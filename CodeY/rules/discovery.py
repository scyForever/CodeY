"""Discover repository-authored instructions without granting them authority.

The scanner treats every file as untrusted evidence.  It records provenance and
scope, while leaving semantic reconciliation and activation to a reviewed rule
patch.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


IGNORED_DIRECTORIES = {
    ".git",
    ".codey",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
SECRET_VALUE_PATTERN = re.compile(
    r"(?i)(-----BEGIN [A-Z ]*PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{8,}|"
    r"\b(?:api[_ -]?key|token|secret|password)\s*[:=]\s*[^\s<]{4,})"
)
MANAGED_BEGIN = "<!-- codey:rules:begin -->"
MANAGED_END = "<!-- codey:rules:end -->"


def _sha256(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strip_managed_block(content: str) -> str:
    """Exclude earlier generated content so repeated plans do not recurse."""
    start = content.find(MANAGED_BEGIN)
    end = content.find(MANAGED_END)
    if start < 0 or end < start:
        return content
    end += len(MANAGED_END)
    return (content[:start] + content[end:]).strip()


def _strip_frontmatter(content: str) -> str:
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return content
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "".join(lines[index + 1 :]).strip()
    return content


@dataclass(frozen=True)
class RuleIssue:
    code: str
    severity: str
    message: str
    paths: tuple[str, ...] = ()

    def to_dict(self):
        payload = asdict(self)
        payload["paths"] = list(self.paths)
        return payload


@dataclass(frozen=True)
class RuleSource:
    source_id: str
    path: str
    ecosystem: str
    kind: str
    scope: str
    precedence: int
    sha256: str
    size_bytes: int
    trust: str
    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def deployable(self):
        return self.kind in {"instructions", "rule", "skill"}

    def to_dict(self, include_content=False):
        payload = asdict(self)
        payload["deployable"] = self.deployable
        if not include_content:
            payload.pop("content", None)
        return payload


@dataclass(frozen=True)
class RuleInventory:
    schema_version: int
    inventory_id: str
    repository_root: str
    revision: str
    git_status: str
    sources: tuple[RuleSource, ...]
    issues: tuple[RuleIssue, ...]

    @property
    def dirty(self):
        return bool(self.git_status.strip())

    def to_dict(self, include_content=False):
        return {
            "schema_version": self.schema_version,
            "inventory_id": self.inventory_id,
            "repository_root": self.repository_root,
            "revision": self.revision,
            "git_status": self.git_status,
            "dirty": self.dirty,
            "sources": [source.to_dict(include_content=include_content) for source in self.sources],
            "issues": [issue.to_dict() for issue in self.issues],
        }


class RuleScanner:
    """Find known coding-agent rule formats inside one repository."""

    def __init__(self, root, max_file_bytes=128 * 1024, max_total_bytes=1024 * 1024):
        self.root = Path(root).resolve()
        self.max_file_bytes = int(max_file_bytes)
        self.max_total_bytes = int(max_total_bytes)
        if not self.root.is_dir():
            raise ValueError(f"repository root is not a directory: {self.root}")

    def scan(self):
        sources = []
        issues = []
        total_bytes = 0
        for path in self._candidate_paths():
            relative = path.relative_to(self.root).as_posix()
            classification = self._classify(relative)
            if classification is None:
                continue
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(self.root)
            except (OSError, ValueError):
                issues.append(
                    RuleIssue(
                        "path_escape",
                        "error",
                        "Rule source resolves outside the repository and was skipped.",
                        (relative,),
                    )
                )
                continue
            try:
                if not resolved.is_file():
                    continue
                size = resolved.stat().st_size
            except OSError:
                issues.append(
                    RuleIssue(
                        "unreadable_source",
                        "warning",
                        "Rule source became unavailable during scanning and was skipped.",
                        (relative,),
                    )
                )
                continue
            if size > self.max_file_bytes:
                issues.append(
                    RuleIssue(
                        "file_too_large",
                        "error",
                        f"Rule source exceeds the {self.max_file_bytes}-byte per-file limit.",
                        (relative,),
                    )
                )
                continue
            if total_bytes + size > self.max_total_bytes:
                issues.append(
                    RuleIssue(
                        "inventory_too_large",
                        "error",
                        f"Rule inventory exceeds the {self.max_total_bytes}-byte total limit.",
                        (relative,),
                    )
                )
                continue
            try:
                raw = resolved.read_bytes()
                content = raw.decode("utf-8")
            except OSError:
                issues.append(
                    RuleIssue(
                        "unreadable_source",
                        "warning",
                        "Rule source could not be read and was skipped.",
                        (relative,),
                    )
                )
                continue
            except UnicodeDecodeError:
                issues.append(
                    RuleIssue(
                        "invalid_encoding",
                        "error",
                        "Rule source is not valid UTF-8 and was skipped.",
                        (relative,),
                    )
                )
                continue
            if len(raw) != size:
                issues.append(
                    RuleIssue(
                        "source_changed_during_scan",
                        "warning",
                        "Rule source changed while it was being scanned and was skipped.",
                        (relative,),
                    )
                )
                continue
            total_bytes += size
            ecosystem, kind, scope, precedence = classification
            cleaned = _strip_managed_block(content) if kind != "config" else content
            if kind == "skill" or relative.casefold().endswith(".mdc"):
                cleaned = _strip_frontmatter(cleaned)
            metadata = self._metadata(relative, ecosystem, kind, content, issues)
            digest = _sha256(content)
            sources.append(
                RuleSource(
                    source_id="src_" + _sha256(f"{relative}\0{digest}")[:16],
                    path=relative,
                    ecosystem=ecosystem,
                    kind=kind,
                    scope=scope,
                    precedence=precedence,
                    sha256=digest,
                    size_bytes=size,
                    trust="repository_untrusted",
                    content=cleaned,
                    metadata=metadata,
                )
            )
            if kind in {"instructions", "rule", "skill"} and SECRET_VALUE_PATTERN.search(cleaned):
                issues.append(
                    RuleIssue(
                        "secret_shaped_content",
                        "error",
                        "Rule source contains secret-shaped content and cannot enter a patch.",
                        (relative,),
                    )
                )

        sources.sort(key=lambda item: (item.precedence, item.ecosystem, item.scope, item.path))
        issues.extend(self._cross_source_issues(sources))
        revision = self._git("rev-parse", "HEAD") or "unversioned"
        status = self._git("status", "--short")
        inventory_payload = [
            (source.path, source.sha256, source.scope, source.ecosystem, source.kind)
            for source in sources
        ]
        inventory_id = "inv_" + _sha256(
            json.dumps(inventory_payload, ensure_ascii=True, separators=(",", ":"))
        )[:16]
        return RuleInventory(
            schema_version=1,
            inventory_id=inventory_id,
            repository_root=str(self.root),
            revision=revision,
            git_status=status,
            sources=tuple(sources),
            issues=tuple(issues),
        )

    def _candidate_paths(self):
        for directory, names, filenames in os.walk(self.root, followlinks=False):
            base = Path(directory)
            names[:] = sorted(
                name
                for name in names
                if name not in IGNORED_DIRECTORIES and not (base / name).is_symlink()
            )
            for filename in sorted(filenames):
                yield base / filename

    def _classify(self, relative):
        parts = tuple(Path(relative).parts)
        lowered = tuple(part.casefold() for part in parts)
        name = lowered[-1]
        parent = Path(relative).parent.as_posix()
        scope = "." if parent == "." else parent
        depth = 0 if scope == "." else len(Path(scope).parts)

        if name == "agents.md":
            return "codex", "instructions", scope, 100 + depth
        if name == "claude.md":
            return "claude", "instructions", scope, 100 + depth
        if name == "gemini.md":
            return "gemini", "instructions", scope, 100 + depth
        if lowered == (".github", "copilot-instructions.md"):
            return "copilot", "instructions", ".", 90
        if len(lowered) >= 3 and lowered[:2] == (".github", "instructions") and name.endswith(".instructions.md"):
            return "copilot", "rule", ".", 110
        if len(lowered) >= 3 and lowered[:2] == (".cursor", "rules") and name.endswith(".mdc"):
            return "cursor", "rule", ".", 110
        if len(lowered) >= 4 and lowered[:2] == (".cursor", "skills") and name == "skill.md":
            return "cursor", "skill", ".", 110
        if len(lowered) >= 3 and lowered[:2] == (".claude", "rules") and name.endswith(".md"):
            return "claude", "rule", ".", 110
        if len(lowered) >= 4 and lowered[:2] == (".claude", "skills") and name == "skill.md":
            return "claude", "skill", ".", 110
        if len(lowered) >= 3 and lowered[:2] in {
            (".claude", "agents"),
            (".claude", "commands"),
        } and name.endswith(".md"):
            return "claude", "rule", ".", 110
        if len(lowered) >= 3 and lowered[:2] == (".codex", "rules") and name.endswith((".md", ".rules")):
            return "codex", "rule", ".", 110
        if lowered == (".codex", "config.toml"):
            return "codex", "config", ".", 10
        if lowered == (".claude", "settings.json"):
            return "claude", "config", ".", 10
        return None

    def _metadata(self, relative, ecosystem, kind, content, issues):
        metadata = {}
        if relative.casefold().endswith(".mdc"):
            metadata.update(self._frontmatter(content))
        if ecosystem == "codex" and kind == "config":
            try:
                config = tomllib.loads(content)
            except (tomllib.TOMLDecodeError, ValueError):
                issues.append(RuleIssue("invalid_config", "warning", "Codex config is invalid TOML.", (relative,)))
                return metadata
            sandbox = str(config.get("sandbox_mode", ""))
            approval = str(config.get("approval_policy", ""))
            metadata.update({"sandbox_mode": sandbox, "approval_policy": approval})
            if sandbox == "danger-full-access" or approval == "never":
                issues.append(
                    RuleIssue(
                        "permissive_agent_config",
                        "warning",
                        "Repository Codex config requests a permissive execution posture; trials override it.",
                        (relative,),
                    )
                )
        if ecosystem == "claude" and kind == "config":
            try:
                config = json.loads(content)
            except json.JSONDecodeError:
                issues.append(RuleIssue("invalid_config", "warning", "Claude settings are invalid JSON.", (relative,)))
                return metadata
            hooks = config.get("hooks", {}) if isinstance(config, dict) else {}
            metadata["has_hooks"] = bool(hooks)
            if hooks:
                issues.append(
                    RuleIssue(
                        "repository_hooks",
                        "warning",
                        "Repository Claude hooks are executable configuration; isolated trials disable them.",
                        (relative,),
                    )
                )
        return metadata

    @staticmethod
    def _frontmatter(content):
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}
        metadata = {}
        for line in lines[1:]:
            if line.strip() == "---":
                break
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"alwaysApply", "description", "globs"}:
                metadata[key.strip()] = value.strip().strip("\"'")
        return metadata

    @staticmethod
    def _cross_source_issues(sources):
        issues = []
        by_hash = {}
        by_scope = {}
        for source in sources:
            if not source.deployable or not source.content.strip():
                continue
            by_hash.setdefault(_sha256(source.content.strip()), []).append(source.path)
            by_scope.setdefault((source.ecosystem, source.scope), []).append(source.path)
        for paths in by_hash.values():
            if len(paths) > 1:
                issues.append(
                    RuleIssue(
                        "duplicate_rule_content",
                        "info",
                        "Multiple rule sources contain identical managed content; planning deduplicates them.",
                        tuple(sorted(paths)),
                    )
                )
        for paths in by_scope.values():
            if len(paths) > 1:
                issues.append(
                    RuleIssue(
                        "overlapping_scope",
                        "info",
                        "Multiple rule sources share an ecosystem and scope; human review must resolve semantics.",
                        tuple(sorted(paths)),
                    )
                )
        return issues

    def _git(self, *args):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return result.stdout.rstrip() if result.returncode == 0 else ""


__all__ = [
    "MANAGED_BEGIN",
    "MANAGED_END",
    "RuleInventory",
    "RuleIssue",
    "RuleScanner",
    "RuleSource",
]
