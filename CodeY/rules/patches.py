"""Reviewable rule patches for CodeY and external coding-agent adapters."""

from __future__ import annotations

import base64
import difflib
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .discovery import MANAGED_BEGIN, MANAGED_END, RuleInventory


PATCH_ID_PATTERN = re.compile(r"^rulepatch_[0-9a-f]{32}$")
TARGET_PATHS = {
    "codey": ".codey/rules/active.md",
    "codex": "AGENTS.md",
    "claude": "CLAUDE.md",
    "cursor": ".cursor/rules/codey-managed.mdc",
}
PATCH_STATUSES = {"review_required", "active", "rolled_back", "rejected"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _sync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _sync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload):
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _sync_parent(path: Path):
    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _patch_identity(patch):
    repository = patch["repository"]
    return {
        "schema_version": patch["schema_version"],
        "objective": patch["objective"],
        "repository": {
            "root": repository["root"],
            "base_revision": repository["base_revision"],
            "dirty_at_plan": repository["dirty_at_plan"],
            "git_status_at_plan": repository["git_status_at_plan"],
            "dirty_paths_at_plan": repository["dirty_paths_at_plan"],
        },
        "inventory_id": patch["inventory_id"],
        "source_refs": patch["source_refs"],
        "inventory_issues": patch["inventory_issues"],
        "artifacts": [
            {
                "target": artifact["target"],
                "path": artifact["path"],
                "operation": artifact["operation"],
                "before_sha256": artifact["before_sha256"],
                "candidate_sha256": artifact["candidate_sha256"],
            }
            for artifact in patch["artifacts"]
        ],
    }


def _patch_id_for(patch):
    encoded = json.dumps(
        _patch_identity(patch),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "rulepatch_" + _sha256(encoded)[:32]


class _RepositoryLock:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is None:
            return False
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        return False


def _replace_managed_block(existing: str, managed: str) -> str:
    start = existing.find(MANAGED_BEGIN)
    end = existing.find(MANAGED_END)
    if start >= 0 and end >= start:
        end += len(MANAGED_END)
        prefix = existing[:start].rstrip()
        suffix = existing[end:].lstrip()
        pieces = [piece for piece in (prefix, managed, suffix) if piece]
        return "\n\n".join(pieces).rstrip() + "\n"
    if existing.strip():
        return existing.rstrip() + "\n\n" + managed.rstrip() + "\n"
    return managed.rstrip() + "\n"


class RulePatchStore:
    """Create, inspect, apply, and roll back repository rule patches."""

    def __init__(self, repository_root):
        self.root = Path(repository_root).resolve()
        self.store_root = self.root / ".codey" / "rules"
        self.patch_root = self.store_root / "patches"
        self.trial_root = self.store_root / "trials"
        self.lock_path = self.store_root / ".lock"
        self.transaction_path = self.store_root / "transaction.json"

    def create_plan(
        self,
        inventory: RuleInventory,
        targets=("codey", "codex", "claude", "cursor"),
        objective="Unify repository coding-agent rules without changing their declared scope.",
        source_paths=None,
        max_bundle_chars=48_000,
    ):
        if Path(inventory.repository_root).resolve() != self.root:
            raise ValueError("inventory belongs to a different repository")
        normalized_targets = tuple(dict.fromkeys(str(target).strip().lower() for target in targets))
        if not normalized_targets:
            raise ValueError("at least one target agent is required")
        unknown = sorted(set(normalized_targets) - set(TARGET_PATHS))
        if unknown:
            raise ValueError(f"unknown rule target(s): {', '.join(unknown)}")

        requested_paths = None if source_paths is None else {str(path).replace("\\", "/") for path in source_paths}
        sources = [
            source
            for source in inventory.sources
            if source.deployable and source.content.strip() and (requested_paths is None or source.path in requested_paths)
        ]
        if requested_paths is not None:
            missing = sorted(requested_paths - {source.path for source in sources})
            if missing:
                raise ValueError(f"unknown or non-deployable rule source(s): {', '.join(missing)}")
        if not sources:
            raise ValueError("no deployable repository rules were discovered")

        selected_paths = {source.path for source in sources}
        blocking = [
            issue
            for issue in inventory.issues
            if issue.severity == "error" and (not issue.paths or selected_paths.intersection(issue.paths))
        ]
        if blocking:
            codes = ", ".join(sorted({issue.code for issue in blocking}))
            raise ValueError(f"rule plan blocked by inventory errors: {codes}")

        source_refs = [
            {
                "source_id": source.source_id,
                "path": source.path,
                "ecosystem": source.ecosystem,
                "kind": source.kind,
                "scope": source.scope,
                "precedence": source.precedence,
                "sha256": source.sha256,
                "trust": source.trust,
            }
            for source in sources
        ]
        artifacts = []
        for target in normalized_targets:
            relative = TARGET_PATHS[target]
            target_path = self._target_path(relative)
            before_bytes = target_path.read_bytes() if target_path.exists() else b""
            try:
                before = before_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"rule target is not valid UTF-8: {relative}") from exc
            managed = self._render_managed_bundle(
                sources,
                objective=objective,
                target=target,
                excluded_path=relative,
            )
            if len(managed) > int(max_bundle_chars):
                raise ValueError(
                    f"managed bundle for {target} exceeds {int(max_bundle_chars)} characters; select fewer sources"
                )
            candidate = _replace_managed_block(before, managed)
            if target == "cursor" and not before.strip():
                candidate = (
                    "---\n"
                    "description: CodeY governed repository instructions\n"
                    "alwaysApply: true\n"
                    "---\n\n"
                    + candidate
                )
            artifacts.append(
                {
                    "target": target,
                    "path": relative,
                    "operation": "upsert_managed_block",
                    "before_sha256": _sha256(before_bytes) if target_path.exists() else None,
                    "before_content": before,
                    "before_bytes_b64": base64.b64encode(before_bytes).decode("ascii")
                    if target_path.exists()
                    else None,
                    "candidate_sha256": _sha256(candidate),
                    "candidate_content": candidate,
                    "diff": "".join(
                        difflib.unified_diff(
                            before.splitlines(keepends=True),
                            candidate.splitlines(keepends=True),
                            fromfile=f"a/{relative}",
                            tofile=f"b/{relative}",
                        )
                    ),
                }
            )

        repository = {
            "root": ".",
            "base_revision": inventory.revision,
            "dirty_at_plan": inventory.dirty,
            "git_status_at_plan": inventory.git_status,
            "dirty_paths_at_plan": self._dirty_paths(inventory.git_status),
        }
        inventory_issues = [issue.to_dict() for issue in inventory.issues]
        identity_record = {
            "schema_version": 2,
            "objective": str(objective).strip(),
            "repository": repository,
            "inventory_id": inventory.inventory_id,
            "source_refs": source_refs,
            "inventory_issues": inventory_issues,
            "artifacts": artifacts,
        }
        patch_id = _patch_id_for(identity_record)
        existing = self.patch_root / f"{patch_id}.json"
        if existing.exists():
            return self.load(patch_id)
        created_at = _now()
        payload = {
            "schema_version": 2,
            "id": patch_id,
            "created_at": created_at,
            "updated_at": created_at,
            "status": "review_required",
            "objective": str(objective).strip(),
            "repository": repository,
            "inventory_id": inventory.inventory_id,
            "source_refs": source_refs,
            "inventory_issues": inventory_issues,
            "artifacts": artifacts,
            "history": [
                {
                    "at": created_at,
                    "event": "planned",
                    "status": "review_required",
                    "reason": "Repository-authored rules require explicit human review.",
                }
            ],
        }
        self.patch_root.mkdir(parents=True, exist_ok=True)
        self._validate_patch(payload, expected_id=patch_id)
        _atomic_write_json(existing, payload)
        return payload

    def list_patches(self):
        if not self.patch_root.exists():
            return []
        patches = []
        for path in sorted(self.patch_root.glob("rulepatch_*.json")):
            try:
                patches.append(self.load(path.stem))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return patches

    def load(self, patch_id):
        patch_id = self._validate_patch_id(patch_id)
        path = self.patch_root / f"{patch_id}.json"
        if not path.exists():
            raise ValueError(f"unknown rule patch: {patch_id}")
        patch = self._read_json(path)
        self._validate_patch(patch, expected_id=patch_id)
        return patch

    def apply(self, patch_id, approved=False):
        if not approved:
            raise ValueError("applying a rule patch requires explicit approval")
        with self._transaction_lock():
            patch = self.load(patch_id)
            if patch["status"] != "review_required":
                raise ValueError(f"rule patch is not reviewable: {patch['status']}")
            self._verify_sources(patch)
            self._verify_artifacts(patch, expected="before")
            self._begin_transaction("apply", patch)
            try:
                for artifact in patch["artifacts"]:
                    target = self._target_path(artifact["path"])
                    self._verify_one_artifact(artifact, expected="before")
                    _atomic_write_text(target, artifact["candidate_content"])
                patch["status"] = "active"
                patch["updated_at"] = _now()
                patch["history"].append(
                    {
                        "at": patch["updated_at"],
                        "event": "applied",
                        "status": "active",
                        "reason": "Explicit CLI approval after exact source and target hash verification.",
                    }
                )
                self._save(patch)
                self.transaction_path.unlink(missing_ok=True)
            except Exception:
                self._recover_transaction()
                raise
            return patch

    def rollback(self, patch_id, approved=False):
        if not approved:
            raise ValueError("rolling back a rule patch requires explicit approval")
        with self._transaction_lock():
            patch = self.load(patch_id)
            if patch["status"] != "active":
                raise ValueError(f"only active rule patches can be rolled back: {patch['status']}")
            self._verify_artifacts(patch, expected="candidate")
            self._begin_transaction("rollback", patch)
            try:
                for artifact in patch["artifacts"]:
                    self._verify_one_artifact(artifact, expected="candidate")
                    self._restore_before(self._target_path(artifact["path"]), artifact)
                patch["status"] = "rolled_back"
                patch["updated_at"] = _now()
                patch["history"].append(
                    {
                        "at": patch["updated_at"],
                        "event": "rolled_back",
                        "status": "rolled_back",
                        "reason": "Explicit CLI rollback after exact candidate hash verification.",
                    }
                )
                self._save(patch)
                self.transaction_path.unlink(missing_ok=True)
            except Exception:
                self._recover_transaction()
                raise
            return patch

    def list_trials(self, patch_id=None):
        if not self.trial_root.exists():
            return []
        trials = []
        for path in sorted(self.trial_root.glob("trial_*/result.json")):
            try:
                result = self._read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if patch_id is None or result.get("patch_id") == patch_id:
                trials.append(result)
        return trials

    def active_artifact(self, target):
        target = str(target).strip().lower()
        if target not in TARGET_PATHS:
            raise ValueError(f"unknown rule target: {target}")
        path = self._target_path(TARGET_PATHS[target])
        if not path.is_file():
            return None
        if path.stat().st_size > 128 * 1024:
            return None
        current_sha256 = _sha256(path.read_bytes())
        matches = []
        for patch in self.list_patches():
            if patch["status"] != "active":
                continue
            artifact = next(
                (item for item in patch["artifacts"] if item["target"] == target),
                None,
            )
            if artifact is not None and artifact["candidate_sha256"] == current_sha256:
                matches.append((patch["updated_at"], patch["id"], artifact))
        if not matches:
            return None
        _, patch_id, artifact = max(matches, key=lambda item: (item[0], item[1]))
        return {
            "patch_id": patch_id,
            "path": artifact["path"],
            "sha256": artifact["candidate_sha256"],
            "content": artifact["candidate_content"],
        }

    @contextmanager
    def _transaction_lock(self):
        with _RepositoryLock(self.lock_path):
            self._recover_transaction()
            yield

    def _begin_transaction(self, operation, patch):
        _atomic_write_json(
            self.transaction_path,
            {
                "schema_version": 2,
                "operation": operation,
                "patch_id": patch["id"],
                "artifacts": self._transaction_artifacts(patch),
                "started_at": _now(),
            },
        )

    def _recover_transaction(self):
        if not self.transaction_path.exists():
            return
        journal = self._read_json(self.transaction_path)
        operation = journal.get("operation")
        patch_id = journal.get("patch_id")
        if journal.get("schema_version") != 2 or operation not in {"apply", "rollback"}:
            raise ValueError("invalid rule patch transaction journal")
        patch = self.load(patch_id)
        if journal.get("artifacts") != self._transaction_artifacts(patch):
            raise ValueError("rule patch transaction journal does not match the patch identity")
        if operation == "apply":
            if patch["status"] == "active":
                self._verify_artifacts(patch, expected="candidate")
                self.transaction_path.unlink(missing_ok=True)
                return
            if patch["status"] != "review_required":
                raise ValueError(f"invalid apply recovery state: {patch['status']}")
            for artifact in patch["artifacts"]:
                target = self._target_path(artifact["path"])
                current = _sha256(target.read_bytes()) if target.exists() else None
                if current == artifact["before_sha256"]:
                    continue
                if current != artifact["candidate_sha256"]:
                    raise ValueError(
                        f"rule patch transaction recovery conflict: {artifact['path']}"
                    )
                self._restore_before(target, artifact)
        else:
            if patch["status"] == "rolled_back":
                self._verify_artifacts(patch, expected="before")
                self.transaction_path.unlink(missing_ok=True)
                return
            if patch["status"] != "active":
                raise ValueError(f"invalid rollback recovery state: {patch['status']}")
            for artifact in patch["artifacts"]:
                target = self._target_path(artifact["path"])
                current = _sha256(target.read_bytes()) if target.exists() else None
                if current == artifact["candidate_sha256"]:
                    continue
                if current != artifact["before_sha256"]:
                    raise ValueError(
                        f"rule patch transaction recovery conflict: {artifact['path']}"
                    )
                _atomic_write_text(
                    target,
                    artifact["candidate_content"],
                )
        self.transaction_path.unlink(missing_ok=True)

    @staticmethod
    def _transaction_artifacts(patch):
        return [
            {
                "path": artifact["path"],
                "before_sha256": artifact["before_sha256"],
                "candidate_sha256": artifact["candidate_sha256"],
            }
            for artifact in patch["artifacts"]
        ]

    def _render_managed_bundle(self, sources, objective, target, excluded_path):
        lines = [
            MANAGED_BEGIN,
            "# CodeY governed repository rules",
            "",
            f"Objective: {str(objective).strip()}",
            f"Adapter target: {target}",
            "",
            "This block is generated from repository-authored, human-review-required sources.",
            "Apply each source only inside its declared scope. More specific directory scope wins",
            "within the same ecosystem; cross-ecosystem conflicts require human judgment.",
        ]
        seen = set()
        selected = 0
        for source in sources:
            if source.path == excluded_path:
                continue
            normalized = source.content.strip()
            digest = _sha256(normalized)
            if not normalized or digest in seen:
                continue
            seen.add(digest)
            selected += 1
            source_directory = Path(source.path).parent.as_posix()
            lines.extend(
                [
                    "",
                    f"## Source `{source.path}`",
                    f"Ecosystem: {source.ecosystem}; scope: `{source.scope}`; source hash: `{source.sha256}`.",
                    f"Resolve relative path references in this section against `{source_directory}`.",
                    "",
                    normalized,
                ]
            )
        if not selected:
            lines.extend(["", "No cross-adapter source content was selected for this target."])
        lines.extend(["", MANAGED_END])
        return "\n".join(lines)

    def _verify_sources(self, patch):
        for source in patch["source_refs"]:
            path = self._target_path(source["path"])
            if not path.is_file():
                raise ValueError(f"rule patch is stale; source is missing: {source['path']}")
            current = _sha256(path.read_bytes())
            if current != source["sha256"]:
                raise ValueError(f"rule patch is stale; source changed: {source['path']}")

    def _verify_artifacts(self, patch, expected):
        for artifact in patch["artifacts"]:
            self._verify_one_artifact(artifact, expected=expected)

    def _verify_one_artifact(self, artifact, expected):
        path = self._target_path(artifact["path"])
        current = _sha256(path.read_bytes()) if path.exists() else None
        expected_hash = artifact["before_sha256"] if expected == "before" else artifact["candidate_sha256"]
        if current != expected_hash:
            raise ValueError(f"rule patch target drifted: {artifact['path']}")

    @staticmethod
    def _restore_before(path, artifact):
        if artifact["before_sha256"] is None:
            if path.exists():
                path.unlink()
            return
        encoded = artifact.get("before_bytes_b64")
        if encoded is not None:
            _atomic_write_bytes(path, base64.b64decode(encoded.encode("ascii")))
        else:
            _atomic_write_text(path, artifact["before_content"])

    def _target_path(self, relative):
        normalized = str(relative).replace("\\", "/")
        if normalized.startswith("/") or ":" in normalized.split("/")[0]:
            raise ValueError(f"absolute rule path is not allowed: {relative}")
        candidate = self.root.joinpath(*Path(normalized).parts)
        parent = candidate.parent.resolve()
        try:
            parent.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"rule path escapes repository: {relative}") from exc
        if candidate.exists():
            try:
                candidate.resolve(strict=True).relative_to(self.root)
            except (OSError, ValueError) as exc:
                raise ValueError(f"rule path resolves outside repository: {relative}") from exc
        return candidate

    def _save(self, patch):
        patch_id = self._validate_patch_id(patch["id"])
        self._validate_patch(patch, expected_id=patch_id)
        _atomic_write_json(self.patch_root / f"{patch_id}.json", patch)

    def _validate_patch(self, patch, expected_id):
        if not isinstance(patch, dict) or patch.get("schema_version") != 2:
            raise ValueError(f"unsupported rule patch schema: {expected_id}")
        required = {
            "id",
            "created_at",
            "updated_at",
            "status",
            "objective",
            "repository",
            "inventory_id",
            "source_refs",
            "inventory_issues",
            "artifacts",
            "history",
        }
        if required - set(patch):
            raise ValueError(f"invalid rule patch record: {expected_id}")
        if patch.get("id") != expected_id or patch.get("status") not in PATCH_STATUSES:
            raise ValueError(f"invalid rule patch record: {expected_id}")
        if not isinstance(patch["repository"], dict) or not isinstance(patch["history"], list):
            raise ValueError(f"invalid rule patch record: {expected_id}")
        repository = patch["repository"]
        repository_keys = {
            "root",
            "base_revision",
            "dirty_at_plan",
            "git_status_at_plan",
            "dirty_paths_at_plan",
        }
        if repository_keys - set(repository) or repository.get("root") != ".":
            raise ValueError(f"invalid rule patch repository: {expected_id}")
        if not isinstance(repository.get("base_revision"), str) or not repository["base_revision"]:
            raise ValueError(f"invalid rule patch repository: {expected_id}")
        if not isinstance(repository.get("dirty_at_plan"), bool) or not isinstance(
            repository.get("git_status_at_plan"), str
        ):
            raise ValueError(f"invalid rule patch repository: {expected_id}")
        dirty_paths = repository.get("dirty_paths_at_plan")
        if (
            not isinstance(dirty_paths, list)
            or not all(isinstance(path, str) and path for path in dirty_paths)
            or dirty_paths != sorted(set(dirty_paths))
            or dirty_paths != self._dirty_paths(repository["git_status_at_plan"])
            or repository["dirty_at_plan"] != bool(repository["git_status_at_plan"].strip())
        ):
            raise ValueError(f"invalid rule patch dirty baseline: {expected_id}")
        if not isinstance(patch["source_refs"], list) or not patch["source_refs"]:
            raise ValueError(f"invalid rule patch sources: {expected_id}")
        if not isinstance(patch["inventory_issues"], list):
            raise ValueError(f"invalid rule patch inventory issues: {expected_id}")
        if not isinstance(patch["artifacts"], list) or not patch["artifacts"]:
            raise ValueError(f"invalid rule patch artifacts: {expected_id}")

        for issue in patch["inventory_issues"]:
            if not isinstance(issue, dict) or not all(
                isinstance(issue.get(key), str) for key in ("code", "severity", "message")
            ):
                raise ValueError(f"invalid rule patch inventory issue: {expected_id}")
            if not isinstance(issue.get("paths"), list) or not all(
                isinstance(path, str) for path in issue["paths"]
            ):
                raise ValueError(f"invalid rule patch inventory issue: {expected_id}")

        source_paths = set()
        for source in patch["source_refs"]:
            if not isinstance(source, dict):
                raise ValueError(f"invalid rule patch source: {expected_id}")
            string_keys = ("source_id", "path", "ecosystem", "kind", "scope", "sha256", "trust")
            if not all(isinstance(source.get(key), str) for key in string_keys):
                raise ValueError(f"invalid rule patch source: {expected_id}")
            if (
                source["path"] in source_paths
                or not re.fullmatch(r"src_[0-9a-f]{16}", source["source_id"])
                or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
                or source["kind"] not in {"instructions", "rule", "skill"}
                or source["trust"] != "repository_untrusted"
                or not isinstance(source.get("precedence"), int)
                or isinstance(source.get("precedence"), bool)
            ):
                raise ValueError(f"invalid rule patch source: {expected_id}")
            source_paths.add(source["path"])
            self._target_path(source["path"])

        targets = set()
        for artifact in patch["artifacts"]:
            if not isinstance(artifact, dict):
                raise ValueError(f"invalid rule patch artifact: {expected_id}")
            target = artifact.get("target")
            path = artifact.get("path")
            if target not in TARGET_PATHS or path != TARGET_PATHS[target] or target in targets:
                raise ValueError(f"invalid rule patch artifact target: {expected_id}")
            targets.add(target)
            if artifact.get("operation") != "upsert_managed_block":
                raise ValueError(f"invalid rule patch operation: {expected_id}")
            candidate = artifact.get("candidate_content")
            before = artifact.get("before_content")
            if not isinstance(candidate, str) or not isinstance(before, str):
                raise ValueError(f"invalid rule patch content: {expected_id}")
            if _sha256(candidate) != artifact.get("candidate_sha256"):
                raise ValueError(f"rule patch candidate hash mismatch: {path}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("candidate_sha256", ""))):
                raise ValueError(f"invalid rule patch candidate hash: {path}")
            before_hash = artifact.get("before_sha256")
            before_b64 = artifact.get("before_bytes_b64")
            if before_hash is None:
                if before_b64 is not None or before != "":
                    raise ValueError(f"invalid missing-target baseline: {path}")
            else:
                if not re.fullmatch(r"[0-9a-f]{64}", str(before_hash)):
                    raise ValueError(f"invalid target baseline hash: {path}")
                try:
                    before_bytes = base64.b64decode(str(before_b64).encode("ascii"), validate=True)
                    decoded = before_bytes.decode("utf-8")
                except (UnicodeDecodeError, ValueError) as exc:
                    raise ValueError(f"invalid target baseline: {path}") from exc
                if _sha256(before_bytes) != before_hash or decoded != before:
                    raise ValueError(f"rule patch baseline hash mismatch: {path}")
            expected_diff = "".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    candidate.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )
            if artifact.get("diff") != expected_diff:
                raise ValueError(f"rule patch review diff mismatch: {path}")

        if not re.fullmatch(r"inv_[0-9a-f]{16}", str(patch["inventory_id"])):
            raise ValueError(f"invalid rule inventory id: {expected_id}")
        calculated_id = _patch_id_for(patch)
        if calculated_id != expected_id:
            raise ValueError(f"rule patch identity mismatch: {expected_id}")

    @staticmethod
    def _dirty_paths(status):
        paths = []
        for line in str(status or "").splitlines():
            if len(line) >= 3 and line[2] == " ":
                path = line[3:].strip()
            elif len(line) >= 2 and line[1] == " ":
                path = line[2:].strip()
            else:
                continue
            if " -> " in path:
                old, new = path.split(" -> ", 1)
                paths.extend([old.strip(), new.strip()])
            elif path:
                paths.append(path)
        return sorted(set(paths))

    @staticmethod
    def _read_json(path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object: {path}")
        return payload

    @staticmethod
    def _validate_patch_id(patch_id):
        patch_id = str(patch_id)
        if not PATCH_ID_PATTERN.fullmatch(patch_id):
            raise ValueError(f"invalid rule patch id: {patch_id}")
        return patch_id


__all__ = ["RulePatchStore", "TARGET_PATHS"]
