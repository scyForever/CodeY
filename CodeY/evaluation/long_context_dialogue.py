"""Privacy-safe replay evaluation for long Codex conversations."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from ..providers.clients import ModelCompletion


SCHEMA_VERSION = 1
RESULTS_SCHEMA_VERSION = 2
PROTOCOL_VERSION = 3
SCORER_VERSION = 4
VARIANTS = (
    "full_dialogue",
    "codex_summary",
    "tail_window",
    "structured_ledger",
    "oracle",
)

WINDOWS_PATH_RE = re.compile(r"(?i)(?:[A-Z]:[\\/])[^\s<>\"|]+")
POSIX_PATH_RE = re.compile(r"(?<!\w)/(?:Users|home|mnt|tmp|var|etc|opt)/[^\s<>\"|]+")
UNC_PATH_RE = re.compile(r"\\\\[^\s\\/]+[\\/][^\s<>\"|]+")
URL_RE = re.compile(r"(?i)\b(?:https?|file|ssh|scp|sftp)://[^\s<>\"]+")
SCP_LOCATION_RE = re.compile(
    r"(?i)\b[A-Z0-9_.-]+@(?:\[[^\]]+\]|[A-Z0-9_.-]+):[^\s<>\"]+"
)
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
IPV4_RE = re.compile(
    r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"
)
IPV6_CANDIDATE_RE = re.compile(
    r"(?i)(?<![0-9a-f:])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![0-9a-f:])"
)
PHONE_RE = re.compile(
    r"(?<!\d)(?:(?:\+?86[- ]?)?1[3-9]\d{9}|"
    r"(?:\+?\d{1,3}[- .]?)?\(?\d{3}\)?[- .]\d{3}[- .]\d{4})(?!\d)"
)
LONG_ID_RE = re.compile(r"(?i)(?<!\d)(?:\d{17}[\dX]|(?:\d[ -]?){15,19}\d)(?!\d)")
RELATIVE_PATH_RE = re.compile(
    r"(?<![\w<])(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,10}\b"
)
FILE_NAME_RE = re.compile(
    r"(?i)\b[A-Za-z0-9_.-]+\.(?:py|md|jsonl?|toml|ya?ml|docx?|pdf|png|jpe?g|drawio|html?|cpp|hpp|c|h)\b"
)
UUID_RE = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
)
HASH_RE = re.compile(r"(?i)\b[0-9a-f]{32,64}\b")
INTERNAL_AGENT_PATH_RE = re.compile(r"(?<![\w<])/(?:root|agent)(?:/[A-Za-z0-9_.-]+)+\b")
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
COMMON_TOKEN_RE = re.compile(
    r"(?i)\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{16,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|"
    r"AIza[0-9A-Za-z_-]{30,})\b"
)
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret|authorization)"
    r"\b\s*[:=]\s*(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\r\n]*)"
)
PEM_RE = re.compile(r"-----BEGIN [^-]+-----.*?-----END [^-]+-----", flags=re.DOTALL)
CODE_BLOCK_RE = re.compile(r"```.*?```", flags=re.DOTALL)
SPACE_RE = re.compile(r"[ \t]+")
SAFE_ID_RE = re.compile(r"[a-z][a-z0-9_-]{0,95}\Z")
PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z0-9_]{0,63}>\Z")
SAFE_PROVIDER_METADATA_FIELDS = {
    "cache_hit",
    "cached_tokens",
    "input_tokens",
    "output_tokens",
    "prompt_cache_supported",
    "request_attempts",
    "total_tokens",
}
PUBLIC_HASH_FIELDS = {"dataset_sha256", "prompt_sha256"}


class LongContextEvaluationError(ValueError):
    """Raised when source data, annotations, or model output is invalid."""


@dataclass(frozen=True)
class DialogueEvent:
    source_line: int
    role: str
    text: str


def _canonical_json(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _load_json(path):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LongContextEvaluationError(f"cannot read JSON {path}: {exc}") from exc


def _require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise LongContextEvaluationError(f"{field} must be a non-empty string")
    return value.strip()


def _require_identifier(value, field):
    value = _require_text(value, field)
    if not SAFE_ID_RE.fullmatch(value):
        raise LongContextEvaluationError(
            f"{field} must use lowercase letters, digits, underscores, or hyphens"
        )
    return value


def _safe_replacements(value, field):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise LongContextEvaluationError(f"{field} must be an object")
    replacements = {}
    for source, target in value.items():
        source = _require_text(source, f"{field} key")
        target = _require_text(target, f"{field}[{source!r}]")
        if not PLACEHOLDER_RE.fullmatch(target):
            raise LongContextEvaluationError(
                f"{field} replacement values must be typed placeholders"
            )
        replacements[source] = target
    return replacements


def _safe_source_path(codex_root, relative_path):
    root = Path(codex_root).resolve()
    path = (root / _require_text(relative_path, "source")).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise LongContextEvaluationError(
            "source must stay inside the Codex root"
        ) from exc
    if not path.is_file():
        raise LongContextEvaluationError(f"source rollout does not exist: {path}")
    return path


def _read_rollout(path):
    rows = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LongContextEvaluationError(
                        f"invalid rollout JSON at line {line_number}"
                    ) from exc
                if not isinstance(item, dict):
                    raise LongContextEvaluationError(
                        f"rollout line {line_number} must contain an object"
                    )
                rows.append((line_number, item))
    except (OSError, UnicodeError) as exc:
        raise LongContextEvaluationError(f"cannot read rollout {path}: {exc}") from exc
    return rows


def _content_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"input_text", "output_text", "text"}:
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _strip_attachment_preamble(text):
    marker = "## My request for Codex:"
    if marker in text:
        return "[attachments omitted]\n" + text.split(marker, 1)[1].strip()
    return text


def _is_ipv6(value):
    try:
        return ipaddress.ip_address(value).version == 6
    except ValueError:
        return False


def _redact_ipv6(text):
    return IPV6_CANDIDATE_RE.sub(
        lambda match: "<IP>" if _is_ipv6(match.group(0)) else match.group(0), text
    )


def sanitize_text(text, replacements=None, max_chars=12000):
    """Remove high-risk identifiers while keeping task semantics readable."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = _strip_attachment_preamble(text)
    text = PEM_RE.sub("<SECRET>", text)
    text = CODE_BLOCK_RE.sub("<CODE_BLOCK>", text)
    text = JWT_RE.sub("<SECRET>", text)
    text = BEARER_RE.sub("<SECRET>", text)
    text = COMMON_TOKEN_RE.sub("<SECRET>", text)
    text = SECRET_ASSIGNMENT_RE.sub("<SECRET>", text)
    text = SCP_LOCATION_RE.sub("<REMOTE_LOCATION>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = URL_RE.sub("<URL>", text)
    text = IPV4_RE.sub("<IP>", text)
    text = _redact_ipv6(text)
    text = PHONE_RE.sub("<PHONE>", text)
    text = LONG_ID_RE.sub("<SENSITIVE_NUMBER>", text)
    text = WINDOWS_PATH_RE.sub("<PATH>", text)
    text = UNC_PATH_RE.sub("<PATH>", text)
    text = POSIX_PATH_RE.sub("<PATH>", text)
    text = INTERNAL_AGENT_PATH_RE.sub("<INTERNAL_AGENT>", text)
    text = RELATIVE_PATH_RE.sub("<REL_PATH>", text)
    text = FILE_NAME_RE.sub("<FILE>", text)
    text = UUID_RE.sub("<UUID>", text)
    text = HASH_RE.sub("<HASH>", text)
    for source, target in sorted(
        (replacements or {}).items(), key=lambda item: len(str(item[0])), reverse=True
    ):
        if str(source):
            text = re.sub(
                re.escape(str(source)), str(target), text, flags=re.IGNORECASE
            )
    lines = [SPACE_RE.sub(" ", line).rstrip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line.strip()).strip()
    if len(text) > int(max_chars):
        text = text[: max(0, int(max_chars) - 15)].rstrip() + "\n<TRUNCATED>"
    assert_no_direct_leak(text)
    return text


def assert_no_direct_leak(text):
    checks = (
        (WINDOWS_PATH_RE, "Windows path"),
        (POSIX_PATH_RE, "POSIX path"),
        (UNC_PATH_RE, "UNC path"),
        (INTERNAL_AGENT_PATH_RE, "internal agent path"),
        (RELATIVE_PATH_RE, "relative path"),
        (FILE_NAME_RE, "file name"),
        (URL_RE, "URL"),
        (SCP_LOCATION_RE, "remote location"),
        (EMAIL_RE, "email"),
        (IPV4_RE, "IPv4 address"),
        (PHONE_RE, "phone number"),
        (LONG_ID_RE, "long sensitive number"),
        (UUID_RE, "UUID"),
        (HASH_RE, "hash"),
        (JWT_RE, "JWT"),
        (BEARER_RE, "bearer token"),
        (COMMON_TOKEN_RE, "provider token"),
        (SECRET_ASSIGNMENT_RE, "secret assignment"),
        (PEM_RE, "private key"),
    )
    for pattern, label in checks:
        if pattern.search(str(text)):
            raise LongContextEvaluationError(f"sanitized text still contains a {label}")
    if any(_is_ipv6(match.group(0)) for match in IPV6_CANDIDATE_RE.finditer(str(text))):
        raise LongContextEvaluationError(
            "sanitized text still contains an IPv6 address"
        )


def assert_public_payload_safe(value, field="payload"):
    if isinstance(value, str):
        assert_no_direct_leak(value)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_public_payload_safe(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            key = str(key)
            assert_no_direct_leak(key)
            if key in PUBLIC_HASH_FIELDS:
                if not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item):
                    raise LongContextEvaluationError(
                        f"{field}.{key} must be a lowercase SHA-256 digest"
                    )
                continue
            assert_public_payload_safe(item, f"{field}.{key}")
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    raise LongContextEvaluationError(f"{field} has unsupported public value type")


def _visible_events(rows, cutoff_line, replacements):
    events = []
    for line_number, item in rows:
        if line_number > cutoff_line:
            break
        payload = item.get("payload") or {}
        if item.get("type") != "event_msg":
            continue
        payload_type = payload.get("type")
        if payload_type == "user_message":
            role = "user"
        elif payload_type == "agent_message":
            role = "assistant"
        else:
            continue
        text = sanitize_text(payload.get("message", ""), replacements=replacements)
        if text:
            events.append(DialogueEvent(line_number, role, text))
    return events


def _latest_compaction(rows, cutoff_line, replacements):
    latest = None
    for line_number, item in rows:
        if line_number > cutoff_line:
            break
        if item.get("type") != "compacted":
            continue
        payload = item.get("payload") or {}
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            summary = sanitize_text(message, replacements=replacements)
        else:
            readable = []
            for history_item in payload.get("replacement_history") or []:
                if not isinstance(history_item, dict):
                    continue
                role = history_item.get("role")
                if history_item.get("type") != "message" or role not in {
                    "user",
                    "assistant",
                }:
                    continue
                text = sanitize_text(
                    _content_text(history_item.get("content")),
                    replacements=replacements,
                )
                if text:
                    readable.append(f"[{role}] {text}")
            summary = "\n".join(readable)
        if summary:
            latest = (line_number, summary)
    return latest


def _render_events(events):
    if not events:
        return "[dialogue empty]"
    return "\n\n".join(
        f"[event {index} | {event.role}]\n{event.text}"
        for index, event in enumerate(events, start=1)
    )


def _tail_events(events, budget):
    selected = []
    for event in reversed(events):
        candidate = [event, *selected]
        rendered = _render_events(candidate)
        if selected and len(rendered) > budget:
            break
        selected = candidate
        if len(rendered) >= budget:
            break
    return _render_events(selected)


def _validate_probe(case_id, probe):
    if not isinstance(probe, dict):
        raise LongContextEvaluationError(f"{case_id}.probe must be an object")
    probe_type = probe.get("type")
    if probe_type not in {"attention_qa", "constraint_state"}:
        raise LongContextEvaluationError(f"{case_id}.probe has unsupported type")
    _require_text(probe.get("question"), f"{case_id}.probe.question")
    if probe_type == "attention_qa":
        expected = probe.get("expected") or {}
        groups = expected.get("required_groups")
        if not isinstance(groups, list) or not groups:
            raise LongContextEvaluationError(
                f"{case_id}.probe.expected.required_groups must be non-empty"
            )
        for group in groups:
            if not isinstance(group, list) or not all(
                isinstance(value, str) and value.strip() for value in group
            ):
                raise LongContextEvaluationError(
                    f"{case_id}.probe required groups must contain aliases"
                )
        forbidden = expected.get("forbidden_terms", [])
        if not isinstance(forbidden, list) or not all(
            isinstance(value, str) and value.strip() for value in forbidden
        ):
            raise LongContextEvaluationError(
                f"{case_id}.probe.expected.forbidden_terms must contain strings"
            )


def load_selection_manifest(path, codex_root):
    payload = _load_json(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise LongContextEvaluationError("unsupported selection manifest schema")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise LongContextEvaluationError("selection manifest requires cases")
    seen = set()
    normalized = []
    for index, raw in enumerate(cases):
        if not isinstance(raw, dict):
            raise LongContextEvaluationError(f"case {index} must be an object")
        case_id = _require_identifier(raw.get("id"), f"case {index}.id")
        if case_id in seen:
            raise LongContextEvaluationError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        source_path = _safe_source_path(codex_root, raw.get("source"))
        try:
            cutoff_line = int(raw.get("cutoff_line") or 0)
        except (TypeError, ValueError) as exc:
            raise LongContextEvaluationError(
                f"{case_id}.cutoff_line must be an integer"
            ) from exc
        if cutoff_line < 1:
            raise LongContextEvaluationError(f"{case_id}.cutoff_line must be positive")
        _safe_replacements(raw.get("redact_terms"), f"{case_id}.redact_terms")
        _validate_probe(case_id, raw.get("probe"))
        normalized.append({**raw, "_source_path": source_path})
    return {
        "selection_id": _require_identifier(
            payload.get("selection_id"), "selection_id"
        ),
        "cases": normalized,
    }


def _source_stats(rows, cutoff_line):
    user_turns = 0
    compact_count = 0
    max_input = 0
    context_window = 0
    for line_number, item in rows:
        if line_number > cutoff_line:
            break
        payload = item.get("payload") or {}
        if item.get("type") == "event_msg":
            payload_type = payload.get("type")
            if payload_type == "user_message":
                user_turns += 1
            elif payload_type == "context_compacted":
                compact_count += 1
            elif payload_type == "token_count":
                info = payload.get("info") or {}
                usage = info.get("last_token_usage") or {}
                max_input = max(max_input, int(usage.get("input_tokens") or 0))
                context_window = max(
                    context_window, int(info.get("model_context_window") or 0)
                )
    return {
        "user_turns": user_turns,
        "context_compactions": compact_count,
        "max_input_tokens": max_input,
        "model_context_window": context_window,
        "max_context_utilization": (
            max_input / context_window if context_window else None
        ),
    }


def _require_visible_position(value, field, cutoff_line, visible_lines):
    try:
        position = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise LongContextEvaluationError(f"{field} must be an integer") from exc
    if position < 1 or position > cutoff_line:
        raise LongContextEvaluationError(
            f"{field} must be a visible source line at or before cutoff_line"
        )
    if position not in visible_lines:
        raise LongContextEvaluationError(
            f"{field} does not point to a visible dialogue event"
        )
    return position


def _constraint_rows(raw_case, cutoff_line, replacements, visible_lines):
    constraints = raw_case.get("constraints") or []
    normalized = []
    seen = set()
    for index, item in enumerate(constraints):
        if not isinstance(item, dict):
            raise LongContextEvaluationError(
                f"{raw_case['id']}.constraints[{index}] must be an object"
            )
        constraint_id = _require_identifier(
            item.get("id"), f"{raw_case['id']}.constraints[{index}].id"
        )
        if constraint_id in seen:
            raise LongContextEvaluationError(
                f"duplicate constraint id in {raw_case['id']}: {constraint_id}"
            )
        seen.add(constraint_id)
        status = item.get("status")
        if status not in {"active", "superseded", "revoked"}:
            raise LongContextEvaluationError(
                f"{raw_case['id']}.{constraint_id} has invalid status"
            )
        introduced_at = _require_visible_position(
            item.get("introduced_at"),
            f"{raw_case['id']}.{constraint_id}.introduced_at",
            cutoff_line,
            visible_lines,
        )
        changed_at = item.get("changed_at")
        if status == "active" and changed_at is not None:
            raise LongContextEvaluationError(
                f"{raw_case['id']}.{constraint_id}.active constraint cannot have changed_at"
            )
        if status != "active" and changed_at is None:
            raise LongContextEvaluationError(
                f"{raw_case['id']}.{constraint_id} must record when it changed"
            )
        if changed_at is not None:
            changed_at = _require_visible_position(
                changed_at,
                f"{raw_case['id']}.{constraint_id}.changed_at",
                cutoff_line,
                visible_lines,
            )
            if changed_at < introduced_at:
                raise LongContextEvaluationError(
                    f"{raw_case['id']}.{constraint_id}.changed_at precedes introduction"
                )
        supersedes = item.get("supersedes")
        if supersedes is not None:
            supersedes = _require_identifier(
                supersedes, f"{raw_case['id']}.{constraint_id}.supersedes"
            )
        normalized.append(
            {
                "id": constraint_id,
                "text": sanitize_text(
                    _require_text(
                        item.get("text"), f"{raw_case['id']}.{constraint_id}.text"
                    ),
                    replacements=replacements,
                ),
                "introduced_at": introduced_at,
                "changed_at": changed_at,
                "status": status,
                "supersedes": supersedes,
            }
        )
    by_id = {item["id"]: item for item in normalized}
    for item in normalized:
        supersedes = item["supersedes"]
        if supersedes is None:
            continue
        prior = by_id.get(supersedes)
        if prior is None or prior["changed_at"] is None:
            raise LongContextEvaluationError(
                f"{raw_case['id']}.{item['id']}.supersedes must reference a changed constraint"
            )
        if prior["changed_at"] > item["introduced_at"]:
            raise LongContextEvaluationError(
                f"{raw_case['id']}.{item['id']}.supersedes precedes the prior change"
            )
    return normalized


def _memory_event_rows(raw_case, cutoff_line, replacements, visible_lines):
    rows = []
    for index, item in enumerate(raw_case.get("memory_events") or []):
        if not isinstance(item, dict):
            raise LongContextEvaluationError(
                f"{raw_case['id']}.memory_events[{index}] must be an object"
            )
        rows.append(
            {
                "at": _require_visible_position(
                    item.get("at"),
                    f"{raw_case['id']}.memory_events[{index}].at",
                    cutoff_line,
                    visible_lines,
                ),
                "text": sanitize_text(
                    _require_text(
                        item.get("text"),
                        f"{raw_case['id']}.memory_events[{index}].text",
                    ),
                    replacements=replacements,
                ),
            }
        )
    return rows


def _oracle_evidence(raw_case, cutoff_line, replacements, visible_lines):
    if raw_case["probe"]["type"] != "attention_qa":
        return None
    _require_visible_position(
        raw_case.get("oracle_evidence_at"),
        f"{raw_case['id']}.oracle_evidence_at",
        cutoff_line,
        visible_lines,
    )
    return sanitize_text(
        _require_text(
            raw_case.get("oracle_evidence"), f"{raw_case['id']}.oracle_evidence"
        ),
        replacements=replacements,
    )


def _render_structured_ledger(memory_events, constraints, events):
    lines = ["Structured conversation memory:"]
    for event in memory_events:
        lines.append(f"- At source event {event['at']}: {event['text']}")
    for constraint in constraints:
        lines.append(
            f"- At source event {constraint['introduced_at']}, constraint "
            f"{constraint['id']} was introduced: {constraint['text']}"
        )
        if constraint["status"] != "active":
            lines.append(
                f"- At source event {constraint['changed_at']}, constraint "
                f"{constraint['id']} became {constraint['status']}."
            )
    recent = events[-4:]
    if recent:
        lines.append("\nRecent visible dialogue:")
        lines.append(_render_events(recent))
    return "\n".join(lines)


def _render_oracle(probe, constraints, oracle_evidence):
    probe_type = probe["type"]
    if probe_type == "attention_qa":
        return "Minimal sufficient evidence:\n" + oracle_evidence
    active = [item for item in constraints if item["status"] == "active"]
    return "Current active constraints:\n" + "\n".join(
        f"- {item['id']}: {item['text']}" for item in active
    )


def _safe_probe(probe, replacements):
    safe = {
        "type": probe["type"],
        "question": sanitize_text(probe["question"], replacements=replacements),
    }
    if probe["type"] == "attention_qa":
        expected = probe["expected"]
        safe["expected"] = {
            "required_groups": [
                [sanitize_text(alias, replacements=replacements) for alias in group]
                for group in expected["required_groups"]
            ],
            "forbidden_terms": [
                sanitize_text(term, replacements=replacements)
                for term in expected.get("forbidden_terms", [])
            ],
        }
    return safe


def build_dataset(selection_manifest, codex_root):
    selection = load_selection_manifest(selection_manifest, codex_root)
    dataset_cases = []
    source_groups = {}
    replacements = {}
    replacement_targets = {}
    for raw_case in selection["cases"]:
        case_replacements = _safe_replacements(
            raw_case.get("redact_terms"), f"{raw_case['id']}.redact_terms"
        )
        for source, target in case_replacements.items():
            normalized_source = unicodedata.normalize("NFKC", source).casefold()
            existing = replacement_targets.get(normalized_source)
            if existing is not None and existing != target:
                raise LongContextEvaluationError(
                    f"conflicting global replacement for {source!r}"
                )
            replacement_targets[normalized_source] = target
            replacements[source] = target
    for raw_case in selection["cases"]:
        source_path = raw_case.pop("_source_path")
        source_key = str(source_path)
        source_group = source_groups.setdefault(
            source_key, f"conversation_{len(source_groups) + 1:03d}"
        )
        rows = _read_rollout(source_path)
        cutoff_line = int(raw_case["cutoff_line"])
        if not rows or cutoff_line > rows[-1][0]:
            raise LongContextEvaluationError(
                f"{raw_case['id']}.cutoff_line exceeds the source rollout"
            )
        events = _visible_events(rows, cutoff_line, replacements)
        if not events:
            raise LongContextEvaluationError(
                f"{raw_case['id']} has no visible dialogue before cutoff"
            )
        compaction = _latest_compaction(rows, cutoff_line, replacements)
        if compaction is None:
            raise LongContextEvaluationError(
                f"{raw_case['id']} has no readable compaction before cutoff"
            )
        compact_line, compact_summary = compaction
        post_compact = [event for event in events if event.source_line > compact_line]
        codex_context = "[Codex compaction summary]\n" + compact_summary
        if post_compact:
            codex_context += "\n\n[Dialogue after compaction]\n" + _render_events(
                post_compact
            )
        full_context = _render_events(events)
        budget = max(2000, len(codex_context), len(_render_events(events[-1:])))
        visible_lines = {event.source_line for event in events}
        constraints = _constraint_rows(
            raw_case, cutoff_line, replacements, visible_lines
        )
        memory_events = _memory_event_rows(
            raw_case, cutoff_line, replacements, visible_lines
        )
        safe_probe = _safe_probe(raw_case["probe"], replacements)
        oracle_evidence = _oracle_evidence(
            raw_case, cutoff_line, replacements, visible_lines
        )
        contexts = {
            "full_dialogue": full_context,
            "codex_summary": codex_context,
            "tail_window": _tail_events(events, budget),
            "structured_ledger": _render_structured_ledger(
                memory_events, constraints, events
            ),
            "oracle": _render_oracle(safe_probe, constraints, oracle_evidence),
        }
        for text in contexts.values():
            assert_no_direct_leak(text)
        dataset_cases.append(
            {
                "id": raw_case["id"],
                "source_group": source_group,
                "topic": sanitize_text(
                    _require_text(raw_case.get("topic"), "topic"),
                    replacements=replacements,
                ),
                "risk": sanitize_text(
                    _require_text(raw_case.get("risk"), "risk"),
                    replacements=replacements,
                ),
                "selection_reason": sanitize_text(
                    _require_text(raw_case.get("selection_reason"), "selection_reason"),
                    replacements=replacements,
                ),
                "source_stats": _source_stats(rows, cutoff_line),
                "probe": safe_probe,
                "constraints": constraints,
                "gold_active_constraints": [
                    item["id"] for item in constraints if item["status"] == "active"
                ],
                "context_budget_chars": budget,
                "context_chars": {
                    variant: len(text) for variant, text in contexts.items()
                },
                "contexts": contexts,
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "codex-long-context-dialogue-dataset-v1",
        "protocol_version": PROTOCOL_VERSION,
        "selection_id": selection["selection_id"],
        "privacy": {
            "source_paths_exported": False,
            "reasoning_exported": False,
            "tool_outputs_exported": False,
            "images_exported": False,
            "encrypted_content_exported": False,
            "visible_messages_sanitized": True,
            "annotation_text_sanitized": True,
            "source_groups_dataset_local": True,
        },
        "dataset_hash_definition": (
            "sha256 of canonical pretty JSON with a trailing newline and without "
            "the dataset_sha256 self-field"
        ),
        "variants": list(VARIANTS),
        "cases": dataset_cases,
    }
    refresh_dataset_hash(payload)
    assert_public_payload_safe(payload)
    return payload


def refresh_dataset_hash(payload):
    payload.pop("dataset_sha256", None)
    payload["dataset_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload["dataset_sha256"]


def build_probe_prompt(case, variant):
    if variant not in VARIANTS:
        raise LongContextEvaluationError(f"unsupported replay variant: {variant}")
    context = case["contexts"][variant]
    probe = case["probe"]
    header = (
        "You are participating in a controlled long-conversation evaluation. "
        "Use only the supplied context. Do not use tools or outside knowledge. "
        "Return one JSON object and no Markdown or prose outside it.\n\n"
    )
    if probe["type"] == "attention_qa":
        contract = '{"answer":"your concise answer"}'
        task = f"Question: {probe['question']}"
    else:
        candidates = "\n".join(
            f"- {item['id']}: {item['text']}" for item in case["constraints"]
        )
        contract = '{"active_constraints":["constraint-id"]}'
        task = (
            f"Question: {probe['question']}\n"
            "Candidate constraints (the list does not indicate which are current):\n"
            f"{candidates}"
        )
    return (
        header
        + f"Required JSON schema: {contract}\n\n"
        + f"Conversation context ({variant}):\n<context>\n{context}\n</context>\n\n"
        + task
    )


def _extract_json_value(text):
    text = str(text or "").strip()

    def reject_duplicate_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise LongContextEvaluationError(
                    "model response has a duplicate JSON key"
                )
            value[key] = item
        return value

    try:
        return json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise LongContextEvaluationError(
            "model response must contain only one valid JSON value"
        ) from exc


def parse_probe_response(text, case):
    value = _extract_json_value(text)
    if not isinstance(value, dict):
        raise LongContextEvaluationError("model response must be a JSON object")
    if case["probe"]["type"] == "attention_qa":
        if set(value) != {"answer"}:
            raise LongContextEvaluationError(
                "attention response must contain only the answer field"
            )
        answer = value.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise LongContextEvaluationError("attention response requires answer")
        return {"answer": sanitize_text(answer)}
    if set(value) != {"active_constraints"}:
        raise LongContextEvaluationError(
            "state response must contain only the active_constraints field"
        )
    selected = value.get("active_constraints")
    if not isinstance(selected, list) or any(
        not isinstance(item, str) for item in selected
    ):
        raise LongContextEvaluationError(
            "state response requires active_constraints string array"
        )
    if any(not item or item != item.strip() for item in selected):
        raise LongContextEvaluationError(
            "state response constraint IDs must be non-empty and unpadded"
        )
    if len(set(selected)) != len(selected):
        raise LongContextEvaluationError(
            "state response contains duplicate constraint IDs"
        )
    if any(not SAFE_ID_RE.fullmatch(item) for item in selected):
        raise LongContextEvaluationError(
            "state response constraint IDs must use the declared identifier format"
        )
    return {"active_constraints": list(selected)}


def _normalize(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(value.split())


def _contains_alias(answer, alias):
    normalized_alias = _normalize(alias)
    if normalized_alias.isdigit():
        return (
            re.search(rf"(?<!\d){re.escape(normalized_alias)}(?!\d)", answer)
            is not None
        )
    return normalized_alias in answer


def score_probe(case, parsed):
    probe = case["probe"]
    if not isinstance(parsed, dict):
        if probe["type"] == "attention_qa":
            return {
                "correct": False,
                "format_correct": False,
                "parse_valid": False,
                "required_groups_matched": 0,
                "required_groups_total": len(probe["expected"]["required_groups"]),
                "forbidden_matches": [],
            }
        expected = set(case["gold_active_constraints"])
        return {
            "correct": False,
            "format_correct": False,
            "parse_valid": False,
            "precision": 0.0,
            "recall": 0.0 if expected else 1.0,
            "f1": 0.0,
            "stale_selected": 0,
            "unknown_selected": [],
            "semantic_repairs": {},
            "inactive_total": len(case["constraints"]) - len(expected),
            "false_positive": 0,
            "false_negative": len(expected),
        }
    if probe["type"] == "attention_qa":
        answer = _normalize(parsed["answer"])
        expected = probe["expected"]
        groups = expected["required_groups"]
        matched = [
            any(_contains_alias(answer, alias) for alias in group) for group in groups
        ]
        forbidden = [
            term
            for term in expected.get("forbidden_terms", [])
            if _contains_alias(answer, term)
        ]
        return {
            "correct": all(matched) and not forbidden,
            "format_correct": True,
            "parse_valid": True,
            "required_groups_matched": sum(matched),
            "required_groups_total": len(groups),
            "forbidden_matches": forbidden,
        }
    expected = set(case["gold_active_constraints"])
    predicted_raw = set(parsed["active_constraints"])
    valid_ids = {item["id"] for item in case["constraints"]}
    unknown = sorted(predicted_raw - valid_ids)
    repairs = {}
    predicted = set(predicted_raw & valid_ids)
    for unknown_id in unknown:
        candidates = [
            valid_id
            for valid_id in valid_ids
            if _edit_distance(unknown_id, valid_id) <= 1
        ]
        if len(candidates) == 1:
            repairs[unknown_id] = candidates[0]
            predicted.add(candidates[0])
        else:
            predicted.add(unknown_id)
    true_positive = len(expected & predicted)
    false_positive = len(predicted - expected)
    false_negative = len(expected - predicted)
    precision = (
        true_positive / len(predicted) if predicted else (1.0 if not expected else 0.0)
    )
    recall = (
        true_positive / len(expected) if expected else (1.0 if not predicted else 0.0)
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    inactive = {item["id"] for item in case["constraints"]} - expected
    return {
        "correct": expected == predicted,
        "format_correct": expected == predicted_raw,
        "parse_valid": True,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "stale_selected": len(predicted & inactive),
        "unknown_selected": unknown,
        "semantic_repairs": repairs,
        "inactive_total": len(inactive),
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _edit_distance(left, right):
    left = str(left)
    right = str(right)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _safe_provider_metadata(metadata):
    metadata = dict(metadata or {})
    return {
        key: value
        for key, value in metadata.items()
        if key in SAFE_PROVIDER_METADATA_FIELDS
        and isinstance(value, (bool, int, float))
    }


def score_first_attempt(case, attempts):
    if not attempts:
        return score_probe(case, None)
    parsed = attempts[0].get("parsed")
    if not isinstance(parsed, dict):
        return score_probe(case, None)
    return score_probe(case, parsed)


def run_probe(client, prompt, case, max_output_tokens=800, parse_retries=1):
    attempts = []
    current_prompt = prompt
    for attempt in range(int(parse_retries) + 1):
        started = time.perf_counter()
        completion = client.complete(current_prompt, max_new_tokens=max_output_tokens)
        if not isinstance(completion, ModelCompletion):
            raise TypeError("long-context evaluation clients must return ModelCompletion")
        response = completion.text
        elapsed = time.perf_counter() - started
        record = {
            "attempt": attempt + 1,
            "elapsed_seconds": elapsed,
            "response_chars": len(response),
            "provider_metadata": _safe_provider_metadata(completion.metadata),
        }
        try:
            parsed = parse_probe_response(response, case)
        except LongContextEvaluationError as exc:
            record["parse_error"] = sanitize_text(str(exc))
            record["parse_valid"] = False
            record["raw_response"] = sanitize_text(response)
            attempts.append(record)
            if attempt >= int(parse_retries):
                return None, attempts
            current_prompt = (
                prompt
                + "\n\nYour previous response was invalid: "
                + str(exc)
                + " Return the complete JSON object again."
            )
            continue
        record["parse_valid"] = True
        record["parsed"] = parsed
        record["raw_response"] = sanitize_text(response)
        attempts.append(record)
        return parsed, attempts
    return None, attempts


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _first_attempt_score(row):
    value = row.get("first_attempt_score")
    return value if isinstance(value, dict) else row["score"]


def aggregate_results(dataset, rows, expected_variants=VARIANTS):
    expected_variants = tuple(dict.fromkeys(expected_variants))
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], {})[row["variant"]] = row
    attribution = []
    for case in dataset["cases"]:
        variants = by_case.get(case["id"], {})
        oracle = variants.get("oracle")
        full = variants.get("full_dialogue")
        missing = [variant for variant in expected_variants if variant not in variants]
        format_failed = [
            variant
            for variant, row in variants.items()
            if not row["score"].get("format_correct", row["score"]["correct"])
        ]
        first_format_failed = [
            variant
            for variant, row in variants.items()
            if not _first_attempt_score(row).get(
                "format_correct", _first_attempt_score(row)["correct"]
            )
        ]
        if missing:
            primary = "incomplete"
        elif "oracle" not in expected_variants:
            primary = "ungated"
        elif oracle is None:
            primary = "incomplete"
        elif not oracle["score"]["correct"]:
            primary = "capability_fail"
        elif (
            "full_dialogue" in expected_variants
            and full
            and not full["score"]["correct"]
        ):
            primary = "full_context_failure"
        else:
            failed = [
                variant
                for variant in ("codex_summary", "tail_window", "structured_ledger")
                if variant in expected_variants
                and variant in variants
                and not variants[variant]["score"]["correct"]
            ]
            if failed:
                primary = "compression_sensitive"
            elif format_failed or first_format_failed:
                primary = "output_format_sensitive"
            else:
                primary = "robust"
        attribution.append(
            {
                "case_id": case["id"],
                "source_group": case["source_group"],
                "primary": primary,
                "missing_variants": missing,
                "failed_variants": [
                    variant
                    for variant, row in variants.items()
                    if not row["score"]["correct"]
                ],
                "format_failed_variants": format_failed,
                "first_attempt_format_failed_variants": first_format_failed,
            }
        )

    summaries = []
    oracle_expected = "oracle" in expected_variants
    for variant in (item for item in VARIANTS if item in expected_variants):
        variant_rows = [row for row in rows if row["variant"] == variant]
        gated_rows = []
        for row in variant_rows:
            oracle = by_case.get(row["case_id"], {}).get("oracle")
            if oracle and oracle["score"]["correct"]:
                gated_rows.append(row)
        metric_rows = (
            gated_rows if oracle_expected and variant != "oracle" else variant_rows
        )
        attention = [row for row in metric_rows if row["probe_type"] == "attention_qa"]
        state = [row for row in metric_rows if row["probe_type"] == "constraint_state"]
        summaries.append(
            {
                "variant": variant,
                "case_count": len(variant_rows),
                "expected_case_count": len(dataset["cases"]),
                "missing_case_count": len(dataset["cases"]) - len(variant_rows),
                "oracle_gated_count": len(gated_rows),
                "metric_denominator_count": len(metric_rows),
                "gating_mode": "oracle" if oracle_expected else "ungated",
                "accuracy": _mean(row["score"]["correct"] for row in metric_rows),
                "format_accuracy": _mean(
                    row["score"].get("format_correct", row["score"]["correct"])
                    for row in metric_rows
                ),
                "first_attempt_format_accuracy": _mean(
                    _first_attempt_score(row).get(
                        "format_correct", _first_attempt_score(row)["correct"]
                    )
                    for row in metric_rows
                ),
                "parser_retry_recovered_count": sum(
                    bool(row["score"].get("format_correct", row["score"]["correct"]))
                    and not bool(
                        _first_attempt_score(row).get(
                            "format_correct", _first_attempt_score(row)["correct"]
                        )
                    )
                    for row in metric_rows
                ),
                "attention_accuracy": _mean(
                    row["score"]["correct"] for row in attention
                ),
                "state_accuracy": _mean(row["score"]["correct"] for row in state),
                "state_recall": _mean(row["score"].get("recall", 0.0) for row in state),
                "stale_selection_rate": (
                    sum(row["score"].get("stale_selected", 0) for row in state)
                    / sum(row["score"].get("inactive_total", 0) for row in state)
                    if sum(row["score"].get("inactive_total", 0) for row in state)
                    else None
                ),
                "avg_prompt_chars": _mean(row["prompt_chars"] for row in variant_rows),
            }
        )
    parser_extra_attempts = sum(
        max(0, len(row.get("attempts") or []) - 1) for row in rows
    )
    provider_transport_extra_attempts = sum(
        max(
            0,
            int(attempt.get("provider_metadata", {}).get("request_attempts") or 1) - 1,
        )
        for row in rows
        for attempt in row.get("attempts") or []
    )
    return {
        "variant_summaries": summaries,
        "attribution": attribution,
        "run_diagnostics": {
            "parser_extra_attempts": parser_extra_attempts,
            "provider_transport_extra_attempts": provider_transport_extra_attempts,
        },
    }


def render_markdown_report(payload):
    def percentage(value):
        return "N/A" if value is None else f"{value:.2%}"

    def rounded(value):
        return "N/A" if value is None else f"{value:.0f}"

    lines = [
        "# Long-Context Dialogue Replay Evaluation",
        "",
        f"- Provider: `{payload['provider']['provider']}`",
        f"- Model: `{payload['provider']['model']}`",
        "- Dataset canonical hash (excluding its self-field): "
        f"`{payload['dataset_sha256']}`",
        f"- Cases: {len(payload['dataset']['cases'])}",
        "- Scope: sanitized visible dialogue only; private reasoning, raw tool output, images, and encrypted content are excluded.",
        "- Parser extra attempts: "
        f"{payload['summary']['run_diagnostics']['parser_extra_attempts']}",
        "- Provider transport extra attempts: "
        f"{payload['summary']['run_diagnostics']['provider_transport_extra_attempts']}",
        "",
        "| Variant | Cases | Missing | Scored | Oracle-gated | Semantic accuracy | Final format exact | First-attempt format exact | Parser retry recoveries | Attention | State | State recall | Stale selection | Avg prompt chars |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]["variant_summaries"]:
        lines.append(
            f"| `{row['variant']}` | {row['case_count']} | {row['missing_case_count']} | "
            f"{row['metric_denominator_count']} | {row['oracle_gated_count']} | "
            f"{percentage(row['accuracy'])} | {percentage(row['format_accuracy'])} | "
            f"{percentage(row['first_attempt_format_accuracy'])} | "
            f"{row['parser_retry_recovered_count']} | "
            f"{percentage(row['attention_accuracy'])} | "
            f"{percentage(row['state_accuracy'])} | {percentage(row['state_recall'])} | "
            f"{percentage(row['stale_selection_rate'])} | {rounded(row['avg_prompt_chars'])} |"
        )
    lines.extend(["", "## Case Attribution", ""])
    for row in payload["summary"]["attribution"]:
        failed = ", ".join(row["failed_variants"]) or "none"
        format_failed = ", ".join(row["format_failed_variants"]) or "none"
        first_format_failed = (
            ", ".join(row["first_attempt_format_failed_variants"]) or "none"
        )
        missing = ", ".join(row["missing_variants"]) or "none"
        lines.append(
            f"- `{row['case_id']}`: `{row['primary']}`; missing: {missing}; "
            f"semantic failures: {failed}; final format failures: {format_failed}; "
            f"first-attempt format failures: {first_format_failed}."
        )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- `full_dialogue` contains visible user/assistant messages, not hidden reasoning or raw tool outputs.",
            "- `codex_summary` uses the latest readable Codex compaction summary plus later visible messages.",
            "- `structured_ledger` is built from supervised annotations and is an upper-bound memory mechanism, not an automatic summarizer result.",
            "- A completed oracle failure is treated as a capability failure and removed from forgetting-rate denominators; an unrun oracle is marked incomplete.",
            "- A uniquely repairable constraint ID typo at edit distance one counts as semantically correct but remains a strict format failure.",
            "- Final format exact requires one JSON object with exactly the requested field and is measured after configured parser retries; first-attempt format exact and parser retry recoveries are reported separately.",
            "- Parser retries repair schema/JSON failures; provider transport retries are counted separately and do not change scoring semantics.",
            "- This pilot is a selected-case audit, not a population estimate over all Codex sessions.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_dataset(dataset, path):
    assert_public_payload_safe(dataset)
    _atomic_write_text(path, _canonical_json(dataset))


def write_results(payload, json_path, markdown_path):
    assert_public_payload_safe(payload)
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    _atomic_write_text(json_path, _canonical_json(payload))
    _atomic_write_text(markdown_path, render_markdown_report(payload))


def shuffled_variants(seed):
    variants = list(VARIANTS)
    random.Random(int(seed)).shuffle(variants)
    return tuple(variants)
