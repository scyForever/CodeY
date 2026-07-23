"""Rule-supervised post-task learning and patch rollout.

The loop intentionally consumes only structured runtime signals. It never parses
the model's final prose into knowledge and never stores tool output, commands, or
source diffs. This keeps self-evolution reviewable and bounded by deterministic
rules.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .hybrid import HybridEvolutionAdvisor


OUTCOMES = {"correct", "incorrect", "partial", "harmful"}
PATCH_TYPES = {
    "policy",
    "strategy",
    "action_chain",
    "knowledge_definition",
    "knowledge_experience",
}
PATCH_STATUSES = {"draft", "shadow", "review_required", "active", "expired"}
AUTO_ACTIVATE_TYPES = {"strategy", "action_chain", "knowledge_experience"}
REVIEW_REQUIRED_TYPES = {"policy", "knowledge_definition"}
ALLOWED_TRANSITIONS = {
    "draft": {"shadow", "review_required"},
    "shadow": {"active", "expired"},
    "review_required": {"active", "expired"},
    "active": {"expired"},
    "expired": set(),
}
REDACTED_PATH = "<redacted-path>"
SECRET_SHAPED_PATH_PATTERN = re.compile(
    r"(?i)(api[_ .-]?key|token|secret|password|sk-[A-Za-z0-9_-]{6,})"
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _clip(value, limit=240):
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _safe_path(value):
    text = _clip(value, 220)
    if text and SECRET_SHAPED_PATH_PATTERN.search(text):
        return REDACTED_PATH
    return text


def _dedupe(items):
    seen = set()
    result = []
    for item in items:
        marker = json.dumps(item, sort_keys=True, ensure_ascii=True) if isinstance(item, dict) else str(item)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result


@dataclass(frozen=True)
class EvolutionThresholds:
    """Safety and rollout thresholds for shadow patches."""

    canary_fraction: float = 0.20
    min_canary_hits: int = 3
    min_canary_hit_rate: float = 0.10
    min_canary_success_rate: float = 0.80
    expiry_min_hits: int = 100
    expiry_success_rate: float = 0.40

    @classmethod
    def from_value(cls, value=None):
        if value is None:
            result = cls()
        elif isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            allowed = set(cls.__dataclass_fields__)
            unknown = sorted(set(value) - allowed)
            if unknown:
                raise ValueError(f"unknown evolution threshold(s): {', '.join(unknown)}")
            result = cls(**dict(value))
        else:
            raise TypeError("evolution_thresholds must be a mapping or EvolutionThresholds")
        result.validate()
        return result

    def validate(self):
        for name in (
            "canary_fraction",
            "min_canary_hit_rate",
            "min_canary_success_rate",
            "expiry_success_rate",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if int(self.min_canary_hits) < 1:
            raise ValueError("min_canary_hits must be at least 1")
        if int(self.expiry_min_hits) < 1:
            raise ValueError("expiry_min_hits must be at least 1")

    def to_dict(self):
        return asdict(self)


class TraceCollector:
    """Reduce a run to non-sensitive events suitable for learning."""

    @staticmethod
    def collect(task_state, tool_events, stale_paths=()):
        events = []
        for index, raw in enumerate(tool_events or (), start=1):
            metadata = dict(raw.get("metadata", {}) or {})
            args = dict(raw.get("args", {}) or {})
            path = _safe_path(args.get("path", ""))
            affected_paths = [
                _safe_path(item)
                for item in metadata.get("affected_paths", [])
                if str(item or "").strip()
            ]
            content = str(raw.get("content", "") or "")
            events.append(
                {
                    "evidence_id": f"tool_{index:03d}",
                    "name": _clip(raw.get("name", ""), 80),
                    "status": _clip(metadata.get("tool_status", ""), 40) or "unknown",
                    "error_code": _clip(metadata.get("tool_error_code", ""), 80),
                    "security_event_type": _clip(metadata.get("security_event_type", ""), 80),
                    "risk_level": _clip(metadata.get("risk_level", ""), 20),
                    "path": path,
                    "affected_paths": affected_paths,
                    "workspace_changed": bool(metadata.get("workspace_changed", False)),
                    "has_content": bool(content.strip()) and not content.lstrip().lower().startswith("error:"),
                }
            )
        safe_stale_paths = sorted({_safe_path(path) for path in stale_paths if str(path or "").strip()})
        return {
            "run_id": str(task_state.run_id),
            "task_id": str(task_state.task_id),
            "task_status": str(task_state.status),
            "stop_reason": str(task_state.stop_reason),
            "terminal_evidence_id": "task_terminal",
            "scope": {
                "skill_name": str(task_state.skill_name or ""),
                "route_id": str(task_state.route_id or ""),
            },
            "tool_events": events,
            "stale_paths": safe_stale_paths,
            "stale_evidence": [
                {"evidence_id": f"stale_{index:03d}", "path": path}
                for index, path in enumerate(safe_stale_paths, start=1)
            ],
        }


class OutcomeEvaluator:
    """Normalize terminal runtime signals to four outcome labels."""

    @staticmethod
    def evaluate(trace):
        events = list(trace.get("tool_events", []))
        harmful_events = [
            event for event in events if event["status"] == "partial_success" and event["workspace_changed"]
        ]
        if harmful_events:
            return {
                "label": "harmful",
                "reason": "failed_action_changed_workspace",
                "evidence_refs": [event["evidence_id"] for event in harmful_events],
                "decisive": True,
                "decision_source": "rules",
            }
        if trace.get("task_status") == "failed":
            return {
                "label": "incorrect",
                "reason": trace.get("stop_reason") or "task_failed",
                "evidence_refs": [trace.get("terminal_evidence_id", "task_terminal")],
                "decisive": True,
                "decision_source": "rules",
            }
        if trace.get("task_status") != "completed":
            return {
                "label": "incorrect",
                "reason": trace.get("stop_reason") or "task_not_completed",
                "evidence_refs": [trace.get("terminal_evidence_id", "task_terminal")],
                "decisive": True,
                "decision_source": "rules",
            }
        failed_events = [
            event
            for event in events
            if event["status"] in {"partial_success", "error", "rejected", "unknown"}
        ]
        if failed_events:
            return {
                "label": "partial",
                "reason": "completed_after_tool_failure",
                "evidence_refs": [event["evidence_id"] for event in failed_events],
                "decisive": False,
                "decision_source": "rules",
            }
        return {
            "label": "correct",
            "reason": "completed_without_structured_failure",
            "evidence_refs": [trace.get("terminal_evidence_id", "task_terminal")],
            "decisive": True,
            "decision_source": "rules",
        }


class RootCauseAnalyzer:
    """Classify structured failures into policy, strategy, chain, or execution."""

    POLICY_CODES = {"tool_not_allowed", "approval_denied"}

    @classmethod
    def analyze(cls, trace, outcome):
        del outcome
        candidates = cls.candidates(trace)
        return candidates[0] if candidates else None

    @classmethod
    def candidates(cls, trace):
        """Return ordered, evidence-backed candidates for optional LLM disambiguation."""
        failures = [event for event in trace.get("tool_events", []) if event.get("status") != "ok"]
        if not failures:
            if trace.get("stop_reason") == "retry_limit_reached":
                return [
                    cls._result(
                        "strategy",
                        "retry_limit_reached",
                        trace.get("terminal_evidence_id", "task_terminal"),
                    )
                ]
            if trace.get("stop_reason") == "step_limit_reached":
                return [
                    cls._result(
                        "chain",
                        "step_limit_reached",
                        trace.get("terminal_evidence_id", "task_terminal"),
                    )
                ]
            return []

        policy_event = next(
            (
                event
                for event in failures
                if event.get("error_code") in cls.POLICY_CODES or event.get("security_event_type")
            ),
            None,
        )
        if policy_event:
            return [cls._event_result("policy", policy_event)]

        candidates = []
        strategy_event = next(
            (event for event in failures if event.get("error_code") == "repeated_identical_call"),
            None,
        )
        if strategy_event or trace.get("stop_reason") == "retry_limit_reached":
            candidates.append(cls._event_result("strategy", strategy_event or failures[-1]))

        if trace.get("stop_reason") == "step_limit_reached" or len(trace.get("tool_events", [])) > 1:
            candidates.append(cls._event_result("chain", failures[-1]))
        candidates.append(cls._event_result("execution", failures[-1]))
        return candidates

    @staticmethod
    def _result(level, trigger, evidence_ref="task_terminal"):
        return {
            "level": level,
            "trigger": trigger,
            "tool": "",
            "error_code": "",
            "security_event_type": "",
            "evidence_refs": [evidence_ref],
        }

    @staticmethod
    def _event_result(level, event):
        return {
            "level": level,
            "trigger": event.get("error_code") or event.get("security_event_type") or event.get("status"),
            "tool": event.get("name", ""),
            "error_code": event.get("error_code", ""),
            "security_event_type": event.get("security_event_type", ""),
            "evidence_refs": [event.get("evidence_id", "")],
        }


class PatchGenerator:
    """Create candidates from controlled templates, never from model prose."""

    ARCHITECTURE_MARKERS = ("architecture", "design", "boundary", "overview", "system")

    def generate(self, trace, outcome, root_cause):
        candidates = []
        if root_cause:
            candidates.append(self._failure_patch(trace, outcome, root_cause))
        if trace.get("stale_paths"):
            candidates.append(self._stale_patch(trace, outcome))
        if outcome.get("label") == "correct":
            candidates.extend(self._knowledge_patches(trace, outcome))
        return _dedupe(candidates)

    def knowledge_facts(self, trace, outcome):
        if outcome.get("label") != "correct":
            return []
        facts = []
        seen_paths = set()
        for event in trace.get("tool_events", []):
            path = event.get("path", "")
            if event.get("status") != "ok" or not path or path == REDACTED_PATH or not event.get("has_content"):
                continue
            if event.get("name") not in {"read_file", "write_file", "patch_file"}:
                continue
            if path in seen_paths:
                continue
            seen_paths.add(path)
            facts.append(
                {
                    "kind": "verified_repository_path",
                    "path": path,
                    "tool": event.get("name", ""),
                    "evidence_id": event.get("evidence_id", ""),
                }
            )
        return _dedupe(facts)[:2]

    def _failure_patch(self, trace, outcome, root_cause):
        level = root_cause["level"]
        tool = root_cause.get("tool") or "the selected tool"
        patch_type = {
            "policy": "policy",
            "strategy": "strategy",
            "chain": "action_chain",
            "execution": "strategy",
        }[level]
        if level == "policy":
            security_event = root_cause.get("security_event_type", "")
            if security_event == "path_escape":
                action = "Resolve tool paths inside the workspace before execution; do not retry an escaping path."
            elif root_cause.get("error_code") == "approval_denied":
                action = "Do not retry a denied risky action; request approval or choose a permitted read-only action."
            else:
                action = "Use only actions permitted by the current tool and approval policy."
        elif level == "strategy":
            action = "Gather a new observation or change approach before repeating an unsuccessful action."
        elif level == "chain":
            action = "Add a verification step before continuing a chain after an unsuccessful tool action."
        else:
            action = f"Validate arguments and prerequisites for {tool} before execution."
        return self._candidate(
            trace,
            outcome,
            root_cause,
            patch_type=patch_type,
            correction_kind=f"{level}_guard",
            action=action,
            trigger_conditions=(
                [{"signal": "tool_name", "equals": root_cause.get("tool", "")}]
                if root_cause.get("tool")
                else [
                    {
                        "signal": "task_scope",
                        "equals": trace.get("scope", {}).get("route_id") or "workspace",
                    }
                ]
            ),
            target_tool=root_cause.get("tool", ""),
            evidence_refs=root_cause.get("evidence_refs", []),
        )

    def _stale_patch(self, trace, outcome):
        paths = list(trace.get("stale_paths", []))[:3]
        evidence_by_path = {
            item.get("path", ""): item.get("evidence_id", "")
            for item in trace.get("stale_evidence", [])
        }
        path_text = ", ".join(f"`{path}`" for path in paths)
        return self._candidate(
            trace,
            outcome,
            {"level": "execution", "trigger": "stale_memory", "tool": ""},
            patch_type="strategy",
            correction_kind="freshness_guard",
            action=f"Revalidate {path_text} before relying on a previous memory summary.",
            trigger_conditions=[{"signal": "path", "equals": path} for path in paths],
            evidence_refs=[evidence_by_path[path] for path in paths if evidence_by_path.get(path)],
        )

    def _knowledge_patches(self, trace, outcome):
        patches = []
        for fact in self.knowledge_facts(trace, outcome):
            path = fact["path"]
            lowered = path.lower()
            is_definition = lowered.endswith(".md") and any(marker in lowered for marker in self.ARCHITECTURE_MARKERS)
            route_label = trace.get("scope", {}).get("route_id") or "workspace"
            if is_definition:
                patch_type = "knowledge_definition"
                kind = "architecture_reference"
                action = f"Treat `{path}` as a verified architecture reference for route `{route_label}`."
            else:
                patch_type = "knowledge_experience"
                kind = "verified_path_usage"
                action = f"Reuse `{path}` as a verified repository path for route `{route_label}`."
            patches.append(
                self._candidate(
                    trace,
                    outcome,
                    None,
                    patch_type=patch_type,
                    correction_kind=kind,
                    action=action,
                    trigger_conditions=[{"signal": "path", "equals": path}],
                    target_tool=fact.get("tool", ""),
                    evidence_refs=[fact.get("evidence_id", "")],
                )
            )
        return patches

    @staticmethod
    def _candidate(
        trace,
        outcome,
        root_cause,
        *,
        patch_type,
        correction_kind,
        action,
        trigger_conditions,
        target_tool="",
        evidence_refs=(),
    ):
        scope = dict(trace.get("scope", {}))
        if target_tool:
            scope["target_tool"] = target_tool
        candidate = {
            "type": patch_type,
            "scope": scope,
            "correction": {
                "kind": correction_kind,
                "action": _clip(action, 500),
            },
            "trigger_conditions": list(trigger_conditions),
            "source": {
                "run_id": trace.get("run_id", ""),
                "task_id": trace.get("task_id", ""),
                "outcome": outcome.get("label", ""),
                "root_cause_level": (root_cause or {}).get("level", ""),
                "root_cause_trigger": (root_cause or {}).get("trigger", ""),
                "root_cause_tool": (root_cause or {}).get("tool", ""),
                "proposal_origin": "rules",
                "evidence_refs": [str(item) for item in evidence_refs if str(item)],
            },
        }
        identity_payload = {
            key: candidate[key]
            for key in ("type", "scope", "correction", "trigger_conditions")
        }
        candidate["source"]["rule_candidate_fingerprint"] = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        return candidate


class EvolutionStore:
    """Persist structured patches and derived human-readable knowledge views."""

    SCHEMA_VERSION = 2

    def __init__(self, root):
        self.root = Path(root)
        self.patches_dir = self.root / "patches"

    def list_patches(self):
        if not self.patches_dir.exists():
            return []
        patches = []
        for path in sorted(self.patches_dir.glob("patch_*.json")):
            patches.append(json.loads(path.read_text(encoding="utf-8")))
        return patches

    def load_patch(self, patch_id):
        path = self.patches_dir / f"{patch_id}.json"
        if not path.exists():
            raise KeyError(f"unknown cognitive patch: {patch_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def create_candidate(self, candidate):
        self._validate_candidate(candidate)
        fingerprint_payload = {
            key: candidate[key]
            for key in ("type", "scope", "correction", "trigger_conditions")
        }
        content_fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        rule_fingerprint = str(candidate.get("source", {}).get("rule_candidate_fingerprint", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", rule_fingerprint):
            rule_fingerprint = content_fingerprint
        patch_id = "patch_" + rule_fingerprint[:16]
        path = self.patches_dir / f"{patch_id}.json"
        if path.exists():
            return self.load_patch(patch_id), False
        timestamp = _utc_now()
        generation_reason = (
            "hybrid_supervised_generation"
            if candidate.get("source", {}).get("proposal_origin") == "hybrid_llm"
            else "rule_supervised_generation"
        )
        patch = {
            "schema_version": self.SCHEMA_VERSION,
            "patch_id": patch_id,
            "fingerprint": content_fingerprint,
            "rule_candidate_fingerprint": rule_fingerprint,
            **candidate,
            "status": "draft",
            "metrics": {
                "eligible_count": 0,
                "hit_count": 0,
                "success_count": 0,
                "harmful_count": 0,
                "hit_rate": 0.0,
                "success_rate": 0.0,
            },
            "created_at": timestamp,
            "updated_at": timestamp,
            "history": [
                {
                    "from": "",
                    "to": "draft",
                    "reason": generation_reason,
                    "at": timestamp,
                }
            ],
            "observed_run_ids": [],
            "materialized": False,
        }
        self.save_patch(patch)
        return patch, True

    def save_patch(self, patch):
        patch_id = str(patch.get("patch_id", ""))
        if not patch_id.startswith("patch_"):
            raise ValueError("invalid cognitive patch id")
        if patch.get("status") not in PATCH_STATUSES:
            raise ValueError(f"invalid cognitive patch status: {patch.get('status')}")
        self.patches_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(self.patches_dir / f"{patch_id}.json", patch)
        return patch

    def transition(self, patch, target_status, reason):
        current = str(patch.get("status", ""))
        if target_status not in ALLOWED_TRANSITIONS.get(current, set()):
            raise ValueError(f"invalid cognitive patch transition: {current} -> {target_status}")
        timestamp = _utc_now()
        patch["status"] = target_status
        patch["updated_at"] = timestamp
        patch.setdefault("history", []).append(
            {"from": current, "to": target_status, "reason": str(reason), "at": timestamp}
        )
        if target_status == "active":
            patch["materialized"] = True
            patch["materialized_at"] = timestamp
        self.save_patch(patch)
        return patch

    def refresh_materialized_views(self):
        patches = self.list_patches()
        active_policies = [patch for patch in patches if patch["type"] == "policy" and patch["status"] == "active"]
        active_decisions = [
            patch
            for patch in patches
            if patch["type"] in {"strategy", "action_chain"} and patch["status"] == "active"
        ]
        behavior_path = self.root / "behavior" / "policies.md"
        decisions_path = self.root / "decisions.md"
        if active_policies or behavior_path.exists():
            self._write_collection(behavior_path, "Active Cognitive Policies", active_policies)
        if active_decisions or decisions_path.exists():
            self._write_collection(decisions_path, "Active Cognitive Decisions", active_decisions)

        for patch in patches:
            if patch["type"] not in {"knowledge_definition", "knowledge_experience"}:
                continue
            leaf = "definition" if patch["type"] == "knowledge_definition" else "experience"
            path = self.root / "knowledge" / leaf / f"{patch['patch_id']}.md"
            if patch.get("materialized") and patch["status"] == "active":
                self._write_text_atomic(path, self._render_patch_document(patch))
            elif path.exists():
                path.unlink()

    @staticmethod
    def _validate_candidate(candidate):
        if candidate.get("type") not in PATCH_TYPES:
            raise ValueError(f"invalid cognitive patch type: {candidate.get('type')}")
        if not isinstance(candidate.get("scope"), dict):
            raise ValueError("cognitive patch scope must be an object")
        correction = candidate.get("correction")
        if not isinstance(correction, dict) or not str(correction.get("action", "")).strip():
            raise ValueError("cognitive patch correction must contain an action")
        if not isinstance(candidate.get("trigger_conditions"), list) or not candidate["trigger_conditions"]:
            raise ValueError("cognitive patch requires trigger conditions")

    @staticmethod
    def _write_collection(path, title, patches):
        lines = [f"# {title}", ""]
        if not patches:
            lines.append("- none")
        for patch in patches:
            lines.extend(
                [
                    f"## {patch['patch_id']}",
                    "",
                    f"- scope: `{json.dumps(patch['scope'], sort_keys=True, ensure_ascii=True)}`",
                    f"- trigger_conditions: `{json.dumps(patch['trigger_conditions'], sort_keys=True, ensure_ascii=True)}`",
                    f"- correction: {patch['correction']['action']}",
                    "",
                ]
            )
        EvolutionStore._write_text_atomic(path, "\n".join(lines).rstrip() + "\n")

    @staticmethod
    def _render_patch_document(patch):
        return "\n".join(
            [
                f"# {patch['type'].replace('_', ' ').title()}",
                "",
                f"- patch_id: {patch['patch_id']}",
                f"- status: {patch['status']}",
                f"- scope: `{json.dumps(patch['scope'], sort_keys=True, ensure_ascii=True)}`",
                f"- trigger_conditions: `{json.dumps(patch['trigger_conditions'], sort_keys=True, ensure_ascii=True)}`",
                f"- conclusion: {patch['correction']['action']}",
                "",
            ]
        )

    @staticmethod
    def _write_json_atomic(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temp_name = handle.name
        Path(temp_name).replace(path)

    @staticmethod
    def _write_text_atomic(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        ) as handle:
            handle.write(text)
            temp_name = handle.name
        Path(temp_name).replace(path)


class SafetyGate:
    """Apply status transitions without allowing policy self-promotion."""

    def __init__(self, store, thresholds):
        self.store = store
        self.thresholds = thresholds

    def initialize(self, patch):
        if patch["type"] in REVIEW_REQUIRED_TYPES:
            return self.store.transition(patch, "review_required", "human_review_required")
        return self.store.transition(patch, "shadow", "eligible_for_shadow_rollout")

    def observe(self, patch, *, run_id, eligible, hit, success, harmful):
        observed_run_ids = patch.setdefault("observed_run_ids", [])
        if run_id and run_id in observed_run_ids:
            return patch, None
        if run_id:
            observed_run_ids.append(run_id)
        metrics = patch.setdefault("metrics", {})
        if eligible:
            metrics["eligible_count"] = int(metrics.get("eligible_count", 0)) + 1
        if hit:
            metrics["hit_count"] = int(metrics.get("hit_count", 0)) + 1
            if success:
                metrics["success_count"] = int(metrics.get("success_count", 0)) + 1
            if harmful:
                metrics["harmful_count"] = int(metrics.get("harmful_count", 0)) + 1
        eligible_count = int(metrics.get("eligible_count", 0))
        hit_count = int(metrics.get("hit_count", 0))
        success_count = int(metrics.get("success_count", 0))
        metrics["hit_rate"] = hit_count / eligible_count if eligible_count else 0.0
        metrics["success_rate"] = success_count / hit_count if hit_count else 0.0
        patch["updated_at"] = _utc_now()
        self.store.save_patch(patch)

        status = patch["status"]
        if not hit or status not in {"shadow", "active"}:
            return patch, None
        if harmful:
            return self.store.transition(patch, "expired", "harmful_canary_outcome"), "expired"
        if (
            hit_count >= self.thresholds.expiry_min_hits
            and metrics["success_rate"] < self.thresholds.expiry_success_rate
        ):
            return self.store.transition(patch, "expired", "success_rate_below_expiry_threshold"), "expired"
        if status == "shadow" and patch["type"] in AUTO_ACTIVATE_TYPES:
            if (
                hit_count >= self.thresholds.min_canary_hits
                and metrics["hit_rate"] >= self.thresholds.min_canary_hit_rate
                and metrics["success_rate"] >= self.thresholds.min_canary_success_rate
            ):
                return self.store.transition(patch, "active", "canary_thresholds_met"), "active"
        return patch, None

    def approve(self, patch):
        if patch["status"] != "review_required":
            raise ValueError(f"patch {patch['patch_id']} is not awaiting review")
        return self.store.transition(patch, "active", "human_review_approved")


class CognitiveLoop:
    """Orchestrate trace collection, evaluation, attribution, and rollout."""

    def __init__(self, root, thresholds=None, llm_client=None, llm_config=None):
        self.thresholds = EvolutionThresholds.from_value(thresholds)
        self.store = EvolutionStore(Path(root) / ".codey" / "evolution")
        self.generator = PatchGenerator()
        self.safety_gate = SafetyGate(self.store, self.thresholds)
        self.advisor = HybridEvolutionAdvisor(client=llm_client, config=llm_config)

    def prepare_run(self, task_state):
        eligible_ids = []
        active_ids = []
        shadow_ids = []
        guidance = []
        for patch in self.store.list_patches():
            if patch["status"] not in {"shadow", "active"}:
                continue
            if not self._scope_matches(patch, task_state):
                continue
            eligible_ids.append(patch["patch_id"])
            if patch["status"] == "active":
                active_ids.append(patch["patch_id"])
                guidance.append(("active", patch))
            elif self._selected_for_canary(task_state.run_id, patch["patch_id"]):
                shadow_ids.append(patch["patch_id"])
                guidance.append(("shadow", patch))
        context = {
            "enabled": True,
            "eligible_patch_ids": eligible_ids,
            "active_patch_ids": active_ids,
            "shadow_patch_ids": shadow_ids,
            "thresholds": self.thresholds.to_dict(),
            "llm_advisor": self.advisor.config.to_dict(),
        }
        lines = ["Adaptive cognitive patches:"]
        for mode, patch in guidance[:8]:
            lines.append(f"- [{mode}:{patch['type']}:{patch['patch_id']}] {patch['correction']['action']}")
        return context, "\n".join(lines) if guidance else ""

    def complete_run(self, task_state, tool_events, stale_paths=(), redactor=None):
        trace = TraceCollector.collect(task_state, tool_events, stale_paths=stale_paths)
        if redactor is not None:
            trace["scope"] = redactor(trace.get("scope", {}))
            trace["stale_paths"] = redactor(trace.get("stale_paths", []))
            for item in trace.get("stale_evidence", []):
                item["path"] = redactor(item.get("path", ""))
            for event in trace.get("tool_events", []):
                event["path"] = redactor(event.get("path", ""))
                event["affected_paths"] = redactor(event.get("affected_paths", []))
        rule_outcome = OutcomeEvaluator.evaluate(trace)
        rule_root_cause = RootCauseAnalyzer.analyze(trace, rule_outcome)
        root_candidates = RootCauseAnalyzer.candidates(trace)
        try:
            outcome, root_cause, diagnostic_audit, patch_eligible = self.advisor.diagnose(
                trace,
                rule_outcome,
                rule_root_cause,
                root_candidates,
            )
        except Exception as exc:
            outcome = rule_outcome
            root_cause = rule_root_cause
            patch_eligible = True
            diagnostic_audit = {
                "mode": self.advisor.config.mode,
                "status": "fallback_internal_error",
                "error_type": type(exc).__name__,
            }
        if outcome["label"] not in OUTCOMES:
            raise ValueError(f"invalid normalized outcome: {outcome['label']}")
        transitions = self._observe_existing(task_state, trace, outcome)
        rule_candidates = self.generator.generate(trace, outcome, root_cause)
        try:
            candidates, patch_audit = self.advisor.refine_patches(
                trace,
                rule_candidates,
                patch_eligible=patch_eligible,
            )
        except Exception as exc:
            candidates = rule_candidates
            patch_audit = {
                "mode": self.advisor.config.mode,
                "status": "fallback_internal_error",
                "error_type": type(exc).__name__,
            }
        generated = []
        for candidate in candidates:
            patch, created = self.store.create_candidate(candidate)
            if created:
                patch = self.safety_gate.initialize(patch)
            generated.append(
                {
                    "patch_id": patch["patch_id"],
                    "type": patch["type"],
                    "status": patch["status"],
                    "created": created,
                    "scope": dict(patch["scope"]),
                    "correction": dict(patch["correction"]),
                    "trigger_conditions": list(patch["trigger_conditions"]),
                    "proposal_origin": patch.get("source", {}).get("proposal_origin", "rules"),
                    "evidence_refs": list(patch.get("source", {}).get("evidence_refs", [])),
                }
            )
        if transitions:
            self.store.refresh_materialized_views()
        reflection = self._reflection(trace, outcome, root_cause)
        return {
            "status": "completed",
            "reflection": reflection,
            "outcome": outcome,
            "root_cause": root_cause,
            "generated_patches": generated,
            "patch_transitions": transitions,
            "decision_audit": {
                "diagnostic": diagnostic_audit,
                "patch_generation": patch_audit,
            },
        }

    def approve_patch(self, patch_id):
        patch = self.store.load_patch(patch_id)
        patch = self.safety_gate.approve(patch)
        self.store.refresh_materialized_views()
        return patch

    def _observe_existing(self, task_state, trace, outcome):
        context = dict(getattr(task_state, "evolution_context", {}) or {})
        eligible_ids = set(context.get("eligible_patch_ids", []))
        hit_ids = set(context.get("active_patch_ids", [])) | set(context.get("shadow_patch_ids", []))
        transitions = []
        for patch_id in sorted(eligible_ids):
            try:
                patch = self.store.load_patch(patch_id)
            except KeyError:
                continue
            before = patch["status"]
            hit = patch_id in hit_ids and self._trigger_matches(patch, trace)
            patch, transition = self.safety_gate.observe(
                patch,
                run_id=str(task_state.run_id),
                eligible=True,
                hit=hit,
                success=outcome["label"] == "correct",
                harmful=outcome["label"] == "harmful",
            )
            if transition:
                transitions.append(
                    {
                        "patch_id": patch_id,
                        "from": before,
                        "to": patch["status"],
                        "metrics": dict(patch["metrics"]),
                    }
                )
        return transitions

    @staticmethod
    def _trigger_matches(patch, trace):
        conditions = list(patch.get("trigger_conditions", []) or [])
        events = list(trace.get("tool_events", []) or [])
        trace_paths = {
            path
            for event in events
            for path in [event.get("path", ""), *event.get("affected_paths", [])]
            if path
        }
        for condition in conditions:
            signal = str(condition.get("signal", ""))
            expected = str(condition.get("equals", ""))
            if signal == "tool_name" and any(event.get("name") == expected for event in events):
                return True
            if signal in {"path", "stale_path"} and expected != REDACTED_PATH and expected in trace_paths:
                return True
            if signal == "route_id" and trace.get("scope", {}).get("route_id", "") == expected:
                return True
            if signal == "tool_failure" and any(
                expected in {event.get("error_code"), event.get("security_event_type"), event.get("status")}
                for event in events
            ):
                return True
            if signal == "task_scope":
                return True
        return False

    def _reflection(self, trace, outcome, root_cause):
        knowledge = self.generator.knowledge_facts(trace, outcome)
        mistakes = []
        for event in trace.get("tool_events", []):
            if event.get("status") == "ok":
                continue
            mistakes.append(
                {
                    "tool": event.get("name", ""),
                    "status": event.get("status", ""),
                    "error_code": event.get("error_code", ""),
                }
            )
        stale_paths = list(trace.get("stale_paths", []))
        return {
            "new_knowledge": {
                "answer": "yes" if knowledge else "no",
                "items": knowledge,
            },
            "mistakes": {
                "answer": "yes" if mistakes else "no",
                "items": mistakes,
                "root_cause_level": (root_cause or {}).get("level", ""),
            },
            "stale_knowledge": {
                "answer": "yes" if stale_paths else "not_observed",
                "items": stale_paths,
            },
        }

    def _selected_for_canary(self, run_id, patch_id):
        fraction = float(self.thresholds.canary_fraction)
        if fraction <= 0.0:
            return False
        if fraction >= 1.0:
            return True
        digest = hashlib.sha256(f"{run_id}:{patch_id}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        return value < fraction

    @staticmethod
    def _scope_matches(patch, task_state):
        scope = dict(patch.get("scope", {}) or {})
        skill_name = str(scope.get("skill_name", ""))
        route_id = str(scope.get("route_id", ""))
        if skill_name and skill_name != str(task_state.skill_name or ""):
            return False
        if route_id and route_id != str(task_state.route_id or ""):
            return False
        return True
