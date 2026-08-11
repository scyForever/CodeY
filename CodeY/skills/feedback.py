"""Auditable Skill routing decisions, outcomes, and Description review patches."""

from __future__ import annotations

import json
import tempfile
import threading
import uuid
from pathlib import Path

from ..context.workspace import now


ROUTING_EVENT_SCHEMA_VERSION = 1
DESCRIPTION_PATCH_SCHEMA_VERSION = 2
ROUTE_DECISION_FIELDS = {
    "routing_event_id",
    "skill_name",
    "route_id",
    "fallback",
    "matched_terms",
    "loaded_paths",
    "selection_source",
    "selector_confidence",
    "selector_reason",
    "candidates",
}
CANDIDATE_FIELDS = {
    "skill_name",
    "activation_phrases",
    "near_misses",
    "matched_activation_phrases",
    "matched_near_misses",
    "name_matched",
    "lexical_score",
    "selected",
}


class SkillFeedbackStore:
    def __init__(self, root):
        self.root = Path(root)
        self.events_dir = self.root / "events"
        self.patches_dir = self.root / "description-patches"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.patches_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def start(self, *, session_id, run_id, task_id, request, route):
        event_id = "route_" + uuid.uuid4().hex[:16]
        timestamp = now()
        decision = dict(route)
        _validate_decision(decision)
        event = {
            "schema_version": ROUTING_EVENT_SCHEMA_VERSION,
            "event_id": event_id,
            "session_id": str(session_id),
            "run_id": str(run_id),
            "task_id": str(task_id),
            "request": str(request),
            "decision": decision,
            "execution": None,
            "feedback": [],
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        with self._lock:
            self._write_json(self.event_path(event_id), event)
        return event_id

    def complete(self, event_id, *, status, stop_reason, outcome, tool_steps):
        with self._lock:
            event = self.load_event(event_id)
            event["execution"] = {
                "status": str(status),
                "stop_reason": str(stop_reason),
                "outcome": str(outcome),
                "tool_steps": int(tool_steps),
                "recorded_at": now(),
            }
            event["feedback"] = [
                feedback
                for feedback in event["feedback"]
                if feedback.get("source") != "runtime_execution"
            ]
            event["feedback"].append(
                {
                    "source": "runtime_execution",
                    "verdict": _execution_verdict(outcome),
                    "note": "Task execution outcome; this is not proof that the Skill route was correct.",
                    "created_at": now(),
                }
            )
            event["updated_at"] = now()
            self._write_json(self.event_path(event_id), event)
            return event

    def submit_user_feedback(self, event_id, *, correct, expected_skill_name="", note=""):
        if not isinstance(correct, bool):
            raise TypeError("correct must be a boolean")
        with self._lock:
            event = self.load_event(event_id)
            event["feedback"].append(
                {
                    "source": "user_route_feedback",
                    "verdict": "positive" if correct else "negative",
                    "expected_skill_name": str(expected_skill_name).strip(),
                    "note": str(note).strip(),
                    "created_at": now(),
                }
            )
            event["updated_at"] = now()
            self._write_json(self.event_path(event_id), event)
            return event

    def propose_description_patch(self, skill_name, min_samples=3):
        skill_name = str(skill_name).strip()
        if not skill_name:
            raise ValueError("skill_name is required")
        evidence = []
        positives = 0
        false_activations = 0
        missed_activations = 0
        for event in self.list_events():
            explicit = [
                feedback
                for feedback in event["feedback"]
                if feedback.get("source") == "user_route_feedback"
            ]
            if not explicit:
                continue
            feedback = explicit[-1]
            selected = str(event["decision"].get("skill_name", ""))
            expected = str(feedback.get("expected_skill_name", ""))
            if selected == skill_name and feedback["verdict"] == "positive":
                evidence_kind = "correct_activation"
                positives += 1
            elif selected == skill_name:
                evidence_kind = "false_activation"
                false_activations += 1
            elif expected == skill_name:
                evidence_kind = "missed_activation"
                missed_activations += 1
            else:
                continue
            evidence.append(
                {
                    "event_id": event["event_id"],
                    "request": event["request"],
                    "kind": evidence_kind,
                    "selected_skill_name": selected,
                    "expected_skill_name": expected,
                    "note": feedback.get("note", ""),
                    "candidates": list(event["decision"].get("candidates", [])),
                }
            )
        total = positives + false_activations + missed_activations
        if total < max(1, int(min_samples)):
            raise ValueError("not enough explicit routing feedback to propose a Description patch")
        patch_id = "skill_description_" + uuid.uuid4().hex[:16]
        patch = {
            "schema_version": DESCRIPTION_PATCH_SCHEMA_VERSION,
            "patch_id": patch_id,
            "type": "skill_description",
            "skill_name": skill_name,
            "status": "review_required",
            "metrics": {
                "explicit_feedback_count": total,
                "positive_count": positives,
                "false_activation_count": false_activations,
                "missed_activation_count": missed_activations,
                "precision_proxy": positives / (positives + false_activations)
                if positives + false_activations
                else 0.0,
                "recall_proxy": positives / (positives + missed_activations)
                if positives + missed_activations
                else 0.0,
            },
            "evidence": evidence,
            "proposed_changes": {
                "activation_examples": [
                    item["request"]
                    for item in evidence
                    if item["kind"] == "missed_activation"
                ],
                "near_miss_examples": [
                    item["request"]
                    for item in evidence
                    if item["kind"] == "false_activation"
                ],
                "constraints": {
                    "activation_phrase_count": 2,
                    "near_miss_count": 3,
                    "english_only": True,
                    "auto_apply": False,
                },
            },
            "instruction": (
                "Review false activations and misses, then revise the two exclusive activation phrases "
                "or the three near-miss exclusions. Do not apply automatically."
            ),
            "created_at": now(),
        }
        with self._lock:
            self._write_json(self.patches_dir / f"{patch_id}.json", patch)
        return patch

    def event_path(self, event_id):
        event_id = str(event_id).strip()
        if not re_safe_id(event_id):
            raise ValueError("invalid routing event id")
        return self.events_dir / f"{event_id}.json"

    def load_event(self, event_id):
        event = json.loads(self.event_path(event_id).read_text(encoding="utf-8"))
        _validate_event(event)
        return event

    def list_events(self):
        events = []
        for path in sorted(self.events_dir.glob("route_*.json")):
            event = json.loads(path.read_text(encoding="utf-8"))
            _validate_event(event)
            events.append(event)
        return events

    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            temp_name = handle.name
        Path(temp_name).replace(path)


def _execution_verdict(outcome):
    if outcome == "correct":
        return "positive"
    if outcome in {"incorrect", "harmful"}:
        return "negative"
    return "mixed"


def re_safe_id(value):
    return bool(value) and all(char.isalnum() or char in {"_", "-"} for char in value)


def _validate_event(event):
    required = {
        "schema_version",
        "event_id",
        "session_id",
        "run_id",
        "task_id",
        "request",
        "decision",
        "execution",
        "feedback",
        "created_at",
        "updated_at",
    }
    if not isinstance(event, dict) or event.get("schema_version") != ROUTING_EVENT_SCHEMA_VERSION:
        raise ValueError("routing event schema is invalid")
    if set(event) != required:
        raise ValueError("routing event has unknown or missing fields")
    _validate_decision(event["decision"])
    if not isinstance(event["feedback"], list):
        raise ValueError("routing event decision/feedback is invalid")
    execution = event["execution"]
    if execution is not None:
        if not isinstance(execution, dict) or set(execution) != {
            "status",
            "stop_reason",
            "outcome",
            "tool_steps",
            "recorded_at",
        }:
            raise ValueError("routing event execution is invalid")
        if isinstance(execution["tool_steps"], bool) or not isinstance(execution["tool_steps"], int):
            raise ValueError("routing event tool_steps is invalid")
    for feedback in event["feedback"]:
        if not isinstance(feedback, dict):
            raise ValueError("routing feedback entry is invalid")
        source = feedback.get("source")
        expected_fields = {"source", "verdict", "note", "created_at"}
        if source == "user_route_feedback":
            expected_fields.add("expected_skill_name")
        if set(feedback) != expected_fields or source not in {
            "runtime_execution",
            "user_route_feedback",
        }:
            raise ValueError("routing feedback entry schema is invalid")
        if feedback.get("verdict") not in {"positive", "negative", "mixed"}:
            raise ValueError("routing feedback verdict is invalid")
    return event


def _validate_decision(decision):
    if not isinstance(decision, dict) or set(decision) != ROUTE_DECISION_FIELDS:
        raise ValueError("routing decision schema is invalid")
    if not isinstance(decision["fallback"], bool):
        raise ValueError("routing decision fallback is invalid")
    if not isinstance(decision["selector_confidence"], (int, float)) or isinstance(
        decision["selector_confidence"], bool
    ):
        raise ValueError("routing decision confidence is invalid")
    if not 0.0 <= float(decision["selector_confidence"]) <= 1.0:
        raise ValueError("routing decision confidence is invalid")
    for key in ("matched_terms", "loaded_paths", "candidates"):
        if not isinstance(decision[key], list):
            raise ValueError(f"routing decision {key} is invalid")
    for candidate in decision["candidates"]:
        if not isinstance(candidate, dict) or set(candidate) != CANDIDATE_FIELDS:
            raise ValueError("routing candidate schema is invalid")
        for key in (
            "activation_phrases",
            "near_misses",
            "matched_activation_phrases",
            "matched_near_misses",
        ):
            if not isinstance(candidate[key], list):
                raise ValueError(f"routing candidate {key} is invalid")
        if not isinstance(candidate["name_matched"], bool) or not isinstance(candidate["selected"], bool):
            raise ValueError("routing candidate boolean evidence is invalid")
        if isinstance(candidate["lexical_score"], bool) or not isinstance(candidate["lexical_score"], int):
            raise ValueError("routing candidate lexical_score is invalid")
    return decision
