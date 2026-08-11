"""Versioned transcript state and asynchronous conversation summarization."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass

from ..context.workspace import now
from ..providers.clients import ModelCompletion


TRANSCRIPT_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 1
SUMMARY_PROMPT_VERSION = "conversation-summary-v1"
DEFAULT_RECENT_TURNS = 6
DEFAULT_SUMMARY_MAX_CHARS = 4000


def new_transcript_state():
    return {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "next_sequence": 1,
        "entries": [],
    }


def new_summary_state():
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "committed": {
            "generation": 0,
            "text": "",
            "covered_through_sequence": 0,
            "source_hash": "",
            "model": "",
            "prompt_version": SUMMARY_PROMPT_VERSION,
            "created_at": "",
        },
        "pending": None,
        "last_error": None,
    }


def validate_transcript_state(state):
    if not isinstance(state, dict) or state.get("schema_version") != TRANSCRIPT_SCHEMA_VERSION:
        raise ValueError(f"transcript schema_version must be {TRANSCRIPT_SCHEMA_VERSION}")
    if set(state) != {"schema_version", "next_sequence", "entries"}:
        raise ValueError("transcript state has unknown or missing fields")
    entries = state["entries"]
    next_sequence = state["next_sequence"]
    if not isinstance(entries, list) or not isinstance(next_sequence, int) or next_sequence < 1:
        raise ValueError("transcript entries/next_sequence are invalid")
    expected = 1
    message_ids = set()
    for entry in entries:
        _validate_entry(entry, expected)
        message_id = entry["message_id"]
        if message_id in message_ids:
            raise ValueError("transcript message_id values must be unique")
        message_ids.add(message_id)
        expected += 1
    if next_sequence != expected:
        raise ValueError("transcript next_sequence is not contiguous")
    return state


def validate_summary_state(state):
    if not isinstance(state, dict) or state.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError(f"conversation summary schema_version must be {SUMMARY_SCHEMA_VERSION}")
    if set(state) != {"schema_version", "committed", "pending", "last_error"}:
        raise ValueError("conversation summary has unknown or missing fields")
    committed = state["committed"]
    required = {
        "generation",
        "text",
        "covered_through_sequence",
        "source_hash",
        "model",
        "prompt_version",
        "created_at",
    }
    if not isinstance(committed, dict) or set(committed) != required:
        raise ValueError("conversation summary committed state is invalid")
    if not isinstance(committed["generation"], int) or committed["generation"] < 0:
        raise ValueError("conversation summary generation is invalid")
    if not isinstance(committed["covered_through_sequence"], int) or committed["covered_through_sequence"] < 0:
        raise ValueError("conversation summary coverage is invalid")
    for key in ("text", "source_hash", "model", "prompt_version", "created_at"):
        if not isinstance(committed[key], str):
            raise ValueError(f"conversation summary {key} must be a string")
    pending = state["pending"]
    if pending is not None:
        pending_required = {
            "generation",
            "covered_through_sequence",
            "source_hash",
            "started_at",
        }
        if not isinstance(pending, dict) or set(pending) != pending_required:
            raise ValueError("conversation summary pending state is invalid")
        if not isinstance(pending["generation"], int) or pending["generation"] <= committed["generation"]:
            raise ValueError("pending summary generation must advance committed generation")
        if (
            not isinstance(pending["covered_through_sequence"], int)
            or pending["covered_through_sequence"] < 0
        ):
            raise ValueError("pending summary coverage is invalid")
        if not isinstance(pending["source_hash"], str) or not isinstance(pending["started_at"], str):
            raise ValueError("pending summary metadata is invalid")
    if state["last_error"] is not None and not isinstance(state["last_error"], dict):
        raise ValueError("conversation summary last_error must be an object or null")
    return state


def validate_summary_against_transcript(summary, entries):
    """Validate summary coverage and hashes against the current transcript."""
    validate_summary_state(summary)
    max_sequence = entries[-1]["sequence"] if entries else 0
    committed = summary["committed"]
    if committed["covered_through_sequence"] > max_sequence:
        raise ValueError("conversation summary committed coverage exceeds transcript")
    if committed["covered_through_sequence"]:
        expected_hash = transcript_prefix_hash(entries, committed["covered_through_sequence"])
        if committed["source_hash"] != expected_hash:
            raise ValueError("conversation summary committed source hash does not match transcript")
    pending = summary.get("pending")
    if pending is not None:
        if pending["covered_through_sequence"] <= committed["covered_through_sequence"]:
            raise ValueError("pending summary coverage must advance committed coverage")
        if pending["covered_through_sequence"] > max_sequence:
            raise ValueError("conversation summary pending coverage exceeds transcript")
        expected_hash = transcript_prefix_hash(entries, pending["covered_through_sequence"])
        if pending["source_hash"] != expected_hash:
            raise ValueError("conversation summary pending source hash does not match transcript")
    return summary


def append_transcript_entry(state, item, turn_id):
    validate_transcript_state(state)
    role = str(item.get("role", "")).strip()
    if role not in {"user", "assistant", "tool"}:
        raise ValueError("transcript role must be user, assistant, or tool")
    turn_id = str(turn_id or "").strip()
    if not turn_id:
        raise ValueError("transcript entries require a turn_id")
    entry = {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "sequence": state["next_sequence"],
        "message_id": "msg_" + uuid.uuid4().hex,
        "turn_id": turn_id,
        "role": role,
        "created_at": str(item.get("created_at", "") or now()),
        "content": str(item.get("content", "")),
    }
    if role == "tool":
        name = str(item.get("name", "")).strip()
        args = item.get("args")
        if not name or not isinstance(args, dict):
            raise ValueError("tool transcript entries require name and args")
        entry["name"] = name
        entry["args"] = dict(args)
    state["entries"].append(entry)
    state["next_sequence"] += 1
    return entry


def completed_turn_ids(entries, current_turn_id=""):
    current_turn_id = str(current_turn_id or "")
    completed = {
        entry["turn_id"]
        for entry in entries
        if entry["role"] == "assistant"
    }
    ordered = []
    seen = set()
    for entry in entries:
        turn_id = entry["turn_id"]
        if turn_id == current_turn_id or turn_id in seen or turn_id not in completed:
            continue
        seen.add(turn_id)
        ordered.append(turn_id)
    return ordered


def split_for_context(entries, recent_turns=DEFAULT_RECENT_TURNS, current_turn_id=""):
    recent_turns = max(1, int(recent_turns))
    turn_ids = completed_turn_ids(entries, current_turn_id=current_turn_id)
    completed_ids = set(turn_ids)
    recent_ids = set(turn_ids[-recent_turns:])
    older = []
    recent = []
    current = []
    for entry in entries:
        if entry["turn_id"] == current_turn_id:
            current.append(entry)
        elif entry["turn_id"] in recent_ids or entry["turn_id"] not in completed_ids:
            recent.append(entry)
        else:
            older.append(entry)
    return older, recent, current


def transcript_prefix_hash(entries, covered_through_sequence):
    prefix = [entry for entry in entries if entry["sequence"] <= int(covered_through_sequence)]
    payload = json.dumps(prefix, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_transcript_entries(entries):
    lines = []
    for entry in entries:
        if entry["role"] == "tool":
            args = json.dumps(entry["args"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            lines.append(f"[tool:{entry['name']}] {args}")
            lines.append(entry["content"])
        else:
            lines.append(f"[{entry['role']}] {entry['content']}")
    return "\n".join(lines)


def _validate_entry(entry, expected_sequence):
    common = {
        "schema_version",
        "sequence",
        "message_id",
        "turn_id",
        "role",
        "created_at",
        "content",
    }
    if not isinstance(entry, dict) or entry.get("schema_version") != TRANSCRIPT_SCHEMA_VERSION:
        raise ValueError("transcript entry schema is invalid")
    role = entry.get("role")
    expected_fields = common | ({"name", "args"} if role == "tool" else set())
    if set(entry) != expected_fields:
        raise ValueError("transcript entry has unknown or missing fields")
    if entry["sequence"] != expected_sequence:
        raise ValueError("transcript sequences must be contiguous")
    if role not in {"user", "assistant", "tool"}:
        raise ValueError("transcript role is invalid")
    for key in ("message_id", "turn_id", "created_at", "content"):
        if not isinstance(entry[key], str) or (key != "content" and not entry[key]):
            raise ValueError(f"transcript {key} is invalid")
    if role == "tool" and (not isinstance(entry["name"], str) or not entry["name"] or not isinstance(entry["args"], dict)):
        raise ValueError("tool transcript entry is invalid")


@dataclass(frozen=True)
class SummaryJob:
    epoch: int
    generation: int
    covered_through_sequence: int
    source_hash: str
    previous_summary: str
    new_entries: tuple[dict, ...]


class AsyncConversationSummarizer:
    """Refresh a committed summary without blocking prompt construction."""

    def __init__(self, agent, recent_turns=DEFAULT_RECENT_TURNS, max_chars=DEFAULT_SUMMARY_MAX_CHARS):
        self.agent = agent
        self.recent_turns = max(1, int(recent_turns))
        self.max_chars = max(200, int(max_chars))
        self._threads = {}
        self._threads_lock = threading.Lock()
        self._epoch = 0

    def invalidate(self):
        """Invalidate in-flight jobs without waiting for provider I/O."""
        with self._threads_lock:
            self._epoch += 1
            return self._epoch

    def recover_persisted_pending(self):
        """Clear a persisted job that has no worker in the resumed process."""
        self.invalidate()
        with self.agent._session_lock:
            state = self.agent.session["conversation_summary"]
            pending = state.get("pending")
            if pending is None:
                return False
            previous_pending = dict(pending)
            previous_error = dict(state["last_error"]) if state.get("last_error") else None
            state["pending"] = None
            state["last_error"] = {
                "generation": int(pending["generation"]),
                "error_type": "SummaryInterruptedOnResume",
                "created_at": now(),
            }
            try:
                self.agent.save_session()
            except Exception:
                state["pending"] = previous_pending
                state["last_error"] = previous_error
                raise
            return True

    def schedule(self):
        with self.agent._session_lock:
            entries = list(self.agent.session["transcript"]["entries"])
            older, _, _ = split_for_context(
                entries,
                self.recent_turns,
                current_turn_id=getattr(self.agent, "_active_turn_id", ""),
            )
            if not older:
                return None
            cutoff = older[-1]["sequence"]
            summary_state = self.agent.session["conversation_summary"]
            committed = summary_state["committed"]
            if cutoff <= committed["covered_through_sequence"]:
                return None
            generation = max(
                committed["generation"],
                int((summary_state.get("pending") or {}).get("generation", 0)),
                int((summary_state.get("last_error") or {}).get("generation", 0)),
            ) + 1
            source_hash = transcript_prefix_hash(entries, cutoff)
            new_entries = tuple(
                dict(entry)
                for entry in entries
                if committed["covered_through_sequence"] < entry["sequence"] <= cutoff
            )
            job = SummaryJob(
                epoch=self._epoch,
                generation=generation,
                covered_through_sequence=cutoff,
                source_hash=source_hash,
                previous_summary=committed["text"],
                new_entries=new_entries,
            )
            summary_state["pending"] = {
                "generation": generation,
                "covered_through_sequence": cutoff,
                "source_hash": source_hash,
                "started_at": now(),
            }
            summary_state["last_error"] = None
            self.agent.save_session()

        thread = threading.Thread(
            target=self._run,
            args=(job,),
            name=f"codey-summary-{generation}",
            daemon=True,
        )
        with self._threads_lock:
            self._threads[(job.epoch, job.generation)] = thread
        thread.start()
        return generation

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while True:
            with self._threads_lock:
                threads = list(self._threads.values())
            alive = [thread for thread in threads if thread.is_alive()]
            if not alive:
                return True
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining == 0.0:
                return False
            alive[0].join(remaining)

    def _run(self, job):
        try:
            prompt = self._build_prompt(job)
            completion = self.agent.complete_summary_model(prompt)
            if not isinstance(completion, ModelCompletion):
                raise TypeError("summary model client must return ModelCompletion")
            text = _parse_summary(completion.text, self.max_chars)
            self._commit(job, text)
        except Exception as exc:
            self._fail(job, exc)
        finally:
            with self._threads_lock:
                self._threads.pop((job.epoch, job.generation), None)

    def _build_prompt(self, job):
        previous = self.agent.redact_text(job.previous_summary)
        delta = self.agent.redact_text(render_transcript_entries(job.new_entries))
        return (
            "Update the durable conversation summary for a coding agent. Preserve active goals, "
            "constraints, decisions, unresolved failures, and repository facts. Remove obsolete or "
            "superseded details. Do not invent facts. Return exactly one JSON object with the single "
            "key summary.\n\n"
            f"Previous summary:\n{previous or '- none'}\n\n"
            f"New transcript:\n{delta}\n"
        )

    def _commit(self, job, text):
        with self.agent._session_lock:
            state = self.agent.session["conversation_summary"]
            pending = state["pending"]
            if (
                job.epoch != self._epoch
                or not pending
                or pending["generation"] != job.generation
                or pending["covered_through_sequence"] != job.covered_through_sequence
                or pending["source_hash"] != job.source_hash
            ):
                return False
            entries = self.agent.session["transcript"]["entries"]
            if transcript_prefix_hash(entries, job.covered_through_sequence) != job.source_hash:
                previous_pending = dict(pending)
                previous_error = dict(state["last_error"]) if state.get("last_error") else None
                state["pending"] = None
                state["last_error"] = {
                    "generation": job.generation,
                    "error_type": "TranscriptChanged",
                    "created_at": now(),
                }
                try:
                    self.agent.save_session()
                except Exception:
                    state["pending"] = previous_pending
                    state["last_error"] = previous_error
                return False
            previous_committed = dict(state["committed"])
            previous_pending = dict(pending)
            previous_error = dict(state["last_error"]) if state.get("last_error") else None
            state["committed"] = {
                "generation": job.generation,
                "text": text,
                "covered_through_sequence": job.covered_through_sequence,
                "source_hash": job.source_hash,
                "model": self.agent.summary_model_name(),
                "prompt_version": SUMMARY_PROMPT_VERSION,
                "created_at": now(),
            }
            state["pending"] = None
            state["last_error"] = None
            try:
                self.agent.save_session()
            except Exception:
                state["committed"] = previous_committed
                state["pending"] = previous_pending
                state["last_error"] = previous_error
                raise
            return True

    def _fail(self, job, exc):
        with self.agent._session_lock:
            state = self.agent.session["conversation_summary"]
            pending = state["pending"]
            if (
                job.epoch != self._epoch
                or not pending
                or pending["generation"] != job.generation
                or pending["covered_through_sequence"] != job.covered_through_sequence
                or pending["source_hash"] != job.source_hash
            ):
                return
            previous_pending = dict(pending)
            previous_error = dict(state["last_error"]) if state.get("last_error") else None
            state["pending"] = None
            state["last_error"] = {
                "generation": job.generation,
                "error_type": type(exc).__name__,
                "created_at": now(),
            }
            try:
                self.agent.save_session()
            except Exception:
                # CAS conflicts are recorded by the owner process; the daemon must not crash.
                state["pending"] = previous_pending
                state["last_error"] = previous_error

    def cancel_pending(self, error_type="SummaryCancelled"):
        """Clear a pending job after a bounded shutdown wait."""
        self.invalidate()
        with self.agent._session_lock:
            state = self.agent.session["conversation_summary"]
            pending = state.get("pending")
            if pending is None:
                return False
            previous_pending = dict(pending)
            previous_error = dict(state["last_error"]) if state.get("last_error") else None
            state["pending"] = None
            state["last_error"] = {
                "generation": int(pending["generation"]),
                "error_type": str(error_type),
                "created_at": now(),
            }
            try:
                self.agent.save_session()
            except Exception:
                state["pending"] = previous_pending
                state["last_error"] = previous_error
                return False
            return True


def _parse_summary(raw, max_chars):
    try:
        payload = json.loads(str(raw).strip())
    except json.JSONDecodeError as exc:
        raise ValueError("summary model output must be strict JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"summary"}:
        raise ValueError("summary model output must contain only summary")
    text = payload["summary"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("summary must be a non-empty string")
    text = text.strip()
    if len(text) > max_chars:
        raise ValueError("summary exceeds configured character limit")
    return text
