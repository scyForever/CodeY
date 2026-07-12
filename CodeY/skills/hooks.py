"""In-process lifecycle hooks for session context restoration."""

from __future__ import annotations

from dataclasses import dataclass

SESSION_START_REASONS = {"startup", "resume", "reset", "compact"}


@dataclass(frozen=True)
class SessionStartEvent:
    reason: str
    session_id: str
    generation: int


class HookManager:
    def __init__(self, callbacks=None):
        self.callbacks = tuple(callbacks or ())

    def session_start(self, agent, reason):
        if reason not in SESSION_START_REASONS:
            raise ValueError(f"unknown SessionStart reason: {reason}")
        context = agent.skill_router.session_start()
        generation = int(agent.session.get("session_context", {}).get("generation", 0)) + 1
        payload = {
            "reason": reason,
            "generation": generation,
            "skill_fingerprint": context.fingerprint,
            "skill_names": list(context.skill_names),
            "loaded_paths": list(context.loaded_paths),
            "text": context.text,
        }
        agent.session["session_context"] = payload
        event = SessionStartEvent(reason=reason, session_id=agent.session["id"], generation=generation)
        for callback in self.callbacks:
            callback(event, dict(payload))
        return payload
