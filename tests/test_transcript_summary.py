import threading

import pytest

from CodeY import CodeYAgent, FakeModelClient, ModelCompletion, SessionStore, WorkspaceContext
from CodeY.context.transcript import (
    append_transcript_entry,
    new_transcript_state,
    split_for_context,
    transcript_prefix_hash,
)


class ControlledSummaryClient:
    supports_prompt_cache = False
    model = "controlled-summary"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.started = [threading.Event() for _ in self.outputs]
        self.release = [threading.Event() for _ in self.outputs]
        self._lock = threading.Lock()
        self._index = 0

    def complete(self, prompt, max_new_tokens, **kwargs):
        del prompt, max_new_tokens, kwargs
        with self._lock:
            index = self._index
            self._index += 1
        self.started[index].set()
        assert self.release[index].wait(timeout=5)
        return ModelCompletion(text=self.outputs[index], metadata={})


def build_agent(tmp_path, summary_client, recent_turns=1):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return CodeYAgent(
        model_client=FakeModelClient([]),
        summary_model_client=summary_client,
        summary_recent_turns=recent_turns,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".codey" / "sessions"),
        approval_policy="auto",
        skill_mode="off",
        feature_flags={"self_evolution": False},
    )


def record_turn(agent, turn_id, user_text, assistant_text):
    agent.record_transcript(
        {"role": "user", "content": user_text, "created_at": "2026-08-11T10:00:00+00:00"},
        turn_id=turn_id,
    )
    agent.record_transcript(
        {
            "role": "assistant",
            "content": assistant_text,
            "created_at": "2026-08-11T10:00:01+00:00",
        },
        turn_id=turn_id,
    )


def test_split_for_context_counts_completed_turns_and_preserves_incomplete_entries():
    state = new_transcript_state()
    append_transcript_entry(state, {"role": "user", "content": "u1"}, "turn-1")
    append_transcript_entry(state, {"role": "assistant", "content": "a1"}, "turn-1")
    append_transcript_entry(state, {"role": "user", "content": "incomplete"}, "turn-2")
    append_transcript_entry(state, {"role": "user", "content": "u3"}, "turn-3")
    append_transcript_entry(state, {"role": "assistant", "content": "a3"}, "turn-3")
    append_transcript_entry(state, {"role": "user", "content": "current"}, "turn-4")

    older, recent, current = split_for_context(
        state["entries"],
        recent_turns=1,
        current_turn_id="turn-4",
    )

    assert {entry["turn_id"] for entry in older} == {"turn-1"}
    assert {entry["turn_id"] for entry in recent} == {"turn-2", "turn-3"}
    assert {entry["turn_id"] for entry in current} == {"turn-4"}


def test_manual_transcript_entry_requires_explicit_turn_id(tmp_path):
    agent = build_agent(tmp_path, FakeModelClient([]))

    with pytest.raises(ValueError, match="explicit turn_id"):
        agent.record_transcript({"role": "user", "content": "orphan"})


def test_async_summary_commits_older_turns_and_keeps_recent_original(tmp_path):
    agent = build_agent(
        tmp_path,
        FakeModelClient(['{"summary":"durable constraints"}']),
        recent_turns=1,
    )
    record_turn(agent, "turn-1", "old user", "old answer")
    record_turn(agent, "turn-2", "middle user", "middle answer")
    record_turn(agent, "turn-3", "recent user", "recent answer")

    assert agent.refresh_conversation_summary_async() == 1
    assert agent.wait_for_conversation_summary(timeout=5)

    state = agent.conversation_summary_state()
    assert state["committed"]["text"] == "durable constraints"
    assert state["committed"]["covered_through_sequence"] == 4
    assert state["pending"] is None
    prompt = agent.prompt("new request")
    assert "durable constraints" in prompt
    assert "old user" not in prompt
    assert "middle answer" not in prompt
    assert "recent user" in prompt
    assert "recent answer" in prompt


def test_summary_failure_keeps_unsummarized_transcript_as_fallback(tmp_path):
    agent = build_agent(tmp_path, FakeModelClient(["not-json"]), recent_turns=1)
    record_turn(agent, "turn-1", "fallback user", "fallback answer")
    record_turn(agent, "turn-2", "recent user", "recent answer")

    agent.refresh_conversation_summary_async()
    assert agent.wait_for_conversation_summary(timeout=5)

    state = agent.conversation_summary_state()
    assert state["committed"]["generation"] == 0
    assert state["pending"] is None
    assert state["last_error"]["error_type"] == "ValueError"
    prompt, metadata = agent._build_prompt_and_metadata("new request")
    assert "fallback user" in prompt
    assert metadata["transcript"]["pending_fallback_entries"] == 2


def test_reset_invalidates_old_generation_without_clearing_new_pending_job(tmp_path):
    client = ControlledSummaryClient(
        ['{"summary":"old summary"}', '{"summary":"new summary"}']
    )
    agent = build_agent(tmp_path, client, recent_turns=1)
    record_turn(agent, "old-1", "old one", "old one answer")
    record_turn(agent, "old-2", "old two", "old two answer")
    agent.refresh_conversation_summary_async()
    assert client.started[0].wait(timeout=5)

    agent.reset()
    record_turn(agent, "new-1", "new one", "new one answer")
    record_turn(agent, "new-2", "new two", "new two answer")
    agent.refresh_conversation_summary_async()
    assert client.started[1].wait(timeout=5)

    client.release[0].set()
    client.release[1].set()
    assert agent.wait_for_conversation_summary(timeout=5)
    assert agent.conversation_summary_state()["committed"]["text"] == "new summary"
    assert agent.conversation_summary_state()["last_error"] is None


def test_close_cancels_pending_summary_after_bounded_wait(tmp_path):
    client = ControlledSummaryClient(['{"summary":"too late"}'])
    agent = build_agent(tmp_path, client, recent_turns=1)
    record_turn(agent, "turn-1", "one", "one answer")
    record_turn(agent, "turn-2", "two", "two answer")
    agent.refresh_conversation_summary_async()
    assert client.started[0].wait(timeout=5)

    assert agent.close(timeout=0) is False
    state = agent.conversation_summary_state()
    assert state["pending"] is None
    assert state["last_error"]["error_type"] == "SummaryFlushTimeout"

    client.release[0].set()
    assert agent.wait_for_conversation_summary(timeout=5)
    assert agent.conversation_summary_state()["committed"]["generation"] == 0


def test_resume_clears_persisted_summary_job_without_a_worker(tmp_path):
    agent = build_agent(tmp_path, FakeModelClient([]), recent_turns=1)
    record_turn(agent, "turn-1", "old user", "old answer")
    record_turn(agent, "turn-2", "recent user", "recent answer")
    entries = agent.transcript_entries()
    agent.session["conversation_summary"]["pending"] = {
        "generation": 1,
        "covered_through_sequence": 2,
        "source_hash": transcript_prefix_hash(entries, 2),
        "started_at": "2026-08-11T10:00:02+00:00",
    }
    agent.save_session()

    resumed = CodeYAgent.from_session(
        model_client=FakeModelClient([]),
        summary_model_client=FakeModelClient([]),
        summary_recent_turns=1,
        workspace=agent.workspace,
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
        skill_mode="off",
        feature_flags={"self_evolution": False},
    )

    state = resumed.conversation_summary_state()
    assert state["pending"] is None
    assert state["last_error"]["generation"] == 1
    assert state["last_error"]["error_type"] == "SummaryInterruptedOnResume"
    persisted = resumed.session_store.load(resumed.session["id"])
    assert persisted["conversation_summary"] == state
