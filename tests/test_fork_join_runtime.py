import json
import threading
from pathlib import Path

import pytest

from CodeY import CodeYAgent, FakeModelClient, ModelCompletion, SessionStore, WorkspaceContext
from CodeY.core.task_state import TaskState


class BarrierFinalClient:
    supports_prompt_cache = False

    def __init__(self, branch_id, barrier):
        self.branch_id = branch_id
        self.barrier = barrier

    def complete(self, prompt, max_new_tokens, **kwargs):
        del prompt, max_new_tokens, kwargs
        self.barrier.wait(timeout=5)
        return ModelCompletion(
            text=f"<final>result-{self.branch_id}</final>",
            metadata={"branch_marker": self.branch_id},
        )


class FailingClient:
    supports_prompt_cache = False

    def complete(self, prompt, max_new_tokens, **kwargs):
        del prompt, max_new_tokens, kwargs
        raise RuntimeError("branch failed")


def fork_call(tasks):
    return "<tool>" + json.dumps(
        {
            "name": "fork_join",
            "args": {
                "tasks": tasks,
                "max_steps": 2,
                "join_policy": "all_settled",
            },
        }
    ) + "</tool>"


def build_agent(tmp_path, outputs, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return CodeYAgent(
        model_client=FakeModelClient(outputs),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".codey" / "sessions"),
        approval_policy="auto",
        skill_mode="off",
        feature_flags={"self_evolution": False},
        **kwargs,
    )


def trace_events(agent):
    path = agent.run_store.trace_path(agent.current_task_state.run_id)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_agent_loop_runs_as_state_graph_and_preserves_terminal_artifacts(tmp_path):
    agent = build_agent(tmp_path, ["<final>done</final>"])

    assert agent.ask("finish") == "done"

    state = agent.current_task_state
    report = agent.run_store.load_report(state.run_id)
    events = trace_events(agent)
    assert state.status == "completed"
    assert state.phase == "finalize"
    assert state.graph_thread_id
    assert report["task_state"]["graph_thread_id"] == state.graph_thread_id
    assert report["fork_summary"] == {}
    assert any(event["event"] == "graph_routed" for event in events)
    assert events[-1]["event"] == "run_finished"


def test_fork_join_runs_homogeneous_children_in_parallel_and_joins_stably(tmp_path):
    tasks = [
        {"id": "api", "objective": "inspect the API"},
        {"id": "tests", "objective": "inspect tests"},
    ]
    barrier = threading.Barrier(2)

    def model_client_factory(spec):
        return BarrierFinalClient(spec.branch_id, barrier)

    agent = build_agent(
        tmp_path,
        [fork_call(tasks), "<final>joined</final>"],
        model_client_factory=model_client_factory,
    )

    assert agent.ask("inspect in parallel") == "joined"

    state = agent.current_task_state
    events = trace_events(agent)
    tool_record = next(
        item
        for item in agent.transcript_entries()
        if item.get("role") == "tool" and item.get("name") == "fork_join"
    )
    joined = json.loads(tool_record["content"])
    assert [branch["branch_id"] for branch in joined["branches"]] == ["api", "tests"]
    assert [branch["final_answer"] for branch in joined["branches"]] == ["result-api", "result-tests"]
    assert joined["status"] == "completed"
    assert state.fork_count == 1
    assert state.fork_summary["succeeded"] == 2
    assert agent.session["forks"]["items"][joined["fork_id"]]["status"] == "completed"
    assert sum(event["event"] == "branch_finished" for event in events) == 2
    assert any(event["event"] == "join_completed" for event in events)
    assert events[-1]["event"] == "run_finished"

    for branch in joined["branches"]:
        result_path = Path(agent.current_run_dir) / branch["result_path"]
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        assert saved["thread_id"].endswith("/" + joined["fork_id"] + "/" + branch["branch_id"])
        assert saved["run_id"] != state.run_id


def test_fork_join_isolates_branch_failure_and_keeps_successful_result(tmp_path):
    tasks = [
        {"id": "good", "objective": "inspect the API"},
        {"id": "bad", "objective": "inspect tests"},
    ]

    def model_client_factory(spec):
        if spec.branch_id == "bad":
            return FailingClient()
        return FakeModelClient(["<final>usable evidence</final>"])

    agent = build_agent(
        tmp_path,
        [fork_call(tasks), "<final>partial evidence handled</final>"],
        model_client_factory=model_client_factory,
    )

    assert agent.ask("inspect with failure isolation") == "partial evidence handled"

    tool_record = next(
        item
        for item in agent.transcript_entries()
        if item.get("role") == "tool" and item.get("name") == "fork_join"
    )
    joined = json.loads(tool_record["content"])
    by_id = {branch["branch_id"]: branch for branch in joined["branches"]}
    assert joined["status"] == "partial"
    assert joined["succeeded"] == 1
    assert joined["failed"] == 1
    assert by_id["good"]["status"] == "completed"
    assert by_id["bad"]["status"] == "failed"
    assert any(event["event"] == "branch_failed" for event in trace_events(agent))
    assert agent.current_task_state.status == "completed"


def test_fork_join_validation_and_legacy_task_state_rejection(tmp_path):
    agent = build_agent(tmp_path, ["<final>unused</final>"])

    result = agent.run_tool(
        "fork_join",
        {"tasks": [{"id": "only", "objective": "one task"}]},
    )

    assert "requires at least two tasks" in result
    with pytest.raises(ValueError, match="legacy task states are not supported"):
        TaskState.from_dict(
            {
                "run_id": "run-old",
                "task_id": "task-old",
                "user_request": "legacy",
            }
        )


def test_four_long_branch_results_remain_valid_bounded_json(tmp_path):
    tasks = [
        {"id": f"branch-{index}", "objective": "inspect " + ("o" * 200)}
        for index in range(4)
    ]

    def model_client_factory(spec):
        return FakeModelClient(["<final>" + (spec.branch_id + "-") * 300 + "</final>"])

    agent = build_agent(
        tmp_path,
        [fork_call(tasks), "<final>joined</final>"],
        model_client_factory=model_client_factory,
    )

    assert agent.ask("long parallel results") == "joined"
    tool_record = next(
        item
        for item in agent.transcript_entries()
        if item.get("role") == "tool" and item.get("name") == "fork_join"
    )
    joined = json.loads(tool_record["content"])
    assert len(tool_record["content"]) < 4000
    assert len(joined["branches"]) == 4
    assert all(branch["answer_truncated"] for branch in joined["branches"])


def test_parent_session_receives_redacted_branch_results(tmp_path, monkeypatch):
    secret = "fork-secret-value-123456"
    monkeypatch.setenv("CODEY_TEST_SECRET", secret)
    tasks = [
        {"id": "secret", "objective": "inspect secret"},
        {"id": "safe", "objective": "inspect safe"},
    ]

    def model_client_factory(spec):
        answer = secret if spec.branch_id == "secret" else "safe result"
        return FakeModelClient([f"<final>{answer}</final>"])

    agent = build_agent(
        tmp_path,
        [fork_call(tasks), "<final>joined</final>"],
        model_client_factory=model_client_factory,
    )

    agent.ask("redact child evidence")

    tool_record = next(
        item
        for item in agent.transcript_entries()
        if item.get("role") == "tool" and item.get("name") == "fork_join"
    )
    assert secret not in tool_record["content"]
    assert "<redacted>" in tool_record["content"]


def test_fork_child_session_redacts_inherited_parent_history(tmp_path, monkeypatch):
    secret = "inherited-parent-secret-123456"
    monkeypatch.setenv("CODEY_TEST_SECRET", secret)
    tasks = [
        {"id": "a", "objective": "inspect a"},
        {"id": "b", "objective": "inspect b"},
    ]

    def model_client_factory(spec):
        return FakeModelClient([f"<final>result-{spec.branch_id}</final>"])

    agent = build_agent(
        tmp_path,
        [fork_call(tasks), "<final>joined</final>"],
        model_client_factory=model_client_factory,
    )
    agent.record_transcript(
        {"role": "user", "content": secret, "created_at": "test"},
        turn_id="inherited-parent-turn",
    )

    agent.ask("redact inherited history")

    child_sessions = [
        path
        for path in agent.session_store.root.glob("*.json")
        if path.stem != agent.session["id"]
    ]
    assert len(child_sessions) == 2
    for path in child_sessions:
        content = path.read_text(encoding="utf-8")
        assert secret not in content
        assert "<redacted>" in content


def test_fork_children_cannot_execute_risky_write_tools(tmp_path):
    tasks = [
        {"id": "writer", "objective": "try to write"},
        {"id": "reader", "objective": "inspect safely"},
    ]

    def model_client_factory(spec):
        if spec.branch_id == "writer":
            return FakeModelClient(
                [
                    '<tool>{"name":"write_file","args":{"path":"blocked.txt","content":"no"}}</tool>',
                    "<final>write was blocked</final>",
                ]
            )
        return FakeModelClient(["<final>safe result</final>"])

    agent = build_agent(
        tmp_path,
        [fork_call(tasks), "<final>joined</final>"],
        model_client_factory=model_client_factory,
    )

    assert agent.ask("enforce read only") == "joined"
    assert not (tmp_path / "blocked.txt").exists()
    tool_record = next(
        item
        for item in agent.transcript_entries()
        if item.get("role") == "tool" and item.get("name") == "fork_join"
    )
    writer = next(
        branch
        for branch in json.loads(tool_record["content"])["branches"]
        if branch["branch_id"] == "writer"
    )
    child_events = [
        json.loads(line)
        for line in agent.run_store.trace_path(writer["run_id"]).read_text(encoding="utf-8").splitlines()
    ]
    write_event = next(
        event
        for event in child_events
        if event["event"] == "tool_executed" and event["name"] == "write_file"
    )
    assert write_event["tool_error_code"] == "approval_denied"
    assert write_event["security_event_type"] == "read_only_block"


def test_fork_infrastructure_failure_is_closed_in_state_and_trace(tmp_path):
    tasks = [
        {"id": "a", "objective": "inspect a"},
        {"id": "b", "objective": "inspect b"},
    ]
    agent = build_agent(tmp_path, [fork_call(tasks), "<final>handled</final>"])

    def fail_child_creation(**kwargs):
        raise RuntimeError("cannot create child")

    agent.create_fork_child = fail_child_creation
    assert agent.ask("handle infrastructure failure") == "handled"

    fork_state = next(iter(agent.session["forks"]["items"].values()))
    assert fork_state["status"] == "failed"
    assert fork_state["error_type"] == "RuntimeError"
    assert any(event["event"] == "join_failed" for event in trace_events(agent))


def test_legacy_session_fork_shape_is_rejected_and_thread_ids_cannot_be_reused(tmp_path):
    agent = build_agent(tmp_path, ["<final>first</final>"])
    agent.session["forks"] = {"items": []}
    agent.session["checkpoints"] = {
        "current_id": "broken",
        "items": {"broken": []},
    }
    agent.session_store.save(agent.session)
    with pytest.raises(ValueError, match="session checkpoint items are invalid"):
        CodeYAgent.from_session(
            model_client=FakeModelClient(["<final>resumed</final>"]),
            workspace=WorkspaceContext.build(tmp_path),
            session_store=agent.session_store,
            session_id=agent.session["id"],
            approval_policy="auto",
            skill_mode="off",
            feature_flags={"self_evolution": False},
        )

    resumed = build_agent(tmp_path, ["<final>resumed</final>"])
    assert resumed.ask("resume", thread_id="fixed-thread") == "resumed"
    with pytest.raises(ValueError, match="thread_id must be unique"):
        resumed.ask("reuse", thread_id="fixed-thread")


def test_persistent_checkpointer_requires_explicit_sensitive_state_opt_in(tmp_path):
    agent = build_agent(
        tmp_path,
        ["<final>unused</final>"],
        graph_checkpointer=object(),
    )

    with pytest.raises(ValueError, match="persistent graph checkpointers require explicit opt-in"):
        agent.ask("unsafe persistence")
