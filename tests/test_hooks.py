import shutil
from pathlib import Path

from CodeY import CodeYAgent, FakeModelClient, SessionStore, WorkspaceContext


SOURCE_SKILL = Path(__file__).parents[1] / "skills" / "codey"


def build_agent(tmp_path, outputs=None, callbacks=None):
    skill = tmp_path / "skills" / "codey"
    skill.parent.mkdir(parents=True)
    shutil.copytree(SOURCE_SKILL, skill)
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".codey" / "sessions")
    return CodeYAgent(
        model_client=FakeModelClient(outputs or ["<final>done</final>"]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        hook_callbacks=callbacks,
    )


def test_startup_reset_and_resume_hooks(tmp_path):
    events = []
    agent = build_agent(tmp_path, callbacks=[lambda event, payload: events.append((event.reason, payload["generation"]))])
    assert agent.session["session_context"]["reason"] == "startup"

    agent.reset()
    assert agent.session["session_context"]["reason"] == "reset"
    resumed = CodeYAgent.from_session(
        model_client=FakeModelClient(["<final>done</final>"]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=agent.session_store,
        session_id=agent.session["id"],
        approval_policy="auto",
    )
    assert resumed.session["session_context"]["reason"] == "resume"
    assert [reason for reason, _ in events] == ["startup", "reset"]


def test_compact_hook_is_once_per_run(tmp_path):
    agent = build_agent(tmp_path)
    before = agent.session["session_context"]["generation"]

    assert agent.compact_session_context("run-1") is True
    assert agent.compact_session_context("run-1") is False
    assert agent.session["session_context"]["generation"] == before + 1
    assert agent.session["session_context"]["reason"] == "compact"


def test_hook_never_loads_route_specific_body(tmp_path):
    agent = build_agent(tmp_path)
    context = agent.session["session_context"]["text"]

    assert "Change the agent runtime" not in context
    assert "Change tools safely" not in context
    assert "Project rules" in context
