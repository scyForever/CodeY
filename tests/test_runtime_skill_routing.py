import json
import shutil
from pathlib import Path

from CodeY import CodeYAgent, FakeModelClient, SessionStore, WorkspaceContext
from CodeY.context.manager import ContextManager


SOURCE_SKILL = Path(__file__).parents[1] / "skills" / "codey"


def build_agent(tmp_path, outputs):
    skill = tmp_path / "skills" / "codey"
    skill.parent.mkdir(parents=True)
    shutil.copytree(SOURCE_SKILL, skill)
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    return CodeYAgent(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=SessionStore(tmp_path / ".codey" / "sessions"),
        approval_policy="auto",
    )


def test_same_session_reroutes_each_new_request(tmp_path):
    client_outputs = ["<final>first</final>", "<final>second</final>"]
    agent = build_agent(tmp_path, client_outputs)
    stable_hash = agent.prefix_state.hash

    assert agent.ask("fix prompt context") == "first"
    first_run = agent.current_task_state.run_id
    assert agent.current_task_state.route_id == "prompt-context"
    assert agent.ask("加强工具安全") == "second"
    assert agent.current_task_state.route_id == "tools-security"
    assert agent.prefix_state.hash == stable_hash

    first_state = json.loads((tmp_path / ".codey" / "runs" / first_run / "task_state.json").read_text(encoding="utf-8"))
    assert first_state["route_id"] == "prompt-context"


def test_prompt_xml_and_route_section_order(tmp_path):
    agent = build_agent(tmp_path, ["<final>done</final>"])
    agent.route_task("fix prompt context")
    prompt, metadata = agent._build_prompt_and_metadata("fix prompt context")

    assert prompt.count("<always-applicable>") >= 1
    assert prompt.count("<task-routing>") >= 1
    assert prompt.index("Selected task route") < prompt.index("Memory:")
    assert prompt.endswith("Current user request:\nfix prompt context")
    assert metadata["route"]["route_id"] == "prompt-context"
    assert metadata["route"]["loaded_paths"]


def test_budget_reduction_preserves_current_request_and_route_floor(tmp_path):
    agent = build_agent(tmp_path, ["<final>done</final>"])
    agent.route_task("fix prompt context")
    agent.context_manager = ContextManager(agent, total_budget=5000)
    for index in range(15):
        agent.record({"role": "assistant", "content": f"history-{index}-" + "x" * 500, "created_at": "now"})

    prompt, metadata = agent._build_prompt_and_metadata("TOKEN-MUST-SURVIVE")

    assert prompt.endswith("Current user request:\nTOKEN-MUST-SURVIVE")
    assert metadata["sections"]["route_context"]["rendered_chars"] >= agent.context_manager.section_floors["route_context"]
