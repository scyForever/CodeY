import json
from pathlib import Path

import pytest

from CodeY import CodeYAgent, FakeModelClient, SessionStore, SkillConfigurationError, SkillRouter, WorkspaceContext
from CodeY.context.manager import ContextManager


def write_skill(root, name="demo"):
    skill = root / "skills" / name
    (skill / "rules").mkdir(parents=True)
    (skill / "workflows").mkdir()
    (skill / "rules" / "core.md").write_text("CORE-ONLY\n", encoding="utf-8")
    (skill / "rules" / "prompt.md").write_text("PROMPT-DETAIL\n", encoding="utf-8")
    (skill / "rules" / "tools.md").write_text("TOOLS-DETAIL\n", encoding="utf-8")
    (skill / "workflows" / "prompt.md").write_text("PROMPT-WORKFLOW\n", encoding="utf-8")
    (skill / "SKILL.md").write_text(
        f'''---
name: {name}
description: Demo coding agent skill
triggers: ["demo", "演示"]
---
<always-applicable>
Always Read:
- rules/core.md
</always-applicable>
<task-routing>
```json
{{"tasks":[
{{"id":"prompt","label":"Prompt / 提示词","triggers":["prompt","提示词"],"workflow":"workflows/prompt.md","reads":["rules/prompt.md"]}},
{{"id":"tools","label":"Tools / 工具","triggers":["tool","工具"],"reads":["rules/tools.md"]}},
{{"id":"other","label":"Other","triggers":[],"reads":[]}}
]}}
```
</task-routing>
''',
        encoding="utf-8",
    )
    return skill


def build_agent(root, outputs=None, callbacks=None):
    workspace = WorkspaceContext.build(root)
    return CodeYAgent(
        model_client=FakeModelClient(outputs or ["<final>done</final>"]),
        workspace=workspace,
        session_store=SessionStore(root / ".codey" / "sessions"),
        approval_policy="auto",
        hook_callbacks=callbacks,
    )


def test_progressive_two_level_routing(tmp_path):
    write_skill(tmp_path)
    router = SkillRouter(tmp_path)
    context = router.session_start()
    assert "CORE-ONLY" in context.text
    assert "PROMPT-DETAIL" not in context.text

    match = router.route("请修改提示词 prompt")
    assert match.skill_name == "demo"
    assert match.route_id == "prompt"
    assert "PROMPT-WORKFLOW" in match.route_context
    assert "PROMPT-DETAIL" in match.route_context
    assert "TOOLS-DETAIL" not in match.route_context

    fallback = router.route("demo unrelated request")
    assert fallback.route_id == "other"
    assert fallback.fallback is True


def test_explicit_unknown_and_path_escape_fail(tmp_path):
    write_skill(tmp_path)
    router = SkillRouter(tmp_path)
    with pytest.raises(SkillConfigurationError, match="unknown skill"):
        router.route("/missing do work")

    skill_file = tmp_path / "skills" / "demo" / "SKILL.md"
    text = skill_file.read_text(encoding="utf-8").replace('"reads":["rules/tools.md"]', '"reads":["../../outside.md"]')
    skill_file.write_text(text, encoding="utf-8")
    router = SkillRouter(tmp_path)
    with pytest.raises(SkillConfigurationError, match="invalid skill read path|escapes"):
        router.route("tool")


def test_session_start_lifecycle_and_xml_prefix(tmp_path):
    write_skill(tmp_path)
    events = []
    agent = build_agent(tmp_path, callbacks=[lambda event, payload: events.append((event.reason, payload["generation"]))])
    assert events == [("startup", 1)]
    assert "<always-applicable>" in agent.prefix
    assert "<task-routing>" in agent.prefix
    assert "PROMPT-DETAIL" not in agent.prefix

    agent.reset()
    assert events[-1][0] == "reset"
    assert agent.session["session_context"]["reason"] == "reset"
    assert agent.compact_session_context("run-1") is True
    assert agent.compact_session_context("run-1") is False
    assert events[-1][0] == "compact"


def test_runtime_reroutes_each_ask_and_persists_metadata(tmp_path):
    write_skill(tmp_path)
    agent = build_agent(tmp_path, ["<final>one</final>", "<final>two</final>"])
    assert agent.ask("demo prompt change") == "one"
    first_hash = agent.prefix_state.hash
    first_run = agent.current_task_state.run_id
    assert agent.current_task_state.route_id == "prompt"

    assert agent.ask("demo tool change") == "two"
    assert agent.current_task_state.route_id == "tools"
    assert agent.prefix_state.hash == first_hash

    first_state = json.loads((tmp_path / ".codey" / "runs" / first_run / "task_state.json").read_text(encoding="utf-8"))
    assert first_state["route_id"] == "prompt"
    report = json.loads((tmp_path / ".codey" / "runs" / agent.current_task_state.run_id / "report.json").read_text(encoding="utf-8"))
    assert report["skill_route"]["route_id"] == "tools"


def test_route_context_is_between_prefix_and_memory_and_request_is_preserved(tmp_path):
    write_skill(tmp_path)
    agent = build_agent(tmp_path)
    agent.route_task("demo prompt")
    manager = ContextManager(agent, total_budget=9000)
    request = "CURRENT-REQUEST-UNTRUNCATED"
    prompt, metadata = manager.build(request)
    assert prompt.index("<always-applicable>") < prompt.index("Selected task route: prompt") < prompt.index("Memory:")
    assert prompt.endswith(f"Current user request:\n{request}")
    assert metadata["route"]["route_id"] == "prompt"
