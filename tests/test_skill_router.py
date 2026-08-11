import shutil
from pathlib import Path

import pytest

from CodeY.skills.router import SkillConfigurationError, SkillRouter


SOURCE_SKILL = Path(__file__).parents[1] / "skills" / "codey"


def copy_skill(root):
    destination = root / "skills" / "codey"
    destination.parent.mkdir(parents=True)
    shutil.copytree(SOURCE_SKILL, destination)
    return destination


def test_routes_english_and_chinese_without_loading_other_routes(tmp_path):
    copy_skill(tmp_path)
    router = SkillRouter(tmp_path)

    prompt_route = router.route("CodeY agent runtime engineering: Fix the prompt context budget")
    tool_route = router.route("local coding assistant harness: 加强工具安全和审批")

    assert prompt_route.route_id == "prompt-context"
    assert "rules/prompt-context.md" in prompt_route.loaded_paths
    assert "tools-security.md" not in prompt_route.route_context
    assert tool_route.route_id == "tools-security"
    assert "rules/tools-security.md" in tool_route.loaded_paths


def test_single_skill_requires_description_match_or_explicit_invocation(tmp_path):
    copy_skill(tmp_path)
    router = SkillRouter(tmp_path)

    unmatched = router.route("unclassified maintenance")
    assert unmatched.skill_name == ""
    assert unmatched.selection_source == "no_description_match"
    explicit = router.route("/codey inspect something")
    assert explicit.skill_name == "codey"
    assert explicit.route_id == "other"
    with pytest.raises(SkillConfigurationError, match="unknown skill"):
        router.route("/missing inspect something")


def test_session_start_loads_only_always_read_files(tmp_path):
    copy_skill(tmp_path)
    context = SkillRouter(tmp_path).session_start()

    assert "project-rules.md" in context.loaded_paths
    assert "runtime-architecture.md" in context.loaded_paths
    assert "prompt-context.md" not in context.text
    assert context.text.count("<always-applicable>") >= 1
    assert context.text.count("<task-routing>") >= 1


def test_fingerprint_changes_only_with_skill_core(tmp_path):
    skill = copy_skill(tmp_path)
    first = SkillRouter(tmp_path).session_start().fingerprint
    (tmp_path / "README.md").write_text("unrelated", encoding="utf-8")
    second = SkillRouter(tmp_path).session_start().fingerprint
    (skill / "rules" / "project-rules.md").write_text("changed", encoding="utf-8")
    third = SkillRouter(tmp_path).session_start().fingerprint

    assert first == second
    assert third != second


def test_rejects_route_path_escape(tmp_path):
    skill = copy_skill(tmp_path)
    skill_file = skill / "SKILL.md"
    text = skill_file.read_text(encoding="utf-8").replace('"reads": ["rules/prompt-context.md"]', '"reads": ["../outside.md"]')
    skill_file.write_text(text, encoding="utf-8")

    with pytest.raises(SkillConfigurationError, match="invalid skill read path"):
        SkillRouter(tmp_path).route("/codey prompt")


def test_multiple_skills_without_match_returns_auditable_no_selection(tmp_path):
    first = copy_skill(tmp_path)
    second = tmp_path / "skills" / "other-skill"
    shutil.copytree(first, second)
    text = (second / "SKILL.md").read_text(encoding="utf-8").replace(
        "name: codey",
        "name: other-skill",
    ).replace(
        "description: This skill should be used when the user's primary objective is to maintain or extend the CodeY local coding-agent runtime, and the request is best characterized as \"CodeY agent runtime engineering\" or \"local coding assistant harness\". It should not activate for generic source-code questions, unrelated agent products, or ordinary application feature work.",
        "description: This skill should be used when the user's primary objective is to maintain another local runtime and the request is best characterized as \"other runtime lifecycle maintenance\" or \"alternative assistant harness engineering\". It should not activate for generic source-code questions, CodeY runtime changes, or ordinary application feature work.",
    )
    (second / "SKILL.md").write_text(text, encoding="utf-8")

    match = SkillRouter(tmp_path).route("unknown request")
    assert match.skill_name == ""
    assert match.selection_source == "no_description_match"
    assert len(match.candidates) == 2
