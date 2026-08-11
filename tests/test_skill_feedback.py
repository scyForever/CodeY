import pytest

from CodeY.cli import propose_description_patch_command, submit_route_feedback_command
from CodeY.skills.feedback import SkillFeedbackStore
from CodeY.skills.router import SkillConfigurationError, SkillRouter, SkillSelection


def write_skill(root, name):
    skill = root / "skills" / name
    skill.mkdir(parents=True)
    description = (
        "This skill should be used when the user's primary objective concerns "
        f"the {name} runtime and the request matches \"{name} lifecycle engineering\" "
        f"or \"{name} harness diagnosis\". It should not activate for generic source review, "
        "unrelated product planning, or ordinary interface styling."
    )
    (skill / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "---\n"
        "<always-applicable>\nCore constraints.\n</always-applicable>\n"
        "<task-routing>\n"
        "```json\n"
        '{"tasks":[{"id":"primary","label":"Primary","triggers":["runtime"],"reads":[]},'
        '{"id":"other","label":"Other","triggers":[],"reads":[]}]}\n'
        "```\n"
        "</task-routing>\n",
        encoding="utf-8",
    )
    return skill


def route_decision(skill_name="", candidates=None):
    return {
        "routing_event_id": "",
        "skill_name": skill_name,
        "route_id": "primary" if skill_name else "",
        "fallback": False,
        "matched_terms": [],
        "loaded_paths": [],
        "selection_source": "description_model",
        "selector_confidence": 0.9 if skill_name else 0.0,
        "selector_reason": "test decision",
        "candidates": list(candidates or []),
    }


def candidate(skill_name, selected):
    return {
        "skill_name": skill_name,
        "activation_phrases": [],
        "near_misses": [],
        "matched_activation_phrases": [],
        "matched_near_misses": [],
        "name_matched": False,
        "lexical_score": 0,
        "selected": selected,
    }


def test_frontmatter_triggers_are_rejected_instead_of_compatibly_loaded(tmp_path):
    skill = write_skill(tmp_path, "alpha")
    path = skill / "SKILL.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "description:",
            'triggers: ["legacy"]\ndescription:',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(SkillConfigurationError, match="unknown SKILL.md fields: triggers"):
        SkillRouter(tmp_path)


def test_model_selection_rejects_near_miss_and_low_confidence_candidates(tmp_path):
    write_skill(tmp_path, "alpha")
    write_skill(tmp_path, "beta")
    router = SkillRouter(tmp_path)

    near_miss = router.route(
        "ordinary interface styling",
        selector=lambda request, skills, candidates: SkillSelection(
            skill_name="alpha",
            confidence=0.95,
            reason="model selected alpha",
            source="description_model",
        ),
    )
    assert near_miss.skill_name == ""
    assert near_miss.selection_source == "description_model_rejected"
    assert near_miss.candidates[0].matched_near_misses == ("ordinary interface styling",)

    low_confidence = router.route(
        "a semantically ambiguous runtime request",
        selector=lambda request, skills, candidates: SkillSelection(
            skill_name="beta",
            confidence=0.2,
            reason="weak model guess",
            source="description_model",
        ),
    )
    assert low_confidence.skill_name == ""
    assert low_confidence.selection_source == "description_model_rejected"
    assert low_confidence.selector_confidence == 0.2


def test_description_model_selection_persists_complete_candidate_evidence(tmp_path):
    write_skill(tmp_path, "alpha")
    write_skill(tmp_path, "beta")
    match = SkillRouter(tmp_path).route(
        "choose the appropriate runtime specialty",
        selector=lambda request, skills, candidates: SkillSelection(
            skill_name="beta",
            confidence=0.9,
            reason="beta owns this request",
            source="description_model",
        ),
    )

    payload = match.to_dict()
    assert payload["skill_name"] == "beta"
    assert payload["selection_source"] == "description_model"
    assert len(payload["candidates"]) == 2
    assert {item["skill_name"] for item in payload["candidates"]} == {"alpha", "beta"}
    assert [item["skill_name"] for item in payload["candidates"] if item["selected"]] == ["beta"]


def test_description_model_failure_degrades_to_no_selection_with_evidence(tmp_path):
    write_skill(tmp_path, "alpha")
    write_skill(tmp_path, "beta")

    match = SkillRouter(tmp_path).route(
        "choose a specialty",
        selector=lambda request, skills, candidates: {"unexpected": True},
    )

    assert match.skill_name == ""
    assert match.selection_source == "description_model_error"
    assert "SkillConfigurationError" in match.selector_reason
    assert len(match.candidates) == 2


def test_unknown_model_selection_degrades_to_no_selection_with_evidence(tmp_path):
    write_skill(tmp_path, "alpha")

    match = SkillRouter(tmp_path).route(
        "choose a specialty",
        selector=lambda request, skills, candidates: SkillSelection(
            skill_name="invented-skill",
            confidence=0.91,
            reason="hallucinated catalog entry",
            source="description_model",
        ),
    )

    assert match.skill_name == ""
    assert match.selection_source == "description_model_error"
    assert "unknown Skill" in match.selector_reason
    assert len(match.candidates) == 1


def test_explicit_feedback_updates_false_activation_and_missed_skill_evidence(tmp_path):
    store = SkillFeedbackStore(tmp_path / "skill-routing")
    event_id = store.start(
        session_id="session-1",
        run_id="run-1",
        task_id="task-1",
        request="route this request",
        route=route_decision(
            "alpha",
            [candidate("alpha", True), candidate("beta", False)],
        ),
    )
    store.submit_user_feedback(
        event_id,
        correct=False,
        expected_skill_name="beta",
        note="beta should have handled this request",
    )

    alpha_patch = store.propose_description_patch("alpha", min_samples=1)
    beta_patch = store.propose_description_patch("beta", min_samples=1)
    assert alpha_patch["status"] == "review_required"
    assert alpha_patch["metrics"]["false_activation_count"] == 1
    assert alpha_patch["metrics"]["precision_proxy"] == 0.0
    assert alpha_patch["evidence"][0]["kind"] == "false_activation"
    assert beta_patch["metrics"]["missed_activation_count"] == 1
    assert beta_patch["metrics"]["recall_proxy"] == 0.0
    assert beta_patch["evidence"][0]["kind"] == "missed_activation"
    assert alpha_patch["proposed_changes"]["near_miss_examples"] == [
        "route this request"
    ]
    assert beta_patch["proposed_changes"]["activation_examples"] == [
        "route this request"
    ]
    assert beta_patch["proposed_changes"]["constraints"]["auto_apply"] is False


def test_skill_feedback_requires_a_real_boolean(tmp_path):
    store = SkillFeedbackStore(tmp_path / "skill-routing")
    event_id = store.start(
        session_id="session-1",
        run_id="run-1",
        task_id="task-1",
        request="route this request",
        route=route_decision("alpha"),
    )

    with pytest.raises(TypeError, match="boolean"):
        store.submit_user_feedback(event_id, correct="false")


class FeedbackCommandAgent:
    def __init__(self):
        self.feedback = None

    def submit_skill_feedback(self, correct, expected_skill_name="", note=""):
        self.feedback = (correct, expected_skill_name, note)
        return {"event_id": "route_test"}

    def propose_skill_description_patch(self, skill_name, min_samples=3):
        return {
            "skill_name": skill_name,
            "min_samples": min_samples,
            "status": "review_required",
        }


def test_repl_feedback_and_description_patch_commands_are_user_reachable():
    agent = FeedbackCommandAgent()

    feedback = submit_route_feedback_command(
        agent,
        '/feedback incorrect beta "beta owns this request"',
    )
    patch = propose_description_patch_command(agent, "/description-patch beta 4")

    assert agent.feedback == (False, "beta", "beta owns this request")
    assert feedback == {
        "event_id": "route_test",
        "verdict": "negative",
        "expected_skill_name": "beta",
    }
    assert patch == {
        "skill_name": "beta",
        "min_samples": 4,
        "status": "review_required",
    }
