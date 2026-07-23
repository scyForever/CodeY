import json

from CodeY import CodeYAgent, FakeModelClient, SessionStore, WorkspaceContext
from CodeY.core.task_state import TaskState


FAST_THRESHOLDS = {
    "canary_fraction": 1.0,
    "min_canary_hits": 1,
    "min_canary_hit_rate": 1.0,
    "min_canary_success_rate": 1.0,
    "expiry_min_hits": 100,
    "expiry_success_rate": 0.4,
}


def build_agent(tmp_path, outputs, thresholds=None, feature_flags=None):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    client = FakeModelClient(outputs)
    agent = CodeYAgent(
        model_client=client,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".codey" / "sessions"),
        approval_policy="auto",
        skill_mode="off",
        feature_flags=feature_flags,
        evolution_thresholds=thresholds or FAST_THRESHOLDS,
    )
    return agent, client


def patches_by_type(agent, patch_type):
    return [patch for patch in agent.cognitive_loop.store.list_patches() if patch["type"] == patch_type]


def test_policy_patch_requires_human_review_before_materialization(tmp_path):
    agent, _ = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"../outside.txt","start":1,"end":1}}</tool>',
            "<final>recovered</final>",
            "<final>clean follow-up</final>",
        ],
    )

    agent.ask("inspect a path")
    policy = patches_by_type(agent, "policy")[0]
    assert policy["status"] == "review_required"
    assert agent.last_cognitive_loop["outcome"]["label"] == "partial"
    assert agent.last_cognitive_loop["root_cause"]["level"] == "policy"
    assert not (tmp_path / ".codey" / "evolution" / "behavior" / "policies.md").exists()

    agent.ask("inspect another path")
    policy = agent.cognitive_loop.store.load_patch(policy["patch_id"])
    assert policy["status"] == "review_required"
    assert policy["metrics"]["hit_count"] == 0

    approved = agent.approve_cognitive_patch(policy["patch_id"])
    policy_path = tmp_path / ".codey" / "evolution" / "behavior" / "policies.md"
    assert approved["status"] == "active"
    assert policy["patch_id"] in policy_path.read_text(encoding="utf-8")


def test_strategy_patch_uses_shadow_guidance_and_auto_activates(tmp_path):
    agent, client = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"start":1,"end":2}}</tool>',
            "<final>recovered</final>",
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":2}}</tool>',
            "<final>clean follow-up</final>",
        ],
    )

    agent.ask("inspect runtime")
    strategy = patches_by_type(agent, "strategy")[0]
    assert strategy["status"] == "shadow"
    assert set(strategy) >= {
        "patch_id",
        "type",
        "scope",
        "correction",
        "trigger_conditions",
        "status",
        "metrics",
    }

    agent.ask("inspect runtime again")
    strategy = agent.cognitive_loop.store.load_patch(strategy["patch_id"])
    assert strategy["status"] == "active"
    assert strategy["metrics"]["hit_rate"] == 1.0
    assert strategy["metrics"]["success_rate"] == 1.0
    assert f"[shadow:strategy:{strategy['patch_id']}]" in client.prompts[-1]
    decisions = (tmp_path / ".codey" / "evolution" / "decisions.md").read_text(encoding="utf-8")
    assert strategy["patch_id"] in decisions

    metrics_before_replay = dict(strategy["metrics"])
    agent.cognitive_loop.complete_run(
        agent.current_task_state,
        [],
        redactor=agent.redact_artifact,
    )
    strategy = agent.cognitive_loop.store.load_patch(strategy["patch_id"])
    assert strategy["metrics"] == metrics_before_replay

    serialized = json.dumps(strategy, sort_keys=True)
    assert "old_text" not in serialized
    assert "new_text" not in serialized


def test_unrelated_success_does_not_count_as_a_patch_hit(tmp_path):
    agent, _ = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"start":1,"end":2}}</tool>',
            "<final>recovered</final>",
            "<final>unrelated success</final>",
        ],
    )

    agent.ask("inspect runtime")
    strategy = patches_by_type(agent, "strategy")[0]
    agent.ask("inspect runtime again")

    strategy = agent.cognitive_loop.store.load_patch(strategy["patch_id"])
    assert strategy["status"] == "shadow"
    assert strategy["metrics"]["eligible_count"] == 1
    assert strategy["metrics"]["hit_count"] == 0


def test_verified_path_is_materialized_as_experience_after_canary(tmp_path):
    agent, _ = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":2}}</tool>',
            "<final>read complete</final>",
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":2}}</tool>',
            "<final>clean follow-up</final>",
        ],
    )

    agent.ask("inspect README")
    experience = patches_by_type(agent, "knowledge_experience")[0]
    assert experience["status"] == "shadow"
    assert agent.last_cognitive_loop["reflection"]["new_knowledge"]["answer"] == "yes"

    agent.ask("inspect README again")
    experience = agent.cognitive_loop.store.load_patch(experience["patch_id"])
    experience_path = (
        tmp_path
        / ".codey"
        / "evolution"
        / "knowledge"
        / "experience"
        / f"{experience['patch_id']}.md"
    )
    assert experience["status"] == "active"
    assert "README.md" in experience_path.read_text(encoding="utf-8")


def test_architecture_definition_requires_review_before_materialization(tmp_path):
    agent, _ = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"architecture.md","start":1,"end":2}}</tool>',
            "<final>architecture read</final>",
        ],
    )
    (tmp_path / "architecture.md").write_text("# Runtime Architecture\n", encoding="utf-8")

    agent.ask("inspect architecture")
    definition = patches_by_type(agent, "knowledge_definition")[0]
    definition_path = (
        tmp_path
        / ".codey"
        / "evolution"
        / "knowledge"
        / "definition"
        / f"{definition['patch_id']}.md"
    )
    assert definition["status"] == "review_required"
    assert not definition_path.exists()

    agent.approve_cognitive_patch(definition["patch_id"])
    assert "architecture.md" in definition_path.read_text(encoding="utf-8")


def test_stale_summary_is_reported_without_storing_the_diff(tmp_path):
    agent, _ = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":2}}</tool>',
            '<tool name="patch_file" path="README.md"><old_text>demo</old_text><new_text>RAW_DIFF_SENTINEL_12345</new_text></tool>',
            "<final>updated</final>",
        ],
    )

    agent.ask("update README")

    stale = agent.last_cognitive_loop["reflection"]["stale_knowledge"]
    assert stale == {"answer": "yes", "items": ["README.md"]}
    strategy = [
        patch
        for patch in patches_by_type(agent, "strategy")
        if patch["correction"]["kind"] == "freshness_guard"
    ][0]
    serialized = json.dumps(strategy, sort_keys=True)
    assert strategy["status"] == "shadow"
    assert "old_text" not in serialized
    assert "new_text" not in serialized
    assert "RAW_DIFF_SENTINEL_12345" not in serialized


def test_action_chain_patch_can_auto_activate(tmp_path):
    agent, _ = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"start":1,"end":2}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":2}}</tool>',
            "<final>recovered</final>",
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":2}}</tool>',
            "<final>clean follow-up</final>",
        ],
    )

    agent.ask("inspect a chain")
    chain = patches_by_type(agent, "action_chain")[0]
    assert chain["status"] == "shadow"

    agent.ask("inspect a chain again")
    chain = agent.cognitive_loop.store.load_patch(chain["patch_id"])
    assert chain["status"] == "active"
    assert chain["patch_id"] in (
        tmp_path / ".codey" / "evolution" / "decisions.md"
    ).read_text(encoding="utf-8")


def test_expired_knowledge_is_removed_from_the_active_view(tmp_path):
    thresholds = {
        **FAST_THRESHOLDS,
        "expiry_min_hits": 2,
        "expiry_success_rate": 0.75,
    }
    agent, _ = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":2}}</tool>',
            "<final>read complete</final>",
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":2}}</tool>',
            "<final>clean follow-up</final>",
        ],
        thresholds=thresholds,
    )

    agent.ask("inspect README")
    experience = patches_by_type(agent, "knowledge_experience")[0]
    agent.ask("inspect README again")
    experience_path = (
        tmp_path
        / ".codey"
        / "evolution"
        / "knowledge"
        / "experience"
        / f"{experience['patch_id']}.md"
    )
    assert experience_path.exists()

    agent.model_client.outputs.append(
        '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":2}}</tool>'
    )
    agent.ask("inspect README once more")

    experience = agent.cognitive_loop.store.load_patch(experience["patch_id"])
    assert experience["status"] == "expired"
    assert not experience_path.exists()


def test_shadow_patch_expires_after_repeated_low_success(tmp_path):
    thresholds = {
        **FAST_THRESHOLDS,
        "min_canary_hits": 999,
        "expiry_min_hits": 2,
        "expiry_success_rate": 0.5,
    }
    agent, _ = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"start":1,"end":2}}</tool>',
            "<final>recovered</final>",
        ],
        thresholds=thresholds,
    )

    agent.ask("inspect runtime")
    strategy = patches_by_type(agent, "strategy")[0]
    agent.model_client.outputs.append(
        '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":2}}</tool>'
    )
    agent.ask("inspect runtime again")
    agent.model_client.outputs.append(
        '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":2}}</tool>'
    )
    agent.ask("inspect runtime once more")

    strategy = agent.cognitive_loop.store.load_patch(strategy["patch_id"])
    assert strategy["status"] == "expired"
    assert strategy["metrics"]["hit_count"] == 2
    assert strategy["metrics"]["success_rate"] == 0.0
    assert strategy["history"][-1]["reason"] == "success_rate_below_expiry_threshold"


def test_disabled_loop_does_not_create_evolution_artifacts(tmp_path):
    agent, _ = build_agent(
        tmp_path,
        ["<final>done</final>"],
        feature_flags={"self_evolution": False},
    )

    agent.ask("finish")

    assert agent.last_cognitive_loop == {"status": "disabled"}
    assert not (tmp_path / ".codey" / "evolution").exists()


def test_secret_shaped_path_is_redacted_before_patch_generation(tmp_path):
    secret_path = "secret-sk-ABCDEF123456.txt"
    agent, _ = build_agent(
        tmp_path,
        [
            f'<tool>{{"name":"read_file","args":{{"path":"{secret_path}","start":1,"end":2}}}}</tool>',
            "<final>read complete</final>",
        ],
    )
    (tmp_path / secret_path).write_text("sensitive\n", encoding="utf-8")

    agent.ask("inspect a sensitive path")

    assert secret_path not in json.dumps(agent.last_cognitive_loop, sort_keys=True)
    assert agent.cognitive_loop.store.list_patches() == []


def test_cognitive_failure_is_isolated_and_run_finished_is_terminal(tmp_path):
    agent, _ = build_agent(tmp_path, ["<final>done</final>"])

    def fail_cognitive_loop(*args, **kwargs):
        raise RuntimeError("cognitive failure")

    agent.cognitive_loop.complete_run = fail_cognitive_loop
    assert agent.ask("finish") == "done"

    run_id = agent.current_task_state.run_id
    report = agent.run_store.load_report(run_id)
    events = [
        json.loads(line)["event"]
        for line in agent.run_store.trace_path(run_id).read_text(encoding="utf-8").splitlines()
    ]
    assert report["cognitive_loop"]["status"] == "error"
    assert "cognitive_loop_failed" in events
    assert events[-1] == "run_finished"


def test_old_task_state_without_evolution_context_is_still_loadable():
    state = TaskState.from_dict(
        {
            "run_id": "run-old",
            "task_id": "task-old",
            "user_request": "old task",
            "status": "completed",
        }
    )

    assert state.evolution_context == {}
    assert state.to_dict()["evolution_context"] == {}
