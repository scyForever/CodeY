import hashlib
import json

import pytest

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


def build_agent(
    tmp_path,
    outputs,
    thresholds=None,
    feature_flags=None,
    evolution_llm_config=None,
    evolution_llm_client=None,
):
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
        evolution_llm_config=evolution_llm_config,
        evolution_llm_client=evolution_llm_client,
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
    assert policy["metrics"]["eligible_count"] == 0
    assert policy["metrics"]["exposed_count"] == 0
    assert policy["metrics"]["triggered_count"] == 0

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
    assert strategy["metrics"]["triggered_count"] == 0
    assert strategy["metrics"]["exposure_rate"] == 1.0
    assert strategy["metrics"]["hit_rate"] == 0.0


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
    assert strategy["metrics"]["triggered_count"] == 2
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


def test_legacy_task_state_is_rejected_without_implicit_migration():
    with pytest.raises(ValueError, match="legacy task states are not supported"):
        TaskState.from_dict(
            {
                "run_id": "run-old",
                "task_id": "task-old",
                "user_request": "old task",
                "status": "completed",
            }
        )


def test_hybrid_advisor_disambiguates_root_cause_and_refines_patch(tmp_path):
    advisor = FakeModelClient(
        [
            json.dumps(
                {
                    "outcome": {
                        "label": "incorrect",
                        "confidence": 0.94,
                        "evidence_refs": ["tool_001"],
                        "reason_code": "critical_tool_failure",
                        "patch_eligible": True,
                    },
                    "root_cause": {
                        "level": "execution",
                        "confidence": 0.91,
                        "evidence_refs": ["tool_001"],
                        "cause_code": "invalid_tool_arguments",
                    },
                }
            ),
            json.dumps(
                {
                    "patches": [
                        {
                            "candidate_index": 0,
                            "correction_action": "Validate read_file path arguments before execution.",
                            "trigger_conditions": [{"signal": "tool_name", "equals": "read_file"}],
                            "confidence": 0.9,
                            "evidence_refs": ["tool_001"],
                        }
                    ]
                }
            ),
        ]
    )
    agent, main_client = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"start":1,"end":2}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":2}}</tool>',
            "<final>recovered</final>",
        ],
        evolution_llm_config={"mode": "hybrid", "min_confidence": 0.8},
        evolution_llm_client=advisor,
    )

    assert agent.ask("inspect runtime") == "recovered"

    result = agent.last_cognitive_loop
    assert result["outcome"]["label"] == "incorrect"
    assert result["outcome"]["decision_source"] == "hybrid_llm"
    assert result["root_cause"]["level"] == "execution"
    assert result["root_cause"]["decision_source"] == "hybrid_llm"
    assert result["decision_audit"]["diagnostic"]["status"] == "accepted"
    assert result["decision_audit"]["patch_generation"]["status"] == "accepted"

    strategy = patches_by_type(agent, "strategy")[0]
    assert strategy["status"] == "shadow"
    assert strategy["correction"]["kind"] == "execution_guard"
    assert strategy["correction"]["action"] == "Validate read_file path arguments before execution."
    assert strategy["source"]["proposal_origin"] == "hybrid_llm"
    assert strategy["source"]["advisor_evidence_refs"] == ["tool_001"]
    assert len(advisor.prompts) == 2
    assert len(main_client.prompts) == 3


def test_hybrid_advisor_rejects_unverifiable_or_secret_shaped_advice(tmp_path):
    leaked_value = "sk-LEAKEDVALUE123456"
    advisor = FakeModelClient(
        [
            json.dumps(
                {
                    "outcome": {
                        "label": "incorrect",
                        "confidence": 0.99,
                        "evidence_refs": ["invented_evidence"],
                        "reason_code": "unsupported_claim",
                        "patch_eligible": True,
                    },
                    "root_cause": {
                        "level": "execution",
                        "confidence": 0.99,
                        "evidence_refs": ["invented_evidence"],
                        "cause_code": "unsupported_cause",
                    },
                }
            ),
            json.dumps(
                {
                    "patches": [
                        {
                            "candidate_index": 0,
                            "correction_action": f"Reuse secret token {leaked_value} before every read.",
                            "trigger_conditions": [{"signal": "tool_name", "equals": "read_file"}],
                            "confidence": 0.99,
                            "evidence_refs": ["tool_001"],
                        }
                    ]
                }
            ),
        ]
    )
    agent, _ = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"start":1,"end":2}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":2}}</tool>',
            "<final>recovered</final>",
        ],
        evolution_llm_config={"mode": "hybrid"},
        evolution_llm_client=advisor,
    )

    agent.ask("inspect runtime")

    result = agent.last_cognitive_loop
    assert result["outcome"]["label"] == "partial"
    assert result["root_cause"]["level"] == "chain"
    assert result["decision_audit"]["diagnostic"]["status"] == "fallback_validation_failed"
    assert result["decision_audit"]["patch_generation"]["status"] == "fallback_validation_failed"
    chain = patches_by_type(agent, "action_chain")[0]
    assert chain["source"]["proposal_origin"] == "rules"
    serialized = json.dumps({"result": result, "patch": chain}, sort_keys=True)
    assert leaked_value not in serialized
    assert "invented_evidence" not in serialized


def test_hybrid_patch_identity_is_stable_across_llm_paraphrases(tmp_path):
    diagnostic = json.dumps(
        {
            "outcome": {
                "label": "partial",
                "confidence": 0.9,
                "evidence_refs": ["tool_001"],
                "reason_code": "completed_after_tool_failure",
                "patch_eligible": True,
            },
            "root_cause": None,
        }
    )
    first_patch = json.dumps(
        {
            "patches": [
                {
                    "candidate_index": 0,
                    "correction_action": "Validate the read path before invoking read_file.",
                    "trigger_conditions": [{"signal": "tool_name", "equals": "read_file"}],
                    "confidence": 0.9,
                    "evidence_refs": ["tool_001"],
                }
            ]
        }
    )
    paraphrased_patch = json.dumps(
        {
            "patches": [
                {
                    "candidate_index": 0,
                    "correction_action": "Check the requested path before calling read_file.",
                    "trigger_conditions": [{"signal": "tool_name", "equals": "read_file"}],
                    "confidence": 0.92,
                    "evidence_refs": ["tool_001"],
                }
            ]
        }
    )
    advisor = FakeModelClient([diagnostic, first_patch, diagnostic, paraphrased_patch])
    agent, _ = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"start":1,"end":2}}</tool>',
            "<final>recovered</final>",
            '<tool>{"name":"read_file","args":{"start":1,"end":2}}</tool>',
            "<final>recovered again</final>",
        ],
        evolution_llm_config={"mode": "hybrid"},
        evolution_llm_client=advisor,
    )

    agent.ask("inspect runtime")
    first = patches_by_type(agent, "strategy")[0]
    agent.ask("inspect runtime again")

    patches = patches_by_type(agent, "strategy")
    assert len(patches) == 1
    assert patches[0]["patch_id"] == first["patch_id"]
    assert patches[0]["correction"]["action"] == "Validate the read path before invoking read_file."
    assert agent.last_cognitive_loop["generated_patches"][0]["created"] is False


def test_hybrid_advisor_cannot_override_policy_review_state(tmp_path):
    advisor = FakeModelClient(
        [
            json.dumps(
                {
                    "outcome": {
                        "label": "partial",
                        "confidence": 0.95,
                        "evidence_refs": ["tool_001"],
                        "reason_code": "blocked_policy_violation",
                        "patch_eligible": True,
                    },
                    "root_cause": {
                        "level": "execution",
                        "confidence": 0.99,
                        "evidence_refs": ["tool_001"],
                        "cause_code": "downgrade_policy",
                    },
                }
            ),
            json.dumps(
                {
                    "patches": [
                        {
                            "candidate_index": 0,
                            "correction_action": "Resolve paths inside the workspace before reading.",
                            "trigger_conditions": [{"signal": "tool_name", "equals": "read_file"}],
                            "confidence": 0.95,
                            "evidence_refs": ["tool_001"],
                            "status": "active",
                        }
                    ]
                }
            ),
        ]
    )
    agent, _ = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"../outside.txt","start":1,"end":1}}</tool>',
            "<final>recovered</final>",
        ],
        evolution_llm_config={"mode": "hybrid"},
        evolution_llm_client=advisor,
    )

    agent.ask("inspect a path")

    policy = patches_by_type(agent, "policy")[0]
    assert agent.last_cognitive_loop["root_cause"]["level"] == "policy"
    assert policy["status"] == "review_required"
    assert policy["source"]["proposal_origin"] == "rules"
    assert agent.last_cognitive_loop["decision_audit"]["patch_generation"]["status"] == (
        "fallback_validation_failed"
    )
    assert not (tmp_path / ".codey" / "evolution" / "behavior" / "policies.md").exists()


def test_hybrid_advisor_is_not_called_when_rules_are_decisive_and_no_patch_exists(tmp_path):
    advisor = FakeModelClient(["unused"])
    agent, _ = build_agent(
        tmp_path,
        ["<final>done</final>"],
        evolution_llm_config={"mode": "hybrid"},
        evolution_llm_client=advisor,
    )

    assert agent.ask("finish") == "done"
    assert advisor.prompts == []
    assert agent.last_cognitive_loop["outcome"]["label"] == "correct"
    assert agent.last_cognitive_loop["decision_audit"]["diagnostic"]["status"] == "skipped_rule_decisive"
    assert agent.last_cognitive_loop["decision_audit"]["patch_generation"]["status"] == "skipped_no_candidates"


def test_legacy_patch_schema_and_old_metric_fields_are_rejected(tmp_path):
    agent, _ = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"start":1,"end":2}}</tool>',
            "<final>recovered</final>",
        ],
    )
    agent.ask("inspect runtime")
    patch = patches_by_type(agent, "strategy")[0]
    path = agent.cognitive_loop.store.patches_dir / f"{patch['patch_id']}.json"

    legacy = dict(patch)
    legacy["schema_version"] = 2
    path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version must be 3"):
        agent.cognitive_loop.store.load_patch(patch["patch_id"])

    invalid_metrics = dict(patch)
    invalid_metrics["metrics"] = dict(patch["metrics"])
    invalid_metrics["metrics"]["hit_count"] = invalid_metrics["metrics"].pop("triggered_count")
    path.write_text(json.dumps(invalid_metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics schema is invalid"):
        agent.cognitive_loop.store.load_patch(patch["patch_id"])


def test_only_eight_eligible_patches_are_exposed_and_observed_per_run(tmp_path):
    agent, _ = build_agent(tmp_path, [])
    store = agent.cognitive_loop.store
    for index in range(9):
        fingerprint = hashlib.sha256(f"candidate-{index}".encode("utf-8")).hexdigest()
        patch, created = store.create_candidate(
            {
                "type": "strategy",
                "scope": {},
                "correction": {
                    "kind": "execution_guard",
                    "action": f"Apply bounded strategy {index}.",
                },
                "trigger_conditions": [{"signal": "task_scope", "equals": "workspace"}],
                "source": {
                    "rule_candidate_fingerprint": fingerprint,
                    "proposal_origin": "rules",
                },
            }
        )
        assert created is True
        agent.cognitive_loop.safety_gate.initialize(patch)

    state = TaskState.create("task-exposure", "exercise patch exposure", run_id="run-exposure")
    state.finish_success("done")
    context, guidance = agent.cognitive_loop.prepare_run(state)
    state.evolution_context = context

    assert len(context["eligible_patch_ids"]) == 9
    assert len(context["exposed_patch_ids"]) == 8
    assert context["exposure_scheduler"] == "least_exposed_first_v1"
    assert guidance.count("[shadow:strategy:") == 8

    agent.cognitive_loop.complete_run(state, [])
    exposed = set(context["exposed_patch_ids"])
    for patch_id in context["eligible_patch_ids"]:
        metrics = store.load_patch(patch_id)["metrics"]
        assert metrics["eligible_count"] == 1
        if patch_id in exposed:
            assert metrics["exposed_count"] == 1
            assert metrics["triggered_count"] == 1
            assert metrics["success_count"] == 1
        else:
            assert metrics["exposed_count"] == 0
            assert metrics["triggered_count"] == 0
            assert metrics["success_count"] == 0

    deferred = (set(context["eligible_patch_ids"]) - exposed).pop()
    next_state = TaskState.create(
        "task-exposure-next",
        "exercise patch exposure again",
        run_id="run-exposure-next",
    )
    next_context, _ = agent.cognitive_loop.prepare_run(next_state)
    assert deferred in next_context["exposed_patch_ids"]


def test_patch_trigger_conditions_use_strict_and_semantics():
    patch = {
        "trigger_conditions": [
            {"signal": "tool_name", "equals": "read_file"},
            {"signal": "path", "equals": "README.md"},
        ]
    }
    partial_trace = {
        "tool_events": [
            {"name": "read_file", "path": "other.txt", "affected_paths": []}
        ]
    }
    complete_trace = {
        "tool_events": [
            {"name": "read_file", "path": "README.md", "affected_paths": []}
        ]
    }

    assert agent_trigger_matches(patch, partial_trace) is False
    assert agent_trigger_matches(patch, complete_trace) is True


def agent_trigger_matches(patch, trace):
    from CodeY.evolution.cognitive import CognitiveLoop

    return CognitiveLoop._trigger_matches(patch, trace)
