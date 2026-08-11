from CodeY.cli import build_arg_parser
from CodeY.evaluation.metrics import (
    build_stress_agent_metrics,
    run_context_ablation_v3,
    run_recovery_ablation_v2,
)


def test_context_ablation_compares_budgeted_and_unbounded_prompts():
    metrics = build_stress_agent_metrics()

    assert set(metrics) == {"budgeted", "unbounded", "no_memory"}
    assert metrics["budgeted"]["prompt_chars"] < metrics["unbounded"]["prompt_chars"]
    assert metrics["budgeted"]["budget_reduction_count"] >= 0
    assert metrics["budgeted"]["current_request_preserved"] is True
    assert metrics["unbounded"]["current_request_preserved"] is True


def test_cli_exposes_summary_and_skill_selector_controls():
    args = build_arg_parser().parse_args(
        [
            "--summary-model",
            "summary-model",
            "--summary-recent-turns",
            "8",
            "--summary-max-new-tokens",
            "700",
            "--summary-max-chars",
            "5000",
            "--summary-flush-timeout",
            "12.5",
            "--skill-selector-model",
            "selector-model",
            "--skill-selector-max-new-tokens",
            "300",
        ]
    )

    assert args.summary_model == "summary-model"
    assert args.summary_recent_turns == 8
    assert args.summary_max_new_tokens == 700
    assert args.summary_max_chars == 5000
    assert args.summary_flush_timeout == 12.5
    assert args.skill_selector_model == "selector-model"
    assert args.skill_selector_max_new_tokens == 300


def test_versioned_context_and_recovery_artifacts_execute_end_to_end(tmp_path):
    context = run_context_ablation_v3(tmp_path / "context.json", repetitions=1)
    recovery = run_recovery_ablation_v2(tmp_path / "recovery.json", repetitions=1)

    assert context["artifact_type"] == "context-ablation-v3"
    assert context["config_count"] == 12
    assert context["summary"]["avg_budgeted_prompt_chars"] < context["summary"][
        "avg_unbounded_prompt_chars"
    ]
    enabled = recovery["variants"]["resume_enabled"]["summary"]
    disabled = recovery["variants"]["resume_disabled"]["summary"]
    assert recovery["artifact_type"] == "recovery-ablation-v2"
    assert enabled["resume_success_rate"] == 1.0
    assert enabled["resume_false_accept_rate"] == 0.0
    assert disabled["resume_success_rate"] < enabled["resume_success_rate"]
