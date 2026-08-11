import argparse
from copy import deepcopy
import json
from pathlib import Path
import urllib.error

import pytest

from CodeY.evaluation.real_skill_routing import (
    build_selection_prompt,
    cases_for_scale,
    load_benchmark_dataset,
    materialize_skill_documents,
    merge_artifact_axes,
    parse_predictions,
    render_flat_catalog,
    render_structured_catalog,
    score_predictions,
    validate_skill_descriptions,
)
from CodeY.providers.clients import (
    AnthropicCompatibleModelClient,
    OpenAICompatibleModelClient,
    _http_retry_delay,
    _is_retryable_http_error,
)
from CodeY.skills.router import SkillRouter
from scripts.run_real_skill_routing_experiment import (
    EXPERIMENT_PROTOCOL_VERSION,
    artifact_identity_matches,
    build_run_config,
)


ROOT = Path(__file__).parents[1]
SKILLS = ROOT / "benchmarks" / "real-skill-routing" / "skills.json"
REQUESTS = ROOT / "benchmarks" / "real-skill-routing" / "requests.json"


def load_dataset():
    return load_benchmark_dataset(SKILLS, REQUESTS)


def test_resumed_scale_slices_preserve_complete_artifact_axes():
    payload = {
        "skill_scales": [5, 15, 25, 50],
        "modes": ["flat_full", "structured_index"],
    }
    merge_artifact_axes(payload, (100,), ("structured_index",))
    assert payload["skill_scales"] == [5, 15, 25, 50, 100]
    assert payload["modes"] == ["flat_full", "structured_index"]


def test_resume_identity_includes_provider_and_model_call_configuration():
    args = argparse.Namespace(
        batch_size=25,
        delay_seconds=3.0,
        max_output_tokens=8192,
        parse_retries=1,
        timeout=300,
        transport_attempts=1,
    )
    expected = {
        "artifact_type": "real-model-skill-routing-v1",
        "dataset_sha256": "dataset-hash",
        "experiment_protocol_version": EXPERIMENT_PROTOCOL_VERSION,
        "provider": {
            "base_url": "https://example.test/v1",
            "client_type": "anthropic",
            "model": "model-v1",
            "provider": "anthropic",
        },
        "repetitions": 1,
        "run_config": build_run_config(args),
        "seed": 20260712,
    }
    assert artifact_identity_matches(deepcopy(expected), expected)

    changed_batch = deepcopy(expected)
    changed_batch["run_config"]["batch_size"] = 10
    assert not artifact_identity_matches(changed_batch, expected)

    changed_backend = deepcopy(expected)
    changed_backend["provider"]["base_url"] = "https://other.test/v1"
    assert not artifact_identity_matches(changed_backend, expected)

    legacy_artifact = deepcopy(expected)
    legacy_artifact.pop("experiment_protocol_version")
    assert not artifact_identity_matches(legacy_artifact, expected)


def test_real_dataset_covers_required_scales_and_request_types():
    dataset = load_dataset()
    assert len(dataset.skills) == 100
    assert dataset.scales == (5, 15, 25, 50, 100)
    assert [len(cases_for_scale(dataset, scale)) for scale in dataset.scales] == [
        12,
        24,
        35,
        61,
        115,
    ]
    assert sum(case.anchor for case in dataset.requests) == 5
    assert {case.kind for case in dataset.requests} == {"single", "multi", "negative"}


def test_skill_descriptions_follow_seven_activation_rules():
    dataset = load_dataset()
    phrases = validate_skill_descriptions(dataset.skills)
    assert len(phrases) == 200
    assert all(
        skill.description.startswith("This skill should be used when")
        for skill in dataset.skills
    )


def test_materialized_documents_are_valid_project_skills(tmp_path):
    dataset = load_dataset()
    skills_root = materialize_skill_documents(dataset, tmp_path / "skills")
    router = SkillRouter(tmp_path)
    assert len(router.skills) == 100
    assert {skill.name for skill in router.skills} == {skill.id for skill in dataset.skills}
    for path in skills_root.glob("*/SKILL.md"):
        frontmatter = path.read_text(encoding="utf-8").split("---", 2)[1]
        keys = {
            line.split(":", 1)[0].strip()
            for line in frontmatter.splitlines()
            if line.strip()
        }
        assert keys == {"name", "description"}


def test_structured_index_is_smaller_but_preserves_every_skill_id():
    dataset = load_dataset()
    flat = render_flat_catalog(dataset, 100)
    structured = render_structured_catalog(dataset, 100)
    assert len(structured) < len(flat)
    for skill in dataset.skills:
        assert skill.id in flat
        assert skill.id in structured


def test_prompt_uses_identical_cases_and_strict_json_contract():
    dataset = load_dataset()
    cases = cases_for_scale(dataset, 5)
    flat = build_selection_prompt(dataset, 5, "flat_full", cases)
    structured = build_selection_prompt(dataset, 5, "structured_index", cases)
    assert '"predictions"' in flat
    assert '"predictions"' in structured
    for case in cases:
        assert case.id in flat
        assert case.id in structured


def test_prediction_parser_accepts_fenced_json_and_scores_multilabel_sets():
    dataset = load_dataset()
    cases = cases_for_scale(dataset, 5)
    payload = {
        "predictions": [
            {"id": case.id, "skills": list(case.expected_skills)} for case in cases
        ]
    }
    predictions = parse_predictions(
        "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```", cases
    )
    score = score_predictions(
        cases, predictions, [skill.id for skill in dataset.skills[:5]]
    )
    assert score["exact_match"]["value"] == 1.0
    assert score["anchor_exact_match"]["value"] == 1.0
    assert score["micro_f1"] == 1.0


def test_unknown_skill_is_retained_and_penalized():
    dataset = load_dataset()
    case = cases_for_scale(dataset, 5)[0]
    predictions = {case.id: (*case.expected_skills, "hallucinated-skill")}
    score = score_predictions(
        (case,), predictions, [skill.id for skill in dataset.skills[:5]]
    )
    assert score["exact_match"]["value"] == 0.0
    assert score["false_activations"] == 1
    assert score["rows"][0]["unknown_skills"] == ["hallucinated-skill"]


def test_rate_limit_retry_uses_retry_after_or_bounded_backoff():
    explicit = urllib.error.HTTPError(
        "https://example.test", 429, "rate limited", {"Retry-After": "7"}, None
    )
    fallback = urllib.error.HTTPError(
        "https://example.test", 429, "rate limited", {}, None
    )
    origin_timeout = urllib.error.HTTPError(
        "https://example.test", 524, "origin timeout", {}, None
    )
    transient_not_found = urllib.error.HTTPError(
        "https://example.test", 404, "not found", {}, None
    )
    body_backoff = urllib.error.HTTPError(
        "https://example.test", 502, "bad gateway", {}, None
    )
    assert _http_retry_delay(explicit, 0) == 7.0
    assert _http_retry_delay(fallback, 0) == 10.0
    assert _http_retry_delay(fallback, 10) == 60.0
    assert _http_retry_delay(origin_timeout, 0) == 120.0
    assert _http_retry_delay(transient_not_found, 0) == 180.0
    assert _http_retry_delay(body_backoff, 0, '{"retry_after":60}') == 60.0


def test_only_bare_proxy_404_is_retried():
    not_found = urllib.error.HTTPError(
        "https://example.test", 404, "not found", {}, None
    )
    assert _is_retryable_http_error(not_found, "404 page not found") is True
    assert _is_retryable_http_error(not_found, '{"error":"model not found"}') is False


def test_openai_compatible_timeout_is_wrapped(monkeypatch):
    client = OpenAICompatibleModelClient(
        model="test-model",
        base_url="https://example.test/v1",
        api_key="test-key",
        temperature=0.0,
        timeout=1,
        attempts=1,
    )

    def raise_timeout(*args, **kwargs):
        del args, kwargs
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", raise_timeout)
    with pytest.raises(RuntimeError, match="Could not reach"):
        client.complete("prompt", max_new_tokens=16)


def test_anthropic_compatible_uses_claude_cli_user_agent(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

        def read(self):
            return json.dumps(
                {
                    "content": [{"type": "text", "text": "OK"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ).encode("utf-8")

    def capture_request(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", capture_request)
    client = AnthropicCompatibleModelClient(
        model="claude-opus-5[1M]",
        base_url="https://example.test/v1",
        api_key="test-key",
        temperature=0.0,
        timeout=3,
        attempts=1,
    )

    completion = client.complete("prompt", max_new_tokens=16)

    assert completion.text == "OK"
    assert captured["request"].get_header("User-agent") == (
        "claude-cli/2.1.218 (external, cli)"
    )
    assert captured["timeout"] == 3
