import pytest

from CodeY.evaluation import fork_merge_efficiency as benchmark


def _row(parallelism, *, overlap=True):
    is_serial = parallelism == 1
    return {
        "parallelism": parallelism,
        "delay_ms": 1,
        "e2e_ms": 200 if is_serial else 100,
        "fork_merge_ms": 180 if is_serial else 90,
        "provider_calls": 4,
        "max_active": 1 if is_serial or not overlap else 2,
        "status": "merged",
        "validation_status": "passed",
        "fixture_content_digest": "same-content",
        "correct": True,
    }


def test_efficiency_gate_requires_correctness_conserved_work_and_overlap(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "_run_condition",
        lambda parallelism, delay_ms: _row(parallelism),
    )
    monkeypatch.setattr(benchmark, "_source_provenance", lambda: {"test": True})

    artifact = benchmark.run_experiment(
        repetitions=2,
        warmups=0,
        delay_ms=1,
        parallelism=2,
    )

    assert artifact["result"]["correctness_gate"] is True
    assert artifact["result"]["work_conservation_gate"] is True
    assert artifact["result"]["overlap_gate"] is True
    assert artifact["result"]["p05_e2e_speedup"] == 2.0
    assert artifact["result"]["experiment_gate"] is True


def test_one_non_overlapping_parallel_trial_fails_the_experiment_gate(monkeypatch):
    parallel_calls = 0

    def fake_condition(parallelism, delay_ms):
        nonlocal parallel_calls
        del delay_ms
        if parallelism == 2:
            parallel_calls += 1
            return _row(parallelism, overlap=parallel_calls != 1)
        return _row(parallelism)

    monkeypatch.setattr(benchmark, "_run_condition", fake_condition)
    monkeypatch.setattr(benchmark, "_source_provenance", lambda: {"test": True})

    artifact = benchmark.run_experiment(
        repetitions=2,
        warmups=0,
        delay_ms=1,
        parallelism=2,
    )

    assert artifact["result"]["correctness_gate"] is True
    assert artifact["result"]["overlap_gate"] is False
    assert artifact["result"]["efficiency_gain_observed"] is False
    assert artifact["result"]["experiment_gate"] is False


def test_cli_exit_code_uses_the_complete_experiment_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "_run_condition",
        lambda parallelism, delay_ms: _row(parallelism, overlap=False),
    )
    monkeypatch.setattr(benchmark, "_source_provenance", lambda: {"test": True})
    artifact = benchmark.run_experiment(
        repetitions=1,
        warmups=0,
        delay_ms=1,
        parallelism=2,
    )
    monkeypatch.setattr(benchmark, "run_experiment", lambda **kwargs: artifact)

    exit_code = benchmark.main(
        [
            "--repetitions",
            "1",
            "--warmups",
            "0",
            "--delay-ms",
            "1",
            "--parallelism",
            "2",
            "--output-json",
            str(tmp_path / "result.json"),
            "--output-markdown",
            str(tmp_path / "result.md"),
        ]
    )

    assert artifact["result"]["correctness_gate"] is True
    assert artifact["result"]["experiment_gate"] is False
    assert exit_code == 1


def test_two_branch_protocol_rejects_misleading_parallelism():
    with pytest.raises(ValueError, match="parallelism=2"):
        benchmark.run_experiment(
            repetitions=1,
            warmups=0,
            delay_ms=1,
            parallelism=4,
        )
