import hashlib
import json

import pytest

from CodeY.evaluation.long_context_dialogue import (
    DialogueEvent,
    LongContextEvaluationError,
    VARIANTS,
    _tail_events,
    aggregate_results,
    assert_public_payload_safe,
    build_dataset,
    build_probe_prompt,
    parse_probe_response,
    run_probe,
    sanitize_text,
    score_first_attempt,
    score_probe,
)
from CodeY.providers.clients import FakeModelClient
from scripts.run_long_context_dialogue_experiment import (
    _identity_matches,
    _validate_resumed_rows,
)


def _write_fixture(tmp_path):
    codex_root = tmp_path / ".codex"
    source = codex_root / "sessions" / "sample.jsonl"
    source.parent.mkdir(parents=True)
    rows = [
        {
            "type": "session_meta",
            "payload": {
                "id": "11111111-1111-4111-8111-111111111111",
                "cwd": "C:\\Users\\Private\\repo",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "user_message",
                "message": "Keep C:\\Users\\Private\\repo\\main.py unchanged; api_key=top-secret.",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": "private chain of thought",
                "encrypted_content": "encrypted-private-content",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "message": "I will preserve the file.",
            },
        },
        {
            "type": "compacted",
            "payload": {
                "message": "Earlier rule: preserve C:\\Users\\Private\\repo\\main.py.",
                "replacement_history": [
                    {
                        "type": "compaction",
                        "encrypted_content": "must-never-export",
                    }
                ],
            },
        },
        {"type": "event_msg", "payload": {"type": "context_compacted"}},
        {
            "type": "event_msg",
            "payload": {
                "type": "user_message",
                "message": "The old preserve rule is revoked. The current rule is to modify only docs.",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {"input_tokens": 80},
                    "model_context_window": 100,
                },
            },
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_id": "fixture",
                "cases": [
                    {
                        "id": "state-case",
                        "source": "sessions/sample.jsonl",
                        "cutoff_line": 8,
                        "topic": "fixture",
                        "risk": "test-only",
                        "selection_reason": "explicit revision across compaction",
                        "constraints": [
                            {
                                "id": "preserve_file",
                                "text": "Preserve the original file.",
                                "introduced_at": 2,
                                "changed_at": 7,
                                "status": "revoked",
                            },
                            {
                                "id": "modify_docs_only",
                                "text": "Modify only documentation.",
                                "introduced_at": 7,
                                "status": "active",
                                "supersedes": "preserve_file",
                            },
                        ],
                        "probe": {
                            "type": "constraint_state",
                            "question": "Which constraints are active?",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return codex_root, selection, source


def test_sanitizer_removes_paths_credentials_urls_ids_and_code():
    common_token = "ghp_" + "A" * 36
    raw = (
        "C:\\Users\\Private\\repo\\a.py https://private.example/a?q=1 "
        'person@example.com api_key="abc" 11111111-1111-4111-8111-111111111111 '
        "private/module.py notes.docx ```python\nprint('private')\n```"
        " internal /root/desc_001_034 13800138000 192.168.1.10 "
        f'2001:db8::1 {common_token} password="secret phrase" user@host:private/a '
        "6222 0200 0000 0000 000\npassword=correct horse battery staple"
    )
    cleaned = sanitize_text(raw)
    assert "C:\\Users" not in cleaned
    assert "private.example" not in cleaned
    assert "person@example.com" not in cleaned
    assert "api_key=abc" not in cleaned
    assert "11111111-1111" not in cleaned
    assert "private/module.py" not in cleaned
    assert "notes.docx" not in cleaned
    assert "print('private')" not in cleaned
    assert "/root/desc_001_034" not in cleaned
    assert "13800138000" not in cleaned
    assert "192.168.1.10" not in cleaned
    assert "2001:db8::1" not in cleaned
    assert common_token not in cleaned
    assert "secret phrase" not in cleaned
    assert "correct horse battery staple" not in cleaned
    assert "horse battery staple" not in cleaned
    assert "user@host:private/a" not in cleaned
    assert "6222 0200" not in cleaned
    assert "<PATH>" in cleaned
    assert "<URL>" in cleaned
    assert "<EMAIL>" in cleaned
    assert "<SECRET>" in cleaned
    assert "<CODE_BLOCK>" in cleaned
    assert "<INTERNAL_AGENT>" in cleaned
    assert "<PHONE>" in cleaned
    assert "<IP>" in cleaned
    assert "<REMOTE_LOCATION>" in cleaned
    assert "<SENSITIVE_NUMBER>" in cleaned


def test_dataset_builds_five_paired_variants_without_private_payloads(tmp_path):
    codex_root, selection, source = _write_fixture(tmp_path)
    dataset = build_dataset(selection, codex_root)
    assert dataset["variants"] == list(VARIANTS)
    assert len(dataset["cases"]) == 1
    case = dataset["cases"][0]
    assert set(case["contexts"]) == set(VARIANTS)
    assert "Earlier rule" in case["contexts"]["codex_summary"]
    assert "current rule" in case["contexts"]["codex_summary"]
    assert "became revoked" in case["contexts"]["structured_ledger"]
    assert "modify_docs_only" in case["contexts"]["oracle"]
    assert case["source_stats"]["context_compactions"] == 1
    assert case["source_stats"]["max_context_utilization"] == 0.8
    serialized = json.dumps(dataset, ensure_ascii=False)
    assert str(source) not in serialized
    assert "C:\\Users\\Private" not in serialized
    assert "top-secret" not in serialized
    assert "private chain of thought" not in serialized
    assert "must-never-export" not in serialized
    assert case["source_group"] == "conversation_001"


def test_annotation_fields_are_sanitized_before_export(tmp_path):
    codex_root, selection, _ = _write_fixture(tmp_path)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    case = payload["cases"][0]
    case["topic"] = "See C:\\Users\\Private\\topic.txt"
    case["selection_reason"] = "Contact person@example.com with api_key=private"
    case["constraints"][1]["text"] = "Modify notes.docx only; digest " + "a" * 64
    case["probe"]["question"] = "Read private/module.py and list active constraints."
    selection.write_text(json.dumps(payload), encoding="utf-8")

    serialized = json.dumps(build_dataset(selection, codex_root), ensure_ascii=False)
    for secret in (
        "C:\\Users\\Private",
        "person@example.com",
        "api_key=private",
        "notes.docx",
        "a" * 64,
        "private/module.py",
    ):
        assert secret not in serialized


def test_redact_terms_apply_globally_across_cases(tmp_path):
    codex_root, selection, _ = _write_fixture(tmp_path)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    first = payload["cases"][0]
    first["topic"] = "Alice task"
    second = json.loads(json.dumps(first))
    second["id"] = "state-case-two"
    second["redact_terms"] = {"Alice": "<PERSON_01>"}
    payload["cases"].append(second)
    selection.write_text(json.dumps(payload), encoding="utf-8")
    dataset = build_dataset(selection, codex_root)
    assert dataset["cases"][0]["topic"] == "<PERSON_01> task"


def test_annotations_cannot_reference_future_or_hidden_events(tmp_path):
    codex_root, selection, _ = _write_fixture(tmp_path / "attention")
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["cases"][0]["constraints"][1]["introduced_at"] = 9
    selection.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LongContextEvaluationError, match="at or before cutoff_line"):
        build_dataset(selection, codex_root)

    payload["cases"][0]["constraints"][1]["introduced_at"] = 3
    selection.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LongContextEvaluationError, match="visible dialogue event"):
        build_dataset(selection, codex_root)

    codex_root, selection, _ = _write_fixture(tmp_path)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    case = payload["cases"][0]
    case.pop("constraints")
    case["memory_events"] = [{"at": 2, "text": "Remember alpha."}]
    case["oracle_evidence"] = "The answer is alpha."
    case["oracle_evidence_at"] = 9
    case["probe"] = {
        "type": "attention_qa",
        "question": "What should be remembered?",
        "expected": {"required_groups": [["alpha"]], "forbidden_terms": []},
    }
    selection.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LongContextEvaluationError, match="oracle_evidence_at"):
        build_dataset(selection, codex_root)

    codex_root, selection, _ = _write_fixture(tmp_path / "past-eof")
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["cases"][0]["cutoff_line"] = 999
    selection.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LongContextEvaluationError, match="exceeds the source rollout"):
        build_dataset(selection, codex_root)


def test_tail_window_is_a_contiguous_suffix():
    events = [
        DialogueEvent(1, "user", "old-short"),
        DialogueEvent(2, "assistant", "middle-" * 20),
        DialogueEvent(3, "user", "new-short"),
    ]
    tail = _tail_events(events, budget=40)
    assert "new-short" in tail
    assert "middle-" not in tail
    assert "old-short" not in tail
    assert len(tail) <= 40


def test_state_parser_and_score_penalize_stale_constraints(tmp_path):
    codex_root, selection, _ = _write_fixture(tmp_path)
    case = build_dataset(selection, codex_root)["cases"][0]
    parsed = parse_probe_response(
        '{"active_constraints":["preserve_file","modify_docs_only"]}', case
    )
    score = score_probe(case, parsed)
    assert score["correct"] is False
    assert score["recall"] == 1.0
    assert score["stale_selected"] == 1

    unknown = parse_probe_response('{"active_constraints":["modify_docs_onl"]}', case)
    unknown_score = score_probe(case, unknown)
    assert unknown_score["correct"] is True
    assert unknown_score["format_correct"] is False
    assert unknown_score["unknown_selected"] == ["modify_docs_onl"]
    assert unknown_score["semantic_repairs"] == {"modify_docs_onl": "modify_docs_only"}

    unrelated = score_probe(case, {"active_constraints": ["hallucinated_constraint"]})
    assert unrelated["correct"] is False


def test_parser_rejects_prose_extra_fields_duplicates_and_padding(tmp_path):
    codex_root, selection, _ = _write_fixture(tmp_path)
    case = build_dataset(selection, codex_root)["cases"][0]
    invalid = (
        'Here is JSON: {"active_constraints":["modify_docs_only"]}',
        '{"active_constraints":["modify_docs_only"]} trailing',
        '{"active_constraints":["modify_docs_only"],"note":"extra"}',
        '{"active_constraints":["modify_docs_only"],"active_constraints":[]}',
        '{"active_constraints":[" modify_docs_only"]}',
        '{"active_constraints":["modify_docs_only","modify_docs_only"]}',
    )
    for response in invalid:
        with pytest.raises(LongContextEvaluationError):
            parse_probe_response(response, case)


def test_attention_probe_requires_every_alias_group():
    case = {
        "probe": {
            "type": "attention_qa",
            "expected": {
                "required_groups": [["alpha"], ["beta", "b"]],
                "forbidden_terms": ["gamma"],
            },
        }
    }
    assert score_probe(case, {"answer": "alpha and beta"})["correct"] is True
    assert score_probe(case, {"answer": "alpha only"})["correct"] is False
    assert score_probe(case, {"answer": "alpha beta gamma"})["correct"] is False

    numeric = {
        "probe": {
            "type": "attention_qa",
            "expected": {
                "required_groups": [["5"], ["15"]],
                "forbidden_terms": [],
            },
        }
    }
    assert score_probe(numeric, {"answer": "15"})["correct"] is False


def test_probe_prompt_keeps_variant_context_but_not_gold_status(tmp_path):
    codex_root, selection, _ = _write_fixture(tmp_path)
    case = build_dataset(selection, codex_root)["cases"][0]
    prompt = build_probe_prompt(case, "full_dialogue")
    assert "Candidate constraints" in prompt
    assert "modify_docs_only" in prompt
    assert '"active_constraints"' in prompt
    candidate_section = prompt.split("Candidate constraints", 1)[1]
    assert "became revoked" not in candidate_section


def test_aggregation_uses_oracle_as_capability_gate(tmp_path):
    codex_root, selection, _ = _write_fixture(tmp_path)
    dataset = build_dataset(selection, codex_root)
    rows = []
    for variant in VARIANTS:
        correct = variant not in {"codex_summary", "tail_window"}
        rows.append(
            {
                "case_id": "state-case",
                "source_group": dataset["cases"][0]["source_group"],
                "probe_type": "constraint_state",
                "variant": variant,
                "prompt_chars": 100,
                "score": {
                    "correct": correct,
                    "recall": 1.0 if correct else 0.0,
                    "stale_selected": 0,
                    "inactive_total": 1,
                },
            }
        )
    summary = aggregate_results(dataset, rows)
    codex = next(
        row for row in summary["variant_summaries"] if row["variant"] == "codex_summary"
    )
    assert codex["oracle_gated_count"] == 1
    assert codex["accuracy"] == 0.0
    assert summary["attribution"][0]["primary"] == "compression_sensitive"


def test_aggregation_marks_missing_oracle_as_incomplete(tmp_path):
    codex_root, selection, _ = _write_fixture(tmp_path)
    dataset = build_dataset(selection, codex_root)
    rows = [
        {
            "case_id": "state-case",
            "source_group": dataset["cases"][0]["source_group"],
            "probe_type": "constraint_state",
            "variant": variant,
            "prompt_chars": 100,
            "score": {"correct": True, "format_correct": True},
        }
        for variant in VARIANTS
        if variant != "oracle"
    ]
    summary = aggregate_results(dataset, rows)
    attribution = summary["attribution"][0]
    assert attribution["primary"] == "incomplete"
    assert attribution["missing_variants"] == ["oracle"]


def test_ungated_variant_subset_still_reports_raw_metrics(tmp_path):
    codex_root, selection, _ = _write_fixture(tmp_path)
    dataset = build_dataset(selection, codex_root)
    rows = [
        {
            "case_id": "state-case",
            "source_group": dataset["cases"][0]["source_group"],
            "probe_type": "constraint_state",
            "variant": variant,
            "prompt_chars": 100,
            "score": {"correct": variant == "full_dialogue", "format_correct": True},
        }
        for variant in ("full_dialogue", "tail_window")
    ]
    summary = aggregate_results(
        dataset, rows, expected_variants=("full_dialogue", "tail_window")
    )
    by_variant = {row["variant"]: row for row in summary["variant_summaries"]}
    assert by_variant["full_dialogue"]["accuracy"] == 1.0
    assert by_variant["tail_window"]["accuracy"] == 0.0
    assert by_variant["full_dialogue"]["metric_denominator_count"] == 1
    assert by_variant["full_dialogue"]["oracle_gated_count"] == 0
    assert summary["attribution"][0]["primary"] == "ungated"


def test_first_attempt_format_is_separate_from_retry_success(tmp_path):
    codex_root, selection, _ = _write_fixture(tmp_path)
    case = build_dataset(selection, codex_root)["cases"][0]
    client = FakeModelClient(
        ["not JSON", '{"active_constraints":["modify_docs_only"]}']
    )
    parsed, attempts = run_probe(client, "prompt", case, parse_retries=1)
    assert score_probe(case, parsed)["format_correct"] is True
    first = score_first_attempt(case, attempts)
    assert first["correct"] is False
    assert first["format_correct"] is False
    assert first["parse_valid"] is False
    assert len(attempts) == 2


def test_unrecoverable_parse_failure_is_scored_instead_of_dropped(tmp_path):
    codex_root, selection, _ = _write_fixture(tmp_path)
    case = build_dataset(selection, codex_root)["cases"][0]
    parsed, attempts = run_probe(
        FakeModelClient(["not JSON", "still not JSON"]),
        "prompt",
        case,
        parse_retries=1,
    )
    assert parsed is None
    assert len(attempts) == 2
    score = score_probe(case, parsed)
    assert score["correct"] is False
    assert score["format_correct"] is False
    assert score["parse_valid"] is False


def test_resume_identity_and_prompt_hash_are_strict(tmp_path):
    expected = {
        "schema_version": 2,
        "artifact_type": "result",
        "dataset_sha256": "hash",
        "protocol_version": 3,
        "scorer_version": 4,
        "provider": {"provider": "test", "model": "model"},
        "run_config": {},
    }
    assert _identity_matches(dict(expected), expected)
    mismatched = dict(expected)
    mismatched["scorer_version"] = 2
    assert not _identity_matches(mismatched, expected)

    codex_root, selection, _ = _write_fixture(tmp_path)
    case = build_dataset(selection, codex_root)["cases"][0]
    with pytest.raises(LongContextEvaluationError, match="stale prompt hash"):
        _validate_resumed_rows(
            [
                {
                    "case_id": case["id"],
                    "variant": "oracle",
                    "parsed": {"active_constraints": ["modify_docs_only"]},
                    "attempts": [
                        {
                            "parse_valid": True,
                            "parsed": {"active_constraints": ["modify_docs_only"]},
                            "raw_response": (
                                '{"active_constraints":["modify_docs_only"]}'
                            ),
                        }
                    ],
                    "prompt_sha256": "stale",
                }
            ],
            {case["id"]: case},
            VARIANTS,
        )

    with pytest.raises(LongContextEvaluationError, match="unexpected fields"):
        _validate_resumed_rows(
            [
                {
                    "case_id": case["id"],
                    "variant": "oracle",
                    "parsed": {"active_constraints": ["modify_docs_only"]},
                    "attempts": [{"parse_valid": True}],
                    "prompt_sha256": "0" * 64,
                    "notes_sha256": "C:\\Users\\Private\\secret.txt",
                }
            ],
            {case["id"]: case},
            VARIANTS,
        )

    with pytest.raises(LongContextEvaluationError):
        assert_public_payload_safe({"notes_sha256": "C:\\Users\\Private\\secret.txt"})
    with pytest.raises(LongContextEvaluationError):
        assert_public_payload_safe({"C:\\Users\\Private\\secret.txt": "value"})


def test_resume_revalidates_attempts_and_top_level_parsed_output(tmp_path):
    codex_root, selection, _ = _write_fixture(tmp_path)
    case = build_dataset(selection, codex_root)["cases"][0]
    expected_parsed = {"active_constraints": ["modify_docs_only"]}
    alternate_parsed = {"active_constraints": []}
    prompt = build_probe_prompt(case, "oracle")
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def row(parsed, attempt_parsed=expected_parsed, raw=None):
        return {
            "case_id": case["id"],
            "variant": "oracle",
            "parsed": parsed,
            "attempts": [
                {
                    "parse_valid": True,
                    "parsed": attempt_parsed,
                    "raw_response": raw
                    or json.dumps(expected_parsed, separators=(",", ":")),
                }
            ],
            "prompt_sha256": prompt_hash,
        }

    _validate_resumed_rows([row(expected_parsed)], {case["id"]: case}, VARIANTS)

    with pytest.raises(LongContextEvaluationError, match="last valid attempt"):
        _validate_resumed_rows([row(alternate_parsed)], {case["id"]: case}, VARIANTS)

    with pytest.raises(LongContextEvaluationError, match="raw response"):
        _validate_resumed_rows(
            [row(expected_parsed, attempt_parsed=alternate_parsed)],
            {case["id"]: case},
            VARIANTS,
        )

    with pytest.raises(LongContextEvaluationError, match="last valid attempt"):
        _validate_resumed_rows([row({})], {case["id"]: case}, VARIANTS)

    _validate_resumed_rows(
        [
            {
                "case_id": case["id"],
                "variant": "oracle",
                "parsed": None,
                "attempts": [
                    {
                        "parse_valid": False,
                        "raw_response": "not JSON",
                    }
                ],
                "prompt_sha256": prompt_hash,
            }
        ],
        {case["id"]: case},
        VARIANTS,
    )
