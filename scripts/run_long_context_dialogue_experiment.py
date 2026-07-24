#!/usr/bin/env python3
"""Build and run the privacy-safe Codex history replay benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CodeY.evaluation.long_context_dialogue import (  # noqa: E402
    PROTOCOL_VERSION,
    RESULTS_SCHEMA_VERSION,
    SCORER_VERSION,
    VARIANTS,
    LongContextEvaluationError,
    aggregate_results,
    build_dataset,
    build_probe_prompt,
    parse_probe_response,
    refresh_dataset_hash,
    run_probe,
    sanitize_text,
    score_probe,
    score_first_attempt,
    shuffled_variants,
    write_dataset,
    write_results,
)
from CodeY.evaluation.real_skill_routing import (  # noqa: E402
    RealSkillEvaluationError,
    build_external_model_client,
)


DEFAULT_CODEX_ROOT = Path.home() / ".codex"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "long-context-dialogue"
DEFAULT_SELECTION = DEFAULT_OUTPUT_ROOT / "selection.local.json"
DEFAULT_DATASET = DEFAULT_OUTPUT_ROOT / "dataset.json"
DEFAULT_RESULTS = DEFAULT_OUTPUT_ROOT / "results.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_ROOT / "results.md"
RESUME_ROW_FIELDS = {
    "attempts",
    "case_id",
    "first_attempt_score",
    "parsed",
    "probe_type",
    "prompt_chars",
    "prompt_sha256",
    "score",
    "source_group",
    "variant",
}


def _csv_values(raw, allowed=None):
    values = tuple(
        dict.fromkeys(item.strip() for item in str(raw).split(",") if item.strip())
    )
    if not values:
        raise argparse.ArgumentTypeError("value list must not be empty")
    if allowed is not None:
        unknown = sorted(set(values) - set(allowed))
        if unknown:
            raise argparse.ArgumentTypeError(
                "unsupported values: " + ", ".join(unknown)
            )
    return values


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build a sanitized dataset from local Codex rollout history and replay "
            "paired long-context variants against the configured external model."
        )
    )
    parser.add_argument("--codex-root", default=str(DEFAULT_CODEX_ROOT))
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION))
    parser.add_argument("--dataset-json", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-json", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output-markdown", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated replay variants.",
    )
    parser.add_argument(
        "--case-ids",
        default="",
        help="Optional comma-separated case IDs. Empty means every selected case.",
    )
    parser.add_argument("--provider", choices=("openai", "anthropic", "deepseek"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--transport-attempts", type=int, default=3)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=15.0,
        help="Delay between successful provider calls to respect rate limits.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=800)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed matching case/variant rows.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build the sanitized dataset without contacting an external model.",
    )
    return parser


def _dataset_summary(dataset):
    return {
        "artifact_type": dataset["artifact_type"],
        "selection_id": dataset["selection_id"],
        "case_count": len(dataset["cases"]),
        "cases": [
            {
                "id": case["id"],
                "source_group": case["source_group"],
                "topic": case["topic"],
                "risk": case["risk"],
                "probe_type": case["probe"]["type"],
                "source_stats": case["source_stats"],
                "context_chars": case["context_chars"],
            }
            for case in dataset["cases"]
        ],
    }


def _run_config(args, variants, case_ids):
    return {
        "case_ids": list(case_ids),
        "delay_seconds": args.delay_seconds,
        "max_output_tokens": args.max_output_tokens,
        "parse_retries": args.parse_retries,
        "seed": args.seed,
        "timeout": args.timeout,
        "transport_attempts": args.transport_attempts,
        "variants": list(variants),
    }


def _identity_matches(existing, expected):
    fields = (
        "schema_version",
        "artifact_type",
        "dataset_sha256",
        "protocol_version",
        "scorer_version",
        "provider",
        "run_config",
    )
    return all(existing.get(field) == expected.get(field) for field in fields)


def _checkpoint(payload, output_json, output_markdown):
    payload["summary"] = aggregate_results(
        payload["_dataset"],
        payload["rows"],
        expected_variants=payload["run_config"]["variants"],
    )
    public_payload = {key: value for key, value in payload.items() if key != "_dataset"}
    write_results(public_payload, output_json, output_markdown)


def _public_provider_profile(profile):
    return {
        key: profile[key]
        for key in ("provider", "model", "client_type")
        if key in profile
    }


def _validate_resumed_rows(rows, case_map, variants):
    seen = set()
    allowed_variants = set(variants)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise LongContextEvaluationError(f"resume row {index} must be an object")
        unexpected = set(row) - RESUME_ROW_FIELDS
        if unexpected:
            raise LongContextEvaluationError(
                f"resume row {index} has unexpected fields: {sorted(unexpected)}"
            )
        case_id = row.get("case_id")
        variant = row.get("variant")
        key = (case_id, variant)
        if case_id not in case_map or variant not in allowed_variants:
            raise LongContextEvaluationError(f"resume row {index} is outside this run")
        if key in seen:
            raise LongContextEvaluationError(
                f"duplicate resume row: {case_id}/{variant}"
            )
        seen.add(key)
        parsed = row.get("parsed")
        if parsed is not None and not isinstance(parsed, dict):
            raise LongContextEvaluationError(
                f"resume row {index} has invalid parsed output"
            )
        attempts = row.get("attempts")
        if (
            not isinstance(attempts, list)
            or not attempts
            or any(not isinstance(attempt, dict) for attempt in attempts)
        ):
            raise LongContextEvaluationError(f"resume row {index} has no attempts")
        valid_attempt_parsed = []
        for attempt_index, attempt in enumerate(attempts):
            parse_valid = attempt.get("parse_valid")
            if parse_valid is True:
                raw_response = attempt.get("raw_response")
                attempt_parsed = attempt.get("parsed")
                if not isinstance(raw_response, str) or not isinstance(
                    attempt_parsed, dict
                ):
                    raise LongContextEvaluationError(
                        f"resume row {index} attempt {attempt_index} has incomplete valid output"
                    )
                try:
                    reparsed = parse_probe_response(raw_response, case_map[case_id])
                except Exception as exc:
                    raise LongContextEvaluationError(
                        f"resume row {index} attempt {attempt_index} cannot be re-parsed"
                    ) from exc
                if reparsed != attempt_parsed:
                    raise LongContextEvaluationError(
                        f"resume row {index} attempt {attempt_index} parsed output does not match raw response"
                    )
                valid_attempt_parsed.append(attempt_parsed)
            elif parse_valid is False:
                if "parsed" in attempt and attempt.get("parsed") is not None:
                    raise LongContextEvaluationError(
                        f"resume row {index} attempt {attempt_index} has parsed output despite parse failure"
                    )
            else:
                raise LongContextEvaluationError(
                    f"resume row {index} attempt {attempt_index} must declare parse_valid"
                )
        if parsed is None:
            if valid_attempt_parsed or attempts[-1].get("parse_valid") is not False:
                raise LongContextEvaluationError(
                    f"resume row {index} has an unexplained parse failure"
                )
        elif not valid_attempt_parsed or parsed != valid_attempt_parsed[-1]:
            raise LongContextEvaluationError(
                f"resume row {index} parsed output does not match the last valid attempt"
            )
        prompt = build_probe_prompt(case_map[case_id], variant)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if row.get("prompt_sha256") != prompt_hash:
            raise LongContextEvaluationError(
                f"resume row {case_id}/{variant} has a stale prompt hash"
            )


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    try:
        variants = _csv_values(args.variants, VARIANTS)
        if (
            args.timeout < 1
            or args.transport_attempts < 1
            or args.max_output_tokens < 1
        ):
            raise LongContextEvaluationError(
                "timeouts, attempts, and token limits must be positive"
            )
        if args.parse_retries < 0 or args.delay_seconds < 0:
            raise LongContextEvaluationError(
                "parse retries and delay must be non-negative"
            )
        dataset = build_dataset(args.selection, args.codex_root)
        requested_case_ids = (
            _csv_values(args.case_ids)
            if str(args.case_ids).strip()
            else tuple(case["id"] for case in dataset["cases"])
        )
        known_case_ids = {case["id"] for case in dataset["cases"]}
        unknown_case_ids = sorted(set(requested_case_ids) - known_case_ids)
        if unknown_case_ids:
            raise LongContextEvaluationError(
                "unknown case IDs: " + ", ".join(unknown_case_ids)
            )
        dataset["cases"] = [
            case for case in dataset["cases"] if case["id"] in requested_case_ids
        ]
        refresh_dataset_hash(dataset)
        write_dataset(dataset, args.dataset_json)
    except (LongContextEvaluationError, OSError) as exc:
        print(f"long-context dataset error: {exc}", file=sys.stderr)
        return 1

    print(
        f"built sanitized dataset cases={len(dataset['cases'])} "
        f"sha256={dataset['dataset_sha256']} output={args.dataset_json}"
    )
    if args.build_only:
        return 0

    try:
        client, private_provider = build_external_model_client(
            ROOT,
            provider=args.provider,
            timeout=args.timeout,
            transport_attempts=args.transport_attempts,
        )
    except RealSkillEvaluationError as exc:
        print(f"long-context provider error: {exc}", file=sys.stderr)
        return 1
    provider = _public_provider_profile(private_provider)

    expected = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "artifact_type": "codex-long-context-dialogue-results-v1",
        "protocol_version": PROTOCOL_VERSION,
        "scorer_version": SCORER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset["dataset_sha256"],
        "provider": provider,
        "run_config": _run_config(args, variants, requested_case_ids),
        "dataset": _dataset_summary(dataset),
        "rows": [],
        "summary": {"variant_summaries": [], "attribution": []},
    }
    output_json = Path(args.output_json)
    payload = {**expected, "_dataset": dataset}
    if args.resume and output_json.is_file():
        try:
            existing = json.loads(output_json.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            print(f"cannot resume result artifact: {exc}", file=sys.stderr)
            return 1
        if not _identity_matches(existing, expected):
            print(
                "cannot resume: existing artifact identity does not match",
                file=sys.stderr,
            )
            return 1
        payload = {
            **expected,
            "created_at": existing.get("created_at", expected["created_at"]),
            "rows": existing.get("rows", []),
            "_dataset": dataset,
        }

    case_map = {case["id"]: case for case in dataset["cases"]}
    try:
        _validate_resumed_rows(payload.get("rows", []), case_map, variants)
        for row in payload.get("rows", []):
            case = case_map[row["case_id"]]
            row["score"] = score_probe(case, row["parsed"])
            row["first_attempt_score"] = score_first_attempt(
                case, row.get("attempts") or []
            )
    except LongContextEvaluationError as exc:
        print(f"cannot resume result rows: {exc}", file=sys.stderr)
        return 1

    completed = {(row["case_id"], row["variant"]) for row in payload.get("rows", [])}
    try:
        for case_index, case_id in enumerate(requested_case_ids):
            case = case_map[case_id]
            ordered = [
                variant
                for variant in shuffled_variants(args.seed + case_index)
                if variant in variants
            ]
            for variant in ordered:
                if (case_id, variant) in completed:
                    print(f"resuming: skip case={case_id} variant={variant}")
                    continue
                prompt = build_probe_prompt(case, variant)
                print(
                    f"running provider={provider['provider']} model={provider['model']} "
                    f"case={case_id} variant={variant} prompt_chars={len(prompt)}"
                )
                parsed, attempts = run_probe(
                    client,
                    prompt,
                    case,
                    max_output_tokens=args.max_output_tokens,
                    parse_retries=args.parse_retries,
                )
                payload["rows"].append(
                    {
                        "case_id": case_id,
                        "source_group": case["source_group"],
                        "probe_type": case["probe"]["type"],
                        "variant": variant,
                        "prompt_chars": len(prompt),
                        "prompt_sha256": hashlib.sha256(
                            prompt.encode("utf-8")
                        ).hexdigest(),
                        "parsed": parsed,
                        "score": score_probe(case, parsed),
                        "first_attempt_score": score_first_attempt(case, attempts),
                        "attempts": attempts,
                    }
                )
                _checkpoint(payload, args.output_json, args.output_markdown)
                if args.delay_seconds:
                    time.sleep(args.delay_seconds)
    except (LongContextEvaluationError, RuntimeError, OSError) as exc:
        payload["failure"] = {
            "case_id": case_id,
            "variant": variant,
            "error": sanitize_text(str(exc)),
        }
        _checkpoint(payload, args.output_json, args.output_markdown)
        print(f"long-context evaluation failed: {exc}", file=sys.stderr)
        return 1

    payload.pop("failure", None)
    _checkpoint(payload, args.output_json, args.output_markdown)
    print(f"wrote results: {args.output_json}")
    print(f"wrote report: {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
