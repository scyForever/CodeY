#!/usr/bin/env python3
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CodeY.evaluation.real_skill_routing import (  # noqa: E402
    DEFAULT_SCALES,
    MODES,
    RealSkillEvaluationError,
    build_external_model_client,
    build_selection_prompt,
    cases_for_scale,
    dataset_fingerprint,
    load_benchmark_dataset,
    materialize_skill_documents,
    merge_artifact_axes,
    run_model_batch,
    score_predictions,
    write_artifacts,
)


DEFAULT_SKILLS = ROOT / "benchmarks" / "real-skill-routing" / "skills.json"
DEFAULT_REQUESTS = ROOT / "benchmarks" / "real-skill-routing" / "requests.json"
DEFAULT_JSON = ROOT / "artifacts" / "real-skill-routing" / "results.json"
DEFAULT_MARKDOWN = ROOT / "artifacts" / "real-skill-routing" / "results.md"
DEFAULT_MATERIALIZED_SKILLS = ROOT / "artifacts" / "real-skill-routing" / "skills"
EXPERIMENT_PROTOCOL_VERSION = 2


def _csv_values(raw, cast=str):
    try:
        values = tuple(cast(item.strip()) for item in str(raw).split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values:
        raise argparse.ArgumentTypeError("value list must not be empty")
    return values


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate flat full Skill documents against a structured compact Skill index "
            "with the external model configured in .env."
        )
    )
    parser.add_argument("--skills", default=str(DEFAULT_SKILLS))
    parser.add_argument("--requests", default=str(DEFAULT_REQUESTS))
    parser.add_argument(
        "--scales",
        default=",".join(str(value) for value in DEFAULT_SCALES),
        help="Comma-separated Skill counts. Supported: 5,15,25,50,100.",
    )
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help="Comma-separated modes: flat_full,structured_index.",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--provider", choices=("openai", "anthropic", "deepseek"))
    parser.add_argument("--model", default=None, help="Explicit model override for the selected provider.")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--transport-attempts", type=int, default=1)
    parser.add_argument("--delay-seconds", type=float, default=3.0)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Maximum requests classified in one external-model call.",
    )
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-markdown", default=str(DEFAULT_MARKDOWN))
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse matching completed runs from output-json.",
    )
    parser.add_argument(
        "--materialized-skills",
        default=str(DEFAULT_MATERIALIZED_SKILLS),
        help="Directory where the 100 benchmark SKILL.md files are materialized.",
    )
    return parser


def _comparison_by_scale(runs):
    comparisons = []
    keys = {
        (run["scale"], run["repetition"]): run
        for run in runs
        if run["mode"] == "flat_full"
    }
    for structured in runs:
        if structured["mode"] != "structured_index":
            continue
        flat = keys.get((structured["scale"], structured["repetition"]))
        if flat is None:
            continue
        comparisons.append(
            {
                "scale": structured["scale"],
                "repetition": structured["repetition"],
                "structured_minus_flat": {
                    "exact_match": structured["score"]["exact_match"]["value"]
                    - flat["score"]["exact_match"]["value"],
                    "anchor_exact_match": structured["score"]["anchor_exact_match"]["value"]
                    - flat["score"]["anchor_exact_match"]["value"],
                    "micro_f1": structured["score"]["micro_f1"]
                    - flat["score"]["micro_f1"],
                    "prompt_chars": structured["prompt_chars"] - flat["prompt_chars"],
                    "elapsed_seconds": structured["elapsed_seconds"]
                    - flat["elapsed_seconds"],
                },
            }
        )
    return comparisons


def build_run_config(args):
    return {
        "batch_size": args.batch_size,
        "delay_seconds": args.delay_seconds,
        "max_output_tokens": args.max_output_tokens,
        "parse_retries": args.parse_retries,
        "timeout": args.timeout,
        "transport_attempts": args.transport_attempts,
    }


def artifact_identity_matches(existing, expected):
    identity_fields = (
        "artifact_type",
        "dataset_sha256",
        "experiment_protocol_version",
        "provider",
        "repetitions",
        "run_config",
        "seed",
    )
    return all(existing.get(field) == expected.get(field) for field in identity_fields)


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    try:
        scales = _csv_values(args.scales, int)
        modes = _csv_values(args.modes)
        if any(scale not in DEFAULT_SCALES for scale in scales):
            raise RealSkillEvaluationError(
                f"scales must be selected from {list(DEFAULT_SCALES)}"
            )
        if any(mode not in MODES for mode in modes):
            raise RealSkillEvaluationError(f"modes must be selected from {list(MODES)}")
        if args.repetitions < 1:
            raise RealSkillEvaluationError("repetitions must be positive")
        if (
            args.timeout < 1
            or args.transport_attempts < 1
            or args.delay_seconds < 0
            or args.max_output_tokens < 1
            or args.batch_size < 1
            or args.parse_retries < 0
        ):
            raise RealSkillEvaluationError(
                "timeouts/attempts/tokens must be positive and delays/retries non-negative"
            )
        dataset = load_benchmark_dataset(args.skills, args.requests)
        materialize_skill_documents(dataset, args.materialized_skills)
        client, provider = build_external_model_client(
            ROOT,
            provider=args.provider,
            model_override=args.model,
            timeout=args.timeout,
            transport_attempts=args.transport_attempts,
        )
    except RealSkillEvaluationError as exc:
        print(f"real-Skill evaluation error: {exc}", file=sys.stderr)
        return 1

    new_payload = {
        "schema_version": 1,
        "artifact_type": "real-model-skill-routing-v1",
        "experiment_protocol_version": EXPERIMENT_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "dataset_sha256": dataset_fingerprint(args.skills, args.requests),
        "skill_scales": list(scales),
        "modes": list(modes),
        "repetitions": args.repetitions,
        "seed": args.seed,
        "run_config": build_run_config(args),
        "runs": [],
        "comparisons": [],
    }

    output_json = Path(args.output_json)
    output_markdown = Path(args.output_markdown)
    payload = new_payload
    if args.resume and output_json.is_file():
        try:
            existing = json.loads(output_json.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            print(f"cannot resume result artifact: {exc}", file=sys.stderr)
            return 1
        if not artifact_identity_matches(existing, new_payload):
            print(
                "cannot resume: existing artifact identity does not match this run",
                file=sys.stderr,
            )
            return 1
        payload = existing
        merge_artifact_axes(payload, scales, modes)
        payload.pop("failure", None)
    completed = {
        (run["scale"], run["mode"], run["repetition"])
        for run in payload.get("runs", [])
    }
    for repetition in range(1, args.repetitions + 1):
        for scale in scales:
            cases = cases_for_scale(dataset, scale)
            mode_order = list(modes)
            if (repetition + scale) % 2:
                mode_order.reverse()
            for mode in mode_order:
                run_key = (scale, mode, repetition)
                if run_key in completed:
                    print(
                        f"resuming: skip completed skills={scale} mode={mode} "
                        f"repetition={repetition}"
                    )
                    continue
                print(
                    f"running provider={provider['provider']} model={provider['model']} "
                    f"skills={scale} mode={mode} cases={len(cases)}"
                )
                predictions = {}
                attempts = []
                prompt_chars = 0
                max_prompt_chars = 0
                batches = [
                    cases[index : index + args.batch_size]
                    for index in range(0, len(cases), args.batch_size)
                ]
                current_batch_index = 0
                try:
                    for batch_index, batch_cases in enumerate(batches, start=1):
                        current_batch_index = batch_index
                        prompt = build_selection_prompt(
                            dataset,
                            scale,
                            mode,
                            batch_cases,
                            seed=args.seed + repetition + batch_index - 2,
                        )
                        prompt_chars += len(prompt)
                        max_prompt_chars = max(max_prompt_chars, len(prompt))
                        batch_predictions, batch_attempts = run_model_batch(
                            client,
                            prompt,
                            batch_cases,
                            max_output_tokens=args.max_output_tokens,
                            parse_retries=args.parse_retries,
                        )
                        predictions.update(batch_predictions)
                        for attempt in batch_attempts:
                            attempt["batch"] = batch_index
                            attempt["batch_case_ids"] = [
                                case.id for case in batch_cases
                            ]
                        attempts.extend(batch_attempts)
                        print(
                            f"batch {batch_index}/{len(batches)} complete "
                            f"skills={scale} mode={mode}"
                        )
                        if args.delay_seconds and batch_index < len(batches):
                            time.sleep(args.delay_seconds)
                except (RealSkillEvaluationError, RuntimeError, OSError) as exc:
                    payload["failure"] = {
                        "scale": scale,
                        "mode": mode,
                        "repetition": repetition,
                        "batch": current_batch_index,
                        "error": str(exc),
                    }
                    payload["comparisons"] = _comparison_by_scale(payload["runs"])
                    write_artifacts(payload, output_json, output_markdown)
                    print(f"real-Skill evaluation failed: {exc}", file=sys.stderr)
                    return 1
                score = score_predictions(
                    cases,
                    predictions,
                    [skill.id for skill in dataset.skills[:scale]],
                )
                run = {
                    "scale": scale,
                    "mode": mode,
                    "repetition": repetition,
                    "max_output_tokens": args.max_output_tokens,
                    "batch_size": args.batch_size,
                    "batch_count": len(batches),
                    "case_ids": [case.id for case in cases],
                    "prompt_chars": prompt_chars,
                    "max_prompt_chars": max_prompt_chars,
                    "elapsed_seconds": sum(
                        attempt["elapsed_seconds"] for attempt in attempts
                    ),
                    "attempts": attempts,
                    "score": score,
                }
                payload["runs"].append(run)
                completed.add(run_key)
                payload["comparisons"] = _comparison_by_scale(payload["runs"])
                write_artifacts(payload, output_json, output_markdown)
                print(
                    "result exact={:.2%} anchor={:.2%} micro_f1={:.2%}".format(
                        score["exact_match"]["value"],
                        score["anchor_exact_match"]["value"],
                        score["micro_f1"],
                    )
                )
                if args.delay_seconds:
                    time.sleep(args.delay_seconds)
    payload["comparisons"] = _comparison_by_scale(payload["runs"])
    write_artifacts(payload, output_json, output_markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
