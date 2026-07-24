"""Command-line workflow for repository rule governance."""

from __future__ import annotations

import argparse
import json
import sys

from ..context.workspace import WorkspaceContext
from .discovery import RuleScanner
from .patches import RulePatchStore
from .runners import ExternalAgentRunner, TrialRequest


def build_rules_parser():
    parser = argparse.ArgumentParser(
        prog="codey rules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Scan, review, trial, and apply repository coding-agent rule patches.",
    )
    parser.add_argument("--cwd", default=".", help="Repository directory.")
    commands = parser.add_subparsers(dest="rules_command", required=True)

    agents = commands.add_parser("agents", help="Probe local CodeY, Codex, and Claude runners.")
    agents.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    scan = commands.add_parser("scan", help="Discover distributed repository rule sources.")
    scan.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    scan.add_argument(
        "--include-content",
        action="store_true",
        help="Include untrusted rule text in JSON output.",
    )

    plan = commands.add_parser("plan", help="Create a human-review-required rule patch.")
    plan.add_argument(
        "--target",
        action="append",
        choices=("codey", "codex", "claude", "cursor"),
        default=[],
        help="Target adapter; repeat to select several. Defaults to all.",
    )
    plan.add_argument(
        "--source",
        action="append",
        default=None,
        help="Repository-relative source path; repeat to select a subset.",
    )
    plan.add_argument(
        "--objective",
        default="Unify repository coding-agent rules without changing their declared scope.",
        help="Review objective stored with the patch.",
    )
    plan.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    status = commands.add_parser("status", help="Show rule patches and isolated observations.")
    status.add_argument("patch_id", nargs="?", help="Optional rule patch id.")
    status.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    diff = commands.add_parser("diff", help="Print the proposed adapter diffs for review.")
    diff.add_argument("patch_id", help="Rule patch id.")
    diff.add_argument(
        "--target",
        choices=("codey", "codex", "claude", "cursor"),
        default=None,
        help="Only print one target adapter.",
    )

    trial = commands.add_parser("trial", help="Run an inspect-only isolated rule trial.")
    _add_trial_arguments(trial, runners=("codey", "codex", "claude"), mode="inspect")

    delegate = commands.add_parser(
        "delegate",
        help="Ask local Codex or Claude to edit an isolated worktree and return a diff.",
    )
    _add_trial_arguments(delegate, runners=("codex", "claude"), mode="edit")

    apply = commands.add_parser("apply", help="Apply an exact reviewed rule patch.")
    apply.add_argument("patch_id", help="Rule patch id.")
    apply.add_argument(
        "--approve",
        action="store_true",
        help="Confirm human review and exact application.",
    )
    apply.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    rollback = commands.add_parser("rollback", help="Restore the pre-patch rule files.")
    rollback.add_argument("patch_id", help="Rule patch id.")
    rollback.add_argument(
        "--approve",
        action="store_true",
        help="Confirm exact rollback of the active patch.",
    )
    rollback.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def _add_trial_arguments(parser, runners, mode):
    parser.set_defaults(trial_mode=mode)
    parser.add_argument("patch_id", help="Rule patch id.")
    parser.add_argument("--runner", required=True, choices=runners, help="Local agent adapter.")
    parser.add_argument(
        "--variant",
        choices=("baseline", "candidate", "canary"),
        default="candidate",
        help="Rule exposure variant.",
    )
    parser.add_argument("--cohort-key", default="", help="Stable key required for canary assignment.")
    parser.add_argument(
        "--candidate-percent",
        type=float,
        default=20.0,
        help="Candidate share for deterministic canary assignment.",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Runner timeout in seconds.")
    parser.add_argument("--max-changed-files", type=int, default=12)
    parser.add_argument("--max-diff-lines", type=int, default=1200)
    parser.add_argument("--max-diff-bytes", type=int, default=256 * 1024)
    parser.add_argument("--max-output-bytes", type=int, default=256 * 1024)
    parser.add_argument(
        "--allow-dirty-base",
        action="store_true",
        help="Use the recorded committed revision even though planning saw uncommitted files.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("task", nargs="+", help="Task sent to the isolated agent.")


def main(argv=None):
    args = build_rules_parser().parse_args(argv)
    try:
        workspace = WorkspaceContext.build(args.cwd)
        root = workspace.repo_root
        store = RulePatchStore(root)
        runner = ExternalAgentRunner(root, patch_store=store)
        command = args.rules_command

        if command == "agents":
            payload = runner.probe_all()
            return _emit(payload, args.json, _render_agents)
        if command == "scan":
            inventory = RuleScanner(root).scan()
            payload = inventory.to_dict(include_content=args.include_content)
            return _emit(payload, args.json, _render_inventory)
        if command == "plan":
            inventory = RuleScanner(root).scan()
            targets = args.target or ("codey", "codex", "claude", "cursor")
            patch = store.create_plan(
                inventory,
                targets=targets,
                objective=args.objective,
                source_paths=args.source,
            )
            summary = _patch_summary(patch, trials=store.list_trials(patch["id"]))
            return _emit(summary, args.json, _render_patch)
        if command == "status":
            if args.patch_id:
                patch = store.load(args.patch_id)
                payload = _patch_summary(patch, trials=store.list_trials(args.patch_id))
                return _emit(payload, args.json, _render_patch)
            payload = [
                _patch_summary(patch, trials=store.list_trials(patch["id"]))
                for patch in store.list_patches()
            ]
            return _emit(payload, args.json, _render_patch_list)
        if command == "diff":
            patch = store.load(args.patch_id)
            artifacts = [
                artifact
                for artifact in patch["artifacts"]
                if args.target is None or artifact["target"] == args.target
            ]
            if not artifacts:
                raise ValueError("patch has no matching target artifact")
            for index, artifact in enumerate(artifacts):
                if index:
                    print()
                print(artifact["diff"].rstrip())
            return 0
        if command in {"trial", "delegate"}:
            request = TrialRequest(
                patch_id=args.patch_id,
                runner=args.runner,
                task=" ".join(args.task).strip(),
                variant=args.variant,
                mode=args.trial_mode,
                cohort_key=args.cohort_key,
                candidate_fraction=float(args.candidate_percent) / 100.0,
                timeout_seconds=args.timeout,
                max_changed_files=args.max_changed_files,
                max_diff_lines=args.max_diff_lines,
                max_diff_bytes=args.max_diff_bytes,
                max_output_bytes=args.max_output_bytes,
                allow_dirty_base=args.allow_dirty_base,
            )
            result = runner.run(request)
            _emit(result, args.json, _render_trial)
            return 0 if result["status"] == "completed" else 1
        if command == "apply":
            patch = store.apply(args.patch_id, approved=args.approve)
            return _emit(
                _patch_summary(patch, trials=store.list_trials(patch["id"])),
                args.json,
                _render_patch,
            )
        if command == "rollback":
            patch = store.rollback(args.patch_id, approved=args.approve)
            return _emit(
                _patch_summary(patch, trials=store.list_trials(patch["id"])),
                args.json,
                _render_patch,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"codey rules: {exc}", file=sys.stderr)
        return 2
    return 0


def _patch_summary(patch, trials):
    return {
        "id": patch["id"],
        "schema_version": patch["schema_version"],
        "status": patch["status"],
        "objective": patch["objective"],
        "created_at": patch["created_at"],
        "updated_at": patch["updated_at"],
        "base_revision": patch["repository"]["base_revision"],
        "dirty_at_plan": patch["repository"]["dirty_at_plan"],
        "git_status_at_plan": patch["repository"]["git_status_at_plan"],
        "dirty_paths_at_plan": patch["repository"]["dirty_paths_at_plan"],
        "inventory_id": patch["inventory_id"],
        "source_count": len(patch["source_refs"]),
        "source_refs": [
            {
                "source_id": source["source_id"],
                "path": source["path"],
                "ecosystem": source["ecosystem"],
                "kind": source["kind"],
                "scope": source["scope"],
                "precedence": source["precedence"],
                "sha256": source["sha256"],
                "trust": source["trust"],
            }
            for source in patch["source_refs"]
        ],
        "inventory_issues": patch["inventory_issues"],
        "targets": [
            {
                "target": artifact["target"],
                "path": artifact["path"],
                "before_sha256": artifact["before_sha256"],
                "candidate_sha256": artifact["candidate_sha256"],
                "diff_lines": len(artifact["diff"].splitlines()),
            }
            for artifact in patch["artifacts"]
        ],
        "trial_count": len(trials),
        "trials": [
            {
                "id": trial.get("id"),
                "runner": trial.get("runner"),
                "mode": trial.get("mode"),
                "requested_variant": trial.get("requested_variant"),
                "selected_variant": trial.get("selected_variant"),
                "status": trial.get("status"),
                "base_revision": trial.get("base_revision"),
                "budget_violations": trial.get("budget_violations", []),
                "diff_stats": trial.get("diff_stats", {}),
                "artifact_dir": trial.get("artifact_dir"),
            }
            for trial in trials
        ],
    }


def _emit(payload, as_json, renderer):
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(renderer(payload))
    return 0


def _render_agents(items):
    lines = []
    for item in items:
        state = "available" if item["available"] else "missing"
        version = f" ({item['version']})" if item["version"] else ""
        lines.append(f"{item['runner']}: {state}{version}")
    return "\n".join(lines)


def _render_inventory(payload):
    lines = [
        f"inventory: {payload['inventory_id']}",
        f"revision: {payload['revision']}",
        f"sources: {len(payload['sources'])}",
        f"issues: {len(payload['issues'])}",
    ]
    for source in payload["sources"]:
        lines.append(
            f"- {source['path']} [{source['ecosystem']}/{source['kind']}] scope={source['scope']}"
        )
    for issue in payload["issues"]:
        paths = ", ".join(issue["paths"])
        lines.append(f"! {issue['severity']} {issue['code']}: {paths or issue['message']}")
    return "\n".join(lines)


def _render_patch(payload):
    lines = [
        f"patch: {payload['id']}",
        f"status: {payload['status']}",
        f"base: {payload['base_revision']}",
        f"dirty_at_plan: {payload['dirty_at_plan']}",
        f"sources: {payload['source_count']}",
        f"trials: {payload['trial_count']}",
    ]
    for source in payload["source_refs"]:
        lines.append(
            f"- source {source['path']} scope={source['scope']} sha256={source['sha256']}"
        )
    for target in payload["targets"]:
        lines.append(
            f"- target {target['target']}: {target['path']} ({target['diff_lines']} diff lines)"
        )
    return "\n".join(lines)


def _render_patch_list(items):
    if not items:
        return "no rule patches"
    return "\n".join(
        f"{item['id']} {item['status']} targets={len(item['targets'])} trials={item['trial_count']}"
        for item in items
    )


def _render_trial(payload):
    lines = [
        f"trial: {payload['id']}",
        f"status: {payload['status']}",
        f"runner: {payload['runner']} ({payload['selected_variant']})",
        f"duration_ms: {payload['duration_ms']}",
        f"changed_files: {payload['diff_stats']['changed_files']}",
        f"artifact_dir: {payload['artifact_dir']}",
    ]
    if payload.get("budget_violations"):
        lines.append("budget_violations: " + ", ".join(payload["budget_violations"]))
    if payload["stdout_excerpt"].strip():
        lines.extend(["", "output:", payload["stdout_excerpt"].strip()])
    if payload["stderr_excerpt"].strip():
        lines.extend(["", "stderr:", payload["stderr_excerpt"].strip()])
    return "\n".join(lines)


__all__ = ["build_rules_parser", "main"]
