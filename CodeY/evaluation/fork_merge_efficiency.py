"""Deterministic serial-vs-parallel benchmark for scoped writable Forks.

The benchmark deliberately uses fixed-delay fake model clients.  It measures
whether CodeY overlaps otherwise identical child-agent waits while preserving
the same calls, candidate changes, validation, and fixture contents.  It does not
claim production model throughput or network-provider speedup.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time

from CodeY.context.workspace import WorkspaceContext
from CodeY.core.runtime import CodeYAgent
from CodeY.providers.clients import FakeModelClient, ModelCompletion
from CodeY.storage.session import SessionStore


ARTIFACT_SCHEMA_VERSION = "fork-merge-efficiency-v1"


class _CallTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def enter(self):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)

    def exit(self):
        with self.lock:
            self.active -= 1


class _DelayedPatchClient:
    supports_prompt_cache = False

    def __init__(self, branch_id, path, old_text, new_text, delay_seconds, tracker):
        self.branch_id = branch_id
        self.path = path
        self.old_text = old_text
        self.new_text = new_text
        self.delay_seconds = float(delay_seconds)
        self.tracker = tracker
        self.call_index = 0

    def complete(self, prompt, max_new_tokens, **kwargs):
        del prompt, max_new_tokens, kwargs
        self.tracker.enter()
        try:
            time.sleep(self.delay_seconds)
        finally:
            self.tracker.exit()
        self.call_index += 1
        if self.call_index == 1:
            text = "<tool>" + json.dumps(
                {
                    "name": "patch_file",
                    "args": {
                        "path": self.path,
                        "old_text": self.old_text,
                        "new_text": self.new_text,
                    },
                }
            ) + "</tool>"
        else:
            text = f"<final>updated {self.path}</final>"
        return ModelCompletion(
            text=text,
            metadata={
                "provider_kind": "deterministic-fixed-delay",
                "branch_id": self.branch_id,
                "call_index": self.call_index,
            },
        )


def _git(repo, *args):
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _source_provenance():
    source_root = Path(__file__).resolve().parents[2]

    def optional_git(*args):
        try:
            return _git(source_root, *args)
        except Exception:
            return ""

    status = optional_git("status", "--porcelain=v1", "--untracked-files=all")
    try:
        git_version = subprocess.run(
            ["git", "--version"],
            cwd=source_root,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        ).stdout.strip()
    except Exception:
        git_version = ""
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit_sha": optional_git("rev-parse", "HEAD"),
        "source_branch": optional_git("branch", "--show-current"),
        "source_dirty": bool(status),
        "source_status_entry_count": len(status.splitlines()) if status else 0,
        "source_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os_platform": platform.platform(),
        "git_version": git_version,
        "core_autocrlf": optional_git("config", "--get", "core.autocrlf") or "unset",
    }


def _initialize_fixture(repo):
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    (repo / ".gitignore").write_text(".codey/\n", encoding="utf-8")
    (repo / "api.txt").write_text("api=old\n", encoding="utf-8")
    (repo / "tests.txt").write_text("tests=old\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "api.txt", "tests.txt")
    _git(
        repo,
        "-c",
        "user.name=Benchmark",
        "-c",
        "user.email=benchmark@example.com",
        "commit",
        "-m",
        "fixture",
    )


def _fork_merge_call():
    return "<tool>" + json.dumps(
        {
            "name": "fork_merge",
            "args": {
                "tasks": [
                    {
                        "id": "api",
                        "objective": "update the API marker",
                        "allowed_paths": ["api.txt"],
                    },
                    {
                        "id": "tests",
                        "objective": "update the tests marker",
                        "allowed_paths": ["tests.txt"],
                    },
                ],
                "max_steps": 3,
                "merge_policy": "atomic_disjoint",
            },
        }
    ) + "</tool>"


def _fixture_content_digest(repo):
    digest = hashlib.sha256()
    for name in ("api.txt", "tests.txt"):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((repo / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_condition(parallelism, delay_ms):
    with tempfile.TemporaryDirectory(prefix="codey-fork-efficiency-") as temp_root:
        repo = Path(temp_root) / "repo"
        _initialize_fixture(repo)
        tracker = _CallTracker()
        edits = {
            "api": ("api.txt", "api=old", "api=new"),
            "tests": ("tests.txt", "tests=old", "tests=new"),
        }

        def factory(spec):
            return _DelayedPatchClient(
                spec.branch_id,
                *edits[spec.branch_id],
                delay_seconds=delay_ms / 1000.0,
                tracker=tracker,
            )

        validation = (
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "assert Path('api.txt').read_text() == 'api=new\\n'; "
                "assert Path('tests.txt').read_text() == 'tests=new\\n'"
            ),
        )
        agent = CodeYAgent(
            model_client=FakeModelClient([_fork_merge_call(), "<final>done</final>"]),
            model_client_factory=factory,
            workspace=WorkspaceContext.build(repo),
            session_store=SessionStore(repo / ".codey" / "sessions"),
            approval_policy="auto",
            skill_mode="off",
            feature_flags={"self_evolution": False},
            max_parallel_branches=parallelism,
            fork_merge_checks=[validation],
            fork_merge_check_timeout=30,
        )
        started_at = time.monotonic()
        final = agent.ask("run the deterministic fork merge benchmark")
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        tool_entry = next(
            item
            for item in agent.transcript_entries()
            if item.get("role") == "tool" and item.get("name") == "fork_merge"
        )
        summary = json.loads(tool_entry["content"])
        expected_digest = hashlib.sha256(
            b"api.txt\0api=new\r\n\0tests.txt\0tests=new\r\n\0"
            if (repo / "api.txt").read_bytes().endswith(b"\r\n")
            else b"api.txt\0api=new\n\0tests.txt\0tests=new\n\0"
        ).hexdigest()
        correctness = (
            final == "done"
            and summary.get("status") == "merged"
            and summary.get("validation", {}).get("status") == "passed"
            and tracker.calls == 4
            and _fixture_content_digest(repo) == expected_digest
            and not _git(repo, "status", "--porcelain")
        )
        return {
            "parallelism": int(parallelism),
            "delay_ms": int(delay_ms),
            "e2e_ms": elapsed_ms,
            "fork_merge_ms": int(summary.get("duration_ms", 0)),
            "provider_calls": tracker.calls,
            "max_active": tracker.max_active,
            "status": summary.get("status", ""),
            "validation_status": summary.get("validation", {}).get("status", ""),
            "fixture_content_digest": _fixture_content_digest(repo),
            "correct": bool(correctness),
        }


def _percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _condition_summary(rows):
    e2e = [row["e2e_ms"] for row in rows]
    fork = [row["fork_merge_ms"] for row in rows]
    return {
        "runs": len(rows),
        "correct_runs": sum(bool(row["correct"]) for row in rows),
        "median_e2e_ms": statistics.median(e2e),
        "mean_e2e_ms": statistics.fmean(e2e),
        "p95_e2e_ms": _percentile(e2e, 0.95),
        "median_fork_merge_ms": statistics.median(fork),
        "provider_calls": sorted({row["provider_calls"] for row in rows}),
        "max_active": max(row["max_active"] for row in rows),
        "fixture_content_digests": sorted(
            {row["fixture_content_digest"] for row in rows}
        ),
    }


def run_experiment(*, repetitions=8, warmups=2, delay_ms=200, parallelism=2):
    repetitions = max(1, int(repetitions))
    warmups = max(0, int(warmups))
    delay_ms = max(1, int(delay_ms))
    parallelism = int(parallelism)
    if parallelism != 2:
        raise ValueError("this two-branch protocol requires parallelism=2")
    for _ in range(warmups):
        _run_condition(1, delay_ms)
        _run_condition(parallelism, delay_ms)

    paired = []
    serial_rows = []
    parallel_rows = []
    for index in range(repetitions):
        order = (1, parallelism) if index % 2 == 0 else (parallelism, 1)
        by_parallelism = {}
        for condition in order:
            by_parallelism[condition] = _run_condition(condition, delay_ms)
        serial = by_parallelism[1]
        parallel = by_parallelism[parallelism]
        serial_rows.append(serial)
        parallel_rows.append(parallel)
        paired.append(
            {
                "trial": index + 1,
                "serial": serial,
                "parallel": parallel,
                "e2e_speedup": serial["e2e_ms"] / max(1, parallel["e2e_ms"]),
                "fork_merge_speedup": serial["fork_merge_ms"]
                / max(1, parallel["fork_merge_ms"]),
            }
        )

    serial_summary = _condition_summary(serial_rows)
    parallel_summary = _condition_summary(parallel_rows)
    speedups = [row["e2e_speedup"] for row in paired]
    fork_speedups = [row["fork_merge_speedup"] for row in paired]
    correctness_gate = (
        serial_summary["correct_runs"] == repetitions
        and parallel_summary["correct_runs"] == repetitions
    )
    work_conservation_gate = (
        serial_summary["provider_calls"] == parallel_summary["provider_calls"] == [4]
        and serial_summary["fixture_content_digests"]
        == parallel_summary["fixture_content_digests"]
    )
    overlap_gate = all(row["max_active"] == 1 for row in serial_rows) and all(
        row["max_active"] >= 2 for row in parallel_rows
    )
    median_speedup = statistics.median(speedups)
    p05_speedup = _percentile(speedups, 0.05)
    efficiency_gain_observed = bool(
        correctness_gate
        and work_conservation_gate
        and overlap_gate
        and p05_speedup > 1.0
    )
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "fork-merge-efficiency",
        "claim_scope": (
            "Synthetic fixed-delay fake child provider; two disjoint writable branches; "
            "real CodeY child AgentLoop, Git worktrees, candidate commits, integration validation, "
            "and local fast-forward. Not production model throughput or provider QPS."
        ),
        "provenance": _source_provenance(),
        "protocol": {
            "repetitions": repetitions,
            "warmups": warmups,
            "delay_ms_per_child_model_call": delay_ms,
            "child_provider_calls_per_trial": 4,
            "untracked_parent_fake_completions_per_trial": 2,
            "serial_parallelism": 1,
            "parallel_parallelism": parallelism,
            "condition_order": "alternating paired trials",
        },
        "conditions": {
            "serial": serial_summary,
            "parallel": parallel_summary,
        },
        "paired_trials": paired,
        "result": {
            "correctness_gate": correctness_gate,
            "work_conservation_gate": work_conservation_gate,
            "overlap_gate": overlap_gate,
            "median_e2e_speedup": median_speedup,
            "median_fork_merge_speedup": statistics.median(fork_speedups),
            "p05_e2e_speedup": p05_speedup,
            "efficiency_gain_observed": efficiency_gain_observed,
            "experiment_gate": efficiency_gain_observed,
        },
    }
    return artifact


def render_markdown(artifact):
    serial = artifact["conditions"]["serial"]
    parallel = artifact["conditions"]["parallel"]
    result = artifact["result"]
    protocol = artifact["protocol"]
    return "\n".join(
        [
            "# Fork Merge Efficiency Experiment",
            "",
            f"- Scope: {artifact['claim_scope']}",
            f"- Paired measured trials: {protocol['repetitions']}",
            f"- Warmups: {protocol['warmups']}",
            f"- Fixed delay per child model call: {protocol['delay_ms_per_child_model_call']} ms",
            f"- Serial median end-to-end: {serial['median_e2e_ms']:.1f} ms",
            f"- Parallel median end-to-end: {parallel['median_e2e_ms']:.1f} ms",
            f"- Median paired end-to-end speedup: {result['median_e2e_speedup']:.3f}x",
            f"- Median paired fork-merge speedup: {result['median_fork_merge_speedup']:.3f}x",
            f"- 5th-percentile paired end-to-end speedup: {result['p05_e2e_speedup']:.3f}x",
            f"- Correctness gate: {result['correctness_gate']}",
            f"- Work-conservation/content-equivalence gate: {result['work_conservation_gate']}",
            f"- Overlap gate: {result['overlap_gate']}",
            f"- Efficiency gain observed: {result['efficiency_gain_observed']}",
            f"- Experiment gate: {result['experiment_gate']}",
            "",
        ]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--delay-ms", type=int, default=200)
    parser.add_argument("--parallelism", type=int, choices=(2,), default=2)
    parser.add_argument(
        "--output-json",
        default="artifacts/fork-merge-efficiency-v1.json",
    )
    parser.add_argument(
        "--output-markdown",
        default="artifacts/fork-merge-efficiency-v1.md",
    )
    args = parser.parse_args(argv)
    artifact = run_experiment(
        repetitions=args.repetitions,
        warmups=args.warmups,
        delay_ms=args.delay_ms,
        parallelism=args.parallelism,
    )
    output_json = Path(args.output_json)
    output_markdown = Path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(render_markdown(artifact), encoding="utf-8")
    print(render_markdown(artifact), end="")
    return 0 if artifact["result"]["experiment_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
