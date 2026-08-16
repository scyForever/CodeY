"""Scoped writable Fork orchestration backed by disposable Git worktrees.

The ordinary :mod:`CodeY.core.fork` path deliberately stays read-only.  This
module is the separate, explicitly risky protocol for parallel code changes:
each homogeneous child edits an exact path lease in its own detached worktree,
the coordinator turns successful changes into candidate commits, integrates
them in another disposable worktree, runs user-configured fixed-argv checks,
and only then fast-forwards the original target branch.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import uuid

from ..context.workspace import MAX_TOOL_OUTPUT, WorkspaceContext, clip, now
from ..tools import registry as toolkit
from ..tools.security import detected_secret_env_items
from .fork import _safe_branch_id, compact_join_summary


MAX_CANDIDATE_PATCH_BYTES = 512_000
SCOPED_WRITE_TOOLS = ("list_files", "read_file", "search", "write_file", "patch_file")
_PROCESS_REPOSITORY_LOCK = threading.Lock()


class ForkMergeRefused(RuntimeError):
    """A fail-closed protocol result that leaves the target worktree untouched."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


class GitCommandError(RuntimeError):
    def __init__(self, args, result):
        self.args_list = tuple(str(item) for item in args)
        self.returncode = int(result.returncode)
        self.stdout = str(result.stdout or "")
        self.stderr = str(result.stderr or "")
        super().__init__(
            f"git command failed with exit code {self.returncode}: "
            + " ".join(self.args_list)
        )


@dataclass(frozen=True)
class MergeBranchSpec:
    branch_id: str
    objective: str
    index: int
    allowed_paths: tuple[str, ...]

    @classmethod
    def from_payload(cls, item, index):
        return cls(
            branch_id=_safe_branch_id(item.get("id"), index),
            objective=str(item.get("objective", "")).strip(),
            index=index,
            allowed_paths=tuple(
                toolkit.normalize_scoped_path(path)
                for path in item.get("allowed_paths", [])
            ),
        )


@dataclass
class MergeBranchResult:
    branch_id: str
    objective: str
    index: int
    allowed_paths: tuple[str, ...]
    status: str
    final_answer: str = ""
    run_id: str = ""
    thread_id: str = ""
    stop_reason: str = ""
    error_code: str = ""
    error_type: str = ""
    error_message: str = ""
    duration_ms: int = 0
    changed_paths: list[str] = field(default_factory=list)
    candidate_commit: str = ""
    patch_sha256: str = ""
    patch_path: str = ""
    merge_status: str = "pending"
    result_path: str = ""
    _patch_text: str = field(default="", repr=False)

    def to_dict(self):
        return {
            "branch_id": self.branch_id,
            "objective": self.objective,
            "index": self.index,
            "allowed_paths": list(self.allowed_paths),
            "status": self.status,
            "final_answer": self.final_answer,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "stop_reason": self.stop_reason,
            "error_code": self.error_code,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "changed_paths": list(self.changed_paths),
            "candidate_commit": self.candidate_commit,
            "patch_sha256": self.patch_sha256,
            "patch_path": self.patch_path,
            "merge_status": self.merge_status,
            "result_path": self.result_path,
        }


class _RepositoryLock:
    """Cross-process lock for one repository's worktree/ref mutation protocol."""

    def __init__(self, path, timeout=10.0):
        self.path = Path(path)
        self.timeout = float(timeout)
        self.handle = None

    def __enter__(self):
        if not _PROCESS_REPOSITORY_LOCK.acquire(timeout=self.timeout):
            raise ForkMergeRefused("repository_busy", "another fork_merge job is active")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("a+b")
            self.handle.seek(0, os.SEEK_END)
            if self.handle.tell() == 0:
                self.handle.write(b"\0")
                self.handle.flush()
            deadline = time.monotonic() + self.timeout
            while True:
                try:
                    self._lock_once()
                    return self
                except OSError:
                    if time.monotonic() >= deadline:
                        raise ForkMergeRefused(
                            "repository_busy", "another fork_merge process holds the repository lock"
                        )
                    time.sleep(0.05)
        except Exception:
            if self.handle is not None:
                self.handle.close()
                self.handle = None
            _PROCESS_REPOSITORY_LOCK.release()
            raise

    def _lock_once(self):
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        try:
            if self.handle is not None:
                self.handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            if self.handle is not None:
                self.handle.close()
                self.handle = None
            _PROCESS_REPOSITORY_LOCK.release()


class WorktreeForkCoordinator:
    """Run scoped writable children and atomically integrate their changes."""

    def __init__(self, parent):
        self.parent = parent
        self.repo_root = Path(parent.root).resolve()
        self.job_root = None
        self.hooks_root = None
        self.empty_git_config = None
        self.worktrees = []
        self.safe_git_config = []

    def run(self, args):
        parent = self.parent
        parent_state = parent.current_task_state
        if parent_state is None:
            raise RuntimeError("fork_merge requires an active parent run")

        fork_id = "merge_" + uuid.uuid4().hex[:10]
        specs = [
            MergeBranchSpec.from_payload(item, index)
            for index, item in enumerate(args["tasks"])
        ]
        if len({spec.branch_id.casefold() for spec in specs}) != len(specs):
            raise ValueError("branch ids collide after normalization")
        max_steps = int(args.get("max_steps", 4))
        started_at = time.monotonic()
        fork_state = {
            "fork_id": fork_id,
            "kind": "worktree_merge",
            "source_run_id": parent_state.run_id,
            "source_task_id": parent_state.task_id,
            "status": "running",
            "merge_policy": "atomic_disjoint",
            "branch_ids": [spec.branch_id for spec in specs],
            "created_at": now(),
            "updated_at": now(),
        }
        parent.save_fork_state(fork_state)
        parent.emit_trace(
            parent_state,
            "merge_fork_started",
            {
                "fork_id": fork_id,
                "branch_ids": list(fork_state["branch_ids"]),
                "merge_policy": "atomic_disjoint",
                "max_steps": max_steps,
            },
        )

        results = {}
        summary = None
        cleanup = {"status": "not_started", "pending_paths": []}
        try:
            self._prepare_job_root(fork_id)
            common_dir = self._git_common_dir()
            with _RepositoryLock(common_dir / "codey-fork-merge.lock"):
                summary = self._execute_locked(
                    specs=specs,
                    fork_id=fork_id,
                    max_steps=max_steps,
                    results=results,
                    started_at=started_at,
                    fork_state=fork_state,
                )
        except ForkMergeRefused as exc:
            summary = self._base_summary(
                fork_id,
                specs,
                status=exc.code,
                started_at=started_at,
                error_type=type(exc).__name__,
                message=str(exc),
            )
        except Exception as exc:
            fork_state.update(
                {
                    "status": "failed",
                    "updated_at": now(),
                    "error_type": type(exc).__name__,
                }
            )
            try:
                parent.save_fork_state(fork_state)
                parent.emit_trace(
                    parent_state,
                    "merge_failed",
                    {"fork_id": fork_id, "error_type": type(exc).__name__},
                )
            except Exception:
                pass
            raise
        finally:
            cleanup = self._cleanup_worktrees()

        summary["cleanup"] = cleanup
        if cleanup["status"] == "pending" and summary["status"] == "merged":
            summary["status"] = "merged_cleanup_pending"
        summary = self._compact_summary(summary)
        self._finalize_results(parent_state, fork_id, results, summary)
        fork_state.update(
            {
                "status": summary["status"],
                "updated_at": now(),
                "base_commit": summary.get("base_commit", ""),
                "target_ref": summary.get("target_ref", ""),
                "integration_commit": summary.get("integration_commit", ""),
                "succeeded": summary.get("succeeded", 0),
                "failed": summary.get("failed", len(specs)),
                "result_paths": [
                    results[spec.branch_id].result_path
                    for spec in specs
                    if spec.branch_id in results
                ],
                "cleanup": cleanup,
            }
        )
        parent.save_fork_state(fork_state)
        parent_state.fork_count += 1
        parent_state.fork_summary = {
            key: value for key, value in summary.items() if key != "branches"
        }
        parent.run_store.write_task_state(parent_state)
        event = "merge_completed" if summary["status"].startswith("merged") else "merge_failed"
        parent.emit_trace(parent_state, event, dict(parent_state.fork_summary))
        return summary

    def _prepare_job_root(self, fork_id):
        self.job_root = Path(tempfile.mkdtemp(prefix=f"codey-{fork_id}-")).resolve()
        self.hooks_root = self.job_root / "empty-hooks"
        self.hooks_root.mkdir()
        self.empty_git_config = self.job_root / "empty-gitconfig"
        self.empty_git_config.touch()
        self.safe_git_config = self._read_safe_effective_git_config()

    def _read_safe_effective_git_config(self):
        allowed = {
            "core.autocrlf": {"true", "false", "input"},
            "core.eol": {"lf", "crlf", "native"},
            "core.safecrlf": {"true", "false", "warn"},
            "core.ignorecase": {"true", "false"},
            "core.symlinks": {"true", "false"},
            "core.filemode": {"true", "false"},
        }
        resolved = []
        env = {
            name: value
            for name, value in os.environ.items()
            if not name.upper().startswith("GIT_")
        }
        env["GIT_TERMINAL_PROMPT"] = "0"
        for key, accepted in allowed.items():
            result = subprocess.run(
                ["git", "config", "--get", key],
                cwd=self.repo_root,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                env=env,
            )
            value = result.stdout.strip().casefold()
            if result.returncode == 0 and value in accepted:
                resolved.append((key, value))
        return resolved

    def _git_common_dir(self):
        result = self._git(self.repo_root, ["rev-parse", "--git-common-dir"])
        common = Path(result.stdout.strip())
        if not common.is_absolute():
            common = self.repo_root / common
        return common.resolve()

    def _execute_locked(self, specs, fork_id, max_steps, results, started_at, fork_state):
        base_commit, target_ref = self._preflight(specs)
        parent_state = self.parent.current_task_state
        prepared = []
        for spec in specs:
            worktree = self.job_root / f"branch-{spec.index + 1}-{spec.branch_id}"
            self._add_worktree(worktree, base_commit)
            workspace = WorkspaceContext.build(
                worktree,
                repo_root_override=worktree,
                git_prefix_args=self._git_prefix_args(),
                git_env=self._git_env(),
            )
            thread_id = f"{parent_state.graph_thread_id}/{fork_id}/{spec.branch_id}"
            child = self.parent.create_fork_merge_child(
                spec=spec,
                fork_id=fork_id,
                thread_id=thread_id,
                max_steps=max_steps,
                workspace=workspace,
            )
            prepared.append((spec, thread_id, child, worktree))
            self.parent.emit_trace(
                parent_state,
                "merge_branch_started",
                {
                    "fork_id": fork_id,
                    "branch_id": spec.branch_id,
                    "thread_id": thread_id,
                    "allowed_paths": list(spec.allowed_paths),
                },
            )

        worker_count = min(len(prepared), self.parent.max_parallel_branches)
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="codey-fork-merge",
        ) as executor:
            futures = {
                executor.submit(
                    self._run_branch,
                    spec,
                    thread_id,
                    child,
                    worktree,
                    base_commit,
                ): spec
                for spec, thread_id, child, worktree in prepared
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = MergeBranchResult(
                        branch_id=spec.branch_id,
                        objective=spec.objective,
                        index=spec.index,
                        allowed_paths=spec.allowed_paths,
                        status="failed",
                        error_code="branch_infrastructure_error",
                        error_type=type(exc).__name__,
                    )
                results[spec.branch_id] = result
                self._persist_result(parent_state, fork_id, result, emit=True)

        ordered = [results[spec.branch_id] for spec in specs]
        candidate_count = sum(result.status == "candidate_ready" for result in ordered)
        if candidate_count != len(ordered):
            for result in ordered:
                result.merge_status = "blocked_by_branch_failure"
            return self._summary_from_results(
                fork_id,
                ordered,
                status="branch_failed",
                started_at=started_at,
                base_commit=base_commit,
                target_ref=target_ref,
            )

        integration = self.job_root / "integration"
        self._add_worktree(integration, base_commit)
        self.parent.emit_trace(
            parent_state,
            "integration_started",
            {"fork_id": fork_id, "base_commit": base_commit},
        )
        conflict = self._integrate_candidates(integration, ordered)
        if conflict is not None:
            conflict_branch, conflict_paths = conflict
            for result in ordered:
                if result.branch_id == conflict_branch:
                    result.merge_status = "conflict"
                elif result.merge_status == "pending":
                    result.merge_status = "blocked_by_conflict"
            return self._summary_from_results(
                fork_id,
                ordered,
                status="merge_conflict",
                started_at=started_at,
                base_commit=base_commit,
                target_ref=target_ref,
                conflict_paths=conflict_paths,
            )

        integration_commit = self._commit_integration(integration, ordered)
        validation = self._run_validation(integration, base_commit, integration_commit)
        if validation["status"] != "passed":
            for result in ordered:
                result.merge_status = "blocked_by_validation"
            return self._summary_from_results(
                fork_id,
                ordered,
                status="validation_failed",
                started_at=started_at,
                base_commit=base_commit,
                target_ref=target_ref,
                integration_commit=integration_commit,
                validation=validation,
            )

        if not self._target_unchanged(base_commit, target_ref):
            for result in ordered:
                result.merge_status = "stale_target"
            return self._summary_from_results(
                fork_id,
                ordered,
                status="stale_target",
                started_at=started_at,
                base_commit=base_commit,
                target_ref=target_ref,
                integration_commit=integration_commit,
                validation=validation,
            )

        fork_state.update(
            {
                "status": "merge_ready",
                "updated_at": now(),
                "base_commit": base_commit,
                "target_ref": target_ref,
                "integration_commit": integration_commit,
                "validation_status": "passed",
            }
        )
        self.parent.save_fork_state(fork_state)
        try:
            self._git(
                self.repo_root,
                ["merge", "--ff-only", "--no-edit", integration_commit],
            )
        except (GitCommandError, subprocess.TimeoutExpired) as exc:
            for result in ordered:
                result.merge_status = "target_update_failed"
            return self._summary_from_results(
                fork_id,
                ordered,
                status="target_update_failed",
                started_at=started_at,
                base_commit=base_commit,
                target_ref=target_ref,
                integration_commit=integration_commit,
                validation=validation,
                message=f"target fast-forward failed: {type(exc).__name__}",
            )

        persistence_error = ""
        fork_state.update(
            {
                "status": "target_updated",
                "updated_at": now(),
                "target_head": integration_commit,
            }
        )
        try:
            self.parent.save_fork_state(fork_state)
        except Exception as exc:
            persistence_error = type(exc).__name__

        try:
            target_verified = self._target_unchanged(integration_commit, target_ref)
        except Exception as exc:
            target_verified = False
            persistence_error = persistence_error or type(exc).__name__
        if not target_verified:
            for result in ordered:
                result.merge_status = "target_state_unknown"
            return self._summary_from_results(
                fork_id,
                ordered,
                status="target_state_unknown",
                started_at=started_at,
                base_commit=base_commit,
                target_ref=target_ref,
                integration_commit=integration_commit,
                validation=validation,
                message=(
                    "target fast-forward returned success but the final worktree/ref state "
                    "could not be verified"
                ),
            )
        for result in ordered:
            result.merge_status = "merged"
        return self._summary_from_results(
            fork_id,
            ordered,
            status="merged_persistence_pending" if persistence_error else "merged",
            started_at=started_at,
            base_commit=base_commit,
            target_ref=target_ref,
            integration_commit=integration_commit,
            validation=validation,
            message=(
                f"target merged but fork-state persistence needs reconciliation: {persistence_error}"
                if persistence_error
                else ""
            ),
        )

    def _preflight(self, specs):
        inside = self._git(self.repo_root, ["rev-parse", "--is-inside-work-tree"]).stdout.strip()
        if inside != "true":
            raise ForkMergeRefused("not_git_repository", "fork_merge requires a Git worktree")
        target = self._git(
            self.repo_root,
            ["symbolic-ref", "--quiet", "HEAD"],
            check=False,
        )
        target_ref = target.stdout.strip()
        if target.returncode != 0 or not target_ref.startswith("refs/heads/"):
            raise ForkMergeRefused("detached_target", "fork_merge requires a checked-out local branch")
        base_commit = self._git(self.repo_root, ["rev-parse", "HEAD"]).stdout.strip()
        target_changes = self._git_status(self.repo_root)
        if target_changes:
            raise ForkMergeRefused(
                "dirty_target",
                "fork_merge requires a clean tracked and untracked target worktree: "
                + ", ".join(target_changes[:20]),
            )
        unsafe_config = self._git(
            self.repo_root,
            [
                "config",
                "--local",
                "--get-regexp",
                (
                    r"^(filter\..*\.(clean|smudge|process)|diff\.external|"
                    r"diff\..*\.(command|textconv)|merge\..*\.driver)$"
                ),
            ],
            check=False,
        )
        if unsafe_config.returncode not in {0, 1}:
            raise GitCommandError(("config", "--local", "--get-regexp"), unsafe_config)
        if unsafe_config.stdout.strip():
            raise ForkMergeRefused(
                "unsafe_git_config",
                "fork_merge refuses local filters, external diff/textconv, or custom merge drivers",
            )
        for spec in specs:
            for path in spec.allowed_paths:
                ignored = self._git(
                    self.repo_root,
                    ["check-ignore", "--quiet", "--", path],
                    check=False,
                )
                if ignored.returncode == 0:
                    raise ForkMergeRefused(
                        "ignored_allowed_path",
                        f"allowed path is ignored by Git: {path}",
                    )
                if ignored.returncode not in {0, 1}:
                    raise GitCommandError(("check-ignore", "--", path), ignored)
        return base_commit, target_ref

    def _add_worktree(self, path, revision):
        resolved = Path(path).resolve()
        if resolved.parent != self.job_root or not resolved.name:
            raise RuntimeError("refusing worktree path outside the fork job root")
        self.worktrees.append(resolved)
        self._git(
            self.repo_root,
            ["worktree", "add", "--detach", str(resolved), revision],
            timeout=60,
        )

    def _run_branch(self, spec, thread_id, child, worktree, base_commit):
        started_at = time.monotonic()
        final = child.ask(
            self._branch_objective(spec),
            thread_id=thread_id,
        )
        task_state = child.current_task_state
        result = MergeBranchResult(
            branch_id=spec.branch_id,
            objective=spec.objective,
            index=spec.index,
            allowed_paths=spec.allowed_paths,
            status="failed",
            final_answer=clip(final, 1000),
            run_id=task_state.run_id,
            thread_id=thread_id,
            stop_reason=task_state.stop_reason,
            duration_ms=int((time.monotonic() - started_at) * 1000),
        )
        if task_state.status != "completed":
            result.error_code = "child_run_failed"
            return result

        changed_paths = self._changed_paths(worktree)
        result.changed_paths = changed_paths
        allowed = set(spec.allowed_paths)
        out_of_scope = [path for path in changed_paths if path not in allowed]
        if out_of_scope:
            result.error_code = "out_of_scope_changes"
            return result
        if not changed_paths:
            result.error_code = "no_changes"
            return result

        if self._changed_files_contain_secret(worktree, changed_paths):
            result.error_code = "secret_detected"
            return result

        self._git(worktree, ["add", "--", *changed_paths])
        staged = self._split_z(
            self._git(
                worktree,
                [
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--cached",
                    "--name-only",
                    "--no-renames",
                    "-z",
                ],
            ).stdout
        )
        if sorted(staged) != sorted(changed_paths):
            result.error_code = "staged_path_mismatch"
            return result
        check = self._git(
            worktree,
            ["diff", "--no-ext-diff", "--no-textconv", "--cached", "--check"],
            check=False,
        )
        if check.returncode != 0:
            result.error_code = "candidate_diff_check_failed"
            result.error_message = clip(check.stderr or check.stdout, 300)
            return result
        patch = self._git(
            worktree,
            [
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--cached",
                "--binary",
                "--full-index",
                base_commit,
                "--",
            ],
        ).stdout
        patch_bytes = patch.encode("utf-8")
        if len(patch_bytes) > MAX_CANDIDATE_PATCH_BYTES:
            result.error_code = "candidate_patch_too_large"
            return result
        redacted_patch = self.parent.redact_text(patch)
        result._patch_text = redacted_patch
        result.patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()
        if redacted_patch != patch:
            result.error_code = "secret_detected"
            return result

        self._git(
            worktree,
            [
                "commit",
                "--no-gpg-sign",
                "--no-verify",
                "-m",
                f"CodeY candidate: {spec.branch_id}",
            ],
        )
        candidate_commit = self._git(worktree, ["rev-parse", "HEAD"]).stdout.strip()
        candidate_parent = self._git(worktree, ["rev-parse", "HEAD^"]).stdout.strip()
        if candidate_parent != base_commit:
            result.error_code = "candidate_parent_mismatch"
            return result
        result.candidate_commit = candidate_commit
        result.status = "candidate_ready"
        result.merge_status = "pending"
        result.duration_ms = int((time.monotonic() - started_at) * 1000)
        return result

    @staticmethod
    def _branch_objective(spec):
        paths = "\n".join(f"- {path}" for path in spec.allowed_paths)
        return (
            f"{spec.objective}\n\n"
            "Writable fork contract:\n"
            "- Modify only the exact allowed paths listed below.\n"
            "- Use write_file or patch_file; shell and nested fork tools are unavailable.\n"
            "- Finish only after the requested code change is present in the worktree.\n"
            f"Allowed paths:\n{paths}"
        )

    def _changed_paths(self, worktree):
        tracked = self._split_z(
            self._git(
                worktree,
                [
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--name-only",
                    "--no-renames",
                    "-z",
                    "HEAD",
                    "--",
                ],
            ).stdout
        )
        untracked = self._split_z(
            self._git(
                worktree,
                ["ls-files", "--others", "--exclude-standard", "-z"],
            ).stdout
        )
        return sorted(set(tracked) | set(untracked), key=lambda value: value.casefold())

    def _changed_files_contain_secret(self, worktree, changed_paths):
        needles = [
            value.encode("utf-8")
            for _, value in detected_secret_env_items(
                secret_env_names=self.parent.secret_env_names
            )
            if value
        ]
        if not needles:
            return False
        max_needle = max(len(needle) for needle in needles)
        overlap_size = max_needle - 1
        for relative in changed_paths:
            path = Path(worktree) / relative
            if not path.is_file():
                continue
            overlap = b""
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(64 * 1024)
                    if not chunk:
                        break
                    block = overlap + chunk
                    if any(needle in block for needle in needles):
                        return True
                    overlap = block[-overlap_size:] if overlap_size else b""
        return False

    def _integrate_candidates(self, integration, ordered):
        for result in ordered:
            cherry_pick = self._git(
                integration,
                ["cherry-pick", "--no-commit", result.candidate_commit],
                check=False,
                timeout=60,
            )
            if cherry_pick.returncode != 0:
                conflicts = self._split_z(
                    self._git(
                        integration,
                    [
                        "diff",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--name-only",
                        "--diff-filter=U",
                        "-z",
                    ],
                        check=False,
                    ).stdout
                )
                return result.branch_id, conflicts
            result.merge_status = "integration_applied"
        check = self._git(
            integration,
            ["diff", "--no-ext-diff", "--no-textconv", "--cached", "--check"],
            check=False,
        )
        if check.returncode != 0:
            raise ForkMergeRefused(
                "integration_diff_check_failed",
                "combined candidate failed git diff --check",
            )
        return None

    def _commit_integration(self, integration, ordered):
        branch_ids = ", ".join(result.branch_id for result in ordered)
        self._git(
            integration,
            [
                "commit",
                "--no-gpg-sign",
                "--no-verify",
                "-m",
                f"CodeY parallel merge: {branch_ids}",
            ],
        )
        return self._git(integration, ["rev-parse", "HEAD"]).stdout.strip()

    def _run_validation(self, integration, base_commit, integration_commit):
        initial_state = self._validation_state(integration, integration_commit)
        if initial_state is not None:
            return initial_state
        diff_check = self._git(
            integration,
            [
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--check",
                f"{base_commit}..{integration_commit}",
            ],
            check=False,
        )
        results = [
            {
                "argv": [
                    "git",
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--check",
                    f"{base_commit}..{integration_commit}",
                ],
                "returncode": diff_check.returncode,
                "duration_ms": 0,
                "stdout": clip(self.parent.redact_text(diff_check.stdout), 500),
                "stderr": clip(self.parent.redact_text(diff_check.stderr), 500),
            }
        ]
        if diff_check.returncode != 0:
            return {"status": "failed", "results": results}

        for command in self.parent.fork_merge_checks:
            started_at = time.monotonic()
            try:
                completed = subprocess.run(
                    list(command),
                    cwd=integration,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=self.parent.fork_merge_check_timeout,
                    env=self._validation_env(integration),
                )
                result = {
                    "argv": list(command),
                    "returncode": completed.returncode,
                    "duration_ms": int((time.monotonic() - started_at) * 1000),
                    "stdout": clip(self.parent.redact_text(completed.stdout), 500),
                    "stderr": clip(self.parent.redact_text(completed.stderr), 500),
                }
            except subprocess.TimeoutExpired as exc:
                result = {
                    "argv": list(command),
                    "returncode": None,
                    "duration_ms": int((time.monotonic() - started_at) * 1000),
                    "stdout": clip(self.parent.redact_text(exc.stdout or ""), 500),
                    "stderr": clip(self.parent.redact_text(exc.stderr or ""), 500),
                    "error_code": "validation_timeout",
                }
            results.append(result)
            changed_state = self._validation_state(integration, integration_commit, results)
            if changed_state is not None:
                return changed_state
            if result.get("returncode") != 0:
                return {"status": "failed", "results": results}
        return {"status": "passed", "results": results}

    def _validation_state(self, integration, integration_commit, results=None):
        head = self._git(integration, ["rev-parse", "HEAD"], check=False)
        if head.returncode != 0 or head.stdout.strip() != integration_commit:
            return {
                "status": "failed",
                "error_code": "validation_modified_git_state",
                "expected_head": integration_commit,
                "observed_head": head.stdout.strip(),
                "results": list(results or []),
            }
        side_effects = self._git_status(integration)
        if side_effects:
            return {
                "status": "failed",
                "error_code": "validation_modified_worktree",
                "side_effects": side_effects,
                "results": list(results or []),
            }
        return None

    def _validation_env(self, integration):
        env = self.parent.shell_env()
        env = {
            name: value
            for name, value in env.items()
            if not name.upper().startswith("GIT_")
        }
        for name in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"):
            if os.environ.get(name):
                env[name] = os.environ[name]
        env["PWD"] = str(integration)
        env["CODEY_FORK_MERGE"] = "1"
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        if self.empty_git_config is not None:
            env["GIT_CONFIG_GLOBAL"] = str(self.empty_git_config)
        return env

    def _target_unchanged(self, expected_commit, expected_ref):
        head = self._git(self.repo_root, ["rev-parse", "HEAD"], check=False)
        ref = self._git(
            self.repo_root,
            ["symbolic-ref", "--quiet", "HEAD"],
            check=False,
        )
        if head.returncode != 0 or ref.returncode != 0:
            return False
        return (
            head.stdout.strip() == expected_commit
            and ref.stdout.strip() == expected_ref
            and not self._git_status(self.repo_root)
        )

    def _persist_result(self, parent_state, fork_id, result, *, emit):
        result_path = self.parent.run_store.branch_result_path(
            parent_state.run_id,
            fork_id,
            result.branch_id,
        )
        result.result_path = result_path.relative_to(self.parent.current_run_dir).as_posix()
        if result._patch_text:
            patch_path = self.parent.run_store.write_branch_patch(
                parent_state.run_id,
                fork_id,
                result.branch_id,
                result._patch_text,
            )
            result.patch_path = patch_path.relative_to(self.parent.current_run_dir).as_posix()
        safe_result = self.parent.redact_artifact(result.to_dict())
        self.parent.run_store.write_branch_result(
            parent_state.run_id,
            fork_id,
            result.branch_id,
            safe_result,
        )
        if not emit:
            return
        event = "merge_branch_candidate" if result.status == "candidate_ready" else "merge_branch_failed"
        self.parent.emit_trace(
            parent_state,
            event,
            {
                "fork_id": fork_id,
                "branch_id": result.branch_id,
                "child_run_id": result.run_id,
                "status": result.status,
                "error_code": result.error_code,
                "error_type": result.error_type,
                "error_message": result.error_message,
                "changed_paths": list(result.changed_paths),
                "candidate_commit": result.candidate_commit,
                "result_path": result.result_path,
                "duration_ms": result.duration_ms,
            },
        )

    def _finalize_results(self, parent_state, fork_id, results, summary):
        merge_by_id = {
            branch["branch_id"]: str(branch.get("merge_status", ""))
            for branch in summary.get("branches", [])
        }
        for branch_id, result in results.items():
            if branch_id in merge_by_id:
                result.merge_status = merge_by_id[branch_id]
            self._persist_result(parent_state, fork_id, result, emit=False)

    def _summary_from_results(
        self,
        fork_id,
        ordered,
        *,
        status,
        started_at,
        base_commit,
        target_ref,
        integration_commit="",
        validation=None,
        conflict_paths=None,
        message="",
    ):
        succeeded = sum(result.status == "candidate_ready" for result in ordered)
        return {
            "fork_id": fork_id,
            "kind": "worktree_merge",
            "status": status,
            "merge_policy": "atomic_disjoint",
            "base_commit": base_commit,
            "target_ref": target_ref,
            "integration_commit": integration_commit,
            "branch_count": len(ordered),
            "succeeded": succeeded,
            "failed": len(ordered) - succeeded,
            "duration_ms": int((time.monotonic() - started_at) * 1000),
            "validation": validation or {"status": "not_run", "results": []},
            "conflict_paths": list(conflict_paths or []),
            "message": message,
            "branches": [self.parent.redact_artifact(result.to_dict()) for result in ordered],
        }

    @staticmethod
    def _base_summary(
        fork_id,
        specs,
        *,
        status,
        started_at,
        error_type="",
        message="",
    ):
        return {
            "fork_id": fork_id,
            "kind": "worktree_merge",
            "status": status,
            "merge_policy": "atomic_disjoint",
            "base_commit": "",
            "target_ref": "",
            "integration_commit": "",
            "branch_count": len(specs),
            "succeeded": 0,
            "failed": len(specs),
            "duration_ms": int((time.monotonic() - started_at) * 1000),
            "error_type": error_type,
            "message": message,
            "validation": {"status": "not_run", "results": []},
            "conflict_paths": [],
            "branches": [],
        }

    def _compact_summary(self, summary):
        payload = self.parent.redact_artifact(summary)
        payload = compact_join_summary(payload, limit=MAX_TOOL_OUTPUT - 300)
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) < MAX_TOOL_OUTPUT:
            return payload
        validation = dict(payload.get("validation", {}))
        validation["results"] = [
            {
                "command": str((result.get("argv") or [""])[0]),
                "argv_sha256": hashlib.sha256(
                    json.dumps(
                        result.get("argv", []),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "returncode": result.get("returncode"),
                "duration_ms": result.get("duration_ms", 0),
                "error_code": result.get("error_code", ""),
            }
            for result in validation.get("results", [])
        ]
        payload["validation"] = validation
        payload = compact_join_summary(payload, limit=MAX_TOOL_OUTPUT - 300)
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) < MAX_TOOL_OUTPUT:
            return payload
        return {
            "fork_id": str(payload.get("fork_id", ""))[:80],
            "kind": "worktree_merge",
            "status": str(payload.get("status", ""))[:80],
            "merge_policy": "atomic_disjoint",
            "base_commit": str(payload.get("base_commit", ""))[:64],
            "target_ref": str(payload.get("target_ref", ""))[:200],
            "integration_commit": str(payload.get("integration_commit", ""))[:64],
            "branch_count": int(payload.get("branch_count", 0)),
            "succeeded": int(payload.get("succeeded", 0)),
            "failed": int(payload.get("failed", 0)),
            "duration_ms": int(payload.get("duration_ms", 0)),
            "message": clip(payload.get("message", ""), 200),
            "validation": {
                "status": str(validation.get("status", ""))[:80],
                "error_code": str(validation.get("error_code", ""))[:80],
            },
            "cleanup": {
                "status": str(payload.get("cleanup", {}).get("status", ""))[:80],
            },
            "branches": [
                {
                    "branch_id": str(branch.get("branch_id", ""))[:80],
                    "status": str(branch.get("status", ""))[:80],
                    "merge_status": str(branch.get("merge_status", ""))[:80],
                    "result_path": str(branch.get("result_path", ""))[:240],
                }
                for branch in payload.get("branches", [])
            ],
        }

    def _cleanup_worktrees(self):
        if self.job_root is None:
            return {"status": "not_started", "pending_paths": []}
        pending = []
        for path in reversed(self.worktrees):
            resolved = Path(path).resolve()
            if resolved.parent != self.job_root:
                pending.append(resolved.name)
                continue
            try:
                result = self._git(
                    self.repo_root,
                    ["worktree", "remove", "--force", str(resolved)],
                    check=False,
                    timeout=60,
                )
            except Exception:
                pending.append(resolved.name)
                continue
            if result.returncode != 0 or resolved.exists():
                pending.append(resolved.name)
        for child in (self.empty_git_config, self.hooks_root):
            try:
                if child is not None and child.is_file():
                    child.unlink()
                elif child is not None and child.is_dir():
                    child.rmdir()
            except OSError:
                pending.append(child.name)
        try:
            self.job_root.rmdir()
        except OSError:
            pending.append(self.job_root.name)
        return {
            "status": "pending" if pending else "completed",
            "pending_paths": sorted(set(pending)),
        }

    def _git_prefix_args(self):
        argv = [
            "-c",
            f"core.hooksPath={self.hooks_root or ''}",
            "-c",
            "credential.helper=",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "user.name=CodeY Fork Coordinator",
            "-c",
            "user.email=codey-fork@localhost",
        ]
        for key, value in self.safe_git_config:
            argv.extend(("-c", f"{key}={value}"))
        return argv

    def _git_env(self):
        env = {
            name: value
            for name, value in os.environ.items()
            if not name.upper().startswith("GIT_")
        }
        env.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_EDITOR": "true",
                "GIT_SEQUENCE_EDITOR": "true",
            }
        )
        if self.empty_git_config is not None:
            env["GIT_CONFIG_GLOBAL"] = str(self.empty_git_config)
        return env

    def _git(self, cwd, args, *, check=True, timeout=30):
        argv = ["git", *self._git_prefix_args(), *(str(item) for item in args)]
        result = subprocess.run(
            argv,
            cwd=Path(cwd),
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=self._git_env(),
        )
        if check and result.returncode != 0:
            raise GitCommandError(args, result)
        return result

    def _git_status(self, cwd):
        result = self._git(
            cwd,
            [
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignore-submodules=none",
                "--",
                ".",
            ],
            check=False,
        )
        if result.returncode != 0:
            raise GitCommandError(("status", "--porcelain=v1"), result)
        return self._split_z(result.stdout)

    @staticmethod
    def _split_z(value):
        return [item for item in str(value or "").split("\0") if item]
