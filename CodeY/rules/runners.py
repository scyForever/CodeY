"""Isolated CodeY, Codex, and Claude trials for reviewed rule patches."""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..tools.security import redact_text
from .discovery import SECRET_VALUE_PATTERN
from .patches import RulePatchStore, _atomic_write_json, _atomic_write_text


RUNNERS = {"codey", "codex", "claude"}
VARIANTS = {"baseline", "candidate", "canary"}
MODES = {"inspect", "edit"}
RUNNER_ENV_ALLOWLIST = {
    "APPDATA",
    "CLAUDE_CONFIG_DIR",
    "CODEX_HOME",
    "COMSPEC",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOCALAPPDATA",
    "PATH",
    "PATHEXT",
    "PROGRAMDATA",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "USERNAME",
    "WINDIR",
}


def _sha256(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sanitize_text(value, limit=None):
    text = redact_text(str(value))
    text = SECRET_VALUE_PATTERN.sub("<redacted>", text)
    if limit is not None and len(text) > int(limit):
        return text[: int(limit)] + f"\n...[truncated {len(text) - int(limit)} chars]"
    return text


@dataclass(frozen=True)
class TrialRequest:
    patch_id: str
    runner: str
    task: str
    variant: str = "candidate"
    mode: str = "inspect"
    cohort_key: str = ""
    candidate_fraction: float = 0.20
    timeout_seconds: int = 300
    max_output_bytes: int = 256 * 1024
    max_changed_files: int = 12
    max_diff_lines: int = 1200
    max_diff_bytes: int = 256 * 1024
    allow_dirty_base: bool = False

    def validate(self):
        if self.runner not in RUNNERS:
            raise ValueError(f"unknown trial runner: {self.runner}")
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown trial variant: {self.variant}")
        if self.mode not in MODES:
            raise ValueError(f"unknown trial mode: {self.mode}")
        if not str(self.task).strip():
            raise ValueError("trial task must not be empty")
        if self.variant == "canary" and not str(self.cohort_key).strip():
            raise ValueError("canary trials require a stable cohort key")
        if not 0.0 <= float(self.candidate_fraction) <= 1.0:
            raise ValueError("candidate_fraction must be between 0 and 1")
        for name in (
            "timeout_seconds",
            "max_output_bytes",
            "max_changed_files",
            "max_diff_lines",
            "max_diff_bytes",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.runner == "codey" and self.mode != "inspect":
            raise ValueError("CodeY rule trials are inspect-only; use the normal agent for small edits")


class ExternalAgentRunner:
    """Execute one bounded agent task in a disposable Git worktree."""

    def __init__(self, repository_root, patch_store=None):
        self.root = Path(repository_root).resolve()
        self.store = patch_store or RulePatchStore(self.root)
        self.git_safety_root = self.store.store_root / "git-safety"

    def probe(self, runner):
        runner = str(runner).lower()
        if runner == "codey":
            return {
                "runner": "codey",
                "available": True,
                "executable": Path(sys.executable).name,
                "version": f"Python {sys.version.split()[0]}",
                "capabilities": ["inspect", "repository_rules", "no_risky_tools"],
            }
        if runner not in {"codex", "claude"}:
            raise ValueError(f"unknown trial runner: {runner}")
        executable, version = self._resolve_executable(runner, required=False)
        capabilities = (
            ["inspect", "edit", "read_only_sandbox", "workspace_write_sandbox", "ephemeral"]
            if runner == "codex"
            else ["inspect", "edit", "project_settings_only", "no_session_persistence"]
        )
        return {
            "runner": runner,
            "available": executable is not None,
            "executable": Path(executable).name if executable else "",
            "version": version,
            "capabilities": capabilities,
        }

    def probe_all(self):
        return [self.probe(name) for name in ("codey", "codex", "claude")]

    def run(self, request: TrialRequest):
        request.validate()
        patch = self.store.load(request.patch_id)
        target_paths = {artifact["path"] for artifact in patch["artifacts"]}
        dirty_targets = sorted(
            target_paths.intersection(patch["repository"].get("dirty_paths_at_plan", []))
        )
        if dirty_targets:
            raise ValueError(
                "rule patch target was dirty during planning and cannot produce a valid baseline: "
                + ", ".join(dirty_targets)
            )
        if patch["repository"].get("dirty_at_plan") and not request.allow_dirty_base:
            raise ValueError(
                "rule patch was planned from a dirty workspace; commit the baseline or explicitly allow a committed-base trial"
            )
        selected_variant = self._select_variant(request)
        trial_id = "trial_" + uuid.uuid4().hex[:16]
        started = time.monotonic()
        executable_info = self.probe(request.runner)
        if not executable_info["available"]:
            raise ValueError(f"local {request.runner} executable is not available")

        git_env = self._git_env()
        with _DetachedWorktree(
            self.root,
            patch["repository"]["base_revision"],
            git_env=git_env,
        ) as workspace:
            self._prepare_workspace(workspace, patch, request.runner, selected_variant, request.mode)
            argv, process_cwd = self._build_argv(request, workspace)
            process = self._run_process(
                argv,
                cwd=process_cwd,
                timeout_seconds=request.timeout_seconds,
                max_output_bytes=request.max_output_bytes,
                input_text=request.task if request.runner in {"codex", "claude"} else None,
            )
            diff, diff_stats = self._collect_diff(workspace, request.max_diff_bytes)

        violations = []
        if diff_stats["changed_files"] > request.max_changed_files:
            violations.append("max_changed_files")
        if diff_stats["diff_lines"] > request.max_diff_lines:
            violations.append("max_diff_lines")
        if diff_stats["diff_bytes"] > request.max_diff_bytes or diff_stats["diff_truncated"]:
            violations.append("max_diff_bytes")
        if process["output_limit_exceeded"]:
            violations.append("max_output_bytes")
        sensitive_diff = bool(SECRET_VALUE_PATTERN.search(diff)) or redact_text(diff) != diff
        if sensitive_diff:
            violations.append("secret_shaped_diff")

        if sensitive_diff:
            status = "blocked_sensitive_diff"
        elif process["timed_out"]:
            status = "timed_out"
        elif process["output_limit_exceeded"]:
            status = "output_limit_exceeded"
        elif process["exit_code"] != 0:
            status = "failed"
        elif violations:
            status = "budget_exceeded"
        elif request.mode == "inspect" and diff_stats["changed_files"]:
            status = "unexpected_changes"
        else:
            status = "completed"

        result = {
            "schema_version": 1,
            "id": trial_id,
            "patch_id": patch["id"],
            "runner": request.runner,
            "runner_version": executable_info["version"],
            "mode": request.mode,
            "requested_variant": request.variant,
            "selected_variant": selected_variant,
            "candidate_fraction": float(request.candidate_fraction),
            "allowed_dirty_base": bool(request.allow_dirty_base),
            "base_revision": patch["repository"]["base_revision"],
            "cohort_key_sha256": _sha256(request.cohort_key) if request.cohort_key else "",
            "task_sha256": _sha256(request.task),
            "task_excerpt": _sanitize_text(request.task, 240),
            "status": status,
            "exit_code": process["exit_code"],
            "timed_out": process["timed_out"],
            "duration_ms": int((time.monotonic() - started) * 1000),
            "output_truncated": process["output_truncated"],
            "output_limit_exceeded": process["output_limit_exceeded"],
            "stdout_sha256": _sha256(process["stdout"]),
            "stderr_sha256": _sha256(process["stderr"]),
            "stdout_excerpt": _sanitize_text(process["stdout"], 4000),
            "stderr_excerpt": _sanitize_text(process["stderr"], 2000),
            "diff_stats": diff_stats,
            "budget_violations": violations,
            "diff_saved": bool(diff) and not sensitive_diff and not diff_stats["diff_truncated"],
            "artifact_dir": (Path(".codey") / "rules" / "trials" / trial_id).as_posix(),
        }
        trial_dir = self.store.trial_root / trial_id
        trial_dir.mkdir(parents=True, exist_ok=False)
        if diff and not sensitive_diff and not diff_stats["diff_truncated"]:
            _atomic_write_text(trial_dir / "changes.patch", diff)
        _atomic_write_json(trial_dir / "result.json", result)
        return result

    def _select_variant(self, request):
        if request.variant != "canary":
            return request.variant
        digest = hashlib.sha256(
            f"{request.patch_id}:{request.runner}:{request.cohort_key}".encode("utf-8")
        ).digest()
        sample = int.from_bytes(digest[:8], "big") / float(2**64)
        return "candidate" if sample < float(request.candidate_fraction) else "baseline"

    def _prepare_workspace(self, workspace, patch, runner, variant, mode):
        if variant == "candidate":
            artifact = next(
                (item for item in patch["artifacts"] if item["target"] == runner),
                None,
            )
            if artifact is None:
                raise ValueError(f"rule patch has no artifact for runner: {runner}")
            target = self._workspace_path(workspace, artifact["path"])
            _atomic_write_text(target, artifact["candidate_content"])

        # Trial safety configuration is part of the disposable baseline, not the
        # agent-produced diff. Repository hooks are never executed in trials.
        if runner == "codex":
            sandbox = "read-only" if mode == "inspect" else "workspace-write"
            _atomic_write_text(
                self._workspace_path(workspace, ".codex/config.toml"),
                f'sandbox_mode = "{sandbox}"\napproval_policy = "never"\n',
            )
        if runner == "claude":
            _atomic_write_text(self._workspace_path(workspace, ".claude/settings.json"), "{}\n")
        self._git(workspace, "add", "-A", "-f", check=True)
        self._git(
            workspace,
            "-c",
            "user.name=CodeY Trial",
            "-c",
            "user.email=codey-trial@localhost",
            "commit",
            "--no-verify",
            "--no-gpg-sign",
            "--allow-empty",
            "-m",
            "codey isolated trial baseline",
            check=True,
        )

    def _build_argv(self, request, workspace):
        if request.runner == "codey":
            return (
                [
                    sys.executable,
                    "-m",
                    "CodeY",
                    "--cwd",
                    str(workspace),
                    "--approval",
                    "never",
                    "--skill",
                    "off",
                    "--evolution-mode",
                    "rules",
                    request.task,
                ],
                self.root,
            )
        executable, _ = self._resolve_executable(request.runner, required=True)
        if request.runner == "codex":
            sandbox = "read-only" if request.mode == "inspect" else "workspace-write"
            return (
                [
                    executable,
                    "exec",
                    "--sandbox",
                    sandbox,
                    "--ephemeral",
                    "--ignore-user-config",
                    "--color",
                    "never",
                    "-c",
                    'approval_policy="never"',
                    "--json",
                    "-C",
                    str(workspace),
                    "-",
                ],
                workspace,
            )
        permission = "plan" if request.mode == "inspect" else "acceptEdits"
        tools = "Read,Glob,Grep" if request.mode == "inspect" else "Read,Glob,Grep,Edit,Write"
        return (
            [
                executable,
                "-p",
                "--output-format",
                "json",
                "--no-session-persistence",
                "--setting-sources",
                "project",
                "--strict-mcp-config",
                "--mcp-config",
                '{"mcpServers":{}}',
                "--no-chrome",
                "--tools",
                tools,
                "--permission-mode",
                permission,
            ],
            workspace,
        )

    def _resolve_executable(self, runner, required):
        candidates = (f"{runner}.exe", runner) if os.name == "nt" else (runner,)
        for candidate in candidates:
            path = shutil.which(candidate)
            if not path:
                continue
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    timeout=8,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if result.returncode == 0:
                version = (result.stdout or result.stderr).strip().splitlines()[0]
                return path, _sanitize_text(version, 200)
        if required:
            raise ValueError(f"local {runner} executable is not available or not runnable")
        return None, ""

    def _run_process(self, argv, cwd, timeout_seconds, max_output_bytes, input_text=None):
        with tempfile.TemporaryDirectory(prefix="codey-runner-output-") as output_root:
            stdout_path = Path(output_root) / "stdout.bin"
            stderr_path = Path(output_root) / "stderr.bin"
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=self._runner_env(cwd),
                    stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    shell=False,
                    creationflags=creationflags,
                    start_new_session=os.name != "nt",
                )
                stdin_thread = None
                if input_text is not None and process.stdin is not None:
                    stdin_thread = threading.Thread(
                        target=self._write_stdin,
                        args=(process.stdin, str(input_text).encode("utf-8")),
                        name="codey-runner-stdin",
                        daemon=True,
                    )
                    stdin_thread.start()
                timed_out = False
                output_limit_exceeded = False
                deadline = time.monotonic() + int(timeout_seconds)
                while process.poll() is None:
                    if self._file_size(stdout_path) + self._file_size(stderr_path) > max_output_bytes:
                        output_limit_exceeded = True
                        self._terminate_process_tree(process)
                        break
                    if time.monotonic() >= deadline:
                        timed_out = True
                        self._terminate_process_tree(process)
                        break
                    time.sleep(0.05)
                exit_code = self._bounded_wait(process)
                if stdin_thread is not None:
                    stdin_thread.join(timeout=1)
                output_limit_exceeded = output_limit_exceeded or (
                    self._file_size(stdout_path) + self._file_size(stderr_path) > max_output_bytes
                )
            with stdout_path.open("rb") as handle:
                stdout_raw = handle.read(max_output_bytes + 1)
            with stderr_path.open("rb") as handle:
                stderr_raw = handle.read(max_output_bytes + 1)
        truncated = (
            output_limit_exceeded
            or len(stdout_raw) > max_output_bytes
            or len(stderr_raw) > max_output_bytes
        )
        stdout = stdout_raw[:max_output_bytes].decode("utf-8", errors="replace")
        stderr = stderr_raw[:max_output_bytes].decode("utf-8", errors="replace")
        return {
            "exit_code": int(exit_code),
            "timed_out": timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "output_truncated": truncated,
            "output_limit_exceeded": output_limit_exceeded,
        }

    @staticmethod
    def _write_stdin(handle, payload):
        try:
            for offset in range(0, len(payload), 16 * 1024):
                handle.write(payload[offset : offset + 16 * 1024])
                handle.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                handle.close()
            except OSError:
                pass

    @staticmethod
    def _terminate_process_tree(process):
        if process.poll() is not None:
            return
        if os.name == "nt":
            try:
                result = subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                    timeout=15,
                )
                if result.returncode != 0 and process.poll() is None:
                    process.kill()
            except (OSError, subprocess.SubprocessError):
                try:
                    process.kill()
                except OSError:
                    pass
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass

    @staticmethod
    def _bounded_wait(process):
        try:
            return process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            try:
                return process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return -9

    @staticmethod
    def _file_size(path):
        try:
            return path.stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _runner_env(cwd):
        env = {
            name: value
            for name, value in os.environ.items()
            if name.upper() in RUNNER_ENV_ALLOWLIST
        }
        env["PWD"] = str(cwd)
        return env

    def _git_env(self):
        self.git_safety_root.mkdir(parents=True, exist_ok=True)
        hooks_root = self.git_safety_root / "empty-hooks"
        hooks_root.mkdir(exist_ok=True)
        empty_config = self.git_safety_root / "empty.gitconfig"
        empty_attributes = self.git_safety_root / "empty.gitattributes"
        for path in (empty_config, empty_attributes):
            if not path.exists():
                _atomic_write_text(path, "")

        env = {name: value for name, value in os.environ.items() if not name.upper().startswith("GIT_")}
        env.update(
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": str(empty_config),
                "GIT_ATTR_NOSYSTEM": "1",
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_OPTIONAL_LOCKS": "0",
            }
        )
        overrides = [
            ("core.hooksPath", str(hooks_root)),
            ("core.attributesFile", str(empty_attributes)),
            ("core.fsmonitor", "false"),
        ]
        overrides.extend(self._disabled_local_filters(env))
        env["GIT_CONFIG_COUNT"] = str(len(overrides))
        for index, (key, value) in enumerate(overrides):
            env[f"GIT_CONFIG_KEY_{index}"] = key
            env[f"GIT_CONFIG_VALUE_{index}"] = value
        return env

    def _disabled_local_filters(self, env):
        result = subprocess.run(
            [
                "git",
                "config",
                "--local",
                "--includes",
                "--name-only",
                "--get-regexp",
                r"^filter\..*\.(clean|smudge|process|required)$",
            ],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=30,
        )
        if result.returncode not in {0, 1}:
            raise RuntimeError("unable to inspect repository-local Git filter configuration")
        drivers = set()
        for key in result.stdout.splitlines():
            if not key.casefold().startswith("filter.") or "." not in key[7:]:
                continue
            drivers.add(key[7:].rsplit(".", 1)[0])
        if len(drivers) > 128:
            raise ValueError("repository config declares too many Git filter drivers for a safe trial")
        overrides = []
        for driver in sorted(drivers):
            overrides.extend(
                [
                    (f"filter.{driver}.clean", ""),
                    (f"filter.{driver}.smudge", ""),
                    (f"filter.{driver}.process", ""),
                    (f"filter.{driver}.required", "false"),
                ]
            )
        return overrides

    def _collect_diff(self, workspace, max_diff_bytes):
        status = self._git(
            workspace,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            check=True,
        )
        untracked = []
        entries = [entry for entry in status.split("\0") if entry]
        for entry in entries:
            if entry.startswith("?? "):
                untracked.append(entry[3:])
        if untracked:
            self._git(workspace, "add", "-N", "--", *untracked, check=True)
        names = self._git(workspace, "diff", "--name-only", "-z", "HEAD", "--", check=True)
        changed_files = len([name for name in names.split("\0") if name])
        diff, diff_truncated = self._limited_git_diff(workspace, int(max_diff_bytes))
        return diff, {
            "changed_files": changed_files,
            "diff_lines": len(diff.splitlines()),
            "diff_bytes": len(diff.encode("utf-8")),
            "diff_truncated": diff_truncated,
        }

    def _limited_git_diff(self, workspace, max_bytes):
        process = subprocess.Popen(
            ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
            cwd=workspace,
            env=self._git_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        assert process.stdout is not None
        data = process.stdout.read(max_bytes + 1)
        truncated = len(data) > max_bytes
        if truncated:
            try:
                process.kill()
            except OSError:
                pass
            data = data[:max_bytes]
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            truncated = True
            return_code = process.returncode
        if return_code != 0:
            assert process.stderr is not None
            error = process.stderr.read(4096).decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"git diff failed: {error or return_code}")
        return data.decode("utf-8", errors="replace"), truncated

    def _workspace_path(self, workspace, relative):
        candidate = Path(workspace).joinpath(*Path(str(relative).replace("\\", "/")).parts)
        parent = candidate.parent.resolve()
        try:
            parent.relative_to(Path(workspace).resolve())
        except ValueError as exc:
            raise ValueError(f"trial path escapes isolated workspace: {relative}") from exc
        return candidate

    def _git(self, cwd, *args, check=False):
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=self._git_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=60,
        )
        if check and result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"git {' '.join(args)} failed: {message}")
        return result.stdout


class _DetachedWorktree:
    def __init__(self, repository_root, revision, git_env):
        self.root = Path(repository_root).resolve()
        self.revision = str(revision)
        self.git_env = git_env
        self.path = None

    def __enter__(self):
        parent = self.root.parent
        self.path = Path(tempfile.mkdtemp(prefix=".codey-trial-", dir=parent)).resolve()
        self.path.rmdir()
        result = subprocess.run(
            ["git", "worktree", "add", "--detach", str(self.path), self.revision],
            cwd=self.root,
            env=self.git_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=120,
        )
        if result.returncode != 0:
            if self.path.exists():
                shutil.rmtree(self.path, ignore_errors=True)
            raise RuntimeError(f"unable to create isolated worktree: {(result.stderr or result.stdout).strip()}")
        return self.path

    def __exit__(self, exc_type, exc, traceback):
        if self.path is None:
            return False
        resolved = self.path.resolve()
        try:
            resolved.relative_to(self.root.parent.resolve())
        except ValueError:
            return False
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(resolved)],
            cwd=self.root,
            env=self.git_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=120,
        )
        if resolved.exists():
            shutil.rmtree(resolved, ignore_errors=True)
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=self.root,
            env=self.git_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=30,
        )
        return False


__all__ = ["ExternalAgentRunner", "TrialRequest"]
