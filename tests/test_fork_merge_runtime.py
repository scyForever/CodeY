import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from CodeY import CodeYAgent, FakeModelClient, ModelCompletion, SessionStore, WorkspaceContext
from CodeY.context.workspace import MAX_TOOL_OUTPUT
from CodeY.core.worktree_fork import GitCommandError, WorktreeForkCoordinator
from CodeY.tools import registry as tool_registry


class ScopedEditClient:
    supports_prompt_cache = False

    def __init__(self, branch_id, path, old_text, new_text, *, delay=0.0, tracker=None):
        self.branch_id = branch_id
        self.path = path
        self.old_text = old_text
        self.new_text = new_text
        self.delay = float(delay)
        self.tracker = tracker
        self.call_index = 0

    def complete(self, prompt, max_new_tokens, **kwargs):
        del prompt, max_new_tokens, kwargs
        if self.tracker is not None:
            self.tracker.enter(self.branch_id)
        try:
            if self.delay:
                time.sleep(self.delay)
        finally:
            if self.tracker is not None:
                self.tracker.exit(self.branch_id)
        self.call_index += 1
        if self.call_index == 1:
            return ModelCompletion(
                text=(
                    "<tool>"
                    + json.dumps(
                        {
                            "name": "patch_file",
                            "args": {
                                "path": self.path,
                                "old_text": self.old_text,
                                "new_text": self.new_text,
                            },
                        }
                    )
                    + "</tool>"
                ),
                metadata={"branch_id": self.branch_id, "call_index": self.call_index},
            )
        return ModelCompletion(
            text=f"<final>updated {self.path}</final>",
            metadata={"branch_id": self.branch_id, "call_index": self.call_index},
        )


class ConcurrencyTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = []

    def enter(self, branch_id):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append((branch_id, "start"))

    def exit(self, branch_id):
        with self.lock:
            self.calls.append((branch_id, "end"))
            self.active -= 1


def git(repo, *args):
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def initialize_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    (repo / ".gitignore").write_text(".codey/\n", encoding="utf-8")
    (repo / "api.txt").write_text("api=old\n", encoding="utf-8")
    (repo / "tests.txt").write_text("tests=old\n", encoding="utf-8")
    (repo / "other.txt").write_text("other=old\n", encoding="utf-8")
    git(repo, "add", ".gitignore", "api.txt", "tests.txt", "other.txt")
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "initial",
    )
    return repo


def merge_call(tasks, max_steps=3):
    return "<tool>" + json.dumps(
        {
            "name": "fork_merge",
            "args": {
                "tasks": tasks,
                "max_steps": max_steps,
                "merge_policy": "atomic_disjoint",
            },
        }
    ) + "</tool>"


def default_tasks():
    return [
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
    ]


def build_agent(repo, parent_outputs, factory, checks, *, max_parallel_branches=2):
    return CodeYAgent(
        model_client=FakeModelClient(parent_outputs),
        model_client_factory=factory,
        workspace=WorkspaceContext.build(repo),
        session_store=SessionStore(repo / ".codey" / "sessions"),
        approval_policy="auto",
        skill_mode="off",
        feature_flags={"self_evolution": False},
        max_parallel_branches=max_parallel_branches,
        fork_merge_checks=checks,
        fork_merge_check_timeout=20,
    )


def joined_merge_result(agent):
    tool_record = next(
        item
        for item in agent.transcript_entries()
        if item.get("role") == "tool" and item.get("name") == "fork_merge"
    )
    return json.loads(tool_record["content"])


def test_fork_merge_edits_disjoint_worktrees_validates_and_fast_forwards_all_success(tmp_path):
    repo = initialize_repo(tmp_path)
    base_commit = git(repo, "rev-parse", "HEAD")
    tracker = ConcurrencyTracker()

    def factory(spec):
        mapping = {
            "api": ("api.txt", "api=old", "api=new"),
            "tests": ("tests.txt", "tests=old", "tests=new"),
        }
        return ScopedEditClient(spec.branch_id, *mapping[spec.branch_id], delay=0.05, tracker=tracker)

    check = (
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            "assert Path('api.txt').read_text() == 'api=new\\n'; "
            "assert Path('tests.txt').read_text() == 'tests=new\\n'"
        ),
    )
    agent = build_agent(
        repo,
        [merge_call(default_tasks()), "<final>merged</final>"],
        factory,
        [check],
    )

    assert agent.ask("parallel edit and merge") == "merged"

    joined = joined_merge_result(agent)
    assert joined["status"] == "merged"
    assert joined["validation"]["status"] == "passed"
    assert joined["base_commit"] == base_commit
    assert joined["integration_commit"] == git(repo, "rev-parse", "HEAD")
    assert joined["integration_commit"] != base_commit
    assert joined["succeeded"] == 2
    assert joined["failed"] == 0
    assert tracker.max_active == 2
    assert sum(event == "start" for _, event in tracker.calls) == 4
    assert (repo / "api.txt").read_text(encoding="utf-8") == "api=new\n"
    assert (repo / "tests.txt").read_text(encoding="utf-8") == "tests=new\n"
    assert git(repo, "status", "--porcelain") == ""
    assert all(branch["candidate_commit"] for branch in joined["branches"])
    assert all(branch["merge_status"] == "merged" for branch in joined["branches"])
    assert all(branch["patch_sha256"] for branch in joined["branches"])
    assert joined["cleanup"]["status"] == "completed"
    for branch in joined["branches"]:
        result_path = Path(agent.current_run_dir) / branch["result_path"]
        patch_path = Path(agent.current_run_dir) / branch["patch_path"]
        assert result_path.is_file()
        assert patch_path.is_file()


def test_fork_merge_rejects_overlapping_or_protected_path_leases_before_running(tmp_path):
    repo = initialize_repo(tmp_path)
    agent = build_agent(
        repo,
        ["<final>unused</final>"],
        lambda spec: FakeModelClient(["<final>unused</final>"]),
        [(sys.executable, "-c", "pass")],
    )

    overlap = agent.run_tool(
        "fork_merge",
        {
            "tasks": [
                {"id": "a", "objective": "a", "allowed_paths": ["api.txt"]},
                {"id": "b", "objective": "b", "allowed_paths": ["API.txt"]},
            ]
        },
    )
    protected = agent.run_tool(
        "fork_merge",
        {
            "tasks": [
                {"id": "a", "objective": "a", "allowed_paths": [".env"]},
                {"id": "b", "objective": "b", "allowed_paths": ["tests.txt"]},
            ]
        },
    )
    branch_id_collision = agent.run_tool(
        "fork_merge",
        {
            "tasks": [
                {"id": "api", "objective": "a", "allowed_paths": ["api.txt"]},
                {"id": "API", "objective": "b", "allowed_paths": ["tests.txt"]},
            ]
        },
    )

    assert "allowed path lease conflict" in overlap
    assert "environment files" in protected
    assert "duplicate branch id" in branch_id_collision
    assert git(repo, "status", "--porcelain") == ""


def test_fork_merge_blocks_out_of_scope_child_write_and_keeps_target_unchanged(tmp_path):
    repo = initialize_repo(tmp_path)
    base_commit = git(repo, "rev-parse", "HEAD")

    def factory(spec):
        if spec.branch_id == "api":
            return ScopedEditClient(spec.branch_id, "other.txt", "other=old", "other=bad")
        return ScopedEditClient(spec.branch_id, "tests.txt", "tests=old", "tests=new")

    agent = build_agent(
        repo,
        [merge_call(default_tasks()), "<final>handled</final>"],
        factory,
        [(sys.executable, "-c", "pass")],
    )

    assert agent.ask("enforce branch scope") == "handled"

    joined = joined_merge_result(agent)
    assert joined["status"] == "branch_failed"
    assert git(repo, "rev-parse", "HEAD") == base_commit
    assert (repo / "other.txt").read_text(encoding="utf-8") == "other=old\n"
    api = next(branch for branch in joined["branches"] if branch["branch_id"] == "api")
    assert api["error_code"] == "no_changes"
    child_trace = agent.run_store.trace_path(api["run_id"]).read_text(encoding="utf-8")
    assert "write_scope_violation" in child_trace


def test_fork_merge_validation_failure_preserves_original_branch_and_files(tmp_path):
    repo = initialize_repo(tmp_path)
    base_commit = git(repo, "rev-parse", "HEAD")

    def factory(spec):
        mapping = {
            "api": ("api.txt", "api=old", "api=new"),
            "tests": ("tests.txt", "tests=old", "tests=new"),
        }
        return ScopedEditClient(spec.branch_id, *mapping[spec.branch_id])

    agent = build_agent(
        repo,
        [merge_call(default_tasks()), "<final>validation handled</final>"],
        factory,
        [(sys.executable, "-c", "raise SystemExit(7)")],
    )

    assert agent.ask("fail closed on validation") == "validation handled"

    joined = joined_merge_result(agent)
    assert joined["status"] == "validation_failed"
    assert joined["validation"]["status"] == "failed"
    assert joined["validation"]["results"][-1]["returncode"] == 7
    assert git(repo, "rev-parse", "HEAD") == base_commit
    assert (repo / "api.txt").read_text(encoding="utf-8") == "api=old\n"
    assert (repo / "tests.txt").read_text(encoding="utf-8") == "tests=old\n"


def test_fork_merge_refuses_dirty_target_without_creating_candidates(tmp_path):
    repo = initialize_repo(tmp_path)
    base_commit = git(repo, "rev-parse", "HEAD")
    (repo / "other.txt").write_text("user change\n", encoding="utf-8")

    agent = build_agent(
        repo,
        [merge_call(default_tasks()), "<final>dirty target handled</final>"],
        lambda spec: FakeModelClient(["<final>must not run</final>"]),
        [(sys.executable, "-c", "pass")],
    )

    assert agent.ask("do not overwrite user changes") == "dirty target handled"

    joined = joined_merge_result(agent)
    assert joined["status"] == "dirty_target"
    assert joined["branches"] == []
    assert git(repo, "rev-parse", "HEAD") == base_commit
    assert (repo / "other.txt").read_text(encoding="utf-8") == "user change\n"


def test_fork_merge_is_disabled_without_user_configured_validation_checks(tmp_path):
    repo = initialize_repo(tmp_path)
    agent = CodeYAgent(
        model_client=FakeModelClient(["<final>unused</final>"]),
        workspace=WorkspaceContext.build(repo),
        session_store=SessionStore(repo / ".codey" / "sessions"),
        approval_policy="auto",
        skill_mode="off",
        feature_flags={"self_evolution": False},
    )

    assert "fork_join" in agent.tools
    assert "fork_merge" not in agent.tools


def test_fork_merge_target_drift_gate_runs_after_validation_and_before_fast_forward(
    tmp_path,
    monkeypatch,
):
    repo = initialize_repo(tmp_path)
    base_commit = git(repo, "rev-parse", "HEAD")

    def factory(spec):
        mapping = {
            "api": ("api.txt", "api=old", "api=new"),
            "tests": ("tests.txt", "tests=old", "tests=new"),
        }
        return ScopedEditClient(spec.branch_id, *mapping[spec.branch_id])

    monkeypatch.setattr(
        WorktreeForkCoordinator,
        "_target_unchanged",
        lambda self, expected_commit, expected_ref: False,
    )
    agent = build_agent(
        repo,
        [merge_call(default_tasks()), "<final>stale target handled</final>"],
        factory,
        [(sys.executable, "-c", "pass")],
    )

    assert agent.ask("detect target drift") == "stale target handled"

    joined = joined_merge_result(agent)
    assert joined["status"] == "stale_target"
    assert joined["validation"]["status"] == "passed"
    assert git(repo, "rev-parse", "HEAD") == base_commit
    assert (repo / "api.txt").read_text(encoding="utf-8") == "api=old\n"
    assert (repo / "tests.txt").read_text(encoding="utf-8") == "tests=old\n"


def test_fork_merge_rejects_validation_that_changes_the_integration_head(tmp_path):
    repo = initialize_repo(tmp_path)
    base_commit = git(repo, "rev-parse", "HEAD")

    def factory(spec):
        mapping = {
            "api": ("api.txt", "api=old", "api=new"),
            "tests": ("tests.txt", "tests=old", "tests=new"),
        }
        return ScopedEditClient(spec.branch_id, *mapping[spec.branch_id])

    reset_head = (
        sys.executable,
        "-c",
        (
            "import subprocess; "
            "subprocess.run(['git', 'reset', '--hard', 'HEAD^'], check=True)"
        ),
    )
    agent = build_agent(
        repo,
        [merge_call(default_tasks()), "<final>validation state handled</final>"],
        factory,
        [reset_head],
    )

    assert agent.ask("reject validation Git state changes") == "validation state handled"

    joined = joined_merge_result(agent)
    assert joined["status"] == "validation_failed"
    assert joined["validation"]["error_code"] == "validation_modified_git_state"
    assert git(repo, "rev-parse", "HEAD") == base_commit
    assert (repo / "api.txt").read_text(encoding="utf-8") == "api=old\n"


def test_fork_merge_scans_environment_secrets_before_git_add(tmp_path, monkeypatch):
    repo = initialize_repo(tmp_path)
    secret = "codey-secret-value-for-prestage-test"
    monkeypatch.setenv("CODEY_TEST_SECRET", secret)

    def factory(spec):
        if spec.branch_id == "api":
            tool = "<tool>" + json.dumps(
                {
                    "name": "write_file",
                    "args": {"path": "api.txt", "content": f"api={secret}\n"},
                }
            ) + "</tool>"
            return FakeModelClient([tool, "<final>wrote secret candidate</final>"])
        return ScopedEditClient(spec.branch_id, "tests.txt", "tests=old", "tests=new")

    agent = build_agent(
        repo,
        [merge_call(default_tasks()), "<final>secret handled</final>"],
        factory,
        [(sys.executable, "-c", "pass")],
    )

    assert agent.ask("reject the secret candidate") == "secret handled"

    joined = joined_merge_result(agent)
    api = next(branch for branch in joined["branches"] if branch["branch_id"] == "api")
    assert joined["status"] == "branch_failed"
    assert api["error_code"] == "secret_detected"
    assert (repo / "api.txt").read_text(encoding="utf-8") == "api=old\n"
    object_id = subprocess.run(
        ["git", "hash-object", "--stdin"],
        cwd=repo,
        input=f"api={secret}\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    object_probe = subprocess.run(
        ["git", "cat-file", "-e", object_id],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert object_probe.returncode != 0


@pytest.mark.skipif(os.name == "nt", reason="Windows resolves case aliases to one file")
def test_writable_scope_authorization_uses_exact_path_spelling(tmp_path):
    repo = initialize_repo(tmp_path)
    agent = CodeYAgent(
        model_client=FakeModelClient(["<final>unused</final>"]),
        workspace=WorkspaceContext.build(repo),
        session_store=SessionStore(repo / ".codey" / "sessions"),
        approval_policy="auto",
        skill_mode="off",
        feature_flags={"self_evolution": False},
        write_allowed_paths=("api.txt",),
    )

    agent.authorize_write_path("api.txt")
    with pytest.raises(ValueError, match="not allowed by branch scope"):
        agent.authorize_write_path("API.txt")


def test_git_status_failure_is_not_treated_as_clean(tmp_path, monkeypatch):
    repo = initialize_repo(tmp_path)
    agent = build_agent(
        repo,
        ["<final>unused</final>"],
        lambda spec: FakeModelClient(["<final>unused</final>"]),
        [(sys.executable, "-c", "pass")],
    )
    coordinator = WorktreeForkCoordinator(agent)
    monkeypatch.setattr(
        coordinator,
        "_git",
        lambda cwd, args, **kwargs: subprocess.CompletedProcess(
            ["git", *args], 2, stdout="", stderr="status unavailable"
        ),
    )

    with pytest.raises(GitCommandError):
        coordinator._git_status(repo)


def test_cleanup_failure_is_reported_without_raising(tmp_path, monkeypatch):
    repo = initialize_repo(tmp_path)
    agent = build_agent(
        repo,
        ["<final>unused</final>"],
        lambda spec: FakeModelClient(["<final>unused</final>"]),
        [(sys.executable, "-c", "pass")],
    )
    coordinator = WorktreeForkCoordinator(agent)
    coordinator.job_root = (tmp_path / "job").resolve()
    coordinator.job_root.mkdir()
    coordinator.hooks_root = coordinator.job_root / "empty-hooks"
    coordinator.hooks_root.mkdir()
    coordinator.empty_git_config = coordinator.job_root / "empty-gitconfig"
    coordinator.empty_git_config.touch()
    worktree = coordinator.job_root / "integration"
    worktree.mkdir()
    coordinator.worktrees = [worktree]

    def fail_remove(cwd, args, **kwargs):
        raise subprocess.TimeoutExpired(["git", *args], timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(coordinator, "_git", fail_remove)
    cleanup = coordinator._cleanup_worktrees()

    assert cleanup["status"] == "pending"
    assert "integration" in cleanup["pending_paths"]


def test_search_terminates_rg_options_and_uses_filtered_environment(tmp_path, monkeypatch):
    captured = {}

    class SearchContext:
        root = tmp_path

        @staticmethod
        def path(raw):
            return tmp_path / raw

        @staticmethod
        def shell_env():
            return {"PATH": "filtered"}

    monkeypatch.setattr(tool_registry.shutil, "which", lambda name: "rg")

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout="(no matches)", stderr="")

    monkeypatch.setattr(tool_registry.subprocess, "run", fake_run)

    assert tool_registry.tool_search(SearchContext(), {"pattern": "--version", "path": "."})
    assert captured["argv"][-3:] == ["--", "--version", str(tmp_path)]
    assert captured["kwargs"]["env"] == {"PATH": "filtered"}
    assert captured["kwargs"]["timeout"] == 20


def test_git_environment_discards_inherited_git_overrides(tmp_path, monkeypatch):
    repo = initialize_repo(tmp_path)
    agent = build_agent(
        repo,
        ["<final>unused</final>"],
        lambda spec: FakeModelClient(["<final>unused</final>"]),
        [(sys.executable, "-c", "pass")],
    )
    coordinator = WorktreeForkCoordinator(agent)
    coordinator.job_root = (tmp_path / "job-git-env").resolve()
    coordinator.job_root.mkdir()
    coordinator.hooks_root = coordinator.job_root / "empty-hooks"
    coordinator.hooks_root.mkdir()
    coordinator.empty_git_config = coordinator.job_root / "empty-gitconfig"
    coordinator.empty_git_config.touch()
    monkeypatch.setenv("GIT_INDEX_FILE", "wrong-index")
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", "unsafe-diff")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")

    env = coordinator._git_env()

    assert "GIT_INDEX_FILE" not in env
    assert "GIT_EXTERNAL_DIFF" not in env
    assert "GIT_CONFIG_COUNT" not in env
    assert env["GIT_CONFIG_GLOBAL"] == str(coordinator.empty_git_config)
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"


def test_reused_branch_client_identity_shares_one_provider_lock(tmp_path):
    repo = initialize_repo(tmp_path)
    shared_client = FakeModelClient(["<final>unused</final>"])
    agent = build_agent(
        repo,
        ["<final>unused</final>"],
        lambda spec: shared_client,
        [(sys.executable, "-c", "pass")],
    )

    first_client, first_lock = agent._branch_model_client(object())
    second_client, second_lock = agent._branch_model_client(object())

    assert first_client is shared_client
    assert second_client is shared_client
    assert first_lock is second_lock


def test_fork_merge_summary_compaction_preserves_valid_json(tmp_path):
    repo = initialize_repo(tmp_path)
    agent = build_agent(
        repo,
        ["<final>unused</final>"],
        lambda spec: FakeModelClient(["<final>unused</final>"]),
        [(sys.executable, "-c", "pass")],
    )
    coordinator = WorktreeForkCoordinator(agent)
    summary = {
        "fork_id": "merge_compaction",
        "kind": "worktree_merge",
        "status": "merged",
        "merge_policy": "atomic_disjoint",
        "base_commit": "a" * 40,
        "target_ref": "refs/heads/main",
        "integration_commit": "b" * 40,
        "branch_count": 2,
        "succeeded": 2,
        "failed": 0,
        "duration_ms": 1,
        "validation": {
            "status": "passed",
            "results": [
                {
                    "argv": ["python", "x" * 1000, "y" * 1000],
                    "returncode": 0,
                    "duration_ms": 1,
                }
                for _ in range(8)
            ],
        },
        "conflict_paths": [],
        "cleanup": {"status": "completed", "pending_paths": []},
        "branches": [],
    }

    compact = coordinator._compact_summary(summary)
    encoded = json.dumps(compact, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    assert len(encoded) < MAX_TOOL_OUTPUT
    assert json.loads(encoded)["status"] == "merged"
