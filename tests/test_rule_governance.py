import base64
import difflib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from CodeY import CodeYAgent, FakeModelClient, SessionStore, WorkspaceContext
from CodeY.cli import main as codey_main
from CodeY.rules.cli import main as rules_main
from CodeY.rules.discovery import MANAGED_BEGIN, RuleScanner
from CodeY.rules.patches import RulePatchStore
from CodeY.rules.runners import ExternalAgentRunner, TrialRequest


def git(root, *args):
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def init_repository(root):
    git(root, "init")
    git(root, "config", "user.name", "CodeY Tests")
    git(root, "config", "user.email", "codey-tests@localhost")
    (root / ".gitignore").write_text(".codey/\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "initial")


def test_rule_scanner_records_ecosystem_scope_and_config_risk(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Root instruction.\n", encoding="utf-8")
    nested = tmp_path / "src" / "api"
    nested.mkdir(parents=True)
    (nested / "AGENTS.md").write_text("API instruction.\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("Claude instruction.\n", encoding="utf-8")
    cursor = tmp_path / ".cursor" / "rules"
    cursor.mkdir(parents=True)
    (cursor / "python.mdc").write_text(
        "---\ndescription: Python rules\nglobs: '**/*.py'\nalwaysApply: false\n---\nUse Python.\n",
        encoding="utf-8",
    )
    cursor_skill = tmp_path / ".cursor" / "skills" / "demo"
    cursor_skill.mkdir(parents=True)
    (cursor_skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill\n---\n\nFollow the skill body.\n",
        encoding="utf-8",
    )
    codex = tmp_path / ".codex"
    codex.mkdir()
    (codex / "config.toml").write_text(
        'sandbox_mode = "danger-full-access"\napproval_policy = "never"\n',
        encoding="utf-8",
    )
    init_repository(tmp_path)

    inventory = RuleScanner(tmp_path).scan()
    by_path = {source.path: source for source in inventory.sources}

    assert inventory.revision == git(tmp_path, "rev-parse", "HEAD")
    assert by_path["AGENTS.md"].scope == "."
    assert by_path["src/api/AGENTS.md"].scope == "src/api"
    assert by_path["src/api/AGENTS.md"].precedence > by_path["AGENTS.md"].precedence
    assert by_path[".cursor/rules/python.mdc"].metadata["globs"] == "**/*.py"
    assert by_path[".cursor/skills/demo/SKILL.md"].content == "Follow the skill body."
    assert by_path[".codex/config.toml"].deployable is False
    assert "permissive_agent_config" in {issue.code for issue in inventory.issues}


@pytest.mark.skipif(os.name == "nt", reason="creating symlinks is not reliably permitted on Windows")
def test_rule_scanner_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").symlink_to(outside)

    inventory = RuleScanner(tmp_path).scan()

    assert inventory.sources == ()
    assert [issue.code for issue in inventory.issues] == ["path_escape"]


def test_secret_shaped_rule_blocks_plan(tmp_path):
    (tmp_path / "AGENTS.md").write_text("api_key = should-not-enter-patch\n", encoding="utf-8")
    init_repository(tmp_path)
    inventory = RuleScanner(tmp_path).scan()

    with pytest.raises(ValueError, match="secret_shaped_content"):
        RulePatchStore(tmp_path).create_plan(inventory, targets=("codey",))


def test_plan_apply_and_exact_rollback(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Keep tests deterministic.\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("Read implementation before edits.\n", encoding="utf-8")
    init_repository(tmp_path)
    store = RulePatchStore(tmp_path)
    patch = store.create_plan(
        RuleScanner(tmp_path).scan(),
        targets=("codey", "codex", "claude", "cursor"),
        objective="Share validated repository rules.",
    )

    assert patch["status"] == "review_required"
    assert [item["path"] for item in patch["artifacts"]] == [
        ".codey/rules/active.md",
        "AGENTS.md",
        "CLAUDE.md",
        ".cursor/rules/codey-managed.mdc",
    ]
    assert all(MANAGED_BEGIN in item["candidate_content"] for item in patch["artifacts"])
    assert "Read implementation before edits." in patch["artifacts"][1]["candidate_content"]
    with pytest.raises(ValueError, match="explicit approval"):
        store.apply(patch["id"])

    active = store.apply(patch["id"], approved=True)

    assert active["status"] == "active"
    assert MANAGED_BEGIN in (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert (tmp_path / ".codey" / "rules" / "active.md").is_file()

    rolled_back = store.rollback(patch["id"], approved=True)

    assert rolled_back["status"] == "rolled_back"
    assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8") == "Keep tests deterministic.\n"
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == "Read implementation before edits.\n"
    assert not (tmp_path / ".codey" / "rules" / "active.md").exists()
    assert not (tmp_path / ".cursor" / "rules" / "codey-managed.mdc").exists()


def test_replanning_does_not_reingest_generated_cursor_wrapper(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("Keep changes scoped.\n", encoding="utf-8")
    init_repository(tmp_path)
    store = RulePatchStore(tmp_path)
    first = store.create_plan(
        RuleScanner(tmp_path).scan(),
        targets=("codey", "codex", "claude", "cursor"),
    )
    store.apply(first["id"], approved=True)

    inventory = RuleScanner(tmp_path).scan()
    generated = next(
        source
        for source in inventory.sources
        if source.path == ".cursor/rules/codey-managed.mdc"
    )
    second = store.create_plan(inventory, targets=("codey",))

    assert generated.content == ""
    assert ".cursor/rules/codey-managed.mdc" not in {
        source["path"] for source in second["source_refs"]
    }


def test_apply_refuses_source_and_target_drift(tmp_path):
    source = tmp_path / "CLAUDE.md"
    source.write_text("Original rule.\n", encoding="utf-8")
    init_repository(tmp_path)
    store = RulePatchStore(tmp_path)
    patch = store.create_plan(RuleScanner(tmp_path).scan(), targets=("codex",))
    source.write_text("Changed rule.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source changed"):
        store.apply(patch["id"], approved=True)

    source.write_text("Original rule.\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("Concurrent target.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="target drifted"):
        store.apply(patch["id"], approved=True)


def test_active_rule_patch_has_audited_prompt_section(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Use the repository verifier.\n", encoding="utf-8")
    init_repository(tmp_path)
    store = RulePatchStore(tmp_path)
    patch = store.create_plan(RuleScanner(tmp_path).scan(), targets=("codey",))
    store.apply(patch["id"], approved=True)
    agent = CodeYAgent(
        model_client=FakeModelClient(["<final>done</final>"]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".codey" / "sessions"),
        skill_mode="off",
        feature_flags={"self_evolution": False},
    )
    agent.route_task("inspect")

    prompt, metadata = agent._build_prompt_and_metadata("inspect")

    assert "Reviewed repository rule patch:\n" in prompt
    assert "Use the repository verifier." in prompt
    assert prompt.index("Reviewed repository rule patch") < prompt.index("Selected task route")
    assert metadata["rules"]["loaded"] is True
    assert metadata["rules"]["state"] == "active"

    active = tmp_path / ".codey" / "rules" / "active.md"
    active.write_text("Unreviewed replacement.\n", encoding="utf-8")
    prompt, metadata = agent._build_prompt_and_metadata("inspect again")

    assert "Unreviewed replacement." not in prompt
    assert "Reviewed repository rule patch:\n- unavailable" in prompt
    assert metadata["rules"]["loaded"] is False
    assert metadata["rules"]["state"] == "unavailable"


def test_rules_cli_scan_plan_status_and_approval(tmp_path, capsys):
    (tmp_path / "CLAUDE.md").write_text("Keep edits scoped.\n", encoding="utf-8")
    init_repository(tmp_path)

    assert rules_main(["--cwd", str(tmp_path), "scan", "--json"]) == 0
    scan_payload = json.loads(capsys.readouterr().out)
    assert scan_payload["sources"][0]["path"] == "CLAUDE.md"

    assert codey_main(["--cwd", str(tmp_path), "rules", "scan", "--json"]) == 0
    routed_scan = json.loads(capsys.readouterr().out)
    assert routed_scan["inventory_id"] == scan_payload["inventory_id"]

    assert rules_main(["--cwd", str(tmp_path), "plan", "--target", "codex", "--json"]) == 0
    plan_payload = json.loads(capsys.readouterr().out)
    patch_id = plan_payload["id"]
    assert plan_payload["status"] == "review_required"
    assert plan_payload["dirty_paths_at_plan"] == []
    assert plan_payload["source_refs"][0]["path"] == "CLAUDE.md"
    assert len(plan_payload["source_refs"][0]["sha256"]) == 64
    assert plan_payload["targets"][0]["before_sha256"] is None

    assert rules_main(["--cwd", str(tmp_path), "apply", patch_id]) == 2
    assert "explicit approval" in capsys.readouterr().err
    assert rules_main(["--cwd", str(tmp_path), "apply", patch_id, "--approve", "--json"]) == 0
    active_payload = json.loads(capsys.readouterr().out)
    assert active_payload["status"] == "active"


class ScriptedRunner(ExternalAgentRunner):
    def __init__(self, *args, script, **kwargs):
        super().__init__(*args, **kwargs)
        self.script = script

    def probe(self, runner):
        return {
            "runner": runner,
            "available": True,
            "executable": Path(sys.executable).name,
            "version": "scripted-test-runner",
            "capabilities": ["inspect", "edit"],
        }

    def _build_argv(self, request, workspace):
        return [sys.executable, "-c", self.script], workspace


def make_runner_patch(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("Read before changing.\n", encoding="utf-8")
    init_repository(tmp_path)
    store = RulePatchStore(tmp_path)
    patch = store.create_plan(RuleScanner(tmp_path).scan(), targets=("codex",))
    return store, patch


def test_isolated_inspect_trial_keeps_original_workspace_clean(tmp_path):
    store, patch = make_runner_patch(tmp_path)
    before = git(tmp_path, "status", "--short")
    runner = ScriptedRunner(tmp_path, patch_store=store, script="print('inspected')")

    result = runner.run(
        TrialRequest(
            patch_id=patch["id"],
            runner="codex",
            task="Inspect repository rules.",
            mode="inspect",
        )
    )

    assert result["status"] == "completed"
    assert result["diff_stats"]["changed_files"] == 0
    assert git(tmp_path, "status", "--short") == before
    stored = store.list_trials(patch["id"])[0]
    assert stored["selected_variant"] == "candidate"
    assert stored["artifact_dir"] == result["artifact_dir"]


def test_inspect_write_fails_and_delegate_returns_bounded_diff(tmp_path):
    store, patch = make_runner_patch(tmp_path)
    script = "from pathlib import Path; Path('result.txt').write_text('done', encoding='utf-8')"
    runner = ScriptedRunner(tmp_path, patch_store=store, script=script)

    inspected = runner.run(
        TrialRequest(
            patch_id=patch["id"],
            runner="codex",
            task="Inspect only.",
            mode="inspect",
        )
    )
    delegated = runner.run(
        TrialRequest(
            patch_id=patch["id"],
            runner="codex",
            task="Create result.txt.",
            mode="edit",
        )
    )

    assert inspected["status"] == "unexpected_changes"
    assert delegated["status"] == "completed"
    assert delegated["diff_stats"]["changed_files"] == 1
    patch_path = tmp_path / delegated["artifact_dir"] / "changes.patch"
    assert "result.txt" in patch_path.read_text(encoding="utf-8")
    assert not (tmp_path / "result.txt").exists()


def test_canary_assignment_is_stable(tmp_path):
    runner = ExternalAgentRunner(tmp_path)
    request = TrialRequest(
        patch_id="rulepatch_0123456789abcdef",
        runner="codex",
        task="Inspect.",
        variant="canary",
        cohort_key="task-42",
        candidate_fraction=0.5,
    )

    assert runner._select_variant(request) == runner._select_variant(request)


def test_claude_runner_uses_valid_empty_mcp_config(tmp_path, monkeypatch):
    runner = ExternalAgentRunner(tmp_path)
    monkeypatch.setattr(
        runner,
        "_resolve_executable",
        lambda name, required: ("claude", "test"),
    )
    request = TrialRequest(
        patch_id="rulepatch_0123456789abcdef",
        runner="claude",
        task="Inspect.",
    )

    argv, _ = runner._build_argv(request, tmp_path)

    assert argv[argv.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert argv[argv.index("--tools") + 1] == "Read,Glob,Grep"
    assert "Inspect." not in argv


def test_patch_load_rejects_candidate_and_diff_tampering(tmp_path):
    store, patch = make_runner_patch(tmp_path)
    path = store.patch_root / f"{patch['id']}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifacts"][0]["candidate_content"] += "\nUnreviewed instruction.\n"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate hash mismatch"):
        store.load(patch["id"])

    path.write_text(json.dumps(patch), encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifacts"][0]["diff"] = "benign-looking diff"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="review diff mismatch"):
        store.load(patch["id"])

    path.write_text(json.dumps(patch), encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact = payload["artifacts"][0]
    forged_before = b"Forged rollback baseline.\n"
    artifact["before_content"] = forged_before.decode("utf-8")
    artifact["before_bytes_b64"] = base64.b64encode(forged_before).decode("ascii")
    artifact["before_sha256"] = hashlib.sha256(forged_before).hexdigest()
    artifact["diff"] = "".join(
        difflib.unified_diff(
            artifact["before_content"].splitlines(keepends=True),
            artifact["candidate_content"].splitlines(keepends=True),
            fromfile=f"a/{artifact['path']}",
            tofile=f"b/{artifact['path']}",
        )
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity mismatch"):
        store.load(patch["id"])

    path.write_text(json.dumps(patch), encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["repository"].update(
        {
            "dirty_at_plan": True,
            "git_status_at_plan": "?? unrelated.txt",
            "dirty_paths_at_plan": ["unrelated.txt"],
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity mismatch"):
        store.load(patch["id"])


def test_interrupted_apply_and_rollback_are_recovered(tmp_path):
    store, patch = make_runner_patch(tmp_path)
    artifact = patch["artifacts"][0]
    target = tmp_path / artifact["path"]

    target.write_text(artifact["candidate_content"], encoding="utf-8", newline="\n")
    store._begin_transaction("apply", patch)
    active = store.apply(patch["id"], approved=True)

    assert active["status"] == "active"
    assert target.is_file()
    assert not store.transaction_path.exists()

    target.unlink()
    store._begin_transaction("rollback", active)
    rolled_back = store.rollback(patch["id"], approved=True)

    assert rolled_back["status"] == "rolled_back"
    assert not target.exists()
    assert not store.transaction_path.exists()


def test_transaction_recovery_preserves_conflicting_user_change(tmp_path):
    store, patch = make_runner_patch(tmp_path)
    artifact = patch["artifacts"][0]
    target = tmp_path / artifact["path"]
    store._begin_transaction("apply", patch)
    target.write_text("New user content after interruption.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="transaction recovery conflict"):
        store.apply(patch["id"], approved=True)

    assert target.read_text(encoding="utf-8") == "New user content after interruption.\n"
    assert store.transaction_path.is_file()


def test_secret_environment_value_in_diff_is_not_saved(tmp_path, monkeypatch):
    store, patch = make_runner_patch(tmp_path)
    secret = "opaque-secret-value-94721"
    monkeypatch.setenv("TEST_SECRET_TOKEN", secret)
    script = (
        "from pathlib import Path; "
        f"Path('result.txt').write_text({secret!r}, encoding='utf-8')"
    )
    runner = ScriptedRunner(tmp_path, patch_store=store, script=script)

    result = runner.run(
        TrialRequest(
            patch_id=patch["id"],
            runner="codex",
            task="Create a result.",
            mode="edit",
        )
    )

    assert result["status"] == "blocked_sensitive_diff"
    assert result["diff_saved"] is False
    assert not (tmp_path / result["artifact_dir"] / "changes.patch").exists()


def test_output_limit_terminates_runner_and_records_violation(tmp_path):
    store, patch = make_runner_patch(tmp_path)
    runner = ScriptedRunner(
        tmp_path,
        patch_store=store,
        script="import sys; sys.stdout.write('x' * 200000); sys.stdout.flush()",
    )

    result = runner.run(
        TrialRequest(
            patch_id=patch["id"],
            runner="codex",
            task="Produce output.",
            max_output_bytes=1024,
        )
    )

    assert result["status"] == "output_limit_exceeded"
    assert "max_output_bytes" in result["budget_violations"]
    assert result["output_truncated"] is True


def test_output_limit_counts_stdout_and_stderr_together(tmp_path):
    store, patch = make_runner_patch(tmp_path)
    runner = ScriptedRunner(
        tmp_path,
        patch_store=store,
        script=(
            "import sys; "
            "sys.stdout.write('o' * 700); sys.stdout.flush(); "
            "sys.stderr.write('e' * 700); sys.stderr.flush()"
        ),
    )

    result = runner.run(
        TrialRequest(
            patch_id=patch["id"],
            runner="codex",
            task="Produce split output.",
            max_output_bytes=1024,
        )
    )

    assert result["status"] == "output_limit_exceeded"
    assert "max_output_bytes" in result["budget_violations"]


def test_large_stdin_does_not_block_runner_or_raise_broken_pipe(tmp_path):
    store, patch = make_runner_patch(tmp_path)
    runner = ScriptedRunner(tmp_path, patch_store=store, script="pass")

    result = runner.run(
        TrialRequest(
            patch_id=patch["id"],
            runner="codex",
            task="x" * (1024 * 1024),
            timeout_seconds=5,
        )
    )

    assert result["status"] == "completed"


@pytest.mark.skipif(os.name != "nt", reason="taskkill fallback is Windows-specific")
def test_windows_taskkill_failure_falls_back_to_parent_kill(monkeypatch):
    class Process:
        pid = 12345

        def __init__(self):
            self.killed = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], returncode=1),
    )
    process = Process()

    ExternalAgentRunner._terminate_process_tree(process)

    assert process.killed is True


def test_trials_disable_repository_git_hooks_and_filters(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("Read before changing.\n", encoding="utf-8")
    (tmp_path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    (tmp_path / ".gitattributes").write_text("*.txt filter=sideeffect\n", encoding="utf-8")
    init_repository(tmp_path)
    store = RulePatchStore(tmp_path)
    patch = store.create_plan(RuleScanner(tmp_path).scan(), targets=("codex",))

    sentinel = tmp_path.parent / f"{tmp_path.name}-git-side-effect"
    hooks = tmp_path / ".test-hooks"
    hooks.mkdir()
    escaped_sentinel = sentinel.as_posix().replace("'", "'\"'\"'")
    hook_body = f"#!/bin/sh\nprintf hook >> '{escaped_sentinel}'\n"
    for name in ("post-checkout", "pre-commit", "prepare-commit-msg", "commit-msg"):
        hook = hooks / name
        hook.write_text(hook_body, encoding="utf-8", newline="\n")
        hook.chmod(0o755)
    filter_script = hooks / "filter.sh"
    filter_script.write_text(
        f"#!/bin/sh\nprintf filter >> '{escaped_sentinel}'\ncat\n",
        encoding="utf-8",
        newline="\n",
    )
    filter_script.chmod(0o755)
    git(tmp_path, "config", "core.hooksPath", str(hooks))
    git(tmp_path, "config", "filter.sideeffect.clean", filter_script.as_posix())
    git(tmp_path, "config", "filter.sideeffect.smudge", "cat")
    git(tmp_path, "config", "filter.sideeffect.required", "true")

    runner = ScriptedRunner(tmp_path, patch_store=store, script="print('safe')")
    result = runner.run(
        TrialRequest(
            patch_id=patch["id"],
            runner="codex",
            task="Inspect safely.",
        )
    )

    assert result["status"] == "completed"
    assert not sentinel.exists()


def test_dirty_target_is_never_allowed_as_trial_baseline(tmp_path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("Committed rule.\n", encoding="utf-8")
    init_repository(tmp_path)
    target.write_text("Uncommitted rule.\n", encoding="utf-8")
    store = RulePatchStore(tmp_path)
    patch = store.create_plan(RuleScanner(tmp_path).scan(), targets=("claude",))
    runner = ScriptedRunner(tmp_path, patch_store=store, script="print('unused')")

    with pytest.raises(ValueError, match="target was dirty"):
        runner.run(
            TrialRequest(
                patch_id=patch["id"],
                runner="claude",
                task="Inspect.",
                allow_dirty_base=True,
            )
        )


def test_trial_refuses_ambiguous_dirty_baseline(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("Read before changing.\n", encoding="utf-8")
    init_repository(tmp_path)
    (tmp_path / "unrelated.txt").write_text("uncommitted\n", encoding="utf-8")
    store = RulePatchStore(tmp_path)
    patch = store.create_plan(RuleScanner(tmp_path).scan(), targets=("codex",))
    runner = ScriptedRunner(tmp_path, patch_store=store, script="print('unused')")

    with pytest.raises(ValueError, match="dirty workspace"):
        runner.run(
            TrialRequest(
                patch_id=patch["id"],
                runner="codex",
                task="Inspect.",
            )
        )
