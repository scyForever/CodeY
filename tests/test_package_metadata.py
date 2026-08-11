import subprocess
import sys
from pathlib import Path

import CodeY
from CodeY import cli


ROOT = Path(__file__).parents[1]


def test_public_package_and_cli_help():
    assert CodeY.CodeYAgent
    result = subprocess.run(
        [sys.executable, "-m", "CodeY", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--skill" in result.stdout
    assert "--evolution-mode" in result.stdout
    assert "--max-fork-branches" in result.stdout
    assert "--max-parallel-branches" in result.stdout


def test_package_metadata_and_metrics_imports():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'codey = "CodeY.cli:main"' in pyproject
    assert 'include = ["CodeY*"]' in pyproject
    for path in (ROOT / "scripts").glob("*.py"):
        assert "from CodeY.evaluation." in path.read_text(encoding="utf-8")


def test_cli_builds_distinct_clients_for_parallel_fork_branches(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    created = []

    def build_client(args, model_override=None):
        del args, model_override
        client = CodeY.FakeModelClient([])
        created.append(client)
        return client

    monkeypatch.setattr(cli, "_build_model_client", build_client)
    args = cli.build_arg_parser().parse_args(
        ["--cwd", str(tmp_path), "--provider", "ollama", "--skill", "off"]
    )

    agent = cli.build_agent(args)
    branch_a = agent.model_client_factory(None)
    branch_b = agent.model_client_factory(None)

    assert branch_a is not agent.model_client
    assert branch_b is not agent.model_client
    assert branch_a is not branch_b
    assert len(created) == 3
