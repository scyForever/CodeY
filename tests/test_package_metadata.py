import subprocess
import sys
from pathlib import Path

import CodeY


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


def test_package_metadata_and_metrics_imports():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'codey = "CodeY.cli:main"' in pyproject
    assert 'include = ["CodeY*"]' in pyproject
    for path in (ROOT / "scripts").glob("*.py"):
        assert "from CodeY.evaluation.metrics import" in path.read_text(encoding="utf-8")
