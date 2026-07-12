import subprocess
import sys
from pathlib import Path

import CodeY


def test_public_package_and_cli():
    assert CodeY.CodeYAgent.__name__ == "CodeYAgent"
    result = subprocess.run([sys.executable, "-m", "CodeY", "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "--skill" in result.stdout


def test_no_legacy_name_in_project_text():
    root = Path(__file__).resolve().parents[1]
    extensions = {".py", ".md", ".toml", ".json", ".yaml", ".yml", ".example"}
    offenders = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in extensions:
            continue
        if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
            continue
        if "p" + "ico" in path.read_text(encoding="utf-8", errors="ignore").casefold():
            offenders.append(path.relative_to(root).as_posix())
    assert offenders == []
