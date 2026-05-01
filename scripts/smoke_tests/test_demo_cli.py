from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_demo_cli_help():
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, str(repo_root / "demo_flashsvd_v15.py"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "FlashSVD v1.5 generation demo" in proc.stdout
    assert "--warmup-tokens" in proc.stdout
    assert "LowRankArena::" in proc.stdout
