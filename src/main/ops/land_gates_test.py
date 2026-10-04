"""`scripts/land.sh`'s gate step must FAIL the landing when a gate fails.

Until 2026-10-04 the gates ran as `( ruff; mypy; pytest; echo OK ) || { GATE FAILED; exit 1; }`. Bash
ignores `set -e` inside the left side of `||`, so the subshell's status was the closing `echo`'s and no
gate could ever stop a push (found when pytest refused a bare worktree with a UsageError and the step
still printed OK). These tests drive the real script, gates only (`_LAND_GATES_ONLY`), against a
throwaway tree whose gate fails, so nothing is fetched, pushed or removed.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from utils.paths import repo_path

LAND = repo_path("scripts", "land.sh")


def _run_gates(tree: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ,
               _LAND_GATES_ONLY="1",
               _LAND_MAIN_CHECKOUT=str(tree),
               GEN3AI_PYTHON=sys.executable,
               PYTHONPATH="")
    return subprocess.run(["bash", str(LAND), "no-such-branch", str(tree)],
                          cwd=tree, env=env, capture_output=True, text=True, timeout=120)


def _tree(tmp_path: Path, *, ruff_bad: bool, gate_bad: bool) -> Path:
    model = tmp_path / "src" / "agents" / "model"
    model.mkdir(parents=True)
    (model / "__init__.py").write_text("")
    (tmp_path / "src" / "main").mkdir()
    (tmp_path / "src" / "utils").mkdir()
    if ruff_bad:
        (model / "bad.py").write_text("value = undefined_name\n")  # ruff F821
    body = "assert False, 'planted gate failure'" if gate_bad else "pass"
    (tmp_path / "src" / "planted_gate_test.py").write_text(f"def test_planted():\n    {body}\n")
    return tmp_path


def test_a_ruff_failure_stops_the_landing(tmp_path):
    proc = _run_gates(_tree(tmp_path, ruff_bad=True, gate_bad=False))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "GATE FAILED (ruff)" in proc.stdout
    assert "gates: ruff + mypy + src/*_gate_test.py OK" not in proc.stdout


def test_a_static_gate_failure_stops_the_landing(tmp_path):
    # The incident's shape: the LAST gate before the closing echo fails.
    proc = _run_gates(_tree(tmp_path, ruff_bad=False, gate_bad=True))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "GATE FAILED (src/*_gate_test.py)" in proc.stdout
    assert "gates: ruff + mypy + src/*_gate_test.py OK" not in proc.stdout


def test_green_gates_pass(tmp_path):
    # The control: the same tree with nothing planted reaches the seam and exits 0.
    proc = _run_gates(_tree(tmp_path, ruff_bad=False, gate_bad=False))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "gates: ruff + mypy + src/*_gate_test.py OK" in proc.stdout
