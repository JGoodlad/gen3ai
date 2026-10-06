"""`scripts/ops/watch_run.sh`'s failure grep fires on REAL failure lines and ONLY those.

2026-10-06: the grep was `-qiE 'OutOfMemory|CUDA out of memory|FATAL|Traceback|FATAL_CONFIG'`, case
insensitive and unanchored, so it false-fired on every current run: Python's harmless "Enable
tracemalloc to get the object allocation traceback" warning, the `[CompileCanary] armed` banner (which
says "before it FATALs") and the `[SUPPLY] ... else FATAL_SUPPLY (5)` banner. The training agent had to
run a patched copy. Each banner test below fails on a revert of the anchored pattern.
"""
import os
import subprocess
import time

import pytest

from utils.paths import repo_path

SCRIPT = repo_path("scripts", "ops", "watch_run.sh")

BANNERS = [
    "/x/site-packages/foo.py:12: ResourceWarning: unclosed file\n"
    "  Enable tracemalloc to get the object allocation traceback",
    "RuntimeWarning: Enable tracemalloc to get the object allocation traceback",
    "🐤 [CompileCanary] armed: R1 (the learner micro-step) compiled vs eager on the K9 learner golden's "
    "real labelled rows at update 1, then every 50 updates (loss + every policy gradient, B=1024), at the "
    "startup gate's bars; a disagreement is confirmed before it FATALs",
    "🛡️  [SUPPLY] --cf-coef: must deliver labels within 3 consecutive live cycles, else FATAL_SUPPLY (5)",
    "⚠️  [SUPPLY] --cf-coef: the in-flight supply guard is DISABLED (--supply-starve-cycles cf=0)",
]

REAL_FAILURES = [
    "Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\nValueError: boom",
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
    "[CompileCanary] FATAL: compiled R1 disagrees with eager (confirmed)",
    "\n[SUPPLY] FATAL: --cf-coef is live but delivered NO labels for 3 consecutive cycles",
    "\n[ModelVersion] FATAL: checkpoint arch-family mismatch",
    "🛑 [TorchFloor] FATAL: torch 2.5.1 < 2.8",
    "[LearnerLifecycle] FATAL — GLOBAL RESEED: np.random.seed called",
    "🛑 FATAL ERROR DETECTED - FAILING FAST",
]


def _run(tmp_path, log_text, *, seconds):
    run = tmp_path / "models" / "r"
    run.mkdir(parents=True)
    (run / "launcher_child.log").write_text(log_text + "\n")
    status = tmp_path / "status.txt"
    env = dict(os.environ, GEN3AI_MODELS_DIR=str(tmp_path / "models"))
    proc = subprocess.Popen(
        ["bash", str(SCRIPT), "r", "--no-pid-check", "--status-file", str(status), "--interval", "1"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    return status.read_text() if status.exists() else ""


@pytest.mark.parametrize("line", BANNERS)
def test_a_harmless_banner_does_not_fire(tmp_path, line):
    status = _run(tmp_path, line, seconds=4)
    assert "FAILURE" not in status, status
    assert " ok step=" in status, status      # the watcher ticked and kept going


@pytest.mark.parametrize("line", REAL_FAILURES)
def test_a_real_failure_line_fires(tmp_path, line):
    t0 = time.time()
    status = _run(tmp_path, line, seconds=20)
    assert "FAILURE: error text in child log" in status, status
    assert "WATCHER EXIT" in status
    assert time.time() - t0 < 15
