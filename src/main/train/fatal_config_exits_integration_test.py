"""Post-parse config refusals with REAL processes: the trainer's own exit code, the real validator.

* a `--bot-weights` typo exits ``FATAL_CONFIG`` (3) — measured 2026-09-30 on the parent commit:
  rc 1 (CRASH), which the launcher restarts into the same typo.

The trainer is started through `runpy` with its module name split, so no argv on this box carries
the trainer's literal name (watchers key on it). Every run dir is under `tmp_path`.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from main.exit_codes import TrainExitCode
from utils.contention import scale_timeout
from utils.paths import repo_root, src_root

pytestmark = [pytest.mark.integration]

_RUNNER = ("import runpy, sys; sys.argv = ['trainer'] + sys.argv[1:]; "
           "runpy.run_module('main.train_rl' '_agent', run_name='__main__', alter_sys=True)")


def _trainer(tmp_path, *args, timeout=600):
    """Run the real trainer entry point to its exit; return ``(rc, log_text)``."""
    log = tmp_path / "trainer.log"
    with open(log, "wb") as fh:
        proc = subprocess.Popen([sys.executable, "-c", _RUNNER, "--debug", *args],
                                stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(repo_root()),
                                env={**os.environ, "PYTHONPATH": str(src_root()),
                                     "CUDA_VISIBLE_DEVICES": ""})
        try:
            rc = proc.wait(timeout=scale_timeout(timeout))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=60)
    return rc, log.read_text(errors="replace")



def test_a_bot_weights_typo_exits_FATAL_CONFIG_not_CRASH(tmp_path):
    rc, text = _trainer(tmp_path, "--steps", "1000", "--bot-weights", "stallr=3",
                        "--run-dir", str(tmp_path / "run"), timeout=180)
    assert rc == int(TrainExitCode.FATAL_CONFIG), (rc, text[-3000:])
    assert "unknown --bot-weights names ['stallr']" in text
    assert not (tmp_path / "run").exists(), "refused before the run dir exists"
