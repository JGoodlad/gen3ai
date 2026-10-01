"""`gen3_supply_guard_v1` END TO END — the real trainer, the real producer it spawns, the real exit code.

`cf_supply_test.py` pins each part; this pins the composition on the `--debug` trainer (serverless,
the rust bridge): a live `--cf-winprob-coef` run STARTS its producer, and when that producer dies
mid-run the TRAINER exits `FATAL_SUPPLY` (5) — the code the launcher does not restart. Reverting the
callback wiring, the exit mapping or the spawn makes this fail (the run completes, or exits 1/3).

The trainer is started through `runpy` with its module name split, so no argv on this box carries
the trainer's literal name (watchers key on it).
"""
import os
import re
import signal
import subprocess
import sys
import time

import pytest

from main.exit_codes import TrainExitCode
from utils.contention import scale_timeout
from utils.paths import src_root

pytestmark = [pytest.mark.slow, pytest.mark.sim]

_RUNNER = ("import runpy, sys; sys.argv = ['trainer'] + sys.argv[1:]; "
           "runpy.run_module('main.train_rl' '_agent', run_name='__main__', alter_sys=True)")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_producer_that_dies_mid_run_stops_the_trainer_with_fatal_supply(tmp_path):
    run_dir = tmp_path / "run"
    log_path = tmp_path / "trainer.log"
    argv = [sys.executable, "-c", _RUNNER, "--debug", "--steps", "40000", "--cf-records",
            "--cf-winprob-coef", "0.5", "--win-prob-mode", "read_only",
            "--checkpoint-every-steps", "2000",
            "--cf-producer-args", "--cycle-seconds 5 --no-compile-extractor",
            "--run-dir", str(run_dir)]
    with open(log_path, "wb") as log:
        trainer = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,
                                   env={**os.environ, "PYTHONPATH": str(src_root())})
    producer = None
    try:
        deadline = time.monotonic() + scale_timeout(900)
        while time.monotonic() < deadline and trainer.poll() is None:
            text = log_path.read_text(errors="replace")
            m = re.search(r"spawned cf_producer pid (\d+)", text)
            if m and list((run_dir / "checkpoints").glob("*.zip")):
                producer = int(m.group(1))
                break
            time.sleep(2)
        assert producer is not None, log_path.read_text(errors="replace")[-4000:]
        assert _alive(producer)
        os.kill(producer, signal.SIGKILL)                  # the supply dies; the trainer must notice
        rc = trainer.wait(timeout=scale_timeout(900))
        text = log_path.read_text(errors="replace")
        assert rc == int(TrainExitCode.FATAL_SUPPLY), (rc, text[-4000:])
        assert "[SUPPLY] FATAL" in text and f"pid {producer}" in text
        assert "Training complete" not in text
    finally:
        if trainer.poll() is None:
            trainer.kill()
            trainer.wait(timeout=60)
