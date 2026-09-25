"""An INTERRUPTED `slow` test must never bank as PASS — the recorder's second writer invariant.

**The defect (2026-09-24, found by Rust core M3).** Stopping a MILESTONE run with SIGTERM made the
root `conftest.py` write PASS rows with 0.0 s durations into `designs/ops/slow_tier_status.json` for
two tests that were still IN FLIGHT. The mechanism: a test's first report is its SETUP, a setup that
succeeds classifies `pass`, and two routes then reach the bank with nothing else —

* **KeyboardInterrupt** (Ctrl-C, or anything that raises it): pytest re-raises it out of the call
  WITHOUT making a call report, then runs `pytest_sessionfinish`, whose sweep banks the setup row;
* **a SIGTERM'd xdist worker**: the controller's "worker crashed" report carries NO keywords, so the
  recorder never sees it as `slow`, and the controller's sweep banks the setup row it relayed.

The fix is by CLASS (`utils.slow_tier_status.settle`): a PASS needs a CALL phase, so a test with none
is `inconclusive` — reported on every routine run, never fatal.

WHY A SUBPROCESS. Both routes are properties of a whole pytest SESSION being stopped, which cannot be
observed from inside a test running in one. So, as `deps_guard_test.py` does, this copies the ACTUAL
root `conftest.py` into a temp tree, runs a second pytest over one finished and one sleeping `slow`
test, interrupts it once the sleeper has provably STARTED (it writes a marker — no fixed sleep), and
asserts on the rows written to a TEMP status file (`$GEN3AI_SLOW_STATUS_FILE`), never the committed
one. Both cases FAIL on the pre-fix conftest: each banks the sleeper as `pass` at 0.0 s.

`integration`: it needs out-of-process pytest (and `pytest-xdist` workers), and plays no battles —
a few seconds, so it rides the routine gate.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from utils.contention import scale_timeout
from utils.paths import repo_root, src_root
from utils.slow_tier_status import INTERRUPTED_DETAIL, SCHEMA, make_row

pytestmark = pytest.mark.integration

_CONFTEST = repo_root() / "conftest.py"
_DONE = "sleeper_test.py::test_a_finished_slow_test"
_IN_FLIGHT = "sleeper_test.py::test_an_in_flight_slow_test"

_SLEEPER = '''\
import os, time, pytest

@pytest.mark.slow
def test_a_finished_slow_test():
    pass

@pytest.mark.slow
def test_an_in_flight_slow_test():
    with open(os.environ["SLEEPER_STARTED"] + ".tmp", "w") as f:
        f.write(str(os.getpid()))
    os.replace(os.environ["SLEEPER_STARTED"] + ".tmp", os.environ["SLEEPER_STARTED"])
    time.sleep(600)       # killed long before this; never a pass
'''


def _interrupt_a_session(tmp_path: Path, *, xdist: bool, seed: "dict | None" = None) -> dict:
    """Run the real conftest over the sleeper, interrupt it in flight, return the rows written.

    ``seed`` pre-plants ``{nodeid: row}`` in the temp status file, as a previous run would have."""
    shutil.copy(_CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: slow\n")
    (tmp_path / "sleeper_test.py").write_text(_SLEEPER)
    status, started = tmp_path / "status.json", tmp_path / "started.pid"
    if seed:
        status.write_text(json.dumps({"schema": SCHEMA, "tests": seed}))
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("PYTEST_") and k != "GEN3AI_SKIP_SLOW_STATUS_RECORD"}
    env.update(GEN3AI_SLOW_STATUS_FILE=str(status), SLEEPER_STARTED=str(started),
               GEN3AI_SKIP_DEPS_GUARD="1", PYTHONPATH=str(src_root()))
    # `-p no:randomly` is not needed: the finished test is first in the file, and under xdist it is
    # on whichever worker — its row is asserted either way. `-n 2` rather than 1 so the two tests
    # can land on separate workers, the shape of the real tier run.
    argv = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "sleeper_test.py"]
    if xdist:
        argv += ["-n", "2"]
    proc = subprocess.Popen(
        argv, cwd=str(tmp_path), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
        # A parent started in the background inherits SIGINT as IGNORED, and Python then leaves it
        # ignored — the KeyboardInterrupt case would silently become a 600 s sleep. Restore default.
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    try:
        deadline = time.monotonic() + scale_timeout(60.0)
        while not started.exists():
            assert proc.poll() is None, f"the sleeper session exited early:\n{proc.stdout.read()}"
            assert time.monotonic() < deadline, "the sleeper never started (INCONCLUSIVE)"
            time.sleep(0.05)
        if xdist:
            os.kill(int(started.read_text()), signal.SIGTERM)    # the WORKER, not the controller
        else:
            proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=scale_timeout(60.0))
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    assert status.exists(), f"the interrupted session recorded nothing at all:\n{out}"
    return json.loads(status.read_text())["tests"]


@pytest.mark.parametrize("xdist", [False, True], ids=["serial-SIGINT", "xdist-worker-SIGTERM"])
def test_an_interrupted_slow_test_banks_INCONCLUSIVE_never_PASS(tmp_path, xdist):
    rows = _interrupt_a_session(tmp_path, xdist=xdist)
    assert rows[_IN_FLIGHT]["status"] == "inconclusive", (
        f"an in-flight test was banked {rows[_IN_FLIGHT]['status']!r} — a gate that reads an "
        f"interrupted test as green is lying: {rows[_IN_FLIGHT]}")
    assert rows[_IN_FLIGHT]["detail"] == INTERRUPTED_DETAIL
    # ...and the demotion is by class, not a blanket one: a test that finished its CALL keeps PASS.
    assert rows[_DONE]["status"] == "pass", rows[_DONE]


def test_an_interrupted_RERUN_of_a_RED_test_leaves_it_RED(tmp_path):
    """The sticky-red rule through the REAL recorder: a test recorded `fail`, then re-run and
    killed in flight, must still read `fail` — the interrupted attempt is kept as `held_over`, and
    the gate stays red. Before `merge_row` the re-run banked `inconclusive` over the red."""
    red = make_row("fail", commit="c0ffee", duration_s=3.0, contention=1.0,
                   detail="AssertionError: 2 != 3")
    rows = _interrupt_a_session(tmp_path, xdist=False, seed={_IN_FLIGHT: red})
    assert rows[_IN_FLIGHT]["status"] == "fail" and rows[_IN_FLIGHT]["commit"] == "c0ffee", (
        f"an interrupted re-run cleared a recorded red: {rows[_IN_FLIGHT]}")
    assert rows[_IN_FLIGHT]["held_over"]["status"] == "inconclusive"
    assert rows[_IN_FLIGHT]["held_over"]["detail"] == INTERRUPTED_DETAIL
    assert rows[_DONE]["status"] == "pass", rows[_DONE]
