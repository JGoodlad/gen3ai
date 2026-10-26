"""`scripts/ops/watch_run.sh` knows a GPU WINDOW (T46) pauses the run on purpose (review 2026-10-20).

A window holds the run with no step progress for up to its `--max-window-min` (the doc's example: 130 min),
far past the watcher's 35-min wedge limit, so the unaware watcher declared `FAILURE: WEDGED`, exited, and the
run went unwatched from its resume on; and the window's HOLD (the one state that needs a human) said nothing.
Each assertion fails on a revert of the window block in the script.

TIMING (measured 12.6 s for the one test it was before, 2026-10-26, against the routine tier's 8 s bar): the watcher
runs at a 0.5 s interval with a 1 s wedge limit, and "an open window is not a wedge" is proved by the status file's OWN
timestamps, not by a sleep. `now - last_move` is whole seconds and the wedge fires at `> 1`, so an unaware watcher
ticking at a status stamp >= 2 s after its last progress has already died; a `window=` line stamped >= 3 s after the
first tick is therefore proof it did not read the window as a wedge, however slowly the box scheduled the ticks.
"""
import json
import time

from main.launcher.gpu_window import protocol as P
from main.ops.watch_run_resume_test import Harness, harness  # noqa: F401 — the fixture


def _window(run_dir, phase, reason="r"):
    P.write_json_atomic(P.window_root(run_dir) / P.CURRENT_FILE,
                        {"window_id": "20261020T120000_abc123", "phase": phase, "reason": reason})


FAST = ("--wedge-seconds", "1", "--interval", "0.5")        # a later --interval wins over the harness's 1 s


def _stamp(line):
    """The status line's own `[YYYY-MM-DD HH:MM:SS]`, as epoch seconds (whole seconds, like the script's `date +%s`)."""
    return time.mktime(time.strptime(line[1:20], "%Y-%m-%d %H:%M:%S"))


def _lines(h, needle):
    return [ln for ln in h.status().splitlines() if needle in ln]


def _wait_for_lines(h, needle, n, timeout=25):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(_lines(h, needle)) >= n:
            return
        if h.proc.poll() is not None:
            break
        time.sleep(0.1)
    raise AssertionError(f"never saw {n} x {needle!r} in:\n{h.status()}")


def test_an_open_window_is_not_a_wedge(harness):  # noqa: F811
    h = harness(extra=FAST)
    h.wait_for("ok step=1000")
    t0 = _stamp(_lines(h, "ok step=1000")[0])
    _window(h.run, P.CLAIMED)
    h.wait_for("window=claimed")
    # a window= tick stamped >= 3 s after the first tick: an unaware watcher (wedge limit 1 s) was dead by +2 s
    deadline = time.time() + 25
    while time.time() < deadline and not any(_stamp(ln) >= t0 + 3 for ln in _lines(h, "window=claimed")):
        assert h.proc.poll() is None, f"an open window read as a wedge:\n{h.status()}"
        time.sleep(0.1)
    assert any(_stamp(ln) >= t0 + 3 for ln in _lines(h, "window=claimed")), h.status()
    assert h.proc.poll() is None and "WEDGED" not in h.status(), f"an open window read as a wedge:\n{h.status()}"


def test_a_window_hold_alerts_once_and_a_closed_window_is_a_wedge_again(harness):  # noqa: F811
    h = harness(extra=FAST)
    h.wait_for("ok step=1000")
    _window(h.run, P.HOLD, reason='the guest exited rc 3 ("x")')
    h.wait_for("ALERT: GPU WINDOW HOLD 20261020T120000_abc123")
    assert "the guest exited rc 3" in h.status()
    # three further ticks IN the hold: the watcher lives on, and the alert is said once (a revert repeats it per tick)
    n_alert = len(_lines(h, "ALERT: GPU WINDOW HOLD"))
    n_ok = len(_lines(h, "window=hold"))
    _wait_for_lines(h, "window=hold", n_ok + 3)
    assert h.proc.poll() is None, f"a HOLD ended the watcher:\n{h.status()}"
    assert len(_lines(h, "ALERT: GPU WINDOW HOLD")) == n_alert == 1, "the HOLD alert repeats every tick"

    # the window closes and the run still makes no progress: THAT is a wedge again
    _window(h.run, P.RESUMED)
    h.wait_for("FAILURE: WEDGED")
    assert json.loads((P.window_root(h.run) / P.CURRENT_FILE).read_text())["phase"] == P.RESUMED
