"""`scripts/ops/watch_run.sh` knows a GPU WINDOW (T46) pauses the run on purpose (review 2026-10-20).

A window holds the run with no step progress for up to its `--max-window-min` (the doc's example: 130 min),
far past the watcher's 35-min wedge limit, so the unaware watcher declared `FAILURE: WEDGED`, exited, and the
run went unwatched from its resume on; and the window's HOLD (the one state that needs a human) said nothing.
Each assertion fails on a revert of the window block in the script.
"""
import json

from main.launcher.gpu_window import protocol as P
from main.ops.watch_run_resume_test import Harness, harness  # noqa: F401 — the fixture


def _window(run_dir, phase, reason="r"):
    P.write_json_atomic(P.window_root(run_dir) / P.CURRENT_FILE,
                        {"window_id": "20261020T120000_abc123", "phase": phase, "reason": reason})


def test_an_open_window_is_not_a_wedge_and_its_hold_alerts_once(harness):  # noqa: F811
    h = harness(extra=("--wedge-seconds", "2"))
    h.wait_for("ok step=1000")
    _window(h.run, P.CLAIMED)
    h.wait_for("window=claimed")
    assert h.alive_for(4), f"an open window read as a wedge:\n{h.status()}"
    assert "WEDGED" not in h.status()

    _window(h.run, P.HOLD, reason='the guest exited rc 3 ("x")')
    h.wait_for("ALERT: GPU WINDOW HOLD 20261020T120000_abc123")
    assert "the guest exited rc 3" in h.status()
    assert h.alive_for(3), f"a HOLD ended the watcher:\n{h.status()}"
    assert h.status().count("ALERT: GPU WINDOW HOLD") == 1, "the HOLD alert repeats every tick"

    # the window closes and the run still makes no progress: THAT is a wedge again
    _window(h.run, P.RESUMED)
    h.wait_for("FAILURE: WEDGED")
    assert json.loads((P.window_root(h.run) / P.CURRENT_FILE).read_text())["phase"] == P.RESUMED
