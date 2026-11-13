"""A quit whose child did not stop cleanly is an UNSAVED STOP with its own exit code, never "Quit complete" / 0.

rb_b1c_s5101 (2026-11-12): the launcher's own SIGTERM killed the child FATAL (exit 3, no save); the launcher,
with a quit pending, logged "Quit complete" and exited 0 before it ever classified the child's exit. Now:

* `_supervise`'s two quit ends (the quit itself, and a quit during a GPU window) return
  `exit_codes.LAUNCHER_UNSAVED_STOP` and say which checkpoint `latest.txt` names, unless the child ended
  INTERRUPTED (15), the one saved stop;
* `run()` reads the supervisor's verdict (`_session_exit_code`), never the app's initial 0, after an abnormal
  app exit.

Every test here fails on revert of the part it names. No marker: mocked children, no process, milliseconds.
"""
import inspect
import io
import os
import queue
import signal
import subprocess
import threading
import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from main.exit_codes import LAUNCHER_UNSAVED_STOP, TrainExitCode
from main.launcher import run as run_module_fn        # the package re-exports the FUNCTION `run`
from main.launcher.run import (_resume_point, _session_exit_code, _SessionCtx, _stop_was_unsaved,
                               _supervise)
from main.launcher.state import LauncherState
from main.ops.arm_chain.armend import LauncherSummary, completeness
from main.ops.arm_chain.launch import Hold
from utils.paths import repo_path

_SENTINEL_FATAL = ("[CompileSentinel] FATAL: the 'update end' update raised a recompile rejection after the "
                   "compile lock (fail_on_recompile)")


# ── the code ───────────────────────────────────────────────────────────────────────────────────────

def test_the_code_is_the_launchers_own_distinct_and_documented():
    assert LAUNCHER_UNSAVED_STOP == 9
    assert LAUNCHER_UNSAVED_STOP not in {int(c) for c in TrainExitCode}, "a child code would be ambiguous"
    doc = repo_path("designs", "launcher", "crashes_and_exit_codes.md").read_text(encoding="utf-8")
    row = next(ln for ln in doc.splitlines() if ln.startswith("| 9 |"))
    assert "LAUNCHER_UNSAVED_STOP" in row and "UNSAVED stop" in row and "latest.txt" in row, row


def test_only_INTERRUPTED_without_a_FATAL_line_is_a_saved_stop():
    assert _stop_was_unsaved(int(TrainExitCode.INTERRUPTED), ["[ABORT] Checkpoint saved → x.zip"]) is False
    assert _stop_was_unsaved(int(TrainExitCode.INTERRUPTED), None) is False
    for rc in (3, 1, 4, 8, -15, -9):
        assert _stop_was_unsaved(rc, []) is True, rc
    assert _stop_was_unsaved(int(TrainExitCode.INTERRUPTED), ["chatter", _SENTINEL_FATAL]) is True


# ── the quit ───────────────────────────────────────────────────────────────────────────────────────

def _proc(rc: int) -> MagicMock:
    p = MagicMock()
    p.pid = 4343
    p.returncode = rc
    p.stdout = io.BytesIO(b"")
    # alive for one poll (so the quit is read and the SIGTERM sent), then exited
    p.wait.side_effect = [subprocess.TimeoutExpired("c", 0.5), None, None, None]
    return p


def _run_dir(tmp_path):
    run_dir = tmp_path / "models" / "rb_unsaved_toy"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "checkpoint_1000_steps.zip").write_text("x")
    (run_dir / "latest.txt").write_text("checkpoints/checkpoint_1000_steps.zip")
    return run_dir


def _drive(tmp_path, rc: int, *, log=(), via: str = "key", window=None):
    """`_supervise` over ONE mocked child that exits `rc` after the launcher's SIGTERM, in a real run dir
    (the real `latest.txt` read, the real crash log). Returns (code, events, signals sent, run_dir)."""
    run_dir = _run_dir(tmp_path)
    ctx = _SessionCtx(child_args=["--run-dir", str(run_dir)], child_env={}, train_script="train.py",
                      src_dir="/src", run_dir=str(run_dir), run_dir_box=[str(run_dir)],
                      session_start=time.time(), interval_seconds=0, grace_seconds=1200.0, pin=False)
    q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    if via == "key":
        q.put("__quit__")                       # `q` → `y`, or a SIGTERM / SIGHUP to the launcher (app.py)
    state = LauncherState(interval_hours=0)
    for line in log:
        state.add_log(line)
    if window is not None:
        state.child_has_reported = lambda: True     # the window pauses only a child that has reported
    with ExitStack() as st:
        popen = st.enter_context(patch("main.launcher.child.subprocess.Popen", side_effect=[_proc(rc)]))
        st.enter_context(patch("main.launcher.child.threading.Thread"))
        st.enter_context(patch("main.launcher.child.os.pipe", return_value=(3, 4)))
        st.enter_context(patch("main.launcher.child.os.close"))
        st.enter_context(patch("main.launcher.run._git_hash", return_value="abc1234"))
        st.enter_context(patch("main.launcher.run.time.sleep"))
        st.enter_context(patch("main.launcher.run.atexit.register"))
        kill = st.enter_context(patch("main.launcher.run.os.kill"))
        if via == "teardown":
            # the app goes down (`run()`'s finally sets `shutdown`) while the child is running: set at the
            # launch, since the loop's top check returns before launching once it is set
            popen.side_effect = lambda *a, **k: (shutdown.set(), _proc(rc))[1]
        code = _supervise(state, q, ctx, interval_hours=0, grace_minutes=20.0, max_crash_restarts=3,
                          shutdown=shutdown, window=window)
    return code, state.snapshot().events, [c.args[1] for c in kill.call_args_list], run_dir


@pytest.mark.parametrize("rc", [3, 1, 4, -15, -9])
def test_a_quit_whose_child_did_not_end_INTERRUPTED_is_an_unsaved_stop_never_quit_complete(tmp_path, rc):
    """The incident's shape at rc 3 (the sentinel FATAL); the others are the same lie: a crash (1), a
    non-finite stop (4), a child killed before its handlers were up (-15) or after the SIGKILL grace (-9)."""
    code, events, sent, run_dir = _drive(tmp_path, rc, log=[_SENTINEL_FATAL] if rc == 3 else [])
    assert sent == [signal.SIGTERM]
    assert code == LAUNCHER_UNSAVED_STOP, (code, events)
    assert not any("Quit complete" in e for e in events), events
    line = next(e for e in events if "UNSAVED stop (quit)" in e)
    assert f"the child exited {rc}, not INTERRUPTED (15)" in line, line
    assert "latest.txt → checkpoints/checkpoint_1000_steps.zip" in line, line
    assert f"launcher exit {LAUNCHER_UNSAVED_STOP}" in line, line
    crash_logs = os.listdir(run_dir / "crashes")
    assert len(crash_logs) == 1 and f"crashes/{crash_logs[0]}" in line, (crash_logs, line)
    if rc == 3:
        assert any("[CompileSentinel] FATAL" in e for e in events), events     # the child's reason, on screen


def test_a_FATAL_line_makes_even_an_INTERRUPTED_exit_an_unsaved_stop(tmp_path):
    code, events, _sent, _ = _drive(tmp_path, int(TrainExitCode.INTERRUPTED), log=[_SENTINEL_FATAL])
    assert code == LAUNCHER_UNSAVED_STOP and any("UNSAVED stop (quit)" in e for e in events), events


def test_control_a_quit_whose_child_ends_INTERRUPTED_is_still_quit_complete_and_zero(tmp_path):
    code, events, sent, run_dir = _drive(tmp_path, int(TrainExitCode.INTERRUPTED),
                                         log=["[ABORT] Checkpoint saved → final_model_interrupted.zip"])
    assert code == 0 and sent == [signal.SIGTERM]
    assert any("Quit complete." in e for e in events) and not any("UNSAVED" in e for e in events), events
    assert not (run_dir / "crashes").exists()


def test_the_apps_teardown_quit_is_judged_the_same_way(tmp_path):
    code, events, sent, _ = _drive(tmp_path, 3, via="teardown")
    assert code == LAUNCHER_UNSAVED_STOP and sent == [signal.SIGTERM], (code, sent, events)
    assert any("UNSAVED stop (quit)" in e for e in events), events


# ── a quit during a GPU window ─────────────────────────────────────────────────────────────────────

class _HeldWindow:
    """A window that is due at once and whose HOLD the operator ends with a quit (what `run_window` returns
    for a paused child that did not exit INTERRUPTED, then a quit)."""

    def __init__(self) -> None:
        self.rcs: list = []

    def due(self):
        return SimpleNamespace(window_id="w1")

    def run_window(self, req, rc, sigterm_epoch, should_quit) -> str:
        self.rcs.append(rc)
        return "quit"

    def abandon(self, req, why) -> None:
        raise AssertionError(f"no quit was pending when the child exited: {why}")


def test_a_quit_during_a_window_held_by_an_unclean_pause_is_an_unsaved_stop(tmp_path):
    window = _HeldWindow()
    code, events, sent, _ = _drive(tmp_path, 3, via="window", window=window)
    assert window.rcs == [3] and sent == [signal.SIGTERM]
    assert code == LAUNCHER_UNSAVED_STOP, (code, events)
    assert any("UNSAVED stop (quit during a GPU window)" in e for e in events), events
    assert not any("Quit complete" in e for e in events), events


def test_control_a_quit_during_a_window_after_a_clean_pause_is_still_zero(tmp_path):
    window = _HeldWindow()
    code, events, _sent, _ = _drive(tmp_path, int(TrainExitCode.INTERRUPTED), via="window", window=window)
    assert window.rcs == [15] and code == 0
    assert any("Quit complete (during a GPU window)." in e for e in events), events


# ── where the run stands ───────────────────────────────────────────────────────────────────────────

def test_the_resume_point_is_what_latest_txt_names_or_what_is_on_disk(tmp_path):
    run_dir = _run_dir(tmp_path)
    assert _resume_point(str(run_dir)) == "latest.txt → checkpoints/checkpoint_1000_steps.zip"
    (run_dir / "latest.txt").write_text("checkpoints/checkpoint_9999_steps.zip")      # names no file
    assert _resume_point(str(run_dir)) == ("latest.txt names no file; the newest checkpoint on disk is "
                                           "checkpoints/checkpoint_1000_steps.zip")
    (run_dir / "latest.txt").unlink()
    (run_dir / "checkpoints" / "checkpoint_1000_steps.zip").unlink()
    assert _resume_point(str(run_dir)) == "NO checkpoint in the run dir"
    assert _resume_point(None) == "no run dir, so no checkpoint"


# ── run(): the supervisor's verdict, never the app's initial 0 ─────────────────────────────────────

class _VerdictLandsDuringTheWait(threading.Event):
    """The supervisor records its verdict while `run()` waits for it (the abnormal-app-exit order)."""

    def __init__(self, app) -> None:
        super().__init__()
        self._app = app

    def wait(self, timeout=None) -> bool:
        self._app.exit_code = LAUNCHER_UNSAVED_STOP
        self.set()
        return True


def test_the_session_exit_code_waits_for_the_supervisors_verdict():
    app = SimpleNamespace(exit_code=0)                     # LauncherApp's initial value
    assert _session_exit_code(app, [object()], _VerdictLandsDuringTheWait(app)) == LAUNCHER_UNSAVED_STOP

    done = threading.Event()
    done.set()                                             # the normal order: verdict first, then the app exits
    assert _session_exit_code(SimpleNamespace(exit_code=3), [object()], done) == 3
    # no verdict in time is never "clean"
    assert _session_exit_code(SimpleNamespace(exit_code=0), [object()], threading.Event(), wait_s=0.0) == 1
    # no child was ever launched: nothing to wait for
    assert _session_exit_code(SimpleNamespace(exit_code=0), [], threading.Event(), wait_s=0.0) == 0


def test_run_exits_through_the_verdict_and_the_supervisor_signals_it():
    """Static, on `run()`: the exit reads `_session_exit_code`, and the supervisor sets the event right after
    it records its code (the pre-fix shape was `sys.exit(getattr(app, "exit_code", 0))`)."""
    src = inspect.getsource(run_module_fn)
    assert "sys.exit(_session_exit_code(app, proc_box, supervisor_done))" in src
    assert 'getattr(app, "exit_code"' not in src
    recorded = src.index("app.exit_code = code")
    assert src.index("supervisor_done.set()") > recorded
    assert src.index("supervisor_done.set()") < src.index("app.call_from_thread(app.exit)")


# ── the callers that read the launcher's exit code ─────────────────────────────────────────────────

def test_the_arm_chain_reads_the_new_code_as_a_HOLD(tmp_path):
    """`arm_chain` enumerates no codes: any non-zero `launcher_rc` is "not a clean end". Before the fix an
    unsaved stop reached it as rc 0 and fell through to the completeness arithmetic."""
    with pytest.raises(Hold, match=rf"launcher_rc={LAUNCHER_UNSAVED_STOP} \(not a clean end\)"):
        completeness(str(tmp_path), 1000, LAUNCHER_UNSAVED_STOP,
                     LauncherSummary(last_step=None, fps=None, restarts=None, pins=[]))
