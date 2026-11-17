"""The LAUNCHER and the DETACHED ladder updater never re-create a gone run dir; each has its own exit semantics.

They are other processes than the trainer, so a child's `RunDirGoneError` does not cover them. Before: the
launcher's crash log ``makedirs``'d ``<run>/crashes`` (it runs right after a child that died FATAL on the gone dir),
its exit footer ``makedirs``'d the run dir, each child start's log ``makedirs``'d it again, and a restart decision
looked for a checkpoint in it; the updater ``makedirs``'d ``snapshot_ladder/`` and wrote ``ladder.json`` through
``atomic_write``'s ``makedirs``. Now (designs/launcher/crashes_and_exit_codes.md):

* the launcher never restarts into a gone run dir: it stops with FATAL_CONFIG (3), the code a child's own
  `RunDirGoneError` maps to, and its crash log, child log and exit footer are skipped, never re-creating it;
* the updater exits FATAL_CONFIG (3) with ``[RunDir] FATAL``; nobody waits on it.

Every test deletes the run dir (in a `run_archive`) and fails on a revert of its site.
"""
from __future__ import annotations

import io
import queue
import shutil
import sys
import time
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from agents.training import snapshot_ladder as SL
from main.exit_codes import TrainExitCode
from main.launcher.child import _open_child_log
from main.launcher.run import _dump_logs_on_exit, _save_crash_log, _SessionCtx, _supervise
from main.launcher.state import LauncherState


def _run(run_archive):
    run = run_archive / "rb_launch_s1"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "checkpoint_1000_steps.zip").write_text("x")
    return run


def _proc(rc: int) -> MagicMock:
    p = MagicMock()
    p.pid = 4242
    p.returncode = rc
    p.stdout = io.BytesIO(b"")
    p.wait.return_value = None
    return p


def test_the_launcher_never_restarts_into_a_gone_run_dir(run_archive):
    """The child crashes (1) after its run dir was deleted: the launcher stops FATAL_CONFIG, re-creating nothing."""
    run = _run(run_archive)
    ckpt = run / "checkpoints" / "checkpoint_1000_steps.zip"
    ctx = _SessionCtx(child_args=["--model", str(ckpt), "--run-dir", str(run)], child_env={},
                      train_script="train.py", src_dir="/src", run_dir=str(run), run_dir_box=[str(run)],
                      session_start=time.time(), interval_seconds=3 * 3600, grace_seconds=1200.0, pin=False)

    def _spawn(*_a, **_k):
        shutil.rmtree(run)                        # deleted under the live child
        return _proc(int(TrainExitCode.CRASH))

    with ExitStack() as st:
        popen = st.enter_context(patch("main.launcher.child.subprocess.Popen", side_effect=[_spawn(),
                                                                                           _proc(0)]))
        st.enter_context(patch("main.launcher.child.threading.Thread"))
        st.enter_context(patch("main.launcher.child.os.pipe", return_value=(3, 4)))
        st.enter_context(patch("main.launcher.child.os.close"))
        st.enter_context(patch("main.launcher.run._git_hash", return_value="abc1234"))
        st.enter_context(patch("main.launcher.run.time.sleep"))
        st.enter_context(patch("main.launcher.run.atexit.register"))
        state = LauncherState(interval_hours=3.0)
        state.run_dir = str(run)
        code = _supervise(state, queue.Queue(), ctx, interval_hours=3.0, grace_minutes=20.0, max_crash_restarts=3)
    events = state.snapshot().events
    assert code == int(TrainExitCode.FATAL_CONFIG) and popen.call_count == 1, (code, popen.call_count, events)
    assert any("[RunDir] FATAL" in e and "will NOT restart" in e for e in events), events
    assert not run.exists(), "the launcher re-created the deleted run dir"


def test_the_crash_log_child_log_and_exit_footer_skip_a_gone_run_dir(run_archive, capsys):
    run = _run(run_archive)
    shutil.rmtree(run)
    state = LauncherState(interval_hours=3.0)
    state.run_dir = str(run)
    assert _save_crash_log(str(run), state, 3) is None
    assert any("[RunDir]" in e and "NOT saved" in e for e in state.snapshot().events)
    assert _open_child_log(state) is None
    _dump_logs_on_exit(str(run), state)
    assert not run.exists(), "a launcher log write re-created the deleted run dir"
    assert "[RunDir]" in capsys.readouterr().err


def test_control_a_live_run_still_gets_its_crash_log(run_archive):
    run = _run(run_archive)
    path = _save_crash_log(str(run), LauncherState(interval_hours=3.0), 1)
    assert path is not None and (run / "crashes").is_dir()


def test_the_detached_ladder_updater_exits_FATAL_CONFIG_on_a_gone_run_dir(run_archive, monkeypatch, capsys):
    run = _run(run_archive)
    shutil.rmtree(run)
    try:
        SL._append_game(str(run), 1000, 2000, 1, 2)
        raise AssertionError("_append_game wrote into a gone run dir")
    except SL.RunDirGoneError:
        pass
    monkeypatch.setattr(sys, "argv", ["snapshot_ladder", str(run), "--fit-only"])
    assert SL.main() == int(TrainExitCode.FATAL_CONFIG)
    assert "[RunDir] FATAL (snapshot ladder)" in capsys.readouterr().out
    assert not run.exists(), "the ladder updater re-created the deleted run dir"
