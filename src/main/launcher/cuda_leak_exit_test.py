"""K6's CUDA memory trend STOP is RESTARTED by the launcher — capped — never mislabelled a config error.

`agents.training.learner_lifecycle.CudaMemoryLeakError` (a SUSTAINED growth of live CUDA memory that
projects an OOM inside the horizon) is raised at an update's end; the trainer's exception handler
saves `final_model_exception.zip` and exits `exit_codes.exit_code_for(exc)`:

* the mapping is its OWN code, ``FATAL_CUDA_LEAK`` (6) — it used to be ``FATAL_CONFIG`` (3), which
  sent whoever triaged it to the argv and made the launcher stop for good;
* the launcher RESTARTS from that checkpoint up to ``CUDA_LEAK_RESTART_CAP`` (2) times per session,
  each with a loud event, and STOPS on the next one (a reproducible leak wants a human).

Every test here fails on revert of the part it names.
"""
import io
import queue
import time
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from main.exit_codes import CUDA_LEAK_RESTART_CAP, TrainExitCode, exit_code_for
from main.launcher.run import _fatal_config_reason, _SessionCtx, _supervise
from main.launcher.state import LauncherState


def test_the_leak_stop_maps_to_its_own_code_not_fatal_config():
    from agents.training.learner_lifecycle import CudaMemoryLeakError
    from main.exit_codes import FatalConfigError
    assert int(TrainExitCode.FATAL_CUDA_LEAK) == 6
    assert len({int(c) for c in TrainExitCode}) == len(TrainExitCode), "exit codes must be distinct"
    assert exit_code_for(CudaMemoryLeakError("[LearnerLifecycle] STOP — CUDA MEMORY LEAK: ...")) == 6
    assert not issubclass(CudaMemoryLeakError, FatalConfigError)
    assert _fatal_config_reason(6, ["[LearnerLifecycle] STOP — CUDA MEMORY LEAK: x"]) is None
    assert CUDA_LEAK_RESTART_CAP == 2


def _proc(rc: int) -> MagicMock:
    p = MagicMock()
    p.pid = 4343
    p.returncode = rc
    p.stdout = io.BytesIO(b"")
    p.wait.return_value = None
    return p


def _drive(tmp_path, rcs):
    run_dir = tmp_path / "models" / "leak_run"
    (run_dir / "checkpoints").mkdir(parents=True)
    ckpt = run_dir / "final_model_exception.zip"
    ckpt.write_text("x")
    ctx = _SessionCtx(child_args=["--model", str(ckpt), "--run-dir", str(run_dir)], child_env={},
                      train_script="train.py", src_dir="/src", run_dir=str(run_dir),
                      run_dir_box=[str(run_dir)], session_start=time.time(), interval_seconds=3 * 3600,
                      grace_seconds=1200.0, pin=False)
    with ExitStack() as st:
        popen = st.enter_context(patch("main.launcher.child.subprocess.Popen",
                                       side_effect=[_proc(rc) for rc in rcs]))
        st.enter_context(patch("main.launcher.child.threading.Thread"))
        st.enter_context(patch("main.launcher.child.os.pipe", return_value=(3, 4)))
        st.enter_context(patch("main.launcher.child.os.close"))
        st.enter_context(patch("main.launcher.run._git_hash", return_value="abc1234"))
        st.enter_context(patch("main.launcher.run.find_latest_checkpoint", return_value=str(ckpt)))
        st.enter_context(patch("main.launcher.run._save_crash_log", return_value=None))
        st.enter_context(patch("main.launcher.run.time.sleep"))
        st.enter_context(patch("main.launcher.run.atexit.register"))
        state = LauncherState(interval_hours=3.0)
        code = _supervise(state, queue.Queue(), ctx, interval_hours=3.0, grace_minutes=20.0,
                          max_crash_restarts=1)
    return code, popen.call_count, state.snapshot().events


def test_a_leak_stop_is_restarted_from_the_checkpoint_and_the_run_continues(tmp_path):
    leak = int(TrainExitCode.FATAL_CUDA_LEAK)
    code, spawned, events = _drive(tmp_path, [leak, int(TrainExitCode.COMPLETE)])
    assert code == 0 and spawned == 2, (code, spawned)
    assert any("CUDA memory leak STOP #1" in e and "RESTARTING" in e for e in events), events


def test_the_cap_holds_two_restarts_then_it_stops_for_good(tmp_path):
    """Two restarts (with --max-crash-restarts 1, so the fast-crash breaker is NOT what lets them
    through), then the third leak stop ends the session with code 6."""
    leak = int(TrainExitCode.FATAL_CUDA_LEAK)
    code, spawned, events = _drive(tmp_path, [leak] * (CUDA_LEAK_RESTART_CAP + 1))
    assert code == leak and spawned == CUDA_LEAK_RESTART_CAP + 1, (code, spawned)
    assert sum("RESTARTING" in e for e in events) == CUDA_LEAK_RESTART_CAP, events
    assert any(f"STOP #{CUDA_LEAK_RESTART_CAP + 1}" in e and "will NOT restart" in e for e in events)
