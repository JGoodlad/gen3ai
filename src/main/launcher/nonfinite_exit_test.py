"""A NON-FINITE learner stops the launcher; it is never restarted into the same update.

Lane K's K9 guard raises ``NonFiniteLearnerError`` on a NaN / Inf loss or gradient (fail CLOSED). As a
generic exception it exited ``CRASH`` (1), so the launcher resumed the checkpoint that produced it,
replayed the same update and crash-looped until ``--max-crash-restarts`` ran out. Now:

* the trainer's fail-fast handlers exit ``exit_codes.exit_code_for(exc)`` — ``FATAL_NONFINITE`` (4) for
  the error or anything whose cause chain holds it, ``CRASH`` for everything else;
* the launcher's ``_supervise`` gives up on 4 exactly as on ``FATAL_CONFIG`` — one child, no restart,
  the code propagated, a distinct event.

Every test here fails on revert of the part it names.
"""
import ast
import io
import queue
import time
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from main.exit_codes import NonFiniteLearnerError, TrainExitCode, exit_code_for, fatal_exit_code_for
from main.launcher.run import _fatal_config_reason, _SessionCtx, _supervise
from main.launcher.state import LauncherState
from utils.paths import src_path


# ── the mapping ────────────────────────────────────────────────────────────────

def test_a_non_finite_learner_maps_to_its_own_fatal_code_and_everything_else_crashes():
    assert int(TrainExitCode.FATAL_NONFINITE) == 4
    assert len({int(c) for c in TrainExitCode}) == len(TrainExitCode), "exit codes must be distinct"
    assert exit_code_for(NonFiniteLearnerError("loss is nan")) == 4
    assert exit_code_for(RuntimeError("boom")) == int(TrainExitCode.CRASH) == 1
    assert exit_code_for(FloatingPointError("a plain FP error is NOT the guard")) == 1
    assert exit_code_for(None) == 1


def test_the_mapping_follows_the_cause_chain_and_a_same_named_subclass():
    try:
        try:
            raise NonFiniteLearnerError("grad inf")
        except NonFiniteLearnerError as inner:
            raise RuntimeError("callback wrapper") from inner
    except RuntimeError as outer:
        assert exit_code_for(outer) == 4

    class NonFiniteLearnerError_(Exception):     # noqa: N801 — a foreign class, not the name
        pass

    ForeignSameName = type("NonFiniteLearnerError", (RuntimeError,), {})   # a guard's own class
    assert fatal_exit_code_for(ForeignSameName("nan")) == 4
    assert fatal_exit_code_for(NonFiniteLearnerError_("x")) is None


def test_both_trainer_fail_fast_handlers_exit_through_the_mapping():
    """Static: `global_exception_handler` and `asyncio_exception_handler` in the trainer entry point
    call `os._exit(exit_code_for(...))`, never a literal 1 (the pre-fix shape)."""
    tree = ast.parse(src_path("main", "train_rl_agent.py").read_text())
    handlers = {f.name: f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
                and f.name in ("global_exception_handler", "asyncio_exception_handler")}
    assert set(handlers) == {"global_exception_handler", "asyncio_exception_handler"}
    for name, fn in handlers.items():
        exits = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                 and c.func.attr == "_exit"]
        assert exits, name
        for c in exits:
            arg = c.args[0]
            assert isinstance(arg, ast.Call) and getattr(arg.func, "id", "") == "exit_code_for", (
                f"{name} exits {ast.unparse(arg)} — a non-finite learner would read as a restartable CRASH")


def test_the_launcher_classifies_code_4_as_fatal():
    lines = ["...", "main.exit_codes.NonFiniteLearnerError: loss is nan at update 312"]
    reason = _fatal_config_reason(int(TrainExitCode.FATAL_NONFINITE), lines)
    assert reason and "non-finite learner" in reason[0] and "update 312" in reason[-1]
    assert _fatal_config_reason(int(TrainExitCode.CRASH), ["plain traceback"]) is None


# ── the restart loop really stops ──────────────────────────────────────────────

def _proc(rc: int) -> MagicMock:
    p = MagicMock()
    p.pid = 4343
    p.returncode = rc
    p.stdout = io.BytesIO(b"")
    p.wait.return_value = None
    return p


def _drive(tmp_path, first_rc: int):
    run_dir = tmp_path / "models" / "nonfinite_run"
    (run_dir / "checkpoints").mkdir(parents=True)
    ckpt = run_dir / "checkpoints" / "checkpoint_1000_steps.zip"
    ckpt.write_text("x")
    ctx = _SessionCtx(child_args=["--model", str(ckpt), "--run-dir", str(run_dir)], child_env={},
                      train_script="train.py", src_dir="/src", run_dir=str(run_dir),
                      run_dir_box=[str(run_dir)], session_start=time.time(), interval_seconds=3 * 3600,
                      grace_seconds=1200.0, pin=False)
    with ExitStack() as st:
        popen = st.enter_context(patch("main.launcher.child.subprocess.Popen",
                                       side_effect=[_proc(first_rc), _proc(TrainExitCode.COMPLETE)]))
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
                          max_crash_restarts=3)
    return code, popen.call_count, state.snapshot().events


def test_a_non_finite_exit_is_never_restarted(tmp_path):
    code, spawned, events = _drive(tmp_path, int(TrainExitCode.FATAL_NONFINITE))
    assert code == 4 and spawned == 1, (code, spawned)
    assert any("Non-finite learner — will NOT restart" in e for e in events), events


@pytest.mark.parametrize("rc", [TrainExitCode.CRASH])
def test_control_an_ordinary_crash_still_restarts(tmp_path, rc):
    code, spawned, _events = _drive(tmp_path, int(rc))
    assert code == 0 and spawned == 2
