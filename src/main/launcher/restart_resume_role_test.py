"""A same-run RESTART builds the child argv from the RESUME role; a FRESH launch never lands on a run.

THE INCIDENT (2026-09-26 12:23 PT). The fresh run ``ai_v14_01_base`` was launched with
``--arch production`` — a FRESH-run-only flag. At its first ``--restart-interval-hours`` restart the
launcher re-passed the ORIGINAL argv plus ``--model <latest>``; the trainer refused ``--arch`` beside
``--model`` (exit 2) three times and the circuit-breaker gave up — ~40 GPU-min lost. The same argv
with ``--arch`` merely dropped and no ``--model`` resolved as a FRESH run from step 0 INTO the
existing run dir — a silent clobber.

Both halves are pinned here against temporary run dirs only: (1) every restart kind (interval,
crash) strips the trainer's own ``fresh_only_flags()`` and points ``--model`` at the checkpoint;
(2) ``resolve_launch_run_dir`` REFUSES a fresh launch into a dir holding a checkpoint or a
``model_config.json``, with ``FATAL_CONFIG`` and a message naming both remedies. The ``--dry-run``
half is in ``dry_run_test.py`` (it needs that file's git-repo fixture).
"""

import io
import queue
import time
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from main.exit_codes import TrainExitCode
from main.launcher.checkpoint import (
    FreshRunDirHasProgress,
    resolve_launch_run_dir,
    resume_child_args,
    run_dir_progress,
)
from main.launcher.run import _SessionCtx, _supervise
from main.launcher.state import LauncherState

TS = "20260926_122300"

#: The shape of the incident's argv (abridged): a FRESH named run under the production umbrella.
FRESH_ARCH_ARGV = ["--arch", "production", "--run-name", "ai_v14_01_base",
                   "--steps", "50000000", "--device", "cuda"]


# ── resume_child_args: the pure builder ──────────────────────────────────────────

@pytest.mark.parametrize("spelling", [["--arch", "production"], ["--arch=production"]])
def test_resume_argv_strips_arch_in_both_spellings(spelling):
    args = [*spelling, "--steps", "100", "--run-dir", "models/x"]
    out, stripped = resume_child_args(args, "models/x/checkpoints/checkpoint_9_steps.zip", "models/x")
    assert "--arch" not in " ".join(out) and "production" not in out
    assert stripped == ["--arch"]
    assert out[out.index("--model") + 1] == "models/x/checkpoints/checkpoint_9_steps.zip"
    assert out[out.index("--run-dir") + 1] == "models/x"
    assert out[out.index("--steps") + 1] == "100", "neighbouring flags must survive the strip"


def test_resume_argv_is_a_no_op_strip_for_a_fork_argv():
    args = ["--model", "models/p/c.zip", "--run-name", "f", "--fork-lr", "2.8e-5"]
    out, stripped = resume_child_args(args, "models/f/checkpoints/c2.zip", "models/f")
    assert stripped == []
    assert out[out.index("--model") + 1] == "models/f/checkpoints/c2.zip"
    assert out.count("--model") == 1 and "--fork-lr" in out


# ── _supervise: the restart loop really re-launches with the RESUME argv ────────

def _proc(rc: int) -> MagicMock:
    p = MagicMock()
    p.pid = 4242
    p.returncode = rc
    p.stdout = io.BytesIO(b"")
    p.wait.return_value = None
    return p


def _run_loop(tmp_path, first_rc: int, *, interval_hours: float):
    """Drive `_supervise` through ONE restart (first child exits `first_rc`, second COMPLETEs)
    with the incident argv, and return the argv each child was spawned with."""
    run_dir = tmp_path / "models" / "ai_v14_01_base"
    (run_dir / "checkpoints").mkdir(parents=True)
    ckpt = run_dir / "checkpoints" / "checkpoint_1000_steps.zip"
    ckpt.write_text("x")
    ctx = _SessionCtx(
        child_args=[*FRESH_ARCH_ARGV, "--run-dir", str(run_dir)],
        child_env={}, train_script="train.py", src_dir="/src",
        run_dir=str(run_dir), run_dir_box=[str(run_dir)], session_start=time.time(),
        interval_seconds=interval_hours * 3600, grace_seconds=1200.0, pin=False,
    )
    with ExitStack() as st:
        popen = st.enter_context(patch("main.launcher.child.subprocess.Popen",
                                       side_effect=[_proc(first_rc),
                                                    _proc(TrainExitCode.COMPLETE)]))
        st.enter_context(patch("main.launcher.child.threading.Thread"))
        st.enter_context(patch("main.launcher.child.os.pipe", return_value=(3, 4)))
        st.enter_context(patch("main.launcher.child.os.close"))
        st.enter_context(patch("main.launcher.run._git_hash", return_value="abc1234"))
        st.enter_context(patch("main.launcher.run.find_latest_checkpoint",
                               return_value=str(ckpt)))
        st.enter_context(patch("main.launcher.run._save_crash_log", return_value=None))
        st.enter_context(patch("main.launcher.run.time.sleep"))
        st.enter_context(patch("main.launcher.run.atexit.register"))
        state = LauncherState(interval_hours=interval_hours)
        code = _supervise(state, queue.Queue(), ctx, interval_hours=interval_hours,
                          grace_minutes=20.0, max_crash_restarts=3)
    assert code == 0
    argvs = [c.args[0][2:] for c in popen.call_args_list]   # drop [python, train_script]
    return argvs, str(ckpt), str(run_dir), state


@pytest.mark.parametrize("first_rc,interval_hours,kind", [
    (TrainExitCode.INTERRUPTED, 3.0, "interval"),
    (TrainExitCode.CRASH, 3.0, "crash"),
])
def test_restart_of_an_arch_production_run_resumes_without_arch(tmp_path, first_rc,
                                                                interval_hours, kind):
    argvs, ckpt, run_dir, state = _run_loop(tmp_path, first_rc, interval_hours=interval_hours)
    assert len(argvs) == 2, f"{kind}: expected exactly one restart"
    first, restart = argvs
    assert "--arch" in first and "--model" not in first, "the FIRST launch is the fresh one"
    assert "--arch" not in restart and "production" not in restart, (
        f"{kind} restart re-passed --arch beside --model — the trainer refuses that "
        f"(arch_umbrella_is_fresh_only) and the run crash-loops out, 2026-09-26")
    assert restart[restart.index("--model") + 1] == ckpt
    assert restart[restart.index("--run-dir") + 1] == run_dir
    assert any("dropped FRESH-only --arch" in e for e in state.snapshot().events)


# ── the FRESH-into-an-existing-run refusal ──────────────────────────────────────

def _existing_run(tmp_path, *, checkpoint=True, model_config=True):
    run_dir = tmp_path / "models" / "ai_v14_01_base"
    (run_dir / "checkpoints").mkdir(parents=True)
    if checkpoint:
        (run_dir / "checkpoints" / "checkpoint_5000_steps.zip").write_text("x")
        (run_dir / "latest.txt").write_text("checkpoints/checkpoint_5000_steps.zip")
    if model_config:
        (run_dir / "model_config.json").write_text("{}")
    return run_dir


@pytest.mark.parametrize("checkpoint,model_config", [(True, True), (True, False), (False, True)])
def test_fresh_launch_into_a_run_with_progress_is_refused(tmp_path, monkeypatch,
                                                         checkpoint, model_config):
    monkeypatch.chdir(tmp_path)
    _existing_run(tmp_path, checkpoint=checkpoint, model_config=model_config)
    argv = [a for a in FRESH_ARCH_ARGV if a not in ("--arch", "production")]   # the 2nd hazard
    with pytest.raises(FreshRunDirHasProgress) as e:
        resolve_launch_run_dir(argv, TS)
    assert e.value.exit_code == int(TrainExitCode.FATAL_CONFIG)
    msg = str(e.value)
    assert "--model" in msg and "--run-name" in msg, "the refusal must name both remedies"
    if checkpoint:
        assert "checkpoint_5000_steps.zip" in msg, "name the checkpoint to resume from"


def test_fresh_launch_via_run_dir_into_a_run_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = _existing_run(tmp_path)
    with pytest.raises(FreshRunDirHasProgress):
        resolve_launch_run_dir(["--run-dir", str(run_dir), "--steps", "1"], TS)


def test_fresh_launch_into_an_empty_or_seed_only_dir_is_allowed(tmp_path, monkeypatch):
    """A dir with only the step-0 self-play seed (written before any checkpoint) is no run yet."""
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "models" / "new_run"
    (run_dir / "snapshots").mkdir(parents=True)
    (run_dir / "snapshots" / "snapshot_0.zip").write_text("x")
    assert run_dir_progress(str(run_dir)) == []
    assert resolve_launch_run_dir(["--run-name", "new_run"], TS) == "models/new_run"


def test_restart_resume_of_the_same_run_is_not_refused(tmp_path, monkeypatch):
    """The guard is FRESH-only: the RESUME argv the restart loop builds must still resolve."""
    monkeypatch.chdir(tmp_path)
    run_dir = _existing_run(tmp_path)
    ckpt = str(run_dir / "checkpoints" / "checkpoint_5000_steps.zip")
    argv, _ = resume_child_args(FRESH_ARCH_ARGV, ckpt, str(run_dir))
    # --run-name + --model is the FORK signal; its idempotent-resume path accepts a dir with
    # progress, and the injected --run-dir wins over --run-name.
    assert resolve_launch_run_dir(argv, TS) == str(run_dir)
