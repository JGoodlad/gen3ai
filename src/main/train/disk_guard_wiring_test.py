"""THE DISK GUARD's wiring: the checkpoint callback's per-save check, the clean stop's exit code, the
trainer's preflight and the run's wiring (`utils.disk_guard` owns the arithmetic and its own tests).

Every test fails on revert of the part it names.
"""
from __future__ import annotations

import errno
import os
import threading
from types import SimpleNamespace

import pytest

from main.exit_codes import TrainExitCode
from main.train import lifecycle
from main.train.deferred_abort import DeferredAbort
from main.train.run_io import _TrackingCheckpointCallback
from utils import disk_guard as dg
from utils.disk_guard import GiB, MiB
from utils.paths import src_path

SAVE = 60 * MiB


class _Model:
    """What the checkpointer reads: a step counter and a `save(path)` that writes `nbytes`."""

    def __init__(self, nbytes=1000, fail=None):
        self.num_timesteps = 0
        self.nbytes = nbytes
        self.fail = fail
        self.saves = []
        self.dumps = 0

    def save(self, path):
        with open(path, "wb") as f:
            f.write(b"\0" * (self.nbytes // 2))        # (a torn file if we then fail)
            if self.fail is not None:
                raise self.fail
            f.write(b"\0" * (self.nbytes - self.nbytes // 2))
        self.saves.append(path)

    def dump_logs(self):
        self.dumps += 1


def _cb(tmp_path, model, *, free, stop_enabled=True):
    cb = _TrackingCheckpointCallback(interval_env_steps=100, save_path=str(tmp_path / "run" / "checkpoints"))
    cb.model = model
    cb._init_callback()
    cb._on_training_start()
    cb.stops = []
    cb.disk_guard = dg.InRunGuard(str(tmp_path), stop_enabled=stop_enabled, free_fn=lambda p: free[0])
    cb.disk_stop_fn = cb.stops.append
    return cb


def _step_past_a_boundary(cb, model, to=150):
    model.num_timesteps = to
    return cb._on_step()


def test_a_roomy_disk_is_silent_and_costs_one_disk_read_per_save(tmp_path, capsys):
    reads = []
    model = _Model(nbytes=SAVE)
    cb = _cb(tmp_path, model, free=[1000 * GiB])
    cb.disk_guard = dg.InRunGuard(str(tmp_path), free_fn=lambda p: reads.append(p) or 1000 * GiB)
    model.num_timesteps = 50
    cb._on_step()                                       # no boundary: no save, no disk read
    assert reads == []
    _step_past_a_boundary(cb, model)
    assert len(reads) == 1 and capsys.readouterr().out == "" and cb.stops == []


def test_below_two_saves_warns_loudly_and_keeps_running(tmp_path, capsys):
    model = _Model(nbytes=SAVE)
    cb = _cb(tmp_path, model, free=[int(1.5 * SAVE)])
    assert _step_past_a_boundary(cb, model) is True
    out = capsys.readouterr().out
    assert "[DiskGuard]" in out and "LOW DISK" in out
    assert cb.stops == [], "a warning must not stop the run"


def test_below_one_save_takes_the_graceful_stop_after_the_save_is_on_disk(tmp_path, capsys):
    model = _Model(nbytes=SAVE)
    cb = _cb(tmp_path, model, free=[SAVE - 1])
    _step_past_a_boundary(cb, model)
    assert len(cb.stops) == 1 and "FATAL_DISK" in cb.stops[0]
    ckpt = tmp_path / "run" / "checkpoints" / "checkpoint_150_steps.zip"
    assert ckpt.stat().st_size == SAVE, "the checkpoint just written must stand"
    assert (tmp_path / "run" / "latest.txt").read_text().strip().endswith("checkpoint_150_steps.zip")


def test_the_opt_out_warns_but_never_stops(tmp_path, capsys):
    model = _Model(nbytes=SAVE)
    cb = _cb(tmp_path, model, free=[10], stop_enabled=False)
    _step_past_a_boundary(cb, model)
    assert cb.stops == [] and "LOW DISK" in capsys.readouterr().out


def test_a_save_that_dies_on_a_full_disk_removes_the_torn_file_and_stops(tmp_path):
    model = _Model(nbytes=SAVE, fail=OSError(errno.ENOSPC, "No space left on device"))
    cb = _cb(tmp_path, model, free=[1000 * GiB])
    with pytest.raises(OSError):                        # (the recorded stop returns; the error is re-raised)
        _step_past_a_boundary(cb, model)
    assert len(cb.stops) == 1 and "FAILED with a full disk" in cb.stops[0]
    assert not (tmp_path / "run" / "checkpoints" / "checkpoint_150_steps.zip").exists()
    assert not (tmp_path / "run" / "latest.txt").exists(), "latest.txt must keep naming the previous checkpoint"


def test_any_other_write_error_is_not_swallowed(tmp_path):
    model = _Model(nbytes=SAVE, fail=OSError(errno.EACCES, "denied"))
    cb = _cb(tmp_path, model, free=[1000 * GiB])
    with pytest.raises(OSError):
        _step_past_a_boundary(cb, model)
    assert cb.stops == []


def test_build_callbacks_installs_the_guard_and_honours_the_opt_out(tmp_path):
    from main.train.callbacks import build_callbacks
    from main.train.config import resolve_config
    from main.train_rl_agent import build_parser

    def build(*flags):
        p = build_parser()
        args = p.parse_args(["--steps", "1", "--debug-eval", *flags])
        resolve_config(args, p)
        args.debug_eval = False          # (checkpoint_cadence_test's recipe: skip the eval callback)
        args.debug = True
        b = build_callbacks(args=args, model_dir=str(tmp_path), annealing_mode=False, _pool=None,
                            _fixed_opponents=None, _bot_weight_vec=None, OPPONENT_NAMES=(),
                            _specialist_team_str=None, _promote_threshold=0.6, _heuristic_floor=0.0,
                            _sp_start_wr=0.5, _sp_full_wr=0.9)
        return b, b.callbacks[0]

    b, cb = build()
    assert isinstance(cb, _TrackingCheckpointCallback)
    assert cb.disk_guard is not None and cb.disk_guard.stop_enabled
    _b, cb = build("--allow-low-disk")
    assert cb.disk_guard is not None and not cb.disk_guard.stop_enabled
    # the stop reaches the run's DeferredAbort through the graceful-restart callback it is wired onto
    called = []
    b.graceful_restart_callback.disk_stop_fn = called.append
    b.callbacks[0].disk_stop_fn("why")
    assert called == ["why"]


# --- the stop itself ----------------------------------------------------------------------------

def test_disk_stop_exits_fatal_disk_without_a_save_and_claims_the_exit_once():
    exits, commits = [], []
    ab = DeferredAbort(lambda r: pytest.fail("a disk stop must not run the save-and-exit-15 body"),
                       disk_commit=commits.append, exit_fn=exits.append)
    ab.disk_stop("disk is full")
    assert exits == [int(TrainExitCode.FATAL_DISK)] == [8] and commits == ["disk is full"]
    assert ab.claimed_by == "disk guard"
    ab.close()


def test_the_runs_signal_wiring_gives_the_stop_a_body_that_never_saves(tmp_path, monkeypatch):
    saves = []
    monkeypatch.setattr(lifecycle, "save_model_snapshot", lambda *a, **k: saves.append("snap"))
    monkeypatch.setattr(lifecycle, "record_checkpoint", lambda *a, **k: saves.append("rec"))
    model = _Model()
    exits = []
    ev = threading.Event()
    ab = lifecycle._setup_signal_handlers(model, str(tmp_path), ev, None, lambda: 1e-3, lambda: 1,
                                          exit_fn=exits.append, start_watchdog=False)
    try:
        ab.disk_stop("no room")
        assert exits == [8] and model.saves == [] and saves == [] and model.dumps == 1 and ev.is_set()
    finally:
        ab.close()


def test_the_exit_code_is_distinct_and_documented():
    assert int(TrainExitCode.FATAL_DISK) == 8
    assert len({int(c) for c in TrainExitCode}) == len(TrainExitCode)
    doc = src_path("main", "launcher", "CLAUDE.md").read_text()
    assert "FATAL_DISK" in doc and "--allow-low-disk" in doc


def test_model_build_wires_the_stop_at_both_construction_sites():
    text = src_path("main", "train", "model_build.py").read_text()
    assert text.count("graceful_restart_callback.disk_stop_fn = _abort_fn.disk_stop") == 2


# --- the trainer's preflight --------------------------------------------------------------------

def test_the_trainer_refuses_fatal_config_before_it_creates_the_run_dir():
    """Static, on the entry point: the verdict is read and a refusal exits FATAL_CONFIG BEFORE
    `os.makedirs(model_dir` (a refusal leaves nothing behind), and the verdict is recorded."""
    text = src_path("main", "train_rl_agent.py").read_text()
    at_verdict = text.index("_disk_verdict(args, model_dir)")
    at_refusal = text.index("if _disk.refused:\n        sys.exit(int(TrainExitCode.FATAL_CONFIG))")
    at_makedirs = text.index("os.makedirs(model_dir, exist_ok=True)")
    assert at_verdict < at_refusal < at_makedirs
    assert 'cli_args["_disk_guard"] = _disk.to_record()' in text


def test_the_namespace_verdict_refuses_tolerates_and_exempts(tmp_path):
    from main.train.disk_preflight import verdict_for_namespace
    ns = SimpleNamespace(steps=15_000_000, checkpoint_every_steps=1_000_000, eval_freq=None, model=None,
                         self_play=True, device="cuda", debug=False, allow_low_disk=False)
    run = str(tmp_path / "rb_fresh")

    def tiny(_p):
        return 1 * GiB

    assert verdict_for_namespace(ns, run, free_fn=tiny).refused
    ns.allow_low_disk = True
    assert verdict_for_namespace(ns, run, free_fn=tiny).status == dg.OPTED_OUT
    ns.allow_low_disk, ns.debug = False, True
    assert verdict_for_namespace(ns, run, free_fn=tiny).status == dg.EXEMPT_DEBUG


def test_a_resumed_run_is_sized_from_its_own_checkpoint_and_remaining_steps(tmp_path):
    from main.train.disk_preflight import plan_for_namespace
    run = tmp_path / "rb_resume"
    ck = run / "checkpoints" / "checkpoint_9000100_steps.zip"
    os.makedirs(ck.parent)
    ck.write_bytes(b"\0" * (50 * MiB))
    ns = SimpleNamespace(steps=15_000_000, checkpoint_every_steps=1_000_000, eval_freq=2_000_000,
                         model=str(ck), self_play=True, device="cuda")
    plan = plan_for_namespace(ns, str(run))
    assert plan.resume_steps == 9_000_100 and plan.ckpt_bytes == 50 * MiB and plan.fork is False
    t = {label: b for label, b, _how in dg.requirement(plan).terms}
    assert t["periodic checkpoints"] == 6 * 50 * MiB    # 10M..15M
