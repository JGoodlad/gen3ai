"""The deferred abort (`main.train.deferred_abort`, gen3_deferred_abort_v1; P10 review F1).

A SIGINT / SIGTERM / SIGHUP only RECORDS a stop request; the main thread runs the abort (dump + save +
exit 15) at the next SAFE POINT; a watchdog exits WITHOUT a save if none comes within the deadline.

Each signal-level test drives the REAL handlers `lifecycle._setup_signal_handlers` installs and FAILS on a
revert to the handler-does-everything form (the handler calling the abort directly):

* a signal raised INSIDE a TensorBoard flush (the writer's non-reentrant lock held) never re-enters that
  lock — the old handler's dump did, which is the F1 deadlock (detected here by an owner-checking lock
  instead of hanging);
* a signal raised INSIDE an update saves nothing until the update has returned;
* a signal with no safe point to follow ends in the fallback's exit, with no save.

Synchronised by events and by the order of calls, never by sleeps.
"""
from __future__ import annotations

import os
import signal
import threading
from typing import Any, List, Optional

import pytest

from main.train import lifecycle
from main.train.deferred_abort import SAFE_POINT_DEADLINE_SEC, SAVE_BUDGET_SEC, DeferredAbort
from utils.contention import scale_timeout

N_STEPS, N_ENVS = 8, 2
ROWS = N_STEPS * N_ENVS
_SIGNALS = [s for s in ("SIGINT", "SIGTERM", "SIGHUP", "SIGUSR1", "SIGUSR2") if hasattr(signal, s)]
# A hang guard only (the outcome is decided by call order, never by this bound).
_GUARD_S = scale_timeout(30.0)


class _Exited(BaseException):
    """What the injected exit raises on the MAIN thread (unwinds learn() the way os._exit would end it)."""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


class _Exit:
    """The injected `exit_fn`: records (code, thread); raises `_Exited` on the main thread only."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []
        self.called = threading.Event()

    def __call__(self, code: int) -> None:
        main = threading.current_thread() is threading.main_thread()
        self.calls.append((int(code), "main" if main else threading.current_thread().name))
        self.called.set()
        if main:
            raise _Exited(int(code))


@pytest.fixture
def restore_signals():
    saved = {name: signal.getsignal(getattr(signal, name)) for name in _SIGNALS}
    try:
        yield
    finally:
        for name, handler in saved.items():
            signal.signal(getattr(signal, name), handler)


# ---------------------------------------------------------------------------- the class, no signals
def test_the_deadline_fits_inside_the_launchers_sigkill_grace():
    from main.launcher.run import KILL_GRACE_SECONDS

    # a safe point reached just before the deadline still has SAVE_BUDGET_SEC to save before the SIGKILL
    assert SAFE_POINT_DEADLINE_SEC + SAVE_BUDGET_SEC <= KILL_GRACE_SECONDS
    assert SAVE_BUDGET_SEC > 0


def _controlled(commit: Optional[List[str]] = None):
    """A DeferredAbort whose watchdog waits on `release` (set by the test) instead of the deadline."""
    commits: List[str] = [] if commit is None else commit
    in_wait, release = threading.Event(), threading.Event()

    def wait_fn(_seconds: float) -> None:
        in_wait.set()
        assert release.wait(_GUARD_S), "hang guard: the test never released the watchdog"

    ex = _Exit()
    abort = DeferredAbort(commits.append, exit_fn=ex, wait_fn=wait_fn)
    return abort, commits, ex, in_wait, release


def test_no_safe_point_means_the_fallback_exits_without_a_commit():
    abort, commits, ex, in_wait, release = _controlled()
    abort.start()
    try:
        abort.request("SIGTERM received")
        assert in_wait.wait(_GUARD_S)
        release.set()
        assert ex.called.wait(_GUARD_S)
        assert ex.calls == [(15, "deferred-abort-watchdog")]
        assert commits == [] and abort.claimed_by == "fallback"
    finally:
        release.set()
        abort.close()


def test_a_safe_point_inside_the_deadline_wins_and_the_fallback_stands_down():
    abort, commits, ex, in_wait, release = _controlled()
    abort.start()
    try:
        abort.request("SIGTERM received")
        assert in_wait.wait(_GUARD_S)                 # the watchdog is waiting out its deadline
        with pytest.raises(_Exited) as e:
            abort.safe_point()
        assert e.value.code == 15 and commits == ["SIGTERM received"]
        release.set()                                 # the deadline "passes" after the safe point claimed
        abort._thread.join(_GUARD_S)
        assert not abort._thread.is_alive()
        assert ex.calls == [(15, "main")] and abort.claimed_by == "safe point"
    finally:
        release.set()
        abort.close()


def test_the_normal_end_of_training_claims_the_exit_first():
    abort, commits, ex, in_wait, release = _controlled()
    abort.start()
    try:
        assert abort.stand_down() is True
        abort.request("SIGTERM received")             # lands during the final save
        assert in_wait.wait(_GUARD_S)
        release.set()
        abort._thread.join(_GUARD_S)
        assert not abort._thread.is_alive()
        assert ex.calls == [] and commits == [] and abort.claimed_by == "training complete"
    finally:
        release.set()
        abort.close()


def test_a_safe_point_with_nothing_requested_does_nothing():
    abort, commits, ex, _in_wait, _release = _controlled()
    abort.safe_point()
    assert commits == [] and ex.calls == [] and abort.claimed_by is None
    abort.close()


# ---------------------------------------------------------------------------- the real handlers
class _StubModel:
    """Enough of a learner for the fallback test: it records dumps and saves."""

    def __init__(self) -> None:
        self.saves: List[str] = []
        self.dumps = 0
        self.num_timesteps = 0

    def dump_logs(self) -> None:
        self.dumps += 1

    def save(self, path: str) -> None:
        self.saves.append(path)


def test_a_signal_with_no_safe_point_falls_back_without_a_save(tmp_path, restore_signals, monkeypatch):
    monkeypatch.setattr(lifecycle, "save_model_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "record_checkpoint", lambda *a, **k: None)
    model, ex = _StubModel(), _Exit()
    abort = lifecycle._setup_signal_handlers(model, str(tmp_path), threading.Event(), None,
                                             lambda: 1e-3, lambda: 1, exit_fn=ex, wait_fn=lambda _s: None)
    try:
        try:
            signal.raise_signal(signal.SIGTERM)
        except _Exited:
            pass                                      # (the reverted form exits from inside the handler)
        assert ex.called.wait(_GUARD_S)
        assert model.saves == [] and model.dumps == 0, "the handler saved / dumped: not deferred"
        assert ex.calls == [(15, "deferred-abort-watchdog")]
    finally:
        abort.close()


def _learner(tmp_path):
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    from agents.training.instrumented_ppo_test import _CounterDictEnv
    from agents.training.rust_rollout.testkit import ToyVecEnv, attach_vec_collector
    from agents.training.train_logger import configure

    env = ToyVecEnv([(lambda: _CounterDictEnv()) for _ in range(N_ENVS)])
    model = attach_vec_collector(InstrumentedMaskablePPO(
        "MultiInputPolicy", env, n_steps=N_STEPS, batch_size=4, n_epochs=1, device="cpu", seed=0))
    model.set_logger(configure(str(tmp_path / "tb"), ["tensorboard"]))
    return model


def _wire(model, tmp_path, monkeypatch, ex):
    """The run's wiring (model_build's): the handlers + the graceful-restart callback's safe points."""
    from agents.training.graceful_restart_callback import GracefulRestartCallback

    monkeypatch.setattr(lifecycle, "save_model_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "record_checkpoint", lambda *a, **k: None)
    abort = lifecycle._setup_signal_handlers(model, str(tmp_path), threading.Event(), None,
                                             lambda: 1e-3, lambda: 1, exit_fn=ex)
    cb = GracefulRestartCallback()
    cb.abort_fn = abort
    cb.safe_point_fn = abort.safe_point
    return abort, cb


class _OwnerCheckingLock:
    """Stands in for TensorBoard `_AsyncWriter._lock` (a NON-reentrant `threading.Lock`): an acquire by the
    thread that already holds it is the F1 deadlock — recorded and raised instead of blocking forever."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: Optional[int] = None
        self.reentrant: List[str] = []
        self.acquires = 0

    def __enter__(self) -> "_OwnerCheckingLock":
        me = threading.get_ident()
        if self._owner == me:
            self.reentrant.append("re-entered by its holder")
            raise RuntimeError("F1: the TensorBoard writer lock re-entered by the thread holding it")
        self._lock.acquire()
        self._owner = me
        self.acquires += 1
        return self

    def __exit__(self, *exc: Any) -> None:
        self._owner = None
        self._lock.release()


def _tb_scalars(folder) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(folder), size_guidance={"scalars": 0})
    acc.Reload()
    return {tag: [(e.step, e.value) for e in acc.Scalars(tag)] for tag in acc.Tags()["scalars"]}


def test_a_signal_inside_a_tensorboard_flush_never_reenters_the_writer_lock(tmp_path, restore_signals,
                                                                            monkeypatch):
    model, ex = _learner(tmp_path), _Exit()
    abort, cb = _wire(model, tmp_path, monkeypatch, ex)
    tb = next(f for f in model.logger.output_formats if hasattr(f, "writer"))
    aw = tb.writer.file_writer.event_writer._async_writer
    lock = _OwnerCheckingLock()
    aw._lock = lock
    record_flush, fired = aw._writer.flush, []

    def flush_that_takes_a_signal() -> None:          # runs INSIDE `with aw._lock` (`_AsyncWriter.flush`)
        if not fired:
            fired.append(model.num_timesteps)
            assert lock._owner == threading.get_ident()   # the precondition: the lock is held right here
            signal.raise_signal(signal.SIGTERM)
        record_flush()

    aw._writer.flush = flush_that_takes_a_signal
    try:
        with pytest.raises(_Exited) as e:
            model.learn(total_timesteps=4 * ROWS, callback=[cb], reset_num_timesteps=False)
        assert fired == [ROWS], fired                  # the first dump's flush took the signal
        assert lock.reentrant == [], "the abort re-entered the TensorBoard lock (F1)"
        assert e.value.code == 15 and ex.calls == [(15, "main")]
        assert model._n_updates == 1                   # update 1 ran; the abort came at rollout 2's start
        assert os.path.isfile(tmp_path / "final_model_interrupted.zip")
        tb.writer.flush()
        scalars = _tb_scalars(tmp_path / "tb")
        # the pending scalars: update 1's `train/*`, written by the abort's dump at the step rollout 2 reached
        train_tags = sorted(t for t in scalars if t.startswith("train/"))
        assert "train/value_loss" in train_tags, train_tags
        assert {t: [s for s, _ in scalars[t]] for t in train_tags} == {t: [ROWS] for t in train_tags}
    finally:
        abort.close()


def test_a_signal_mid_update_saves_nothing_until_the_update_returns(tmp_path, restore_signals, monkeypatch):
    model, ex = _learner(tmp_path), _Exit()
    abort, cb = _wire(model, tmp_path, monkeypatch, ex)
    order: List[str] = []
    save, train = model.save, model.train

    def recording_save(path, *a, **k):
        order.append("save")
        return save(path, *a, **k)

    def train_that_takes_a_signal() -> None:
        order.append("update start")
        if model._n_updates == 0:
            signal.raise_signal(signal.SIGTERM)
            order.append("signal handled")
        train()
        order.append("update end")

    model.save = recording_save
    model.train = train_that_takes_a_signal
    try:
        with pytest.raises(_Exited):
            model.learn(total_timesteps=4 * ROWS, callback=[cb], reset_num_timesteps=False)
        assert order == ["update start", "signal handled", "update end", "save"], order
        assert os.path.isfile(tmp_path / "final_model_interrupted.zip")
        assert model._n_updates == 1 and ex.calls == [(15, "main")]
    finally:
        abort.close()
