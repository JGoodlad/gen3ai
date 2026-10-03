"""The deferred abort (`main.train.deferred_abort`, gen3_deferred_abort_v1; P10 review F1).

A SIGINT / SIGTERM / SIGHUP only RECORDS a stop request; the main thread runs the abort (dump + save +
exit 15) at the next SAFE POINT; a watchdog exits WITHOUT a save if none comes within the deadline.

Each signal-level test drives the REAL handlers `lifecycle._setup_signal_handlers` installs and FAILS on a
revert to the handler-does-everything form (the handler calling the abort directly):

* a signal raised INSIDE a TensorBoard flush (the writer's non-reentrant lock held) never re-enters that
  lock — the old handler's dump did, which is the F1 deadlock (detected here by an owner-checking lock
  instead of hanging);
* a signal raised INSIDE an update saves nothing until the update has returned;
* a signal with no safe point to follow ends in the fallback's exit, with no save;
* (P10-A2) a SIGUSR1 raised INSIDE an update saves its forced checkpoint only after the update returned,
  and training continues;
* (P10-A2) a signal raised INSIDE an in-process eval cycle is honoured at the cycle's own safe point —
  saved and exited before the cycle's next host step.

P10-A2 also pins: the graceful restart's CALL of the run's `DeferredAbort` (it was not callable), the
exception path's exit claim, and the grace / deadline relations (here and in `launcher/reap_grace_test`).

Synchronised by events and by the order of calls, never by sleeps.
"""
from __future__ import annotations

import os
import signal
import threading
import zipfile
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


def test_the_deadline_covers_twice_the_worst_measured_stretch_without_a_safe_point():
    # `train/train_ms` max 50.6 s over 151 updates of the five `sizing_*` runs (2026-10-02/03), and the
    # stretch around an update ~8 s longer than its `train_ms` (P10-A2 pre-flight: 48.3 s beside
    # 40.4 s): that stretch slowed 2x by a contended box must still reach its safe point in time.
    worst_stretch = 50.6 + (48.3 - 40.4)
    assert SAFE_POINT_DEADLINE_SEC >= 2 * worst_stretch


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
        assert e.value.code == 15 and len(commits) == 1 and commits[0].startswith("SIGTERM received")
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
    train = model.train
    # The save is recorded where lifecycle marks it DONE (`_write_latest_txt`, after `model.save`
    # returned) — never by an instance-level `model.save` wrapper, which sb3 then tries to pickle INTO
    # the zip (a "save" that fails: the P10-A form of this test passed on exactly that).
    write_latest = lifecycle._write_latest_txt

    def recording_write_latest(*a, **k):
        order.append("save")
        return write_latest(*a, **k)

    monkeypatch.setattr(lifecycle, "_write_latest_txt", recording_write_latest)

    def train_that_takes_a_signal() -> None:
        order.append("update start")
        if model._n_updates == 0:
            signal.raise_signal(signal.SIGTERM)
            order.append("signal handled")
        train()
        order.append("update end")

    model.train = train_that_takes_a_signal
    try:
        with pytest.raises(_Exited):
            model.learn(total_timesteps=4 * ROWS, callback=[cb], reset_num_timesteps=False)
        assert order == ["update start", "signal handled", "update end", "save"], order
        assert zipfile.is_zipfile(tmp_path / "final_model_interrupted.zip")   # a REAL save, not a failed one
        assert model._n_updates == 1 and ex.calls == [(15, "main")]
    finally:
        abort.close()


# ---------------------------------------------------------------------------- P10-A2
def test_the_graceful_restart_calls_the_runs_deferred_abort(monkeypatch):
    """model_build wires `abort_fn = <the DeferredAbort>` and the interval restart CALLS it with a reason —
    a non-callable abort raised TypeError at the first interval restart (P10-A2)."""
    import agents.training.graceful_restart_callback as grc_mod
    from types import SimpleNamespace

    clock = SimpleNamespace(t=0.0)
    monkeypatch.setattr(grc_mod, "time", SimpleNamespace(monotonic=lambda: clock.t))
    monkeypatch.setenv(grc_mod._INTERVAL_ENV, "100")
    commits: List[str] = []
    ex = _Exit()
    abort = DeferredAbort(commits.append, exit_fn=ex)
    cb = grc_mod.GracefulRestartCallback()
    cb.abort_fn, cb.safe_point_fn = abort, abort.safe_point
    try:
        cb._on_training_start()
        clock.t = 100.0
        with pytest.raises(_Exited) as e:
            cb._on_rollout_end()
        assert e.value.code == 15 and ex.calls == [(15, "main")] and abort.claimed_by == "safe point"
        assert len(commits) == 1 and commits[0].startswith("restart interval")
    finally:
        abort.close()


def test_a_learn_that_raised_claims_the_exit_so_the_fallback_stands_down():
    abort, commits, ex, in_wait, release = _controlled()
    abort.start()
    try:
        abort.request("SIGTERM received")
        assert in_wait.wait(_GUARD_S)
        assert abort.stand_down("exception") is True   # model_build's except: claim, then the forensic save
        release.set()
        abort._thread.join(_GUARD_S)
        assert not abort._thread.is_alive()
        assert ex.calls == [] and commits == [] and abort.claimed_by == "exception"
    finally:
        release.set()
        abort.close()


def test_a_pending_stop_wins_over_a_pending_forced_checkpoint():
    saves: List[str] = []
    ex = _Exit()
    abort = DeferredAbort(lambda r: saves.append("abort"), checkpoint=lambda: saves.append("checkpoint"),
                          exit_fn=ex)
    try:
        abort.request_checkpoint("SIGUSR1 received")
        abort.request("SIGTERM received")
        with pytest.raises(_Exited):
            abort.safe_point("rollout_start")
        assert saves == ["abort"], saves             # the abort's save is the newer checkpoint
    finally:
        abort.close()


def test_a_forced_checkpoint_saves_once_at_a_safe_point_and_does_not_exit():
    saves: List[str] = []
    ex = _Exit()
    abort = DeferredAbort(lambda r: saves.append("abort"), checkpoint=lambda: saves.append("checkpoint"),
                          exit_fn=ex)
    try:
        abort.request_checkpoint("SIGUSR1 received")
        assert saves == []                           # the request saves nothing
        abort.safe_point("step")
        abort.safe_point("step")
        assert saves == ["checkpoint"] and ex.calls == [] and abort.claimed_by is None
    finally:
        abort.close()


def test_a_safe_point_off_the_main_thread_does_nothing():
    saves: List[str] = []
    abort = DeferredAbort(lambda r: saves.append("abort"), checkpoint=lambda: saves.append("checkpoint"),
                          exit_fn=_Exit())
    try:
        abort.request_checkpoint("SIGUSR1 received")
        abort.request("SIGTERM received")
        t = threading.Thread(target=abort.safe_point, args=("a thread",))
        t.start()
        t.join(_GUARD_S)
        assert saves == [] and abort.claimed_by is None
    finally:
        abort.close()


def test_sigusr1_mid_update_saves_only_after_the_update_returns(tmp_path, restore_signals, monkeypatch):
    """The real SIGUSR1 handler: the forced checkpoint is RECORDED; it is saved at the next safe point
    after the update, and training goes on (a revert to the in-handler save writes it mid-update)."""
    model, ex = _learner(tmp_path), _Exit()
    abort, cb = _wire(model, tmp_path, monkeypatch, ex)
    order: List[str] = []
    train = model.train
    # The save is recorded where lifecycle marks it DONE (`_write_latest_txt`, after `model.save`
    # returned) — never by an instance-level `model.save` wrapper, which sb3 then tries to pickle INTO
    # the zip (a "save" that fails: the P10-A form of this test passed on exactly that).
    write_latest = lifecycle._write_latest_txt

    def recording_write_latest(*a, **k):
        order.append("save")
        return write_latest(*a, **k)

    monkeypatch.setattr(lifecycle, "_write_latest_txt", recording_write_latest)

    def train_that_takes_a_signal() -> None:
        order.append("update start")
        if model._n_updates == 0:
            signal.raise_signal(signal.SIGUSR1)
            order.append("signal handled")
        train()
        order.append("update end")

    model.train = train_that_takes_a_signal
    try:
        model.learn(total_timesteps=3 * ROWS, callback=[cb], reset_num_timesteps=False)
        assert order[:4] == ["update start", "signal handled", "update end", "save"], order
        assert order.count("save") == 1 and ex.calls == [] and abort.claimed_by is None
        forced = [f for f in os.listdir(tmp_path / "checkpoints") if f.startswith("checkpoint_forced_")]
        assert len(forced) == 1 and zipfile.is_zipfile(tmp_path / "checkpoints" / forced[0]), forced
        assert forced == [f"checkpoint_forced_{ROWS:010d}_" + forced[0].rsplit("_", 1)[1]], forced
        assert (tmp_path / "latest.txt").read_text().strip() == os.path.join("checkpoints", forced[0])
        assert model._n_updates == 3
    finally:
        abort.close()


class _EvalModel(_StubModel):
    """A learner stub with what `run_rust_eval_cycle` reads: a policy, a gamma and a Rust collector whose
    `evaluator` is the test's (the executor's host-step loop, reduced to its safe-point contract)."""

    def __init__(self, evaluator: Any, order: List[str]) -> None:
        from types import SimpleNamespace

        super().__init__()
        self._order = order
        self.gamma = 0.99
        self.policy = SimpleNamespace(training=True, eval=lambda: None, train=lambda: None)
        self._rust_collector = SimpleNamespace(evaluator=evaluator, cfg=SimpleNamespace(run_seed=1))

    def save(self, path: str) -> None:
        self._order.append("save")
        super().save(path)


class _HostStepEvaluator:
    """`RustEvalCore.run_cycle`'s safe-point contract: `safe_point` at the top of every host step. The
    signal lands INSIDE host step 1 (as it would inside `svc.drain` / `core.step`)."""

    def __init__(self, order: List[str], host_steps: int = 4) -> None:
        self.order, self.host_steps = order, host_steps

    def run_cycle(self, pool: Any, run_dir: str, **kw: Any) -> Any:
        safe_point = kw.get("safe_point")
        for i in range(self.host_steps):
            if safe_point is not None:
                safe_point("eval cycle (step 7)")
            self.order.append(f"host step {i}")
            if i == 1:
                signal.raise_signal(signal.SIGTERM)
                self.order.append("signal handled")
        self.order.append("cycle end")
        raise AssertionError("the cycle finished: the stop was not honoured inside it")


def test_a_signal_during_an_eval_cycle_is_honoured_at_the_cycles_safe_point(tmp_path, restore_signals,
                                                                          monkeypatch):
    """The real handlers + the real `run_rust_eval_cycle` plumbing (cb.safe_point_fn -> run_cycle): a
    SIGTERM inside the in-process eval cycle saves and exits 15 at the NEXT HOST STEP, not after the
    cycle. FAILS on a revert of the plumbing (the cycle runs to its end)."""
    from types import SimpleNamespace

    from agents.training.rust_eval.launch import run_rust_eval_cycle

    monkeypatch.setattr(lifecycle, "save_model_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "record_checkpoint", lambda *a, **k: None)
    order: List[str] = []
    model, ex = _EvalModel(_HostStepEvaluator(order), order), _Exit()
    abort = lifecycle._setup_signal_handlers(model, str(tmp_path), threading.Event(), None,
                                             lambda: 1e-3, lambda: 1, exit_fn=ex)
    cb = SimpleNamespace(model=model, safe_point_fn=abort.safe_point, _model_dir=None)
    try:
        with pytest.raises(_Exited) as e:
            run_rust_eval_cycle(cb, pool=SimpleNamespace(items=[]), run_dir=str(tmp_path), step=7,
                                forensic=False, record=False)
        assert order == ["host step 0", "host step 1", "signal handled", "save"], order
        assert e.value.code == 15 and ex.calls == [(15, "main")] and abort.claimed_by == "safe point"
        assert model.saves == [os.path.join(str(tmp_path), "final_model_interrupted")]
        assert (tmp_path / "latest.txt").read_text().strip() == "final_model_interrupted.zip"
    finally:
        abort.close()
