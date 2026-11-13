"""A signal that lands INSIDE a compiled call under the compile lock still reaches the trainer's handler.

rb_b1c_s5101 (2026-11-12): the launcher's SIGTERM arrived while the compiled update was running under
`fail_on_recompile`. Python runs a signal handler in whatever frame the main thread is executing, here inside the
compiled call, where dynamo rejects every Python frame it has not compiled: the handler's body never ran, the
rejection surfaced in the update, the sentinel read it as a recompile and the child exited FATAL_CONFIG with no
save. `lifecycle._setup_signal_handlers` now registers every handler through `compile_control.lock_immune`.

Each test drives the REAL handlers and FAILS on a revert to a plain registration (the compiled call raises
"Detected recompile ..." instead of returning). CPU only and decided by call order: the signal is raised by the
compiled function at itself (`signal.raise_signal` is not traceable, so it runs natively between two compiled
halves of the frame), which is the one place the handler is guaranteed to run inside the call.
"""
from __future__ import annotations

import signal
import threading
from typing import Any, Callable, Iterator

import pytest
import torch

from agents.model.compile_control import CompileControl
from agents.training.eval_callback import consume_forced_eval_request
from main.exit_codes import TrainExitCode
from main.train import lifecycle
from main.train.deferred_abort_test import _SIGNALS, _Exit, _Exited, _StubModel
from main.train.deferred_abort_test import restore_signals  # noqa: F401 — the fixture

# dynamo says once per process that it cannot trace `raise_signal`: that graph break is the test's mechanism
pytestmark = pytest.mark.filterwarnings("ignore:Dynamo does not know how to trace the builtin")


@pytest.fixture
def control() -> Iterator[CompileControl]:
    """An installed `CompileControl` on an empty graph cache; released (stance, config, cache) on the way out."""
    torch._dynamo.reset()
    prev = (torch._dynamo.config.error_on_recompile, torch._dynamo.config.suppress_errors)
    ctl = CompileControl().install()
    try:
        yield ctl
    finally:
        ctl.uninstall()
        torch._dynamo.config.error_on_recompile, torch._dynamo.config.suppress_errors = prev
        torch._dynamo.reset()


def _update_that_signals_itself(sig: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """A compiled "update" that raises `sig` at its own process halfway through, compiled (both halves) BEFORE the
    lock with the signal ignored, as the learner's regions are prewarmed before the lock."""
    def update(x: torch.Tensor) -> torch.Tensor:
        y = x + 1
        signal.raise_signal(sig)             # not traceable: a graph break, run natively inside the compiled frame
        return y * 2

    compiled: Callable[[torch.Tensor], torch.Tensor] = torch.compile(update, backend="eager")
    prev = signal.signal(sig, signal.SIG_IGN)
    try:
        for _ in range(2):
            compiled(torch.ones(4))
    finally:
        signal.signal(sig, prev)
    return compiled


def _handlers(tmp_path: Any, monkeypatch: pytest.MonkeyPatch, ex: _Exit) -> tuple:
    monkeypatch.setattr(lifecycle, "save_model_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "record_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_model_hparams", lambda _model: {})
    model = _StubModel()
    abort = lifecycle._setup_signal_handlers(model, str(tmp_path), threading.Event(), None,
                                             lambda: 1e-3, lambda: 1, exit_fn=ex, start_watchdog=False)
    return model, abort


@pytest.mark.parametrize("name", _SIGNALS)
def test_a_signal_inside_a_compiled_call_under_the_lock_reaches_its_handler_and_the_call_returns(
        name, tmp_path, restore_signals, monkeypatch, control):  # noqa: F811
    sig = getattr(signal, name)
    update = _update_that_signals_itself(sig)
    consume_forced_eval_request()                     # (no request left over from another test)
    ex = _Exit()
    model, abort = _handlers(tmp_path, monkeypatch, ex)
    try:
        control.lock("the test's startup")
        out = update(torch.ones(4))                   # a plain handler: RuntimeError "Detected recompile ..." HERE
        assert torch.equal(out, torch.full((4,), 4.0)), "the compiled call did not finish its second half"
        if name == "SIGUSR2":
            assert consume_forced_eval_request() is True
            assert abort.reason is None and abort.checkpoint_reason is None
        elif name == "SIGUSR1":
            assert abort.checkpoint_reason == "SIGUSR1 received" and abort.reason is None
        else:
            assert abort.reason is not None and abort.reason.startswith(f"{name} received")
        assert model.saves == [] and ex.calls == [], "the handler only RECORDS; nothing is saved inside the call"
        control.check("after the signal")             # and the sentinel saw no compile and no rejection
        assert control.compiles_after_lock == 0 and control.rejected_after_lock == 0
    finally:
        abort.close()
        consume_forced_eval_request()


def test_a_SIGTERM_mid_update_under_the_lock_ends_in_a_save_and_INTERRUPTED(
        tmp_path, restore_signals, monkeypatch, control):  # noqa: F811
    """The incident end to end: the update returns, the next safe point saves and the exit is 15, never FATAL."""
    update = _update_that_signals_itself(signal.SIGTERM)
    ex = _Exit()
    model, abort = _handlers(tmp_path, monkeypatch, ex)
    try:
        control.lock("the test's startup")
        update(torch.ones(4))
        with control.guard("update end"):             # the trainer's own guard: nothing to report
            pass
        with pytest.raises(_Exited) as e:
            abort.safe_point("update end")
        assert e.value.code == int(TrainExitCode.INTERRUPTED) and ex.calls == [(15, "main")]
        assert len(model.saves) == 1 and model.dumps == 1
        saved = tmp_path / "final_model_interrupted.zip"
        assert saved.is_file() and (tmp_path / "latest.txt").read_text().strip() == saved.name
    finally:
        abort.close()


def test_setup_installs_a_handler_for_exactly_the_signals_delivered_above(
        tmp_path, restore_signals, monkeypatch):  # noqa: F811
    """A NEW handler in `_setup_signal_handlers` fails here until it joins `_SIGNALS`, where the test above
    delivers it inside a compiled call: no handler of the trainer goes unexercised under the lock."""
    before = {s: signal.getsignal(s) for s in signal.valid_signals()}
    _model, abort = _handlers(tmp_path, monkeypatch, _Exit())
    try:
        changed = {getattr(s, "name", str(s)) for s in before if signal.getsignal(s) is not before[s]}
        assert changed == set(_SIGNALS)
    finally:
        abort.close()
