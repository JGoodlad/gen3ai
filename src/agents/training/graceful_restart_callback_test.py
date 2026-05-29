"""Unit tests for GracefulRestartCallback (no server, no env required)."""

import agents.training.graceful_restart_callback as grc_mod
from agents.training.graceful_restart_callback import (
    GracefulRestartCallback,
    _INTERVAL_ENV,
)


class _Clock:
    """Monkeypatchable monotonic clock."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


def _make(monkeypatch, interval_value: str | None):
    if interval_value is None:
        monkeypatch.delenv(_INTERVAL_ENV, raising=False)
    else:
        monkeypatch.setenv(_INTERVAL_ENV, interval_value)
    clock = _Clock()
    monkeypatch.setattr(grc_mod.time, "monotonic", clock.monotonic)
    return GracefulRestartCallback(), clock


def test_inert_without_env(monkeypatch):
    cb, clock = _make(monkeypatch, None)
    assert not cb.armed
    fired = []
    cb.abort_fn = lambda reason: fired.append(reason)
    cb._on_training_start()
    clock.t = 10_000.0
    cb._on_rollout_end()
    assert fired == []


def test_inert_when_interval_non_positive(monkeypatch):
    cb, clock = _make(monkeypatch, "0")
    assert not cb.armed
    fired = []
    cb.abort_fn = lambda reason: fired.append(reason)
    cb._on_training_start()
    clock.t = 10_000.0
    cb._on_rollout_end()
    assert fired == []


def test_inert_on_garbage_env(monkeypatch):
    cb, _ = _make(monkeypatch, "not-a-number")
    assert not cb.armed


def test_fires_once_after_interval(monkeypatch):
    cb, clock = _make(monkeypatch, "100")
    assert cb.armed
    fired = []
    cb.abort_fn = lambda reason: fired.append(reason)
    cb._on_training_start()  # records start at t=0

    # Before the interval elapses: no fire.
    clock.t = 50.0
    cb._on_rollout_end()
    assert fired == []

    # At/after the interval: fires exactly once.
    clock.t = 100.0
    cb._on_rollout_end()
    assert len(fired) == 1

    # Subsequent boundaries do not re-fire.
    clock.t = 250.0
    cb._on_rollout_end()
    assert len(fired) == 1


def test_does_not_fire_before_abort_fn_wired(monkeypatch):
    cb, clock = _make(monkeypatch, "100")
    cb._on_training_start()
    clock.t = 500.0
    # abort_fn is still None (signal handlers not wired yet) — must not crash.
    cb._on_rollout_end()
    assert not cb._fired


def test_on_step_is_noop(monkeypatch):
    cb, _ = _make(monkeypatch, "100")
    assert cb._on_step() is True
