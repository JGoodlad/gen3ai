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


def test_every_loop_event_is_a_safe_point_and_runs_first(monkeypatch):
    """gen3_deferred_abort_v1: a signal-requested abort runs at these events, before the callback's own work
    (so a rollout end with BOTH a pending abort and an elapsed interval aborts as the signal asked)."""
    cb, clock = _make(monkeypatch, "100")
    seen = []
    fired = []
    cb.safe_point_fn = seen.append
    cb.abort_fn = lambda reason: fired.append(reason)
    cb._on_training_start()
    cb._on_rollout_start()
    assert cb._on_step() is True
    clock.t = 100.0
    cb._on_rollout_end()
    cb._on_training_end()
    assert seen == ["training_start", "rollout_start", "step", "rollout_end", "training_end"]
    assert len(fired) == 1                              # the interval restart still fires after it


def test_safe_points_are_inert_until_wired(monkeypatch):
    cb, _ = _make(monkeypatch, None)
    assert cb.safe_point_fn is None
    cb._on_training_start()
    cb._on_rollout_start()
    assert cb._on_step() is True
    cb._on_rollout_end()
    cb._on_training_end()
