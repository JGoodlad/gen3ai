"""The rank-tripwire state machine (gen3_distill_target_gate_v1;
design_advantage_gated_distillation.md §4.1 — implemented verbatim).

Pinned here: baseline capture (median of readings [W_SKIP, W_SKIP+W_BASE)), the EMA half-life,
the ×3 persistence rules for WARN and TRIP, the latch, the abort channel (`_on_step` returns
False ONLY under mode="abort" after a trip), and the missing-reading semantics — "no reading" is
never a trip, never an all-clear, and freezes (does not reset) every counter.
"""
import math
import types

import pytest
from stable_baselines3.common.logger import Logger

from agents.training.rank_tripwire import RankTripwireCallback

SKIP = RankTripwireCallback.W_SKIP
BASE = RankTripwireCallback.W_BASE


def _cb(mode="warn", drop=0.20):
    cb = RankTripwireCallback(mode=mode, drop=drop)
    cb.model = types.SimpleNamespace(logger=Logger(folder=None, output_formats=[]))
    return cb


def _feed(cb, value):
    """One train()-worth of signal: record rank/policy_pr (as train() would), then the rollout
    boundary where the callback reads it back."""
    cb.model.logger.record(RankTripwireCallback.SIGNAL, value)
    cb._on_rollout_end()


def _vals(cb):
    return cb.model.logger.name_to_value


# ------------------------------------------------------------------------------ baseline capture

def test_baseline_is_the_median_of_the_window_after_the_skip():
    cb = _cb()
    for _ in range(SKIP):
        _feed(cb, 100.0)                       # the resume/compile transient — must NOT enter
    assert cb._baseline is None
    for v in [float(i) for i in range(1, BASE + 1)]:   # 1..20 → median 10.5
        _feed(cb, v)
    assert cb._baseline == pytest.approx(10.5)
    _feed(cb, 10.5)
    assert _vals(cb)["rank/policy_pr_baseline"] == pytest.approx(10.5)
    assert "rank/policy_pr_ratio" in _vals(cb)


def test_no_ratio_and_no_thresholds_before_the_baseline_exists():
    cb = _cb()
    for _ in range(SKIP + BASE - 1):
        _feed(cb, 1e-3)                        # tiny readings — would trip instantly if judged
    assert cb._baseline is None
    assert "rank/policy_pr_ratio" not in _vals(cb)
    assert not cb._fired and cb._on_step() is True


# ------------------------------------------------------------------------ WARN / TRIP persistence

def _established(mode="warn", level=20.0):
    """A callback with baseline == ema == `level` (constant readings through skip + window)."""
    cb = _cb(mode=mode)
    for _ in range(SKIP + BASE):
        _feed(cb, level)
    assert cb._baseline == pytest.approx(level)
    assert cb._ema == pytest.approx(level)
    return cb


def _steps_until(pred, cb, value=0.0, cap=50):
    for n in range(1, cap + 1):
        _feed(cb, value)
        if pred(cb):
            return n
    raise AssertionError(f"condition not reached in {cap} readings")


def test_warn_fires_only_after_three_consecutive_readings_below_090(monkeypatch):
    events = []
    monkeypatch.setattr("agents.training.rank_tripwire.emit", events.append)
    cb = _established()
    d = cb._decay
    # Feed 17s: the EMA converges 20 → 17, so the ratio settles at 0.85 — inside the WARN band,
    # never inside the TRIP band (a warn-only degradation). ema_k = 17 + 3·d^k < 18 ⇔ d^k < 1/3.
    k_first_below = next(k for k in range(1, 40) if 17.0 + 3.0 * d ** k < 0.9 * 20.0)
    fire_at = k_first_below + cb.PERSISTENCE - 1
    for k in range(1, fire_at):
        _feed(cb, 17.0)
        assert not any("WARN" in e for e in events), f"warned early at reading {k}"
    _feed(cb, 17.0)
    assert sum("WARN" in e for e in events) == 1
    assert not cb._fired                       # 0.85 never crosses the 0.80 trip band
    # It does not repeat while still degraded…
    _feed(cb, 17.0)
    assert sum("WARN" in e for e in events) == 1
    # …but re-arms after a recovery.
    for _ in range(40):
        _feed(cb, 40.0)                        # pull the EMA back above the band
    assert cb._warn_streak == 0 and not cb._warned
    _steps_until(lambda c: sum("WARN" in e for e in events) == 2, cb, 17.0, cap=80)


def test_trip_latches_fires_the_loud_event_and_warn_mode_keeps_training(monkeypatch):
    events = []
    monkeypatch.setattr("agents.training.rank_tripwire.emit", events.append)
    cb = _established(mode="warn")
    _steps_until(lambda c: c._fired, cb, 0.0)
    assert any("TRIP" in e for e in events)
    assert _vals(cb)["rank/tripwire_fired"] == 1.0
    assert cb._on_step() is True               # warn mode NEVER stops learn()
    # The latch: a full recovery does not clear it.
    for _ in range(40):
        _feed(cb, 40.0)
    assert cb._fired and _vals(cb)["rank/tripwire_fired"] == 1.0


def test_abort_mode_returns_false_from_on_step_after_the_trip(monkeypatch):
    monkeypatch.setattr("agents.training.rank_tripwire.emit", lambda m: None)
    cb = _established(mode="abort")
    assert cb._on_step() is True
    _steps_until(lambda c: c._fired, cb, 0.0)
    assert cb._on_step() is False              # SB3 stops learn() cleanly; final save still runs


def test_trip_needs_three_consecutive_readings_not_three_total(monkeypatch):
    monkeypatch.setattr("agents.training.rank_tripwire.emit", lambda m: None)
    cb = _established()
    d = cb._decay
    k_below_trip = next(k for k in range(1, 30) if 20.0 * d ** k < 0.8 * 20.0)
    for _ in range(k_below_trip + 1):          # two consecutive below-trip readings…
        _feed(cb, 0.0)
    assert cb._trip_streak == 2 and not cb._fired
    # …one STRONG recovery reading lifts the EMA clear of the trip band in a single step (a mild
    # one would not — the EMA lags, and a still-below-band reading legitimately extends the
    # streak), resetting the counter…
    _feed(cb, 200.0)
    assert cb._trip_streak == 0 and not cb._fired
    _feed(cb, 0.0)                             # …so the next low reading starts over, no fire
    assert not cb._fired


# ------------------------------------------------------------------------------- missing reading

def test_a_missing_reading_is_no_reading_never_a_trip_and_never_an_all_clear(monkeypatch):
    monkeypatch.setattr("agents.training.rank_tripwire.emit", lambda m: None)
    cb = _cb()
    cb._on_rollout_end()                       # nothing recorded yet (the very first rollout)
    assert _vals(cb)["rank/tripwire_no_reading"] == 1.0
    assert cb._n_readings == 0 and not cb._fired

    cb = _established()
    while cb._trip_streak < cb.PERSISTENCE - 1:
        _feed(cb, 0.0)
    ema_before, streak_before = cb._ema, cb._trip_streak
    del cb.model.logger.name_to_value[RankTripwireCallback.SIGNAL]
    cb._on_rollout_end()                       # a capture failure mid-degradation
    assert cb._trip_streak == streak_before, "a missing reading advanced the persistence counter"
    assert cb._ema == ema_before, "a missing reading moved the EMA"
    assert not cb._fired, "a missing reading completed a trip"
    _feed(cb, 0.0)                             # the streak was frozen, not reset → this completes it
    assert cb._fired


def test_a_non_finite_reading_counts_as_no_reading():
    cb = _cb()
    _feed(cb, math.nan)
    assert cb._n_readings == 0
    assert _vals(cb)["rank/tripwire_no_reading"] == 1.0


# ------------------------------------------------------------------------------------ the wiring

def test_off_is_not_a_constructor_mode():
    """'off' means the callback is never registered (main.train.callbacks gates on it); the class
    itself refuses it so a mis-wired 'off' instance cannot sit silently doing warn-work."""
    with pytest.raises(ValueError):
        RankTripwireCallback(mode="off")


def test_callbacks_assembly_registers_the_tripwire_gated_on_off():
    import inspect

    import main.train.callbacks as callbacks_mod

    src = inspect.getsource(callbacks_mod)
    assert "RankTripwireCallback" in src, "the tripwire is not wired into build_callbacks"
    assert '!= "off"' in src, "the 'off' gate left the assembly — off would still register"


def test_thresholds_follow_the_drop_flag():
    cb = _cb(drop=0.30)
    assert cb.trip_threshold == pytest.approx(0.70)
    assert cb.warn_threshold == pytest.approx(0.85)
