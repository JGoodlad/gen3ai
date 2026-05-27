import math
from unittest.mock import MagicMock

import pytest

from agents.training.adaptive_lr_callback import AdaptivePPOCallback, AnnealingLRCallback


# ---------------------------------------------------------------------------
# AnnealingLRCallback
# ---------------------------------------------------------------------------

def _make_annealing(initial_lr=3e-4, total_steps=100_000, anneal_start_steps=40_000, anneal_min_lr=1e-5):
    return AnnealingLRCallback(
        initial_lr=initial_lr,
        total_steps=total_steps,
        anneal_start_steps=anneal_start_steps,
        anneal_min_lr=anneal_min_lr,
    )


def test_anneal_at_start_equals_initial_lr():
    cb = _make_annealing()
    assert cb.lr_at(40_000) == pytest.approx(3e-4)


def test_anneal_at_total_steps_equals_min_lr():
    cb = _make_annealing()
    assert cb.lr_at(100_000) == pytest.approx(1e-5)


def test_anneal_before_start_holds_initial_lr():
    cb = _make_annealing()
    assert cb.lr_at(0) == pytest.approx(3e-4)
    assert cb.lr_at(39_999) == pytest.approx(3e-4)


def test_anneal_after_total_steps_clamped_at_min():
    cb = _make_annealing()
    assert cb.lr_at(100_001) == pytest.approx(1e-5)
    assert cb.lr_at(999_999) == pytest.approx(1e-5)


def test_anneal_midpoint_is_midpoint_of_range():
    # At x=0.5 cosine gives exactly the midpoint of [min_lr, initial_lr].
    cb = _make_annealing(initial_lr=3e-4, total_steps=100_000, anneal_start_steps=0, anneal_min_lr=1e-4)
    mid = cb.lr_at(50_000)
    expected = 1e-4 + (3e-4 - 1e-4) * 0.5  # cos(π*0.5)=0, so 0.5*(1+0)=0.5
    assert mid == pytest.approx(expected)


def test_anneal_zero_duration_returns_min_lr():
    cb = _make_annealing(total_steps=40_000, anneal_start_steps=40_000)
    assert cb.lr_at(40_000) == pytest.approx(1e-5)


def test_anneal_current_lr_uses_model_timesteps():
    cb = _make_annealing()
    cb.model = MagicMock()
    cb.model.num_timesteps = 70_000  # halfway through the annealing window
    assert cb.current_lr == pytest.approx(cb.lr_at(70_000))


def _make_callback(
    initial_lr=3e-4,
    target_kl=0.015,
    kl_tolerance=0.3,
    lr_factor=1.5,
    min_lr=1e-5,
    max_lr=None,
    ema_alpha=0.3,
    verbose=0,
):
    cb = AdaptivePPOCallback(
        initial_lr=initial_lr,
        target_kl=target_kl,
        kl_tolerance=kl_tolerance,
        lr_factor=lr_factor,
        min_lr=min_lr,
        max_lr=max_lr,
        ema_alpha=ema_alpha,
        verbose=verbose,
    )
    cb.model = MagicMock()
    cb.model.logger.name_to_value = {}
    return cb


def _fire(cb, kl=None):
    cb.model.logger.name_to_value = {}
    if kl is not None:
        cb.model.logger.name_to_value["train/approx_kl"] = kl
    cb._on_rollout_end()


def test_no_adjustment_when_kl_in_band():
    cb = _make_callback(initial_lr=3e-4, target_kl=0.015, kl_tolerance=0.3)
    _fire(cb, kl=0.015)  # exactly at target, inside band
    assert cb.current_lr == pytest.approx(3e-4)


def test_lr_decreases_when_kl_too_high():
    cb = _make_callback(initial_lr=3e-4, target_kl=0.015, kl_tolerance=0.3, lr_factor=1.5)
    _fire(cb, kl=0.022)  # above 0.015 * 1.3 = 0.0195
    assert cb.current_lr == pytest.approx(3e-4 / 1.5)


def test_lr_increases_when_kl_too_low():
    cb = _make_callback(initial_lr=3e-4, target_kl=0.015, kl_tolerance=0.3, lr_factor=1.5)
    _fire(cb, kl=0.009)  # below 0.015 * 0.7 = 0.0105
    assert cb.current_lr == pytest.approx(3e-4 * 1.5)


def test_lr_clamped_at_min():
    cb = _make_callback(initial_lr=1.1e-5, min_lr=1e-5, lr_factor=1.5)
    _fire(cb, kl=0.999)  # very high KL → would go below min
    assert cb.current_lr == pytest.approx(1e-5)


def test_lr_clamped_at_max():
    cb = _make_callback(initial_lr=3e-4, max_lr=3e-4, lr_factor=1.5)
    _fire(cb, kl=0.0)  # very low KL → would exceed max
    assert cb.current_lr == pytest.approx(3e-4)


def test_default_max_lr_is_2x_initial():
    cb = _make_callback(initial_lr=3e-4, max_lr=None)
    assert cb.max_lr == pytest.approx(6e-4)


def test_lr_schedule_updated_on_model():
    cb = _make_callback(initial_lr=3e-4, lr_factor=1.5)
    _fire(cb, kl=0.999)  # high KL → LR decreases, schedule is set
    assert cb.model.lr_schedule is not None


def test_multiple_adjustments_accumulate():
    cb = _make_callback(initial_lr=3e-4, lr_factor=1.5)
    _fire(cb, kl=0.022)  # 3e-4 → 2e-4
    _fire(cb, kl=0.022)  # 2e-4 → ~1.33e-4
    assert cb.current_lr == pytest.approx(3e-4 / 1.5 / 1.5)


def test_no_adjustment_when_metric_absent():
    cb = _make_callback(initial_lr=3e-4)
    _fire(cb, kl=None)  # no approx_kl in log
    assert cb.current_lr == pytest.approx(3e-4)


# --- EMA smoothing behaviour ---

def test_single_spike_does_not_trigger_after_warmup():
    # Warm up: EMA converges to in-band value.
    # target=0.015, tolerance=0.3 → hi=0.0195
    # One spike to 0.030: EMA = 0.3*0.030 + 0.7*0.015 = 0.0195 (== hi, not > hi) → no change.
    cb = _make_callback(initial_lr=3e-4, ema_alpha=0.3)
    for _ in range(5):
        _fire(cb, kl=0.015)
    assert cb.current_lr == pytest.approx(3e-4)  # no change during warmup
    _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4)  # single spike absorbed by EMA


def test_sustained_high_kl_triggers_after_warmup():
    # After warmup at 0.015, two consecutive fires at 0.030 push EMA past hi=0.0195.
    # fire 1: EMA = 0.3*0.030 + 0.7*0.015 = 0.0195 → no change
    # fire 2: EMA = 0.3*0.030 + 0.7*0.0195 = 0.02265 → above 0.0195 → LR drops
    cb = _make_callback(initial_lr=3e-4, ema_alpha=0.3)
    for _ in range(5):
        _fire(cb, kl=0.015)
    _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4)   # still no change after one spike
    _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4 / 1.5)  # triggered on second sustained fire


def test_ema_resets_to_none_implicitly_on_first_value():
    # Cold-start: first observed KL seeds the EMA directly.
    cb = _make_callback(initial_lr=3e-4, ema_alpha=0.3)
    assert cb._kl_ema is None
    _fire(cb, kl=0.015)
    assert cb._kl_ema == pytest.approx(0.015)


def test_ema_alpha_controls_smoothing_rate():
    # With alpha=1.0 the EMA equals the raw value every rollout (no smoothing).
    cb = _make_callback(initial_lr=3e-4, ema_alpha=1.0)
    for _ in range(5):
        _fire(cb, kl=0.015)
    _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4 / 1.5)  # triggers immediately
