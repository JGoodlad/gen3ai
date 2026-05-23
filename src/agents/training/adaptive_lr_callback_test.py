from unittest.mock import MagicMock

import pytest

from agents.training.adaptive_lr_callback import AdaptivePPOCallback


def _make_callback(
    initial_lr=3e-4,
    target_kl=0.015,
    kl_tolerance=0.3,
    lr_factor=1.5,
    min_lr=1e-5,
    max_lr=None,
    verbose=0,
):
    cb = AdaptivePPOCallback(
        initial_lr=initial_lr,
        target_kl=target_kl,
        kl_tolerance=kl_tolerance,
        lr_factor=lr_factor,
        min_lr=min_lr,
        max_lr=max_lr,
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
