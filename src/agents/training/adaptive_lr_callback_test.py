from unittest.mock import MagicMock

import pytest

from agents.training.adaptive_lr_callback import AdaptivePPOCallback


def _make_callback(
    initial_lr=3e-4,
    initial_epochs=10,
    clip_lo=0.05,
    clip_hi=0.15,
    lr_factor=1.5,
    min_lr=1e-5,
    max_lr=None,
    kl_lo=0.010,
    kl_hi=0.020,
    min_epochs=2,
    max_epochs=12,
    verbose=0,
):
    cb = AdaptivePPOCallback(
        initial_lr=initial_lr,
        initial_epochs=initial_epochs,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        lr_factor=lr_factor,
        min_lr=min_lr,
        max_lr=max_lr,
        kl_lo=kl_lo,
        kl_hi=kl_hi,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        verbose=verbose,
    )
    cb.model = MagicMock()
    cb.model.logger.name_to_value = {}
    cb.model.n_epochs = initial_epochs
    return cb


def _fire(cb, clip=None, kl=None):
    cb.model.logger.name_to_value = {}
    if clip is not None:
        cb.model.logger.name_to_value["train/clip_fraction"] = clip
    if kl is not None:
        cb.model.logger.name_to_value["train/approx_kl"] = kl
    cb._on_rollout_end()


# --- LR lever (clip_fraction) ---

def test_no_lr_adjustment_when_clip_in_band():
    cb = _make_callback(initial_lr=3e-4)
    _fire(cb, clip=0.10)  # inside [0.05, 0.15]
    assert cb.current_lr == pytest.approx(3e-4)


def test_lr_decreases_when_clip_too_high():
    cb = _make_callback(initial_lr=3e-4, lr_factor=1.5)
    _fire(cb, clip=0.20)  # above 0.15
    assert cb.current_lr == pytest.approx(3e-4 / 1.5)


def test_lr_increases_when_clip_too_low():
    cb = _make_callback(initial_lr=1e-4, lr_factor=1.5)
    _fire(cb, clip=0.02)  # below 0.05
    assert cb.current_lr == pytest.approx(1e-4 * 1.5)


def test_lr_clamped_at_min():
    cb = _make_callback(initial_lr=1.1e-5, min_lr=1e-5, lr_factor=1.5)
    _fire(cb, clip=0.99)
    assert cb.current_lr == pytest.approx(1e-5)


def test_lr_clamped_at_max():
    cb = _make_callback(initial_lr=3e-4, max_lr=3e-4, lr_factor=1.5)
    _fire(cb, clip=0.0)
    assert cb.current_lr == pytest.approx(3e-4)


def test_default_max_lr_is_2x_initial():
    cb = _make_callback(initial_lr=3e-4, max_lr=None)
    assert cb.max_lr == pytest.approx(6e-4)


def test_lr_schedule_updated_on_model():
    cb = _make_callback(initial_lr=3e-4, lr_factor=1.5)
    _fire(cb, clip=0.99)
    assert cb.model.lr_schedule is not None


def test_lr_multiple_adjustments_accumulate():
    cb = _make_callback(initial_lr=3e-4, lr_factor=1.5)
    _fire(cb, clip=0.20)
    _fire(cb, clip=0.20)
    assert cb.current_lr == pytest.approx(3e-4 / 1.5 / 1.5)


# --- n_epochs lever (approx_kl) ---

def test_no_epochs_adjustment_when_kl_in_band():
    cb = _make_callback(initial_epochs=6)
    _fire(cb, kl=0.015)  # inside [0.010, 0.020]
    assert cb.current_epochs == 6


def test_epochs_decrease_when_kl_too_high():
    cb = _make_callback(initial_epochs=6)
    _fire(cb, kl=0.025)  # above 0.020
    assert cb.current_epochs == 5


def test_epochs_increase_when_kl_too_low():
    cb = _make_callback(initial_epochs=6)
    _fire(cb, kl=0.005)  # below 0.010
    assert cb.current_epochs == 7


def test_epochs_clamped_at_min():
    cb = _make_callback(initial_epochs=2, min_epochs=2)
    _fire(cb, kl=0.999)
    assert cb.current_epochs == 2


def test_epochs_clamped_at_max():
    cb = _make_callback(initial_epochs=12, max_epochs=12)
    _fire(cb, kl=0.0)
    assert cb.current_epochs == 12


def test_epochs_updated_on_model():
    cb = _make_callback(initial_epochs=6)
    _fire(cb, kl=0.025)
    assert cb.model.n_epochs == 5


def test_epochs_multiple_adjustments_accumulate():
    cb = _make_callback(initial_epochs=6)
    _fire(cb, kl=0.025)  # 6 → 5
    _fire(cb, kl=0.025)  # 5 → 4
    assert cb.current_epochs == 4


# --- Silent when both signals absent ---

def test_no_adjustment_on_first_rollout():
    cb = _make_callback(initial_lr=3e-4, initial_epochs=6)
    cb.model.logger.name_to_value = {}
    cb._on_rollout_end()
    assert cb.current_lr == pytest.approx(3e-4)
    assert cb.current_epochs == 6


# --- Both levers fire independently in the same rollout ---

def test_both_levers_adjust_independently():
    cb = _make_callback(initial_lr=3e-4, initial_epochs=6, lr_factor=1.5)
    _fire(cb, clip=0.20, kl=0.025)  # clip high → LR down; kl high → epochs down
    assert cb.current_lr == pytest.approx(3e-4 / 1.5)
    assert cb.current_epochs == 5


def test_levers_can_move_in_opposite_directions():
    cb = _make_callback(initial_lr=1e-4, initial_epochs=6, lr_factor=1.5)
    _fire(cb, clip=0.02, kl=0.025)  # clip low → LR up; kl high → epochs down
    assert cb.current_lr == pytest.approx(1e-4 * 1.5)
    assert cb.current_epochs == 5
