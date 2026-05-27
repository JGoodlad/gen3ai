import math
from unittest.mock import MagicMock

import pytest

from agents.training.adaptive_lr_callback import AdaptivePPOCallback, TwoPhaseLRCallback


# ---------------------------------------------------------------------------
# TwoPhaseLRCallback — cosine math (Phase 2 in isolation)
# ---------------------------------------------------------------------------


def _make_two_phase(
    initial_lr=3e-4,
    total_steps=100_000,
    anneal_start_steps=40_000,
    anneal_min_lr=1e-5,
    min_lr=1e-5,
    max_lr=None,
    handoff_lr=None,
    cooldown_rollouts=0,
):
    # Most tests fire-and-check immediately. Default cooldown_rollouts=0 so
    # those tests aren't gated by the cooldown; dedicated cooldown tests pass
    # a positive value explicitly.
    return TwoPhaseLRCallback(
        initial_lr=initial_lr,
        total_steps=total_steps,
        anneal_start_steps=anneal_start_steps,
        anneal_min_lr=anneal_min_lr,
        min_lr=min_lr,
        max_lr=max_lr,
        ema_alpha=0.3,
        cooldown_rollouts=cooldown_rollouts,
        verbose=0,
        handoff_lr=handoff_lr,
    )


def test_cosine_at_anneal_start_equals_handoff():
    cb = _make_two_phase(handoff_lr=3e-4)
    assert cb._cosine_lr_at(40_000) == pytest.approx(3e-4)


def test_cosine_at_total_steps_equals_anneal_min():
    cb = _make_two_phase(handoff_lr=3e-4)
    assert cb._cosine_lr_at(100_000) == pytest.approx(1e-5)


def test_cosine_after_total_steps_clamped_at_min():
    cb = _make_two_phase(handoff_lr=3e-4)
    assert cb._cosine_lr_at(100_001) == pytest.approx(1e-5)
    assert cb._cosine_lr_at(999_999) == pytest.approx(1e-5)


def test_cosine_midpoint_is_midpoint_of_range():
    # At x=0.5 cosine gives exactly the midpoint of [anneal_min_lr, handoff_lr].
    cb = _make_two_phase(
        total_steps=100_000, anneal_start_steps=0, anneal_min_lr=1e-4, handoff_lr=3e-4
    )
    mid = cb._cosine_lr_at(50_000)
    expected = 1e-4 + (3e-4 - 1e-4) * 0.5
    assert mid == pytest.approx(expected)


def test_cosine_zero_duration_returns_anneal_min():
    cb = _make_two_phase(total_steps=40_000, anneal_start_steps=40_000, handoff_lr=3e-4)
    assert cb._cosine_lr_at(40_000) == pytest.approx(1e-5)


def test_cosine_uses_dynamic_handoff_not_initial_lr():
    # Critical property: the cosine starts at handoff_lr (the LR Phase 1
    # settled on), NOT at the constructor's initial_lr.
    cb = _make_two_phase(initial_lr=3e-4, handoff_lr=1.5e-4)
    assert cb._cosine_lr_at(40_000) == pytest.approx(1.5e-4)


# ---------------------------------------------------------------------------
# TwoPhaseLRCallback — phase classification
# ---------------------------------------------------------------------------


def test_phase_is_1_before_anneal_start():
    cb = _make_two_phase(anneal_start_steps=40_000)
    assert cb.phase(0) == 1
    assert cb.phase(39_999) == 1


def test_phase_is_2_at_anneal_start():
    cb = _make_two_phase(anneal_start_steps=40_000)
    assert cb.phase(40_000) == 2
    assert cb.phase(100_000) == 2


# ---------------------------------------------------------------------------
# TwoPhaseLRCallback — Phase 1: KL-driven adaptation
# ---------------------------------------------------------------------------


def _attach_mock_model(cb, num_timesteps=0, optimizer_lr=3e-4):
    cb.model = MagicMock()
    cb.model.num_timesteps = num_timesteps
    cb.model.logger.name_to_value = {}
    cb.model.n_epochs = 4
    cb.model.policy.optimizer.param_groups = [{"lr": optimizer_lr}]
    return cb


def _fire_two_phase(cb, kl=None, num_timesteps=None):
    if num_timesteps is not None:
        cb.model.num_timesteps = num_timesteps
    cb.model.logger.name_to_value = {}
    if kl is not None:
        cb.model.logger.name_to_value["train/approx_kl"] = kl
    cb._on_rollout_end()


def test_phase1_lr_decreases_when_kl_too_high():
    cb = _make_two_phase(initial_lr=3e-4, max_lr=6e-4)
    _attach_mock_model(cb, num_timesteps=1000)
    # ema_alpha=0.3, target=0.01, hi=0.013. kl=0.030 → first fire seeds ema at 0.030,
    # which is well above hi → immediate drop by default lr_factor=1.2.
    _fire_two_phase(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4 / 1.2)


def test_phase1_lr_increases_when_kl_too_low():
    cb = _make_two_phase(initial_lr=3e-4, max_lr=6e-4)
    _attach_mock_model(cb, num_timesteps=1000)
    # kl=0.003 → below lo=0.007 → increase by default lr_factor=1.2.
    _fire_two_phase(cb, kl=0.003)
    assert cb.current_lr == pytest.approx(3e-4 * 1.2)


def test_phase1_no_change_when_kl_in_band():
    cb = _make_two_phase(initial_lr=3e-4, max_lr=6e-4)
    _attach_mock_model(cb, num_timesteps=1000)
    _fire_two_phase(cb, kl=0.010)
    assert cb.current_lr == pytest.approx(3e-4)


def test_phase1_handoff_lr_stays_none():
    """Before crossing anneal_start_steps, handoff_lr must remain None."""
    cb = _make_two_phase(initial_lr=3e-4, anneal_start_steps=40_000)
    _attach_mock_model(cb, num_timesteps=10_000)
    _fire_two_phase(cb, kl=0.005)
    _fire_two_phase(cb, kl=0.020, num_timesteps=39_999)
    assert cb.handoff_lr is None


# ---------------------------------------------------------------------------
# TwoPhaseLRCallback — Phase 1 → Phase 2 handoff
# ---------------------------------------------------------------------------


def test_handoff_lr_captured_at_first_phase2_rollout():
    """When num_timesteps crosses anneal_start_steps, the optimizer's
    current LR (the value Phase 1 settled on) is captured as the cosine
    starting point — NOT args.lr from the constructor."""
    cb = _make_two_phase(initial_lr=3e-4, anneal_start_steps=40_000)
    # Phase 1 settled at 4.5e-4 (Phase 1 increased it from 3e-4 due to low KL).
    _attach_mock_model(cb, num_timesteps=40_001, optimizer_lr=4.5e-4)
    _fire_two_phase(cb, kl=0.010)
    assert cb.handoff_lr == pytest.approx(4.5e-4)


def test_handoff_lr_captured_once_only():
    """Subsequent Phase 2 rollouts do not overwrite the captured handoff_lr."""
    cb = _make_two_phase(initial_lr=3e-4, anneal_start_steps=40_000, total_steps=100_000)
    _attach_mock_model(cb, num_timesteps=40_001, optimizer_lr=4.5e-4)
    _fire_two_phase(cb, kl=0.010)
    captured = cb.handoff_lr
    # Now the optimizer LR drops due to cosine decay; handoff_lr must not change.
    cb.model.policy.optimizer.param_groups[0]["lr"] = 2.0e-4
    _fire_two_phase(cb, kl=0.010, num_timesteps=70_000)
    assert cb.handoff_lr == captured


def test_phase2_uses_cosine_not_kl():
    """In Phase 2, KL is ignored — the LR follows the cosine deterministically."""
    cb = _make_two_phase(
        initial_lr=3e-4, anneal_start_steps=40_000, total_steps=100_000, handoff_lr=3e-4
    )
    _attach_mock_model(cb, num_timesteps=70_000)
    # Very high KL — would shrink LR drastically in Phase 1.
    _fire_two_phase(cb, kl=0.999)
    # Phase 2 ignores KL: LR must match the cosine value at step 70_000.
    expected = cb._cosine_lr_at(70_000)
    assert cb.current_lr == pytest.approx(expected)


def test_phase2_with_preloaded_handoff_lr_skips_capture():
    """If handoff_lr is preloaded (e.g. from the sidecar on resume), the
    Phase 2 rollout must use that value verbatim — not the optimizer's."""
    cb = _make_two_phase(initial_lr=3e-4, anneal_start_steps=40_000, handoff_lr=4.5e-4)
    # Optimizer LR may differ from handoff_lr after the run was paused.
    _attach_mock_model(cb, num_timesteps=70_000, optimizer_lr=2.0e-4)
    _fire_two_phase(cb, kl=0.010)
    # Cosine should be computed from handoff_lr=4.5e-4, not optimizer_lr=2.0e-4.
    duration = 100_000 - 40_000
    x = (70_000 - 40_000) / duration
    expected = 1e-5 + (4.5e-4 - 1e-5) * 0.5 * (1.0 + math.cos(math.pi * x))
    assert cb.current_lr == pytest.approx(expected)
    # And handoff_lr stays at the preloaded value.
    assert cb.handoff_lr == pytest.approx(4.5e-4)


def test_on_training_start_in_phase2_without_handoff_uses_optimizer():
    """Resuming into Phase 2 with no persisted handoff_lr falls back to the
    optimizer's current LR (graceful migration for legacy runs)."""
    cb = _make_two_phase(initial_lr=3e-4, anneal_start_steps=40_000, handoff_lr=None)
    _attach_mock_model(cb, num_timesteps=70_000, optimizer_lr=2.5e-4)
    cb._on_training_start()
    assert cb.handoff_lr == pytest.approx(2.5e-4)


# ---------------------------------------------------------------------------
# AdaptivePPOCallback (unchanged behaviour)
# ---------------------------------------------------------------------------


def _make_callback(
    initial_lr=3e-4,
    target_kl=0.015,
    kl_tolerance=0.3,
    lr_factor=1.5,
    min_lr=1e-5,
    max_lr=None,
    ema_alpha=0.3,
    cooldown_rollouts=0,
    verbose=0,
):
    # Default cooldown_rollouts=0 — keeps the rollout-by-rollout tests below
    # exercising raw EMA/threshold logic. Dedicated cooldown tests opt in.
    cb = AdaptivePPOCallback(
        initial_lr=initial_lr,
        target_kl=target_kl,
        kl_tolerance=kl_tolerance,
        lr_factor=lr_factor,
        min_lr=min_lr,
        max_lr=max_lr,
        ema_alpha=ema_alpha,
        cooldown_rollouts=cooldown_rollouts,
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
    cb = _make_callback(initial_lr=3e-4, ema_alpha=0.3)
    for _ in range(5):
        _fire(cb, kl=0.015)
    assert cb.current_lr == pytest.approx(3e-4)
    _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4)


def test_sustained_high_kl_triggers_after_warmup():
    cb = _make_callback(initial_lr=3e-4, ema_alpha=0.3)
    for _ in range(5):
        _fire(cb, kl=0.015)
    _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4)
    _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4 / 1.5)


def test_ema_resets_to_none_implicitly_on_first_value():
    cb = _make_callback(initial_lr=3e-4, ema_alpha=0.3)
    assert cb._kl_ema is None
    _fire(cb, kl=0.015)
    assert cb._kl_ema == pytest.approx(0.015)


def test_ema_alpha_controls_smoothing_rate():
    cb = _make_callback(initial_lr=3e-4, ema_alpha=1.0)
    for _ in range(5):
        _fire(cb, kl=0.015)
    _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4 / 1.5)


# ---------------------------------------------------------------------------
# Cooldown after adjustment — prevents per-rollout compounding overshoot
# ---------------------------------------------------------------------------


def test_cooldown_blocks_consecutive_adjustments():
    """After an adjustment, the next ``cooldown_rollouts`` rollouts must NOT
    fire another adjustment, even if approx_kl stays out of band."""
    cb = _make_callback(initial_lr=3e-4, lr_factor=1.5, ema_alpha=1.0, cooldown_rollouts=5)
    cb._cooldown_remaining = 0  # bypass startup cooldown; this test targets post-adjustment cooldown
    # alpha=1.0 means EMA tracks the raw value with no smoothing.
    _fire(cb, kl=0.030)  # first fire: triggers ↓ and arms cooldown of 5
    assert cb.current_lr == pytest.approx(3e-4 / 1.5)
    # The next 5 rollouts at the same high KL must be suppressed.
    for _ in range(5):
        _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4 / 1.5)
    # The 6th rollout after the first fire is allowed to adjust again.
    _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4 / 1.5 / 1.5)


def test_ema_still_updates_during_cooldown():
    """EMA must keep tracking during cooldown so the post-cooldown decision
    reflects current KL, not stale pre-cooldown KL."""
    cb = _make_callback(initial_lr=3e-4, lr_factor=1.5, ema_alpha=0.5, cooldown_rollouts=3)
    cb._cooldown_remaining = 0  # bypass startup cooldown
    _fire(cb, kl=0.030)  # seed EMA at 0.030, fire ↓
    assert cb.current_lr == pytest.approx(3e-4 / 1.5)
    # During cooldown: kl drops back into band. EMA should track.
    for _ in range(3):
        _fire(cb, kl=0.010)
    # EMA should now be close to 0.010, in band — next fire should NOT adjust.
    assert cb._kl_ema == pytest.approx(0.5 * 0.010 + 0.5 * (0.5 * 0.010 + 0.5 * (0.5 * 0.010 + 0.5 * 0.030)))
    _fire(cb, kl=0.010)  # post-cooldown, EMA in band → no adjust
    assert cb.current_lr == pytest.approx(3e-4 / 1.5)


def test_cooldown_zero_means_no_gating():
    """cooldown_rollouts=0 keeps the old per-rollout behaviour for callers
    that want it (e.g. existing test fixtures)."""
    cb = _make_callback(initial_lr=3e-4, lr_factor=1.5, ema_alpha=1.0, cooldown_rollouts=0)
    _fire(cb, kl=0.030)
    _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4 / 1.5 / 1.5)


def test_cooldown_blocks_both_directions():
    """Cooldown applies whether the last fire was ↑ or ↓ — symmetric."""
    cb = _make_callback(initial_lr=3e-4, lr_factor=1.5, ema_alpha=1.0, cooldown_rollouts=3)
    cb._cooldown_remaining = 0  # bypass startup cooldown
    _fire(cb, kl=0.001)  # very low → ↑ to 4.5e-4 and arm cooldown
    assert cb.current_lr == pytest.approx(3e-4 * 1.5)
    # Cooldown should also block a ↓ fire, not just same-direction.
    for _ in range(3):
        _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4 * 1.5)
    # 4th rollout post-fire is unblocked → ↓ fires.
    _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4 * 1.5 / 1.5)


def test_two_phase_cooldown_applies():
    """TwoPhaseLRCallback's Phase 1 honors cooldown the same way."""
    cb = _make_two_phase(initial_lr=3e-4, max_lr=6e-4, cooldown_rollouts=5)
    _attach_mock_model(cb, num_timesteps=1000)
    cb._cooldown_remaining = 0  # bypass startup cooldown
    _fire_two_phase(cb, kl=0.030)  # fires ↓
    assert cb.current_lr == pytest.approx(3e-4 / 1.2)  # default lr_factor=1.2
    # Next 5 rollouts at the same high KL must be suppressed.
    for _ in range(5):
        _fire_two_phase(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4 / 1.2)
    # 6th rollout is unblocked.
    _fire_two_phase(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4 / 1.2 / 1.2)


def test_startup_cooldown_blocks_first_adjustment():
    """On a fresh callback (and on every launcher restart, which constructs a
    fresh callback), the first ``cooldown_rollouts`` rollouts must NOT fire
    an adjustment — the EMA hasn't had time to settle on representative state."""
    cb = _make_callback(initial_lr=3e-4, lr_factor=1.5, ema_alpha=1.0, cooldown_rollouts=5)
    # Even with extreme KL on the very first rollouts, no adjustment fires.
    for _ in range(5):
        _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4)
    # The 6th rollout is the first one allowed to adjust.
    _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4 / 1.5)


def test_startup_cooldown_zero_means_immediate_fire():
    """cooldown_rollouts=0 disables both startup and post-adjustment gating."""
    cb = _make_callback(initial_lr=3e-4, lr_factor=1.5, ema_alpha=1.0, cooldown_rollouts=0)
    _fire(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4 / 1.5)


def test_startup_cooldown_lets_ema_settle():
    """During startup cooldown, the EMA tracks. When the cooldown ends, the
    decision uses the settled EMA, not the very first noisy KL."""
    cb = _make_callback(initial_lr=3e-4, lr_factor=1.5, ema_alpha=0.5, cooldown_rollouts=3)
    # Noisy spike on the first rollout, then KL settles into the in-band region.
    _fire(cb, kl=0.030)  # would normally trigger ↓ — but cooldown blocks it
    _fire(cb, kl=0.010)
    _fire(cb, kl=0.010)
    assert cb.current_lr == pytest.approx(3e-4)  # untouched during startup
    # 4th rollout: cooldown elapsed. EMA has settled toward 0.010, in-band → no adjust.
    _fire(cb, kl=0.010)
    assert cb.current_lr == pytest.approx(3e-4)


def test_two_phase_startup_cooldown_applies():
    """TwoPhaseLRCallback's Phase 1 also honors startup cooldown."""
    cb = _make_two_phase(initial_lr=3e-4, max_lr=6e-4, cooldown_rollouts=4)
    _attach_mock_model(cb, num_timesteps=1000)
    for _ in range(4):
        _fire_two_phase(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4)  # no adjust during startup
    _fire_two_phase(cb, kl=0.030)
    assert cb.current_lr == pytest.approx(3e-4 / 1.2)  # 5th rollout: first fire


def test_two_phase_phase2_ignores_cooldown():
    """Cooldown is a Phase 1 concept. Phase 2 cosine ignores it — the cosine
    schedule fires every rollout regardless."""
    cb = _make_two_phase(
        initial_lr=3e-4,
        anneal_start_steps=40_000,
        total_steps=100_000,
        handoff_lr=3e-4,
        cooldown_rollouts=5,
    )
    _attach_mock_model(cb, num_timesteps=50_000)
    # Phase 2: every rollout sets LR to the cosine value.
    _fire_two_phase(cb, kl=0.010, num_timesteps=50_000)
    lr_a = cb.current_lr
    _fire_two_phase(cb, kl=0.010, num_timesteps=60_000)
    lr_b = cb.current_lr
    assert lr_a != lr_b  # cosine kept advancing
