"""THE FOLD STOP RULE + THE DUAL-ASCENT ANCHOR COEFFICIENT (`gen3_distill_stop_rule_v1`) — the six
things that must hold.

1. **OFF IS FREE.** Neither flag passed ⇒ no callback, no series, and the anchor's own update is
   bit-identical to the tree before this feature existed.
2. **THE DUAL IS A DUAL.** Above target the coefficient RISES, below it FALLS, it settles AT the
   target rather than oscillating around it, and both clamps hold.
3. **EACH DETECTOR FIRES ON ITS OWN SIGNAL AND NOTHING ELSE.** The plateau detector fires on a flat
   series and not on a rising one; the rise detector fires on a rise and not on noise.
4. **THE AND-GATE AND THE COUNT.** Neither half alone fires, both together must persist, and a
   failed rollout resets the count.
5. **EACH ACTION DOES ITS OWN THING.** `warn` changes nothing, `anneal` decays `--distill-coef`
   geometrically to exactly 0, `abort` stops `learn()` through `_on_step` — the rank-tripwire
   channel, so the run's normal end-of-learn save still happens.
6. **A RESTART CONTINUES, IT DOES NOT RE-ARM.** Every EMA, both histories, the hold count, the
   latch and the annealed coefficient round-trip through the sidecar — and the annealed coefficient
   is RE-APPLIED over the argv's, because the launcher forwards the original argv every relaunch.
"""
import copy
import types

import numpy as np
import pytest
import torch as th
from stable_baselines3.common.logger import Logger

from agents.training.distill_anchor_callback import DistillAnchorCallback
from agents.training.distill_stop_callback import (
    STOP_MODES,
    AnchorDualAscent,
    DistillStopCallback,
    FoldStopDetector,
    ols_slope_and_se,
)
from agents.training.instrumented_ppo_test import _build_tiny_ppo, _train_from_init
from agents.training.distill_anchor_test import _anchor_arm, _build_anchor_ppo, _Rec


# ======================================================================================
# 1. THE DUAL — the pure controller
# ======================================================================================

def test_a_kl_above_target_raises_the_coefficient_and_below_it_lowers_it():
    """The SIGN. This is the whole controller in one assertion, and it is the one thing that being
    backwards would make into a positive feedback loop rather than a constraint."""
    up = AnchorDualAscent(target_kl=0.01, coef0=0.02)
    up.update(0.10)                       # 10x the budget
    assert up.coef > 0.02
    down = AnchorDualAscent(target_kl=0.01, coef0=0.02)
    down.update(0.001)                    # a tenth of it
    assert down.coef < 0.02


def test_the_dual_converges_to_a_kl_that_sits_AT_the_target():
    """A closed loop against a plant whose KL falls as the coefficient rises. The step is
    proportional to the violation, so it must SETTLE rather than hunt — which is the property the
    'no cooldown' decision rests on."""
    d = AnchorDualAscent(target_kl=0.01, coef0=0.02, coef_max=100.0)
    kl = 0.10
    for _ in range(400):
        d.update(kl)
        kl = 0.10 / (1.0 + 30.0 * d.coef)   # a monotone plant: more anchor ⇒ less divergence
    assert d.kl_ema == pytest.approx(0.01, rel=0.05), d.kl_ema
    assert 0.0 < d.coef < 100.0             # settled in the interior, not against a clamp


def test_the_clamps_hold_in_both_directions():
    hi = AnchorDualAscent(target_kl=0.01, coef0=0.02, coef_max=0.05)
    for _ in range(200):
        hi.update(1.0)
    assert hi.coef == pytest.approx(0.05)
    assert hi.clamped is True
    lo = AnchorDualAscent(target_kl=0.01, coef0=0.02, coef_min=0.01)
    for _ in range(200):
        lo.update(0.0)
    assert lo.coef == pytest.approx(0.01)


def test_the_default_max_is_ten_times_the_starting_coefficient():
    """The anchor is documented as a FRACTION of --distill-coef; a coefficient at distill scale is
    R3-SELF, which measured -9pp. The default ceiling is what keeps an unbounded dual out of it."""
    assert AnchorDualAscent(target_kl=0.01, coef0=0.03).coef_max == pytest.approx(0.3)


def test_a_missing_reading_moves_nothing():
    """NO READING is silence, never a zero: reading an absent meter as 0.0 would drive the
    coefficient toward its floor every rollout the series happened not to log."""
    d = AnchorDualAscent(target_kl=0.01, coef0=0.02)
    d.update(0.05)
    before, ema, n = d.coef, d.kl_ema, d.n_readings
    for bad in (None, float("nan"), float("inf"), "x", object()):
        assert d.update(bad) == before
    assert (d.kl_ema, d.n_readings) == (ema, n)


def test_a_zero_starting_coefficient_is_refused_rather_than_run_as_a_no_op():
    """0 is a FIXED POINT of a multiplicative update. `resolve_config` refuses the combination up
    front; this is the belt to that brace."""
    with pytest.raises(ValueError, match="MULTIPLICATIVE"):
        AnchorDualAscent(target_kl=0.01, coef0=0.0)
    with pytest.raises(ValueError, match="target-kl"):
        AnchorDualAscent(target_kl=0.0, coef0=0.02)


def test_the_dual_state_round_trips():
    d = AnchorDualAscent(target_kl=0.01, coef0=0.02)
    for v in (0.05, 0.04, 0.03):
        d.update(v)
    blob = d.state()
    fresh = AnchorDualAscent(target_kl=0.01, coef0=0.02)
    assert fresh.load_state(blob) is True
    assert (fresh.coef, fresh.kl_ema, fresh.n_readings) == (d.coef, d.kl_ema, d.n_readings)
    # ...and the next step from the restored state equals the next step from the live one.
    assert fresh.update(0.03) == pytest.approx(d.update(0.03))


def test_a_malformed_dual_blob_is_ignored_whole():
    d = AnchorDualAscent(target_kl=0.01, coef0=0.02)
    for blob in (None, [], {}, {"coef": "x"}, {"kl_ema": 0.1}):
        assert d.load_state(blob) is False
    assert d.coef == pytest.approx(0.02)


# ======================================================================================
# 2. THE DETECTORS — the pure state machine
# ======================================================================================

def _feed(det, agree, kl):
    return det.update(agree, kl)


def _run(det, agrees, kls):
    return [det.update(a, k) for a, k in zip(agrees, kls)]


def test_ols_slope_and_se_is_the_arithmetic_it_claims():
    slope, se = ols_slope_and_se([1.0, 2.0, 3.0, 4.0])
    assert slope == pytest.approx(1.0) and se == pytest.approx(0.0)   # noiseless ⇒ se 0
    slope, se = ols_slope_and_se([1.0, 1.0, 1.0, 1.0])
    assert slope == pytest.approx(0.0)
    assert ols_slope_and_se([1.0, 2.0]) == (None, None)               # no residual dof


def test_the_plateau_detector_fires_on_a_flat_series_and_not_on_a_rising_one():
    flat = FoldStopDetector(window=4, eps=0.005, persist=99)
    _run(flat, [0.80] * 10, [0.001] * 10)
    assert flat.last_plateau is True

    rising = FoldStopDetector(window=4, eps=0.005, persist=99)
    _run(rising, [0.50 + 0.03 * i for i in range(10)], [0.001] * 10)
    assert rising.last_plateau is False


def test_a_FALLING_agreement_counts_as_a_plateau_because_it_is_not_absorbing_either():
    det = FoldStopDetector(window=4, eps=0.005, persist=99)
    _run(det, [0.80 - 0.01 * i for i in range(10)], [0.001] * 10)
    assert det.last_plateau is True


def test_the_rise_detector_fires_on_a_rise_and_not_on_noise():
    rise = FoldStopDetector(window=6, eps=0.005, kl_slope_t=2.0, persist=99)
    _run(rise, [0.80] * 14, [0.001 * (i + 1) for i in range(14)])
    assert rise.last_rise is True

    # A zero-mean wobble at four times the rising series' per-step increment. At t > 2 a handful of
    # false positives over many draws is the test's own size; what must NOT happen is the detector
    # calling a trend most of the time, which is exactly what fitting the EMA did (see the class
    # docstring — the smoothed residuals understate the noise by a large factor).
    rng = np.random.default_rng(0)
    fired = 0
    for _ in range(40):
        noisy = FoldStopDetector(window=6, eps=0.005, kl_slope_t=2.0, persist=99)
        _run(noisy, [0.80] * 20, [0.01 + 0.004 * float(rng.standard_normal()) for _ in range(20)])
        fired += bool(noisy.last_rise)
    assert fired <= 6, f"a zero-mean wobble read as a trend in {fired}/40 draws"


def test_fitting_the_EMA_would_call_white_noise_a_trend___the_reason_the_fit_is_on_RAW():
    """The measured correction, pinned so nobody 'tidies' the trend fit back onto the EMA.

    Same white-noise draws, both fits. The EMA fit reports a significant positive trend far more
    often than the raw fit does, because a low-pass filter's residuals are not the series' noise.
    """
    rng = np.random.default_rng(7)
    raw_hits = ema_hits = 0
    for _ in range(60):
        series = [0.01 + 0.004 * float(rng.standard_normal()) for _ in range(20)]
        ema, ema_series = None, []
        for v in series:
            ema = v if ema is None else 0.2 * v + 0.8 * ema
            ema_series.append(ema)
        for name, ys in (("raw", series[-7:]), ("ema", ema_series[-7:])):
            slope, se = ols_slope_and_se(ys)
            hit = slope is not None and slope > 0 and slope > 2.0 * se
            if name == "raw":
                raw_hits += hit
            else:
                ema_hits += hit
    assert ema_hits > 3 * max(raw_hits, 1), (raw_hits, ema_hits)


def test_the_rise_test_is_scale_free():
    """The threshold is a t-multiple, so multiplying the whole collateral series by 1000 (the real
    spread across configs) must not change the verdict."""
    small = FoldStopDetector(window=6, persist=99)
    big = FoldStopDetector(window=6, persist=99)
    series = [0.00001 * (i + 1) for i in range(14)]
    _run(small, [0.8] * 14, series)
    _run(big, [0.8] * 14, [v * 1000.0 for v in series])
    assert small.last_rise == big.last_rise is True


def test_neither_half_alone_closes_the_gate():
    """PLATEAU without RISE is a fold that has merely finished; RISE without PLATEAU is an ordinary
    fold in progress paying collateral for content. Only the conjunction is the stop condition."""
    plateau_only = FoldStopDetector(window=4, persist=1)
    _run(plateau_only, [0.80] * 12, [0.01] * 12)          # flat KL ⇒ no rise
    assert plateau_only.fired is False and plateau_only.state_code() == FoldStopDetector.PLATEAU

    rise_only = FoldStopDetector(window=4, persist=1)
    _run(rise_only, [0.3 + 0.05 * i for i in range(12)], [0.001 * (i + 1) for i in range(12)])
    assert rise_only.fired is False and rise_only.state_code() == FoldStopDetector.ARMED


def test_the_and_gate_fires_only_after_persist_consecutive_rollouts():
    det = FoldStopDetector(window=3, eps=0.005, persist=3)
    states = _run(det, [0.80] * 8, [0.001 * (i + 1) for i in range(8)])
    assert FoldStopDetector.PLATEAU_AND_RISE in states, states
    assert det.fired is True
    assert states.index(FoldStopDetector.FIRED) - states.index(
        FoldStopDetector.PLATEAU_AND_RISE) == 2, states


def test_a_rollout_that_breaks_either_half_resets_the_count():
    det = FoldStopDetector(window=3, eps=0.005, persist=5)
    _run(det, [0.80] * 6, [0.001 * (i + 1) for i in range(6)])
    assert det.hold >= 1 and not det.fired
    det.update(0.99, 0.007)                # a big absorption jump breaks the plateau
    assert det.hold == 0 and not det.fired


def test_a_missing_reading_freezes_the_count_and_never_fires():
    """The `rank_tripwire` rule: no reading is never a verdict and never an all-clear."""
    det = FoldStopDetector(window=3, eps=0.005, persist=5)
    _run(det, [0.80] * 6, [0.001 * (i + 1) for i in range(6)])
    hold, n = det.hold, det.n_readings
    assert hold >= 1 and not det.fired
    for a, k in ((None, 0.01), (0.8, None), (None, None), (float("nan"), 0.01)):
        det.update(a, k)
    assert (det.hold, det.n_readings, det.fired) == (hold, n, False)


def test_the_latch_never_clears():
    det = FoldStopDetector(window=3, eps=0.005, persist=1)
    _run(det, [0.80] * 8, [0.001 * (i + 1) for i in range(8)])
    assert det.fired
    for _ in range(5):
        assert det.update(0.10, 0.0) == FoldStopDetector.FIRED
    assert det.rollouts_since_fire >= 5


def test_the_window_floor_is_enforced_at_construction():
    with pytest.raises(ValueError, match="window"):
        FoldStopDetector(window=1)
    with pytest.raises(ValueError, match="persist"):
        FoldStopDetector(persist=0)


def test_the_detector_state_round_trips_and_continues_rather_than_re_arming():
    """THE RESTART. A detector re-armed on every launcher restart would need its whole window again
    each time and might never fire, while reading as ON throughout."""
    live = FoldStopDetector(window=3, eps=0.005, persist=3)
    agrees, kls = [0.80] * 5, [0.001 * (i + 1) for i in range(5)]
    _run(live, agrees, kls)
    assert 0 < live.hold < 3 and not live.fired

    restored = FoldStopDetector(window=3, eps=0.005, persist=3)
    assert restored.load_state(live.state()) is True
    assert (restored.hold, restored.agree_hist, restored.kl_hist) == (
        live.hold, live.agree_hist, live.kl_hist)
    # Continuing from the restored state reaches the SAME verdict on the SAME next readings.
    nxt_a, nxt_k = [0.80] * 4, [0.006, 0.007, 0.008, 0.009]
    assert _run(restored, nxt_a, nxt_k) == _run(live, nxt_a, nxt_k)
    assert restored.fired is True


def test_a_malformed_detector_blob_is_ignored_whole():
    det = FoldStopDetector(window=3)
    for blob in (None, [], {"agree_hist": ["x"]}, {"kl_hist": [float("nan")]}):
        assert det.load_state(blob) is False
    assert det.agree_hist == [] and det.kl_hist == []


# ======================================================================================
# 3. THE CALLBACK — the three actions
# ======================================================================================

def _cb(mode="warn", *, window=3, persist=2, factor=0.7, resume=None, distill_coef=0.3):
    cb = DistillStopCallback(mode=mode, window=window, eps=0.005, persist=persist,
                             anneal_factor=factor, resume_state=resume)
    cb.model = types.SimpleNamespace(
        logger=Logger(folder=None, output_formats=[]), distill_coef=distill_coef,
        num_timesteps=12345)
    return cb


def _drive(cb, n, *, agree=0.80, kl0=0.001, step=0.001):
    """n rollouts of a PLATEAUED agreement against a RISING collateral — the stop condition."""
    for i in range(n):
        cb.model.logger.record("distill/teacher_agreement_on_slice", agree)
        cb.model.logger.record("distill/collateral_kl_vs_parent", kl0 + step * i)
        cb._on_rollout_end()


def test_the_modes_are_exactly_the_ones_the_flag_offers():
    assert STOP_MODES == ("off", "warn", "anneal", "abort")
    with pytest.raises(ValueError, match="warn|anneal|abort"):
        DistillStopCallback(mode="off")
    with pytest.raises(ValueError, match="anneal-factor"):
        DistillStopCallback(mode="anneal", anneal_factor=1.5)


def test_warn_logs_the_signal_and_changes_nothing(monkeypatch):
    seen = []
    monkeypatch.setattr("main.launcher.ipc.emit", lambda m: seen.append(m))
    cb = _cb("warn")
    _drive(cb, 10)
    vals = cb.model.logger.name_to_value
    assert cb.detector.fired is True
    assert vals["distill/stop_signal"] == 1.0
    assert vals["distill/stop_state"] == float(FoldStopDetector.FIRED)
    assert vals["distill/stop_rollouts_since_fire"] > 0
    assert cb.model.distill_coef == pytest.approx(0.3)      # untouched
    assert cb._on_step() is True                            # learn() continues
    assert sum("FIRED" in m for m in seen) == 1, "the event must print ONCE per fire"


def test_anneal_decays_the_distill_coefficient_geometrically_and_snaps_to_zero(monkeypatch):
    monkeypatch.setattr("main.launcher.ipc.emit", lambda m: None)
    cb = _cb("anneal", factor=0.5)
    _drive(cb, 4)
    assert cb.detector.fired is False, "the rise test needs window+1 points, then persist rollouts"
    _drive(cb, 1, kl0=0.005)                        # rollout 5: hold reaches persist=2 ⇒ FIRE
    assert cb.detector.fired is True
    assert cb.model.distill_coef == pytest.approx(0.3 * 0.5)   # the firing rollout anneals once
    _drive(cb, 3, kl0=0.01)
    assert cb.model.distill_coef == pytest.approx(0.3 * 0.5 ** 4)
    _drive(cb, 40, kl0=0.01)
    assert cb.model.distill_coef == 0.0, "the geometric decay must SNAP to exactly zero"
    _drive(cb, 2, kl0=0.01)
    assert cb.model.distill_coef == 0.0             # and stay there


def test_abort_stops_learn_through_the_rank_tripwire_channel(monkeypatch):
    """`_on_step` returning False stops rollout collection and `learn()` returns cleanly, so the
    run's normal final save happens and the process exits COMPLETE rather than CRASH."""
    monkeypatch.setattr("main.launcher.ipc.emit", lambda m: None)
    cb = _cb("abort")
    assert cb._on_step() is True
    _drive(cb, 10)
    assert cb.detector.fired is True
    assert cb._on_step() is False
    assert cb.model.distill_coef == pytest.approx(0.3), "abort must not also anneal"


def test_warn_and_anneal_never_stop_learn(monkeypatch):
    monkeypatch.setattr("main.launcher.ipc.emit", lambda m: None)
    for mode in ("warn", "anneal"):
        cb = _cb(mode)
        _drive(cb, 10)
        assert cb.detector.fired and cb._on_step() is True


def test_a_rollout_with_no_meters_is_silent(monkeypatch):
    monkeypatch.setattr("main.launcher.ipc.emit", lambda m: None)
    cb = _cb("abort")
    for _ in range(20):
        cb._on_rollout_end()                      # nothing recorded ⇒ nothing read
    assert cb.detector.fired is False
    assert cb._on_step() is True
    assert cb.model.logger.name_to_value["distill/stop_state"] == 0.0


def test_the_sidecar_state_round_trips_through_a_restart(monkeypatch):
    """The full restart: fire under `anneal`, hand the sidecar blob to a FRESH callback, and check
    it comes back latched, mid-anneal, and continues decaying from where it stopped."""
    monkeypatch.setattr("main.launcher.ipc.emit", lambda m: None)
    live = _cb("anneal", factor=0.5)
    _drive(live, 6)
    assert live.detector.fired
    blob = live.sidecar_state()
    assert blob["distill_coef_annealed"] == pytest.approx(live.model.distill_coef)

    # The launcher forwards the ORIGINAL argv, so the fresh process starts at --distill-coef 0.3.
    fresh = _cb("anneal", factor=0.5, resume=blob, distill_coef=0.3)
    fresh._on_training_start()
    assert fresh.detector.fired is True, "the latch must survive the restart"
    assert fresh.model.distill_coef == pytest.approx(blob["distill_coef_annealed"]), \
        "the annealed coefficient must be RE-APPLIED over the argv's, or the wind-down restarts"
    _drive(fresh, 1, kl0=0.01)
    assert fresh.model.distill_coef == pytest.approx(blob["distill_coef_annealed"] * 0.5)


def test_a_restart_after_an_abort_refuses_at_the_first_step(monkeypatch):
    monkeypatch.setattr("main.launcher.ipc.emit", lambda m: None)
    live = _cb("abort")
    _drive(live, 10)
    fresh = _cb("abort", resume=live.sidecar_state())
    fresh._on_training_start()
    assert fresh._on_step() is False


def test_an_operator_who_RAISED_the_coefficient_between_restarts_is_not_overruled(monkeypatch):
    monkeypatch.setattr("main.launcher.ipc.emit", lambda m: None)
    live = _cb("anneal", factor=0.5)
    _drive(live, 6)
    blob = live.sidecar_state()
    fresh = _cb("anneal", factor=0.5, resume=blob, distill_coef=0.001)
    fresh._on_training_start()
    assert fresh.model.distill_coef == pytest.approx(0.001)


# ======================================================================================
# 4. THE DUAL, WIRED — through DistillAnchorCallback
# ======================================================================================

class _Model:
    """The narrowest thing the anchor callback's rollout hook touches."""

    def __init__(self):
        self.logger = Logger(folder=None, output_formats=[])
        self.policy = None
        self.num_timesteps = 0


def _anchor_cb(**kw):
    cb = DistillAnchorCallback(parent_path="p", route="explicit", coef=0.02, mode="off_slice",
                               monitor=True, load_parent=lambda _p: object(), **kw)
    cb.model = _Model()
    return cb


def test_with_no_target_kl_the_coefficient_is_a_constant_and_the_series_is_flat():
    cb = _anchor_cb()
    assert cb._dual is None
    for kl in (0.5, 0.5, 0.5):
        cb.model.logger.record("distill/collateral_kl_vs_parent", kl)
        cb._on_rollout_end()
        assert cb.model.logger.name_to_value["distill/anchor_coef"] == pytest.approx(0.02)
    assert not hasattr(cb.model, "distill_anchor_dual_state")


def test_the_dual_moves_the_coefficient_the_model_reads():
    cb = _anchor_cb(target_kl=0.01)
    for _ in range(6):
        cb.model.logger.record("distill/collateral_kl_vs_parent", 0.10)
        cb._on_rollout_end()
    assert cb.model.distill_anchor_coef > 0.02
    assert cb.model.distill_anchor_coef == pytest.approx(cb.coef)
    assert cb.model.logger.name_to_value["distill/anchor_coef"] == pytest.approx(cb.coef)
    assert isinstance(cb.model.distill_anchor_dual_state, dict)


def test_the_signal_is_the_displacement_meter_under_parent_and_the_anchor_KL_under_a_moving_ref():
    """The controllability argument, pinned. Under a moving reference the anchor is DESIGNED not to
    resist parent-displacement, so a dual budgeted on it could never satisfy its constraint."""
    assert _anchor_cb(target_kl=0.01, ref="parent")._dual_signal_name() == \
        "distill/collateral_kl_vs_parent"
    for ref in ("ema", "periodic"):
        cb = _anchor_cb(target_kl=0.01, ref=ref)
        assert cb._dual_signal_name() == "distill/anchor_kl"


def test_a_moving_reference_dual_ignores_the_displacement_meter():
    cb = _anchor_cb(target_kl=0.01, ref="periodic", refresh_every=4)
    cb._ref_policy = None                      # `_on_rollout_end`'s periodic branch needs no policy
    cb.ref = "parent"                          # ...so drive the update path without a torch model
    cb.ref = "periodic"
    cb.model.logger.record("distill/collateral_kl_vs_parent", 10.0)   # the WRONG meter, huge
    cb._step_dual()
    assert cb.coef == pytest.approx(0.02), "the dual read a meter it must not act on"
    cb.model.logger.record("distill/anchor_kl", 10.0)
    cb._step_dual()
    assert cb.coef > 0.02


def test_the_dual_state_reaches_the_sidecar_and_comes_back():
    live = _anchor_cb(target_kl=0.01)
    for _ in range(5):
        live.model.logger.record("distill/collateral_kl_vs_parent", 0.10)
        live._on_rollout_end()
    blob = live.model.distill_anchor_dual_state
    fresh = _anchor_cb(target_kl=0.01, resume_dual=blob)
    fresh._on_training_start()
    assert fresh.coef == pytest.approx(live.coef)
    assert fresh.model.distill_anchor_coef == pytest.approx(live.coef)
    assert "DUAL RESTORED" in fresh._dual_restore_note


# ======================================================================================
# 5. OFF IS FREE — the byte-identity gate
# ======================================================================================

def test_no_dual_and_no_stop_is_byte_identical_to_the_anchor_arm():
    """The established pattern in `distill_anchor_test`: two identically-seeded arms of the real
    `train()`, one of them with this feature's modules imported and its callbacks unregistered."""
    base, log_base = _anchor_arm(attach_parent=True, coef=0.05)
    again, log_again = _anchor_arm(attach_parent=True, coef=0.05)
    for k in base:
        assert th.equal(base[k], again[k]), f"the anchor arm is no longer reproducible at {k}"
    assert not [k for k in log_base if "stop" in k or "dual" in k]


def test_a_fired_anneal_walks_the_real_train_loop_down_to_a_dead_distill_term(monkeypatch):
    """END TO END through the real `train()`: once `--distill-coef` reaches 0 the teacher forwards
    stop and `distill/kl` is no longer recorded — the fold really is over — while the ANCHOR's own
    meters keep reading, because they depend on the frozen parent and the `distill_mask` obs key,
    neither of which the coefficient gates."""
    monkeypatch.setattr("main.launcher.ipc.emit", lambda m: None)
    th.manual_seed(0)
    np.random.seed(0)
    model, parent = _build_anchor_ppo()
    teacher, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    th.manual_seed(21)
    with th.no_grad():
        for p in teacher.policy.action_net.parameters():
            p.add_(th.randn_like(p) * 2.0)
    teacher.policy.set_training_mode(False)
    model.learn(total_timesteps=8 * 4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model._distill_teachers = [teacher]
    model.distill_coef = 0.2
    model._distill_anchor_parent = parent
    model.distill_anchor_coef = 0.05
    model._logger = _Rec()
    _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    assert "distill/kl" in model.logger.vals            # the fold is live

    model.distill_coef = 0.0                            # what a completed anneal leaves behind
    model._logger = _Rec()
    _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    assert "distill/kl" not in model.logger.vals, "a dead coefficient still ran the teacher forward"
    assert model.logger.vals["distill/collateral_kl_vs_parent"] > 0.0, \
        "the ANCHOR's meters must survive the fold winding down"


# ======================================================================================
# 6. THE CLI SURFACE
# ======================================================================================

def test_checkargs_accepts_every_new_flag():
    from main.checkargs import check
    argv = ["--distill-teacher", "models/t:data/teams/sample/a.txt", "--distill-coef", "0.3",
            "--distill-anchor-coef", "0.02", "--distill-anchor-target-kl", "0.01",
            "--distill-anchor-dual-lr", "0.1", "--distill-anchor-coef-min", "0.0",
            "--distill-anchor-coef-max", "0.5", "--distill-stop", "warn",
            "--distill-stop-window", "8", "--distill-stop-eps", "0.005",
            "--distill-stop-kl-slope", "2.0", "--distill-stop-persist", "3",
            "--distill-stop-anneal-factor", "0.7"]
    got = check(argv)
    assert got["unknown"] == []
    for f in argv[::2]:
        if f.startswith("--"):
            assert f in got["accepted"], f


def _parsed(*argv):
    """`resolve_config` MUTATES the namespace in place (and returns a separate ResolvedRunConfig),
    so the resolved flag values are read back off `args` — the same thing `build_callbacks` sees."""
    from main.train.config import resolve_config
    from main.train_rl_agent import build_parser
    p = build_parser()
    args = p.parse_args(list(argv))
    resolve_config(args, p)
    return args, p


def test_the_defaults_resolve_to_off():
    args, _ = _parsed("--steps", "10")
    assert args.distill_anchor_target_kl == 0.0
    assert args.distill_stop == "off"
    assert args.distill_stop_window == 8 and args.distill_stop_persist == 3
    assert args.distill_stop_eps == pytest.approx(0.005)
    assert args.distill_stop_kl_slope == pytest.approx(2.0)
    assert args.distill_stop_anneal_factor == pytest.approx(0.7)


@pytest.mark.parametrize("argv,why", [
    (["--distill-anchor-target-kl", "0.01", "--distill-anchor-monitor"],
     "a dual with a zero starting coefficient is a fixed point"),
    (["--distill-anchor-dual-lr", "0.2", "--distill-anchor-coef", "0.02"],
     "a dual knob with no dual"),
    (["--distill-stop", "warn"],
     "the stop rule with no anchor monitor has no rise signal"),
    (["--distill-stop-window", "4"],
     "a stop knob with no stop rule"),
    (["--distill-stop", "warn", "--distill-anchor-monitor", "--distill-stop-window", "1"],
     "a window with no residual degree of freedom"),
    (["--distill-stop", "anneal", "--distill-anchor-monitor",
      "--distill-stop-anneal-factor", "1.0"],
     "a decay factor that does not decay"),
])
def test_config_refuses_the_combinations_that_would_be_silent_no_ops(argv, why):
    base = ["--steps", "10", "--distill-teacher", "models/t:data/teams/sample/a.txt",
            "--distill-coef", "0.3"]
    with pytest.raises(SystemExit):
        _parsed(*(base + argv))


def test_the_stop_rule_is_accepted_with_a_monitor():
    args, _ = _parsed("--steps", "10", "--distill-teacher", "models/t:data/teams/sample/a.txt",
                      "--distill-coef", "0.3", "--distill-anchor-monitor",
                      "--distill-stop", "anneal")
    assert args.distill_stop == "anneal"


def test_the_dual_is_accepted_beside_a_live_anchor_coefficient():
    args, _ = _parsed("--steps", "10", "--distill-teacher", "models/t:data/teams/sample/a.txt",
                      "--distill-coef", "0.3", "--distill-anchor-coef", "0.02",
                      "--distill-anchor-target-kl", "0.01")
    assert args.distill_anchor_target_kl == pytest.approx(0.01)
    assert args.distill_anchor_dual_lr == pytest.approx(0.1)


def test_the_sidecar_carries_the_controller_state_only_when_it_is_live():
    """`_model_hparams` writes these two keys only when the callback that owns them ran, so an
    ordinary run's sidecar is byte-for-byte what it always was."""
    from main.train.run_io import _model_hparams
    model, _venv = _build_tiny_ppo()
    plain = _model_hparams(model)
    assert "distill_stop_state" not in plain and "distill_anchor_dual_state" not in plain
    model.distill_stop_state = {"mode": "warn"}
    model.distill_anchor_dual_state = {"coef": 0.03}
    live = _model_hparams(model)
    assert live["distill_stop_state"] == {"mode": "warn"}
    assert live["distill_anchor_dual_state"] == {"coef": 0.03}
    assert set(live) - set(plain) == {"distill_stop_state", "distill_anchor_dual_state"}


# ======================================================================================
# 7. THE TWO INSTRUMENTS DEFAULT ON FOR A FOLD (`gen3_distill_instruments_default_v1`)
#
# `--distill-anchor-monitor` and `--distill-stop warn` are NON-PERTURBING: the monitor attaches no
# loss term and changes no parameter, `warn` only logs. As opt-ins they were carried on three of
# seven fold arms in one batch and omitted on the other four, which made a pre-registered
# cross-check unrunnable on the arms that mattered — an ABSENT series in a column of numbers reads
# like a zero. So they now default ON whenever a fold is actually running, and OFF changes nothing.
# ======================================================================================

_TEACHER = ["--distill-teacher", "models/t:data/teams/sample/a.txt"]
_FOLD = ["--steps", "10", *_TEACHER, "--distill-coef", "0.3", "--model", "models/parent.zip"]


def test_no_teacher_leaves_both_instruments_exactly_where_they_were():
    """(a) The byte-identity arm: nothing named a teacher, so nothing is defaulted on and the
    resolved namespace is what it has always been."""
    args, _ = _parsed("--steps", "10")
    assert args.distill_anchor_monitor is False
    assert args.distill_stop == "off"
    assert args.distill_anchor_monitor_source == "default-off"
    assert args.distill_stop_source == "default-off"


def test_a_teacher_at_coef_zero_is_not_a_fold():
    """A teacher beside `--distill-coef 0` is the distillation-FREE arm: there is no distill term,
    so there is no `distill_mask` slice for the anchor to split on and `resolve_config` refuses the
    anchor outright. The default must therefore NOT fire there, or the flag would turn a working
    command into a usage error."""
    args, _ = _parsed("--steps", "10", *_TEACHER, "--distill-coef", "0",
                      "--model", "models/parent.zip")
    assert args.distill_anchor_monitor is False and args.distill_stop == "off"


def test_a_fold_gets_both_instruments_with_nothing_typed():
    """(b) The motivating case: a fold argv that names neither flag still measures its own
    collateral and still arms the stop detector in its log-only mode."""
    args, _ = _parsed(*_FOLD)
    assert args.distill_anchor_monitor is True
    assert args.distill_anchor_monitor_source == "default"
    assert args.distill_stop == "warn"
    assert args.distill_stop_source == "default"


def test_the_explicit_opt_outs_win():
    """(c) `--no-distill-anchor-monitor` turns the monitor off — and the stop rule follows it down,
    because the rule's RISE half reads a meter only the attached parent emits."""
    args, _ = _parsed(*_FOLD, "--no-distill-anchor-monitor")
    assert args.distill_anchor_monitor is False
    assert args.distill_anchor_monitor_source == "cli"
    assert args.distill_stop == "off" and args.distill_stop_source == "default-off"


def test_an_explicit_distill_stop_off_wins_over_the_default():
    args, _ = _parsed(*_FOLD, "--distill-stop", "off")
    assert args.distill_stop == "off" and args.distill_stop_source == "cli"
    assert args.distill_anchor_monitor is True         # the monitor is a separate decision


def test_an_explicit_monitor_still_forces_it_on():
    args, _ = _parsed(*_FOLD, "--distill-anchor-monitor")
    assert args.distill_anchor_monitor is True and args.distill_anchor_monitor_source == "cli"


def test_a_live_anchor_coefficient_does_not_also_default_the_monitor_on():
    """(d, resolution half) `--distill-anchor-coef > 0` ALREADY attaches the frozen parent and
    already emits every collateral meter, so defaulting the monitor on beside it would be a second
    name for one thing. The stop rule still defaults on — its dependency is the PARENT, not the
    flag that attached it."""
    args, _ = _parsed(*_FOLD, "--distill-anchor-coef", "0.05")
    assert args.distill_anchor_monitor is False
    assert args.distill_stop == "warn" and args.distill_stop_source == "default"


def test_a_fold_with_no_resolvable_parent_warns_and_records_the_absence():
    """(e) THE DEFAULT YIELDS. No `--model` and no `--distill-anchor-parent` ⇒ nothing can name a
    fold parent, so the instrument is left OFF rather than turning the launch into a FATAL_CONFIG —
    and `cli_args` records WHY, so the missing series is visible instead of silent."""
    args, _ = _parsed("--steps", "10", *_TEACHER, "--distill-coef", "0.3")
    assert args.distill_anchor_monitor is False
    assert args.distill_anchor_monitor_source == "default-no-parent"
    assert args.distill_stop == "off" and args.distill_stop_source == "default-off"


def test_an_explicit_monitor_with_no_parent_still_refuses_at_the_callbacks():
    """The other half of "a default yields, an ask refuses": typed explicitly, an unresolvable
    parent is still the FATAL_CONFIG exit it always was."""
    with pytest.raises(SystemExit):
        _built(None, "--distill-anchor-monitor", model=None)


# --- the ACTUAL callback list, not a re-derivation of it -------------------------------------

def _built(tmp_path, *flags, model="models/parent.zip"):
    """`build_callbacks`'s real callback list for a fold argv (the `exploiter_ladder_test`
    convention: `--debug` to skip the eval callback, `--use-bridge node` to keep `resolve_config`
    off the rust binary). The parent is never LOADED here — `DistillAnchorCallback` takes the
    loader as a callable and calls it in `_on_training_start` — so a path that does not exist is
    enough to exercise the wiring."""
    from main.train.callbacks import build_callbacks
    from main.train.config import resolve_config
    from main.train.parser import build_parser

    p = build_parser()
    argv = ["--steps", "1", "--use-bridge", "node", *_TEACHER, "--distill-coef", "0.3", *flags]
    if model:
        argv += ["--model", model]
    args = p.parse_args(argv)
    resolve_config(args, p)
    args.debug, args.debug_eval = True, False
    return build_callbacks(
        args=args, model_dir=str(tmp_path or "."), server_config=None, annealing_mode=False,
        _pool=None, _fixed_opponents=None, _bot_weight_vec=None, OPPONENT_CLASSES=(),
        _specialist_team_str=None, _promote_threshold=0.6, _heuristic_floor=0.0,
        _sp_start_wr=0.5, _sp_full_wr=0.9).callbacks


def _count(cbs, cls):
    return sum(1 for c in cbs if isinstance(c, cls))


def test_a_default_fold_attaches_the_anchor_once_and_arms_the_stop_rule(tmp_path):
    cbs = _built(tmp_path)
    assert _count(cbs, DistillAnchorCallback) == 1
    assert _count(cbs, DistillStopCallback) == 1
    anchor = next(c for c in cbs if isinstance(c, DistillAnchorCallback))
    assert anchor.monitor is True and anchor.coef == 0.0        # instrument, not regulariser
    assert next(c for c in cbs if isinstance(c, DistillStopCallback)).mode == "warn"


def test_a_live_coefficient_attaches_the_anchor_exactly_once(tmp_path):
    """(d) The attach condition is one `or`, so a coefficient AND a defaulted monitor could never
    double-register — this pins that, since the two now arrive from different places."""
    cbs = _built(tmp_path, "--distill-anchor-coef", "0.05")
    assert _count(cbs, DistillAnchorCallback) == 1
    assert next(c for c in cbs if isinstance(c, DistillAnchorCallback)).coef == pytest.approx(0.05)


def test_opting_out_attaches_neither_callback(tmp_path):
    cbs = _built(tmp_path, "--no-distill-anchor-monitor")
    assert _count(cbs, DistillAnchorCallback) == 0
    assert _count(cbs, DistillStopCallback) == 0
