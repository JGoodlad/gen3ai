"""gen3_signal_rate_metrics_v1 — the `signal/` group: advantage density × outcome entropy.

Four things are pinned here:

1. **The estimators are ARITHMETIC**, not "whatever numpy did" — every expectation below is an
   independently written closed form or a hand-computed constant, so a refactor that changes the
   moment definition (population vs sample variance, Fisher vs Pearson kurtosis) fails.
2. **Degenerate input never crashes and never fabricates.** A constant rollout has an undefined
   kurtosis; it must read NaN, not 0.0 — a 0.0 there is indistinguishable from a real "advantage
   mass is evenly smeared" reading, which is one of the two conclusions the metric exists to
   support.
3. **The kind-splits route.** A `signal/outcome_entropy_target` curve that silently carried pool
   games would be worse than no curve.
4. **It is OBSERVABILITY** — the advantage-density read leaves the parameter update and the
   advantages PPO fits byte-identical.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch as th

from agents.training.instrumented_ppo.signal_metrics import (
    OutcomeEntropyTracker,
    advantage_density_metrics,
    outcome_entropy,
)
from agents.training.signal_callback import OPP_CLASS_SUFFIX, SignalMetricsCallback

# ══ 1. Advantage density — the moments ════════════════════════════════════════


def test_known_array_matches_hand_computed_moments():
    """[-2,-1,0,1,2]: mean 0, population var 2, E|Â| 1.2, m4 6.8 ⇒ excess kurtosis 6.8/4 − 3."""
    m = advantage_density_metrics(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert m["adv_raw_std"] == pytest.approx(np.sqrt(2.0), abs=1e-12)
    assert m["adv_raw_abs_mean"] == pytest.approx(1.2, abs=1e-12)   # 6/5
    assert m["adv_kurtosis"] == pytest.approx(6.8 / 4.0 - 3.0, abs=1e-12)   # = −1.3, a flat-topped set


def test_moments_match_an_independent_closed_form_on_a_random_sample():
    """A second implementation of the same definitions, written from the formulas rather than from
    the code under test."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal(5000) * 3.5 + 1.25
    m = advantage_density_metrics(a)
    c = a - a.mean()
    assert m["adv_raw_std"] == pytest.approx(float(np.sqrt((c ** 2).mean())), rel=1e-12)
    assert m["adv_raw_abs_mean"] == pytest.approx(float(np.abs(a).mean()), rel=1e-12)
    expected_k = float((c ** 4).mean() / ((c ** 2).mean() ** 2) - 3.0)
    assert m["adv_kurtosis"] == pytest.approx(expected_k, rel=1e-12)
    # A normal sample is mesokurtic: EXCESS kurtosis ~0 (Fisher convention, not Pearson's 3).
    assert abs(m["adv_kurtosis"]) < 0.3


def test_kurtosis_is_positive_when_signal_is_concentrated_in_a_few_decisions():
    """The whole reason the third moment is here: a rollout whose advantage mass sits on a handful
    of decisive turns must read HEAVY-TAILED, and a rollout that spreads the same std evenly must
    not. Same `adv_raw_std` to 3 significant figures in both — so kurtosis is carrying information
    std cannot."""
    n = 4000
    sparse = np.zeros(n)
    sparse[:20] = 10.0                      # 0.5% of decisions carry everything
    spread = np.full(n, sparse.std())       # constant-magnitude, alternating sign
    spread[1::2] *= -1.0

    ks = advantage_density_metrics(sparse)
    kd = advantage_density_metrics(spread)
    assert ks["adv_raw_std"] == pytest.approx(kd["adv_raw_std"], rel=1e-9)
    assert ks["adv_kurtosis"] > 20.0, "a 0.5%-support rollout must read strongly heavy-tailed"
    assert kd["adv_kurtosis"] == pytest.approx(-2.0, abs=1e-9), "a two-point set is maximally flat"


def test_kurtosis_is_scale_free_but_std_is_not():
    """The UNITS claim in the docstring, as a test: PopArt's σ moves over a run, so only the shape
    metric is comparable across two rollouts measured in different return units."""
    rng = np.random.default_rng(11)
    a = rng.standard_normal(2000) ** 3          # something with a real tail
    base = advantage_density_metrics(a)
    scaled = advantage_density_metrics(a * 137.0)
    assert scaled["adv_kurtosis"] == pytest.approx(base["adv_kurtosis"], rel=1e-9)
    assert scaled["adv_raw_std"] == pytest.approx(base["adv_raw_std"] * 137.0, rel=1e-9)


# ══ 1b. Degenerate input — NaN-safe, never a crash, never a fabricated 0 ══════


def test_constant_advantages_report_nan_kurtosis_not_zero():
    m = advantage_density_metrics(np.full(64, 2.5))
    assert m["adv_raw_std"] == pytest.approx(0.0, abs=1e-12)
    assert m["adv_raw_abs_mean"] == pytest.approx(2.5, abs=1e-12)
    assert np.isnan(m["adv_kurtosis"]), "0/0 must be NaN — a 0.0 would read as 'evenly smeared'"


def test_all_zero_advantages_are_safe():
    m = advantage_density_metrics(np.zeros((16, 4)))
    assert m["adv_raw_std"] == 0.0 and m["adv_raw_abs_mean"] == 0.0
    assert np.isnan(m["adv_kurtosis"])


def test_empty_buffer_publishes_nothing():
    assert advantage_density_metrics(np.array([])) == {}


def test_a_tiny_sample_reports_nan_kurtosis():
    """Fewer than 4 points cannot support a fourth moment."""
    assert np.isnan(advantage_density_metrics(np.array([1.0, -1.0, 3.0]))["adv_kurtosis"])


def test_non_finite_entries_are_dropped_rather_than_poisoning_every_moment():
    finite = advantage_density_metrics(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    mixed = advantage_density_metrics(np.array([-2.0, -1.0, np.nan, 0.0, np.inf, 1.0, 2.0]))
    assert mixed == pytest.approx(finite)


def test_an_entirely_non_finite_rollout_reports_nan_across_the_board():
    m = advantage_density_metrics(np.array([np.nan, np.inf, -np.inf, np.nan]))
    assert set(m) == {"adv_raw_mean", "adv_raw_std", "adv_raw_abs_mean", "adv_kurtosis"}
    assert all(np.isnan(v) for v in m.values())


def test_a_2d_buffer_shaped_array_is_flattened():
    """`rollout_buffer.advantages` is [n_steps, n_envs]; the moments are over every transition."""
    a = np.arange(24, dtype=np.float64).reshape(6, 4)
    assert advantage_density_metrics(a) == pytest.approx(advantage_density_metrics(a.ravel()))


def test_float32_input_is_promoted_before_the_fourth_power():
    """`rollout_buffer.advantages` is float32, whose fourth power saturates to inf above ~4.3e9.
    Promotion to float64 is what keeps the shape metric readable on a large-advantage rollout."""
    a = np.array([-1e10, 0.0, 0.0, 1e10], dtype=np.float32)
    with np.errstate(over="ignore"):        # the overflow IS the premise being asserted
        assert not np.isfinite((a ** 4).mean()), "float32 alone would saturate to inf here"
    m = advantage_density_metrics(a)
    assert np.isfinite(m["adv_kurtosis"]) and np.isfinite(m["adv_raw_std"])
    assert m["adv_kurtosis"] == pytest.approx(-1.0, abs=1e-9)   # a symmetric two-spike set


# ══ 2. Outcome entropy — p(1−p) ═══════════════════════════════════════════════


@pytest.mark.parametrize("p,expected", [(0.5, 0.25), (0.0, 0.0), (1.0, 0.0),
                                        (0.25, 0.1875), (0.9, 0.09)])
def test_outcome_entropy_is_the_bernoulli_variance(p, expected):
    assert outcome_entropy(p) == pytest.approx(expected, abs=1e-12)


def test_outcome_entropy_peaks_at_one_half():
    grid = np.linspace(0.0, 1.0, 101)
    vals = [outcome_entropy(float(p)) for p in grid]
    assert grid[int(np.argmax(vals))] == pytest.approx(0.5)


def test_unknown_win_rate_is_nan_not_zero():
    """A window with no episodes yet must leave a GAP in TensorBoard, not report the same 0.0 a
    100%-loss wall would."""
    assert np.isnan(outcome_entropy(None))
    assert np.isnan(outcome_entropy(float("nan")))


# ══ 2b. The rolling tracker ═══════════════════════════════════════════════════


def test_rolling_window_tracks_a_known_win_pattern():
    t = OutcomeEntropyTracker(window=10)
    t.observe_many([True, True, False, False])          # p = 0.5
    m = t.metrics()
    assert m["outcome_win_rate"] == pytest.approx(0.5)
    assert m["outcome_entropy"] == pytest.approx(0.25)
    assert m["outcome_n"] == 4.0

    t.observe_many([True, True, True, True])            # 6/8
    assert t.metrics()["outcome_entropy"] == pytest.approx(0.75 * 0.25)


def test_the_window_evicts_so_the_meter_tracks_the_present():
    """A curriculum change must move the curve, not be averaged away by ancient games."""
    t = OutcomeEntropyTracker(window=4)
    t.observe_many([False] * 4)
    assert t.metrics()["outcome_entropy"] == pytest.approx(0.0)
    t.observe_many([True] * 2)                          # 2 wins, 2 losses left in a 4-window
    assert t.metrics()["outcome_entropy"] == pytest.approx(0.25)
    t.observe_many([True] * 2)                          # window is now all wins
    assert t.metrics()["outcome_entropy"] == pytest.approx(0.0)
    assert t.metrics()["outcome_n"] == 4.0


def test_nothing_is_published_before_the_first_episode():
    assert OutcomeEntropyTracker().metrics() == {}


def test_kind_splits_route_and_the_pool_sees_everything():
    t = OutcomeEntropyTracker(window=100)
    t.observe_many([True, True, True, True], kind="bots")      # 4/4 vs bots
    t.observe_many([True, False], kind="target")               # 1/2 vs the target
    t.observe(False)                                           # an untagged episode
    m = t.metrics()
    assert m["outcome_entropy_bots"] == pytest.approx(0.0)
    assert m["outcome_entropy_target"] == pytest.approx(0.25)
    assert m["outcome_n_bots"] == 4.0 and m["outcome_n_target"] == 2.0
    # POOLED = every episode including the untagged one: 5 wins of 7.
    assert m["outcome_n"] == 7.0
    assert m["outcome_win_rate"] == pytest.approx(5.0 / 7.0)
    assert "outcome_entropy_pool" not in m, "a kind with no episodes must publish no curve"


def test_per_kind_windows_are_independent():
    """A rare kind must not be starved out of its own window by a common one."""
    t = OutcomeEntropyTracker(window=3)
    t.observe_many([True, True, True], kind="bots")
    t.observe(False, kind="target")
    t.observe_many([True, True, True], kind="bots")
    m = t.metrics()
    assert m["outcome_n_target"] == 1.0 and m["outcome_entropy_target"] == pytest.approx(0.0)
    assert m["outcome_n"] == 3.0, "the POOLED window is the one that evicts"


# ══ 3. The callback — both rollout paths, and the kind mapping ════════════════


class _Logger:
    def __init__(self):
        self.rows = {}

    def record(self, key, value, exclude=None):
        self.rows[key] = value

    def dump(self, *a, **k):
        pass

    def close(self, *a, **k):
        pass


def _cb():
    """`logger` is a read-only BaseCallback property (it reads `model.logger`), so the sink is
    injected through a stand-in model — the same idiom as `exploiter_temp_callback_test`."""
    from types import SimpleNamespace

    cb = SignalMetricsCallback()
    cb.model = SimpleNamespace(logger=_Logger())
    return cb


def _done_info(won, opp_class):
    return {"win_outcome": 1.0 if won else 0.0, "opponent_class": opp_class}


def test_sync_locals_are_consumed_and_recorded_under_the_signal_prefix():
    cb = _cb()
    cb.locals = {"infos": [_done_info(True, 3), {}, _done_info(False, 3)],
                 "dones": [True, False, True]}
    cb._on_step()
    cb._on_rollout_end()
    assert cb.model.logger.rows["signal/outcome_entropy"] == pytest.approx(0.25)
    assert cb.model.logger.rows["signal/outcome_entropy_target"] == pytest.approx(0.25)
    assert cb.model.logger.rows["signal/outcome_n"] == 2.0
    assert all(k.startswith("signal/") for k in cb.model.logger.rows)


def test_async_wave_locals_are_consumed():
    """`--async-rollout` publishes wave_infos/wave_dones instead of infos/dones — the outcome meter
    covers it, because it needs no (step, env) buffer row (unlike WinProbLabelCallback)."""
    cb = _cb()
    cb.locals = {"wave_infos": [_done_info(True, 1), _done_info(True, 1)],
                 "wave_dones": [True, True]}
    cb._on_step()
    cb._on_rollout_end()
    assert cb.model.logger.rows["signal/outcome_entropy_pool"] == pytest.approx(0.0)
    assert cb.model.logger.rows["signal/outcome_n"] == 2.0


def test_a_non_done_step_contributes_nothing():
    cb = _cb()
    cb.locals = {"infos": [_done_info(True, 0)], "dones": [False]}
    cb._on_step()
    cb._on_rollout_end()
    assert cb.model.logger.rows == {}


def test_missing_locals_and_malformed_infos_are_survivable():
    cb = _cb()
    cb.locals = {}
    assert cb._on_step() is True
    cb.locals = {"infos": [None, {"no_outcome": 1}], "dones": [True, True]}
    assert cb._on_step() is True
    cb._on_rollout_end()
    assert cb.model.logger.rows == {}


def test_an_untagged_outcome_still_feeds_the_pooled_window():
    cb = _cb()
    cb.locals = {"infos": [{"win_outcome": 1.0}], "dones": [True]}
    cb._on_step()
    cb._on_rollout_end()
    assert cb.model.logger.rows["signal/outcome_n"] == 1.0
    assert not any(k.startswith("signal/outcome_entropy_") for k in cb.model.logger.rows)


def test_the_opponent_class_map_matches_the_wrapper_constants():
    """The integer is all that crosses the env pipe, so a renumbering in `MaskableAgentWrapper` must
    not silently relabel a curve."""
    from agents.training.wrappers import MaskableAgentWrapper as W
    assert OPP_CLASS_SUFFIX == {W.OPP_CLASS_BOT: "bots", W.OPP_CLASS_POOL: "pool",
                                W.OPP_CLASS_STABLE: "stable", W.OPP_CLASS_EXPLOITER: "target"}
    assert len(OPP_CLASS_SUFFIX) == W.N_OPP_CLASSES


def test_the_wrapper_tags_the_episode_end_info_with_the_selected_class():
    """The producer half of the contract, asserted on the real `step` source rather than a mock —
    the tag must be set in the same block as `win_outcome`, off `_opponent_class`."""
    import inspect

    from agents.training.wrappers import MaskableAgentWrapper
    src = inspect.getsource(MaskableAgentWrapper.step)
    assert 'info["opponent_class"]' in src and "_opponent_class" in src
    assert src.index('info["win_outcome"]') < src.index('info["opponent_class"]')


# ══ 4. It is OBSERVABILITY — the training path is untouched ═══════════════════


def test_train_is_byte_identical_with_and_without_the_advantage_read():
    """The `signal/adv_*` read must not perturb the update. Compared against the SAME train() with
    the estimator monkeypatched to a no-op — the only difference between the arms is whether the
    diagnostic ran."""
    from agents.training.instrumented_ppo import ppo as ppo_mod
    from agents.training.instrumented_ppo_test import _build_tiny_ppo, _train_from_init

    model, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    live = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)

    real = ppo_mod.advantage_density_metrics
    try:
        ppo_mod.advantage_density_metrics = lambda _a: {}          # the diagnostic removed entirely
        muted = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    finally:
        ppo_mod.advantage_density_metrics = real

    for k in live:
        assert th.allclose(live[k], muted[k], atol=0.0), f"the signal/ read perturbed {k}"


def test_the_advantages_ppo_fits_are_left_untouched():
    """Read-only over the buffer: not merely equal afterwards but the SAME bytes, and the estimator
    must not have promoted/normalized in place."""
    from agents.training.instrumented_ppo_test import _build_tiny_ppo

    model, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    model.learn(total_timesteps=8 * 4)
    adv = model.rollout_buffer.advantages
    before = adv.copy()
    m = advantage_density_metrics(adv)
    assert np.array_equal(adv, before) and adv.dtype == before.dtype
    assert set(m) == {"adv_raw_mean", "adv_raw_std", "adv_raw_abs_mean", "adv_kurtosis"}


def test_a_real_train_publishes_the_three_scalars():
    """End to end through `train()`: the keys exist, carry the `signal/` prefix, and are finite on a
    real (non-degenerate) rollout."""
    from agents.training.instrumented_ppo_test import _build_tiny_ppo

    model, _ = _build_tiny_ppo(n_steps=16, n_envs=4)
    model.learn(total_timesteps=16 * 4)
    rows = _Logger()
    model.set_logger(rows)
    model.train()
    for key in ("signal/adv_raw_std", "signal/adv_raw_abs_mean", "signal/adv_kurtosis"):
        assert key in rows.rows, f"{key} was not published by train()"
    assert rows.rows["signal/adv_raw_std"] >= 0.0


def test_the_signal_callback_is_registered_unconditionally():
    """ALWAYS-ON is the contract, asserted on the callback list `build_callbacks` ACTUALLY returns
    for a flagless argv — not on a source scan, which would pass on a list that never runs. The
    `--debug` path is the one exercised, so the meter is live on a smoke too."""
    import tempfile

    from main.train.callbacks import build_callbacks
    from main.train.config import resolve_config
    from main.train.parser import build_parser

    p = build_parser()
    args = p.parse_args(["--steps", "1", "--use-bridge", "node", "--debug-eval"])
    resolve_config(args, p)
    args.debug_eval, args.debug = False, True     # skip the eval callback (needs a server/pool)
    bundle = build_callbacks(
        args=args, model_dir=tempfile.mkdtemp(prefix="signal_metrics_test_"), server_config=None, annealing_mode=False,
        _pool=None, _fixed_opponents=None, _bot_weight_vec=None, OPPONENT_CLASSES=(),
        _specialist_team_str=None, _promote_threshold=0.6, _heuristic_floor=0.0,
        _sp_start_wr=0.5, _sp_full_wr=0.9)
    hits = [c for c in bundle.callbacks if isinstance(c, SignalMetricsCallback)]
    assert len(hits) == 1, f"expected exactly one SignalMetricsCallback, got {len(hits)}"


def test_the_ladder_rung_entropy_rides_the_gate_window():
    """`signal/outcome_entropy_rung` is emitted by the ladder callback (which owns the per-rung
    window) and is exactly p(1−p) of the number its promotion gate reads."""
    from agents.training.exploiter_ladder import ExploiterLadderCallback

    from types import SimpleNamespace

    cb = ExploiterLadderCallback.__new__(ExploiterLadderCallback)
    cb._idx, cb._promotions, cb._last_wr = 1, [], 0.6
    cb.model = SimpleNamespace(logger=_Logger())
    cb._record()
    assert cb.model.logger.rows["train/exploiter_rung_wr"] == pytest.approx(0.6)
    assert cb.model.logger.rows["signal/outcome_entropy_rung"] == pytest.approx(0.24)

    cb.model = SimpleNamespace(logger=_Logger())
    cb._last_wr = float("nan")           # no completed window yet
    cb._record()
    assert "signal/outcome_entropy_rung" not in cb.model.logger.rows
