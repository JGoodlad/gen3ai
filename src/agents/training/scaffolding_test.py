"""The SCAFFOLDING GAUGE — pinned on synthetic data with KNOWN answers, then on the live scalar.

Three things have to hold and none of them is checkable on real traces:

1. **The math is right where the answer is known.** A perfectly rank-agreeing pair must read
   exactly 0, an inverted one exactly 1, an independent one ~0.5 — and a DEGENERATE slice must
   read NaN rather than one of those three, because "no variance to rank with" is the PBRS
   endpoint the gauge is supposed to survive, not a divergence finding.
2. **The affine gauge's residual is separable.** The whole honesty claim of the calibrated gauge
   is that ``readout_penalty`` tells a reader how much of ``rms`` is the affine family failing
   rather than the heads disagreeing. That is only true if the number behaves — so the two
   regimes (linear-in-V outcome vs a sharply nonlinear one) are constructed here and asserted.
3. **The live scalar is OBSERVABILITY.** It must not perturb the update by a single bit, must
   publish nothing at all when there is no win-prob head, and must survive NaNs.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch as th

from agents.training.scaffolding import (
    affine_gauge, cluster_bootstrap_ci, constancy_row, gauge_slice, live_gauge_metrics,
    rank_gauge, spearman_rho,
)


class _Logger:
    """Captures `logger.record(key, value)` — the same stand-in `signal_metrics_test` uses."""

    def __init__(self):
        self.rows = {}

    def record(self, key, value, exclude=None):
        self.rows[key] = value

    def __getattr__(self, _name):
        return lambda *a, **k: None


# ══ 1. RANK GAUGE — known relationships, known values ═════════════════════════


def test_a_monotone_pair_reads_exactly_zero_divergence():
    """The definition, at the endpoint. Any strictly increasing map of V onto P(win) — here a
    sigmoid, which is the shape the real relationship actually has — is rank-identical, so the
    gauge must be exactly 0. This is the property that makes the rank form PopArt-proof."""
    v = np.linspace(-4.0, 4.0, 200)
    p = 1.0 / (1.0 + np.exp(-v))
    g = rank_gauge(v, p)
    assert g["rho"] == pytest.approx(1.0)
    assert g["gauge"] == pytest.approx(0.0)
    assert g["n"] == 200.0


def test_an_inverted_pair_reads_exactly_one():
    v = np.linspace(-4.0, 4.0, 200)
    g = rank_gauge(v, -v)
    assert g["rho"] == pytest.approx(-1.0)
    assert g["gauge"] == pytest.approx(1.0)


def test_an_independent_pair_reads_about_a_half():
    rng = np.random.default_rng(0)
    g = rank_gauge(rng.normal(size=4000), rng.normal(size=4000))
    assert abs(g["rho"]) < 0.06                      # ~N(0, 1/sqrt(n)) = 0.016
    assert g["gauge"] == pytest.approx(0.5, abs=0.03)


def test_an_affine_rescale_of_either_axis_changes_nothing():
    """PopArt moves V's scale over training. If the published curve moved with it, every trend
    reading would be an artifact of the normalizer."""
    rng = np.random.default_rng(1)
    v = rng.normal(size=300)
    p = 1.0 / (1.0 + np.exp(-(0.5 * v + rng.normal(scale=0.5, size=300))))
    base = rank_gauge(v, p)["gauge"]
    assert rank_gauge(17.3 * v - 42.0, p)["gauge"] == pytest.approx(base)
    assert rank_gauge(v, 0.001 * p + 0.5)["gauge"] == pytest.approx(base)


def test_a_constant_axis_is_NaN_and_not_zero_and_not_a_half():
    """THE PBRS ENDPOINT. Under a good frozen potential V_shaped is driven toward a constant; a
    correlation against a constant is 0/0. Reporting 0.0 would read as 'perfect agreement' and
    reporting 0.5 as 'independent' — both are confident statements about a slice that carries no
    information. NaN leaves a GAP in the curve, which is the honest rendering."""
    p = np.linspace(0.1, 0.9, 50)
    assert np.isnan(rank_gauge(np.full(50, 3.0), p)["rho"])
    assert np.isnan(rank_gauge(np.full(50, 3.0), p)["gauge"])
    assert np.isnan(rank_gauge(p, np.full(50, 0.5))["gauge"])


def test_ties_use_average_ranks_and_short_or_empty_input_is_NaN():
    assert spearman_rho([1.0, 1.0, 2.0, 2.0], [5.0, 5.0, 9.0, 9.0]) == pytest.approx(1.0)
    assert np.isnan(spearman_rho([1.0, 2.0], [1.0, 2.0]))          # n < 3
    assert np.isnan(spearman_rho([], []))
    assert rank_gauge([], [])["n"] == 0.0


def test_non_finite_rows_are_dropped_not_propagated():
    """A run with the head off records an all-NaN `win_probs`; a run WITH the head can still carry
    a stray NaN row. One NaN must not blank the whole slice."""
    v = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 6.0])
    p = np.array([0.1, 0.2, 0.3, np.inf, 0.5, 0.6])
    g = rank_gauge(v, p)
    assert g["n"] == 4.0 and g["gauge"] == pytest.approx(0.0)
    assert np.isnan(rank_gauge(np.full(6, np.nan), p)["gauge"])


def test_length_mismatch_raises_rather_than_silently_truncating():
    with pytest.raises(ValueError):
        rank_gauge([1.0, 2.0, 3.0], [0.1, 0.2])


# ══ 2. AFFINE GAUGE — the residual is separable ═══════════════════════════════


def test_when_the_head_equals_the_affine_readout_the_divergence_is_zero():
    """Construct the exact agreement case: outcomes ARE a linear function of V (so the fit
    recovers it), and the head publishes exactly that. rms must be 0 and the readout penalty 0."""
    rng = np.random.default_rng(2)
    n = 20_000
    v = rng.uniform(-2.0, 2.0, size=n)
    q = np.clip(0.2 * v + 0.5, 0.0, 1.0)             # the true P(win|V)
    y = (rng.uniform(size=n) < q).astype(float)
    a = affine_gauge(v, q, y)
    assert a["rms"] == pytest.approx(0.0, abs=0.01)
    assert a["bias"] == pytest.approx(0.0, abs=0.01)
    assert abs(a["readout_penalty"]) < 0.01
    assert a["a"] == pytest.approx(0.2, abs=0.02) and a["b"] == pytest.approx(0.5, abs=0.02)


def test_a_head_offset_by_a_constant_shows_up_as_exactly_that_bias():
    """The gauge is in PROBABILITY units, so a head that is uniformly 0.1 too optimistic must read
    bias = −0.1 and rms = 0.1 — not some rescaled quantity."""
    rng = np.random.default_rng(3)
    v = rng.uniform(-2.0, 2.0, size=2000)
    q = np.clip(0.2 * v + 0.5, 0.02, 0.98)
    y = (rng.uniform(size=2000) < q).astype(float)
    a = affine_gauge(v, np.clip(q + 0.1, 0, 1), y)
    assert a["bias"] == pytest.approx(-0.1, abs=0.02)
    assert a["rms"] == pytest.approx(0.1, abs=0.02)


def test_the_readout_penalty_convicts_the_AFFINE_FAMILY_not_the_heads():
    """THE HONESTY CLAIM, as a test. Make V→outcome sharply nonlinear (a step) and let the head be
    PERFECT. The heads do not disagree at all, yet the affine readout cannot express the step, so
    `rms` is large. `readout_penalty` must be the thing that says so: strongly positive, and a
    large share of the squared residual. Without this separation a reader would file a readout
    limitation as a divergence finding."""
    rng = np.random.default_rng(4)
    v = rng.uniform(-3.0, 3.0, size=4000)
    q = np.where(v > 0.0, 0.95, 0.05)                # the TRUE map — a step, unreachable by a line
    y = (rng.uniform(size=4000) < q).astype(float)
    a = affine_gauge(v, q, y)                        # head == truth, so any residual is the readout
    assert a["rms"] > 0.15
    # The penalty exceeds rms² — i.e. the readout's excess Brier accounts for the whole residual
    # and then some. Nothing here is a divergence between the two heads.
    assert a["readout_penalty"] > a["rms"] ** 2 > 0.02
    assert a["brier_v_affine"] > a["brier_head"]

    # The control: same V, same outcomes, but a LINEAR truth — the readout can express it, so the
    # penalty collapses while the machinery is otherwise identical.
    q_lin = np.clip(0.5 + 0.15 * v, 0.02, 0.98)
    y_lin = (rng.uniform(size=4000) < q_lin).astype(float)
    b = affine_gauge(v, q_lin, y_lin)
    assert b["readout_penalty"] < 0.01 and b["rms"] < a["rms"]


def test_brier_reference_scales_are_real_brier_scores():
    v = np.array([0.0, 1.0, 2.0, 3.0])
    p = np.array([0.25, 0.25, 0.75, 0.75])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    a = affine_gauge(v, p, y)
    assert a["brier_head"] == pytest.approx(np.mean((p - y) ** 2))
    assert a["base_rate"] == pytest.approx(0.5)
    assert a["brier_base"] == pytest.approx(0.25)


def test_a_constant_V_pins_the_slope_to_zero_and_FLAGS_itself():
    """The constancy endpoint again, on the affine side: with no variance in V the best linear
    readout IS the base rate. The result is legitimate, but a reader must be told the slope was
    not estimated — hence `v_constant`."""
    v = np.full(100, 2.5)
    y = np.array([1.0] * 60 + [0.0] * 40)
    a = affine_gauge(v, np.full(100, 0.6), y)
    assert a["v_constant"] == 1.0
    assert a["a"] == pytest.approx(0.0, abs=1e-9)
    assert a["b"] == pytest.approx(0.6, abs=1e-9)     # ȳ = 0.6, which the head also says
    assert a["rms"] == pytest.approx(0.0, abs=1e-9)


def test_an_empty_slice_is_all_NaN_and_never_a_fabricated_zero():
    a = affine_gauge([], [], [])
    assert a["n"] == 0.0
    for key in ("rms", "bias", "brier_head", "readout_penalty"):
        assert np.isnan(a[key]), key


# ══ 3. CONSTANCY ROW — the db9bb5c prediction ═════════════════════════════════


def test_the_constancy_row_reports_the_spread_and_its_within_between_split():
    """Two battles, V constant WITHIN each and different BETWEEN them: v_std is non-zero but
    within_frac is 0 — the look-alike FAILURE the raw std cannot distinguish from a flattened
    potential."""
    v = np.array([1.0] * 10 + [3.0] * 10)
    b = np.array(["A"] * 10 + ["B"] * 10)
    row = constancy_row(v, groups=b)
    assert row["v_mean"] == pytest.approx(2.0)
    assert row["v_std"] == pytest.approx(1.0)
    assert row["n_groups"] == 2.0
    assert row["within_std"] == pytest.approx(0.0)
    assert row["between_std"] == pytest.approx(1.0)
    assert row["within_frac"] == pytest.approx(0.0)


def test_a_fully_flattened_V_reads_zero_spread_which_CONFIRMS_the_theory():
    row = constancy_row(np.full(50, 4.2), groups=np.repeat(["A", "B"], 25))
    assert row["v_std"] == pytest.approx(0.0)
    assert row["v_iqr"] == pytest.approx(0.0)
    assert row["dispersion"] == pytest.approx(0.0)


def test_dispersion_is_scale_free_where_the_raw_std_is_not():
    """PopArt σ moves; the cross-run companion must not."""
    rng = np.random.default_rng(5)
    v = rng.normal(loc=5.0, scale=1.0, size=1000)
    assert constancy_row(10 * v)["v_std"] == pytest.approx(10 * constancy_row(v)["v_std"])
    assert constancy_row(10 * v)["dispersion"] == pytest.approx(
        constancy_row(v)["dispersion"], rel=1e-9)


def test_groups_must_be_one_label_per_value():
    with pytest.raises(ValueError):
        constancy_row([1.0, 2.0, 3.0], groups=["a", "b"])


# ══ 4. CLUSTER BOOTSTRAP — over BATTLES, not states ═══════════════════════════


def test_the_bootstrap_resamples_BATTLES_so_the_interval_is_not_fabricated_tight():
    """The recorded Simpson-trap lesson, as a test. 20 battles x 50 identical states each: an
    i.i.d. bootstrap over the 1000 states would report a near-zero-width interval, because every
    resample would land the same battle mixture. Resampling the 20 CLUSTERS must produce a
    visibly wide one."""
    rng = np.random.default_rng(6)
    per_battle = rng.normal(size=20)
    v = np.repeat(per_battle, 50)
    b = np.repeat(np.arange(20), 50)
    lo, hi = cluster_bootstrap_ci(lambda idx: float(np.mean(v[idx])), b, n_boot=500, seed=0)
    iid_se = float(np.std(v)) / np.sqrt(v.size)       # what an i.i.d. interval would have claimed
    assert np.isfinite(lo) and np.isfinite(hi)
    # The ratio is ~sqrt(states-per-battle) = sqrt(50) ≈ 7; 4x is the loose, unarguable form.
    assert (hi - lo) > 4 * (2 * 1.96 * iid_se)
    cluster_se = float(np.std(per_battle, ddof=1)) / np.sqrt(20)
    assert (hi - lo) == pytest.approx(2 * 1.96 * cluster_se, rel=0.35)


def test_a_single_cluster_returns_no_interval_rather_than_a_zero_width_one():
    lo, hi = cluster_bootstrap_ci(lambda idx: 1.0, np.zeros(100), n_boot=100, seed=0)
    assert np.isnan(lo) and np.isnan(hi)


def test_a_statistic_that_is_always_degenerate_yields_no_interval():
    lo, hi = cluster_bootstrap_ci(lambda idx: float("nan"), np.repeat(np.arange(5), 10),
                                  n_boot=100, seed=0)
    assert np.isnan(lo) and np.isnan(hi)


def test_gauge_slice_folds_all_three_blocks_with_intervals():
    rng = np.random.default_rng(7)
    n_b = 30
    v, p, y, b = [], [], [], []
    for i in range(n_b):
        won = float(i % 2)
        k = 12
        base = 2.0 * won - 1.0
        v.append(base + rng.normal(scale=0.3, size=k))
        p.append(np.clip(0.5 + 0.3 * (2 * won - 1) + rng.normal(scale=0.05, size=k), 0, 1))
        y.append(np.full(k, won))
        b.append(np.full(k, f"battle_{i}"))
    out = gauge_slice(np.concatenate(v), np.concatenate(p), np.concatenate(y), np.concatenate(b),
                      n_boot=200, seed=0)
    assert set(out) == {"rank", "affine", "constancy"}
    assert out["rank"]["gauge"] < 0.15                     # constructed to agree strongly
    assert out["rank"]["ci_lo"] <= out["rank"]["gauge"] <= out["rank"]["ci_hi"]
    assert out["affine"]["ci_lo"] <= out["affine"]["rms"] <= out["affine"]["ci_hi"]
    assert out["constancy"]["n_groups"] == float(n_b)


def test_gauge_slice_refuses_non_parallel_arrays():
    with pytest.raises(ValueError):
        gauge_slice([1.0, 2.0], [0.1, 0.2], [1.0, 0.0], ["a"])


# ══ 5. THE LIVE SCALAR — pure math half ══════════════════════════════════════


def test_the_live_form_is_the_rank_form_over_LOGITS_and_matches_the_probabilities():
    """The sigmoid is monotone, so passing logits must give bit-identical ρ to passing the
    probabilities. This is what licenses skipping the conversion in the hot path."""
    rng = np.random.default_rng(8)
    v = rng.normal(size=500)
    z = 0.8 * v + rng.normal(scale=0.4, size=500)
    p = 1.0 / (1.0 + np.exp(-z))
    assert live_gauge_metrics(v, z)["scaffolding_gauge"] == pytest.approx(
        rank_gauge(v, p)["gauge"])


def test_the_live_form_publishes_NOTHING_when_there_is_no_head():
    """A run with `--win-prob-mode none` must leave the curve absent, not flat at zero."""
    assert live_gauge_metrics(np.zeros(5), None) == {}
    assert live_gauge_metrics(None, np.zeros(5)) == {}
    assert live_gauge_metrics([], []) == {}
    assert live_gauge_metrics(np.zeros(5), np.zeros(3)) == {}      # shape mismatch, not a crash


def test_the_live_form_is_NaN_safe_and_reports_the_surviving_count():
    v = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    z = np.array([0.1, 0.2, 0.3, np.inf, 0.5])
    m = live_gauge_metrics(v, z)
    assert m["scaffolding_n"] == 3.0 and m["scaffolding_gauge"] == pytest.approx(0.0)
    all_nan = live_gauge_metrics(np.full(5, np.nan), np.full(5, np.nan))
    assert all_nan == {}                                          # nothing finite ⇒ no key at all
    const = live_gauge_metrics(np.full(5, 1.0), np.arange(5.0))
    assert np.isnan(const["scaffolding_gauge"]) and const["scaffolding_n"] == 5.0


# ══ 6. THE LIVE SCALAR — inside train(), and it is OBSERVABILITY ═════════════


def _attach_fake_win_head(model, *, mode="read_only", broken=False):
    """Give the stock `MultiInputPolicy`'s extractor the two attributes the gauge reads.

    A forward hook sets `last_win_prob_logits` per minibatch from the extractor's own OUTPUT, so
    the pair varies across minibatches exactly as the real head's would — a constant stub would
    make the whole rank read degenerate and the assertion vacuous. `broken=True` publishes NaNs,
    which is the NaN-safety arm."""
    fe = model.policy.features_extractor
    fe.win_prob_mode = mode

    def _hook(_mod, _inp, out):
        n = out.shape[0]
        if broken:
            fe.last_win_prob_logits = th.full((n, 1), float("nan"))
        else:
            fe.last_win_prob_logits = out.detach().sum(dim=-1, keepdim=True) * 0.37
    fe.register_forward_hook(_hook)
    return fe


def test_train_publishes_the_scaffolding_scalars_when_a_win_head_exists():
    from agents.training.instrumented_ppo_test import _build_tiny_ppo

    model, _ = _build_tiny_ppo(n_steps=16, n_envs=4)
    model.learn(total_timesteps=16 * 4)
    _attach_fake_win_head(model)
    rows = _Logger()
    model.set_logger(rows)
    model.train()
    assert "train/scaffolding_gauge" in rows.rows
    assert "train/scaffolding_rho" in rows.rows
    assert rows.rows["train/scaffolding_n"] == 64.0        # the whole rollout, epoch 0 only
    g = rows.rows["train/scaffolding_gauge"]
    assert np.isnan(g) or 0.0 <= g <= 1.0


def test_train_publishes_NO_scaffolding_key_without_a_win_head():
    """ALWAYS-ON means always-on WHEN THE HEAD EXISTS. A run without one must write no curve."""
    from agents.training.instrumented_ppo_test import _build_tiny_ppo

    model, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    model.learn(total_timesteps=8 * 4)
    rows = _Logger()
    model.set_logger(rows)
    model.train()
    assert not any(k.startswith("train/scaffolding") for k in rows.rows)


def test_a_NaN_win_head_leaves_a_GAP_and_never_crashes_train():
    from agents.training.instrumented_ppo_test import _build_tiny_ppo

    model, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    model.learn(total_timesteps=8 * 4)
    _attach_fake_win_head(model, broken=True)
    rows = _Logger()
    model.set_logger(rows)
    model.train()                                          # must not raise
    assert not any(k.startswith("train/scaffolding") for k in rows.rows)


def test_train_is_byte_identical_with_and_without_the_scaffolding_read():
    """It is a DIAGNOSTIC. Compared against the same train() with the estimator monkeypatched to a
    no-op — the only difference between the arms is whether the read ran at all."""
    from agents.training.instrumented_ppo import ppo as ppo_mod
    from agents.training.instrumented_ppo_test import _build_tiny_ppo, _train_from_init

    model, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    _attach_fake_win_head(model)
    # 🚨 The init snapshot MUST be taken BEFORE `learn()`, and that is not a style choice: after a
    # train() the optimizer's Adam state is populated, `deepcopy`ing it hands `load_state_dict`
    # tensors it aliases rather than copies, and the next train() mutates the very snapshot it was
    # restored from. Two identical calls then drift by ~1e-3 and every byte-identity claim built on
    # them is vacuous. Measured here before this test was believed; the `signal/` byte-identity
    # test takes its snapshot before `learn()` for the same reason.
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)

    live = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    real = ppo_mod.live_gauge_metrics
    try:
        ppo_mod.live_gauge_metrics = lambda _v, _z: {}      # the diagnostic removed entirely
        muted = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=1)
    finally:
        ppo_mod.live_gauge_metrics = real

    for k in live:
        assert th.allclose(live[k], muted[k], atol=0.0), f"the scaffolding read perturbed {k}"


def test_the_gauge_is_read_from_EPOCH_ZERO_only():
    """Mixing epochs would attribute a pair to a policy that did not produce it. With n_epochs=3
    the published `scaffolding_n` must still be one pass over the rollout."""
    from agents.training.instrumented_ppo_test import _build_tiny_ppo

    model, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    model.n_epochs = 3
    model.learn(total_timesteps=8 * 4)
    _attach_fake_win_head(model)
    rows = _Logger()
    model.set_logger(rows)
    model.train()
    assert rows.rows["train/scaffolding_n"] == 32.0        # 8 x 4, once — not 96
