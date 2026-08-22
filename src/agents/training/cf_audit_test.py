"""Pure unit tests for cf_audit's statistics and bookkeeping — no battles, no bridge, no model.

The estimator tests are the load-bearing ones. ``sd_true_excess`` is the program's PRIMARY
meter, and an estimator that reports structure where there is none would license a lever on
noise; so it is validated **at zero true effect** (a synthetic cell whose entire spread IS the
binomial floor must return ~0) as well as at a known nonzero one.
"""

from __future__ import annotations

import json
import os

import numpy as np

from agents.training.cf_audit import (CONVICTION_DECILE, Decision, LabelRow, SCHEMA_VERSION,
                                      cluster_bootstrap_ci, obs_digest, sd_true_excess,
                                      stratified_sample, stratum_of, turn_tercile_edges,
                                      wilson_ci, write_labels)


def _dec(**kw):
    base = dict(battle="/t/b1", short="heuristic/b1", opponent="heuristic", opp_class="bot",
                outcome="loss", inv=3, turn=10, win_prob=0.5, value=1.0, action=6,
                move_rank=1, n_moves=8)
    base.update(kw)
    return Decision(**base)


# --------------------------------------------------------------------- Wilson

def test_wilson_ci_is_not_degenerate_at_zero_wins():
    lo, hi = wilson_ci(0, 8)
    assert lo == 0.0 and 0.0 < hi < 1.0, "a normal approximation would give the useless [0, 0]"
    lo, hi = wilson_ci(8, 8)
    assert hi == 1.0 and 0.0 < lo < 1.0


def test_wilson_ci_brackets_the_point_estimate_and_narrows_with_n():
    lo, hi = wilson_ci(4, 8)
    assert lo < 0.5 < hi
    lo2, hi2 = wilson_ci(400, 800)
    assert (hi2 - lo2) < (hi - lo)


def test_wilson_ci_of_no_samples_is_maximally_uninformative():
    assert wilson_ci(0, 0) == (0.0, 1.0)


# ------------------------------------------------------- sd_true_excess (THE meter)

def test_sd_true_excess_returns_zero_when_the_spread_IS_the_binomial_noise():
    """ZERO TRUE EFFECT. Every state in the cell shares one true p, so all the observed
    spread of the R-rollout means is sampling noise and there is nothing to resolve. An
    estimator that reports excess here would license a lever on noise."""
    rng = np.random.default_rng(0)
    R = 8
    excesses = []
    for p in (0.2, 0.5, 0.85):
        for _ in range(40):
            mc = rng.binomial(R, p, size=300) / R
            excesses.append(sd_true_excess(mc, R)["sd_true_excess"])
    # The estimator is unbiased in VARIANCE and clamped at 0, so the sd reads slightly
    # positive on average — and it carries the sampling noise of a variance estimate
    # (relative sd ~sqrt(2/n)), which at n=300 puts an occasional cell near 0.1. The
    # property that matters is therefore a DISTRIBUTIONAL one, stated against the real
    # spreads the G0 map reports (0.11-0.36): spurious excess is small on average and its
    # worst cell never reaches the smallest genuine signal.
    assert np.mean(excesses) < 0.03, f"mean spurious sd_true_excess {np.mean(excesses):.4f}"
    assert np.percentile(excesses, 95) < 0.08, f"p95 {np.percentile(excesses, 95):.4f}"
    assert max(excesses) < 0.11, f"max {max(excesses):.4f}"


def test_sd_true_excess_recovers_a_KNOWN_spread():
    """Nonzero true effect: true p is spread with sd 0.20, and the estimator must find it
    after subtracting the R=8 floor (which alone is ~0.17 and would otherwise swamp it)."""
    rng = np.random.default_rng(1)
    R = 8
    true_p = np.clip(rng.normal(0.5, 0.20, size=4000), 0.01, 0.99)
    mc = rng.binomial(R, true_p) / R
    st = sd_true_excess(mc, R)
    assert st["sd_observed"] > st["sd_true_excess"] > 0.17, st
    assert abs(st["sd_true_excess"] - 0.20) < 0.03, st
    assert 0.4 < st["frac_variance_real"] < 0.75, st


def test_sd_true_excess_is_degenerate_on_a_tiny_cell():
    st = sd_true_excess([0.5, 0.25], 8)
    assert st["sd_true_excess"] is None and st["n"] == 2


def test_sd_true_excess_weights_recombine_subcells_at_population_shares():
    """A cell sampled 50/50 from two sub-populations that occur 90/10 must be recombined at
    the POPULATION shares — otherwise this probe's own oversampling inflates the spread."""
    a = [0.9] * 50                      # the common sub-population, tight
    b = [0.1] * 50                      # the rare one, far away
    unweighted = sd_true_excess(a + b, 8)["sd_true_excess"]
    weighted = sd_true_excess(a + b, 8, weights=[0.9 / 50] * 50 + [0.1 / 50] * 50)["sd_true_excess"]
    assert weighted < unweighted, (weighted, unweighted)


# --------------------------------------------------------------- clustered CI

def test_cluster_bootstrap_ci_is_wider_than_a_state_level_ci_when_battles_correlate():
    """Decisions inside a battle are not independent. Resampling STATES would understate the
    width; resampling BATTLES is the honest unit (the pooled-correlation Simpson lesson)."""
    rng = np.random.default_rng(3)
    values, clusters = [], []
    for b in range(12):                                  # a strong per-battle offset
        off = rng.normal(0, 1.0)
        for _ in range(20):
            values.append(off + rng.normal(0, 0.05))
            clusters.append(f"b{b}")
    lo_c, hi_c = cluster_bootstrap_ci(values, clusters, draws=800, seed=1)
    lo_s, hi_s = cluster_bootstrap_ci(values, [f"s{i}" for i in range(len(values))],
                                      draws=800, seed=1)
    assert (hi_c - lo_c) > 3 * (hi_s - lo_s)


def test_cluster_bootstrap_ci_refuses_a_single_cluster():
    assert cluster_bootstrap_ci([1.0, 2.0], ["b", "b"]) == (None, None)


# ----------------------------------------------------------------- stratifier

def test_stratum_is_confidence_decile_by_outcome_by_turn_tercile():
    edges = (10.0, 23.0)
    assert stratum_of(_dec(win_prob=0.0, turn=5), edges) == (0, "loss", 0)
    assert stratum_of(_dec(win_prob=0.99, turn=30, outcome="win"), edges) == (9, "win", 2)
    assert stratum_of(_dec(win_prob=1.0, turn=15), edges) == (9, "loss", 1), "1.0 must not overflow"


def test_turn_terciles_are_read_off_the_data():
    frame = [_dec(turn=t) for t in range(1, 31)]
    lo, hi = turn_tercile_edges(frame)
    assert 9 <= lo <= 12 and 19 <= hi <= 22


def test_stratifier_oversamples_the_high_confidence_loss_region():
    """The conviction class is the population R1 supervises, and it is RARE — so it is
    deliberately over-sampled relative to its frame share (and corrected for at aggregation)."""
    frame = ([_dec(battle=f"/b{i}", win_prob=0.9, outcome="loss", inv=i) for i in range(200)]
             + [_dec(battle=f"/c{i}", win_prob=0.2, outcome="win", inv=i) for i in range(200)])
    sample, design = stratified_sample(frame, 200, seed=5, max_per_battle=99)
    conv_frame = 0.5
    conv_sample = np.mean([d.win_prob >= 0.75 and d.outcome == "loss" for d in sample])
    assert conv_sample > conv_frame * 1.4, conv_sample
    assert design["conviction_decile_floor"] == CONVICTION_DECILE
    assert design["sampler_version"] and design["seed"] == 5


def test_stratifier_caps_per_battle_so_one_long_game_cannot_carry_a_stratum():
    frame = [_dec(battle="/only", inv=i, win_prob=0.5) for i in range(400)]
    sample, _ = stratified_sample(frame, 200, seed=5, max_per_battle=12)
    assert len(sample) == 12


def test_stratifier_is_deterministic_under_a_seed():
    frame = [_dec(battle=f"/b{i % 20}", inv=i, win_prob=(i % 10) / 10) for i in range(300)]
    a, _ = stratified_sample(frame, 50, seed=7)
    b, _ = stratified_sample(frame, 50, seed=7)
    c, _ = stratified_sample(frame, 50, seed=8)
    assert [d.inv for d in a] == [d.inv for d in b]
    assert [d.inv for d in a] != [d.inv for d in c]


def test_stratifier_design_records_every_populated_cell():
    frame = [_dec(battle=f"/b{i}", inv=i, win_prob=(i % 10) / 10,
                  outcome="win" if i % 2 else "loss", turn=1 + i % 40) for i in range(400)]
    _, design = stratified_sample(frame, 100, seed=1)
    assert design["cells"] and sum(c["frame_n"] for c in design["cells"]) == len(frame)


# --------------------------------------------------------------- schema writer

def test_label_row_serializes_the_v1_schema_exactly(tmp_path):
    obs = np.arange(8, dtype=np.float32)
    row = LabelRow(battle="/t/b_reconstruction.json", decision_idx=4, obs_sha1=obs_digest(obs),
                   label=0.625, n_rollouts=8, wilson_lo=0.3, wilson_hi=0.86,
                   policy_step=24000000, opponent="heuristic", obs_npz="/t/b_states.npz::obs")
    path = write_labels([row], str(tmp_path), producer="cf_audit", seq=24000000)
    assert os.path.basename(path) == "labels_cf_audit_24000000.jsonl"
    assert os.path.basename(os.path.dirname(path)) == "cf_labels"
    with open(path) as f:
        d = json.loads(f.readline())
    assert set(d) == {"schema", "kind", "battle", "decision_idx", "obs_sha1", "obs_npz",
                      "obs_inline", "label", "n_rollouts", "wilson_lo", "wilson_hi",
                      "policy_step", "opponent", "created_unix"}
    assert d["schema"] == SCHEMA_VERSION and d["kind"] == "mc_winprob"
    assert d["obs_inline"] is None and d["obs_npz"].endswith("::obs")
    assert 0.0 <= d["label"] <= 1.0 and d["wilson_lo"] <= d["label"] <= d["wilson_hi"]


def test_obs_digest_is_over_the_float32_BYTES_so_a_consumer_can_verify_its_row():
    a = np.arange(8, dtype=np.float32)
    assert obs_digest(a) == obs_digest(a.astype(np.float64))     # cast is normalized
    b = a.copy()
    b[3] += np.float32(1e-6)
    assert obs_digest(a) != obs_digest(b), "a one-ulp change must change the digest"


def test_obs_digest_matches_a_hand_computed_sha1():
    import hashlib
    a = np.array([1.0, 2.0], dtype=np.float32)
    assert obs_digest(a) == hashlib.sha1(a.tobytes()).hexdigest()
