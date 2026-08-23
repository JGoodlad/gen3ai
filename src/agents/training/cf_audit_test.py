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
import pytest

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


# ---------------------------------------------------------- the EVIDENTIAL read
#
# The pre-registered success meter for `--cf-evidential` is not the loss — it is whether the
# confessed Beta WIDTH tracks the measured `sd_true_excess` per stratum. Nothing computed that
# correlation until this batch, which meant the experiment had a declared meter with no reader.

def _evid_labels(width_of, *, n_per_decile=16, n_battles=8, rng_seed=0):
    """Synthetic labels whose per-decile MC spread is controlled, plus a width the caller chooses
    as a function of that decile — so a test can construct a head that tracks the blur and one
    that does not, and demand opposite verdicts."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for dec in range(10):
        wp = dec / 10 + 0.05
        # spread GROWS with the decile: the ground truth the width is supposed to discover
        spread = 0.02 + 0.04 * dec
        for i in range(n_per_decile):
            mc = float(np.clip(rng.normal(wp, spread), 0.0, 1.0))
            rows.append({"battle": f"/t/b{i % n_battles}", "inv": i, "turn": 10 + i,
                         "opponent": "heuristic", "opp_class": "bot",
                         "outcome": "win" if i % 2 else "loss", "win_prob": wp, "value": 0.0,
                         "mc": mc, "wins": int(mc * 8), "n": 8, "opponent_source": "bot",
                         "evid_width": float(width_of(dec)), "evid_precision": 20.0})
    return rows


def _frame_mass(labels):
    from collections import Counter
    c = Counter()
    for r in labels:
        c[(min(9, int(r["win_prob"] * 10)), r["outcome"])] += 1
    return c


def test_spearman_is_none_when_a_side_is_FLAT_not_zero():
    """"Wide everywhere" and "width unrelated to blur" are DIFFERENT findings, and the more damning
    one is the first. A constant input must not be reported as a correlation of 0."""
    from agents.training.cf_audit import spearman
    assert spearman([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) is None
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0]) is None          # mismatched / too short
    assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == 1.0
    assert spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == -1.0
    # RANK, not Pearson: a monotone but wildly nonlinear relation is still a perfect 1.0.
    assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 10.0, 1e3, 1e9]) == 1.0


def test_a_head_whose_width_TRACKS_the_blur_scores_positive():
    labels = _evid_labels(lambda dec: 0.05 + 0.03 * dec)
    from agents.training.cf_audit import evidential_read
    ev = evidential_read(labels, _frame_mass(labels), 8, draws=200)
    assert ev["n_strata"] >= 8
    assert ev["width_vs_blur_spearman"] > 0.7, ev
    lo, hi = ev["width_vs_blur_ci"]
    assert lo is not None and lo > 0.0, "the CI does not exclude 'no relation'"


def test_a_head_that_is_WIDE_EVERYWHERE_scores_the_null_and_says_so():
    """The failure mode the meter exists to catch: a confessed width that is a constant carries no
    information about which states the critic cannot separate."""
    labels = _evid_labels(lambda dec: 0.11)
    from agents.training.cf_audit import evidential_read
    ev = evidential_read(labels, _frame_mass(labels), 8, draws=50)
    assert ev["width_vs_blur_spearman"] is None
    assert ev["width_vs_blur_ci"] == [None, None]
    assert ev["evid_width_mean"] == pytest.approx(0.11)   # …the LEVEL is still reported


def test_a_head_whose_width_ANTI_tracks_the_blur_scores_negative():
    labels = _evid_labels(lambda dec: 0.5 - 0.03 * dec)
    from agents.training.cf_audit import evidential_read
    ev = evidential_read(labels, _frame_mass(labels), 8, draws=200)
    assert ev["width_vs_blur_spearman"] < -0.7, ev


def test_the_bias_map_carries_the_evidential_block_ONLY_when_the_head_was_read():
    """Absent, never zero: 'this checkpoint has no head' and 'this head claims no uncertainty' are
    opposite findings, and a column of zeros renders them identically."""
    from agents.training.cf_audit import bias_map
    labels = _evid_labels(lambda dec: 0.05 + 0.03 * dec)
    frame = [_dec(win_prob=r["win_prob"], outcome=r["outcome"], battle=r["battle"], turn=r["turn"])
             for r in labels]
    design = {"turn_tercile_edges": [12.0, 18.0], "sampler_version": "test", "seed": 0}
    with_head = bias_map(labels, frame, n_rollouts=8, design=design, accounting={})
    assert with_head["evidential"] is not None
    assert any(c.get("evid_width_mean") is not None for c in with_head["resolution"])

    headless = [{k: v for k, v in r.items() if not k.startswith("evid_")} for r in labels]
    without = bias_map(headless, frame, n_rollouts=8, design=design, accounting={})
    assert without["evidential"] is None
    assert all("evid_width_mean" not in c for c in without["resolution"])


def test_the_markdown_says_ABSENT_rather_than_rendering_zeros():
    from agents.training.cf_audit import bias_map, render_markdown
    labels = _evid_labels(lambda dec: 0.05 + 0.03 * dec)
    frame = [_dec(win_prob=r["win_prob"], outcome=r["outcome"], battle=r["battle"], turn=r["turn"])
             for r in labels]
    design = {"turn_tercile_edges": [12.0, 18.0], "sampler_version": "test", "seed": 0}
    headless = [{k: v for k, v in r.items() if not k.startswith("evid_")} for r in labels]
    md = render_markdown(bias_map(headless, frame, n_rollouts=8, design=design, accounting={}),
                         run_dir="/t", step=1, ckpt=None)
    assert "carries no `cf_evid_head`" in md
    assert "0.000" not in md.split("## EVIDENTIAL")[1].split("## Caveats")[0]

    md2 = render_markdown(bias_map(labels, frame, n_rollouts=8, design=design, accounting={}),
                          run_dir="/t", step=1, ckpt=None)
    assert "width_vs_blur_spearman" in md2 and "Beta width" in md2


def test_attach_evidential_is_a_NO_OP_when_the_checkpoint_has_no_head(capsys):
    """A model that will not load, or one without the head, must cost the audit its evidential
    columns and NOTHING ELSE — the labels and the bias map are the product."""
    from agents.training.cf_audit import attach_evidential

    class _NoHead:
        def cf_evidential_batch(self, obs, masks=None):
            return None

    class _Session:
        def probe_model(self, battle_id):
            return _NoHead(), None

    labels = [{"battle": "/t/b1", "inv": 0}]
    cache = {"/t/b1_states.npz": np.zeros((3, 4), dtype=np.float32)}
    assert attach_evidential(_Session(), labels, cache) == 0
    assert "evid_width" not in labels[0]
    assert "no cf_evid_head" in capsys.readouterr().out

    class _Broken:
        def probe_model(self, battle_id):
            raise FileNotFoundError("no checkpoint at this architecture")

    assert attach_evidential(_Broken(), labels, cache) == 0
    assert "no model" in capsys.readouterr().out
    # a session with no such method at all (an injected test double) is silently fine
    assert attach_evidential(object(), labels, cache) == 0


def test_attach_evidential_writes_the_width_and_precision_it_was_given():
    from agents.training.cf_audit import attach_evidential

    class _Head:
        def cf_evidential_batch(self, obs, masks=None):
            n = len(obs)
            return np.full(n, 2.0), np.full(n, 6.0)      # Beta(2, 6)

    class _Session:
        def probe_model(self, battle_id):
            return _Head(), None

    labels = [{"battle": "/t/b1", "inv": i} for i in range(3)]
    cache = {"/t/b1_states.npz": np.zeros((3, 4), dtype=np.float32)}
    assert attach_evidential(_Session(), labels, cache) == 3
    # std of Beta(2,6) = sqrt(2*6 / (8^2 * 9)) = sqrt(12/576)
    assert labels[0]["evid_width"] == float(np.sqrt(12 / 576))
    assert labels[0]["evid_precision"] == 8.0


# --------------------------------------------------------- TWIN HEADS + SHADOW (v99)
# gen3_cf_twin_heads_v1. The amended R1 primary is a WITHIN-RUN paired difference, so the tests
# here are about the two properties that make a paired read trustworthy: the SIGN convention (these
# are ERROR scores, so lower is better and a reader gets this backwards), and the refusal to report
# a comparison it cannot make.

def _twin_labels(n=60, b_err=0.30, c_err=0.05, a_err=0.30, shadow=None, live=None):
    """Rows where head C is deliberately CLOSER to the MC truth than A and B.

    A and B are given the SAME error on purpose: the coverage arm and the control should look alike
    unless the extra states bought something, so a test fixture that made them differ would let a
    broken contrast pass on the fixture's own asymmetry.
    """
    rows = []
    for i in range(n):
        mc = 0.1 + 0.8 * ((i * 7) % 10) / 10.0
        sign = 1.0 if i % 2 else -1.0
        r = {"battle": f"/t/b{i % 6}", "inv": i, "turn": 10 + i % 20, "opponent": "heuristic",
             "opp_class": "bot", "outcome": "loss" if i % 3 else "win", "mc": mc, "n": 8,
             "value": 1.0,
             "win_prob": float(np.clip(mc + sign * a_err, 0.0, 1.0)),
             "twin_b_pred": float(np.clip(mc + sign * b_err, 0.0, 1.0)),
             "twin_c_pred": float(np.clip(mc + sign * c_err, 0.0, 1.0))}
        if shadow is not None:
            r["shadow_value"] = shadow(i)
        if live is not None:
            r["live_v"] = live(i)
        rows.append(r)
    return rows


def test_the_paired_read_scores_the_better_labelled_head_as_BETTER():
    """SIGN CONVENTION, pinned: these are ERROR scores, so a NEGATIVE difference means the
    first-named head is better. Getting this backwards is the single easiest way to read the arm's
    result as its own opposite."""
    from agents.training.cf_audit import paired_head_read
    out = paired_head_read(_twin_labels(), draws=200)
    assert out["heads_present"] == ["A", "B", "C"]
    c_minus_b = next(c for c in out["contrasts"] if c["contrast"] == "C_minus_B")
    assert c_minus_b["brier"] < 0, "the tighter-labelled head scored as WORSE — sign flipped?"
    lo, hi = c_minus_b["brier_ci"]
    assert lo is not None and hi < 0, "the CI does not exclude zero on a fixture built to"
    assert "PRECISION" in c_minus_b["isolates"]


def test_the_coverage_contrast_reads_NULL_when_B_and_A_are_alike():
    """B−A must be ~0 when the coverage arm bought nothing. A fixture in which A and B have the
    SAME error is the honest null, and a contrast that reported an effect here would be reporting
    its own arithmetic."""
    from agents.training.cf_audit import paired_head_read
    out = paired_head_read(_twin_labels(a_err=0.3, b_err=0.3), draws=200)
    b_minus_a = next(c for c in out["contrasts"] if c["contrast"] == "B_minus_A")
    assert abs(b_minus_a["brier"]) < 1e-9
    assert b_minus_a["mean_abs_pred_diff"] < 1e-9


def test_a_near_zero_contrast_with_no_prediction_divergence_is_flagged_by_the_pred_diff():
    """The reading that a flat contrast is AMBIGUOUS. `mean_abs_pred_diff` is the field that tells
    "the labels did not separate the heads" (a coverage/dosage fact) apart from "they separated and
    it bought nothing" (the pre-registered kill)."""
    from agents.training.cf_audit import paired_head_read
    same = paired_head_read(_twin_labels(b_err=0.3, c_err=0.3), draws=100)
    moved = paired_head_read(_twin_labels(b_err=0.3, c_err=0.05), draws=100)
    cb_same = next(c for c in same["contrasts"] if c["contrast"] == "C_minus_B")
    cb_moved = next(c for c in moved["contrasts"] if c["contrast"] == "C_minus_B")
    assert cb_same["mean_abs_pred_diff"] < 1e-9
    assert cb_moved["mean_abs_pred_diff"] > 0.1


def test_the_paired_read_REFUSES_a_comparison_it_cannot_make():
    """A checkpoint without the twin heads must produce no contrasts at all — not zeros. "This run
    has no twin heads" and "the twins agree exactly" are opposite findings."""
    from agents.training.cf_audit import paired_head_read
    bare = [{k: v for k, v in r.items() if not k.startswith("twin_")}
            for r in _twin_labels(n=20)]
    out = paired_head_read(bare, draws=50)
    assert out["contrasts"] == [] and out["heads_present"] == ["A"]


def test_the_bias_map_carries_the_twin_blocks_ONLY_when_the_heads_were_read():
    from agents.training.cf_audit import bias_map
    labels = _twin_labels()
    frame = [_dec(win_prob=r["win_prob"], outcome=r["outcome"], battle=r["battle"], turn=r["turn"])
             for r in labels]
    design = {"turn_tercile_edges": [12.0, 18.0], "sampler_version": "test", "seed": 0}
    with_heads = bias_map(labels, frame, n_rollouts=8, design=design, accounting={})
    assert with_heads["twin_paired"]["contrasts"]
    assert set(with_heads["twin_resolution"]["by_head"]) >= {"A", "C"}
    # The weighting caveat must travel WITH the numbers, not live only in a docstring: absolute
    # levels here are not comparable with the bias map's population-weighted headline.
    assert "UNWEIGHTED" in with_heads["twin_resolution"]["weighting"]

    bare = [{k: v for k, v in r.items() if not k.startswith("twin_")} for r in labels]
    without = bias_map(bare, frame, n_rollouts=8, design=design, accounting={})
    assert "twin_paired" not in without and "twin_resolution" not in without


def test_the_shadow_block_reports_a_SIGNED_divergence_against_the_live_critic():
    """THE staged-promotion meter. A shadow sitting systematically BELOW the live critic says the
    live critic is optimistic about the states the factory samples — so the SIGN is the reading and
    an absolute value would destroy it."""
    from agents.training.cf_audit import shadow_read
    out = shadow_read(_twin_labels(shadow=lambda i: 1.0, live=lambda i: 3.0), draws=200)
    assert out["shadow_vs_live_v"] == pytest.approx(-2.0)
    assert out["shadow_vs_live_v_abs"] == pytest.approx(2.0)
    lo, hi = out["shadow_vs_live_v_ci"]
    assert lo is not None and hi < 0


def test_the_shadow_block_is_ABSENT_without_a_shadow_head():
    from agents.training.cf_audit import bias_map, shadow_read
    assert shadow_read(_twin_labels()) is None
    labels = _twin_labels()
    frame = [_dec(win_prob=r["win_prob"], outcome=r["outcome"], battle=r["battle"], turn=r["turn"])
             for r in labels]
    out = bias_map(labels, frame, n_rollouts=8,
                   design={"turn_tercile_edges": [12.0, 18.0], "sampler_version": "t", "seed": 0},
                   accounting={})
    assert "shadow" not in out


def test_attach_twin_heads_is_a_NO_OP_when_the_checkpoint_has_no_heads(capsys):
    """Same best-effort contract as `attach_evidential`: a model that will not load (79 of 79
    archived runs, at any given HEAD) costs the audit these columns and NOTHING ELSE."""
    from agents.training.cf_audit import attach_twin_heads

    class _NoModel:
        def probe_model(self, _p):
            raise RuntimeError("ArchDriftError")

    labels = _twin_labels(n=4)
    assert attach_twin_heads(_NoModel(), labels, {}) == 0
    assert "columns omitted" in capsys.readouterr().out
    assert attach_twin_heads(object(), labels, {}) == 0      # no probe_model at all
    assert attach_twin_heads(_NoModel(), [], {}) == 0        # no labels
