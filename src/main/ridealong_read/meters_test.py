"""Pure unit tests of the ride-along reader's meters (synthetic arrays; no model, no bank)."""

from __future__ import annotations

import numpy as np
import pytest

from main.policy_spectrum.qhat_report import Boot
from main.ridealong_read import meters as M


def test_outcome_target_draw_policies():
    z, keep = M.outcome_target(["win", "loss", "draw"])
    assert z.tolist() == [1.0, 0.0, 0.0] and keep.tolist() == [True, True, False]
    z, keep = M.outcome_target(["win", "loss", "draw"], "loss")
    assert keep.all() and z.tolist() == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        M.outcome_target(["win", "tie"])
    with pytest.raises(ValueError):
        M.outcome_target(["win"], "half")


def test_binary_entropy_peaks_at_half():
    h = M.binary_entropy(np.array([0.0, 0.1, 0.5, 0.9, 1.0]))
    assert int(np.argmax(h)) == 2 and h[1] == pytest.approx(h[3]) and h[0] < 1e-9


def test_cluster_boot_auroc_perfect_and_degenerate():
    score = np.arange(40, dtype=float)
    lab = score >= 20
    cl = np.repeat(np.arange(10), 4)
    assert M.cluster_boot_auroc(score, lab, cl) == [1.0, 1.0]
    assert M.cluster_boot_auroc(score, np.zeros(40, bool), cl) is None
    assert M.cluster_boot_auroc(score, lab, np.zeros(40)) is None       # one cluster


def test_score_vs_error_perfect_score():
    rng = np.random.default_rng(0)
    err = rng.uniform(0, 1, 500)
    out = M.score_vs_error(err, err, np.arange(500) // 5)
    assert out["auroc"] == 1.0 and out["spearman"] == pytest.approx(1.0)
    dec = out["err_by_decile"]
    assert len(dec) == 10 and all(a < b for a, b in zip(dec, dec[1:]))
    assert out["n_err"] == int((err > 0.5).sum())


def test_conditional_auroc_separates_a_ref_proxy_from_new_information():
    rng = np.random.default_rng(1)
    ref = rng.uniform(0, 1, 2000)
    extra = rng.uniform(0, 1, 2000)
    lab = (ref + extra) > 1.0
    # A score that is only a coarse function of ref (constant inside each ref quintile) carries
    # nothing beyond ref: within-quintile AUROC is exactly 0.5 even though its raw AUROC is high.
    proxy = np.digitize(ref, np.quantile(ref, [.2, .4, .6, .8])).astype(float)
    assert M.rank_auroc(proxy, lab) > 0.75
    assert M.conditional_auroc(proxy, lab, ref) == pytest.approx(0.5)
    # A score that IS the missing information keeps a high AUROC inside the bins.
    assert M.conditional_auroc(extra, lab, ref) > 0.8


def test_stratified_uncertainty_shapes():
    rng = np.random.default_rng(2)
    n = 300
    err = rng.uniform(0, 1, n)
    strata = {"phase": np.array(["opening", "midgame", "endgame"] * 100)}
    out = M.stratified_uncertainty({"s": err}, err, np.arange(n) // 3, strata, ref=rng.uniform(0, 1, n))
    assert set(out) == {"all", "phase"}
    assert set(out["phase"]) == {"opening", "midgame", "endgame"}
    assert out["phase"]["opening"]["s"]["n"] == 100
    assert "auroc_within_ref_quintiles" in out["all"]["s"]


def _truth_row(tid: str, outcomes: dict) -> dict:
    return {"id": tid, "ok": True, "outcomes": {str(a): list(v) for a, v in outcomes.items()}}


def _fixture():
    """Two battles, two turns each. Truth values (±1) per legal action; action 1 is always the best
    and action 4 always dominated; action 2 is near-best and STARVED by the policy (π < 1 %)."""
    rows, id_to_row, battle_of = [], {}, {}
    S = 40
    pi = np.zeros((4, 11))
    for r in range(4):
        tid = f"b{r // 2}#p1#{r}"
        rows.append(_truth_row(tid, {1: [1.0] * S, 2: [1.0] * (S - 2) + [-1.0] * 2,
                                     3: [1.0, -1.0] * (S // 2), 4: [-1.0] * S}))
        id_to_row[tid] = r
        battle_of[tid] = f"b{r // 2}"
        pi[r, [1, 2, 3, 4]] = [0.6, 0.005, 0.3, 0.095]
    return rows, id_to_row, battle_of, pi


def test_truth_turns_classes():
    rows, id_to_row, battle_of, _ = _fixture()
    turns = M.truth_turns(rows, id_to_row, battle_of)
    assert len(turns) == 4
    T = turns[0]
    assert T.acts.tolist() == [1, 2, 3, 4]
    assert T.near.tolist() == [True, True, False, False]
    assert T.dom[3] and T.decisive and T.vstar == 1.0
    # a limited read keeps only its own turns
    assert len(M.truth_turns(rows, {rows[0]["id"]: 0}, battle_of)) == 1


def test_within_turn_reads_truth_score_vs_reversed():
    rows, id_to_row, battle_of, pi = _fixture()
    turns = M.truth_turns(rows, id_to_row, battle_of)
    B = Boot([T.battle for T in turns])
    good = np.zeros((4, 11))
    good[:, [1, 2, 3, 4]] = [4.0, 3.0, 2.0, 1.0]
    out = M.within_turn_reads(turns, good, pi, B)
    assert out["spearman"]["mean"] == pytest.approx(1.0)
    assert out["argmax_regret_all"]["mean"] == 0.0 and out["argmax_near"]["mean"] == 1.0
    assert out["starved_top2"]["mean"] == 1.0 and out["starved_top2"]["n"] == 4
    assert out["starved_top2"]["chance"] == pytest.approx(0.5)
    bad = -good
    out = M.within_turn_reads(turns, bad, pi, B)
    assert out["spearman"]["mean"] == pytest.approx(-1.0)
    assert out["argmax_regret_decisive"]["mean"] == pytest.approx(2.0)
    assert out["starved_top2"]["mean"] == 0.0 and out["fed_top2"]["mean"] == 0.0


def test_adv_std_flags_detects_a_spread_on_starved_near_best():
    rows, id_to_row, battle_of, pi = _fixture()
    turns = M.truth_turns(rows, id_to_row, battle_of)
    B = Boot([T.battle for T in turns])
    sd = np.full((4, 11), 0.1)
    sd[:, 2] = 1.0                                   # the starved near-best action
    out = M.adv_std_flags(turns, sd, pi, B)
    assert out["starved_near"]["mean"] == 1.0 and out["fed_near"]["mean"] == 0.1
    assert out["auroc_starved_near_vs_other_legal"]["auroc"] == 1.0
    # nothing is starved-and-not-near in this fixture: the control AUROC has no negatives
    assert out["auroc_starved_near_vs_starved_other"]["auroc"] is None


def test_bank_starved_spread():
    pi = np.array([[0.0, 0.005, 0.995], [0.5, 0.5, 0.0]])
    masks = np.array([[False, True, True], [True, True, False]])
    sd = np.array([[9.0, 2.0, 1.0], [1.0, 1.0, 9.0]])
    out = M.bank_starved_spread(sd, pi, masks)
    assert out["n_starved"] == 1 and out["starved_mean"] == 2.0 and out["fed_mean"] == 1.0
    assert out["auroc_starved_vs_fed"] == 1.0


def test_greedy_truth_value_and_truth_value_error():
    rows, id_to_row, battle_of, pi = _fixture()
    turns = M.truth_turns(rows, id_to_row, battle_of)
    logits = np.zeros((4, 11))
    logits[:, 4] = 5.0                               # the continuation plays the dominated action
    vt = M.greedy_truth_value(turns, logits)
    assert vt.tolist() == [0.0] * 4
    v = np.array([0.9, 0.9, 0.1, 0.1])
    score = np.array([1.0, 1.0, 0.0, 0.0])
    out = M.truth_value_error(turns, v, logits, {"s": score})
    assert out["turns"] == 4 and out["mean_signed_v_minus_truth"] == pytest.approx(0.5)
    assert out["s"]["auroc"] == 1.0
    assert M.truth_value_error([], v, logits, {"s": score}) == {"turns": 0}


def test_score_vs_error_reports_every_interval():
    rng = np.random.default_rng(5)
    err = rng.uniform(0, 1, 400)
    score = err + rng.normal(0, 0.3, 400)
    out = M.score_vs_error(score, err, np.arange(400) // 4)
    for k in ("auroc_ci", "spearman_ci", "decile_ratio_ci"):
        lo, hi = out[k]
        assert lo <= hi
    assert out["auroc_ci"][0] <= out["auroc"] <= out["auroc_ci"][1]
    assert out["decile_ratio_top_over_bottom"] > 1.0


def test_paired_spearman_diff():
    rows, id_to_row, battle_of, pi = _fixture()
    turns = M.truth_turns(rows, id_to_row, battle_of)
    B = Boot([T.battle for T in turns])
    good = np.zeros((4, 11))
    good[:, [1, 2, 3, 4]] = [4.0, 3.0, 2.0, 1.0]
    out = M.paired_spearman_diff(turns, good, -good, B)
    assert out["n"] == 4 and out["delta"] == pytest.approx(2.0)


def test_the_within_quintile_auroc_has_a_clustered_interval_that_brackets_its_point():
    from main.ridealong_read.meters import cluster_boot_conditional_auroc, conditional_auroc

    rng = np.random.default_rng(0)
    n = 2000
    ref = rng.random(n)
    lab = rng.random(n) < 0.3
    score = lab * 0.5 + rng.random(n)            # informative beyond ref
    cl = np.repeat(np.arange(200), 10)
    pt = conditional_auroc(score, lab, ref)
    lo, hi = cluster_boot_conditional_auroc(score, lab, ref, cl, n_boot=200)
    assert lo <= pt <= hi and lo > 0.6
    noise = rng.random(n)                          # carries nothing
    lo2, hi2 = cluster_boot_conditional_auroc(noise, lab, ref, cl, n_boot=200)
    assert lo2 < 0.52 and hi2 > 0.48 and hi2 - lo2 < 0.1   # centred near chance, not above it
    assert cluster_boot_conditional_auroc(score, lab, ref, np.zeros(n)) is None
