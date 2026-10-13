"""Unit tests of TIER 1's read (pure numpy, synthetic data; no archive read). Run from the repo root:

    GEN3AI_SKIP_DEPS_GUARD=1 python -m pytest designs/research_state/measurements/hidden_info_tier1_2026-10-13/t1_analysis_test.py -q

They pin: the tie rule; the dose family's endpoints and its placebo identity at p = 0; the stratified bootstrap; the
rule-8 comparisons; and the registered R-MOVES verdict logic on planted effects (each verdict reachable, each planted
effect read the right way round).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_blind as SB  # noqa: E402
import t1_data as D  # noqa: E402
import t1_stats as ST  # noqa: E402
import unblind_t1 as U  # noqa: E402


def test_pick_ties_toward_policy_order():
    s = np.array([0.5, 0.7, 0.7 - 1e-7, 0.1])
    assert D.pick(s, [2, 1, 0, 3]) == 2           # within TIE of the max: the policy's earlier action wins
    assert D.pick(s, [1, 2, 0, 3]) == 1
    S = s[[2, 1, 0, 3]]
    Tp = np.array([0.3, 0.6, 0.2, 0.9])
    assert SB.imp_of(S[None, :], Tp)[0] == 0.0     # index 0 = top1 picked


def _synthetic_state(rng, A=5):
    acts = [str(i) for i in range(A)]
    order = list(rng.permutation(A))
    S1 = {"acts": acts, "order": order, "top1": order[0], "turn": 10, "T": rng.random(A),
          "W": rng.integers(0, 2, size=(A, 16)).astype(float),
          "Xd": {d: rng.random((A, 16)) for d in (1, 2, 4)}, "crit": rng.random((A, 8))}
    Sb = {"W": rng.integers(0, 2, size=(A, 8)).astype(float), "Xd": {d: rng.random((A, 8)) for d in (1, 2, 4)},
          "crit": rng.random((A, 8))}
    return S1, Sb


def test_dose_family_endpoints_and_placebo_identity():
    rng = np.random.default_rng(1)
    for _ in range(25):
        S1, Sb = _synthetic_state(rng)
        fam = SB.dk_family(S1, Sb, worst=3)
        Tv, t1 = S1["T"], S1["top1"]
        o = D.scorer_values(S1)
        b = D.scorer_values(Sb)
        for z, grid_last in (("LEAF-K8", 100), ("TRUNC-4-K8", 20), ("CRITIC", 20)):
            assert fam[f"M8|{z}"][0] == pytest.approx(Tv[D.pick(b[z], S1["order"])] - Tv[t1])
            assert fam[f"M8|{z}"][grid_last] == pytest.approx(Tv[D.pick(o[z], S1["order"])] - Tv[t1])
            # p = 0: every placebo IS the honest 4-world mean (each 4-subset weighted equally)
            assert fam[f"M5W|{z}"][0] == pytest.approx(fam[f"M5|{z}"][0])
            assert fam[f"M5X|{z}"][0] == pytest.approx(
                np.mean([SB.imp_of(Sb_sub, Tv[np.array(S1["order"])]) for Sb_sub in
                         [Sb_mat[np.array(S1["order"])] for Sb_mat in
                          [_base_mat(Sb, z)[list(J)].mean(0) for J in SB.OTHER4[3]]]]))


def _base_mat(Sb, z):
    if z == "LEAF-K8":
        return Sb["W"].T
    if z == "TRUNC-4-K8":
        return Sb["Xd"][4].T
    return Sb["crit"].T


def test_stratified_bootstrap_keeps_strata_sizes():
    battles = [f"a{i}" for i in range(7)] + [f"b{i}" for i in range(4)]
    strata = ["end2"] * 7 + ["prod"] * 4
    bt = ST.Boot(battles, strata)
    ia = [i for i, b in enumerate(bt.ub) if b.startswith("a")]
    ib = [i for i, b in enumerate(bt.ub) if b.startswith("b")]
    assert np.all(bt.C[:, ia].sum(1) == 7) and np.all(bt.C[:, ib].sum(1) == 4)
    est, vec = bt.mean(np.arange(11.0))
    assert est == pytest.approx(5.0) and len(vec) == ST.B


def test_rule8_comparisons():
    assert ST.gt(0.0006, 0.0005) and not ST.gt(0.0005, 0.0005)      # the zero edge carries its margin already
    assert not ST.gt(0.0203, 0.02) and ST.gt(0.0206, 0.02)            # a boundary result is not clear
    assert ST.ge(0.5006, 0.5) and not ST.ge(0.5004, 0.5)
    assert ST.lt(0.2494, 0.25) and not ST.lt(0.2496, 0.25)
    assert ST.inside([-0.009, 0.009], 0.01) and not ST.inside([-0.0096, 0.009], 0.01)


# ------------------------------------------------------------------------------------------------
# R-MOVES on planted effects
# ------------------------------------------------------------------------------------------------

LABELS = ["F", "F+M", "O-M", "F+U", "O-U", "D50", "D90", "W90"]


def _tab(sM, nM, sU, nU, d90=None, w90=0.0, n=600, seed=7, noise=0.05):
    """Per state IMP (LEAF-K8): O = F + G_s; the knock-ins / knock-outs move a share of G_s."""
    rng = np.random.default_rng(seed)
    name = {"O": "O", **{lab: f"C{i}" for i, lab in enumerate(LABELS)}}
    tab = {}
    for cp in ("end2", "prod"):
        recs = []
        for i in range(n):
            g = 0.07 + rng.normal(0, 0.15)
            f = rng.normal(0.0, 0.1)
            o = f + g
            vals = {"O": o, "F": f, "F+M": f + sM * g, "O-M": o - nM * g, "F+U": f + sU * g, "O-U": o - nU * g,
                    "D50": f + 0.5 * (sM if d90 is None else d90) * g, "D90": f + (sM if d90 is None else d90) * 0.9 * g,
                    "W90": f + w90 * g}
            src = {}
            for lab, v in vals.items():
                v = v + (0 if lab in ("O", "F") else rng.normal(0, noise))
                src[name[lab]] = {"ok": True, "imp": {"LEAF-K8": float(v)}, "pick": {"LEAF-K8": 0}}
            recs.append({"id": f"{cp}{i}", "cp": cp, "set": "registered", "battle": f"{cp}:{i // 2}", "k": 1 + i % 6,
                         "turn": 2 + i % 38, "material": 2 + i % 11, "opp_class": "heuristic", "src": src, "dk": {}})
        tab[cp] = {"recs": recs}
    return tab, name


def _verdict(**kw):
    tab, name = _tab(**kw)
    fr = U.frames(tab, "registered")
    return U.r_moves(fr, name, gigo_pass=True)


def test_r_moves_binding_pattern_is_pending_replication():
    r = _verdict(sM=0.8, nM=0.8, sU=0.2, nU=0.2)
    assert r["B"]["B3"]["holds"] and r["B"]["B4"]["holds"] and r["B"]["B5"]["holds"]
    assert r["B"]["B6"]["holds"] and r["B"]["B7"]["holds"] and r["B"]["B8"]["holds"]
    assert r["verdict"].startswith("B2-B8 HOLD")                    # B1 needs the independent replication


def test_r_moves_refuted_when_moves_carry_nothing():
    r = _verdict(sM=0.0, nM=0.0, sU=0.9, nU=0.9, w90=0.0)
    assert r["verdict"].startswith("MOVES-REFUTED")
    assert r["R"]["R1"] and r["R"]["R3"]


def test_r_moves_placebo_equivalent_refutes():
    r = _verdict(sM=0.45, nM=0.45, sU=0.4, nU=0.4, d90=0.0, w90=0.0)
    assert r["R"]["R2"]
    assert "R2" in r["verdict"]


def test_r_moves_shared():
    r = _verdict(sM=0.35, nM=0.35, sU=0.3, nU=0.3, w90=0.0)
    assert not (r["R"]["R1"] or r["R"]["R2"] or r["R"]["R3"])
    assert r["verdict"] == "MOVES-SHARED"


def test_r_moves_necessary_not_sufficient():
    r = _verdict(sM=0.0, nM=0.8, sU=0.0, nU=0.6, w90=0.0, d90=0.3)
    assert r["verdict"] == "MOVES-NECESSARY-NOT-SUFFICIENT"


def test_eq_p_and_persistent_first():
    grid = np.array([0.0, 0.5, 1.0])
    assert U.eq_p(np.array([0.0, 0.04, 0.08]), 0.02, grid) == pytest.approx(0.25)
    assert U.eq_p(np.array([0.0, 0.04, 0.08]), -0.1, grid) == 0.0
    assert U.eq_p(np.array([0.0, 0.04, 0.08]), 0.1, grid) is None
    assert U.persistent_first(np.array([True, False, True, True])) == 2
    assert U.persistent_first(np.array([True, True, False])) is None
