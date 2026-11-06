"""B4's delivery meter (PREREG §5): Δagree, the deal-clustered bootstrap, DELIVERED iff LB > 0 with the boundary
rule, and the gate file the read rule consumes."""
from __future__ import annotations

import json

import numpy as np
import pytest

from main.x65 import bank as BK
from main.x65 import delivery as DV


def _panel(delta_rows: int, n_deals: int = 300, rows: int = 30, seed: int = 0):
    """``n_deals`` deals of ``rows`` label rows; P\\* agrees on half the rows, the student on half plus
    ``delta_rows`` extra rows per deal."""
    deal = np.repeat(np.array([f"{i:020x}" for i in range(n_deals)]), rows)
    a_p = np.tile((np.arange(rows) % 2 == 0).astype(int), n_deals)
    a_s = a_p.copy()
    for d in range(n_deals):
        off = np.flatnonzero(a_s[d * rows:(d + 1) * rows] == 0)[: max(delta_rows, 0)]
        a_s[d * rows + off] = 1
    return {"deal": deal, "diff": a_s - a_p}


def test_the_pooled_delta_is_the_mean_of_the_panels_row_means():
    per = {p: _panel(k) for p, k in zip(BK.U_PANELS, (0, 3, 6))}
    g = DV.bootstrap_lb(per, B=200)
    assert g["per_panel"] == {"heldout72_hpfix": 0.0, "forum72": 0.1, "p72": pytest.approx(0.2)}
    assert g["pooled"] == pytest.approx(0.1)
    assert g["lb"] == pytest.approx(0.1) and g["delivered"] and not g["boundary"]   # every deal alike: no spread


def test_no_delta_is_not_delivered_and_the_bootstrap_is_seeded():
    rng = np.random.default_rng(1)
    per = {}
    for p in BK.U_PANELS:
        deal = np.repeat(np.array([f"{i:020x}" for i in range(200)]), 20)
        per[p] = {"deal": deal, "diff": rng.integers(-1, 2, len(deal))}
    a, b = DV.bootstrap_lb(per, B=300), DV.bootstrap_lb(per, B=300)
    assert a == b and a["lb"] < a["pooled"] < a["ub"]
    assert a != DV.bootstrap_lb(per, B=300, seed=7)


def test_a_lower_bound_within_rule8_of_zero_is_never_delivered(monkeypatch):
    per = {p: _panel(1) for p in BK.U_PANELS}
    for lb, want, boundary in ((0.0004, False, True), (-0.0004, False, True), (0.0006, True, False),
                               (-0.01, False, False)):
        monkeypatch.setattr(DV.np, "percentile", lambda a, q, _lb=lb: _lb if q == 2.5 else 1.0)
        g = DV.bootstrap_lb(per, B=5)
        assert (g["delivered"], g["boundary"]) == (want, boundary), lb


def test_the_bootstrap_resamples_whole_deals():
    """One deal of many rows carries all the signal: a row bootstrap would be narrow, the deal bootstrap is wide."""
    per = {}
    for p in BK.U_PANELS:
        deal = np.array(["a" * 20] * 500 + [f"{i:020x}" for i in range(50)])
        per[p] = {"deal": deal, "diff": np.array([1] * 500 + [0] * 50)}
    g = DV.bootstrap_lb(per, B=400)
    assert g["lb"] < 0.5 < g["pooled"]


def test_the_gate_writes_the_rule_s_file_and_refuses_another_bank(tmp_path):
    rng = np.random.default_rng(3)
    for i, p in enumerate(BK.U_PANELS):
        n = 600
        deal = np.repeat(np.array([f"{p[:2]}{j:018x}" for j in range(60)]), 10)
        a_p = (rng.random(n) < 0.6).astype(np.int8)
        a_s = np.maximum(a_p, (rng.random(n) < 0.2)).astype(np.int8)
        np.savez(tmp_path / f"{p}.npz", schema=DV.SCORE_SCHEMA, fold=p, student=f"S{i}", zip_sha256=f"{i}" * 64,
                 bank_manifest_sha256="b" * 64, deal=deal, a_student=a_s, a_pstar=a_p, departure=a_p == 0,
                 leak_attr=rng.random(n) < 0.3, leak_rewr=rng.random(n) < 0.1, leak_resid=rng.random(n) < 0.05,
                 v_student=rng.random(n).astype(np.float32), v_pstar=rng.random(n).astype(np.float32),
                 pre_agree_pstar=np.ones(40, np.int8))
    doc = DV.gate(tmp_path, "b" * 64)
    assert doc["schema"] == "gen3_x65_delivery_v1" and doc["B"] == 4000 and doc["seed"] == 20261105
    assert doc["students"]["forum72"]["zip_sha256"] == "1" * 64 and doc["delivered"] is True
    assert set(doc["per_panel"]["p72"]) == {"n_rows", "n_deals", "agree_student", "agree_pstar", "delta"}
    assert doc["per_panel"]["p72"]["n_deals"] == 60
    assert set(doc["reported"]["p72"]["by_exposure"]) == {"attribution_miss", "rewritten_moves",
                                                          "residual_counters", "none"}
    # the read rule accepts exactly this file (its key set, its sha256s, its own LB test)
    from main.reads.rules_distill_play import read_delivery

    (tmp_path / "delivery.json").write_text(json.dumps(doc))
    got = read_delivery({"path": str(tmp_path / "delivery.json"), "B": 4000, "seed": 20261105},
                        students={p: f"{i}" * 64 for i, p in enumerate(BK.U_PANELS)}, u_panels=list(BK.U_PANELS),
                        bank_manifest="b" * 64, eps=DV.RULE8_EPS)
    assert got["problems"] == [] and got["delivered"] is True
    with pytest.raises(DV.DeliveryError):
        DV.gate(tmp_path, "c" * 64)
