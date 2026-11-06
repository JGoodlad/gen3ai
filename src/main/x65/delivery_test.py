"""B4's delivery meter (PREREG §5): Δagree, the deal-clustered bootstrap, DELIVERED iff LB > 0 with the boundary
rule, and the gate file the read rule consumes."""
from __future__ import annotations

import json
from pathlib import Path

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
                 leak_attr=rng.random(n) < 0.3, leak_rewr=rng.random(n) < 0.1,
                 v_student=rng.random(n).astype(np.float32), v_pstar=rng.random(n).astype(np.float32),
                 pre_agree_pstar=np.ones(40, np.int8))
    doc = DV.gate(tmp_path, "b" * 64)
    assert doc["schema"] == "gen3_x65_delivery_v1" and doc["B"] == 4000 and doc["seed"] == 20261105
    assert doc["students"]["forum72"]["zip_sha256"] == "1" * 64 and doc["delivered"] is True
    assert set(doc["per_panel"]["p72"]) == {"n_rows", "n_deals", "agree_student", "agree_pstar", "delta"}
    assert doc["per_panel"]["p72"]["n_deals"] == 60
    assert set(doc["reported"]["p72"]["by_exposure"]) == {"attribution_miss", "rewritten_moves", "neither"}
    # the read rule accepts exactly this file (its key set, its sha256s, its own LB test)
    from main.reads.rules_distill_play import read_delivery

    (tmp_path / "delivery.json").write_text(json.dumps(doc))
    got = read_delivery({"path": str(tmp_path / "delivery.json"), "B": 4000, "seed": 20261105},
                        students={p: f"{i}" * 64 for i, p in enumerate(BK.U_PANELS)}, u_panels=list(BK.U_PANELS),
                        bank_manifest="b" * 64, eps=DV.RULE8_EPS)
    assert got["problems"] == [] and got["delivered"] is True
    # the independent recompute from the same score files agrees (REVIEW R-M1)
    bh = _by_hand()
    assert bh.compare(bh.recompute(str(tmp_path)), doc) == []
    with pytest.raises(DV.DeliveryError):
        DV.gate(tmp_path, "c" * 64)


def _by_hand():
    import importlib.util

    from utils.paths import repo_path

    path = repo_path("designs", "research_state", "measurements", "x65_exit0_2026-11-05", "delivery_by_hand.py")
    spec = importlib.util.spec_from_file_location("x65_delivery_by_hand", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_by_hand_recompute_catches_a_wrong_gate(tmp_path):
    """REVIEW R-M1: a gate file whose Δagree, pooled Δ, LB or DELIVERED is not what the score files say is named."""
    rng = np.random.default_rng(4)
    for i, p in enumerate(BK.U_PANELS):
        deal = np.repeat(np.array([f"{p[:2]}{j:018x}" for j in range(40)]), 8)
        a_p = (rng.random(320) < 0.55).astype(np.int8)
        a_s = (rng.random(320) < 0.6).astype(np.int8)
        np.savez(tmp_path / f"{p}.npz", deal=deal, a_student=a_s, a_pstar=a_p)
    bh = _by_hand()
    mine = bh.recompute(str(tmp_path))
    gate = {"per_panel": {p: {"delta": mine["delta"][p]} for p in BK.U_PANELS}, "pooled": mine["pooled"],
            "lb": mine["lb"], "delivered": mine["delivered"], "boundary": mine["boundary"]}
    assert bh.compare(mine, gate) == []
    for k, v in (("pooled", mine["pooled"] + 1e-6), ("lb", mine["lb"] - 1e-3), ("delivered", not mine["delivered"])):
        assert bh.compare(mine, dict(gate, **{k: v})), k
    wrong = json.loads(json.dumps(gate))
    wrong["per_panel"]["forum72"]["delta"] += 0.01
    assert bh.compare(mine, wrong)


# ------------------------------------------------------------------------------------------------ score(), stubbed
def _score_bank(tmp_path):
    """One read panel (forum72), the searched arm: 4 deals x 4 rows, label rows past turn 10, two pre-turn-10 rows,
    one past-turn-10 non-label row (an exclusion), plus a raw-arm shard score() must not read. The obs row encodes
    what the stub forward returns: column 0 = the student's argmax, 1 = its V, 2 = P*'s V."""
    from main.x65.student_test import N_ACT

    rows = []
    #        deal, label, turn, played, pi_argmax, student, attr, rewr
    spec = [(0, 1, 12, 2, 2, 2, 0, 0), (0, 1, 13, 3, 1, 3, 1, 0), (0, 0, 4, 1, 1, 2, 0, 0), (0, 1, 14, 0, 0, 1, 0, 1),
            (1, 1, 12, 1, 1, 1, 0, 0), (1, 0, 11, 2, 2, 2, 0, 0), (1, 1, 15, 3, 2, 2, 2, 0), (1, 0, 6, 0, 0, 0, 0, 0),
            (2, 1, 12, 0, 3, 0, 0, 0), (2, 1, 12, 1, 1, 0, 0, 0), (3, 1, 20, 2, 0, 2, 0, 3), (3, 1, 21, 3, 3, 3, 0, 0)]
    for k, (d, lab, turn, pl, pia, stu, attr, rewr) in enumerate(spec):
        rows.append({"deal": f"{d:020x}", "label": bool(lab), "turn": turn, "played": pl, "pi_argmax": pia,
                     "stu": stu, "attr": attr, "rewr": rewr, "v_s": 0.1 * k, "v_p": 0.05 * k})
    n = len(rows)
    obs = np.zeros((n, 8), np.float32)
    obs[:, 0] = [r["stu"] for r in rows]
    obs[:, 1] = [r["v_s"] for r in rows]
    obs[:, 2] = [r["v_p"] for r in rows]
    mask = np.zeros((n, N_ACT), np.int8)
    mask[:, :4] = 1
    arr = {"obs": obs, "mask": mask, "label": np.array([r["label"] for r in rows]),
           "played": np.array([r["played"] for r in rows], np.int8),
           "pi_argmax": np.array([r["pi_argmax"] for r in rows], np.int8),
           "deal": np.array([r["deal"] for r in rows], "U20"), "turn": np.array([r["turn"] for r in rows], np.int16),
           "arm": np.ones(n, np.int8), "leak_attr": np.array([r["attr"] for r in rows], np.int16),
           "leak_rewr": np.array([r["rewr"] for r in rows], np.int16)}
    shards = []
    for name, a in (("pstar_s__forum72__300", arr), ("pstar__forum72__300", dict(arr, arm=np.zeros(n, np.int8)))):
        np.savez(tmp_path / f"{name}.npz", **a)
        shards.append({"name": name, "label": name.split("__")[0], "panel": "forum72", "unit_seed": 300, "rows": n,
                       "labels": int(a["label"].sum()), "sha256": BK.sha256_file(tmp_path / f"{name}.npz")})
    man = {"schema": BK.MANIFEST_SCHEMA, "void": None, "engine": BK.E4, "dirty": False, "shards": shards,
           "rows": 2 * n, "label_rows": 2 * int(arr["label"].sum())}
    (tmp_path / "bank_manifest.json").write_text(json.dumps(man))
    return rows


def test_score_reads_every_label_row_of_the_read_panel_and_p_star_s_recorded_choice(tmp_path, monkeypatch):
    """REVIEW R-M1 (mutation M15): ``score()`` itself, its model forward stubbed: per label row a(student) =
    1[student argmax = played], a(P*) = 1[pi_argmax = played], the deal, the departures, the leak counters and both
    V; the pre-turn-10 agreement with P*'s recorded choice; nothing from the raw arm or a non-label row."""
    from main.x65 import student as ST

    bank = tmp_path / "bank"
    bank.mkdir()
    rows = _score_bank(bank)
    zp, init = tmp_path / "student.zip", tmp_path / "pstar.zip"
    zp.write_bytes(b"student")
    init.write_bytes(b"pstar")
    (tmp_path / "export.json").write_text(json.dumps({"zip": str(zp), "zip_sha256": BK.sha256_file(zp),
                                                      "bit_identical": True, "student": "S_F"}))
    monkeypatch.setattr(ST, "PSTAR_SHA256", BK.sha256_file(init))

    def fake_forward(path, obs, mask, threads=2):
        out = np.full((len(obs), BK.N_ACT + 1), -5.0, np.float32)
        if Path(path) == zp:
            out[np.arange(len(obs)), obs[:, 0].astype(int)] = 3.0
            out[:, BK.N_ACT] = obs[:, 1]
        else:
            out[np.arange(len(obs)), (obs[:, 0].astype(int) + 1) % 4] = 3.0    # P*'s forward: never the student's
            out[:, BK.N_ACT] = obs[:, 2]
        return out

    monkeypatch.setattr(DV, "_forward", fake_forward)
    got = DV.score(bank, BK.sha256_file(bank / "bank_manifest.json"), "forum72", tmp_path / "export.json", init,
                   tmp_path / "scores" / "forum72.npz")
    z = np.load(tmp_path / "scores" / "forum72.npz")
    lab = [r for r in rows if r["label"]]
    pre = [r for r in rows if r["turn"] < 10]
    assert got == {"fold": "forum72", "rows": len(lab), "pre_rows": len(pre)}
    assert z["deal"].tolist() == [r["deal"] for r in lab]
    assert z["a_student"].tolist() == [int(r["stu"] == r["played"]) for r in lab]
    assert z["a_pstar"].tolist() == [int(r["pi_argmax"] == r["played"]) for r in lab]
    assert z["departure"].tolist() == [r["pi_argmax"] != r["played"] for r in lab]
    assert z["leak_attr"].tolist() == [r["attr"] >= 1 for r in lab]
    assert z["leak_rewr"].tolist() == [r["rewr"] >= 1 for r in lab]
    assert np.allclose(z["v_student"], [r["v_s"] for r in lab]) and np.allclose(z["v_pstar"], [r["v_p"] for r in lab])
    assert z["pre_agree_pstar"].tolist() == [int(r["stu"] == r["pi_argmax"]) for r in pre]
    assert str(z["zip_sha256"]) == BK.sha256_file(zp) and str(z["fold"]) == "forum72"
    # the hand-made rows are not degenerate: the two agreements differ on some rows
    assert z["a_student"].tolist() != z["a_pstar"].tolist()
