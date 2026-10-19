"""The reader on PLANTED effects (no archive): a synthetic table where the oracle - honest gap shrinks and the network's
belief rises between two anchors must read GAP-SHRINKS and BELIEF-RISES; a null series must read neither; the common
set must drop a state that ties or fails at ANY anchor; LEAN never reads a FRONTIER.

    python -I read_mat_test.py          (or under pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import read_mat as RM  # noqa: E402

KB = ("1-2", "3", "4", "5", "6")
TB = ("<=7", "8-14", "15-21", ">=22")
MB = ("<=7", "8-9", "10-11", "12")


def _table(shift_g: float, fn_imp_last: float, n: int = 320, mode: str = "lean", seed: int = 3):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, 101)
    tab = {}
    for lab, step, last in (("a_025M", 25, False), ("a_225M", 225, True)):
        recs = []
        for i in range(n):
            f = rng.normal(0.0, 0.15)
            o = f + 0.07 - (shift_g if last else 0.0) + rng.normal(0, 0.05)
            fn = (fn_imp_last if last else 0.0) + f + rng.normal(0, 0.01)
            curve = f + 0.07 * grid                       # IMP_p rises linearly from F to O's mean
            recs.append({"id": f"s{i:04d}", "battle": f"b{i // 2:04d}", "tie": i == 5 and last,
                         "k_bin": KB[i % 5], "turn_bin": TB[i % 4], "mat_bin": MB[i % 4],
                         "stratum": ("snapshot", "bot")[i % 2],
                         "src": {"O": {"ok": True, "imp": {z: o for z in RM.SCORERS}},
                                 "F": {"ok": i != 7, "imp": {z: f for z in RM.SCORERS}},
                                 "FN": {"ok": True, "imp": {z: fn for z in RM.SCORERS}}},
                         "dk": {"F": {"M8|LEAF-K8": curve}}})
        tab[lab] = {"label": lab, "step": step, "mode": mode, "recs": recs, "counts": {}}
    return tab


def test_planted_shrink_and_belief_rise():
    res = RM.read(_table(shift_g=0.05, fn_imp_last=0.035))
    rm = res["R-MATURITY"]
    assert res["n_common"] == 318 and res["drops"] == {"tie": 1, "truth": 0, "F": 1}
    assert rm["GAP"]["all"]["verdict"] == "GAP-SHRINKS"
    assert rm["BELIEF"]["verdict"] == "BELIEF-RISES"
    assert rm["FRONTIER"]["verdict"].startswith("NOT READ (LEAN")
    assert res["per_anchor"]["a_225M"]["TODAY"]["p_eq"] > 0.3


def test_null_series_reads_nothing():
    rm = RM.read(_table(shift_g=0.0, fn_imp_last=0.0))["R-MATURITY"]
    assert rm["GAP"]["all"]["verdict"] == "NOT DETECTED"
    assert rm["BELIEF"]["verdict"] == "NOT DETECTED"


def test_full_anchors_read_a_frontier():
    rm = RM.read(_table(shift_g=0.0, fn_imp_last=0.0, n=200, mode="full"))["R-MATURITY"]
    assert rm["FRONTIER"]["verdict"] in ("FRONTIER-STATIC", "FRONTIER-MOVES-EARLIER", "FRONTIER-RECEDES",
                                         "FRONTIER-MIXED")



# ---- amendment A1 (2026-10-19): the plateau step's read ----

def _hide(tab, u_rev=0.02, u_hide=0.02, seed=5):
    """Add R3's arms (F+M, F+U, O−M, O−U) to every rec of a table, with a planted U margin on both sides."""
    rng = np.random.default_rng(seed)
    for t in tab.values():
        t["mode"] = "full_hide"
        for r in t["recs"]:
            o, f = r["src"]["O"]["imp"]["LEAF-K8"], r["src"]["F"]["imp"]["LEAF-K8"]
            vals = {"F+M": f + 0.01, "F+U": f + 0.01 + u_rev, "O-M": o - 0.03, "O-U": o - 0.03 - u_hide}
            for k, v in vals.items():
                r["src"][k] = {"ok": True, "imp": {z: v + rng.normal(0, 0.005) for z in RM.SCORERS}}
    return tab


def test_a1_mixed_lean_and_full_reads_no_frontier():
    import copy

    tab = _table(shift_g=0.0, fn_imp_last=0.0, n=200, mode="full")
    mid = copy.deepcopy(tab["a_025M"])
    tab["a_100M"] = {**mid, "label": "a_100M", "step": 100, "mode": "lean"}
    for r in tab["a_100M"]["recs"]:
        r["cp"] = "a_100M"
    # a LEAN anchor BETWEEN two FULL ones: the read runs on the LEAN states (FINDING 4), so no FRONTIER (the registered
    # first-and-last rule would have read one on states where no bucket can open)
    assert RM.read(tab)["R-MATURITY"]["FRONTIER"]["verdict"].startswith("NOT READ")
    assert RM.read(tab, ["a_025M", "a_225M"])["R-MATURITY"]["FRONTIER"]["verdict"].startswith("FRONTIER-")


def test_a1_one_bank_converged_is_not_read_and_r3_reads_u_first():
    tab = _hide(_table(shift_g=0.05, fn_imp_last=0.0, mode="full"))
    rm = RM.read(tab, converged_on_this_bank=False)["R-MATURITY"]
    assert rm["CONVERGED"]["verdict"].startswith("NOT READ (amendment A1")
    assert rm["STILL-MOVING"]["verdict"] == "STILL-SHRINKING"
    r3 = RM.r3_at(tab, "a_225M")
    assert r3["verdict"] == "U-FIRST" and r3["stratum:snapshot"]["verdict"] == "U-FIRST"
    assert r3["n_valid"] == 319                        # the tie at the last anchor is dropped (truth ok elsewhere)
    assert RM.r3_at(_table(0.0, 0.0), "a_225M")["verdict"].startswith("NOT READ")    # no hide-only rows


def test_a1_converged_pooled_needs_both_banks_at_both_anchors():
    ti = _table(shift_g=0.0, fn_imp_last=0.0, n=900, seed=3)
    tp = _table(shift_g=0.0, fn_imp_last=0.0, n=900, seed=4)
    for r in tp["a_025M"]["recs"] + tp["a_225M"]["recs"]:
        r["battle"] = "p" + r["battle"]               # a battle belongs to one bank
    out = RM.converged_pooled({"interim": ti, "plateau": tp}, "a_225M", "a_025M")
    assert out["n_common_per_bank"] == {"interim": 898, "plateau": 898}
    assert out["verdict"] == "CONVERGED"              # no planted change, ~1,800 common states: inside ±0.01
    del tp["a_025M"]
    miss = RM.converged_pooled({"interim": ti, "plateau": tp}, "a_225M", "a_025M")
    assert miss["verdict"].startswith("NOT READ (amendment A1") and "a_025M" in miss["verdict"]


def test_a1_bank_dependent_on_opposite_reads():
    shrink = RM.read(_table(shift_g=0.05, fn_imp_last=0.0))
    grow = RM.read(_table(shift_g=-0.05, fn_imp_last=0.0))
    assert RM._bank_dep(shrink, grow)["GAP:all"] is True
    assert RM._bank_dep(shrink, shrink)["GAP:all"] is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
