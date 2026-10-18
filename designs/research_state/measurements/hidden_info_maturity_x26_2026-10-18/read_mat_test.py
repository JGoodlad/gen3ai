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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
