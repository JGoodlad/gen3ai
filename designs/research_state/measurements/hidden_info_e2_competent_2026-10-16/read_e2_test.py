"""Planted-effect tests of E2's read (`read_e2.py`): the §3.4 verdicts must come out right on synthetic per-state IMPs,
and the k = 6 identity check must catch a planted difference. Pure; no archive is read.

    python -I -m pytest designs/research_state/measurements/hidden_info_e2_competent_2026-10-16/read_e2_test.py -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import read_e2 as R  # noqa: E402

Z = R.PRIMARY


def recs(effects, n=600, noise=0.02, seed=0, k=None):
    """Synthetic records: IMP of each source = its planted mean + noise; 2 states per battle, two checkpoints."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        cp = "end2" if i % 2 else "prod"
        src = {}
        for s in R.SOURCES:
            v = effects.get(s, 0.0) + rng.normal(0, noise)
            src[s] = {"status": "ok", "imp": {z: v for z in R.scorers()}, "out": {"0": "++++++++"}}
        out.append({"id": f"s{i}", "cp": cp, "battle": f"{cp}:b{i // 2}", "k": k if k is not None else 1 + i % 6,
                    "k_bin": "1-2", "outcome": "win", "move_slots": 5.0, "src": src,
                    "truth_out8": {"0": "++++++++"}})
    return out


def verdict_of(effects, **kw):
    return R.read_frame(recs(effects, **kw), True, Z)["verdict"]


def test_u_first():
    # σ_U = F+U − F large, σ_M small; ν_U = O − O−U large, ν_M small
    assert verdict_of({"O": 0.07, "F": 0.0, "F+U": 0.03, "F+M": 0.005, "O-U": 0.02, "O-M": 0.06}) == "U-FIRST"


def test_m_first():
    assert verdict_of({"O": 0.07, "F": 0.0, "F+U": 0.005, "F+M": 0.03, "O-U": 0.06, "O-M": 0.02}) == "M-FIRST"


def test_shared_when_both_detected_and_equal():
    assert verdict_of({"O": 0.07, "F": 0.0, "F+U": 0.02, "F+M": 0.02, "O-U": 0.04, "O-M": 0.04}) == "SHARED"


def test_not_detected_when_nothing():
    assert verdict_of({}) == "NOT DETECTED"


def test_sides_disagree_is_shared_with_label():
    fr = R.read_frame(recs({"O": 0.07, "F": 0.0, "F+U": 0.03, "F+M": 0.005, "O-U": 0.06, "O-M": 0.02}), True, Z)
    assert fr["verdict"] == "SHARED" and fr["qualifiers"]["sides_disagree"] and "SIDES-DISAGREE" in fr["label"]


def test_rule8_edge_is_not_clear():
    # a reveal-only numerator whose whole interval sits at ~+0.0003 (inside the ±0.0005 band) is not a direction
    fr = R.read_frame(recs({"F+U": 0.0003, "O-M": 0.0003}, noise=1e-6), True, Z)
    assert fr["qualifiers"]["direction_reveal_only"] == "none"
    assert fr["qualifiers"]["direction_hide_only"] == "none"


def test_k6_identity_catches_a_difference():
    rs = recs({}, n=20, k=6)
    assert R.validity(rs)["valid"]
    rs[3]["src"]["F+U"]["out"] = {"0": "+++++++-"}
    v = R.validity(rs)
    assert not v["valid"] and v["k6_identities"]["F+U ≡ F at k = 6"]["differing"] == 1
