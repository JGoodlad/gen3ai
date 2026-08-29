"""Synthetic validation of PROBE G's estimators.

Three claims, each checked against a construction whose truth is known:
 1. the per-decision identity  mean_a e^2 == offset^2 + mean_a resid^2
 2. the split-half noise floor RECOVERS the true residual/offset MSE when the critic is a PURE
    OFFSET of the truth (true differential MSE == 0, so the corrected number must be ~0)
 3. it does NOT eat real differential error (a critic with a known differential error is
    recovered at the right magnitude)
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from critic_bias_split_analyze import per_decision

RNG = np.random.default_rng(0)
R_HALF = 32


def synth(n_dec, *, differential_sd, offset_sd, A=7):
    rows = []
    for d in range(n_dec):
        p = np.clip(RNG.uniform(0.05, 0.95) + RNG.normal(0, 0.15, A), 0.02, 0.98)   # true P(win) per action
        off = RNG.normal(0, offset_sd)
        C = np.clip(p + off + RNG.normal(0, differential_sd, A), 0, 1)              # the "critic"
        LA = RNG.binomial(R_HALF, p) / R_HALF
        LB = RNG.binomial(R_HALF, p) / R_HALF
        rows.append({"battle": f"b{d//3}", "short": "", "inv": d, "turn": 10, "opponent": "x",
                     "opp_class": "bot", "outcome": "win", "stratum": "ordinary", "n_legal": A,
                     "n_used": A, "td_delta": 0.0, "rec_win_prob": 0.5, "chosen": 0, "R": 2*R_HALF,
                     "actions": list(range(A)), "C": C, "LA": LA, "LB": LB,
                     "terminal": [None]*A, "_p": p, "_off": off, "_dsd": differential_sd})
    return rows


def check(name, differential_sd, offset_sd, n=4000):
    rows = per_decision(synth(n, differential_sd=differential_sd, offset_sd=offset_sd))
    raw_r = np.mean([r["mse_resid"] for r in rows])
    nf_r = np.mean([r["noise_mse_resid"] for r in rows])
    raw_o = np.mean([r["mse_offset"] for r in rows])
    nf_o = np.mean([r["noise_mse_offset"] for r in rows])
    # GROUND TRUTH read off the construction itself (C vs the true p), not off the nominal sd --
    # the clip() that keeps C a probability truncates the injected noise, so the nominal formula
    # over-states the truth at large sd. The estimator is being checked against what was actually
    # built, which is the only honest reference.
    et = [r["C"] - r["_p"] for r in rows]
    true_r = float(np.mean([((e - e.mean()) ** 2).mean() for e in et]))
    true_o = float(np.mean([e.mean() ** 2 for e in et]))
    ident = max(abs(r["mse_e"] - (r["mse_offset"] + r["mse_resid"])) for r in rows)
    print(f"{name}\n  identity max|Δ| = {ident:.3e}")
    print(f"  residual: raw {raw_r:.5f}  - floor {nf_r:.5f}  = {raw_r-nf_r:+.5f}   (true {true_r:.5f})")
    print(f"  offset  : raw {raw_o:.5f}  - floor {nf_o:.5f}  = {raw_o-nf_o:+.5f}   (true {true_o:.5f})")
    assert ident < 1e-12, "decomposition identity broken"
    return raw_r - nf_r, true_r


a, ta = check("PURE OFFSET critic (true differential = 0)", differential_sd=0.0, offset_sd=0.10)
assert abs(a - ta) < 0.0015, f"floor did not zero out: {a} vs {ta}"
b, tb = check("REAL differential error sd=0.08", differential_sd=0.08, offset_sd=0.10)
assert abs(b - tb) / tb < 0.06, f"floor ate real signal: {b} vs {tb}"
c, tc = check("LARGE differential error sd=0.20", differential_sd=0.20, offset_sd=0.05)
assert abs(c - tc) / tc < 0.06, f"{c} vs {tc}"
print("\nALL SELF-TESTS PASS")
