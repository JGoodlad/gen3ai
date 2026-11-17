"""DRAFT (i)'s coefficient rule: which small ``--ent-coef`` should end a P* + 20M fork at the swap's entropy.

A frozen record of a DRAFT, not a tool. It reads the banked end entropies (the mean of a run's last 20 points of
-train/entropy_loss, the definition X67, CE-3d and X70 share) and X70's four, which do not exist yet and are typed.

    python coef_rule.py selfcheck                      the banked inputs and the rule on hypothetical X70 values
    python coef_rule.py pick  <e1> <e2> <e3> <e4>      c* from X70's four end entropies (class (i) only)
    python coef_rule.py check <c*> <h1> <e1> .. <e4>   after the first matched fork: KEEP, or c** from its point

The rule (DRAFT_PREREG_i_matched_entropy.md §4). Three points of end entropy against the coefficient, all at the
same horizon (+20M from P*): (0, H0) = X70, (0.0125, H_X67), (0.05, H_C0). Fit the one monotone curve
H(c) = H0 + a * c**p through all three (p from the two banked points and H0), and invert it at the target
T = the mean of the swap's three end entropies. Deterministic: no draw, no fit tolerance, the output rounded to
1e-5 (a coefficient typed with five decimals).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
X67_JSON = HERE.parent / "x67_entropy_screen_2026-11-09" / "record" / "diag9_entropy.json"
CE3D_JSON = HERE.parent / "ce3d_magnet_screen_2026-11-11" / "record" / "diag9_ce3d.json"
C_X67 = 0.0125
C_C0 = 0.05
#: KEEP band after the first matched fork, nats: twice the largest within-arm SD of a banked end entropy
#: (X67's three, 0.0148; the swap's 0.0063; C0's eight 0.0040), rounded up.
TOL = 0.030
RULE8_EPS = 1e-9


def _ends(path: Path, prefix: str) -> List[float]:
    runs = json.loads(path.read_text())["runs"]
    return [float(v["train/entropy_loss"]["end_mean"]) for k, v in sorted(runs.items()) if k.startswith(prefix)]


def banked() -> Dict[str, List[float]]:
    return {"c0": _ends(X67_JSON, "rb_x35_c0_"), "x67": _ends(X67_JSON, "rb_x35_x67ent_"),
            "swap": _ends(CE3D_JSON, "rb_x35_ce3dswap_")}


def power_fit(h0: float, h_x67: float, h_c0: float) -> Tuple[float, float]:
    """(p, a) of H(c) = h0 + a * c**p through (0, h0), (C_X67, h_x67), (C_C0, h_c0)."""
    if not h0 < h_x67 < h_c0:
        raise ValueError(f"not monotone: H(0) {h0} < H(0.0125) {h_x67} < H(0.05) {h_c0} fails")
    p = math.log((h_x67 - h0) / (h_c0 - h0)) / math.log(C_X67 / C_C0)
    a = (h_x67 - h0) / C_X67 ** p
    return p, a


def invert(target: float, h0: float, h_x67: float, p: float) -> float:
    return C_X67 * ((target - h0) / (h_x67 - h0)) ** (1.0 / p)


def pick(x70: Sequence[float]) -> Dict[str, float]:
    b = banked()
    h0, h_x67, h_c0, t = mean(x70), mean(b["x67"]), mean(b["c0"]), mean(b["swap"])
    if not max(x70) < min(b["swap"]) - RULE8_EPS:
        raise ValueError(f"X70 is not ENT-BELOW-SWAP (highest {max(x70):.4f} vs the swap's lowest "
                         f"{min(b['swap']):.4f}): class (i) does not apply and no coefficient is picked")
    if t - h0 <= TOL + RULE8_EPS:
        raise ValueError(f"X70-IS-MATCHED: X70's mean {h0:.4f} is within the KEEP band ({TOL}) of the target "
                         f"{t:.4f}, so X70's own cross already compared the anchor at a matched entropy; no arm")
    p, a = power_fit(h0, h_x67, h_c0)
    c = invert(t, h0, h_x67, p)
    return {"H0": h0, "H_x67": h_x67, "H_c0": h_c0, "target": t, "p": p, "a": a, "c_star": round(c, 5),
            "c_linear_p1": round(invert(t, h0, h_x67, 1.0), 5), "c_sqrt_p05": round(invert(t, h0, h_x67, 0.5), 5)}


def check(c_star: float, h1: float, x70: Sequence[float]) -> Dict[str, object]:
    """After the first matched fork ends at h1: KEEP if |h1 - T| <= TOL, else re-pick ONCE by linear interpolation
    between the two measured points that bracket T (X70's mean, the fork's, X67's mean)."""
    b = banked()
    t = mean(b["swap"])
    if abs(h1 - t) <= TOL - RULE8_EPS:
        return {"target": t, "h1": h1, "verdict": "KEEP", "c": c_star}
    pts = sorted([(0.0, mean(x70)), (c_star, h1), (C_X67, mean(b["x67"]))])
    if any(pts[i][1] >= pts[i + 1][1] for i in range(2)):
        return {"target": t, "h1": h1, "verdict": "HOLD", "why": "the three points are not monotone", "pts": pts}
    for (c_lo, h_lo), (c_hi, h_hi) in zip(pts, pts[1:]):
        if h_lo <= t <= h_hi:
            c2 = c_lo + (c_hi - c_lo) * (t - h_lo) / (h_hi - h_lo)
            return {"target": t, "h1": h1, "verdict": "REPICK", "c": round(c2, 5), "between": [c_lo, c_hi]}
    return {"target": t, "h1": h1, "verdict": "HOLD", "why": "the target is outside the measured points", "pts": pts}


def selfcheck() -> None:
    b = banked()
    for k, v in b.items():
        print(f"{k:5s} n={len(v)} mean={mean(v):.4f} sd={stdev(v):.4f} min={min(v):.4f} max={max(v):.4f}")
    print(f"target T = mean(swap) = {mean(b['swap']):.5f}; KEEP band +-{TOL}")
    print("hypothetical X70 mean H0 -> p, c* (power law), c (p = 1), c (p = 0.5)")
    for h0 in (0.10, 0.20, 0.30, 0.342, 0.40, 0.45, 0.48):
        p, _ = power_fit(h0, mean(b["x67"]), mean(b["c0"]))
        t = mean(b["swap"])
        print(f"  H0 {h0:.3f}: p {p:.3f}  c* {invert(t, h0, mean(b['x67']), p):.5f}  "
              f"linear {invert(t, h0, mean(b['x67']), 1.0):.5f}  sqrt {invert(t, h0, mean(b['x67']), 0.5):.5f}")
    # the two banked points alone, extrapolated to c = 0 under each form (a prediction for X70, binding nothing)
    hx, hc = mean(b["x67"]), mean(b["c0"])
    print(f"banked-only extrapolation to c = 0: linear {hx - (hc - hx) / 3.0:.3f} nats; "
          f"sqrt {hx - (hc - hx) / (2.0 - 1.0):.3f} nats")


def main(argv: Sequence[str]) -> int:
    if not argv or argv[0] == "selfcheck":
        selfcheck()
        return 0
    if argv[0] == "pick" and len(argv) == 5:
        try:
            print(json.dumps(pick([float(x) for x in argv[1:]]), indent=1))
        except ValueError as e:
            print(f"REFUSED: {e}")
            return 3
        return 0
    if argv[0] == "check" and len(argv) == 7:
        print(json.dumps(check(float(argv[1]), float(argv[2]), [float(x) for x in argv[3:]]), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
