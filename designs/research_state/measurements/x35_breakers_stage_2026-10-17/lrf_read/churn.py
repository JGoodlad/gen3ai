"""LR-F look 1, AM-LRF-L1 item 8: greedy churn per 20M on ``bank_x26`` (REPORTED; STAGED_PLAN §4.1.2 predicts
0.33-0.42 for LR-F, from ~0.30). Never a decision.

Input: a ``main.policy_spectrum read`` dir over ``bank_x26`` holding ``pstar`` (P*, the fork start) and each final
(``lrf_s350x``, ``c0_s350x``). Churn = the share of the bank's decisions whose greedy (argmax) action differs between
P* and the final (X26 collapse battery (f)'s definition; every bank decision has >= 2 legal actions); also the mean
total-variation distance between the two action distributions. Prints one JSON document.

    python lrf_read/churn.py <reads dir> > lrf_read/churn.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

LABELS = [f"{a}_s{s}" for a in ("lrf", "c0") for s in (3501, 3502, 3503)]


def load(d: Path, lab: str) -> np.ndarray:
    return np.load(d / f"{lab}.probs.npz")["probs"].astype(np.float64)


def main(argv: list) -> int:
    d = Path(argv[0])
    p0 = load(d, "pstar")
    out = {"schema": "x35_lrf_churn_v1", "reported_only": True, "reads": str(d), "n": int(p0.shape[0]),
           "predicted_lrf": [0.33, 0.42], "vs": "pstar", "rows": {}}
    for lab in LABELS:
        p = load(d, lab)
        if p.shape != p0.shape:
            raise SystemExit(f"{lab}: shape {p.shape} != pstar's {p0.shape}")
        out["rows"][lab] = {"greedy_flip_share": float(np.mean(p.argmax(1) != p0.argmax(1))),
                            "mean_tv": float(0.5 * np.abs(p - p0).sum(1).mean())}
    for arm in ("lrf", "c0"):
        v = [out["rows"][f"{arm}_s{s}"]["greedy_flip_share"] for s in (3501, 3502, 3503)]
        out[f"{arm}_mean_flip_share"] = float(np.mean(v))
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
