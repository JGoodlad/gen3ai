"""The EXTREME-VALUE read behind the TF32 max condition's persistence rule (``tail.py``'s rows).

The statistic the gate reads each update is the max |Δ log π| over ONE 2,048-row micro-batch (the
first). From ``parts/tail_u*.npz`` (independent seeds: fresh rollout rows, a fresh perturbation each):

1. EMPIRICAL — the micro-batch maxima: n, quantiles, the largest; and the per-row tail.
2. PEAKS OVER THRESHOLD on the ROWS (the primary model): a generalized Pareto above the per-row
   ``U_Q`` quantile, so q(x) = P(row |Δ| > x); a micro-batch max exceeds x with p(x) = 1 − (1 − q)^2048
   (rows i.i.d. within a micro-batch — ASSUMED).
3. GEV on the micro-batch maxima (a cross-check; few blocks).
Then, per update p = p(BAR): P(≥ 1 crossing in N updates) = 1 − (1 − p)^N and P(≥ 1 pair of
CONSECUTIVE crossings) ≈ 1 − (1 − p²)^(N − 1) (updates independent — ASSUMED: each update scores fresh
rollout rows), and the expected max over N updates (the 1/N return level). A seed-level bootstrap gives
the 95 % interval on p. ``--write`` stores ``tail_result.json``.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
BAR, MICRO, N_UPDATES = 0.071, 2048, 10_000
U_Q = 0.999
PERSISTENCE = (2, 3, 4)
P99_BAR = 3.6e-3


def load(paths=None):
    paths = sorted(glob.glob(str(HERE / "parts" / "tail_u*.npz"))) if paths is None else paths
    units = [np.load(p) for p in paths if ".partial" not in p]
    rows = [u["rows"].astype(np.float64) for u in units]
    seeds = np.concatenate([u["seeds"] for u in units])
    per_seed = [r for u in rows for r in np.split(u, len(u) // 6144)]
    micro = np.concatenate([u["micro_max"] for u in units])
    return per_seed, micro, seeds


def pot_p(rows: np.ndarray, x: float):
    u = np.quantile(rows, U_Q)
    exc = rows[rows > u] - u
    xi, _loc, sigma = stats.genpareto.fit(exc, floc=0.0)
    zeta = exc.size / rows.size
    q = zeta * stats.genpareto.sf(x - u, xi, 0.0, sigma)
    p = 1.0 - (1.0 - q) ** MICRO
    # the per-update max's 1/N return level: the row quantile at 1 − 1/(MICRO·N)
    tail = 1.0 / (MICRO * N_UPDATES)
    ret = u + stats.genpareto.isf(tail / zeta, xi, 0.0, sigma)
    return {"u": float(u), "xi": float(xi), "sigma": float(sigma), "n_exceed": int(exc.size), "q_row": float(q),
            "p_update": float(p), "return_level_10k": float(ret)}


def rates(p: float) -> dict:
    return {"p_update": p, "P_at_least_one_crossing_10k": 1.0 - (1.0 - p) ** N_UPDATES,
            **{f"P_{k}_consecutive_10k": 1.0 - (1.0 - p ** k) ** (N_UPDATES - k + 1) for k in PERSISTENCE}}


def derive() -> dict:
    per_seed, micro, seeds = load()
    rows = np.concatenate(per_seed)
    out = {"setup": {"bar": BAR, "micro": MICRO, "n_updates": N_UPDATES, "pot_threshold_quantile": U_Q,
                     "seeds": int(seeds.size), "rows": int(rows.size), "micro_batches": int(micro.size),
                     "assumptions": ["rows i.i.d. within a micro-batch", "updates independent (fresh rows each)",
                                     "a fresh perturbed production learner stands in for a trained one",
                                     "TF32 eager on the RTX 3080 Ti, torch 2.5.1"]}}
    out["empirical"] = {"micro_max": {"n": int(micro.size), "median": float(np.median(micro)),
                                      "q90": float(np.quantile(micro, 0.9)), "q99": float(np.quantile(micro, 0.99)),
                                      "max": float(micro.max()), "n_over_bar": int((micro >= BAR).sum())},
                        "rows": {"n": int(rows.size), "p99_9": float(np.quantile(rows, 0.999)),
                                 "p99_99": float(np.quantile(rows, 0.9999)), "max": float(rows.max()),
                                 "n_over_bar": int((rows >= BAR).sum())}}
    pot = pot_p(rows, BAR)
    out["pot_rows"] = {**pot, **rates(pot["p_update"])}
    # the single-shot p99 condition, on the same rows: each seed's rows are 3 micro-batches in order
    p99s = np.array([np.quantile(r[i:i + MICRO], 0.99) for r in per_seed for i in range(0, r.size, MICRO)])
    c9, l9, s9 = stats.genextreme.fit(p99s)
    p9 = float(stats.genextreme.sf(P99_BAR, c9, l9, s9))
    out["p99_single_shot"] = {"n_micro": int(p99s.size), "median": float(np.median(p99s)), "max": float(p99s.max()),
                              "headroom": P99_BAR / float(p99s.max()), "gev_p_update": p9,
                              "P_at_least_one_crossing_10k": 1.0 - (1.0 - p9) ** N_UPDATES}
    c, loc, sc = stats.genextreme.fit(micro)
    pg = float(stats.genextreme.sf(BAR, c, loc, sc))
    out["gev_micro_max"] = {"shape_scipy_c": float(c), "loc": float(loc), "scale": float(sc),
                            "return_level_10k": float(stats.genextreme.isf(1.0 / N_UPDATES, c, loc, sc)), **rates(pg)}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(200):
        pick = rng.integers(0, len(per_seed), len(per_seed))
        boots.append(pot_p(np.concatenate([per_seed[i] for i in pick]), BAR)["p_update"])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    out["pot_rows"]["p_update_95ci_seed_bootstrap"] = [float(lo), float(hi)]
    out["pot_rows"]["rates_at_ci_high"] = rates(float(hi))
    return out


if __name__ == "__main__":
    res = derive()
    print(json.dumps(res, indent=1))
    if "--write" in sys.argv:
        (HERE / "tail_result.json").write_text(json.dumps(res, indent=1) + "\n")
