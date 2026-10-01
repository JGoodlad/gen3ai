#!/usr/bin/env python3
"""The FEWER-ACTIVE-SNAPSHOTS construction check (M5 sizing study, REGISTRATION §2.3 O7) — a DESCRIPTOR.

The owner (2026-09-30) accepts a rotating active K-subset of the self-play pool ONLY if each opponent's
episode frequency equals the target weighting p. The program doc's Decision-record construction:

1. inclusion probabilities by iterative capping: pi_i = 1 for every i with K * p_i >= 1 after
   rescaling over the uncapped, else pi_i = c * p_i, with c such that sum(pi) = K;
2. a FIXED-SIZE SYSTEMATIC PPS design that realises pi exactly (cumulate pi in a fixed order, draw
   u ~ U[0, 1), take the unit whose cumulative interval holds u + j for j = 0 .. K-1);
3. within the subset, pick i proportional to p_i / pi_i.

Then sum_{j in S} p_j / pi_j = 1 for EVERY subset S (capped units are always in S; each uncapped one
contributes 1/c, and there are K - m of them), so P(i) = pi_i * (p_i / pi_i) = p_i exactly.

This script checks it two ways for three weightings: EXACTLY, by integrating the systematic design over
u (its sample is a step function of u — every breakpoint enumerated), and by SIMULATION (10^6 episode
draws, max |realised - p_i| against its binomial SE). It also shows the naive min(1, K p_i) rule's
error, which is what step 1's capping fixes. Pure numpy; no training, no GPU.

    python3 pps_check.py [--draws 1000000] [--seed 20261001] [--out pps_check.json]
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np


def capped_inclusion(p: np.ndarray, k: int) -> np.ndarray:
    """pi with sum K: iterative capping at 1, the rest proportional to p."""
    p = np.asarray(p, dtype=np.float64)
    if not (0 < k <= p.size):
        raise ValueError(f"K={k} for {p.size} units")
    pi = np.zeros_like(p)
    capped = np.zeros(p.size, dtype=bool)
    while True:
        m = int(capped.sum())
        rest = p[~capped].sum()
        c = (k - m) / rest
        new = (~capped) & (c * p >= 1.0)
        if not new.any():
            pi[capped] = 1.0
            pi[~capped] = c * p[~capped]
            return pi
        capped |= new


def naive_inclusion(p: np.ndarray, k: int) -> np.ndarray:
    return np.minimum(1.0, k * np.asarray(p, dtype=np.float64))


def systematic_sample(pi: np.ndarray, u: float) -> np.ndarray:
    """Units hit by u + j, j = 0..K-1, on the cumulative pi line (sum(pi) = K)."""
    cum = np.concatenate(([0.0], np.cumsum(pi)))
    k = int(round(cum[-1]))
    cum[-1] = float(k)                       # sum(pi) = K up to rounding: pin the end of the line
    pts = u + np.arange(k)
    return np.minimum(np.searchsorted(cum, pts, side="right") - 1, pi.size - 1)


def exact_marginal(p: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """P(opponent i) under (systematic design over pi) x (pick ∝ p/pi within the subset), integrated
    EXACTLY over u: the sample changes only where u + j crosses a cumulative boundary."""
    cum = np.concatenate(([0.0], np.cumsum(pi)))
    br = np.unique(np.concatenate(([0.0, 1.0], np.mod(cum, 1.0))))
    br = br[(br >= 0) & (br <= 1)]
    out = np.zeros_like(p, dtype=np.float64)
    w = p / pi
    for a, b in zip(br[:-1], br[1:]):
        if b - a <= 0:
            continue
        s = systematic_sample(pi, 0.5 * (a + b))
        out[s] += (b - a) * w[s] / w[s].sum()
    return out


def simulate(p: np.ndarray, pi: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    w = p / pi
    counts = np.zeros(p.size, dtype=np.int64)
    us = rng.random(draws)
    vs = rng.random(draws)
    for u, v in zip(us, vs):
        s = systematic_sample(pi, float(u))
        q = np.cumsum(w[s] / w[s].sum())
        counts[s[min(int(np.searchsorted(q, v, side="right")), s.size - 1)]] += 1
    return counts / draws


def weightings() -> Dict[str, np.ndarray]:
    n = 20
    uniform = np.full(n, 1.0 / n)
    # PFSP-like: hard opponents weighted up; one dominant (K * p > 1 at K = 8) ⇒ capped
    hard = np.linspace(0.05, 0.95, n) ** 3
    hard[-1] *= 6.0
    recency = np.exp(np.arange(n) / 5.0)
    return {"uniform20": uniform, "pfsp_skew_capped": hard / hard.sum(), "recency": recency / recency.sum()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--draws", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    res: Dict[str, dict] = {}
    for name, p in weightings().items():
        pi = capped_inclusion(p, a.k)
        ex = exact_marginal(p, pi)
        sim = simulate(p, pi, a.draws, rng)
        se = np.sqrt(p * (1 - p) / a.draws)
        npi = naive_inclusion(p, a.k)
        naive = None
        if abs(npi.sum() - a.k) > 1e-9:          # the naive rule is not even a fixed-size-K design here
            naive = {"sum_pi": float(npi.sum()), "note": "naive min(1, K p) does not sum to K: no fixed-size design realises it"}
        res[name] = {"K": a.k, "n_capped": int((pi >= 1 - 1e-12).sum()), "sum_pi": float(pi.sum()),
                     "exact_max_abs_err": float(np.abs(ex - p).max()),
                     "sim_draws": a.draws, "sim_max_abs_err": float(np.abs(sim - p).max()),
                     "sim_max_z": float((np.abs(sim - p) / se).max()),
                     "naive": naive}
        print(f"{name:18s} capped={res[name]['n_capped']} exact max|P-p|={res[name]['exact_max_abs_err']:.2e} "
              f"sim max|f-p|={res[name]['sim_max_abs_err']:.2e} (max z {res[name]['sim_max_z']:.2f})"
              + (f"  naive sum(pi)={naive['sum_pi']:.3f}" if naive else ""))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
