"""STAGE 1's REGISTERED RULES as code (``PREREGISTRATION.md`` §6; design §6.2). Pure numpy; no archive, no engine.

Rule 8 (root ``CLAUDE.md``; Tier 1's ``t1_stats``): a bound within ±0.0005 of its boundary is NOT clear. A boundary of
exactly ±0.0005 already carries the margin (the zero edge is written "> +0.0005"); every other boundary c is clear only
beyond c ± 0.0005. "Inside ±m" means lower > −m + 0.0005 and upper < m − 0.0005.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

B = 2000
SEED = 20261019                 # design §6.2: the stage-1 bootstrap seed
EDGE = 0.0005
NULL_MARGIN = 0.01
MATERIAL_R = 0.25
TRIGGER_1B = 0.010
PF = np.round(np.linspace(0.0, 1.0, 21), 2)


class Boot:
    """Tier 1's battle-clustered bootstrap (``maturity_rules.Boot``) with stage 1's seed: ``C[r, b]`` = how often
    battle b is drawn in replicate r; every quantity of the read is evaluated on the SAME draws (paired)."""

    def __init__(self, battles: Sequence[str], seed: int = SEED, n: int = B):
        self.ub = sorted(set(battles))
        ix = {b: i for i, b in enumerate(self.ub)}
        self.bi = np.array([ix[b] for b in battles])
        nb = len(self.ub)
        rng = np.random.default_rng(seed)
        self.C = np.zeros((n, nb))
        draws = rng.integers(0, nb, size=(n, nb))
        for r in range(n):
            np.add.at(self.C[r], draws[r], 1.0)
        self.nb = nb

    def mean(self, x, sel=None) -> Tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        m = np.ones(len(x), bool) if sel is None else np.asarray(sel, bool).copy()
        m &= np.isfinite(x)
        S = np.bincount(self.bi[m], weights=x[m], minlength=self.nb)
        N = np.bincount(self.bi[m], minlength=self.nb).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = (self.C @ S) / (self.C @ N)
        est = float(S.sum() / N.sum()) if N.sum() > 0 else float("nan")
        return est, vec

    def ratio(self, num, den, sel=None) -> Tuple[float, np.ndarray]:
        a, b = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
        m = np.ones(len(a), bool) if sel is None else np.asarray(sel, bool).copy()
        m &= np.isfinite(a) & np.isfinite(b)
        Sa = np.bincount(self.bi[m], weights=a[m], minlength=self.nb)
        Sb = np.bincount(self.bi[m], weights=b[m], minlength=self.nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = (self.C @ Sa) / (self.C @ Sb)
            est = float(Sa.sum() / Sb.sum()) if m.any() and Sb.sum() != 0 else float("nan")
        return est, vec


def ci(vec, level: float = 95.0) -> List[Optional[float]]:
    v = np.asarray(vec, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return [None, None]
    a = (100.0 - level) / 2.0
    lo, hi = np.percentile(v, [a, 100.0 - a])
    return [float(lo), float(hi)]


def _margin(c: float) -> float:
    return 0.0 if abs(abs(c) - EDGE) < 1e-12 else EDGE


def gt(x: Optional[float], c: float) -> bool:
    return x is not None and bool(np.isfinite(x)) and x > c + _margin(c)


def lt(x: Optional[float], c: float) -> bool:
    return x is not None and bool(np.isfinite(x)) and x < c - _margin(c)


def inside(lohi: Sequence[Optional[float]], m: float) -> bool:
    lo, hi = lohi
    return lo is not None and hi is not None and lo > -m + EDGE and hi < m - EDGE


def summary(est: float, vec, n: Optional[int] = None) -> dict:
    lo, hi = ci(vec)
    out: dict = {"mean": None if not np.isfinite(est) else round(float(est), 5),
                 "ci": [None if lo is None else round(lo, 5), None if hi is None else round(hi, 5)]}
    if n is not None:
        out["n"] = int(n)
    return out


def delta_verdict(leaf_ci: Sequence[Optional[float]], t4_ci: Sequence[Optional[float]]) -> str:
    """PAYS (lower > +0.0005 at LEAF-K8 AND TRUNC-4-K8) > HURTS (upper < −0.0005 at both) > NULL (the LEAF-K8
    interval inside ±0.01) > NOT DETECTED."""
    if gt(leaf_ci[0], EDGE) and gt(t4_ci[0], EDGE):
        return "PAYS"
    if lt(leaf_ci[1], -EDGE) and lt(t4_ci[1], -EDGE):
        return "HURTS"
    if inside(leaf_ci, NULL_MARGIN):
        return "NULL"
    return "NOT DETECTED"


def material(verdict: str, r_ci: Sequence[Optional[float]]) -> bool:
    """MATERIAL iff it PAYS and r_X = Δ_X / G_F_legal (LEAF-K8) has a lower bound >= 0.25 (clear of it)."""
    return verdict == "PAYS" and gt(r_ci[0], MATERIAL_R)


def search_enabled(imp_leaf_ci: Sequence[Optional[float]], t4_minus_crit_ci: Sequence[Optional[float]]) -> bool:
    """lower95[IMP_X] > +0.0005 at LEAF-K8 AND lower95[IMP_X − IMP_CRITIC,X] > +0.0005 at TRUNC-4-K8."""
    return gt(imp_leaf_ci[0], EDGE) and gt(t4_minus_crit_ci[0], EDGE)


def eq_p(curve, target: float, grid=PF) -> Optional[float]:
    """The p at which the dose curve first reaches ``target`` (linear interpolation); 0 at or below the p = 0 value;
    None above the p = 1 value (Tier 1's ``unblind_t1.eq_p``, ``maturity_rules.eq_p``)."""
    c = np.asarray(curve, dtype=np.float64)
    g = np.asarray(grid, dtype=np.float64)
    if not np.isfinite(target) or not np.all(np.isfinite(c)):
        return None
    if target <= c[0]:
        return 0.0
    for i in range(1, len(c)):
        if c[i] >= target:
            a, b = c[i - 1], c[i]
            return float(g[i - 1] + (g[i] - g[i - 1]) * (target - a) / (b - a)) if b > a else float(g[i])
    return None


def today_clamped(curve_est, target_est: float, curve_draws, target_draws, grid=PF):
    """(TODAY's point, its per-draw vector), every value clamped to [0, 1] (above the curve -> 1.0)."""
    def one(c, t) -> float:
        if not np.isfinite(t) or not np.all(np.isfinite(c)):
            return float("nan")
        p = eq_p(c, t, grid)
        return 1.0 if p is None else min(1.0, max(0.0, p))

    pt = one(np.asarray(curve_est), target_est)
    vec = np.array([one(c, t) for c, t in zip(np.asarray(curve_draws), np.asarray(target_draws))])
    return (None if not np.isfinite(pt) else pt), vec


def contains(today_ci: Sequence[Optional[float]], q: float) -> str:
    """Does TODAY_X's interval contain q_SM,X? CONTAINS / EXCLUDES / BOUNDARY (rule 8). TODAY is clamped to [0, 1]: a
    lower bound AT 0 (or an upper bound at 1) is the clamp, open on that side."""
    lo, hi = today_ci
    if lo is None or hi is None or not np.isfinite(q):
        return "BOUNDARY"
    low_ok = lo <= 0.0 or lo < q - EDGE
    high_ok = hi >= 1.0 or hi > q + EDGE
    low_out = lo > 0.0 and q < lo - EDGE
    high_out = hi < 1.0 and q > hi + EDGE
    if low_out or high_out:
        return "EXCLUDES"
    return "CONTAINS" if (low_ok and high_ok) else "BOUNDARY"


def calibrated(per_arm: Dict[str, str]) -> str:
    vals = list(per_arm.values())
    if not vals:
        return "NOT READ"
    if any(v == "EXCLUDES" for v in vals):
        return "NOT CALIBRATED"
    return "CALIBRATED" if all(v == "CONTAINS" for v in vals) else "UNDECIDED"


def trigger_1b(deltas: Dict[str, Dict[str, float]], enabled: Dict[str, bool]) -> dict:
    """Design §6.2: 1b triggers on any arm with Δ_X >= +0.010 (the point) at either scorer, or any C / NS arm
    SEARCH-ENABLED."""
    why = [f"{x}: Δ {z} {v:+.4f}" for x, d in deltas.items() for z, v in d.items()
           if v is not None and np.isfinite(v) and v >= TRIGGER_1B]
    why += [f"{x}: SEARCH-ENABLED" for x, e in enabled.items() if e and (x.startswith("C_") or x == "NS")]
    return {"trigger": bool(why), "why": why}


#: design §6.2 "What each result means", printed beside the verdicts (descriptive)
MEANING = {
    "C_MATERIAL": "a team-level model is the path: next owner Q1, the dice-marginalised check (e), a ladder-population read",
    "ONLY_POOL": "team identification works but our corpora do not cover the opponents: the lever is coverage or (f)",
    "NOTHING": "information is not the limit at this leaf: re-read with the matrix (§5) and the evaluator",
    "B_PAYS": "the Smogon-only joint goes into F as the new baseline",
    "PARTIAL": "worlds that are partly right carry the value: the target is decision-relevant correctness, not exactness",
    "EXACT": "exactness IS the requirement: the honest path runs through the opponent's choices (f) or public teams",
}


def meaning(verdicts: Dict[str, str], mat: Dict[str, bool], qsm: Dict[str, float]) -> List[str]:
    out = []
    pays = {x for x, v in verdicts.items() if v == "PAYS"}
    if any(mat.get(x) for x in ("C_ladder", "C_forum_strict")):
        out.append(MEANING["C_MATERIAL"])
    if pays == {"C_pool"}:
        out.append(MEANING["ONLY_POOL"])
    if not pays:
        out.append(MEANING["NOTHING"])
    if "B" in pays:
        out.append(MEANING["B_PAYS"])
    if any(x in pays and qsm.get(x, 1.0) < 0.01 for x in ("NS", "C_ladder")):
        out.append(MEANING["PARTIAL"])
    elif pays and pays <= {"C_pool"}:
        out.append(MEANING["EXACT"])
    return out
