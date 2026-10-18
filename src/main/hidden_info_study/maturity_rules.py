"""THE MATURITY BATTERY's decision rules as code: R-MATURITY (`design_hidden_info_binding.md` §12.13.4, read over the
§12.13.7 trajectory) and the per-checkpoint gate it consumes (R-STAGE-SEARCH, re-applied to ONE checkpoint).

Conventions (Tier 1's, `hidden_info_tier1_2026-10-13/t1_stats.py`; `maturity_rules_test` checks the bootstrap draws
equal Tier 1's exactly):

* the bootstrap: 2,000 draws, ``numpy.random.default_rng(20261011)``, battles resampled with replacement, percentile
  intervals; every quantity of one read is evaluated on the SAME draws. The battery scores ONE frozen bank at every
  checkpoint, so one :class:`Boot` over the bank's battles makes every across-checkpoint contrast PAIRED;
* RULE 8: a bound within ±0.0005 of its boundary is a BOUNDARY result and counts as not clear. A boundary of exactly
  ±0.0005 already carries the margin (the registration writes the zero edge as "> +0.0005"); any other boundary c is
  clear only beyond c ± 0.0005; "inside ±m" means lower > −m + 0.0005 and upper < m − 0.0005.

R-MATURITY (first = the series' first anchor, last = its last; per-draw vectors paired):

* GAP-SHRINKS iff upper95 [G_F(last) − G_F(first)] < −0.0005; GAP-GROWS iff lower95 > +0.0005; else NOT DETECTED.
  Per cell: the all-state cell and every stage bucket alike.
* BELIEF-RISES iff lower95 [TODAY(last) − TODAY(first)] > +0.0005, else NOT DETECTED. TODAY per draw is the
  p-equivalent of FN on the M8 curve, CLAMPED to [0, 1] (FN at or below the p = 0 value reads 0; above the p = 1 value
  reads 1), so a difference is defined on every draw.
* FRONTIER (on each stage axis; "earlier" = lower k, lower turn, MORE material): E(cp) = the earliest-game GATE-OPEN
  bucket, or "none" (later than every bucket). FRONTIER-MOVES-EARLIER iff on some axis E(last) is strictly earlier
  than E(first); FRONTIER-RECEDES iff on some axis E(last) is strictly later; both (on different axes) = FRONTIER-MIXED;
  neither = FRONTIER-STATIC. GATE-WEAK is not open.
* CONVERGED iff the interval of G_F(last) − G_F(previous anchor) on the all-state cell lies inside ±0.01.

The per-checkpoint GATE (R-STAGE-SEARCH with ONE network, the registration's single-checkpoint reading): GATE-OPEN iff
n >= 150 AND lower [TRUNC-4-K8 − CRITIC] > +0.0005 AND lower [TRUNC-4-K8] > +0.0005 at the 99.6 % level (Bonferroni
over the 13 primary buckets); GATE-WEAK iff both hold at 95 % only; else GATE-CLOSED. ("Each checkpoint separately" and
"pooled" coincide for one network.)

PURE: numpy + stdlib, no ``main`` / ``agents`` import (the reader loads this file by path).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

B = 2000
SEED = 20261011
EDGE = 0.0005
BONF_LEVEL = 99.6                        # Tier 1's §12.13.4 level (13 primary buckets; `unblind_t1.BONF_LEVEL`)
GATE_MIN_N = 150
CONVERGED_MARGIN = 0.01

#: the stage axes: (field, buckets in EARLIEST-GAME-FIRST order). k and turn ascend; material descends (12 alive is
#: the earliest).
AXES: dict[str, tuple[str, tuple[str, ...]]] = {
    "k": ("k_bin", ("1-2", "3", "4", "5", "6")),
    "turn": ("turn_bin", ("<=7", "8-14", "15-21", ">=22")),
    "material": ("mat_bin", ("12", "10-11", "8-9", "<=7")),
}


class Boot:
    """The battle-clustered bootstrap of Tier 1 (`t1_stats.Boot`, unstratified): ``C[r, b]`` = how often battle b is
    drawn in replicate r."""

    def __init__(self, battles: Sequence[str]):
        self.ub = sorted(set(battles))
        ix = {b: i for i, b in enumerate(self.ub)}
        self.bi = np.array([ix[b] for b in battles])
        nb = len(self.ub)
        rng = np.random.default_rng(SEED)
        self.C = np.zeros((B, nb))
        draws = rng.integers(0, nb, size=(B, nb))
        for r in range(B):
            np.add.at(self.C[r], draws[r], 1.0)
        self.nb = nb

    def mean(self, x: npt.ArrayLike, sel: npt.ArrayLike | None = None) -> tuple[float, np.ndarray]:
        """(estimate, per-draw vector) of the mean of x over sel (non-finite entries dropped)."""
        x = np.asarray(x, dtype=np.float64)
        m = np.ones(len(x), bool) if sel is None else np.asarray(sel, bool).copy()
        m &= np.isfinite(x)
        S = np.bincount(self.bi[m], weights=x[m], minlength=self.nb)
        N = np.bincount(self.bi[m], minlength=self.nb).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = (self.C @ S) / (self.C @ N)
        est = float(S.sum() / N.sum()) if N.sum() > 0 else float("nan")
        return est, vec


def ci(vec: npt.ArrayLike, level: float = 95.0) -> list[float | None]:
    v = np.asarray(vec, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return [None, None]
    a = (100.0 - level) / 2.0
    lo, hi = np.percentile(v, [a, 100.0 - a])
    return [float(lo), float(hi)]


def _margin(c: float) -> float:
    return 0.0 if abs(abs(c) - EDGE) < 1e-12 else EDGE


def gt(x: float | None, c: float) -> bool:
    """x > c, CLEAR of the boundary (rule 8)."""
    return x is not None and bool(np.isfinite(x)) and x > c + _margin(c)


def lt(x: float | None, c: float) -> bool:
    return x is not None and bool(np.isfinite(x)) and x < c - _margin(c)


def inside(lohi: Sequence[float | None], m: float) -> bool:
    lo, hi = lohi
    return lo is not None and hi is not None and lo > -m + EDGE and hi < m - EDGE


def summary(est: float, vec: npt.ArrayLike, n: int | None = None, level: float = 95.0) -> dict:
    lo, hi = ci(vec, level)
    out: dict = {"mean": None if not np.isfinite(est) else round(float(est), 5),
                 "ci": [None if lo is None else round(lo, 5), None if hi is None else round(hi, 5)]}
    if n is not None:
        out["n"] = int(n)
    return out


# ------------------------------------------------------------------------------------------------
# TODAY: the p-equivalent of FN on the M8 curve
# ------------------------------------------------------------------------------------------------

def eq_p(curve: npt.ArrayLike, target: float, grid: npt.ArrayLike) -> float | None:
    """The p at which the curve first reaches ``target`` (linear interpolation); 0 at or below the p = 0 value; None
    above the p = 1 value (Tier 1's `unblind_t1.eq_p`)."""
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


def today_clamped(curve_est: npt.ArrayLike, target_est: float, curve_draws: npt.ArrayLike, target_draws: npt.ArrayLike,
                  grid: npt.ArrayLike) -> tuple[float | None, np.ndarray]:
    """(TODAY's point, its per-draw vector) with every value clamped to [0, 1] (None above the curve -> 1.0; a draw
    with a non-finite input -> NaN, dropped by the interval)."""
    def one(c: npt.ArrayLike, t: float) -> float:
        if not np.isfinite(t) or not np.all(np.isfinite(c)):
            return float("nan")
        p = eq_p(c, t, grid)
        return 1.0 if p is None else min(1.0, max(0.0, p))

    pt = one(np.asarray(curve_est), target_est)
    vec = np.array([one(c, t) for c, t in zip(np.asarray(curve_draws), np.asarray(target_draws))])
    return (None if not np.isfinite(pt) else pt), vec


# ------------------------------------------------------------------------------------------------
# the per-checkpoint GATE (R-STAGE-SEARCH on one network)
# ------------------------------------------------------------------------------------------------

def gate_status(n: int, t4_minus_crit: np.ndarray, t4: np.ndarray, n_min: int = GATE_MIN_N) -> str:
    """GATE-OPEN / GATE-WEAK / GATE-CLOSED from the per-draw vectors of [TRUNC-4-K8 − CRITIC] and [TRUNC-4-K8]."""
    bonf = gt(ci(t4_minus_crit, BONF_LEVEL)[0], EDGE) and gt(ci(t4, BONF_LEVEL)[0], EDGE)
    p95 = gt(ci(t4_minus_crit)[0], EDGE) and gt(ci(t4)[0], EDGE)
    if n >= n_min and bonf:
        return "GATE-OPEN"
    return "GATE-WEAK" if p95 else "GATE-CLOSED"


def earliest_open(gates: Mapping[str, str], axis: str) -> int:
    """The index (earliest-game first) of the earliest GATE-OPEN bucket on ``axis``; len(buckets) = none open."""
    buckets = AXES[axis][1]
    for i, b in enumerate(buckets):
        if gates.get(b) == "GATE-OPEN":
            return i
    return len(buckets)


def _bucket_name(buckets: Sequence[str], i: int) -> str:
    return buckets[i] if i < len(buckets) else "none"


def frontier_change(first: Mapping[str, Mapping[str, str]], last: Mapping[str, Mapping[str, str]]) -> dict:
    """{axis: {first, last, moved}} and the verdict (see the module docstring)."""
    per, earlier, later = {}, [], []
    for ax, (_f, buckets) in AXES.items():
        e0, e1 = earliest_open(first.get(ax, {}), ax), earliest_open(last.get(ax, {}), ax)
        per[ax] = {"first": _bucket_name(buckets, e0), "last": _bucket_name(buckets, e1),
                   "moved": "earlier" if e1 < e0 else "later" if e1 > e0 else "static"}
        if e1 < e0:
            earlier.append(ax)
        elif e1 > e0:
            later.append(ax)
    v = ("FRONTIER-MIXED" if earlier and later else "FRONTIER-MOVES-EARLIER" if earlier
         else "FRONTIER-RECEDES" if later else "FRONTIER-STATIC")
    return {"axes": per, "verdict": v}


# ------------------------------------------------------------------------------------------------
# R-MATURITY over the series
# ------------------------------------------------------------------------------------------------

@dataclass
class CheckpointRead:
    """One anchor's read on the frozen bank. Every vector is a per-draw vector of the SAME :class:`Boot`.

    ``g``: {cell: (estimate, vector)} of G_F = IMP_O − IMP_F at LEAF-K8 ("all" and "<axis>:<bucket>");
    ``today``: (point, clamped vector) or None when FN is not read; ``gates``: {axis: {bucket: status}}."""

    label: str
    step: int
    g: dict[str, tuple[float, np.ndarray]]
    today: tuple[float | None, np.ndarray] | None = None
    gates: dict[str, dict[str, str]] = field(default_factory=dict)


def gap_change(first: tuple[float, np.ndarray], last: tuple[float, np.ndarray]) -> dict:
    d_est = last[0] - first[0]
    d_vec = np.asarray(last[1]) - np.asarray(first[1])
    lo, hi = ci(d_vec)
    v = "GAP-SHRINKS" if lt(hi, -EDGE) else "GAP-GROWS" if gt(lo, EDGE) else "NOT DETECTED"
    return {"change": summary(d_est, d_vec), "verdict": v}


def belief_change(first: tuple[float | None, np.ndarray], last: tuple[float | None, np.ndarray]) -> dict:
    d_vec = np.asarray(last[1]) - np.asarray(first[1])
    d_est = (last[0] - first[0]) if (last[0] is not None and first[0] is not None) else float("nan")
    lo, _hi = ci(d_vec)
    return {"change": summary(d_est, d_vec), "verdict": "BELIEF-RISES" if gt(lo, EDGE) else "NOT DETECTED"}


def converged(prev: tuple[float, np.ndarray], last: tuple[float, np.ndarray]) -> dict:
    d_vec = np.asarray(last[1]) - np.asarray(prev[1])
    iv = ci(d_vec)
    return {"change": summary(last[0] - prev[0], d_vec), "verdict": "CONVERGED" if inside(iv, CONVERGED_MARGIN)
            else "NOT CONVERGED"}


def r_maturity(reads: Sequence[CheckpointRead]) -> dict:
    """R-MATURITY over the anchors (sorted by step here; at least two)."""
    rs = sorted(reads, key=lambda r: r.step)
    if len(rs) < 2:
        raise ValueError("R-MATURITY needs at least two anchors")
    if len({r.step for r in rs}) != len(rs):
        raise ValueError("two reads at the same step")
    first, last, prev = rs[0], rs[-1], rs[-2]
    cells = sorted(set(first.g) & set(last.g))
    out: dict = {"first": first.label, "last": last.label, "anchors": [r.label for r in rs],
                 "GAP": {c: gap_change(first.g[c], last.g[c]) for c in cells}}
    out["BELIEF"] = (belief_change(first.today, last.today) if first.today is not None and last.today is not None
                     else {"verdict": "NOT READ"})
    out["FRONTIER"] = (frontier_change(first.gates, last.gates) if first.gates and last.gates
                       else {"verdict": "NOT READ"})
    out["CONVERGED"] = (converged(prev.g["all"], last.g["all"]) if "all" in prev.g and "all" in last.g
                        else {"verdict": "NOT READ"})
    out["CONVERGED"]["between"] = [prev.label, last.label]
    return out
