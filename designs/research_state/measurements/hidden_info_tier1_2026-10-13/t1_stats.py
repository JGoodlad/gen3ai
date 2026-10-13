"""TIER 1's statistics: the battle-clustered bootstrap and the rule-8 comparisons (registration §12.3, §12.10, §12.13.4).

* The bootstrap: 2,000 draws, ``numpy.random.default_rng(20261011)``, percentile intervals, every contrast PAIRED (all
  quantities of one read are evaluated on the same draws). Per checkpoint: battles resampled with replacement. POOLED
  over the checkpoints: battles resampled WITHIN each checkpoint (§6.3: "the bootstrap resampling battles within each
  checkpoint"), states weighted equally.
* A mean over a selection is the ratio of per-draw sums (states weighted equally); a share is a ratio of means on the
  same draws (§6.3: "ratios of means on the same bootstrap draws").
* RULE 8 (§12): "a bound within ±0.0005 of its boundary is a BOUNDARY result and counts as not clear". A comparison with
  a boundary of exactly ±0.0005 already carries the margin (the registration writes the zero edge as "> +0.0005"); every
  other boundary c is clear only beyond c ± 0.0005. "Inside ±m" means lower > −m + 0.0005 and upper < m − 0.0005.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

B = 2000
SEED = 20261011
EDGE = 0.0005


class Boot:
    def __init__(self, battles: Sequence[str], strata: Optional[Sequence[str]] = None):
        self.ub = sorted(set(battles))
        ix = {b: i for i, b in enumerate(self.ub)}
        self.bi = np.array([ix[b] for b in battles])
        nb = len(self.ub)
        rng = np.random.default_rng(SEED)
        self.C = np.zeros((B, nb))
        if strata is None:
            draws = rng.integers(0, nb, size=(B, nb))
            for r in range(B):
                np.add.at(self.C[r], draws[r], 1.0)
        else:
            st_of_b = {}
            for b, s in zip(battles, strata):
                if st_of_b.setdefault(b, s) != s:
                    raise ValueError(f"battle {b} in two strata")
            groups = [np.array([ix[b] for b in self.ub if st_of_b[b] == s]) for s in sorted(set(strata))]
            for r in range(B):
                for g in groups:
                    np.add.at(self.C[r], g[rng.integers(0, len(g), size=len(g))], 1.0)
        self.nb = nb

    def _sum(self, x, m):
        return np.bincount(self.bi[m], weights=np.asarray(x, dtype=np.float64)[m], minlength=self.nb)

    def mean(self, x, sel=None, weights=None):
        """(estimate, per-draw vector) of the (weighted) mean of x over sel."""
        x = np.asarray(x, dtype=np.float64)
        m = np.ones(len(x), bool) if sel is None else np.asarray(sel, bool).copy()
        m &= np.isfinite(x)
        wt = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
        S = np.bincount(self.bi[m], weights=(x * wt)[m], minlength=self.nb)
        N = np.bincount(self.bi[m], weights=wt[m], minlength=self.nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = (self.C @ S) / (self.C @ N)
        est = float(S.sum() / N.sum()) if N.sum() > 0 else float("nan")
        return est, vec

    def ratio(self, num, den, sel=None):
        """(Σ num / Σ den, per-draw vector) over sel (a ratio of means: the common n cancels)."""
        num = np.asarray(num, dtype=np.float64)
        den = np.asarray(den, dtype=np.float64)
        m = np.ones(len(num), bool) if sel is None else np.asarray(sel, bool).copy()
        m &= np.isfinite(num) & np.isfinite(den)
        Sn, Sd = self._sum(num, m), self._sum(den, m)
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = (self.C @ Sn) / (self.C @ Sd)
        est = float(Sn.sum() / Sd.sum()) if Sd.sum() != 0 else float("nan")
        return est, vec

    def ols(self, y, Z, sel=None):
        """OLS (intercept first) of y on the columns of Z; per-draw coefficient matrix [B', p]."""
        y = np.asarray(y, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        m = np.ones(len(y), bool) if sel is None else np.asarray(sel, bool).copy()
        m &= np.isfinite(y) & np.isfinite(Z).all(1)
        X = np.column_stack([np.ones(m.sum()), Z[m]])
        yy = y[m]
        bi = self.bi[m]
        p = X.shape[1]
        XtX = np.zeros((self.nb, p, p))
        Xty = np.zeros((self.nb, p))
        np.add.at(XtX, bi, X[:, :, None] * X[:, None, :])
        np.add.at(Xty, bi, X * yy[:, None])
        beta = np.linalg.solve(XtX.sum(0), Xty.sum(0))
        A = (self.C @ XtX.reshape(self.nb, p * p)).reshape(-1, p, p)
        bv = self.C @ Xty
        ok = np.array([np.linalg.matrix_rank(a) == p for a in A])
        bs = np.full((len(A), p), np.nan)
        bs[ok] = np.linalg.solve(A[ok], bv[ok][:, :, None])[:, :, 0]
        return beta, bs


def ci(vec, level: float = 95.0):
    v = np.asarray(vec, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return [None, None]
    a = (100.0 - level) / 2.0
    lo, hi = np.percentile(v, [a, 100.0 - a])
    return [float(lo), float(hi)]


def summary(est, vec, n=None, level: float = 95.0) -> dict:
    lo, hi = ci(vec, level)
    out = {"mean": None if not np.isfinite(est) else round(float(est), 5),
           "ci": [None if lo is None else round(lo, 5), None if hi is None else round(hi, 5)]}
    if n is not None:
        out["n"] = int(n)
    return out


# ------------------------------------------------------------------------------------------------
# rule-8 comparisons
# ------------------------------------------------------------------------------------------------

def _margin(c: float) -> float:
    return 0.0 if abs(abs(c) - EDGE) < 1e-12 else EDGE


def gt(x: Optional[float], c: float) -> bool:
    """x > c, CLEAR of the boundary (rule 8)."""
    return x is not None and np.isfinite(x) and x > c + _margin(c)


def lt(x: Optional[float], c: float) -> bool:
    return x is not None and np.isfinite(x) and x < c - _margin(c)


def ge(x: Optional[float], c: float) -> bool:
    """x ≥ c, clear of the boundary: a value within 0.0005 of c is a BOUNDARY result (not clear)."""
    return x is not None and np.isfinite(x) and x >= c + _margin(c)


def inside(lohi: List[Optional[float]], m: float) -> bool:
    lo, hi = lohi
    return lo is not None and hi is not None and lo > -m + EDGE and hi < m - EDGE


def boundary(x: Optional[float], c: float) -> bool:
    return x is not None and np.isfinite(x) and abs(x - c) <= EDGE and _margin(c) > 0
