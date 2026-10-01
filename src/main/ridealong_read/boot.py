"""The BATTLE-CLUSTERED BOOTSTRAP for rank statistics (AUROC, Spearman, the top/bottom-decile error
ratio) — B = 1000 percentile resamples, fixed seed, the X4 pre-read's method.

A resample draws battles with replacement; a row's MULTIPLICITY is its battle's draw count. Rather
than materialising the replicated arrays (1000 × ~19k rows × every cell), each statistic is computed
from the rows' WEIGHTS over one precomputed sort: the weighted mid-rank of a tie group occupying
positions ``before + 1 … before + w`` is ``before + (w + 1) / 2`` — exactly the average rank the
replicated copies would get — so the AUROC (Mann–Whitney on those ranks) and the Spearman ρ (the
weighted Pearson of the two mid-rank vectors) EQUAL the replicated computation. The decile ratio places
each row at the midpoint of its cumulative weight (a copy group straddling a decile edge lands on one
side), an approximation that touches only the edge rows; its point estimate is the exact unweighted one.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import numpy as np

N_BOOT = 1000
BOOT_SEED = 20260930
CHUNK = 50
METHOD = {"method": "percentile bootstrap resampling BATTLES (clusters)", "B": N_BOOT,
          "seed": BOOT_SEED, "interval": "95 % (2.5th, 97.5th percentiles)"}


def weight_chunks(clusters: Any, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                  chunk: int = CHUNK) -> Iterator[np.ndarray]:
    """``[b, N]`` float row weights (the battle's draw count), ``n_boot`` rows in chunks."""
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    for s in range(0, n_boot, chunk):
        counts = np.stack([np.bincount(r, minlength=len(uniq)) for r in draws[s:s + chunk]])
        yield counts.astype(np.float64)[:, inv]


class SortedMidranks:
    """One sort of ``x``; weighted mid-ranks for any weight matrix."""

    def __init__(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.order = np.argsort(x, kind="stable")
        xs = x[self.order]
        new = np.r_[True, xs[1:] != xs[:-1]] if len(xs) else np.zeros(0, bool)
        self.starts = np.flatnonzero(new)
        self.group = np.cumsum(new) - 1

    def __call__(self, W: np.ndarray) -> np.ndarray:
        Ws = W[:, self.order]
        g = np.add.reduceat(Ws, self.starts, axis=1)
        before = np.cumsum(g, axis=1) - g
        out = np.empty_like(W)
        out[:, self.order] = (before + (g + 1.0) / 2.0)[:, self.group]
        return out


def _auroc_w(r: np.ndarray, lab: np.ndarray, W: np.ndarray) -> np.ndarray:
    n_pos = (W * lab).sum(1)
    n_neg = (W * ~lab).sum(1)
    u = (W * lab * r).sum(1) - n_pos * (n_pos + 1) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((n_pos > 0) & (n_neg > 0), u / (n_pos * n_neg), np.nan)


def _pearson_w(a: np.ndarray, b: np.ndarray, W: np.ndarray) -> np.ndarray:
    s = W.sum(1, keepdims=True)
    ma = (W * a).sum(1, keepdims=True) / s
    mb = (W * b).sum(1, keepdims=True) / s
    da, db = a - ma, b - mb
    cov = (W * da * db).sum(1)
    den = np.sqrt((W * da * da).sum(1) * (W * db * db).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, cov / den, np.nan)


def _decile_ratio_w(order: np.ndarray, err: np.ndarray, W: np.ndarray) -> np.ndarray:
    Ws = W[:, order]
    es = err[order][None, :]
    c = np.cumsum(Ws, axis=1)
    tot = c[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.clip(np.floor((c - Ws / 2.0) / tot * 10.0), 0, 9)
        top = (Ws * es * (dec == 9)).sum(1) / (Ws * (dec == 9)).sum(1)
        bot = (Ws * es * (dec == 0)).sum(1) / (Ws * (dec == 0)).sum(1)
        return np.asarray(top / bot, dtype=np.float64)


def boot_rank_stats(score: np.ndarray, clusters: Any, *, label: Optional[np.ndarray] = None,
                    err: Optional[np.ndarray] = None, n_boot: int = N_BOOT,
                    seed: int = BOOT_SEED) -> Dict[str, np.ndarray]:
    """The resampled distributions ``{"auroc", "spearman", "decile_ratio"}`` (each ``[n_boot]``, NaN
    where a resample is degenerate): AUROC of ``score`` for ``label``; Spearman of ``score`` vs
    ``err`` and the mean ``err`` in the top score decile ÷ the bottom one."""
    score = np.asarray(score, dtype=np.float64)
    rs = SortedMidranks(score)
    re_ = SortedMidranks(err) if err is not None else None
    out: Dict[str, List[np.ndarray]] = {}
    for W in weight_chunks(clusters, n_boot, seed):
        r = rs(W)
        if label is not None:
            out.setdefault("auroc", []).append(_auroc_w(r, np.asarray(label, bool)[None, :], W))
        if err is not None and re_ is not None:
            e = np.asarray(err, dtype=np.float64)
            out.setdefault("spearman", []).append(_pearson_w(r, re_(W), W))
            out.setdefault("decile_ratio", []).append(_decile_ratio_w(rs.order, e, W))
    return {k: np.concatenate(v) for k, v in out.items()}


def ci(dist: Optional[np.ndarray], min_valid: float = 0.5) -> Optional[List[float]]:
    """95 % percentile interval, or None when fewer than ``min_valid`` of the resamples are defined."""
    if dist is None or len(dist) == 0:
        return None
    ok = dist[np.isfinite(dist)]
    if len(ok) < min_valid * len(dist):
        return None
    lo, hi = np.percentile(ok, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def n_clusters(clusters: Any) -> int:
    return int(len(np.unique(np.asarray(clusters)))) if len(clusters) else 0


__all__ = ["N_BOOT", "BOOT_SEED", "METHOD", "weight_chunks", "SortedMidranks", "boot_rank_stats",
           "ci", "n_clusters"]
