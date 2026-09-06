"""stats — the training package's SHARED small-sample statistics: pure NumPy in, floats out.

Every function here is stateless and carries no domain concept: no labels, no battles, no
checkpoints, no filesystem, no torch, no RNG except an explicitly seeded bootstrap. That is the
admission rule for this module — a helper that needs to know what a *decision* or a *bias map* is
belongs beside the instrument that owns the concept, not here.

Extracted from ``cf_audit.py`` (2026-09-06) as the file-size ratchet's first cut of the 1,000–2,000
band; every function keeps the docstring it was written with, and each notes where it came from.
The arithmetic is unchanged — ``cf_audit_test.py`` pins the whole audit's JSON-serialised readouts
against a golden captured BEFORE the move.

What lives here
---------------
* :func:`wilson_ci` — the small-N binomial interval.
* :func:`spearman` — rank correlation, ``None`` when undefined.
* :func:`cluster_bootstrap_ci` / :func:`cluster_bootstrap_diff_ci` — percentile CIs that resample
  the unit of independence (a battle), for a mean and for a difference of means.
* :func:`sd_true_excess` — the within-cell spread net of the binomial sampling floor, and
  the ``MIN_CELL_N`` / ``MIN_SUBCELL_N`` sample-size floors below which it is not reported.

Related statistics elsewhere in this package, and why they are NOT merged here
-----------------------------------------------------------------------------
Three near-siblings exist and each one differs in a way that is load-bearing rather than
accidental, so folding them together would silently change a shipped instrument's output:

* ``scaffolding.py`` — ``spearman_rho`` and its own ``cluster_bootstrap_ci``. Both use the **NaN**
  refusal convention (TensorBoard drops NaN, so a degenerate slice leaves a GAP in the live
  ``train/scaffolding_gauge`` curve), where this module returns ``None`` for a JSON report. Its
  bootstrap is also strictly more general — it resamples ROW INDICES and evaluates an arbitrary
  ``stat_fn``, which is what lets ``reliability_table`` compose with it; the pair here is
  specialised to a mean and a difference of means. Two conventions, two return types, two
  audiences.
* ``winprob_finetune.label_noise_variance`` — the same ``p̂(1−p̂)/(n−1)`` identity
  :func:`sd_true_excess` subtracts, but PER ROW with a heterogeneous ``n`` per row, and it excludes
  ``n < 2`` rows rather than refusing the whole cell.
* ``main/q_amortization.spearman`` — the same shape as :func:`spearman` with an exact
  ``std() == 0`` flatness test instead of the relative-tolerance ``_is_flat`` here. Its call site
  should move to this module, but that is a behaviour change (a *near*-flat row starts refusing
  instead of reporting float noise) and belongs in its own pass with its own evidence.

``hodge.py`` and ``elo.py`` carry no general-purpose statistics — everything in them is
Bradley-Terry or Hodge-decomposition machinery bound to the rating model.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Binomial intervals
# ---------------------------------------------------------------------------

def wilson_ci(wins: float, n: int, z: float = 1.96) -> "tuple[float, float]":
    """Wilson score interval — the right small-N binomial CI (a normal approximation gives
    the degenerate [0, 0] at 0 wins). ``(0.0, 1.0)`` for n == 0.

    ``wins`` may be FRACTIONAL: `cf_producer` scores a draw-at-the-turn-cap 0.5, so its success
    total is a sum over ``{0, 0.5, 1}``. The arithmetic is unchanged and well-defined, but such a
    sample is not Bernoulli, so the interval becomes an approximation that errs NARROW — which is
    why the label row carries `n_capped` beside it.

    *Moved from ``cf_audit.py``.*
    """
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------

def _ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared) — the transform that turns Pearson into Spearman.

    *Moved from ``cf_audit.py``.*
    """
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    # average tied ranks, so a head that emits one constant width scores 0 rather than an artifact
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def _is_flat(a: np.ndarray, rtol: float = 1e-9) -> bool:
    """Constant to a RELATIVE tolerance — :func:`spearman`'s degeneracy test.

    *Moved from ``cf_audit.py``.*
    """
    return bool(np.ptp(a) <= rtol * (float(np.abs(a).max()) + 1e-30))


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation, or ``None`` when it is undefined (n < 3, or either side flat).

    Hand-rolled rather than `scipy.stats.spearmanr` so the audit's headline does not acquire a
    dependency the training package does not otherwise have. A CONSTANT input returns None, not 0:
    "the head claims the same width everywhere" is a different finding from "the widths it claims
    are unrelated to the blur", and collapsing them would hide the more damning one.

    *Moved from ``cf_audit.py``.*
    """
    a, b = np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)
    if len(a) < 3 or len(a) != len(b):
        return None
    # FLAT to a relative tolerance, not `std() == 0`. A constant that arrives through a weighted
    # average is constant to ~1 ulp, not exactly — and an exact test there lets a genuinely flat
    # width fall through to `corrcoef`, which divides by ~1e-17 and reports a confident correlation
    # of pure float noise.
    if _is_flat(a) or _is_flat(b):
        return None
    ra, rb = _ranks(a), _ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# Cluster bootstrap — resample the unit of independence, never the state
# ---------------------------------------------------------------------------

def cluster_bootstrap_ci(values: Sequence[float], clusters: Sequence[str], *,
                         draws: int = 2000, seed: int = 7,
                         ) -> "tuple[Optional[float], Optional[float]]":
    """95% CI by resampling CLUSTERS (battles) with replacement, not states.

    Decisions inside one battle share a board, a team matchup and a dice stream, so a
    state-level CI understates the width by however much that correlation is worth. This is
    the same discipline the pooled-correlation Simpson trap taught: resample the unit of
    independence, which here is the battle.

    *Moved from ``cf_audit.py``.*
    """
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    by_c: "Dict[str, List[float]]" = defaultdict(list)
    for v, c in zip(values, clusters):
        by_c[c].append(float(v))
    keys = list(by_c)
    if len(keys) < 2:
        return None, None
    pools = [np.asarray(by_c[k]) for k in keys]
    means = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, len(pools), size=len(pools))
        means[i] = float(np.concatenate([pools[j] for j in idx]).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _cluster_pools(values: Sequence[float], clusters: Sequence[str]) -> "Optional[List[np.ndarray]]":
    """Group ``values`` by cluster key; None when there is nothing to resample (< 2 clusters).

    *Moved from ``cf_audit.py``.*
    """
    by_c: "Dict[str, List[float]]" = defaultdict(list)
    for v, c in zip(values, clusters):
        by_c[c].append(float(v))
    if len(by_c) < 2:
        return None
    return [np.asarray(by_c[k]) for k in by_c]


def cluster_bootstrap_diff_ci(values_a: Sequence[float], clusters_a: Sequence[str],
                              values_b: Sequence[float], clusters_b: Sequence[str], *,
                              draws: int = 2000, seed: int = 7,
                              ) -> "tuple[Optional[float], Optional[float], Optional[float]]":
    """``mean(a) − mean(b)`` and its 95% cluster-bootstrap CI — returned TOGETHER, on purpose.

    Returns ``(point, lo, hi)``. The two groups' clusters are resampled INDEPENDENTLY (they are
    disjoint at every call site: a battle has one outcome, so a decision is in the loss arm or the
    win arm, never both), and the bootstrap statistic is the SAME functional as ``point`` — which
    is what makes the interval bracket the estimate instead of describing a different quantity.

    ⚠️ **THE DEFECT THIS EXISTS TO PREVENT.** The conviction-class readout used to get its point
    estimate from ``mean(a) − mean(b)`` while getting its CI from
    ``cluster_bootstrap_ci(a + [−x for x in b], …)`` — the mean of the CONCATENATION, which is
    ``(Σa − Σb) / (n_a + n_b)``, a SIZE-WEIGHTED pooled mean and not a difference of means at all.
    The two agree only when ``n_a == n_b``; at the real 3:1 imbalance they came apart badly enough
    that the published interval did not contain its own point estimate (+0.205 vs [+0.070,
    +0.158]). A CI that does not bracket its estimate is not a wide CI, it is a different
    statistic — and it reads as a precise result.

    ``(None, None, None)`` when either arm has fewer than 2 clusters (nothing to resample), matching
    :func:`cluster_bootstrap_ci`'s refusal convention.

    *Moved from ``cf_audit.py``.*
    """
    if len(values_a) < 1 or len(values_b) < 1:
        return None, None, None
    point = float(np.mean(np.asarray(values_a, dtype=float))
                  - np.mean(np.asarray(values_b, dtype=float)))
    pools_a, pools_b = _cluster_pools(values_a, clusters_a), _cluster_pools(values_b, clusters_b)
    if pools_a is None or pools_b is None:
        return point, None, None
    rng = np.random.default_rng(seed)
    stat = np.empty(draws)
    for i in range(draws):
        ia = rng.integers(0, len(pools_a), size=len(pools_a))
        ib = rng.integers(0, len(pools_b), size=len(pools_b))
        stat[i] = (float(np.concatenate([pools_a[j] for j in ia]).mean())
                   - float(np.concatenate([pools_b[j] for j in ib]).mean()))
    return point, float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


# ---------------------------------------------------------------------------
# Spread net of the binomial sampling floor
# ---------------------------------------------------------------------------

#: The sample-size FLOORS that guard :func:`sd_true_excess` from reporting its own noise: a cell
#: needs `MIN_CELL_N` observations (and `MIN_SUBCELL_N` per sub-cell) before its spread is worth
#: printing. They live here rather than beside one reader because `cf_audit.resolution_cells` and
#: `cf_audit_twin.twin_resolution_read` bin different things and must refuse at the SAME n — two
#: copies of a floor is two thresholds that drift. They are the one pair of constants in this
#: module, admitted for the same reason as the estimator they guard: sample size is not a domain
#: concept, and neither of them knows what a label or a decile is.
MIN_CELL_N, MIN_SUBCELL_N = 12, 3


def sd_true_excess(mc: Sequence[float], n_rollouts: int,
                   weights: "Optional[Sequence[float]]" = None) -> dict:
    """THE PRIMARY METER — the within-cell spread of the TRUE win probability, net of the
    R-rollout binomial noise floor.

    ``Var(MC) = Var(true p) + E[sampling var]``, and for an R-rollout mean the sampling
    variance is ``p(1−p)/R``. The plug-in ``E[p̂(1−p̂)]/(R−1)`` is EXACTLY unbiased for it
    (``E[p̂(1−p̂)] = (1−1/R)·p(1−p)``), so the subtraction leaves the real heterogeneity and
    nothing else. Clamped at 0 — a negative estimate means the cell's spread is at or below
    the floor, i.e. no resolvable structure, which is what 0 says.

    ``weights`` recombines sub-cells (e.g. win/loss) at their POPULATION shares rather than
    at this probe's deliberately balanced sampling shares:
    ``Var = Σ w_o (Var_o + (m_o − m)²)``. Pass one weight per element.

    *Moved from ``cf_audit.py``, where it is the counterfactual audit's PRIMARY meter.*
    """
    a = np.asarray(list(mc), dtype=float)
    n = len(a)
    if n < 3 or n_rollouts < 2:
        return {"n": int(n), "mean": float(a.mean()) if n else None, "sd_observed": None,
                "sd_binomial_floor": None, "sd_true_excess": None, "frac_variance_real": None}
    w = (np.ones(n) if weights is None else np.asarray(list(weights), dtype=float))
    w = w / w.sum()
    mean = float((w * a).sum())
    # Unbiased weighted variance (reliability weights): divide by (1 - Σw²).
    denom = 1.0 - float((w ** 2).sum())
    var = float((w * (a - mean) ** 2).sum() / denom) if denom > 0 else float(a.var(ddof=1))
    binom = float((w * (a * (1 - a) / (n_rollouts - 1))).sum())
    excess = max(0.0, var - binom)
    return {
        "n": int(n),
        "mean": mean,
        "sd_observed": math.sqrt(max(0.0, var)),
        "sd_binomial_floor": math.sqrt(max(0.0, binom)),
        "sd_true_excess": math.sqrt(excess),
        "frac_variance_real": (excess / var) if var > 0 else None,
    }
