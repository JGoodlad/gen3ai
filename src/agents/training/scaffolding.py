"""The SCAFFOLDING GAUGE — how far the shaped critic and the win-prob head still disagree.

Registered as an instrument by the value-function foundations ruling (ledger 2026-08-29). The two
value readouts this tree carries answer two DIFFERENT questions and neither is a repair of the
other:

* ``V`` — the critic. Estimates the **shaped** return, in PopArt-normalized units, discounted at
  ``gamma``. Definitionally correct for its job: GAE advantages must be estimated in the units of
  the reward stream actually being optimized.
* ``P(win)`` — the win-prob head. Estimates the **game** value: outcome units, no discount
  distortion, no PopArt drift.

The GAP between them is the reward scaffolding — the part of ``V`` that is shaping rather than
game. As a generation matures the two should order states more and more alike, and the gauge's
TRAJECTORY is the registered signal for when shaping coefficients can begin annealing toward the
pure game.

═══ 🚨 UNITS HONESTY — read before quoting any number from this module ════════════════════════

**A direct unit conversion between ``V`` and ``P(win)`` is not generally possible.** ``V`` is a
PopArt-normalized *shaped* return: its scale is set by a normalizer that moves over training, its
composition by whichever reward terms are enabled, and its horizon by ``gamma``. ``P(win)`` is a
probability. There is no fixed affine map between them, and under PBRS with a good potential there
is not even a monotone one to recover — the classic φ=V* result drives ``V_shaped`` toward a
CONSTANT, all evaluative content having migrated into the reward stream (ledger db9bb5c). So this
module ships **two** gauges and labels each with exactly what it can and cannot claim:

┌─ ``rank_gauge`` — Spearman ρ between V and P(win) over a slice of states ─────────────────────┐
│ CAN claim: whether the two readouts ORDER states alike. Unit-free, PopArt-proof, always valid, │
│   invariant to any monotone reparameterization of either axis.                                │
│ CANNOT claim: anything about magnitude, calibration, or "how many win-percent" the gap is. A   │
│   ρ of 1.0 is compatible with the two heads disagreeing wildly about every absolute level.     │
│   It is also blind by construction to the constancy endpoint: as V_shaped flattens, ρ          │
│   degenerates into noise — so a FALLING ρ late in a shaped run is ambiguous between "the heads  │
│   diverged" and "V ran out of variance to rank with". Read it beside `constancy_row`.          │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

┌─ ``affine_gauge`` — fit a per-checkpoint affine V→outcome map on OUTCOMES, then compare ───────┐
│ CAN claim: how far the best LINEAR outcome-readout of V sits from the win-prob head, in        │
│   probability units, on this slice. The map is fit on realized win/loss labels, so both sides  │
│   of the comparison are in outcome units by construction rather than by assumption.            │
│ CANNOT claim: that V and P(win) are natively commensurable — the map is a fit, refit per        │
│   checkpoint, and it is NOT transportable to another checkpoint or another run. Nor that the   │
│   whole residual is disagreement: the affine FAMILY is a restriction (the true V→outcome map   │
│   is sigmoid-shaped at best), so part of every residual is the readout being a worse outcome   │
│   predictor rather than the heads disagreeing. That part is separated and reported as          │
│   ``readout_penalty`` (= Brier(affine readout) − Brier(head)); a large ``rms`` with a large     │
│   ``readout_penalty`` is a bad readout, not a divergence finding.                              │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

**Outcome labels are per-BATTLE and broadcast to every state of that battle**, so states are
heavily clustered. Every confidence interval here is a CLUSTER bootstrap over battles, never an
i.i.d. one over states — the pooled-correlation Simpson trap is a recorded failure in this tree
and an i.i.d. interval on this data would be a fabricated tightness of roughly sqrt(states/battle).

Everything in this module is PURE — numpy in, floats out. No torch, no RNG except the explicitly
seeded bootstrap, no filesystem. It is shared by the offline CLI (`python -m main.scaffolding_gauge`)
and the live `train/scaffolding_gauge` scalar so the two can never drift apart.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Below this the axis carries no ordering information and a rank correlation is 0/0. Reported as
#: NaN rather than a fabricated 0.0 — TensorBoard drops NaN points, so a degenerate slice leaves a
#: GAP in the curve, which is the honest rendering (the `signal/` group's convention).
_DEGENERATE_STD = 1e-12

#: Minimum states before a rank correlation means anything. Three is the smallest n at which
#: Spearman is not forced to ±1 by the sample size alone.
_MIN_RANK_N = 3


# ══════════════════════════════════════════════════════════════ rank statistics (unit-free) ══

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks, 1-based. A local implementation so this module needs no scipy — the
    tie handling is load-bearing (a slice where many states share a win-prob is routine)."""
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    # Average the ranks inside each run of equal values.
    srt = a[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and srt[j] == srt[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def spearman_rho(x, y) -> float:
    """Spearman rank correlation. NaN when undefined rather than 0.0.

    Undefined means: fewer than `_MIN_RANK_N` finite pairs, or either axis constant (every rank
    tied ⇒ zero rank variance ⇒ 0/0). A constant axis is NOT correlation zero; reporting it as
    zero would read as "the heads disagree completely", which is the opposite of the truth.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"spearman_rho: length mismatch {a.size} vs {b.size}")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < _MIN_RANK_N:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    da, db = float(np.sqrt(np.mean(ra * ra))), float(np.sqrt(np.mean(rb * rb)))
    if da <= _DEGENERATE_STD or db <= _DEGENERATE_STD:
        return float("nan")
    return float(np.mean(ra * rb) / (da * db))


def rank_gauge(values, win_probs) -> Dict[str, float]:
    """The UNIT-FREE gauge: ``{"rho", "gauge", "n"}``.

    ``gauge = (1 − ρ) / 2`` ∈ [0, 1] — 0 when the two readouts order states identically (no
    scaffolding divergence visible in the ordering), 0.5 at independence, 1 at perfect inversion.
    The affine rescale of ρ exists so the published scalar SHRINKS as the run matures, matching
    every other divergence meter in the tree; ``rho`` is carried alongside so nothing is hidden by
    the transform. Both are NaN on a degenerate slice, never 0.

    Claims: ORDERING only. See the module docstring — this number says nothing about magnitude,
    and it goes ambiguous exactly where V_shaped flattens.
    """
    a = np.asarray(values, dtype=np.float64).ravel()
    b = np.asarray(win_probs, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"rank_gauge: length mismatch {a.size} vs {b.size}")
    ok = np.isfinite(a) & np.isfinite(b)
    rho = spearman_rho(a[ok], b[ok])
    return {"rho": rho,
            "gauge": float("nan") if not np.isfinite(rho) else (1.0 - rho) / 2.0,
            "n": float(int(ok.sum()))}


# ═══════════════════════════════════════════════════ the calibrated-affine gauge (outcome units) ══

def _affine_fit(v: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Least-squares ``a·v + b ≈ y``, pinned to ``(0, ȳ)`` when V carries no variance.

    The pin is explicit rather than left to `lstsq`: on a rank-deficient design the minimum-norm
    solution splits the intercept between the two columns (a constant V=2.5 with ȳ=0.6 comes back
    as a=0.207, b=0.083), which predicts the identical q but publishes a SLOPE that reads as "V
    informs the outcome" when it cannot. The fitted values agree; the reported coefficients would
    lie, and this is exactly the PBRS constancy endpoint where a reader is looking at them.
    """
    if float(np.std(v)) <= _DEGENERATE_STD:
        return 0.0, float(np.mean(y))
    design = np.column_stack([v, np.ones_like(v)])
    sol, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(sol[0]), float(sol[1])


def affine_gauge(values, win_probs, outcomes, *, clip: bool = True) -> Dict[str, float]:
    """The CALIBRATED-AFFINE gauge, in probability units.

    Fits ``q = clip(a·V + b, 0, 1)`` against the realized per-state outcome labels ``y ∈ {0,1}``,
    then reports how far that V-implied outcome sits from the win-prob head ``p``.

    Returned keys:

    ``rms``               √mean (q − p)²  — the headline divergence, in probability units.
    ``bias``              mean (q − p)    — signed: positive = V-implied reads MORE optimistic.
    ``mad``               mean |q − p|    — the outlier-robust companion.
    ``a`` / ``b``         the fitted map. NOT transportable off this slice.
    ``brier_head``        mean (p − y)²   — the win-prob head's own Brier score.
    ``brier_v_affine``    mean (q − y)²   — the V-implied readout's Brier score.
    ``brier_base``        mean (ȳ − y)²   — the constant base-rate reference.
    ``readout_penalty``   brier_v_affine − brier_head. **The disclaimer, as a number.** The affine
                          family cannot express the true V→outcome map, so some of ``rms`` is the
                          readout simply being worse rather than the heads disagreeing. When this
                          is a large fraction of ``rms``, ``rms`` is not a divergence finding.
    ``base_rate``         ȳ on this slice — a reminder that a lopsided slice inflates everything.
    ``v_constant``        1.0 when V carried no variance (the PBRS constancy endpoint), so a reader
                          knows ``a`` was pinned to 0 and ``q`` is the base rate.
    ``n``                 finite rows used.

    Degenerate input (no finite rows) returns every statistic as NaN — never a fabricated 0.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    p = np.asarray(win_probs, dtype=np.float64).ravel()
    y = np.asarray(outcomes, dtype=np.float64).ravel()
    if not (v.size == p.size == y.size):
        raise ValueError(f"affine_gauge: length mismatch {v.size}/{p.size}/{y.size}")
    ok = np.isfinite(v) & np.isfinite(p) & np.isfinite(y)
    v, p, y = v[ok], p[ok], y[ok]
    nan = float("nan")
    if v.size == 0:
        return {k: nan for k in ("rms", "bias", "mad", "a", "b", "brier_head", "brier_v_affine",
                                 "brier_base", "readout_penalty", "base_rate", "v_constant",
                                 "n")} | {"n": 0.0}
    v_const = float(np.std(v)) <= _DEGENERATE_STD
    a, b = _affine_fit(v, y)
    q = a * v + b
    if clip:
        q = np.clip(q, 0.0, 1.0)
    diff = q - p
    ybar = float(y.mean())
    brier_head = float(np.mean((p - y) ** 2))
    brier_q = float(np.mean((q - y) ** 2))
    return {
        "rms": float(np.sqrt(np.mean(diff ** 2))),
        "bias": float(np.mean(diff)),
        "mad": float(np.mean(np.abs(diff))),
        "a": a, "b": b,
        "brier_head": brier_head,
        "brier_v_affine": brier_q,
        "brier_base": float(np.mean((ybar - y) ** 2)),
        "readout_penalty": brier_q - brier_head,
        "base_rate": ybar,
        "v_constant": 1.0 if v_const else 0.0,
        "n": float(v.size),
    }


# ══════════════════════════════════════════════ the CONSTANCY sanity row (ledger db9bb5c) ══

def constancy_row(values, groups: Optional[Sequence] = None) -> Dict[str, float]:
    """The db9bb5c prediction as a one-line check: does ``V_shaped`` flatten?

    Under PBRS with a good frozen potential φ, all evaluative content migrates into the reward
    stream and the shaped critic is driven toward a CONSTANT. That is a checkable prediction, and
    this is the cheap row a frozen-φ arm's battery quotes. The corresponding meter is V's SPREAD
    across states, not its level (a constant is free — the critic baseline absorbs it).

    ``groups`` (one label per state, typically the battle id) turns on the within/between
    decomposition, which is what separates the two ways V can look flat:

    * ``within_frac`` → 1 means V varies inside a battle but not between battles — the critic is
      tracking turn-by-turn shaping payments only.
    * ``within_frac`` → 0 means V is a per-battle constant — the critic has become a matchup
      lookup, which is the FAILURE mode that a raw ``v_std`` alone cannot tell from the theory's
      prediction.

    🚨 UNITS: ``v_std`` rides PopArt, whose σ moves over training, so the RAW value compares within
    a run and only cautiously across runs. ``dispersion`` (= v_std / E|V|) is the scale-free
    companion for a cross-run read; the STRONG form of the check is arm-vs-control at matched step.
    """
    raw = np.asarray(values, dtype=np.float64).ravel()
    finite = np.isfinite(raw)
    v = raw[finite]
    nan = float("nan")
    if v.size == 0:
        return {"n": 0.0, "n_groups": 0.0, "v_mean": nan, "v_std": nan, "v_abs_mean": nan,
                "v_iqr": nan, "dispersion": nan, "within_std": nan, "between_std": nan,
                "within_frac": nan}
    std = float(np.std(v))
    abs_mean = float(np.mean(np.abs(v)))
    q75, q25 = np.percentile(v, [75.0, 25.0])
    out = {"n": float(v.size), "n_groups": 0.0,
           "v_mean": float(np.mean(v)), "v_std": std, "v_abs_mean": abs_mean,
           "v_iqr": float(q75 - q25),
           "dispersion": std / abs_mean if abs_mean > _DEGENERATE_STD else nan,
           "within_std": nan, "between_std": nan, "within_frac": nan}
    if groups is None:
        return out
    g = np.asarray(groups).ravel()
    if g.size != raw.size:
        raise ValueError("constancy_row: `groups` must be one label per value")
    g = g[finite]
    uniq = np.unique(g)
    out["n_groups"] = float(uniq.size)
    means, within_ss, within_n = [], 0.0, 0
    for u in uniq:
        vals = v[g == u]
        if vals.size == 0:
            continue
        means.append(float(np.mean(vals)))
        within_ss += float(np.sum((vals - np.mean(vals)) ** 2))
        within_n += vals.size
    if within_n:
        within_var = within_ss / within_n
        out["within_std"] = float(np.sqrt(within_var))
        out["between_std"] = float(np.std(means)) if means else nan
        total_var = float(np.var(v))
        out["within_frac"] = within_var / total_var if total_var > _DEGENERATE_STD else nan
    return out


# ══════════════════════════════════════════════════════════ cluster bootstrap over battles ══

def cluster_bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    groups: Sequence,
    *,
    n_boot: int = 400,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile CI for ``stat_fn(row_indices)``, resampling BATTLES with replacement.

    ``stat_fn`` receives an int array of row indices (a battle appearing twice contributes its rows
    twice) and returns one float. Non-finite draws are dropped rather than poisoning the quantiles;
    fewer than 20 usable draws returns ``(nan, nan)`` — a CI computed from a handful of degenerate
    resamples is worse than no CI.

    The cluster is the BATTLE and not the state, deliberately: outcome labels are per-battle and
    broadcast, so an i.i.d. bootstrap over states would report an interval roughly
    ``sqrt(states-per-battle)`` times too tight. That error has cost this tree a finding before.
    """
    g = np.asarray(groups).ravel()
    uniq, inverse = np.unique(g, return_inverse=True)
    if uniq.size < 2:
        return float("nan"), float("nan")
    by_group: List[np.ndarray] = [np.nonzero(inverse == k)[0] for k in range(uniq.size)]
    rng = np.random.default_rng(seed)
    draws: List[float] = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, uniq.size, size=uniq.size)
        idx = np.concatenate([by_group[k] for k in pick])
        try:
            val = float(stat_fn(idx))
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            continue
        if np.isfinite(val):
            draws.append(val)
    if len(draws) < 20:
        return float("nan"), float("nan")
    lo, hi = np.percentile(draws, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    return float(lo), float(hi)


# ══════════════════════════════════════════════════════════════════════ the slice fold ══

def gauge_slice(
    values,
    win_probs,
    outcomes,
    battles,
    *,
    n_boot: int = 400,
    seed: int = 0,
) -> Dict[str, object]:
    """Both gauges + the constancy row over one step-slice, with cluster-bootstrap CIs.

    ``battles`` is one label per state (the trace path or short id). The four arrays are parallel.
    Returns a JSON-ready dict: ``{"rank": {...}, "affine": {...}, "constancy": {...}}``, each gauge
    block carrying ``ci_lo`` / ``ci_hi`` for its headline statistic (``gauge`` and ``rms``).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    p = np.asarray(win_probs, dtype=np.float64).ravel()
    y = np.asarray(outcomes, dtype=np.float64).ravel()
    b = np.asarray(battles).ravel()
    if not (v.size == p.size == y.size == b.size):
        raise ValueError(
            f"gauge_slice: parallel arrays required, got {v.size}/{p.size}/{y.size}/{b.size}")

    rank = rank_gauge(v, p)
    affine = affine_gauge(v, p, y)
    const = constancy_row(v, groups=b)

    rank["ci_lo"], rank["ci_hi"] = cluster_bootstrap_ci(
        lambda idx: rank_gauge(v[idx], p[idx])["gauge"], b, n_boot=n_boot, seed=seed)
    affine["ci_lo"], affine["ci_hi"] = cluster_bootstrap_ci(
        lambda idx: affine_gauge(v[idx], p[idx], y[idx])["rms"], b, n_boot=n_boot, seed=seed + 1)
    return {"rank": rank, "affine": affine, "constancy": const}


# ══════════════════════════════════════════════════════════════════════ the live scalar ══

#: Below this many paired rows, IDENTICAL orderings are a COINCIDENCE, not an identity, and
#: `_same_ordering` refuses to conclude anything from them. Two unrelated readouts agree on the
#: order of n distinct values with probability 1/n!: that is 1-in-6 at n = 3 and 1-in-2 at n = 2,
#: but ~5e-14 at 16 — beyond any plausible accident, while a real rollout's paired read is ~1e5
#: rows (`train/scaffolding_n` reads 131072 on `models/ai_v12_01_winprob_critic`). So the floor
#: costs the production path nothing and keeps the gate from silencing a genuine tiny sample.
_MIN_SAME_ORDERING_N = 16


def _same_ordering(a: np.ndarray, b: np.ndarray) -> bool:
    """Do these two arrays induce the IDENTICAL ranking (ties included)?

    `gen3_tb_relevance_v1`. True ⟺ one is an exact monotone (non-decreasing) transform of the
    other over at least `_MIN_SAME_ORDERING_N` rows — the condition under which a rank correlation
    between them is 1.0 by construction and therefore content-free. Uses the same average-rank
    convention the Spearman path does, so the answer agrees with "ρ would be exactly 1.0" rather
    than approximating it, and it is a claim about SAMENESS of the two readouts, which is why the
    sample floor is part of it rather than a separate guard.
    """
    if a.size < _MIN_SAME_ORDERING_N:
        return False
    return bool(np.array_equal(_rankdata(a), _rankdata(b)))


def live_gauge_metrics(values, win_prob_logits) -> Dict[str, float]:
    """`train/scaffolding_*` from one rollout's paired (V, win-prob-logit) reads.

    The RANK form only — deliberately. The live path has no realized outcome labels at hand for
    the states it is scoring (the win-prob target is a delayed MC label that only some rows carry),
    so the calibrated-affine gauge is an OFFLINE instrument by construction; publishing a live
    number labelled as if it were the calibrated one would be the worse error.

    The logit is passed through unconverted: the sigmoid is monotone, so ρ over logits equals ρ
    over probabilities exactly, and skipping it avoids saturating float32 ranks at ±1.

    Returns keys WITHOUT the ``train/`` prefix (the caller adds it — the `signal/` idiom), and an
    EMPTY dict when there is nothing to publish (no head, no rows), so a run without the win-prob
    head writes no curve at all rather than a flat zero.

    🚨 **AND an empty dict when the two readouts are THE SAME QUANTITY** (`gen3_tb_relevance_v1`).
    Under ``--critic winprob`` the deployed value IS ``sigmoid(win_prob_logit)``, so ρ is 1.0 and
    the gauge is 0 *by construction* — a tautology, not a measurement, and it published one for
    every rollout of the first win-prob arm (``train/scaffolding_rho`` a flat 1.0,
    ``train/scaffolding_gauge`` a flat 5.5e-13). The test is the RANK VECTORS being identical,
    which is exactly "V is an exact monotone map of the logit"; over a rollout's ~10⁵ paired rows
    two genuinely different readouts do not produce it, and if they somehow did, the gauge would
    have nothing to say either.
    """
    if values is None or win_prob_logits is None:
        return {}
    v = np.asarray(values, dtype=np.float64).ravel()
    z = np.asarray(win_prob_logits, dtype=np.float64).ravel()
    if v.size == 0 or v.size != z.size:
        return {}
    ok = np.isfinite(v) & np.isfinite(z)
    n = int(ok.sum())
    if n == 0:
        return {}
    if _same_ordering(v[ok], z[ok]):
        return {}
    g = rank_gauge(v[ok], z[ok])
    return {"scaffolding_gauge": g["gauge"], "scaffolding_rho": g["rho"],
            "scaffolding_n": float(n)}


# ═══════════════════════════════ the RELIABILITY table (calibration of P(win) vs outcome) ══

#: Default bin count for the reliability table: ten equal-width bins over [0, 1]. EQUAL-WIDTH and
#: not equal-mass on purpose — "when the head says 0.85, does it happen 85% of the time?" is a
#: question about the FORECAST axis, and equal-mass bins move their own boundaries between arms,
#: so two arms' tables would stop being comparable row by row.
_DEFAULT_RELIABILITY_BINS = 10


def reliability_table(win_probs, outcomes, *, bins: int = _DEFAULT_RELIABILITY_BINS,
                      weights=None) -> Dict[str, object]:
    """The reliability curve + Brier / ECE / Murphy decomposition of ``P(win)`` against outcome.

    This is the CALIBRATION half of the pair this module opened with. ``affine_gauge`` asks how far
    the two READOUTS sit apart; this asks how far the win-prob head sits from the TRUTH, and it is
    the one that yields a bar a future critic must beat.

    Returned keys:

    ``brier``            mean (p − y)² — the head's Brier score. LOWER is better.
    ``brier_base``       ȳ(1 − ȳ) — the constant base-rate forecaster's Brier on this slice.
    ``skill``            1 − brier/brier_base, the Brier SKILL SCORE. **Zero means the head is no
                         better than always predicting the slice's base rate; NEGATIVE means it is
                         worse.** This is the headline, and it is not implied by the rank gauge: a
                         forecaster can ORDER states well (high Spearman) and still score below
                         zero here, because ordering is invariant to the level errors this
                         penalizes.
    ``ece``              Σ (n_k/N)·|p̄_k − ȳ_k| — the count-weighted expected calibration error.
    ``mce``              max_k |p̄_k − ȳ_k| over non-empty bins — the worst bin.
    ``reliability``      Σ (n_k/N)(p̄_k − ȳ_k)²  — Murphy's REL. LOWER is better.
    ``resolution``       Σ (n_k/N)(ȳ_k − ȳ)²    — Murphy's RES. **HIGHER is better**: the head's
                         ability to SEPARATE outcomes, and the component
                         ``designs/learning/win_prob_decomposition.md`` names as the disease. A
                         base-rate forecaster scores a perfect 0 reliability and a useless 0
                         resolution, which is exactly why REL alone is never the meter.
    ``uncertainty``      ȳ(1 − ȳ) — Murphy's UNC, a property of the SLICE, not of the forecaster.
    ``decomp_residual``  brier − (reliability − resolution + uncertainty). The identity is EXACT
                         only when states are grouped by distinct forecast value; binning makes it
                         approximate, and the size of the approximation is reported rather than
                         hidden. A residual comparable to ``reliability`` means the bins are too
                         coarse to trust the split.
    ``base_rate``        ȳ on this slice.
    ``n`` / ``n_bins``   finite rows used / bins that carried at least one row.
    ``table``            per-bin rows ``{lo, hi, n, p_mean, y_rate, gap}``, ``gap`` = p̄_k − ȳ_k
                         (POSITIVE = the head was OPTIMISTIC in that bin). Empty bins are KEPT with
                         ``n = 0`` and NaN statistics so two slices' tables align row by row.

    Degenerate input (no finite rows) returns every statistic as NaN and an empty table — never a
    fabricated 0. A slice whose outcomes are all one class returns ``brier_base = 0`` and therefore
    ``skill = NaN``: a skill score against a zero-variance reference is undefined, not infinite.

    ⚠️ **The labels are per-BATTLE and broadcast**, so every row of a battle carries that battle's
    outcome. Nothing here is an i.i.d. sample; pair it with ``cluster_bootstrap_ci`` over battles
    for any interval, and read ``main.scaffolding_gauge``'s footer on the eval quota, which selects
    the battles this table is computed over and is not a random sample of play.

    🚨 **``weights`` is the answer to that selection, and on this tree's eval traces it is not
    optional.** The recorder's quota keeps a NON-REPRESENTATIVE win/loss mix — measured on
    ``ai_v9_59_R2ACTION_0827`` the captured slice's outcome rate is 0.46 while the run's own
    ``eval_results.jsonl`` records 0.90 against bots — so an UNWEIGHTED table over captured traces
    scores the head against a loss-enriched population and reports a large positive ECE that is
    mostly the quota. Passing per-row importance weights (true-outcome share / captured-outcome
    share, within an opponent) restores the eval population. Every statistic above becomes its
    weighted analogue and ``ess`` — Kish's effective sample size, ``(Σw)² / Σw²`` — is returned
    beside ``n`` so a reader can see how much of the sample the reweighting spent. An ``ess`` far
    below ``n`` means the correction is carried by a handful of battles and the interval, not the
    point estimate, is the honest output.
    """
    p = np.asarray(win_probs, dtype=np.float64).ravel()
    y = np.asarray(outcomes, dtype=np.float64).ravel()
    if p.size != y.size:
        raise ValueError(f"reliability_table: length mismatch {p.size} vs {y.size}")
    if weights is None:
        w_all = np.ones(p.size, dtype=np.float64)
    else:
        w_all = np.asarray(weights, dtype=np.float64).ravel()
        if w_all.size != p.size:
            raise ValueError(f"reliability_table: weights length {w_all.size} vs {p.size}")
        if np.any(w_all < 0.0):
            raise ValueError("reliability_table: negative weights are not a reweighting")
    nb = int(bins)
    if nb < 1:
        raise ValueError(f"reliability_table: bins must be >= 1, got {bins}")
    ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(w_all)
    p, y, w_all = p[ok], y[ok], w_all[ok]
    nan = float("nan")
    edges = np.linspace(0.0, 1.0, nb + 1)
    wsum = float(w_all.sum())
    if p.size == 0 or wsum <= _DEGENERATE_STD:
        return {k: nan for k in ("brier", "brier_base", "skill", "ece", "mce", "reliability",
                                 "resolution", "uncertainty", "decomp_residual", "base_rate",
                                 "ess")} | {"n": int(p.size), "n_bins": 0, "table": []}

    ybar = float((w_all * y).sum() / wsum)
    brier = float((w_all * (p - y) ** 2).sum() / wsum)
    unc = ybar * (1.0 - ybar)
    # digitize against the INTERIOR edges: p == 1.0 then lands in the last bin rather than a
    # phantom one above it, and p == 0.0 in the first. Clipping is belt-and-braces for -0.0.
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, nb - 1)

    table: List[Dict[str, float]] = []
    rel = res = ece = 0.0
    mce = nan
    n_used = 0
    for k in range(nb):
        sel = idx == k
        nk = int(sel.sum())
        wk = float(w_all[sel].sum())
        row: Dict[str, float] = {"lo": float(edges[k]), "hi": float(edges[k + 1]), "n": nk,
                                 "p_mean": nan, "y_rate": nan, "gap": nan}
        if nk and wk > _DEGENERATE_STD:
            pk = float((w_all[sel] * p[sel]).sum() / wk)
            yk = float((w_all[sel] * y[sel]).sum() / wk)
            gap = pk - yk
            row.update({"p_mean": pk, "y_rate": yk, "gap": gap})
            w = wk / wsum
            rel += w * gap * gap
            res += w * (yk - ybar) ** 2
            ece += w * abs(gap)
            mce = abs(gap) if (not np.isfinite(mce) or abs(gap) > mce) else mce
            n_used += 1
        table.append(row)

    return {
        "brier": brier,
        "brier_base": float(unc),
        "skill": float(1.0 - brier / unc) if unc > _DEGENERATE_STD else nan,
        "ece": float(ece),
        "mce": float(mce),
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "decomp_residual": float(brier - (rel - res + unc)),
        "base_rate": ybar,
        "n": int(p.size),
        "ess": float(wsum * wsum / float((w_all * w_all).sum())),
        "n_bins": n_used,
        "table": table,
    }
