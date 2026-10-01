"""The RND VARIANTS' pre-registered comparisons (X26 amendment; `gen3_ridealong_rnd_variants_read_v1`).

Pure NumPy over per-row columns the reader already computed. Nothing here loads a model. The torch
half (the identification probes' errors, the predictor-reset floor, feat's drift scoring) is in
:mod:`main.ridealong_read.variants_probe`.

KEYS: ``rnd`` = base (the unchanged ``--ridealong-rnd`` head, the REFERENCE), ``rndv_<name>`` = each
declared variant (``agents.model.ridealong_heads.RND_VARIANT_DECLS``).

COMMON MACHINERY. Every interval is the battle-clustered percentile bootstrap of
:mod:`main.ridealong_read.boot` (B = 1000, seed 20260930). A PAIRED difference is computed on the
SAME resamples for both scores: ``boot.weight_chunks`` yields identical row weights for the same
clusters and seed, so every key (and every checkpoint, which shares the bank's rows) is evaluated
on one draw per resample. Bootstrap p-values:

* two-sided  p = min(1, 2 · min(#(d ≤ 0) + 1, #(d ≥ 0) + 1) / (B + 1));
* one-sided  p(θ > θ0) = (#(d ≤ θ0) + 1) / (B + 1), and the mirror for θ < θ0,

over the B resamples where the statistic is defined. Holm step-down at family α = 0.05; raw and
Holm-adjusted p are both reported, and an untestable member counts in the family as p = 1 (the
family size is the pre-registered one, never shrunk by a degenerate read).

(a) V-error beyond V's own uncertainty — each variant's within-V-entropy-quintile AUROC (R1's
    primary meter) minus base's, paired; Holm across the variants.
(b) Coverage — the AUROC of each key's novelty for exploiter rows vs the rest, and for late-game
    rows (turn > the bank's own 95th percentile) vs the rest, ALL rows (draws included); paired
    differences vs base, Holm across variants × 2 contrasts.
(c) Saturation — the raw error's median, IQR and rel_spread = IQR / median per checkpoint; the
    ratio r of rel_spread between two checkpoints, paired resamples, Holm across the 5 keys.
(d) feat's drift — checkpoint A's feat head scored through every later B's value_pooled.
(e) Identification — does a predictor learn the frozen target EVERYWHERE, or only on the states it
    visits? Errors on fixed real probe rows (i) and on their ``chimera_v1`` recombinations (ii),
    each against the same heads with the predictors reset to their declared init (the floor).

Quantiles are the weighted INVERTED-CDF quantile (numpy ``method="inverted_cdf"`` at unit weights:
the smallest value whose cumulative weight reaches q of the total), point and resample alike.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from main.ridealong_read.boot import BOOT_SEED, N_BOOT, SortedMidranks, _auroc_w, ci, weight_chunks

SCHEMA = "gen3_ridealong_rnd_variants_read_v1"
ALPHA = 0.05
BASE = "rnd"
#: (e)'s probe set: rows drawn from the bank with this seed, in a seeded shuffled order.
N_PROBE = 4096
PROBE_SEED = 20260930
CHIMERA_GENERATOR = "chimera_v1"
#: (b2)'s late-game contrast: turn > this quantile of the read's own turn stamps.
LATE_TURN_Q = 0.95
#: (c): SATURATED needs the Holm rejection with r < 1 AND a point r at or below this.
SAT_R_MAX = 0.5
#: (e): the transfer-index split (0 = learns only what it visits, 1 = identifies the target) and
#: the on-distribution learning floor below which nothing is judged.
S_SPLIT = 0.5
F_I_INDETERMINATE = 0.7
#: (d): the offline read's predicted feature-RND inflation range (rnd_input_choice, 2026-09-30).
OFFLINE_DRIFT_PREDICTION = (2.5, 8.0)
QUANTILE_METHOD = "inverted_cdf (weighted; numpy method='inverted_cdf' at unit weights)"


def variant_key(name: str) -> str:
    return f"rndv_{name}"


def base_arch() -> str:
    from agents.model.arch_constants import RIDEALONG_RND_HIDDEN as H
    from agents.model.arch_constants import RIDEALONG_RND_OUT as O

    return (f"obs→{H}→{H}→{O} predictor vs obs→{H}→{O} target, same ReLU-MLP family, predictor one "
            "layer deeper")


def declarations(names: Sequence[str]) -> Dict[str, dict]:
    """Per key: input, lr_mult, half-life, the predictor-vs-target architecture and why."""
    from agents.model.ridealong_heads import RND_VARIANT_BY_NAME

    out: Dict[str, dict] = {BASE: {"input": "obs", "lr_mult": 1.0, "decay_half_life_updates": None,
                                   "predictor_vs_target": base_arch(),
                                   "why": "the REFERENCE: the unchanged --ridealong-rnd head"}}
    for n in names:
        d = RND_VARIANT_BY_NAME[n]
        out[variant_key(n)] = {"input": d.input, "lr_mult": d.lr_mult,
                               "decay_half_life_updates": d.decay_half_life_updates,
                               "predictor_vs_target": d.predictor_vs_target, "why": d.why}
    return out


def _g(x: Any, nd: int = 6) -> Optional[float]:
    """A JSON float with ``nd`` significant digits, or None (not finite / absent)."""
    if x is None:
        return None
    x = float(x)
    return float(f"{x:.{nd}g}") if math.isfinite(x) else None


def _ci(dist: Optional[np.ndarray]) -> Optional[List[float]]:
    out = ci(dist)
    return None if out is None else [float(v) for v in out]


# ---------------------------------------------------------------------------------------------
# p-values and Holm
# ---------------------------------------------------------------------------------------------

def _finite(d: Any) -> np.ndarray:
    a = np.asarray(d, dtype=np.float64).reshape(-1)
    return a[np.isfinite(a)]


def p_two_sided(d: Any, null: float = 0.0) -> Optional[float]:
    """Two-sided bootstrap p of H0: θ = ``null`` from the resampled θ's (B = the defined ones)."""
    a = _finite(d) - null
    if len(a) == 0:
        return None
    return min(1.0, 2.0 * min(int((a <= 0).sum()) + 1, int((a >= 0).sum()) + 1) / (len(a) + 1))


def p_greater(d: Any, null: float = 0.0) -> Optional[float]:
    """One-sided p of H0: θ ≤ ``null`` (the alternative θ > ``null``)."""
    a = _finite(d) - null
    return None if len(a) == 0 else (int((a <= 0).sum()) + 1) / (len(a) + 1)


def p_less(d: Any, null: float = 0.0) -> Optional[float]:
    """One-sided p of H0: θ ≥ ``null`` (the alternative θ < ``null``)."""
    a = _finite(d) - null
    return None if len(a) == 0 else (int((a >= 0).sum()) + 1) / (len(a) + 1)


def holm(pvals: Mapping[str, Optional[float]], alpha: float = ALPHA) -> Dict[str, dict]:
    """Holm step-down over the family ``pvals`` (insertion order breaks ties). An untestable member
    (None) enters as p = 1 so the family keeps its registered size. Returns per member
    ``{"p", "p_holm", "reject"}``: the adjusted p is the running max of (m − i)·p_(i), capped at 1."""
    keys = list(pvals)
    m = len(keys)
    eff = [1.0 if pvals[k] is None else float(pvals[k]) for k in keys]   # type: ignore[arg-type]
    order = sorted(range(m), key=lambda i: (eff[i], i))
    out: Dict[str, dict] = {}
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * eff[i]))
        out[keys[i]] = {"p": _g(pvals[keys[i]]), "p_holm": _g(run), "reject": bool(run <= alpha)}
    return {k: out[k] for k in keys}


# ---------------------------------------------------------------------------------------------
# paired resampling
# ---------------------------------------------------------------------------------------------

def paired_boot(clusters: Any, fn: Callable[[np.ndarray], np.ndarray], n_boot: int = N_BOOT,
                seed: int = BOOT_SEED) -> np.ndarray:
    """``fn(W) -> [b, k]`` over every chunk of resample weights → ``[n_boot, k]``. Every statistic
    computed through one call sees the SAME resamples (and so does any later call with the same
    clusters and seed — that is what pairs checkpoints)."""
    return np.concatenate([np.asarray(fn(W), dtype=np.float64)
                           for W in weight_chunks(clusters, n_boot, seed)], axis=0)


class SortedQuantiles:
    """One sort of ``x``; weighted inverted-CDF quantiles for any weight matrix."""

    def __init__(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.order = np.argsort(x, kind="stable")
        self.xs = x[self.order]

    def __call__(self, W: np.ndarray, qs: Sequence[float]) -> np.ndarray:
        """``[b, len(qs)]``: per weight row, the smallest value whose cumulative weight ≥ q·total."""
        c = np.cumsum(W[:, self.order], axis=1)
        tot = c[:, -1:]
        out = np.empty((W.shape[0], len(qs)))
        n = len(self.xs)
        for j, q in enumerate(qs):
            idx = (c < q * tot).sum(1)
            out[:, j] = self.xs[np.minimum(idx, n - 1)]
        return out

    def point(self, qs: Sequence[float]) -> np.ndarray:
        out: np.ndarray = self(np.ones((1, len(self.xs))), qs)[0]
        return out


def weighted_quantile(x: np.ndarray, W: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    out: np.ndarray = SortedQuantiles(x)(W, qs)
    return out


# ---------------------------------------------------------------------------------------------
# (a) V-error beyond V's own uncertainty
# ---------------------------------------------------------------------------------------------

def _cond_aurocs(scores: Sequence[np.ndarray], lab: np.ndarray, ref: np.ndarray,
                 bins: int) -> List[Optional[float]]:
    """``meters.conditional_auroc`` for several scores sharing one ``ref`` sort (identical algorithm)."""
    from agents.training.instrumented_ppo.ridealong_terms import rank_auroc

    if len(ref) < bins * 2:
        return [None] * len(scores)
    chunks = np.array_split(np.argsort(ref, kind="stable"), bins)
    out: List[Optional[float]] = []
    for s in scores:
        tot, w = 0.0, 0
        for ch in chunks:
            a = rank_auroc(s[ch], lab[ch])
            if a is not None:
                tot += a * len(ch)
                w += len(ch)
        out.append(tot / w if w else None)
    return out


def ranked_vs_base(point: Mapping[str, Optional[float]], dist: Mapping[str, np.ndarray],
                   base: str = BASE, alpha: float = ALPHA) -> dict:
    """Paired differences variant − base (point, CI, two-sided p), Holm across the variants, and
    the verdict per variant: BEATS_BASE / WORSE_THAN_BASE (Holm rejects, by the sign of the point
    difference) or NOT_DETECTED. ``best`` = the BEATS_BASE variant with the largest point difference,
    else "BASE STANDS"."""
    variants = [k for k in point if k != base]
    diffs: Dict[str, dict] = {}
    ps: Dict[str, Optional[float]] = {}
    for k in variants:
        d = dist[k] - dist[base]
        pt = (None if point[k] is None or point[base] is None
              else float(point[k]) - float(point[base]))     # type: ignore[arg-type]
        diffs[k] = {"diff": _g(pt, 5), "diff_ci": _ci(d)}
        ps[k] = p_two_sided(d) if pt is not None else None
    h = holm(ps, alpha)
    best, best_d = "BASE STANDS", -np.inf
    for k in variants:
        dk = diffs[k]["diff"]
        rej = h[k]["reject"] and dk is not None
        verdict = ("BEATS_BASE" if rej and dk > 0 else "WORSE_THAN_BASE" if rej and dk < 0
                   else "NOT_DETECTED")
        diffs[k].update({**h[k], "verdict": verdict})
        if verdict == "BEATS_BASE" and dk > best_d:
            best, best_d = k, dk
    ranking = sorted(point, key=lambda k: np.inf if point[k] is None else -float(point[k]))  # type: ignore[arg-type]
    return {"vs_base": diffs, "best": best,
            "ranking_descriptive": [{"key": k, "value": _g(point[k], 5)} for k in ranking]}


def compare_v_error(zs: Mapping[str, np.ndarray], err: np.ndarray, ref: np.ndarray, clusters: Any,
                    threshold: float = 0.5, bins: int = 5, n_boot: int = N_BOOT,
                    seed: int = BOOT_SEED) -> dict:
    """(a): within-``ref``-quintile AUROC of each key's z for ``err > threshold`` (the caller passes
    draw-excluded rows), every key on the SAME battle resamples (each re-cuts the quintiles)."""
    keys = list(zs)
    lab = np.asarray(err) > threshold
    ref = np.asarray(ref, dtype=np.float64)
    S = [np.asarray(zs[k], dtype=np.float64) for k in keys]
    point = dict(zip(keys, _cond_aurocs(S, lab, ref, bins)))
    n = len(lab)

    def fn(W: np.ndarray) -> np.ndarray:
        out = np.full((W.shape[0], len(keys)), np.nan)
        for b in range(W.shape[0]):
            idx = np.repeat(np.arange(n), W[b].astype(np.int64))
            vals = _cond_aurocs([s[idx] for s in S], lab[idx], ref[idx], bins)
            out[b] = [np.nan if v is None else v for v in vals]
        return out

    D = paired_boot(clusters, fn, n_boot, seed)
    dist = {k: D[:, j] for j, k in enumerate(keys)}
    res = ranked_vs_base(point, dist)
    return {"meter": "within-V-entropy-quintile AUROC of the key's z for |V - z| > 0.5 (draws "
                     "excluded) — R1's primary meter",
            "n": int(n), "n_err": int(lab.sum()),
            "auroc_within_ref_quintiles": {k: {"value": _g(point[k], 5), "ci": _ci(dist[k])}
                                           for k in keys},
            **res}


# ---------------------------------------------------------------------------------------------
# (b) coverage
# ---------------------------------------------------------------------------------------------

def late_game_mask(turns: Any, q: float = LATE_TURN_Q) -> Tuple[np.ndarray, float]:
    t = np.asarray(turns, dtype=np.float64)
    thr = float(np.percentile(t, 100.0 * q))
    return t > thr, thr


def compare_coverage(errs: Mapping[str, np.ndarray], contrasts: Mapping[str, np.ndarray],
                     clusters: Any, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                     base: str = BASE, alpha: float = ALPHA) -> dict:
    """(b): per contrast (a boolean positive class over ALL rows), the AUROC of every key's novelty,
    and the paired difference vs base. One Holm family over (variant × contrast). Per variant:
    COVERS_BETTER iff ≥ 1 contrast rejects positive and none negative; COVERS_WORSE the mirror;
    MIXED; NOT_DETECTED."""
    from agents.training.instrumented_ppo.ridealong_terms import rank_auroc

    keys = list(errs)
    variants = [k for k in keys if k != base]
    ranks = {k: SortedMidranks(np.asarray(errs[k], dtype=np.float64)) for k in keys}
    labs = {c: np.asarray(m, dtype=bool) for c, m in contrasts.items()}
    cnames = list(labs)

    def fn(W: np.ndarray) -> np.ndarray:
        cols = []
        for c in cnames:
            for k in keys:
                cols.append(_auroc_w(ranks[k](W), labs[c][None, :], W))
        return np.stack(cols, axis=1)

    D = paired_boot(clusters, fn, n_boot, seed)
    out: dict = {"contrasts": {}}
    ps: Dict[str, Optional[float]] = {}
    pts: Dict[str, Optional[float]] = {}
    for ci_, c in enumerate(cnames):
        lab = labs[c]
        blk: dict = {"n_pos": int(lab.sum()), "n_neg": int((~lab).sum()), "auroc": {}, "vs_base": {}}
        dist = {k: D[:, ci_ * len(keys) + j] for j, k in enumerate(keys)}
        pt = {k: rank_auroc(np.asarray(errs[k], dtype=np.float64), lab) for k in keys}
        for k in keys:
            blk["auroc"][k] = {"value": _g(pt[k], 5), "ci": _ci(dist[k])}
        for k in variants:
            fam = f"{k}|{c}"
            d = (None if pt[k] is None or pt[base] is None
                 else float(pt[k]) - float(pt[base]))     # type: ignore[arg-type]
            pts[fam] = d
            ps[fam] = p_two_sided(dist[k] - dist[base]) if d is not None else None
            blk["vs_base"][k] = {"diff": _g(d, 5), "diff_ci": _ci(dist[k] - dist[base])}
        out["contrasts"][c] = blk
    h = holm(ps, alpha)
    out["verdicts"] = {}
    for k in variants:
        pos = neg = False
        for c in cnames:
            hk = h[f"{k}|{c}"]
            out["contrasts"][c]["vs_base"][k].update(hk)
            d = pts[f"{k}|{c}"]
            pos |= bool(hk["reject"] and d is not None and d > 0)
            neg |= bool(hk["reject"] and d is not None and d < 0)
        out["verdicts"][k] = coverage_verdict(pos, neg)
    out["holm_family"] = f"{len(variants)} variants x {len(cnames)} contrasts = {len(ps)}"
    return out


def coverage_verdict(pos: bool, neg: bool) -> str:
    if pos and neg:
        return "MIXED"
    return "COVERS_BETTER" if pos else "COVERS_WORSE" if neg else "NOT_DETECTED"


# ---------------------------------------------------------------------------------------------
# (c) saturation
# ---------------------------------------------------------------------------------------------

def spread_point_and_dist(err: np.ndarray, clusters: Any, n_boot: int = N_BOOT,
                          seed: int = BOOT_SEED) -> Tuple[dict, np.ndarray]:
    """One key's raw error on every bank row: ``(json, rel_spread_dist [n_boot])`` — median, IQR and
    rel_spread = IQR / median, each with its battle-clustered CI."""
    sq = SortedQuantiles(np.asarray(err, dtype=np.float64))
    q25, q50, q75 = sq.point((0.25, 0.5, 0.75))
    D = paired_boot(clusters, lambda W: sq(W, (0.25, 0.5, 0.75)), n_boot, seed)
    med, iqr = D[:, 1], D[:, 2] - D[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(med > 0, iqr / med, np.nan)
    rel_pt = (q75 - q25) / q50 if q50 > 0 else None
    return ({"n": int(len(err)), "median": _g(q50), "median_ci": _ci(med), "iqr": _g(q75 - q25),
             "iqr_ci": _ci(iqr), "rel_spread": _g(rel_pt), "rel_spread_ci": _ci(rel)}, rel)


def saturation_compare(first: Mapping[str, Tuple[Optional[float], np.ndarray]],
                       last: Mapping[str, Tuple[Optional[float], np.ndarray]],
                       alpha: float = ALPHA) -> dict:
    """``first`` / ``last``: per key ``(rel_spread point, its resampled dist)`` from the SAME
    resamples. r = rel_spread(last) / rel_spread(first); two-sided p of log r; Holm across the keys.
    SATURATED iff Holm rejects with r < 1 AND point r ≤ :data:`SAT_R_MAX`. Guard: if base, fast AND
    decay are all SATURATED, the verdict reads STREAM_HOMOGENISED (the data, not a predictor)."""
    keys = [k for k in first if k in last]
    rows: Dict[str, dict] = {}
    ps: Dict[str, Optional[float]] = {}
    for k in keys:
        a, da = first[k]
        b, db = last[k]
        r = None if a is None or b is None or a <= 0 else float(b) / float(a)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(db / da)
        rows[k] = {"rel_spread_first": _g(a), "rel_spread_last": _g(b), "r": _g(r),
                   "r_ci": _ci(np.exp(lr))}
        ps[k] = p_two_sided(lr) if r is not None else None
    h = holm(ps, alpha)
    for k in keys:
        r = rows[k]["r"]
        sat = bool(h[k]["reject"] and r is not None and r < 1 and r <= SAT_R_MAX)
        rows[k].update({**h[k], "verdict_raw": "SATURATED" if sat else "NOT_DETECTED"})
    guard_keys = [BASE, variant_key("fast"), variant_key("decay")]
    homog = all(k in rows and rows[k]["verdict_raw"] == "SATURATED" for k in guard_keys)
    for k in keys:
        rows[k]["verdict"] = ("STREAM_HOMOGENISED" if homog and rows[k]["verdict_raw"] == "SATURATED"
                              else rows[k]["verdict_raw"])
    return {"keys": rows, "stream_homogenised": homog,
            "guard": "base, fast and decay all SATURATED ⇒ the stream homogenised (the data, not "
                     "a predictor): every SATURATED reads STREAM_HOMOGENISED",
            "rule": f"SATURATED iff Holm rejects log r = 0 with r < 1 and point r <= {SAT_R_MAX}"}


# ---------------------------------------------------------------------------------------------
# (d) feat's drift
# ---------------------------------------------------------------------------------------------

def drift_one(err_a: np.ndarray, err_b: np.ndarray, clusters: Any, n_boot: int = N_BOOT,
              seed: int = BOOT_SEED) -> dict:
    """inflation = median err through B ÷ median err through A on the same rows; its CI on paired
    resamples; ``frac_above_a_p90``; the one-sided p of H0: inflation ≤ 1."""
    ea, eb = np.asarray(err_a, dtype=np.float64), np.asarray(err_b, dtype=np.float64)
    sa, sb = SortedQuantiles(ea), SortedQuantiles(eb)
    p90 = float(sa.point((0.9,))[0])
    above = (eb > p90).astype(np.float64)

    def fn(W: np.ndarray) -> np.ndarray:
        return np.stack([sa(W, (0.5,))[:, 0], sb(W, (0.5,))[:, 0],
                         (W * above).sum(1) / W.sum(1)], axis=1)

    D = paired_boot(clusters, fn, n_boot, seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        infl = D[:, 1] / D[:, 0]
    ma, mb = float(sa.point((0.5,))[0]), float(sb.point((0.5,))[0])
    pt = mb / ma if ma > 0 else None
    return {"median_err_A": _g(ma), "median_err_B": _g(mb), "inflation": _g(pt),
            "inflation_ci": _ci(infl), "frac_above_a_p90": _g(float(above.mean())),
            "frac_above_a_p90_ci": _ci(D[:, 2]),
            "p_inflation_gt_1": _g(p_greater(np.log(infl)) if pt is not None else None)}


def drift_compare(err_a: np.ndarray, through_b: Mapping[str, Tuple[np.ndarray, np.ndarray]],
                  clusters: Any, alpha: float = ALPHA) -> dict:
    """(d): per later checkpoint B, ``(err via A's feat over B's pooled, the same with the feature
    statistics re-fitted on B)``. Holm across the pairs on the one-sided p; DRIFT_CONFIRMED iff
    rejected; whether the CI overlaps the offline prediction."""
    rows: Dict[str, dict] = {}
    for name, (eb, ebn) in through_b.items():
        rows[name] = {**drift_one(err_a, eb, clusters),
                      "renormalised": drift_one(err_a, ebn, clusters)}
    h = holm({k: v["p_inflation_gt_1"] for k, v in rows.items()}, alpha)
    lo_p, hi_p = OFFLINE_DRIFT_PREDICTION
    for k, v in rows.items():
        v["holm"] = h[k]
        v["verdict"] = "DRIFT_CONFIRMED" if h[k]["reject"] else "NOT_DETECTED"
        c = v["inflation_ci"]
        v["ci_overlaps_offline_prediction"] = (None if c is None
                                               else bool(c[0] <= hi_p and c[1] >= lo_p))
    return {"pairs": rows, "offline_prediction": list(OFFLINE_DRIFT_PREDICTION),
            "rule": "one-sided H0: inflation <= 1, Holm across the pairs; DRIFT_CONFIRMED iff rejected"}


# ---------------------------------------------------------------------------------------------
# (e) identification
# ---------------------------------------------------------------------------------------------

def probe_indices(n_rows: int, n_probe: int = N_PROBE, seed: int = PROBE_SEED) -> np.ndarray:
    """``min(n_probe, n_rows)`` distinct bank rows chosen with ``seed``, in a seeded SHUFFLED order."""
    rng = np.random.default_rng(seed)
    n = min(int(n_probe), int(n_rows))
    idx = np.sort(rng.choice(int(n_rows), size=n, replace=False))
    return idx[rng.permutation(n)]


def chimera_donors(n: int, edges: Sequence[int]) -> np.ndarray:
    """``[n_blocks, n]``: the row whose block k chimera row j copies (``block_chimera``'s rule)."""
    nb = len(edges) - 1
    stride = max(1, n // max(1, nb))
    j = np.arange(n)
    return np.stack([j if k == 0 else (j + k * stride) % max(1, n) for k in range(nb)])


def identification_arith(m_i: Any, m_ii: Any, m_i0: Any, m_ii0: Any) -> Dict[str, np.ndarray]:
    """R = m_ii / m_i, R0 = m_ii0 / m_i0, f_i = m_i / m_i0, f_ii = m_ii / m_ii0, ρ = R / R0 =
    f_ii / f_i and the transfer index S = log f_ii / log f_i (elementwise; NaN where undefined)."""
    a = [np.asarray(x, dtype=np.float64) for x in (m_i, m_ii, m_i0, m_ii0)]
    with np.errstate(divide="ignore", invalid="ignore"):
        R, R0 = a[1] / a[0], a[3] / a[2]
        f_i, f_ii = a[0] / a[2], a[1] / a[3]
        rho = f_ii / f_i
        S = np.log(f_ii) / np.log(f_i)
    S = np.where(np.isfinite(S), S, np.nan)
    return {"R": R, "R0": R0, "f_i": f_i, "f_ii": f_ii, "rho": rho, "S": S}


def identification_verdict(f_i: Optional[float], s_ci: Optional[Sequence[float]]) -> str:
    """INDETERMINATE iff f_i > 0.7 (too little on-distribution learning to judge); IDENTIFIES iff
    S's CI lower bound > 0.5; SELECTIVE iff its upper bound < 0.5; else UNDECIDED."""
    if f_i is None or not math.isfinite(f_i) or f_i > F_I_INDETERMINATE:
        return "INDETERMINATE"
    if s_ci is None:
        return "UNDECIDED"
    if s_ci[0] > S_SPLIT:
        return "IDENTIFIES"
    if s_ci[1] < S_SPLIT:
        return "SELECTIVE"
    return "UNDECIDED"


def identification_read(errs: Mapping[str, Mapping[str, np.ndarray]], clusters: Any,
                        archs: Mapping[str, str], n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                        alpha: float = ALPHA) -> dict:
    """(e) per key from ``errs[key] = {"i", "ii", "i0", "ii0"}`` (raw errors on the probe rows, in
    probe order; the cluster of probe row j — for (i) AND (ii) — is the battle of (i)'s row j)."""
    keys = list(errs)
    qs: Dict[str, Dict[str, SortedQuantiles]] = {k: {s: SortedQuantiles(errs[k][s])
                                                     for s in ("i", "ii", "i0", "ii0")}
                                                 for k in keys}

    def fn(W: np.ndarray) -> np.ndarray:
        return np.stack([qs[k][s](W, (0.5,))[:, 0] for k in keys for s in ("i", "ii", "i0", "ii0")],
                        axis=1)

    D = paired_boot(clusters, fn, n_boot, seed)
    out: Dict[str, dict] = {}
    p_gt: Dict[str, Optional[float]] = {}
    p_lt: Dict[str, Optional[float]] = {}
    for j, k in enumerate(keys):
        pt = {s: float(qs[k][s].point((0.5,))[0]) for s in ("i", "ii", "i0", "ii0")}
        A = identification_arith(pt["i"], pt["ii"], pt["i0"], pt["ii0"])
        B = identification_arith(*(D[:, 4 * j + t] for t in range(4)))
        row: Dict[str, Any] = {"arch": archs.get(k), "m_i": _g(pt["i"]), "m_ii": _g(pt["ii"]),
               "m_i0": _g(pt["i0"]), "m_ii0": _g(pt["ii0"])}
        for name in ("R", "R0", "f_i", "f_ii", "rho", "S"):
            row[name] = _g(A[name])
            row[f"{name}_ci"] = _ci(B[name])
        p_gt[k] = p_greater(B["S"], S_SPLIT) if row["S"] is not None else None
        p_lt[k] = p_less(B["S"], S_SPLIT) if row["S"] is not None else None
        row["p_S_gt_half"], row["p_S_lt_half"] = _g(p_gt[k]), _g(p_lt[k])
        row["verdict_unadjusted"] = identification_verdict(row["f_i"], row["S_ci"])
        out[k] = row
    h_gt, h_lt = holm(p_gt, alpha), holm(p_lt, alpha)
    for k in keys:
        # THE VERDICT IS HOLM'S (across the keys, one family per direction): the unadjusted CI
        # rule above is reported beside it, never instead of it.
        f_i = out[k]["f_i"]
        if f_i is None or not math.isfinite(f_i) or f_i > F_I_INDETERMINATE:
            out[k]["verdict"] = "INDETERMINATE"
        elif h_gt[k]["reject"]:
            out[k]["verdict"] = "IDENTIFIES"
        elif h_lt[k]["reject"]:
            out[k]["verdict"] = "SELECTIVE"
        else:
            out[k]["verdict"] = "UNDECIDED"
    return {"keys": out, "holm_S_gt_half": h_gt, "holm_S_lt_half": h_lt,
            "rule": (f"INDETERMINATE iff f_i > {F_I_INDETERMINATE}; else IDENTIFIES iff Holm "
                     f"(across the keys) rejects S <= {S_SPLIT} (one-sided); SELECTIVE iff Holm "
                     f"rejects S >= {S_SPLIT}; else UNDECIDED. `verdict_unadjusted` is the "
                     "per-key 95 % CI rule, reported only. S = log f_ii / log f_i: 0 = learns "
                     "only what it visits, 1 = identifies the target everywhere")}


# ---------------------------------------------------------------------------------------------
# one checkpoint's NumPy half, and the cross-checkpoint series
# ---------------------------------------------------------------------------------------------

_STEP_RE = re.compile(r"_(\d+)_steps(?:\.zip)?$")


def parse_step(path: Any) -> Optional[int]:
    m = _STEP_RE.search(Path(str(path)).name)
    return int(m.group(1)) if m else None


@dataclass
class Carry:
    """What the cross-checkpoint reads need from one checkpoint (kept in memory, never written)."""
    label: str
    path: str
    heads: str
    keys: List[str]
    errs: Dict[str, np.ndarray] = field(default_factory=dict)        # raw error, every bank row
    spread: Dict[str, Tuple[Optional[float], np.ndarray]] = field(default_factory=dict)
    ident: Optional[dict] = None
    feat: Any = None            # this checkpoint's feat head (a deep copy), or None

    @property
    def step(self) -> Optional[int]:
        return parse_step(self.path)


def per_checkpoint(decisions: Sequence[dict], v: np.ndarray, errs: Mapping[str, np.ndarray],
                   zs: Mapping[str, np.ndarray]) -> Tuple[dict, Dict[str, Tuple[Optional[float], np.ndarray]]]:
    """(a), (b) and (c)'s per-checkpoint half over the bank's rows. ``errs`` / ``zs``: per key the
    raw error / its z-score, bank order."""
    from main.ridealong_read.meters import DRAW_POLICY, binary_entropy, outcome_target

    battles = np.array([d["battle"] for d in decisions])
    z, keep = outcome_target([d["outcome"] for d in decisions], DRAW_POLICY)
    verr = np.abs(np.asarray(v, dtype=np.float64) - z)
    ref = binary_entropy(v)
    out: dict = {}
    if BASE in zs and len(zs) > 1:
        out["a_v_error_beyond_v_uncertainty"] = compare_v_error(
            {k: np.asarray(s)[keep] for k, s in zs.items()}, verr[keep], ref[keep], battles[keep])
    else:
        out["a_v_error_beyond_v_uncertainty"] = {"skipped": "needs base RND and >= 1 variant"}
    late, thr = late_game_mask([d["turn"] for d in decisions])
    contrasts = {"b1_exploiter_vs_rest": np.array([d["opp_class"] == "exploiter" for d in decisions]),
                 "b2_late_game_vs_rest": late}
    if BASE in errs and len(errs) > 1:
        cov = compare_coverage(errs, contrasts, battles)
        cov["b2_turn_threshold"] = {"quantile": LATE_TURN_Q, "turn": _g(thr),
                                    "rule": "turn > the read's own 95th percentile turn (np.percentile)"}
        cov["rows"] = "ALL bank rows (draws included)"
        out["b_coverage"] = cov
    else:
        out["b_coverage"] = {"skipped": "needs base RND and >= 1 variant"}
    spread: Dict[str, Tuple[Optional[float], np.ndarray]] = {}
    sat: Dict[str, dict] = {}
    for k, e in errs.items():
        sat[k], dist = spread_point_and_dist(e, battles)
        spread[k] = (sat[k]["rel_spread"], dist)
    out["c_saturation"] = {"keys": sat, "quantiles": QUANTILE_METHOD,
                           "rows": "ALL bank rows (draws included), the RAW error"}
    return out, spread


def identification_series(carries: Sequence[Carry]) -> Dict[str, List[dict]]:
    """Per key the time series of (e) over the checkpoints, in CLI order."""
    out: Dict[str, List[dict]] = {}
    for c in carries:
        if not c.ident or "keys" not in c.ident:
            continue
        for k, row in c.ident["keys"].items():
            out.setdefault(k, []).append({
                "label": c.label, "step": c.step,
                **{f: row.get(f) for f in ("m_i", "m_ii", "R", "rho", "f_i", "f_ii", "S", "verdict")}})
    return out


def cross_checkpoint(carries: Sequence[Carry], drift: Optional[dict]) -> dict:
    """The ``rnd_variants_series.json`` body: (c) first vs last and every consecutive pair, (d) as
    computed by the caller, and (e)'s per-key series."""
    body: dict = {"schema": SCHEMA,
                  "checkpoints": [{"label": c.label, "path": c.path, "step": c.step,
                                   "heads": c.heads, "keys": c.keys} for c in carries]}
    if len(carries) >= 2:
        first, last = carries[0], carries[-1]
        body["c_saturation"] = {
            "first_vs_last": {"first": first.label, "last": last.label,
                              **saturation_compare(first.spread, last.spread)},
            "consecutive": [{"first": a.label, "last": b.label, **saturation_compare(a.spread, b.spread)}
                            for a, b in zip(carries, carries[1:])]}
    else:
        body["c_saturation"] = {"skipped": "needs >= 2 checkpoints"}
    body["d_feat_drift"] = drift if drift is not None else {"skipped": "needs >= 2 checkpoints with feat"}
    body["e_identification_series"] = identification_series(carries)
    return body


__all__ = ["SCHEMA", "ALPHA", "BASE", "N_PROBE", "PROBE_SEED", "CHIMERA_GENERATOR", "variant_key",
           "declarations", "base_arch", "p_two_sided", "p_greater", "p_less", "holm", "paired_boot",
           "SortedQuantiles", "weighted_quantile", "compare_v_error", "ranked_vs_base",
           "late_game_mask", "compare_coverage", "coverage_verdict", "spread_point_and_dist",
           "saturation_compare", "drift_one", "drift_compare", "probe_indices", "chimera_donors",
           "identification_arith", "identification_verdict", "identification_read", "parse_step",
           "Carry", "per_checkpoint", "identification_series", "cross_checkpoint"]
