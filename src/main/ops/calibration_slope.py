"""``main.ops.calibration_slope`` — the CALIBRATION SLOPE of ``V``, and the support it is fitted on.

Split out of :mod:`main.ops.conditioning_meters` for the reason :mod:`main.ops.team_conditioning`
was: the meters module owns the extraction, the reweighting, the spread identity and the decodes,
and one more block of estimator arithmetic would push it past the file-size gate's reporting band.
The METERS are still DECLARED in ``conditioning_meters`` — this module holds only the arithmetic.

**THE QUANTITY.** Regress the realized outcome ``y`` on the forecast, in the LOGIT of the forecast,
by weighted logistic regression::

    logit P(y = 1) = a + b · logit(V)

``(a, b)`` is the classic Cox (1958) RECALIBRATION pair — *calibration-in-the-large* and the
*calibration slope*. A perfectly calibrated forecast reads ``b = 1``, ``a = 0``. The slope is the
DISPERSION reading and it is the whole reason this row exists:

* ``b > 1`` — **UNDER-dispersed / SHRUNK**. The forecasts are squeezed toward the base rate: where
  the head says 0.7 the realized rate is *above* 0.7 and where it says 0.3 it is *below*. Stretching
  the predictions away from the base rate would improve them.
* ``b ≈ 1`` — correctly dispersed.
* ``b < 1`` — **OVER-dispersed**: the head's opinions are too extreme for the evidence behind them.

**WHY IT IS THE SHARP TEST OF SHRINKAGE.** A λ-return target blends the bootstrapped ``V`` into the
label, so the thing being fitted is compressed toward the base rate relative to a raw 0/1 outcome.
Fitting a compressed target is a shrinkage estimator: it can IMPROVE the rank order of what is
emitted while REDUCING the amplitude. Rank order is measured by an out-of-fold decode (scale
invariant — it cannot see shrinkage at all) and amplitude by a spread ratio (which sees it but also
sees every other reason a spread moves). The calibration slope sees the compression DIRECTLY and in
the units the compression happens in, so an arm that is shrunk reads ``b > 1`` and one that merely
learned more reads ``b ≈ 1``.

🚨 **THE SLOPE'S OWN STANDARD ERROR SCALES AS 1/SD(logit V).** An arm whose predictions are more
compressed has a SHORTER LEVER ARM, so it gets a WIDER interval — from the very effect under test.
That bias is conservative (it hides a real difference, it cannot manufacture one) but it must be
VISIBLE, so two things ride beside every slope row and are never optional:

* :func:`support_stats` — ``sd(V)`` and ``sd(logit V)`` per side, printed like the cell census;
* the **COMMON-SUPPORT** companion, :func:`common_window` — both sides restricted to the
  intersection of their central 95% of ``V`` and re-fitted there, which removes the lever-arm
  difference by construction. A slope difference that survives the common-support row is not the
  lever arm; one that vanishes there was.

🚨 **``V`` IS CLIPPED BEFORE THE LOGIT** at :data:`CALIB_EPS`, because ``logit(0)`` is not a number
and because a forecast at 0.9999 is a leverage point worth five ordinary states in a regression
whose x-axis is the logit. The clipped SHARE is reported beside the row: a row computed on a column
that is 20% clipped is a statement about the clip, not about the head.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: ``V`` is clipped into ``[CALIB_EPS, 1 - CALIB_EPS]`` before the logit, bounding |logit V| at
#: ~6.9. Looser (1e-6) admits leverage points worth ~13.8/2.9 ≈ 5 ordinary states each; tighter
#: starts compressing the very dispersion the slope is measuring. The clipped share is REPORTED.
CALIB_EPS = 1e-3
#: Newton/IRLS budget. Two to six parameters on a few thousand rows converges in well under ten
#: iterations unless the data are separable — which is exactly the case that must return NaN.
IRLS_MAX_ITER = 60
IRLS_TOL = 1e-9
#: a fitted |slope| past this is separation, not a measurement.
SLOPE_CAP = 50.0
#: the central mass each side contributes to the COMMON-SUPPORT window.
SUPPORT_Q = (2.5, 97.5)
#: the two turn windows a slope is reported on, and the bucket the within-stratum and
#: common-support companions are computed on.
BUCKETS = ("all", "t1_3")
PRIMARY_BUCKET = "all"


def logit(v: np.ndarray, eps: float = CALIB_EPS) -> np.ndarray:
    """``log(p / (1 - p))`` with ``p`` clipped away from 0 and 1 first."""
    p = np.clip(np.asarray(v, dtype=float), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def _fit(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Optional[np.ndarray]:
    """Weighted logistic regression by Newton/IRLS -> the coefficient vector, or ``None``.

    ``None`` — never a number — for a degenerate fit: one outcome class, fewer rows than the fit
    has parameters and slack, a singular Hessian, a non-finite iterate, or no convergence inside
    :data:`IRLS_MAX_ITER` (which for this model means the data are separable). A separated fit's
    coefficient diverges, and a diverging coefficient reported as a slope is a wrong reading with
    no tell.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n, k = X.shape
    if n < k + 5 or w.sum() <= 0:
        return None
    if float(y.min()) == float(y.max()):
        return None
    beta = np.zeros(k)
    for _ in range(IRLS_MAX_ITER):
        eta = np.clip(X @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        g = X.T @ (w * (y - p))
        H = X.T @ (X * (w * p * (1.0 - p))[:, None])
        # a negligible ridge only so a numerically singular Hessian solves rather than raising;
        # it is 1e-9 of the mean curvature and cannot move a reported slope.
        H.flat[:: k + 1] += 1e-9 * max(1.0, float(np.trace(H)) / k)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(step)):
            return None
        beta = beta + step
        if float(np.max(np.abs(step))) < IRLS_TOL:
            return beta
    return None


def slope_intercept(v: np.ndarray, y: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    """The Cox recalibration pair ``(intercept, slope)`` of ``y`` on ``logit(V)``."""
    nan = float("nan")
    x = logit(v)
    beta = _fit(np.stack([x, np.ones(x.size)], axis=1), y, w)
    if beta is None or abs(float(beta[0])) > SLOPE_CAP:
        return {"slope": nan, "intercept": nan}
    return {"slope": float(beta[0]), "intercept": float(beta[1])}


def within_stratum_slope(v: np.ndarray, y: np.ndarray, w: np.ndarray,
                         stratum: np.ndarray) -> float:
    """One SHARED slope on ``logit(V)`` with a free intercept PER STRATUM — the calibration slope
    computed WITHIN the coarse own-team strength strata rather than across them.

    Shrinkage across teams and shrinkage inside one are different statements: a head could be
    correctly dispersed inside each stratum and merely compressed BETWEEN them (only the pooled row
    then reads high), or compressed everywhere (both rows read high). The stratum intercepts absorb
    the between-stratum part exactly, so this row is the within part and nothing else.

    🚨 A stratum whose outcomes are ALL wins or ALL losses is DROPPED, not fitted. Its own dummy
    would diverge (perfect separation on that column), taking the shared slope's convergence with
    it — the fit would return NaN for a reason that has nothing to do with the slope.
    """
    v = np.asarray(v, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    s = np.asarray(stratum, dtype=int)
    keep = s >= 0
    if not keep.any():
        return float("nan")
    v, y, w, s = v[keep], y[keep], w[keep], s[keep]
    codes, inv = np.unique(s, return_inverse=True)
    n_s = codes.size
    ymin = np.full(n_s, np.inf)
    ymax = np.full(n_s, -np.inf)
    np.minimum.at(ymin, inv, y)
    np.maximum.at(ymax, inv, y)
    live = ymin < ymax
    if not live.any():
        return float("nan")
    m = live[inv]
    v, y, w, inv = v[m], y[m], w[m], inv[m]
    codes2, inv2 = np.unique(inv, return_inverse=True)
    D = np.zeros((v.size, codes2.size))
    D[np.arange(v.size), inv2] = 1.0
    beta = _fit(np.concatenate([logit(v)[:, None], D], axis=1), y, w)
    if beta is None or abs(float(beta[0])) > SLOPE_CAP:
        return float("nan")
    return float(beta[0])


# --------------------------------------------------------------------------- the support

def _wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted quantile of ``x`` at percentile ``q``, linearly interpolated on the weight CDF."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not m.any():
        return float("nan")
    x, w = x[m], w[m]
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    c = np.cumsum(w)
    tot = float(c[-1])
    if tot <= 0:
        return float("nan")
    # the midpoint convention, so a single heavy state cannot own a whole tail
    pos = (c - 0.5 * w) / tot
    return float(np.interp(q / 100.0, pos, x))


def support_stats(v: np.ndarray, w: np.ndarray, *, eps: float = CALIB_EPS) -> Dict[str, float]:
    """The LEVER ARM the slope is fitted on — printed beside every slope row, never optional.

    ``sd_logit_V`` is the one that matters: the slope's standard error scales as ``1/sd(logit V)``,
    so a head whose predictions are compressed buys a wider interval from the very effect under
    test. ``q_lo``/``q_hi`` are the central 95% of ``V``, which is what the common-support window
    is intersected from, and ``clipped_share`` says how much of the column the clip touched.
    """
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    nan = float("nan")
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if int(m.sum()) < 2:
        return {"n_states": int(m.sum()), "sd_V": nan, "sd_logit_V": nan,
                "mean_V": nan, "q_lo": nan, "q_hi": nan, "clipped_share": nan}
    v, w = v[m], w[m]
    x = logit(v, eps)
    mv = float(np.average(v, weights=w))
    mx = float(np.average(x, weights=w))
    return {"n_states": int(v.size),
            "sd_V": float(np.sqrt(np.average((v - mv) ** 2, weights=w))),
            "sd_logit_V": float(np.sqrt(np.average((x - mx) ** 2, weights=w))),
            "mean_V": mv,
            "q_lo": _wquantile(v, w, SUPPORT_Q[0]), "q_hi": _wquantile(v, w, SUPPORT_Q[1]),
            "clipped_share": float(np.average(((v < eps) | (v > 1.0 - eps)).astype(float),
                                              weights=w))}


def common_window(a: Dict[str, float], c: Dict[str, float]) -> Optional[Tuple[float, float]]:
    """The overlap of the two sides' central 95% of ``V``, or ``None`` when they do not overlap.

    A window is refused rather than widened: two heads whose central masses do not meet have no
    common support, and a slope difference read across disjoint ranges is an extrapolation.
    """
    lo = max(float(a.get("q_lo", np.nan)), float(c.get("q_lo", np.nan)))
    hi = min(float(a.get("q_hi", np.nan)), float(c.get("q_hi", np.nan)))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    return (lo, hi)


# --------------------------------------------------------------------------- the block

def payload(*, n_battles: int, opp_of_battle: np.ndarray,
            buckets: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """The per-state columns the slope rows are computed from, in the shape :func:`block` wants.

    Kept as a standalone document so :mod:`main.ops.critic_read` can CACHE it and re-fit the
    common-support companion for a PAIR without re-reading either trace tree — the window is a
    function of both sides, so it cannot be computed inside a single run's block.
    """
    return {"n_battles": int(n_battles),
            "opp_of_battle": np.asarray(opp_of_battle, dtype=int),
            "buckets": {k: {kk: np.asarray(vv) for kk, vv in v.items()}
                        for k, v in buckets.items()}}


def to_json(pay: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not pay:
        return None
    return {"n_battles": int(pay["n_battles"]),
            "opp_of_battle": [int(x) for x in pay["opp_of_battle"]],
            "buckets": {k: {kk: [float(x) for x in vv] for kk, vv in v.items()}
                        for k, v in pay["buckets"].items()}}


def from_json(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not doc:
        return None
    ints = ("battle", "stratum")
    return {"n_battles": int(doc["n_battles"]),
            "opp_of_battle": np.asarray(doc["opp_of_battle"], dtype=int),
            "buckets": {k: {kk: np.asarray(vv, dtype=(int if kk in ints else float))
                            for kk, vv in v.items()}
                        for k, v in doc["buckets"].items()}}


def _csr(battle: np.ndarray, n_battles: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """State rows grouped by battle as a flat (order, start, count) triple — the same gather form
    :mod:`main.ops.conditioning_meters`'s bootstrap uses, so a resampled draw is one fancy index
    rather than a python loop over ~950 battles × 2,000 draws."""
    o = np.argsort(battle, kind="stable")
    cnt = np.bincount(battle, minlength=n_battles).astype(np.int64)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    return o, start, cnt


def _gather(sel: np.ndarray, flat: np.ndarray, start: np.ndarray,
            cnt: np.ndarray) -> np.ndarray:
    c = cnt[sel]
    tot = int(c.sum())
    if tot == 0:
        return np.empty(0, dtype=int)
    base = np.repeat(start[sel], c)
    within = np.arange(tot) - np.repeat(np.cumsum(c) - c, c)
    return flat[base + within]


def _slots(opp_of_battle: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Battle indices grouped by OPPONENT cell, as (all_slots, start_per_slot, size_per_slot).

    The bootstrap resamples battles WITHIN each opponent cell, because the roster is a fixed pinned
    set and not a sample — the same convention every other conditioning interval follows.
    """
    o = np.argsort(opp_of_battle, kind="stable")
    cells = opp_of_battle[o]
    bounds = np.flatnonzero(np.concatenate([[True], cells[1:] != cells[:-1], [True]]))
    sizes = np.diff(bounds)
    return o, np.repeat(bounds[:-1], sizes), np.repeat(sizes, sizes).astype(float)


def _rows(pay: Dict[str, Any], window: Optional[Tuple[float, float]],
          want: Sequence[str], idx: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
    """Every requested calibration point on one (possibly resampled, possibly windowed) frame.

    Row names are ``<coefficient>.<bucket>`` plus the one companion ``slope.within_stratum``; the
    caller maps them onto meter keys (:func:`block`'s ``names``), so this module never has to know
    what the meters are called.
    """
    out: Dict[str, float] = {}
    want = set(want)
    for bucket in BUCKETS:
        bk = pay["buckets"].get(bucket)
        wanted = [r for r in want if r.endswith(f".{bucket}")]
        within = bucket == PRIMARY_BUCKET and "slope.within_stratum" in want
        if bk is None or not (wanted or within):
            continue
        g = idx[bucket] if idx is not None else np.arange(bk["v"].size)
        v, y, w = bk["v"][g], bk["y"][g], bk["w"][g]
        if window is not None:
            m = (v >= window[0]) & (v <= window[1])
            v, y, w, g = v[m], y[m], w[m], g[m]
        if wanted:
            si = slope_intercept(v, y, w)
            out[f"slope.{bucket}"] = si["slope"]
            out[f"intercept.{bucket}"] = si["intercept"]
        if within:
            out["slope.within_stratum"] = within_stratum_slope(v, y, w, bk["stratum"][g])
    return {k: v for k, v in out.items() if k in want}


def _buckets_of(want: Sequence[str]) -> List[str]:
    """Which turn windows a requested row set touches — what :func:`block` reports SUPPORT for."""
    out = [b for b in BUCKETS if any(r.endswith(f".{b}") for r in want)]
    if "slope.within_stratum" in want and PRIMARY_BUCKET not in out:
        out.append(PRIMARY_BUCKET)
    return out


def block(pay: Optional[Dict[str, Any]], *, names: Dict[str, str], boot: int, seed: int,
          window: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """Every requested calibration row for ONE side, with its battle-clustered bootstrap draws.

    ``names`` maps this module's row names onto the caller's meter keys, which is the whole of the
    coupling between the two: the AS-TRACED rows and the COMMON-SUPPORT companion are the SAME
    estimator called twice with different names and a different ``window``, so they can never
    drift apart.

    ``window`` restricts both the point and every draw to a range of ``V``. It is passed in rather
    than derived because the common-support window is a property of the PAIR — a single run has no
    way to know it.
    """
    out: Dict[str, Any] = {"points": {}, "draws": {}, "support": {},
                           "window": None if window is None else [float(window[0]),
                                                                  float(window[1])]}
    if not pay or not pay.get("buckets"):
        return out
    want = list(names)

    for bucket in _buckets_of(want):
        bk = pay["buckets"].get(bucket)
        if bk is None:
            continue
        v, w, bat = bk["v"], bk["w"], bk["battle"]
        if window is not None:
            m = (v >= window[0]) & (v <= window[1])
            v, w, bat = v[m], w[m], bat[m]
        out["support"][bucket] = dict(support_stats(v, w),
                                      n_battles=int(np.unique(bat).size) if bat.size else 0)

    for row, val in _rows(pay, window, want).items():
        if np.isfinite(val):
            out["points"][names[row]] = float(val)
    if not out["points"]:
        return out

    n_b = int(pay["n_battles"])
    all_slots, start_per_slot, size_per_slot = _slots(pay["opp_of_battle"])
    csr = {b: _csr(pay["buckets"][b]["battle"], n_b) for b in _buckets_of(want)
           if b in pay["buckets"]}
    rng = np.random.default_rng(seed)
    acc: Dict[str, List[float]] = {}
    for _ in range(int(boot)):
        j = start_per_slot + (rng.random(all_slots.size) * size_per_slot).astype(int)
        sel = all_slots[j]
        idx = {b: _gather(sel, *csr[b]) for b in csr}
        for row, val in _rows(pay, window, want, idx).items():
            if np.isfinite(val):
                acc.setdefault(row, []).append(float(val))
    for row, vals in acc.items():
        if names[row] in out["points"]:
            out["draws"][names[row]] = np.asarray(vals, dtype=float)
    return out


def support_note() -> str:
    return ("the calibration slope's standard error scales as 1/sd(logit V), so a head whose "
            "predictions are COMPRESSED gets a wider interval from the very effect under test — a "
            "conservative bias, never a manufacturing one. sd(V) and sd(logit V) are printed per "
            "side for exactly that reason, and the COMMON-SUPPORT row re-fits both sides on the "
            "intersection of their central 95% of V, which removes the lever-arm difference by "
            "construction.")
