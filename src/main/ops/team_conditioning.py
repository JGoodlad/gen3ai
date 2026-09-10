"""``main.ops.team_conditioning`` — the OWN-TEAM cell arithmetic behind the (A)/(B) separation.

Split out of :mod:`main.ops.conditioning_meters` for the same reason
:mod:`main.ops.critic_read_render` is split out of the read: the meters module already owns the
extraction, the reweighting, the spread identity and the decodes, and one more block of
second-moment arithmetic would push it past the file-size gate's reporting band. The METERS are
still DECLARED in ``conditioning_meters`` — this module holds only the arithmetic they call.

**THE QUESTION THESE ROWS SEPARATE.** Arm 8 (``ai_v12_19_ladder_lambda09``) is the first ladder
arm whose ``V`` decodes its OWN TEAM at turn 1 (R² ≈ 0.15 against ≈ 0 on every control; ledger
2026-09-10 · *READ · critic ladder arm 8*). Two readings fit that fact and the existing rows do
not separate them:

* **(A) CONDITIONING** — the critic now uses its team as it should. Teams differ in strength, so a
  critic that knows which one it is holding predicts better. Within a team it still discriminates
  by the board.
* **(B) SUBSTITUTION** — the critic has replaced board state with team identity. It is right on
  average per team and blind INSIDE one.

The separation is the classic decomposition of a forecast's skill into a BETWEEN-cell part and a
WITHIN-cell part, with the own team as the cell:

| | within-team resolution | between-team spread |
|---|---|---|
| (A) conditioning | not lower, ideally higher | UP |
| (B) substitution | **DOWN** | UP |
| neither | flat | flat |

🚨 **THE CELLS ARE SMALL AND THAT IS THE WHOLE HAZARD.** The pool carries 719 teams, so a
4,800-battle frame averages ~7 episodes per team. A binned resolution computed inside a 7-episode
cell is mostly the binning's own sampling noise, and it is *positively* biased — each bin's
observed rate departs from the cell's base rate by chance — so a within-team row read on tiny
cells is not evidence of (B) or against it. Two things are therefore mandatory and both are done
here: only cells clearing ``MIN_TEAM_BATTLES`` are used, and the CELL CENSUS (how many cells, how
many battles and states inside them, the median per-cell N) is printed beside every row. The
COARSE companion — the same resolution over five team-STRENGTH strata, hundreds of episodes each
— carries the same (A)/(B) sign logic with none of the small-cell inflation, and disagreement
between the two rows is itself the reading.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

#: reliability-diagram bins for a within-cell Murphy resolution — the same 10 the identity's
#: Murphy decomposition uses, so a resolution here and a resolution there are the same estimator.
RESOLUTION_BINS = 10
#: how many team-STRENGTH strata the coarse companion cuts the qualifying teams into. Quantiles of
#: the team's leave-one-battle-out win rate, cut at equal BATTLE mass rather than equal team count
#: so no stratum is a handful of rarely-drawn teams.
TEAM_STRATA = 5
#: a cell needs this many battles IN A BOOTSTRAP DRAW before its mean has a sampling variance to
#: subtract. The cell SET is fixed on the full frame (see :func:`team_spread`); this is the extra
#: condition a resampled draw must meet.
MIN_CELL_BATTLES_IN_DRAW = 2


def bin_codes(v: np.ndarray, nbins: int = RESOLUTION_BINS) -> np.ndarray:
    """Reliability-diagram bin of each forecast, clipped into [0, 1] first."""
    return np.minimum(nbins - 1,
                      (np.clip(np.asarray(v, dtype=float), 0.0, 1.0) * nbins).astype(int))


def cellwise_resolution(v: np.ndarray, y: np.ndarray, w: np.ndarray, cell: np.ndarray,
                        cell_weight: np.ndarray, n_cells: int, *,
                        nbins: int = RESOLUTION_BINS) -> float:
    """Murphy RESOLUTION computed INSIDE each cell against that cell's OWN base rate, then a
    ``cell_weight``-weighted mean over the cells.

    ``RES_c = Σ_bins (W_bin / W_c) · (ō_bin − ō_c)²`` — how far the outcome rate of the states the
    critic put in one bin departs from the base rate of the cell they came from. Pooled over
    everything this is the ordinary Murphy resolution; conditioned on the own team it is exactly
    the *within-team discrimination* reading (B) predicts will be low.

    Fully vectorised over (cell × bin) because the battle-clustered bootstrap evaluates it 2,000
    times; a per-cell python loop over ~700 teams made the read minutes rather than seconds.
    """
    v = np.asarray(v, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    cell = np.asarray(cell, dtype=int)
    cw = np.asarray(cell_weight, dtype=float)
    keep = (cell >= 0) & (cell < n_cells) & np.isfinite(v) & np.isfinite(w) & (w > 0)
    if n_cells <= 0 or not keep.any():
        return float("nan")
    v, y, w, cell = v[keep], y[keep], w[keep], cell[keep]
    g = cell * nbins + bin_codes(v, nbins)
    size = n_cells * nbins
    Wg = np.bincount(g, weights=w, minlength=size)
    Yg = np.bincount(g, weights=w * y, minlength=size)
    Wc = np.bincount(cell, weights=w, minlength=n_cells)
    Yc = np.bincount(cell, weights=w * y, minlength=n_cells)
    ok_c = Wc > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        obar = np.where(ok_c, Yc / np.where(ok_c, Wc, 1.0), 0.0)
        ok_g = Wg > 0
        obar_g = np.where(ok_g, Yg / np.where(ok_g, Wg, 1.0), 0.0)
    contrib = np.where(ok_g, Wg * (obar_g - np.repeat(obar, nbins)) ** 2, 0.0)
    res = contrib.reshape(n_cells, nbins).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        res = np.where(ok_c, res / np.where(ok_c, Wc, 1.0), np.nan)
    m = ok_c & np.isfinite(res) & np.isfinite(cw) & (cw > 0)
    den = float(cw[m].sum())
    return float((cw[m] * res[m]).sum() / den) if den > 0 else float("nan")


def cell_census(cell_of_state: np.ndarray, battle_of_state: np.ndarray,
                n_cells: int) -> Dict[str, Any]:
    """The CENSUS printed beside a within-cell row: how many cells, and how much is inside them.

    🚨 Printed because the row cannot be read without it. Within-cell resolution on cells of ~7
    episodes is close to the binning's own noise, so `n_cells`, the battles and states they hold,
    and the MEDIAN per-cell N are what tell a reader whether the number is a measurement.
    """
    cell = np.asarray(cell_of_state, dtype=int)
    keep = (cell >= 0) & (cell < n_cells)
    nan = float("nan")
    if not keep.any():
        return {"n_cells": 0, "n_battles": 0, "n_states": 0,
                "median_battles_per_cell": nan, "median_states_per_cell": nan,
                "min_battles_per_cell": nan, "max_battles_per_cell": nan}
    cell = cell[keep]
    battles = np.asarray(battle_of_state)[keep]
    states_per = np.bincount(cell, minlength=n_cells).astype(float)
    pairs = np.unique(np.stack([cell, np.unique(battles, return_inverse=True)[1]], axis=1), axis=0)
    battles_per = np.bincount(pairs[:, 0].astype(int), minlength=n_cells).astype(float)
    live = states_per > 0
    return {"n_cells": int(live.sum()), "n_battles": int(battles_per[live].sum()),
            "n_states": int(states_per[live].sum()),
            "median_battles_per_cell": float(np.median(battles_per[live])),
            "median_states_per_cell": float(np.median(states_per[live])),
            "min_battles_per_cell": float(battles_per[live].min()),
            "max_battles_per_cell": float(battles_per[live].max())}


def strata_of(rate: np.ndarray, n_battles: np.ndarray, qualifies: np.ndarray,
              n_strata: int = TEAM_STRATA) -> np.ndarray:
    """Assign each QUALIFYING team a strength stratum — quantiles of its own win rate, cut at
    equal BATTLE mass. Non-qualifying teams get ``-1``.

    Equal battle mass, not equal team count: the trainee draws teams unevenly, and a stratum that
    is five rarely-drawn teams has the same small-cell problem the per-team row has.
    """
    rate = np.asarray(rate, dtype=float)
    nb = np.asarray(n_battles, dtype=float)
    ok = np.asarray(qualifies, dtype=bool) & np.isfinite(rate) & (nb > 0)
    out = np.full(rate.size, -1, dtype=int)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out
    order = idx[np.lexsort((idx, rate[idx]))]
    cum = np.cumsum(nb[order])
    total = float(cum[-1])
    if total <= 0:
        return out
    # the stratum of a team is the battle-mass quantile its own battles' MIDPOINT falls in, so a
    # single very heavy team cannot swallow two strata's worth of mass and leave one empty.
    mid = (cum - nb[order] / 2.0) / total
    out[order] = np.minimum(n_strata - 1, (mid * n_strata).astype(int))
    return out


def team_spread(*, w: np.ndarray, n_states: np.ndarray, sum_v: np.ndarray, y: np.ndarray,
                cell: np.ndarray, n_cells: int, keep_cell: Optional[np.ndarray] = None,
                min_battles: int = MIN_CELL_BATTLES_IN_DRAW) -> Dict[str, float]:
    """BETWEEN-TEAM spread of ``V`` against the team's own win rate, noise-corrected, plus the raw
    companion — the OWN-TEAM analogue of the between-opponent spread identity.

    Every argument is per SELECTED BATTLE and aligned 1:1 with the selection (the same rule
    :func:`~main.ops.conditioning_meters.cell_stats` carries: a bootstrap draw has exactly as many
    entries as the original, so a size test cannot tell a resampled selection from an unresampled
    one). ``cell`` is that battle's team code, ``-1`` for a team the frame does not qualify.

    **What the two sides are.** ``sd_V`` is the between-team spread of the team's IPW mean ``V``
    over the bucket's states; ``sd_y`` is the between-team spread of the team's OWN win rate. The
    per-battle leave-one-battle-out rate the decoder uses averages, within a team, to exactly that
    win rate when the team's battles carry equal weight — the leave-one-out exists to stop a
    DECODE from carrying its own answer, and there is no decode here, only two marginal second
    moments.

    **The noise correction** subtracts each side's own IPW sampling variance, estimated on the same
    battles, exactly as the opponent identity does. Unlike the opponent side there is no external
    100-game denominator to draw a Binomial from — the traced battles ARE the team's sample — so
    the outcome side's noise is estimated rather than simulated, and ``ratio_raw`` is reported
    beside it as the uncorrected, unclamped, monotone companion.

    🚨 The ratio is CLAMPED the same way the opponent one is (each corrected variance floored at
    zero), which makes it biased and non-monotone: **the interval is the read, and a 0.000 is not
    "no spread"**.
    """
    w = np.asarray(w, dtype=float)
    ns = np.asarray(n_states, dtype=float)
    sv = np.asarray(sum_v, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.asarray(cell, dtype=int)
    nan = float("nan")
    empty = {k: nan for k in ("sd_V", "sd_y", "ratio", "ratio_raw", "delta",
                              "noise_V", "noise_y", "n_cells", "n_battles",
                              "median_battles_per_cell")}
    has = (ns > 0) & (c >= 0) & (c < n_cells) & (w > 0)
    if n_cells <= 0 or not has.any():
        return empty
    vb = np.where(ns > 0, sv / np.where(ns > 0, ns, 1.0), np.nan)
    cc, ww, vv, yy = c[has], w[has], vb[has], y[has]
    W = np.bincount(cc, weights=ww, minlength=n_cells)
    nb = np.bincount(cc, minlength=n_cells).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = W > 0
        mV = np.where(ok, np.bincount(cc, weights=ww * vv, minlength=n_cells)
                      / np.where(ok, W, 1.0), np.nan)
        mY = np.where(ok, np.bincount(cc, weights=ww * yy, minlength=n_cells)
                      / np.where(ok, W, 1.0), np.nan)
        s2V = np.bincount(cc, weights=(ww * (vv - mV[cc])) ** 2, minlength=n_cells)
        s2Y = np.bincount(cc, weights=(ww * (yy - mY[cc])) ** 2, minlength=n_cells)
        corr = np.where(nb > 1, nb / np.maximum(nb - 1, 1.0), 1.0)
        seV2 = np.where(ok & (nb > 1), s2V / np.where(ok, W, 1.0) ** 2 * corr, np.nan)
        seY2 = np.where(ok & (nb > 1), s2Y / np.where(ok, W, 1.0) ** 2 * corr, np.nan)
    m = (ok & (nb >= max(int(min_battles), 2)) & np.isfinite(mV) & np.isfinite(mY)
         & np.isfinite(seV2) & np.isfinite(seY2))
    if keep_cell is not None:
        m = m & np.asarray(keep_cell, dtype=bool)
    if int(m.sum()) < 3:
        return empty
    V, Y = mV[m], mY[m]
    nV, nY = float(np.mean(seV2[m])), float(np.mean(seY2[m]))
    vV, vY = float(np.var(V, ddof=1)), float(np.var(Y, ddof=1))
    sV, sY = np.sqrt(max(vV - nV, 0.0)), np.sqrt(max(vY - nY, 0.0))
    rV, rY = np.sqrt(vV), np.sqrt(vY)
    return {"sd_V": float(sV), "sd_y": float(sY),
            "ratio": float(sV / sY) if sY > 0 else nan,
            "ratio_raw": float(rV / rY) if rY > 0 else nan,
            "delta": float(sV - sY), "noise_V": float(np.sqrt(nV)), "noise_y": float(np.sqrt(nY)),
            "n_cells": float(m.sum()), "n_battles": float(nb[m].sum()),
            "median_battles_per_cell": float(np.median(nb[m]))}


def reading_of(within_team: Optional[float], within_stratum: Optional[float],
               between: Optional[float], *, tol: float = 0.0) -> str:
    """Which reading a set of DELTAS supports — (A), (B), or neither. Text, never a label.

    Deliberately a helper on the three signs and nothing else: the labels the report prints come
    from the registered DETECTED / WITHIN FLOOR / NOT DETECTED machinery, and this sentence is a
    reading of those rows, not a verdict of its own.
    """
    def sign(x: Optional[float]) -> int:
        if x is None or not np.isfinite(float(x)):
            return 0
        return 1 if float(x) > tol else (-1 if float(x) < -tol else 0)

    wt, ws, bt = sign(within_team), sign(within_stratum), sign(between)
    within = wt if wt == ws or ws == 0 else (0 if wt == 0 else wt)
    if bt < 0 and within < 0:
        # 🚨 The case arm 8 actually produced. The spread ratio is an AMPLITUDE and the own-team
        # R^2 is an ALIGNMENT — a monotone out-of-fold decode, invariant to scale — so a head can
        # order its teams better than the control while emitting a SMALLER between-team spread.
        # Neither reading as posed covers that, and saying so is the honest output.
        return ("neither as posed — BOTH components are DOWN: less dispersion BETWEEN teams and "
                "less discrimination INSIDE one. Note that the spread ratio is an AMPLITUDE while "
                "the own-team R^2 is an ALIGNMENT (a monotone decode, invariant to scale), so a "
                "head can ORDER its teams better while emitting a smaller between-team spread — "
                "read the two together, never the R^2 alone")
    if bt > 0 and within < 0:
        return ("(B) SUBSTITUTION — the between-team spread is UP while the WITHIN-team "
                "discrimination is DOWN: the head is right about which team it holds and worse "
                "inside one")
    if bt > 0 and within >= 0:
        return ("(A) CONDITIONING — the between-team spread is UP and the within-team "
                "discrimination is not lower: the head has ADDED team information without "
                "spending board information")
    if within > 0 and bt <= 0:
        return ("neither cleanly — the WITHIN-team discrimination is up without a between-team "
                "spread to go with it, which is a board-reading gain, not a conditioning one")
    if wt == ws == bt == 0:
        return "neither — no component moved"
    return ("neither cleanly — the two components do not line up with either reading; read the "
            "cell census and the intervals before quoting a direction")
