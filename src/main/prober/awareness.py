"""'Did the model KNOW?' — battle-level awareness verdicts from the distributional critic.

The per-decision recording already exists (`RLPlayer` writes the per-atom `value_dist` row and
the scalar `values` into every trace; `build_value_dist` renders one decision). This module adds
the fold the stall/timeout investigations kept re-deriving by hand: walk a battle's decisions
and answer, with pre-registered definitions, whether the model saw the loss coming — the
readout that turns "positive V the turn before a cap loss" from a post-mortem discovery into a
column (gen-11 runbook §3).

Definitions (fixed here so every probe means the same thing):

  * ``p_loss[i]``   — P(return < 0) under decision i's predicted distribution, in REAL return
                      units (see the denorm note below).
  * ``p_tail[i]``   — P(return ≤ tail_threshold): the catastrophic band mass. Default
                      threshold = vmin + 10% of the support range (with support [−12, 12] a cap
                      loss's clamped return lands in the bottom atoms).
  * ``knew_by_turn`` — the first game turn from which ``p_loss > 0.5`` holds SUSTAINED through
                      the battle's end (the longest such suffix). None if never.
  * ``lead_time``   — last decision turn − knew_by_turn (how much warning). None if blind.
  * ``blind_loss``  — a LOSS with ``knew_by_turn`` None: the model never saw it coming.
  * ``mean_tail_divergence`` — max over decisions of ``p_tail`` restricted to decisions whose
                      distribution MEAN is still positive: large when the mean stays optimistic
                      while the tail piles up — the stall signature (a scalar critic reads the
                      mean; only the distribution shows the pile-up).

PopArt / denormalization: on a ``--use-popart`` run the dist support is in the critic's
NORMALIZED return space, where "< 0" means "below the running mean", not "loss". The trace
never records (mu, sigma) — but it records the DEnormalized scalar ``values`` beside every
dist, and under ``value_from_dist`` the scalar IS ``mean_norm·sigma + mu``, so a per-battle
least-squares over (mean_norm, recorded_v) pairs recovers the frozen snapshot's (mu, sigma) to
machine precision (``fit_denorm``). Under a SEPARATE scalar head (``value_dist_mode
shaping``) the two heads target the same return, so the fit is an approximation — adequate for
thresholding, and only the placement of the zero bar depends on it. Without popart the fit
lands on ≈(0, 1) and changes nothing. Model-free either way.

Pure numpy, no model, no I/O — feed it decisions from any trace (arrays) or live forwards.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class AwarenessVerdict:
    n_decisions: int                   # decisions that carried a distribution (the fold's rows)
    outcome: str                       # "win" | "loss" | "draw" | "?"
    turns: "tuple[int, ...]"           # game turn of each folded decision (parallel arrays below)
    p_loss: "tuple[float, ...]"
    p_tail: "tuple[float, ...]"
    knew_by_turn: Optional[int]
    lead_time: Optional[int]
    blind_loss: bool
    mean_tail_divergence: float
    divergence_turn: Optional[int]     # where the stall signature peaked; None if it never fired
    denorm: "tuple[float, float]"      # the (mu, sigma) actually applied


def fit_denorm(mean_norms: Sequence[float],
               recorded_values: Sequence[float]) -> "tuple[float, float]":
    """Recover the popart (mu, sigma) from paired (normalized dist mean, denormalized scalar V)
    rows: least squares on ``v = mean·sigma + mu``. Falls back to (0, 1) when the pairs are too
    few/degenerate (constant means) or the fit is unusable (sigma ≤ 0) — i.e. identity, which is
    also the exact answer on a no-popart run."""
    m = np.asarray(mean_norms, dtype=np.float64)
    v = np.asarray(recorded_values, dtype=np.float64)
    ok = ~(np.isnan(m) | np.isnan(v))
    m, v = m[ok], v[ok]
    if len(m) < 2 or float(np.ptp(m)) < 1e-9:
        return (0.0, 1.0)
    sigma, mu = np.polyfit(m, v, 1)
    if not np.isfinite(sigma) or not np.isfinite(mu) or sigma <= 0:
        return (0.0, 1.0)
    return (float(mu), float(sigma))


def build_awareness(dists: Sequence[Optional[np.ndarray]],
                    turns: Sequence[int],
                    outcome: str,
                    support: "tuple[float, float, int]",
                    values: "Optional[Sequence[float]]" = None,
                    tail_threshold: Optional[float] = None) -> Optional[AwarenessVerdict]:
    """Fold one battle's per-decision distributions into the awareness verdict.

    ``dists``: per-decision per-atom probs (None rows skipped — old traces / uncaptured rows);
    ``turns``: the game turn of each decision (parallel); ``support`` = (vmin, vmax, bins) in
    the critic's own (possibly normalized) space; ``values``: the recorded DEnormalized scalar
    V per decision (parallel), used only to fit the popart denorm — omit for identity.
    Returns None when fewer than 2 decisions carry a distribution (no trajectory to judge).
    """
    vmin, vmax, bins = float(support[0]), float(support[1]), int(support[2])
    z = np.linspace(vmin, vmax, bins)

    rows = []
    for j, (t, d) in enumerate(zip(turns, dists)):
        if d is None:
            continue
        probs = np.asarray(d, dtype=np.float64)
        if probs.size != bins or np.isnan(probs).any():
            continue
        rows.append((j, int(t) if t is not None else -1, probs / max(float(probs.sum()), 1e-8)))
    if len(rows) < 2:
        return None

    means_norm = [float((p * z).sum()) for _j, _t, p in rows]
    if values is not None:
        paired_v = [values[j] if j < len(values) else np.nan for j, _t, _p in rows]
        mu, sigma = fit_denorm(means_norm, paired_v)
    else:
        mu, sigma = 0.0, 1.0

    z_real = z * sigma + mu
    if tail_threshold is None:
        tail_threshold = float(z_real[0] + 0.10 * (z_real[-1] - z_real[0]))

    p_loss, p_tail, div = [], [], []
    for (_j, _t, p), mean_norm in zip(rows, means_norm):
        mean_real = mean_norm * sigma + mu
        pl = float(p[z_real < 0.0].sum())
        pt = float(p[z_real <= tail_threshold].sum())
        p_loss.append(pl)
        p_tail.append(pt)
        # the stall signature: tail mass accumulating WHILE the mean is still positive —
        # a scalar critic (the mean) reads healthy exactly when this is large
        div.append(pt if mean_real > 0.0 else 0.0)

    # knew_by_turn: sustained P(loss)>0.5 through the end == the longest all-the-way-to-the-end
    # suffix, i.e. walk backward while p_loss stays above the bar.
    knew_idx = None
    for i in range(len(rows) - 1, -1, -1):
        if p_loss[i] > 0.5:
            knew_idx = i
        else:
            break

    knew_by_turn = rows[knew_idx][1] if knew_idx is not None else None
    last_turn = rows[-1][1]
    lead = (last_turn - knew_by_turn) if knew_by_turn is not None else None
    d_peak = int(np.argmax(div)) if div and max(div) > 0 else None
    return AwarenessVerdict(
        n_decisions=len(rows), outcome=outcome,
        turns=tuple(t for _j, t, _p in rows),
        p_loss=tuple(round(x, 4) for x in p_loss),
        p_tail=tuple(round(x, 4) for x in p_tail),
        knew_by_turn=knew_by_turn, lead_time=lead,
        blind_loss=(outcome == "loss" and knew_by_turn is None),
        mean_tail_divergence=float(round(max(div), 4)) if div else 0.0,
        divergence_turn=(rows[d_peak][1] if d_peak is not None else None),
        denorm=(round(mu, 6), round(sigma, 6)),
    )


def coverage_stats(dists: Sequence[Optional[np.ndarray]],
                   returns: Sequence[float],
                   support: "tuple[float, float, int]",
                   values: "Optional[Sequence[float]]" = None) -> Optional[dict]:
    """QUANTILE COVERAGE of the distributional head vs the realized outcome (runbook §3):
    per decision, the PIT ``F(G_i)`` — the predicted CDF evaluated at the realized discounted
    return ``G_i`` (the MC value target; real reward units, same denorm fit as
    ``build_awareness``). A calibrated head has PIT ~ Uniform(0,1): ``pit_mean`` ≈ 0.5
    (persistently < 0.5 ⇒ the head is OPTIMISTIC — outcomes land in its lower tail) and
    ``coverage80`` ≈ 0.80 (fraction of realized returns inside the predicted 10–90 band;
    below ⇒ over-confident/narrow, above ⇒ under-confident/wide). Returns None when fewer
    than 2 decisions carry a distribution. Model-free, pure numpy."""
    vmin, vmax, bins = float(support[0]), float(support[1]), int(support[2])
    z = np.linspace(vmin, vmax, bins)
    rows = []
    for j, d in enumerate(dists):
        if d is None or j >= len(returns) or returns[j] is None:
            continue
        probs = np.asarray(d, dtype=np.float64)
        if probs.size != bins or np.isnan(probs).any():
            continue
        rows.append((j, probs / max(float(probs.sum()), 1e-8)))
    if len(rows) < 2:
        return None
    means_norm = [float((p * z).sum()) for _j, p in rows]
    if values is not None:
        mu, sigma = fit_denorm(means_norm, [values[j] if j < len(values) else np.nan
                                            for j, _p in rows])
    else:
        mu, sigma = 0.0, 1.0
    z_real = z * sigma + mu
    pits = []
    for j, p in rows:
        # mid-PIT (continuity correction for a discrete predictive dist): count HALF the mass
        # of the atom the return lands on — the raw CDF would bias every PIT high by ~p(atom)/2,
        # reading a perfectly-centered prediction as ≈0.5 + half the center atom.
        mid_cdf = np.cumsum(p) - p / 2.0
        pits.append(float(np.interp(float(returns[j]), z_real, mid_cdf, left=0.0, right=1.0)))
    pits_a = np.asarray(pits)
    return {
        "n": len(pits),
        "pits": tuple(round(x, 4) for x in pits),   # per-decision, for run-level pooling
        "pit_mean": float(round(pits_a.mean(), 4)),
        "pit_std": float(round(pits_a.std(), 4)),
        "coverage80": float(round(float(((pits_a >= 0.10) & (pits_a <= 0.90)).mean()), 4)),
        "denorm": (round(mu, 6), round(sigma, 6)),
    }


def coverage_from_npz(npz, returns: Sequence[float],
                      support: "tuple[float, float, int]") -> Optional[dict]:
    """Trace-file form of ``coverage_stats`` (mirrors ``awareness_from_npz``)."""
    arr = npz.get("value_dist") if hasattr(npz, "get") else None
    if arr is None:
        return None
    dists = [np.asarray(arr[i]) if i < len(arr) else None for i in range(len(returns))]
    values = npz.get("values")
    return coverage_stats(dists, returns, support,
                          values=(list(np.asarray(values, dtype=np.float64))
                                  if values is not None else None))


def awareness_from_npz(npz, turns: Sequence[int], outcome: str,
                       support: "tuple[float, float, int]") -> Optional[AwarenessVerdict]:
    """Convenience: the trace-file form (`value_dist` [N, bins]; NaN rows = uncaptured; the
    sibling `values` array feeds the denorm fit)."""
    arr = npz.get("value_dist") if hasattr(npz, "get") else None
    if arr is None:
        return None
    dists = [np.asarray(arr[i]) if i < len(arr) else None for i in range(len(turns))]
    values = npz.get("values")
    return build_awareness(dists, turns, outcome, support,
                           values=(list(np.asarray(values, dtype=np.float64))
                                   if values is not None else None))
