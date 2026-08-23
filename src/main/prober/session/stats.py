"""Pure statistics behind the bait-loop fold and the critic-calibration probe.

Kept out of the session so a rate can never be computed one way for the total and another way for
a split: `_loop_aggregate` is called once for the whole population and once per split, and every
rate ships `{n, d, rate}` because "13.9%" over 843 battles and over 7 are different claims.
"""

from __future__ import annotations

import collections

import numpy as np

from main.prober.loops import median as _median


# --- critic-calibration pure helpers (model-free; the calibration probe) --------

def _ratio(k: int, n: int) -> dict:
    """A rate that CANNOT be quoted without its numerator and denominator. Every bait-loop rate
    ships this shape, because "13.9%" over 843 battles and over 7 are different claims and the
    percentage alone hides which one you are reading."""
    return {"n": int(k), "d": int(n), "rate": (round(k / n, 4) if n else None)}


def _rate_by(rows, key: str) -> dict:
    """`_ratio` over a boolean attribute, skipping rows where it is None (undecidable ≠ wrong)."""
    sel = [r for r in rows if getattr(r, key) is not None]
    return _ratio(sum(1 for r in sel if getattr(r, key)), len(sel))


def _loop_aggregate(folds) -> dict:
    """Aggregate a list of `loops.BattleLoops` folds. Pure over the folds — the session calls it
    once for the whole population and once per split, so a split can never be computed a different
    way from the total."""
    baits = [b for f in folds for b in f.baits]
    whiffs = [b for b in baits if b.whiff]
    loop_steps = [b for b in whiffs if b.loop_step]
    reads = [r for f in folds for r in f.reads]
    n_dec = sum(f.n_decisions for f in folds)
    deltas = [d for f in folds for d in f.turn_deltas]

    def _bucket(name):
        sel = [d for d in deltas if d["bucket"] == name]
        return {"n": len(sel),
                "median_delta_v": _median([d["delta_v"] for d in sel if d["delta_v"] is not None]),
                "median_delta_win_prob": _median(
                    [d["delta_win_prob"] for d in sel if d["delta_win_prob"] is not None])}

    return {
        "n_battles": len(folds),
        "opp_voluntary_pivots": sum(f.opp_voluntary_pivots for f in folds),
        "moved_into_pivots": len(baits),
        "misses": sum(1 for b in baits if b.kind == "miss"),
        "whiff_kinds": dict(collections.Counter(b.kind for b in whiffs)),
        # THE THREE HEADLINE RATES, each on a different denominator on purpose (see caveats).
        "whiff_rate_per_pivot": _ratio(len(whiffs), len(baits)),
        "whiff_rate_per_decision": _ratio(len(whiffs), n_dec),
        "reclick_rate": _ratio(sum(1 for b in whiffs if b.reclick), len(whiffs)),
        "loop_battle_rate": _ratio(sum(1 for f in folds if f.loop_battle), len(folds)),
        "loop_ge3_battle_rate": _ratio(sum(1 for f in folds if f.worst_loop >= 3), len(folds)),
        "loop_steps_per_decision": _ratio(len(loop_steps), n_dec),
        "worst_loop_max": max((f.worst_loop for f in folds), default=0),
        # The confidence half: a loop step taken at p≈0.96 is not exploration.
        "median_chosen_prob": {
            "loop_steps": _median([b.chosen_prob for b in loop_steps]),
            "other_whiffs": _median([b.chosen_prob for b in whiffs if not b.loop_step]),
            "all_baits": _median([b.chosen_prob for b in baits]),
        },
        "critic_delta": {name: _bucket(name)
                         for name in ("loop_step", "other_bait", "other")},
        # PERCEPTION — expected FLAT or up. These were never the broken half; a fall here is a lost
        # belief, not a fixed policy.
        "readouts": {
            "beta_slot_accuracy": {
                "first_time": _rate_by([r for r in reads if r.first_time], "slot_correct"),
                "repeat": _rate_by([r for r in reads if not r.first_time], "slot_correct"),
                "loop_step": _rate_by([r for r in reads if r.loop_step], "slot_correct"),
            },
            "beta_species_correct": _rate_by(reads, "species_correct"),
            "alpha_switch_top1": {
                "all_pivots": _rate_by(reads, "alpha_top_is_switch"),
                "loop_steps": _rate_by([r for r in reads if r.loop_step], "alpha_top_is_switch"),
            },
            "alpha_switch_p_median_on_loop_steps": _median(
                [r.alpha_switch_p for r in reads if r.loop_step and r.alpha_switch_p is not None]),
        },
        # THE CONTROL: the same detector with the sides swapped. It measures the OPPONENT.
        "mirror": {
            "our_voluntary_pivots": sum(f.our_voluntary_pivots for f in folds),
            "whiff_rate_per_pivot": _ratio(sum(f.mirror_whiffs for f in folds),
                                           sum(f.mirror_moved_into for f in folds)),
            "loop_battle_rate": _ratio(sum(1 for f in folds if f.mirror_loop_battle), len(folds)),
        },
        "median_turns": _median([f.n_turns for f in folds if f.n_turns is not None]),
    }


def _discounted_returns(rewards: "list", gamma: float) -> "list[float]":
    """Realized discounted return from each decision to terminal:
    ``G_i = Σ_{j≥i} γ^{j−i} r_j`` (a None reward counts as 0). This is the
    Monte-Carlo value TARGET the critic is trained to predict, so V(s_i) vs G_i is
    a calibration residual."""
    n = len(rewards)
    out = [0.0] * n
    acc = 0.0
    for i in range(n - 1, -1, -1):
        acc = (rewards[i] if rewards[i] is not None else 0.0) + gamma * acc
        out[i] = acc
    return out


def _reliability_curve(values, returns, n_bins: int) -> "list[dict]":
    """Equal-count reliability bins over (V, G) pairs, ascending by V. Per bin:
    ``v_lo/v_hi/v_mean/g_mean/n`` and ``gap = v_mean − g_mean`` (the critic's
    SYSTEMATIC over-valuation at that value level; >0 ⇒ V runs above realized G).
    Binning by V (not by outcome) is what makes it selection-free — a loss-only
    V−G is biased positive by construction (losses are the below-V tail)."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(returns, dtype=float)
    if v.size == 0:
        return []
    order = np.argsort(v, kind="stable")
    v, g = v[order], g[order]
    bins = []
    for idx in np.array_split(np.arange(v.size), min(max(1, n_bins), v.size)):
        if idx.size == 0:
            continue
        vb, gb = v[idx], g[idx]
        bins.append({
            "v_lo": float(vb.min()), "v_hi": float(vb.max()),
            "v_mean": float(vb.mean()), "g_mean": float(gb.mean()),
            "n": int(idx.size), "gap": float(vb.mean() - gb.mean()),
        })
    return bins


def _reliability_gap_at(bins: "list[dict]", v: float) -> "float | None":
    """The critic's systematic over-valuation (bin ``gap``) at value level ``v``:
    among the bins whose [v_lo, v_hi] contain ``v`` (equal-count bins can OVERLAP
    when V has ties at a boundary) the one whose center ``v_mean`` is nearest;
    if none contains ``v``, the nearest bin overall (clamps to the ends). None
    when there are no bins."""
    if not bins:
        return None
    containing = [b for b in bins if b["v_lo"] <= v <= b["v_hi"]]
    return min(containing or bins, key=lambda b: abs(b["v_mean"] - v))["gap"]


def _calibration_stats(values, returns) -> dict:
    """Aggregate critic calibration of recorded V against realized G:
    ``bias`` = E[V−G] (systematic over/under-valuation; ~0 if unbiased over BOTH
    outcomes), ``mae`` = E[|V−G|], ``ev`` = 1−Var(G−V)/Var(G) (how much of the
    return variance V explains), ``slope`` = OLS G~V (1 = calibrated spread, <1 =
    over-confident)."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(returns, dtype=float)
    if v.size == 0:
        return {"n": 0, "bias": None, "mae": None, "ev": None, "slope": None,
                "v_mean": None, "g_mean": None}
    resid = v - g
    var_g = float(np.var(g))
    var_v = float(np.var(v))
    return {
        "n": int(v.size),
        "bias": float(resid.mean()),
        "mae": float(np.abs(resid).mean()),
        "ev": (1.0 - float(np.var(resid)) / var_g) if var_g > 1e-9 else None,
        "slope": float(np.cov(v, g, bias=True)[0, 1] / var_v) if var_v > 1e-9 else None,
        "v_mean": float(v.mean()), "g_mean": float(g.mean()),
    }
