"""THE RUN FLOOR ON THE HEAD-TO-HEAD SCALE: σ_h from a round-robin of runs' finals.

X5 design §7.3 (P0). Two independent runs of one recipe differ in strength; the head-to-head meter reads that
difference. This module turns a round-robin of edges (``main.h2h.stats.edge_summary`` rows) into σ_h, the SD
of a run's strength on this scale, with the METER's own variance subtracted — never left in.

THE SCALE AND THE MODEL. An edge is the player's win rate against the opponent, in percentage points above 50:
``d_ij = 100 (score_ij - 1/2)``. For runs with strengths ``s_i`` (pp of head-to-head win rate: small
differences are additive, ``d_ij = s_i - s_j``) the model is

    d_ij = s_i - s_j + u + e_ij,      s_i ~ (mu, sigma_h^2) independent,   e_ij ~ (0, v_ij) independent,

where ``v_ij`` is the edge's squared pair-clustered standard error (``stats.pair_se``) and ``u`` is the SEAT
EFFECT: the player keeps seat p1 in both games of a pair (``rust_eval.seeds.pair_game``), so a seat advantage
adds to every edge and an unmirrored-seat comparison would carry it. ``u`` is measured, not assumed — a
checkpoint against ITSELF has ``s_i - s_j = 0`` — and subtracted from every edge before anything else.

THE ESTIMATOR (a complete round-robin of K runs, each pair played at least once; the edges of a pair are
combined by inverse variance after orienting them). With ``dbar_ij`` the combined, ``u``-corrected edge and
``dbar_ji = -dbar_ij``:

    s_hat_i = (1/K) sum_{j != i} dbar_ij              (the least-squares strengths; they sum to 0)
    sigma_hat^2 = ( sum_i s_hat_i^2 - (2/K^2) sum_{edges} v_e ) / (K - 1)

``E[sum_i s_hat_i^2] = (K-1) sigma_h^2 + (2/K^2) sum_e v_e`` — the second term is the meter's variance riding
the strengths, subtracted exactly. For K = 2 it is the note's per-edge formula ``(d^2 - v) / 2``.

THE INTERVAL. With equal edge variance ``v`` and Gaussian strengths, ``s_hat ~ N(0, (sigma^2 + v/K)(I - J/K))``,
so ``sum_i s_hat_i^2 / (sigma^2 + v/K) ~ chi^2_{K-1}`` EXACTLY, and the interval for ``sigma^2`` is the chi-square
one for ``sigma^2 + v/K`` shifted down by ``v/K`` (floored at 0). The edges' variances differ by a few percent,
so ``v`` is their mean — an approximation, stated. With K - 1 = 1 or 2 degrees of freedom it is WIDE, and the
one-sided upper bound is the number a power calculation should respect. ``sigma_hat^2 < 0`` (the data are
inside the meter's noise) is reported as such and the point estimate floored at 0.

WHAT IT DOES NOT CARRY. (1) ``K`` here counts RUNS, not edges: the edges of one run are not independent. (2) The
runs must be replicates of one recipe — the caller chooses the set and says what differs (a different N or pin
is a LEVER inside the estimate, not noise). (3) Normality of the strengths cannot be checked at 2-3 df.
(4) The meter reads the PLAYED regime (greedy vs greedy, mirrored, the trainee's team distribution), so
sigma_h is a run floor of THAT meter.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class EdgeRead:
    """One edge in pp: ``player`` (the measured side, seat p1) vs ``opponent``; ``d`` = 100 (score - 1/2),
    ``v`` = the squared pair-clustered standard error in pp^2."""

    player: str
    opponent: str
    d: float
    v: float
    pairs: int = 0


def edge_read(player: str, opponent: str, summary: Mapping[str, Any]) -> EdgeRead:
    """An :class:`EdgeRead` from ``stats.edge_summary`` output."""
    if summary.get("score") is None or summary.get("se_pp") is None:
        raise ValueError(f"{player} vs {opponent}: an edge with no score / SE cannot be read")
    return EdgeRead(player, opponent, 100.0 * (float(summary["score"]) - 0.5), float(summary["se_pp"]) ** 2,
                    int(summary["pairs"]))


def seat_effect(self_edges: Sequence[EdgeRead]) -> Dict[str, Any]:
    """``u`` from edges of a run against ITSELF: the inverse-variance mean of their ``d`` and its SE. EMPTY
    input gives ``u = 0`` with ``measured = False`` — the caller must say so."""
    own = [e for e in self_edges if e.player == e.opponent]
    if not own:
        return {"u": 0.0, "se": None, "n_edges": 0, "measured": False}
    w = [1.0 / e.v for e in own]
    u = sum(wi * e.d for wi, e in zip(w, own)) / sum(w)
    return {"u": u, "se": math.sqrt(1.0 / sum(w)), "n_edges": len(own), "measured": True}


def seat_effect_from_reversals(reads: Sequence[EdgeRead]) -> Dict[str, Any]:
    """``u`` from edges played in BOTH directions: ``d_ij + d_ji = 2u + (e_ij + e_ji)`` since the strengths
    cancel. Returns the inverse-variance mean of ``(d_ij + d_ji) / 2`` over the pairs held in both
    directions; ``measured = False`` when no pair is."""
    by = {(e.player, e.opponent): e for e in reads if e.player != e.opponent}
    vals: List[Tuple[float, float]] = []
    for (i, j), e in by.items():
        if i < j and (j, i) in by:
            f = by[(j, i)]
            vals.append(((e.d + f.d) / 2.0, (e.v + f.v) / 4.0))
    if not vals:
        return {"u": 0.0, "se": None, "n_pairs_both_ways": 0, "measured": False}
    w = [1.0 / v for _d, v in vals]
    u = sum(wi * d for wi, (d, _v) in zip(w, vals)) / sum(w)
    return {"u": u, "se": math.sqrt(1.0 / sum(w)), "n_pairs_both_ways": len(vals), "measured": True}


def combine_pairs(runs: Sequence[str], reads: Sequence[EdgeRead], u: float = 0.0) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """``{(i, j): (dbar_ij, v_ij)}`` for every unordered pair of ``runs`` (``i`` before ``j`` in ``runs``): the
    ``u``-corrected edges of the pair, oriented ``i`` over ``j`` and combined by inverse variance. A pair with
    no edge is a ``ValueError`` — the estimator needs a complete round-robin."""
    order = {r: k for k, r in enumerate(runs)}
    acc: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for e in reads:
        if e.player == e.opponent or e.player not in order or e.opponent not in order:
            continue
        i, j = (e.player, e.opponent) if order[e.player] < order[e.opponent] else (e.opponent, e.player)
        d = (e.d - u) if e.player == i else -(e.d - u)
        acc.setdefault((i, j), []).append((d, e.v))
    out: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for i, j in itertools.combinations(runs, 2):
        obs = acc.get((i, j))
        if not obs:
            raise ValueError(f"no edge between {i} and {j}: the run-floor estimator needs a complete round-robin")
        w = [1.0 / v for _d, v in obs]
        out[(i, j)] = (sum(wi * d for wi, (d, _v) in zip(w, obs)) / sum(w), 1.0 / sum(w))
    return out


def run_strengths(runs: Sequence[str], pairs: Mapping[Tuple[str, str], Tuple[float, float]]) -> Dict[str, float]:
    """The least-squares strengths of a complete round-robin (they sum to 0)."""
    k = len(runs)
    s = {r: 0.0 for r in runs}
    for (i, j), (d, _v) in pairs.items():
        s[i] += d / k
        s[j] -= d / k
    return s


def _chi2_ppf(q: float, df: int) -> float:
    from scipy import stats

    return float(stats.chi2.ppf(q, df))


def sigma_h(runs: Sequence[str], reads: Sequence[EdgeRead], u: float = 0.0, level: float = 0.95) -> Dict[str, Any]:
    """σ_h over ``runs`` (module docs). Returns the strengths, the raw spread, the subtracted meter variance,
    σ̂² and σ̂ (floored at 0), df = K - 1 and the interval: two-sided at ``level`` and the one-sided upper
    bound at ``level`` (both on σ, pp)."""
    k = len(runs)
    if k < 2:
        raise ValueError("a run floor needs at least two runs")
    pairs = combine_pairs(runs, reads, u)
    s = run_strengths(runs, pairs)
    df = k - 1
    ss = sum(x * x for x in s.values())
    noise = (2.0 / (k * k)) * sum(v for _d, v in pairs.values())
    vbar = sum(v for _d, v in pairs.values()) / len(pairs)
    s2 = ss / df                                   # E = sigma^2 + v/K
    est2 = (ss - noise) / df
    a = (1.0 - level) / 2.0
    lo2 = df * s2 / _chi2_ppf(1.0 - a, df) - vbar / k
    hi2 = df * s2 / _chi2_ppf(a, df) - vbar / k
    up2 = df * s2 / _chi2_ppf(1.0 - level, df) - vbar / k
    root = lambda x: math.sqrt(max(0.0, x))        # noqa: E731 - a floor at zero, not a model
    edges = {f"{i} - {j}": {"d_pp": d, "se_pp": math.sqrt(v), "sigma_hat_sq": (d * d - v) / 2.0}
             for (i, j), (d, v) in pairs.items()}
    return {
        "runs": list(runs), "k": k, "df": df, "u_subtracted_pp": u, "strengths_pp": s,
        "sum_sq_strengths": ss, "meter_variance_subtracted": noise, "mean_edge_variance": vbar,
        "sigma_hat_sq": est2, "sigma_hat": root(est2), "inside_meter_noise": est2 <= 0.0,
        "sigma_ci": [root(lo2), root(hi2)], "sigma_upper_one_sided": root(up2), "level": level,
        "raw_sd_of_strengths": math.sqrt(s2), "edges": edges,
        "max_abs_pairwise_pp": max(abs(d) for d, _v in pairs.values()),
    }
