"""Reading the head-to-head rows: pool an edge's batches into one pentanomial, its win rate and a PAIR-clustered
interval. Pure stdlib (``agents.training.mirrored_pairs`` for the pentanomial); the rows come from a declared
``eval_ledger.read`` (v2 views: a v1 row's near-tie COUNT is ``compute.near_tie_game_count`` there).

THE PAIR IS THE UNIT. The win rate of a mirrored read is the mean per-GAME score over PAIRS (a draw worth 1/2 —
the Fishtest convention; it equals ``W / games`` when there are no draws, and the row says how many there are).
Its interval is the pentanomial one over pairs (``mirrored_pairs.pair_score_ci``, 95 % normal); a per-game
binomial interval on the same games would treat the two games of a pair as independent draws. ``se`` here is the
UNBIASED sample standard error over pairs (``/ (n - 1)``), the quantity the run-floor estimator subtracts.

A reader REFUSES to pool across regimes (design_evaluation.md §0c rule 2): rows of one edge must carry one
``regime_id``.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.training import mirrored_pairs as MP


class MixedRegimeError(ValueError):
    """Rows of different regimes (or different players) offered to one pooled estimate."""


def pair_se(counts: Sequence[int]) -> Optional[float]:
    """Unbiased standard error of the mean per-pair score (score units, 0..1); ``None`` below two pairs."""
    n = MP.n_pairs(counts)
    if n < 2:
        return None
    mu = MP.pair_score(counts)
    assert mu is not None
    ss = sum(c * (i / 4.0 - mu) ** 2 for i, c in enumerate(counts))
    return math.sqrt(ss / (n - 1) / n)


def pair_sd(counts: Sequence[int]) -> Optional[float]:
    """Unbiased standard deviation of the per-pair score (score units)."""
    n = MP.n_pairs(counts)
    if n < 2:
        return None
    mu = MP.pair_score(counts)
    assert mu is not None
    return math.sqrt(sum(c * (i / 4.0 - mu) ** 2 for i, c in enumerate(counts)) / (n - 1))


def edge_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    return row["player"]["sha256"], row["opponent"]["sha256"]


def group_edges(rows: Sequence[Mapping[str, Any]], regime_id: Optional[str] = None
                ) -> Dict[Tuple[str, str], List[Mapping[str, Any]]]:
    """``{(player sha, opponent sha): rows}``. Within an edge more than one ``regime_id`` is a
    :class:`MixedRegimeError` unless ``regime_id`` picks one."""
    out: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        if regime_id is not None and r["regime"]["regime_id"] != regime_id:
            continue
        out[edge_key(r)].append(r)
    for k, rs in out.items():
        ids = sorted({r["regime"]["regime_id"] for r in rs})
        if len(ids) > 1:
            raise MixedRegimeError(f"edge {rs[0]['player']['id']} vs {rs[0]['opponent']['id']} holds rows of "
                                   f"{len(ids)} regimes {ids}; name one with regime_id (rows of different regimes "
                                   "are never pooled)")
    return dict(out)


def edge_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The pooled read of ONE edge's rows (same players, one regime): counts, the pentanomial, the win rate
    and its pair-clustered 95 % interval, the throughput and the near-tie census."""
    if not rows:
        raise ValueError("an edge needs at least one row")
    if len({edge_key(r) for r in rows}) != 1:
        raise MixedRegimeError("rows of more than one (player, opponent) edge")
    if len({r["regime"]["regime_id"] for r in rows}) != 1:
        raise MixedRegimeError("rows of more than one regime")
    counts = MP.add_counts(*[r["pairs"]["counts"] for r in rows]) or MP.empty_counts()
    w = sum(r["counts"]["w"] for r in rows)
    l = sum(r["counts"]["l"] for r in rows)
    d = sum(r["counts"]["d"] for r in rows)
    games = w + l + d
    ci = MP.pair_score_ci(counts)
    se = pair_se(counts)
    wall = sum(float(r["compute"].get("wall_s") or 0.0) for r in rows)
    return {
        "player": rows[0]["player"]["id"], "opponent": rows[0]["opponent"]["id"],
        "player_sha256": rows[0]["player"]["sha256"], "opponent_sha256": rows[0]["opponent"]["sha256"],
        "regime_id": rows[0]["regime"]["regime_id"], "purpose": sorted({r["purpose"] for r in rows}),
        "batches": len(rows), "pairs": MP.n_pairs(counts), "games": games, "w": w, "l": l, "d": d,
        "pair_counts": [int(c) for c in counts],
        "win_rate": (w / games) if games else None,
        "score": ci[0] if ci else None, "score_ci95": [ci[1], ci[2]] if ci else None,
        "se_pp": 100.0 * se if se is not None else None,
        "pair_sd": pair_sd(counts),
        "wall_s": wall, "games_per_s": (games / wall) if wall > 0 else None,
        "near_tie_games": sum(int(r["compute"].get("near_tie_game_count") or 0) for r in rows),
        "near_tie_decisions": sum(int(r["compute"].get("near_tie_decisions") or 0) for r in rows),
        "devices": sorted({f"{r['compute'].get('device')}/{r['compute'].get('backend')}" for r in rows}),
    }


def format_edge(s: Mapping[str, Any]) -> str:
    """One human line for an edge summary."""
    if not s["pairs"]:
        return f"{s['player']} vs {s['opponent']}: no pairs"
    lo, hi = s["score_ci95"]
    gps = f"; {s['games_per_s']:.2f} games/s" if s.get("games_per_s") else ""
    se = f"{s['se_pp']:.3f}" if s.get("se_pp") is not None else "n/a"
    return (f"{s['player']} vs {s['opponent']}: score {100 * s['score']:.2f}% [{100 * lo:.2f}, {100 * hi:.2f}] "
            f"(pair-clustered 95%), SE {se} pp over {s['pairs']} pairs / {s['games']} games; "
            f"W/L/D {s['w']}/{s['l']}/{s['d']}; pentanomial {s['pair_counts']}{gps}")
