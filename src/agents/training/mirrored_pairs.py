"""MIRRORED TEAM PAIRS — the pair as the statistical unit of a head-to-head eval (T17, ``gen3_mirrored_pairs_v1``).

A MIRRORED PAIR is two games of one TEAM PAIRING ``(A, B)`` played from BOTH sides with the SAME battle
seed: game 1 the measured side pilots ``A`` against the opponent on ``B``, game 2 the measured side pilots
``B`` against the opponent on ``A``. Team-draw luck — "my team happened to counter yours" — then cancels
inside the pair instead of riding the win rate as noise (or, at small ``n``, as bias).

🚨 **THE PAIR IS THE UNIT, never the game.** The two games of a pair share their teams and their dice, so
they are not independent draws: a per-game binomial CI on them states a precision the data does not have
(it is wrong in either direction depending on the sign of the within-pair correlation). Every interval and
every decision made on mirrored games is taken over PAIRS — the pentanomial below.

**The pentanomial.** A game is worth ``0`` (loss), ``1`` (draw) or ``2`` (win) HALF-POINTS to the measured
side, so a pair is worth ``0..4`` half-points — five categories. ``pair_counts[c]`` is how many pairs scored
``c``. The pair's normalized SCORE is ``c / 4`` (a per-game score in ``[0, 1]``, a draw worth ½ — the
Fishtest convention). A draw is not a win in ``win_rate`` (``n_won / n_finished``); the score and the win
rate differ by ``½ · draw rate`` (measured ≈ 0 in eval: 0 draws in 145k archived traces), and a reader is
told which one it holds.

Pure stdlib + numpy-free on purpose: the eval workers, the Rust eval core, the offline meters and the SPRT
all import it.
"""
from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Sequence, Tuple

SCHEMA = "gen3_mirrored_pairs_v1"

#: The five pair categories, in half-points (0 = both games lost … 4 = both won).
N_CATEGORIES = 5

#: Half-points of one game to the measured side.
WIN_POINTS, DRAW_POINTS, LOSS_POINTS = 2, 1, 0


def game_points(won: bool, lost: bool) -> int:
    """A finished game's half-points to the measured side: win 2, draw (neither won nor lost) 1, loss 0."""
    if won and lost:
        raise ValueError("a game cannot be both won and lost")
    return WIN_POINTS if won else LOSS_POINTS if lost else DRAW_POINTS


def result_points(result: str) -> int:
    """Half-points from a ``trace_result`` result (``WIN`` / ``LOSS`` / ``DRAW``)."""
    from agents.training.trace_result import DRAW, LOSS, WIN

    try:
        return {WIN: WIN_POINTS, DRAW: DRAW_POINTS, LOSS: LOSS_POINTS}[result]
    except KeyError:
        raise ValueError(f"unknown game result {result!r}") from None


def empty_counts() -> List[int]:
    return [0] * N_CATEGORIES


def pair_counts(points: Sequence[Optional[int]]) -> List[int]:
    """The pentanomial counts of a sequence of per-GAME half-points in PAIR ORDER (games ``2k``, ``2k+1``
    are pair ``k``). A pair with a missing game (``None`` — a timed-out / unfinished game) is DROPPED whole,
    never half-counted: half a pair carries the team-draw luck the pair exists to cancel. An odd trailing
    game is refused — pairs are even by construction."""
    if len(points) % 2:
        raise ValueError(f"{len(points)} games is not a whole number of mirrored pairs")
    out = empty_counts()
    for k in range(0, len(points), 2):
        a, b = points[k], points[k + 1]
        if a is None or b is None:
            continue
        for p in (a, b):
            if p not in (0, 1, 2):
                raise ValueError(f"game half-points must be 0, 1 or 2, got {p!r}")
        out[a + b] += 1
    return out


def add_counts(*vecs: Optional[Sequence[int]]) -> Optional[List[int]]:
    """Element-wise sum of pentanomial vectors; ``None`` entries are skipped, all-``None`` gives ``None``."""
    present = [v for v in vecs if v is not None]
    if not present:
        return None
    out = empty_counts()
    for v in present:
        if len(v) != N_CATEGORIES:
            raise ValueError(f"a pentanomial vector has {N_CATEGORIES} entries, got {len(v)}")
        for i, c in enumerate(v):
            out[i] += int(c)
    return out


def n_pairs(counts: Sequence[int]) -> int:
    return int(sum(counts))


def pair_score(counts: Sequence[int]) -> Optional[float]:
    """The mean per-game SCORE over the pairs (draw = ½), or ``None`` with no pairs."""
    n = n_pairs(counts)
    if not n:
        return None
    return sum(c * i for i, c in enumerate(counts)) / (4.0 * n)


def pair_score_ci(counts: Sequence[int], z: float = 1.96) -> Optional[Tuple[float, float, float]]:
    """``(score, lo, hi)`` — the mean per-game score with a PAIR-LEVEL normal interval: the variance is the
    sample variance of the per-pair score ``c/4`` over pairs (the pentanomial), divided by the PAIR count.
    ``None`` with no pairs; a single pair (or zero spread) gives a zero-width interval — honest about what
    was measured, and the reason a reader is told ``n_pairs`` beside it."""
    n = n_pairs(counts)
    if not n:
        return None
    mu = pair_score(counts)
    assert mu is not None
    var = sum(c * (i / 4.0 - mu) ** 2 for i, c in enumerate(counts)) / n
    se = math.sqrt(var / n)
    return mu, max(0.0, mu - z * se), min(1.0, mu + z * se)


def pair_bootstrap_ci(counts: Sequence[int], *, draws: int = 2000, seed: int = 0,
                      level: float = 0.95) -> Optional[Tuple[float, float, float]]:
    """``(score, lo, hi)`` — a PAIR-resampling percentile bootstrap of the mean per-game score (pairs are
    the exchangeable unit). Deterministic for a given ``seed``."""
    n = n_pairs(counts)
    if not n:
        return None
    pts = [i for i, c in enumerate(counts) for _ in range(int(c))]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(pts) for _ in range(n)) / (4.0 * n) for _ in range(int(draws)))
    a = (1.0 - level) / 2.0
    lo = means[max(0, int(math.floor(a * draws)))]
    hi = means[min(len(means) - 1, int(math.ceil((1.0 - a) * draws)) - 1)]
    return pair_score(counts), lo, hi  # type: ignore[return-value]


def summary(counts: Optional[Sequence[int]]) -> Optional[dict]:
    """The JSON block a record carries for one opponent's mirrored pairs (``None`` = not mirrored)."""
    if counts is None:
        return None
    ci = pair_score_ci(counts)
    return {"schema": SCHEMA, "pair_counts": [int(c) for c in counts], "n_pairs": n_pairs(counts),
            "score": None if ci is None else round(ci[0], 6),
            "score_ci95": None if ci is None else [round(ci[1], 6), round(ci[2], 6)]}


def pooled(blocks: Iterable[Optional[Sequence[int]]]) -> Optional[List[int]]:
    """Pool several opponents' pentanomials (e.g. the pool's sentinels) — ``add_counts`` over an iterable."""
    return add_counts(*list(blocks))
