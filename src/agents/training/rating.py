"""Rating-model seam — the extension point for skill rating (ELO today, Glicko-2 / TrueSkill later).

The live skill rating is an anchored Bradley-Terry ELO (`elo.py`), fit in one global batch over the
win-records every eval cycle produces. That works, but it bakes in two BT-specific assumptions that
a future Glicko-2 / TrueSkill cannot meet: (1) a *global batch* fit (Glicko-2 updates sequentially
by rating period, carrying each player's RD forward), and (2) "pinned ratings" as the anchoring
mechanism (no clean Glicko analog). So rather than prematurely route the live path through a
one-size abstraction, this module defines the **data** and **interface** a new model plugs into,
implements the *batch* family (ELO/BT) against it, and documents — but does NOT build — the
*sequential* family. The live `record_elo` path is deliberately unchanged (zero risk); this seam is
the ready drop-in for when a different rating model is actually wanted.

What's here:

  - ``MatchRecord``  — one (player_a vs player_b) pairing over a set of games in one rating period.
                       Carries exact counts + draws + optional priors, so it is sufficient for BT,
                       Glicko-2 AND TrueSkill (the extra fields default to None/0 and BT ignores them).
  - ``RatingResult`` — a model's output: ratings + per-player uncertainty + provenance.
  - ``RatingModel``  — the BATCH protocol: ``fit(records, pinned) -> RatingResult``. ELO/BT satisfy
                       it; Glicko-2/TrueSkill need the *sequential* sibling sketched at the bottom.
  - ``BradleyTerryRating`` — the anchored-BT implementation, delegating to ``elo.fit_pairwise`` so
                       its numbers are byte-identical to the live ELO fit (pinned by a test).
  - ``eval_rows_to_match_records`` — adapt the existing ``elo.EvalRow`` history into ``MatchRecord``s.
                       The reusable bridge a Glicko backfill starts from.

Data fidelity for a future backfill: ``eval_results.jsonl`` now carries the exact per-opponent
``counts`` ({name: [n_won, n_finished]}) — see ``snapshot.append_eval_result_row``. Recovering
counts from ``win_rate × n_games`` is exact only at full shard coverage; the stored counts are the
fidelity-preserving record (the win/loss sequence is not retained, which Glicko-2 tolerates — it
needs the period's aggregate W/L, not the order).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agents.training import elo as _elo


@dataclass
class MatchRecord:
    """One pairing of (``player_a`` vs ``player_b``) over ``n_games`` games in one rating period.

    Skill-model-agnostic — it carries the minimum union BT, Glicko-2 and TrueSkill all need:
    exact counts (``n_won_a`` / ``n_drawn``) + a ``period_id`` (Glicko-2's discrete epoch; the
    eval ``step`` is the natural one) + optional opponent priors (Glicko-2 RD / TrueSkill σ at
    period start). BT uses only ``player_a/player_b/n_games/n_won_a``; the rest default away so a
    BT fit over these records is identical to the current ``elo`` path. Player ids follow the elo
    convention: ``"bot:<name>"`` and ``"snap:<step>"`` (see ``elo.bot_key`` / ``elo.snap_key``)."""
    player_a: str
    player_b: str
    n_games: int
    n_won_a: int
    n_drawn: int = 0
    period_id: int | None = None
    # Priors for sequential / multi-period models (Glicko-2 RD, TrueSkill σ); None ⇒ model default.
    player_a_rating: float | None = None
    player_a_rd: float | None = None
    player_b_rating: float | None = None
    player_b_rd: float | None = None


@dataclass
class RatingResult:
    """A rating model's output. ``ratings``/``uncertainty`` are per-player dicts; ``uncertainty``
    is the standard error for BT, the rating deviation for Glicko-2, σ for TrueSkill — all "how
    sure are we" in the model's own units. ``base`` is the gauge/anchor reference point."""
    ratings: dict[str, float]
    uncertainty: dict[str, float]
    converged: bool = True
    algorithm: str = "bradley-terry"
    base: float = _elo.DEFAULT_BASE

    def rating_for(self, player: str) -> tuple[float, float] | None:
        """(rating, uncertainty) for ``player``, or None if absent from the fit."""
        if player not in self.ratings:
            return None
        return self.ratings[player], self.uncertainty.get(player, 0.0)


@runtime_checkable
class RatingModel(Protocol):
    """The BATCH rating interface: fit one global set of match records, return ratings.

    ELO / Bradley-Terry satisfy this directly. ``pinned`` fixes some players' ratings (the bot
    anchors that make snapshot ratings cross-run-comparable). Glicko-2 / TrueSkill are *sequential*
    (period-by-period RD carry-forward) and do NOT fit this single-batch shape — see the module
    footer for the ``SequentialRatingModel`` sketch they would implement instead."""

    def fit(self, records: list[MatchRecord],
            pinned: dict[str, float] | None = None) -> RatingResult: ...


class BradleyTerryRating:
    """Anchored Bradley-Terry ELO as a ``RatingModel`` — a thin adapter over ``elo.fit_pairwise``.

    Lowers each ``MatchRecord`` to the ``(player_a, player_b, n_won_a, n_games)`` pairwise form the
    solver already consumes and returns its ratings/SE unchanged, so a fit over records produced by
    ``eval_rows_to_match_records`` is **identical** to the live ``elo`` path (asserted in
    ``rating_test.py``). Draws are split half-and-half into the win count (BT has no native draw),
    matching the standard convention; with ``n_drawn == 0`` (the eval case) this is a no-op."""

    algorithm = "bradley-terry"

    def __init__(self, *, base: float = _elo.DEFAULT_BASE, prior_sd: float = _elo.DEFAULT_PRIOR_SD,
                 max_iter: int = 100, tol: float = 1e-7):
        self.base = base
        self.prior_sd = prior_sd
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, records: list[MatchRecord],
            pinned: dict[str, float] | None = None) -> RatingResult:
        # Draws fold as half-wins (the standard BT convention), rounded to an int because the
        # solver's edge counts are integer. Exact for the eval path (n_drawn == 0 → just n_won_a);
        # genuinely fractional draws would need elo._aggregate to accept float wins (a future item,
        # alongside the sequential Glicko model — see the module footer).
        results = [(r.player_a, r.player_b, int(round(r.n_won_a + 0.5 * r.n_drawn)), r.n_games)
                   for r in records]
        ratings, se, converged = _elo.fit_pairwise(
            results, pinned=pinned, base=self.base, prior_sd=self.prior_sd,
            max_iter=self.max_iter, tol=self.tol)
        return RatingResult(ratings=ratings, uncertainty=se, converged=converged,
                            algorithm=self.algorithm, base=self.base)


def eval_rows_to_match_records(rows: list["_elo.EvalRow"]) -> list[MatchRecord]:
    """Adapt the accumulated ``elo.EvalRow`` history into ``MatchRecord``s (one per pairing).

    Mirrors ``elo._rows_to_results`` exactly — trainee (``snap:<step>``) vs each bot
    (``bot:<name>``) and each sentinel (``snap:<sentinel_step>``), win count recovered as
    ``round(win_rate × n_games)`` — so the records feed any ``RatingModel`` and a BT fit over them
    reproduces the live ladder. ``period_id`` is the cycle step (the natural Glicko-2 rating
    period). When a future loader reads the exact ``counts`` field from ``eval_results.jsonl`` it
    can build records with the true ``n_won_a`` directly instead of the rounded rate."""
    out: list[MatchRecord] = []
    for row in rows:
        n = max(1, int(row.n_games))
        subj = _elo.snap_key(row.step)
        for name, wr in row.bots.items():
            out.append(MatchRecord(subj, _elo.bot_key(name), n,
                                   int(round(_clamp01(wr) * n)), period_id=int(row.step)))
        for m_step, wr in row.sentinels:
            out.append(MatchRecord(subj, _elo.snap_key(m_step), n,
                                   int(round(_clamp01(wr) * n)), period_id=int(row.step)))
    return out


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


# ── Future: the SEQUENTIAL family (Glicko-2 / TrueSkill) ───────────────────────────────────────
#
# Glicko-2 and TrueSkill are NOT batch-global like BT — they update a player's rating AND its
# uncertainty (RD / σ) one rating PERIOD at a time, carrying the post-period RD forward as the next
# period's prior. They therefore do not fit `RatingModel.fit(all_records)`; they want a sibling:
#
#   class SequentialRatingModel(Protocol):
#       def fit_period(self, period_id, records, priors: dict[str, PlayerState]) -> dict[str, PlayerState]: ...
#       def finalize(self) -> RatingResult: ...   # last-period states → the same RatingResult shape
#
# A `Glicko2Rating` would group `MatchRecord`s by `period_id`, seed period 0 from the bot anchors
# (their RD pinned small, the Glicko analog of `pinned`), and fold periods in step order. Everything
# it needs is already on `MatchRecord` (counts, draws, period_id, opponent priors) and in the
# enriched `eval_results.jsonl` (exact `counts`), so landing it is: implement `Glicko2Rating`, add
# the protocol above, and switch `record_elo` to construct the chosen model — no upstream re-plumbing.
# Deliberately not built now: it buys nothing until a rating model is actually wanted, and baking the
# sequential shape into the live path prematurely would be the wrong abstraction.
