"""The results schema — and the rule that gives this module its shape.

🚨 **A NUMBER NEVER LEAVES THIS TOOL WITHOUT ITS REGIME.** The 2026-09-14 matched-regime battery
established that "temperature 1.0" is not one setting across models — at T = 1.0 `SmallRL` plays
its own argmax 64.7% of the time and `SyntheticRLV2` 88.0%, so the same nominal knob perturbs 35%
of one policy's decisions and 12% of the other's, and the two models' temperature effects come out
with OPPOSITE SIGNS because of it. The de-risk's headline `0.742` fell to `0.520` on the
like-for-like matched cell, −22.2 pp [−34.1, −9.4]. A win rate reported without the regime it was
taken under is therefore not a weaker number, it is a **different** number that nobody can place.

So :class:`GameRow` carries, on EVERY row and not once per file:

* the regime, and the ``argmax_match_rate`` that VERIFIES it — the sampling cells are the positive
  control (an instrument reading 1.000 in both regimes would have no power);
* the team set, and the SIZE of both sides' team sources, so an asymmetric draw is visible;
* the opponent's name, version and commit — neither checkout is ours to pin;
* our checkpoint's resolved ``.zip``, its step count and how ``resolve_model_ref`` chose it;
* the search budget and the REALIZED visit count, because Foul Play's only budget is WALL CLOCK
  (UNDERSTANDING rule 23: a wall-clock budget on a contended box is a width meter, not a setting —
  realized width was 1.21–1.53 M visits/decision at a constant 1000 ms).

Intervals are **Wilson** for a single proportion and **Newcombe's hybrid-score interval** for a
difference. Ties count in the denominator and not the numerator, as both 2026-09-14 batteries did.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

Z95 = 1.959963985


def wilson(k: int, n: int, z: float = Z95) -> "tuple[float, float, float]":
    """``(p, lo, hi)`` — the Wilson score interval for ``k`` successes in ``n``.

    Wilson rather than Wald because every cell here is n ~= 100 and some sit near a boundary,
    where the normal approximation's coverage is simply wrong.
    """
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def newcombe(k1: int, n1: int, k2: int, n2: int) -> "tuple[float, float, float]":
    """95% CI for ``p1 - p2`` (Newcombe method 10, hybrid score).

    The difference interval CONSISTENT with the Wilson cell intervals, which is what the project's
    equivalence rule needs: a bar compared against a point estimate is vacuous, so an effect is
    reported with its OWN interval and is NOT DETECTED whenever that interval covers zero.
    """
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"), float("nan"))
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (d, max(-1.0, lo), min(1.0, hi))


@dataclass(frozen=True)
class CellSpec:
    """Everything that makes this cell THIS cell — stamped onto every row it produces.

    Frozen and dataclass-shaped on purpose: a row is built by :meth:`GameRow.build` from one of
    these plus the game's own outcome, so no code path can write a row that forgot the regime.
    """

    opponent: str                  # "metamon:SmallRL" / "foulplay"
    opponent_kind: str             # "metamon" / "foulplay"
    opponent_agent: str            # "SmallRL" / "" for foulplay
    opponent_version: str          # checkpoint number, engine version, whatever names the policy
    opponent_commit: str           # the checkout's git HEAD, short
    our_regime: str                # "greedy" / "t1"
    their_regime: str              # "greedy" / "t1" / "search:1000ms"
    regime_matched: bool           # both sides moved together?
    teamset: str                   # "home" / "away"
    our_team_source: str
    our_team_count: int
    their_team_source: str
    their_team_count: int
    battle_format: str
    model_spec: str                # what the CLI was given
    model_zip: str                 # what resolve_model_ref chose
    model_step: Optional[int]
    model_rung: str                # HOW it was chosen ("explicit_zip", "latest_txt", ...)
    search_time_ms: Optional[int]  # foulplay only
    search_parallelism: Optional[int]
    forfeit_turn_limit: int
    server_uri: str
    showdown_pin: str
    #: WHO our side is — "model" (a checkpoint through `main.play`) or "bot:<name>" (one of the
    #: nine pinned eval bots). A bot has no sampling knob at all, exactly like Foul Play, so a
    #: bot cell is stamped `regime_matched = False` and `our_regime = "bot:<name>"`; that is the
    #: honest label, not a defect, and it is the reason this field is on the ROW.
    our_side: str = "model"
    #: HOW the checkpoint was loaded: "bare" (`MaskablePPO.load`) or "foreign"
    #: (`load_foreign_opponent`, which verifies the arch_signature and reads the zip's own config).
    #: A cross-run frozen snapshot FAILS a bare load, so which loader ran is part of what the row
    #: says about the policy that played.
    model_loader: str = ""

    def stamp(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GameRow:
    """One battle, as OUR side saw it, with the whole cell stamped onto it."""

    cell: CellSpec
    half: str                      # "ours_challenge" / "peer_challenge"
    index: int                     # 1-based within the whole series
    battle_tag: str
    turns: int
    result: str                    # "win" / "loss" / "tie"
    won: Optional[bool]
    hit_forfeit_limit: bool
    our_team: Optional[str]
    n_decisions: Optional[int]
    n_defaults: Optional[int]
    n_redecides: Optional[int]
    t_finished: float
    # Regime VERIFICATION, series-level but written per row so a row is self-contained.
    our_stochastic_kwargs: List[bool] = field(default_factory=list)
    their_sample_kwargs: List[bool] = field(default_factory=list)
    their_argmax_match_rate: Optional[float] = None
    # Realized search WIDTH (foulplay). None everywhere else; never silently zero.
    their_visits_mean: Optional[float] = None
    their_visits_n: Optional[int] = None

    def to_json(self) -> Dict[str, Any]:
        row = self.cell.stamp()
        row.update({
            "half": self.half,
            "index": self.index,
            "battle_tag": self.battle_tag,
            "turns": self.turns,
            "result": self.result,
            "won": self.won,
            "hit_forfeit_limit": self.hit_forfeit_limit,
            "our_team": self.our_team,
            "n_decisions": self.n_decisions,
            "n_defaults": self.n_defaults,
            "n_redecides": self.n_redecides,
            "t_finished": self.t_finished,
            "our_stochastic_kwargs": self.our_stochastic_kwargs,
            "their_sample_kwargs": self.their_sample_kwargs,
            "their_argmax_match_rate": self.their_argmax_match_rate,
            "their_visits_mean": self.their_visits_mean,
            "their_visits_n": self.their_visits_n,
        })
        return row


#: Fields every written row MUST carry. Pinned by a test, because the point of this module is that
#: the regime cannot fall off a row — and "it is in the dataclass" is not the same as "it is in the
#: file" once anyone writes a second producer.
REQUIRED_ROW_FIELDS = (
    "opponent", "opponent_version", "opponent_commit",
    "our_side", "model_loader",
    "our_regime", "their_regime", "regime_matched",
    "teamset", "our_team_count", "their_team_count",
    "model_zip", "model_step", "model_rung",
    "search_time_ms", "their_visits_mean",
    "their_argmax_match_rate", "our_stochastic_kwargs",
    "result", "turns", "half", "index",
)


def write_games(path: Path, rows: Iterable[GameRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row.to_json()) + "\n")
            n += 1
    return n


def summarize(cell: CellSpec, rows: List[GameRow], *, status: str,
              failure: Optional[Dict[str, Any]] = None,
              provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The cell's one number, its Wilson interval, and every integrity counter beside it.

    ``status`` is ``"OK"`` or ``"FAILED"``; a FAILED summary still carries whatever games did
    finish, because "n games completed, then the peer died" is a far more useful report than a
    missing file, and a partial n is only honest when it is LABELLED partial.
    """
    wins = sum(1 for r in rows if r.result == "win")
    losses = sum(1 for r in rows if r.result == "loss")
    ties = sum(1 for r in rows if r.result == "tie")
    n = len(rows)
    p, lo, hi = wilson(wins, n)

    by_half: Dict[str, Dict[str, Any]] = {}
    for half in ("ours_challenge", "peer_challenge"):
        sub = [r for r in rows if r.half == half]
        hw = sum(1 for r in sub if r.result == "win")
        hp, hlo, hhi = wilson(hw, len(sub))
        by_half[half] = {"n": len(sub), "wins": hw, "win_rate": hp,
                         "wilson_lo": hlo, "wilson_hi": hhi}

    caps = sum(1 for r in rows if r.hit_forfeit_limit)
    turns = sorted(r.turns for r in rows)
    visits = [r.their_visits_mean for r in rows if r.their_visits_mean is not None]
    rates = sorted({r.their_argmax_match_rate for r in rows
                    if r.their_argmax_match_rate is not None})

    out: Dict[str, Any] = {
        "status": status,
        "cell": cell.stamp(),
        "n": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        # THE headline, and it is never reported alone — `cell` above carries the regime, the team
        # set, the opponent version and the search budget, and they travel together by construction.
        "win_rate": p,
        "wilson95": [lo, hi],
        "by_half": by_half,
        "mean_turns": (sum(turns) / n) if n else None,
        "median_turns": turns[n // 2] if n else None,
        "max_turns": turns[-1] if n else None,
        "hit_forfeit_limit": caps,
        "n_defaults": sum(r.n_defaults or 0 for r in rows),
        "n_redecides": sum(r.n_redecides or 0 for r in rows),
        "distinct_our_teams": len({r.our_team for r in rows if r.our_team}),
        # The realized search WIDTH, reported because the nominal budget is wall clock and
        # therefore not a reproducible opponent across boxes (UNDERSTANDING rule 23).
        "realized_visits_per_decision_mean": (sum(visits) / len(visits)) if visits else None,
        "realized_visits_samples": len(visits),
        "their_argmax_match_rates": rates,
        "team_source_asymmetry": cell.our_team_count != cell.their_team_count,
    }
    if failure is not None:
        out["failure"] = failure
    if provenance is not None:
        out["provenance"] = provenance
    return out


def write_summary(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


def render(summary: Dict[str, Any]) -> str:
    """The one block a human reads. Regime first, then the number — in that order on purpose."""
    cell = summary["cell"]
    n = summary["n"]
    lo, hi = summary["wilson95"]
    lines = [
        f"  opponent    {cell['opponent']} @ {cell['opponent_version']} "
        f"({cell['opponent_commit']})",
        f"  regime      ours={cell['our_regime']}  theirs={cell['their_regime']}  "
        f"matched={cell['regime_matched']}",
        f"  team set    {cell['teamset']}  (ours {cell['our_team_count']} teams / "
        f"theirs {cell['their_team_count']})",
        f"  our side    {cell.get('our_side', 'model')}"
        + (f"  loader={cell['model_loader']}" if cell.get("model_loader") else ""),
        f"  our model   {cell['model_zip'] or '(none — our side is not a checkpoint)'}"
        + (f" @ step {cell['model_step']} (resolved via {cell['model_rung']})"
           if cell['model_zip'] else ""),
    ]
    if cell.get("search_time_ms"):
        v = summary.get("realized_visits_per_decision_mean")
        lines.append(f"  search      {cell['search_time_ms']} ms/decision nominal, realized "
                     + (f"{v:,.0f} visits/decision" if v else "visits NOT RECORDED"))
    rates = summary.get("their_argmax_match_rates") or []
    if rates:
        lines.append("  verified    peer argmax_match_rate = "
                     + ", ".join(f"{r:.4f}" for r in rates))
    lines += [
        f"  status      {summary['status']}",
        f"  WIN RATE    {summary['win_rate']:.3f}   Wilson 95% [{lo:.3f}, {hi:.3f}]   "
        f"n={n} (W{summary['wins']}/L{summary['losses']}/T{summary['ties']})",
    ]
    if summary.get("failure"):
        lines.append(f"  FAILURE     {summary['failure'].get('cause')}: "
                     f"{summary['failure'].get('detail')}")
    return "\n".join(lines)
