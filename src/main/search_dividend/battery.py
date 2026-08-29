"""The evaluation battery: play matched games per (arm, budget, opponent) and append the rows.

**Matched games are the whole point.** Three arms are compared, so every cell must play the SAME
battles: the same opponent, the same team draw, the same dice. The driver therefore derives a
per-game SEED from ``(opponent, game_index, --games-seed)`` and pins it on the bridge ``START``,
and it draws both teams from a per-game RNG rather than the players' own teambuilders. Two arms
that disagree on a game then disagree about the DECISION, not about the battle they were handed —
without that, an arm difference at N=10 is mostly team-draw noise (the exploiter work measured
exactly that trap: an apparent edge that vanished under an equal-pilot mirror).

**The MIRROR mode, and why it needs side-swap.** ``--opponents self`` plays the searched side
against the SAME network with search structurally off, so the two sides differ in exactly one
thing. That pins the no-effect point at 50% by construction — unlike the scripted roster, which
saturates near 90% and hides a dividend in its ceiling. But a mirror game's TEAM DRAW is
asymmetric, and at small n that asymmetry is most of the variance: the searched side simply gets a
better or worse team. ``--side-swap`` plays every game index in BOTH orientations off one pinned
seed, so the summary can difference the pair and report a win rate the team draw cannot fake.

**Resumable, append-only.** Each finished game appends one JSON line. A relaunch reads the file,
counts the ``(arm, budget, opponent)`` cells' finished ORIENTATION-GAMES, and plays only what is
missing. The file is never rewritten — the same discipline ``eval_results.jsonl`` follows, and for
the same reason: a summary can always be recomputed from rows, but a row lost to a crashed rewrite
is gone.

**Every row carries an outcome XOR a named error** (:func:`finalize_row`). A row that says nothing
went wrong on a battle that never happened is the one shape this file must not be able to hold.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from main.search_dividend.defensive import fold_defensive
from main.search_dividend.player import SearchDividendPlayer, play_one_battle
from main.search_dividend.playoff import PlayoffConfig, PlayoffRunner, fold_playoff
from main.search_dividend.racing import fold_racing
from main.search_dividend.record import install_choice_tap
from main.search_dividend.search import SearchConfig, SearchEngine

BATTLE_FORMAT = "gen3ou"
# v2 ADDED `orientation` / `tied` / `turns` / `max_depth` and let `result` be ``"tie"``. Every
# reader defaults the new keys, so a v1 file still summarizes and still resumes — an append-only
# file outlives its schema by construction, and a version that could not be read back would make
# the append-only discipline pointless.
#
# v3 adds NO field. It is a MEANING change: before it, dice draw 0 was the sim's ``"original"``
# seed — the realized stream, which reproduced the actual turn byte-for-byte in 11 of 12 live
# decisions — so every score mixed one clairvoyant ply with `R-1` honest ones and a cell's reading
# tracked its realized `r_dice`. A version bump is the only thing that can tell those rows apart
# from these in an append-only file, and `summary.format_report` prints a banner when it sees one.
ROW_VERSION = 3


def game_seed(opponent: str, game: int, salt: int) -> str:
    """The sim seed for one CELL-INDEPENDENT game.

    Derived from ``(opponent, game, salt)`` and NOT from the arm or the budget, which is what
    makes the arms matched: cell (base, 1 s, heuristic) game 7 and cell (oracle, 3 s, heuristic)
    game 7 are the same dice."""
    h = hashlib.sha256(f"{salt}|{opponent}|{game}".encode()).hexdigest()
    return "sodium," + h[:32]


def team_pair(opponent: str, game: int, salt: int, packed_teams: Sequence[str],
              orientation: int = 0) -> tuple:
    """The (our, their) packed teams for one game-orientation — matched across arms as above.

    **Orientation 1 SWAPS them, and that is the whole point of side-swap pairing.** A mirror
    game's team draw is asymmetric: two copies of one network pilot two different teams, so at
    small n the reported win rate is mostly a measurement of which team is stronger. Playing each
    ``game`` in both orientations lets the summary difference them out — the same paired design the
    exploiter work needed when an apparent edge turned out to be pure team draw. The SEED is
    deliberately NOT varied with orientation, so the pair starts from one dice stream.
    """
    rng = random.Random(f"{salt}|{opponent}|{game}|teams")
    a, b = rng.choice(list(packed_teams)), rng.choice(list(packed_teams))
    return (a, b) if not int(orientation) else (b, a)


#: The MIRROR opponent's name. ``--opponents self`` means "the same network with search off" —
#: the sensitive contrast, because the scripted roster saturates near 90% and pins nothing, while
#: a mirror pins the no-effect point at exactly 50% by construction.
MIRROR = "self"


@dataclass
class Cell:
    arm: str
    budget: float
    opponent: str

    def key(self) -> str:
        return f"{self.arm}|{self.budget:g}|{self.opponent}"

    @property
    def is_mirror(self) -> bool:
        return self.opponent == MIRROR


def row_unit(row: dict) -> tuple:
    """A row's ``(game, orientation)`` unit of account, defaulting a pre-side-swap row to 0."""
    return (int(row["game"]), int(row.get("orientation", 0) or 0))


def _cell_tag(cell: Cell) -> str:
    """A short, STABLE account-name suffix for a cell.

    ``hash()`` on a str is salted per process (``PYTHONHASHSEED``), so using it here would give
    the same cell a different account name on every relaunch — which is invisible on the bridge
    (no server, no matchmaking) right up until someone points this driver at ``--impl off`` and
    spends an afternoon on a username collision."""
    return hashlib.md5(cell.key().encode()).hexdigest()[:6]


class ResultsFile:
    """Append-only JSONL of per-game rows, with the resume index built by reading it back."""

    def __init__(self, path: str):
        self.path = path
        self._done: Dict[str, set] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A row truncated by a kill mid-write. Skipping it is right — it will simply
                    # be replayed — but a SILENT skip would make a short file look complete, so
                    # the loader stops at the first bad line rather than reading past it.
                    break
                self._index(row)

    def _index(self, row: dict) -> None:
        """A game counts as DONE only when it FINISHED. An unfinished row (a crashed bridge
        child, a killed process, the 2026-08-23 pruned-worktree incident that scored 8 games in
        0.1 s each) stays in the file as evidence but is REPLAYED on resume — the battery's unit
        of account is finished games per cell, and marking a crash done would quietly shrink a
        cell's n. A deterministic crash then costs one extra error row per relaunch, which is
        bounded and visible, against silently under-powered cells, which is neither.

        The unit is the ORIENTATION-GAME, not the game: under side-swap a ``game`` is two battles
        and finishing one of them finishes half a pair. Rows written before side-swap existed
        carry no ``orientation`` and read as 0, so an old file resumes exactly as it always did."""
        if not int(row.get("finished", 0)):
            return
        k = Cell(row["arm"], float(row["budget"]), row["opponent"]).key()
        self._done.setdefault(k, set()).add(row_unit(row))

    def done_units(self, cell: Cell) -> set:
        """The ``(game, orientation)`` pairs already finished for this cell."""
        return set(self._done.get(cell.key(), ()))

    def done_games(self, cell: Cell) -> set:
        """The game INDEXES with at least one finished orientation (the pre-side-swap view)."""
        return {g for (g, _o) in self._done.get(cell.key(), ())}

    def n_done(self, cell: Cell) -> int:
        """How many orientation-games are finished — the battery's unit of account."""
        return len(self._done.get(cell.key(), ()))

    def append(self, row: dict) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._index(row)

    def rows(self) -> List[dict]:
        out: List[dict] = []
        if not os.path.exists(self.path):
            return out
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        break
        return out


def summarize_decisions(decisions: Sequence[dict]) -> dict:
    """Fold one game's per-decision rows into the counters the report needs.

    ``fallbacks`` is a HISTOGRAM by reason, never a single total: "the search fell back" is not a
    finding, "the search fell back because every determinized world failed the prefix gate" is."""
    fallbacks: Dict[str, int] = {}
    # `depth` and `beam` are a schema ADDITION, not a fork: ladder requirement 3 says the ladder's
    # per-decision search trace is THIS format extended, never a second one, so a new signal joins
    # the dict a reader already walks.
    realized = {"m_opp": [], "k_worlds": [], "r_dice": [], "arms": [], "elapsed": [],
                "depth": [], "beam": []}
    errors: List[str] = []
    n_changed = 0
    n_searched = 0
    deepened = 0
    truncated = 0
    gate_failed = 0
    for d in decisions:
        fb = d.get("fallback")
        if fb:
            fallbacks[fb] = fallbacks.get(fb, 0) + 1
            det = d.get("error_detail")
            if det and det not in errors and len(errors) < 3:
                errors.append(det)
            continue
        n_searched += 1
        n_changed += 1 if d.get("changed") else 0
        w = d.get("widths") or {}
        realized["m_opp"].append(w.get("opp_candidates", 0))
        realized["k_worlds"].append(w.get("worlds_gated_ok", 0))
        realized["r_dice"].append(w.get("dice", 0))
        realized["arms"].append(w.get("arms_scored", 0))
        realized["elapsed"].append(w.get("elapsed_s", 0.0))
        realized["depth"].append(w.get("depth_realized", 1))
        realized["beam"].append(w.get("beam_m", 0))
        deepened += 1 if int(w.get("depth_realized", 1) or 1) > 1 else 0
        truncated += 1 if w.get("deadline_truncated") else 0
        gate_failed += int(w.get("worlds_gate_failed", 0))
    return {
        "n_decisions": len(decisions),
        "n_searched": n_searched,
        "n_changed": n_changed,
        "n_deepened": deepened,
        "max_depth_realized": max((int(d) for d in realized["depth"]), default=0),
        "fallbacks": fallbacks,
        "fallback_details": errors,
        "deadline_truncated": truncated,
        "worlds_gate_failed": gate_failed,
        "realized_mean": {k: (round(sum(v) / len(v), 3) if v else 0.0)
                          for k, v in realized.items()},
        # ADDITIVE (ladder requirement 3, 87a3f91). Zero on every arm but `playoff`, so a row
        # written by any other cell — or by any earlier version of this file — folds identically.
        **fold_playoff(decisions),
        **fold_racing(decisions),
        **fold_defensive(decisions),
    }


#: The error a game gets when it produced neither an exception nor an outcome. Named, because the
#: alternative is a row that says nothing went wrong on a battle that never happened — measured
#: 2026-08-23, when a pruned worktree took ``local_sim_bridge.js`` out from under a running battery
#: and 8 straight games recorded clean-looking and empty.
NEVER_FINISHED = ("battle_never_finished: no exception, no result — the bridge child likely died "
                  "at spawn (deleted worktree? missing node/dist?)")
NEVER_CREATED = ("battle_never_created: run_local_battles returned without ever creating a battle "
                 "object for this player — the bridge child produced no protocol at all")


def finalize_row(row: dict) -> dict:
    """Enforce the one invariant every result row obeys, and repair it rather than trust it.

    **A row carries a real outcome XOR a named error. Never neither.** A game that neither threw
    nor produced a winner is the single most dangerous shape this file can hold: it reads as a
    played battle to every consumer, it dilutes a cell's win rate toward whatever it happened to
    be, and nothing in it points at a cause. So the invariant is checked HERE, at the one place
    rows are made, instead of being an expectation about the call sites above.
    """
    result = row.get("result")
    if result in ("win", "loss", "tie"):
        return row
    if not row.get("error"):
        row["error"] = NEVER_CREATED if not row.get("battle_created", True) else NEVER_FINISHED
    row["result"] = "unfinished"
    row["finished"] = 0
    return row


def build_players(model, mappings, cfg: SearchConfig, opponent_name: str, *,
                  pool_packed: Optional[Sequence[str]] = None, tag: str = "",
                  playoff_cfg: Optional[PlayoffConfig] = None):
    """The trainee (search-wrapped) + one scripted bot, both bridge-transport (no server).

    Teams are injected per game by :func:`set_teams`, so the teambuilders here are placeholders
    that the driver overrides — the battery must not let each player draw its own team."""
    from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration
    from agents.training.eval_callback import build_eval_opponents
    from utils.teambuilder import Gen3Teambuilder
    from utils.team_loader import TeamLoader

    teams = list(pool_packed) if pool_packed else None
    tb = Gen3Teambuilder(TeamLoader().get_all_teams()) if teams is None else _FixedTeam(teams[0])
    if opponent_name == "self":
        # The MIRROR opponent (owner-ordered): the SAME network with search structurally OFF —
        # a base-arm SearchDividendPlayer, so the two sides differ in exactly one thing, the
        # search. This is the sensitive contrast: the scripted roster is saturated (~90% either
        # way, the dividend hides in the ceiling), while a mirror pins the no-effect point at
        # 50% by construction. `dataclasses.replace` keeps every cap/score knob identical to
        # the searched side's config rather than re-deriving a second one that could drift.
        from dataclasses import replace
        opp_engine = SearchEngine(model=model, mappings=mappings,
                                  cfg=replace(cfg, arm="base", budget_s=0.0),
                                  pool_packed=pool_packed)
        opp = SearchDividendPlayer(
            model=model, team=tb, battle_format=BATTLE_FORMAT,
            server_configuration=LocalhostServerConfiguration, mappings=mappings,
            account_configuration=AccountConfiguration(f"SDivSelf{tag}", "password"),
            start_listening=False, engine=opp_engine, opp_player=None)
    else:
        (_name, opp) = build_eval_opponents(
            LocalhostServerConfiguration, tb, [opponent_name], tag=tag, start_listening=False)[0]
    # The second-stage scorer exists ONLY where a second stage runs: the `playoff` ARM, and the
    # `defensive` ROOT STRATEGY when --defensive-confirm asked for one. Building it
    # unconditionally would be harmless (it is never called) but would misdescribe every other
    # cell's engine. The defensive path overrides `rollouts` with its OWN flag, so one runner
    # cannot silently inherit the playoff arm's much larger R.
    runner = None
    dcfg = cfg.defensive_cfg()
    if cfg.arm == "playoff":
        runner = PlayoffRunner(model=model, mappings=mappings, battle_format=BATTLE_FORMAT,
                               cfg=playoff_cfg or PlayoffConfig(), tag=tag)
    elif dcfg is not None and int(dcfg.confirm_rollouts) > 0:
        from dataclasses import replace as _replace
        runner = PlayoffRunner(
            model=model, mappings=mappings, battle_format=BATTLE_FORMAT,
            cfg=_replace(playoff_cfg or PlayoffConfig(), rollouts=int(dcfg.confirm_rollouts)),
            tag=tag)
    engine = SearchEngine(model=model, mappings=mappings, cfg=cfg, pool_packed=pool_packed,
                          playoff=runner)
    me = SearchDividendPlayer(
        model=model, team=tb, battle_format=BATTLE_FORMAT,
        server_configuration=LocalhostServerConfiguration, mappings=mappings,
        account_configuration=AccountConfiguration(f"SDiv{tag}", "password"),
        start_listening=False, engine=engine, opp_player=opp)
    if opponent_name == "self":
        # Each side's in-flight record builder names the OTHER side; wire the back-reference
        # only after both exist (the ctor merely stores it).
        opp._opp_player = me
    return me, opp, engine


class _FixedTeam:
    """A teambuilder whose team the driver sets per game (poke-env calls ``yield_team``)."""

    def __init__(self, packed: str):
        self.packed = packed

    def yield_team(self) -> str:
        return self.packed


def set_teams(me, opp, our_packed: str, their_packed: str) -> None:
    """Pin both sides' teams for the next battle. ``run_local_battles`` calls ``get_next_team()``
    on each player, which reads ``player._team.yield_team()``."""
    me._team = _FixedTeam(our_packed)
    opp._team = _FixedTeam(their_packed)


async def run_cell(cell: Cell, *, model, mappings, cfg: SearchConfig, games: int,
                   results: ResultsFile, salt: int, impl: str,
                   pool_packed: Sequence[str], progress=None,
                   side_swap: bool = False, games_start: int = 0,
                   playoff_cfg: Optional[PlayoffConfig] = None) -> int:
    """Play the missing orientation-games of one cell, appending a row each.

    With ``side_swap`` every ``game`` index is played TWICE — orientation 0 and orientation 1 — so
    the summary can difference out the team draw. ``games`` therefore keeps meaning "distinct team
    draws" and the battle count doubles, which is the honest way round: halving ``games`` to keep
    the battle count would halve the number of independent draws, which is the thing that actually
    bounds the interval.

    ``games_start`` plays the HALF-OPEN index window ``[games_start, games_start + games)`` instead
    of ``[0, games)``. It is a SHARDING knob and nothing more: :func:`game_seed` and
    :func:`team_pair` are functions of the index alone, so game 500 is the same dice and the same
    team draw whichever process plays it, and two shards over disjoint windows concatenate into
    exactly the file one process would have written. That is what lets a cell whose per-game wall
    exceeds the session budget be split across cores without any arm becoming a different
    experiment — the alternative (a different ``--games-seed`` per shard) would re-use index 0 for
    two different battles and silently break the pairing the summary does on it.
    """
    install_choice_tap()
    orientations = (0, 1) if side_swap else (0,)
    done = results.done_units(cell)
    lo = int(games_start)
    todo = [(g, o) for g in range(lo, lo + games) for o in orientations if (g, o) not in done]
    if not todo:
        return 0
    me, opp, engine = build_players(model, mappings, cfg, cell.opponent,
                                    pool_packed=pool_packed, tag=_cell_tag(cell),
                                    playoff_cfg=playoff_cfg)
    played = 0
    try:
        for g, orient in todo:
            ours, theirs = team_pair(cell.opponent, g, salt, pool_packed, orient)
            set_teams(me, opp, ours, theirs)
            seed = game_seed(cell.opponent, g, salt)
            t0 = time.monotonic()
            try:
                out = await play_one_battle(me, opp, battle_format=BATTLE_FORMAT,
                                            seed=seed, impl=impl)
                err = None
            except Exception as e:                    # noqa: BLE001
                # A crashed game is RECORDED as an error row, not dropped. A dropped game biases
                # the win rate by whatever made it crash.
                out = {"outcome": "unfinished", "won": 0, "tied": 0, "finished": 0,
                       "battle_created": True, "decisions": []}
                err = f"{type(e).__name__}: {e}"
            row = finalize_row({
                "v": ROW_VERSION, "arm": cell.arm, "budget": cell.budget,
                "opponent": cell.opponent, "game": g, "orientation": int(orient),
                "result": out.get("outcome"),
                "finished": int(out.get("finished", 0)), "won": int(out.get("won", 0)),
                "tied": int(out.get("tied", 0)), "turns": int(out.get("turns", 0) or 0),
                "battle_created": bool(out.get("battle_created", False)),
                "wall_s": round(time.monotonic() - t0, 3),
                "seed": seed,
                # The score mode the search was ASKED for — `effective_score`, not the raw flag,
                # because `--root-strategy defensive` names its own head and a row recording
                # "auto" would misdescribe which leaf the cell was measured on.
                "score_mode": cfg.effective_score(), "search_impl": cfg.search_impl,
                "root_strategy": cfg.root_strategy,
                "max_depth": int(getattr(cfg, "max_depth", 1)),
                "playoff_rollouts": (int(playoff_cfg.rollouts) if playoff_cfg
                                     and cfg.arm == "playoff" else 0),
                "error": err,
                **summarize_decisions(out.get("decisions") or []),
            })
            results.append(row)
            played += 1
            if progress is not None:
                progress(row)
    finally:
        engine.close()
    return played
