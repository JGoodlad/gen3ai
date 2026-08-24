"""The evaluation battery: play matched games per (arm, budget, opponent) and append the rows.

**Matched games are the whole point.** Three arms are compared, so every cell must play the SAME
battles: the same opponent, the same team draw, the same dice. The driver therefore derives a
per-game SEED from ``(opponent, game_index, --games-seed)`` and pins it on the bridge ``START``,
and it draws both teams from a per-game RNG rather than the players' own teambuilders. Two arms
that disagree on a game then disagree about the DECISION, not about the battle they were handed —
without that, an arm difference at N=10 is mostly team-draw noise (the exploiter work measured
exactly that trap: an apparent edge that vanished under an equal-pilot mirror).

**Resumable, append-only.** Each finished game appends one JSON line. A relaunch reads the file,
counts the ``(arm, budget, opponent)`` cells already complete, and plays only what is missing. The
file is never rewritten — the same discipline ``eval_results.jsonl`` follows, and for the same
reason: a summary can always be recomputed from rows, but a row lost to a crashed rewrite is gone.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from main.search_dividend.player import SearchDividendPlayer, play_one_battle
from main.search_dividend.record import install_choice_tap
from main.search_dividend.search import SearchConfig, SearchEngine

BATTLE_FORMAT = "gen3ou"
ROW_VERSION = 1


def game_seed(opponent: str, game: int, salt: int) -> str:
    """The sim seed for one CELL-INDEPENDENT game.

    Derived from ``(opponent, game, salt)`` and NOT from the arm or the budget, which is what
    makes the arms matched: cell (base, 1 s, heuristic) game 7 and cell (oracle, 3 s, heuristic)
    game 7 are the same dice."""
    h = hashlib.sha256(f"{salt}|{opponent}|{game}".encode()).hexdigest()
    return "sodium," + h[:32]


def team_pair(opponent: str, game: int, salt: int, packed_teams: Sequence[str]) -> tuple:
    """The (our, their) packed teams for one game — matched across arms for the same reason."""
    rng = random.Random(f"{salt}|{opponent}|{game}|teams")
    return rng.choice(list(packed_teams)), rng.choice(list(packed_teams))


@dataclass
class Cell:
    arm: str
    budget: float
    opponent: str

    def key(self) -> str:
        return f"{self.arm}|{self.budget:g}|{self.opponent}"


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
        bounded and visible, against silently under-powered cells, which is neither."""
        if not int(row.get("finished", 0)):
            return
        k = Cell(row["arm"], float(row["budget"]), row["opponent"]).key()
        self._done.setdefault(k, set()).add(int(row["game"]))

    def done_games(self, cell: Cell) -> set:
        return set(self._done.get(cell.key(), ()))

    def n_done(self, cell: Cell) -> int:
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
    realized = {"m_opp": [], "k_worlds": [], "r_dice": [], "arms": [], "elapsed": []}
    errors: List[str] = []
    n_changed = 0
    n_searched = 0
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
        truncated += 1 if w.get("deadline_truncated") else 0
        gate_failed += int(w.get("worlds_gate_failed", 0))
    return {
        "n_decisions": len(decisions),
        "n_searched": n_searched,
        "n_changed": n_changed,
        "fallbacks": fallbacks,
        "fallback_details": errors,
        "deadline_truncated": truncated,
        "worlds_gate_failed": gate_failed,
        "realized_mean": {k: (round(sum(v) / len(v), 3) if v else 0.0)
                          for k, v in realized.items()},
    }


def build_players(model, mappings, cfg: SearchConfig, opponent_name: str, *,
                  pool_packed: Optional[Sequence[str]] = None, tag: str = ""):
    """The trainee (search-wrapped) + one scripted bot, both bridge-transport (no server).

    Teams are injected per game by :func:`set_teams`, so the teambuilders here are placeholders
    that the driver overrides — the battery must not let each player draw its own team."""
    from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration
    from agents.training.eval_callback import build_eval_opponents
    from utils.teambuilder import Gen3Teambuilder
    from utils.team_loader import TeamLoader

    teams = list(pool_packed) if pool_packed else None
    tb = Gen3Teambuilder(TeamLoader().get_all_teams()) if teams is None else _FixedTeam(teams[0])
    (_name, opp) = build_eval_opponents(
        LocalhostServerConfiguration, tb, [opponent_name], tag=tag, start_listening=False)[0]
    engine = SearchEngine(model=model, mappings=mappings, cfg=cfg, pool_packed=pool_packed)
    me = SearchDividendPlayer(
        model=model, team=tb, battle_format=BATTLE_FORMAT,
        server_configuration=LocalhostServerConfiguration, mappings=mappings,
        account_configuration=AccountConfiguration(f"SDiv{tag}", "password"),
        start_listening=False, engine=engine, opp_player=opp)
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
                   pool_packed: Sequence[str], progress=None) -> int:
    """Play the missing games of one cell, appending a row each. Returns how many were played."""
    install_choice_tap()
    done = results.done_games(cell)
    todo = [g for g in range(games) if g not in done]
    if not todo:
        return 0
    me, opp, engine = build_players(model, mappings, cfg, cell.opponent,
                                    pool_packed=pool_packed, tag=_cell_tag(cell))
    played = 0
    try:
        for g in todo:
            ours, theirs = team_pair(cell.opponent, g, salt, pool_packed)
            set_teams(me, opp, ours, theirs)
            t0 = time.monotonic()
            try:
                out = await play_one_battle(me, opp, battle_format=BATTLE_FORMAT,
                                            seed=game_seed(cell.opponent, g, salt), impl=impl)
                err = None
            except Exception as e:                    # noqa: BLE001
                # A crashed game is RECORDED as an error row, not dropped. A dropped game biases
                # the win rate by whatever made it crash.
                out, err = {"won": 0, "finished": 0, "decisions": []}, f"{type(e).__name__}: {e}"
            if err is None and not out["finished"]:
                # `run_local_battles` returns cleanly on a bridge child that dies at spawn (EOF
                # in `_demux` breaks the loop with no exception), so a battle can "complete" in
                # 0.1 s without ever being created. Measured 2026-08-23: a pruned worktree took
                # `local_sim_bridge.js` out from under a running battery and 8 straight games
                # recorded as unfinished with error=None — a row that says nothing went wrong on
                # a game that never happened. Name it, so the summary's error census sees it.
                err = ("battle_never_finished: no exception, no result — the bridge child "
                      "likely died at spawn (deleted worktree? missing node/dist?)")
                out["decisions"] = out.get("decisions") or []
            row = {
                "v": ROW_VERSION, "arm": cell.arm, "budget": cell.budget,
                "opponent": cell.opponent, "game": g,
                "result": ("win" if out["won"] else ("loss" if out["finished"] else "unfinished")),
                "finished": int(out["finished"]), "won": int(out["won"]),
                "wall_s": round(time.monotonic() - t0, 3),
                "seed": game_seed(cell.opponent, g, salt),
                "score_mode": cfg.score, "search_impl": cfg.search_impl,
                "error": err,
                **summarize_decisions(out.get("decisions") or []),
            }
            results.append(row)
            played += 1
            if progress is not None:
                progress(row)
    finally:
        engine.close()
    return played
