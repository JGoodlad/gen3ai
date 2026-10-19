"""THE SCHEDULED PANEL BUILDER: a teambuilder whose every draw in a head-to-head batch is NAMED in advance.

The Rust eval core draws each side's team with ``rust_eval.seeds.draw_team``: it re-seeds the builder's ``_rng`` with
``random.Random(derived_seed(key, side))`` and calls ``yield_team()``. A game's key is a pure function of
``(cycle seed, item key, game)`` (``seeds.pair_game``: a mirrored pair's two games share game ``2k``'s key). So the
draw a given (pair, side) will make is known before the batch: :meth:`ScheduledTeams.schedule` computes, for every
planned pair ``k`` and side, the first 64 bits the re-seeded RNG will produce, and maps them to the team that pair
must play. ``yield_team`` reads those bits and returns the mapped team.

Nothing is trusted on faith:

* bits that map to no planned draw are a typed :class:`ScheduleMiss` (a changed seed rule, a draw outside the plan);
* two planned draws whose bits collide are refused at :meth:`schedule` time;
* the caller checks every played game's team indices against the plan (``main.team_strength.x26_lane``).

The builder is handed to ``main.h2h.play.H2HEngine(team_builder=...)``: the trainee's builder, which a GREEDY
sentinel draws from too, so BOTH sides of every game draw here. Its ``packed_teams`` (the panel, packed by the
project's own ``Gen3Teambuilder`` — validator, Hidden Power IV fix) are declared into the core's team table at
startup."""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from agents.training.rust_eval import seeds as SD

SCHEMA = "gen3_tsr_scheduled_teams_v1"


class ScheduleMiss(RuntimeError):
    """A draw the schedule did not plan (or a plan the seed rule cannot keep apart)."""


def draw_token(key: Tuple[int, str, int], side: str) -> int:
    """The first 64 bits ``seeds.draw_team`` 's re-seeded RNG produces for ``(key, side)``."""
    return random.Random(SD.derived_seed(key, side)).getrandbits(64)


class ScheduledTeams:
    """See the module docs. ``teams`` maps team id -> paste text; every team must pass the gen3ou validator."""

    _block_episodes = 1

    def __init__(self, teams: Mapping[str, str], *, team_source: str = "", item_key: str = "h2h"):
        from utils.teambuilder import Gen3Teambuilder

        ids = sorted(teams)
        if not ids:
            raise ScheduleMiss("a scheduled builder needs at least one team")
        tb = Gen3Teambuilder([teams[i] for i in ids])
        if len(tb.packed_teams) != len(ids):
            raise ScheduleMiss(f"{len(ids) - len(tb.packed_teams)} of {len(ids)} panel teams fail the gen3ou "
                               "validator — a panel team that cannot be packed cannot be rated")
        self.packed_by_id: Dict[str, str] = dict(zip(ids, tb.packed_teams))
        if len(set(self.packed_by_id.values())) != len(ids):
            raise ScheduleMiss("two panel team ids pack to the SAME packed team — one team, two identities")
        self.id_by_packed = {p: i for i, p in self.packed_by_id.items()}
        self.packed_teams: List[str] = [self.packed_by_id[i] for i in ids]
        self.bias_packed_teams: List[str] = []
        self.item_key = str(item_key)
        self.team_set_id = "tsr:" + hashlib.sha256(json.dumps(
            {"schema": SCHEMA, "teams": self.packed_teams}, separators=(",", ":")).encode()).hexdigest()[:16]
        self.team_source = team_source or f"{SCHEMA}: every game's two teams named by the schedule"
        self._plan: Dict[int, str] = {}
        self._rng: Any = random.Random(0)

    def schedule(self, cycle_seed: int, pairs: Sequence[Tuple[str, str]]) -> None:
        """Plan one batch: pair ``k`` = ``(team id the PLAYER pilots in game 2k, the opponent's)``; game ``2k+1``
        hands them over (the executor's mirrored-pair rule)."""
        plan: Dict[int, str] = {}
        for k, (a, b) in enumerate(pairs):
            for tid in (a, b):
                if tid not in self.packed_by_id:
                    raise ScheduleMiss(f"pair {k}: team {tid} is not a panel team")
            key, swapped = SD.pair_game(int(cycle_seed), self.item_key, 2 * k, True)
            assert not swapped
            for side, tid in ((SD.TRAINEE, a), (SD.OPPONENT, b)):
                tok = draw_token(key, side)
                if tok in plan:
                    raise ScheduleMiss(f"pair {k} {side}: its draw token collides with another planned draw")
                plan[tok] = self.packed_by_id[tid]
        self._plan = plan

    def yield_team(self) -> str:
        tok = self._rng.getrandbits(64)
        try:
            return self._plan[tok]
        except KeyError:
            raise ScheduleMiss("a draw the schedule did not plan (the seed rule or the batch is not the planned one)"
                               ) from None
