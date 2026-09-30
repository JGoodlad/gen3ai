"""Per-episode TEAMS and SEEDS for the Rust env core (M5 Lane G; F-LE-5).

The core plays a battle between two entries of a TEAM TABLE declared at startup (``spec.teams``,
every team validated by use in ``Core::new``) and a Showdown seed, both STAGED by the host one
episode ahead (``ep_team`` / ``ep_seed``, consumed at every start with ``ep_opp``). This module owns
that staging for training:

* :class:`TeamTable` — every packed team any builder of the run can yield (the trainee builder's pool
  and bias teams, the opponent builder's pool, each pinned stable / exploiter team — F-LE-5), in one
  deduplicated, index-stable list. A draw that is not in the table is a typed `TeamTableMiss`.
* :class:`TeamStager` — per env, a COPY of the run's trainee and opponent ``Gen3Teambuilder`` (today
  each ``SubprocVecEnv`` worker unpickles its own copy) and of each pinned route's builder, every copy
  SEEDED by declaration (today's builders draw from each worker's unseeded global ``random`` — a
  declared change of stream, not of distribution, like F-LE-3); the draw is the builder's own
  ``yield_team`` (block episodes, bias, PFSP weights all compose). The opponent's team follows the
  episode's ROUTE exactly as ``MaskableAgentWrapper._apply_opponent_team`` does (its pin, else the
  pool builder).
* the SEED of env ``e``'s episode ``k`` is four 16-bit words of the keyed hash
  ``(run seed, STREAM_BATTLE, e, k, 0)`` — a pure function, so a replay reproduces it.
* the per-team WIN-RATE tracking (``--team-wr-tracking``, default on): an ended episode's outcome is
  recorded against the team that episode PLAYED — tracked per env across the one-episode-ahead
  staging, since the builder's own ``_last_pool_idx`` already names the NEXT episode's team — and
  drained per env in the builder's own format (``drain_team_wr_counts``).
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from agents.training import keyed_draw as KD

#: The keyed-hash stream of the Showdown battle seeds (the trainee and opponent streams are 0 / 1).
STREAM_BATTLE = 2
#: The keyed-hash stream of the per-env teambuilder RNG seeds.
STREAM_TEAM_RNG = 3


class TeamTableMiss(RuntimeError):
    """A builder yielded a team the startup table does not hold."""


class TeamTable:
    def __init__(self) -> None:
        self.teams: List[str] = []
        self._index: Dict[str, int] = {}

    def add(self, packed: str) -> int:
        i = self._index.get(packed)
        if i is None:
            i = len(self.teams)
            self.teams.append(packed)
            self._index[packed] = i
        return i

    def add_builder(self, builder: Any) -> None:
        for t in list(getattr(builder, "packed_teams", ())) + list(getattr(builder, "bias_packed_teams", ()) or ()):
            self.add(t)

    def index(self, packed: str, where: str) -> int:
        i = self._index.get(packed)
        if i is None:
            raise TeamTableMiss(f"{where}: a yielded team is not in the startup team table "
                                f"({len(self.teams)} teams) — every builder must be declared before the core starts")
        return i

    def __len__(self) -> int:
        return len(self.teams)


def battle_seed(run_seed: int, env: int, episode: int) -> List[int]:
    h = int(KD.draw_keys(run_seed, STREAM_BATTLE, env, episode, 0))
    return [(h >> s) & 0xFFFF for s in (0, 16, 32, 48)]


def _builder_copy(builder: Any, seed: int) -> Any:
    if not hasattr(builder, "packed_teams") or not hasattr(builder, "yield_team"):
        raise TypeError(f"the Rust collector stages teams from a Gen3Teambuilder; got {type(builder).__name__}")
    b = copy.deepcopy(builder)
    b._rng = random.Random(int(seed))
    return b


def pinned_builder(team_strs: Sequence[str]) -> Any:
    """A stable / exploiter opponent's pinned builder, built as ``env_factory`` builds it."""
    from utils.teambuilder import Gen3Teambuilder

    return Gen3Teambuilder(list(team_strs))


class TeamStager:
    """Per-env seeded team draws + battle seeds (module docs). ``route_builders`` maps a route index
    to the pinned builder its episodes' opponent pilots (routes absent pilot the opponent builder)."""

    def __init__(self, table: TeamTable, n_envs: int, *, trainee_builder: Any, opponent_builder: Any,
                 route_builders: Optional[Mapping[int, Any]] = None, run_seed: int,
                 team_wr_tracking: bool = True, n_opp_classes: int = 4):
        self.table, self.n, self.run_seed = table, int(n_envs), int(run_seed)
        self.team_wr_tracking = bool(team_wr_tracking)
        self.n_opp_classes = int(n_opp_classes)

        def seed_of(env: int, which: int) -> int:
            return int(KD.draw_keys(self.run_seed, STREAM_TEAM_RNG, env, which, 0)) & 0x7FFFFFFFFFFFFFFF

        self.trainee = [_builder_copy(trainee_builder, seed_of(i, 0)) for i in range(self.n)]
        self.opponent = [_builder_copy(opponent_builder, seed_of(i, 1)) for i in range(self.n)]
        rb = dict(route_builders or {})
        self.route: List[Dict[int, Any]] = [
            {r: _builder_copy(b, seed_of(i, 2 + k)) for k, (r, b) in enumerate(sorted(rb.items()))}
            for i in range(self.n)]
        for b in [trainee_builder, opponent_builder, *rb.values()]:
            table.add_builder(b)
        self.staged_pool_idx: List[Optional[int]] = [None] * self.n
        self.cur_pool_idx: List[Optional[int]] = [None] * self.n
        self.staged_teams = np.full((self.n, 2), -1, dtype=np.int64)
        self.cur_teams = np.full((self.n, 2), -1, dtype=np.int64)
        self.draws = 0

    def stage(self, cols: Mapping[str, np.ndarray], envs: Sequence[int], routes: Sequence[int],
              next_episode: Sequence[int]) -> None:
        """Stage env ``envs[j]``'s next episode (ordinal ``next_episode[j]``) against ``routes[j]``."""
        for e, r, k in zip(envs, routes, next_episode):
            e, r, k = int(e), int(r), int(k)
            tb = self.trainee[e]
            p1 = self.table.index(tb.yield_team(), f"env {e} trainee")
            self.staged_pool_idx[e] = getattr(tb, "_last_pool_idx", None)
            ob = self.route[e].get(r, self.opponent[e])
            p2 = self.table.index(ob.yield_team(), f"env {e} opponent route {r}")
            cols["ep_team"][e] = (p1, p2)
            cols["ep_seed"][e] = battle_seed(self.run_seed, e, k)
            self.staged_teams[e] = (p1, p2)
            self.draws += 1

    def started(self, envs: Sequence[int]) -> None:
        """``envs`` consumed their staged row (their ``episode`` column moved)."""
        for e in envs:
            e = int(e)
            self.cur_pool_idx[e] = self.staged_pool_idx[e]
            self.cur_teams[e] = self.staged_teams[e]

    def record_outcome(self, env: int, won: float, opp_class: int) -> None:
        """The per-team win-rate record of the episode env ``env`` just finished (module docs)."""
        if not self.team_wr_tracking:
            return
        tb = self.trainee[int(env)]
        if not hasattr(tb, "record_team_wr_outcome"):
            return
        staged = getattr(tb, "_last_pool_idx", None)
        tb._last_pool_idx = self.cur_pool_idx[int(env)]
        try:
            tb.record_team_wr_outcome(float(won), int(opp_class), self.n_opp_classes)
        finally:
            tb._last_pool_idx = staged

    def drain_team_wr_counts(self) -> List[Any]:
        """Per env, the builder's ``drain_team_wr_counts()`` (``(counts, keys)``) or None when off."""
        if not self.team_wr_tracking:
            return [None] * self.n
        return [tb.drain_team_wr_counts() if hasattr(tb, "drain_team_wr_counts") else None for tb in self.trainee]
