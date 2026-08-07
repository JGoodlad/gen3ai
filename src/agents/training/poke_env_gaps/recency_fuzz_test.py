"""E9 recency fuzz (gen3_entity_recency_v1) — the per-mon recency obs block vs PROTOCOL TRUTH.

Real battles in-process via the local BattleStream bridge (no server). At EVERY decision the
player runs the real tracker protocol (EpisodeTracker.record → update_progress_clock → encode
with recency threaded, exactly the RLPlayer/Gen3Env path) and validates the three encoded
per-mon recency dims against an INDEPENDENT from-scratch recount over the battle's FULL event
log: last MOVE turn (acted), last SWITCH-or-active turn (seen), last DAMAGE turn (was_hit),
counter = current turn − last-event turn, same 10-cap log-saturation. Any mismatch raises with
the offending (side, species, channel).

Run directly:
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/recency_fuzz_test.py [n_battles]
"""
from __future__ import annotations

import asyncio
import math
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.battle_event import OURS, OPP, EventKind
from agents.battle.gen3_battle import Gen3Battle
from agents.observation.constants import (
    OFFSET_OUR_TEAM, OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_RECENCY_OFFSET, TEAM_SIZE,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.episode_tracker import EpisodeTracker
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder
from utils.bridge.local_battle_runner import run_local_battles

_SAT = 10
_TOL = 1e-6


def _norm(n: int) -> float:
    return math.log1p(min(max(n, 0), _SAT)) / math.log(11.0)


@dataclass
class _Stats:
    decisions: int = 0
    checked: int = 0
    failures: list = field(default_factory=list)


class _RecencyFuzzPlayer(Player):
    def __init__(self, *args, stats: _Stats, **kwargs):
        kwargs.setdefault("battle_class", Gen3Battle)
        super().__init__(*args, **kwargs)
        self.stats = stats
        self.encoder = Gen3ObservationEncoder(load_mappings())
        self._trackers: dict = {}
        self._active_log: dict = {}   # tag -> {(side, species): last decision turn ON FIELD}

    def _oracle(self, battle, side: str, species: str, cur_turn: int):
        """From-scratch recount over the FULL event log (independent of RecencyTracker)."""
        last_seen = last_acted = last_hit = None
        for e in battle.events_since(0):
            if getattr(e, "side", None) != side or getattr(e, "actor_species", None) != species:
                continue
            if e.kind is EventKind.MOVE:
                last_acted = e.turn
                last_seen = e.turn if last_seen is None else max(last_seen, e.turn)
            elif e.kind is EventKind.SWITCH:
                last_seen = e.turn if last_seen is None else max(last_seen, e.turn)
            elif e.kind is EventKind.DAMAGE:
                last_hit = e.turn
        live = battle.strict_view().live
        active_sp = (live.ours.active.species if side == OURS and live.ours.active else
                     live.opp.active.species if side == OPP and live.opp.active else None)
        # A mon can be ON FIELD without generating events (its opponent acts around it), so
        # `seen` also consults the test's OWN decision-time active log (independent bookkeeping
        # of the same public fact the tracker anchors on).
        logged = self._active_log.get(battle.battle_tag, {}).get((side, species))
        if logged is not None:
            last_seen = logged if last_seen is None else max(last_seen, logged)
        def v(last):
            # The turn-anchored convention: counter = cur − event_turn (last turn reads 1).
            return _norm(cur_turn - last) if last is not None else 1.0
        seen = 0.0 if active_sp == species else v(last_seen)
        return seen, v(last_acted), v(last_hit)

    def choose_move(self, battle):
        tag = battle.battle_tag
        tracker = self._trackers.setdefault(tag, EpisodeTracker())
        mask = np.ones(11, dtype=np.float32)
        tracker.record(battle, mask)
        tracker.update_progress_clock(battle, None)
        obs = self.encoder.encode(battle, legal=None, recency=tracker.recency)
        live = battle.strict_view().live
        cur_turn = live.turn
        alog = self._active_log.setdefault(tag, {})
        if live.ours.active:
            alog[(OURS, live.ours.active.species)] = cur_turn
        if live.opp.active:
            alog[(OPP, live.opp.active.species)] = cur_turn
        s = self.stats
        s.decisions += 1
        for side, base_off, side_obj in ((OURS, OFFSET_OUR_TEAM, live.ours),
                                         (OPP, OFFSET_OPP_TEAM, live.opp)):
            team = self.encoder.get_team_list(battle, is_opponent=(side == OPP))
            for i in range(TEAM_SIZE):
                mon = team[i] if i < len(team) else None
                if mon is None or not mon.species:
                    continue
                off = base_off + i * POKEMON_FULL_DIM + POKEMON_RECENCY_OFFSET
                got = tuple(float(x) for x in obs[off:off + 3])
                want = self._oracle(battle, side, mon.species, cur_turn)
                s.checked += 1
                for ch, (g, w) in enumerate(zip(got, want)):
                    if abs(g - w) > _TOL:
                        s.failures.append(
                            dict(tag=tag, turn=cur_turn, side=side, species=mon.species,
                                 channel=("seen", "acted", "was_hit")[ch], got=g, want=w))
        if battle.finished:
            self._trackers.pop(tag, None)
            self._active_log.pop(tag, None)
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        self._trackers.pop(battle.battle_tag, None)
        return super()._battle_finished_callback(battle)


def main(n_battles: int = 30) -> int:
    stats = _Stats()
    teams = TeamLoader().get_all_teams()
    p1 = _RecencyFuzzPlayer(stats=stats, team=Gen3Teambuilder(teams), battle_format="gen3ou",
                            server_configuration=LocalhostServerConfiguration,
                            account_configuration=AccountConfiguration(f"RcyA{int(time.time()) % 10000}", None),
                            start_listening=False, max_concurrent_battles=1)
    p2 = _RecencyFuzzPlayer(stats=_Stats(), team=Gen3Teambuilder(teams), battle_format="gen3ou",
                            server_configuration=LocalhostServerConfiguration,
                            account_configuration=AccountConfiguration(f"RcyB{int(time.time()) % 10000}", None),
                            start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(p1, p2, n_battles))
    print(f"[recency fuzz] {n_battles} battles, {stats.decisions} decisions, "
          f"{stats.checked} per-mon checks, {len(stats.failures)} failures")
    if stats.failures:
        for f in stats.failures[:10]:
            print("  FAIL", f)
        raise SystemExit(1)
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 30))
