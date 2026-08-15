"""Tier H-A fuzz (gen3_pair_history_v1) — the compiled-history obs blocks vs PROTOCOL TRUTH.

Real battles in-process via the local BattleStream bridge (no server). At EVERY decision the
player runs the real tracker protocol (EpisodeTracker.record → update_progress_clock → encode
with pair_history threaded, exactly the RLPlayer/Gen3Env path) and validates BOTH H-A blocks
against an INDEPENDENT from-scratch recount over the battle's FULL event log:

  * H-A1 — each side's ACTIVE slot carries that side's last action
    [move_num, was_switch, hit, miss, fail, crit]: last MOVE or SWITCH/DRAG event per side;
    MISS/FAIL/CRIT attach to the same side's same-turn move; a CANT window leaves the
    previous action standing; bench slots carry all-zeros.
  * H-A2 — the 6×6×5 pair block: for every (their mon i, our mon j) with both species known
    to the test, the switch-in / attack / status-click counters (their chosen actions while
    our j was active, actives replayed event-ordered), distinct shared-field turns (events +
    this test's OWN decision-time pairing log — the recency fuzz's `_active_log` pattern),
    and the last-pairing recency, all log-saturated over the 10 cap.

Any mismatch raises with the offending (pair/side, channel, got, want).

Run directly:
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/pair_history_fuzz_test.py [n_battles]
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
from agents.gen3_data import moves as gen3_movedex
from agents.observation.constants import (
    OFFSET_OUR_TEAM, OFFSET_OPP_TEAM, OFFSET_PAIR_HISTORY,
    PAIR_HISTORY_CELL_DIM, POKEMON_FULL_DIM, POKEMON_LAST_ACTION_OFFSET,
    POKEMON_LAST_ACTION_DIM, TEAM_SIZE,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.episode_tracker import EpisodeTracker, _move_is_damaging
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


class _PairHistoryFuzzPlayer(Player):
    def __init__(self, *args, stats: _Stats, **kwargs):
        kwargs.setdefault("battle_class", Gen3Battle)
        super().__init__(*args, **kwargs)
        self.stats = stats
        self.encoder = Gen3ObservationEncoder(load_mappings())
        self._trackers: dict = {}
        # tag -> list of (turn, our_active_species, opp_active_species) at each decision —
        # the test's OWN record of the decision-time resync marks the tracker also makes.
        self._decision_log: dict = {}

    # ---------------------------------------------------------------- the independent oracle
    def _oracle(self, battle, decision_log, cur_turn):
        """From-scratch recount over the FULL event log (independent of PairHistoryTracker):
        returns (last_action[side] dict, pair dict[(opp_sp, our_sp)] -> raw counters)."""
        last = {}
        our_active = opp_active = None
        switch_ins: dict = {}
        attacks: dict = {}
        status_clicks: dict = {}
        pair_turns: dict = {}          # (i, j) -> set of turns
        decisions_by_turn = {}
        for (t, o, p) in decision_log:
            decisions_by_turn.setdefault(t, []).append((o, p))

        def mark(t):
            if our_active and opp_active:
                pair_turns.setdefault((opp_active, our_active), set()).add(t)

        for e in battle.events_since(0):
            side = getattr(e, "side", None)
            sp = getattr(e, "actor_species", None)
            et = getattr(e, "turn", None)
            if et is None:
                continue
            k = e.kind
            # LEADS (turn-0 events) are outside every tracker window by construction — a lead
            # is a simultaneous placement, not a switch-in RESPONSE — so pre-turn-1 events
            # advance the replayed actives but never count and never become a last action.
            countable = et >= 1
            if k is EventKind.MOVE and side is not None and sp:
                if countable:
                    last[side] = dict(move_id=e.move_id, was_switch=False,
                                      missed=False, failed=False, crit=False, turn=et)
                    if side == OPP and our_active:
                        key = (sp, our_active)
                        d = attacks if _move_is_damaging(e.move_id) else status_clicks
                        d[key] = d.get(key, 0) + 1
                    mark(et)
            elif k in (EventKind.SWITCH, EventKind.DRAG) and side is not None and sp:
                if countable:
                    if k is EventKind.SWITCH and side == OPP and our_active:
                        key = (sp, our_active)
                        switch_ins[key] = switch_ins.get(key, 0) + 1
                    last[side] = dict(move_id=None, was_switch=True,
                                      missed=False, failed=False, crit=False, turn=et)
                if side == OURS:
                    our_active = sp
                else:
                    opp_active = sp
                if countable:
                    mark(et)
            elif k in (EventKind.MISS, EventKind.FAIL, EventKind.CRIT) and side is not None:
                la = last.get(side)
                if la and not la["was_switch"] and la["turn"] == et:
                    if k is EventKind.MISS:
                        la["missed"] = True
                    elif k is EventKind.FAIL:
                        la["failed"] = True
                    else:
                        la["crit"] = True
            elif k is EventKind.FAINT and side is not None and sp:
                if side == OURS and our_active == sp:
                    our_active = None
                elif side == OPP and opp_active == sp:
                    opp_active = None

        # The decision-time pairing marks (the tracker's LiveView resync, replayed from the
        # test's own log — including THIS decision, which the tracker marks before encode).
        for t, pairs in decisions_by_turn.items():
            for (o, p) in pairs:
                if o and p:
                    pair_turns.setdefault((p, o), set()).add(t)
        return last, dict(switch_ins=switch_ins, attacks=attacks,
                          status_clicks=status_clicks, pair_turns=pair_turns)

    def _want_last_tuple(self, la):
        if la is None:
            return (0.0,) * POKEMON_LAST_ACTION_DIM
        if la["was_switch"]:
            return (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        num = 0.0
        if la["move_id"]:
            md = gen3_movedex.get(la["move_id"])
            num = float(md.num) if md is not None else 0.0
        hit = 0.0 if (la["missed"] or la["failed"]) else 1.0
        return (num, 0.0, hit,
                1.0 if la["missed"] else 0.0,
                1.0 if (la["failed"] and not la["missed"]) else 0.0,
                1.0 if la["crit"] else 0.0)

    # ----------------------------------------------------------------------- the decision hook
    def choose_move(self, battle):
        tag = battle.battle_tag
        tracker = self._trackers.setdefault(tag, EpisodeTracker())
        mask = np.ones(11, dtype=np.float32)
        tracker.record(battle, mask)
        tracker.update_progress_clock(battle, None)
        live = battle.strict_view().live
        cur_turn = live.turn
        dlog = self._decision_log.setdefault(tag, [])
        _oa, _pa = live.ours.active, live.opp.active
        dlog.append((cur_turn,
                     _oa.species if (_oa is not None and not _oa.fainted) else None,
                     _pa.species if (_pa is not None and not _pa.fainted) else None))
        obs = self.encoder.encode(battle, legal=None, recency=tracker.recency,
                                  pair_history=tracker.pair_history)
        s = self.stats
        s.decisions += 1
        last, pairs = self._oracle(battle, dlog, cur_turn)

        # --- H-A1: the active slot carries the side's last action; bench slots are zero ---
        our_team = self.encoder.get_team_list(battle, is_opponent=False)
        opp_team = self.encoder.get_team_list(battle, is_opponent=True)
        for side, base_off, team, active_sp in (
                (OURS, OFFSET_OUR_TEAM, our_team,
                 live.ours.active.species if live.ours.active else None),
                (OPP, OFFSET_OPP_TEAM, opp_team,
                 live.opp.active.species if live.opp.active else None)):
            want_active = self._want_last_tuple(last.get(side))
            for i in range(TEAM_SIZE):
                mon = team[i] if i < len(team) else None
                if mon is None or not mon.species:
                    continue
                off = base_off + i * POKEMON_FULL_DIM + POKEMON_LAST_ACTION_OFFSET
                got = tuple(float(x) for x in obs[off:off + POKEMON_LAST_ACTION_DIM])
                want = want_active if mon.species == active_sp else (0.0,) * POKEMON_LAST_ACTION_DIM
                s.checked += 1
                if any(abs(g - w) > _TOL for g, w in zip(got, want)):
                    s.failures.append(dict(tag=tag, turn=cur_turn, kind="last_action",
                                           side=side, species=mon.species, got=got, want=want))

        # --- H-A2: every (opp slot i, our slot j) cell vs the recount ---
        for i in range(TEAM_SIZE):
            opp_mon = opp_team[i] if i < len(opp_team) else None
            opp_sp = opp_mon.species if opp_mon is not None else None
            for j in range(TEAM_SIZE):
                our_mon = our_team[j] if j < len(our_team) else None
                our_sp = our_mon.species if our_mon is not None else None
                off = OFFSET_PAIR_HISTORY + (i * TEAM_SIZE + j) * PAIR_HISTORY_CELL_DIM
                got = tuple(float(x) for x in obs[off:off + PAIR_HISTORY_CELL_DIM])
                if not opp_sp or not our_sp:
                    # An empty slot's cell is written from pair_values(None, ...): zero counts
                    # + never-paired recency 1.0.
                    want = (0.0, 0.0, 0.0, 0.0, 1.0)
                else:
                    key = (opp_sp, our_sp)
                    turns = pairs["pair_turns"].get(key, set())
                    rec = 1.0 if not turns else _norm(cur_turn - max(turns))
                    want = (_norm(pairs["switch_ins"].get(key, 0)),
                            _norm(pairs["attacks"].get(key, 0)),
                            _norm(pairs["status_clicks"].get(key, 0)),
                            _norm(len(turns)), rec)
                s.checked += 1
                if any(abs(g - w) > _TOL for g, w in zip(got, want)):
                    rec = dict(tag=tag, turn=cur_turn, kind="pair",
                               pair=(opp_sp, our_sp), got=got, want=want)
                    if not s.failures:      # first failure: dump the evidence trail
                        rec["trace"] = [
                            (str(e.kind).replace("EventKind.", ""), getattr(e, "turn", None),
                             getattr(e, "side", None), getattr(e, "actor_species", None),
                             getattr(e, "move_id", None), getattr(e, "seq", None))
                            for e in battle.events_since(0)
                            if getattr(e, "kind", None) in (EventKind.MOVE, EventKind.SWITCH,
                                                            EventKind.DRAG, EventKind.FAINT)]
                        rec["dlog"] = list(dlog)
                    s.failures.append(rec)

        if battle.finished:
            self._trackers.pop(tag, None)
            self._decision_log.pop(tag, None)
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        self._trackers.pop(battle.battle_tag, None)
        return super()._battle_finished_callback(battle)


def main(n_battles: int = 30) -> int:
    stats = _Stats()
    teams = TeamLoader().get_all_teams()
    p1 = _PairHistoryFuzzPlayer(stats=stats, team=Gen3Teambuilder(teams), battle_format="gen3ou",
                                server_configuration=LocalhostServerConfiguration,
                                account_configuration=AccountConfiguration(f"PhA{int(time.time()) % 10000}", None),
                                start_listening=False, max_concurrent_battles=1)
    p2 = _PairHistoryFuzzPlayer(stats=_Stats(), team=Gen3Teambuilder(teams), battle_format="gen3ou",
                                server_configuration=LocalhostServerConfiguration,
                                account_configuration=AccountConfiguration(f"PhB{int(time.time()) % 10000}", None),
                                start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(p1, p2, n_battles))
    print(f"[pair-history fuzz] {n_battles} battles, {stats.decisions} decisions, "
          f"{stats.checked} checks, {len(stats.failures)} failures")
    if stats.failures:
        for f in stats.failures[:10]:
            print("  FAIL", f)
        raise SystemExit(1)
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 30))
