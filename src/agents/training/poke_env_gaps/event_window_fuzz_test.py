"""Tier H-B fuzz (gen3_event_window_v1) — the event-window obs block vs PROTOCOL TRUTH.

Real battles in-process via the local BattleStream bridge (no server). At EVERY decision the
player runs the real tracker protocol (EpisodeTracker.record → update_progress_clock → encode
with event_window threaded — the RLPlayer/Gen3Env path) and validates the whole 32×19 block
against an INDEPENDENT from-scratch fold over the battle's FULL event log: the H-B type
vocabulary, the modifier-attach rules (clause-free target-matched damage; miss/fail/crit;
the effectiveness trio tagged on the defender), we_first, forced windows, the id columns
(species/move dex nums), recency and the front-padding convention.

Any mismatch raises with (row, column, got, want) + an event trace on the first failure.

Run directly:
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/event_window_fuzz_test.py [n_battles]
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

from agents import gen3_data
from agents.battle.battle_event import OURS, OPP, EventKind
from agents.battle.gen3_battle import Gen3Battle
from agents.gen3_data import moves as gen3_movedex
from agents.observation.constants import (
    EVENT_T_BOOST, EVENT_T_FAINT, EVENT_T_HAZARD, EVENT_T_ITEM_REVEAL, EVENT_T_MOVE,
    EVENT_T_STATUS_APPLIED, EVENT_T_STATUS_CURED, EVENT_T_SWITCH_IN, EVENT_T_SWITCH_REJECTED,
    EVENT_TOKEN_DIM, EVENT_WINDOW_DIM, EVENT_WINDOW_N, OFFSET_EVENT_WINDOW,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.episode_tracker import EpisodeTracker, _EVENT_STATUS_IDS
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder
from utils.bridge.local_battle_runner import run_local_battles

_TOL = 1e-5


def _norm(n: int) -> float:
    return math.log1p(min(max(n, 0), 10)) / math.log(11.0)


def _sp_num(sp):
    d = gen3_data.species.get(sp) if sp else None
    return float(d.num) if d is not None else 0.0


def _mv_num(mid):
    d = gen3_movedex.get(mid) if mid else None
    return float(d.num) if d is not None else 0.0


@dataclass
class _Stats:
    decisions: int = 0
    checked: int = 0
    failures: list = field(default_factory=list)


def _oracle_rows(battle, resync_log):
    """From-scratch fold over the FULL log → the expected record dicts (all of them; the
    caller windows to the last EVENT_WINDOW_N). ``resync_log`` = this test's own record of the
    (turn, our_active, opp_active) marks the tracker makes at each decision, replayed in
    event order by seq so the running actives match the tracker's."""
    rows = []
    our_active = opp_active = None
    forced = {"ours": False, "opp": False}
    open_move = {}
    first_mover = {}
    resync = sorted(resync_log, key=lambda r: r[0])     # by seq watermark

    def flags():
        return 1.0 if (forced["ours"] or forced["opp"]) else 0.0

    ri = 0
    # THE H-A CONVENTION, inherited: the tracker's first decision folds an EMPTY window
    # (`_cursors` is empty until the second record), so events before the FIRST decision's
    # cursor — the leads and battle-init lines — never become RECORDS (a lead is a placement,
    # not an action; the compiled state carries the lineup). They DO advance the oracle's
    # running STATE (actives), which is also what the tracker's decision-time resync achieves.
    start_seq = resync[0][0] if resync else 0

    def emit(seq):
        return seq >= start_seq

    for e in battle.events_since(0):
        seq = e.seq
        # apply every decision-time resync that happened BEFORE this event
        while ri < len(resync) and resync[ri][0] <= seq:
            _, o, p = resync[ri]
            if o:
                our_active, forced["ours"] = o, False
            if p:
                opp_active, forced["opp"] = p, False
            ri += 1
        side, sp, et = e.side, e.actor_species, e.turn
        k = e.kind
        if k is EventKind.MOVE and side and sp and emit(seq):
            if first_mover.get(et) is None:
                first_mover[et] = side
            r = dict(t=EVENT_T_MOVE, actor=sp, side=side,
                     target=(opp_active if side == OURS else our_active),
                     move=e.move_id, mag=0.0, hit=1.0, miss=0.0, fail=0.0, crit=0.0,
                     eff=0, wf=(side == first_mover[et]), status=0, turn=et, fw=flags())
            rows.append(r)
            open_move[side] = r
        elif k is EventKind.DAMAGE and side:
            mover = OPP if side == OURS else OURS
            om = open_move.get(mover)
            if (om is not None and om["turn"] == et and not e.value.get("from")
                    and sp and om["target"] == sp and e.amount is not None):
                om["mag"] += float(e.amount)
        elif k in (EventKind.MISS, EventKind.FAIL, EventKind.CRIT) and side:
            om = open_move.get(side)
            if om is not None and om["turn"] == et:
                if k is EventKind.MISS:
                    om["miss"], om["hit"] = 1.0, 0.0
                elif k is EventKind.FAIL:
                    om["fail"], om["hit"] = 1.0, 0.0
                else:
                    om["crit"] = 1.0
        elif k in (EventKind.IMMUNE, EventKind.RESISTED, EventKind.SUPEREFFECTIVE) and side:
            om = open_move.get(OPP if side == OURS else OURS)
            if om is not None and om["turn"] == et:
                om["eff"] = {EventKind.SUPEREFFECTIVE: 1, EventKind.RESISTED: 2,
                             EventKind.IMMUNE: 3}[k]
        elif k in (EventKind.SWITCH, EventKind.DRAG) and side and sp:
            if emit(seq):
                rows.append(dict(t=EVENT_T_SWITCH_IN, actor=sp, side=side,
                                 target=(opp_active if side == OURS else our_active),
                                 move=None, mag=0.0, hit=0.0, miss=0.0, fail=0.0, crit=0.0,
                                 eff=0, wf=False, status=0, turn=et, fw=flags()))
            forced["ours" if side == OURS else "opp"] = False
            if side == OURS:
                our_active = sp
            else:
                opp_active = sp
        elif k is EventKind.FAINT and side and sp:
            if emit(seq):
                rows.append(dict(t=EVENT_T_FAINT, actor=sp, side=side, target=None, move=None,
                                 mag=0.0, hit=0.0, miss=0.0, fail=0.0, crit=0.0, eff=0,
                                 wf=False, status=0, turn=et, fw=flags()))
            if side == OURS and our_active == sp:
                our_active, forced["ours"] = None, True
            elif side == OPP and opp_active == sp:
                opp_active, forced["opp"] = None, True
        elif k in (EventKind.STATUS, EventKind.CURESTATUS) and sp and emit(seq):
            rows.append(dict(t=(EVENT_T_STATUS_APPLIED if k is EventKind.STATUS
                                else EVENT_T_STATUS_CURED),
                             actor=sp, side=side, target=None, move=None, mag=0.0,
                             hit=0.0, miss=0.0, fail=0.0, crit=0.0, eff=0, wf=False,
                             status=_EVENT_STATUS_IDS.get(e.status or "", 0),
                             turn=et, fw=flags()))
        elif k in (EventKind.BOOST, EventKind.UNBOOST) and sp and emit(seq):
            amt = float(e.amount or 0.0)
            rows.append(dict(t=EVENT_T_BOOST, actor=sp, side=side, target=None, move=None,
                             mag=(amt if k is EventKind.BOOST else -amt), hit=0.0, miss=0.0,
                             fail=0.0, crit=0.0, eff=0, wf=False, status=0, turn=et,
                             fw=flags()))
        elif k in (EventKind.ITEM, EventKind.ENDITEM) and sp and emit(seq):
            rows.append(dict(t=EVENT_T_ITEM_REVEAL, actor=sp, side=side, target=None,
                             move=None, mag=0.0, hit=0.0, miss=0.0, fail=0.0, crit=0.0,
                             eff=0, wf=False, status=0, turn=et, fw=flags()))
        elif k is EventKind.SIDE and side and emit(seq):
            rows.append(dict(t=EVENT_T_HAZARD, actor=None, side=side, target=None, move=None,
                             mag=0.0, hit=0.0, miss=0.0, fail=0.0, crit=0.0, eff=0,
                             wf=False, status=0, turn=et, fw=flags()))
        elif k is EventKind.CHOICE_REJECTED and emit(seq):
            rows.append(dict(t=EVENT_T_SWITCH_REJECTED, actor=our_active, side=OURS,
                             target=None, move=None, mag=0.0, hit=0.0, miss=0.0, fail=0.0,
                             crit=0.0, eff=0, wf=False, status=0, turn=et, fw=flags()))
    return rows


def _want_vec(r, cur_turn):
    side = 1.0 if r["side"] == OURS else (-1.0 if r["side"] == OPP else 0.0)
    is_move = r["t"] == EVENT_T_MOVE
    mag = max(-1.0, min(1.0, r["mag"])) if is_move else max(-1.0, min(1.0, r["mag"] / 6.0))
    eff = [0.0] * 4
    if is_move:
        eff[int(r["eff"])] = 1.0
    return (float(r["t"]), _sp_num(r["actor"]), side, _sp_num(r["target"]),
            _mv_num(r["move"]), mag,
            (r["hit"] if is_move else 0.0), (r["miss"] if is_move else 0.0),
            (r["fail"] if is_move else 0.0), (r["crit"] if is_move else 0.0),
            *eff, (1.0 if r["wf"] else 0.0), float(r["status"]),
            _norm(max(0, cur_turn - int(r["turn"]))), float(r["fw"]), 1.0)


class _EventWindowFuzzPlayer(Player):
    def __init__(self, *args, stats: _Stats, **kwargs):
        kwargs.setdefault("battle_class", Gen3Battle)
        super().__init__(*args, **kwargs)
        self.stats = stats
        self.encoder = Gen3ObservationEncoder(load_mappings())
        self._trackers: dict = {}
        self._resync_log: dict = {}      # tag -> [(seq watermark, our_sp, opp_sp)]

    def choose_move(self, battle):
        tag = battle.battle_tag
        tracker = self._trackers.setdefault(tag, EpisodeTracker())
        mask = np.ones(11, dtype=np.float32)
        tracker.record(battle, mask)
        tracker.update_progress_clock(battle, None)
        live = battle.strict_view().live
        cur_turn = live.turn
        _oa, _pa = live.ours.active, live.opp.active
        rlog = self._resync_log.setdefault(tag, [])
        rlog.append((battle.event_cursor,
                     _oa.species if (_oa is not None and not _oa.fainted) else None,
                     _pa.species if (_pa is not None and not _pa.fainted) else None))
        obs = self.encoder.encode(battle, legal=None, event_window=tracker.event_window)
        s = self.stats
        s.decisions += 1

        rows = _oracle_rows(battle, rlog)[-EVENT_WINDOW_N:]
        block = obs[OFFSET_EVENT_WINDOW:OFFSET_EVENT_WINDOW + EVENT_WINDOW_DIM] \
            .reshape(EVENT_WINDOW_N, EVENT_TOKEN_DIM)
        n_pad = EVENT_WINDOW_N - len(rows)
        s.checked += 1
        if float(np.abs(block[:n_pad]).sum()) > _TOL:
            s.failures.append(dict(tag=tag, turn=cur_turn, kind="pad_nonzero"))
        for ri, r in enumerate(rows):
            got = tuple(float(x) for x in block[n_pad + ri])
            want = _want_vec(r, cur_turn)
            s.checked += 1
            if any(abs(g - w) > _TOL for g, w in zip(got, want)):
                rec = dict(tag=tag, turn=cur_turn, row=ri, got=got, want=want, record=r)
                if not s.failures:
                    rec["trace"] = [
                        (str(e.kind).replace("EventKind.", ""), e.turn, e.side,
                         e.actor_species, e.value)
                        for e in battle.events_since(0)]
                s.failures.append(rec)

        if battle.finished:
            self._trackers.pop(tag, None)
            self._resync_log.pop(tag, None)
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        self._trackers.pop(battle.battle_tag, None)
        return super()._battle_finished_callback(battle)


def main(n_battles: int = 30) -> int:
    stats = _Stats()
    teams = TeamLoader().get_all_teams()
    p1 = _EventWindowFuzzPlayer(stats=stats, team=Gen3Teambuilder(teams), battle_format="gen3ou",
                                server_configuration=LocalhostServerConfiguration,
                                account_configuration=AccountConfiguration(f"EwA{int(time.time()) % 10000}", None),
                                start_listening=False, max_concurrent_battles=1)
    p2 = _EventWindowFuzzPlayer(stats=_Stats(), team=Gen3Teambuilder(teams), battle_format="gen3ou",
                                server_configuration=LocalhostServerConfiguration,
                                account_configuration=AccountConfiguration(f"EwB{int(time.time()) % 10000}", None),
                                start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(p1, p2, n_battles))
    print(f"[event-window fuzz] {n_battles} battles, {stats.decisions} decisions, "
          f"{stats.checked} checks, {len(stats.failures)} failures")
    if stats.failures:
        for f in stats.failures[:6]:
            print("  FAIL", {k: v for k, v in f.items() if k != "trace"})
        if "trace" in stats.failures[0]:
            for line in stats.failures[0]["trace"][-40:]:
                print("    ", line)
        raise SystemExit(1)
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 30))
