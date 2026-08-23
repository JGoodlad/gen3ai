"""Tier H-B fuzz (gen3_event_window_v1) — the event-window obs block vs PROTOCOL TRUTH.

Real battles in-process via the local BattleStream bridge (no server). At EVERY decision the
player runs the real tracker protocol (EpisodeTracker.record → update_progress_clock → encode
with event_window threaded — the RLPlayer/Gen3Env path) and validates the whole 32×22 block —
EVERY column, none declared unmodelled — against an INDEPENDENT from-scratch fold over the
battle's FULL event log: the H-B type vocabulary, the modifier-attach rules (clause-free
target-matched damage; miss/fail/crit; the effectiveness trio tagged on the MOVER — the
producer's "attach to the resolving mover" convention, which this oracle MIRRORED as
defender-tagged until 2026-08-19 and therefore never caught the tracker's identical flip),
we_first, forced windows, the id columns (species/move dex nums), the three derived id columns
(cant reason, faint CAUSE, item TRANSITION), recency and the front-padding convention.

Any mismatch raises with (row, column, got, want) + an event trace on the first failure.

Run directly:
    python src/agents/training/poke_env_gaps/event_window_fuzz_test.py [n_battles]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
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
    EVENT_T_BOOST, EVENT_T_CANT, EVENT_T_FAINT, EVENT_T_HAZARD, EVENT_T_ITEM_REVEAL,
    EVENT_T_MOVE, EVENT_T_STATUS_APPLIED, EVENT_T_STATUS_CURED, EVENT_T_SWITCH_IN,
    EVENT_T_SWITCH_REJECTED,
    EVENT_STATUS_IDS, EVENT_TOKEN_DIM, EVENT_WINDOW_DIM, EVENT_WINDOW_N,
    ITEM_TR_CONSUMED, ITEM_TR_REMOVED, ITEM_TR_REVEALED, ITEM_TR_SWAPPED,
    OFFSET_EVENT_WINDOW, EVENT_EFF_GROUP, EventCol as C,
)
from agents.observation.gen3_effects import cant_reason_id
from agents.battle.turn_view import FAINT_CAUSE_VOCAB
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.episode_tracker import EpisodeTracker
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
    # Per-DERIVED-column exercise counts. "0 failures" over a column no row ever sets is a
    # vacuous pass, and these three are exactly the ones that were unchecked before — so the
    # run reports how often each was actually put to work rather than leaving it to be assumed.
    derived: dict = field(default_factory=lambda: {"cant": 0, "faint_cause": 0, "item_tr": 0})


def attributable_damage(e) -> bool:
    """Does this DAMAGE event count toward the mover's move magnitude?

    Only clause-free damage does: a `[from]` clause means sandstorm / burn / Leech Seed /
    Recoil, not the move's hit. **Read it through `from_clause`, NEVER `value["from"]`** — the
    parser writes the clause to `value["reason"]` on DAMAGE/HEAL/SETHP/STATUS and to
    `value["from"]` on the effect kinds, so the raw key is unconditionally None here.

    Module-level and named precisely so this file's ORACLE can be unit-tested against the trap
    rather than only against a live battle. It read `value.get("from")` until the
    positional-binding sweep: an always-falsy guard, which made the independent oracle repeat
    the very consumer-side key drift (`from` vs `reason`) the tracker had already been fixed
    for — a fuzz that mirrors its subject's mistake cannot catch that mistake coming back.
    """
    return e.from_clause is None


# --------------------------------------------------------------------------------------- #
# The three DERIVED columns. Where the oracle's independence starts and stops, stated:      #
#                                                                                            #
#   INDEPENDENT (the modelling, and where the real bugs live) — WHICH event a row is derived #
#     from, which mon it is attributed to, and the running state a cause needs: what last    #
#     damaged each side, whether that side's last move was a self-KO, and the RESETS (a mon  #
#     that switches out or faints leaves no chip history to the next occupant — a documented #
#     defect class in the tracker, so the oracle keeps that ledger itself rather than        #
#     reading `_last_dmg_cause` / `_used_selfko`).                                           #
#   SHARED (the declared vocabulary, a contract both sides must agree on by construction) —  #
#     the label→id maps: `FAINT_CAUSE_VOCAB`, the `ITEM_TR_*` ids, `cant_reason_id`'s        #
#     normalization. Same call the STATUS column already makes on `EVENT_STATUS_IDS`.        #
# --------------------------------------------------------------------------------------- #
_ORACLE_SELF_KO_MOVES = frozenset({"explosion", "selfdestruct"})


def oracle_faint_cause_id(from_clause, used_selfko: bool) -> int:
    """WHY a mon fainted → its 1-based `FAINT_CAUSE_VOCAB` id, classified from the `[from]`
    clause of the last damage it took. Written out here rather than calling the battle layer's
    classifier: a residual death (weather / status / hazard / Leech Seed) emits no preceding
    event to infer from, so this column IS the signal for the slow-attrition class and a
    silently-agreeing copy of the producer's branch would test nothing."""
    if used_selfko:
        label = "selfko"
    elif from_clause is None:
        label = "attack"
    else:
        fc = from_clause.strip().lower()
        if fc == "spikes":
            label = "hazard"
        elif fc in ("sandstorm", "hail"):
            label = "weather"
        elif fc in ("psn", "tox", "brn", "burn"):
            label = "status"
        elif "recoil" in fc:
            label = "recoil"
        elif "leech seed" in fc or "leechseed" in fc:
            label = "leechseed"
        else:
            label = "other"
    return FAINT_CAUSE_VOCAB.index(label) + 1


def oracle_item_transition(kind, from_clause) -> int:
    """`|-item|` / `|-enditem|` (+ its `[from]`) → an `ITEM_TR_*` id. Gen3 has three ways an
    item stops being held and they mean different things: a CONSUMED berry was spent, a Knock
    Off REMOVAL is permanent in ADV, and a Trick/Thief/Covet SWAP tells you the opponent now
    holds it. Derived here from the event kind + clause alone."""
    if kind is EventKind.ITEM:
        return ITEM_TR_REVEALED
    fc = (from_clause or "").strip().lower()
    if "knock off" in fc or "knockoff" in fc:
        return ITEM_TR_REMOVED
    if any(w in fc for w in ("trick", "thief", "covet", "switcheroo")):
        return ITEM_TR_SWAPPED
    return ITEM_TR_CONSUMED


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
    # The oracle's OWN cause ledger (see the note above `oracle_faint_cause_id`): per side, the
    # `[from]` clause of the last damage it took and whether its last move self-KO'd. Both are
    # CLEARED when that side's mon leaves the field — a fresh mon inherits no chip history, and
    # getting that wrong makes an incoming mon's first faint read the previous occupant's cause.
    last_dmg_cause = {}
    used_selfko = {}
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
            used_selfko[side] = e.move_id in _ORACLE_SELF_KO_MOVES
        elif k is EventKind.DAMAGE and side:
            last_dmg_cause[side] = e.from_clause        # None ⇒ a direct hit
            mover = OPP if side == OURS else OURS
            om = open_move.get(mover)
            if (om is not None and om["turn"] == et and attributable_damage(e)
                    and sp and om["target"] == sp and e.amount is not None):
                om["mag"] += float(e.amount)
        elif k in (EventKind.MISS, EventKind.FAIL, EventKind.CRIT) and side:
            om = open_move.get(side)
            external_cause = (k is EventKind.FAIL
                              and e.from_clause not in (None, "move-suffix"))
            if om is not None and om["turn"] == et and not external_cause:
                if k is EventKind.MISS:
                    om["miss"], om["hit"] = 1.0, 0.0
                elif k is EventKind.FAIL:
                    om["fail"], om["hit"] = 1.0, 0.0
                else:
                    om["crit"] = 1.0
        elif k in (EventKind.IMMUNE, EventKind.RESISTED, EventKind.SUPEREFFECTIVE) and side:
            om = open_move.get(side)                      # producer tags the MOVER, like crit/miss/fail
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
            last_dmg_cause.pop(side, None)
            used_selfko.pop(side, None)
        elif k is EventKind.FAINT and side and sp:
            if emit(seq):
                rows.append(dict(t=EVENT_T_FAINT, actor=sp, side=side, target=None, move=None,
                                 mag=0.0, hit=0.0, miss=0.0, fail=0.0, crit=0.0, eff=0,
                                 wf=False, status=0, turn=et, fw=flags(),
                                 faint=oracle_faint_cause_id(last_dmg_cause.get(side),
                                                             bool(used_selfko.get(side)))))
            last_dmg_cause.pop(side, None)
            used_selfko.pop(side, None)
            if side == OURS and our_active == sp:
                our_active, forced["ours"] = None, True
            elif side == OPP and opp_active == sp:
                opp_active, forced["opp"] = None, True
        elif k in (EventKind.STATUS, EventKind.CURESTATUS) and sp and emit(seq):
            rows.append(dict(t=(EVENT_T_STATUS_APPLIED if k is EventKind.STATUS
                                else EVENT_T_STATUS_CURED),
                             actor=sp, side=side, target=None, move=None, mag=0.0,
                             hit=0.0, miss=0.0, fail=0.0, crit=0.0, eff=0, wf=False,
                             status=EVENT_STATUS_IDS.get(str(e.status or "").lower(), 0),
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
                             eff=0, wf=False, status=0, turn=et, fw=flags(),
                             item_tr=oracle_item_transition(k, e.from_clause)))
        elif k is EventKind.SIDE and side and emit(seq):
            rows.append(dict(t=EVENT_T_HAZARD, actor=None, side=side, target=None, move=None,
                             mag=0.0, hit=0.0, miss=0.0, fail=0.0, crit=0.0, eff=0,
                             wf=False, status=0, turn=et, fw=flags()))
        elif k is EventKind.CANT and sp and emit(seq):
            # "this mon could not move, and why". ATTRIBUTED TO THE MON THAT LOST ITS TURN, not
            # to the ability holder the protocol files it against (Damp blocking someone else's
            # Explosion is filed on the Damp holder) — `blocked_actor`/`blocked_side` are the
            # event's own typed fields, so this is read from the log, not from the fold.
            rows.append(dict(t=EVENT_T_CANT, actor=(e.blocked_actor or sp),
                             side=(e.blocked_side or side), target=None, move=e.cant_move,
                             mag=0.0, hit=0.0, miss=0.0, fail=0.0, crit=0.0, eff=0, wf=False,
                             status=0, turn=et, fw=flags(), cant=e.reason))
        elif k is EventKind.CHOICE_REJECTED and emit(seq):
            rows.append(dict(t=EVENT_T_SWITCH_REJECTED, actor=our_active, side=OURS,
                             target=None, move=None, mag=0.0, hit=0.0, miss=0.0, fail=0.0,
                             crit=0.0, eff=0, wf=False, status=0, turn=et, fw=flags()))
    return rows


# The columns this oracle does NOT model — now EMPTY, and the constant stays because the
# coverage assert below is what keeps it that way: a new EventCol member must be modelled or
# declared here, never silently unchecked.
#
# gen3_event_col_names_v1 declared `CANT` / `FAINT_CAUSE` / `ITEM_TRANSITION` unmodelled after
# finding them silently unchecked (`_want_vec` returned a 19-tuple compared with `zip(got,
# want)` against a 22-wide row, and `zip` stops at the shorter). The missing CANT ROW was worse
# than the missing columns: the oracle emitted one fewer record per `|cant|` than the tracker,
# so once the 32-row window SATURATED its last-32 started earlier in the timeline and EVERY row
# compared against its neighbour — 8209 failures over 5 battles, all one root. All three are
# modelled now (`oracle_faint_cause_id` / `oracle_item_transition` / the CANT branch).
_ORACLE_UNMODELED_COLS: frozenset = frozenset()


def _want_vec(r, cur_turn):
    """The oracle's expected row, keyed by NAMED column — never a positional tuple."""
    side = 1.0 if r["side"] == OURS else (-1.0 if r["side"] == OPP else 0.0)
    is_move = r["t"] == EVENT_T_MOVE
    mag = max(-1.0, min(1.0, r["mag"])) if is_move else max(-1.0, min(1.0, r["mag"] / 6.0))
    want = {
        C.TYPE: float(r["t"]),
        C.ACTOR_SPECIES: _sp_num(r["actor"]),
        C.ACTOR_SIDE: side,
        C.TARGET_SPECIES: _sp_num(r["target"]),
        C.MOVE: _mv_num(r["move"]),
        C.MAGNITUDE: mag,
        C.OUT_HIT: (r["hit"] if is_move else 0.0),
        C.OUT_MISS: (r["miss"] if is_move else 0.0),
        C.OUT_FAIL: (r["fail"] if is_move else 0.0),
        C.CRIT: (r["crit"] if is_move else 0.0),
        C.WE_FIRST: (1.0 if r["wf"] else 0.0),
        C.STATUS: float(r["status"]),
        C.TURNS_AGO: _norm(max(0, cur_turn - int(r["turn"]))),
        C.FORCED_WINDOW: float(r["fw"]),
        C.VALID: 1.0,
        # The three derived columns — `.get` because each is set by exactly ONE row type and
        # every other type must read a clean 0 (the encoder writes them under the same rule).
        C.CANT: float(cant_reason_id(r.get("cant"))),
        C.FAINT_CAUSE: float(r.get("faint", 0)),
        C.ITEM_TRANSITION: float(r.get("item_tr", 0)),
    }
    for i, col in enumerate(EVENT_EFF_GROUP):
        want[col] = 1.0 if (is_move and int(r["eff"]) == i) else 0.0
    assert set(want) | _ORACLE_UNMODELED_COLS == set(C), (
        "the oracle must state an expectation for every EventCol member or declare it "
        f"unmodelled — missing: {sorted(set(C) - set(want) - _ORACLE_UNMODELED_COLS)}")
    return want


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
            for _key, _col in (("cant", C.CANT), ("faint_cause", C.FAINT_CAUSE),
                               ("item_tr", C.ITEM_TRANSITION)):
                if want[_col] > 0.0:
                    s.derived[_key] += 1
            if any(abs(got[int(c)] - w) > _TOL for c, w in want.items()):
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
    print(f"  derived-column rows exercised: {stats.derived}")
    _idle = [k for k, v in stats.derived.items() if v == 0]
    if _idle:
        # Not a failure: an item transition needs a battle where an item is actually spent or
        # knocked off, which a short run can legitimately miss. But a green run over a column
        # nothing set proves nothing about it, so say which ones.
        print(f"  ⚠️ COVERAGE: no row ever set {_idle} — those columns were not exercised; "
              f"re-run with more battles before reading this as a pass on them.")
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
