"""Tier H-B fuzz (gen3_event_window_v1 → gen3_event_record_v2) — the event-window obs block vs
PROTOCOL TRUTH.

Real battles in-process via the local BattleStream bridge (no server). At EVERY decision the
player runs the real tracker protocol (EpisodeTracker.record → update_progress_clock → encode
with event_window threaded → advance(action) — the RLPlayer/Gen3Env path) and validates the whole
32×30 block — EVERY column, none declared unmodelled — against an INDEPENDENT from-scratch fold
over the battle's FULL event log: the H-B type vocabulary, the modifier-attach rules (clause-free
target-matched damage; miss/fail/crit; the effectiveness trio tagged on the MOVER — the
producer's "attach to the resolving mover" convention, which this oracle MIRRORED as
defender-tagged until 2026-08-19 and therefore never caught the tracker's identical flip), the
MOVE row's implied target (user / foe / none from the move's dex target class), we_first, forced
windows, the id columns (species/move dex nums), the derived id columns (cant reason, faint
CAUSE over the LIVE vocabulary incl. destinybond / perishsong, item TRANSITION incl. the
transfer DIRECTION), recency and the front-padding convention.

gen3_event_record_v2 (E12) columns 22–29, all modelled: REL species + side (the replaced /
passer / dragged-out mon on a SWITCH_IN, the KOer on an `attack` FAINT, the denier on a DENIED
row, the other party of an item transfer / removal), the SWITCH_IN ENTRY reason (chosen /
replacement / drag / Baton Pass, with the causing move on the row's MOVE and the Spikes entry
chip on its MAGNITUDE), the DENIED row type and its DENIAL reason (fainted first / TURN CUT —
gen 3 singles: any action-phase faint cancels every queued action), CALLER, BOOST STAT, Spikes
LAYERS and PURSUIT_SWITCH. The refused-switch (E4) TARGET is derived from the switch this
player actually SENT (its own BattleOrder), never from the tracker's decode.

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
    EVENT_T_BOOST, EVENT_T_CANT, EVENT_T_DENIED, EVENT_T_FAINT, EVENT_T_HAZARD,
    EVENT_T_ITEM_REVEAL, EVENT_T_MOVE, EVENT_T_STATUS_APPLIED, EVENT_T_STATUS_CURED,
    EVENT_T_SWITCH_IN, EVENT_T_SWITCH_REJECTED,
    EVENT_STAT_IDS, EVENT_STATUS_IDS, EVENT_TOKEN_DIM, EVENT_WINDOW_DIM, EVENT_WINDOW_N,
    ENTRY_BATON_PASS, ENTRY_CHOSEN, ENTRY_DRAG, ENTRY_REPLACEMENT,
    DENIAL_FAINTED_FIRST, DENIAL_TURN_CUT,
    ITEM_TR_CONSUMED, ITEM_TR_RECEIVED, ITEM_TR_REMOVED, ITEM_TR_REVEALED, ITEM_TR_SWAPPED,
    OFFSET_EVENT_WINDOW, EVENT_EFF_GROUP, EventCol as C,
)
from agents.observation.gen3_effects import cant_reason_id
from agents.battle.turn_view import FAINT_CAUSE_VOCAB, FAINT_CAUSE_VOCAB_LIVE
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
    # gen3_event_record_v2 (E12): the same exercise ledger for the new columns and row type, split
    # by the VALUE that matters (a Baton Pass entry, a turn cut, a received item …), because each
    # is a separate branch of the fold and "0 failures" over an unset branch is a vacuous pass.
    e12: dict = field(default_factory=dict)


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


def oracle_faint_cause_id(from_clause, used_selfko: bool, lethal: bool = True) -> int:
    """WHY a mon fainted → its 1-based `FAINT_CAUSE_VOCAB` id, classified from the `[from]`
    clause of the last damage it took. Written out here rather than calling the battle layer's
    classifier: a residual death (weather / status / hazard / Leech Seed) emits no preceding
    event to infer from, so this column IS the signal for the slow-attrition class and a
    silently-agreeing copy of the producer's branch would test nothing."""
    if used_selfko:
        label = "selfko"
    elif not lethal:
        # no damage line took it to 0 HP (Destiny Bond, Perish Song, Memento) — not an attack
        # (gen3_event_window_semantics_fixes_v1, W2)
        label = "other"
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
    Off REMOVAL is permanent in ADV, and a Trick/Thief/Covet transfer tells you the opponent now
    holds it. Derived here from the event kind + clause alone.

    gen3_event_record_v2 (E12) splits a transfer by DIRECTION: the `|-enditem|` side LOST the
    item (SWAPPED — taken from this mon), the `|-item|` side RECEIVED one (RECEIVED; Trick writes
    two receipts)."""
    fc = (from_clause or "").strip().lower()
    if any(w in fc for w in ("trick", "thief", "covet", "switcheroo")):
        return ITEM_TR_SWAPPED if kind is EventKind.ENDITEM else ITEM_TR_RECEIVED
    if kind is EventKind.ITEM:
        return ITEM_TR_REVEALED
    if "knock off" in fc or "knockoff" in fc:
        return ITEM_TR_REMOVED
    return ITEM_TR_CONSUMED


def oracle_live_faint_cause_id(label: str) -> int:
    """A LIVE-vocabulary faint label → its 1-based id (`FAINT_CAUSE_VOCAB_LIVE`: the archive's
    eight, then `destinybond` / `perishsong`). The archive prefix is shared, so an id
    `oracle_faint_cause_id` returns is already a live id."""
    return FAINT_CAUSE_VOCAB_LIVE.index(label) + 1


# The dex `target` classes whose `|move|` line names the USER (Showdown writes the target into
# the line before any `[still]` blanks it): self-targeting, the user's side, the whole field.
# `adjacentAlly` (Helping Hand) has no target in singles. Curse is the one gen-3 move whose target
# depends on the USER's typing (a non-Ghost Curse targets itself). Stated here from the dex, not
# imported from the battle layer's classifier.
_ORACLE_USER_TARGETS = frozenset({"self", "allySide", "allyTeam", "adjacentAllyOrSelf",
                                  "allies", "all"})


def oracle_move_target(move_id, user_sp, foe_sp):
    d = gen3_movedex.get(move_id) if move_id else None
    if d is None:
        return foe_sp
    if d.id == "curse":
        u = gen3_data.species.get(user_sp) if user_sp else None
        return user_sp if (u is not None and "GHOST" not in u.types) else foe_sp
    if d.target in _ORACLE_USER_TARGETS:
        return user_sp
    if d.target == "adjacentAlly":
        return None
    return foe_sp


# An end-of-turn RESIDUAL line, recognised by its `[from]` (prefix `item:` / `move:` / `ability:`
# stripped): the action phase is over once one prints, so a faint after it cuts nothing
# (designs/ARCHITECTURE.md §1.6, ACTION DENIAL).
_ORACLE_RESIDUAL = frozenset({"psn", "tox", "brn", "sandstorm", "hail", "leech seed",
                              "nightmare", "curse", "leftovers", "wish", "ingrain",
                              "future sight", "doom desire"})


def oracle_is_residual(from_clause) -> bool:
    fc = (from_clause or "").strip().lower()
    if ":" in fc and fc.split(":", 1)[0] in ("item", "move", "ability"):
        fc = fc.split(":", 1)[1].strip()
    return fc in _ORACLE_RESIDUAL


_ACTION_KINDS = (EventKind.MOVE, EventKind.SWITCH, EventKind.DRAG, EventKind.CANT,
                 EventKind.CHOICE_REJECTED)


def _blank(t, actor, side, turn, fw, **kw):
    r = dict(t=t, actor=actor, side=side, target=None, move=None, mag=0.0, hit=0.0, miss=0.0,
             fail=0.0, crit=0.0, eff=0, wf=False, status=0, turn=turn, fw=fw,
             rel=None, rel_side=None, entry=0, denial=0, caller=None, stat=0, layers=0,
             pursuit=False)
    r.update(kw)
    return r


class _TurnLedger:
    """ONE turn's action phase, as the oracle reads the gen-3 rule: the ACTORS are the two
    actives when the turn's first line prints; any faint while the phase is open cancels every
    action still queued. The phase is a small ledger (who acted, which faints happened, which
    denial rows are owed) that the fold SETTLES at the next settle point — an action line, a
    residual line, the next turn, or a decision — rather than a state machine threaded through
    the event loop."""

    def __init__(self, turn, actors):
        self.turn = turn
        self.actors = dict(actors)          # side -> species (None ⇒ no actor)
        self.done = {s: False for s in actors}
        self.faints = []                    # [(species, side, cause_id, action)] in order
        self.owed = []                      # fainted-first rows not yet placed (their fields)
        self.open = True

    def acted(self, side, species):
        if side in self.actors and species and self.actors[side] == species:
            self.done[side] = True


def _oracle_rows(battle, resync_log):
    """From-scratch fold over the FULL log → the expected record dicts (all of them; the
    caller windows to the last EVENT_WINDOW_N). ``resync_log`` = this test's own record of the
    (seq watermark, our_active, opp_active, switch_sent) marks it made at each decision,
    replayed in event order by seq so the running actives match the tracker's; ``switch_sent``
    is the species of the switch this player's order at that decision aimed at (or None)."""
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
    last_dmg_lethal = {}
    used_selfko = {}
    # E12 state, each keyed by side: the fainted mon a side's next entry replaces; the turn a
    # side's (unfailed) Baton Pass is pending on; Spikes layers; the SWITCH_IN row still taking
    # its entry chip; a Perish count at 0 (the species) and a triggered Destiny Bond (the victim
    # side). `last_action` is the latest MOVE as (actor, side, move) — a switch / cant / refusal
    # / a new turn clears it, so a faint's KOer is only ever the action that printed before it.
    replaces, baton_on, layers, entry_row = {}, {}, {}, {}
    perish_at_zero, bonded = {}, {}
    last_action = None
    ledger = None
    switch_sent = None
    mover_now = None       # the side of the latest |move| — whose bare -damage is a hit
    resync = sorted(resync_log, key=lambda r: r[0])     # by seq watermark

    def flags():
        return 1.0 if (forced["ours"] or forced["opp"]) else 0.0

    def active(side):
        return our_active if side == OURS else opp_active

    def other(side):
        return OPP if side == OURS else OURS if side == OPP else None

    def settle(cut, close):
        """Place what the ledger owes: its fainted-first rows (in faint order), then — when a
        faint has happened and the settle point cuts — one TURN-CUT row per actor (ours, then
        theirs) that neither acted nor fainted. `close` ends the action phase."""
        if ledger is None:
            return
        for kw in ledger.owed:
            rows.append(_blank(EVENT_T_DENIED, fw=flags(), **kw))
        ledger.owed = []
        if cut and ledger.faints:
            f_sp, f_side, f_cause, f_act = ledger.faints[0]
            for sd in (OURS, OPP):
                if ledger.actors.get(sd) and not ledger.done[sd]:
                    ledger.done[sd] = True
                    rows.append(_blank(EVENT_T_DENIED, ledger.actors[sd], sd, ledger.turn,
                                       flags(), denial=DENIAL_TURN_CUT, rel=f_sp,
                                       rel_side=f_side, faint=f_cause,
                                       move=(f_act[2] if f_act else None)))
        if close:
            ledger.open = False

    def decision(mark):
        """A decision closes the fold's window: a faint that cut the turn ends its action phase
        here; otherwise (a mid-turn decision) the phase stays open. Then the actives resync."""
        nonlocal our_active, opp_active, switch_sent
        cut = ledger is not None and bool(ledger.faints)
        settle(cut=cut, close=cut)
        _, o, p, sent = mark
        if o:
            our_active, forced["ours"] = o, False
        if p:
            opp_active, forced["opp"] = p, False
        switch_sent = sent

    def forget(side):
        for d in (last_dmg_cause, last_dmg_lethal, used_selfko, perish_at_zero, bonded):
            d.pop(side, None)

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
        # apply every decision that happened BEFORE this event
        while ri < len(resync) and resync[ri][0] <= seq:
            decision(resync[ri])
            ri += 1
        side, sp, et = e.side, e.actor_species, e.turn
        k = e.kind
        fc_low = (e.from_clause or "").strip().lower()
        # ---- the turn's action phase: open a new ledger on a new turn, settle at action lines
        # and at the first residual line.
        if ledger is None or ledger.turn != et:
            if ledger is not None and ledger.open and ledger.turn > 0:
                settle(cut=True, close=True)
            ledger = _TurnLedger(et, {OURS: our_active, OPP: opp_active})
            last_action = None
        if k in _ACTION_KINDS:
            had_faint = bool(ledger.faints)
            settle(cut=True, close=had_faint)
        elif ledger.open and oracle_is_residual(e.from_clause):
            settle(cut=True, close=True)

        if k is EventKind.MOVE and side and sp:
            if emit(seq):
                if first_mover.get(et) is None:
                    first_mover[et] = side
                r = _blank(EVENT_T_MOVE, sp, side, et, flags(),
                           target=oracle_move_target(e.move_id, sp, active(other(side))),
                           move=e.move_id, hit=1.0, wf=(side == first_mover[et]),
                           caller=e.from_move, pursuit=bool(e.value.get("pursuit_switch")))
                rows.append(r)
                open_move[side] = r
            used_selfko[side] = e.move_id in _ORACLE_SELF_KO_MOVES
            last_action = (sp, side, e.move_id)
            ledger.acted(side, sp)
            if e.move_id == "batonpass":
                baton_on[side] = et
        if k is EventKind.MOVE and side:
            mover_now = side
        elif k is EventKind.DAMAGE and side:
            last_dmg_cause[side] = e.from_clause        # None ⇒ a direct hit
            _ha = e.value.get("hp_after")
            last_dmg_lethal[side] = _ha is None or float(_ha) <= 0.0
            if fc_low == "confusion":
                ledger.acted(side, sp)                   # hit itself: it DID take its turn
            er = entry_row.get(side)
            if (fc_low == "spikes" and er is not None and er["turn"] == et
                    and sp and er["actor"] == sp and e.amount is not None):
                er["mag"] += float(e.amount)             # the entry chip rides the entry row
            mover = OPP if side == OURS else OURS
            om = open_move.get(mover)
            if (om is not None and om["turn"] == et and attributable_damage(e)
                    and mover_now == mover
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
                    if om["move"] == "batonpass":
                        baton_on.pop(side, None)         # a failed pass passes nothing
                else:
                    om["crit"] = 1.0
        elif k is EventKind.ACTIVATE and side and str(e.effect or "").strip().lower() in (
                "protect", "detect", "move: protect", "move: detect"):
            # a Protect / Detect BLOCK names the protector; the moving side's move failed (W4)
            mover = OPP if side == OURS else OURS
            om = open_move.get(mover)
            if om is not None and om["turn"] == et and mover_now == mover:
                om["fail"], om["hit"] = 1.0, 0.0
        elif k is EventKind.ACTIVATE and side and "destiny" in str(e.effect or "").lower():
            bonded[other(side)] = True                   # the bond's USER names it; the foe dies
        elif (k is EventKind.VOLATILE_START and side and sp
              and str(e.effect or "").strip().lower() == "perish0"):
            perish_at_zero[side] = sp
        elif k in (EventKind.IMMUNE, EventKind.RESISTED, EventKind.SUPEREFFECTIVE) and side:
            om = open_move.get(side)                      # producer tags the MOVER, like crit/miss/fail
            if om is not None and om["turn"] == et:
                om["eff"] = {EventKind.SUPEREFFECTIVE: 1, EventKind.RESISTED: 2,
                             EventKind.IMMUNE: 3}[k]
        elif k in (EventKind.SWITCH, EventKind.DRAG) and side and sp:
            leaving = active(side)
            cause_move = None
            if k is EventKind.DRAG:
                entry, rel = ENTRY_DRAG, leaving
                phazer = open_move.get(other(side))
                if phazer is not None and phazer["turn"] == et:
                    cause_move = phazer["move"]
            elif replaces.get(side):
                entry, rel = ENTRY_REPLACEMENT, replaces[side]
            elif baton_on.get(side) == et:
                entry, rel, cause_move = ENTRY_BATON_PASS, leaving, "batonpass"
            else:
                entry, rel = ENTRY_CHOSEN, leaving
            if entry in (ENTRY_CHOSEN, ENTRY_BATON_PASS):
                ledger.acted(side, leaving)
            baton_on.pop(side, None)
            replaces.pop(side, None)
            if emit(seq):
                r = _blank(EVENT_T_SWITCH_IN, sp, side, et, flags(),
                           target=active(other(side)), move=cause_move, entry=entry,
                           rel=rel, rel_side=(side if rel else None),
                           layers=layers.get(side, 0))
                rows.append(r)
                entry_row[side] = r
            forced["ours" if side == OURS else "opp"] = False
            if side == OURS:
                our_active = sp
            else:
                opp_active = sp
            forget(side)
        elif k is EventKind.FAINT and side and sp:
            if perish_at_zero.get(side) == sp:
                cause = oracle_live_faint_cause_id("perishsong")
            elif bonded.get(side):
                cause = oracle_live_faint_cause_id("destinybond")
            else:
                cause = oracle_faint_cause_id(last_dmg_cause.get(side),
                                              bool(used_selfko.get(side)),
                                              bool(last_dmg_lethal.get(side)))
            koer = (last_action if (cause == oracle_live_faint_cause_id("attack")
                                    and last_action is not None and last_action[1] != side)
                    else None)
            if emit(seq):
                rows.append(_blank(EVENT_T_FAINT, sp, side, et, flags(), faint=cause,
                                   rel=(koer[0] if koer else None),
                                   rel_side=(koer[1] if koer else None),
                                   layers=(layers.get(side, 0)
                                           if cause == oracle_live_faint_cause_id("hazard")
                                           else 0)))
            if ledger.open and et > 0:
                ledger.faints.append((sp, side, cause, last_action))
                if ledger.actors.get(side) == sp and not ledger.done[side]:
                    ledger.done[side] = True
                    la = last_action
                    ledger.owed.append(dict(actor=sp, side=side, turn=et,
                                            denial=DENIAL_FAINTED_FIRST, faint=cause,
                                            move=(la[2] if la else None),
                                            rel=(la[0] if la else None),
                                            rel_side=(la[1] if la else None)))
            forget(side)
            entry_row.pop(side, None)
            replaces[side] = sp
            if side == OURS and our_active == sp:
                our_active, forced["ours"] = None, True
            elif side == OPP and opp_active == sp:
                opp_active, forced["opp"] = None, True
        elif k in (EventKind.STATUS, EventKind.CURESTATUS) and sp and emit(seq):
            rows.append(_blank((EVENT_T_STATUS_APPLIED if k is EventKind.STATUS
                                else EVENT_T_STATUS_CURED), sp, side, et, flags(),
                               status=EVENT_STATUS_IDS.get(str(e.status or "").lower(), 0)))
        elif k in (EventKind.BOOST, EventKind.UNBOOST) and sp and emit(seq):
            rows.append(_blank(EVENT_T_BOOST, sp, side, et, flags(),
                               mag=float(e.amount or 0.0),      # amount is already SIGNED (W1)
                               stat=(EVENT_STAT_IDS[str(e.value["stat"]).strip().lower()]
                                     if e.value.get("stat") else 0)))
        elif k in (EventKind.ITEM, EventKind.ENDITEM) and sp and emit(seq):
            tr = oracle_item_transition(k, e.from_clause)
            partner_side = (other(side) if tr in (ITEM_TR_SWAPPED, ITEM_TR_RECEIVED,
                                                  ITEM_TR_REMOVED) else None)
            partner = active(partner_side) if partner_side else None
            rows.append(_blank(EVENT_T_ITEM_REVEAL, sp, side, et, flags(), item_tr=tr,
                               rel=partner, rel_side=(partner_side if partner else None)))
        elif k is EventKind.SIDE and side:
            starts = e.value.get("op") != "sideend"
            spikes = "spikes" in str(e.value.get("condition") or "").lower()
            if spikes:
                layers[side] = min(3, layers.get(side, 0) + 1) if starts else 0
            if emit(seq):
                rows.append(_blank(EVENT_T_HAZARD, None, side, et, flags(),
                                   mag=(1.0 if starts else -1.0),
                                   layers=(layers.get(side, 0) if spikes else 0)))
        elif k is EventKind.CANT and sp:
            # "this mon could not move, and why". ATTRIBUTED TO THE MON THAT LOST ITS TURN, not
            # to the ability holder the protocol files it against (Damp blocking someone else's
            # Explosion is filed on the Damp holder) — `blocked_actor`/`blocked_side` are the
            # event's own typed fields, so this is read from the log, not from the fold.
            lost_sp, lost_side = (e.blocked_actor or sp), (e.blocked_side or side)
            ledger.acted(lost_side, lost_sp)             # a refused action still took its turn
            if emit(seq):
                rows.append(_blank(EVENT_T_CANT, lost_sp, lost_side, et, flags(),
                                   move=e.cant_move, cant=e.reason))
        elif k is EventKind.CHOICE_REJECTED and emit(seq):
            # E4: the target is the switch THIS player sent at its latest decision — read from
            # its own order (`switch_sent`), which the server's `|error|` never names.
            rows.append(_blank(EVENT_T_SWITCH_REJECTED, our_active, OURS, et, flags(),
                               target=switch_sent))
        if k in (EventKind.SWITCH, EventKind.DRAG, EventKind.CANT, EventKind.CHOICE_REJECTED):
            last_action = None
    # the decisions at/after the last event — the CURRENT one settles what the window owes
    while ri < len(resync):
        decision(resync[ri])
        ri += 1
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
# modelled now (`oracle_faint_cause_id` / `oracle_item_transition` / the CANT branch), and so
# are gen3_event_record_v2's eight (REL species/side, ENTRY, DENIAL, CALLER, STAT, LAYERS,
# PURSUIT_SWITCH) plus its DENIED row type — the same row-COUNT hazard, which is why the
# DENIED rows are placed exactly where the gen-3 rule settles them.
_ORACLE_UNMODELED_COLS: frozenset = frozenset()


def _e12_keys(r):
    """The E12 branches a row exercises — the coverage ledger's keys."""
    t = r["t"]
    if t == EVENT_T_SWITCH_IN:
        yield f"entry={int(r['entry'])}"
        if r["mag"]:
            yield "spikes_entry_chip"
        if r["layers"]:
            yield "switch_in_layers"
    elif t == EVENT_T_DENIED:
        yield f"denial={int(r['denial'])}"
    elif t == EVENT_T_FAINT:
        if r["rel"]:
            yield "faint_koer"
        if r.get("faint", 0) > len(FAINT_CAUSE_VOCAB):
            yield f"faint={FAINT_CAUSE_VOCAB_LIVE[int(r['faint']) - 1]}"
    elif t == EVENT_T_ITEM_REVEAL and r["rel"]:
        yield f"item_rel_tr={int(r['item_tr'])}"
    elif t == EVENT_T_MOVE:
        if r["caller"]:
            yield "caller"
        if r["pursuit"]:
            yield "pursuit_switch"
    elif t == EVENT_T_BOOST and r["stat"]:
        yield "boost_stat"
    elif t == EVENT_T_HAZARD and r["layers"]:
        yield "hazard_layers"
    elif t == EVENT_T_SWITCH_REJECTED:
        yield "switch_rejected" + ("_target" if r["target"] else "_no_target")


def _want_vec(r, cur_turn):
    """The oracle's expected row, keyed by NAMED column — never a positional tuple."""
    side = 1.0 if r["side"] == OURS else (-1.0 if r["side"] == OPP else 0.0)
    is_move = r["t"] == EVENT_T_MOVE
    # only a BOOST row carries stage units; a HAZARD row's ±1 is written as is
    mag = max(-1.0, min(1.0, r["mag"] / 6.0)) if r["t"] == EVENT_T_BOOST else max(-1.0, min(1.0, r["mag"]))
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
        # gen3_event_record_v2 (E12) — every row type states all eight; the ones a type does not
        # set are 0 by `_blank`, and must READ 0.
        C.REL_SPECIES: _sp_num(r["rel"]),
        C.REL_SIDE: (1.0 if r["rel_side"] == OURS else (-1.0 if r["rel_side"] == OPP else 0.0)),
        C.ENTRY: float(r["entry"]),
        C.DENIAL: float(r["denial"]),
        C.CALLER: _mv_num(r["caller"]),
        C.STAT: float(r["stat"]),
        C.LAYERS: float(r["layers"]) / 3.0,
        C.PURSUIT_SWITCH: 1.0 if r["pursuit"] else 0.0,
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
        self._resync_log: dict = {}      # tag -> [[seq watermark, our_sp, opp_sp, switch_sent]]

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
        rlog.append([battle.event_cursor,
                     _oa.species if (_oa is not None and not _oa.fainted) else None,
                     _pa.species if (_pa is not None and not _pa.fainted) else None,
                     None])
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
            for _key in _e12_keys(r):
                s.e12[_key] = s.e12.get(_key, 0) + 1
            if any(abs(got[int(c)] - w) > _TOL for c, w in want.items()):
                rec = dict(tag=tag, turn=cur_turn, row=ri, got=got, want=want, record=r)
                if not s.failures:
                    rec["trace"] = [
                        (str(e.kind).replace("EventKind.", ""), e.turn, e.side,
                         e.actor_species, e.value)
                        for e in battle.events_since(0)]
                s.failures.append(rec)

        order = self.choose_random_move(battle)
        # The RLPlayer path's `advance(action)`: the tracker decodes a refused switch's TARGET
        # (E4) from the action index against its own team order. The ORACLE never sees that
        # index — it records the species of the switch this order actually SENT.
        mon = getattr(order, "order", None)
        team = list(battle.team.values())
        if mon is not None and hasattr(mon, "species") and any(m is mon for m in team):
            tracker.advance(next(i for i, m in enumerate(team) if m is mon))
            rlog[-1][3] = mon.species
        else:
            tracker.advance(6)                   # a move (or struggle / default): not a switch
        if battle.finished:
            self._trackers.pop(tag, None)
            self._resync_log.pop(tag, None)
        return order

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
    print(f"  E12 branches exercised (row-checks): {dict(sorted(stats.e12.items()))}")
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
