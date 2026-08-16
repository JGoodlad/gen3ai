from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from poke_env.battle.abstract_battle import DamagingMoveEvent

from agents.enums import PokemonType

from agents.action.constants import MOVE_START, MOVE_END
from agents.action.ordering_integrity import (
    reorder_move_bits_to_sorted,
    assert_sorted_validity_correct,
)
from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta
from agents.training.hidden_power_tracker import HiddenPowerTracker
from agents.training.progress_clock import ProgressClock
from agents.training.slot_registry import SlotRegistry

if TYPE_CHECKING:
    from agents.enums import Status

    from agents.battle.live_view import LiveView
    from agents.observation.turn_delta_encoder import TurnDeltaEncoder


@dataclass(frozen=True)
class _HpTargetMon:
    """Minimal mon-shaped object for HiddenPowerTracker.observe().

    Carries exactly the attributes effective_multiplier() reads. The status
    field is the target's status AT MOVE-FIRE TIME (sourced from the
    DamagingMoveEvent captured by the protocol parser), not the current live
    status — needed because Gen 3 Fire moves thaw their target in the same hit
    that resolves them.
    """
    species: str
    type_1: "PokemonType"
    type_2: "PokemonType | None"
    ability: "str | None"
    status: "Status | None"


def _wrap_hp_target(live: "LiveView", event: DamagingMoveEvent) -> Optional[_HpTargetMon]:
    """Look up the target's current type/ability from the LiveView and overlay the
    status captured at the moment HP fired.

    The target of an opponent's Hidden Power is always one of OUR mons, so resolve it
    against ``live.ours``. Returns None if the species can't be resolved (shouldn't
    happen — every HP hit names a real teammate).

    The defender's types are reconstructed from ``LivePokemon.types`` (current-board,
    lowercased names) into the ``PokemonType`` enums ``effective_multiplier`` reads. This
    is value-identical to the old ``mon.type_1``/``mon.type_2`` reads: poke-env defines all
    three properties off the same ``_temporary_types`` / ``_type_1`` / ``_type_2`` state, so
    ``types`` reconstructed to ``(type_1, type_2)`` equals the pair the properties return —
    type-change (Conversion / Camouflage) cases included.
    """
    live_mon = live.ours.get(event.target_species)
    if live_mon is None or not live_mon.types:
        return None
    type_1 = PokemonType[live_mon.types[0].upper()]
    type_2 = (
        PokemonType[live_mon.types[1].upper()] if len(live_mon.types) > 1 else None
    )
    return _HpTargetMon(
        species=live_mon.species,
        type_1=type_1,
        type_2=type_2,
        ability=live_mon.ability,
        status=event.target_status,
    )


class RecencyTracker:
    """E9 step 1 (roadmap §3.9): per-(side, species) RECENCY — the first
    history-attaches-to-entities increment. Three per-mon facts:

      * seen    — last ON FIELD (the live actives + SWITCH events)
      * acted   — last EXECUTED a move (MOVE events)
      * was_hit — last TOOK damage (DAMAGE events — the event's (side, actor_species)
                  attribution names the mon AFFECTED)

    TURN-ANCHORED: we store each fact's latest EVENT TURN and the counter is simply
    `cur_turn − event_turn` (an event from LAST turn reads 1 — "one turn ago"; the on-field
    mon reads 0). Crucially the value is INVARIANT to which decision's window processed the
    event (the earlier tick-then-reset form read differently across forced-switch turns —
    caught by the recency fuzz). Events come from the SAME per-decision
    window the TurnDelta fold reads; both sides PUBLIC. ``values()`` log-saturates over a
    10-turn cap (the ``turns_since_progress`` convention); a never-tracked mon reads 1.0 —
    max staleness, the honest default for a mon that has not appeared this episode."""

    _SAT = 10

    def __init__(self):
        self._turn: int = 0
        self._seen: dict = {}
        self._acted: dict = {}
        self._hit: dict = {}

    def update(self, turn: int, events, our_active: Optional[str],
               opp_active: Optional[str]) -> None:
        from agents.battle.battle_event import OURS, OPP, EventKind
        self._turn = max(self._turn, int(turn))
        for e in events or []:
            if not getattr(e, "actor_species", None) or getattr(e, "side", None) is None:
                continue
            key = (e.side, e.actor_species)
            et = int(getattr(e, "turn", turn))
            if e.kind is EventKind.MOVE:
                self._acted[key] = max(self._acted.get(key, et), et)
                self._seen[key] = max(self._seen.get(key, et), et)
            elif e.kind is EventKind.SWITCH:
                self._seen[key] = max(self._seen.get(key, et), et)
            elif e.kind is EventKind.DAMAGE:
                self._hit[key] = max(self._hit.get(key, et), et)
        for side, sp in ((OURS, our_active), (OPP, opp_active)):
            if sp:
                key = (side, sp)
                self._seen[key] = max(self._seen.get(key, 0), self._turn)  # on field NOW → 0

    def values(self, side: str, species: Optional[str]):
        """→ (seen, acted, was_hit), each log-saturated to [0, 1]."""
        import math
        out = []
        for d in (self._seen, self._acted, self._hit):
            last = None if species is None else d.get((side, species))
            n = self._SAT if last is None else max(0, self._turn - last)
            out.append(math.log1p(min(n, self._SAT)) / math.log(11.0))
        return tuple(out)


def _pair_sat_norm(n: int, sat: int = 10) -> float:
    """The H-A log-saturation convention (design_history_entity.md §3):
    ``log(1 + min(n, 10)) / log(11)`` — same curve the recency triplet and
    ``turns_since_progress`` use."""
    import math
    return math.log1p(min(max(int(n), 0), sat)) / math.log(sat + 1.0)


# Gen3 damaging moves the dex records at basePower 0 (fixed / variable / calculated power).
# Without this set they would classify as "status clicks" in the H-A2 fold. Bare + typed
# Hidden Power are handled by prefix (the bare dex entry also reads 0 BP).
_ZERO_BP_DAMAGING = frozenset({
    "seismictoss", "nightshade", "sonicboom", "dragonrage", "psywave", "superfang",
    "counter", "mirrorcoat", "bide", "endeavor", "return", "frustration", "flail",
    "reversal", "magnitude", "present", "lowkick", "spitup", "beatup",
})


def _move_is_damaging(move_id: "str | None") -> bool:
    """Dex-based damaging-vs-status split for the H-A2 attack/status click counters.
    ``gen3_data.moves.is_damaging`` (base_power > 0) plus the fixed/variable-power overlay
    above — the dex records those at BP 0, and a Seismic Toss click is an attack, not a
    status click."""
    if not move_id:
        return False
    if move_id.startswith("hiddenpower") or move_id in _ZERO_BP_DAMAGING:
        return True
    from agents import gen3_data
    return gen3_data.moves.is_damaging(move_id)


class PairHistoryTracker:
    """Tier H-A (gen3_pair_history_v1, `designs/ai_v9/design_history_entity.md` §3): the
    compiled last-action facts (H-A1) and the pair-history counters (H-A2), folded from
    the SAME per-decision event window the TurnDelta fold and the recency triplet read.
    PUBLIC protocol events only; within-battle only.

    **H-A1 — per-side last action.** A side's most recent MOVE or SWITCH/DRAG event:
    ``last_action(side)`` → ``(move_id | None, was_switch, outcome | None, crit)``.
    ``outcome`` mirrors ``TurnView.SideTurn.outcome`` exactly ("miss" > "fail" > "hit"
    precedence; ``None`` when the last action was a switch). A DRAG (phaze) counts as a
    switch, like ``TurnView.switched``. CANT windows do not replace the last action (a
    fully-prevented turn leaves the previous action standing — recency's ``acted`` carries
    staleness). MISS/FAIL/CRIT events attach to their side's current same-turn move (the
    protocol emits them adjacent to the |move| line, attributed to the MOVER's side).
    ``move_id`` is the protocol-reported EXECUTED move for both sides (delegation-aware:
    the last |move| line wins, so Sleep Talk's called move replaces it) — our own typed
    Hidden Power stays the bare wire id, matching what both sides observe.

    **H-A2 — pair counters, per (their mon i, our mon j).** Keyed by SPECIES within the
    battle (species ↔ team slot is a stable 1:1 within a battle — the species clause —
    and the encoder joins values onto slots through the same team-list order the recency
    block uses, so the obs cell (i, j) always names the same two entities all battle):

      * switch_ins — chosen SWITCH events by their mon i while our mon j was on field
        (DRAG — being phazed in — is not a choice and is not counted)
      * attacks — damaging MOVE clicks by i while j was our active (dex damaging split)
      * status_clicks — non-damaging MOVE clicks by i while j was our active
      * shared_field_turns — distinct game turns i and j were observed sharing the field
      * last paired turn — feeds ``recency_of_last_pairing`` (cur_turn − last, like
        ``since_seen``; never-paired reads 1.0)

    "While j active" is the event-ordered fold: the running actives are advanced by
    SWITCH/DRAG events (the arriving mon), cleared by the active's FAINT, and resynced to
    the decision-time LiveView actives each update. Pairings are observed at switch,
    move, and decision points, counted once per distinct turn per pair.

    **Idempotent by seq**: only events with ``seq > _max_seq`` are processed, so an
    overlapping window or a rolled-back opponent RE-DECIDE can never double-count
    (the counters are sums, unlike recency's max-anchored turns).

    ``pair_values()`` log-saturates every cell over the 10 cap (`_pair_sat_norm`)."""

    _SAT = 10

    def __init__(self):
        self._max_seq: int = -1
        self._turn: int = 0
        self._our_active: Optional[str] = None
        self._opp_active: Optional[str] = None
        self._last: dict = {}          # side -> last-action record (mutable dict)
        self._switch_ins: dict = {}    # (opp_sp, our_sp) -> count
        self._attacks: dict = {}
        self._status_clicks: dict = {}
        self._shared_count: dict = {}
        self._shared_last_turn: dict = {}  # (opp_sp, our_sp) -> last turn counted

    def _observe_pairing(self, t: int) -> None:
        i, j = self._opp_active, self._our_active
        if not i or not j:
            return
        key = (i, j)
        last = self._shared_last_turn.get(key)
        if last is None or t > last:
            self._shared_count[key] = self._shared_count.get(key, 0) + 1
            self._shared_last_turn[key] = t

    def update(self, turn: int, events, our_active: Optional[str],
               opp_active: Optional[str]) -> None:
        from agents.battle.battle_event import OURS, OPP, EventKind
        self._turn = max(self._turn, int(turn))
        for e in events or []:
            seq = getattr(e, "seq", None)
            if seq is not None:
                if seq <= self._max_seq:
                    continue          # already folded (overlap / re-decide replay)
                self._max_seq = seq
            side = getattr(e, "side", None)
            sp = getattr(e, "actor_species", None)
            et = int(getattr(e, "turn", turn))
            k = e.kind
            if k is EventKind.MOVE and side is not None and sp:
                self._last[side] = {"move_id": e.move_id, "was_switch": False,
                                    "missed": False, "failed": False, "crit": False,
                                    "turn": et}
                if side == OPP and self._our_active:
                    key = (sp, self._our_active)
                    d = self._attacks if _move_is_damaging(e.move_id) else self._status_clicks
                    d[key] = d.get(key, 0) + 1
                self._observe_pairing(et)
            elif k in (EventKind.SWITCH, EventKind.DRAG) and side is not None and sp:
                if k is EventKind.SWITCH and side == OPP and self._our_active:
                    key = (sp, self._our_active)
                    self._switch_ins[key] = self._switch_ins.get(key, 0) + 1
                self._last[side] = {"move_id": None, "was_switch": True,
                                    "missed": False, "failed": False, "crit": False,
                                    "turn": et}
                if side == OURS:
                    self._our_active = sp
                else:
                    self._opp_active = sp
                self._observe_pairing(et)
            elif k in (EventKind.MISS, EventKind.FAIL, EventKind.CRIT) and side is not None:
                la = self._last.get(side)
                if la and not la["was_switch"] and la["turn"] == et:
                    if k is EventKind.MISS:
                        la["missed"] = True
                    elif k is EventKind.FAIL:
                        la["failed"] = True
                    else:
                        la["crit"] = True
            elif k is EventKind.FAINT and side is not None and sp:
                if side == OURS and self._our_active == sp:
                    self._our_active = None
                elif side == OPP and self._opp_active == sp:
                    self._opp_active = None
        # Decision-time resync: the LiveView actives are the same public fact the events
        # advance; they also cover pairing on quiet windows (recency's `seen` precedent).
        if our_active:
            self._our_active = our_active
        if opp_active:
            self._opp_active = opp_active
        self._observe_pairing(self._turn)

    def last_action(self, side: str) -> tuple:
        """→ ``(move_id | None, was_switch: float, outcome: str | None, crit: float)``."""
        la = self._last.get(side)
        if la is None:
            return (None, 0.0, None, 0.0)
        if la["was_switch"]:
            return (None, 1.0, None, 0.0)
        outcome = "miss" if la["missed"] else ("fail" if la["failed"] else "hit")
        return (la["move_id"], 0.0, outcome, 1.0 if la["crit"] else 0.0)

    def pair_values(self, opp_species: Optional[str], our_species: Optional[str]) -> tuple:
        """→ the 5-cell ``h[i, j]`` for (their mon i, our mon j), each in [0, 1]:
        ``(switch_ins, attacks, status_clicks, shared_field_turns, recency_of_last_pairing)``."""
        if not opp_species or not our_species:
            return (0.0, 0.0, 0.0, 0.0, 1.0)
        key = (opp_species, our_species)
        last = self._shared_last_turn.get(key)
        rec = 1.0 if last is None else _pair_sat_norm(max(0, self._turn - last))
        return (
            _pair_sat_norm(self._switch_ins.get(key, 0)),
            _pair_sat_norm(self._attacks.get(key, 0)),
            _pair_sat_norm(self._status_clicks.get(key, 0)),
            _pair_sat_norm(self._shared_count.get(key, 0)),
            rec,
        )


# Tier H-B event-type vocabulary — SINGLE-SOURCED in `agents.observation.constants` (it is
# the obs contract: column 0 of every event row). Re-imported here for the fold that emits it.
from agents.observation.constants import (          # noqa: E402
    EVENT_T_PAD, EVENT_T_MOVE, EVENT_T_SWITCH_IN, EVENT_T_FAINT, EVENT_T_STATUS_APPLIED,
    EVENT_T_STATUS_CURED, EVENT_T_BOOST, EVENT_T_ITEM_REVEAL, EVENT_T_HAZARD,
    EVENT_T_SWITCH_REJECTED, N_EVENT_TYPES,
)

# Status-id axis for event records (0 = none/pad). Mirrors the per-mon condition one-hot's
# vocabulary; kept as an ID here (the consumer embeds) rather than a one-hot (obs stays lean).
_EVENT_STATUS_IDS = {"brn": 1, "par": 2, "slp": 3, "frz": 4, "psn": 5, "tox": 6}


class EventWindowTracker:
    """Tier H-B (`designs/ai_v9/design_history_entity.md` §3 H-B): the last-N DECISION-RELEVANT
    events as typed records — the sequential residue the compiled tiers (recency, H-A) cannot
    carry, made queryable. PUBLIC protocol events only; within-battle only; **seq-idempotent**
    (the PairHistoryTracker convention), so an overlapping window or a rolled-back opponent
    RE-DECIDE can never append twice.

    One record per event in the H-B vocabulary (moves, switch-ins, faints, status
    applied/cured, boosts, item reveals, hazards, our rejected switches). MODIFIER events
    (DAMAGE / MISS / FAIL / CRIT / IMMUNE / RESISTED / SUPEREFFECTIVE) do not get records —
    they ATTACH to their side's open same-turn MOVE record (the H-A attach rule, extended):
    damage lands on the target side and accumulates into the move's ``hp_delta``; the
    effectiveness trio sets ``eff``; the outcome trio sets flags. ``we_first`` marks the
    records of whichever side MOVED first that turn (speed-inversion evidence, §2).
    ``forced_window`` tags events emitted while a side's active slot was empty after a faint
    (the "turn framing dissolves" phase tag). v1 TRIMS, recorded deliberately: no faint-cause
    multi-hot (adjacent events + compiled state carry it), no item/hazard CONTENT ids (the
    reveal event + per-mon state carry them), SETBOOST/CLEARBOOST skipped (rare, compiled
    boosts are current-state).

    Records are plain dicts (species/move ids as STRINGS — the encoder maps to nums, exactly
    like H-A's last-action). The window is bounded (``maxlen``); reads return
    most-recent-LAST so the encoder's padding convention is stable."""

    def __init__(self, maxlen: int = 32):
        self.maxlen = int(maxlen)
        self._events: deque = deque(maxlen=self.maxlen)
        self._max_seq: int = -1
        self._turn: int = 0
        self._our_active: Optional[str] = None
        self._opp_active: Optional[str] = None
        self._open_move: dict = {}          # side -> the record modifiers attach to
        self._first_mover_turn: int = -1    # the turn whose first mover is recorded
        self._first_mover_side: Optional[str] = None
        self._forced: dict = {}             # side -> active slot empty (post-faint window)

    def _append(self, rec: dict) -> dict:
        rec["forced_window"] = 1.0 if (self._forced.get("our") or self._forced.get("opp")) else 0.0
        self._events.append(rec)
        return rec

    def update(self, turn: int, events, our_active: Optional[str],
               opp_active: Optional[str]) -> None:
        from agents.battle.battle_event import OURS, OPP, EventKind
        self._turn = max(self._turn, int(turn))
        for e in events or []:
            seq = getattr(e, "seq", None)
            if seq is not None:
                if seq <= self._max_seq:
                    continue
                self._max_seq = seq
            side = getattr(e, "side", None)
            sp = getattr(e, "actor_species", None)
            et = int(getattr(e, "turn", turn))
            k = e.kind
            if k is EventKind.MOVE and side is not None and sp:
                if self._first_mover_turn != et:
                    self._first_mover_turn = et
                    self._first_mover_side = side
                rec = self._append({
                    "t": EVENT_T_MOVE, "actor": sp, "side": side,
                    "target": (self._opp_active if side == OURS else self._our_active),
                    "move_id": e.move_id, "hp_delta": 0.0,
                    "missed": False, "failed": False, "crit": False,
                    "eff": 0, "we_first": side == self._first_mover_side,
                    "status": 0, "turn": et,
                })
                self._open_move[side] = rec
            elif k is EventKind.DAMAGE and side is not None:
                # damage lands ON `side`; it attaches to the OTHER side's open move this turn
                # ONLY when it is the move's own hit: no `[from]` clause (recoil / Sandstorm /
                # status / item residuals all carry one) AND the damaged mon IS the move's
                # recorded target (a switched-in replacement taking hazard chip is not the hit).
                mover = OPP if side == OURS else OURS
                om = self._open_move.get(mover)
                if (om is not None and om["turn"] == et
                        and not e.value.get("from")
                        and sp and om["target"] == sp):
                    amt = e.amount
                    if amt is not None:
                        om["hp_delta"] += float(amt)
            elif k in (EventKind.MISS, EventKind.FAIL, EventKind.CRIT) and side is not None:
                om = self._open_move.get(side)
                if om is not None and om["turn"] == et:
                    if k is EventKind.MISS:
                        om["missed"] = True
                    elif k is EventKind.FAIL:
                        om["failed"] = True
                    else:
                        om["crit"] = True
            elif k in (EventKind.IMMUNE, EventKind.RESISTED, EventKind.SUPEREFFECTIVE) \
                    and side is not None:
                mover = OPP if side == OURS else OURS       # tagged on the DEFENDER
                om = self._open_move.get(mover)
                if om is not None and om["turn"] == et:
                    om["eff"] = {EventKind.SUPEREFFECTIVE: 1, EventKind.RESISTED: 2,
                                 EventKind.IMMUNE: 3}[k]
            elif k in (EventKind.SWITCH, EventKind.DRAG) and side is not None and sp:
                # append BEFORE clearing the forced flag: the arriving replacement IS the
                # forced-window event (the tag is what lets a reader see "this switch-in was
                # the post-faint replacement, not a chosen pivot").
                self._append({
                    "t": EVENT_T_SWITCH_IN, "actor": sp, "side": side,
                    "target": (self._opp_active if side == OURS else self._our_active),
                    "move_id": None, "hp_delta": 0.0, "missed": False, "failed": False,
                    "crit": False, "eff": 0, "we_first": False, "status": 0, "turn": et,
                })
                self._forced["our" if side == OURS else "opp"] = False
                if side == OURS:
                    self._our_active = sp
                else:
                    self._opp_active = sp
            elif k is EventKind.FAINT and side is not None and sp:
                self._append({
                    "t": EVENT_T_FAINT, "actor": sp, "side": side, "target": None,
                    "move_id": None, "hp_delta": 0.0, "missed": False, "failed": False,
                    "crit": False, "eff": 0, "we_first": False, "status": 0, "turn": et,
                })
                if side == OURS and self._our_active == sp:
                    self._our_active = None
                    self._forced["our"] = True
                elif side == OPP and self._opp_active == sp:
                    self._opp_active = None
                    self._forced["opp"] = True
            elif k in (EventKind.STATUS, EventKind.CURESTATUS) and sp:
                self._append({
                    "t": (EVENT_T_STATUS_APPLIED if k is EventKind.STATUS
                          else EVENT_T_STATUS_CURED),
                    "actor": sp, "side": side, "target": None, "move_id": None,
                    "hp_delta": 0.0, "missed": False, "failed": False, "crit": False,
                    "eff": 0, "we_first": False,
                    "status": _EVENT_STATUS_IDS.get(e.status or "", 0), "turn": et,
                })
            elif k in (EventKind.BOOST, EventKind.UNBOOST) and sp:
                amt = float(e.amount or 0.0)
                self._append({
                    "t": EVENT_T_BOOST, "actor": sp, "side": side, "target": None,
                    "move_id": None,
                    "hp_delta": (amt if k is EventKind.BOOST else -amt),   # the magnitude col
                    "missed": False, "failed": False, "crit": False, "eff": 0,
                    "we_first": False, "status": 0, "turn": et,
                })
            elif k in (EventKind.ITEM, EventKind.ENDITEM) and sp:
                self._append({
                    "t": EVENT_T_ITEM_REVEAL, "actor": sp, "side": side, "target": None,
                    "move_id": None, "hp_delta": 0.0, "missed": False, "failed": False,
                    "crit": False, "eff": 0, "we_first": False, "status": 0, "turn": et,
                })
            elif k is EventKind.SIDE and side is not None:
                self._append({
                    "t": EVENT_T_HAZARD, "actor": None, "side": side, "target": None,
                    "move_id": None, "hp_delta": 0.0, "missed": False, "failed": False,
                    "crit": False, "eff": 0, "we_first": False, "status": 0, "turn": et,
                })
            elif k is EventKind.CHOICE_REJECTED:
                self._append({
                    "t": EVENT_T_SWITCH_REJECTED, "actor": self._our_active, "side": OURS,
                    "target": None, "move_id": None, "hp_delta": 0.0, "missed": False,
                    "failed": False, "crit": False, "eff": 0, "we_first": False,
                    "status": 0, "turn": et,
                })
        # Decision-time resync (the H-A alive-filter is applied by the CALLER, which passes
        # None for a fainted "active" — same contract as PairHistoryTracker.update).
        if our_active:
            self._our_active = our_active
            self._forced["our"] = False
        if opp_active:
            self._opp_active = opp_active
            self._forced["opp"] = False

    @property
    def turn(self) -> int:
        return self._turn

    def window(self) -> list:
        """The folded records, oldest-first (≤ ``maxlen``)."""
        return list(self._events)


class EpisodeTracker:
    """
    Tracks per-episode state needed to build observations and reward signals.

    Owns slot registries, a rolling history of BattleContexts, and a parallel
    list of actions taken at each context so historical TurnDeltas can be
    reconstructed for the N-turn history observation feature.
    """

    def __init__(self, history_cap: Optional[int] = None):
        """``history_cap`` = the N passed to :meth:`prev_N_delta_vecs` (i.e.
        ``N_HISTORY_TURNS``). When set, the rolling history is bounded so a long game /
        250-turn stall can't accumulate hundreds of ``BattleContext``s (each carrying
        ``obs``+``mask``). ``prev_N_delta_vecs`` reaches ``_history[-2-i]`` (→ -(N+1)) and
        ``_actions``/``_cursors[-1-i]`` (→ -N), so we keep N+1 contexts and N aux entries;
        the ``len(actions)==len(cursors)==len(history)-1`` invariant is preserved by those
        maxlens. ``None`` ⇒ unbounded (short-episode callers: inference, tests)."""
        self._our_slots = SlotRegistry()
        self._opp_slots = SlotRegistry()
        _hist_max = (history_cap + 1) if history_cap else None
        _aux_max = history_cap if history_cap else None
        self._history: "deque[BattleContext]" = deque(maxlen=_hist_max)
        self._actions: "deque[int]" = deque(maxlen=_aux_max)   # _actions[i]=action FROM _history[i]
        self._cursors: "deque[int]" = deque(maxlen=_aux_max)   # _cursors[i]=event_cursor at _history[i]
        self._last_action: int = -1
        self._last_cursor: int = 0      # event_cursor at the last record() call
        self._hidden_power_tracker = HiddenPowerTracker()
        # Episode-scoped no-progress counter (design §5.1). Updated at record()/embed time so the
        # obs is fresh; read by BOTH the obs encoder (value()) and the reward (last_penalty).
        self._progress_clock = ProgressClock()
        # E9 step 1 (roadmap §3.9): per-entity recency, fed by the same decision window.
        self._recency = RecencyTracker()
        # Tier H-A (gen3_pair_history_v1): last-action + pair-history counters, same window.
        self._pair_history = PairHistoryTracker()
        self._event_window = EventWindowTracker()
        # Memoized turn-history: encoded TurnDelta vectors, oldest-left/newest-right.
        # A past turn's window is bounded and immutable (see prev_N_delta_vecs), so its
        # encoded vector never changes — we encode only the NEWEST delta each step and
        # reuse the rest, instead of re-folding+re-encoding all N slots every step.
        self._hist_vec_cache: "deque[np.ndarray]" = deque(maxlen=_aux_max)
        self._n_cached_deltas: int = 0   # total completed deltas the cache has encoded
        # MONOTONIC count of completed turn transitions this episode. NOT len(_history)-1,
        # which caps once the (bounded) history deque starts dropping — using the capped
        # length would make the cache miss new turns and serve stale ones.
        self._n_transitions: int = 0

    @property
    def hidden_power_tracker(self) -> HiddenPowerTracker:
        return self._hidden_power_tracker

    @property
    def progress_clock(self) -> ProgressClock:
        return self._progress_clock

    @property
    def recency(self) -> RecencyTracker:
        return self._recency

    @property
    def event_window(self) -> EventWindowTracker:
        return self._event_window

    @property
    def pair_history(self) -> PairHistoryTracker:
        return self._pair_history

    @property
    def last_ctx(self) -> Optional[BattleContext]:
        return self._history[-1] if self._history else None

    @property
    def prev_mask(self) -> np.ndarray:
        """Previous turn's action mask, as an obs feature.

        The MOVE bits are reordered from action/request order into sorted-by-id
        order so they line up with the active mon's move slots in the feature
        extractor (which reads moves via ``get_sorted_moves``). Without this the
        validity bit for one move lands on a different move's embedding — silent
        when all moves are legal, wrong on disabled/zero-PP turns. All-ones if no
        previous turn recorded yet.

        If our active mon CHANGED since the previous decision (a switch / forced
        replacement), the previous mask's MOVE bits describe the PREVIOUS mon's moves
        (sorted by that mon's ids), so they don't correspond to THIS mon's sorted move
        slots — the validity bit lands on an unrelated move. In that case the move bits
        are reset to the no-prior-info default (all-ones, like the first-turn case); the
        SWITCH bits stay (team-ordered, mon-independent). ``active_move_ids`` is a safe
        discriminator: it differs whenever the moveset differs, and is equal only when
        the sorted order already aligns. (gen3_move_slot_align_v1 — same alignment class.)"""
        if len(self._history) >= 2:
            prev_ctx = self._history[-2]
            reordered = reorder_move_bits_to_sorted(
                prev_ctx.mask.astype(np.float32), prev_ctx.active_move_ids
            )
            assert_sorted_validity_correct(
                reordered, prev_ctx.mask, prev_ctx.active_move_ids
            )
            cur_ctx = self._history[-1]
            if cur_ctx.active_move_ids != prev_ctx.active_move_ids:
                reordered[MOVE_START:MOVE_END] = 1.0
            return reordered
        return np.ones(11, dtype=np.float32)

    def record(self, battle, mask: np.ndarray, legal=None) -> BattleContext:
        """Build and store a context snapshot for the current turn.

        Also commits the pending _last_action and event_cursor as the action
        and window-start taken FROM the previous context, so prev_N_delta_vecs()
        can reconstruct all N deltas. Updates the HiddenPowerTracker BEFORE the
        env encodes the observation, so the encoded obs includes the just-fired
        HP's narrowing.

        ``legal`` is the per-decision :class:`LegalActions` snapshot the masker built
        the mask from; threaded onto the stored context so the action mapper decodes
        against the same snapshot the model saw.
        """
        if self._history:
            self._actions.append(self._last_action)
            self._cursors.append(self._last_cursor)
            self._n_transitions += 1   # one more completed transition becomes available
        # Capture cursor NOW (before we build the context snapshot) so it marks
        # the start of the window for the NEXT decision — events emitted between
        # this record() and the next one are the delta for this turn.
        self._last_cursor = getattr(battle, "event_cursor", 0)
        ctx = BattleContext.from_battle(battle, mask, self._our_slots, self._opp_slots, legal)

        # Current-board reads (the HP-candidate moveset scan + the HP-target type/ability
        # lookup) go through our LiveView, never the raw poke-env Battle/Pokemon objects
        # (ai_v4 Phase 3 — encapsulation behind the strict boundary).
        live = battle.strict_view().live
        self._maybe_observe_hidden_power(live, ctx)
        self._scan_opp_movesets_for_no_hp(live)

        self._history.append(ctx)
        return ctx

    # ------------------------------------------------------------------ #
    # Rolling-history snapshot / restore — for the opponent's stale RE-DECIDE
    # ------------------------------------------------------------------ #
    def snapshot(self) -> tuple:
        """Cheap shallow snapshot of the rolling-history state, taken before a self-play
        opponent decides so a stale RE-DECIDE can roll the superseded attempt back out
        (``RLPlayer.choose_move``). Without it, each re-decide would leave a phantom turn —
        the record() of the decision that never happened — in the opponent's turn-history obs.

        The bounded deques are short (≈``N_HISTORY_TURNS``) and ``BattleContext``s are immutable,
        so snapshotting the deques as plain lists is cheap and *shares* (never clones) the
        contexts. The ``HiddenPowerTracker`` and slot registries are deliberately NOT snapshotted:
        their updates are of real protocol events and idempotent (probability narrowing /
        stable slot ids), so a rolled-back attempt leaves them correct — re-observing the same
        events on the retry is a no-op. The trainee never re-decides (a stale trainee decision
        crashes), so only the opponent path uses this."""
        return (
            list(self._history),
            list(self._actions),
            list(self._cursors),
            list(self._hist_vec_cache),
            self._n_transitions,
            self._last_cursor,
            self._last_action,
        )

    def restore(self, snap: tuple) -> None:
        """Restore the state captured by :meth:`snapshot`, undoing the ``record()`` (and any
        memoized delta) that a stale, re-decided decision applied — so the committed
        turn-history never keeps a phantom turn. Rebuilding each deque from its snapshot list
        re-establishes the *exact* pre-attempt contents, including an entry ``maxlen`` would
        have since dropped, so the rollback is exact even when a deque sat at its cap."""
        history, actions, cursors, hist_vec, n_transitions, last_cursor, last_action = snap
        self._history.clear()
        self._history.extend(history)
        self._actions.clear()
        self._actions.extend(actions)
        self._cursors.clear()
        self._cursors.extend(cursors)
        self._hist_vec_cache.clear()
        self._hist_vec_cache.extend(hist_vec)
        self._n_transitions = n_transitions
        self._last_cursor = last_cursor
        self._last_action = last_action

    def _scan_opp_movesets_for_no_hp(self, live: "LiveView") -> None:
        """Mark any opponent species whose four moves are fully revealed and
        none is Hidden Power as definitively HP-less.

        This converts the previously ambiguous "all-zero HP probs" state into a
        positive signal (hp_revealed=1, probs all zero in the encoder).
        Idempotent and cheap (~12 lookups per turn).

        Reads the opponent's revealed movesets off the current-board ``LiveView``
        (``live.opp.mons`` / ``LivePokemon.move_ids``) rather than the raw
        ``battle.opponent_team`` — value-identical, since ``move_ids`` is exactly the
        revealed move-dict keys the old path iterated.
        """
        for mon in live.opp.mons:
            if not mon.species:
                continue
            move_ids = mon.move_ids
            if len(move_ids) >= 4 and not any(
                mid.startswith("hiddenpower") for mid in move_ids
            ):
                self._hidden_power_tracker.mark_no_hp(mon.species)

    def _maybe_observe_hidden_power(self, live: "LiveView", ctx: BattleContext) -> None:
        """Feed an HP observation to the tracker when opp's last damaging move
        was Hidden Power.

        Everything we need — firer species, target species, target status at
        move-fire time, effectiveness — comes from ctx.opp_last_damaging_event,
        which is set by poke-env's protocol parser at the |move| line and
        finalized by the matching |-supereffective|/|-resisted|/|-immune|
        event. The event is turn-gated to the just-ended turn, so a stale HP
        from earlier in the battle never leaks. The target's current type/ability
        (for the effectiveness filter) is resolved through the current-board
        ``live`` view. No before/after inference, no resolver, no edge cases for
        switches / Roar / phazing / faint chains — the protocol stated the facts
        and we just record them.
        """
        if ctx.phase != "move_selection":
            return
        event = ctx.opp_last_damaging_event
        if event is None or event.move_id != "hiddenpower":
            return
        target = _wrap_hp_target(live, event)
        if target is None:
            return
        # FEASIBILITY GUARD (default since gen3_typed_hp_belief_v1). Two different things can make an
        # observation eliminate every candidate, and they deserve opposite responses:
        #
        #   * NO Hidden Power type could produce this effectiveness against this target. Then the
        #     TARGET IDENTIFICATION is wrong (the classic case: a switch resolved on our side in the
        #     same window, so the mon we resolved isn't the mon that was hit) — the observation is
        #     junk about a mon that was never involved. DISCARD it: narrowing on it would zero
        #     perfectly possible types, and raising would crash a run over a misattribution.
        #   * Some type could have produced it, but none of THIS SPECIES' surviving candidates can.
        #     That is a real contradiction — a tracker bug or a gap in the HP-type priors — and
        #     `observe` still RAISES on it, with its full per-species observation-log dump.
        #
        # So the guard removes the crash for the misattribution class WITHOUT weakening the GIGO
        # detector for the genuine one. Discards are counted rather than silent (see the tracker's
        # `infeasible_observations`) — a rising count means the target resolution is drifting.
        if not self._hidden_power_tracker.is_feasible(event.effectiveness, target):
            self._hidden_power_tracker.note_infeasible()
            return
        self._hidden_power_tracker.observe(
            event.user_species, event.effectiveness, target
        )

    def advance(self, action: int) -> None:
        """Record the action chosen this turn, before the game steps forward."""
        self._last_action = action

    def _get_events_for_window(self, battle, cursor: int) -> list:
        """Return events from ``cursor`` to now using battle.events_since, or []."""
        if battle is None or not hasattr(battle, "events_since"):
            return []
        return battle.events_since(cursor)

    def build_delta(self, battle=None) -> TurnDelta:
        """Fold the most-recent turn's TurnDelta from the event log. Returns an empty delta
        at episode start.

        ``battle`` (a Gen3Battle) supplies the event window via ``events_since``; the single
        event-fold path (``TurnDelta.build_from_events``) is always used. A standalone caller
        without an event log passes ``battle=None`` ⇒ an empty window, which folds the
        current-board snapshot for HP (see ``build_from_events``).
        """
        if len(self._history) < 2:
            return TurnDelta.empty()
        prev_ctx = self._history[-2]
        curr_ctx = self._history[-1]
        cursor = self._cursors[-1] if self._cursors else 0
        events = self._get_events_for_window(battle, cursor)
        return TurnDelta.build_from_events(prev_ctx, curr_ctx, self._last_action, events)

    def update_progress_clock(self, battle, legal) -> TurnDelta:
        """Fold the just-completed window's delta and advance the shared ``ProgressClock``, so the obs
        scalar (``value()``) and the reward's no-progress penalty (``last_penalty``) key on ONE value.

        Call from ``embed_battle`` AFTER :meth:`record`, BEFORE ``encode`` — poke-env runs
        ``embed_battle`` before ``calc_reward``, so updating here (not at reward time) keeps the obs
        fresh. Returns the folded delta so the caller can reuse it (the env caches it for
        ``calc_reward``, avoiding a second fold). The penalty magnitude lives on the clock itself
        (set once from the reward config), so this stays an obs-side call with no reward param.
        Single home for the 3-step protocol the env + inference players both need (no copy-paste).
        """
        delta = self.build_delta(battle=battle)
        live = battle.strict_view().live
        self._progress_clock.update(delta, live, legal)
        # E9 recency: the SAME per-decision window the newest TurnDelta slot folds
        # ([cursors[-1], now)), plus the live actives for the seen reset.
        if isinstance(live.turn, int):        # a mocked/partial battle (tests) skips recency
            if self._cursors and hasattr(battle, "events_since"):
                _ev = battle.events_since(self._cursors[-1])
            else:
                _ev = []
            self._recency.update(
                live.turn, _ev,
                live.ours.active.species if live.ours.active else None,
                live.opp.active.species if live.opp.active else None)
            # Tier H-A: same window, ALIVE live actives only (seq-idempotent, so the shared
            # window plumbing needs no extra bookkeeping here). The alive filter is load-
            # bearing: at a FORCED-SWITCH decision poke-env still reports the fainted mon as
            # active, and an unfiltered resync would RESURRECT an active the FAINT event
            # correctly cleared — pairing the fresh replacement (or their next switch-in)
            # against a dead mon. Caught by pair_history_fuzz_test on a double-KO Explosion.
            _oa = live.ours.active
            _pa = live.opp.active
            self._pair_history.update(
                live.turn, _ev,
                _oa.species if (_oa is not None and not _oa.fainted) else None,
                _pa.species if (_pa is not None and not _pa.fainted) else None)
            # Tier H-B: the SAME window and the SAME alive-filtered resync contract.
            self._event_window.update(
                live.turn, _ev,
                _oa.species if (_oa is not None and not _oa.fainted) else None,
                _pa.species if (_pa is not None and not _pa.fainted) else None)
        return delta

    def _encode_delta_slot(self, i: int, encoder, battle) -> np.ndarray:
        """Encode the TurnDelta for history slot ``i`` (0 = most-recent), folded over its
        OWN bounded decision window: ``[cursors[-1-i] : cursors[-i])`` (``end=None`` ⇒ to
        "now" for the most-recent slot). The upper bound is what makes the slot represent
        its own turn and the per-step cost bounded; the most-recent slot's ``end=None``
        resolves to the same cursor that bounds it on the NEXT step, so a cached vector is
        bit-identical to recomputing it later — which is why the deque memoization is safe.

        Always folds via ``build_from_events``. ``battle=None`` (or a battle without an event
        log) ⇒ an empty window per slot — the standalone/test path.
        """
        action = self._actions[-1 - i]
        ctx_prev = self._history[-2 - i]
        ctx_curr = self._history[-1 - i]
        if battle is not None and hasattr(battle, "events_between") and self._cursors:
            start = self._cursors[-1 - i]
            end = None if i == 0 else self._cursors[-i]
            events = battle.events_between(start, end)
        else:
            events = []
        delta = TurnDelta.build_from_events(ctx_prev, ctx_curr, action, events)
        return encoder.encode(delta)

    def prev_N_delta_vecs(
        self, n: int, encoder: "TurnDeltaEncoder", battle=None
    ) -> np.ndarray:
        """Return (n, TURN_DELTA_DIM) array of encoded TurnDeltas, oldest-first.

        Index n-1 is the most recent delta (same data as build_delta()). Turns not yet
        played are zero-padded. Pass ``battle`` (Gen3Battle) to use the event-fold path.

        Each past turn's bounded window is immutable, so on the event path we **encode only
        the newly-completed delta(s) and reuse the cached rest** (deque memoization) — one
        fold+encode per step instead of N. The output is identical to recomputing every
        slot from scratch.
        """
        result = np.zeros((n, encoder.dimension), dtype=np.float32)
        use_events = battle is not None and hasattr(battle, "events_between")
        available = min(n, len(self._history) - 1, len(self._actions))
        if use_events:
            available = min(available, len(self._cursors))
        if available <= 0:
            return result

        if not use_events:
            # Standalone / no-event-log path: recompute every slot, uncached (each folds
            # over an empty window — the current-board snapshot for HP).
            for i in range(available):
                result[n - 1 - i] = self._encode_delta_slot(i, encoder, battle)
            return result

        # Event path: extend the deque by the newly-completed delta(s) only. Use the
        # MONOTONIC transition count (not len(_history)-1, which caps with the deque).
        total_completed = self._n_transitions
        new = total_completed - self._n_cached_deltas
        if new == 1:
            # Common case: one turn finished since last call → encode just slot 0.
            self._hist_vec_cache.append(
                self._encode_delta_slot(0, encoder, battle)
            )
        elif new != 0:
            # Rare (calls skipped, or first call mid-episode): rebuild the kept tail.
            self._hist_vec_cache.clear()
            for i in range(available - 1, -1, -1):  # oldest-kept first → newest appended last
                self._hist_vec_cache.append(
                    self._encode_delta_slot(i, encoder, battle)
                )
        self._n_cached_deltas = total_completed

        for k, vec in enumerate(reversed(self._hist_vec_cache)):  # newest first
            if k >= n:
                break
            result[n - 1 - k] = vec
        return result

    def reset(self) -> None:
        self._our_slots.reset()
        self._opp_slots.reset()
        self._history.clear()
        self._actions.clear()
        self._cursors.clear()
        self._hist_vec_cache.clear()
        self._n_cached_deltas = 0
        self._n_transitions = 0
        self._last_action = -1
        self._last_cursor = 0
        self._hidden_power_tracker.reset()
        self._progress_clock.reset()
        # Cross-turn trackers are WITHIN-BATTLE only (design_history_entity.md §0.3). The
        # recency tracker was missing here (a cross-episode leak on the env path, where one
        # EpisodeTracker is reset per episode rather than recreated — the recency fuzz built
        # fresh trackers per battle so it never saw it); fixed alongside pair-history.
        self._recency = RecencyTracker()
        self._pair_history = PairHistoryTracker()
        self._event_window = EventWindowTracker()
