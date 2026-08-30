from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from poke_env.battle.abstract_battle import DamagingMoveEvent

from agents.enums import PokemonType

from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta
from agents.training.hidden_power_tracker import HiddenPowerTracker
from agents.training.progress_clock import ProgressClock
from agents.training.slot_registry import SlotRegistry
from agents.observation.assembler import ObsAssembler, SAT_LUT as _SAT_LUT

_SAT_CAP = len(_SAT_LUT) - 1

if TYPE_CHECKING:
    from agents.enums import Status

    from agents.battle.live_view import LiveView


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
        """→ (seen, acted, was_hit), each log-saturated to [0, 1].

        The saturation has an 11-value codomain, so it reads out of `assembler.SAT_LUT`
        (precomputed from the identical float64 expression — bit-for-bit the same numbers, and
        it is on the hot path 12× per encode)."""
        turn = self._turn
        sat = self._SAT
        lut = _SAT_LUT
        out = []
        for d in (self._seen, self._acted, self._hit):
            last = None if species is None else d.get((side, species))
            n = sat if last is None else max(0, turn - last)
            out.append(lut[n if n < sat else sat])
        return tuple(out)


def _pair_sat_norm(n: int, sat: int = 10) -> float:
    """The H-A log-saturation convention (design_history_entity.md §3):
    ``log(1 + min(n, 10)) / log(11)`` — same curve the recency triplet and
    ``turns_since_progress`` use.

    The default ``sat`` reads out of `assembler.SAT_LUT` (same float64 expression, evaluated
    once): this is called 5× per pair cell × 36 cells per encode, the single largest call-count
    family the obs benchmark reports."""
    if sat == _SAT_CAP:
        i = int(n)
        return _SAT_LUT[i if 0 < i < _SAT_CAP else (0 if i <= 0 else _SAT_CAP)]
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
        # The saturation is INLINED (not `_pair_sat_norm`) on purpose: this runs 36× per encode
        # and every cell ticks every turn, so it is the largest remaining call family on the
        # warm obs path. Identical numbers — `_SAT_LUT` is the same float64 table, and
        # `assembler_test` pins the two against the arithmetic they replaced.
        lut = _SAT_LUT
        cap = _SAT_CAP
        last = self._shared_last_turn.get(key)
        if last is None:
            rec = 1.0
        else:
            _d = self._turn - last
            rec = lut[_d if 0 < _d < cap else (0 if _d <= 0 else cap)]
        out = []
        for d in (self._switch_ins, self._attacks, self._status_clicks, self._shared_count):
            n = d.get(key, 0)
            out.append(lut[n if n < cap else cap])
        out.append(rec)
        return tuple(out)


# Tier H-B event-type vocabulary — SINGLE-SOURCED in `agents.observation.constants` (it is
# the obs contract: column 0 of every event row). Re-imported here for the fold that emits it.
from agents.observation.constants import (          # noqa: E402
    EVENT_T_MOVE, EVENT_T_SWITCH_IN, EVENT_T_FAINT, EVENT_T_STATUS_APPLIED,
    EVENT_T_STATUS_CURED, EVENT_T_BOOST, EVENT_T_ITEM_REVEAL, EVENT_T_HAZARD,
    EVENT_T_SWITCH_REJECTED, EVENT_T_CANT,
    ITEM_TR_REVEALED, ITEM_TR_CONSUMED, ITEM_TR_REMOVED, ITEM_TR_SWAPPED,
)

# Status-id axis for event records (0 = none/pad). Mirrors the per-mon condition one-hot's
# vocabulary; kept as an ID here (the consumer embeds) rather than a one-hot (obs stays lean).
def _event_status_id(status: "str | None") -> int:
    """The H-B column-15 id for a status name — CRASH, don't drop (the `normalize_cant_reason`
    contract). An absent status is legitimately 0; an unrecognised NAME is a parser or
    vocabulary drift, and silently coding it as 0 would tell the model "no status" on a turn
    the opponent was, say, badly poisoned — an unfalsifiable GIGO with no metric to show it.
    Gen-3's status set is closed, so an unrecognised WORD is always a defect somewhere upstream.

    **A `[...]` token is ABSENCE, not a bad name, and the distinction is the whole subtlety.**
    `_build_event` reads the status positionally (`sm[3]`), and some real Showdown lines put a
    protocol modifier there instead — measured live: `|-curestatus|` from Heal Bell /
    Aromatherapy yields `'[from] move: Aromatherapy'`. That line genuinely carries no status
    name, so 0 is the correct id (and is what shipped); what must never pass silently is a
    plain word the vocabulary does not know.
    """
    from agents.observation.constants import EVENT_STATUS_IDS

    if not status:
        return 0
    key = str(status).strip().lower()
    if key.startswith("["):        # a protocol modifier in the status slot ⇒ no status on the line
        return 0
    if key not in EVENT_STATUS_IDS:
        raise ValueError(
            f"unknown status {status!r} for the H-B event window — the vocabulary is "
            f"{sorted(EVENT_STATUS_IDS)} (agents.observation.constants.EVENT_STATUS_IDS). "
            "Add it there (and widen EventSeats' table) rather than letting it read as 'none'.")
    return EVENT_STATUS_IDS[key]


# gen3_event_semantics_v1 — the two classifiers the H-B rows need.
# `_classify_faint_cause` is IMPORTED from turn_view rather than reimplemented: the frames and
# the event window must never disagree about what "weather" means, and one copy cannot drift.
from agents.battle.turn_view import _classify_faint_cause

_SELF_KO_MOVE_IDS: frozenset = frozenset({"explosion", "selfdestruct"})
def _classify_item_transition(kind, from_clause: "Optional[str]") -> int:
    """|-item| / |-enditem| (+ its `[from]`) -> one of the ITEM_TR_* ids.

    Gen3 has THREE ways an item stops being held and they mean different things to a player:
    a CONSUMED berry is spent (the mon is now itemless by its own design), a Knock Off REMOVAL
    is permanent in ADV (unlike later gens), and a Trick/Thief/Covet SWAP means the OPPONENT is
    now holding it — which is information about their set, not just about ours. Collapsing them
    into one "item gone" bit is the conflation this column exists to end."""
    from agents.battle.battle_event import EventKind
    if kind is EventKind.ITEM:
        return ITEM_TR_REVEALED
    fc = (from_clause or "").strip().lower()
    if "knock off" in fc or "knockoff" in fc:
        return ITEM_TR_REMOVED
    if any(w in fc for w in ("trick", "thief", "covet", "switcheroo")):
        return ITEM_TR_SWAPPED
    return ITEM_TR_CONSUMED


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
        self._open_move: dict = {}
        # gen3_event_semantics_v1: the FAINT row's cause needs to know what last damaged
        # the mon — the protocol says it on the DAMAGE line's `[from]`, not on the faint.
        # Per SIDE, reset when that side's mon leaves the field (a fresh mon's death has
        # nothing to do with the previous occupant's last chip).
        self._last_dmg_cause: dict = {}
        self._used_selfko: dict = {}   # side -> did its last move self-KO
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
                # Explosion / Self-Destruct kill their OWN user; the lethal damage carries no
                # `[from]`, so without this the faint would classify as a plain `attack`.
                self._used_selfko[side] = (e.move_id in _SELF_KO_MOVE_IDS)
            elif k is EventKind.DAMAGE and side is not None:
                # damage lands ON `side`; it attaches to the OTHER side's open move this turn
                # ONLY when it is the move's own hit: no `[from]` clause (recoil / Sandstorm /
                # status / item residuals all carry one) AND the damaged mon IS the move's
                # recorded target (a switched-in replacement taking hazard chip is not the hit).
                self._last_dmg_cause[side] = e.from_clause   # None ⇒ a direct hit
                mover = OPP if side == OURS else OURS
                om = self._open_move.get(mover)
                # `from_clause`, NOT `value.get("from")`. On a DAMAGE event the parser stores
                # the `[from]` clause under `value["reason"]`, so the raw key is ALWAYS absent
                # here and this guard never fired: every sandstorm/burn/poison/Leech-Seed/recoil
                # residual landing on the move's target was folded into the move's attributed
                # magnitude. Shipped v81, trained on for two generations.
                if (om is not None and om["turn"] == et
                        and not e.from_clause
                        and sp and om["target"] == sp):
                    amt = e.amount
                    if amt is not None:
                        om["hp_delta"] += float(amt)
            elif k in (EventKind.MISS, EventKind.FAIL, EventKind.CRIT) and side is not None:
                om = self._open_move.get(side)
                # A `-fail` carrying a real `[from]` cause is NOT the open move's outcome — it is
                # some other effect fizzling while this side happened to be the current mover
                # (`|-fail|p2a: Metagross|unboost|[from] ability: Clear Body` = Intimidate blocked
                # on a switch-in, observed mis-attributed onto a full-damage Earthquake). The
                # synthetic "move-suffix" tag (`|move|…|[miss]`) IS the move's own outcome and
                # still counts. Same `[from]` class as the v91 DAMAGE guard.
                external_cause = (k is EventKind.FAIL
                                  and e.from_clause not in (None, "move-suffix"))
                if om is not None and om["turn"] == et and not external_cause:
                    if k is EventKind.MISS:
                        om["missed"] = True
                    elif k is EventKind.FAIL:
                        om["failed"] = True
                    else:
                        om["crit"] = True
            elif k in (EventKind.IMMUNE, EventKind.RESISTED, EventKind.SUPEREFFECTIVE) \
                    and side is not None:
                # The producer tags these on the MOVER ("attach to the resolving mover",
                # gen3_battle), the same convention as CRIT/MISS/FAIL one branch up. This
                # consumer flipped it to the defender until 2026-08-19, so on every one-sided
                # turn (the immune-on-pivot case above all) the lookup hit the side with no
                # open move and the eff dropped — measured: EVERY move row in a gen-15 trace
                # read neutral, including an immune whiff and a 4× KO. The fuzz oracle and the
                # unit fixture shared the flip, which is why 73k checks stayed green.
                om = self._open_move.get(side)
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
                # A fresh mon inherits no chip history from the one it replaced — without this
                # the incoming mon's first faint would read the PREVIOUS occupant's last cause.
                self._last_dmg_cause.pop(side, None)
                self._used_selfko.pop(side, None)
            elif k is EventKind.FAINT and side is not None and sp:
                self._append({
                    "t": EVENT_T_FAINT, "actor": sp, "side": side, "target": None,
                    "move_id": None, "hp_delta": 0.0, "missed": False, "failed": False,
                    "crit": False, "eff": 0, "we_first": False, "status": 0, "turn": et,
                    # gen3_event_semantics_v1: WHY it fainted. Reuses `turn_view`'s classifier
                    # so the event window and the TurnDelta fold cannot drift on the vocabulary.
                    "faint_cause": _classify_faint_cause(
                        self._last_dmg_cause.get(side), bool(self._used_selfko.get(side))),
                })
                self._last_dmg_cause.pop(side, None)
                self._used_selfko.pop(side, None)
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
                    "status": _event_status_id(e.status), "turn": et,
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
                    "item_tr": _classify_item_transition(k, e.from_clause),
                })
            elif k is EventKind.SIDE and side is not None:
                self._append({
                    "t": EVENT_T_HAZARD, "actor": None, "side": side, "target": None,
                    "move_id": None, "hp_delta": 0.0, "missed": False, "failed": False,
                    "crit": False, "eff": 0, "we_first": False, "status": 0, "turn": et,
                })
            elif k is EventKind.CANT and sp:
                # gen3_frame_deletion_v1: "this mon could not move, and why" (full paralysis /
                # sleep / flinch / recharge). The battle event log always carried it and the
                # TurnDelta fold read it into `*_cant_reason`, but it reached the model ONLY
                # through the lag frames — this window emitted no row. With those frames
                # deleted this is the fact's only route, so it gets its own type + column.
                # gen3_damp_cant_v1: attribute the row to the mon that LOST ITS TURN, not to
                # the ability holder the protocol files it against. Falls back to the event's own
                # actor/side for ordinary self-inflicted cants, where they are the same mon.
                _cs = e.blocked_side or side
                _ca = e.blocked_actor or sp
                self._append({
                    "t": EVENT_T_CANT, "actor": _ca, "side": _cs, "target": None,
                    "move_id": e.cant_move, "hp_delta": 0.0, "missed": False,
                    "failed": True, "crit": False, "eff": 0, "we_first": False,
                    "status": 0, "cant": e.reason, "turn": et,
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

    def open_records(self) -> tuple:
        """The ≤2 records that may still be MUTATED IN PLACE (one open MOVE per side).

        Every in-place write in :meth:`update` — accumulated ``hp_delta``, the
        ``missed``/``failed``/``crit`` outcome trio, the ``eff`` code — goes through
        ``self._open_move``, so this tuple IS the mutation surface. The incremental obs
        assembler re-writes exactly these rows each decision instead of tracking a per-record
        version, which makes "a row changed after it was appended" unrepresentable rather than
        merely handled. A new mutation site that did not route through ``_open_move`` would
        break that; today there is no other way to reach an already-appended record.
        """
        return tuple(self._open_move.values())


class EpisodeTracker:
    """
    Tracks per-episode state needed to build observations and reward signals.

    Owns slot registries, a rolling history of BattleContexts, and a parallel
    list of actions taken at each context so historical TurnDeltas can be
    reconstructed for the N-turn history observation feature.
    """

    def __init__(self, history_cap: Optional[int] = None):
        """``history_cap`` bounds the rolling context window so a long game / 250-turn
        stall can't accumulate hundreds of ``BattleContext``s (each carrying ``obs``+``mask``).

        gen3_frame_deletion_v1 SHRANK what needs keeping. The deep reach was
        ``prev_N_delta_vecs``, which walked back N slots to build the obs lag frames; with
        those frames deleted the deepest read left is ``build_delta``'s ``_history[-2]``
        (plus ``_cursors[-1]``), so a cap of 1 already satisfies every consumer. The
        parameter stays because the ``_hist_max = cap + 1`` / ``_aux_max = cap`` sizing is
        what preserves the ``len(actions)==len(cursors)==len(history)-1`` invariant, and
        because a future consumer wanting depth should ASK for it rather than inherit a
        window sized for a deleted feature. ``None`` ⇒ unbounded (inference, tests)."""
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
        # gen3_obs_assembler_v1: the incremental obs cache. It lives HERE — inside the object
        # the materializer deep-copies per counterfactual arm — so a restored arm carries a
        # cache consistent with its own rolled-back state (design §5.3). Built lazily because
        # its width is the encoder's, which this module does not import.
        self._obs_assembler: Optional[ObsAssembler] = None
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

    def obs_assembler(self, dimension: int) -> ObsAssembler:
        """The episode's :class:`~agents.observation.assembler.ObsAssembler`, sized to the
        encoder's ``dimension`` on first use. A width change (only possible across encoders)
        rebuilds it rather than reusing a mis-sized buffer."""
        asm = self._obs_assembler
        if asm is None or asm.buf.shape[0] != dimension:
            asm = self._obs_assembler = ObsAssembler(dimension)
        return asm

    @property
    def last_ctx(self) -> Optional[BattleContext]:
        return self._history[-1] if self._history else None

    def record(self, battle, mask: np.ndarray, legal=None) -> BattleContext:
        """Build and store a context snapshot for the current turn.

        Also commits the pending _last_action and event_cursor as the action
        and window-start taken FROM the previous context, so ``build_delta`` can fold the
        completed turn. Updates the HiddenPowerTracker BEFORE the
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
            self._n_transitions,
            self._last_cursor,
            self._last_action,
        )

    def restore(self, snap: tuple) -> None:
        """Restore the state captured by :meth:`snapshot`, undoing the ``record()`` that a
        stale, re-decided decision applied — so the committed
        turn-history never keeps a phantom turn. Rebuilding each deque from its snapshot list
        re-establishes the *exact* pre-attempt contents, including an entry ``maxlen`` would
        have since dropped, so the rollback is exact even when a deque sat at its cap."""
        history, actions, cursors, n_transitions, last_cursor, last_action = snap
        self._history.clear()
        self._history.extend(history)
        self._actions.clear()
        self._actions.extend(actions)
        self._cursors.clear()
        self._cursors.extend(cursors)
        self._n_transitions = n_transitions
        self._last_cursor = last_cursor
        self._last_action = last_action
        # gen3_obs_assembler_v1: the obs cache is NOT idempotent under rollback the way the HP
        # tracker and the slot registries are — a superseded decision's ring appends and dirty
        # bits would survive it. Marking everything dirty costs one full rebuild on the next
        # encode (re-decides are rare) and removes the need to reason about which half of the
        # cache the rollback invalidated.
        if self._obs_assembler is not None:
            self._obs_assembler.mark_all_dirty()

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
        # `legal` is the legality of the request the caller is about to answer — decision t+1 for
        # the window being folded. The clock's trapped-vs-wall gate was specified against "a switch
        # being legal THIS decision", so hand it the OPENING decision's snapshot too; which of the
        # two it reads is `--progress-decision-tense`'s business (ProgressClock._gates). The
        # context that opened the window is the same `_history[-2]` `build_delta` just folded from.
        legal_prev = self._history[-2].legal if len(self._history) >= 2 else None
        self._progress_clock.update(delta, live, legal, legal_prev=legal_prev)
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

    def reset(self) -> None:
        self._our_slots.reset()
        self._opp_slots.reset()
        self._history.clear()
        self._actions.clear()
        self._cursors.clear()
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
        if self._obs_assembler is not None:
            self._obs_assembler.reset()
