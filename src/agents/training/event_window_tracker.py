"""Tier H-B — the EVENT-WINDOW fold (`EventWindowTracker`): the last-N decision-relevant events
as typed records, folded from PUBLIC protocol events.

Split out of `episode_tracker.py` by `gen3_event_record_v2` (E12, the event-block reshape), which
widened each record to carry the Rust core's native per-action record faithfully: switch-in entry
reasons, DENIED rows (fainted first / the gen-3 turn cut), callers, Pursuit on a switch, boost
stats, item-transfer direction, Spikes layers and the refused-switch target (E4). The Rust mirror
is `src/rust_sim/src/trackers/history.rs::EventWindow`, held equal per decision by slice T and,
through the encoded row, slice O. `episode_tracker` re-exports every public name.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

# Tier H-B event-type vocabulary — SINGLE-SOURCED in `agents.observation.constants` (it is
# the obs contract: column 0 of every event row). Re-imported here for the fold that emits it.
from agents.observation.constants import (          # noqa: E402
    EVENT_T_MOVE, EVENT_T_SWITCH_IN, EVENT_T_FAINT, EVENT_T_STATUS_APPLIED,
    EVENT_T_STATUS_CURED, EVENT_T_BOOST, EVENT_T_ITEM_REVEAL, EVENT_T_HAZARD,
    EVENT_T_SWITCH_REJECTED, EVENT_T_CANT, EVENT_T_DENIED,
    ITEM_TR_REVEALED, ITEM_TR_CONSUMED, ITEM_TR_REMOVED, ITEM_TR_SWAPPED, ITEM_TR_RECEIVED,
    ENTRY_CHOSEN, ENTRY_REPLACEMENT, ENTRY_DRAG, ENTRY_BATON_PASS,
    DENIAL_FAINTED_FIRST, DENIAL_TURN_CUT, EVENT_STAT_IDS,
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
from agents.battle.turn_view import _classify_faint_cause, damage_is_lethal, is_protect_block

_SELF_KO_MOVE_IDS: frozenset = frozenset({"explosion", "selfdestruct"})
# The moves whose item lines move an item from one mon to the other (W3 above).
_ITEM_TRANSFER_WORDS: tuple = ("trick", "thief", "covet", "switcheroo")

def _classify_item_transition(kind, from_clause: "Optional[str]") -> int:
    """|-item| / |-enditem| (+ its `[from]`) -> one of the ITEM_TR_* ids.

    Gen3 has THREE ways an item stops being held and they mean different things to a player:
    a CONSUMED berry is spent (the mon is now itemless by its own design), a Knock Off REMOVAL
    is permanent in ADV (unlike later gens), and a Trick/Thief/Covet SWAP means the OPPONENT is
    now holding it — which is information about their set, not just about ours. Collapsing them
    into one "item gone" bit is the conflation this column exists to end."""
    from agents.battle.battle_event import EventKind
    fc = (from_clause or "").strip().lower()
    # gen3_event_window_semantics_fixes_v1 (W3): a transfer move writes an item line on BOTH
    # mons — Trick `|-item|` ×2, Thief / Covet `|-enditem|` on the victim + `|-item|` on the
    # taker. Every one of them is the item CHANGING HANDS, so the `|-item|` side is SWAPPED too;
    # reading it as REVEALED ("disclosed, still held") recorded a Trick as two plain reveals.
    if any(w in fc for w in _ITEM_TRANSFER_WORDS):
        # gen3_event_record_v2 (E12): the DIRECTION of the transfer — the `|-enditem|` side lost
        # it (SWAPPED: taken from this mon), the `|-item|` side RECEIVED it (Trick's two lines are
        # both receipts).
        return ITEM_TR_SWAPPED if kind is EventKind.ENDITEM else ITEM_TR_RECEIVED
    if kind is EventKind.ITEM:
        return ITEM_TR_REVEALED
    if "knock off" in fc or "knockoff" in fc:
        return ITEM_TR_REMOVED
    return ITEM_TR_CONSUMED


# gen3_event_record_v2 (E12): the end-of-turn RESIDUAL block, recognised by its causes (the Rust
# record's `residual_cause`, over the event's `[from]` clause with any `item:` / `move:` prefix
# stripped). A faint in the residual block is not an action-phase faint: it cuts no turn.
_RESIDUAL_CAUSES: frozenset = frozenset({
    "psn", "tox", "brn", "sandstorm", "hail", "leech seed", "nightmare", "curse", "leftovers",
    "wish", "ingrain", "future sight", "doom desire"})


def _is_residual_cause(fc: "Optional[str]") -> bool:
    if not fc:
        return False
    low = fc.strip().lower()
    for prefix in ("item:", "move:", "ability:"):
        if low.startswith(prefix):
            low = low[len(prefix):].strip()
            break
    return low in _RESIDUAL_CAUSES


def _event_stat_id(stat: "Optional[str]") -> int:
    """A BOOST row's column-27 id — CRASH, don't drop, on a stat outside gen 3's seven."""
    if not stat:
        return 0
    key = str(stat).strip().lower()
    if key not in EVENT_STAT_IDS:
        raise ValueError(f"unknown stat {stat!r} for the H-B event window (EVENT_STAT_IDS)")
    return EVENT_STAT_IDS[key]


def _other(side: "Optional[str]") -> "Optional[str]":
    from agents.battle.battle_event import OPP, OURS
    return OPP if side == OURS else OURS if side == OPP else None


class EventWindowTracker:
    """Tier H-B (`designs/ai_v9/design_history_entity.md` §3 H-B): the last-N DECISION-RELEVANT
    events as typed records — the sequential residue the compiled tiers (recency, H-A) cannot
    carry, made queryable. PUBLIC protocol events only; within-battle only; **seq-idempotent**
    (the PairHistoryTracker convention), so an overlapping window or a rolled-back opponent
    RE-DECIDE can never append twice.

    One record per event in the H-B vocabulary (moves, switch-ins, faints, status
    applied/cured, boosts, item reveals, hazards, our rejected switches, cants, DENIALS). MODIFIER
    events (DAMAGE / MISS / FAIL / CRIT / IMMUNE / RESISTED / SUPEREFFECTIVE) do not get records —
    they ATTACH to their side's open same-turn MOVE record: damage lands on the target side and
    accumulates into the move's ``hp_delta``; the effectiveness trio sets ``eff``; the outcome trio
    sets flags. ``we_first`` marks the records of whichever side MOVED first that turn.
    ``forced_window`` tags events emitted while a side's active slot was empty after a faint.

    **gen3_event_record_v2 (E12, the reshape).** Each record also carries the Rust core's native
    per-action record faithfully (`record::Window`), entity-aligned: a switch-in's ENTRY reason
    (chosen / replacement / drag / Baton Pass), the mon it replaced (REL), the move that caused it,
    the Spikes layers it met and the chip it took (``hp_delta``); a DENIED row for every turn ACTOR
    that was chosen for the turn and never acted — it fainted first, or (gen 3 singles) a faint CUT
    the turn and cancelled every queued action — with the mon, the move and the faint cause that
    denied it; the CALLER of a called move; Pursuit on a switching target; a boost's STAT; an item
    transfer's direction and partner; Destiny Bond / Perish Song faint causes; and (E4) which bench
    mon a refused switch aimed at. 🚨 The fold receives NO choices, so a DENIED row cannot carry the
    denied side's intent — the information boundary holds by construction.

    Records are plain dicts (species/move ids as STRINGS — the encoder maps to nums). The window is
    bounded (``maxlen``); reads return most-recent-LAST so the encoder's padding is stable."""

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
        self._last_dmg_cause: dict = {}
        self._last_dmg_lethal: dict = {}
        self._used_selfko: dict = {}   # side -> did its last move self-KO
        self._last_mover: Optional[str] = None
        self._first_mover_turn: int = -1
        self._first_mover_side: Optional[str] = None
        self._forced: dict = {}             # side -> active slot empty (post-faint window)
        # --- gen3_event_record_v2 (E12) ---
        self._fainted_pending: dict = {}    # side -> the fainted mon its next entry replaces
        self._baton_turn: dict = {}         # side -> the turn its Baton Pass is pending on
        self._spikes: dict = {}             # side -> Spikes layers
        self._open_entry: dict = {}         # side -> the SWITCH_IN row still taking its entry chip
        self._perish_pending: dict = {}     # side -> species at perish0
        self._db_pending: dict = {}         # side -> its mon dies to a Destiny Bond now
        self._act_turn: int = -1            # the turn whose ACTION phase is being folded
        self._turn_actors: dict = {}        # side -> its active at that turn's first event
        self._acted: dict = {}              # side -> the turn actor acted (or was denied)
        self._last_act: Optional[tuple] = None   # (actor, side, move_id) of the latest MOVE action
        self._first_faint: Optional[tuple] = None  # (species, side, cause, last_act)
        self._pending_denials: list = []    # fainted-first rows, placed after their action
        self._actions_closed: bool = False

    @staticmethod
    def _rec(t: int, actor, side, turn: int, **kw) -> dict:
        rec = {"t": t, "actor": actor, "side": side, "target": None, "move_id": None,
               "hp_delta": 0.0, "missed": False, "failed": False, "crit": False, "eff": 0,
               "we_first": False, "status": 0, "turn": turn,
               "rel": None, "rel_side": None, "entry": 0, "denial": 0, "caller": None,
               "stat": 0, "layers": 0, "pursuit": False}
        rec.update(kw)
        return rec

    def _append(self, rec: dict) -> dict:
        rec["forced_window"] = 1.0 if (self._forced.get("our") or self._forced.get("opp")) else 0.0
        self._events.append(rec)
        return rec

    def _active(self, side) -> Optional[str]:
        from agents.battle.battle_event import OURS
        return self._our_active if side == OURS else self._opp_active

    # ---- E12: the turn's action phase and its DENIALS ----------------------------------------
    def _start_turn(self, et: int) -> None:
        from agents.battle.battle_event import OPP, OURS
        self._act_turn = et
        self._turn_actors = {OURS: self._our_active, OPP: self._opp_active}
        self._acted = {OURS: False, OPP: False}
        self._first_faint = None
        self._last_act = None
        self._actions_closed = False

    def _mark_acted(self, side, species) -> None:
        if side is not None and species and self._turn_actors.get(side) == species:
            self._acted[side] = True

    def _flush_denials(self, cut: bool) -> None:
        """Place the pending fainted-first rows, then — once a faint has CUT the turn (gen 3
        singles: any faint cancels every remaining queued action) — one TURN-CUT row per turn
        actor that neither acted nor fainted."""
        from agents.battle.battle_event import OPP, OURS
        for rec in self._pending_denials:
            self._append(rec)
        self._pending_denials = []
        if not cut or self._first_faint is None:
            return
        f_sp, f_side, f_cause, f_act = self._first_faint
        for side in (OURS, OPP):
            actor = self._turn_actors.get(side)
            if not actor or self._acted.get(side):
                continue
            self._acted[side] = True
            self._append(self._rec(
                EVENT_T_DENIED, actor, side, self._act_turn, denial=DENIAL_TURN_CUT,
                rel=f_sp, rel_side=f_side, faint_cause=f_cause,
                move_id=(f_act[2] if f_act is not None else None)))

    def _close_actions(self) -> None:
        if self._actions_closed or self._act_turn <= 0:
            self._actions_closed = True
            return
        self._flush_denials(cut=True)
        self._actions_closed = True

    def _faint_cause(self, side, sp) -> str:
        if self._perish_pending.get(side) == sp:
            return "perishsong"
        if self._db_pending.get(side):
            return "destinybond"
        return _classify_faint_cause(
            self._last_dmg_cause.get(side), bool(self._used_selfko.get(side)),
            bool(self._last_dmg_lethal.get(side)))

    def _clear_side(self, side) -> None:
        self._last_dmg_cause.pop(side, None)
        self._last_dmg_lethal.pop(side, None)
        self._used_selfko.pop(side, None)
        self._perish_pending.pop(side, None)
        self._db_pending.pop(side, None)

    def update(self, turn: int, events, our_active: Optional[str],
               opp_active: Optional[str], attempted_switch: Optional[str] = None) -> None:
        """Fold the events since the last decision. ``attempted_switch`` (E4) is the species OUR
        previous decision tried to switch to — the target a ``CHOICE_REJECTED`` row names (the
        server's refusal does not say it; the caller knows what it sent)."""
        from agents.battle.battle_event import (
            IMPLIED_NONE, IMPLIED_USER, OURS, OPP, EventKind, implied_move_target)
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
            if et != self._act_turn:
                self._close_actions()
                self._start_turn(et)
            if k in (EventKind.MOVE, EventKind.SWITCH, EventKind.DRAG, EventKind.CANT,
                     EventKind.CHOICE_REJECTED):
                # an ACTION opens: the previous one closed, so its denials take their place
                self._flush_denials(cut=True)
                if self._first_faint is not None:
                    self._actions_closed = True
            elif not self._actions_closed and _is_residual_cause(e.from_clause):
                self._close_actions()
            if k is EventKind.MOVE and side is not None and sp:
                if self._first_mover_turn != et:
                    self._first_mover_turn = et
                    self._first_mover_side = side
                _implied = implied_move_target(e.move_id, sp)
                rec = self._append(self._rec(
                    EVENT_T_MOVE, sp, side, et,
                    target=(sp if _implied == IMPLIED_USER
                            else None if _implied == IMPLIED_NONE
                            else (self._opp_active if side == OURS else self._our_active)),
                    move_id=e.move_id, we_first=side == self._first_mover_side,
                    caller=e.from_move,
                    pursuit=bool(e.value.get("pursuit_switch"))))
                self._open_move[side] = rec
                self._last_mover = side
                self._used_selfko[side] = (e.move_id in _SELF_KO_MOVE_IDS)
                self._last_act = (sp, side, e.move_id)
                self._mark_acted(side, sp)
                if e.move_id == "batonpass":
                    self._baton_turn[side] = et
            elif k is EventKind.DAMAGE and side is not None:
                self._last_dmg_cause[side] = e.from_clause   # None ⇒ a direct hit
                self._last_dmg_lethal[side] = damage_is_lethal(e)
                fc = (e.from_clause or "").strip().lower()
                if fc == "confusion":
                    # a confused mon that hit itself DID take its turn (no `|move|` line prints)
                    self._mark_acted(side, sp)
                ent = self._open_entry.get(side)
                if (fc == "spikes" and ent is not None and ent["turn"] == et and sp
                        and ent["actor"] == sp and e.amount is not None):
                    ent["hp_delta"] += float(e.amount)       # the entry chip, on the entry row
                mover = OPP if side == OURS else OURS
                om = self._open_move.get(mover)
                if (om is not None and om["turn"] == et
                        and not e.from_clause
                        and self._last_mover == mover
                        and sp and om["target"] == sp):
                    amt = e.amount
                    if amt is not None:
                        om["hp_delta"] += float(amt)
            elif k in (EventKind.MISS, EventKind.FAIL, EventKind.CRIT) and side is not None:
                om = self._open_move.get(side)
                external_cause = (k is EventKind.FAIL
                                  and e.from_clause not in (None, "move-suffix"))
                if om is not None and om["turn"] == et and not external_cause:
                    if k is EventKind.MISS:
                        om["missed"] = True
                    elif k is EventKind.FAIL:
                        om["failed"] = True
                        if om["move_id"] == "batonpass":
                            self._baton_turn.pop(side, None)   # a failed pass carries nothing
                    else:
                        om["crit"] = True
            elif k is EventKind.ACTIVATE and side is not None and is_protect_block(e.effect):
                mover = OPP if side == OURS else OURS
                om = self._open_move.get(mover)
                if om is not None and om["turn"] == et and self._last_mover == mover:
                    om["failed"] = True
            elif k is EventKind.ACTIVATE and side is not None and "destiny" in (e.effect or "").lower():
                # `|-activate|<DB user>|move: Destiny Bond` — the OTHER side's mon faints next
                self._db_pending[_other(side)] = True
            elif k is EventKind.VOLATILE_START and side is not None and sp \
                    and (e.effect or "").strip().lower() == "perish0":
                self._perish_pending[side] = sp
            elif k in (EventKind.IMMUNE, EventKind.RESISTED, EventKind.SUPEREFFECTIVE) \
                    and side is not None:
                om = self._open_move.get(side)
                if om is not None and om["turn"] == et:
                    om["eff"] = {EventKind.SUPEREFFECTIVE: 1, EventKind.RESISTED: 2,
                                 EventKind.IMMUNE: 3}[k]
            elif k in (EventKind.SWITCH, EventKind.DRAG) and side is not None and sp:
                out = self._active(side)
                move_id = None
                if k is EventKind.DRAG:
                    entry, rel = ENTRY_DRAG, out
                    om = self._open_move.get(_other(side))
                    if om is not None and om["turn"] == et:
                        move_id = om["move_id"]               # the phazing move (Roar / Whirlwind)
                elif self._fainted_pending.get(side):
                    entry, rel = ENTRY_REPLACEMENT, self._fainted_pending[side]
                elif self._baton_turn.get(side) == et:
                    entry, rel, move_id = ENTRY_BATON_PASS, out, "batonpass"
                else:
                    entry, rel = ENTRY_CHOSEN, out
                if entry in (ENTRY_CHOSEN, ENTRY_BATON_PASS):
                    self._mark_acted(side, out)
                self._baton_turn.pop(side, None)
                self._fainted_pending.pop(side, None)
                # append BEFORE clearing the forced flag: the arriving replacement IS the
                # forced-window event.
                rec = self._append(self._rec(
                    EVENT_T_SWITCH_IN, sp, side, et,
                    target=(self._opp_active if side == OURS else self._our_active),
                    move_id=move_id, entry=entry, rel=rel, rel_side=(side if rel else None),
                    layers=int(self._spikes.get(side, 0))))
                self._open_entry[side] = rec
                self._forced["our" if side == OURS else "opp"] = False
                if side == OURS:
                    self._our_active = sp
                else:
                    self._opp_active = sp
                self._clear_side(side)
            elif k is EventKind.FAINT and side is not None and sp:
                cause = self._faint_cause(side, sp)
                la = self._last_act
                by_attack = (cause == "attack" and la is not None and la[1] != side)
                self._append(self._rec(
                    EVENT_T_FAINT, sp, side, et, faint_cause=cause,
                    rel=(la[0] if by_attack else None), rel_side=(la[1] if by_attack else None),
                    layers=(int(self._spikes.get(side, 0)) if cause == "hazard" else 0)))
                if not self._actions_closed and et > 0:
                    if self._first_faint is None:
                        self._first_faint = (sp, side, cause, la)
                    if self._turn_actors.get(side) == sp and not self._acted.get(side):
                        self._acted[side] = True
                        self._pending_denials.append(self._rec(
                            EVENT_T_DENIED, sp, side, et, denial=DENIAL_FAINTED_FIRST,
                            faint_cause=cause, move_id=(la[2] if la is not None else None),
                            rel=(la[0] if la is not None else None),
                            rel_side=(la[1] if la is not None else None)))
                self._clear_side(side)
                self._open_entry.pop(side, None)
                self._fainted_pending[side] = sp
                if side == OURS and self._our_active == sp:
                    self._our_active = None
                    self._forced["our"] = True
                elif side == OPP and self._opp_active == sp:
                    self._opp_active = None
                    self._forced["opp"] = True
            elif k in (EventKind.STATUS, EventKind.CURESTATUS) and sp:
                self._append(self._rec(
                    (EVENT_T_STATUS_APPLIED if k is EventKind.STATUS else EVENT_T_STATUS_CURED),
                    sp, side, et, status=_event_status_id(e.status)))
            elif k in (EventKind.BOOST, EventKind.UNBOOST) and sp:
                # W1: `amount` is ALREADY SIGNED (an `|-unboost|` reads negative).
                self._append(self._rec(
                    EVENT_T_BOOST, sp, side, et, hp_delta=float(e.amount or 0.0),
                    stat=_event_stat_id(e.value.get("stat"))))
            elif k in (EventKind.ITEM, EventKind.ENDITEM) and sp:
                tr = _classify_item_transition(k, e.from_clause)
                other = (_other(side) if tr in (ITEM_TR_SWAPPED, ITEM_TR_RECEIVED, ITEM_TR_REMOVED)
                         else None)
                rel = self._active(other) if other is not None else None
                self._append(self._rec(
                    EVENT_T_ITEM_REVEAL, sp, side, et, item_tr=tr,
                    rel=rel, rel_side=(other if rel else None)))
            elif k is EventKind.SIDE and side is not None:
                start = e.value.get("op") != "sideend"
                is_spikes = "spikes" in str(e.value.get("condition") or "").lower()
                if is_spikes:
                    self._spikes[side] = min(3, self._spikes.get(side, 0) + 1) if start else 0
                # W5: +1 a side condition STARTED, −1 one ENDED.
                self._append(self._rec(
                    EVENT_T_HAZARD, None, side, et, hp_delta=(1.0 if start else -1.0),
                    layers=(int(self._spikes.get(side, 0)) if is_spikes else 0)))
            elif k is EventKind.CANT and sp:
                # gen3_damp_cant_v1: the row is the mon that LOST ITS TURN.
                _cs = e.blocked_side or side
                _ca = e.blocked_actor or sp
                self._mark_acted(_cs, _ca)
                self._last_act = None
                self._append(self._rec(
                    EVENT_T_CANT, _ca, _cs, et, move_id=e.cant_move, failed=True, cant=e.reason))
            elif k is EventKind.CHOICE_REJECTED:
                # E4: the refused switch's TARGET — which bench mon the trapped switch aimed at.
                self._append(self._rec(
                    EVENT_T_SWITCH_REJECTED, self._our_active, OURS, et, target=attempted_switch))
            if k in (EventKind.SWITCH, EventKind.DRAG, EventKind.CHOICE_REJECTED):
                self._last_act = None
        # A DECISION closes the window. A faint that cut the turn ends its action phase here (the
        # forced replacement follows); a mid-turn Baton Pass decision leaves it open.
        self._flush_denials(cut=self._first_faint is not None)
        if self._first_faint is not None:
            self._actions_closed = True
        # Decision-time resync (the alive-filter is applied by the CALLER).
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
        """The records that may still be MUTATED IN PLACE: one open MOVE per side, and one open
        SWITCH_IN per side (E12: its Spikes entry chip). The incremental obs assembler re-writes
        exactly these rows each decision."""
        return tuple(self._open_move.values()) + tuple(self._open_entry.values())
