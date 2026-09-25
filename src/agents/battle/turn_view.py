"""``TurnView`` — the ergonomic per-turn read model over the event log.

This is the **long-term consumer API**. Today ``TurnDelta.build`` is a detective: it
diffs two ``BattleContext`` snapshots and recovers facts the snapshot dropped with a
pile of heuristics (KO-before-acting, phaze recovery, ``opp_all_last_move_ids``,
effectiveness alignment). Once the event log exists, "what happened this turn" is a
**fold**, not a reconstruction — and both ``TurnDelta`` *and* the reward manager should
read it through one surface so they can never disagree.

``TurnView`` is that surface. Build it once per turn:

    view = TurnView.for_turn(battle, turn)        # from a Gen3Battle
    view = TurnView.from_events(events, OURS)      # or from a raw event list (tests)

then read intent-level fields per side (``view.ours`` / ``view.opp``) and turn-level
facts (``view.we_moved_first``, ``view.damage_on(species)``). It is deliberately pure
(depends only on :class:`BattleEvent`), so it unit-tests on hand-built logs and the
eventual ``TurnDelta`` migration is a thin adapter rather than a rewrite.

Side labels are ``"ours"`` / ``"opp"`` (see :data:`agents.battle.battle_event.OURS`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from agents.battle.battle_event import OPP, OURS, BattleEvent, EventKind

# ---------------------------------------------------------------------------
# Faint-cause vocabulary (Step 4)
# ---------------------------------------------------------------------------
# The cause of each KO in a decision window, derived from the [from] clause
# of the lethal DAMAGE event. "other" is a catch-all for unrecognised sources
# — crash-don't-drop is NOT applied here: a novel [from] string is annotated
# "other" rather than raising, because it's an annotation gap, not a parser
# error. The encoder encodes causes as a multi-hot so multiple KOs in one
# window can contribute different bits.
FAINT_CAUSE_VOCAB: tuple = (
    "attack",    # KO'd by a direct move (no [from] clause on the lethal damage)
    "hazard",    # Entry hazard (Spikes / Stealth Rock)
    "weather",   # Residual weather damage (Sandstorm / Hail)
    "status",    # Residual status damage (Burn / Poison / Toxic)
    "recoil",    # Recoil damage
    "selfko",    # User of Explosion / Selfdestruct kills themselves
    "leechseed", # Leech Seed residual
    "other",     # Any unclassified [from] source
)
FAINT_CAUSE_DIM: int = len(FAINT_CAUSE_VOCAB)  # 8
_FAINT_CAUSE_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(FAINT_CAUSE_VOCAB)}

def damage_is_lethal(e: BattleEvent) -> bool:
    """False iff a DAMAGE event is KNOWN to have left its mon above 0 HP. ``hp_after`` is an
    OPTIONAL key (``EVENT_OPTIONAL_KEYS``; the live builder always writes it), so an event without
    one is unknown and keeps the classic reading — only a faint whose last damage demonstrably did
    not KO (or that had no damage line at all) is re-read as "not an attack"."""
    hp_after = e.value.get("hp_after")
    return hp_after is None or float(hp_after) <= 0.0


# `|-activate|<target>|move: Protect` — the target's Protect / Detect STOPPED the move being used
# (gen3_event_window_semantics_fixes_v1, W4). Endure's `-activate` is NOT a block: the move still
# hits, to 1 HP. Bare and `move:`-prefixed forms both occur in the gen-3 sim.
_PROTECT_BLOCK_EFFECTS: frozenset = frozenset({"protect", "detect", "move: protect", "move: detect"})


def is_protect_block(effect: Optional[str]) -> bool:
    """True iff an ``|-activate|`` effect is a Protect / Detect BLOCK of the move being used.
    ONE predicate for the H-B event window and :class:`TurnView`, so the two cannot disagree on
    what a block is (the Rust core's `record.rs` / `history.rs` / `turnview.rs` match the same set)."""
    return (effect or "").strip().lower() in _PROTECT_BLOCK_EFFECTS


# Self-KO moves: the user always faints (even on a miss/immune target in Gen 3
# they still die), so a faint on the user's own side this turn is classified selfko.
_SELF_KO_MOVES: frozenset = frozenset({"explosion", "selfdestruct"})


@dataclass(frozen=True)
class FaintDetail:
    """Cause-annotated faint for one Pokémon in a decision window."""
    species: str
    side: str   # OURS or OPP
    cause: str  # element of FAINT_CAUSE_VOCAB


def faint_cause_id(cause: Optional[str]) -> int:
    """Faint-cause label -> a 1-based id into :data:`FAINT_CAUSE_VOCAB`; ``None`` -> 0.

    The EMBEDDING-routed counterpart of the lag frames' multi-hot, for the H-B event window's
    `faint_cause_id` column (gen3_event_semantics_v1). 1-based so 0 can mean "not a FAINT row",
    which every other record type must read. Lives HERE, beside the vocabulary and
    :func:`_classify_faint_cause`, so the frames-era encoding and the event window cannot drift
    on what a cause label means — one vocabulary, one index map, two consumers.

    Raises on an unknown label: the vocabulary is closed, and a silent 0 would read as "no faint"
    on a row that IS one."""
    if cause is None:
        return 0
    return _FAINT_CAUSE_TO_IDX[cause] + 1


def _classify_faint_cause(from_clause: Optional[str], used_selfko: bool,
                          lethal: bool = True) -> str:
    """Map a DAMAGE event's ``[from]`` clause + self-KO flag → faint cause label.

    ``lethal`` is whether the fainting mon's LAST damage line took it to 0 HP. A faint that no
    damage line caused — Destiny Bond (``|-activate|<user>|move: Destiny Bond`` then the attacker's
    ``|faint|``), Perish Song (``|-start|<mon>|perish0`` then ``|faint|``), Memento — has no ``[from]``
    to read, and the old "no ``[from]`` ⇒ attack" fall-through called it a KO by a direct move
    (gen3_event_window_semantics_fixes_v1, W2). The vocabulary has no ``perishsong`` /
    ``destinybond`` entry (that is the next layout change), so it is ``"other"`` — the honest
    catch-all — never ``"attack"``."""
    if used_selfko:
        return "selfko"
    if not lethal:
        return "other"
    if from_clause is None:
        return "attack"
    fc = from_clause.strip().lower()
    if fc == "spikes":
        return "hazard"
    if fc in ("sandstorm", "hail"):
        return "weather"
    if fc in ("psn", "tox", "brn", "burn"):
        return "status"
    if "recoil" in fc:
        return "recoil"
    if "leech seed" in fc or "leechseed" in fc:
        return "leechseed"
    return "other"


@dataclass(frozen=True)
class DamagingMove:
    """A damaging move resolved from the log — the event-sourced replacement for
    poke-env's ``DamagingMoveEvent`` (same fields, sourced from the ordered log)."""

    user_species: Optional[str]
    target_species: Optional[str]
    target_status: Optional[str]  # status of the target AT MOVE-FIRE TIME
    move_id: Optional[str]
    effectiveness: Optional[float]  # 0/0.5/1/2; None if the protocol never disclosed it


@dataclass
class SideTurn:
    """Everything one side did on one turn, folded from its events."""

    side: str
    moved: bool = False  # a MOVE event resolved for this side
    move_id: Optional[str] = None  # the *executed* move (delegation-aware)
    called_via: Optional[str] = None  # delegating move (Sleep Talk / Metronome / …)
    switched: bool = False
    switched_to: Optional[str] = None
    drag: bool = False  # phazed in (Roar / Whirlwind), as opposed to a chosen switch
    fainted: bool = False
    cant_reason: Optional[str] = None  # verbatim |cant| reason ("slp", "Focus Punch", …)
    cant_move: Optional[str] = None  # the move that was prevented, if the line named one
    crit: bool = False
    missed: bool = False
    failed: bool = False
    effectiveness: Optional[float] = None
    target_species: Optional[str] = None
    damaging_move: Optional[DamagingMove] = None
    boosts: Dict[str, int] = field(default_factory=dict)  # net stat-stage change this turn
    status_applied: Optional[str] = None  # status this side's active GAINED this turn
    status_cured: Optional[str] = None  # status this side's active LOST this turn
    item_lost: Optional[str] = None  # item consumed/knocked off this turn (|-enditem|)
    item_gained: Optional[str] = None  # item revealed/gained this turn (|-item|)
    blocked: bool = False  # this side's move was STOPPED by the target's Protect / Detect
    #   (`|-activate|<target>|move: Protect`) — outcome "fail" (gen3_event_window_semantics_fixes_v1)
    hit_dealt: float = 0.0  # HP fraction THIS side's moves' own hits took off the other side
    #   (≤ 0): a `-damage` with no `[from]` on the other side while THIS side is the current mover
    #   (the latest `|move|`). Sand / poison / recoil / Spikes on the entrant / the opponent's own
    #   Substitute or Belly Drum cost are not our hit (T1).
    choice_overridden: bool = False  # an Encore landed on this side's mon BEFORE it moved this
    #   turn, so the move it executed is the encored one, not the one it chose (L5)
    attempted_rejected: bool = False  # an action this side chose was REFUSED by the server
    #   this window (|error|[Unavailable choice] — a switch tried while trapped). Folded from
    #   the out-of-band CHOICE_REJECTED event; only ever set on OUR side (the opponent's
    #   rejections are not observable).

    @property
    def failed_to_move(self) -> bool:
        """True when a ``|cant|`` prevented this side from acting."""
        return self.cant_reason is not None

    @property
    def outcome(self) -> Optional[str]:
        """``"hit"`` / ``"miss"`` / ``"fail"``, or ``None`` when no move resolved
        (switch / cant). Mirrors the old ``_derive_move_outcome`` precedence."""
        if not self.moved:
            return None
        if self.missed:
            return "miss"
        if self.failed or self.blocked:
            return "fail"
        return "hit"


class TurnView:
    """Folded, read-only view of a single turn's events."""

    def __init__(self, turn: int, events: Sequence[BattleEvent], our_side: str = OURS):
        self.turn = turn
        self.events: List[BattleEvent] = list(events)
        self.our_side = our_side
        self.ours = self._fold_side(OURS)
        self.opp = self._fold_side(OPP)
        self._fold_attribution()
        self._move_order: List[str] = self._compute_move_order()

    # ---- constructors ----
    @classmethod
    def from_events(
        cls, events: Sequence[BattleEvent], our_side: str = OURS
    ) -> "TurnView":
        turn = events[0].turn if events else 0
        return cls(turn, events, our_side)

    @classmethod
    def for_turn(cls, battle, turn: int, our_side: str = OURS) -> "TurnView":
        return cls(turn, battle.events_for_turn(turn), our_side)

    # ---- per-side fold ----
    def _fold_side(self, side: str) -> SideTurn:
        st = SideTurn(side=side)
        side_events = [e for e in self.events if e.side == side]

        move_events = [e for e in side_events if e.kind is EventKind.MOVE]
        if move_events:
            st.moved = True
            primary = move_events[-1]  # the executed move (Sleep Talk -> called move)
            st.move_id = primary.move_id
            st.called_via = primary.value.get("from_move")
            st.target_species = primary.target_species

        for e in side_events:
            if e.kind is EventKind.SWITCH:
                st.switched = True
                st.switched_to = e.actor_species
            elif e.kind is EventKind.DRAG:
                st.switched = True
                st.drag = True
                st.switched_to = e.actor_species
            elif e.kind is EventKind.FAINT:
                st.fainted = True
            elif e.kind is EventKind.CANT:
                st.cant_reason = e.reason
                st.cant_move = e.cant_move
            elif e.kind is EventKind.CRIT:
                st.crit = True
            elif e.kind is EventKind.MISS:
                st.missed = True
            elif e.kind is EventKind.FAIL:
                # A `-fail` carrying a real `[from]` cause (`ability: Clear Body` blocking
                # Intimidate on a switch-in) is another effect fizzling, not this side's move
                # failing; the synthetic "move-suffix" outcome IS the move's own and counts.
                if e.from_clause in (None, "move-suffix"):
                    st.failed = True
            elif e.kind in (
                EventKind.IMMUNE,
                EventKind.RESISTED,
                EventKind.SUPEREFFECTIVE,
            ):
                st.effectiveness = e.multiplier
            elif e.kind in (
                EventKind.BOOST,
                EventKind.UNBOOST,
                EventKind.SETBOOST,
            ):
                if e.stat is not None:
                    st.boosts[e.stat] = st.boosts.get(e.stat, 0) + int(e.amount or 0)
            elif e.kind is EventKind.STATUS:
                st.status_applied = e.status
            elif e.kind is EventKind.CURESTATUS:
                st.status_cured = e.status
            elif e.kind is EventKind.ENDITEM:
                st.item_lost = e.item
            elif e.kind is EventKind.ITEM:
                st.item_gained = e.item
            elif e.kind is EventKind.CHOICE_REJECTED:
                st.attempted_rejected = True

        if st.moved and move_events:
            primary = move_events[-1]
            # A move is "damaging" here if the protocol disclosed effectiveness for it
            # or it dealt HP damage to the named target — that is exactly when the old
            # DamagingMoveEvent was populated.
            target = primary.target_species
            # the target sits on the OTHER side; pass it so a mirror match resolves
            target_side = OPP if side == OURS else OURS
            dealt = self.damage_on(target, side=target_side) if target else 0.0
            if st.effectiveness is not None or dealt < 0.0:
                st.damaging_move = DamagingMove(
                    user_species=primary.actor_species,
                    target_species=target,
                    target_status=primary.value.get("target_status"),
                    move_id=primary.move_id,
                    effectiveness=st.effectiveness
                    if st.effectiveness is not None
                    else (1.0 if dealt < 0.0 else None),
                )
        return st

    def _fold_attribution(self) -> None:
        """The facts that need the CROSS-SIDE event order (gen3_event_window_semantics_fixes_v1):

        * **the current mover's own hits** (``hit_dealt``) — a ``-damage`` with no ``[from]`` on the
          side opposite the latest ``|move|``'s user. A ``[from]`` marks every residual, recoil,
          hazard, weather and item chip; the current-mover rule excludes the other side's own
          no-``[from]`` HP costs (Substitute, Belly Drum, a Ghost's Curse, which print a bare
          ``-damage`` on their user while THEY are moving).
        * **a Protect / Detect block** (``blocked``) — ``|-activate|<target>|move: Protect`` stops
          the current mover's move; the activate line names the PROTECTOR, so it is the other
          side's move that is blocked.
        * **an Encore override** (``choice_overridden``) — an Encore ``-start`` on a side's mon
          BEFORE that side's move in the same turn: the move it executes is the encored one.
        """
        mover: Optional[str] = None
        encored_turn: Dict[str, int] = {}
        for e in self.events:
            k = e.kind
            if k is EventKind.MOVE and e.side in (OURS, OPP):
                mover = e.side
                if encored_turn.get(e.side) == e.turn:
                    self._side(e.side).choice_overridden = True
            elif k is EventKind.DAMAGE and mover is not None and e.side in (OURS, OPP) \
                    and e.side != mover and not e.from_clause:
                self._side(mover).hit_dealt += float(e.value.get("amount", 0.0))
            elif k is EventKind.ACTIVATE and e.side in (OURS, OPP) and mover is not None \
                    and e.side != mover and is_protect_block(e.effect):
                self._side(mover).blocked = True
            elif k is EventKind.VOLATILE_START and e.side in (OURS, OPP) \
                    and (e.effect or "").strip().lower() in ("encore", "move: encore"):
                encored_turn[e.side] = e.turn

    def _side(self, side: str) -> "SideTurn":
        return self.ours if side == OURS else self.opp

    def _compute_move_order(self) -> List[str]:
        """Sides in the order they first acted this turn (first element moved first)."""
        order: List[str] = []
        for e in self.events:
            if e.kind is EventKind.MOVE and e.side in (OURS, OPP) and e.side not in order:
                order.append(e.side)
        return order

    # ---- turn-level reads ----
    @property
    def move_order(self) -> List[str]:
        return list(self._move_order)

    @property
    def we_moved_first(self) -> Optional[bool]:
        """True if our side acted before the opponent. ``None`` when fewer than both
        sides used a move (e.g. one or both switched)."""
        if len(self._move_order) < 2:
            return None
        return self._move_order[0] == OURS

    def damage_on(self, species: Optional[str], side: Optional[str] = None) -> float:
        """Net HP-fraction change on ``species`` this turn (negative = net damage),
        summed across all DAMAGE/HEAL events naming it. ``0.0`` if unknown.

        Pass ``side`` ("ours"/"opp") to disambiguate a **mirror match** where both
        sides field the same species — without it, a bare species name sums across
        both Tyranitars. The reward manager (step 5) should always pass ``side``.
        """
        if species is None:
            return 0.0
        total = 0.0
        for e in self.events:
            if (
                e.actor_species == species
                and (side is None or e.side == side)
                # SETHP (Pain Split) emits no -damage/-heal of its own, so it must be
                # counted here for the HP trajectory to be complete.
                and e.kind in (EventKind.DAMAGE, EventKind.HEAL, EventKind.SETHP)
            ):
                total += float(e.value.get("amount", 0.0))
        return total

    def faints(self) -> List[str]:
        """Species that fainted this turn, in revealed order."""
        return [
            e.actor_species
            for e in self.events
            if e.kind is EventKind.FAINT and e.actor_species is not None
        ]

    @property
    def someone_fainted(self) -> bool:
        """True if any mon fainted this turn (the reward manager's KO trigger)."""
        return any(e.kind is EventKind.FAINT for e in self.events)

    @property
    def both_attacked(self) -> bool:
        """True if both sides executed a move (no switch / cant on either side)."""
        return self.ours.moved and self.opp.moved

    @property
    def anyone_switched(self) -> bool:
        """True if either side switched or was phazed."""
        return self.ours.switched or self.opp.switched

    # ---- board-effect accessors (typed, so consumers never walk raw events) ----
    def hazards_set(self) -> List[BattleEvent]:
        """SIDE-condition starts this turn (Spikes / Reflect / Light Screen / …).
        Each event's ``.side`` is the side the hazard was laid on, ``.effect`` the
        condition id, and ``.value['op']`` is ``sidestart``/``sideend``."""
        return [e for e in self.events if e.kind is EventKind.SIDE]

    def weather_set(self) -> Optional[str]:
        """The weather id set/continued this turn, or None."""
        for e in self.events:
            if e.kind is EventKind.WEATHER:
                return e.weather
        return None

    def statuses_applied(self) -> List[BattleEvent]:
        """STATUS-inflicted events this turn (``.side``/``.actor_species``/``.status``)."""
        return [e for e in self.events if e.kind is EventKind.STATUS]

    def items_lost(self) -> List[BattleEvent]:
        """ENDITEM events this turn — Berry pops, Knock Off, Trick
        (``.actor_species``/``.item``/``.from_cause``/``.of_source``)."""
        return [e for e in self.events if e.kind is EventKind.ENDITEM]

    def events_of(self, kind: EventKind) -> List[BattleEvent]:
        """Escape hatch: every event of ``kind`` this turn, in order. Lets a consumer
        reach any fact the typed accessors don't name without walking ``self.events``."""
        return [e for e in self.events if e.kind is kind]

    def faint_details(self) -> List[FaintDetail]:
        """Cause-annotated list of every faint in this decision window.

        For each FAINT event the cause is derived from the most recent DAMAGE
        event on the same (species, side) pair preceding the FAINT (the
        ``[from]`` clause of that damage, stored as ``event.reason``). An
        unrecognised ``[from]`` maps to ``"other"`` — crash-don't-drop is NOT
        applied here because a novel source is an annotation gap, not a parser
        error.

        Self-KO (Explosion / Selfdestruct): if the fainting side used a self-KO
        move this turn, that side's faint is classified ``"selfko"`` regardless
        of the last damage reason.
        """
        # Build (species, side) → last DAMAGE [from] clause, in event order, and whether that
        # damage took the mon to 0 HP (a faint no damage line caused is not an `attack`, W2).
        last_damage_reason: Dict[tuple, Optional[str]] = {}
        last_damage_lethal: Dict[tuple, bool] = {}
        for e in self.events:
            if e.kind is EventKind.DAMAGE and e.actor_species and e.side:
                last_damage_reason[(e.actor_species, e.side)] = e.reason
                last_damage_lethal[(e.actor_species, e.side)] = damage_is_lethal(e)

        details: List[FaintDetail] = []
        for e in self.events:
            if e.kind is not EventKind.FAINT or not e.actor_species or not e.side:
                continue
            side_turn = self.ours if e.side == OURS else self.opp
            used_selfko = (
                side_turn.moved
                and side_turn.move_id is not None
                and side_turn.move_id in _SELF_KO_MOVES
            )
            from_clause = last_damage_reason.get((e.actor_species, e.side))
            lethal = last_damage_lethal.get((e.actor_species, e.side), False)
            cause = _classify_faint_cause(from_clause, used_selfko, lethal)
            details.append(FaintDetail(species=e.actor_species, side=e.side, cause=cause))
        return details

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"TurnView(turn={self.turn}, order={self._move_order}, "
            f"ours={self.ours.move_id or ('switch' if self.ours.switched else None)}, "
            f"opp={self.opp.move_id or ('switch' if self.opp.switched else None)})"
        )
