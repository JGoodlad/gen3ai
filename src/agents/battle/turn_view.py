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

# Self-KO moves: the user always faints (even on a miss/immune target in Gen 3
# they still die), so a faint on the user's own side this turn is classified selfko.
_SELF_KO_MOVES: frozenset = frozenset({"explosion", "selfdestruct"})


@dataclass(frozen=True)
class FaintDetail:
    """Cause-annotated faint for one Pokémon in a decision window."""
    species: str
    side: str   # OURS or OPP
    cause: str  # element of FAINT_CAUSE_VOCAB


def _classify_faint_cause(from_clause: Optional[str], used_selfko: bool) -> str:
    """Map a DAMAGE event's ``[from]`` clause + self-KO flag → faint cause label."""
    if used_selfko:
        return "selfko"
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
        if self.failed:
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
        # Build (species, side) → last DAMAGE [from] clause, in event order.
        last_damage_reason: Dict[tuple, Optional[str]] = {}
        for e in self.events:
            if e.kind is EventKind.DAMAGE and e.actor_species and e.side:
                last_damage_reason[(e.actor_species, e.side)] = e.reason

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
            cause = _classify_faint_cause(from_clause, used_selfko)
            details.append(FaintDetail(species=e.actor_species, side=e.side, cause=cause))
        return details

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"TurnView(turn={self.turn}, order={self._move_order}, "
            f"ours={self.ours.move_id or ('switch' if self.ours.switched else None)}, "
            f"opp={self.opp.move_id or ('switch' if self.opp.switched else None)})"
        )
