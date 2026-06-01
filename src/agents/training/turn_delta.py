"""``TurnDelta`` — the per-decision HISTORY fold.

"What happened since the agent was last asked to act", folded from the event log
(``battle.events_since(cursor)`` → :class:`~agents.battle.turn_view.TurnView`) plus the
current-board reads of two consecutive :class:`~agents.training.battle_snapshot.BattleContext`
snapshots (HP-after, boosts, slot maps, phase, prev-active). The numeric per-turn quantities
(per-slot HP deltas, target-HP, faint causes, status transitions, move outcome, effectiveness)
come from the event log; the snapshot supplies only current-board values that are LiveView
projections (never a diff-detective reconstruction).

The field layout is FROZEN — it is consumed by ``turn_delta_encoder.py`` (the obs block) and
the reward manager. Changing a field is retrain-class.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import numpy as np

from poke_env.battle.abstract_battle import DamagingMoveEvent

from agents.gen3_mechanics import BOOST_DIM

if TYPE_CHECKING:
    from agents.enums import Status
    from agents.training.battle_snapshot import BattleContext


# Moves whose user always faints and which always connect when used (a neutral
# hit emits no effectiveness event, so the damaging-event "connected" signal would
# miss them; an immune target DOES emit, so it's covered by the event either way).
SELF_KO_MOVES = frozenset({"explosion", "selfdestruct"})


def _resolve_target_hp_delta(
    event: Optional[DamagingMoveEvent],
    hp_delta: np.ndarray,
    slot_map: dict,
) -> Optional[float]:
    """Look up the HP delta on the species named by event.target_species.

    Returns None when the event is None or the target species isn't in the
    slot map (shouldn't happen for confirmed damaging events, but defensive).
    """
    if event is None:
        return None
    slot = slot_map.get(event.target_species)
    if slot is None:
        return None
    return float(hp_delta[slot])


def _fold_hp_deltas(
    events: list,
    our_slot_map: dict,
    opp_slot_map: dict,
    prev_our_hp: np.ndarray,
    prev_opp_hp: np.ndarray,
    prev_opp_slot_map: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold per-slot HP deltas from the event log — bit-identical to ``curr_hp − prev_hp``.

    Each DAMAGE/HEAL/SETHP event carries the post-line HP FRACTION (``hp_after`` for
    DAMAGE/HEAL, ``hp`` for SETHP — see ``gen3_battle._build_event``). The LAST such
    event for a (side, species) in the window therefore holds that mon's HP at the
    next decision — exactly the fraction the LiveView snapshot stores as ``curr_hp``.
    So we fold a per-slot END HP from those ``hp_after`` values (last wins), default
    each slot to ``prev_hp`` (no event ⇒ unchanged), and subtract ``prev_hp`` ONCE.

    Why end-HP-then-subtract and NOT a sum of signed ``amount``s: a per-event-amount
    SUM accumulates float rounding differently from the single endpoint subtraction
    (~6e-8), which is enough to flip a discrete reward threshold (``opp_hp_delta.sum()
    >= 0`` in the futile-attack penalty). Reading the last ``hp_after`` and casting it
    to float32 reproduces ``curr_hp`` bit-for-bit, so ``end − prev_hp`` equals the
    snapshot diff exactly — event-sourced AND value-identical.

    FAINT-completeness (FINDING): a self-KO move (Explosion / Selfdestruct) faints the
    user with **no** ``|-damage|`` line — only ``|faint|`` — so its HP→0 never appears
    as an ``hp_after``. So a FAINT pins the slot's end HP to 0 (its true value), the
    one HP fact the damage stream alone cannot supply.

    Newly-revealed-opponent zeroing: a mon first revealed THIS window had
    ``prev_hp = 0`` (the unrevealed sentinel), so the old snapshot path zeroed its
    (necessarily-positive) delta as a spurious "gain". Reproduced by zeroing any opp
    slot whose species was not in the previous slot map — lossless, since gen3 entry
    hazards (Spikes ≤ 25%) can never KO a freshly-revealed mon from full.
    """
    from agents.battle.battle_event import OURS, OPP, EventKind
    our_end = prev_our_hp.astype(np.float32, copy=True)
    opp_end = prev_opp_hp.astype(np.float32, copy=True)
    for e in events:
        if e.kind in (EventKind.DAMAGE, EventKind.HEAL):
            hp_after = e.value.get("hp_after")
        elif e.kind is EventKind.SETHP:
            hp_after = e.value.get("hp")
        else:
            continue
        if hp_after is None:
            continue
        if e.side == OURS:
            slot = our_slot_map.get(e.actor_species)
            if slot is not None:
                our_end[slot] = np.float32(hp_after)
        elif e.side == OPP:
            slot = opp_slot_map.get(e.actor_species)
            if slot is not None:
                opp_end[slot] = np.float32(hp_after)
    for e in events:
        if e.kind is not EventKind.FAINT or not e.actor_species or not e.side:
            continue
        if e.side == OURS:
            slot = our_slot_map.get(e.actor_species)
            if slot is not None:
                our_end[slot] = np.float32(0.0)
        elif e.side == OPP:
            slot = opp_slot_map.get(e.actor_species)
            if slot is not None:
                opp_end[slot] = np.float32(0.0)
    our_delta = our_end - prev_our_hp
    opp_delta = opp_end - prev_opp_hp
    for species, slot in opp_slot_map.items():
        if species not in prev_opp_slot_map:
            opp_delta[slot] = 0.0
    return our_delta, opp_delta


@dataclass
class TurnDelta:
    """
    Diff between two consecutive BattleContexts.

    Built after each turn completes, capturing what actions were taken and what
    changed. Passed to the reward function and written into the info dict for
    callbacks to consume without touching the battle object.
    """
    # What we did this turn
    our_move_id: str | None       # move ID (e.g. "rockslide"), None if we switched
    our_switch_to: str | None     # species we switched to, None if we moved
    our_prev_active: str          # species that was active at turn start

    # What they did this turn
    opp_move_id: str | None       # move ID from poke-env's last_move tracking; None if switched
    opp_switch_to: str | None     # species they switched to, None if they moved
    opp_prev_active: str
    opp_move_known: bool          # False only when we know they attacked but have no move ID
                                  # (e.g. Explosion aftermath where the attacker is no longer active)

    # HP outcomes per slot (indexed by slot_map from BattleContext)
    our_hp_delta: np.ndarray      # (6,) float32 — negative means damage taken
    opp_hp_delta: np.ndarray      # (6,) float32

    # Faint events this turn
    we_fainted: bool
    opp_fainted: bool

    # Did each side fail to act this turn?
    # Derived from curr_ctx cant_reason — True whenever |cant| fired for that side.
    our_failed_to_move: bool
    our_cant_reason: str | None
    opp_failed_to_move: bool
    opp_cant_reason: str | None

    # Stat-stage deltas for each side's active Pokémon (BOOST_STATS order).
    # Positive = gained a stage, negative = lost a stage this turn.
    # Zero when the active mon switched (new mon starts from its own current stages).
    our_boost_delta: np.ndarray   # (7,) int8
    opp_boost_delta: np.ndarray   # (7,) int8

    # Type-effectiveness of each side's last damaging move (snapshotted from curr_ctx).
    # 0.0=immune, 0.5=resisted, 1.0=neutral, 2.0=super-effective.
    # None when the side switched, used a non-damaging move, or the battle just started.
    our_effectiveness: float | None
    opp_effectiveness: float | None

    # True = we executed our action before the opponent this turn.
    # None when one or both sides performed a normal switch.
    we_moved_first: bool | None

    # Full per-side damaging-move record (user / target / target_status at
    # fire time / move_id / effectiveness), pass-through from the curr_ctx
    # snapshot. None when the side didn't use a damaging move whose
    # effectiveness was confirmed by an explicit emission — preserves the
    # "skip on uncertainty" semantics of the underlying property. Use this
    # instead of (move_id, effectiveness) tuples for attribution-sensitive
    # consumers (reward shaping, replay recording) where the protocol-truth
    # user/target identity matters; the bare opp_move_id / opp_effectiveness
    # fields stay for callers that just need any signal.
    our_damaging_event: Optional[DamagingMoveEvent] = None
    opp_damaging_event: Optional[DamagingMoveEvent] = None

    # True when this delta closes on a forced_switch input request (mid-turn
    # replacement after a faint, end-of-turn replacement). False for normal
    # move-selection turns. Lets the model distinguish half-turn replacement
    # slots from full action-pair slots — without this, the absence of an opp
    # move in a forced-switch slot looks like "opp voluntarily passed."
    phase_is_forced_switch: bool = False

    # Per-slot HP levels AT THE END OF THE TURN (i.e. from curr_ctx). Carried
    # alongside the deltas so the encoder can expose the full HP trajectory
    # to the model across the history window without forcing the transformer
    # to inverse-cumsum delta scalars across attention positions. In slot_map
    # order; zeros for unrevealed slots.
    our_hp_after: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    opp_hp_after: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))

    # HP delta on the named target species of each side's damaging move this
    # turn. our_target_hp_delta = opp's damaging move's target (i.e. our mon
    # that got hit); opp_target_hp_delta = our damaging move's target. None
    # when the side didn't use a damaging move or the target slot can't be
    # resolved. Pairs with the actor/target species IDs in the encoder's
    # slot — gives the model "how hard the named target got hit."
    our_target_hp_delta: Optional[float] = None
    opp_target_hp_delta: Optional[float] = None

    # Per-side move outcome for the turn, one of "hit" / "miss" / "fail", or
    # None when the side switched, was prevented from moving (|cant| — covered
    # by the separate cant one-hot), or used no identifiable move. "hit" means
    # the move connected / did its thing (damage or status applied); "miss" =
    # accuracy miss (|-miss|); "fail" = the move executed but did nothing
    # (Protect on repeat, Substitute on existing sub, |-fail|/|-notarget|/
    # |-nothing|). crit is orthogonal — a "hit" may also crit.
    our_move_outcome: Optional[str] = None
    opp_move_outcome: Optional[str] = None
    our_move_crit: bool = False
    opp_move_crit: bool = False

    # --- multi-KO + cause ------------------------------------------------
    # Count of mons that fainted on each side in this decision window.
    # (we_fainted / opp_fainted are kept as quick bool checks; these carry
    # the exact count for multi-KO turns.)
    our_faint_count: int = 0
    opp_faint_count: int = 0

    # Multi-hot over FAINT_CAUSE_VOCAB (8 dims). A turn with two faints of
    # different causes will have two bits set. Zeros when no mon fainted.
    # Shape: (FAINT_CAUSE_DIM,) float32.
    our_faint_causes: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float32)
    )
    opp_faint_causes: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float32)
    )

    # --- attempted action ------------------------------------------------
    # The move WE pressed, preserved even when it never fired (cant / frozen /
    # KO-before-acting). Decoded from the raw action index against prev_ctx at
    # build time, so it always reflects genuine intent. Only the MOVE is kept:
    # a pressed switch always executes (switches aren't subject to freeze/sleep/
    # flinch/cant and the mask gates legality), so an attempted_switch_to would
    # always equal our_switch_to. Opp attempted action is not observable.
    our_attempted_move_id: Optional[str] = None

    # --- Status transitions THIS window (folded from the event log) ------
    # The status each side's active GAINED (status_applied) or LOST
    # (status_cured) this decision window — the *event*, distinct from the
    # current-status snapshot in the per-mon block. Lets the history window
    # carry temporal patterns the snapshot can't ("Tyranitar's Toxic was
    # cured, then it Dragon Danced" — Lum-Berry-enabled setup). None when no
    # status change on that side. Stored as poke-env Status enums (the encoder
    # one-hots them). The CAUSE (item vs ability vs natural) is NOT stored
    # here — it lives in the per-mon item/ability block (single source of
    # truth); this is purely the transition.
    our_status_applied: Optional["Status"] = None
    our_status_cured: Optional["Status"] = None
    opp_status_applied: Optional["Status"] = None
    opp_status_cured: Optional["Status"] = None

    # --- Item consumed/removed THIS window (folded from |-enditem|) -------
    # The item id each side's active LOST this window (Berry eaten, Knock Off,
    # Trick). Stored as the id string (reward/replay can use the identity); the
    # ENCODER emits only a BIT ("an item was used") — the WHICH lives in the
    # per-mon item block ([item_id, known, consumed=1]), so putting the id here
    # too would duplicate it. The history just marks the resource event +
    # timing; the per-mon block names it. Parity with ability_activated. None
    # when no item was lost this window.
    our_item_lost: Optional[str] = None
    opp_item_lost: Optional[str] = None

    # --- Trapping: rejected switch (gen3_trapping_signals_v1) ------------
    # ``attempted_switch_rejected``: the server REFUSED a switch we chose this window
    # (|error|[Unavailable choice]) — we tried to pivot and were trapped (Arena Trap /
    # Shadow Tag / Magnet Pull / Mean Look). Folded from the out-of-band CHOICE_REJECTED
    # event (TurnView.ours.attempted_rejected). The "switches always execute" assumption that
    # dropped attempted_switch_to is exactly false here. ``attempted_switch_to``: the species
    # we PRESSED a switch to (intent), decoded from the action index — parity with
    # ``our_attempted_move_id``; set whenever a switch was attempted, so on a rejected pivot it
    # names the mon we tried to bring in (our_switch_to is None — the switch never happened).
    # Only OUR side: the opponent's attempted action is not observable.
    attempted_switch_rejected: bool = False
    attempted_switch_to: Optional[str] = None

    @property
    def opp_resolved_move_id(self) -> Optional[str]:
        """Opp's move id with protocol-truth preference.

        Returns `opp_damaging_event.move_id` when the event is set (the
        |move| line of the just-resolved turn, captured before any |faint|
        or active-slot reshuffle). Falls back to `opp_move_id` for
        non-damaging moves (status, Roar, Calm Mind, BP, switches) where no
        event ever promotes.

        Attribution-sensitive callers (reward shaping, replay recording)
        should use this instead of raw `opp_move_id` — the latter is
        vulnerable to stale `last_move` reads when opp's mon switches
        between snapshots.
        """
        if self.opp_damaging_event is not None:
            return self.opp_damaging_event.move_id
        return self.opp_move_id

    @classmethod
    def build_from_events(
        cls,
        prev_ctx: "BattleContext",
        curr_ctx: "BattleContext",
        action: int,
        events: list,
    ) -> "TurnDelta":
        """Build TurnDelta by folding the event log (the primary path).

        ``events`` must be ``battle.events_since(cursor)`` — the per-decision
        window (NOT ``events_for_turn``, which slices by protocol turn).
        """
        from agents.battle.battle_event import OURS, OPP  # noqa: F401
        from agents.battle.turn_view import (
            TurnView, FAINT_CAUSE_DIM, FAINT_CAUSE_VOCAB,
        )
        from agents.enums import Status
        _cause_idx = {c: i for i, c in enumerate(FAINT_CAUSE_VOCAB)}

        # --- Attempted action (decoded before anything fires) ---
        # Both the attempted MOVE and the attempted SWITCH are captured as INTENT, so they
        # survive when the action never executes. A pressed move can fail to fire
        # (freeze/sleep/flinch/cant/KO-before-act); a pressed switch can be REFUSED by the
        # server (trapped → |error|[Unavailable choice]) — the case that makes
        # attempted_switch_to worth keeping (the "switches always execute" assumption is
        # false here). On a successful switch attempted_switch_to == our_switch_to (redundant,
        # parity with attempted_move == move_id on a normal move).
        if action < 6:
            our_attempted_move_id: Optional[str] = None  # a switch — no attempted move
            our_attempted_switch_to: Optional[str] = (
                prev_ctx.our_team_order[action]
                if action < len(prev_ctx.our_team_order) else None
            )
        elif action < 10:
            slot = action - 6
            ids = prev_ctx.active_move_ids
            our_attempted_move_id = ids[slot] if slot < len(ids) else None
            our_attempted_switch_to = None
        else:
            our_attempted_move_id = "struggle"
            our_attempted_switch_to = None

        # Per-slot HP deltas — FOLDED from the event log's DAMAGE/HEAL/SETHP + FAINT events
        # (HP-complete; no poke-env Pokémon read), bit-identical to ``curr_hp − prev_hp``.
        our_hp_delta, opp_hp_delta = _fold_hp_deltas(
            events, curr_ctx.our_slot_map, curr_ctx.opp_slot_map,
            prev_ctx.our_hp, prev_ctx.opp_hp, prev_ctx.opp_slot_map,
        )

        empty_faint_causes = np.zeros(FAINT_CAUSE_DIM, dtype=np.float32)

        if not events:
            # No event window: standalone / no-Gen3Battle callers, or a genuinely empty
            # decision window. The event-fold above already yields all-zero HP here (no
            # events ⇒ no change), but fall back to the current-board snapshot diff so the
            # crafted-context unit tests (which inject HP with no event log) still surface
            # their injected delta. In a real battle an empty window has no HP change, so
            # this is identical to the fold's zeros — it is NOT a per-turn diff heuristic.
            our_hp_delta = curr_ctx.our_hp - prev_ctx.our_hp
            opp_hp_delta = curr_ctx.opp_hp - prev_ctx.opp_hp
            for species, slot in curr_ctx.opp_slot_map.items():
                if species not in prev_ctx.opp_slot_map and opp_hp_delta[slot] > 0:
                    opp_hp_delta[slot] = 0.0
            we_fainted = curr_ctx.our_fainted_count > prev_ctx.our_fainted_count
            opp_fainted = curr_ctx.opp_fainted_count > prev_ctx.opp_fainted_count
            return cls(
                our_move_id=None, our_switch_to=None,
                our_prev_active=prev_ctx.our_active,
                opp_move_id=None, opp_switch_to=None,
                opp_prev_active=prev_ctx.opp_active,
                opp_move_known=False,
                our_hp_delta=our_hp_delta, opp_hp_delta=opp_hp_delta,
                we_fainted=we_fainted, opp_fainted=opp_fainted,
                our_failed_to_move=False, our_cant_reason=None,
                opp_failed_to_move=False, opp_cant_reason=None,
                our_boost_delta=np.zeros(BOOST_DIM, dtype=np.int8),
                opp_boost_delta=np.zeros(BOOST_DIM, dtype=np.int8),
                our_effectiveness=None, opp_effectiveness=None,
                we_moved_first=None,
                phase_is_forced_switch=(curr_ctx.phase == "forced_switch"),
                our_hp_after=curr_ctx.our_hp.copy(),
                opp_hp_after=curr_ctx.opp_hp.copy(),
                our_faint_count=int(we_fainted),
                opp_faint_count=int(opp_fainted),
                our_faint_causes=empty_faint_causes.copy(),
                opp_faint_causes=empty_faint_causes.copy(),
                our_attempted_move_id=our_attempted_move_id,
                attempted_switch_rejected=False,  # no events ⇒ no rejection this window
                attempted_switch_to=our_attempted_switch_to,
            )

        view = TurnView.from_events(events)
        our = view.ours
        opp = view.opp

        # --- Move / switch from event log (delegation-aware) ---
        our_move_id = our.move_id
        our_switch_to = our.switched_to if our.switched else None
        opp_move_id = opp.move_id
        opp_switch_to = opp.switched_to if opp.switched else None
        opp_move_known = opp.moved or opp.switched

        # --- Cant reasons ---
        our_cant_reason = our.cant_reason
        opp_cant_reason = opp.cant_reason
        our_failed_to_move = our_cant_reason is not None
        opp_failed_to_move = opp_cant_reason is not None

        # --- Status transitions (protocol id string → Status enum) ---
        def _status(s):
            return Status.__members__.get(s.upper()) if s else None
        our_status_applied = _status(our.status_applied)
        our_status_cured = _status(our.status_cured)
        opp_status_applied = _status(opp.status_applied)
        opp_status_cured = _status(opp.status_cured)

        # --- Item consumed/removed this window (id kept; encoder emits a bit) ---
        our_item_lost = our.item_lost
        opp_item_lost = opp.item_lost

        # --- Boost deltas (current-board LiveView stage diff, NOT an event-sum) ---
        # FINDING: boost deltas cannot be value-identically folded from the event log.
        # BOOST/UNBOOST carry a signed amount, but the realized stage clamps at ±6 (a +2
        # Swords Dance at +5 nets +1, not +2) and CLEARBOOST/-invertboost/-copyboost/
        # -swapboost/Belly-Drum SETBOOST carry only an ``op`` (no realized amount) — the
        # log simply does not record the post-op stage values. So the exact, lossless
        # source for the net stage change is the difference of the active mon's clamped
        # stages at the two decision endpoints, read from the LiveView-equivalent
        # snapshot boosts. Zeroed on switch (the switch-in's stages are its own baseline).
        our_boost_delta = (
            np.zeros(BOOST_DIM, dtype=np.int8) if our_switch_to is not None
            else (curr_ctx.our_boosts - prev_ctx.our_boosts).astype(np.int8)
        )
        opp_boost_delta = (
            np.zeros(BOOST_DIM, dtype=np.int8) if opp_switch_to is not None
            else (curr_ctx.opp_boosts - prev_ctx.opp_boosts).astype(np.int8)
        )

        # --- Damaging events — convert TurnView.DamagingMove → DamagingMoveEvent ---
        def _to_dme(dm):
            if dm is None:
                return None
            ts_str = dm.target_status
            target_status = Status.__members__.get(ts_str) if ts_str else None
            return DamagingMoveEvent(
                user_species=dm.user_species or "",
                target_species=dm.target_species or "",
                target_status=target_status,
                move_id=dm.move_id or "",
                effectiveness=dm.effectiveness if dm.effectiveness is not None else 1.0,
            )

        our_damaging_event = _to_dme(our.damaging_move)
        opp_damaging_event = _to_dme(opp.damaging_move)

        # --- Faint counts + causes ---
        faints = view.faint_details()
        our_faint_list = [f for f in faints if f.side == OURS]
        opp_faint_list = [f for f in faints if f.side == OPP]
        our_faint_count = len(our_faint_list)
        opp_faint_count = len(opp_faint_list)

        # Direct index (KeyError on miss) — every faint HAS a cause (no None
        # sentinel), so a cause outside the vocab is a true silent drop. Both
        # _classify_faint_cause and _cause_idx derive from FAINT_CAUSE_VOCAB, so
        # this can only fire if a future edit desyncs them — which we WANT to crash.
        our_faint_causes_arr = np.zeros(FAINT_CAUSE_DIM, dtype=np.float32)
        for f in our_faint_list:
            our_faint_causes_arr[_cause_idx[f.cause]] = 1.0

        opp_faint_causes_arr = np.zeros(FAINT_CAUSE_DIM, dtype=np.float32)
        for f in opp_faint_list:
            opp_faint_causes_arr[_cause_idx[f.cause]] = 1.0

        # --- Target HP deltas ---
        our_target_hp_delta = _resolve_target_hp_delta(
            opp_damaging_event, our_hp_delta, curr_ctx.our_slot_map
        )
        opp_target_hp_delta = _resolve_target_hp_delta(
            our_damaging_event, opp_hp_delta, curr_ctx.opp_slot_map
        )

        return cls(
            our_move_id=our_move_id,
            our_switch_to=our_switch_to,
            our_prev_active=prev_ctx.our_active,
            opp_move_id=opp_move_id,
            opp_switch_to=opp_switch_to,
            opp_prev_active=prev_ctx.opp_active,
            opp_move_known=opp_move_known,
            our_hp_delta=our_hp_delta,
            opp_hp_delta=opp_hp_delta,
            we_fainted=our_faint_count > 0,
            opp_fainted=opp_faint_count > 0,
            our_failed_to_move=our_failed_to_move,
            our_cant_reason=our_cant_reason,
            opp_failed_to_move=opp_failed_to_move,
            opp_cant_reason=opp_cant_reason,
            our_boost_delta=our_boost_delta,
            opp_boost_delta=opp_boost_delta,
            our_effectiveness=our.effectiveness,
            opp_effectiveness=opp.effectiveness,
            we_moved_first=view.we_moved_first,
            our_damaging_event=our_damaging_event,
            opp_damaging_event=opp_damaging_event,
            phase_is_forced_switch=(curr_ctx.phase == "forced_switch"),
            our_hp_after=curr_ctx.our_hp.copy(),
            opp_hp_after=curr_ctx.opp_hp.copy(),
            our_target_hp_delta=our_target_hp_delta,
            opp_target_hp_delta=opp_target_hp_delta,
            our_move_outcome=our.outcome,
            opp_move_outcome=opp.outcome,
            our_move_crit=our.crit,
            opp_move_crit=opp.crit,
            our_faint_count=our_faint_count,
            opp_faint_count=opp_faint_count,
            our_faint_causes=our_faint_causes_arr,
            opp_faint_causes=opp_faint_causes_arr,
            our_attempted_move_id=our_attempted_move_id,
            our_status_applied=our_status_applied,
            our_status_cured=our_status_cured,
            opp_status_applied=opp_status_applied,
            opp_status_cured=opp_status_cured,
            our_item_lost=our_item_lost,
            opp_item_lost=opp_item_lost,
            attempted_switch_rejected=view.ours.attempted_rejected,
            attempted_switch_to=our_attempted_switch_to,
        )

    @classmethod
    def empty(cls) -> "TurnDelta":
        return cls(
            our_move_id=None, our_switch_to=None, our_prev_active="NULL",
            opp_move_id=None, opp_switch_to=None, opp_prev_active="NULL",
            opp_move_known=False,
            our_hp_delta=np.zeros(6, dtype=np.float32),
            opp_hp_delta=np.zeros(6, dtype=np.float32),
            we_fainted=False, opp_fainted=False,
            our_failed_to_move=False, our_cant_reason=None,
            opp_failed_to_move=False, opp_cant_reason=None,
            our_boost_delta=np.zeros(BOOST_DIM, dtype=np.int8),
            opp_boost_delta=np.zeros(BOOST_DIM, dtype=np.int8),
            our_effectiveness=None,
            opp_effectiveness=None,
            we_moved_first=None,
            our_damaging_event=None,
            opp_damaging_event=None,
            phase_is_forced_switch=False,
            our_hp_after=np.zeros(6, dtype=np.float32),
            opp_hp_after=np.zeros(6, dtype=np.float32),
            our_target_hp_delta=None,
            opp_target_hp_delta=None,
            our_move_outcome=None,
            opp_move_outcome=None,
            our_move_crit=False,
            opp_move_crit=False,
        )
