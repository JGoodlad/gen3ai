from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np

from poke_env.battle.abstract_battle import DamagingMoveEvent

from agents.training.slot_registry import SlotRegistry
from agents.gen3_mechanics import PHAZING_MOVES, BOOST_DIM, boosts_array


# Moves whose user always faints and which always connect when used (a neutral
# hit emits no effectiveness event, so the damaging-event "connected" signal would
# miss them; an immune target DOES emit, so it's covered by the event either way).
SELF_KO_MOVES = frozenset({"explosion", "selfdestruct"})


def _moves_match(a: Optional[str], b: Optional[str]) -> bool:
    """True if two move ids refer to the same move. Treats all Hidden Power
    variants ('hiddenpower', 'hiddenpowerfire', …) as one move."""
    if a == b:
        return True
    return bool(a and b and a.startswith("hiddenpower") and b.startswith("hiddenpower"))


def _align_effectiveness(move_id, effectiveness, event):
    """Keep a side's (effectiveness, damaging_event) only when they describe the
    SAME move we recorded as having fired. They are turn-gated independently of
    move_id and on same-turn forced-switch / faint double-records can lag at a
    different move — feeding the wrong effectiveness/target would corrupt the obs
    one-hot, the actor/target attribution, and the immune reward term. On
    disagreement we drop to "unknown" (None) rather than serve wrong data.
    Applied identically for our and opp sides."""
    if (event is not None and move_id is not None
            and not _moves_match(event.move_id, move_id)):
        return None, None
    return effectiveness, event


def _ko_before_acting(*, fainted, switched_voluntarily, move_resolved,
                      other_side_moved_first, cant_reason) -> bool:
    """A side was KO'd BEFORE it could act: it fainted this turn, did not choose
    to switch, no move of its own resolved (no damaging event / miss / fail), and
    the OTHER side moved first (Gen 3: mover lands the KO before the slower mon's
    turn). When True the side did NOTHING — its move_id must be None and its
    "didn't move" reason is 'fainted'. Symmetric for both sides; only the
    per-side inputs differ (our faint splits across two turns so we never
    voluntarily switch on it; an opp faint folds the forced replacement into the
    same delta, so its other_side_moved_first is our we_moved_first)."""
    return bool(
        fainted and not switched_voluntarily and not move_resolved
        and other_side_moved_first is True and cant_reason is None
    )


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


def _derive_move_outcome(
    move_used: bool,
    missed: bool,
    failed: bool,
    suppressed: bool,
    connected: bool = False,
) -> Optional[str]:
    """Collapse the per-side protocol outcome flags into one category.

    Returns "miss" / "fail" / "hit", or None when no move resolved.

    `suppressed` is True when the side was prevented from acting by a |cant|
    (sleep/par/flinch/etc.) — no move resolved, so the outcome is None (the cant
    reason is carried separately).

    The outcome describes a MOVE, so it is None unless a move was actually used
    (`move_used`). The miss/fail flags are turn-gated and can leak onto a turn the
    move did NOT have them — a no-move turn (a switch on the same game turn as an
    earlier sub-turn move) or a self-faint move like Explosion. So:
      - gate on `move_used` (a switch is never "miss"/"fail"), and
      - `connected` (a damaging event resolved this turn) means the move DEALT
        damage, i.e. it landed — that overrides a stale miss/fail flag (Explosion
        can't both deal damage and "miss").
    Precedence among real outcomes is miss > fail > hit; the protocol never emits
    more than one for a single move.
    """
    if suppressed or not move_used:
        return None
    if connected:
        return "hit"
    if missed:
        return "miss"
    if failed:
        return "fail"
    return "hit"


@dataclass
class BattleContext:
    """
    Frozen per-turn snapshot of agent-relevant battle state.

    Built once per turn in EpisodeTracker.record() and stored as self._last_ctx on the env.
    Gives the reward function, action mapper, and callbacks a single stable source
    of truth for what the model saw when it chose its action.
    """
    turn: int
    phase: Literal["move_selection", "forced_switch"]
    mask: np.ndarray            # (11,) int8 — valid actions this turn
    our_slot_map: dict[str, int]   # species -> stable slot index 0-5
    opp_slot_map: dict[str, int]   # species -> stable slot index 0-5

    # Per-slot HP fractions (in slot_map order, zeros for unrevealed slots)
    our_hp: np.ndarray          # (6,) float32
    opp_hp: np.ndarray          # (6,) float32

    # Active Pokémon species at the moment this context was built
    our_active: str             # "NONE" if no active mon (between faints)
    opp_active: str             # "NONE" if not yet revealed

    # Fainted counts — used to detect new faints via delta
    our_fainted_count: int
    opp_fainted_count: int

    # Move IDs in request-slot order (len 4, None for empty/missing slots).
    # Used by TurnDelta.build() to resolve which move action indices 6-9 map to.
    active_move_ids: list       # list[str | None]

    # The move the opponent's active Pokémon used on the PREVIOUS turn, sourced
    # from battle.opponent_active_pokemon.last_move (poke-env parses the Showdown
    # |-move| protocol and sets Move._is_last_used on the correct move object).
    # None if the opponent hasn't moved yet, switched in this snapshot's turn,
    # or the move cannot be identified (e.g. Explosion aftermath).
    #
    # Known behaviors (confirmed by transition_fuzz_test.py):
    #   - Sleep Talk: when delegation succeeds, stores the delegated move (e.g. "surf").
    #     When delegation FAILS (all PP depleted), stores "sleeptalk" directly.
    #   - Recharge turns (Hyper Beam): persists as "hyperbeam" on the |cant|recharge turn.
    #   - |cant| turns (par, flinch, frz, sleep-no-talk): persists from the prior turn
    #     IF the mon has already used a move; None if first active turn.
    opp_last_move_id: str | None

    # last_move snapshot for every revealed opponent Pokémon (not just the active one).
    # Used by TurnDelta.build() to recover the phazed mon's move when Roar/Whirlwind fires:
    # poke-env swaps the active slot to the new mon before we snapshot, so
    # opp_last_move_id (which reads from the NEW active mon) would be None. The phazed mon
    # still has last_move set in battle.opponent_team, captured here.
    opp_all_last_move_ids: dict  # dict[str, str | None] — species → last_move_id

    # All move IDs the opponent's current active Pokémon has revealed so far.
    # Kept for potential use by the observation encoder and diagnostics.
    opp_active_revealed_moves: frozenset  # frozenset[str]

    # Cant-move reason for each side's active Pokémon, snapshotted after the turn resolves.
    # Set by |cant| messages (paralysis, flinch, freeze, sleep-no-sleep-talk, confusion).
    # Cleared by |move| messages (Pokemon.moved() resets _last_cant_reason to None).
    # None means the mon moved normally this turn (or is a fresh switch-in with no history).
    our_cant_reason: str | None
    opp_cant_reason: str | None

    # Stat stages for each side's active Pokémon in BOOST_STATS order
    # (atk/def/spa/spd/spe/accuracy/evasion).  All zeros if mon is None or not yet revealed.
    our_boosts: np.ndarray   # (7,) int8
    opp_boosts: np.ndarray   # (7,) int8

    # Effectiveness of the last damaging move used by each side (from the turn that
    # just ended).  0.0=immune, 0.5=resisted, 1.0=neutral, 2.0=super-effective.
    # None when the side switched, used a non-damaging move, or the battle just started.
    # Sourced from AbstractBattle.our/opp_last_effectiveness, which gates on turn-1.
    our_last_effectiveness: float | None
    opp_last_effectiveness: float | None

    # True = we executed our action before the opponent in the turn that just ended.
    # None when one or both sides performed a normal switch (no move competition).
    we_moved_first: bool | None

    # Per-side set of fainted species — used to identify which specific mon
    # newly fainted between two contexts (the count alone is ambiguous, e.g. for
    # resolving the actual target of opponent Hidden Power when our side switched
    # and the switch-in fainted in the same turn).
    our_fainted_species: frozenset = field(default_factory=frozenset)  # frozenset[str]
    opp_fainted_species: frozenset = field(default_factory=frozenset)  # frozenset[str]

    # Per-mon status at this context's snapshot point. Used to evaluate Gen 3
    # ability quirks that depend on status — currently the Flash Fire-vs-frozen
    # interaction (a frozen target's Flash Fire does NOT block incoming Fire moves,
    # and the move itself thaws the target). Without a historical snapshot we'd
    # only see the post-thaw status and mis-evaluate the just-fired move.
    our_team_status: dict = field(default_factory=dict)  # dict[str, Status | None]

    # Species in battle.team iteration order — the same order the action mapper
    # uses to interpret switch actions 0–5. Captured at snapshot time so a later
    # TurnDelta can recover the INTENT of a switch action (which slot we picked)
    # even if the switch-in later dies and forced-replacements cycle in more
    # mons by the time the next snapshot is built. Without this, our_switch_to
    # would point at the end-of-chain active mon rather than the mon we sent in.
    our_team_order: tuple = field(default_factory=tuple)  # tuple[str, ...]

    # Full per-side record of the last damaging move resolved last turn —
    # carries user_species, target_species, target_status (at move-fire time),
    # move_id, and effectiveness. Lets callers attribute moves directly without
    # inferring "who fired / who got hit" from before/after snapshot diffs.
    # None when the side didn't use a damaging move last turn. Mirrors
    # our/opp_last_effectiveness (set together at the same protocol events).
    our_last_damaging_event: Optional[DamagingMoveEvent] = None
    opp_last_damaging_event: Optional[DamagingMoveEvent] = None

    # The move OUR active Pokémon actually used on the turn that just ended,
    # from battle.active_pokemon.last_move — the mirror of opp_last_move_id.
    # Protocol-truth and DELEGATION-AWARE: Sleep Talk stores the called move
    # (e.g. "surf"), not "sleeptalk" (see opp_last_move_id notes). Used by
    # TurnDelta.build() to set our_move_id from the protocol instead of the
    # action index — immune to the action-bookkeeping desync on faint /
    # forced-switch turns, and captures delegated moves first-class.
    our_last_move_id: Optional[str] = None

    # Per-side move outcome flags for the turn that just ended, sourced from
    # AbstractBattle.our/opp_move_{crit,missed,failed} (each turn-gated on
    # turn-1). crit is orthogonal to miss/fail (a hit can crit); missed and
    # failed are mutually exclusive in practice. All False when the side
    # switched, was prevented from moving (|cant|), or used a move that simply
    # connected. Derived into a single outcome category by TurnDelta.build.
    our_move_crit: bool = False
    opp_move_crit: bool = False
    our_move_missed: bool = False
    opp_move_missed: bool = False
    our_move_failed: bool = False
    opp_move_failed: bool = False

    def __post_init__(self):
        if self.mask.shape != (11,):
            raise RuntimeError(
                f"BattleContext mask shape {self.mask.shape} != (11,) at turn {self.turn}"
            )
        if len(self.active_move_ids) != 4:
            raise RuntimeError(
                f"BattleContext active_move_ids length {len(self.active_move_ids)} != 4 at turn {self.turn}"
            )
        if self.our_boosts.shape != (BOOST_DIM,):
            raise RuntimeError(
                f"BattleContext our_boosts shape {self.our_boosts.shape} != ({BOOST_DIM},) at turn {self.turn}"
            )
        if self.opp_boosts.shape != (BOOST_DIM,):
            raise RuntimeError(
                f"BattleContext opp_boosts shape {self.opp_boosts.shape} != ({BOOST_DIM},) at turn {self.turn}"
            )

    @classmethod
    def from_battle(
        cls,
        battle,
        mask: np.ndarray,
        our_slots: SlotRegistry,
        opp_slots: SlotRegistry,
    ) -> BattleContext:
        """Build a context snapshot from a live battle, updating slot registries in place."""
        for mon in battle.team.values():
            our_slots.assign(mon.species)
        for mon in battle.opponent_team.values():
            opp_slots.assign(mon.species)

        our_hp = np.zeros(6, dtype=np.float32)
        for mon in battle.team.values():
            slot = our_slots.get(mon.species)
            if slot is not None:
                our_hp[slot] = mon.current_hp_fraction

        opp_hp = np.zeros(6, dtype=np.float32)
        for mon in battle.opponent_team.values():
            slot = opp_slots.get(mon.species)
            if slot is not None:
                opp_hp[slot] = mon.current_hp_fraction

        our_active = (
            battle.active_pokemon.species
            if battle.active_pokemon and not battle.active_pokemon.fainted
            else "NONE"
        )
        opp_mon = battle.opponent_active_pokemon
        opp_active = opp_mon.species if opp_mon else "NONE"

        # Build active_move_ids: 4-element list mirroring the masker's move-slot assignment.
        # Prefer _gen3_decision_context (latched by the masker) to guarantee slot ordering
        # is identical to what the action mask was built from.
        dec_ctx = getattr(battle, "_gen3_decision_context", None)
        if dec_ctx and dec_ctx.get("turn") == battle.turn:
            raw_ids = dec_ctx.get("move_ids", [])
            active_move_ids = (list(raw_ids) + [None, None, None, None])[:4]
        else:
            active_move_ids: list = [None, None, None, None]
            try:
                active_request = battle.last_request.get("active", [{}])[0]
                request_moves = active_request.get("moves", [])
                for i, move_data in enumerate(request_moves):
                    if i < 4 and move_data.get("id") != "struggle":
                        if not move_data.get("disabled", False):
                            active_move_ids[i] = move_data.get("id")
            except (AttributeError, IndexError, TypeError):
                pass

        opp_last_move = opp_mon.last_move if opp_mon else None
        opp_last_move_id = opp_last_move.id if opp_last_move else None
        our_active_mon = battle.active_pokemon
        our_last_move = our_active_mon.last_move if our_active_mon else None
        our_last_move_id = our_last_move.id if our_last_move else None
        opp_active_revealed_moves = frozenset(opp_mon.moves.keys() if opp_mon else [])
        opp_all_last_move_ids: dict = {}
        for mon in battle.opponent_team.values():
            lm = mon.last_move
            opp_all_last_move_ids[mon.species] = lm.id if lm else None
        our_cant_reason = (
            battle.active_pokemon.last_cant_reason
            if battle.active_pokemon else None
        )
        opp_cant_reason = opp_mon.last_cant_reason if opp_mon else None

        our_boosts = boosts_array(battle.active_pokemon)
        opp_boosts = boosts_array(opp_mon)

        return cls(
            turn=battle.turn,
            phase="forced_switch" if battle.force_switch else "move_selection",
            mask=mask,
            our_slot_map=our_slots.snapshot(),
            opp_slot_map=opp_slots.snapshot(),
            our_hp=our_hp,
            opp_hp=opp_hp,
            our_active=our_active,
            opp_active=opp_active,
            our_fainted_count=sum(1 for m in battle.team.values() if m.fainted),
            opp_fainted_count=sum(1 for m in battle.opponent_team.values() if m.fainted),
            our_fainted_species=frozenset(m.species for m in battle.team.values() if m.fainted),
            opp_fainted_species=frozenset(m.species for m in battle.opponent_team.values() if m.fainted),
            our_team_status={m.species: m.status for m in battle.team.values()},
            active_move_ids=active_move_ids,
            opp_last_move_id=opp_last_move_id,
            opp_all_last_move_ids=opp_all_last_move_ids,
            opp_active_revealed_moves=opp_active_revealed_moves,
            our_cant_reason=our_cant_reason,
            opp_cant_reason=opp_cant_reason,
            our_boosts=our_boosts,
            opp_boosts=opp_boosts,
            our_last_effectiveness=battle.our_last_effectiveness,
            opp_last_effectiveness=battle.opp_last_effectiveness,
            our_last_damaging_event=battle.our_last_damaging_move,
            opp_last_damaging_event=battle.opp_last_damaging_move,
            our_last_move_id=our_last_move_id,
            we_moved_first=battle.we_moved_first,
            our_team_order=tuple(m.species for m in battle.team.values()),
            our_move_crit=battle.our_move_crit,
            opp_move_crit=battle.opp_move_crit,
            our_move_missed=battle.our_move_missed,
            opp_move_missed=battle.opp_move_missed,
            our_move_failed=battle.our_move_failed,
            opp_move_failed=battle.opp_move_failed,
        )


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
    # |-nothing|). Derived in build() from the move/switch/cant context plus
    # the curr_ctx outcome flags. crit is orthogonal — a "hit" may also crit.
    our_move_outcome: Optional[str] = None
    opp_move_outcome: Optional[str] = None
    our_move_crit: bool = False
    opp_move_crit: bool = False

    # --- Step 4: multi-KO + cause ----------------------------------------
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

    # --- Step 4: attempted action ----------------------------------------
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
    def build(cls, prev_ctx: BattleContext, curr_ctx: BattleContext, action: int) -> TurnDelta:
        our_hp_delta = curr_ctx.our_hp - prev_ctx.our_hp
        opp_hp_delta = curr_ctx.opp_hp - prev_ctx.opp_hp

        # When an opponent mon is revealed for the first time (not in prev slot_map),
        # its HP slot transitions from 0 (unrevealed default) to its actual value.
        # This looks like the opponent "gained" HP, but we didn't cause that — we just
        # learned about it. Zero out the delta for newly-revealed slots to prevent false
        # hp_opp penalties in compute_base_reward.
        for species, slot in curr_ctx.opp_slot_map.items():
            if species not in prev_ctx.opp_slot_map and opp_hp_delta[slot] > 0:
                opp_hp_delta[slot] = 0.0

        we_fainted = curr_ctx.our_fainted_count > prev_ctx.our_fainted_count
        opp_fainted = curr_ctx.opp_fainted_count > prev_ctx.opp_fainted_count

        # --- Our action ---
        if action < 6:
            # The action's INTENT — which team-slot species we sent in — is the
            # only correct answer here. Reading curr_ctx.our_active instead
            # gets bitten when the switch-in dies and forced-replacements cycle
            # more mons before the next snapshot (our_active ends up pointing
            # at the final replacement, not the mon we switched to).
            our_switch_to = (
                prev_ctx.our_team_order[action]
                if action < len(prev_ctx.our_team_order)
                else None
            )
            our_move_id = None
        elif action < 10:
            our_switch_to = None
            slot = action - 6
            ids = prev_ctx.active_move_ids
            our_move_id = ids[slot] if slot < len(ids) else None
        else:
            # action == 10: Struggle
            our_switch_to = None
            our_move_id = "struggle"

        # --- Protocol-truth override for our_move_id ---
        # The action-derived id above is vulnerable to the action-bookkeeping
        # desync (_last_action can be a different turn's action on faint /
        # forced-switch cadence — confirmed mislabeling moves in training). The
        # protocol is authoritative for the move OUR mon actually used; we take it
        # from two turn-gated sources depending on whether the mon survived:
        we_stayed_in = (
            prev_ctx.our_active == curr_ctx.our_active
            and curr_ctx.our_active != "NONE"
        )
        if curr_ctx.our_cant_reason is None:
            if (our_switch_to is None and we_stayed_in
                    and curr_ctx.our_last_move_id is not None):
                # Survived: active_pokemon.last_move is the move we used — fresh
                # and DELEGATION-AWARE (Sleep Talk / Metronome store the CALLED
                # move, e.g. "surf"), so the delegated move becomes first-class.
                our_move_id = curr_ctx.our_last_move_id
            elif we_fainted and curr_ctx.our_last_damaging_event is not None:
                # Used a (damaging) move and fainted THIS turn — active_pokemon now
                # reads the replacement (last_move wrong) and the action is desynced.
                # The turn-gated DamagingMoveEvent names the move the fainted mon
                # actually used. This also corrects a move↔switch misclassification
                # when the desynced action looked like a switch.
                our_move_id = curr_ctx.our_last_damaging_event.move_id
                our_switch_to = None

        # --- Protocol-accuracy guards (shared with the opp side below) ---
        # The action-derived id would misrepresent a KO'd-before-acting turn as
        # "used the clicked move" (and the outcome as "hit"). _ko_before_acting
        # reports the truth: nothing fired (move_id None, reason "fainted").
        our_ko = _ko_before_acting(
            fainted=we_fainted,
            switched_voluntarily=our_switch_to is not None,
            move_resolved=(curr_ctx.our_last_damaging_event is not None
                           or curr_ctx.our_move_missed or curr_ctx.our_move_failed),
            other_side_moved_first=(curr_ctx.we_moved_first is not True),  # opp first
            cant_reason=curr_ctx.our_cant_reason,
        )
        if our_ko:
            our_move_id = None
        our_cant_reason = curr_ctx.our_cant_reason  # KO'd-before-acting → None, not "fainted"
        our_effectiveness, our_damaging_event = _align_effectiveness(
            our_move_id, curr_ctx.our_last_effectiveness, curr_ctx.our_last_damaging_event,
        )

        # --- Opponent action ---
        # opp_last_move_id in curr_ctx was read from battle.opponent_active_pokemon.last_move
        # AFTER the turn resolved. Guard against contamination from a newly switched-in
        # Pokémon's prior-appearance last_move by checking if the opponent switched.
        opp_switched = prev_ctx.opp_active != curr_ctx.opp_active
        opp_switch_to = curr_ctx.opp_active if opp_switched and curr_ctx.opp_active != "NONE" else None

        if opp_switched:
            if our_move_id in PHAZING_MOVES:
                # Phaze case (Roar/Whirlwind): the opponent moved first (Gen 3 phazing moves
                # have -6 priority), then was forced out. opp_last_move_id reads from the NEW
                # active mon (which hasn't moved), so we recover the phazed mon's last_move
                # from the full-team snapshot in opp_all_last_move_ids.
                opp_move_id = curr_ctx.opp_all_last_move_ids.get(prev_ctx.opp_active)
                opp_move_known = opp_move_id is not None
            elif opp_fainted:
                # Forced switch after their mon fainted: they may have moved before dying.
                # Recover from the full-team snapshot (opp_last_move_id reads the NEW active
                # mon, which hasn't moved yet, so we must use the per-species snapshot).
                opp_move_id = curr_ctx.opp_all_last_move_ids.get(prev_ctx.opp_active)
                opp_move_known = True   # switch was forced (faint), whether or not we know the move
            else:
                opp_move_id = None
                opp_move_known = True   # voluntary switch — no move was used
        else:
            opp_move_id = curr_ctx.opp_last_move_id
            opp_move_known = opp_move_id is not None

        # --- Same protocol-accuracy guards, opp side ---
        # The faint-recovery above may have pulled a STALE prior move from
        # opp_all_last_move_ids when the opp was KO'd before acting; correct it.
        # (Per-side inputs differ: an opp faint folds the forced replacement into
        # this delta, so its "switched_voluntarily" is only true off a faint, and
        # "other side moved first" is OUR we_moved_first.)
        opp_ko = _ko_before_acting(
            fainted=opp_fainted,
            switched_voluntarily=(opp_switch_to is not None and not opp_fainted),
            move_resolved=(curr_ctx.opp_last_damaging_event is not None
                           or curr_ctx.opp_move_missed or curr_ctx.opp_move_failed),
            other_side_moved_first=(curr_ctx.we_moved_first is True),  # we went first
            cant_reason=curr_ctx.opp_cant_reason,
        )
        if opp_ko:
            opp_move_id = None
            opp_move_known = True  # we positively know nothing fired
        opp_cant_reason = curr_ctx.opp_cant_reason  # KO'd-before-acting → None, not "fainted"
        opp_effectiveness, opp_damaging_event = _align_effectiveness(
            opp_move_id, curr_ctx.opp_last_effectiveness, curr_ctx.opp_last_damaging_event,
        )

        our_failed_to_move = our_cant_reason is not None
        opp_failed_to_move = opp_cant_reason is not None

        # Boost deltas: meaningful when the same mon stayed in; zeroed when switched
        # (the switch-in's boosts are its own baseline, not a change from the prev mon).
        our_boost_delta = (
            np.zeros(BOOST_DIM, dtype=np.int8) if our_switch_to is not None
            else (curr_ctx.our_boosts - prev_ctx.our_boosts).astype(np.int8)
        )
        opp_boost_delta = (
            np.zeros(BOOST_DIM, dtype=np.int8) if opp_switch_to is not None
            else (curr_ctx.opp_boosts - prev_ctx.opp_boosts).astype(np.int8)
        )

        # Target-HP-delta attribution: look the named target's species up in
        # the current slot map (where it actually lives now, post-turn) so
        # newly-revealed opp mons resolve correctly. Returns None when no
        # damaging move fired or the target species isn't in the slot map.
        our_target_hp_delta = _resolve_target_hp_delta(
            opp_damaging_event, our_hp_delta, curr_ctx.our_slot_map
        )
        opp_target_hp_delta = _resolve_target_hp_delta(
            our_damaging_event, opp_hp_delta, curr_ctx.opp_slot_map
        )

        # Per-side move outcome. `suppressed` covers the |cant| case where an
        # action was selected (our_move_id set) but no move resolved. For the
        # opponent, opp_move_id is already None on a voluntary switch and is
        # recovered on a faint-forced switch, so move_used alone is the right
        # signal — no extra switch suppression needed.
        our_move_outcome = _derive_move_outcome(
            move_used=our_move_id is not None,
            missed=curr_ctx.our_move_missed,
            failed=curr_ctx.our_move_failed,
            suppressed=our_failed_to_move,
            connected=our_damaging_event is not None or our_move_id in SELF_KO_MOVES,
        )
        opp_move_outcome = _derive_move_outcome(
            move_used=opp_move_id is not None,
            missed=curr_ctx.opp_move_missed,
            failed=curr_ctx.opp_move_failed,
            suppressed=opp_failed_to_move,
            connected=opp_damaging_event is not None or opp_move_id in SELF_KO_MOVES,
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
            we_fainted=we_fainted,
            opp_fainted=opp_fainted,
            our_failed_to_move=our_failed_to_move,
            our_cant_reason=our_cant_reason,
            opp_failed_to_move=opp_failed_to_move,
            opp_cant_reason=opp_cant_reason,
            our_boost_delta=our_boost_delta,
            opp_boost_delta=opp_boost_delta,
            our_effectiveness=our_effectiveness,
            opp_effectiveness=opp_effectiveness,
            we_moved_first=curr_ctx.we_moved_first,
            our_damaging_event=our_damaging_event,
            opp_damaging_event=opp_damaging_event,
            phase_is_forced_switch=(curr_ctx.phase == "forced_switch"),
            our_hp_after=curr_ctx.our_hp.copy(),
            opp_hp_after=curr_ctx.opp_hp.copy(),
            our_target_hp_delta=our_target_hp_delta,
            opp_target_hp_delta=opp_target_hp_delta,
            our_move_outcome=our_move_outcome,
            opp_move_outcome=opp_move_outcome,
            our_move_crit=curr_ctx.our_move_crit,
            opp_move_crit=curr_ctx.opp_move_crit,
        )

    @classmethod
    def build_from_events(
        cls,
        prev_ctx: "BattleContext",
        curr_ctx: "BattleContext",
        action: int,
        events: list,
    ) -> "TurnDelta":
        """Build TurnDelta by folding the event log (Step 4 primary path).

        ``events`` must be ``battle.events_since(cursor)`` — the per-decision
        window (NOT ``events_for_turn``, which slices by protocol turn).
        """
        from agents.battle.battle_event import OURS, OPP  # noqa: F401
        from agents.battle.turn_view import (
            TurnView, FAINT_CAUSE_DIM, FAINT_CAUSE_VOCAB,
        )
        from poke_env.battle.status import Status
        _cause_idx = {c: i for i, c in enumerate(FAINT_CAUSE_VOCAB)}

        # --- Attempted move (decoded before anything fires) ---
        # Only the move is kept — a pressed switch always executes, so it would
        # always equal our_switch_to (see TurnDelta field doc). A pressed move,
        # by contrast, may never fire (freeze/sleep/flinch/cant/KO-before-act).
        if action < 6:
            our_attempted_move_id: Optional[str] = None  # a switch — no attempted move
        elif action < 10:
            slot = action - 6
            ids = prev_ctx.active_move_ids
            our_attempted_move_id = ids[slot] if slot < len(ids) else None
        else:
            our_attempted_move_id = "struggle"

        # HP deltas from snapshot (still the right source for continuous HP).
        our_hp_delta = curr_ctx.our_hp - prev_ctx.our_hp
        opp_hp_delta = curr_ctx.opp_hp - prev_ctx.opp_hp
        for species, slot in curr_ctx.opp_slot_map.items():
            if species not in prev_ctx.opp_slot_map and opp_hp_delta[slot] > 0:
                opp_hp_delta[slot] = 0.0

        empty_faint_causes = np.zeros(FAINT_CAUSE_DIM, dtype=np.float32)

        if not events:
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

        # --- Boost deltas ---
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
        )

    @classmethod
    def empty(cls) -> TurnDelta:
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
