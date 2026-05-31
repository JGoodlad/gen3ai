"""``BattleContext`` — the per-decision snapshot of agent-relevant battle state.

This is the **decision-capture seam**: built once per decision (in
``EpisodeTracker.record`` / ``BattleRecorder.record`` / ``RewardTracker``), it freezes
the current board (per-slot HP, active species, boosts, slot maps), the legality surface
(``mask`` / ``legal`` / ``active_move_ids``), and the phase the model saw when it chose
its action. Reward (``record_action``), the action mapper, the ``prev_mask`` obs feature,
the forensic recorder, and the Hidden-Power tracker all read it as a single stable source
of truth for "what the model saw".

It is NOT the history-fold: "what happened this turn" is folded from the event log by
:class:`~agents.training.turn_delta.TurnDelta` (``build_from_events``), which reads the
current-board fields of two consecutive snapshots but does **no** diff-detective
reconstruction. The snapshot's remaining poke-env reads (``opp_last_damaging_event`` for
the HP tracker; ``we_moved_first`` / ``our_move_crit`` / effectiveness, probed by the
poke-env-gap fuzz tests) are the documented seam this module owns — it is exempted in
``strict_api_lock_test`` exactly as the old ``battle_context.py`` was.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, TYPE_CHECKING
import numpy as np

from poke_env.battle.abstract_battle import DamagingMoveEvent

from agents.training.slot_registry import SlotRegistry
from agents.gen3_mechanics import BOOST_DIM, boosts_array

if TYPE_CHECKING:
    from agents.battle.live_view import LegalActions


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
    # Used by TurnDelta to resolve which move action indices 6-9 map to.
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
    # (e.g. "surf"), not "sleeptalk" (see opp_last_move_id notes).
    our_last_move_id: Optional[str] = None

    # Per-side move outcome flags for the turn that just ended, sourced from
    # AbstractBattle.our/opp_move_{crit,missed,failed} (each turn-gated on
    # turn-1). crit is orthogonal to miss/fail (a hit can crit); missed and
    # failed are mutually exclusive in practice. All False when the side
    # switched, was prevented from moving (|cant|), or used a move that simply
    # connected.
    our_move_crit: bool = False
    opp_move_crit: bool = False
    our_move_missed: bool = False
    opp_move_missed: bool = False
    our_move_failed: bool = False
    opp_move_failed: bool = False

    # The server-authoritative LegalActions snapshot captured at THIS decision — the
    # immutable per-decision legality surface the masker built the mask from. Carried so
    # the action mapper can decode the chosen action against the SAME snapshot the model
    # saw, replacing the old battle._gen3_decision_context stash. None for fallback /
    # standalone callers that didn't supply it (active_move_ids then reads last_request).
    legal: Optional["LegalActions"] = None

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
        legal: Optional["LegalActions"] = None,
    ) -> BattleContext:
        """Build a context snapshot from a live battle, updating slot registries in place.

        ``legal`` is the per-decision :class:`LegalActions` snapshot captured by the
        caller (the masker built the mask from it). When supplied it is the source of
        ``active_move_ids`` (request order, struggle excluded) AND is stored on the
        context for the mapper. When omitted (standalone callers), ``active_move_ids``
        falls back to reading the raw request.
        """
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

        # Build active_move_ids: 4-element list mirroring the masker's move-slot
        # assignment. The per-decision LegalActions snapshot is the source of truth
        # (request order, struggle already excluded) — identical to what the mask was
        # built from. Without it (standalone callers), fall back to the raw request.
        if legal is not None:
            active_move_ids = (list(legal.move_ids) + [None, None, None, None])[:4]
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
            legal=legal,
        )
