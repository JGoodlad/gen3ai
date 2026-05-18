from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np

from agents.training.slot_registry import SlotRegistry
from agents.gen3_mechanics import PHAZING_MOVES, BOOST_DIM, boosts_array


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
    obs: np.ndarray             # (OBS_DIM,) float32 — encoded observation
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
    # Known behaviors (confirmed by transition_fuzz_e2e_test.py):
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
        obs: np.ndarray,
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
            obs=obs,
            our_slot_map=our_slots.snapshot(),
            opp_slot_map=opp_slots.snapshot(),
            our_hp=our_hp,
            opp_hp=opp_hp,
            our_active=our_active,
            opp_active=opp_active,
            our_fainted_count=sum(1 for m in battle.team.values() if m.fainted),
            opp_fainted_count=sum(1 for m in battle.opponent_team.values() if m.fainted),
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
            we_moved_first=battle.we_moved_first,
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
            our_switch_to = curr_ctx.our_active if curr_ctx.our_active != "NONE" else None
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

        our_failed_to_move = curr_ctx.our_cant_reason is not None
        opp_failed_to_move = curr_ctx.opp_cant_reason is not None

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
            our_cant_reason=curr_ctx.our_cant_reason,
            opp_failed_to_move=opp_failed_to_move,
            opp_cant_reason=curr_ctx.opp_cant_reason,
            our_boost_delta=our_boost_delta,
            opp_boost_delta=opp_boost_delta,
            our_effectiveness=curr_ctx.our_last_effectiveness,
            opp_effectiveness=curr_ctx.opp_last_effectiveness,
            we_moved_first=curr_ctx.we_moved_first,
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
        )
