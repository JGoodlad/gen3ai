from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np

from agents.training.slot_registry import SlotRegistry


@dataclass
class BattleContext:
    """
    Frozen per-turn snapshot of agent-relevant battle state.

    Built once per turn in embed_battle() and stored as self._last_ctx on the env.
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

    # Move IDs in action-slot order (indices 0-3 correspond to actions 6-9).
    # None for disabled/PP-depleted slots or when no moves are available (e.g. forced switch).
    # Always length 4.
    our_moves: list  # list[str | None], length 4

    def __post_init__(self):
        if self.mask.shape != (11,):
            raise RuntimeError(
                f"BattleContext mask shape {self.mask.shape} != (11,) at turn {self.turn}"
            )
        if len(self.our_moves) != 4:
            raise RuntimeError(
                f"BattleContext our_moves length {len(self.our_moves)} != 4 at turn {self.turn}"
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
        opp_active = (
            battle.opponent_active_pokemon.species
            if battle.opponent_active_pokemon
            else "NONE"
        )

        # Build our_moves: 4-element list mirroring the masker's move-slot assignment.
        # The masker reads request_moves from battle.last_request and assigns slot i+6
        # to request_moves[i], skipping disabled moves and struggle. We do the same here
        # so that our_moves[i] is the move ID for action slot i+6 (None if disabled).
        our_moves: list = [None, None, None, None]
        try:
            active_request = battle.last_request.get("active", [{}])[0]
            request_moves = active_request.get("moves", [])
            for i, move_data in enumerate(request_moves):
                if i < 4 and move_data.get("id") != "struggle":
                    if not move_data.get("disabled", False):
                        our_moves[i] = move_data.get("id")
        except (AttributeError, IndexError, TypeError):
            # No request available (e.g. forced switch phase) — all slots remain None
            pass

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
            our_moves=our_moves,
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

    # What they did this turn (may be None if not visible from battle log)
    opp_move_id: str | None
    opp_switch_to: str | None
    opp_prev_active: str

    # HP outcomes per slot (indexed by slot_map from BattleContext)
    our_hp_delta: np.ndarray      # (6,) float32 — negative means damage taken
    opp_hp_delta: np.ndarray      # (6,) float32

    # Faint events this turn
    we_fainted: bool
    opp_fainted: bool

    # TODO: status and stat-stage deltas — see designs/ai_v3/todo.md
    # Aromatherapy clears the entire team at once; Calm Mind / Curse track
    # stat stages that reset on switch. Needs per-slot before/after snapshots
    # rather than a delta list.

    @classmethod
    def build(cls, prev_ctx: BattleContext, curr_ctx: BattleContext, action: int) -> TurnDelta:
        our_hp_delta = curr_ctx.our_hp - prev_ctx.our_hp
        opp_hp_delta = curr_ctx.opp_hp - prev_ctx.opp_hp

        we_fainted = curr_ctx.our_fainted_count > prev_ctx.our_fainted_count
        opp_fainted = curr_ctx.opp_fainted_count > prev_ctx.opp_fainted_count

        if action < 6:
            our_switch_to = curr_ctx.our_active if curr_ctx.our_active != "NONE" else None
            our_move_id = None
        elif action == 10:
            our_switch_to = None
            our_move_id = "struggle"
        else:
            # action is 6-9; map to move slot index 0-3 in prev_ctx.our_moves
            our_switch_to = None
            our_move_id = prev_ctx.our_moves[action - 6]

        opp_switched = prev_ctx.opp_active != curr_ctx.opp_active
        opp_switch_to = curr_ctx.opp_active if opp_switched and curr_ctx.opp_active != "NONE" else None
        # poke-env exposes Pokemon.last_move (the last move used by that pokemon).
        # curr_ctx does not hold a battle reference, so opp_move_id must be sourced
        # externally (e.g. from battle.opponent_active_pokemon.last_move after the turn
        # resolves). TurnDelta.build() operates on frozen contexts only, so we leave
        # opp_move_id = None here. Callers with access to the live battle object can
        # populate this field themselves if needed.
        opp_move_id = None

        return cls(
            our_move_id=our_move_id,
            our_switch_to=our_switch_to,
            our_prev_active=prev_ctx.our_active,
            opp_move_id=opp_move_id,
            opp_switch_to=opp_switch_to,
            opp_prev_active=prev_ctx.opp_active,
            our_hp_delta=our_hp_delta,
            opp_hp_delta=opp_hp_delta,
            we_fainted=we_fainted,
            opp_fainted=opp_fainted,
        )

    @classmethod
    def empty(cls) -> TurnDelta:
        return cls(
            our_move_id=None, our_switch_to=None, our_prev_active="NULL",
            opp_move_id=None, opp_switch_to=None, opp_prev_active="NULL",
            our_hp_delta=np.zeros(6, dtype=np.float32),
            opp_hp_delta=np.zeros(6, dtype=np.float32),
            we_fainted=False, opp_fainted=False,
        )
