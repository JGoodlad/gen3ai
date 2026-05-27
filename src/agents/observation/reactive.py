import numpy as np
from .base import ObservationEncoder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.side_condition import SideCondition
from .constants import REACTIVE_DIM, TEAM_SIZE
from agents.gen3_mechanics import effective_multiplier
from typing import Any, Dict, List, Tuple


def _hp_expected_multiplier(move, attacker, opp, hp_tracker) -> float:
    """Effectiveness of `move` against `opp`, with Hidden Power's unknown type
    handled by averaging across the tracker's narrowed distribution.

    For bare "hiddenpower" (opponent pre-reveal), poke-env's `move.type` defaults
    to Normal, so the plain `effective_multiplier` is wrong. When a tracker is
    available, compute the expected effectiveness over the 16 candidate types
    weighted by their current probabilities. For typed HP variants (own team)
    `move.type` is already correct, so we fall through to the plain path —
    which is mathematically identical to the weighted sum against a one-hot.
    """
    if (
        move.id == "hiddenpower"
        and hp_tracker is not None
        and attacker is not None
    ):
        from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER
        probs = hp_tracker.get_probs(attacker.species)
        if probs is not None and probs.sum() > 0:
            return float(sum(
                probs[i] * effective_multiplier(HIDDEN_POWER_TYPE_ORDER[i], opp)
                for i in range(16) if probs[i] > 0
            ))
    return effective_multiplier(move.type, opp)


class ReactiveEncoder(ObservationEncoder):
    """
    Encodes reactive features:
    - Base Power of 4 active moves (4)
    - Damage multipliers of 4 active moves (4)
    - Fainted counts (2)
    - Status flag (1)
    - Forced Struggle flag (1)
    - Matchup Matrix: Our moves vs Their mons (144)
    - Matchup Matrix: Their moves vs Our mons (144)
    Total: 300 dims
    (HP and Spikes removed — duplicated in per-Pokémon vector and global env respectively)
    """
    
    @property
    def dimension(self) -> int:
        return REACTIVE_DIM

    def encode(self, battle: AbstractBattle, hp_tracker=None) -> np.ndarray:
        """Encode the 300-dim reactive block.

        hp_tracker: optional HiddenPowerTracker. When supplied, matchup scalars
        for bare "hiddenpower" move slots are replaced with the expected
        effectiveness across the tracker's narrowed distribution. Without it,
        bare HP slots fall back to poke-env's default move.type (typically
        Normal), preserving legacy behaviour when called outside training.
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        if battle is None:
            return vec

        # 1. Active Moves (Power and Multiplier)
        moves_base_power = np.zeros(4)
        moves_dmg_multiplier = np.ones(4)

        mon_move_ids = []
        if battle.active_pokemon:
            mon_move_ids = [m.id for m in battle.active_pokemon.moves.values()]

        # Skip Struggle — it has a dedicated action (10) and a dedicated flag (vec[15]).
        # Filling the move slots with Struggle's stats would create a confusing alias
        # between slot 0 (action 6) and the Struggle action (10).
        is_forced_struggle = (
            len(battle.available_moves) == 1
            and battle.available_moves[0].id == "struggle"
        )

        if not is_forced_struggle:
            for i, move in enumerate(battle.available_moves):
                if i >= 4:
                    break
                moves_base_power[i] = move.base_power / 200.0
                if battle.opponent_active_pokemon is not None:
                    mult = _hp_expected_multiplier(
                        move, battle.active_pokemon, battle.opponent_active_pokemon, hp_tracker
                    )
                    moves_dmg_multiplier[i] = mult / 4.0

        vec[0:4] = moves_base_power
        vec[4:8] = moves_dmg_multiplier

        # 2. Fainted Counts
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6.0
        fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6.0
        vec[8] = fainted_mon_team
        vec[9] = fainted_mon_opponent

        # 3. Status (HP and Spikes removed — available in per-Pokémon vector and global env)
        vec[10] = 1.0 if battle.active_pokemon and battle.active_pokemon.status else 0.0

        # 4. Forced Struggle
        vec[11] = 1.0 if is_forced_struggle else 0.0

        # --- Matchup Matrices ---
        our_team = self.get_team_list(battle, is_opponent=False)
        their_team = self.get_team_list(battle, is_opponent=True)

        # 5. Our moves vs Their mons (144 dims)
        cursor = 12
        for i in range(TEAM_SIZE):
            our_mon = our_team[i] if i < len(our_team) else None
            our_moves = self.get_sorted_moves(our_mon)
            for move_idx in range(4):
                move = our_moves[move_idx] if move_idx < len(our_moves) else None
                for j in range(TEAM_SIZE):
                    their_mon = their_team[j] if j < len(their_team) else None
                    if move and their_mon:
                        # Normalize by 4.0 to keep values in [0, 1] range for better MLP convergence
                        vec[cursor] = _hp_expected_multiplier(move, our_mon, their_mon, hp_tracker) / 4.0
                    cursor += 1

        # 6. Their moves vs Our mons (144 dims)
        for i in range(TEAM_SIZE):
            their_mon = their_team[i] if i < len(their_team) else None
            their_moves = self.get_sorted_moves(their_mon)
            for move_idx in range(4):
                move = their_moves[move_idx] if move_idx < len(their_moves) else None
                for j in range(TEAM_SIZE):
                    our_mon = our_team[j] if j < len(our_team) else None
                    if move and our_mon:
                        # Normalize by 4.0 to keep values in [0, 1] range for better MLP convergence
                        vec[cursor] = _hp_expected_multiplier(move, their_mon, our_mon, hp_tracker) / 4.0
                    cursor += 1
        
        return vec

    def get_layout(self) -> Dict[str, Any]:
        return {
            "move_power": {"offset": 0, "dim": 4},
            "move_multiplier": {"offset": 4, "dim": 4},
            "fainted": {"offset": 8, "dim": 2},
            "active_status": {"offset": 10, "dim": 1},
            "forced_struggle": {"offset": 11, "dim": 1},
            "our_matchups": {"offset": 12, "dim": 144},
            "their_matchups": {"offset": 156, "dim": 144}
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        # Extract matrices and scale back up by 4.0 for human-readable display
        our_m = vector[12:156].reshape(TEAM_SIZE, 4, TEAM_SIZE) * 4.0
        their_m = vector[156:300].reshape(TEAM_SIZE, 4, TEAM_SIZE) * 4.0
        
        return {
            "fainted_our": int(vector[8] * 6),
            "fainted_opp": int(vector[9] * 6),
            "active_move_mults": [f"{m*4.0:.1f}x" for m in vector[4:8].tolist()],
            "struggle": bool(vector[11]),
            "our_vs_their": our_m, # Full matrix for deeper trace
            "their_vs_our": their_m
        }
