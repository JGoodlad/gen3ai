import numpy as np
from .base import ObservationEncoder
from .pokemon import PokemonEncoder
from .active_context import ActiveContextEncoder
from .global_env import GlobalEnvEncoder
from .species import SpeciesEncoder
from .items import ItemsEncoder
from .types import TypeEncoder
from .abilities import AbilitiesEncoder
from .moves import MovesEncoder
from .reactive import ReactiveEncoder
from .constants import (
    POKEMON_VECTOR_DIM,
    POKEMON_FULL_DIM,
    TEAM_SIZE,
    OFFSET_OUR_TEAM,
    OFFSET_OPP_TEAM,
    OFFSET_CONTEXT,
    OFFSET_GLOBAL,
    OFFSET_REACTIVE,
    REACTIVE_DIM,
    ACTIVE_CONTEXT_DIM,
    GLOBAL_ENV_DIM
)
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Dict, Any, List, Tuple
import json
import os
from agents.action.mask_generator import Gen3ActionMasker
from agents.observation.reactive import ReactiveEncoder as _ReactiveEncoder

def load_mappings():
    """Loads move, species, item, ability, and nature mappings with validation."""
    mappings = {}
    mapping_files = {
        "species": "data/pokemon/gen3_species.json",
        "moves": "data/pokemon/gen3_moves.json",
        "abilities": "data/pokemon/gen3_abilities.json",
        "items": "data/pokemon/gen3_items.json"
    }
    for key, path in mapping_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"CRITICAL: Mapping file missing: {path}. Run data generation script first!")

        with open(path, "r") as f:
            data = json.load(f)
            if not data:
                raise ValueError(f"CRITICAL: Mapping file is empty: {path}")
            # Normalize data: Ensure every entry is a dict with a 'num' key
            normalized = {}
            for name, val in data.items():
                if isinstance(val, dict):
                    normalized[name] = val
                else:
                    normalized[name] = {"num": int(val)}
            mappings[key] = normalized

    # Load ability priors (Smogon Gen3 OU usage) — used for opp-unrevealed encoding.
    # Format: {species_name: {ability_id: probability}}.
    _ability_priors_path = "data/pokemon/gen3_ability_priors.json"
    if not os.path.exists(_ability_priors_path):
        raise FileNotFoundError(
            f"CRITICAL: {_ability_priors_path} missing. Run "
            "tools/smogon_stats_downloader/compute_priors.py first."
        )
    with open(_ability_priors_path, "r") as f:
        mappings["ability_priors"] = json.load(f)

    # Load natures from poke_env static data — used for spread encoding
    _natures_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "poke_env", "data", "static", "natures.json"))
    if not os.path.exists(_natures_path):
        raise FileNotFoundError(f"CRITICAL: natures.json missing at {_natures_path}. poke_env static data may be corrupted.")
    with open(_natures_path, "r") as f:
        raw_natures = json.load(f)
    # Keep only the stat multipliers (atk/def/spa/spd/spe); drop "num"
    mappings["natures"] = {
        name: {k: float(v) for k, v in entry.items() if k in ("atk", "def", "spa", "spd", "spe")}
        for name, entry in raw_natures.items()
    }

    # Pre-compute reverse mappings for IDs to names
    mappings["reverse"] = {}
    for category in ["species", "moves", "abilities", "items"]:
        rev = {}
        for name, data in mappings[category].items():
            if isinstance(data, dict) and "num" in data:
                num = data["num"]
                # For moves, prefer the base "hiddenpower" name over typed HP variants
                # (all 17 HP variants share num=237; the base name is the honest display)
                if num not in rev or name == "hiddenpower":
                    rev[num] = name
            elif isinstance(data, (int, float)):
                rev[int(data)] = name
        mappings["reverse"][category] = rev

    return mappings

def get_observation_encoder(mappings):
    return Gen3ObservationEncoder(mappings)

class Gen3ObservationEncoder(ObservationEncoder):
    """
    Top-level encoder that orchestrates the entire Gen 3 observation vector.
    Total dimensions: base_dimension + 11 (prev_mask) + N_HISTORY_TURNS * TURN_DELTA_DIM
    """
    
    def __init__(self, mappings: Dict[str, Any] = None):
        self.mappings = mappings or {}
        mappings = self.mappings
        
        # Sub-encoders
        rev = self.mappings.get("reverse", {})
        self.species_encoder = SpeciesEncoder(mappings.get("species"), rev.get("species"))
        self.items_encoder = ItemsEncoder(mappings.get("items"), rev.get("items"))
        self.type_encoder = TypeEncoder()
        # Smogon-derived per-species ability priors, loaded by load_mappings()
        # from data/pokemon/gen3_ability_priors.json. The encoder picks the
        # top-2 abilities by Smogon usage for opp-unrevealed slots and writes
        # the dominance probability of ability1 alongside.
        self.abilities_encoder = AbilitiesEncoder(
            mappings.get("abilities"),
            rev.get("abilities"),
            species_to_ability_priors=mappings.get("ability_priors", {}),
        )
        self.moves_encoder = MovesEncoder(rev.get("moves"))
        
        self.pokemon_encoder = PokemonEncoder(
            self.species_encoder,
            self.items_encoder,
            self.type_encoder,
            self.abilities_encoder,
            self.moves_encoder,
            natures=mappings.get("natures", {}),
        )
        
        self.active_context_encoder = ActiveContextEncoder(mappings.get("moves"))
        self.global_env_encoder = GlobalEnvEncoder()
        # ability_priors threads into reactive so matchup cells against
        # unrevealed opp abilities show expected effectiveness instead of
        # the live (None → 1.0×) fallback. Mirrors the AbilitiesEncoder wiring.
        self.reactive_encoder = ReactiveEncoder(
            ability_priors=mappings.get("ability_priors", {}),
        )
        from agents.observation.turn_delta_encoder import TurnDeltaEncoder, TURN_DELTA_DIM as _TD_DIM
        self.turn_delta_encoder = TurnDeltaEncoder(
            mappings.get("moves", {}),
            mappings.get("species", {}),
        )
        self._turn_delta_dim = _TD_DIM
        self.current_battle_id = None

    @property
    def base_dimension(self) -> int:
        """Raw encoder output dimension, before the previous-turn mask is appended."""
        return OFFSET_REACTIVE + REACTIVE_DIM

    @property
    def dimension(self) -> int:
        """Full observation dimension including the 11-dim prev-mask and N-turn history block."""
        from agents.observation.turn_delta_encoder import TURN_DELTA_DIM
        from agents.model.features_extractor import N_HISTORY_TURNS
        return self.base_dimension + 11 + N_HISTORY_TURNS * TURN_DELTA_DIM

    def encode(self, battle: AbstractBattle, hp_tracker=None) -> np.ndarray:
        """Encode the full base observation vector.

        hp_tracker: optional HiddenPowerTracker whose per-species probability
        vectors are written into each opponent mon's 17-dim HP block. None
        leaves the blocks at zero (e.g. when called outside the training env).
        """
        vec = np.zeros(self.base_dimension, dtype=np.float32)

        # Build the current-board read-model ONCE per decision through the strict boundary
        # (`battle.strict_view().live`) and thread it to every sub-encoder. Requires a
        # Gen3Battle; a plain poke-env Battle / unit-test mock (no strict_view/live_view)
        # falls back to None, and each sub-encoder then reads the raw object — byte-identical.
        strict = battle.strict_view() if hasattr(battle, "strict_view") else None
        if strict is not None:
            live = strict.live
        else:
            live = battle.live_view() if hasattr(battle, "live_view") else None

        # 1. Our Team — current-board per-mon facts read through the LiveView slot.
        our_team_list = self.get_team_list(battle, is_opponent=False)
        for i in range(TEAM_SIZE):
            mon = our_team_list[i] if i < len(our_team_list) else None
            live_mon = live.ours.get(mon.species) if (live is not None and mon is not None) else None
            mon_vec = self.pokemon_encoder.encode(mon, battle, is_own=True, live_mon=live_mon)
            is_active = 1.0 if (mon and mon.active) else 0.0

            start = OFFSET_OUR_TEAM + (i * POKEMON_FULL_DIM)
            vec[start : start + POKEMON_VECTOR_DIM] = mon_vec
            vec[start + POKEMON_VECTOR_DIM] = is_active

        # 2. Opponent Team — HP block populated from the tracker when supplied.
        opponents = self.get_team_list(battle, is_opponent=True)

        for i in range(TEAM_SIZE):
            mon = opponents[i] if i < len(opponents) else None
            live_mon = live.opp.get(mon.species) if (live is not None and mon is not None) else None
            if hp_tracker is not None and mon is not None:
                hp_probs = hp_tracker.get_probs(mon.species)
                hp_known = hp_tracker.is_known(mon.species)
            else:
                hp_probs = None
                hp_known = False
            mon_vec = self.pokemon_encoder.encode(
                mon, battle, is_own=False, hp_probs=hp_probs, hp_known=hp_known, live_mon=live_mon
            )
            # Active flag through the LiveView slot (LivePokemon.active is set at fold time
            # from poke-env's opponent_active_pokemon accessor, so this is byte-identical to
            # the old `mon is battle.opponent_active_pokemon` identity check). Fall back to the
            # raw check only on the plain-Battle / unit-test path where there is no LiveView.
            if live_mon is not None:
                is_active = 1.0 if live_mon.active else 0.0
            else:
                is_active = 1.0 if (mon and mon is battle.opponent_active_pokemon) else 0.0

            start = OFFSET_OPP_TEAM + (i * POKEMON_FULL_DIM)
            vec[start : start + POKEMON_VECTOR_DIM] = mon_vec
            vec[start + POKEMON_VECTOR_DIM] = is_active

        # 3. Active Context + 4. Global Environment — sourced from the same LiveView
        # (current-board read-model folded from the event log).
        our_active = live.ours.active if live else None
        opp_active = live.opp.active if live else None
        vec[OFFSET_CONTEXT : OFFSET_CONTEXT + ACTIVE_CONTEXT_DIM] = \
            self.active_context_encoder.encode(our_active)
        vec[OFFSET_CONTEXT + ACTIVE_CONTEXT_DIM : OFFSET_CONTEXT + (2 * ACTIVE_CONTEXT_DIM)] = \
            self.active_context_encoder.encode(opp_active)

        # 4. Global Environment
        if live is not None:
            vec[OFFSET_GLOBAL : OFFSET_GLOBAL + GLOBAL_ENV_DIM] = \
                self.global_env_encoder.encode(live)
        
        # 5. Reactive Features — fainted counts + active-status read through the LiveView;
        # the move-effectiveness matrices stay on the raw battle (see reactive.py).
        vec[OFFSET_REACTIVE : OFFSET_REACTIVE + REACTIVE_DIM] = \
            self.reactive_encoder.encode(battle, hp_tracker=hp_tracker, live=live)
        
        return vec

    def get_observation(self, battle: AbstractBattle) -> Dict[str, Any]:
        """
        Standardized entry point for getting the full observation dictionary.
        Includes both the encoded state vector and the action mask.
        """
        if battle.wait:
            error_msg = f"⚠️ [OBSERVATION] CRITICAL: Observation requested while battle.wait is True for {battle.battle_tag}"
            print(error_msg)
            raise RuntimeError(error_msg)
            
        obs = self.encode(battle)
        mask = Gen3ActionMasker.get_mask(battle)
        return {
            "observation": obs,
            "action_mask": mask
        }

    def get_layout(self) -> Dict[str, Any]:
        from agents.observation.turn_delta_encoder import TURN_DELTA_DIM as _TD_DIM
        from agents.model.features_extractor import N_HISTORY_TURNS as _N_HIST
        pokemon_layout = self.pokemon_encoder.get_layout()
        return {
            "parts": {
                "our_team": {
                    "start": OFFSET_OUR_TEAM, 
                    "end": OFFSET_OPP_TEAM, 
                    "reshape": (TEAM_SIZE, POKEMON_FULL_DIM)
                },
                "opp_team": {
                    "start": OFFSET_OPP_TEAM, 
                    "end": OFFSET_CONTEXT, 
                    "reshape": (TEAM_SIZE, POKEMON_FULL_DIM)
                },
                "context": {
                    "start": OFFSET_CONTEXT, 
                    "end": OFFSET_GLOBAL, 
                    "reshape": (2, self.active_context_encoder.dimension)
                },
                "global": {
                    "start": OFFSET_GLOBAL, 
                    "end": OFFSET_REACTIVE, 
                    "dim": self.global_env_encoder.dimension
                },
                "reactive": {
                    "start": OFFSET_REACTIVE, 
                    "end": self.dimension, 
                    "dim": self.reactive_encoder.dimension
                }
            },
            "pokemon": pokemon_layout,
            "total_dim": self.dimension,    # base + 11 prev_mask + N * TURN_DELTA_DIM (157) turn_history
            "base_dim": self.base_dimension, # raw encoder output without prev_mask or turn_history
            "prev_mask_dim": 11,
            "turn_delta_dim": _TD_DIM,
            "n_history_turns": _N_HIST,
            "turn_history_offset": self.base_dimension + 11,
            "turn_history_dim": _N_HIST * _TD_DIM,
            "active_context_dim": ACTIVE_CONTEXT_DIM,
            "reactive_layout": _ReactiveEncoder().get_layout(),
            "global_layout": self.global_env_encoder.get_layout(),
            "max_species": 400,
            "species_embedding_dim": 32,
            "max_moves": 400,
            "move_embedding_dim": 16,
            "max_items": 600,
            "item_embedding_dim": 16,
            "max_abilities": 100,
            "ability_embedding_dim": 16,
            "max_types": 20, # 18 types + placeholders
            "type_embedding_dim": 16
        }

    def get_features_extractor_kwargs(self) -> Dict[str, Any]:
        return {
            "layout": self.get_layout(),
            "mappings": self.mappings
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        desc = {"our_team": [], "opp_team": []}
        
        # 1. Teams
        for i in range(TEAM_SIZE):
            start = OFFSET_OUR_TEAM + (i * POKEMON_FULL_DIM)
            mon_vec = vector[start : start + POKEMON_VECTOR_DIM]
            is_active = vector[start + POKEMON_VECTOR_DIM] > 0.5
            if np.any(mon_vec):
                mon_desc = self.pokemon_encoder.describe_vector(mon_vec)
                mon_desc["active"] = is_active
                desc["our_team"].append(mon_desc)
                
            start_opp = OFFSET_OPP_TEAM + (i * POKEMON_FULL_DIM)
            opp_vec = vector[start_opp : start_opp + POKEMON_VECTOR_DIM]
            is_active_opp = vector[start_opp + POKEMON_VECTOR_DIM] > 0.5
            if np.any(opp_vec):
                opp_desc = self.pokemon_encoder.describe_vector(opp_vec)
                opp_desc["active"] = is_active_opp
                desc["opp_team"].append(opp_desc)
        
        # 2. Context
        our_active_ctx = vector[OFFSET_CONTEXT : OFFSET_CONTEXT + ACTIVE_CONTEXT_DIM]
        opp_active_ctx = vector[OFFSET_CONTEXT + ACTIVE_CONTEXT_DIM : OFFSET_CONTEXT + (2 * ACTIVE_CONTEXT_DIM)]
        desc["our_active"] = self.active_context_encoder.describe_vector(our_active_ctx)
        desc["opp_active"] = self.active_context_encoder.describe_vector(opp_active_ctx)
        
        # 3. Global
        global_vec = vector[OFFSET_GLOBAL : OFFSET_GLOBAL + GLOBAL_ENV_DIM]
        desc["world"] = self.global_env_encoder.describe_vector(global_vec)
        
        # 4. Reactive
        reactive_vec = vector[OFFSET_REACTIVE : OFFSET_REACTIVE + REACTIVE_DIM]
        desc["momentum"] = self.reactive_encoder.describe_vector(reactive_vec)

        # 5. TurnDelta tail (present when the full obs is passed, not just base)
        td_start = self.base_dimension + 11
        if len(vector) >= td_start + self._turn_delta_dim:
            td_vec = vector[td_start : td_start + self._turn_delta_dim]
            desc["turn_delta"] = self.turn_delta_encoder.describe_vector(td_vec)

        return desc

    def integrity_check(self, vector: np.ndarray) -> Tuple[List[str], bool]:
        warnings = []
        is_critical = False
        desc = self.describe_vector(vector)
        
        # 1. Active Pokémon Check
        our_active = [mon for mon in desc['our_team'] if mon.get('active')]
        if len(our_active) > 1:
            warnings.append(f"CRITICAL: Multiple active Pokémon on our team: {[m['species'] for m in our_active]}")
            is_critical = True
        elif len(our_active) == 0:
            warnings.append("Note: No active Pokémon found on our team.")
            
        opp_active = [mon for mon in desc['opp_team'] if mon.get('active')]
        if len(opp_active) > 1:
            warnings.append(f"CRITICAL: Multiple active Pokémon on opponent team: {[m['species'] for m in opp_active]}")
            is_critical = True
            
        # 2. HP/Fainted Consistency
        fainted_our_list = len([mon for mon in desc['our_team'] if float(mon['hp'].strip('%')) == 0])
        fainted_our_momentum = desc['momentum']['fainted_our']
        if fainted_our_list != fainted_our_momentum:
             warnings.append(f"CRITICAL: Our fainted count mismatch! Team list ({fainted_our_list}) != momentum ({fainted_our_momentum})")
             is_critical = True
             
        fainted_opp_list = [mon['species'] for mon in desc['opp_team'] if float(mon['hp'].strip('%')) == 0]
        fainted_opp_count = len(fainted_opp_list)
        fainted_opp_momentum = desc['momentum']['fainted_opp']
        if fainted_opp_momentum > fainted_opp_count:
             warnings.append(f"CRITICAL: Opponent fainted count (momentum={fainted_opp_momentum}) > seen in team list ({fainted_opp_count}). Team fainted: {fainted_opp_list}")
             is_critical = True
        elif fainted_opp_momentum < fainted_opp_count:
             warnings.append(f"Mismatch: Opponent fainted count (momentum={fainted_opp_momentum}) < seen in team list ({fainted_opp_count}). Team fainted: {fainted_opp_list}")
             is_critical = True

        return warnings, is_critical
