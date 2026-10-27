"""``Gen3ObservationEncoder`` — the observation LAYOUT the model reads (``get_layout()``), and ``load_mappings``.

Every offset comes from ``agents/observation/constants.py``; the Rust encoder's ``layout.rs`` is held equal to it
by ``rust_core_obs_layout_test.py``. Never hardcode an index — read ``get_layout()``.
"""
from __future__ import annotations
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
    POKEMON_ACTIVE_OFFSET,
    TEAM_SIZE,
    OFFSET_OUR_TEAM,
    OFFSET_OPP_TEAM,
    OFFSET_CONTEXT,
    OFFSET_GLOBAL,
    OFFSET_REACTIVE,
    REACTIVE_DIM,
    OFFSET_PAIR_HISTORY,
    PAIR_HISTORY_DIM,
    PAIR_HISTORY_CELL_DIM,
    OFFSET_EVENT_WINDOW,
    EVENT_WINDOW_N,
    EVENT_TOKEN_DIM,
    EVENT_WINDOW_DIM,
    ACTIVE_CONTEXT_DIM,
    GLOBAL_ENV_DIM,
    OFFSET_OBS_FACTS,
    OBS_FACTS_DIM,
    FACTS_SEEN_OFFSET,
    FACTS_SEEN_ROW_DIM,
    FACTS_CHOICE_OFFSET,
    FACTS_CHOICE_DIM,
    FACTS_VOL_OFFSET,
    FACTS_VOL_SIDE_DIM,
    FACTS_VOL_CELL_DIM,
    FACTS_VOL_EFFECTS,
    FACTS_SCREENS_OFFSET,
    FACTS_SCREENS,
)
from .obs_facts import describe as _describe_obs_facts
from typing import Dict, Any, List, Optional, Tuple
from agents.observation.reactive import ReactiveEncoder as _ReactiveEncoder


def load_mappings() -> Dict[str, Any]:
    """Assemble the observation encoder's reference mappings from the ``gen3_data`` facade.

    Each ``data/pokemon/`` file is parsed once by its concept module (``gen3_data.species``,
    ``.moves``, ``.items``, ``.abilities``, ``.priors``, ``.natures``); this borrows their raw
    dicts and inverts id→name reverse maps. The three upstreams (poke-env / Showdown / Smogon)
    stay hidden behind the facade — this loader speaks only domain concepts, no file paths."""
    from agents import gen3_data
    mappings: Dict[str, Any] = {
        # Reference dexes — the raw {id: {num, …}} dicts the sub-encoders read directly. A fresh
        # outer dict per call (matching the previous loader's semantics); inner records are the
        # shared, immutable-by-convention singletons.
        "species": dict(gen3_data.species.raw()),
        "moves": dict(gen3_data.moves.raw()),
        "abilities": dict(gen3_data.abilities.raw()),
        "items": dict(gen3_data.items.raw()),
        # Smogon usage priors for opp-unrevealed ability encoding ({species: {ability_id: prob}}).
        "ability_priors": gen3_data.priors.ability_raw(),
        # Nature stat multipliers ({nature: {atk/def/spa/spd/spe: mult}}) for spread encoding.
        "natures": gen3_data.natures.multipliers(),
    }

    # Pre-compute reverse mappings for IDs to names
    mappings["reverse"] = {}
    for category in ["species", "moves", "abilities", "items"]:
        rev: Dict[int, str] = {}
        for name, data in mappings[category].items():
            if isinstance(data, dict) and "num" in data:
                # gen3_species_formes_v1: an alternate/cosmetic FORME (Deoxys-Speed,
                # Unown-B, Castform-Sunny) shares its BASE's national-dex num, and the obs
                # species channel IS that num — so a forme can never be the decode of a
                # num. Skip forme rows outright rather than relying on "base sorts first".
                if data.get("baseSpecies"):
                    continue
                num = data["num"]
                # Hidden Power (gen3_typed_hidden_power_ids_v1): the OPPONENT's bare HP keeps num 237 →
                # decode it as the bare "hiddenpower" (its type is unknowable); OUR typed HP have
                # DISTINCT nums (355-370), each mapping uniquely to its typed name. The `or name ==
                # "hiddenpower"` only matters for 237 (it has no other claimant now), kept harmless.
                if num not in rev or name == "hiddenpower":
                    rev[num] = name
            elif isinstance(data, (int, float)):
                rev[int(data)] = name
        mappings["reverse"][category] = rev

    return mappings


def get_observation_encoder(mappings: Dict[str, Any]) -> "Gen3ObservationEncoder":
    return Gen3ObservationEncoder(mappings)


class Gen3ObservationEncoder(ObservationEncoder):
    """
    The observation's LAYOUT, as the MODEL reads it: ``dimension`` (the row length), ``get_layout()`` (every
    block's offset / shape — the extractor slices by it), ``get_features_extractor_kwargs()``, and the
    read-back ``describe_vector`` / ``integrity_check`` (the prober). The row itself is written by the Rust
    encoder (``src/rust_sim/src/encoder/``, the only encoder); the Python ``encode`` / ``get_observation`` over
    a poke-env battle are DELETED (T27 P6 slice 6d-2). gen3_frame_deletion_v1 deleted the two tail blocks (the
    prev-turn action mask and the lag frames), so ``dimension == base_dimension``.
    """
    
    def __init__(self, mappings: Optional[Dict[str, Any]] = None) -> None:
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

    @property
    def base_dimension(self) -> int:
        """Raw encoder output dimension, before the previous-turn mask is appended."""
        # gen3_obs_facts_v1: the OBS-FACTS block follows the H-B event window and closes base.
        return OFFSET_OBS_FACTS + OBS_FACTS_DIM

    @property
    def dimension(self) -> int:
        """Full observation dimension. gen3_frame_deletion_v1: identical to `base_dimension` —
        the prev-mask and lag-frame tail blocks are gone. Kept as a distinct property because
        every consumer reads `dimension`, and a future tail block would land here again."""
        return self.base_dimension

    def get_layout(self) -> Dict[str, Any]:
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
                },
                "pair_history": {
                    "start": OFFSET_PAIR_HISTORY,
                    "end": OFFSET_PAIR_HISTORY + PAIR_HISTORY_DIM,
                    "reshape": (TEAM_SIZE, TEAM_SIZE, PAIR_HISTORY_CELL_DIM)
                }
            },
            "pokemon": pokemon_layout,
            # gen3_frame_deletion_v1: total_dim == base_dim; the prev_mask_dim / turn_delta_dim /
            # n_history_turns / turn_history_offset / turn_history_dim keys are DELETED with their
            # blocks. Consumers that sliced by them (ObsUnpack, the prober's offsets, the schema)
            # are updated in the same pass — a key left behind reading 0 would be sliced silently.
            "total_dim": self.dimension,
            "base_dim": self.base_dimension,
            "active_context_dim": ACTIVE_CONTEXT_DIM,
            "pair_history_offset": OFFSET_PAIR_HISTORY,
            "pair_history_dim": PAIR_HISTORY_DIM,
            "pair_history_cell_dim": PAIR_HISTORY_CELL_DIM,
            "event_window_offset": OFFSET_EVENT_WINDOW,
            "event_window_dim": EVENT_WINDOW_DIM,
            "event_window_n": EVENT_WINDOW_N,
            "event_token_dim": EVENT_TOKEN_DIM,
            # gen3_obs_facts_v1: the OBS-FACTS block and its four sub-blocks (offsets INSIDE it).
            "obs_facts_offset": OFFSET_OBS_FACTS,
            "obs_facts_dim": OBS_FACTS_DIM,
            "obs_facts": {
                "seen": {"offset": FACTS_SEEN_OFFSET, "dim": TEAM_SIZE * FACTS_SEEN_ROW_DIM,
                         "rows": TEAM_SIZE, "row_dim": FACTS_SEEN_ROW_DIM},
                "choice": {"offset": FACTS_CHOICE_OFFSET, "dim": FACTS_CHOICE_DIM},
                "vol": {"offset": FACTS_VOL_OFFSET, "dim": 2 * FACTS_VOL_SIDE_DIM, "sides": 2,
                        "side_dim": FACTS_VOL_SIDE_DIM, "cell_dim": FACTS_VOL_CELL_DIM,
                        "effects": list(FACTS_VOL_EFFECTS)},
                "screens": {"offset": FACTS_SCREENS_OFFSET, "dim": 2 * len(FACTS_SCREENS),
                            "sides": 2, "side_dim": len(FACTS_SCREENS),
                            "conditions": list(FACTS_SCREENS)},
            },
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
        desc: Dict[str, Any] = {"our_team": [], "opp_team": []}
        
        # 1. Teams
        for i in range(TEAM_SIZE):
            start = OFFSET_OUR_TEAM + (i * POKEMON_FULL_DIM)
            mon_vec = vector[start : start + POKEMON_VECTOR_DIM]
            is_active = vector[start + POKEMON_ACTIVE_OFFSET] > 0.5
            if np.any(mon_vec):
                mon_desc = self.pokemon_encoder.describe_vector(mon_vec)
                mon_desc["active"] = is_active
                desc["our_team"].append(mon_desc)
                
            start_opp = OFFSET_OPP_TEAM + (i * POKEMON_FULL_DIM)
            opp_vec = vector[start_opp : start_opp + POKEMON_VECTOR_DIM]
            is_active_opp = vector[start_opp + POKEMON_ACTIVE_OFFSET] > 0.5
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

        # 5. gen3_obs_facts_v1
        desc["facts"] = _describe_obs_facts(vector[OFFSET_OBS_FACTS:OFFSET_OBS_FACTS + OBS_FACTS_DIM])

        # gen3_frame_deletion_v1: there is no TurnDelta tail to describe — the obs ends at base.
        # What HAPPENED last turn is read from the H-B event window instead (`event_window`),
        # which is the block that replaced these frames.

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
