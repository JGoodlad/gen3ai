import numpy as np
from .base import ObservationEncoder
from .constants import (
    POKEMON_VECTOR_DIM,
    POKEMON_SPECIES_OFFSET,
    POKEMON_ITEMS_OFFSET,
    POKEMON_TYPES_OFFSET,
    POKEMON_ABILITIES_OFFSET,
    POKEMON_CONDITION_OFFSET,
    POKEMON_MOVES_OFFSET,
    POKEMON_HP_OFFSET,
    POKEMON_SPECIES_KNOWN_OFFSET,
    POKEMON_COUNTER_OFFSET,
    POKEMON_SPREAD_OFFSET,
    POKEMON_SPREAD_DIM,
    POKEMON_HP_REVEALED_OFFSET,
    POKEMON_HP_PROBS_OFFSET,
    POKEMON_HP_BLOCK_DIM,
)
from .species import SpeciesEncoder
from .items import ItemsEncoder
from .types import TypeEncoder
from .abilities import AbilitiesEncoder
from .moves import MovesEncoder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.status import Status
from typing import Any, Dict

# Imported lazily inside encode() to avoid a hard import cycle with the training package.
_HP_TYPE_INDEX: "dict[str, int] | None" = None


def _hp_type_index() -> "dict[str, int]":
    """Lazy-built mapping from PokemonType.name (uppercase) → index in
    HIDDEN_POWER_TYPE_ORDER. Used to translate a known HP type into its
    one-hot slot in the 16-dim probability block."""
    global _HP_TYPE_INDEX
    if _HP_TYPE_INDEX is None:
        from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER
        _HP_TYPE_INDEX = {t.name: i for i, t in enumerate(HIDDEN_POWER_TYPE_ORDER)}
    return _HP_TYPE_INDEX


def _own_hp_type_index(mon: Any) -> "int | None":
    """If `mon` has a Hidden Power move with its type preserved (via the
    poke-env raw_id fix), return the index in HIDDEN_POWER_TYPE_ORDER for that
    type. Returns None if the mon has no HP, or only has bare "hiddenpower"
    with no type info."""
    for move_id, move in mon.moves.items():
        if not move_id.startswith("hiddenpower"):
            continue
        if move_id == "hiddenpower":
            continue  # type-less variant — opponent path, shouldn't reach here for own
        type_name = getattr(move.type, "name", None)
        if type_name is None:
            continue
        idx = _hp_type_index().get(type_name)
        if idx is not None:
            return idx
    return None

class PokemonEncoder(ObservationEncoder):
    """
    Aggregates all Pokémon-level encoders into a single POKEMON_VECTOR_DIM-wide vector.
    Layout: species(7) + items(3) + types(2) + abilities(2) + condition(7) + moves(36)
            + hp(1) + species_known(1) + status_counters(2) + spread(18) + hp_block(17) = 96 dims.
    The active flag (1 dim) is appended by state_encoder, making POKEMON_FULL_DIM = 98.

    hp_block carries Hidden Power information as `hp_revealed (1) + type_probs (16)`:
      - opponent unknown: hp_revealed=0, probs all zero
      - opponent narrowed (observed HP, types ruled out): hp_revealed=1, sparse probs
      - opponent ruled out (4 moves seen, none is HP): hp_revealed=1, all-zero probs
      - our mon with HP: hp_revealed=1, one-hot at the known type
      - our mon without HP: hp_revealed=1, all-zero probs

    The four states are linearly separable by the (hp_revealed, probs.any()) pair,
    so the model gets a clean signal that distinguishes "no information yet" from
    "no HP possible."
    """
    
    _NATURE_STAT_ORDER = ("atk", "def", "spa", "spd", "spe")

    def __init__(self, species_encoder, items_encoder, type_encoder, abilities_encoder, moves_encoder, natures: dict = None):
        self.species_encoder = species_encoder
        self.items_encoder = items_encoder
        self.type_encoder = type_encoder
        self.abilities_encoder = abilities_encoder
        self.moves_encoder = moves_encoder
        self._natures = natures or {}

    @property
    def dimension(self) -> int:
        return POKEMON_VECTOR_DIM

    def encode(
        self,
        mon: Any,
        battle: AbstractBattle,
        is_own: bool = False,
        hp_probs: "np.ndarray | None" = None,
        hp_known: bool = False,
    ) -> np.ndarray:
        """Encode a single Pokémon slot.

        hp_probs: optional (16,) float32 from HiddenPowerTracker.get_probs(species)
        for opponent mons. Ignored for own mons (we derive the one-hot ourselves
        from mon.moves).
        hp_known: for opponent mons, True when the tracker has made a determination
        — either narrowed via observation, or ruled out (4 moves revealed without HP).
        Ignored for own mons (always known).
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        if mon is None:
            return vec
            
        # 1. Species (1 ID + 6 Stats)
        species_vec = self.species_encoder.encode(mon, battle)
        vec[POKEMON_SPECIES_OFFSET : POKEMON_SPECIES_OFFSET + len(species_vec)] = species_vec
        
        # 2. Items (16 + 1)
        item_vec = self.items_encoder.encode(mon, battle)
        vec[POKEMON_ITEMS_OFFSET : POKEMON_ITEMS_OFFSET + len(item_vec)] = item_vec
        
        # 3. Combined Types (8)
        type_vec = self.type_encoder.encode(mon, battle)
        vec[POKEMON_TYPES_OFFSET : POKEMON_TYPES_OFFSET + len(type_vec)] = type_vec
        
        # 4. Abilities (3): [ability1_id, ability2_id, known_flag]
        # For unrevealed opp mons, ability1/2 hold the species' dex-possible
        # Gen 3 abilities (e.g. Snorlax = [Immunity, Thick Fat]); once revealed,
        # ability1 holds the actual ability and known flips to 1.
        ability_vec = self.abilities_encoder.encode(mon, battle)
        vec[POKEMON_ABILITIES_OFFSET : POKEMON_ABILITIES_OFFSET + len(ability_vec)] = ability_vec
        
        # 5. Condition (8)
        cursor = POKEMON_CONDITION_OFFSET
        status = mon.status
        if status:
            status_map = {
                Status.BRN: 1, Status.PAR: 2, Status.SLP: 3, 
                Status.FRZ: 4, Status.PSN: 5, Status.TOX: 6
            }
            idx = status_map.get(status, 0)
            if idx > 0:
                vec[cursor + idx] = 1.0
        
        # 6. Moves (36)
        moves_vec = self.moves_encoder.encode(mon, battle)
        vec[POKEMON_MOVES_OFFSET : POKEMON_MOVES_OFFSET + len(moves_vec)] = moves_vec
        
        # 7. HP (1)
        vec[POKEMON_HP_OFFSET] = mon.current_hp_fraction

        # 8. Species known (1) — always 1.0 for a real slot; 0.0 for absent slots
        # (the None path returned early above, so all populated slots hit this)
        vec[POKEMON_SPECIES_KNOWN_OFFSET] = 1.0

        # 9. Status counters (2): sleep duration, toxic severity
        # Gen 3 sleep: 1–4 turns. Sleep Talk/Snore increment the counter but the
        # increment is discarded on switch-out (engine oversight), so this is approximate.
        # Toxic: resets to 1 on switch-in; practical max ~8 turns before fainting.
        status = mon.status
        ctr = getattr(mon, "status_counter", 0) or 0
        vec[POKEMON_COUNTER_OFFSET]     = min(ctr, 4) / 4.0 if status == Status.SLP else 0.0
        vec[POKEMON_COUNTER_OFFSET + 1] = min(ctr, 8) / 8.0 if status == Status.TOX else 0.0

        # 10. Spread block (18 dims): IVs (6) + EVs (6) + spread_known (1) + nature (5)
        # Own team: actual values from the teambuilder. Opponent: all zeros + spread_known=0
        # so the model can distinguish "unknown opponent" from "zero EVs on own Pokémon".
        #
        # mon.ivs / mon.evs should always be non-None for own team after the poke_env fix
        # (TeambuilderPokemon defaults: ivs=[31]*6, evs=[0]*6). The None guards below are
        # defensive fallbacks in case the Pokemon was never passed through teambuilder init.
        if is_own:
            ivs = mon.ivs  # list[int] | None: [HP, Atk, Def, SpA, SpD, Spe]
            evs = mon.evs  # list[int] | None: [HP, Atk, Def, SpA, SpD, Spe]
            off = POKEMON_SPREAD_OFFSET
            # IVs: fallback to all-31 (competitive standard) rather than all-0
            # — encoding 0.0 for unknown IVs would be silently wrong
            iv_vals = ivs if ivs is not None else [31] * 6
            for j, iv in enumerate(iv_vals):
                vec[off + j] = iv / 31.0
            if evs is not None:
                for j, ev in enumerate(evs):
                    vec[off + 6 + j] = ev / 252.0
            vec[off + 12] = 1.0  # spread_known flag
            nature_name = mon.nature  # str | None; fallback to "serious" (all 1.0 modifiers)
            nature_mods = self._natures.get(nature_name or "serious", {})
            for j, stat in enumerate(self._NATURE_STAT_ORDER):
                vec[off + 13 + j] = float(nature_mods.get(stat, 1.0))
        # Opponent slots: all 18 dims remain 0.0 (spread_known=0 is the signal)

        # 11. Hidden Power candidate block (17 dims): hp_revealed (1) + 16 type probs.
        # Own mons: state is always known (set hp_revealed=1.0). If the mon has HP,
        # derive the type from the move object (preserved via the poke-env raw_id
        # fix in Pokemon._update_from_teambuilder) and write a one-hot. Otherwise
        # leave probs at zero — encoding "we know this mon definitively has no HP."
        if is_own:
            vec[POKEMON_HP_REVEALED_OFFSET] = 1.0
            idx = _own_hp_type_index(mon)
            if idx is not None:
                vec[POKEMON_HP_PROBS_OFFSET + idx] = 1.0
        elif hp_known:
            # Opponent: tracker has either narrowed (sparse probs) or ruled out
            # (probs all zero). Either way, set hp_revealed and copy the (possibly
            # all-zero) vector so the model can distinguish from "never observed."
            vec[POKEMON_HP_REVEALED_OFFSET] = 1.0
            if hp_probs is not None:
                vec[POKEMON_HP_PROBS_OFFSET : POKEMON_HP_PROBS_OFFSET + 16] = hp_probs

        return vec

    def get_layout(self) -> Dict[str, Any]:
        return {
            "species": {
                "offset": POKEMON_SPECIES_OFFSET, 
                "dim": self.species_encoder.dimension,
                "layout": self.species_encoder.get_layout()
            },
            "items": {
                "offset": POKEMON_ITEMS_OFFSET, 
                "dim": self.items_encoder.dimension,
                "layout": self.items_encoder.get_layout()
            },
            "types": {
                "offset": POKEMON_TYPES_OFFSET, 
                "dim": self.type_encoder.dimension,
                "layout": self.type_encoder.get_layout()
            },
            "abilities": {
                "offset": POKEMON_ABILITIES_OFFSET, 
                "dim": self.abilities_encoder.dimension,
                "layout": self.abilities_encoder.get_layout()
            },
            "condition": {"offset": POKEMON_CONDITION_OFFSET, "dim": 7},
            "moves": {
                "offset": POKEMON_MOVES_OFFSET, 
                "dim": self.moves_encoder.dimension,
                "layout": self.moves_encoder.get_layout()
            },
            "hp": {"offset": POKEMON_HP_OFFSET, "dim": 1},
            "species_known": {"offset": POKEMON_SPECIES_KNOWN_OFFSET, "dim": 1},
            "status_counters": {"offset": POKEMON_COUNTER_OFFSET, "dim": 2},
            "spread": {
                "offset": POKEMON_SPREAD_OFFSET,
                "dim": POKEMON_SPREAD_DIM,
                "layout": {
                    "ivs": {"offset": 0, "dim": 6, "stats": ["hp", "atk", "def", "spa", "spd", "spe"]},
                    "evs": {"offset": 6, "dim": 6, "stats": ["hp", "atk", "def", "spa", "spd", "spe"]},
                    "spread_known": {"offset": 12, "dim": 1},
                    "nature": {"offset": 13, "dim": 5, "stats": list(self._NATURE_STAT_ORDER)},
                }
            },
            "hp_block": {
                "offset": POKEMON_HP_REVEALED_OFFSET,
                "dim": POKEMON_HP_BLOCK_DIM,
                "layout": {
                    "hp_revealed": {"offset": 0, "dim": 1},
                    "hp_type_probs": {"offset": 1, "dim": 16},
                },
            },
            "pokemon_vector_dim": POKEMON_VECTOR_DIM,
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        species_part = vector[POKEMON_SPECIES_OFFSET : POKEMON_SPECIES_OFFSET + 7]
        species_desc = self.species_encoder.describe_vector(species_part)
        
        item_part = vector[POKEMON_ITEMS_OFFSET : POKEMON_ITEMS_OFFSET + self.items_encoder.dimension]
        item_name = self.items_encoder.describe_vector(item_part)
        
        type_part = vector[POKEMON_TYPES_OFFSET : POKEMON_TYPES_OFFSET + self.type_encoder.dimension]
        type_name = self.type_encoder.describe_vector(type_part)
        
        ability_part = vector[POKEMON_ABILITIES_OFFSET : POKEMON_ABILITIES_OFFSET + self.abilities_encoder.dimension]
        ability_name = self.abilities_encoder.describe_vector(ability_part)
        
        moves_part = vector[POKEMON_MOVES_OFFSET : POKEMON_MOVES_OFFSET + self.moves_encoder.dimension]
        moves_desc = self.moves_encoder.describe_vector(moves_part)
        
        return {
            "species": species_desc["name"],
            "hp": f"{vector[POKEMON_HP_OFFSET]*100:.1f}%",
            "types": type_name,
            "stats": {k: v for k, v in species_desc.items() if k != "name"},
            "status": self._decode_status(vector[POKEMON_CONDITION_OFFSET : POKEMON_CONDITION_OFFSET + 7]),
            "item": item_name,
            "ability": ability_name,
            "moves": moves_desc["moves"]
        }

    def _decode_status(self, vec: np.ndarray) -> str:
        names = ["NONE", "BRN", "PAR", "SLP", "FRZ", "PSN", "TOX"]
        for i, val in enumerate(vec):
            if val > 0.5:
                return names[i]
        return "NONE"
