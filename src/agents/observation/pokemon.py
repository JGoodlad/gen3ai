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

# Status → condition one-hot slot, in both the read-model's id form (LivePokemon.status,
# e.g. "brn") and poke-env's Status enum (raw / unit-test fallback). Both map to the SAME
# slot so the emitted vector is byte-identical regardless of which source is read.
_STATUS_STR_IDX = {"brn": 1, "par": 2, "slp": 3, "frz": 4, "psn": 5, "tox": 6}
_STATUS_ENUM_IDX = {
    Status.BRN: 1, Status.PAR: 2, Status.SLP: 3,
    Status.FRZ: 4, Status.PSN: 5, Status.TOX: 6,
}

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
      - our mon (HP or not): hp_revealed=1, all-zero probs

    Own mons always read hp_revealed=1 with all-zero probs. The own HP *type* is not
    recoverable from the current-board read-model: in a live battle the server request
    re-keys a typed Hidden Power under the bare ``"hiddenpower"`` id (the move's own typed
    ``.id`` survives only on the raw poke-env ``Move`` object, which this boundary does not
    expose). The previous `_own_hp_type_index` therefore resolved to None on every real
    decision — the own type one-hot was already dead — so dropping it keeps the emitted
    vector byte-identical while removing the last raw ``mon.moves`` read here.
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
        live_mon=None,
    ) -> np.ndarray:
        """Encode a single Pokémon slot.

        live_mon: the :class:`~agents.battle.live_view.LivePokemon` current-board snapshot
        for this slot (built once per decision by ``state_encoder`` from ``battle.live_view``).
        When supplied, the *current-board* per-mon facts read here — status, HP fraction,
        status counter, spread (IVs/EVs/nature) and species/base-stats (via the species
        sub-encoder) — come through the read-model instead of the raw poke-env ``Pokemon``.
        ``None`` (unit-test / plain-Battle path) falls back to the raw ``mon``; both surfaces
        carry the same values, so the emitted vector is byte-identical. The item / type /
        ability / move sub-encoders are not part of this boundary and still read ``mon``.

        hp_probs: optional (16,) float32 from HiddenPowerTracker.get_probs(species)
        for opponent mons. Ignored for own mons.
        hp_known: for opponent mons, True when the tracker has made a determination
        — either narrowed via observation, or ruled out (4 moves revealed without HP).
        Ignored for own mons (always known).
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        if mon is None:
            return vec

        # 1. Species (1 ID + 6 Stats) — current-board read goes through LivePokemon
        species_vec = self.species_encoder.encode(mon, battle, live_mon=live_mon)
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

        # 5. Condition (8) — status one-hot, sourced from the read-model when available.
        # `status` below is reused by the status-counter block (9); keep it as the raw
        # source's value so both blocks agree.
        cursor = POKEMON_CONDITION_OFFSET
        if live_mon is not None:
            status_idx = _STATUS_STR_IDX.get(live_mon.status, 0)
        else:
            status_idx = _STATUS_ENUM_IDX.get(mon.status, 0)
        if status_idx > 0:
            vec[cursor + status_idx] = 1.0

        # 6. Moves (36)
        moves_vec = self.moves_encoder.encode(mon, battle)
        vec[POKEMON_MOVES_OFFSET : POKEMON_MOVES_OFFSET + len(moves_vec)] = moves_vec

        # 7. HP (1)
        vec[POKEMON_HP_OFFSET] = (
            live_mon.hp_fraction if live_mon is not None else mon.current_hp_fraction
        )

        # 8. Species known (1) — always 1.0 for a real slot; 0.0 for absent slots
        # (the None path returned early above, so all populated slots hit this)
        vec[POKEMON_SPECIES_KNOWN_OFFSET] = 1.0

        # 9. Status counters (2): sleep duration, toxic severity
        # Gen 3 sleep: 1–4 turns. Sleep Talk/Snore increment the counter but the
        # increment is discarded on switch-out (engine oversight), so this is approximate.
        # Toxic: resets to 1 on switch-in; practical max ~8 turns before fainting.
        if live_mon is not None:
            is_slp = live_mon.status == "slp"
            is_tox = live_mon.status == "tox"
            ctr = live_mon.status_counter
        else:
            is_slp = mon.status == Status.SLP
            is_tox = mon.status == Status.TOX
            ctr = getattr(mon, "status_counter", 0) or 0
        vec[POKEMON_COUNTER_OFFSET]     = min(ctr, 4) / 4.0 if is_slp else 0.0
        vec[POKEMON_COUNTER_OFFSET + 1] = min(ctr, 8) / 8.0 if is_tox else 0.0

        # 10. Spread block (18 dims): IVs (6) + EVs (6) + spread_known (1) + nature (5)
        # Own team: actual values from the teambuilder. Opponent: all zeros + spread_known=0
        # so the model can distinguish "unknown opponent" from "zero EVs on own Pokémon".
        #
        # mon.ivs / mon.evs should always be non-None for own team after the poke_env fix
        # (TeambuilderPokemon defaults: ivs=[31]*6, evs=[0]*6). The None guards below are
        # defensive fallbacks in case the Pokemon was never passed through teambuilder init.
        # LivePokemon mirrors mon.ivs/evs/nature (own side only, spread_known gated), so the
        # read-model and raw paths produce the same spread block.
        if is_own:
            if live_mon is not None:
                ivs = live_mon.ivs       # tuple[int] | None: [HP, Atk, Def, SpA, SpD, Spe]
                evs = live_mon.evs       # tuple[int] | None
                nature_name = live_mon.nature
            else:
                ivs = mon.ivs            # list[int] | None
                evs = mon.evs            # list[int] | None
                nature_name = mon.nature  # str | None
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
            # nature_name: str | None; fallback to "serious" (all 1.0 modifiers)
            nature_mods = self._natures.get(nature_name or "serious", {})
            for j, stat in enumerate(self._NATURE_STAT_ORDER):
                vec[off + 13 + j] = float(nature_mods.get(stat, 1.0))
        # Opponent slots: all 18 dims remain 0.0 (spread_known=0 is the signal)

        # 11. Hidden Power candidate block (17 dims): hp_revealed (1) + 16 type probs.
        # Own mons: state is always known → hp_revealed=1.0, probs all zero. The own HP
        # *type* is not recoverable from the read-model (the live request re-keys typed HP
        # to bare "hiddenpower"; the typed id survives only on the raw Move object this
        # boundary hides), and the former type one-hot resolved to None on every real
        # decision — so leaving probs zero is byte-identical and drops the last mon.moves read.
        if is_own:
            vec[POKEMON_HP_REVEALED_OFFSET] = 1.0
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
