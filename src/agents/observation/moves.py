import numpy as np
from .base import ObservationEncoder
from .constants import MOVE_SLOT_DIM, MAX_PP
from .types import TypeEncoder
from agents import gen3_movedex
from poke_env.battle.abstract_battle import AbstractBattle
from agents.enums import MoveCategory
from typing import Any, List, Dict

# Hidden Power's Pokémon ID. All 16 typed variants ("hiddenpowergrass" etc.)
# share this num with the bare "hiddenpower" in data/pokemon/gen3_moves.json,
# so a single equality check detects any HP slot regardless of how the type
# was (or wasn't) revealed. Used by the feature extractor to route HP slots
# through the weighted-type-embedding path.
HIDDEN_POWER_MOVE_NUM = 237

# Per-move-id cache of the encoded category scalar (0=status, 1=physical, 2=special).
# poke-env's `Move.category` is a property that, for gen3, walks `move.entry` ->
# `GenData.from_gen(3).moves[id]` and resolves the pre-split type-based category on every
# read — the #1 tottime line in the obs profile (~199 `move.entry` calls/encode, since the
# per-mon move encoder runs over ~48 slots/encode). The category is a *pure function of the
# move id* within our fixed gen3 data (it never changes mid-battle), so we memoize the
# already-computed scalar keyed by id and skip the property chain on every subsequent read.
#
# Byte-identity: the cached value IS the live `move.category`-derived scalar (computed the
# first time each id is seen, exactly as before), NOT a re-derivation from gen3_movedex —
# the dex's base_power<=0 => STATUS rule disagrees with poke-env for fixed-/variable-power
# moves (Seismic Toss, Endeavor, Return, and bare "hiddenpower" all read PHYSICAL live but
# STATUS in the dex), so only caching the live result stays identical.
_CATEGORY_VAL_CACHE: Dict[str, float] = {}


def _category_val(move) -> float:
    """Encoded category scalar for a move, memoized by move id off the live category.

    First read of an id resolves poke-env's `move.category` (the gen3 type-split property)
    and caches the 0/1/2 scalar; later reads of the same id return it without touching the
    `move.entry` / `GenData.from_gen` chain. Identical output to reading `move.category`
    every time, because the category is invariant per id in our gen3 data."""
    mid = move.id
    val = _CATEGORY_VAL_CACHE.get(mid)
    if val is None:
        cat = move.category
        if cat == MoveCategory.PHYSICAL:
            val = 1.0
        elif cat == MoveCategory.SPECIAL:
            val = 2.0
        else:
            val = 0.0
        _CATEGORY_VAL_CACHE[mid] = val
    return val


class MovesEncoder(ObservationEncoder):
    """
    Encodes move IDs and reveal status for 4 move slots.

    Move reference data (num / power / type / secondary / recoil / accuracy / never-miss)
    is read from the ``gen3_movedex`` concept module — the single source of truth, built
    from the same ``data/pokemon/gen3_moves.json`` this encoder used to be handed as a
    dict. Category and PP still come from the live move object (poke-env), unchanged.
    ``reverse_mapping`` (move num → name) is kept for ``describe_vector``.
    """

    def __init__(self, reverse_mapping=None):
        self.reverse_mapping = reverse_mapping or {}

    @property
    def dimension(self) -> int:
        return 4 * MOVE_SLOT_DIM

    def encode(self, mon: Any, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if mon is None:
            return vec
            
        # Get moves and SORT them by ID to ensure stable mapping across workers
        moves = self.get_sorted_moves(mon)
        
        for i in range(4):
            if i < len(moves):
                move = moves[i]
                move_id = move.id
                
                # Reference metadata from the gen3 move dex (single source of truth).
                md = gen3_movedex.get(move_id)
                if md is None:
                    raise ValueError(f"Unrecognized move: {move_id}. Update data/pokemon/gen3_moves.json")
                num = md.num
                secondary = 1.0 if md.has_secondary else 0.0
                recoil = 1.0 if md.has_recoil else 0.0

                if move_id == "hiddenpower":
                    # Bare "hiddenpower" id = type unknown (opponent before reveal).
                    # Our own HP arrives here as a typed variant (e.g. "hiddenpowergrass"),
                    # routed through the else branch where the dex supplies the real type
                    # and 70bp. The dex's bare-HP entry has basePower=0 and type=Normal —
                    # neither correct — so we override here.
                    power = 70
                    type_id = 0
                else:
                    power = md.base_power
                    # TYPE_TO_IDX keys typeless moves as "???" (Curse); PokemonType
                    # spells that member THREE_QUESTION_MARKS — translate so the type
                    # slot matches the data string the index was built from.
                    type_name = "???" if md.type.name == "THREE_QUESTION_MARKS" else md.type.name
                    type_id = TypeEncoder.TYPE_TO_IDX.get(type_name, 0)
                
                base_idx = i * MOVE_SLOT_DIM
                # 1. Move ID
                vec[base_idx] = float(num)
                # 2. Base Power (Normalized 0-200)
                vec[base_idx + 1] = float(power) / 200.0
                # 3. Secondary Effect Flag
                vec[base_idx + 2] = secondary
                # 4. Recoil Flag
                vec[base_idx + 3] = recoil
                # 5. Type ID
                vec[base_idx + 4] = float(type_id)
                # 6. Category ID (0=Status, 1=Physical, 2=Special) — memoized per move id
                # off the live `move.category` (skips the per-read move.entry/GenData chain).
                vec[base_idx + 5] = _category_val(move)
                
                # 7. Known Flag (Binary) - Now interleaved
                vec[base_idx + 6] = 1.0

                # 8-9. PP: current and max, each normalized by MAX_PP.
                # Encoding both separately lets the model distinguish move spammability (max)
                # from depletion state (current). Opponent moves in Gen 3 always show full PP
                # since Showdown doesn't track opponent PP for Gen 3.
                vec[base_idx + 7] = float(move.current_pp) / MAX_PP
                vec[base_idx + 8] = float(move.max_pp) / MAX_PP

                # 10-11. Accuracy normalized to [0, 1] (raw% / 100) plus a
                # categorical never_miss bit. Never-miss moves carry accuracy=100 in
                # the mapping, so they encode as [1.0, 1] while a genuine
                # 100%-accuracy move encodes as [1.0, 0]: same scalar, distinguished
                # only by the bit. A 100%-accuracy move can still miss into evasion
                # (Double Team) or after Sand-Attack, whereas a never-miss move
                # (Swift, Aerial Ace, all status/self moves) bypasses the
                # accuracy/evasion check entirely — a kind difference, not a 1% one,
                # so it gets its own bit rather than the "101" trick.
                vec[base_idx + 9] = float(md.accuracy) / 100.0
                vec[base_idx + 10] = 1.0 if md.never_miss else 0.0

        return vec

    def get_layout(self) -> Dict[str, Any]:
        return {
            "slots": [{"offset": i * MOVE_SLOT_DIM, "dim": MOVE_SLOT_DIM} for i in range(4)],
            "slot_layout": {
                "id": {"offset": 0, "dim": 1},
                "power": {"offset": 1, "dim": 1},
                "secondary": {"offset": 2, "dim": 1},
                "recoil": {"offset": 3, "dim": 1},
                "type": {"offset": 4, "dim": 1},
                "category": {"offset": 5, "dim": 1},
                "known": {"offset": 6, "dim": 1},
                "current_pp": {"offset": 7, "dim": 1},
                "max_pp": {"offset": 8, "dim": 1},
                "accuracy": {"offset": 9, "dim": 1},
                "never_miss": {"offset": 10, "dim": 1}
            }
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        move_names = []
        for i in range(4):
            if vector[i * MOVE_SLOT_DIM + 6] > 0.5:
                mid = int(vector[i * MOVE_SLOT_DIM])
                name = self.reverse_mapping.get(mid, f"Move({mid})")
                move_names.append(name)
        return {"moves": move_names}
