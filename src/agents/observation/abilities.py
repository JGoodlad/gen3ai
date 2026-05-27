import numpy as np
from .base import ObservationEncoder
from .constants import ABILITY_SLOT_DIM, ABILITY_KNOWN_DIM
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Any, Dict, List, Optional


class AbilitiesEncoder(ObservationEncoder):
    """Encodes the ability block: [ability1_id, ability2_id, known_flag].

    Three states the model must distinguish:
      - Empty slot (mon is None): all zeros.
      - Ability revealed (own team always, opp once an ability message fires):
        known=1, ability1=revealed_id, ability2=0.
      - Opp ability not yet revealed: known=0, ability1 + ability2 = the
        species' two dex-possible Gen 3 abilities (from gen3_species.json).
        For single-ability species (Shedinja, Salamence, etc.) ability2=0.
    """

    def __init__(
        self,
        ability_to_id: Optional[Dict[str, Any]] = None,
        reverse_mapping: Optional[Dict[int, str]] = None,
        species_to_abilities: Optional[Dict[str, List[Optional[str]]]] = None,
    ):
        if not ability_to_id:
            raise ValueError("AbilitiesEncoder requires a non-empty ability mapping!")
        self.ability_to_id = ability_to_id
        self.reverse_mapping = reverse_mapping or {}
        # Pre-compute species → (ability_num_1, ability_num_2) for fast lookup.
        # Missing species (or species with no Gen 3-valid abilities) get (0, 0).
        self._species_ability_nums: Dict[str, tuple[int, int]] = {}
        for sp, ab_list in (species_to_abilities or {}).items():
            if not ab_list:
                continue
            nums = []
            for ab_id in ab_list[:2]:
                if ab_id is None:
                    nums.append(0)
                else:
                    nums.append(int(self.ability_to_id.get(ab_id, {}).get("num", 0)))
            while len(nums) < 2:
                nums.append(0)
            self._species_ability_nums[sp] = (nums[0], nums[1])

    @property
    def dimension(self) -> int:
        return ABILITY_SLOT_DIM + ABILITY_KNOWN_DIM  # 2 + 1 = 3

    def _normalize(self, name: str) -> str:
        return name.lower().replace(" ", "").replace("_", "")

    def encode(self, mon: Any, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if mon is None:
            return vec

        ability = mon.ability
        if ability:
            ability_key = self._normalize(ability)
            # poke-env stores "unknownability" as the sentinel for unrevealed
            # opp abilities. Fall through to the species-prior path.
            if ability_key != "unknownability":
                if ability_key not in self.ability_to_id:
                    raise ValueError(
                        f"Unrecognized ability: {ability_key}. "
                        "Update data/pokemon/gen3_abilities.json"
                    )
                vec[0] = float(self.ability_to_id[ability_key].get("num", 0))
                vec[1] = 0.0
                vec[2] = 1.0  # known
                return vec

        # Not revealed — encode the species' two dex-possible abilities so the
        # model has prior knowledge instead of a flat "unknown" signal.
        species = getattr(mon, "species", None)
        if species:
            nums = self._species_ability_nums.get(species)
            if nums is not None:
                vec[0] = float(nums[0])
                vec[1] = float(nums[1])
        # vec[2] stays 0.0 (not known)
        return vec

    def get_layout(self) -> dict:
        return {
            "id1":   {"offset": 0, "dim": 1},
            "id2":   {"offset": 1, "dim": 1},
            "known": {"offset": ABILITY_SLOT_DIM, "dim": ABILITY_KNOWN_DIM},
        }

    def describe_vector(self, vector: np.ndarray) -> str:
        known = vector[ABILITY_SLOT_DIM] >= 0.5
        ab1_id = int(vector[0])
        ab2_id = int(vector[1])
        if ab1_id == 0 and ab2_id == 0:
            return "ABLY-UNKN" if not known else "NONE"

        def name(n: int) -> str:
            if n == 0:
                return ""
            return self.reverse_mapping.get(n, f"Ably({n})").upper()

        if known:
            return name(ab1_id)
        # Unrevealed: show the candidate set
        names = [name(ab1_id)]
        if ab2_id != 0:
            names.append(name(ab2_id))
        return "?(" + "|".join(names) + ")"
