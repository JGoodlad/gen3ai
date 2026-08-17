import numpy as np
from .base import ObservationEncoder
from .constants import ABILITY_SLOT_DIM, ABILITY_DOMINANCE_DIM, ABILITY_KNOWN_DIM
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Any, Dict, Optional


class AbilitiesEncoder(ObservationEncoder):
    """Encodes the ability block: [ability1_id, ability2_id, dominance, known].

    Priors come from data/pokemon/gen3_ability_priors.json (per-species Smogon
    usage distributions, mirrors the gen3_hidden_power_priors.json pattern).
    Layout:

      - ability1_id / ability2_id : the top 2 Smogon-observed abilities for
        the species, sorted by usage (ability1 is the more common one). For
        single-ability species (Salamence → only Intimidate, Shedinja → only
        Wonder Guard) ability2_id stays 0.
      - dominance ∈ [0, 1] : the Smogon-observed probability share of ability1.
        For known revelations or single-ability species it's 1.0.
      - known flag : 1.0 if the actual ability has been confirmed (own team
        always; opp once an ability message fires), else 0.0.

    Three observable states:
      - Empty slot (mon is None) — all zeros.
      - Revealed (own team, or opp once an ability triggers) —
        [revealed_id, 0, 1.0, 1.0].
      - Opp unrevealed — [top1_id, top2_id_or_0, dominance, 0.0].
    """

    def __init__(
        self,
        ability_to_id: Optional[Dict[str, Any]] = None,
        reverse_mapping: Optional[Dict[int, str]] = None,
        species_to_ability_priors: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        if not ability_to_id:
            raise ValueError("AbilitiesEncoder requires a non-empty ability mapping!")
        self.ability_to_id = ability_to_id
        self.reverse_mapping = reverse_mapping or {}
        # Pre-compute species → (ability1_num, ability2_num, dominance) so the
        # hot path is a single dict lookup. Missing species (no Smogon data) get
        # (0, 0, 0.0).
        self._species_priors: Dict[str, tuple[int, int, float]] = {}
        for sp, ab_probs in (species_to_ability_priors or {}).items():
            if not ab_probs:
                continue
            # Sort by probability descending; ties broken by ability name for
            # determinism across runs.
            ranked = sorted(ab_probs.items(), key=lambda kv: (-kv[1], kv[0]))
            top1_id, top1_p = ranked[0]
            top2_id = ranked[1][0] if len(ranked) > 1 else None
            num1 = int(self.ability_to_id.get(top1_id, {}).get("num", 0))
            num2 = (
                int(self.ability_to_id.get(top2_id, {}).get("num", 0))
                if top2_id is not None else 0
            )
            self._species_priors[sp] = (num1, num2, float(top1_p))

    @property
    def dimension(self) -> int:
        return ABILITY_SLOT_DIM + ABILITY_DOMINANCE_DIM + ABILITY_KNOWN_DIM  # 4

    def _normalize(self, name: str) -> str:
        return name.lower().replace(" ", "").replace("_", "")

    def encode(self, mon: Any, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if mon is None:
            return vec

        ability = mon.ability
        if ability:
            ability_key = self._normalize(ability)
            if ability_key != "unknownability":
                if ability_key not in self.ability_to_id:
                    raise ValueError(
                        f"Unrecognized ability: {ability_key}. "
                        "Update data/pokemon/gen3_abilities.json"
                    )
                vec[0] = float(self.ability_to_id[ability_key].get("num", 0))
                vec[1] = 0.0
                vec[2] = 1.0  # dominance forced to 1.0 — alternative no longer hypothetical
                vec[3] = 1.0  # known
                return vec

        # Opp unrevealed — emit Smogon-derived priors so the model has a head
        # start instead of a flat "unknown" signal.
        species = getattr(mon, "species", None)
        if species:
            priors = self._species_priors.get(species)
            if priors is not None:
                num1, num2, dominance = priors
                vec[0] = float(num1)
                vec[1] = float(num2)
                vec[2] = float(dominance)
        # vec[3] stays 0.0 (not known)
        return vec

    def get_layout(self) -> dict:
        return {
            "id1":       {"offset": 0, "dim": 1},
            "id2":       {"offset": 1, "dim": 1},
            "dominance": {"offset": ABILITY_SLOT_DIM, "dim": ABILITY_DOMINANCE_DIM},
            "known":     {"offset": ABILITY_SLOT_DIM + ABILITY_DOMINANCE_DIM,
                          "dim": ABILITY_KNOWN_DIM},
        }

    # Why the `type: ignore[override]` below — compact-string sub-encoder; see TypeEncoder.describe_vector.
    def describe_vector(self, vector: np.ndarray) -> str:  # type: ignore[override]
        dom = float(vector[ABILITY_SLOT_DIM])
        known = vector[ABILITY_SLOT_DIM + ABILITY_DOMINANCE_DIM] >= 0.5
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
        # Unrevealed: show the candidate set with the dominance probability
        n1 = name(ab1_id)
        n2 = name(ab2_id)
        if ab2_id == 0:
            return f"?({n1} p={dom:.2f})"
        return f"?({n1} p={dom:.2f} | {n2} p={1-dom:.2f})"
