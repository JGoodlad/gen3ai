"""
Per-species Hidden Power candidate distribution tracker.

Maintains a (16,) probability vector per opponent species. On construction the vector
is loaded from competitive usage priors (gen3_hidden_power_priors.json). Each time the
opponent uses Hidden Power and we observe the effectiveness tier, types incompatible
with that observation are zeroed out.

The 16 types are indexed in fixed alphabetical order (HIDDEN_POWER_TYPE_ORDER); this
matches the 16-dim block in the observation encoding.
"""
from __future__ import annotations

import json
import os

import numpy as np
from poke_env.battle.pokemon_type import PokemonType

from agents.gen3_mechanics import effective_multiplier
from utils.git import get_repo_root

# Fixed alphabetical order — index is canonical across tracker, encoder, and design doc
HIDDEN_POWER_TYPE_ORDER: list[PokemonType] = [
    PokemonType.BUG,
    PokemonType.DARK,
    PokemonType.DRAGON,
    PokemonType.ELECTRIC,
    PokemonType.FIGHTING,
    PokemonType.FIRE,
    PokemonType.FLYING,
    PokemonType.GHOST,
    PokemonType.GRASS,
    PokemonType.GROUND,
    PokemonType.ICE,
    PokemonType.POISON,
    PokemonType.PSYCHIC,
    PokemonType.ROCK,
    PokemonType.STEEL,
    PokemonType.WATER,
]

_PRIOR_TYPE_NAMES: list[str] = [t.name.lower() for t in HIDDEN_POWER_TYPE_ORDER]

_FLAT_PRIOR = 1.0 / 16


class HiddenPowerTracker:
    """
    Tracks per-species Hidden Power candidate distributions within one episode.

    Call observe() each time the opponent uses Hidden Power and the effectiveness
    tier is known. Types incompatible with the observed effectiveness are zeroed
    out of the candidate vector.

    get_probs(species) returns a (16,) float32 array. All-zero means HP has not yet
    been observed for that species this episode.
    """

    def __init__(
        self,
        priors_path: str | None = None,
        _priors: dict | None = None,
    ) -> None:
        if _priors is not None:
            self._priors: dict = _priors
        else:
            if priors_path is None:
                priors_path = os.path.join(
                    get_repo_root(), "data", "pokemon", "gen3_hidden_power_priors.json"
                )
            with open(priors_path) as f:
                self._priors = json.load(f)
        self._state: dict[str, np.ndarray] = {}

    def observe(self, species: str, effectiveness: float, target_mon) -> None:
        """Filter the candidate distribution for species after HP hits target_mon.

        effectiveness must be 0.0, 0.5, 1.0, or 2.0 — the observed damage multiplier.

        Raises ValueError if the observation eliminates all candidates. This should
        never happen in correct operation; it indicates either a tracker bug (when the
        species has a prior entry) or a missing entry in gen3_hidden_power_priors.json
        (data gap).
        """
        had_prior_entry = species in self._priors

        if species not in self._state:
            prior_dict = self._priors.get(species, {})
            if prior_dict:
                vec = np.array(
                    [prior_dict.get(name, 0.0) for name in _PRIOR_TYPE_NAMES],
                    dtype=np.float32,
                )
            else:
                vec = np.full(16, _FLAT_PRIOR, dtype=np.float32)
            self._state[species] = vec

        for i, hidden_power_type in enumerate(HIDDEN_POWER_TYPE_ORDER):
            if self._state[species][i] != 0.0:
                if effective_multiplier(hidden_power_type, target_mon) != effectiveness:
                    self._state[species][i] = 0.0

        if not np.any(self._state[species]):
            mon_desc = (
                f"{getattr(target_mon, 'species', '?')} "
                f"(type1={getattr(target_mon, 'type_1', '?')}, "
                f"type2={getattr(target_mon, 'type_2', '?')}, "
                f"ability={getattr(target_mon, 'ability', '?')})"
            )
            if had_prior_entry:
                raise ValueError(
                    f"HiddenPowerTracker: all candidates eliminated for '{species}' "
                    f"after observing {effectiveness}× on {mon_desc}. "
                    f"Species has a prior entry — this is likely a tracker bug."
                )
            else:
                raise ValueError(
                    f"HiddenPowerTracker: all candidates eliminated for '{species}' "
                    f"after observing {effectiveness}× on {mon_desc}. "
                    f"Species has no prior entry (flat 1/16 used) — data gap: "
                    f"add '{species}' to gen3_hidden_power_priors.json."
                )

    def is_feasible(self, effectiveness: float, target_mon) -> bool:
        """Return True if at least one HP type produces this effectiveness against target_mon.

        Use this guard before calling observe() to skip observations where the target
        identification is suspect (e.g. a switch occurred on our side in the same turn).
        If no HP type can possibly produce the observed effectiveness, the target is wrong
        and the observation should be discarded rather than raising ValueError.
        """
        return any(
            effective_multiplier(hp_type, target_mon) == effectiveness
            for hp_type in HIDDEN_POWER_TYPE_ORDER
        )

    def get_probs(self, species: str) -> np.ndarray:
        """Return (16,) float32 candidate probability vector for species.

        All-zero if HP has not been observed for this species this episode.
        """
        if species in self._state:
            return self._state[species].copy()
        return np.zeros(16, dtype=np.float32)

    def reset(self) -> None:
        self._state.clear()
