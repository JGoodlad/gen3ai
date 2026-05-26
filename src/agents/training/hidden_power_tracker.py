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

from agents.gen3_mechanics import bucket_effectiveness, effective_multiplier
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
        # Species whose moveset has been fully revealed without Hidden Power —
        # we know definitively they cannot use HP this episode.
        self._ruled_out: set[str] = set()
        # Per-species observation log — kept so a ValueError dump can show
        # exactly which earlier observations narrowed the candidate set to nothing.
        self._obs_log: dict[str, list] = {}

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

        # poke-env reports effectiveness in the {0, 0.5, 1, 2} buckets — a 4× SE
        # hit on a quad-weak target reads as 2.0, not 4.0. Bucket our computed
        # multipliers the same way before comparing.
        for i, hidden_power_type in enumerate(HIDDEN_POWER_TYPE_ORDER):
            if self._state[species][i] != 0.0:
                calc = bucket_effectiveness(
                    effective_multiplier(hidden_power_type, target_mon)
                )
                if calc != effectiveness:
                    self._state[species][i] = 0.0

        # Record this observation for the diagnostic dump below (and for callers
        # introspecting the narrowing history).
        self._obs_log.setdefault(species, []).append((
            float(effectiveness),
            getattr(target_mon, "species", "?"),
            str(getattr(target_mon, "type_1", "?")),
            str(getattr(target_mon, "type_2", None)),
            getattr(target_mon, "ability", None),
            str(getattr(target_mon, "status", None)),
        ))

        if not np.any(self._state[species]):
            mon_desc = (
                f"{getattr(target_mon, 'species', '?')} "
                f"(type1={getattr(target_mon, 'type_1', '?')}, "
                f"type2={getattr(target_mon, 'type_2', '?')}, "
                f"ability={getattr(target_mon, 'ability', '?')}, "
                f"status={getattr(target_mon, 'status', None)})"
            )
            history = "\n  ".join(
                f"eff={e}× target={s} types=({t1}/{t2}) ability={a} status={st}"
                for (e, s, t1, t2, a, st) in self._obs_log[species]
            )
            why = (
                "Species has a prior entry — this is likely a tracker bug."
                if had_prior_entry
                else "Species has no prior entry (flat 1/16 used) — data gap: "
                     f"add '{species}' to gen3_hidden_power_priors.json."
            )
            raise ValueError(
                f"HiddenPowerTracker: all candidates eliminated for '{species}' "
                f"after observing {effectiveness}× on {mon_desc}. {why}\n"
                f"Observation log for '{species}':\n  {history}"
            )

    def is_feasible(self, effectiveness: float, target_mon) -> bool:
        """Return True if at least one HP type produces this effectiveness against target_mon.

        Use this guard before calling observe() to skip observations where the target
        identification is suspect (e.g. a switch occurred on our side in the same turn).
        If no HP type can possibly produce the observed effectiveness, the target is wrong
        and the observation should be discarded rather than raising ValueError.
        """
        return any(
            bucket_effectiveness(effective_multiplier(hp_type, target_mon)) == effectiveness
            for hp_type in HIDDEN_POWER_TYPE_ORDER
        )

    def get_probs(self, species: str) -> np.ndarray:
        """Return (16,) float32 candidate probability vector for species.

        All-zero if HP has not been observed for this species this episode,
        or if the species was marked ruled out via mark_no_hp().
        """
        if species in self._ruled_out:
            return np.zeros(16, dtype=np.float32)
        if species in self._state:
            return self._state[species].copy()
        return np.zeros(16, dtype=np.float32)

    def mark_no_hp(self, species: str) -> None:
        """Mark a species as definitively lacking Hidden Power in its moveset.

        Called when an opponent has revealed all 4 moves and none is Hidden Power.
        After this call, is_known(species) returns True and get_probs(species)
        returns all-zero, signalling the model that HP is impossible (vs. the
        all-zero default meaning "haven't observed yet").
        """
        self._ruled_out.add(species)

    def is_known(self, species: str) -> bool:
        """True if we have made a determination about this species' HP — either
        narrowed via observation, or ruled out via mark_no_hp().

        The encoder uses this to decide whether to set hp_revealed=1.0; the
        probability vector itself may be sparse (narrowed) or all-zero (ruled out).
        """
        return species in self._ruled_out or species in self._state

    def reset(self) -> None:
        self._state.clear()
        self._ruled_out.clear()
        self._obs_log.clear()
