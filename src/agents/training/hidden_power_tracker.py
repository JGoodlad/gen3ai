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

import numpy as np

from agents.enums import PokemonType
from agents.gen3_mechanics import bucket_effectiveness, effective_multiplier

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
        elif priors_path is not None:
            with open(priors_path) as f:
                self._priors = json.load(f)
        else:
            # Default: the project's HP-type usage priors, via the gen3_data facade (parsed once
            # and shared, instead of re-read from disk per episode).
            from agents.gen3_data import priors as _gen3_priors
            self._priors = _gen3_priors.hidden_power_raw()
        self._state: dict[str, np.ndarray] = {}
        # Species whose moveset has been fully revealed without Hidden Power —
        # we know definitively they cannot use HP this episode.
        self._ruled_out: set[str] = set()
        # Count of observations DISCARDED by the caller's is_feasible() guard — i.e. no HP type at
        # all could have produced the reported effectiveness against the resolved target, so the
        # target identification was wrong. Counted rather than silent: this should sit at ~0, and a
        # rising count is the signal that target resolution is drifting (no silent caps).
        self._infeasible_observations: int = 0
        # Per-species observation log — kept so a ValueError dump can show
        # exactly which earlier observations narrowed the candidate set to nothing.
        self._obs_log: dict[str, list] = {}
        # gen3_hp_prior_support_v1: the same observations as REPLAYABLE (effectiveness, target)
        # pairs, so a usage prior the evidence contradicts can be replaced by the flat prior and the
        # whole log re-applied (see `observe`).
        self._obs_targets: dict[str, list] = {}
        # Species whose usage-prior row was CONTRADICTED by their own observations (every type the
        # row gives mass was eliminated while some Hidden Power type still explains them all) and so
        # was replaced by the flat prior. Counted, never silent: on the training pool this sits at 0.
        self._prior_discarded: set[str] = set()
        # gen3_obs_assembler_v1: a monotone counter bumped by EVERY writer of the state
        # `get_probs` / `is_known` read. The obs assembler caches an opponent's 122-dim slot
        # (17 of which are this tracker's HP-candidate block) and needs to know when the block
        # moved. Narrowing is triggered by protocol events, so an event-only dirty rule would
        # *usually* be right — and "usually" is exactly the shape of a silent-staleness bug, so
        # the tracker states it outright instead. See `revision`.
        self._revision: int = 0

    def observe(self, species: str, effectiveness: float, target_mon) -> None:
        """Filter the candidate distribution for species after HP hits target_mon.

        effectiveness must be 0.0, 0.5, 1.0, or 2.0 — the observed damage multiplier.

        **A usage prior's zero is not an impossibility** (`gen3_hp_prior_support_v1`). A species'
        row in gen3_hidden_power_priors.json is Smogon USAGE: a type nobody used there has mass 0.0,
        but every one of the 16 types is a legal Hidden Power for every user — the type is the IVs'
        (`sim/dex.ts` `getHiddenPower`), and an unset IV is 31 (`sim/pokemon.ts:387-394`), so a set
        that states no IVs is HP DARK. When this species' observations eliminate every type its row
        gives mass while SOME type still explains them all, the row is what the evidence refuted:
        the species restarts from the flat 1/16 prior and its whole observation log is re-applied
        (counted in :attr:`prior_discarded`). Measured: 0 of the training pool's 1,912 HP users carry
        a zero-prior type; 289 of the ladder corpus's 52,007 do (all HP Dark).

        Raises ValueError if the observations eliminate every candidate even under the flat prior —
        no Hidden Power type explains them, so a target or an effectiveness was misattributed: a
        tracker bug, and it stays loud.
        """
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

        self._narrow(self._state[species], effectiveness, target_mon)

        self._revision += 1

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
        self._obs_targets.setdefault(species, []).append((effectiveness, target_mon))

        if np.any(self._state[species]):
            return
        if self._priors.get(species) and species not in self._prior_discarded:
            # gen3_hp_prior_support_v1: the usage row excluded the true type — re-run the whole
            # log from the flat prior (every legal type).
            flat = np.full(16, _FLAT_PRIOR, dtype=np.float32)
            for eff, tgt in self._obs_targets[species]:
                self._narrow(flat, eff, tgt)
            if np.any(flat):
                self._state[species] = flat
                self._prior_discarded.add(species)
                return
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
        raise ValueError(
            f"HiddenPowerTracker: all candidates eliminated for '{species}' "
            f"after observing {effectiveness}× on {mon_desc}. No Hidden Power type explains "
            f"this species' observations even under the flat prior — a tracker bug "
            f"(a misattributed target or effectiveness).\n"
            f"Observation log for '{species}':\n  {history}"
        )

    @staticmethod
    def _narrow(vec: np.ndarray, effectiveness: float, target_mon) -> None:
        """Zero every candidate of ``vec`` that could not produce ``effectiveness`` on ``target_mon``.

        poke-env reports effectiveness in the {0, 0.5, 1, 2} buckets — a 4× SE hit on a quad-weak
        target reads as 2.0, not 4.0 — so the computed multipliers are bucketed the same way."""
        for i, hidden_power_type in enumerate(HIDDEN_POWER_TYPE_ORDER):
            if vec[i] != 0.0:
                calc = bucket_effectiveness(
                    effective_multiplier(hidden_power_type, target_mon)
                )
                if calc != effectiveness:
                    vec[i] = 0.0

    def is_feasible(self, effectiveness: float, target_mon) -> bool:
        """Return True if at least one HP type produces this effectiveness against target_mon.

        **Called by default before every observe()** (`EpisodeTracker._maybe_observe_hidden_power`,
        gen3_typed_hp_belief_v1) to skip observations where the target identification is suspect
        (e.g. a switch resolved on our side in the same window). If NO HP type can produce the
        observed effectiveness, the target is wrong — the observation is junk about a mon that was
        never hit, so it is discarded (and counted, see :attr:`infeasible_observations`) rather
        than narrowing on it or raising.

        Note the deliberate asymmetry with :meth:`observe`. An observation that is feasible in
        general but eliminates every candidate THIS species' usage prior supports refutes the PRIOR
        (a type nobody used on Smogon is still legal): the species falls back to the flat prior and
        its log is replayed (`gen3_hp_prior_support_v1`). Only a log that NO Hidden Power type
        explains — each observation feasible alone, their conjunction not — still RAISES: that is a
        genuine contradiction (a tracker bug), and it must stay loud.
        """
        return any(
            bucket_effectiveness(effective_multiplier(hp_type, target_mon)) == effectiveness
            for hp_type in HIDDEN_POWER_TYPE_ORDER
        )

    def note_infeasible(self) -> None:
        """Record that the caller's :meth:`is_feasible` guard discarded an observation."""
        self._infeasible_observations += 1

    @property
    def revision(self) -> int:
        """Monotone counter of state changes visible through :meth:`get_probs` /
        :meth:`is_known` (`gen3_obs_assembler_v1`).

        Bumped by :meth:`observe` and :meth:`mark_no_hp` — the only two writers — and reset to 0
        by :meth:`reset`. A cache holding an encoded opponent slot compares this instead of
        inferring "the HP block cannot have moved" from the event stream.
        """
        return self._revision

    @property
    def prior_discarded(self) -> frozenset:
        """Species whose usage-prior row their own observations refuted, now on the flat prior
        (`gen3_hp_prior_support_v1`; expected empty on the training pool)."""
        return frozenset(self._prior_discarded)

    @property
    def infeasible_observations(self) -> int:
        """How many observations the feasibility guard discarded this episode (expected ~0)."""
        return self._infeasible_observations

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
        if species not in self._ruled_out:
            self._ruled_out.add(species)
            self._revision += 1

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
        self._obs_targets.clear()
        self._prior_discarded.clear()
        self._infeasible_observations = 0
        self._revision = 0
