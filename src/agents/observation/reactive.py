import numpy as np
from .base import ObservationEncoder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.side_condition import SideCondition
from .constants import REACTIVE_DIM, TEAM_SIZE
from agents.gen3_mechanics import effective_multiplier_by_types
from typing import Any, Dict, List, Optional, Tuple


def _resolve_ability_distribution(opp, ability_priors):
    """Return a list of (ability_name, probability) pairs describing the
    defender's ability — a singleton for known abilities, the Smogon-derived
    prior distribution for unrevealed opponents, or a sentinel singleton
    `(None, 1.0)` when we have no information (no priors loaded, or species
    absent from the priors file). The `None` sentinel means "pass the mon
    through unchanged" — recovers the pre-fix behaviour without a special case.

    Treats both `None` *and* poke-env's `"unknownability"` sentinel as
    unrevealed — the `AbilitiesEncoder` uses the same gate, so the two
    encoders stay consistent about which mons are "known".
    """
    if opp is None:
        return [(None, 1.0)]
    ability = getattr(opp, "ability", None)
    if ability and ability.lower().replace(" ", "").replace("_", "") != "unknownability":
        return [(ability, 1.0)]
    if ability_priors:
        species = getattr(opp, "species", None)
        if species:
            priors = ability_priors.get(species)
            if priors:
                return list(priors.items())
    return [(None, 1.0)]


def _expected_multiplier(move, attacker, opp, hp_tracker, ability_priors) -> float:
    """Joint-expectation effectiveness handling two independent uncertainties:

    - **Attacker-side**: bare "hiddenpower" with an unrevealed type → the
      `hp_tracker` exposes a 16-dim distribution narrowed by past observations.
    - **Defender-side**: an opponent whose ability hasn't fired yet → the
      Smogon-derived `ability_priors` give the per-species ability distribution.

    Each axis collapses to a singleton when known: own-team typed HP gives a
    1-entry type distribution; revealed opp ability gives a 1-entry ability
    distribution. The joint expectation is therefore identical to the plain
    `effective_multiplier(move.type, opp)` when nothing is uncertain.

    Without `ability_priors` (no opp ability data available, e.g. evaluation
    outside training), the defender path falls through to the live `mon.ability`
    → preserves legacy behaviour for any caller that has never been wired up.
    """
    # Attacker-side: enumerate possible move types
    if (
        move.id == "hiddenpower"
        and hp_tracker is not None
        and attacker is not None
    ):
        from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER
        probs = hp_tracker.get_probs(attacker.species)
        if probs is not None and probs.sum() > 0:
            type_dist = [
                (HIDDEN_POWER_TYPE_ORDER[i], float(probs[i]))
                for i in range(16) if probs[i] > 0
            ]
        else:
            type_dist = [(move.type, 1.0)]
    else:
        type_dist = [(move.type, 1.0)]

    # Defender-side: enumerate possible abilities
    ability_dist = _resolve_ability_distribution(opp, ability_priors)

    # Read the defender's type/status/ability ONCE here, then feed values to the memoized
    # primitive. Previously each (type, ability) term wrapped `opp` in an _AbilityOverrideMon
    # whose __getattr__ re-resolved type_1/type_2/status through poke-env per term — the
    # proxy + property thrash dominated the matchup-encoder profile. `ability is None` is the
    # "pass through unchanged" sentinel → use the defender's real ability.
    d1, d2 = opp.type_1, opp.type_2
    dstatus = getattr(opp, "status", None)
    opp_ability = getattr(opp, "ability", None)

    # Joint expectation
    total = 0.0
    for move_type, p in type_dist:
        for ability, q in ability_dist:
            ab = opp_ability if ability is None else ability
            total += p * q * effective_multiplier_by_types(move_type, d1, d2, ab, dstatus)
    return total


class ReactiveEncoder(ObservationEncoder):
    """
    Encodes reactive features:
    - Base Power of 4 active moves (4)
    - Damage multipliers of 4 active moves (4)
    - Fainted counts (2)
    - Status flag (1)
    - Forced Struggle flag (1)
    - Matchup Matrix: Our moves vs Their mons (144)
    - Matchup Matrix: Their moves vs Our mons (144)
    Total: 300 dims
    (HP and Spikes removed — duplicated in per-Pokémon vector and global env respectively)
    """
    
    def __init__(self, ability_priors: Optional[Dict[str, Dict[str, float]]] = None):
        """ability_priors: Smogon-derived per-species ability distributions
        keyed by lowercase species name → {ability_id: probability}. Used to
        compute expected matchup effectiveness against opponents whose ability
        hasn't yet fired. None recovers legacy behaviour (no ability expectation —
        equivalent to assuming the live `mon.ability` is authoritative).
        """
        self._ability_priors = ability_priors or {}

    @property
    def dimension(self) -> int:
        return REACTIVE_DIM

    def encode(self, battle: AbstractBattle, hp_tracker=None, live=None) -> np.ndarray:
        """Encode the 300-dim reactive block.

        live: optional :class:`~agents.battle.live_view.LiveView` snapshot for this
        decision. When supplied, the structural reads — per-side fainted counts and the
        active mon's status flag — are taken through the read-model; ``None`` (unit-test /
        plain-Battle path) falls back to the raw battle, byte-identically.

        **Why the effectiveness core stays on the raw battle.** The active-move loop and the
        two 6×4×6 matchup matrices read moves off the raw poke-env ``Move`` objects
        (``battle.available_moves`` / ``get_sorted_moves``) and the defender's
        ``type_1``/``type_2``/``ability``/``status``. They are *not* migrated to the read-model
        on purpose, for three independent reasons:
          1. *Typed Hidden Power id.* Our own HP move keeps its typed id (``hiddenpowerfire``)
             only on the raw ``Move`` object; the live request — and therefore ``LiveView`` /
             ``LegalActions`` — re-keys it to bare ``hiddenpower``. Reading the move list off the
             read-model would silently collapse own HP to Normal/tracker, changing the emitted
             effectiveness. (See the matching note in ``pokemon.py``.)
          2. *Hot path needs enums.* ``effective_multiplier_by_types`` is keyed on
             ``PokemonType`` enums; ``LiveView`` exposes lowercased *strings*, so feeding it
             would add a per-cell string→enum conversion to the #1 obs hot loop (288 cells).
          3. *Pinned by the alignment test.* ``alignment_test`` asserts each matrix cell carries
             its move's ``move.type`` effectiveness — i.e. the move-object type, not a dex lookup.

        hp_tracker: optional HiddenPowerTracker. When supplied, matchup scalars
        for bare "hiddenpower" move slots are replaced with the expected
        effectiveness across the tracker's narrowed distribution. Without it,
        bare HP slots fall back to poke-env's default move.type (typically
        Normal), preserving legacy behaviour when called outside training.

        Ability priors are taken from `self._ability_priors` (set in __init__)
        and applied to every matchup cell — for opponents with an unrevealed
        ability, the cell is the expected effectiveness across the prior
        distribution; for revealed/own-team mons it collapses to the exact
        value via the proxy in `_expected_multiplier`.
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        if battle is None:
            return vec

        # 1. Active Moves (Power and Multiplier) — see the docstring: kept on the raw battle
        # so own Hidden Power retains its typed id and the hot loop stays enum-keyed.
        moves_base_power = np.zeros(4)
        moves_dmg_multiplier = np.ones(4)

        # Skip Struggle — it has a dedicated action (10) and a dedicated flag (vec[15]).
        # Filling the move slots with Struggle's stats would create a confusing alias
        # between slot 0 (action 6) and the Struggle action (10).
        is_forced_struggle = (
            len(battle.available_moves) == 1
            and battle.available_moves[0].id == "struggle"
        )

        if not is_forced_struggle:
            for i, move in enumerate(battle.available_moves):
                if i >= 4:
                    break
                moves_base_power[i] = move.base_power / 200.0
                if battle.opponent_active_pokemon is not None:
                    mult = _expected_multiplier(
                        move,
                        battle.active_pokemon,
                        battle.opponent_active_pokemon,
                        hp_tracker,
                        self._ability_priors,
                    )
                    moves_dmg_multiplier[i] = mult / 4.0

        vec[0:4] = moves_base_power
        vec[4:8] = moves_dmg_multiplier

        # 2. Fainted Counts — read through the LiveView (m.fainted) when available, else raw.
        if live is not None:
            fainted_mon_team = sum(1 for m in live.ours.mons if m.fainted) / 6.0
            fainted_mon_opponent = sum(1 for m in live.opp.mons if m.fainted) / 6.0
        else:
            fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6.0
            fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6.0
        vec[8] = fainted_mon_team
        vec[9] = fainted_mon_opponent

        # 3. Status — active mon currently has a status condition (HP and Spikes removed —
        # available in per-Pokémon vector and global env). Read through the LiveView's active
        # slot when available; both paths resolve the active off battle.active_pokemon, so the
        # truthiness is identical.
        if live is not None:
            active_live = live.ours.active
            vec[10] = 1.0 if (active_live is not None and active_live.status) else 0.0
        else:
            vec[10] = 1.0 if battle.active_pokemon and battle.active_pokemon.status else 0.0

        # 4. Forced Struggle
        vec[11] = 1.0 if is_forced_struggle else 0.0

        # --- Matchup Matrices (raw battle — see the docstring's three reasons) ---
        our_team = self.get_team_list(battle, is_opponent=False)
        their_team = self.get_team_list(battle, is_opponent=True)

        # 5. Our moves vs Their mons (144 dims)
        cursor = 12
        for i in range(TEAM_SIZE):
            our_mon = our_team[i] if i < len(our_team) else None
            our_moves = self.get_sorted_moves(our_mon)
            for move_idx in range(4):
                move = our_moves[move_idx] if move_idx < len(our_moves) else None
                for j in range(TEAM_SIZE):
                    their_mon = their_team[j] if j < len(their_team) else None
                    if move and their_mon:
                        # Normalize by 4.0 to keep values in [0, 1] range for better MLP convergence
                        vec[cursor] = _expected_multiplier(
                            move, our_mon, their_mon, hp_tracker, self._ability_priors
                        ) / 4.0
                    cursor += 1

        # 6. Their moves vs Our mons (144 dims)
        for i in range(TEAM_SIZE):
            their_mon = their_team[i] if i < len(their_team) else None
            their_moves = self.get_sorted_moves(their_mon)
            for move_idx in range(4):
                move = their_moves[move_idx] if move_idx < len(their_moves) else None
                for j in range(TEAM_SIZE):
                    our_mon = our_team[j] if j < len(our_team) else None
                    if move and our_mon:
                        # Normalize by 4.0 to keep values in [0, 1] range for better MLP convergence
                        vec[cursor] = _expected_multiplier(
                            move, their_mon, our_mon, hp_tracker, self._ability_priors
                        ) / 4.0
                    cursor += 1
        
        return vec

    def get_layout(self) -> Dict[str, Any]:
        return {
            "move_power": {"offset": 0, "dim": 4},
            "move_multiplier": {"offset": 4, "dim": 4},
            "fainted": {"offset": 8, "dim": 2},
            "active_status": {"offset": 10, "dim": 1},
            "forced_struggle": {"offset": 11, "dim": 1},
            "our_matchups": {"offset": 12, "dim": 144},
            "their_matchups": {"offset": 156, "dim": 144}
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        # Extract matrices and scale back up by 4.0 for human-readable display
        our_m = vector[12:156].reshape(TEAM_SIZE, 4, TEAM_SIZE) * 4.0
        their_m = vector[156:300].reshape(TEAM_SIZE, 4, TEAM_SIZE) * 4.0
        
        return {
            "fainted_our": int(vector[8] * 6),
            "fainted_opp": int(vector[9] * 6),
            "active_move_mults": [f"{m*4.0:.1f}x" for m in vector[4:8].tolist()],
            "struggle": bool(vector[11]),
            "our_vs_their": our_m, # Full matrix for deeper trace
            "their_vs_our": their_m
        }
