import numpy as np
from .base import ObservationEncoder
from poke_env.battle.abstract_battle import AbstractBattle
from .constants import (
    REACTIVE_DIM, TEAM_SIZE,
    REACTIVE_SCALAR_DIM, REACTIVE_MATCHUP_OFFSET,
    MOVE_EFFECTS_DIM, MOVE_EFFECT_FEATURES,
    INCOMING_DMG_DIM, INCOMING_DMG_OFFSET,
)
from agents.enums import PokemonType
from agents import gen3_data
from agents.gen3_mechanics import effective_multiplier_by_types, status_land_estimate
from agents.observation.incoming_damage_encoder import encode_block as encode_incoming_block
from typing import Any, Dict, List, Optional


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


def _attacker_type_dist(move, attacker, hp_tracker):
    """Attacker-side type distribution for one (move, attacker) pair.

    A singleton ``[(move.type, 1.0)]`` for typed moves (own typed Hidden Power and
    every normal move); for a bare ``hiddenpower`` whose type is unrevealed, the
    ``hp_tracker``'s 16-dim distribution narrowed by past observations. Constant
    across every defender the move can hit, so the matchup encoder computes it once
    per (attacker, move) and reuses it down the inner defender loop.
    """
    if (
        move.id == "hiddenpower"
        and hp_tracker is not None
        and attacker is not None
    ):
        from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER
        probs = hp_tracker.get_probs(attacker.species)
        if probs is not None and probs.sum() > 0:
            return [
                (HIDDEN_POWER_TYPE_ORDER[i], float(probs[i]))
                for i in range(16) if probs[i] > 0
            ]
    return [(move.type, 1.0)]


def _defender_terms(opp, ability_priors):
    """Defender-side terms for one mon: ``(type_1, type_2, status, ability,
    ability_dist)``.

    These are constant for a given defender across every attacker/move that targets
    it, so the matchup encoder computes them ONCE per team mon and reuses them for
    all attacker-move combinations — instead of re-reading the poke-env
    ``type_1``/``type_2``/``status``/``ability`` properties and rebuilding the
    ability distribution on every one of the 144 matrix cells. `ability is None`
    inside the distribution is the "pass through unchanged" sentinel → use the
    defender's real ability.
    """
    ability_dist = _resolve_ability_distribution(opp, ability_priors)
    d1, d2 = opp.type_1, opp.type_2
    dstatus = getattr(opp, "status", None)
    opp_ability = getattr(opp, "ability", None)
    return d1, d2, dstatus, opp_ability, ability_dist


def _joint_expectation(type_dist, terms) -> float:
    """Joint effectiveness expectation over the attacker type distribution and the
    defender ability distribution.

    Byte-identical to the pre-hoist ``_expected_multiplier`` body: same iteration
    order (types outer, abilities inner) and the same
    ``p * q * effective_multiplier_by_types(...)`` accumulation. The memoized
    primitive is read once per (type, ability) term.
    """
    d1, d2, dstatus, opp_ability, ability_dist = terms
    total = 0.0
    for move_type, p in type_dist:
        for ability, q in ability_dist:
            ab = opp_ability if ability is None else ability
            total += p * q * effective_multiplier_by_types(move_type, d1, d2, ab, dstatus)
    return total


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

    Thin single-cell wrapper over the hoistable primitives (`_attacker_type_dist` +
    `_defender_terms` → `_joint_expectation`). The matchup matrices call those
    directly so the per-defender / per-(attacker,move) reads happen once instead of
    per cell; this wrapper is retained for the active-move slot loop and the unit
    tests, which need the single-cell entry point.
    """
    return _joint_expectation(
        _attacker_type_dist(move, attacker, hp_tracker),
        _defender_terms(opp, ability_priors),
    )


class ReactiveEncoder(ObservationEncoder):
    """
    Encodes reactive features:
    - Base Power of 4 active moves (4)
    - Damage multipliers of 4 active moves (4)
    - Fainted counts (2)
    - Status flag (1)
    - Forced Struggle flag (1)
    - Trapped flag (1)        — server-authoritative `legal.trapped` (cannot switch)
    - Maybe-trapped flag (1)  — server-authoritative `legal.maybe_trapped` (opponent MIGHT trap)
    - Move-effect flags (36)  — gen3_move_effects_v1: 4 request-order move slots × 9 flags
      [is_boost, is_heal, is_protect, is_phaze, is_hazard, inflicts_status,
      status_will_land, pp_fraction, status_will_land_known] so the policy head can tell a
      setup move from a heal from a wasted status (otherwise indistinguishable: base power 0 +
      neutral multiplier). status_will_land is a prior-weighted probability; the trailing
      *_known bit flags confirmed-vs-prior, mirroring the ability block's `known` flag.
    - Matchup Matrix: Our moves vs Their mons (144)
    - Matchup Matrix: Their moves vs Our mons (144)
    Total: 338 dims
    (HP and Spikes removed — duplicated in per-Pokémon vector and global env respectively)

    The trapped / maybe_trapped bits (gen3_trapping_signals_v1) are sourced from the
    per-decision :class:`~agents.battle.live_view.LegalActions` snapshot (``legal``), the same
    server-authoritative surface the action mask is built from. ``trapped`` is redundant with
    the mask (which already zeroes the switch bits), but giving the policy/value nets an
    explicit feature beats forcing them to infer "can't switch" from masked logits.
    ``maybe_trapped`` carries genuinely NEW information: the switch bits stay legal (correct —
    we don't KNOW we're trapped), so without this bit the model attempts the pivot blind and
    eats a server rejection; with it, it can learn "switching here is risky — the opponent
    might be Dugtrio/Arena Trap" and weigh the pivot.
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

    def encode(self, battle: AbstractBattle, hp_tracker=None, live=None, legal=None,
               progress_clock=None) -> np.ndarray:
        """Encode the reactive block.

        live: optional :class:`~agents.battle.live_view.LiveView` snapshot for this
        decision. When supplied, the structural reads — per-side fainted counts and the
        active mon's status flag — are taken through the read-model; ``None`` (unit-test /
        plain-Battle path) falls back to the raw battle, byte-identically.

        legal: optional :class:`~agents.battle.live_view.LegalActions` snapshot for this
        decision (server-authoritative legality, same surface the mask is built from). Its
        ``trapped`` / ``maybe_trapped`` flags become two obs bits (vec[12], vec[13]). ``None``
        (unit-test / plain-Battle path, or a caller that hasn't threaded it) leaves both at 0.

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
        # gen3_move_effects_v1: per-move EFFECT flags, request-order (slot i ↔ action 6+i),
        # so the policy head can tell a setup move from a heal from a wasted status. Static
        # flags come from the data facade (MoveData); status_will_land and Curse's
        # type-conditional setup are resolved LIVE here.
        move_effects = np.zeros((4, MOVE_EFFECT_FEATURES), dtype=np.float32)

        # Skip Struggle — it has a dedicated action (10) and a dedicated flag (vec[11]).
        # Filling the move slots with Struggle's stats would create a confusing alias
        # between slot 0 (action 6) and the Struggle action (10).
        is_forced_struggle = (
            len(battle.available_moves) == 1
            and battle.available_moves[0].id == "struggle"
        )

        active = battle.active_pokemon
        opp = battle.opponent_active_pokemon
        # Curse is a setup move (+atk/+def/-spe) ONLY for a non-Ghost user; for a Ghost
        # user it is a self-HP-cost trap. Resolve from the live user's type (the static
        # MoveData.is_boost is False for Curse precisely because it is type-conditional).
        user_is_ghost = active is not None and PokemonType.GHOST in (active.type_1, active.type_2)

        if not is_forced_struggle:
            for i, move in enumerate(battle.available_moves):
                if i >= 4:
                    break
                moves_base_power[i] = move.base_power / 200.0
                if opp is not None:
                    mult = _expected_multiplier(
                        move, active, opp, hp_tracker, self._ability_priors,
                    )
                    moves_dmg_multiplier[i] = mult / 4.0

                eff = move_effects[i]
                md = gen3_data.moves.get(move.id)
                if md is not None:
                    eff[0] = 1.0 if (md.is_boost or (move.id == "curse" and not user_is_ghost)) else 0.0
                    eff[1] = 1.0 if md.is_heal else 0.0
                    eff[2] = 1.0 if md.is_protect else 0.0
                    eff[3] = 1.0 if md.is_phaze else 0.0
                    eff[4] = 1.0 if md.is_hazard else 0.0
                    if md.status_inflicted is not None:
                        eff[5] = 1.0  # inflicts_status (it IS a status move)
                        if opp is not None:
                            # status_will_land (6): prior-weighted probability in [0,1] — priors
                            # first (Smogon ability distribution for an unrevealed opp), then
                            # collapses to 0/1 once the ability is revealed. Same ability-
                            # distribution source the matchup cells use (`_resolve_ability_
                            # distribution`). status_will_land_known (8): the prior-vs-confirmed
                            # flag, routed with the SAME predicate as the per-mon ability block's
                            # `known` bit (revealed ability OR a type-certain hard block) so the
                            # model can tell a confirmed outcome from a Smogon-prior estimate —
                            # parity with how abilities are routed.
                            ability_dist = _resolve_ability_distribution(opp, self._ability_priors)
                            eff[6], known = status_land_estimate(
                                move.id, md.status_inflicted, opp, ability_dist
                            )
                            eff[8] = 1.0 if known else 0.0
                # pp_fraction: this move's own remaining PP (depletion → Struggle awareness).
                max_pp = getattr(move, "max_pp", 0) or 0
                eff[7] = (move.current_pp / max_pp) if max_pp else 0.0

        vec[0:4] = moves_base_power
        vec[4:8] = moves_dmg_multiplier
        vec[REACTIVE_SCALAR_DIM:REACTIVE_SCALAR_DIM + MOVE_EFFECTS_DIM] = move_effects.reshape(-1)

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

        # 4b. Trapping signals (gen3_trapping_signals_v1) — server-authoritative legality.
        # trapped: confirmed cannot switch (Mean Look / Arena Trap / Magnet Pull revealed) —
        # redundant with the mask but an explicit feature. maybe_trapped: the opponent MIGHT
        # be trapping us (switches still legal in the mask) — the highest-value bit, the only
        # signal here the model has no other way to see.
        if legal is not None:
            vec[12] = 1.0 if legal.trapped else 0.0
            vec[13] = 1.0 if legal.maybe_trapped else 0.0

        # 4d. turns_since_progress (gen3_markovian_progress_v1, vec[14]) — the log-saturated
        # no-progress clock (design §5.1). Sourced from the EpisodeTracker-owned ProgressClock
        # (NOT LiveView — it is cross-turn state), threaded in like the HP tracker; None on the
        # plain-Battle / unit-test path leaves it 0. The reward's no_progress_tax keys on the SAME
        # clock instance, so obs and reward share one value.
        if progress_clock is not None:
            vec[14] = float(progress_clock.value())

        # --- Matchup Matrices (raw battle — see the docstring's three reasons) ---
        our_team = self.get_team_list(battle, is_opponent=False)
        their_team = self.get_team_list(battle, is_opponent=True)

        # Hoist the per-defender reads (type_1/type_2/status/ability + the ability
        # distribution) out of the inner cell loop: they are constant for a given mon
        # across every attacker/move that targets it, so compute them ONCE per team
        # mon instead of re-reading poke-env properties on all 144 cells per matrix.
        our_terms = [
            _defender_terms(m, self._ability_priors) if m is not None else None
            for m in our_team
        ]
        their_terms = [
            _defender_terms(m, self._ability_priors) if m is not None else None
            for m in their_team
        ]

        # 4c. Incoming-damage / OHKO belief block (incoming_damage_v2) — per our mon, the
        # opponent active's KO/chip/outspeed threat under the hidden-set belief. Sits before the
        # matchups so the extractor routes it through non_matchup_rest to both heads. Sourced from
        # the LiveView read-model; the HP tracker is threaded for the revealed-HP typed expansion.
        vec[INCOMING_DMG_OFFSET:INCOMING_DMG_OFFSET + INCOMING_DMG_DIM] = \
            encode_incoming_block(live, hp_tracker=hp_tracker)

        # 5. Our moves vs Their mons (144 dims), starting at REACTIVE_MATCHUP_OFFSET (after the
        # 14 scalars + 36 move-effects + the incoming-damage block).
        cursor = REACTIVE_MATCHUP_OFFSET
        for i in range(TEAM_SIZE):
            our_mon = our_team[i] if i < len(our_team) else None
            our_moves = self.get_sorted_moves(our_mon)
            for move_idx in range(4):
                move = our_moves[move_idx] if move_idx < len(our_moves) else None
                # type_dist is constant across the inner defender loop → hoist it.
                type_dist = _attacker_type_dist(move, our_mon, hp_tracker) if move is not None else None
                for j in range(TEAM_SIZE):
                    their_mon = their_team[j] if j < len(their_team) else None
                    if move and their_mon:
                        # Normalize by 4.0 to keep values in [0, 1] range for better MLP convergence
                        vec[cursor] = _joint_expectation(type_dist, their_terms[j]) / 4.0
                    cursor += 1

        # 6. Their moves vs Our mons (144 dims)
        for i in range(TEAM_SIZE):
            their_mon = their_team[i] if i < len(their_team) else None
            their_moves = self.get_sorted_moves(their_mon)
            for move_idx in range(4):
                move = their_moves[move_idx] if move_idx < len(their_moves) else None
                # type_dist is constant across the inner defender loop → hoist it.
                type_dist = _attacker_type_dist(move, their_mon, hp_tracker) if move is not None else None
                for j in range(TEAM_SIZE):
                    our_mon = our_team[j] if j < len(our_team) else None
                    if move and our_mon:
                        # Normalize by 4.0 to keep values in [0, 1] range for better MLP convergence
                        vec[cursor] = _joint_expectation(type_dist, our_terms[j]) / 4.0
                    cursor += 1
        
        return vec

    def get_layout(self) -> Dict[str, Any]:
        mo = REACTIVE_MATCHUP_OFFSET  # 84 = 15 scalars + 36 move-effects + 33 incoming-damage
        return {
            "move_power": {"offset": 0, "dim": 4},
            "move_multiplier": {"offset": 4, "dim": 4},
            "fainted": {"offset": 8, "dim": 2},
            "active_status": {"offset": 10, "dim": 1},
            "forced_struggle": {"offset": 11, "dim": 1},
            "trapped": {"offset": 12, "dim": 1},
            "maybe_trapped": {"offset": 13, "dim": 1},
            "turns_since_progress": {"offset": 14, "dim": 1},  # gen3_markovian_progress_v1
            # gen3_move_effects_v1: 4 move slots × MOVE_EFFECT_FEATURES (9), slot-major,
            # request order. Per slot: [is_boost, is_heal, is_protect, is_phaze, is_hazard,
            # inflicts_status, status_will_land, pp_fraction, status_will_land_known].
            "move_effects": {"offset": REACTIVE_SCALAR_DIM, "dim": MOVE_EFFECTS_DIM,
                             "per_slot": MOVE_EFFECT_FEATURES},
            # incoming_damage_v1: per-mon [phys_expdmg, spec_expdmg, phys_pko, spec_pko,
            # p_outspeed] × TEAM_SIZE, then [recovery_rate, cures_status, recovery_known].
            "incoming_damage": {"offset": INCOMING_DMG_OFFSET, "dim": INCOMING_DMG_DIM,
                                "per_mon": 5, "recovery": 3},
            "our_matchups": {"offset": mo, "dim": 144},
            "their_matchups": {"offset": mo + 144, "dim": 144},
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        # Extract matrices and scale back up by 4.0 for human-readable display
        mo = REACTIVE_MATCHUP_OFFSET
        our_m = vector[mo:mo + 144].reshape(TEAM_SIZE, 4, TEAM_SIZE) * 4.0
        their_m = vector[mo + 144:mo + 288].reshape(TEAM_SIZE, 4, TEAM_SIZE) * 4.0

        # gen3_move_effects_v1: decode the 4×8 per-move effect block (slot-major).
        eff = vector[REACTIVE_SCALAR_DIM:REACTIVE_SCALAR_DIM + MOVE_EFFECTS_DIM].reshape(4, MOVE_EFFECT_FEATURES)
        eff_names = ["boost", "heal", "protect", "phaze", "hazard", "status",
                     "status_lands", "pp", "status_known"]
        # status_lands (6) and pp (7) are continuous in [0,1]; the rest are bits.
        move_effects = [
            {eff_names[f]: (round(float(eff[s, f]), 2) if f in (6, 7) else bool(eff[s, f]))
             for f in range(MOVE_EFFECT_FEATURES)}
            for s in range(4)
        ]

        return {
            "fainted_our": int(vector[8] * 6),
            "fainted_opp": int(vector[9] * 6),
            "active_move_mults": [f"{m*4.0:.1f}x" for m in vector[4:8].tolist()],
            "struggle": bool(vector[11]),
            "trapped": bool(vector[12]),
            "maybe_trapped": bool(vector[13]),
            "move_effects": move_effects,
            "our_vs_their": our_m, # Full matrix for deeper trace
            "their_vs_our": their_m
        }
