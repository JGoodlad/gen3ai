import numpy as np
from .base import ObservationEncoder
from poke_env.battle.abstract_battle import AbstractBattle
from .constants import (
    REACTIVE_DIM, TEAM_SIZE,
    REACTIVE_SCALAR_DIM, REACTIVE_MATCHUP_OFFSET,
    ACTIVE_REQ_MOVES_OFFSET, ACTIVE_REQ_MOVES_PER, ACTIVE_REQ_MOVES_DIM,
)
from .types import TypeEncoder
from agents.enums import PokemonType
from agents import gen3_data
from agents.gen3_mechanics import (
    effective_multiplier_by_types, status_land_estimate, protect_success_probability,
)
from agents.observation.wish_belief import build_wish_pending, wish_floating_value
from agents.battle.battle_event import OURS, OPP
from typing import Any, Dict, List, Optional

# Neutral fill for an UNWRITTEN active-move multiplier slot (a mon with <4 moves, or no opp active).
# Multipliers are stored as effectiveness/4, so 1× → 0.25; the old np.ones(4) default stored raw 1.0,
# which decodes to a phantom 4× super-effective KO threat on a non-existent move (the move slot that
# the model + prober then read as the scariest possible matchup). 0.25 is the "no modifier" point.
_NEUTRAL_MULT = 0.25


def _request_slot_moves(battle, legal):
    """Per-request-slot ``Move`` objects in ACTION order — slot i ↔ action logit ``6+i`` — with
    DISABLED moves KEPT, so the per-move obs features (base power, type multiplier, effect flags)
    align with the action mask / mapper, which both index ``legal.move_slots[i] → action 6+i``.

    poke-env's ``battle.available_moves`` DROPS disabled moves (``available_moves_from_request``
    filters ``disabled``: Disable / Taunt-on-status / Imprison / 0-PP), which LEFT-SHIFTS every later
    slot relative to the request order and leaves the trailing slot unwritten — the misalignment this
    replaces. A disabled move is still a real, identity-bearing option the model should see at its
    true logit position (legality is the mask's job, not the feature's).

    Our own Hidden Power arrives in the request as the bare ``hiddenpower``; the TYPED ``Move``
    (``hiddenpowerfire`` …, carrying the correct effectiveness) lives on the active mon's moveset, so
    resolve it the same way poke-env's ``available_moves_from_request`` does (single ``hiddenpower*``).

    Falls back to ``battle.available_moves`` order ONLY when ``legal is None`` (unit-test / plain-
    ``Battle`` callers) — never the trainee obs path, where ``state_encoder`` always threads a real
    ``legal`` built from the strict view. Byte-identical to the old behaviour on that fallback path.
    """
    active = battle.active_pokemon
    if legal is None or active is None:
        return list(battle.available_moves)[:4]
    moveset = active.moves
    slot_moves = []
    for lm in legal.move_slots[:4]:
        mv = moveset.get(lm.id)
        if mv is None and lm.id == "hiddenpower":
            hps = [v for mid, v in moveset.items() if mid.startswith("hiddenpower")]
            mv = hps[0] if len(hps) == 1 else None
        slot_moves.append(mv)
    return slot_moves


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
    - turns_since_progress (1) — gen3_markovian_progress_v1: the log-saturated no-progress clock (vec[6])
    - Protect-success odds (2) — gen3_protect_odds_v1: P(Protect/Detect/Endure succeeds NOW) for our
      active (vec[7]) and the opp active (vec[8]) — gen3 floored doubling (100/50/25/12.5, 1/8 floor)
      from each mon's LiveView protect_counter; the only obs signal for the stall counter.
      [is_boost, is_heal, is_protect, is_phaze, is_hazard, inflicts_status,
      status_will_land, pp_fraction, status_will_land_known] so the policy head can tell a
      setup move from a heal from a wasted status (otherwise indistinguishable: base power 0 +
      neutral multiplier). status_will_land is a prior-weighted probability; the trailing
      *_known bit flags confirmed-vs-prior, mirroring the ability block's `known` flag.
    - Matchup Matrix: Our moves vs Their mons (144)
    - Matchup Matrix: Their moves vs Our mons (144)
    - Active request-move id/type/legality (12) — gen3_op_move_align_v1: OUR active's 4 moves in
      REQUEST order (action 6+k), [move_num ×4, resolved_type_id ×4, legal_now ×4], consumed by the
      DamageOperator's OUTGOING per-move blocks so their output aligns with the action logits.
    Total: REACTIVE_DIM (19 scalars + 44 move-effects + 51 incoming-damage + 288 matchup
    + 12 active-req-moves = 414).
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
        ``trapped`` / ``maybe_trapped`` flags become two obs bits (vec[4], vec[5]). ``None``
        (unit-test / plain-Battle path, or a caller that hasn't threaded it) leaves both at 0.

        **Active-move slot order + why the effectiveness core stays on the raw battle.** The
        active-move loop iterates the per-decision **request slots** (``legal.move_slots``, action
        order: slot i ↔ action logit ``6+i``, disabled moves KEPT) via ``_request_slot_moves``, and
        resolves each slot to its raw poke-env ``Move`` off the active mon's moveset — NOT
        ``battle.available_moves``, which drops disabled moves and would shift every later feature
        out of action alignment (gen3_move_slot_align_v1). The two 6×4×6 matchup matrices still read
        moves off the raw ``Move`` objects (``get_sorted_moves``) and the defender's
        ``type_1``/``type_2``/``ability``/``status``. The effectiveness core is *not* migrated to the
        read-model on purpose, for three independent reasons:
          1. *Typed Hidden Power id.* Our own HP move keeps its typed id (``hiddenpowerfire``)
             only on the raw ``Move`` object; the live request — and therefore ``LiveView`` /
             ``LegalActions`` — re-keys it to bare ``hiddenpower``. The slot resolver re-derives the
             typed ``Move`` from the moveset (single ``hiddenpower*``), preserving the emitted
             effectiveness; reading the read-model's id directly would collapse own HP to
             Normal/tracker. (See the matching note in ``pokemon.py``.)
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
        # gen3_cpu_damage_deleted_v1: the per-move base-power / type-multiplier scalars and the
        # 44-dim move-effect block used to be built here. Both are now GPU-side (the op's OUTGOING
        # per-move block + the move latent), so the CPU no longer computes them at all.
        active_req_move_ids = np.zeros(ACTIVE_REQ_MOVES_PER, dtype=np.float32)
        active_req_move_type_ids = np.zeros(ACTIVE_REQ_MOVES_PER, dtype=np.float32)
        active_req_move_legal = np.zeros(ACTIVE_REQ_MOVES_PER, dtype=np.float32)

        # Skip Struggle — it has a dedicated action (10) and a dedicated flag (vec[3]).
        # Filling the move slots with Struggle's stats would create a confusing alias
        # between slot 0 (action 6) and the Struggle action (10).
        # Forced Struggle (all PP gone). Prefer the server-authoritative `legal.struggle` when the
        # snapshot is threaded (avoids a `battle.available_moves` property rebuild on the hot path);
        # the plain-Battle / unit-test fallback reproduces the original check byte-identically.
        if legal is not None:
            is_forced_struggle = legal.struggle
        else:
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
            # REQUEST-slot order (action 6+i ↔ slot i), disabled moves KEPT — so each per-move
            # feature lands at its true action-logit position instead of being shifted by a
            # disabled non-last slot (see `_request_slot_moves`). gen3_move_slot_align_v1.
            for i, move in enumerate(_request_slot_moves(battle, legal)):
                if i >= 4:
                    break
                if move is None:
                    continue  # empty / unresolved request slot → neutral defaults (masked anyway)
                md = gen3_data.moves.get(move.id)

                # gen3_op_move_align_v1: request-order move NUM + resolved TYPE-id for the op's OUTGOING
                # blocks. NUM mirrors moves.py (HP → 237 regardless of type, so the op's HP branch fires);
                # the resolved type drives STAB / effectiveness (our own Hidden Power arrives typed, via
                # _request_slot_moves, so the else branch supplies the real type — a bare 'hiddenpower'
                # would be type-unknown, which never happens for our own active).
                if md is not None:
                    active_req_move_ids[i] = float(md.num)
                    if move.id == "hiddenpower":
                        type_id = 0
                    else:
                        type_name = "???" if md.type.name == "THREE_QUESTION_MARKS" else md.type.name
                        type_id = TypeEncoder.TYPE_TO_IDX.get(type_name, 0)
                    active_req_move_type_ids[i] = float(type_id)
                # legal_now: current-decision choosability in request order — the EXACT action-mask move
                # bit (`not legal.move_slots[i].disabled`). The legal-is-None fallback path drew moves from
                # battle.available_moves (already disabled-filtered) → every resolved move is choosable.
                active_req_move_legal[i] = (
                    1.0 if (legal is not None and i < len(legal.move_slots)
                            and not legal.move_slots[i].disabled)
                    else (1.0 if legal is None else 0.0)
                )

        # gen3_op_move_align_v1: request-order active-move id/type/legality block (sits AFTER the
        # matchups; consumed only by the DamageOperator's outgoing methods via ObsUnpack, never the
        # raw-scalar path). Layout: [ids ×4, type_ids ×4, legal ×4].
        _ar = ACTIVE_REQ_MOVES_OFFSET
        _p = ACTIVE_REQ_MOVES_PER
        vec[_ar:_ar + _p] = active_req_move_ids
        vec[_ar + _p:_ar + 2 * _p] = active_req_move_type_ids
        vec[_ar + 2 * _p:_ar + 3 * _p] = active_req_move_legal

        # 2. Fainted Counts — read through the LiveView (m.fainted) when available, else raw.
        if live is not None:
            fainted_mon_team = sum(1 for m in live.ours.mons if m.fainted) / 6.0
            fainted_mon_opponent = sum(1 for m in live.opp.mons if m.fainted) / 6.0
        else:
            fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6.0
            fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6.0
        vec[0] = fainted_mon_team
        vec[1] = fainted_mon_opponent

        # 3. Status — active mon currently has a status condition (HP and Spikes removed —
        # available in per-Pokémon vector and global env). Read through the LiveView's active
        # slot when available; both paths resolve the active off battle.active_pokemon, so the
        # truthiness is identical.
        if live is not None:
            active_live = live.ours.active
            vec[2] = 1.0 if (active_live is not None and active_live.status) else 0.0
        else:
            vec[2] = 1.0 if battle.active_pokemon and battle.active_pokemon.status else 0.0

        # 4. Forced Struggle
        vec[3] = 1.0 if is_forced_struggle else 0.0

        # 4b. Trapping signals (gen3_trapping_signals_v1) — server-authoritative legality.
        # trapped: confirmed cannot switch (Mean Look / Arena Trap / Magnet Pull revealed) —
        # redundant with the mask but an explicit feature. maybe_trapped: the opponent MIGHT
        # be trapping us (switches still legal in the mask) — the highest-value bit, the only
        # signal here the model has no other way to see.
        if legal is not None:
            vec[4] = 1.0 if legal.trapped else 0.0
            vec[5] = 1.0 if legal.maybe_trapped else 0.0

        # 4d. turns_since_progress (gen3_markovian_progress_v1, vec[6]) — the log-saturated
        # no-progress clock (design §5.1). Sourced from the EpisodeTracker-owned ProgressClock
        # (NOT LiveView — it is cross-turn state), threaded in like the HP tracker; None on the
        # plain-Battle / unit-test path leaves it 0. The reward's no_progress_tax keys on the SAME
        # clock instance, so obs and reward share one value.
        if progress_clock is not None:
            vec[6] = float(progress_clock.value())

        # 4e. Protect-success odds (gen3_protect_odds_v1, vec[7] our active / vec[8] opp active).
        # P(a Protect/Detect/Endure succeeds NOW) from each active mon's consecutive-stall counter,
        # read through the LiveView read-model (NOT raw poke-env). Showdown gen3 = floored doubling
        # (100/50/25/12.5, 1/8 floor); the gen3 mechanic lives in protect_success_probability(). This
        # is the only obs signal for the stall counter — poke-env doesn't enumerate the 'stall'
        # volatile, and history saliency decays before the model can count a chain. Public both sides
        # (the opp's counter derives entirely from their revealed move stream → no leak). 0.0 on the
        # None / no-active-mon paths (unit-test only — a real decision always has an active mon).
        if live is not None:
            our_active = live.ours.active
            opp_active = live.opp.active
            if our_active is not None:
                vec[7] = protect_success_probability(our_active.protect_counter)
            if opp_active is not None:
                vec[8] = protect_success_probability(opp_active.protect_counter)

        # 4f. Wish "floating heal" (gen3_wish_wired_v1, vec[9] our side / vec[10] opp). P(KO)-style
        # belief reconstructed from OUR event log (poke-env doesn't track pending Wish): a gen3 Wish cast
        # last turn heals the slot mon ~50% of its max HP at the END of this turn (slot-keyed, so it
        # survives faint/phaze/switch). The value is the flat WISH_HEAL_FRACTION (≈recipient maxhp/2 — no
        # max-HP read, GIGO-proof) when pending, else 0.0. Folded from `battle` (a Gen3Battle); on the
        # mock / non-Gen3Battle path `battle.events` is absent → both stay 0.0.
        wish_pending = build_wish_pending(battle)
        vec[9] = wish_floating_value(wish_pending[OURS])
        vec[10] = wish_floating_value(wish_pending[OPP])

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

        # 5. Our moves vs Their mons (144 dims), starting at REACTIVE_MATCHUP_OFFSET (after the
        # 17 scalars + 36 move-effects + the 51-dim incoming-damage block).
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
        mo = REACTIVE_MATCHUP_OFFSET  # 11 — the scalars are now the only thing before the matchups
        return {
            # gen3_cpu_damage_deleted_v1: move_power / move_multiplier (the 8 active-move scalars),
            # the 44-dim move_effects block and the 51-dim incoming_damage block are GONE — the
            # DamageOperator computes all three GPU-side from the LEARNED belief. Scalars below are
            # the post-deletion indices (each shifted down by 8).
            "fainted": {"offset": 0, "dim": 2},
            "active_status": {"offset": 2, "dim": 1},
            "forced_struggle": {"offset": 3, "dim": 1},
            "trapped": {"offset": 4, "dim": 1},
            "maybe_trapped": {"offset": 5, "dim": 1},
            "turns_since_progress": {"offset": 6, "dim": 1},  # gen3_markovian_progress_v1
            # gen3_protect_odds_v1: P(Protect/Detect/Endure succeeds NOW), our active then opp active.
            "protect_odds_our": {"offset": 7, "dim": 1},
            "protect_odds_opp": {"offset": 8, "dim": 1},
            # gen3_wish_wired_v1: pending-Wish "floating heal" — one dim per side.
            "wish_floating_our": {"offset": 9, "dim": 1},
            "wish_floating_opp": {"offset": 10, "dim": 1},
            "our_matchups": {"offset": mo, "dim": 144},
            "their_matchups": {"offset": mo + 144, "dim": 144},
            # gen3_op_move_align_v1: request-order active-move id/type/legality (after the matchups).
            # Sub-blocks are contiguous: ids[0:4], type_ids[4:8], legal[8:12] within the block.
            "active_req_moves": {"offset": ACTIVE_REQ_MOVES_OFFSET, "dim": ACTIVE_REQ_MOVES_DIM,
                                 "per": ACTIVE_REQ_MOVES_PER},
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        # Extract matrices and scale back up by 4.0 for human-readable display
        mo = REACTIVE_MATCHUP_OFFSET
        our_m = vector[mo:mo + 144].reshape(TEAM_SIZE, 4, TEAM_SIZE) * 4.0
        their_m = vector[mo + 144:mo + 288].reshape(TEAM_SIZE, 4, TEAM_SIZE) * 4.0

        return {
            "fainted_our": int(vector[0] * 6),
            "fainted_opp": int(vector[1] * 6),
            "struggle": bool(vector[3]),
            "trapped": bool(vector[4]),
            "maybe_trapped": bool(vector[5]),
            "turns_since_progress": round(float(vector[6]), 3),
            "protect_odds_our": round(float(vector[7]), 3),
            "protect_odds_opp": round(float(vector[8]), 3),
            "our_vs_their": our_m, # Full matrix for deeper trace
            "their_vs_our": their_m
        }
