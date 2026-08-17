import numpy as np
from .base import ObservationEncoder
from poke_env.battle.abstract_battle import AbstractBattle
from .constants import (
    REACTIVE_DIM, ACTIVE_REQ_MOVES_OFFSET, ACTIVE_REQ_MOVES_PER, ACTIVE_REQ_MOVES_DIM,
)
from .types import TypeEncoder
from agents import gen3_data
from agents.observation.wish_belief import build_wish_pending, wish_floating_value
from agents.battle.battle_event import OURS, OPP
from typing import Any, Dict, List, Optional

def _request_slot_moves(battle: Any, legal: Any) -> List[Any]:
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
    
    def __init__(self, ability_priors: Optional[Dict[str, Dict[str, float]]] = None) -> None:
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

    # Why the `type: ignore[override]` below — LiveView-subject encoder; see ActiveContextEncoder.encode.
    def encode(self, battle: AbstractBattle, hp_tracker: Any = None,  # type: ignore[override]
               live: Any = None, legal: Any = None,
               progress_clock: Any = None) -> np.ndarray:
        """Encode the reactive block.

        live: optional :class:`~agents.battle.live_view.LiveView` snapshot for this
        decision. When supplied, the structural reads — per-side fainted counts and the
        active mon's status flag — are taken through the read-model; ``None`` (unit-test /
        plain-Battle path) falls back to the raw battle, byte-identically.

        legal: optional :class:`~agents.battle.live_view.LegalActions` snapshot for this
        decision (server-authoritative legality, same surface the mask is built from). Drives
        the request-slot resolution below and the ``legal_now`` bits; the trapped/maybe_trapped
        facts it carries are written by state_encoder onto OUR ACTIVE's mon slot
        (gen3_entity_rehome_v1), not here. ``None`` = the unit-test / plain-Battle path.

        **Active-move slot order.** The active-move loop iterates the per-decision **request
        slots** (``legal.move_slots``, action order: slot i ↔ action logit ``6+i``, disabled
        moves KEPT) via ``_request_slot_moves``, and resolves each slot to its raw poke-env
        ``Move`` off the active mon's moveset — NOT ``battle.available_moves``, which drops
        disabled moves and would shift every later feature out of action alignment
        (gen3_move_slot_align_v1). Our own Hidden Power keeps its typed id
        (``hiddenpowerfire``) only on the raw ``Move`` object, so the resolver re-derives the
        typed ``Move`` from the moveset (see the matching note in ``pokemon.py``).

        hp_tracker: accepted for call-compat (state_encoder threads it) but UNUSED since
        gen3_entity_rehome_v1 — the matchup matrices it fed are deleted; the HP-type
        distribution reaches the model via the per-mon HP block + the GPU belief instead.

        Ability priors (`self._ability_priors`) are likewise retained on the instance for
        call-compat but no longer read here — the ability-expectation matchup cells they fed
        are deleted with the matrices.
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

        # gen3_entity_rehome_v1: active_status is DELETED (byte-redundant with the active mon's
        # per-slot condition one-hot); forced_struggle is DELETED (derivable — the
        # active_req_moves legal bits are all zero on a move-request exactly when struggle is
        # forced, and the action mask stays authoritative); trapped/maybe_trapped moved to the
        # OUR-ACTIVE mon slot and protect_odds to EVERY mon slot (state_encoder / pokemon.py —
        # the facts now ride the entities they describe).

        # 3. turns_since_progress (gen3_markovian_progress_v1, vec[2]) — the log-saturated
        # no-progress clock (design §5.1). Sourced from the EpisodeTracker-owned ProgressClock
        # (NOT LiveView — it is cross-turn state), threaded in like the HP tracker; None on the
        # plain-Battle / unit-test path leaves it 0. The reward's no_progress_tax keys on the SAME
        # clock instance, so obs and reward share one value.
        if progress_clock is not None:
            vec[2] = float(progress_clock.value())

        # 4. Wish "floating heal" (gen3_wish_wired_v1, vec[3] our side / vec[4] opp). P(KO)-style
        # belief reconstructed from OUR event log (poke-env doesn't track pending Wish): a gen3 Wish cast
        # last turn heals the slot mon ~50% of its max HP at the END of this turn (slot-keyed, so it
        # survives faint/phaze/switch). The value is the flat WISH_HEAL_FRACTION (≈recipient maxhp/2 — no
        # max-HP read, GIGO-proof) when pending, else 0.0. Folded from `battle` (a Gen3Battle); on the
        # mock / non-Gen3Battle path `battle.events` is absent → both stay 0.0.
        wish_pending = build_wish_pending(battle)
        vec[3] = wish_floating_value(wish_pending[OURS])
        vec[4] = wish_floating_value(wish_pending[OPP])

        # gen3_entity_rehome_v1: the two 144-dim matchup matrices are DELETED — the D/V edge
        # families compute a strict superset GPU-side (real gen3 physics + the learned move
        # belief per pair, [low, high, crit, pko, type_mult, revealed] cells) from raw facts,
        # so the CPU no longer prices a single matchup cell per decision.
        return vec

    def get_layout(self) -> Dict[str, Any]:
        # gen3_entity_rehome_v1: the matchup matrices, active_status, forced_struggle,
        # trapped/maybe_trapped and protect_odds entries are GONE from this block — deleted or
        # re-homed onto the per-mon slots (see constants.py POKEMON_PROTECT_OFFSET /
        # POKEMON_TRAPPED_OFFSET). What remains is the raw BOARD state: side facts (fainted
        # counts, pending Wish), the cross-turn progress clock, and the request-order
        # active-move ID block.
        return {
            "fainted": {"offset": 0, "dim": 2},
            "turns_since_progress": {"offset": 2, "dim": 1},  # gen3_markovian_progress_v1
            # gen3_wish_wired_v1: pending-Wish "floating heal" — one dim per side.
            "wish_floating_our": {"offset": 3, "dim": 1},
            "wish_floating_opp": {"offset": 4, "dim": 1},
            # gen3_op_move_align_v1: request-order active-move id/type/legality.
            # Sub-blocks are contiguous: ids[0:4], type_ids[4:8], legal[8:12] within the block.
            "active_req_moves": {"offset": ACTIVE_REQ_MOVES_OFFSET, "dim": ACTIVE_REQ_MOVES_DIM,
                                 "per": ACTIVE_REQ_MOVES_PER},
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        return {
            "fainted_our": int(vector[0] * 6),
            "fainted_opp": int(vector[1] * 6),
            "turns_since_progress": round(float(vector[2]), 3),
            "wish_floating_our": round(float(vector[3]), 3),
            "wish_floating_opp": round(float(vector[4]), 3),
        }
