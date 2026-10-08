from __future__ import annotations
import numpy as np
from .base import ObservationEncoder
from .constants import (
    REACTIVE_DIM, ACTIVE_REQ_MOVES_OFFSET, ACTIVE_REQ_MOVES_PER, ACTIVE_REQ_MOVES_DIM,
)
from typing import Any, Dict, Optional


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
