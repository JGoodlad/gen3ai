"""gen3_pair_outcome_v1 — the UNIFIED PER-PAIR OUTCOME GRID's coordinate table, and the α split that reads it.

The op (`DamageOperatorBlocks.pair_outcome_coords`, stashed as `DamageOperator.last_pair_in`) computes one vector per
(our mon j, their believed move seat k) carrying damage AND status in the same units: the op's own damage cell
`[low, high, crit, ko_ramp, acc, is_phys]` (`last_pair_cells`, verbatim), then the six per-identity status landing
probabilities (`p_par … p_tox`), then the two exact status FACTS (`STATUS_FACT_COORDS`). The move-resolution family
(`move_resolution.gather_ops`) is its one consumer.

What survives here is the CONTRACT (`PAIR_OUTCOME_COORDS` / `PAIR_OUTCOME_IDX` / `STATUS_FACT_COORDS`, never a
re-spelled index), the Ghost type id, and `pair_alpha_full` (the published α's three-way split). The two per-action
cells that delivered the reduced grid to the pointer cells (`PairOutcomeMoveCell`, `PairOutcomeSwitchCell`), their
reducers (`reduce_pair_in` / `_all`), the R1 fallback (`pair_alpha`) and the switch-cell `spin_denied` coordinate were
deleted with `--pair-outcome-cell` / `--pair-outcome-switch` (legacy deletion U3, `designs/deleted_flags.md`); their
design history is `design_opponent_intent.md` §5 and `design_pair_reduction.md`. The two derivable collapses
(`p_status_land`, `p_immobilize`) stay deliberately absent: both are linear in the six identities.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from agents.model.arch_constants import _PAIR_OUTCOME_DMG, _PAIR_OUTCOME_RAW

#: The reduced row's coordinate names, in order. Index 0.._PAIR_OUTCOME_DMG-1 are the op's own
#: pair-cell channels (its `last_pair_cells` F axis, unchanged); the rest are gen3_pair_outcome_v1's.
#: This tuple IS the contract the producer, the move-resolution family and the tests read — never re-spell an index.
PAIR_OUTCOME_COORDS: Tuple[str, ...] = (
    # --- the op's existing per-(defender, seat) damage cell, verbatim ---
    "low", "high", "crit", "ko_ramp", "acc", "is_phys",
    # --- NEW: status, per IDENTITY (SECONDARY_COLS' major prefix order) ---
    "p_par", "p_brn", "p_frz", "p_slp", "p_psn", "p_tox",
    # --- the two exact status FACTS (`STATUS_FACT_COORDS`; the JUDGMENTS `neutralization` / `tempo_cost` that held
    #     these positions were deleted by the legacy deletion's U1) ---
    "e_burn_dmg_lost", "e_par_outspeed_lost",
)
#: gen3_endstate_facts_v1 (the exact status facts): the two coordinates that REPLACED `neutralization` / `tempo_cost`
#: in the same two positions (a width-neutral swap: `PAIR_OUTCOME_COORDS[12:14]`), each a FACT in its own unit —
#: the expected HP fraction (of the target's max HP) a burn from this seat costs our mon j's physical damage, and the
#: expected loss of P(j outspeeds their active) a paralysis from this seat costs. `DamageOperatorBlocks.
#: pair_outcome_coords` is the producer; the cure availability rides `status_facts.cure_flags` on the mon tokens.
STATUS_FACT_COORDS: Tuple[str, ...] = ("e_burn_dmg_lost", "e_par_outspeed_lost")
assert PAIR_OUTCOME_COORDS[12:14] == STATUS_FACT_COORDS
#: Name → index over `PAIR_OUTCOME_COORDS`. The op reads `"high"` off it when building the vector,
#: so a coordinate reorder cannot silently feed the status physics the wrong damage column.
PAIR_OUTCOME_IDX = {name: i for i, name in enumerate(PAIR_OUTCOME_COORDS)}

assert len(PAIR_OUTCOME_COORDS) == _PAIR_OUTCOME_RAW, \
    "PAIR_OUTCOME_COORDS and _PAIR_OUTCOME_RAW disagree — one was edited without the other."
assert PAIR_OUTCOME_COORDS[_PAIR_OUTCOME_DMG - 1] == "is_phys", \
    "_PAIR_OUTCOME_DMG no longer names the op's damage-cell prefix."

#: The TypeEncoder id of GHOST (`agents.observation.types.TypeEncoder.TYPE_TO_IDX`). Resolved from
#: that table at import rather than written as `14`, so a renumbering fails at import instead of
#: silently pricing Dragon-typed switch-ins as spinblockers.
def _ghost_type_idx() -> int:
    from agents.observation.types import TypeEncoder
    idx = TypeEncoder.TYPE_TO_IDX.get("GHOST")
    if idx is None:
        raise ValueError("TypeEncoder has no GHOST entry — the spinblock predicate has no axis.")
    return int(idx)


GHOST_TYPE_IDX = _ghost_type_idx()


def pair_alpha_full(alpha_logits: torch.Tensor,
                    seat_live: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three-way split of the PUBLISHED α, for a consumer that needs the SWITCH branch.
    `alpha_logits` `[B,K+1]` (last class = SWITCH) → `(alpha_moves [B,K], a_switch [B,1],
    a_stay [B,1])`.

    The move-resolution family needs the switch mass as a first-class number, so this returns it explicitly
    rather than making each consumer re-softmax. Three properties, all deliberate:

    * **stop-grad**, unconditionally — these are POLICY-side consumers, and resting on
      `--belief-grad-mode label_only` to cut the PPO→`alpha_head` route makes the route's EXISTENCE
      a function of a TRAINING flag (the v87 pattern, codified in `model/CLAUDE.md`).
    * **masked seats' mass is NOT reassigned.** An unmodeled 5th+ seat describes nothing, so its
      usage mass is simply not spent — which is why `a_stay` is computed as `Σ_k alpha_moves` and
      NOT as `1 − a_switch`. The two differ exactly when a seat is masked, and the sum form is the
      honest one (`a_stay + a_switch ≤ 1`).
    * **no channel axis and no defender axis** — Contract W, enforced by the return shapes.
    """
    full = torch.softmax(alpha_logits.detach().float(), dim=-1).to(alpha_logits.dtype)
    k = alpha_logits.shape[-1] - 1
    alpha = full[:, :k]
    if seat_live is not None:
        alpha = alpha * seat_live.to(alpha.dtype)
    return alpha, full[:, -1:], alpha.sum(-1, keepdim=True)
