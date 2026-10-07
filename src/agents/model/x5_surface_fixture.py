"""TEST SUPPORT: the extractor kwargs that build the opponent-belief family as it exists since the X5 version
break (config v144, `gen3_x5_version_break_v1`).

X5's hypothesis tokens are the ONLY belief representation, so `opp_belief_slots` / `opp_intent` build them and
`extractor_build` REFUSES a configuration missing one of their requirements (both toggles, `t0_species_prior`,
`move_belief_mode != off`, `entity_topk_seats >= 4`, `entity_tail_seats`, `damage_topk_k == entity_topk_seats`
when the incoming matrix is on, `damage_candidate_k == 0`). A test that turns the family on takes
:func:`x5_kwargs` over its own kwargs; a test of the belief-OFF ablation leaves both toggles off. The
production surface itself is `main.fresh_checkpoint._production_policy_kwargs` — use that when the subject is
"what production builds", this when the subject is one cell on a hand-built extractor.

Not collected by pytest (no ``_test`` suffix) and imported only by tests.
"""
from __future__ import annotations

from typing import Any, Dict

#: The seat count X5 needs at least (`extractor_build`: revealed moves take the first seats).
X5_MIN_SEATS = 4
#: The seat count a test gets when it names none (production's `entity_topk_seats`).
X5_DEFAULT_SEATS = 6


def x5_kwargs(**kw: Any) -> Dict[str, Any]:
    """``kw`` with the opponent-belief family ON and every requirement X5 refuses without. A seat count the
    caller named (``entity_topk_seats``, else ``damage_topk_k``) is kept — it must be >= 4 — and the op's
    top-K follows it when the incoming matrix is on; ``damage_op`` + ``move_latent`` (what E4 seats need) and
    ``move_prior_fusion`` (what the T0 prior's hidden-slot move mixture reads) are turned on. ``move_belief_mode`` is kept unless it is ``off`` (then ``revealed``)."""
    out = dict(kw)
    out.update(opp_belief_slots=True, opp_intent=True, t0_species_prior=True, entity_tail_seats=True,
               attend_unrevealed_opponents=True, damage_op=True, move_latent=True, damage_candidate_k=0,
               # NOT refused at build, but the forward needs it: the T0 prior's hidden-slot move mixture
               # reads MoveBelief's per-species prior (`move_prior_probs`), built only under fusion.
               move_prior_fusion=True)
    if out.get("move_belief_mode", "off") == "off":
        out["move_belief_mode"] = "revealed"
    k = int(out.get("entity_topk_seats") or out.get("damage_topk_k") or X5_DEFAULT_SEATS)
    if k < X5_MIN_SEATS:
        raise ValueError(f"X5 needs entity_topk_seats >= {X5_MIN_SEATS}, the test named {k}")
    out["entity_topk_seats"] = k
    if out.get("damage_matrices_incoming"):
        if out.get("damage_topk_k") and int(out["damage_topk_k"]) != k:
            raise ValueError(f"X5 needs damage_topk_k == entity_topk_seats, the test named "
                             f"{out['damage_topk_k']} and {k}")
        out["damage_topk_k"] = k
    return out


__all__ = ["x5_kwargs", "X5_MIN_SEATS", "X5_DEFAULT_SEATS"]
