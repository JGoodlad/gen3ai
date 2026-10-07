"""OPPONENT INTENT — `α` (what will they click) and `β` (who will they bring).

`design_opponent_intent.md`. The model can price *what a move would do* but has never been able to
say *which move they are likely to click*. Everything downstream of that sentence — hedging,
switching into a likely attack rather than a possible one, and (the reason this ships even if it
buys no Elo) a HUMAN-READABLE statement of what the model expects — needs a distribution over
their options, not a set of them.

**The measured case for it** (`tmp/g2b_alpha_baseline.py`, gen-8 @26M, n=1676 attack decisions):
the believed-move top-K *contains* the move they actually clicked **85.8%** of the time, but the
belief's own ordering puts it first only **51.8%** of the time — 24.6% of the time the true move is
sitting at rank 1, one slot away. That **34.0 pp** gap is mass already inside the seats and merely
mis-ranked, which is exactly what a learned re-weighting can move and no new information is needed
to move it. (For scale, the hidden-team belief was greenlit on ~8-10 pp of top-1 headroom.)

## The two hard constraints, and how the shapes enforce them

**1. Discrete, named options only — the owner constraint.** `α` is a distribution over the belief's
OWN K seats plus one `SWITCH` option; there is no `UNKNOWN` slot and no free-form move head. If the
belief does not hold it, `α` cannot name it, and the target is MASKED rather than smeared onto a
neighbour. That is what makes the output legible: every probability points at a move with a name.

**2. Equivariance.** Both heads are POINTER-style — seat `k`'s logit is scored from seat `k`'s own
features through a SHARED scorer, and bench slot `j`'s from slot `j`'s own token through another.
Neither head has a weight indexed by seat position, so permuting their moves permutes `α` and
permuting their bench permutes `β`, exactly. A flat `Linear(ctx, K)` would have been simpler and
would have quietly learned "seat 0 is usually the best move" from the belief's own sort order —
memorising the ordering we are trying to correct.

## Matching is by canonical id, never by index

Seats are `w.topk(K)` and therefore **permute every turn**. The environment cannot know them (they
are built by the model mid-forward), so the env emits the opponent's raw move NUM and the loss
locates it among the seats at loss time. A miss is masked, and the mask rate is logged as a
first-class diagnostic — it is the belief's coverage failure showing up in `α`'s denominator, and
conflating it with `α` being wrong would hide which component to fix.

## What is live

The opponent-intent readout is X5's flat pointer (`agents.model.flat_intent`, folded by
`agents.training.instrumented_ppo.flat_intent_fold`); its consumers read `α` / `β` as its
RE-EXPRESSION. The blob path's two separate heads (`AlphaIntentHead` / `BetaSwitchHead`) are DELETED
(the version break, config v144, part 2). What this module still serves live: the label sentinel
`INTENT_IGNORE`, the opponent-class table, the canonical-id seat matching, content-addressed slot
resolution and the α render.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch

# Target sentinel for "no supervised label this decision" — the belief did not hold the move they
# clicked, or the trace could not name their action. Must be negative (a valid class is >= 0) and
# is what `cross_entropy(ignore_index=...)` skips.
INTENT_IGNORE = -100


def match_seats_to_move_num(seat_nums: torch.Tensor, chosen_num: torch.Tensor,
                            kind: torch.Tensor, n_seats: int) -> torch.Tensor:
    """Build `α`'s integer target by CANONICAL ID, the only stable key across a permuting seat set.

    `seat_nums` [B,K] int — the move nums the op's top-K actually holds (`op.last_topk_idx`).
    `chosen_num` [B] int   — the move num they clicked (env label; ignored when kind != MOVE).
    `kind` [B] int         — 0 MOVE, 1 SWITCH, 2 UNKNOWN/unnameable.

    Returns [B] with the seat index, `n_seats` for SWITCH, or `INTENT_IGNORE` when the belief did
    not hold their move (a coverage miss) or the action could not be named. Masking rather than
    guessing is the point: a smeared target would train `α` toward a move they did not pick and
    make its accuracy uninterpretable.
    """
    B = chosen_num.shape[0]
    tgt = torch.full((B,), INTENT_IGNORE, dtype=torch.long, device=seat_nums.device)
    hit = seat_nums == chosen_num[:, None]                                       # [B,K]
    any_hit = hit.any(dim=-1)
    first = hit.float().argmax(dim=-1)                                           # first match
    is_move = kind == 0
    tgt = torch.where(is_move & any_hit, first.long(), tgt)
    tgt = torch.where(kind == 1, torch.full_like(tgt, n_seats), tgt)
    return tgt


def resolve_believed_slot_by_content(
    species_logits: torch.Tensor,
    believed_mask: torch.Tensor,
    switch_species: torch.Tensor,
    min_prob: float = 0.05,
) -> torch.Tensor:
    """CONTENT-ADDRESSED `β` target — which believed slot does the MODEL think holds this mon?

    A believed slot is an ANONYMOUS query. The species loss matches the k slots to the k hidden
    mons by HUNGARIAN assignment, which discards any label-side ordering — so there is no
    index-based answer to "which slot is Blissey", and the Pokedex-sorted canonicalisation in
    `assign_hidden_to_slots` is NOT that answer (see `opp_intent_labels`). The only coherent target
    is the slot the model's OWN species posterior puts that mon in:

        target_j = argmax over BELIEVED slots j of  P_j(species = the mon that came in)

    That makes `β` and the species head refer to the same object by construction — "β says slot 4,
    the species head says slot 4 is Blissey" — which is what makes the rendered sentence
    *"they will switch to Blissey"* mean anything.

    It also dissolves the positional worry that motivated it. `BeliefSlots` gives each believed slot
    its own learned query (`unknown_slot_emb`, 6 distinct rows — load-bearing for DETR-style set
    coverage), so a slot's token DOES carry its index, and an index-based target would let `β` score
    slots by position instead of content. Under a content-addressed target position is no longer
    PREDICTIVE, so the shortcut earns nothing — the objective removes the incentive rather than the
    architecture removing the capability.

    **Masks on belief miss**, the same rule `α` follows: if no believed slot gives the mon at least
    `min_prob`, the model does not believe it is there and there is nothing coherent to point at.
    Supervising anyway would train `β` toward whichever slot happened to hold the argmax of a
    near-uniform posterior — noise. The mask rate is reported, since it is the BELIEF's failure and
    must stay attributable to the belief rather than to `β`.

    `species_logits` [B, n_slots, n_species]; `believed_mask` [B, n_slots] (1 = a believed slot);
    `switch_species` [B] species nums (0 = not a switch). Returns [B] slot index or INTENT_IGNORE.
    """
    B, n_slots, _ = species_logits.shape
    probs = torch.softmax(species_logits.float(), dim=-1)
    idx = switch_species.clamp(min=0, max=probs.shape[-1] - 1)
    p_species = probs.gather(-1, idx[:, None, None].expand(B, n_slots, 1)).squeeze(-1)  # [B,slots]
    p_species = p_species.masked_fill(believed_mask < 0.5, -1.0)
    best_p, best_j = p_species.max(dim=-1)
    ok = (switch_species > 0) & (best_p >= min_prob)
    return torch.where(ok, best_j, torch.full_like(best_j, INTENT_IGNORE))


#: `opp_class` code -> the suffix its stratified metrics carry. Mirrors
#: `agents.training.opponent_classes`; kept as a plain table here so the model package does not
#: import the training package (pinned equal by `opponent_classes_test`).
OPP_CLASS_NAMES = {0: "bot", 1: "pool", 2: "stable", 3: "exploiter"}

#: The one class the label weight below discounts. Named rather than spelled `0` at the use site,
#: because `OPP_CLASS_NAMES[0]` and this constant must always be the same row.
OPP_CLASS_BOT = 0


def render_alpha(alpha_probs: torch.Tensor, seat_nums: torch.Tensor,
                 move_name: Callable[[int], Optional[str]],
                 top: int = 5) -> List[Dict[str, Any]]:
    """G3b — `α` as a ranked list of NAMED options, the interpretability deliverable.

    This is not a debug helper: the owner constraint is that the model may only ever point at
    things it can name, and this function is what makes that checkable by a human and assertable
    by a test. Returns [{'name': str, 'p': float}], SWITCH included, highest first.
    """
    k = alpha_probs.shape[-1] - 1
    rows: List[Dict[str, Any]] = []
    for i in range(k):
        num = int(seat_nums[i])
        if num <= 0:
            continue
        rows.append({"name": move_name(num) or f"move#{num}", "p": float(alpha_probs[i])})
    rows.append({"name": "SWITCH", "p": float(alpha_probs[k])})
    rows.sort(key=lambda r: -r["p"])
    return rows[:top]
