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

## Gradient policy

The losses are SUPERVISION ONLY: `α`/`β` are trained by cross-entropy against what the opponent
actually did, and their gradient is **stop-gradiented out of the trunk** in v1 (`detach_input`).
A null is then interpretable — it says the head cannot predict the opponent, not that predicting
the opponent perturbed the policy. Letting `α` shape the trunk is a later, separate experiment.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

# Target sentinel for "no supervised label this decision" — the belief did not hold the move they
# clicked, or the trace could not name their action. Must be negative (a valid class is >= 0) and
# is what `cross_entropy(ignore_index=...)` skips.
INTENT_IGNORE = -100


class AlphaIntentHead(torch.nn.Module):
    """`α` — a distribution over [their K believed-move seats] + [SWITCH].

    Pointer-style: one SHARED scorer maps (seat features ‖ board context) → one logit, applied to
    every seat, so the head is equivariant under permuting their moves. `SWITCH` gets its own
    scorer over the board context alone, because there is no per-seat object to point at — it is
    the "none of these" option and must be scored from the position, not from a move.
    """

    def __init__(self, seat_dim: int, ctx_dim: int, hidden: int = 64):
        super().__init__()
        self.seat_dim = int(seat_dim)
        self.ctx_dim = int(ctx_dim)
        self.seat_scorer = torch.nn.Sequential(
            torch.nn.Linear(self.seat_dim + self.ctx_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.switch_scorer = torch.nn.Sequential(
            torch.nn.Linear(self.ctx_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, seat_feats: torch.Tensor, ctx: torch.Tensor,
                seat_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`seat_feats` [B,K,seat_dim], `ctx` [B,ctx_dim] → logits [B,K+1] (last column = SWITCH).

        `seat_valid` [B,K] (1 = a real seat) masks padding to -inf so the softmax never puts mass
        on a slot the belief did not fill.
        """
        B, K, _ = seat_feats.shape
        ctx_b = ctx[:, None, :].expand(B, K, ctx.shape[-1])
        seat_logits = self.seat_scorer(torch.cat([seat_feats, ctx_b], dim=-1)).squeeze(-1)  # [B,K]
        if seat_valid is not None:
            seat_logits = seat_logits.masked_fill(seat_valid < 0.5, float("-inf"))
        switch_logit = self.switch_scorer(ctx)                                              # [B,1]
        return torch.cat([seat_logits, switch_logit], dim=-1)                               # [B,K+1]


class BetaSwitchHead(torch.nn.Module):
    """`β` — given they switch, WHICH of their mons comes in.

    Pointer-style over their six team tokens, so it is equivariant under permuting their bench.
    Alive-and-not-active is enforced by the mask, not learned: an illegal target must be
    unrepresentable rather than merely unlikely, or `β` will spend capacity learning the rules.
    """

    def __init__(self, token_dim: int, ctx_dim: int, hidden: int = 64):
        super().__init__()
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(token_dim + ctx_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, their_tokens: torch.Tensor, ctx: torch.Tensor,
                candidate_mask: torch.Tensor) -> torch.Tensor:
        """`their_tokens` [B,6,D], `ctx` [B,C], `candidate_mask` [B,6] (1 = a legal switch-in)."""
        B, T, _ = their_tokens.shape
        ctx_b = ctx[:, None, :].expand(B, T, ctx.shape[-1])
        logits = self.scorer(torch.cat([their_tokens, ctx_b], dim=-1)).squeeze(-1)          # [B,6]
        out = logits.masked_fill(candidate_mask < 0.5, float("-inf"))
        # A row with NO legal switch-in (their last mon is the active one) would be all -inf, and
        # `log_softmax` of that is NaN — which propagates through `cross_entropy` and poisons the
        # WHOLE batch even though such a row's target is always IGNORE. Measured: a smoke reported
        # `beta_loss = nan` with 130 supervised rows. Leave those rows FLAT (uniform, all-finite):
        # they contribute nothing to the loss via ignore_index, and nothing can be NaN.
        dead = (candidate_mask > 0.5).sum(dim=-1) == 0                                      # [B]
        return torch.where(dead[:, None], torch.zeros_like(out), out)


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


def intent_losses(alpha_logits: Optional[torch.Tensor], alpha_target: Optional[torch.Tensor],
                  beta_logits: Optional[torch.Tensor], beta_target: Optional[torch.Tensor],
                  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Cross-entropy on both axes, each over only its supervised rows, plus the diagnostics.

    Every rate is reported against a denominator that says what it is over. `alpha_mask_rate` is
    the fraction of decisions with NO usable label — the belief's coverage failure, surfaced here
    because otherwise a rising `α` accuracy on a shrinking supervised subset looks like progress.
    """
    metrics: Dict[str, float] = {}
    total = None

    if alpha_logits is not None and alpha_target is not None:
        sup = alpha_target != INTENT_IGNORE
        n_sup = int(sup.sum())
        metrics["opp_intent/alpha_mask_rate"] = 1.0 - n_sup / max(alpha_target.numel(), 1)
        metrics["opp_intent/alpha_n_supervised"] = float(n_sup)
        if n_sup > 0:
            la = torch.nn.functional.cross_entropy(
                alpha_logits, alpha_target, ignore_index=INTENT_IGNORE)
            total = la if total is None else total + la
            with torch.no_grad():
                pred = alpha_logits[sup].argmax(dim=-1)
                tgt = alpha_target[sup]
                metrics["opp_intent/alpha_loss"] = float(la.detach())
                metrics["opp_intent/alpha_acc"] = float((pred == tgt).float().mean())
                # SWITCH is the last class; split the accuracy so a head that only learns
                # "they attack" cannot hide behind the attack-heavy base rate.
                k = alpha_logits.shape[-1] - 1
                is_sw = tgt == k
                if int(is_sw.sum()):
                    metrics["opp_intent/alpha_acc_switch"] = float(
                        (pred[is_sw] == tgt[is_sw]).float().mean())
                if int((~is_sw).sum()):
                    metrics["opp_intent/alpha_acc_move"] = float(
                        (pred[~is_sw] == tgt[~is_sw]).float().mean())
                    # THE BASELINE alpha exists to beat, and the ONLY number that makes
                    # `alpha_acc_move` interpretable. Seats are `topk(w)` DESCENDING, so seat 0 IS
                    # argmax(w) — "the belief's own top-ranked move" — and the fraction of supervised
                    # MOVE rows whose target is seat 0 is exactly how often that free guess is right.
                    # Without it a reader cannot tell IMPROVEMENT from REPRODUCTION, and reproduction
                    # is the likely default: `w` rides in the E4 seat header the alpha head reads, so
                    # the head can recover argmax(w) from its own input almost by construction.
                    # Compare LIKE FOR LIKE: this is measured on the SAME supervised-and-covered
                    # subset as `alpha_acc_move`, so it is NOT the 51.8% whole-population figure
                    # (that one divides by every move row, covered or not).
                    metrics["opp_intent/alpha_acc_move_baseline_argmax_w"] = float(
                        (tgt[~is_sw] == 0).float().mean())
                metrics["opp_intent/alpha_switch_rate"] = float(is_sw.float().mean())

    if beta_logits is not None and beta_target is not None:
        sup = beta_target != INTENT_IGNORE
        n_sup = int(sup.sum())
        metrics["opp_intent/beta_mask_rate"] = 1.0 - n_sup / max(beta_target.numel(), 1)
        metrics["opp_intent/beta_n_supervised"] = float(n_sup)
        if n_sup > 0:
            lb = torch.nn.functional.cross_entropy(
                beta_logits, beta_target, ignore_index=INTENT_IGNORE)
            total = lb if total is None else total + lb
            with torch.no_grad():
                metrics["opp_intent/beta_loss"] = float(lb.detach())
                metrics["opp_intent/beta_acc"] = float(
                    (beta_logits[sup].argmax(dim=-1) == beta_target[sup]).float().mean())

    if total is None:
        total = torch.zeros((), device=(alpha_logits.device if alpha_logits is not None
                                        else torch.device("cpu")))
    return total, metrics


def render_alpha(alpha_probs: torch.Tensor, seat_nums: torch.Tensor,
                 move_name: callable, top: int = 5) -> list:
    """G3b — `α` as a ranked list of NAMED options, the interpretability deliverable.

    This is not a debug helper: the owner constraint is that the model may only ever point at
    things it can name, and this function is what makes that checkable by a human and assertable
    by a test. Returns [{'name': str, 'p': float}], SWITCH included, highest first.
    """
    k = alpha_probs.shape[-1] - 1
    rows = []
    for i in range(k):
        num = int(seat_nums[i])
        if num <= 0:
            continue
        rows.append({"name": move_name(num) or f"move#{num}", "p": float(alpha_probs[i])})
    rows.append({"name": "SWITCH", "p": float(alpha_probs[k])})
    rows.sort(key=lambda r: -r["p"])
    return rows[:top]
