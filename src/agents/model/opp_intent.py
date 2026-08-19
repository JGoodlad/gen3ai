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

from typing import Any, Callable, Dict, List, Optional, Tuple

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


def info_gain_nats(logits: torch.Tensor, target: torch.Tensor) -> float:
    """`H(empirical marginal) − CE(model)` in nats, over the rows given. THE headline number.

    Accuracy is the wrong instrument for a belief about an opponent, and the reason is not
    fastidiousness — it is that accuracy conflates two different things a reader needs kept apart:

    * **Irreducible entropy.** Against a uniform-random opponent the Bayes-OPTIMAL prediction IS
      uniform, and its accuracy is 1/n. A perfectly calibrated model scores 1/n there and looks
      terrible. Under a proper scoring rule it scores ~0 gain — which is the honest reading:
      *nothing here was learnable.* Our opponent pool spans a random bot, several heuristics and
      frozen selves, so the same accuracy number means something different per opponent; a scoring
      rule makes them comparable, and comparable is the whole point.
    * **Confident guessing.** Accuracy rewards a peaked wrong answer exactly as much as it punishes
      a hedged right one. Log-loss does not.

    The reference is the **empirical marginal of the targets in this batch** — "predict the base
    rate and nothing else". Chosen because it needs no extra plumbing (it is a function of `target`
    alone) and it is the honest floor: a head that has learned only *how often* they switch, and
    nothing about *when*, scores ~0.

    **It CAN go negative**, and that is a real signal rather than a bug: the model is doing worse
    than the base rate. `alpha_move_recall_top1` sitting below its `argmax(w)` baseline is exactly
    the situation where that matters, since both of those numbers move during training and their
    difference is not interpretable on its own.

    NOTE this is deliberately NOT gain over `w`, the belief's own move ranking, which would be the
    stronger reference. `w` is not passed here and threading it is a separate change; the
    `alpha_move_baseline_argmax_w` comparison remains the (imperfect) stand-in for it — and it is
    only meaningful against `alpha_move_recall_top1`, which is restricted to the MOVE classes and so
    asks the same question the baseline does. Comparing it against a number that also had to decide
    move-vs-switch is what produced a wrong read once.
    """
    tgt = target.reshape(-1)
    n_classes = logits.shape[-1]
    counts = torch.bincount(tgt, minlength=n_classes).float()
    p = counts / counts.sum().clamp_min(1.0)
    nz = p > 0
    h_marginal = float(-(p[nz] * p[nz].log()).sum())
    ce_model = float(torch.nn.functional.cross_entropy(logits, tgt))
    return h_marginal - ce_model


#: `opp_class` code -> the suffix its stratified metrics carry. Mirrors
#: `MaskableAgentWrapper.OPP_CLASS_*`; kept as a plain table here so the model package does not
#: import the training package.
OPP_CLASS_NAMES = {0: "bot", 1: "pool", 2: "stable", 3: "exploiter"}

#: The one class the label weight below discounts. Named rather than spelled `0` at the use site,
#: because `OPP_CLASS_NAMES[0]` and this constant must always be the same row.
OPP_CLASS_BOT = 0


def intent_label_weights(opp_class: Optional[torch.Tensor], bot_weight: float,
                         like: torch.Tensor) -> Optional[torch.Tensor]:
    """Per-row supervision weight for the α/β labels: ``bot_weight`` on bot rows, 1.0 elsewhere.

    Returns ``None`` — meaning "do not weight at all" — when there is nothing to key on
    (``opp_class`` absent) or when the weight is exactly 1.0. That ``None`` is load-bearing: the
    callers use it to take the ORIGINAL unweighted `cross_entropy` call, so the default config is
    bit-identical rather than merely numerically equal.

    ``like`` supplies dtype/device (the logits).
    """
    if opp_class is None or bot_weight == 1.0:
        return None
    oc = opp_class.reshape(-1).to(like.device)
    ones = torch.ones(oc.shape, dtype=like.dtype, device=like.device)
    return torch.where(oc == OPP_CLASS_BOT, ones * float(bot_weight), ones)


def set_valued_switch_loss(beta_logits: torch.Tensor, believed_mask: torch.Tensor,
                           rows: torch.Tensor) -> Optional[torch.Tensor]:
    """`−log Σ_{j believed} p_j` over `rows` — PARTIAL CREDIT for "someone we have not seen".

    The gap this closes. `β`'s target is the believed slot whose species posterior holds the mon
    that came in (`resolve_believed_slot_by_content`). When no slot clears the floor — we did not
    believe that mon was there — the row is MASKED and contributes nothing. But we still know
    something true and useful about it: **they switched to a mon we had not revealed**, and `β`
    should have put mass on the believed set rather than on the revealed bench.

    A set-valued target says exactly that and no more. It grades the coarse call — *"you were right
    that it would be someone unseen"* — without asserting which member, which is the part we
    genuinely cannot label. Sharpening within the set is left to the species belief, whose job that
    is; this term must not smear a fake per-slot target across it.

    This is the one concrete thing the flat-11 redesign would have bought, obtained additively
    instead: the loss is a proper log-loss on the aggregated probability of a SET, so it is the
    same object a flat head's set-valued loss would optimize, applied to the head we already have.

    Rows with no believed-AND-legal slot are dropped rather than scored: their target set is empty,
    so `−log 0` would charge an unbounded loss for a constraint no parameter setting can satisfy.
    Returns None when no row qualifies (the caller must not add a zero — see the NaN note in
    `BetaSwitchHead`).
    """
    finite = torch.isfinite(beta_logits)
    avail = (believed_mask > 0.5) & finite                       # [B,6] believed AND legal
    rows = rows & (avail.sum(dim=-1) > 0)
    if not bool(rows.any()):
        return None
    p = torch.softmax(beta_logits[rows].float(), dim=-1)         # illegal slots are -inf -> 0
    mass = (p * avail[rows].float()).sum(dim=-1)
    return -(mass.clamp_min(1e-8).log()).mean()


def _alpha_subset_metrics(logits: torch.Tensor, tgt: torch.Tensor, k: int,
                          sfx: str = "") -> Dict[str, float]:
    """Every `α` diagnostic over ONE row subset. `sfx` is `""` for the pooled read, `"_<class>"`
    for a stratified one — so the pooled and per-opponent numbers are the SAME computation and
    cannot drift apart, which is the failure a second hand-written copy would eventually produce.

    Callers pass rows already selected (supervised, and optionally masked to one opponent class).
    """
    out: Dict[str, float] = {}
    pred = logits.argmax(dim=-1)
    out[f"opp_intent/alpha_acc{sfx}"] = float((pred == tgt).float().mean())
    # NAMING. Everything below that conditions on the TRUE class is a RECALL and is named one;
    # `alpha_acc` above is the only genuine ACCURACY here (exact class over all K+1, conditioned
    # on nothing).
    #
    # PRECISION is added for the KIND decision and ONLY there, because that is the only place it
    # carries information. Kind is binary, so "when we say switch, how often is it a switch" is a
    # different question from "of the switches, how many did we catch" — and a switch-biased head
    # scores well on the second while failing the first. The TARGET metrics are single-label top-1
    # over 6 slots / K seats, where exactly one class is predicted per row, so micro-precision,
    # micro-recall and accuracy are the SAME NUMBER; emitting a "precision" there would be a
    # duplicate column wearing a different name.
    is_sw = tgt == k
    said_sw = pred == k

    # --- axis 1 + 2: the KIND decision, both directions -----------------------------
    if int(is_sw.sum()):
        out[f"opp_intent/alpha_switch_recall{sfx}"] = float((pred[is_sw] == k).float().mean())
    if int(said_sw.sum()):
        out[f"opp_intent/alpha_switch_precision{sfx}"] = float((tgt[said_sw] == k).float().mean())
    if int((~is_sw).sum()):
        out[f"opp_intent/alpha_move_kind_recall{sfx}"] = float((pred[~is_sw] != k).float().mean())
    if int((~said_sw).sum()):
        out[f"opp_intent/alpha_move_kind_precision{sfx}"] = float(
            (tgt[~said_sw] != k).float().mean())
    out[f"opp_intent/alpha_pred_switch_rate{sfx}"] = float(said_sw.float().mean())

    # --- axis 4: WHICH move, given they moved ---------------------------------------
    if int((~is_sw).sum()):
        mv = ~is_sw
        move_logits = logits[mv][:, :k]
        mv_tgt = tgt[mv]
        out[f"opp_intent/alpha_move_recall_top1{sfx}"] = float(
            (move_logits.argmax(dim=-1) == mv_tgt).float().mean())
        # TOP-2: "nearly had it" is a different state from "no idea", and with K seats the two are
        # indistinguishable in top-1 alone. A head whose top-2 covers the truth is one a consumer
        # can use as a DISTRIBUTION even when its argmax is wrong — which is exactly how
        # --intent-value-reduce and --intent-move-cell consume alpha (as weights, not a decision).
        if move_logits.shape[-1] >= 2:
            top2 = move_logits.topk(2, dim=-1).indices
            out[f"opp_intent/alpha_move_recall_top2{sfx}"] = float(
                (top2 == mv_tgt[:, None]).any(dim=-1).float().mean())
        # The baseline this must beat: seats are topk(w) DESCENDING, so seat 0 IS the belief's own
        # top-ranked move, and this is how often that free guess is right. Compared LIKE FOR LIKE
        # with top1 above — both are "given they moved".
        out[f"opp_intent/alpha_move_baseline_argmax_w{sfx}"] = float((mv_tgt == 0).float().mean())
    out[f"opp_intent/alpha_switch_rate{sfx}"] = float(is_sw.float().mean())

    # THE HEADLINE. See `info_gain_nats`: a proper scoring rule, so an unpredictable opponent
    # scores ~0 instead of scoring "wrong", and the number is comparable across opponents and over
    # time in a way an accuracy delta between two moving quantities is not.
    out[f"opp_intent/alpha_info_gain_nats{sfx}"] = info_gain_nats(logits, tgt)
    # Split it the same way the accuracy is split, and for the same reason: pooling the (easy,
    # base-rate-dominated) switch axis with the (hard) move axis produces a number that moves when
    # the SWITCH RATE moves, which is not a fact about alpha.
    if int((~is_sw).sum()) > 1:
        out[f"opp_intent/alpha_info_gain_nats_move{sfx}"] = info_gain_nats(
            logits[~is_sw], tgt[~is_sw])
    return out


def _beta_subset_metrics(logits: torch.Tensor, tgt: torch.Tensor,
                         sfx: str = "") -> Dict[str, float]:
    """Every `β` diagnostic over ONE row subset — the `_alpha_subset_metrics` contract, other axis."""
    out: Dict[str, float] = {}
    # RECALL, not "accuracy": conditioned on the TRUE class (a supervised switch). Single-label
    # top-1 over 6 slots, so micro-precision == micro-recall == accuracy here — a separate
    # "precision" column would be the same number twice. Precision only earns its place on the
    # binary KIND decision (see the alpha block).
    out[f"opp_intent/beta_recall_top1{sfx}"] = float((logits.argmax(dim=-1) == tgt).float().mean())
    # TOP-2 for the same reason alpha gets one: with 6 slots, "narrowed it to two" and "no idea"
    # are indistinguishable in top-1, and they mean very different things for a reader deciding
    # whether the head has learned anything.
    if logits.shape[-1] >= 2:
        b2 = logits.topk(2, dim=-1).indices
        out[f"opp_intent/beta_recall_top2{sfx}"] = float((b2 == tgt[:, None]).any(dim=-1)
                                                         .float().mean())
    # The base rate beta must beat is "switch to whichever bench slot is most common", which is far
    # from uniform — so raw accuracy flatters it. Same instrument as alpha.
    out[f"opp_intent/beta_info_gain_nats{sfx}"] = info_gain_nats(logits, tgt)
    return out


def switch_coverage_metrics(kind: torch.Tensor, need: Optional[torch.Tensor],
                            content: Optional[torch.Tensor],
                            rows: Optional[torch.Tensor] = None,
                            sfx: str = "") -> Dict[str, float]:
    """THE SWITCH-COVERAGE MATRIX over one row subset. Every voluntary switch falls in exactly one
    of three buckets, and only the third is a failure — but with just a mask rate and a miss rate a
    reader cannot tell their SIZES, and "beta is masked 73% of the time" reads as a crisis when ~62
    of those points are simply "they attacked". These are fractions of VOLUNTARY SWITCHES, so they
    sum to 1.

      revealed      the mon was already on the board -> exact slot, no belief needed. The easiest
                    label, and previously invisible.
      hidden_found  still hidden, and the species posterior placed it -> the content-addressed
                    target. This is what that path BUYS.
      hidden_missed the belief could not name it -> masked. The BELIEF's failure, and the only
                    bucket that is lost supervision.

    Module-level rather than a closure in the PPO loop so it can be tested at all: nothing covered
    this matrix, and a metric with no test is a metric that can silently read zero.

    `need`/`content` are None when the belief head is off — then every switch is `revealed` by
    construction, which is the truth, not a fallback.
    """
    out: Dict[str, float] = {}
    if rows is not None:
        kind = kind[rows]
        need = None if need is None else need[rows]
        content = None if content is None else content[rows]
    n_sw = float((kind == 1).float().sum())
    if n_sw <= 0:
        return out
    want = 0.0 if need is None else float(need.float().sum())
    got = 0.0 if (need is None or content is None) else float((need & (content >= 0)).float().sum())
    out[f"opp_intent/beta_switch_n{sfx}"] = n_sw
    out[f"opp_intent/beta_switch_to_revealed{sfx}"] = (n_sw - want) / n_sw
    out[f"opp_intent/beta_switch_to_hidden_found{sfx}"] = got / n_sw
    out[f"opp_intent/beta_switch_to_hidden_missed{sfx}"] = (want - got) / n_sw
    # SPLIT `beta_mask_rate`, which conflates two failures with opposite meanings. Its denominator
    # is every row, so it is dominated by "this decision was not a switch at all" — expected,
    # uninteresting, roughly constant. Buried inside it is the one a reader wants: of the switches
    # that NEEDED the belief, how often was the belief too cold to name the mon? That is the
    # BELIEF's failure and must stay attributable to the belief. Sliced per opponent for the same
    # reason as everything else here — a cold belief against ONE class is a different diagnosis
    # from a uniformly cold one.
    if want > 0:
        out[f"opp_intent/beta_belief_miss_rate{sfx}"] = 1.0 - got / want
    return out


def intent_losses(alpha_logits: Optional[torch.Tensor], alpha_target: Optional[torch.Tensor],
                  beta_logits: Optional[torch.Tensor], beta_target: Optional[torch.Tensor],
                  opp_class: Optional[torch.Tensor] = None,
                  bot_label_weight: float = 1.0,
                  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Cross-entropy on both axes, each over only its supervised rows, plus the diagnostics.

    Every rate is reported against a denominator that says what it is over. `alpha_mask_rate` is
    the fraction of decisions with NO usable label — the belief's coverage failure, surfaced here
    because otherwise a rising `α` accuracy on a shrinking supervised subset looks like progress.

    **`bot_label_weight` (`--intent-label-bot-weight`, `gen3_intent_label_bot_weight_v1`) is a
    per-sample weight on rows whose opponent was a heuristic BOT** (`opp_class == OPP_CLASS_BOT`);
    every other class keeps 1.0. It exists because a bot's tendencies are not the meta's: the
    self-play ramp trains 100% vs bots until the pool seeds, so early α/β supervision is
    bot-DOMINATED (measured on gen-11: supervised rows 100% bot at 2M, ~7% from 6M on) and the head
    can imprint on a decision tree before it ever sees a player.

    **Semantics: folded BEFORE the mean, and the denominator stays `n_sup`** — `Σ w_i·ce_i / n_sup`,
    NOT `Σ w_i·ce_i / Σ w_i`. Normalising by `Σw` would make a 100%-bot minibatch identical to an
    unweighted one, i.e. exactly nothing during the ramp this knob exists for. With the chosen
    denominator, `w ≡ 1` reproduces the plain mean, so the existing `opp_intent_coef` semantics are
    unchanged and a mixed batch is discounted in proportion to its bot share.

    **It COMPOSES with the masks, it does not interact with them.** A row masked by `INTENT_IGNORE`
    (unmodeled seat, unrevealed β switch-in, non-switch decision) is dropped FIRST; the weight then
    multiplies only the survivors. A masked bot row contributes nothing at any weight, and `w = 0`
    is a legal setting that means "score bot rows for the metrics but train on none of them".

    **It is deliberately confined to α/β.** The other supervised beliefs — species / move / item /
    spread / HP-type — are TEAM truth: what the opponent's team IS does not depend on who is
    piloting it, so discounting a bot's rows there would throw away valid labels. Only INTENT is
    behaviour, and only behaviour is opponent-specific.
    """
    metrics: Dict[str, float] = {}
    total = None
    # None ⇒ "no weighting" ⇒ take the original unweighted call below (bit-identical default).
    w_all = intent_label_weights(
        opp_class, bot_label_weight,
        alpha_logits if alpha_logits is not None else (
            beta_logits if beta_logits is not None else torch.zeros(())))

    if alpha_logits is not None and alpha_target is not None:
        sup = alpha_target != INTENT_IGNORE
        n_sup = int(sup.sum())
        metrics["opp_intent/alpha_mask_rate"] = 1.0 - n_sup / max(alpha_target.numel(), 1)
        metrics["opp_intent/alpha_n_supervised"] = float(n_sup)
        if n_sup > 0:
            if w_all is None:
                la = torch.nn.functional.cross_entropy(
                    alpha_logits, alpha_target, ignore_index=INTENT_IGNORE)
            else:
                # Mask FIRST (the survivors), weight SECOND, and divide by `n_sup` — see the
                # docstring for why the denominator is the row COUNT and not `Σw`.
                per = torch.nn.functional.cross_entropy(
                    alpha_logits[sup], alpha_target[sup], reduction="none")
                la = (per * w_all[sup]).sum() / n_sup
            total = la if total is None else total + la
            with torch.no_grad():
                tgt = alpha_target[sup]
                k = alpha_logits.shape[-1] - 1
                metrics["opp_intent/alpha_loss"] = float(la.detach())
                if opp_class is not None:
                    # THE EXPOSURE NUMBER for `--intent-label-bot-weight`: what share of the rows
                    # α actually trained on this minibatch came from a bot. The per-class
                    # `alpha_n_supervised_*` counts below carry the same information, but they are
                    # gated on ≥2 rows and are counts, so nothing reports the RATIO — and a ratio
                    # nobody reports is a ratio nobody reads. Emitted whether or not the weight is
                    # set, because the decision to set it is made off this number.
                    metrics["opp_intent/label_bot_frac"] = float(
                        (opp_class.reshape(-1)[sup] == OPP_CLASS_BOT).float().mean())
                metrics.update(_alpha_subset_metrics(alpha_logits[sup], tgt, k))
                # STRATIFY BY OPPONENT KIND — EVERY axis, not just the headline. The pooled number
                # averages over populations where "predict their move" is a different problem:
                # against the RANDOM bot the optimal prediction is uniform and the achievable gain
                # is ~0 BY CONSTRUCTION; against a heuristic it is high but measures a decision
                # tree; only `pool` (frozen selves) measures the thing high-quality opponent
                # reasoning is for. Reporting one mean over all three is how a real deficit and a
                # favourable mix become indistinguishable. A near-zero `bot` gain is the EXPECTED
                # reading, not a failure.
                #
                # The split used to cover only acc / info-gain / n, which left the four AXIS
                # metrics — the ones a reader actually uses to locate a deficit — pooled. That is
                # not a cosmetic gap: the bot share of supervised rows on gen-11 ran 100% at 2M and
                # ~7% from 6M on, and bot rows score DIFFERENTLY (measured 2026-08-15: info gain
                # 0.124 nats vs pool 0.254). So a pooled axis metric RISES as the mix shifts toward
                # the pool, and that rise is indistinguishable from the head getting better.
                if opp_class is not None:
                    oc = opp_class.reshape(-1)[sup]
                    for _code, _name in OPP_CLASS_NAMES.items():
                        m = oc == _code
                        n_m = int(m.sum())
                        if n_m < 2:                       # a 1-row marginal has zero entropy
                            continue
                        metrics[f"opp_intent/alpha_n_supervised_{_name}"] = float(n_m)
                        metrics.update(_alpha_subset_metrics(
                            alpha_logits[sup][m], tgt[m], k, f"_{_name}"))

    if beta_logits is not None and beta_target is not None:
        sup = beta_target != INTENT_IGNORE
        n_sup = int(sup.sum())
        metrics["opp_intent/beta_mask_rate"] = 1.0 - n_sup / max(beta_target.numel(), 1)
        metrics["opp_intent/beta_n_supervised"] = float(n_sup)
        if n_sup > 0:
            if w_all is None:
                lb = torch.nn.functional.cross_entropy(
                    beta_logits, beta_target, ignore_index=INTENT_IGNORE)
            else:
                # Same rule as α, and the SAME weight vector — β's supervised subset is only the
                # voluntary switches, so its bot share differs from α's, but the per-ROW weight
                # cannot: both losses grade the same opponent's behaviour on the same decision.
                per = torch.nn.functional.cross_entropy(
                    beta_logits[sup], beta_target[sup], reduction="none")
                lb = (per * w_all[sup]).sum() / n_sup
            total = lb if total is None else total + lb
            with torch.no_grad():
                metrics["opp_intent/beta_loss"] = float(lb.detach())
                _bl, _bt = beta_logits[sup], beta_target[sup]
                metrics.update(_beta_subset_metrics(_bl, _bt))
                # Same stratification, same reason as alpha — and beta needs it MORE, because its
                # supervised rows are only the switches, so one opponent class can dominate the
                # subset even when it is a minority of decisions.
                if opp_class is not None:
                    _oc = opp_class.reshape(-1)[sup]
                    for _code, _name in OPP_CLASS_NAMES.items():
                        _m = _oc == _code
                        _n_m = int(_m.sum())
                        if _n_m < 2:
                            continue
                        metrics[f"opp_intent/beta_n_supervised_{_name}"] = float(_n_m)
                        metrics.update(_beta_subset_metrics(_bl[_m], _bt[_m], f"_{_name}"))
                # THE FALSIFIER for keeping the alpha/beta split (2026-08-13). The split imposes a
                # HIERARCHY the game does not: it factors P(switch to j) = P(SWITCH)·P(j | SWITCH),
                # which is exact for disjoint outcomes and therefore loses nothing IN PRINCIPLE. The
                # way it could still cost something in PRACTICE is if beta only works when alpha has
                # already decided — i.e. beta is riding alpha's confidence rather than reading the
                # board. Then the hierarchy is doing harm and a flat head over the true action space
                # gets a real argument. Bucket beta's accuracy by alpha's switch confidence: the two
                # numbers should be COMPARABLE. A large gap is the signal to revisit the decision.
                if alpha_logits is not None:
                    with torch.no_grad():
                        p_sw = torch.softmax(alpha_logits.float(), dim=-1)[:, -1]   # [B]
                        b_ok = _bl.argmax(dim=-1) == _bt
                        conf = p_sw[sup] >= 0.5
                        for _name, _m in (("alpha_confident", conf), ("alpha_unsure", ~conf)):
                            if int(_m.sum()) > 1:
                                metrics[f"opp_intent/beta_recall_{_name}"] = float(
                                    b_ok[_m].float().mean())

    if total is None:
        total = torch.zeros((), device=(alpha_logits.device if alpha_logits is not None
                                        else torch.device("cpu")))
    return total, metrics


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
