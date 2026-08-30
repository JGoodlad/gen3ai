"""The PER-ACTION win-probability loss terms (`gen3_q_winprob_head_v1`, E5 step 2: GROUND).

A self-contained VERTICAL in the `cf_terms` / `td_aux` / `belief_bank` mould: everything here reads
the label rows the counterfactual buffer already sampled, applies `fe.q_winprob_head` to the
pointer stash of the SAME forward, and returns `(weighted_term, metrics)`. It touches the PPO
update only through the `loss = loss + term` lines that call it.

It RIDES the existing cf sample and the existing cf forward (`cf_terms.cf_sample_and_forward`) —
not as an optimization but for the same reason the twin heads do: the Q head is supervised on the
same recorded states as every other counterfactual readout, and a second sample would make the
terms disagree about which states they saw for no benefit at all.

**TWO TERMS, TWO COEFFICIENTS, AND THE SPLIT IS THE POINT.**

``q_winprob_term``          the COUNTERFACTUAL term — a masked binomial likelihood over exactly the
                            (state, action) pairs a row's ``q_labels`` covers. This is the one the
                            head exists for.
``q_winprob_onpolicy_term`` the WEAK FALLBACK — the recorded battle's realized outcome as a
                            single-sample label for the ONE action that was actually taken.
                            Default OFF, and separately weighted so the two can never be confused
                            in a run's provenance.

🚨 **WHY THE FALLBACK IS DANGEROUS AND WHY IT IS NOT THE DEFAULT.** On-policy data labels exactly
one action per state, and the measured preferred-alternative rate is p≈0.002 — so a Q head trained
on that stream teaches itself only where the policy already goes and stays **untrained precisely on
the never-tried moves**, which is the entire set a per-action readout would be consulted about. A
head in that state is not merely uninformative there; it is *confidently* wrong, because the shared
scorer generalizes the taken-action signal onto the unvisited columns with nothing to correct it.
The term exists because it is cheap and because a starved-factory run should have *something* to
show, not because it is a substitute for counterfactual labels. Read
``q_winprob/label_coverage`` and ``q_winprob/labels_per_row`` before believing any number this head
produces (ledger 229e9f1).

**Both terms are HEAD-ONLY, structurally, with no switch.** The head's inputs were detached inside
the extractor forward (`q_winprob_mode` has no `shaping` value), so no coefficient can route a
gradient into the trunk and ``grad/q_winprob_share`` reads exactly 0.0 by construction — the
verification, not a defect.

Design: `designs/ai_v10/design_counterfactual_value_grounding.md` (the label factory);
ledger 229e9f1 / 5edbd05 (the E5 loop this is step 2 of).
"""
from __future__ import annotations

import torch as th
from torch.nn import functional as F


def q_masked_binomial_nll(logits, labels, n_rollouts, mask):
    """MASKED binomial NLL over a [B, A] grid of per-action labels, normalized by Σ(mask·n).

    The strict generalization of `cf_terms.cf_binomial_nll` to a partially-labelled matrix: with
    ``mask`` all ones this returns EXACTLY that function's value, which is a pinned property rather
    than a claimed one — it is what makes 'the Q head's loss is the same likelihood, restricted'
    a true statement instead of an analogy.

    Per labelled cell, with ``w = round(label·n)`` and ``q = sigmoid(logit)``::

        NLL = −[ w·log q + (n − w)·log(1 − q) ]

    i.e. ``n`` times the per-observation cross-entropy, so a 16-rollout label pulls exactly 4× a
    4-rollout one. Folded as ``Σ mask·NLL / Σ mask·n`` — mean NLL per ROLLOUT, so the coefficient
    keeps its meaning across producers with different R AND across minibatches with different
    label DENSITY. That second invariance is the one this masked form adds and it matters: a
    normalizer of ``Σn`` over the whole grid would make the term shrink as coverage fell, which is
    the opposite of what a starving factory should do to a loss.

    An UNLABELLED cell contributes exactly zero to both the numerator and the denominator — never a
    zero target, which is indistinguishable from a confident "this action loses" and is the single
    most dangerous silent label this schema could produce.

    Computed through ``softplus`` for the stability reason `cf_binomial_nll` states: −log σ(z) =
    softplus(−z) and −log(1−σ(z)) = softplus(z), both finite at large |z| where the naive form
    underflows to ``log(0)``.
    """
    n = n_rollouts.clamp(min=1.0)
    wins = th.minimum(th.round(labels * n).clamp(min=0.0), n)
    per_cell = wins * F.softplus(-logits) + (n - wins) * F.softplus(logits)
    return (per_cell * mask).sum() / (n * mask).sum().clamp(min=1.0)


def _q_logits(model, ctx):
    """Re-apply the Q head to THIS cf forward's pointer stash. ``None`` when it cannot be read.

    WHY IT RE-APPLIES THE HEAD RATHER THAN READING ``last_q_winprob_logits``. The cf forward runs
    under ``th.no_grad()`` whenever nothing downstream of it needs a graph (`cf_sample_and_forward`
    computes that condition from the *scalar* term's settings, which know nothing about this one),
    so the stashed logits may carry no graph at all — and a term folded from them would train
    exactly nothing while every metric looked healthy. Re-applying the head OUTSIDE that context
    gives the head's own parameters their gradients; the inputs it reads were detached inside the
    forward regardless, which is precisely the head-only contract.

    This is `cf_winprob_term`'s reasoning arriving at the same place from a different premise: that
    term re-applies its head because the extractor's ``win_prob_mode`` governs a *different*
    decision than ``cf_head_only``; this one because a ``no_grad`` forward is not a place to take a
    gradient from.
    """
    fe = model.policy.features_extractor
    head = getattr(fe, "q_winprob_head", None)
    if head is None:
        return None
    inputs = fe.last_pointer_inputs
    if inputs is None:                              # pragma: no cover - defensive
        return None
    pooled = ctx.value_pooled
    if inputs.move_tokens.shape[0] != pooled.shape[0]:
        # Structurally impossible — one forward writes both stashes — so a disagreement means the
        # cf fold and the pointer stash are wired to different forwards, which would silently
        # supervise the head on one batch's actions and another batch's board. Fail loud, the
        # `_critic_value` stale-stash precedent.
        raise RuntimeError(
            f"stale pointer stash in the Q fold: pointer batch {inputs.move_tokens.shape[0]} vs "
            f"value_pooled {pooled.shape[0]} — the cf forward and this read are from different "
            f"batches.")
    return head(pooled.detach(), inputs.move_tokens.detach(), inputs.move_valid.detach(),
                inputs.team_tokens.detach(), inputs.move_cells.detach(),
                inputs.switch_cells.detach())


def q_winprob_term(model, ctx):
    """The PER-ACTION counterfactual grounding term for ONE minibatch.

    Returns ``(weighted_term, metrics)`` — ``(None, {...})`` when there is no sample, no head, or
    no row in the sample carries a per-action label. The no-coverage case publishes a
    ``coverage: 0.0`` metric rather than returning silently: a Q head whose producer ships no
    ``q_labels`` is the starvation case, and it must not look like a healthy head with nothing to
    say (the `cf_shadow_term` precedent, and the search-teacher's silent starvation before it).
    """
    if ctx is None:
        return None, {}
    logits = _q_logits(model, ctx)
    if logits is None:
        return None, {}
    b = ctx.batch
    mask, labels, n_roll = b.q_mask, b.q_label, b.q_n
    n_cells = float(mask.sum())
    n_rows = float(ctx.n_rows)
    coverage = {
        # The fraction of the eleven actions an average sampled ROW carries a label for. This is
        # the number that separates "the factory is running" from "the factory is running at one
        # action per state" — i.e. from the on-policy starvation this head exists to avoid.
        "labels_per_row": n_cells / max(1.0, n_rows),
        # The fraction of sampled rows carrying ANY per-action label.
        "label_coverage": float((mask.sum(-1) > 0).float().mean()),
        "n": n_rows,
    }
    if n_cells < 1.0:
        return None, coverage
    loss = q_masked_binomial_nll(logits, labels, n_roll, mask)
    metrics = dict(coverage)
    metrics["loss"] = float(loss)
    with th.no_grad():
        probs = th.sigmoid(logits)
        sel = mask > 0.5
        metrics["n_rollouts_mean"] = float((n_roll * mask).sum() / mask.sum())
        metrics["pred_mean"] = float(probs[sel].mean())
        metrics["label_mean"] = float(labels[sel].mean())
        metrics["bias"] = float((probs - labels)[sel].mean())
        metrics["abs_err"] = float((probs - labels)[sel].abs().mean())
        # THE AMORTIZATION READ, live and per minibatch (E5 step 5). `spread` is how far apart the
        # head puts the labelled actions of a state; a head that has learned nothing per-ACTION
        # can still score well on `abs_err` by predicting each state's mean, and this is the
        # column that tells the two apart. Computed only over rows with >= 2 labelled actions,
        # because a one-action row's spread is 0 by construction and would dilute the mean toward
        # exactly the wrong conclusion.
        multi = mask.sum(-1) >= 2.0
        if bool(multi.any()):
            big = th.where(sel, probs, th.full_like(probs, -1.0))
            small = th.where(sel, probs, th.full_like(probs, 2.0))
            pred_spread = (big.amax(-1) - small.amin(-1))[multi]
            big_l = th.where(sel, labels, th.full_like(labels, -1.0))
            small_l = th.where(sel, labels, th.full_like(labels, 2.0))
            label_spread = (big_l.amax(-1) - small_l.amin(-1))[multi]
            metrics["pred_spread"] = float(pred_spread.mean())
            metrics["label_spread"] = float(label_spread.mean())
            metrics["multi_action_rows"] = float(multi.float().mean())
    return model.q_winprob_coef * loss, metrics


def q_winprob_onpolicy_term(model, ctx):
    """The WEAK on-policy fallback — the recorded outcome, at the ONE action that was taken.

    Returns ``(weighted_term, metrics)`` — ``(None, {...})`` when no sampled row carries both a
    ``taken_action`` and an ``outcome_label``.

    🚨 **BIASED BY CONSTRUCTION.** See this module's header: an on-policy label covers one action of
    eleven, drawn from the policy's own choices, so this term teaches the head where the policy
    already goes. Its coefficient is separate from the counterfactual one, defaults to 0.0, and
    every metric it publishes is prefixed ``onpolicy_`` so no reader can mistake its numbers for
    the grounded stream's.

    n ≡ 1 by construction — a single realized outcome IS one observation — which under
    `q_masked_binomial_nll`'s Σ(mask·n) normalization gives each of its rows the same per-row
    gradient magnitude as a counterfactual row, with only the TARGET differing. That is deliberate:
    if the two streams are ever run together, the difference between them must be the label, not an
    effective learning rate.
    """
    if ctx is None:
        return None, {}
    logits = _q_logits(model, ctx)
    if logits is None:
        return None, {}
    b = ctx.batch
    row_mask = b.taken_mask
    n_on = float(row_mask.sum())
    metrics = {"onpolicy_coverage": n_on / max(1.0, float(ctx.n_rows)),
               "n": float(ctx.n_rows)}
    if n_on < 1.0:
        return None, metrics
    # Scatter the one taken action per row into a [B, A] mask, so the SAME masked likelihood folds
    # both streams. One loss function, two label sources — the twin heads' rule.
    cell_mask = th.zeros_like(b.q_mask)
    cell_mask.scatter_(1, b.taken_action.long()[:, None], row_mask[:, None])
    targets = th.zeros_like(b.q_label)
    targets.scatter_(1, b.taken_action.long()[:, None], b.outcome[:, None])
    ones = th.ones_like(cell_mask)
    loss = q_masked_binomial_nll(logits, targets, ones, cell_mask)
    metrics["onpolicy_loss"] = float(loss)
    with th.no_grad():
        sel = cell_mask > 0.5
        probs = th.sigmoid(logits)
        metrics["onpolicy_pred_mean"] = float(probs[sel].mean())
        metrics["onpolicy_label_mean"] = float(targets[sel].mean())
        metrics["onpolicy_bias"] = float((probs - targets)[sel].mean())
    return model.q_winprob_onpolicy_coef * loss, metrics
