"""The COUNTERFACTUAL loss terms — one sample, one forward, four consumers.

Split out of `instrumented_ppo.py` (which the file-size ratchet caught growing by 378 lines when
the twin heads landed) for the reason `belief_bank.py` and `td_aux.py` were: this is a
self-contained VERTICAL. Everything here reads label rows off `model._cf_buffer`, applies a head to
the extractor's stashed `value_pooled`, and returns `(weighted_term, metrics)` — it touches the PPO
update only through the `loss = loss + term` line that calls it.

The functions take the MODEL as their first argument rather than living on it, so each is testable
against a small stand-in; `InstrumentedMaskablePPO` keeps one-line delegating methods so the call
sites and the existing `model._cf_*` tests are unchanged.

The four consumers, and what each is FOR:

  `cf_winprob_term`    the scalar R1 term  — tight-MC labels into head A's `win_head`
  `cf_evidential_term` the Beta readout    — the head that CONFESSES its blur (v98)
  `cf_twin_terms`      heads B and C       — the WITHIN-RUN paired comparison (v99)
  `cf_shadow_term`     the shadow critic   — an MC-grounded value twin, passive (v99)

Design: `designs/ai_v10/design_counterfactual_value_grounding.md`; the arm's rules:
`designs/research_state/cf_r1_runbook.md`.
"""
from __future__ import annotations

import contextlib
from typing import Any, NamedTuple

import torch as th
from torch.nn import functional as F


class CfForward(NamedTuple):
    """The ONE sample + ONE extractor forward that every counterfactual term shares.

    Named rather than positional because the block now feeds FOUR consumers off streams that are
    optional per row — and an anonymous tuple whose arity grows is precisely the order-mismatch
    shape this tree treats as drop-everything.

    ``batch``        the :class:`~agents.training.cf_label_buffer.CfBatch` — every label stream
    ``value_pooled`` [B, D_MODEL] the trunk summary every cf head reads (carries grad only in the
                     trunk-open arm; every head detaches it for itself as its contract requires)
    ``n_rows``       how many rows were sampled (the `cf/rows_sampled` numerator)
    ``vf_features``  the forward's critic features, kept ONLY so the shadow critic's divergence
                     meter can read the LIVE V on these states without a second forward. None when
                     a test double's forward does not return the pair.
    """
    batch: Any
    value_pooled: Any
    n_rows: int
    vf_features: Any


def cf_binomial_nll(logits, labels, n_rollouts):
    """EXACT binomial negative log-likelihood of the labels' win COUNTS, normalized by Σn.

    ``label`` is a RATIO (wins/rollouts) and the flat BCE ate it as if every row carried one
    observation. It does not: with ``w = round(label·n)`` and ``q = sigmoid(logit)``,

        NLL_i = −[ w_i·log q_i + (n_i − w_i)·log(1 − q_i) ]

    which is ``n_i`` times the per-observation cross-entropy — so an R=16 label pulls exactly 4×
    an R=4 one. That is not an emphasis choice; it is the likelihood of the data.

    NORMALIZATION: ``Σ NLL_i / Σ n_i`` — mean NLL per ROLLOUT. Chosen (over ``/mean(n)`` or a
    bare mean) because it makes the coefficient keep its meaning across producers with different
    R, and because at ``n ≡ 1`` it reduces EXACTLY to the mean BCE the flat path computes (a
    1-rollout label is already 0 or 1, so the round is the identity and Σn = B). That exact
    agreement is pinned by a test, so 'binomial' is a strict generalisation rather than a
    different objective that happens to look similar.

    Computed through ``softplus`` rather than ``log(sigmoid(·))``: −log σ(z) = softplus(−z) and
    −log(1−σ(z)) = softplus(z), both stable at large |z| where the naive form underflows to
    ``log(0)``.
    """
    n = n_rollouts.clamp(min=1.0)
    # round-then-clamp: a producer's label is in [0,1] (the buffer enforces it), so w lands in
    # [0, n] already; the clamp is belt-and-braces against a float-rounding edge, and keeps
    # `n - w` non-negative, which lgamma-free though this path is, the caller relies on.
    wins = th.minimum(th.round(labels * n).clamp(min=0.0), n)
    total = wins * F.softplus(-logits) + (n - wins) * F.softplus(logits)
    return total.sum() / n.sum().clamp(min=1.0)

def cf_sample_and_forward(model):
    """Sample label rows and run the ONE extractor forward both cf terms share.

    Returns ``(labels, n_rollouts, value_pooled, n_rows)`` or ``None`` when the buffer is empty
    or absent — so a starving producer costs the train loop nothing but a `len()`.

    WHY ITS OWN SAMPLE AND ITS OWN FORWARD. The labelled states are *recorded past decisions*
    re-scored offline; they are not in this rollout at all, so there is no way to ride
    `rollout_data` — the same shape the search-teacher / OPD folds already establish here. It
    runs PER MINIBATCH for the same reason `_td_aux_term` does: a once-per-``train()`` fold
    would give the term one gradient contribution against the value loss's
    ``n_epochs x n_minibatches``, so the coefficient would stop meaning what it means for every
    other aux.

    WHY IT IS SHARED. The scalar win-prob term and the evidential Beta term supervise two
    readouts of the SAME `value_pooled` on the SAME rows; giving each its own sample would pay
    twice for the extractor forward (the whole cost of this block) and would additionally make
    the two terms disagree about which states they saw, for no reason at all.

    NOTE it CLOBBERS the extractor stashes for this minibatch (its forward replaces them), so
    it must be run after every loss that reads them — it sits beside `_td_aux_term`, which
    carries the identical constraint.

    TWO GUARDS ride on the forward itself:

    * **`no_grad` when nothing downstream needs the graph.** Under `cf_head_only` the win-prob
      term detaches its input and the evidential term detaches unconditionally, so the whole
      extractor graph is built and thrown away. The condition is computed exactly rather than
      assumed (`cf_head_only` OR a dead win-prob coefficient), because the ONE configuration
      that does need it — `--no-cf-head-only` with a live `cf_winprob_coef` — is precisely the
      trunk-open arm, where silently dropping the graph would turn the lever into a no-op.
      The heads still train either way: `head(value_pooled)` is applied OUTSIDE this context,
      so their own parameters keep their gradients whatever the input tensor carries.
    * **The ObservationDebugger is SUPPRESSED for it.** These are recorded FOREIGN states —
      other episodes, other policy steps, read off disk — and the debugger's whole premise is
      that it is looking at the board this process is about to act on. Feeding it 256 replayed
      rows per minibatch would have it report their integrity against the live env's
      expectations. Suppressed and restored, never permanently dropped (that is the compile
      path's trade, and it costs the run its only live obs-integrity check).
    """
    from agents.training.cf_label_buffer import CF_SAMPLE_SIZE, batch_tensors

    buf = getattr(model, "_cf_buffer", None)
    rows = buf.sample(CF_SAMPLE_SIZE) if buf is not None else []
    if not rows:
        return None

    fe = model.policy.features_extractor
    batch = batch_tensors(rows, model.device)
    obs = batch.obs
    needs_graph = (not model.cf_head_only) and float(getattr(model, "cf_winprob_coef", 0.0)) != 0.0
    grad_ctx = contextlib.nullcontext() if needs_graph else th.no_grad()
    # `getattr` + nullcontext: a test double / a non-Gen3 extractor has no debugger to suppress,
    # and the cf fold must not require one to exist.
    dbg_ctx = getattr(fe, "suppress_observation_debugger", contextlib.nullcontext)()
    with dbg_ctx, grad_ctx:
        # The EAGER forward, deliberately. `compile_trainer_extractor` patches the BOUND
        # `fe.forward`, so `type(fe).forward` is always the uncompiled one — and this call passes
        # an obs dict with ONLY the "observation" key (the sole key the model reads; a label
        # carries nothing else), a structure the production forward never sees. Routing it
        # through the compiled entry point would add a second graph shape, and would ask dynamo
        # to trace the `model.stash` write on a shape it exists nowhere else. The term is 256 rows
        # once per minibatch; eager is the cheap, boring answer.
        feats = type(fe).forward(fe, {"observation": obs})
    value_pooled = fe.stash.value_pooled
    if value_pooled is None:                       # pragma: no cover - defensive
        return None
    # gen3_cf_twin_heads_v1: keep the forward's `vf_features` so the SHADOW critic's divergence
    # meter can read the LIVE critic on these exact states WITHOUT paying for a second extractor
    # forward (the whole cost of this block). It is routed through the policy's own
    # `_critic_value`, never a hand-rolled value path — that method is what handles PopArt
    # de-normalization and the --value-from-dist route, and reading `value_net` directly would
    # compare the shadow against a critic the run does not use.
    vf_features = feats[1] if isinstance(feats, tuple) and len(feats) == 2 else None
    return CfForward(batch=batch, value_pooled=value_pooled, n_rows=len(rows),
                     vf_features=vf_features)

def cf_winprob_term(model, ctx):
    """The COUNTERFACTUAL win-prob grounding term for ONE minibatch.

    Returns ``(weighted_term, metrics)`` — or ``(None, {})`` when there is no sample or no head.

    WHY IT DOES NOT READ ``last_win_prob_logits``. That stash is produced under the extractor's
    own ``win_prob_mode`` (``read_only`` stop-grads, ``shaping`` does not), which governs the
    ON-POLICY win-prob BCE. This term's trunk exposure is governed by ``cf_head_only``, which is
    a separate decision — so it re-applies the head to ``stash.value_pooled`` itself, detaching
    iff ``cf_head_only``. The two settings are then independent by construction, and head-only
    really is head-only whatever the mode says.
    """
    if ctx is None:
        return None, {}
    labels, n_rollouts = ctx.batch.label, ctx.batch.n_rollouts
    value_pooled, n_rows = ctx.value_pooled, ctx.n_rows

    fe = model.policy.features_extractor
    head = getattr(fe, "win_head", None)
    if head is None:
        # Guarded at the call site too; belt and braces, because a silent no-op here would be
        # a coefficient that reads as "on" and teaches nothing.
        return None, {}

    logits = head(value_pooled.detach() if model.cf_head_only else value_pooled).flatten()
    if str(getattr(model, "cf_label_likelihood", "binomial")) == "bce":
        loss = th.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    else:
        loss = cf_binomial_nll(logits, labels, n_rollouts)
    with th.no_grad():
        probs = th.sigmoid(logits)
        metrics = {
            "loss": float(loss),
            "n": float(n_rows),
            # The mean EVIDENCE behind the sampled labels. Read beside `loss`: under the
            # binomial likelihood the loss is per-rollout, so a producer that quietly changed R
            # would otherwise move the loss with no visible cause.
            "n_rollouts_mean": float(n_rollouts.mean()),
            "pred_mean": float(probs.mean()),
            "label_mean": float(labels.mean()),
            # The G0 meter's live counterpart: SIGNED mean(pred - MC). The G0 bias map's
            # amendment says the head's defect is RESOLUTION, not this offset — so read it as
            # a no-harm watch, NOT as the thing the arm is trying to move.
            "bias": float((probs - labels).mean()),
            "abs_err": float((probs - labels).abs().mean()),
        }
    return model.cf_winprob_coef * loss, metrics

def cf_evidential_term(model, ctx):
    """The EVIDENTIAL Beta term for ONE minibatch — the head's UNCERTAINTY, supervised.

    Returns ``(weighted_term, metrics)`` — or ``(None, {})`` when there is no sample or the head
    is not built.

    WHAT IT IS FOR. G0 convicted the scalar win-prob head of BLUR, not bias: within a confidence
    decile the true P(win) spread is 0.11–0.36, which the point estimate cannot represent at
    all. This term does not remove the blur — it reads the same `value_pooled` — it makes the
    head CONFESS it, by fitting a Beta whose width can be wide exactly where the states behind a
    bin disagree.

    THE LOSS is the Beta-Binomial MARGINAL likelihood of the row's counts (p integrated out, not
    plugged in), which is the correct evidential objective for count data: it pulls the mean
    toward w/n AND grows the precision α+β only as far as consistency across states supports.
    Normalized by Σn like the scalar term, so the two coefficients are in the same units (nats
    per rollout). The KL pull toward Beta(1,1) rides INSIDE the coefficient at
    ``cf_evidential_reg``, so coefficient zero kills the whole term including the regularizer.

    ALWAYS DETACHED. Unlike `_cf_winprob_term` there is no `head_only` switch: this head is a
    pure supervised readout that feeds nothing forward, so letting it shape the trunk would be
    a training change with no consumer to justify it. `grad/cf_evidential_share` therefore reads
    exactly 0.0 by construction — the verification, not a defect.
    """
    if ctx is None:
        return None, {}
    labels, n_rollouts = ctx.batch.label, ctx.batch.n_rollouts
    value_pooled, n_rows = ctx.value_pooled, ctx.n_rows

    fe = model.policy.features_extractor
    head = getattr(fe, "cf_evid_head", None)
    if head is None:
        return None, {}

    alpha, beta = head(value_pooled.detach())
    n = n_rollouts.clamp(min=1.0)
    wins = th.minimum(th.round(labels * n).clamp(min=0.0), n)
    nll = head.beta_binomial_nll(alpha, beta, wins, n).sum() / n.sum().clamp(min=1.0)
    reg = head.kl_to_uniform(alpha, beta).mean()
    term = nll + float(model.cf_evidential_reg) * reg
    with th.no_grad():
        precision = alpha + beta
        metrics = {
            "nll": float(nll),
            "reg": float(reg),
            "n": float(n_rows),
            "alpha_mean": float(alpha.mean()),
            # α+β is the EVIDENCE the head claims. Watch it against `cf_evidential_reg`: an
            # unbounded climb is the evidential-overconfidence failure the KL exists to bound.
            "precision_mean": float(precision.mean()),
            # THE HEADLINE. The pre-registered read for the future A/B is that this width,
            # per stratum, tracks `cf_audit`'s measured `sd_true_excess` for that stratum.
            "epistemic_std_mean": float(head.epistemic_std(alpha, beta).mean()),
            "pred_mean": float((alpha / precision).mean()),
        }
    # A stash for a future per-decision trace capture (the prober reads `win_probs` from the
    # npz the same way). It lives on the EXTRACTOR, beside `last_win_prob_logits`, for the
    # reason that convention exists: a plain attribute there rides no state_dict and no
    # checkpoint, whereas the same attribute on the model would be pickled into every save.
    # Detached + on CPU so nothing holds a graph or device memory.
    fe.last_cf_evidential = (alpha.detach().cpu(), beta.detach().cpu())
    return model.cf_evidential_coef * term, metrics

# -- gen3_cf_twin_heads_v1: the TWIN heads and the SHADOW critic ---------------------------
#
# THE AMENDMENT, in one paragraph. R1's signed pre-registration compared two RUNS. Two runs
# differ in every random draw they ever make, and the primary meter's own measured floor is
# ~40% of its variance — so a cross-run difference has to clear noise the design cannot
# control. Three win-prob heads on ONE trunk delete that noise by construction: identical
# trunk, identical states, and the ONLY thing that differs is which label stream trains each
# head. Authorized by the owner 2026-08-22 (ledger, "Three owner sign-offs" item 3).

def cf_twin_onpolicy_terms(model, rollout_data):
    """Head A's OWN loss, mirrored onto twins B and C on the SAME rollout minibatch.

    Returns ``(term_or_None, metrics)``. This is the "A's loss PLUS" half of the factorial: B
    and C must carry a bit-identical copy of the control's objective, or B−A would confound
    "extra states" with "a different base objective" and the factorial would decompose nothing.

    It re-applies each twin to the extractor's STASHED `value_pooled` rather than reading a
    per-head stash, for `_cf_winprob_term`'s reason: the stash `last_win_prob_logits` is
    produced under the extractor's own `win_prob_mode`, and the twins' trunk exposure is not
    that decision. **Both twins detach unconditionally** (head-only ALWAYS in v1), which is what
    makes this a measurement of the LABEL effect on a trunk frozen with respect to them.

    The weight is `win_prob_coef` — head A's own — deliberately NOT `cf_twin_coef`. The block
    as a whole is gated on `cf_twin_coef != 0` by the caller, so at coefficient zero the twins
    take no gradient at all and every parameter update is byte-identical to not building them.
    """
    fe = model.policy.features_extractor
    heads = [(n, getattr(fe, n, None)) for n in ("cf_twin_head_b", "cf_twin_head_c")]
    pooled = getattr(fe, "last_value_pooled", None)
    if pooled is None or any(h is None for _n, h in heads):
        return None, {}
    target = rollout_data.observations.get("win_target")
    mask = rollout_data.observations.get("win_mask")
    margin = rollout_data.observations.get("win_margin")
    if target is None or mask is None:
        return None, {}
    term = None
    metrics: "dict[str, float]" = {}
    for name, head in heads:
        out = model._win_prob_loss(head(pooled.detach()), target, mask, margin)
        if out is None:
            continue
        loss, m = out
        weighted = model.win_prob_coef * loss
        term = weighted if term is None else term + weighted
        tag = name[-1]                                   # 'b' | 'c'
        for k in ("loss", "brier", "acc"):
            if k in m:
                metrics[f"{tag}_onpolicy_{k}"] = float(m[k])
    return term, metrics

def cf_twin_terms(model, ctx):
    """The CF folds that make B and C different: the SAME states, two LABEL STREAMS.

    Returns ``(weighted_term, metrics)`` — ``(None, {})`` when there is no sample or the heads
    are not built.

    * **head B** eats the row's ``outcome_label`` — the RECORDED battle's realized win/loss,
      one Monte-Carlo sample — at **n ≡ 1**.
    * **head C** eats the row's ``label`` / ``n_rollouts`` — the tight-MC ratio and its
      evidence, the same pair head A's cf term would eat.

    THE ROUTING IS THE GIGO HERE and it is pinned by a test: B must never see a tight-MC label
    and C must never see a single-outcome one. A swap would leave every scalar looking healthy
    while the factorial silently measured its own mirror image.

    WHY THE SAME LOSS FUNCTION FOR BOTH. `_cf_binomial_nll` normalizes by ``Σ n``, so a row's
    gradient magnitude is ``(q − target)/B`` regardless of its n. B's n≡1 rows and C's n=R rows
    therefore pull EQUALLY HARD; only the target differs (a 0/1 draw vs its own tight mean).
    That equality is what makes C−B a read of label PRECISION rather than of effective learning
    rate — with two different loss forms it would be neither.

    MASKED, per row. A row that carries no ``outcome_label`` (an older producer, or `cf_audit`'s
    output) supervises B on nothing rather than on a zero — a zero-filled absent label is
    indistinguishable from a confident "you lose", which is the most dangerous silent target
    this schema could produce. B's term is skipped entirely when no row in the sample carries
    one, and `cf/twin_b_coverage` says so.

    ``metrics["loss"]`` is the COMBINED unweighted fold — C's, plus B's when B's arm actually
    ran. It is the key `train/cf_twin_loss` publishes, and it exists so the twin block reports
    the same one-scalar-per-term shape as `train/cf_loss` / `train/cf_evidential_loss` /
    `train/cf_shadow_loss`: this whole function contributes ONE `loss = loss + term` to the
    optimizer, so it gets ONE headline. The per-arm split is not lost — `cf/twin_c_loss` and
    `cf/twin_b_loss` already publish it, and they are the arm-level instrument. Summing here
    rather than in the logger is deliberate: the two arms' per-minibatch lists have DIFFERENT
    lengths (B skips a starved minibatch entirely), so `mean(c) + mean(b)` computed downstream
    would not be the mean of the term on any minibatch that actually folded.
    """
    if ctx is None:
        return None, {}
    fe = model.policy.features_extractor
    head_b = getattr(fe, "cf_twin_head_b", None)
    head_c = getattr(fe, "cf_twin_head_c", None)
    if head_b is None or head_c is None:
        return None, {}
    b = ctx.batch
    # ALWAYS detached: head-only is not a mode for the twins, it is their definition in v1.
    pooled = ctx.value_pooled.detach()
    metrics: "dict[str, float]" = {"n": float(ctx.n_rows)}
    term = None

    # ---- head C: TIGHT-MC labels (the treatment) ----
    logits_c = head_c(pooled).flatten()
    loss_c = cf_binomial_nll(logits_c, b.label, b.n_rollouts)
    term = model.cf_twin_coef * loss_c
    combined = float(loss_c)                          # the headline; += B's below when B runs
    with th.no_grad():
        q_c = th.sigmoid(logits_c)
        metrics["c_loss"] = float(loss_c)
        metrics["c_pred_mean"] = float(q_c.mean())
        metrics["c_abs_err"] = float((q_c - b.label).abs().mean())
        metrics["c_bias"] = float((q_c - b.label).mean())

    # ---- head B: SINGLE-OUTCOME labels (the coverage arm) ----
    n_outcome = float(b.outcome_mask.sum())
    metrics["b_coverage"] = n_outcome / max(1.0, float(ctx.n_rows))
    if n_outcome >= 1.0:
        sel = b.outcome_mask > 0.5
        logits_b = head_b(pooled).flatten()[sel]
        outcome = b.outcome[sel]
        # n ≡ 1: a realized outcome IS one observation. Passing `ones_like` rather than reusing
        # `n_rollouts` is the whole point of the arm — B is A's evidence on C's states.
        ones = th.ones_like(outcome)
        loss_b = cf_binomial_nll(logits_b, outcome, ones)
        term = term + model.cf_twin_coef * loss_b
        combined += float(loss_b)
        with th.no_grad():
            q_b = th.sigmoid(logits_b)
            metrics["b_loss"] = float(loss_b)
            metrics["b_pred_mean"] = float(q_b.mean())
            metrics["b_abs_err"] = float((q_b - outcome).abs().mean())
            metrics["b_bias"] = float((q_b - outcome).mean())
            # THE PAIRED READ, live and per minibatch: |B−C| on the SAME states, which is the
            # within-run difference the whole amendment exists to measure. The AUDIT's
            # held-out, battle-clustered version is the result; this is the launch-window tell
            # that the two heads have actually diverged rather than converged to one function.
            metrics["b_vs_c_abs"] = float((q_b - q_c[sel]).abs().mean())
            metrics["b_minus_c"] = float((q_b - q_c[sel]).mean())
    metrics["loss"] = combined
    return term, metrics

def cf_shadow_term(model, ctx, popart):
    """The SHADOW CRITIC's masked MSE against tight-MC ``mc_return`` labels.

    Returns ``(weighted_term, metrics)`` — ``(None, {})`` when there is no sample, no head, or
    no row in the sample carries an ``mc_return``.

    THE FRAME. Under PopArt the live critic trains in normalized space, so the shadow does too:
    the head's raw output IS the normalized value and the target is
    ``popart.normalize(mc_return)``. That mirrors `_value_distill_mse` exactly, and for the
    same reason — the coefficient stays scale-comparable with the value loss, and a raw-unit
    MSE against a normalizer that moves is a moving target. With PopArt off both maps are the
    identity.

    THE METER, which is the point of the head. `shadow_vs_live_v` is the SIGNED mean of
    (shadow − live V) in REAL units on the same minibatch states, read through the policy's own
    `_critic_value` (the method that handles PopArt de-normalization and the --value-from-dist
    route). That divergence, accumulated over a run, is the staged-promotion evidence: a shadow
    that sits systematically BELOW the live critic is a live critic that is optimistic about
    the states the factory sampled, measured against ground truth rather than argued from a
    calibration curve. It is a MEASUREMENT and nothing else — the head never computes an
    advantage, never enters GAE, and feeds nothing forward.
    """
    if ctx is None:
        return None, {}
    fe = model.policy.features_extractor
    head = getattr(fe, "cf_shadow_head", None)
    if head is None:
        return None, {}
    b = ctx.batch
    m = b.mc_return_mask
    n_on = m.sum()
    if float(n_on) < 1.0:
        # Published as a zero-coverage metric rather than silence: a shadow critic with a
        # producer that ships no mc_return is the starvation case, and it must not look like a
        # healthy head with nothing to say.
        return None, {"coverage": 0.0, "n": float(ctx.n_rows)}
    pred = head(ctx.value_pooled.detach()).flatten()          # NORMALIZED frame (see docstring)
    target = popart.normalize(b.mc_return) if popart is not None else b.mc_return
    se = (pred - target) ** 2
    loss = (se * m).sum() / n_on.clamp(min=1e-6)
    metrics = {"loss": float(loss), "n": float(ctx.n_rows),
               "coverage": float(n_on) / max(1.0, float(ctx.n_rows))}
    with th.no_grad():
        real = popart.denormalize(pred) if popart is not None else pred
        real_target = b.mc_return
        metrics["pred_mean"] = float((real * m).sum() / n_on)
        metrics["label_mean"] = float((real_target * m).sum() / n_on)
        metrics["abs_err"] = float(((real - real_target).abs() * m).sum() / n_on)
        live = cf_live_values(model, ctx)
        if live is not None:
            # SIGNED, and the sign is the whole reading: negative = the shadow says these states
            # are worth LESS than the live critic thinks.
            metrics["shadow_vs_live_v"] = float(((real - live) * m).sum() / n_on)
            metrics["shadow_vs_live_v_abs"] = float(((real - live).abs() * m).sum() / n_on)
            metrics["live_v_vs_label"] = float(((live - real_target) * m).sum() / n_on)
    return model.cf_shadow_coef * loss, metrics

def cf_live_values(model, ctx):
    """The LIVE critic's real-unit V on the cf rows, or None when it cannot be read.

    Routed through `policy._critic_value` on the cf forward's own `vf_features` — never a
    hand-rolled `value_net` call, which under `--value-from-dist` would read a frozen head the
    run does not use, and never a second `predict_values` forward, which would double the cf
    block's cost for a diagnostic.
    """
    vf = ctx.vf_features
    if vf is None or not hasattr(model.policy, "_critic_value"):
        return None
    try:
        with th.no_grad():
            latent_vf = model.policy.mlp_extractor.forward_critic(vf)
            return model.policy._critic_value(latent_vf).flatten()
    except Exception:                                  # pragma: no cover - diagnostic only
        # A meter must never be able to fail a training step. A test double whose policy has no
        # mlp_extractor loses the column; the run keeps going.
        return None

