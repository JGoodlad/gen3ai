"""THE OFF-SLICE DISTILL ANCHOR (`gen3_distill_offslice_anchor_v1`) — a trust region to the FROZEN
fold parent on the states the teachers do not cover, plus the live COLLATERAL-KL meters.

WHY IT EXISTS. A distillation fold's net effect is *teacher content MINUS overshoot damage on the
untaught distribution*. The 2026-08-31 licensing probe
(`designs/research_state/measurements/lr_licensing_probe_2026-08-31.md`) measured that lowering the
distill step cut OFF-SLICE collateral by 39% with on-slice absorption unchanged — i.e. the damage
is a SYSTEMATIC direction the distill gradient carries off the taught slice, not noise. Two things
follow, and this module is both of them:

  * **The meter.** `distill/collateral_kl` is the probe's off-slice divergence, live, every
    `train()`. It is emitted whenever a parent snapshot is attached — including at
    `--distill-anchor-coef 0` under `--distill-anchor-monitor`, which is the pure-instrument arm.
  * **The regulariser.** `--distill-anchor-coef` folds `coef · mean_off-slice KL(pi_parent || pi_student)`
    into the loss, penalising exactly the drift the meter reads.

🚨 **THIS IS NOT R3-SELF, AND THE DIFFERENCE IS THE WHOLE DESIGN.** The rev-3 SELF fold used
self-distillation as the fold TARGET at production dose and measured -9pp: a target DRIVES steps,
and driving a policy toward its own frozen copy at a dose sized for teacher content is destructive.
The anchor is a small-coefficient REGULARISER toward the frozen STARTING policy, on states no
teacher covers, folded BESIDE a real teacher term — it removes freedom the fold did not need, it
does not supply a direction the fold is meant to follow. A coefficient anywhere near the distill
coefficient is misuse; the intended regime is a fraction of it.

**THE KL DIRECTION.** Both the loss and the meter use FORWARD `KL(parent || student)`, matching the
exploiter-distillation term's own direction (`KL(teacher || student)`) so the two terms are read on
one scale. The licensing probe reported the REVERSE, `KL(now || original)`; the two agree in sign
and to first order near a small drift but are not the same number, so a probe figure and a
`distill/collateral_kl` figure are comparable in TREND and not in absolute value.

**THE SLICE.** On-slice = the sample's OUR-team is one of some teacher's pinned teams. That is
exactly the `distill_mask` obs key the exploiter-distillation term already masks on (the INTEGER
teacher-id: 0 = no teacher, k = teacher k), reused rather than re-derived — a second team-identity
path is a second thing to drift. It therefore requires an active distillation
(`--distill-coef > 0`): the env only emits `distill_mask` when `_distill_species` is populated,
which `matchup_setup.apply_distill_team_bias` gates on the coefficient.
"""
import torch as th
from torch.nn import functional as F

#: `--distill-anchor-mode` values. `off_slice` (the default) anchors only where no teacher teaches;
#: `all` anchors everywhere, and exists so a future arm can test whether excluding the taught slice
#: is what makes the trust region work, rather than assuming it.
ANCHOR_MODES = ("off_slice", "all")


def anchor_row_weights(distill_mask, mode, dtype):
    """``(weights, off_indicator)`` — both ``[B]`` float, from the INTEGER-team-id ``distill_mask``.

    ``off_indicator`` is 1.0 exactly where ``distill_mask == 0`` (no teacher pins this state's team)
    and is the meter's denominator regardless of mode. ``weights`` is what the LOSS sums over: the
    same thing under ``off_slice``, all-ones under ``all``.
    """
    tid = distill_mask.reshape(-1).to(dtype)
    off = (tid < 0.5).to(dtype)
    if mode == "all":
        return th.ones_like(off), off
    return off, off


def masked_kl_rows(p_logits, q_logits, action_mask):
    """Per-row forward ``KL(p || q)`` over the LEGAL actions — ``[B]``.

    Illegal logits go to -inf on BOTH sides first, so both distributions normalise over the same
    legal set and illegal actions contribute exactly 0 (identical masking to `_distill_loss`).
    """
    neg = (action_mask.to(q_logits.dtype) - 1.0) * 1e9
    logq = F.log_softmax(q_logits + neg, dim=-1)
    p = F.softmax(p_logits + neg, dim=-1)
    return (p * (th.log(p.clamp_min(1e-9)) - logq)).sum(-1)


def anchor_loss_and_metrics(parent_logits, student_logits, action_mask, distill_mask,
                            *, mode: str = "off_slice"):
    """``(anchor_kl | None, metrics)`` — the trust-region KL and the licensing probe's live meters.

    ``parent_logits`` [B, n_actions] is the FROZEN fold parent's (already under ``no_grad``);
    ``student_logits`` [B, n_actions] carries grad. ``distill_mask`` [B] or [B,1] is the integer
    teacher-id. Returns ``None`` for the KL when the selected slice is empty this minibatch — an
    empty subset must never NaN-poison the loss (the `_distill_loss` None-guard convention).

    The metrics are computed on the SAME masked rows either way, so `collateral_kl` /
    `on_slice_kl` read identically whether or not the loss folds — that is what makes
    `--distill-anchor-coef 0 --distill-anchor-monitor` a real instrument arm rather than a
    different measurement.
    """
    if parent_logits is None or student_logits is None or distill_mask is None:
        return None, {}
    dtype = student_logits.dtype
    w, off = anchor_row_weights(distill_mask, mode, dtype)
    on = 1.0 - off
    row = masked_kl_rows(parent_logits, student_logits, action_mask)   # [B], grad through student
    n_w = w.sum()
    kl = (row * w).sum() / n_w.clamp(min=1e-6) if float(n_w) >= 1.0 else None
    with th.no_grad():
        n_off, n_on = off.sum(), on.sum()
        metrics = {"off_slice_frac": float(off.mean())}
        if float(n_off) >= 1.0:
            metrics["collateral_kl"] = float((row * off).sum() / n_off)
        if float(n_on) >= 1.0:
            metrics["on_slice_kl"] = float((row * on).sum() / n_on)
        metrics["anchor_n"] = float(n_w)
    return kl, metrics


def distill_anchor_step(model, rollout_data, student_pi, metrics_out):
    """THE ONE SEAM `ppo.py` calls. Returns the weighted anchor term (or ``None``) and folds every
    meter into ``metrics_out`` (the `distill/` metric dict, so the names land as `distill/*`).

    ``student_pi`` is the distribution ``evaluate_actions`` already built
    (``policy._last_pi_distribution``) — reused rather than re-forwarded, exactly as the distill
    term reuses it; ``None`` (or a policy without the stash) is tolerated and skips the block.

    Runs ONE frozen ``no_grad`` forward of the parent per minibatch — the `_distill_teachers` shape
    exactly, obs keys filtered to the parent's own space so a parent from an older obs generation
    still loads. Absent parent / absent `distill_mask` => ``None`` and nothing logged, so a run
    without the flag pays nothing and its loss expression is unchanged.

    `anchor_loss` is recorded whenever the meters are, INCLUDING as an exact 0.0 under
    monitor-only: a reader comparing a monitored control arm against a folded arm needs the series
    to exist in both, and a missing series reads as "not measured" rather than "measured zero".
    """
    parent = getattr(model, "_distill_anchor_parent", None)
    if parent is None:
        return None
    dmask = rollout_data.observations.get("distill_mask")
    if dmask is None:
        return None
    # Reuse the stash when there is one; fall back to a fresh forward otherwise. The fallback is
    # not defensive padding — `_last_pi_distribution` is set by `Gen3DualHeadMaskablePolicy` and by
    # nothing else, so a stock SB3 policy (every unit-test toy, and any future policy) takes it. The
    # two are equal over the legal set, which is all the masked KL reads. Same fallback the distill
    # term carries.
    student_logits = getattr(getattr(student_pi, "distribution", None), "logits", None)
    if student_logits is None:
        student_logits = model.policy.get_distribution(
            rollout_data.observations).distribution.logits
    p_obs = {k: v for k, v in rollout_data.observations.items()
             if k in parent.observation_space.spaces}
    with th.no_grad():
        p_logits = parent.policy.get_distribution(p_obs).distribution.logits
    mode = str(getattr(model, "distill_anchor_mode", "off_slice"))
    kl, metrics = anchor_loss_and_metrics(
        p_logits, student_logits, rollout_data.action_masks, dmask, mode=mode)
    coef = float(getattr(model, "distill_anchor_coef", 0.0) or 0.0)
    term = coef * kl if (kl is not None and coef != 0.0) else None
    for _k, _v in metrics.items():
        metrics_out.setdefault(_k, []).append(float(_v))
    if metrics:
        metrics_out.setdefault("anchor_loss", []).append(
            float(term) if term is not None else 0.0)
        if kl is not None:
            metrics_out.setdefault("anchor_kl", []).append(float(kl))
    return term
