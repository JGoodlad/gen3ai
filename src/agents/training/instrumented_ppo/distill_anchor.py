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

**THE REFERENCE — `--distill-anchor-ref {parent,ema,periodic}`, default `parent`.** WHICH policy
the trust region is measured against, and the answer is not the one PPO's clip uses. The clip bounds
the per-update RATE against the policy that collected the data, and it re-reads that policy every
rollout; the anchor bounds the ACCUMULATED DISPLACEMENT from the fold start, which is the quantity
the licensing probe measured and the quantity rev-4's untaught robbery is made of. That collateral is
SYSTEMATIC — the same off-slice direction every step — and a reference that follows the student
barely resists a systematic drift at all, because it moves with it. So the default is FIXED, which is
Learning-without-Forgetting's design (Li & Hoiem 2016: distil against a snapshot of the model taken
before the new task).

The two alternatives exist because a fixed reference cannot tell a GIFT from a ROBBERY — v8's fold
CHANGED off-slice switching behaviour by +5.4pp and that change was GOOD, but to a fixed anchor it is
displacement like any other, so a large enough coefficient suppresses it. A Polyak/EMA-averaged
reference is ACER's trust region (Wang et al. 2016: KL to an average-policy network, alpha ~ 0.99); it
lets slow consistent improvement through — the average follows it — while still taxing fast overshoot,
which the average lags behind. `periodic` is the coarse-grained version of the same idea, and its
`--distill-anchor-refresh-every 0` degenerates to `parent`.

**THE DISPLACEMENT METER IS EMITTED IN EVERY MODE.** `distill/collateral_kl` reads the ANCHOR's own
reference, so under `ema`/`periodic` it is a rate, not a displacement — and the displacement is the
number the untaught-team meter correlates with. `distill/collateral_kl_vs_parent` is therefore always
`KL(frozen PARENT || student)` on the off-slice rows, whatever the anchor is anchored to; the frozen
parent stays loaded in all three modes for exactly this (~2M params — memory is not the constraint).
In `parent` mode the two are the SAME number computed once, so the default arm pays no second forward.
`distill/anchor_ref_age_rollouts` says what the anchor is anchored to: rollouts since the reference was
last refreshed under `parent`/`periodic`, and the nominal EMA window `1/(1-tau)` under `ema`.

**THE SLICE.** On-slice = the sample's OUR-team is one of some teacher's pinned teams. That is
exactly the `distill_mask` obs key the exploiter-distillation term already masks on (the INTEGER
teacher-id: 0 = no teacher, k = teacher k), reused rather than re-derived — a second team-identity
path is a second thing to drift. It therefore requires an active distillation
(`--distill-coef > 0`): the env only emits `distill_mask` when `_distill_species` is populated,
which `matchup_setup.apply_distill_team_bias` gates on the coefficient.
"""
import torch as th
from torch.nn import functional as F

from agents.training.instrumented_ppo.distill_grad_project import GRAD_PROJECT_MODE

#: `--distill-anchor-mode` values. `off_slice` (the default) anchors only where no teacher teaches;
#: `all` anchors everywhere, and exists so a future arm can test whether excluding the taught slice
#: is what makes the trust region work, rather than assuming it. `grad_project`
#: (`gen3_distill_grad_project_v1`, `distill_grad_project.py`) is the SOURCE-SEPARATED mode: it
#: additionally projects the DISTILL gradient off the off-slice behaviour subspace at every step,
#: which is the thing an OUTPUT anchor structurally cannot do — see that module's docstring. For the
#: OUTPUT half it behaves exactly like `off_slice`, so `--distill-anchor-coef 0` is
#: projection-only and a positive coefficient COMPOSES an off-slice output anchor on top (the
#: projection is first-order per step; the output anchor catches the second-order accumulation).
ANCHOR_MODES = ("off_slice", "all", GRAD_PROJECT_MODE)

#: `--distill-anchor-ref` values — WHICH policy the anchor is measured against. `parent` (the
#: default) is the FIXED frozen fold parent and is byte-identical to the behaviour this feature
#: shipped with; `ema` is a Polyak average of the student; `periodic` re-snapshots the student every
#: N rollouts. Both moving forms are INITIALISED FROM THE PARENT, so at fold start all three modes
#: hold the same reference and only diverge as the student moves.
ANCHOR_REFS = ("parent", "ema", "periodic")


def anchor_row_weights(distill_mask, mode, dtype):
    """``(weights, off_indicator)`` — both ``[B]`` float, from the INTEGER-team-id ``distill_mask``.

    ``off_indicator`` is 1.0 exactly where ``distill_mask == 0`` (no teacher pins this state's team)
    and is the meter's denominator regardless of mode. ``weights`` is what the LOSS sums over: the
    same thing under ``off_slice``, all-ones under ``all``.

    ``grad_project`` takes ``off_slice``'s weights deliberately — the mode's own contribution is the
    GRADIENT projection, and any output anchor folded beside it belongs on the same rows the
    projection constrains. Naming the mode here rather than defaulting to it keeps a typo in an
    unrecognised value falling through to ``off_slice`` visible in exactly one place.
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


def frozen_logits(reference, observations):
    """One ``no_grad`` forward of a frozen REFERENCE over ``observations`` -> ``[B, n_actions]``.

    ``reference`` is anything carrying ``.policy`` and ``.observation_space`` — the frozen parent
    MODEL, or the `MovingAnchorReference` holder the ema/periodic modes build. The obs keys are
    filtered to the reference's own space so a parent from an older obs generation still loads (and
    so the training-only ``distill_mask`` never reaches a policy whose extractor was not built with
    it).
    """
    obs = {k: v for k, v in observations.items()
           if k in reference.observation_space.spaces}
    with th.no_grad():
        return reference.policy.get_distribution(obs).distribution.logits


def offslice_kl(ref_logits, student_logits, action_mask, distill_mask):
    """The OFF-SLICE mean ``KL(ref || student)`` as a float, under ``no_grad`` — or ``None`` when
    this minibatch holds no off-slice row.

    This is `anchor_loss_and_metrics`'s `collateral_kl` computed against an ARBITRARY reference, and
    it exists for exactly one caller: `distill/collateral_kl_vs_parent`, the accumulated-displacement
    meter that must keep reading the FROZEN parent even when the anchor's own reference is moving.
    """
    if ref_logits is None or student_logits is None or distill_mask is None:
        return None
    with th.no_grad():
        _, off = anchor_row_weights(distill_mask, "off_slice", student_logits.dtype)
        n_off = off.sum()
        if float(n_off) < 1.0:
            return None
        rows = masked_kl_rows(ref_logits, student_logits, action_mask)
        return float((rows * off).sum() / n_off)


def distill_anchor_step(model, rollout_data, student_pi, metrics_out):
    """THE ONE SEAM `ppo.py` calls. Returns the weighted anchor term (or ``None``) and folds every
    meter into ``metrics_out`` (the `distill/` metric dict, so the names land as `distill/*`).

    ``student_pi`` is the distribution ``evaluate_actions`` already built
    (``policy._last_pi_distribution``) — reused rather than re-forwarded, exactly as the distill
    term reuses it; ``None`` (or a policy without the stash) is tolerated and skips the block.

    Runs ONE frozen ``no_grad`` forward of the REFERENCE per minibatch — the `_distill_teachers`
    shape exactly, obs keys filtered to the reference's own space so a parent from an older obs
    generation still loads — plus a SECOND one of the frozen parent under `ema`/`periodic`, where
    the reference is no longer the parent and `collateral_kl_vs_parent` still has to read it. Under
    the default `parent` reference there is exactly one forward, as there always was. Absent parent
    / absent `distill_mask` => ``None`` and nothing logged, so a run without the flag pays nothing
    and its loss expression is unchanged.

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
    # THE REFERENCE the trust region is measured against: the frozen parent under the default
    # `--distill-anchor-ref parent`, else the moving one `DistillAnchorCallback` maintains. Absent
    # ⇒ the parent, so every pre-`--distill-anchor-ref` caller behaves exactly as it did.
    ref = getattr(model, "_distill_anchor_ref", None) or parent
    p_logits = frozen_logits(ref, rollout_data.observations)
    mode = str(getattr(model, "distill_anchor_mode", "off_slice"))
    kl, metrics = anchor_loss_and_metrics(
        p_logits, student_logits, rollout_data.action_masks, dmask, mode=mode)
    # THE DISPLACEMENT METER, in every mode. Under a MOVING reference `collateral_kl` reads a rate
    # rather than a displacement, and the displacement is the number the untaught-team meter tracks
    # — so `collateral_kl_vs_parent` always reads the FROZEN parent. When the reference IS the
    # parent it is the number already in hand, so the default arm pays no second forward.
    if ref is parent:
        if "collateral_kl" in metrics:
            metrics["collateral_kl_vs_parent"] = metrics["collateral_kl"]
    else:
        _vs_parent = offslice_kl(frozen_logits(parent, rollout_data.observations),
                                 student_logits, rollout_data.action_masks, dmask)
        if _vs_parent is not None:
            metrics["collateral_kl_vs_parent"] = _vs_parent
    # WHAT the anchor is anchored to, as a number a reader can see: rollouts since the reference was
    # last refreshed (`parent` = since fold start, and therefore ever-rising), or the nominal EMA
    # window under `ema`. Absent on a model the callback never touched.
    _age = getattr(model, "distill_anchor_ref_age", None)
    if _age is not None:
        metrics["anchor_ref_age_rollouts"] = float(_age)
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
