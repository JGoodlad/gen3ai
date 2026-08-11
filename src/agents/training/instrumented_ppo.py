"""InstrumentedMaskablePPO — MaskablePPO with `train/clip_fraction_vf` added.

sb3_contrib's MaskablePPO logs `train/clip_fraction` (policy clip fraction) but
not the equivalent metric for value-function clipping, even though VF clipping
is applied when `clip_range_vf` is non-None. This subclass copies the upstream
`train()` method verbatim and adds three lines: a per-batch fraction
computation, accumulation, and one final `self.logger.record(...)` call.

# Drift detection

`train()` is a vendored copy of upstream code. If sb3_contrib ever updates the
method, our copy will silently become stale and we'd run different training
logic than upstream. To prevent that, `_verify_upstream_unchanged()` runs at
import time and computes the SHA256 of `inspect.getsource(MaskablePPO.train)`.
A mismatch raises immediately with both hashes and instructions.

If upstream changes (e.g. after a `pip install -U sb3_contrib`):
1. Diff the new upstream `train()` against the one this subclass was vendored
   from (last known hash is in `_EXPECTED_UPSTREAM_TRAIN_HASH`).
2. Port any non-instrumentation changes into the `train()` override below.
3. Recompute the hash and update `_EXPECTED_UPSTREAM_TRAIN_HASH`.
"""

import hashlib
import inspect
import itertools

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.utils import explained_variance


class _GroupGradAccumulator:
    """Per-param-GROUP gradient accumulation across OPTIMIZER steps (--film-grad-accum-steps).

    ``gate(k)`` is called at each optimizer-step boundary (after clipping, before ``step()``): it
    adds the group's current post-clip grads into a persistent buffer; on the k-th capture it
    writes the AVERAGED accumulated grad back into ``p.grad`` (one clean large-batch update for the
    group); on non-apply steps it sets ``p.grad = None`` — torch optimizers SKIP None-grad params,
    so Adam takes no stale-momentum step and its moment state is untouched between applies.
    Capturing POST-clip keeps the global clip semantics identical to baseline (the group's grads
    participate in the global norm every step) and bounds the average by construction. k<=1 is a
    pure passthrough (never touches grads — byte-identical)."""

    def __init__(self, params):
        self.params = list(params)
        self.buf = None
        self.count = 0

    def gate(self, k: int) -> bool:
        if k <= 1:
            return True
        import torch as th
        if self.buf is None:
            self.buf = [th.zeros_like(p) for p in self.params]
        for b, p in zip(self.buf, self.params):
            if p.grad is not None:
                b.add_(p.grad)
        self.count += 1
        if self.count >= k:
            for b, p in zip(self.buf, self.params):
                b.div_(self.count)
                p.grad = b
            self.buf = None
            self.count = 0
            return True
        for p in self.params:
            p.grad = None
        return False
from torch.nn import functional as F

from sb3_contrib import MaskablePPO

from agents.training.async_vec_env import AsyncSubprocVecEnv, collect_rollouts_async
from agents.training.grad_balance import (
    grad_balance_metrics,
    shared_trunk_parameters,
    value_scale_metrics,
)
from agents.training.rank_metrics import rank_probe


# SHA256 of inspect.getsource(MaskablePPO.train) at the time this file was
# written. If sb3_contrib updates and this no longer matches, _verify_...
# raises at import time.
_EXPECTED_UPSTREAM_TRAIN_HASH = (
    "79500464b6a71d5adcfdf10028df56fbaf72b7754952e760f9e377610b9cf809"
)

# Fraction of each minibatch that forms the "tail" for the tail-weighted value loss — the worst
# _VALUE_TAIL_FRAC by squared value error (the V-tail craters the critic under-prices). 0.1 = worst
# 10%; loosely tracks the eval/td_resid_tail CVaR@5% diagnostic the loss is meant to pull down.
_VALUE_TAIL_FRAC = 0.1

# Win-prob closeness threshold: a decision is "contested" (the band where the head's value lives — a
# blowout's P(win) is trivially recoverable from material) when |normalized material margin| < this.
# margin ∈ [−1,1] = Φ_mat/bound; bound ≈ 19.5, so 0.25 ≈ a material lead of up to ~1.5 mons.
_WIN_CONTESTED_TAU = 0.25

# Latent-belief VICReg variance floor: a hinge `relu(_LATENT_STD_TARGET - std)` per latent dim pushes
# the predicted latents to stay spread (≈unit std), the belt-and-braces collapse guard on top of the
# stop-grad + task-anchored target. Weighted by _LATENT_VICREG_WEIGHT inside the latent loss. The
# `belief_latent_std` metric (mean per-dim std) is the NO-GO monitor: std→0 while cosine→1 is collapse.
_LATENT_STD_TARGET = 1.0
_LATENT_VICREG_WEIGHT = 1.0

# Gradient-noise-scale EMA decay (McCandlish et al. 2018, "An Empirical Model of Large-Batch
# Training"). The single-step estimates of |G|² (true-gradient squared norm) and tr(Σ) (per-example
# gradient-variance trace) are noisy; their RATIO B_simple = tr(Σ)/|G|² is unstable per step, so we
# EMA the numerator and denominator SEPARATELY (this constant) and divide the smoothed values. 0.99
# ≈ a few-hundred-train()-call window — long enough to denoise, short enough to track drift.
_NOISE_SCALE_EMA_DECAY = 0.99

# Spread-belief loss (gen3_unified_spread_belief_v1): the believed/true DERIVED stat VALUES are ~80-450,
# so normalise by this before smooth_l1 to keep the term O(1) (the coef then sets its true weight). The
# MAE metric is reported in RAW stat points (not normalised) for interpretability.
_SPREAD_LOSS_SCALE = 100.0

# Nature/EV-belief loss (gen3_nature_ev_belief_v1): the nature is a 25-way CE; the EV is a smooth_l1 over EV
# VALUES (0..252) normalised by this so the term stays O(1) (the spread coef then sets its true weight). The
# CE + EV sub-terms combine with these internal weights (the CE is the load-bearing decomposition signal; the
# EV-MAE metric is reported in RAW EV points).
_EV_LOSS_SCALE = 64.0
_NATURE_CE_WEIGHT = 1.0
_EV_LOSS_WEIGHT = 1.0


def _verify_upstream_unchanged() -> None:
    """Fail-loud at import time if the upstream `MaskablePPO.train()` source
    has drifted from what this subclass was vendored against."""
    source = inspect.getsource(MaskablePPO.train)
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual != _EXPECTED_UPSTREAM_TRAIN_HASH:
        upstream_file = inspect.getfile(MaskablePPO)
        raise RuntimeError(
            "[InstrumentedMaskablePPO] DRIFT DETECTED: upstream "
            "`sb3_contrib.MaskablePPO.train()` source has changed since this "
            "subclass was vendored.\n"
            f"  Expected SHA256: {_EXPECTED_UPSTREAM_TRAIN_HASH}\n"
            f"  Actual SHA256:   {actual}\n"
            f"  Upstream file:   {upstream_file}\n\n"
            "ACTION REQUIRED: diff the upstream train() against this subclass, "
            "port any non-instrumentation changes into the override in "
            f"{__file__}, then update _EXPECTED_UPSTREAM_TRAIN_HASH to silence "
            "this check."
        )


_verify_upstream_unchanged()


class InstrumentedMaskablePPO(MaskablePPO):
    """MaskablePPO with `train/clip_fraction_vf` instrumentation added.

    Behaviour-identical to `MaskablePPO` except for the additional TensorBoard
    metric. See module docstring for drift-detection details.

    Also dispatches rollout collection to the **non-barrier async collector** when
    ``self._async_rollout`` is set and the env is an ``AsyncSubprocVecEnv`` (``--async-rollout``);
    otherwise it is the unchanged stock ``MaskablePPO.collect_rollouts``.
    """

    # Set by train_rl_agent after construction; opt-in (default off → stock sync collection).
    _async_rollout: bool = False

    # Set by train_rl_agent after construction (like _async_rollout). GRADIENT ACCUMULATION: do K
    # forward/backward passes over `batch_size`-sized MICRO-batches, summing their gradients, and
    # call optimizer.step() only ONCE per group of K. The accumulated gradient is the EXACT gradient
    # of one (batch_size·K) batch (gradients are additive + each micro-loss is scaled by 1/K), but the
    # backward graph only ever holds ONE micro-batch's activations → an effective batch of batch_size·K
    # at the GPU-memory cost of batch_size. The memory lever stock MaskablePPO can't give: it steps once
    # per minibatch, so `batch_size` alone couples the effective-batch size to the activation peak. 1 =
    # OFF (one step per minibatch, byte-identical to upstream — `loss / 1` is exact). A pure train-loop
    # knob (no forward change) → NOT version-locked / NOT in model_config.json (forwarded as a CLI flag
    # each resume, like batch_size / n_epochs).
    #   EXACTNESS: the accumulation math is BIT-EXACT to a literal batch_size·K batch when batch_size
    #   divides the rollout (n_steps·n_envs) AND accum divides the minibatch count — every group is then
    #   `accum` equal-size micro-batches (verified to ~3e-8 in instrumented_ppo_test). Two bounded,
    #   negligible deviations otherwise: (1) per-MICRO-batch advantage normalization — stock normalizes
    #   per-minibatch too, so the only change is the normalization sample size (batch_size vs batch_size·K),
    #   immaterial for batches of thousands; (2) a NON-divisible rollout slightly mis-weights the single
    #   remainder minibatch in the final group of each epoch (no worse than stock's full-weight step on it).
    #   For a bit-exact effective batch, pick batch_size | rollout and accum | minibatch-count.
    grad_accum_steps: int = 1

    # +NOISE-SCALE: running (EMA) estimates of the McCandlish gradient-noise-scale numerator/denominator
    # — tr(Σ) and |G|² — accumulated across train() calls (one sample per call). None until the first
    # measurable call. Only updated when grad_accum_steps >= 2 (the diagnostic needs gradients at TWO
    # batch sizes: one micro-batch = batch_size, and the accumulated group = batch_size·accum). Process-
    # local (reset to None on a launcher restart → re-converges in a few hundred calls); not saved.
    _noise_ema_s: float = None    # EMA of tr(Σ)  (per-example gradient-variance trace)
    _noise_ema_g2: float = None   # EMA of |G|²   (true-gradient squared norm)
    _noise_film_ema_s: float = None   # film-group EMA of tr(Σ)  (+FILM-NOISE-SCALE)
    _noise_film_ema_g2: float = None  # film-group EMA of |G|²

    @staticmethod
    def _global_grad_sq(params) -> float:
        """Squared global L2 norm ‖g‖² of the CURRENT .grad over all params (one device→host sync).
        Mirrors what clip_grad_norm_ computes, but read-only (no clipping)."""
        sq = None
        for p in params:
            g = p.grad
            if g is not None:
                s = g.detach().pow(2).sum()
                sq = s if sq is None else sq + s
        return float(sq) if sq is not None else 0.0

    @staticmethod
    def _noise_scale_estimate(g_small_sq, g_big_sq, b_small, b_big):
        """McCandlish et al. 2018 'simple' gradient-noise-scale building blocks from squared gradient
        norms at TWO batch sizes. Since E‖Ĝ_B‖² = ‖G‖² + tr(Σ)/B, two (B, ‖Ĝ_B‖²) points pin both
        unknowns:
            |G|²   = (b_big·g_big_sq − b_small·g_small_sq) / (b_big − b_small)        # true-grad norm²
            tr(Σ)  = (g_small_sq − g_big_sq) / (1/b_small − 1/b_big)                   # per-example noise
        Returns ``(tr_sigma, g2)`` (single-sample, pre-EMA; either can be negative under noise — the
        caller EMAs them separately before the B_simple = tr(Σ)/|G|² ratio). Pure → unit-testable."""
        g2 = (b_big * g_big_sq - b_small * g_small_sq) / (b_big - b_small)
        tr_sigma = (g_small_sq - g_big_sq) / (1.0 / b_small - 1.0 / b_big)
        return tr_sigma, g2

    # Set by train_rl_agent after construction (like _async_rollout); resume-immutable (recorded +
    # version-checked). 0.0 = plain MSE value loss (byte-identical to upstream). >0 blends in the CVaR
    # of the worst value misses — see _value_loss_from_se.
    value_tail_weight: float = 0.0

    # Set by train_rl_agent after construction (like value_tail_weight). The hidden-opponent belief
    # aux-loss coefficient: opp_belief_aux_coef * (species_CE + moves_weight·moves_BCE) over the
    # believed opp slots is added to each minibatch loss. 0.0 = OFF (no aux term, byte-identical loss).
    # A TRAINING hparam (affects the loss only, never a forward pass) → NOT version-locked / NOT in
    # check_compatible (treat like ent_coef; a frozen eval/pool/distill opponent never runs train()).
    # Class default 0.0 so a smoke/test/frozen-opponent path that never sets it reads a safe no-op.
    opp_belief_aux_coef: float = 0.0
    # Relative weight of the moves multi-label BCE vs the species CE inside the aux term. Both are now
    # on a per-believed-slot scale (CE per slot ≈ log S; BCE = mean over the M move classes per slot),
    # so the species term dominates by default (moves is the weaker secondary signal); raise this to
    # up-weight move prediction. Training-only, like the coef.
    opp_belief_moves_weight: float = 1.0

    # Set by train_rl_agent (like opp_belief_aux_coef). The MOVE-belief reinjection-head loss weight:
    # move_belief_coef * (BCE over the scored opp slots) is added to each minibatch. 0.0 = OFF
    # (byte-identical). Training-only (scales the loss, never a forward pass) → NOT version-locked. The
    # MODE (which slots) lives on the extractor (move_belief_mode) — read from there, single source.
    move_belief_coef: float = 0.0

    # Set by train_rl_agent (gen3_unified_spread_belief_v1). The SPREAD-belief supervision weight:
    # spread_belief_coef * smooth_l1(believed derived stats, TRUE derived stats) over the REVEALED opp
    # slots. 0.0 = OFF (byte-identical loss — the SpreadBelief head then gets only the indirect op-damage
    # gradient and sits at the usage-mean prior, the "over-estimates the largest EV" miscalibration). It
    # READS the extractor's last_spread_belief (the believed stats the op consumes) + the training-only
    # belief_spread/_mask label keys. Training-only (scales the loss, never a forward pass) → NOT
    # version-locked; the SpreadBelief module is gated by the version-checked spread_belief arch toggle.
    spread_belief_coef: float = 0.0

    # Set by train_rl_agent (gen3_opp_hp_type_belief_v1). The HP-TYPE-belief supervision weight:
    # hp_type_belief_coef * cross_entropy(HPTypeBelief posterior logits, TRUE HP type) over the REVEALED
    # opp slots whose species runs Hidden Power. 0.0 = OFF (byte-identical loss — the head then gets only
    # the indirect op-damage gradient + sits at the Smogon HP-type prior). READS the extractor's
    # last_hp_type_logits + the training-only hp_type_label/hp_type_mask keys. Training-only (NOT
    # version-locked; the HPTypeBelief module is gated by the version-checked hp_type_belief_mode toggle).
    hp_type_belief_coef: float = 0.0

    # Set by train_rl_agent (gen3_unified_move_system_v1). The MOVE-belief LATENT-grading weight:
    # move_belief_latent_coef * (cosine of the predicted move distribution's expected move-latent toward
    # the true moveset's mean latent + VICReg floor) over the revealed scored slots. 0.0 = OFF
    # (byte-identical loss). Soft complement to the per-ID BCE so near-moves (Rock Slide ≈ HP Rock) grade
    # as near. Training-only (scales the loss, never a forward pass) → NOT version-locked; it READS the
    # extractor's last_move_latent_table, whose state_dict-changing module is gated by the version-checked
    # move_latent arch toggle.
    move_belief_latent_coef: float = 0.0

    # Set by train_rl_agent (like opp_belief_aux_coef). The LATENT-belief loss weight: opp_belief_latent_coef
    # * (cosine-to-encoder-role-token + VICReg variance floor) over the believed opp slots, matched on the
    # SAME Hungarian assignment as the species CE. 0.0 = OFF (no term; byte-identical loss). Training-only
    # (scales the loss, never a forward pass) → NOT version-locked. The latent PREDICTOR (a state_dict
    # change) is gated by the version-checked opp_belief_latent arch toggle, not this coef.
    opp_belief_latent_coef: float = 0.0

    # Set by train_rl_agent (gen3_defensive_entropy_v1). STATE-CONDITIONED entropy boost: on decisions the env
    # flags `defensive_opportunity`=1 (a productive recovery/cure move is legal), the per-decision entropy bonus
    # is multiplied by `defensive_entropy_boost` so the policy keeps EXPLORING defensive moves instead of
    # collapsing to attacking — WITHOUT touching the reward (no stall incentive; the draw penalty + clock stay
    # the guardrail). 1.0 = OFF (byte-identical entropy term). Annealed B→1 over `defensive_entropy_anneal_frac`
    # of training (0 = constant). Training-only (like ent_coef): NOT version-locked, settable on resume.
    defensive_entropy_boost: float = 1.0
    defensive_entropy_anneal_frac: float = 0.0

    # Set by train_rl_agent (like opp_belief_aux_coef). The WIN-PROBABILITY head's BCE loss weight:
    # win_prob_coef * BCE(win_logit, MC outcome) over the transitions whose episode finished in-buffer.
    # Training-only (scales the loss, never a forward pass) → NOT version-locked. The MODE (none /
    # read_only / shaping — which also controls whether the grad reaches the trunk) lives on the
    # extractor (win_prob_mode); the loss is added whenever the mode is on AND this coef != 0.
    win_prob_coef: float = 1.0

    # Set by train_rl_agent (gen3_pubval_aux_v1, like win_prob_coef). The PUBLIC-VALUE aux head's
    # soft-target BCE weight: pubval_coef * BCE(pubval_logit, V_pub) where V_pub is the FROZEN
    # human-replay-calibrated public value riding the training-only `pubval_target` obs key.
    # Training-only → NOT version-locked; the MODE (none/read_only/shaping) lives on the extractor
    # (pubval_mode); the loss is added whenever the mode is on AND this coef != 0.
    pubval_coef: float = 0.0

    # Set by train_rl_agent (gen3_zarch_film_v1, like spread_belief_coef). The z_arch aux weights:
    # zarch_recon_coef * BCE(recon_logits, our-team species multi-hot) — the ANTI-COLLAPSE anchor (a
    # constant z cannot reconstruct different teams; Species Clause ⇒ multi-hot is lossless) — plus
    # zarch_vicreg_coef * relu(1 − std(z, batch)).mean() — the VICReg per-dim variance floor. Both READ
    # the extractor's last_zarch/last_zarch_recon_logits/last_zarch_species_ids stashes (grad-gated,
    # training epochs only). The gradients touch ONLY the ZArchEncoder's own params (the encoder reads
    # DETACHED embedding tables + raw obs slices — zero shared-trunk pull, so no grad-balance entry).
    # 0.0/0.0 = OFF (byte-identical loss). Training-only → NOT version-locked; the ZArchEncoder + FiLM
    # modules are gated by the version-checked zarch_film/zarch_dim arch toggles. NOTE: auto-zeroed by
    # train_rl_agent on a single-team (pinned --trainee-team) run — z is a constant there, so the
    # cross-batch variance floor is degenerate and the recon target is constant (harmless but useless).
    zarch_recon_coef: float = 0.0
    zarch_vicreg_coef: float = 0.0

    # Set by train_rl_agent (like win_prob_coef). The DISTRIBUTIONAL value head's HL-Gauss cross-entropy
    # loss weight: value_dist_coef * CE(value_dist_logits, return) over the rollout. Training-only (scales
    # the loss, never a forward pass) → NOT version-locked (recorded for provenance + flagless-resume
    # read-back). The MODE (none/read_only/shaping — which controls whether the grad reaches the trunk)
    # lives on the extractor (value_dist_mode); the loss is added whenever the mode is on AND coef != 0.
    value_dist_coef: float = 1.0

    # Set by train_rl_agent (like win_prob_coef). The SEARCH-TEACHER AWR policy-distillation weight:
    # search_teacher_coef * advantage-weighted CE toward the verified-better action A*, over a minibatch
    # sampled from the standalone `_correction_buffer` (NOT the rollout buffer — searched states are
    # off-policy). 0.0 = OFF (loss byte-identical even if the buffer fills). Training-only (scales the
    # loss, no forward/weight change) → NOT version-locked. `_correction_buffer` / `_search_teacher_on`
    # are runtime attrs set externally (like `_async_rollout`). Design: design_search_teacher.md.
    search_teacher_coef: float = 0.0
    # AWR temperature β: weight w = clamp(exp(advantage/β), max=w_clip). Higher β → flatter weighting.
    search_teacher_beta: float = 1.0
    # Corrections sampled per train() for the AWR forward (its OWN forward — small, e.g. 256).
    search_teacher_batch_size: int = 256
    # OFF-POLICY value term (DEFAULT 0 — soundness): the search value is V^π*(s), so regressing the
    # current critic (which feeds GAE) toward it biases advantages. Only enable for the joint-ExIt A/B.
    search_teacher_value_coef: float = 0.0

    # ON-POLICY SELF-DISTILLATION (OPD), modelled EXACTLY on search_teacher_coef. Upgrades the
    # distillation TARGET from the single verified-better action A* (AWR) to the FULL improved
    # distribution π' via KL(π' ‖ π_student) — π' is the softmax of the beam's per-action backed-up
    # values (built in produce.py when the workers run OPD). Samples the SAME standalone
    # `_correction_buffer` as the search-teacher (its own get_distribution forward). 0.0 = OFF (loss
    # byte-identical even if the buffer fills). Training-only (scales the loss, no forward/weight change)
    # → NOT version-locked / NOT in ModelVersion / check_compatible. `_opd_on` is a runtime attr set
    # externally (like `_search_teacher_on`). OPD requires --search-teacher (it fills the buffer + its
    # workers build π'); a run can A/B AWR vs KL since a Correction carries BOTH.
    opd_coef: float = 0.0
    # OPD softmax temperature β for π' (built worker-side in produce.py); recorded here for provenance.
    opd_beta: float = 1.0

    # SEED VICReg (gen3_seed_vicreg_v1, v62): the variance+COVARIANCE floor on the multi-seed critic
    # readout's outputs (agents/model/seed_vicreg.py) — the pre-registered collapse trigger in
    # seed_diagnostics.py FIRED on gen-5 (seeds/out_effective_rank 1.0 sustained: the k=4 seeds pay
    # for 4 reads and deliver 1). 0.0 = OFF (loss byte-identical, stash never read). UNLIKE the
    # training-only coefs above this one is resume-IMMUTABLE (the vf_coef class — recorded in
    # ModelVersion, enforced by check_value_seed_vicreg on the training-resume path only).
    value_seed_vicreg_coef: float = 0.0

    # PER-SEED QUANTILE (gen3_seed_quantile_v1, v63): seed k of the MultiSeedValueReadout predicts
    # quantile tau_k of the return through ONE SHARED Linear, so k different predictions REQUIRE k
    # different seed READS — the positive counterpart to the VICReg repulsion above (which gen-6
    # measured as 1-D spread with three seeds still identical). Regressed against the SAME rollout
    # return the critic sees, in the critic's own (PopArt-normalized) frame. 0.0 = OFF, and the HEAD
    # is a structural version-checked toggle, so an off run builds no module at all.
    seed_quantile_coef: float = 0.0

    # EXPLOITER DISTILLATION (gen3_exploiter_distill_v1). The ON-POLICY KL that pours a frozen per-team
    # SPECIALIST (an --exploiter checkpoint) into the generalist: for rollout states where the trainee
    # pilots the teacher's team (the training-only `distill_mask` obs key = 1), a forward of the frozen
    # teacher gives π_teacher, and distill_coef * KL(π_teacher ‖ π_student) is added — carving the
    # specialist's per-team play into the shared trunk WITHOUT touching the value head (policy-only). The
    # non-teacher-team states (distill_mask = 0) are the rehearsal that guards against forgetting. Training-
    # only (scales the loss, no forward/weight change) → NOT version-locked / NOT in check_compatible.
    # `_distill_teacher` is a runtime attr (a loaded frozen model) set externally (like `_correction_buffer`).
    # 0.0 = OFF (byte-identical: the whole block is guarded on coef != 0 AND teacher present).
    distill_coef: float = 0.0
    # VALUE distillation (gen3_exploiter_value_distill_v1) — the missing head: policy-only distill leaves
    # the student piloting the teacher's team with its OWN amortized (~4-dim) critic, so it mimics the
    # teacher's MOVES but never gets its per-team VALUE understanding (confirmed: value_cls rank flat
    # _14→_18→_19). This adds distill_value_coef * MSE(V_teacher, V_student) on the SAME teacher-team
    # states, in the student's PopArt-normalized frame (same frame as the value loss). It is COHERENT
    # despite V^π being policy-relative because the policy KL simultaneously drives π_student→π_teacher
    # there, so V_teacher becomes the right value. Requires distill_coef > 0 (the policy-match validates
    # the value-match). 0.0 = OFF (byte-identical); training-only, NOT version-locked. The A/B lever:
    # policy-only (=0) vs policy+value (>0), read out by the value_cls rank probe.
    distill_value_coef: float = 0.0
    # FITNETS VALUE-FEATURE distillation (gen3_exploiter_value_feat_distill_v1) — the "hint" upgrade of scalar
    # value distill. Matching only the teacher's SCALAR V crystallizes the critic (value_cls rank DROPS, A/B
    # confirmed on ai_v7_20: value_mse falls but rank 4.15→3.55). This adds distill_value_feat_coef ·
    # (1 − cos(value_pooled_student, value_pooled_teacher)) on the SAME teacher-team states — regressing the
    # INTERMEDIATE 128-dim value-CLS pool (the FitNets hint layer) instead of the collapsed scalar, so the trunk
    # inherits the teacher's per-team value STRUCTURE. COSINE (scale-free) chosen from the geometry analysis
    # (complementary, non-competing, low-rank teacher subspaces — see _value_feat_distill). Requires
    # distill_coef > 0 (the policy KL makes V_teacher the right target). 0.0 = OFF (byte-identical, no teacher
    # value_pooled read); training-only, NOT version-locked. Composes with / is an A/B alternative to
    # distill_value_coef (scalar) — read out by the value_cls effective-rank probe (does the HINT enrich it?).
    distill_value_feat_coef: float = 0.0

    def _excluded_save_params(self):
        # The search-teacher's `_correction_buffer` lives on the model (the callback↔train() hand-off),
        # but it is TRANSIENT scaffolding like the rollout buffer — and it holds a threading.Lock that
        # cloudpickle can't serialize (model.save would crash). Exclude it from the checkpoint (also keeps
        # checkpoints small — a full buffer is hundreds of MB of obs); it's re-created on resume, empty,
        # and the workers/cycle refill it. Mirrors SB3 excluding `rollout_buffer`.
        # `_distill_teacher` is a full frozen model (a foreign exploiter) attached at setup — never pickle
        # it into our checkpoint; it is re-loaded from its own path on resume (like a stable opponent).
        # `_film_grad_accumulator` holds CUDA gradient clones: pickled into the zip's data section they
        # deserialize WITHOUT map_location, so any process that loads the snapshot (env/eval workers,
        # device="cpu") silently initializes a 252 MiB GPU context — dozens of workers ≈ the whole card
        # (the 2026-07-20 OOM cascade). Transient like the rollout buffer; train() lazily recreates it.
        return super()._excluded_save_params() + ["_correction_buffer", "_distill_teacher",
                                                  "_distill_teachers", "_film_grad_accumulator"]

    @staticmethod
    def _searchteacher_loss(logits, action_mask, better_action, advantage,
                            beta_awr: float = 1.0, w_clip: float = 20.0):
        """ADVANTAGE-WEIGHTED CE (AWR) toward the verified-better action A* (the search-teacher signal).

        ``logits`` [B, n_actions] = ``policy.get_distribution(obs_dict).distribution.logits`` on the
        CORRECTION obs (its own forward); ``better_action`` [B] = A*; ``advantage`` [B] = the CONFIRMED
        win-rate improvement of A* vs the EXACT opponent (> 0, already CI-gated) — NOT a critic
        advantage. Weight ``w = clamp(exp(adv/β), max=w_clip)`` up-weights high-margin corrections. The
        policy CE is in logit space (no PopArt). Returns ``(loss, metrics)`` or ``None`` on empty.
        """
        if logits is None or better_action is None or better_action.numel() == 0:
            return None
        masked = logits + (action_mask.to(logits.dtype) - 1.0) * 1e9   # illegal → −inf (A* is always legal)
        w = th.exp(advantage / beta_awr).clamp(max=w_clip)
        ce = F.cross_entropy(masked, better_action.long(), reduction="none")   # [B]
        loss = (w * ce).sum() / w.sum().clamp(min=1e-6)
        with th.no_grad():
            agree = (masked.argmax(-1) == better_action).float().mean()
            metrics = {"loss": float(loss), "agree_rate": float(agree),
                       "mean_adv": float(advantage.mean()), "mean_w": float(w.mean()),
                       "ce": float(ce.mean()), "n": int(better_action.numel())}
        return loss, metrics

    @staticmethod
    def _opd_loss(logits, action_mask, pi_target):
        """ON-POLICY SELF-DISTILLATION (OPD) — KL(π' ‖ π_student) toward the FULL improved distribution
        π' (the beam's per-action backed-up values, softmaxed over legal actions). The KL-form upgrade of
        the AWR :meth:`_searchteacher_loss` (which distils only the single action A*).

        ``logits`` [B, n_actions] = ``policy.get_distribution(obs_dict).distribution.logits`` on the
        CORRECTION obs (its own forward); ``action_mask`` [B, n_actions] = the legal mask; ``pi_target``
        [B, n_actions] = π' (already over LEGAL actions, 0 on illegal, L1-normed). Illegal logits are
        masked to −∞ so the student log-probs are over the legal set (matching π'). Forward KL
        ``Σ p_tgt·(log p_tgt − log p_student)``, mean over the batch. Returns ``(kl, metrics)`` or
        ``None`` on an empty / absent π' (the buffer had no OPD targets — an AWR-only run). Pure + static
        so it unit-tests without a full PPO."""
        if logits is None or pi_target is None or pi_target.numel() == 0:
            return None
        masked = logits + (action_mask.to(logits.dtype) - 1.0) * 1e9   # illegal → −inf (π' is 0 there)
        logp = F.log_softmax(masked, dim=-1)                           # student log-probs over legal
        p_tgt = pi_target                                              # already legal-only, illegal 0
        kl = (p_tgt * (th.log(p_tgt.clamp_min(1e-9)) - logp)).sum(-1).mean()
        with th.no_grad():
            ent = -(p_tgt * th.log(p_tgt.clamp_min(1e-9))).sum(-1).mean()   # π' sharpness (low = decisive)
            agree = (masked.argmax(-1) == p_tgt.argmax(-1)).float().mean()  # student ↔ π' mode agreement
            metrics = {"kl": float(kl), "pi_target_entropy": float(ent),
                       "agree_rate": float(agree), "n": int(pi_target.shape[0])}
        return kl, metrics

    @staticmethod
    def _distill_loss(student_logits, teacher_logits, action_mask, distill_mask):
        """EXPLOITER DISTILLATION — masked ON-POLICY KL(π_teacher ‖ π_student) over the rollout minibatch.

        ``student_logits`` / ``teacher_logits`` [B, n_actions] = raw ``get_distribution(...).distribution.
        logits`` (the teacher's ALREADY under no_grad — it is frozen). ``action_mask`` [B, n_actions] = the
        legal mask; ``distill_mask`` [B] or [B,1] = 1 on states where the trainee pilots the TEACHER's team
        (the only states where the teacher's advice is on-distribution — elsewhere it would corrupt the
        other teams, so those rows are excluded). Illegal logits → −∞ so both sides normalise over the legal
        set; forward KL ``Σ p_teacher·(log p_teacher − log p_student)`` per row, masked-mean over the
        teacher-team rows. Returns ``(kl, metrics)`` or ``None`` when the minibatch has no teacher-team rows
        (the None guard keeps an empty subset from NaN-poisoning the loss). Pure + static → unit-testable."""
        if student_logits is None or teacher_logits is None or distill_mask is None:
            return None
        m = distill_mask.reshape(-1).to(student_logits.dtype)              # [B] 1.0 on teacher-team states
        n_on = m.sum()
        if float(n_on) < 1.0:
            return None                                                   # no teacher-team states this batch
        neg = (action_mask.to(student_logits.dtype) - 1.0) * 1e9          # illegal → −inf (both sides)
        logp_s = F.log_softmax(student_logits + neg, dim=-1)              # student log-probs over legal
        p_t = F.softmax(teacher_logits + neg, dim=-1)                     # teacher probs over legal (detached)
        kl_row = (p_t * (th.log(p_t.clamp_min(1e-9)) - logp_s)).sum(-1)   # [B] forward KL per state
        loss = (kl_row * m).sum() / n_on.clamp(min=1e-6)                  # masked-mean over teacher-team rows
        with th.no_grad():
            agree_row = ((student_logits + neg).argmax(-1) == (teacher_logits + neg).argmax(-1)).float()
            metrics = {"kl": float(loss),
                       "agree_rate": float((agree_row * m).sum() / n_on.clamp(min=1e-6)),
                       "coverage": float(m.mean()),                       # fraction of minibatch on teacher team
                       "n": int(n_on.item())}
        return loss, metrics

    @staticmethod
    def _value_distill_mse(student_values, teacher_values, distill_mask, popart=None):
        """VALUE DISTILLATION — masked MSE(V_teacher ‖ V_student) over the teacher-team rows.

        ``student_values`` [B] carries grad; ``teacher_values`` [B] is the frozen teacher's (real-unit,
        already under no_grad). ``distill_mask`` [B]/[B,1] = 1 on teacher-team states. When a PopArt
        normalizer is given, both are mapped to the student's normalized frame first (so the coef is
        scale-comparable with the value loss); else a raw-unit MSE. Returns the masked-mean SE, or None
        when no teacher-team rows (the None guard, like _distill_loss). Pure + static → unit-testable."""
        if student_values is None or teacher_values is None or distill_mask is None:
            return None
        m = distill_mask.reshape(-1).to(student_values.dtype)
        n_on = m.sum()
        if float(n_on) < 1.0:
            return None
        sv, tv = student_values.reshape(-1), teacher_values.reshape(-1)
        if popart is not None:
            se = (popart.normalize(sv) - popart.normalize(tv)) ** 2
        else:
            se = (sv - tv) ** 2
        return (se * m).sum() / n_on.clamp(min=1e-6)

    @staticmethod
    def _value_feat_distill(student_feat, teacher_feat, distill_mask):
        """FITNETS VALUE-FEATURE DISTILLATION — masked COSINE distance between the value-CLS pools.

        The FitNets (Romero 2015) "hint" upgrade of scalar value distillation: instead of only matching the
        teacher's scalar V (which collapses to a ~4-dim critic — `_value_distill_mse` CRYSTALLIZES the head,
        value_cls rank DROPS), regress the student's INTERMEDIATE 128-dim `value_pooled` (the extractor's
        `last_value_pooled` HINT layer) toward the teacher's on the teacher-team states, so the trunk inherits
        the teacher's per-team value STRUCTURE, not just its output.

        ``student_feat`` [B,D] carries grad (the student's live `value_pooled`); ``teacher_feat`` [B,D] is the
        frozen teacher's (already under no_grad). ``distill_mask`` [B]/[B,1] = 1 on teacher-team states. COSINE
        (not MSE): the geometry analysis (`tmp/fitnet_analysis.py`) found the teachers' value subspaces are
        low-rank, COMPLEMENTARY (TSS orthogonal, collective effR ~12), and NON-competing (all pull-cosines
        positive) — so a scale-free directional pull transfers the correct structure without over-constraining
        a low-rank target the way an MSE on raw magnitudes would (the student/teacher live in separately-rotated
        128-dim bases, so absolute coordinates aren't comparable; direction is). Loss = ``1 − cos`` per row,
        masked-mean over the teacher-team rows. Returns the masked-mean cosine distance, or None when no
        teacher-team rows (the None guard, like `_value_distill_mse`). Pure + static → unit-testable."""
        if student_feat is None or teacher_feat is None or distill_mask is None:
            return None
        m = distill_mask.reshape(-1).to(student_feat.dtype)
        n_on = m.sum()
        if float(n_on) < 1.0:
            return None
        cos = F.cosine_similarity(student_feat, teacher_feat, dim=-1, eps=1e-6)   # [B] direction match per row
        dist = 1.0 - cos                                                          # [B] cosine DISTANCE (0=aligned)
        return (dist * m).sum() / n_on.clamp(min=1e-6)

    @staticmethod
    def _win_prob_loss(logits, target, mask, margin=None):
        """Supervised BCE loss for the auxiliary WIN-PROBABILITY head (``last_win_prob_logits`` [B,1]).

        ``target`` [B,1] = the Monte-Carlo episode OUTCOME (win=1 / loss=0) propagated to every step of
        the episode by the ``WinProbLabelCallback`` (it overwrites the obs-dict placeholder post-collection);
        ``mask`` [B,1] = 1 where that label is KNOWN (the step's episode finished within the rollout buffer)
        and 0 for the trailing in-progress episode (no outcome yet) — those transitions are excluded so the
        head is never trained toward a fabricated label. BCE-with-logits, masked-mean. Returns
        ``(loss, metrics)`` or ``None`` when nothing is scorable (head off / labels absent / a minibatch
        with zero known labels — the None guard keeps an empty minibatch from NaN-poisoning the loss). Pure
        + static so it unit-tests without a full PPO.

        When ``margin`` [B,1] (the normalized material margin ∈ [−1,1], from gen3_env's ``win_margin`` obs
        key) is given, ALSO reports the INFORMATION VALUE the aggregate Brier hides: the head's skill on
        CLOSE games (``|margin| < _WIN_CONTESTED_TAU`` — a blowout's P(win) is trivially recoverable from
        material), and a Brier SKILL SCORE vs a material-only baseline (``P_mat = clip(0.5 + 0.5·margin)``):
        ``skill_vs_material`` > 0 ⇒ the head beats 'just count the mons'."""
        if logits is None or target is None or mask is None:
            return None
        logits = logits.reshape(-1)
        target = target.to(logits.device).reshape(-1)
        mask = mask.to(logits.device).reshape(-1)
        n_known = mask.sum()
        if float(n_known) == 0.0:
            return None
        per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (per * mask).sum() / n_known
        with th.no_grad():
            p = th.sigmoid(logits)
            sq = (p - target) ** 2
            correct = ((p > 0.5).float() == target).float()
            brier = (sq * mask).sum() / n_known                             # calibration (lower better)
            acc = (correct * mask).sum() / n_known
            pred_mean = (p * mask).sum() / n_known                          # mean predicted P(win)
            label_mean = (target * mask).sum() / n_known                    # actual win base rate
        metrics = {
            "loss": float(loss.item()),
            "acc": float(acc.item()),
            "brier": float(brier.item()),
            "pred_mean": float(pred_mean.item()),
            "label_mean": float(label_mean.item()),
            "coverage": float((n_known / mask.numel()).item()),             # fraction of minibatch labeled
        }
        # Information value the aggregate Brier hides (only when the material margin is available): the
        # head's skill on CLOSE games + a skill score beyond a material-only baseline.
        if margin is not None:
            with th.no_grad():
                margin = margin.to(logits.device).reshape(-1)
                close = (margin.abs() < _WIN_CONTESTED_TAU).float() * mask
                n_close = close.sum()
                metrics["contested_frac"] = float((n_close / n_known).item())
                if float(n_close) > 0.0:
                    # Brier/acc restricted to material-EVEN decisions — where a good P(win) is non-trivial
                    # (the aggregate is inflated by blowouts). Judge brier_contested vs a 50/50 game's
                    # ~0.25 no-skill floor; contested_label_mean ≈ 0.5 confirms these are genuinely even.
                    metrics["brier_contested"] = float((sq * close).sum() / n_close)
                    metrics["acc_contested"] = float((correct * close).sum() / n_close)
                    metrics["contested_label_mean"] = float((target * close).sum() / n_close)
                # Brier SKILL SCORE vs a material-only baseline P_mat = clip(0.5 + 0.5·margin) — the trivial
                # "predict win from the material lead" forecaster. >0 ⇒ the head adds info BEYOND material;
                # ≤0 ⇒ it's no better than counting mons. The headline "information value" number.
                p_mat = (0.5 + 0.5 * margin).clamp(1e-6, 1.0 - 1e-6)
                brier_mat = (((p_mat - target) ** 2) * mask).sum() / n_known
                metrics["brier_material"] = float(brier_mat.item())
                metrics["skill_vs_material"] = (
                    float((1.0 - brier / brier_mat).item()) if float(brier_mat) > 0.0 else 0.0)
        return loss, metrics

    @staticmethod
    def _pubval_loss(logits, target, mask):
        """Soft-target BCE for the PUBLIC-VALUE aux head (gen3_pubval_aux_v1; ``last_pubval_logits``
        [B,1]). ``target`` [B,1] = the frozen human-replay-calibrated V_pub ∈ (0,1) evaluated on the
        decision's PUBLIC board (a REAL per-step value the env computed — no callback back-fill);
        ``mask`` [B,1] = 1 where the target was computable (0 only pre-battle). BCE-with-logits
        against a SOFT probability target — its minimizer is exactly sigmoid(logit) = V_pub, i.e. the
        head distills the human public value into the trunk's value_pooled read. Returns
        ``(loss, metrics)`` or ``None`` when nothing is scorable. Pure + static → unit-testable.
        NOTE the loss floor is the target's own entropy (V_pub ∈ (0,1) ⇒ BCE > 0 even at a perfect
        fit) — watch ``mae`` (|sigmoid − target|, → 0 as it fits) rather than the raw BCE level."""
        if logits is None or target is None or mask is None:
            return None
        logits = logits.reshape(-1)
        target = target.to(logits.device).reshape(-1)
        mask = mask.to(logits.device).reshape(-1)
        n_known = mask.sum()
        if float(n_known) == 0.0:
            return None
        per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (per * mask).sum() / n_known
        with th.no_grad():
            p = th.sigmoid(logits)
            mae = ((p - target).abs() * mask).sum() / n_known
            pred_mean = (p * mask).sum() / n_known
            target_mean = (target * mask).sum() / n_known
        metrics = {
            "loss": float(loss.item()),
            "bce": float(loss.item()),
            "mae": float(mae.item()),                 # the fit signal: |head − V_pub| → 0
            "pred_mean": float(pred_mean.item()),
            "target_mean": float(target_mean.item()),
            "coverage": float((n_known / mask.numel()).item()),
        }
        return loss, metrics

    # +NSR-ADVISOR state: rate-limited Events-panel warnings when a noise-scale ratio is out of
    # band (see _noise_scale_advice). Process-local (resets each child — fine, it re-warns).
    _nsr_warn_last: dict = None
    _nsr_samples: int = 0

    # Set by train_rl_agent (--film-grad-accum-steps; 1 = off, byte-identical). Per-GROUP gradient
    # accumulation for the FiLM generators: their grads are accumulated across N consecutive
    # OPTIMIZER steps and applied once, averaged — so each film update is computed from N× the
    # effective batch while every other parameter updates normally. The precision counter to the
    # residual film-group noise (film/noise_scale_ratio): pick N ≈ that ratio so each film update
    # lands at the group's critical batch. Training-only, NOT version-locked, resume-forwarded.
    film_grad_accum_steps: int = 1
    _film_grad_accumulator = None      # persistent across train() calls (partial groups carry over)

    @staticmethod
    def _noise_scale_advice(global_ratio, film_ratio, film_accum, b_eff):
        """PURE advisory logic for the noise-scale ratios → list of (key, warning) pairs; [] when
        healthy. The TUI-warning half of the McCandlish instrumentation: a ratio ≫ 1 means updates
        are noise-dominated (each step's direction is mostly sideways — and under Adam the noise
        still moves params at full speed, so spurious content gets WRITTEN, not just slowed); ≪ 1
        means samples are being spent polishing an already-clean gradient instead of taking more
        steps. Each warning names the concrete fix. The film check accounts for the CONFIGURED
        --film-grad-accum-steps (applied film updates see film_accum× the batch), so a covered
        ratio warns nothing."""
        import math
        out = []
        if global_ratio is not None:
            if global_ratio > 2.0:
                out.append(("global_high", (
                    f"⚠️ [NOISE] train/noise_scale_ratio {global_ratio:.1f} — gradient NOISE-LIMITED "
                    f"(critical batch ≈ {global_ratio * b_eff / 1000:.0f}k vs effective {b_eff / 1000:.0f}k; "
                    f"updates are mostly sideways). Fix: raise --grad-accum-steps ~{math.ceil(global_ratio)}× "
                    f"(free — no VRAM/FPS cost, same rollout).")))
            elif global_ratio < 0.5:
                out.append(("global_low", (
                    f"⚠️ [NOISE] train/noise_scale_ratio {global_ratio:.2f} — OVER-BATCHED (effective "
                    f"{b_eff / 1000:.0f}k is ≫ the critical batch; samples polish an already-clean gradient). "
                    f"Fix: lower --grad-accum-steps for more update steps per sample.")))
        if film_ratio is not None:
            applied = film_ratio / max(1, int(film_accum))
            if applied > 2.0:
                out.append(("film_high", (
                    f"⚠️ [NOISE] film/noise_scale_ratio {film_ratio:.1f} with --film-grad-accum-steps "
                    f"{int(film_accum)} → APPLIED film updates still {applied:.1f}× under the FiLM group's "
                    f"critical batch (conditioning grads mostly noise → spurious per-team content). Fix: "
                    f"set --film-grad-accum-steps ~{math.ceil(film_ratio)}.")))
        return out

    def _emit_noise_scale_warnings(self, global_ratio, film_ratio, b_eff):
        """Rate-limited (30 min per key) Events-panel emit of _noise_scale_advice, after an EMA
        warm-up (first ~20 samples are settling and would false-alarm)."""
        import time
        self._nsr_samples += 1
        if self._nsr_samples < 20:
            return
        if self._nsr_warn_last is None:
            self._nsr_warn_last = {}
        advice = self._noise_scale_advice(
            global_ratio, film_ratio, getattr(self, "film_grad_accum_steps", 1), b_eff)
        now = time.time()
        for key, msg in advice:
            if now - self._nsr_warn_last.get(key, 0.0) >= 1800.0:
                self._nsr_warn_last[key] = now
                try:
                    from main.launcher.ipc import emit
                    emit(msg)
                except Exception:
                    print(msg, flush=True)

    @staticmethod
    def _zarch_participation_ratio(z):
        """Effective #axes of the batch z_arch cloud — the LIVE LUT-vs-style dial
        (gen3_zarch_film_v1; the archetype-latent note's `rank/archetype_cls_*` TODO). PR =
        (Σλ)²/Σλ² over the covariance eigenvalues of the minibatch z rows: near zarch_dim = the
        teams spread toward orthogonal identity codes (LUT-leaning — linear FiLM can then treat
        them independently); low-but-alive = compressed shared style axes; →1 = collapse. A
        batch-sampled estimate of the offline 719-team probe (tmp/zarch_neighbors_probe.py
        machinery). Returns None on a degenerate batch (too few rows / zero variance)."""
        if z is None or z.shape[0] < 3:
            return None
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / (z.shape[0] - 1)
        lam = th.linalg.eigvalsh(cov).clamp(min=0)
        s = lam.sum()
        if float(s) <= 1e-9:
            return None
        return float((s * s / (lam * lam).sum()).item())

    @staticmethod
    def _zarch_loss(z, recon_logits, species_ids):
        """The z_arch aux terms (gen3_zarch_film_v1): species multi-hot reconstruction BCE (the
        anti-collapse anchor) + the VICReg per-dim variance floor, returned SEPARATELY so the caller
        folds each at its own coef.

        ``z`` [B, D] = the team-archetype latent; ``recon_logits`` [B, n_species] = the
        ZArchEncoder.recon_head output; ``species_ids`` [B, 6] = OUR team's species ids (public —
        our own roster, no privileged label). The multi-hot target is exact under Species Clause
        (no duplicate species on a legal team); row 0 (the pad/sentinel species) is zeroed out of
        the target so a placeholder id never trains a spurious positive. The variance floor is the
        latent-belief convention: relu(1 − std(z, dim=0)).mean() — a per-dim hinge across the batch
        (z is LayerNorm'd per-sample, which does NOT prevent cross-batch collapse). Returns
        ``(recon_loss, vicreg_loss, metrics)`` or ``None``. Pure + static → unit-testable."""
        if z is None or recon_logits is None or species_ids is None:
            return None
        if z.shape[0] < 2:
            return None                                  # a 1-row batch has no cross-batch variance
        target = th.zeros_like(recon_logits)
        target.scatter_(1, species_ids.long(), 1.0)
        target[:, 0] = 0.0                               # pad/sentinel row — never a real team member
        recon = F.binary_cross_entropy_with_logits(recon_logits, target)
        std = z.std(dim=0)
        vicreg = F.relu(1.0 - std).mean()
        with th.no_grad():
            # Recon health: does the true species rank in the top-6 logits? (multi-label top-k acc)
            k = species_ids.shape[1]
            topk = recon_logits.topk(k, dim=1).indices                      # [B, 6]
            hits = (topk.unsqueeze(2) == species_ids.long().unsqueeze(1)).any(dim=1)  # [B, 6]
            valid = species_ids.long() != 0
            n_valid = valid.sum().clamp(min=1)
            recon_acc = (hits & valid).sum().float() / n_valid.float()
        metrics = {
            "recon_bce": float(recon.item()),
            "recon_topk_acc": float(recon_acc.item()),   # → 1 as z reconstructs the roster
            "std": float(std.mean().item()),             # the collapse monitor (→0 = collapsed)
            "vicreg": float(vicreg.item()),
        }
        return recon, vicreg, metrics

    @staticmethod
    def _value_dist_loss(logits, target, atoms):
        """HL-Gauss cross-entropy for the distributional VALUE head (Farebrother et al. 2024) + the
        interpretability diagnostics (v29; designs/ai_v6/design_distributional_value_critic.md).

        ``logits`` [B, N] are the head's per-atom logits; ``target`` [B] (or [B,1]) is the return in the
        SAME space as ``atoms`` [N] (the fixed support — the caller PopArt-normalizes the return when the
        scalar critic does, so the support lives in normalized units). Builds a Gaussian-smoothed soft
        target by integrating N(target, σ_g²) over each bin (σ_g = 0.75·Δ), with the two EDGE bins
        absorbing the outer tails (graceful out-of-support handling — an out-of-range return reads as
        "near the edge", not lost), then cross-entropy against ``log_softmax(logits)``. Returns
        (loss, metrics): ``entropy``/``std``/``pit_mean``/``mean_abs_err`` are the per-decision reads the
        prober renders, aggregated here for the launcher (``pit_mean`` ≈ 0.5 ⟺ calibrated). Pure + static
        → unit-testable without a full PPO. Returns None when nothing is scorable."""
        if logits is None or target is None or atoms is None:
            return None
        z = atoms.to(logits.device).reshape(-1)                      # [N] fixed support
        n = z.numel()
        if n < 2:
            return None
        t = target.to(logits.device).reshape(-1, 1)                  # [B, 1] return (already in z-space)
        delta = (z[-1] - z[0]) / (n - 1)                             # bin width (z is a linspace)
        sigma_g = 0.75 * delta                                       # HL-Gauss smoothing (σ/ς = 0.75)
        inv = 1.0 / (sigma_g * (2.0 ** 0.5))                        # 1/(σ_g·√2) for the erf-CDF
        # Standard-normal CDF Φ(u) = ½(1+erf(u/√2)), evaluated at each bin's upper / lower edge.
        cdf_hi = 0.5 * (1.0 + th.erf((z + 0.5 * delta - t) * inv))   # [B, N]
        cdf_lo = 0.5 * (1.0 + th.erf((z - 0.5 * delta - t) * inv))   # [B, N]
        p = cdf_hi - cdf_lo                                          # [B, N] interior bin masses
        # Edge-bin tail absorption: bin 0 = all mass below its upper edge; bin N-1 = all mass above its
        # lower edge. (Concatenation, not in-place, so it stays autograd-clean — p carries no grad anyway.)
        p = th.cat([cdf_hi[:, :1], p[:, 1:-1], 1.0 - cdf_lo[:, -1:]], dim=1)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-8)             # renormalize (numerical safety)
        logp = th.log_softmax(logits, dim=-1)                       # [B, N]
        loss = -(p * logp).sum(-1).mean()                           # masked-mean CE
        with th.no_grad():
            probs = th.softmax(logits, dim=-1)                      # [B, N]
            mean = (probs * z).sum(-1)                              # [B] E[Z]
            std = th.sqrt((probs * (z - mean.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0.0))
            entropy = -(probs * logp).sum(-1)                      # [B] nats
            tt = t.reshape(-1)
            pit = (probs * (z.unsqueeze(0) <= tt.unsqueeze(-1)).float()).sum(-1)  # F_pred(target) ≈ PIT
            mean_abs_err = (mean - tt).abs()
        metrics = {
            "ce": float(loss.item()),
            "entropy": float(entropy.mean().item()),
            "std": float(std.mean().item()),
            "pit_mean": float(pit.mean().item()),                  # ≈0.5 ⟺ calibrated
            "mean_abs_err": float(mean_abs_err.mean().item()),
        }
        return loss, metrics

    @staticmethod
    def _move_belief_loss(ml, known_moves, belief_moves, mode: str):
        """Supervised loss for the MoveBelief REINJECTION head (``last_move_belief_logits`` [B,6,M]).

        Two DISJOINT slot populations, selected by ``mode``:
        - REVEALED slots (mode revealed|both): seen mons. ``known_moves`` [B,6,4] holds each revealed
          slot's FULL privileged moveset; those slots are supervised DIRECTLY (slot==species, no
          matching) by a multi-label BCE — the head learns the mon's as-yet-UNREVEALED moves
          (the surprise-OHKO new-move gap).
        - UNREVEALED slots (mode unrevealed|both): hidden mons. ``belief_moves`` [B,6,4] holds the hidden
          movesets at the believed (anonymous) slots; the k believed-slot predictions are
          order-invariantly matched to the k hidden movesets (per-sample min-cost assignment — the
          slots are interchangeable, so a fixed slot↔mon target would chase a reveal-shifting
          assignment, the same defect the species aux fixed). The matching cost is the
          assignment-relevant part of BCE, ``-(pred·target)`` (the per-slot constant terms drop out of
          the argmin), so it is a cheap einsum, not a full pairwise BCE.

        The two label tensors PAD each other's slots (known_moves PADs believed slots; belief_moves
        PADs revealed slots), so 'both' simply scores each population with its own rule. A slot whose
        moveset is all-PAD (unknown moves) is NOT supervised. Returns (loss, metrics) or None
        (off / labels absent / nothing scorable). FAILS LOUD on an out-of-vocab move id. Pure + static
        (unit-tests without a full PPO)."""
        if ml is None:
            return None
        device = ml.device
        n_moves = ml.shape[-1]
        terms = []
        mv_tp = mv_pred_pos = mv_true_pos = 0
        n_revealed = n_unrevealed = 0

        def _vocab_check(ids):
            if ids.numel() and bool((ids >= n_moves).any()):
                raise ValueError(
                    f"move-belief label out of vocab: max {int(ids.max())} (n_moves {n_moves}) — "
                    "the embedding-num pipeline is corrupt (real Gen-3 move nums are all < 400)."
                )

        # ---- REVEALED: direct multi-label BCE (slot identity == revealed species) ----
        if mode in ("revealed", "both") and known_moves is not None:
            km = known_moves.long().to(device)                                     # [B, 6, 4]
            valid = km >= 0
            _vocab_check(km[valid])
            slot_has = valid.any(-1)                                               # [B, 6]
            if bool(slot_has.any()):
                multi_hot = th.zeros_like(ml)                                      # [B, 6, M]
                bb, ss, _ = valid.nonzero(as_tuple=True)
                multi_hot[bb, ss, km[valid]] = 1.0
                per_slot = F.binary_cross_entropy_with_logits(
                    ml, multi_hot, reduction="none").mean(-1)                      # [B, 6]
                terms.append(per_slot[slot_has].reshape(-1))
                with th.no_grad():
                    sel = slot_has.unsqueeze(-1)
                    pp = (ml > 0.0) & sel
                    mh = multi_hot.bool() & sel
                    mv_tp += int((pp & mh).sum()); mv_pred_pos += int(pp.sum()); mv_true_pos += int(mh.sum())
                n_revealed += int(slot_has.sum())

        # ---- UNREVEALED: order-invariant (Hungarian) multi-label BCE over the believed slots ----
        if mode in ("unrevealed", "both") and belief_moves is not None:
            bm = belief_moves.long().to(device)                                    # [B, 6, 4]
            valid = bm >= 0
            _vocab_check(bm[valid])
            slot_has = valid.any(-1)                                               # [B, 6] believed slots w/ a moveset
            counts = slot_has.sum(1)                                               # [B] k per sample
            for k in range(1, ml.shape[1] + 1):
                sel = (counts == k).nonzero(as_tuple=True)[0]
                if sel.numel() == 0:
                    continue
                n = sel.numel()
                slot_idx = slot_has[sel].nonzero(as_tuple=False)[:, 1].view(n, k)  # [n, k] believed positions
                rows = sel.view(n, 1).expand(n, k)
                preds = ml[rows, slot_idx]                                         # [n, k, M] logits
                tgt = th.zeros((n, k, n_moves), device=device)                     # [n, k, M] target multi-hots
                ids = bm[rows, slot_idx]                                           # [n, k, 4]
                vmask = ids >= 0
                if bool(vmask.any()):
                    aa, kk, _ = vmask.nonzero(as_tuple=True)
                    tgt[aa, kk, ids[vmask]] = 1.0
                # cost[a,i,j] = assignment-relevant part of BCE(pred_i, tgt_j) = -(pred_i · tgt_j). The
                # per-slot constant BCE terms are independent of the assignment → drop from the argmin.
                cost = -th.einsum('akm,ajm->akj', preds, tgt)                      # [n, k, k]
                perms = th.tensor(list(itertools.permutations(range(k))), dtype=th.long, device=device)
                ii = th.arange(k, device=device).view(1, k).expand(perms.shape[0], k)
                best_perm = perms[cost[:, ii, perms].sum(-1).argmin(1)]            # [n, k] min-cost assignment
                matched = th.gather(tgt, 1, best_perm[:, :, None].expand(n, k, n_moves))  # [n, k, M]
                per_slot = F.binary_cross_entropy_with_logits(
                    preds, matched, reduction="none").mean(-1)                     # [n, k]
                terms.append(per_slot.reshape(-1))
                with th.no_grad():
                    pp = preds > 0.0
                    mh = matched.bool()
                    mv_tp += int((pp & mh).sum()); mv_pred_pos += int(pp.sum()); mv_true_pos += int(mh.sum())
                n_unrevealed += n * k

        if not terms:
            return None
        loss = th.cat(terms).mean()
        metrics = {
            "bce": float(loss.item()),
            "precision": (mv_tp / mv_pred_pos) if mv_pred_pos else 0.0,
            "recall": (mv_tp / mv_true_pos) if mv_true_pos else 0.0,
            "revealed_slots": float(n_revealed),
            "unrevealed_slots": float(n_unrevealed),
        }
        return loss, metrics

    @staticmethod
    def _spread_belief_loss(sb, belief_spread, belief_spread_mask):
        """Supervised loss for the SPREAD belief (gen3_unified_spread_belief_v1).

        `sb` = the extractor's stashed `last_spread_belief` [B,6,5] — the believed DERIVED stats
        {atk,def,spa,spd,spe} the DamageOperator consumes. Supervise the REVEALED opp slots
        (`belief_spread_mask` [B,6]==1) toward their TRUE derived stats (`belief_spread` [B,6,5], a
        training-only privileged label) with a smooth-L1 in scale-normalised stat units, so the head
        learns the opponent's HIDDEN EV spread instead of sitting at the usage-mean prior (the
        "over-estimates the largest EV" miscalibration). The gradient flows believed → stat_head →
        opp tokens → trunk, so it joins the aux pull on the grad-balance probe.

        Returns (loss, metrics) or None (off / labels absent / no scored slot). LEAK-SAFE: the believed
        stats are a MODEL OUTPUT (the op's own input), not a label; the true-spread LABEL rides a
        training-only obs key read ONLY here, never in the pi/vf forward. Pure + static (unit-testable
        without a full PPO)."""
        if sb is None or belief_spread is None or belief_spread_mask is None:
            return None
        device = sb.device
        target = belief_spread.to(device).float()                          # [B,6,5]
        mask = belief_spread_mask.to(device).float() > 0.5                 # [B,6]
        if not bool(mask.any()):
            return None
        sb_sel = sb[mask]                                                  # [N,5]
        tgt_sel = target[mask]                                             # [N,5]
        loss = F.smooth_l1_loss(sb_sel / _SPREAD_LOSS_SCALE, tgt_sel / _SPREAD_LOSS_SCALE)
        with th.no_grad():
            mae = (sb_sel - tgt_sel).abs().mean()
            # The "over-estimate the largest EV" diagnostic: signed error on each mon's LARGEST true stat
            # (>0 ⇒ over-estimating it). Should trend toward 0 as the head learns off the prior.
            amax = tgt_sel.argmax(dim=1)
            rows = th.arange(tgt_sel.shape[0], device=device)
            largest_bias = (sb_sel[rows, amax] - tgt_sel[rows, amax]).mean()
        return loss, {"mae": float(mae.item()), "largest_bias": float(largest_bias.item()),
                      "n_slots": int(mask.sum().item())}

    @staticmethod
    def _nature_ev_belief_loss(nat_logits, ev_pred, belief_nature, belief_nature_mask, belief_ev, belief_ev_mask):
        """Supervised loss for the NATURE/EV decomposition of the generative spread belief
        (`gen3_nature_ev_belief_v1`). `nat_logits` [B,6,25] + `ev_pred` [B,6,5] are the extractor's stashed
        `last_spread_nature_logits` / `last_spread_ev` (BOTH None when the additive head is used → returns None,
        so the term is auto-skipped unless --spread-belief-nature is on). Supervise the REVEALED opp slots
        toward the privileged INVERTED (nature, EVs) label: a 25-way CE on the nature + a scale-normalised
        smooth_l1 on the EVs. This trains the decomposition DIRECTLY — the derived-stat smooth_l1 alone is
        many-to-one (many (nature, EV) reproduce one derived stat) so it can't pin the nature/EV; this is what
        actually fixes the "over-estimates the largest EV" order-statistic bias. The gradient flows
        head → opp tokens → trunk. LEAK-SAFE: the labels are training-only, read ONLY here, never in pi/vf.
        Returns (loss, metrics) or None (off / labels absent / no scored slot)."""
        if nat_logits is None or ev_pred is None or belief_nature is None or belief_ev is None:
            return None
        device = nat_logits.device
        if belief_nature_mask is None:
            return None
        nmask = belief_nature_mask.to(device).float() > 0.5                 # [B,6]
        if not bool(nmask.any()):
            return None
        nat_true = belief_nature.to(device).long()                         # [B,6]
        nl = nat_logits[nmask]                                              # [N,25]
        nt = nat_true[nmask]                                               # [N]
        nat_ce = F.cross_entropy(nl, nt)
        ev_true = belief_ev.to(device).float()                             # [B,6,5]
        evm = (belief_ev_mask.to(device).float() > 0.5) if belief_ev_mask is not None else nmask
        if bool(evm.any()):
            ev_loss = F.smooth_l1_loss(ev_pred[evm] / _EV_LOSS_SCALE, ev_true[evm] / _EV_LOSS_SCALE)
            ev_mae = (ev_pred[evm] - ev_true[evm]).abs().mean().detach()
        else:
            ev_loss = nl.new_zeros(())
            ev_mae = nl.new_zeros(())
        loss = _NATURE_CE_WEIGHT * nat_ce + _EV_LOSS_WEIGHT * ev_loss
        with th.no_grad():
            acc = (nl.argmax(dim=-1) == nt).float().mean()
        return loss, {"nature_acc": float(acc.item()), "nature_ce": float(nat_ce.item()),
                      "ev_mae": float(ev_mae.item()), "n_slots": int(nmask.sum().item())}

    @staticmethod
    def _hp_type_belief_loss(logits, hp_type_label, hp_type_mask):
        """Supervised CROSS-ENTROPY for the HP-TYPE belief (gen3_opp_hp_type_belief_v1).

        `logits` = the extractor's stashed `last_hp_type_logits` [B,6,16] — the HPTypeBelief head's
        per-opp-slot HP-type posterior logits (Smogon prior ⊕ learned delta) the DamageOperator consumes as
        its typed-HP candidate weights. Supervise the slots whose REVEALED species runs Hidden Power
        (`hp_type_mask` [B,6]==1) toward the TRUE HP type index (`hp_type_label` [B,6] ∈ 0..15, a
        training-only privileged label — Gen 3 never reveals the opp HP type, so it can't ride the obs
        vector). The gradient flows posterior → hp_type_head → opp tokens → trunk, joining the aux pull on
        the grad-balance probe.

        Returns (loss, metrics) or None (off / labels absent / no scored slot). LEAK-SAFE: the posterior is
        a MODEL OUTPUT (the op's own input), not a label; the true-type LABEL rides a training-only obs key
        read ONLY here, never in pi/vf. Pure + static (unit-testable without a full PPO)."""
        if logits is None or hp_type_label is None or hp_type_mask is None:
            return None
        device = logits.device
        label = hp_type_label.to(device).long()                            # [B,6]
        mask = hp_type_mask.to(device).float() > 0.5                       # [B,6]
        if not bool(mask.any()):
            return None
        sel_logits = logits[mask]                                          # [N,16]
        sel_label = label[mask].clamp(min=0)                               # [N] (PAD -1 is masked out already)
        loss = F.cross_entropy(sel_logits, sel_label)
        with th.no_grad():
            acc = (sel_logits.argmax(dim=-1) == sel_label).float().mean()
        return loss, {"acc": float(acc.item()), "n_slots": int(mask.sum().item())}

    @staticmethod
    def _move_belief_latent_loss(ml, latent_table, known_moves):
        """LATENT-space grading of the move belief (gen3_unified_move_system_v1) — the soft complement to
        the per-ID BCE so near-moves (Rock Slide ≈ Hidden Power Rock) grade as near.

        For each REVEALED scored slot (slot==species, like the BCE's revealed branch): the predicted
        distribution's EXPECTED move-latent ``softmax(ml) @ latent_table`` (softmax over the move axis →
        floor-robust, concentrates on the believed moves) is regressed by COSINE toward the slot's true
        moveset MEAN latent (stop-grad — the grading TARGET), plus a VICReg variance floor on the
        predictions. ``latent_table`` ``[M, D]`` is the context-free `MoveLatentEncoder.latent_table`;
        ``ml`` ``[B,6,M]``; ``known_moves``
        ``[B,6,4]`` (move nums, -1 PAD). Returns (loss, metrics) or None (off / labels absent / nothing
        scorable). Pure + static (unit-testable without a full PPO).

        COLLAPSE NOTE (differs from the species latent head): unlike `_belief_aux_loss`, whose target is an
        EXTERNAL stop-grad (the pokemon_encoder role-token), here the target is derived from the SAME
        `latent_table` whose params the prediction gradient updates (the `.detach()` only severs the
        per-step path, not the shared table). So the cosine could in principle be gamed by collapsing the
        table rows to one DIRECTION. The VICReg term below is NOT the guard against that — it floors per-dim
        MAGNITUDE variance, which an angular collapse can satisfy by scaling magnitudes. The REAL
        anti-collapse pressure is the RL TASK-ANCHORING: the same per-move latent feeds the move network
        (`PokemonEncoder.forward`), so the table can't freely collapse without hurting the policy/value
        objective. (Empirically full direction-collapse is not a reachable attractor of this loss; the
        `movelatent_std` metric monitors magnitude, not angle — a future change that weakens the move-net
        anchoring would remove the only real guard, so add a row-decorrelation term then.)"""
        if ml is None or latent_table is None or known_moves is None:
            return None
        device = ml.device
        km = known_moves.long().to(device)                                # [B,6,4]
        valid = km >= 0
        slot_has = valid.any(-1)                                          # [B,6]
        if not bool(slot_has.any()):
            return None
        # predicted distribution's expected latent (softmax kills the ~0.02 prior floor → the latent is
        # the believed moves', not the global mean), [B,6,D].
        pred_latent = F.softmax(ml, dim=-1) @ latent_table               # [B,6,D]
        # target = mean of the slot's TRUE moves' latents (stop-grad). gather + zero pads + mean.
        move_lat = latent_table[km.clamp(min=0)] * valid.unsqueeze(-1).float()   # [B,6,4,D] (pads → 0)
        tgt = (move_lat.sum(2) / valid.sum(-1, keepdim=True).clamp(min=1).float()).detach()  # [B,6,D]
        pl = pred_latent[slot_has]                                        # [N,D]
        tl = tgt[slot_has]                                                # [N,D]
        cos = F.cosine_similarity(pl, tl, dim=-1)                         # [N]
        cos_loss = (1.0 - cos).mean()
        std = th.sqrt(pl.var(dim=0, unbiased=False) + 1e-4)              # [D] per-dim MAGNITUDE floor (see COLLAPSE NOTE)
        vicreg = F.relu(_LATENT_STD_TARGET - std).mean()
        loss = cos_loss + _LATENT_VICREG_WEIGHT * vicreg
        metrics = {
            "cosine": float(cos.mean().item()),
            "std": float(std.mean().item()),
            "slots": float(int(slot_has.sum())),
        }
        return loss, metrics

    @staticmethod
    def _belief_aux_loss(bl, sp_labels, mv_labels, moves_weight: float = 1.0, latent_target=None):
        """Order-invariant (Hungarian / DETR-style) hidden-opponent belief aux loss.

        bl = {"species": [B,6,S], "moves": [B,6,M], ["latent": [B,6,D]]} (the stashed BeliefHead
        logits); sp_labels [B,6] and mv_labels [B,6,4] are the privileged int labels (-1 = revealed/pad).
        The k believed-slot predictions of each sample are matched to its k hidden-mon targets by
        **per-sample min-cost assignment** (so the anonymous slot tokens collectively cover the hidden
        SET instead of each chasing a reveal-shifting fixed slot↔mon target), then species cross-entropy
        + moves multi-label BCE are taken over the matched pairs. The matching is exact: for k ≤ TEAM_SIZE
        the k! permutations are enumerated and the min-CE-cost one chosen (vectorised per distinct k — no
        per-sample Python loop, no scipy).

        **Latent term (`latent_target` [B,6,D] given AND bl carries "latent").** On the SAME species-CE
        Hungarian assignment (one mon per slot across all heads — no conflicting pulls), a cosine loss
        regresses each believed slot's predicted latent toward the STOP-GRAD encoder role-token of its
        matched true hidden mon, plus a VICReg variance floor on the predictions (collapse guard). It is
        returned SEPARATELY (third tuple element) so the caller weights it by its own coef.

        Perf: the species log-softmax is taken on the GATHERED believed slots ([n,k,S]) not the full
        [B,6,S] (the non-believed slots are never read); the moves branch is skipped entirely when
        moves_weight==0; accuracy + moves P/R are diagnostics computed under no_grad.

        Returns (aux_tensor, metrics_dict, latent_loss_or_None) or None when nothing to score (belief
        off / labels absent / a minibatch with zero believed slots — the None guard keeps an empty
        minibatch from NaN-poisoning the loss). FAILS LOUD on an out-of-vocab label id (impossible on
        real data → a corrupt embedding-num pipeline), rather than silently dropping it. Pure + static so
        it unit-tests without a full PPO."""
        if bl is None or sp_labels is None or mv_labels is None:
            return None
        sp_logits = bl["species"]
        mv_logits = bl["moves"]
        device = sp_logits.device
        sp_labels = sp_labels.long().to(device)
        mv_labels = mv_labels.long().to(device)
        n_species = sp_logits.shape[-1]
        n_moves = mv_logits.shape[-1]
        believed = sp_labels >= 0                                                  # [B, 6] (-1 = not scored)
        counts = believed.sum(1)                                                   # [B] k per sample
        if int(counts.sum()) == 0:
            return None
        # FAIL LOUD: a believed label id must fit the vocab. Every real Gen-3 species/move num is well
        # inside max=400, so a violation means the label↔embedding num space is corrupt — crash, don't
        # silently filter and train on a hole. (-1 pads were already excluded by `believed`.) A SINGLE
        # host-sync on the happy path (the per-element max()/message only run on the failure path).
        sp_believed = sp_labels[believed]
        mv_believed = mv_labels[believed]
        sp_bad = (sp_believed >= n_species).any()
        mv_bad = (mv_believed >= n_moves).any() if mv_believed.numel() else sp_bad.new_zeros(())
        if bool(sp_bad | mv_bad):
            raise ValueError(
                f"belief label out of vocab: species max {int(sp_believed.max())} (n_species {n_species}) / "
                f"move max {int(mv_believed.max()) if mv_believed.numel() else -1} (n_moves {n_moves}) — "
                "the embedding-num pipeline is corrupt (real Gen-3 nums are all < 400)."
            )
        do_moves = moves_weight != 0.0
        latent_pred = bl.get("latent")
        do_latent = latent_pred is not None and latent_target is not None
        if do_latent:
            latent_target = latent_target.to(device)
        ce_terms, bce_terms = [], []
        # latent: per-slot cosine distance + matched preds (VICReg) + matched targets (above-chance anchor)
        cos_terms, lat_pred_terms, lat_tgt_terms = [], [], []
        n_correct = th.zeros((), device=device)
        n_slots = 0
        mv_tp = mv_pred_pos = mv_true_pos = 0  # moves precision/recall accumulators (diagnostic)
        # Group samples by their believed-slot count k; within a group every cost matrix is k×k so the
        # whole group is matched with one vectorised permutation-enumeration.
        for k in range(1, sp_labels.shape[1] + 1):
            sel = (counts == k).nonzero(as_tuple=True)[0]                          # samples with k believed
            if sel.numel() == 0:
                continue
            n = sel.numel()
            slot_idx = believed[sel].nonzero(as_tuple=False)[:, 1].view(n, k)      # [n, k] believed positions
            rows = sel.view(n, 1).expand(n, k)                                     # [n, k] sample indices
            pred_logp = th.log_softmax(sp_logits[rows, slot_idx], dim=-1)          # [n, k, S] softmax on gathered
            tgt_sp = sp_labels[rows, slot_idx]                                     # [n, k] target species
            # cost[a,i,j] = CE of predicting believed-slot i's logits at target j = -logp[a,i,tgt[a,j]]
            cost = -th.gather(pred_logp, 2, tgt_sp[:, None, :].expand(n, k, k))    # [n, k, k]
            perms = th.tensor(list(itertools.permutations(range(k))), dtype=th.long, device=device)  # [P,k]
            ii = th.arange(k, device=device).view(1, k).expand(perms.shape[0], k)
            best_perm = perms[cost[:, ii, perms].sum(-1).argmin(1)]               # [n, k] min-cost assignment
            matched_sp = th.gather(tgt_sp, 1, best_perm)                           # [n, k] matched species target
            ce_terms.append((-th.gather(pred_logp, 2, matched_sp[:, :, None]).squeeze(-1)).reshape(-1))
            with th.no_grad():
                n_correct = n_correct + (pred_logp.argmax(-1) == matched_sp).sum()
            n_slots += n * k
            if do_moves or do_latent:
                matched_label_slot = th.gather(slot_idx, 1, best_perm)             # [n, k] matched label slot
            if do_latent:
                # Same assignment as species: pred at believed slot i ↔ target role-token at the
                # matched believed slot. Cosine distance (1 − cos) + collect preds for the VICReg floor.
                pred_l = latent_pred[rows, slot_idx]                               # [n, k, D] predictions
                tgt_l = latent_target[rows, matched_label_slot]                    # [n, k, D] matched targets
                cos = (F.normalize(pred_l, dim=-1) * F.normalize(tgt_l, dim=-1)).sum(-1)  # [n, k]
                cos_terms.append((1.0 - cos).reshape(-1))
                lat_pred_terms.append(pred_l.reshape(-1, pred_l.shape[-1]))        # [n*k, D]
                lat_tgt_terms.append(tgt_l.reshape(-1, tgt_l.shape[-1]))           # [n*k, D] matched targets
            if do_moves:
                mv_pred = mv_logits[rows, slot_idx]                                # [n, k, M] predictions
                mv_ids = mv_labels[rows, matched_label_slot]                      # [n, k, 4] matched move ids
                mvalid = mv_ids >= 0                                               # [n, k, 4] (pad excluded)
                multi_hot = th.zeros_like(mv_pred)                                 # [n, k, M]
                if bool(mvalid.any()):
                    aa, kk, _ = mvalid.nonzero(as_tuple=True)
                    multi_hot[aa, kk, mv_ids[mvalid]] = 1.0
                # per-slot BCE = mean over the M move classes (same per-slot scale as the per-slot CE),
                # but ONLY for slots with ≥1 labeled move — a slot whose moves are all-pad (unknown
                # moveset) must NOT be supervised toward "predict no moves" (all-negative).
                slot_has_moves = mvalid.any(-1)                                    # [n, k]
                per_slot_bce = F.binary_cross_entropy_with_logits(
                    mv_pred, multi_hot, reduction="none").mean(-1)                 # [n, k]
                bce_terms.append(per_slot_bce[slot_has_moves].reshape(-1))
                with th.no_grad():
                    pred_present = mv_pred > 0.0                                   # sigmoid>0.5 ⇒ predicted present
                    mv_tp += int((pred_present & multi_hot.bool()).sum())
                    mv_pred_pos += int(pred_present.sum())
                    mv_true_pos += int(multi_hot.sum())
        ce = th.cat(ce_terms).mean()
        bce_cat = th.cat(bce_terms) if bce_terms else th.zeros(0, device=device)
        # numel guard: every believed slot could have an unknown moveset (all-pad) → no BCE terms → 0
        # (not NaN). In practice hidden mons have mapped moves so this is the degenerate edge.
        bce = bce_cat.mean() if bce_cat.numel() else th.zeros((), device=device)
        aux = ce + moves_weight * bce
        n_samples = int((counts > 0).sum())
        acc = float((n_correct.float() / max(1, n_slots)).item())
        metrics = {
            "species_ce": float(ce.item()),
            "moves_bce": float(bce.item()),
            "species_acc": acc,
            "species_acc_above_chance": acc - 1.0 / n_species,
            "moves_precision": (mv_tp / mv_pred_pos) if mv_pred_pos else 0.0,
            "moves_recall": (mv_tp / mv_true_pos) if mv_true_pos else 0.0,
            "k_mean": n_slots / max(1, n_samples),
            "coverage": n_samples / sp_labels.shape[0],
        }
        # Latent-belief: mean cosine distance to the matched stop-grad role-token + a VICReg variance
        # floor on the predictions. Returned separately so the caller scales it by opp_belief_latent_coef.
        latent_loss = None
        if do_latent and cos_terms:
            cos_dist = th.cat(cos_terms).mean()
            lat_pred = th.cat(lat_pred_terms, dim=0)                               # [N, D] matched preds
            lat_std = th.sqrt(lat_pred.var(dim=0, unbiased=False) + 1e-4)          # [D] per-dim std
            vicreg = th.relu(_LATENT_STD_TARGET - lat_std).mean()
            latent_loss = cos_dist + _LATENT_VICREG_WEIGHT * vicreg
            similarity = float(1.0 - cos_dist.item())
            metrics["latent_cosine"] = similarity                                 # similarity (higher better)
            metrics["latent_loss"] = float(latent_loss.item())
            metrics["latent_std"] = float(lat_std.mean().item())                  # collapse monitor (→0 = NO-GO)
            metrics["latent_vicreg"] = float(vicreg.item())
            # Interpretability anchor (the latent analog of `species_acc_above_chance`): role-tokens are
            # task-anchored, NOT orthogonal, so the raw cosine has a non-zero null. The baseline is the
            # cosine each prediction scores against a MISMATCHED true target (the within-batch roll-by-1
            # "background" a predictor would get by regressing to a typical role-token). `above_chance` =
            # matched − mismatched is the discriminative signal — small-but-positive with a healthy
            # `latent_std` means the head predicts the SET's mean role, not the per-mon identity.
            with th.no_grad():
                lat_tgt = th.cat(lat_tgt_terms, dim=0)                            # [N, D] matched targets
                pn = F.normalize(lat_pred, dim=-1)
                tn = F.normalize(lat_tgt, dim=-1)
                baseline = float((pn * th.roll(tn, 1, 0)).sum(-1).mean().item())  # cos to a non-matched target
            metrics["latent_cosine_baseline"] = baseline
            metrics["latent_cosine_above_chance"] = similarity - baseline
        return aux, metrics, latent_loss

    def _value_loss_from_se(self, se: "th.Tensor") -> "th.Tensor":
        """Tail-weighted value loss from per-sample squared errors `se` (in whatever space the branch
        uses — NORMALIZED under PopArt, so the tail selection is on the same scale the loss trains in).

        value_tail_weight == 0 → plain `se.mean()`, byte-identical to `F.mse_loss`. >0 → blend
        `(1-w)·MSE + w·CVaR`, where CVaR = mean of the worst `_VALUE_TAIL_FRAC` squared errors — it
        upweights the big value misses (the V-tail craters a probe found the critic under-prices)
        WITHOUT biasing the mean (symmetric in error sign), so the de-normalized V the GAE advantages
        read stays unbiased. A scheduling/weighting change, not a new target."""
        mse = se.mean()
        w = self.value_tail_weight
        if w <= 0.0:
            return mse
        flat = se.reshape(-1)
        k = max(1, int(_VALUE_TAIL_FRAC * flat.numel()))
        tail = th.topk(flat, k).values.mean()   # mean of the worst-k squared errors (CVaR)
        return (1.0 - w) * mse + w * tail

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
        if self._async_rollout and isinstance(env, AsyncSubprocVecEnv):
            return collect_rollouts_async(
                self, env, callback, rollout_buffer, n_rollout_steps, use_masking)
        return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps, use_masking)

    def _defensive_entropy_boost_eff(self) -> float:
        """gen3_defensive_entropy_v1: the entropy-boost multiplier at the CURRENT step. Constant
        `defensive_entropy_boost` if `defensive_entropy_anneal_frac` == 0; else linearly annealed toward 1.0,
        reaching 1.0 once `anneal_frac` of training has elapsed (uses SB3's `_current_progress_remaining`,
        which runs 1.0 at the start → 0.0 at the end). Pure → unit-testable."""
        B, af = float(self.defensive_entropy_boost), float(self.defensive_entropy_anneal_frac)
        if af <= 0.0 or B == 1.0:
            return B
        done = 1.0 - float(getattr(self, "_current_progress_remaining", 1.0))   # 0 → 1 over training
        return 1.0 + (B - 1.0) * max(0.0, 1.0 - done / af)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.

        Vendored from `sb3_contrib.MaskablePPO.train` (hash pinned in
        `_EXPECTED_UPSTREAM_TRAIN_HASH`). The only deltas vs upstream are
        marked with `# +INSTRUMENTATION` comments.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        # gen3_defensive_entropy_v1: per-minibatch diagnostics for the state-conditioned entropy boost.
        defent_flag_fracs, defent_boost_eff, defent_ent_flagged, defent_ent_unflagged = [], [], [], []
        clip_fractions = []
        vf_clip_fractions: list[float] = []  # +INSTRUMENTATION
        belief_metrics: dict[str, list[float]] = {}  # +BELIEF: per-minibatch aux diagnostics (dict of lists)
        win_prob_metrics: dict[str, list[float]] = {}  # +WIN-PROB: per-minibatch diagnostics (dict of lists)
        pubval_metrics: dict[str, list[float]] = {}    # +PUBVAL: per-minibatch diagnostics (dict of lists)
        teacher_metrics: dict[str, list[float]] = {}    # +SEARCH-TEACHER: AWR per-minibatch diagnostics
        opd_metrics: dict[str, list[float]] = {}         # +OPD: on-policy self-distillation KL diagnostics
        seed_vicreg_metrics: dict[str, list[float]] = {}  # +SEED-VICREG: per-minibatch var/cov terms
        distill_metrics: dict[str, list[float]] = {}     # +DISTILL: exploiter-distillation KL diagnostics
        value_dist_metrics: dict[str, list[float]] = {}  # +VALUE-DIST: per-minibatch HL-Gauss diagnostics
        zarch_metrics: dict[str, list[float]] = {}       # +ZARCH: recon/VICReg diagnostics (gen3_zarch_film_v1)
        # Compute once: the aux path is fully skipped when off → loss stays byte-identical to upstream.
        belief_aux_on = self.opp_belief_aux_coef > 0.0
        move_belief_on = self.move_belief_coef > 0.0  # +MOVE-BELIEF reinjection-head supervised loss
        latent_belief_on = self.opp_belief_latent_coef > 0.0  # +LATENT belief (rides the species aux call)
        move_latent_on = self.move_belief_latent_coef > 0.0  # +MOVE-LATENT grading (gen3_unified_move_system_v1)
        spread_belief_on = self.spread_belief_coef > 0.0  # +SPREAD-belief supervision (gen3_unified_spread_belief_v1)
        hp_type_belief_on = self.hp_type_belief_coef > 0.0  # +HP-TYPE belief CE (gen3_opp_hp_type_belief_v1)
        # +WIN-PROB: the head's MODE (none/read_only/shaping) lives on the extractor; the loss is added
        # whenever the mode is on AND the coef is non-zero. read_only vs shaping differ only in whether the
        # extractor stop-grads the head's input (the trunk gradient) — the loss term itself is identical.
        win_prob_on = (
            getattr(self.policy.features_extractor, "win_prob_mode", "none") != "none"
            and self.win_prob_coef != 0.0
        )
        # +PUBVAL: the public-value aux head (gen3_pubval_aux_v1). On when the mode is set AND the coef
        # is non-zero. read_only vs shaping differ only in the extractor's stop-grad of the head's input
        # — the loss term itself is identical. OFF → skipped (loss byte-identical to upstream).
        pubval_on = (
            getattr(self.policy.features_extractor, "pubval_mode", "none") != "none"
            and self.pubval_coef != 0.0
        )
        # +ZARCH (gen3_zarch_film_v1): the z_arch recon + VICReg aux. On when the extractor built the
        # encoder AND either coef is non-zero. Gradients reach ONLY the ZArchEncoder's own params
        # (detached embedding reads) — no trunk pull, hence no grad-balance entry. OFF → skipped.
        zarch_on = (
            getattr(self.policy.features_extractor, "zarch_film", "off") != "off"
            and (self.zarch_recon_coef != 0.0 or self.zarch_vicreg_coef != 0.0)
        )
        # +VALUE-DIST: the distributional value head's HL-Gauss CE aux loss. On when the mode is set AND
        # the coef is non-zero. read_only vs shaping differ only in the extractor's stop-grad of the head's
        # input — the loss term is identical. OFF → skipped (loss byte-identical to upstream).
        # gen3_dist_critic_v1 (Phase B): the distributional head IS the critic — GAE reads E[Z]
        # (policy._critic_value), the HL-Gauss CE is the PRIMARY value loss (weighted by vf_coef,
        # not value_dist_coef), and the scalar MSE term is dropped (value_net freezes as a fallback).
        value_from_dist = bool(getattr(self.policy, "_value_from_dist", False))
        value_dist_on = (
            getattr(self.policy.features_extractor, "value_dist_mode", "none") != "none"
            and (self.value_dist_coef != 0.0 or value_from_dist)   # Phase B forces the CE on
        )
        # +SEARCH-TEACHER: AWR policy distillation. On when enabled, the coef is non-zero, AND the
        # standalone correction buffer has been populated (the callback fills it from worker shards).
        # Each minibatch samples its OWN correction batch + does its OWN policy forward (off-policy
        # states not in the rollout). OFF / empty buffer → skipped (loss byte-identical to upstream).
        search_teacher_on = (
            getattr(self, "_search_teacher_on", False) and self.search_teacher_coef != 0.0
            and getattr(self, "_correction_buffer", None) is not None
            and len(self._correction_buffer) > 0
        )
        # +OPD: on-policy self-distillation. On when enabled, the coef is non-zero, AND the SAME
        # standalone correction buffer (filled by the SearchTeacherCallback, its workers building π')
        # is populated. Its OWN get_distribution forward, like the search-teacher AWR. A sampled batch
        # with no π' (an AWR-only buffer) is skipped by the None-guard. OFF → byte-identical to upstream.
        opd_on = (
            getattr(self, "_opd_on", False) and self.opd_coef != 0.0
            and getattr(self, "_correction_buffer", None) is not None
            and len(self._correction_buffer) > 0
        )
        # +DISTILL (gen3_exploiter_distill_v1): exploiter distillation, N teachers. On when a non-empty list
        # of frozen teacher models is attached AND the coef is non-zero. Per minibatch: ONE student forward
        # + one forward per teacher, each KL masked to that teacher's team states (the `distill_mask` obs key
        # holds an INTEGER team-id — 0 = none, k = teacher k, 1-indexed). Per-teacher mean-KLs are averaged
        # (per-archetype balancing → no teacher dominates). N=1 is byte-identical to the single-teacher form
        # (id ∈ {0,1}). OFF (empty list / coef 0) → byte-identical to upstream.
        distill_on = (
            bool(getattr(self, "_distill_teachers", None)) and self.distill_coef != 0.0
        )

        continue_training = True

        # +INSTRUMENTATION: gradient-balance + value-scale diagnostics (grad_balance.py).
        # The dual-head extractor shares one trunk; both losses' gradients compete there. We
        # sample that pull ONCE per train() call (first minibatch) so vf_coef / return
        # normalization (PopArt) can be tuned to a number rather than inferred from KL.
        shared_trunk = shared_trunk_parameters(self.policy.features_extractor)
        grad_balance: dict[str, float] = {}
        rank_metrics: dict[str, float] = {}  # effective rank of trunk / value_cls / policy reps (once/train)
        grad_norms: list[float] = []  # pre-clip total grad norm (shows grad-clip activity)

        # +PopArt: advance the value-target normalizer once per train() (before the epochs) from
        # this rollout's returns; update() also POP-rescales value_net so its de-normalized outputs
        # are preserved. The value loss below then trains in normalized space. No-op when disabled.
        popart = getattr(self.policy, "popart", None)
        if popart is not None:
            popart.update(
                th.as_tensor(self.rollout_buffer.returns, device=self.device), self.policy.value_net
            )

        # +GRAD-ACCUM: number of `batch_size` micro-batches whose gradients are summed before one
        # optimizer.step() (1 = OFF, stock one-step-per-minibatch). See the class attr docstring.
        accum = max(1, int(getattr(self, "grad_accum_steps", 1)))

        # +NOISE-SCALE: when accumulating (accum>=2) we get gradient norms at two batch sizes for free —
        # one micro-batch (batch_size) and the full first group (batch_size·accum) — which is exactly
        # what the McCandlish gradient-noise-scale estimator needs. Captured once per train() (group 0 of
        # epoch 0) so |G_small|² and |G_big|² come from the SAME data; folded into the EMAs after the epochs.
        noise_g_small_sq = None   # ‖single micro-batch gradient‖²  (B = batch_size)
        noise_g_big_sq = None     # ‖accumulated group gradient‖²   (B = batch_size·accum)
        # +FILM-NOISE-SCALE (gen3_zarch_film_v1 follow-up): the SAME two-point estimator restricted
        # to the FiLM generator params — the per-GROUP critical batch size. The global noise scale
        # cannot resolve whether the ~33k-param conditioning gradient is signal or noise (it is
        # drowned by the ~10M-param total); B_simple(film) ≫ B_simple(global) ≫ effective batch would
        # say the per-team RL gradient into the conditioners sits below its noise floor at our batch —
        # the quantitative form of the sample-starvation/"persistent net cost" hypothesis.
        noise_film_small_sq = None
        noise_film_big_sq = None
        _fe_ns = self.policy.features_extractor
        film_noise_params = (
            list(_fe_ns.film_pi.parameters()) + list(_fe_ns.film_vf.parameters())
            if getattr(_fe_ns, "zarch_film", "off") != "off" else []
        )
        # +FILM-GRAD-ACCUM (--film-grad-accum-steps): per-group accumulation for the FiLM
        # generators — see _GroupGradAccumulator. Persistent on self so partial groups carry
        # across train() calls; None (off / no film group) ⇒ every step site is byte-identical.
        _film_accum_k = int(getattr(self, "film_grad_accum_steps", 1) or 1)
        film_grad_accum = None
        if film_noise_params and _film_accum_k > 1:
            if self._film_grad_accumulator is None:
                self._film_grad_accumulator = _GroupGradAccumulator(film_noise_params)
            film_grad_accum = self._film_grad_accumulator

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # +GRAD-ACCUM: start each accumulation group with a clean grad buffer; count micro-batches.
            self.policy.optimizer.zero_grad()
            micro_in_group = 0
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if popart is not None:
                    # +PopArt: value loss in NORMALIZED space (both target and prediction scaled by
                    # the running mu/sigma) → O(1) gradient, no longer swamping the shared trunk.
                    # Mutually exclusive with vf-clipping (enforced at startup).
                    # +TAIL: per-sample SE in normalized space → _value_loss_from_se (w=0 ⇒ MSE).
                    value_loss = self._value_loss_from_se(
                        (popart.normalize(rollout_data.returns) - popart.normalize(values)) ** 2
                    )
                elif self.clip_range_vf is None:
                    # No clipping
                    value_loss = self._value_loss_from_se((rollout_data.returns - values) ** 2)
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                    # +INSTRUMENTATION: fraction of value updates that hit the clip bound
                    vf_clip_fraction = th.mean(
                        (th.abs(values - rollout_data.old_values) > clip_range_vf).float()
                    ).item()
                    vf_clip_fractions.append(vf_clip_fraction)
                    # +TAIL: per-sample SE on the clipped prediction → _value_loss_from_se.
                    value_loss = self._value_loss_from_se((rollout_data.returns - values_pred) ** 2)
                value_losses.append(value_loss.item())

                # Entropy loss favors exploration. gen3_defensive_entropy_v1: when defensive_entropy_boost > 1,
                # multiply the per-decision entropy bonus by the (annealed) boost on decisions the env flagged
                # `defensive_opportunity` — keeping the policy exploratory on recovery/cure choices instead of
                # collapsing to attacking, WITHOUT touching the reward. OFF (boost == 1) byte-identical.
                ent_per = -log_prob if entropy is None else entropy          # [B] per-decision entropy (nats)
                entropy_loss = -th.mean(ent_per)                             # standard (unweighted) metric
                entropy_losses.append(entropy_loss.item())
                do_flag = rollout_data.observations.get("defensive_opportunity")
                if self.defensive_entropy_boost != 1.0 and do_flag is not None:
                    b_eff = self._defensive_entropy_boost_eff()
                    flag = do_flag.to(ent_per.device).reshape(-1).float()    # [B] 1.0 on defensive decisions
                    ent_loss_used = -th.mean((1.0 + (b_eff - 1.0) * flag) * ent_per)
                    with th.no_grad():
                        fm = flag > 0.5
                        defent_flag_fracs.append(float(fm.float().mean().item()))
                        defent_boost_eff.append(b_eff)
                        if bool(fm.any()):    defent_ent_flagged.append(float(ent_per[fm].mean().item()))
                        if bool((~fm).any()): defent_ent_unflagged.append(float(ent_per[~fm].mean().item()))
                else:
                    ent_loss_used = entropy_loss

                # Phase B (value_from_dist): the scalar MSE value term is DROPPED (value_net frozen —
                # the CE below at vf_coef is the critic). value_loss is still logged as the
                # E[Z]-mean-vs-return diagnostic. Off → the standard vf_coef·MSE term.
                _vf_term = 0.0 if value_from_dist else self.vf_coef * value_loss
                loss = policy_loss + self.ent_coef * ent_loss_used + _vf_term

                # +BELIEF: hidden-opponent belief aux loss. evaluate_actions(rollout_data.observations,
                # …) ran the extractor forward just above, stashing per-slot logits for THIS minibatch;
                # the privileged labels ride the same obs dict (training-only keys). Masked to the
                # believed slots, folded in at opp_belief_aux_coef. OFF → skipped (loss byte-identical).
                belief_aux_term = None  # the WEIGHTED aux contribution, for the grad-balance probe
                latent_belief_term = None  # the WEIGHTED latent contribution (rides the same probe)
                if belief_aux_on:
                    aux_out = self._belief_aux_loss(
                        self.policy.features_extractor.last_belief_logits,
                        rollout_data.observations.get("belief_species"),
                        rollout_data.observations.get("belief_moves"),
                        moves_weight=self.opp_belief_moves_weight,
                        # The latent target (stop-grad encoder role-tokens) the extractor stashed this
                        # minibatch; passed only when the latent term is on (else no latent loss computed).
                        latent_target=(self.policy.features_extractor.last_belief_target_latent
                                       if latent_belief_on else None),
                    )
                    if aux_out is not None:
                        aux, belief_m, latent_loss = aux_out
                        belief_aux_term = self.opp_belief_aux_coef * aux
                        loss = loss + belief_aux_term
                        belief_m["aux_loss"] = float(aux.item())
                        if latent_loss is not None and latent_belief_on:
                            latent_belief_term = self.opp_belief_latent_coef * latent_loss
                            loss = loss + latent_belief_term
                        for _bk, _bv in belief_m.items():
                            belief_metrics.setdefault(_bk, []).append(float(_bv))

                # +MOVE-BELIEF: predict+reinject the opp moveset. The extractor forward (run by
                # evaluate_actions above) stashed last_move_belief_logits; known_moves/belief_moves ride
                # the same training-only obs dict. Folded at move_belief_coef; mode read off the extractor
                # (single source). OFF → skipped (byte-identical). Its gradient ALSO flows into the trunk
                # via the reinjection, so it joins the aux pull on the grad-balance probe.
                move_belief_term = None
                if move_belief_on:
                    mb_out = self._move_belief_loss(
                        self.policy.features_extractor.last_move_belief_logits,
                        rollout_data.observations.get("known_moves"),
                        rollout_data.observations.get("belief_moves"),
                        self.policy.features_extractor.move_belief_mode,
                    )
                    if mb_out is not None:
                        mb_loss, mb_m = mb_out
                        move_belief_term = self.move_belief_coef * mb_loss
                        loss = loss + move_belief_term
                        mb_m["loss"] = float(mb_loss.item())
                        for _mk, _mv in mb_m.items():
                            belief_metrics.setdefault("move_" + _mk, []).append(float(_mv))

                # +SEED-VICREG (gen3_seed_vicreg_v1): variance+covariance floor on the multi-seed
                # critic readout's [B,k,D] outputs — reads the LIVE (grad-carrying) stash the
                # evaluate_actions forward above just wrote, so the gradient flows into the seed
                # queries/kv_proj via the vf path. coef 0 → never entered (byte-identical). A coef>0
                # with no readout was already rejected at startup (assert_seed_vicreg_wirable).
                if self.value_seed_vicreg_coef > 0.0:
                    _svo = getattr(getattr(getattr(self.policy.features_extractor, "assembler", None),
                                           "seed_readout", None), "last_outputs", None)
                    if _svo is not None:
                        from agents.model.seed_vicreg import seed_vicreg_loss
                        svr_loss, svr_m = seed_vicreg_loss(_svo)
                        loss = loss + self.value_seed_vicreg_coef * svr_loss
                        for _sk, _sv in svr_m.items():
                            seed_vicreg_metrics.setdefault(_sk, []).append(_sv)
                        seed_vicreg_metrics.setdefault("value_seeds/vicreg_loss", []).append(
                            float(svr_loss.detach()))

                # +SEED-QUANTILE: the per-seed pinball fold. `last_seed_quantile_preds` [B,k] is the
                # shared head's read of THIS minibatch's seed outputs (grad-carrying). The target is
                # the rollout return in the critic's frame — PopArt-normalized when the critic is, so
                # the taus land in the same units the value head learns in.
                if self.seed_quantile_coef > 0.0:
                    _sqp = getattr(self.policy.features_extractor, "last_seed_quantile_preds", None)
                    _sqh = getattr(self.policy.features_extractor, "seed_quantile_head", None)
                    if _sqp is not None and _sqh is not None:
                        from agents.model.seed_quantile import seed_quantile_loss
                        _tgt = (popart.normalize(rollout_data.returns) if popart is not None
                                else rollout_data.returns)
                        sq_loss, sq_m = seed_quantile_loss(_sqp, _tgt.flatten(), _sqh.taus)
                        loss = loss + self.seed_quantile_coef * sq_loss
                        for _qk, _qv in sq_m.items():
                            seed_vicreg_metrics.setdefault(_qk, []).append(_qv)

                # +MOVE-LATENT (gen3_unified_move_system_v1): grade the move belief in latent space so
                # near-moves (Rock Slide ≈ HP Rock) grade as near — the soft complement to the per-ID BCE.
                # Reads the extractor's context-free move-latent table (stashed this minibatch) + the same
                # known_moves labels. Its gradient flows into the move-belief head AND the MoveLatentEncoder
                # (the table) → it joins the aux pull on the trunk. OFF → skipped (byte-identical).
                move_latent_term = None
                if move_latent_on:
                    ml_out = self._move_belief_latent_loss(
                        self.policy.features_extractor.last_move_belief_logits,
                        self.policy.features_extractor.last_move_latent_table,
                        rollout_data.observations.get("known_moves"),
                    )
                    if ml_out is not None:
                        ml_loss, ml_m = ml_out
                        move_latent_term = self.move_belief_latent_coef * ml_loss
                        loss = loss + move_latent_term
                        ml_m["loss"] = float(ml_loss.item())
                        for _lk, _lv in ml_m.items():
                            belief_metrics.setdefault("movelatent_" + _lk, []).append(float(_lv))

                # +SPREAD-BELIEF (gen3_unified_spread_belief_v1): supervise the believed opp spread toward
                # the TRUE derived stats so the DamageOperator prices damage against the real bulk/offense/
                # speed instead of the usage-mean prior. evaluate_actions stashed last_spread_belief; the
                # belief_spread/_mask labels ride the same training-only obs dict. Its gradient flows into
                # the SpreadBelief head + reinjection → the trunk, so it joins the aux pull. OFF → skipped.
                spread_belief_term = None
                if spread_belief_on:
                    sb_out = self._spread_belief_loss(
                        self.policy.features_extractor.last_spread_belief,
                        rollout_data.observations.get("belief_spread"),
                        rollout_data.observations.get("belief_spread_mask"),
                    )
                    if sb_out is not None:
                        sb_loss, sb_m = sb_out
                        spread_belief_term = self.spread_belief_coef * sb_loss
                        loss = loss + spread_belief_term
                        sb_m["loss"] = float(sb_loss.item())
                        for _sk, _sv in sb_m.items():
                            belief_metrics.setdefault("spread_" + _sk, []).append(float(_sv))

                # +NATURE/EV BELIEF (gen3_nature_ev_belief_v1): supervise the generative spread belief's NATURE
                # categorical + EV decomposition toward the privileged inverted (nature, EVs) label, so the head
                # learns the RIGHT decomposition (the derived loss above is many-to-one). The stashed
                # last_spread_nature_logits/_ev are None unless --spread-belief-nature → the loss is None →
                # skipped. Folded at the SAME spread_belief_coef (one knob supervises the whole spread belief).
                nature_ev_term = None
                if spread_belief_on:
                    ne_out = self._nature_ev_belief_loss(
                        self.policy.features_extractor.last_spread_nature_logits,
                        self.policy.features_extractor.last_spread_ev,
                        rollout_data.observations.get("belief_nature"),
                        rollout_data.observations.get("belief_nature_mask"),
                        rollout_data.observations.get("belief_ev"),
                        rollout_data.observations.get("belief_ev_mask"),
                    )
                    if ne_out is not None:
                        ne_loss, ne_m = ne_out
                        nature_ev_term = self.spread_belief_coef * ne_loss
                        loss = loss + nature_ev_term
                        ne_m["loss"] = float(ne_loss.item())
                        for _nk, _nv in ne_m.items():
                            belief_metrics.setdefault("natureev_" + _nk, []).append(float(_nv))

                # +HP-TYPE BELIEF (gen3_opp_hp_type_belief_v1): supervise the HPTypeBelief posterior toward the
                # TRUE opp HP type so the DamageOperator prices the right typed-HP threat (the "opp HP reads
                # immune" fix). evaluate_actions stashed last_hp_type_logits; the hp_type_label/_mask labels
                # ride the same training-only obs dict. Its gradient flows into the HPTypeBelief head → the
                # trunk, joining the aux pull. OFF → skipped (loss byte-identical).
                hp_type_term = None
                if hp_type_belief_on:
                    hp_out = self._hp_type_belief_loss(
                        self.policy.features_extractor.last_hp_type_logits,
                        rollout_data.observations.get("hp_type_label"),
                        rollout_data.observations.get("hp_type_mask"),
                    )
                    if hp_out is not None:
                        hp_loss, hp_m = hp_out
                        hp_type_term = self.hp_type_belief_coef * hp_loss
                        loss = loss + hp_type_term
                        hp_m["loss"] = float(hp_loss.item())
                        for _hk, _hv in hp_m.items():
                            belief_metrics.setdefault("hptype_" + _hk, []).append(float(_hv))

                # +WIN-PROB: auxiliary win-probability BCE. evaluate_actions ran the extractor forward
                # above, stashing last_win_prob_logits for THIS minibatch; the MC outcome label + its
                # known-mask ride the same obs dict (the WinProbLabelCallback overwrote the placeholders
                # post-collection). Folded at win_prob_coef. Under read_only the head's input was
                # stop-grad'd in the extractor, so this term trains only the head's own params (no trunk
                # gradient); under shaping it also pulls the trunk. OFF → skipped (loss byte-identical).
                win_prob_term = None
                if win_prob_on:
                    wp_out = self._win_prob_loss(
                        self.policy.features_extractor.last_win_prob_logits,
                        rollout_data.observations.get("win_target"),
                        rollout_data.observations.get("win_mask"),
                        rollout_data.observations.get("win_margin"),
                    )
                    if wp_out is not None:
                        wp_loss, wp_m = wp_out
                        win_prob_term = self.win_prob_coef * wp_loss
                        loss = loss + win_prob_term
                        for _wk, _wv in wp_m.items():
                            win_prob_metrics.setdefault(_wk, []).append(float(_wv))

                # +PUBVAL: public-value aux soft-BCE (gen3_pubval_aux_v1). evaluate_actions stashed
                # last_pubval_logits for THIS minibatch; the frozen human-replay V_pub target + mask ride
                # the obs dict as REAL per-step values (env-computed at decision time — no back-fill).
                # Folded at pubval_coef. Under read_only the head's input was stop-grad'd in the extractor
                # (head-only training); under shaping the human positional prior also pulls the trunk.
                # OFF → skipped (loss byte-identical).
                pubval_term = None
                if pubval_on:
                    pv_out = self._pubval_loss(
                        self.policy.features_extractor.last_pubval_logits,
                        rollout_data.observations.get("pubval_target"),
                        rollout_data.observations.get("pubval_mask"),
                    )
                    if pv_out is not None:
                        pv_loss, pv_m = pv_out
                        pubval_term = self.pubval_coef * pv_loss
                        loss = loss + pubval_term
                        for _pk, _pv in pv_m.items():
                            pubval_metrics.setdefault(_pk, []).append(float(_pv))

                # +VALUE-DIST: distributional value head HL-Gauss CE. evaluate_actions ran the extractor
                # forward above, stashing last_value_dist_logits for THIS minibatch; the target is the
                # rollout return, PopArt-normalized when the scalar critic is (so it lands in the head's
                # support space). Folded at value_dist_coef. Under read_only the head's input was
                # stop-grad'd in the extractor (head-only training, no trunk gradient); under shaping it
                # also pulls the trunk. OFF → skipped (loss byte-identical).
                value_dist_term = None
                if value_dist_on:
                    _vd_head = self.policy.features_extractor.value_dist_head
                    _vd_logits = self.policy.features_extractor.last_value_dist_logits
                    if _vd_head is not None and _vd_logits is not None:
                        _vd_target = (
                            popart.normalize(rollout_data.returns) if popart is not None
                            else rollout_data.returns
                        )
                        vd_out = self._value_dist_loss(_vd_logits, _vd_target, _vd_head.atoms)
                        if vd_out is not None:
                            vd_loss, vd_m = vd_out
                            # Phase B: the CE is the PRIMARY critic loss (vf_coef weight); else the aux coef.
                            _ce_w = self.vf_coef if value_from_dist else self.value_dist_coef
                            value_dist_term = _ce_w * vd_loss
                            loss = loss + value_dist_term
                            for _vk, _vv in vd_m.items():
                                value_dist_metrics.setdefault(_vk, []).append(float(_vv))

                # +ZARCH (gen3_zarch_film_v1): the z_arch recon BCE + VICReg variance floor. The
                # evaluate_actions forward above stashed last_zarch + the grad-gated recon
                # logits/species ids for THIS minibatch; each term folds at its own coef. Touches
                # only the ZArchEncoder's params (detached embedding reads → zero trunk gradient).
                # OFF → skipped (loss byte-identical).
                if zarch_on:
                    _fe = self.policy.features_extractor
                    z_out = self._zarch_loss(
                        _fe.last_zarch, _fe.last_zarch_recon_logits, _fe.last_zarch_species_ids)
                    if z_out is not None:
                        z_recon, z_vicreg, z_m = z_out
                        loss = loss + (self.zarch_recon_coef * z_recon
                                       + self.zarch_vicreg_coef * z_vicreg)
                        for _zk, _zv in z_m.items():
                            zarch_metrics.setdefault(_zk, []).append(float(_zv))

                # +DISTILL (gen3_exploiter_distill_v1): ON-POLICY KL toward a frozen per-team SPECIALIST,
                # masked to the rollout states where the trainee pilots the teacher's team (`distill_mask`).
                # Its own get_distribution forwards — the student's (fresh, so its extractor re-stash can't
                # clobber the aux losses above, which are already folded) + the FROZEN teacher's under
                # no_grad. Folded at distill_coef; policy-only (never touches the value head). OFF (coef 0 /
                # no teacher) → the whole block is skipped, loss byte-identical.
                distill_term = None
                if distill_on:
                    _tid = rollout_data.observations.get("distill_mask")   # INTEGER team-id [B,1]: 0=none, k=teacher k
                    if _tid is not None and float(_tid.reshape(-1).max()) >= 1.0:
                        _tid_flat = _tid.reshape(-1)
                        # ONE student forward, reused across all teachers (the teacher forwards are frozen).
                        # gen3_exploiter_distill_v1 optimization: REUSE the student pi distribution the
                        # evaluate_actions forward above already built (self.policy._last_pi_distribution),
                        # instead of a redundant second get_distribution — the KL is bit-identical (masked
                        # vs raw logits agree over legal actions; illegal contribute 0). Fall back to a fresh
                        # forward if the stash is somehow absent (defensive; evaluate_actions always sets it).
                        _last_pi = getattr(self.policy, "_last_pi_distribution", None)
                        _s_logits = (_last_pi.distribution.logits if _last_pi is not None
                                     else self.policy.get_distribution(
                                         rollout_data.observations).distribution.logits)
                        # +VALUE-DISTILL (gen3_exploiter_value_distill_v1): also pour the teacher's per-team
                        # VALUE into the student. Requires policy distill (coherence). OFF (coef 0) → the
                        # teacher predict_values forward is skipped, loss byte-identical.
                        _vd_on = self.distill_value_coef != 0.0
                        _s_val = values.flatten() if _vd_on else None        # student V (real-unit, WITH grad)
                        # +FITNETS VALUE-FEATURE distill (gen3_exploiter_value_feat_distill_v1): match the
                        # teacher's INTERMEDIATE value-CLS pool (the 128-dim hint) instead of the collapsed
                        # scalar. The student's `last_value_pooled` from the evaluate_actions forward above
                        # (WITH grad) — the teacher forwards below run on their OWN extractors, so this student
                        # stash is not clobbered. OFF (coef 0) → no teacher value_pooled read, loss byte-identical.
                        _vfd_on = self.distill_value_feat_coef != 0.0
                        _s_vfeat = self.policy.features_extractor.last_value_pooled if _vfd_on else None
                        _per_teacher_kl, _per_teacher_vd, _per_teacher_vfd = [], [], []
                        for _k, _teacher in enumerate(self._distill_teachers, start=1):
                            _sel = (_tid_flat == _k).to(_s_logits.dtype)      # states on teacher k's team
                            if float(_sel.sum()) < 1.0:
                                continue
                            # Each frozen teacher has its OWN (older) obs space — pass only the keys it knows
                            # (SB3's preprocess_obs iterates obs keys against the space; it needs just
                            # observation + action_mask). See gen3_exploiter_distill_v1 invariance (Δ=0).
                            _t_obs = {key: v for key, v in rollout_data.observations.items()
                                      if key in _teacher.observation_space.spaces}
                            with th.no_grad():
                                _t_logits = _teacher.policy.get_distribution(_t_obs).distribution.logits
                                # gen3_exploiter_value_feat_distill_v1: the get_distribution forward above ran
                                # the teacher's FULL extractor, so its `last_value_pooled` (the hint) is set for
                                # THESE states — capture it now, BEFORE the predict_values forward below re-runs
                                # + overwrites it. Under no_grad → detached (the FitNets target is frozen).
                                _t_vfeat = (_teacher.policy.features_extractor.last_value_pooled
                                            if _vfd_on else None)
                            _d_out = self._distill_loss(_s_logits, _t_logits, rollout_data.action_masks, _sel)
                            if _d_out is not None:
                                _kl_k, _m_k = _d_out
                                _per_teacher_kl.append(_kl_k)
                                for _mk, _mv in _m_k.items():   # per-teacher diagnostics (distill/t{k}_*)
                                    distill_metrics.setdefault(f"t{_k}_{_mk}", []).append(float(_mv))
                            if _vfd_on:
                                # Masked cosine distance between the student + teacher value-CLS pools on
                                # teacher-k's states (the FitNets hint match).
                                _vfd_k = self._value_feat_distill(_s_vfeat, _t_vfeat, _sel)
                                if _vfd_k is not None:
                                    _per_teacher_vfd.append(_vfd_k)
                                    distill_metrics.setdefault(f"t{_k}_value_feat_cos", []).append(float(_vfd_k))
                            if _vd_on:
                                # Teacher V (real-unit, frozen); masked MSE vs student V in the PopArt frame.
                                with th.no_grad():
                                    _t_val = _teacher.policy.predict_values(_t_obs).flatten()
                                _vd_k = self._value_distill_mse(_s_val, _t_val, _sel, popart)
                                if _vd_k is not None:
                                    _per_teacher_vd.append(_vd_k)
                                    distill_metrics.setdefault(f"t{_k}_value_mse", []).append(float(_vd_k))
                        if _per_teacher_kl:
                            # Per-archetype balancing: average the per-teacher mean-KLs so a teacher with
                            # fewer states still contributes comparable gradient (not swamped by a big one).
                            _distill_kl = th.stack(_per_teacher_kl).mean()
                            distill_term = self.distill_coef * _distill_kl
                            loss = loss + distill_term
                            distill_metrics.setdefault("kl", []).append(float(_distill_kl))
                            distill_metrics.setdefault("n_teachers_active", []).append(float(len(_per_teacher_kl)))
                        if _per_teacher_vd:
                            _distill_vd = th.stack(_per_teacher_vd).mean()    # balanced like the policy KL
                            loss = loss + self.distill_value_coef * _distill_vd
                            distill_metrics.setdefault("value_mse", []).append(float(_distill_vd))
                        if _per_teacher_vfd:
                            _distill_vfd = th.stack(_per_teacher_vfd).mean()  # balanced like the policy KL
                            loss = loss + self.distill_value_feat_coef * _distill_vfd
                            distill_metrics.setdefault("value_feat_cos", []).append(float(_distill_vfd))

                # +SEARCH-TEACHER: AWR policy distillation toward the verified-better action. The
                # corrections are OFF-POLICY (searched eval-trace states, not in this rollout), so this
                # samples its OWN minibatch from the standalone _correction_buffer and runs its OWN policy
                # forward (get_distribution → masked logits). Folded at search_teacher_coef; the CE
                # gradient pulls the trunk (measured by grad/searchteacher_share). The OPTIONAL value term
                # (default coef 0) is off-policy (the search value is V^π*) — kept behind its own coef.
                # OFF / empty buffer → skipped (loss byte-identical).
                searchteacher_term = None
                if search_teacher_on:
                    _batch = self._correction_buffer.sample(self.search_teacher_batch_size)
                    if _batch:
                        from agents.training.teacher.buffer import CorrectionBuffer as _CB
                        _td = _CB.to_tensors(_batch, self.device)
                        _dist = self.policy.get_distribution(_td["obs_dict"])
                        _st = self._searchteacher_loss(
                            _dist.distribution.logits, _td["action_mask"], _td["better_action"],
                            _td["advantage"], beta_awr=self.search_teacher_beta)
                        if _st is not None:
                            _st_loss, _st_m = _st
                            searchteacher_term = self.search_teacher_coef * _st_loss
                            if self.search_teacher_value_coef != 0.0:   # OFF by default (soundness)
                                _vt = self.policy.predict_values(_td["obs_dict"]).flatten()
                                _vtgt = (popart.normalize(_td["confirmed_value"]) if popart is not None
                                         else _td["confirmed_value"])
                                searchteacher_term = searchteacher_term + \
                                    self.search_teacher_value_coef * ((_vt - _vtgt) ** 2).mean()
                            loss = loss + searchteacher_term
                            for _tk, _tv in _st_m.items():
                                teacher_metrics.setdefault(_tk, []).append(float(_tv))

                # +OPD: on-policy self-distillation KL(π' ‖ π_student). Like the search-teacher AWR above,
                # this samples the SAME standalone _correction_buffer + runs its OWN get_distribution
                # forward — but distils the FULL improved distribution π' (the beam's per-action
                # backed-up values, built worker-side) instead of only the single action A*. Folded at
                # opd_coef; the KL gradient pulls the trunk (measured by grad/opd_share). A sampled batch
                # with no π' (an AWR-only buffer) → to_tensors sets pi_target None → the loss None-guards
                # (skipped). OFF / empty buffer → skipped (loss byte-identical).
                opd_term = None
                if opd_on:
                    _obatch = self._correction_buffer.sample(self.search_teacher_batch_size)
                    if _obatch:
                        from agents.training.teacher.buffer import CorrectionBuffer as _CB
                        _otd = _CB.to_tensors(_obatch, self.device)
                        if _otd.get("pi_target") is not None:   # skip an AWR-only (π'-less) sample
                            _odist = self.policy.get_distribution(_otd["obs_dict"])
                            _opd = self._opd_loss(
                                _odist.distribution.logits, _otd["action_mask"], _otd["pi_target"])
                            if _opd is not None:
                                _opd_loss_t, _opd_m = _opd
                                opd_term = self.opd_coef * _opd_loss_t
                                loss = loss + opd_term
                                for _ok, _ov in _opd_m.items():
                                    opd_metrics.setdefault(_ok, []).append(float(_ov))

                # Per-term auxiliary pull on the shared trunk, for the grad-balance probe — EVERY
                # active scaffold competes with policy/value there, so each is broken out INDIVIDUALLY
                # (not lumped into one "belief" norm) and the probe puts them on one common denominator
                # so policy/value/each-aux are mutually comparable + sum to ~1 (grad_balance.py). Only
                # the terms set this minibatch are included (a belief term is None on a zero-believed
                # minibatch; win_prob/value_dist None when their head is off).
                aux_probe_terms: dict[str, th.Tensor] = {}
                if belief_aux_term is not None:    aux_probe_terms["species_belief"] = belief_aux_term
                if move_belief_term is not None:   aux_probe_terms["move_belief"] = move_belief_term
                if latent_belief_term is not None: aux_probe_terms["latent"] = latent_belief_term
                if move_latent_term is not None:   aux_probe_terms["move_latent"] = move_latent_term
                if spread_belief_term is not None: aux_probe_terms["spread_belief"] = spread_belief_term
                if nature_ev_term is not None:     aux_probe_terms["nature_ev"] = nature_ev_term
                if hp_type_term is not None:       aux_probe_terms["hp_type"] = hp_type_term
                if win_prob_term is not None:      aux_probe_terms["win_prob"] = win_prob_term
                if pubval_term is not None:        aux_probe_terms["pubval"] = pubval_term
                if value_dist_term is not None:    aux_probe_terms["value_dist"] = value_dist_term
                if searchteacher_term is not None: aux_probe_terms["searchteacher"] = searchteacher_term
                if opd_term is not None:           aux_probe_terms["opd"] = opd_term
                aux_on = belief_aux_on or move_belief_on or latent_belief_on or move_latent_on
                # The belief terms only materialize on a minibatch with scored (believed = HIDDEN) slots;
                # wait for one so their shares aren't silently dropped from the single per-train() sample.
                # spread_belief scores on REVEALED slots (near-always present) so it does NOT gate this —
                # it rides whichever minibatch the probe samples (incl. the first, for a spread-only run).
                belief_present = any(
                    k in aux_probe_terms for k in ("species_belief", "move_belief", "latent", "move_latent")
                )

                # +INSTRUMENTATION: sample the shared-trunk gradient balance on the first
                # minibatch (graph alive here; the probe uses read-only autograd.grad with
                # retain_graph, so loss.backward() below is unaffected). Skipped when the
                # extractor exposes no shared-trunk params (non-Gen3 policy).
                # Sample once per train(). When an aux is ON, wait for a minibatch that actually HAS
                # scored slots (belief_present) so the per-aux shares aren't silently dropped for the
                # call; when off, sample on the first minibatch as before.
                if (shared_trunk and not grad_balance
                        and (not aux_on or belief_present)
                        and (not win_prob_on or win_prob_term is not None)):  # don't drop grad/win_prob_share
                    grad_balance = grad_balance_metrics(
                        policy_loss + self.ent_coef * entropy_loss,
                        # Phase B: the REAL critic term is the CE (value_dist_term); the scalar
                        # vf_coef·value_loss is dropped from the loss, so measure the CE instead.
                        (value_dist_term if (value_from_dist and value_dist_term is not None)
                         else self.vf_coef * value_loss),
                        shared_trunk,
                        # Each ACTIVE scaffold broken out on the trunk: species/move/latent/move-latent
                        # belief + win-prob (≈0 under read_only) + value-dist. Empty → RL-heads-only.
                        aux_terms=aux_probe_terms or None,
                    )

                # +INSTRUMENTATION: effective-rank of the trunk / value_cls / policy reps, sampled
                # ONCE per train() (first minibatch) via one no_grad forward — how many dims each
                # readout actually uses (rank_metrics.py). {} for a non-Gen3 extractor.
                if shared_trunk and not rank_metrics:
                    rank_metrics = rank_probe(
                        self.policy.features_extractor,
                        rollout_data.observations,
                        self.policy.extract_features,
                    )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    # +GRAD-ACCUM: discard the partial accumulation group — a true (batch_size·accum)
                    # batch checks KL over the whole effective batch and would discard it as one unit,
                    # mirroring stock's discard-the-current-minibatch on a KL trip.
                    self.policy.optimizer.zero_grad()
                    micro_in_group = 0
                    break

                # Optimization step. +GRAD-ACCUM: accumulate the 1/accum-scaled gradient (accum
                # micro-batches of size batch_size sum to the exact (batch_size·accum) gradient) and
                # step only when the group is full. accum==1 ⇒ one step per minibatch (upstream).
                (loss / accum).backward()
                micro_in_group += 1
                # +NOISE-SCALE: after the FIRST micro-batch of group 0 (epoch 0), .grad holds exactly
                # g_1/accum (this micro's gradient, scaled) → ‖g_1‖² = accum²·‖.grad‖². The single
                # micro-batch (B=batch_size) sample for the noise-scale estimate.
                if accum >= 2 and epoch == 0 and micro_in_group == 1 and noise_g_small_sq is None:
                    noise_g_small_sq = (accum ** 2) * self._global_grad_sq(self.policy.parameters())
                    if film_noise_params:
                        noise_film_small_sq = (accum ** 2) * self._global_grad_sq(film_noise_params)
                if micro_in_group == accum:
                    # +FILM-NOISE-SCALE: the film-group accumulated norm must be read BEFORE
                    # clip_grad_norm_ (which rescales grads IN PLACE when the global norm trips the
                    # clip — the global path is safe because clip returns the PRE-clip value).
                    if (accum >= 2 and epoch == 0 and noise_film_big_sq is None
                            and noise_film_small_sq is not None):
                        noise_film_big_sq = self._global_grad_sq(film_noise_params)
                    grad_norm = float(  # +INSTRUMENTATION: pre-clip total grad norm (per step)
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    )
                    grad_norms.append(grad_norm)
                    # +NOISE-SCALE: the accumulated group gradient (B=batch_size·accum) — pre-clip norm
                    # from clip_grad_norm_. Captured on group 0 (same data as the micro-batch above).
                    if accum >= 2 and epoch == 0 and noise_g_big_sq is None:
                        noise_g_big_sq = grad_norm * grad_norm
                    # +FILM-GRAD-ACCUM: gate the film group's grads (accumulate-or-apply) after the
                    # clip (baseline-identical clip semantics) and before the optimizer step.
                    if film_grad_accum is not None:
                        film_grad_accum.gate(_film_accum_k)
                    self.policy.optimizer.step()
                    self.policy.optimizer.zero_grad()
                    micro_in_group = 0

            # +GRAD-ACCUM: flush a trailing partial group (#minibatches not divisible by accum).
            # Rescale its accumulated grad from 1/accum to 1/micro_in_group so the short group's step
            # has the right magnitude. EXACT when its micro-batches are equal-size (the common case —
            # only the buffer's final minibatch can be smaller than batch_size); if that smaller
            # remainder lands in a group with full-size micro-batches it is weighted as if full-size,
            # a tiny bounded mis-weighting of one remainder per epoch (≈8e-5 on params in a toy probe,
            # negligible vs a 100k-sample rollout, and no worse than stock SB3's full-weight step on the
            # same remainder minibatch). ZERO when batch_size divides the rollout AND accum divides the
            # minibatch count → every group is `accum` equal-size micro-batches and the gradient is
            # bit-exact (verified: instrumented_ppo_test.test_grad_accum_matches_full_batch).
            if micro_in_group > 0:
                if micro_in_group < accum:
                    _rescale = accum / micro_in_group
                    for _p in self.policy.parameters():
                        if _p.grad is not None:
                            _p.grad.mul_(_rescale)
                grad_norms.append(float(
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                ))
                # +FILM-GRAD-ACCUM: same gate at the trailing-group step site.
                if film_grad_accum is not None:
                    film_grad_accum.gate(_film_accum_k)
                self.policy.optimizer.step()
                self.policy.optimizer.zero_grad()
                micro_in_group = 0

            self._n_updates += 1
            if not continue_training:
                break
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        # gen3_defensive_entropy_v1: did the boost fire, and is entropy actually higher on flagged decisions?
        if defent_flag_fracs:
            self.logger.record("defent/flagged_frac", float(np.mean(defent_flag_fracs)))
            self.logger.record("defent/boost_eff", float(np.mean(defent_boost_eff)))
            if defent_ent_flagged:
                self.logger.record("defent/entropy_flagged", float(np.mean(defent_ent_flagged)))
            if defent_ent_unflagged:
                self.logger.record("defent/entropy_unflagged", float(np.mean(defent_ent_unflagged)))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
            # +INSTRUMENTATION: average fraction of value updates that hit the clip bound
            if vf_clip_fractions:
                self.logger.record("train/clip_fraction_vf", float(np.mean(vf_clip_fractions)))

        # +INSTRUMENTATION: gradient-balance + value-scale diagnostics. These prepare for
        # reducing vf_coef and adding return normalization (PopArt) — see grad_balance.py and
        # src/agents/training/CLAUDE.md. All ride the standard logger → TensorBoard + launcher TUI.
        for _key, _val in grad_balance.items():
            self.logger.record(_key, _val)
        for _key, _val in rank_metrics.items():   # rank/{trunk,value_cls,policy}_* effective-rank probe
            self.logger.record(_key, _val)
        for _key, _val in value_scale_metrics(
            self.rollout_buffer.returns, self.rollout_buffer.values
        ).items():
            self.logger.record(_key, _val)
        if grad_norms:
            self.logger.record("train/grad_norm", float(np.mean(grad_norms)))

        # +NOISE-SCALE: fold this call's two-batch-size sample into the EMAs and log the smoothed
        # McCandlish 'simple' gradient noise scale B_simple = tr(Σ)/|G|² — the critical batch size.
        # Read it against your EFFECTIVE batch (batch_size·accum): `train/noise_scale_ratio` = B_simple /
        # effective; ≫1 ⇒ noise-limited (a bigger batch buys ~linear per-step progress), ≪1 ⇒
        # diminishing returns (could shrink for more update steps). Only when accumulating (needs two
        # batch sizes) AND both norms were captured (a full first group formed).
        _nsr_global = None   # +NSR-ADVISOR: this call's smoothed ratios (None until EMAs positive)
        _nsr_film = None
        if accum >= 2 and noise_g_small_sq is not None and noise_g_big_sq is not None:
            b_small = float(self.batch_size)
            b_big = b_small * accum
            tr_sigma, g2 = self._noise_scale_estimate(noise_g_small_sq, noise_g_big_sq, b_small, b_big)
            d = _NOISE_SCALE_EMA_DECAY
            self._noise_ema_s = tr_sigma if self._noise_ema_s is None else d * self._noise_ema_s + (1 - d) * tr_sigma
            self._noise_ema_g2 = g2 if self._noise_ema_g2 is None else d * self._noise_ema_g2 + (1 - d) * g2
            if self._noise_ema_g2 > 1e-12 and self._noise_ema_s > 0.0:
                b_simple = self._noise_ema_s / self._noise_ema_g2
                self.logger.record("train/noise_scale", float(b_simple))
                self.logger.record("train/noise_scale_ratio", float(b_simple / b_big))
                _nsr_global = float(b_simple / b_big)
        # +FILM-NOISE-SCALE: the FiLM-generator-group critical batch, same EMA treatment. Compare
        # `film/noise_scale` to `train/noise_scale` AND the effective batch: film ≫ global means the
        # conditioning gradient is far more noise-dominated than the aggregate — the direct
        # sample-starvation read for the per-team routing.
        if accum >= 2 and noise_film_small_sq is not None and noise_film_big_sq is not None:
            b_small = float(self.batch_size)
            b_big = b_small * accum
            tr_f, g2_f = self._noise_scale_estimate(noise_film_small_sq, noise_film_big_sq, b_small, b_big)
            d = _NOISE_SCALE_EMA_DECAY
            self._noise_film_ema_s = tr_f if self._noise_film_ema_s is None else d * self._noise_film_ema_s + (1 - d) * tr_f
            self._noise_film_ema_g2 = g2_f if self._noise_film_ema_g2 is None else d * self._noise_film_ema_g2 + (1 - d) * g2_f
            if self._noise_film_ema_g2 > 1e-12 and self._noise_film_ema_s > 0.0:
                b_film = self._noise_film_ema_s / self._noise_film_ema_g2
                self.logger.record("film/noise_scale", float(b_film))
                self.logger.record("film/noise_scale_ratio", float(b_film / b_big))
                _nsr_film = float(b_film / b_big)
                # The APPLIED-update view: each film update the optimizer actually takes is computed
                # from film_grad_accum_steps× the effective batch, so this is the ratio a FiLM step
                # really experiences — readable without the accumulation factor in your head.
                # ≈1 = at the group's critical batch; >2 = raise --film-grad-accum-steps.
                self.logger.record("film/noise_scale_ratio_applied",
                                   float(b_film / b_big) / max(1, int(getattr(self, "film_grad_accum_steps", 1) or 1)))
        # +NSR-ADVISOR: rate-limited TUI Events warnings when a smoothed noise-scale ratio is out
        # of band, with the concrete fix in the message (see _noise_scale_advice). Only on the
        # accumulating path (the estimator needs two batch sizes).
        if accum >= 2 and (_nsr_global is not None or _nsr_film is not None):
            self._emit_noise_scale_warnings(_nsr_global, _nsr_film, float(self.batch_size) * accum)

        # +BELIEF: hidden-opponent belief-aux diagnostics under their OWN `belief/` TB prefix (NOT
        # `train/`, which is crowded — matches the dedicated `grad/`/`popart/`/`win_prob/`/`eval/`
        # groups). Only when the aux is on AND some minibatch had believed slots. `species_acc` is the
        # headline: top-1 accuracy of predicting a hidden mon's species — rises as the model learns to
        # anticipate the un-revealed party.
        if belief_metrics:
            for _bk, _bvals in belief_metrics.items():
                self.logger.record(f"belief/{_bk}", float(np.mean(_bvals)))

        # +WIN-PROB: auxiliary win-probability diagnostics under their OWN `win_prob/` TB prefix (NOT
        # `train/`, which is crowded — matches the dedicated `grad/`/`popart/`/`eval/` groups). Only when
        # the head is on AND some minibatch had a known label. Calibration: `acc` (top-1 win/loss) +
        # `brier` (lower = P(win) tracks the win rate); `pred_mean` vs `label_mean` watches a base-rate
        # collapse; `coverage` = fraction with a known label. INFORMATION VALUE (the aggregate hides it —
        # blowouts are trivial): `brier_contested`/`acc_contested` on CLOSE games (|margin|<τ; judge vs the
        # ~0.25 no-skill floor of a 50/50 game), `contested_frac`/`contested_label_mean`, and
        # `skill_vs_material` (Brier skill vs a material-only baseline — >0 ⇒ beats counting mons). The
        # shared-trunk pull rides `grad/win_prob_share` (≈0 under read_only; real under shaping).
        if win_prob_metrics:
            for _wk, _wvals in win_prob_metrics.items():
                self.logger.record(f"win_prob/{_wk}", float(np.mean(_wvals)))

        # +PUBVAL: public-value aux diagnostics under their OWN `pubval/` TB prefix (gen3_pubval_aux_v1).
        # `mae` is the fit signal (|sigmoid − V_pub| → 0 as the trunk learns the human public value —
        # the raw BCE floors at the soft target's own entropy, so watch mae not bce); `pred_mean` vs
        # `target_mean` watches a collapse-to-base-rate. The shared-trunk pull rides `grad/pubval_share`
        # (≈0 under read_only; real under shaping — the credit-assignment experiment's lever).
        if pubval_metrics:
            for _pk, _pvals in pubval_metrics.items():
                self.logger.record(f"pubval/{_pk}", float(np.mean(_pvals)))

        # +VALUE-DIST: distributional value head diagnostics under their OWN `value_dist/` TB prefix (the
        # interpretability head's aggregate health, complementing the prober's per-decision histogram).
        # `entropy`/`std` fall as the critic sharpens; `pit_mean` ≈ 0.5 ⟺ calibrated; `mean_abs_err` =
        # |E[Z] − return| in support units. Ride the generic logger → TensorBoard + launcher TUI.
        if value_dist_metrics:
            for _vk, _vvals in value_dist_metrics.items():
                self.logger.record(f"value_dist/{_vk}", float(np.mean(_vvals)))

        # +ZARCH (gen3_zarch_film_v1): z_arch diagnostics under their OWN `zarch/` TB prefix.
        # `recon_topk_acc` → 1 as z carries the roster; `std` is the collapse monitor (→0 = collapsed
        # — the NO-GO signal); the `film/*` generator norms are the deviation-from-identity read
        # ("is FiLM alive or dead" — 0 at init, grows as conditioning is used; per-head so the
        # policy-vs-value routing is separately attributable).
        if zarch_metrics:
            for _zk, _zvals in zarch_metrics.items():
                self.logger.record(f"zarch/{_zk}", float(np.mean(_zvals)))
        _zfe = self.policy.features_extractor
        if getattr(_zfe, "zarch_film", "off") != "off":
            with th.no_grad():
                _z = _zfe.last_zarch                      # last minibatch's z [B, zdim] (many teams)
                # The LIVE LUT-vs-style dial: participation ratio of the minibatch z cloud (see
                # _zarch_participation_ratio). Watch the TREND — drifting toward zarch_dim =
                # LUT-ward identity spread; falling = style compression; →1 = collapse (pair
                # with zarch/std, the VICReg floor monitor).
                _pr = self._zarch_participation_ratio(_z)
                if _pr is not None:
                    self.logger.record("zarch/pr", _pr)
                # +LUT (gen3_zarch_lut_v1, v46): the GIGO CANARY. The per-team code is keyed by a
                # species⊕moves signature computed from the OBSERVATION, so a signature that fails to
                # match sends the decision to row 0 (unconditioned) — silently turning the whole
                # experiment into a no-op. On a --trainee-teams run this MUST sit at ~1.0; anything
                # lower means the lookup is broken, not that the LUT "didn't help".
                _lut_idx = getattr(_zfe, "last_zarch_lut_idx", None)
                if _lut_idx is not None:
                    self.logger.record("zarch/lut_hit_frac", float((_lut_idx > 0).float().mean()))
                    self.logger.record("zarch/lut_teams_seen",
                                       float(th.unique(_lut_idx[_lut_idx > 0]).numel()))
                    # Per-team code SPREAD — the LUT's analogue of film/*_team_std: the mean pairwise
                    # cosine DISTANCE between the learned rows. Random-init starts near 1.0
                    # (~orthogonal, the intended large-ε geometry); collapsing toward 0 would mean the
                    # codes merged and the conditioning went back to one shared direction.
                    _W = _zfe.zarch_lut_emb.weight[1:]
                    # Guard the ZERO-INIT case: normalizing all-zero rows yields cos=0, which would
                    # print code_dist=1.0 (MAXIMUM spread) for codes that have no spread at all —
                    # an actively misleading reading at exactly the moment we would be watching them
                    # grow. Report the raw norm alongside so "0 = not grown yet" is unambiguous.
                    self.logger.record("zarch/lut_code_norm", float(_W.norm(dim=1).mean()))
                    if _W.shape[0] > 1 and float(_W.norm(dim=1).mean()) > 1e-6:
                        _Wn = _W / (_W.norm(dim=1, keepdim=True) + 1e-8)
                        _cos = _Wn @ _Wn.T
                        _off = ~th.eye(_Wn.shape[0], dtype=th.bool, device=_Wn.device)
                        self.logger.record("zarch/lut_code_dist",
                                           float(1.0 - _cos[_off].mean().item()))
                for _side, _gen in (("pi", _zfe.film_pi), ("vf", _zfe.film_vf)):
                    _P = _gen.out_features // 2
                    self.logger.record(f"film/{_side}_gamma_norm",
                                       float(_gen.weight[:_P].norm().item()))
                    self.logger.record(f"film/{_side}_beta_norm",
                                       float(_gen.weight[_P:].norm().item()))
                    # GENERIC vs CONDITIONING split: the generators can grow by exploiting the SHARED
                    # component of z (a team-generic scale/shift = free capacity, not routing) while
                    # the per-team DIFFERENTIAL stays weak — the norms alone can't tell. `*_dev` =
                    # mean |modulation| (deviation-from-identity, generic + differential); `*_team_std`
                    # = per-dim std of the modulation ACROSS the minibatch's teams (the true
                    # conditioning signal — ≈0 while dev grows = the lazy generic mode; distillation
                    # pressure is the lever that sharpens it).
                    if _z is not None and _z.shape[0] > 1:
                        _mod = _gen(_z)                   # [B, 2P] = [Δγ ‖ Δβ]
                        self.logger.record(f"film/{_side}_dev",
                                           float(_mod.abs().mean().item()))
                        self.logger.record(f"film/{_side}_team_std",
                                           float(_mod.std(dim=0).mean().item()))

        # +SEARCH-TEACHER: AWR diagnostics under their OWN `teacher/` TB prefix. `agree_rate` (policy ↔
        # A* — should RISE as the distillation lands), `mean_adv` (the confirmed win-rate improvement of
        # the corrections), `mean_w` (AWR weight), `ce`, `loss`, `n`; `buffer_size` = the standalone ring
        # depth. The shared-trunk pull rides `grad/searchteacher_share` (+ `_policy_cosine` — the live
        # "is the teacher fighting the actor" signal). `teacher/yield` + `/corrections_per_cycle` are
        # emitted by SearchTeacherCallback (cross-process facts). Empty (off / empty buffer) → not logged.
        if teacher_metrics:
            for _tk, _tvals in teacher_metrics.items():
                self.logger.record(f"teacher/{_tk}", float(np.mean(_tvals)))
            cb = getattr(self, "_correction_buffer", None)
            if cb is not None:
                self.logger.record("teacher/buffer_size", float(len(cb)))

        # +OPD: on-policy self-distillation KL diagnostics under their OWN `opd/` TB prefix. `kl` = the
        # forward KL(π' ‖ π_student) being minimized (should FALL as the student matches π'),
        # `pi_target_entropy` = π' sharpness (low = decisive target), `agree_rate` = student ↔ π' mode
        # agreement (should RISE), `n` = the sampled correction count. The shared-trunk pull rides
        # `grad/opd_share`. Empty (off / empty buffer / an AWR-only π'-less sample) → not logged.
        if opd_metrics:
            for _ok, _ovals in opd_metrics.items():
                self.logger.record(f"opd/{_ok}", float(np.mean(_ovals)))

        # +DISTILL: exploiter-distillation KL diagnostics under their OWN `distill/` TB prefix. `kl` = the
        # masked forward KL(π_teacher ‖ π_student) being minimized (should FALL as the student matches the
        # specialist), `agree_rate` = student ↔ teacher mode agreement on teacher-team states (should RISE),
        # `coverage` = fraction of the minibatch on the teacher's team, `n` = teacher-team state count.
        # Empty (off / no teacher-team states in any minibatch) → not logged.
        if distill_metrics:
            for _dk, _dvals in distill_metrics.items():
                self.logger.record(f"distill/{_dk}", float(np.mean(_dvals)))

        # +PopArt diagnostics: mu/sigma should TRACK train/return_mean/return_std (the running
        # normalizer estimate); value_weight_norm watches the POP rescale stay bounded (an explosion
        # signals a degenerate sigma / broken preservation). With PopArt on, train/value_loss is the
        # NORMALIZED loss (≈O(1)) and grad/value_policy_logratio should fall toward ~0 (the
        # aux-independent value/policy balance; grad/value_share also drops but moves with the aux count).
        if popart is not None:
            self.logger.record("popart/mu", float(self.policy.popart.mu))
            self.logger.record("popart/sigma", float(self.policy.popart.sigma))
            self.logger.record("popart/value_weight_norm", float(self.policy.value_net.weight.norm()))
        # gen3_no_concat_v1 (v61): the multi-seed critic readout's collapse monitors — the TB
        # contract in agents/model/seed_diagnostics.py (query/output cosine, UNCENTERED effective
        # rank, the VICReg variance target), logged every train() so a collapsing seed set is
        # visible from step 0 (the z_arch post-hoc-discovery failure, never again). The
        # pre-registered VICReg trigger lives in that module's docstring — decide from the plot.
        _sr = getattr(getattr(self.policy.features_extractor, "assembler", None), "seed_readout", None)
        if _sr is not None and _sr.last_outputs is not None:
            from agents.model.seed_diagnostics import seed_collapse_diagnostics
            for k, v in seed_collapse_diagnostics(_sr.queries.detach(),
                                                  _sr.last_outputs.detach()).items():
                self.logger.record(k, v)
        # +SEED-VICReg terms (only when the regularizer is active): watch vicreg_var_term fall as
        # seeds/out_effective_rank rises toward k — the un-collapse confirmation the gen-6 enable
        # is judged by.
        for _sk, _svals in seed_vicreg_metrics.items():
            self.logger.record(_sk, float(np.mean(_svals)))
