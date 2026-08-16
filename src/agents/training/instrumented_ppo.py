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


from torch.nn import functional as F

from sb3_contrib import MaskablePPO

from agents.training.async_vec_env import AsyncSubprocVecEnv, collect_rollouts_async
from agents.training.grad_balance import (
    edge_family_metrics,
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

# MOVE-latent VICReg variance floor (the belief-latent leg that also used it is DELETED, v75): a
# hinge `relu(_LATENT_STD_TARGET - std)` per latent dim pushes the predicted latents to stay spread
# (≈unit std), the belt-and-braces collapse guard on top of the stop-grad + task-anchored target.
# Weighted by _LATENT_VICREG_WEIGHT inside the move-latent loss. The `movelatent_std` metric (mean
# per-dim std) is the NO-GO monitor: std→0 while cosine→1 is collapse.
# (moved to belief_bank with the latent loss; re-exported below for old imports)

# Gradient-noise-scale EMA decay (McCandlish et al. 2018, "An Empirical Model of Large-Batch
# Training"). The single-step estimates of |G|² (true-gradient squared norm) and tr(Σ) (per-example
# gradient-variance trace) are noisy; their RATIO B_simple = tr(Σ)/|G|² is unstable per step, so we
# EMA the numerator and denominator SEPARATELY (this constant) and divide the smoothed values. 0.99
# ≈ a few-hundred-train()-call window — long enough to denoise, short enough to track drift.
_NOISE_SCALE_EMA_DECAY = 0.99

# The supervised belief-head losses + their scale constants live in `belief_bank` (the
# declarative fold of design_unified_belief.md §4 — one ROW per head instead of one inline
# vertical per head). Re-exported here because tests and older call sites import them from
# this module.
from agents.training.belief_bank import (   # noqa: F401  (re-exports)
    _EV_LOSS_SCALE, _NATURE_CE_WEIGHT, _EV_LOSS_WEIGHT, _SPREAD_LOSS_SCALE,
    _LATENT_STD_TARGET, _LATENT_VICREG_WEIGHT,
)
from agents.training import belief_bank as _belief_bank


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
    item_belief_coef: float = 0.0

    # Set by train_rl_agent (gen3_unified_move_system_v1). The MOVE-belief LATENT-grading weight:
    # move_belief_latent_coef * (cosine of the predicted move distribution's expected move-latent toward
    # the true moveset's mean latent + VICReg floor) over the revealed scored slots. 0.0 = OFF
    # (byte-identical loss). Soft complement to the per-ID BCE so near-moves (Rock Slide ≈ HP Rock) grade
    # as near. Training-only (scales the loss, never a forward pass) → NOT version-locked; it READS the
    # extractor's last_move_latent_table, whose state_dict-changing module is gated by the version-checked
    # move_latent arch toggle.
    move_belief_latent_coef: float = 0.0


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

    opp_intent_coef: float = 0.0
    # SET-VALUED partial credit on beta's belief-miss rows (see `set_valued_switch_loss`). Scales
    # ON TOP of opp_intent_coef, so it is a share of the intent budget rather than a second one.
    # 0.0 = OFF and the loss is byte-identical; training-only, resume-mutable (no module changes).
    beta_setvalued_coef: float = 0.0

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
        return super()._excluded_save_params() + ["_correction_buffer", "_distill_teacher",
                                                  "_distill_teachers"]

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

    @staticmethod
    def _noise_scale_advice(global_ratio, b_eff):
        """PURE advisory logic for the noise-scale ratios → list of (key, warning) pairs; [] when
        healthy. The TUI-warning half of the McCandlish instrumentation: a ratio ≫ 1 means updates
        are noise-dominated (each step's direction is mostly sideways — and under Adam the noise
        still moves params at full speed, so spurious content gets WRITTEN, not just slowed); ≪ 1
        means samples are being spent polishing an already-clean gradient instead of taking more
        steps. Each warning names the concrete fix."""
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
        return out

    def _emit_noise_scale_warnings(self, global_ratio, b_eff):
        """Rate-limited (30 min per key) Events-panel emit of _noise_scale_advice, after an EMA
        warm-up (first ~20 samples are settling and would false-alarm)."""
        import time
        self._nsr_samples += 1
        if self._nsr_samples < 20:
            return
        if self._nsr_warn_last is None:
            self._nsr_warn_last = {}
        advice = self._noise_scale_advice(global_ratio, b_eff)
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

    # ALL FIVE supervised belief losses now live in `belief_bank` (the declarative fold);
    # these aliases keep every existing call site and test resolving unchanged.
    _move_belief_loss = staticmethod(_belief_bank.move_belief_loss)
    # The three revealed-slot supervised losses MOVED to `belief_bank` (the declarative fold);
    # these aliases keep every existing call site and test resolving unchanged.
    _spread_belief_loss = staticmethod(_belief_bank.spread_belief_loss)
    _nature_ev_belief_loss = staticmethod(_belief_bank.nature_ev_belief_loss)
    _hp_type_belief_loss = staticmethod(_belief_bank.hp_type_belief_loss)

    _move_belief_latent_loss = staticmethod(_belief_bank.move_belief_latent_loss)

    _belief_aux_loss = staticmethod(_belief_bank.belief_aux_loss)

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
        # +OPPONENT INTENT (gen3_opp_intent_v1): ALIGN the labels to the predictions, ONCE, BEFORE
        # `get()` flattens and shuffles. The env emits, at buffer row i, what the opponent did at
        # decision i-1 (their turn-t action is only observable while building the obs for t+1), so
        # row i's own label sits at row i+1 of the same env column. Shifting here — while the
        # [n_steps, n_envs] structure and `episode_starts` still exist — is the only place the
        # episode-boundary drop is even expressible; after the shuffle the adjacency is gone.
        # Idempotent per rollout: collect_rollouts refills these keys every time.
        if getattr(self, "opp_intent_coef", 0.0) > 0.0:
            _obs_buf = getattr(self.rollout_buffer, "observations", None)
            if isinstance(_obs_buf, dict) and "opp_action_kind" in _obs_buf:
                from agents.training.opp_intent_labels import (KIND_UNKNOWN, SWITCH_SLOT_NONE,
                                                               align_labels_to_predictions)
                _starts = self.rollout_buffer.episode_starts
                # EVERY one-ahead intent key must be shifted, including `opp_switch_species`.
                # It was omitted originally, so beta's CONTENT-ADDRESSED target read the species of
                # decision t-1 against the kind/slot of decision t. That is not merely wrong, it is
                # INVISIBLE: on most rows the stale species is 0 -> resolve_believed_slot_by_content
                # returns INTENT_IGNORE and the path silently no-ops, which reads exactly like the
                # documented "the belief is too cold to clear the floor" case below. Two consecutive
                # switch-ins is the one shape where it resolves — to the PREVIOUS switch-in's slot.
                for _k, _fill in (("opp_action_kind", KIND_UNKNOWN), ("opp_action_num", 0),
                                  ("opp_switch_slot", SWITCH_SLOT_NONE),
                                  ("opp_switch_species", 0),
                                  # `opp_class` is CONSTANT within an episode, so the shift is a
                                  # semantic no-op — included anyway so every intent label is
                                  # row-aligned by the same rule. A reader should never have to
                                  # remember which of these keys was shifted and which was not;
                                  # that asymmetry is what produced the bug documented above.
                                  ("opp_class", 0)):
                    _obs_buf[_k] = align_labels_to_predictions(_obs_buf[_k], _starts, _fill)

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
        # Shared sink for the per-minibatch aux diagnostics that already carry their OWN full TB
        # key (`value_seeds/*` from the seed-collapse contract, `opp_intent/*`), so they are recorded
        # verbatim rather than under a prefix.
        aux_metrics: dict[str, list[float]] = {}
        distill_metrics: dict[str, list[float]] = {}     # +DISTILL: exploiter-distillation KL diagnostics
        value_dist_metrics: dict[str, list[float]] = {}  # +VALUE-DIST: per-minibatch HL-Gauss diagnostics
        # Compute once: the aux path is fully skipped when off → loss stays byte-identical to upstream.
        belief_aux_on = self.opp_belief_aux_coef > 0.0
        move_belief_on = self.move_belief_coef > 0.0  # +MOVE-BELIEF reinjection-head supervised loss
        move_latent_on = self.move_belief_latent_coef > 0.0  # +MOVE-LATENT grading (gen3_unified_move_system_v1)
        spread_belief_on = self.spread_belief_coef > 0.0  # +SPREAD-belief supervision (gen3_unified_spread_belief_v1)
        hp_type_belief_on = self.hp_type_belief_coef > 0.0  # +HP-TYPE belief CE (gen3_opp_hp_type_belief_v1)
        item_belief_on = self.item_belief_coef > 0.0  # +ITEM belief CE (gen3_item_belief_v1)
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
        edge_metrics: dict[str, float] = {}  # edge/<fam>_{weight,grad}_norm — per-family liveness
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
                # +BELIEF BANK site "hidden_move": the hidden-team Hungarian aux
                # (masked to the BELIEVED slots, folded at opp_belief_aux_coef — metrics
                # UNPREFIXED with the historic `aux_loss` key) and the move-belief BCE
                # (revealed direct + unrevealed order-invariant, folded at move_belief_coef,
                # `move_` prefix). Same two blocks, one loop — rows in belief_bank.ROWS.
                belief_aux_term = None  # the WEIGHTED aux contribution, for the grad-balance probe
                move_belief_term = None
                for _brow, _bterm, _bm in _belief_bank.compute(
                        self.policy.features_extractor, rollout_data.observations,
                        coefs={"opp_belief_aux_coef": self.opp_belief_aux_coef,
                               "move_belief_coef": self.move_belief_coef},
                        gates={"hidden_team": belief_aux_on, "move_belief": move_belief_on},
                        site="hidden_move",
                        params={"moves_weight": self.opp_belief_moves_weight}):
                    loss = loss + _bterm
                    if _brow.name == "hidden_team":
                        belief_aux_term = _bterm
                    elif _brow.name == "move_belief":
                        move_belief_term = _bterm
                    for _bk, _bv in _bm.items():
                        belief_metrics.setdefault(_brow.prefix + _bk, []).append(float(_bv))

                # +OPPONENT INTENT (gen3_opp_intent_v1): supervise ALPHA/BETA against what the
                # opponent actually did. The label lives in the obs, one row AHEAD of the prediction
                # (the env can only see their turn-t action while building the obs for t+1), so the
                # buffer's label block was shifted back by one — and pairs spanning an episode
                # boundary DROPPED — before `get()` shuffled it. See `align_labels_to_predictions`.
                # Initialised HERE, not beside the other `*_term` locals ~370 lines below:
                # the intent block runs FIRST in this minibatch, so a later `= None` would wipe it
                # and the gradient probe would silently see no intent term.
                opp_intent_term = None
                if self.opp_intent_coef > 0.0:
                    # gen3_belief_label_only_v1: alpha's LIVE logits — `last_alpha_logits` is the
                    # stop-grad publication under label_only (it feeds the critic under
                    # --intent-value-reduce), so the intent loss must read the supervision view.
                    _al = self.policy.features_extractor.belief_supervision("alpha_logits")
                    _bl = getattr(self.policy.features_extractor, "last_beta_logits", None)
                    _sn = getattr(self.policy.features_extractor, "last_alpha_seat_nums", None)
                    _obs = rollout_data.observations
                    if _al is not None and _sn is not None and "opp_action_kind" in _obs:
                        from agents.model.opp_intent import (INTENT_IGNORE, OPP_CLASS_NAMES,
                                                             intent_losses,
                                                             match_seats_to_move_num,
                                                             switch_coverage_metrics)
                        _kind = _obs["opp_action_kind"].long().flatten()
                        _num = _obs["opp_action_num"].long().flatten()
                        _atgt = match_seats_to_move_num(_sn, _num, _kind, _sn.shape[-1])
                        _btgt = _obs["opp_switch_slot"].long().flatten()
                        # beta learns ONLY from genuine voluntary switches; every other row is masked
                        # at the label builder, and a switch we cannot address is masked here.
                        _btgt = th.where(_kind == 1, _btgt,
                                         th.full_like(_btgt, INTENT_IGNORE))
                        # CONTENT-ADDRESSED believed-slot resolution. A switch-in that was still
                        # HIDDEN at the decision has no valid slot INDEX: the believed slots are
                        # anonymous DETR queries the species loss re-matches by Hungarian
                        # assignment, so the label's Pokedex-sorted canonicalisation names a slot
                        # whose learned content is a different mon. Ask the model's OWN species
                        # posterior instead — "which believed slot do you think holds this mon" —
                        # so beta and the species head refer to the same object. Masked on belief
                        # miss, exactly as alpha masks on seat miss.
                        _sp = _obs.get("opp_switch_species")
                        _bel = getattr(self.policy.features_extractor,
                                       "last_opp_believed_mask", None)
                        _blog = getattr(self.policy.features_extractor,
                                        "last_belief_logits", None)
                        if _sp is not None and _bel is not None and _blog is not None \
                                and "species" in _blog:
                            from agents.model.opp_intent import (resolve_believed_slot_by_content,
                                                                set_valued_switch_loss)
                            _content = resolve_believed_slot_by_content(
                                _blog["species"].detach(), _bel.float(),
                                _sp.long().flatten())
                            # Prefer the EXACT revealed slot; fall back to the content-addressed
                            # believed slot only where the label had none.
                            _need = (_kind == 1) & (_btgt < 0)
                            _btgt = th.where(_need, _content, _btgt)
                            oi_extra_believed = float(
                                ((_need) & (_content >= 0)).float().sum())
                            # SEPARATE the two failure modes. `wanted` counts rows that ASKED for
                            # content-addressing (a switch to a mon with no revealed slot);
                            # `believed_targets` counts rows it RESOLVED. wanted=0 => the label
                            # never emits SWITCH_SLOT_NONE (plumbing); wanted>0 with resolved=0 =>
                            # the belief is too cold to clear the floor (expected early, and the
                            # reason a cold smoke cannot validate this path).
                            oi_wanted_content = float(_need.float().sum())
                        else:
                            oi_extra_believed = 0.0
                            oi_wanted_content = 0.0
                            # Defined on BOTH paths: the set-valued term below reads them, and a
                            # name that exists on only one branch is a NameError waiting for the
                            # first run whose belief head is off.
                            _content = None
                            _need = None
                        # beta v1 supervises only switch-ins the head could actually POINT AT.
                        # The label's slot is resolved on the board at t+1; the logits come from the
                        # board at t. A mon UNREVEALED at t has no addressable slot there, so its
                        # target lands on a -inf logit => +inf loss (measured: beta_loss=inf).
                        # Dropping those rows IS design_opponent_intent.md §4.3's stated v1 scope
                        # (revealed slots only; ~46% of switches masked, rate logged). B1 is the
                        # named upgrade that turns the mask into a posterior soft-target.
                        if _bl is not None:
                            _safe = _btgt.clamp(min=0, max=_bl.shape[-1] - 1)
                            _reach = th.isfinite(_bl.detach().gather(1, _safe[:, None]).squeeze(1))
                            _btgt = th.where(_reach, _btgt, th.full_like(_btgt, INTENT_IGNORE))
                        # SET-VALUED partial credit for a switch to a mon we did not believe.
                        # These rows are the ones `_content` could not name, so today they are
                        # dropped entirely — yet they carry a true fact (`they brought someone
                        # UNSEEN`) that beta should be graded on. Off (coef 0.0) leaves the loss
                        # byte-identical.
                        _sv, oi_m_extra_rows = None, 0.0
                        if self.beta_setvalued_coef > 0.0 and _bl is not None \
                                and _bel is not None and _need is not None \
                                and _content is not None:
                            _miss = _need & (_content < 0)
                            _sv = set_valued_switch_loss(_bl, _bel.float(), _miss)
                            oi_m_extra_rows = float(_miss.float().sum())
                        _ocls = _obs.get("opp_class")
                        oi_loss, oi_m = intent_losses(
                            _al, _atgt, _bl, _btgt,
                            opp_class=(_ocls.long() if _ocls is not None else None))
                        # THE number that says whether content-addressing recovered anything.
                        # Without it, a no-op looks identical to a working feature (the same
                        # blindness that let a zero-supervision alpha pass a green smoke).
                        oi_m["opp_intent/beta_believed_targets"] = oi_extra_believed
                        oi_m["opp_intent/beta_wanted_content"] = oi_wanted_content
                        # SPLIT `beta_mask_rate`, which conflates two failures with opposite
                        # meanings. Its denominator is every row, so it is dominated by "this
                        # decision was not a switch at all" — expected, uninteresting, and roughly
                        # constant. Buried inside it is the one a reader actually wants: of the
                        # switches that NEEDED the belief, how often was the belief too cold to
                        # name the mon? That is the BELIEF's failure, not beta's, and it must stay
                        # attributable to the belief. (Measured on gen-9: 356 resolved of 390
                        # wanted => 0.087. Recoverable from the two counters above, but nobody
                        # computes a ratio off a dashboard, so a rate nobody reports is a rate
                        # nobody reads.) EMITTED BY `_switch_coverage` below, which owns the
                        # want/got counters for the pooled read and every opponent slice alike.
                        # THE SWITCH-COVERAGE MATRIX. Every voluntary switch falls in exactly one of
                        # three buckets, and only the third is a failure — but with just a mask rate
                        # and a miss rate a reader cannot tell their SIZES, and "beta is masked 73%
                        # of the time" reads as a crisis when ~62 of those points are simply "they
                        # attacked". These are fractions of VOLUNTARY SWITCHES, so they sum to 1.
                        #
                        #   revealed      the mon was already on the board -> exact slot, no belief
                        #                 needed. The easiest label, and previously invisible.
                        #   hidden_found  still hidden, and the species posterior placed it -> the
                        #                 content-addressed target. This is what that path BUYS.
                        #   hidden_missed the belief could not name it -> masked. The BELIEF's
                        #                 failure, and the only bucket that is lost supervision.
                        oi_m.update(switch_coverage_metrics(_kind, _need, _content))
                        if _ocls is not None:
                            _ocf = _ocls.long().reshape(-1)
                            for _code, _name in OPP_CLASS_NAMES.items():
                                _rows = _ocf == _code
                                if int(_rows.sum()) < 2:
                                    continue
                                oi_m.update(switch_coverage_metrics(
                                    _kind, _need, _content, _rows, f"_{_name}"))
                        if _sv is not None:
                            loss = loss + self.opp_intent_coef * self.beta_setvalued_coef * _sv
                            oi_m["opp_intent/beta_setvalued_loss"] = float(_sv.detach())
                            oi_m["opp_intent/beta_setvalued_rows"] = oi_m_extra_rows
                        opp_intent_term = self.opp_intent_coef * oi_loss
                        loss = loss + opp_intent_term
                        for _ok, _ov in oi_m.items():
                            aux_metrics.setdefault(_ok, []).append(_ov)

                # +MOVE-LATENT (gen3_unified_move_system_v1): grade the move belief in latent space so
                # near-moves (Rock Slide ≈ HP Rock) grade as near — the soft complement to the per-ID BCE.
                # Reads the extractor's context-free move-latent table (stashed this minibatch) + the same
                # known_moves labels. Its gradient flows into the move-belief head AND the MoveLatentEncoder
                # (the table) → it joins the aux pull on the trunk. OFF → skipped (byte-identical).
                # +BELIEF BANK site "latent": the move-latent grading (soft complement to the
                # per-ID BCE — near-moves grade as near; `movelatent_` prefix).
                move_latent_term = None
                for _brow, _bterm, _bm in _belief_bank.compute(
                        self.policy.features_extractor, rollout_data.observations,
                        coefs={"move_belief_latent_coef": self.move_belief_latent_coef},
                        gates={"move_latent": move_latent_on}, site="latent"):
                    loss = loss + _bterm
                    move_latent_term = _bterm
                    for _bk, _bv in _bm.items():
                        belief_metrics.setdefault(_brow.prefix + _bk, []).append(float(_bv))

                # +BELIEF BANK (design_unified_belief.md §4, the code-shape fold): the three
                # revealed-slot supervised heads — SPREAD (gen3_unified_spread_belief_v1),
                # NATURE/EV (gen3_nature_ev_belief_v1, folded at the SAME spread coef — one knob
                # supervises the whole spread belief), HP-TYPE (gen3_opp_hp_type_belief_v1) —
                # folded by ONE loop over `belief_bank.ROWS`. Registry order == the old inline
                # block order, so `loss = loss + term` accumulates bit-identically; each head's
                # per-row docstring (stash keys, labels, leak-safety) lives in belief_bank.
                # A sixth supervised belief is now a ROW there, not another inline vertical.
                spread_belief_term = None
                nature_ev_term = None
                hp_type_term = None
                item_belief_term = None
                for _brow, _bterm, _bm in _belief_bank.compute(
                        self.policy.features_extractor, rollout_data.observations,
                        coefs={"spread_belief_coef": self.spread_belief_coef,
                               "hp_type_belief_coef": self.hp_type_belief_coef,
                               "item_belief_coef": self.item_belief_coef},
                        gates={"spread": spread_belief_on, "hp_type": hp_type_belief_on,
                               "item": item_belief_on},
                        site="revealed"):
                    loss = loss + _bterm
                    if _brow.name == "spread":
                        spread_belief_term = _bterm
                    elif _brow.name == "nature_ev":
                        nature_ev_term = _bterm
                    elif _brow.name == "hp_type":
                        hp_type_term = _bterm
                    elif _brow.name == "item":
                        item_belief_term = _bterm
                    for _bk, _bv in _bm.items():
                        belief_metrics.setdefault(_brow.prefix + _bk, []).append(float(_bv))

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
                if move_latent_term is not None:   aux_probe_terms["move_latent"] = move_latent_term
                if spread_belief_term is not None: aux_probe_terms["spread_belief"] = spread_belief_term
                if nature_ev_term is not None:     aux_probe_terms["nature_ev"] = nature_ev_term
                if hp_type_term is not None:       aux_probe_terms["hp_type"] = hp_type_term
                if item_belief_term is not None:   aux_probe_terms["item_belief"] = item_belief_term
                if win_prob_term is not None:      aux_probe_terms["win_prob"] = win_prob_term
                if pubval_term is not None:        aux_probe_terms["pubval"] = pubval_term
                if value_dist_term is not None:    aux_probe_terms["value_dist"] = value_dist_term
                if searchteacher_term is not None: aux_probe_terms["searchteacher"] = searchteacher_term
                # THE FIGHT DETECTOR. Registering the intent term here is what produces
                # `grad/opp_intent_policy_cosine` — the angle between the intent objective's pull on
                # the shared trunk and the policy's. Under `--opp-intent-grad-mode detached` the
                # intent gradient cannot reach the trunk at all and this reads ~0 BY CONSTRUCTION,
                # which is the correct and expected value, not a bug. It only becomes informative
                # under `shaping`, which is precisely when you need to know.
                if opp_intent_term is not None:    aux_probe_terms["opp_intent"] = opp_intent_term
                if opd_term is not None:           aux_probe_terms["opd"] = opd_term
                aux_on = belief_aux_on or move_belief_on or move_latent_on
                # The belief terms only materialize on a minibatch with scored (believed = HIDDEN) slots;
                # wait for one so their shares aren't silently dropped from the single per-train() sample.
                # spread_belief scores on REVEALED slots (near-always present) so it does NOT gate this —
                # it rides whichever minibatch the probe samples (incl. the first, for a spread-only run).
                belief_present = any(
                    k in aux_probe_terms for k in ("species_belief", "move_belief", "move_latent")
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
                        # Each ACTIVE scaffold broken out on the trunk: species/move/move-latent
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
                # +INSTRUMENTATION: per-edge-family liveness, sampled ONCE per train() and read
                # HERE because it wants `.grad` populated but not yet cleared by the optimizer
                # step. Parameters only — no forward touched, so the hot path pays nothing.
                if not edge_metrics:
                    edge_metrics = edge_family_metrics(self.policy.features_extractor)
                # +NOISE-SCALE: after the FIRST micro-batch of group 0 (epoch 0), .grad holds exactly
                # g_1/accum (this micro's gradient, scaled) → ‖g_1‖² = accum²·‖.grad‖². The single
                # micro-batch (B=batch_size) sample for the noise-scale estimate.
                if accum >= 2 and epoch == 0 and micro_in_group == 1 and noise_g_small_sq is None:
                    noise_g_small_sq = (accum ** 2) * self._global_grad_sq(self.policy.parameters())
                if micro_in_group == accum:
                    grad_norm = float(  # +INSTRUMENTATION: pre-clip total grad norm (per step)
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    )
                    grad_norms.append(grad_norm)
                    # +NOISE-SCALE: the accumulated group gradient (B=batch_size·accum) — pre-clip norm
                    # from clip_grad_norm_. Captured on group 0 (same data as the micro-batch above).
                    if accum >= 2 and epoch == 0 and noise_g_big_sq is None:
                        noise_g_big_sq = grad_norm * grad_norm
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
        for _key, _val in edge_metrics.items():   # edge/<fam>_{weight,grad}_norm — is each family ALIVE?
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
        # +NSR-ADVISOR: rate-limited TUI Events warnings when a smoothed noise-scale ratio is out
        # of band, with the concrete fix in the message (see _noise_scale_advice). Only on the
        # accumulating path (the estimator needs two batch sizes).
        if accum >= 2 and _nsr_global is not None:
            self._emit_noise_scale_warnings(_nsr_global, float(self.batch_size) * accum)

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
        for _sk, _svals in aux_metrics.items():
            self.logger.record(_sk, float(np.mean(_svals)))
