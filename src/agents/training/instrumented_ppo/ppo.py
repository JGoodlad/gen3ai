"""`InstrumentedMaskablePPO` — the class, and `train()`: the whole loss fold in ONE module.

⚠️ **`train()` is not split, and that is the design.** It is a vendored copy of upstream
`sb3_contrib.MaskablePPO.train` (hash-pinned in the hub) with our terms folded in, and the ORDER
in which those terms are folded is a CONTRACT — see `train()`'s own docstring for the numbered
version. Splitting the sequence across modules would make an ordering that is currently
straight-line source order into something a reader has to reassemble, and the one property that
matters about it (no flag combination reorders these) would stop being visible.

Everything that is NOT the sequence has moved out: the knobs (`hparams`), the per-term losses
(`distill_terms`, `value_terms`, `aux_terms`), the noise-scale machinery (`noise_scale`).
"""
import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance

from agents.training.async_vec_env import AsyncSubprocVecEnv, collect_rollouts_async
from agents.training.grad_balance import (
    cell_family_metrics,
    edge_family_metrics,
    grad_balance_metrics,
    shared_trunk_parameters,
    value_scale_metrics,
)
from agents.training import belief_bank as _belief_bank
from agents.training.instrumented_ppo.aux_terms import AuxTerms
from agents.training.instrumented_ppo.capacity_terms import CapacityTerms
from agents.training.instrumented_ppo.constants import _NOISE_SCALE_EMA_DECAY
from agents.training.instrumented_ppo.distill_terms import DistillTerms
from agents.training.instrumented_ppo.hparams import PpoHyperparameters
from agents.training.instrumented_ppo.noise_scale import NoiseScaleDiagnostics
from agents.training.instrumented_ppo.signal_metrics import advantage_density_metrics
from agents.training.instrumented_ppo.value_terms import ValueTerms
from agents.training.rank_metrics import rank_probe
from agents.training.scaffolding import live_gauge_metrics


class InstrumentedMaskablePPO(PpoHyperparameters,
                              NoiseScaleDiagnostics,
                              DistillTerms,
                              ValueTerms,
                              AuxTerms,
                              CapacityTerms,
                              MaskablePPO):
    """MaskablePPO with `train/clip_fraction_vf` instrumentation added.

    Behaviour-identical to `MaskablePPO` except for the additional TensorBoard
    metric. See module docstring for drift-detection details.

    Also dispatches rollout collection to the **non-barrier async collector** when
    ``self._async_rollout`` is set and the env is an ``AsyncSubprocVecEnv`` (``--async-rollout``);
    otherwise it is the unchanged stock ``MaskablePPO.collect_rollouts``.
    """

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
        if self._async_rollout and isinstance(env, AsyncSubprocVecEnv):
            ok = collect_rollouts_async(
                self, env, callback, rollout_buffer, n_rollout_steps, use_masking)
        else:
            ok = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps, use_masking)
        # +WIN-PROB PBRS (ai_v12 route 1, gen3_winprob_pbrs_v1): add coef·(γ·φ(s′) − φ(s)) to this
        # rollout's rewards and RE-RUN GAE, with φ = the DETACHED win-prob head. It has to happen HERE
        # — after collection, before train() — because both collectors compute GAE as their last act
        # and PopArt reads `returns` at the top of train(), so this is the one window in which the
        # shaping can land in RAW reward space and still reach the advantages. Env workers hold no
        # model, so the reward cannot be shaped where it is produced. Covers BOTH collectors
        # identically (see winprob_pbrs.py on why the φ read is a batched re-forward). At coef 0 —
        # the default — not even the import runs, so an OFF run is byte-identical.
        if ok and float(getattr(self, "win_prob_pbrs_coef", 0.0) or 0.0) != 0.0:
            from agents.training.winprob_pbrs import apply_winprob_pbrs
            self._pbrs_metrics = apply_winprob_pbrs(self, rollout_buffer)
        return ok

    def _annealed_entropy_boost(self, B: float, af: float) -> float:
        """The state-conditioned entropy-boost multiplier at the CURRENT step. Constant `B` if the anneal
        fraction is 0; else linearly annealed toward 1.0, reaching 1.0 once `af` of training has elapsed
        (uses SB3's `_current_progress_remaining`, which runs 1.0 at the start → 0.0 at the end). Shared by
        the defensive (`gen3_defensive_entropy_v1`) and bait (`gen3_bait_entropy_v1`) boosts — ONE schedule,
        so the two flags can never drift apart. Pure → unit-testable."""
        B, af = float(B), float(af)
        if af <= 0.0 or B == 1.0:
            return B
        done = 1.0 - float(getattr(self, "_current_progress_remaining", 1.0))   # 0 → 1 over training
        return 1.0 + (B - 1.0) * max(0.0, 1.0 - done / af)

    def _defensive_entropy_boost_eff(self) -> float:
        """gen3_defensive_entropy_v1: this run's defensive boost at the current step."""
        return InstrumentedMaskablePPO._annealed_entropy_boost(
            self, self.defensive_entropy_boost, self.defensive_entropy_anneal_frac)

    def _bait_entropy_boost_eff(self) -> float:
        """gen3_bait_entropy_v1: this run's bait boost at the current step (same schedule)."""
        return InstrumentedMaskablePPO._annealed_entropy_boost(
            self, self.bait_entropy_boost, self.bait_entropy_anneal_frac)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.

        Vendored from `sb3_contrib.MaskablePPO.train` (hash pinned in
        `_EXPECTED_UPSTREAM_TRAIN_HASH`). The only deltas vs upstream are
        marked with `# +INSTRUMENTATION` comments.

        ------------------------------------------------------------------------------------
        THE FOLD ORDER IS A CONTRACT, and it is STRAIGHT-LINE SOURCE ORDER
        ------------------------------------------------------------------------------------
        Every `loss = loss + <term>` below runs in the order it is written, unconditionally —
        **no flag combination reorders them.** Each term is guarded by its own `if <x>_on:`,
        and a term that is off contributes nothing rather than moving anyone else. That is why
        this method is NOT split across modules even though it is ~1,250 lines: the ordering is
        checkable by reading only while it is one straight line, and the numbered contract below
        would otherwise have to be reassembled from six files.

        The sequence, per minibatch:

          1. `loss = pg_term + ent_coef * ent_loss_used + vf_term`   (the upstream PPO loss;
             `pg_term` is the UNSCALED `policy_loss` tensor at `policy_grad_coef == 1.0` — the default,
             byte-identical to upstream — else `policy_grad_coef * policy_loss` (`--policy-grad-coef`; 0.0 removes
             the policy-gradient term alone, the arm-F pure-distill/aux phase — entropy and the
             value term keep their own coefficients). `_value_loss_from_se` is the only other
             delta, and at `value_tail_weight == 0` it is `F.mse_loss` byte-for-byte)
          2. the BELIEF bank — species/moves aux, opponent-intent (+ the set-valued beta term),
             move belief, spread belief, nature/EV, HP-type, item belief, move-latent
          3. the WIN-PROB BCE, then the CF-TWIN on-policy mirror
          4. the VALUE-DIST HL-Gauss CE
          5. the DISTILL family — the policy term (full KL, or the top-K/action-CE form with the
             optional advantage gate under `--distill-target action` — gen3_distill_target_gate_v1),
             value MSE, the FitNets value-feature hint
          6. SEARCH-TEACHER AWR, then OPD
          7. TD-AUX (the Bellman-residual consistency term)
          8. the COUNTERFACTUAL block — cf-winprob, cf-evidential, cf-twin, cf-shadow

        **Why 7 and 8 are LAST, and in that order.** Steps 2-4 read the extractor STASHES that
        this minibatch's `evaluate_actions` forward left behind (`last_win_prob_logits`,
        `last_spread_belief`, …). Steps 7 and 8 each run their OWN extractor forward, which
        CLOBBERS those stashes. So every stash-reading term must be folded before them, and the
        CF block — which additionally samples foreign recorded states off disk — goes after
        `_td_aux_term` for the same reason. Moving a stash-reading fold below step 7 does not
        crash: it silently scores the wrong states. `instrumented_ppo_hub_contract_test.py`
        pins the 7-before-8 half of this by reading the source.

        The steps AFTER the loop (the grad-accum flush, the noise-scale fold, and the ~260 lines
        of `self.logger.record`) are diagnostics and carry no gradient.

        **CAPACITY TELEMETRY is NOT a fold step, and is placed to make that unarguable.**
        `--capacity-telemetry` (gen3_capacity_telemetry_v1) runs entirely AFTER the optimizer step
        for the minibatch, so it appears nowhere in the sequence above and no `loss = loss + …`
        line belongs to it. Its three probes carry no gradient into the policy by construction —
        the canary owns its own optimizer over its own params on a detached input, the half-batch
        cosine reads gradients with `autograd.grad` (which never writes `.grad`), and the velocity
        probe is `no_grad`. The one thing it needs from inside the fold is a SNAPSHOT of this
        minibatch's `value_pooled`, taken right after `evaluate_actions` for the same reason steps
        2-4 sit where they do: the own-forward folds replace the stash.
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
        # gen3_bait_entropy_v1: the same four, for the bait-opportunity boost.
        baitent_flag_fracs, baitent_boost_eff, baitent_ent_flagged, baitent_ent_unflagged = [], [], [], []
        clip_fractions = []
        vf_clip_fractions: list[float] = []  # +INSTRUMENTATION
        belief_metrics: dict[str, list[float]] = {}  # +BELIEF: per-minibatch aux diagnostics (dict of lists)
        win_prob_metrics: dict[str, list[float]] = {}  # +WIN-PROB: per-minibatch diagnostics (dict of lists)
        # +SCAFFOLDING GAUGE: paired (V, win-prob logit) reads for `train/scaffolding_gauge`. Two
        # lists, filled ONLY during epoch 0 so the gauge describes ONE policy over the whole
        # rollout rather than mixing epochs (by epoch 3 the policy that produced the pair is not
        # the policy the pair is attributed to). Empty when the head is off → nothing published.
        scaffold_v: list[np.ndarray] = []
        scaffold_z: list[np.ndarray] = []
        teacher_metrics: dict[str, list[float]] = {}    # +SEARCH-TEACHER: AWR per-minibatch diagnostics
        opd_metrics: dict[str, list[float]] = {}         # +OPD: on-policy self-distillation KL diagnostics
        # Shared sink for the per-minibatch aux diagnostics that already carry their OWN full TB
        # key (`opp_intent/*`), so they are recorded verbatim rather than under a prefix.
        aux_metrics: dict[str, list[float]] = {}
        distill_metrics: dict[str, list[float]] = {}     # +DISTILL: exploiter-distillation KL diagnostics
        td_aux_metrics: dict[str, list[float]] = {}      # +TD-AUX: Bellman-residual diagnostics
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
        # +SCAFFOLDING GAUGE: gated on the HEAD's existence alone, NOT on `win_prob_coef` — the
        # gauge is an observability read of whatever the head currently says, and a `read_only`
        # head at coef 0 still says something worth curving. ALWAYS ON when the head exists;
        # there is no flag, matching the `signal/` group.
        scaffolding_on = getattr(self.policy.features_extractor, "win_prob_mode", "none") != "none"
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
        # +DISTILL-SHARE (gen3_grad_distill_share_v1): does THIS rollout hold any teacher-team rows
        # at all? Decides whether the grad-balance probe below WAITS for a minibatch with a live
        # distill term (so `grad/distill_share` — the §6.2 dose meter of
        # designs/ai_v10/design_advantage_gated_distillation.md — isn't silently dropped from the
        # per-train() sample) or samples immediately: an all-zero buffer would otherwise suppress
        # the WHOLE probe for the call (the cf starving-buffer lesson at the probe's gate). One
        # np.max over the buffer per train(); off (no teachers / coef 0) it short-circuits free.
        _buf_obs = getattr(self.rollout_buffer, "observations", None) if distill_on else None
        distill_rows_in_buffer = (
            distill_on and isinstance(_buf_obs, dict) and "distill_mask" in _buf_obs
            and float(np.max(_buf_obs["distill_mask"])) >= 1.0
        )
        # +PG-COEF (gen3_policy_grad_coef_v1, `--policy-grad-coef`): the PPO policy-gradient term's own weight.
        # 1.0 (default) takes the UNSCALED `policy_loss` tensor — the loss expression is then
        # byte-identical to upstream; 0.0 removes the policy-gradient contribution alone (the
        # arm-F pure-distill/aux phase). Scales ONLY `policy_loss` — entropy and the value term
        # keep their own coefficients (`ent_coef`, `vf_coef`).
        policy_grad_coef = float(getattr(self, "policy_grad_coef", 1.0))
        # +TD-AUX (gen3_td_consistency_aux_v1): the Bellman-residual consistency term over CONTIGUOUS
        # buffer pairs. 0.0 → the block is skipped entirely (no sampler, no extra forward, loss
        # byte-identical to today). See `_td_aux_term`.
        td_aux_on = float(getattr(self, "td_aux_coef", 0.0)) > 0.0
        # +CF-WINPROB (gen3_cf_label_plumbing_v1): the counterfactual MC win-prob grounding term.
        # On when the coef is non-zero, a label buffer is attached, AND the extractor actually has a
        # win-prob head to supervise (`--win-prob-mode` != none). 0.0 / no buffer / no head → the
        # block is skipped entirely: no disk poll, no sample, no forward, loss byte-identical.
        cf_buffer = getattr(self, "_cf_buffer", None)
        cf_winprob_on = (
            float(getattr(self, "cf_winprob_coef", 0.0)) != 0.0
            and cf_buffer is not None
            and getattr(self.policy.features_extractor, "win_head", None) is not None
        )
        # +CF-EVIDENTIAL (gen3_cf_evidential_head_v1): the Beta uncertainty readout, on the SAME
        # labels and the SAME forward. Independent of the scalar term — either, both or neither may
        # be live — but it needs the STRUCTURAL head (`--cf-evidential`), a launch-time decision.
        cf_evid_on = (
            float(getattr(self, "cf_evidential_coef", 0.0)) != 0.0
            and cf_buffer is not None
            and getattr(self.policy.features_extractor, "cf_evid_head", None) is not None
        )
        # +CF-TWIN (gen3_cf_twin_heads_v1): the twin win-prob heads B/C — the within-run paired
        # comparison. Needs the STRUCTURAL heads (`--cf-twin-heads`) AND a live coefficient; at
        # coefficient 0 the whole block is skipped INCLUDING the on-policy mirror, so a built-but-
        # unused pair leaves every parameter update byte-identical to not building them.
        cf_twin_on = (
            float(getattr(self, "cf_twin_coef", 0.0)) != 0.0
            and cf_buffer is not None
            and getattr(self.policy.features_extractor, "cf_twin_head_b", None) is not None
        )
        # +CF-SHADOW (gen3_cf_twin_heads_v1): the passive value twin on `mc_return` labels.
        cf_shadow_on = (
            float(getattr(self, "cf_shadow_coef", 0.0)) != 0.0
            and cf_buffer is not None
            and getattr(self.policy.features_extractor, "cf_shadow_head", None) is not None
        )
        cf_any_on = cf_winprob_on or cf_evid_on or cf_twin_on or cf_shadow_on
        if cf_any_on:
            # ONE disk poll per train() (= per rollout), not per minibatch: the producer writes at
            # its own pace and re-globbing a directory 240 times an update buys nothing.
            cf_buffer.poll(int(self.num_timesteps))
        cf_metrics: dict[str, list[float]] = {}
        cf_evid_metrics: dict[str, list[float]] = {}
        cf_twin_metrics: dict[str, list[float]] = {}     # +CF-TWIN (gen3_cf_twin_heads_v1)
        cf_shadow_metrics: dict[str, list[float]] = {}   # +CF-SHADOW (gen3_cf_twin_heads_v1)
        cf_rows_sampled = 0

        continue_training = True

        # +INSTRUMENTATION: gradient-balance + value-scale diagnostics (grad_balance.py).
        # The dual-head extractor shares one trunk; both losses' gradients compete there. We
        # sample that pull ONCE per train() call (first minibatch) so vf_coef / return
        # normalization (PopArt) can be tuned to a number rather than inferred from KL.
        shared_trunk = shared_trunk_parameters(self.policy.features_extractor)
        grad_balance: dict[str, float] = {}
        rank_metrics: dict[str, float] = {}  # effective rank of trunk / value_cls / policy reps (once/train)
        edge_metrics: dict[str, float] = {}  # edge/<fam>_{weight,grad}_norm — per-family liveness
        cell_metrics: dict[str, float] = {}  # cell/<name>_{weight,grad}_norm — per-CELL liveness
        grad_norms: list[float] = []  # pre-clip total grad norm (shows grad-clip activity)

        # +CAPACITY TELEMETRY (gen3_capacity_telemetry_v1): the plasticity canary, the half-batch
        # trunk-gradient cosine and the fixed-probe feature velocity. `None` when the flag is off,
        # and OFF is the whole cost — no head, no optimizer, no projection matrix, no probe batch,
        # and no extra forward or backward anywhere below. See `capacity_telemetry.py`.
        capacity = self._capacity()
        capacity_metrics: dict[str, float] = {}

        # +PopArt: advance the value-target normalizer once per train() (before the epochs) from
        # this rollout's returns; update() also POP-rescales value_net so its de-normalized outputs
        # are preserved. The value loss below then trains in normalized space. No-op when disabled.
        popart = getattr(self.policy, "popart", None)
        if popart is not None:
            popart.update(
                th.as_tensor(self.rollout_buffer.returns, device=self.device), self.policy.value_net
            )

        # +SIGNAL (gen3_signal_rate_metrics_v1): ADVANTAGE DENSITY — how much action-attributable
        # learning signal this rollout carries. Read ONCE per train() off the buffer's RAW GAE
        # advantages, HERE, because this is the last point at which they still exist unmodified:
        # the minibatch loop below applies `normalize_advantage`, which forces std→1 per minibatch
        # and so erases the very quantity being measured. Read-only numpy over the buffer — no
        # torch, no RNG, no gradient path, and the advantages PPO fits are untouched.
        # ⚠️ UNITS: these ride the run's PopArt-normalized returns, whose σ moves over training, so
        # `adv_raw_std`/`adv_raw_abs_mean` compare WITHIN a run and only cautiously across runs
        # (`adv_kurtosis` is scale-free and does compare). Must be read WITH `signal/outcome_entropy`
        # — see signal_metrics.py's module docstring for the mirror paradox and the 2x2 reading.
        signal_metrics = advantage_density_metrics(self.rollout_buffer.advantages)

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

                # +CAPACITY: snapshot THIS forward's `value_pooled` before the TD-aux / CF folds
                # replace the stash. Detached in the snapshot itself, so nothing downstream can
                # accidentally hand the canary a live graph.
                cap_features = (
                    self._capacity_snapshot_features(self.policy.features_extractor,
                                                     int(values.shape[0]))
                    if capacity is not None else None)

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
                # gen3_bait_entropy_v1 adds the SECOND such flag (`bait_opportunity`) on the same mechanism.
                # The two weights MULTIPLY: each is 1 off its own flag, so one boost alone is byte-identical
                # to running it alone, and a decision flagged by both gets the product (they are near-disjoint
                # in practice — "a heal is legal" vs "our attack is dead into their bench").
                ent_per = -log_prob if entropy is None else entropy          # [B] per-decision entropy (nats)
                entropy_loss = -th.mean(ent_per)                             # standard (unweighted) metric
                entropy_losses.append(entropy_loss.item())
                ent_weight = None                                            # None ⇒ nothing on ⇒ unweighted
                do_flag = rollout_data.observations.get("defensive_opportunity")
                if self.defensive_entropy_boost != 1.0 and do_flag is not None:
                    b_eff = self._defensive_entropy_boost_eff()
                    flag = do_flag.to(ent_per.device).reshape(-1).float()    # [B] 1.0 on defensive decisions
                    ent_weight = 1.0 + (b_eff - 1.0) * flag
                    with th.no_grad():
                        fm = flag > 0.5
                        defent_flag_fracs.append(float(fm.float().mean().item()))
                        defent_boost_eff.append(b_eff)
                        if bool(fm.any()):    defent_ent_flagged.append(float(ent_per[fm].mean().item()))
                        if bool((~fm).any()): defent_ent_unflagged.append(float(ent_per[~fm].mean().item()))
                bait_flag = rollout_data.observations.get("bait_opportunity")
                if self.bait_entropy_boost != 1.0 and bait_flag is not None:
                    bb_eff = self._bait_entropy_boost_eff()
                    bflag = bait_flag.to(ent_per.device).reshape(-1).float()  # [B] 1.0 on bait-opportunity rows
                    bw = 1.0 + (bb_eff - 1.0) * bflag
                    ent_weight = bw if ent_weight is None else ent_weight * bw
                    with th.no_grad():
                        bm = bflag > 0.5
                        baitent_flag_fracs.append(float(bm.float().mean().item()))
                        baitent_boost_eff.append(bb_eff)
                        if bool(bm.any()):    baitent_ent_flagged.append(float(ent_per[bm].mean().item()))
                        if bool((~bm).any()): baitent_ent_unflagged.append(float(ent_per[~bm].mean().item()))
                ent_loss_used = entropy_loss if ent_weight is None else -th.mean(ent_weight * ent_per)

                # Phase B (value_from_dist): the scalar MSE value term is DROPPED (value_net frozen —
                # the CE below at vf_coef is the critic). value_loss is still logged as the
                # E[Z]-mean-vs-return diagnostic. Off → the standard vf_coef·MSE term.
                _vf_term = 0.0 if value_from_dist else self.vf_coef * value_loss
                # +PG-COEF: at 1.0 (the default) `_policy_grad_term` IS the `policy_loss` tensor, so the
                # line below is literally the old `loss = policy_loss + …` expression —
                # byte-identical. Any other value scales ONLY the policy-gradient term.
                _policy_grad_term = policy_loss if policy_grad_coef == 1.0 else policy_grad_coef * policy_loss
                loss = _policy_grad_term + self.ent_coef * ent_loss_used + _vf_term

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
                    # beta is published since gen3_intent_conditional_v1 (the boom cell reads
                    # it forward-side), so its CE must read the supervision view too — the
                    # attribute is the stop-grad publication under label_only.
                    _bl = self.policy.features_extractor.belief_supervision("beta_logits")
                    _sn = self.policy.features_extractor.last_alpha_seat_nums
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
                        # gen3_intent_label_bot_weight_v1: `--intent-label-bot-weight` discounts the
                        # α/β labels produced against a heuristic BOT (bots play strategies that are
                        # not the meta, and the self-play ramp makes early supervision bot-dominated).
                        # It lands HERE and NOWHERE ELSE. The BeliefBank rows below — species, move,
                        # item, spread, nature/EV, HP-type — are TEAM truth: what their team IS holds
                        # regardless of who pilots it, so weighting those by opponent class would
                        # discard valid labels. Only INTENT is behaviour. Default 1.0 takes the
                        # original unweighted `cross_entropy` call, bit-identical.
                        oi_loss, oi_m = intent_losses(
                            _al, _atgt, _bl, _btgt,
                            opp_class=(_ocls.long() if _ocls is not None else None),
                            bot_label_weight=float(
                                getattr(self, "intent_label_bot_weight", 1.0)))
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

                # +SCAFFOLDING GAUGE (registered 2026-08-29): the two value readouts this tree
                # carries answer DIFFERENT questions — the critic estimates the SHAPED return (in
                # PopArt units, discounted), the win-prob head estimates the GAME. Their divergence
                # is the reward scaffolding still doing work, and its trajectory is the registered
                # signal for when shaping coefficients can begin annealing toward the pure game.
                # Read here because this is the one place both readouts exist for the SAME states
                # from the SAME forward: `evaluate_actions` above produced `values` and stashed
                # `last_win_prob_logits`.
                # 🚨 RANK FORM ONLY. V is a PopArt-normalized shaped return, so there is no unit
                # conversion to a probability; the live path additionally has no realized outcome
                # labels for these states, so the calibrated-affine gauge is OFFLINE by
                # construction (`python -m main.scaffolding_gauge`). The logit is NOT sigmoided —
                # the sigmoid is monotone, so the rank correlation is identical and float32 ranks
                # never saturate. Read-only: detached clones, no gradient path, no RNG.
                if scaffolding_on and epoch == 0:
                    _wz = getattr(self.policy.features_extractor, "last_win_prob_logits", None)
                    if _wz is not None:
                        scaffold_v.append(values.detach().reshape(-1).cpu().numpy())
                        scaffold_z.append(_wz.detach().reshape(-1).cpu().numpy())

                # +CF-TWIN, half one of two (gen3_cf_twin_heads_v1): head A's OWN loss, mirrored
                # onto twins B and C on THIS minibatch. It must run HERE, beside A's fold and
                # BEFORE the cf block below clobbers the extractor stashes with its own forward —
                # the twins read the same `value_pooled` A read, which is the entire premise of
                # "identical trunk, identical states". Weighted at `win_prob_coef` (A's own), so
                # all three heads carry a bit-identical control objective; gated on `cf_twin_coef`
                # so coefficient zero is byte-identical.
                cf_twin_op_term = None
                if cf_twin_on:
                    cf_twin_op_term, _ctm = self._cf_twin_onpolicy_terms(rollout_data)
                    if cf_twin_op_term is not None:
                        loss = loss + cf_twin_op_term
                        for _ck, _cv in _ctm.items():
                            cf_twin_metrics.setdefault(_ck, []).append(float(_cv))

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
                        # +DISTILL TARGET FORM (gen3_distill_target_gate_v1,
                        # design_advantage_gated_distillation.md §3.1/§3.3): WHAT the policy term
                        # asks for. "kl" (the default) takes the literal `_distill_loss` call below
                        # — byte-identical to every run before the flag existed. "action" dispatches
                        # to `_gated_action_distill_loss` (teacher top-K renormalized target, K=1 =
                        # argmax CE, AWR-weighted by |Â|), optionally row-gated on the student's OWN
                        # normalized advantage (`--distill-gate advantage`: teacher disagrees AND
                        # Â < -τ). `advantages`/`actions` are the very tensors the clip objective
                        # uses, so τ is in clip-objective units. Everything else — the teacher
                        # forwards, the per-teacher balancing, every value-side term — is untouched.
                        _d_target = str(getattr(self, "distill_target", "kl"))
                        _gate_n = _gate_agree = _gate_adv = 0.0   # §4.3 liveness, summed over teachers
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
                            if _d_target == "kl":
                                _d_out = self._distill_loss(_s_logits, _t_logits, rollout_data.action_masks, _sel)
                            else:
                                _d_out = self._gated_action_distill_loss(
                                    _s_logits, _t_logits, rollout_data.action_masks, _sel,
                                    advantages, actions,
                                    top_k=int(getattr(self, "distill_topk", 1)),
                                    tau=float(getattr(self, "distill_gate_tau", 0.0)),
                                    beta=float(getattr(self, "distill_beta", 1.0)),
                                    gate=str(getattr(self, "distill_gate", "none")))
                            if _d_out is not None:
                                _kl_k, _m_k = _d_out
                                _per_teacher_kl.append(_kl_k)
                                if _d_target != "kl":
                                    _gate_n += _m_k["n_gated"]
                                    _gate_agree += _m_k["gate_agree_rate"] * _m_k["n_gated"]
                                    _gate_adv += _m_k["mean_gate_adv"] * _m_k["n_gated"]
                                for _mk, _mv in _m_k.items():   # per-teacher diagnostics (distill/t{k}_*)
                                    distill_metrics.setdefault(f"t{_k}_{_mk}", []).append(float(_mv))
                            if _vfd_on:
                                # Masked cosine distance between the student + teacher value-CLS pools on
                                # teacher-k's states (the FitNets hint match).
                                _vfd_k = self._value_feat_distill(_s_vfeat, _t_vfeat, _sel)
                                if _vfd_k is not None:
                                    _per_teacher_vfd.append(_vfd_k)
                                    # NAMING (read this before quoting the number): the recorded value is the
                                    # cosine DISTANCE `1 − cos`, i.e. the loss term — it FALLS toward 0 as the
                                    # student and teacher hints align, so a reading of 0.005 means cos ≈ 0.995
                                    # (near-PARALLEL), not near-orthogonal. `*_value_feat_dist` is the canonical
                                    # key; `*_value_feat_cos` is the historical spelling, which reads as its own
                                    # opposite and is kept ONE release for TensorBoard continuity.
                                    for _vfd_key in (f"t{_k}_value_feat_dist", f"t{_k}_value_feat_cos"):
                                        distill_metrics.setdefault(_vfd_key, []).append(float(_vfd_k))
                            if _vd_on:
                                # Teacher V (real-unit, frozen); masked MSE vs student V in the PopArt frame.
                                with th.no_grad():
                                    _t_val = _teacher.policy.predict_values(_t_obs).flatten()
                                _vd_k = self._value_distill_mse(_s_val, _t_val, _sel, popart)
                                if _vd_k is not None:
                                    _per_teacher_vd.append(_vd_k)
                                    distill_metrics.setdefault(f"t{_k}_value_mse", []).append(float(_vd_k))
                        if _d_target != "kl":
                            # +GATE LIVENESS (§4.3): the aggregate-across-teachers row for THIS
                            # minibatch. `n_gated == 0` is a READING — the gate found nothing to
                            # teach here — not an absence; the rate metrics are gated on n>0
                            # because a 0/0 agree-rate would be a fabricated perfect score.
                            _B_rows = float(_tid_flat.shape[0])
                            distill_metrics.setdefault("n_gated", []).append(_gate_n)
                            distill_metrics.setdefault("gated_frac", []).append(
                                _gate_n / max(_B_rows, 1.0))
                            if _gate_n > 0:
                                distill_metrics.setdefault("gate_agree_rate", []).append(
                                    _gate_agree / _gate_n)
                                distill_metrics.setdefault("mean_gate_adv", []).append(
                                    _gate_adv / _gate_n)
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
                            # Same naming note as the per-teacher site above: DISTANCE (1 − cos), lower =
                            # better aligned. `value_feat_dist` is canonical; `value_feat_cos` is the
                            # deprecated alias kept one release.
                            for _vfd_key in ("value_feat_dist", "value_feat_cos"):
                                distill_metrics.setdefault(_vfd_key, []).append(float(_distill_vfd))

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

                # +TD-AUX: the TD-consistency auxiliary. Its OWN contiguous sample + its OWN critic
                # forward (the minibatch is shuffled — it holds no adjacent pairs), so it must run
                # AFTER every loss that reads an extractor stash from THIS minibatch's
                # evaluate_actions forward: the forward below replaces those stashes. Placed here,
                # beside the other own-forward folds (search-teacher / OPD), for exactly that reason.
                # `rank_probe` further below re-forwards `rollout_data.observations` itself, so it is
                # unaffected. OFF → skipped (loss byte-identical).
                td_aux_term = None
                if td_aux_on:
                    td_aux_term, _tdm = self._td_aux_term(popart)
                    if td_aux_term is not None:
                        loss = loss + td_aux_term
                        for _tdk, _tdv in _tdm.items():
                            td_aux_metrics.setdefault(_tdk, []).append(float(_tdv))

                # +CF-WINPROB: ground-truth Monte-Carlo P(win) supervision of the win-prob head on
                # OFF-DISTRIBUTION recorded states (its own sample + its own extractor forward, so
                # it belongs here beside td_aux/search-teacher/OPD — after every loss that reads a
                # stash from THIS minibatch's evaluate_actions forward, which its forward replaces).
                # OFF / empty buffer → skipped (loss byte-identical).
                cf_term = None
                cf_evid_term = None
                cf_twin_term = None
                cf_shadow_term = None
                if cf_any_on:
                    # ONE sample + ONE extractor forward, shared by both readouts (see
                    # `_cf_sample_and_forward`). With the evidential half off this is exactly the
                    # call the scalar term used to make on its own, which is what keeps the
                    # coefficient-zero byte-identity pins meaningful.
                    _cf_ctx = self._cf_sample_and_forward()
                    # Rows the fold actually CONSUMED this train(), summed over minibatches. Not a
                    # duplicate of `cf/buffer_fill` (residency) nor of `cf/n` (the per-fold mean):
                    # this is the only number that answers "how much label did this update eat",
                    # which is what a starving producer starves — a buffer of 40 rows sampled by 40
                    # minibatches still reports fill 40 while delivering 40x the same handful.
                    if _cf_ctx is not None:
                        cf_rows_sampled += int(_cf_ctx.n_rows)
                    if cf_winprob_on:
                        cf_term, _cfm = self._cf_winprob_term(_cf_ctx)
                        if cf_term is not None:
                            loss = loss + cf_term
                            for _cfk, _cfv in _cfm.items():
                                cf_metrics.setdefault(_cfk, []).append(float(_cfv))
                    if cf_evid_on:
                        cf_evid_term, _cfem = self._cf_evidential_term(_cf_ctx)
                        if cf_evid_term is not None:
                            loss = loss + cf_evid_term
                            for _cek, _cev in _cfem.items():
                                cf_evid_metrics.setdefault(_cek, []).append(float(_cev))
                    # +CF-TWIN, half two of two (gen3_cf_twin_heads_v1): the folds that make B and
                    # C DIFFER — the same states through the same shared forward, B on the recorded
                    # SINGLE OUTCOME and C on the TIGHT-MC label. Riding the shared sample is not an
                    # optimization here, it is the design: two samples would make the two arms
                    # disagree about which states they scored, and the paired difference would stop
                    # being paired.
                    if cf_twin_on:
                        cf_twin_term, _cftm = self._cf_twin_terms(_cf_ctx)
                        if cf_twin_term is not None:
                            loss = loss + cf_twin_term
                        for _ck, _cv in _cftm.items():
                            cf_twin_metrics.setdefault(_ck, []).append(float(_cv))
                    # +CF-SHADOW: the passive value twin on `mc_return`. Same sample, same forward.
                    if cf_shadow_on:
                        cf_shadow_term, _cfsm = self._cf_shadow_term(_cf_ctx, popart)
                        if cf_shadow_term is not None:
                            loss = loss + cf_shadow_term
                        for _sk, _sv in _cfsm.items():
                            cf_shadow_metrics.setdefault(_sk, []).append(float(_sv))

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
                if value_dist_term is not None:    aux_probe_terms["value_dist"] = value_dist_term
                if searchteacher_term is not None: aux_probe_terms["searchteacher"] = searchteacher_term
                # +DISTILL-SHARE (gen3_grad_distill_share_v1): the exploiter-distillation KL's own
                # shared-trunk pull — `grad/distill_share`, on the SAME policy+value+Σaux
                # denominator as every other `grad/*_share` (grad_balance.py), like
                # `grad/searchteacher_share` / `grad/opd_share`. THE dose meter §6.2 of
                # designs/ai_v10/design_advantage_gated_distillation.md dose-matches the G1/G2
                # arms on (gradient share, not coefficient). The POLICY KL term only,
                # deliberately: the value-side distill coefficients are held fixed across those
                # arms (§6.1), so folding them in would compress the very differences the meter
                # exists to read. None (distill off / no teacher-team rows this minibatch) → not
                # logged; a non-distill run pays nothing.
                if distill_term is not None:       aux_probe_terms["distill"] = distill_term
                # THE FIGHT DETECTOR. Registering the intent term here is what produces
                # `grad/opp_intent_policy_cosine` — the angle between the intent objective's pull on
                # the shared trunk and the policy's. Under `--opp-intent-grad-mode detached` the
                # intent gradient cannot reach the trunk at all and this reads ~0 BY CONSTRUCTION,
                # which is the correct and expected value, not a bug. It only becomes informative
                # under `shaping`, which is precisely when you need to know.
                if opp_intent_term is not None:    aux_probe_terms["opp_intent"] = opp_intent_term
                if opd_term is not None:           aux_probe_terms["opd"] = opd_term
                # The TD term pulls the trunk through the CRITIC path only, so `grad/td_aux_share`
                # against `grad/value_share` is the read for "is the consistency term crowding out
                # the level regression it is supposed to complement".
                if td_aux_term is not None:        aux_probe_terms["td_aux"] = td_aux_term
                # The CF term's trunk pull. Under `cf_head_only` (the default) its input is
                # stop-grad'd, so `grad/cf_winprob_share` reads exactly 0.0 BY CONSTRUCTION — that
                # is the correct value and the gate the head-only stage is verified by, not a bug.
                if cf_term is not None:            aux_probe_terms["cf_winprob"] = cf_term
                # The evidential term's input is detached UNCONDITIONALLY (no head_only switch), so
                # `grad/cf_evidential_share` reads exactly 0.0 BY CONSTRUCTION — it is registered
                # here precisely so that zero is PUBLISHED rather than assumed.
                if cf_evid_term is not None:       aux_probe_terms["cf_evidential"] = cf_evid_term
                # gen3_cf_twin_heads_v1: the twins and the shadow all read a DETACHED value_pooled
                # unconditionally, so `grad/cf_twin_share` and `grad/cf_shadow_share` read exactly
                # 0.0 BY CONSTRUCTION. Registered here for the evidential head's reason: the
                # head-only contract is the arm's single most load-bearing claim, and a published
                # zero is a live measurement of it where a docstring is not. (Both twin halves ride
                # ONE probe entry — the on-policy mirror and the cf fold pull the same two heads.)
                # `sum` rather than a length branch: the probe must not encode the arity, or a
                # third twin term would silently drop out of a scalar published precisely to make
                # the head-only contract a measurement instead of a docstring claim.
                _twin_terms = [t for t in (cf_twin_op_term, cf_twin_term) if t is not None]
                if _twin_terms:
                    aux_probe_terms["cf_twin"] = sum(_twin_terms[1:], _twin_terms[0])
                if cf_shadow_term is not None:     aux_probe_terms["cf_shadow"] = cf_shadow_term
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
                        and (not win_prob_on or win_prob_term is not None)   # don't drop grad/win_prob_share
                        # …nor grad/cf_winprob_share. A STARVING buffer yields a None term on every
                        # minibatch, so waiting for one would suppress the whole grad probe for the
                        # rest of the run — the `len(cf_buffer) == 0` escape says "there are no
                        # labels at all, sample anyway"; `cf/buffer_fill` is where that is read.
                        and (not cf_any_on or cf_term is not None or cf_evid_term is not None
                             or len(cf_buffer) == 0)
                        # +DISTILL-SHARE: wait for a minibatch with a live distill term so
                        # `grad/distill_share` isn't dropped from the per-train() sample — but
                        # ONLY when the rollout holds teacher-team rows at all
                        # (`distill_rows_in_buffer`); a row-less rollout samples immediately
                        # rather than suppressing the whole probe (the cf escape's reason).
                        and (not distill_rows_in_buffer or distill_term is not None)):
                    grad_balance = grad_balance_metrics(
                        # +PG-COEF: the probe measures the terms AS FOLDED — `_policy_grad_term`, not the
                        # raw `policy_loss` (at the 1.0 default they are the same tensor).
                        _policy_grad_term + self.ent_coef * entropy_loss,
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
                # +INSTRUMENTATION: the same read for the zero-init POINTER CELLS (switch-branch,
                # pair-outcome move/switch, conditional-threat). Same window, same reason: a cell
                # that never comes off its zero init is invisible without it.
                if not cell_metrics:
                    cell_metrics = cell_family_metrics(self.policy.features_extractor)
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

                # +CAPACITY TELEMETRY: LAST in the minibatch body, deliberately — after the loss
                # fold, after `loss.backward()`, after the optimizer step. Nothing it does can
                # reach `loss` or `.grad` from here, which is the point: the placement is the
                # proof. It costs one `if` per minibatch when the flag is off.
                if capacity is not None:
                    self._capacity_observe(capacity, rollout_data, actions, advantages,
                                           shared_trunk, clip_range, cap_features)

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

        # +CAPACITY TELEMETRY: the once-per-train() half — fold the per-minibatch canary/cosine
        # samples and (on cadence) run the frozen probe batch through the extractor for the
        # feature-velocity read. Outside the epoch loop, no gradient, `{}` when the flag is off.
        if capacity is not None:
            capacity_metrics = self._capacity_finish(capacity)

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
        # gen3_bait_entropy_v1: same four for the bait boost. `flagged_frac` is also the probe's EXPOSURE
        # reading — how much of the rollout is actually a bait board (a boost cannot work on states the
        # policy never reaches), so a flat behavioural result at a near-zero flagged_frac is a DOSE
        # finding, not a mechanism finding.
        if baitent_flag_fracs:
            self.logger.record("baitent/flagged_frac", float(np.mean(baitent_flag_fracs)))
            self.logger.record("baitent/boost_eff", float(np.mean(baitent_boost_eff)))
            if baitent_ent_flagged:
                self.logger.record("baitent/entropy_flagged", float(np.mean(baitent_ent_flagged)))
            if baitent_ent_unflagged:
                self.logger.record("baitent/entropy_unflagged", float(np.mean(baitent_ent_unflagged)))
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
        for _key, _val in cell_metrics.items():   # cell/<name>_{weight,grad}_norm — is each cell ALIVE?
            self.logger.record(_key, _val)
        for _key, _val in value_scale_metrics(
            self.rollout_buffer.returns, self.rollout_buffer.values
        ).items():
            self.logger.record(_key, _val)
        if grad_norms:
            self.logger.record("train/grad_norm", float(np.mean(grad_norms)))

        # +SIGNAL (gen3_signal_rate_metrics_v1): the ADVANTAGE-DENSITY half of the `signal/` group,
        # measured above off the RAW pre-normalization advantages. `adv_raw_std` = how much the critic
        # thinks this rollout's actions mattered; `adv_raw_abs_mean` = its outlier-robust companion
        # (std rising alone ⇒ a few runaway points, not a broader density); `adv_kurtosis` = EXCESS
        # kurtosis, POSITIVE when the signal is concentrated in a few decisive turns (which is what
        # exploit signal looks like) and ≈0 when advantage mass is smeared evenly across decisions.
        # Read WITH `signal/outcome_entropy` (SignalMetricsCallback) — high outcome entropy with LOW
        # density is the mirror paradox, not health. NaN on a degenerate (constant) rollout.
        for _sk, _sv in signal_metrics.items():
            self.logger.record(f"signal/{_sk}", float(_sv))

        # +SCAFFOLDING GAUGE: `train/scaffolding_gauge` = (1 − Spearman ρ(V, P(win))) / 2 over
        # epoch 0's paired reads. 0 = the shaped critic and the win-prob head order states
        # identically (no scaffolding divergence visible in the ordering); 0.5 = independent.
        # It should SHRINK as a generation matures, and that trajectory is the registered signal
        # for annealing the shaping coefficients toward the pure game.
        # ⚠️ ORDERING ONLY — it claims nothing about magnitude, and it goes AMBIGUOUS exactly
        # where PBRS drives V_shaped toward a constant (the critic then has no variance left to
        # rank with). Read it beside `train/value_std`; the magnitude question is the offline
        # `python -m main.scaffolding_gauge`, which fits a per-checkpoint affine V→outcome map on
        # realized outcomes. NaN on a degenerate rollout, and NO key at all when the run carries
        # no win-prob head — a run without the head must leave a GAP, not a flat zero.
        if scaffold_v:
            for _gk, _gv in live_gauge_metrics(np.concatenate(scaffold_v),
                                               np.concatenate(scaffold_z)).items():
                self.logger.record(f"train/{_gk}", float(_gv))

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

        # +WIN-PROB PBRS (gen3_winprob_pbrs_v1, ai_v12 route 1): the shaping term's magnitude for THIS
        # rollout, computed in `collect_rollouts` (not here — the term edits rewards, not the loss, so
        # it has no per-minibatch existence). Under `train/` deliberately: it is a property of the
        # reward stream PPO is fitting, not of the win-prob head, and it belongs beside the other
        # train-loop quantities a reader checks when the loss moves. `pbrs_reward_share` is the one to
        # watch — the shaping's mean |magnitude| as a fraction of the UNSHAPED reward's, i.e. how much
        # of the return signal this coefficient has replaced.
        if self._pbrs_metrics:
            for _pk, _pv in self._pbrs_metrics.items():
                self.logger.record(f"train/pbrs_{_pk}", float(_pv))

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
        # Under `--distill-target action` (gen3_distill_target_gate_v1) the §4.3 liveness row rides the
        # same prefix: `gated_frac` / `n_gated` (0 is a reading: the gate found nothing) /
        # `gate_agree_rate` (student argmax == teacher argmax ON GATED ROWS) / `mean_gate_adv` — the
        # dose meters G2's share-matching is read against (with grad/distill_share).
        # Empty (off / no teacher-team states in any minibatch) → not logged.
        if distill_metrics:
            for _dk, _dvals in distill_metrics.items():
                self.logger.record(f"distill/{_dk}", float(np.mean(_dvals)))

        # +TD-AUX: Bellman-residual diagnostics under their OWN `td_aux/` TB prefix. `resid_rms` is
        # the headline — the quantity the term minimises, and the live counterpart of the offline
        # ΔV-dispersion instrument the rung-1 gate used; it should FALL. `resid_mean` (SIGNED) is the
        # no-harm watch: rung 1's decomposition says this is dispersion suppression, so a bias that
        # drifts away from ~0 means the residual-gradient (Baird) term is shifting the level rather
        # than tightening it — read it beside `train/explained_variance`. `scale` is the unit the
        # residual is expressed in (PopArt's sigma; 1.0 with PopArt off), and `pair_drop_frac` is the
        # fraction of candidate pairs lost to episode boundaries. Empty (off) → not logged.
        if td_aux_metrics:
            for _tdk, _tdvals in td_aux_metrics.items():
                self.logger.record(f"td_aux/{_tdk}", float(np.mean(_tdvals)))

        # +CF-WINPROB (gen3_cf_label_plumbing_v1). Two blocks, and they answer DIFFERENT questions:
        #
        #  * `cf/*` is PRODUCER LIVENESS — published on every train() the moment a buffer exists,
        #    whether or not a single label ever arrived. An empty buffer that does not announce
        #    itself is this tree's oldest failure mode (the search-teacher's silent starvation), so
        #    `cf/buffer_fill` == 0 with a flat `cf/labels_ingested_total` is a first-class reading,
        #    not an absence of readings.
        #  * `train/cf_loss` + `train/cf_grad_share` are the TERM — only when it actually folded.
        #    `cf_grad_share` is lifted from the grad-balance probe's shared denominator so it is
        #    directly comparable with grad/policy_share et al; it reads 0.0 under `cf_head_only`
        #    (the default) because the head's input is stop-grad'd, which is the head-only stage's
        #    verification, not a defect.
        if cf_buffer is not None:
            for _ck, _cv in cf_buffer.stats(int(self.num_timesteps)).items():
                self.logger.record(_ck, _cv)
        if cf_any_on:
            # Rows CONSUMED per train() — the throughput half of liveness, which residency alone
            # cannot report. Recorded whenever the fold is enabled (0 is the reading that matters:
            # the term is on and the buffer gave it nothing).
            self.logger.record("cf/rows_sampled", float(cf_rows_sampled))
        if cf_metrics:
            for _ck2, _cvals in cf_metrics.items():
                self.logger.record(f"cf/{_ck2}", float(np.mean(_cvals)))
            self.logger.record("train/cf_loss", float(np.mean(cf_metrics.get("loss", [0.0]))))
        if cf_winprob_on:
            self.logger.record("train/cf_grad_share",
                               float(grad_balance.get("grad/cf_winprob_share", 0.0)))
        # +CF-EVIDENTIAL (gen3_cf_evidential_head_v1) — `cf/evid_*`, its own sub-prefix so a reader
        # can tell the Beta readout's numbers from the scalar head's at a glance.
        #
        #  * `nll` is the term being minimised (Beta-Binomial marginal, per rollout) and `reg` the
        #    KL pull toward Beta(1,1).
        #  * `precision_mean` (α+β) is the EVIDENCE the head claims. An unbounded climb is the
        #    evidential-overconfidence failure `--cf-evidential-reg` exists to bound; read the two
        #    together, because a falling `nll` with a runaway precision is the head buying its loss
        #    with certainty it has not earned.
        #  * `epistemic_std_mean` is THE HEADLINE and the pre-registered read: the confessed width
        #    should CORRELATE, per stratum, with `cf_audit`'s measured `sd_true_excess` for that
        #    stratum. Wide everywhere and wide nowhere are the same null.
        if cf_evid_metrics:
            for _ek, _evals in cf_evid_metrics.items():
                self.logger.record(f"cf/evid_{_ek}", float(np.mean(_evals)))
            self.logger.record("train/cf_evidential_loss",
                               float(np.mean(cf_evid_metrics.get("nll", [0.0]))))
        if cf_evid_on:
            # Reads 0.0 by construction (the head's input is always detached). Published so the
            # always-detached contract is a LIVE measurement rather than a claim in a docstring.
            self.logger.record("train/cf_evidential_grad_share",
                               float(grad_balance.get("grad/cf_evidential_share", 0.0)))
        # +CF-TWIN (gen3_cf_twin_heads_v1) — `cf/twin_*`. The PAIRED read, live.
        #
        #  * `c_loss` / `b_loss` are the two arms' cf folds; `b_coverage` is the fraction of the
        #    sampled rows that carried a single-outcome label AT ALL. **Read `b_coverage` first.**
        #    A twin-heads run whose producer ships no `outcome_label` trains B on nothing, B then
        #    equals A, and the C−B contrast silently becomes C−A while every other scalar reads
        #    healthy. That is the one way this arm can produce a confident wrong answer.
        #  * `b_vs_c_abs` / `b_minus_c` are the two heads' predictions differing on the SAME states.
        #    A `b_vs_c_abs` pinned near 0 means the label streams have not separated the heads and
        #    there is nothing to decompose yet — a coverage/dosage reading, not a result.
        #  * `*_onpolicy_*` are the mirrored control objective. They should track head A's
        #    `win_prob/*` closely; a persistent gap means the twins are NOT carrying a bit-identical
        #    copy of A's loss and the factorial's base is not shared.
        #  THE RESULT IS NONE OF THESE. It is `cf_audit`'s held-out, battle-clustered paired
        #  differences (runbook §2 as amended); these are the launch-window instrument.
        if cf_twin_metrics:
            for _tk, _tvals in cf_twin_metrics.items():
                self.logger.record(f"cf/twin_{_tk}", float(np.mean(_tvals)))
            # ABSENT, never zero — the shadow head's rule, for the shadow head's reason. The
            # COMBINED (C + B) unweighted fold, summed per minibatch inside `cf_twin_terms` and
            # only meaned here: the two arms' lists differ in length when B starves, so
            # mean(c)+mean(b) would be the mean of no minibatch that ever folded. One scalar,
            # because this block contributes ONE term to the loss; the per-arm split is already
            # live at `cf/twin_c_loss` / `cf/twin_b_loss`, which is where you read the arms.
            # The key is missing (not 0.0) when the CF fold never ran — `cf_twin_metrics` can be
            # non-empty from the ON-POLICY mirror alone, and a defaulted 0.0 would then publish
            # a perfect score for an arm that saw no counterfactual label at all.
            if "loss" in cf_twin_metrics:
                self.logger.record("train/cf_twin_loss",
                                   float(np.mean(cf_twin_metrics["loss"])))
        if cf_twin_on:
            self.logger.record("train/cf_twin_grad_share",
                               float(grad_balance.get("grad/cf_twin_share", 0.0)))
        # +CF-SHADOW (gen3_cf_twin_heads_v1) — `cf/shadow_*`.
        #
        #  * `loss` is the MSE against `mc_return` in the PopArt-normalized frame; `abs_err`,
        #    `pred_mean` and `label_mean` are the same quantities de-normalized to real shaped-return
        #    units, which is the only frame a reader can interpret.
        #  * **`shadow_vs_live_v` is THE METER** — the SIGNED mean of (shadow − live V) in real
        #    units on the same states. It is the staged-promotion evidence: a shadow sitting
        #    systematically BELOW the live critic is a live critic that is optimistic about the
        #    states the factory sampled, measured against ground truth rather than argued from a
        #    calibration curve. `live_v_vs_label` is its direct half (live V minus the MC label);
        #    read them together, since the shadow is itself a fitted head and can be wrong too.
        #  * `coverage` is `mc_return`'s label coverage — the same first-read rule as `b_coverage`.
        if cf_shadow_metrics:
            for _sk, _svals in cf_shadow_metrics.items():
                self.logger.record(f"cf/shadow_{_sk}", float(np.mean(_svals)))
            # ABSENT, never zero. A starved shadow (no `mc_return` arrived, or every row's reward
            # digest was refused) folds NO term, so there is no `loss` key — and defaulting it to
            # 0.0 would publish a PERFECT SCORE for a head that trained on nothing, which is
            # indistinguishable on a TB chart from a perfectly-fit one. `cf/shadow_coverage` still
            # publishes, so the starvation is visible rather than silent.
            if "loss" in cf_shadow_metrics:
                self.logger.record("train/cf_shadow_loss",
                                   float(np.mean(cf_shadow_metrics["loss"])))
        if cf_shadow_on:
            # 0.0 by construction (always-detached), published for the same reason as the
            # evidential head's — this head is a promotion PATH, so its passivity is the contract.
            self.logger.record("train/cf_shadow_grad_share",
                               float(grad_balance.get("grad/cf_shadow_share", 0.0)))

        # +CAPACITY TELEMETRY (gen3_capacity_telemetry_v1). Read them as TRENDS, never as levels —
        # every one of these is a saturation EARLY WARNING and none has a meaningful absolute value:
        #   canary_loss / canary_recovery / canary_age  the plasticity canary. `canary_recovery` is
        #       the one-number read (post-reset loss ÷ pre-reset loss for the target that was last
        #       re-seeded); compare it at a MATCHED `canary_age`, since it decays with age by design.
        #   canary_steps  how many canary updates this train() actually took. 0 with the flag ON
        #       means the `value_pooled` snapshot never arrived (a non-Gen3 extractor, or a stash
        #       that stopped being populated) — the tell that would otherwise be a silent gap.
        #   halfbatch_cosine  the two half-batches' agreement on the shared trunk. Falling toward
        #       0 / negative = the batch is fighting itself. Read with halfbatch_grad_norm_ratio.
        #   feature_velocity{,_cos,_rel}  how far the FROZEN probe batch's features moved since the
        #       last measurement. Falling velocity at constant `train/grad_norm` = weights move but
        #       functions do not.
        for _capk, _capv in capacity_metrics.items():
            self.logger.record(f"capacity/{_capk}", float(_capv))

        # +PopArt diagnostics: mu/sigma should TRACK train/return_mean/return_std (the running
        # normalizer estimate); value_weight_norm watches the POP rescale stay bounded (an explosion
        # signals a degenerate sigma / broken preservation). With PopArt on, train/value_loss is the
        # NORMALIZED loss (≈O(1)) and grad/value_policy_logratio should fall toward ~0 (the
        # aux-independent value/policy balance; grad/value_share also drops but moves with the aux count).
        if popart is not None:
            self.logger.record("popart/mu", float(self.policy.popart.mu))
            self.logger.record("popart/sigma", float(self.policy.popart.sigma))
            self.logger.record("popart/value_weight_norm", float(self.policy.value_net.weight.norm()))
        # (v61's `value_seeds/*` seed-collapse contract was logged here. The multi-seed critic
        # readout it monitored is DELETED — dV 0.0000 bit-exact on two consecutive end-of-run
        # audits — so the monitor went with it. Its finding survives in designs/CHANGELOG.md.)
        for _sk, _svals in aux_metrics.items():
            self.logger.record(_sk, float(np.mean(_svals)))
