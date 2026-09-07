"""`TrainSetup` — everything `train()` resolves BEFORE the epoch loop, and nothing else.

Three methods, each returning what the fold then reads: the opponent-intent label alignment (a
buffer edit with no result), the FOLD FLAGS (which terms are live this call), and the PROBE SETUP
(the once-per-`train()` diagnostics, PopArt's advance, and the two gradient samplers).

None of this is the fold. Each flag is computed once and read by the `if <x>_on:` guard of the term
it owns, so the sequence in `ppo.train()` stays straight-line source with its guards inline — the
property `instrumented_ppo_hub_contract_test` reads. The two result containers are `NamedTuple`s
carrying the SAME names the fold uses, so `train()` unpacks them back into the locals the loop was
written against and the loop body is unchanged.
"""
from typing import Any, NamedTuple

import numpy as np
import torch as th

from agents.model.critic_mode import is_winprob
from agents.training.grad_balance import shared_trunk_parameters
from agents.training.instrumented_ppo.constants import _NOISE_PER_TERM_EVERY
from agents.training.instrumented_ppo.distill_grad_project import make_projector
from agents.training.instrumented_ppo.noise_scale_terms import (
    NULL_TAGGER,
    PerTermNoiseSampler,
    per_term_enabled,
)
from agents.training.instrumented_ppo.signal_metrics import advantage_density_metrics


class FoldFlags(NamedTuple):
    """WHICH terms this `train()` call folds. Every field is read by exactly the guard of the term
    it names; the reasoning for each one is the comment above its computation below."""
    belief_aux_on: Any
    move_belief_on: Any
    move_latent_on: Any
    spread_belief_on: Any
    hp_type_belief_on: Any
    item_belief_on: Any
    critic_winprob: Any
    win_prob_on: Any
    scaffolding_on: Any
    value_from_dist: Any
    value_dist_on: Any
    search_teacher_on: Any
    opd_on: Any
    distill_on: Any
    distill_rows_in_buffer: Any
    policy_grad_coef: Any
    td_aux_on: Any
    cf_buffer: Any
    cf_winprob_on: Any
    cf_evid_on: Any
    cf_twin_on: Any
    cf_shadow_on: Any
    q_winprob_on: Any
    q_onpolicy_on: Any
    cf_any_on: Any


class ProbeSetup(NamedTuple):
    """The once-per-`train()` diagnostics state. `ns_terms` / `dgp` are `_ns_terms` / `_dgp` in
    `train()`; a `NamedTuple` field cannot start with an underscore, and the local names are what
    the fold's `_ntg.add(...)` / `_dgp.add(...)` seams are written against."""
    shared_trunk: Any
    grad_balance: Any
    rank_metrics: Any
    edge_metrics: Any
    cell_metrics: Any
    grad_norms: Any
    capacity: Any
    capacity_metrics: Any
    popart: Any
    signal_metrics: Any
    accum: Any
    noise_g_small_sq: Any
    noise_g_big_sq: Any
    ns_terms: Any
    dgp: Any


class TrainSetup:
    """Mixin: the pre-loop half of `train()`."""

    def _align_opp_intent_labels(self) -> None:
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

    def _resolve_fold_flags(self) -> FoldFlags:
        """WHICH terms are live this call. Pure resolution plus ONE side effect — the counterfactual
        buffer's single disk poll, which sits where it always did: right after `cf_any_on` is known."""
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
        # +CRITIC MODE (gen3_winprob_critic_mode_v1): under `--critic winprob` the win-prob head IS
        # the value function, so the BCE below stops being an auxiliary and becomes THE value loss
        # — at `vf_coef`, tagged "value" (never "aux": §1.4 of the design records that
        # `train/noise_scale_value` spent the distributional-critic era describing a zero-weighted
        # term). The scalar `value_loss` survives as a diagnostic; its TERM is dropped, as under
        # Phase B. `shaped` (the default) is unchanged.
        critic_winprob = is_winprob(getattr(self.policy, "_critic_mode", "shaped"))
        win_prob_on = (
            getattr(self.policy.features_extractor, "win_prob_mode", "none") != "none"
            and (self.win_prob_coef != 0.0 or critic_winprob)   # winprob forces the BCE on
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
        # +Q-WINPROB (gen3_q_winprob_head_v1, v107): the PER-ACTION win-prob head — the amortized
        # one-ply search leaf (E5 step 2, GROUND). Two INDEPENDENT coefficients over one head: the
        # counterfactual per-action likelihood, and the WEAK taken-action fallback whose bias is
        # documented at its flag. Either being live turns the block on; both zero, no buffer or no
        # head (`--q-winprob-mode none`) skips it entirely — no sample, no forward, loss
        # byte-identical.
        q_head_built = getattr(self.policy.features_extractor, "q_winprob_head", None) is not None
        q_winprob_on = (
            float(getattr(self, "q_winprob_coef", 0.0)) != 0.0
            and cf_buffer is not None and q_head_built
        )
        q_onpolicy_on = (
            float(getattr(self, "q_winprob_onpolicy_coef", 0.0)) != 0.0
            and cf_buffer is not None and q_head_built
        )
        cf_any_on = (cf_winprob_on or cf_evid_on or cf_twin_on or cf_shadow_on
                     or q_winprob_on or q_onpolicy_on)
        if cf_any_on:
            # ONE disk poll per train() (= per rollout), not per minibatch: the producer writes at
            # its own pace and re-globbing a directory 240 times an update buys nothing.
            cf_buffer.poll(int(self.num_timesteps))
        return FoldFlags(
            belief_aux_on=belief_aux_on, move_belief_on=move_belief_on, move_latent_on=move_latent_on,
            spread_belief_on=spread_belief_on, hp_type_belief_on=hp_type_belief_on, item_belief_on=item_belief_on,
            critic_winprob=critic_winprob, win_prob_on=win_prob_on, scaffolding_on=scaffolding_on,
            value_from_dist=value_from_dist, value_dist_on=value_dist_on, search_teacher_on=search_teacher_on,
            opd_on=opd_on, distill_on=distill_on, distill_rows_in_buffer=distill_rows_in_buffer,
            policy_grad_coef=policy_grad_coef, td_aux_on=td_aux_on, cf_buffer=cf_buffer,
            cf_winprob_on=cf_winprob_on, cf_evid_on=cf_evid_on, cf_twin_on=cf_twin_on,
            cf_shadow_on=cf_shadow_on, q_winprob_on=q_winprob_on, q_onpolicy_on=q_onpolicy_on,
            cf_any_on=cf_any_on,
        )

    def _train_probe_setup(self, distill_metrics: dict) -> ProbeSetup:
        """The once-per-`train()` probes, PopArt's advance, and the two gradient samplers. Takes
        `distill_metrics` because the grad-projector writes its diagnostics straight into it."""
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
        # +NOISE-SCALE PER-TERM: the same two points, taken per LOSS GROUP so the total reading can
        # be told apart from the PPO policy term's own (noise_scale_terms.py's docstring is the why).
        # Built only on a sampled call — the cadence divides its cost — and NULL otherwise, in which
        # case every `_ntg.add(...)` below is a passthrough and no extra gradient is ever taken.
        self._noise_per_term_calls += 1
        _ns_terms = NULL_TAGGER
        if (accum >= 2 and per_term_enabled(self)
                and self._noise_per_term_calls % max(1, _NOISE_PER_TERM_EVERY) == 0):
            _ns_terms = PerTermNoiseSampler(list(self.policy.parameters()))
        # +DISTILL-GRAD-PROJECT (gen3_distill_grad_project_v1): SOURCE-SEPARATED anchoring — project
        # the DISTILL gradient off the off-slice behaviour subspace and leave PPO's gradient free.
        # NULL unless `--distill-anchor-mode grad_project`, in which case `_dgp.add(...)` below is a
        # passthrough and the two step-side hooks do nothing (update bit-identical). The whole
        # mechanism lives in `distill_grad_project.py`; this file holds only the seam.
        _dgp = make_projector(self, distill_metrics, list(self.policy.parameters()))
        return ProbeSetup(
            shared_trunk=shared_trunk, grad_balance=grad_balance, rank_metrics=rank_metrics,
            edge_metrics=edge_metrics, cell_metrics=cell_metrics, grad_norms=grad_norms,
            capacity=capacity, capacity_metrics=capacity_metrics, popart=popart,
            signal_metrics=signal_metrics, accum=accum, noise_g_small_sq=noise_g_small_sq,
            noise_g_big_sq=noise_g_big_sq, ns_terms=_ns_terms, dgp=_dgp,
        )
