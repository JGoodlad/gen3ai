"""`InstrumentedMaskablePPO` — the class, and `train()`: the whole FOLD SEQUENCE in ONE module.

⚠️ **The fold sequence is not split, and that is the design.** `train()` is a vendored copy of
upstream `sb3_contrib.MaskablePPO.train` (hash-pinned in the hub) with our terms folded in, and the
ORDER in which those terms are folded is a CONTRACT — see `train()`'s own docstring for the
numbered version. Splitting the sequence across modules would make an ordering that is currently
straight-line source order into something a reader has to reassemble, and the one property that
matters about it (no flag combination reorders these) would stop being visible.

Everything that is NOT the sequence has moved out. The per-term losses live in `distill_terms`,
`value_terms` and `aux_terms`; the knobs in `hparams`; the noise-scale machinery in `noise_scale`.
Three modules hold the rest of what `train()` used to spell out inline, and each is a mixin whose
methods `train()` calls in place:

    train_setup.py      the pre-loop half — the opponent-intent label alignment, the FOLD FLAGS
                        (`FoldFlags`) and the once-per-call probes (`ProbeSetup`). Both containers
                        are unpacked back into the locals the loop is written against, so the fold
                        body is unchanged by their existence.
    metrics_export.py   the ~400-line `self.logger.record` tail — diagnostics, no gradient. One
                        method per TB prefix group, each taking the accumulators this call filled.
    rollout_probes.py   `collect_rollouts`, the entropy-boost schedule, the episode-start read —
                        per-ROLLOUT work that is not part of the fold at all.

**A source-level pin that says "in `train()`" should read `train_step_source()`** (below), which is
`train()` plus those delegates. The fold, its setup and its export are one train step; which of the
three a given line sits in is a decomposition detail, and a pin that depends on it breaks on a move
that changed nothing.
"""
import inspect
import time

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance

from agents.training.grad_balance import (
    cell_family_metrics,
    edge_family_metrics,
    grad_balance_metrics,
)
from agents.training import belief_bank as _belief_bank
from agents.training.instrumented_ppo.aux_terms import AuxTerms
from agents.training.instrumented_ppo.capacity_terms import CapacityTerms
from agents.training.instrumented_ppo.calibration import (   # the MODULE path, never the hub:
    CalibrationAccumulator as _CalibrationAccumulator,        # a submodule importing the package
    as_numpy as _calib_as_numpy,                              # __init__ back closes the import
    contested_mask as _calib_contested_mask,                  # cycle `ppo` sits at the end of
    sigmoid as _calib_sigmoid,                                # (pinned by the hub-contract test).
)
from agents.training.instrumented_ppo.constants import _WIN_CONTESTED_TAU
from agents.training.instrumented_ppo.distill_anchor import distill_anchor_step
from agents.training.instrumented_ppo.distill_terms import DistillTerms
from agents.training.instrumented_ppo.hparams import PpoHyperparameters
from agents.training.instrumented_ppo.metrics_export import TrainMetricsExport
from agents.training.instrumented_ppo.noise_scale import NoiseScaleDiagnostics
from agents.training.instrumented_ppo.noise_scale_terms import NULL_TAGGER
from agents.training.instrumented_ppo.rollout_probes import RolloutProbes
from agents.training.instrumented_ppo.train_setup import TrainSetup
from agents.training.instrumented_ppo.value_terms import ValueTerms
from agents.training.rank_metrics import rank_probe


def train_step_source() -> str:
    """`train()` plus every method it delegates a piece of the train step to, concatenated.

    THE unit a source-level pin should read. Before the setup and the metrics export moved out of
    `train()`, `inspect.getsource(InstrumentedMaskablePPO.train)` WAS the train step, and a dozen
    tests in this tree pin properties of it that way — that a flag is resolved with `is_winprob`,
    that a term is tagged `value` and not `aux`, that the noise-scale fold goes through the shared
    debiased EMA. Those properties are about the train step, not about which of three files a line
    ended up in, so they read this. The fold's own ORDERING pins stay on `train()` itself, where
    straight-line source order is the thing being checked.
    """
    return "\n".join(inspect.getsource(fn) for fn in (
        InstrumentedMaskablePPO.train,
        TrainSetup._align_opp_intent_labels,
        TrainSetup._resolve_fold_flags,
        TrainSetup._train_probe_setup,
        TrainMetricsExport._record_grad_balance_metrics,
        TrainMetricsExport._record_signal_metrics,
        TrainMetricsExport._record_noise_scale_metrics,
        TrainMetricsExport._record_head_metrics,
        TrainMetricsExport._record_term_metrics,
        TrainMetricsExport._record_cf_metrics,
        TrainMetricsExport._record_capacity_and_popart_metrics,
    ))


class InstrumentedMaskablePPO(PpoHyperparameters,
                              NoiseScaleDiagnostics,
                              DistillTerms,
                              ValueTerms,
                              AuxTerms,
                              CapacityTerms,
                              TrainSetup,
                              TrainMetricsExport,
                              RolloutProbes,
                              MaskablePPO):
    """MaskablePPO with `train/clip_fraction_vf` instrumentation added.

    Behaviour-identical to `MaskablePPO` except for the additional TensorBoard
    metric. See module docstring for drift-detection details.

    Also dispatches rollout collection to the **non-barrier async collector** when
    ``self._async_rollout`` is set and the env is an ``AsyncSubprocVecEnv`` (``--async-rollout``);
    otherwise it is the unchanged stock ``MaskablePPO.collect_rollouts``.
    """

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
        # +INSTRUMENTATION: the wall clock of the WHOLE call, recorded as `train/train_ms`. It is
        # the denominator every "this probe costs X% of the train step" claim in this file needs,
        # and reading it live is the only way that claim can stay true as the fold grows.
        _t_train0 = time.perf_counter()
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # +OPPONENT INTENT (gen3_opp_intent_v1): row-align the one-ahead intent labels to the
        # predictions ONCE, here, while the [n_steps, n_envs] structure and `episode_starts` still
        # exist — after `get()` shuffles, the adjacency is gone. See `train_setup.py`.
        self._align_opp_intent_labels()

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
        # +WIN-PROB CALIBRATION: reliability-diagram BIN COUNTS over epoch 0, pooled and restricted
        # to material-EVEN decisions. Bin counts rather than per-minibatch ECEs because an ECE is
        # nonlinear in the populations (see `calibration.CalibrationAccumulator`).
        calib_all = _CalibrationAccumulator()
        calib_contested = _CalibrationAccumulator()
        teacher_metrics: dict[str, list[float]] = {}    # +SEARCH-TEACHER: AWR per-minibatch diagnostics
        opd_metrics: dict[str, list[float]] = {}         # +OPD: on-policy self-distillation KL diagnostics
        # Shared sink for the per-minibatch aux diagnostics that already carry their OWN full TB
        # key (`opp_intent/*`), so they are recorded verbatim rather than under a prefix.
        aux_metrics: dict[str, list[float]] = {}
        distill_metrics: dict[str, list[float]] = {}     # +DISTILL: exploiter-distillation KL diagnostics
        td_aux_metrics: dict[str, list[float]] = {}      # +TD-AUX: Bellman-residual diagnostics
        value_dist_metrics: dict[str, list[float]] = {}  # +VALUE-DIST: per-minibatch HL-Gauss diagnostics
        # Compute once: WHICH terms this call folds — and, for the counterfactual family, the one
        # per-rollout buffer poll. Every flag is read by exactly the guard of the term it names,
        # and the reasoning for each sits beside its computation in `train_setup._resolve_fold_flags`.
        # Unpacked back into locals so the fold below reads exactly as it was written.
        _f = self._resolve_fold_flags()
        belief_aux_on, move_belief_on = _f.belief_aux_on, _f.move_belief_on
        move_latent_on, spread_belief_on = _f.move_latent_on, _f.spread_belief_on
        hp_type_belief_on, item_belief_on = _f.hp_type_belief_on, _f.item_belief_on
        critic_winprob, win_prob_on = _f.critic_winprob, _f.win_prob_on
        scaffolding_on, value_from_dist = _f.scaffolding_on, _f.value_from_dist
        value_dist_on, search_teacher_on = _f.value_dist_on, _f.search_teacher_on
        opd_on, distill_on = _f.opd_on, _f.distill_on
        distill_rows_in_buffer, policy_grad_coef = _f.distill_rows_in_buffer, _f.policy_grad_coef
        td_aux_on, cf_buffer, cf_winprob_on = _f.td_aux_on, _f.cf_buffer, _f.cf_winprob_on
        cf_evid_on, cf_twin_on, cf_shadow_on = _f.cf_evid_on, _f.cf_twin_on, _f.cf_shadow_on
        q_winprob_on, q_onpolicy_on, cf_any_on = _f.q_winprob_on, _f.q_onpolicy_on, _f.cf_any_on
        cf_metrics: dict[str, list[float]] = {}
        cf_evid_metrics: dict[str, list[float]] = {}
        cf_twin_metrics: dict[str, list[float]] = {}     # +CF-TWIN (gen3_cf_twin_heads_v1)
        cf_shadow_metrics: dict[str, list[float]] = {}   # +CF-SHADOW (gen3_cf_twin_heads_v1)
        q_metrics: dict[str, list[float]] = {}           # +Q-WINPROB (gen3_q_winprob_head_v1)
        cf_rows_sampled = 0

        continue_training = True

        # The once-per-train() probes, PopArt's advance and the two gradient samplers —
        # `train_setup._train_probe_setup`, which takes `distill_metrics` because the grad-projector
        # writes straight into it. Unpacked into the names the fold's `_ntg`/`_dgp` seams use.
        _p = self._train_probe_setup(distill_metrics)
        shared_trunk, grad_balance = _p.shared_trunk, _p.grad_balance
        rank_metrics, edge_metrics = _p.rank_metrics, _p.edge_metrics
        cell_metrics, grad_norms, capacity = _p.cell_metrics, _p.grad_norms, _p.capacity
        capacity_metrics, popart = _p.capacity_metrics, _p.popart
        signal_metrics, accum, noise_g_small_sq = _p.signal_metrics, _p.accum, _p.noise_g_small_sq
        noise_g_big_sq, _ns_terms, _dgp = _p.noise_g_big_sq, _p.ns_terms, _p.dgp
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # +GRAD-ACCUM: start each accumulation group with a clean grad buffer; count micro-batches.
            self.policy.optimizer.zero_grad()
            micro_in_group = 0
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # +NOISE-SCALE PER-TERM: collect on epoch 0's FIRST accumulation group only — the
                # same window the total's two points are read from, so both readings score the very
                # same data and a disagreement can only be the gradient. NULL elsewhere ⇒ the
                # `_ntg.add(...)` calls threaded through the fold below are pure passthroughs.
                _ntg = _ns_terms if (epoch == 0 and _ns_terms.micros < accum) else NULL_TAGGER
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
                elif critic_winprob or self.clip_range_vf is None:
                    # No clipping. Under `winprob` the scalar MSE is a DIAGNOSTIC (its term is
                    # dropped below), so clipping would only make `train/value_loss` read as a
                    # clipped quantity in probability units. `critic_winprob` is False on the
                    # `shaped` path, so this reads `self.clip_range_vf is None` there.
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
                # gen3_winprob_critic_mode_v1 adds the SECOND such case, for the same reason: the
                # critic is the win-prob head and `value_net` is in no loss graph, so the scalar
                # term would train a readout nothing reads. Its BCE joins the SAME "value" group.
                _vf_term = 0.0 if (value_from_dist or critic_winprob) else self.vf_coef * value_loss
                # +PG-COEF: at 1.0 (the default) `_policy_grad_term` IS the `policy_loss` tensor, so the
                # line below is literally the old `loss = policy_loss + …` expression —
                # byte-identical. Any other value scales ONLY the policy-gradient term.
                _policy_grad_term = policy_loss if policy_grad_coef == 1.0 else policy_grad_coef * policy_loss
                # +NOISE-SCALE PER-TERM: `_ent_term` names the sub-expression that was already
                # there — `a + b * c + d` and `a + (b*c) + d` are the same operations in the same
                # order — so the fold is byte-identical while the three RL groups become taggable.
                _ent_term = self.ent_coef * ent_loss_used
                loss = (_ntg.add("policy", _policy_grad_term) + _ntg.add("entropy", _ent_term)
                        + _ntg.add("value", _vf_term))

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
                    loss = loss + _ntg.add("aux", _bterm)
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
                            loss = loss + _ntg.add(
                                "aux", self.opp_intent_coef * self.beta_setvalued_coef * _sv)
                            oi_m["opp_intent/beta_setvalued_loss"] = float(_sv.detach())
                            oi_m["opp_intent/beta_setvalued_rows"] = oi_m_extra_rows
                        opp_intent_term = self.opp_intent_coef * oi_loss
                        loss = loss + _ntg.add("aux", opp_intent_term)
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
                    loss = loss + _ntg.add("aux", _bterm)
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
                    loss = loss + _ntg.add("aux", _bterm)
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
                        if critic_winprob:
                            # THE VALUE LOSS. One critic, one coefficient: `vf_coef`, never
                            # `win_prob_coef` (the `_ce_w` conditional the design retires).
                            win_prob_term = self.vf_coef * wp_loss
                            loss = loss + _ntg.add("value", win_prob_term)
                        else:
                            win_prob_term = self.win_prob_coef * wp_loss
                            loss = loss + _ntg.add("aux", win_prob_term)
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

                # +WIN-PROB CALIBRATION (gen3_winprob_calibration_export_v1): the reliability half
                # of the head's diagnostics. Brier is a PROPER score and decomposes as
                # reliability − resolution + uncertainty, so it can stay flat while calibration
                # drifts; ECE/MCE/the per-bin gaps isolate the reliability term. Accumulated in BIN
                # COUNTS across the minibatches of EPOCH 0 (an ECE is nonlinear in the bin
                # populations — the mean of per-minibatch ECEs is not the pooled ECE) and folded
                # once at the end. Read-only: detached, no gradient, no RNG.
                if scaffolding_on and epoch == 0:
                    _cz = getattr(self.policy.features_extractor, "last_win_prob_logits", None)
                    _ct = rollout_data.observations.get("win_target")
                    _cm = rollout_data.observations.get("win_mask")
                    if _cz is not None and _ct is not None and _cm is not None:
                        _cp = _calib_sigmoid(_calib_as_numpy(_cz).reshape(-1))
                        _cy = _calib_as_numpy(_ct).reshape(-1)
                        _ck_mask = _calib_as_numpy(_cm).reshape(-1)
                        calib_all.observe(_cp, _cy, _ck_mask)
                        _cmar = _calib_contested_mask(
                            _calib_as_numpy(rollout_data.observations.get("win_margin")),
                            _WIN_CONTESTED_TAU)
                        if _cmar is not None and _cmar.size == _cp.size:
                            calib_contested.observe(_cp, _cy, _ck_mask * _cmar)

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
                        loss = loss + _ntg.add("aux", cf_twin_op_term)
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
                            loss = loss + _ntg.add("aux", value_dist_term)
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
                        # gen3_distill_offslice_anchor_v1: the licensing probe's ON-SLICE half —
                        # student↔teacher top-1 agreement, averaged over the ACTIVE teachers, so
                        # `distill/teacher_agreement_on_slice` (absorption) is readable beside
                        # `distill/collateral_kl` (damage) without expanding the per-teacher rows.
                        _on_agree, _on_agree_n = 0.0, 0
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
                                _a_k = _m_k.get("agree_rate", _m_k.get("gate_agree_rate"))
                                if _a_k is not None:
                                    _on_agree += float(_a_k)
                                    _on_agree_n += 1
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
                            # +DISTILL-GRAD-PROJECT: `_dgp.add` records the TEACHER terms (this one
                            # and the two value-side ones below) as the gradient source to project.
                            # It returns its argument unchanged, so the fold is the one that was
                            # here before; the anchor term deliberately does NOT get this wrapper.
                            loss = loss + _ntg.add("distill", _dgp.add(distill_term))
                            distill_metrics.setdefault("kl", []).append(float(_distill_kl))
                            distill_metrics.setdefault("n_teachers_active", []).append(float(len(_per_teacher_kl)))
                            if _on_agree_n:
                                distill_metrics.setdefault("teacher_agreement_on_slice", []).append(
                                    _on_agree / _on_agree_n)
                        if _per_teacher_vd:
                            _distill_vd = th.stack(_per_teacher_vd).mean()    # balanced like the policy KL
                            loss = loss + _ntg.add(
                                "distill", _dgp.add(self.distill_value_coef * _distill_vd))
                            distill_metrics.setdefault("value_mse", []).append(float(_distill_vd))
                        if _per_teacher_vfd:
                            _distill_vfd = th.stack(_per_teacher_vfd).mean()  # balanced like the policy KL
                            loss = loss + _ntg.add(
                                "distill", _dgp.add(self.distill_value_feat_coef * _distill_vfd))
                            # Same naming note as the per-teacher site above: DISTANCE (1 − cos), lower =
                            # better aligned. `value_feat_dist` is canonical; `value_feat_cos` is the
                            # deprecated alias kept one release.
                            for _vfd_key in ("value_feat_dist", "value_feat_cos"):
                                distill_metrics.setdefault(_vfd_key, []).append(float(_distill_vfd))

                # +DISTILL-ANCHOR (gen3_distill_offslice_anchor_v1): the OFF-SLICE trust region to
                # the FROZEN fold parent, and the live collateral-KL meters. ONE call — everything
                # (the frozen forward, the slice split, the loss, every `distill/*` meter) lives in
                # `distill_anchor.py`. `_distill_anchor_parent` absent (no flag) ⇒ returns None
                # having done nothing, so the loss expression is byte-identical; attached at
                # coefficient 0 (`--distill-anchor-monitor`) ⇒ meters only, still no term. It rides
                # the `distill` noise-scale group because it is part of the fold's dose, not an aux
                # head. The student's π is the one `evaluate_actions` already built, as the distill
                # term reuses it.
                anchor_term = distill_anchor_step(
                    self, rollout_data,
                    getattr(self.policy, "_last_pi_distribution", None), distill_metrics)
                if anchor_term is not None:
                    loss = loss + _ntg.add("distill", anchor_term)

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
                            loss = loss + _ntg.add("aux", searchteacher_term)
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
                                loss = loss + _ntg.add("aux", opd_term)
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
                        loss = loss + _ntg.add("aux", td_aux_term)
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
                q_term = None
                q_op_term = None
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
                            loss = loss + _ntg.add("aux", cf_term)
                            for _cfk, _cfv in _cfm.items():
                                cf_metrics.setdefault(_cfk, []).append(float(_cfv))
                    if cf_evid_on:
                        cf_evid_term, _cfem = self._cf_evidential_term(_cf_ctx)
                        if cf_evid_term is not None:
                            loss = loss + _ntg.add("aux", cf_evid_term)
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
                            loss = loss + _ntg.add("aux", cf_twin_term)
                        for _ck, _cv in _cftm.items():
                            cf_twin_metrics.setdefault(_ck, []).append(float(_cv))
                    # +CF-SHADOW: the passive value twin on `mc_return`. Same sample, same forward.
                    if cf_shadow_on:
                        cf_shadow_term, _cfsm = self._cf_shadow_term(_cf_ctx, popart)
                        if cf_shadow_term is not None:
                            loss = loss + _ntg.add("aux", cf_shadow_term)
                        for _sk, _sv in _cfsm.items():
                            cf_shadow_metrics.setdefault(_sk, []).append(float(_sv))
                    # +Q-WINPROB (gen3_q_winprob_head_v1): the PER-ACTION head, on the SAME sample
                    # and the SAME forward. Both halves collect metrics unconditionally — the
                    # coverage columns are the starvation tell and must be published even (and
                    # especially) on a minibatch where the term itself did not fold.
                    if q_winprob_on:
                        q_term, _qm = self._q_winprob_term(_cf_ctx)
                        if q_term is not None:
                            loss = loss + _ntg.add("aux", q_term)
                        for _qk, _qv in _qm.items():
                            q_metrics.setdefault(_qk, []).append(float(_qv))
                    if q_onpolicy_on:
                        q_op_term, _qom = self._q_winprob_onpolicy_term(_cf_ctx)
                        if q_op_term is not None:
                            loss = loss + _ntg.add("aux", q_op_term)
                        for _qk, _qv in _qom.items():
                            q_metrics.setdefault(_qk, []).append(float(_qv))

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
                # +ANCHOR-SHARE (gen3_distill_offslice_anchor_v1): `grad/distill_anchor_share` on
                # the SAME denominator as `grad/distill_share` — the pair IS the dose reading a
                # trust region has to be sized by (how hard is the anchor pulling, relative to the
                # teacher content it is protecting?). Absent when no anchor folded.
                if anchor_term is not None:        aux_probe_terms["distill_anchor"] = anchor_term
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
                # gen3_q_winprob_head_v1: the Q head's inputs are detached INSIDE the extractor
                # forward (`q_winprob_mode` has no `shaping` value), so `grad/q_winprob_share`
                # reads exactly 0.0 BY CONSTRUCTION. Registered for the evidential head's reason:
                # "this readout cannot perturb the policy" is the flag's load-bearing claim, and a
                # published zero is a live measurement of it where a docstring is not. Both halves
                # ride ONE entry (they pull the same head) and are summed by the same arity-free
                # `sum` the twins use.
                _q_terms = [t for t in (q_term, q_op_term) if t is not None]
                if _q_terms:
                    aux_probe_terms["q_winprob"] = sum(_q_terms[1:], _q_terms[0])
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
                        (win_prob_term if (critic_winprob and win_prob_term is not None)
                         else value_dist_term if (value_from_dist and value_dist_term is not None)
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
                # +NOISE-SCALE PER-TERM: take the per-group gradients LAST, while the graph is
                # still alive and `.grad` still holds only what previous micro-batches put there.
                # `autograd.grad` writes no `.grad`, so the accumulation below is untouched.
                _ntg.flush_micro()
                # +DISTILL-GRAD-PROJECT: the removal vector is computed while the graph is alive
                # (read-only `autograd.grad`, no `.grad` written) and applied to `.grad` immediately
                # after the real backward — so `.grad` goes from `g_ppo + g_distill` to
                # `g_ppo + P_perp g_distill` with PPO's contribution bit-for-bit untouched. Both are
                # no-ops unless `--distill-anchor-mode grad_project`.
                _dgp.before_backward(self.policy, rollout_data)
                (loss / accum).backward()
                _dgp.after_backward(accum)
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

        self._record_grad_balance_metrics(grad_balance, rank_metrics, edge_metrics, cell_metrics,
                                          grad_norms)
        self._record_signal_metrics(signal_metrics, scaffold_v, scaffold_z)
        self._record_noise_scale_metrics(accum, noise_g_small_sq, noise_g_big_sq, _ns_terms)
        self._record_head_metrics(belief_metrics, win_prob_metrics, calib_all, calib_contested,
                                  critic_winprob, scaffolding_on, grad_balance)
        self._record_term_metrics(value_dist_metrics, teacher_metrics, opd_metrics,
                                  distill_metrics, td_aux_metrics)
        self._record_cf_metrics(cf_buffer, cf_any_on, cf_rows_sampled, cf_metrics, cf_winprob_on,
                                cf_evid_metrics, cf_evid_on, cf_twin_metrics, cf_twin_on,
                                cf_shadow_metrics, cf_shadow_on, q_metrics, q_winprob_on,
                                q_onpolicy_on, grad_balance)
        self._record_capacity_and_popart_metrics(capacity_metrics, popart, aux_metrics)
        # +INSTRUMENTATION: LAST line of train(), so it bounds the whole call — the honest
        # denominator for `train/noise_per_term_ms` and for every other probe's cost claim.
        self.logger.record("train/train_ms", 1000.0 * (time.perf_counter() - _t_train0))
