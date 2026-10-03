"""`InstrumentedMaskablePPO` — the class, and `train()`: the whole FOLD SEQUENCE in ONE module.

⚠️ **The fold sequence is not split, and that is the design.** `train()` is a vendored copy of
upstream `sb3_contrib.MaskablePPO.train` (hash-pinned in the hub) with our terms folded in, and the
ORDER in which those terms are folded is a CONTRACT — see `train()`'s own docstring for the
numbered version. Splitting the sequence across modules would make an ordering that is currently
straight-line source order into something a reader has to reassemble, and the one property that
matters about it (no flag combination reorders these) would stop being visible.

Everything that is NOT the sequence has moved out. The per-term losses live in `value_terms`
and `aux_terms`; the knobs in `hparams`; the noise-scale machinery in `noise_scale`.
Three modules hold the rest of what `train()` used to spell out inline, and each is a mixin whose
methods `train()` calls in place:

    train_setup.py      the pre-loop half — the opponent-intent label alignment, the FOLD FLAGS
                        (`FoldFlags`) and the once-per-call probes (`ProbeSetup`). Both containers
                        are unpacked back into the locals the loop is written against, so the fold
                        body is unchanged by their existence.
    metrics_export.py   the ~400-line `self.logger.record` tail — diagnostics, no gradient. One
                        method per TB prefix group, each taking the accumulators this call filled.
    rollout_probes.py   `collect_rollouts` and the episode-start read — per-ROLLOUT work that is
                        not part of the fold at all.

**A source-level pin that says "in `train()`" should read `train_step_source()`** (below), which is
`train()` plus those delegates. The fold, its setup and its export are one train step; which of the
three a given line sits in is a decomposition detail, and a pin that depends on it breaks on a move
that changed nothing.
"""
import inspect
import time

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance

from agents.training.grad_balance import (
    cell_family_metrics,
    edge_family_metrics,
    grad_balance_metrics,
)
from agents.training.instrumented_ppo.aux_terms import AuxTerms
from agents.training.instrumented_ppo.capacity_terms import CapacityTerms
from agents.training.instrumented_ppo.calibration import (   # the MODULE path, never the hub:
    CalibrationAccumulator as _CalibrationAccumulator,        # a submodule importing the package
    as_numpy as _calib_as_numpy,                              # __init__ back closes the import
    contested_mask as _calib_contested_mask,                  # cycle `ppo` sits at the end of
    sigmoid as _calib_sigmoid,                                # (pinned by the hub-contract test).
)
from agents.training.instrumented_ppo.constants import _WIN_CONTESTED_TAU
from agents.training.instrumented_ppo.hparams import PpoHyperparameters
from agents.training.instrumented_ppo.loop import OwnedLoop   # gen3_owned_ppo_loop_v1: the loop is ours
from agents.training.instrumented_ppo.learner_gates import (   # K9(b) dispatch + K9(c)
    behaviour_gate_mode,
    check_buffer_finite,
    check_kl_finite,
    check_loss_finite,
    clip_grad_norm_checked,
)
from agents.training.instrumented_ppo.metrics_export import TrainMetricsExport
from agents.training.instrumented_ppo.micro_step import pack as _pack_micro
from agents.training.instrumented_ppo.micro_step import unpack as _unpack_micro
from agents.model.masked_categorical import MaskedPi as _MaskedPi
from agents.training.instrumented_ppo.noise_scale import NoiseScaleDiagnostics
from agents.training.instrumented_ppo.noise_scale_terms import NULL_TAGGER
from agents.training.instrumented_ppo.phase_hook import current as _current_phase_hook
from agents.training.instrumented_ppo.ridealong_terms import (RideAlongAccumulator,
                                                             RideAlongTerms)
from agents.training.instrumented_ppo.rollout_probes import RolloutProbes
from agents.training.instrumented_ppo.train_setup import TrainSetup
from agents.training.instrumented_ppo.value_terms import ValueTerms
from agents.training.rank_metrics import rank_probe_from_stash
from agents.training.instrumented_ppo.device_batches import install as _devb_install
from agents.training.instrumented_ppo.device_batches import uninstall as _devb_uninstall
from agents.training.instrumented_ppo.device_batches import DEFAULT_MODE as _DEVB_DEFAULT


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
    import agents.training.instrumented_ppo.micro_step as _ms
    return "\n".join(inspect.getsource(fn) for fn in (
        InstrumentedMaskablePPO.train,
        _ms.micro_step,                       # K8 region R1: fold steps 1-3a (gen3_learner_micro_step_v1)
        _ms.win_prob_terms,
        TrainSetup._micro_static,
        TrainSetup._align_opp_intent_labels,
        TrainSetup._resolve_fold_flags,
        TrainSetup._train_probe_setup,
        TrainMetricsExport._record_grad_balance_metrics,
        TrainMetricsExport._record_signal_metrics,
        TrainMetricsExport._record_noise_scale_metrics,
        TrainMetricsExport._record_head_metrics,
        TrainMetricsExport._record_capacity_metrics,
    ))


class InstrumentedMaskablePPO(PpoHyperparameters,
                              NoiseScaleDiagnostics,
                              ValueTerms,
                              AuxTerms,
                              CapacityTerms,
                              RideAlongTerms,
                              TrainSetup,
                              TrainMetricsExport,
                              RolloutProbes,
                              OwnedLoop,
                              MaskablePPO):
    """MaskablePPO with `train/clip_fraction_vf` instrumentation added.

    Behaviour-identical to `MaskablePPO` except for the additional TensorBoard
    metric. See module docstring for drift-detection details.

    Rollout collection is the Rust collector's (`RolloutProbes.collect_rollouts`).
    """

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.

        Vendored from `sb3_contrib.MaskablePPO.train` (hash pinned in
        `_EXPECTED_UPSTREAM_TRAIN_HASH`). The only deltas vs upstream are
        marked with `# +INSTRUMENTATION` comments.

        ------------------------------------------------------------------------------------
        THE FOLD ORDER IS A CONTRACT, and it is STRAIGHT-LINE SOURCE ORDER — in TWO parts
        ------------------------------------------------------------------------------------
        Every `loss = loss + <term>` runs in the order it is written, unconditionally — **no flag
        combination reorders them.** Each term is guarded by its own flag, and a term that is off
        contributes nothing rather than moving anyone else. Since K8 (`gen3_learner_micro_step_v1`)
        the sequence is TWO straight lines read in order: steps 1 to 3a (the PPO loss, the belief
        bank, opponent intent, the win-prob BCE) are the body of `micro_step.micro_step` — ONE
        function, compiled as the declared region R1 under --compile-trainer — and the steps after
        3a are folded below, onto R1's loss, as the DECLARED EAGER TAIL. Each part is checkable by
        reading; `instrumented_ppo_hub_contract_test` pins both orders.

        The sequence, per minibatch (1 to 3a inside R1, `micro_step.micro_step`):

          1. `loss = pg_term + ent_coef * entropy_loss + vf_term`   (the upstream PPO loss;
             `pg_term` is the UNSCALED `policy_loss` tensor at `policy_grad_coef == 1.0` — the default,
             byte-identical to upstream — else `policy_grad_coef * policy_loss` (`--policy-grad-coef`; 0.0 removes
             the policy-gradient term alone — entropy and the
             value term keep their own coefficients))
          2. the BELIEF bank — species/moves aux, opponent-intent (+ the set-valued beta term),
             move belief, spread belief, nature/EV, HP-type, item belief, move-latent
          3. (3a) the WIN-PROB BCE — the last R1 term

        The declared EAGER TAIL after R1 folds NO loss term: what follows (the ride-along heads'
        update, the capacity probes, the rank probe) READS the extractor STASHES that this
        minibatch's `evaluate_actions` forward left behind (`last_win_prob_logits`,
        `last_value_pooled`, …) and writes no `.grad` into the policy. Any future tail fold that
        runs its OWN extractor forward would CLOBBER those stashes, so it must follow every
        stash reader — moving a reader below it does not crash, it silently scores the wrong
        states. `instrumented_ppo_hub_contract_test.py` pins R1's order and that the R1 call
        precedes the tail's readers by reading the source. (The TD-consistency auxiliary and the
        counterfactual block — cf-winprob, cf-evidential, cf-twin, cf-shadow — were the tail's
        own-forward folds; deleted in deletion passes P11c and L4.)

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
        # +PHASE HOOK (gen3_learner_phase_hook_v1): BENCHMARK-ONLY segment marks, read ONCE here.
        # None in production, so every `if _ph is not None:` below is the whole cost; see
        # `phase_hook.py` for what each name books. No mark sits inside a compiled region.
        _ph = _current_phase_hook()
        if _ph is not None: _ph("start")
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # +OPPONENT INTENT (gen3_opp_intent_v1): row-align the one-ahead intent labels to the
        # predictions ONCE, here, while the [n_steps, n_envs] structure and `episode_starts` still
        # exist — after `get()` shuffles, the adjacency is gone. See `train_setup.py`.
        self._align_opp_intent_labels()
        # +K9(c) FAIL-CLOSED: the buffer's trained quantities (rewards, values, log-probs, advantages,
        # returns, every float label key) are finite, ONCE per update and BEFORE any forward
        # (`learner_gates`).
        check_buffer_finite(self.rollout_buffer)
        # +K9(b) / STALENESS (M5 Lane G): before any optimizer step; a no-op unless --behaviour-check.
        # Lane G's pre-loop probe (its own forward, age-bucketed by the rows' policy versions; a buffer
        # with no version record is judged as every row current) — the ONE implementation
        # (`learner_gates.behaviour_gate_mode`; the python core's in-loop variant went with it, U4).
        if behaviour_gate_mode(self) == "probe":
            self._behaviour_probe()

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        vf_clip_fractions: list[float] = []  # +INSTRUMENTATION
        belief_metrics: dict[str, list[float]] = {}  # +BELIEF: per-minibatch aux diagnostics (dict of lists)
        win_prob_metrics: dict[str, list[float]] = {}  # +WIN-PROB: per-minibatch diagnostics (dict of lists)
        # +WIN-PROB CALIBRATION: reliability-diagram BIN COUNTS over epoch 0, pooled and restricted
        # to material-EVEN decisions. Bin counts rather than per-minibatch ECEs because an ECE is
        # nonlinear in the populations (see `calibration.CalibrationAccumulator`).
        calib_all = _CalibrationAccumulator()
        calib_contested = _CalibrationAccumulator()
        # Shared sink for the per-minibatch aux diagnostics that already carry their OWN full TB
        # key (`opp_intent/*`), so they are recorded verbatim rather than under a prefix.
        aux_metrics: dict[str, list[float]] = {}
        # Compute once: WHICH terms this call folds — and, for the counterfactual family, the one
        # per-rollout buffer poll. Every flag is read by exactly the guard of the term it names,
        # and the reasoning for each sits beside its computation in `train_setup._resolve_fold_flags`.
        # Unpacked back into locals so the EAGER TAIL below reads as it was written; the R1 terms'
        # flags (the belief rows, the fork mask, the policy-gradient coefficient) are read by
        # `_micro_static` instead — the region's static flags.
        _f = self._resolve_fold_flags()
        belief_aux_on, move_belief_on = _f.belief_aux_on, _f.move_belief_on
        move_latent_on = _f.move_latent_on
        critic_winprob, win_prob_on = _f.critic_winprob, _f.win_prob_on
        # +FORK ARM (gen3_fork_v1) — computed in the Rust collector's fork pass
        # (`rust_rollout/fork.py`, which BLOCKS on the branch continuations before the epochs
        # begin) and stashed on the model. Recorded under its OWN `fork/` prefix rather than
        # folded into `win_prob/`: these are facts about the COLLECTION, not about the head's loss,
        # and the two families are read at different times by different people. Only when a pass
        # actually RAN: an absent `fork/*` family means --fork-fraction is 0.0 (or the arm
        # disabled itself, which announces itself once) and nothing else. Cleared at every
        # `_on_rollout_start`, so it can never be a stale rollout's.
        _fork_metrics = getattr(self, "_fork_metrics", None)
        if _fork_metrics:
            for _fk, _fv in _fork_metrics.items():
                if float(_fv) == float(_fv):      # a NaN is an empty slice; omit, never log it
                    self.logger.record(f"fork/{_fk}", float(_fv))
        # +RIDE-ALONG (gen3_ridealong_heads_v1): the detached heads' `ridealong/*` sink.
        ridealong_acc = RideAlongAccumulator()

        continue_training = True

        # The once-per-train() probes and the gradient sampler —
        # `train_setup._train_probe_setup`. Unpacked into the names the fold's `_ntg` seam uses.
        _p = self._train_probe_setup()
        shared_trunk, grad_balance = _p.shared_trunk, _p.grad_balance
        rank_metrics, edge_metrics = _p.rank_metrics, _p.edge_metrics
        cell_metrics, grad_norms, capacity = _p.cell_metrics, _p.grad_norms, _p.capacity
        capacity_metrics = _p.capacity_metrics
        signal_metrics, accum, noise_g_small_sq = _p.signal_metrics, _p.accum, _p.noise_g_small_sq
        noise_g_big_sq, _ns_terms = _p.noise_g_big_sq, _p.ns_terms
        diag = _p.diag   # gen3_diagnostics_cadence_v1: which optional probes run on THIS call
        # +PER-EPOCH (gen3_ppo_per_epoch_diag_v1): one (approx_kl, clip_fraction) pair per epoch the
        # loop actually ran, folded from the SAME per-minibatch numbers the stock tags already average
        # — no extra forward, no extra device sync. An early KL stop leaves fewer than n_epochs rows.
        epoch_approx_kl: list[float] = []
        epoch_clip_fraction: list[float] = []
        # +R1 (gen3_learner_micro_step_v1): the region's static flags, resolved ONCE, and the
        # per-update lists its diagnostics are routed into (by name).
        _micro_st = self._micro_static(_f)
        if getattr(self, "_compiled_micro_step", None) is not None:
            # K8: the compiled R1 runs ONLY at its startup declaration; a lever that moved since is
            # a typed FATAL naming the field (the sentinel's guard dump is the backstop).
            from agents.model.compile_regions import check_r1_declared
            check_r1_declared(self, _micro_st)
        _ppo_lists = {"pg_losses": pg_losses, "clip_fractions": clip_fractions,
                      "value_losses": value_losses, "entropy_losses": entropy_losses,
                      "vf_clip_fractions": vf_clip_fractions}
        if _ph is not None: _ph("setup")
        # +K8 (gen3_device_batches_v1 / gen3_device_batch_mode_v1): how the micro-batches reach the
        # device — `--device-batch` (`device_batches.MODES`: one resident copy of the flattened buffer,
        # or each micro-batch STAGED by a prefetch thread, or sb3's host path); the same permutation
        # draw and bit-identical batches in every mode; removed after the epoch loop.
        _devb = _devb_install(self.rollout_buffer, mode=getattr(self, "device_batch_mode", _DEVB_DEFAULT))
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            _epoch_cf_start = len(clip_fractions)   # +PER-EPOCH: this epoch's slice of the running list
            # +GRAD-ACCUM: start each accumulation group with a clean grad buffer; count micro-batches.
            self.policy.optimizer.zero_grad()
            micro_in_group = 0
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                if _ph is not None: _ph("batch")
                # +NOISE-SCALE PER-TERM: collect on epoch 0's FIRST accumulation group only — the
                # same window the total's two points are read from, so both readings score the very
                # same data and a disagreement can only be the gradient. NULL elsewhere ⇒ the
                # `_ntg.add(...)` calls threaded through the fold below are pure passthroughs.
                _ntg = _ns_terms if (epoch == 0 and _ns_terms.micros < accum) else NULL_TAGGER
                # +R1 (gen3_learner_micro_step_v1, M5 Lane K8): `evaluate_actions` AND fold steps 1-3a —
                # the upstream PPO loss, the belief bank's hidden_move site, the opponent-intent fold,
                # the latent and revealed sites, the win-prob BCE — as ONE function
                # (`micro_step.micro_step`), in exactly this source order inside it. Compiled as one
                # `fullgraph=True` region under --compile-trainer (`self._micro_region`), eager
                # otherwise; the steps after 3a below are the DECLARED EAGER TAIL, in contract order.
                _mo = self._micro_region()(self.policy, rollout_data.observations,
                                         rollout_data.actions, rollout_data.action_masks,
                                         rollout_data.old_log_prob, rollout_data.old_values,
                                         rollout_data.advantages, rollout_data.returns,
                                         _micro_st)
                actions = (rollout_data.actions.long().flatten() if _micro_st.discrete
                           else rollout_data.actions)
                values = _mo.values
                advantages = _mo.advantages
                loss = _mo.loss
                # The ride-along readers' stash, built OUTSIDE the region.
                if _mo.logp is not None:
                    self.policy._last_pi_distribution = _MaskedPi(_mo.logp, _mo.masks_bool)
                # +INSTRUMENTATION: effective rank of the trunk / value_cls / policy / vf reps, ONCE per
                # train() (first minibatch) — read from R1's OWN forward's stashes (K8,
                # gen3_rank_device_v1: no second forward; the spectra on the device, one host read).
                # HERE, before the tail: a tail fold that re-forwards would overwrite the stashes.
                if shared_trunk and diag.rank and not rank_metrics:
                    rank_metrics = rank_probe_from_stash(self.policy.features_extractor)
                # THE ONE host read of this micro-batch's diagnostics (every metric + presence + the
                # loss's finiteness + the approx-KL), routed into the per-update lists below.
                _mvals, _mpres, _mfinite = _unpack_micro(*_pack_micro(_mo))
                _approx_kl = _mvals.pop("approx_kl/")
                for _mk, _mv in _mvals.items():
                    _grp, _, _key = _mk.partition("/")
                    if _grp == "belief":
                        belief_metrics.setdefault(_key, []).append(_mv)
                    elif _grp == "aux":
                        aux_metrics.setdefault(_key, []).append(_mv)
                    elif _grp == "win_prob":
                        win_prob_metrics.setdefault(_key, []).append(_mv)
                    else:
                        _ppo_lists[_grp].append(_mv)
                for _tn, _tt in _mo.terms.items():          # the noise-scale per-term tagger
                    if _mpres.get(_tn, True):
                        _ntg.add(_mo.term_groups[_tn], _tt)
                _policy_grad_term, _ent_term = _mo.terms["policy"], _mo.terms["entropy"]
                _vf_term = _mo.terms.get("value", 0.0)

                # +CAPACITY: snapshot THIS forward's `value_pooled` before any later forward
                # replaces the stash. Detached in the snapshot itself, so nothing downstream can
                # accidentally hand the canary a live graph.
                cap_features = (
                    self._capacity_snapshot_features(self.policy.features_extractor,
                                                     int(values.shape[0]))
                    if capacity is not None else None)
                if _ph is not None: _ph("forward")
                # +RIDE-ALONG (gen3_ridealong_heads_v1): the DETACHED heads' own step, on THIS
                # forward's stashes. Every input is stop-grad and the heads have their own optimizer;
                # their grads are back to None when it returns, so nothing below — the tail, the clip,
                # the probes — can see them. A no-op (one attribute read) when the policy has no heads.
                self._ridealong_update(rollout_data, values, actions, epoch, ridealong_acc)
                if _ph is not None: _ph("ridealong")

                # (The in-training SCAFFOLDING GAUGE — a paired (V, win-prob logit) rank read here — was
                # RETIRED in P11d: under the win-prob critic V IS sigmoid(logit), a tautology. The offline
                # `python -m main.scaffolding_gauge` still reads old shaped runs' traces.)

                # +WIN-PROB CALIBRATION (gen3_winprob_calibration_export_v1): the reliability half
                # of the head's diagnostics. Brier is a PROPER score and decomposes as
                # reliability − resolution + uncertainty, so it can stay flat while calibration
                # drifts; ECE/MCE/the per-bin gaps isolate the reliability term. Accumulated in BIN
                # COUNTS across the minibatches of EPOCH 0 (an ECE is nonlinear in the bin
                # populations — the mean of per-minibatch ECEs is not the pooled ECE) and folded
                # once at the end. Read-only: detached, no gradient, no RNG.
                if win_prob_on and epoch == 0:
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

                # Per-term auxiliary pull on the shared trunk, for the grad-balance probe — EVERY
                # active scaffold competes with policy/value there, so each is broken out INDIVIDUALLY
                # (not lumped into one "belief" norm) and the probe puts them on one common denominator
                # so policy/value/each-aux are mutually comparable + sum to ~1 (grad_balance.py). Only
                # the terms set this minibatch are included (a belief term is None on a zero-believed
                # minibatch; win_prob None when its head is off).
                aux_probe_terms: dict[str, th.Tensor] = {}
                # R1's terms (gen3_learner_micro_step_v1), in the inline fold's registration order;
                # a term is registered exactly when the inline fold had one (`present`).
                for _pn in ("species_belief", "move_belief", "move_latent", "spread_belief",
                            "nature_ev", "hp_type", "item_belief", "win_prob"):
                    if _pn in _mo.terms and _mpres.get(_pn, False):
                        aux_probe_terms[_pn] = _mo.terms[_pn]
                win_prob_term = aux_probe_terms.get("win_prob")
                opp_intent_term = (_mo.terms["opp_intent"]
                                   if "opp_intent" in _mo.terms and _mpres.get("opp_intent", False)
                                   else None)
                # THE FIGHT DETECTOR. Registering the intent term here is what produces
                # `grad/opp_intent_policy_cosine` — the angle between the intent objective's pull on
                # the shared trunk and the policy's. Under `--opp-intent-grad-mode detached` the
                # intent gradient cannot reach the trunk at all and this reads ~0 BY CONSTRUCTION,
                # which is the correct and expected value, not a bug. It only becomes informative
                # under `shaping`, which is precisely when you need to know.
                if opp_intent_term is not None:    aux_probe_terms["opp_intent"] = opp_intent_term
                aux_on = belief_aux_on or move_belief_on or move_latent_on
                # The belief terms only materialize on a minibatch with scored (believed = HIDDEN) slots;
                # wait for one so their shares aren't silently dropped from the single per-train() sample.
                # spread_belief scores on REVEALED slots (near-always present) so it does NOT gate this —
                # it rides whichever minibatch the probe samples (incl. the first, for a spread-only run).
                belief_present = any(
                    k in aux_probe_terms for k in ("species_belief", "move_belief", "move_latent")
                )
                # +K9(c) FAIL-CLOSED: the assembled loss must be finite before anything reads its graph
                # (the grad-balance / noise probes below, the backward). A NaN/Inf is a typed FATAL
                # naming the term(s), never a skipped step (`learner_gates.check_loss_finite`).
                # R1's finiteness rode the micro-batch's one host read; the full check (its own read)
                # runs only when the eager tail folded something onto the region's loss, or to NAME
                # the non-finite term(s) on a failure.
                if loss is not _mo.loss or not _mfinite:
                    check_loss_finite(loss, {"policy": _policy_grad_term, "entropy": _ent_term,
                                             "value": _vf_term, **aux_probe_terms},
                                      epoch=epoch, micro=len(pg_losses) - 1)
                if _ph is not None: _ph("loss")

                # +INSTRUMENTATION: sample the shared-trunk gradient balance on the first
                # minibatch (graph alive here; the probe uses read-only autograd.grad with
                # retain_graph, so loss.backward() below is unaffected). Skipped when the
                # extractor exposes no shared-trunk params (non-Gen3 policy).
                # Sample once per train(). When an aux is ON, wait for a minibatch that actually HAS
                # scored slots (belief_present) so the per-aux shares aren't silently dropped for the
                # call; when off, sample on the first minibatch as before.
                if (shared_trunk and diag.grad_balance and not grad_balance
                        and (not aux_on or belief_present)
                        and (not win_prob_on or win_prob_term is not None)):   # don't drop grad/win_prob_share
                    grad_balance = grad_balance_metrics(
                        # +PG-COEF: the probe measures the terms AS FOLDED — `_policy_grad_term`, not the
                        # raw `policy_loss` (at the 1.0 default they are the same tensor).
                        _policy_grad_term + self.ent_coef * _mo.entropy_loss,
                        # Under the win-prob critic the REAL critic term is the head's BCE; the scalar
                        # vf_coef·value_loss is dropped from the loss, so measure the BCE instead.
                        (win_prob_term if (critic_winprob and win_prob_term is not None)
                         else self.vf_coef * _mo.value_loss),
                        shared_trunk,
                        # Each ACTIVE scaffold broken out on the trunk: species/move/move-latent
                        # belief + win-prob (≈0 under read_only). Empty → RL-heads-only.
                        aux_terms=aux_probe_terms or None,
                    )

                if _ph is not None: _ph("probes")

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                # (computed inside R1; read with the micro-batch's one host read — a float32 scalar,
                # as sb3's `.cpu().numpy()` produced, so `np.mean` folds it in float32 exactly as before)
                approx_kl_div = np.float32(_approx_kl)
                approx_kl_divs.append(approx_kl_div)
                # +K9(c): a NaN/Inf KL can ride a FINITE loss (an overflowed ratio on a positive-
                # advantage row takes the clipped branch) and would pin the KL->LR controller's EMA
                # forever; the host read above is the micro-batch's own, so the check is free.
                check_kl_finite(float(approx_kl_div), epoch=epoch)

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
                if _ph is not None: _ph("kl")

                # Optimization step. +GRAD-ACCUM: accumulate the 1/accum-scaled gradient (accum
                # micro-batches of size batch_size sum to the exact (batch_size·accum) gradient) and
                # step only when the group is full. accum==1 ⇒ one step per minibatch (upstream).
                # +NOISE-SCALE PER-TERM: take the per-group gradients LAST, while the graph is
                # still alive and `.grad` still holds only what previous micro-batches put there.
                # `autograd.grad` writes no `.grad`, so the accumulation below is untouched.
                _ntg.flush_micro()
                if _ph is not None: _ph("noise_probe")
                (loss / accum).backward()
                micro_in_group += 1
                if _ph is not None: _ph("backward")
                # +INSTRUMENTATION: per-edge-family liveness, sampled ONCE per train() and read
                # HERE because it wants `.grad` populated but not yet cleared by the optimizer
                # step. Parameters only — no forward touched, so the hot path pays nothing.
                if diag.liveness and not edge_metrics:
                    edge_metrics = edge_family_metrics(self.policy.features_extractor)
                # +INSTRUMENTATION: the same read for the zero-init POINTER CELLS (switch-branch,
                # pair-outcome move/switch, conditional-threat). Same window, same reason: a cell
                # that never comes off its zero init is invisible without it.
                if diag.liveness and not cell_metrics:
                    cell_metrics = cell_family_metrics(self.policy.features_extractor)
                if _ph is not None: _ph("probes")
                # +NOISE-SCALE: after the FIRST micro-batch of group 0 (epoch 0), .grad holds exactly
                # g_1/accum (this micro's gradient, scaled) → ‖g_1‖² = accum²·‖.grad‖². The single
                # micro-batch (B=batch_size) sample for the noise-scale estimate.
                if accum >= 2 and epoch == 0 and micro_in_group == 1 and noise_g_small_sq is None:
                    noise_g_small_sq = (accum ** 2) * self._global_grad_sq(self.policy.parameters())
                if _ph is not None: _ph("noise_base")
                if micro_in_group == accum:
                    # +INSTRUMENTATION: pre-clip total grad norm (per step). +K9(c): a NaN/Inf norm is a
                    # typed FATAL BEFORE the in-place clip and the optimizer step (`learner_gates`).
                    grad_norm = clip_grad_norm_checked(self.policy, self.max_grad_norm, epoch=epoch)
                    grad_norms.append(grad_norm)
                    # +NOISE-SCALE: the accumulated group gradient (B=batch_size·accum) — pre-clip norm
                    # from clip_grad_norm_. Captured on group 0 (same data as the micro-batch above).
                    if accum >= 2 and epoch == 0 and noise_g_big_sq is None:
                        noise_g_big_sq = grad_norm * grad_norm
                    self.policy.optimizer.step()
                    self.policy.optimizer.zero_grad()
                    micro_in_group = 0
                if _ph is not None: _ph("optim")

                # +CAPACITY TELEMETRY: LAST in the minibatch body, deliberately — after the loss
                # fold, after `loss.backward()`, after the optimizer step. Nothing it does can
                # reach `loss` or `.grad` from here, which is the point: the placement is the
                # proof. It costs one `if` per minibatch when the flag is off.
                if capacity is not None:
                    self._capacity_observe(capacity, rollout_data, actions, advantages,
                                           shared_trunk, clip_range, cap_features)
                if _ph is not None: _ph("capacity")

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
                grad_norms.append(clip_grad_norm_checked(self.policy, self.max_grad_norm, epoch=epoch))
                self.policy.optimizer.step()
                self.policy.optimizer.zero_grad()
                micro_in_group = 0

            # +PER-EPOCH: close this epoch's pair. Placed AFTER the minibatch loop so a KL early stop
            # (which `break`s out of it) still records the partial epoch, tripping minibatch included.
            if approx_kl_divs:
                epoch_approx_kl.append(float(np.mean(approx_kl_divs)))
            if len(clip_fractions) > _epoch_cf_start:
                epoch_clip_fraction.append(float(np.mean(clip_fractions[_epoch_cf_start:])))

            self._n_updates += 1
            if _ph is not None: _ph("epoch_end")
            if not continue_training:
                break
        if _devb is not None and _devb.nbytes:   # gen3_cuda_ledger_v1: the update's device copy
            self.logger.record("lifecycle/device_batch_mib", _devb.nbytes / (1 << 20))
        _devb_uninstall(self.rollout_buffer)   # +K8: the device copy is the update's, not the run's

        # +CAPACITY TELEMETRY: the once-per-train() half — fold the per-minibatch canary/cosine
        # samples and (on cadence) run the frozen probe batch through the extractor for the
        # feature-velocity read. Outside the epoch loop, no gradient, `{}` when the flag is off.
        if capacity is not None:
            capacity_metrics = self._capacity_finish(capacity)

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        # +PER-EPOCH (gen3_ppo_per_epoch_diag_v1): `train/approx_kl` above is the LAST epoch's mean
        # (stock SB3 resets its list per epoch) while `train/clip_fraction` pools EVERY epoch — the
        # per-epoch series is what shows how the policy drifts across the n_epochs passes.
        for _k, _kl in enumerate(epoch_approx_kl):
            self.logger.record(f"train/approx_kl_epoch_{_k}", _kl)
        for _k, _cf in enumerate(epoch_clip_fraction):
            self.logger.record(f"train/clip_fraction_epoch_{_k}", _cf)
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
        self._record_signal_metrics(signal_metrics)
        self._record_noise_scale_metrics(accum, noise_g_small_sq, noise_g_big_sq, _ns_terms)
        self._record_head_metrics(belief_metrics, win_prob_metrics, calib_all, calib_contested,
                                  critic_winprob, win_prob_on, grad_balance)
        self._record_capacity_metrics(capacity_metrics, aux_metrics)
        self._record_ridealong_metrics(ridealong_acc)
        # +INSTRUMENTATION: LAST line of train(), so it bounds the whole call — the honest
        # denominator for `train/noise_per_term_ms` and for every other probe's cost claim.
        if _ph is not None: _ph("logging")
        self.logger.record("train/train_ms", 1000.0 * (time.perf_counter() - _t_train0))
