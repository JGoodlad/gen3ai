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
from agents.training.instrumented_ppo.distill_anchor import distill_anchor_step
from agents.training.instrumented_ppo.distill_terms import DistillTerms
from agents.training.instrumented_ppo.hparams import PpoHyperparameters
from agents.training.instrumented_ppo.learner_gates import (   # K9(b) python path + K9(c)
    behaviour_gate_mode,
    check_behaviour_first_micro,
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
from agents.model.compile_trainer import eager_extractor as _eager_fe  # gen3_compile_sentinel_v1


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
        _ms.value_loss_from_se,
        _ms._flag_entropy,
        TrainSetup._micro_static,
        TrainSetup._micro_var,
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
                              RideAlongTerms,
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

          1. `loss = pg_term + ent_coef * ent_loss_used + vf_term`   (the upstream PPO loss;
             `pg_term` is the UNSCALED `policy_loss` tensor at `policy_grad_coef == 1.0` — the default,
             byte-identical to upstream — else `policy_grad_coef * policy_loss` (`--policy-grad-coef`; 0.0 removes
             the policy-gradient term alone, the arm-F pure-distill/aux phase — entropy and the
             value term keep their own coefficients). `_value_loss_from_se` is the only other
             delta, and at `value_tail_weight == 0` it is `F.mse_loss` byte-for-byte)
          2. the BELIEF bank — species/moves aux, opponent-intent (+ the set-valued beta term),
             move belief, spread belief, nature/EV, HP-type, item belief, move-latent
          3. (3a) the WIN-PROB BCE — the last R1 term — then (3b, the tail's first) the dense aux
             head and the CF-TWIN on-policy mirror
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
        # returns, every float label key) are finite, ONCE per update and BEFORE PopArt's advance —
        # which rewrites `value_net` outside the optimizer — or any forward (`learner_gates`).
        check_buffer_finite(self.rollout_buffer)
        # +K9(b) / STALENESS (M5 Lane G): before any optimizer step; a no-op unless --behaviour-check.
        # A buffer that carries per-row policy versions (the Rust collector) gets Lane G's pre-loop
        # probe (its own forward, age-bucketed); one that does not (python env core: every row is
        # current) gets the in-loop gate on the FIRST micro-batch's own forward — never both
        # (`learner_gates.behaviour_gate_mode`).
        _bgate_mode = behaviour_gate_mode(self)
        if _bgate_mode == "probe":
            self._behaviour_probe()
        _bgate_pending = _bgate_mode == "in_loop"

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
        # Unpacked back into locals so the EAGER TAIL below reads as it was written; the R1 terms'
        # flags (the belief rows, the fork mask, the policy-gradient coefficient) are read by
        # `_micro_static` instead — the region's static flags.
        _f = self._resolve_fold_flags()
        belief_aux_on, move_belief_on = _f.belief_aux_on, _f.move_belief_on
        move_latent_on = _f.move_latent_on
        critic_winprob, win_prob_on = _f.critic_winprob, _f.win_prob_on
        scaffolding_on, value_from_dist = _f.scaffolding_on, _f.value_from_dist
        value_dist_on, search_teacher_on = _f.value_dist_on, _f.search_teacher_on
        opd_on, distill_on = _f.opd_on, _f.distill_on
        distill_rows_in_buffer = _f.distill_rows_in_buffer
        td_aux_on, cf_buffer, cf_winprob_on = _f.td_aux_on, _f.cf_buffer, _f.cf_winprob_on
        cf_evid_on, cf_twin_on, cf_shadow_on = _f.cf_evid_on, _f.cf_twin_on, _f.cf_shadow_on
        dense_aux_on = _f.dense_aux_on
        q_winprob_on, q_onpolicy_on, cf_any_on = _f.q_winprob_on, _f.q_onpolicy_on, _f.cf_any_on
        # +WIN-PROB STRATA (gen3_winprob_strata_weight_v1) — the per-opponent-CLASS weights for the
        # win-prob BCE, computed ONCE here over the WHOLE rollout buffer and held constant for
        # every epoch and minibatch of this call. Per-BUFFER and not per-minibatch on purpose: the
        # normalisation that makes the mean weight 1 is only meaningful over the population the
        # frequencies were measured on, and a per-minibatch recomputation would make the class
        # balance itself a sampling-noise term. Read here rather than in `_resolve_fold_flags`
        # because it needs the buffer, and `_align_opp_intent_labels` above has already run — the
        # only writer of `opp_class` in this call (a semantic no-op: the class is constant within
        # an episode). Under `--critic winprob` only; every other configuration passed None and
        # takes the unweighted expression unchanged.
        strata_w = None
        if win_prob_on and critic_winprob and float(getattr(self, "win_prob_strata_weight", 0.0)) > 0.0:
            _sb = self.rollout_buffer.observations
            _sout = self._win_prob_strata_weights(
                _sb.get("opp_class"), _sb.get("win_mask"), float(self.win_prob_strata_weight))
            if _sout is not None:
                # `_smetrics` arrives even when no weighting applies (one class present, labels
                # not filled yet), carrying `strata_active` 0/1 — so an ABSENT `win_prob/strata_*`
                # family means the flag is off and cannot be confused with a plumbing break.
                strata_w, _smetrics = _sout
                for _sk, _sv in _smetrics.items():
                    win_prob_metrics.setdefault(_sk, []).append(float(_sv))
        # +WIN-PROB ANCHOR WEIGHT (gen3_winprob_rollout_weight_v1) — the per-ROW weight on the
        # rollout-ANCHORED rows of the win-prob BCE. The vector itself is built per rollout by
        # `WinProbLabelCallback._apply_rollout_weight` and rides the buffer's own `win_row_w` obs
        # key (the only carrier that survives `get()`'s shuffle aligned to its row), so all that is
        # decided here is WHETHER to read it. Both halves of the predicate: the weight above 1.0,
        # and the rollout fraction that produces the anchors it weighs — a weight with no anchors
        # would be a vector of ones, and reading it would cost a gather per minibatch to change
        # nothing. Under `--critic winprob` only, exactly like the strata weight.
        rollout_weight_on = (
            win_prob_on and critic_winprob
            and float(getattr(self, "win_prob_rollout_weight", 1.0) or 1.0) > 1.0
            and float(getattr(self, "win_prob_rollout_target", 0.0) or 0.0) > 0.0
            and "win_row_w" in self.rollout_buffer.observations)
        # +WIN-PROB λ-RETURN (gen3_winprob_lambda_v1) — the family is COMPUTED in
        # `WinProbLabelCallback._on_rollout_end` (it needs the buffer's [n_steps, n_envs] shape,
        # before `get()` shuffles it flat, and the same `model._last_obs` forward SB3's own GAE
        # bootstrap takes) and stashed on the model. Folded in here so it rides the ordinary
        # `win_prob/` prefix, and only when the recursion actually RAN: an absent
        # `win_prob/lambda_*` family means λ = 1.0 (off, and the targets are the terminal bit) and
        # nothing else. Cleared at every `_on_rollout_start`, so it can never be a stale rollout's.
        _lam_metrics = getattr(self, "_win_prob_lambda_metrics", None)
        if _lam_metrics:
            for _lk, _lv in _lam_metrics.items():
                # A NaN is what an empty slice reports (no scored rows at all); it is a real state
                # and it is REPORTED by omitting the tag, never by logging a NaN that TensorBoard
                # renders as a gap in a series that also has honest gaps.
                if float(_lv) == float(_lv):
                    win_prob_metrics.setdefault(_lk, []).append(float(_lv))
        # +DENSE-AUX (gen3_dense_aux_v1) — the PLUMBING half of the `win_prob/aux_*` family is
        # COMPUTED in `DenseAuxLabelCallback._on_rollout_end` (it needs the buffer's
        # [n_steps, n_envs] shape, before `get()` shuffles it flat) and stashed on the model.
        # Folded in here so it rides the ordinary `win_prob/` prefix. Cleared at every
        # `_on_rollout_start`, so it can never be a stale rollout's; an absent `win_prob/aux_*`
        # family means the head is off, and nothing else.
        _daux_metrics = getattr(self, "_dense_aux_metrics", None)
        if _daux_metrics:
            for _dk, _dv in _daux_metrics.items():
                if float(_dv) == float(_dv):      # a NaN is REPORTED by omission, never logged
                    win_prob_metrics.setdefault(_dk, []).append(float(_dv))
        # +WIN-PROB R-ROLLOUT TARGETS (gen3_winprob_rollout_target_v1) — computed in
        # `WinProbLabelCallback._on_rollout_end` (it needs the buffer's [n_steps, n_envs] shape and
        # it has to BLOCK on the continuations before the epochs begin) and stashed on the model.
        # Folded in here so it rides the ordinary `win_prob/` prefix, and only when the labelling
        # actually RAN: an absent `win_prob/rollout_*` family means the fraction is 0.0 (or the run
        # has no cf_records ring, which announces itself once) and nothing else. Cleared at every
        # `_on_rollout_start`, so it can never be a stale rollout's.
        _roll_metrics = getattr(self, "_win_prob_rollout_metrics", None)
        if _roll_metrics:
            for _rk, _rv in _roll_metrics.items():
                if float(_rv) == float(_rv):      # a NaN is an empty slice; omit, never log it
                    win_prob_metrics.setdefault(_rk, []).append(float(_rv))
        # +FORK ARM (gen3_fork_v1) — computed in `ForkArmCallback._on_rollout_end` (it needs the
        # buffer's [n_steps, n_envs] shape and it BLOCKS on the branch continuations before the
        # epochs begin) and stashed on the model. Recorded under its OWN `fork/` prefix rather than
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
        cf_metrics: dict[str, list[float]] = {}
        cf_evid_metrics: dict[str, list[float]] = {}
        cf_twin_metrics: dict[str, list[float]] = {}     # +CF-TWIN (gen3_cf_twin_heads_v1)
        cf_shadow_metrics: dict[str, list[float]] = {}   # +CF-SHADOW (gen3_cf_twin_heads_v1)
        q_metrics: dict[str, list[float]] = {}           # +Q-WINPROB (gen3_q_winprob_head_v1)
        # +RIDE-ALONG (gen3_ridealong_heads_v1): the detached heads' `ridealong/*` sink.
        ridealong_acc = RideAlongAccumulator()
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
        diag = _p.diag   # gen3_diagnostics_cadence_v1: which optional probes run on THIS call
        # +PER-EPOCH (gen3_ppo_per_epoch_diag_v1): one (approx_kl, clip_fraction) pair per epoch the
        # loop actually ran, folded from the SAME per-minibatch numbers the stock tags already average
        # — no extra forward, no extra device sync. An early KL stop leaves fewer than n_epochs rows.
        epoch_approx_kl: list[float] = []
        epoch_clip_fraction: list[float] = []
        # +R1 (gen3_learner_micro_step_v1): the region's static flags + per-update tensors, resolved
        # ONCE, and the per-update lists its diagnostics are routed into (by name).
        _micro_st = self._micro_static(_f, popart, strata_w, rollout_weight_on)
        _micro_var = self._micro_var(_micro_st, strata_w)
        _ppo_lists = {"pg_losses": pg_losses, "clip_fractions": clip_fractions,
                      "value_losses": value_losses, "entropy_losses": entropy_losses,
                      "vf_clip_fractions": vf_clip_fractions,
                      "defent_flag_fracs": defent_flag_fracs, "defent_boost_eff": defent_boost_eff,
                      "defent_ent_flagged": defent_ent_flagged,
                      "defent_ent_unflagged": defent_ent_unflagged,
                      "baitent_flag_fracs": baitent_flag_fracs, "baitent_boost_eff": baitent_boost_eff,
                      "baitent_ent_flagged": baitent_ent_flagged,
                      "baitent_ent_unflagged": baitent_ent_unflagged}
        if _ph is not None: _ph("setup")
        # +K8 (gen3_device_batches_v1): every micro-batch gathered from ONE device copy of the
        # flattened buffer (made at the first micro-batch) instead of a host gather + H2D copy each
        # — the same permutation draw, bit-identical batches; removed after the epoch loop.
        _devb_install(self.rollout_buffer)
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
                _mo = self._micro_region()(self.policy, popart, rollout_data.observations,
                                         rollout_data.actions, rollout_data.action_masks,
                                         rollout_data.old_log_prob, rollout_data.old_values,
                                         rollout_data.advantages, rollout_data.returns,
                                         _micro_var, _micro_st)
                actions = (rollout_data.actions.long().flatten() if _micro_st.discrete
                           else rollout_data.actions)
                values, log_prob = _mo.values, _mo.log_prob
                advantages = _mo.advantages
                loss = _mo.loss
                # The distill / anchor / ride-along readers' stash, built OUTSIDE the region.
                if _mo.logp is not None:
                    self.policy._last_pi_distribution = _MaskedPi(_mo.logp, _mo.masks_bool)
                # +INSTRUMENTATION: effective rank of the trunk / value_cls / policy / vf reps, ONCE per
                # train() (first minibatch) — read from R1's OWN forward's stashes (K8,
                # gen3_rank_device_v1: no second forward; the spectra on the device, one host read).
                # HERE, before the tail: TD-aux and the cf block re-forward and overwrite the stashes.
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

                # +CAPACITY: snapshot THIS forward's `value_pooled` before the TD-aux / CF folds
                # replace the stash. Detached in the snapshot itself, so nothing downstream can
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

                # +K9(b), python path: the first micro-batch of epoch 0 runs before any optimizer step,
                # so its recomputed log π must equal the rollout's stored behaviour log-prob (one host
                # read per update, no forward of its own; `learner_gates` module docs).
                if _bgate_pending:
                    check_behaviour_first_micro(self, log_prob, rollout_data.old_log_prob,
                                                actions, rollout_data.action_masks)
                    _bgate_pending = False

                # +DENSE-AUX (gen3_dense_aux_v1, v117): the DENSE AUXILIARY loss — per-slot
                # survival, per-slot final HP and turns-left, all END-OF-BATTLE facts back-filled
                # to every state the way the win bit is. It is arm 9 of the critic ladder and
                # KataGo's (Wu 2019) answer to a one-bit terminal signal: ~10% of that bit's
                # variance lies between opponents, so the head shrinks the weak axes toward the
                # marginal; 25 per-ENTITY targets put gradient on those axes directly.
                #
                # The head is NOT in the forward (the `CfEvidentialHead` contract), so it is
                # applied here to the `value_pooled` this minibatch's `evaluate_actions` stashed.
                # 🚨 That tensor is NOT detached, and that is the arm: the aux gradient reaches
                # the shared trunk exactly as the win-prob loss does under `shaping` (which
                # `--critic winprob` implies). `pi` is untouched in the only sense that matters
                # for a readout — the head's OUTPUT never enters the policy path, at any weight.
                # Folded as an `aux` term at `--win-prob-dense-aux`, never at `vf_coef`: there is
                # one critic and these are not it.
                dense_aux_term = None
                if dense_aux_on:
                    _fe = self.policy.features_extractor
                    _pooled = _fe.last_value_pooled
                    if _pooled is not None:
                        _da_out = self._dense_aux_loss(
                            _fe.dense_aux_head(_pooled),
                            rollout_data.observations.get("aux_target"),
                            rollout_data.observations.get("aux_mask"),
                        )
                        if _da_out is not None:
                            _da_loss, _da_m = _da_out
                            dense_aux_term = self.win_prob_dense_aux * _da_loss
                            loss = loss + _ntg.add("aux", dense_aux_term)
                            for _dk, _dv in _da_m.items():
                                win_prob_metrics.setdefault(_dk, []).append(float(_dv))

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
                        # EAGER: a (obs, action_mask)-only key set at a ring-sized batch is a
                        # signature the compile lock would kill (gen3_compile_sentinel_v1).
                        with _eager_fe(getattr(self.policy, "features_extractor", None)):
                            _dist = self.policy.get_distribution(_td["obs_dict"])
                        _st = self._searchteacher_loss(
                            _dist.distribution.logits, _td["action_mask"], _td["better_action"],
                            _td["advantage"], beta_awr=self.search_teacher_beta)
                        if _st is not None:
                            _st_loss, _st_m = _st
                            searchteacher_term = self.search_teacher_coef * _st_loss
                            if self.search_teacher_value_coef != 0.0:   # OFF by default (soundness)
                                with _eager_fe(getattr(self.policy, "features_extractor", None)):
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
                            with _eager_fe(getattr(self.policy, "features_extractor", None)):   # see search-teacher
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
                # The rank probe reads R1's stashes right after R1 (above), so it is unaffected.
                # OFF → skipped (loss byte-identical).
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
                # gen3_dense_aux_v1: `grad/dense_aux_share` is the VERIFICATION that the dense
                # targets actually pull the shared trunk — the one number that separates "the arm
                # ran" from "the arm did what it was built to do". It is a live (un-detached)
                # readout, so unlike `grad/cf_evidential_share` it must NOT read 0.
                if dense_aux_term is not None:     aux_probe_terms["dense_aux"] = dense_aux_term
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
                        _policy_grad_term + self.ent_coef * _mo.entropy_loss,
                        # Phase B: the REAL critic term is the CE (value_dist_term); the scalar
                        # vf_coef·value_loss is dropped from the loss, so measure the CE instead.
                        (win_prob_term if (critic_winprob and win_prob_term is not None)
                         else value_dist_term if (value_from_dist and value_dist_term is not None)
                         else self.vf_coef * _mo.value_loss),
                        shared_trunk,
                        # Each ACTIVE scaffold broken out on the trunk: species/move/move-latent
                        # belief + win-prob (≈0 under read_only) + value-dist. Empty → RL-heads-only.
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
                # +DISTILL-GRAD-PROJECT: the removal vector is computed while the graph is alive
                # (read-only `autograd.grad`, no `.grad` written) and applied to `.grad` immediately
                # after the real backward — so `.grad` goes from `g_ppo + g_distill` to
                # `g_ppo + P_perp g_distill` with PPO's contribution bit-for-bit untouched. Both are
                # no-ops unless `--distill-anchor-mode grad_project`.
                _dgp.before_backward(self.policy, rollout_data)
                (loss / accum).backward()
                _dgp.after_backward(accum)
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
        _devb_uninstall(self.rollout_buffer)   # +K8: the device copy is the update's, not the run's

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
        self._record_ridealong_metrics(ridealong_acc)
        # +INSTRUMENTATION: LAST line of train(), so it bounds the whole call — the honest
        # denominator for `train/noise_per_term_ms` and for every other probe's cost claim.
        if _ph is not None: _ph("logging")
        self.logger.record("train/train_ms", 1000.0 * (time.perf_counter() - _t_train0))
