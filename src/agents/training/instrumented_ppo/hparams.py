"""`PpoHyperparameters` — every knob `train_rl_agent` sets on the model AFTER construction.

A mixin of class attributes, not a config object, because that is exactly what these are: SB3
constructs the algorithm and the entry point then assigns `model.<name> = ...`. Each carries the
comment that says what it costs, what OFF means, and whether it is version-locked — which is the
reason they are worth 300 lines and worth keeping in one place.

The class defaults are all no-ops, so a smoke, a unit test, or a frozen eval/pool
opponent that never sets them runs the byte-identical-to-upstream loss.
"""
class PpoHyperparameters:
    """The after-construction knobs, and the save-exclusion list that goes with them."""

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
    # +PER-TERM NOISE SCALE (`noise_scale_terms.py`): also estimate the McCandlish critical batch
    # SEPARATELY per loss group (policy / value / entropy / aux), not just on the total
    # gradient. Pure DIAGNOSTIC — it takes read-only `autograd.grad` snapshots and never touches
    # `.grad`, the loss, or the optimizer step, so it is neither version-locked nor recorded. Only
    # does anything when `grad_accum_steps >= 2` (it needs the same two batch sizes the total
    # estimator does). An ENV knob rather than a CLI flag — `$GEN3AI_NOISE_SCALE_PER_TERM=0`
    # disables it and wins over this default (see `noise_scale_terms.per_term_enabled`), because a
    # switch that changes no training math should not have to survive a launcher resume's argv.
    noise_scale_per_term: bool = True
    # +DIAGNOSTICS CADENCE (gen3_diagnostics_cadence_v1, `--diagnostics-every`, config v124): the
    # OPTIONAL probes (per-term noise, grad balance, rank, edge/cell liveness) run on every Nth
    # update only — `diagnostics_cadence.py` owns the set, the phase and the first-update rule. The
    # class default 1 is every update (what a bare construction and every pre-v124 run did); the
    # CLI resolves a FRESH run to `DIAGNOSTICS_EVERY_DEFAULT` and a flagless resume to the recorded
    # value. Changes no training math (bit-identity pinned by `diagnostics_cadence_test.py`).
    diagnostics_every: int = 1
    # The two LOAD-BEARING exemptions, DERIVED from the run's flags by `apply_training_hparams`:
    # a consumer that reads the probe every update keeps it every update.
    rank_probe_every_update: bool = False     # `--rank-tripwire warn|abort`
    noise_terms_every_update: bool = False    # `--adaptive-batch policy`
    # POLICY-GRADIENT term weight (gen3_policy_grad_coef_v1, `--policy-grad-coef`). Multiplies ONLY `policy_loss`
    # (the clipped PPO surrogate) in the loss fold — never entropy, never the value term, never
    # any aux. 1.0 (default) takes the UNSCALED `policy_loss` tensor itself, so the loss
    # expression is byte-identical to upstream; 0.0 removes the policy-gradient contribution
    # entirely (every other term keeps its own coefficient).
    # TRAINING-only (scales a loss, never a forward pass) -> NOT version-locked / NOT in
    # check_compatible; recorded on ModelVersion for provenance + flagless-resume read-back,
    # like td_aux_coef.
    policy_grad_coef: float = 1.0

    # Set by train_rl_agent after construction. The hidden-opponent belief
    # aux-loss coefficient: opp_belief_aux_coef * (species_CE + moves_weight·moves_BCE) over the
    # believed opp slots is added to each minibatch loss. 0.0 = OFF (no aux term, byte-identical loss).
    # A TRAINING hparam (affects the loss only, never a forward pass) → NOT version-locked / NOT in
    # check_compatible (treat like ent_coef; a frozen eval/pool opponent never runs train()).
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

    # TD-CONSISTENCY AUXILIARY (gen3_td_consistency_aux_v1; the live-training half of
    # designs/research_state/levers/td_consistency_aux.md, ledger C5). The per-state value MSE never
    # constrains ADJACENT-state differences, so ΔV inherits ~2x the state noise exactly where the truth
    # is nearly constant. This adds an explicit Bellman residual over CONTIGUOUS pairs drawn from the
    # rollout buffer's own [n_steps, n_envs] structure (PPO's minibatches are shuffled and contain no
    # adjacent pairs at all):
    #     td_aux_coef * mean[ ( V(s_t) - r_t - gamma*V(s_{t+1}) )^2 ]
    # 0.0 = OFF and the whole block is skipped (loss byte-identical to today — `_td_aux_term` is not
    # even called, so a broken sampler could not perturb an off run). Rung-1's pre-registered band is
    # 1.0-3.0; lambda <= 0.1 measured WORSE than control, so do not use the small-coef regime.
    # TRAINING-only (scales the loss, never a forward pass) -> NOT version-locked / NOT in
    # check_compatible; recorded on ModelVersion for provenance + flagless-resume read-back, like
    # opp_belief_aux_coef.
    td_aux_coef: float = 0.0
    # Process-local RNG for the contiguous-pair sampler, seeded from the global numpy stream at first
    # use so a seeded run stays reproducible. Not saved (like _noise_ema_*).
    _td_aux_rng = None

    opp_intent_coef: float = 0.0
    # SET-VALUED partial credit on beta's belief-miss rows (see `set_valued_switch_loss`). Scales
    # ON TOP of opp_intent_coef, so it is a share of the intent budget rather than a second one.
    # 0.0 = OFF and the loss is byte-identical; training-only, resume-mutable (no module changes).
    beta_setvalued_coef: float = 0.0
    # gen3_intent_label_bot_weight_v1: per-sample weight on α/β label rows whose opponent was a
    # heuristic BOT (`opp_class == 0`); every other class stays 1.0. 1.0 = OFF and the loss is
    # bit-identical (the unweighted `cross_entropy` call is taken unchanged). Training-only,
    # resume-mutable. Applies to the INTENT losses only — never to the BeliefBank rows, which are
    # team truth rather than behaviour. See `agents.model.opp_intent.intent_losses`.
    intent_label_bot_weight: float = 1.0
    # gen3_winprob_strata_weight_v1: OPPONENT-STRATIFIED weighting of the WIN-PROB BCE, in [0, 1].
    # 0.0 = OFF and the loss is bit-identical (the unweighted masked mean is taken unchanged);
    # 1.0 = each opponent CLASS contributes to the objective in equal proportion rather than in
    # episode proportion (inverse-frequency `f ** -s`, capped, renormalised so the mean weight over
    # the buffer is 1). It exists because only ~10-14% of the terminal label's variance lies BETWEEN
    # (cycle, opponent) cells, so the head buys its resolution from the board instead
    # (`winprob_head_refit_2026-09-09`). Requires `--critic winprob` (refused otherwise, never a
    # silent no-op). Training-only, resume-mutable; scales a loss, touches no forward pass.
    win_prob_strata_weight: float = 0.0

    # COUNTERFACTUAL WIN-PROB GROUNDING (gen3_cf_label_plumbing_v1; G3 of
    # designs/ai_v10/design_counterfactual_value_grounding.md, rung R1). A background producer
    # re-rolls recorded training decisions to termination and drops tight Monte-Carlo P(win) labels
    # as JSONL; `_cf_buffer` (an `agents.training.cf_label_buffer.CfLabelBuffer`, attached
    # externally like `_async_rollout`) ingests them, and this coefficient folds
    #     cf_winprob_coef * BCE( win_head(value_pooled(s)), MC_label(s) )
    # over its OWN sample and its OWN extractor forward — the labelled states are OFF-DISTRIBUTION
    # w.r.t. this rollout, so they cannot ride the minibatch.
    #
    # 0.0 = OFF and the whole block is skipped: no poll, no sample, no forward, loss byte-identical.
    # TRAINING-only (a loss weight; no forward/weight-shape change) → NOT version-locked, NOT in
    # check_compatible, resume-mutable — the `td_aux_coef` class.
    cf_winprob_coef: float = 0.0
    # THE SAFE STAGE, and the DEFAULT. True → the head's input `value_pooled` is stop-grad'd for
    # this term, so it trains the win-prob head's own params ONLY and cannot perturb the trunk (a
    # pure, risk-free delivery — `grad/cf_winprob_share` reads exactly 0.0 by construction). False
    # → the ground-truth objective also shapes the shared trunk. Independent of the extractor's own
    # `win_prob_mode` read_only/shaping split, which governs the ON-POLICY win-prob BCE, not this.
    cf_head_only: bool = True
    # Set by train_rl_agent alongside the buffer; the buffer itself owns the bound (this is the
    # value it was constructed with, kept here only for the record).
    cf_label_lag_steps: int = 0
    # gen3_cf_binomial_likelihood_v1: WHICH likelihood the scalar cf term uses.
    #   'binomial' (the DEFAULT) — the exact binomial NLL of the row's win COUNT:
    #       w = round(label*n), NLL_i = -[w*log q + (n-w)*log(1-q)], folded as sum(NLL)/sum(n).
    #     Each row is weighted by its evidence, so an R=16 label pulls 4x an R=4 label. That is not
    #     a heuristic weighting — it is what the likelihood of the data actually is, and the flat
    #     form was implicitly asserting every label carries one observation.
    #   'bce' — the flat per-row BCE on the scalar `label`, i.e. the pre-2026-08-22 behaviour, kept
    #     as the A/B arm. The two are EXACTLY equal when every n == 1 (a 1-rollout label is already
    #     0 or 1, so the round is the identity and sum(n) == B).
    # TRAINING-only (a loss FORM, no forward and no weight shape) -> not version-locked, not in
    # check_compatible, NOT read back on a flagless resume: the `td_aux_coef` class.
    cf_label_likelihood: str = "binomial"
    # gen3_cf_evidential_head_v1: the EVIDENTIAL Beta term's weight. Folds
    #     cf_evidential_coef * ( BetaBinomialNLL(alpha,beta; w,n)/sum(n)
    #                            + cf_evidential_reg * mean KL(Beta(a,b) || Beta(1,1)) )
    # over the SAME sampled rows and the SAME extractor forward as the scalar cf term. 0.0 = OFF and
    # the whole block is skipped. TRAINING-only; the STRUCTURAL half is the extractor's
    # `cf_evidential` kwarg (v98), which decides whether the head's params exist at all.
    cf_evidential_coef: float = 0.0
    # The evidential-overconfidence guard's weight, RELATIVE to the NLL (it sits inside the coef).
    # Evidential heads inflate alpha+beta without bound on locally-consistent data; a small pull
    # back toward the uninformative Beta(1,1) is the standard remedy.
    cf_evidential_reg: float = 1e-3
    # gen3_cf_twin_heads_v1: the TWIN win-prob heads' cf weight — the owner-authorized amendment to
    # the signed R1 pre-registration (ledger 2026-08-22 evening, "Three owner sign-offs" item 3).
    # ONE coefficient for BOTH twins on purpose: B and C must differ in their LABEL STREAM and in
    # nothing else, and two knobs would eventually be set to two numbers.
    #
    #   head A (`win_head`)       : the on-policy single-outcome BCE ONLY — the CONTROL, untouched
    #   head B (`cf_twin_head_b`) : A's loss + cf_twin_coef * NLL(B; SINGLE-OUTCOME labels, n=1)
    #   head C (`cf_twin_head_c`) : A's loss + cf_twin_coef * NLL(C; TIGHT-MC labels, n=R)
    #
    # B−A isolates COVERAGE (the same loss form on extra states); C−B isolates pure VARIANCE
    # REDUCTION (the same states, the same form, a tighter target). The twins' half of "A's loss" is
    # folded at head A's own weight (1.0), not at this coefficient, so all three heads carry a bit-identical
    # A-term. 0.0 = OFF and the WHOLE twin block is skipped (including the on-policy mirror), so a
    # built-but-unused pair of heads leaves every parameter update byte-identical.
    # TRAINING-only (a loss weight) → the `td_aux_coef` class; the STRUCTURAL half is the extractor's
    # `cf_twin_heads` kwarg (v99), which decides whether the heads' params exist at all.
    cf_twin_coef: float = 0.0
    # gen3_cf_twin_heads_v1: the SHADOW CRITIC's weight. Folds
    #     cf_shadow_coef * masked-MSE( shadow(value_pooled.detach()), normalize(mc_return) )
    # over the same sampled rows and the same extractor forward. The head never computes an
    # advantage and never enters GAE — it is the staged PROMOTION PATH for critic surgery (a critic
    # ROUTE change owes C4), so what it produces is evidence, not a training change to the critic.
    # 0.0 = OFF, whole block skipped. TRAINING-only; the STRUCTURAL half is `cf_shadow_critic` (v99).
    cf_shadow_coef: float = 0.0
    # ---- gen3_q_winprob_head_v1 — THE PER-ACTION WIN-PROB HEAD (`q_winprob/*`) -----------------
    # The COUNTERFACTUAL term's weight. Folds
    #     q_winprob_coef * masked-binomial-NLL( q_head(pointer tokens), q_labels ; Sum(mask*n) )
    # over the same sampled rows and the same extractor forward as every cf term. 0.0 = OFF, whole
    # block skipped. TRAINING-only; the STRUCTURAL half is `q_winprob_mode` (v107), which decides
    # whether the head's params exist at all.
    q_winprob_coef: float = 0.0
    # The WEAK on-policy fallback's weight — SEPARATE on purpose. 🚨 It labels ONE action of
    # eleven, drawn from the policy's own choices (measured preferred-alternative rate p≈0.002), so
    # it teaches the head where the policy already goes and leaves it confidently wrong on the
    # never-tried moves. That is the exact failure the counterfactual stream exists to avoid, which
    # is why this defaults to 0.0 and why one coefficient could never have covered both.
    q_winprob_onpolicy_coef: float = 0.0
    # ---- gen3_capacity_telemetry_v1 — LIVE CAPACITY TELEMETRY (`capacity/*`) -------------------
    # The master switch for all three probes (plasticity canary / half-batch trunk cosine / feature
    # velocity). TRAINING-only and, uniquely in this file, it is not even that: it folds NO term
    # into `loss` and touches no `.grad`, so the policy's parameter updates are bit-identical
    # whether it is on or off. It buys scalars. OFF holds no state and pays one boolean per
    # minibatch. Detail: `agents/training/capacity_telemetry.py`.
    capacity_telemetry: bool = False
    # ENV steps between canary resets. Each reset re-seeds ONE of the K=4 synthetic targets,
    # round-robin, and the RE-FIT that follows is the supply-side measurement. Too small and the
    # head never converges between resets (recovery is noise); too large and a run yields two
    # points. 1M is ~16 resets per 3-hour launcher window at production throughput.
    canary_reset_steps: int = 1_000_000
    # Minibatches between half-batch cosine measurements. The probe costs two extra half-batch
    # forward+backwards ≈ one extra full one, so 50 amortizes it to ~2% of the train step.
    capacity_cosine_every: int = 50
    # `train()` calls between feature-velocity measurements. One 256-row no_grad forward.
    capacity_velocity_every: int = 50

    def _excluded_save_params(self):
        # `_cf_buffer` is the TRANSIENT scaffolding genre (like SB3's `rollout_buffer`): refilled from disk by
        # the producer, holding a threading.Lock cloudpickle can't serialize and hundreds of MB of obs if
        # pickled. Excluded for both reasons; re-created on resume, empty.
        # `_capacity_state` (gen3_capacity_telemetry_v1) is excluded DELIBERATELY and the
        # consequence is documented rather than hidden: it holds the canary's head, its Adam state,
        # the projection matrix and the frozen probe batch, so a resume re-inits the canary and its
        # loss/recovery curves restart. Persisting a diagnostic's optimizer into every checkpoint
        # is a worse trade than reading recoveries WITHIN a restart window.
        # `_vf_scale_announced` (gen3_winprob_critic_mode_v1) is the once-per-PROCESS latch on the
        # `[CRITIC] winprob` scale readout. Excluded so a launcher restart re-prints it: the line
        # belongs beside the startup banner, which every restart also re-prints, and the run's
        # `--vf-coef` may have been changed between them — a latch that rode the checkpoint would
        # silence the reading for the rest of the run's life after its first three hours.
        # `_diagnostics_ran_in_process` (gen3_diagnostics_cadence_v1) is a once-per-PROCESS latch like
        # `_vf_scale_announced`: the first update of every process runs every optional probe, so
        # the compile lock (taken after that update) has seen their signatures. Pickled, a
        # restarted process would skip them on its first update and reach them after the lock.
        # `collect_rollouts` / `train` / `learn` / `_compile_control` (gen3_compile_sentinel_v1) are the compile
        # sentinel's INSTANCE wrappers (`CompileControl.attach`): closures over a process-local
        # CompileControl holding a logging handler and dynamo callbacks. Pickled, every save after
        # the first update would carry (or fail on) them, and a loaded model would re-install a
        # dead process's sentinel. Re-attached fresh by every process that compiles.
        # `_learner_freeze` (gen3_learner_freeze_v1, K6) is the declared-lifecycle FREEZE GUARD
        # (`learner_lifecycle.attach`): it holds identity snapshots, torch's global registration
        # hooks and an optimizer step hook — process-local, re-attached by every process; it also
        # owns the `collect_rollouts` / `train` / `learn` wrappers above when it is the outermost.
        # `_compiled_micro_step` (K8 region R1, `compile_regions.install`) is a process-local compiled
        # callable — re-installed by every process that compiles, never pickled.
        # `_rust_collector` (M5 Lane G, `--env-core rust`) is the live env core + inference service
        # + row arena (locks, a child process, GPU slot storage): process-local by construction and
        # rebuilt by every process's startup. `_rust_fill` / `_rust_row_versions` / `_rust_version` /
        # `_behaviour_probe_metrics` are the last update's staleness record — transient like the buffer.
        # `_env_core_stamp` is written to metadata.json by `_model_hparams` on every save instead.
        # `_ppo_loop_mode` (gen3_owned_ppo_loop_v1) is which loop THIS process's `learn()` ran — the
        # owned one or the `GEN3AI_PPO_LOOP=sb3_reference` test seam — resolved per call; a checkpoint
        # must not carry it, so the `.zip`'s `data` stays exactly what it was before the loop moved.
        # `_loop_hooks` (gen3_declared_loop_hooks_v1) is the loop's hook TABLE — closures over this
        # process's freeze guard and compile sentinel; built fresh by every `_setup_model`.
        # `_rust_row_provenance` is the last fill's per-row provenance (K9(b)'s dump) — transient too.
        return super()._excluded_save_params() + ["_rust_collector", "_rust_fill", "_rust_row_versions",
                                                  "_rust_row_provenance",
                                                  "_rust_version", "_behaviour_probe_metrics", "_env_core_stamp",
                                                  "_cf_buffer",
                                                  "_capacity_state",
                                                  "_vf_scale_announced", "_diagnostics_ran_in_process",
                                                  "collect_rollouts",
                                                  "train", "learn", "_compile_control",
                                                  "_learner_freeze", "_compiled_micro_step",
                                                  "_ppo_loop_mode", "_loop_hooks"]
