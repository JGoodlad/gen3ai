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

    # Set by train_rl_agent after construction. GRADIENT ACCUMULATION: do K
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
    # (`winprob_head_refit_2026-09-09`). Requires the win-prob critic (refused otherwise, never a
    # silent no-op). Training-only, resume-mutable; scales a loss, touches no forward pass.
    win_prob_strata_weight: float = 0.0

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
        # `_rust_collector` (M5 Lane G, the Rust env core) is the live env core + inference service
        # + row arena (locks, a child process, GPU slot storage): process-local by construction and
        # rebuilt by every process's startup. `_rust_fill` / `_rust_row_versions` / `_rust_version` /
        # `_behaviour_probe_metrics` are the last update's staleness record — transient like the buffer.
        # `_env_core_stamp` is written to metadata.json by `_model_hparams` on every save instead.
        # `rollout_buffer_class` (gen3_owned_rollout_buffer_v1) is FORCED to the owned buffer by every
        # `_setup_model` (`OwnedLoop`), so a `.zip` never needs to name one — and a reader that loads with
        # plain sb3 builds sb3's own buffer, exactly as before the buffer moved.
        # `_loop_hooks` (gen3_declared_loop_hooks_v1) is the loop's hook TABLE — closures over this
        # process's freeze guard and compile sentinel; built fresh by every `_setup_model`.
        # `_rust_row_provenance` is the last fill's per-row provenance (K9(b)'s dump) — transient too.
        return super()._excluded_save_params() + ["_rust_collector", "_rust_fill", "_rust_row_versions",
                                                  "_rust_row_provenance",
                                                  "_rust_version", "_behaviour_probe_metrics", "_env_core_stamp",
                                                  "_capacity_state",
                                                  "_vf_scale_announced", "_diagnostics_ran_in_process",
                                                  "collect_rollouts",
                                                  "train", "learn", "_compile_control",
                                                  "_learner_freeze", "_compiled_micro_step",
                                                  "rollout_buffer_class", "_loop_hooks"]
