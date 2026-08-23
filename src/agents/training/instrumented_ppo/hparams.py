"""`PpoHyperparameters` — every knob `train_rl_agent` sets on the model AFTER construction.

A mixin of class attributes, not a config object, because that is exactly what these are: SB3
constructs the algorithm and the entry point then assigns `model.<name> = ...`. Each carries the
comment that says what it costs, what OFF means, and whether it is version-locked — which is the
reason they are worth 300 lines and worth keeping in one place.

The class defaults are all no-ops, so a smoke, a unit test, or a frozen eval/pool/distill
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

    # Set by train_rl_agent (gen3_bait_entropy_v1). The SECOND state-conditioned entropy boost — same
    # mechanism, different flag: on decisions the env flags `bait_opportunity`=1 (the attack we would click
    # is zero-damage against a revealed opponent BENCH mon), the per-decision entropy bonus is multiplied by
    # `bait_entropy_boost`. This is the PROBE of the bait verdict's stated mechanism — "exploration
    # starvation at a saturated action" (the whiff sits at p≈0.97, so the alternatives at p≈0.01-0.03 are
    # never sampled and their advantage is never realized). 1.0 = OFF (byte-identical entropy term).
    # Annealed B→1 over `bait_entropy_anneal_frac` (0 = constant); the anneal is what makes the probe
    # two-sided — behaviour that survives the anneal means sampling was the block, behaviour that reverts
    # convicts CREDIT. COMPOSES with the defensive boost MULTIPLICATIVELY (each factor is 1 off its own
    # flag, so either boost alone is byte-identical to running it alone). Training-only: NOT version-locked,
    # settable on resume.
    bait_entropy_boost: float = 1.0
    bait_entropy_anneal_frac: float = 0.0

    # Set by train_rl_agent (like opp_belief_aux_coef). The WIN-PROBABILITY head's BCE loss weight:
    # win_prob_coef * BCE(win_logit, MC outcome) over the transitions whose episode finished in-buffer.
    # Training-only (scales the loss, never a forward pass) → NOT version-locked. The MODE (none /
    # read_only / shaping — which also controls whether the grad reaches the trunk) lives on the
    # extractor (win_prob_mode); the loss is added whenever the mode is on AND this coef != 0.
    win_prob_coef: float = 1.0


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

    # COUNTERFACTUAL WIN-PROB GROUNDING (gen3_cf_label_plumbing_v1; G3 of
    # designs/ai_v10/design_counterfactual_value_grounding.md, rung R1). A background producer
    # re-rolls recorded training decisions to termination and drops tight Monte-Carlo P(win) labels
    # as JSONL; `_cf_buffer` (an `agents.training.cf_label_buffer.CfLabelBuffer`, attached
    # externally like `_correction_buffer`) ingests them, and this coefficient folds
    #     cf_winprob_coef * BCE( win_head(value_pooled(s)), MC_label(s) )
    # over its OWN sample and its OWN extractor forward — the labelled states are OFF-DISTRIBUTION
    # w.r.t. this rollout, so they cannot ride the minibatch.
    #
    # 0.0 = OFF and the whole block is skipped: no poll, no sample, no forward, loss byte-identical.
    # TRAINING-only (a loss weight; no forward/weight-shape change) → NOT version-locked, NOT in
    # check_compatible, resume-mutable — the `opd_coef` class.
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
    # check_compatible, NOT read back on a flagless resume: the `--opd-coef` class.
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
    # folded at `win_prob_coef`, not at this coefficient, so all three heads carry a bit-identical
    # A-term. 0.0 = OFF and the WHOLE twin block is skipped (including the on-policy mirror), so a
    # built-but-unused pair of heads leaves every parameter update byte-identical.
    # TRAINING-only (a loss weight) → the `--opd-coef` class; the STRUCTURAL half is the extractor's
    # `cf_twin_heads` kwarg (v99), which decides whether the heads' params exist at all.
    cf_twin_coef: float = 0.0
    # gen3_cf_twin_heads_v1: the SHADOW CRITIC's weight. Folds
    #     cf_shadow_coef * masked-MSE( shadow(value_pooled.detach()), normalize(mc_return) )
    # over the same sampled rows and the same extractor forward. The head never computes an
    # advantage and never enters GAE — it is the staged PROMOTION PATH for critic surgery (a critic
    # ROUTE change owes C4), so what it produces is evidence, not a training change to the critic.
    # 0.0 = OFF, whole block skipped. TRAINING-only; the STRUCTURAL half is `cf_shadow_critic` (v99).
    cf_shadow_coef: float = 0.0

    def _excluded_save_params(self):
        # The search-teacher's `_correction_buffer` lives on the model (the callback↔train() hand-off),
        # but it is TRANSIENT scaffolding like the rollout buffer — and it holds a threading.Lock that
        # cloudpickle can't serialize (model.save would crash). Exclude it from the checkpoint (also keeps
        # checkpoints small — a full buffer is hundreds of MB of obs); it's re-created on resume, empty,
        # and the workers/cycle refill it. Mirrors SB3 excluding `rollout_buffer`.
        # `_distill_teacher` is a full frozen model (a foreign exploiter) attached at setup — never pickle
        # it into our checkpoint; it is re-loaded from its own path on resume (like a stable opponent).
        # `_cf_buffer` is the same genre: transient scaffolding refilled from disk by the producer,
        # and hundreds of MB of obs if pickled. Excluded for the same two reasons.
        return super()._excluded_save_params() + ["_correction_buffer", "_distill_teacher",
                                                  "_distill_teachers", "_cf_buffer"]
