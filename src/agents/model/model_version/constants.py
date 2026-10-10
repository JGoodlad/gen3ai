"""The live identity constants, the error type, and the reward-immutable field tables.

`MODEL_CONFIG_VERSION` and `ARCH_SIGNATURE` are the two numbers every other module in this
package compares against, so they live alone in the leaf of the import graph: nothing here
imports a sibling, which is what lets `migrations`, `fields`, `compat` and `resume_checks` all
read them without a cycle.

The per-version narratives above each constant are HISTORY and are preserved verbatim.
"""
from typing import Any, Dict


# Bump this whenever the ModelVersion schema changes (fields added/renamed/removed).
# Also add a migration case in _migrate_config().
#
# The per-version narrative (v3 -> v88: what each field means, which gate enforces it, and
# why) lives in designs/CHANGELOG.md under 'The MODEL_CONFIG_VERSION narrative' — moved
# there 2026-08-16; it is history, and this file keeps only the live machinery.
# v89 (gen3_value_pooled_routes_v1): the five value routes (intent_value_reduce v74,
#   value_entity_pool v80/82, intent_threshold's vf half v84, value_clock/value_intent v87 —
#   four of the five are DELETED at v96; `value_entity_pool` is the one that carried)
#   INJECT into `value_pooled` instead of the post-assembler vf concat, which
#   `--value-from-dist` structurally bypassed (gen-12 proof: their zero-init projections
#   bit-exact ZERO after 25M steps). Route out-widths become D_MODEL and the vf concat
#   narrows for flag-ON configs, so a <v89 checkpoint recording ANY of them ON carries
#   shapes the surviving code cannot load — REFUSED (the v75 rule); OFF stamps forward.
# v92 (gen3_td_consistency_aux_v1): `td_aux_coef` — the TD-consistency auxiliary's weight. A
#   TRAINING-only loss coefficient (the opp_belief_aux_coef class): recorded for provenance and
#   for flagless-resume read-back, never gated. A pre-v92 config defaults it to 0.0 = OFF.
#   ⚠️ Built as v90 and RENUMBERED: v90 (gen3_frame_deletion_v1) and v91
#   (gen3_event_semantics_v1) landed while this sat on a branch. No ARCH_SIGNATURE bump —
#   the term is computed in the PPO step, never in the extractor forward, so a coef-0 build
#   is byte-identical and there is nothing for `check_compatible` to gate.
# v95 (substrate Phase C): `conditional_threat_cell` (OA1 — the defensive-pivot coordinates on the
#   pointer SWITCH cell) and `pair_value_route` (PV — the α-reduced outcome row as TOKEN CONTENT on
#   the critic's copy of our team tokens). Both opt-in, zero-init, OFF byte-identical. The same bump
#   carries `gen3_status_economy_v1`, which AMENDS `tempo_cost`'s coordinate semantics under the
#   existing `pair_outcome_*` flags (the Natural Cure ability + the bench-cleric path become undo
#   paths; the reduction becomes a MIN over available paths). No ARCH_SIGNATURE bump — with the
#   flags OFF the forward is byte-identical — but a <v95 config recording either pair_outcome flag
#   ON is REFUSED rather than migrated, since it trained against different numbers.
# v96 (gen3_critic_route_wave_v1) — THE CRITIC-ROUTE DELETION WAVE. Seven audited-dead critic
#   routes are deleted in one pass, and with them the whole post-assembler vf tail:
#     * the v61 MultiSeedValueReadout + seed_diagnostics + the `value_seeds/*` TB contract
#       (dV 0.0000 bit-exact, gen-13 AND gen-14)
#     * the hidden-opp belief's VF half ONLY — its PI half flips 39.6% of argmaxes and STAYS
#     * the `non_matchup_rest` VF concat (0.0000; C1 measured the content substituting
#       through the global token) — its PI concat STAYS
#     * `value_intent` (0.156) · `intent_threshold`'s vf half (0.155/0.136) ·
#       `intent_value_reduce` (0.3176 at 2x) · `value_clock` (0.2169 at 2x), all vs a 0.39 bar
#   Three FIELDS go with them (`intent_value_reduce`, `value_clock`, `value_intent`); the other
#   four were unconditional or ride a surviving flag. `vf_combined` is now `value_pooled` alone,
#   which is what bumps ARCH_SIGNATURE: `value_projection` narrows and `assembler.seed_readout.*`
#   leaves the state_dict, and NOTHING in the config records either, so the signature is the only
#   gate that can reject a pre-v96 checkpoint with a diagnosis instead of an opaque torch error.
# v97 (gen3_intent_label_bot_weight_v1): `intent_label_bot_weight` — the per-sample weight on the
#   opponent-intent (alpha/beta) LABELS produced against a heuristic BOT. A TRAINING-only loss
#   weight (the td_aux_coef class): recorded for provenance and for flagless-resume read-back,
#   never gated. It scales a loss term computed in the PPO step and touches no forward pass, so a
#   default (1.0) build is bit-identical and there is nothing for `check_compatible` to compare.
#   A pre-v97 config defaults it to 1.0 = OFF. No ARCH_SIGNATURE bump.
# v98 (gen3_cf_evidential_head_v1): `cf_evidential` — the EVIDENTIAL Beta readout over P(win|state)
#   off `value_pooled` (the counterfactual label factory's rung R1). STRUCTURAL, exactly the
#   win_prob_mode / value_dist_mode precedent: building it adds the head's params to the state_dict,
#   so a resume that flips it has either no weights for the head or orphan weights, and a bool
#   compare in check_compatible is the gate. It is never called from the extractor forward (the
#   training-side loss applies it to the stashed `value_pooled`, always detached), so OFF is
#   byte-for-byte the baseline and ON-at-coefficient-0 is bit-identical in pi/vf too — hence NO
#   ARCH_SIGNATURE bump, the optional-side-head rule. A pre-v98 config defaults it to False = OFF.
#   Its two coefficients (`cf_evidential_coef`, `cf_evidential_reg`) are TRAINING-only argparse in
#   the `--td-aux-coef` / `--cf-winprob-coef` class and appear nowhere here.
# v99 (gen3_cf_twin_heads_v1): `cf_twin_heads` + `cf_shadow_critic` — the TWIN WIN-PROB HEADS and
#   the passive SHADOW CRITIC (the owner-authorized amendment to the signed R1 pre-registration;
#   ledger 2026-08-22 evening, "Three owner sign-offs" item 3). Two STRUCTURAL bools in exactly the
#   v98 mould: each builds modules whose params ARE the state_dict delta, neither is ever called
#   from the extractor forward (the training-side terms apply them to the stashed `value_pooled`,
#   always detached), so OFF is byte-for-byte the baseline and ON-at-coefficient-0 is bit-identical
#   in pi/vf. NO ARCH_SIGNATURE bump — optional side heads, obs family unchanged. Both are gated by
#   a bool compare in check_compatible, and the gate is the ONLY thing that can catch a flipped
#   flag, because a head the forward never calls produces no shape error anywhere. A pre-v99 config
#   defaults BOTH to False = OFF (not a guess: the modules did not exist). Their coefficients
#   (`cf_twin_coef`, `cf_shadow_coef`) are TRAINING-only argparse in the `--td-aux-coef` class and
#   appear nowhere here. TWO fields in ONE bump because they ship as one amendment.
# v100 (gen3_cf_coef_provenance_v1): the TEN counterfactual COEFFICIENTS (cf_records,
#   cf_records_keep, cf_winprob_coef, cf_head_only, cf_label_lag_steps, cf_label_likelihood,
#   cf_evidential_coef, cf_evidential_reg, cf_twin_coef, cf_shadow_coef) leave the `--td-aux-coef`
#   genre for the td_aux_coef one: all TRAINING-only (a loss in the PPO step; no forward read, no
#   weight shape) ⇒ RECORDED for provenance + flagless-resume read-back, NEVER gated.
#   ⚠️ THE DEFECT IS SILENT: an R1 arm resumed without re-typing `--cf-winprob-coef 1.0` kept
#   training and simply stopped applying the term it was launched to measure. Their three
#   STRUCTURAL companions (cf_evidential v98, cf_twin_heads/cf_shadow_critic v99) were already
#   recorded and GATED, so a resume could keep the head and lose the coefficient driving it.
#   A pre-v100 config defaults each to its ARGPARSE default — not a guess: the fields did not
#   exist, so that is what every such run ran with. No ARCH_SIGNATURE bump, no floor change.
# v101 (gen3_capacity_telemetry_v1): the FOUR live-capacity-telemetry knobs (capacity_telemetry,
#   canary_reset_steps, capacity_cosine_every, capacity_velocity_every) — v100's shape exactly:
#   TRAINING-only, RECORDED for provenance + flagless-resume read-back, NEVER gated. They are
#   weaker than v100's even: a cf coefficient at least scales a loss, whereas these fold nothing
#   into `loss` and write no `.grad`, so a run's parameter updates are bit-identical on or off.
#   They are recorded anyway because a DIAGNOSTIC whose provenance is unrecoverable is a number
#   nobody can interpret six months later — `metadata.json`'s `cli_args` is overwritten by every
#   resuming process, so `model_config.json` is the only durable record of what a run measured.
#   A pre-v101 config defaults each to its ARGPARSE default (not a guess: the fields did not
#   exist). No ARCH_SIGNATURE bump — the canary head is owned by the PPO object, not the
#   extractor, so no state_dict key and no forward changes.
# v102 (gen3_policy_grad_coef_v1): `policy_grad_coef` — the weight on the PPO POLICY-GRADIENT term itself
#   (`policy_grad_coef * policy_loss`; scales ONLY the clipped surrogate — never entropy, never the value
#   term, never an aux). A TRAINING-only loss coefficient, the td_aux_coef class exactly: recorded
#   for provenance and for flagless-resume read-back, never gated. 1.0 = the upstream expression
#   (byte-identical — the unscaled tensor is used); 0.0 = arm F's pure-distill/aux phase, the
#   value it exists for (design_advantage_gated_distillation.md §5 needed a way to run PPO with
#   the policy-gradient term OFF, and no flag could zero it). A pre-v102 config defaults it to
#   1.0 = upstream — not a guess: the term entered at an implicit 1.0 in every run ever made.
#   No ARCH_SIGNATURE bump — computed in the PPO step, never in the extractor forward.
# v103 (gen3_distill_target_gate_v1): the ADVANTAGE-GATED / ACTION-FORM DISTILLATION family + the
#   RANK TRIPWIRE (design_advantage_gated_distillation.md §3.1/§3.3/§4.1/§7.1). Seven TRAINING-only
#   knobs, the td_aux_coef class exactly — recorded for provenance + flagless-resume read-back,
#   never gated. `distill_target` ("kl" = the untouched full-distribution KL, the byte-identical
#   default; "action" = the teacher's top-K probabilities renormalized over the legal set —
#   `distill_topk`=1 ⇒ pure argmax CE, the §3.3 axis no arm had ever manipulated; K >= n_actions
#   recovers the KL), `distill_gate` "advantage" + `distill_gate_tau` (rung (a): a row fires only
#   where the teacher's argmax disagrees with the SAMPLED action AND the student's own NORMALIZED
#   advantage reads it as a mistake, Â < -τ — so the distill gradient pushes a logit PPO is
#   already pushing down, by construction), `distill_beta` (the AWR |Â| temperature, mirroring
#   search_teacher_beta), and `rank_tripwire`/`rank_tripwire_drop` (§4.1: the rank/policy_pr
#   EMA-vs-own-baseline watchdog, default "warn"; "abort" may stop learn() cleanly — it changes
#   WHEN training ends, never what a step computes). A pre-v103 config defaults each to its
#   argparse default — not a guess: "kl" is the one loss every such run trained with, and the
#   tripwire did not exist. No ARCH_SIGNATURE bump — nothing here touches a forward pass or a
#   weight shape.
# v104 (gen3_winprob_pbrs_v1; ai_v12 route 1 — designs/ai_v12/design_winprob_behavior_coupling.md):
#   `win_prob_pbrs_coef` — POTENTIAL-BASED REWARD SHAPING from the win-prob head. Every
#   transition's reward gains `coef * (gamma*phi(s') - phi(s))`, phi = the DETACHED sigmoid of the
#   win-prob logit, applied trainer-side to the rollout buffer before GAE. It is the FIRST knob in
#   this family that edits the REWARD STREAM rather than a loss term — worth saying, because the
#   provenance class is nevertheless td_aux_coef's exactly: no forward pass reads it, no weight
#   shape depends on it, so it is recorded for provenance + flagless-resume read-back and never
#   gated. 0.0 = OFF and the shaping module is not even imported (byte-identical). A pre-v104
#   config defaults it to 0.0 — not a guess: the flag did not exist, so no run could have used it.
#   No ARCH_SIGNATURE bump — the reward stream is not the network.
# v105 (gen3_clean_world_config_v1 + gen3_winprob_pbrs_source_v1): FIVE keys for the CLEAN-WORLD
#   arm. Four are resume-immutable VALUE-meaning reward fields — `hand_shaping` (the master
#   off-switch for all eight hand PBRS potentials AND the whole BIAS class, the composition
#   `--no-all-shaping-pbrs` could not reach because that flag is ALSO `_bias_term_active`'s master
#   gate), `pbrs_material` / `pbrs_belief` (the two potentials that had no flag at all) and
#   `victory_value` (the ±30 terminal, promoted off a module constant so ±1 is reachable by flag).
#   The fifth, `win_prob_pbrs_source`, is TRAINING-only provenance: the frozen checkpoint whose
#   win-prob head supplies φ. Every default IS the pre-v105 behaviour, so the migration is a plain
#   setdefault and a flagless run is byte-identical. No ARCH_SIGNATURE bump — no weight shape moves.
# v106 (gen3_progress_clock_intent_v1): `progress_decision_tense` + `progress_switch_freeze` — the
#   two OPT-IN intent-restoring fixes to the no-progress clock (probes M/N, 2026-08-29). Same
#   provenance class as v14/v15's reward switches: resume-immutable VALUE-meaning, checked by
#   `check_reward_config`, excluded from the weight-shape check, no ARCH_SIGNATURE bump. A pre-v106
#   config defaults BOTH to False — not a guess: the flags did not exist, so no run can have used
#   them, and False reproduces the behaviour every generation through gen-15 trained under.
# v107 (gen3_q_winprob_head_v1; ai_v12 E5 — ledger 229e9f1 / 5edbd05): `q_winprob_mode` plus its
#   two coefficients `q_winprob_coef` / `q_winprob_onpolicy_coef`. The MODE is STRUCTURAL in the
#   `win_prob_mode` mould: 'none' builds nothing (byte-for-byte the baseline), 'read_only' builds
#   a `QWinProbHead` whose params ARE the state_dict delta, and a string compare in
#   check_compatible is the gate. It differs from `win_prob_mode` in exactly two ways, both
#   deliberate. (1) There is NO 'shaping' value — every input is detached unconditionally, so no
#   coefficient can make a per-action readout carrying a COUNTERFACTUAL label reshape the trunk;
#   trunk exposure is a later decision that owes its own gate. (2) It reads the POINTER stash as
#   well as `value_pooled`, which is why the module is built LAST in `__init__` (after every
#   module that widens a pointer cell) and called at the END of the forward.
#   NO ARCH_SIGNATURE bump: with the mode 'none' the forward is byte-identical, and with it on the
#   only output is a stash — pi/vf are bit-identical either way. The two coefficients are
#   TRAINING-only (the `cf_winprob_coef` class): recorded for provenance + flagless-resume
#   read-back, never gated. A pre-v107 config defaults the mode to 'none' and both coefficients to
#   0.0 — not a guess: none of the three existed, so that is what every such run ran with.
# v108: gen3_dead_flag_purge_v2 — the STAMP only, for `threat_prob_outspeed`. Its POP/REFUSE half is
#   version-INDEPENDENT (a stale key TypeErrors in `cls(**data)` whatever vintage wrote it), so it
#   lives with the other sanitizers in `migrations.py` rather than in a `version <` branch.
#   NO ARCH_SIGNATURE bump, and that is the safety rule rather than a convenience: the flag built no
#   parameters, so every state_dict key is exactly where it was and no weight becomes unplaceable.
#   The same fact is why True is REFUSED instead of popped — see the migration.
# v109: gen3_winprob_critic_mode_v1 (designs/ai_v12/design_winprob_only_critic.md) — `critic`,
#   a STRUCTURAL string in the `win_prob_mode` mould, plus the two resume-immutable reward fields
#   the 'winprob' value it introduces implies (`terminal_indicator`, `no_progress_tax_armed`).
#   'shaped' is the DEFAULT and is today's critic exactly, so a flagless run is byte-identical and
#   there is NO ARCH_SIGNATURE bump: no module is added or removed, no state_dict key moves, and
#   the forward is unchanged. The signature bump belongs to the DEFAULT FLIP, which is a separate
#   commit — there it is forced, because a critic trained to predict a shaped return cannot be
#   warm-started into predicting a probability. A pre-v109 config defaults all three to today's
#   values; not a guess, since none of the three existed.
# v110: gen3_frozen_phi_actor_only_v1 (designs/ai_v12/design_winprob_only_critic.md §3.7) — ONE
#   training-only path, `win_prob_pbrs_frozen`, in the v105 `win_prob_pbrs_source` mould. It names
#   the FROZEN checkpoint whose win-prob head supplies the ACTOR-ONLY potential: the shaping is
#   applied to `rollout_buffer.advantages` alone, so the critic keeps regressing the unshaped
#   terminal indicator and V(s) = P(win|s) is preserved bit-for-bit. NO ARCH_SIGNATURE bump — no
#   module is added or removed, no state_dict key moves, and no forward pass changes; the flag
#   edits a numpy array between collection and train(). None is the default and is every pre-v110
#   run's only possible past.
# v111: gen3_arch_surface_guard_v1 — ONE provenance-only string, `arch_source`, in the v105
#   `win_prob_pbrs_source` mould. It records WHERE this run's architecture surface came from:
#   `--arch production` stamps `production_config@<12 hex of the mirror's git blob hash>`,
#   `--allow-nonproduction-arch` stamps the deliberate-drift form, and a run that used neither
#   records None. It gates NOTHING — absent from `_WEIGHT_FIELDS` and from every `check_*` — and
#   carries NO ARCH_SIGNATURE bump: no module, no state_dict key, no forward. It exists because
#   the 2026-09-06 incident's run recorded a near-bare architecture with nothing on disk saying
#   whether that was a decision or an accident.
# v112: gen3_eval_sentinel_greedy_default_v1 — the EVAL OPPONENT REGIME (`eval_sentinel_greedy`)
#   and the promotion gate it derives (`promote_threshold`), in the v101 capacity-telemetry mould:
#   TRAINING/EVAL-only, RECORDED for provenance + flagless-resume read-back, NEVER gated. Neither is
#   read by any forward and neither changes a weight shape, so there is NO ARCH_SIGNATURE bump and
#   a frozen eval/pool opponent (which runs no eval cycle) passes trivially.
#   The bump exists because the UNRECORDED version of this flag already cost a year of
#   comparability: `--eval-sentinel-greedy` was ON for 49 runs (v5.5–v8) and dropped without a note
#   at the v9 launch, and the asymmetric regime it left behind reads +8.9 pp [+7.0, +10.7] in the
#   trainee's favour on the same frozen pair the dense ladder plays symmetrically. THE DEFAULT IS
#   FLIPPED in the same commit (greedy + symmetric teams), which is exactly why the field must be
#   recorded and inherited: without it, every resume of a v9-era run on this code would silently
#   cross an opponent-regime boundary mid-run (rule of evidence 15). A pre-v112 config defaults to
#   False and derives its gate from that — see the migration for why that is a record, not a guess.
# v113 (gen3_teacher_scan_limit_flag_v1): `teacher_scan_limit` — how many loss traces of the newest
#   eval cycle the search-teacher's SELECTION half falsify-gates per cycle. The v101/v112 mould:
#   TRAINING-only, RECORDED for provenance + flagless-resume read-back, NEVER gated. No forward
#   reads it, no state_dict key depends on it, so there is NO ARCH_SIGNATURE bump and a frozen
#   eval/pool opponent (which runs no teacher cycle) passes trivially.
#   The bump exists because the value was a HARD-CODED 60 inside `SearchTeacherCallback` with no
#   flag at all, and it is simultaneously the selection half's COST (~3 s of re-rolls per trace,
#   ~100 s per cycle at 60 — which is what the same commit moves off the training step) and its
#   SUPPLY (the crater pool a cycle's candidates are drawn from). A pre-v113 config defaults to 60,
#   which is not a guess: no run could set anything else, because nothing could set it.
# v114 (gen3_value_true_team_v1): `value_true_team` — the PRIVILEGED (true-opponent-team) VALUE
#   route, arm 5 of the critic ladder (designs/research_state/winprob_critic_ladder_2026-09-08.md).
#   The value_entity_pool mould: STRUCTURAL, recorded, and gated by a bool compare in
#   check_compatible because the injection is ADDITIVE into `value_pooled` and therefore changes no
#   width that a shape error could catch. NO ARCH_SIGNATURE bump, and that is a claim worth making
#   precisely: the 2501-dim observation VECTOR is unchanged (the privileged block rides a SEPARATE
#   Dict key, `opp_true_team`, the win_target/belief_species precedent), no existing module moves,
#   and the readout is built LAST — so an OFF run on this code is bit-identical to the same run on
#   v113 and every existing checkpoint still resumes. A pre-v114 config defaults to False, which is
#   not a guess: no run could set anything else, because nothing could set it.
# v115 (gen3_winprob_strata_weight_v1): `win_prob_strata_weight` — OPPONENT-STRATIFIED weighting
#   of the win-prob BCE, arm 7 of the critic ladder
#   (designs/research_state/measurements/winprob_head_refit_2026-09-09/ §11). ONE TRAINING-only
#   loss weight, the td_aux_coef / v97 shape exactly: it multiplies each state's BCE term by its
#   opponent CLASS's inverse-frequency weight, so it reweights a loss computed in the PPO step and
#   touches no forward pass and no weight shape — a default (0.0) build is bit-identical and there
#   is nothing for `check_compatible` to compare. RECORDED anyway, for v100's reason: a resume that
#   dropped it would keep training and silently stop applying the only thing the arm exists to
#   measure. A pre-v115 config defaults to 0.0 = OFF, which is not a guess but the only possible
#   past — the field did not exist. No ARCH_SIGNATURE bump, no MIGRATION_FLOOR change.
# v116 (gen3_winprob_lambda_v1): `win_prob_lambda` + `win_prob_lambda_truncated` — λ-RETURN
#   targets for the win-prob BCE, arm 8 of the critic ladder
#   (designs/research_state/measurements/winprob_head_refit_2026-09-09/ §11). TRAINING-only, the
#   td_aux_coef / v115 shape exactly: the pair changes what the BCE regresses TOWARD, computed in a
#   post-collection callback over the rollout buffer, and touches no forward pass and no weight
#   shape — a default (1.0) build is bit-identical and there is nothing for `check_compatible` to
#   compare. RECORDED anyway, for v100's reason: a resume that dropped them would keep training and
#   silently return the arm to the terminal-bit target it exists to contest. A pre-v116 config
#   defaults to 1.0 / "bootstrap", which is not a guess but the only possible past — λ = 1.0 IS the
#   terminal-bit target every prior run used, and the truncation mode is inert at that λ. No
#   ARCH_SIGNATURE bump, no MIGRATION_FLOOR change.
# v117 (gen3_dense_aux_v1): `dense_aux` (STRUCTURAL bool) + `win_prob_dense_aux` (its training
#   COEFFICIENT) — the DENSE AUXILIARY head, arm 9 of the critic ladder. The `opp_belief_slots` /
#   `opp_intent` SPLIT: one CLI flag, `--win-prob-dense-aux <coef>`, whose positivity BUILDS the
#   head and whose magnitude doses its loss. The BOOL is the value_true_team mould exactly —
#   structural, recorded, gated by a bool compare in check_compatible because the head's params are
#   the whole state_dict delta and its only output is a training-side loss, so no shape error
#   anywhere would catch a flipped flag. The COEFFICIENT is the td_aux_coef class — recorded for
#   provenance and flagless-resume read-back, never compared — which is what makes a resume free to
#   RE-DOSE the arm but not to add or drop its parameters. NO ARCH_SIGNATURE bump, for
#   `value_true_team`'s reasons: the 2501-dim observation VECTOR is unchanged (the labels ride
#   SEPARATE Dict keys, the win_target precedent), no existing module moves, the head is built LAST
#   and is not called by the forward at all — so an OFF run on this code is bit-identical to the
#   same run on v116 and every existing checkpoint still resumes. A pre-v117 config defaults to
#   False / 0.0, which is not a guess: no run could set anything else, because nothing could.
# v118 (gen3_winprob_rollout_target_v1): `win_prob_rollout_target` + `win_prob_rollout_r` +
#   `win_prob_rollout_mode` — R-ROLLOUT MONTE-CARLO targets for the win-prob BCE, arm 10 of the
#   critic ladder. TRAINING-only, the v116 shape exactly: the three change what a SUBSAMPLE of the
#   buffer's states regresses toward, computed in a post-collection callback, and touch no forward
#   pass and no weight shape — a default (0.0) build is bit-identical and there is nothing for
#   `check_compatible` to compare. RECORDED anyway, for v100's reason, and with an extra edge here:
#   the treatment also carries a large WALL-CLOCK cost, so a resume that dropped it would look like
#   a speed-up rather than like a lost arm. A pre-v118 config defaults to 0.0 / 8 / "replace",
#   which is not a guess but the only possible past — the terminal bit IS what every prior run
#   trained against, and the other two are inert at that fraction. No ARCH_SIGNATURE bump, no
#   MIGRATION_FLOOR change.
# v119 (gen3_winprob_rollout_weight_v1): `win_prob_rollout_weight` — the per-row LOSS WEIGHT on
#   the rows v118's fraction ANCHORED, and the arithmetic half of arm 10. v118's shape exactly: it
#   re-prices rows of a loss computed in a post-collection callback, touches no forward pass and no
#   weight shape, and a default (1.0) build is bit-identical — the obs key it rides is not even
#   declared. RECORDED for v100's reason and for v118's: a resume that dropped it would keep paying
#   for every continuation while delivering ~1/50th of the dose the arm was registered at, which is
#   the most expensive way to read a null. A pre-v119 config defaults to 1.0, which is not a guess
#   but the only possible past — no run could weigh a row it had no flag to weigh with. No
#   ARCH_SIGNATURE bump, no MIGRATION_FLOOR change.
# v120 (gen3_fork_v1): `fork_fraction` + `fork_branches` + `fork_contested_gap` +
#   `fork_contested_absv` + `fork_max_per_battle` + `fork_crn` — the FORK ARM, registered by
#   `designs/research_state/measurements/paired_refit_discrimination_2026-09-14/`. TRAINING-only,
#   the v118 shape exactly: they steer a post-collection callback that plays contested-state
#   branches and injects their transitions into the rollout buffer, and touch no forward pass and
#   no weight shape — a default (0.0) build is bit-identical (no module imported, no obs key
#   declared, no callback attached, no row injected), and there is nothing for `check_compatible`
#   to compare. RECORDED for v100's reason and for v118's — the arm carries a ~1.5-2x WALL-CLOCK
#   cost, so a resume that dropped it would look like a speed-up rather than like a lost arm — and
#   for one more that is specific to this arm: its registered endpoint is a comparison against an
#   OFFLINE baseline measured at one particular selector quantile and one particular CRN regime, so
#   all six are recorded and a run that cannot say which it used cannot be read against that
#   baseline. A pre-v120 config defaults to 0.0 / 3 / 0.40 / 0.0 / 1 / "dice_and_draws", which is
#   not a guess but the only possible past — no run could fork a decision it had no flag to fork
#   with, and the other five are inert at fraction 0. No ARCH_SIGNATURE bump, no MIGRATION_FLOOR
#   change.
# v121 (gen3_event_record_v2) — THE OBSERVATION-ARCHITECTURE BATCH, one deliberate retrain boundary
#   (owner, 2026-09-25): the E12 event-row reshape (22 → 30 columns, a DENIED row type), the E4
#   refused-switch target, the Mud Sport / Water Sport volatile slots (and the damage op honouring
#   them), and the E10 parameter-free Smogon mixture for the hidden-slot move prior. The obs grows
#   2501 → 2761, `EventSeats`' projection and the `r` edge cell (2 → 3) change shape, and the hidden
#   slots' move posterior changes meaning — so ARCH_SIGNATURE bumps and MIGRATION_FLOOR rises to
#   121. No field is added.
# v122 (gen3_shaped_reward_deletion_v1) — THE SHAPED REWARD PATH IS DELETED (program_rust_core §4 M3
#   row, owner-approved 2026-09-26). FOURTEEN resume-immutable reward fields LEAVE the config:
#   bias_additivity, mat_alive_weight, bias_redesign, switch_bias_weight, self_ko_hp_penalty,
#   drop_redundant_bias, drop_switch_bias, all_shaping_pbrs, stall_pbrs, no_progress_penalty,
#   hand_shaping, pbrs_material, pbrs_belief, no_progress_tax_armed. `_migrate_config` POPs them
#   (version-independent, so a frozen load of any vintage works), and a RESUME or FORK of a config
#   that recorded a shaped reward is REFUSED by `model_version.shaped_reward` (never switched
#   silently to the terminal alone). No weight shape moves: no ARCH_SIGNATURE bump, no
#   MIGRATION_FLOOR change — a v121 production checkpoint loads and resumes unchanged.
# v123 (gen3_policy_gae_lambda_v1): `policy_gae_lambda` — the PPO POLICY's GAE λ, until now
#   HARDCODED to 0.80 at both model_build sites and so unrecorded. TRAINING-only, the td_aux_coef
#   class exactly: it shapes the rollout buffer's advantages / returns, touches no forward pass and
#   no weight shape, so there is nothing for `check_compatible` to compare. RECORDED for v100's
#   reason — a resume that dropped it would silently return the policy to 0.80 — and read back by
#   `_resolve` on a flagless resume. A pre-v123 config defaults to 0.80, which is not a guess but
#   the only possible past. No ARCH_SIGNATURE bump, no MIGRATION_FLOOR change.
# v124 (gen3_diagnostics_cadence_v1): `diagnostics_every` — `--diagnostics-every N`, the cadence of
#   the OPTIONAL learner telemetry (per-term noise scale, grad balance, rank, edge/cell liveness;
#   M5 Lane K2). Changes no training math (bit-identity pinned by `diagnostics_cadence_test.py`),
#   so there is nothing for `check_compatible` to compare. RECORDED as the REGIME of those TB
#   series and read back by `_resolve` on a flagless resume. A pre-v124 config defaults to 1 —
#   every run before the flag ran them every update. A FRESH run resolves to 10. No ARCH_SIGNATURE
#   bump, no MIGRATION_FLOOR change.
# v125 (gen3_opp_intent_coef_recorded_v1): `opp_intent_coef` — the dose that ENABLES the derived
#   `opp_intent` toggle (`--opp-intent-coef > 0` builds the alpha/beta heads). TRAINING-only, the
#   td_aux_coef class: recorded for flagless-resume read-back, never gated. Before v125
#   `model_config.json` recorded only the bool, so a launcher RESTART of a fresh `--arch production`
#   run (the restart strips the FRESH-only `--arch`) resolved the coefficient to 0.0 and FATALed at
#   check_compatible on `opp_intent`. A pre-v125 config migrates to 0.0 when `opp_intent` is OFF (the
#   only possible past) and stays UNRECORDED (None) when ON — a resume then takes the run's
#   `metadata.json:cli_args` dose or is REFUSED. No ARCH_SIGNATURE bump, no MIGRATION_FLOOR change.
# v126 (gen3_ridealong_heads_v1): `ridealong_ensemble` / `ridealong_rnd` / `ridealong_adv` /
#   `ridealong_opp` — the DETACHED RIDE-ALONG heads (V ensemble, RND novelty, A, B; owner
#   2026-09-30, EXPERIMENT_BACKLOG X25 / X4a). STRUCTURAL (their params are the state_dict delta,
#   gated in check_compatible) but INERT to training: every input is detached, the heads have their
#   own optimizer and a private init RNG, so a run with them learns exactly what the same run
#   without them learns (`ridealong_heads_test`). Built on the POLICY after SB3's `_build`, never
#   called by the forward. A pre-v126 config defaults every toggle OFF (the only possible past).
#   No ARCH_SIGNATURE bump (the observation vector and every existing module are unchanged), no
#   MIGRATION_FLOOR change.
# v127 (gen3_ridealong_rnd_variants_v1): `ridealong_rnd_variants` — the RND VARIANT ENSEMBLE beside
#   the base `ridealong_rnd` head (fast / decay / small / feat; owner 2026-09-30, X26). A canonical
#   comma-list STRING, STRUCTURAL (the variants' predictors are the state_dict delta, gated in
#   check_compatible), INERT to training exactly like the four v126 toggles. A pre-v127 config
#   migrates to "off" (the only possible past). No ARCH_SIGNATURE bump, no MIGRATION_FLOOR change.
# v128 (gen3_mirrored_pairs_v1): `eval_mirrored_pairs` — `--eval-mirrored-pairs` (T17), the in-loop
#   eval's PAIRING REGIME: every team pairing of the bot and pool eval played from both sides on one
#   battle seed, the pair the statistical unit. EVAL-only (the v112 `eval_sentinel_greedy` class): never
#   compared by check_compatible, RECORDED as a regime boundary and read back by `_resolve` on a flagless
#   resume. A pre-v128 config migrates to False (the only possible past). No ARCH_SIGNATURE bump, no
#   MIGRATION_FLOOR change.
# v129 (gen3_sprt_promotion_v1): `promotion_sprt` — `--promotion-sprt` (T6), pool promotion by a
#   per-candidate SPRT on fresh mirrored pairs. The v128 class: never compared, RECORDED as a regime
#   boundary, read back by `_resolve`. A pre-v129 config migrates to False. No ARCH_SIGNATURE bump, no
#   MIGRATION_FLOOR change.
# v130 (gen3_bare_argv_winprob_v1, deletion pass D2, owner 2026-10-02): the BARE-ARGV DEFAULT FLIP — a
#   fresh argv without `--arch production` now defaults to the win-prob critic, the win-indicator terminal
#   (indicator ON, victory 1.0, draw 0.0) and the Rust core (all constants since P11b). A PROVENANCE boundary only: no
#   field is added, every one of those is recorded explicitly, and an ABSENT record still means the
#   historical value (`critic_mode.CRITIC_UNRECORDED`, `_REWARD_IMMUTABLE_FIELDS`), so no checkpoint
#   loads differently. No ARCH_SIGNATURE bump (designs/model/versioning.md says why), no
#   MIGRATION_FLOOR change.
# v131 (deletion pass L1, owner-approved 2026-10-02 — designs/ops/deletion_pass_manifest.md §2): the
#   levers that ran only on the Python core or only under the shaped critic LEAVE the config, with the
#   code behind them: use_popart, value_dist_mode (+ value_dist_bins / vmin / vmax / coef),
#   value_from_dist, value_tail_weight, win_prob_coef, win_prob_pbrs_coef, win_prob_pbrs_source,
#   win_prob_pbrs_frozen. `_migrate_config` POPs them version-independently; a recorded ON value that
#   named parameters or a critic route (PopArt, the distributional head, value_from_dist) is refused on
#   EVERY load, and a RESUME or FORK of a run that recorded any lever ON is refused by
#   `model_version.retired_levers` (never silently continued without it). Every v121+ run on record
#   recorded them all OFF, so no checkpoint loads differently: no ARCH_SIGNATURE bump, no
#   MIGRATION_FLOOR change. Later deletion units APPEND their levers to that module's table.
# v132 (deletion pass L2, owner-approved 2026-10-02): the next slice of Python-core-only levers leaves the
#   config: win_prob_lambda (+ win_prob_lambda_truncated), win_prob_rollout_target (+ _r / _mode /
#   _weight), win_prob_dense_aux and the STRUCTURAL dense_aux bool, and the STRUCTURAL value_true_team
#   bool. (The two entropy boosts were never recorded fields.) Same machinery as v131: `_migrate_config`
#   POPs them from any vintage; a recorded dense_aux / value_true_team ON is refused on EVERY load (a
#   `DenseAuxHead` / `TrueTeamValueReadout` in the state_dict has no home), and a RESUME or FORK of a
#   run that recorded any lever ON is refused by `model_version.retired_levers`. Every v121+ run on
#   record recorded them all OFF, so no checkpoint loads differently: no ARCH_SIGNATURE bump, no
#   MIGRATION_FLOOR change.
# v133 (deletion pass L3, owner-approved 2026-10-02): DISTILLATION and the SEARCH TEACHER leave the config:
#   the five distillation loss knobs (distill_target / _topk / _gate / _gate_tau / _beta, v103) and
#   teacher_scan_limit (v113). All TRAINING-only (no extractor forward reads them, no weight shape
#   depends on them), so no lever of this slice is STRUCTURAL: `_migrate_config` POPs them from any
#   vintage, and a RESUME or FORK of a run that recorded a non-default one (`distill_target != "kl"`,
#   `distill_gate != "none"`, `teacher_scan_limit != 60`; the top-K / tau / beta values are inert
#   without those) is refused by `model_version.retired_levers`. `--distill-coef`, `--distill-teacher`,
#   `--search-teacher` and the rest were never recorded fields, so a run that used them under the
#   DEFAULT knobs leaves no trace here (its recorded argv fails argparse on an unpinned resume
#   instead). Every v121+ run on record recorded the knobs at their defaults: no ARCH_SIGNATURE
#   bump, no MIGRATION_FLOOR change.
# v134 (deletion pass L4, owner-approved 2026-10-02): the COUNTERFACTUAL TRAINING HALF leaves the config —
#   the ten v100 cf coefficients (cf_records, cf_records_keep, cf_winprob_coef, cf_head_only,
#   cf_label_lag_steps, cf_label_likelihood, cf_evidential_coef, cf_evidential_reg, cf_twin_coef,
#   cf_shadow_coef), the two v107 Q coefficients (q_winprob_coef, q_winprob_onpolicy_coef) and the FOUR
#   STRUCTURAL head toggles they supervised (cf_evidential v98, cf_twin_heads / cf_shadow_critic v99,
#   q_winprob_mode v107). The coefficients are TRAINING-only: `_migrate_config` POPs them from any
#   vintage and a RESUME or FORK of a run that recorded one live is refused by
#   `model_version.retired_levers`. The four toggles named MODULES in the state_dict that the
#   surviving extractor has no home for, so a config recording one ON is REFUSED on every load
#   (`refuse_structural`) and `snapshot._DEAD_FEK_JUDGED` carries the pickled extractor kwargs.
#   `--team-pfsp` and `--exploiter-ladder` were never recorded fields. Every v121+ run on record
#   recorded all of it OFF: no ARCH_SIGNATURE bump, no MIGRATION_FLOOR change.
# v135 (deletion pass P11c, owner rule 2026-10-03: a flag with no live user is deleted): THREE recorded fields
#   leave the config — `pair_value_route` (v95, STRUCTURAL: a zero-init `PairValueInject` projection inside
#   CLSPool, a state_dict key), `td_aux_coef` (v90) and `win_prob_strata_weight` (v115), both TRAINING-only
#   loss knobs. `_migrate_config` POPs all three from any vintage; a config recording `pair_value_route`
#   ON is REFUSED on every load (`refuse_structural`) and `snapshot._DEAD_FEK_JUDGED` carries the pickled
#   extractor kwarg; a RESUME or FORK of a run that recorded a non-zero `td_aux_coef` /
#   `win_prob_strata_weight` is refused by `model_version.retired_levers`. Every v121+ run on record
#   recorded all three OFF: no ARCH_SIGNATURE bump, no MIGRATION_FLOOR change.
# v136 (gen3_x5_hypothesis_set_v1, X5 build unit U2; designs/endstate/design_x5_belief_tokens.md §3.8):
#   `belief_tokens` — `--belief-tokens {blob,fixed_mass}`. STRUCTURAL: `fixed_mass` builds the T0
#   hypothesis builder (δ_θ, OTHER — the state_dict delta) and re-targets the hidden-team belief
#   supervision to the set BCE; `blob` (the default, production until the X5 A/B rules) builds nothing
#   and is byte-identical to v135. A pre-v136 config migrates to "blob" (the only possible past). No
#   ARCH_SIGNATURE bump while both arms build at one commit — it comes with the losing arm's deletion
#   (design §3.8). No MIGRATION_FLOOR change.
# v137 (gen3_oracle_reveal_v1, the X5 A/B's oracle reference arms; designs/endstate/design_x5_belief_tokens.md §7.6):
#   `oracle_reveal` — `--oracle-reveal {off,species,full}`, a DIAGNOSTIC observation mode (never production). The Rust
#   encoder writes the opponent's true species into the opponent block of the OBSERVATION itself. RESUME-IMMUTABLE:
#   no module and no weight, so the forward is bit-identical and `check_compatible` does not gate it; what the
#   input MEANS differs, so `check_oracle_reveal` refuses a resume that flips it (a flagless resume inherits it).
#   A pre-v137 config migrates to "off" (the only possible past). No ARCH_SIGNATURE bump, no MIGRATION_FLOOR change.
#   The `full` level (the opponent's whole set) joined the same field without a bump: the recorded value is a string.
# v138 (gen3_policy_readout_trunk_v1, architecture audit F2; designs/endstate/design_arch_audit.md):
#   `policy_readout` — `--policy-readout {tower,trunk}`. STRUCTURAL: `trunk` retires the flat policy tower
#   (`pre_proj_norm` / `projection` + `mlp_extractor.policy_net`), builds the trunk state query
#   (`pools.PolicyStateQuery`) and widens the pointer scorers to TRUNK_POINTER_HIDDEN; `tower` (the default,
#   production) builds nothing and is byte-identical to v137. A pre-v138 config migrates to "tower" (the only
#   possible past). No ARCH_SIGNATURE bump while both modes build at one commit; no MIGRATION_FLOOR change.
# v139 (gen3_static_tokens_v1; designs/endstate/design_static_tokens.md): `token_encoding` —
#   `--token-encoding {legacy,static}`. STRUCTURAL: `static` builds `StaticTokenEncoder` (the static identity S
#   + the dynamic per-mon state D, no board fact) in `pokemon_encoder`'s place; `legacy` (the default,
#   production until the screen rules) builds `PokemonEncoder` and is byte-identical to v138. A pre-v139 config
#   migrates to "legacy" (the only possible past). No ARCH_SIGNATURE bump while both encodings build at one
#   commit. No MIGRATION_FLOOR change.
# v140 (gen3_static_board_v1, static-token build stage 2; designs/endstate/design_static_tokens.md §4): no new
#   field — what `token_encoding="static"` BUILDS changed: the global token is replaced by three BOARD tokens
#   (OUR SIDE / THEIR SIDE through one shared `side_proj`, FIELD), `x` → own side and `g` / `c4` → FIELD, the
#   critic's pool reads the three board rows, the tower's `non_matchup_rest` bypass is deleted, and the per-mon
#   OP CONTENT (`op_content`) is added on both sides. A pre-v140 config recording `static` (a stage-1 layout, never
#   trained) is REFUSED (its state_dict has no home); `legacy` stamps through unchanged (byte-identical). No
#   ARCH_SIGNATURE bump while both encodings build. No MIGRATION_FLOOR change.
# v141 (gen3_move_resolution_v1): `move_resolution` — F11's MOVE-RESOLUTION family (architecture audit §9; owner
#   2026-10-06: facts kept, judgments dropped). A STRUCTURAL string {off,on}: 'on' builds `MoveResolutionCell` and
#   retires the seven per-action blocks, gated in check_compatible. A pre-v141 config migrates to "off" (the only
#   possible past). No ARCH_SIGNATURE bump ('off' is byte-identical), no MIGRATION_FLOOR change.
# v142 (gen3_value_threat_inject_off_v1): no new field — what `value_threat_inject=False` BUILDS changed (architecture
#   audit F10, owner 2026-10-06: delete the critic's token-content threat route, read critic discrimination). With the
#   op built, OFF now constructs the projection (its init draw) and the policy RETIRES it after SB3's orthogonal
#   re-init, so every other parameter's initial bytes equal production's (ON) and `--value-threat-inject off` is a
#   one-lever arm. Before v142 OFF skipped the Linear and shifted ~185 later tensors' init draws. The state_dict is
#   unchanged in both modes, so a pre-v142 config stamps through (a load never re-draws an init). No ARCH_SIGNATURE
#   bump, no MIGRATION_FLOOR change.
# v143 (gen3_speed_physics_v1): `speed_physics` — architecture audit F7b (owner 2026-10-06): the op's P(we act
#   first) from the speed BELIEF + the exact gen-3 order rules (`move_order`) instead of `_DMG_SPEED_SCALE`'s
#   logistic. A STRUCTURAL string {off,on} with NO parameters, gated in check_compatible. A pre-v143 config migrates
#   to "off" (the only possible past). No ARCH_SIGNATURE bump ('off' is byte-identical), no MIGRATION_FLOOR change.
# v144 (gen3_x5_version_break_v1 — THE X5 VERSION BREAK; designs/endstate/design_x5_belief_tokens.md Decision record
#   2026-10-07 + §3.8, `model_version/version_break.py`): the ONE planned checkpoint break after X5's adoption. ONE
#   bump carries every part of the break (later parts append below, no further bump). ARCH_SIGNATURE ->
#   "gen3_x5_version_break_v1" and MIGRATION_FLOOR -> 144: NO pre-break checkpoint (blob or fixed_mass) is
#   reproducible at HEAD, so every one is refused at the floor with the belief-specific reason and runs PINNED
#   (`version_break.LAST_BLOB_COMMIT`).
#   Part 1 — the X5 flip + the blob deletion: X5's hypothesis tokens (the old `--belief-tokens fixed_mass`) are the
#     ONLY belief representation. `belief_tokens` leaves the config, the flag, the extractor kwarg and the registry;
#     the blob path's forward and loss branches are deleted. A pickled `belief_tokens='fixed_mass'` pops, any other
#     value is refused (`snapshot._DEAD_FEK_JUDGED`). The opponent-belief family (`opp_belief_slots` /
#     `opp_intent`) now builds X5 whenever it is on and REFUSES a configuration missing one of X5's requirements
#     (`t0_species_prior`, `move_belief_mode`, `move_prior_fusion`, `opp_intent`, `opp_belief_slots`,
#     `entity_tail_seats`). The
#     production model is byte-identical to the pre-break fixed_mass arm (K9 init / post hashes unchanged).
#   Part 2 — the EXACT-refactor bundle (architecture audit F1 / F6a / F7a / F16b + the blob leftovers): the dead SB3
#     value tower is DELETED (`value_pre_norm` / `value_projection`, `mlp_extractor.value_net`, `value_net`: 592,129
#     parameters; the extractor's value half IS `value_pooled`, the policy builds its own actor-only stack and a
#     `critic` other than winprob is refused), the flat pointer's shared scorer has no bias (F16b, -1), every
#     value-reduction max is `index_max.max_by_index` (F6a), the off-path speed-spread lookups are gone (F7a), and
#     `BeliefSlots` / `AlphaIntentHead` / `BetaSwitchHead` (constructed only for their RNG draws) and
#     `--beta-setvalued-coef` are deleted. 3,111,176 -> 2,519,046 production parameters. The INIT bytes move (the
#     K9 golden is re-recorded once, at the end of the break); the weight-mapping identity proof
#     (`designs/research_state/measurements/version_break_identity_2026-10-07/`) is bitwise except F16b's
#     softmax-shift rounding (log pi max |d| 2.4e-7).
#   Part 4 — the slot-tied `out_gain` (design_arch_audit §9.4): the op's learned gain is ONE scalar per (block region,
#     channel), shared across REQUEST SLOTS / move seats (`damage_op_layout.out_gain_channel_keys`, expanded by a
#     non-persistent one-hot `_out_gain_tie`); `damage_op.out_gain` 138 -> 99 on the production op (2,519,046 ->
#     2,519,007 production parameters). The mon-axis replicates stay per position. Init forward bitwise equal.
#   Part 5 — the pre-gain read: every op consumer that reads an op value AS physics reads it PRE-gain through the
#     op's live `last_raw_tensors` view — `intent_conditional` (high roll, P(first), flinch) in BOTH speed modes
#     (the `--speed-physics on`-only special case deleted) and the move-resolution family (P(first), was the detached
#     `last_raw_block`). F7a completed: the X5 OTHER roster's dead `spe_std` / `with_spe_std` deleted.
#   Part 3 — the OBS-FACTS APPEND (gen3_obs_facts_v1; designs/endstate/design_entity_coverage_audit.md §8): the
#     84-dim OBS-FACTS block (what the opponent has seen of our team, the opponent active's Choice-lock evidence,
#     the actives' Encore / Taunt / Disable / Uproar / partial-trap turns, each side's screen turns) is APPENDED as
#     the observation's last block (2761 -> 2845; the 2761-dim prefix byte-identical), and `obs_facts` —
#     `--obs-facts {off,v1}` — records whether the model READS it. STRUCTURAL: `v1` builds the zero-init
#     `ObsFactsInject` (`agents/model/obs_facts_inject.py`); `off` (production) builds nothing. `total_dim` (a
#     `_WEIGHT_FIELDS` entry) carries the observation break too. No migration branch: the field is new AT v144,
#     and every pre-144 config is refused at the floor.
# v145 (gen3_mon_tied_gain_v1; owner 2026-10-07: "fix those non-equivariant knobs"): the op's learned `out_gain` is
#   tied across EVERY position axis — our TEAM SLOTS and their mons as well as the request slots / move seats part 4
#   tied: the incoming per-mon rows 72 -> 12, the Choice-Band tail 12 -> 2, and in the render arm the outgoing
#   matrix's per-their-mon cells / revealed bits and the incoming matrix's per-our-mon cells. Production
#   `damage_op.out_gain` 99 -> 29 (render arm 222 -> 92). No lead-mon special case (owner: "in human games it isn't
#   super strategic"). Landed before any v144 checkpoint was trained; ARCH_SIGNATURE -> "gen3_mon_tied_gain_v1" and
#   MIGRATION_FLOOR -> 145, so a v144 config is refused with `version_break.v144_reason()` and runs PINNED
#   (`version_break.LAST_V144_COMMIT`). Init forward bitwise equal (each tied channel's init was equal per slot).
# v146 (gen3_op_reduction_principled_v1): `op_reduction` — architecture audit F6b (owner 2026-10-08): the op's
#   per-channel hard maxima over the opponent's believed moves (the incoming per-mon row, its argmax-picked accuracy /
#   provenance, the C1b / C2 / C3 / D4 kernels, the E5 tail's worst-phys/spec, the Pursuit presence) replaced by the
#   alpha-weighted EXPECTATION (alpha = presence / total presence, one mixture per attacker) and the noisy-OR worst
#   case (P(some move KOs), delivered by the zero-init `op_worst_proj`; P(some mon holds Pursuit)). A STRUCTURAL
#   string {max,principled}, gated in check_compatible; a v145 config migrates to "max" (the only possible past). No
#   ARCH_SIGNATURE bump ('max' is byte-identical), no MIGRATION_FLOOR change.
# v147 (gen3_static_port_v1): `--token-encoding static` ported to the post-break graph, made EQUIVARIANT where it was
#   not (S reads the type SET as a sum, not an alphabetical concat; the op content's outgoing route is a Deep Sets
#   function of our moves, not a request-ordered Linear(24 -> 128)), plus two narrow facts for it, each a STRUCTURAL
#   string {off,on} gated in check_compatible: `mon_hazard_cost` (every mon's own side's Spikes layers + its switch-in
#   HP cost, the op's ONE entry rule) and `move_actor_state` (our active's HP + status onto its E3 move seats). A
#   v145 / v146 `static` record is REFUSED (its weights have no home: no such checkpoint exists, archive scan
#   2026-10-09); `legacy` stamps through and both new fields migrate to "off". No ARCH_SIGNATURE bump (production is
#   `legacy`, byte-identical), no MIGRATION_FLOOR change.
# v148 (gen3_spikes_entry_base_types_v1; GIGO fix 2026-10-09): no field, no weight shape. `DamageOperator.spikes_entry`
#   (the `x` edge cell's chip + grounded bit, and `--mon-hazard-cost`'s fraction) decides Flying immunity from the
#   mon's BASE (species) types, not the obs type columns, which hold CURRENT types (Color Change, Transform,
#   Conversion, Castform's Forecast): a switch-IN follows `clearVolatile` -> `setSpecies(baseSpecies)`. Production's
#   `x` cell changes for an ACTIVE mon whose current types differ from its species'; the state_dict is unchanged and
#   every past config stamps through. No ARCH_SIGNATURE bump, no MIGRATION_FLOOR change.
# v149 (gen3_spikes_entry_species_levitate_v1): no field, no weight shape. The same rule's Levitate half: `spikes_entry`
#   reads Levitate from the SPECIES (`SPECIES_TRAP_PRIOR[:, 3]`, exactly 0 or 1 in gen 3 -- Levitate is the sole ability
#   of its 17 species) instead of the CURRENT-ability column, which Trace / Role Play / Skill Swap / Transform change
#   while a switch-in reverts them. Production's `x` cell changes for an active mon whose current ability differs from
#   its species'; every past config stamps through. No ARCH_SIGNATURE bump, no MIGRATION_FLOOR change.
# v150 (gen3_static_recovery_v1): three STATIC-RECOVERY levers, each its own field, production at the old value:
#   `trunk_layers` (int; 2 = production; 3 / 4 append identity-init pre-LN trunk rounds), `switch_hazard_cost` {off,on}
#   (the switch pointer cell's entry-hazard block, read by the pointer head's zero-init `switch_extra_proj`) and
#   `eot_residual` {off,on} (every mon's end-of-turn HP change, zero-init token content). Each is STRUCTURAL and
#   gated in check_compatible; a pre-v150 config migrates to 2 / off / off (the only possible past). No ARCH_SIGNATURE
#   bump (production builds byte-identically), no MIGRATION_FLOOR change.
# v151 (gen3_toxic_stage_scale_v1 + gen3_wish_flag_truth_v1): no field, no weight shape; two OBSERVATION-VALUE fixes.
#   (1) The per-mon toxic counter cell
#   (`POKEMON_COUNTER_OFFSET + 1`) was min(stage, 8) / 8 while Showdown's `tox.onResidual` runs the stage to 15 (damage
#   floor(maxhp/16) * stage), so every tick from the 9th on read as the 8th and every reader of the cell (the `g` ledger,
#   `--eot-residual`) under-priced it. It is now min(stage, 15) / 15 (`TOXIC_STAGE_MAX`); `damage_op_pairwise.toxic_next_tick`
#   decodes it (next tick = min(stage + 1, 15) / 16). Every toxic mon's cell value changes (n/15, not n/8), so a checkpoint
#   trained on the old scale reads a re-scaled input from here on. (2) The board's "Wish pending" flag read "a Wish cast last
#   turn" at every decision, which is wrong at a replacement decision AFTER the end-of-turn faint (the turn number is the one
#   whose residual already ran): it kept a Wish that had just landed and missed one cast that turn; it is now "a Wish lands
#   at the next end-of-turn residual" (`WishFold::pending(turn, residual_done)`), equal to the engine's slot condition.
#   Every past config stamps through. No ARCH_SIGNATURE bump, no MIGRATION_FLOOR change, state_dict unchanged. The obs
#   golden moved on 107 of 991 decisions: 106 on the toxic cell alone and 1 on the toxic cell plus the Wish flag.
# v152 (gen3_endstate_facts_v1): five FACT-COMPLETION levers, each its own STRUCTURAL field, production at the old
#   value: `move_resolution_facts` {off,full}, `status_facts` {off,exact}, `ko_ramp` {ramp,exact},
#   `drop_progress_clock` {off,on}, `g_ledger` {coarse,eot}; each gated in check_compatible; a pre-v152 config
#   migrates to off / off / ramp / off / coarse (the only possible past). No ARCH_SIGNATURE bump (production builds
#   byte-identically), no MIGRATION_FLOOR change.
# v153 (gen3_probe_facts_v1): two STRUCTURAL fields, production at the old value: `effective_stats` {off,on} (each side's
#   ACTIVE mon's stage-applied stats as zero-init token content) and `move_target_state` {off,on} (their active's HP +
#   status onto our E3 seats). Each gated in check_compatible; a pre-v153 config migrates to off / off. No
#   ARCH_SIGNATURE bump, no MIGRATION_FLOOR change.
MODEL_CONFIG_VERSION = 153

# THE OBS-SEMANTICS MARKER (declared 2026-10-09, with v151): the FIRST config version whose OBSERVATION VALUES
# (the vector the encoder writes, and the model-input cells derived from it) MEAN something different from every
# earlier version's, with or without a shape change. `ARCH_SIGNATURE` / `MIGRATION_FLOOR` catch a checkpoint whose
# WEIGHTS no longer fit; nothing caught one whose weights still load and whose INPUTS were re-scaled or re-defined
# underneath it (v151's Toxic counter cell n/8 -> n/15 and the Wish flag; v148 / v149's `x` cell fixes before it). A
# reader that must answer "is this checkpoint reading the observation it trained on" (the prober's model views,
# `main.prober.arch_status`) compares the checkpoint's RECORDED `config_version` against this: below it = older
# input semantics = NOT current, even when the state_dict loads. Nothing here gates a resume or an opponent load
# (a resumed run is meant to continue on the new inputs; `check_compatible` is unchanged).
#
# THE RULE: a commit that changes what an observation cell MEANS without changing a shape sets this to the config
# version it stamps, and re-pins `src/agents/model/obs_semantics_test.py`'s golden hash. That test fails when the obs
# golden (`agents/training/golden_obs_fixture.json`) moves without it. A change the golden cannot see (a
# model-INTERNAL input such as an op edge cell) must be raised here by hand.
OBS_SEMANTICS_VERSION = 151

# What changed at `OBS_SEMANTICS_VERSION`, in one clause — rendered in the prober's drift diagnosis.
OBS_SEMANTICS_REASON = ("the Toxic counter cell is now stage/15 where it was stage/8, and the Wish flag now means "
                        "'lands at the next end-of-turn residual'")

# The one-line effect of each `belief_grad_mode`, for the migration notice. Keyed by the SAME strings
# as `features_extractor.BELIEF_GRAD_MODES` (which owns the legal set + the ValueError); the two are
# pinned to agree by `belief_grad_mode_test.py::test_every_mode_has_a_migration_notice`, so a fourth
# mode cannot ship with a silently generic notice.
_BELIEF_GRAD_MODE_EFFECT = {
    "shaping": "the belief-aux gradient now SHAPES the shared trunk, and PPO trains the heads.",
    "detached": "the belief-aux gradient now STOPS at the heads (the trunk is stop-grad on the read).",
    "label_only": "the belief heads are now trained by their SUPERVISED LABELS ALONE — no policy/value "
                  "gradient reaches them (their outputs are published stop-grad to every consumer).",
}

# Change this when the neural architecture changes structurally in a way that makes
# weights from a different signature incompatible (e.g. adding LSTM, replacing attention).
# Same-family dim changes (role_token_size 128→256) don't need a new signature —
# check_compatible() catches those via the dim fields.
#
# The signature-by-signature history (v2 -> gen3_ctx_dedup_v1: what broke weight
# compatibility each time, and why) lives in designs/CHANGELOG.md under 'The
# ARCH_SIGNATURE narrative' — moved there 2026-08-16.
ARCH_SIGNATURE = "gen3_mon_tied_gain_v1"
class ModelVersionError(Exception):
    pass


# The resume-IMMUTABLE reward hparams, in the order `check_reward_config` reports them, each mapped
# to the value a config-shaped object is read with when it lacks the field. The DEFAULTS here track
# `agents.training.reward_manager.RewardConfig` and are pinned against it by
# `src/main/reward_defaults_test.py` — a divergence would make an absent field mean one thing to the
# reward and another to the version record, which is the drift class this whole file guards.
_REWARD_IMMUTABLE_FIELDS: Dict[str, Any] = {
    "draw_penalty": -35.0,
    "victory_value": 30.0,
    "progress_decision_tense": False,
    "progress_switch_freeze": False,
    # gen3_winprob_critic_mode_v1 — the default is today's behaviour, so a pre-v109 config
    # migrates to it and a flagless resume of any existing run is unchanged.
    "terminal_indicator": False,
    # (The 14 SHAPED-reward fields left this table at v122, gen3_shaped_reward_deletion_v1 — see
    # `model_version.shaped_reward.DELETED_SHAPED_REWARD_FIELDS`.)
}
