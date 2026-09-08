# Memory audit — 2026-09-08

A full pass over the orchestrator's persistent memory
(`~/.claude/projects/-home-goodlad-dev-gen3ai/memory/`): every file read, classified, and acted on.
The goal was a **smaller, truer** index — `MEMORY.md` loads into every session, including every
subagent's, so a line there is paid for on every turn of every agent.

**Nothing was deleted.** Every retired file is in
[`../memory_archive/`](../memory_archive/) with a one-line `> **Archived 2026-09-08**` header
naming what supersedes it; the merged one lives on inside its survivor.

## Summary

| status | meaning | n |
|---|---|---|
| **LIVE** | still changes what a session does today — kept as-is | 75 |
| **POINTER** | body is already a pointer at `designs/ops/ORCHESTRATOR_SOP.md` / `TRAINING_RUN_SOP.md` (converted 2026-09-07) — kept, they are linked by name | 21 |
| **STALE-REFERENCE (fixed)** | named a file / flag / command that no longer resolves — reference repaired or the sentence marked `**STALE:**` | 12 |
| **CLOSED-ERA** | a research finding from an era `UNDERSTANDING.md` §1 marks CLOSED, whose operative fact an always-current doc now states — **archived** | 55 |
| **SUPERSEDED** | a later memory states a newer verdict (each of these says so in its own banner) — **archived** | 3 |
| **DUPLICATE → merged** | overlapped a survivor; merged into it, then archived as the record | 1 |
| | **total memory files audited** | **167** |

`MEMORY.md` itself is the 168th file: **16,447 bytes → 16,447 bytes**.

**Two things the audit found that are worth carrying:**

1. **`ledger_index.md` begins at 2026-08-01.** For every pre-August memory the ledger does *not*
   hold the verdict — the memory file is the only record. That is the argument for the archive
   directory rather than deletion, and it is why every pre-August retirement below cites an
   always-current *document* (an `ARCHITECTURE.md` fact, a leaf `CLAUDE.md`, a standing
   `feedback_*` rule) rather than a ledger line.
2. **The `probes/` directory does not exist anywhere.** Seven memories cite it as the reference
   implementation for the extraction / fold / teacher-ceiling readouts. It was a session
   scratchpad. The numbers those memories record stand; the scripts do not, and the committed
   successors are `python -m main.exploitability`, `main.untaught_meter`, `main.dose`.

## The table

Archived files are linked to their new home. Sizes are of the file as audited.

| memory | type | bytes | last touched | status | note |
|---|---|---:|---|---|---|
| `feedback_agent_report_hazards_are_findings` | feedback | 659 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_agent_stream_stalls` | feedback | 794 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_arm_labels_need_human_description` | feedback | 655 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_cd_before_worktree_remove` | feedback | 1,953 | 2026-09-03 | LIVE |  |
| `feedback_challenge_the_owner` | feedback | 2,229 | 2026-09-08 | LIVE |  |
| `feedback_check_banked_factorials_before_promoting` | feedback | 1,440 | 2026-09-06 | LIVE |  |
| `feedback_data_source_of_truth` | feedback | 1,672 | 2026-09-08 | LIVE |  |
| [`feedback_delta_interval_before_writing`](../memory_archive/feedback_delta_interval_before_writing.md) | feedback | 2,175 | — | DUPLICATE → merged | MERGED into `feedback_equivalence_needs_delta_ci` — same rule, same two catches |
| `feedback_design_block_is_not_a_launch_command` | feedback | 3,704 | 2026-09-06 | LIVE |  |
| `feedback_determinism_means_fixable` | feedback | 4,899 | 2026-09-08 | LIVE |  |
| `feedback_docs_auto_update` | feedback | 1,380 | 2026-05-29 | LIVE |  |
| `feedback_dont_kill_training_server` | feedback | 2,206 | 2026-09-08 | LIVE |  |
| `feedback_dv_ablation_scope_limit` | feedback | 2,708 | 2026-09-08 | LIVE |  |
| `feedback_edge_case_regression_tests` | feedback | 1,895 | 2026-09-08 | LIVE |  |
| `feedback_edit_in_worktree_path` | feedback | 1,711 | 2026-05-31 | LIVE |  |
| `feedback_elo_reading_rules` | feedback | 2,984 | 2026-09-08 | LIVE |  |
| `feedback_equivalence_needs_delta_ci` | feedback | 6,396 | 2026-09-08 | LIVE |  |
| `feedback_explain_to_teach` | feedback | 2,782 | 2026-08-16 | LIVE |  |
| `feedback_fallback_cron_55m` | feedback | 635 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_fuzz_tests` | feedback | 1,882 | 2026-09-08 | LIVE |  |
| `feedback_gigo_order_bugs_asap` | feedback | 3,423 | 2026-09-08 | LIVE |  |
| `feedback_git_workflow` | feedback | 1,970 | 2026-05-31 | LIVE |  |
| `feedback_ideation_session_operating_pattern` | feedback | 781 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_idle_cpu_means_deadlock` | feedback | 2,174 | 2026-09-08 | LIVE |  |
| `feedback_kill_by_pid` | feedback | 1,638 | 2026-08-26 | LIVE |  |
| `feedback_learning_notes_inline_by_default` | feedback | 1,380 | 2026-09-01 | LIVE |  |
| `feedback_long_run_sop` | feedback | 602 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_main_scratch_ignore` | feedback | 1,970 | 2026-08-01 | LIVE |  |
| `feedback_matched_extraction_row` | feedback | 2,634 | 2026-09-08 | STALE-REFERENCE (fixed) | `probes/extraction_matched.py` gone → `main.exploitability` named |
| `feedback_matched_noise_control` | feedback | 1,698 | 2026-08-28 | LIVE |  |
| `feedback_night_autonomy_keep_gpu_busy` | feedback | 698 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_no_auto_ship` | feedback | 5,859 | 2026-09-01 | LIVE |  |
| `feedback_no_auto_tech_debt` | feedback | 728 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_no_busywork_checks` | feedback | 688 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_no_circular_unwinnable_claims` | feedback | 2,582 | 2026-09-08 | LIVE |  |
| `feedback_no_training_run_monitors` | feedback | 721 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_notification_standing_order` | feedback | 569 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_notify_then_act_15m` | feedback | 646 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_offline_probe_checkpoint_resolution` | feedback | 1,420 | 2026-09-05 | LIVE |  |
| `feedback_owner_autonomy_no_cap` | feedback | 742 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_pooled_correlation_simpson_trap` | feedback | 1,617 | 2026-08-17 | LIVE |  |
| `feedback_prober_self_improvement` | feedback | 2,417 | 2026-09-08 | STALE-REFERENCE (fixed) | Textual TUI retired + `list/overview/find` → the real subcommand list; `prober/web/` named |
| `feedback_provide_vs_learn` | feedback | 2,348 | 2026-09-08 | LIVE |  |
| `feedback_ptrace_debug` | feedback | 1,191 | 2026-06-02 | LIVE |  |
| `feedback_quota_pacing` | feedback | 647 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_relay_direct_to_training_session` | feedback | 735 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_replicate_means_pin_and_seed` | feedback | 2,124 | 2026-09-08 | LIVE |  |
| `feedback_report_at_design_doc_level` | feedback | 697 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_research_state` | feedback | 4,697 | 2026-09-08 | LIVE |  |
| `feedback_role_division_ideation_vs_running` | feedback | 811 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_smogon_priors_only` | feedback | 2,267 | 2026-09-08 | STALE-REFERENCE (fixed) | `tmp/belief_coupling_lift.py` gone; measurement JSON kept |
| `feedback_strength_first_principles` | feedback | 3,277 | 2026-09-08 | LIVE |  |
| `feedback_subagents_opus_only` | feedback | 642 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_training_run_autonomy_grant` | feedback | 620 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_validate_by_executing` | feedback | 5,287 | 2026-09-06 | LIVE |  |
| `feedback_validate_observable_bytes` | feedback | 2,598 | 2026-09-08 | LIVE |  |
| `feedback_waiting_on_background_work` | feedback | 762 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `feedback_what_is_the_baseline` | feedback | 2,340 | 2026-09-06 | LIVE |  |
| `feedback_workflow_stall_ms` | feedback | 736 | 2026-09-07 | POINTER | body is a pointer to `designs/ops/*_SOP.md` |
| `project_2x2_teacher_content_batch` | project | 8,110 | 2026-09-08 | STALE-REFERENCE (fixed) | `tc_readout.py` path resolved into `measurements/teacher_content_2x2_2026-09-04/` |
| [`project_anti_stall_fix`](../memory_archive/project_anti_stall_fix.md) | project | 4,957 | — | CLOSED-ERA | ai_v5_5-era reward fix, shipped d7aa983 + a11f234; the mechanism lives in reward_manager.py / progress_clock.py and the win-prob critic (ai_v12) removes hand shaping entirely |
| [`project_arch_compute_decision`](../memory_archive/project_arch_compute_decision.md) | project | 4,478 | — | CLOSED-ERA | ai_v5-era 23-token transformer decision; the entity-graph generation replaced that trunk — designs/ARCHITECTURE.md is the doc of record |
| `project_arch_transfer_validation` | project | 11,427 | 2026-09-06 | LIVE |  |
| [`project_archetype_competence_gradient`](../memory_archive/project_archetype_competence_gradient.md) | project | 9,083 | — | CLOSED-ERA | ai_v6_13/ai_v7_02-era archetype probe; era closed per UNDERSTANDING.md §1 |
| `project_bait_verdict_final` | project | 4,569 | 2026-09-02 | LIVE |  |
| `project_baton_pass_gigo` | project | 1,928 | 2026-08-23 | LIVE |  |
| [`project_battle_reconstruction`](../memory_archive/project_battle_reconstruction.md) | project | 9,093 | — | CLOSED-ERA | reconstruction + falsifier SHIPPED; src/main/prober/CLAUDE.md is the always-current doc of record |
| [`project_belief_latent_role_probe`](../memory_archive/project_belief_latent_role_probe.md) | project | 8,410 | — | CLOSED-ERA | ai_v5-era latent-belief design exploration; superseded by the shipped belief stack and designs/ARCHITECTURE.md |
| [`project_belief_shaping_experiment`](../memory_archive/project_belief_shaping_experiment.md) | project | 9,629 | — | CLOSED-ERA | ai_v7/ai_v8-era belief-grad-mode arms; era closed, --belief-grad-mode semantics live in src/agents/training/CLAUDE.md |
| [`project_belief_toggle_flags`](../memory_archive/project_belief_toggle_flags.md) | project | 3,232 | — | CLOSED-ERA | ai_v5 Step-3 design note; the toggles shipped and live in designs/ARCHITECTURE.md's flag table |
| [`project_better_line_search`](../memory_archive/project_better_line_search.md) | project | 4,878 | — | CLOSED-ERA | BUILT+shipped; superseded operationally by project_rust_search_driver and src/main/prober/CLAUDE.md |
| [`project_bridge_training_transport`](../memory_archive/project_bridge_training_transport.md) | project | 9,209 | — | CLOSED-ERA | --use-showdown-bridge era; superseded by --use-bridge rust as the DEFAULT transport (root CLAUDE.md) |
| `project_claude_md_restructure` | project | 3,288 | 2026-09-06 | LIVE |  |
| [`project_code_rank_ceiling`](../memory_archive/project_code_rank_ceiling.md) | project | 9,702 | — | SUPERSEDED | SUPERSEDED by project_lut_conditioning_ceiling_result (its own banner says so) and project_count_dominates_conditioning |
| `project_contention_robust_timeouts` | project | 6,829 | 2026-09-08 | LIVE |  |
| `project_count_dominates_conditioning` | project | 4,957 | 2026-07-28 | LIVE |  |
| `project_counterfactual_label_costs` | project | 4,077 | 2026-08-22 | LIVE |  |
| [`project_counterfactual_prober`](../memory_archive/project_counterfactual_prober.md) | project | 4,890 | — | CLOSED-ERA | BUILT+shipped; src/main/prober/CLAUDE.md is the doc of record |
| `project_cpu_starvation_tmux_cgroups` | project | 4,377 | 2026-08-14 | LIVE |  |
| [`project_damage_op_block_audit`](../memory_archive/project_damage_op_block_audit.md) | project | 14,218 | — | CLOSED-ERA | ai_v7-era op audit; the refine loop it audits was DELETED at v50 and the era is closed |
| [`project_damage_op_prefuse_v50`](../memory_archive/project_damage_op_prefuse_v50.md) | project | 3,270 | — | CLOSED-ERA | v50 build record; prefuse is production and stated in designs/ARCHITECTURE.md |
| [`project_defensive_entropy_built`](../memory_archive/project_defensive_entropy_built.md) | project | 4,011 | — | CLOSED-ERA | ai_v6-era exploration lever, built and never adopted; era closed |
| `project_distill_retention_ablation` | project | 2,771 | 2026-07-26 | LIVE |  |
| `project_distillability_index` | project | 2,529 | 2026-08-27 | LIVE |  |
| [`project_distributional_critic_verdict`](../memory_archive/project_distributional_critic_verdict.md) | project | 2,810 | — | CLOSED-ERA | verdict NO; the value_dist head is DELETED under the ai_v12 win-prob critic |
| [`project_double_sided_recipe`](../memory_archive/project_double_sided_recipe.md) | project | 2,337 | — | CLOSED-ERA | ai_v7-era distill+stable-opponent recipe; the flywheel account is restated in UNDERSTANDING.md §2 |
| [`project_event_sourced_migration`](../memory_archive/project_event_sourced_migration.md) | project | 4,622 | — | CLOSED-ERA | ai_v4 closed out and shipped; src/agents/battle/CLAUDE.md is the always-current doc of record |
| `project_exploiter_fork_vs_scratch` | project | 6,546 | 2026-09-08 | LIVE |  |
| [`project_exploiter_league_tooling`](../memory_archive/project_exploiter_league_tooling.md) | project | 7,343 | — | CLOSED-ERA | ai_v6-era tooling, shipped; flags documented in src/agents/training/CLAUDE.md |
| [`project_exploiter_no_team_advantage`](../memory_archive/project_exploiter_no_team_advantage.md) | project | 3,484 | — | CLOSED-ERA | the live rule survives as feedback_no_circular_unwinnable_claims |
| [`project_floor_leak_critic_selfko`](../memory_archive/project_floor_leak_critic_selfko.md) | project | 3,240 | — | CLOSED-ERA | ai_v5-era critic forensics; era closed, critic replaced by the win-prob head (ai_v12) |
| `project_fold_failure_eliminations` | project | 4,868 | 2026-09-08 | STALE-REFERENCE (fixed) | `probes/` scratchpad dir no longer exists — marked **STALE:** |
| `project_fold_transfer_is_local` | project | 3,483 | 2026-09-08 | STALE-REFERENCE (fixed) | `probes/` scratchpad dir no longer exists — marked **STALE:** |
| `project_fork_inert_flags` | project | 3,609 | 2026-09-08 | STALE-REFERENCE (fixed) | `probes/` scratchpad dir no longer exists — marked **STALE:** |
| [`project_fresh_generation_equivariance`](../memory_archive/project_fresh_generation_equivariance.md) | project | 25,859 | — | CLOSED-ERA | the ai_v9 gen-1..gen-6 build narrative; the resulting architecture is stated in designs/ARCHITECTURE.md and its ELO rule survives as feedback_elo_reading_rules |
| `project_g0_bias_map_verdict` | project | 3,629 | 2026-08-22 | LIVE |  |
| `project_g5d_io_infrastructure` | project | 1,018 | 2026-09-08 | STALE-REFERENCE (fixed) | `scripts/GCP_INFRASTRUCTURE.md` → `scripts/workstation/` |
| [`project_gen3_randoms_byte_parity`](../memory_archive/project_gen3_randoms_byte_parity.md) | project | 53,087 | — | CLOSED-ERA | rust randoms grind; src/rust_sim/CLAUDE.md + designs/rust_sim/port_build_log.md are the always-current docs of record |
| [`project_gpu_damage_op`](../memory_archive/project_gpu_damage_op.md) | project | 15,510 | — | CLOSED-ERA | Stage A/B build record; the damage op is production and stated in designs/ARCHITECTURE.md |
| [`project_hidden_team_belief_built`](../memory_archive/project_hidden_team_belief_built.md) | project | 4,268 | — | CLOSED-ERA | belief-slot build record; the head is production and stated in designs/ARCHITECTURE.md |
| [`project_incoming_damage_outcome`](../memory_archive/project_incoming_damage_outcome.md) | project | 12,156 | — | CLOSED-ERA | ai_v5-era obs-block forensics; era closed |
| [`project_l3_oracle_grind_l4`](../memory_archive/project_l3_oracle_grind_l4.md) | project | 5,976 | — | CLOSED-ERA | the durable rule (the amortizability gate, never search ON the model) lives in feedback_research_state |
| [`project_latent_belief_built`](../memory_archive/project_latent_belief_built.md) | project | 5,035 | — | CLOSED-ERA | branch build record, ai_v5 era; superseded by the shipped belief stack |
| [`project_local_sim_bridge`](../memory_archive/project_local_sim_bridge.md) | project | 2,631 | — | CLOSED-ERA | the bridge is the default transport; root CLAUDE.md and src/utils/bridge/ are the docs of record |
| [`project_loss_analysis_run20260601`](../memory_archive/project_loss_analysis_run20260601.md) | project | 8,453 | — | CLOSED-ERA | June-2026 forensic pass; siblings _run20260531 and _v2 were archived 2026-09-07 |
| `project_lut_conditioning_ceiling_result` | project | 4,892 | 2026-09-08 | LIVE |  |
| [`project_markovian_reward_design`](../memory_archive/project_markovian_reward_design.md) | project | 15,239 | — | CLOSED-ERA | PBRS/hand shaping is DELETED under the ai_v12 win-prob critic (--no-hand-shaping) |
| [`project_model_frontier_roadmap`](../memory_archive/project_model_frontier_roadmap.md) | project | 5,445 | — | CLOSED-ERA | ai_v5-era roadmap; superseded by designs/research_state/UNDERSTANDING.md |
| `project_models_disk_retention` | project | 3,351 | 2026-09-06 | LIVE |  |
| `project_multiteam_distill_payoff` | project | 5,202 | 2026-09-08 | LIVE |  |
| [`project_nature_ev_belief_built`](../memory_archive/project_nature_ev_belief_built.md) | project | 4,159 | — | CLOSED-ERA | v40 build record; the spread belief is production and stated in designs/ARCHITECTURE.md |
| `project_negative_transfer_verdict` | project | 4,791 | 2026-09-03 | LIVE |  |
| [`project_oa_cells_path_forward`](../memory_archive/project_oa_cells_path_forward.md) | project | 11,112 | — | SUPERSEDED | SUPERSEDED 2026-08-08 by the OpTensors path (its own banner says so) |
| [`project_op_move_order_bugclass`](../memory_archive/project_op_move_order_bugclass.md) | project | 3,879 | — | CLOSED-ERA | the durable rule survives as feedback_gigo_order_bugs_asap; the guards are in-tree |
| [`project_opd_built`](../memory_archive/project_opd_built.md) | project | 6,205 | — | CLOSED-ERA | ai_v7-era on-policy self-distillation, verdict NEUTRAL; era closed |
| [`project_opp_action_head_falsified`](../memory_archive/project_opp_action_head_falsified.md) | project | 4,053 | — | CLOSED-ERA | ai_v5-era falsification; era closed and the trunk it probed no longer exists |
| [`project_opp_hp_immune_bug`](../memory_archive/project_opp_hp_immune_bug.md) | project | 6,620 | — | CLOSED-ERA | GIGO fixed and shipped a4aa2bc; the class rule lives in feedback_gigo_order_bugs_asap |
| [`project_opponent_distillation_findings`](../memory_archive/project_opponent_distillation_findings.md) | project | 6,361 | — | CLOSED-ERA | ai_v5-era opponent distillation; superseded by torch.compile'd opponents and the rust bridge |
| `project_opus5_alias_remap` | project | 2,929 | 2026-08-16 | LIVE |  |
| [`project_outgoing_damage_design`](../memory_archive/project_outgoing_damage_design.md) | project | 7,425 | — | CLOSED-ERA | ai_v5-era obs block; era closed |
| `project_owner_direction_replicate_v8_gift` | project | 2,452 | 2026-09-05 | LIVE |  |
| `project_pfsp_phase1_built` | project | 5,750 | 2026-09-08 | LIVE |  |
| `project_plasticity_null` | project | 3,308 | 2026-08-27 | LIVE |  |
| [`project_plateau_research_2026_06_25`](../memory_archive/project_plateau_research_2026_06_25.md) | project | 27,759 | — | CLOSED-ERA | superseded by designs/research_state/UNDERSTANDING.md §1-§2 (the era map and the flywheel account) |
| [`project_popart`](../memory_archive/project_popart.md) | project | 4,680 | — | CLOSED-ERA | PopArt is SHIPPED and is OFF by construction under the ai_v12 win-prob critic; mechanism in designs/learning/popart_value_scale_and_currencies.md |
| [`project_public_value_poc`](../memory_archive/project_public_value_poc.md) | project | 6,325 | — | CLOSED-ERA | v43 pubval; owner verdict "did nothing" (recorded in project_damage_op_block_audit K9/K10) and the era is closed |
| [`project_refresh_status_cure_gap`](../memory_archive/project_refresh_status_cure_gap.md) | project | 8,337 | — | CLOSED-ERA | ai_v5/v6-era obs bits, shipped 92cf1ca/4ce3c72; era closed |
| [`project_representation_probe`](../memory_archive/project_representation_probe.md) | project | 4,607 | — | CLOSED-ERA | ai_v5-era probe harness, shipped; src/main/prober/CLAUDE.md is the doc of record |
| [`project_resume_optimizer_realign_bug`](../memory_archive/project_resume_optimizer_realign_bug.md) | project | 7,860 | — | CLOSED-ERA | fixed and hardened (name-keyed remap, ac5b93f); the guard is in-tree |
| [`project_reward_shaping_verification`](../memory_archive/project_reward_shaping_verification.md) | project | 1,884 | — | CLOSED-ERA | May-2026 method note; superseded by the reward golden (ledger 2026-09-08 gen3_reward_golden_v1) |
| [`project_rust_bridge_incremental`](../memory_archive/project_rust_bridge_incremental.md) | project | 14,951 | — | CLOSED-ERA | rust bridge wedges fixed; src/rust_sim/CLAUDE.md is the always-current doc of record |
| [`project_rust_bridge_training_enablement`](../memory_archive/project_rust_bridge_training_enablement.md) | project | 12,487 | — | CLOSED-ERA | rust IS the default training transport now (root CLAUDE.md); src/rust_sim/CLAUDE.md owns the detail |
| `project_rust_search_driver` | project | 2,353 | 2026-09-08 | LIVE |  |
| [`project_rust_sim_port`](../memory_archive/project_rust_sim_port.md) | project | 127,672 | — | CLOSED-ERA | 127 KB build log; src/rust_sim/CLAUDE.md + designs/rust_sim/port_build_log.md are the always-current docs of record |
| `project_sampling_snr_analysis` | project | 4,123 | 2026-09-08 | LIVE |  |
| `project_sb3_ortho_init_clobber` | project | 3,058 | 2026-09-08 | LIVE |  |
| `project_search_teacher` | project | 5,828 | 2026-09-08 | LIVE |  |
| `project_shape_and_exploitability` | project | 5,057 | 2026-09-08 | STALE-REFERENCE (fixed) | `probes/` scratchpad dir no longer exists — marked **STALE:** |
| [`project_spread_belief_supervision`](../memory_archive/project_spread_belief_supervision.md) | project | 5,726 | — | SUPERSEDED | SUPERSEDED by project_nature_ev_belief_built (generative nature/EV head replaced the point estimate) |
| `project_step_size_controllers_built` | project | 5,838 | 2026-09-03 | LIVE |  |
| `project_stream_idle_timeout_rootcause` | project | 20,324 | 2026-09-06 | LIVE |  |
| `project_substrate_before_flywheel` | project | 10,757 | 2026-09-08 | LIVE |  |
| `project_target_form_breakthrough` | project | 2,628 | 2026-08-26 | LIVE |  |
| `project_teacher_ceiling` | project | 4,523 | 2026-09-08 | STALE-REFERENCE (fixed) | `probes/` scratchpad dir no longer exists — marked **STALE:** |
| `project_teacher_fleet_geometry` | project | 4,295 | 2026-09-08 | STALE-REFERENCE (fixed) | `probes/` scratchpad dir no longer exists — marked **STALE:** |
| `project_teacher_resolution_last_snapshot` | project | 1,844 | 2026-09-06 | LIVE |  |
| [`project_team_pool_weighting_fix`](../memory_archive/project_team_pool_weighting_fix.md) | project | 2,872 | — | CLOSED-ERA | data fix landed; the 719-team pool count is stated in root CLAUDE.md |
| `project_three_axis_value_variance` | project | 5,803 | 2026-09-08 | LIVE |  |
| [`project_throughput_compile`](../memory_archive/project_throughput_compile.md) | project | 33,848 | — | CLOSED-ERA | 33 KB throughput archaeology; the operative facts (--compile-opponents / --compile-trainer ON by default) are in root CLAUDE.md and the training runbook |
| [`project_throughput_profile`](../memory_archive/project_throughput_profile.md) | project | 3,412 | — | CLOSED-ERA | ai_v5-era py-spy profile; superseded by project_throughput_compile and the rust bridge |
| `project_tick1_verdict_and_teacher_gate` | project | 2,467 | 2026-08-25 | LIVE |  |
| [`project_topk_incoming_moves`](../memory_archive/project_topk_incoming_moves.md) | project | 4,154 | — | CLOSED-ERA | v30 build record; the top-K block is production and stated in designs/ARCHITECTURE.md |
| `project_training_versions` | project | 2,838 | 2026-09-08 | STALE-REFERENCE (fixed) | 2026-06 phase snapshot marked **STALE:**; the rule survives |
| [`project_tss_specialist_poc`](../memory_archive/project_tss_specialist_poc.md) | project | 36,114 | — | CLOSED-ERA | ai_v7-era single-team specialist arc, double-corrected; era closed |
| [`project_unified_move_system`](../memory_archive/project_unified_move_system.md) | project | 6,268 | — | CLOSED-ERA | v24/v25 build record; the move system is production and stated in designs/ARCHITECTURE.md |
| `project_untaught_meter_axes` | project | 5,011 | 2026-09-06 | LIVE |  |
| `project_v8_gift_is_a_transient_hump` | project | 5,773 | 2026-09-06 | LIVE |  |
| `project_v8_line_replication` | project | 8,760 | 2026-09-08 | LIVE |  |
| `project_v8_reproduction_scorecard` | project | 3,541 | 2026-08-30 | LIVE |  |
| [`project_value_dist_head_status`](../memory_archive/project_value_dist_head_status.md) | project | 4,803 | — | CLOSED-ERA | the ValueDistHead is DELETED under the ai_v12 win-prob critic |
| [`project_value_distill_fitnet`](../memory_archive/project_value_distill_fitnet.md) | project | 3,015 | — | CLOSED-ERA | ai_v7-era value-distill A/B; era closed, and no gen fold ever ran v8_14's literal value-feat coef |
| `project_wake_cadence_measurement` | project | 1,437 | 2026-09-06 | LIVE |  |
| `project_week_goal_winprob_validation` | project | 4,878 | 2026-09-07 | LIVE |  |
| `project_winprob_era_live_run` | project | 8,778 | 2026-09-07 | LIVE |  |
| `project_winprob_only_critic` | project | 14,436 | 2026-09-06 | LIVE |  |
| `reference_wang2024_thesis` | reference | 4,100 | 2026-05-31 | LIVE |  |
| `user_gen3ou_ladder_context` | user | 1,635 | 2026-08-23 | LIVE |  |

