# Archived Claude memories

Memories retired from the agent memory index on **2026-09-07**, preserved verbatim.

They were removed because their subject is a CLOSED era, a landed fix, or a fact the
repository now records directly (a `CLAUDE.md`, `designs/ARCHITECTURE.md`, or the ledger) —
per the memory rule that a memory must not duplicate what the repo already says.

They are kept here rather than deleted because each records what was believed at the time,
which the ledger's append-only history depends on. **Nothing here is current.** Read the
ledger and `designs/research_state/UNDERSTANDING.md` for what is true now.

| file | retired because |
|---|---|
| `project_current_run.md` | GEN-14/17 era run notes, self-labelled historical; superseded by the win-prob critic era |
| `project_gen13_launch_plan.md` | gen-13 launched and closed |
| `project_gen14_preregistered.md` | gen-14 pre-registration, since executed |
| `project_loss_analysis_run20260531.md` + `_v2.md` | May-2026 forensic passes; superseded by `falsify` / `calibration` |
| `project_stall_recovery_analysis.md` | superseded by the anti-stall fix and the contention work |
| `project_plateau_diagnosis_2026_06_09.md` | superseded by `project_plateau_research_2026_06_25` |
| `project_loss_triage_tool.md` | the `triage` tool now documented in `src/main/prober/CLAUDE.md` |
| `project_eval_item_workstealing.md` | shipped; documented in `src/agents/training/CLAUDE.md` |
| `project_stable_opponents_design.md` | shipped; documented in `src/agents/training/CLAUDE.md` |
| `project_env_worker_memory_leak.md` | fixed |
| `project_bridge_unique_battle_tags.md` | fixed |
| `project_showdown_server_memory_growth.md` | moot — the in-process bridge is the default, no server |
| `project_human_agreement_probe.md` | ai_v6 behaviour-cloning track, not a live line |
| `project_next_run_plan.md` | pointed at `designs/ai_v8/next_run_plan.md`; that era is closed |
| `project_opponent_system_parity.md` | audit whose debts now live in `designs/ops/TECH_DEBT_BACKLOG.md` |
| `project_positional_grind_decomposition.md` | the live rule survives as `feedback_no_circular_unwinnable_claims` |
| `feedback_progress_cron_notifications.md` | fully covered by `feedback_long_run_sop` (mechanism) + `feedback_notification_standing_order` (policy) |

---

## Second retirement wave — **2026-09-08** (the memory audit)

Audit note: [`../measurements/memory_audit_2026-09-08.md`](../measurements/memory_audit_2026-09-08.md).
Ledger entry: *2026-09-08 · MEMORY AUDIT*.

Criterion for this wave: the file's subject is an **ai_v5 / ai_v6 / ai_v7 / ai_v8 era** build or
verdict (eras `UNDERSTANDING.md` §1 marks CLOSED), or the ai_v9 gen-1…gen-6 build narrative, **and**
the operative fact it carries is now stated by an always-current document — `designs/ARCHITECTURE.md`,
a leaf `CLAUDE.md`, `designs/research_state/UNDERSTANDING.md`, or a standing `feedback_*` memory.
Note that `ledger_index.md` begins at **2026-08-01**, so for the pre-August files this archive is the
ONLY record — which is exactly why they are archived rather than deleted.

Each file carries a one-line `> **Archived 2026-09-08** — …` header after its front matter.

| file | retired because |
|---|---|
| `project_anti_stall_fix.md` | ai_v5_5-era reward fix, shipped d7aa983 + a11f234; the mechanism lives in reward_manager.py / progress_clock.py and the win-prob critic (ai_v12) removes hand shaping entirely |
| `project_arch_compute_decision.md` | ai_v5-era 23-token transformer decision; the entity-graph generation replaced that trunk — designs/ARCHITECTURE.md is the doc of record |
| `project_archetype_competence_gradient.md` | ai_v6_13/ai_v7_02-era archetype probe; era closed per UNDERSTANDING.md §1 |
| `project_battle_reconstruction.md` | reconstruction + falsifier SHIPPED; src/main/prober/CLAUDE.md is the always-current doc of record |
| `project_belief_latent_role_probe.md` | ai_v5-era latent-belief design exploration; superseded by the shipped belief stack and designs/ARCHITECTURE.md |
| `project_belief_shaping_experiment.md` | ai_v7/ai_v8-era belief-grad-mode arms; era closed, --belief-grad-mode semantics live in src/agents/training/CLAUDE.md |
| `project_belief_toggle_flags.md` | ai_v5 Step-3 design note; the toggles shipped and live in designs/ARCHITECTURE.md's flag table |
| `project_better_line_search.md` | BUILT+shipped; superseded operationally by project_rust_search_driver and src/main/prober/CLAUDE.md |
| `project_bridge_training_transport.md` | --use-showdown-bridge era; superseded by --use-bridge rust as the DEFAULT transport (root CLAUDE.md) |
| `project_code_rank_ceiling.md` | SUPERSEDED by project_lut_conditioning_ceiling_result (its own banner says so) and project_count_dominates_conditioning |
| `project_counterfactual_prober.md` | BUILT+shipped; src/main/prober/CLAUDE.md is the doc of record |
| `project_damage_op_block_audit.md` | ai_v7-era op audit; the refine loop it audits was DELETED at v50 and the era is closed |
| `project_damage_op_prefuse_v50.md` | v50 build record; prefuse is production and stated in designs/ARCHITECTURE.md |
| `project_defensive_entropy_built.md` | ai_v6-era exploration lever, built and never adopted; era closed |
| `project_distributional_critic_verdict.md` | verdict NO; the value_dist head is DELETED under the ai_v12 win-prob critic |
| `project_double_sided_recipe.md` | ai_v7-era distill+stable-opponent recipe; the flywheel account is restated in UNDERSTANDING.md §2 |
| `project_event_sourced_migration.md` | ai_v4 closed out and shipped; src/agents/battle/CLAUDE.md is the always-current doc of record |
| `project_exploiter_league_tooling.md` | ai_v6-era tooling, shipped; flags documented in src/agents/training/CLAUDE.md |
| `project_exploiter_no_team_advantage.md` | the live rule survives as feedback_no_circular_unwinnable_claims |
| `project_floor_leak_critic_selfko.md` | ai_v5-era critic forensics; era closed, critic replaced by the win-prob head (ai_v12) |
| `project_fresh_generation_equivariance.md` | the ai_v9 gen-1..gen-6 build narrative; the resulting architecture is stated in designs/ARCHITECTURE.md and its ELO rule survives as feedback_elo_reading_rules |
| `project_gen3_randoms_byte_parity.md` | rust randoms grind; src/rust_sim/CLAUDE.md + designs/rust_sim/port_build_log.md are the always-current docs of record |
| `project_gpu_damage_op.md` | Stage A/B build record; the damage op is production and stated in designs/ARCHITECTURE.md |
| `project_hidden_team_belief_built.md` | belief-slot build record; the head is production and stated in designs/ARCHITECTURE.md |
| `project_incoming_damage_outcome.md` | ai_v5-era obs-block forensics; era closed |
| `project_l3_oracle_grind_l4.md` | the durable rule (the amortizability gate, never search ON the model) lives in feedback_research_state |
| `project_latent_belief_built.md` | branch build record, ai_v5 era; superseded by the shipped belief stack |
| `project_local_sim_bridge.md` | the bridge is the default transport; root CLAUDE.md and src/utils/bridge/ are the docs of record |
| `project_loss_analysis_run20260601.md` | June-2026 forensic pass; siblings _run20260531 and _v2 were archived 2026-09-07 |
| `project_markovian_reward_design.md` | PBRS/hand shaping is DELETED under the ai_v12 win-prob critic (--no-hand-shaping) |
| `project_model_frontier_roadmap.md` | ai_v5-era roadmap; superseded by designs/research_state/UNDERSTANDING.md |
| `project_nature_ev_belief_built.md` | v40 build record; the spread belief is production and stated in designs/ARCHITECTURE.md |
| `project_oa_cells_path_forward.md` | SUPERSEDED 2026-08-08 by the OpTensors path (its own banner says so) |
| `project_op_move_order_bugclass.md` | the durable rule survives as feedback_gigo_order_bugs_asap; the guards are in-tree |
| `project_opd_built.md` | ai_v7-era on-policy self-distillation, verdict NEUTRAL; era closed |
| `project_opp_action_head_falsified.md` | ai_v5-era falsification; era closed and the trunk it probed no longer exists |
| `project_opp_hp_immune_bug.md` | GIGO fixed and shipped a4aa2bc; the class rule lives in feedback_gigo_order_bugs_asap |
| `project_opponent_distillation_findings.md` | ai_v5-era opponent distillation; superseded by torch.compile'd opponents and the rust bridge |
| `project_outgoing_damage_design.md` | ai_v5-era obs block; era closed |
| `project_plateau_research_2026_06_25.md` | superseded by designs/research_state/UNDERSTANDING.md §1-§2 (the era map and the flywheel account) |
| `project_popart.md` | PopArt is SHIPPED and is OFF by construction under the ai_v12 win-prob critic; mechanism in designs/learning/popart_value_scale_and_currencies.md |
| `project_public_value_poc.md` | v43 pubval; owner verdict "did nothing" (recorded in project_damage_op_block_audit K9/K10) and the era is closed |
| `project_refresh_status_cure_gap.md` | ai_v5/v6-era obs bits, shipped 92cf1ca/4ce3c72; era closed |
| `project_representation_probe.md` | ai_v5-era probe harness, shipped; src/main/prober/CLAUDE.md is the doc of record |
| `project_resume_optimizer_realign_bug.md` | fixed and hardened (name-keyed remap, ac5b93f); the guard is in-tree |
| `project_reward_shaping_verification.md` | May-2026 method note; superseded by the reward golden (ledger 2026-09-08 gen3_reward_golden_v1) |
| `project_rust_bridge_incremental.md` | rust bridge wedges fixed; src/rust_sim/CLAUDE.md is the always-current doc of record |
| `project_rust_bridge_training_enablement.md` | rust IS the default training transport now (root CLAUDE.md); src/rust_sim/CLAUDE.md owns the detail |
| `project_rust_sim_port.md` | 127 KB build log; src/rust_sim/CLAUDE.md + designs/rust_sim/port_build_log.md are the always-current docs of record |
| `project_spread_belief_supervision.md` | SUPERSEDED by project_nature_ev_belief_built (generative nature/EV head replaced the point estimate) |
| `project_team_pool_weighting_fix.md` | data fix landed; the 719-team pool count is stated in root CLAUDE.md |
| `project_throughput_compile.md` | 33 KB throughput archaeology; the operative facts (--compile-opponents / --compile-trainer ON by default) are in root CLAUDE.md and the training runbook |
| `project_throughput_profile.md` | ai_v5-era py-spy profile; superseded by project_throughput_compile and the rust bridge |
| `project_topk_incoming_moves.md` | v30 build record; the top-K block is production and stated in designs/ARCHITECTURE.md |
| `project_tss_specialist_poc.md` | ai_v7-era single-team specialist arc, double-corrected; era closed |
| `project_unified_move_system.md` | v24/v25 build record; the move system is production and stated in designs/ARCHITECTURE.md |
| `project_value_dist_head_status.md` | the ValueDistHead is DELETED under the ai_v12 win-prob critic |
| `project_value_distill_fitnet.md` | ai_v7-era value-distill A/B; era closed, and no gen fold ever ran v8_14's literal value-feat coef |
| `feedback_delta_interval_before_writing.md` | MERGED into `feedback_equivalence_needs_delta_ci` — same rule, same two catches |
