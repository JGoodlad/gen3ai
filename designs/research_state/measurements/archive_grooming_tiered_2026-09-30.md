# Archive-grooming DRY RUN — `models/`

*policy: **tiered***

*Generated 2026-09-30T11:45:14-0700 · `/home/goodlad/dev/gen3ai/models`*

> **NOTHING WAS DELETED IN THIS PASS.** This is a census; the plan below is what the retention policy *would* do, and it was produced with `--apply` absent.

## Headline

| | |
|---|---|
| runs in the archive | **276** |
| total size | **332.499 GB** |
| … of which physically under `models/` | 332.499 GB |
| … in 0 SYMLINKED run dirs (elsewhere on disk) | 0.0 GB |
| the policy would free | **109.963 GB** (33.1%) |
| entries in the plan | 3818 |
| runs with a non-empty plan | 162 |
| LIVE / REFERENCED / CLOSED | 27 / 248 / 1 |
| runs vetoed by a named file | 68 |
| **CLOSED runs needing review** | **37** |
| tier-4 runs REFUSED (no resolvable final model) | 0 |

## The TIERED policy

> **The owner's reason for tier 4, verbatim (2026-09-06):** *"Yes, please work on a reasonable retention policy, especially pre ai_v8 eras, as we are unlikely to need anything from them as there wasn't a 'novel' outcome, more getting the pattern established and us able to make meaningful progress."*

| tier | who | GB now | GB freed | runs | what happens |
|---:|---|---:|---:|---:|---|
| 0 | LIVE | 40.026 | 0.0 | 27 | live, or reached for by something live — untouched |
| 1 | REFERENCED | 291.845 | 109.66 | 248 | named by the BASELINE REGISTRY, a script, a measurement artifact, the ledger tail, or another run's model graph — standing policy + snapshots rule (a registry-NAMED file is additionally kept at every tier) |
| 2 | v9+ CLOSED | 0.628 | 0.303 | 1 | standing policy + snapshots rule |
| 3 | v8 CLOSED | 0.0 | 0.0 | 0 | first + last + latest.txt pin (no every-10th) + snapshots rule |
| 4 | PRE-v8 | 0.0 | 0.0 | 0 | AGGRESSIVE keep-list — the era's record survives, the weights do not |

### The snapshots rule

A self-play pool is kept **only** when some run forks this run — a fork auto-seeds its parent's pool, so the zips *and* `summary.json` / `win_rate_vs_bots.txt` / `model_config.json` are load-bearing — or a committed script names the run as a `--stable-opponents` / `--exploiter` / pool source. Otherwise `snapshots/` goes whole.

| | |
|---|---:|
| pools KEPT | 42 (18.264 GB) |
| pools FREED | 69 (25.187 GB) |
| runs with no pool | 67 |
| tier-0 runs (rule not applied) | 98 |
| **PROPOSED** further thinning of the KEPT pools (every 4th + newest) | **11.994 GB** |

The thinning is a **proposal, not a plan** — no kept pool loses a byte in this policy. It is reported so the 11.994 GB is a number the owner can decide on rather than a discovery made later.

### Every pool decision, per run

| run | tier | pool GB | decision | why |
|---|---:|---:|---|---|
| `ai_v6_13_outgoing_dmg_0620` | 1 | 0.903 | KEEP | KEPT — a fork parent / pool source: ai_v6_13_outgoing_dmg_0620_exp_v1 (argv fork_parent), ai_v6_13_outgoing_dmg_0620_exp_v1 (fork_parent), ai_v6_13_outgoing_dmg_0620_exploiter_v1 (argv fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v6_13_outgoing_dmg_0620_exp_v1` | 1 | 0.903 | KEEP | KEPT — a committed script names this run: src/main/launcher_test.py |
| `ai_v6_11_typed_hp_0619` | 1 | 0.876 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v7_04_opd_selfdistill_0702` | 1 | 0.868 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v7_02_critic_shape_0627` | 1 | 0.868 | KEEP | KEPT — a fork parent / pool source: ai_v7_05_tss_specialist_0703 (argv pool_source), ai_v7_05_tss_specialist_0703_aborted_noeval (argv pool_source), ai_v7_06_tss_temp_anneal_0706 (argv pool_source) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v7_03_belief_shape_0630` | 1 | 0.868 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v6_09_dmg_reattend_N_0617` | 1 | 0.78 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_27_extremedial_probe_0823` | 1 | 0.711 | KEEP | KEPT — a committed script names this run: src/agents/training/exploiter_ladder.py, src/agents/training/exploiter_ladder_test.py |
| `ai_v9_25_E4_baitbot_0822` | 1 | 0.711 | KEEP | KEPT — a fork parent / pool source: ai_v9_27_extremedial_probe_0823 (argv fork_parent), ai_v9_27_extremedial_probe_0823 (fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v13_16_teach5_offense_dist` | 1 | 0.706 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/population_loop_r1_2026-09-23/launch_RB.sh, designs/research_state/measurements/population_loop_r1_2026-09-23/launch_RC.sh |
| `ai_v13_17_fold_k1` | 1 | 0.706 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v13_18_fold_k3` | 1 | 0.706 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v13_19_fold_k11` | 1 | 0.706 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v13_20_fold_k11_sharematched` | 1 | 0.706 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v13_16_teach5_offense_dist_ABANDONED_forklr2p8` | 1 | 0.706 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v12_04_pfsp_fork25M` | 1 | 0.706 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v6_03_win_pred_N_0614` | 1 | 0.644 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_24_E3_substrate_on_0822` | 1 | 0.64 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_23_E2_substrate_on_0822` | 1 | 0.64 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_22_E1_substrate_on_0821` | 1 | 0.64 | KEEP | KEPT — a fork parent / pool source: ai_v9_25_E4_baitbot_0822 (argv fork_parent), ai_v9_25_E4_baitbot_0822 (fork_parent), ai_v9_26_baitent_probe_0823 (argv fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v6_11_unified_obs_fixed_0618` | 1 | 0.608 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v5_11_tail2_53m_0611` | 1 | 0.587 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_09_gen8_beliefs_threat_inject_0811` | 1 | 0.585 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_172_G1SHORT_0905` | 1 | 0.582 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_76_R4ACTION_0830` | 1 | 0.582 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/resolve_sets.py |
| `ai_v9_91_COMPFOLD_0831` | 1 | 0.582 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/axis_split_taught_untaught.py |
| `ai_v9_71_R3ACTIONHI_0828` | 1 | 0.582 | KEEP | KEPT — a committed script names this run: designs/ai_v12/team_slate_build.py |
| `ai_v9_70_R3ACTION_0828` | 1 | 0.582 | KEEP | KEPT — a committed script names this run: designs/ai_v12/team_slate_build.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py |
| `ai_v5_9_attend_unrevealed_56m_0610` | 1 | 0.571 | KEEP | KEPT — a fork parent / pool source: ai_v5_10_tail1_23_0611 (argv pool_source), ai_v5_11_tail2_53m_0611 (argv pool_source), ai_v5_12_bias_05_N_0612 (argv pool_source) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v5_6_stable_70m_0608` | 1 | 0.566 | KEEP | KEPT — a fork parent / pool source: ai_v5_7_switch_bias_41m_0609 (argv pool_source) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v5_5_popart_50m_0607` | 1 | 0.566 | KEEP | KEPT — a fork parent / pool source: ai_v5_6_stable_70m_0608 (argv pool_source), ai_v5_7_switch_bias_41m_0609 (argv pool_source) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v6_01_belief_53m_0613` | 1 | 0.551 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v5_13_shape_pbrs_43m_0612` | 1 | 0.528 | KEEP | KEPT — a fork parent / pool source: ai_v6_01_belief_53m_0613 (argv pool_source) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v9_12_gen10_t0prior_0814` | 1 | 0.528 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_15_gen13_hb_events_stack_0817` | 1 | 0.517 | KEEP | KEPT — a fork parent / pool source: RETIRED_c5fork_control_gen13base_0817 (argv fork_parent), RETIRED_c5fork_control_gen13base_0817 (fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v9_14_gen12_h_entitypool_shaping_0816` | 1 | 0.514 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/obs_conditioning_probe.py |
| `.dryrun_K6A_1788581936` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_195_G5PLAINA_0906` | 1 | 0.509 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py, src/main/untaught_meter.py |
| `ai_v9_196_G5PLAINB_0906` | 1 | 0.509 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py, src/main/untaught_meter.py |
| `ai_v9_197_G5PLAINC_0906` | 1 | 0.509 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py, src/main/untaught_meter.py |
| `ai_v9_72_R3SELF_0828` | 1 | 0.509 | KEEP | KEPT — a committed script names this run: designs/ai_v12/team_slate_build.py, designs/research_state/measurements/plain_training_robbery.py |
| `ai_v9_60_R2TOPK_0827` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_61_R2KL_0827` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_49_G2_advgate_0826` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_52_G1p_matched_0826` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_38_fdA_coef03_0825` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_39_fdB_lossonly_0825` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `.aborted_R4DOSE12_nometa_1401` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_42_fdE_single_0825` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_62_R2PLAIN_0827` | 1 | 0.509 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/plain_training_robbery.py, designs/research_state/measurements/representational_richness_transfer_forward.py |
| `ai_v9_40_fdC_ecology_0825` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_16_gen14_framedel_v91_0817` | 1 | 0.505 | KEEP | KEPT — a fork parent / pool source: DISCARDED_tdaux_control_n16_0818 (argv fork_parent), DISCARDED_tdaux_control_n16_0818 (fork_parent), ai_v9_17_tdaux_control_0818 (argv fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v9_18_gen15_v8rewards_0818` | 1 | 0.504 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/obs_conditioning_probe.py, src/main/prober/loops.py |
| `ai_v6_04_unified_inc_N_0615` | 1 | 0.501 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_20_tdaux_rung2_lam30_0820` | 1 | 0.498 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_20_tdaux_rung2_lam10_0820` | 1 | 0.498 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_20_tdaux_rung2_lam00_0820` | 1 | 0.498 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_13_gen11_labelonly_winprob_0815` | 1 | 0.496 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v5_3_vf_coef_clip_50m_0606` | 1 | 0.481 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_10_gen9_intent_distcritic_0813` | 1 | 0.475 | KEEP | KEPT — a committed script names this run: src/agents/model/intent_move_cell_test.py |
| `ai_v9_50_fdF_p1c_0826` | 1 | 0.473 | KEEP | KEPT — a fork parent / pool source: ai_v9_51_fdF_p2c_0826 (argv fork_parent), ai_v9_51_fdF_p2c_0826 (fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v9_51_fdF_p2c_0826` | 1 | 0.473 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v5_8_split_inc_dmg_38m_0610` | 1 | 0.457 | KEEP | KEPT — a fork parent / pool source: ai_v5_10_tail1_23_0611 (argv pool_source), ai_v5_11_tail2_53m_0611 (argv pool_source), ai_v5_12_bias_05_N_0612 (argv pool_source) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v5_7_switch_bias_41m_0609` | 1 | 0.453 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v7_01_teacher_0626` | 1 | 0.434 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_19_gen16_mechanics_0819` | 1 | 0.427 | KEEP | KEPT — a fork parent / pool source: ai_v9_20_tdaux_rung2_lam00_0820 (argv fork_parent), ai_v9_20_tdaux_rung2_lam00_0820 (fork_parent), ai_v9_20_tdaux_rung2_lam10_0820 (argv fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v9_21_gen17_pfspoff_0820` | 1 | 0.427 | KEEP | KEPT — a fork parent / pool source: ai_v9_22_E1_substrate_on_0821 (argv fork_parent), ai_v9_22_E1_substrate_on_0821 (fork_parent), ai_v9_23_E2_substrate_on_0822 (argv fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v8_03_zarch_control_0718` | 1 | 0.416 | KEEP | KEPT — a fork parent / pool source: ai_v8_04_distill_4teacher_0722 (argv fork_parent), ai_v8_04_distill_4teacher_0722 (fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v6_07_unified_topk_N_0616` | 1 | 0.38 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v6_08_unmasked_floor_N_0617` | 1 | 0.304 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v5_4_pbrs_opp_threat_50m_0607` | 1 | 0.283 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v12_01_winprob_critic` | 1 | 0.268 | KEEP | KEPT — a committed script names this run: src/agents/model/model_version/fields.py, src/agents/training/instrumented_ppo/calibration.py |
| `ai_v5_12_bias_05_N_0612` | 1 | 0.235 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v5_10_tail1_23_0611` | 1 | 0.205 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_34_tick1_0824` | 1 | 0.182 | KEEP | KEPT — a fork parent / pool source: ai_v9_35_tick1_exploit_0824 (argv fork_parent), ai_v9_35_tick1_exploit_0824 (argv pool_source), ai_v9_35_tick1_exploit_0824 (fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v6_06_unified_all_N_0616` | 1 | 0.174 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v6_02_belief_lat_16m_0614` | 1 | 0.125 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v8_15_retention_A_frozen_0726` | 1 | 0.089 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v8_02_zarch_teampfsp_0718` | 1 | 0.089 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v7_19_combined_0716` | 1 | 0.087 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v7_14_league_capstone_0712` | 1 | 0.087 | KEEP | KEPT — a fork parent / pool source: ai_v7_15_tss_exploiter_vs14_0713 (argv pool_source), ai_v7_16_distill_tss_mvp_0715 (argv fork_parent), ai_v7_16_distill_tss_mvp_0715 (fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v5_2_native_selfplay_50m_0606` | 1 | 0.085 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_17_tdaux_lam1_0818` | 1 | 0.084 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_17_tdaux_lam3_0818` | 1 | 0.084 | KEEP | KEPT — a committed script names this run: src/agents/training/poke_env_gaps/faint_attribution_fuzz_test.py |
| `ai_v9_82_REFOLD1_0830` | 1 | 0.073 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/axis_split_taught_untaught.py |
| `ai_v9_37_tick1_dosext_0825` | 1 | 0.073 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_58_R2CTRL_0827` | 1 | 0.073 | KEEP | KEPT — a committed script names this run: designs/ai_v12/team_slate_build.py, designs/research_state/measurements/plain_training_robbery.py |
| `v8rep_p1_A_0905` | 1 | 0.045 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `v8rep_p1_C_0905` | 1 | 0.045 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `v8rep_p1_B_0905` | 1 | 0.045 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `v8rep_p2loss_B_0905` | 1 | 0.045 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `v8rep_p2loss_A_0905` | 1 | 0.045 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `v8rep_p2loss_C_0905` | 1 | 0.045 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `v8rep_p2self_B_0905` | 1 | 0.045 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py |
| `v8rep_p2self_C_0905` | 1 | 0.045 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py |
| `v8rep_p2self_A_0905` | 1 | 0.045 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py |
| `ai_v7_21_fitnet_valuefeat_ab_0717` | 1 | 0.043 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v7_20_valuedistill_ab_0717` | 1 | 0.043 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v7_18_distill_4teacher_0716` | 1 | 0.043 | KEEP | KEPT — a fork parent / pool source: ai_v7_19_combined_0716 (argv fork_parent), ai_v7_19_combined_0716 (fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v7_16_distill_tss_mvp_0715` | 1 | 0.043 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_17_tdaux_control_0818` | 1 | 0.042 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `DISCARDED_tdaux_control_n16_0818` | 1 | 0.041 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_11_gen10_intentfull_compiled_0814` | 1 | 0.041 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v12_26_ladder_ctrl10M_shaped` | 1 | 0.039 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py |
| `ai_v6_04_unified_all_half_batch_N_0616` | 1 | 0.035 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v8_01_zarch_film_0717` | 1 | 0.017 | KEEP | KEPT — a fork parent / pool source: ai_v8_02_zarch_teampfsp_0718 (argv fork_parent), ai_v8_02_zarch_teampfsp_0718 (fork_parent), ai_v8_03_zarch_control_0718 (argv fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v12_27_ladder_ctrl10M_shaped_dense` | 2 | 0.0 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v6_10_unified_obs_0618` | 1 | 0.0 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `.aborted_R4DOSE12_poolless_1355` | 1 | 0.0 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v12_02_winprob_critic.OOM_4096` | 1 | 0.0 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v7_01_teacher_0626_oom1` | 1 | 0.0 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |

### Tier 4 — what each pre-v8 run keeps

The keep-list is the policy. `resolve_model_ref` picks the ONE model file, and the **rung** it fired on is recorded, because a bare run dir has meant different files at different times (`gen3_last_snapshot_resolution_v1`).

| run | GB freed | the kept model | rung | steps |
|---|---:|---|---|---:|
| — | | *(no run graded tier 4)* | | |

**Consequence, stated plainly:** a tier-4 run becomes un-probeable except at its final checkpoint. That costs less than it sounds: root `CLAUDE.md` records that on 2026-08-13 **79 of 79 archived runs could not be re-loaded** at the then-current architecture, and the drift has only grown since — so every model-loading prober view (`analyze` / `lookahead` / `better-line` / `replay-counterfactual` / `probe`) already returns an `ArchDriftError` on these runs. What survives is exactly what still works on them: `tb/`, the ELO ladder, `eval_results.jsonl`, and the model-free prober views.

## The policy

Applied per TIER (above). Tiers 1-3 stay inside `checkpoints`, `eval_traces` plus `snapshots/`; tier 4 works from a KEEP-LIST instead and is guarded by `assert_safe_tiered`. The rules below describe the tier-1/2 body of the policy, which is the standing one verbatim (tier 3 differs only in taking no every-10th stride).

- **`checkpoints/`** — keep the FIRST, the LAST, every 10th, whatever `latest.txt` pins, and any checkpoint another run's `lineage` block resolved to. A `.json` sidecar is kept or dropped with its `.zip`, by STEP.
- **`eval_traces/`** — `main.prober.groom` at 3/1. The groomer's own planner is called, not re-implemented, so the two can never drift.
- **Never touched**: `best_model`, `cf_labels`, `cf_records`, `crashes`, `elo`, `snapshot_ladder`, `snapshots`, `stalls`, `tb`, `tb_imgs`, and the run-root files `capacity_battery.json`, `command.txt`, `eval_results.jsonl`, `latest.txt`, `launcher_child.log`, `metadata.json`, `model_config.json`, `team_winrates.json`, `team_winrates_history.jsonl`. `_assert_safe` re-checks every planned path against these before the plan is reported or executed.
- A run is **tier 0** if a launcher process names it, its training output was written within 7 days, its run dir is a symlink, or it is a (transitive) model-graph ancestor of any of those. It is **tier 1** if the ledger's last 1500 lines name it, a committed **script** names it, a committed **measurement artifact** names it, or another run's model graph names it. The v8-era blanket is RETIRED — the model graph replaces it, and reads `original_command` as well as `lineage`.
- Prose that merely *mentions* a run does **not** protect it — the historical record names nearly every run forever, so a `.md` mention as a live reference would close nothing. A committed script does protect it (a script names a run dir in order to load it), and prose still **vetoes** when it names an exact path the plan would delete.
- `snapshots/` (the self-play pool) HAS a rule here — see *The snapshots rule* above. It is the second-largest consumer in the archive and the standing policy leaves it entirely alone.

## Top 20 runs by GB freed

| # | run | generation | GB freed | ckpts deleted | trace steps deleted |
|---|---|---|---:|---:|---:|
| 1 | `ai_v8_03_zarch_control_0718` | ai_v8 | 5.789 | 196 | 19 |
| 2 | `ai_v9_75_R4S3c_0829` | ai_v9 | 2.324 | 116 | 4 |
| 3 | `ai_v9_74_R4S3b_0829` | ai_v9 | 2.313 | 116 | 4 |
| 4 | `ai_v9_34_tick1_0824` | ai_v9 | 2.308 | 116 | 4 |
| 5 | `ai_v9_73_R4S3a_0829` | ai_v9 | 2.274 | 114 | 4 |
| 6 | `ai_v13_19_fold_k11` | ai_v13 | 1.676 | 38 | 5 |
| 7 | `ai_v13_17_fold_k1` | ai_v13 | 1.674 | 38 | 5 |
| 8 | `ai_v13_18_fold_k3` | ai_v13 | 1.671 | 38 | 5 |
| 9 | `ai_v13_20_fold_k11_sharematched` | ai_v13 | 1.642 | 36 | 5 |
| 10 | `ai_v9_09_gen8_beliefs_threat_inject_0811` | ai_v9 | 1.629 | 10 | 12 |
| 11 | `ai_v9_12_gen10_t0prior_0814` | ai_v9 | 1.541 | 12 | 12 |
| 12 | `ai_v8_01_zarch_film_0717` | ai_v8 | 1.507 | 36 | 11 |
| 13 | `ai_v9_13_gen11_labelonly_winprob_0815` | ai_v9 | 1.49 | 12 | 11 |
| 14 | `ai_v8_12_defensive20_exploiter_0724` | ai_v8 | 1.458 | 36 | 11 |
| 15 | `ai_v7_04_opd_selfdistill_0702` | ai_v7 | 1.346 | 22 | 0 |
| 16 | `ai_v8_07_semistall564_scratch_0722` | ai_v8 | 1.167 | 32 | 10 |
| 17 | `ai_v9_37_tick1_dosext_0825` | ai_v9 | 1.165 | 56 | 2 |
| 18 | `ai_v9_60_R2TOPK_0827` | ai_v9 | 1.165 | 34 | 1 |
| 19 | `ai_v9_61_R2KL_0827` | ai_v9 | 1.165 | 34 | 1 |
| 20 | `ai_v9_52_G1p_matched_0826` | ai_v9 | 1.165 | 34 | 1 |

## Runs vetoed because a committed file or the ledger names a file in the plan

These are excluded from the deletion set automatically.

| run | GB it would have freed | example named path | named by |
|---|---:|---|---|
| `ai_v12_02_winprob_critic` | 1.672 | `checkpoints/checkpoint_12369408_steps.json` | designs/ARCHITECTURE.md, designs/CHANGELOG.md |
| `ai_v12_10_ladder_vf15` | 0.227 | `eval_traces/step_6000000/snapshot.zip` | designs/research_state/ledger.md, designs/research_state/ledger_index.md |
| `ai_v12_11_ladder_ctrl10M` | 0.266 | `eval_traces/step_4000032` | designs/CHANGELOG.md, designs/research_state/UNDERSTANDING.md |
| `ai_v12_12_ladder_cflabels` | 0.888 | `eval_traces/step_6000000/snapshot.zip` | designs/CHANGELOG.md, designs/research_state/ledger.md |
| `ai_v12_13_ladder_tdaux` | 0.239 | `eval_traces/step_6000000/snapshot.zip` | designs/research_state/ledger.md, designs/research_state/ledger_index.md |
| `ai_v12_14_ladder_truevalue` | 0.349 | `eval_traces/step_6000000/snapshot.zip` | designs/CHANGELOG.md, designs/model/file_layout.md |
| `ai_v12_15_ladder_ctrl10M_b` | 0.385 | `eval_traces/step_4000032` | designs/research_state/UNDERSTANDING.md, designs/research_state/ledger.md |
| `ai_v12_16_ladder_ctrl10M_c` | 0.352 | `eval_traces/step_4000032` | designs/research_state/ledger.md, designs/research_state/ledger_index.md |
| `ai_v12_17_ladder_strata` | 0.353 | `eval_traces/step_6000000/snapshot.zip` | designs/research_state/UNDERSTANDING.md, designs/research_state/ledger.md |
| `ai_v12_19_ladder_lambda09` | 0.393 | `eval_traces/step_4000032` | designs/CHANGELOG.md, designs/research_state/ledger.md |
| `ai_v12_20_ladder_denseaux` | 0.494 | `eval_traces/step_6000000/snapshot.zip` | designs/research_state/ledger.md, designs/research_state/ledger_index.md |
| `ai_v12_21_ladder_lambda09_b` | 0.346 | `eval_traces/step_6000000/snapshot.zip` | designs/research_state/ledger.md, designs/research_state/ledger_index.md |
| `ai_v12_22_ladder_lambda095` | 0.36 | `eval_traces/step_6000000/snapshot.zip` | designs/research_state/ledger.md, designs/research_state/ledger_index.md |
| `ai_v12_23_ladder_rollout` | 0.338 | `eval_traces/step_6000000/snapshot.zip` | designs/research_state/ledger.md, designs/research_state/ledger_index.md |
| `ai_v12_24_ladder_strata_b` | 0.314 | `eval_traces/step_6000000/snapshot.zip` | designs/research_state/UNDERSTANDING.md, designs/research_state/ledger.md |
| `ai_v12_25_ladder_vf15_b` | 0.246 | `eval_traces/step_6000000/snapshot.zip` | designs/research_state/flywheel_era_pair_2026-09-12.md, designs/research_state/ledger.md |
| `ai_v12_28_ladder_ent05` | 0.24 | `eval_traces/step_6000000/snapshot.zip` | designs/research_state/UNDERSTANDING.md, designs/research_state/flywheel_era_pair_2026-09-12.md |
| `ai_v12_29_ladder_vf025` | 0.245 | `eval_traces/step_6000000/snapshot.zip` | designs/research_state/flywheel_era_pair_2026-09-12.md, designs/research_state/ledger.md |
| `ai_v13_01_flywheel_shaped` | 1.781 | `eval_traces/step_70000032/snapshot.zip` | designs/research_state/UNDERSTANDING.md, designs/research_state/flywheel_era_pair_2026-09-12.md |
| `ai_v13_03_fork` | 0.254 | `eval_traces/step_6000000/snapshot.zip` | designs/ops/EXTERNAL_ANCHORS_SOP.md, designs/research_state/UNDERSTANDING.md |
| `ai_v13_04_flywheel_winprob_b` | 1.799 | `eval_traces/step_70000032/snapshot.zip` | designs/research_state/ledger.md, designs/research_state/ledger_index.md |
| `ai_v13_05_exploit_big5starmie` | 0.164 | `eval_traces/step_78000000/snapshot.zip` | designs/ops/TECH_DEBT_BACKLOG.md, designs/research_state/ledger.md |
| `ai_v13_07_fold1` | 0.388 | `checkpoints/checkpoint_76005984_steps.zip` | designs/research_state/UNDERSTANDING.md, designs/research_state/ledger.md |
| `ai_v13_08_fold1_cont` | 0.388 | `checkpoints/checkpoint_82600848_steps.zip` | designs/research_state/ledger.md, designs/research_state/ledger_index.md |
| `ai_v13_11_split_lossoff` | 1.024 | `checkpoints/checkpoint_78006048_steps.zip` | designs/research_state/ledger.md, designs/research_state/ledger_index.md |
| `ai_v8_04_distill_4teacher_0722` | 0.424 | `eval_traces/step_272000006/snapshot.zip` | designs/baselines.json, designs/research_state/UNDERSTANDING.md |
| `ai_v8_06_semistall_3team_exploiter_0722` | 0.381 | `checkpoints/checkpoint_279699602_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/v8_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/README.md |
| `ai_v8_09_pool10_exploiter_0723` | 1.115 | `checkpoints/checkpoint_279688455_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/v8_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/README.md |
| `ai_v8_13_defensive10_exploiter_0725` | 0.772 | `checkpoints/checkpoint_279671587_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/v8_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/README.md |
| `ai_v8_14_distill3_0725` | 1.03 | `checkpoints/checkpoint_279661705_steps.json` | designs/baselines.json, designs/research_state/flywheel_era_pair_2026-09-12.md |
| `ai_v9_100_R5F08_0831` | 0.656 | `checkpoints/checkpoint_25367760_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py |
| `ai_v9_102_R5F10_0831` | 0.656 | `checkpoints/checkpoint_25367760_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py |
| `ai_v9_104_R5F12_0831` | 0.656 | `checkpoints/checkpoint_25367760_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py |
| `ai_v9_106_R5F14_0831` | 0.656 | `checkpoints/checkpoint_25367760_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/README.md |
| `ai_v9_120_R5FUND00_0901` | 0.364 | `checkpoints/checkpoint_29015184_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py |
| `ai_v9_122_R5FUND02_0901` | 0.364 | `checkpoints/checkpoint_29015184_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py |
| `ai_v9_124_R5FUND04_0901` | 0.364 | `checkpoints/checkpoint_29015184_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py |
| `ai_v9_126_R5FUND06_0901` | 0.364 | `checkpoints/checkpoint_29015184_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py |
| `ai_v9_128_R5FUND08_0901` | 0.364 | `checkpoints/checkpoint_29015184_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py |
| `ai_v9_130_R5FUND10_0901` | 0.364 | `checkpoints/checkpoint_29015184_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py |
| `ai_v9_132_R5FUND12_0901` | 0.364 | `checkpoints/checkpoint_29015184_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py |
| `ai_v9_134_R5FUND14_0901` | 0.364 | `checkpoints/checkpoint_29015184_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py |
| `ai_v9_140_B2_0901` | 0.947 | `checkpoints/checkpoint_29165184_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/ledger.md |
| `ai_v9_141_C1_0901` | 0.983 | `checkpoints/checkpoint_29165184_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/ledger.md |
| `ai_v9_142_N1_0901` | 0.947 | `checkpoints/checkpoint_29165184_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json |
| `ai_v9_143_N2_0901` | 0.947 | `checkpoints/checkpoint_29165184_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json |
| `ai_v9_150_R4DOSE12_0901` | 0.255 | `checkpoints/checkpoint_29115216_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/README.md |
| `ai_v9_151_R4DOSE6_0901` | 0.255 | `checkpoints/checkpoint_29115216_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/README.md |
| `ai_v9_152_R4DOSE3_0901` | 0.255 | `checkpoints/checkpoint_29115216_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/README.md |
| `ai_v9_160_TCFUNDA_0903` | 0.255 | `checkpoints/checkpoint_29115216_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_competence/README.md |
| `ai_v9_161_TCFUNDB_0903` | 0.255 | `checkpoints/checkpoint_29115216_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py |
| `ai_v9_162_TCUNFA_0903` | 0.255 | `checkpoints/checkpoint_29115216_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/ledger.md |
| `ai_v9_163_TCUNFB_0903` | 0.255 | `checkpoints/checkpoint_29115216_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py |
| `ai_v9_170_TCUNFK6A_0904` | 0.255 | `checkpoints/checkpoint_29115216_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/ledger.md |
| `ai_v9_171_TCUNFK6B_0904` | 0.255 | `checkpoints/checkpoint_29115216_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/ledger.md |
| `ai_v9_29_rev1_0823` | 4.751 | `checkpoints/checkpoint_9995088_steps.zip` | designs/ai_v12/design_winprob_behavior_coupling.md, designs/ai_v12/team_slate_40.md |
| `ai_v9_31_tock1_k4_0824` | 0.656 | `eval_traces/step_26000016/snapshot.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/ai_v9_34_tick1_0824_endofrun.json |
| `ai_v9_44_tock2_v8shape_0825` | 2.077 | `eval_traces/step_30000000/snapshot.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/archive_grooming_dryrun_2026-09-06.json |
| `ai_v9_53_R2F5a_0826` | 0.656 | `checkpoints/checkpoint_27467760_steps.zip` | designs/ai_v12/promotion_exclusions.json, designs/research_state/era_boundary_deprecation_2026-09-06.md |
| `ai_v9_54_R2F5b_0826` | 0.656 | `checkpoints/checkpoint_27467760_steps.zip` | designs/ai_v12/promotion_exclusions.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_rev2.json |
| `ai_v9_55_R2F5c_0826` | 0.656 | `eval_traces/step_26000016/snapshot.zip` | designs/ai_v12/promotion_exclusions.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_rev2.json |
| `ai_v9_56_R2F5d_0826` | 0.656 | `eval_traces/step_26000016/snapshot.zip` | designs/ai_v12/promotion_exclusions.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_rev2.json |
| `ai_v9_57_R2F5e_0826` | 0.656 | `eval_traces/step_26000016/snapshot.zip` | designs/ai_v12/promotion_exclusions.json, designs/research_state/era_boundary_deprecation_2026-09-06.md |
| `ai_v9_59_R2ACTION_0827` | 0.656 | `checkpoints/checkpoint_25367760_steps.zip` | designs/ai_v12/design_winprob_only_critic.md, designs/ai_v12/team_slate_build.py |
| `ai_v9_92_R5F00_0831` | 0.656 | `checkpoints/checkpoint_25367760_steps.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py |
| `ai_v9_94_R5F02_0831` | 0.656 | `checkpoints/checkpoint_25367760_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/README.md |
| `ai_v9_96_R5F04_0831` | 0.656 | `checkpoints/checkpoint_25367760_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py |
| `ai_v9_98_R5F06_0831` | 0.656 | `checkpoints/checkpoint_25367760_steps.zip` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py |

## ⚠️ SYMLINKED run dirs — the data is NOT under `models/`

These entries are symlinks into launcher worktrees, so `du -sh models/` does not see them and a deletion "in `models/`" would physically land under `.claude/worktrees/`. They are held out of the plan by default; `--follow-symlinked-runs` opts in after you have confirmed the targets are still the ones you mean.

*(none)*

## ⚠️ REVIEW BEFORE APPLYING — CLOSED runs the ledger names outside its tail

The tail window is what protects a run; this section makes its EDGE visible rather than silent. Each of these has a non-empty plan **and** is named somewhere higher up `ledger.md`, so a banked result may still rest on it. They are still in the deletion set — read them before running `--apply`, and widen `--ledger-tail-lines` (or delete the run's entry from the plan) if any should be kept.

| run | generation | GB freed | named at | prose mentions |
|---|---|---:|---|---:|
| `ai_v12_01_winprob_critic` | ai_v12 | 0.522 | `ledger.md:11534` | 28 |
| `ai_v12_02_winprob_critic.OOM_4096` | ai_v12 | 0.0 | `ledger.md:11844` | 3 |
| `ai_v12_04_pfsp_fork25M` | ai_v12 | 0.776 | `ledger.md:15208` | 7 |
| `ai_v12_26_ladder_ctrl10M_shaped` | ai_v12 | 0.485 | `ledger.md:17918` | 6 |
| `ai_v12_27_ladder_ctrl10M_shaped_dense` | ai_v12 | 0.303 | `ledger.md:18008` | 3 |
| `ai_v13_06_exploit_ddtar_spikes` | ai_v13 | 0.127 | `ledger.md:19319` | 18 |
| `ai_v13_10_exploit_stall` | ai_v13 | 0.136 | `ledger.md:20184` | 6 |
| `ai_v13_14_exploit5_balance` | ai_v13 | 0.142 | `ledger.md:20489` | 6 |
| `ai_v13_15_exploit5_stall` | ai_v13 | 0.143 | `ledger.md:20503` | 6 |
| `ai_v13_16_teach5_offense_dist` | ai_v13 | 0.14 | `ledger.md:20547` | 19 |
| `ai_v13_16_teach5_offense_dist_ABANDONED_forklr2p8` | ai_v13 | 0.706 | `ledger.md:20557` | 1 |
| `ai_v13_17_fold_k1` | ai_v13 | 1.674 | `ledger.md:20583` | 8 |
| `ai_v13_18_fold_k3` | ai_v13 | 1.671 | `ledger.md:20658` | 2 |
| `ai_v13_19_fold_k11` | ai_v13 | 1.676 | `ledger.md:20692` | 1 |
| `ai_v13_20_fold_k11_sharematched` | ai_v13 | 1.642 | `ledger.md:21163` | 1 |
| `ai_v5_11_tail2_53m_0611` | ai_v5 | 0.763 | `ledger.md:85` | 11 |
| `ai_v5_12_bias_05_N_0612` | ai_v5 | 0.294 | `ledger.md:87` | 10 |
| `ai_v5_5_popart_50m_0607` | ai_v5 | 0.198 | `ledger.md:13470` | 13 |
| `ai_v6_13_outgoing_dmg_0620` | ai_v6 | 0.406 | `ledger.md:21505` | 15 |
| `ai_v6_13_outgoing_dmg_0620_exp_v1` | ai_v6 | 0.045 | `ledger.md:21501` | 11 |
| `ai_v7_14_league_capstone_0712` | ai_v7 | 0.13 | `ledger.md:14285` | 10 |
| `ai_v8_01_zarch_film_0717` | ai_v8 | 1.507 | `ledger.md:14283` | 19 |
| `ai_v8_03_zarch_control_0718` | ai_v8 | 5.789 | `ledger.md:18141` | 45 |
| `ai_v8_15_retention_A_frozen_0726` | ai_v8 | 1.135 | `ledger.md:13470` | 11 |
| `ai_v9_107_R5F15_0831` | ai_v9 | 0.656 | `ledger.md:8497` | 12 |
| `ai_v9_172_G1SHORT_0905` | ai_v9 | 0.837 | `ledger.md:10935` | 16 |
| `ai_v9_21_gen17_pfspoff_0820` | ai_v9 | 0.845 | `ledger.md:2265` | 47 |
| `ai_v9_25_E4_baitbot_0822` | ai_v9 | 0.323 | `ledger.md:2039` | 25 |
| `ai_v9_27_extremedial_probe_0823` | ai_v9 | 0.036 | `ledger.md:3722` | 18 |
| `ai_v9_34_tick1_0824` | ai_v9 | 2.308 | `ledger.md:11017` | 28 |
| `ai_v9_37_tick1_dosext_0825` | ai_v9 | 1.165 | `ledger.md:11018` | 22 |
| `ai_v9_38_fdA_coef03_0825` | ai_v9 | 1.165 | `ledger.md:11018` | 19 |
| `ai_v9_50_fdF_p1c_0826` | ai_v9 | 0.291 | `ledger.md:17963` | 19 |
| `ai_v9_51_fdF_p2c_0826` | ai_v9 | 0.728 | `ledger.md:21051` | 18 |
| `ai_v9_58_R2CTRL_0827` | ai_v9 | 0.656 | `ledger.md:11147` | 38 |
| `ai_v9_62_R2PLAIN_0827` | ai_v9 | 0.656 | `ledger.md:8151` | 35 |
| `v8rep_p1_A_0905` | ai_v8 (replication) | 0.045 | `ledger.md:10541` | 6 |

## Per-run census

Sizes in GB. `plan GB` is 0 for every run that is not CLOSED.

| run | gen | cfg | tier | status | ckpts | best | snaps | traces | tb | other | total | plan GB |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ai_v8_03_zarch_control_0718` | ai_v8 | 45 | 1 | REFERENCED | 4.938 | 0.045 | 0.416 | 1.631 | 0.011 | 0.412 | 7.459 | 5.789 |
| `ai_v9_29_rev1_0823` | ai_v9 | 101 | 1 | REFERENCED | 4.625 | 0.036 | 0.436 | 0.793 | 0.005 | 0.075 | 6.092 | 0.0 |
| `ai_v13_33_core_burnin` | ai_v13 | 120 | 0 | LIVE | 2.258 | 0.035 | 0.706 | 1.394 | 0.031 | 0.206 | 4.636 | 0.0 |
| `ai_v5_6_stable_70m_0608` | ai_v5 | 7 | 1 | REFERENCED | 0.311 | 0.028 | 0.566 | 0.075 | 0.001 | 2.691 | 3.987 | 0.255 |
| `ai_v13_02_flywheel_winprob` | ai_v13 | 119 | 0 | LIVE | 0.882 | 0.035 | 0.706 | 1.342 | 0.016 | 0.451 | 3.441 | 0.0 |
| `ai_v13_04_flywheel_winprob_b` | ai_v13 | 119 | 1 | REFERENCED | 0.847 | 0.035 | 0.706 | 1.27 | 0.016 | 0.451 | 3.333 | 0.0 |
| `ai_v14_01_base` | ai_v14 | 121 | 0 | LIVE | 0.851 | 0.035 | 0.709 | 1.257 | 0.016 | 0.451 | 3.329 | 0.0 |
| `ai_v9_34_tick1_0824` | ai_v9 | 101 | 1 | REFERENCED | 2.404 | 0.036 | 0.182 | 0.331 | 0.007 | 0.075 | 3.165 | 2.308 |
| `ai_v5_9_attend_unrevealed_56m_0610` | ai_v5 | 8 | 1 | REFERENCED | 0.257 | 0.029 | 0.571 | 0.1 | 0.001 | 2.115 | 3.08 | 0.2 |
| `ai_v6_01_belief_53m_0613` | ai_v6 | 16 | 1 | REFERENCED | 0.245 | 0.031 | 0.551 | 0.117 | 0.002 | 2.087 | 3.039 | 0.735 |
| `ai_v5_5_popart_50m_0607` | ai_v5 | 6 | 1 | REFERENCED | 0.255 | 0.028 | 0.566 | 0.071 | 0.001 | 2.068 | 3.001 | 0.198 |
| `ai_v5_11_tail2_53m_0611` | ai_v5 | 11 | 1 | REFERENCED | 0.235 | 0.029 | 0.587 | 0.117 | 0.001 | 2.027 | 3.001 | 0.763 |
| `ai_v13_01_flywheel_shaped` | ai_v13 | 119 | 1 | REFERENCED | 0.854 | 0.036 | 0.711 | 1.233 | 0.018 | 0.091 | 2.951 | 0.0 |
| `ai_v9_75_R4S3c_0829` | ai_v9 | 107 | 1 | REFERENCED | 2.404 | 0.036 | 0.0 | 0.342 | 0.007 | 0.074 | 2.896 | 2.324 |
| `ai_v9_74_R4S3b_0829` | ai_v9 | 107 | 1 | REFERENCED | 2.404 | 0.036 | 0.0 | 0.322 | 0.007 | 0.074 | 2.875 | 2.313 |
| `ai_v12_02_winprob_critic` | ai_v12 | 110 | 1 | REFERENCED | 0.882 | 0.035 | 0.706 | 1.118 | 0.016 | 0.072 | 2.836 | 0.0 |
| `ai_v9_73_R4S3a_0829` | ai_v9 | 107 | 1 | REFERENCED | 2.367 | 0.036 | 0.0 | 0.304 | 0.007 | 0.074 | 2.819 | 2.274 |
| `ai_v9_44_tock2_v8shape_0825` | ai_v9 | 101 | 1 | REFERENCED | 2.149 | 0.036 | 0.0 | 0.285 | 0.006 | 0.074 | 2.573 | 0.0 |
| `ai_v5_3_vf_coef_clip_50m_0606` | ai_v5 | 3 | 1 | REFERENCED | 0.198 | 0.028 | 0.481 | 0.067 | 0.001 | 1.755 | 2.534 | 0.623 |
| `ai_v5_13_shape_pbrs_43m_0612` | ai_v5 | 15 | 1 | REFERENCED | 0.206 | 0.029 | 0.528 | 0.112 | 0.001 | 1.645 | 2.532 | 0.147 |
| `ai_v5_7_switch_bias_41m_0609` | ai_v5 | 7 | 1 | REFERENCED | 0.198 | 0.028 | 0.453 | 0.099 | 0.001 | 1.502 | 2.291 | 0.595 |
| `ai_v13_11_split_lossoff` | ai_v13 | 119 | 1 | REFERENCED | 0.812 | 0.035 | 0.706 | 0.563 | 0.019 | 0.133 | 2.275 | 0.0 |
| `ai_v5_8_split_inc_dmg_38m_0610` | ai_v5 | 7 | 1 | REFERENCED | 0.171 | 0.029 | 0.457 | 0.089 | 0.001 | 1.431 | 2.19 | 0.143 |
| `ai_v13_17_fold_k1` | ai_v13 | 119 | 1 | REFERENCED | 0.812 | 0.035 | 0.706 | 0.46 | 0.024 | 0.142 | 2.185 | 1.674 |
| `ai_v13_19_fold_k11` | ai_v13 | 119 | 1 | REFERENCED | 0.812 | 0.035 | 0.706 | 0.455 | 0.024 | 0.142 | 2.18 | 1.676 |
| `ai_v13_18_fold_k3` | ai_v13 | 119 | 1 | REFERENCED | 0.812 | 0.035 | 0.706 | 0.453 | 0.024 | 0.142 | 2.177 | 1.671 |
| `ai_v13_21_wcont_b` | ai_v13 | 119 | 0 | LIVE | 0.776 | 0.035 | 0.706 | 0.497 | 0.019 | 0.133 | 2.174 | 0.0 |
| `ai_v13_09_wcont` | ai_v13 | 119 | 0 | LIVE | 0.776 | 0.035 | 0.706 | 0.488 | 0.019 | 0.132 | 2.165 | 0.0 |
| `ai_v13_20_fold_k11_sharematched` | ai_v13 | 119 | 1 | REFERENCED | 0.776 | 0.035 | 0.706 | 0.461 | 0.024 | 0.143 | 2.151 | 1.642 |
| `ai_v9_14_gen12_h_entitypool_shaping_0816` | ai_v9 | 80 | 1 | REFERENCED | 0.385 | 0.043 | 0.514 | 1.014 | 0.003 | 0.086 | 2.053 | 1.11 |
| `ai_v9_09_gen8_beliefs_threat_inject_0811` | ai_v9 | 64 | 1 | REFERENCED | 0.315 | 0.045 | 0.585 | 1.035 | 0.002 | 0.046 | 2.043 | 1.629 |
| `ai_v9_16_gen14_framedel_v91_0817` | ai_v9 | 91 | 1 | REFERENCED | 0.379 | 0.042 | 0.505 | 1.016 | 0.004 | 0.085 | 2.039 | 1.087 |
| `ai_v7_04_opd_selfdistill_0702` | ai_v7 | 42 | 1 | REFERENCED | 0.608 | 0.043 | 0.868 | 0.227 | 0.01 | 0.219 | 1.983 | 1.346 |
| `ai_v9_70_R3ACTION_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.093 | 0.036 | 0.582 | 0.145 | 0.007 | 0.074 | 1.963 | 0.983 |
| `ai_v9_76_R4ACTION_0830` | ai_v9 | 107 | 1 | REFERENCED | 1.056 | 0.036 | 0.582 | 0.18 | 0.007 | 0.074 | 1.962 | 0.947 |
| `ai_v9_140_B2_0901` | ai_v9 | 107 | 1 | REFERENCED | 1.056 | 0.036 | 0.582 | 0.172 | 0.002 | 0.074 | 1.949 | 0.0 |
| `ai_v9_141_C1_0901` | ai_v9 | 107 | 1 | REFERENCED | 1.093 | 0.036 | 0.582 | 0.162 | 0.002 | 0.038 | 1.939 | 0.0 |
| `ai_v9_143_N2_0901` | ai_v9 | 107 | 1 | REFERENCED | 1.056 | 0.036 | 0.546 | 0.191 | 0.002 | 0.074 | 1.932 | 0.0 |
| `ai_v9_91_COMPFOLD_0831` | ai_v9 | 107 | 1 | REFERENCED | 1.056 | 0.036 | 0.582 | 0.148 | 0.007 | 0.074 | 1.93 | 0.947 |
| `ai_v9_71_R3ACTIONHI_0828` | ai_v9 | 104 | 1 | REFERENCED | 1.056 | 0.036 | 0.582 | 0.145 | 0.007 | 0.074 | 1.927 | 0.947 |
| `ai_v8_01_zarch_film_0717` | ai_v8 | 44 | 1 | REFERENCED | 0.938 | 0.045 | 0.017 | 0.843 | 0.002 | 0.063 | 1.914 | 1.507 |
| `ai_v9_15_gen13_hb_events_stack_0817` | ai_v9 | 89 | 1 | REFERENCED | 0.258 | 0.043 | 0.517 | 0.989 | 0.004 | 0.087 | 1.906 | 0.947 |
| `ai_v9_72_R3SELF_0828` | ai_v9 | 107 | 1 | REFERENCED | 1.093 | 0.036 | 0.509 | 0.155 | 0.007 | 0.074 | 1.899 | 0.983 |
| `ai_v9_12_gen10_t0prior_0814` | ai_v9 | 77 | 1 | REFERENCED | 0.325 | 0.041 | 0.528 | 0.953 | 0.002 | 0.042 | 1.894 | 1.541 |
| `ai_v9_13_gen11_labelonly_winprob_0815` | ai_v9 | 77 | 1 | REFERENCED | 0.331 | 0.041 | 0.496 | 0.933 | 0.002 | 0.084 | 1.894 | 1.49 |
| `ai_v9_142_N1_0901` | ai_v9 | 107 | 1 | REFERENCED | 1.056 | 0.036 | 0.509 | 0.183 | 0.002 | 0.074 | 1.887 | 0.0 |
| `ai_v6_13_outgoing_dmg_0620` | ai_v6 | 41 | 1 | REFERENCED | 0.542 | 0.045 | 0.903 | 0.203 | 0.007 | 0.137 | 1.841 | 0.406 |
| `ai_v9_18_gen15_v8rewards_0818` | ai_v9 | 95 | 1 | REFERENCED | 0.336 | 0.042 | 0.504 | 0.851 | 0.004 | 0.085 | 1.83 | 0.946 |
| `ai_v5_4_pbrs_opp_threat_50m_0607` | ai_v5 | 3 | 1 | REFERENCED | 0.17 | 0.028 | 0.283 | 0.07 | 0.001 | 1.274 | 1.829 | 0.396 |
| `ai_v8_12_defensive20_exploiter_0724` | ai_v8 | 45 | 1 | REFERENCED | 0.938 | 0.045 | 0.0 | 0.77 | 0.014 | 0.046 | 1.819 | 1.458 |
| `ai_v13_22_popr1_loop` | ai_v13 | 119 | 0 | LIVE | 0.565 | 0.035 | 0.706 | 0.361 | 0.022 | 0.113 | 1.809 | 0.0 |
| `ai_v13_28_popr2_ctrl` | ai_v13 | 119 | 0 | LIVE | 0.529 | 0.035 | 0.706 | 0.384 | 0.024 | 0.113 | 1.798 | 0.0 |
| `ai_v13_27_popr2_loop` | ai_v13 | 119 | 0 | LIVE | 0.529 | 0.035 | 0.706 | 0.379 | 0.024 | 0.113 | 1.794 | 0.0 |
| `ai_v13_23_popr1_ctrl` | ai_v13 | 119 | 0 | LIVE | 0.529 | 0.035 | 0.706 | 0.364 | 0.022 | 0.113 | 1.777 | 0.0 |
| `ai_v9_10_gen9_intent_distcritic_0813` | ai_v9 | 69 | 1 | REFERENCED | 0.256 | 0.037 | 0.475 | 0.943 | 0.002 | 0.038 | 1.756 | 0.931 |
| `ai_v14_08_g0p_k3` | ai_v14 | 123 | 0 | LIVE | 0.568 | 0.035 | 0.709 | 0.295 | 0.022 | 0.113 | 1.753 | 0.0 |
| `ai_v14_05_lbat_l95` | ai_v14 | 123 | 0 | LIVE | 0.532 | 0.035 | 0.709 | 0.33 | 0.018 | 0.113 | 1.747 | 0.0 |
| `ai_v14_02_lbat_ctrl` | ai_v14 | 123 | 0 | LIVE | 0.532 | 0.035 | 0.709 | 0.327 | 0.018 | 0.113 | 1.743 | 0.0 |
| `ai_v14_03_lbat_e5` | ai_v14 | 123 | 0 | LIVE | 0.532 | 0.035 | 0.709 | 0.324 | 0.018 | 0.113 | 1.74 | 0.0 |
| `ai_v13_12_plateau` | ai_v13 | 119 | 0 | LIVE | 0.529 | 0.035 | 0.706 | 0.327 | 0.02 | 0.113 | 1.738 | 0.0 |
| `ai_v14_06_lbat_ctrl_fix` | ai_v14 | 123 | 0 | LIVE | 0.532 | 0.035 | 0.709 | 0.313 | 0.018 | 0.113 | 1.73 | 0.0 |
| `ai_v14_07_g0p_k2` | ai_v14 | 123 | 0 | LIVE | 0.532 | 0.035 | 0.709 | 0.31 | 0.02 | 0.113 | 1.729 | 0.0 |
| `ai_v7_02_critic_shape_0627` | ai_v7 | 42 | 1 | REFERENCED | 0.478 | 0.043 | 0.868 | 0.209 | 0.008 | 0.088 | 1.701 | 0.391 |
| `ai_v9_37_tick1_dosext_0825` | ai_v9 | 101 | 1 | REFERENCED | 1.202 | 0.036 | 0.073 | 0.185 | 0.008 | 0.074 | 1.658 | 1.165 |
| `ai_v9_19_gen16_mechanics_0819` | ai_v9 | 97 | 1 | REFERENCED | 0.285 | 0.036 | 0.427 | 0.799 | 0.004 | 0.072 | 1.629 | 0.853 |
| `ai_v9_21_gen17_pfspoff_0820` | ai_v9 | 97 | 1 | REFERENCED | 0.285 | 0.036 | 0.427 | 0.789 | 0.004 | 0.072 | 1.618 | 0.845 |
| `ai_v12_12_ladder_cflabels` | ai_v12 | 114 | 1 | REFERENCED | 0.67 | 0.035 | 0.141 | 0.631 | 0.002 | 0.117 | 1.603 | 0.0 |
| `ai_v13_07_fold1` | ai_v13 | 119 | 1 | REFERENCED | 0.388 | 0.035 | 0.706 | 0.291 | 0.018 | 0.111 | 1.555 | 0.0 |
| `ai_v8_14_distill3_0725` | ai_v8 | 45 | 1 | REFERENCED | 0.625 | 0.045 | 0.089 | 0.725 | 0.013 | 0.046 | 1.551 | 0.0 |
| `ai_v13_08_fold1_cont` | ai_v13 | 119 | 1 | REFERENCED | 0.388 | 0.035 | 0.706 | 0.285 | 0.019 | 0.111 | 1.55 | 0.0 |
| `ai_v8_15_retention_A_frozen_0726` | ai_v8 | 45 | 1 | REFERENCED | 0.715 | 0.045 | 0.089 | 0.604 | 0.015 | 0.046 | 1.518 | 1.135 |
| `ai_v9_67_R3F6e_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.205 | 0.006 | 0.038 | 1.516 | 1.093 |
| `ai_v9_40_fdC_ecology_0825` | ai_v9 | 101 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.174 | 0.006 | 0.038 | 1.514 | 1.165 |
| `ai_v7_03_belief_shape_0630` | ai_v7 | 42 | 1 | REFERENCED | 0.304 | 0.043 | 0.868 | 0.229 | 0.004 | 0.045 | 1.502 | 1.085 |
| `ai_v9_62_R2PLAIN_0827` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.154 | 0.006 | 0.037 | 1.495 | 0.656 |
| `ai_v9_42_fdE_single_0825` | ai_v9 | 101 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.15 | 0.006 | 0.038 | 1.491 | 1.165 |
| `ai_v9_38_fdA_coef03_0825` | ai_v9 | 101 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.151 | 0.006 | 0.037 | 1.489 | 1.165 |
| `ai_v9_60_R2TOPK_0827` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.143 | 0.006 | 0.037 | 1.484 | 1.165 |
| `ai_v9_59_R2ACTION_0827` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.143 | 0.006 | 0.037 | 1.484 | 0.0 |
| `ai_v9_68_R3F6f_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.177 | 0.006 | 0.037 | 1.482 | 1.093 |
| `ai_v9_61_R2KL_0827` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.142 | 0.006 | 0.037 | 1.481 | 1.165 |
| `ai_v9_48_G1_action_0826` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.141 | 0.006 | 0.037 | 1.48 | 0.0 |
| `ai_v9_49_G2_advgate_0826` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.139 | 0.006 | 0.037 | 1.478 | 1.165 |
| `ai_v9_52_G1p_matched_0826` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.138 | 0.006 | 0.037 | 1.477 | 1.165 |
| `ai_v9_39_fdB_lossonly_0825` | ai_v9 | 101 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.137 | 0.006 | 0.037 | 1.476 | 1.165 |
| `ai_v9_63_R3F6a_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.167 | 0.006 | 0.038 | 1.468 | 1.093 |
| `ai_v9_66_R3F6d_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.167 | 0.006 | 0.037 | 1.468 | 1.093 |
| `ai_v9_64_R3F6b_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.167 | 0.006 | 0.038 | 1.467 | 1.093 |
| `ai_v9_65_R3F6c_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.161 | 0.006 | 0.038 | 1.464 | 1.093 |
| `ai_v9_69_R3F6CURR_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.157 | 0.006 | 0.038 | 1.455 | 1.093 |
| `ai_v8_09_pool10_exploiter_0723` | ai_v8 | 45 | 1 | REFERENCED | 0.715 | 0.045 | 0.0 | 0.631 | 0.013 | 0.046 | 1.454 | 0.0 |
| `ai_v8_07_semistall564_scratch_0722` | ai_v8 | 45 | 1 | REFERENCED | 0.661 | 0.035 | 0.0 | 0.672 | 0.002 | 0.071 | 1.452 | 1.167 |
| `ai_v12_20_ladder_denseaux` | ai_v12 | 117 | 1 | REFERENCED | 0.186 | 0.062 | 0.249 | 0.769 | 0.002 | 0.175 | 1.447 | 0.0 |
| `ai_v9_82_REFOLD1_0830` | ai_v9 | 107 | 1 | REFERENCED | 1.056 | 0.036 | 0.073 | 0.133 | 0.007 | 0.074 | 1.404 | 0.947 |
| `ai_v9_24_E3_substrate_on_0822` | ai_v9 | 97 | 1 | REFERENCED | 0.107 | 0.036 | 0.64 | 0.536 | 0.006 | 0.072 | 1.403 | 1.017 |
| `ai_v6_11_typed_hp_0619` | ai_v6 | 38 | 1 | REFERENCED | 0.263 | 0.044 | 0.876 | 0.168 | 0.003 | 0.045 | 1.402 | 1.051 |
| `ai_v9_25_E4_baitbot_0822` | ai_v9 | 98 | 1 | REFERENCED | 0.107 | 0.036 | 0.711 | 0.448 | 0.007 | 0.072 | 1.386 | 0.323 |
| `ai_v5_12_bias_05_N_0612` | ai_v5 | 12 | 1 | REFERENCED | 0.117 | 0.029 | 0.235 | 0.107 | 0.0 | 0.882 | 1.38 | 0.294 |
| `ai_v9_23_E2_substrate_on_0822` | ai_v9 | 97 | 1 | REFERENCED | 0.107 | 0.036 | 0.64 | 0.501 | 0.006 | 0.072 | 1.369 | 1.0 |
| `ai_v6_13_outgoing_dmg_0620_exp_v1` | ai_v6 | 41 | 1 | REFERENCED | 0.135 | 0.045 | 0.903 | 0.202 | 0.009 | 0.046 | 1.345 | 0.045 |
| `ai_v5_10_tail1_23_0611` | ai_v5 | 11 | 1 | REFERENCED | 0.117 | 0.029 | 0.205 | 0.106 | 0.0 | 0.852 | 1.32 | 0.264 |
| `ai_v9_22_E1_substrate_on_0821` | ai_v9 | 97 | 1 | REFERENCED | 0.107 | 0.036 | 0.64 | 0.432 | 0.006 | 0.072 | 1.297 | 0.321 |
| `ai_v9_26_baitent_probe_0823` | ai_v9 | 100 | 1 | REFERENCED | 0.0 | 0.036 | 0.711 | 0.377 | 0.006 | 0.107 | 1.24 | 0.0 |
| `ai_v13_16_teach5_offense_dist` | ai_v13 | 119 | 1 | REFERENCED | 0.071 | 0.035 | 0.706 | 0.277 | 0.022 | 0.112 | 1.225 | 0.14 |
| `ai_v8_17_rand20_nolut_0726` | ai_v8 | 46 | 1 | REFERENCED | 0.581 | 0.045 | 0.0 | 0.53 | 0.013 | 0.045 | 1.218 | 0.872 |
| `ai_v6_09_dmg_reattend_N_0617` | ai_v6 | 35 | 1 | REFERENCED | 0.217 | 0.043 | 0.78 | 0.122 | 0.002 | 0.044 | 1.215 | 0.91 |
| `ai_v9_162_TCUNFA_0903` | ai_v9 | 107 | 1 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.165 | 0.002 | 0.11 | 1.213 | 0.0 |
| `ai_v8_20_rand10_nolut_0727` | ai_v8 | 46 | 1 | REFERENCED | 0.581 | 0.045 | 0.0 | 0.519 | 0.013 | 0.045 | 1.207 | 0.864 |
| `ai_v8_16_def20_lut_0726` | ai_v8 | 46 | 1 | REFERENCED | 0.564 | 0.04 | 0.0 | 0.535 | 0.013 | 0.041 | 1.199 | 0.869 |
| `ai_v9_81_REVIVE1c_0830` | ai_v9 | 107 | 1 | REFERENCED | 0.947 | 0.036 | 0.0 | 0.135 | 0.007 | 0.037 | 1.197 | 0.838 |
| `ai_v9_150_R4DOSE12_0901` | ai_v9 | 107 | 1 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.181 | 0.002 | 0.074 | 1.193 | 0.0 |
| `ai_v8_19_def20_lut_zeroinit_0727` | ai_v8 | 46 | 1 | REFERENCED | 0.56 | 0.04 | 0.0 | 0.525 | 0.013 | 0.041 | 1.185 | 0.856 |
| `ai_v9_151_R4DOSE6_0901` | ai_v9 | 107 | 1 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.173 | 0.002 | 0.074 | 1.184 | 0.0 |
| `ai_v9_161_TCFUNDB_0903` | ai_v9 | 107 | 1 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.172 | 0.002 | 0.074 | 1.184 | 0.0 |
| `ai_v9_80_REVIVE1b_0830` | ai_v9 | 107 | 1 | REFERENCED | 0.947 | 0.036 | 0.0 | 0.125 | 0.007 | 0.037 | 1.183 | 0.838 |
| `ai_v9_160_TCFUNDA_0903` | ai_v9 | 107 | 1 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.166 | 0.002 | 0.074 | 1.179 | 0.0 |
| `ai_v9_172_G1SHORT_0905` | ai_v9 | 107 | 1 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.167 | 0.002 | 0.074 | 1.179 | 0.837 |
| `ai_v9_170_TCUNFK6A_0904` | ai_v9 | 107 | 1 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.166 | 0.002 | 0.074 | 1.178 | 0.0 |
| `ai_v9_163_TCUNFB_0903` | ai_v9 | 107 | 1 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.166 | 0.002 | 0.074 | 1.177 | 0.0 |
| `ai_v8_18_rand20_lut_0726` | ai_v8 | 46 | 1 | REFERENCED | 0.564 | 0.04 | 0.0 | 0.513 | 0.013 | 0.041 | 1.176 | 0.851 |
| `ai_v9_171_TCUNFK6B_0904` | ai_v9 | 107 | 1 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.162 | 0.002 | 0.074 | 1.176 | 0.0 |
| `ai_v8_10_offense20_exploiter_0724` | ai_v8 | 45 | 1 | REFERENCED | 0.581 | 0.045 | 0.0 | 0.489 | 0.013 | 0.046 | 1.176 | 0.838 |
| `ai_v9_79_REVIVE1a_0830` | ai_v9 | 107 | 1 | REFERENCED | 0.947 | 0.036 | 0.0 | 0.12 | 0.007 | 0.037 | 1.175 | 0.838 |
| `ai_v9_77_G1LEAN_0830` | ai_v9 | 107 | 1 | REFERENCED | 0.911 | 0.036 | 0.0 | 0.139 | 0.006 | 0.037 | 1.164 | 0.801 |
| `ai_v12_23_ladder_rollout` | ai_v12 | 119 | 1 | REFERENCED | 0.076 | 0.039 | 0.19 | 0.682 | 0.002 | 0.136 | 1.149 | 0.0 |
| `ai_v5_2_native_selfplay_50m_0606` | ai_v5 | 2 | 1 | REFERENCED | 0.113 | 0.028 | 0.085 | 0.058 | 0.0 | 0.85 | 1.141 | 0.141 |
| `ai_v8_13_defensive10_exploiter_0725` | ai_v8 | 45 | 1 | REFERENCED | 0.536 | 0.045 | 0.0 | 0.493 | 0.013 | 0.045 | 1.136 | 0.0 |
| `ai_v12_19_ladder_lambda09` | ai_v12 | 116 | 1 | REFERENCED | 0.106 | 0.035 | 0.176 | 0.683 | 0.002 | 0.119 | 1.126 | 0.0 |
| `ai_v9_152_R4DOSE3_0901` | ai_v9 | 107 | 1 | REFERENCED | 0.291 | 0.036 | 0.509 | 0.182 | 0.002 | 0.074 | 1.122 | 0.0 |
| `ai_v6_03_win_pred_N_0614` | ai_v6 | 22 | 1 | REFERENCED | 0.29 | 0.032 | 0.644 | 0.114 | 0.004 | 0.034 | 1.121 | 0.869 |
| `ai_v12_04_pfsp_fork25M` | ai_v12 | 113 | 1 | REFERENCED | 0.071 | 0.035 | 0.706 | 0.247 | 0.007 | 0.038 | 1.108 | 0.776 |
| `ai_v7_01_teacher_0626` | ai_v7 | 42 | 1 | REFERENCED | 0.217 | 0.043 | 0.434 | 0.212 | 0.002 | 0.175 | 1.092 | 0.564 |
| `ai_v12_22_ladder_lambda095` | ai_v12 | 116 | 1 | REFERENCED | 0.106 | 0.035 | 0.176 | 0.649 | 0.002 | 0.119 | 1.092 | 0.0 |
| `ai_v12_26_ladder_ctrl10M_shaped` | ai_v12 | 113 | 1 | REFERENCED | 0.118 | 0.04 | 0.039 | 0.8 | 0.002 | 0.083 | 1.087 | 0.485 |
| `ai_v12_15_ladder_ctrl10M_b` | ai_v12 | 114 | 1 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.679 | 0.002 | 0.117 | 1.084 | 0.0 |
| `ai_v6_11_unified_obs_fixed_0618` | ai_v6 | 37 | 1 | REFERENCED | 0.217 | 0.043 | 0.608 | 0.156 | 0.002 | 0.044 | 1.075 | 0.738 |
| `ai_v12_16_ladder_ctrl10M_c` | ai_v12 | 114 | 1 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.635 | 0.002 | 0.117 | 1.04 | 0.0 |
| `ai_v12_21_ladder_lambda09_b` | ai_v12 | 116 | 1 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.629 | 0.002 | 0.119 | 1.036 | 0.0 |
| `ai_v12_17_ladder_strata` | ai_v12 | 115 | 1 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.627 | 0.002 | 0.117 | 1.032 | 0.0 |
| `ai_v12_14_ladder_truevalue` | ai_v12 | 115 | 1 | REFERENCED | 0.109 | 0.036 | 0.145 | 0.617 | 0.002 | 0.119 | 1.031 | 0.0 |
| `ai_v9_58_R2CTRL_0827` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.073 | 0.114 | 0.005 | 0.037 | 1.019 | 0.656 |
| `ai_v9_50_fdF_p1c_0826` | ai_v9 | 103 | 1 | REFERENCED | 0.364 | 0.036 | 0.473 | 0.07 | 0.005 | 0.038 | 1.011 | 0.291 |
| `ai_v12_24_ladder_strata_b` | ai_v12 | 115 | 1 | REFERENCED | 0.071 | 0.035 | 0.141 | 0.64 | 0.002 | 0.116 | 1.01 | 0.0 |
| `ai_v9_45_fdF_p1_0826` | ai_v9 | 102 | 1 | REFERENCED | 0.364 | 0.036 | 0.473 | 0.068 | 0.005 | 0.038 | 1.006 | 0.0 |
| `ai_v12_01_winprob_critic` | ai_v12 | 109 | 1 | REFERENCED | 0.146 | 0.024 | 0.268 | 0.53 | 0.002 | 0.025 | 1.001 | 0.522 |
| `ai_v9_51_fdF_p2c_0826` | ai_v9 | 103 | 1 | REFERENCED | 0.328 | 0.036 | 0.473 | 0.079 | 0.006 | 0.037 | 0.981 | 0.728 |
| `ai_v9_102_R5F10_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.138 | 0.005 | 0.037 | 0.973 | 0.0 |
| `ai_v9_94_R5F02_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.135 | 0.005 | 0.037 | 0.972 | 0.0 |
| `ai_v9_95_R5F03_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.131 | 0.005 | 0.037 | 0.966 | 0.656 |
| `ai_v9_110_R5F18_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.129 | 0.006 | 0.037 | 0.963 | 0.656 |
| `ai_v9_111_R5F19_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.127 | 0.006 | 0.037 | 0.96 | 0.656 |
| `ai_v9_100_R5F08_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.124 | 0.005 | 0.037 | 0.958 | 0.0 |
| `ai_v14_04b_lbat_t32` | ai_v14 | 123 | 0 | LIVE | 0.071 | 0.035 | 0.709 | 0.078 | 0.016 | 0.043 | 0.958 | 0.0 |
| `ai_v9_98_R5F06_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.12 | 0.005 | 0.037 | 0.955 | 0.0 |
| `ai_v9_106_R5F14_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.121 | 0.005 | 0.037 | 0.954 | 0.0 |
| `ai_v9_92_R5F00_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.123 | 0.005 | 0.037 | 0.953 | 0.0 |
| `ai_v9_96_R5F04_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.121 | 0.005 | 0.037 | 0.95 | 0.0 |
| `ai_v9_104_R5F12_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.122 | 0.005 | 0.037 | 0.949 | 0.0 |
| `ai_v9_103_R5F11_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.119 | 0.005 | 0.037 | 0.948 | 0.656 |
| `ai_v9_107_R5F15_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.121 | 0.006 | 0.037 | 0.948 | 0.656 |
| `ai_v9_27_extremedial_probe_0823` | ai_v9 | 100 | 1 | REFERENCED | 0.0 | 0.036 | 0.711 | 0.154 | 0.007 | 0.036 | 0.946 | 0.036 |
| `ai_v9_101_R5F09_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.112 | 0.005 | 0.037 | 0.946 | 0.656 |
| `ai_v9_97_R5F05_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.116 | 0.005 | 0.037 | 0.945 | 0.656 |
| `ai_v9_36_tock1c_q6_0824` | ai_v9 | 101 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.116 | 0.005 | 0.037 | 0.943 | 0.656 |
| `ai_v9_57_R2F5e_0826` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.115 | 0.005 | 0.037 | 0.942 | 0.0 |
| `ai_v9_93_R5F01_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.113 | 0.005 | 0.037 | 0.941 | 0.656 |
| `ai_v9_105_R5F13_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.111 | 0.005 | 0.037 | 0.938 | 0.656 |
| `ai_v9_108_R5F16_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.113 | 0.006 | 0.037 | 0.938 | 0.656 |
| `ai_v9_56_R2F5d_0826` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.111 | 0.005 | 0.037 | 0.938 | 0.0 |
| `ai_v9_55_R2F5c_0826` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.109 | 0.005 | 0.037 | 0.938 | 0.0 |
| `ai_v9_109_R5F17_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.111 | 0.006 | 0.037 | 0.936 | 0.656 |
| `ai_v9_31_tock1_k4_0824` | ai_v9 | 101 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.11 | 0.005 | 0.037 | 0.936 | 0.0 |
| `ai_v9_32_tock1b_rain_0824` | ai_v9 | 101 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.108 | 0.005 | 0.037 | 0.936 | 0.656 |
| `ai_v9_99_R5F07_0831` | ai_v9 | 107 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.11 | 0.005 | 0.037 | 0.935 | 0.656 |
| `ai_v9_54_R2F5b_0826` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.109 | 0.005 | 0.037 | 0.935 | 0.0 |
| `ai_v9_53_R2F5a_0826` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.109 | 0.005 | 0.037 | 0.935 | 0.0 |
| `ai_v6_04_unified_inc_N_0615` | ai_v6 | 23 | 1 | REFERENCED | 0.201 | 0.033 | 0.501 | 0.113 | 0.002 | 0.034 | 0.891 | 0.635 |
| `ai_v13_03_fork` | ai_v13 | 120 | 1 | REFERENCED | 0.114 | 0.039 | 0.153 | 0.355 | 0.002 | 0.131 | 0.817 | 0.0 |
| `ai_v6_07_unified_topk_N_0616` | ai_v6 | 30 | 1 | REFERENCED | 0.152 | 0.038 | 0.38 | 0.144 | 0.001 | 0.08 | 0.804 | 0.494 |
| `ai_v8_04_distill_4teacher_0722` | ai_v8 | 45 | 1 | REFERENCED | 0.357 | 0.045 | 0.045 | 0.284 | 0.012 | 0.046 | 0.798 | 0.0 |
| `ai_v9_20_tdaux_rung2_lam00_0820` | ai_v9 | 97 | 1 | REFERENCED | 0.036 | 0.036 | 0.498 | 0.151 | 0.005 | 0.036 | 0.764 | 0.533 |
| `ai_v9_20_tdaux_rung2_lam30_0820` | ai_v9 | 97 | 1 | REFERENCED | 0.036 | 0.036 | 0.498 | 0.15 | 0.005 | 0.036 | 0.764 | 0.533 |
| `ai_v9_20_tdaux_rung2_lam10_0820` | ai_v9 | 97 | 1 | REFERENCED | 0.036 | 0.036 | 0.498 | 0.148 | 0.005 | 0.036 | 0.763 | 0.533 |
| `ai_v13_16_teach5_offense_dist_ABANDONED_forklr2p8` | ai_v13 | 119 | 1 | REFERENCED | 0.0 | 0.0 | 0.706 | 0.0 | 0.02 | 0.035 | 0.761 | 0.706 |
| `ai_v12_25_ladder_vf15_b` | ai_v12 | 113 | 1 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.371 | 0.002 | 0.074 | 0.733 | 0.0 |
| `ai_v7_05_tss_specialist_0703` | ai_v7 | 42 | 1 | REFERENCED | 0.478 | 0.043 | 0.0 | 0.149 | 0.008 | 0.045 | 0.724 | 0.391 |
| `ai_v12_11_ladder_ctrl10M` | ai_v12 | 113 | 1 | REFERENCED | 0.141 | 0.035 | 0.141 | 0.32 | 0.002 | 0.074 | 0.717 | 0.0 |
| `ai_v12_28_ladder_ent05` | ai_v12 | 113 | 1 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.351 | 0.002 | 0.073 | 0.712 | 0.0 |
| `ai_v12_29_ladder_vf025` | ai_v12 | 113 | 1 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.35 | 0.002 | 0.074 | 0.712 | 0.0 |
| `ai_v12_13_ladder_tdaux` | ai_v12 | 113 | 1 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.329 | 0.002 | 0.074 | 0.691 | 0.0 |
| `ai_v12_10_ladder_vf15` | ai_v12 | 113 | 1 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.325 | 0.002 | 0.074 | 0.687 | 0.0 |
| `ai_v8_06_semistall_3team_exploiter_0722` | ai_v8 | 45 | 1 | REFERENCED | 0.313 | 0.045 | 0.0 | 0.252 | 0.012 | 0.046 | 0.674 | 0.0 |
| `ai_v9_130_R5FUND10_0901` | ai_v9 | 107 | 1 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.067 | 0.006 | 0.037 | 0.648 | 0.0 |
| `ai_v9_128_R5FUND08_0901` | ai_v9 | 107 | 1 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.068 | 0.006 | 0.037 | 0.648 | 0.0 |
| `ai_v9_195_G5PLAINA_0906` | ai_v9 | 107 | 1 | REFERENCED | 0.073 | 0.0 | 0.509 | 0.0 | 0.001 | 0.037 | 0.645 | 0.0 |
| `ai_v9_122_R5FUND02_0901` | ai_v9 | 107 | 1 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.068 | 0.001 | 0.037 | 0.645 | 0.0 |
| `ai_v9_197_G5PLAINC_0906` | ai_v9 | 107 | 1 | REFERENCED | 0.073 | 0.0 | 0.509 | 0.0 | 0.001 | 0.037 | 0.643 | 0.0 |
| `ai_v9_30_rev1_exploit_0824` | ai_v9 | 101 | 1 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.061 | 0.005 | 0.037 | 0.643 | 0.364 |
| `ai_v9_196_G5PLAINB_0906` | ai_v9 | 107 | 1 | REFERENCED | 0.073 | 0.0 | 0.509 | 0.0 | 0.001 | 0.037 | 0.643 | 0.0 |
| `ai_v9_35_tick1_exploit_0824` | ai_v9 | 101 | 1 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.06 | 0.008 | 0.037 | 0.641 | 0.364 |
| `ai_v9_132_R5FUND12_0901` | ai_v9 | 107 | 1 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.063 | 0.006 | 0.037 | 0.636 | 0.0 |
| `ai_v9_126_R5FUND06_0901` | ai_v9 | 107 | 1 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.062 | 0.001 | 0.037 | 0.636 | 0.0 |
| `ai_v9_134_R5FUND14_0901` | ai_v9 | 107 | 1 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.058 | 0.006 | 0.037 | 0.634 | 0.0 |
| `ai_v9_120_R5FUND00_0901` | ai_v9 | 107 | 1 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.062 | 0.001 | 0.037 | 0.632 | 0.0 |
| `ai_v9_124_R5FUND04_0901` | ai_v9 | 107 | 1 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.061 | 0.001 | 0.037 | 0.63 | 0.0 |
| `ai_v12_27_ladder_ctrl10M_shaped_dense` | ai_v12 | 113 | 2 | CLOSED | 0.158 | 0.039 | 0.0 | 0.382 | 0.002 | 0.042 | 0.628 | 0.303 |
| `ai_v6_08_unmasked_floor_N_0617` | ai_v6 | 30 | 1 | REFERENCED | 0.114 | 0.038 | 0.304 | 0.122 | 0.001 | 0.039 | 0.624 | 0.342 |
| `ai_v7_14_league_capstone_0712` | ai_v7 | 43 | 1 | REFERENCED | 0.217 | 0.043 | 0.087 | 0.16 | 0.011 | 0.044 | 0.567 | 0.13 |
| `.aborted_R4DOSE12_nometa_1401` | ai_v9 (via lineage) | 107 | 1 | REFERENCED | 0.0 | 0.0 | 0.509 | 0.0 | 0.001 | 0.037 | 0.56 | 0.509 |
| `ai_v13_15_exploit5_stall` | ai_v13 | 119 | 1 | REFERENCED | 0.071 | 0.035 | 0.0 | 0.3 | 0.022 | 0.112 | 0.554 | 0.143 |
| `ai_v7_09_tss_bots_pubval_0708` | ai_v7 | 43 | 1 | REFERENCED | 0.262 | 0.044 | 0.0 | 0.152 | 0.004 | 0.089 | 0.551 | 0.175 |
| `ai_v7_15_tss_exploiter_vs14_0713` | ai_v7 | 43 | 1 | REFERENCED | 0.35 | 0.044 | 0.0 | 0.087 | 0.006 | 0.045 | 0.537 | 0.263 |
| `.dryrun_K6A_1788581936` | ai_v9 (via lineage) | 107 | 1 | REFERENCED | 0.0 | 0.0 | 0.509 | 0.0 | 0.001 | 0.0 | 0.522 | 0.509 |
| `ai_v7_19_combined_0716` | ai_v7 | 43 | 1 | REFERENCED | 0.13 | 0.043 | 0.087 | 0.188 | 0.013 | 0.044 | 0.522 | 0.13 |
| `ai_v13_10_exploit_stall` | ai_v13 | 119 | 1 | REFERENCED | 0.071 | 0.035 | 0.0 | 0.274 | 0.018 | 0.112 | 0.52 | 0.136 |
| `ai_v13_14_exploit5_balance` | ai_v13 | 119 | 1 | REFERENCED | 0.071 | 0.035 | 0.0 | 0.271 | 0.022 | 0.112 | 0.519 | 0.142 |
| `ai_v7_08_tss_bots_0707` | ai_v7 | 42 | 1 | REFERENCED | 0.26 | 0.043 | 0.0 | 0.163 | 0.004 | 0.045 | 0.515 | 0.174 |
| `ai_v8_11_offense10_exploiter_0724` | ai_v8 | 45 | 1 | REFERENCED | 0.223 | 0.045 | 0.0 | 0.183 | 0.012 | 0.045 | 0.509 | 0.223 |
| `ai_v6_06_unified_all_N_0616` | ai_v6 | 28 | 1 | REFERENCED | 0.139 | 0.035 | 0.174 | 0.119 | 0.001 | 0.035 | 0.508 | 0.243 |
| `ai_v13_05_exploit_big5starmie` | ai_v13 | 119 | 1 | REFERENCED | 0.106 | 0.035 | 0.0 | 0.223 | 0.018 | 0.112 | 0.5 | 0.0 |
| `ai_v13_24_popr1_read_loop` | ai_v13 | 119 | 0 | LIVE | 0.106 | 0.035 | 0.0 | 0.219 | 0.024 | 0.112 | 0.497 | 0.0 |
| `ai_v6_13_outgoing_dmg_0620_exploiter_v2` | ai_v6 | 41 | 1 | REFERENCED | 0.226 | 0.045 | 0.0 | 0.121 | 0.002 | 0.092 | 0.493 | 0.181 |
| `ai_v13_18_teach5_offense_hidose` | ai_v13 | 119 | 0 | LIVE | 0.106 | 0.035 | 0.0 | 0.215 | 0.022 | 0.112 | 0.491 | 0.0 |
| `ai_v13_30_popr2_read_ctrl` | ai_v13 | 119 | 0 | LIVE | 0.071 | 0.035 | 0.0 | 0.219 | 0.026 | 0.112 | 0.465 | 0.0 |
| `ai_v13_13_exploit5_offense` | ai_v13 | 119 | 0 | LIVE | 0.071 | 0.035 | 0.0 | 0.221 | 0.022 | 0.112 | 0.462 | 0.0 |
| `ai_v13_29_popr2_read_loop` | ai_v13 | 119 | 0 | LIVE | 0.071 | 0.035 | 0.0 | 0.217 | 0.026 | 0.112 | 0.462 | 0.0 |
| `ai_v13_26_popr0_exploit5_offense_s1002` | ai_v13 | 119 | 0 | LIVE | 0.071 | 0.035 | 0.0 | 0.219 | 0.022 | 0.112 | 0.461 | 0.0 |
| `ai_v13_06_exploit_ddtar_spikes` | ai_v13 | 119 | 1 | REFERENCED | 0.071 | 0.035 | 0.0 | 0.217 | 0.018 | 0.112 | 0.454 | 0.127 |
| `ai_v13_25_popr1_read_ctrl` | ai_v13 | 119 | 0 | LIVE | 0.071 | 0.035 | 0.0 | 0.211 | 0.024 | 0.112 | 0.454 | 0.0 |
| `ai_v6_13_outgoing_dmg_0620_exploiter_v1` | ai_v6 | 41 | 1 | REFERENCED | 0.226 | 0.045 | 0.0 | 0.12 | 0.002 | 0.046 | 0.442 | 0.135 |
| `ai_v7_06_tss_temp_anneal_0706` | ai_v7 | 42 | 1 | REFERENCED | 0.174 | 0.043 | 0.0 | 0.152 | 0.002 | 0.044 | 0.416 | 0.087 |
| `ai_v8_02_zarch_teampfsp_0718` | ai_v8 | 44 | 1 | REFERENCED | 0.089 | 0.045 | 0.089 | 0.129 | 0.002 | 0.045 | 0.404 | 0.134 |
| `ai_v7_17_stall_exploiter_0715` | ai_v7 | 43 | 1 | REFERENCED | 0.174 | 0.044 | 0.0 | 0.112 | 0.002 | 0.044 | 0.385 | 0.087 |
| `ai_v8_08_defensive_6team_exploiter_0723` | ai_v8 | 45 | 1 | REFERENCED | 0.134 | 0.045 | 0.0 | 0.132 | 0.012 | 0.045 | 0.377 | 0.089 |
| `ai_v6_02_belief_lat_16m_0614` | ai_v6 | 21 | 1 | REFERENCED | 0.094 | 0.031 | 0.125 | 0.083 | 0.001 | 0.032 | 0.371 | 0.156 |
| `ai_v9_17_tdaux_lam1_0818` | ai_v9 | 95 | 1 | REFERENCED | 0.042 | 0.042 | 0.084 | 0.149 | 0.005 | 0.043 | 0.369 | 0.126 |
| `ai_v8_05_semistall564_exploiter_0722` | ai_v8 | 45 | 1 | REFERENCED | 0.134 | 0.045 | 0.0 | 0.122 | 0.012 | 0.045 | 0.365 | 0.089 |
| `ai_v9_17_tdaux_lam3_0818` | ai_v9 | 95 | 1 | REFERENCED | 0.042 | 0.042 | 0.084 | 0.145 | 0.005 | 0.043 | 0.364 | 0.042 |
| `ai_v7_07_tss_temp_ratchet_0707` | ai_v7 | 42 | 1 | REFERENCED | 0.13 | 0.043 | 0.0 | 0.14 | 0.001 | 0.044 | 0.36 | 0.043 |
| `ai_v7_11_tss_exploiter_nopubval` | ai_v7 | 43 | 1 | REFERENCED | 0.174 | 0.043 | 0.0 | 0.086 | 0.002 | 0.044 | 0.354 | 0.087 |
| `ai_v7_21_fitnet_valuefeat_ab_0717` | ai_v7 | 43 | 1 | REFERENCED | 0.087 | 0.043 | 0.043 | 0.113 | 0.012 | 0.044 | 0.348 | 0.043 |
| `ai_v7_13_cmpass_exploiter_0711` | ai_v7 | 43 | 1 | REFERENCED | 0.174 | 0.044 | 0.0 | 0.08 | 0.002 | 0.044 | 0.345 | 0.087 |
| `ai_v7_18_distill_4teacher_0716` | ai_v7 | 43 | 1 | REFERENCED | 0.087 | 0.043 | 0.043 | 0.109 | 0.011 | 0.044 | 0.344 | 0.0 |
| `ai_v7_16_distill_tss_mvp_0715` | ai_v7 | 43 | 1 | REFERENCED | 0.087 | 0.043 | 0.043 | 0.1 | 0.011 | 0.044 | 0.334 | 0.043 |
| `ai_v7_12_trap_exploiter_0711` | ai_v7 | 43 | 1 | REFERENCED | 0.131 | 0.044 | 0.0 | 0.095 | 0.001 | 0.044 | 0.326 | 0.044 |
| `ai_v9_17_tdaux_control_0818` | ai_v9 | 95 | 1 | REFERENCED | 0.042 | 0.042 | 0.042 | 0.147 | 0.005 | 0.043 | 0.325 | 0.084 |
| `v8rep_p1_A_0905` | ai_v8 (replication) | 45 | 1 | REFERENCED | 0.089 | 0.045 | 0.045 | 0.086 | 0.0 | 0.045 | 0.323 | 0.045 |
| `ai_v7_20_valuedistill_ab_0717` | ai_v7 | 43 | 1 | REFERENCED | 0.087 | 0.043 | 0.043 | 0.082 | 0.011 | 0.044 | 0.319 | 0.043 |
| `ai_v6_04_unified_all_half_batch_N_0616` | ai_v6 | 28 | 1 | REFERENCED | 0.104 | 0.035 | 0.035 | 0.101 | 0.001 | 0.036 | 0.316 | 0.069 |
| `ai_v6_10_unified_obs_0618` | ai_v6 | 37 | 1 | REFERENCED | 0.087 | 0.043 | 0.0 | 0.134 | 0.001 | 0.044 | 0.316 | 0.0 |
| `ai_v7_10_tss_exploiter_fixed_0709` | ai_v7 | 43 | 1 | REFERENCED | 0.131 | 0.044 | 0.0 | 0.089 | 0.002 | 0.045 | 0.315 | 0.044 |
| `DISCARDED_tdaux_control_n16_0818` | ai_v9 (via lineage) | 95 | 1 | REFERENCED | 0.082 | 0.041 | 0.041 | 0.074 | 0.005 | 0.042 | 0.294 | 0.041 |
| `v8rep_p1_C_0905` | ai_v8 (replication) | 45 | 1 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.088 | 0.0 | 0.045 | 0.274 | 0.045 |
| `v8rep_p1_B_0905` | ai_v8 (replication) | 45 | 1 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.088 | 0.0 | 0.045 | 0.273 | 0.045 |
| `v8rep_p2loss_B_0905` | ai_v8 (replication) | 45 | 1 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.086 | 0.0 | 0.045 | 0.27 | 0.045 |
| `v8rep_p2loss_C_0905` | ai_v8 (replication) | 45 | 1 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.087 | 0.0 | 0.045 | 0.269 | 0.045 |
| `v8rep_p2loss_A_0905` | ai_v8 (replication) | 45 | 1 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.085 | 0.0 | 0.045 | 0.267 | 0.045 |
| `ai_v13_31_popr1_read_loop_ext` | ai_v13 | 119 | 0 | LIVE | 0.035 | 0.035 | 0.0 | 0.11 | 0.025 | 0.057 | 0.264 | 0.0 |
| `ai_v13_32_popr1_read_ctrl_ext` | ai_v13 | 119 | 0 | LIVE | 0.035 | 0.035 | 0.0 | 0.107 | 0.025 | 0.057 | 0.259 | 0.0 |
| `v8rep_p2self_C_0905` | ai_v8 (replication) | 45 | 1 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.067 | 0.0 | 0.045 | 0.249 | 0.0 |
| `v8rep_p2self_A_0905` | ai_v8 (replication) | 45 | 1 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.068 | 0.0 | 0.045 | 0.249 | 0.0 |
| `v8rep_p2self_B_0905` | ai_v8 (replication) | 45 | 1 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.068 | 0.0 | 0.045 | 0.248 | 0.0 |
| `ai_v9_11_gen10_intentfull_compiled_0814` | ai_v9 | 77 | 1 | REFERENCED | 0.041 | 0.041 | 0.041 | 0.076 | 0.0 | 0.041 | 0.246 | 0.041 |
| `ai_v14_09_r0_offense_a` | ai_v14 | 124 | 0 | LIVE | 0.0 | 0.035 | 0.0 | 0.053 | 0.02 | 0.041 | 0.15 | 0.0 |
| `ai_v7_01_teacher_0626_oom1` | ai_v7 | 42 | 1 | REFERENCED | 0.043 | 0.0 | 0.0 | 0.0 | 0.0 | 0.044 | 0.1 | 0.0 |
| `.aborted_R4DOSE12_poolless_1355` | ai_v9 (via lineage) | 107 | 1 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.001 | 0.037 | 0.053 | 0.0 |
| `ai_v7_20_valuedistill_SMOKE` | ai_v7 | 43 | 1 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.011 | 0.042 | 0.053 | 0.0 |
| `ai_v7_05_tss_specialist_0703_aborted_noeval` | ai_v7 | 42 | 1 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.044 | 0.044 | 0.0 |
| `warmstart_generic_0715` | unknown | 43 | 1 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.043 | 0.043 | 0.0 |
| `run_20260830_184043` | unknown | 107 | 1 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.027 | 0.027 | 0.0 |
| `run_20260830_180409` | unknown | 107 | 1 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.027 | 0.027 | 0.0 |
| `run_20260906_083317` | unknown | 107 | 1 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.01 | 0.01 | 0.0 |
| `run_20260830_183828` | unknown | 107 | 1 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.01 | 0.01 | 0.0 |
| `RETIRED_c5fork_control_gen13base_0817` | ai_v9 (via lineage) | 90 | 1 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.004 | 0.0 | 0.004 | 0.0 |
| `RETIRED_gen14_framedel_v90_0817` | unknown | 90 | 1 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.001 | 0.001 | 0.0 |
| `ai_v12_02_winprob_critic.OOM_4096` | ai_v12 | 110 | 1 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.001 | 0.0 |

## Why each non-CLOSED run is protected

- **`.aborted_R4DOSE12_nometa_1401`** — REFERENCED: named by 4 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`.aborted_R4DOSE12_poolless_1355`** — REFERENCED: named by 4 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`.dryrun_K6A_1788581936`** — REFERENCED: named by 4 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`DISCARDED_tdaux_control_n16_0818`** — REFERENCED: named by 3 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`RETIRED_c5fork_control_gen13base_0817`** — REFERENCED: named by 3 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`RETIRED_gen14_framedel_v90_0817`** — REFERENCED: named by 3 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v12_01_winprob_critic`** — REFERENCED: loaded by 13 committed script(s): src/agents/model/model_version/fields.py, src/agents/training/instrumented_ppo/calibration.py, src/agents/training/instrumented_ppo/value_terms.py …; named by 9 committed measurement artifact(s): designs/research_state/measurements/anchors_p2_batch_2026-09-22/README.md, designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`ai_v12_02_winprob_critic`** — REFERENCED: named in the ledger's last 1500 lines; loaded by 56 committed script(s): designs/research_state/measurements/elo_calibration_external_anchors_2026-09-14/cells.py, designs/research_state/measurements/elo_calibration_external_anchors_2026-09-14/fit_joint.py, designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/entropy_forensics.py …; named by 164 committed measurement artifact(s): designs/research_state/measurements/anchor_ab_continuation_2026-09-20/README.md, designs/research_state/measurements/docs_staleness_census_2026-09-08.md, designs/research_state/measurements/elo_calibration_external_anchors_2026-09-14/PREDICTION.md …; named by another run's model graph: ai_v12_02_winprob_critic.OOM_4096 (argv mention), ai_v12_04_pfsp_fork25M (argv fork_parent), ai_v12_04_pfsp_fork25M (argv mention) …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_02_winprob_critic.OOM_4096`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json
- **`ai_v12_04_pfsp_fork25M`** — REFERENCED: named by 3 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/winprob_critic_75M_read_2026-09-08/critic_gate.md, designs/research_state/measurements/winprob_critic_75M_read_2026-09-08/identity_test.md
- **`ai_v12_10_ladder_vf15`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, designs/research_state/measurements/repr_class_decode_2026-09-11/run.sh, designs/research_state/measurements/repr_class_decode_2026-09-11/tabulate.py …; named by 33 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/vf15_vs_ctrl10M_2026-09-08/README.md, designs/research_state/measurements/critic_ladder_reads/vf15_vs_ctrl10M_2026-09-08/critic_read.json, designs/research_state/measurements/critic_ladder_reads/vf15_vs_ctrl10M_2026-09-08/critic_read.md …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_11_ladder_ctrl10M`** — REFERENCED: loaded by 31 committed script(s): designs/research_state/measurements/critic_ladder_reads/cflabels_vs_ctrl10M_2026-09-09/matched_quota/subsample.py, designs/research_state/measurements/critic_ladder_reads/ent05_vs_ctrl10M_2026-09-12/hp_ent05.sh, designs/research_state/measurements/critic_ladder_reads/strata_b_vs_ctrl10M_2026-09-12/hp_strata_b.sh …; named by 223 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/cflabels_vs_ctrl10M_2026-09-09/README.md, designs/research_state/measurements/critic_ladder_reads/cflabels_vs_ctrl10M_2026-09-09/critic_read.json, designs/research_state/measurements/critic_ladder_reads/cflabels_vs_ctrl10M_2026-09-09/critic_read.md …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_12_ladder_cflabels`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_ladder_reads/cflabels_vs_ctrl10M_2026-09-09/matched_quota/subsample.py, designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, designs/research_state/measurements/search_dividend_winprob_heads_2026-09-11/run_battery3.sh …; named by 28 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/cflabels_vs_ctrl10M_2026-09-09/README.md, designs/research_state/measurements/critic_ladder_reads/cflabels_vs_ctrl10M_2026-09-09/critic_read.json, designs/research_state/measurements/critic_ladder_reads/cflabels_vs_ctrl10M_2026-09-09/critic_read.md …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_13_ladder_tdaux`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py; named by 26 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/tdaux_vs_ctrl10M_2026-09-09/README.md, designs/research_state/measurements/critic_ladder_reads/tdaux_vs_ctrl10M_2026-09-09/critic_read.json, designs/research_state/measurements/critic_ladder_reads/tdaux_vs_ctrl10M_2026-09-09/critic_read.md …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_14_ladder_truevalue`** — REFERENCED: loaded by 5 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, src/agents/model/compile_preload.py, src/agents/model/compile_trainer.py …; named by 22 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/truevalue_vs_ctrl10M_2026-09-10/README.md, designs/research_state/measurements/critic_ladder_reads/truevalue_vs_ctrl10M_2026-09-10/critic_read_hp400.json, designs/research_state/measurements/critic_ladder_reads/truevalue_vs_ctrl10M_2026-09-10/critic_read_hp400.md …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_15_ladder_ctrl10M_b`** — REFERENCED: loaded by 10 committed script(s): designs/research_state/measurements/critic_ladder_reads/ent05_vs_ctrl10M_2026-09-12/hp_ent05.sh, designs/research_state/measurements/critic_ladder_reads/strata_b_vs_ctrl10M_2026-09-12/hp_strata_b.sh, designs/research_state/measurements/critic_ladder_reads/vf025_vs_ctrl10M_2026-09-13/hp_vf025.sh …; named by 96 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/ctrl10M_b_vs_ctrl10M_2026-09-09_FLOOR/README.md, designs/research_state/measurements/critic_ladder_reads/ctrl10M_b_vs_ctrl10M_2026-09-09_FLOOR/cond_readout.json, designs/research_state/measurements/critic_ladder_reads/ctrl10M_b_vs_ctrl10M_2026-09-09_FLOOR/critic_read.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_16_ladder_ctrl10M_c`** — REFERENCED: loaded by 9 committed script(s): designs/research_state/measurements/critic_ladder_reads/ent05_vs_ctrl10M_2026-09-12/hp_ent05.sh, designs/research_state/measurements/critic_ladder_reads/strata_b_vs_ctrl10M_2026-09-12/hp_strata_b.sh, designs/research_state/measurements/critic_ladder_reads/vf025_vs_ctrl10M_2026-09-13/hp_vf025.sh …; named by 81 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/ctrl10M_c_vs_ctrl10M_2026-09-09_FLOOR2/README.md, designs/research_state/measurements/critic_ladder_reads/ctrl10M_c_vs_ctrl10M_2026-09-09_FLOOR2/cond_readout.json, designs/research_state/measurements/critic_ladder_reads/ctrl10M_c_vs_ctrl10M_2026-09-09_FLOOR2/critic_read.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_17_ladder_strata`** — REFERENCED: loaded by 6 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, designs/research_state/measurements/repr_class_decode_2026-09-11/run.sh, designs/research_state/measurements/repr_class_decode_2026-09-11/tabulate.py …; named by 35 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/strata_b_vs_ctrl10M_2026-09-12/README.md, designs/research_state/measurements/critic_ladder_reads/strata_vs_ctrl10M_2026-09-09/README.md, designs/research_state/measurements/critic_ladder_reads/strata_vs_ctrl10M_2026-09-09/cond_readout.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_19_ladder_lambda09`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, designs/research_state/measurements/search_dividend_winprob_heads_2026-09-11/run_battery.sh, designs/research_state/measurements/search_dividend_winprob_heads_2026-09-11/run_extension.sh …; named by 60 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/lambda09_interim_4M_2026-09-10/README.md, designs/research_state/measurements/critic_ladder_reads/lambda09_interim_4M_2026-09-10/critic_read.json, designs/research_state/measurements/critic_ladder_reads/lambda09_interim_4M_2026-09-10/critic_read.md …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_20_ladder_denseaux`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, designs/research_state/measurements/search_dividend_winprob_heads_2026-09-11/run_battery2.sh, designs/research_state/measurements/search_dividend_winprob_heads_2026-09-11/run_battery3.sh; named by 33 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/denseaux_vs_ctrl10M_2026-09-10/README.md, designs/research_state/measurements/critic_ladder_reads/denseaux_vs_ctrl10M_2026-09-10/identity_payload.json, designs/research_state/measurements/critic_ladder_reads/denseaux_vs_ctrl10M_2026-09-10/run_readout.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_21_ladder_lambda09_b`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py; named by 23 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/lambda09_b_vs_ctrl10M_2026-09-11/README.md, designs/research_state/measurements/critic_ladder_reads/lambda09_b_vs_ctrl10M_2026-09-11/vs_ctrl10M/critic_read_hp400.json, designs/research_state/measurements/critic_ladder_reads/lambda09_b_vs_ctrl10M_2026-09-11/vs_ctrl10M/critic_read_hp400.md …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_22_ladder_lambda095`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py; named by 21 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/lambda095_vs_ctrl10M_2026-09-11/README.md, designs/research_state/measurements/critic_ladder_reads/lambda095_vs_ctrl10M_2026-09-11/vs_ctrl10M/critic_read_hp400.json, designs/research_state/measurements/critic_ladder_reads/lambda095_vs_ctrl10M_2026-09-11/vs_ctrl10M/critic_read_hp400.md …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_23_ladder_rollout`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, designs/research_state/measurements/paired_refit_discrimination_2026-09-14/run_battery.sh; named by 21 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/rollout_vs_lambda_pair_2026-09-11/README.md, designs/research_state/measurements/critic_ladder_reads/rollout_vs_lambda_pair_2026-09-11/vs_ctrl10M/critic_read_hp400.json, designs/research_state/measurements/critic_ladder_reads/rollout_vs_lambda_pair_2026-09-11/vs_ctrl10M/critic_read_hp400.md …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_24_ladder_strata_b`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_ladder_reads/strata_b_vs_ctrl10M_2026-09-12/hp_strata_b.sh, designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, designs/research_state/measurements/repr_class_decode_strata_b_2026-09-12/run.sh …; named by 28 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/strata_b_vs_ctrl10M_2026-09-12/README.md, designs/research_state/measurements/critic_ladder_reads/strata_b_vs_ctrl10M_2026-09-12/hp400/vs_ctrl10M/cond_readout.json, designs/research_state/measurements/critic_ladder_reads/strata_b_vs_ctrl10M_2026-09-12/hp400/vs_ctrl10M/critic_read.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_25_ladder_vf15_b`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/critic_ladder_reads/vf15_b_vs_ctrl10M_2026-09-12/hp_vf15_b.sh, designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py; named by 24 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/vf15_b_vs_ctrl10M_2026-09-12/README.md, designs/research_state/measurements/critic_ladder_reads/vf15_b_vs_ctrl10M_2026-09-12/hp800/vs_ctrl10M/cond_readout.json, designs/research_state/measurements/critic_ladder_reads/vf15_b_vs_ctrl10M_2026-09-12/hp800/vs_ctrl10M/critic_read.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_26_ladder_ctrl10M_shaped`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py; named by 3 committed measurement artifact(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/PREDICTION.md, designs/research_state/measurements/ladder_strength_table_2026-09-12/README.md, designs/research_state/measurements/ladder_strength_table_2026-09-12/ladder_strength.json
- **`ai_v12_28_ladder_ent05`** — REFERENCED: loaded by 5 committed script(s): designs/research_state/measurements/critic_ladder_reads/ent05_vs_ctrl10M_2026-09-12/hp_ent05.sh, designs/research_state/measurements/flywheel_armS_reads_2026-09-14/scripts/entropy_read.py, designs/research_state/measurements/flywheel_pair_read_2026-09-15/scripts/pair_entropy_read.py …; named by 25 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/ent05_vs_ctrl10M_2026-09-12/README.md, designs/research_state/measurements/critic_ladder_reads/ent05_vs_ctrl10M_2026-09-12/hp800/vs_ctrl10M/cond_readout.json, designs/research_state/measurements/critic_ladder_reads/ent05_vs_ctrl10M_2026-09-12/hp800/vs_ctrl10M/critic_read.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v12_29_ladder_vf025`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/critic_ladder_reads/vf025_vs_ctrl10M_2026-09-13/hp_vf025.sh, designs/research_state/measurements/paired_refit_discrimination_2026-09-14/run_battery.sh; named by 22 committed measurement artifact(s): designs/research_state/measurements/critic_ladder_reads/vf025_vs_ctrl10M_2026-09-13/README.md, designs/research_state/measurements/critic_ladder_reads/vf025_vs_ctrl10M_2026-09-13/hp800/vs_ctrl10M/cond_readout.json, designs/research_state/measurements/critic_ladder_reads/vf025_vs_ctrl10M_2026-09-13/hp800/vs_ctrl10M/critic_read.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v13_01_flywheel_shaped`** — REFERENCED: loaded by 26 committed script(s): designs/research_state/measurements/elo_calibration_external_anchors_2026-09-14/cells.py, designs/research_state/measurements/elo_calibration_external_anchors_2026-09-14/fit_joint.py, designs/research_state/measurements/flywheel_armS_reads_2026-09-14/scripts/critic_levels.py …; named by 44 committed measurement artifact(s): designs/research_state/measurements/elo_calibration_external_anchors_2026-09-14/PREDICTION.md, designs/research_state/measurements/elo_calibration_external_anchors_2026-09-14/README.md, designs/research_state/measurements/elo_calibration_external_anchors_2026-09-14/games.jsonl …; a committed file / the ledger names a file the plan would delete
- **`ai_v13_02_flywheel_winprob`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v13_21_wcont_b
- **`ai_v13_03_fork`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/fork_arm_read_2026-09-16/run_battery.sh, designs/research_state/measurements/fork_arm_read_2026-09-16/run_forks.sh, designs/research_state/measurements/fork_arm_read_2026-09-16/run_guard.sh; named by 10 committed measurement artifact(s): designs/research_state/measurements/flywheel_pair_read_2026-09-15/README.md, designs/research_state/measurements/fork_arm_read_2026-09-16/PREDICTION.md, designs/research_state/measurements/fork_arm_read_2026-09-16/README.md …; a committed file / the ledger names a file the plan would delete
- **`ai_v13_04_flywheel_winprob_b`** — REFERENCED: loaded by 17 committed script(s): designs/research_state/measurements/flywheel_wb_floor_read_2026-09-18/scripts/floor_away_anchor.py, designs/research_state/measurements/flywheel_wb_floor_read_2026-09-18/scripts/floor_critic_levels.py, designs/research_state/measurements/flywheel_wb_floor_read_2026-09-18/scripts/floor_entropy_read.py …; named by 36 committed measurement artifact(s): designs/research_state/measurements/flywheel_wb_floor_read_2026-09-18/PREDICTION.md, designs/research_state/measurements/flywheel_wb_floor_read_2026-09-18/README.md, designs/research_state/measurements/flywheel_wb_floor_read_2026-09-18/out/away/games.jsonl …; a committed file / the ledger names a file the plan would delete
- **`ai_v13_05_exploit_big5starmie`** — REFERENCED: named in the ledger's last 1500 lines; loaded by 11 committed script(s): designs/research_state/measurements/exploiter_discrimination_2026-09-18/gen_tree.py, designs/research_state/measurements/exploiter_discrimination_2026-09-18/run_battery.sh, designs/research_state/measurements/exploiter_discrimination_2026-09-18/run_forks.sh …; named by 23 committed measurement artifact(s): designs/research_state/measurements/best_response_gap_meter_2026-09-22/README.md, designs/research_state/measurements/best_response_gap_meter_2026-09-22/archive_read.json, designs/research_state/measurements/best_response_gap_meter_2026-09-22/archive_read.md …; named by another run's model graph: ai_v13_07_fold1 (argv mention), ai_v13_07_fold1 (argv pool_source), ai_v13_07_fold1 (argv teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v13_06_exploit_ddtar_spikes`** — REFERENCED: named in the ledger's last 1500 lines; loaded by 7 committed script(s): designs/research_state/measurements/fold1_cont_read_2026-09-19/scripts/cont_slice_read.py, designs/research_state/measurements/fold1_read_2026-09-19/scripts/fold_slice_read.py, designs/research_state/measurements/fold1_read_2026-09-19/scripts/harness/run_slice.sh …; named by 16 committed measurement artifact(s): designs/research_state/measurements/best_response_gap_meter_2026-09-22/README.md, designs/research_state/measurements/best_response_gap_meter_2026-09-22/archive_read.json, designs/research_state/measurements/best_response_gap_meter_2026-09-22/archive_read.md …; named by another run's model graph: ai_v13_07_fold1 (argv mention), ai_v13_07_fold1 (argv pool_source), ai_v13_07_fold1 (argv teacher) …
- **`ai_v13_07_fold1`** — REFERENCED: named in the ledger's last 1500 lines; loaded by 13 committed script(s): designs/research_state/measurements/fold1_cont_read_2026-09-19/scripts/cont_run_rows.py, designs/research_state/measurements/fold1_cont_read_2026-09-19/scripts/cont_slice_read.py, designs/research_state/measurements/fold1_cont_read_2026-09-19/scripts/cont_untaught_delta.py …; named by 26 committed measurement artifact(s): designs/research_state/measurements/fold1_cont_read_2026-09-19/PREDICTION.md, designs/research_state/measurements/fold1_cont_read_2026-09-19/README.md, designs/research_state/measurements/fold1_cont_read_2026-09-19/out/cont_run_rows.json …; named by another run's model graph: ai_v13_08_fold1_cont (argv fork_parent), ai_v13_08_fold1_cont (fork_parent); a committed file / the ledger names a file the plan would delete
- **`ai_v13_08_fold1_cont`** — REFERENCED: named in the ledger's last 1500 lines; loaded by 10 committed script(s): designs/research_state/measurements/anchor_ab_continuation_2026-09-20/scripts/analyze.py, designs/research_state/measurements/anchor_ab_continuation_2026-09-20/scripts/run_lane.sh, designs/research_state/measurements/fold1_cont_read_2026-09-19/scripts/cont_run_rows.py …; named by 50 committed measurement artifact(s): designs/research_state/measurements/anchor_ab_continuation_2026-09-20/PREDICTION.md, designs/research_state/measurements/anchor_ab_continuation_2026-09-20/README.md, designs/research_state/measurements/anchor_ab_continuation_2026-09-20/out/F_SmallRL_away_s20260919/summary.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v13_09_wcont`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v13_12_plateau
- **`ai_v13_10_exploit_stall`** — REFERENCED: named in the ledger's last 1500 lines; loaded by 2 committed script(s): src/main/best_response_gap.py, src/main/best_response_gap_integration_test.py; named by 4 committed measurement artifact(s): designs/research_state/measurements/best_response_gap_meter_2026-09-22/README.md, designs/research_state/measurements/best_response_gap_meter_2026-09-22/archive_read.json, designs/research_state/measurements/best_response_gap_meter_2026-09-22/archive_read.md …
- **`ai_v13_11_split_lossoff`** — REFERENCED: named in the ledger's last 1500 lines; loaded by 5 committed script(s): designs/research_state/measurements/split_lossoff_read_2026-09-20/scripts/harness/run_slice.sh, designs/research_state/measurements/split_lossoff_read_2026-09-20/scripts/harness/run_untaught.sh, designs/research_state/measurements/split_lossoff_read_2026-09-20/scripts/render_tables.py …; named by 14 committed measurement artifact(s): designs/research_state/measurements/fold1_read_2026-09-19/PREDICTION.md, designs/research_state/measurements/ladder_refit_audit_2026-09-22/README.md, designs/research_state/measurements/ladder_refit_audit_2026-09-22/_all_runs.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v13_12_plateau`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v13_23_popr1_ctrl
- **`ai_v13_13_exploit5_offense`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v13_27_popr2_loop
- **`ai_v13_14_exploit5_balance`** — REFERENCED: named in the ledger's last 1500 lines; loaded by 3 committed script(s): src/main/best_response_gap.py, src/main/best_response_gap_integration_test.py, src/main/train/fork_lr_inherit_guard_test.py; named by 5 committed measurement artifact(s): designs/research_state/measurements/best_response_gap_meter_2026-09-22/README.md, designs/research_state/measurements/best_response_gap_meter_2026-09-22/archive_read.json, designs/research_state/measurements/best_response_gap_meter_2026-09-22/archive_read.md …
- **`ai_v13_15_exploit5_stall`** — REFERENCED: named in the ledger's last 1500 lines; loaded by 3 committed script(s): src/main/best_response_gap.py, src/main/best_response_gap_integration_test.py, src/main/train/fork_lr_inherit_guard_test.py; named by 5 committed measurement artifact(s): designs/research_state/measurements/best_response_gap_meter_2026-09-22/README.md, designs/research_state/measurements/best_response_gap_meter_2026-09-22/archive_read.json, designs/research_state/measurements/best_response_gap_meter_2026-09-22/archive_read.md …
- **`ai_v13_16_teach5_offense_dist`** — REFERENCED: named in the ledger's last 1500 lines; loaded by 5 committed script(s): designs/research_state/measurements/population_loop_r1_2026-09-23/launch_RB.sh, designs/research_state/measurements/population_loop_r1_2026-09-23/launch_RC.sh, designs/research_state/measurements/population_loop_r1_2026-09-23/scripts/build_argvs.py …; named by 14 committed measurement artifact(s): designs/research_state/measurements/hidose_teacher_2026-09-21/PREDICTION.md, designs/research_state/measurements/hidose_teacher_2026-09-21/out/dose.txt, designs/research_state/measurements/ladder_refit_audit_2026-09-22/README.md …; named by another run's model graph: ai_v13_16_teach5_offense_dist_ABANDONED_forklr2p8 (argv mention)
- **`ai_v13_16_teach5_offense_dist_ABANDONED_forklr2p8`** — REFERENCED: named in the ledger's last 1500 lines
- **`ai_v13_17_fold_k1`** — REFERENCED: named in the ledger's last 1500 lines; named by 6 committed measurement artifact(s): designs/research_state/measurements/hidose_teacher_2026-09-21/PREDICTION.md, designs/research_state/measurements/hidose_teacher_2026-09-21/README.md, designs/research_state/measurements/kladder_read_2026-09-21/PREDICTION.md …
- **`ai_v13_18_fold_k3`** — REFERENCED: named in the ledger's last 1500 lines; named by 1 committed measurement artifact(s): designs/research_state/measurements/kladder_read_2026-09-21/PREDICTION.md
- **`ai_v13_18_teach5_offense_hidose`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v13_27_popr2_loop
- **`ai_v13_19_fold_k11`** — REFERENCED: named in the ledger's last 1500 lines
- **`ai_v13_20_fold_k11_sharematched`** — REFERENCED: named in the ledger's last 1500 lines
- **`ai_v13_21_wcont_b`** — LIVE: training output written within 7 days (tb/, 2026-09-23) — an arm launched between ledger updates has no other signal
- **`ai_v13_22_popr1_loop`** — LIVE: training output written within 7 days (tb/, 2026-09-23) — an arm launched between ledger updates has no other signal
- **`ai_v13_23_popr1_ctrl`** — LIVE: training output written within 7 days (tb/, 2026-09-24) — an arm launched between ledger updates has no other signal
- **`ai_v13_24_popr1_read_loop`** — LIVE: training output written within 7 days (tb/, 2026-09-24) — an arm launched between ledger updates has no other signal
- **`ai_v13_25_popr1_read_ctrl`** — LIVE: training output written within 7 days (tb/, 2026-09-24) — an arm launched between ledger updates has no other signal
- **`ai_v13_26_popr0_exploit5_offense_s1002`** — LIVE: training output written within 7 days (tb/, 2026-09-24) — an arm launched between ledger updates has no other signal
- **`ai_v13_27_popr2_loop`** — LIVE: training output written within 7 days (tb/, 2026-09-24) — an arm launched between ledger updates has no other signal
- **`ai_v13_28_popr2_ctrl`** — LIVE: training output written within 7 days (tb/, 2026-09-25) — an arm launched between ledger updates has no other signal
- **`ai_v13_29_popr2_read_loop`** — LIVE: training output written within 7 days (tb/, 2026-09-25) — an arm launched between ledger updates has no other signal
- **`ai_v13_30_popr2_read_ctrl`** — LIVE: training output written within 7 days (tb/, 2026-09-25) — an arm launched between ledger updates has no other signal
- **`ai_v13_31_popr1_read_loop_ext`** — LIVE: training output written within 7 days (tb/, 2026-09-25) — an arm launched between ledger updates has no other signal
- **`ai_v13_32_popr1_read_ctrl_ext`** — LIVE: training output written within 7 days (tb/, 2026-09-25) — an arm launched between ledger updates has no other signal
- **`ai_v13_33_core_burnin`** — LIVE: training output written within 7 days (tb/, 2026-09-26) — an arm launched between ledger updates has no other signal
- **`ai_v14_01_base`** — LIVE: training output written within 7 days (tb/, 2026-09-27) — an arm launched between ledger updates has no other signal
- **`ai_v14_02_lbat_ctrl`** — LIVE: training output written within 7 days (tb/, 2026-09-28) — an arm launched between ledger updates has no other signal
- **`ai_v14_03_lbat_e5`** — LIVE: training output written within 7 days (tb/, 2026-09-28) — an arm launched between ledger updates has no other signal
- **`ai_v14_04b_lbat_t32`** — LIVE: training output written within 7 days (tb/, 2026-09-28) — an arm launched between ledger updates has no other signal
- **`ai_v14_05_lbat_l95`** — LIVE: training output written within 7 days (tb/, 2026-09-28) — an arm launched between ledger updates has no other signal
- **`ai_v14_06_lbat_ctrl_fix`** — LIVE: training output written within 7 days (tb/, 2026-09-28) — an arm launched between ledger updates has no other signal
- **`ai_v14_07_g0p_k2`** — LIVE: training output written within 7 days (tb/, 2026-09-29) — an arm launched between ledger updates has no other signal
- **`ai_v14_08_g0p_k3`** — LIVE: training output written within 7 days (tb/, 2026-09-29) — an arm launched between ledger updates has no other signal
- **`ai_v14_09_r0_offense_a`** — LIVE: training output written within 7 days (tb/, 2026-09-29) — an arm launched between ledger updates has no other signal
- **`ai_v5_10_tail1_23_0611`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v5_11_tail2_53m_0611`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v5_12_bias_05_N_0612`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v5_13_shape_pbrs_43m_0612`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: ai_v6_01_belief_53m_0613 (argv mention), ai_v6_01_belief_53m_0613 (argv pool_source)
- **`ai_v5_2_native_selfplay_50m_0606`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v5_3_vf_coef_clip_50m_0606`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v5_4_pbrs_opp_threat_50m_0607`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v5_5_popart_50m_0607`** — REFERENCED: loaded by 1 committed script(s): src/main/launcher_app_test.py; named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: ai_v5_6_stable_70m_0608 (argv mention), ai_v5_6_stable_70m_0608 (argv pool_source), ai_v5_7_switch_bias_41m_0609 (argv mention) …
- **`ai_v5_6_stable_70m_0608`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: ai_v5_7_switch_bias_41m_0609 (argv mention), ai_v5_7_switch_bias_41m_0609 (argv pool_source)
- **`ai_v5_7_switch_bias_41m_0609`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v5_8_split_inc_dmg_38m_0610`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: ai_v5_10_tail1_23_0611 (argv mention), ai_v5_10_tail1_23_0611 (argv pool_source), ai_v5_11_tail2_53m_0611 (argv mention) …
- **`ai_v5_9_attend_unrevealed_56m_0610`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: ai_v5_10_tail1_23_0611 (argv mention), ai_v5_10_tail1_23_0611 (argv pool_source), ai_v5_11_tail2_53m_0611 (argv mention) …
- **`ai_v6_01_belief_53m_0613`** — REFERENCED: named by 2 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md, designs/research_state/measurements/tech_debt_census_2026-09-07/worktrees.md
- **`ai_v6_02_belief_lat_16m_0614`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v6_03_win_pred_N_0614`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v6_04_unified_all_half_batch_N_0616`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v6_04_unified_inc_N_0615`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v6_06_unified_all_N_0616`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v6_07_unified_topk_N_0616`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v6_08_unmasked_floor_N_0617`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v6_09_dmg_reattend_N_0617`** — REFERENCED: named by 2 committed measurement artifact(s): designs/research_state/measurements/docs_staleness_census_2026-09-08.md, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v6_10_unified_obs_0618`** — REFERENCED: named by 2 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v6_11_typed_hp_0619`** — REFERENCED: named by 2 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v6_11_unified_obs_fixed_0618`** — REFERENCED: named by 3 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md, designs/research_state/measurements/tech_debt_census_2026-09-07/worktrees.md
- **`ai_v6_13_outgoing_dmg_0620`** — REFERENCED: named in the ledger's last 1500 lines; loaded by 3 committed script(s): src/main/launcher_test.py, src/main/run_name_test.py, src/main/train/parser/eval_subprocess.py; named by 2 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: ai_v6_13_outgoing_dmg_0620_exp_v1 (argv fork_parent), ai_v6_13_outgoing_dmg_0620_exp_v1 (fork_parent), ai_v6_13_outgoing_dmg_0620_exploiter_v1 (argv fork_parent) …
- **`ai_v6_13_outgoing_dmg_0620_exp_v1`** — REFERENCED: named in the ledger's last 1500 lines; loaded by 1 committed script(s): src/main/launcher_test.py; named by 2 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v6_13_outgoing_dmg_0620_exploiter_v1`** — REFERENCED: named by 2 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v6_13_outgoing_dmg_0620_exploiter_v2`** — REFERENCED: named by 2 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v7_01_teacher_0626`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: ai_v7_01_teacher_0626_oom1 (argv mention), ai_v7_01_teacher_0626_oom1 (argv mention)
- **`ai_v7_01_teacher_0626_oom1`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v7_02_critic_shape_0627`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: ai_v7_04_opd_selfdistill_0702 (argv mention), ai_v7_05_tss_specialist_0703 (argv mention), ai_v7_05_tss_specialist_0703 (argv pool_source) …
- **`ai_v7_03_belief_shape_0630`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v7_04_opd_selfdistill_0702`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v7_05_tss_specialist_0703`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: ai_v7_05_tss_specialist_0703_aborted_noeval (argv mention)
- **`ai_v7_05_tss_specialist_0703_aborted_noeval`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v7_06_tss_temp_anneal_0706`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v7_07_tss_temp_ratchet_0707`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v7_08_tss_bots_0707`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v7_09_tss_bots_pubval_0708`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v7_10_tss_exploiter_fixed_0709`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: ai_v7_14_league_capstone_0712 (argv mention), ai_v7_14_league_capstone_0712 (argv pool_source), ai_v7_16_distill_tss_mvp_0715 (argv mention) …
- **`ai_v7_11_tss_exploiter_nopubval`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v7_12_trap_exploiter_0711`** — REFERENCED: named by 2 committed measurement artifact(s): designs/research_state/measurements/plain_training_robbery_2026-08-31.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: ai_v7_14_league_capstone_0712 (argv mention), ai_v7_14_league_capstone_0712 (argv pool_source), ai_v7_18_distill_4teacher_0716 (argv mention) …
- **`ai_v7_13_cmpass_exploiter_0711`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: ai_v7_14_league_capstone_0712 (argv mention), ai_v7_14_league_capstone_0712 (argv pool_source), ai_v7_18_distill_4teacher_0716 (argv mention) …
- **`ai_v7_14_league_capstone_0712`** — REFERENCED: named by 2 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md, designs/research_state/measurements/tech_debt_census_2026-09-07/worktrees.md; named by another run's model graph: ai_v7_15_tss_exploiter_vs14_0713 (argv mention), ai_v7_15_tss_exploiter_vs14_0713 (argv pool_source), ai_v7_16_distill_tss_mvp_0715 (argv fork_parent) …
- **`ai_v7_15_tss_exploiter_vs14_0713`** — REFERENCED: named by 2 committed measurement artifact(s): designs/research_state/measurements/plain_training_robbery_2026-08-31.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: ai_v8_04_distill_4teacher_0722 (argv mention), ai_v8_04_distill_4teacher_0722 (argv teacher), ai_v8_04_distill_4teacher_0722 (teacher)
- **`ai_v7_16_distill_tss_mvp_0715`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v7_17_stall_exploiter_0715`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: ai_v7_18_distill_4teacher_0716 (argv mention), ai_v7_18_distill_4teacher_0716 (teacher), ai_v7_19_combined_0716 (argv mention) …
- **`ai_v7_18_distill_4teacher_0716`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: ai_v7_19_combined_0716 (argv fork_parent), ai_v7_19_combined_0716 (fork_parent)
- **`ai_v7_19_combined_0716`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v7_20_valuedistill_SMOKE`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v7_20_valuedistill_ab_0717`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v7_21_fitnet_valuefeat_ab_0717`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v8_01_zarch_film_0717`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.py, designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/entropy_forensics.py, src/agents/training/tb_inherit.py …; named by 6 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/README.md, designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/dose.json, designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json …; named by another run's model graph: ai_v8_02_zarch_teampfsp_0718 (argv fork_parent), ai_v8_02_zarch_teampfsp_0718 (fork_parent), ai_v8_03_zarch_control_0718 (argv fork_parent) …
- **`ai_v8_02_zarch_teampfsp_0718`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v8_03_zarch_control_0718`** — REFERENCED: loaded by 16 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/v8_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/v8_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/sharing_kernel/gen_states.py …; named by 31 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/v8_era_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/v8_era_n9.json …; named by another run's model graph: ai_v8_04_distill_4teacher_0722 (argv fork_parent), ai_v8_04_distill_4teacher_0722 (fork_parent)
- **`ai_v8_04_distill_4teacher_0722`** — REFERENCED: NAMED in designs/baselines.json (gen3_baselines_registry_v1): final_model_interrupted.zip — a baseline name must still resolve a year from now; a committed file / the ledger names a file the plan would delete
- **`ai_v8_05_semistall564_exploiter_0722`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v8_06_semistall_3team_exploiter_0722`** — REFERENCED: loaded by 6 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/v8_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/resolve_teachers.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/v8_era_locality_v2.py …; named by 17 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/resolved_teachers.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/resolved_teachers.log …; named by another run's model graph: ai_v8_14_distill3_0725 (argv mention), ai_v8_14_distill3_0725 (argv pool_source), ai_v8_14_distill3_0725 (argv teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v8_07_semistall564_scratch_0722`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v8_08_defensive_6team_exploiter_0723`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v8_09_pool10_exploiter_0723`** — REFERENCED: loaded by 6 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/v8_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/resolve_teachers.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/v8_era_locality_v2.py …; named by 19 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/resolved_teachers.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/resolved_teachers.log …; named by another run's model graph: ai_v8_14_distill3_0725 (argv mention), ai_v8_14_distill3_0725 (argv pool_source), ai_v8_14_distill3_0725 (argv teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v8_10_offense20_exploiter_0724`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v8_11_offense10_exploiter_0724`** — REFERENCED: named by 2 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md, designs/research_state/measurements/tech_debt_census_2026-09-07/worktrees.md
- **`ai_v8_12_defensive20_exploiter_0724`** — REFERENCED: named by 2 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md, designs/research_state/measurements/tech_debt_census_2026-09-07/worktrees.md
- **`ai_v8_13_defensive10_exploiter_0725`** — REFERENCED: loaded by 6 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/v8_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/resolve_teachers.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/v8_era_locality_v2.py …; named by 17 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/resolved_teachers.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/resolved_teachers.log …; named by another run's model graph: ai_v8_14_distill3_0725 (argv mention), ai_v8_14_distill3_0725 (argv pool_source), ai_v8_14_distill3_0725 (argv teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v8_14_distill3_0725`** — REFERENCED: NAMED in designs/baselines.json (gen3_baselines_registry_v1): final_model_interrupted.zip — a baseline name must still resolve a year from now; a committed file / the ledger names a file the plan would delete
- **`ai_v8_15_retention_A_frozen_0726`** — REFERENCED: named by 2 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md, designs/research_state/measurements/v8_redistribution_pfsp_2026-08-30.md
- **`ai_v8_16_def20_lut_0726`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v8_17_rand20_nolut_0726`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v8_18_rand20_lut_0726`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v8_19_def20_lut_zeroinit_0727`** — REFERENCED: named by 2 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md, designs/research_state/measurements/tech_debt_census_2026-09-07/worktrees.md
- **`ai_v8_20_rand10_nolut_0727`** — REFERENCED: named by 2 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md, designs/research_state/measurements/tech_debt_census_2026-09-07/worktrees.md
- **`ai_v9_09_gen8_beliefs_threat_inject_0811`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/ladder_refit_audit_2026-09-22/README.md, designs/research_state/measurements/ladder_refit_audit_2026-09-22/_all_runs.json …
- **`ai_v9_100_R5F08_0831`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 29 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n9.json …; named by another run's model graph: .dryrun_K6A_1788581936 (argv mention), .dryrun_K6A_1788581936 (argv teacher), .dryrun_K6A_1788581936 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_101_R5F09_0831`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/r5_fleet_teams.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`ai_v9_102_R5F10_0831`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 29 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n9.json …; named by another run's model graph: .dryrun_K6A_1788581936 (argv mention), .dryrun_K6A_1788581936 (argv teacher), .dryrun_K6A_1788581936 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_103_R5F11_0831`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/r5_fleet_teams.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`ai_v9_104_R5F12_0831`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 29 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n9.json …; named by another run's model graph: .dryrun_K6A_1788581936 (argv mention), .dryrun_K6A_1788581936 (argv teacher), .dryrun_K6A_1788581936 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_105_R5F13_0831`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/r5_fleet_teams.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`ai_v9_106_R5F14_0831`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 30 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log …; named by another run's model graph: .dryrun_K6A_1788581936 (argv mention), .dryrun_K6A_1788581936 (teacher), ai_v9_134_R5FUND14_0901 (argv fork_parent) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_107_R5F15_0831`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/r5_fleet_teams.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`ai_v9_108_R5F16_0831`** — REFERENCED: named by 5 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/r5_fleet_teams.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`ai_v9_109_R5F17_0831`** — REFERENCED: named by 5 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/r5_fleet_teams.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`ai_v9_10_gen9_intent_distcritic_0813`** — REFERENCED: loaded by 1 committed script(s): src/agents/model/intent_move_cell_test.py; named by 7 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/ladder_refit_audit_2026-09-22/README.md …
- **`ai_v9_110_R5F18_0831`** — REFERENCED: named by 5 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/r5_fleet_teams.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`ai_v9_111_R5F19_0831`** — REFERENCED: named by 5 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/r5_fleet_teams.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`ai_v9_11_gen10_intentfull_compiled_0814`** — REFERENCED: named by 3 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`ai_v9_120_R5FUND00_0901`** — REFERENCED: loaded by 7 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 28 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log …; named by another run's model graph: ai_v9_160_TCFUNDA_0903 (argv mention), ai_v9_160_TCFUNDA_0903 (argv teacher), ai_v9_160_TCFUNDA_0903 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_122_R5FUND02_0901`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 26 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n9.json …; named by another run's model graph: ai_v9_160_TCFUNDA_0903 (argv mention), ai_v9_160_TCFUNDA_0903 (argv teacher), ai_v9_160_TCFUNDA_0903 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_124_R5FUND04_0901`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 26 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n9.json …; named by another run's model graph: ai_v9_160_TCFUNDA_0903 (argv mention), ai_v9_160_TCFUNDA_0903 (argv teacher), ai_v9_160_TCFUNDA_0903 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_126_R5FUND06_0901`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 26 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n9.json …; named by another run's model graph: ai_v9_160_TCFUNDA_0903 (argv mention), ai_v9_160_TCFUNDA_0903 (argv teacher), ai_v9_160_TCFUNDA_0903 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_128_R5FUND08_0901`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 26 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n9.json …; named by another run's model graph: ai_v9_160_TCFUNDA_0903 (argv mention), ai_v9_160_TCFUNDA_0903 (argv teacher), ai_v9_160_TCFUNDA_0903 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_12_gen10_t0prior_0814`** — REFERENCED: named by 8 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/gen11_label_only_winprob_verdict.json …
- **`ai_v9_130_R5FUND10_0901`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 26 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n9.json …; named by another run's model graph: ai_v9_160_TCFUNDA_0903 (argv mention), ai_v9_160_TCFUNDA_0903 (argv teacher), ai_v9_160_TCFUNDA_0903 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_132_R5FUND12_0901`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 26 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n9.json …; named by another run's model graph: ai_v9_160_TCFUNDA_0903 (argv mention), ai_v9_160_TCFUNDA_0903 (teacher), ai_v9_161_TCFUNDB_0903 (argv mention) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_134_R5FUND14_0901`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 26 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n9.json …; named by another run's model graph: ai_v9_160_TCFUNDA_0903 (argv mention), ai_v9_160_TCFUNDA_0903 (teacher), ai_v9_161_TCFUNDB_0903 (argv mention) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_13_gen11_labelonly_winprob_0815`** — REFERENCED: named by 9 committed measurement artifact(s): designs/research_state/measurements/ai_v9_14_gen12_h_entitypool_shaping_0816_endofrun.json, designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …
- **`ai_v9_140_B2_0901`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/offline_collateral_kl.py; named by 14 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/displacement.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_141_C1_0901`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/offline_collateral_kl.py; named by 15 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/displacement.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_142_N1_0901`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/teacher_content_2x2_2026-09-04/tc_readout.py; named by 9 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_143_N2_0901`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/teacher_content_2x2_2026-09-04/tc_readout.py; named by 13 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_14_gen12_h_entitypool_shaping_0816`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/obs_conditioning_probe.py; named by 15 committed measurement artifact(s): designs/research_state/measurements/ai_v9_14_gen12_h_entitypool_shaping_0816_endofrun.json, designs/research_state/measurements/ai_v9_14_gen12_h_entitypool_shaping_0816_endofrun.md, designs/research_state/measurements/ai_v9_15_gen13_hb_events_stack_0817_endofrun.json …
- **`ai_v9_150_R4DOSE12_0901`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/offline_collateral_kl.py; named by 14 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/displacement.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: .aborted_R4DOSE12_nometa_1401 (argv mention), .aborted_R4DOSE12_poolless_1355 (argv mention); a committed file / the ledger names a file the plan would delete
- **`ai_v9_151_R4DOSE6_0901`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/offline_collateral_kl.py; named by 14 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/displacement.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_152_R4DOSE3_0901`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/offline_collateral_kl.py; named by 10 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/displacement.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_15_gen13_hb_events_stack_0817`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/gen13_stall_coverage.py, designs/research_state/measurements/gen14_paired_bt_refit.py, designs/research_state/measurements/obs_conditioning_probe.py; named by 18 committed measurement artifact(s): designs/research_state/measurements/ai_v9_15_gen13_hb_events_stack_0817_endofrun.json, designs/research_state/measurements/ai_v9_15_gen13_hb_events_stack_0817_endofrun.md, designs/research_state/measurements/ai_v9_16_gen14_framedel_v91_0817_endofrun.json …; named by another run's model graph: RETIRED_c5fork_control_gen13base_0817 (argv fork_parent), RETIRED_c5fork_control_gen13base_0817 (argv mention), RETIRED_c5fork_control_gen13base_0817 (argv mention) …
- **`ai_v9_160_TCFUNDA_0903`** — REFERENCED: loaded by 9 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_competence/inventory.py, designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/sharing_kernel/control_funded_vs_unfunded.py …; named by 18 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_competence/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_competence/inventory.json, designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/displacement.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_161_TCFUNDB_0903`** — REFERENCED: loaded by 7 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/sharing_kernel/control_funded_vs_unfunded.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py …; named by 16 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/displacement.json, designs/research_state/measurements/arch_transfer_2026-09-05/sharing_kernel/control_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_162_TCUNFA_0903`** — REFERENCED: loaded by 13 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_competence/inventory.py, designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/sharing_kernel/control_funded_vs_unfunded.py …; named by 19 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_competence/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_competence/inventory.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_163_TCUNFB_0903`** — REFERENCED: loaded by 8 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/sharing_kernel/control_funded_vs_unfunded.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py …; named by 16 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/displacement.json, designs/research_state/measurements/arch_transfer_2026-09-05/sharing_kernel/control_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_16_gen14_framedel_v91_0817`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/gen13_stall_coverage.py, designs/research_state/measurements/gen14_paired_bt_refit.py, designs/research_state/measurements/obs_conditioning_probe.py; named by 16 committed measurement artifact(s): designs/research_state/measurements/README.md, designs/research_state/measurements/ai_v9_16_gen14_framedel_v91_0817_endofrun.json, designs/research_state/measurements/ai_v9_16_gen14_framedel_v91_0817_endofrun.md …; named by another run's model graph: DISCARDED_tdaux_control_n16_0818 (argv fork_parent), DISCARDED_tdaux_control_n16_0818 (argv mention), DISCARDED_tdaux_control_n16_0818 (fork_parent) …
- **`ai_v9_170_TCUNFK6A_0904`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/teacher_content_2x2_2026-09-04/k6_readout.py, designs/research_state/measurements/teacher_content_2x2_2026-09-04/taught_readout.py; named by 14 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …; named by another run's model graph: .dryrun_K6A_1788581936 (argv mention); a committed file / the ledger names a file the plan would delete
- **`ai_v9_171_TCUNFK6B_0904`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/teacher_content_2x2_2026-09-04/k6_readout.py, designs/research_state/measurements/teacher_content_2x2_2026-09-04/taught_readout.py; named by 14 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_172_G1SHORT_0905`** — REFERENCED: named by 10 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_competence/README.md, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`ai_v9_17_tdaux_control_0818`** — REFERENCED: named by 3 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md; named by another run's model graph: DISCARDED_tdaux_control_n16_0818 (argv mention)
- **`ai_v9_17_tdaux_lam1_0818`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/ladder_refit_audit_2026-09-22/README.md …
- **`ai_v9_17_tdaux_lam3_0818`** — REFERENCED: loaded by 1 committed script(s): src/agents/training/poke_env_gaps/faint_attribution_fuzz_test.py; named by 5 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/ladder_refit_audit_2026-09-22/README.md, designs/research_state/measurements/ladder_refit_audit_2026-09-22/_all_runs.json …
- **`ai_v9_18_gen15_v8rewards_0818`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/obs_conditioning_probe.py, src/main/prober/loops.py; named by 16 committed measurement artifact(s): designs/research_state/measurements/ai_v9_18_gen15_v8rewards_0818_endofrun.json, designs/research_state/measurements/ai_v9_18_gen15_v8rewards_0818_endofrun.md, designs/research_state/measurements/ai_v9_19_gen16_mechanics_0819_endofrun.json …
- **`ai_v9_195_G5PLAINA_0906`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py, src/main/untaught_meter.py; named by 17 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift_gen.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …
- **`ai_v9_196_G5PLAINB_0906`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py, src/main/untaught_meter.py; named by 17 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift_gen.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …
- **`ai_v9_197_G5PLAINC_0906`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py, src/main/untaught_meter.py; named by 17 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift_gen.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …
- **`ai_v9_19_gen16_mechanics_0819`** — REFERENCED: named by 13 committed measurement artifact(s): designs/research_state/measurements/ai_v9_19_gen16_mechanics_0819_endofrun.json, designs/research_state/measurements/ai_v9_19_gen16_mechanics_0819_endofrun.md, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …; named by another run's model graph: ai_v9_20_tdaux_rung2_lam00_0820 (argv fork_parent), ai_v9_20_tdaux_rung2_lam00_0820 (argv mention), ai_v9_20_tdaux_rung2_lam00_0820 (fork_parent) …
- **`ai_v9_20_tdaux_rung2_lam00_0820`** — REFERENCED: named by 10 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_20_tdaux_rung2_lam10_0820`** — REFERENCED: named by 10 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_20_tdaux_rung2_lam30_0820`** — REFERENCED: named by 10 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_21_gen17_pfspoff_0820`** — REFERENCED: NAMED in designs/baselines.json (gen3_baselines_registry_v1): final_model.zip — a baseline name must still resolve a year from now
- **`ai_v9_22_E1_substrate_on_0821`** — REFERENCED: named by 9 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …; named by another run's model graph: ai_v9_25_E4_baitbot_0822 (argv fork_parent), ai_v9_25_E4_baitbot_0822 (fork_parent), ai_v9_26_baitent_probe_0823 (argv fork_parent) …
- **`ai_v9_23_E2_substrate_on_0822`** — REFERENCED: named by 9 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_24_E3_substrate_on_0822`** — REFERENCED: named by 9 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_25_E4_baitbot_0822`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/maturity_harm_trend.py; named by 14 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …; named by another run's model graph: ai_v9_27_extremedial_probe_0823 (argv fork_parent), ai_v9_27_extremedial_probe_0823 (argv mention), ai_v9_27_extremedial_probe_0823 (fork_parent)
- **`ai_v9_26_baitent_probe_0823`** — REFERENCED: REVIEW HOLD (partial) — the capacity baseline IS banked (designs/research_state/capacity_battery.md:153ff), but the P2 bait-entropy per-leg result (boost_eff 3.0, flagged 5.9%, B1 0.056 -> 0.229, leg-vs-leg z=-2.55, ledger.md:3722) is in no committed artifact, and the Baton Pass GIGO reproducer decodes loss_s0_003_states.npz from this run's traces (ledger.md:3595). ladder_readiness.md:269 also loads its legB_final_model.zip.
- **`ai_v9_27_extremedial_probe_0823`** — REFERENCED: loaded by 2 committed script(s): src/agents/training/exploiter_ladder.py, src/agents/training/exploiter_ladder_test.py; named by 8 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_29_rev1_0823`** — REFERENCED: NAMED in designs/baselines.json (gen3_baselines_registry_v1): final_model.zip, snapshots/model_config.json, snapshots/snapshot_000024000000.zip — a baseline name must still resolve a year from now; a committed file / the ledger names a file the plan would delete
- **`ai_v9_30_rev1_exploit_0824`** — REFERENCED: named by 5 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_31_tock1_k4_0824`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/teacher_sharpness_probe.py; named by 11 committed measurement artifact(s): designs/research_state/measurements/ai_v9_34_tick1_0824_endofrun.json, designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.json, designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.md …; named by another run's model graph: ai_v9_34_tick1_0824 (argv mention), ai_v9_34_tick1_0824 (argv pool_source), ai_v9_34_tick1_0824 (argv teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_32_tock1b_rain_0824`** — REFERENCED: named by 9 committed measurement artifact(s): designs/research_state/measurements/ai_v9_34_tick1_0824_endofrun.json, designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.json, designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.md …; named by another run's model graph: ai_v9_34_tick1_0824 (argv mention), ai_v9_34_tick1_0824 (argv pool_source), ai_v9_34_tick1_0824 (argv teacher) …
- **`ai_v9_34_tick1_0824`** — REFERENCED: named by 17 committed measurement artifact(s): designs/research_state/measurements/README.md, designs/research_state/measurements/ai_v9_34_tick1_0824_endofrun.json, designs/research_state/measurements/ai_v9_34_tick1_0824_endofrun.md …; named by another run's model graph: ai_v9_35_tick1_exploit_0824 (argv fork_parent), ai_v9_35_tick1_exploit_0824 (argv mention), ai_v9_35_tick1_exploit_0824 (argv pool_source) …
- **`ai_v9_35_tick1_exploit_0824`** — REFERENCED: named by 5 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_36_tock1c_q6_0824`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/teacher_sharpness_probe.py; named by 9 committed measurement artifact(s): designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.json, designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.md, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …; named by another run's model graph: ai_v9_42_fdE_single_0825 (argv mention), ai_v9_42_fdE_single_0825 (argv teacher), ai_v9_42_fdE_single_0825 (teacher)
- **`ai_v9_37_tick1_dosext_0825`** — REFERENCED: named by 12 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_38_fdA_coef03_0825`** — REFERENCED: named by 10 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_39_fdB_lossonly_0825`** — REFERENCED: named by 10 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_40_fdC_ecology_0825`** — REFERENCED: named by 10 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_42_fdE_single_0825`** — REFERENCED: named by 10 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_44_tock2_v8shape_0825`** — REFERENCED: named by 8 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/pilot_T2_n300.json, designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.json, designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.md …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_45_fdF_p1_0826`** — REFERENCED: REVIEW HOLD — the NUMBERS are banked (designs/ai_v10/design_advantage_gated_distillation.md:459-467 carries the entropy 0.892 -> 1.354 dissolution and the subtraction rule), so this is not a data dependency; it is held because the ledger records an explicit owner decision to preserve it as the entropy-dissolution SPECIMEN (ledger.md:4937).
- **`ai_v9_48_G1_action_0826`** — REFERENCED: REVIEW HOLD — the program's first POSITIVE distill arm (pooled +0.0398 [+0.016,+0.064] z=+3.29; G2-fdB +0.0762 z=+6.01, ledger.md:4943ff). NO committed artifact carries the per-arm numbers: fold_capacity_telemetry.md has fdA/fdB/fdC/fdE rows and no G1/G2 row, and no ai_v9_48_*_endofrun.json exists — the claim rests on this run's eval_results.jsonl + eval_traces. Bank an endofrun artifact and this hold can be released.
- **`ai_v9_49_G2_advgate_0826`** — REFERENCED: named by 9 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_50_fdF_p1c_0826`** — REFERENCED: named by 11 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/README.md, designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …; named by another run's model graph: ai_v9_51_fdF_p2c_0826 (argv fork_parent), ai_v9_51_fdF_p2c_0826 (argv mention), ai_v9_51_fdF_p2c_0826 (fork_parent)
- **`ai_v9_51_fdF_p2c_0826`** — REFERENCED: named in the ledger's last 1500 lines; named by 9 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_52_G1p_matched_0826`** — REFERENCED: named by 9 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_53_R2F5a_0826`** — REFERENCED: loaded by 8 committed script(s): designs/research_state/measurements/distillability_index_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py …; named by 40 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_rev2.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_rev2.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_58_R2CTRL_0827 (argv mention), ai_v9_58_R2CTRL_0827 (argv teacher), ai_v9_58_R2CTRL_0827 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_54_R2F5b_0826`** — REFERENCED: loaded by 7 committed script(s): designs/research_state/measurements/distillability_index_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py …; named by 27 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_rev2.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_rev2.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_58_R2CTRL_0827 (argv mention), ai_v9_58_R2CTRL_0827 (argv teacher), ai_v9_58_R2CTRL_0827 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_55_R2F5c_0826`** — REFERENCED: loaded by 6 committed script(s): designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/per_team_gradient_geometry_2026-08-28/probeF_build_states.py …; named by 24 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_rev2.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_rev2.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_58_R2CTRL_0827 (argv mention), ai_v9_58_R2CTRL_0827 (argv teacher), ai_v9_58_R2CTRL_0827 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_56_R2F5d_0826`** — REFERENCED: loaded by 6 committed script(s): designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/per_team_gradient_geometry_2026-08-28/probeF_build_states.py …; named by 23 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_rev2.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_rev2.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_58_R2CTRL_0827 (argv mention), ai_v9_58_R2CTRL_0827 (argv teacher), ai_v9_58_R2CTRL_0827 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_57_R2F5e_0826`** — REFERENCED: loaded by 6 committed script(s): designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/per_team_gradient_geometry_2026-08-28/probeF_build_states.py …; named by 23 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_rev2.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_rev2.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_58_R2CTRL_0827 (argv mention), ai_v9_58_R2CTRL_0827 (argv teacher), ai_v9_58_R2CTRL_0827 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_58_R2CTRL_0827`** — REFERENCED: named in the ledger's last 1500 lines; loaded by 7 committed script(s): designs/ai_v12/team_slate_build.py, designs/research_state/measurements/plain_training_robbery.py, designs/research_state/measurements/representational_richness_transfer_forward.py …; named by 25 committed measurement artifact(s): designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.json, designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.md, designs/research_state/measurements/dark_knowledge_decomposition_2026-08-28.json …
- **`ai_v9_59_R2ACTION_0827`** — REFERENCED: NAMED in designs/baselines.json (gen3_baselines_registry_v1): final_model.zip — a baseline name must still resolve a year from now; a committed file / the ledger names a file the plan would delete
- **`ai_v9_60_R2TOPK_0827`** — REFERENCED: named by 15 committed measurement artifact(s): designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.json, designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.md, designs/research_state/measurements/dark_knowledge_decomposition_2026-08-28.json …
- **`ai_v9_61_R2KL_0827`** — REFERENCED: named by 15 committed measurement artifact(s): designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.json, designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.md, designs/research_state/measurements/dark_knowledge_decomposition_2026-08-28.json …
- **`ai_v9_62_R2PLAIN_0827`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/plain_training_robbery.py, designs/research_state/measurements/representational_richness_transfer_forward.py, designs/research_state/measurements/representational_richness_transfer_locus.py; named by 25 committed measurement artifact(s): designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.json, designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.md, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …
- **`ai_v9_63_R3F6a_0828`** — REFERENCED: loaded by 5 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py …; named by 17 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_70_R3ACTION_0828 (argv mention), ai_v9_70_R3ACTION_0828 (argv teacher), ai_v9_70_R3ACTION_0828 (teacher) …
- **`ai_v9_64_R3F6b_0828`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py …; named by 14 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_70_R3ACTION_0828 (argv mention), ai_v9_70_R3ACTION_0828 (argv teacher), ai_v9_70_R3ACTION_0828 (teacher) …
- **`ai_v9_65_R3F6c_0828`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py …; named by 14 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_70_R3ACTION_0828 (argv mention), ai_v9_70_R3ACTION_0828 (argv teacher), ai_v9_70_R3ACTION_0828 (teacher) …
- **`ai_v9_66_R3F6d_0828`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py …; named by 14 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_70_R3ACTION_0828 (argv mention), ai_v9_70_R3ACTION_0828 (argv teacher), ai_v9_70_R3ACTION_0828 (teacher) …
- **`ai_v9_67_R3F6e_0828`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py …; named by 13 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_70_R3ACTION_0828 (argv mention), ai_v9_70_R3ACTION_0828 (argv teacher), ai_v9_70_R3ACTION_0828 (teacher) …
- **`ai_v9_68_R3F6f_0828`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py …; named by 28 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_70_R3ACTION_0828 (argv mention), ai_v9_70_R3ACTION_0828 (argv teacher), ai_v9_70_R3ACTION_0828 (teacher) …
- **`ai_v9_69_R3F6CURR_0828`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/rev3_untaught_pulldown.py; named by 6 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_70_R3ACTION_0828`** — REFERENCED: loaded by 12 committed script(s): designs/ai_v12/team_slate_build.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/resolve_sets.py …; named by 32 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json, designs/research_state/measurements/axis_split_inputs/cov_R3ACTION.json, designs/research_state/measurements/axis_split_inputs/pilot_R3ACTION_n300.json …
- **`ai_v9_71_R3ACTIONHI_0828`** — REFERENCED: loaded by 1 committed script(s): designs/ai_v12/team_slate_build.py; named by 12 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/cov_R3ACTIONHI.json, designs/research_state/measurements/axis_split_inputs/pilot_R3ACTIONHI_n300.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …
- **`ai_v9_72_R3SELF_0828`** — REFERENCED: loaded by 4 committed script(s): designs/ai_v12/team_slate_build.py, designs/research_state/measurements/plain_training_robbery.py, designs/research_state/measurements/starmie_ood_control_traces.py …; named by 17 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/cov_R3SELF.json, designs/research_state/measurements/axis_split_inputs/pilot_R3SELF_n300.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …
- **`ai_v9_73_R4S3a_0829`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/lr_licensing_probe.py; named by 37 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: .aborted_R4DOSE12_nometa_1401 (argv mention), .aborted_R4DOSE12_nometa_1401 (argv teacher), .aborted_R4DOSE12_nometa_1401 (teacher) …
- **`ai_v9_74_R4S3b_0829`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/lr_licensing_probe.py; named by 29 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: .aborted_R4DOSE12_nometa_1401 (argv mention), .aborted_R4DOSE12_nometa_1401 (argv teacher), .aborted_R4DOSE12_nometa_1401 (teacher) …
- **`ai_v9_75_R4S3c_0829`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/lr_licensing_probe.py …; named by 31 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: .aborted_R4DOSE12_nometa_1401 (argv mention), .aborted_R4DOSE12_nometa_1401 (argv teacher), .aborted_R4DOSE12_nometa_1401 (teacher) …
- **`ai_v9_76_R4ACTION_0830`** — REFERENCED: loaded by 8 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/resolve_sets.py, designs/research_state/measurements/axis_split_taught_untaught.py …; named by 30 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json, designs/research_state/measurements/axis_split_inputs/cov_R4ACTION.json, designs/research_state/measurements/axis_split_inputs/pilot_R4ACTION_n300.json …
- **`ai_v9_77_G1LEAN_0830`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/starmie_ood_control_traces.py; named by 8 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_79_REVIVE1a_0830`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/axis_split_taught_untaught_2026-08-31.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …; named by another run's model graph: ai_v9_82_REFOLD1_0830 (argv mention), ai_v9_82_REFOLD1_0830 (argv teacher), ai_v9_82_REFOLD1_0830 (teacher)
- **`ai_v9_80_REVIVE1b_0830`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/axis_split_taught_untaught_2026-08-31.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …; named by another run's model graph: ai_v9_82_REFOLD1_0830 (argv mention), ai_v9_82_REFOLD1_0830 (argv teacher), ai_v9_82_REFOLD1_0830 (teacher)
- **`ai_v9_81_REVIVE1c_0830`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/axis_split_taught_untaught_2026-08-31.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …; named by another run's model graph: ai_v9_82_REFOLD1_0830 (argv mention), ai_v9_82_REFOLD1_0830 (argv teacher), ai_v9_82_REFOLD1_0830 (teacher)
- **`ai_v9_82_REFOLD1_0830`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/axis_split_taught_untaught.py; named by 11 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/cov_REFOLD1.json, designs/research_state/measurements/axis_split_inputs/pilot_REFOLD1.json, designs/research_state/measurements/axis_split_taught_untaught_2026-08-31.json …
- **`ai_v9_91_COMPFOLD_0831`** — REFERENCED: loaded by 5 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/axis_split_taught_untaught.py, designs/research_state/measurements/obs_conditioning_probe.py …; named by 16 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json, designs/research_state/measurements/axis_split_inputs/cov_COMPFOLD.json, designs/research_state/measurements/axis_split_inputs/pilot_COMPFOLD_n300.json …
- **`ai_v9_92_R5F00_0831`** — REFERENCED: loaded by 11 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 31 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log …; named by another run's model graph: .dryrun_K6A_1788581936 (argv mention), .dryrun_K6A_1788581936 (argv teacher), .dryrun_K6A_1788581936 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_93_R5F01_0831`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/r5_fleet_teams.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`ai_v9_94_R5F02_0831`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 30 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log …; named by another run's model graph: .dryrun_K6A_1788581936 (argv mention), .dryrun_K6A_1788581936 (argv teacher), .dryrun_K6A_1788581936 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_95_R5F03_0831`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/r5_fleet_teams.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`ai_v9_96_R5F04_0831`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 29 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n9.json …; named by another run's model graph: .dryrun_K6A_1788581936 (argv mention), .dryrun_K6A_1788581936 (argv teacher), .dryrun_K6A_1788581936 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_97_R5F05_0831`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/r5_fleet_teams.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`ai_v9_98_R5F06_0831`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py …; named by 29 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.json, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n3.log, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_v2_n9.json …; named by another run's model graph: .dryrun_K6A_1788581936 (argv mention), .dryrun_K6A_1788581936 (argv teacher), .dryrun_K6A_1788581936 (teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_99_R5F07_0831`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/r5_fleet_teams.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`run_20260830_180409`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`run_20260830_183828`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`run_20260830_184043`** — REFERENCED: named by 6 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`run_20260906_083317`** — REFERENCED: named by 5 committed measurement artifact(s): designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json …
- **`v8rep_p1_A_0905`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`v8rep_p1_B_0905`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`v8rep_p1_C_0905`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`v8rep_p2loss_A_0905`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`v8rep_p2loss_B_0905`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`v8rep_p2loss_C_0905`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`v8rep_p2self_A_0905`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py; named by 3 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift_v8.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`v8rep_p2self_B_0905`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py; named by 3 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift_v8.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`v8rep_p2self_C_0905`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py; named by 3 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift_v8.json, designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md
- **`warmstart_generic_0715`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/tech_debt_census_2026-09-07/lineage_review.md

## What would be KEPT, and why (every CLOSED run with a plan)

<details><summary><code>.aborted_R4DOSE12_nometa_1401</code> — 0.509 GB freed, 1 entries deleted</summary>

**KEEP**

- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>.aborted_R4DOSE12_poolless_1355</code> — 0.0 GB freed, 1 entries deleted</summary>

**KEEP**

- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>.dryrun_K6A_1788581936</code> — 0.509 GB freed, 1 entries deleted</summary>

**KEEP**

- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>DISCARDED_tdaux_control_n16_0818</code> — 0.041 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25867520_steps.json` — first, every-10th
- `checkpoints/checkpoint_25867520_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26667520_steps.json` — last
- `checkpoints/checkpoint_26667520_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>ai_v12_01_winprob_critic</code> — 0.522 GB freed, 19 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_20872192_steps.json` — last
- `checkpoints/checkpoint_20872192_steps.zip` — last
- `checkpoints/checkpoint_3200000_steps.json` — first, every-10th
- `checkpoints/checkpoint_3200000_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_14472192_steps.json`
- `checkpoints/checkpoint_14472192_steps.zip`
- `checkpoints/checkpoint_17672192_steps.json`
- `checkpoints/checkpoint_17672192_steps.zip`
- `checkpoints/checkpoint_6400000_steps.json`
- `checkpoints/checkpoint_6400000_steps.zip`
- `checkpoints/checkpoint_9600000_steps.json`
- `checkpoints/checkpoint_9600000_steps.zip`
- `eval_traces/step_22000000/snapshot.zip`
- `eval_traces/step_20000000/snapshot.zip`
- `eval_traces/step_18000000`
- `eval_traces/step_16000000`
- `eval_traces/step_14000000`
- `eval_traces/step_12000000`
- `eval_traces/step_10000000`
- `eval_traces/step_8000000`
- `eval_traces/step_6000000`
- `eval_traces/step_4000000`
- `eval_traces/step_2000000`

</details>

<details><summary><code>ai_v12_02_winprob_critic.OOM_4096</code> — 0.0 GB freed, 1 entries deleted</summary>

**KEEP**

- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>ai_v12_04_pfsp_fork25M</code> — 0.776 GB freed, 3 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_27254016_steps.json` — first, every-10th
- `checkpoints/checkpoint_27254016_steps.zip` — first, every-10th
- `checkpoints/checkpoint_29654016_steps.json` — last
- `checkpoints/checkpoint_29654016_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_28000032/snapshot.zip`
- `eval_traces/step_26000016/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v12_26_ladder_ctrl10M_shaped</code> — 0.485 GB freed, 6 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_2400000_steps.json` — first, every-10th
- `checkpoints/checkpoint_2400000_steps.zip` — first, every-10th
- `checkpoints/checkpoint_7200000_steps.json` — last
- `checkpoints/checkpoint_7200000_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_4800000_steps.json`
- `checkpoints/checkpoint_4800000_steps.zip`
- `eval_traces/step_8000016/snapshot.zip`
- `eval_traces/step_6000000/snapshot.zip`
- `eval_traces/step_4000032`
- `eval_traces/step_2000016`

</details>

<details><summary><code>ai_v12_27_ladder_ctrl10M_shaped_dense</code> — 0.303 GB freed, 9 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_2400000_steps.json` — first, every-10th
- `checkpoints/checkpoint_2400000_steps.zip` — first, every-10th
- `checkpoints/checkpoint_9600000_steps.json` — last
- `checkpoints/checkpoint_9600000_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_4800000_steps.json`
- `checkpoints/checkpoint_4800000_steps.zip`
- `checkpoints/checkpoint_7200000_steps.json`
- `checkpoints/checkpoint_7200000_steps.zip`
- `eval_traces/step_8000016/snapshot.zip`
- `eval_traces/step_6000000/snapshot.zip`
- `eval_traces/step_4000032`
- `eval_traces/step_2000016`
- `snapshots`

</details>

<details><summary><code>ai_v13_06_exploit_ddtar_spikes</code> — 0.127 GB freed, 3 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_77405952_steps.json` — first, every-10th
- `checkpoints/checkpoint_77405952_steps.zip` — first, every-10th
- `checkpoints/checkpoint_79805952_steps.json` — last
- `checkpoints/checkpoint_79805952_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_80000016/snapshot.zip`
- `eval_traces/step_78000000/snapshot.zip`
- `eval_traces/step_76000032`

</details>

<details><summary><code>ai_v13_10_exploit_stall</code> — 0.136 GB freed, 3 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_77405952_steps.json` — first, every-10th
- `checkpoints/checkpoint_77405952_steps.zip` — first, every-10th
- `checkpoints/checkpoint_82124544_steps.json` — last
- `checkpoints/checkpoint_82124544_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_80000016/snapshot.zip`
- `eval_traces/step_78000000/snapshot.zip`
- `eval_traces/step_76000032`

</details>

<details><summary><code>ai_v13_14_exploit5_balance</code> — 0.142 GB freed, 3 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_97558272_steps.json` — first, every-10th
- `checkpoints/checkpoint_97558272_steps.zip` — first, every-10th
- `checkpoints/checkpoint_99958272_steps.json` — last
- `checkpoints/checkpoint_99958272_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_100000032/snapshot.zip`
- `eval_traces/step_98000016/snapshot.zip`
- `eval_traces/step_96000000`

</details>

<details><summary><code>ai_v13_15_exploit5_stall</code> — 0.143 GB freed, 3 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_97558272_steps.json` — first, every-10th
- `checkpoints/checkpoint_97558272_steps.zip` — first, every-10th
- `checkpoints/checkpoint_99958272_steps.json` — last
- `checkpoints/checkpoint_99958272_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_100000032/snapshot.zip`
- `eval_traces/step_98000016/snapshot.zip`
- `eval_traces/step_96000000`

</details>

<details><summary><code>ai_v13_16_teach5_offense_dist</code> — 0.14 GB freed, 3 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_97558272_steps.json` — first, every-10th
- `checkpoints/checkpoint_97558272_steps.zip` — first, every-10th
- `checkpoints/checkpoint_99958272_steps.json` — last
- `checkpoints/checkpoint_99958272_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_100000032/snapshot.zip`
- `eval_traces/step_98000016/snapshot.zip`
- `eval_traces/step_96000000`

</details>

<details><summary><code>ai_v13_16_teach5_offense_dist_ABANDONED_forklr2p8</code> — 0.706 GB freed, 1 entries deleted</summary>

**KEEP**

- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>ai_v13_17_fold_k1</code> — 1.674 GB freed, 44 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_100778592_steps.json` — every-10th
- `checkpoints/checkpoint_100778592_steps.zip` — every-10th
- `checkpoints/checkpoint_105800592_steps.json` — every-10th
- `checkpoints/checkpoint_105800592_steps.zip` — every-10th
- `checkpoints/checkpoint_106800624_steps.json` — last
- `checkpoints/checkpoint_106800624_steps.zip` — last
- `checkpoints/checkpoint_95658288_steps.json` — first, every-10th
- `checkpoints/checkpoint_95658288_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_100278576_steps.json`
- `checkpoints/checkpoint_100278576_steps.zip`
- `checkpoints/checkpoint_101278608_steps.json`
- `checkpoints/checkpoint_101278608_steps.zip`
- `checkpoints/checkpoint_101778624_steps.json`
- `checkpoints/checkpoint_101778624_steps.zip`
- `checkpoints/checkpoint_102278640_steps.json`
- `checkpoints/checkpoint_102278640_steps.zip`
- `checkpoints/checkpoint_102778656_steps.json`
- `checkpoints/checkpoint_102778656_steps.zip`
- `checkpoints/checkpoint_103278672_steps.json`
- `checkpoints/checkpoint_103278672_steps.zip`
- `checkpoints/checkpoint_103778688_steps.json`
- `checkpoints/checkpoint_103778688_steps.zip`
- `checkpoints/checkpoint_104278704_steps.json`
- `checkpoints/checkpoint_104278704_steps.zip`
- `checkpoints/checkpoint_104800560_steps.json`
- `checkpoints/checkpoint_104800560_steps.zip`
- `checkpoints/checkpoint_105300576_steps.json`
- `checkpoints/checkpoint_105300576_steps.zip`
- `checkpoints/checkpoint_106300608_steps.json`
- `checkpoints/checkpoint_106300608_steps.zip`
- `checkpoints/checkpoint_96158304_steps.json`
- `checkpoints/checkpoint_96158304_steps.zip`
- `checkpoints/checkpoint_96658320_steps.json`
- `checkpoints/checkpoint_96658320_steps.zip`
- `checkpoints/checkpoint_97158336_steps.json`
- `checkpoints/checkpoint_97158336_steps.zip`
- `checkpoints/checkpoint_97658352_steps.json`
- `checkpoints/checkpoint_97658352_steps.zip`
- `checkpoints/checkpoint_98158368_steps.json`
- `checkpoints/checkpoint_98158368_steps.zip`
- `checkpoints/checkpoint_98658384_steps.json`
- `checkpoints/checkpoint_98658384_steps.zip`
- `checkpoints/checkpoint_99158400_steps.json`
- `checkpoints/checkpoint_99158400_steps.zip`
- `checkpoints/checkpoint_99658416_steps.json`
- `checkpoints/checkpoint_99658416_steps.zip`
- `eval_traces/step_104000016/snapshot.zip`
- `eval_traces/step_102000000/snapshot.zip`
- `eval_traces/step_100000032`
- `eval_traces/step_98000016`
- `eval_traces/step_96000000`
- `snapshots`

</details>

<details><summary><code>ai_v13_18_fold_k3</code> — 1.671 GB freed, 44 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_100778592_steps.json` — every-10th
- `checkpoints/checkpoint_100778592_steps.zip` — every-10th
- `checkpoints/checkpoint_105898896_steps.json` — every-10th
- `checkpoints/checkpoint_105898896_steps.zip` — every-10th
- `checkpoints/checkpoint_106898928_steps.json` — last
- `checkpoints/checkpoint_106898928_steps.zip` — last
- `checkpoints/checkpoint_95658288_steps.json` — first, every-10th
- `checkpoints/checkpoint_95658288_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_100278576_steps.json`
- `checkpoints/checkpoint_100278576_steps.zip`
- `checkpoints/checkpoint_101278608_steps.json`
- `checkpoints/checkpoint_101278608_steps.zip`
- `checkpoints/checkpoint_101778624_steps.json`
- `checkpoints/checkpoint_101778624_steps.zip`
- `checkpoints/checkpoint_102278640_steps.json`
- `checkpoints/checkpoint_102278640_steps.zip`
- `checkpoints/checkpoint_102778656_steps.json`
- `checkpoints/checkpoint_102778656_steps.zip`
- `checkpoints/checkpoint_103278672_steps.json`
- `checkpoints/checkpoint_103278672_steps.zip`
- `checkpoints/checkpoint_103778688_steps.json`
- `checkpoints/checkpoint_103778688_steps.zip`
- `checkpoints/checkpoint_104278704_steps.json`
- `checkpoints/checkpoint_104278704_steps.zip`
- `checkpoints/checkpoint_104898864_steps.json`
- `checkpoints/checkpoint_104898864_steps.zip`
- `checkpoints/checkpoint_105398880_steps.json`
- `checkpoints/checkpoint_105398880_steps.zip`
- `checkpoints/checkpoint_106398912_steps.json`
- `checkpoints/checkpoint_106398912_steps.zip`
- `checkpoints/checkpoint_96158304_steps.json`
- `checkpoints/checkpoint_96158304_steps.zip`
- `checkpoints/checkpoint_96658320_steps.json`
- `checkpoints/checkpoint_96658320_steps.zip`
- `checkpoints/checkpoint_97158336_steps.json`
- `checkpoints/checkpoint_97158336_steps.zip`
- `checkpoints/checkpoint_97658352_steps.json`
- `checkpoints/checkpoint_97658352_steps.zip`
- `checkpoints/checkpoint_98158368_steps.json`
- `checkpoints/checkpoint_98158368_steps.zip`
- `checkpoints/checkpoint_98658384_steps.json`
- `checkpoints/checkpoint_98658384_steps.zip`
- `checkpoints/checkpoint_99158400_steps.json`
- `checkpoints/checkpoint_99158400_steps.zip`
- `checkpoints/checkpoint_99658416_steps.json`
- `checkpoints/checkpoint_99658416_steps.zip`
- `eval_traces/step_104000016/snapshot.zip`
- `eval_traces/step_102000000/snapshot.zip`
- `eval_traces/step_100000032`
- `eval_traces/step_98000016`
- `eval_traces/step_96000000`
- `snapshots`

</details>

<details><summary><code>ai_v13_19_fold_k11</code> — 1.676 GB freed, 44 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_100778592_steps.json` — every-10th
- `checkpoints/checkpoint_100778592_steps.zip` — every-10th
- `checkpoints/checkpoint_105800592_steps.json` — every-10th
- `checkpoints/checkpoint_105800592_steps.zip` — every-10th
- `checkpoints/checkpoint_106800624_steps.json` — last
- `checkpoints/checkpoint_106800624_steps.zip` — last
- `checkpoints/checkpoint_95658288_steps.json` — first, every-10th
- `checkpoints/checkpoint_95658288_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_100278576_steps.json`
- `checkpoints/checkpoint_100278576_steps.zip`
- `checkpoints/checkpoint_101278608_steps.json`
- `checkpoints/checkpoint_101278608_steps.zip`
- `checkpoints/checkpoint_101778624_steps.json`
- `checkpoints/checkpoint_101778624_steps.zip`
- `checkpoints/checkpoint_102278640_steps.json`
- `checkpoints/checkpoint_102278640_steps.zip`
- `checkpoints/checkpoint_102778656_steps.json`
- `checkpoints/checkpoint_102778656_steps.zip`
- `checkpoints/checkpoint_103278672_steps.json`
- `checkpoints/checkpoint_103278672_steps.zip`
- `checkpoints/checkpoint_103778688_steps.json`
- `checkpoints/checkpoint_103778688_steps.zip`
- `checkpoints/checkpoint_104278704_steps.json`
- `checkpoints/checkpoint_104278704_steps.zip`
- `checkpoints/checkpoint_104800560_steps.json`
- `checkpoints/checkpoint_104800560_steps.zip`
- `checkpoints/checkpoint_105300576_steps.json`
- `checkpoints/checkpoint_105300576_steps.zip`
- `checkpoints/checkpoint_106300608_steps.json`
- `checkpoints/checkpoint_106300608_steps.zip`
- `checkpoints/checkpoint_96158304_steps.json`
- `checkpoints/checkpoint_96158304_steps.zip`
- `checkpoints/checkpoint_96658320_steps.json`
- `checkpoints/checkpoint_96658320_steps.zip`
- `checkpoints/checkpoint_97158336_steps.json`
- `checkpoints/checkpoint_97158336_steps.zip`
- `checkpoints/checkpoint_97658352_steps.json`
- `checkpoints/checkpoint_97658352_steps.zip`
- `checkpoints/checkpoint_98158368_steps.json`
- `checkpoints/checkpoint_98158368_steps.zip`
- `checkpoints/checkpoint_98658384_steps.json`
- `checkpoints/checkpoint_98658384_steps.zip`
- `checkpoints/checkpoint_99158400_steps.json`
- `checkpoints/checkpoint_99158400_steps.zip`
- `checkpoints/checkpoint_99658416_steps.json`
- `checkpoints/checkpoint_99658416_steps.zip`
- `eval_traces/step_104000016/snapshot.zip`
- `eval_traces/step_102000000/snapshot.zip`
- `eval_traces/step_100000032`
- `eval_traces/step_98000016`
- `eval_traces/step_96000000`
- `snapshots`

</details>

<details><summary><code>ai_v13_20_fold_k11_sharematched</code> — 1.642 GB freed, 42 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_100983696_steps.json` — every-10th
- `checkpoints/checkpoint_100983696_steps.zip` — every-10th
- `checkpoints/checkpoint_106415904_steps.json` — every-10th
- `checkpoints/checkpoint_106415904_steps.zip` — every-10th
- `checkpoints/checkpoint_106915920_steps.json` — last
- `checkpoints/checkpoint_106915920_steps.zip` — last
- `checkpoints/checkpoint_95658288_steps.json` — first, every-10th
- `checkpoints/checkpoint_95658288_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_100483680_steps.json`
- `checkpoints/checkpoint_100483680_steps.zip`
- `checkpoints/checkpoint_101483712_steps.json`
- `checkpoints/checkpoint_101483712_steps.zip`
- `checkpoints/checkpoint_101983728_steps.json`
- `checkpoints/checkpoint_101983728_steps.zip`
- `checkpoints/checkpoint_102483744_steps.json`
- `checkpoints/checkpoint_102483744_steps.zip`
- `checkpoints/checkpoint_102983760_steps.json`
- `checkpoints/checkpoint_102983760_steps.zip`
- `checkpoints/checkpoint_103915824_steps.json`
- `checkpoints/checkpoint_103915824_steps.zip`
- `checkpoints/checkpoint_104415840_steps.json`
- `checkpoints/checkpoint_104415840_steps.zip`
- `checkpoints/checkpoint_104915856_steps.json`
- `checkpoints/checkpoint_104915856_steps.zip`
- `checkpoints/checkpoint_105415872_steps.json`
- `checkpoints/checkpoint_105415872_steps.zip`
- `checkpoints/checkpoint_105915888_steps.json`
- `checkpoints/checkpoint_105915888_steps.zip`
- `checkpoints/checkpoint_96158304_steps.json`
- `checkpoints/checkpoint_96158304_steps.zip`
- `checkpoints/checkpoint_96658320_steps.json`
- `checkpoints/checkpoint_96658320_steps.zip`
- `checkpoints/checkpoint_97158336_steps.json`
- `checkpoints/checkpoint_97158336_steps.zip`
- `checkpoints/checkpoint_97658352_steps.json`
- `checkpoints/checkpoint_97658352_steps.zip`
- `checkpoints/checkpoint_98158368_steps.json`
- `checkpoints/checkpoint_98158368_steps.zip`
- `checkpoints/checkpoint_98658384_steps.json`
- `checkpoints/checkpoint_98658384_steps.zip`
- `checkpoints/checkpoint_99158400_steps.json`
- `checkpoints/checkpoint_99158400_steps.zip`
- `checkpoints/checkpoint_99983664_steps.json`
- `checkpoints/checkpoint_99983664_steps.zip`
- `eval_traces/step_104000016/snapshot.zip`
- `eval_traces/step_102000000/snapshot.zip`
- `eval_traces/step_100000032`
- `eval_traces/step_98000016`
- `eval_traces/step_96000000`
- `snapshots`

</details>

<details><summary><code>ai_v5_10_tail1_23_0611</code> — 0.264 GB freed, 5 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_23530799_steps.json` — last
- `checkpoints/checkpoint_23530799_steps.zip` — last
- `checkpoints/checkpoint_957397_steps.json` — first, every-10th
- `checkpoints/checkpoint_957397_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10317874_steps.json`
- `checkpoints/checkpoint_10317874_steps.zip`
- `checkpoints/checkpoint_17892968_steps.json`
- `checkpoints/checkpoint_17892968_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v5_11_tail2_53m_0611</code> — 0.763 GB freed, 13 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_52056146_steps.json` — last
- `checkpoints/checkpoint_52056146_steps.zip` — last
- `checkpoints/checkpoint_955745_steps.json` — first, every-10th
- `checkpoints/checkpoint_955745_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10091682_steps.json`
- `checkpoints/checkpoint_10091682_steps.zip`
- `checkpoints/checkpoint_18029970_steps.json`
- `checkpoints/checkpoint_18029970_steps.zip`
- `checkpoints/checkpoint_25414312_steps.json`
- `checkpoints/checkpoint_25414312_steps.zip`
- `checkpoints/checkpoint_32991848_steps.json`
- `checkpoints/checkpoint_32991848_steps.zip`
- `checkpoints/checkpoint_40202626_steps.json`
- `checkpoints/checkpoint_40202626_steps.zip`
- `checkpoints/checkpoint_47275909_steps.json`
- `checkpoints/checkpoint_47275909_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v5_12_bias_05_N_0612</code> — 0.294 GB freed, 5 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_23309510_steps.json` — last
- `checkpoints/checkpoint_23309510_steps.zip` — last
- `checkpoints/checkpoint_951950_steps.json` — first, every-10th
- `checkpoints/checkpoint_951950_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_17204319_steps.json`
- `checkpoints/checkpoint_17204319_steps.zip`
- `checkpoints/checkpoint_9890482_steps.json`
- `checkpoints/checkpoint_9890482_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v5_13_shape_pbrs_43m_0612</code> — 0.147 GB freed, 10 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_43434034_steps.json` — last
- `checkpoints/checkpoint_43434034_steps.zip` — last
- `checkpoints/checkpoint_908672_steps.json` — first, every-10th
- `checkpoints/checkpoint_908672_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10233250_steps.json`
- `checkpoints/checkpoint_10233250_steps.zip`
- `checkpoints/checkpoint_18203428_steps.json`
- `checkpoints/checkpoint_18203428_steps.zip`
- `checkpoints/checkpoint_25559196_steps.json`
- `checkpoints/checkpoint_25559196_steps.zip`
- `checkpoints/checkpoint_33189756_steps.json`
- `checkpoints/checkpoint_33189756_steps.zip`
- `checkpoints/checkpoint_40686976_steps.json`
- `checkpoints/checkpoint_40686976_steps.zip`

</details>

<details><summary><code>ai_v5_2_native_selfplay_50m_0606</code> — 0.141 GB freed, 5 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1022179_steps.json` — first, every-10th
- `checkpoints/checkpoint_1022179_steps.zip` — first, every-10th
- `checkpoints/checkpoint_29865854_steps.json` — last
- `checkpoints/checkpoint_29865854_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_11576311_steps.json`
- `checkpoints/checkpoint_11576311_steps.zip`
- `checkpoints/checkpoint_21267940_steps.json`
- `checkpoints/checkpoint_21267940_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v5_3_vf_coef_clip_50m_0606</code> — 0.623 GB freed, 11 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_50056021_steps.json` — last
- `checkpoints/checkpoint_50056021_steps.zip` — last
- `checkpoints/checkpoint_998727_steps.json` — first, every-10th
- `checkpoints/checkpoint_998727_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10895401_steps.json`
- `checkpoints/checkpoint_10895401_steps.zip`
- `checkpoints/checkpoint_20760103_steps.json`
- `checkpoints/checkpoint_20760103_steps.zip`
- `checkpoints/checkpoint_28792110_steps.json`
- `checkpoints/checkpoint_28792110_steps.zip`
- `checkpoints/checkpoint_36130928_steps.json`
- `checkpoints/checkpoint_36130928_steps.zip`
- `checkpoints/checkpoint_43241595_steps.json`
- `checkpoints/checkpoint_43241595_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v5_4_pbrs_opp_threat_50m_0607</code> — 0.396 GB freed, 9 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_35315608_steps.json` — last
- `checkpoints/checkpoint_35315608_steps.zip` — last
- `checkpoints/checkpoint_909273_steps.json` — first, every-10th
- `checkpoints/checkpoint_909273_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10223552_steps.json`
- `checkpoints/checkpoint_10223552_steps.zip`
- `checkpoints/checkpoint_19172258_steps.json`
- `checkpoints/checkpoint_19172258_steps.zip`
- `checkpoints/checkpoint_26654947_steps.json`
- `checkpoints/checkpoint_26654947_steps.zip`
- `checkpoints/checkpoint_33388083_steps.json`
- `checkpoints/checkpoint_33388083_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v5_5_popart_50m_0607</code> — 0.198 GB freed, 14 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_52052399_steps.json` — last
- `checkpoints/checkpoint_52052399_steps.zip` — last
- `checkpoints/checkpoint_884999_steps.json` — first, every-10th
- `checkpoints/checkpoint_884999_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10025810_steps.json`
- `checkpoints/checkpoint_10025810_steps.zip`
- `checkpoints/checkpoint_16858256_steps.json`
- `checkpoints/checkpoint_16858256_steps.zip`
- `checkpoints/checkpoint_23063353_steps.json`
- `checkpoints/checkpoint_23063353_steps.zip`
- `checkpoints/checkpoint_29275474_steps.json`
- `checkpoints/checkpoint_29275474_steps.zip`
- `checkpoints/checkpoint_35934118_steps.json`
- `checkpoints/checkpoint_35934118_steps.zip`
- `checkpoints/checkpoint_43682849_steps.json`
- `checkpoints/checkpoint_43682849_steps.zip`
- `checkpoints/checkpoint_51375672_steps.json`
- `checkpoints/checkpoint_51375672_steps.zip`

</details>

<details><summary><code>ai_v5_6_stable_70m_0608</code> — 0.255 GB freed, 18 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_70001818_steps.json` — last, every-10th
- `checkpoints/checkpoint_70001818_steps.zip` — last, every-10th
- `checkpoints/checkpoint_953844_steps.json` — first, every-10th
- `checkpoints/checkpoint_953844_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_17583545_steps.json`
- `checkpoints/checkpoint_17583545_steps.zip`
- `checkpoints/checkpoint_24944812_steps.json`
- `checkpoints/checkpoint_24944812_steps.zip`
- `checkpoints/checkpoint_32086652_steps.json`
- `checkpoints/checkpoint_32086652_steps.zip`
- `checkpoints/checkpoint_39741911_steps.json`
- `checkpoints/checkpoint_39741911_steps.zip`
- `checkpoints/checkpoint_46658999_steps.json`
- `checkpoints/checkpoint_46658999_steps.zip`
- `checkpoints/checkpoint_53575532_steps.json`
- `checkpoints/checkpoint_53575532_steps.zip`
- `checkpoints/checkpoint_60816966_steps.json`
- `checkpoints/checkpoint_60816966_steps.zip`
- `checkpoints/checkpoint_67891233_steps.json`
- `checkpoints/checkpoint_67891233_steps.zip`
- `checkpoints/checkpoint_9957951_steps.json`
- `checkpoints/checkpoint_9957951_steps.zip`

</details>

<details><summary><code>ai_v5_7_switch_bias_41m_0609</code> — 0.595 GB freed, 11 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_40992544_steps.json` — last
- `checkpoints/checkpoint_40992544_steps.zip` — last
- `checkpoints/checkpoint_989468_steps.json` — first, every-10th
- `checkpoints/checkpoint_989468_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10071937_steps.json`
- `checkpoints/checkpoint_10071937_steps.zip`
- `checkpoints/checkpoint_17846156_steps.json`
- `checkpoints/checkpoint_17846156_steps.zip`
- `checkpoints/checkpoint_25257234_steps.json`
- `checkpoints/checkpoint_25257234_steps.zip`
- `checkpoints/checkpoint_32484321_steps.json`
- `checkpoints/checkpoint_32484321_steps.zip`
- `checkpoints/checkpoint_40296240_steps.json`
- `checkpoints/checkpoint_40296240_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v5_8_split_inc_dmg_38m_0610</code> — 0.143 GB freed, 9 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_37392713_steps.json` — last, latest.txt pin
- `checkpoints/checkpoint_37392713_steps.zip` — last, latest.txt pin
- `checkpoints/checkpoint_948331_steps.json` — first, every-10th
- `checkpoints/checkpoint_948331_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_17301015_steps.json`
- `checkpoints/checkpoint_17301015_steps.zip`
- `checkpoints/checkpoint_24480645_steps.json`
- `checkpoints/checkpoint_24480645_steps.zip`
- `checkpoints/checkpoint_31913706_steps.json`
- `checkpoints/checkpoint_31913706_steps.zip`
- `checkpoints/checkpoint_9600815_steps.json`
- `checkpoints/checkpoint_9600815_steps.zip`
- `eval_traces/step_36000004/snapshot.zip`

</details>

<details><summary><code>ai_v5_9_attend_unrevealed_56m_0610</code> — 0.2 GB freed, 14 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_56555334_steps.json` — last
- `checkpoints/checkpoint_56555334_steps.zip` — last
- `checkpoints/checkpoint_972659_steps.json` — first, every-10th
- `checkpoints/checkpoint_972659_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10139971_steps.json`
- `checkpoints/checkpoint_10139971_steps.zip`
- `checkpoints/checkpoint_17793530_steps.json`
- `checkpoints/checkpoint_17793530_steps.zip`
- `checkpoints/checkpoint_25364744_steps.json`
- `checkpoints/checkpoint_25364744_steps.zip`
- `checkpoints/checkpoint_32537417_steps.json`
- `checkpoints/checkpoint_32537417_steps.zip`
- `checkpoints/checkpoint_40368403_steps.json`
- `checkpoints/checkpoint_40368403_steps.zip`
- `checkpoints/checkpoint_48088654_steps.json`
- `checkpoints/checkpoint_48088654_steps.zip`
- `checkpoints/checkpoint_55091450_steps.json`
- `checkpoints/checkpoint_55091450_steps.zip`

</details>

<details><summary><code>ai_v6_01_belief_53m_0613</code> — 0.735 GB freed, 13 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_52910114_steps.json` — last
- `checkpoints/checkpoint_52910114_steps.zip` — last
- `checkpoints/checkpoint_976687_steps.json` — first, every-10th
- `checkpoints/checkpoint_976687_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10797773_steps.json`
- `checkpoints/checkpoint_10797773_steps.zip`
- `checkpoints/checkpoint_18883182_steps.json`
- `checkpoints/checkpoint_18883182_steps.zip`
- `checkpoints/checkpoint_26108605_steps.json`
- `checkpoints/checkpoint_26108605_steps.zip`
- `checkpoints/checkpoint_33238595_steps.json`
- `checkpoints/checkpoint_33238595_steps.zip`
- `checkpoints/checkpoint_40618773_steps.json`
- `checkpoints/checkpoint_40618773_steps.zip`
- `checkpoints/checkpoint_48091911_steps.json`
- `checkpoints/checkpoint_48091911_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v6_02_belief_lat_16m_0614</code> — 0.156 GB freed, 3 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_16510131_steps.json` — last
- `checkpoints/checkpoint_16510131_steps.zip` — last
- `checkpoints/checkpoint_983006_steps.json` — first, every-10th
- `checkpoints/checkpoint_983006_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10260721_steps.json`
- `checkpoints/checkpoint_10260721_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v6_03_win_pred_N_0614</code> — 0.869 GB freed, 15 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_58917105_steps.json` — last
- `checkpoints/checkpoint_58917105_steps.zip` — last
- `checkpoints/checkpoint_990898_steps.json` — first, every-10th
- `checkpoints/checkpoint_990898_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10388758_steps.json`
- `checkpoints/checkpoint_10388758_steps.zip`
- `checkpoints/checkpoint_17984199_steps.json`
- `checkpoints/checkpoint_17984199_steps.zip`
- `checkpoints/checkpoint_25226792_steps.json`
- `checkpoints/checkpoint_25226792_steps.zip`
- `checkpoints/checkpoint_32373631_steps.json`
- `checkpoints/checkpoint_32373631_steps.zip`
- `checkpoints/checkpoint_39602721_steps.json`
- `checkpoints/checkpoint_39602721_steps.zip`
- `checkpoints/checkpoint_46725020_steps.json`
- `checkpoints/checkpoint_46725020_steps.zip`
- `checkpoints/checkpoint_53869147_steps.json`
- `checkpoints/checkpoint_53869147_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v6_04_unified_all_half_batch_N_0616</code> — 0.069 GB freed, 3 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1177011_steps.json` — first, every-10th
- `checkpoints/checkpoint_1177011_steps.zip` — first, every-10th
- `checkpoints/checkpoint_13838366_steps.json` — last
- `checkpoints/checkpoint_13838366_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12793484_steps.json`
- `checkpoints/checkpoint_12793484_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v6_04_unified_inc_N_0615</code> — 0.635 GB freed, 9 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1091845_steps.json` — first, every-10th
- `checkpoints/checkpoint_1091845_steps.zip` — first, every-10th
- `checkpoints/checkpoint_38937343_steps.json` — last
- `checkpoints/checkpoint_38937343_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_11858658_steps.json`
- `checkpoints/checkpoint_11858658_steps.zip`
- `checkpoints/checkpoint_19438316_steps.json`
- `checkpoints/checkpoint_19438316_steps.zip`
- `checkpoints/checkpoint_28023692_steps.json`
- `checkpoints/checkpoint_28023692_steps.zip`
- `checkpoints/checkpoint_35946535_steps.json`
- `checkpoints/checkpoint_35946535_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v6_06_unified_all_N_0616</code> — 0.243 GB freed, 5 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1180485_steps.json` — first, every-10th
- `checkpoints/checkpoint_1180485_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26107495_steps.json` — last
- `checkpoints/checkpoint_26107495_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12836693_steps.json`
- `checkpoints/checkpoint_12836693_steps.zip`
- `checkpoints/checkpoint_23422064_steps.json`
- `checkpoints/checkpoint_23422064_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v6_07_unified_topk_N_0616</code> — 0.494 GB freed, 6 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1187322_steps.json` — first, every-10th
- `checkpoints/checkpoint_1187322_steps.zip` — first, every-10th
- `checkpoints/checkpoint_31634159_steps.json` — last
- `checkpoints/checkpoint_31634159_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_13922816_steps.json`
- `checkpoints/checkpoint_13922816_steps.zip`
- `checkpoints/checkpoint_24553381_steps.json`
- `checkpoints/checkpoint_24553381_steps.zip`
- `eval_traces/step_30000003/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v6_08_unmasked_floor_N_0617</code> — 0.342 GB freed, 3 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1181770_steps.json` — first, every-10th
- `checkpoints/checkpoint_1181770_steps.zip` — first, every-10th
- `checkpoints/checkpoint_22002567_steps.json` — last
- `checkpoints/checkpoint_22002567_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12933045_steps.json`
- `checkpoints/checkpoint_12933045_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v6_09_dmg_reattend_N_0617</code> — 0.91 GB freed, 7 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1196649_steps.json` — first, every-10th
- `checkpoints/checkpoint_1196649_steps.zip` — first, every-10th
- `checkpoints/checkpoint_42458933_steps.json` — last
- `checkpoints/checkpoint_42458933_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12464402_steps.json`
- `checkpoints/checkpoint_12464402_steps.zip`
- `checkpoints/checkpoint_22345697_steps.json`
- `checkpoints/checkpoint_22345697_steps.zip`
- `checkpoints/checkpoint_32837718_steps.json`
- `checkpoints/checkpoint_32837718_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v6_10_unified_obs_0618</code> — 0.0 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1197172_steps.json` — first, every-10th
- `checkpoints/checkpoint_1197172_steps.zip` — first, every-10th
- `checkpoints/checkpoint_9572669_steps.json` — last
- `checkpoints/checkpoint_9572669_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>ai_v6_11_typed_hp_0619</code> — 1.051 GB freed, 9 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1197110_steps.json` — first, every-10th
- `checkpoints/checkpoint_1197110_steps.zip` — first, every-10th
- `checkpoints/checkpoint_45793883_steps.json` — last
- `checkpoints/checkpoint_45793883_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_11910141_steps.json`
- `checkpoints/checkpoint_11910141_steps.zip`
- `checkpoints/checkpoint_22676708_steps.json`
- `checkpoints/checkpoint_22676708_steps.zip`
- `checkpoints/checkpoint_32864563_steps.json`
- `checkpoints/checkpoint_32864563_steps.zip`
- `checkpoints/checkpoint_42953465_steps.json`
- `checkpoints/checkpoint_42953465_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v6_11_unified_obs_fixed_0618</code> — 0.738 GB freed, 7 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1196924_steps.json` — first, every-10th
- `checkpoints/checkpoint_1196924_steps.zip` — first, every-10th
- `checkpoints/checkpoint_33633775_steps.json` — last
- `checkpoints/checkpoint_33633775_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12022153_steps.json`
- `checkpoints/checkpoint_12022153_steps.zip`
- `checkpoints/checkpoint_22258877_steps.json`
- `checkpoints/checkpoint_22258877_steps.zip`
- `checkpoints/checkpoint_31731498_steps.json`
- `checkpoints/checkpoint_31731498_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v6_13_outgoing_dmg_0620</code> — 0.406 GB freed, 18 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_103698064_steps.json` — last
- `checkpoints/checkpoint_103698064_steps.zip` — last
- `checkpoints/checkpoint_1197099_steps.json` — first, every-10th
- `checkpoints/checkpoint_1197099_steps.zip` — first, every-10th
- `checkpoints/checkpoint_95809715_steps.json` — every-10th
- `checkpoints/checkpoint_95809715_steps.zip` — every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12864301_steps.json`
- `checkpoints/checkpoint_12864301_steps.zip`
- `checkpoints/checkpoint_19593297_steps.json`
- `checkpoints/checkpoint_19593297_steps.zip`
- `checkpoints/checkpoint_26541236_steps.json`
- `checkpoints/checkpoint_26541236_steps.zip`
- `checkpoints/checkpoint_36537058_steps.json`
- `checkpoints/checkpoint_36537058_steps.zip`
- `checkpoints/checkpoint_46438240_steps.json`
- `checkpoints/checkpoint_46438240_steps.zip`
- `checkpoints/checkpoint_56175452_steps.json`
- `checkpoints/checkpoint_56175452_steps.zip`
- `checkpoints/checkpoint_65881349_steps.json`
- `checkpoints/checkpoint_65881349_steps.zip`
- `checkpoints/checkpoint_75907912_steps.json`
- `checkpoints/checkpoint_75907912_steps.zip`
- `checkpoints/checkpoint_85782148_steps.json`
- `checkpoints/checkpoint_85782148_steps.zip`

</details>

<details><summary><code>ai_v6_13_outgoing_dmg_0620_exp_v1</code> — 0.045 GB freed, 2 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_105692114_steps.json` — first, every-10th
- `checkpoints/checkpoint_105692114_steps.zip` — first, every-10th
- `checkpoints/checkpoint_124697151_steps.json` — last
- `checkpoints/checkpoint_124697151_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_116069255_steps.json`
- `checkpoints/checkpoint_116069255_steps.zip`

</details>

<details><summary><code>ai_v6_13_outgoing_dmg_0620_exploiter_v1</code> — 0.135 GB freed, 6 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_127523438_steps.json` — last
- `checkpoints/checkpoint_127523438_steps.zip` — last
- `checkpoints/checkpoint_96917276_steps.json` — first, every-10th
- `checkpoints/checkpoint_96917276_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_106204357_steps.json`
- `checkpoints/checkpoint_106204357_steps.zip`
- `checkpoints/checkpoint_115479375_steps.json`
- `checkpoints/checkpoint_115479375_steps.zip`
- `checkpoints/checkpoint_124763606_steps.json`
- `checkpoints/checkpoint_124763606_steps.zip`

</details>

<details><summary><code>ai_v6_13_outgoing_dmg_0620_exploiter_v2</code> — 0.181 GB freed, 7 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_35605573_steps.json` — last
- `checkpoints/checkpoint_35605573_steps.zip` — last
- `checkpoints/checkpoint_5727495_steps.json` — first, every-10th
- `checkpoints/checkpoint_5727495_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_15627261_steps.json`
- `checkpoints/checkpoint_15627261_steps.zip`
- `checkpoints/checkpoint_25106888_steps.json`
- `checkpoints/checkpoint_25106888_steps.zip`
- `checkpoints/checkpoint_34673207_steps.json`
- `checkpoints/checkpoint_34673207_steps.zip`
- `eval_traces/step_34000015/snapshot.zip`

</details>

<details><summary><code>ai_v7_01_teacher_0626</code> — 0.564 GB freed, 7 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1197315_steps.json` — first, every-10th
- `checkpoints/checkpoint_1197315_steps.zip` — first, every-10th
- `checkpoints/checkpoint_33392145_steps.json` — last
- `checkpoints/checkpoint_33392145_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12570909_steps.json`
- `checkpoints/checkpoint_12570909_steps.zip`
- `checkpoints/checkpoint_22351408_steps.json`
- `checkpoints/checkpoint_22351408_steps.zip`
- `checkpoints/checkpoint_32438549_steps.json`
- `checkpoints/checkpoint_32438549_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v7_01_teacher_0626_oom1</code> — 0.0 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1197011_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_1197011_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>ai_v7_02_critic_shape_0627</code> — 0.391 GB freed, 18 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_106685764_steps.json` — last, every-10th, referenced by another run's lineage
- `checkpoints/checkpoint_106685764_steps.zip` — last, every-10th, referenced by another run's lineage
- `checkpoints/checkpoint_1197490_steps.json` — first, every-10th
- `checkpoints/checkpoint_1197490_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_13070322_steps.json`
- `checkpoints/checkpoint_13070322_steps.zip`
- `checkpoints/checkpoint_22872762_steps.json`
- `checkpoints/checkpoint_22872762_steps.zip`
- `checkpoints/checkpoint_33480321_steps.json`
- `checkpoints/checkpoint_33480321_steps.zip`
- `checkpoints/checkpoint_43968969_steps.json`
- `checkpoints/checkpoint_43968969_steps.zip`
- `checkpoints/checkpoint_54574533_steps.json`
- `checkpoints/checkpoint_54574533_steps.zip`
- `checkpoints/checkpoint_65944051_steps.json`
- `checkpoints/checkpoint_65944051_steps.zip`
- `checkpoints/checkpoint_76349880_steps.json`
- `checkpoints/checkpoint_76349880_steps.zip`
- `checkpoints/checkpoint_86931944_steps.json`
- `checkpoints/checkpoint_86931944_steps.zip`
- `checkpoints/checkpoint_97532917_steps.json`
- `checkpoints/checkpoint_97532917_steps.zip`

</details>

<details><summary><code>ai_v7_03_belief_shape_0630</code> — 1.085 GB freed, 11 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1197121_steps.json` — first, every-10th
- `checkpoints/checkpoint_1197121_steps.zip` — first, every-10th
- `checkpoints/checkpoint_59697301_steps.json` — last
- `checkpoints/checkpoint_59697301_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12872291_steps.json`
- `checkpoints/checkpoint_12872291_steps.zip`
- `checkpoints/checkpoint_23545725_steps.json`
- `checkpoints/checkpoint_23545725_steps.zip`
- `checkpoints/checkpoint_34144809_steps.json`
- `checkpoints/checkpoint_34144809_steps.zip`
- `checkpoints/checkpoint_44530677_steps.json`
- `checkpoints/checkpoint_44530677_steps.zip`
- `checkpoints/checkpoint_55823083_steps.json`
- `checkpoints/checkpoint_55823083_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v7_04_opd_selfdistill_0702</code> — 1.346 GB freed, 23 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_107652399_steps.json` — every-10th
- `checkpoints/checkpoint_107652399_steps.zip` — every-10th
- `checkpoints/checkpoint_1197490_steps.json` — first, every-10th
- `checkpoints/checkpoint_1197490_steps.zip` — first, every-10th
- `checkpoints/checkpoint_135694065_steps.json` — last
- `checkpoints/checkpoint_135694065_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_117444375_steps.json`
- `checkpoints/checkpoint_117444375_steps.zip`
- `checkpoints/checkpoint_127557393_steps.json`
- `checkpoints/checkpoint_127557393_steps.zip`
- `checkpoints/checkpoint_13070322_steps.json`
- `checkpoints/checkpoint_13070322_steps.zip`
- `checkpoints/checkpoint_22872762_steps.json`
- `checkpoints/checkpoint_22872762_steps.zip`
- `checkpoints/checkpoint_33480321_steps.json`
- `checkpoints/checkpoint_33480321_steps.zip`
- `checkpoints/checkpoint_43968969_steps.json`
- `checkpoints/checkpoint_43968969_steps.zip`
- `checkpoints/checkpoint_54574533_steps.json`
- `checkpoints/checkpoint_54574533_steps.zip`
- `checkpoints/checkpoint_65944051_steps.json`
- `checkpoints/checkpoint_65944051_steps.zip`
- `checkpoints/checkpoint_76349880_steps.json`
- `checkpoints/checkpoint_76349880_steps.zip`
- `checkpoints/checkpoint_86931944_steps.json`
- `checkpoints/checkpoint_86931944_steps.zip`
- `checkpoints/checkpoint_97532917_steps.json`
- `checkpoints/checkpoint_97532917_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v7_05_tss_specialist_0703</code> — 0.391 GB freed, 18 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1085943_steps.json` — first, every-10th
- `checkpoints/checkpoint_1085943_steps.zip` — first, every-10th
- `checkpoints/checkpoint_111279963_steps.json` — last, every-10th
- `checkpoints/checkpoint_111279963_steps.zip` — last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_102974861_steps.json`
- `checkpoints/checkpoint_102974861_steps.zip`
- `checkpoints/checkpoint_12408602_steps.json`
- `checkpoints/checkpoint_12408602_steps.zip`
- `checkpoints/checkpoint_23765874_steps.json`
- `checkpoints/checkpoint_23765874_steps.zip`
- `checkpoints/checkpoint_35034481_steps.json`
- `checkpoints/checkpoint_35034481_steps.zip`
- `checkpoints/checkpoint_46286357_steps.json`
- `checkpoints/checkpoint_46286357_steps.zip`
- `checkpoints/checkpoint_57435929_steps.json`
- `checkpoints/checkpoint_57435929_steps.zip`
- `checkpoints/checkpoint_68686341_steps.json`
- `checkpoints/checkpoint_68686341_steps.zip`
- `checkpoints/checkpoint_79847848_steps.json`
- `checkpoints/checkpoint_79847848_steps.zip`
- `checkpoints/checkpoint_91128788_steps.json`
- `checkpoints/checkpoint_91128788_steps.zip`

</details>

<details><summary><code>ai_v7_06_tss_temp_anneal_0706</code> — 0.087 GB freed, 4 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1059516_steps.json` — first, every-10th
- `checkpoints/checkpoint_1059516_steps.zip` — first, every-10th
- `checkpoints/checkpoint_32470224_steps.json` — last
- `checkpoints/checkpoint_32470224_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12309055_steps.json`
- `checkpoints/checkpoint_12309055_steps.zip`
- `checkpoints/checkpoint_23777456_steps.json`
- `checkpoints/checkpoint_23777456_steps.zip`

</details>

<details><summary><code>ai_v7_07_tss_temp_ratchet_0707</code> — 0.043 GB freed, 2 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1025468_steps.json` — first, every-10th
- `checkpoints/checkpoint_1025468_steps.zip` — first, every-10th
- `checkpoints/checkpoint_18609109_steps.json` — last
- `checkpoints/checkpoint_18609109_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12330312_steps.json`
- `checkpoints/checkpoint_12330312_steps.zip`

</details>

<details><summary><code>ai_v7_08_tss_bots_0707</code> — 0.174 GB freed, 8 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1196467_steps.json` — first, every-10th
- `checkpoints/checkpoint_1196467_steps.zip` — first, every-10th
- `checkpoints/checkpoint_56813073_steps.json` — last
- `checkpoints/checkpoint_56813073_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_13165413_steps.json`
- `checkpoints/checkpoint_13165413_steps.zip`
- `checkpoints/checkpoint_25223633_steps.json`
- `checkpoints/checkpoint_25223633_steps.zip`
- `checkpoints/checkpoint_37476753_steps.json`
- `checkpoints/checkpoint_37476753_steps.zip`
- `checkpoints/checkpoint_49631583_steps.json`
- `checkpoints/checkpoint_49631583_steps.zip`

</details>

<details><summary><code>ai_v7_09_tss_bots_pubval_0708</code> — 0.175 GB freed, 8 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1196959_steps.json` — first, every-10th
- `checkpoints/checkpoint_1196959_steps.zip` — first, every-10th
- `checkpoints/checkpoint_57421191_steps.json` — last
- `checkpoints/checkpoint_57421191_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_13165365_steps.json`
- `checkpoints/checkpoint_13165365_steps.zip`
- `checkpoints/checkpoint_25222836_steps.json`
- `checkpoints/checkpoint_25222836_steps.zip`
- `checkpoints/checkpoint_38378958_steps.json`
- `checkpoints/checkpoint_38378958_steps.zip`
- `checkpoints/checkpoint_51435737_steps.json`
- `checkpoints/checkpoint_51435737_steps.zip`

</details>

<details><summary><code>ai_v7_10_tss_exploiter_fixed_0709</code> — 0.044 GB freed, 2 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1021160_steps.json` — first, every-10th
- `checkpoints/checkpoint_1021160_steps.zip` — first, every-10th
- `checkpoints/checkpoint_23350277_steps.json` — last
- `checkpoints/checkpoint_23350277_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12192087_steps.json`
- `checkpoints/checkpoint_12192087_steps.zip`

</details>

<details><summary><code>ai_v7_11_tss_exploiter_nopubval</code> — 0.087 GB freed, 4 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1026448_steps.json` — first, every-10th
- `checkpoints/checkpoint_1026448_steps.zip` — first, every-10th
- `checkpoints/checkpoint_25491946_steps.json` — last
- `checkpoints/checkpoint_25491946_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12108035_steps.json`
- `checkpoints/checkpoint_12108035_steps.zip`
- `checkpoints/checkpoint_23437994_steps.json`
- `checkpoints/checkpoint_23437994_steps.zip`

</details>

<details><summary><code>ai_v7_12_trap_exploiter_0711</code> — 0.044 GB freed, 2 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1039954_steps.json` — first, every-10th
- `checkpoints/checkpoint_1039954_steps.zip` — first, every-10th
- `checkpoints/checkpoint_15546865_steps.json` — last
- `checkpoints/checkpoint_15546865_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12406775_steps.json`
- `checkpoints/checkpoint_12406775_steps.zip`

</details>

<details><summary><code>ai_v7_13_cmpass_exploiter_0711</code> — 0.087 GB freed, 4 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1043932_steps.json` — first, every-10th
- `checkpoints/checkpoint_1043932_steps.zip` — first, every-10th
- `checkpoints/checkpoint_25536809_steps.json` — last
- `checkpoints/checkpoint_25536809_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12301857_steps.json`
- `checkpoints/checkpoint_12301857_steps.zip`
- `checkpoints/checkpoint_23463204_steps.json`
- `checkpoints/checkpoint_23463204_steps.zip`

</details>

<details><summary><code>ai_v7_14_league_capstone_0712</code> — 0.13 GB freed, 6 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_107882936_steps.json` — first, every-10th
- `checkpoints/checkpoint_107882936_steps.zip` — first, every-10th
- `checkpoints/checkpoint_148223095_steps.json` — last
- `checkpoints/checkpoint_148223095_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_117874682_steps.json`
- `checkpoints/checkpoint_117874682_steps.zip`
- `checkpoints/checkpoint_128352969_steps.json`
- `checkpoints/checkpoint_128352969_steps.zip`
- `checkpoints/checkpoint_139670117_steps.json`
- `checkpoints/checkpoint_139670117_steps.zip`

</details>

<details><summary><code>ai_v7_15_tss_exploiter_vs14_0713</code> — 0.263 GB freed, 12 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1038052_steps.json` — first, every-10th
- `checkpoints/checkpoint_1038052_steps.zip` — first, every-10th
- `checkpoints/checkpoint_74729364_steps.json` — last
- `checkpoints/checkpoint_74729364_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12242664_steps.json`
- `checkpoints/checkpoint_12242664_steps.zip`
- `checkpoints/checkpoint_23731899_steps.json`
- `checkpoints/checkpoint_23731899_steps.zip`
- `checkpoints/checkpoint_35107909_steps.json`
- `checkpoints/checkpoint_35107909_steps.zip`
- `checkpoints/checkpoint_46367828_steps.json`
- `checkpoints/checkpoint_46367828_steps.zip`
- `checkpoints/checkpoint_57436562_steps.json`
- `checkpoints/checkpoint_57436562_steps.zip`
- `checkpoints/checkpoint_68718377_steps.json`
- `checkpoints/checkpoint_68718377_steps.zip`

</details>

<details><summary><code>ai_v7_16_distill_tss_mvp_0715</code> — 0.043 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_149598552_steps.json` — first, every-10th
- `checkpoints/checkpoint_149598552_steps.zip` — first, every-10th
- `checkpoints/checkpoint_154637567_steps.json` — last
- `checkpoints/checkpoint_154637567_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>ai_v7_17_stall_exploiter_0715</code> — 0.087 GB freed, 4 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1030227_steps.json` — first, every-10th
- `checkpoints/checkpoint_1030227_steps.zip` — first, every-10th
- `checkpoints/checkpoint_27246773_steps.json` — last
- `checkpoints/checkpoint_27246773_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_11914412_steps.json`
- `checkpoints/checkpoint_11914412_steps.zip`
- `checkpoints/checkpoint_23087573_steps.json`
- `checkpoints/checkpoint_23087573_steps.zip`

</details>

<details><summary><code>ai_v7_19_combined_0716</code> — 0.13 GB freed, 3 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_159499550_steps.json` — first, every-10th
- `checkpoints/checkpoint_159499550_steps.zip` — first, every-10th
- `checkpoints/checkpoint_175523633_steps.json` — last
- `checkpoints/checkpoint_175523633_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_170014290_steps.json`
- `checkpoints/checkpoint_170014290_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v7_20_valuedistill_ab_0717</code> — 0.043 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_149598474_steps.json` — first, every-10th
- `checkpoints/checkpoint_149598474_steps.zip` — first, every-10th
- `checkpoints/checkpoint_152670073_steps.json` — last
- `checkpoints/checkpoint_152670073_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>ai_v7_21_fitnet_valuefeat_ab_0717</code> — 0.043 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_149598992_steps.json` — first, every-10th
- `checkpoints/checkpoint_149598992_steps.zip` — first, every-10th
- `checkpoints/checkpoint_159925420_steps.json` — last
- `checkpoints/checkpoint_159925420_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>ai_v8_01_zarch_film_0717</code> — 1.507 GB freed, 47 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_149411773_steps.json` — first, every-10th
- `checkpoints/checkpoint_149411773_steps.zip` — first, every-10th
- `checkpoints/checkpoint_160128178_steps.json` — every-10th
- `checkpoints/checkpoint_160128178_steps.zip` — every-10th
- `checkpoints/checkpoint_170604749_steps.json` — last, every-10th
- `checkpoints/checkpoint_170604749_steps.zip` — last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_150406238_steps.json`
- `checkpoints/checkpoint_150406238_steps.zip`
- `checkpoints/checkpoint_151398089_steps.json`
- `checkpoints/checkpoint_151398089_steps.zip`
- `checkpoints/checkpoint_152394751_steps.json`
- `checkpoints/checkpoint_152394751_steps.zip`
- `checkpoints/checkpoint_153388378_steps.json`
- `checkpoints/checkpoint_153388378_steps.zip`
- `checkpoints/checkpoint_154383978_steps.json`
- `checkpoints/checkpoint_154383978_steps.zip`
- `checkpoints/checkpoint_155376532_steps.json`
- `checkpoints/checkpoint_155376532_steps.zip`
- `checkpoints/checkpoint_156369295_steps.json`
- `checkpoints/checkpoint_156369295_steps.zip`
- `checkpoints/checkpoint_158143622_steps.json`
- `checkpoints/checkpoint_158143622_steps.zip`
- `checkpoints/checkpoint_159135144_steps.json`
- `checkpoints/checkpoint_159135144_steps.zip`
- `checkpoints/checkpoint_161119646_steps.json`
- `checkpoints/checkpoint_161119646_steps.zip`
- `checkpoints/checkpoint_162116025_steps.json`
- `checkpoints/checkpoint_162116025_steps.zip`
- `checkpoints/checkpoint_163105940_steps.json`
- `checkpoints/checkpoint_163105940_steps.zip`
- `checkpoints/checkpoint_164095413_steps.json`
- `checkpoints/checkpoint_164095413_steps.zip`
- `checkpoints/checkpoint_165079964_steps.json`
- `checkpoints/checkpoint_165079964_steps.zip`
- `checkpoints/checkpoint_166149384_steps.json`
- `checkpoints/checkpoint_166149384_steps.zip`
- `checkpoints/checkpoint_167658514_steps.json`
- `checkpoints/checkpoint_167658514_steps.zip`
- `checkpoints/checkpoint_168642371_steps.json`
- `checkpoints/checkpoint_168642371_steps.zip`
- `checkpoints/checkpoint_169621340_steps.json`
- `checkpoints/checkpoint_169621340_steps.zip`
- `eval_traces/step_168000019/snapshot.zip`
- `eval_traces/step_166000019/snapshot.zip`
- `eval_traces/step_164000012`
- `eval_traces/step_162000010`
- `eval_traces/step_160000015`
- `eval_traces/step_158000003`
- `eval_traces/step_156000013`
- `eval_traces/step_154000014`
- `eval_traces/step_152000006`
- `eval_traces/step_150000008`
- `eval_traces/step_148401357`

</details>

<details><summary><code>ai_v8_02_zarch_teampfsp_0718</code> — 0.134 GB freed, 2 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_172994305_steps.json` — first, every-10th
- `checkpoints/checkpoint_172994305_steps.zip` — first, every-10th
- `checkpoints/checkpoint_173989527_steps.json` — last
- `checkpoints/checkpoint_173989527_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_171990511/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v8_03_zarch_control_0718</code> — 5.789 GB freed, 215 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_149598621_steps.json` — first, every-10th
- `checkpoints/checkpoint_149598621_steps.zip` — first, every-10th
- `checkpoints/checkpoint_159979843_steps.json` — every-10th
- `checkpoints/checkpoint_159979843_steps.zip` — every-10th
- `checkpoints/checkpoint_170703863_steps.json` — every-10th
- `checkpoints/checkpoint_170703863_steps.zip` — every-10th
- `checkpoints/checkpoint_181490346_steps.json` — every-10th
- `checkpoints/checkpoint_181490346_steps.zip` — every-10th
- `checkpoints/checkpoint_192630904_steps.json` — every-10th
- `checkpoints/checkpoint_192630904_steps.zip` — every-10th
- `checkpoints/checkpoint_203279623_steps.json` — every-10th
- `checkpoints/checkpoint_203279623_steps.zip` — every-10th
- `checkpoints/checkpoint_213387445_steps.json` — every-10th
- `checkpoints/checkpoint_213387445_steps.zip` — every-10th
- `checkpoints/checkpoint_224282776_steps.json` — every-10th
- `checkpoints/checkpoint_224282776_steps.zip` — every-10th
- `checkpoints/checkpoint_235426415_steps.json` — every-10th
- `checkpoints/checkpoint_235426415_steps.zip` — every-10th
- `checkpoints/checkpoint_246350536_steps.json` — every-10th
- `checkpoints/checkpoint_246350536_steps.zip` — every-10th
- `checkpoints/checkpoint_257329363_steps.json` — every-10th
- `checkpoints/checkpoint_257329363_steps.zip` — every-10th
- `checkpoints/checkpoint_267612744_steps.json` — last
- `checkpoints/checkpoint_267612744_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_150795562_steps.json`
- `checkpoints/checkpoint_150795562_steps.zip`
- `checkpoints/checkpoint_151792082_steps.json`
- `checkpoints/checkpoint_151792082_steps.zip`
- `checkpoints/checkpoint_152763859_steps.json`
- `checkpoints/checkpoint_152763859_steps.zip`
- `checkpoints/checkpoint_153736836_steps.json`
- `checkpoints/checkpoint_153736836_steps.zip`
- `checkpoints/checkpoint_154712668_steps.json`
- `checkpoints/checkpoint_154712668_steps.zip`
- `checkpoints/checkpoint_155685986_steps.json`
- `checkpoints/checkpoint_155685986_steps.zip`
- `checkpoints/checkpoint_156661897_steps.json`
- `checkpoints/checkpoint_156661897_steps.zip`
- `checkpoints/checkpoint_157636321_steps.json`
- `checkpoints/checkpoint_157636321_steps.zip`
- `checkpoints/checkpoint_159005944_steps.json`
- `checkpoints/checkpoint_159005944_steps.zip`
- `checkpoints/checkpoint_160953060_steps.json`
- `checkpoints/checkpoint_160953060_steps.zip`
- `checkpoints/checkpoint_161924598_steps.json`
- `checkpoints/checkpoint_161924598_steps.zip`
- `checkpoints/checkpoint_162899305_steps.json`
- `checkpoints/checkpoint_162899305_steps.zip`
- `checkpoints/checkpoint_163869113_steps.json`
- `checkpoints/checkpoint_163869113_steps.zip`
- `checkpoints/checkpoint_164840317_steps.json`
- `checkpoints/checkpoint_164840317_steps.zip`
- `checkpoints/checkpoint_165812896_steps.json`
- `checkpoints/checkpoint_165812896_steps.zip`
- `checkpoints/checkpoint_167759303_steps.json`
- `checkpoints/checkpoint_167759303_steps.zip`
- `checkpoints/checkpoint_168736100_steps.json`
- `checkpoints/checkpoint_168736100_steps.zip`
- `checkpoints/checkpoint_169713517_steps.json`
- `checkpoints/checkpoint_169713517_steps.zip`
- `checkpoints/checkpoint_171695686_steps.json`
- `checkpoints/checkpoint_171695686_steps.zip`
- `checkpoints/checkpoint_172686395_steps.json`
- `checkpoints/checkpoint_172686395_steps.zip`
- `checkpoints/checkpoint_173679402_steps.json`
- `checkpoints/checkpoint_173679402_steps.zip`
- `checkpoints/checkpoint_174674264_steps.json`
- `checkpoints/checkpoint_174674264_steps.zip`
- `checkpoints/checkpoint_176518484_steps.json`
- `checkpoints/checkpoint_176518484_steps.zip`
- `checkpoints/checkpoint_177510998_steps.json`
- `checkpoints/checkpoint_177510998_steps.zip`
- `checkpoints/checkpoint_178506677_steps.json`
- `checkpoints/checkpoint_178506677_steps.zip`
- `checkpoints/checkpoint_179499965_steps.json`
- `checkpoints/checkpoint_179499965_steps.zip`
- `checkpoints/checkpoint_180498162_steps.json`
- `checkpoints/checkpoint_180498162_steps.zip`
- `checkpoints/checkpoint_182481877_steps.json`
- `checkpoints/checkpoint_182481877_steps.zip`
- `checkpoints/checkpoint_183473102_steps.json`
- `checkpoints/checkpoint_183473102_steps.zip`
- `checkpoints/checkpoint_185076735_steps.json`
- `checkpoints/checkpoint_185076735_steps.zip`
- `checkpoints/checkpoint_186071495_steps.json`
- `checkpoints/checkpoint_186071495_steps.zip`
- `checkpoints/checkpoint_187024505_steps.json`
- `checkpoints/checkpoint_187024505_steps.zip`
- `checkpoints/checkpoint_188002427_steps.json`
- `checkpoints/checkpoint_188002427_steps.zip`
- `checkpoints/checkpoint_189167401_steps.json`
- `checkpoints/checkpoint_189167401_steps.zip`
- `checkpoints/checkpoint_190661853_steps.json`
- `checkpoints/checkpoint_190661853_steps.zip`
- `checkpoints/checkpoint_191651059_steps.json`
- `checkpoints/checkpoint_191651059_steps.zip`
- `checkpoints/checkpoint_193604489_steps.json`
- `checkpoints/checkpoint_193604489_steps.zip`
- `checkpoints/checkpoint_194582391_steps.json`
- `checkpoints/checkpoint_194582391_steps.zip`
- `checkpoints/checkpoint_195557382_steps.json`
- `checkpoints/checkpoint_195557382_steps.zip`
- `checkpoints/checkpoint_196531370_steps.json`
- `checkpoints/checkpoint_196531370_steps.zip`
- `checkpoints/checkpoint_197502734_steps.json`
- `checkpoints/checkpoint_197502734_steps.zip`
- `checkpoints/checkpoint_199391957_steps.json`
- `checkpoints/checkpoint_199391957_steps.zip`
- `checkpoints/checkpoint_200364858_steps.json`
- `checkpoints/checkpoint_200364858_steps.zip`
- `checkpoints/checkpoint_201335910_steps.json`
- `checkpoints/checkpoint_201335910_steps.zip`
- `checkpoints/checkpoint_202308445_steps.json`
- `checkpoints/checkpoint_202308445_steps.zip`
- `checkpoints/checkpoint_204252906_steps.json`
- `checkpoints/checkpoint_204252906_steps.zip`
- `checkpoints/checkpoint_205219264_steps.json`
- `checkpoints/checkpoint_205219264_steps.zip`
- `checkpoints/checkpoint_206176070_steps.json`
- `checkpoints/checkpoint_206176070_steps.zip`
- `checkpoints/checkpoint_207642484_steps.json`
- `checkpoints/checkpoint_207642484_steps.zip`
- `checkpoints/checkpoint_208607304_steps.json`
- `checkpoints/checkpoint_208607304_steps.zip`
- `checkpoints/checkpoint_209560974_steps.json`
- `checkpoints/checkpoint_209560974_steps.zip`
- `checkpoints/checkpoint_210511234_steps.json`
- `checkpoints/checkpoint_210511234_steps.zip`
- `checkpoints/checkpoint_211466529_steps.json`
- `checkpoints/checkpoint_211466529_steps.zip`
- `checkpoints/checkpoint_212430390_steps.json`
- `checkpoints/checkpoint_212430390_steps.zip`
- `checkpoints/checkpoint_214351506_steps.json`
- `checkpoints/checkpoint_214351506_steps.zip`
- `checkpoints/checkpoint_215608896_steps.json`
- `checkpoints/checkpoint_215608896_steps.zip`
- `checkpoints/checkpoint_216576424_steps.json`
- `checkpoints/checkpoint_216576424_steps.zip`
- `checkpoints/checkpoint_218324007_steps.json`
- `checkpoints/checkpoint_218324007_steps.zip`
- `checkpoints/checkpoint_219317323_steps.json`
- `checkpoints/checkpoint_219317323_steps.zip`
- `checkpoints/checkpoint_220313851_steps.json`
- `checkpoints/checkpoint_220313851_steps.zip`
- `checkpoints/checkpoint_221303866_steps.json`
- `checkpoints/checkpoint_221303866_steps.zip`
- `checkpoints/checkpoint_222300365_steps.json`
- `checkpoints/checkpoint_222300365_steps.zip`
- `checkpoints/checkpoint_223293912_steps.json`
- `checkpoints/checkpoint_223293912_steps.zip`
- `checkpoints/checkpoint_225258000_steps.json`
- `checkpoints/checkpoint_225258000_steps.zip`
- `checkpoints/checkpoint_226873517_steps.json`
- `checkpoints/checkpoint_226873517_steps.zip`
- `checkpoints/checkpoint_227862988_steps.json`
- `checkpoints/checkpoint_227862988_steps.zip`
- `checkpoints/checkpoint_228855795_steps.json`
- `checkpoints/checkpoint_228855795_steps.zip`
- `checkpoints/checkpoint_229845658_steps.json`
- `checkpoints/checkpoint_229845658_steps.zip`
- `checkpoints/checkpoint_230839767_steps.json`
- `checkpoints/checkpoint_230839767_steps.zip`
- `checkpoints/checkpoint_231831986_steps.json`
- `checkpoints/checkpoint_231831986_steps.zip`
- `checkpoints/checkpoint_232823452_steps.json`
- `checkpoints/checkpoint_232823452_steps.zip`
- `checkpoints/checkpoint_233812731_steps.json`
- `checkpoints/checkpoint_233812731_steps.zip`
- `checkpoints/checkpoint_237182720_steps.json`
- `checkpoints/checkpoint_237182720_steps.zip`
- `checkpoints/checkpoint_238178477_steps.json`
- `checkpoints/checkpoint_238178477_steps.zip`
- `checkpoints/checkpoint_239165498_steps.json`
- `checkpoints/checkpoint_239165498_steps.zip`
- `checkpoints/checkpoint_240160922_steps.json`
- `checkpoints/checkpoint_240160922_steps.zip`
- `checkpoints/checkpoint_241223062_steps.json`
- `checkpoints/checkpoint_241223062_steps.zip`
- `checkpoints/checkpoint_242407731_steps.json`
- `checkpoints/checkpoint_242407731_steps.zip`
- `checkpoints/checkpoint_243388049_steps.json`
- `checkpoints/checkpoint_243388049_steps.zip`
- `checkpoints/checkpoint_244380939_steps.json`
- `checkpoints/checkpoint_244380939_steps.zip`
- `checkpoints/checkpoint_245361028_steps.json`
- `checkpoints/checkpoint_245361028_steps.zip`
- `checkpoints/checkpoint_247334344_steps.json`
- `checkpoints/checkpoint_247334344_steps.zip`
- `checkpoints/checkpoint_248323494_steps.json`
- `checkpoints/checkpoint_248323494_steps.zip`
- `checkpoints/checkpoint_249303948_steps.json`
- `checkpoints/checkpoint_249303948_steps.zip`
- `checkpoints/checkpoint_251375877_steps.json`
- `checkpoints/checkpoint_251375877_steps.zip`
- `checkpoints/checkpoint_252371694_steps.json`
- `checkpoints/checkpoint_252371694_steps.zip`
- `checkpoints/checkpoint_253360840_steps.json`
- `checkpoints/checkpoint_253360840_steps.zip`
- `checkpoints/checkpoint_254356874_steps.json`
- `checkpoints/checkpoint_254356874_steps.zip`
- `checkpoints/checkpoint_255346317_steps.json`
- `checkpoints/checkpoint_255346317_steps.zip`
- `checkpoints/checkpoint_256344828_steps.json`
- `checkpoints/checkpoint_256344828_steps.zip`
- `checkpoints/checkpoint_258324868_steps.json`
- `checkpoints/checkpoint_258324868_steps.zip`
- `checkpoints/checkpoint_259538409_steps.json`
- `checkpoints/checkpoint_259538409_steps.zip`
- `checkpoints/checkpoint_260536030_steps.json`
- `checkpoints/checkpoint_260536030_steps.zip`
- `checkpoints/checkpoint_261526199_steps.json`
- `checkpoints/checkpoint_261526199_steps.zip`
- `checkpoints/checkpoint_262526136_steps.json`
- `checkpoints/checkpoint_262526136_steps.zip`
- `checkpoints/checkpoint_264606845_steps.json`
- `checkpoints/checkpoint_264606845_steps.zip`
- `checkpoints/checkpoint_265605469_steps.json`
- `checkpoints/checkpoint_265605469_steps.zip`
- `checkpoints/checkpoint_266613827_steps.json`
- `checkpoints/checkpoint_266613827_steps.zip`
- `eval_traces/step_266000014/snapshot.zip`
- `eval_traces/step_264000019/snapshot.zip`
- `eval_traces/step_262000006`
- `eval_traces/step_260000012`
- `eval_traces/step_258000018`
- `eval_traces/step_256000014`
- `eval_traces/step_254000006`
- `eval_traces/step_252000012`
- `eval_traces/step_250000010`
- `eval_traces/step_248000024`
- `eval_traces/step_246000019`
- `eval_traces/step_244000009`
- `eval_traces/step_242000001`
- `eval_traces/step_240000001`
- `eval_traces/step_238000008`
- `eval_traces/step_236000002`
- `eval_traces/step_234000002`
- `eval_traces/step_232000014`
- `eval_traces/step_230000003`

</details>

<details><summary><code>ai_v8_05_semistall564_exploiter_0722</code> — 0.089 GB freed, 3 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278642638_steps.json` — first, every-10th
- `checkpoints/checkpoint_278642638_steps.zip` — first, every-10th
- `checkpoints/checkpoint_280757412_steps.json` — last
- `checkpoints/checkpoint_280757412_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_279702211_steps.json`
- `checkpoints/checkpoint_279702211_steps.zip`
- `eval_traces/step_278000005/snapshot.zip`

</details>

<details><summary><code>ai_v8_07_semistall564_scratch_0722</code> — 1.167 GB freed, 42 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1049346_steps.json` — first, every-10th
- `checkpoints/checkpoint_1049346_steps.zip` — first, every-10th
- `checkpoints/checkpoint_12137833_steps.json` — every-10th
- `checkpoints/checkpoint_12137833_steps.zip` — every-10th
- `checkpoints/checkpoint_21112628_steps.json` — last
- `checkpoints/checkpoint_21112628_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_11081568_steps.json`
- `checkpoints/checkpoint_11081568_steps.zip`
- `checkpoints/checkpoint_13189098_steps.json`
- `checkpoints/checkpoint_13189098_steps.zip`
- `checkpoints/checkpoint_14245635_steps.json`
- `checkpoints/checkpoint_14245635_steps.zip`
- `checkpoints/checkpoint_15296480_steps.json`
- `checkpoints/checkpoint_15296480_steps.zip`
- `checkpoints/checkpoint_16349041_steps.json`
- `checkpoints/checkpoint_16349041_steps.zip`
- `checkpoints/checkpoint_17399775_steps.json`
- `checkpoints/checkpoint_17399775_steps.zip`
- `checkpoints/checkpoint_18452543_steps.json`
- `checkpoints/checkpoint_18452543_steps.zip`
- `checkpoints/checkpoint_19502910_steps.json`
- `checkpoints/checkpoint_19502910_steps.zip`
- `checkpoints/checkpoint_2110004_steps.json`
- `checkpoints/checkpoint_2110004_steps.zip`
- `checkpoints/checkpoint_3184463_steps.json`
- `checkpoints/checkpoint_3184463_steps.zip`
- `checkpoints/checkpoint_4248474_steps.json`
- `checkpoints/checkpoint_4248474_steps.zip`
- `checkpoints/checkpoint_5312985_steps.json`
- `checkpoints/checkpoint_5312985_steps.zip`
- `checkpoints/checkpoint_6376742_steps.json`
- `checkpoints/checkpoint_6376742_steps.zip`
- `checkpoints/checkpoint_7447226_steps.json`
- `checkpoints/checkpoint_7447226_steps.zip`
- `checkpoints/checkpoint_8518090_steps.json`
- `checkpoints/checkpoint_8518090_steps.zip`
- `checkpoints/checkpoint_9587645_steps.json`
- `checkpoints/checkpoint_9587645_steps.zip`
- `eval_traces/step_20000020/snapshot.zip`
- `eval_traces/step_18000002/snapshot.zip`
- `eval_traces/step_16000010`
- `eval_traces/step_14000017`
- `eval_traces/step_12000018`
- `eval_traces/step_10000021`
- `eval_traces/step_8000004`
- `eval_traces/step_6000022`
- `eval_traces/step_4000021`
- `eval_traces/step_2000001`

</details>

<details><summary><code>ai_v8_08_defensive_6team_exploiter_0723</code> — 0.089 GB freed, 3 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278641713_steps.json` — first, every-10th
- `checkpoints/checkpoint_278641713_steps.zip` — first, every-10th
- `checkpoints/checkpoint_280753159_steps.json` — last
- `checkpoints/checkpoint_280753159_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_279699715_steps.json`
- `checkpoints/checkpoint_279699715_steps.zip`
- `eval_traces/step_278000014/snapshot.zip`

</details>

<details><summary><code>ai_v8_10_offense20_exploiter_0724</code> — 0.838 GB freed, 27 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278624867_steps.json` — first, every-10th
- `checkpoints/checkpoint_278624867_steps.zip` — first, every-10th
- `checkpoints/checkpoint_289605804_steps.json` — every-10th
- `checkpoints/checkpoint_289605804_steps.zip` — every-10th
- `checkpoints/checkpoint_291684967_steps.json` — last
- `checkpoints/checkpoint_291684967_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_279680027_steps.json`
- `checkpoints/checkpoint_279680027_steps.zip`
- `checkpoints/checkpoint_280727756_steps.json`
- `checkpoints/checkpoint_280727756_steps.zip`
- `checkpoints/checkpoint_281771325_steps.json`
- `checkpoints/checkpoint_281771325_steps.zip`
- `checkpoints/checkpoint_282812169_steps.json`
- `checkpoints/checkpoint_282812169_steps.zip`
- `checkpoints/checkpoint_283868151_steps.json`
- `checkpoints/checkpoint_283868151_steps.zip`
- `checkpoints/checkpoint_284903874_steps.json`
- `checkpoints/checkpoint_284903874_steps.zip`
- `checkpoints/checkpoint_285944457_steps.json`
- `checkpoints/checkpoint_285944457_steps.zip`
- `checkpoints/checkpoint_286989241_steps.json`
- `checkpoints/checkpoint_286989241_steps.zip`
- `checkpoints/checkpoint_288560353_steps.json`
- `checkpoints/checkpoint_288560353_steps.zip`
- `checkpoints/checkpoint_290644027_steps.json`
- `checkpoints/checkpoint_290644027_steps.zip`
- `eval_traces/step_290000017/snapshot.zip`
- `eval_traces/step_288000018/snapshot.zip`
- `eval_traces/step_286000010`
- `eval_traces/step_284000003`
- `eval_traces/step_282000002`
- `eval_traces/step_280000003`
- `eval_traces/step_278000015`

</details>

<details><summary><code>ai_v8_11_offense10_exploiter_0724</code> — 0.223 GB freed, 8 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278630457_steps.json` — first, every-10th
- `checkpoints/checkpoint_278630457_steps.zip` — first, every-10th
- `checkpoints/checkpoint_282840653_steps.json` — last
- `checkpoints/checkpoint_282840653_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_279689990_steps.json`
- `checkpoints/checkpoint_279689990_steps.zip`
- `checkpoints/checkpoint_280739452_steps.json`
- `checkpoints/checkpoint_280739452_steps.zip`
- `checkpoints/checkpoint_281790034_steps.json`
- `checkpoints/checkpoint_281790034_steps.zip`
- `eval_traces/step_280000018/snapshot.zip`
- `eval_traces/step_278000001/snapshot.zip`

</details>

<details><summary><code>ai_v8_12_defensive20_exploiter_0724</code> — 1.458 GB freed, 47 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278631107_steps.json` — first, every-10th
- `checkpoints/checkpoint_278631107_steps.zip` — first, every-10th
- `checkpoints/checkpoint_289605699_steps.json` — every-10th
- `checkpoints/checkpoint_289605699_steps.zip` — every-10th
- `checkpoints/checkpoint_300470594_steps.json` — last, every-10th
- `checkpoints/checkpoint_300470594_steps.zip` — last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_279680708_steps.json`
- `checkpoints/checkpoint_279680708_steps.zip`
- `checkpoints/checkpoint_280728560_steps.json`
- `checkpoints/checkpoint_280728560_steps.zip`
- `checkpoints/checkpoint_281774366_steps.json`
- `checkpoints/checkpoint_281774366_steps.zip`
- `checkpoints/checkpoint_282816532_steps.json`
- `checkpoints/checkpoint_282816532_steps.zip`
- `checkpoints/checkpoint_283867174_steps.json`
- `checkpoints/checkpoint_283867174_steps.zip`
- `checkpoints/checkpoint_284910752_steps.json`
- `checkpoints/checkpoint_284910752_steps.zip`
- `checkpoints/checkpoint_285956559_steps.json`
- `checkpoints/checkpoint_285956559_steps.zip`
- `checkpoints/checkpoint_287001289_steps.json`
- `checkpoints/checkpoint_287001289_steps.zip`
- `checkpoints/checkpoint_288556473_steps.json`
- `checkpoints/checkpoint_288556473_steps.zip`
- `checkpoints/checkpoint_290656911_steps.json`
- `checkpoints/checkpoint_290656911_steps.zip`
- `checkpoints/checkpoint_291702659_steps.json`
- `checkpoints/checkpoint_291702659_steps.zip`
- `checkpoints/checkpoint_292741537_steps.json`
- `checkpoints/checkpoint_292741537_steps.zip`
- `checkpoints/checkpoint_293790395_steps.json`
- `checkpoints/checkpoint_293790395_steps.zip`
- `checkpoints/checkpoint_294835288_steps.json`
- `checkpoints/checkpoint_294835288_steps.zip`
- `checkpoints/checkpoint_295882320_steps.json`
- `checkpoints/checkpoint_295882320_steps.zip`
- `checkpoints/checkpoint_296926462_steps.json`
- `checkpoints/checkpoint_296926462_steps.zip`
- `checkpoints/checkpoint_298380865_steps.json`
- `checkpoints/checkpoint_298380865_steps.zip`
- `checkpoints/checkpoint_299424886_steps.json`
- `checkpoints/checkpoint_299424886_steps.zip`
- `eval_traces/step_298000014/snapshot.zip`
- `eval_traces/step_296000019/snapshot.zip`
- `eval_traces/step_294000000`
- `eval_traces/step_292000006`
- `eval_traces/step_290000017`
- `eval_traces/step_288000018`
- `eval_traces/step_286000017`
- `eval_traces/step_284000001`
- `eval_traces/step_282000007`
- `eval_traces/step_280000018`
- `eval_traces/step_278000005`

</details>

<details><summary><code>ai_v8_15_retention_A_frozen_0726</code> — 1.135 GB freed, 34 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_293086053_steps.json` — first, every-10th
- `checkpoints/checkpoint_293086053_steps.zip` — first, every-10th
- `checkpoints/checkpoint_303420516_steps.json` — every-10th
- `checkpoints/checkpoint_303420516_steps.zip` — every-10th
- `checkpoints/checkpoint_308372456_steps.json` — last
- `checkpoints/checkpoint_308372456_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_294076733_steps.json`
- `checkpoints/checkpoint_294076733_steps.zip`
- `checkpoints/checkpoint_295064873_steps.json`
- `checkpoints/checkpoint_295064873_steps.zip`
- `checkpoints/checkpoint_296053718_steps.json`
- `checkpoints/checkpoint_296053718_steps.zip`
- `checkpoints/checkpoint_297046418_steps.json`
- `checkpoints/checkpoint_297046418_steps.zip`
- `checkpoints/checkpoint_298035682_steps.json`
- `checkpoints/checkpoint_298035682_steps.zip`
- `checkpoints/checkpoint_299025420_steps.json`
- `checkpoints/checkpoint_299025420_steps.zip`
- `checkpoints/checkpoint_300011478_steps.json`
- `checkpoints/checkpoint_300011478_steps.zip`
- `checkpoints/checkpoint_301440943_steps.json`
- `checkpoints/checkpoint_301440943_steps.zip`
- `checkpoints/checkpoint_302431625_steps.json`
- `checkpoints/checkpoint_302431625_steps.zip`
- `checkpoints/checkpoint_304412698_steps.json`
- `checkpoints/checkpoint_304412698_steps.zip`
- `checkpoints/checkpoint_305403690_steps.json`
- `checkpoints/checkpoint_305403690_steps.zip`
- `checkpoints/checkpoint_306393432_steps.json`
- `checkpoints/checkpoint_306393432_steps.zip`
- `checkpoints/checkpoint_307380178_steps.json`
- `checkpoints/checkpoint_307380178_steps.zip`
- `eval_traces/step_306000001/snapshot.zip`
- `eval_traces/step_304000005/snapshot.zip`
- `eval_traces/step_302000004`
- `eval_traces/step_300000002`
- `eval_traces/step_298000016`
- `eval_traces/step_296000023`
- `eval_traces/step_294000004`
- `snapshots`

</details>

<details><summary><code>ai_v8_16_def20_lut_0726</code> — 0.869 GB freed, 27 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278639657_steps.json` — first, every-10th
- `checkpoints/checkpoint_278639657_steps.zip` — first, every-10th
- `checkpoints/checkpoint_289528972_steps.json` — every-10th
- `checkpoints/checkpoint_289528972_steps.zip` — every-10th
- `checkpoints/checkpoint_291642709_steps.json` — last
- `checkpoints/checkpoint_291642709_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_279701687_steps.json`
- `checkpoints/checkpoint_279701687_steps.zip`
- `checkpoints/checkpoint_280756136_steps.json`
- `checkpoints/checkpoint_280756136_steps.zip`
- `checkpoints/checkpoint_281815161_steps.json`
- `checkpoints/checkpoint_281815161_steps.zip`
- `checkpoints/checkpoint_282870857_steps.json`
- `checkpoints/checkpoint_282870857_steps.zip`
- `checkpoints/checkpoint_283928693_steps.json`
- `checkpoints/checkpoint_283928693_steps.zip`
- `checkpoints/checkpoint_284982492_steps.json`
- `checkpoints/checkpoint_284982492_steps.zip`
- `checkpoints/checkpoint_286041340_steps.json`
- `checkpoints/checkpoint_286041340_steps.zip`
- `checkpoints/checkpoint_287098071_steps.json`
- `checkpoints/checkpoint_287098071_steps.zip`
- `checkpoints/checkpoint_288468591_steps.json`
- `checkpoints/checkpoint_288468591_steps.zip`
- `checkpoints/checkpoint_290585165_steps.json`
- `checkpoints/checkpoint_290585165_steps.zip`
- `eval_traces/step_290000009/snapshot.zip`
- `eval_traces/step_288000012/snapshot.zip`
- `eval_traces/step_286000013`
- `eval_traces/step_284000019`
- `eval_traces/step_282000004`
- `eval_traces/step_280000003`
- `eval_traces/step_278000003`

</details>

<details><summary><code>ai_v8_17_rand20_nolut_0726</code> — 0.872 GB freed, 27 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278635878_steps.json` — first, every-10th
- `checkpoints/checkpoint_278635878_steps.zip` — first, every-10th
- `checkpoints/checkpoint_289428098_steps.json` — every-10th
- `checkpoints/checkpoint_289428098_steps.zip` — every-10th
- `checkpoints/checkpoint_291540051_steps.json` — last
- `checkpoints/checkpoint_291540051_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_279694696_steps.json`
- `checkpoints/checkpoint_279694696_steps.zip`
- `checkpoints/checkpoint_280746936_steps.json`
- `checkpoints/checkpoint_280746936_steps.zip`
- `checkpoints/checkpoint_281803818_steps.json`
- `checkpoints/checkpoint_281803818_steps.zip`
- `checkpoints/checkpoint_282855229_steps.json`
- `checkpoints/checkpoint_282855229_steps.zip`
- `checkpoints/checkpoint_283912087_steps.json`
- `checkpoints/checkpoint_283912087_steps.zip`
- `checkpoints/checkpoint_284963472_steps.json`
- `checkpoints/checkpoint_284963472_steps.zip`
- `checkpoints/checkpoint_286021233_steps.json`
- `checkpoints/checkpoint_286021233_steps.zip`
- `checkpoints/checkpoint_287071196_steps.json`
- `checkpoints/checkpoint_287071196_steps.zip`
- `checkpoints/checkpoint_288370378_steps.json`
- `checkpoints/checkpoint_288370378_steps.zip`
- `checkpoints/checkpoint_290482604_steps.json`
- `checkpoints/checkpoint_290482604_steps.zip`
- `eval_traces/step_290000004/snapshot.zip`
- `eval_traces/step_288000010/snapshot.zip`
- `eval_traces/step_286000018`
- `eval_traces/step_284000010`
- `eval_traces/step_282000005`
- `eval_traces/step_280000017`
- `eval_traces/step_278000015`

</details>

<details><summary><code>ai_v8_18_rand20_lut_0726</code> — 0.851 GB freed, 27 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278638971_steps.json` — first, every-10th
- `checkpoints/checkpoint_278638971_steps.zip` — first, every-10th
- `checkpoints/checkpoint_289536403_steps.json` — every-10th
- `checkpoints/checkpoint_289536403_steps.zip` — every-10th
- `checkpoints/checkpoint_291654679_steps.json` — last
- `checkpoints/checkpoint_291654679_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_279703135_steps.json`
- `checkpoints/checkpoint_279703135_steps.zip`
- `checkpoints/checkpoint_280762233_steps.json`
- `checkpoints/checkpoint_280762233_steps.zip`
- `checkpoints/checkpoint_281820566_steps.json`
- `checkpoints/checkpoint_281820566_steps.zip`
- `checkpoints/checkpoint_282881457_steps.json`
- `checkpoints/checkpoint_282881457_steps.zip`
- `checkpoints/checkpoint_283944497_steps.json`
- `checkpoints/checkpoint_283944497_steps.zip`
- `checkpoints/checkpoint_285002073_steps.json`
- `checkpoints/checkpoint_285002073_steps.zip`
- `checkpoints/checkpoint_286060764_steps.json`
- `checkpoints/checkpoint_286060764_steps.zip`
- `checkpoints/checkpoint_287118748_steps.json`
- `checkpoints/checkpoint_287118748_steps.zip`
- `checkpoints/checkpoint_288472741_steps.json`
- `checkpoints/checkpoint_288472741_steps.zip`
- `checkpoints/checkpoint_290595934_steps.json`
- `checkpoints/checkpoint_290595934_steps.zip`
- `eval_traces/step_290000010/snapshot.zip`
- `eval_traces/step_288000017/snapshot.zip`
- `eval_traces/step_286000027`
- `eval_traces/step_284000001`
- `eval_traces/step_282000018`
- `eval_traces/step_280000005`
- `eval_traces/step_278000014`

</details>

<details><summary><code>ai_v8_19_def20_lut_zeroinit_0727</code> — 0.856 GB freed, 27 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278671536_steps.json` — first, every-10th
- `checkpoints/checkpoint_278671536_steps.zip` — first, every-10th
- `checkpoints/checkpoint_290083231_steps.json` — every-10th
- `checkpoints/checkpoint_290083231_steps.zip` — every-10th
- `checkpoints/checkpoint_292189835_steps.json` — last
- `checkpoints/checkpoint_292189835_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_279771767_steps.json`
- `checkpoints/checkpoint_279771767_steps.zip`
- `checkpoints/checkpoint_280863504_steps.json`
- `checkpoints/checkpoint_280863504_steps.zip`
- `checkpoints/checkpoint_281961296_steps.json`
- `checkpoints/checkpoint_281961296_steps.zip`
- `checkpoints/checkpoint_283052446_steps.json`
- `checkpoints/checkpoint_283052446_steps.zip`
- `checkpoints/checkpoint_284149040_steps.json`
- `checkpoints/checkpoint_284149040_steps.zip`
- `checkpoints/checkpoint_285241796_steps.json`
- `checkpoints/checkpoint_285241796_steps.zip`
- `checkpoints/checkpoint_286328916_steps.json`
- `checkpoints/checkpoint_286328916_steps.zip`
- `checkpoints/checkpoint_287976324_steps.json`
- `checkpoints/checkpoint_287976324_steps.zip`
- `checkpoints/checkpoint_289028629_steps.json`
- `checkpoints/checkpoint_289028629_steps.zip`
- `checkpoints/checkpoint_291139360_steps.json`
- `checkpoints/checkpoint_291139360_steps.zip`
- `eval_traces/step_290000010/snapshot.zip`
- `eval_traces/step_288000001/snapshot.zip`
- `eval_traces/step_286000012`
- `eval_traces/step_284000003`
- `eval_traces/step_282000007`
- `eval_traces/step_280000007`
- `eval_traces/step_278000009`

</details>

<details><summary><code>ai_v8_20_rand10_nolut_0727</code> — 0.864 GB freed, 27 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278635763_steps.json` — first, every-10th
- `checkpoints/checkpoint_278635763_steps.zip` — first, every-10th
- `checkpoints/checkpoint_289520500_steps.json` — every-10th
- `checkpoints/checkpoint_289520500_steps.zip` — every-10th
- `checkpoints/checkpoint_291624819_steps.json` — last
- `checkpoints/checkpoint_291624819_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_279697593_steps.json`
- `checkpoints/checkpoint_279697593_steps.zip`
- `checkpoints/checkpoint_280752237_steps.json`
- `checkpoints/checkpoint_280752237_steps.zip`
- `checkpoints/checkpoint_281809621_steps.json`
- `checkpoints/checkpoint_281809621_steps.zip`
- `checkpoints/checkpoint_282865142_steps.json`
- `checkpoints/checkpoint_282865142_steps.zip`
- `checkpoints/checkpoint_283922996_steps.json`
- `checkpoints/checkpoint_283922996_steps.zip`
- `checkpoints/checkpoint_284975899_steps.json`
- `checkpoints/checkpoint_284975899_steps.zip`
- `checkpoints/checkpoint_286034337_steps.json`
- `checkpoints/checkpoint_286034337_steps.zip`
- `checkpoints/checkpoint_287087329_steps.json`
- `checkpoints/checkpoint_287087329_steps.zip`
- `checkpoints/checkpoint_288466155_steps.json`
- `checkpoints/checkpoint_288466155_steps.zip`
- `checkpoints/checkpoint_290568792_steps.json`
- `checkpoints/checkpoint_290568792_steps.zip`
- `eval_traces/step_290000014/snapshot.zip`
- `eval_traces/step_288000006/snapshot.zip`
- `eval_traces/step_286000006`
- `eval_traces/step_284000012`
- `eval_traces/step_282000001`
- `eval_traces/step_280000001`
- `eval_traces/step_278000015`

</details>

<details><summary><code>ai_v9_09_gen8_beliefs_threat_inject_0811</code> — 1.629 GB freed, 23 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_2400000_steps.json` — first, every-10th
- `checkpoints/checkpoint_2400000_steps.zip` — first, every-10th
- `checkpoints/checkpoint_25599744_steps.json` — last
- `checkpoints/checkpoint_25599744_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10362624_steps.json`
- `checkpoints/checkpoint_10362624_steps.zip`
- `checkpoints/checkpoint_14196480_steps.json`
- `checkpoints/checkpoint_14196480_steps.zip`
- `checkpoints/checkpoint_18030336_steps.json`
- `checkpoints/checkpoint_18030336_steps.zip`
- `checkpoints/checkpoint_21765888_steps.json`
- `checkpoints/checkpoint_21765888_steps.zip`
- `checkpoints/checkpoint_6430464_steps.json`
- `checkpoints/checkpoint_6430464_steps.zip`
- `eval_traces/step_24000000/snapshot.zip`
- `eval_traces/step_22000032/snapshot.zip`
- `eval_traces/step_20000016`
- `eval_traces/step_18000000`
- `eval_traces/step_16000032`
- `eval_traces/step_14000016`
- `eval_traces/step_12000000`
- `eval_traces/step_10000032`
- `eval_traces/step_8000016`
- `eval_traces/step_6000000`
- `eval_traces/step_4000032`
- `eval_traces/step_2000016`
- `snapshots`

</details>

<details><summary><code>ai_v9_101_R5F09_0831</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_103_R5F11_0831</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_105_R5F13_0831</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_107_R5F15_0831</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_108_R5F16_0831</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_109_R5F17_0831</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_10_gen9_intent_distcritic_0813</code> — 0.931 GB freed, 22 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_2400000_steps.json` — first, every-10th
- `checkpoints/checkpoint_2400000_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26386176_steps.json` — last
- `checkpoints/checkpoint_26386176_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10657536_steps.json`
- `checkpoints/checkpoint_10657536_steps.zip`
- `checkpoints/checkpoint_14491392_steps.json`
- `checkpoints/checkpoint_14491392_steps.zip`
- `checkpoints/checkpoint_18423552_steps.json`
- `checkpoints/checkpoint_18423552_steps.zip`
- `checkpoints/checkpoint_22454016_steps.json`
- `checkpoints/checkpoint_22454016_steps.zip`
- `checkpoints/checkpoint_6725376_steps.json`
- `checkpoints/checkpoint_6725376_steps.zip`
- `eval_traces/step_24000000/snapshot.zip`
- `eval_traces/step_22000032/snapshot.zip`
- `eval_traces/step_20000016`
- `eval_traces/step_18000000`
- `eval_traces/step_16000032`
- `eval_traces/step_14000016`
- `eval_traces/step_12000000`
- `eval_traces/step_10000032`
- `eval_traces/step_8000016`
- `eval_traces/step_6000000`
- `eval_traces/step_4000032`
- `eval_traces/step_2000016`

</details>

<details><summary><code>ai_v9_110_R5F18_0831</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_111_R5F19_0831</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_11_gen10_intentfull_compiled_0814</code> — 0.041 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_2400000_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_2400000_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>ai_v9_12_gen10_t0prior_0814</code> — 1.541 GB freed, 25 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_23182848_steps.json` — last
- `checkpoints/checkpoint_23182848_steps.zip` — last
- `checkpoints/checkpoint_2400000_steps.json` — first, every-10th
- `checkpoints/checkpoint_2400000_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_11091456_steps.json`
- `checkpoints/checkpoint_11091456_steps.zip`
- `checkpoints/checkpoint_14786304_steps.json`
- `checkpoints/checkpoint_14786304_steps.zip`
- `checkpoints/checkpoint_17186304_steps.json`
- `checkpoints/checkpoint_17186304_steps.zip`
- `checkpoints/checkpoint_20782848_steps.json`
- `checkpoints/checkpoint_20782848_steps.zip`
- `checkpoints/checkpoint_4800000_steps.json`
- `checkpoints/checkpoint_4800000_steps.zip`
- `checkpoints/checkpoint_8691456_steps.json`
- `checkpoints/checkpoint_8691456_steps.zip`
- `eval_traces/step_24000000/snapshot.zip`
- `eval_traces/step_22000032/snapshot.zip`
- `eval_traces/step_20000016`
- `eval_traces/step_18000000`
- `eval_traces/step_16000032`
- `eval_traces/step_14000016`
- `eval_traces/step_12000000`
- `eval_traces/step_10000032`
- `eval_traces/step_8000016`
- `eval_traces/step_6000000`
- `eval_traces/step_4000032`
- `eval_traces/step_2000016`
- `snapshots`

</details>

<details><summary><code>ai_v9_13_gen11_labelonly_winprob_0815</code> — 1.49 GB freed, 24 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_23379456_steps.json` — last
- `checkpoints/checkpoint_23379456_steps.zip` — last
- `checkpoints/checkpoint_2400000_steps.json` — first, every-10th
- `checkpoints/checkpoint_2400000_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_11484672_steps.json`
- `checkpoints/checkpoint_11484672_steps.zip`
- `checkpoints/checkpoint_15081216_steps.json`
- `checkpoints/checkpoint_15081216_steps.zip`
- `checkpoints/checkpoint_17481216_steps.json`
- `checkpoints/checkpoint_17481216_steps.zip`
- `checkpoints/checkpoint_20979456_steps.json`
- `checkpoints/checkpoint_20979456_steps.zip`
- `checkpoints/checkpoint_4800000_steps.json`
- `checkpoints/checkpoint_4800000_steps.zip`
- `checkpoints/checkpoint_9084672_steps.json`
- `checkpoints/checkpoint_9084672_steps.zip`
- `eval_traces/step_22000032/snapshot.zip`
- `eval_traces/step_20000016/snapshot.zip`
- `eval_traces/step_18000000`
- `eval_traces/step_16000032`
- `eval_traces/step_14000016`
- `eval_traces/step_12000000`
- `eval_traces/step_10000032`
- `eval_traces/step_8000016`
- `eval_traces/step_6000000`
- `eval_traces/step_4000032`
- `eval_traces/step_2000016`
- `snapshots`

</details>

<details><summary><code>ai_v9_14_gen12_h_entitypool_shaping_0816</code> — 1.11 GB freed, 25 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_2400000_steps.json` — first, every-10th
- `checkpoints/checkpoint_2400000_steps.zip` — first, every-10th
- `checkpoints/checkpoint_24715008_steps.json` — last
- `checkpoints/checkpoint_24715008_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10993152_steps.json`
- `checkpoints/checkpoint_10993152_steps.zip`
- `checkpoints/checkpoint_14196480_steps.json`
- `checkpoints/checkpoint_14196480_steps.zip`
- `checkpoints/checkpoint_16596480_steps.json`
- `checkpoints/checkpoint_16596480_steps.zip`
- `checkpoints/checkpoint_19504896_steps.json`
- `checkpoints/checkpoint_19504896_steps.zip`
- `checkpoints/checkpoint_21904896_steps.json`
- `checkpoints/checkpoint_21904896_steps.zip`
- `checkpoints/checkpoint_4800000_steps.json`
- `checkpoints/checkpoint_4800000_steps.zip`
- `checkpoints/checkpoint_8593152_steps.json`
- `checkpoints/checkpoint_8593152_steps.zip`
- `eval_traces/step_22000032/snapshot.zip`
- `eval_traces/step_20000016/snapshot.zip`
- `eval_traces/step_18000000`
- `eval_traces/step_16000032`
- `eval_traces/step_14000016`
- `eval_traces/step_12000000`
- `eval_traces/step_10000032`
- `eval_traces/step_8000016`
- `eval_traces/step_6000000`
- `eval_traces/step_4000032`
- `eval_traces/step_2000016`

</details>

<details><summary><code>ai_v9_15_gen13_hb_events_stack_0817</code> — 0.947 GB freed, 19 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_21765888_steps.json` — last
- `checkpoints/checkpoint_21765888_steps.zip` — last
- `checkpoints/checkpoint_2400000_steps.json` — first, every-10th
- `checkpoints/checkpoint_2400000_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_12623616_steps.json`
- `checkpoints/checkpoint_12623616_steps.zip`
- `checkpoints/checkpoint_17243904_steps.json`
- `checkpoints/checkpoint_17243904_steps.zip`
- `checkpoints/checkpoint_4800000_steps.json`
- `checkpoints/checkpoint_4800000_steps.zip`
- `checkpoints/checkpoint_7905024_steps.json`
- `checkpoints/checkpoint_7905024_steps.zip`
- `eval_traces/step_22000032/snapshot.zip`
- `eval_traces/step_20000016/snapshot.zip`
- `eval_traces/step_18000000`
- `eval_traces/step_16000032`
- `eval_traces/step_14000016`
- `eval_traces/step_12000000`
- `eval_traces/step_10000032`
- `eval_traces/step_8000016`
- `eval_traces/step_6000000`
- `eval_traces/step_4000032`
- `eval_traces/step_2000016`

</details>

<details><summary><code>ai_v9_16_gen14_framedel_v91_0817</code> — 1.087 GB freed, 25 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_2400000_steps.json` — first, every-10th
- `checkpoints/checkpoint_2400000_steps.zip` — first, every-10th
- `checkpoints/checkpoint_24813312_steps.json` — last
- `checkpoints/checkpoint_24813312_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10894848_steps.json`
- `checkpoints/checkpoint_10894848_steps.zip`
- `checkpoints/checkpoint_13999872_steps.json`
- `checkpoints/checkpoint_13999872_steps.zip`
- `checkpoints/checkpoint_16399872_steps.json`
- `checkpoints/checkpoint_16399872_steps.zip`
- `checkpoints/checkpoint_19406592_steps.json`
- `checkpoints/checkpoint_19406592_steps.zip`
- `checkpoints/checkpoint_21806592_steps.json`
- `checkpoints/checkpoint_21806592_steps.zip`
- `checkpoints/checkpoint_4800000_steps.json`
- `checkpoints/checkpoint_4800000_steps.zip`
- `checkpoints/checkpoint_8494848_steps.json`
- `checkpoints/checkpoint_8494848_steps.zip`
- `eval_traces/step_22000032/snapshot.zip`
- `eval_traces/step_20000016/snapshot.zip`
- `eval_traces/step_18000000`
- `eval_traces/step_16000032`
- `eval_traces/step_14000016`
- `eval_traces/step_12000000`
- `eval_traces/step_10000032`
- `eval_traces/step_8000016`
- `eval_traces/step_6000000`
- `eval_traces/step_4000032`
- `eval_traces/step_2000016`

</details>

<details><summary><code>ai_v9_172_G1SHORT_0905</code> — 0.837 GB freed, 14 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_28615200_steps.json` — first, every-10th
- `checkpoints/checkpoint_28615200_steps.zip` — first, every-10th
- `checkpoints/checkpoint_32474544_steps.json` — last
- `checkpoints/checkpoint_32474544_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_29115216_steps.json`
- `checkpoints/checkpoint_29115216_steps.zip`
- `checkpoints/checkpoint_29615232_steps.json`
- `checkpoints/checkpoint_29615232_steps.zip`
- `checkpoints/checkpoint_30115248_steps.json`
- `checkpoints/checkpoint_30115248_steps.zip`
- `checkpoints/checkpoint_30974496_steps.json`
- `checkpoints/checkpoint_30974496_steps.zip`
- `checkpoints/checkpoint_31474512_steps.json`
- `checkpoints/checkpoint_31474512_steps.zip`
- `checkpoints/checkpoint_31974528_steps.json`
- `checkpoints/checkpoint_31974528_steps.zip`
- `eval_traces/step_30000000/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_17_tdaux_control_0818</code> — 0.084 GB freed, 2 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_27467520_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_27467520_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_26000016/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_17_tdaux_lam1_0818</code> — 0.126 GB freed, 2 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_27467520_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_27467520_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_26000016/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_17_tdaux_lam3_0818</code> — 0.042 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_27467520_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_27467520_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_18_gen15_v8rewards_0818</code> — 0.946 GB freed, 23 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_22101504_steps.json` — last
- `checkpoints/checkpoint_22101504_steps.zip` — last
- `checkpoints/checkpoint_2400000_steps.json` — first, every-10th
- `checkpoints/checkpoint_2400000_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10796544_steps.json`
- `checkpoints/checkpoint_10796544_steps.zip`
- `checkpoints/checkpoint_13999872_steps.json`
- `checkpoints/checkpoint_13999872_steps.zip`
- `checkpoints/checkpoint_16399872_steps.json`
- `checkpoints/checkpoint_16399872_steps.zip`
- `checkpoints/checkpoint_19701504_steps.json`
- `checkpoints/checkpoint_19701504_steps.zip`
- `checkpoints/checkpoint_4800000_steps.json`
- `checkpoints/checkpoint_4800000_steps.zip`
- `checkpoints/checkpoint_8396544_steps.json`
- `checkpoints/checkpoint_8396544_steps.zip`
- `eval_traces/step_22000032/snapshot.zip`
- `eval_traces/step_20000016/snapshot.zip`
- `eval_traces/step_18000000`
- `eval_traces/step_16000032`
- `eval_traces/step_14000016`
- `eval_traces/step_12000000`
- `eval_traces/step_10000032`
- `eval_traces/step_8000016`
- `eval_traces/step_6000000`
- `eval_traces/step_4000032`
- `eval_traces/step_2000016`

</details>

<details><summary><code>ai_v9_19_gen16_mechanics_0819</code> — 0.853 GB freed, 23 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_22396416_steps.json` — last
- `checkpoints/checkpoint_22396416_steps.zip` — last
- `checkpoints/checkpoint_2400000_steps.json` — first, every-10th
- `checkpoints/checkpoint_2400000_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_10993152_steps.json`
- `checkpoints/checkpoint_10993152_steps.zip`
- `checkpoints/checkpoint_14393088_steps.json`
- `checkpoints/checkpoint_14393088_steps.zip`
- `checkpoints/checkpoint_16793088_steps.json`
- `checkpoints/checkpoint_16793088_steps.zip`
- `checkpoints/checkpoint_19996416_steps.json`
- `checkpoints/checkpoint_19996416_steps.zip`
- `checkpoints/checkpoint_4800000_steps.json`
- `checkpoints/checkpoint_4800000_steps.zip`
- `checkpoints/checkpoint_8593152_steps.json`
- `checkpoints/checkpoint_8593152_steps.zip`
- `eval_traces/step_22000032/snapshot.zip`
- `eval_traces/step_20000016/snapshot.zip`
- `eval_traces/step_18000000`
- `eval_traces/step_16000032`
- `eval_traces/step_14000016`
- `eval_traces/step_12000000`
- `eval_traces/step_10000032`
- `eval_traces/step_8000016`
- `eval_traces/step_6000000`
- `eval_traces/step_4000032`
- `eval_traces/step_2000016`

</details>

<details><summary><code>ai_v9_20_tdaux_rung2_lam00_0820</code> — 0.533 GB freed, 2 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_27467520_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_27467520_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_26000016/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_20_tdaux_rung2_lam10_0820</code> — 0.533 GB freed, 2 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_27467520_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_27467520_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_26000016/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_20_tdaux_rung2_lam30_0820</code> — 0.533 GB freed, 2 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_27467520_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_27467520_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_26000016/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_21_gen17_pfspoff_0820</code> — 0.845 GB freed, 23 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_22887936_steps.json` — last
- `checkpoints/checkpoint_22887936_steps.zip` — last
- `checkpoints/checkpoint_2400000_steps.json` — first, every-10th
- `checkpoints/checkpoint_2400000_steps.zip` — first, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_11484672_steps.json`
- `checkpoints/checkpoint_11484672_steps.zip`
- `checkpoints/checkpoint_14786304_steps.json`
- `checkpoints/checkpoint_14786304_steps.zip`
- `checkpoints/checkpoint_17186304_steps.json`
- `checkpoints/checkpoint_17186304_steps.zip`
- `checkpoints/checkpoint_20487936_steps.json`
- `checkpoints/checkpoint_20487936_steps.zip`
- `checkpoints/checkpoint_4800000_steps.json`
- `checkpoints/checkpoint_4800000_steps.zip`
- `checkpoints/checkpoint_9084672_steps.json`
- `checkpoints/checkpoint_9084672_steps.zip`
- `eval_traces/step_22000032/snapshot.zip`
- `eval_traces/step_20000016/snapshot.zip`
- `eval_traces/step_18000000`
- `eval_traces/step_16000032`
- `eval_traces/step_14000016`
- `eval_traces/step_12000000`
- `eval_traces/step_10000032`
- `eval_traces/step_8000016`
- `eval_traces/step_6000000`
- `eval_traces/step_4000032`
- `eval_traces/step_2000016`

</details>

<details><summary><code>ai_v9_22_E1_substrate_on_0821</code> — 0.321 GB freed, 7 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_27467520_steps.json` — first, every-10th
- `checkpoints/checkpoint_27467520_steps.zip` — first, every-10th
- `checkpoints/checkpoint_32914944_steps.json` — last
- `checkpoints/checkpoint_32914944_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_30514944_steps.json`
- `checkpoints/checkpoint_30514944_steps.zip`
- `eval_traces/step_31500000/snapshot.zip`
- `eval_traces/step_30000000/snapshot.zip`
- `eval_traces/step_28500000`
- `eval_traces/step_28000032`
- `eval_traces/step_26000016`

</details>

<details><summary><code>ai_v9_23_E2_substrate_on_0822</code> — 1.0 GB freed, 8 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_27467520_steps.json` — first, every-10th
- `checkpoints/checkpoint_27467520_steps.zip` — first, every-10th
- `checkpoints/checkpoint_32874240_steps.json` — last
- `checkpoints/checkpoint_32874240_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_29867520_steps.json`
- `checkpoints/checkpoint_29867520_steps.zip`
- `eval_traces/step_31500000/snapshot.zip`
- `eval_traces/step_30000000/snapshot.zip`
- `eval_traces/step_28500000`
- `eval_traces/step_27000000`
- `eval_traces/step_25500000`
- `snapshots`

</details>

<details><summary><code>ai_v9_24_E3_substrate_on_0822</code> — 1.017 GB freed, 8 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_27467520_steps.json` — first, every-10th
- `checkpoints/checkpoint_27467520_steps.zip` — first, every-10th
- `checkpoints/checkpoint_32775936_steps.json` — last
- `checkpoints/checkpoint_32775936_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_29867520_steps.json`
- `checkpoints/checkpoint_29867520_steps.zip`
- `eval_traces/step_31500000/snapshot.zip`
- `eval_traces/step_30000000/snapshot.zip`
- `eval_traces/step_28500000`
- `eval_traces/step_27000000`
- `eval_traces/step_25500000`
- `snapshots`

</details>

<details><summary><code>ai_v9_25_E4_baitbot_0822</code> — 0.323 GB freed, 7 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_36511488_steps.json` — first, every-10th
- `checkpoints/checkpoint_36511488_steps.zip` — first, every-10th
- `checkpoints/checkpoint_41721600_steps.json` — last
- `checkpoints/checkpoint_41721600_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_38911488_steps.json`
- `checkpoints/checkpoint_38911488_steps.zip`
- `eval_traces/step_40500000/snapshot.zip`
- `eval_traces/step_39000000/snapshot.zip`
- `eval_traces/step_37500000`
- `eval_traces/step_36000000`
- `eval_traces/step_34500000`

</details>

<details><summary><code>ai_v9_27_extremedial_probe_0823</code> — 0.036 GB freed, 1 entries deleted</summary>

**KEEP**

- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_43500000/snapshot.zip`

</details>

<details><summary><code>ai_v9_30_rev1_exploit_0824</code> — 0.364 GB freed, 20 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_27017760_steps.json` — last
- `checkpoints/checkpoint_27017760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`

</details>

<details><summary><code>ai_v9_32_tock1b_rain_0824</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_34_tick1_0824</code> — 2.308 GB freed, 120 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28217760_steps.json` — every-10th
- `checkpoints/checkpoint_28217760_steps.zip` — every-10th
- `checkpoints/checkpoint_29739744_steps.json` — every-10th
- `checkpoints/checkpoint_29739744_steps.zip` — every-10th
- `checkpoints/checkpoint_31239744_steps.json` — every-10th
- `checkpoints/checkpoint_31239744_steps.zip` — every-10th
- `checkpoints/checkpoint_32739744_steps.json` — every-10th
- `checkpoints/checkpoint_32739744_steps.zip` — every-10th
- `checkpoints/checkpoint_34318512_steps.json` — every-10th
- `checkpoints/checkpoint_34318512_steps.zip` — every-10th
- `checkpoints/checkpoint_35068512_steps.json` — last
- `checkpoints/checkpoint_35068512_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `checkpoints/checkpoint_28067760_steps.json`
- `checkpoints/checkpoint_28067760_steps.zip`
- `checkpoints/checkpoint_28367760_steps.json`
- `checkpoints/checkpoint_28367760_steps.zip`
- `checkpoints/checkpoint_28517760_steps.json`
- `checkpoints/checkpoint_28517760_steps.zip`
- `checkpoints/checkpoint_28667760_steps.json`
- `checkpoints/checkpoint_28667760_steps.zip`
- `checkpoints/checkpoint_28817760_steps.json`
- `checkpoints/checkpoint_28817760_steps.zip`
- `checkpoints/checkpoint_28967760_steps.json`
- `checkpoints/checkpoint_28967760_steps.zip`
- `checkpoints/checkpoint_29117760_steps.json`
- `checkpoints/checkpoint_29117760_steps.zip`
- `checkpoints/checkpoint_29267760_steps.json`
- `checkpoints/checkpoint_29267760_steps.zip`
- `checkpoints/checkpoint_29417760_steps.json`
- `checkpoints/checkpoint_29417760_steps.zip`
- `checkpoints/checkpoint_29567760_steps.json`
- `checkpoints/checkpoint_29567760_steps.zip`
- `checkpoints/checkpoint_29889744_steps.json`
- `checkpoints/checkpoint_29889744_steps.zip`
- `checkpoints/checkpoint_30039744_steps.json`
- `checkpoints/checkpoint_30039744_steps.zip`
- `checkpoints/checkpoint_30189744_steps.json`
- `checkpoints/checkpoint_30189744_steps.zip`
- `checkpoints/checkpoint_30339744_steps.json`
- `checkpoints/checkpoint_30339744_steps.zip`
- `checkpoints/checkpoint_30489744_steps.json`
- `checkpoints/checkpoint_30489744_steps.zip`
- `checkpoints/checkpoint_30639744_steps.json`
- `checkpoints/checkpoint_30639744_steps.zip`
- `checkpoints/checkpoint_30789744_steps.json`
- `checkpoints/checkpoint_30789744_steps.zip`
- `checkpoints/checkpoint_30939744_steps.json`
- `checkpoints/checkpoint_30939744_steps.zip`
- `checkpoints/checkpoint_31089744_steps.json`
- `checkpoints/checkpoint_31089744_steps.zip`
- `checkpoints/checkpoint_31389744_steps.json`
- `checkpoints/checkpoint_31389744_steps.zip`
- `checkpoints/checkpoint_31539744_steps.json`
- `checkpoints/checkpoint_31539744_steps.zip`
- `checkpoints/checkpoint_31689744_steps.json`
- `checkpoints/checkpoint_31689744_steps.zip`
- `checkpoints/checkpoint_31839744_steps.json`
- `checkpoints/checkpoint_31839744_steps.zip`
- `checkpoints/checkpoint_31989744_steps.json`
- `checkpoints/checkpoint_31989744_steps.zip`
- `checkpoints/checkpoint_32139744_steps.json`
- `checkpoints/checkpoint_32139744_steps.zip`
- `checkpoints/checkpoint_32289744_steps.json`
- `checkpoints/checkpoint_32289744_steps.zip`
- `checkpoints/checkpoint_32439744_steps.json`
- `checkpoints/checkpoint_32439744_steps.zip`
- `checkpoints/checkpoint_32589744_steps.json`
- `checkpoints/checkpoint_32589744_steps.zip`
- `checkpoints/checkpoint_32889744_steps.json`
- `checkpoints/checkpoint_32889744_steps.zip`
- `checkpoints/checkpoint_33039744_steps.json`
- `checkpoints/checkpoint_33039744_steps.zip`
- `checkpoints/checkpoint_33189744_steps.json`
- `checkpoints/checkpoint_33189744_steps.zip`
- `checkpoints/checkpoint_33339744_steps.json`
- `checkpoints/checkpoint_33339744_steps.zip`
- `checkpoints/checkpoint_33489744_steps.json`
- `checkpoints/checkpoint_33489744_steps.zip`
- `checkpoints/checkpoint_33639744_steps.json`
- `checkpoints/checkpoint_33639744_steps.zip`
- `checkpoints/checkpoint_33868512_steps.json`
- `checkpoints/checkpoint_33868512_steps.zip`
- `checkpoints/checkpoint_34018512_steps.json`
- `checkpoints/checkpoint_34018512_steps.zip`
- `checkpoints/checkpoint_34168512_steps.json`
- `checkpoints/checkpoint_34168512_steps.zip`
- `checkpoints/checkpoint_34468512_steps.json`
- `checkpoints/checkpoint_34468512_steps.zip`
- `checkpoints/checkpoint_34618512_steps.json`
- `checkpoints/checkpoint_34618512_steps.zip`
- `checkpoints/checkpoint_34768512_steps.json`
- `checkpoints/checkpoint_34768512_steps.zip`
- `checkpoints/checkpoint_34918512_steps.json`
- `checkpoints/checkpoint_34918512_steps.zip`
- `eval_traces/step_32000016/snapshot.zip`
- `eval_traces/step_30000000/snapshot.zip`
- `eval_traces/step_28000032`
- `eval_traces/step_26000016`

</details>

<details><summary><code>ai_v9_35_tick1_exploit_0824</code> — 0.364 GB freed, 20 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_35244768_steps.json` — first, every-10th
- `checkpoints/checkpoint_35244768_steps.zip` — first, every-10th
- `checkpoints/checkpoint_36744768_steps.json` — every-10th
- `checkpoints/checkpoint_36744768_steps.zip` — every-10th
- `checkpoints/checkpoint_37044768_steps.json` — last
- `checkpoints/checkpoint_37044768_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_35394768_steps.json`
- `checkpoints/checkpoint_35394768_steps.zip`
- `checkpoints/checkpoint_35544768_steps.json`
- `checkpoints/checkpoint_35544768_steps.zip`
- `checkpoints/checkpoint_35694768_steps.json`
- `checkpoints/checkpoint_35694768_steps.zip`
- `checkpoints/checkpoint_35844768_steps.json`
- `checkpoints/checkpoint_35844768_steps.zip`
- `checkpoints/checkpoint_35994768_steps.json`
- `checkpoints/checkpoint_35994768_steps.zip`
- `checkpoints/checkpoint_36144768_steps.json`
- `checkpoints/checkpoint_36144768_steps.zip`
- `checkpoints/checkpoint_36294768_steps.json`
- `checkpoints/checkpoint_36294768_steps.zip`
- `checkpoints/checkpoint_36444768_steps.json`
- `checkpoints/checkpoint_36444768_steps.zip`
- `checkpoints/checkpoint_36594768_steps.json`
- `checkpoints/checkpoint_36594768_steps.zip`
- `checkpoints/checkpoint_36894768_steps.json`
- `checkpoints/checkpoint_36894768_steps.zip`

</details>

<details><summary><code>ai_v9_36_tock1c_q6_0824</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_37_tick1_dosext_0825</code> — 1.165 GB freed, 59 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_35244768_steps.json` — first, every-10th
- `checkpoints/checkpoint_35244768_steps.zip` — first, every-10th
- `checkpoints/checkpoint_36744768_steps.json` — every-10th
- `checkpoints/checkpoint_36744768_steps.zip` — every-10th
- `checkpoints/checkpoint_38244768_steps.json` — every-10th
- `checkpoints/checkpoint_38244768_steps.zip` — every-10th
- `checkpoints/checkpoint_39766752_steps.json` — every-10th
- `checkpoints/checkpoint_39766752_steps.zip` — every-10th
- `checkpoints/checkpoint_40066752_steps.json` — last
- `checkpoints/checkpoint_40066752_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_35394768_steps.json`
- `checkpoints/checkpoint_35394768_steps.zip`
- `checkpoints/checkpoint_35544768_steps.json`
- `checkpoints/checkpoint_35544768_steps.zip`
- `checkpoints/checkpoint_35694768_steps.json`
- `checkpoints/checkpoint_35694768_steps.zip`
- `checkpoints/checkpoint_35844768_steps.json`
- `checkpoints/checkpoint_35844768_steps.zip`
- `checkpoints/checkpoint_35994768_steps.json`
- `checkpoints/checkpoint_35994768_steps.zip`
- `checkpoints/checkpoint_36144768_steps.json`
- `checkpoints/checkpoint_36144768_steps.zip`
- `checkpoints/checkpoint_36294768_steps.json`
- `checkpoints/checkpoint_36294768_steps.zip`
- `checkpoints/checkpoint_36444768_steps.json`
- `checkpoints/checkpoint_36444768_steps.zip`
- `checkpoints/checkpoint_36594768_steps.json`
- `checkpoints/checkpoint_36594768_steps.zip`
- `checkpoints/checkpoint_36894768_steps.json`
- `checkpoints/checkpoint_36894768_steps.zip`
- `checkpoints/checkpoint_37044768_steps.json`
- `checkpoints/checkpoint_37044768_steps.zip`
- `checkpoints/checkpoint_37194768_steps.json`
- `checkpoints/checkpoint_37194768_steps.zip`
- `checkpoints/checkpoint_37344768_steps.json`
- `checkpoints/checkpoint_37344768_steps.zip`
- `checkpoints/checkpoint_37494768_steps.json`
- `checkpoints/checkpoint_37494768_steps.zip`
- `checkpoints/checkpoint_37644768_steps.json`
- `checkpoints/checkpoint_37644768_steps.zip`
- `checkpoints/checkpoint_37794768_steps.json`
- `checkpoints/checkpoint_37794768_steps.zip`
- `checkpoints/checkpoint_37944768_steps.json`
- `checkpoints/checkpoint_37944768_steps.zip`
- `checkpoints/checkpoint_38094768_steps.json`
- `checkpoints/checkpoint_38094768_steps.zip`
- `checkpoints/checkpoint_38394768_steps.json`
- `checkpoints/checkpoint_38394768_steps.zip`
- `checkpoints/checkpoint_38544768_steps.json`
- `checkpoints/checkpoint_38544768_steps.zip`
- `checkpoints/checkpoint_38694768_steps.json`
- `checkpoints/checkpoint_38694768_steps.zip`
- `checkpoints/checkpoint_38844768_steps.json`
- `checkpoints/checkpoint_38844768_steps.zip`
- `checkpoints/checkpoint_38994768_steps.json`
- `checkpoints/checkpoint_38994768_steps.zip`
- `checkpoints/checkpoint_39144768_steps.json`
- `checkpoints/checkpoint_39144768_steps.zip`
- `checkpoints/checkpoint_39294768_steps.json`
- `checkpoints/checkpoint_39294768_steps.zip`
- `checkpoints/checkpoint_39444768_steps.json`
- `checkpoints/checkpoint_39444768_steps.zip`
- `checkpoints/checkpoint_39594768_steps.json`
- `checkpoints/checkpoint_39594768_steps.zip`
- `checkpoints/checkpoint_39916752_steps.json`
- `checkpoints/checkpoint_39916752_steps.zip`
- `eval_traces/step_38000016/snapshot.zip`
- `eval_traces/step_36000000/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_38_fdA_coef03_0825</code> — 1.165 GB freed, 36 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_39_fdB_lossonly_0825</code> — 1.165 GB freed, 36 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_40_fdC_ecology_0825</code> — 1.165 GB freed, 36 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_42_fdE_single_0825</code> — 1.165 GB freed, 36 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_49_G2_advgate_0826</code> — 1.165 GB freed, 36 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_50_fdF_p1c_0826</code> — 0.291 GB freed, 16 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26567760_steps.json` — last
- `checkpoints/checkpoint_26567760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`

</details>

<details><summary><code>ai_v9_51_fdF_p2c_0826</code> — 0.728 GB freed, 15 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_26790624_steps.json` — first, every-10th
- `checkpoints/checkpoint_26790624_steps.zip` — first, every-10th
- `checkpoints/checkpoint_27990624_steps.json` — last
- `checkpoints/checkpoint_27990624_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_26940624_steps.json`
- `checkpoints/checkpoint_26940624_steps.zip`
- `checkpoints/checkpoint_27090624_steps.json`
- `checkpoints/checkpoint_27090624_steps.zip`
- `checkpoints/checkpoint_27240624_steps.json`
- `checkpoints/checkpoint_27240624_steps.zip`
- `checkpoints/checkpoint_27390624_steps.json`
- `checkpoints/checkpoint_27390624_steps.zip`
- `checkpoints/checkpoint_27540624_steps.json`
- `checkpoints/checkpoint_27540624_steps.zip`
- `checkpoints/checkpoint_27690624_steps.json`
- `checkpoints/checkpoint_27690624_steps.zip`
- `checkpoints/checkpoint_27840624_steps.json`
- `checkpoints/checkpoint_27840624_steps.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_52_G1p_matched_0826</code> — 1.165 GB freed, 36 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_58_R2CTRL_0827</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_60_R2TOPK_0827</code> — 1.165 GB freed, 36 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_61_R2KL_0827</code> — 1.165 GB freed, 36 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v9_62_R2PLAIN_0827</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_63_R3F6a_0828</code> — 1.093 GB freed, 58 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28217760_steps.json` — every-10th
- `checkpoints/checkpoint_28217760_steps.zip` — every-10th
- `checkpoints/checkpoint_29717760_steps.json` — every-10th
- `checkpoints/checkpoint_29717760_steps.zip` — every-10th
- `checkpoints/checkpoint_30017760_steps.json` — last
- `checkpoints/checkpoint_30017760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `checkpoints/checkpoint_28067760_steps.json`
- `checkpoints/checkpoint_28067760_steps.zip`
- `checkpoints/checkpoint_28367760_steps.json`
- `checkpoints/checkpoint_28367760_steps.zip`
- `checkpoints/checkpoint_28517760_steps.json`
- `checkpoints/checkpoint_28517760_steps.zip`
- `checkpoints/checkpoint_28667760_steps.json`
- `checkpoints/checkpoint_28667760_steps.zip`
- `checkpoints/checkpoint_28817760_steps.json`
- `checkpoints/checkpoint_28817760_steps.zip`
- `checkpoints/checkpoint_28967760_steps.json`
- `checkpoints/checkpoint_28967760_steps.zip`
- `checkpoints/checkpoint_29117760_steps.json`
- `checkpoints/checkpoint_29117760_steps.zip`
- `checkpoints/checkpoint_29267760_steps.json`
- `checkpoints/checkpoint_29267760_steps.zip`
- `checkpoints/checkpoint_29417760_steps.json`
- `checkpoints/checkpoint_29417760_steps.zip`
- `checkpoints/checkpoint_29567760_steps.json`
- `checkpoints/checkpoint_29567760_steps.zip`
- `checkpoints/checkpoint_29867760_steps.json`
- `checkpoints/checkpoint_29867760_steps.zip`
- `eval_traces/step_28000032/snapshot.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_64_R3F6b_0828</code> — 1.093 GB freed, 58 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28217760_steps.json` — every-10th
- `checkpoints/checkpoint_28217760_steps.zip` — every-10th
- `checkpoints/checkpoint_29717760_steps.json` — every-10th
- `checkpoints/checkpoint_29717760_steps.zip` — every-10th
- `checkpoints/checkpoint_30017760_steps.json` — last
- `checkpoints/checkpoint_30017760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `checkpoints/checkpoint_28067760_steps.json`
- `checkpoints/checkpoint_28067760_steps.zip`
- `checkpoints/checkpoint_28367760_steps.json`
- `checkpoints/checkpoint_28367760_steps.zip`
- `checkpoints/checkpoint_28517760_steps.json`
- `checkpoints/checkpoint_28517760_steps.zip`
- `checkpoints/checkpoint_28667760_steps.json`
- `checkpoints/checkpoint_28667760_steps.zip`
- `checkpoints/checkpoint_28817760_steps.json`
- `checkpoints/checkpoint_28817760_steps.zip`
- `checkpoints/checkpoint_28967760_steps.json`
- `checkpoints/checkpoint_28967760_steps.zip`
- `checkpoints/checkpoint_29117760_steps.json`
- `checkpoints/checkpoint_29117760_steps.zip`
- `checkpoints/checkpoint_29267760_steps.json`
- `checkpoints/checkpoint_29267760_steps.zip`
- `checkpoints/checkpoint_29417760_steps.json`
- `checkpoints/checkpoint_29417760_steps.zip`
- `checkpoints/checkpoint_29567760_steps.json`
- `checkpoints/checkpoint_29567760_steps.zip`
- `checkpoints/checkpoint_29867760_steps.json`
- `checkpoints/checkpoint_29867760_steps.zip`
- `eval_traces/step_28000032/snapshot.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_65_R3F6c_0828</code> — 1.093 GB freed, 58 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28217760_steps.json` — every-10th
- `checkpoints/checkpoint_28217760_steps.zip` — every-10th
- `checkpoints/checkpoint_29717760_steps.json` — every-10th
- `checkpoints/checkpoint_29717760_steps.zip` — every-10th
- `checkpoints/checkpoint_30017760_steps.json` — last
- `checkpoints/checkpoint_30017760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `checkpoints/checkpoint_28067760_steps.json`
- `checkpoints/checkpoint_28067760_steps.zip`
- `checkpoints/checkpoint_28367760_steps.json`
- `checkpoints/checkpoint_28367760_steps.zip`
- `checkpoints/checkpoint_28517760_steps.json`
- `checkpoints/checkpoint_28517760_steps.zip`
- `checkpoints/checkpoint_28667760_steps.json`
- `checkpoints/checkpoint_28667760_steps.zip`
- `checkpoints/checkpoint_28817760_steps.json`
- `checkpoints/checkpoint_28817760_steps.zip`
- `checkpoints/checkpoint_28967760_steps.json`
- `checkpoints/checkpoint_28967760_steps.zip`
- `checkpoints/checkpoint_29117760_steps.json`
- `checkpoints/checkpoint_29117760_steps.zip`
- `checkpoints/checkpoint_29267760_steps.json`
- `checkpoints/checkpoint_29267760_steps.zip`
- `checkpoints/checkpoint_29417760_steps.json`
- `checkpoints/checkpoint_29417760_steps.zip`
- `checkpoints/checkpoint_29567760_steps.json`
- `checkpoints/checkpoint_29567760_steps.zip`
- `checkpoints/checkpoint_29867760_steps.json`
- `checkpoints/checkpoint_29867760_steps.zip`
- `eval_traces/step_28000032/snapshot.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_66_R3F6d_0828</code> — 1.093 GB freed, 58 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28217760_steps.json` — every-10th
- `checkpoints/checkpoint_28217760_steps.zip` — every-10th
- `checkpoints/checkpoint_29717760_steps.json` — every-10th
- `checkpoints/checkpoint_29717760_steps.zip` — every-10th
- `checkpoints/checkpoint_30017760_steps.json` — last
- `checkpoints/checkpoint_30017760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `checkpoints/checkpoint_28067760_steps.json`
- `checkpoints/checkpoint_28067760_steps.zip`
- `checkpoints/checkpoint_28367760_steps.json`
- `checkpoints/checkpoint_28367760_steps.zip`
- `checkpoints/checkpoint_28517760_steps.json`
- `checkpoints/checkpoint_28517760_steps.zip`
- `checkpoints/checkpoint_28667760_steps.json`
- `checkpoints/checkpoint_28667760_steps.zip`
- `checkpoints/checkpoint_28817760_steps.json`
- `checkpoints/checkpoint_28817760_steps.zip`
- `checkpoints/checkpoint_28967760_steps.json`
- `checkpoints/checkpoint_28967760_steps.zip`
- `checkpoints/checkpoint_29117760_steps.json`
- `checkpoints/checkpoint_29117760_steps.zip`
- `checkpoints/checkpoint_29267760_steps.json`
- `checkpoints/checkpoint_29267760_steps.zip`
- `checkpoints/checkpoint_29417760_steps.json`
- `checkpoints/checkpoint_29417760_steps.zip`
- `checkpoints/checkpoint_29567760_steps.json`
- `checkpoints/checkpoint_29567760_steps.zip`
- `checkpoints/checkpoint_29867760_steps.json`
- `checkpoints/checkpoint_29867760_steps.zip`
- `eval_traces/step_28000032/snapshot.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_67_R3F6e_0828</code> — 1.093 GB freed, 58 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28217760_steps.json` — every-10th
- `checkpoints/checkpoint_28217760_steps.zip` — every-10th
- `checkpoints/checkpoint_29717760_steps.json` — every-10th
- `checkpoints/checkpoint_29717760_steps.zip` — every-10th
- `checkpoints/checkpoint_30017760_steps.json` — last
- `checkpoints/checkpoint_30017760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `checkpoints/checkpoint_28067760_steps.json`
- `checkpoints/checkpoint_28067760_steps.zip`
- `checkpoints/checkpoint_28367760_steps.json`
- `checkpoints/checkpoint_28367760_steps.zip`
- `checkpoints/checkpoint_28517760_steps.json`
- `checkpoints/checkpoint_28517760_steps.zip`
- `checkpoints/checkpoint_28667760_steps.json`
- `checkpoints/checkpoint_28667760_steps.zip`
- `checkpoints/checkpoint_28817760_steps.json`
- `checkpoints/checkpoint_28817760_steps.zip`
- `checkpoints/checkpoint_28967760_steps.json`
- `checkpoints/checkpoint_28967760_steps.zip`
- `checkpoints/checkpoint_29117760_steps.json`
- `checkpoints/checkpoint_29117760_steps.zip`
- `checkpoints/checkpoint_29267760_steps.json`
- `checkpoints/checkpoint_29267760_steps.zip`
- `checkpoints/checkpoint_29417760_steps.json`
- `checkpoints/checkpoint_29417760_steps.zip`
- `checkpoints/checkpoint_29567760_steps.json`
- `checkpoints/checkpoint_29567760_steps.zip`
- `checkpoints/checkpoint_29867760_steps.json`
- `checkpoints/checkpoint_29867760_steps.zip`
- `eval_traces/step_28000032/snapshot.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_68_R3F6f_0828</code> — 1.093 GB freed, 58 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28217760_steps.json` — every-10th
- `checkpoints/checkpoint_28217760_steps.zip` — every-10th
- `checkpoints/checkpoint_29717760_steps.json` — every-10th
- `checkpoints/checkpoint_29717760_steps.zip` — every-10th
- `checkpoints/checkpoint_30017760_steps.json` — last
- `checkpoints/checkpoint_30017760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `checkpoints/checkpoint_28067760_steps.json`
- `checkpoints/checkpoint_28067760_steps.zip`
- `checkpoints/checkpoint_28367760_steps.json`
- `checkpoints/checkpoint_28367760_steps.zip`
- `checkpoints/checkpoint_28517760_steps.json`
- `checkpoints/checkpoint_28517760_steps.zip`
- `checkpoints/checkpoint_28667760_steps.json`
- `checkpoints/checkpoint_28667760_steps.zip`
- `checkpoints/checkpoint_28817760_steps.json`
- `checkpoints/checkpoint_28817760_steps.zip`
- `checkpoints/checkpoint_28967760_steps.json`
- `checkpoints/checkpoint_28967760_steps.zip`
- `checkpoints/checkpoint_29117760_steps.json`
- `checkpoints/checkpoint_29117760_steps.zip`
- `checkpoints/checkpoint_29267760_steps.json`
- `checkpoints/checkpoint_29267760_steps.zip`
- `checkpoints/checkpoint_29417760_steps.json`
- `checkpoints/checkpoint_29417760_steps.zip`
- `checkpoints/checkpoint_29567760_steps.json`
- `checkpoints/checkpoint_29567760_steps.zip`
- `checkpoints/checkpoint_29867760_steps.json`
- `checkpoints/checkpoint_29867760_steps.zip`
- `eval_traces/step_28000032/snapshot.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_69_R3F6CURR_0828</code> — 1.093 GB freed, 58 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28217760_steps.json` — every-10th
- `checkpoints/checkpoint_28217760_steps.zip` — every-10th
- `checkpoints/checkpoint_29717760_steps.json` — every-10th
- `checkpoints/checkpoint_29717760_steps.zip` — every-10th
- `checkpoints/checkpoint_30017760_steps.json` — last
- `checkpoints/checkpoint_30017760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `checkpoints/checkpoint_28067760_steps.json`
- `checkpoints/checkpoint_28067760_steps.zip`
- `checkpoints/checkpoint_28367760_steps.json`
- `checkpoints/checkpoint_28367760_steps.zip`
- `checkpoints/checkpoint_28517760_steps.json`
- `checkpoints/checkpoint_28517760_steps.zip`
- `checkpoints/checkpoint_28667760_steps.json`
- `checkpoints/checkpoint_28667760_steps.zip`
- `checkpoints/checkpoint_28817760_steps.json`
- `checkpoints/checkpoint_28817760_steps.zip`
- `checkpoints/checkpoint_28967760_steps.json`
- `checkpoints/checkpoint_28967760_steps.zip`
- `checkpoints/checkpoint_29117760_steps.json`
- `checkpoints/checkpoint_29117760_steps.zip`
- `checkpoints/checkpoint_29267760_steps.json`
- `checkpoints/checkpoint_29267760_steps.zip`
- `checkpoints/checkpoint_29417760_steps.json`
- `checkpoints/checkpoint_29417760_steps.zip`
- `checkpoints/checkpoint_29567760_steps.json`
- `checkpoints/checkpoint_29567760_steps.zip`
- `checkpoints/checkpoint_29867760_steps.json`
- `checkpoints/checkpoint_29867760_steps.zip`
- `eval_traces/step_28000032/snapshot.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_70_R3ACTION_0828</code> — 0.983 GB freed, 53 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_28265184_steps.json` — first, every-10th
- `checkpoints/checkpoint_28265184_steps.zip` — first, every-10th
- `checkpoints/checkpoint_29765184_steps.json` — every-10th
- `checkpoints/checkpoint_29765184_steps.zip` — every-10th
- `checkpoints/checkpoint_31271088_steps.json` — every-10th
- `checkpoints/checkpoint_31271088_steps.zip` — every-10th
- `checkpoints/checkpoint_32621088_steps.json` — last
- `checkpoints/checkpoint_32621088_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_28415184_steps.json`
- `checkpoints/checkpoint_28415184_steps.zip`
- `checkpoints/checkpoint_28565184_steps.json`
- `checkpoints/checkpoint_28565184_steps.zip`
- `checkpoints/checkpoint_28715184_steps.json`
- `checkpoints/checkpoint_28715184_steps.zip`
- `checkpoints/checkpoint_28865184_steps.json`
- `checkpoints/checkpoint_28865184_steps.zip`
- `checkpoints/checkpoint_29015184_steps.json`
- `checkpoints/checkpoint_29015184_steps.zip`
- `checkpoints/checkpoint_29165184_steps.json`
- `checkpoints/checkpoint_29165184_steps.zip`
- `checkpoints/checkpoint_29315184_steps.json`
- `checkpoints/checkpoint_29315184_steps.zip`
- `checkpoints/checkpoint_29465184_steps.json`
- `checkpoints/checkpoint_29465184_steps.zip`
- `checkpoints/checkpoint_29615184_steps.json`
- `checkpoints/checkpoint_29615184_steps.zip`
- `checkpoints/checkpoint_29915184_steps.json`
- `checkpoints/checkpoint_29915184_steps.zip`
- `checkpoints/checkpoint_30065184_steps.json`
- `checkpoints/checkpoint_30065184_steps.zip`
- `checkpoints/checkpoint_30215184_steps.json`
- `checkpoints/checkpoint_30215184_steps.zip`
- `checkpoints/checkpoint_30365184_steps.json`
- `checkpoints/checkpoint_30365184_steps.zip`
- `checkpoints/checkpoint_30515184_steps.json`
- `checkpoints/checkpoint_30515184_steps.zip`
- `checkpoints/checkpoint_30665184_steps.json`
- `checkpoints/checkpoint_30665184_steps.zip`
- `checkpoints/checkpoint_30821088_steps.json`
- `checkpoints/checkpoint_30821088_steps.zip`
- `checkpoints/checkpoint_30971088_steps.json`
- `checkpoints/checkpoint_30971088_steps.zip`
- `checkpoints/checkpoint_31121088_steps.json`
- `checkpoints/checkpoint_31121088_steps.zip`
- `checkpoints/checkpoint_31421088_steps.json`
- `checkpoints/checkpoint_31421088_steps.zip`
- `checkpoints/checkpoint_31571088_steps.json`
- `checkpoints/checkpoint_31571088_steps.zip`
- `checkpoints/checkpoint_31721088_steps.json`
- `checkpoints/checkpoint_31721088_steps.zip`
- `checkpoints/checkpoint_31871088_steps.json`
- `checkpoints/checkpoint_31871088_steps.zip`
- `checkpoints/checkpoint_32021088_steps.json`
- `checkpoints/checkpoint_32021088_steps.zip`
- `checkpoints/checkpoint_32171088_steps.json`
- `checkpoints/checkpoint_32171088_steps.zip`
- `checkpoints/checkpoint_32321088_steps.json`
- `checkpoints/checkpoint_32321088_steps.zip`
- `checkpoints/checkpoint_32471088_steps.json`
- `checkpoints/checkpoint_32471088_steps.zip`
- `eval_traces/step_30000000/snapshot.zip`

</details>

<details><summary><code>ai_v9_71_R3ACTIONHI_0828</code> — 0.947 GB freed, 51 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_28265184_steps.json` — first, every-10th
- `checkpoints/checkpoint_28265184_steps.zip` — first, every-10th
- `checkpoints/checkpoint_29765184_steps.json` — every-10th
- `checkpoints/checkpoint_29765184_steps.zip` — every-10th
- `checkpoints/checkpoint_31364304_steps.json` — every-10th
- `checkpoints/checkpoint_31364304_steps.zip` — every-10th
- `checkpoints/checkpoint_32564304_steps.json` — last
- `checkpoints/checkpoint_32564304_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_28415184_steps.json`
- `checkpoints/checkpoint_28415184_steps.zip`
- `checkpoints/checkpoint_28565184_steps.json`
- `checkpoints/checkpoint_28565184_steps.zip`
- `checkpoints/checkpoint_28715184_steps.json`
- `checkpoints/checkpoint_28715184_steps.zip`
- `checkpoints/checkpoint_28865184_steps.json`
- `checkpoints/checkpoint_28865184_steps.zip`
- `checkpoints/checkpoint_29015184_steps.json`
- `checkpoints/checkpoint_29015184_steps.zip`
- `checkpoints/checkpoint_29165184_steps.json`
- `checkpoints/checkpoint_29165184_steps.zip`
- `checkpoints/checkpoint_29315184_steps.json`
- `checkpoints/checkpoint_29315184_steps.zip`
- `checkpoints/checkpoint_29465184_steps.json`
- `checkpoints/checkpoint_29465184_steps.zip`
- `checkpoints/checkpoint_29615184_steps.json`
- `checkpoints/checkpoint_29615184_steps.zip`
- `checkpoints/checkpoint_29915184_steps.json`
- `checkpoints/checkpoint_29915184_steps.zip`
- `checkpoints/checkpoint_30065184_steps.json`
- `checkpoints/checkpoint_30065184_steps.zip`
- `checkpoints/checkpoint_30215184_steps.json`
- `checkpoints/checkpoint_30215184_steps.zip`
- `checkpoints/checkpoint_30365184_steps.json`
- `checkpoints/checkpoint_30365184_steps.zip`
- `checkpoints/checkpoint_30515184_steps.json`
- `checkpoints/checkpoint_30515184_steps.zip`
- `checkpoints/checkpoint_30665184_steps.json`
- `checkpoints/checkpoint_30665184_steps.zip`
- `checkpoints/checkpoint_30815184_steps.json`
- `checkpoints/checkpoint_30815184_steps.zip`
- `checkpoints/checkpoint_30965184_steps.json`
- `checkpoints/checkpoint_30965184_steps.zip`
- `checkpoints/checkpoint_31214304_steps.json`
- `checkpoints/checkpoint_31214304_steps.zip`
- `checkpoints/checkpoint_31514304_steps.json`
- `checkpoints/checkpoint_31514304_steps.zip`
- `checkpoints/checkpoint_31664304_steps.json`
- `checkpoints/checkpoint_31664304_steps.zip`
- `checkpoints/checkpoint_31814304_steps.json`
- `checkpoints/checkpoint_31814304_steps.zip`
- `checkpoints/checkpoint_31964304_steps.json`
- `checkpoints/checkpoint_31964304_steps.zip`
- `checkpoints/checkpoint_32114304_steps.json`
- `checkpoints/checkpoint_32114304_steps.zip`
- `checkpoints/checkpoint_32264304_steps.json`
- `checkpoints/checkpoint_32264304_steps.zip`
- `checkpoints/checkpoint_32414304_steps.json`
- `checkpoints/checkpoint_32414304_steps.zip`
- `eval_traces/step_30000000/snapshot.zip`

</details>

<details><summary><code>ai_v9_72_R3SELF_0828</code> — 0.983 GB freed, 53 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_28265184_steps.json` — first, every-10th
- `checkpoints/checkpoint_28265184_steps.zip` — first, every-10th
- `checkpoints/checkpoint_29765184_steps.json` — every-10th
- `checkpoints/checkpoint_29765184_steps.zip` — every-10th
- `checkpoints/checkpoint_31265184_steps.json` — every-10th
- `checkpoints/checkpoint_31265184_steps.zip` — every-10th
- `checkpoints/checkpoint_32615184_steps.json` — last
- `checkpoints/checkpoint_32615184_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_28415184_steps.json`
- `checkpoints/checkpoint_28415184_steps.zip`
- `checkpoints/checkpoint_28565184_steps.json`
- `checkpoints/checkpoint_28565184_steps.zip`
- `checkpoints/checkpoint_28715184_steps.json`
- `checkpoints/checkpoint_28715184_steps.zip`
- `checkpoints/checkpoint_28865184_steps.json`
- `checkpoints/checkpoint_28865184_steps.zip`
- `checkpoints/checkpoint_29015184_steps.json`
- `checkpoints/checkpoint_29015184_steps.zip`
- `checkpoints/checkpoint_29165184_steps.json`
- `checkpoints/checkpoint_29165184_steps.zip`
- `checkpoints/checkpoint_29315184_steps.json`
- `checkpoints/checkpoint_29315184_steps.zip`
- `checkpoints/checkpoint_29465184_steps.json`
- `checkpoints/checkpoint_29465184_steps.zip`
- `checkpoints/checkpoint_29615184_steps.json`
- `checkpoints/checkpoint_29615184_steps.zip`
- `checkpoints/checkpoint_29915184_steps.json`
- `checkpoints/checkpoint_29915184_steps.zip`
- `checkpoints/checkpoint_30065184_steps.json`
- `checkpoints/checkpoint_30065184_steps.zip`
- `checkpoints/checkpoint_30215184_steps.json`
- `checkpoints/checkpoint_30215184_steps.zip`
- `checkpoints/checkpoint_30365184_steps.json`
- `checkpoints/checkpoint_30365184_steps.zip`
- `checkpoints/checkpoint_30515184_steps.json`
- `checkpoints/checkpoint_30515184_steps.zip`
- `checkpoints/checkpoint_30665184_steps.json`
- `checkpoints/checkpoint_30665184_steps.zip`
- `checkpoints/checkpoint_30815184_steps.json`
- `checkpoints/checkpoint_30815184_steps.zip`
- `checkpoints/checkpoint_30965184_steps.json`
- `checkpoints/checkpoint_30965184_steps.zip`
- `checkpoints/checkpoint_31115184_steps.json`
- `checkpoints/checkpoint_31115184_steps.zip`
- `checkpoints/checkpoint_31415184_steps.json`
- `checkpoints/checkpoint_31415184_steps.zip`
- `checkpoints/checkpoint_31565184_steps.json`
- `checkpoints/checkpoint_31565184_steps.zip`
- `checkpoints/checkpoint_31715184_steps.json`
- `checkpoints/checkpoint_31715184_steps.zip`
- `checkpoints/checkpoint_31865184_steps.json`
- `checkpoints/checkpoint_31865184_steps.zip`
- `checkpoints/checkpoint_32015184_steps.json`
- `checkpoints/checkpoint_32015184_steps.zip`
- `checkpoints/checkpoint_32165184_steps.json`
- `checkpoints/checkpoint_32165184_steps.zip`
- `checkpoints/checkpoint_32315184_steps.json`
- `checkpoints/checkpoint_32315184_steps.zip`
- `checkpoints/checkpoint_32465184_steps.json`
- `checkpoints/checkpoint_32465184_steps.zip`
- `eval_traces/step_30000000/snapshot.zip`

</details>

<details><summary><code>ai_v9_73_R4S3a_0829</code> — 2.274 GB freed, 118 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28217760_steps.json` — every-10th
- `checkpoints/checkpoint_28217760_steps.zip` — every-10th
- `checkpoints/checkpoint_29717760_steps.json` — every-10th
- `checkpoints/checkpoint_29717760_steps.zip` — every-10th
- `checkpoints/checkpoint_31217760_steps.json` — every-10th
- `checkpoints/checkpoint_31217760_steps.zip` — every-10th
- `checkpoints/checkpoint_32859216_steps.json` — every-10th
- `checkpoints/checkpoint_32859216_steps.zip` — every-10th
- `checkpoints/checkpoint_34359216_steps.json` — every-10th
- `checkpoints/checkpoint_34359216_steps.zip` — every-10th
- `checkpoints/checkpoint_34959216_steps.json` — last
- `checkpoints/checkpoint_34959216_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `checkpoints/checkpoint_28067760_steps.json`
- `checkpoints/checkpoint_28067760_steps.zip`
- `checkpoints/checkpoint_28367760_steps.json`
- `checkpoints/checkpoint_28367760_steps.zip`
- `checkpoints/checkpoint_28517760_steps.json`
- `checkpoints/checkpoint_28517760_steps.zip`
- `checkpoints/checkpoint_28667760_steps.json`
- `checkpoints/checkpoint_28667760_steps.zip`
- `checkpoints/checkpoint_28817760_steps.json`
- `checkpoints/checkpoint_28817760_steps.zip`
- `checkpoints/checkpoint_28967760_steps.json`
- `checkpoints/checkpoint_28967760_steps.zip`
- `checkpoints/checkpoint_29117760_steps.json`
- `checkpoints/checkpoint_29117760_steps.zip`
- `checkpoints/checkpoint_29267760_steps.json`
- `checkpoints/checkpoint_29267760_steps.zip`
- `checkpoints/checkpoint_29417760_steps.json`
- `checkpoints/checkpoint_29417760_steps.zip`
- `checkpoints/checkpoint_29567760_steps.json`
- `checkpoints/checkpoint_29567760_steps.zip`
- `checkpoints/checkpoint_29867760_steps.json`
- `checkpoints/checkpoint_29867760_steps.zip`
- `checkpoints/checkpoint_30017760_steps.json`
- `checkpoints/checkpoint_30017760_steps.zip`
- `checkpoints/checkpoint_30167760_steps.json`
- `checkpoints/checkpoint_30167760_steps.zip`
- `checkpoints/checkpoint_30317760_steps.json`
- `checkpoints/checkpoint_30317760_steps.zip`
- `checkpoints/checkpoint_30467760_steps.json`
- `checkpoints/checkpoint_30467760_steps.zip`
- `checkpoints/checkpoint_30617760_steps.json`
- `checkpoints/checkpoint_30617760_steps.zip`
- `checkpoints/checkpoint_30767760_steps.json`
- `checkpoints/checkpoint_30767760_steps.zip`
- `checkpoints/checkpoint_30917760_steps.json`
- `checkpoints/checkpoint_30917760_steps.zip`
- `checkpoints/checkpoint_31067760_steps.json`
- `checkpoints/checkpoint_31067760_steps.zip`
- `checkpoints/checkpoint_31509216_steps.json`
- `checkpoints/checkpoint_31509216_steps.zip`
- `checkpoints/checkpoint_31659216_steps.json`
- `checkpoints/checkpoint_31659216_steps.zip`
- `checkpoints/checkpoint_31809216_steps.json`
- `checkpoints/checkpoint_31809216_steps.zip`
- `checkpoints/checkpoint_31959216_steps.json`
- `checkpoints/checkpoint_31959216_steps.zip`
- `checkpoints/checkpoint_32109216_steps.json`
- `checkpoints/checkpoint_32109216_steps.zip`
- `checkpoints/checkpoint_32259216_steps.json`
- `checkpoints/checkpoint_32259216_steps.zip`
- `checkpoints/checkpoint_32409216_steps.json`
- `checkpoints/checkpoint_32409216_steps.zip`
- `checkpoints/checkpoint_32559216_steps.json`
- `checkpoints/checkpoint_32559216_steps.zip`
- `checkpoints/checkpoint_32709216_steps.json`
- `checkpoints/checkpoint_32709216_steps.zip`
- `checkpoints/checkpoint_33009216_steps.json`
- `checkpoints/checkpoint_33009216_steps.zip`
- `checkpoints/checkpoint_33159216_steps.json`
- `checkpoints/checkpoint_33159216_steps.zip`
- `checkpoints/checkpoint_33309216_steps.json`
- `checkpoints/checkpoint_33309216_steps.zip`
- `checkpoints/checkpoint_33459216_steps.json`
- `checkpoints/checkpoint_33459216_steps.zip`
- `checkpoints/checkpoint_33609216_steps.json`
- `checkpoints/checkpoint_33609216_steps.zip`
- `checkpoints/checkpoint_33759216_steps.json`
- `checkpoints/checkpoint_33759216_steps.zip`
- `checkpoints/checkpoint_33909216_steps.json`
- `checkpoints/checkpoint_33909216_steps.zip`
- `checkpoints/checkpoint_34059216_steps.json`
- `checkpoints/checkpoint_34059216_steps.zip`
- `checkpoints/checkpoint_34209216_steps.json`
- `checkpoints/checkpoint_34209216_steps.zip`
- `checkpoints/checkpoint_34509216_steps.json`
- `checkpoints/checkpoint_34509216_steps.zip`
- `checkpoints/checkpoint_34659216_steps.json`
- `checkpoints/checkpoint_34659216_steps.zip`
- `checkpoints/checkpoint_34809216_steps.json`
- `checkpoints/checkpoint_34809216_steps.zip`
- `eval_traces/step_32000016/snapshot.zip`
- `eval_traces/step_30000000/snapshot.zip`
- `eval_traces/step_28000032`
- `eval_traces/step_26000016`

</details>

<details><summary><code>ai_v9_74_R4S3b_0829</code> — 2.313 GB freed, 120 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28217760_steps.json` — every-10th
- `checkpoints/checkpoint_28217760_steps.zip` — every-10th
- `checkpoints/checkpoint_29717760_steps.json` — every-10th
- `checkpoints/checkpoint_29717760_steps.zip` — every-10th
- `checkpoints/checkpoint_31312608_steps.json` — every-10th
- `checkpoints/checkpoint_31312608_steps.zip` — every-10th
- `checkpoints/checkpoint_32812608_steps.json` — every-10th
- `checkpoints/checkpoint_32812608_steps.zip` — every-10th
- `checkpoints/checkpoint_34312608_steps.json` — every-10th
- `checkpoints/checkpoint_34312608_steps.zip` — every-10th
- `checkpoints/checkpoint_35062608_steps.json` — last
- `checkpoints/checkpoint_35062608_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `checkpoints/checkpoint_28067760_steps.json`
- `checkpoints/checkpoint_28067760_steps.zip`
- `checkpoints/checkpoint_28367760_steps.json`
- `checkpoints/checkpoint_28367760_steps.zip`
- `checkpoints/checkpoint_28517760_steps.json`
- `checkpoints/checkpoint_28517760_steps.zip`
- `checkpoints/checkpoint_28667760_steps.json`
- `checkpoints/checkpoint_28667760_steps.zip`
- `checkpoints/checkpoint_28817760_steps.json`
- `checkpoints/checkpoint_28817760_steps.zip`
- `checkpoints/checkpoint_28967760_steps.json`
- `checkpoints/checkpoint_28967760_steps.zip`
- `checkpoints/checkpoint_29117760_steps.json`
- `checkpoints/checkpoint_29117760_steps.zip`
- `checkpoints/checkpoint_29267760_steps.json`
- `checkpoints/checkpoint_29267760_steps.zip`
- `checkpoints/checkpoint_29417760_steps.json`
- `checkpoints/checkpoint_29417760_steps.zip`
- `checkpoints/checkpoint_29567760_steps.json`
- `checkpoints/checkpoint_29567760_steps.zip`
- `checkpoints/checkpoint_29867760_steps.json`
- `checkpoints/checkpoint_29867760_steps.zip`
- `checkpoints/checkpoint_30017760_steps.json`
- `checkpoints/checkpoint_30017760_steps.zip`
- `checkpoints/checkpoint_30167760_steps.json`
- `checkpoints/checkpoint_30167760_steps.zip`
- `checkpoints/checkpoint_30317760_steps.json`
- `checkpoints/checkpoint_30317760_steps.zip`
- `checkpoints/checkpoint_30467760_steps.json`
- `checkpoints/checkpoint_30467760_steps.zip`
- `checkpoints/checkpoint_30617760_steps.json`
- `checkpoints/checkpoint_30617760_steps.zip`
- `checkpoints/checkpoint_30767760_steps.json`
- `checkpoints/checkpoint_30767760_steps.zip`
- `checkpoints/checkpoint_30917760_steps.json`
- `checkpoints/checkpoint_30917760_steps.zip`
- `checkpoints/checkpoint_31067760_steps.json`
- `checkpoints/checkpoint_31067760_steps.zip`
- `checkpoints/checkpoint_31462608_steps.json`
- `checkpoints/checkpoint_31462608_steps.zip`
- `checkpoints/checkpoint_31612608_steps.json`
- `checkpoints/checkpoint_31612608_steps.zip`
- `checkpoints/checkpoint_31762608_steps.json`
- `checkpoints/checkpoint_31762608_steps.zip`
- `checkpoints/checkpoint_31912608_steps.json`
- `checkpoints/checkpoint_31912608_steps.zip`
- `checkpoints/checkpoint_32062608_steps.json`
- `checkpoints/checkpoint_32062608_steps.zip`
- `checkpoints/checkpoint_32212608_steps.json`
- `checkpoints/checkpoint_32212608_steps.zip`
- `checkpoints/checkpoint_32362608_steps.json`
- `checkpoints/checkpoint_32362608_steps.zip`
- `checkpoints/checkpoint_32512608_steps.json`
- `checkpoints/checkpoint_32512608_steps.zip`
- `checkpoints/checkpoint_32662608_steps.json`
- `checkpoints/checkpoint_32662608_steps.zip`
- `checkpoints/checkpoint_32962608_steps.json`
- `checkpoints/checkpoint_32962608_steps.zip`
- `checkpoints/checkpoint_33112608_steps.json`
- `checkpoints/checkpoint_33112608_steps.zip`
- `checkpoints/checkpoint_33262608_steps.json`
- `checkpoints/checkpoint_33262608_steps.zip`
- `checkpoints/checkpoint_33412608_steps.json`
- `checkpoints/checkpoint_33412608_steps.zip`
- `checkpoints/checkpoint_33562608_steps.json`
- `checkpoints/checkpoint_33562608_steps.zip`
- `checkpoints/checkpoint_33712608_steps.json`
- `checkpoints/checkpoint_33712608_steps.zip`
- `checkpoints/checkpoint_33862608_steps.json`
- `checkpoints/checkpoint_33862608_steps.zip`
- `checkpoints/checkpoint_34012608_steps.json`
- `checkpoints/checkpoint_34012608_steps.zip`
- `checkpoints/checkpoint_34162608_steps.json`
- `checkpoints/checkpoint_34162608_steps.zip`
- `checkpoints/checkpoint_34462608_steps.json`
- `checkpoints/checkpoint_34462608_steps.zip`
- `checkpoints/checkpoint_34612608_steps.json`
- `checkpoints/checkpoint_34612608_steps.zip`
- `checkpoints/checkpoint_34762608_steps.json`
- `checkpoints/checkpoint_34762608_steps.zip`
- `checkpoints/checkpoint_34912608_steps.json`
- `checkpoints/checkpoint_34912608_steps.zip`
- `eval_traces/step_32000016/snapshot.zip`
- `eval_traces/step_30000000/snapshot.zip`
- `eval_traces/step_28000032`
- `eval_traces/step_26000016`

</details>

<details><summary><code>ai_v9_75_R4S3c_0829</code> — 2.324 GB freed, 120 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28217760_steps.json` — every-10th
- `checkpoints/checkpoint_28217760_steps.zip` — every-10th
- `checkpoints/checkpoint_29717760_steps.json` — every-10th
- `checkpoints/checkpoint_29717760_steps.zip` — every-10th
- `checkpoints/checkpoint_31224480_steps.json` — every-10th
- `checkpoints/checkpoint_31224480_steps.zip` — every-10th
- `checkpoints/checkpoint_32724480_steps.json` — every-10th
- `checkpoints/checkpoint_32724480_steps.zip` — every-10th
- `checkpoints/checkpoint_34224480_steps.json` — every-10th
- `checkpoints/checkpoint_34224480_steps.zip` — every-10th
- `checkpoints/checkpoint_34974480_steps.json` — last
- `checkpoints/checkpoint_34974480_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `checkpoints/checkpoint_28067760_steps.json`
- `checkpoints/checkpoint_28067760_steps.zip`
- `checkpoints/checkpoint_28367760_steps.json`
- `checkpoints/checkpoint_28367760_steps.zip`
- `checkpoints/checkpoint_28517760_steps.json`
- `checkpoints/checkpoint_28517760_steps.zip`
- `checkpoints/checkpoint_28667760_steps.json`
- `checkpoints/checkpoint_28667760_steps.zip`
- `checkpoints/checkpoint_28817760_steps.json`
- `checkpoints/checkpoint_28817760_steps.zip`
- `checkpoints/checkpoint_28967760_steps.json`
- `checkpoints/checkpoint_28967760_steps.zip`
- `checkpoints/checkpoint_29117760_steps.json`
- `checkpoints/checkpoint_29117760_steps.zip`
- `checkpoints/checkpoint_29267760_steps.json`
- `checkpoints/checkpoint_29267760_steps.zip`
- `checkpoints/checkpoint_29417760_steps.json`
- `checkpoints/checkpoint_29417760_steps.zip`
- `checkpoints/checkpoint_29567760_steps.json`
- `checkpoints/checkpoint_29567760_steps.zip`
- `checkpoints/checkpoint_29867760_steps.json`
- `checkpoints/checkpoint_29867760_steps.zip`
- `checkpoints/checkpoint_30017760_steps.json`
- `checkpoints/checkpoint_30017760_steps.zip`
- `checkpoints/checkpoint_30167760_steps.json`
- `checkpoints/checkpoint_30167760_steps.zip`
- `checkpoints/checkpoint_30317760_steps.json`
- `checkpoints/checkpoint_30317760_steps.zip`
- `checkpoints/checkpoint_30467760_steps.json`
- `checkpoints/checkpoint_30467760_steps.zip`
- `checkpoints/checkpoint_30624480_steps.json`
- `checkpoints/checkpoint_30624480_steps.zip`
- `checkpoints/checkpoint_30774480_steps.json`
- `checkpoints/checkpoint_30774480_steps.zip`
- `checkpoints/checkpoint_30924480_steps.json`
- `checkpoints/checkpoint_30924480_steps.zip`
- `checkpoints/checkpoint_31074480_steps.json`
- `checkpoints/checkpoint_31074480_steps.zip`
- `checkpoints/checkpoint_31374480_steps.json`
- `checkpoints/checkpoint_31374480_steps.zip`
- `checkpoints/checkpoint_31524480_steps.json`
- `checkpoints/checkpoint_31524480_steps.zip`
- `checkpoints/checkpoint_31674480_steps.json`
- `checkpoints/checkpoint_31674480_steps.zip`
- `checkpoints/checkpoint_31824480_steps.json`
- `checkpoints/checkpoint_31824480_steps.zip`
- `checkpoints/checkpoint_31974480_steps.json`
- `checkpoints/checkpoint_31974480_steps.zip`
- `checkpoints/checkpoint_32124480_steps.json`
- `checkpoints/checkpoint_32124480_steps.zip`
- `checkpoints/checkpoint_32274480_steps.json`
- `checkpoints/checkpoint_32274480_steps.zip`
- `checkpoints/checkpoint_32424480_steps.json`
- `checkpoints/checkpoint_32424480_steps.zip`
- `checkpoints/checkpoint_32574480_steps.json`
- `checkpoints/checkpoint_32574480_steps.zip`
- `checkpoints/checkpoint_32874480_steps.json`
- `checkpoints/checkpoint_32874480_steps.zip`
- `checkpoints/checkpoint_33024480_steps.json`
- `checkpoints/checkpoint_33024480_steps.zip`
- `checkpoints/checkpoint_33174480_steps.json`
- `checkpoints/checkpoint_33174480_steps.zip`
- `checkpoints/checkpoint_33324480_steps.json`
- `checkpoints/checkpoint_33324480_steps.zip`
- `checkpoints/checkpoint_33474480_steps.json`
- `checkpoints/checkpoint_33474480_steps.zip`
- `checkpoints/checkpoint_33624480_steps.json`
- `checkpoints/checkpoint_33624480_steps.zip`
- `checkpoints/checkpoint_33774480_steps.json`
- `checkpoints/checkpoint_33774480_steps.zip`
- `checkpoints/checkpoint_33924480_steps.json`
- `checkpoints/checkpoint_33924480_steps.zip`
- `checkpoints/checkpoint_34074480_steps.json`
- `checkpoints/checkpoint_34074480_steps.zip`
- `checkpoints/checkpoint_34374480_steps.json`
- `checkpoints/checkpoint_34374480_steps.zip`
- `checkpoints/checkpoint_34524480_steps.json`
- `checkpoints/checkpoint_34524480_steps.zip`
- `checkpoints/checkpoint_34674480_steps.json`
- `checkpoints/checkpoint_34674480_steps.zip`
- `checkpoints/checkpoint_34824480_steps.json`
- `checkpoints/checkpoint_34824480_steps.zip`
- `eval_traces/step_32000016/snapshot.zip`
- `eval_traces/step_30000000/snapshot.zip`
- `eval_traces/step_28000032`
- `eval_traces/step_26000016`

</details>

<details><summary><code>ai_v9_76_R4ACTION_0830</code> — 0.947 GB freed, 51 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_28265184_steps.json` — first, every-10th
- `checkpoints/checkpoint_28265184_steps.zip` — first, every-10th
- `checkpoints/checkpoint_29765184_steps.json` — every-10th
- `checkpoints/checkpoint_29765184_steps.zip` — every-10th
- `checkpoints/checkpoint_31265184_steps.json` — every-10th
- `checkpoints/checkpoint_31265184_steps.zip` — every-10th
- `checkpoints/checkpoint_32595648_steps.json` — last
- `checkpoints/checkpoint_32595648_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_28415184_steps.json`
- `checkpoints/checkpoint_28415184_steps.zip`
- `checkpoints/checkpoint_28565184_steps.json`
- `checkpoints/checkpoint_28565184_steps.zip`
- `checkpoints/checkpoint_28715184_steps.json`
- `checkpoints/checkpoint_28715184_steps.zip`
- `checkpoints/checkpoint_28865184_steps.json`
- `checkpoints/checkpoint_28865184_steps.zip`
- `checkpoints/checkpoint_29015184_steps.json`
- `checkpoints/checkpoint_29015184_steps.zip`
- `checkpoints/checkpoint_29165184_steps.json`
- `checkpoints/checkpoint_29165184_steps.zip`
- `checkpoints/checkpoint_29315184_steps.json`
- `checkpoints/checkpoint_29315184_steps.zip`
- `checkpoints/checkpoint_29465184_steps.json`
- `checkpoints/checkpoint_29465184_steps.zip`
- `checkpoints/checkpoint_29615184_steps.json`
- `checkpoints/checkpoint_29615184_steps.zip`
- `checkpoints/checkpoint_29915184_steps.json`
- `checkpoints/checkpoint_29915184_steps.zip`
- `checkpoints/checkpoint_30065184_steps.json`
- `checkpoints/checkpoint_30065184_steps.zip`
- `checkpoints/checkpoint_30215184_steps.json`
- `checkpoints/checkpoint_30215184_steps.zip`
- `checkpoints/checkpoint_30365184_steps.json`
- `checkpoints/checkpoint_30365184_steps.zip`
- `checkpoints/checkpoint_30515184_steps.json`
- `checkpoints/checkpoint_30515184_steps.zip`
- `checkpoints/checkpoint_30665184_steps.json`
- `checkpoints/checkpoint_30665184_steps.zip`
- `checkpoints/checkpoint_30815184_steps.json`
- `checkpoints/checkpoint_30815184_steps.zip`
- `checkpoints/checkpoint_30965184_steps.json`
- `checkpoints/checkpoint_30965184_steps.zip`
- `checkpoints/checkpoint_31115184_steps.json`
- `checkpoints/checkpoint_31115184_steps.zip`
- `checkpoints/checkpoint_31415184_steps.json`
- `checkpoints/checkpoint_31415184_steps.zip`
- `checkpoints/checkpoint_31565184_steps.json`
- `checkpoints/checkpoint_31565184_steps.zip`
- `checkpoints/checkpoint_31715184_steps.json`
- `checkpoints/checkpoint_31715184_steps.zip`
- `checkpoints/checkpoint_31865184_steps.json`
- `checkpoints/checkpoint_31865184_steps.zip`
- `checkpoints/checkpoint_32015184_steps.json`
- `checkpoints/checkpoint_32015184_steps.zip`
- `checkpoints/checkpoint_32295648_steps.json`
- `checkpoints/checkpoint_32295648_steps.zip`
- `checkpoints/checkpoint_32445648_steps.json`
- `checkpoints/checkpoint_32445648_steps.zip`
- `eval_traces/step_30000000/snapshot.zip`

</details>

<details><summary><code>ai_v9_77_G1LEAN_0830</code> — 0.801 GB freed, 43 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28217760_steps.json` — every-10th
- `checkpoints/checkpoint_28217760_steps.zip` — every-10th
- `checkpoints/checkpoint_28817760_steps.json` — last
- `checkpoints/checkpoint_28817760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `checkpoints/checkpoint_28067760_steps.json`
- `checkpoints/checkpoint_28067760_steps.zip`
- `checkpoints/checkpoint_28367760_steps.json`
- `checkpoints/checkpoint_28367760_steps.zip`
- `checkpoints/checkpoint_28517760_steps.json`
- `checkpoints/checkpoint_28517760_steps.zip`
- `checkpoints/checkpoint_28667760_steps.json`
- `checkpoints/checkpoint_28667760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_79_REVIVE1a_0830</code> — 0.838 GB freed, 45 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_35244768_steps.json` — first, every-10th
- `checkpoints/checkpoint_35244768_steps.zip` — first, every-10th
- `checkpoints/checkpoint_36744768_steps.json` — every-10th
- `checkpoints/checkpoint_36744768_steps.zip` — every-10th
- `checkpoints/checkpoint_38244768_steps.json` — every-10th
- `checkpoints/checkpoint_38244768_steps.zip` — every-10th
- `checkpoints/checkpoint_38994768_steps.json` — last
- `checkpoints/checkpoint_38994768_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_35394768_steps.json`
- `checkpoints/checkpoint_35394768_steps.zip`
- `checkpoints/checkpoint_35544768_steps.json`
- `checkpoints/checkpoint_35544768_steps.zip`
- `checkpoints/checkpoint_35694768_steps.json`
- `checkpoints/checkpoint_35694768_steps.zip`
- `checkpoints/checkpoint_35844768_steps.json`
- `checkpoints/checkpoint_35844768_steps.zip`
- `checkpoints/checkpoint_35994768_steps.json`
- `checkpoints/checkpoint_35994768_steps.zip`
- `checkpoints/checkpoint_36144768_steps.json`
- `checkpoints/checkpoint_36144768_steps.zip`
- `checkpoints/checkpoint_36294768_steps.json`
- `checkpoints/checkpoint_36294768_steps.zip`
- `checkpoints/checkpoint_36444768_steps.json`
- `checkpoints/checkpoint_36444768_steps.zip`
- `checkpoints/checkpoint_36594768_steps.json`
- `checkpoints/checkpoint_36594768_steps.zip`
- `checkpoints/checkpoint_36894768_steps.json`
- `checkpoints/checkpoint_36894768_steps.zip`
- `checkpoints/checkpoint_37044768_steps.json`
- `checkpoints/checkpoint_37044768_steps.zip`
- `checkpoints/checkpoint_37194768_steps.json`
- `checkpoints/checkpoint_37194768_steps.zip`
- `checkpoints/checkpoint_37344768_steps.json`
- `checkpoints/checkpoint_37344768_steps.zip`
- `checkpoints/checkpoint_37494768_steps.json`
- `checkpoints/checkpoint_37494768_steps.zip`
- `checkpoints/checkpoint_37644768_steps.json`
- `checkpoints/checkpoint_37644768_steps.zip`
- `checkpoints/checkpoint_37794768_steps.json`
- `checkpoints/checkpoint_37794768_steps.zip`
- `checkpoints/checkpoint_37944768_steps.json`
- `checkpoints/checkpoint_37944768_steps.zip`
- `checkpoints/checkpoint_38094768_steps.json`
- `checkpoints/checkpoint_38094768_steps.zip`
- `checkpoints/checkpoint_38394768_steps.json`
- `checkpoints/checkpoint_38394768_steps.zip`
- `checkpoints/checkpoint_38544768_steps.json`
- `checkpoints/checkpoint_38544768_steps.zip`
- `checkpoints/checkpoint_38694768_steps.json`
- `checkpoints/checkpoint_38694768_steps.zip`
- `checkpoints/checkpoint_38844768_steps.json`
- `checkpoints/checkpoint_38844768_steps.zip`
- `eval_traces/step_36000000/snapshot.zip`

</details>

<details><summary><code>ai_v9_80_REVIVE1b_0830</code> — 0.838 GB freed, 45 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_35244768_steps.json` — first, every-10th
- `checkpoints/checkpoint_35244768_steps.zip` — first, every-10th
- `checkpoints/checkpoint_36744768_steps.json` — every-10th
- `checkpoints/checkpoint_36744768_steps.zip` — every-10th
- `checkpoints/checkpoint_38244768_steps.json` — every-10th
- `checkpoints/checkpoint_38244768_steps.zip` — every-10th
- `checkpoints/checkpoint_38994768_steps.json` — last
- `checkpoints/checkpoint_38994768_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_35394768_steps.json`
- `checkpoints/checkpoint_35394768_steps.zip`
- `checkpoints/checkpoint_35544768_steps.json`
- `checkpoints/checkpoint_35544768_steps.zip`
- `checkpoints/checkpoint_35694768_steps.json`
- `checkpoints/checkpoint_35694768_steps.zip`
- `checkpoints/checkpoint_35844768_steps.json`
- `checkpoints/checkpoint_35844768_steps.zip`
- `checkpoints/checkpoint_35994768_steps.json`
- `checkpoints/checkpoint_35994768_steps.zip`
- `checkpoints/checkpoint_36144768_steps.json`
- `checkpoints/checkpoint_36144768_steps.zip`
- `checkpoints/checkpoint_36294768_steps.json`
- `checkpoints/checkpoint_36294768_steps.zip`
- `checkpoints/checkpoint_36444768_steps.json`
- `checkpoints/checkpoint_36444768_steps.zip`
- `checkpoints/checkpoint_36594768_steps.json`
- `checkpoints/checkpoint_36594768_steps.zip`
- `checkpoints/checkpoint_36894768_steps.json`
- `checkpoints/checkpoint_36894768_steps.zip`
- `checkpoints/checkpoint_37044768_steps.json`
- `checkpoints/checkpoint_37044768_steps.zip`
- `checkpoints/checkpoint_37194768_steps.json`
- `checkpoints/checkpoint_37194768_steps.zip`
- `checkpoints/checkpoint_37344768_steps.json`
- `checkpoints/checkpoint_37344768_steps.zip`
- `checkpoints/checkpoint_37494768_steps.json`
- `checkpoints/checkpoint_37494768_steps.zip`
- `checkpoints/checkpoint_37644768_steps.json`
- `checkpoints/checkpoint_37644768_steps.zip`
- `checkpoints/checkpoint_37794768_steps.json`
- `checkpoints/checkpoint_37794768_steps.zip`
- `checkpoints/checkpoint_37944768_steps.json`
- `checkpoints/checkpoint_37944768_steps.zip`
- `checkpoints/checkpoint_38094768_steps.json`
- `checkpoints/checkpoint_38094768_steps.zip`
- `checkpoints/checkpoint_38394768_steps.json`
- `checkpoints/checkpoint_38394768_steps.zip`
- `checkpoints/checkpoint_38544768_steps.json`
- `checkpoints/checkpoint_38544768_steps.zip`
- `checkpoints/checkpoint_38694768_steps.json`
- `checkpoints/checkpoint_38694768_steps.zip`
- `checkpoints/checkpoint_38844768_steps.json`
- `checkpoints/checkpoint_38844768_steps.zip`
- `eval_traces/step_36000000/snapshot.zip`

</details>

<details><summary><code>ai_v9_81_REVIVE1c_0830</code> — 0.838 GB freed, 45 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_35244768_steps.json` — first, every-10th
- `checkpoints/checkpoint_35244768_steps.zip` — first, every-10th
- `checkpoints/checkpoint_36744768_steps.json` — every-10th
- `checkpoints/checkpoint_36744768_steps.zip` — every-10th
- `checkpoints/checkpoint_38244768_steps.json` — every-10th
- `checkpoints/checkpoint_38244768_steps.zip` — every-10th
- `checkpoints/checkpoint_38994768_steps.json` — last
- `checkpoints/checkpoint_38994768_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_35394768_steps.json`
- `checkpoints/checkpoint_35394768_steps.zip`
- `checkpoints/checkpoint_35544768_steps.json`
- `checkpoints/checkpoint_35544768_steps.zip`
- `checkpoints/checkpoint_35694768_steps.json`
- `checkpoints/checkpoint_35694768_steps.zip`
- `checkpoints/checkpoint_35844768_steps.json`
- `checkpoints/checkpoint_35844768_steps.zip`
- `checkpoints/checkpoint_35994768_steps.json`
- `checkpoints/checkpoint_35994768_steps.zip`
- `checkpoints/checkpoint_36144768_steps.json`
- `checkpoints/checkpoint_36144768_steps.zip`
- `checkpoints/checkpoint_36294768_steps.json`
- `checkpoints/checkpoint_36294768_steps.zip`
- `checkpoints/checkpoint_36444768_steps.json`
- `checkpoints/checkpoint_36444768_steps.zip`
- `checkpoints/checkpoint_36594768_steps.json`
- `checkpoints/checkpoint_36594768_steps.zip`
- `checkpoints/checkpoint_36894768_steps.json`
- `checkpoints/checkpoint_36894768_steps.zip`
- `checkpoints/checkpoint_37044768_steps.json`
- `checkpoints/checkpoint_37044768_steps.zip`
- `checkpoints/checkpoint_37194768_steps.json`
- `checkpoints/checkpoint_37194768_steps.zip`
- `checkpoints/checkpoint_37344768_steps.json`
- `checkpoints/checkpoint_37344768_steps.zip`
- `checkpoints/checkpoint_37494768_steps.json`
- `checkpoints/checkpoint_37494768_steps.zip`
- `checkpoints/checkpoint_37644768_steps.json`
- `checkpoints/checkpoint_37644768_steps.zip`
- `checkpoints/checkpoint_37794768_steps.json`
- `checkpoints/checkpoint_37794768_steps.zip`
- `checkpoints/checkpoint_37944768_steps.json`
- `checkpoints/checkpoint_37944768_steps.zip`
- `checkpoints/checkpoint_38094768_steps.json`
- `checkpoints/checkpoint_38094768_steps.zip`
- `checkpoints/checkpoint_38394768_steps.json`
- `checkpoints/checkpoint_38394768_steps.zip`
- `checkpoints/checkpoint_38544768_steps.json`
- `checkpoints/checkpoint_38544768_steps.zip`
- `checkpoints/checkpoint_38694768_steps.json`
- `checkpoints/checkpoint_38694768_steps.zip`
- `checkpoints/checkpoint_38844768_steps.json`
- `checkpoints/checkpoint_38844768_steps.zip`
- `eval_traces/step_36000000/snapshot.zip`

</details>

<details><summary><code>ai_v9_82_REFOLD1_0830</code> — 0.947 GB freed, 51 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_28265184_steps.json` — first, every-10th
- `checkpoints/checkpoint_28265184_steps.zip` — first, every-10th
- `checkpoints/checkpoint_29765184_steps.json` — every-10th
- `checkpoints/checkpoint_29765184_steps.zip` — every-10th
- `checkpoints/checkpoint_31265184_steps.json` — every-10th
- `checkpoints/checkpoint_31265184_steps.zip` — every-10th
- `checkpoints/checkpoint_32465184_steps.json` — last
- `checkpoints/checkpoint_32465184_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_28415184_steps.json`
- `checkpoints/checkpoint_28415184_steps.zip`
- `checkpoints/checkpoint_28565184_steps.json`
- `checkpoints/checkpoint_28565184_steps.zip`
- `checkpoints/checkpoint_28715184_steps.json`
- `checkpoints/checkpoint_28715184_steps.zip`
- `checkpoints/checkpoint_28865184_steps.json`
- `checkpoints/checkpoint_28865184_steps.zip`
- `checkpoints/checkpoint_29015184_steps.json`
- `checkpoints/checkpoint_29015184_steps.zip`
- `checkpoints/checkpoint_29165184_steps.json`
- `checkpoints/checkpoint_29165184_steps.zip`
- `checkpoints/checkpoint_29315184_steps.json`
- `checkpoints/checkpoint_29315184_steps.zip`
- `checkpoints/checkpoint_29465184_steps.json`
- `checkpoints/checkpoint_29465184_steps.zip`
- `checkpoints/checkpoint_29615184_steps.json`
- `checkpoints/checkpoint_29615184_steps.zip`
- `checkpoints/checkpoint_29915184_steps.json`
- `checkpoints/checkpoint_29915184_steps.zip`
- `checkpoints/checkpoint_30065184_steps.json`
- `checkpoints/checkpoint_30065184_steps.zip`
- `checkpoints/checkpoint_30215184_steps.json`
- `checkpoints/checkpoint_30215184_steps.zip`
- `checkpoints/checkpoint_30365184_steps.json`
- `checkpoints/checkpoint_30365184_steps.zip`
- `checkpoints/checkpoint_30515184_steps.json`
- `checkpoints/checkpoint_30515184_steps.zip`
- `checkpoints/checkpoint_30665184_steps.json`
- `checkpoints/checkpoint_30665184_steps.zip`
- `checkpoints/checkpoint_30815184_steps.json`
- `checkpoints/checkpoint_30815184_steps.zip`
- `checkpoints/checkpoint_30965184_steps.json`
- `checkpoints/checkpoint_30965184_steps.zip`
- `checkpoints/checkpoint_31115184_steps.json`
- `checkpoints/checkpoint_31115184_steps.zip`
- `checkpoints/checkpoint_31415184_steps.json`
- `checkpoints/checkpoint_31415184_steps.zip`
- `checkpoints/checkpoint_31565184_steps.json`
- `checkpoints/checkpoint_31565184_steps.zip`
- `checkpoints/checkpoint_31715184_steps.json`
- `checkpoints/checkpoint_31715184_steps.zip`
- `checkpoints/checkpoint_31865184_steps.json`
- `checkpoints/checkpoint_31865184_steps.zip`
- `checkpoints/checkpoint_32015184_steps.json`
- `checkpoints/checkpoint_32015184_steps.zip`
- `checkpoints/checkpoint_32165184_steps.json`
- `checkpoints/checkpoint_32165184_steps.zip`
- `checkpoints/checkpoint_32315184_steps.json`
- `checkpoints/checkpoint_32315184_steps.zip`
- `eval_traces/step_30000000/snapshot.zip`

</details>

<details><summary><code>ai_v9_91_COMPFOLD_0831</code> — 0.947 GB freed, 51 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_28265184_steps.json` — first, every-10th
- `checkpoints/checkpoint_28265184_steps.zip` — first, every-10th
- `checkpoints/checkpoint_29765184_steps.json` — every-10th
- `checkpoints/checkpoint_29765184_steps.zip` — every-10th
- `checkpoints/checkpoint_31265184_steps.json` — every-10th
- `checkpoints/checkpoint_31265184_steps.zip` — every-10th
- `checkpoints/checkpoint_32497344_steps.json` — last
- `checkpoints/checkpoint_32497344_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_28415184_steps.json`
- `checkpoints/checkpoint_28415184_steps.zip`
- `checkpoints/checkpoint_28565184_steps.json`
- `checkpoints/checkpoint_28565184_steps.zip`
- `checkpoints/checkpoint_28715184_steps.json`
- `checkpoints/checkpoint_28715184_steps.zip`
- `checkpoints/checkpoint_28865184_steps.json`
- `checkpoints/checkpoint_28865184_steps.zip`
- `checkpoints/checkpoint_29015184_steps.json`
- `checkpoints/checkpoint_29015184_steps.zip`
- `checkpoints/checkpoint_29165184_steps.json`
- `checkpoints/checkpoint_29165184_steps.zip`
- `checkpoints/checkpoint_29315184_steps.json`
- `checkpoints/checkpoint_29315184_steps.zip`
- `checkpoints/checkpoint_29465184_steps.json`
- `checkpoints/checkpoint_29465184_steps.zip`
- `checkpoints/checkpoint_29615184_steps.json`
- `checkpoints/checkpoint_29615184_steps.zip`
- `checkpoints/checkpoint_29915184_steps.json`
- `checkpoints/checkpoint_29915184_steps.zip`
- `checkpoints/checkpoint_30065184_steps.json`
- `checkpoints/checkpoint_30065184_steps.zip`
- `checkpoints/checkpoint_30215184_steps.json`
- `checkpoints/checkpoint_30215184_steps.zip`
- `checkpoints/checkpoint_30365184_steps.json`
- `checkpoints/checkpoint_30365184_steps.zip`
- `checkpoints/checkpoint_30515184_steps.json`
- `checkpoints/checkpoint_30515184_steps.zip`
- `checkpoints/checkpoint_30665184_steps.json`
- `checkpoints/checkpoint_30665184_steps.zip`
- `checkpoints/checkpoint_30815184_steps.json`
- `checkpoints/checkpoint_30815184_steps.zip`
- `checkpoints/checkpoint_30965184_steps.json`
- `checkpoints/checkpoint_30965184_steps.zip`
- `checkpoints/checkpoint_31115184_steps.json`
- `checkpoints/checkpoint_31115184_steps.zip`
- `checkpoints/checkpoint_31415184_steps.json`
- `checkpoints/checkpoint_31415184_steps.zip`
- `checkpoints/checkpoint_31565184_steps.json`
- `checkpoints/checkpoint_31565184_steps.zip`
- `checkpoints/checkpoint_31715184_steps.json`
- `checkpoints/checkpoint_31715184_steps.zip`
- `checkpoints/checkpoint_31865184_steps.json`
- `checkpoints/checkpoint_31865184_steps.zip`
- `checkpoints/checkpoint_32015184_steps.json`
- `checkpoints/checkpoint_32015184_steps.zip`
- `checkpoints/checkpoint_32197344_steps.json`
- `checkpoints/checkpoint_32197344_steps.zip`
- `checkpoints/checkpoint_32347344_steps.json`
- `checkpoints/checkpoint_32347344_steps.zip`
- `eval_traces/step_30000000/snapshot.zip`

</details>

<details><summary><code>ai_v9_93_R5F01_0831</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_95_R5F03_0831</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_97_R5F05_0831</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_99_R5F07_0831</code> — 0.656 GB freed, 35 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_25217760_steps.json` — first, every-10th
- `checkpoints/checkpoint_25217760_steps.zip` — first, every-10th
- `checkpoints/checkpoint_26717760_steps.json` — every-10th
- `checkpoints/checkpoint_26717760_steps.zip` — every-10th
- `checkpoints/checkpoint_28067760_steps.json` — last
- `checkpoints/checkpoint_28067760_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_25367760_steps.json`
- `checkpoints/checkpoint_25367760_steps.zip`
- `checkpoints/checkpoint_25517760_steps.json`
- `checkpoints/checkpoint_25517760_steps.zip`
- `checkpoints/checkpoint_25667760_steps.json`
- `checkpoints/checkpoint_25667760_steps.zip`
- `checkpoints/checkpoint_25817760_steps.json`
- `checkpoints/checkpoint_25817760_steps.zip`
- `checkpoints/checkpoint_25967760_steps.json`
- `checkpoints/checkpoint_25967760_steps.zip`
- `checkpoints/checkpoint_26117760_steps.json`
- `checkpoints/checkpoint_26117760_steps.zip`
- `checkpoints/checkpoint_26267760_steps.json`
- `checkpoints/checkpoint_26267760_steps.zip`
- `checkpoints/checkpoint_26417760_steps.json`
- `checkpoints/checkpoint_26417760_steps.zip`
- `checkpoints/checkpoint_26567760_steps.json`
- `checkpoints/checkpoint_26567760_steps.zip`
- `checkpoints/checkpoint_26867760_steps.json`
- `checkpoints/checkpoint_26867760_steps.zip`
- `checkpoints/checkpoint_27017760_steps.json`
- `checkpoints/checkpoint_27017760_steps.zip`
- `checkpoints/checkpoint_27167760_steps.json`
- `checkpoints/checkpoint_27167760_steps.zip`
- `checkpoints/checkpoint_27317760_steps.json`
- `checkpoints/checkpoint_27317760_steps.zip`
- `checkpoints/checkpoint_27467760_steps.json`
- `checkpoints/checkpoint_27467760_steps.zip`
- `checkpoints/checkpoint_27617760_steps.json`
- `checkpoints/checkpoint_27617760_steps.zip`
- `checkpoints/checkpoint_27767760_steps.json`
- `checkpoints/checkpoint_27767760_steps.zip`
- `checkpoints/checkpoint_27917760_steps.json`
- `checkpoints/checkpoint_27917760_steps.zip`
- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>v8rep_p1_A_0905</code> — 0.045 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278669097_steps.json` — first, every-10th
- `checkpoints/checkpoint_278669097_steps.zip` — first, every-10th
- `checkpoints/checkpoint_279651074_steps.json` — last
- `checkpoints/checkpoint_279651074_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>v8rep_p1_B_0905</code> — 0.045 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278665344_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_278665344_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>v8rep_p1_C_0905</code> — 0.045 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278665177_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_278665177_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>v8rep_p2loss_A_0905</code> — 0.045 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278664287_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_278664287_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>v8rep_p2loss_B_0905</code> — 0.045 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278671312_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_278671312_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

<details><summary><code>v8rep_p2loss_C_0905</code> — 0.045 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278670065_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_278670065_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `snapshots`

</details>

## To actually apply this

```bash
cd /home/goodlad/dev/gen3ai && \
export PYTHONPATH=$PYTHONPATH:src && \
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
  designs/research_state/measurements/archive_grooming_dryrun.py \
  --policy tiered --apply
```

Run it from the **main checkout** — `models/` exists only there — and read *REVIEW BEFORE APPLYING* first. `--policy standing` (the default) is the gentler pass and is still available unchanged.

**Nothing was deleted in this pass — this was a dry run, and it wrote only the two report files.**
