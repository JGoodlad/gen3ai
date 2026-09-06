# Archive-grooming DRY RUN — `models/`

*policy: **tiered***

*Generated 2026-09-06T15:33:31-0700 · `/home/goodlad/dev/gen3ai/models`*

> **NOTHING WAS DELETED IN THIS PASS.** This is a census; the plan below is what the retention policy *would* do, and it was produced with `--apply` absent.

## Headline

| | |
|---|---|
| runs in the archive | **218** |
| total size | **257.428 GB** |
| … of which physically under `models/` | 242.818 GB |
| … in 8 SYMLINKED run dirs (elsewhere on disk) | 14.61 GB |
| the policy would free | **87.881 GB** (34.1%) |
| entries in the plan | 2685 |
| runs with a non-empty plan | 108 |
| LIVE / REFERENCED / CLOSED | 100 / 69 / 49 |
| runs vetoed by a named file | 3 |
| **CLOSED runs needing review** | **10** |
| tier-4 runs REFUSED (no resolvable final model) | 1 |

## The TIERED policy

> **The owner's reason for tier 4, verbatim (2026-09-06):** *"Yes, please work on a reasonable retention policy, especially pre ai_v8 eras, as we are unlikely to need anything from them as there wasn't a 'novel' outcome, more getting the pattern established and us able to make meaningful progress."*

| tier | who | GB now | GB freed | runs | what happens |
|---:|---|---:|---:|---:|---|
| 0 | LIVE | 112.334 | 0.0 | 100 | live, or reached for by something live — untouched |
| 1 | REFERENCED | 97.578 | 45.591 | 69 | named by a script, a measurement artifact, the ledger tail, or another run's model graph — standing policy + snapshots rule |
| 2 | v9+ CLOSED | 2.043 | 1.629 | 1 | standing policy + snapshots rule |
| 3 | v8 CLOSED | 11.969 | 8.647 | 12 | first + last + latest.txt pin (no every-10th) + snapshots rule |
| 4 | PRE-v8 | 33.503 | 32.014 | 36 | AGGRESSIVE keep-list — the era's record survives, the weights do not |

### The snapshots rule

A self-play pool is kept **only** when some run forks this run — a fork auto-seeds its parent's pool, so the zips *and* `summary.json` / `win_rate_vs_bots.txt` / `model_config.json` are load-bearing — or a committed script names the run as a `--stable-opponents` / `--exploiter` / pool source. Otherwise `snapshots/` goes whole.

| | |
|---|---:|
| pools KEPT | 26 (12.963 GB) |
| pools FREED | 50 (19.085 GB) |
| runs with no pool | 35 |
| tier-0 runs (rule not applied) | 107 |
| **PROPOSED** further thinning of the KEPT pools (every 4th + newest) | **8.695 GB** |

The thinning is a **proposal, not a plan** — no kept pool loses a byte in this policy. It is reported so the 8.695 GB is a number the owner can decide on rather than a discovery made later.

### Every pool decision, per run

| run | tier | pool GB | decision | why |
|---|---:|---:|---|---|
| `ai_v6_13_outgoing_dmg_0620` | 1 | 0.903 | KEEP | KEPT — a fork parent / pool source: ai_v6_13_outgoing_dmg_0620_exp_v1 (argv fork_parent), ai_v6_13_outgoing_dmg_0620_exp_v1 (fork_parent), ai_v6_13_outgoing_dmg_0620_exploiter_v1 (argv fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v6_13_outgoing_dmg_0620_exp_v1` | 1 | 0.903 | KEEP | KEPT — a committed script names this run: src/main/launcher_test.py |
| `ai_v6_11_typed_hp_0619` | 4 | 0.876 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v7_04_opd_selfdistill_0702` | 4 | 0.868 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v7_03_belief_shape_0630` | 4 | 0.868 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v6_09_dmg_reattend_N_0617` | 4 | 0.78 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v9_27_extremedial_probe_0823` | 1 | 0.711 | KEEP | KEPT — a committed script names this run: src/agents/training/exploiter_ladder.py, src/agents/training/exploiter_ladder_test.py |
| `ai_v9_25_E4_baitbot_0822` | 1 | 0.711 | KEEP | KEPT — a fork parent / pool source: ai_v9_27_extremedial_probe_0823 (argv fork_parent), ai_v9_27_extremedial_probe_0823 (fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v6_03_win_pred_N_0614` | 4 | 0.644 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v9_24_E3_substrate_on_0822` | 1 | 0.64 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_23_E2_substrate_on_0822` | 1 | 0.64 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_22_E1_substrate_on_0821` | 1 | 0.64 | KEEP | KEPT — a fork parent / pool source: ai_v9_25_E4_baitbot_0822 (argv fork_parent), ai_v9_25_E4_baitbot_0822 (fork_parent), ai_v9_26_baitent_probe_0823 (argv fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v6_11_unified_obs_fixed_0618` | 4 | 0.608 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v5_11_tail2_53m_0611` | 4 | 0.587 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v9_09_gen8_beliefs_threat_inject_0811` | 2 | 0.585 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_71_R3ACTIONHI_0828` | 1 | 0.582 | KEEP | KEPT — a committed script names this run: designs/ai_v12/team_slate_build.py |
| `ai_v9_70_R3ACTION_0828` | 1 | 0.582 | KEEP | KEPT — a committed script names this run: designs/ai_v12/team_slate_build.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py |
| `ai_v5_9_attend_unrevealed_56m_0610` | 1 | 0.571 | KEEP | KEPT — a fork parent / pool source: ai_v5_10_tail1_23_0611 (argv pool_source), ai_v5_11_tail2_53m_0611 (argv pool_source), ai_v5_12_bias_05_N_0612 (argv pool_source) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v5_6_stable_70m_0608` | 1 | 0.566 | KEEP | KEPT — a fork parent / pool source: ai_v5_7_switch_bias_41m_0609 (argv pool_source) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v5_5_popart_50m_0607` | 1 | 0.566 | KEEP | KEPT — a fork parent / pool source: ai_v5_6_stable_70m_0608 (argv pool_source), ai_v5_7_switch_bias_41m_0609 (argv pool_source) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v6_01_belief_53m_0613` | 4 | 0.551 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v5_13_shape_pbrs_43m_0612` | 1 | 0.528 | KEEP | KEPT — a fork parent / pool source: ai_v6_01_belief_53m_0613 (argv pool_source) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v9_12_gen10_t0prior_0814` | 1 | 0.528 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_15_gen13_hb_events_stack_0817` | 1 | 0.517 | KEEP | KEPT — a fork parent / pool source: RETIRED_c5fork_control_gen13base_0817 (argv fork_parent), RETIRED_c5fork_control_gen13base_0817 (fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v9_14_gen12_h_entitypool_shaping_0816` | 1 | 0.514 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/obs_conditioning_probe.py |
| `ai_v9_72_R3SELF_0828` | 1 | 0.509 | KEEP | KEPT — a committed script names this run: designs/ai_v12/team_slate_build.py, designs/research_state/measurements/plain_training_robbery.py |
| `ai_v9_60_R2TOPK_0827` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_61_R2KL_0827` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_49_G2_advgate_0826` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_52_G1p_matched_0826` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_38_fdA_coef03_0825` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_39_fdB_lossonly_0825` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_42_fdE_single_0825` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_62_R2PLAIN_0827` | 1 | 0.509 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/plain_training_robbery.py, designs/research_state/measurements/representational_richness_transfer_forward.py |
| `ai_v9_40_fdC_ecology_0825` | 1 | 0.509 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_16_gen14_framedel_v91_0817` | 1 | 0.505 | KEEP | KEPT — a fork parent / pool source: DISCARDED_tdaux_control_n16_0818 (argv fork_parent), DISCARDED_tdaux_control_n16_0818 (fork_parent), ai_v9_17_tdaux_control_0818 (argv fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v9_18_gen15_v8rewards_0818` | 1 | 0.504 | KEEP | KEPT — a committed script names this run: designs/research_state/measurements/obs_conditioning_probe.py, src/main/prober/loops.py |
| `ai_v6_04_unified_inc_N_0615` | 4 | 0.501 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v9_20_tdaux_rung2_lam30_0820` | 1 | 0.498 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_20_tdaux_rung2_lam10_0820` | 1 | 0.498 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_20_tdaux_rung2_lam00_0820` | 1 | 0.498 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_13_gen11_labelonly_winprob_0815` | 1 | 0.496 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v5_3_vf_coef_clip_50m_0606` | 4 | 0.481 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v9_10_gen9_intent_distcritic_0813` | 1 | 0.475 | KEEP | KEPT — a committed script names this run: src/agents/model/intent_move_cell_test.py |
| `ai_v9_50_fdF_p1c_0826` | 1 | 0.473 | KEEP | KEPT — a fork parent / pool source: ai_v9_51_fdF_p2c_0826 (argv fork_parent), ai_v9_51_fdF_p2c_0826 (fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v9_51_fdF_p2c_0826` | 1 | 0.473 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v5_8_split_inc_dmg_38m_0610` | 1 | 0.457 | KEEP | KEPT — a fork parent / pool source: ai_v5_10_tail1_23_0611 (argv pool_source), ai_v5_11_tail2_53m_0611 (argv pool_source), ai_v5_12_bias_05_N_0612 (argv pool_source) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v5_7_switch_bias_41m_0609` | 4 | 0.453 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v7_01_teacher_0626` | 1 | 0.434 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_19_gen16_mechanics_0819` | 1 | 0.427 | KEEP | KEPT — a fork parent / pool source: ai_v9_20_tdaux_rung2_lam00_0820 (argv fork_parent), ai_v9_20_tdaux_rung2_lam00_0820 (fork_parent), ai_v9_20_tdaux_rung2_lam10_0820 (argv fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v9_21_gen17_pfspoff_0820` | 1 | 0.427 | KEEP | KEPT — a fork parent / pool source: ai_v9_22_E1_substrate_on_0821 (argv fork_parent), ai_v9_22_E1_substrate_on_0821 (fork_parent), ai_v9_23_E2_substrate_on_0822 (argv fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v6_07_unified_topk_N_0616` | 4 | 0.38 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v6_08_unmasked_floor_N_0617` | 4 | 0.304 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v5_4_pbrs_opp_threat_50m_0607` | 4 | 0.283 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v5_12_bias_05_N_0612` | 4 | 0.235 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v5_10_tail1_23_0611` | 4 | 0.205 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v9_34_tick1_0824` | 1 | 0.182 | KEEP | KEPT — a fork parent / pool source: ai_v9_35_tick1_exploit_0824 (argv fork_parent), ai_v9_35_tick1_exploit_0824 (argv pool_source), ai_v9_35_tick1_exploit_0824 (fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v6_06_unified_all_N_0616` | 4 | 0.174 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v6_02_belief_lat_16m_0614` | 4 | 0.125 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v8_15_retention_A_frozen_0726` | 1 | 0.089 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v8_02_zarch_teampfsp_0718` | 3 | 0.089 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v7_19_combined_0716` | 4 | 0.087 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v5_2_native_selfplay_50m_0606` | 4 | 0.085 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v9_17_tdaux_lam1_0818` | 1 | 0.084 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_17_tdaux_lam3_0818` | 1 | 0.084 | KEEP | KEPT — a committed script names this run: src/agents/training/poke_env_gaps/faint_attribution_fuzz_test.py |
| `ai_v9_37_tick1_dosext_0825` | 1 | 0.073 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_58_R2CTRL_0827` | 1 | 0.073 | KEEP | KEPT — a committed script names this run: designs/ai_v12/team_slate_build.py, designs/research_state/measurements/plain_training_robbery.py |
| `ai_v7_21_fitnet_valuefeat_ab_0717` | 4 | 0.043 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v7_20_valuedistill_ab_0717` | 4 | 0.043 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v7_18_distill_4teacher_0716` | 1 | 0.043 | KEEP | KEPT — a fork parent / pool source: ai_v7_19_combined_0716 (argv fork_parent), ai_v7_19_combined_0716 (fork_parent) (a fork auto-seeds its parent's pool, so the zips AND the metadata are load-bearing) |
| `ai_v7_16_distill_tss_mvp_0715` | 4 | 0.043 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v9_17_tdaux_control_0818` | 1 | 0.042 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `DISCARDED_tdaux_control_n16_0818` | 1 | 0.041 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v9_11_gen10_intentfull_compiled_0814` | 1 | 0.041 | DELETE | FREED — no run forks it, no committed script names it as a --stable-opponents / --exploiter / pool source |
| `ai_v6_04_unified_all_half_batch_N_0616` | 4 | 0.035 | DELETE | tier 4 — the whole run is reduced to its record + final model |
| `ai_v6_10_unified_obs_0618` | 4 | 0.0 | DELETE | tier 4 — the whole run is reduced to its record + final model |

### Tier 4 — what each pre-v8 run keeps

The keep-list is the policy. `resolve_model_ref` picks the ONE model file, and the **rung** it fired on is recorded, because a bare run dir has meant different files at different times (`gen3_last_snapshot_resolution_v1`).

| run | GB freed | the kept model | rung | steps |
|---|---:|---|---|---:|
| `ai_v6_01_belief_53m_0613` | 3.006 | `final_model_interrupted.zip` | latest_txt | 53254107 |
| `ai_v5_11_tail2_53m_0611` | 2.97 | `final_model_interrupted.zip` | latest_txt | 52690944 |
| `ai_v5_3_vf_coef_clip_50m_0606` | 2.505 | `final_model_interrupted.zip` | latest_txt | 50292601 |
| `ai_v5_7_switch_bias_41m_0609` | 2.261 | `final_model_interrupted.zip` | latest_txt | 41241048 |
| `ai_v7_04_opd_selfdistill_0702` | 1.928 | `final_model_interrupted.zip` | latest_txt | 135882052 |
| `ai_v5_4_pbrs_opp_threat_50m_0607` | 1.8 | `final_model_interrupted.zip` | latest_txt | 35805245 |
| `ai_v7_03_belief_shape_0630` | 1.454 | `final_model_interrupted.zip` | latest_txt | 59979799 |
| `ai_v6_11_typed_hp_0619` | 1.355 | `final_model_interrupted.zip` | latest_txt | 46055756 |
| `ai_v5_12_bias_05_N_0612` | 1.32 | `checkpoint_23309510_steps.zip` | highest_checkpoint | 23309510 |
| `ai_v5_10_tail1_23_0611` | 1.29 | `final_model_interrupted.zip` | latest_txt | 23918406 |
| `ai_v6_09_dmg_reattend_N_0617` | 1.169 | `final_model_interrupted.zip` | latest_txt | 42663936 |
| `ai_v5_2_native_selfplay_50m_0606` | 1.11 | `final_model_interrupted.zip` | latest_txt | 30272421 |
| `ai_v6_03_win_pred_N_0614` | 1.085 | `final_model_interrupted.zip` | latest_txt | 59001270 |
| `ai_v6_11_unified_obs_fixed_0618` | 1.03 | `final_model_interrupted.zip` | latest_txt | 34549487 |
| `ai_v6_04_unified_inc_N_0615` | 0.855 | `final_model_interrupted.zip` | latest_txt | 39268523 |
| `ai_v6_07_unified_topk_N_0616` | 0.764 | `final_model_interrupted.zip` | latest_txt | 32036565 |
| `ai_v6_08_unmasked_floor_N_0617` | 0.585 | `final_model_interrupted.zip` | latest_txt | 22856749 |
| `ai_v7_09_tss_bots_pubval_0708` | 0.503 | `final_model_interrupted.zip` | latest_txt | 57984653 |
| `ai_v6_06_unified_all_N_0616` | 0.472 | `final_model_interrupted.zip` | latest_txt | 26208576 |
| `ai_v7_08_tss_bots_0707` | 0.468 | `final_model_interrupted.zip` | latest_txt | 56915088 |
| `ai_v7_19_combined_0716` | 0.465 | `final_model_interrupted.zip` | latest_txt | 175988511 |
| `ai_v6_13_outgoing_dmg_0620_exploiter_v2` | 0.446 | `final_model_interrupted.zip` | latest_txt | 36047373 |
| `ai_v6_13_outgoing_dmg_0620_exploiter_v1` | 0.395 | `final_model_interrupted.zip` | latest_txt | 128209189 |
| `ai_v7_06_tss_temp_anneal_0706` | 0.37 | `final_model_interrupted.zip` | latest_txt | 33130852 |
| `ai_v6_02_belief_lat_16m_0614` | 0.339 | `final_model_interrupted.zip` | latest_txt | 16837925 |
| `ai_v7_07_tss_temp_ratchet_0707` | 0.315 | `final_model_interrupted.zip` | latest_txt | 18874368 |
| `ai_v7_11_tss_exploiter_nopubval` | 0.308 | `final_model_interrupted.zip` | latest_txt | 26499040 |
| `ai_v7_21_fitnet_valuefeat_ab_0717` | 0.292 | `final_model_interrupted.zip` | latest_txt | 160296140 |
| `ai_v6_04_unified_all_half_batch_N_0616` | 0.28 | `final_model_interrupted.zip` | latest_txt | 14188106 |
| `ai_v7_16_distill_tss_mvp_0715` | 0.28 | `final_model_interrupted.zip` | latest_txt | 154979822 |
| `ai_v6_10_unified_obs_0618` | 0.272 | `final_model_interrupted.zip` | latest_txt | 10724858 |
| `ai_v7_20_valuedistill_ab_0717` | 0.265 | `final_model_interrupted.zip` | latest_txt | 152957470 |
| `ai_v7_01_teacher_0626_oom1` | 0.057 | `final_model_interrupted.zip` | latest_txt | 1430572 |
| `ai_v7_05_tss_specialist_0703_aborted_noeval` | 0.001 | `final_model_interrupted.zip` | latest_txt | 196608 |
| `ai_v7_20_valuedistill_SMOKE` | 0.0 | `final_model_interrupted.zip` | latest_txt | 148402976 |
| `warmstart_generic_0715` | 0 | **REFUSED** — no final model resolves for /home/goodlad/dev/gen3ai/models/warmstart_generic_0715: FileNotFoundError: run spec: no model .zip found for '/home/goodlad/dev/gen3ai/models/warmstart_generic_0715' (expected a run dir carrying latest.txt / checkpoints/ / final_model.zip / best_model/best_model.zip, a direct .zip, or a run dir + @step). | — | — |

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
| 1 | `ai_v6_01_belief_53m_0613` | ai_v6 | 3.006 | 75 | 1 |
| 2 | `ai_v5_11_tail2_53m_0611` | ai_v5 | 2.97 | 76 | 1 |
| 3 | `ai_v5_3_vf_coef_clip_50m_0606` | ai_v5 | 2.505 | 68 | 1 |
| 4 | `ai_v9_34_tick1_0824` | ai_v9 | 2.308 | 116 | 4 |
| 5 | `ai_v5_7_switch_bias_41m_0609` | ai_v5 | 2.261 | 59 | 1 |
| 6 | `ai_v7_04_opd_selfdistill_0702` | ai_v7 | 1.928 | 14 | 1 |
| 7 | `ai_v5_4_pbrs_opp_threat_50m_0607` | ai_v5 | 1.8 | 50 | 1 |
| 8 | `ai_v9_09_gen8_beliefs_threat_inject_0811` | ai_v9 | 1.629 | 10 | 12 |
| 9 | `ai_v9_12_gen10_t0prior_0814` | ai_v9 | 1.541 | 12 | 12 |
| 10 | `ai_v8_12_defensive20_exploiter_0724` | ai_v8 | 1.503 | 38 | 11 |
| 11 | `ai_v9_13_gen11_labelonly_winprob_0815` | ai_v9 | 1.49 | 12 | 11 |
| 12 | `ai_v7_03_belief_shape_0630` | ai_v7 | 1.454 | 7 | 1 |
| 13 | `ai_v6_11_typed_hp_0619` | ai_v6 | 1.355 | 6 | 1 |
| 14 | `ai_v5_12_bias_05_N_0612` | ai_v5 | 1.32 | 32 | 1 |
| 15 | `ai_v5_10_tail1_23_0611` | ai_v5 | 1.29 | 32 | 1 |
| 16 | `ai_v8_07_semistall564_scratch_0722` | ai_v8 | 1.202 | 34 | 10 |
| 17 | `ai_v6_09_dmg_reattend_N_0617` | ai_v6 | 1.169 | 5 | 1 |
| 18 | `ai_v9_37_tick1_dosext_0825` | ai_v9 | 1.165 | 56 | 2 |
| 19 | `ai_v9_60_R2TOPK_0827` | ai_v9 | 1.165 | 34 | 1 |
| 20 | `ai_v9_61_R2KL_0827` | ai_v9 | 1.165 | 34 | 1 |

## Runs vetoed because a committed file or the ledger names a file in the plan

These are excluded from the deletion set automatically.

| run | GB it would have freed | example named path | named by |
|---|---:|---|---|
| `ai_v8_14_distill3_0725` | 1.03 | `checkpoints/checkpoint_279661705_steps.json` | designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/v8_era_locality_v2.py |
| `ai_v9_31_tock1_k4_0824` | 0.656 | `eval_traces/step_26000016/snapshot.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/ai_v9_34_tick1_0824_endofrun.json |
| `ai_v9_44_tock2_v8shape_0825` | 2.077 | `eval_traces/step_30000000/snapshot.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/archive_grooming_dryrun_2026-09-06.json |

## ⚠️ SYMLINKED run dirs — the data is NOT under `models/`

These entries are symlinks into launcher worktrees, so `du -sh models/` does not see them and a deletion "in `models/`" would physically land under `.claude/worktrees/`. They are held out of the plan by default; `--follow-symlinked-runs` opts in after you have confirmed the targets are still the ones you mean.

| run | generation | GB | status | data actually lives at |
|---|---|---:|---|---|
| `ai_v9_01_gen1_edges6_40m_0804` | ai_v9 | 2.53 | LIVE | `/home/goodlad/dev/gen3ai/.claude/worktrees/gen1-run-0804/models/run_20260804_090512` |
| `ai_v9_02_gen2_full11_40m_0805` | ai_v9 | 2.48 | LIVE | `/home/goodlad/dev/gen3ai/.claude/worktrees/gen2-run-0805/models/run_20260805_060807` |
| `ai_v9_03_gen25_consequence_25m_0806` | ai_v9 | 1.737 | LIVE | `/home/goodlad/dev/gen3ai/.claude/worktrees/gen25-run-0806/models/run_20260806_160611` |
| `ai_v9_04_gen3_k6_recency_40m_0807` | ai_v9 | 2.596 | LIVE | `/home/goodlad/dev/gen3ai/.claude/worktrees/gen3-run-0807/models/run_20260807_135637` |
| `ai_v9_05_gen4_rehome_25m_0808` | ai_v9 | 1.68 | LIVE | `/home/goodlad/dev/gen3ai/.claude/worktrees/gen4-run-0808/models/run_20260808_212910` |
| `ai_v9_06_gen5_no_concat_0809` | ai_v9 | 1.474 | LIVE | `/home/goodlad/dev/gen3ai/.claude/worktrees/gen5-run-0809/models/ai_v9_06_gen5_no_concat_0809` |
| `ai_v9_07_gen6_seed_vicreg_0810` | ai_v9 | 1.193 | LIVE | `/home/goodlad/dev/gen3ai/.claude/worktrees/gen6-run-0810/models/ai_v9_07_gen6_seed_vicreg_0810` |
| `ai_v9_08_gen7_seed_quantile_0811` | ai_v9 | 0.919 | LIVE | `/home/goodlad/dev/gen3ai/.claude/worktrees/gen7-run-0811/models/ai_v9_08_gen7_seed_quantile_0811` |

## ⚠️ REVIEW BEFORE APPLYING — CLOSED runs the ledger names outside its tail

The tail window is what protects a run; this section makes its EDGE visible rather than silent. Each of these has a non-empty plan **and** is named somewhere higher up `ledger.md`, so a banked result may still rest on it. They are still in the deletion set — read them before running `--apply`, and widen `--ledger-tail-lines` (or delete the run's entry from the plan) if any should be kept.

| run | generation | GB freed | named at | prose mentions |
|---|---|---:|---|---:|
| `ai_v5_11_tail2_53m_0611` | ai_v5 | 2.97 | `ledger.md:85` | 6 |
| `ai_v5_12_bias_05_N_0612` | ai_v5 | 1.32 | `ledger.md:87` | 6 |
| `ai_v9_21_gen17_pfspoff_0820` | ai_v9 | 0.845 | `ledger.md:2265` | 29 |
| `ai_v9_25_E4_baitbot_0822` | ai_v9 | 0.323 | `ledger.md:2039` | 18 |
| `ai_v9_27_extremedial_probe_0823` | ai_v9 | 0.036 | `ledger.md:3722` | 12 |
| `ai_v9_34_tick1_0824` | ai_v9 | 2.308 | `ledger.md:11019` | 20 |
| `ai_v9_37_tick1_dosext_0825` | ai_v9 | 1.165 | `ledger.md:11020` | 16 |
| `ai_v9_38_fdA_coef03_0825` | ai_v9 | 1.165 | `ledger.md:11020` | 12 |
| `ai_v9_58_R2CTRL_0827` | ai_v9 | 0.656 | `ledger.md:11149` | 31 |
| `ai_v9_62_R2PLAIN_0827` | ai_v9 | 0.656 | `ledger.md:8151` | 28 |

## Per-run census

Sizes in GB. `plan GB` is 0 for every run that is not CLOSED.

| run | gen | cfg | tier | status | ckpts | best | snaps | traces | tb | other | total | plan GB |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ai_v8_03_zarch_control_0718` | ai_v8 | 45 | 0 | LIVE | 4.938 | 0.045 | 0.416 | 1.631 | 0.011 | 0.412 | 7.459 | 0.0 |
| `ai_v9_29_rev1_0823` | ai_v9 | 101 | 0 | LIVE | 4.625 | 0.036 | 0.436 | 0.793 | 0.005 | 0.075 | 6.092 | 0.0 |
| `ai_v5_6_stable_70m_0608` | ai_v5 | 7 | 1 | REFERENCED | 0.311 | 0.028 | 0.566 | 0.075 | 0.001 | 2.691 | 3.987 | 0.255 |
| `ai_v9_34_tick1_0824` | ai_v9 | 101 | 1 | REFERENCED | 2.404 | 0.036 | 0.182 | 0.331 | 0.002 | 0.075 | 3.161 | 2.308 |
| `ai_v5_9_attend_unrevealed_56m_0610` | ai_v5 | 8 | 1 | REFERENCED | 0.257 | 0.029 | 0.571 | 0.1 | 0.001 | 2.115 | 3.08 | 0.2 |
| `ai_v6_01_belief_53m_0613` | ai_v6 | 16 | 4 | CLOSED | 0.245 | 0.031 | 0.551 | 0.117 | 0.002 | 2.087 | 3.039 | 3.006 |
| `ai_v5_5_popart_50m_0607` | ai_v5 | 6 | 1 | REFERENCED | 0.255 | 0.028 | 0.566 | 0.071 | 0.001 | 2.068 | 3.001 | 0.198 |
| `ai_v5_11_tail2_53m_0611` | ai_v5 | 11 | 4 | CLOSED | 0.235 | 0.029 | 0.587 | 0.117 | 0.001 | 2.027 | 3.001 | 2.97 |
| `ai_v9_75_R4S3c_0829` | ai_v9 | 107 | 0 | LIVE | 2.404 | 0.036 | 0.0 | 0.342 | 0.002 | 0.074 | 2.891 | 0.0 |
| `ai_v9_74_R4S3b_0829` | ai_v9 | 107 | 0 | LIVE | 2.404 | 0.036 | 0.0 | 0.322 | 0.002 | 0.074 | 2.87 | 0.0 |
| `ai_v9_73_R4S3a_0829` | ai_v9 | 107 | 0 | LIVE | 2.367 | 0.036 | 0.0 | 0.304 | 0.002 | 0.074 | 2.814 | 0.0 |
| `ai_v9_04_gen3_k6_recency_40m_0807` | ai_v9 | 59 | 0 | LIVE | 0.598 | 0.037 | 0.747 | 1.169 | 0.002 | 0.037 | 2.596 | 0.0 |
| `ai_v9_44_tock2_v8shape_0825` | ai_v9 | 101 | 1 | REFERENCED | 2.149 | 0.036 | 0.0 | 0.285 | 0.002 | 0.074 | 2.568 | 0.0 |
| `ai_v5_3_vf_coef_clip_50m_0606` | ai_v5 | 3 | 4 | CLOSED | 0.198 | 0.028 | 0.481 | 0.067 | 0.001 | 1.755 | 2.534 | 2.505 |
| `ai_v5_13_shape_pbrs_43m_0612` | ai_v5 | 15 | 1 | REFERENCED | 0.206 | 0.029 | 0.528 | 0.112 | 0.001 | 1.645 | 2.532 | 0.147 |
| `ai_v9_01_gen1_edges6_40m_0804` | ai_v9 | 56 | 0 | LIVE | 0.58 | 0.036 | 0.688 | 1.182 | 0.002 | 0.036 | 2.53 | 0.0 |
| `ai_v9_02_gen2_full11_40m_0805` | ai_v9 | 57 | 0 | LIVE | 0.58 | 0.036 | 0.689 | 1.132 | 0.002 | 0.036 | 2.48 | 0.0 |
| `ai_v5_7_switch_bias_41m_0609` | ai_v5 | 7 | 4 | CLOSED | 0.198 | 0.028 | 0.453 | 0.099 | 0.001 | 1.502 | 2.291 | 2.261 |
| `ai_v5_8_split_inc_dmg_38m_0610` | ai_v5 | 7 | 1 | REFERENCED | 0.171 | 0.029 | 0.457 | 0.089 | 0.001 | 1.431 | 2.19 | 0.143 |
| `ai_v9_14_gen12_h_entitypool_shaping_0816` | ai_v9 | 80 | 1 | REFERENCED | 0.385 | 0.043 | 0.514 | 1.014 | 0.003 | 0.086 | 2.053 | 1.11 |
| `ai_v9_09_gen8_beliefs_threat_inject_0811` | ai_v9 | 64 | 2 | CLOSED | 0.315 | 0.045 | 0.585 | 1.035 | 0.002 | 0.046 | 2.043 | 1.629 |
| `ai_v9_16_gen14_framedel_v91_0817` | ai_v9 | 91 | 1 | REFERENCED | 0.379 | 0.042 | 0.505 | 1.016 | 0.004 | 0.085 | 2.039 | 1.087 |
| `ai_v7_04_opd_selfdistill_0702` | ai_v7 | 42 | 4 | CLOSED | 0.608 | 0.043 | 0.868 | 0.227 | 0.01 | 0.219 | 1.983 | 1.928 |
| `ai_v9_70_R3ACTION_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.093 | 0.036 | 0.582 | 0.145 | 0.001 | 0.074 | 1.957 | 0.983 |
| `ai_v9_76_R4ACTION_0830` | ai_v9 | 107 | 0 | LIVE | 1.056 | 0.036 | 0.582 | 0.18 | 0.001 | 0.074 | 1.956 | 0.0 |
| `ai_v9_140_B2_0901` | ai_v9 | 107 | 0 | LIVE | 1.056 | 0.036 | 0.582 | 0.172 | 0.001 | 0.074 | 1.948 | 0.0 |
| `ai_v9_141_C1_0901` | ai_v9 | 107 | 0 | LIVE | 1.093 | 0.036 | 0.582 | 0.162 | 0.001 | 0.038 | 1.939 | 0.0 |
| `ai_v9_143_N2_0901` | ai_v9 | 107 | 0 | LIVE | 1.056 | 0.036 | 0.546 | 0.191 | 0.001 | 0.074 | 1.931 | 0.0 |
| `ai_v9_91_COMPFOLD_0831` | ai_v9 | 107 | 0 | LIVE | 1.056 | 0.036 | 0.582 | 0.148 | 0.001 | 0.074 | 1.924 | 0.0 |
| `ai_v9_71_R3ACTIONHI_0828` | ai_v9 | 104 | 1 | REFERENCED | 1.056 | 0.036 | 0.582 | 0.145 | 0.001 | 0.074 | 1.921 | 0.947 |
| `ai_v8_01_zarch_film_0717` | ai_v8 | 44 | 0 | LIVE | 0.938 | 0.045 | 0.017 | 0.843 | 0.002 | 0.063 | 1.914 | 0.0 |
| `ai_v9_15_gen13_hb_events_stack_0817` | ai_v9 | 89 | 1 | REFERENCED | 0.258 | 0.043 | 0.517 | 0.989 | 0.004 | 0.087 | 1.906 | 0.947 |
| `ai_v9_12_gen10_t0prior_0814` | ai_v9 | 77 | 1 | REFERENCED | 0.325 | 0.041 | 0.528 | 0.953 | 0.002 | 0.042 | 1.894 | 1.541 |
| `ai_v9_13_gen11_labelonly_winprob_0815` | ai_v9 | 77 | 1 | REFERENCED | 0.331 | 0.041 | 0.496 | 0.933 | 0.002 | 0.084 | 1.894 | 1.49 |
| `ai_v9_72_R3SELF_0828` | ai_v9 | 107 | 1 | REFERENCED | 1.093 | 0.036 | 0.509 | 0.155 | 0.001 | 0.074 | 1.893 | 0.983 |
| `ai_v9_142_N1_0901` | ai_v9 | 107 | 0 | LIVE | 1.056 | 0.036 | 0.509 | 0.183 | 0.001 | 0.074 | 1.886 | 0.0 |
| `ai_v6_13_outgoing_dmg_0620` | ai_v6 | 41 | 1 | REFERENCED | 0.542 | 0.045 | 0.903 | 0.203 | 0.007 | 0.137 | 1.841 | 0.406 |
| `ai_v9_18_gen15_v8rewards_0818` | ai_v9 | 95 | 1 | REFERENCED | 0.336 | 0.042 | 0.504 | 0.851 | 0.004 | 0.085 | 1.83 | 0.946 |
| `ai_v5_4_pbrs_opp_threat_50m_0607` | ai_v5 | 3 | 4 | CLOSED | 0.17 | 0.028 | 0.283 | 0.07 | 0.001 | 1.274 | 1.829 | 1.8 |
| `ai_v8_12_defensive20_exploiter_0724` | ai_v8 | 45 | 3 | CLOSED | 0.938 | 0.045 | 0.0 | 0.77 | 0.002 | 0.046 | 1.808 | 1.503 |
| `ai_v9_10_gen9_intent_distcritic_0813` | ai_v9 | 69 | 1 | REFERENCED | 0.256 | 0.037 | 0.475 | 0.943 | 0.002 | 0.038 | 1.756 | 0.931 |
| `ai_v9_03_gen25_consequence_25m_0806` | ai_v9 | 58 | 0 | LIVE | 0.363 | 0.036 | 0.435 | 0.842 | 0.001 | 0.036 | 1.737 | 0.0 |
| `ai_v7_02_critic_shape_0627` | ai_v7 | 42 | 0 | LIVE | 0.478 | 0.043 | 0.868 | 0.209 | 0.008 | 0.088 | 1.701 | 0.0 |
| `ai_v9_05_gen4_rehome_25m_0808` | ai_v9 | 60 | 0 | LIVE | 0.374 | 0.037 | 0.411 | 0.8 | 0.001 | 0.037 | 1.68 | 0.0 |
| `ai_v9_37_tick1_dosext_0825` | ai_v9 | 101 | 1 | REFERENCED | 1.202 | 0.036 | 0.073 | 0.185 | 0.001 | 0.074 | 1.651 | 1.165 |
| `ai_v9_19_gen16_mechanics_0819` | ai_v9 | 97 | 1 | REFERENCED | 0.285 | 0.036 | 0.427 | 0.799 | 0.004 | 0.072 | 1.629 | 0.853 |
| `ai_v9_21_gen17_pfspoff_0820` | ai_v9 | 97 | 1 | REFERENCED | 0.285 | 0.036 | 0.427 | 0.789 | 0.004 | 0.072 | 1.618 | 0.845 |
| `ai_v8_14_distill3_0725` | ai_v8 | 45 | 1 | REFERENCED | 0.625 | 0.045 | 0.089 | 0.725 | 0.001 | 0.046 | 1.54 | 0.0 |
| `ai_v9_67_R3F6e_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.205 | 0.001 | 0.038 | 1.511 | 1.093 |
| `ai_v9_40_fdC_ecology_0825` | ai_v9 | 101 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.174 | 0.001 | 0.038 | 1.509 | 1.165 |
| `ai_v8_15_retention_A_frozen_0726` | ai_v8 | 45 | 1 | REFERENCED | 0.715 | 0.045 | 0.089 | 0.604 | 0.002 | 0.046 | 1.505 | 1.135 |
| `ai_v7_03_belief_shape_0630` | ai_v7 | 42 | 4 | CLOSED | 0.304 | 0.043 | 0.868 | 0.229 | 0.004 | 0.045 | 1.502 | 1.454 |
| `ai_v9_62_R2PLAIN_0827` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.154 | 0.001 | 0.037 | 1.49 | 0.656 |
| `ai_v9_42_fdE_single_0825` | ai_v9 | 101 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.15 | 0.001 | 0.038 | 1.486 | 1.165 |
| `ai_v9_38_fdA_coef03_0825` | ai_v9 | 101 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.151 | 0.001 | 0.037 | 1.485 | 1.165 |
| `ai_v9_60_R2TOPK_0827` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.143 | 0.001 | 0.037 | 1.479 | 1.165 |
| `ai_v9_59_R2ACTION_0827` | ai_v9 | 103 | 0 | LIVE | 0.728 | 0.036 | 0.509 | 0.143 | 0.001 | 0.037 | 1.479 | 0.0 |
| `ai_v9_68_R3F6f_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.177 | 0.001 | 0.037 | 1.477 | 1.093 |
| `ai_v9_61_R2KL_0827` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.142 | 0.001 | 0.037 | 1.477 | 1.165 |
| `ai_v9_48_G1_action_0826` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.141 | 0.001 | 0.037 | 1.475 | 0.0 |
| `ai_v9_06_gen5_no_concat_0809` | ai_v9 | 61 | 0 | LIVE | 0.308 | 0.031 | 0.37 | 0.711 | 0.001 | 0.032 | 1.474 | 0.0 |
| `ai_v9_49_G2_advgate_0826` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.139 | 0.001 | 0.037 | 1.473 | 1.165 |
| `ai_v9_52_G1p_matched_0826` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.138 | 0.001 | 0.037 | 1.472 | 1.165 |
| `ai_v9_39_fdB_lossonly_0825` | ai_v9 | 101 | 1 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.137 | 0.001 | 0.037 | 1.471 | 1.165 |
| `ai_v9_63_R3F6a_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.167 | 0.001 | 0.038 | 1.463 | 1.093 |
| `ai_v9_66_R3F6d_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.167 | 0.001 | 0.037 | 1.463 | 1.093 |
| `ai_v9_64_R3F6b_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.167 | 0.001 | 0.038 | 1.462 | 1.093 |
| `ai_v9_65_R3F6c_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.161 | 0.001 | 0.038 | 1.459 | 1.093 |
| `ai_v8_07_semistall564_scratch_0722` | ai_v8 | 45 | 3 | CLOSED | 0.661 | 0.035 | 0.0 | 0.672 | 0.002 | 0.071 | 1.452 | 1.202 |
| `ai_v9_69_R3F6CURR_0828` | ai_v9 | 103 | 1 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.157 | 0.001 | 0.038 | 1.451 | 1.093 |
| `ai_v8_09_pool10_exploiter_0723` | ai_v8 | 45 | 0 | LIVE | 0.715 | 0.045 | 0.0 | 0.631 | 0.002 | 0.046 | 1.442 | 0.0 |
| `ai_v6_11_typed_hp_0619` | ai_v6 | 38 | 4 | CLOSED | 0.263 | 0.044 | 0.876 | 0.168 | 0.003 | 0.045 | 1.402 | 1.355 |
| `ai_v9_24_E3_substrate_on_0822` | ai_v9 | 97 | 1 | REFERENCED | 0.107 | 0.036 | 0.64 | 0.536 | 0.002 | 0.072 | 1.399 | 1.017 |
| `ai_v9_82_REFOLD1_0830` | ai_v9 | 107 | 0 | LIVE | 1.056 | 0.036 | 0.073 | 0.133 | 0.001 | 0.074 | 1.398 | 0.0 |
| `ai_v9_25_E4_baitbot_0822` | ai_v9 | 98 | 1 | REFERENCED | 0.107 | 0.036 | 0.711 | 0.448 | 0.002 | 0.072 | 1.381 | 0.323 |
| `ai_v5_12_bias_05_N_0612` | ai_v5 | 12 | 4 | CLOSED | 0.117 | 0.029 | 0.235 | 0.107 | 0.0 | 0.882 | 1.38 | 1.32 |
| `ai_v9_23_E2_substrate_on_0822` | ai_v9 | 97 | 1 | REFERENCED | 0.107 | 0.036 | 0.64 | 0.501 | 0.002 | 0.072 | 1.364 | 1.0 |
| `ai_v6_13_outgoing_dmg_0620_exp_v1` | ai_v6 | 41 | 1 | REFERENCED | 0.135 | 0.045 | 0.903 | 0.202 | 0.001 | 0.046 | 1.338 | 0.045 |
| `ai_v5_10_tail1_23_0611` | ai_v5 | 11 | 4 | CLOSED | 0.117 | 0.029 | 0.205 | 0.106 | 0.0 | 0.852 | 1.32 | 1.29 |
| `ai_v9_22_E1_substrate_on_0821` | ai_v9 | 97 | 1 | REFERENCED | 0.107 | 0.036 | 0.64 | 0.432 | 0.002 | 0.072 | 1.293 | 0.321 |
| `ai_v9_26_baitent_probe_0823` | ai_v9 | 100 | 1 | REFERENCED | 0.0 | 0.036 | 0.711 | 0.377 | 0.0 | 0.107 | 1.234 | 0.0 |
| `ai_v6_09_dmg_reattend_N_0617` | ai_v6 | 35 | 4 | CLOSED | 0.217 | 0.043 | 0.78 | 0.122 | 0.002 | 0.044 | 1.215 | 1.169 |
| `ai_v9_162_TCUNFA_0903` | ai_v9 | 107 | 0 | LIVE | 0.291 | 0.036 | 0.582 | 0.165 | 0.001 | 0.11 | 1.213 | 0.0 |
| `ai_v8_17_rand20_nolut_0726` | ai_v8 | 46 | 3 | CLOSED | 0.581 | 0.045 | 0.0 | 0.53 | 0.001 | 0.045 | 1.206 | 0.917 |
| `ai_v8_20_rand10_nolut_0727` | ai_v8 | 46 | 3 | CLOSED | 0.581 | 0.045 | 0.0 | 0.519 | 0.001 | 0.045 | 1.195 | 0.909 |
| `ai_v9_07_gen6_seed_vicreg_0810` | ai_v9 | 62 | 0 | LIVE | 0.249 | 0.031 | 0.28 | 0.585 | 0.001 | 0.031 | 1.193 | 0.0 |
| `ai_v9_150_R4DOSE12_0901` | ai_v9 | 107 | 0 | LIVE | 0.291 | 0.036 | 0.582 | 0.181 | 0.001 | 0.074 | 1.192 | 0.0 |
| `ai_v9_81_REVIVE1c_0830` | ai_v9 | 107 | 0 | LIVE | 0.947 | 0.036 | 0.0 | 0.135 | 0.001 | 0.037 | 1.19 | 0.0 |
| `ai_v8_16_def20_lut_0726` | ai_v8 | 46 | 3 | CLOSED | 0.564 | 0.04 | 0.0 | 0.535 | 0.001 | 0.041 | 1.187 | 0.909 |
| `ai_v9_151_R4DOSE6_0901` | ai_v9 | 107 | 0 | LIVE | 0.291 | 0.036 | 0.582 | 0.173 | 0.001 | 0.074 | 1.184 | 0.0 |
| `ai_v9_161_TCFUNDB_0903` | ai_v9 | 107 | 0 | LIVE | 0.291 | 0.036 | 0.582 | 0.172 | 0.001 | 0.074 | 1.183 | 0.0 |
| `ai_v9_160_TCFUNDA_0903` | ai_v9 | 107 | 0 | LIVE | 0.291 | 0.036 | 0.582 | 0.166 | 0.001 | 0.074 | 1.178 | 0.0 |
| `ai_v9_172_G1SHORT_0905` | ai_v9 | 107 | 0 | LIVE | 0.291 | 0.036 | 0.582 | 0.167 | 0.001 | 0.074 | 1.178 | 0.0 |
| `ai_v9_170_TCUNFK6A_0904` | ai_v9 | 107 | 0 | LIVE | 0.291 | 0.036 | 0.582 | 0.166 | 0.001 | 0.074 | 1.178 | 0.0 |
| `ai_v9_163_TCUNFB_0903` | ai_v9 | 107 | 0 | LIVE | 0.291 | 0.036 | 0.582 | 0.166 | 0.001 | 0.074 | 1.176 | 0.0 |
| `ai_v9_80_REVIVE1b_0830` | ai_v9 | 107 | 0 | LIVE | 0.947 | 0.036 | 0.0 | 0.125 | 0.001 | 0.037 | 1.176 | 0.0 |
| `ai_v9_171_TCUNFK6B_0904` | ai_v9 | 107 | 0 | LIVE | 0.291 | 0.036 | 0.582 | 0.162 | 0.001 | 0.074 | 1.175 | 0.0 |
| `ai_v8_19_def20_lut_zeroinit_0727` | ai_v8 | 46 | 3 | CLOSED | 0.56 | 0.04 | 0.0 | 0.525 | 0.001 | 0.041 | 1.173 | 0.896 |
| `ai_v9_79_REVIVE1a_0830` | ai_v9 | 107 | 0 | LIVE | 0.947 | 0.036 | 0.0 | 0.12 | 0.001 | 0.037 | 1.168 | 0.0 |
| `ai_v8_18_rand20_lut_0726` | ai_v8 | 46 | 3 | CLOSED | 0.564 | 0.04 | 0.0 | 0.513 | 0.001 | 0.041 | 1.165 | 0.892 |
| `ai_v8_10_offense20_exploiter_0724` | ai_v8 | 45 | 3 | CLOSED | 0.581 | 0.045 | 0.0 | 0.489 | 0.001 | 0.046 | 1.164 | 0.883 |
| `ai_v9_77_G1LEAN_0830` | ai_v9 | 107 | 0 | LIVE | 0.911 | 0.036 | 0.0 | 0.139 | 0.001 | 0.037 | 1.159 | 0.0 |
| `ai_v5_2_native_selfplay_50m_0606` | ai_v5 | 2 | 4 | CLOSED | 0.113 | 0.028 | 0.085 | 0.058 | 0.0 | 0.85 | 1.141 | 1.11 |
| `ai_v8_13_defensive10_exploiter_0725` | ai_v8 | 45 | 0 | LIVE | 0.536 | 0.045 | 0.0 | 0.493 | 0.001 | 0.045 | 1.124 | 0.0 |
| `ai_v9_152_R4DOSE3_0901` | ai_v9 | 107 | 0 | LIVE | 0.291 | 0.036 | 0.509 | 0.182 | 0.001 | 0.074 | 1.122 | 0.0 |
| `ai_v6_03_win_pred_N_0614` | ai_v6 | 22 | 4 | CLOSED | 0.29 | 0.032 | 0.644 | 0.114 | 0.004 | 0.034 | 1.121 | 1.085 |
| `ai_v7_01_teacher_0626` | ai_v7 | 42 | 1 | REFERENCED | 0.217 | 0.043 | 0.434 | 0.212 | 0.002 | 0.175 | 1.092 | 0.564 |
| `ai_v6_11_unified_obs_fixed_0618` | ai_v6 | 37 | 4 | CLOSED | 0.217 | 0.043 | 0.608 | 0.156 | 0.002 | 0.044 | 1.075 | 1.03 |
| `ai_v9_58_R2CTRL_0827` | ai_v9 | 103 | 1 | REFERENCED | 0.728 | 0.036 | 0.073 | 0.114 | 0.001 | 0.037 | 1.014 | 0.656 |
| `ai_v9_50_fdF_p1c_0826` | ai_v9 | 103 | 1 | REFERENCED | 0.364 | 0.036 | 0.473 | 0.07 | 0.0 | 0.038 | 1.006 | 0.291 |
| `ai_v9_45_fdF_p1_0826` | ai_v9 | 102 | 1 | REFERENCED | 0.364 | 0.036 | 0.473 | 0.068 | 0.0 | 0.038 | 1.001 | 0.0 |
| `ai_v9_51_fdF_p2c_0826` | ai_v9 | 103 | 1 | REFERENCED | 0.328 | 0.036 | 0.473 | 0.079 | 0.0 | 0.037 | 0.976 | 0.728 |
| `ai_v9_102_R5F10_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.138 | 0.001 | 0.037 | 0.968 | 0.0 |
| `ai_v9_94_R5F02_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.135 | 0.001 | 0.037 | 0.967 | 0.0 |
| `ai_v9_95_R5F03_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.131 | 0.001 | 0.037 | 0.961 | 0.0 |
| `ai_v9_110_R5F18_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.129 | 0.001 | 0.037 | 0.958 | 0.0 |
| `ai_v9_111_R5F19_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.127 | 0.001 | 0.037 | 0.955 | 0.0 |
| `ai_v9_100_R5F08_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.124 | 0.001 | 0.037 | 0.953 | 0.0 |
| `ai_v9_98_R5F06_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.12 | 0.001 | 0.037 | 0.95 | 0.0 |
| `ai_v9_106_R5F14_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.121 | 0.001 | 0.037 | 0.949 | 0.0 |
| `ai_v9_92_R5F00_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.123 | 0.001 | 0.037 | 0.948 | 0.0 |
| `ai_v9_96_R5F04_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.121 | 0.001 | 0.037 | 0.945 | 0.0 |
| `ai_v9_104_R5F12_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.122 | 0.001 | 0.037 | 0.944 | 0.0 |
| `ai_v9_103_R5F11_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.119 | 0.001 | 0.037 | 0.943 | 0.0 |
| `ai_v9_107_R5F15_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.121 | 0.001 | 0.037 | 0.943 | 0.0 |
| `ai_v9_101_R5F09_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.112 | 0.001 | 0.037 | 0.941 | 0.0 |
| `ai_v9_97_R5F05_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.116 | 0.001 | 0.037 | 0.941 | 0.0 |
| `ai_v9_27_extremedial_probe_0823` | ai_v9 | 100 | 1 | REFERENCED | 0.0 | 0.036 | 0.711 | 0.154 | 0.0 | 0.036 | 0.939 | 0.036 |
| `ai_v9_36_tock1c_q6_0824` | ai_v9 | 101 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.116 | 0.001 | 0.037 | 0.938 | 0.656 |
| `ai_v9_57_R2F5e_0826` | ai_v9 | 103 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.115 | 0.001 | 0.037 | 0.937 | 0.0 |
| `ai_v9_93_R5F01_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.113 | 0.001 | 0.037 | 0.936 | 0.0 |
| `ai_v9_105_R5F13_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.111 | 0.001 | 0.037 | 0.933 | 0.0 |
| `ai_v9_108_R5F16_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.113 | 0.001 | 0.037 | 0.933 | 0.0 |
| `ai_v9_56_R2F5d_0826` | ai_v9 | 103 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.111 | 0.001 | 0.037 | 0.933 | 0.0 |
| `ai_v9_55_R2F5c_0826` | ai_v9 | 103 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.109 | 0.001 | 0.037 | 0.933 | 0.0 |
| `ai_v9_109_R5F17_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.111 | 0.001 | 0.037 | 0.931 | 0.0 |
| `ai_v9_31_tock1_k4_0824` | ai_v9 | 101 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.11 | 0.001 | 0.037 | 0.931 | 0.0 |
| `ai_v9_32_tock1b_rain_0824` | ai_v9 | 101 | 1 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.108 | 0.001 | 0.037 | 0.931 | 0.656 |
| `ai_v9_99_R5F07_0831` | ai_v9 | 107 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.11 | 0.001 | 0.037 | 0.93 | 0.0 |
| `ai_v9_54_R2F5b_0826` | ai_v9 | 103 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.109 | 0.001 | 0.037 | 0.93 | 0.0 |
| `ai_v9_53_R2F5a_0826` | ai_v9 | 103 | 0 | LIVE | 0.728 | 0.036 | 0.0 | 0.109 | 0.001 | 0.037 | 0.93 | 0.0 |
| `ai_v9_08_gen7_seed_quantile_0811` | ai_v9 | 63 | 0 | LIVE | 0.187 | 0.031 | 0.218 | 0.431 | 0.001 | 0.031 | 0.919 | 0.0 |
| `ai_v6_04_unified_inc_N_0615` | ai_v6 | 23 | 4 | CLOSED | 0.201 | 0.033 | 0.501 | 0.113 | 0.002 | 0.034 | 0.891 | 0.855 |
| `ai_v6_07_unified_topk_N_0616` | ai_v6 | 30 | 4 | CLOSED | 0.152 | 0.038 | 0.38 | 0.144 | 0.001 | 0.08 | 0.804 | 0.764 |
| `ai_v8_04_distill_4teacher_0722` | ai_v8 | 45 | 0 | LIVE | 0.357 | 0.045 | 0.045 | 0.284 | 0.001 | 0.046 | 0.787 | 0.0 |
| `ai_v9_20_tdaux_rung2_lam00_0820` | ai_v9 | 97 | 1 | REFERENCED | 0.036 | 0.036 | 0.498 | 0.151 | 0.001 | 0.036 | 0.76 | 0.533 |
| `ai_v9_20_tdaux_rung2_lam30_0820` | ai_v9 | 97 | 1 | REFERENCED | 0.036 | 0.036 | 0.498 | 0.15 | 0.001 | 0.036 | 0.76 | 0.533 |
| `ai_v9_20_tdaux_rung2_lam10_0820` | ai_v9 | 97 | 1 | REFERENCED | 0.036 | 0.036 | 0.498 | 0.148 | 0.001 | 0.036 | 0.759 | 0.533 |
| `ai_v7_05_tss_specialist_0703` | ai_v7 | 42 | 1 | REFERENCED | 0.478 | 0.043 | 0.0 | 0.149 | 0.008 | 0.045 | 0.724 | 0.391 |
| `ai_v8_06_semistall_3team_exploiter_0722` | ai_v8 | 45 | 0 | LIVE | 0.313 | 0.045 | 0.0 | 0.252 | 0.001 | 0.046 | 0.662 | 0.0 |
| `ai_v9_122_R5FUND02_0901` | ai_v9 | 107 | 0 | LIVE | 0.473 | 0.036 | 0.0 | 0.068 | 0.0 | 0.037 | 0.644 | 0.0 |
| `ai_v9_195_G5PLAINA_0906` | ai_v9 | 107 | 0 | LIVE | 0.073 | 0.0 | 0.509 | 0.0 | 0.0 | 0.037 | 0.644 | 0.0 |
| `ai_v9_197_G5PLAINC_0906` | ai_v9 | 107 | 0 | LIVE | 0.073 | 0.0 | 0.509 | 0.0 | 0.0 | 0.037 | 0.643 | 0.0 |
| `ai_v9_130_R5FUND10_0901` | ai_v9 | 107 | 0 | LIVE | 0.473 | 0.036 | 0.0 | 0.067 | 0.0 | 0.037 | 0.643 | 0.0 |
| `ai_v9_128_R5FUND08_0901` | ai_v9 | 107 | 0 | LIVE | 0.473 | 0.036 | 0.0 | 0.068 | 0.0 | 0.037 | 0.643 | 0.0 |
| `ai_v9_196_G5PLAINB_0906` | ai_v9 | 107 | 0 | LIVE | 0.073 | 0.0 | 0.509 | 0.0 | 0.0 | 0.037 | 0.642 | 0.0 |
| `ai_v9_30_rev1_exploit_0824` | ai_v9 | 101 | 1 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.061 | 0.0 | 0.037 | 0.638 | 0.364 |
| `ai_v9_126_R5FUND06_0901` | ai_v9 | 107 | 0 | LIVE | 0.473 | 0.036 | 0.0 | 0.062 | 0.0 | 0.037 | 0.635 | 0.0 |
| `ai_v9_35_tick1_exploit_0824` | ai_v9 | 101 | 1 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.06 | 0.0 | 0.037 | 0.634 | 0.364 |
| `ai_v9_120_R5FUND00_0901` | ai_v9 | 107 | 0 | LIVE | 0.473 | 0.036 | 0.0 | 0.062 | 0.0 | 0.037 | 0.632 | 0.0 |
| `ai_v9_132_R5FUND12_0901` | ai_v9 | 107 | 0 | LIVE | 0.473 | 0.036 | 0.0 | 0.063 | 0.0 | 0.037 | 0.63 | 0.0 |
| `ai_v9_124_R5FUND04_0901` | ai_v9 | 107 | 0 | LIVE | 0.473 | 0.036 | 0.0 | 0.061 | 0.0 | 0.037 | 0.63 | 0.0 |
| `ai_v9_134_R5FUND14_0901` | ai_v9 | 107 | 0 | LIVE | 0.473 | 0.036 | 0.0 | 0.058 | 0.0 | 0.037 | 0.629 | 0.0 |
| `ai_v6_08_unmasked_floor_N_0617` | ai_v6 | 30 | 4 | CLOSED | 0.114 | 0.038 | 0.304 | 0.122 | 0.001 | 0.039 | 0.624 | 0.585 |
| `.aborted_R4DOSE12_nometa_1401` | ai_v9 (via lineage) | 107 | 0 | LIVE | 0.0 | 0.0 | 0.509 | 0.0 | 0.0 | 0.037 | 0.56 | 0.0 |
| `ai_v7_14_league_capstone_0712` | ai_v7 | 43 | 0 | LIVE | 0.217 | 0.043 | 0.087 | 0.16 | 0.003 | 0.044 | 0.559 | 0.0 |
| `ai_v7_09_tss_bots_pubval_0708` | ai_v7 | 43 | 4 | CLOSED | 0.262 | 0.044 | 0.0 | 0.152 | 0.004 | 0.089 | 0.551 | 0.503 |
| `ai_v7_15_tss_exploiter_vs14_0713` | ai_v7 | 43 | 0 | LIVE | 0.35 | 0.044 | 0.0 | 0.087 | 0.006 | 0.045 | 0.537 | 0.0 |
| `.dryrun_K6A_1788581936` | ai_v9 (via lineage) | 107 | 0 | LIVE | 0.0 | 0.0 | 0.509 | 0.0 | 0.0 | 0.0 | 0.522 | 0.0 |
| `ai_v7_08_tss_bots_0707` | ai_v7 | 42 | 4 | CLOSED | 0.26 | 0.043 | 0.0 | 0.163 | 0.004 | 0.045 | 0.515 | 0.468 |
| `ai_v7_19_combined_0716` | ai_v7 | 43 | 4 | CLOSED | 0.13 | 0.043 | 0.087 | 0.188 | 0.001 | 0.044 | 0.51 | 0.465 |
| `ai_v6_06_unified_all_N_0616` | ai_v6 | 28 | 4 | CLOSED | 0.139 | 0.035 | 0.174 | 0.119 | 0.001 | 0.035 | 0.508 | 0.472 |
| `ai_v8_11_offense10_exploiter_0724` | ai_v8 | 45 | 3 | CLOSED | 0.223 | 0.045 | 0.0 | 0.183 | 0.001 | 0.045 | 0.497 | 0.223 |
| `ai_v6_13_outgoing_dmg_0620_exploiter_v2` | ai_v6 | 41 | 4 | CLOSED | 0.226 | 0.045 | 0.0 | 0.121 | 0.002 | 0.092 | 0.493 | 0.446 |
| `ai_v6_13_outgoing_dmg_0620_exploiter_v1` | ai_v6 | 41 | 4 | CLOSED | 0.226 | 0.045 | 0.0 | 0.12 | 0.002 | 0.046 | 0.442 | 0.395 |
| `ai_v7_06_tss_temp_anneal_0706` | ai_v7 | 42 | 4 | CLOSED | 0.174 | 0.043 | 0.0 | 0.152 | 0.002 | 0.044 | 0.416 | 0.37 |
| `ai_v8_02_zarch_teampfsp_0718` | ai_v8 | 44 | 3 | CLOSED | 0.089 | 0.045 | 0.089 | 0.129 | 0.0 | 0.045 | 0.402 | 0.134 |
| `ai_v7_17_stall_exploiter_0715` | ai_v7 | 43 | 0 | LIVE | 0.174 | 0.044 | 0.0 | 0.112 | 0.002 | 0.044 | 0.385 | 0.0 |
| `ai_v6_02_belief_lat_16m_0614` | ai_v6 | 21 | 4 | CLOSED | 0.094 | 0.031 | 0.125 | 0.083 | 0.001 | 0.032 | 0.371 | 0.339 |
| `ai_v8_08_defensive_6team_exploiter_0723` | ai_v8 | 45 | 3 | CLOSED | 0.134 | 0.045 | 0.0 | 0.132 | 0.0 | 0.045 | 0.365 | 0.089 |
| `ai_v9_17_tdaux_lam1_0818` | ai_v9 | 95 | 1 | REFERENCED | 0.042 | 0.042 | 0.084 | 0.149 | 0.001 | 0.043 | 0.365 | 0.126 |
| `ai_v9_17_tdaux_lam3_0818` | ai_v9 | 95 | 1 | REFERENCED | 0.042 | 0.042 | 0.084 | 0.145 | 0.001 | 0.043 | 0.36 | 0.042 |
| `ai_v7_07_tss_temp_ratchet_0707` | ai_v7 | 42 | 4 | CLOSED | 0.13 | 0.043 | 0.0 | 0.14 | 0.001 | 0.044 | 0.36 | 0.315 |
| `ai_v7_11_tss_exploiter_nopubval` | ai_v7 | 43 | 4 | CLOSED | 0.174 | 0.043 | 0.0 | 0.086 | 0.002 | 0.044 | 0.354 | 0.308 |
| `ai_v8_05_semistall564_exploiter_0722` | ai_v8 | 45 | 3 | CLOSED | 0.134 | 0.045 | 0.0 | 0.122 | 0.0 | 0.045 | 0.354 | 0.089 |
| `ai_v7_13_cmpass_exploiter_0711` | ai_v7 | 43 | 0 | LIVE | 0.174 | 0.044 | 0.0 | 0.08 | 0.002 | 0.044 | 0.345 | 0.0 |
| `ai_v7_21_fitnet_valuefeat_ab_0717` | ai_v7 | 43 | 4 | CLOSED | 0.087 | 0.043 | 0.043 | 0.113 | 0.001 | 0.044 | 0.337 | 0.292 |
| `ai_v7_18_distill_4teacher_0716` | ai_v7 | 43 | 1 | REFERENCED | 0.087 | 0.043 | 0.043 | 0.109 | 0.001 | 0.044 | 0.334 | 0.0 |
| `ai_v12_01_winprob_critic` | ai_v12 | 109 | 0 | LIVE | 0.049 | 0.024 | 0.073 | 0.175 | 0.001 | 0.001 | 0.329 | 0.0 |
| `ai_v7_12_trap_exploiter_0711` | ai_v7 | 43 | 0 | LIVE | 0.131 | 0.044 | 0.0 | 0.095 | 0.001 | 0.044 | 0.326 | 0.0 |
| `ai_v7_16_distill_tss_mvp_0715` | ai_v7 | 43 | 4 | CLOSED | 0.087 | 0.043 | 0.043 | 0.1 | 0.0 | 0.044 | 0.323 | 0.28 |
| `v8rep_p1_A_0905` | ai_v8 (replication) | 45 | 0 | LIVE | 0.089 | 0.045 | 0.045 | 0.086 | 0.0 | 0.045 | 0.323 | 0.0 |
| `ai_v9_17_tdaux_control_0818` | ai_v9 | 95 | 1 | REFERENCED | 0.042 | 0.042 | 0.042 | 0.147 | 0.0 | 0.043 | 0.321 | 0.084 |
| `ai_v6_04_unified_all_half_batch_N_0616` | ai_v6 | 28 | 4 | CLOSED | 0.104 | 0.035 | 0.035 | 0.101 | 0.001 | 0.036 | 0.316 | 0.28 |
| `ai_v6_10_unified_obs_0618` | ai_v6 | 37 | 4 | CLOSED | 0.087 | 0.043 | 0.0 | 0.134 | 0.001 | 0.044 | 0.316 | 0.272 |
| `ai_v7_10_tss_exploiter_fixed_0709` | ai_v7 | 43 | 0 | LIVE | 0.131 | 0.044 | 0.0 | 0.089 | 0.002 | 0.045 | 0.315 | 0.0 |
| `ai_v7_20_valuedistill_ab_0717` | ai_v7 | 43 | 4 | CLOSED | 0.087 | 0.043 | 0.043 | 0.082 | 0.0 | 0.044 | 0.308 | 0.265 |
| `DISCARDED_tdaux_control_n16_0818` | ai_v9 (via lineage) | 95 | 1 | REFERENCED | 0.082 | 0.041 | 0.041 | 0.074 | 0.001 | 0.042 | 0.29 | 0.041 |
| `v8rep_p1_C_0905` | ai_v8 (replication) | 45 | 0 | LIVE | 0.045 | 0.045 | 0.045 | 0.088 | 0.0 | 0.045 | 0.274 | 0.0 |
| `v8rep_p1_B_0905` | ai_v8 (replication) | 45 | 0 | LIVE | 0.045 | 0.045 | 0.045 | 0.088 | 0.0 | 0.045 | 0.273 | 0.0 |
| `v8rep_p2loss_B_0905` | ai_v8 (replication) | 45 | 0 | LIVE | 0.045 | 0.045 | 0.045 | 0.086 | 0.0 | 0.045 | 0.27 | 0.0 |
| `v8rep_p2loss_C_0905` | ai_v8 (replication) | 45 | 0 | LIVE | 0.045 | 0.045 | 0.045 | 0.087 | 0.0 | 0.045 | 0.269 | 0.0 |
| `v8rep_p2loss_A_0905` | ai_v8 (replication) | 45 | 0 | LIVE | 0.045 | 0.045 | 0.045 | 0.085 | 0.0 | 0.045 | 0.267 | 0.0 |
| `v8rep_p2self_C_0905` | ai_v8 (replication) | 45 | 0 | LIVE | 0.045 | 0.045 | 0.045 | 0.067 | 0.0 | 0.045 | 0.249 | 0.0 |
| `v8rep_p2self_A_0905` | ai_v8 (replication) | 45 | 0 | LIVE | 0.045 | 0.045 | 0.045 | 0.068 | 0.0 | 0.045 | 0.249 | 0.0 |
| `v8rep_p2self_B_0905` | ai_v8 (replication) | 45 | 0 | LIVE | 0.045 | 0.045 | 0.045 | 0.068 | 0.0 | 0.045 | 0.248 | 0.0 |
| `ai_v9_11_gen10_intentfull_compiled_0814` | ai_v9 | 77 | 1 | REFERENCED | 0.041 | 0.041 | 0.041 | 0.076 | 0.0 | 0.041 | 0.246 | 0.041 |
| `ai_v7_01_teacher_0626_oom1` | ai_v7 | 42 | 4 | CLOSED | 0.043 | 0.0 | 0.0 | 0.0 | 0.0 | 0.044 | 0.1 | 0.057 |
| `.aborted_R4DOSE12_poolless_1355` | ai_v9 (via lineage) | 107 | 0 | LIVE | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.037 | 0.053 | 0.0 |
| `ai_v7_05_tss_specialist_0703_aborted_noeval` | ai_v7 | 42 | 4 | CLOSED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.044 | 0.044 | 0.001 |
| `warmstart_generic_0715` | unknown | 43 | 4 | CLOSED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.043 | 0.043 | 0.0 |
| `ai_v7_20_valuedistill_SMOKE` | ai_v7 | 43 | 4 | CLOSED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.042 | 0.042 | 0.0 |
| `run_20260830_184043` | unknown | 107 | 0 | LIVE | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.027 | 0.027 | 0.0 |
| `run_20260830_180409` | unknown | 107 | 0 | LIVE | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.027 | 0.027 | 0.0 |
| `run_20260906_083317` | unknown | 107 | 0 | LIVE | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.01 | 0.01 | 0.0 |
| `run_20260830_183828` | unknown | 107 | 0 | LIVE | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.01 | 0.01 | 0.0 |
| `RETIRED_gen14_framedel_v90_0817` | unknown | 90 | 1 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.001 | 0.001 | 0.0 |
| `RETIRED_c5fork_control_gen13base_0817` | ai_v9 (via lineage) | 90 | 1 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Why each non-CLOSED run is protected

- **`.aborted_R4DOSE12_nometa_1401`** — LIVE: training output written within 7 days (tb/, 2026-09-02) — an arm launched between ledger updates has no other signal
- **`.aborted_R4DOSE12_poolless_1355`** — LIVE: training output written within 7 days (tb/, 2026-09-02) — an arm launched between ledger updates has no other signal
- **`.dryrun_K6A_1788581936`** — LIVE: training output written within 7 days (tb/, 2026-09-04) — an arm launched between ledger updates has no other signal
- **`DISCARDED_tdaux_control_n16_0818`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json
- **`RETIRED_c5fork_control_gen13base_0817`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json
- **`RETIRED_gen14_framedel_v90_0817`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json
- **`ai_v12_01_winprob_critic`** — LIVE: a LIVE launcher/trainer process names this run dir; training output written within 7 days (tb/, 2026-09-06) — an arm launched between ledger updates has no other signal
- **`ai_v5_13_shape_pbrs_43m_0612`** — REFERENCED: named by another run's model graph: ai_v6_01_belief_53m_0613 (argv mention), ai_v6_01_belief_53m_0613 (argv pool_source)
- **`ai_v5_5_popart_50m_0607`** — REFERENCED: loaded by 1 committed script(s): src/main/launcher_app_test.py; named by another run's model graph: ai_v5_6_stable_70m_0608 (argv mention), ai_v5_6_stable_70m_0608 (argv pool_source), ai_v5_7_switch_bias_41m_0609 (argv mention) …
- **`ai_v5_6_stable_70m_0608`** — REFERENCED: named by another run's model graph: ai_v5_7_switch_bias_41m_0609 (argv mention), ai_v5_7_switch_bias_41m_0609 (argv pool_source)
- **`ai_v5_8_split_inc_dmg_38m_0610`** — REFERENCED: named by another run's model graph: ai_v5_10_tail1_23_0611 (argv mention), ai_v5_10_tail1_23_0611 (argv pool_source), ai_v5_11_tail2_53m_0611 (argv mention) …
- **`ai_v5_9_attend_unrevealed_56m_0610`** — REFERENCED: named by another run's model graph: ai_v5_10_tail1_23_0611 (argv mention), ai_v5_10_tail1_23_0611 (argv pool_source), ai_v5_11_tail2_53m_0611 (argv mention) …
- **`ai_v6_13_outgoing_dmg_0620`** — REFERENCED: loaded by 3 committed script(s): src/main/launcher_test.py, src/main/run_name_test.py, src/main/train/parser/eval_subprocess.py; named by another run's model graph: ai_v6_13_outgoing_dmg_0620_exp_v1 (argv fork_parent), ai_v6_13_outgoing_dmg_0620_exp_v1 (fork_parent), ai_v6_13_outgoing_dmg_0620_exploiter_v1 (argv fork_parent) …
- **`ai_v6_13_outgoing_dmg_0620_exp_v1`** — REFERENCED: loaded by 1 committed script(s): src/main/launcher_test.py
- **`ai_v7_01_teacher_0626`** — REFERENCED: named by another run's model graph: ai_v7_01_teacher_0626_oom1 (argv mention), ai_v7_01_teacher_0626_oom1 (argv mention)
- **`ai_v7_02_critic_shape_0627`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v7_13_cmpass_exploiter_0711
- **`ai_v7_05_tss_specialist_0703`** — REFERENCED: named by another run's model graph: ai_v7_05_tss_specialist_0703_aborted_noeval (argv mention)
- **`ai_v7_10_tss_exploiter_fixed_0709`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v7_14_league_capstone_0712
- **`ai_v7_12_trap_exploiter_0711`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v8_04_distill_4teacher_0722
- **`ai_v7_13_cmpass_exploiter_0711`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v8_04_distill_4teacher_0722
- **`ai_v7_14_league_capstone_0712`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v7_17_stall_exploiter_0715
- **`ai_v7_15_tss_exploiter_vs14_0713`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v8_04_distill_4teacher_0722
- **`ai_v7_17_stall_exploiter_0715`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v8_04_distill_4teacher_0722
- **`ai_v7_18_distill_4teacher_0716`** — REFERENCED: named by another run's model graph: ai_v7_19_combined_0716 (argv fork_parent), ai_v7_19_combined_0716 (fork_parent)
- **`ai_v8_01_zarch_film_0717`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v8_03_zarch_control_0718
- **`ai_v8_03_zarch_control_0718`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v8_04_distill_4teacher_0722
- **`ai_v8_04_distill_4teacher_0722`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run v8rep_p2self_C_0905
- **`ai_v8_06_semistall_3team_exploiter_0722`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run v8rep_p2self_C_0905
- **`ai_v8_09_pool10_exploiter_0723`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run v8rep_p2self_C_0905
- **`ai_v8_13_defensive10_exploiter_0725`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run v8rep_p2self_C_0905
- **`ai_v8_14_distill3_0725`** — REFERENCED: loaded by 9 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/v8_era_locality_v2.py, designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/representational_richness_transfer_locus.py …; named by 19 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/README.md, designs/research_state/measurements/arch_transfer_2026-09-05/cross_era_head_to_head/README.md …; named by another run's model graph: ai_v8_15_retention_A_frozen_0726 (argv fork_parent), ai_v8_15_retention_A_frozen_0726 (fork_parent); a committed file / the ledger names a file the plan would delete
- **`ai_v8_15_retention_A_frozen_0726`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/v8_redistribution_pfsp_2026-08-30.md
- **`ai_v9_01_gen1_edges6_40m_0804`** — LIVE: run dir is a SYMLINK into a launcher worktree — the bytes are at /home/goodlad/dev/gen3ai/.claude/worktrees/gen1-run-0804/models/run_20260804_090512, outside models/ (--follow-symlinked-runs to include it)
- **`ai_v9_02_gen2_full11_40m_0805`** — LIVE: run dir is a SYMLINK into a launcher worktree — the bytes are at /home/goodlad/dev/gen3ai/.claude/worktrees/gen2-run-0805/models/run_20260805_060807, outside models/ (--follow-symlinked-runs to include it)
- **`ai_v9_03_gen25_consequence_25m_0806`** — LIVE: run dir is a SYMLINK into a launcher worktree — the bytes are at /home/goodlad/dev/gen3ai/.claude/worktrees/gen25-run-0806/models/run_20260806_160611, outside models/ (--follow-symlinked-runs to include it)
- **`ai_v9_04_gen3_k6_recency_40m_0807`** — LIVE: run dir is a SYMLINK into a launcher worktree — the bytes are at /home/goodlad/dev/gen3ai/.claude/worktrees/gen3-run-0807/models/run_20260807_135637, outside models/ (--follow-symlinked-runs to include it)
- **`ai_v9_05_gen4_rehome_25m_0808`** — LIVE: run dir is a SYMLINK into a launcher worktree — the bytes are at /home/goodlad/dev/gen3ai/.claude/worktrees/gen4-run-0808/models/run_20260808_212910, outside models/ (--follow-symlinked-runs to include it)
- **`ai_v9_06_gen5_no_concat_0809`** — LIVE: run dir is a SYMLINK into a launcher worktree — the bytes are at /home/goodlad/dev/gen3ai/.claude/worktrees/gen5-run-0809/models/ai_v9_06_gen5_no_concat_0809, outside models/ (--follow-symlinked-runs to include it)
- **`ai_v9_07_gen6_seed_vicreg_0810`** — LIVE: run dir is a SYMLINK into a launcher worktree — the bytes are at /home/goodlad/dev/gen3ai/.claude/worktrees/gen6-run-0810/models/ai_v9_07_gen6_seed_vicreg_0810, outside models/ (--follow-symlinked-runs to include it)
- **`ai_v9_08_gen7_seed_quantile_0811`** — LIVE: run dir is a SYMLINK into a launcher worktree — the bytes are at /home/goodlad/dev/gen3ai/.claude/worktrees/gen7-run-0811/models/ai_v9_08_gen7_seed_quantile_0811, outside models/ (--follow-symlinked-runs to include it)
- **`ai_v9_100_R5F08_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-01) — an arm launched between ledger updates has no other signal
- **`ai_v9_101_R5F09_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-01) — an arm launched between ledger updates has no other signal
- **`ai_v9_102_R5F10_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-01) — an arm launched between ledger updates has no other signal
- **`ai_v9_103_R5F11_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-01) — an arm launched between ledger updates has no other signal
- **`ai_v9_104_R5F12_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-01) — an arm launched between ledger updates has no other signal
- **`ai_v9_105_R5F13_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-01) — an arm launched between ledger updates has no other signal
- **`ai_v9_106_R5F14_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-01) — an arm launched between ledger updates has no other signal
- **`ai_v9_107_R5F15_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-01) — an arm launched between ledger updates has no other signal
- **`ai_v9_108_R5F16_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-01) — an arm launched between ledger updates has no other signal
- **`ai_v9_109_R5F17_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-02) — an arm launched between ledger updates has no other signal
- **`ai_v9_10_gen9_intent_distcritic_0813`** — REFERENCED: loaded by 1 committed script(s): src/agents/model/intent_move_cell_test.py; named by 1 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json
- **`ai_v9_110_R5F18_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-02) — an arm launched between ledger updates has no other signal
- **`ai_v9_111_R5F19_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-02) — an arm launched between ledger updates has no other signal
- **`ai_v9_11_gen10_intentfull_compiled_0814`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json
- **`ai_v9_120_R5FUND00_0901`** — LIVE: training output written within 7 days (tb/, 2026-09-02) — an arm launched between ledger updates has no other signal
- **`ai_v9_122_R5FUND02_0901`** — LIVE: training output written within 7 days (tb/, 2026-09-02) — an arm launched between ledger updates has no other signal
- **`ai_v9_124_R5FUND04_0901`** — LIVE: training output written within 7 days (tb/, 2026-09-02) — an arm launched between ledger updates has no other signal
- **`ai_v9_126_R5FUND06_0901`** — LIVE: training output written within 7 days (tb/, 2026-09-02) — an arm launched between ledger updates has no other signal
- **`ai_v9_128_R5FUND08_0901`** — LIVE: training output written within 7 days (tb/, 2026-09-02) — an arm launched between ledger updates has no other signal
- **`ai_v9_12_gen10_t0prior_0814`** — REFERENCED: named by 2 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/gen11_label_only_winprob_verdict.json
- **`ai_v9_130_R5FUND10_0901`** — LIVE: training output written within 7 days (tb/, 2026-09-02) — an arm launched between ledger updates has no other signal
- **`ai_v9_132_R5FUND12_0901`** — LIVE: training output written within 7 days (tb/, 2026-09-02) — an arm launched between ledger updates has no other signal
- **`ai_v9_134_R5FUND14_0901`** — LIVE: training output written within 7 days (tb/, 2026-09-02) — an arm launched between ledger updates has no other signal
- **`ai_v9_13_gen11_labelonly_winprob_0815`** — REFERENCED: named by 3 committed measurement artifact(s): designs/research_state/measurements/ai_v9_14_gen12_h_entitypool_shaping_0816_endofrun.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/gen11_label_only_winprob_verdict.json
- **`ai_v9_140_B2_0901`** — LIVE: training output written within 7 days (tb/, 2026-09-03) — an arm launched between ledger updates has no other signal
- **`ai_v9_141_C1_0901`** — LIVE: training output written within 7 days (tb/, 2026-09-03) — an arm launched between ledger updates has no other signal
- **`ai_v9_142_N1_0901`** — LIVE: training output written within 7 days (tb/, 2026-09-03) — an arm launched between ledger updates has no other signal
- **`ai_v9_143_N2_0901`** — LIVE: training output written within 7 days (tb/, 2026-09-03) — an arm launched between ledger updates has no other signal
- **`ai_v9_14_gen12_h_entitypool_shaping_0816`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/obs_conditioning_probe.py; named by 10 committed measurement artifact(s): designs/research_state/measurements/ai_v9_14_gen12_h_entitypool_shaping_0816_endofrun.json, designs/research_state/measurements/ai_v9_14_gen12_h_entitypool_shaping_0816_endofrun.md, designs/research_state/measurements/ai_v9_15_gen13_hb_events_stack_0817_endofrun.json …
- **`ai_v9_150_R4DOSE12_0901`** — LIVE: training output written within 7 days (tb/, 2026-09-02) — an arm launched between ledger updates has no other signal
- **`ai_v9_151_R4DOSE6_0901`** — LIVE: training output written within 7 days (tb/, 2026-09-02) — an arm launched between ledger updates has no other signal
- **`ai_v9_152_R4DOSE3_0901`** — LIVE: training output written within 7 days (tb/, 2026-09-03) — an arm launched between ledger updates has no other signal
- **`ai_v9_15_gen13_hb_events_stack_0817`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/gen13_stall_coverage.py, designs/research_state/measurements/gen14_paired_bt_refit.py, designs/research_state/measurements/obs_conditioning_probe.py; named by 12 committed measurement artifact(s): designs/research_state/measurements/ai_v9_15_gen13_hb_events_stack_0817_endofrun.json, designs/research_state/measurements/ai_v9_15_gen13_hb_events_stack_0817_endofrun.md, designs/research_state/measurements/ai_v9_16_gen14_framedel_v91_0817_endofrun.json …; named by another run's model graph: RETIRED_c5fork_control_gen13base_0817 (argv fork_parent), RETIRED_c5fork_control_gen13base_0817 (argv mention), RETIRED_c5fork_control_gen13base_0817 (argv mention) …
- **`ai_v9_160_TCFUNDA_0903`** — LIVE: training output written within 7 days (tb/, 2026-09-04) — an arm launched between ledger updates has no other signal
- **`ai_v9_161_TCFUNDB_0903`** — LIVE: training output written within 7 days (tb/, 2026-09-04) — an arm launched between ledger updates has no other signal
- **`ai_v9_162_TCUNFA_0903`** — LIVE: training output written within 7 days (tb/, 2026-09-05) — an arm launched between ledger updates has no other signal
- **`ai_v9_163_TCUNFB_0903`** — LIVE: training output written within 7 days (tb/, 2026-09-04) — an arm launched between ledger updates has no other signal
- **`ai_v9_16_gen14_framedel_v91_0817`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/gen13_stall_coverage.py, designs/research_state/measurements/gen14_paired_bt_refit.py, designs/research_state/measurements/obs_conditioning_probe.py; named by 10 committed measurement artifact(s): designs/research_state/measurements/README.md, designs/research_state/measurements/ai_v9_16_gen14_framedel_v91_0817_endofrun.json, designs/research_state/measurements/ai_v9_16_gen14_framedel_v91_0817_endofrun.md …; named by another run's model graph: DISCARDED_tdaux_control_n16_0818 (argv fork_parent), DISCARDED_tdaux_control_n16_0818 (argv mention), DISCARDED_tdaux_control_n16_0818 (fork_parent) …
- **`ai_v9_170_TCUNFK6A_0904`** — LIVE: training output written within 7 days (tb/, 2026-09-05) — an arm launched between ledger updates has no other signal
- **`ai_v9_171_TCUNFK6B_0904`** — LIVE: training output written within 7 days (tb/, 2026-09-05) — an arm launched between ledger updates has no other signal
- **`ai_v9_172_G1SHORT_0905`** — LIVE: training output written within 7 days (tb/, 2026-09-06) — an arm launched between ledger updates has no other signal
- **`ai_v9_17_tdaux_control_0818`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json; named by another run's model graph: DISCARDED_tdaux_control_n16_0818 (argv mention)
- **`ai_v9_17_tdaux_lam1_0818`** — REFERENCED: named by 1 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json
- **`ai_v9_17_tdaux_lam3_0818`** — REFERENCED: loaded by 1 committed script(s): src/agents/training/poke_env_gaps/faint_attribution_fuzz_test.py; named by 1 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json
- **`ai_v9_18_gen15_v8rewards_0818`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/obs_conditioning_probe.py, src/main/prober/loops.py; named by 7 committed measurement artifact(s): designs/research_state/measurements/ai_v9_18_gen15_v8rewards_0818_endofrun.json, designs/research_state/measurements/ai_v9_18_gen15_v8rewards_0818_endofrun.md, designs/research_state/measurements/ai_v9_19_gen16_mechanics_0819_endofrun.json …
- **`ai_v9_195_G5PLAINA_0906`** — LIVE: training output written within 7 days (tb/, 2026-09-06) — an arm launched between ledger updates has no other signal
- **`ai_v9_196_G5PLAINB_0906`** — LIVE: training output written within 7 days (tb/, 2026-09-06) — an arm launched between ledger updates has no other signal
- **`ai_v9_197_G5PLAINC_0906`** — LIVE: training output written within 7 days (tb/, 2026-09-06) — an arm launched between ledger updates has no other signal
- **`ai_v9_19_gen16_mechanics_0819`** — REFERENCED: named by 8 committed measurement artifact(s): designs/research_state/measurements/ai_v9_19_gen16_mechanics_0819_endofrun.json, designs/research_state/measurements/ai_v9_19_gen16_mechanics_0819_endofrun.md, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …; named by another run's model graph: ai_v9_20_tdaux_rung2_lam00_0820 (argv fork_parent), ai_v9_20_tdaux_rung2_lam00_0820 (argv mention), ai_v9_20_tdaux_rung2_lam00_0820 (fork_parent) …
- **`ai_v9_20_tdaux_rung2_lam00_0820`** — REFERENCED: named by 5 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_20_tdaux_rung2_lam10_0820`** — REFERENCED: named by 5 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_20_tdaux_rung2_lam30_0820`** — REFERENCED: named by 5 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_21_gen17_pfspoff_0820`** — REFERENCED: named in the ledger's last 1500 lines; loaded by 2 committed script(s): designs/research_state/measurements/obs_conditioning_probe.py, src/agents/model/audit_states_test.py; named by 15 committed measurement artifact(s): designs/research_state/measurements/README.md, designs/research_state/measurements/ai_v9_21_gen17_pfspoff_0820_endofrun.json, designs/research_state/measurements/ai_v9_29_rev1_0823_endofrun.json …; named by another run's model graph: ai_v9_22_E1_substrate_on_0821 (argv fork_parent), ai_v9_22_E1_substrate_on_0821 (fork_parent), ai_v9_23_E2_substrate_on_0822 (argv fork_parent) …
- **`ai_v9_22_E1_substrate_on_0821`** — REFERENCED: named by 4 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …; named by another run's model graph: ai_v9_25_E4_baitbot_0822 (argv fork_parent), ai_v9_25_E4_baitbot_0822 (fork_parent), ai_v9_26_baitent_probe_0823 (argv fork_parent) …
- **`ai_v9_23_E2_substrate_on_0822`** — REFERENCED: named by 4 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_24_E3_substrate_on_0822`** — REFERENCED: named by 4 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_25_E4_baitbot_0822`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/maturity_harm_trend.py; named by 9 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …; named by another run's model graph: ai_v9_27_extremedial_probe_0823 (argv fork_parent), ai_v9_27_extremedial_probe_0823 (argv mention), ai_v9_27_extremedial_probe_0823 (fork_parent)
- **`ai_v9_26_baitent_probe_0823`** — REFERENCED: REVIEW HOLD (partial) — the capacity baseline IS banked (designs/research_state/capacity_battery.md:153ff), but the P2 bait-entropy per-leg result (boost_eff 3.0, flagged 5.9%, B1 0.056 -> 0.229, leg-vs-leg z=-2.55, ledger.md:3722) is in no committed artifact, and the Baton Pass GIGO reproducer decodes loss_s0_003_states.npz from this run's traces (ledger.md:3595). ladder_readiness.md:269 also loads its legB_final_model.zip.
- **`ai_v9_27_extremedial_probe_0823`** — REFERENCED: loaded by 2 committed script(s): src/agents/training/exploiter_ladder.py, src/agents/training/exploiter_ladder_test.py; named by 4 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_29_rev1_0823`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v9_107_R5F15_0831
- **`ai_v9_30_rev1_exploit_0824`** — REFERENCED: named by 4 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_31_tock1_k4_0824`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/teacher_sharpness_probe.py; named by 10 committed measurement artifact(s): designs/research_state/measurements/ai_v9_34_tick1_0824_endofrun.json, designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.json, designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.md …; named by another run's model graph: ai_v9_34_tick1_0824 (argv mention), ai_v9_34_tick1_0824 (argv pool_source), ai_v9_34_tick1_0824 (argv teacher) …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_32_tock1b_rain_0824`** — REFERENCED: named by 8 committed measurement artifact(s): designs/research_state/measurements/ai_v9_34_tick1_0824_endofrun.json, designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.json, designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.md …; named by another run's model graph: ai_v9_34_tick1_0824 (argv mention), ai_v9_34_tick1_0824 (argv pool_source), ai_v9_34_tick1_0824 (argv teacher) …
- **`ai_v9_34_tick1_0824`** — REFERENCED: named in the ledger's last 1500 lines; named by 11 committed measurement artifact(s): designs/research_state/measurements/ai_v9_34_tick1_0824_endofrun.json, designs/research_state/measurements/ai_v9_34_tick1_0824_endofrun.md, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …; named by another run's model graph: ai_v9_35_tick1_exploit_0824 (argv fork_parent), ai_v9_35_tick1_exploit_0824 (argv mention), ai_v9_35_tick1_exploit_0824 (argv pool_source) …
- **`ai_v9_35_tick1_exploit_0824`** — REFERENCED: named by 4 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_36_tock1c_q6_0824`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/teacher_sharpness_probe.py; named by 8 committed measurement artifact(s): designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.json, designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.md, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …; named by another run's model graph: ai_v9_42_fdE_single_0825 (argv mention), ai_v9_42_fdE_single_0825 (argv teacher), ai_v9_42_fdE_single_0825 (teacher)
- **`ai_v9_37_tick1_dosext_0825`** — REFERENCED: named in the ledger's last 1500 lines; named by 8 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_38_fdA_coef03_0825`** — REFERENCED: named in the ledger's last 1500 lines; named by 5 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_39_fdB_lossonly_0825`** — REFERENCED: named by 5 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_40_fdC_ecology_0825`** — REFERENCED: named by 5 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_42_fdE_single_0825`** — REFERENCED: named by 5 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_44_tock2_v8shape_0825`** — REFERENCED: named by 7 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/pilot_T2_n300.json, designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.json, designs/research_state/measurements/differentiation_vs_breadth_2026-08-28.md …; a committed file / the ledger names a file the plan would delete
- **`ai_v9_45_fdF_p1_0826`** — REFERENCED: REVIEW HOLD — the NUMBERS are banked (designs/ai_v10/design_advantage_gated_distillation.md:459-467 carries the entropy 0.892 -> 1.354 dissolution and the subtraction rule), so this is not a data dependency; it is held because the ledger records an explicit owner decision to preserve it as the entropy-dissolution SPECIMEN (ledger.md:4937).
- **`ai_v9_48_G1_action_0826`** — REFERENCED: REVIEW HOLD — the program's first POSITIVE distill arm (pooled +0.0398 [+0.016,+0.064] z=+3.29; G2-fdB +0.0762 z=+6.01, ledger.md:4943ff). NO committed artifact carries the per-arm numbers: fold_capacity_telemetry.md has fdA/fdB/fdC/fdE rows and no G1/G2 row, and no ai_v9_48_*_endofrun.json exists — the claim rests on this run's eval_results.jsonl + eval_traces. Bank an endofrun artifact and this hold can be released.
- **`ai_v9_49_G2_advgate_0826`** — REFERENCED: named by 4 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_50_fdF_p1c_0826`** — REFERENCED: named by 4 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …; named by another run's model graph: ai_v9_51_fdF_p2c_0826 (argv fork_parent), ai_v9_51_fdF_p2c_0826 (argv mention), ai_v9_51_fdF_p2c_0826 (fork_parent)
- **`ai_v9_51_fdF_p2c_0826`** — REFERENCED: named by 4 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_52_G1p_matched_0826`** — REFERENCED: named by 4 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_53_R2F5a_0826`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v9_59_R2ACTION_0827
- **`ai_v9_54_R2F5b_0826`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v9_59_R2ACTION_0827
- **`ai_v9_55_R2F5c_0826`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v9_59_R2ACTION_0827
- **`ai_v9_56_R2F5d_0826`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v9_59_R2ACTION_0827
- **`ai_v9_57_R2F5e_0826`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v9_59_R2ACTION_0827
- **`ai_v9_58_R2CTRL_0827`** — REFERENCED: named in the ledger's last 1500 lines; loaded by 7 committed script(s): designs/ai_v12/team_slate_build.py, designs/research_state/measurements/plain_training_robbery.py, designs/research_state/measurements/representational_richness_transfer_forward.py …; named by 20 committed measurement artifact(s): designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.json, designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.md, designs/research_state/measurements/dark_knowledge_decomposition_2026-08-28.json …
- **`ai_v9_59_R2ACTION_0827`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v9_107_R5F15_0831
- **`ai_v9_60_R2TOPK_0827`** — REFERENCED: named by 10 committed measurement artifact(s): designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.json, designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.md, designs/research_state/measurements/dark_knowledge_decomposition_2026-08-28.json …
- **`ai_v9_61_R2KL_0827`** — REFERENCED: named by 10 committed measurement artifact(s): designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.json, designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.md, designs/research_state/measurements/dark_knowledge_decomposition_2026-08-28.json …
- **`ai_v9_62_R2PLAIN_0827`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/plain_training_robbery.py, designs/research_state/measurements/representational_richness_transfer_forward.py, designs/research_state/measurements/representational_richness_transfer_locus.py; named by 20 committed measurement artifact(s): designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.json, designs/research_state/measurements/bias_tax_head_alignment_2026-08-29.md, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …
- **`ai_v9_63_R3F6a_0828`** — REFERENCED: loaded by 5 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py …; named by 16 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_70_R3ACTION_0828 (argv mention), ai_v9_70_R3ACTION_0828 (argv teacher), ai_v9_70_R3ACTION_0828 (teacher) …
- **`ai_v9_64_R3F6b_0828`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py …; named by 13 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_70_R3ACTION_0828 (argv mention), ai_v9_70_R3ACTION_0828 (argv teacher), ai_v9_70_R3ACTION_0828 (teacher) …
- **`ai_v9_65_R3F6c_0828`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py …; named by 13 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_70_R3ACTION_0828 (argv mention), ai_v9_70_R3ACTION_0828 (argv teacher), ai_v9_70_R3ACTION_0828 (teacher) …
- **`ai_v9_66_R3F6d_0828`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py …; named by 13 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_70_R3ACTION_0828 (argv mention), ai_v9_70_R3ACTION_0828 (argv teacher), ai_v9_70_R3ACTION_0828 (teacher) …
- **`ai_v9_67_R3F6e_0828`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py …; named by 12 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_70_R3ACTION_0828 (argv mention), ai_v9_70_R3ACTION_0828 (argv teacher), ai_v9_70_R3ACTION_0828 (teacher) …
- **`ai_v9_68_R3F6f_0828`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py …; named by 27 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.json, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/dist_gen.log, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json …; named by another run's model graph: ai_v9_70_R3ACTION_0828 (argv mention), ai_v9_70_R3ACTION_0828 (argv teacher), ai_v9_70_R3ACTION_0828 (teacher) …
- **`ai_v9_69_R3F6CURR_0828`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/rev3_untaught_pulldown.py; named by 5 committed measurement artifact(s): designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json, designs/research_state/measurements/era_boundary_2026-09-06/loadability.json, designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json …
- **`ai_v9_70_R3ACTION_0828`** — REFERENCED: loaded by 12 committed script(s): designs/ai_v12/team_slate_build.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/resolve_sets.py …; named by 27 committed measurement artifact(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.json, designs/research_state/measurements/axis_split_inputs/cov_R3ACTION.json, designs/research_state/measurements/axis_split_inputs/pilot_R3ACTION_n300.json …
- **`ai_v9_71_R3ACTIONHI_0828`** — REFERENCED: loaded by 1 committed script(s): designs/ai_v12/team_slate_build.py; named by 7 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/cov_R3ACTIONHI.json, designs/research_state/measurements/axis_split_inputs/pilot_R3ACTIONHI_n300.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …
- **`ai_v9_72_R3SELF_0828`** — REFERENCED: loaded by 4 committed script(s): designs/ai_v12/team_slate_build.py, designs/research_state/measurements/plain_training_robbery.py, designs/research_state/measurements/starmie_ood_control_traces.py …; named by 16 committed measurement artifact(s): designs/research_state/measurements/axis_split_inputs/cov_R3SELF.json, designs/research_state/measurements/axis_split_inputs/pilot_R3SELF_n300.json, designs/research_state/measurements/era_boundary_2026-09-06/flag_archive_census.json …
- **`ai_v9_73_R4S3a_0829`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v9_142_N1_0901
- **`ai_v9_74_R4S3b_0829`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v9_142_N1_0901
- **`ai_v9_75_R4S3c_0829`** — LIVE: model-graph ancestor (transitively) of the LIVE/recent run ai_v9_142_N1_0901
- **`ai_v9_76_R4ACTION_0830`** — LIVE: training output written within 7 days (tb/, 2026-08-30) — an arm launched between ledger updates has no other signal
- **`ai_v9_77_G1LEAN_0830`** — LIVE: training output written within 7 days (best_model/, 2026-08-30) — an arm launched between ledger updates has no other signal
- **`ai_v9_79_REVIVE1a_0830`** — LIVE: training output written within 7 days (tb/, 2026-08-30) — an arm launched between ledger updates has no other signal
- **`ai_v9_80_REVIVE1b_0830`** — LIVE: training output written within 7 days (tb/, 2026-08-31) — an arm launched between ledger updates has no other signal
- **`ai_v9_81_REVIVE1c_0830`** — LIVE: training output written within 7 days (tb/, 2026-08-31) — an arm launched between ledger updates has no other signal
- **`ai_v9_82_REFOLD1_0830`** — LIVE: training output written within 7 days (tb/, 2026-08-31) — an arm launched between ledger updates has no other signal
- **`ai_v9_91_COMPFOLD_0831`** — LIVE: training output written within 7 days (tb/, 2026-08-31) — an arm launched between ledger updates has no other signal
- **`ai_v9_92_R5F00_0831`** — LIVE: training output written within 7 days (tb/, 2026-08-31) — an arm launched between ledger updates has no other signal
- **`ai_v9_93_R5F01_0831`** — LIVE: training output written within 7 days (tb/, 2026-08-31) — an arm launched between ledger updates has no other signal
- **`ai_v9_94_R5F02_0831`** — LIVE: training output written within 7 days (tb/, 2026-08-31) — an arm launched between ledger updates has no other signal
- **`ai_v9_95_R5F03_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-01) — an arm launched between ledger updates has no other signal
- **`ai_v9_96_R5F04_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-01) — an arm launched between ledger updates has no other signal
- **`ai_v9_97_R5F05_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-01) — an arm launched between ledger updates has no other signal
- **`ai_v9_98_R5F06_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-01) — an arm launched between ledger updates has no other signal
- **`ai_v9_99_R5F07_0831`** — LIVE: training output written within 7 days (tb/, 2026-09-01) — an arm launched between ledger updates has no other signal
- **`run_20260830_180409`** — LIVE: training output written within 7 days (tb/, 2026-08-30) — an arm launched between ledger updates has no other signal
- **`run_20260830_183828`** — LIVE: training output written within 7 days (tb/, 2026-08-30) — an arm launched between ledger updates has no other signal
- **`run_20260830_184043`** — LIVE: training output written within 7 days (tb/, 2026-08-30) — an arm launched between ledger updates has no other signal
- **`run_20260906_083317`** — LIVE: training output written within 7 days (tb/, 2026-09-06) — an arm launched between ledger updates has no other signal
- **`v8rep_p1_A_0905`** — LIVE: training output written within 7 days (checkpoints/, 2026-09-05) — an arm launched between ledger updates has no other signal
- **`v8rep_p1_B_0905`** — LIVE: training output written within 7 days (tb/, 2026-09-05) — an arm launched between ledger updates has no other signal
- **`v8rep_p1_C_0905`** — LIVE: training output written within 7 days (checkpoints/, 2026-09-05) — an arm launched between ledger updates has no other signal
- **`v8rep_p2loss_A_0905`** — LIVE: training output written within 7 days (tb/, 2026-09-05) — an arm launched between ledger updates has no other signal
- **`v8rep_p2loss_B_0905`** — LIVE: training output written within 7 days (checkpoints/, 2026-09-05) — an arm launched between ledger updates has no other signal
- **`v8rep_p2loss_C_0905`** — LIVE: training output written within 7 days (checkpoints/, 2026-09-05) — an arm launched between ledger updates has no other signal
- **`v8rep_p2self_A_0905`** — LIVE: training output written within 7 days (checkpoints/, 2026-09-05) — an arm launched between ledger updates has no other signal
- **`v8rep_p2self_B_0905`** — LIVE: training output written within 7 days (tb/, 2026-09-06) — an arm launched between ledger updates has no other signal
- **`v8rep_p2self_C_0905`** — LIVE: training output written within 7 days (checkpoints/, 2026-09-06) — an arm launched between ledger updates has no other signal

## What would be KEPT, and why (every CLOSED run with a plan)

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

<details><summary><code>ai_v5_10_tail1_23_0611</code> — 1.29 GB freed, 38 entries deleted</summary>

**KEEP**

- `checkpoint_10317874_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_11211275_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_12115544_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_12841299_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_13562801_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_14298595_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_15017067_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_15733931_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_16469981_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_17183712_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_17892968_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_18610457_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_1915569_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_19304769_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_19996745_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_20710810_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_21407519_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_22120056_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_23530799_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_2875071_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_3828190_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_4757916_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_5646893_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_6589322_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_7528787_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_8458162_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_9388178_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_957397_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_10317874_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_17892968_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_23530799_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_957397_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoint_10317874_steps.zip`
- `checkpoint_11211275_steps.zip`
- `checkpoint_12115544_steps.zip`
- `checkpoint_12841299_steps.zip`
- `checkpoint_13562801_steps.zip`
- `checkpoint_14298595_steps.zip`
- `checkpoint_15017067_steps.zip`
- `checkpoint_15733931_steps.zip`
- `checkpoint_16469981_steps.zip`
- `checkpoint_17183712_steps.zip`
- `checkpoint_17892968_steps.zip`
- `checkpoint_18610457_steps.zip`
- `checkpoint_1915569_steps.zip`
- `checkpoint_19304769_steps.zip`
- `checkpoint_19996745_steps.zip`
- `checkpoint_20710810_steps.zip`
- `checkpoint_21407519_steps.zip`
- `checkpoint_22120056_steps.zip`
- `checkpoint_23530799_steps.zip`
- `checkpoint_2875071_steps.zip`
- `checkpoint_3828190_steps.zip`
- `checkpoint_4757916_steps.zip`
- `checkpoint_5646893_steps.zip`
- `checkpoint_6589322_steps.zip`
- `checkpoint_7528787_steps.zip`
- `checkpoint_8458162_steps.zip`
- `checkpoint_9388178_steps.zip`
- `checkpoint_957397_steps.zip`
- `checkpoints/checkpoint_10317874_steps.zip`
- `checkpoints/checkpoint_17892968_steps.zip`
- `checkpoints/checkpoint_23530799_steps.zip`
- `checkpoints/checkpoint_957397_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v5_11_tail2_53m_0611</code> — 2.97 GB freed, 82 entries deleted</summary>

**KEEP**

- `checkpoint_10091682_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_10850742_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_11607840_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_12372698_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_13791933_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_14522855_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_15230331_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_15929512_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_16644318_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_17335616_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_18029970_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_18736058_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_1905874_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_19425627_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_20132749_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_20824505_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_21515147_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_22626861_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_23320359_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_24014406_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_24726151_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_25414312_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_26121713_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_26811299_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_27497675_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_28208238_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_2853805_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_28896609_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_29581664_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_30965662_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_31628713_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_32359408_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_32991848_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_33671827_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_34370071_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_35054057_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_35737638_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_36446198_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_37134208_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_37818344_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_3793180_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_38823565_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_39505014_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_40202626_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_40872940_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_41549695_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_42250034_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_42928523_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_43606394_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_44311261_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_44994144_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_45674230_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_46369706_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_47275909_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_4734755_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_47953061_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_48648497_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_49325807_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_50006031_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_50702741_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_51375531_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_52056146_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_5625505_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_6545249_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_7477583_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_8380984_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_9230631_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_955745_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_10091682_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_18029970_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_25414312_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_32991848_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_40202626_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_47275909_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_52056146_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_955745_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoint_10091682_steps.zip`
- `checkpoint_10850742_steps.zip`
- `checkpoint_11607840_steps.zip`
- `checkpoint_12372698_steps.zip`
- `checkpoint_13791933_steps.zip`
- `checkpoint_14522855_steps.zip`
- `checkpoint_15230331_steps.zip`
- `checkpoint_15929512_steps.zip`
- `checkpoint_16644318_steps.zip`
- `checkpoint_17335616_steps.zip`
- `checkpoint_18029970_steps.zip`
- `checkpoint_18736058_steps.zip`
- `checkpoint_1905874_steps.zip`
- `checkpoint_19425627_steps.zip`
- `checkpoint_20132749_steps.zip`
- `checkpoint_20824505_steps.zip`
- `checkpoint_21515147_steps.zip`
- `checkpoint_22626861_steps.zip`
- `checkpoint_23320359_steps.zip`
- `checkpoint_24014406_steps.zip`
- `checkpoint_24726151_steps.zip`
- `checkpoint_25414312_steps.zip`
- `checkpoint_26121713_steps.zip`
- `checkpoint_26811299_steps.zip`
- `checkpoint_27497675_steps.zip`
- `checkpoint_28208238_steps.zip`
- `checkpoint_2853805_steps.zip`
- `checkpoint_28896609_steps.zip`
- `checkpoint_29581664_steps.zip`
- `checkpoint_30965662_steps.zip`
- `checkpoint_31628713_steps.zip`
- `checkpoint_32359408_steps.zip`
- `checkpoint_32991848_steps.zip`
- `checkpoint_33671827_steps.zip`
- `checkpoint_34370071_steps.zip`
- `checkpoint_35054057_steps.zip`
- `checkpoint_35737638_steps.zip`
- `checkpoint_36446198_steps.zip`
- `checkpoint_37134208_steps.zip`
- `checkpoint_37818344_steps.zip`
- `checkpoint_3793180_steps.zip`
- `checkpoint_38823565_steps.zip`
- `checkpoint_39505014_steps.zip`
- `checkpoint_40202626_steps.zip`
- `checkpoint_40872940_steps.zip`
- `checkpoint_41549695_steps.zip`
- `checkpoint_42250034_steps.zip`
- `checkpoint_42928523_steps.zip`
- `checkpoint_43606394_steps.zip`
- `checkpoint_44311261_steps.zip`
- `checkpoint_44994144_steps.zip`
- `checkpoint_45674230_steps.zip`
- `checkpoint_46369706_steps.zip`
- `checkpoint_47275909_steps.zip`
- `checkpoint_4734755_steps.zip`
- `checkpoint_47953061_steps.zip`
- `checkpoint_48648497_steps.zip`
- `checkpoint_49325807_steps.zip`
- `checkpoint_50006031_steps.zip`
- `checkpoint_50702741_steps.zip`
- `checkpoint_51375531_steps.zip`
- `checkpoint_52056146_steps.zip`
- `checkpoint_5625505_steps.zip`
- `checkpoint_6545249_steps.zip`
- `checkpoint_7477583_steps.zip`
- `checkpoint_8380984_steps.zip`
- `checkpoint_9230631_steps.zip`
- `checkpoint_955745_steps.zip`
- `checkpoints/checkpoint_10091682_steps.zip`
- `checkpoints/checkpoint_18029970_steps.zip`
- `checkpoints/checkpoint_25414312_steps.zip`
- `checkpoints/checkpoint_32991848_steps.zip`
- `checkpoints/checkpoint_40202626_steps.zip`
- `checkpoints/checkpoint_47275909_steps.zip`
- `checkpoints/checkpoint_52056146_steps.zip`
- `checkpoints/checkpoint_955745_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v5_12_bias_05_N_0612</code> — 1.32 GB freed, 38 entries deleted</summary>

**KEEP**

- `checkpoint_10649102_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_11379178_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_12112473_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_12912642_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_13617713_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_14346810_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_15051309_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_15766256_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_16491086_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_17204319_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_17902105_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_18624267_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_1899235_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_19450973_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_20302095_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_21134241_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_21893534_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_22609369_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_23309510_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_23309510_steps.zip` — THE final model — what resolve_model_ref picks
- `checkpoint_2849670_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_3788183_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_4724500_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_5631091_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_6545991_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_7461562_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_8348321_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_9109129_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_951950_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_9890482_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_17204319_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_23309510_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_951950_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_9890482_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — latest.txt's target — so the pin still resolves
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoint_10649102_steps.zip`
- `checkpoint_11379178_steps.zip`
- `checkpoint_12112473_steps.zip`
- `checkpoint_12912642_steps.zip`
- `checkpoint_13617713_steps.zip`
- `checkpoint_14346810_steps.zip`
- `checkpoint_15051309_steps.zip`
- `checkpoint_15766256_steps.zip`
- `checkpoint_16491086_steps.zip`
- `checkpoint_17204319_steps.zip`
- `checkpoint_17902105_steps.zip`
- `checkpoint_18624267_steps.zip`
- `checkpoint_1899235_steps.zip`
- `checkpoint_19450973_steps.zip`
- `checkpoint_20302095_steps.zip`
- `checkpoint_21134241_steps.zip`
- `checkpoint_21893534_steps.zip`
- `checkpoint_22609369_steps.zip`
- `checkpoint_2849670_steps.zip`
- `checkpoint_3788183_steps.zip`
- `checkpoint_4724500_steps.zip`
- `checkpoint_5631091_steps.zip`
- `checkpoint_6545991_steps.zip`
- `checkpoint_7461562_steps.zip`
- `checkpoint_8348321_steps.zip`
- `checkpoint_9109129_steps.zip`
- `checkpoint_951950_steps.zip`
- `checkpoint_9890482_steps.zip`
- `checkpoints/checkpoint_17204319_steps.zip`
- `checkpoints/checkpoint_23309510_steps.zip`
- `checkpoints/checkpoint_951950_steps.zip`
- `checkpoints/checkpoint_9890482_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

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

<details><summary><code>ai_v5_2_native_selfplay_50m_0606</code> — 1.11 GB freed, 39 entries deleted</summary>

**KEEP**

- `checkpoint_1022179_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_10614570_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_11576311_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_12523319_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_13510266_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_14479853_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_15446961_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_16416001_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_17384394_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_18359246_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_19327250_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_20302635_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_2043596_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_21267940_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_22752707_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_23715478_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_24682557_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_25638088_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_26588484_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_27512561_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_29031445_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_29865854_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_3047186_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_4064618_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_5074574_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_6068458_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_7635256_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_8632303_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_9621381_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_1022179_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_11576311_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_21267940_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_29865854_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- `tb_imgs` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoint_1022179_steps.zip`
- `checkpoint_10614570_steps.zip`
- `checkpoint_11576311_steps.zip`
- `checkpoint_12523319_steps.zip`
- `checkpoint_13510266_steps.zip`
- `checkpoint_14479853_steps.zip`
- `checkpoint_15446961_steps.zip`
- `checkpoint_16416001_steps.zip`
- `checkpoint_17384394_steps.zip`
- `checkpoint_18359246_steps.zip`
- `checkpoint_19327250_steps.zip`
- `checkpoint_20302635_steps.zip`
- `checkpoint_2043596_steps.zip`
- `checkpoint_21267940_steps.zip`
- `checkpoint_22752707_steps.zip`
- `checkpoint_23715478_steps.zip`
- `checkpoint_24682557_steps.zip`
- `checkpoint_25638088_steps.zip`
- `checkpoint_26588484_steps.zip`
- `checkpoint_27512561_steps.zip`
- `checkpoint_29031445_steps.zip`
- `checkpoint_29865854_steps.zip`
- `checkpoint_3047186_steps.zip`
- `checkpoint_4064618_steps.zip`
- `checkpoint_5074574_steps.zip`
- `checkpoint_6068458_steps.zip`
- `checkpoint_7635256_steps.zip`
- `checkpoint_8632303_steps.zip`
- `checkpoint_9621381_steps.zip`
- `checkpoints/checkpoint_1022179_steps.zip`
- `checkpoints/checkpoint_11576311_steps.zip`
- `checkpoints/checkpoint_21267940_steps.zip`
- `checkpoints/checkpoint_29865854_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v5_3_vf_coef_clip_50m_0606</code> — 2.505 GB freed, 74 entries deleted</summary>

**KEEP**

- `checkpoint_10895401_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_11869713_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_12848692_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_13816852_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_14784757_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_16622450_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_17463134_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_18308312_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_19128561_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_1993495_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_19938636_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_20760103_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_21564615_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_22378557_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_23129647_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_23884527_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_24680257_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_25485556_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_26302197_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_27003153_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_28039400_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_28792110_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_29537514_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_2997280_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_30304137_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_31041375_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_31774169_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_32527692_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_33248810_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_33971053_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_34700055_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_35405574_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_36130928_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_36845070_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_37571126_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_38298735_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_38998971_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_39702149_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_3990184_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_40418253_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_41115808_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_41819391_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_42542917_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_43241595_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_43933342_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_44630016_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_45317725_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_46005516_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_46691669_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_47355994_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_48022057_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_48703454_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_49378803_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_4986194_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_50056021_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_5975927_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_6970599_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_7948064_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_8936272_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_9915841_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_998727_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_10895401_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_20760103_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_28792110_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_36130928_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_43241595_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_50056021_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_998727_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoint_10895401_steps.zip`
- `checkpoint_11869713_steps.zip`
- `checkpoint_12848692_steps.zip`
- `checkpoint_13816852_steps.zip`
- `checkpoint_14784757_steps.zip`
- `checkpoint_16622450_steps.zip`
- `checkpoint_17463134_steps.zip`
- `checkpoint_18308312_steps.zip`
- `checkpoint_19128561_steps.zip`
- `checkpoint_1993495_steps.zip`
- `checkpoint_19938636_steps.zip`
- `checkpoint_20760103_steps.zip`
- `checkpoint_21564615_steps.zip`
- `checkpoint_22378557_steps.zip`
- `checkpoint_23129647_steps.zip`
- `checkpoint_23884527_steps.zip`
- `checkpoint_24680257_steps.zip`
- `checkpoint_25485556_steps.zip`
- `checkpoint_26302197_steps.zip`
- `checkpoint_27003153_steps.zip`
- `checkpoint_28039400_steps.zip`
- `checkpoint_28792110_steps.zip`
- `checkpoint_29537514_steps.zip`
- `checkpoint_2997280_steps.zip`
- `checkpoint_30304137_steps.zip`
- `checkpoint_31041375_steps.zip`
- `checkpoint_31774169_steps.zip`
- `checkpoint_32527692_steps.zip`
- `checkpoint_33248810_steps.zip`
- `checkpoint_33971053_steps.zip`
- `checkpoint_34700055_steps.zip`
- `checkpoint_35405574_steps.zip`
- `checkpoint_36130928_steps.zip`
- `checkpoint_36845070_steps.zip`
- `checkpoint_37571126_steps.zip`
- `checkpoint_38298735_steps.zip`
- `checkpoint_38998971_steps.zip`
- `checkpoint_39702149_steps.zip`
- `checkpoint_3990184_steps.zip`
- `checkpoint_40418253_steps.zip`
- `checkpoint_41115808_steps.zip`
- `checkpoint_41819391_steps.zip`
- `checkpoint_42542917_steps.zip`
- `checkpoint_43241595_steps.zip`
- `checkpoint_43933342_steps.zip`
- `checkpoint_44630016_steps.zip`
- `checkpoint_45317725_steps.zip`
- `checkpoint_46005516_steps.zip`
- `checkpoint_46691669_steps.zip`
- `checkpoint_47355994_steps.zip`
- `checkpoint_48022057_steps.zip`
- `checkpoint_48703454_steps.zip`
- `checkpoint_49378803_steps.zip`
- `checkpoint_4986194_steps.zip`
- `checkpoint_50056021_steps.zip`
- `checkpoint_5975927_steps.zip`
- `checkpoint_6970599_steps.zip`
- `checkpoint_7948064_steps.zip`
- `checkpoint_8936272_steps.zip`
- `checkpoint_9915841_steps.zip`
- `checkpoint_998727_steps.zip`
- `checkpoints/checkpoint_10895401_steps.zip`
- `checkpoints/checkpoint_20760103_steps.zip`
- `checkpoints/checkpoint_28792110_steps.zip`
- `checkpoints/checkpoint_36130928_steps.zip`
- `checkpoints/checkpoint_43241595_steps.zip`
- `checkpoints/checkpoint_50056021_steps.zip`
- `checkpoints/checkpoint_998727_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v5_4_pbrs_opp_threat_50m_0607</code> — 1.8 GB freed, 57 entries deleted</summary>

**KEEP**

- `checkpoint_10223552_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_11138165_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_12065092_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_13761029_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_14637896_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_15466694_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_16292129_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_16984980_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_17685453_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_1817673_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_18414438_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_19172258_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_19928087_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_20689507_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_21415834_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_22140769_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_22821103_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_23496260_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_24611167_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_25280428_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_25961029_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_26654947_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_27326557_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_2770519_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_28001146_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_28696084_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_29395559_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_30088947_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_30750353_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_31401766_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_32061721_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_32719512_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_33388083_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_34030253_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_34694930_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_35315608_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_3711711_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_4656275_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_5596258_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_6545657_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_7481487_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_8417356_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_909273_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_9336186_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_10223552_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_19172258_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_26654947_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_33388083_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_35315608_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_909273_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoint_10223552_steps.zip`
- `checkpoint_11138165_steps.zip`
- `checkpoint_12065092_steps.zip`
- `checkpoint_13761029_steps.zip`
- `checkpoint_14637896_steps.zip`
- `checkpoint_15466694_steps.zip`
- `checkpoint_16292129_steps.zip`
- `checkpoint_16984980_steps.zip`
- `checkpoint_17685453_steps.zip`
- `checkpoint_1817673_steps.zip`
- `checkpoint_18414438_steps.zip`
- `checkpoint_19172258_steps.zip`
- `checkpoint_19928087_steps.zip`
- `checkpoint_20689507_steps.zip`
- `checkpoint_21415834_steps.zip`
- `checkpoint_22140769_steps.zip`
- `checkpoint_22821103_steps.zip`
- `checkpoint_23496260_steps.zip`
- `checkpoint_24611167_steps.zip`
- `checkpoint_25280428_steps.zip`
- `checkpoint_25961029_steps.zip`
- `checkpoint_26654947_steps.zip`
- `checkpoint_27326557_steps.zip`
- `checkpoint_2770519_steps.zip`
- `checkpoint_28001146_steps.zip`
- `checkpoint_28696084_steps.zip`
- `checkpoint_29395559_steps.zip`
- `checkpoint_30088947_steps.zip`
- `checkpoint_30750353_steps.zip`
- `checkpoint_31401766_steps.zip`
- `checkpoint_32061721_steps.zip`
- `checkpoint_32719512_steps.zip`
- `checkpoint_33388083_steps.zip`
- `checkpoint_34030253_steps.zip`
- `checkpoint_34694930_steps.zip`
- `checkpoint_35315608_steps.zip`
- `checkpoint_3711711_steps.zip`
- `checkpoint_4656275_steps.zip`
- `checkpoint_5596258_steps.zip`
- `checkpoint_6545657_steps.zip`
- `checkpoint_7481487_steps.zip`
- `checkpoint_8417356_steps.zip`
- `checkpoint_909273_steps.zip`
- `checkpoint_9336186_steps.zip`
- `checkpoints/checkpoint_10223552_steps.zip`
- `checkpoints/checkpoint_19172258_steps.zip`
- `checkpoints/checkpoint_26654947_steps.zip`
- `checkpoints/checkpoint_33388083_steps.zip`
- `checkpoints/checkpoint_35315608_steps.zip`
- `checkpoints/checkpoint_909273_steps.zip`
- `crashes`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

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

<details><summary><code>ai_v5_7_switch_bias_41m_0609</code> — 2.261 GB freed, 65 entries deleted</summary>

**KEEP**

- `checkpoint_10071937_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_10810631_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_11523551_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_12818082_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_13553255_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_14291693_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_15017509_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_15729231_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_16458146_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_17147059_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_17846156_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_18588178_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_19320375_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_1997973_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_20053904_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_20779995_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_21651197_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_22380468_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_23099433_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_23812133_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_24544535_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_25257234_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_25964463_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_26697794_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_27410568_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_28139673_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_28853228_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_29562627_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_2962108_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_30321082_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_31040085_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_31752151_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_32484321_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_33198926_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_33915339_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_34647901_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_35366773_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_36090614_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_36812277_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_37526717_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_38855174_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_3901797_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_39566178_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_40296240_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_40992544_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_4844295_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_5789341_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_6736917_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_7675255_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_8529945_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_9287169_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_989468_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_10071937_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_17846156_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_25257234_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_32484321_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_40296240_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_40992544_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_989468_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoint_10071937_steps.zip`
- `checkpoint_10810631_steps.zip`
- `checkpoint_11523551_steps.zip`
- `checkpoint_12818082_steps.zip`
- `checkpoint_13553255_steps.zip`
- `checkpoint_14291693_steps.zip`
- `checkpoint_15017509_steps.zip`
- `checkpoint_15729231_steps.zip`
- `checkpoint_16458146_steps.zip`
- `checkpoint_17147059_steps.zip`
- `checkpoint_17846156_steps.zip`
- `checkpoint_18588178_steps.zip`
- `checkpoint_19320375_steps.zip`
- `checkpoint_1997973_steps.zip`
- `checkpoint_20053904_steps.zip`
- `checkpoint_20779995_steps.zip`
- `checkpoint_21651197_steps.zip`
- `checkpoint_22380468_steps.zip`
- `checkpoint_23099433_steps.zip`
- `checkpoint_23812133_steps.zip`
- `checkpoint_24544535_steps.zip`
- `checkpoint_25257234_steps.zip`
- `checkpoint_25964463_steps.zip`
- `checkpoint_26697794_steps.zip`
- `checkpoint_27410568_steps.zip`
- `checkpoint_28139673_steps.zip`
- `checkpoint_28853228_steps.zip`
- `checkpoint_29562627_steps.zip`
- `checkpoint_2962108_steps.zip`
- `checkpoint_30321082_steps.zip`
- `checkpoint_31040085_steps.zip`
- `checkpoint_31752151_steps.zip`
- `checkpoint_32484321_steps.zip`
- `checkpoint_33198926_steps.zip`
- `checkpoint_33915339_steps.zip`
- `checkpoint_34647901_steps.zip`
- `checkpoint_35366773_steps.zip`
- `checkpoint_36090614_steps.zip`
- `checkpoint_36812277_steps.zip`
- `checkpoint_37526717_steps.zip`
- `checkpoint_38855174_steps.zip`
- `checkpoint_3901797_steps.zip`
- `checkpoint_39566178_steps.zip`
- `checkpoint_40296240_steps.zip`
- `checkpoint_40992544_steps.zip`
- `checkpoint_4844295_steps.zip`
- `checkpoint_5789341_steps.zip`
- `checkpoint_6736917_steps.zip`
- `checkpoint_7675255_steps.zip`
- `checkpoint_8529945_steps.zip`
- `checkpoint_9287169_steps.zip`
- `checkpoint_989468_steps.zip`
- `checkpoints/checkpoint_10071937_steps.zip`
- `checkpoints/checkpoint_17846156_steps.zip`
- `checkpoints/checkpoint_25257234_steps.zip`
- `checkpoints/checkpoint_32484321_steps.zip`
- `checkpoints/checkpoint_40296240_steps.zip`
- `checkpoints/checkpoint_40992544_steps.zip`
- `checkpoints/checkpoint_989468_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

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

<details><summary><code>ai_v6_01_belief_53m_0613</code> — 3.006 GB freed, 81 entries deleted</summary>

**KEEP**

- `checkpoint_10797773_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_11684086_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_12513274_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_13256992_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_14002863_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_14754109_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_15502632_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_16252211_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_16975696_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_17694914_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_18883182_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_1953511_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_19606239_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_20343859_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_21076871_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_21805036_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_22544249_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_23245030_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_23907736_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_24679050_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_25390057_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_26108605_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_26927724_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_27640638_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_28372354_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_29075664_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_2924609_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_29759218_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_30460472_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_31144171_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_31829213_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_32536509_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_33238595_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_33939667_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_34891706_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_35603003_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_36331522_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_37037509_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_37751311_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_3843214_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_38477773_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_39183115_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_39890013_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_40618773_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_41327147_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_42042620_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_43139686_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_43841813_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_44558215_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_45259540_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_45960331_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_46673202_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_47374060_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_4807200_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_48091911_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_48802763_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_49503903_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_50218159_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_51494096_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_52219390_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_52910114_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_5765317_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_6729118_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_7673369_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_8967447_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_976687_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoint_9897399_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_10797773_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_18883182_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_26108605_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_33238595_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_40618773_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_48091911_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_52910114_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_976687_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `review_notes.json` — small run-root bookkeeping (not a .zip or a .log)
- `review_notes.md` — small run-root bookkeeping (not a .zip or a .log)
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoint_10797773_steps.zip`
- `checkpoint_11684086_steps.zip`
- `checkpoint_12513274_steps.zip`
- `checkpoint_13256992_steps.zip`
- `checkpoint_14002863_steps.zip`
- `checkpoint_14754109_steps.zip`
- `checkpoint_15502632_steps.zip`
- `checkpoint_16252211_steps.zip`
- `checkpoint_16975696_steps.zip`
- `checkpoint_17694914_steps.zip`
- `checkpoint_18883182_steps.zip`
- `checkpoint_1953511_steps.zip`
- `checkpoint_19606239_steps.zip`
- `checkpoint_20343859_steps.zip`
- `checkpoint_21076871_steps.zip`
- `checkpoint_21805036_steps.zip`
- `checkpoint_22544249_steps.zip`
- `checkpoint_23245030_steps.zip`
- `checkpoint_23907736_steps.zip`
- `checkpoint_24679050_steps.zip`
- `checkpoint_25390057_steps.zip`
- `checkpoint_26108605_steps.zip`
- `checkpoint_26927724_steps.zip`
- `checkpoint_27640638_steps.zip`
- `checkpoint_28372354_steps.zip`
- `checkpoint_29075664_steps.zip`
- `checkpoint_2924609_steps.zip`
- `checkpoint_29759218_steps.zip`
- `checkpoint_30460472_steps.zip`
- `checkpoint_31144171_steps.zip`
- `checkpoint_31829213_steps.zip`
- `checkpoint_32536509_steps.zip`
- `checkpoint_33238595_steps.zip`
- `checkpoint_33939667_steps.zip`
- `checkpoint_34891706_steps.zip`
- `checkpoint_35603003_steps.zip`
- `checkpoint_36331522_steps.zip`
- `checkpoint_37037509_steps.zip`
- `checkpoint_37751311_steps.zip`
- `checkpoint_3843214_steps.zip`
- `checkpoint_38477773_steps.zip`
- `checkpoint_39183115_steps.zip`
- `checkpoint_39890013_steps.zip`
- `checkpoint_40618773_steps.zip`
- `checkpoint_41327147_steps.zip`
- `checkpoint_42042620_steps.zip`
- `checkpoint_43139686_steps.zip`
- `checkpoint_43841813_steps.zip`
- `checkpoint_44558215_steps.zip`
- `checkpoint_45259540_steps.zip`
- `checkpoint_45960331_steps.zip`
- `checkpoint_46673202_steps.zip`
- `checkpoint_47374060_steps.zip`
- `checkpoint_4807200_steps.zip`
- `checkpoint_48091911_steps.zip`
- `checkpoint_48802763_steps.zip`
- `checkpoint_49503903_steps.zip`
- `checkpoint_50218159_steps.zip`
- `checkpoint_51494096_steps.zip`
- `checkpoint_52219390_steps.zip`
- `checkpoint_52910114_steps.zip`
- `checkpoint_5765317_steps.zip`
- `checkpoint_6729118_steps.zip`
- `checkpoint_7673369_steps.zip`
- `checkpoint_8967447_steps.zip`
- `checkpoint_976687_steps.zip`
- `checkpoint_9897399_steps.zip`
- `checkpoints/checkpoint_10797773_steps.zip`
- `checkpoints/checkpoint_18883182_steps.zip`
- `checkpoints/checkpoint_26108605_steps.zip`
- `checkpoints/checkpoint_33238595_steps.zip`
- `checkpoints/checkpoint_40618773_steps.zip`
- `checkpoints/checkpoint_48091911_steps.zip`
- `checkpoints/checkpoint_52910114_steps.zip`
- `checkpoints/checkpoint_976687_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v6_02_belief_lat_16m_0614</code> — 0.339 GB freed, 9 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_10260721_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_16510131_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_983006_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `review_notes.json` — small run-root bookkeeping (not a .zip or a .log)
- `review_notes.md` — small run-root bookkeeping (not a .zip or a .log)
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_10260721_steps.zip`
- `checkpoints/checkpoint_16510131_steps.zip`
- `checkpoints/checkpoint_983006_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v6_03_win_pred_N_0614</code> — 1.085 GB freed, 16 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_10388758_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_17984199_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_25226792_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_32373631_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_39602721_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_46725020_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_53869147_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_58917105_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_990898_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `review_notes.json` — small run-root bookkeeping (not a .zip or a .log)
- `review_notes.md` — small run-root bookkeeping (not a .zip or a .log)
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_10388758_steps.zip`
- `checkpoints/checkpoint_17984199_steps.zip`
- `checkpoints/checkpoint_25226792_steps.zip`
- `checkpoints/checkpoint_32373631_steps.zip`
- `checkpoints/checkpoint_39602721_steps.zip`
- `checkpoints/checkpoint_46725020_steps.zip`
- `checkpoints/checkpoint_53869147_steps.zip`
- `checkpoints/checkpoint_58917105_steps.zip`
- `checkpoints/checkpoint_990898_steps.zip`
- `crashes`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v6_04_unified_all_half_batch_N_0616</code> — 0.28 GB freed, 10 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1177011_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_12793484_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_13838366_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_1177011_steps.zip`
- `checkpoints/checkpoint_12793484_steps.zip`
- `checkpoints/checkpoint_13838366_steps.zip`
- `crashes`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v6_04_unified_inc_N_0615</code> — 0.855 GB freed, 12 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1091845_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_11858658_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_19438316_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_28023692_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_35946535_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_38937343_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_1091845_steps.zip`
- `checkpoints/checkpoint_11858658_steps.zip`
- `checkpoints/checkpoint_19438316_steps.zip`
- `checkpoints/checkpoint_28023692_steps.zip`
- `checkpoints/checkpoint_35946535_steps.zip`
- `checkpoints/checkpoint_38937343_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v6_06_unified_all_N_0616</code> — 0.472 GB freed, 10 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1180485_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_12836693_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_23422064_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_26107495_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_1180485_steps.zip`
- `checkpoints/checkpoint_12836693_steps.zip`
- `checkpoints/checkpoint_23422064_steps.zip`
- `checkpoints/checkpoint_26107495_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v6_07_unified_topk_N_0616</code> — 0.764 GB freed, 10 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1187322_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_13922816_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_24553381_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_31634159_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_1187322_steps.zip`
- `checkpoints/checkpoint_13922816_steps.zip`
- `checkpoints/checkpoint_24553381_steps.zip`
- `checkpoints/checkpoint_31634159_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v6_08_unmasked_floor_N_0617</code> — 0.585 GB freed, 9 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1181770_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_12933045_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_22002567_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_1181770_steps.zip`
- `checkpoints/checkpoint_12933045_steps.zip`
- `checkpoints/checkpoint_22002567_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v6_09_dmg_reattend_N_0617</code> — 1.169 GB freed, 11 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1196649_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_12464402_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_22345697_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_32837718_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_42458933_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_1196649_steps.zip`
- `checkpoints/checkpoint_12464402_steps.zip`
- `checkpoints/checkpoint_22345697_steps.zip`
- `checkpoints/checkpoint_32837718_steps.zip`
- `checkpoints/checkpoint_42458933_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v6_10_unified_obs_0618</code> — 0.272 GB freed, 8 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1197172_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_9572669_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_1197172_steps.zip`
- `checkpoints/checkpoint_9572669_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v6_11_typed_hp_0619</code> — 1.355 GB freed, 12 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_11910141_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_1197110_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_22676708_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_32864563_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_42953465_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_45793883_steps.json` — checkpoint sidecar — the record, not the weights
- `elo` — the era's training record — never thinned
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `review_notes.json` — small run-root bookkeeping (not a .zip or a .log)
- `review_notes.md` — small run-root bookkeeping (not a .zip or a .log)
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_11910141_steps.zip`
- `checkpoints/checkpoint_1197110_steps.zip`
- `checkpoints/checkpoint_22676708_steps.zip`
- `checkpoints/checkpoint_32864563_steps.zip`
- `checkpoints/checkpoint_42953465_steps.zip`
- `checkpoints/checkpoint_45793883_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v6_11_unified_obs_fixed_0618</code> — 1.03 GB freed, 11 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1196924_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_12022153_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_22258877_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_31731498_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_33633775_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `review_notes.json` — small run-root bookkeeping (not a .zip or a .log)
- `review_notes.md` — small run-root bookkeeping (not a .zip or a .log)
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_1196924_steps.zip`
- `checkpoints/checkpoint_12022153_steps.zip`
- `checkpoints/checkpoint_22258877_steps.zip`
- `checkpoints/checkpoint_31731498_steps.zip`
- `checkpoints/checkpoint_33633775_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

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

<details><summary><code>ai_v6_13_outgoing_dmg_0620_exploiter_v1</code> — 0.395 GB freed, 10 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_106204357_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_115479375_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_124763606_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_127523438_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_96917276_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_106204357_steps.zip`
- `checkpoints/checkpoint_115479375_steps.zip`
- `checkpoints/checkpoint_124763606_steps.zip`
- `checkpoints/checkpoint_127523438_steps.zip`
- `checkpoints/checkpoint_96917276_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `stalls`

</details>

<details><summary><code>ai_v6_13_outgoing_dmg_0620_exploiter_v2</code> — 0.446 GB freed, 10 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_15627261_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_25106888_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_34673207_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_35605573_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_5727495_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_15627261_steps.zip`
- `checkpoints/checkpoint_25106888_steps.zip`
- `checkpoints/checkpoint_34673207_steps.zip`
- `checkpoints/checkpoint_35605573_steps.zip`
- `checkpoints/checkpoint_5727495_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `stalls`

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

<details><summary><code>ai_v7_01_teacher_0626_oom1</code> — 0.057 GB freed, 7 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1197011_steps.json` — checkpoint sidecar — the record, not the weights
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_1197011_steps.zip`
- `crashes`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v7_03_belief_shape_0630</code> — 1.454 GB freed, 13 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1197121_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_12872291_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_23545725_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_34144809_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_44530677_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_55823083_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_59697301_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_1197121_steps.zip`
- `checkpoints/checkpoint_12872291_steps.zip`
- `checkpoints/checkpoint_23545725_steps.zip`
- `checkpoints/checkpoint_34144809_steps.zip`
- `checkpoints/checkpoint_44530677_steps.zip`
- `checkpoints/checkpoint_55823083_steps.zip`
- `checkpoints/checkpoint_59697301_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v7_04_opd_selfdistill_0702</code> — 1.928 GB freed, 23 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_107652399_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_117444375_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_1197490_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_127557393_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_13070322_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_135694065_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_22872762_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_33480321_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_43968969_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_54574533_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_65944051_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_76349880_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_86931944_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_97532917_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_107652399_steps.zip`
- `checkpoints/checkpoint_117444375_steps.zip`
- `checkpoints/checkpoint_1197490_steps.zip`
- `checkpoints/checkpoint_127557393_steps.zip`
- `checkpoints/checkpoint_13070322_steps.zip`
- `checkpoints/checkpoint_135694065_steps.zip`
- `checkpoints/checkpoint_22872762_steps.zip`
- `checkpoints/checkpoint_33480321_steps.zip`
- `checkpoints/checkpoint_43968969_steps.zip`
- `checkpoints/checkpoint_54574533_steps.zip`
- `checkpoints/checkpoint_65944051_steps.zip`
- `checkpoints/checkpoint_76349880_steps.zip`
- `checkpoints/checkpoint_86931944_steps.zip`
- `checkpoints/checkpoint_97532917_steps.zip`
- `crashes`
- `eval_traces`
- `final_model_exception.zip`
- `launcher_child.log`
- `snapshots`
- `stalls`
- `teacher_persist`

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

<details><summary><code>ai_v7_05_tss_specialist_0703_aborted_noeval</code> — 0.001 GB freed, 4 entries deleted</summary>

**KEEP**

- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints`
- `launcher_child.log`

</details>

<details><summary><code>ai_v7_06_tss_temp_anneal_0706</code> — 0.37 GB freed, 9 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1059516_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_12309055_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_23777456_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_32470224_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_1059516_steps.zip`
- `checkpoints/checkpoint_12309055_steps.zip`
- `checkpoints/checkpoint_23777456_steps.zip`
- `checkpoints/checkpoint_32470224_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `stalls`

</details>

<details><summary><code>ai_v7_07_tss_temp_ratchet_0707</code> — 0.315 GB freed, 8 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1025468_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_12330312_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_18609109_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `exploiter_temp_state.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_1025468_steps.zip`
- `checkpoints/checkpoint_12330312_steps.zip`
- `checkpoints/checkpoint_18609109_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `stalls`

</details>

<details><summary><code>ai_v7_08_tss_bots_0707</code> — 0.468 GB freed, 11 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1196467_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_13165413_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_25223633_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_37476753_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_49631583_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_56813073_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_1196467_steps.zip`
- `checkpoints/checkpoint_13165413_steps.zip`
- `checkpoints/checkpoint_25223633_steps.zip`
- `checkpoints/checkpoint_37476753_steps.zip`
- `checkpoints/checkpoint_49631583_steps.zip`
- `checkpoints/checkpoint_56813073_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `stalls`

</details>

<details><summary><code>ai_v7_09_tss_bots_pubval_0708</code> — 0.503 GB freed, 11 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1196959_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_13165365_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_25222836_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_38378958_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_51435737_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_57421191_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_1196959_steps.zip`
- `checkpoints/checkpoint_13165365_steps.zip`
- `checkpoints/checkpoint_25222836_steps.zip`
- `checkpoints/checkpoint_38378958_steps.zip`
- `checkpoints/checkpoint_51435737_steps.zip`
- `checkpoints/checkpoint_57421191_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `stalls`

</details>

<details><summary><code>ai_v7_11_tss_exploiter_nopubval</code> — 0.308 GB freed, 9 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1026448_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_12108035_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_23437994_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_25491946_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `exploiter_temp_state.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_1026448_steps.zip`
- `checkpoints/checkpoint_12108035_steps.zip`
- `checkpoints/checkpoint_23437994_steps.zip`
- `checkpoints/checkpoint_25491946_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `stalls`

</details>

<details><summary><code>ai_v7_16_distill_tss_mvp_0715</code> — 0.28 GB freed, 8 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_149598552_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_154637567_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_149598552_steps.zip`
- `checkpoints/checkpoint_154637567_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v7_19_combined_0716</code> — 0.465 GB freed, 9 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_159499550_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_170014290_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_175523633_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_159499550_steps.zip`
- `checkpoints/checkpoint_170014290_steps.zip`
- `checkpoints/checkpoint_175523633_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v7_20_valuedistill_SMOKE</code> — 0.0 GB freed, 1 entries deleted</summary>

**KEEP**

- `command.txt` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `checkpoints`

</details>

<details><summary><code>ai_v7_20_valuedistill_ab_0717</code> — 0.265 GB freed, 8 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_149598474_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_152670073_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_149598474_steps.zip`
- `checkpoints/checkpoint_152670073_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v7_21_fitnet_valuefeat_ab_0717</code> — 0.292 GB freed, 8 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_149598992_steps.json` — checkpoint sidecar — the record, not the weights
- `checkpoints/checkpoint_159925420_steps.json` — checkpoint sidecar — the record, not the weights
- `eval_results.jsonl` — the run's identity + result record
- `final_model_interrupted.json` — small run-root bookkeeping (not a .zip or a .log)
- `final_model_interrupted.zip` — THE final model — what resolve_model_ref picks
- `latest.txt` — the run's identity + result record
- `metadata.json` — the run's identity + result record
- `model_config.json` — the run's identity + result record
- `tb` — the era's training record — never thinned
- everything else in the run dir is DELETED — the keep-list above IS the policy

**DELETE**

- `.eval_runs`
- `best_model`
- `checkpoints/checkpoint_149598992_steps.zip`
- `checkpoints/checkpoint_159925420_steps.zip`
- `eval_traces`
- `launcher_child.log`
- `snapshots`
- `stalls`

</details>

<details><summary><code>ai_v8_02_zarch_teampfsp_0718</code> — 0.134 GB freed, 2 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_172994305_steps.json` — first
- `checkpoints/checkpoint_172994305_steps.zip` — first
- `checkpoints/checkpoint_173989527_steps.json` — last
- `checkpoints/checkpoint_173989527_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_171990511/snapshot.zip`
- `snapshots`

</details>

<details><summary><code>ai_v8_05_semistall564_exploiter_0722</code> — 0.089 GB freed, 3 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278642638_steps.json` — first
- `checkpoints/checkpoint_278642638_steps.zip` — first
- `checkpoints/checkpoint_280757412_steps.json` — last
- `checkpoints/checkpoint_280757412_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_279702211_steps.json`
- `checkpoints/checkpoint_279702211_steps.zip`
- `eval_traces/step_278000005/snapshot.zip`

</details>

<details><summary><code>ai_v8_07_semistall564_scratch_0722</code> — 1.202 GB freed, 44 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_1049346_steps.json` — first
- `checkpoints/checkpoint_1049346_steps.zip` — first
- `checkpoints/checkpoint_21112628_steps.json` — last
- `checkpoints/checkpoint_21112628_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_11081568_steps.json`
- `checkpoints/checkpoint_11081568_steps.zip`
- `checkpoints/checkpoint_12137833_steps.json`
- `checkpoints/checkpoint_12137833_steps.zip`
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

- `checkpoints/checkpoint_278641713_steps.json` — first
- `checkpoints/checkpoint_278641713_steps.zip` — first
- `checkpoints/checkpoint_280753159_steps.json` — last
- `checkpoints/checkpoint_280753159_steps.zip` — last
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `checkpoints/checkpoint_279699715_steps.json`
- `checkpoints/checkpoint_279699715_steps.zip`
- `eval_traces/step_278000014/snapshot.zip`

</details>

<details><summary><code>ai_v8_10_offense20_exploiter_0724</code> — 0.883 GB freed, 29 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278624867_steps.json` — first
- `checkpoints/checkpoint_278624867_steps.zip` — first
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
- `checkpoints/checkpoint_289605804_steps.json`
- `checkpoints/checkpoint_289605804_steps.zip`
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

- `checkpoints/checkpoint_278630457_steps.json` — first
- `checkpoints/checkpoint_278630457_steps.zip` — first
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

<details><summary><code>ai_v8_12_defensive20_exploiter_0724</code> — 1.503 GB freed, 49 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278631107_steps.json` — first
- `checkpoints/checkpoint_278631107_steps.zip` — first
- `checkpoints/checkpoint_300470594_steps.json` — last
- `checkpoints/checkpoint_300470594_steps.zip` — last
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
- `checkpoints/checkpoint_289605699_steps.json`
- `checkpoints/checkpoint_289605699_steps.zip`
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

<details><summary><code>ai_v8_16_def20_lut_0726</code> — 0.909 GB freed, 29 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278639657_steps.json` — first
- `checkpoints/checkpoint_278639657_steps.zip` — first
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
- `checkpoints/checkpoint_289528972_steps.json`
- `checkpoints/checkpoint_289528972_steps.zip`
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

<details><summary><code>ai_v8_17_rand20_nolut_0726</code> — 0.917 GB freed, 29 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278635878_steps.json` — first
- `checkpoints/checkpoint_278635878_steps.zip` — first
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
- `checkpoints/checkpoint_289428098_steps.json`
- `checkpoints/checkpoint_289428098_steps.zip`
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

<details><summary><code>ai_v8_18_rand20_lut_0726</code> — 0.892 GB freed, 29 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278638971_steps.json` — first
- `checkpoints/checkpoint_278638971_steps.zip` — first
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
- `checkpoints/checkpoint_289536403_steps.json`
- `checkpoints/checkpoint_289536403_steps.zip`
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

<details><summary><code>ai_v8_19_def20_lut_zeroinit_0727</code> — 0.896 GB freed, 29 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278671536_steps.json` — first
- `checkpoints/checkpoint_278671536_steps.zip` — first
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
- `checkpoints/checkpoint_290083231_steps.json`
- `checkpoints/checkpoint_290083231_steps.zip`
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

<details><summary><code>ai_v8_20_rand10_nolut_0727</code> — 0.909 GB freed, 29 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_278635763_steps.json` — first
- `checkpoints/checkpoint_278635763_steps.zip` — first
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
- `checkpoints/checkpoint_289520500_steps.json`
- `checkpoints/checkpoint_289520500_steps.zip`
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
