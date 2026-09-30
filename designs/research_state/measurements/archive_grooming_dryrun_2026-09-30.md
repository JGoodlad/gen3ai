# Archive-grooming DRY RUN — `models/`

*policy: **standing***

*Generated 2026-09-30T11:45:11-0700 · `/home/goodlad/dev/gen3ai/models`*

> **NOTHING WAS DELETED IN THIS PASS.** This is a census; the plan below is what the retention policy *would* do, and it was produced with `--apply` absent.

## Headline

| | |
|---|---|
| runs in the archive | **276** |
| total size | **332.499 GB** |
| … of which physically under `models/` | 332.499 GB |
| … in 0 SYMLINKED run dirs (elsewhere on disk) | 0.0 GB |
| the policy would free | **28.789 GB** (8.7%) |
| entries in the plan | 1368 |
| runs with a non-empty plan | 81 |
| LIVE / REFERENCED / CLOSED | 0 / 176 / 100 |
| runs vetoed by a named file | 1 |
| **CLOSED runs needing review** | **7** |

## The policy

Applied to **CLOSED runs only**, and only inside `checkpoints`, `eval_traces`.

- **`checkpoints/`** — keep the FIRST, the LAST, every 10th, whatever `latest.txt` pins, and any checkpoint another run's `lineage` block resolved to. A `.json` sidecar is kept or dropped with its `.zip`, by STEP.
- **`eval_traces/`** — `main.prober.groom` at 3/1. The groomer's own planner is called, not re-implemented, so the two can never drift.
- **Never touched**: `best_model`, `cf_labels`, `cf_records`, `crashes`, `elo`, `snapshot_ladder`, `snapshots`, `stalls`, `tb`, `tb_imgs`, and the run-root files `capacity_battery.json`, `command.txt`, `eval_results.jsonl`, `latest.txt`, `launcher_child.log`, `metadata.json`, `model_config.json`, `team_winrates.json`, `team_winrates_history.jsonl`. `_assert_safe` re-checks every planned path against these before the plan is reported or executed.
- A run is REFERENCED — and therefore untouched — if a launcher process names it, the ledger's last 1000 lines name it, a committed **script** names it, it is v8-era, it was touched within 7 days, or it is a (transitive) fork parent of a LIVE run.
- Prose that merely *mentions* a run does **not** protect it — the historical record names nearly every run forever, so a `.md` mention as a live reference would close nothing. A committed script does protect it (a script names a run dir in order to load it), and prose still **vetoes** when it names an exact path the plan would delete.
- `snapshots/` (the self-play pool) is **out of scope** even though it is the second-largest consumer — the standing policy says nothing about it, so this pass measures it and proposes nothing.

## Top 20 runs by GB freed

| # | run | generation | GB freed | ckpts deleted | trace steps deleted |
|---|---|---|---:|---:|---:|
| 1 | `ai_v9_09_gen8_beliefs_threat_inject_0811` | ai_v9 | 1.043 | 10 | 10 |
| 2 | `ai_v9_12_gen10_t0prior_0814` | ai_v9 | 1.013 | 12 | 10 |
| 3 | `ai_v9_13_gen11_labelonly_winprob_0815` | ai_v9 | 0.994 | 12 | 9 |
| 4 | `ai_v9_19_gen16_mechanics_0819` | ai_v9 | 0.853 | 12 | 9 |
| 5 | `ai_v9_79_REVIVE1a_0830` | ai_v9 | 0.838 | 44 | 0 |
| 6 | `ai_v9_80_REVIVE1b_0830` | ai_v9 | 0.838 | 44 | 0 |
| 7 | `ai_v9_81_REVIVE1c_0830` | ai_v9 | 0.838 | 44 | 0 |
| 8 | `ai_v9_60_R2TOPK_0827` | ai_v9 | 0.656 | 34 | 0 |
| 9 | `ai_v9_61_R2KL_0827` | ai_v9 | 0.656 | 34 | 0 |
| 10 | `ai_v9_111_R5F19_0831` | ai_v9 | 0.656 | 34 | 0 |
| 11 | `ai_v9_110_R5F18_0831` | ai_v9 | 0.656 | 34 | 0 |
| 12 | `ai_v9_52_G1p_matched_0826` | ai_v9 | 0.656 | 34 | 0 |
| 13 | `ai_v9_49_G2_advgate_0826` | ai_v9 | 0.656 | 34 | 0 |
| 14 | `ai_v9_38_fdA_coef03_0825` | ai_v9 | 0.656 | 34 | 0 |
| 15 | `ai_v9_109_R5F17_0831` | ai_v9 | 0.656 | 34 | 0 |
| 16 | `ai_v9_108_R5F16_0831` | ai_v9 | 0.656 | 34 | 0 |
| 17 | `ai_v9_39_fdB_lossonly_0825` | ai_v9 | 0.656 | 34 | 0 |
| 18 | `ai_v9_107_R5F15_0831` | ai_v9 | 0.656 | 34 | 0 |
| 19 | `ai_v9_42_fdE_single_0825` | ai_v9 | 0.656 | 34 | 0 |
| 20 | `ai_v9_95_R5F03_0831` | ai_v9 | 0.656 | 34 | 0 |

## Runs vetoed because a committed file or the ledger names a file in the plan

These are excluded from the deletion set automatically.

| run | GB it would have freed | example named path | named by |
|---|---:|---|---|
| `ai_v9_44_tock2_v8shape_0825` | 2.077 | `eval_traces/step_30000000/snapshot.zip` | designs/research_state/era_boundary_deprecation_2026-09-06.md, designs/research_state/measurements/archive_grooming_dryrun_2026-09-06.json |

## ⚠️ SYMLINKED run dirs — the data is NOT under `models/`

These entries are symlinks into launcher worktrees, so `du -sh models/` does not see them and a deletion "in `models/`" would physically land under `.claude/worktrees/`. They are held out of the plan by default; `--follow-symlinked-runs` opts in after you have confirmed the targets are still the ones you mean.

*(none)*

## ⚠️ REVIEW BEFORE APPLYING — CLOSED runs the ledger names outside its tail

The tail window is what protects a run; this section makes its EDGE visible rather than silent. Each of these has a non-empty plan **and** is named somewhere higher up `ledger.md`, so a banked result may still rest on it. They are still in the deletion set — read them before running `--apply`, and widen `--ledger-tail-lines` (or delete the run's entry from the plan) if any should be kept.

| run | generation | GB freed | named at | prose mentions |
|---|---|---:|---|---:|
| `ai_v12_04_pfsp_fork25M` | ai_v12 | 0.071 | `ledger.md:15208` | 7 |
| `ai_v12_27_ladder_ctrl10M_shaped_dense` | ai_v12 | 0.303 | `ledger.md:18008` | 3 |
| `ai_v7_14_league_capstone_0712` | ai_v7 | 0.13 | `ledger.md:14285` | 10 |
| `ai_v9_107_R5F15_0831` | ai_v9 | 0.656 | `ledger.md:8497` | 12 |
| `ai_v9_172_G1SHORT_0905` | ai_v9 | 0.255 | `ledger.md:10935` | 16 |
| `ai_v9_38_fdA_coef03_0825` | ai_v9 | 0.656 | `ledger.md:11018` | 19 |
| `ai_v9_50_fdF_p1c_0826` | ai_v9 | 0.291 | `ledger.md:17963` | 19 |

## Per-run census

Sizes in GB. `plan GB` is 0 for every run that is not CLOSED.

| run | gen | cfg | status | ckpts | best | snaps | traces | tb | other | total | plan GB |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ai_v8_03_zarch_control_0718` | ai_v8 | 45 | REFERENCED | 4.938 | 0.045 | 0.416 | 1.631 | 0.011 | 0.412 | 7.459 | 0.0 |
| `ai_v9_29_rev1_0823` | ai_v9 | 101 | REFERENCED | 4.625 | 0.036 | 0.436 | 0.793 | 0.005 | 0.075 | 6.092 | 0.0 |
| `ai_v13_33_core_burnin` | ai_v13 | 120 | REFERENCED | 2.258 | 0.035 | 0.706 | 1.394 | 0.031 | 0.206 | 4.636 | 0.0 |
| `ai_v5_6_stable_70m_0608` | ai_v5 | 7 | CLOSED | 0.311 | 0.028 | 0.566 | 0.075 | 0.001 | 2.691 | 3.987 | 0.255 |
| `ai_v13_02_flywheel_winprob` | ai_v13 | 119 | REFERENCED | 0.882 | 0.035 | 0.706 | 1.342 | 0.016 | 0.451 | 3.441 | 0.0 |
| `ai_v13_04_flywheel_winprob_b` | ai_v13 | 119 | REFERENCED | 0.847 | 0.035 | 0.706 | 1.27 | 0.016 | 0.451 | 3.333 | 0.0 |
| `ai_v14_01_base` | ai_v14 | 121 | REFERENCED | 0.851 | 0.035 | 0.709 | 1.257 | 0.016 | 0.451 | 3.329 | 0.0 |
| `ai_v9_34_tick1_0824` | ai_v9 | 101 | REFERENCED | 2.404 | 0.036 | 0.182 | 0.331 | 0.007 | 0.075 | 3.165 | 0.0 |
| `ai_v5_9_attend_unrevealed_56m_0610` | ai_v5 | 8 | CLOSED | 0.257 | 0.029 | 0.571 | 0.1 | 0.001 | 2.115 | 3.08 | 0.2 |
| `ai_v6_01_belief_53m_0613` | ai_v6 | 16 | CLOSED | 0.245 | 0.031 | 0.551 | 0.117 | 0.002 | 2.087 | 3.039 | 0.184 |
| `ai_v5_5_popart_50m_0607` | ai_v5 | 6 | REFERENCED | 0.255 | 0.028 | 0.566 | 0.071 | 0.001 | 2.068 | 3.001 | 0.0 |
| `ai_v5_11_tail2_53m_0611` | ai_v5 | 11 | REFERENCED | 0.235 | 0.029 | 0.587 | 0.117 | 0.001 | 2.027 | 3.001 | 0.0 |
| `ai_v13_01_flywheel_shaped` | ai_v13 | 119 | REFERENCED | 0.854 | 0.036 | 0.711 | 1.233 | 0.018 | 0.091 | 2.951 | 0.0 |
| `ai_v9_75_R4S3c_0829` | ai_v9 | 107 | REFERENCED | 2.404 | 0.036 | 0.0 | 0.342 | 0.007 | 0.074 | 2.896 | 0.0 |
| `ai_v9_74_R4S3b_0829` | ai_v9 | 107 | REFERENCED | 2.404 | 0.036 | 0.0 | 0.322 | 0.007 | 0.074 | 2.875 | 0.0 |
| `ai_v12_02_winprob_critic` | ai_v12 | 110 | REFERENCED | 0.882 | 0.035 | 0.706 | 1.118 | 0.016 | 0.072 | 2.836 | 0.0 |
| `ai_v9_73_R4S3a_0829` | ai_v9 | 107 | REFERENCED | 2.367 | 0.036 | 0.0 | 0.304 | 0.007 | 0.074 | 2.819 | 0.0 |
| `ai_v9_44_tock2_v8shape_0825` | ai_v9 | 101 | REFERENCED | 2.149 | 0.036 | 0.0 | 0.285 | 0.006 | 0.074 | 2.573 | 0.0 |
| `ai_v5_3_vf_coef_clip_50m_0606` | ai_v5 | 3 | CLOSED | 0.198 | 0.028 | 0.481 | 0.067 | 0.001 | 1.755 | 2.534 | 0.141 |
| `ai_v5_13_shape_pbrs_43m_0612` | ai_v5 | 15 | CLOSED | 0.206 | 0.029 | 0.528 | 0.112 | 0.001 | 1.645 | 2.532 | 0.147 |
| `ai_v5_7_switch_bias_41m_0609` | ai_v5 | 7 | CLOSED | 0.198 | 0.028 | 0.453 | 0.099 | 0.001 | 1.502 | 2.291 | 0.142 |
| `ai_v13_11_split_lossoff` | ai_v13 | 119 | REFERENCED | 0.812 | 0.035 | 0.706 | 0.563 | 0.019 | 0.133 | 2.275 | 0.0 |
| `ai_v5_8_split_inc_dmg_38m_0610` | ai_v5 | 7 | CLOSED | 0.171 | 0.029 | 0.457 | 0.089 | 0.001 | 1.431 | 2.19 | 0.143 |
| `ai_v13_17_fold_k1` | ai_v13 | 119 | REFERENCED | 0.812 | 0.035 | 0.706 | 0.46 | 0.024 | 0.142 | 2.185 | 0.0 |
| `ai_v13_19_fold_k11` | ai_v13 | 119 | REFERENCED | 0.812 | 0.035 | 0.706 | 0.455 | 0.024 | 0.142 | 2.18 | 0.0 |
| `ai_v13_18_fold_k3` | ai_v13 | 119 | REFERENCED | 0.812 | 0.035 | 0.706 | 0.453 | 0.024 | 0.142 | 2.177 | 0.0 |
| `ai_v13_21_wcont_b` | ai_v13 | 119 | REFERENCED | 0.776 | 0.035 | 0.706 | 0.497 | 0.019 | 0.133 | 2.174 | 0.0 |
| `ai_v13_09_wcont` | ai_v13 | 119 | REFERENCED | 0.776 | 0.035 | 0.706 | 0.488 | 0.019 | 0.132 | 2.165 | 0.0 |
| `ai_v13_20_fold_k11_sharematched` | ai_v13 | 119 | REFERENCED | 0.776 | 0.035 | 0.706 | 0.461 | 0.024 | 0.143 | 2.151 | 0.0 |
| `ai_v9_14_gen12_h_entitypool_shaping_0816` | ai_v9 | 80 | REFERENCED | 0.385 | 0.043 | 0.514 | 1.014 | 0.003 | 0.086 | 2.053 | 0.0 |
| `ai_v9_09_gen8_beliefs_threat_inject_0811` | ai_v9 | 64 | CLOSED | 0.315 | 0.045 | 0.585 | 1.035 | 0.002 | 0.046 | 2.043 | 1.043 |
| `ai_v9_16_gen14_framedel_v91_0817` | ai_v9 | 91 | REFERENCED | 0.379 | 0.042 | 0.505 | 1.016 | 0.004 | 0.085 | 2.039 | 0.0 |
| `ai_v7_04_opd_selfdistill_0702` | ai_v7 | 42 | CLOSED | 0.608 | 0.043 | 0.868 | 0.227 | 0.01 | 0.219 | 1.983 | 0.478 |
| `ai_v9_70_R3ACTION_0828` | ai_v9 | 103 | REFERENCED | 1.093 | 0.036 | 0.582 | 0.145 | 0.007 | 0.074 | 1.963 | 0.0 |
| `ai_v9_76_R4ACTION_0830` | ai_v9 | 107 | REFERENCED | 1.056 | 0.036 | 0.582 | 0.18 | 0.007 | 0.074 | 1.962 | 0.0 |
| `ai_v9_140_B2_0901` | ai_v9 | 107 | REFERENCED | 1.056 | 0.036 | 0.582 | 0.172 | 0.002 | 0.074 | 1.949 | 0.0 |
| `ai_v9_141_C1_0901` | ai_v9 | 107 | REFERENCED | 1.093 | 0.036 | 0.582 | 0.162 | 0.002 | 0.038 | 1.939 | 0.0 |
| `ai_v9_143_N2_0901` | ai_v9 | 107 | REFERENCED | 1.056 | 0.036 | 0.546 | 0.191 | 0.002 | 0.074 | 1.932 | 0.0 |
| `ai_v9_91_COMPFOLD_0831` | ai_v9 | 107 | REFERENCED | 1.056 | 0.036 | 0.582 | 0.148 | 0.007 | 0.074 | 1.93 | 0.0 |
| `ai_v9_71_R3ACTIONHI_0828` | ai_v9 | 104 | REFERENCED | 1.056 | 0.036 | 0.582 | 0.145 | 0.007 | 0.074 | 1.927 | 0.0 |
| `ai_v8_01_zarch_film_0717` | ai_v8 | 44 | REFERENCED | 0.938 | 0.045 | 0.017 | 0.843 | 0.002 | 0.063 | 1.914 | 0.0 |
| `ai_v9_15_gen13_hb_events_stack_0817` | ai_v9 | 89 | REFERENCED | 0.258 | 0.043 | 0.517 | 0.989 | 0.004 | 0.087 | 1.906 | 0.0 |
| `ai_v9_72_R3SELF_0828` | ai_v9 | 107 | REFERENCED | 1.093 | 0.036 | 0.509 | 0.155 | 0.007 | 0.074 | 1.899 | 0.0 |
| `ai_v9_12_gen10_t0prior_0814` | ai_v9 | 77 | CLOSED | 0.325 | 0.041 | 0.528 | 0.953 | 0.002 | 0.042 | 1.894 | 1.013 |
| `ai_v9_13_gen11_labelonly_winprob_0815` | ai_v9 | 77 | CLOSED | 0.331 | 0.041 | 0.496 | 0.933 | 0.002 | 0.084 | 1.894 | 0.994 |
| `ai_v9_142_N1_0901` | ai_v9 | 107 | REFERENCED | 1.056 | 0.036 | 0.509 | 0.183 | 0.002 | 0.074 | 1.887 | 0.0 |
| `ai_v6_13_outgoing_dmg_0620` | ai_v6 | 41 | REFERENCED | 0.542 | 0.045 | 0.903 | 0.203 | 0.007 | 0.137 | 1.841 | 0.0 |
| `ai_v9_18_gen15_v8rewards_0818` | ai_v9 | 95 | REFERENCED | 0.336 | 0.042 | 0.504 | 0.851 | 0.004 | 0.085 | 1.83 | 0.0 |
| `ai_v5_4_pbrs_opp_threat_50m_0607` | ai_v5 | 3 | CLOSED | 0.17 | 0.028 | 0.283 | 0.07 | 0.001 | 1.274 | 1.829 | 0.113 |
| `ai_v8_12_defensive20_exploiter_0724` | ai_v8 | 45 | REFERENCED | 0.938 | 0.045 | 0.0 | 0.77 | 0.014 | 0.046 | 1.819 | 0.0 |
| `ai_v13_22_popr1_loop` | ai_v13 | 119 | REFERENCED | 0.565 | 0.035 | 0.706 | 0.361 | 0.022 | 0.113 | 1.809 | 0.0 |
| `ai_v13_28_popr2_ctrl` | ai_v13 | 119 | REFERENCED | 0.529 | 0.035 | 0.706 | 0.384 | 0.024 | 0.113 | 1.798 | 0.0 |
| `ai_v13_27_popr2_loop` | ai_v13 | 119 | REFERENCED | 0.529 | 0.035 | 0.706 | 0.379 | 0.024 | 0.113 | 1.794 | 0.0 |
| `ai_v13_23_popr1_ctrl` | ai_v13 | 119 | REFERENCED | 0.529 | 0.035 | 0.706 | 0.364 | 0.022 | 0.113 | 1.777 | 0.0 |
| `ai_v9_10_gen9_intent_distcritic_0813` | ai_v9 | 69 | REFERENCED | 0.256 | 0.037 | 0.475 | 0.943 | 0.002 | 0.038 | 1.756 | 0.0 |
| `ai_v14_08_g0p_k3` | ai_v14 | 123 | REFERENCED | 0.568 | 0.035 | 0.709 | 0.295 | 0.022 | 0.113 | 1.753 | 0.0 |
| `ai_v14_05_lbat_l95` | ai_v14 | 123 | REFERENCED | 0.532 | 0.035 | 0.709 | 0.33 | 0.018 | 0.113 | 1.747 | 0.0 |
| `ai_v14_02_lbat_ctrl` | ai_v14 | 123 | REFERENCED | 0.532 | 0.035 | 0.709 | 0.327 | 0.018 | 0.113 | 1.743 | 0.0 |
| `ai_v14_03_lbat_e5` | ai_v14 | 123 | REFERENCED | 0.532 | 0.035 | 0.709 | 0.324 | 0.018 | 0.113 | 1.74 | 0.0 |
| `ai_v13_12_plateau` | ai_v13 | 119 | REFERENCED | 0.529 | 0.035 | 0.706 | 0.327 | 0.02 | 0.113 | 1.738 | 0.0 |
| `ai_v14_06_lbat_ctrl_fix` | ai_v14 | 123 | REFERENCED | 0.532 | 0.035 | 0.709 | 0.313 | 0.018 | 0.113 | 1.73 | 0.0 |
| `ai_v14_07_g0p_k2` | ai_v14 | 123 | REFERENCED | 0.532 | 0.035 | 0.709 | 0.31 | 0.02 | 0.113 | 1.729 | 0.0 |
| `ai_v7_02_critic_shape_0627` | ai_v7 | 42 | CLOSED | 0.478 | 0.043 | 0.868 | 0.209 | 0.008 | 0.088 | 1.701 | 0.391 |
| `ai_v9_37_tick1_dosext_0825` | ai_v9 | 101 | REFERENCED | 1.202 | 0.036 | 0.073 | 0.185 | 0.008 | 0.074 | 1.658 | 0.0 |
| `ai_v9_19_gen16_mechanics_0819` | ai_v9 | 97 | CLOSED | 0.285 | 0.036 | 0.427 | 0.799 | 0.004 | 0.072 | 1.629 | 0.853 |
| `ai_v9_21_gen17_pfspoff_0820` | ai_v9 | 97 | REFERENCED | 0.285 | 0.036 | 0.427 | 0.789 | 0.004 | 0.072 | 1.618 | 0.0 |
| `ai_v12_12_ladder_cflabels` | ai_v12 | 114 | REFERENCED | 0.67 | 0.035 | 0.141 | 0.631 | 0.002 | 0.117 | 1.603 | 0.0 |
| `ai_v13_07_fold1` | ai_v13 | 119 | REFERENCED | 0.388 | 0.035 | 0.706 | 0.291 | 0.018 | 0.111 | 1.555 | 0.0 |
| `ai_v8_14_distill3_0725` | ai_v8 | 45 | REFERENCED | 0.625 | 0.045 | 0.089 | 0.725 | 0.013 | 0.046 | 1.551 | 0.0 |
| `ai_v13_08_fold1_cont` | ai_v13 | 119 | REFERENCED | 0.388 | 0.035 | 0.706 | 0.285 | 0.019 | 0.111 | 1.55 | 0.0 |
| `ai_v8_15_retention_A_frozen_0726` | ai_v8 | 45 | REFERENCED | 0.715 | 0.045 | 0.089 | 0.604 | 0.015 | 0.046 | 1.518 | 0.0 |
| `ai_v9_67_R3F6e_0828` | ai_v9 | 103 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.205 | 0.006 | 0.038 | 1.516 | 0.0 |
| `ai_v9_40_fdC_ecology_0825` | ai_v9 | 101 | CLOSED | 0.728 | 0.036 | 0.509 | 0.174 | 0.006 | 0.038 | 1.514 | 0.656 |
| `ai_v7_03_belief_shape_0630` | ai_v7 | 42 | CLOSED | 0.304 | 0.043 | 0.868 | 0.229 | 0.004 | 0.045 | 1.502 | 0.217 |
| `ai_v9_62_R2PLAIN_0827` | ai_v9 | 103 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.154 | 0.006 | 0.037 | 1.495 | 0.0 |
| `ai_v9_42_fdE_single_0825` | ai_v9 | 101 | CLOSED | 0.728 | 0.036 | 0.509 | 0.15 | 0.006 | 0.038 | 1.491 | 0.656 |
| `ai_v9_38_fdA_coef03_0825` | ai_v9 | 101 | CLOSED | 0.728 | 0.036 | 0.509 | 0.151 | 0.006 | 0.037 | 1.489 | 0.656 |
| `ai_v9_60_R2TOPK_0827` | ai_v9 | 103 | CLOSED | 0.728 | 0.036 | 0.509 | 0.143 | 0.006 | 0.037 | 1.484 | 0.656 |
| `ai_v9_59_R2ACTION_0827` | ai_v9 | 103 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.143 | 0.006 | 0.037 | 1.484 | 0.0 |
| `ai_v9_68_R3F6f_0828` | ai_v9 | 103 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.177 | 0.006 | 0.037 | 1.482 | 0.0 |
| `ai_v9_61_R2KL_0827` | ai_v9 | 103 | CLOSED | 0.728 | 0.036 | 0.509 | 0.142 | 0.006 | 0.037 | 1.481 | 0.656 |
| `ai_v9_48_G1_action_0826` | ai_v9 | 103 | REFERENCED | 0.728 | 0.036 | 0.509 | 0.141 | 0.006 | 0.037 | 1.48 | 0.0 |
| `ai_v9_49_G2_advgate_0826` | ai_v9 | 103 | CLOSED | 0.728 | 0.036 | 0.509 | 0.139 | 0.006 | 0.037 | 1.478 | 0.656 |
| `ai_v9_52_G1p_matched_0826` | ai_v9 | 103 | CLOSED | 0.728 | 0.036 | 0.509 | 0.138 | 0.006 | 0.037 | 1.477 | 0.656 |
| `ai_v9_39_fdB_lossonly_0825` | ai_v9 | 101 | CLOSED | 0.728 | 0.036 | 0.509 | 0.137 | 0.006 | 0.037 | 1.476 | 0.656 |
| `ai_v9_63_R3F6a_0828` | ai_v9 | 103 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.167 | 0.006 | 0.038 | 1.468 | 0.0 |
| `ai_v9_66_R3F6d_0828` | ai_v9 | 103 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.167 | 0.006 | 0.037 | 1.468 | 0.0 |
| `ai_v9_64_R3F6b_0828` | ai_v9 | 103 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.167 | 0.006 | 0.038 | 1.467 | 0.0 |
| `ai_v9_65_R3F6c_0828` | ai_v9 | 103 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.161 | 0.006 | 0.038 | 1.464 | 0.0 |
| `ai_v9_69_R3F6CURR_0828` | ai_v9 | 103 | REFERENCED | 1.202 | 0.036 | 0.0 | 0.157 | 0.006 | 0.038 | 1.455 | 0.0 |
| `ai_v8_09_pool10_exploiter_0723` | ai_v8 | 45 | REFERENCED | 0.715 | 0.045 | 0.0 | 0.631 | 0.013 | 0.046 | 1.454 | 0.0 |
| `ai_v8_07_semistall564_scratch_0722` | ai_v8 | 45 | REFERENCED | 0.661 | 0.035 | 0.0 | 0.672 | 0.002 | 0.071 | 1.452 | 0.0 |
| `ai_v12_20_ladder_denseaux` | ai_v12 | 117 | REFERENCED | 0.186 | 0.062 | 0.249 | 0.769 | 0.002 | 0.175 | 1.447 | 0.0 |
| `ai_v9_82_REFOLD1_0830` | ai_v9 | 107 | REFERENCED | 1.056 | 0.036 | 0.073 | 0.133 | 0.007 | 0.074 | 1.404 | 0.0 |
| `ai_v9_24_E3_substrate_on_0822` | ai_v9 | 97 | CLOSED | 0.107 | 0.036 | 0.64 | 0.536 | 0.006 | 0.072 | 1.403 | 0.377 |
| `ai_v6_11_typed_hp_0619` | ai_v6 | 38 | CLOSED | 0.263 | 0.044 | 0.876 | 0.168 | 0.003 | 0.045 | 1.402 | 0.175 |
| `ai_v9_25_E4_baitbot_0822` | ai_v9 | 98 | REFERENCED | 0.107 | 0.036 | 0.711 | 0.448 | 0.007 | 0.072 | 1.386 | 0.0 |
| `ai_v5_12_bias_05_N_0612` | ai_v5 | 12 | REFERENCED | 0.117 | 0.029 | 0.235 | 0.107 | 0.0 | 0.882 | 1.38 | 0.0 |
| `ai_v9_23_E2_substrate_on_0822` | ai_v9 | 97 | CLOSED | 0.107 | 0.036 | 0.64 | 0.501 | 0.006 | 0.072 | 1.369 | 0.36 |
| `ai_v6_13_outgoing_dmg_0620_exp_v1` | ai_v6 | 41 | REFERENCED | 0.135 | 0.045 | 0.903 | 0.202 | 0.009 | 0.046 | 1.345 | 0.0 |
| `ai_v5_10_tail1_23_0611` | ai_v5 | 11 | CLOSED | 0.117 | 0.029 | 0.205 | 0.106 | 0.0 | 0.852 | 1.32 | 0.059 |
| `ai_v9_22_E1_substrate_on_0821` | ai_v9 | 97 | CLOSED | 0.107 | 0.036 | 0.64 | 0.432 | 0.006 | 0.072 | 1.297 | 0.321 |
| `ai_v9_26_baitent_probe_0823` | ai_v9 | 100 | REFERENCED | 0.0 | 0.036 | 0.711 | 0.377 | 0.006 | 0.107 | 1.24 | 0.0 |
| `ai_v13_16_teach5_offense_dist` | ai_v13 | 119 | REFERENCED | 0.071 | 0.035 | 0.706 | 0.277 | 0.022 | 0.112 | 1.225 | 0.0 |
| `ai_v8_17_rand20_nolut_0726` | ai_v8 | 46 | REFERENCED | 0.581 | 0.045 | 0.0 | 0.53 | 0.013 | 0.045 | 1.218 | 0.0 |
| `ai_v6_09_dmg_reattend_N_0617` | ai_v6 | 35 | CLOSED | 0.217 | 0.043 | 0.78 | 0.122 | 0.002 | 0.044 | 1.215 | 0.13 |
| `ai_v9_162_TCUNFA_0903` | ai_v9 | 107 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.165 | 0.002 | 0.11 | 1.213 | 0.0 |
| `ai_v8_20_rand10_nolut_0727` | ai_v8 | 46 | REFERENCED | 0.581 | 0.045 | 0.0 | 0.519 | 0.013 | 0.045 | 1.207 | 0.0 |
| `ai_v8_16_def20_lut_0726` | ai_v8 | 46 | REFERENCED | 0.564 | 0.04 | 0.0 | 0.535 | 0.013 | 0.041 | 1.199 | 0.0 |
| `ai_v9_81_REVIVE1c_0830` | ai_v9 | 107 | CLOSED | 0.947 | 0.036 | 0.0 | 0.135 | 0.007 | 0.037 | 1.197 | 0.838 |
| `ai_v9_150_R4DOSE12_0901` | ai_v9 | 107 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.181 | 0.002 | 0.074 | 1.193 | 0.0 |
| `ai_v8_19_def20_lut_zeroinit_0727` | ai_v8 | 46 | REFERENCED | 0.56 | 0.04 | 0.0 | 0.525 | 0.013 | 0.041 | 1.185 | 0.0 |
| `ai_v9_151_R4DOSE6_0901` | ai_v9 | 107 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.173 | 0.002 | 0.074 | 1.184 | 0.0 |
| `ai_v9_161_TCFUNDB_0903` | ai_v9 | 107 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.172 | 0.002 | 0.074 | 1.184 | 0.0 |
| `ai_v9_80_REVIVE1b_0830` | ai_v9 | 107 | CLOSED | 0.947 | 0.036 | 0.0 | 0.125 | 0.007 | 0.037 | 1.183 | 0.838 |
| `ai_v9_160_TCFUNDA_0903` | ai_v9 | 107 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.166 | 0.002 | 0.074 | 1.179 | 0.0 |
| `ai_v9_172_G1SHORT_0905` | ai_v9 | 107 | CLOSED | 0.291 | 0.036 | 0.582 | 0.167 | 0.002 | 0.074 | 1.179 | 0.255 |
| `ai_v9_170_TCUNFK6A_0904` | ai_v9 | 107 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.166 | 0.002 | 0.074 | 1.178 | 0.0 |
| `ai_v9_163_TCUNFB_0903` | ai_v9 | 107 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.166 | 0.002 | 0.074 | 1.177 | 0.0 |
| `ai_v8_18_rand20_lut_0726` | ai_v8 | 46 | REFERENCED | 0.564 | 0.04 | 0.0 | 0.513 | 0.013 | 0.041 | 1.176 | 0.0 |
| `ai_v9_171_TCUNFK6B_0904` | ai_v9 | 107 | REFERENCED | 0.291 | 0.036 | 0.582 | 0.162 | 0.002 | 0.074 | 1.176 | 0.0 |
| `ai_v8_10_offense20_exploiter_0724` | ai_v8 | 45 | REFERENCED | 0.581 | 0.045 | 0.0 | 0.489 | 0.013 | 0.046 | 1.176 | 0.0 |
| `ai_v9_79_REVIVE1a_0830` | ai_v9 | 107 | CLOSED | 0.947 | 0.036 | 0.0 | 0.12 | 0.007 | 0.037 | 1.175 | 0.838 |
| `ai_v9_77_G1LEAN_0830` | ai_v9 | 107 | REFERENCED | 0.911 | 0.036 | 0.0 | 0.139 | 0.006 | 0.037 | 1.164 | 0.0 |
| `ai_v12_23_ladder_rollout` | ai_v12 | 119 | REFERENCED | 0.076 | 0.039 | 0.19 | 0.682 | 0.002 | 0.136 | 1.149 | 0.0 |
| `ai_v5_2_native_selfplay_50m_0606` | ai_v5 | 2 | CLOSED | 0.113 | 0.028 | 0.085 | 0.058 | 0.0 | 0.85 | 1.141 | 0.057 |
| `ai_v8_13_defensive10_exploiter_0725` | ai_v8 | 45 | REFERENCED | 0.536 | 0.045 | 0.0 | 0.493 | 0.013 | 0.045 | 1.136 | 0.0 |
| `ai_v12_19_ladder_lambda09` | ai_v12 | 116 | REFERENCED | 0.106 | 0.035 | 0.176 | 0.683 | 0.002 | 0.119 | 1.126 | 0.0 |
| `ai_v9_152_R4DOSE3_0901` | ai_v9 | 107 | REFERENCED | 0.291 | 0.036 | 0.509 | 0.182 | 0.002 | 0.074 | 1.122 | 0.0 |
| `ai_v6_03_win_pred_N_0614` | ai_v6 | 22 | CLOSED | 0.29 | 0.032 | 0.644 | 0.114 | 0.004 | 0.034 | 1.121 | 0.225 |
| `ai_v12_04_pfsp_fork25M` | ai_v12 | 113 | CLOSED | 0.071 | 0.035 | 0.706 | 0.247 | 0.007 | 0.038 | 1.108 | 0.071 |
| `ai_v7_01_teacher_0626` | ai_v7 | 42 | CLOSED | 0.217 | 0.043 | 0.434 | 0.212 | 0.002 | 0.175 | 1.092 | 0.13 |
| `ai_v12_22_ladder_lambda095` | ai_v12 | 116 | REFERENCED | 0.106 | 0.035 | 0.176 | 0.649 | 0.002 | 0.119 | 1.092 | 0.0 |
| `ai_v12_26_ladder_ctrl10M_shaped` | ai_v12 | 113 | REFERENCED | 0.118 | 0.04 | 0.039 | 0.8 | 0.002 | 0.083 | 1.087 | 0.0 |
| `ai_v12_15_ladder_ctrl10M_b` | ai_v12 | 114 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.679 | 0.002 | 0.117 | 1.084 | 0.0 |
| `ai_v6_11_unified_obs_fixed_0618` | ai_v6 | 37 | CLOSED | 0.217 | 0.043 | 0.608 | 0.156 | 0.002 | 0.044 | 1.075 | 0.13 |
| `ai_v12_16_ladder_ctrl10M_c` | ai_v12 | 114 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.635 | 0.002 | 0.117 | 1.04 | 0.0 |
| `ai_v12_21_ladder_lambda09_b` | ai_v12 | 116 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.629 | 0.002 | 0.119 | 1.036 | 0.0 |
| `ai_v12_17_ladder_strata` | ai_v12 | 115 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.627 | 0.002 | 0.117 | 1.032 | 0.0 |
| `ai_v12_14_ladder_truevalue` | ai_v12 | 115 | REFERENCED | 0.109 | 0.036 | 0.145 | 0.617 | 0.002 | 0.119 | 1.031 | 0.0 |
| `ai_v9_58_R2CTRL_0827` | ai_v9 | 103 | REFERENCED | 0.728 | 0.036 | 0.073 | 0.114 | 0.005 | 0.037 | 1.019 | 0.0 |
| `ai_v9_50_fdF_p1c_0826` | ai_v9 | 103 | CLOSED | 0.364 | 0.036 | 0.473 | 0.07 | 0.005 | 0.038 | 1.011 | 0.291 |
| `ai_v12_24_ladder_strata_b` | ai_v12 | 115 | REFERENCED | 0.071 | 0.035 | 0.141 | 0.64 | 0.002 | 0.116 | 1.01 | 0.0 |
| `ai_v9_45_fdF_p1_0826` | ai_v9 | 102 | REFERENCED | 0.364 | 0.036 | 0.473 | 0.068 | 0.005 | 0.038 | 1.006 | 0.0 |
| `ai_v12_01_winprob_critic` | ai_v12 | 109 | REFERENCED | 0.146 | 0.024 | 0.268 | 0.53 | 0.002 | 0.025 | 1.001 | 0.0 |
| `ai_v9_51_fdF_p2c_0826` | ai_v9 | 103 | REFERENCED | 0.328 | 0.036 | 0.473 | 0.079 | 0.006 | 0.037 | 0.981 | 0.0 |
| `ai_v9_102_R5F10_0831` | ai_v9 | 107 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.138 | 0.005 | 0.037 | 0.973 | 0.0 |
| `ai_v9_94_R5F02_0831` | ai_v9 | 107 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.135 | 0.005 | 0.037 | 0.972 | 0.0 |
| `ai_v9_95_R5F03_0831` | ai_v9 | 107 | CLOSED | 0.728 | 0.036 | 0.0 | 0.131 | 0.005 | 0.037 | 0.966 | 0.656 |
| `ai_v9_110_R5F18_0831` | ai_v9 | 107 | CLOSED | 0.728 | 0.036 | 0.0 | 0.129 | 0.006 | 0.037 | 0.963 | 0.656 |
| `ai_v9_111_R5F19_0831` | ai_v9 | 107 | CLOSED | 0.728 | 0.036 | 0.0 | 0.127 | 0.006 | 0.037 | 0.96 | 0.656 |
| `ai_v9_100_R5F08_0831` | ai_v9 | 107 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.124 | 0.005 | 0.037 | 0.958 | 0.0 |
| `ai_v14_04b_lbat_t32` | ai_v14 | 123 | REFERENCED | 0.071 | 0.035 | 0.709 | 0.078 | 0.016 | 0.043 | 0.958 | 0.0 |
| `ai_v9_98_R5F06_0831` | ai_v9 | 107 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.12 | 0.005 | 0.037 | 0.955 | 0.0 |
| `ai_v9_106_R5F14_0831` | ai_v9 | 107 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.121 | 0.005 | 0.037 | 0.954 | 0.0 |
| `ai_v9_92_R5F00_0831` | ai_v9 | 107 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.123 | 0.005 | 0.037 | 0.953 | 0.0 |
| `ai_v9_96_R5F04_0831` | ai_v9 | 107 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.121 | 0.005 | 0.037 | 0.95 | 0.0 |
| `ai_v9_104_R5F12_0831` | ai_v9 | 107 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.122 | 0.005 | 0.037 | 0.949 | 0.0 |
| `ai_v9_103_R5F11_0831` | ai_v9 | 107 | CLOSED | 0.728 | 0.036 | 0.0 | 0.119 | 0.005 | 0.037 | 0.948 | 0.656 |
| `ai_v9_107_R5F15_0831` | ai_v9 | 107 | CLOSED | 0.728 | 0.036 | 0.0 | 0.121 | 0.006 | 0.037 | 0.948 | 0.656 |
| `ai_v9_27_extremedial_probe_0823` | ai_v9 | 100 | REFERENCED | 0.0 | 0.036 | 0.711 | 0.154 | 0.007 | 0.036 | 0.946 | 0.0 |
| `ai_v9_101_R5F09_0831` | ai_v9 | 107 | CLOSED | 0.728 | 0.036 | 0.0 | 0.112 | 0.005 | 0.037 | 0.946 | 0.656 |
| `ai_v9_97_R5F05_0831` | ai_v9 | 107 | CLOSED | 0.728 | 0.036 | 0.0 | 0.116 | 0.005 | 0.037 | 0.945 | 0.656 |
| `ai_v9_36_tock1c_q6_0824` | ai_v9 | 101 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.116 | 0.005 | 0.037 | 0.943 | 0.0 |
| `ai_v9_57_R2F5e_0826` | ai_v9 | 103 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.115 | 0.005 | 0.037 | 0.942 | 0.0 |
| `ai_v9_93_R5F01_0831` | ai_v9 | 107 | CLOSED | 0.728 | 0.036 | 0.0 | 0.113 | 0.005 | 0.037 | 0.941 | 0.656 |
| `ai_v9_105_R5F13_0831` | ai_v9 | 107 | CLOSED | 0.728 | 0.036 | 0.0 | 0.111 | 0.005 | 0.037 | 0.938 | 0.656 |
| `ai_v9_108_R5F16_0831` | ai_v9 | 107 | CLOSED | 0.728 | 0.036 | 0.0 | 0.113 | 0.006 | 0.037 | 0.938 | 0.656 |
| `ai_v9_56_R2F5d_0826` | ai_v9 | 103 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.111 | 0.005 | 0.037 | 0.938 | 0.0 |
| `ai_v9_55_R2F5c_0826` | ai_v9 | 103 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.109 | 0.005 | 0.037 | 0.938 | 0.0 |
| `ai_v9_109_R5F17_0831` | ai_v9 | 107 | CLOSED | 0.728 | 0.036 | 0.0 | 0.111 | 0.006 | 0.037 | 0.936 | 0.656 |
| `ai_v9_31_tock1_k4_0824` | ai_v9 | 101 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.11 | 0.005 | 0.037 | 0.936 | 0.0 |
| `ai_v9_32_tock1b_rain_0824` | ai_v9 | 101 | CLOSED | 0.728 | 0.036 | 0.0 | 0.108 | 0.005 | 0.037 | 0.936 | 0.656 |
| `ai_v9_99_R5F07_0831` | ai_v9 | 107 | CLOSED | 0.728 | 0.036 | 0.0 | 0.11 | 0.005 | 0.037 | 0.935 | 0.656 |
| `ai_v9_54_R2F5b_0826` | ai_v9 | 103 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.109 | 0.005 | 0.037 | 0.935 | 0.0 |
| `ai_v9_53_R2F5a_0826` | ai_v9 | 103 | REFERENCED | 0.728 | 0.036 | 0.0 | 0.109 | 0.005 | 0.037 | 0.935 | 0.0 |
| `ai_v6_04_unified_inc_N_0615` | ai_v6 | 23 | CLOSED | 0.201 | 0.033 | 0.501 | 0.113 | 0.002 | 0.034 | 0.891 | 0.134 |
| `ai_v13_03_fork` | ai_v13 | 120 | REFERENCED | 0.114 | 0.039 | 0.153 | 0.355 | 0.002 | 0.131 | 0.817 | 0.0 |
| `ai_v6_07_unified_topk_N_0616` | ai_v6 | 30 | CLOSED | 0.152 | 0.038 | 0.38 | 0.144 | 0.001 | 0.08 | 0.804 | 0.114 |
| `ai_v8_04_distill_4teacher_0722` | ai_v8 | 45 | REFERENCED | 0.357 | 0.045 | 0.045 | 0.284 | 0.012 | 0.046 | 0.798 | 0.0 |
| `ai_v9_20_tdaux_rung2_lam00_0820` | ai_v9 | 97 | CLOSED | 0.036 | 0.036 | 0.498 | 0.151 | 0.005 | 0.036 | 0.764 | 0.036 |
| `ai_v9_20_tdaux_rung2_lam30_0820` | ai_v9 | 97 | CLOSED | 0.036 | 0.036 | 0.498 | 0.15 | 0.005 | 0.036 | 0.764 | 0.036 |
| `ai_v9_20_tdaux_rung2_lam10_0820` | ai_v9 | 97 | CLOSED | 0.036 | 0.036 | 0.498 | 0.148 | 0.005 | 0.036 | 0.763 | 0.036 |
| `ai_v13_16_teach5_offense_dist_ABANDONED_forklr2p8` | ai_v13 | 119 | CLOSED | 0.0 | 0.0 | 0.706 | 0.0 | 0.02 | 0.035 | 0.761 | 0.0 |
| `ai_v12_25_ladder_vf15_b` | ai_v12 | 113 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.371 | 0.002 | 0.074 | 0.733 | 0.0 |
| `ai_v7_05_tss_specialist_0703` | ai_v7 | 42 | CLOSED | 0.478 | 0.043 | 0.0 | 0.149 | 0.008 | 0.045 | 0.724 | 0.391 |
| `ai_v12_11_ladder_ctrl10M` | ai_v12 | 113 | REFERENCED | 0.141 | 0.035 | 0.141 | 0.32 | 0.002 | 0.074 | 0.717 | 0.0 |
| `ai_v12_28_ladder_ent05` | ai_v12 | 113 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.351 | 0.002 | 0.073 | 0.712 | 0.0 |
| `ai_v12_29_ladder_vf025` | ai_v12 | 113 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.35 | 0.002 | 0.074 | 0.712 | 0.0 |
| `ai_v12_13_ladder_tdaux` | ai_v12 | 113 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.329 | 0.002 | 0.074 | 0.691 | 0.0 |
| `ai_v12_10_ladder_vf15` | ai_v12 | 113 | REFERENCED | 0.106 | 0.035 | 0.141 | 0.325 | 0.002 | 0.074 | 0.687 | 0.0 |
| `ai_v8_06_semistall_3team_exploiter_0722` | ai_v8 | 45 | REFERENCED | 0.313 | 0.045 | 0.0 | 0.252 | 0.012 | 0.046 | 0.674 | 0.0 |
| `ai_v9_130_R5FUND10_0901` | ai_v9 | 107 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.067 | 0.006 | 0.037 | 0.648 | 0.0 |
| `ai_v9_128_R5FUND08_0901` | ai_v9 | 107 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.068 | 0.006 | 0.037 | 0.648 | 0.0 |
| `ai_v9_195_G5PLAINA_0906` | ai_v9 | 107 | REFERENCED | 0.073 | 0.0 | 0.509 | 0.0 | 0.001 | 0.037 | 0.645 | 0.0 |
| `ai_v9_122_R5FUND02_0901` | ai_v9 | 107 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.068 | 0.001 | 0.037 | 0.645 | 0.0 |
| `ai_v9_197_G5PLAINC_0906` | ai_v9 | 107 | REFERENCED | 0.073 | 0.0 | 0.509 | 0.0 | 0.001 | 0.037 | 0.643 | 0.0 |
| `ai_v9_30_rev1_exploit_0824` | ai_v9 | 101 | CLOSED | 0.473 | 0.036 | 0.0 | 0.061 | 0.005 | 0.037 | 0.643 | 0.364 |
| `ai_v9_196_G5PLAINB_0906` | ai_v9 | 107 | REFERENCED | 0.073 | 0.0 | 0.509 | 0.0 | 0.001 | 0.037 | 0.643 | 0.0 |
| `ai_v9_35_tick1_exploit_0824` | ai_v9 | 101 | CLOSED | 0.473 | 0.036 | 0.0 | 0.06 | 0.008 | 0.037 | 0.641 | 0.364 |
| `ai_v9_132_R5FUND12_0901` | ai_v9 | 107 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.063 | 0.006 | 0.037 | 0.636 | 0.0 |
| `ai_v9_126_R5FUND06_0901` | ai_v9 | 107 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.062 | 0.001 | 0.037 | 0.636 | 0.0 |
| `ai_v9_134_R5FUND14_0901` | ai_v9 | 107 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.058 | 0.006 | 0.037 | 0.634 | 0.0 |
| `ai_v9_120_R5FUND00_0901` | ai_v9 | 107 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.062 | 0.001 | 0.037 | 0.632 | 0.0 |
| `ai_v9_124_R5FUND04_0901` | ai_v9 | 107 | REFERENCED | 0.473 | 0.036 | 0.0 | 0.061 | 0.001 | 0.037 | 0.63 | 0.0 |
| `ai_v12_27_ladder_ctrl10M_shaped_dense` | ai_v12 | 113 | CLOSED | 0.158 | 0.039 | 0.0 | 0.382 | 0.002 | 0.042 | 0.628 | 0.303 |
| `ai_v6_08_unmasked_floor_N_0617` | ai_v6 | 30 | CLOSED | 0.114 | 0.038 | 0.304 | 0.122 | 0.001 | 0.039 | 0.624 | 0.038 |
| `ai_v7_14_league_capstone_0712` | ai_v7 | 43 | CLOSED | 0.217 | 0.043 | 0.087 | 0.16 | 0.011 | 0.044 | 0.567 | 0.13 |
| `.aborted_R4DOSE12_nometa_1401` | ai_v9 (via lineage) | 107 | CLOSED | 0.0 | 0.0 | 0.509 | 0.0 | 0.001 | 0.037 | 0.56 | 0.0 |
| `ai_v13_15_exploit5_stall` | ai_v13 | 119 | REFERENCED | 0.071 | 0.035 | 0.0 | 0.3 | 0.022 | 0.112 | 0.554 | 0.0 |
| `ai_v7_09_tss_bots_pubval_0708` | ai_v7 | 43 | CLOSED | 0.262 | 0.044 | 0.0 | 0.152 | 0.004 | 0.089 | 0.551 | 0.175 |
| `ai_v7_15_tss_exploiter_vs14_0713` | ai_v7 | 43 | CLOSED | 0.35 | 0.044 | 0.0 | 0.087 | 0.006 | 0.045 | 0.537 | 0.263 |
| `.dryrun_K6A_1788581936` | ai_v9 (via lineage) | 107 | CLOSED | 0.0 | 0.0 | 0.509 | 0.0 | 0.001 | 0.0 | 0.522 | 0.0 |
| `ai_v7_19_combined_0716` | ai_v7 | 43 | CLOSED | 0.13 | 0.043 | 0.087 | 0.188 | 0.013 | 0.044 | 0.522 | 0.043 |
| `ai_v13_10_exploit_stall` | ai_v13 | 119 | REFERENCED | 0.071 | 0.035 | 0.0 | 0.274 | 0.018 | 0.112 | 0.52 | 0.0 |
| `ai_v13_14_exploit5_balance` | ai_v13 | 119 | REFERENCED | 0.071 | 0.035 | 0.0 | 0.271 | 0.022 | 0.112 | 0.519 | 0.0 |
| `ai_v7_08_tss_bots_0707` | ai_v7 | 42 | CLOSED | 0.26 | 0.043 | 0.0 | 0.163 | 0.004 | 0.045 | 0.515 | 0.174 |
| `ai_v8_11_offense10_exploiter_0724` | ai_v8 | 45 | REFERENCED | 0.223 | 0.045 | 0.0 | 0.183 | 0.012 | 0.045 | 0.509 | 0.0 |
| `ai_v6_06_unified_all_N_0616` | ai_v6 | 28 | CLOSED | 0.139 | 0.035 | 0.174 | 0.119 | 0.001 | 0.035 | 0.508 | 0.069 |
| `ai_v13_05_exploit_big5starmie` | ai_v13 | 119 | REFERENCED | 0.106 | 0.035 | 0.0 | 0.223 | 0.018 | 0.112 | 0.5 | 0.0 |
| `ai_v13_24_popr1_read_loop` | ai_v13 | 119 | REFERENCED | 0.106 | 0.035 | 0.0 | 0.219 | 0.024 | 0.112 | 0.497 | 0.0 |
| `ai_v6_13_outgoing_dmg_0620_exploiter_v2` | ai_v6 | 41 | CLOSED | 0.226 | 0.045 | 0.0 | 0.121 | 0.002 | 0.092 | 0.493 | 0.181 |
| `ai_v13_18_teach5_offense_hidose` | ai_v13 | 119 | REFERENCED | 0.106 | 0.035 | 0.0 | 0.215 | 0.022 | 0.112 | 0.491 | 0.0 |
| `ai_v13_30_popr2_read_ctrl` | ai_v13 | 119 | REFERENCED | 0.071 | 0.035 | 0.0 | 0.219 | 0.026 | 0.112 | 0.465 | 0.0 |
| `ai_v13_13_exploit5_offense` | ai_v13 | 119 | REFERENCED | 0.071 | 0.035 | 0.0 | 0.221 | 0.022 | 0.112 | 0.462 | 0.0 |
| `ai_v13_29_popr2_read_loop` | ai_v13 | 119 | REFERENCED | 0.071 | 0.035 | 0.0 | 0.217 | 0.026 | 0.112 | 0.462 | 0.0 |
| `ai_v13_26_popr0_exploit5_offense_s1002` | ai_v13 | 119 | REFERENCED | 0.071 | 0.035 | 0.0 | 0.219 | 0.022 | 0.112 | 0.461 | 0.0 |
| `ai_v13_06_exploit_ddtar_spikes` | ai_v13 | 119 | REFERENCED | 0.071 | 0.035 | 0.0 | 0.217 | 0.018 | 0.112 | 0.454 | 0.0 |
| `ai_v13_25_popr1_read_ctrl` | ai_v13 | 119 | REFERENCED | 0.071 | 0.035 | 0.0 | 0.211 | 0.024 | 0.112 | 0.454 | 0.0 |
| `ai_v6_13_outgoing_dmg_0620_exploiter_v1` | ai_v6 | 41 | CLOSED | 0.226 | 0.045 | 0.0 | 0.12 | 0.002 | 0.046 | 0.442 | 0.135 |
| `ai_v7_06_tss_temp_anneal_0706` | ai_v7 | 42 | CLOSED | 0.174 | 0.043 | 0.0 | 0.152 | 0.002 | 0.044 | 0.416 | 0.087 |
| `ai_v8_02_zarch_teampfsp_0718` | ai_v8 | 44 | REFERENCED | 0.089 | 0.045 | 0.089 | 0.129 | 0.002 | 0.045 | 0.404 | 0.0 |
| `ai_v7_17_stall_exploiter_0715` | ai_v7 | 43 | CLOSED | 0.174 | 0.044 | 0.0 | 0.112 | 0.002 | 0.044 | 0.385 | 0.087 |
| `ai_v8_08_defensive_6team_exploiter_0723` | ai_v8 | 45 | REFERENCED | 0.134 | 0.045 | 0.0 | 0.132 | 0.012 | 0.045 | 0.377 | 0.0 |
| `ai_v6_02_belief_lat_16m_0614` | ai_v6 | 21 | CLOSED | 0.094 | 0.031 | 0.125 | 0.083 | 0.001 | 0.032 | 0.371 | 0.031 |
| `ai_v9_17_tdaux_lam1_0818` | ai_v9 | 95 | CLOSED | 0.042 | 0.042 | 0.084 | 0.149 | 0.005 | 0.043 | 0.369 | 0.042 |
| `ai_v8_05_semistall564_exploiter_0722` | ai_v8 | 45 | REFERENCED | 0.134 | 0.045 | 0.0 | 0.122 | 0.012 | 0.045 | 0.365 | 0.0 |
| `ai_v9_17_tdaux_lam3_0818` | ai_v9 | 95 | REFERENCED | 0.042 | 0.042 | 0.084 | 0.145 | 0.005 | 0.043 | 0.364 | 0.0 |
| `ai_v7_07_tss_temp_ratchet_0707` | ai_v7 | 42 | CLOSED | 0.13 | 0.043 | 0.0 | 0.14 | 0.001 | 0.044 | 0.36 | 0.043 |
| `ai_v7_11_tss_exploiter_nopubval` | ai_v7 | 43 | CLOSED | 0.174 | 0.043 | 0.0 | 0.086 | 0.002 | 0.044 | 0.354 | 0.087 |
| `ai_v7_21_fitnet_valuefeat_ab_0717` | ai_v7 | 43 | CLOSED | 0.087 | 0.043 | 0.043 | 0.113 | 0.012 | 0.044 | 0.348 | 0.0 |
| `ai_v7_13_cmpass_exploiter_0711` | ai_v7 | 43 | CLOSED | 0.174 | 0.044 | 0.0 | 0.08 | 0.002 | 0.044 | 0.345 | 0.087 |
| `ai_v7_18_distill_4teacher_0716` | ai_v7 | 43 | CLOSED | 0.087 | 0.043 | 0.043 | 0.109 | 0.011 | 0.044 | 0.344 | 0.0 |
| `ai_v7_16_distill_tss_mvp_0715` | ai_v7 | 43 | CLOSED | 0.087 | 0.043 | 0.043 | 0.1 | 0.011 | 0.044 | 0.334 | 0.0 |
| `ai_v7_12_trap_exploiter_0711` | ai_v7 | 43 | CLOSED | 0.131 | 0.044 | 0.0 | 0.095 | 0.001 | 0.044 | 0.326 | 0.044 |
| `ai_v9_17_tdaux_control_0818` | ai_v9 | 95 | CLOSED | 0.042 | 0.042 | 0.042 | 0.147 | 0.005 | 0.043 | 0.325 | 0.042 |
| `v8rep_p1_A_0905` | ai_v8 (replication) | 45 | REFERENCED | 0.089 | 0.045 | 0.045 | 0.086 | 0.0 | 0.045 | 0.323 | 0.0 |
| `ai_v7_20_valuedistill_ab_0717` | ai_v7 | 43 | CLOSED | 0.087 | 0.043 | 0.043 | 0.082 | 0.011 | 0.044 | 0.319 | 0.0 |
| `ai_v6_04_unified_all_half_batch_N_0616` | ai_v6 | 28 | CLOSED | 0.104 | 0.035 | 0.035 | 0.101 | 0.001 | 0.036 | 0.316 | 0.035 |
| `ai_v6_10_unified_obs_0618` | ai_v6 | 37 | CLOSED | 0.087 | 0.043 | 0.0 | 0.134 | 0.001 | 0.044 | 0.316 | 0.0 |
| `ai_v7_10_tss_exploiter_fixed_0709` | ai_v7 | 43 | CLOSED | 0.131 | 0.044 | 0.0 | 0.089 | 0.002 | 0.045 | 0.315 | 0.044 |
| `DISCARDED_tdaux_control_n16_0818` | ai_v9 (via lineage) | 95 | CLOSED | 0.082 | 0.041 | 0.041 | 0.074 | 0.005 | 0.042 | 0.294 | 0.0 |
| `v8rep_p1_C_0905` | ai_v8 (replication) | 45 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.088 | 0.0 | 0.045 | 0.274 | 0.0 |
| `v8rep_p1_B_0905` | ai_v8 (replication) | 45 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.088 | 0.0 | 0.045 | 0.273 | 0.0 |
| `v8rep_p2loss_B_0905` | ai_v8 (replication) | 45 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.086 | 0.0 | 0.045 | 0.27 | 0.0 |
| `v8rep_p2loss_C_0905` | ai_v8 (replication) | 45 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.087 | 0.0 | 0.045 | 0.269 | 0.0 |
| `v8rep_p2loss_A_0905` | ai_v8 (replication) | 45 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.085 | 0.0 | 0.045 | 0.267 | 0.0 |
| `ai_v13_31_popr1_read_loop_ext` | ai_v13 | 119 | REFERENCED | 0.035 | 0.035 | 0.0 | 0.11 | 0.025 | 0.057 | 0.264 | 0.0 |
| `ai_v13_32_popr1_read_ctrl_ext` | ai_v13 | 119 | REFERENCED | 0.035 | 0.035 | 0.0 | 0.107 | 0.025 | 0.057 | 0.259 | 0.0 |
| `v8rep_p2self_C_0905` | ai_v8 (replication) | 45 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.067 | 0.0 | 0.045 | 0.249 | 0.0 |
| `v8rep_p2self_A_0905` | ai_v8 (replication) | 45 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.068 | 0.0 | 0.045 | 0.249 | 0.0 |
| `v8rep_p2self_B_0905` | ai_v8 (replication) | 45 | REFERENCED | 0.045 | 0.045 | 0.045 | 0.068 | 0.0 | 0.045 | 0.248 | 0.0 |
| `ai_v9_11_gen10_intentfull_compiled_0814` | ai_v9 | 77 | CLOSED | 0.041 | 0.041 | 0.041 | 0.076 | 0.0 | 0.041 | 0.246 | 0.0 |
| `ai_v14_09_r0_offense_a` | ai_v14 | 124 | REFERENCED | 0.0 | 0.035 | 0.0 | 0.053 | 0.02 | 0.041 | 0.15 | 0.0 |
| `ai_v7_01_teacher_0626_oom1` | ai_v7 | 42 | CLOSED | 0.043 | 0.0 | 0.0 | 0.0 | 0.0 | 0.044 | 0.1 | 0.0 |
| `.aborted_R4DOSE12_poolless_1355` | ai_v9 (via lineage) | 107 | CLOSED | 0.0 | 0.0 | 0.0 | 0.0 | 0.001 | 0.037 | 0.053 | 0.0 |
| `ai_v7_20_valuedistill_SMOKE` | ai_v7 | 43 | CLOSED | 0.0 | 0.0 | 0.0 | 0.0 | 0.011 | 0.042 | 0.053 | 0.0 |
| `ai_v7_05_tss_specialist_0703_aborted_noeval` | ai_v7 | 42 | CLOSED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.044 | 0.044 | 0.0 |
| `warmstart_generic_0715` | unknown | 43 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.043 | 0.043 | 0.0 |
| `run_20260830_184043` | unknown | 107 | CLOSED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.027 | 0.027 | 0.0 |
| `run_20260830_180409` | unknown | 107 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.027 | 0.027 | 0.0 |
| `run_20260906_083317` | unknown | 107 | REFERENCED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.01 | 0.01 | 0.0 |
| `run_20260830_183828` | unknown | 107 | CLOSED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.01 | 0.01 | 0.0 |
| `RETIRED_c5fork_control_gen13base_0817` | ai_v9 (via lineage) | 90 | CLOSED | 0.0 | 0.0 | 0.0 | 0.0 | 0.004 | 0.0 | 0.004 | 0.0 |
| `RETIRED_gen14_framedel_v90_0817` | unknown | 90 | CLOSED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.001 | 0.001 | 0.0 |
| `ai_v12_02_winprob_critic.OOM_4096` | ai_v12 | 110 | CLOSED | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.001 | 0.0 |

## Why each non-CLOSED run is protected

- **`ai_v12_01_winprob_critic`** — REFERENCED: loaded by 13 committed script(s): src/agents/model/model_version/fields.py, src/agents/training/instrumented_ppo/calibration.py, src/agents/training/instrumented_ppo/value_terms.py, src/agents/training/scaffolding.py …
- **`ai_v12_02_winprob_critic`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 56 committed script(s): designs/research_state/measurements/elo_calibration_external_anchors_2026-09-14/cells.py, designs/research_state/measurements/elo_calibration_external_anchors_2026-09-14/fit_joint.py, designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/entropy_forensics.py, designs/research_state/measurements/expand_many_2026-09-22/profile_expand_many.py …
- **`ai_v12_10_ladder_vf15`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, designs/research_state/measurements/repr_class_decode_2026-09-11/run.sh, designs/research_state/measurements/repr_class_decode_2026-09-11/tabulate.py, designs/research_state/measurements/repr_class_decode_strata_b_2026-09-12/tabulate.py
- **`ai_v12_11_ladder_ctrl10M`** — REFERENCED: loaded by 31 committed script(s): designs/research_state/measurements/critic_ladder_reads/cflabels_vs_ctrl10M_2026-09-09/matched_quota/subsample.py, designs/research_state/measurements/critic_ladder_reads/ent05_vs_ctrl10M_2026-09-12/hp_ent05.sh, designs/research_state/measurements/critic_ladder_reads/strata_b_vs_ctrl10M_2026-09-12/hp_strata_b.sh, designs/research_state/measurements/critic_ladder_reads/vf025_vs_ctrl10M_2026-09-13/hp_vf025.sh …
- **`ai_v12_12_ladder_cflabels`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_ladder_reads/cflabels_vs_ctrl10M_2026-09-09/matched_quota/subsample.py, designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, designs/research_state/measurements/search_dividend_winprob_heads_2026-09-11/run_battery3.sh, src/main/ops/quota_match.py
- **`ai_v12_13_ladder_tdaux`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py
- **`ai_v12_14_ladder_truevalue`** — REFERENCED: loaded by 5 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, src/agents/model/compile_preload.py, src/agents/model/compile_trainer.py, src/agents/model/extra_obs_keys.py …
- **`ai_v12_15_ladder_ctrl10M_b`** — REFERENCED: loaded by 10 committed script(s): designs/research_state/measurements/critic_ladder_reads/ent05_vs_ctrl10M_2026-09-12/hp_ent05.sh, designs/research_state/measurements/critic_ladder_reads/strata_b_vs_ctrl10M_2026-09-12/hp_strata_b.sh, designs/research_state/measurements/critic_ladder_reads/vf025_vs_ctrl10M_2026-09-13/hp_vf025.sh, designs/research_state/measurements/critic_ladder_reads/vf15_b_vs_ctrl10M_2026-09-12/hp_vf15_b.sh …
- **`ai_v12_16_ladder_ctrl10M_c`** — REFERENCED: loaded by 9 committed script(s): designs/research_state/measurements/critic_ladder_reads/ent05_vs_ctrl10M_2026-09-12/hp_ent05.sh, designs/research_state/measurements/critic_ladder_reads/strata_b_vs_ctrl10M_2026-09-12/hp_strata_b.sh, designs/research_state/measurements/critic_ladder_reads/vf025_vs_ctrl10M_2026-09-13/hp_vf025.sh, designs/research_state/measurements/critic_ladder_reads/vf15_b_vs_ctrl10M_2026-09-12/hp_vf15_b.sh …
- **`ai_v12_17_ladder_strata`** — REFERENCED: loaded by 6 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, designs/research_state/measurements/repr_class_decode_2026-09-11/run.sh, designs/research_state/measurements/repr_class_decode_2026-09-11/tabulate.py, designs/research_state/measurements/repr_class_decode_strata_b_2026-09-12/tabulate.py …
- **`ai_v12_19_ladder_lambda09`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, designs/research_state/measurements/search_dividend_winprob_heads_2026-09-11/run_battery.sh, designs/research_state/measurements/search_dividend_winprob_heads_2026-09-11/run_extension.sh, src/main/ops/team_conditioning.py
- **`ai_v12_20_ladder_denseaux`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, designs/research_state/measurements/search_dividend_winprob_heads_2026-09-11/run_battery2.sh, designs/research_state/measurements/search_dividend_winprob_heads_2026-09-11/run_battery3.sh
- **`ai_v12_21_ladder_lambda09_b`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py
- **`ai_v12_22_ladder_lambda095`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py
- **`ai_v12_23_ladder_rollout`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, designs/research_state/measurements/paired_refit_discrimination_2026-09-14/run_battery.sh
- **`ai_v12_24_ladder_strata_b`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_ladder_reads/strata_b_vs_ctrl10M_2026-09-12/hp_strata_b.sh, designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py, designs/research_state/measurements/repr_class_decode_strata_b_2026-09-12/run.sh, designs/research_state/measurements/repr_class_decode_strata_b_2026-09-12/tabulate.py
- **`ai_v12_25_ladder_vf15_b`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/critic_ladder_reads/vf15_b_vs_ctrl10M_2026-09-12/hp_vf15_b.sh, designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py
- **`ai_v12_26_ladder_ctrl10M_shaped`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/ladder_strength_table_2026-09-12/read_ladder_strength.py
- **`ai_v12_28_ladder_ent05`** — REFERENCED: loaded by 5 committed script(s): designs/research_state/measurements/critic_ladder_reads/ent05_vs_ctrl10M_2026-09-12/hp_ent05.sh, designs/research_state/measurements/flywheel_armS_reads_2026-09-14/scripts/entropy_read.py, designs/research_state/measurements/flywheel_pair_read_2026-09-15/scripts/pair_entropy_read.py, designs/research_state/measurements/flywheel_wb_floor_read_2026-09-18/scripts/floor_entropy_read.py …
- **`ai_v12_29_ladder_vf025`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/critic_ladder_reads/vf025_vs_ctrl10M_2026-09-13/hp_vf025.sh, designs/research_state/measurements/paired_refit_discrimination_2026-09-14/run_battery.sh
- **`ai_v13_01_flywheel_shaped`** — REFERENCED: loaded by 26 committed script(s): designs/research_state/measurements/elo_calibration_external_anchors_2026-09-14/cells.py, designs/research_state/measurements/elo_calibration_external_anchors_2026-09-14/fit_joint.py, designs/research_state/measurements/flywheel_armS_reads_2026-09-14/scripts/critic_levels.py, designs/research_state/measurements/flywheel_armS_reads_2026-09-14/scripts/entropy_read.py …
- **`ai_v13_02_flywheel_winprob`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 55 committed script(s): designs/research_state/measurements/anchor_ab_continuation_2026-09-20/scripts/run_lane.sh, designs/research_state/measurements/anchors_rust_frontend_2026-09-20/scripts/run_cell.sh, designs/research_state/measurements/exploiter_discrimination_2026-09-18/run_battery.sh, designs/research_state/measurements/exploiter_discrimination_2026-09-18/run_forks.sh …
- **`ai_v13_03_fork`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/fork_arm_read_2026-09-16/run_battery.sh, designs/research_state/measurements/fork_arm_read_2026-09-16/run_forks.sh, designs/research_state/measurements/fork_arm_read_2026-09-16/run_guard.sh
- **`ai_v13_04_flywheel_winprob_b`** — REFERENCED: loaded by 17 committed script(s): designs/research_state/measurements/flywheel_wb_floor_read_2026-09-18/scripts/floor_away_anchor.py, designs/research_state/measurements/flywheel_wb_floor_read_2026-09-18/scripts/floor_critic_levels.py, designs/research_state/measurements/flywheel_wb_floor_read_2026-09-18/scripts/floor_entropy_read.py, designs/research_state/measurements/flywheel_wb_floor_read_2026-09-18/scripts/floor_eplen_read.py …
- **`ai_v13_05_exploit_big5starmie`** — REFERENCED: loaded by 11 committed script(s): designs/research_state/measurements/exploiter_discrimination_2026-09-18/gen_tree.py, designs/research_state/measurements/exploiter_discrimination_2026-09-18/run_battery.sh, designs/research_state/measurements/exploiter_discrimination_2026-09-18/run_forks.sh, designs/research_state/measurements/exploiter_discrimination_2026-09-18/run_score.sh …
- **`ai_v13_06_exploit_ddtar_spikes`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 7 committed script(s): designs/research_state/measurements/fold1_cont_read_2026-09-19/scripts/cont_slice_read.py, designs/research_state/measurements/fold1_read_2026-09-19/scripts/fold_slice_read.py, designs/research_state/measurements/fold1_read_2026-09-19/scripts/harness/run_slice.sh, designs/research_state/measurements/split_lossoff_read_2026-09-20/scripts/split_slice_read.py …
- **`ai_v13_07_fold1`** — REFERENCED: loaded by 13 committed script(s): designs/research_state/measurements/fold1_cont_read_2026-09-19/scripts/cont_run_rows.py, designs/research_state/measurements/fold1_cont_read_2026-09-19/scripts/cont_slice_read.py, designs/research_state/measurements/fold1_cont_read_2026-09-19/scripts/cont_untaught_delta.py, designs/research_state/measurements/fold1_cont_read_2026-09-19/scripts/harness/run_slice.sh …
- **`ai_v13_08_fold1_cont`** — REFERENCED: loaded by 10 committed script(s): designs/research_state/measurements/anchor_ab_continuation_2026-09-20/scripts/analyze.py, designs/research_state/measurements/anchor_ab_continuation_2026-09-20/scripts/run_lane.sh, designs/research_state/measurements/fold1_cont_read_2026-09-19/scripts/cont_run_rows.py, designs/research_state/measurements/fold1_cont_read_2026-09-19/scripts/cont_slice_read.py …
- **`ai_v13_09_wcont`** — REFERENCED: loaded by 12 committed script(s): designs/research_state/measurements/anchor_ab_continuation_2026-09-20/scripts/analyze.py, designs/research_state/measurements/anchor_ab_continuation_2026-09-20/scripts/run_lane.sh, designs/research_state/measurements/fold1_cont_read_2026-09-19/scripts/cont_slice_read.py, designs/research_state/measurements/split_lossoff_read_2026-09-20/scripts/render_tables.py …
- **`ai_v13_10_exploit_stall`** — REFERENCED: loaded by 2 committed script(s): src/main/best_response_gap.py, src/main/best_response_gap_integration_test.py
- **`ai_v13_11_split_lossoff`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 5 committed script(s): designs/research_state/measurements/split_lossoff_read_2026-09-20/scripts/harness/run_slice.sh, designs/research_state/measurements/split_lossoff_read_2026-09-20/scripts/harness/run_untaught.sh, designs/research_state/measurements/split_lossoff_read_2026-09-20/scripts/render_tables.py, designs/research_state/measurements/split_lossoff_read_2026-09-20/scripts/split_slice_read.py …
- **`ai_v13_12_plateau`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 21 committed script(s): designs/research_state/measurements/hidose_teacher_2026-09-21/launch.sh, designs/research_state/measurements/hidose_teacher_2026-09-21/scripts/harness/run_admission_hidose.sh, designs/research_state/measurements/hidose_teacher_2026-09-21/scripts/harness/run_untaught_hidose.sh, designs/research_state/measurements/hidose_teacher_2026-09-21/scripts/hidose_delta.py …
- **`ai_v13_13_exploit5_offense`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 12 committed script(s): designs/research_state/measurements/hidose_teacher_2026-09-21/launch.sh, designs/research_state/measurements/population_loop_r1_2026-09-23/launch_B.sh, designs/research_state/measurements/population_loop_r1_2026-09-23/launch_C.sh, designs/research_state/measurements/population_loop_r1_2026-09-23/read/scripts/manipulation_check.py …
- **`ai_v13_14_exploit5_balance`** — REFERENCED: loaded by 3 committed script(s): src/main/best_response_gap.py, src/main/best_response_gap_integration_test.py, src/main/train/fork_lr_inherit_guard_test.py
- **`ai_v13_15_exploit5_stall`** — REFERENCED: loaded by 3 committed script(s): src/main/best_response_gap.py, src/main/best_response_gap_integration_test.py, src/main/train/fork_lr_inherit_guard_test.py
- **`ai_v13_16_teach5_offense_dist`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 5 committed script(s): designs/research_state/measurements/population_loop_r1_2026-09-23/launch_RB.sh, designs/research_state/measurements/population_loop_r1_2026-09-23/launch_RC.sh, designs/research_state/measurements/population_loop_r1_2026-09-23/scripts/build_argvs.py, designs/research_state/measurements/population_loop_r1_2026-09-23/scripts/resolve_opponents.py …
- **`ai_v13_17_fold_k1`** — REFERENCED: named in the ledger's last 1000 lines
- **`ai_v13_18_fold_k3`** — REFERENCED: named in the ledger's last 1000 lines
- **`ai_v13_18_teach5_offense_hidose`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 20 committed script(s): designs/research_state/measurements/hidose_teacher_2026-09-21/launch.sh, designs/research_state/measurements/hidose_teacher_2026-09-21/scripts/harness/run_admission_hidose.sh, designs/research_state/measurements/hidose_teacher_2026-09-21/scripts/harness/run_untaught_hidose.sh, designs/research_state/measurements/hidose_teacher_2026-09-21/scripts/hidose_delta.py …
- **`ai_v13_19_fold_k11`** — REFERENCED: named in the ledger's last 1000 lines
- **`ai_v13_20_fold_k11_sharematched`** — REFERENCED: named in the ledger's last 1000 lines
- **`ai_v13_21_wcont_b`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 3 committed script(s): designs/research_state/measurements/rust_core_phase0_2026-09-23/opponent_inference_bench.py, designs/research_state/measurements/wcont_b_seed1002_read_2026-09-23/scripts/harness/run_away_cell.sh, designs/research_state/measurements/wcont_b_seed1002_read_2026-09-23/scripts/harness/run_untaught.sh; training output written within 7 days (tb/, 2026-09-23)
- **`ai_v13_22_popr1_loop`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 16 committed script(s): designs/research_state/measurements/belief_calibration_2026-09-24/scripts/prep_five_arm.py, designs/research_state/measurements/belief_calibration_2026-09-24/scripts/run_arm.py, designs/research_state/measurements/belief_winrate_ab_2026-09-24/scripts/run_ab.py, designs/research_state/measurements/population_loop_r1_2026-09-23/launch_B.sh …; training output written within 7 days (tb/, 2026-09-23)
- **`ai_v13_23_popr1_ctrl`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 13 committed script(s): designs/research_state/measurements/population_loop_r1_2026-09-23/launch_C.sh, designs/research_state/measurements/population_loop_r1_2026-09-23/launch_RC.sh, designs/research_state/measurements/population_loop_r1_2026-09-23/read/scripts/ga_aggregate.py, designs/research_state/measurements/population_loop_r1_2026-09-23/read/scripts/ga_driver.sh …; training output written within 7 days (tb/, 2026-09-24)
- **`ai_v13_24_popr1_read_loop`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 13 committed script(s): designs/research_state/measurements/population_loop_r1_2026-09-23/launch_RB.sh, designs/research_state/measurements/population_loop_r1_2026-09-23/read/scripts/primary_rule.py, designs/research_state/measurements/population_loop_r1_2026-09-23/scripts/build_argvs.py, designs/research_state/measurements/population_loop_r2_2026-09-24/generalists/launch_B2.sh …; training output written within 7 days (tb/, 2026-09-24)
- **`ai_v13_25_popr1_read_ctrl`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 7 committed script(s): designs/research_state/measurements/population_loop_r1_2026-09-23/launch_RC.sh, designs/research_state/measurements/population_loop_r1_2026-09-23/read/scripts/primary_rule.py, designs/research_state/measurements/population_loop_r1_2026-09-23/scripts/build_argvs.py, designs/research_state/measurements/population_loop_r2_2026-09-24/launch_RCX.sh …; training output written within 7 days (tb/, 2026-09-24)
- **`ai_v13_26_popr0_exploit5_offense_s1002`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 3 committed script(s): designs/research_state/measurements/population_loop_r1_2026-09-23/launch_A2.sh, designs/research_state/measurements/population_loop_r1_2026-09-23/read/scripts/primary_rule.py, designs/research_state/measurements/population_loop_r1_2026-09-23/scripts/build_argvs.py; training output written within 7 days (tb/, 2026-09-24)
- **`ai_v13_27_popr2_loop`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 8 committed script(s): designs/research_state/measurements/population_loop_r2_2026-09-24/generalists/launch_B2.sh, designs/research_state/measurements/population_loop_r2_2026-09-24/launch_RB2.sh, designs/research_state/measurements/population_loop_r2_2026-09-24/read/scripts/ga_aggregate.py, designs/research_state/measurements/population_loop_r2_2026-09-24/read/scripts/ga_driver.sh …; training output written within 7 days (tb/, 2026-09-24)
- **`ai_v13_28_popr2_ctrl`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 8 committed script(s): designs/research_state/measurements/population_loop_r2_2026-09-24/generalists/launch_C2.sh, designs/research_state/measurements/population_loop_r2_2026-09-24/launch_RC2.sh, designs/research_state/measurements/population_loop_r2_2026-09-24/read/scripts/ga_aggregate.py, designs/research_state/measurements/population_loop_r2_2026-09-24/read/scripts/ga_driver.sh …; training output written within 7 days (tb/, 2026-09-25)
- **`ai_v13_29_popr2_read_loop`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 4 committed script(s): designs/research_state/measurements/population_loop_r2_2026-09-24/launch_RB2.sh, designs/research_state/measurements/population_loop_r2_2026-09-24/read/scripts/primary_rule.py, designs/research_state/measurements/population_loop_r2_2026-09-24/read/scripts/stoplist_check.py, designs/research_state/measurements/population_loop_r2_2026-09-24/scripts/build_argvs.py; training output written within 7 days (tb/, 2026-09-25)
- **`ai_v13_30_popr2_read_ctrl`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 4 committed script(s): designs/research_state/measurements/population_loop_r2_2026-09-24/launch_RC2.sh, designs/research_state/measurements/population_loop_r2_2026-09-24/read/scripts/primary_rule.py, designs/research_state/measurements/population_loop_r2_2026-09-24/read/scripts/stoplist_check.py, designs/research_state/measurements/population_loop_r2_2026-09-24/scripts/build_argvs.py; training output written within 7 days (tb/, 2026-09-25)
- **`ai_v13_31_popr1_read_loop_ext`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 4 committed script(s): designs/research_state/measurements/population_loop_r2_2026-09-24/launch_RBX.sh, designs/research_state/measurements/population_loop_r2_2026-09-24/read/scripts/side_check.sh, designs/research_state/measurements/population_loop_r2_2026-09-24/scripts/build_argvs.py, designs/research_state/measurements/population_loop_r2_2026-09-24/scripts/convergence_rule.py; training output written within 7 days (tb/, 2026-09-25)
- **`ai_v13_32_popr1_read_ctrl_ext`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 4 committed script(s): designs/research_state/measurements/population_loop_r2_2026-09-24/launch_RCX.sh, designs/research_state/measurements/population_loop_r2_2026-09-24/read/scripts/side_check.sh, designs/research_state/measurements/population_loop_r2_2026-09-24/scripts/build_argvs.py, designs/research_state/measurements/population_loop_r2_2026-09-24/scripts/convergence_rule.py; training output written within 7 days (tb/, 2026-09-25)
- **`ai_v13_33_core_burnin`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 1 committed script(s): designs/research_state/measurements/new_lineage_2026-09-26/launch_base.sh; training output written within 7 days (tb/, 2026-09-26)
- **`ai_v14_01_base`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 22 committed script(s): designs/research_state/measurements/kakuna_goal_read_2026-09-28/scripts/kq.py, designs/research_state/measurements/learner_battery_2026-09-26/launch_arm.sh, designs/research_state/measurements/learner_battery_2026-09-26/scripts/build_argvs.py, designs/research_state/measurements/learner_battery_2026-09-26/scripts/validate_all.sh …; training output written within 7 days (tb/, 2026-09-27)
- **`ai_v14_02_lbat_ctrl`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 7 committed script(s): designs/research_state/measurements/learner_battery_2026-09-26/launch_arm.sh, designs/research_state/measurements/learner_battery_2026-09-26/scripts/build_argvs.py, designs/research_state/measurements/learner_battery_2026-09-26/scripts/speed_read.py, designs/research_state/measurements/learner_battery_2026-09-26/scripts/validate_all.sh …; training output written within 7 days (tb/, 2026-09-28)
- **`ai_v14_03_lbat_e5`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 6 committed script(s): designs/research_state/measurements/learner_battery_2026-09-26/launch_arm.sh, designs/research_state/measurements/learner_battery_2026-09-26/scripts/build_argvs.py, designs/research_state/measurements/learner_battery_2026-09-26/scripts/speed_read.py, designs/research_state/measurements/learner_battery_2026-09-26/scripts/validate_all.sh …; training output written within 7 days (tb/, 2026-09-28)
- **`ai_v14_04b_lbat_t32`** — REFERENCED: named in the ledger's last 1000 lines; training output written within 7 days (tb/, 2026-09-28)
- **`ai_v14_05_lbat_l95`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 6 committed script(s): designs/research_state/measurements/learner_battery_2026-09-26/launch_arm.sh, designs/research_state/measurements/learner_battery_2026-09-26/scripts/build_argvs.py, designs/research_state/measurements/learner_battery_2026-09-26/scripts/speed_read.py, designs/research_state/measurements/learner_battery_2026-09-26/scripts/validate_all.sh …; training output written within 7 days (tb/, 2026-09-28)
- **`ai_v14_06_lbat_ctrl_fix`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 11 committed script(s): designs/research_state/measurements/m5_t2/probes/cpp/dump.py, designs/research_state/measurements/m5_t2/probes/probe_aot28_swap.py, designs/research_state/measurements/m5_t2/probes/probe_aot28_wi.py, designs/research_state/measurements/m5_t2/probes/probe_backends.py …; training output written within 7 days (tb/, 2026-09-28)
- **`ai_v14_07_g0p_k2`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 4 committed script(s): designs/research_state/measurements/new_lineage_2026-09-26/g0p_blocks_k2_k3/chain_k3.sh, designs/research_state/measurements/new_lineage_2026-09-26/round0_2026-09-29/launch_one.sh, src/main/policy_spectrum/bank.py, src/main/train/matchup_drift_header_test.py; training output written within 7 days (tb/, 2026-09-29)
- **`ai_v14_08_g0p_k3`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 2 committed script(s): designs/research_state/measurements/new_lineage_2026-09-26/g0p_blocks_k2_k3/chain_k3.sh, src/main/policy_spectrum/bank.py; training output written within 7 days (tb/, 2026-09-29)
- **`ai_v14_09_r0_offense_a`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 2 committed script(s): src/main/policy_spectrum/bank.py, src/main/train/matchup_drift_header_test.py; training output written within 7 days (tb/, 2026-09-29)
- **`ai_v5_11_tail2_53m_0611`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/archive_grooming_tiered_test.py, designs/research_state/measurements/archive_grooming_tiers.py
- **`ai_v5_12_bias_05_N_0612`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/archive_grooming_tiered_test.py, designs/research_state/measurements/archive_grooming_tiers.py
- **`ai_v5_5_popart_50m_0607`** — REFERENCED: loaded by 1 committed script(s): src/main/launcher_app_test.py
- **`ai_v6_13_outgoing_dmg_0620`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 3 committed script(s): src/main/launcher_test.py, src/main/run_name_test.py, src/main/train/parser/eval_subprocess.py
- **`ai_v6_13_outgoing_dmg_0620_exp_v1`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 1 committed script(s): src/main/launcher_test.py
- **`ai_v8_01_zarch_film_0717`** — REFERENCED: loaded by 5 committed script(s): designs/research_state/measurements/archive_grooming_tiers.py, designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/ent_coef_archive_survey.py, designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/entropy_forensics.py, src/agents/training/tb_inherit.py …; v8-era (the era replication + head-to-head load these)
- **`ai_v8_02_zarch_teampfsp_0718`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`ai_v8_03_zarch_control_0718`** — REFERENCED: loaded by 17 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/v8_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/v8_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/sharing_kernel/gen_states.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/v8_checkpoint_fix.py …; v8-era (the era replication + head-to-head load these)
- **`ai_v8_04_distill_4teacher_0722`** — REFERENCED: loaded by 18 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/v8_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/resolve_teachers.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/v8_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py …; v8-era (the era replication + head-to-head load these)
- **`ai_v8_05_semistall564_exploiter_0722`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`ai_v8_06_semistall_3team_exploiter_0722`** — REFERENCED: loaded by 6 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/v8_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/resolve_teachers.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/v8_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/v8_checkpoint_fix.py …; v8-era (the era replication + head-to-head load these)
- **`ai_v8_07_semistall564_scratch_0722`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`ai_v8_08_defensive_6team_exploiter_0723`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`ai_v8_09_pool10_exploiter_0723`** — REFERENCED: loaded by 6 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/v8_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/resolve_teachers.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/v8_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/v8_checkpoint_fix.py …; v8-era (the era replication + head-to-head load these)
- **`ai_v8_10_offense20_exploiter_0724`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`ai_v8_11_offense10_exploiter_0724`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`ai_v8_12_defensive20_exploiter_0724`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`ai_v8_13_defensive10_exploiter_0725`** — REFERENCED: loaded by 6 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/v8_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/resolve_teachers.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/v8_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/v8_checkpoint_fix.py …; v8-era (the era replication + head-to-head load these)
- **`ai_v8_14_distill3_0725`** — REFERENCED: named in the ledger's last 1000 lines; loaded by 11 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/v8_era_locality_v2.py, designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/entropy_forensics.py, designs/research_state/measurements/folding_history_probe.py …; v8-era (the era replication + head-to-head load these)
- **`ai_v8_15_retention_A_frozen_0726`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`ai_v8_16_def20_lut_0726`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`ai_v8_17_rand20_nolut_0726`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`ai_v8_18_rand20_lut_0726`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`ai_v8_19_def20_lut_zeroinit_0727`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`ai_v8_20_rand10_nolut_0727`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`ai_v9_100_R5F08_0831`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/lineage_check.py
- **`ai_v9_102_R5F10_0831`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/lineage_check.py
- **`ai_v9_104_R5F12_0831`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/lineage_check.py
- **`ai_v9_106_R5F14_0831`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/lineage_check.py
- **`ai_v9_10_gen9_intent_distcritic_0813`** — REFERENCED: loaded by 1 committed script(s): src/agents/model/intent_move_cell_test.py
- **`ai_v9_120_R5FUND00_0901`** — REFERENCED: loaded by 7 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/inspect_eval.py …
- **`ai_v9_122_R5FUND02_0901`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/lineage_check.py
- **`ai_v9_124_R5FUND04_0901`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/lineage_check.py
- **`ai_v9_126_R5FUND06_0901`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/lineage_check.py
- **`ai_v9_128_R5FUND08_0901`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/lineage_check.py
- **`ai_v9_130_R5FUND10_0901`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/lineage_check.py
- **`ai_v9_132_R5FUND12_0901`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/lineage_check.py
- **`ai_v9_134_R5FUND14_0901`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/lineage_check.py
- **`ai_v9_140_B2_0901`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/offline_collateral_kl.py
- **`ai_v9_141_C1_0901`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/offline_collateral_kl.py
- **`ai_v9_142_N1_0901`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/teacher_content_2x2_2026-09-04/tc_readout.py
- **`ai_v9_143_N2_0901`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/teacher_content_2x2_2026-09-04/tc_readout.py
- **`ai_v9_14_gen12_h_entitypool_shaping_0816`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/obs_conditioning_probe.py
- **`ai_v9_150_R4DOSE12_0901`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/offline_collateral_kl.py
- **`ai_v9_151_R4DOSE6_0901`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/offline_collateral_kl.py
- **`ai_v9_152_R4DOSE3_0901`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/offline_collateral_kl.py
- **`ai_v9_15_gen13_hb_events_stack_0817`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/gen13_stall_coverage.py, designs/research_state/measurements/gen14_paired_bt_refit.py, designs/research_state/measurements/obs_conditioning_probe.py
- **`ai_v9_160_TCFUNDA_0903`** — REFERENCED: loaded by 9 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_competence/inventory.py, designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/sharing_kernel/control_funded_vs_unfunded.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py …
- **`ai_v9_161_TCFUNDB_0903`** — REFERENCED: loaded by 7 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/sharing_kernel/control_funded_vs_unfunded.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/teacher_content_2x2_2026-09-04/k6_readout.py …
- **`ai_v9_162_TCUNFA_0903`** — REFERENCED: loaded by 13 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_competence/inventory.py, designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/sharing_kernel/control_funded_vs_unfunded.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py …
- **`ai_v9_163_TCUNFB_0903`** — REFERENCED: loaded by 8 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/fold_displacement/deltas.py, designs/research_state/measurements/arch_transfer_2026-09-05/sharing_kernel/control_funded_vs_unfunded.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/teacher_content_2x2_2026-09-04/k6_readout.py …
- **`ai_v9_16_gen14_framedel_v91_0817`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/gen13_stall_coverage.py, designs/research_state/measurements/gen14_paired_bt_refit.py, designs/research_state/measurements/obs_conditioning_probe.py
- **`ai_v9_170_TCUNFK6A_0904`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/teacher_content_2x2_2026-09-04/k6_readout.py, designs/research_state/measurements/teacher_content_2x2_2026-09-04/taught_readout.py
- **`ai_v9_171_TCUNFK6B_0904`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/teacher_content_2x2_2026-09-04/k6_readout.py, designs/research_state/measurements/teacher_content_2x2_2026-09-04/taught_readout.py
- **`ai_v9_17_tdaux_lam3_0818`** — REFERENCED: loaded by 1 committed script(s): src/agents/training/poke_env_gaps/faint_attribution_fuzz_test.py
- **`ai_v9_18_gen15_v8rewards_0818`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/obs_conditioning_probe.py, src/main/prober/loops.py
- **`ai_v9_195_G5PLAINA_0906`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py, src/main/untaught_meter.py
- **`ai_v9_196_G5PLAINB_0906`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py, src/main/untaught_meter.py
- **`ai_v9_197_G5PLAINC_0906`** — REFERENCED: loaded by 2 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py, src/main/untaught_meter.py
- **`ai_v9_21_gen17_pfspoff_0820`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/folding_history_probe.py, designs/research_state/measurements/obs_conditioning_probe.py, src/agents/model/audit_states_test.py, src/main/baselines.py
- **`ai_v9_25_E4_baitbot_0822`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/maturity_harm_trend.py
- **`ai_v9_26_baitent_probe_0823`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/archive_grooming_tiers.py
- **`ai_v9_27_extremedial_probe_0823`** — REFERENCED: loaded by 2 committed script(s): src/agents/training/exploiter_ladder.py, src/agents/training/exploiter_ladder_test.py
- **`ai_v9_29_rev1_0823`** — REFERENCED: loaded by 83 committed script(s): designs/ai_v12/team_slate_build.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/combine_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py …
- **`ai_v9_31_tock1_k4_0824`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/teacher_sharpness_probe.py
- **`ai_v9_34_tick1_0824`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/folding_history_probe.py
- **`ai_v9_36_tock1c_q6_0824`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/teacher_sharpness_probe.py
- **`ai_v9_37_tick1_dosext_0825`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/folding_history_probe.py
- **`ai_v9_44_tock2_v8shape_0825`** — REFERENCED: a committed file / the ledger names a file the plan would delete
- **`ai_v9_45_fdF_p1_0826`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/archive_grooming_tiers.py
- **`ai_v9_48_G1_action_0826`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/archive_grooming_tiers.py
- **`ai_v9_51_fdF_p2c_0826`** — REFERENCED: named in the ledger's last 1000 lines
- **`ai_v9_53_R2F5a_0826`** — REFERENCED: loaded by 8 committed script(s): designs/research_state/measurements/distillability_index_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/per_team_gradient_geometry_2026-08-28/probeF_acid.py …
- **`ai_v9_54_R2F5b_0826`** — REFERENCED: loaded by 7 committed script(s): designs/research_state/measurements/distillability_index_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/per_team_gradient_geometry_2026-08-28/probeF_build_states.py …
- **`ai_v9_55_R2F5c_0826`** — REFERENCED: loaded by 6 committed script(s): designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/per_team_gradient_geometry_2026-08-28/probeF_build_states.py, designs/research_state/measurements/per_team_gradient_geometry_2026-08-28/probeF_grads.py …
- **`ai_v9_56_R2F5d_0826`** — REFERENCED: loaded by 6 committed script(s): designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/per_team_gradient_geometry_2026-08-28/probeF_build_states.py, designs/research_state/measurements/per_team_gradient_geometry_2026-08-28/probeF_grads.py …
- **`ai_v9_57_R2F5e_0826`** — REFERENCED: loaded by 6 committed script(s): designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/per_team_gradient_geometry_2026-08-28/probeF_build_states.py, designs/research_state/measurements/per_team_gradient_geometry_2026-08-28/probeF_grads.py …
- **`ai_v9_58_R2CTRL_0827`** — REFERENCED: loaded by 8 committed script(s): designs/ai_v12/team_slate_build.py, designs/research_state/measurements/folding_history_probe.py, designs/research_state/measurements/plain_training_robbery.py, designs/research_state/measurements/representational_richness_transfer_forward.py …
- **`ai_v9_59_R2ACTION_0827`** — REFERENCED: loaded by 46 committed script(s): designs/ai_v12/team_slate_build.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/combine_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py …
- **`ai_v9_62_R2PLAIN_0827`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/folding_history_probe.py, designs/research_state/measurements/plain_training_robbery.py, designs/research_state/measurements/representational_richness_transfer_forward.py, designs/research_state/measurements/representational_richness_transfer_locus.py
- **`ai_v9_63_R3F6a_0828`** — REFERENCED: loaded by 6 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/folding_history_probe.py …
- **`ai_v9_64_R3F6b_0828`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/rev3_untaught_pulldown.py
- **`ai_v9_65_R3F6c_0828`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/rev3_untaught_pulldown.py
- **`ai_v9_66_R3F6d_0828`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/rev3_untaught_pulldown.py
- **`ai_v9_67_R3F6e_0828`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/rev3_untaught_pulldown.py
- **`ai_v9_68_R3F6f_0828`** — REFERENCED: loaded by 5 committed script(s): designs/research_state/measurements/critic_as_transfer_vehicle_probe.py, designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/folding_history_probe.py …
- **`ai_v9_69_R3F6CURR_0828`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/rev3_untaught_pulldown.py
- **`ai_v9_70_R3ACTION_0828`** — REFERENCED: loaded by 13 committed script(s): designs/ai_v12/team_slate_build.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/resolve_sets.py, designs/research_state/measurements/axis_split_taught_untaught.py …
- **`ai_v9_71_R3ACTIONHI_0828`** — REFERENCED: loaded by 1 committed script(s): designs/ai_v12/team_slate_build.py
- **`ai_v9_72_R3SELF_0828`** — REFERENCED: loaded by 5 committed script(s): designs/ai_v12/team_slate_build.py, designs/research_state/measurements/folding_history_probe.py, designs/research_state/measurements/plain_training_robbery.py, designs/research_state/measurements/starmie_ood_control_traces.py …
- **`ai_v9_73_R4S3a_0829`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/lr_licensing_probe.py
- **`ai_v9_74_R4S3b_0829`** — REFERENCED: loaded by 3 committed script(s): designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/lr_licensing_probe.py
- **`ai_v9_75_R4S3c_0829`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/exploitability_taught_untaught.py, designs/research_state/measurements/exploiter_fingerprint_probe.py, designs/research_state/measurements/lr_licensing_probe.py, designs/research_state/measurements/starmie_ood_control_traces.py
- **`ai_v9_76_R4ACTION_0830`** — REFERENCED: loaded by 9 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/resolve_sets.py, designs/research_state/measurements/axis_split_taught_untaught.py, designs/research_state/measurements/folding_history_probe.py …
- **`ai_v9_77_G1LEAN_0830`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/starmie_ood_control_traces.py
- **`ai_v9_82_REFOLD1_0830`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/axis_split_taught_untaught.py
- **`ai_v9_91_COMPFOLD_0831`** — REFERENCED: loaded by 5 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/teacher_distance/fold_table.py, designs/research_state/measurements/axis_split_taught_untaught.py, designs/research_state/measurements/obs_conditioning_probe.py, designs/research_state/measurements/representational_richness_transfer_forward.py …
- **`ai_v9_92_R5F00_0831`** — REFERENCED: loaded by 11 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/inspect_eval.py …
- **`ai_v9_94_R5F02_0831`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/lineage_check.py
- **`ai_v9_96_R5F04_0831`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/lineage_check.py
- **`ai_v9_98_R5F06_0831`** — REFERENCED: loaded by 4 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/content_locality/gen_era_locality.py, designs/research_state/measurements/arch_transfer_2026-09-05/content_locality_v2/gen_era_locality_v2.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/drift.py, designs/research_state/measurements/arch_transfer_2026-09-05/exploiter_drift/lineage_check.py
- **`run_20260830_180409`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/archive_grooming_tiered_test.py
- **`run_20260906_083317`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/archive_grooming_tiered_test.py
- **`v8rep_p1_A_0905`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/archive_grooming_tiered_test.py; v8-era (the era replication + head-to-head load these)
- **`v8rep_p1_B_0905`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`v8rep_p1_C_0905`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`v8rep_p2loss_A_0905`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`v8rep_p2loss_B_0905`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`v8rep_p2loss_C_0905`** — REFERENCED: v8-era (the era replication + head-to-head load these)
- **`v8rep_p2self_A_0905`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py; v8-era (the era replication + head-to-head load these)
- **`v8rep_p2self_B_0905`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py; v8-era (the era replication + head-to-head load these)
- **`v8rep_p2self_C_0905`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/arch_transfer_2026-09-05/continuation_drift/drift.py; v8-era (the era replication + head-to-head load these)
- **`warmstart_generic_0715`** — REFERENCED: loaded by 1 committed script(s): designs/research_state/measurements/archive_grooming_tiered_test.py

## What would be KEPT, and why (every CLOSED run with a plan)

<details><summary><code>ai_v12_04_pfsp_fork25M</code> — 0.071 GB freed, 2 entries deleted</summary>

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

</details>

<details><summary><code>ai_v12_27_ladder_ctrl10M_shaped_dense</code> — 0.303 GB freed, 8 entries deleted</summary>

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

</details>

<details><summary><code>ai_v5_10_tail1_23_0611</code> — 0.059 GB freed, 4 entries deleted</summary>

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

<details><summary><code>ai_v5_2_native_selfplay_50m_0606</code> — 0.057 GB freed, 4 entries deleted</summary>

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

</details>

<details><summary><code>ai_v5_3_vf_coef_clip_50m_0606</code> — 0.141 GB freed, 10 entries deleted</summary>

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

</details>

<details><summary><code>ai_v5_4_pbrs_opp_threat_50m_0607</code> — 0.113 GB freed, 8 entries deleted</summary>

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

<details><summary><code>ai_v5_7_switch_bias_41m_0609</code> — 0.142 GB freed, 10 entries deleted</summary>

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

<details><summary><code>ai_v6_01_belief_53m_0613</code> — 0.184 GB freed, 12 entries deleted</summary>

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

</details>

<details><summary><code>ai_v6_02_belief_lat_16m_0614</code> — 0.031 GB freed, 2 entries deleted</summary>

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

</details>

<details><summary><code>ai_v6_03_win_pred_N_0614</code> — 0.225 GB freed, 14 entries deleted</summary>

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

</details>

<details><summary><code>ai_v6_04_unified_all_half_batch_N_0616</code> — 0.035 GB freed, 2 entries deleted</summary>

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

</details>

<details><summary><code>ai_v6_04_unified_inc_N_0615</code> — 0.134 GB freed, 8 entries deleted</summary>

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

</details>

<details><summary><code>ai_v6_06_unified_all_N_0616</code> — 0.069 GB freed, 4 entries deleted</summary>

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

</details>

<details><summary><code>ai_v6_07_unified_topk_N_0616</code> — 0.114 GB freed, 5 entries deleted</summary>

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

</details>

<details><summary><code>ai_v6_08_unmasked_floor_N_0617</code> — 0.038 GB freed, 2 entries deleted</summary>

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

</details>

<details><summary><code>ai_v6_09_dmg_reattend_N_0617</code> — 0.13 GB freed, 6 entries deleted</summary>

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

</details>

<details><summary><code>ai_v6_11_typed_hp_0619</code> — 0.175 GB freed, 8 entries deleted</summary>

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

</details>

<details><summary><code>ai_v6_11_unified_obs_fixed_0618</code> — 0.13 GB freed, 6 entries deleted</summary>

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

<details><summary><code>ai_v7_01_teacher_0626</code> — 0.13 GB freed, 6 entries deleted</summary>

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

<details><summary><code>ai_v7_03_belief_shape_0630</code> — 0.217 GB freed, 10 entries deleted</summary>

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

</details>

<details><summary><code>ai_v7_04_opd_selfdistill_0702</code> — 0.478 GB freed, 22 entries deleted</summary>

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

<details><summary><code>ai_v7_19_combined_0716</code> — 0.043 GB freed, 2 entries deleted</summary>

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

</details>

<details><summary><code>ai_v9_09_gen8_beliefs_threat_inject_0811</code> — 1.043 GB freed, 22 entries deleted</summary>

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

<details><summary><code>ai_v9_12_gen10_t0prior_0814</code> — 1.013 GB freed, 24 entries deleted</summary>

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

</details>

<details><summary><code>ai_v9_13_gen11_labelonly_winprob_0815</code> — 0.994 GB freed, 23 entries deleted</summary>

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

</details>

<details><summary><code>ai_v9_172_G1SHORT_0905</code> — 0.255 GB freed, 13 entries deleted</summary>

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

</details>

<details><summary><code>ai_v9_17_tdaux_control_0818</code> — 0.042 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_27467520_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_27467520_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_17_tdaux_lam1_0818</code> — 0.042 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_27467520_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_27467520_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_26000016/snapshot.zip`

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

<details><summary><code>ai_v9_20_tdaux_rung2_lam00_0820</code> — 0.036 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_27467520_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_27467520_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_20_tdaux_rung2_lam10_0820</code> — 0.036 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_27467520_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_27467520_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_26000016/snapshot.zip`

</details>

<details><summary><code>ai_v9_20_tdaux_rung2_lam30_0820</code> — 0.036 GB freed, 1 entries deleted</summary>

**KEEP**

- `checkpoints/checkpoint_27467520_steps.json` — first, last, every-10th
- `checkpoints/checkpoint_27467520_steps.zip` — first, last, every-10th
- `best_model/`, `tb/`, `snapshot_ladder/`, `cf_*`, `elo/`, `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl` — never candidates
- the 3 most-recent `eval_traces/step_*` (+ `snapshot.zip` on the newest 1) — `prober.groom` retention

**DELETE**

- `eval_traces/step_26000016/snapshot.zip`

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

<details><summary><code>ai_v9_23_E2_substrate_on_0822</code> — 0.36 GB freed, 7 entries deleted</summary>

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

</details>

<details><summary><code>ai_v9_24_E3_substrate_on_0822</code> — 0.377 GB freed, 7 entries deleted</summary>

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

<details><summary><code>ai_v9_38_fdA_coef03_0825</code> — 0.656 GB freed, 35 entries deleted</summary>

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

<details><summary><code>ai_v9_39_fdB_lossonly_0825</code> — 0.656 GB freed, 35 entries deleted</summary>

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

<details><summary><code>ai_v9_40_fdC_ecology_0825</code> — 0.656 GB freed, 35 entries deleted</summary>

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

<details><summary><code>ai_v9_42_fdE_single_0825</code> — 0.656 GB freed, 35 entries deleted</summary>

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

<details><summary><code>ai_v9_49_G2_advgate_0826</code> — 0.656 GB freed, 35 entries deleted</summary>

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

<details><summary><code>ai_v9_52_G1p_matched_0826</code> — 0.656 GB freed, 35 entries deleted</summary>

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

<details><summary><code>ai_v9_60_R2TOPK_0827</code> — 0.656 GB freed, 35 entries deleted</summary>

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

<details><summary><code>ai_v9_61_R2KL_0827</code> — 0.656 GB freed, 35 entries deleted</summary>

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

## To actually apply this

```bash
export PYTHONPATH=$PYTHONPATH:src && \
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
  designs/research_state/measurements/archive_grooming_dryrun.py --apply
```

**Nothing was deleted in this pass — this was a dry run, and it wrote only the two report files.**
