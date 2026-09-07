# Effective rank — v8 line vs the live win-prob-critic arm

Probe (a) only (representation effective rank). Estimators are a **verbatim copy of the current
main checkout's `src/agents/model/capacity_probes.py` + `audit_states.py`** (`current_battery/`
in this dir), so all three columns come from one estimator. `battery_version` 1 on all three.
`agents/training/rank_metrics.effective_rank` is **byte-identical** at both v8 commits and on
main (diffed; only a blank line), so `rank_summary` is the same function everywhere.

| ref | checkpoint | step | commit / worktree | config_v | arch_signature | obs_dim | n_states |
|---|---|---|---|---|---|---|---|
| v8_line (the fold) | `ai_v8_14_distill3_0725/final_model_interrupted.zip` | 292,623,779 | `b13b30b289c5eaba136a930a4ab63451e209fbe5` | 45 | `gen3_opp_hp_typed_candidates_v1` | 2992 | 3000 |
| v8_parent (fold parent) | `ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip` | 277,583,267 | `ce8bbc954b42fceac0df773eab4040723bd2ffd3` | 45 | `gen3_opp_hp_typed_candidates_v1` | 2992 | 3000 |
| arm (win-prob critic) | `ai_v12_02_winprob_critic/final_model_interrupted.zip` | 20,054,016 | main `ae8e9fa9` | 110 | `gen3_critic_route_wave_v1` | 2501 | 3000 |

## TRAINED (fresh, same-config random init, in parentheses)

| tap | dim | v8_line PR (fresh) | v8_parent PR (fresh) | arm PR (fresh) |
|---|---|---|---|---|
| role_tokens  | 128 | 18.24 (14.36) | 18.73 (14.11) | 15.78 (17.38) |
| team_tokens  | 128 | 30.47 (6.30)  | 27.10 (6.17)  | 11.92 (6.54)  |
| value_pooled | 128 | **4.65** (10.92) | **4.82** (11.00) | **4.89** (4.28) |
| pi_features  | 512 | 40.27 (58.57) | 42.91 (52.64) | 11.01 (8.60) |
| vf_features  | 512 | **4.44** (57.85) | **4.36** (55.93) | **6.41** (5.20) |

### effrank / srank99 / n90 — trained arm

| tap | dim | v8_line effrank·sr99·n90 | v8_parent effrank·sr99·n90 | arm effrank·sr99·n90 |
|---|---|---|---|---|
| role_tokens  | 128 | 30.68 · 81 · 30 | 32.71 · 87 · 33 | 33.28 · 100 · 39 |
| team_tokens  | 128 | 51.65 · 110 · 55 | 49.75 · 110 · 55 | 31.57 · 107 · 47 |
| value_pooled | 128 | 9.08 · 47 · 10 | 10.20 · 51 · 12 | 10.32 · 54 · 14 |
| pi_features  | 512 | 99.34 · 390 · 144 | 107.91 · 403 · 157 | 29.09 · 270 · 51 |
| vf_features  | 512 | 14.53 · 205 · 35 | 15.77 · 218 · 42 | 15.78 · 139 · 23 |

`n_rows`: token taps 36,000 (`[N·12, D]`); the three state-level taps 3,000. Same on all three.

## Method — every deviation from the current battery

1. **Only probe (a) ran** on the v8 side. (b) trainability, (c) decodability and (d) the param
   census were not run: (b)/(c) need the era's obs layout through `Gen3ObservationEncoder.get_layout()`
   and are not what was asked. No tap was impossible to capture — all five hooks fired on both v8 refs.
2. **Tap names and capture path are unchanged.** The v8-era extractor already exposes
   `pokemon_encoder`, `team_transformer` (returning `(our, their)`, concatenated to 12 tokens),
   `last_value_pooled`, and a `(pi_features, vf_features)` forward return — so the current
   `capture_features` runs on it verbatim. No renaming, no substitution.
3. **`restore_identity_init` does not exist at either v8 commit.** `build_fresh_extractor` calls it
   via `getattr(..., None)`, so the v8 fresh arm is SB3 orthogonal init only. That is faithful:
   identity-at-init did not exist in this repo before 2026-08-01, so it *is* the network v8 training
   started from. The arm's fresh column does include the restore step. **The fresh columns are
   therefore not built by an identical procedure across the era wall.**
4. **State sampling is the current `collect_states` (`stratified_round_robin_v1`), seed 0,
   max 3000** — the same sampler and seed as the arm's artifact. Coverage differs by run, as it must
   (each run's own traces):
   - v8_line: 161 files read of 2055 matched, 8 step dirs × 13 opponents (incl. 3 `ext_*` exploiter sentinels)
   - v8_parent: 201 of 754, 4 step dirs × 10 opponents
   - arm: 207 of 2025, 10 step dirs × 14 opponents
5. **obs_dim differs (2992 vs 2501)** and the steps are three orders of magnitude apart
   (292M / 277M vs 20M). This is not a matched-step comparison and `battery_version` matching is
   the only one of the four "reading two artifacts" rules that holds.
6. `--out` paths were explicit; nothing was written under `models/`. Both worktrees were removed.

## The validity caveat (quoted, `designs/research_state/capacity_battery.md`)

> **A participation ratio treats every column as a MAGNITUDE. Before quoting one, check that every
> column in the matrix is consumed as a magnitude.**

The taps above are all post-Linear activations, so the specimen that minted this rule (raw-obs
columns carrying dex numbers) does not apply here. The doc's other three rules do: no kill/build
decision from any number above alone; a single generation's reading is uncalibrated; and every alarm
needs paired behavioural evidence (anchored ELO at matched snapshot COUNT plus a retention read).

Artifacts: `capacity_v8_line.json`, `capacity_v8_parent.json`, driver `run_v8_rank.py`,
estimators `current_battery/`.
