# Ladder refit audit — every committed `ladder.json` under `models/`, 2026-09-22

Landing the ladder RECIPE STAMP (`0f230405`) made **every** `snapshot_ladder/ladder.json` already
on disk read `recipe_status == "absent"`, so every cross-run reader refuses it. That is the
intended, loud outcome of the fix — but it left every run's headline rating unquotable. This
measurement says what each of those files is actually worth: **what the same nodes read when
refit by the current recipe**, which is the number `UNDERSTANDING.md` §7 rule 24 needs.

**Nothing was played and nothing under `models/` was written.** `fit_ladder(..., write=False)`
over each run's append-only `games.jsonl`. Reproduce:

```bash
export PYTHONPATH=$PYTHONPATH:src
nice -n 15 python3 designs/research_state/measurements/ladder_refit_audit_2026-09-22/refit_audit.py
```

One `<run>.json` per run lands beside this file (every node, committed vs refit), plus
`_all_runs.json` and `_ordering.json`.

---

## 1. Inventory

| | |
|---|---|
| runs under `models/` with a `snapshot_ladder/` | **93** |
| with a committed `ladder.json` | **93** (all) |
| carrying the current recipe stamp | **0** — every one reads `absent` |
| with a `games.jsonl`, i.e. refittable with `--fit-only`, playing nothing | **93** (all), 5,235 recorded pair rows |
| whose refit moves at least one node | **68** |
| whose refit reproduces the committed file to 0.0 | **25** |

**The 68/25 split is exactly the `3e6875a5` boundary, and it is clean.** All 68 movers record
`eval_sentinel_edges_dropped: null` (fitted before the sentinel-edge drop); all 25 that reproduce
to 0.0 record a non-null count (6 · 10 · 16 · 20 · 23 · 47 · 48 · 51). So on the files that exist
today the old count-key heuristic would in fact have classified every one correctly.

🚨 **That does not rehabilitate the count key, and rule 24 is right to have dropped it.** It is a
coincidence of this archive: no run on disk happens to have written a `0`, which is the value a
run that never measured a sentinel pair emits and which is indistinguishable from a pre-fix file.
The key also only ever encodes THIS recipe change — the next one is silent again. The recipe block
is what makes the next change loud.

## 2. What the refit moves

Over the 68 movers:

| | |
|---|---|
| median max \|Δ\| across a run's nodes | **53.4 Elo** |
| largest max \|Δ\| | **255.6** (`ai_v9_45_fdF_p1_0826`) |
| runs whose **newest** node moves by more than 10 Elo | **39** |
| newest-node delta sign | **65 of 68 NEGATIVE** — the committed file reads HIGH, as the fix predicted |
| the three positives | `ai_v9_51_fdF_p2c` +3.2, `ai_v9_71_R3ACTIONHI` +2.2, `ai_v9_162_TCUNFA` +0.5 — all inside noise |
| median newest-node delta | **−12.4 Elo** |
| the known case reproduces exactly | `ai_v12_02_winprob_critic` newest node **2057.3 → 1984.2 = −73.1**, matching the independently measured figure in `flywheel_armS_reads_2026-09-14/` §2.1 |

**The delta is not a uniform shift**, which is why the ordering has to be checked rather than
assumed away (§3): it scales with how much of a node's evidence came from eval sentinel edges, so
a densely-played node barely moves while a thinly-connected one moves by 100+.

🚨 **One run loses nodes outright.** `ai_v9_51_fdF_p2c_0826`'s committed 18M and 24M nodes are
**UNRATEABLE** under the current recipe — they carried no dense pair and no surviving bot edge, so
the committed fit rated them on sentinel edges alone. (Its committed 18M reads 1780.2, byte-equal
to its 4M node: the signature of a disconnected node.) Those two committed numbers are not
"inflated", they are unsupported.

**Four runs' committed `n_frozen_pairs_measured` exceeds what the refit can use**
(`ai_v12_02` 494→190, `ai_v9_25_E4_baitbot` 113→99, `ai_v9_26_baitent_probe` 54→45,
`ai_v9_27_extremedial_probe` 29→28). Their `games.jsonl` holds edges to snapshots the self-play
window has since groomed off disk; the committed file counted those rows while rating only the
survivors. The refit here is over the committed node set, so those edges are correctly excluded —
the same slice the live tooling takes.

## 3. Does the ordering change? — the number rule 24 needs

Ordering by each run's **newest node**, within the campaigns that were actually compared to each
other in the ledger:

| sibling group | n | order changes | pairs flipped |
|---|---:|:-:|---:|
| v9 generation ladder (gen1..gen17) | 18 | **YES** | **21 / 153** |
| fold-dial fd* (A/B/C/E/F, F-p1c, F-p2c) | 7 | **YES** | 1 / 21 |
| teacher-content TC* (09-03/09-04) | 6 | **YES** | 1 / 15 |
| tdaux rung2 (lam00/lam10/lam30) | 3 | **YES** | 1 / 3 |
| E-substrate + baitbot (E1..E4) | 4 | no | 0 / 6 |
| R2 rung (CTRL/ACTION/TOPK/KL/PLAIN) | 5 | no | 0 / 10 |
| R3/R4 + DOSE | 5 | no | 0 / 10 |
| 09-01 factorial (B2/C1/N2/G1SHORT) | 4 | no | 0 / 6 |
| G-gate (G1_action / G2_advgate / G1p_matched) | 3 | no | 0 / 3 |
| v12 ladder cells (10..29) | 17 | no | 0 / 136 |
| v13 flywheel family | 8 | no | 0 / 28 |
| winprob critic pair (v12_01 / v12_02) | 2 | no | 0 / 1 |
| **GLOBAL, all 93 runs** | 93 | — | **363 / 4,278 (8.5 %)** |

🚨 **The v9 generation ladder is the damaged one.** 21 of its 153 pairwise orderings reverse, and
several reverse a sign that was read as a result at the time:

| pair | committed Δ | refit Δ |
|---|---:|---:|
| `gen1_edges6` vs `gen7_seed_quantile` | −14.9 | **+10.7** |
| `gen13_hb_events_stack` vs `gen7_seed_quantile` | −5.9 | **+23.6** |
| `gen8_beliefs_threat_inject` vs `gen14_framedel` | +2.0 | **−17.2** |
| `gen3_k6_recency` vs `gen6_seed_vicreg` | −16.0 | **+1.3** |
| `gen4_rehome` vs `gen13_hb_events_stack` | +12.9 | **−2.9** |
| `gen25_consequence` vs `gen14_framedel` | +16.6 | **−3.5** |

Note what changes at the TOP of that group: committed reads `gen2_full11 > gen4_rehome >
gen11_labelonly > gen10_t0prior`; refit reads `gen2_full11 > gen13_hb_events_stack >
gen4_rehome > gen9_intent_distcritic`. `gen13` moves from 7th to 2nd.

**Within a run the curve shape changes too**: the argmax (best) node moves on **33 of 93** runs,
and **60 of 93** have at least one adjacent-node order swap. A "which snapshot was this run's
peak" read taken off a committed file is not safe either.

Globally, the biggest headline rank moves are `ai_v12_02_winprob_critic` (22 → 54 of 93),
`ai_v9_09_gen8` (38 → 60), `ai_v9_08_gen7` (14 → 34).

## 4. Per-run table — all 93 runs, ordered by max |Δ|

`🔁` = the run's best (argmax) node changes on refit. `adj. swaps` = adjacent node pairs that
reverse order. `count key` is the committed file's top-level `eval_sentinel_edges_dropped`.

| run | nodes | pairs (games.jsonl lines) | stamp | count key | max \|Δ\| | newest Δ | argmax node moves | adj. swaps |
|---|---:|---:|---|---|---:|---:|:-:|---:|
| `ai_v9_45_fdF_p1_0826` | 13 | 12 | absent | `null` | 255.6 | -2.3 | 🔁 | 6 |
| `ai_v9_26_baitent_probe_0823` | 20 | 54 | absent | `null` | 185.5 | -104.2 | 🔁 | 10 |
| `ai_v9_27_extremedial_probe_0823` | 20 | 29 | absent | `null` | 158.6 | -27.0 | 🔁 | 6 |
| `ai_v9_50_fdF_p1c_0826` | 13 | 12 | absent | `null` | 119.4 | -1.0 |  | 4 |
| `ai_v9_39_fdB_lossonly_0825` | 14 | 22 | absent | `null` | 113.0 | -11.9 | 🔁 | 5 |
| `ai_v9_24_E3_substrate_on_0822` | 18 | 87 | absent | `null` | 102.9 | -19.3 | 🔁 | 4 |
| `ai_v9_23_E2_substrate_on_0822` | 18 | 87 | absent | `null` | 102.2 | -15.9 | 🔁 | 6 |
| `ai_v9_25_E4_baitbot_0822` | 20 | 113 | absent | `null` | 100.8 | -14.4 | 🔁 | 3 |
| `ai_v9_42_fdE_single_0825` | 14 | 22 | absent | `null` | 97.8 | -11.0 | 🔁 | 6 |
| `ai_v9_62_R2PLAIN_0827` | 14 | 19 | absent | `null` | 95.0 | -9.1 | 🔁 | 4 |
| `ai_v9_49_G2_advgate_0826` | 14 | 23 | absent | `null` | 92.4 | -3.2 | 🔁 | 6 |
| `ai_v9_22_E1_substrate_on_0821` | 18 | 81 | absent | `null` | 91.6 | -16.0 | 🔁 | 7 |
| `ai_v9_20_tdaux_rung2_lam10_0820` | 14 | 20 | absent | `null` | 81.2 | -12.8 | 🔁 | 4 |
| `ai_v9_52_G1p_matched_0826` | 14 | 23 | absent | `null` | 81.1 | -3.4 |  | 0 |
| `ai_v9_70_R3ACTION_0828` | 16 | 29 | absent | `null` | 80.1 | -4.4 | 🔁 | 4 |
| `ai_v9_60_R2TOPK_0827` | 14 | 22 | absent | `null` | 75.5 | -9.4 | 🔁 | 4 |
| `ai_v9_20_tdaux_rung2_lam00_0820` | 14 | 19 | absent | `null` | 75.3 | -6.8 |  | 4 |
| `ai_v12_02_winprob_critic` | 20 | 495 | absent | `null` | 73.1 | -73.1 |  | 3 |
| `ai_v9_160_TCFUNDA_0903` | 16 | 29 | absent | `null` | 66.5 | -5.5 |  | 4 |
| `ai_v9_76_R4ACTION_0830` | 16 | 29 | absent | `null` | 65.9 | -6.0 |  | 5 |
| `ai_v9_38_fdA_coef03_0825` | 14 | 22 | absent | `null` | 65.5 | -1.6 | 🔁 | 3 |
| `ai_v9_20_tdaux_rung2_lam30_0820` | 14 | 20 | absent | `null` | 65.1 | -12.8 | 🔁 | 3 |
| `ai_v9_48_G1_action_0826` | 14 | 22 | absent | `null` | 65.0 | -5.9 |  | 3 |
| `ai_v9_170_TCUNFK6A_0904` | 16 | 29 | absent | `null` | 63.5 | -10.4 |  | 3 |
| `ai_v9_51_fdF_p2c_0826` | 11 | 8 | absent | `null` | 62.3 | +3.2 🚨2 unrateable |  | — |
| `ai_v9_91_COMPFOLD_0831` | 16 | 29 | absent | `null` | 61.6 | -5.6 |  | 4 |
| `ai_v9_161_TCFUNDB_0903` | 16 | 29 | absent | `null` | 61.1 | -6.7 | 🔁 | 3 |
| `ai_v9_61_R2KL_0827` | 14 | 25 | absent | `null` | 59.4 | -5.4 | 🔁 | 7 |
| `ai_v9_141_C1_0901` | 16 | 29 | absent | `null` | 58.9 | -9.5 |  | 2 |
| `ai_v9_171_TCUNFK6B_0904` | 16 | 29 | absent | `null` | 57.8 | -3.6 |  | 2 |
| `ai_v9_163_TCUNFB_0903` | 16 | 29 | absent | `null` | 55.2 | -1.3 |  | 3 |
| `ai_v9_08_gen7_seed_quantile_0811` | 7 | 21 | absent | `null` | 55.0 | -55.0 |  | 0 |
| `ai_v9_40_fdC_ecology_0825` | 14 | 19 | absent | `null` | 54.0 | -5.9 |  | 2 |
| `ai_v9_02_gen2_full11_40m_0805` | 19 | 171 | absent | `null` | 53.5 | -35.6 | 🔁 | 2 |
| `ai_v9_162_TCUNFA_0903` | 16 | 29 | absent | `null` | 53.3 | +0.5 |  | 2 |
| `ai_v9_01_gen1_edges6_40m_0804` | 19 | 171 | absent | `null` | 50.7 | -29.4 |  | 4 |
| `ai_v9_172_G1SHORT_0905` | 16 | 29 | absent | `null` | 50.1 | -3.7 |  | 1 |
| `ai_v9_07_gen6_seed_vicreg_0810` | 9 | 36 | absent | `null` | 49.7 | -49.7 | 🔁 | 1 |
| `ai_v9_71_R3ACTIONHI_0828` | 16 | 29 | absent | `null` | 48.5 | +2.2 |  | 1 |
| `ai_v12_01_winprob_critic` | 11 | 55 | absent | `null` | 47.3 | -47.3 |  | 2 |
| `ai_v9_151_R4DOSE6_0901` | 16 | 29 | absent | `null` | 47.3 | -6.7 |  | 3 |
| `ai_v9_12_gen10_t0prior_0814` | 12 | 66 | absent | `null` | 47.0 | -47.0 | 🔁 | 0 |
| `ai_v9_13_gen11_labelonly_winprob_0815` | 12 | 66 | absent | `null` | 45.8 | -45.8 |  | 1 |
| `ai_v9_03_gen25_consequence_25m_0806` | 12 | 66 | absent | `null` | 45.2 | -45.2 | 🔁 | 4 |
| `ai_v9_29_rev1_0823` | 12 | 66 | absent | `null` | 45.1 | -45.1 | 🔁 | 3 |
| `ai_v9_150_R4DOSE12_0901` | 16 | 29 | absent | `null` | 44.7 | -5.2 |  | 0 |
| `ai_v9_143_N2_0901` | 15 | 14 | absent | `null` | 44.6 | -5.6 | 🔁 | 1 |
| `ai_v9_140_B2_0901` | 16 | 29 | absent | `null` | 44.5 | -3.7 |  | 4 |
| `ai_v9_09_gen8_beliefs_threat_inject_0811` | 13 | 78 | absent | `null` | 44.3 | -44.3 | 🔁 | 3 |
| `ai_v9_06_gen5_no_concat_0809` | 12 | 66 | absent | `null` | 44.1 | -44.1 |  | 1 |
| `ai_v9_34_tick1_0824` | 5 | 10 | absent | `null` | 43.7 | -43.7 | 🔁 | 2 |
| `ai_v9_04_gen3_k6_recency_40m_0807` | 20 | 190 | absent | `null` | 43.6 | -32.4 |  | 1 |
| `ai_v9_18_gen15_v8rewards_0818` | 12 | 66 | absent | `null` | 42.9 | -42.9 |  | 2 |
| `ai_v9_05_gen4_rehome_25m_0808` | 11 | 55 | absent | `null` | 42.3 | -41.3 |  | 1 |
| `ai_v9_10_gen9_intent_distcritic_0813` | 13 | 78 | absent | `null` | 38.7 | -38.7 | 🔁 | 3 |
| `ai_v9_19_gen16_mechanics_0819` | 12 | 66 | absent | `null` | 38.5 | -29.4 | 🔁 | 1 |
| `ai_v9_21_gen17_pfspoff_0820` | 12 | 66 | absent | `null` | 37.2 | -37.2 |  | 2 |
| `ai_v9_14_gen12_h_entitypool_shaping_0816` | 12 | 66 | absent | `null` | 35.0 | -35.0 |  | 1 |
| `ai_v9_59_R2ACTION_0827` | 14 | 21 | absent | `null` | 34.9 | -10.1 | 🔁 | 4 |
| `ai_v9_16_gen14_framedel_v91_0817` | 12 | 132 | absent | `null` | 31.2 | -25.1 |  | 1 |
| `ai_v9_58_R2CTRL_0827` | 2 | 1 | absent | `null` | 25.6 | -25.6 | 🔁 | 1 |
| `ai_v9_15_gen13_hb_events_stack_0817` | 12 | 132 | absent | `null` | 25.5 | -25.5 | 🔁 | 1 |
| `ai_v9_17_tdaux_lam3_0818` | 2 | 1 | absent | `null` | 24.8 | -23.3 | 🔁 | 1 |
| `ai_v9_17_tdaux_lam1_0818` | 2 | 1 | absent | `null` | 22.8 | -21.2 | 🔁 | 1 |
| `ai_v9_82_REFOLD1_0830` | 2 | 1 | absent | `null` | 20.2 | -20.2 |  | 0 |
| `ai_v9_37_tick1_dosext_0825` | 2 | 1 | absent | `null` | 19.8 | -10.2 |  | 0 |
| `ai_v8_03_zarch_control_0718` | 10 | 45 | absent | `null` | 13.9 | -4.4 |  | 3 |
| `ai_v8_14_distill3_0725` | 2 | 1 | absent | `null` | 2.5 | -1.4 |  | 0 |
| `ai_v12_10_ladder_vf15` | 4 | 6 | absent | `6` | 0.0 | +0.0 |  | 0 |
| `ai_v12_11_ladder_ctrl10M` | 4 | 6 | absent | `6` | 0.0 | +0.0 |  | 0 |
| `ai_v12_12_ladder_cflabels` | 4 | 6 | absent | `6` | 0.0 | +0.0 |  | 0 |
| `ai_v12_13_ladder_tdaux` | 4 | 6 | absent | `6` | 0.0 | +0.0 |  | 0 |
| `ai_v12_14_ladder_truevalue` | 4 | 6 | absent | `6` | 0.0 | +0.0 |  | 0 |
| `ai_v12_15_ladder_ctrl10M_b` | 4 | 6 | absent | `6` | 0.0 | +0.0 |  | 0 |
| `ai_v12_16_ladder_ctrl10M_c` | 4 | 6 | absent | `6` | 0.0 | +0.0 |  | 0 |
| `ai_v12_17_ladder_strata` | 4 | 6 | absent | `6` | 0.0 | +0.0 |  | 0 |
| `ai_v12_19_ladder_lambda09` | 5 | 10 | absent | `10` | 0.0 | +0.0 |  | 0 |
| `ai_v12_20_ladder_denseaux` | 4 | 6 | absent | `6` | 0.0 | +0.0 |  | 0 |
| `ai_v12_21_ladder_lambda09_b` | 4 | 6 | absent | `6` | 0.0 | +0.0 |  | 0 |
| `ai_v12_22_ladder_lambda095` | 5 | 10 | absent | `10` | 0.0 | +0.0 |  | 0 |
| `ai_v12_23_ladder_rollout` | 5 | 10 | absent | `10` | 0.0 | +0.0 |  | 0 |
| `ai_v12_24_ladder_strata_b` | 4 | 6 | absent | `6` | 0.0 | +0.0 |  | 0 |
| `ai_v12_25_ladder_vf15_b` | 4 | 6 | absent | `6` | 0.0 | +0.0 |  | 0 |
| `ai_v12_28_ladder_ent05` | 4 | 6 | absent | `6` | 0.0 | +0.0 |  | 0 |
| `ai_v12_29_ladder_vf025` | 4 | 6 | absent | `6` | 0.0 | +0.0 |  | 0 |
| `ai_v13_01_flywheel_shaped` | 20 | 361 | absent | `51` | 0.0 | +0.0 |  | 0 |
| `ai_v13_02_flywheel_winprob` | 20 | 399 | absent | `48` | 0.0 | +0.0 |  | 0 |
| `ai_v13_03_fork` | 4 | 6 | absent | `6` | 0.0 | +0.0 |  | 0 |
| `ai_v13_04_flywheel_winprob_b` | 20 | 494 | absent | `47` | 0.0 | +0.0 |  | 0 |
| `ai_v13_09_wcont` | 20 | 95 | absent | `20` | 0.0 | +0.0 |  | 0 |
| `ai_v13_11_split_lossoff` | 20 | 114 | absent | `23` | 0.0 | +0.0 |  | 0 |
| `ai_v13_12_plateau` | 20 | 76 | absent | `16` | 0.0 | +0.0 |  | 0 |
| `ai_v13_16_teach5_offense_dist` | 20 | 76 | absent | `16` | 0.0 | +0.0 |  | 0 |

## 5. The tool that applies it — built, tested, NOT run on `models/`

```bash
python -m main.elo refit <run_dir>            # read-only: committed vs refit, node by node
python -m main.elo refit --apply <run_dir>    # write the stamped fit, KEEP the old file
```

* It **refits, it never plays** — `games.jsonl` is append-only and never stale.
* It fits the **committed file's own node set** (`--pool` overrides to use the snapshots on disk).
  Bradley–Terry re-solves every node on every add and the newest node of a fit is systematically
  inflated, so a fit over a different node set is a different object: restricting to the committed
  nodes is what makes a delta mean *the recipe moved this node* and nothing else.
* `--apply` writes the refit to `snapshot_ladder/ladder.json` and keeps the file it replaces as
  **`snapshot_ladder/ladder.pre_recipe.json`** — the only surviving evidence of what a banked
  number was quoted from. It **refuses** rather than overwrite a backup that already exists, and
  is a **no-op** on a file that is already current and identical.
* A run with no `games.jsonl` gets a refusal, never a guess (none exist today — all 93 have one).
* `python -m main.elo <run>` now prints a `[ladder]` headline line under its star table, and
  **withholds the dense number** on an unstamped file, printing the refusal in its place. Every
  reader's refusal (`main.critic_gate`, `--exploiter-ladder auto:`, `main.ops.plateau_signal`,
  this one) names both `--fit-only` and `refit --apply`, worded once in
  `snapshot_ladder.recipe_refusal`.

Pinned by `src/main/elo_refit_test.py` (11 tests on a fixture run): the read writes nothing, the
node set is the committed one, `--apply` keeps the old file verbatim, a second apply that would
change the file refuses, an already-current file is a no-op, and the refusal names the tool.

### 🚨 Ledger note — who applies this, and what it costs

**`refit --apply` was NOT run against `models/` by this session** (this session may not write
there). The 93 refits above were done in memory and are recorded in the per-run JSON files beside
this README, so the archive can be converted at any time with no measurement lost.

**The owner or the Training Run session applies it.** The whole archive is ~93 invocations of a
fit that takes ~0.01 s each and plays nothing; it is safe to run on a finished run at any time,
and safe on a LIVE run only in the sense that the run's own pinned updater will overwrite
`ladder.json` at its next promotion with a pre-fix fit again — so a live arm is refit at run END,
not mid-run. After applying, a run's `ladder.pre_recipe.json` is what a historical ledger entry's
number came from; **the ledger entry itself is never edited** — it says what was believed then.
Append the new reading beside it.

## 6. The era's banked claims that quoted a committed file

Every `ladder.json` citation in `ledger.md` dated **2026-09-06 onward**, with what the refit does
to the number. **No ledger edit is made or owed** — this table is the conversion.

| date | ledger line | run(s) quoted | what was quoted | refit | moves > 10 Elo? |
|---|---:|---|---|---|:-:|
| 2026-09-08 | 15346 | `ai_v12_02_winprob_critic` | the arm's full 20-node ladder, 36M 1996.6 … 74M **2057.3**, explicitly flagged in-line as written by the run's pre-fix updater | 36M 1954.5 … 74M **1984.2** | **YES — every node, −38.2 to −73.1** |
| 2026-09-08 | 15453 | `ai_v12_02_winprob_critic` | run-end matched-count −51.0 [−81.5, −20.5] vs `famine_comparator`, off the same committed file | superseded; already registered INVALID as a like-for-like (groomed pool) in the same entry | **YES (already void)** |
| 2026-09-08 | 15757 | `ai_v12_11_ladder_ctrl10M` | 4M 1742.2 · 6M 1895.6 · 8M 2004.5 · 10M **2018.7** | identical | no — **0.0** |
| 2026-09-09 | 15951 | `ai_v12_13_ladder_tdaux` | 4M 1738.7 · 6M 1929.7 · 8M 1987.7 · 10M **2034.7** | identical | no — **0.0** |
| 2026-09-09 | 16128 | `ai_v12_12_ladder_cflabels` | 4M 1779.9 · 6M 1883.6 · 8M 1975.9 · 10M **2021.1** | identical | no — **0.0** |
| 2026-09-07 | 13421 | (live arm, no file read) | "Δ vs the fold parent at 14 snapshots: +87 [+50, +123] (**committed would read +91**)" — the entry states it REFIT rather than read, and says no `ladder.json` on disk was rewritten | already a refit; the +91 is the stale figure it declined | no — the correct number was already used |
| 2026-09-12 | 18065 | `ai_v12_28_ladder_ent05` family | explicitly **NOT** a strength claim; `eval/elo` quoted as a descriptor, ladder named only as the standing rule | n/a | no |
| 2026-09-12 | 17979 | `ai_v9_59_R2ACTION` | a HAZARD note (12 inherited + 2 own nodes share a fork-seeding mtime), no rating quoted | that run's max \|Δ\| is 34.9, newest −10.1 | the hazard stands; no number to move |
| 2026-09-14 | 18360/18370 | `ai_v12_02_winprob_critic` | the discovery entry itself — the +73 trap | this audit reproduces it to the decimal (−73.1) | **confirms** |
| 2026-09-16 | 18671 | `ai_v13_01_flywheel_shaped`, `ai_v13_02_flywheel_winprob` | arm S **2036.6**, arm W **2019.1**, Δ +17.5 [−6.9, +41.9] at matched 20-node count | identical | no — **0.0**, as the entry itself claimed |
| 2026-09-18 | 19083 | S, W, `ai_v13_04_flywheel_winprob_b` | S 2036.6 · W 2019.1 · W_b **2044.3**; finding 17.5 within a 25.2 floor | identical | no — **0.0**, three times |

**Reading.** The flywheel era (2026-09-16 onward) is CLEAN: those entries checked the count key
before quoting and the refits confirm them to 0.0. The v12 ladder-cell campaign (2026-09-08/09) is
CLEAN for the same reason. **The one banked number that moves is `ai_v12_02_winprob_critic`'s
20-node ladder**, and the 2026-09-08 entry that published it already carried the caveat in its own
sentence, while the 2026-09-14 entry converted it. So no *era* claim is overturned by this audit.

🚨 **The exposure is OLDER than the era grepped for.** 68 of the 93 files move, and the ones whose
ordering breaks are the **v9 generation ladder** (gen1..gen17, 2026-08-04 → 2026-08-20) — a
campaign whose per-generation comparisons predate 2026-09-06 and were therefore outside the
requested window. Those comparisons were mostly read at matched snapshot COUNT through
`critic_gate`, which refits, so a file-quoted ordering is not automatically what was banked; but
any gen-vs-gen ordering quoted from a committed file is now known to be unsafe in 21 of 153 pairs.
That is an OPEN item, not a finding, and closing it means re-reading the v9 entries the way this
table re-reads the v12/v13 ones.
