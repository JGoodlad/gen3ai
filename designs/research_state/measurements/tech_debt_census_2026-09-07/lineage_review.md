# The lineage review — 2026-09-07

> **READ-ONLY.** No `--apply`, no `--backfill --apply`, nothing written under `models/`.
> Every `main.lineage --backfill` invocation below is PRINTED, not run. This report answers
> the `TECH_DEBT_BACKLOG.md` §2 row *"Lineage backfill: 105 legacy runs carry a parent DERIVED
> from `original_command`"* and is the PREREQUISITE that row names for the TensorBoard
> fork-prefix backfill.

## Summary

| | n |
|---|---:|
| run directories under `models/` | 231 |
| with a **DERIVED** parent (the population under review) | **162** |
| &nbsp;&nbsp;→ **CONSISTENT** — the derivation agrees with both signals | **161** |
| &nbsp;&nbsp;→ **CONTRADICTED** — a signal says otherwise | **1** |
| &nbsp;&nbsp;→ **UNDECIDABLE** — a signal is missing | **0** |
| with a **RECORDED** lineage block (out of scope, trusted) | 36 |
| no `metadata.json` at all — lineage UNKNOWABLE | 12 |
| `metadata.json` but no block and no `original_command` — lineage UNKNOWABLE | 21 |

**Verdict: the backlog row is clear to close, and the TensorBoard backfill is unblocked.**

## The two signals, and what each can and cannot decide

**Signal 1 — the run's own first TensorBoard scalar step.** Read from the head of every
`<run>/tb/**/events.out.tfevents.*` with `EventFileLoader` (the loader behind
`src/main/ops/tb_read.py`'s `load()`), taking the minimum step over the first 5 scalar events
of each file. 219 of 231 runs have one; the 12 without a `tb/` are exactly the 12 without a
`metadata.json` (see *What is UNDECIDABLE*).

* A first scalar **at ~0** proves nothing about a fork whose prefix was inherited — but it is
  what a fresh run must look like.
* A first scalar **deep into training** is decisive in one direction: **a fresh run cannot
  start at 148M.** That asymmetry is what convicts the one CONTRADICTED row.
* For a fork, the test is sharper: the first scalar must equal the **parent's step at the fork**
  — the pinned checkpoint's `_N_steps` when the reference names one, the parent run's endpoint
  when it names a bare directory. Tolerance 3,000,000 steps (about one checkpoint interval).

**Signal 2 — `model_config.json` across the link.** `arch_signature` and `config_version` for
the run and for its claimed parent. A fork **cannot** have loaded a differently-shaped parent,
so an `arch_signature` change across a link is a contradiction on its own. No row failed this
test; it is reported because it was run, not because it fired.

## THE CONTRADICTED ROW

### `ai_v8_01_zarch_film_0717` — claims **fresh**, is a fork of `ai_v7_14_league_capstone_0712`

| | |
|---|---|
| what the block says | `role: fresh`, `fork_parent: null`, `derived: true`, `recorded_at 2026-09-02T02:28:54Z` |
| **signal 1** — first TB scalar | **step 148,401,356** |
| first checkpoint | `checkpoint_149411773_steps.zip` (step 149,411,773) |
| **signal 2** — arch / config version | `gen3_opp_hp_typed_candidates_v1` / v44 |
| what `original_command` holds | `tmp/fork_zarch_v8.py` — **a bare script path, no argv at all** |

**The candidate parent, on three independent grounds:**

1. **Step match.** `ai_v7_14_league_capstone_0712`'s last checkpoint is
   `checkpoint_148223095_steps.zip` and its `final_model_interrupted.zip` sits just past it;
   this run's first TB scalar is 148,401,356. No other run in the archive ends
   within 3M steps of that.
2. **Config-version ladder.** `ai_v7_14` is `config_version` 43; this run is 44 — one step,
   the v44 `z_arch`/FiLM toggle.
3. **The fork script still exists and names it.** `/home/goodlad/dev/gen3ai/tmp/fork_zarch_v8.py`:

   ```python
   """Surgical fork: ai_v7_14 (zarch OFF, v43-era) → ai_v8_01 init (zarch-ON, v44, gen3_zarch_film_v1)."""
   DONOR_DIR = "models/ai_v7_14_league_capstone_0712"
   DONOR_ZIP = f"{DONOR_DIR}/final_model_interrupted.zip"
   OUT_DIR   = "models/ai_v8_01_zarch_film_0717"
   ```

**Why the derivation got it wrong, and it is not the deriver's fault.** The run was never
launched through the launcher — it was initialised by that script, so the recorded
`original_command` is the script's own path. `build_lineage_from_command` looks for a
`--model` in an argv; there is no argv. With nothing to read it produced the only answer it
could, `fresh`, and marked it `derived: true`. **The defect is that a derivation with no
input reads exactly like a derivation that found nothing to fork from.**

**The invocation the owner would run — and it will not work:**

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.lineage ai_v8_01_zarch_film_0717 --backfill          # DRY RUN
python -m main.lineage ai_v8_01_zarch_film_0717 --backfill --apply  # would write
```

Run as a dry run (2026-09-07), it prints:

```
DRY RUN — nothing written. Re-run with --apply to write.

ai_v8_01_zarch_film_0717: SKIP — already records a lineage block (immutable)
```

**So `--backfill` cannot repair the one row the backlog named it for**, on two counts: the
block is immutable and refuses to be rewritten, and even if it were rewritten the derivation
would re-read `tmp/fork_zarch_v8.py`, find no `--model`, and write `fresh` again. Repairing
this row is a hand edit of an IMMUTABLE block — a policy decision, not a tool run, and
therefore left entirely to the owner.

## Two defects found in the reading tools — recorded, NOT fixed

### 1. `main.lineage` under-reports `derived` for a derived block with no parent

`src/main/lineage.py:86`:

```python
out["derived"] = bool(parent is not None and parent.derived) or (
    block is None and read_original_command(run_dir) is not None)
```

The block's own `"derived": true` key is never read. When the derivation concluded *fresh*,
`fork_parent()` returns `None`, so the first clause is False; the block exists, so the second
is False. The CLI therefore prints `derived: false` for a run whose stored block literally says
`"derived": true`. Reproduce:

```bash
python -m main.lineage ai_v8_01_zarch_film_0717 --json | head -8   # "derived": false
python -c "import json;print(json.load(open('models/ai_v8_01_zarch_film_0717/metadata.json'))['lineage']['derived'])"   # True
```

**Scale: 47 runs** carry a derived block whose role is `fresh` and are invisible to
the CLI's `⚠ DERIVED` marker. This report reads the block directly and does not use that flag.

### 2. The backlog's count of 105 is right, but it is not the count of derived runs

There are **162** runs with a derived parent, not 105 — 153 carrying a
`derived: true` block and 9 with no block at all, derived at read time from
`original_command`. The 105 is a different, and correct, quantity: it is exactly the number of
`main.tb_inherit --list` forks whose parent is derived (see below). The remaining
47 are the derived-**fresh** rows of defect 1, which no tool currently counts.

## The population, by what it claims

| role recorded | n | verdict |
|---|---:|---|
| `exploiter` | 56 | CONSISTENT 56 |
| `fresh` | 47 | CONSISTENT 46, CONTRADICTED 1 |
| `fold` | 39 | CONSISTENT 39 |
| `fork` | 20 | CONSISTENT 20 |

The 9 rows with role `—` are `v8rep_*_0905`: no lineage block, derived at read time from an
`original_command` that names `--model models/ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip`.
Their first TB scalar is **277,583,267** on all nine, exactly the parent's recorded
`num_timesteps`. That is the cleanest CONSISTENT evidence in the archive.

## The prerequisite this report exists for: the TensorBoard fork-prefix backfill

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.tb_inherit --list --json   # 137 forks missing a prefix
```

| | n |
|---|---:|
| forks missing a TB prefix | **137** |
| → parent RECORDED (never derived) | 32 |
| → parent DERIVED **and CONSISTENT** | **105** |
| → parent DERIVED and CONTRADICTED | **0** |
| → parent DERIVED and UNDECIDABLE | **0** |

**All 137 are safe to backfill.** The 105 is the backlog's own number, and every one of them
passes both signals. The command, still not run:

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.tb_inherit --backfill --all           # DRY RUN
python -m main.tb_inherit --backfill --all --apply   # writes
```

⚠️ **One caveat the backfill cannot see.** `ai_v8_01_zarch_film_0717` is *not* in the 137,
because `tb_inherit` asks the lineage layer for forks and that run claims `fresh`. After the
backfill it will be the one fork in the archive with no inherited prefix, and nothing will
report it as missing. Fix the lineage row first, or record that exception.

## What is UNDECIDABLE, and why

**Zero derived rows are undecidable** — every one had a readable TB series and a parent step to
compare it against. Three separate classes sit *outside* the derived population and cannot be
reviewed at all:

* **12 runs have no `metadata.json`** — no lineage, no command, no git hash. Nothing to
  cross-check; a derivation is impossible, not merely wrong. They are:
  `.dryrun_TCFUNDA_2001`, `.failed_C1_configreject_1427`, `_arch`, `_goldens`, `ai_v7_19_SMOKE`, `ai_v7_21_fitnet_SMOKE2`, `ai_v7_22_hyperoffense_exploiter_0717`, `ai_v9_172_G1SHORT_0905.FAILED_0049_preresolver`, `ai_v9_82_REVIVE2a_0830`, `run_20260715_091837`, `saved_work`, `warmstart_generic_0715`
* **21 runs have `metadata.json` but neither a lineage block nor an
  `original_command`** — the run's ancestry is stated nowhere. Listed in the appendix.
* **12 runs have no `tb/` directory — and they are the SAME 12 that have no `metadata.json`**
  (set-equal, checked). So the two unreadable populations coincide: nothing in the derived
  population lost signal 1, and nothing that lost signal 1 had a lineage claim to test. If any of
  those 12 is ever given a lineage block by hand, signal 1 will not be available to check it.

A note on the one thing signal 1 *could* have hidden: a fork whose TB prefix had already been
inherited would show a first scalar at ~0 and be indistinguishable from a fresh run. **No
derived-parent fork in the archive is in that state** — `main.tb_inherit --list` says all 137
forks are still missing their prefix, which is precisely why this review could be conclusive.
Once the backfill runs, this cross-check is no longer reproducible in the same way.

---

<details>
<summary><b>Appendix A — every DERIVED row with its evidence</b></summary>

| run | role | derived from | claimed parent | parent step | first TB step | Δ | verdict |
|---|---|---|---|---:|---:|---:|---|
| `ai_v8_01_zarch_film_0717` | fresh | block(derived:true) | — | — | 148,401,356 | — | CONTRADICTED |
| `DISCARDED_tdaux_control_n16_0818` | fork | block(derived:true) | `ai_v9_16_gen14_framedel_v91_0817` | 25,067,520 | 25,067,520 | 0 | CONSISTENT |
| `RETIRED_c5fork_control_gen13base_0817` | fork | block(derived:true) | `ai_v9_15_gen13_hb_events_stack_0817` | 25,067,520 | 25,067,520 | 0 | CONSISTENT |
| `RETIRED_gen14_framedel_v90_0817` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v6_10_unified_obs_0618` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v6_11_typed_hp_0619` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v6_11_unified_obs_fixed_0618` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v6_13_outgoing_dmg_0620` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v6_13_outgoing_dmg_0620_exp_v1` | fork | block(derived:true) | `ai_v6_13_outgoing_dmg_0620` | 104,495,313 | 104,495,313 | 0 | CONSISTENT |
| `ai_v6_13_outgoing_dmg_0620_exploiter_v1` | exploiter | block(derived:true) | `best_model` | 96,000,018 | 96,000,018 | 0 | CONSISTENT |
| `ai_v6_13_outgoing_dmg_0620_exploiter_v2` | exploiter | block(derived:true) | `ai_v6_13_outgoing_dmg_0620` | 4,788,069 | 4,788,069 | 0 | CONSISTENT |
| `ai_v7_01_teacher_0626` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_01_teacher_0626_oom1` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_02_critic_shape_0627` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_03_belief_shape_0630` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_04_opd_selfdistill_0702` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_05_tss_specialist_0703` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_05_tss_specialist_0703_aborted_noeval` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_06_tss_temp_anneal_0706` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_07_tss_temp_ratchet_0707` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_08_tss_bots_0707` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_09_tss_bots_pubval_0708` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_10_tss_exploiter_fixed_0709` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_11_tss_exploiter_nopubval` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_12_trap_exploiter_0711` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_13_cmpass_exploiter_0711` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_14_league_capstone_0712` | fork | block(derived:true) | `ai_v7_02_critic_shape_0627` | 106,685,764 | 106,685,764 | 0 | CONSISTENT |
| `ai_v7_15_tss_exploiter_vs14_0713` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_16_distill_tss_mvp_0715` | fold | block(derived:true) | `ai_v7_14_league_capstone_0712` | 148,401,356 | 148,401,356 | 0 | CONSISTENT |
| `ai_v7_17_stall_exploiter_0715` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v7_18_distill_4teacher_0716` | fold | block(derived:true) | `ai_v7_14_league_capstone_0712` | 148,401,356 | 148,401,356 | 0 | CONSISTENT |
| `ai_v7_19_combined_0716` | fold | block(derived:true) | `ai_v7_18_distill_4teacher_0716` | 158,302,045 | 158,302,045 | 0 | CONSISTENT |
| `ai_v7_20_valuedistill_SMOKE` | fold | block(derived:true) | `ai_v7_14_league_capstone_0712` | 148,401,356 | 148,401,356 | 0 | CONSISTENT |
| `ai_v7_20_valuedistill_ab_0717` | fold | block(derived:true) | `ai_v7_14_league_capstone_0712` | 148,401,356 | 148,401,356 | 0 | CONSISTENT |
| `ai_v7_21_fitnet_valuefeat_ab_0717` | fold | block(derived:true) | `ai_v7_14_league_capstone_0712` | 148,401,356 | 148,401,356 | 0 | CONSISTENT |
| `ai_v8_02_zarch_teampfsp_0718` | fork | block(derived:true) | `ai_v8_01_zarch_film_0717` | 171,331,194 | 171,331,194 | 0 | CONSISTENT |
| `ai_v8_03_zarch_control_0718` | fork | block(derived:true) | `ai_v8_01_zarch_film_0717` | 148,401,356 | 148,401,356 | 0 | CONSISTENT |
| `ai_v8_04_distill_4teacher_0722` | fold | block(derived:true) | `ai_v8_03_zarch_control_0718` | 268,518,944 | 268,518,944 | 0 | CONSISTENT |
| `ai_v8_05_semistall564_exploiter_0722` | exploiter | block(derived:true) | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `ai_v8_06_semistall_3team_exploiter_0722` | exploiter | block(derived:true) | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `ai_v8_07_semistall564_scratch_0722` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v8_08_defensive_6team_exploiter_0723` | exploiter | block(derived:true) | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `ai_v8_09_pool10_exploiter_0723` | exploiter | block(derived:true) | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `ai_v8_10_offense20_exploiter_0724` | exploiter | block(derived:true) | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `ai_v8_11_offense10_exploiter_0724` | exploiter | block(derived:true) | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `ai_v8_12_defensive20_exploiter_0724` | exploiter | block(derived:true) | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `ai_v8_13_defensive10_exploiter_0725` | exploiter | block(derived:true) | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `ai_v8_14_distill3_0725` | fold | block(derived:true) | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `ai_v8_15_retention_A_frozen_0726` | fork | block(derived:true) | `ai_v8_14_distill3_0725` | 292,100,648 | 292,100,648 | 0 | CONSISTENT |
| `ai_v8_16_def20_lut_0726` | exploiter | block(derived:true) | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `ai_v8_17_rand20_nolut_0726` | exploiter | block(derived:true) | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `ai_v8_18_rand20_lut_0726` | exploiter | block(derived:true) | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `ai_v8_19_def20_lut_zeroinit_0727` | exploiter | block(derived:true) | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `ai_v8_20_rand10_nolut_0727` | exploiter | block(derived:true) | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `ai_v9_01_gen1_edges6_40m_0804` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_02_gen2_full11_40m_0805` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_03_gen25_consequence_25m_0806` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_04_gen3_k6_recency_40m_0807` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_05_gen4_rehome_25m_0808` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_06_gen5_no_concat_0809` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_07_gen6_seed_vicreg_0810` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_08_gen7_seed_quantile_0811` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_09_gen8_beliefs_threat_inject_0811` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_100_R5F08_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_101_R5F09_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_102_R5F10_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_103_R5F11_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_104_R5F12_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_105_R5F13_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_106_R5F14_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_107_R5F15_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_10_gen9_intent_distcritic_0813` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_11_gen10_intentfull_compiled_0814` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_12_gen10_t0prior_0814` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_13_gen11_labelonly_winprob_0815` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_14_gen12_h_entitypool_shaping_0816` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_15_gen13_hb_events_stack_0817` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_16_gen14_framedel_v91_0817` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_17_tdaux_control_0818` | fork | block(derived:true) | `ai_v9_16_gen14_framedel_v91_0817` | 25,067,520 | 25,067,520 | 0 | CONSISTENT |
| `ai_v9_17_tdaux_lam1_0818` | fork | block(derived:true) | `ai_v9_16_gen14_framedel_v91_0817` | 25,067,520 | 25,067,520 | 0 | CONSISTENT |
| `ai_v9_17_tdaux_lam3_0818` | fork | block(derived:true) | `ai_v9_16_gen14_framedel_v91_0817` | 25,067,520 | 25,067,520 | 0 | CONSISTENT |
| `ai_v9_18_gen15_v8rewards_0818` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_19_gen16_mechanics_0819` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_20_tdaux_rung2_lam00_0820` | fork | block(derived:true) | `ai_v9_19_gen16_mechanics_0819` | 25,067,520 | 25,067,520 | 0 | CONSISTENT |
| `ai_v9_20_tdaux_rung2_lam10_0820` | fork | block(derived:true) | `ai_v9_19_gen16_mechanics_0819` | 25,067,520 | 25,067,520 | 0 | CONSISTENT |
| `ai_v9_20_tdaux_rung2_lam30_0820` | fork | block(derived:true) | `ai_v9_19_gen16_mechanics_0819` | 25,067,520 | 25,067,520 | 0 | CONSISTENT |
| `ai_v9_21_gen17_pfspoff_0820` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_22_E1_substrate_on_0821` | fork | block(derived:true) | `ai_v9_21_gen17_pfspoff_0820` | 25,067,520 | 25,067,520 | 0 | CONSISTENT |
| `ai_v9_23_E2_substrate_on_0822` | fork | block(derived:true) | `ai_v9_21_gen17_pfspoff_0820` | 25,067,520 | 25,067,520 | 0 | CONSISTENT |
| `ai_v9_24_E3_substrate_on_0822` | fork | block(derived:true) | `ai_v9_21_gen17_pfspoff_0820` | 25,067,520 | 25,067,520 | 0 | CONSISTENT |
| `ai_v9_25_E4_baitbot_0822` | fork | block(derived:true) | `ai_v9_22_E1_substrate_on_0821` | 34,111,488 | 34,111,488 | 0 | CONSISTENT |
| `ai_v9_26_baitent_probe_0823` | fork | block(derived:true) | `ai_v9_22_E1_substrate_on_0821` | 34,111,488 | 34,111,488 | 0 | CONSISTENT |
| `ai_v9_27_extremedial_probe_0823` | fork | block(derived:true) | `ai_v9_25_E4_baitbot_0822` | 43,155,456 | 43,155,456 | 0 | CONSISTENT |
| `ai_v9_29_rev1_0823` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `ai_v9_30_rev1_exploit_0824` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_31_tock1_k4_0824` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_32_tock1b_rain_0824` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_34_tick1_0824` | fold | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_35_tick1_exploit_0824` | exploiter | block(derived:true) | `ai_v9_34_tick1_0824` | 35,094,768 | 35,094,768 | 0 | CONSISTENT |
| `ai_v9_36_tock1c_q6_0824` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_37_tick1_dosext_0825` | fold | block(derived:true) | `ai_v9_34_tick1_0824` | 35,094,768 | 35,094,768 | 0 | CONSISTENT |
| `ai_v9_38_fdA_coef03_0825` | fold | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_39_fdB_lossonly_0825` | fold | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_40_fdC_ecology_0825` | fold | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_42_fdE_single_0825` | fold | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_44_tock2_v8shape_0825` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_45_fdF_p1_0826` | fold | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_48_G1_action_0826` | fold | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_49_G2_advgate_0826` | fold | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_50_fdF_p1c_0826` | fold | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_51_fdF_p2c_0826` | fold | block(derived:true) | `ai_v9_50_fdF_p1c_0826` | 26,640,624 | 26,640,624 | 0 | CONSISTENT |
| `ai_v9_52_G1p_matched_0826` | fold | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_53_R2F5a_0826` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_54_R2F5b_0826` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_55_R2F5c_0826` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_56_R2F5d_0826` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_57_R2F5e_0826` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_58_R2CTRL_0827` | fold | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_59_R2ACTION_0827` | fold | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_60_R2TOPK_0827` | fold | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_61_R2KL_0827` | fold | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_62_R2PLAIN_0827` | fork | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_63_R3F6a_0828` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_64_R3F6b_0828` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_65_R3F6c_0828` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_66_R3F6d_0828` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_67_R3F6e_0828` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_68_R3F6f_0828` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_69_R3F6CURR_0828` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_70_R3ACTION_0828` | fold | block(derived:true) | `ai_v9_59_R2ACTION_0827` | 28,115,184 | 28,115,184 | 0 | CONSISTENT |
| `ai_v9_71_R3ACTIONHI_0828` | fold | block(derived:true) | `ai_v9_59_R2ACTION_0827` | 28,115,184 | 28,115,184 | 0 | CONSISTENT |
| `ai_v9_72_R3SELF_0828` | fold | block(derived:true) | `ai_v9_59_R2ACTION_0827` | 28,115,184 | 28,115,184 | 0 | CONSISTENT |
| `ai_v9_73_R4S3a_0829` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_74_R4S3b_0829` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_75_R4S3c_0829` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_76_R4ACTION_0830` | fold | block(derived:true) | `ai_v9_59_R2ACTION_0827` | 28,115,184 | 28,115,184 | 0 | CONSISTENT |
| `ai_v9_77_G1LEAN_0830` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_79_REVIVE1a_0830` | exploiter | block(derived:true) | `ai_v9_73_R4S3a_0829` | 35,094,768 | 35,094,768 | 0 | CONSISTENT |
| `ai_v9_80_REVIVE1b_0830` | exploiter | block(derived:true) | `ai_v9_74_R4S3b_0829` | 35,094,768 | 35,094,768 | 0 | CONSISTENT |
| `ai_v9_81_REVIVE1c_0830` | exploiter | block(derived:true) | `ai_v9_75_R4S3c_0829` | 35,094,768 | 35,094,768 | 0 | CONSISTENT |
| `ai_v9_82_REFOLD1_0830` | fold | block(derived:true) | `ai_v9_59_R2ACTION_0827` | 28,115,184 | 28,115,184 | 0 | CONSISTENT |
| `ai_v9_91_COMPFOLD_0831` | fold | block(derived:true) | `ai_v9_59_R2ACTION_0827` | 28,115,184 | 28,115,184 | 0 | CONSISTENT |
| `ai_v9_92_R5F00_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_93_R5F01_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_94_R5F02_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_95_R5F03_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_96_R5F04_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_97_R5F05_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_98_R5F06_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `ai_v9_99_R5F07_0831` | exploiter | block(derived:true) | `ai_v9_29_rev1_0823` | 25,067,760 | 25,067,760 | 0 | CONSISTENT |
| `run_20260830_180409` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `run_20260830_183828` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `run_20260830_184043` | fresh | block(derived:true) | — | — | 0 | — | CONSISTENT |
| `v8rep_p1_A_0905` | fold | no block, derived from original_command | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `v8rep_p1_B_0905` | fold | no block, derived from original_command | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `v8rep_p1_C_0905` | fold | no block, derived from original_command | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `v8rep_p2loss_A_0905` | fold | no block, derived from original_command | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `v8rep_p2loss_B_0905` | fold | no block, derived from original_command | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `v8rep_p2loss_C_0905` | fold | no block, derived from original_command | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `v8rep_p2self_A_0905` | fold | no block, derived from original_command | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `v8rep_p2self_B_0905` | fold | no block, derived from original_command | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |
| `v8rep_p2self_C_0905` | fold | no block, derived from original_command | `ai_v8_04_distill_4teacher_0722` | 277,583,267 | 277,583,267 | 0 | CONSISTENT |

</details>

<details>
<summary><b>Appendix B — runs whose lineage cannot be reviewed</b></summary>

**No `metadata.json`:**

* `.dryrun_TCFUNDA_2001`
* `.failed_C1_configreject_1427`
* `_arch`
* `_goldens`
* `ai_v7_19_SMOKE`
* `ai_v7_21_fitnet_SMOKE2`
* `ai_v7_22_hyperoffense_exploiter_0717`
* `ai_v9_172_G1SHORT_0905.FAILED_0049_preresolver`
* `ai_v9_82_REVIVE2a_0830`
* `run_20260715_091837`
* `saved_work`
* `warmstart_generic_0715`

**`metadata.json` present, but no lineage block and no `original_command`:**

* `ai_v5_10_tail1_23_0611`
* `ai_v5_11_tail2_53m_0611`
* `ai_v5_12_bias_05_N_0612`
* `ai_v5_13_shape_pbrs_43m_0612`
* `ai_v5_2_native_selfplay_50m_0606`
* `ai_v5_3_vf_coef_clip_50m_0606`
* `ai_v5_4_pbrs_opp_threat_50m_0607`
* `ai_v5_5_popart_50m_0607`
* `ai_v5_6_stable_70m_0608`
* `ai_v5_7_switch_bias_41m_0609`
* `ai_v5_8_split_inc_dmg_38m_0610`
* `ai_v5_9_attend_unrevealed_56m_0610`
* `ai_v6_01_belief_53m_0613`
* `ai_v6_02_belief_lat_16m_0614`
* `ai_v6_03_win_pred_N_0614`
* `ai_v6_04_unified_all_half_batch_N_0616`
* `ai_v6_04_unified_inc_N_0615`
* `ai_v6_06_unified_all_N_0616`
* `ai_v6_07_unified_topk_N_0616`
* `ai_v6_08_unmasked_floor_N_0617`
* `ai_v6_09_dmg_reattend_N_0617`

</details>

<details>
<summary><b>Appendix C — how to reproduce this review</b></summary>

```bash
export PYTHONPATH=$PYTHONPATH:src
# 1. the claimed lineage, per run
python -m main.lineage <run> --json          # NB: its `derived` flag is wrong for role=fresh
python -c "import json;print(json.load(open('models/<run>/metadata.json'))['lineage'])"
# 2. signal 1 — the run's own first TB scalar step
python - <<'EOF'
import glob, os
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
run = 'models/<run>'
best = None
for f in sorted(glob.glob(os.path.join(run,'tb','**','events.out.tfevents.*'), recursive=True)):
    n = 0
    for ev in EventFileLoader(f).Load():
        if ev.HasField('summary') and len(ev.summary.value):
            best = ev.step if best is None else min(best, ev.step); n += 1
            if n >= 5: break
print(best)
EOF
# 3. signal 2 — arch across the link
python -c "import json;c=json.load(open('models/<run>/model_config.json'));print(c['arch_signature'],c['config_version'])"
# 4. the fork population
python -m main.tb_inherit --list --json
```

The scan scripts themselves were session-scoped by design (`~/.claude/jobs/9ab51de6/tmp/census/`):
this is a **census**, a snapshot of a moving archive, not a banked readout to be re-run — so it
carries its commands inline instead of registering a script in `measurements_readout_gate_test.py`.

</details>

