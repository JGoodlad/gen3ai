# `models/` retention policy

*Written 2026-09-06. **Nothing has been applied.** Both policies below have been run as dry runs
and their reports are committed beside them; the `--apply` commands are in §7.*

The run archive is the one thing in this project that is never reproducible. A checkpoint is a
weight file that no amount of compute re-derives, because the run that produced it consumed dice,
a pool, an opponent distribution and a commit that no longer all coexist. So the whole tool is
built to **REFUSE rather than to reclaim**: every rule below says what is KEPT and why, the
deletion set is what falls out, and a run the rules cannot classify gets *no plan at all*.

Two policies live in one script, `designs/research_state/measurements/archive_grooming_dryrun.py`:

| policy | what it asks | frees |
|---|---|---|
| **`--policy standing`** (the DEFAULT) | *is this run CLOSED?* — one rule for all of them | 18.8 GB / 7.3% |
| **`--policy tiered`** | *which ERA, and is anything still reaching for it?* | **87.9 GB / 34.1%** |

The standing policy is unchanged and stays the default. The tiered policy is what this document
specifies.

---

## 1. Why a second policy, in three measurements

**(a) The standing policy cannot see 20.8 GB.** It only ever plans inside `checkpoints/` and
`eval_traces/`. But 13 pre-v8 runs use the LEGACY layout with their checkpoints at the run ROOT —
**1,440 files, 20.76 GB** — which is more than the standing policy frees across the entire archive.
No amount of tuning its stride reaches them; it is looking in the wrong directory.

**(b) `snapshots/` has 52.2 GB and no rule at all.** It is the archive's second-largest consumer
and the standing policy says nothing about it, deliberately (it measured it and proposed nothing).

**(c) Whole eras are no longer in play, and the owner said so.** Verbatim, 2026-09-06:

> "Yes, please work on a reasonable retention policy, especially pre ai_v8 eras, as we are unlikely
> to need anything from them as there wasn't a 'novel' outcome, more getting the pattern established
> and us able to make meaningful progress."

That is the licence for tier 4, and it is quoted in the code, in the report, and here — three
places, one sentence, so nobody has to reconstruct the reason from a GB figure.

---

## 2. The tiers

First match wins, and the ORDER is the safety order: a run that is LIVE-adjacent is never graded
on its era.

| tier | who | what happens to it |
|---:|---|---|
| **0 · LIVE** | a launcher/trainer process names it · its training output was written within `--recent-days` (7) · its run dir is a SYMLINK into a launcher worktree · **or it is a (transitive) model-graph ancestor of any of those** | **nothing** |
| **1 · REFERENCED** *(any era)* | the **BASELINE REGISTRY** names it (§3.3 — and its named FILES are kept at every tier) · a committed **script** names it · a committed **measurement artifact** names it · the ledger's last **1500** lines name it · **another run's model graph names it** · it carries a **REVIEW HOLD** (§5) | the standing policy + the snapshots rule (a HOLD suppresses the plan entirely) |
| **2 · v9+ CLOSED** | era ≥ ai_v9, nothing reaches for it | the standing policy + the snapshots rule |
| **3 · v8 CLOSED** | era = ai_v8, nothing reaches for it | first + last + `latest.txt` pin, **no every-10th stride** + the snapshots rule |
| **4 · PRE-v8** | ai_v5 / ai_v6 / ai_v7, and the un-prefixed `run_2026*` dirs dated before 07-17 | **AGGRESSIVE keep-list** (§4) |

**Recency is part of tier 0, not tier 1.** An arm launched between ledger updates has no other
signal, and the tiered policy retires the blanket "v8-era" protection the standing policy leans on
— so the recency window has to be the thing that catches a fresh family. On this archive that is
69 of the 100 tier-0 runs.

**An era the tool cannot read is graded as tier 2, the GENTLEST closed tier — never tier 4.** An
unreadable name must not be able to buy an aggressive plan. `era_of()` reads the generation from
the run name, then the lineage, then a date (`run_YYYYMMDD…` or a trailing `_MMDD`) against the
2026-07-17 cutoff (`ai_v8_01_zarch_film_0717` is the first ai_v8 run); anything else is `unknown`.

---

## 3. The two mechanisms that are NEW, not just re-graded

### 3.1 The reference graph reads `original_command`, not only `lineage`

The `lineage` block postdates most of the archive. **The live `v8rep_*` replication arms carry
`lineage: null`** — their fork parent (`ai_v8_04`) and their three teachers (`ai_v8_09` / `_06` /
`_13`) exist only inside `original_command`. And the ledger tail does not name those teachers
either (measured: the last 1500 lines name 12 runs; none of them is a teacher). Under the standing
policy the blanket "v8-era" rule saved them by accident. The tiered policy retires that blanket, so
it has to see the edges properly.

`build_model_graph` therefore unions `lineage_refs` with `argv_refs`, which scans each run's
`metadata.json` (`original_command`, `launcher_command`, `cli_args`) for any other run's name and
classifies the edge by the nearest preceding flag: `--model` ⇒ a FORK edge, `--stable-opponents` /
`--exploiter` / `--exploiter-ladder` / `--warmstart-consensus` ⇒ a POOL-SOURCE edge,
`--distill-teacher` / `--win-prob-pbrs-source` ⇒ a TEACHER edge, anything else ⇒ a bare mention.
That distinction is load-bearing for the snapshots rule: **a teacher is not a fork parent — it does
not seed a pool.**

### 3.2 An artifact that ENUMERATES the archive is not a reference to anything in it

**The first tiered run put all 118 non-tier-0 runs in tier 1 and graded nothing.** The cause is
self-referential: the committed census `archive_grooming_dryrun_2026-09-06.md` names every run in
`models/` by construction, and so does every lineage / sidecar sweep. A retention policy that reads
its own previous report as evidence can never close a run.

So `is_bookkeeping()` excludes a committed file by BASENAME prefix — `archive_grooming_`,
`fh_lineage`, `folding_history_`, `sidecar_audit`, `run_inventory` — from BOTH the artifact and the
script protection paths. Matching the basename means a future dated copy is covered without an edit.

---

### 3.3 A registry-NAMED baseline survives every tier — file by file

**`designs/baselines.json` (`gen3_baselines_registry_v1`) is a third reference source, and the
FILE-level half of it is the load-bearing one.** A baseline is the thing a result is read against;
a NAME is only worth having if it still resolves a year from now, which is exactly what a retention
pass can quietly end. So `archive_grooming_tiers.baseline_protected()` reads the registry and the
policy uses it twice:

* **RUN level** — a registry-named run is **tier 1 REFERENCED**, with the reason naming the file.
* **FILE level** — every file the registry NAMES is added to the keep-list inside
  `assert_safe_tiered()`, the ONE choke point every tiered plan passes through (the tier-4 plan, the
  pool plan, and `apply_plan` before execution). So the protection covers every tier by
  construction rather than by remembering, and it is a REFUSAL — a plan that names such a file
  raises `TieredRefusal` rather than being silently trimmed.

**MEASURED 2026-09-06, and the measurement is what says which half matters.** All five runs the
seeded registry names are ALREADY tier 1 by the committed-file scan (2–59 scripts and 15–228
measurement artifacts each), so the run-level protection buys nothing *today* and is belt-and-braces
— it stops a baseline's tier depending on incidental script mentions a later cleanup can remove.
The file-level keep-list is different: tier 1 applies the STANDING rule, which keeps first + last +
every 10th checkpoint + the `latest.txt` pin, and `untaught_meter_opponent` is
`snapshots/snapshot_000024000000.zip` — a POOL file that no checkpoint rule covers and that the
snapshots rule (§5) keeps only while some fork or committed script happens to name the run.

The registry is also not free-riding on the committed-file scan: that scan classifies an origin as a
SCRIPT (by extension) or a MEASUREMENT ARTIFACT (by the `measurements/` prefix), and
`designs/baselines.json` is neither.

**A broken registry degrades to NO protection rather than a crash** — a dry run must not be what an
unreadable registry takes down, and `python -m main.baselines check` is what reports it. That
degradation is safe only in the wrong direction (it makes a plan *more* aggressive), so the count is
reported rather than silent. Gates: `archive_grooming_tiered_test.py` → *the BASELINE REGISTRY* (5
tests, including the refusal and the tier-4 keep-list entry).

---

## 4. Tier 4 — the aggressive keep-list

**The keep-list IS the policy**; everything not on it is deleted. Per run:

**KEEP**
- `metadata.json`, `model_config.json`, `latest.txt`, `eval_results.jsonl`;
- every other run-root file that is not a `.zip` and not a `.log` (`command.txt`,
  `team_winrates*.json`, `capacity_battery.json`, …) — kilobytes of bookkeeping;
- **every checkpoint `.json` SIDECAR**, at either layout — the per-checkpoint record (step count,
  git hash, `pin_history`, the eval at that step) is what makes the era's training record readable
  after its weights are gone. **Tier 4 deletes the WEIGHTS, not the LOG of them**;
- `tb/`, `tb_imgs/`, `snapshot_ladder/`, `elo/` — the era's training record, never thinned
  (all `tb/` in the whole archive is 0.27 GB);
- **ONE model file: whatever `resolve_model_ref` picks**, plus `latest.txt`'s target if that is a
  different file, plus any file another run's model graph resolved to.

**DELETE** — every other checkpoint (`checkpoints/` and the legacy run root alike), `best_model/`,
`snapshots/`, `eval_traces/`, `crashes/`, `stalls/`, `launcher_child.log`, and every other
subdirectory (`.eval_runs/`, `teacher_persist/`).

**The one kept model goes through the ONE choke point.** `resolve_model_ref` is CALLED, never
re-derived, so the surviving file is by construction the file a bare `--distill-teacher <run>` /
`--stable-opponents <run>` would load today (`gen3_last_snapshot_resolution_v1`: `latest.txt` →
highest-step checkpoint → `final_model*.zip` → `best_model/` LAST). The **rung** it fired on is
recorded per run in the report, because a bare run dir has meant different files at different
times. Measured on this archive: 34 tier-4 runs resolve on `latest_txt`, 1 on `highest_checkpoint`.

**A tier-4 run whose final model does not resolve gets NO plan.** One did — `warmstart_generic_0715`
(43 MB, a bare `warmstart_generic.zip` with no run layout) — and it is listed as REFUSED rather
than emptied on a guess.

### What tier 4 costs, stated plainly

**A tier-4 run becomes un-probeable except at its final checkpoint.** That costs less than it
sounds. Root `CLAUDE.md` records that on 2026-08-13 **79 of 79 archived runs could not be re-loaded**
at the then-current architecture, and the drift has only grown (the v96 signature bump added to
it) — so every model-loading prober view (`analyze` / `lookahead` / `better-line` /
`replay-counterfactual` / `probe`) already returns an `ArchDriftError` on these runs. What survives
tier 4 is exactly what still works on them: `tb/`, the ELO ladder, `eval_results.jsonl`, the
checkpoint sidecars, and every model-free prober view (`scan`, `triage`, `turns`, `falsify`,
`calibration`).

---

## 5. The snapshots rule (tiers 1–4; tier 0 is exempt)

A self-play pool is **kept only** when:

- **some run forks this run** — a fork auto-seeds its parent's pool, so the zips *and*
  `summary.json` / `win_rate_vs_bots.txt` / `model_config.json` are load-bearing (the zips alone
  read `self_play_fraction=0%`); a kept pool missing any of that metadata is FLAGGED in the report;
  **or**
- **a committed script names the run** (the conservative reading of "referenced as a
  `--stable-opponents` / `--exploiter` / pool source by a committed script").

Otherwise `snapshots/` goes **whole**. Every run's decision, with its one-sentence reason, is in the
report's *Every pool decision, per run* table.

Measured: **26 pools KEPT (12.96 GB) · 50 FREED (19.09 GB)**; 35 runs have no pool; 107 are tier 0
and exempt.

**Thinning a KEPT pool is PROPOSED, never planned.** Every 4th snapshot + the newest would free a
further **8.70 GB**. It is reported so that number is a decision the owner can take rather than a
discovery someone makes later; no kept pool loses a byte under this policy.

---

## 6. The five review runs — verdicts

The 2026-09-06 standing dry run flagged five CLOSED runs the ledger names *outside* its tail window.
Each was read against `ledger.md` and the committed tree. Three are HELD, two are RELEASED.

**A HOLD suppresses the plan entirely** — it is not a softer tier. The reason a run is held is that
we are not certain which of its files its banked claim rests on, and grooming "only" its
`eval_traces` while that is unresolved is the same bet with a smaller stake. **A hold is released by
BANKING the artifact, not by re-reading the ledger.** The holds live in
`archive_grooming_tiers.REVIEW_HOLDS`, each with its finding, so "we looked at it" is a fact in the
code rather than a claim in a report.

| run | tier | verdict | why |
|---|---:|---|---|
| `ai_v9_48_G1_action_0826` | 1 | **HELD** | The program's **first POSITIVE distill arm** (pooled +0.0398 [+0.016,+0.064] z=+3.29; G2−fdB +0.0762 z=+6.01, `ledger.md:4943`ff). **No committed artifact carries the per-arm numbers** — `fold_capacity_telemetry.md` has fdA/fdB/fdC/fdE rows and no G1/G2 row, and no `ai_v9_48_*_endofrun.json` exists. The claim rests on this run's `eval_results.jsonl` + `eval_traces`. *Bank an endofrun artifact and the hold can be released.* |
| `ai_v9_26_baitent_probe_0823` | 1 | **HELD (partial)** | The capacity baseline IS banked (`capacity_battery.md:153`ff), but the P2 bait-entropy per-leg result (boost_eff 3.0, flagged 5.9%, B1 0.056→0.229, leg-vs-leg z=−2.55, `ledger.md:3722`) is in no committed artifact, and the Baton-Pass GIGO reproducer decodes `loss_s0_003_states.npz` from this run's traces (`ledger.md:3595`). `ladder_readiness.md:269` also loads its `legB_final_model.zip`. |
| `ai_v9_45_fdF_p1_0826` | 1 | **HELD** | Not a data dependency — the numbers ARE banked (`designs/ai_v10/design_advantage_gated_distillation.md:459-467` carries the entropy 0.892→1.354 dissolution and the subtraction rule). Held because the ledger records an explicit **owner decision to preserve it as the entropy-dissolution SPECIMEN** (`ledger.md:4937`). |
| `ai_v5_11_tail2_53m_0611` | 4 | **RELEASED** | Its only ledger claim (`value_share` ~0.6, ELO ahead of `ai_v5_10`, `ledger.md:85`) lives in `tb/` and the checkpoint sidecars — **and tier 4 keeps both**, so the claim survives its own tier. The two downstream findings are already banked in `levers/attack_type_mismatch.md:15-31` and `designs/ai_v6/design_differentiable_damage_op.md:39-48`. Frees 2.97 GB. |
| `ai_v5_12_bias_05_N_0612` | 4 | **RELEASED** | `ledger.md:87` asserts **no number at all** ("LIVE (bias-redesign)"), and its one measured contribution is banked at `levers/attack_type_mismatch.md:33-34` (immune-pick V≥0 Δ=+0.0158, z=6.8). The cleanest release of the five. Frees 1.32 GB. |

The tiered run surfaces **10 further** runs in the same review bracket (a plan, plus a ledger
mention outside the tail). They are listed in the report and are the read-before-apply list.

---

## 7. The census, and the exact commands

Measured 2026-09-06 over 218 runs / **257.4 GB** (242.8 GB physically under `models/`, 14.6 GB in
8 symlinked launcher-worktree run dirs held out by default):

| tier | runs | GB now | GB freed |
|---:|---|---:|---:|---:|
| 0 · LIVE | 100 | 112.33 | 0.00 |
| 1 · REFERENCED | 69 | 97.58 | 45.59 |
| 2 · v9+ CLOSED | 1 | 2.04 | 1.63 |
| 3 · v8 CLOSED | 12 | 11.97 | 8.65 |
| **4 · PRE-v8** | **36** | **33.50** | **32.01** |
| **total** | **218** | **257.43** | **87.88 (34.1%)** |

Also: 2,685 planned entries · 3 runs vetoed because a committed file names an exact planned path
(`ai_v8_14_distill3_0725`, `ai_v9_31_tock1_k4_0824`, `ai_v9_44_tock2_v8shape_0825`) · 1 tier-4
refusal · 10 runs in the review bracket · **377 legacy root-level checkpoint zips** reached that the
standing policy structurally cannot see.

**Read `designs/research_state/measurements/archive_grooming_tiered_2026-09-06.md` before applying**
— in particular its *REVIEW BEFORE APPLYING* section.

```bash
# DRY RUN (what was done here — writes only the two report files)
cd /home/goodlad/dev/gen3ai && \
export PYTHONPATH=$PYTHONPATH:src && \
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
  designs/research_state/measurements/archive_grooming_dryrun.py --policy tiered

# APPLY the tiered policy — 87.9 GB
cd /home/goodlad/dev/gen3ai && \
export PYTHONPATH=$PYTHONPATH:src && \
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
  designs/research_state/measurements/archive_grooming_dryrun.py --policy tiered --apply

# APPLY the gentler STANDING policy instead — 18.8 GB, checkpoints/ + eval_traces/ only
cd /home/goodlad/dev/gen3ai && \
export PYTHONPATH=$PYTHONPATH:src && \
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
  designs/research_state/measurements/archive_grooming_dryrun.py --apply
```

**Run it from the MAIN checkout** — `models/` exists only there. The default `--out-prefix` is
named after the policy, so a tiered pass can never clobber the standing report or the other way
round. `--follow-symlinked-runs` opts the 8 worktree-symlinked runs in, after confirming their
targets are still the ones you mean.

---

## 8. What is guarded, and how

- **`_assert_safe`** (tiers 1–3) — every planned path is inside `checkpoints/` or `eval_traces/`,
  on the REALPATH of both sides, and never a protected subdir or run-root file.
- **`assert_safe_tiered`** (tier 4, and the pool deletions) — tier 4 deliberately reaches outside
  those two directories, so the guarantee is restated positively instead: nothing escapes the run
  dir, and nothing the keep-list names (or lives under a kept path) is in the deletion set.
- Both are re-checked **at execution time**, under the rules of the tier that produced the plan,
  never a weaker set.
- **The named-path VETO is shared by both policies** — a committed file or the ledger naming an
  exact path a plan would delete drops that run from the deletion set entirely.
- **Symlinked run dirs are held out by default** in both policies; a deletion "in `models/`" would
  otherwise land physically under `.claude/worktrees/`.
- **Tests**: `archive_grooming_dryrun_test.py` (34, the standing policy) and
  `archive_grooming_tiered_test.py` (47) — synthetic trees under `tmp_path`, which is also the only
  place `--apply` is ever exercised. Run them with:

```bash
python -m pytest designs/research_state/measurements/archive_grooming_dryrun_test.py \
                 designs/research_state/measurements/archive_grooming_tiered_test.py -q
```

(`pytest.ini` sets `testpaths = src tools`, so a test under `designs/` is not collected by the
routine gate and is run directly.)

---

## 9. Standing conventions this replaces

The 2026-08-09 convention (first + last + every 10th checkpoint, `prober.groom 3/1` on traces,
never thin `tb/`) is not retired — it **is** the standing policy, and it is the body of tiers 1–3.
What this document adds is the grading, the snapshots rule, and the pre-v8 keep-list. Disk is still
not the constraint (a ~1 TB volume with hundreds of GB free); this is housekeeping under an explicit
owner instruction, not need — say so before proposing any further deletions.
