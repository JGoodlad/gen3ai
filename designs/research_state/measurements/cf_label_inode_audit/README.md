# CfLabelBuffer inode-reuse audit (boundary `a7627744`, 2026-09-30)

**Verdict: no past data affected, and no banked claim changes.** Only three runs ever fed a cf label to
training. In all three, the recreated-shard path the fix closes could not be reached, and nothing on
disk or in TensorBoard shows a missing row. The audit found a **separate, larger defect**: the
`ai_v12_12_ladder_cflabels` arm never received a single label (§4).

Reproduce (read-only on `models/`): `PYTHONPATH=src python designs/research_state/measurements/cf_label_inode_audit/audit.py > audit.json`.

## 1. The bug, and the exact condition that triggers it

Before `a7627744`, `CfLabelBuffer` kept a byte offset for each file, keyed on `(name, inode)`. If a file
with the same name appeared, got the **same inode number**, and the saved offset was greater than zero,
the buffer skipped the new file's first `offset` bytes without any warning. On ext4 a freed inode number
is often handed straight to the next new file, so a delete followed by a recreate can keep the number.

Three things must all be true to trigger it: a label filename is **reused**; the new file gets the
**old file's inode number**; and the buffer has **already consumed** part of the old file.

## 2. Which runs had the plumbing live

| Population | Count | Evidence |
|---|---|---|
| `model_config.json` records a non-zero cf coefficient | **98** | 97 v9-era runs (plus dry-run/aborted dirs) with `cf_evidential_coef 0.05` (`cf_twin_coef` / `cf_shadow_coef 0.1`, all head-only); `ai_v12_12_ladder_cflabels` with `cf_winprob_coef 0.5` |
| **ever INGESTED a label** (`cf/labels_ingested_total > 0` in the run's OWN TB files, not inherited ones) | **3** | `ai_v9_29_rev1_0823`, `ai_v9_34_tick1_0824` (forked from it), `ai_v9_37_tick1_dosext_0825` (forked from that) |
| had cf scalars but ingested nothing | 92 | an empty `cf_labels/`, because no producer ever ran |

The other ~95 runs are **UNAFFECTED by construction**: with an empty buffer the fold has nothing to
read, so dropping rows is impossible. The v9 descendants inherit the three runs' trained side heads,
but they never ingested a row themselves.

In the three consuming runs, labels reached only **detached side heads**: `cf_evid_head`, twin
heads B/C, and the shadow critic, which reads the same rows' `mc_return` field. All of them are
`--cf-head-only` with their grad shares logged at exactly 0.0. `cf_winprob_coef` was 0. So **the trunk, the policy and V never saw a
label**, and even a real drop could only have affected the R1 side-head reads.

## 3. Were shards recreated? No, and the path could not be reached

**By construction.** The only producer that wrote into these dirs is `cf_producer`.
`write_label_batch` writes each batch to a **new, never-reused name**,
`labels_cf_producer_<step>_<seq>.jsonl`. It writes a `.tmp` file first and then calls `os.replace`. So:

- The name is never reused. `seq` is persisted in `cf_producer_state.json` across producer restarts.
- Even if a name were reused, the `.tmp` is created while the old file still exists, so it cannot get
  the old file's inode number.
- The rename is atomic. The buffer only sees complete files, which it reads in one poll.

**On disk (checked):**

| Run | Label files | Distinct `seq` | max `seq` | Producer `seq` | Rows on disk | Producer `labels_total` |
|---|---|---|---|---|---|---|
| `ai_v9_29_rev1_0823` | 2,200 | 2,200 | 2,200 | 2,200 | 6,600 | 6,600 |
| `ai_v9_34_tick1_0824` | 2,448 | 2,448 | 2,448 | 2,448 | 7,344 | 7,344 |
| `ai_v9_37_tick1_dosext_0825` | 1,342 | 1,342 | 1,342 | 1,342 | 4,026 | 4,026 |

The `seq` values are unique and gap-free, and they equal the producer's own counter. Rows on disk
equal the rows the producer says it wrote. **No file was ever overwritten or recreated.**

## 4. Estimated dropped fraction: 0 by construction, and at most ~1–4% even without that argument

Each trainer process starts with an empty offset map and re-reads every file on disk. So in each
TensorBoard segment:

- rows parsed = `ingested + expired_at_ingest + skipped`
- that sum lies in `[max(ingested, expired) + skipped, ingested + expired + skipped]`

Compare that range with the rows on disk at the segment's last log (file mtime ≤ that log's wall time):

| Run, segment (steps) | Disk rows | Parsed range | Unexplained, at most |
|---|---|---|---|
| rev1 0.2–6.0M | 375 | 363–369 | 12 |
| rev1 6.3–11.4M | 1,344 | 1,272–1,557 | 72 |
| rev1 11.7–16.0M | 3,240 | 3,180–4,104 | 60 |
| rev1 16.2–20.8M | 5,019 | 4,980–5,817 | 39 |
| rev1 21.1–25.1M | 6,597 | 6,537–7,131 | 60 |
| tick1 25.3–29.5M | 2,244 | 2,136–3,009 | 108 |
| tick1 29.8–33.6M | 4,680 | 4,608–5,568 | 72 |
| tick1 33.9–35.1M | 6,018 | 5,925–6,330 | 93 |
| dosext 35.3–39.5M | 3,147 | 3,023–4,208 | 124 |
| dosext 39.8–40.1M | 3,714 | 3,624–3,807 | 90 |

Every segment but the first is consistent with zero drops. The first is 6 rows (2 files) short of its
lower bound. Files written between the last poll and the log timestamp explain that; the files are
3 rows each, so 6 rows is exactly two files. The last column is a loose **upper bound**, dominated by
that poll-versus-log timing slack. It is not a drop estimate. **The dropped-row estimate is 0**; the
bounds alone cap it at 3.2% of the first segment's rows and below 4% in every segment.

The banked R1 figure "~2,646 of 6,600 rows ever ingested" (ledger 2026-08-24 · *R1 FIRST READ*)
equals the sum of rev1's per-segment `ingested` (6 + 285 + 924 + 837 + 594 = 2,646). The shortfall
is lag-bound **expiry**, as that entry attributed it (the duty-cycle era). It is not silent dropping.

## 5. Claims

| Claim (ledger / UNDERSTANDING) | Rests on | Standing |
|---|---|---|
| R1 FIRST READ, 2026-08-24: B−A +0.065, C−B −0.036, C−A +0.029; evidential CLEAN NULL (`r1_first_read_ai_v9_29.md`) | rev1's twin/evid heads, trained through the buffer | **UNAFFECTED**: 0 recreated shards, the ingest accounting reconciles |
| R1 DOSE READ (`r1_dose_read_ai_v9_37.md`) | dosext's (+tick1's) heads | **UNAFFECTED**: same evidence |
| REV-1 HOUR-2 INCIDENT, 2026-08-23 (label path starved ~100×) | expiry counters | **UNAFFECTED**: an expiry mechanism, already correctly attributed |
| tick1 / dosext / every other v9 policy or strength claim | trunk and policy | **UNAFFECTED**: labels reached only detached heads (grad share 0.0) |
| cf label factory COST MODEL, G0 BIAS MAP | offline `cf_audit` / producer; no buffer | **UNAFFECTED**: the buffer is not in the path |
| critic-ladder `cflabels` reads, 2026-09-09 (+ the 2026-09-12 L2 leaf row 0.485) | `ai_v12_12_ladder_cflabels` | **UNAFFECTED by this bug** (0 rows were ingested, so none could be dropped), **but see §6** |

**No past decision needs revisiting on account of `a7627744`.** It is a pre-data boundary for every
reader: no banked number passed through the defective branch.

## 6. FINDING (separate from the inode bug): `ai_v12_12_ladder_cflabels` trained with NO cf labels

The arm launched with `--cf-records --cf-winprob-coef 0.5`, but no label producer ever ran for it.
Four pieces of evidence:

- `cf_labels/` is empty.
- There is no `cf_producer_state.json` and no producer log.
- `cf/labels_ingested_total`, `cf/buffer_fill` and `cf/rows_sampled` read **0 at every point** in
  both TB files, from 0.2M to 10.0M.
- `train/cf_loss` was **never written**, so the fold never ran.

Its 522 `cf_records/` ring files were written and never consumed. The launcher does not start
`cf_producer`, which is a separate out-of-process program. The launch passed the duty-cycle guard,
which checks cadence and cannot detect a missing producer. The ledger's 2026-09-09 *OPS* entry reads
"no expired/starved line … the first live evidence the fix holds". That is the vacuous-check shape
the 2026-08-23 incident already recorded, because 0 ingested also produces no expired line.

**So the `cflabels` arm is a control run plus `--cf-records`, the 40/40/10 forensic quota and a
different commit (377a5aa1 vs f3502568). It is not a cf-label arm.** Standing of its claims:

- Its conditioning, resolution and L2 leaf rows (all NOT DETECTED after the quota-matched retraction)
  are **not evidence about the cf-label lever at all**. They should read "lever never engaged".
- The standing "head reads unbiased against its own continuation where the control is pessimistic
  (Δ +0.040 [+0.007, +0.071])" **cannot be the lever's effect**. It is a difference between two runs
  with the same objective, so it is a noise/commit/quota draw, and is within the later-measured
  replicate floor (identity-bias floor 0.070).
- The UNDERSTANDING sentence "a null here is evidence about the class only at 10M" should read "no
  evidence either way; the class is UNTESTED".

The cf continuation-label lever at production scale is **untested**, not null. Class fix (not done
here): a FATAL_CONFIG guard or live-watch STOP row when a cf coefficient is live and
`cf/labels_ingested_total` is still 0 after N cycles, or have the launcher own the producer.
