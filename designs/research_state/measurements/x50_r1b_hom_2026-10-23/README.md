# X50 rung 1b use (b): the HUMAN OPPONENT MODEL, built and read (2026-10-24)

**Status: BUILT and FROZEN; R-b1 READ: LEARNED** (Δ NLL −0.582 nats [−0.599, −0.565]; §R-b1 below).
The HOM is X26's network fine-tuned from P\* on human choices (actor rated >= 1500), as
registered in [`design_x50_rung1b_human_opponent_model.md`](../../../endstate/design_x50_rung1b_human_opponent_model.md)
§3-§4 (`42f1d9a7`). It is search's opponent-choice likelihood judge ONLY (owner's yes, 2026-10-23); the policy never reads
it. **HOM sha256 `76e29a0176fd450d95496b4722d898e38ca2b98a50de0fc9147a2dea671bb36f`**, committed here
([`hom_sidecar.json`](hom_sidecar.json)) before any held-out row was read. R-b1 and R-b2 read this file and refuse any
other.

| file | what |
|---|---|
| `run_hom.sh` | the driver: `genc`, `train` (extraction), `heldout`, `compact`, `hom` (train + export), `rb1`, `rb2` |
| `genc.json` | G-ENC at the extraction commit: PASS |
| `hom_rows_summary.json` | the rows: counts, the exclusion by heldout team, the held-out extraction |
| `hom_sidecar.json` | the model: init, rows manifest sha256, recipe, validation history, the HOM's sha256 |
| `rb2_run.py`, `rb2_read.py` | R-b2's scoring (two judges, one process, at L3's code `8fe4fb89`) and its read |

Archive: `~/gen3ai_archive/x50_r1b/` (`train/`, `heldout/`, `hom/rows/`, `hom/model/`, `hom_build/` logs).

## The data (design §3)

- **G-ENC (F-1b-6), at the extraction commit `2bbdffa7d`, before any extraction: PASS.** On R1's 2,306 traces, the
  reader's full read equals the rows and masks the trainee played on, bit for bit: 2,106 equal, 9 short by the
  unrecorded last row of a 250-turn game (R1's known case), and 191 skipped because the stream holds a refused choice
  (R0's 191). No DIFF, no error.
- **The training dates, 2026-05-18 → 2026-09-20 of the spectator manifest:** 368,997 logs in 738 parts, 0 part errors.
  - **Team-memory exclusion (F-1b-2): 70,010 battles (19.0 %) excluded, both seats.** Every one of the 72 heldout teams
    is hit, from 7 to 10,605 battles each. A species-only reveal fits, so most hits are common cores, not copies (the
    rule over-excludes by design).
  - Of the 298,987 kept logs: 597,920 seats read, **594,266 completed** (C-GAME). Excluded: no Smogon prior 2,407; no
    consistent team in 32 draws 721; revealed facts no legal set holds 500. **Refused: 27 completed seats** (22 `OwnError`,
    4 reader refusals, 1 label mismatch) plus 54 spectator reads: 0.014 % of seats, under the 1 % stop.
  - Own-action labels: 15,180,478 of 16,808,248 completed decisions (all ratings).
- **The HOM's rows** (mode 2, an own label, >= 2 legal actions, the actor rated >= 1500): **1,762,236 train rows** (100
  shards) and **38,303 validation rows** (by `sha256(battle id) mod 50 == 0`). Rows manifest sha256 `44522bf3…b4ea`.
  The design estimated ≈ 2.0-2.4 M; the exclusion took about a fifth (F-1b-8).
- **The held-out dates 09-21..23, re-extracted at the Mimic fix (`3f5d4b4da`):** 7,413 logs, 14,770 completed seats
  (R0's 14,756 + the 14 Mimic seats), **0 refused**; 389,709 own-action labels. Nothing trained or selected on them.

## The model (design §4)

Init P\* (`rb_x26_s2601` @ 280,000,170, sha256 `814823c5…91ed`). The registered recipe, unchanged: the masked policy
softmax's cross-entropy at the human's action; AdamW, lr 1e-4 cosine to 1e-5 over 3 epochs, weight decay 1e-5, batch 512,
clip 0.5, fp32 on the CPU, seed 51; validation at step 0 and every quarter epoch, the lowest kept, stop after 2
consecutive rises. **10,346 steps, all run; no early stop; the kept model is the last.** The trunk and every head the
logits read moved; the win head and the flat intent head did not (no gradient).

| step | 0 (P\*) | 862 | 1,725 | 2,588 | 3,450 | 5,174 | 6,898 | 8,622 | **10,346** |
|---|---|---|---|---|---|---|---|---|---|
| validation NLL (nats) | 1.766 | 1.260 | 1.228 | 1.209 | 1.197 | 1.179 | 1.168 | 1.159 | **1.156** |

The exported zip reproduces the last validation NLL exactly (1.155700). The validation rows are training-date battles,
so this selects the model and reads nothing held out.

## R-b1: does the HOM out-predict P\* on held-out humans? (§8.2, a GATE) — **LEARNED**

Rows: the held-out dates 09-21..23, completed, (b)'s label, >= 2 legal actions; both models through the judge's masked
softmax in float64. The read refused any HOM whose sha256 differed from the committed sidecar. Every number is in
[`rb1_read.json`](rb1_read.json). 95 % CIs: battle-cluster bootstrap, 2,000 resamples, seed 50.

| stratum | rows | battles | Δ NLL (HOM − P\*) | NLL HOM / P\* | top-1 HOM / P\* |
|---|---|---|---|---|---|
| **PRIMARY: actor >= 1500** | 69,171 | 1,481 | **−0.582 [−0.599, −0.565]** | 1.148 / 1.730 | 0.567 / 0.460 |
| voluntary switches (>= 1500) | 15,807 | 1,454 | −0.874 [−0.912, −0.839] | 1.642 / 2.516 | 0.389 / 0.307 |
| moves (>= 1500) | 46,881 | 1,481 | −0.490 [−0.510, −0.470] | 1.006 / 1.496 | 0.626 / 0.505 |
| forced replacements (>= 1500) | 6,483 | 1,379 | −0.537 [−0.575, −0.502] | 0.969 / 1.506 | 0.578 / 0.512 |
| all ratings | 384,601 | 7,335 | −0.600 [−0.606, −0.593] | 1.223 / 1.823 | 0.538 / 0.435 |
| rating < 1300 | 171,631 | 4,193 | −0.602 [−0.611, −0.593] | 1.291 / 1.893 | 0.513 / 0.417 |
| 1300–1499 | 136,198 | 3,249 | −0.609 [−0.620, −0.598] | 1.168 / 1.777 | 0.559 / 0.448 |
| 1500–1599 | 47,723 | 1,226 | −0.578 [−0.600, −0.558] | 1.156 / 1.734 | 0.563 / 0.460 |
| 1600–1699 | 16,203 | 435 | −0.595 [−0.626, −0.567] | 1.118 / 1.713 | 0.579 / 0.466 |
| >= 1700 | 5,245 | 128 | −0.583 [−0.644, −0.523] | 1.169 / 1.752 | 0.567 / 0.442 |

**Verdict under the registered rule: LEARNED** (the CI's upper end, −0.565, is far below −0.05). R-b2 is unlocked.

**Reading it.**
- The gain is ten times the bar and holds in every stratum and rating band, the largest on voluntary switches
  (−0.87 nats), where R1 found the human gap. Validation (training dates) read 1.156; the held-out humans read 1.148, so
  there is no sign of drift over the three later days.
- **It is a gate, not the evidence (F-1b-5).** Both models read completed rows, and only the HOM trained on them: part
  of the gain may be the completion's signature (F-1b-4). R-b2 reads true rows.
- Reference only, a different conditioning (R0's X26 intent head on H_c, predicting the same humans from the
  OPPONENT's view without their team): 1.785. The HOM, which sees the actor's completed own side, reads 1.148.

## FINDINGS (standing rule 7)

1. **The trainer ran from the worktree before it was committed.** The sidecar's `commit` (`2bbdffa7d`) is the
   extraction's; `hom.py` as run (sha256 `8b9e5790…3ad8`) is the file this commit adds, unchanged.
2. **CPU training is not bit-deterministic.** Two from-scratch runs of 2 steps differ at the 7th digit of the validation
   NLL (a multi-threaded CPU reduction). A resume therefore continues the same plan, not bit for bit the same numbers.
3. **Thread count.** Training began with 6 intra-op threads. When other agents' heavy jobs shared the agent cores,
   OpenMP barrier stalls cut throughput 8-20× (a chunk ran 62 steps in 38 min). From step 6,739 it ran with 3 threads
   (`HOM_THREADS`, speed only). Wall time: ≈ 8.7 h, against ≈ 5 CPU-hours estimated.
4. **The driver was edited while bash ran it.** Bash reads a script as it runs, so the extraction's chained call hit a
   syntax error after its last step had finished (no part lost: all 738 parts, 0 errors). Later edits waited for the
   driver to stop.
5. **The team-memory rule excludes far more than copies** (above). It is the registered rule; R-b2 reports the bank's
   states split by whether their team had training battles removed.
6. **R-b1's forward cache first landed inside the worktree** (`measurements/rb1_cache/`, the default beside `--out`); it
   was moved to the archive before anything was committed, and `--cache` now defaults to the archive.
7. **Process.** CPU only; `models/`, the corpus, the bank and `data/` were only read. Kills: my own driver shells by
   PID; one `pkill -P <my PID>` took the bounded wait child of my own relaunch.
