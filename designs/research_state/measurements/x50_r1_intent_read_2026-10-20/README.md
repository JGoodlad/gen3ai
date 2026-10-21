# X50 R1: X26's intent head on human opponents (registered and read 2026-10-20)

**Verdict under the registered rule: GAP.** X26's opponent-intent head predicts human ladder players (rated ≥ 1500)
worse than it predicts its own pool snapshots, through the same spectator pipeline: **Δ NLL(H − S) = +0.188 nats
[+0.164, +0.213]**. The 95 % CI is entirely above the registered +0.05 margin. Top-1 is 33.8 % on humans against
38.9 % on snapshots (−5.1 pp [−6.2, −3.98]). So human labels carry signal the self-play labels lack. The next step
under the registration is a training-use design, registered separately. This unit runs no training.

**Registration:** [`../../../endstate/design_x50_rung1_intent_labels.md`](../../../endstate/design_x50_rung1_intent_labels.md)
§1–§5, shipped before the extractor existed (`6888a5ec4`). Extractor: `a3b37d8a3`. Driver: `run_r1.sh`. Read:
`src/main/x50_r1/read.py`. Every number below is in `r1_read.json`. **Model:**
`models/rb_x26_s2601/checkpoints/checkpoint_380000195_steps.zip`, CPU, read-only.

## Results

Every population was read through the same spectator pipeline: the own side is public-only, and the decisions are
synthesised. Supervised rows only, Hidden Power excluded. The 95 % CIs come from a battle-cluster bootstrap (2,000
resamples, seed 50).

| population | rows | battles | NLL (nats) | top-1 | uniform floor NLL / top-1 | presence-only NLL / top-1 | switch share |
|---|---|---|---|---|---|---|---|
| **H** humans, labelled ≥ 1500 | 48,579 | 1,481 | **1.937** [1.921, 1.953] | **0.338** | 2.390 / 0.094 | 2.240 / 0.211 | 0.300 |
| **S** own snapshots (sentinels) | 29,953 | 1,122 | **1.748** [1.732, 1.766] | **0.389** | 2.347 / 0.100 | 2.145 / 0.225 | 0.259 |
| **B** the eight archetype bots | 15,063 | 1,182 | 1.682 [1.657, 1.708] | 0.409 | 2.368 / 0.096 | 2.082 / 0.280 | 0.058 |
| S-full (control: the training row) | 26,239 | 1,001 | 1.519 [1.502, 1.536] | 0.476 | 2.351 / 0.099 | 2.151 / 0.223 | 0.261 |
| B-full (control) | 14,027 | 1,112 | 1.545 [1.522, 1.572] | 0.490 | 2.370 / 0.096 | 2.087 / 0.278 | 0.057 |

| contrast | Δ NLL | Δ top-1 |
|---|---|---|
| **H − S (the decision input)** | **+0.188 [+0.164, +0.213]** | −0.051 [−0.062, −0.040] |
| H − B | +0.255 [+0.225, +0.286] | −0.071 [−0.087, −0.055] |
| pipeline penalty S − S-full | +0.229 [+0.206, +0.254] | −0.087 [−0.100, −0.075] |
| pipeline penalty B − B-full | +0.136 [+0.100, +0.172] | −0.081 [−0.101, −0.060] |

**Where the gap is: switches.**
- On a MOVE label, H is 1.692 and S is 1.620, a gap of +0.07 nats (top-1 0.410 vs 0.431).
- On a SWITCH label, H is **2.508** and S is 2.116, a gap of +0.39 nats (top-1 **0.170** vs 0.269). On human
  switches the head is worse than the uniform floor (2.438), and the presence-only floor beats it (2.208).
- Humans also switch more: 30.0 % of labels, against 25.9 % for the snapshots and 5.8 % for the bots.

**By the labelled human's rating** (all rated, Hidden Power excluded): < 1300 2.092; 1300–1499 2.001; 1500–1699 1.933;
1700+ 1.975 (n = 3,810, 128 battles). All rated bands together: 2.030 [2.023, 2.038], top-1 0.311. Stronger players
are a little more predictable, up to the thin top band.

**The label classes** (the share of supervised rows: seat / OTHER_move / slot / OTHER_species):
- H: 63.2 / 6.7 / 22.7 / 7.3 %;
- S: 69.8 / 4.2 / 22.1 / 3.8 %;
- B: 89.4 / 4.8 / 4.4 / 1.4 %.

The share of labels the head can address (not UNMODELED): H 99.8 %, S 97.8 %, B 99.0 %. Human choices fall outside
the belief's top-K seats (OTHER_move) and onto unseen species (OTHER_species) about twice as often as the snapshots'.

**Robustness.**
- Without the Struggle-placeholder rows: H 1.921, S 1.725.
- The Hidden Power stratum (H, ≥ 1500, a bare num-237 label): 1.600 on 1,015 rows. It is reported, not used.

## Extraction (counts from the part files under `/home/goodlad/gen3ai_archive/x50_r1_intent/`, 129 MB)

| | humans (09-21..09-23, the spectator manifest) | traces (X26, cycles ≥ 360M, 9 bots + 5 sentinels, `random` excluded) |
|---|---|---|
| logs / seats read | 7,413 / 14,826 (both seats) | 2,306 / 2,306 (the trainee's seat) |
| refusals | **0** | **0** (decision points = the trainee's real requests on every one) |
| prefix-guard checks | 14,826 passed | 2,306 passed |
| decisions | 432,486 | 80,801 spectator, 72,879 full |
| labels MOVE / SWITCH / UNKNOWN | 197,253 / 76,218 / 159,015 | 40,261 / 8,640 / 31,900 |
| bare Hidden Power labels | 10,038 | 3,042 |
| Struggle-placeholder decisions | 27,968 (6.5 %) | 5,858 (7.2 %) |
| full read skipped (a refused choice in the trainee's stream) | — | 191 battles (8.3 %) |

Wall time: extraction 3 min, read 16 min, each in one heavy slot. The rest of about 2 h end to end was the heavy-slot
queue.

## FINDINGS (standing rule 7)

- **F-R1-1 (the pipeline penalty is as large as the gap).** The spectator rows cost the head 0.23 nats on the
  snapshots' own battles: the public-only own side, the fabricated own spread (F-X50-4) and the synthesised
  legality. The H − S contrast holds the pipeline fixed, so the GAP verdict stands. But the head's absolute numbers
  on humans are pessimistic, and how the penalty interacts with the population is **UNVERIFIED**: no human battle
  has a full read. A training use must close or re-measure this. A Smogon-completed own side is the obvious lever.
- **F-R1-2 (the gap is a switch gap).** On moves the head nearly transfers (+0.07 nats). On human switches it is
  worse than uniform. This is the part human labels would teach. It is also where the census's production rules mask
  the most (switch co-occurrence; F-X50-7).
- **F-R1-3 (the bots barely switch).** 5.8 % of bot labels are switches, against 26 % for the snapshots and 30 % for
  humans. B is a poor stand-in for human switching, which is why H − S is the decision input.
- **F-R1-4 (the traces are loss-enriched).** That is by design (10 losses, 5 draws, 5 wins per opponent per cycle).
  S and B are therefore not random samples of X26's play. The humans are unselected.
- **F-R1-5 (Hidden Power).** 3.7 % of human labels are a bare Hidden Power. They are excluded from every primary row,
  and their stratum is reported only.
- **F-R1-6 (the Struggle placeholder).** It opens 6.5 % of human decisions: the lead before it moves, or a forced
  switch with no revealed bench. Dropping those rows moves H by −0.016 nats and S by −0.023 nats.
- **F-R1-7 (a plumbing dry run before the read).** One trace cycle (380M) and 60 human logs of 09-21 went through
  extract + read in a scratch directory before the registered run. They overlap the registered S, B and H data. Only
  the pipeline-penalty lines were looked at, and nothing in the design or the code's logic changed after it. A mypy
  annotation and the extraction of `read.rule` (the same rule) came after.
- **F-R1-8 (process).** CPU only. `models/`, the corpus and `data/` were only read. The output lives under
  `gen3ai_archive/`. The read used one checkpoint, so "X26's intent heads" (plural) is one head at 380M here.
