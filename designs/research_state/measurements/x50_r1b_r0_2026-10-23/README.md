# X50 rung 1b R0: is F-R1-1 fixed? (registered 2026-10-23, read 2026-10-23)

**Verdict under the registered rule: FIXED.** With the viewer's own side completed (C-GAME), the spectator pipeline
costs X26's intent head **P_c = NLL(S_c) − NLL(S-full) = +0.012 nats [+0.005, +0.020]** on X26's own battles. The
CI's upper end, +0.020, is under the +0.08 bar, and not within rule 8's 0.002 band. The completion removes
**+0.217 nats [+0.204, +0.230]** of R1's +0.229 penalty.

**R1's gap re-read beside it, with R1's rule: GAP, and larger.** Δ NLL(H_c − S_c) = **+0.253 nats [+0.232, +0.276]**
(R1, without the completion: +0.188). Switch labels +0.411 [+0.378, +0.447]; move labels +0.160 [+0.138, +0.184].

**Registration:** [`../../../endstate/design_x50_rung1b_human_opponent_model.md`](../../../endstate/design_x50_rung1b_human_opponent_model.md)
§8.1 (`42f1d9a7`, before any row was extracted). Build: `d66a74ec1` (`src/main/x50_r1/own.py`, `complete.py`,
`read_r0.py`). Driver: `run_r0.sh`. Every number below is in `r0_read.json`. **Model:**
`models/rb_x26_s2601/checkpoints/checkpoint_380000195_steps.zip` (R1's), CPU, read-only. **Populations:** R1's own
(the held-out dates 09-21..23 of the spectator manifest; X26's eval traces at cycles ≥ 360M), re-extracted in one
pass with four row modes per seat.

## Results

Supervised rows only (R1's rules: a flat target not masked, Hidden Power excluded, the last decision dropped).
Humans: the labelled player rated ≥ 1500. 95 % CIs: battle-cluster bootstrap, 2,000 resamples, seed 50; contrasts
within the traces resample the battles JOINTLY.

| population | rows | battles | NLL (nats) | top-1 | switch share |
|---|---|---|---|---|---|
| H humans, spectator (R1's row) | 48,579 | 1,481 | 1.937 [1.921, 1.953] | 0.338 | 0.300 |
| **H_c humans, completed** | 48,542 | 1,481 | **1.785** [1.769, 1.800] | **0.392** | 0.300 |
| S snapshots, spectator | 29,953 | 1,122 | 1.748 [1.730, 1.766] | 0.389 | 0.259 |
| **S_c snapshots, completed** | 29,884 | 1,121 | **1.531** [1.516, 1.547] | **0.471** | 0.260 |
| S-own-true (the true team, synthesised decisions) | 29,953 | 1,122 | 1.513 [1.497, 1.528] | 0.479 | 0.259 |
| S-full (the training row) | 26,239 | 1,001 | 1.519 [1.502, 1.535] | 0.476 | 0.261 |
| B bots, spectator | 15,063 | 1,182 | 1.682 [1.656, 1.709] | 0.409 | 0.058 |
| B_c bots, completed | 15,063 | 1,182 | 1.565 [1.540, 1.591] | 0.483 | 0.058 |
| B-own-true | 15,063 | 1,182 | 1.550 [1.525, 1.575] | 0.486 | 0.058 |
| B-full | 14,027 | 1,112 | 1.545 [1.519, 1.571] | 0.490 | 0.057 |

| contrast | Δ NLL (nats) | reads |
|---|---|---|
| **P_c = S_c − S-full (PRIMARY)** | **+0.012 [+0.005, +0.020]** | **FIXED** (upper < +0.08) |
| reduction S − S_c | +0.217 [+0.204, +0.230] | (REDUCED's test; not needed) |
| R1's penalty S − S-full, re-extracted | +0.229 [+0.215, +0.244] | R1: +0.229 [+0.206, +0.254] |
| completion error S_c − S-own-true | +0.019 [+0.014, +0.023] | descriptive |
| synthesised decision S-own-true − S-full | −0.006 [−0.012, −0.001] | descriptive (finding 3) |
| B_c − B-full | +0.020 [+0.010, +0.031] | descriptive |
| **H_c − S_c (R1's rule, re-read)** | **+0.253 [+0.232, +0.276]** | **GAP** |
| H_c − S_c, moves | +0.160 [+0.138, +0.184] | |
| H_c − S_c, switches | +0.411 [+0.378, +0.447] | |
| H − S (R1, re-extracted) | +0.188 [+0.164, +0.213] | R1's number exactly |
| H_c − B_c | +0.219 [+0.191, +0.249] | |

**Reading it.**
- The completion closes the pipeline penalty to about a hundredth of a nat. What remains splits into a completion
  error of +0.019 (drawn sets against the true team) and nothing measurable from the synthesised decisions: with the
  true team, the synthesised rows read as well as the training rows.
- The gap did not shrink when the pipeline was fixed; it grew from +0.19 to +0.25. The completion helped our own
  battles (−0.217) more than the humans' (−0.152). Under R1's pipeline the penalty hid part of the gap. The move gap
  is now real too: +0.16 nats, against R1's +0.07.
- On human switches the completed head (2.267) is still worse than the presence-only floor (2.208), and close to the
  uniform floor (2.438). On our snapshots' switches it reads 1.856.

## The fidelity gate (before the read)

With the trainee's TRUE team and the own stream's exact HP, the completed rows equal the rows the trainee played on,
byte for byte: 4,581 of 4,582 rows on 150 traces of 380M (a dev check), and every row of the golden trace in
`complete_test.py`. On the public log only the HP cells move (the percent display's ±1 HP). The own-action labels
equal the trainee's real choices on the golden trace.

## Extraction (counts from the parts under `/home/goodlad/gen3ai_archive/x50_r1b/`, commit `d66a74ec1`)

| | humans (7,413 logs, both seats) | traces (2,306 battles, the trainee's seat) |
|---|---|---|
| seats | 14,826 | 2,306 |
| completed (C-GAME) | 14,756 (99.53 %) | 2,305 |
| excluded: a species with no Smogon prior | 47 | 0 |
| excluded: no consistent team in 32 draws | 8 | 1 |
| excluded: revealed facts no legal set holds | 1 | 0 |
| **refused** (the reader: a Mimic slot, finding 1) | **14** (0.09 %) | 0 |
| own-true (the true team; an illegal own action would refuse) | — | 2,306 / 2,306, 0 refusals |
| completed prefix-guard checks | 14,756 passed | 2,305 + 2,306 passed |
| the TRUE team passes Showdown's validator | — | 2,306 / 2,306 |
| draws: first attempt / a redraw | 14,106 / 650 | 2,274 / 31 |
| decisions (completed) | 430,281 | 80,691 |
| (b)'s own-action labels (completed rows) | 389,156 (5,087 with one legal action) | 72,648 |

Not completed in all: 70 of 14,826 human seats (0.47 %) and 1 of 2,306 traces, under the design's 1 % stop. Every
completed seat's opponent-action labels equal its spectator read's (a mismatch would refuse; none did).

**Where (b)'s labels land:** the same part files, field `own_action` (the viewer's own 11-way action index under
the completed request; −1 = no label: a hidden `cant`, or no action) with `n_legal`, on mode-2 rows
(`/home/goodlad/gen3ai_archive/x50_r1b/humans/*.npz`; traces: modes 2 and 3). A row with either label is kept.
These are the HELD-OUT dates: R-b1 may read them only after the HOM is frozen and hashed; nothing may train on them.

Wall time: humans 380 s and traces 97 s in the short lane; the read 7 min of forwards (cached per part) plus 16 s.

## FINDINGS (standing rule 7)

1. **The 14 refusals are one class: Mimic.** After a Mimic the reader's moveset holds the copied move, but the
   synthesised request still lists `mimic`, and the reader refuses it (`Error with move mimic`). The tracker does not
   model Mimic's slot swap. 0.09 % of seats, under the 1 % stop. A fix (swap the slot until the mon leaves) belongs
   to the training-date extraction.
2. **The completion error is measured only on OUR teams.** S_c − S-own-true is +0.019 on pool teams. Human teams may
   sit further from the Smogon prior, and no human battle has a truth, so the completion's cost on H_c is
   UNVERIFIED. If it is larger, H_c is pessimistic and the true gap smaller.
3. **S-own-true reads slightly BETTER than S-full (−0.006 [−0.012, −0.001]).** Two differences remain between them:
   the ±1 HP of the percent rewrite, and the 191 battles (8.3 %) whose trainee stream held a refused choice. Those
   are in S-own-true and not in S-full, so the row sets differ. The difference is a hundredth of the bar and decides
   nothing.
4. **C-GAME's declared leak (F-1b-4) is in H_c and S_c.** Facts revealed later sit beside drawn ones. The frozen X26
   head never trained on such rows, so R0 is not exposed to it; a model trained on these rows would be.
5. **Pressure is a guess where it is unrevealed.** Gen 3 never announces it, so a foe's Pressure is its revealed
   ability, else its species' Smogon majority. On traces the synthesised PP matched the real requests on all but 40
   of 12,115 decisions (dev check, 373 traces); `maybeTrapped` differed on 12.
6. **The packed order of a completed team is the lead, then the seen mons in order of appearance, then the drawn
   ones.** A human's real order is unknown. Training rows carry the player's own order.
7. **Process.** CPU only. `models/`, the corpus and `data/` were only read; the output is under `gen3ai_archive/`.
   The first read attempt crashed on a missing field (`placeholder`) after its forwards were cached. The fix is one
   line in `read_r0.py`'s field list, shipped with this record; nothing else changed between the two attempts, and no
   number was seen before the crash. A plumbing dev run on 60–600 held-out logs of 09-21 and 40–400 traces preceded
   the build. It looked at refusals, legality and row equality against the training rows, never at an NLL.
