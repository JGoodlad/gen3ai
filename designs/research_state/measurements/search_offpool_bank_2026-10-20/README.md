# The OFF-POOL BANK at P\*, and L3 and its decoy test replicated on it (2026-10-20)

**Status: READ 2026-10-20.** READ 2 (L3 replicated): **REPLICATED-SELF-PLAY-ONLY**. READ 3 (the off-pool decoy):
**NO-LIFT**, combined **TRUTH-SPECIFIC**. Rows: `~/gen3ai_archive/search_offpool_bank/rows.jsonl`, `recheck.jsonl`. Result:
[`result_offpool.json`](result_offpool.json).

Registered in [`PREREGISTRATION.md`](PREREGISTRATION.md) (`7642df0f6`; the
registered files' sha256 in [`offpool_registration.json`](offpool_registration.json)). No battle had been played and no
likelihood computed at registration.

**The bank** ([`states_offpool.json`](states_offpool.json), committed before any likelihood):
- **Battles:** 312, all three parts complete. bots: 162 battles in 4.4 min. sentA: 90 in 26.7 min. sentB: 60 in 7.0 min.
- **Gates:** the off-pool gate passed on 312 / 312 battles (every opponent a heldout team, every traced side a pool
  team). Gate ①: 0 failures on 10,395 decisions; the turn decode agrees on all of them.
- **Draw:** 8,047 eligible decisions; every cell filled from the cycle alone, so the extension was not needed. **320
  states (160 snapshot, 160 bot; 64 per k bin)** from 223 battles and 68 of the 72 opponent teams. ids sha256
  `db5edcdc…29f6`, bank content sha256 `18bec65b…3725`.
- **Outcome mix (hazard 2, as declared):** bot stratum 154 wins / 6 losses; snapshot stratum 92 wins / 65 losses /
  3 draws. L3's bank was loss-enriched.

Stage 1c's A4 asks any L3 or 1c claim to replicate on an OFF-POOL opponent bank at the plateau checkpoint before it
steers anything. This unit builds the bank: X26's P\* (280,000,170) plays the plateau cycle's own roster, every
opponent piloting a `heldout72_hpfix` ladder team that X26's lineage never trained against ([`offpool_proof.json`](offpool_proof.json)).
It then runs the two cheapest reads on it: L3 (truth vs honest worlds) and the decoy test with OFF-POOL ladder decoys
(pool memory or real-team coherence?).

| file | what |
|---|---|
| `offpool_common.py` | the constants: models, sentinels, parts, seeds, cells, salts |
| `offpool_refs.py` | `proof` (the off-pool proof, run) and `corpus` (the off-pool decoy corpus, built) |
| `gen_cycle.py`, `run_bank.sh` | the battles: `main.ops.eval_trace_gen` with three patches (§3), three parts, one heavy slot each |
| `make_bank.py` | the maturity battery's draw over the generated cycle (§4) |
| `offpool_run.py`, `run_score.sh` | the scoring: L3's candidates + off-pool decoys under P\* (§5–§6), G3's recheck |
| `offpool_read.py` | the registered read: L3's and the decoy test's registered reads, reused by path and sha-checked (§7–§9) |
| `offpool_test.py` | the tests of the pure parts and the registration digests |

**Checkouts and archive.** Every engine, world and network step runs from a detached checkout of L3's code commit
`8fe4fb89` (`~/.cache/gen3ai/offpool/code`), with `--code`. The archive is `~/gen3ai_archive/search_offpool_bank/`:
`ckpt/`, `traces/`, `bank/`, `census.npz`, `decoy_corpus.json`, `rows.jsonl`, `recheck.jsonl`, and `smoke/` (the
engineering smokes, never read as output).

## In one paragraph

**Off-pool, at the plateau, the opponent's own choices still point at the true team, but only when the opponent is our
own network.**
- **Against X26's earlier snapshots** (each piloting a ladder team X26 never trained on), the truth beats today's honest
  worlds 79 % of the time: AUC 0.794 [0.753, 0.834], exactly L3's pool-bank 0.796.
- **Against the scripted bots it is gone:** 0.472 [0.410, 0.531], NO-EVIDENCE. On the pool bank it was 0.588 (WEAK), so
  that part of L3 was pool memory.
- **The off-pool decoy separates pool memory from real-team coherence.** A real ladder team that is not the truth gets
  NO lift over the honest worlds: 0.503 [0.462, 0.543]. On the pool bank, a pool decoy got 0.575. And the truth beats
  these decoys (0.611 [0.543, 0.680]).

So the pool decoy's lift in the counter-test was pool MEMORY, not coherence, and what remains off-pool is
truth-specific, carried by the snapshot stratum. By A4, L3 is now **SELF-PLAY-ONLY**: it replicates off-pool at the
plateau where the opponent model nearly is the opponent, and not against a non-self opponent.

## 1. Gates (PREREGISTRATION §4, §9): all PASS

| gate | result |
|---|---|
| G-OFF (bank) | 312 / 312 battles: the opponent on a heldout team, the traced side on a pool team |
| gate ①, turn decode | 0 failures on 10,395 decisions; 10,395 / 10,395 agree |
| G1 ids | PASS (sha256 `db5edcdc…29f6`) |
| G2 defects | PASS: 0 (no likelihood guard and no decoy consistency guard raised) |
| G3 determinism | PASS: 10 / 10 states identical in a fresh process, decoys included |
| G4 encoder (descriptive) | 318 / 318 states' truth opponent rows byte-identical at X26's pin `9c033489` |
| exclusions | 2 of 320 (the builder's honest `WorldError`); 318 read (159 bot, 159 snapshot) |

Per state the opponent made 13.6 scored choices (13.5 informative); 417 were excluded as unobserved, and 2 as pending.
Decoys: 118 states hold one (62 bot, 56 snapshot; 412 decoys). On 103 states no off-pool team fits the reveals; on 97
eligible candidates existed but none passed the gate.

## 2. READ 2: L3 replicated (L3's registered read; 95 % battle-clustered intervals)

| quantity | pooled (318) | bot (159) | snapshot (159) | L3 on the pool bank, 225M (pooled / bot / snapshot) |
|---|---|---|---|---|
| **AUC** truth vs honest | **0.633 [0.591, 0.675] WORKS** | **0.472 [0.410, 0.531] NO-EVIDENCE** | **0.794 [0.753, 0.834] WORKS** | 0.692 / 0.588 / 0.796 |
| top1 (beats all 8) | 0.258 [0.210, 0.308] | 0.107 [0.058, 0.157] | 0.409 [0.330, 0.489] | 0.304 / 0.142 / 0.465 |
| p_truth (weight among 9; counterfactual) | 0.238 [0.203, 0.277] | 0.113 [0.076, 0.151] | 0.363 [0.306, 0.422] | 0.249 / 0.137 / 0.360 |
| AUC_conf (familiarity control) | 0.469 [0.430, 0.509] | 0.497 [0.446, 0.551] | 0.440 [0.384, 0.497] | 0.481 |
| AUC_regret | 0.637 [0.596, 0.682] | 0.477 [0.416, 0.535] | 0.798 [0.756, 0.838] | 0.676 |
| lift_moves (reweighting today's 8) | +0.003 [−0.005, +0.012] | −0.001 [−0.013, +0.012] | +0.007 [−0.003, +0.018] | +0.005 |
| lift_moves_exact | −0.000 [−0.004, +0.005] | +0.002 [−0.004, +0.011] | −0.003 [−0.007, 0.000] | −0.000 |
| lift_species | −0.003 [−0.019, +0.015] | −0.012 [−0.036, +0.015] | +0.006 [−0.015, +0.029] | +0.008 |

By the truth's informative choices: 1–4, 0.593 [0.503, 0.681] (47); 5–9, 0.632 [0.556, 0.711] (75); ≥ 10, 0.644 [0.592,
0.697] (194). Spearman +0.07.

**VERDICT (§7): REPLICATED-SELF-PLAY-ONLY.** The pooled AUC is WORKS (lower bound 0.591), but the bot stratum is
NO-EVIDENCE while the snapshot stratum WORKS. **Bot stratum: NO-EVIDENCE.**

## 3. READ 3: the off-pool decoy (the counter-test's registered read, relabelled)

| quantity (states with a decoy) | pooled (118) | bot (62) | snapshot (56) |
|---|---|---|---|
| **AUC decoy vs honest** (PRIMARY) | **0.503 [0.462, 0.543] NO-LIFT** | 0.511 [0.454, 0.571] INCONCLUSIVE | 0.495 [0.439, 0.548] INCONCLUSIVE |
| **AUC truth vs decoy** (co-primary) | **0.611 [0.543, 0.680] TRUTH-ABOVE** | 0.504 [0.406, 0.602] INCONCLUSIVE | 0.729 [0.641, 0.811] TRUTH-ABOVE-MATERIAL |
| AUC truth vs honest, this subset | 0.598 [0.537, 0.656] | 0.490 [0.409, 0.576] | 0.717 [0.643, 0.785] |
| conf: truth vs decoy / decoy vs honest | 0.489 / 0.504 | 0.567 / 0.482 | 0.402 / 0.528 |
| **VERDICT** | **TRUTH-SPECIFIC** | NOT-DETECTED | UNDERPOWERED (< 60) |

**Comparison with the pool decoy (descriptive, hazard 3):** the pool decoy read 0.575 [0.517, 0.631] at 225M on the pool
bank. This interval does not hold 0.575, and the two intervals overlap only in [0.517, 0.543].

**VERDICT (§8): NO-LIFT; combined TRUTH-SPECIFIC.** The registered consequence of NO-LIFT: the pool-bank reads (L3's
0.692, stage 1c on the pool bank) are inflated by pool memory, and only off-pool reads steer.

## 4. Descriptive (no verdict)

- **Clustered by the opponent's TEAM** (68 teams), every verdict holds:
  - truth vs honest: pooled 0.633 [0.581, 0.683], bot 0.472 [0.404, 0.539], snapshot 0.794 [0.743, 0.839];
  - decoy vs honest: 0.503 [0.465, 0.540];
  - truth vs decoy: 0.611 [0.533, 0.681].
- **Outcome mix (hazard 2).** This bank: bots 96 % wins, snapshots 58 % wins. L3's bank: bots 62 % wins, snapshots
  46 % wins.
  - Reweighted to L3's mix per stratum, the truth-vs-honest AUC is 0.703 [0.655, 0.746], L3's level.
  - By outcome: P\*'s LOSSES 0.834 [0.775, 0.885] (70); WINS 0.575 [0.524, 0.623] (245); draws 0.708 (3).
  - The signal lives where the opponent was winning, mostly against the snapshots. Against bots this bank holds only 6
    losses, so the bot stratum's NO-EVIDENCE is a read on P\*'s wins against bots.

## 5. What it means (post hoc)

- **The pool-bank bot signal was pool memory.** On the pool bank the bots read 0.588, and the counter-test's decoys
  matched the truth there. Off-pool they read 0.472. A mis-specified opponent model (our network seated as a scripted
  bot) carries no evidence about a team it has not trained against.
- **Coherence explains nothing.** A real ladder team that fits every reveal ranks with the honest Smogon-usage worlds
  (0.503). The pool decoy's lift was therefore familiarity with pool teams, not coherence. For stage 1c's corpus arms
  (C_ladder) this means no likelihood advantage from coherence alone. It also means the pool bank's corpus arms cannot
  be separated from memory: any pool-bank NEARER needs this bank.
- **What survives is truth-specific, in self-play only.** Against our own earlier snapshots the truth beats both honest
  worlds and off-pool decoys clearly (0.794; 0.729 against decoys). There the opponent model nearly is the opponent. For
  a non-self opponent (bots here, ladder humans in play) the network as the opponent model shows nothing. A
  behaviour-cloned opponent model is a different candidate, never tested.
- **The proposal set still binds.** Reweighting today's 8 honest worlds moves the hidden moves by +0.003 pooled
  (+0.007 snapshots), as on the pool bank.

## 6. FINDINGS

1. **Outcome mix (declared hazard 2).** The bot stratum is 96 % P\* wins (L3's: 62 %). The AUC is far higher in P\*'s
   losses (0.83) than its wins (0.57), so the bot NO-EVIDENCE is measured almost only on wins. Reweighted, the pooled AUC
   is 0.703. A loss-enriched bot bank (the live quota, about 8 CPU-h of battles on this box) would test whether bots
   carry evidence in P\*'s losses. Not run.
2. **Cross-bank comparison (declared hazard 3).** NO-LIFT is absolute (vs 0.5). Calling the pool decoy's 0.575 "memory"
   compares 225M on the pool bank with P\* off-pool; maturity is not separated.
3. **The decoy subset is selected (as in the counter-test).** 118 of 318 states (37 %) hold a decoy, mostly states with
   few reveals; the truth-vs-honest AUC there is 0.598 against 0.633 on all.
4. **The snapshot stratum's decoy read is UNDERPOWERED** (56 < 60 states); its TRUTH-ABOVE-MATERIAL is descriptive.
5. **The truths' hidden parts are partly reconstructions (declared hazard 1)**, from Metamon and the Hidden Power repair.
   That is conservative for READ 2, and the snapshot stratum still matches L3.
6. **72 teams, 68 in the bank, about 4.7 states each (declared hazard 4).** The team-clustered intervals give the same
   verdicts.
7. **One regime change (declared hazard 5):** sentinels drew heldout teams, not the trainee's distribution.
8. **One model, one anchor** (P\*); no other checkpoint.
9. **Process.**
   - Wall time was mostly the heavy-slot queue: about 4.5 h for the three battle parts and the draw, about 2.7 h for the
     scoring. The CPU work was about 40 min of battles and about 45 min of scoring (7.5 s a state).
   - My first 55-minute timer was briefly duplicated (a second waiter, stopped at once by its task id).
   - The usage gate returned HTTP 429 (fail-open) when checked.
   - Every kill was by explicit PID or task id.

## 7. Recommendation: stage 1c phase B on this bank (out of scope; cost only)

Phase B needs phase A's proposals on this bank first:
- **Phase A:** 128 F + 128 C_ladder worlds and the likelihood, about 100 CPU-s a state, so about 9 slot-hours for the
  318 states (12–20 wall-h with this queue).
- **Phase B:** about 13 CPU-s a state per arm, about 1.2 CPU-h per arm, 3–5 arms.
- **Total:** about 12–15 slot-hours, 1–1.5 days on the contended box.

Three changes are needed:
- **P\* as the opponent model** (1c runs 225M).
- **C_ladder must drop each truth's own `hl_05_26` file and its near copies.** The heldout teams ARE corpus files, so
  without that the corpus holds the truth.
- **The playouts must run at a pin with the damage-roll control** (1c's `pin_rolls`).

Given READ 2's bot NO-EVIDENCE, a phase B here can show a gain only on the snapshot stratum (SELF-PLAY-ONLY by A4).
Run it only if 1c's pool-bank phase A reads NEARER on the bot stratum, or as the snapshot-stratum upper bound before a
behaviour-cloned opponent model is built.
