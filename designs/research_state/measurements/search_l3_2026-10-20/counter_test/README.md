# L3's familiarity counter-test: pool decoys in the truth's place (READ 2026-10-20)

Pre-registered in [`PREREGISTRATION.md`](PREREGISTRATION.md) (commit `b39c9cdc`). The read script, the driver and the run
script are byte-identical to the registered ones (sha256 in [`decoy_registration.json`](decoy_registration.json)). The
rows came from [`decoy_run.py`](decoy_run.py) on a detached checkout of L3's code commit `8fe4fb89`, in 2 heavy-slot
chunks plus the recheck (each ≤ 45 min), CPU only, peak 1.39 GB. The rows are in `~/gen3ai_archive/search_l3_decoy/` (`rows.jsonl`,
`recheck.jsonl`, `run.log`). The result is [`result_decoy.json`](result_decoy.json). Re-read it with
`python -I decoy_read.py --rows <archive>/rows.jsonl --recheck <archive>/recheck.jsonl --l3-rows ~/gen3ai_archive/search_l3/rows_0_of_1.jsonl`.

## In one paragraph

**Not settled, and pool-ness is real.** L3's review asked whether the likelihood picks the true team because it is TRUE
or because X26 trained on that pool team (F2). Here a DECOY takes the truth's place: another pool team, consistent with
everything the searched side saw, that is not a near copy of the truth.
- **Decoys clearly beat the honest worlds:** AUC 0.575 [0.517, 0.631] (POOL-LIFT).
- **The truth beats its decoys only by a margin the interval cannot confirm:** AUC 0.578 [0.489, 0.664], INCONCLUSIVE.
- **The registered verdict is NOT-DETECTED.** On the same states the truth beats honest worlds at 0.645. Pool-ness
  carries about half of that (pool share 0.52 [0.15, 0.95]).
- **By opponent (descriptive, each stratum under the registered minimum of 60 states):** against snapshots the truth
  beats its decoys (0.639 [0.518, 0.754]). Against bots it does not (0.515 [0.384, 0.639]), and there the decoys
  (0.589) do as well against honest worlds as the truth (0.588).

L3 stays **PROVISIONAL-POOL**, now with positive evidence that part of its separation is pool-ness (or real-team
coherence: hazard 2). The off-pool bank (the review's A4) is required before L3 or stage 1c steers anything.

## 1. Gates (PREREGISTRATION §6): all PASS

| gate | result |
|---|---|
| G1 the 287 ids | PASS (sha256 `cf8d8c6f…dfb`) |
| G2 defects ≤ 2 | PASS: 0 defects (no likelihood guard and no decoy consistency guard raised) |
| G3 determinism | PASS: 10 / 10 states identical in a fresh process, decoys included |
| G5 reproduction of L3 | PASS: 283 / 283 states' truth and honest log L and conf equal L3's archived values, max \|Δ\| 5.0e-5 nats (bar 1e-4) |
| exclusions | the same 4 builder exclusions as L3; 283 read |

**Decoy availability:** 82 of 283 states hold at least one decoy (244 decoys; 40 bot, 42 snapshot states). On 140 states
no other pool team fits the reveals once the truth and its near copies are dropped. On 61, eligible candidates exist
but none passed the gate.

## 2. Results (95 % battle-clustered intervals; states weighted equally)

| quantity (states with a decoy) | pooled (82) | bot (40) | snapshot (42) |
|---|---|---|---|
| **AUC truth vs decoy** (PRIMARY) | **0.578 [0.489, 0.664] INCONCLUSIVE** | 0.515 [0.384, 0.639] | 0.639 [0.518, 0.754] TRUTH-ABOVE |
| **AUC decoy vs honest** (CO-PRIMARY) | **0.575 [0.517, 0.631] POOL-LIFT** | 0.589 [0.504, 0.671] (lower bound inside the band) | 0.562 [0.482, 0.639] |
| AUC truth vs honest, this subset | 0.645 [0.577, 0.711] | 0.588 [0.484, 0.691] | 0.699 [0.617, 0.773] |
| pool share (AUC_dh − ½) / (AUC_th − ½) | 0.52 [0.15, 0.95] | 1.01 [−2.4, 4.8] | 0.31 [−0.13, 0.70] |
| conf: truth vs decoy | 0.379 [0.295, 0.470] | | |
| conf: decoy vs honest | 0.514 [0.454, 0.570] | | |
| **VERDICT** | **NOT-DETECTED** | UNDERPOWERED (< 60) | UNDERPOWERED (< 60) |

For reference, AUC truth vs honest on all 283 OK states is 0.692 [0.655, 0.728] (L3's own number, reproduced). AUC truth
vs decoy, split by the decoys' mean shared mon sets with the truth: < 2, 0.580 (68 states); ≥ 2, 0.571 (14).

**VERDICT (PREREGISTRATION §5): NOT-DETECTED.** (a) is INCONCLUSIVE: the lower bound 0.489 is below 0.505 and the upper
bound 0.664 is above 0.545. (b) is POOL-LIFT. NOT-DETECTED has no registered consequence of its own, so L3's standing is
unchanged: PROVISIONAL-POOL, with the off-pool replication required.

## 3. What it means (descriptive, post hoc)

- **Part of L3 is pool-ness.** A pool team that is NOT the truth already beats today's honest worlds (0.575, clear).
  Because a decoy agrees with the truth only on what was revealed, this lift does not come from the hidden facts of the
  true team.
- **Whether the truth adds anything beyond pool-ness is not settled pooled.** The point estimate (0.578) sits about where
  half of L3's subset separation would put it. Only the snapshot stratum shows the truth clearly above its decoys.
  There the opponent model nearly is the opponent (a greedy earlier X26), the one case where the truth's own facts
  should show most.
- **Against bots, L3's signal looks like pool-ness only.** There, decoys beat honest worlds as often as the truth does
  (0.589 vs 0.588), and the truth does not beat its decoys (0.515). This is descriptive and underpowered (40 states).
  It agrees with the review's F1: the bot stratum, the more defensible prior for a non-self opponent, carries little
  truth-specific evidence.
- **The confidence control behaves differently from L3's.** The network is LESS sure on the truth's rows than on a
  decoy's (conf AUC 0.379). So whatever separates the truth from its decoys is not the network being surer about the
  true team.

## 4. FINDINGS

1. **The subset is selected (declared hazard 1).** Only 82 of 283 states (29 %) hold a decoy. The 8-state smoke
   suggested about half. These are mostly states with few reveals. On the same subset L3's AUC is 0.645, against 0.692
   on all states.
2. **Pool-ness vs real-team coherence (declared hazard 2) is NOT separated.** The POOL-LIFT could be training
   familiarity, or the network recognising coherent real teams over Smogon-usage draws. An off-pool real-team decoy (the
   ladder corpus, strict) would separate them; it was not run (out of this unit).
3. **The decoys' seen-mon facts are their own (declared hazard 3).** A decoy's item, ability and spread fit the log only
   under some damage rolls, unlike the honest worlds' screened spreads. Decoy vs honest therefore also differs in
   non-binding facts.
4. **Both strata are under the registered minimum of 60 states.** Their verdicts are UNDERPOWERED. Their numbers above
   are descriptive only.
5. **No validator on the decoys (declared hazard 5).** Only the clause check and the consistency guard ran.
6. **One model, one anchor, one pool bank** (X26 225M), as in L3.
7. **Process.** About 4.7 h of the run's wall time was spent waiting for heavy slots (22 min, 2.3 h and 2.0 h; up to 6 jobs queued). The run ran
   from a detached checkout at `/home/goodlad/.cache/gen3ai/l3_decoy/pin`, removed after the read. Its Rust binaries
   were built there, niced, outside a slot (~1 min).
