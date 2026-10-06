# CRN label refit: the result (2026-10-06)

**Verdict, by the registered rule: NOT DETECTED, NOT SHOWN FLAT.** REFIT-K8 − BASE on M1 is
**−0.0144 [−0.0320, +0.0019]**. The interval's UPPER bound excludes the two outcomes the question
was about: LABEL-LIMITED and PARTIAL. Closing half the rollout gap needed +0.038. The interval is
not inside ±0.02, so the registered FLAT label does not apply. The interval leaves room for a
small HARM, not for a gain.

**Reading.** A less noisy label does not raise this head's sibling discrimination. Refit on the
mean of 8 shared-dice rollouts, the frozen-trunk readout ranks siblings no better than the head
as trained, and no better than a refit on ONE rollout. Meanwhile the 8-rollout leaf itself beats
the head by +0.077. The information that separates siblings is in the playouts, not in the
`value_pooled` features of the one-ply successor. **The bound is the representation (or the
coverage of the states it was trained on), not the label.** The refit buys CALIBRATION
(ECE 0.065 → 0.032), not RESOLUTION. That is the 2026-09-18 offline-fit result again, now with
de-noised labels.

Checkpoint `rb_x5ab_blob_s1001/final_model.zip` (15.05M, `e5e660dd`), CPU, greedy continuation
on both sides. Held-out: 563 turns / 191 battles (179 contested turns, 96 battles with a
non-tied pair). Train: 556 turns / 189 battles. Battle-clustered bootstrap, 2,000 draws, 95 %.

## M1: all non-tied sibling pairs, held-out contested turns (2,976 pairs, 96 battles)

| scorer | accuracy [95 %] |
|---|---|
| BASE (the head as trained) | 0.5932 [0.5578, 0.6274] |
| REFIT-K1 (fit seed 0; seeds 1-2) | 0.5857 [0.5498, 0.6199] (0.5848, 0.5865) |
| REFIT-K4 | 0.5771 [0.5443, 0.6099] (0.5791, 0.5811) |
| **REFIT-K8** | **0.5788 [0.5452, 0.6123]** (0.5852, 0.5828) |
| LEAF-K1 | 0.5769 [0.5545, 0.5998] |
| LEAF-K4 | 0.6322 [0.6022, 0.6601] |
| **LEAF-K8 (the ceiling)** | **0.6702 [0.6398, 0.7006]** |

| contrast (paired) | Δ [95 %] | read |
|---|---|---|
| **REFIT-K8 − BASE** (the verdict) | **−0.0144 [−0.0320, +0.0019]** | not detected; a gain ≥ +0.002 excluded |
| LEAF-K8 − BASE (room exists) | +0.0769 [+0.0354, +0.1146] | DETECTED: the K-curve reproduces on this checkpoint |
| LEAF-K4 − BASE | +0.0390 [+0.0002, +0.0772] | BOUNDARY: the lower bound is within 0.0005 of 0, so it counts as not clear (rule 8) |
| REFIT-K8 − REFIT-K1 (label NOISE) | −0.0069 [−0.0181, +0.0046] | not detected: more rollouts per label buy nothing |
| REFIT-K1 − BASE (label ESTIMAND + refit) | −0.0076 [−0.0278, +0.0107] | not detected |
| LEAF-K8 − REFIT-K8 | +0.0914 [+0.0493, +0.1315] | DETECTED |

CLOSED (share of the rollout gap the K8 refit closes) = **−0.19**.

## M2: the leaf column, policy top-1 vs top-2 (103 pairs, 81 battles; too few to decide anything)

BASE 0.524 [0.424, 0.620] · REFIT-K8 0.505 [0.413, 0.598] · LEAF-K8 0.612 [0.543, 0.683].
REFIT-K8 − REFIT-K1 −0.049 [−0.096, −0.010] (detected, against the label). Every other contrast
is not detected.

## M3: the STARVED slice (π < 1 % on an action within 0.1 of the best), held-out free turns

| slice | n pairs / battles | BASE | REFIT-K1 | REFIT-K8 | LEAF-K8 |
|---|---|---|---|---|---|
| M3: starved near-best vs anything | 1,119 / 109 | **0.450** [0.399, 0.502] | 0.481 | 0.468 | **0.685** [0.644, 0.728] |
| M3a: vs the best action | 77 / 55 | 0.597 [0.470, 0.716] | 0.597 | 0.623 | 0.578 |
| M3b: vs a DOMINATED action | 327 / 70 | **0.440** [0.360, 0.520] | 0.492 | 0.465 | **0.885** [0.849, 0.918] |

- On M3, REFIT-K8 − BASE is +0.018 [+0.002, +0.035] and REFIT-K1 − BASE is
  +0.030 [+0.011, +0.051]. Both are DETECTED but tiny, and both leave the refit below 0.5.
  LEAF-K8 − BASE is +0.235 [+0.169, +0.300].
- **The head ranks a starved near-best move BELOW a dominated alternative 56 % of the time**
  (M3b, point estimate; the interval reaches 0.52, so "below chance" is not detected). 8 rollouts
  order the same pairs correctly 88.5 % of the time. This is the X4 pre-read's blind spot on
  another checkpoint. A better label does not repair it.

## Descriptive (registered)

- **M1 by pair class** (BASE / REFIT-K8 / LEAF-K8):
  - move-move (580 pairs): 0.591 / 0.566 / 0.620;
  - move-switch (1,708): 0.605 / 0.591 / 0.692;
  - switch-switch (688): 0.565 / 0.558 / 0.659.

  The leaf's lead is largest where switches are involved, the category the lineage starves most.
- **ECE against the held-out truth mean** (held-out actions): BASE 0.065, REFIT-K1 0.039,
  REFIT-K4 0.033, REFIT-K8 0.032. **Validation BCE**: 0.567 → 0.540 at K8, best epoch 6-12 of
  27-33 for every fit.
- **Checks:** the warm-started head reproduces BASE on the stored features to 2e-7. On the 1,007
  capture branches that ended within one ply, the capture's terminal equals truth playout j's
  outcome in 1,007 of 1,007 (capture and truth are on the same dice). Refused: 0 capture turns,
  0 truth turns.

## Against the prediction

| predicted | observed |
|---|---|
| LEAF-K8 − BASE +0.04 to +0.08, detected | +0.077, detected ✓ |
| REFIT-K8 − BASE FLAT or not detected, point within ±0.02 | −0.014, not detected ✓ (not shown flat) |
| REFIT-K8 − REFIT-K1 small and positive (≤ +0.02) | −0.007, not detected (sign wrong) |
| REFIT-K1 − BASE ≤ +0.02, sign unknown | −0.008, not detected |
| M3: BASE near chance against a*; the refit does not change that | M3a BASE 0.60 [0.47, 0.72] (wide); against DOMINATED partners 0.44. The refit moves M3 by +0.018 (detected, negligible) |

## Cost

**Measured CPU: 4.6 CPU-hours** for the truth playouts (cgroup `cpu.stat`):
- held-out: 8,961 + 3,342 CPU-s over two legs;
- train: 3,486 CPU-s;
- pilot: 863 CPU-s.

160,256 playouts, **0.10 CPU-s per playout** in steady state, about half the pilot's 0.21. Add
~0.3 CPU-h estimated for the pilot's first 6 turns, the capture (2.1 min wall), the root read and
the nine fits. Wall time: 13:41 → 15:44 PDT, at 3 workers x 2 threads, nice 15. No GPU lease,
`--device cpu`.

## Caveats

- **The truth is a greedy continuation on both sides** (blob's own greedy): not Nash, not the
  recorded opponent. The refit labels are on the same estimand, so this does not favour the leaf
  over the refit.
- **The states are the Lane S bank's**: ai_v14-lineage eval traces from before the F-LF-1 bot fix,
  read by an rb-era checkpoint. They are off the blob arm's training distribution, which is
  "coverage" in the rule's sense. What a bank of blob's own states would read is UNVERIFIED.
- **The subsample**: 380 of 549 battles. With the steady-state rate the full set would have cost
  ~6.6 CPU-h, inside the budget. The pilot over-priced it 2x. The rule was followed as registered.
- **The yardstick is 24 truth seeds**, which caps every level (LEAF-K8 0.670 is not 1.0). M2 has
  only 103 pairs.
- **The refit's training set is small**: 3,247 train + 713 validation actions. The K1→K8 null and
  the 2026-09-18 fit on 17,904 positions (which read the same) say data volume is not what binds.
  A refit on many more de-noised labels is still UNVERIFIED.
- **The readout sees the one-ply successor only.** "Representation" here means `value_pooled` of
  the state after one move. A Q readout on (state, action) features was not tried.
