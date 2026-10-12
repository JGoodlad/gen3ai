# Value-discrimination pilot, PHASE 3 (offline per-action Q head): end2

Training: 9629 states / 964 battles (oracle labels, K = 2; validation 145 battles). Evaluation: 600 registered states (599 with Phase 2's honest rows), TRUE-world truth seeds 16–39.

Verdicts: R-QHEAD **NOT DETECTED**, R-QHEAD-CRITIC **BEHIND**

| scorer (argmax, no rollout at decision time unless named) | IMP [95 %] | n | share of states moved off top1 |
|---|---|---|---|
| A-HEAD (ride-along) | +0.0101 [+0.0009, +0.0195] | 600 | 0.202 |
| HONEST CRITIC | +0.0139 [+0.0019, +0.0264] | 599 | 0.444 |
| HONEST LEAF-K8 | +0.0110 [-0.0015, +0.0244] | 599 | 0.457 |
| HONEST TRUNC-4-K4 | +0.0157 [+0.0020, +0.0291] | 599 | 0.559 |
| ORACLE CRITIC | +0.0261 [+0.0143, +0.0392] | 600 | 0.468 |
| ORACLE LEAF-K8 | +0.0806 [+0.0664, +0.0945] | 600 | 0.513 |
| QHEAD@0.25 | -0.0094 [-0.0229, +0.0041] | 600 | 0.462 |
| QHEAD@0.5 | -0.0004 [-0.0131, +0.0121] | 600 | 0.453 |
| QHEAD@1.0 | -0.0044 [-0.0162, +0.0085] | 600 | 0.402 |

| contrast | Δ [95 %] |
|---|---|
| R-QHEAD: QHEAD - policy (all states) | -0.0044 [-0.0162, +0.0085] |
| R-QHEAD-CRITIC: QHEAD - HONEST CRITIC (common) | -0.0179 [-0.0320, -0.0040] |
| QHEAD - HONEST TRUNC-4-K4 (common) | -0.0198 [-0.0340, -0.0044] |
| QHEAD - ORACLE CRITIC | -0.0306 [-0.0443, -0.0181] |
| QHEAD - A-HEAD | -0.0146 [-0.0259, -0.0033] |

| fit share | fit states | validation R² (centred labels) | member epochs |
|---|---|---|---|
| 0.25 | 2048 | -0.0038 | [11, 12, 10, 10, 10] |
| 0.5 | 4096 | -0.0015 | [10, 10, 10, 10, 10] |
| 1.0 | 8181 | +0.0017 | [10, 9, 9, 10, 9] |

| scorer | mean within-state Spearman ρ vs truth | n |
|---|---|---|
| QHEAD@1.0 | +0.0999 | 511 |
| POLICY logits | +0.1425 | 511 |
| ORACLE CRITIC | +0.2001 | 511 |
| HONEST CRITIC | +0.1567 | 510 |
