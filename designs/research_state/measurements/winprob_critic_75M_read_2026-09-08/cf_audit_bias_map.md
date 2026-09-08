# cf_audit — the counterfactual bias map

**Run:** `/home/goodlad/dev/gen3ai/models/ai_v12_02_winprob_critic`  ·  **step:** 74000016  ·  **checkpoint:** `None`  ·  **R:** 8  ·  **sampler:** `cf_audit_strata_v1` (seed 0)

## Headline

```
labels                          800   over 214 battles
population-weighted gap         +0.0979
population-weighted sd_true_excess  0.2550
```

The **gap** is the offset a re-centring would fix; the **sd_true_excess** is the per-state spread the head does not resolve, and it is the primary meter. A lever that moves the first and not the second has not done the thing this program is for.

## Accounting

| | |
|---|---|
| frame decisions | 6083 |
| frame battles | 242 |
| tasks issued | 800 |
| labelled | 800 |
| errors | 0 |
| anchors issued | 142 |
| anchors reproduced | 142 |
| rollouts | 6400 |
| skipped (frame) | {"turn_1_unopenable": 242, "forced_switch_rounds": 936} |

## By battle outcome (a description of two state POPULATIONS, not a calibration verdict)

| stratum | n | n_battles | mean_predicted | mean_mc | mean_gap | 95% CI (battle-clustered) |
|---|---|---|---|---|---|---|
| win | 182 | 88 | +0.8456 | +0.8771 | -0.0315 | [-0.0599, -0.0009] |
| loss | 618 | 126 | +0.7533 | +0.5218 | +0.2315 | [+0.1926, +0.2728] |

## By predicted decile × outcome

| stratum | n | n_battles | mean_predicted | mean_mc | mean_gap | 95% CI (battle-clustered) |
|---|---|---|---|---|---|---|
| decile0/loss | 8 | 6 | +0.0311 | +0.0156 | +0.0155 | [-0.0163, +0.0440] |
| decile1/loss | 8 | 8 | +0.1538 | +0.0938 | +0.0601 | [-0.0648, +0.1597] |
| decile2/loss | 9 | 8 | +0.2467 | +0.1111 | +0.1356 | [+0.0641, +0.2054] |
| decile3/loss | 13 | 12 | +0.3481 | +0.1827 | +0.1654 | [-0.0147, +0.3413] |
| decile4/win | 3 | 3 | +0.4908 | +0.6250 | -0.1342 | [-0.3823, -0.0057] |
| decile4/loss | 21 | 19 | +0.4494 | +0.2381 | +0.2113 | [+0.0986, +0.3152] |
| decile5/win | 8 | 5 | +0.5613 | +0.5469 | +0.0145 | [-0.2071, +0.2089] |
| decile5/loss | 29 | 24 | +0.5479 | +0.4310 | +0.1168 | [+0.0044, +0.2183] |
| decile6/win | 20 | 17 | +0.6679 | +0.7500 | -0.0821 | [-0.1888, +0.0115] |
| decile6/loss | 45 | 36 | +0.6431 | +0.5528 | +0.0903 | [+0.0099, +0.1751] |
| decile7/win | 35 | 32 | +0.7508 | +0.8357 | -0.0850 | [-0.1454, -0.0223] |
| decile7/loss | 197 | 86 | +0.7487 | +0.5425 | +0.2062 | [+0.1570, +0.2579] |
| decile8/win | 38 | 31 | +0.8484 | +0.8882 | -0.0398 | [-0.0950, +0.0372] |
| decile8/loss | 164 | 70 | +0.8438 | +0.6105 | +0.2333 | [+0.1776, +0.2920] |
| decile9/win | 78 | 56 | +0.9752 | +0.9663 | +0.0088 | [-0.0188, +0.0458] |
| decile9/loss | 124 | 40 | +0.9452 | +0.5554 | +0.3897 | [+0.2862, +0.4938] |

## By turn tercile

| stratum | n | n_battles | mean_predicted | mean_mc | mean_gap | 95% CI (battle-clustered) |
|---|---|---|---|---|---|---|
| turn<=10/win | 73 | 57 | +0.7860 | +0.8613 | -0.0753 | [-0.1148, -0.0301] |
| turn<=10/loss | 206 | 99 | +0.7538 | +0.6377 | +0.1160 | [+0.0700, +0.1625] |
| turn 10-19/win | 61 | 49 | +0.8793 | +0.8689 | +0.0104 | [-0.0413, +0.0714] |
| turn 10-19/loss | 190 | 93 | +0.7534 | +0.4605 | +0.2929 | [+0.2363, +0.3496] |
| turn>19/win | 48 | 35 | +0.8934 | +0.9115 | -0.0181 | [-0.0569, +0.0353] |
| turn>19/loss | 222 | 64 | +0.7528 | +0.4668 | +0.2860 | [+0.2048, +0.3745] |

## By opponent

| stratum | n | n_battles | mean_predicted | mean_mc | mean_gap | 95% CI (battle-clustered) |
|---|---|---|---|---|---|---|
| aggressive | 17 | 10 | +0.8095 | +0.5515 | +0.2581 | [+0.0758, +0.4228] |
| aggressive_v2 | 57 | 17 | +0.7812 | +0.6184 | +0.1628 | [+0.0488, +0.2741] |
| heuristic | 69 | 18 | +0.8095 | +0.5761 | +0.2334 | [+0.1145, +0.3649] |
| heuristic2 | 46 | 13 | +0.8218 | +0.7065 | +0.1152 | [-0.0307, +0.2497] |
| random | 18 | 7 | +0.9075 | +1.0000 | -0.0925 | [-0.1199, -0.0665] |
| sentinel_0 | 94 | 17 | +0.7720 | +0.5545 | +0.2175 | [+0.1394, +0.2935] |
| sentinel_1 | 66 | 18 | +0.7025 | +0.4735 | +0.2290 | [+0.1337, +0.3189] |
| sentinel_2 | 78 | 18 | +0.6853 | +0.5913 | +0.0939 | [+0.0381, +0.1565] |
| sentinel_3 | 60 | 20 | +0.7227 | +0.6208 | +0.1019 | [+0.0287, +0.1564] |
| sentinel_4 | 96 | 20 | +0.7632 | +0.6693 | +0.0940 | [+0.0169, +0.1729] |
| setup_sweep | 45 | 16 | +0.8210 | +0.6333 | +0.1877 | [+0.0785, +0.2942] |
| setup_sweep_v2 | 46 | 15 | +0.8193 | +0.5380 | +0.2812 | [+0.0992, +0.3812] |
| staller | 51 | 13 | +0.8286 | +0.6103 | +0.2183 | [+0.0364, +0.3559] |
| staller_v2 | 57 | 12 | +0.7941 | +0.5592 | +0.2349 | [-0.0286, +0.4663] |

## The conviction class — high confidence, lost battle

```
n=379 over 92 battles
predicted 0.861 (median 0.849)  vs tight-MC 0.579 (median 0.625)
gap +0.2818  CI [+0.2288, +0.3394]
LOSS - WIN difference +0.2928  CI [+0.2254, +0.3547]
MC >= 0.75 (the critic was RIGHT; the dice lost it)  40.9%
MC <  0.50 (the critic was genuinely wrong)          30.9%
MC <  0.25 (badly wrong)                             14.5%
```

A single realized outcome cannot separate those two populations. That separation is the whole case for a tight-MC label as an instrument.

## RESOLUTION — within-decile true spread vs the binomial floor

| decile | n | predicted | MC | sd(MC) | binomial floor | **sd_true_excess** | % variance real |
|---|---|---|---|---|---|---|---|
| 3 | 13 | 0.348 | 0.183 | 0.317 | 0.090 | **0.304** | 92.0% |
| 4 | 24 | 0.454 | 0.280 | 0.284 | 0.133 | **0.251** | 78.0% |
| 5 | 37 | 0.551 | 0.457 | 0.272 | 0.159 | **0.221** | 66.0% |
| 6 | 65 | 0.651 | 0.614 | 0.284 | 0.150 | **0.242** | 72.2% |
| 7 | 232 | 0.750 | 0.663 | 0.296 | 0.140 | **0.261** | 77.8% |
| 8 | 202 | 0.846 | 0.744 | 0.279 | 0.127 | **0.249** | 79.3% |
| 9 | 202 | 0.967 | 0.849 | 0.282 | 0.084 | **0.269** | 91.2% |

## EVIDENTIAL — does the confessed width track the blur?

_The audited checkpoint carries no `cf_evid_head` (`--cf-evidential` off, or pre-v98) — the evidential columns are ABSENT, not zero._

## Caveats

- Turn-1 decisions are excluded by construction (the offline replay driver cannot open them) and forced-switch rounds are structurally uncovered by the re-roll anchor.
- The MC label is measured on the EVAL distribution, played greedy; the head was trained on a mostly-self-play mixture with a stochastic actor. Never quote a gap without naming the population — its SIGN depends on the weighting.
- R = 8: a single label's own sd is at most 0.177 (95% half-width ±0.35). Cell aggregates are honest; a single state's label is not a point value.
