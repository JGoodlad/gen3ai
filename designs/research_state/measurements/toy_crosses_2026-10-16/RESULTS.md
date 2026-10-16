# Toy crosses: RESULTS (R-T5 equal compute, R-T6 toy at cap; both against EB 15M)

**Read 2026-10-16 11:42 PDT** by [`read_crosses.py`](read_crosses.py) → [`result/read.json`](result/read.json), as
registered in [`PREREG.md`](PREREG.md) (`72350102`, committed before any game).

**How it was played.**
- 48 cells × 250 mirrored pairs (24,000 games) on CPU, under rule 5b, at `72350102` (its `src/` is identical to
  `66e66237`).
- Families `toyx_rt5_eqcompute` / `toyx_rt6_cap` in `models/_ledger`.
- Wall time: 08:19–11:41 PDT, **3 h 22 min**.
  - That is ≈ 2.7–2.9 games/s with 50-pair batches, not the 20-pair pilot's 0.94 games/s. So the pre-registered cut
    to 250 pairs was more conservative than it needed to be (FINDING 3).
- Every cell completed 250 pairs, with 0 aborted games (84 draws in R-T5, 65 in R-T6) and no INCONCLUSIVE reason.

**DESCRIPTIVE: no verdict rule** (PREREG §5). The number is the toy's score minus 50 pp, where a draw counts ½. The
interval is the declared seeds-as-units rectangular statistic (`rect_stat.py`, NOT T40).

## The numbers

| read | toy checkpoint | pooled toy − 50 | 90 % interval (df) | 95 % interval | per seed (s4001 / s4002 / s4003) |
|---|---|---|---|---|---|
| **R-T5, equal compute** | 48M / 46M / 47M (the toy's wall clock = EB's to 15M) | **+6.48 pp** | [+4.54, +8.43] (8.6) | [+4.08, +8.89] | +5.90 / +6.34 / +7.21 |
| **R-T6, toy at its 100M cap** | `s4001b` / `s4002` / `s4003` final | **+10.20 pp** | [+7.50, +12.91] (4.0) | [+6.68, +13.73] | +8.10 / +11.24 / +11.27 |
| *R-T1, matched steps (W4; retrospective interval)* | 15M | *−14.10 pp* | *[−17.16, −11.04] (4.7)* | | *−16.11 / −14.16 / −12.02* |
| *R-T1b, matched 5M (W4; retrospective interval)* | 5M | *−25.51 pp* | *[−28.04, −22.98] (8.9)* | | *−25.12 / −26.86 / −24.55* |

Per-seed 90 % intervals, over each seed's 8 cells (df 7):
- R-T5: s4001 [+4.02, +7.78], s4002 [+4.12, +8.55], s4003 [+4.87, +9.55];
- R-T6: s4001b [+6.28, +9.92], s4002 [+8.90, +13.58], s4003 [+9.94, +12.61].

**Every one of the 48 cells is above 50 %.**
- R-T5 cells run 51.5–61.5 %. Its EB column means run 53.0–59.9 %, and the column (EB seed) spread dominates:
  s²_C 7.74 against s²_R 0.45.
- R-T6 cells run 55.1–67.2 %.

## What they say

1. **At equal training compute, the toy beats production.** With the same GPU wall clock production needs to reach
   15M (≈ 2.5 h), T-ENT reaches ≈ 46–48M steps and scores **≈ +6.5 pp** against production's 15M finals. All three
   toy seeds agree (+5.9 to +7.2), and the interval is clear of 0.
   - At matched STEPS the same toy is −14.1 pp at 15M and −25.5 pp at 5M (W4).
   - So production's per-step advantage is smaller than the toy's ≈ 3.3× step-throughput advantage over 2.5 GPU-h.
2. **The toy at its 100M cap scores ≈ +10.2 pp against production at 15M.** This is not a ceiling read. Production at
   15M is far from its plateau (X26 is the plateau reference, R-T2).
3. **What this does NOT say.**
   - It does not say the machinery buys nothing. That is G_p(cap) (R-T2, toy cap vs X26 plateau), and §3.6's classes
     are not read from these crosses.
   - It does not say the toy would still lead at a LARGER equal budget. Production keeps climbing past 15M too, and
     the equal-compute comparison was made at only one budget (≈ 2.5 GPU-h).
   - What it does show: at the budget where our 8-seed A/Bs read production (15M), a 1.42M-parameter toy trained for
     the same wall clock is stronger.

## FINDINGS (standing rule 7)

1. **The equal-compute step is ≈ 46–48M, not the brief's ~50–55M.** The toy's speed ratio is 3.33–3.44× at 15M and
   falls as its games lengthen (`time_to_step.json`).
2. **The R-T6 row of `s4001b` inherits ≈ 9.2M post-canary steps** (62.83M → 72.0M; readiness F17). It is the lowest
   R-T6 row (+8.1 vs +11.2 / +11.3). Dropping it moves the pooled R-T6 point to ≈ +11.3 pp. Whether the canary-era
   steps cost it anything is UNVERIFIED (one seed).
3. **The CPU estimate was 3× pessimistic.** The 20-pair pilot (0.94 games/s) under-read the 50-pair batches' rate
   (2.65–2.92 games/s). The 500-pair design would have fit in ≈ 7 h. The 250-pair cut, decided before playing, stands.
4. **Equal compute is GPU wall clock on a shared box**, and the agent CPU load in EB's (10-12/13) and the toys'
   (10-15) windows was not recorded. A FLOP-matched step would be far larger (17.4 vs 95.7 MFLOP per row).
5. **R-T5 / R-T6 are new ids.** The brief's R-T3 / R-T4 already name readiness §3.5's external-units and cost reads.
6. **The interval is not T40's registered estimator.** It is the declared seeds-as-units Welch analogue, conservative
   in the cell-noise term (self-check 8 / 8; MC coverage 0.912 at 3 × 8).
