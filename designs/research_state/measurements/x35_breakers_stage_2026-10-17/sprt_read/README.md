# SPRT (the promotion-rule arm, STAGED_PLAN §4.2) look 1: **NULL, FINAL** (2026-10-24)

SPRT = C0 + `--promotion-sprt`, P\* + 20M at pin `f4b25238`, s3501–s3503. The control is C0 s3501–s3503 at the same
pin. Record: [`../reads/sprt_look1.json`](../reads/sprt_look1.json) (arm `SPRT`). Rule: `main.x35.readout.read`
(`python -m main.x35.readout decide read_look1_input.json` gives the same verdict).

| | s3501 | s3502 | s3503 | mean |
|---|---|---|---|---|
| S, SPRT (pp, se ≈ 0.55) | 56.52 | 56.79 | 56.89 | 56.73 |
| S, C0 (pp, G1's) | 57.52 | 56.60 | 56.81 | 56.98 |
| K9(b) / V-S1, V-S2 | PASS / PASS | PASS / PASS | PASS / PASS | |
| accepted / candidates (abandoned) | 4 / 10 (0) | 2 / 10 (0) | 4 / 10 (0) | 10 / 30 = 0.33 |
| N_prom SPRT / C0 | 4 / 1 | 2 / 1 | 4 / 1 | 10 / 3 |
| final median pool age, SPRT / C0 (M) | 60.1 / 72.1 | 68.1 / 72.1 | 60.1 / 72.1 | |
| steps/s SPRT / C0 (`../dose/`) | 1155 / 1384 | 1243 / 1418 | 1199 / 1424 | ×0.85 |

- **R1 Δ −0.25 pp [−1.65, +1.16]** (Welch 97.5 %, df 2.6), inside ±2 pp: NULL, final at look 1. The prediction was
  0.0, 80 % [−1.0, +1.5].
- **Mechanism:** the accept fraction of 0.33 lies inside the predicted [0.10, 0.90], and above today's rule's 0.04–0.33
  per candidate. N_prom is ≥ 2 × C0's, and the pool is younger than C0's.
- **Guards** G-F / G-C / G-K were not read: R1 is non-qualifying (the A1 ruling (a)). G-E is absent by design.
- **Dose** (reported): +14 % GPU-h (4.75 vs 4.16), above the readout's 10 % confound bar. Equal steps, so S is not
  biased by it.

## How it was read (CPU, `nice 19` on cores 5-7,13-15, one heavy slot per 40-min unit)

- [`s_read.py`](s_read.py) is `../g1/g1_read.py` with only the finals, the family (`x35_sprt_read_2026-10-24`) and the
  output changed. Each final was sha-checked and copied, and read only after its `X35END: DONE` line. All 249 ledger
  rows carry regime `adfdbee824c9eb0a`, the same as G1's. The results are in [`s_results.json`](s_results.json).
- [`assemble.py`](assemble.py) checks each seed's K9(b) record and V-S1 / V-S2 (`batch2_void.read_void`, which is
  `kit void`'s code) and STOPS on a VOID. It decides with the readout and STOPS on a qualifying R1 (the guard inputs
  were not played). It adds the mechanism, read from `sprt_promotion.jsonl` and `snapshots/`, and the dose.
- Raw ledger: `~/gen3ai_archive/x35_sprt_read/`. `models/` was only read.

## Hazards (FINDINGS)

1. **The playing code is this read's worktree** (main `eb99ab362`, + the s3503 records), not the X35 pin, as at G1.
   The regime id equals G1's.
2. **N_prom and age come from `snapshots/`, not TM-2's TB tag** `train/selfplay_promoted_steps`. They are the same
   count only while the pool evicts nothing (< 20 members: 14 seeded + ≤ 4 here). For the SPRT seeds the count equals
   each log's accepted count.
3. **The test's wall cost is ≈ 20× the memo's estimate** (205–263 s per cycle, against ≈ 11 s). That cost is the
   ×0.85 steps/s.
4. **C0's pool age (72M) sits inside TM-3's 55–76M band**, and C0's pooled N_prom of 3 is at the low edge of TM-3's
   3–8.
