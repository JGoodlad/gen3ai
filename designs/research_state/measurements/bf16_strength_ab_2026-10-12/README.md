# bf16 STRENGTH A/B — measurements

Registration: [`designs/endstate/design_bf16_strength_ab.md`](../../../endstate/design_bf16_strength_ab.md)
(REGISTERED 2026-10-12, `981013f4`, before any seed). EB = `--arch endstate --trunk-precision bf16`, E = `--arch endstate`
(fp32), both at P_bf16 `f45d0c27`.

- [`futility_rule.py`](futility_rule.py) — stage 1's decision rule (§4), CODED BEFORE ANY CELL IS PLAYED: the 4 × 4
  cross of the 5M finals (rows EB 2001–2004, columns E 2001–2004), `main.h2h.cross.cross_stat` at n = 4 (df 6),
  the two-sided 90 % interval Δ̂ ± t_{0.95,6} √V̂ (t = 1.9432); STOP (futility) iff its upper end < −2.0 pp, rule 8 at
  1e-9 (a tie does not stop); otherwise CONTINUE to stage 2. `python futility_rule.py self-check` (run from a
  checkout's `src`) reads three constructed matrices: CONTINUE, STOP, and the rule-8 tie (CONTINUE, `near_boundary`).
- Stage 2's rule is the closing test's coded rule, unchanged:
  [`endstate_closing_test_2026-10-09/closing_rule.py`](../endstate_closing_test_2026-10-09/closing_rule.py) `decide`.

The cross artifacts (plan, shas, cells, h, verdict) are added here when each stage's cross is read.

## Stage 1 results (2026-10-12)

The 4 × 4 cross of the 5M finals, played at P_bf16 `f45d0c27` (a detached checkout, `src/` and `data/` clean) by
[`stage1_cross/drive_s1.sh`](stage1_cross/drive_s1.sh) right after the stage-1 chain's CHAIN DONE: 16 cells × 1,000
mirrored pairs (batches of 500), family `bf16_strength_steps`, request `bf16_s1_steps`, label `bf16_ab`, the archive's
eval ledger; 19:21:55 → 19:38:31 PT. Planned by [`stage1_cross/plan_s1.py`](stage1_cross/plan_s1.py) (every one of the
eight runs: `final_model.zip`, one `pin_history` entry at P_bf16 ≥ 5M, equal `init_num_threads`), read by
[`stage1_cross/read_s1.py`](stage1_cross/read_s1.py) with the coded rule [`futility_rule.py`](futility_rule.py)
(`7201f002`, before any cell). Cells: [`stage1_cross/result/cells.json`](stage1_cross/result/cells.json); the
finals' sha256: [`stage1_cross/shas.json`](stage1_cross/shas.json) (each run keeps its 5M final as
`final_model_stage1_5M.zip`, sha-identical, since stage 2's resume overwrites `final_model.zip`).

h (pp) = EB seed's score against E seed:

| EB \ E | E2001 | E2002 | E2003 | E2004 | row mean |
|---|---|---|---|---|---|
| EB2001 | 48.30 | 53.80 | 54.97 | 50.68 | 51.94 |
| EB2002 | 45.05 | 50.12 | 51.98 | 46.40 | 48.39 |
| EB2003 | 46.30 | 48.50 | 53.10 | 46.83 | 48.68 |
| EB2004 | 46.67 | 52.52 | 54.10 | 48.73 | 50.51 |
| col mean | 46.58 | 51.24 | 53.54 | 48.16 | |

The coded rule's output, verbatim ([`stage1_cross/result/verdict.json`](stage1_cross/result/verdict.json)):

```json
{
 "outcome": "CONTINUE",
 "upper_below_minus_2": false,
 "interval90_pp": [
  -3.550693698316416,
  3.3069436983164247
 ],
 "t_0.95": 1.9431802805153042,
 "delta_hat_pp": -0.12187499999999574,
 "se_pp": 1.7645396738006962,
 "df": 6,
 "s2_rows": 2.7633463541666603,
 "s2_cols": 9.691054687500008,
 "near_boundary": [],
 "stat": {
  "n": 4,
  "h_pp": [
   [
    48.3,
    53.800000000000004,
    54.974999999999994,
    50.675000000000004
   ],
   [
    45.050000000000004,
    50.125,
    51.975,
    46.400000000000006
   ],
   [
    46.300000000000004,
    48.5,
    53.1,
    46.825
   ],
   [
    46.675,
    52.525,
    54.1,
    48.725
   ]
  ],
  "delta_hat_pp": -0.12187499999999574,
  "row_means_pp": [
   51.9375,
   48.3875,
   48.681250000000006,
   50.50625
  ],
  "col_means_pp": [
   46.58125,
   51.2375,
   53.5375,
   48.15625
  ],
  "s2_rows": 2.7633463541666603,
  "s2_cols": 9.691054687500008,
  "v_hat": 3.1136002604166673,
  "se_pp": 1.7645396738006962,
  "df": 6,
  "t": 1.9144511456201816,
  "margin_pp": 3.5,
  "ci95_pp": [
   -4.439548039638414,
   4.195798039638422
  ],
  "upper95_one_sided_pp": 3.3069436983164247
 },
 "n_cells_read": 16
}
```

## Stage 2 results (2026-10-14)

The 8 × 8 cross of the 15M finals, played at P_bf16 `f45d0c27` (the same detached checkout, `src/` and `data/` clean)
by [`stage2_cross/drive_s2.sh`](stage2_cross/drive_s2.sh) right after the stage-2 chain's CHAIN DONE: 64 cells × 1,000
mirrored pairs (batches of 500), family `bf16_strength_steps` (registered at stage 1), request `bf16_s2_steps`, label
`bf16_ab`, the archive's eval ledger; 13:31:26 → 14:15:58 PT. Planned by [`stage2_cross/plan_s2.py`](stage2_cross/plan_s2.py)
(every one of the sixteen runs: `final_model.zip`, every `pin_history` entry at P_bf16, ≥ 15M — seeds 2001–2004 were
resumed from their own 5M finals at the same pin, each a single extended entry — and equal `init_num_threads`), read by
[`stage2_cross/read_s2.py`](stage2_cross/read_s2.py) with the closing test's coded rule, unchanged
([`../endstate_closing_test_2026-10-09/closing_rule.py`](../endstate_closing_test_2026-10-09/closing_rule.py) `decide`,
byte-identical at P_bf16 and at this commit). Cells: [`stage2_cross/result/cells.json`](stage2_cross/result/cells.json);
the finals' sha256: [`stage2_cross/shas.json`](stage2_cross/shas.json).

h (pp) = EB seed's score against E seed:

| EB \ E | E2001 | E2002 | E2003 | E2004 | E2005 | E2006 | E2007 | E2008 | row mean |
|---|---|---|---|---|---|---|---|---|---|
| EB2001 | 57.50 | 53.20 | 56.60 | 52.55 | 51.50 | 53.65 | 57.40 | 55.10 | 54.69 |
| EB2002 | 54.75 | 55.02 | 55.02 | 53.55 | 49.58 | 53.75 | 56.83 | 55.73 | 54.28 |
| EB2003 | 51.45 | 49.85 | 51.00 | 47.70 | 46.95 | 49.12 | 51.90 | 50.92 | 49.86 |
| EB2004 | 51.62 | 47.52 | 51.70 | 46.40 | 45.02 | 49.02 | 50.65 | 48.77 | 48.84 |
| EB2005 | 51.48 | 49.10 | 48.85 | 46.50 | 45.52 | 45.67 | 47.12 | 48.18 | 47.80 |
| EB2006 | 55.62 | 51.48 | 55.23 | 50.85 | 49.53 | 49.95 | 54.93 | 54.05 | 52.70 |
| EB2007 | 49.62 | 48.30 | 49.40 | 44.82 | 41.68 | 46.85 | 51.08 | 47.77 | 47.44 |
| EB2008 | 54.20 | 55.38 | 54.50 | 49.38 | 49.02 | 52.20 | 54.52 | 53.87 | 52.88 |
| col mean | 53.28 | 51.23 | 52.79 | 48.97 | 47.35 | 50.03 | 53.05 | 51.80 | |

The coded rule's output, verbatim ([`stage2_cross/result/verdict.json`](stage2_cross/result/verdict.json)):

```json
{
 "outcome": "PASS",
 "clause_a_includes_zero_or_above": true,
 "clause_b_lower_above_minus_2": true,
 "interval90_pp": [
  -1.186187842539114,
  3.311187842539114
 ],
 "t_0.95": 1.761310135774891,
 "delta_hat_pp": 1.0625,
 "se_pp": 1.2767131675818129,
 "df": 14,
 "s2_rows": 8.515711495535717,
 "s2_cols": 4.52426060267857,
 "t_sup": 0.8322151184611434,
 "near_boundary": [],
 "stat": {
  "n": 8,
  "h_pp": [
   [
    57.49999999999999,
    53.2,
    56.599999999999994,
    52.55,
    51.5,
    53.65,
    57.4,
    55.1
   ],
   [
    54.75,
    55.025,
    55.025,
    53.55,
    49.575,
    53.75,
    56.825,
    55.725
   ],
   [
    51.449999999999996,
    49.85,
    51.0,
    47.699999999999996,
    46.949999999999996,
    49.125,
    51.9,
    50.925
   ],
   [
    51.625,
    47.525,
    51.7,
    46.400000000000006,
    45.025,
    49.025,
    50.64999999999999,
    48.775
   ],
   [
    51.475,
    49.1,
    48.85,
    46.5,
    45.525,
    45.675,
    47.125,
    48.175000000000004
   ],
   [
    55.625,
    51.475,
    55.225,
    50.849999999999994,
    49.525000000000006,
    49.95,
    54.925000000000004,
    54.05
   ],
   [
    49.625,
    48.3,
    49.4,
    44.824999999999996,
    41.675000000000004,
    46.85,
    51.075,
    47.775
   ],
   [
    54.2,
    55.375,
    54.50000000000001,
    49.375,
    49.025,
    52.2,
    54.525,
    53.87499999999999
   ]
  ],
  "delta_hat_pp": 1.0625,
  "row_means_pp": [
   54.6875,
   54.278125,
   49.8625,
   48.840625,
   47.803125,
   52.703125,
   47.440625,
   52.884375
  ],
  "col_means_pp": [
   53.28125,
   51.23125,
   52.7875,
   48.96875,
   47.35,
   50.028125,
   53.053125,
   51.8
  ],
  "s2_rows": 8.515711495535717,
  "s2_cols": 4.52426060267857,
  "v_hat": 1.6299965122767859,
  "se_pp": 1.2767131675818129,
  "df": 14,
  "t": 3.5736296263331453,
  "margin_pp": 3.5,
  "ci95_pp": [
   -1.6757774061188444,
   3.8007774061188444
  ],
  "upper95_one_sided_pp": 3.311187842539114
 },
 "n_cells_read": 64
}
```

## Speed and run health (2026-10-14; REPORTED, never decided)

[`speed/train_ms_read.py`](speed/train_ms_read.py) → [`speed/train_ms.json`](speed/train_ms.json), read from the 16 runs' own
TensorBoard event files and `launcher_child.full.log`s (`python train_ms_read.py <models-dir> <out.json>`; read-only on the
models directory). `train_ms`, 153 PPO updates per run: the median of the eight seeds' medians is **54.255 s** for E (fp32; per-seed
medians 54.169–54.449 s) and **42.067 s** for EB (bf16; 41.994–42.152 s), a ratio of **0.7754 (−22.5 %)**; the pooled means are
55.44 s and 43.00 s. Health: 16 runs, **24** launcher-child legs each ending `Training complete` (seeds 2001–2004 trained a 5M leg and
then a same-pin resume to 15M), **0** `Traceback` lines.
