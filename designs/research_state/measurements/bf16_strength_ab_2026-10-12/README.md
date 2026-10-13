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
