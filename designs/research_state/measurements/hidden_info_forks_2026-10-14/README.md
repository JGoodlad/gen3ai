# Hidden-info KNOCK-IN FORKS — measurements

Registration: [`designs/endstate/design_hidden_info_binding.md`](../../../endstate/design_hidden_info_binding.md) §8
(the training axis), §8.4 (the crosses) and §8.5 (the rules). Run by the Training session.

**As played** (the arms, the pin and the deviations from §8.3's argv, each an orchestrator decision, 2026-10-14):

- 15 forks of `rb_es_end2_s2001/final_model.zip` (the closing test's adopted end state, 15,047,835 steps) to
  18,047,835 at P_hib `64a695dc` (the data-landed main commit: the learnset pre-evolution fix), seeds 3101–3103 ×
  arms `moves` (`--oracle-reveal ki_moves` = `moves:seen@0.95`), `items` (`ki_items`), `spreads` (`ki_spreads`),
  `species` (`species:unseen@0.95`, mass-matched) and the NO-CHANGE control `ctrl`, interleaved per seed.
- The levers are OBSERVATION reveals (`--oracle-reveal` + `--diagnostic-fork-consent`), not §8.1's belief-input
  injection (never built; `design_information_levers.md` §6).
- **No wrong-moves placebo** (decision (a)): the training observation refuses a `~placebo` source, so §8.5's K-VALID
  placebo clause and K-L-WORTH's Δ_L − Δ_WRONG are UNREADABLE in this batch.
- **The control-vs-parent (v8) read is CONFOUNDED** (decision (c)) by the learnset data change (the parent trained
  before it) plus the +3M steps.
- `--trunk-precision bf16` on every arm (the bf16 strength A/B's PASS, `../bf16_strength_ab_2026-10-12/`); the
  parent's recorded recipe TYPED on every fork (a fork does not inherit the cli_args recipe: untyped it falls to the
  parser defaults — 32 envs, no self-play, …).
- `--behaviour-check warn` uniformly (Amendment 2 of the closing test, carried over): the K9(b) tie-share clause warns,
  |d log π| < 1e-4 is held BY READ at each fork's bank. The `moves` arm's tie share sits above the 0.15 ceiling
  (median ≈ 0.16 on seeds 3101 and 3102 against 0.005–0.036 on every other arm): the moves reveal raises it.

- [`read_hib.py`](read_hib.py) — §8.5's rules CODED BEFORE ANY CELL IS PLAYED: `main.h2h.cross.cross_stat` at n = 3
  (df 4), the two-sided 95 % interval X5 §7.4 reports, rule 8 at ±0.05 pp; the per-lever verdict
  (`WORTH_FIRST_CLAUSE_ONLY` / `NULL` / `NOT_DETECTED`; K-L-WORTH itself is not declarable, decision (a)), the moves
  EXTENSION trigger, K-RANK on the paired contrast (items and spreads registered; species a declared descriptive
  extension), the v8 read descriptive and stamped. `python read_hib.py rehearse` (from a checkout's `src`) exercises
  every verdict branch and every rule-8 boundary on constructed matrices.

- The HONEST cross (§8.4's secondary read, no rule): each lever's seed i with its reveal WITHHELD at eval
  (`main.h2h --oracle-reveal-mode withheld`, landed `8272c6cb`) vs ctrl seed i — 12 seed-paired cells played at a
  second detached checkout of `8272c6cb` (the primary and v8 crosses stay at `64a695dc`; the checkpoints are the
  same, only the h2h code differs). `read_hib.py` reads it DESCRIPTIVELY: the withheld scores and the told − withheld
  difference on the same seed-paired cells.
- Provenance caution (the orchestrator's FINDING, 2026-10-15): the forks' `metadata.json` records `oracle_reveal: off`
  while `model_config.json` / the cli args hold the lever spec; `main.h2h` reads the level from `model_config.json`, so
  the crosses are unaffected — no bank or read script here reads the reveal from `metadata.json`.

## Results (2026-10-15)

The crosses were played on 2026-10-15, 02:11–03:55 (`crosses/run.log`).
- **Primary:** 36 cells at `64a695dc`, protocol `gen3_eval_protocol_v1_h2h_oracle_one_sided`, family `hib_forks_primary`, requests `hib_primary_<lever>`.
- **v8:** 3 cells at `64a695dc`, `gen3_eval_protocol_v1_h2h`, family `hib_forks_v8`.
- **Honest:** 12 cells at `8272c6cb`, `gen3_eval_protocol_v1_h2h_oracle_withheld`, family `hib_forks_honest`, requests `hib_honest_<lever>`.
- Every cell has 1,000 mirrored pairs, greedy vs greedy, on cuda. The K-VALID audit (`bank_hib.py` on all 15 forks) reads `invalid=[]`.
- The read is `crosses/read_cross_hib.py` applying `read_hib.decide` (`f68f686b`), at the `8272c6cb` checkout's `main.h2h.cross`.

The files:
- `crosses/plan_hib.py` — the plan and its refusals;
- `crosses/shas.json` — each final's sha256;
- `crosses/cells_*.json` — the planned [player, opponent] lists;
- `crosses/drive_hib.sh` — the driver, with its pin checks;
- `crosses/run.log` — every play-many and family-register line;
- `crosses/result/` — the read cells per lever, `h.json` and `verdict.json`.

**A read-script fix, before anything was read.** The driver's own read at 03:55:42 produced an EMPTY honest block. Its honest loop called `matrix_from_cells` per seed with all three of the lever's cells, and that function refuses any cell outside the declared row × column. The reasons landed only in `inconclusive["honest_*"]`, which the lever verdicts do not read. `read_cross_hib.py` now filters each seed pair's cell and was re-run at 03:56 with no new games. The primary, v8 and k_rank blocks are byte-identical to the first read (JSON equality). The first read is kept as `crosses/verdict_run1_honest_bug.json` and `crosses/h_run1_honest_bug.json`.

**The primary matrices** `h[i][j]` (pp; rows = the lever's seeds 3101 / 3102 / 3103; columns = ctrl seeds 3101 / 3102 / 3103; each row's three cells are separated by ` · `):

| lever | lever s3101 | lever s3102 | lever s3103 |
|---|---|---|---|
| moves | 48.625 · 48.550 · 49.200 | 48.350 · 50.025 · 50.150 | 51.000 · 50.725 · 49.950 |
| items | 51.400 · 49.975 · 50.950 | 50.650 · 50.250 · 51.050 | 48.350 · 49.050 · 49.450 |
| spreads | 50.000 · 51.400 · 52.875 | 48.600 · 48.475 · 52.050 | 49.425 · 48.950 · 51.275 |
| species | 48.700 · 48.775 · 49.700 | 49.200 · 49.175 · 49.800 | 48.100 · 51.250 · 49.950 |

**`crosses/result/verdict.json`, VERBATIM:**

```json
{
 "k_valid": {
  "invalid": [],
  "placebo_clause": "UNREADABLE (decision (a)): no wrong-moves arm \u2014 the training observation refuses a placebo source; K-VALID's placebo clause and K-L-WORTH's \u0394_L \u2212 \u0394_WRONG cannot be read in this batch"
 },
 "rule8_pp": 0.05,
 "interval": "two-sided 95 % Student-t (X5 \u00a77.4's reported interval)",
 "levers": {
  "moves": {
   "outcome": "NULL",
   "k_l_worth": "NOT DECLARABLE \u2014 UNREADABLE (decision (a)): no wrong-moves arm \u2014 the training observation refuses a placebo source; K-VALID's placebo clause and K-L-WORTH's \u0394_L \u2212 \u0394_WRONG cannot be read in this batch",
   "near_boundary": [],
   "delta_hat_pp": -0.38055555555556,
   "se_pp": 0.5337165868339965,
   "df": 4,
   "t_0.975": 2.7764451051977934,
   "ci95_pp": [
    -1.8623903606336825,
    1.1012792495225625
   ],
   "half_width_pp": 1.4818348050781225,
   "row_means_pp": [
    48.791666666666664,
    49.508333333333326,
    50.55833333333334
   ],
   "col_means_pp": [
    49.324999999999996,
    49.76666666666666,
    49.76666666666667
   ]
  },
  "items": {
   "outcome": "NULL",
   "k_l_worth": "NOT DECLARABLE \u2014 UNREADABLE (decision (a)): no wrong-moves arm \u2014 the training observation refuses a placebo source; K-VALID's placebo clause and K-L-WORTH's \u0394_L \u2212 \u0394_WRONG cannot be read in this batch",
   "near_boundary": [],
   "delta_hat_pp": 0.125,
   "se_pp": 0.6247221604663781,
   "df": 4,
   "t_0.975": 2.7764451051977934,
   "ci95_pp": [
    -1.6095067845354658,
    1.8595067845354658
   ],
   "half_width_pp": 1.7345067845354658,
   "row_means_pp": [
    50.775,
    50.65,
    48.949999999999996
   ],
   "col_means_pp": [
    50.133333333333326,
    49.758333333333326,
    50.48333333333333
   ]
  },
  "spreads": {
   "outcome": "NOT_DETECTED",
   "k_l_worth": "NOT DECLARABLE \u2014 UNREADABLE (decision (a)): no wrong-moves arm \u2014 the training observation refuses a placebo source; K-VALID's placebo clause and K-L-WORTH's \u0394_L \u2212 \u0394_WRONG cannot be read in this batch",
   "near_boundary": [],
   "delta_hat_pp": 0.3388888888888886,
   "se_pp": 1.0245444636640693,
   "df": 4,
   "t_0.975": 2.7764451051977934,
   "ci95_pp": [
    -2.505702572308715,
    3.183480350086492
   ],
   "half_width_pp": 2.8445914611976035,
   "row_means_pp": [
    51.425000000000004,
    49.708333333333336,
    49.88333333333333
   ],
   "col_means_pp": [
    49.34166666666667,
    49.60833333333333,
    52.06666666666667
   ]
  },
  "species": {
   "outcome": "NULL",
   "k_l_worth": "NOT DECLARABLE \u2014 UNREADABLE (decision (a)): no wrong-moves arm \u2014 the training observation refuses a placebo source; K-VALID's placebo clause and K-L-WORTH's \u0394_L \u2212 \u0394_WRONG cannot be read in this batch",
   "near_boundary": [],
   "delta_hat_pp": -0.5944444444444485,
   "se_pp": 0.42299819046398074,
   "df": 4,
   "t_0.975": 2.7764451051977934,
   "ci95_pp": [
    -1.7688756998656918,
    0.5799868109767947
   ],
   "half_width_pp": 1.1744312554212433,
   "row_means_pp": [
    49.05833333333334,
    49.39166666666667,
    49.76666666666667
   ],
   "col_means_pp": [
    48.666666666666664,
    49.73333333333333,
    49.81666666666666
   ]
  }
 },
 "moves_extension": {
  "extend": false,
  "half_width_pp": 1.4818348050781225,
  "near_boundary": []
 },
 "k_rank": {
  "pairs": {
   "items": {
    "moves_first": false,
    "near_boundary": [],
    "contrast_hat_pp": -0.50555555555556,
    "se_pp": 1.1148478831074342,
    "df": 4,
    "t_0.975": 2.7764451051977934,
    "ci95_pp": [
     -3.6008695036493172,
     2.5897583925381973
    ],
    "half_width_pp": 3.0953139480937573,
    "row_means_pp": [
     48.01666666666667,
     48.85833333333334,
     51.60833333333333
    ],
    "col_means_pp": [
     49.19166666666666,
     50.00833333333333,
     49.28333333333333
    ],
    "registered": true
   },
   "spreads": {
    "moves_first": false,
    "near_boundary": [],
    "contrast_hat_pp": -0.7194444444444485,
    "se_pp": 1.2675313812561524,
    "df": 4,
    "t_0.975": 2.7764451051977934,
    "ci95_pp": [
     -4.238675743617691,
     2.799786854728794
    ],
    "half_width_pp": 3.5192312991732426,
    "row_means_pp": [
     47.36666666666667,
     49.79999999999999,
     50.67499999999999
    ],
    "col_means_pp": [
     49.98333333333333,
     50.15833333333333,
     47.699999999999996
    ],
    "registered": true
   },
   "species": {
    "moves_first": false,
    "near_boundary": [],
    "contrast_hat_pp": 0.21388888888888857,
    "se_pp": 0.3816589842345052,
    "df": 4,
    "t_0.975": 2.7764451051977934,
    "ci95_pp": [
     -0.8457663297437652,
     1.2735441075215423
    ],
    "half_width_pp": 1.0596552186326538,
    "row_means_pp": [
     49.73333333333333,
     50.11666666666667,
     50.791666666666664
    ],
    "col_means_pp": [
     50.65833333333333,
     50.03333333333333,
     49.949999999999996
    ],
    "registered": false
   }
  },
  "MOVES_FIRST": false
 },
 "v8": {
  "descriptive": true,
  "stamp": "CONFOUNDED (decision (c)): the control trained on the learnset-fixed data/ (64a695dc) and its parent did not, so control-vs-parent mixes +3M steps with the data change",
  "cells_pp": [
   55.15,
   52.625,
   55.35
  ],
  "mean_minus_50_pp": 4.375
 },
 "honest": {
  "moves": {
   "descriptive": true,
   "withheld_cells_pp": [
    46.75,
    47.925000000000004,
    48.275
   ],
   "withheld_mean_minus_50_pp": -2.3499999999999943,
   "told_diag_cells_pp": [
    48.625,
    50.025,
    49.95
   ],
   "told_minus_withheld_pp": [
    1.875,
    2.0999999999999943,
    1.6750000000000043
   ],
   "told_minus_withheld_mean_pp": 1.8833333333333329
  },
  "items": {
   "descriptive": true,
   "withheld_cells_pp": [
    51.5,
    50.375,
    49.85
   ],
   "withheld_mean_minus_50_pp": 0.5749999999999957,
   "told_diag_cells_pp": [
    51.4,
    50.24999999999999,
    49.45
   ],
   "told_minus_withheld_pp": [
    -0.10000000000000142,
    -0.1250000000000071,
    -0.3999999999999986
   ],
   "told_minus_withheld_mean_pp": -0.2083333333333357
  },
  "spreads": {
   "descriptive": true,
   "withheld_cells_pp": [
    49.125,
    48.775,
    50.8
   ],
   "withheld_mean_minus_50_pp": -0.4333333333333371,
   "told_diag_cells_pp": [
    50.0,
    48.475,
    51.275000000000006
   ],
   "told_minus_withheld_pp": [
    0.875,
    -0.29999999999999716,
    0.4750000000000085
   ],
   "told_minus_withheld_mean_pp": 0.3500000000000038
  },
  "species": {
   "descriptive": true,
   "withheld_cells_pp": [
    47.775,
    46.25,
    48.25
   ],
   "withheld_mean_minus_50_pp": -2.5749999999999957,
   "told_diag_cells_pp": [
    48.699999999999996,
    49.175000000000004,
    49.95
   ],
   "told_minus_withheld_pp": [
    0.9249999999999972,
    2.9250000000000043,
    1.7000000000000028
   ],
   "told_minus_withheld_mean_pp": 1.8500000000000014
  }
 },
 "outcome": "READ"
}
```
