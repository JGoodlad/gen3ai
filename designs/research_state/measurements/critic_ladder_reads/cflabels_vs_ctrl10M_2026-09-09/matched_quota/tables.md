# matched-quota re-read — 30 subsample seeds, 2000 bootstrap draws per block, block seed 0

## Frames

| view | battles | teams | states | own-team t1 decoder battles | opp-class battles |
|---|---|---|---|---|---|
| control (as traced, 5/10/5) | 198 | 76 | 6080 | 104 | 198 |
| arm MATCHED (8/12/5, median of 30) | 197 | 106 | 6257 | 62 | 197 |
| arm decoder_matched | 247 | 118 | 7692 | 102 | 247 |
| arm x2 | 326 | 132 | 10114 | 176 | 326 |
| arm x4 | 531 | 167 | 16120 | 353 | 531 |
| arm FULL (as traced, 40/40/10) | 627 | 183 | 18739 | 429 | 627 |

## Table A — every conditioning row on the BATTLE-MATCHED frame (caps 8/12/5 — the control's own realized profile)

| row | arm SUBSAMPLED (median [2.5,97.5] over seeds) | arm FULL | control | Δ, per-seed median CI | Δ, POOLED CI | pooled label | seeds whose own CI clears 0 |
|---|---|---|---|---|---|---|---|
| `cond.spread_ratio.t1_3` | +0.0000 [+0.0000, +0.1629] | +0.0000 | +0.1152 | -0.1152 [-0.4367, +0.2747] | -0.1152 [-0.4658, +0.2577] | NOT DETECTED | 0/30 |
| `cond.spread_ratio_raw.t1_3` | +0.2254 [+0.1534, +0.2926] | +0.1077 | +0.3333 | -0.1066 [-0.3961, +0.0977] | -0.1079 [-0.3718, +0.1291] | NOT DETECTED | 1/30 |
| `cond.spread_delta.t1_3` | -0.1026 [-0.1026, -0.0859] | -0.1026 | -0.0886 | -0.0140 [-0.0604, +0.0386] | -0.0140 [-0.0605, +0.0350] | NOT DETECTED | 0/30 |
| `cond.spread_ratio.all` | +0.5027 [+0.3716, +0.5673] | +0.5149 | +0.7895 | -0.2852 [-0.6915, +0.1088] | -0.2867 [-0.7134, +0.1124] | NOT DETECTED | 1/30 |
| `cond.spread_ratio_raw.all` | +0.5456 [+0.4518, +0.6076] | +0.5108 | +0.8053 | -0.2594 [-0.6199, +0.0892] | -0.2597 [-0.6432, +0.1065] | NOT DETECTED | 1/30 |
| `cond.spread_delta.all` | -0.0510 [-0.0644, -0.0444] | -0.0498 | -0.0211 | -0.0298 [-0.0782, +0.0190] | -0.0299 [-0.0783, +0.0180] | NOT DETECTED | 0/30 |
| `cond.elo_slope` | +0.0191 [+0.0169, +0.0232] | +0.0191 | +0.0098 | +0.0095 [-0.0047, +0.0237] | +0.0093 [-0.0044, +0.0242] | NOT DETECTED | 0/30 |
| `cond.own_team_r2.t1` | -0.0250 [-0.1404, +0.2491] | +0.0604 | -0.0237 | +0.0018 [-0.5144, +0.2142] | -0.0013 [-0.2436, +0.3461] | NOT DETECTED | 4/30 |
| `cond.own_team_r2.all` | -0.0467 [-0.1760, +0.1542] | +0.0837 | -0.0098 | -0.0283 [-0.3414, +0.0904] | -0.0369 [-0.3011, +0.2041] | NOT DETECTED | 5/30 |
| `cond.opp_class_auc.t1` | +0.5312 [+0.3845, +0.6233] | +0.5552 | +0.4619 | +0.0693 [-0.0755, +0.2147] | +0.0693 [-0.1317, +0.2431] | NOT DETECTED | 3/30 |

## Table A2 — the same rows on the DECODER-MATCHED frame (caps 11/16/5). The arm carries more distinct teams per battle, so battle-matching UNDER-fills its own-team decoder; this rung matches the decoder's own battle count instead

| row | arm SUBSAMPLED (median [2.5,97.5] over seeds) | arm FULL | control | Δ, per-seed median CI | Δ, POOLED CI | pooled label | seeds whose own CI clears 0 |
|---|---|---|---|---|---|---|---|
| `cond.spread_ratio.t1_3` | +0.0000 [+0.0000, +0.1812] | +0.0000 | +0.1152 | -0.1152 [-0.5371, +0.1322] | -0.1152 [-0.4665, +0.1966] | NOT DETECTED | 0/30 |
| `cond.spread_ratio_raw.t1_3` | +0.2004 [+0.1280, +0.2537] | +0.1077 | +0.3333 | -0.1325 [-0.3958, +0.0441] | -0.1329 [-0.4113, +0.0615] | NOT DETECTED | 5/30 |
| `cond.spread_delta.t1_3` | -0.1026 [-0.1026, -0.0840] | -0.1026 | -0.0886 | -0.0140 [-0.0708, +0.0224] | -0.0140 [-0.0629, +0.0310] | NOT DETECTED | 0/30 |
| `cond.spread_ratio.all` | +0.5056 [+0.3692, +0.5828] | +0.5149 | +0.7895 | -0.2828 [-0.6834, +0.0563] | -0.2839 [-0.7145, +0.0964] | NOT DETECTED | 6/30 |
| `cond.spread_ratio_raw.all` | +0.5308 [+0.4116, +0.6001] | +0.5108 | +0.8053 | -0.2734 [-0.6287, +0.0287] | -0.2745 [-0.6534, +0.0798] | NOT DETECTED | 6/30 |
| `cond.spread_delta.all` | -0.0507 [-0.0647, -0.0428] | -0.0498 | -0.0211 | -0.0295 [-0.0756, +0.0131] | -0.0296 [-0.0793, +0.0170] | NOT DETECTED | 0/30 |
| `cond.elo_slope` | +0.0199 [+0.0168, +0.0234] | +0.0191 | +0.0098 | +0.0102 [-0.0030, +0.0238] | +0.0101 [-0.0041, +0.0235] | NOT DETECTED | 2/30 |
| `cond.own_team_r2.t1` | +0.0038 [-0.0770, +0.2352] | +0.0604 | -0.0237 | +0.0287 [-0.0808, +0.1327] | +0.0275 [-0.1162, +0.3785] | NOT DETECTED | 3/30 |
| `cond.own_team_r2.all` | -0.0034 [-0.0571, +0.1290] | +0.0837 | -0.0098 | +0.0070 [-0.0800, +0.0824] | +0.0063 [-0.1150, +0.1846] | NOT DETECTED | 1/30 |
| `cond.opp_class_auc.t1` | +0.5398 [+0.4019, +0.6406] | +0.5552 | +0.4619 | +0.0825 [-0.0461, +0.2104] | +0.0779 [-0.1199, +0.2454] | NOT DETECTED | 6/30 |

## Table B — frame-size curve (the ARM alone)

| rung | caps (win/loss/draw) | battles | teams | own-team t1 decoder battles | `cond.own_team_r2.t1` median [2.5,97.5] | `cond.own_team_r2.all` |
|---|---|---|---|---|---|---|
| matched | 8/12/5 | 197 | 106 | 62 | -0.0250 [-0.1404, +0.2491] | -0.0467 [-0.1760, +0.1542] |
| decoder_matched | 11/16/5 | 247 | 118 | 102 | +0.0038 [-0.0770, +0.2352] | -0.0034 [-0.0571, +0.1290] |
| x2 | 16/24/10 | 326 | 132 | 176 | +0.0369 [-0.0416, +0.1321] | +0.0263 [-0.0168, +0.0904] |
| x4 | 32/48/10 | 531 | 167 | 353 | +0.0684 [+0.0437, +0.0999] | +0.0596 [+0.0275, +0.0910] |
| full | as traced | 627 | 183 | 429 | +0.0604 [+0.0604, +0.0604] | +0.0837 [+0.0837, +0.0837] |

Control for reference: 198 battles, 76 teams, 104 decoder battles, `cond.own_team_r2.t1` = -0.0237 [-0.11218100414369468, -0.006659875393908483]

## Table C — the arm against ITSELF (matched frame minus full frame)

| row | arm FULL | arm MATCHED (median) | Δ (matched - full), POOLED CI | label |
|---|---|---|---|---|
| `cond.spread_ratio.t1_3` | +0.0000 | +0.0000 | +0.0000 [-0.1282, +0.3577] | NOT DETECTED |
| `cond.spread_ratio_raw.t1_3` | +0.1077 | +0.2254 | +0.1177 [+0.0131, +0.3546] | DETECTED |
| `cond.spread_delta.t1_3` | -0.1026 | -0.1026 | +0.0000 [-0.0311, +0.0517] | NOT DETECTED |
| `cond.spread_ratio.all` | +0.5149 | +0.5027 | -0.0122 [-0.2851, +0.3398] | NOT DETECTED |
| `cond.spread_ratio_raw.all` | +0.5108 | +0.5456 | +0.0348 [-0.2067, +0.3598] | NOT DETECTED |
| `cond.spread_delta.all` | -0.0498 | -0.0510 | -0.0012 [-0.0409, +0.0446] | NOT DETECTED |
| `cond.elo_slope` | +0.0191 | +0.0191 | -0.0001 [-0.0124, +0.0125] | NOT DETECTED |
| `cond.own_team_r2.t1` | +0.0604 | -0.0250 | -0.0855 [-0.3438, +0.2439] | NOT DETECTED |
| `cond.own_team_r2.all` | +0.0837 | -0.0467 | -0.1304 [-0.4017, +0.1034] | NOT DETECTED |
| `cond.opp_class_auc.t1` | +0.5552 | +0.5312 | -0.0239 [-0.2148, +0.1289] | NOT DETECTED |

<!-- machine-readable -->
```json
{
 "n_seeds": 30,
 "boot": 2000,
 "block_seed": 0,
 "frames": {
  "control": {
   "trace_dir": "/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032",
   "step": 10000032,
   "selection_schema": 2,
   "n_states": 6080,
   "n_battles": 198,
   "n_opponents": 12,
   "n_teams": 76,
   "n_draw_battles_excluded": 6,
   "max_abs_values_minus_winprobs": 0.0,
   "refusals": [],
   "n_cells": 12,
   "boot": 2000,
   "seed": 0,
   "states_per_battle_cap": 2,
   "min_team_battles": 4,
   "score_frames": {
    "cond.own_team_r2.t1": {
     "n_states": 108,
     "n_battles": 104
    },
    "cond.own_team_r2.all": {
     "n_states": 208,
     "n_battles": 104
    },
    "cond.opp_class_auc.t1": {
     "n_states": 210,
     "n_battles": 198
    }
   },
   "recorded_v_note": "every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model \u2014 no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles."
  },
  "arm_full": {
   "trace_dir": "/home/goodlad/dev/gen3ai/models/ai_v12_12_ladder_cflabels/eval_traces/step_10000032",
   "step": 10000032,
   "selection_schema": 2,
   "n_states": 18739,
   "n_battles": 627,
   "n_opponents": 12,
   "n_teams": 183,
   "n_draw_battles_excluded": 5,
   "max_abs_values_minus_winprobs": 0.0,
   "refusals": [],
   "n_cells": 12,
   "boot": 2000,
   "seed": 0,
   "states_per_battle_cap": 2,
   "min_team_battles": 4,
   "score_frames": {
    "cond.own_team_r2.t1": {
     "n_states": 443,
     "n_battles": 429
    },
    "cond.own_team_r2.all": {
     "n_states": 858,
     "n_battles": 429
    },
    "cond.opp_class_auc.t1": {
     "n_states": 655,
     "n_battles": 627
    }
   },
   "recorded_v_note": "every conditioning row is computed on ONE eval cycle from the `win_probs` the trace npz RECORDED, which within a cycle IS that cycle's own model \u2014 no model forward is done. Pooling a recorded V across cycles would import the trainee's own improvement as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against 0.066 frozen); this module never pools cycles."
  },
  "arm_matched_median_battles": 197.0
 },
 "table_a": [
  {
   "key": "cond.spread_ratio.t1_3",
   "arm_median": 0.0,
   "arm_q": [
    0.0,
    0.16291725422705255
   ],
   "arm_full": 0.0,
   "control": 0.11523083617537427,
   "delta_median_seed": {
    "delta": -0.11523083617537427,
    "ci": [
     -0.43667247120922753,
     0.27472798331224657
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": -0.11523083617537427,
    "ci": [
     -0.4657890353812985,
     0.25768642682353604
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 0,
   "n_seeds": 30
  },
  {
   "key": "cond.spread_ratio_raw.t1_3",
   "arm_median": 0.22537954791598086,
   "arm_q": [
    0.1534257429558875,
    0.2925696719343823
   ],
   "arm_full": 0.10768909390824295,
   "control": 0.3332903977158929,
   "delta_median_seed": {
    "delta": -0.10663666463287977,
    "ci": [
     -0.3961290769587315,
     0.09772078581405834
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": -0.10791084979991206,
    "ci": [
     -0.37180331553546664,
     0.12909032916145644
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 1,
   "n_seeds": 30
  },
  {
   "key": "cond.spread_delta.t1_3",
   "arm_median": -0.10256845459880869,
   "arm_q": [
    -0.10256845459880869,
    -0.08585828360525868
   ],
   "arm_full": -0.10256845459880869,
   "control": -0.08858650626591769,
   "delta_median_seed": {
    "delta": -0.013981948332891,
    "ci": [
     -0.06035546732654004,
     0.03858142307757983
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": -0.013981948332891,
    "ci": [
     -0.06052754926390079,
     0.03504345978382379
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 0,
   "n_seeds": 30
  },
  {
   "key": "cond.spread_ratio.all",
   "arm_median": 0.502712576588391,
   "arm_q": [
    0.3716427042651436,
    0.5673027676108996
   ],
   "arm_full": 0.5148877315281184,
   "control": 0.7894593261490331,
   "delta_median_seed": {
    "delta": -0.2852109657703078,
    "ci": [
     -0.6915008502322844,
     0.10879696171646924
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": -0.28674674956064217,
    "ci": [
     -0.7133884725446817,
     0.11239301134504935
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 1,
   "n_seeds": 30
  },
  {
   "key": "cond.spread_ratio_raw.all",
   "arm_median": 0.5455920831105022,
   "arm_q": [
    0.45183652801643037,
    0.6076109014356526
   ],
   "arm_full": 0.510782722284472,
   "control": 0.8053156756062707,
   "delta_median_seed": {
    "delta": -0.259410082439697,
    "ci": [
     -0.6198808147492328,
     0.08924720623363416
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": -0.2597235924957685,
    "ci": [
     -0.6432405165989584,
     0.10645106523293621
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 1,
   "n_seeds": 30
  },
  {
   "key": "cond.spread_delta.all",
   "arm_median": -0.05100600251075217,
   "arm_q": [
    -0.06444963675941082,
    -0.04438108643533161
   ],
   "arm_full": -0.049757215684083284,
   "control": -0.021080145518075633,
   "delta_median_seed": {
    "delta": -0.029768334022704047,
    "ci": [
     -0.07823334869741241,
     0.01904820853059853
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": -0.02992585699267654,
    "ci": [
     -0.07829881256072438,
     0.017982275107585176
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 0,
   "n_seeds": 30
  },
  {
   "key": "cond.elo_slope",
   "arm_median": 0.019055893524487923,
   "arm_q": [
    0.016915527974521682,
    0.023214013640559607
   ],
   "arm_full": 0.019110211066039478,
   "control": 0.009760216244029,
   "delta_median_seed": {
    "delta": 0.009484603896476616,
    "ci": [
     -0.004738420159580425,
     0.023680403226127623
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": 0.009295677280458922,
    "ci": [
     -0.004353256089242006,
     0.024229735365538803
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 0,
   "n_seeds": 30
  },
  {
   "key": "cond.own_team_r2.t1",
   "arm_median": -0.025028141437328455,
   "arm_q": [
    -0.14038329032160776,
    0.24908624221829775
   ],
   "arm_full": 0.060426281733685805,
   "control": -0.023692051554759397,
   "delta_median_seed": {
    "delta": 0.0018257206485576827,
    "ci": [
     -0.5143934557752909,
     0.2142070421780261
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": -0.0013360898825690581,
    "ci": [
     -0.24364073544019516,
     0.34608618149005527
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 4,
   "n_seeds": 30
  },
  {
   "key": "cond.own_team_r2.all",
   "arm_median": -0.046707064433747725,
   "arm_q": [
    -0.1760493541463572,
    0.15424252729237833
   ],
   "arm_full": 0.08369569312806924,
   "control": -0.009768495081126094,
   "delta_median_seed": {
    "delta": -0.028251501790905298,
    "ci": [
     -0.3414104040985028,
     0.09042018621586279
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": -0.03693856935262163,
    "ci": [
     -0.3011482132203675,
     0.20405272496386354
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 5,
   "n_seeds": 30
  },
  {
   "key": "cond.opp_class_auc.t1",
   "arm_median": 0.5312341431865744,
   "arm_q": [
    0.3845135402432784,
    0.623282151348623
   ],
   "arm_full": 0.5551605557284238,
   "control": 0.4619072892874479,
   "delta_median_seed": {
    "delta": 0.06933597604729819,
    "ci": [
     -0.07552993063642183,
     0.21466537409831424
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": 0.06932685389912646,
    "ci": [
     -0.1316830890836568,
     0.24311672624269548
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 3,
   "n_seeds": 30
  }
 ],
 "table_a_decoder_matched": [
  {
   "key": "cond.spread_ratio.t1_3",
   "arm_median": 0.0,
   "arm_q": [
    0.0,
    0.18124107726904348
   ],
   "arm_full": 0.0,
   "control": 0.11523083617537427,
   "delta_median_seed": {
    "delta": -0.11523083617537427,
    "ci": [
     -0.5370667384620318,
     0.13224092830392545
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": -0.11523083617537427,
    "ci": [
     -0.466536415251381,
     0.19662136643266048
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 0,
   "n_seeds": 30
  },
  {
   "key": "cond.spread_ratio_raw.t1_3",
   "arm_median": 0.20035360324491155,
   "arm_q": [
    0.12801028271709688,
    0.25368048705676316
   ],
   "arm_full": 0.10768909390824295,
   "control": 0.3332903977158929,
   "delta_median_seed": {
    "delta": -0.13245158407870153,
    "ci": [
     -0.3957739600807439,
     0.044101263577492555
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": -0.13293679447098136,
    "ci": [
     -0.41134841384056553,
     0.06151809698599002
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 5,
   "n_seeds": 30
  },
  {
   "key": "cond.spread_delta.t1_3",
   "arm_median": -0.10256845459880869,
   "arm_q": [
    -0.10256845459880869,
    -0.08397883739349962
   ],
   "arm_full": -0.10256845459880869,
   "control": -0.08858650626591769,
   "delta_median_seed": {
    "delta": -0.013981948332891,
    "ci": [
     -0.07083617231626117,
     0.022392163844037954
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": -0.013981948332891,
    "ci": [
     -0.06287935973670351,
     0.031025751039515447
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 0,
   "n_seeds": 30
  },
  {
   "key": "cond.spread_ratio.all",
   "arm_median": 0.5056019275422023,
   "arm_q": [
    0.3692234926649251,
    0.5828056160210355
   ],
   "arm_full": 0.5148877315281184,
   "control": 0.7894593261490331,
   "delta_median_seed": {
    "delta": -0.2827962810345849,
    "ci": [
     -0.6834488511941514,
     0.05630496284052198
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": -0.2838573986068308,
    "ci": [
     -0.7145226012387221,
     0.09640771022815685
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 6,
   "n_seeds": 30
  },
  {
   "key": "cond.spread_ratio_raw.all",
   "arm_median": 0.5307782736398129,
   "arm_q": [
    0.4116478124248426,
    0.6000548825434159
   ],
   "arm_full": 0.510782722284472,
   "control": 0.8053156756062707,
   "delta_median_seed": {
    "delta": -0.2734035035598542,
    "ci": [
     -0.6287010357510623,
     0.028733432576760608
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": -0.2745374019664578,
    "ci": [
     -0.653410701634468,
     0.07980584074460537
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 6,
   "n_seeds": 30
  },
  {
   "key": "cond.spread_delta.all",
   "arm_median": -0.05070964624862614,
   "arm_q": [
    -0.06469777155459273,
    -0.04279098323202437
   ],
   "arm_full": -0.049757215684083284,
   "control": -0.021080145518075633,
   "delta_median_seed": {
    "delta": -0.029520663541017608,
    "ci": [
     -0.07559442960070682,
     0.013117704169691703
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": -0.029629500730550505,
    "ci": [
     -0.07928256499396201,
     0.016967741522126135
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 0,
   "n_seeds": 30
  },
  {
   "key": "cond.elo_slope",
   "arm_median": 0.019850594107730088,
   "arm_q": [
    0.016763736919083092,
    0.023423569835350187
   ],
   "arm_full": 0.019110211066039478,
   "control": 0.009760216244029,
   "delta_median_seed": {
    "delta": 0.010151933748984796,
    "ci": [
     -0.002958853579865879,
     0.02378176373978964
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": 0.010090377863701087,
    "ci": [
     -0.004140485750866645,
     0.023494839892172648
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 2,
   "n_seeds": 30
  },
  {
   "key": "cond.own_team_r2.t1",
   "arm_median": 0.0038065863274333034,
   "arm_q": [
    -0.07700449487078237,
    0.2351789113544285
   ],
   "arm_full": 0.060426281733685805,
   "control": -0.023692051554759397,
   "delta_median_seed": {
    "delta": 0.02868358701121032,
    "ci": [
     -0.08079546830150423,
     0.1326604328628678
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": 0.0274986378821927,
    "ci": [
     -0.11615041190719218,
     0.37845589949468417
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 3,
   "n_seeds": 30
  },
  {
   "key": "cond.own_team_r2.all",
   "arm_median": -0.003426332749361549,
   "arm_q": [
    -0.057131769832839574,
    0.12895031417909314
   ],
   "arm_full": 0.08369569312806924,
   "control": -0.009768495081126094,
   "delta_median_seed": {
    "delta": 0.007008242610482274,
    "ci": [
     -0.08002281423468904,
     0.08241380454053292
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": 0.006342162331764545,
    "ci": [
     -0.11495309394149227,
     0.18459627201639595
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 1,
   "n_seeds": 30
  },
  {
   "key": "cond.opp_class_auc.t1",
   "arm_median": 0.5398333064793585,
   "arm_q": [
    0.4019233287670812,
    0.6405733656783666
   ],
   "arm_full": 0.5551605557284238,
   "control": 0.4619072892874479,
   "delta_median_seed": {
    "delta": 0.08247173770558608,
    "ci": [
     -0.046108774384471446,
     0.21037056513987548
    ],
    "n_draws": 2000
   },
   "delta_pooled": {
    "delta": 0.07792601719191056,
    "ci": [
     -0.11988071217857008,
     0.24543724141720027
    ],
    "n_draws": 2000,
    "label": "NOT DETECTED",
    "qualifier": "CI covers zero",
    "floor": null,
    "clears_zero": false,
    "clears_floor": false
   },
   "seeds_clearing_zero": 6,
   "n_seeds": 30
  }
 ],
 "table_b": [
  {
   "rung": "matched",
   "caps": [
    8,
    12,
    5
   ],
   "battles": 197.0,
   "teams": 105.5,
   "decoder_battles_t1": 62.0,
   "own_team_t1_median": -0.025028141437328455,
   "own_team_all_median": -0.046707064433747725,
   "n_seeds": 30
  },
  {
   "rung": "decoder_matched",
   "caps": [
    11,
    16,
    5
   ],
   "battles": 247.0,
   "teams": 118.0,
   "decoder_battles_t1": 102.0,
   "own_team_t1_median": 0.0038065863274333034,
   "own_team_all_median": -0.003426332749361549,
   "n_seeds": 30
  },
  {
   "rung": "x2",
   "caps": [
    16,
    24,
    10
   ],
   "battles": 326.0,
   "teams": 131.5,
   "decoder_battles_t1": 176.0,
   "own_team_t1_median": 0.036868234213135564,
   "own_team_all_median": 0.026258657196000734,
   "n_seeds": 20
  },
  {
   "rung": "x4",
   "caps": [
    32,
    48,
    10
   ],
   "battles": 531.0,
   "teams": 167.0,
   "decoder_battles_t1": 353.0,
   "own_team_t1_median": 0.06842611619088174,
   "own_team_all_median": 0.059648600754258485,
   "n_seeds": 20
  },
  {
   "rung": "full",
   "caps": [
    null,
    null,
    null
   ],
   "battles": 627.0,
   "teams": 183.0,
   "decoder_battles_t1": 429.0,
   "own_team_t1_median": 0.060426281733685805,
   "own_team_all_median": 0.08369569312806924,
   "n_seeds": 1
  }
 ],
 "table_c": [
  {
   "key": "cond.spread_ratio.t1_3",
   "arm_full": 0.0,
   "arm_matched": 0.0,
   "delta": 0.0,
   "ci": [
    -0.12818820998134423,
    0.35770660256833336
   ],
   "n_draws": 2000,
   "label": "NOT DETECTED",
   "qualifier": "CI covers zero",
   "floor": null,
   "clears_zero": false,
   "clears_floor": false
  },
  {
   "key": "cond.spread_ratio_raw.t1_3",
   "arm_full": 0.10768909390824295,
   "arm_matched": 0.22537954791598086,
   "delta": 0.11769045400773791,
   "ci": [
    0.013127817649925634,
    0.354581330592128
   ],
   "n_draws": 2000,
   "label": "DETECTED",
   "qualifier": "vs ZERO \u2014 NO FLOOR",
   "floor": null,
   "clears_zero": true,
   "clears_floor": false
  },
  {
   "key": "cond.spread_delta.t1_3",
   "arm_full": -0.10256845459880869,
   "arm_matched": -0.10256845459880869,
   "delta": 0.0,
   "ci": [
    -0.031068705853492227,
    0.051661358120258974
   ],
   "n_draws": 2000,
   "label": "NOT DETECTED",
   "qualifier": "CI covers zero",
   "floor": null,
   "clears_zero": false,
   "clears_floor": false
  },
  {
   "key": "cond.spread_ratio.all",
   "arm_full": 0.5148877315281184,
   "arm_matched": 0.502712576588391,
   "delta": -0.012175154939727428,
   "ci": [
    -0.2851037575402544,
    0.3397997675388037
   ],
   "n_draws": 2000,
   "label": "NOT DETECTED",
   "qualifier": "CI covers zero",
   "floor": null,
   "clears_zero": false,
   "clears_floor": false
  },
  {
   "key": "cond.spread_ratio_raw.all",
   "arm_full": 0.510782722284472,
   "arm_matched": 0.5455920831105022,
   "delta": 0.034809360826030256,
   "ci": [
    -0.2066975256000944,
    0.3598180462081731
   ],
   "n_draws": 2000,
   "label": "NOT DETECTED",
   "qualifier": "CI covers zero",
   "floor": null,
   "clears_zero": false,
   "clears_floor": false
  },
  {
   "key": "cond.spread_delta.all",
   "arm_full": -0.049757215684083284,
   "arm_matched": -0.05100600251075217,
   "delta": -0.0012487868266688879,
   "ci": [
    -0.04093992529827662,
    0.04456377738365299
   ],
   "n_draws": 2000,
   "label": "NOT DETECTED",
   "qualifier": "CI covers zero",
   "floor": null,
   "clears_zero": false,
   "clears_floor": false
  },
  {
   "key": "cond.elo_slope",
   "arm_full": 0.019110211066039478,
   "arm_matched": 0.019055893524487923,
   "delta": -5.431754155155524e-05,
   "ci": [
    -0.012375535024828884,
    0.012475114762199998
   ],
   "n_draws": 2000,
   "label": "NOT DETECTED",
   "qualifier": "CI covers zero",
   "floor": null,
   "clears_zero": false,
   "clears_floor": false
  },
  {
   "key": "cond.own_team_r2.t1",
   "arm_full": 0.060426281733685805,
   "arm_matched": -0.025028141437328455,
   "delta": -0.08545442317101426,
   "ci": [
    -0.343776767665463,
    0.24394201866669357
   ],
   "n_draws": 2000,
   "label": "NOT DETECTED",
   "qualifier": "CI covers zero",
   "floor": null,
   "clears_zero": false,
   "clears_floor": false
  },
  {
   "key": "cond.own_team_r2.all",
   "arm_full": 0.08369569312806924,
   "arm_matched": -0.046707064433747725,
   "delta": -0.13040275756181696,
   "ci": [
    -0.4017338520398392,
    0.10342239980638177
   ],
   "n_draws": 2000,
   "label": "NOT DETECTED",
   "qualifier": "CI covers zero",
   "floor": null,
   "clears_zero": false,
   "clears_floor": false
  },
  {
   "key": "cond.opp_class_auc.t1",
   "arm_full": 0.5551605557284238,
   "arm_matched": 0.5312341431865744,
   "delta": -0.023926412541849462,
   "ci": [
    -0.21480767864213707,
    0.12890566335859444
   ],
   "n_draws": 2000,
   "label": "NOT DETECTED",
   "qualifier": "CI covers zero",
   "floor": null,
   "clears_zero": false,
   "clears_floor": false
  }
 ]
}
```
