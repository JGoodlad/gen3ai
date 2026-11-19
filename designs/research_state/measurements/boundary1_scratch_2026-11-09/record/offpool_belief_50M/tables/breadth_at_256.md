### home72 — IN-POOL reference (72 teams, 144 games, 6785 rows, bank 5207d8c100df)

**species_logp** (hidden species: mean log presence of the true species; 3950 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -0.856 [-0.934, -0.780] | -1.843 [-1.984, -1.705] | +0.987 [+0.853, +1.121] | BETTER | +0.882 [+0.764, +1.002] | (reference) |  |
| WB | 3 | -1.296 [-1.390, -1.207] | -1.843 [-1.984, -1.705] | +0.547 [+0.423, +0.673] | BETTER | +0.486 [+0.381, +0.594] | -0.439 [-0.508, -0.373] | WORSE |

**species_hit** (hidden species: share of the true species in the top k; 3950 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | 0.692 [0.657, 0.726] | 0.429 [0.368, 0.495] | +0.263 [+0.202, +0.321] | BETTER | +0.266 [+0.208, +0.322] | (reference) |  |
| WB | 3 | 0.630 [0.583, 0.675] | 0.429 [0.368, 0.495] | +0.201 [+0.144, +0.255] | BETTER | +0.206 [+0.155, +0.256] | -0.062 [-0.094, -0.032] | WORSE |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -0.682 [-0.723, -0.643] | -0.874 [-0.933, -0.816] | +0.191 [+0.141, +0.245] | BETTER | +0.179 [+0.138, +0.223] | (reference) |  |
| WB | 3 | -0.803 [-0.846, -0.761] | -0.874 [-0.933, -0.816] | +0.070 [+0.035, +0.108] | BETTER | +0.075 [+0.047, +0.106] | -0.121 [-0.144, -0.099] | WORSE |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -4.565 [-4.796, -4.352] | -4.799 [-5.035, -4.574] | +0.234 [+0.038, +0.437] | BETTER | +0.355 [+0.210, +0.507] | (reference) |  |
| WB | 3 | -4.514 [-4.716, -4.321] | -4.799 [-5.035, -4.574] | +0.285 [+0.174, +0.398] | BETTER | +0.329 [+0.242, +0.420] | +0.051 [-0.073, +0.166] | NOT DETECTED |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | 0.658 [0.634, 0.683] | 0.604 [0.572, 0.636] | +0.054 [+0.025, +0.084] | BETTER | +0.052 [+0.028, +0.076] | (reference) |  |
| WB | 3 | 0.620 [0.594, 0.647] | 0.604 [0.572, 0.636] | +0.016 [-0.005, +0.038] | NOT DETECTED | +0.021 [+0.004, +0.040] | -0.038 [-0.056, -0.021] | WORSE |

**items_logp** (unstated item of a revealed mon: log probability; 7887 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -0.314 [-0.399, -0.241] | -0.461 [-0.573, -0.361] | +0.147 [+0.075, +0.225] | BETTER | +0.140 [+0.084, +0.204] | (reference) |  |
| WB | 3 | -0.360 [-0.430, -0.296] | -0.461 [-0.573, -0.361] | +0.101 [+0.037, +0.171] | BETTER | +0.097 [+0.047, +0.154] | -0.046 [-0.086, -0.005] | WORSE |

**items_top1** (unstated item of a revealed mon: top-1 rate; 7887 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | 0.895 [0.857, 0.929] | 0.848 [0.783, 0.906] | +0.047 [+0.006, +0.094] | BETTER | +0.053 [+0.014, +0.096] | (reference) |  |
| WB | 3 | 0.895 [0.860, 0.926] | 0.848 [0.783, 0.906] | +0.047 [+0.002, +0.096] | BETTER | +0.048 [+0.006, +0.094] | -0.001 [-0.023, +0.024] | NOT DETECTED |

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### heldout72_hpfix — OFF-POOL (72 teams, 144 games, 6636 rows, bank 21545e87b1fb)

**species_logp** (hidden species: mean log presence of the true species; 4316 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -3.130 [-3.746, -2.556] | -2.082 [-2.275, -1.895] | -1.048 [-1.530, -0.607] | WORSE | -0.057 [-0.172, +0.061] | (reference) |  |
| WB | 3 | -2.104 [-2.339, -1.883] | -2.082 [-2.275, -1.895] | -0.022 [-0.135, +0.088] | NOT DETECTED | +0.051 [-0.023, +0.126] | +1.026 [+0.639, +1.439] | BETTER |

**species_hit** (hidden species: share of the true species in the top k; 4316 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | 0.369 [0.315, 0.427] | 0.388 [0.327, 0.452] | -0.018 [-0.065, +0.027] | NOT DETECTED | -0.006 [-0.050, +0.037] | (reference) |  |
| WB | 3 | 0.396 [0.342, 0.454] | 0.388 [0.327, 0.452] | +0.009 [-0.036, +0.055] | NOT DETECTED | +0.028 [-0.011, +0.068] | +0.027 [+0.003, +0.054] | BETTER |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -0.909 [-0.965, -0.851] | -0.855 [-0.904, -0.807] | -0.054 [-0.083, -0.023] | WORSE | -0.007 [-0.029, +0.016] | (reference) |  |
| WB | 3 | -0.911 [-0.964, -0.858] | -0.855 [-0.904, -0.807] | -0.056 [-0.077, -0.035] | WORSE | -0.022 [-0.038, -0.005] | -0.002 [-0.021, +0.016] | NOT DETECTED |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -5.737 [-5.969, -5.506] | -4.969 [-5.156, -4.781] | -0.768 [-0.889, -0.647] | WORSE | -0.344 [-0.422, -0.264] | (reference) |  |
| WB | 3 | -5.095 [-5.273, -4.916] | -4.969 [-5.156, -4.781] | -0.127 [-0.219, -0.031] | WORSE | +0.055 [-0.011, +0.124] | +0.642 [+0.526, +0.762] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | 0.508 [0.479, 0.538] | 0.530 [0.497, 0.563] | -0.022 [-0.045, +0.002] | NOT DETECTED | -0.011 [-0.032, +0.011] | (reference) |  |
| WB | 3 | 0.509 [0.482, 0.535] | 0.530 [0.497, 0.563] | -0.021 [-0.044, +0.001] | NOT DETECTED | -0.012 [-0.031, +0.008] | +0.000 [-0.013, +0.014] | NOT DETECTED |

**items_logp** (unstated item of a revealed mon: log probability; 7198 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -0.525 [-0.659, -0.406] | -0.475 [-0.571, -0.387] | -0.050 [-0.140, +0.031] | NOT DETECTED | +0.012 [-0.041, +0.063] | (reference) |  |
| WB | 3 | -0.460 [-0.553, -0.375] | -0.475 [-0.571, -0.387] | +0.016 [-0.044, +0.077] | NOT DETECTED | +0.033 [-0.010, +0.078] | +0.066 [-0.004, +0.146] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 7198 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | 0.815 [0.764, 0.863] | 0.811 [0.746, 0.873] | +0.004 [-0.045, +0.052] | NOT DETECTED | +0.018 [-0.028, +0.063] | (reference) |  |
| WB | 3 | 0.835 [0.783, 0.883] | 0.811 [0.746, 0.873] | +0.024 [-0.025, +0.073] | NOT DETECTED | +0.030 [-0.016, +0.077] | +0.020 [-0.015, +0.054] | NOT DETECTED |

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### forum72 — OFF-POOL (72 teams, 144 games, 6475 rows, bank 0caf6037fee3)

**species_logp** (hidden species: mean log presence of the true species; 3954 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -3.515 [-4.129, -2.977] | -2.247 [-2.434, -2.069] | -1.268 [-1.740, -0.858] | WORSE | -0.111 [-0.223, -0.001] | (reference) |  |
| WB | 3 | -2.318 [-2.541, -2.116] | -2.247 [-2.434, -2.069] | -0.072 [-0.164, +0.025] | NOT DETECTED | +0.016 [-0.049, +0.085] | +1.197 [+0.828, +1.624] | BETTER |

**species_hit** (hidden species: share of the true species in the top k; 3954 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | 0.322 [0.273, 0.370] | 0.319 [0.262, 0.379] | +0.003 [-0.040, +0.044] | NOT DETECTED | +0.009 [-0.033, +0.049] | (reference) |  |
| WB | 3 | 0.314 [0.267, 0.362] | 0.319 [0.262, 0.379] | -0.005 [-0.049, +0.038] | NOT DETECTED | +0.016 [-0.020, +0.051] | -0.008 [-0.033, +0.017] | NOT DETECTED |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -0.900 [-0.983, -0.823] | -0.998 [-1.086, -0.918] | +0.099 [+0.059, +0.140] | BETTER | +0.110 [+0.079, +0.143] | (reference) |  |
| WB | 3 | -0.990 [-1.074, -0.913] | -0.998 [-1.086, -0.918] | +0.009 [-0.020, +0.038] | NOT DETECTED | +0.027 [+0.005, +0.050] | -0.090 [-0.116, -0.065] | WORSE |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -5.320 [-5.668, -5.001] | -5.114 [-5.440, -4.806] | -0.206 [-0.338, -0.066] | WORSE | +0.045 [-0.050, +0.147] | (reference) |  |
| WB | 3 | -5.030 [-5.339, -4.749] | -5.114 [-5.440, -4.806] | +0.084 [-0.015, +0.186] | NOT DETECTED | +0.168 [+0.092, +0.246] | +0.290 [+0.196, +0.382] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | 0.577 [0.548, 0.605] | 0.566 [0.529, 0.601] | +0.011 [-0.015, +0.037] | NOT DETECTED | +0.020 [-0.003, +0.043] | (reference) |  |
| WB | 3 | 0.562 [0.532, 0.591] | 0.566 [0.529, 0.601] | -0.004 [-0.023, +0.015] | NOT DETECTED | +0.002 [-0.012, +0.018] | -0.015 [-0.033, +0.002] | NOT DETECTED |

**items_logp** (unstated item of a revealed mon: log probability; 8013 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -0.664 [-0.901, -0.468] | -0.738 [-0.934, -0.560] | +0.075 [-0.053, +0.199] | NOT DETECTED | +0.135 [+0.059, +0.219] | (reference) |  |
| WB | 3 | -0.633 [-0.808, -0.487] | -0.738 [-0.934, -0.560] | +0.105 [+0.019, +0.197] | BETTER | +0.113 [+0.047, +0.186] | +0.031 [-0.042, +0.120] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 8013 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | 0.779 [0.727, 0.829] | 0.734 [0.657, 0.807] | +0.046 [-0.015, +0.106] | NOT DETECTED | +0.052 [-0.003, +0.107] | (reference) |  |
| WB | 3 | 0.779 [0.721, 0.833] | 0.734 [0.657, 0.807] | +0.045 [-0.010, +0.101] | NOT DETECTED | +0.041 [-0.009, +0.091] | -0.000 [-0.023, +0.024] | NOT DETECTED |

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### p72 — OFF-POOL (72 teams, 144 games, 6643 rows, bank af668a3f65fb)

**species_logp** (hidden species: mean log presence of the true species; 3918 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -6.275 [-6.973, -5.614] | -3.030 [-3.283, -2.797] | -3.245 [-3.776, -2.731] | WORSE | -0.641 [-0.726, -0.549] | (reference) |  |
| WB | 3 | -2.942 [-3.142, -2.753] | -3.030 [-3.283, -2.797] | +0.088 [-0.015, +0.195] | NOT DETECTED | +0.140 [+0.059, +0.226] | +3.333 [+2.805, +3.892] | BETTER |

**species_hit** (hidden species: share of the true species in the top k; 3918 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | 0.128 [0.106, 0.153] | 0.180 [0.143, 0.221] | -0.051 [-0.087, -0.020] | WORSE | -0.046 [-0.080, -0.015] | (reference) |  |
| WB | 3 | 0.166 [0.138, 0.196] | 0.180 [0.143, 0.221] | -0.014 [-0.042, +0.012] | NOT DETECTED | -0.008 [-0.032, +0.014] | +0.037 [+0.018, +0.059] | BETTER |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -1.081 [-1.148, -1.015] | -1.015 [-1.076, -0.951] | -0.066 [-0.093, -0.039] | WORSE | -0.013 [-0.033, +0.006] | (reference) |  |
| WB | 3 | -1.059 [-1.120, -0.994] | -1.015 [-1.076, -0.951] | -0.044 [-0.063, -0.025] | WORSE | -0.012 [-0.026, +0.002] | +0.023 [-0.002, +0.048] | NOT DETECTED |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -6.503 [-6.791, -6.222] | -5.597 [-5.828, -5.369] | -0.906 [-1.011, -0.803] | WORSE | -0.389 [-0.450, -0.329] | (reference) |  |
| WB | 3 | -5.769 [-6.014, -5.527] | -5.597 [-5.828, -5.369] | -0.171 [-0.268, -0.080] | WORSE | +0.054 [-0.005, +0.116] | +0.734 [+0.612, +0.861] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | 0.459 [0.429, 0.489] | 0.486 [0.450, 0.521] | -0.027 [-0.046, -0.008] | WORSE | -0.017 [-0.032, -0.002] | (reference) |  |
| WB | 3 | 0.466 [0.437, 0.494] | 0.486 [0.450, 0.521] | -0.020 [-0.038, -0.002] | WORSE | -0.014 [-0.030, +0.001] | +0.007 [-0.007, +0.021] | NOT DETECTED |

**items_logp** (unstated item of a revealed mon: log probability; 9437 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | -0.827 [-1.083, -0.606] | -0.634 [-0.758, -0.517] | -0.193 [-0.373, -0.040] | WORSE | -0.008 [-0.080, +0.061] | (reference) |  |
| WB | 3 | -0.530 [-0.641, -0.428] | -0.634 [-0.758, -0.517] | +0.105 [+0.055, +0.158] | BETTER | +0.096 [+0.055, +0.142] | +0.297 [+0.154, +0.473] | BETTER |

**items_top1** (unstated item of a revealed mon: top-1 rate; 9437 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − W |  |
|---|---|---|---|---|---|---|---|---|
| W | 3 | 0.763 [0.711, 0.813] | 0.735 [0.670, 0.797] | +0.028 [-0.013, +0.069] | NOT DETECTED | +0.027 [-0.012, +0.066] | (reference) |  |
| WB | 3 | 0.795 [0.743, 0.845] | 0.735 [0.670, 0.797] | +0.060 [+0.021, +0.100] | BETTER | +0.058 [+0.025, +0.093] | +0.032 [+0.003, +0.063] | BETTER |

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0
