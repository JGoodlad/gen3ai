### home72 — IN-POOL reference (72 teams, 144 games, 6785 rows, bank 5207d8c100df)

**species_logp** (hidden species: mean log presence of the true species; 3950 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -1.048 [-1.132, -0.966] | -1.843 [-1.984, -1.705] | +0.795 [+0.674, +0.918] | BETTER | +0.709 [+0.604, +0.818] | (reference) |  |
| ON | 6 | -1.076 [-1.154, -1.000] | -1.843 [-1.984, -1.705] | +0.767 [+0.642, +0.892] | BETTER | +0.684 [+0.576, +0.793] | -0.028 [-0.052, -0.005] | WORSE |

**species_hit** (hidden species: share of the true species in the top k; 3950 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.660 [0.621, 0.699] | 0.429 [0.368, 0.495] | +0.231 [+0.178, +0.283] | BETTER | +0.233 [+0.183, +0.281] | (reference) |  |
| ON | 6 | 0.661 [0.623, 0.698] | 0.429 [0.368, 0.495] | +0.232 [+0.175, +0.286] | BETTER | +0.236 [+0.184, +0.286] | +0.000 [-0.013, +0.014] | NOT DETECTED |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -0.714 [-0.754, -0.674] | -0.874 [-0.933, -0.816] | +0.160 [+0.110, +0.212] | BETTER | +0.160 [+0.120, +0.202] | (reference) |  |
| ON | 6 | -0.743 [-0.783, -0.704] | -0.874 [-0.933, -0.816] | +0.131 [+0.089, +0.176] | BETTER | +0.127 [+0.093, +0.164] | -0.029 [-0.045, -0.013] | WORSE |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -4.897 [-5.161, -4.655] | -4.799 [-5.035, -4.574] | -0.098 [-0.307, +0.105] | NOT DETECTED | +0.214 [+0.071, +0.356] | (reference) |  |
| ON | 6 | -4.539 [-4.745, -4.346] | -4.799 [-5.035, -4.574] | +0.260 [+0.111, +0.413] | BETTER | +0.342 [+0.228, +0.461] | +0.358 [+0.259, +0.467] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.623 [0.599, 0.646] | 0.604 [0.572, 0.636] | +0.019 [-0.007, +0.044] | NOT DETECTED | +0.034 [+0.012, +0.057] | (reference) |  |
| ON | 6 | 0.639 [0.615, 0.663] | 0.604 [0.572, 0.636] | +0.035 [+0.011, +0.059] | BETTER | +0.037 [+0.017, +0.057] | +0.016 [+0.006, +0.028] | BETTER |

**items_logp** (unstated item of a revealed mon: log probability; 7887 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -0.336 [-0.411, -0.269] | -0.461 [-0.573, -0.361] | +0.125 [+0.060, +0.196] | BETTER | +0.121 [+0.068, +0.181] | (reference) |  |
| ON | 6 | -0.337 [-0.411, -0.271] | -0.461 [-0.573, -0.361] | +0.124 [+0.060, +0.195] | BETTER | +0.119 [+0.067, +0.178] | -0.001 [-0.010, +0.009] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 7887 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.890 [0.855, 0.922] | 0.848 [0.783, 0.906] | +0.042 [-0.001, +0.089] | NOT DETECTED | +0.049 [+0.010, +0.092] | (reference) |  |
| ON | 6 | 0.895 [0.861, 0.926] | 0.848 [0.783, 0.906] | +0.047 [+0.006, +0.094] | BETTER | +0.050 [+0.012, +0.094] | +0.005 [-0.004, +0.014] | NOT DETECTED |

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### heldout72_hpfix — OFF-POOL (72 teams, 144 games, 6636 rows, bank 21545e87b1fb)

**species_logp** (hidden species: mean log presence of the true species; 4316 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -2.706 [-3.162, -2.281] | -2.082 [-2.275, -1.895] | -0.624 [-0.938, -0.337] | WORSE | +0.002 [-0.087, +0.092] | (reference) |  |
| ON | 6 | -2.617 [-3.031, -2.226] | -2.082 [-2.275, -1.895] | -0.535 [-0.818, -0.279] | WORSE | -0.003 [-0.087, +0.082] | +0.089 [+0.015, +0.166] | BETTER |

**species_hit** (hidden species: share of the true species in the top k; 4316 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.387 [0.332, 0.446] | 0.388 [0.327, 0.452] | -0.000 [-0.043, +0.043] | NOT DETECTED | +0.011 [-0.028, +0.052] | (reference) |  |
| ON | 6 | 0.383 [0.330, 0.439] | 0.388 [0.327, 0.452] | -0.005 [-0.049, +0.039] | NOT DETECTED | +0.011 [-0.028, +0.050] | -0.004 [-0.018, +0.009] | NOT DETECTED |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -0.936 [-0.992, -0.880] | -0.855 [-0.904, -0.807] | -0.081 [-0.108, -0.052] | WORSE | -0.017 [-0.036, +0.003] | (reference) |  |
| ON | 6 | -0.910 [-0.964, -0.855] | -0.855 [-0.904, -0.807] | -0.055 [-0.079, -0.030] | WORSE | -0.014 [-0.032, +0.005] | +0.026 [+0.010, +0.042] | BETTER |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -6.109 [-6.331, -5.884] | -4.969 [-5.156, -4.781] | -1.140 [-1.267, -1.017] | WORSE | -0.435 [-0.507, -0.361] | (reference) |  |
| ON | 6 | -5.416 [-5.614, -5.216] | -4.969 [-5.156, -4.781] | -0.448 [-0.536, -0.356] | WORSE | -0.144 [-0.209, -0.077] | +0.693 [+0.610, +0.782] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.489 [0.464, 0.513] | 0.530 [0.497, 0.563] | -0.042 [-0.063, -0.021] | WORSE | -0.019 [-0.039, -0.000] | (reference) |  |
| ON | 6 | 0.509 [0.481, 0.535] | 0.530 [0.497, 0.563] | -0.022 [-0.044, +0.000] | NOT DETECTED | -0.011 [-0.031, +0.008] | +0.020 [+0.010, +0.030] | BETTER |

**items_logp** (unstated item of a revealed mon: log probability; 7198 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -0.495 [-0.603, -0.397] | -0.475 [-0.571, -0.387] | -0.020 [-0.085, +0.040] | NOT DETECTED | +0.020 [-0.023, +0.063] | (reference) |  |
| ON | 6 | -0.492 [-0.600, -0.395] | -0.475 [-0.571, -0.387] | -0.017 [-0.082, +0.044] | NOT DETECTED | +0.023 [-0.021, +0.065] | +0.003 [-0.012, +0.018] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 7198 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.819 [0.769, 0.864] | 0.811 [0.746, 0.873] | +0.008 [-0.040, +0.055] | NOT DETECTED | +0.018 [-0.023, +0.059] | (reference) |  |
| ON | 6 | 0.825 [0.777, 0.870] | 0.811 [0.746, 0.873] | +0.014 [-0.032, +0.059] | NOT DETECTED | +0.024 [-0.018, +0.066] | +0.006 [-0.003, +0.016] | NOT DETECTED |

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### forum72 — OFF-POOL (72 teams, 144 games, 6475 rows, bank 0caf6037fee3)

**species_logp** (hidden species: mean log presence of the true species; 3954 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -3.079 [-3.550, -2.665] | -2.247 [-2.434, -2.069] | -0.832 [-1.155, -0.549] | WORSE | -0.066 [-0.155, +0.024] | (reference) |  |
| ON | 6 | -2.917 [-3.325, -2.555] | -2.247 [-2.434, -2.069] | -0.670 [-0.933, -0.439] | WORSE | -0.047 [-0.125, +0.031] | +0.162 [+0.057, +0.282] | BETTER |

**species_hit** (hidden species: share of the true species in the top k; 3954 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.324 [0.275, 0.373] | 0.319 [0.262, 0.379] | +0.005 [-0.037, +0.046] | NOT DETECTED | +0.014 [-0.026, +0.052] | (reference) |  |
| ON | 6 | 0.318 [0.272, 0.364] | 0.319 [0.262, 0.379] | -0.001 [-0.043, +0.040] | NOT DETECTED | +0.012 [-0.025, +0.048] | -0.006 [-0.023, +0.011] | NOT DETECTED |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -0.949 [-1.033, -0.873] | -0.998 [-1.086, -0.918] | +0.049 [+0.011, +0.087] | BETTER | +0.077 [+0.048, +0.107] | (reference) |  |
| ON | 6 | -0.945 [-1.028, -0.869] | -0.998 [-1.086, -0.918] | +0.054 [+0.021, +0.086] | BETTER | +0.069 [+0.043, +0.095] | +0.005 [-0.011, +0.021] | NOT DETECTED |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -5.717 [-6.070, -5.389] | -5.114 [-5.440, -4.806] | -0.604 [-0.741, -0.466] | WORSE | -0.127 [-0.221, -0.029] | (reference) |  |
| ON | 6 | -5.175 [-5.500, -4.880] | -5.114 [-5.440, -4.806] | -0.061 [-0.168, +0.053] | NOT DETECTED | +0.106 [+0.026, +0.192] | +0.542 [+0.464, +0.624] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.550 [0.521, 0.578] | 0.566 [0.529, 0.601] | -0.016 [-0.040, +0.008] | NOT DETECTED | +0.002 [-0.018, +0.023] | (reference) |  |
| ON | 6 | 0.569 [0.542, 0.597] | 0.566 [0.529, 0.601] | +0.003 [-0.017, +0.025] | NOT DETECTED | +0.011 [-0.007, +0.030] | +0.020 [+0.010, +0.029] | BETTER |

**items_logp** (unstated item of a revealed mon: log probability; 8013 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -0.641 [-0.854, -0.470] | -0.738 [-0.934, -0.560] | +0.098 [-0.002, +0.200] | NOT DETECTED | +0.130 [+0.060, +0.209] | (reference) |  |
| ON | 6 | -0.648 [-0.856, -0.480] | -0.738 [-0.934, -0.560] | +0.090 [-0.008, +0.192] | NOT DETECTED | +0.124 [+0.054, +0.201] | -0.008 [-0.025, +0.011] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 8013 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.777 [0.725, 0.827] | 0.734 [0.657, 0.807] | +0.043 [-0.016, +0.102] | NOT DETECTED | +0.050 [-0.002, +0.103] | (reference) |  |
| ON | 6 | 0.779 [0.725, 0.830] | 0.734 [0.657, 0.807] | +0.045 [-0.011, +0.103] | NOT DETECTED | +0.046 [-0.006, +0.097] | +0.002 [-0.008, +0.015] | NOT DETECTED |

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### p72 — OFF-POOL (72 teams, 144 games, 6643 rows, bank af668a3f65fb)

**species_logp** (hidden species: mean log presence of the true species; 3918 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -4.905 [-5.376, -4.450] | -3.030 [-3.283, -2.797] | -1.875 [-2.173, -1.583] | WORSE | -0.290 [-0.349, -0.229] | (reference) |  |
| ON | 6 | -4.609 [-5.034, -4.198] | -3.030 [-3.283, -2.797] | -1.579 [-1.847, -1.319] | WORSE | -0.251 [-0.310, -0.189] | +0.296 [+0.198, +0.398] | BETTER |

**species_hit** (hidden species: share of the true species in the top k; 3918 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.149 [0.123, 0.177] | 0.180 [0.143, 0.221] | -0.031 [-0.062, -0.004] | WORSE | -0.028 [-0.056, -0.003] | (reference) |  |
| ON | 6 | 0.147 [0.123, 0.172] | 0.180 [0.143, 0.221] | -0.033 [-0.063, -0.006] | WORSE | -0.027 [-0.054, -0.002] | -0.002 [-0.011, +0.006] | NOT DETECTED |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -1.085 [-1.148, -1.021] | -1.015 [-1.076, -0.951] | -0.071 [-0.096, -0.045] | WORSE | -0.010 [-0.029, +0.008] | (reference) |  |
| ON | 6 | -1.070 [-1.132, -1.005] | -1.015 [-1.076, -0.951] | -0.055 [-0.075, -0.036] | WORSE | -0.013 [-0.027, +0.002] | +0.015 [-0.002, +0.033] | NOT DETECTED |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -6.886 [-7.169, -6.604] | -5.597 [-5.828, -5.369] | -1.289 [-1.413, -1.164] | WORSE | -0.494 [-0.567, -0.420] | (reference) |  |
| ON | 6 | -6.136 [-6.394, -5.880] | -5.597 [-5.828, -5.369] | -0.539 [-0.616, -0.463] | WORSE | -0.168 [-0.218, -0.118] | +0.751 [+0.659, +0.841] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.443 [0.414, 0.470] | 0.486 [0.450, 0.521] | -0.044 [-0.063, -0.025] | WORSE | -0.025 [-0.041, -0.009] | (reference) |  |
| ON | 6 | 0.463 [0.434, 0.490] | 0.486 [0.450, 0.521] | -0.024 [-0.041, -0.006] | WORSE | -0.016 [-0.030, -0.001] | +0.020 [+0.011, +0.030] | BETTER |

**items_logp** (unstated item of a revealed mon: log probability; 9437 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -0.687 [-0.867, -0.528] | -0.634 [-0.758, -0.517] | -0.053 [-0.147, +0.031] | NOT DETECTED | +0.035 [-0.010, +0.083] | (reference) |  |
| ON | 6 | -0.678 [-0.856, -0.522] | -0.634 [-0.758, -0.517] | -0.044 [-0.142, +0.045] | NOT DETECTED | +0.044 [-0.005, +0.095] | +0.009 [-0.015, +0.035] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 9437 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.773 [0.723, 0.820] | 0.735 [0.670, 0.797] | +0.038 [-0.003, +0.080] | NOT DETECTED | +0.041 [+0.006, +0.079] | (reference) |  |
| ON | 6 | 0.779 [0.729, 0.826] | 0.735 [0.670, 0.797] | +0.044 [+0.007, +0.082] | BETTER | +0.043 [+0.010, +0.076] | +0.006 [-0.002, +0.015] | NOT DETECTED |

Units not scored / flagged (first arm): moves_prec_near_tie 1, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0
