### home72 — IN-POOL reference (72 teams, 144 games, 6785 rows, bank 5207d8c100df)

**species_logp** (hidden species: mean log presence of the true species; 3950 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -1.232 [-1.323, -1.142] | -1.843 [-1.984, -1.705] | +0.611 [+0.487, +0.737] | BETTER | +0.540 [+0.434, +0.649] | (reference) |  |
| WB | 3 | -1.296 [-1.390, -1.207] | -1.843 [-1.984, -1.705] | +0.547 [+0.423, +0.673] | BETTER | +0.486 [+0.381, +0.594] | -0.064 [-0.104, -0.027] | WORSE |

**species_hit** (hidden species: share of the true species in the top k; 3950 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | 0.634 [0.588, 0.680] | 0.429 [0.368, 0.495] | +0.205 [+0.153, +0.257] | BETTER | +0.206 [+0.158, +0.254] | (reference) |  |
| WB | 3 | 0.630 [0.583, 0.675] | 0.429 [0.368, 0.495] | +0.201 [+0.144, +0.255] | BETTER | +0.206 [+0.155, +0.256] | -0.004 [-0.031, +0.021] | NOT DETECTED |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -0.753 [-0.792, -0.715] | -0.874 [-0.933, -0.816] | +0.120 [+0.078, +0.167] | BETTER | +0.122 [+0.088, +0.159] | (reference) |  |
| WB | 3 | -0.803 [-0.846, -0.761] | -0.874 [-0.933, -0.816] | +0.070 [+0.035, +0.108] | BETTER | +0.075 [+0.047, +0.106] | -0.050 [-0.069, -0.031] | WORSE |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -4.648 [-4.893, -4.424] | -4.799 [-5.035, -4.574] | +0.151 [-0.014, +0.311] | NOT DETECTED | +0.317 [+0.201, +0.434] | (reference) |  |
| WB | 3 | -4.514 [-4.716, -4.321] | -4.799 [-5.035, -4.574] | +0.285 [+0.174, +0.398] | BETTER | +0.329 [+0.242, +0.420] | +0.134 [+0.034, +0.244] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | 0.612 [0.589, 0.635] | 0.604 [0.572, 0.636] | +0.008 [-0.017, +0.033] | NOT DETECTED | +0.023 [+0.001, +0.045] | (reference) |  |
| WB | 3 | 0.620 [0.594, 0.647] | 0.604 [0.572, 0.636] | +0.016 [-0.005, +0.038] | NOT DETECTED | +0.021 [+0.004, +0.040] | +0.008 [-0.007, +0.023] | NOT DETECTED |

**items_logp** (unstated item of a revealed mon: log probability; 7887 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -0.365 [-0.443, -0.296] | -0.461 [-0.573, -0.361] | +0.096 [+0.027, +0.168] | BETTER | +0.096 [+0.044, +0.153] | (reference) |  |
| WB | 3 | -0.360 [-0.430, -0.296] | -0.461 [-0.573, -0.361] | +0.101 [+0.037, +0.171] | BETTER | +0.097 [+0.047, +0.154] | +0.005 [-0.010, +0.023] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 7887 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | 0.888 [0.849, 0.923] | 0.848 [0.783, 0.906] | +0.040 [-0.009, +0.093] | NOT DETECTED | +0.047 [+0.003, +0.095] | (reference) |  |
| WB | 3 | 0.895 [0.860, 0.926] | 0.848 [0.783, 0.906] | +0.047 [+0.002, +0.096] | BETTER | +0.048 [+0.006, +0.094] | +0.007 [-0.005, +0.020] | NOT DETECTED |

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### heldout72_hpfix — OFF-POOL (72 teams, 144 games, 6636 rows, bank 21545e87b1fb)

**species_logp** (hidden species: mean log presence of the true species; 4316 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -2.094 [-2.340, -1.863] | -2.082 [-2.275, -1.895] | -0.013 [-0.133, +0.104] | NOT DETECTED | +0.067 [-0.008, +0.142] | (reference) |  |
| WB | 3 | -2.104 [-2.339, -1.883] | -2.082 [-2.275, -1.895] | -0.022 [-0.135, +0.088] | NOT DETECTED | +0.051 [-0.023, +0.126] | -0.010 [-0.053, +0.035] | NOT DETECTED |

**species_hit** (hidden species: share of the true species in the top k; 4316 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | 0.404 [0.346, 0.465] | 0.388 [0.327, 0.452] | +0.016 [-0.025, +0.058] | NOT DETECTED | +0.031 [-0.005, +0.069] | (reference) |  |
| WB | 3 | 0.396 [0.342, 0.454] | 0.388 [0.327, 0.452] | +0.009 [-0.036, +0.055] | NOT DETECTED | +0.028 [-0.011, +0.068] | -0.007 [-0.026, +0.011] | NOT DETECTED |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -0.920 [-0.971, -0.871] | -0.855 [-0.904, -0.807] | -0.065 [-0.091, -0.039] | WORSE | -0.017 [-0.036, +0.002] | (reference) |  |
| WB | 3 | -0.911 [-0.964, -0.858] | -0.855 [-0.904, -0.807] | -0.056 [-0.077, -0.035] | WORSE | -0.022 [-0.038, -0.005] | +0.009 [-0.010, +0.028] | NOT DETECTED |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -5.562 [-5.757, -5.360] | -4.969 [-5.156, -4.781] | -0.593 [-0.700, -0.488] | WORSE | -0.149 [-0.215, -0.081] | (reference) |  |
| WB | 3 | -5.095 [-5.273, -4.916] | -4.969 [-5.156, -4.781] | -0.127 [-0.219, -0.031] | WORSE | +0.055 [-0.011, +0.124] | +0.467 [+0.368, +0.566] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | 0.496 [0.471, 0.521] | 0.530 [0.497, 0.563] | -0.034 [-0.056, -0.012] | WORSE | -0.011 [-0.031, +0.008] | (reference) |  |
| WB | 3 | 0.509 [0.482, 0.535] | 0.530 [0.497, 0.563] | -0.021 [-0.044, +0.001] | NOT DETECTED | -0.012 [-0.031, +0.008] | +0.013 [+0.000, +0.025] | BETTER |

**items_logp** (unstated item of a revealed mon: log probability; 7198 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -0.462 [-0.557, -0.376] | -0.475 [-0.571, -0.387] | +0.014 [-0.045, +0.073] | NOT DETECTED | +0.032 [-0.010, +0.076] | (reference) |  |
| WB | 3 | -0.460 [-0.553, -0.375] | -0.475 [-0.571, -0.387] | +0.016 [-0.044, +0.077] | NOT DETECTED | +0.033 [-0.010, +0.078] | +0.002 [-0.013, +0.017] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 7198 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | 0.829 [0.776, 0.877] | 0.811 [0.746, 0.873] | +0.018 [-0.035, +0.071] | NOT DETECTED | +0.027 [-0.017, +0.072] | (reference) |  |
| WB | 3 | 0.835 [0.783, 0.883] | 0.811 [0.746, 0.873] | +0.024 [-0.025, +0.073] | NOT DETECTED | +0.030 [-0.016, +0.077] | +0.006 [-0.008, +0.020] | NOT DETECTED |

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### forum72 — OFF-POOL (72 teams, 144 games, 6475 rows, bank 0caf6037fee3)

**species_logp** (hidden species: mean log presence of the true species; 3954 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -2.356 [-2.605, -2.133] | -2.247 [-2.434, -2.069] | -0.109 [-0.226, +0.004] | NOT DETECTED | +0.005 [-0.071, +0.081] | (reference) |  |
| WB | 3 | -2.318 [-2.541, -2.116] | -2.247 [-2.434, -2.069] | -0.072 [-0.164, +0.025] | NOT DETECTED | +0.016 [-0.049, +0.085] | +0.037 [-0.017, +0.094] | NOT DETECTED |

**species_hit** (hidden species: share of the true species in the top k; 3954 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | 0.336 [0.283, 0.388] | 0.319 [0.262, 0.379] | +0.017 [-0.025, +0.057] | NOT DETECTED | +0.027 [-0.011, +0.063] | (reference) |  |
| WB | 3 | 0.314 [0.267, 0.362] | 0.319 [0.262, 0.379] | -0.005 [-0.049, +0.038] | NOT DETECTED | +0.016 [-0.020, +0.051] | -0.021 [-0.046, +0.002] | NOT DETECTED |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -0.978 [-1.061, -0.901] | -0.998 [-1.086, -0.918] | +0.021 [-0.014, +0.055] | NOT DETECTED | +0.049 [+0.022, +0.076] | (reference) |  |
| WB | 3 | -0.990 [-1.074, -0.913] | -0.998 [-1.086, -0.918] | +0.009 [-0.020, +0.038] | NOT DETECTED | +0.027 [+0.005, +0.050] | -0.012 [-0.031, +0.007] | NOT DETECTED |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -5.398 [-5.738, -5.084] | -5.114 [-5.440, -4.806] | -0.284 [-0.405, -0.165] | WORSE | +0.037 [-0.051, +0.126] | (reference) |  |
| WB | 3 | -5.030 [-5.339, -4.749] | -5.114 [-5.440, -4.806] | +0.084 [-0.015, +0.186] | NOT DETECTED | +0.168 [+0.092, +0.246] | +0.368 [+0.283, +0.461] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | 0.547 [0.517, 0.577] | 0.566 [0.529, 0.601] | -0.019 [-0.043, +0.005] | NOT DETECTED | -0.000 [-0.021, +0.021] | (reference) |  |
| WB | 3 | 0.562 [0.532, 0.591] | 0.566 [0.529, 0.601] | -0.004 [-0.023, +0.015] | NOT DETECTED | +0.002 [-0.012, +0.018] | +0.015 [+0.000, +0.029] | BETTER |

**items_logp** (unstated item of a revealed mon: log probability; 8013 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -0.624 [-0.798, -0.480] | -0.738 [-0.934, -0.560] | +0.115 [+0.030, +0.207] | BETTER | +0.119 [+0.051, +0.194] | (reference) |  |
| WB | 3 | -0.633 [-0.808, -0.487] | -0.738 [-0.934, -0.560] | +0.105 [+0.019, +0.197] | BETTER | +0.113 [+0.047, +0.186] | -0.009 [-0.025, +0.006] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 8013 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | 0.775 [0.719, 0.828] | 0.734 [0.657, 0.807] | +0.042 [-0.017, +0.101] | NOT DETECTED | +0.055 [+0.005, +0.108] | (reference) |  |
| WB | 3 | 0.779 [0.721, 0.833] | 0.734 [0.657, 0.807] | +0.045 [-0.010, +0.101] | NOT DETECTED | +0.041 [-0.009, +0.091] | +0.004 [-0.010, +0.019] | NOT DETECTED |

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 1, species_off_support 0

### p72 — OFF-POOL (72 teams, 144 games, 6643 rows, bank af668a3f65fb)

**species_logp** (hidden species: mean log presence of the true species; 3918 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -2.998 [-3.201, -2.800] | -3.030 [-3.283, -2.797] | +0.032 [-0.073, +0.144] | NOT DETECTED | +0.097 [+0.017, +0.185] | (reference) |  |
| WB | 3 | -2.942 [-3.142, -2.753] | -3.030 [-3.283, -2.797] | +0.088 [-0.015, +0.195] | NOT DETECTED | +0.140 [+0.059, +0.226] | +0.056 [+0.001, +0.110] | BETTER |

**species_hit** (hidden species: share of the true species in the top k; 3918 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | 0.166 [0.136, 0.200] | 0.180 [0.143, 0.221] | -0.014 [-0.042, +0.013] | NOT DETECTED | -0.010 [-0.036, +0.014] | (reference) |  |
| WB | 3 | 0.166 [0.138, 0.196] | 0.180 [0.143, 0.221] | -0.014 [-0.042, +0.012] | NOT DETECTED | -0.008 [-0.032, +0.014] | -0.001 [-0.017, +0.014] | NOT DETECTED |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -1.076 [-1.138, -1.014] | -1.015 [-1.076, -0.951] | -0.062 [-0.085, -0.038] | WORSE | -0.014 [-0.031, +0.003] | (reference) |  |
| WB | 3 | -1.059 [-1.120, -0.994] | -1.015 [-1.076, -0.951] | -0.044 [-0.063, -0.025] | WORSE | -0.012 [-0.026, +0.002] | +0.018 [-0.006, +0.041] | NOT DETECTED |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -6.307 [-6.568, -6.044] | -5.597 [-5.828, -5.369] | -0.709 [-0.836, -0.584] | WORSE | -0.189 [-0.265, -0.111] | (reference) |  |
| WB | 3 | -5.769 [-6.014, -5.527] | -5.597 [-5.828, -5.369] | -0.171 [-0.268, -0.080] | WORSE | +0.054 [-0.005, +0.116] | +0.538 [+0.427, +0.654] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | 0.447 [0.417, 0.476] | 0.486 [0.450, 0.521] | -0.039 [-0.060, -0.019] | WORSE | -0.023 [-0.040, -0.006] | (reference) |  |
| WB | 3 | 0.466 [0.437, 0.494] | 0.486 [0.450, 0.521] | -0.020 [-0.038, -0.002] | WORSE | -0.014 [-0.030, +0.001] | +0.019 [+0.008, +0.031] | BETTER |

**items_logp** (unstated item of a revealed mon: log probability; 9437 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | -0.536 [-0.647, -0.433] | -0.634 [-0.758, -0.517] | +0.099 [+0.047, +0.153] | BETTER | +0.092 [+0.051, +0.138] | (reference) |  |
| WB | 3 | -0.530 [-0.641, -0.428] | -0.634 [-0.758, -0.517] | +0.105 [+0.055, +0.158] | BETTER | +0.096 [+0.055, +0.142] | +0.006 [-0.008, +0.020] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 9437 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − Br |  |
|---|---|---|---|---|---|---|---|---|
| Br | 3 | 0.788 [0.734, 0.838] | 0.735 [0.670, 0.797] | +0.053 [+0.007, +0.099] | BETTER | +0.060 [+0.020, +0.100] | (reference) |  |
| WB | 3 | 0.795 [0.743, 0.845] | 0.735 [0.670, 0.797] | +0.060 [+0.021, +0.100] | BETTER | +0.058 [+0.025, +0.093] | +0.008 [-0.007, +0.022] | NOT DETECTED |

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0
