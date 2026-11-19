### home72 — IN-POOL reference (72 teams, 144 games, 6785 rows, bank 5207d8c100df)

**species_logp** (hidden species: mean log presence of the true species; 3950 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -0.860 [-0.939, -0.782] | -1.843 [-1.984, -1.705] | +0.983 [+0.855, +1.113] | BETTER | +0.880 [+0.767, +0.997] | (reference) |  |
| ON | 6 | -1.264 [-1.354, -1.175] | -1.843 [-1.984, -1.705] | +0.579 [+0.456, +0.704] | BETTER | +0.513 [+0.408, +0.621] | -0.404 [-0.462, -0.346] | WORSE |

**species_hit** (hidden species: share of the true species in the top k; 3950 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.689 [0.654, 0.724] | 0.429 [0.368, 0.495] | +0.260 [+0.202, +0.317] | BETTER | +0.263 [+0.208, +0.317] | (reference) |  |
| ON | 6 | 0.632 [0.588, 0.676] | 0.429 [0.368, 0.495] | +0.203 [+0.149, +0.254] | BETTER | +0.206 [+0.158, +0.253] | -0.057 [-0.082, -0.034] | WORSE |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -0.678 [-0.720, -0.637] | -0.874 [-0.933, -0.816] | +0.195 [+0.142, +0.253] | BETTER | +0.188 [+0.146, +0.234] | (reference) |  |
| ON | 6 | -0.778 [-0.818, -0.739] | -0.874 [-0.933, -0.816] | +0.095 [+0.057, +0.136] | BETTER | +0.099 [+0.068, +0.132] | -0.100 [-0.122, -0.077] | WORSE |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -4.856 [-5.112, -4.620] | -4.799 [-5.035, -4.574] | -0.056 [-0.275, +0.166] | NOT DETECTED | +0.233 [+0.076, +0.392] | (reference) |  |
| ON | 6 | -4.581 [-4.798, -4.380] | -4.799 [-5.035, -4.574] | +0.218 [+0.087, +0.348] | BETTER | +0.323 [+0.225, +0.424] | +0.275 [+0.166, +0.383] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.646 [0.621, 0.670] | 0.604 [0.572, 0.636] | +0.042 [+0.014, +0.070] | BETTER | +0.049 [+0.025, +0.073] | (reference) |  |
| ON | 6 | 0.616 [0.593, 0.640] | 0.604 [0.572, 0.636] | +0.012 [-0.010, +0.034] | NOT DETECTED | +0.022 [+0.004, +0.041] | -0.030 [-0.042, -0.017] | WORSE |

**items_logp** (unstated item of a revealed mon: log probability; 7887 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -0.311 [-0.394, -0.239] | -0.461 [-0.573, -0.361] | +0.150 [+0.078, +0.230] | BETTER | +0.143 [+0.086, +0.208] | (reference) |  |
| ON | 6 | -0.363 [-0.436, -0.297] | -0.461 [-0.573, -0.361] | +0.099 [+0.033, +0.169] | BETTER | +0.096 [+0.046, +0.154] | -0.052 [-0.098, -0.008] | WORSE |

**items_top1** (unstated item of a revealed mon: top-1 rate; 7887 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.894 [0.856, 0.927] | 0.848 [0.783, 0.906] | +0.046 [+0.005, +0.091] | BETTER | +0.051 [+0.014, +0.094] | (reference) |  |
| ON | 6 | 0.891 [0.855, 0.924] | 0.848 [0.783, 0.906] | +0.043 [-0.003, +0.094] | NOT DETECTED | +0.048 [+0.005, +0.094] | -0.003 [-0.028, +0.023] | NOT DETECTED |

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### heldout72_hpfix — OFF-POOL (72 teams, 144 games, 6636 rows, bank 21545e87b1fb)

**species_logp** (hidden species: mean log presence of the true species; 4316 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -3.224 [-3.868, -2.626] | -2.082 [-2.275, -1.895] | -1.142 [-1.648, -0.680] | WORSE | -0.060 [-0.175, +0.060] | (reference) |  |
| ON | 6 | -2.099 [-2.337, -1.873] | -2.082 [-2.275, -1.895] | -0.017 [-0.133, +0.093] | NOT DETECTED | +0.059 [-0.014, +0.132] | +1.125 [+0.721, +1.563] | BETTER |

**species_hit** (hidden species: share of the true species in the top k; 4316 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.370 [0.315, 0.428] | 0.388 [0.327, 0.452] | -0.018 [-0.063, +0.028] | NOT DETECTED | -0.007 [-0.051, +0.036] | (reference) |  |
| ON | 6 | 0.400 [0.345, 0.458] | 0.388 [0.327, 0.452] | +0.012 [-0.029, +0.056] | NOT DETECTED | +0.030 [-0.007, +0.068] | +0.030 [+0.009, +0.052] | BETTER |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -0.930 [-0.989, -0.870] | -0.855 [-0.904, -0.807] | -0.075 [-0.105, -0.044] | WORSE | -0.012 [-0.034, +0.011] | (reference) |  |
| ON | 6 | -0.916 [-0.967, -0.865] | -0.855 [-0.904, -0.807] | -0.061 [-0.082, -0.039] | WORSE | -0.019 [-0.035, -0.002] | +0.014 [-0.005, +0.033] | NOT DETECTED |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -6.197 [-6.444, -5.951] | -4.969 [-5.156, -4.781] | -1.228 [-1.369, -1.088] | WORSE | -0.532 [-0.616, -0.448] | (reference) |  |
| ON | 6 | -5.329 [-5.507, -5.144] | -4.969 [-5.156, -4.781] | -0.360 [-0.445, -0.274] | WORSE | -0.047 [-0.108, +0.016] | +0.868 [+0.748, +0.990] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.495 [0.468, 0.521] | 0.530 [0.497, 0.563] | -0.035 [-0.057, -0.014] | WORSE | -0.019 [-0.039, +0.001] | (reference) |  |
| ON | 6 | 0.502 [0.478, 0.527] | 0.530 [0.497, 0.563] | -0.028 [-0.049, -0.006] | WORSE | -0.011 [-0.030, +0.007] | +0.008 [-0.002, +0.018] | NOT DETECTED |

**items_logp** (unstated item of a revealed mon: log probability; 7198 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -0.527 [-0.656, -0.411] | -0.475 [-0.571, -0.387] | -0.052 [-0.137, +0.025] | NOT DETECTED | +0.010 [-0.041, +0.059] | (reference) |  |
| ON | 6 | -0.461 [-0.555, -0.375] | -0.475 [-0.571, -0.387] | +0.015 [-0.044, +0.074] | NOT DETECTED | +0.032 [-0.010, +0.077] | +0.067 [+0.005, +0.137] | BETTER |

**items_top1** (unstated item of a revealed mon: top-1 rate; 7198 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.812 [0.760, 0.860] | 0.811 [0.746, 0.873] | +0.001 [-0.048, +0.049] | NOT DETECTED | +0.014 [-0.031, +0.058] | (reference) |  |
| ON | 6 | 0.832 [0.780, 0.880] | 0.811 [0.746, 0.873] | +0.021 [-0.030, +0.071] | NOT DETECTED | +0.029 [-0.016, +0.074] | +0.020 [-0.016, +0.053] | NOT DETECTED |

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### forum72 — OFF-POOL (72 teams, 144 games, 6475 rows, bank 0caf6037fee3)

**species_logp** (hidden species: mean log presence of the true species; 3954 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -3.659 [-4.319, -3.085] | -2.247 [-2.434, -2.069] | -1.412 [-1.923, -0.965] | WORSE | -0.124 [-0.235, -0.016] | (reference) |  |
| ON | 6 | -2.337 [-2.572, -2.127] | -2.247 [-2.434, -2.069] | -0.090 [-0.191, +0.011] | NOT DETECTED | +0.011 [-0.058, +0.081] | +1.322 [+0.928, +1.777] | BETTER |

**species_hit** (hidden species: share of the true species in the top k; 3954 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.317 [0.270, 0.364] | 0.319 [0.262, 0.379] | -0.002 [-0.046, +0.039] | NOT DETECTED | +0.005 [-0.037, +0.045] | (reference) |  |
| ON | 6 | 0.325 [0.276, 0.373] | 0.319 [0.262, 0.379] | +0.006 [-0.036, +0.046] | NOT DETECTED | +0.021 [-0.014, +0.055] | +0.008 [-0.011, +0.026] | NOT DETECTED |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -0.910 [-0.994, -0.833] | -0.998 [-1.086, -0.918] | +0.088 [+0.046, +0.131] | BETTER | +0.108 [+0.075, +0.142] | (reference) |  |
| ON | 6 | -0.984 [-1.067, -0.908] | -0.998 [-1.086, -0.918] | +0.015 [-0.015, +0.045] | NOT DETECTED | +0.038 [+0.015, +0.062] | -0.073 [-0.097, -0.049] | WORSE |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -5.678 [-6.035, -5.342] | -5.114 [-5.440, -4.806] | -0.564 [-0.709, -0.414] | WORSE | -0.123 [-0.221, -0.018] | (reference) |  |
| ON | 6 | -5.214 [-5.535, -4.920] | -5.114 [-5.440, -4.806] | -0.100 [-0.202, +0.002] | NOT DETECTED | +0.102 [+0.024, +0.183] | +0.464 [+0.375, +0.556] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.565 [0.537, 0.592] | 0.566 [0.529, 0.601] | -0.001 [-0.026, +0.024] | NOT DETECTED | +0.012 [-0.009, +0.034] | (reference) |  |
| ON | 6 | 0.554 [0.526, 0.583] | 0.566 [0.529, 0.601] | -0.012 [-0.032, +0.009] | NOT DETECTED | +0.001 [-0.015, +0.018] | -0.010 [-0.022, +0.001] | NOT DETECTED |

**items_logp** (unstated item of a revealed mon: log probability; 8013 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -0.661 [-0.908, -0.462] | -0.738 [-0.934, -0.560] | +0.078 [-0.051, +0.202] | NOT DETECTED | +0.139 [+0.063, +0.222] | (reference) |  |
| ON | 6 | -0.628 [-0.802, -0.483] | -0.738 [-0.934, -0.560] | +0.110 [+0.025, +0.202] | BETTER | +0.116 [+0.049, +0.190] | +0.033 [-0.041, +0.122] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 8013 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.779 [0.726, 0.828] | 0.734 [0.657, 0.807] | +0.045 [-0.015, +0.105] | NOT DETECTED | +0.048 [-0.007, +0.104] | (reference) |  |
| ON | 6 | 0.777 [0.721, 0.830] | 0.734 [0.657, 0.807] | +0.044 [-0.013, +0.100] | NOT DETECTED | +0.048 [-0.002, +0.099] | -0.001 [-0.024, +0.023] | NOT DETECTED |

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### p72 — OFF-POOL (72 teams, 144 games, 6643 rows, bank af668a3f65fb)

**species_logp** (hidden species: mean log presence of the true species; 3918 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -6.543 [-7.282, -5.845] | -3.030 [-3.283, -2.797] | -3.513 [-4.068, -2.974] | WORSE | -0.659 [-0.738, -0.573] | (reference) |  |
| ON | 6 | -2.970 [-3.169, -2.779] | -3.030 [-3.283, -2.797] | +0.060 [-0.040, +0.165] | NOT DETECTED | +0.118 [+0.041, +0.203] | +3.573 [+3.017, +4.163] | BETTER |

**species_hit** (hidden species: share of the true species in the top k; 3918 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.130 [0.107, 0.154] | 0.180 [0.143, 0.221] | -0.050 [-0.085, -0.019] | WORSE | -0.045 [-0.079, -0.015] | (reference) |  |
| ON | 6 | 0.166 [0.138, 0.198] | 0.180 [0.143, 0.221] | -0.014 [-0.041, +0.011] | NOT DETECTED | -0.009 [-0.033, +0.013] | +0.036 [+0.019, +0.055] | BETTER |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -1.088 [-1.153, -1.021] | -1.015 [-1.076, -0.951] | -0.073 [-0.101, -0.046] | WORSE | -0.010 [-0.030, +0.009] | (reference) |  |
| ON | 6 | -1.068 [-1.127, -1.006] | -1.015 [-1.076, -0.951] | -0.053 [-0.070, -0.035] | WORSE | -0.013 [-0.026, +0.000] | +0.020 [-0.000, +0.041] | NOT DETECTED |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -6.985 [-7.281, -6.692] | -5.597 [-5.828, -5.369] | -1.387 [-1.510, -1.264] | WORSE | -0.594 [-0.663, -0.527] | (reference) |  |
| ON | 6 | -6.038 [-6.285, -5.788] | -5.597 [-5.828, -5.369] | -0.440 [-0.539, -0.345] | WORSE | -0.067 [-0.128, -0.004] | +0.947 [+0.833, +1.067] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.449 [0.420, 0.477] | 0.486 [0.450, 0.521] | -0.037 [-0.056, -0.019] | WORSE | -0.021 [-0.036, -0.007] | (reference) |  |
| ON | 6 | 0.456 [0.428, 0.484] | 0.486 [0.450, 0.521] | -0.030 [-0.048, -0.012] | WORSE | -0.019 [-0.034, -0.003] | +0.008 [-0.002, +0.018] | NOT DETECTED |

**items_logp** (unstated item of a revealed mon: log probability; 9437 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | -0.833 [-1.090, -0.611] | -0.634 [-0.758, -0.517] | -0.199 [-0.373, -0.050] | WORSE | -0.015 [-0.083, +0.051] | (reference) |  |
| ON | 6 | -0.533 [-0.644, -0.431] | -0.634 [-0.758, -0.517] | +0.102 [+0.052, +0.155] | BETTER | +0.094 [+0.053, +0.139] | +0.300 [+0.159, +0.473] | BETTER |

**items_top1** (unstated item of a revealed mon: top-1 rate; 9437 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − OFF |  |
|---|---|---|---|---|---|---|---|---|
| OFF | 6 | 0.761 [0.709, 0.810] | 0.735 [0.670, 0.797] | +0.025 [-0.017, +0.067] | NOT DETECTED | +0.025 [-0.014, +0.064] | (reference) |  |
| ON | 6 | 0.792 [0.739, 0.841] | 0.735 [0.670, 0.797] | +0.056 [+0.014, +0.099] | BETTER | +0.059 [+0.023, +0.096] | +0.031 [+0.002, +0.062] | BETTER |

Units not scored / flagged (first arm): moves_prec_near_tie 1, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0
