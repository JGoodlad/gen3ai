### home72 — IN-POOL reference (72 teams, 144 games, 6785 rows, bank 5207d8c100df)

**species_logp** (hidden species: mean log presence of the true species; 3950 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -0.612 [-0.689, -0.544] | -1.843 [-1.984, -1.705] | +1.231 [+1.085, +1.379] | BETTER | +1.099 [+0.967, +1.236] | (reference) |  |
| B | 1 | -0.615 [-0.696, -0.543] | -1.843 [-1.984, -1.705] | +1.227 [+1.077, +1.379] | BETTER | +1.101 [+0.969, +1.236] | -0.003 [-0.023, +0.016] | NOT DETECTED |
| A | 1 | -0.612 [-0.690, -0.541] | -1.843 [-1.984, -1.705] | +1.231 [+1.086, +1.379] | BETTER | +1.100 [+0.971, +1.235] | +0.000 [-0.014, +0.016] | NOT DETECTED |
| AB | 1 | -0.602 [-0.672, -0.537] | -1.843 [-1.984, -1.705] | +1.241 [+1.101, +1.384] | BETTER | +1.103 [+0.975, +1.238] | +0.010 [-0.005, +0.026] | NOT DETECTED |

**species_hit** (hidden species: share of the true species in the top k; 3950 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | 0.776 [0.745, 0.805] | 0.429 [0.368, 0.495] | +0.347 [+0.283, +0.408] | BETTER | +0.351 [+0.289, +0.411] | (reference) |  |
| B | 1 | 0.775 [0.746, 0.803] | 0.429 [0.368, 0.495] | +0.346 [+0.282, +0.407] | BETTER | +0.348 [+0.286, +0.407] | -0.001 [-0.009, +0.008] | NOT DETECTED |
| A | 1 | 0.779 [0.750, 0.806] | 0.429 [0.368, 0.495] | +0.350 [+0.286, +0.411] | BETTER | +0.353 [+0.291, +0.412] | +0.003 [-0.005, +0.012] | NOT DETECTED |
| AB | 1 | 0.779 [0.748, 0.808] | 0.429 [0.368, 0.495] | +0.350 [+0.287, +0.411] | BETTER | +0.353 [+0.292, +0.413] | +0.003 [-0.008, +0.013] | NOT DETECTED |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -0.732 [-0.795, -0.673] | -0.874 [-0.933, -0.816] | +0.141 [+0.069, +0.215] | BETTER | +0.169 [+0.116, +0.225] | (reference) |  |
| B | 1 | -0.743 [-0.806, -0.683] | -0.874 [-0.933, -0.816] | +0.130 [+0.061, +0.201] | BETTER | +0.161 [+0.111, +0.214] | -0.011 [-0.020, -0.002] | WORSE |
| A | 1 | -0.783 [-0.848, -0.720] | -0.874 [-0.933, -0.816] | +0.090 [+0.016, +0.165] | BETTER | +0.129 [+0.076, +0.184] | -0.051 [-0.068, -0.034] | WORSE |
| AB | 1 | -0.783 [-0.849, -0.720] | -0.874 [-0.933, -0.816] | +0.090 [+0.017, +0.164] | BETTER | +0.129 [+0.076, +0.183] | -0.051 [-0.067, -0.036] | WORSE |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -4.405 [-4.798, -4.048] | -4.799 [-5.035, -4.574] | +0.394 [-0.001, +0.755] | NOT DETECTED | +0.723 [+0.528, +0.924] | (reference) |  |
| B | 1 | -4.360 [-4.756, -4.003] | -4.799 [-5.035, -4.574] | +0.440 [+0.043, +0.801] | BETTER | +0.747 [+0.555, +0.944] | +0.046 [+0.015, +0.077] | BETTER |
| A | 1 | -3.874 [-4.241, -3.548] | -4.799 [-5.035, -4.574] | +0.925 [+0.558, +1.265] | BETTER | +1.039 [+0.849, +1.238] | +0.531 [+0.463, +0.599] | BETTER |
| AB | 1 | -3.861 [-4.225, -3.535] | -4.799 [-5.035, -4.574] | +0.938 [+0.571, +1.274] | BETTER | +1.045 [+0.856, +1.242] | +0.544 [+0.475, +0.616] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | 0.658 [0.631, 0.685] | 0.604 [0.572, 0.636] | +0.053 [+0.018, +0.091] | BETTER | +0.068 [+0.036, +0.101] | (reference) |  |
| B | 1 | 0.665 [0.639, 0.692] | 0.604 [0.572, 0.636] | +0.061 [+0.024, +0.099] | BETTER | +0.074 [+0.045, +0.106] | +0.008 [-0.002, +0.017] | NOT DETECTED |
| A | 1 | 0.692 [0.666, 0.718] | 0.604 [0.572, 0.636] | +0.088 [+0.053, +0.124] | BETTER | +0.095 [+0.063, +0.129] | +0.035 [+0.023, +0.047] | BETTER |
| AB | 1 | 0.688 [0.660, 0.715] | 0.604 [0.572, 0.636] | +0.083 [+0.048, +0.119] | BETTER | +0.086 [+0.055, +0.120] | +0.030 [+0.018, +0.042] | BETTER |

**items_logp** (unstated item of a revealed mon: log probability; 7887 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -0.276 [-0.358, -0.207] | -0.461 [-0.573, -0.361] | +0.185 [+0.114, +0.265] | BETTER | +0.169 [+0.111, +0.236] | (reference) |  |
| B | 1 | -0.279 [-0.365, -0.206] | -0.461 [-0.573, -0.361] | +0.182 [+0.113, +0.258] | BETTER | +0.168 [+0.112, +0.231] | -0.003 [-0.012, +0.006] | NOT DETECTED |
| A | 1 | -0.280 [-0.363, -0.211] | -0.461 [-0.573, -0.361] | +0.181 [+0.107, +0.263] | BETTER | +0.167 [+0.108, +0.236] | -0.005 [-0.014, +0.005] | NOT DETECTED |
| AB | 1 | -0.263 [-0.352, -0.189] | -0.461 [-0.573, -0.361] | +0.198 [+0.125, +0.277] | BETTER | +0.179 [+0.121, +0.246] | +0.012 [-0.003, +0.026] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 7887 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | 0.907 [0.869, 0.940] | 0.848 [0.783, 0.906] | +0.059 [+0.018, +0.104] | BETTER | +0.064 [+0.026, +0.108] | (reference) |  |
| B | 1 | 0.904 [0.866, 0.937] | 0.848 [0.783, 0.906] | +0.056 [+0.017, +0.100] | BETTER | +0.066 [+0.028, +0.110] | -0.003 [-0.009, +0.002] | NOT DETECTED |
| A | 1 | 0.907 [0.871, 0.940] | 0.848 [0.783, 0.906] | +0.060 [+0.019, +0.106] | BETTER | +0.072 [+0.031, +0.118] | +0.001 [-0.003, +0.005] | NOT DETECTED |
| AB | 1 | 0.915 [0.876, 0.947] | 0.848 [0.783, 0.906] | +0.067 [+0.024, +0.115] | BETTER | +0.062 [+0.026, +0.104] | +0.008 [+0.000, +0.019] | BETTER |

Observation-semantics rows between a checkpoint's recorded config and the bank's encoding (scored through them, never trained on them): C0: [[162, 163, 165, 166, 167, 168]]; B: [[162, 163, 165, 166, 167, 168]]; A: [[162, 163, 165, 166, 167, 168]]; AB: [[162, 163, 165, 166, 167, 168]]

Units not scored / flagged (first arm): moves_prec_near_tie 50, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### heldout72_hpfix — OFF-POOL (72 teams, 144 games, 6636 rows, bank 21545e87b1fb)

**species_logp** (hidden species: mean log presence of the true species; 4316 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -5.088 [-6.456, -3.828] | -2.082 [-2.275, -1.895] | -3.006 [-4.242, -1.873] | WORSE | -0.158 [-0.297, -0.011] | (reference) |  |
| B | 1 | -5.112 [-6.488, -3.847] | -2.082 [-2.275, -1.895] | -3.030 [-4.271, -1.893] | WORSE | -0.169 [-0.308, -0.024] | -0.024 [-0.088, +0.040] | NOT DETECTED |
| A | 1 | -5.092 [-6.467, -3.835] | -2.082 [-2.275, -1.895] | -3.010 [-4.254, -1.879] | WORSE | -0.161 [-0.302, -0.014] | -0.004 [-0.080, +0.071] | NOT DETECTED |
| AB | 1 | -5.122 [-6.500, -3.853] | -2.082 [-2.275, -1.895] | -3.040 [-4.298, -1.891] | WORSE | -0.163 [-0.304, -0.015] | -0.034 [-0.106, +0.038] | NOT DETECTED |

**species_hit** (hidden species: share of the true species in the top k; 4316 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | 0.365 [0.307, 0.427] | 0.388 [0.327, 0.452] | -0.023 [-0.070, +0.024] | NOT DETECTED | -0.016 [-0.062, +0.030] | (reference) |  |
| B | 1 | 0.362 [0.304, 0.424] | 0.388 [0.327, 0.452] | -0.026 [-0.072, +0.021] | NOT DETECTED | -0.018 [-0.063, +0.028] | -0.003 [-0.009, +0.004] | NOT DETECTED |
| A | 1 | 0.370 [0.312, 0.431] | 0.388 [0.327, 0.452] | -0.017 [-0.067, +0.033] | NOT DETECTED | -0.012 [-0.061, +0.037] | +0.005 [-0.008, +0.020] | NOT DETECTED |
| AB | 1 | 0.365 [0.304, 0.430] | 0.388 [0.327, 0.452] | -0.022 [-0.070, +0.025] | NOT DETECTED | -0.015 [-0.063, +0.031] | +0.000 [-0.010, +0.012] | NOT DETECTED |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -1.185 [-1.295, -1.085] | -0.855 [-0.904, -0.807] | -0.330 [-0.417, -0.252] | WORSE | -0.125 [-0.164, -0.086] | (reference) |  |
| B | 1 | -1.189 [-1.301, -1.088] | -0.855 [-0.904, -0.807] | -0.334 [-0.422, -0.256] | WORSE | -0.128 [-0.166, -0.089] | -0.004 [-0.015, +0.007] | NOT DETECTED |
| A | 1 | -1.252 [-1.362, -1.152] | -0.855 [-0.904, -0.807] | -0.397 [-0.480, -0.321] | WORSE | -0.176 [-0.214, -0.137] | -0.067 [-0.083, -0.051] | WORSE |
| AB | 1 | -1.270 [-1.381, -1.170] | -0.855 [-0.904, -0.807] | -0.416 [-0.500, -0.338] | WORSE | -0.182 [-0.220, -0.144] | -0.086 [-0.102, -0.070] | WORSE |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -6.795 [-7.241, -6.350] | -4.969 [-5.156, -4.781] | -1.826 [-2.201, -1.464] | WORSE | -0.455 [-0.607, -0.302] | (reference) |  |
| B | 1 | -6.749 [-7.197, -6.307] | -4.969 [-5.156, -4.781] | -1.780 [-2.160, -1.416] | WORSE | -0.428 [-0.585, -0.271] | +0.046 [+0.001, +0.088] | BETTER |
| A | 1 | -6.243 [-6.668, -5.821] | -4.969 [-5.156, -4.781] | -1.274 [-1.627, -0.933] | WORSE | -0.168 [-0.311, -0.025] | +0.552 [+0.480, +0.624] | BETTER |
| AB | 1 | -6.301 [-6.725, -5.874] | -4.969 [-5.156, -4.781] | -1.332 [-1.690, -0.984] | WORSE | -0.190 [-0.335, -0.044] | +0.494 [+0.427, +0.559] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | 0.487 [0.456, 0.516] | 0.530 [0.497, 0.563] | -0.043 [-0.068, -0.018] | WORSE | -0.021 [-0.043, +0.003] | (reference) |  |
| B | 1 | 0.494 [0.464, 0.525] | 0.530 [0.497, 0.563] | -0.036 [-0.058, -0.012] | WORSE | -0.011 [-0.033, +0.012] | +0.008 [-0.002, +0.019] | NOT DETECTED |
| A | 1 | 0.504 [0.472, 0.535] | 0.530 [0.497, 0.563] | -0.027 [-0.050, -0.002] | WORSE | -0.014 [-0.036, +0.009] | +0.017 [+0.007, +0.027] | BETTER |
| AB | 1 | 0.500 [0.470, 0.530] | 0.530 [0.497, 0.563] | -0.030 [-0.053, -0.006] | WORSE | -0.012 [-0.033, +0.011] | +0.014 [+0.003, +0.025] | BETTER |

**items_logp** (unstated item of a revealed mon: log probability; 7198 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -0.492 [-0.623, -0.378] | -0.475 [-0.571, -0.387] | -0.017 [-0.101, +0.060] | NOT DETECTED | +0.044 [-0.004, +0.089] | (reference) |  |
| B | 1 | -0.496 [-0.634, -0.377] | -0.475 [-0.571, -0.387] | -0.021 [-0.113, +0.062] | NOT DETECTED | +0.047 [-0.002, +0.095] | -0.004 [-0.022, +0.012] | NOT DETECTED |
| A | 1 | -0.497 [-0.627, -0.382] | -0.475 [-0.571, -0.387] | -0.021 [-0.108, +0.057] | NOT DETECTED | +0.040 [-0.009, +0.088] | -0.005 [-0.021, +0.011] | NOT DETECTED |
| AB | 1 | -0.486 [-0.623, -0.365] | -0.475 [-0.571, -0.387] | -0.010 [-0.101, +0.071] | NOT DETECTED | +0.053 [+0.003, +0.101] | +0.006 [-0.014, +0.025] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 7198 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | 0.827 [0.780, 0.872] | 0.811 [0.746, 0.873] | +0.016 [-0.033, +0.064] | NOT DETECTED | +0.034 [-0.007, +0.077] | (reference) |  |
| B | 1 | 0.823 [0.772, 0.871] | 0.811 [0.746, 0.873] | +0.012 [-0.039, +0.062] | NOT DETECTED | +0.042 [+0.002, +0.086] | -0.004 [-0.018, +0.012] | NOT DETECTED |
| A | 1 | 0.829 [0.780, 0.875] | 0.811 [0.746, 0.873] | +0.018 [-0.032, +0.067] | NOT DETECTED | +0.032 [-0.016, +0.079] | +0.002 [-0.009, +0.017] | NOT DETECTED |
| AB | 1 | 0.819 [0.765, 0.868] | 0.811 [0.746, 0.873] | +0.007 [-0.050, +0.061] | NOT DETECTED | +0.035 [-0.015, +0.084] | -0.009 [-0.029, +0.005] | NOT DETECTED |

Observation-semantics rows between a checkpoint's recorded config and the bank's encoding (scored through them, never trained on them): C0: [[162, 163, 165, 166, 167, 168]]; B: [[162, 163, 165, 166, 167, 168]]; A: [[162, 163, 165, 166, 167, 168]]; AB: [[162, 163, 165, 166, 167, 168]]

Units not scored / flagged (first arm): moves_prec_near_tie 222, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### forum72 — OFF-POOL (72 teams, 144 games, 6475 rows, bank 0caf6037fee3)

**species_logp** (hidden species: mean log presence of the true species; 3954 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -6.268 [-7.786, -4.974] | -2.247 [-2.434, -2.069] | -4.022 [-5.397, -2.850] | WORSE | -0.306 [-0.440, -0.171] | (reference) |  |
| B | 1 | -6.292 [-7.800, -5.002] | -2.247 [-2.434, -2.069] | -4.045 [-5.417, -2.877] | WORSE | -0.303 [-0.437, -0.167] | -0.023 [-0.101, +0.049] | NOT DETECTED |
| A | 1 | -6.262 [-7.785, -4.970] | -2.247 [-2.434, -2.069] | -4.015 [-5.395, -2.840] | WORSE | -0.312 [-0.447, -0.174] | +0.006 [-0.060, +0.074] | NOT DETECTED |
| AB | 1 | -6.218 [-7.734, -4.930] | -2.247 [-2.434, -2.069] | -3.971 [-5.341, -2.803] | WORSE | -0.309 [-0.442, -0.176] | +0.050 [-0.016, +0.121] | NOT DETECTED |

**species_hit** (hidden species: share of the true species in the top k; 3954 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | 0.289 [0.237, 0.342] | 0.319 [0.262, 0.379] | -0.031 [-0.078, +0.017] | NOT DETECTED | -0.023 [-0.070, +0.022] | (reference) |  |
| B | 1 | 0.302 [0.248, 0.357] | 0.319 [0.262, 0.379] | -0.017 [-0.065, +0.030] | NOT DETECTED | -0.012 [-0.060, +0.035] | +0.013 [+0.001, +0.025] | BETTER |
| A | 1 | 0.285 [0.234, 0.339] | 0.319 [0.262, 0.379] | -0.034 [-0.084, +0.016] | NOT DETECTED | -0.027 [-0.076, +0.022] | -0.003 [-0.011, +0.004] | NOT DETECTED |
| AB | 1 | 0.289 [0.238, 0.343] | 0.319 [0.262, 0.379] | -0.030 [-0.078, +0.017] | NOT DETECTED | -0.025 [-0.072, +0.021] | +0.001 [-0.012, +0.011] | NOT DETECTED |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -1.056 [-1.152, -0.967] | -0.998 [-1.086, -0.918] | -0.057 [-0.121, +0.008] | NOT DETECTED | +0.037 [-0.005, +0.081] | (reference) |  |
| B | 1 | -1.050 [-1.145, -0.962] | -0.998 [-1.086, -0.918] | -0.051 [-0.116, +0.015] | NOT DETECTED | +0.040 [-0.004, +0.085] | +0.006 [-0.003, +0.015] | NOT DETECTED |
| A | 1 | -1.097 [-1.196, -1.005] | -0.998 [-1.086, -0.918] | -0.098 [-0.161, -0.037] | WORSE | +0.004 [-0.037, +0.045] | -0.041 [-0.060, -0.023] | WORSE |
| AB | 1 | -1.107 [-1.207, -1.016] | -0.998 [-1.086, -0.918] | -0.109 [-0.173, -0.046] | WORSE | -0.003 [-0.045, +0.039] | -0.051 [-0.067, -0.037] | WORSE |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -5.738 [-6.299, -5.237] | -5.114 [-5.440, -4.806] | -0.624 [-1.074, -0.245] | WORSE | +0.112 [-0.047, +0.265] | (reference) |  |
| B | 1 | -5.645 [-6.212, -5.149] | -5.114 [-5.440, -4.806] | -0.532 [-1.001, -0.144] | WORSE | +0.177 [+0.019, +0.331] | +0.093 [+0.047, +0.140] | BETTER |
| A | 1 | -5.168 [-5.692, -4.700] | -5.114 [-5.440, -4.806] | -0.054 [-0.474, +0.290] | NOT DETECTED | +0.434 [+0.286, +0.580] | +0.571 [+0.497, +0.649] | BETTER |
| AB | 1 | -5.192 [-5.713, -4.727] | -5.114 [-5.440, -4.806] | -0.078 [-0.497, +0.266] | NOT DETECTED | +0.417 [+0.266, +0.565] | +0.546 [+0.476, +0.621] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | 0.566 [0.535, 0.596] | 0.566 [0.529, 0.601] | -0.000 [-0.028, +0.030] | NOT DETECTED | +0.014 [-0.010, +0.039] | (reference) |  |
| B | 1 | 0.572 [0.539, 0.603] | 0.566 [0.529, 0.601] | +0.006 [-0.022, +0.036] | NOT DETECTED | +0.013 [-0.012, +0.039] | +0.006 [+0.000, +0.012] | BETTER |
| A | 1 | 0.578 [0.545, 0.609] | 0.566 [0.529, 0.601] | +0.012 [-0.016, +0.040] | NOT DETECTED | +0.027 [+0.002, +0.053] | +0.012 [+0.000, +0.023] | BETTER |
| AB | 1 | 0.580 [0.547, 0.611] | 0.566 [0.529, 0.601] | +0.014 [-0.013, +0.042] | NOT DETECTED | +0.030 [+0.004, +0.057] | +0.014 [+0.002, +0.026] | BETTER |

**items_logp** (unstated item of a revealed mon: log probability; 8013 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -0.749 [-1.084, -0.481] | -0.738 [-0.934, -0.560] | -0.011 [-0.234, +0.179] | NOT DETECTED | +0.149 [+0.067, +0.238] | (reference) |  |
| B | 1 | -0.754 [-1.082, -0.487] | -0.738 [-0.934, -0.560] | -0.015 [-0.225, +0.166] | NOT DETECTED | +0.139 [+0.060, +0.225] | -0.004 [-0.022, +0.015] | NOT DETECTED |
| A | 1 | -0.749 [-1.079, -0.483] | -0.738 [-0.934, -0.560] | -0.011 [-0.232, +0.180] | NOT DETECTED | +0.148 [+0.060, +0.243] | +0.000 [-0.015, +0.016] | NOT DETECTED |
| AB | 1 | -0.731 [-1.074, -0.457] | -0.738 [-0.934, -0.560] | +0.007 [-0.221, +0.200] | NOT DETECTED | +0.164 [+0.084, +0.251] | +0.018 [-0.006, +0.038] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 8013 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | 0.791 [0.737, 0.843] | 0.734 [0.657, 0.807] | +0.057 [-0.006, +0.122] | NOT DETECTED | +0.064 [+0.009, +0.122] | (reference) |  |
| B | 1 | 0.789 [0.735, 0.841] | 0.734 [0.657, 0.807] | +0.056 [-0.007, +0.120] | NOT DETECTED | +0.069 [+0.020, +0.121] | -0.002 [-0.006, +0.002] | NOT DETECTED |
| A | 1 | 0.797 [0.745, 0.847] | 0.734 [0.657, 0.807] | +0.064 [-0.002, +0.131] | NOT DETECTED | +0.061 [+0.002, +0.122] | +0.006 [-0.001, +0.015] | NOT DETECTED |
| AB | 1 | 0.794 [0.740, 0.846] | 0.734 [0.657, 0.807] | +0.060 [-0.003, +0.125] | NOT DETECTED | +0.084 [+0.034, +0.138] | +0.003 [-0.001, +0.008] | NOT DETECTED |

Observation-semantics rows between a checkpoint's recorded config and the bank's encoding (scored through them, never trained on them): C0: [[162, 163, 165, 166, 167, 168]]; B: [[162, 163, 165, 166, 167, 168]]; A: [[162, 163, 165, 166, 167, 168]]; AB: [[162, 163, 165, 166, 167, 168]]

Units not scored / flagged (first arm): moves_prec_near_tie 56, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### p72 — OFF-POOL (72 teams, 144 games, 6643 rows, bank af668a3f65fb)

**species_logp** (hidden species: mean log presence of the true species; 3918 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -10.834 [-12.191, -9.521] | -3.030 [-3.283, -2.797] | -7.804 [-9.005, -6.630] | WORSE | -0.803 [-0.885, -0.716] | (reference) |  |
| B | 1 | -10.960 [-12.325, -9.653] | -3.030 [-3.283, -2.797] | -7.930 [-9.134, -6.762] | WORSE | -0.831 [-0.908, -0.750] | -0.126 [-0.230, -0.021] | WORSE |
| A | 1 | -10.740 [-12.131, -9.409] | -3.030 [-3.283, -2.797] | -7.711 [-8.931, -6.525] | WORSE | -0.798 [-0.884, -0.706] | +0.094 [-0.009, +0.199] | NOT DETECTED |
| AB | 1 | -10.875 [-12.227, -9.569] | -3.030 [-3.283, -2.797] | -7.846 [-9.037, -6.685] | WORSE | -0.814 [-0.895, -0.727] | -0.041 [-0.129, +0.045] | NOT DETECTED |

**species_hit** (hidden species: share of the true species in the top k; 3918 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | 0.132 [0.106, 0.160] | 0.180 [0.143, 0.221] | -0.048 [-0.083, -0.017] | WORSE | -0.043 [-0.078, -0.012] | (reference) |  |
| B | 1 | 0.131 [0.105, 0.159] | 0.180 [0.143, 0.221] | -0.049 [-0.083, -0.018] | WORSE | -0.046 [-0.080, -0.015] | -0.001 [-0.005, +0.003] | NOT DETECTED |
| A | 1 | 0.133 [0.106, 0.161] | 0.180 [0.143, 0.221] | -0.047 [-0.082, -0.015] | WORSE | -0.045 [-0.079, -0.013] | +0.001 [-0.004, +0.006] | NOT DETECTED |
| AB | 1 | 0.131 [0.106, 0.158] | 0.180 [0.143, 0.221] | -0.049 [-0.084, -0.017] | WORSE | -0.043 [-0.077, -0.012] | -0.001 [-0.008, +0.006] | NOT DETECTED |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -1.356 [-1.449, -1.264] | -1.015 [-1.076, -0.951] | -0.341 [-0.398, -0.285] | WORSE | -0.130 [-0.165, -0.095] | (reference) |  |
| B | 1 | -1.357 [-1.447, -1.267] | -1.015 [-1.076, -0.951] | -0.342 [-0.396, -0.287] | WORSE | -0.134 [-0.167, -0.100] | -0.001 [-0.012, +0.010] | NOT DETECTED |
| A | 1 | -1.424 [-1.517, -1.332] | -1.015 [-1.076, -0.951] | -0.410 [-0.466, -0.353] | WORSE | -0.184 [-0.218, -0.149] | -0.069 [-0.090, -0.050] | WORSE |
| AB | 1 | -1.435 [-1.529, -1.342] | -1.015 [-1.076, -0.951] | -0.421 [-0.478, -0.362] | WORSE | -0.186 [-0.219, -0.151] | -0.079 [-0.099, -0.061] | WORSE |

**moves_ll** (unshown moves of a revealed mon: log-likelihood of the whole candidate set, nats a unit; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -7.823 [-8.291, -7.373] | -5.597 [-5.828, -5.369] | -2.226 [-2.556, -1.920] | WORSE | -0.605 [-0.735, -0.480] | (reference) |  |
| B | 1 | -7.709 [-8.177, -7.259] | -5.597 [-5.828, -5.369] | -2.112 [-2.442, -1.802] | WORSE | -0.557 [-0.689, -0.430] | +0.114 [+0.070, +0.158] | BETTER |
| A | 1 | -7.284 [-7.721, -6.857] | -5.597 [-5.828, -5.369] | -1.686 [-1.989, -1.410] | WORSE | -0.333 [-0.449, -0.220] | +0.540 [+0.452, +0.626] | BETTER |
| AB | 1 | -7.317 [-7.762, -6.887] | -5.597 [-5.828, -5.369] | -1.720 [-2.032, -1.435] | WORSE | -0.344 [-0.462, -0.228] | +0.506 [+0.433, +0.580] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | 0.417 [0.388, 0.445] | 0.486 [0.450, 0.521] | -0.069 [-0.091, -0.047] | WORSE | -0.046 [-0.066, -0.026] | (reference) |  |
| B | 1 | 0.416 [0.386, 0.445] | 0.486 [0.450, 0.521] | -0.070 [-0.092, -0.049] | WORSE | -0.039 [-0.058, -0.019] | -0.001 [-0.010, +0.008] | NOT DETECTED |
| A | 1 | 0.432 [0.401, 0.463] | 0.486 [0.450, 0.521] | -0.054 [-0.076, -0.032] | WORSE | -0.031 [-0.053, -0.010] | +0.015 [+0.006, +0.025] | BETTER |
| AB | 1 | 0.428 [0.397, 0.459] | 0.486 [0.450, 0.521] | -0.058 [-0.080, -0.036] | WORSE | -0.036 [-0.056, -0.015] | +0.011 [+0.002, +0.021] | BETTER |

**items_logp** (unstated item of a revealed mon: log probability; 9437 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | -1.030 [-1.350, -0.750] | -0.634 [-0.758, -0.517] | -0.395 [-0.643, -0.175] | WORSE | -0.033 [-0.109, +0.044] | (reference) |  |
| B | 1 | -1.021 [-1.343, -0.740] | -0.634 [-0.758, -0.517] | -0.387 [-0.634, -0.169] | WORSE | -0.032 [-0.111, +0.047] | +0.008 [-0.014, +0.029] | NOT DETECTED |
| A | 1 | -1.017 [-1.340, -0.737] | -0.634 [-0.758, -0.517] | -0.383 [-0.634, -0.163] | WORSE | -0.026 [-0.104, +0.052] | +0.013 [-0.009, +0.037] | NOT DETECTED |
| AB | 1 | -1.060 [-1.388, -0.772] | -0.634 [-0.758, -0.517] | -0.426 [-0.683, -0.198] | WORSE | -0.040 [-0.119, +0.039] | -0.030 [-0.061, -0.002] | WORSE |

**items_top1** (unstated item of a revealed mon: top-1 rate; 9437 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| C0 | 5 | 0.764 [0.708, 0.816] | 0.735 [0.670, 0.797] | +0.029 [-0.020, +0.076] | NOT DETECTED | +0.028 [-0.019, +0.074] | (reference) |  |
| B | 1 | 0.768 [0.713, 0.819] | 0.735 [0.670, 0.797] | +0.033 [-0.017, +0.081] | NOT DETECTED | +0.031 [-0.017, +0.077] | +0.004 [-0.001, +0.009] | NOT DETECTED |
| A | 1 | 0.767 [0.712, 0.818] | 0.735 [0.670, 0.797] | +0.031 [-0.019, +0.080] | NOT DETECTED | +0.029 [-0.019, +0.075] | +0.003 [-0.002, +0.007] | NOT DETECTED |
| AB | 1 | 0.758 [0.703, 0.810] | 0.735 [0.670, 0.797] | +0.023 [-0.028, +0.072] | NOT DETECTED | +0.018 [-0.031, +0.065] | -0.006 [-0.014, +0.001] | NOT DETECTED |

Observation-semantics rows between a checkpoint's recorded config and the bank's encoding (scored through them, never trained on them): C0: [[162, 163, 165, 166, 167, 168]]; B: [[162, 163, 165, 166, 167, 168]]; A: [[162, 163, 165, 166, 167, 168]]; AB: [[162, 163, 165, 166, 167, 168]]

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0
