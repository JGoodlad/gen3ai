### home72 — IN-POOL reference (72 teams, 144 games, 6785 rows, bank 5207d8c100df)

**species_logp** (hidden species: mean log presence of the true species; 3950 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | -0.627 [-0.707, -0.554] | -1.843 [-1.984, -1.705] | +1.216 [+1.069, +1.363] | BETTER | +1.088 [+0.958, +1.222] | -0.015 [-0.031, +0.002] | NOT DETECTED |
| C0 | 5 | -0.612 [-0.689, -0.544] | -1.843 [-1.984, -1.705] | +1.231 [+1.085, +1.379] | BETTER | +1.099 [+0.967, +1.236] | (reference) |  |
| N1P | 5 | -1.249 [-1.345, -1.158] | -1.843 [-1.984, -1.705] | +0.593 [+0.473, +0.720] | BETTER | +0.518 [+0.415, +0.628] | -0.637 [-0.735, -0.542] | WORSE |
| X60T | 6 | -0.955 [-1.037, -0.879] | -1.843 [-1.984, -1.705] | +0.887 [+0.767, +1.014] | BETTER | +0.772 [+0.664, +0.886] | -0.343 [-0.424, -0.269] | WORSE |

**species_hit** (hidden species: share of the true species in the top k; 3950 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | 0.769 [0.737, 0.797] | 0.429 [0.368, 0.495] | +0.340 [+0.277, +0.400] | BETTER | +0.343 [+0.282, +0.401] | -0.008 [-0.018, +0.003] | NOT DETECTED |
| C0 | 5 | 0.776 [0.745, 0.805] | 0.429 [0.368, 0.495] | +0.347 [+0.283, +0.408] | BETTER | +0.351 [+0.289, +0.411] | (reference) |  |
| N1P | 5 | 0.626 [0.578, 0.673] | 0.429 [0.368, 0.495] | +0.197 [+0.142, +0.249] | BETTER | +0.196 [+0.146, +0.244] | -0.150 [-0.195, -0.107] | WORSE |
| X60T | 6 | 0.705 [0.667, 0.740] | 0.429 [0.368, 0.495] | +0.276 [+0.218, +0.332] | BETTER | +0.278 [+0.223, +0.331] | -0.072 [-0.101, -0.042] | WORSE |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | -0.734 [-0.798, -0.671] | -0.874 [-0.933, -0.816] | +0.140 [+0.066, +0.216] | BETTER | +0.168 [+0.113, +0.226] | -0.001 [-0.010, +0.007] | NOT DETECTED |
| C0 | 5 | -0.732 [-0.795, -0.673] | -0.874 [-0.933, -0.816] | +0.141 [+0.069, +0.215] | BETTER | +0.169 [+0.116, +0.225] | (reference) |  |
| N1P | 5 | -0.790 [-0.847, -0.735] | -0.874 [-0.933, -0.816] | +0.084 [+0.020, +0.150] | BETTER | +0.119 [+0.072, +0.169] | -0.057 [-0.079, -0.035] | WORSE |
| X60T | 6 | -0.709 [-0.765, -0.654] | -0.874 [-0.933, -0.816] | +0.165 [+0.104, +0.228] | BETTER | +0.173 [+0.127, +0.222] | +0.024 [-0.003, +0.049] | NOT DETECTED |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 28057 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | 0.656 [0.629, 0.682] | 0.604 [0.572, 0.636] | +0.052 [+0.015, +0.090] | BETTER | +0.071 [+0.040, +0.104] | -0.002 [-0.013, +0.008] | NOT DETECTED |
| C0 | 5 | 0.658 [0.631, 0.685] | 0.604 [0.572, 0.636] | +0.053 [+0.018, +0.091] | BETTER | +0.068 [+0.036, +0.101] | (reference) |  |
| N1P | 5 | 0.644 [0.616, 0.671] | 0.604 [0.572, 0.636] | +0.040 [+0.006, +0.074] | BETTER | +0.050 [+0.019, +0.081] | -0.014 [-0.029, +0.002] | NOT DETECTED |
| X60T | 6 | 0.649 [0.622, 0.677] | 0.604 [0.572, 0.636] | +0.045 [+0.014, +0.077] | BETTER | +0.052 [+0.023, +0.083] | -0.009 [-0.025, +0.009] | NOT DETECTED |

**items_logp** (unstated item of a revealed mon: log probability; 7887 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | -0.283 [-0.366, -0.215] | -0.461 [-0.573, -0.361] | +0.178 [+0.106, +0.257] | BETTER | +0.164 [+0.105, +0.230] | -0.007 [-0.019, +0.006] | NOT DETECTED |
| C0 | 5 | -0.276 [-0.358, -0.207] | -0.461 [-0.573, -0.361] | +0.185 [+0.114, +0.265] | BETTER | +0.169 [+0.111, +0.236] | (reference) |  |
| N1P | 5 | -0.326 [-0.394, -0.266] | -0.461 [-0.573, -0.361] | +0.135 [+0.072, +0.207] | BETTER | +0.122 [+0.070, +0.182] | -0.050 [-0.083, -0.015] | WORSE |
| X60T | 6 | -0.305 [-0.378, -0.239] | -0.461 [-0.573, -0.361] | +0.157 [+0.103, +0.218] | BETTER | +0.135 [+0.090, +0.188] | -0.029 [-0.065, +0.009] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 7887 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | 0.905 [0.871, 0.936] | 0.848 [0.783, 0.906] | +0.057 [+0.015, +0.104] | BETTER | +0.062 [+0.023, +0.106] | -0.001 [-0.017, +0.019] | NOT DETECTED |
| C0 | 5 | 0.907 [0.869, 0.940] | 0.848 [0.783, 0.906] | +0.059 [+0.018, +0.104] | BETTER | +0.064 [+0.026, +0.108] | (reference) |  |
| N1P | 5 | 0.899 [0.863, 0.932] | 0.848 [0.783, 0.906] | +0.051 [+0.010, +0.098] | BETTER | +0.057 [+0.019, +0.101] | -0.007 [-0.026, +0.015] | NOT DETECTED |
| X60T | 6 | 0.909 [0.872, 0.942] | 0.848 [0.783, 0.906] | +0.062 [+0.024, +0.105] | BETTER | +0.064 [+0.028, +0.105] | +0.003 [-0.016, +0.025] | NOT DETECTED |

Observation-semantics rows between a checkpoint's recorded config and the bank's encoding (scored through them, never trained on them): PSTAR: [[162, 163, 165, 166, 167, 168]]; C0: [[162, 163, 165, 166, 167, 168]]; N1P: [[162, 163, 165, 166, 167, 168]]; X60T: [[162, 163, 165, 166, 167, 168]]

Units not scored / flagged (first arm): moves_prec_near_tie 50, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### heldout72_hpfix — OFF-POOL (72 teams, 144 games, 6636 rows, bank 21545e87b1fb)

**species_logp** (hidden species: mean log presence of the true species; 4316 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | -5.078 [-6.443, -3.826] | -2.082 [-2.275, -1.895] | -2.996 [-4.227, -1.864] | WORSE | -0.174 [-0.315, -0.028] | +0.010 [-0.070, +0.093] | NOT DETECTED |
| C0 | 5 | -5.088 [-6.456, -3.828] | -2.082 [-2.275, -1.895] | -3.006 [-4.242, -1.873] | WORSE | -0.158 [-0.297, -0.011] | (reference) |  |
| N1P | 5 | -2.042 [-2.281, -1.809] | -2.082 [-2.275, -1.895] | +0.040 [-0.058, +0.136] | NOT DETECTED | +0.079 [+0.015, +0.146] | +3.046 [+1.962, +4.231] | BETTER |
| X60T | 6 | -2.131 [-2.426, -1.852] | -2.082 [-2.275, -1.895] | -0.049 [-0.206, +0.098] | NOT DETECTED | +0.092 [+0.003, +0.183] | +2.957 [+1.918, +4.087] | BETTER |

**species_hit** (hidden species: share of the true species in the top k; 4316 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | 0.363 [0.304, 0.425] | 0.388 [0.327, 0.452] | -0.025 [-0.074, +0.025] | NOT DETECTED | -0.018 [-0.064, +0.029] | -0.002 [-0.014, +0.009] | NOT DETECTED |
| C0 | 5 | 0.365 [0.307, 0.427] | 0.388 [0.327, 0.452] | -0.023 [-0.070, +0.024] | NOT DETECTED | -0.016 [-0.062, +0.030] | (reference) |  |
| N1P | 5 | 0.418 [0.358, 0.481] | 0.388 [0.327, 0.452] | +0.031 [-0.007, +0.069] | NOT DETECTED | +0.037 [+0.002, +0.072] | +0.053 [+0.018, +0.089] | BETTER |
| X60T | 6 | 0.411 [0.356, 0.469] | 0.388 [0.327, 0.452] | +0.023 [-0.025, +0.071] | NOT DETECTED | +0.035 [-0.009, +0.080] | +0.046 [+0.016, +0.077] | BETTER |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | -1.187 [-1.296, -1.087] | -0.855 [-0.904, -0.807] | -0.332 [-0.418, -0.255] | WORSE | -0.130 [-0.169, -0.090] | -0.002 [-0.013, +0.007] | NOT DETECTED |
| C0 | 5 | -1.185 [-1.295, -1.085] | -0.855 [-0.904, -0.807] | -0.330 [-0.417, -0.252] | WORSE | -0.125 [-0.164, -0.086] | (reference) |  |
| N1P | 5 | -1.158 [-1.260, -1.064] | -0.855 [-0.904, -0.807] | -0.303 [-0.386, -0.229] | WORSE | -0.115 [-0.152, -0.079] | +0.027 [-0.003, +0.058] | NOT DETECTED |
| X60T | 6 | -1.052 [-1.150, -0.960] | -0.855 [-0.904, -0.807] | -0.197 [-0.272, -0.131] | WORSE | -0.059 [-0.094, -0.024] | +0.133 [+0.103, +0.163] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 25984 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | 0.483 [0.453, 0.513] | 0.530 [0.497, 0.563] | -0.047 [-0.071, -0.022] | WORSE | -0.019 [-0.042, +0.006] | -0.003 [-0.011, +0.004] | NOT DETECTED |
| C0 | 5 | 0.487 [0.456, 0.516] | 0.530 [0.497, 0.563] | -0.043 [-0.068, -0.018] | WORSE | -0.021 [-0.043, +0.003] | (reference) |  |
| N1P | 5 | 0.485 [0.456, 0.514] | 0.530 [0.497, 0.563] | -0.045 [-0.068, -0.022] | WORSE | -0.020 [-0.041, +0.002] | -0.002 [-0.013, +0.010] | NOT DETECTED |
| X60T | 6 | 0.500 [0.471, 0.530] | 0.530 [0.497, 0.563] | -0.030 [-0.051, -0.009] | WORSE | -0.014 [-0.035, +0.007] | +0.014 [+0.002, +0.025] | BETTER |

**items_logp** (unstated item of a revealed mon: log probability; 7198 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | -0.475 [-0.599, -0.367] | -0.475 [-0.571, -0.387] | +0.000 [-0.085, +0.078] | NOT DETECTED | +0.054 [+0.004, +0.103] | +0.017 [-0.002, +0.037] | NOT DETECTED |
| C0 | 5 | -0.492 [-0.623, -0.378] | -0.475 [-0.571, -0.387] | -0.017 [-0.101, +0.060] | NOT DETECTED | +0.044 [-0.004, +0.089] | (reference) |  |
| N1P | 5 | -0.441 [-0.540, -0.354] | -0.475 [-0.571, -0.387] | +0.035 [-0.026, +0.097] | NOT DETECTED | +0.050 [+0.007, +0.098] | +0.051 [-0.009, +0.121] | NOT DETECTED |
| X60T | 6 | -0.418 [-0.523, -0.325] | -0.475 [-0.571, -0.387] | +0.057 [-0.002, +0.118] | NOT DETECTED | +0.067 [+0.026, +0.111] | +0.074 [+0.016, +0.140] | BETTER |

**items_top1** (unstated item of a revealed mon: top-1 rate; 7198 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | 0.830 [0.780, 0.875] | 0.811 [0.746, 0.873] | +0.018 [-0.033, +0.069] | NOT DETECTED | +0.037 [-0.009, +0.084] | +0.002 [-0.010, +0.018] | NOT DETECTED |
| C0 | 5 | 0.827 [0.780, 0.872] | 0.811 [0.746, 0.873] | +0.016 [-0.033, +0.064] | NOT DETECTED | +0.034 [-0.007, +0.077] | (reference) |  |
| N1P | 5 | 0.839 [0.786, 0.888] | 0.811 [0.746, 0.873] | +0.028 [-0.021, +0.076] | NOT DETECTED | +0.040 [-0.001, +0.084] | +0.012 [-0.023, +0.044] | NOT DETECTED |
| X60T | 6 | 0.840 [0.785, 0.889] | 0.811 [0.746, 0.873] | +0.029 [-0.014, +0.072] | NOT DETECTED | +0.038 [+0.002, +0.076] | +0.013 [-0.016, +0.038] | NOT DETECTED |

Observation-semantics rows between a checkpoint's recorded config and the bank's encoding (scored through them, never trained on them): PSTAR: [[162, 163, 165, 166, 167, 168]]; C0: [[162, 163, 165, 166, 167, 168]]; N1P: [[162, 163, 165, 166, 167, 168]]; X60T: [[162, 163, 165, 166, 167, 168]]

Units not scored / flagged (first arm): moves_prec_near_tie 222, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### forum72 — OFF-POOL (72 teams, 144 games, 6475 rows, bank 0caf6037fee3)

**species_logp** (hidden species: mean log presence of the true species; 3954 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | -6.214 [-7.729, -4.915] | -2.247 [-2.434, -2.069] | -3.968 [-5.342, -2.790] | WORSE | -0.304 [-0.438, -0.169] | +0.054 [-0.010, +0.114] | NOT DETECTED |
| C0 | 5 | -6.268 [-7.786, -4.974] | -2.247 [-2.434, -2.069] | -4.022 [-5.397, -2.850] | WORSE | -0.306 [-0.440, -0.171] | (reference) |  |
| N1P | 5 | -2.321 [-2.568, -2.101] | -2.247 [-2.434, -2.069] | -0.074 [-0.171, +0.019] | NOT DETECTED | +0.007 [-0.054, +0.068] | +3.947 [+2.828, +5.265] | BETTER |
| X60T | 6 | -2.541 [-2.873, -2.243] | -2.247 [-2.434, -2.069] | -0.294 [-0.483, -0.122] | WORSE | -0.029 [-0.117, +0.058] | +3.728 [+2.658, +4.989] | BETTER |

**species_hit** (hidden species: share of the true species in the top k; 3954 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | 0.293 [0.240, 0.346] | 0.319 [0.262, 0.379] | -0.027 [-0.074, +0.020] | NOT DETECTED | -0.022 [-0.069, +0.023] | +0.004 [-0.007, +0.015] | NOT DETECTED |
| C0 | 5 | 0.289 [0.237, 0.342] | 0.319 [0.262, 0.379] | -0.031 [-0.078, +0.017] | NOT DETECTED | -0.023 [-0.070, +0.022] | (reference) |  |
| N1P | 5 | 0.324 [0.271, 0.376] | 0.319 [0.262, 0.379] | +0.005 [-0.030, +0.038] | NOT DETECTED | +0.014 [-0.018, +0.044] | +0.035 [-0.001, +0.071] | NOT DETECTED |
| X60T | 6 | 0.322 [0.267, 0.376] | 0.319 [0.262, 0.379] | +0.003 [-0.038, +0.043] | NOT DETECTED | +0.010 [-0.029, +0.048] | +0.033 [+0.008, +0.060] | BETTER |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | -1.045 [-1.143, -0.956] | -0.998 [-1.086, -0.918] | -0.047 [-0.111, +0.018] | NOT DETECTED | +0.043 [-0.000, +0.087] | +0.011 [+0.002, +0.019] | BETTER |
| C0 | 5 | -1.056 [-1.152, -0.967] | -0.998 [-1.086, -0.918] | -0.057 [-0.121, +0.008] | NOT DETECTED | +0.037 [-0.005, +0.081] | (reference) |  |
| N1P | 5 | -1.126 [-1.222, -1.038] | -0.998 [-1.086, -0.918] | -0.128 [-0.188, -0.068] | WORSE | -0.008 [-0.045, +0.029] | -0.070 [-0.098, -0.043] | WORSE |
| X60T | 6 | -1.002 [-1.105, -0.910] | -0.998 [-1.086, -0.918] | -0.004 [-0.062, +0.053] | NOT DETECTED | +0.070 [+0.034, +0.106] | +0.054 [+0.020, +0.086] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 26443 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | 0.572 [0.540, 0.602] | 0.566 [0.529, 0.601] | +0.006 [-0.023, +0.036] | NOT DETECTED | +0.013 [-0.013, +0.039] | +0.006 [-0.002, +0.015] | NOT DETECTED |
| C0 | 5 | 0.566 [0.535, 0.596] | 0.566 [0.529, 0.601] | -0.000 [-0.028, +0.030] | NOT DETECTED | +0.014 [-0.010, +0.039] | (reference) |  |
| N1P | 5 | 0.537 [0.509, 0.564] | 0.566 [0.529, 0.601] | -0.029 [-0.054, -0.004] | WORSE | -0.015 [-0.036, +0.006] | -0.029 [-0.045, -0.013] | WORSE |
| X60T | 6 | 0.559 [0.529, 0.588] | 0.566 [0.529, 0.601] | -0.007 [-0.033, +0.019] | NOT DETECTED | +0.002 [-0.019, +0.023] | -0.007 [-0.021, +0.007] | NOT DETECTED |

**items_logp** (unstated item of a revealed mon: log probability; 8013 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | -0.745 [-1.074, -0.480] | -0.738 [-0.934, -0.560] | -0.006 [-0.224, +0.178] | NOT DETECTED | +0.149 [+0.068, +0.237] | +0.005 [-0.009, +0.018] | NOT DETECTED |
| C0 | 5 | -0.749 [-1.084, -0.481] | -0.738 [-0.934, -0.560] | -0.011 [-0.234, +0.179] | NOT DETECTED | +0.149 [+0.067, +0.238] | (reference) |  |
| N1P | 5 | -0.588 [-0.751, -0.449] | -0.738 [-0.934, -0.560] | +0.151 [+0.063, +0.243] | BETTER | +0.144 [+0.073, +0.221] | +0.162 [-0.002, +0.368] | NOT DETECTED |
| X60T | 6 | -0.591 [-0.740, -0.455] | -0.738 [-0.934, -0.560] | +0.147 [+0.059, +0.244] | BETTER | +0.147 [+0.074, +0.226] | +0.158 [-0.017, +0.392] | NOT DETECTED |

**items_top1** (unstated item of a revealed mon: top-1 rate; 8013 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | 0.787 [0.733, 0.840] | 0.734 [0.657, 0.807] | +0.054 [-0.009, +0.118] | NOT DETECTED | +0.051 [-0.009, +0.111] | -0.004 [-0.008, +0.000] | NOT DETECTED |
| C0 | 5 | 0.791 [0.737, 0.843] | 0.734 [0.657, 0.807] | +0.057 [-0.006, +0.122] | NOT DETECTED | +0.064 [+0.009, +0.122] | (reference) |  |
| N1P | 5 | 0.785 [0.727, 0.840] | 0.734 [0.657, 0.807] | +0.051 [-0.007, +0.110] | NOT DETECTED | +0.059 [+0.012, +0.108] | -0.006 [-0.030, +0.021] | NOT DETECTED |
| X60T | 6 | 0.781 [0.722, 0.837] | 0.734 [0.657, 0.807] | +0.048 [-0.001, +0.097] | NOT DETECTED | +0.056 [+0.017, +0.099] | -0.010 [-0.039, +0.023] | NOT DETECTED |

Observation-semantics rows between a checkpoint's recorded config and the bank's encoding (scored through them, never trained on them): PSTAR: [[162, 163, 165, 166, 167, 168]]; C0: [[162, 163, 165, 166, 167, 168]]; N1P: [[162, 163, 165, 166, 167, 168]]; X60T: [[162, 163, 165, 166, 167, 168]]

Units not scored / flagged (first arm): moves_prec_near_tie 56, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0

### p72 — OFF-POOL (72 teams, 144 games, 6643 rows, bank af668a3f65fb)

**species_logp** (hidden species: mean log presence of the true species; 3918 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | -10.689 [-12.023, -9.389] | -3.030 [-3.283, -2.797] | -7.659 [-8.841, -6.507] | WORSE | -0.793 [-0.875, -0.704] | +0.145 [+0.057, +0.234] | BETTER |
| C0 | 5 | -10.834 [-12.191, -9.521] | -3.030 [-3.283, -2.797] | -7.804 [-9.005, -6.630] | WORSE | -0.803 [-0.885, -0.716] | (reference) |  |
| N1P | 5 | -3.165 [-3.431, -2.918] | -3.030 [-3.283, -2.797] | -0.135 [-0.208, -0.066] | WORSE | -0.054 [-0.100, -0.009] | +7.669 [+6.521, +8.845] | BETTER |
| X60T | 6 | -3.870 [-4.206, -3.551] | -3.030 [-3.283, -2.797] | -0.840 [-1.014, -0.665] | WORSE | -0.334 [-0.408, -0.257] | +6.964 [+5.905, +8.052] | BETTER |

**species_hit** (hidden species: share of the true species in the top k; 3918 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | 0.136 [0.109, 0.165] | 0.180 [0.143, 0.221] | -0.044 [-0.079, -0.013] | WORSE | -0.040 [-0.074, -0.009] | +0.004 [-0.002, +0.010] | NOT DETECTED |
| C0 | 5 | 0.132 [0.106, 0.160] | 0.180 [0.143, 0.221] | -0.048 [-0.083, -0.017] | WORSE | -0.043 [-0.078, -0.012] | (reference) |  |
| N1P | 5 | 0.166 [0.134, 0.202] | 0.180 [0.143, 0.221] | -0.014 [-0.041, +0.012] | NOT DETECTED | -0.012 [-0.038, +0.012] | +0.034 [+0.012, +0.060] | BETTER |
| X60T | 6 | 0.152 [0.122, 0.186] | 0.180 [0.143, 0.221] | -0.028 [-0.061, +0.003] | NOT DETECTED | -0.022 [-0.053, +0.006] | +0.021 [+0.005, +0.037] | BETTER |

**moves_logp** (unshown moves of a revealed mon: mean log probability; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | -1.354 [-1.444, -1.262] | -1.015 [-1.076, -0.951] | -0.339 [-0.396, -0.281] | WORSE | -0.132 [-0.167, -0.096] | +0.002 [-0.008, +0.013] | NOT DETECTED |
| C0 | 5 | -1.356 [-1.449, -1.264] | -1.015 [-1.076, -0.951] | -0.341 [-0.398, -0.285] | WORSE | -0.130 [-0.165, -0.095] | (reference) |  |
| N1P | 5 | -1.380 [-1.463, -1.296] | -1.015 [-1.076, -0.951] | -0.366 [-0.422, -0.307] | WORSE | -0.152 [-0.185, -0.118] | -0.024 [-0.060, +0.011] | NOT DETECTED |
| X60T | 6 | -1.269 [-1.356, -1.184] | -1.015 [-1.076, -0.951] | -0.255 [-0.304, -0.203] | WORSE | -0.092 [-0.123, -0.061] | +0.086 [+0.054, +0.119] | BETTER |

**moves_prec** (unshown moves of a revealed mon: precision at |unshown|; 26966 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | 0.416 [0.388, 0.444] | 0.486 [0.450, 0.521] | -0.070 [-0.093, -0.046] | WORSE | -0.045 [-0.065, -0.024] | -0.001 [-0.009, +0.007] | NOT DETECTED |
| C0 | 5 | 0.417 [0.388, 0.445] | 0.486 [0.450, 0.521] | -0.069 [-0.091, -0.047] | WORSE | -0.046 [-0.066, -0.026] | (reference) |  |
| N1P | 5 | 0.428 [0.399, 0.456] | 0.486 [0.450, 0.521] | -0.058 [-0.082, -0.035] | WORSE | -0.035 [-0.055, -0.014] | +0.011 [-0.001, +0.023] | NOT DETECTED |
| X60T | 6 | 0.439 [0.409, 0.468] | 0.486 [0.450, 0.521] | -0.047 [-0.070, -0.024] | WORSE | -0.033 [-0.053, -0.013] | +0.022 [+0.008, +0.036] | BETTER |

**items_logp** (unstated item of a revealed mon: log probability; 9437 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | -1.020 [-1.338, -0.739] | -0.634 [-0.758, -0.517] | -0.386 [-0.634, -0.167] | WORSE | -0.029 [-0.108, +0.050] | +0.010 [-0.012, +0.032] | NOT DETECTED |
| C0 | 5 | -1.030 [-1.350, -0.750] | -0.634 [-0.758, -0.517] | -0.395 [-0.643, -0.175] | WORSE | -0.033 [-0.109, +0.044] | (reference) |  |
| N1P | 5 | -0.541 [-0.660, -0.432] | -0.634 [-0.758, -0.517] | +0.094 [+0.034, +0.160] | BETTER | +0.094 [+0.045, +0.148] | +0.489 [+0.289, +0.721] | BETTER |
| X60T | 6 | -0.542 [-0.652, -0.440] | -0.634 [-0.758, -0.517] | +0.093 [+0.029, +0.161] | BETTER | +0.097 [+0.049, +0.153] | +0.488 [+0.290, +0.723] | BETTER |

**items_top1** (unstated item of a revealed mon: top-1 rate; 9437 units, higher is better)

| arm | ckpts | belief | Smogon prior | belief − prior |  | mix0.25 − prior | belief − C0 |  |
|---|---|---|---|---|---|---|---|---|
| PSTAR | 1 | 0.770 [0.716, 0.820] | 0.735 [0.670, 0.797] | +0.035 [-0.016, +0.085] | NOT DETECTED | +0.035 [-0.015, +0.082] | +0.006 [-0.002, +0.015] | NOT DETECTED |
| C0 | 5 | 0.764 [0.708, 0.816] | 0.735 [0.670, 0.797] | +0.029 [-0.020, +0.076] | NOT DETECTED | +0.028 [-0.019, +0.074] | (reference) |  |
| N1P | 5 | 0.801 [0.752, 0.846] | 0.735 [0.670, 0.797] | +0.066 [+0.023, +0.109] | BETTER | +0.066 [+0.027, +0.106] | +0.037 [+0.011, +0.065] | BETTER |
| X60T | 6 | 0.799 [0.750, 0.844] | 0.735 [0.670, 0.797] | +0.064 [+0.021, +0.107] | BETTER | +0.057 [+0.020, +0.094] | +0.035 [+0.014, +0.059] | BETTER |

Observation-semantics rows between a checkpoint's recorded config and the bank's encoding (scored through them, never trained on them): PSTAR: [[162, 163, 165, 166, 167, 168]]; C0: [[162, 163, 165, 166, 167, 168]]; N1P: [[162, 163, 165, 166, 167, 168]]; X60T: [[162, 163, 165, 166, 167, 168]]

Units not scored / flagged (first arm): moves_prec_near_tie 0, moves_truth_illegal 0, species_hit_near_tie 0, species_off_support 0
