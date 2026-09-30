# Gate ④ at scale — healthy sharpening vs starvation, by category

## Continuation: K2 final — S64

1600 turns branched, 949 decisive; 3.466 near-best actions per turn. Decisive turns with a near-best action, by category: attack 410, status 195, setup 98, hazard 52, recovery 75, switch 644.

**Levels** — dominated mass (turns with a dominated action of the category) / starved share (decisive turns with a near-best action of it):

| policy | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M | 0.350 / 0.334 | 0.288 / 0.185 | 0.155 / 0.139 | 0.126 / 0.133 | 0.253 / 0.115 | 0.146 / 0.027 | 0.131 / 0.373 |
| C_fix@83.1M | 0.336 / 0.478 | 0.295 / 0.241 | 0.145 / 0.210 | 0.143 / 0.184 | 0.278 / 0.115 | 0.115 / 0.080 | 0.111 / 0.548 |
| K2@91.1M | 0.332 / 0.513 | 0.295 / 0.261 | 0.139 / 0.251 | 0.129 / 0.184 | 0.266 / 0.173 | 0.126 / 0.093 | 0.110 / 0.582 |
| K3@99.2M | 0.331 / 0.487 | 0.298 / 0.244 | 0.133 / 0.226 | 0.124 / 0.225 | 0.319 / 0.135 | 0.130 / 0.133 | 0.106 / 0.557 |

**Paired steps** (Δ dominated mass ; Δ starved share; (+) / (−) = the 95 % battle-clustered interval excludes 0, (0) = it does not):

| step | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M → C_fix@83.1M | -0.014 (−) ; +0.144 (+) | +0.007 (0) ; +0.056 (+) | -0.010 (0) ; +0.072 (+) | +0.018 (0) ; +0.051 (0) | +0.025 (0) ; +0.000 (0) | -0.031 (−) ; +0.053 (0) | -0.019 (−) ; +0.175 (+) |
| C_fix@83.1M → K2@91.1M | -0.004 (0) ; +0.035 (+) | +0.001 (0) ; +0.019 (0) | -0.006 (0) ; +0.041 (+) | -0.014 (−) ; +0.000 (0) | -0.012 (0) ; +0.058 (0) | +0.011 (0) ; +0.013 (0) | -0.002 (0) ; +0.034 (+) |
| K2@91.1M → K3@99.2M | -0.001 (0) ; -0.026 (−) | +0.003 (0) ; -0.017 (0) | -0.006 (0) ; -0.026 (0) | -0.005 (0) ; +0.041 (0) | +0.052 (+) ; -0.038 (0) | +0.003 (0) ; +0.040 (0) | -0.003 (0) ; -0.025 (−) |
| N0@75.0M → K3@99.2M | -0.019 (−) ; +0.153 (+) | +0.010 (0) ; +0.059 (+) | -0.022 (0) ; +0.087 (+) | -0.002 (0) ; +0.092 (0) | +0.066 (0) ; +0.019 (0) | -0.017 (0) ; +0.107 (+) | -0.024 (−) ; +0.185 (+) |

**Strict starvation** (the starved action is near-best even at its upper bound, gap + 1.96·SE ≤ ε) — Δ per step:

| step | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M → C_fix@83.1M | +0.115 (+) n=949 | +0.042 (+) n=283 | +0.082 (+) n=110 | +0.032 (0) n=62 | +0.087 (0) n=23 | +0.065 (0) n=31 | +0.156 (+) n=540 |
| C_fix@83.1M → K2@91.1M | +0.035 (+) n=949 | +0.007 (0) n=283 | +0.073 (+) n=110 | +0.000 (0) n=62 | +0.000 (0) n=23 | -0.032 (0) n=31 | +0.043 (+) n=540 |
| K2@91.1M → K3@99.2M | -0.023 (−) n=949 | -0.025 (0) n=283 | -0.054 (−) n=110 | +0.032 (0) n=62 | +0.000 (0) n=23 | +0.097 (0) n=31 | -0.028 (−) n=540 |
| N0@75.0M → K3@99.2M | +0.126 (+) n=949 | +0.025 (0) n=283 | +0.100 (+) n=110 | +0.065 (0) n=62 | +0.087 (0) n=23 | +0.129 (+) n=31 | +0.170 (+) n=540 |

## Continuation: K2 final — S16(nested)

1600 turns branched, 740 decisive; 2.877 near-best actions per turn. Decisive turns with a near-best action, by category: attack 271, status 108, setup 49, hazard 32, recovery 33, switch 440.

**Levels** — dominated mass (turns with a dominated action of the category) / starved share (decisive turns with a near-best action of it):

| policy | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M | 0.267 / 0.288 | 0.246 / 0.181 | 0.156 / 0.130 | 0.139 / 0.225 | 0.272 / 0.156 | 0.152 / 0.030 | 0.100 / 0.321 |
| C_fix@83.1M | 0.257 / 0.412 | 0.257 / 0.247 | 0.130 / 0.213 | 0.153 / 0.245 | 0.289 / 0.156 | 0.132 / 0.091 | 0.085 / 0.495 |
| K2@91.1M | 0.253 / 0.445 | 0.255 / 0.251 | 0.129 / 0.269 | 0.133 / 0.225 | 0.297 / 0.250 | 0.149 / 0.121 | 0.084 / 0.525 |
| K3@99.2M | 0.258 / 0.427 | 0.258 / 0.243 | 0.115 / 0.241 | 0.134 / 0.245 | 0.386 / 0.188 | 0.155 / 0.091 | 0.087 / 0.509 |

**Paired steps** (Δ dominated mass ; Δ starved share; (+) / (−) = the 95 % battle-clustered interval excludes 0, (0) = it does not):

| step | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M → C_fix@83.1M | -0.010 (−) ; +0.124 (+) | +0.010 (0) ; +0.066 (+) | -0.026 (−) ; +0.083 (+) | +0.015 (0) ; +0.020 (0) | +0.017 (0) ; +0.000 (0) | -0.019 (0) ; +0.061 (0) | -0.015 (−) ; +0.175 (+) |
| C_fix@83.1M → K2@91.1M | -0.004 (0) ; +0.032 (+) | -0.002 (0) ; +0.004 (0) | -0.001 (0) ; +0.056 (0) | -0.020 (−) ; -0.020 (0) | +0.008 (0) ; +0.094 (0) | +0.017 (0) ; +0.030 (0) | -0.001 (0) ; +0.029 (+) |
| K2@91.1M → K3@99.2M | +0.004 (0) ; -0.018 (0) | +0.003 (0) ; -0.007 (0) | -0.013 (−) ; -0.028 (0) | +0.001 (0) ; +0.020 (0) | +0.089 (+) ; -0.062 (0) | +0.006 (0) ; -0.030 (0) | +0.003 (0) ; -0.016 (0) |
| N0@75.0M → K3@99.2M | -0.009 (0) ; +0.139 (+) | +0.012 (0) ; +0.063 (+) | -0.041 (−) ; +0.111 (+) | -0.004 (0) ; +0.020 (0) | +0.114 (+) ; +0.031 (0) | +0.003 (0) ; +0.061 (0) | -0.013 (−) ; +0.189 (+) |

**Strict starvation** (the starved action is near-best even at its upper bound, gap + 1.96·SE ≤ ε) — Δ per step:

| step | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M → C_fix@83.1M | +0.111 (+) n=740 | +0.039 (+) n=231 | +0.080 (+) n=88 | +0.044 (0) n=45 | +0.000 (0) n=24 | +0.000 (0) n=23 | +0.159 (+) n=435 |
| C_fix@83.1M → K2@91.1M | +0.028 (+) n=740 | +0.009 (0) n=231 | +0.034 (0) n=88 | -0.022 (0) n=45 | +0.083 (0) n=24 | +0.043 (0) n=23 | +0.030 (+) n=435 |
| K2@91.1M → K3@99.2M | -0.013 (0) n=740 | -0.009 (0) n=231 | -0.034 (0) n=88 | +0.022 (0) n=45 | -0.042 (0) n=24 | -0.043 (0) n=23 | -0.011 (0) n=435 |
| N0@75.0M → K3@99.2M | +0.126 (+) n=740 | +0.039 (0) n=231 | +0.080 (+) n=88 | +0.044 (0) n=45 | +0.042 (0) n=24 | +0.000 (0) n=23 | +0.177 (+) n=435 |

## Continuation: N0 final — S64

1600 turns branched, 951 decisive; 3.441 near-best actions per turn. Decisive turns with a near-best action, by category: attack 395, status 182, setup 86, hazard 36, recovery 50, switch 653.

**Levels** — dominated mass (turns with a dominated action of the category) / starved share (decisive turns with a near-best action of it):

| policy | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M | 0.379 / 0.348 | 0.313 / 0.182 | 0.181 / 0.187 | 0.125 / 0.140 | 0.292 / 0.167 | 0.176 / 0.040 | 0.134 / 0.378 |
| C_fix@83.1M | 0.367 / 0.470 | 0.323 / 0.248 | 0.177 / 0.231 | 0.120 / 0.233 | 0.300 / 0.139 | 0.128 / 0.160 | 0.119 / 0.514 |
| K2@91.1M | 0.360 / 0.506 | 0.320 / 0.266 | 0.160 / 0.264 | 0.109 / 0.209 | 0.283 / 0.139 | 0.155 / 0.100 | 0.119 / 0.562 |
| K3@99.2M | 0.360 / 0.485 | 0.319 / 0.240 | 0.165 / 0.253 | 0.102 / 0.267 | 0.325 / 0.167 | 0.144 / 0.140 | 0.118 / 0.531 |

**Paired steps** (Δ dominated mass ; Δ starved share; (+) / (−) = the 95 % battle-clustered interval excludes 0, (0) = it does not):

| step | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M → C_fix@83.1M | -0.012 (−) ; +0.122 (+) | +0.010 (0) ; +0.066 (+) | -0.004 (0) ; +0.044 (+) | -0.005 (0) ; +0.093 (+) | +0.008 (0) ; -0.028 (0) | -0.048 (−) ; +0.120 (+) | -0.015 (−) ; +0.136 (+) |
| C_fix@83.1M → K2@91.1M | -0.006 (−) ; +0.036 (+) | -0.003 (0) ; +0.018 (0) | -0.018 (−) ; +0.033 (0) | -0.010 (0) ; -0.023 (0) | -0.017 (0) ; +0.000 (0) | +0.026 (+) ; -0.060 (0) | -0.001 (0) ; +0.048 (+) |
| K2@91.1M → K3@99.2M | +0.000 (0) ; -0.021 (−) | -0.001 (0) ; -0.025 (−) | +0.005 (0) ; -0.011 (0) | -0.007 (0) ; +0.058 (0) | +0.041 (+) ; +0.028 (0) | -0.011 (0) ; +0.040 (0) | -0.001 (0) ; -0.031 (−) |
| N0@75.0M → K3@99.2M | -0.018 (−) ; +0.137 (+) | +0.006 (0) ; +0.058 (+) | -0.017 (0) ; +0.066 (+) | -0.022 (0) ; +0.128 (+) | +0.032 (0) ; +0.000 (0) | -0.032 (0) ; +0.100 (0) | -0.016 (−) ; +0.153 (+) |

**Strict starvation** (the starved action is near-best even at its upper bound, gap + 1.96·SE ≤ ε) — Δ per step:

| step | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M → C_fix@83.1M | +0.101 (+) n=951 | +0.039 (+) n=281 | +0.045 (0) n=112 | +0.047 (0) n=64 | -0.038 (0) n=26 | +0.125 (+) n=32 | +0.138 (+) n=565 |
| C_fix@83.1M → K2@91.1M | +0.034 (+) n=951 | +0.025 (0) n=281 | +0.036 (0) n=112 | +0.000 (0) n=64 | +0.038 (0) n=26 | -0.094 (0) n=32 | +0.041 (+) n=565 |
| K2@91.1M → K3@99.2M | -0.011 (0) n=951 | -0.007 (0) n=281 | -0.009 (0) n=112 | +0.078 (0) n=64 | +0.000 (0) n=26 | +0.062 (0) n=32 | -0.025 (−) n=565 |
| N0@75.0M → K3@99.2M | +0.124 (+) n=951 | +0.057 (+) n=281 | +0.071 (+) n=112 | +0.125 (+) n=64 | +0.000 (0) n=26 | +0.094 (0) n=32 | +0.154 (+) n=565 |

## Continuation: N0 final — S16(nested)

1600 turns branched, 754 decisive; 2.899 near-best actions per turn. Decisive turns with a near-best action, by category: attack 278, status 111, setup 49, hazard 23, recovery 30, switch 476.

**Levels** — dominated mass (turns with a dominated action of the category) / starved share (decisive turns with a near-best action of it):

| policy | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M | 0.303 / 0.308 | 0.271 / 0.183 | 0.185 / 0.162 | 0.121 / 0.225 | 0.274 / 0.130 | 0.218 / 0.000 | 0.119 / 0.355 |
| C_fix@83.1M | 0.295 / 0.416 | 0.286 / 0.230 | 0.187 / 0.234 | 0.129 / 0.245 | 0.282 / 0.130 | 0.167 / 0.133 | 0.102 / 0.506 |
| K2@91.1M | 0.292 / 0.448 | 0.287 / 0.252 | 0.171 / 0.234 | 0.109 / 0.225 | 0.262 / 0.130 | 0.190 / 0.100 | 0.103 / 0.546 |
| K3@99.2M | 0.293 / 0.435 | 0.287 / 0.241 | 0.172 / 0.234 | 0.107 / 0.265 | 0.301 / 0.130 | 0.188 / 0.200 | 0.102 / 0.519 |

**Paired steps** (Δ dominated mass ; Δ starved share; (+) / (−) = the 95 % battle-clustered interval excludes 0, (0) = it does not):

| step | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M → C_fix@83.1M | -0.008 (0) ; +0.109 (+) | +0.015 (0) ; +0.047 (+) | +0.003 (0) ; +0.072 (+) | +0.008 (0) ; +0.020 (0) | +0.008 (0) ; +0.000 (0) | -0.051 (−) ; +0.133 (+) | -0.017 (−) ; +0.151 (+) |
| C_fix@83.1M → K2@91.1M | -0.003 (0) ; +0.032 (+) | +0.001 (0) ; +0.022 (0) | -0.017 (−) ; +0.000 (0) | -0.020 (−) ; -0.020 (0) | -0.020 (0) ; +0.000 (0) | +0.022 (+) ; -0.033 (0) | +0.002 (0) ; +0.040 (+) |
| K2@91.1M → K3@99.2M | +0.001 (0) ; -0.013 (0) | +0.000 (0) ; -0.011 (0) | +0.001 (0) ; +0.000 (0) | -0.002 (0) ; +0.041 (0) | +0.038 (0) ; +0.000 (0) | -0.002 (0) ; +0.100 (0) | -0.001 (0) ; -0.027 (0) |
| N0@75.0M → K3@99.2M | -0.010 (−) ; +0.127 (+) | +0.016 (0) ; +0.058 (+) | -0.013 (0) ; +0.072 (+) | -0.014 (0) ; +0.041 (0) | +0.027 (0) ; +0.000 (0) | -0.030 (0) ; +0.200 (+) | -0.017 (−) ; +0.164 (+) |

**Strict starvation** (the starved action is near-best even at its upper bound, gap + 1.96·SE ≤ ε) — Δ per step:

| step | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M → C_fix@83.1M | +0.111 (+) n=754 | +0.022 (0) n=230 | +0.062 (0) n=97 | +0.023 (0) n=43 | +0.059 (0) n=17 | +0.160 (+) n=25 | +0.154 (+) n=474 |
| C_fix@83.1M → K2@91.1M | +0.028 (+) n=754 | +0.030 (0) n=230 | +0.000 (0) n=97 | +0.000 (0) n=43 | +0.000 (0) n=17 | -0.080 (0) n=25 | +0.032 (+) n=474 |
| K2@91.1M → K3@99.2M | -0.013 (0) n=754 | -0.013 (0) n=230 | +0.010 (0) n=97 | +0.046 (0) n=43 | +0.000 (0) n=17 | +0.120 (0) n=25 | -0.025 (−) n=474 |
| N0@75.0M → K3@99.2M | +0.126 (+) n=754 | +0.039 (+) n=230 | +0.072 (+) n=97 | +0.070 (0) n=43 | +0.059 (0) n=17 | +0.200 (+) n=25 | +0.160 (+) n=474 |

## Continuation: C_fix final — S64

1600 turns branched, 949 decisive; 3.449 near-best actions per turn. Decisive turns with a near-best action, by category: attack 412, status 210, setup 87, hazard 49, recovery 66, switch 629.

**Levels** — dominated mass (turns with a dominated action of the category) / starved share (decisive turns with a near-best action of it):

| policy | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M | 0.345 / 0.311 | 0.269 / 0.158 | 0.163 / 0.095 | 0.154 / 0.126 | 0.269 / 0.102 | 0.150 / 0.061 | 0.134 / 0.353 |
| C_fix@83.1M | 0.332 / 0.473 | 0.276 / 0.218 | 0.149 / 0.200 | 0.167 / 0.230 | 0.268 / 0.102 | 0.116 / 0.106 | 0.119 / 0.533 |
| K2@91.1M | 0.327 / 0.500 | 0.275 / 0.245 | 0.142 / 0.238 | 0.150 / 0.264 | 0.267 / 0.102 | 0.133 / 0.106 | 0.117 / 0.550 |
| K3@99.2M | 0.328 / 0.475 | 0.281 / 0.221 | 0.135 / 0.224 | 0.146 / 0.253 | 0.306 / 0.102 | 0.130 / 0.076 | 0.116 / 0.536 |

**Paired steps** (Δ dominated mass ; Δ starved share; (+) / (−) = the 95 % battle-clustered interval excludes 0, (0) = it does not):

| step | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M → C_fix@83.1M | -0.013 (−) ; +0.162 (+) | +0.006 (0) ; +0.061 (+) | -0.015 (0) ; +0.105 (+) | +0.013 (0) ; +0.103 (+) | -0.001 (0) ; +0.000 (0) | -0.034 (−) ; +0.045 (+) | -0.015 (−) ; +0.180 (+) |
| C_fix@83.1M → K2@91.1M | -0.005 (0) ; +0.027 (+) | -0.001 (0) ; +0.027 (0) | -0.007 (0) ; +0.038 (0) | -0.018 (−) ; +0.035 (0) | -0.002 (0) ; +0.000 (0) | +0.017 (+) ; +0.000 (0) | -0.002 (0) ; +0.018 (0) |
| K2@91.1M → K3@99.2M | +0.002 (0) ; -0.025 (−) | +0.006 (0) ; -0.024 (−) | -0.007 (0) ; -0.014 (0) | -0.004 (0) ; -0.011 (0) | +0.039 (+) ; +0.000 (0) | -0.003 (0) ; -0.030 (0) | -0.001 (0) ; -0.014 (0) |
| N0@75.0M → K3@99.2M | -0.017 (−) ; +0.164 (+) | +0.012 (0) ; +0.063 (+) | -0.029 (−) ; +0.129 (+) | -0.009 (0) ; +0.126 (+) | +0.037 (0) ; +0.000 (0) | -0.020 (0) ; +0.015 (0) | -0.018 (−) ; +0.183 (+) |

**Strict starvation** (the starved action is near-best even at its upper bound, gap + 1.96·SE ≤ ε) — Δ per step:

| step | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M → C_fix@83.1M | +0.119 (+) n=949 | +0.070 (+) n=300 | +0.059 (+) n=118 | +0.083 (0) n=60 | +0.042 (0) n=24 | +0.057 (0) n=35 | +0.156 (+) n=514 |
| C_fix@83.1M → K2@91.1M | +0.024 (+) n=949 | +0.017 (0) n=300 | +0.051 (+) n=118 | +0.033 (0) n=60 | +0.000 (0) n=24 | +0.000 (0) n=35 | +0.021 (0) n=514 |
| K2@91.1M → K3@99.2M | -0.025 (−) n=949 | -0.027 (−) n=300 | -0.034 (0) n=118 | +0.000 (0) n=60 | -0.042 (0) n=24 | -0.029 (0) n=35 | -0.021 (0) n=514 |
| N0@75.0M → K3@99.2M | +0.118 (+) n=949 | +0.060 (+) n=300 | +0.076 (+) n=118 | +0.117 (+) n=60 | +0.000 (0) n=24 | +0.029 (0) n=35 | +0.156 (+) n=514 |

## Continuation: C_fix final — S16(nested)

1600 turns branched, 731 decisive; 2.884 near-best actions per turn. Decisive turns with a near-best action, by category: attack 266, status 107, setup 49, hazard 25, recovery 37, switch 426.

**Levels** — dominated mass (turns with a dominated action of the category) / starved share (decisive turns with a near-best action of it):

| policy | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M | 0.277 / 0.280 | 0.247 / 0.188 | 0.160 / 0.075 | 0.177 / 0.143 | 0.311 / 0.120 | 0.166 / 0.081 | 0.112 / 0.359 |
| C_fix@83.1M | 0.265 / 0.429 | 0.246 / 0.256 | 0.167 / 0.168 | 0.204 / 0.184 | 0.299 / 0.160 | 0.132 / 0.216 | 0.096 / 0.547 |
| K2@91.1M | 0.257 / 0.461 | 0.239 / 0.274 | 0.164 / 0.196 | 0.183 / 0.245 | 0.276 / 0.120 | 0.148 / 0.243 | 0.093 / 0.570 |
| K3@99.2M | 0.258 / 0.442 | 0.239 / 0.248 | 0.159 / 0.187 | 0.184 / 0.225 | 0.345 / 0.120 | 0.148 / 0.189 | 0.092 / 0.559 |

**Paired steps** (Δ dominated mass ; Δ starved share; (+) / (−) = the 95 % battle-clustered interval excludes 0, (0) = it does not):

| step | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M → C_fix@83.1M | -0.011 (−) ; +0.149 (+) | -0.001 (0) ; +0.068 (+) | +0.008 (0) ; +0.093 (+) | +0.028 (0) ; +0.041 (0) | -0.012 (0) ; +0.040 (0) | -0.034 (0) ; +0.135 (+) | -0.016 (−) ; +0.188 (+) |
| C_fix@83.1M → K2@91.1M | -0.009 (−) ; +0.032 (+) | -0.006 (0) ; +0.019 (0) | -0.004 (0) ; +0.028 (0) | -0.022 (−) ; +0.061 (0) | -0.024 (0) ; -0.040 (0) | +0.016 (0) ; +0.027 (0) | -0.003 (0) ; +0.024 (0) |
| K2@91.1M → K3@99.2M | +0.001 (0) ; -0.019 (0) | -0.000 (0) ; -0.026 (0) | -0.005 (0) ; -0.009 (0) | +0.001 (0) ; -0.020 (0) | +0.070 (+) ; +0.000 (0) | +0.000 (0) ; -0.054 (0) | -0.001 (0) ; -0.012 (0) |
| N0@75.0M → K3@99.2M | -0.019 (−) ; +0.161 (+) | -0.008 (0) ; +0.060 (+) | -0.001 (0) ; +0.112 (+) | +0.007 (0) ; +0.082 (0) | +0.034 (0) ; +0.000 (0) | -0.018 (0) ; +0.108 (0) | -0.020 (−) ; +0.200 (+) |

**Strict starvation** (the starved action is near-best even at its upper bound, gap + 1.96·SE ≤ ε) — Δ per step:

| step | all | attack | status | setup | hazard | recovery | switch |
|---|---|---|---|---|---|---|---|
| N0@75.0M → C_fix@83.1M | +0.140 (+) n=731 | +0.076 (+) n=236 | +0.081 (+) n=86 | +0.049 (0) n=41 | +0.053 (0) n=19 | +0.133 (+) n=30 | +0.177 (+) n=418 |
| C_fix@83.1M → K2@91.1M | +0.027 (+) n=731 | +0.017 (0) n=236 | +0.035 (0) n=86 | +0.049 (0) n=41 | -0.053 (0) n=19 | +0.033 (0) n=30 | +0.024 (0) n=418 |
| K2@91.1M → K3@99.2M | -0.018 (0) n=731 | -0.021 (0) n=236 | +0.012 (0) n=86 | -0.049 (0) n=41 | +0.000 (0) n=19 | -0.033 (0) n=30 | -0.014 (0) n=418 |
| N0@75.0M → K3@99.2M | +0.149 (+) n=731 | +0.072 (+) n=236 | +0.128 (+) n=86 | +0.049 (0) n=41 | +0.000 (0) n=19 | +0.133 (+) n=30 | +0.187 (+) n=418 |

## Verdict agreement across continuations (+ / − = interval excludes 0; 0 = straddles)

| step \| category \| measure | K2 final | N0 final | C_fix final |
|---|---|---|---|
| N0@75.0M → C_fix@83.1M | all | dominated_mass | − | − | − |
| N0@75.0M → C_fix@83.1M | all | starved | + | + | + |
| N0@75.0M → C_fix@83.1M | all | starved_strict | + | + | + |
| N0@75.0M → C_fix@83.1M | attack | dominated_mass | 0 | 0 | 0 |
| N0@75.0M → C_fix@83.1M | attack | starved | + | + | + |
| N0@75.0M → C_fix@83.1M | attack | starved_strict | + | + | + |
| N0@75.0M → C_fix@83.1M | status | dominated_mass | 0 | 0 | 0 |
| N0@75.0M → C_fix@83.1M | status | starved | + | + | + |
| N0@75.0M → C_fix@83.1M | status | starved_strict | + | 0 | + |
| N0@75.0M → C_fix@83.1M | setup | dominated_mass | 0 | 0 | 0 |
| N0@75.0M → C_fix@83.1M | setup | starved | 0 | + | + |
| N0@75.0M → C_fix@83.1M | setup | starved_strict | 0 | 0 | 0 |
| N0@75.0M → C_fix@83.1M | hazard | dominated_mass | 0 | 0 | 0 |
| N0@75.0M → C_fix@83.1M | hazard | starved | 0 | 0 | 0 |
| N0@75.0M → C_fix@83.1M | hazard | starved_strict | 0 | 0 | 0 |
| N0@75.0M → C_fix@83.1M | recovery | dominated_mass | − | − | − |
| N0@75.0M → C_fix@83.1M | recovery | starved | 0 | + | + |
| N0@75.0M → C_fix@83.1M | recovery | starved_strict | 0 | + | 0 |
| N0@75.0M → C_fix@83.1M | switch | dominated_mass | − | − | − |
| N0@75.0M → C_fix@83.1M | switch | starved | + | + | + |
| N0@75.0M → C_fix@83.1M | switch | starved_strict | + | + | + |
| C_fix@83.1M → K2@91.1M | all | dominated_mass | 0 | − | 0 |
| C_fix@83.1M → K2@91.1M | all | starved | + | + | + |
| C_fix@83.1M → K2@91.1M | all | starved_strict | + | + | + |
| C_fix@83.1M → K2@91.1M | attack | dominated_mass | 0 | 0 | 0 |
| C_fix@83.1M → K2@91.1M | attack | starved | 0 | 0 | 0 |
| C_fix@83.1M → K2@91.1M | attack | starved_strict | 0 | 0 | 0 |
| C_fix@83.1M → K2@91.1M | status | dominated_mass | 0 | − | 0 |
| C_fix@83.1M → K2@91.1M | status | starved | + | 0 | 0 |
| C_fix@83.1M → K2@91.1M | status | starved_strict | + | 0 | + |
| C_fix@83.1M → K2@91.1M | setup | dominated_mass | − | 0 | − |
| C_fix@83.1M → K2@91.1M | setup | starved | 0 | 0 | 0 |
| C_fix@83.1M → K2@91.1M | setup | starved_strict | 0 | 0 | 0 |
| C_fix@83.1M → K2@91.1M | hazard | dominated_mass | 0 | 0 | 0 |
| C_fix@83.1M → K2@91.1M | hazard | starved | 0 | 0 | 0 |
| C_fix@83.1M → K2@91.1M | hazard | starved_strict | 0 | 0 | 0 |
| C_fix@83.1M → K2@91.1M | recovery | dominated_mass | 0 | + | + |
| C_fix@83.1M → K2@91.1M | recovery | starved | 0 | 0 | 0 |
| C_fix@83.1M → K2@91.1M | recovery | starved_strict | 0 | 0 | 0 |
| C_fix@83.1M → K2@91.1M | switch | dominated_mass | 0 | 0 | 0 |
| C_fix@83.1M → K2@91.1M | switch | starved | + | + | 0 |
| C_fix@83.1M → K2@91.1M | switch | starved_strict | + | + | 0 |
| K2@91.1M → K3@99.2M | all | dominated_mass | 0 | 0 | 0 |
| K2@91.1M → K3@99.2M | all | starved | − | − | − |
| K2@91.1M → K3@99.2M | all | starved_strict | − | 0 | − |
| K2@91.1M → K3@99.2M | attack | dominated_mass | 0 | 0 | 0 |
| K2@91.1M → K3@99.2M | attack | starved | 0 | − | − |
| K2@91.1M → K3@99.2M | attack | starved_strict | 0 | 0 | − |
| K2@91.1M → K3@99.2M | status | dominated_mass | 0 | 0 | 0 |
| K2@91.1M → K3@99.2M | status | starved | 0 | 0 | 0 |
| K2@91.1M → K3@99.2M | status | starved_strict | − | 0 | 0 |
| K2@91.1M → K3@99.2M | setup | dominated_mass | 0 | 0 | 0 |
| K2@91.1M → K3@99.2M | setup | starved | 0 | 0 | 0 |
| K2@91.1M → K3@99.2M | setup | starved_strict | 0 | 0 | 0 |
| K2@91.1M → K3@99.2M | hazard | dominated_mass | + | + | + |
| K2@91.1M → K3@99.2M | hazard | starved | 0 | 0 | 0 |
| K2@91.1M → K3@99.2M | hazard | starved_strict | 0 | 0 | 0 |
| K2@91.1M → K3@99.2M | recovery | dominated_mass | 0 | 0 | 0 |
| K2@91.1M → K3@99.2M | recovery | starved | 0 | 0 | 0 |
| K2@91.1M → K3@99.2M | recovery | starved_strict | 0 | 0 | 0 |
| K2@91.1M → K3@99.2M | switch | dominated_mass | 0 | 0 | 0 |
| K2@91.1M → K3@99.2M | switch | starved | − | − | 0 |
| K2@91.1M → K3@99.2M | switch | starved_strict | − | − | 0 |
| N0@75.0M → K3@99.2M | all | dominated_mass | − | − | − |
| N0@75.0M → K3@99.2M | all | starved | + | + | + |
| N0@75.0M → K3@99.2M | all | starved_strict | + | + | + |
| N0@75.0M → K3@99.2M | attack | dominated_mass | 0 | 0 | 0 |
| N0@75.0M → K3@99.2M | attack | starved | + | + | + |
| N0@75.0M → K3@99.2M | attack | starved_strict | 0 | + | + |
| N0@75.0M → K3@99.2M | status | dominated_mass | 0 | 0 | − |
| N0@75.0M → K3@99.2M | status | starved | + | + | + |
| N0@75.0M → K3@99.2M | status | starved_strict | + | + | + |
| N0@75.0M → K3@99.2M | setup | dominated_mass | 0 | 0 | 0 |
| N0@75.0M → K3@99.2M | setup | starved | 0 | + | + |
| N0@75.0M → K3@99.2M | setup | starved_strict | 0 | + | + |
| N0@75.0M → K3@99.2M | hazard | dominated_mass | 0 | 0 | 0 |
| N0@75.0M → K3@99.2M | hazard | starved | 0 | 0 | 0 |
| N0@75.0M → K3@99.2M | hazard | starved_strict | 0 | 0 | 0 |
| N0@75.0M → K3@99.2M | recovery | dominated_mass | 0 | 0 | 0 |
| N0@75.0M → K3@99.2M | recovery | starved | + | 0 | 0 |
| N0@75.0M → K3@99.2M | recovery | starved_strict | + | 0 | 0 |
| N0@75.0M → K3@99.2M | switch | dominated_mass | − | − | − |
| N0@75.0M → K3@99.2M | switch | starved | + | + | + |
| N0@75.0M → K3@99.2M | switch | starved_strict | + | + | + |
