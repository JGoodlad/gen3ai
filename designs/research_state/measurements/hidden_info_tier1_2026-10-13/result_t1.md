# Tier 1 — the result (rendered from `result_t1.json`)

## R-MOVES (§12.3; registered 600 per checkpoint, pooled over end2 + prod, LEAF-K8)

**Verdict: MOVES-REFUTED (R3)**

G = IMP_O − IMP_F: +0.070 [+0.060, +0.081] (n 1097)

| condition | holds | read |
|---|---|---|
| B1 | False | R-GIGO PASS; R-REPLICATE NOT AVAILABLE (the independent implementation is compared by a third party, §13.1) |
| B2 | True | G_lower 0.05958723754382039 |
| B3 | False | s_M +0.130 [-0.015, +0.252] |
| B4 | False | n_M +0.416 [+0.308, +0.535] |
| B5 | False | classes_compared ['U']; n_U +0.822 [+0.697, +0.957]; not_in_tier1 ['I', 'A', 'S']; s_U +0.314 [+0.179, +0.434] |
| B6 | True | D90_minus_F +0.011 [+0.001, +0.021]; dose_means [0.00485, 0.01263, 0.01624, 0.01392]; note the slope grid holds the Tier-1 doses only (D25 / D75 are Tier 2); slope +0.010 [+0.001, +0.019]; slope_grid [0.0, 0.5, 0.9, 1.0] |
| B7 | True | D90_minus_W90 +0.025 [+0.013, +0.036]; W90_minus_F -0.013 [-0.024, -0.003] |
| B8 | False | end2 holds False (F+M_minus_F +0.008 [-0.006, +0.022], O_minus_O-M +0.028 [+0.017, +0.040]); prod holds False (F+M_minus_F +0.010 [-0.003, +0.024], O_minus_O-M +0.029 [+0.019, +0.039]) |

R1 False · R2 False · R3 True; NON-ADDITIVE False

### Shares (pooled)

| class | s (sufficiency) | n (necessity) | σ = IMP_F+X − IMP_F | ν = IMP_O − IMP_O−X |
|---|---|---|---|---|
| M | +0.130 [-0.015, +0.252] | +0.416 [+0.308, +0.535] | +0.009 [-0.001, +0.019] | +0.029 [+0.021, +0.038] |
| U | +0.314 [+0.179, +0.434] | +0.822 [+0.697, +0.957] | +0.022 [+0.012, +0.032] | +0.058 [+0.048, +0.068] |

Per checkpoint:

- end2: G +0.063 [+0.048, +0.080]; s_M +0.129 [-0.112, +0.337], n_M +0.447 [+0.265, +0.650]; s_U +0.243 [+0.017, +0.433], n_U +0.728 [+0.528, +0.941]
- prod: G +0.077 [+0.063, +0.092]; s_M +0.131 [-0.044, +0.285], n_M +0.391 [+0.268, +0.535]; s_U +0.372 [+0.209, +0.515], n_U +0.898 [+0.748, +1.079]

## R-RIVALS U: **U-SHARED** (s_U +0.314 [+0.179, +0.434], n_U +0.822 [+0.697, +0.957])

## R-RECOVER: **BELIEF-RECOVERABLE (on this distribution)**; PRIOR-MISMATCH False; HONEST-SHARPENING-HELPS False (MAP100 − F -0.002 [-0.014, +0.009]); p½ None

| arm | r = (IMP_X − IMP_F) / G |
|---|---|
| D50 | +0.113 [-0.038, +0.240] |
| D90 | +0.160 [+0.024, +0.287] |
| F+M | +0.130 [-0.015, +0.252] |
| MAP100 | -0.030 [-0.209, +0.120] |
| POOL | +0.799 [+0.729, +0.867] |
| POOLSETS | -0.020 [-0.187, +0.125] |

## R-EVAL: **not EVALUATOR-BOUND**; neither; INFORMATION-BOUND True (g_HE +0.028 [+0.018, +0.037], g_CRIT +0.037 [+0.027, +0.047], ratio +0.75 [+0.52, +1.06])

| worlds | HE-TRUNC-4-K8 − TRUNC-4-K8 | HE-1PLY − CRITIC |
|---|---|---|
| F | +0.007 [-0.001, +0.016] | -0.002 [-0.011, +0.006] |
| P | +0.003 [-0.005, +0.011] | -0.005 [-0.013, +0.004] |

## The stage rules (§12.13.4; all Tier-1S states)

- **R-STAGE-VOI k: TREND-DOWN**; slope per SD -0.021 [-0.030, -0.012]; G by bucket: 1-2 +0.087 [+0.066, +0.108]; 3 +0.093 [+0.071, +0.116]; 4 +0.076 [+0.052, +0.101]; 5 +0.061 [+0.042, +0.079]; 6 +0.025 [+0.010, +0.042]
- **R-STAGE-VOI material: TREND-UP**; slope per SD +0.017 [+0.008, +0.026]; G by bucket: 12 +0.094 [+0.072, +0.115]; 10-11 +0.071 [+0.054, +0.090]; 8-9 +0.081 [+0.060, +0.103]; <=7 +0.041 [+0.027, +0.058]
- **R-STAGE-VOI turn: TREND-DOWN**; slope per SD -0.022 [-0.030, -0.013]; G by bucket: <=7 +0.095 [+0.073, +0.116]; 8-14 +0.078 [+0.060, +0.096]; 15-21 +0.069 [+0.050, +0.089]; >=22 +0.033 [+0.019, +0.049]
- **R-STAGE-AXIS**: k does not carry (-0.010 [-0.023, +0.003]); material does not carry (+0.003 [-0.009, +0.014]); turn CARRIES (-0.014 [-0.024, -0.003])
- **R-STAGE-MECH k: NOT DETECTED**; Δc +0.360 [+0.306, +0.415], Δs +0.013 [-0.021, +0.045]; split Δc·s̄ +0.0394, c̄·Δs +0.0076
- **R-STAGE-MECH material: NOT DETECTED**; Δc +0.427 [+0.370, +0.484], Δs -0.017 [-0.060, +0.027]; split Δc·s̄ +0.0475, c̄·Δs -0.0095
- **R-STAGE-MECH turn: NOT DETECTED**; Δc +0.374 [+0.304, +0.443], Δs +0.042 [-0.002, +0.081]; split Δc·s̄ +0.0352, c̄·Δs +0.0266

### R-STAGE-CLASS (M vs U) and the two-point Shapley credit

| axis | bucket | n | G | σ_M | σ_U | ν_M | ν_U | φ_M | φ_U | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| k | 1-2 | 323 | +0.086 [+0.066, +0.108] | +0.009 [-0.011, +0.029] | +0.046 [+0.025, +0.068] | +0.025 [+0.010, +0.041] | +0.084 [+0.065, +0.104] | +0.017 | +0.065 | U-LED |
| k | 3 | 293 | +0.091 [+0.069, +0.114] | +0.021 [-0.000, +0.040] | +0.044 [+0.023, +0.064] | +0.023 [+0.007, +0.039] | +0.068 [+0.045, +0.091] | +0.022 | +0.056 | U-LED |
| k | 4 | 277 | +0.076 [+0.052, +0.102] | +0.002 [-0.018, +0.019] | +0.032 [+0.008, +0.059] | +0.026 [+0.011, +0.042] | +0.064 [+0.039, +0.089] | +0.014 | +0.048 | U-LED |
| k | 5 | 266 | +0.060 [+0.042, +0.080] | +0.010 [-0.009, +0.028] | -0.006 [-0.026, +0.014] | +0.030 [+0.015, +0.047] | +0.049 [+0.031, +0.069] | +0.020 | +0.022 | SIDES-DISAGREE |
| k | 6 | | | | | | | | | UNDEFINED (no never-seen mon at k = 6) |
| material | 12 | 254 | +0.094 [+0.071, +0.116] | +0.018 [-0.005, +0.041] | +0.033 [+0.011, +0.055] | +0.033 [+0.012, +0.053] | +0.087 [+0.065, +0.110] | +0.025 | +0.060 | SIDES-DISAGREE |
| material | 10-11 | 450 | +0.070 [+0.053, +0.088] | +0.011 [-0.004, +0.027] | +0.028 [+0.012, +0.045] | +0.028 [+0.014, +0.040] | +0.056 [+0.039, +0.074] | +0.019 | +0.042 | SIDES-DISAGREE |
| material | 8-9 | 344 | +0.081 [+0.059, +0.103] | +0.012 [-0.005, +0.028] | +0.031 [+0.009, +0.052] | +0.032 [+0.016, +0.048] | +0.055 [+0.036, +0.073] | +0.022 | +0.043 | SIDES-DISAGREE |
| material | <=7 | 372 | +0.040 [+0.026, +0.056] | +0.009 [-0.005, +0.025] | +0.009 [-0.004, +0.022] | +0.013 [+0.005, +0.022] | +0.031 [+0.018, +0.045] | +0.011 | +0.020 | SIDES-DISAGREE |
| turn | <=7 | 341 | +0.095 [+0.072, +0.116] | +0.013 [-0.009, +0.035] | +0.045 [+0.024, +0.066] | +0.030 [+0.013, +0.047] | +0.082 [+0.062, +0.102] | +0.022 | +0.064 | U-LED |
| turn | 8-14 | 436 | +0.079 [+0.061, +0.097] | +0.012 [-0.002, +0.025] | +0.041 [+0.026, +0.057] | +0.022 [+0.010, +0.034] | +0.069 [+0.052, +0.087] | +0.017 | +0.055 | U-LED |
| turn | 15-21 | 322 | +0.067 [+0.049, +0.088] | +0.024 [+0.006, +0.042] | +0.012 [-0.007, +0.029] | +0.036 [+0.023, +0.051] | +0.041 [+0.024, +0.060] | +0.030 | +0.026 | NOT DETECTED |
| turn | >=22 | 321 | +0.030 [+0.015, +0.045] | -0.002 [-0.017, +0.014] | -0.006 [-0.021, +0.010] | +0.016 [+0.004, +0.031] | +0.020 [+0.008, +0.032] | +0.007 | +0.007 | NOT DETECTED |

### R-STAGE-SEARCH: FRONTIER = EMPTY

| axis | bucket | n pooled | TRUNC-4-K8 − CRITIC (95 %) | TRUNC-4-K8 | end2 / prod hold | verdict |
|---|---|---|---|---|---|---|
| k | 1-2 | 331 | +0.007 [-0.012, +0.024] | +0.003 [-0.016, +0.022] | False / False | GATE-CLOSED |
| k | 3 | 299 | -0.010 [-0.028, +0.010] | -0.011 [-0.033, +0.010] | False / False | GATE-CLOSED |
| k | 4 | 286 | +0.001 [-0.021, +0.021] | +0.007 [-0.016, +0.029] | False / False | GATE-CLOSED |
| k | 5 | 285 | -0.014 [-0.033, +0.006] | -0.015 [-0.035, +0.006] | False / False | GATE-CLOSED |
| k | 6 | 276 | +0.024 [+0.006, +0.044] | +0.029 [+0.012, +0.049] | False / False | GATE-WEAK |
| material | 12 | 257 | -0.016 [-0.034, +0.004] | -0.009 [-0.030, +0.010] | False / False | GATE-CLOSED |
| material | 10-11 | 468 | +0.004 [-0.011, +0.020] | +0.015 [-0.001, +0.031] | False / False | GATE-CLOSED |
| material | 8-9 | 357 | +0.011 [-0.010, +0.031] | -0.004 [-0.025, +0.018] | False / False | GATE-CLOSED |
| material | <=7 | 395 | +0.001 [-0.014, +0.017] | +0.000 [-0.016, +0.017] | False / False | GATE-CLOSED |
| turn | <=7 | 346 | -0.003 [-0.020, +0.017] | -0.001 [-0.020, +0.018] | False / False | GATE-CLOSED |
| turn | 8-14 | 451 | -0.000 [-0.015, +0.015] | -0.003 [-0.019, +0.012] | False / False | GATE-CLOSED |
| turn | 15-21 | 335 | -0.003 [-0.023, +0.017] | +0.001 [-0.022, +0.025] | False / False | GATE-CLOSED |
| turn | >=22 | 345 | +0.012 [-0.004, +0.030] | +0.014 [-0.002, +0.030] | False / False | GATE-CLOSED |

## R-DOSE (§6.9; M8 = the TRUE world at weight p, F's worlds at 1 − p; LEAF-K8)

Pooled: P-STAR {'vs_critic_same_p': 0.1, 'vs_critic_today': 0.1, 'vs_policy': 0.1}; TODAY (FN) 0.009070464767616198 (upper95 0.0322); WITHIN-REACH; SHAPE CONCAVE (Δ² +0.025 [+0.020, +0.031]); p-half 0.25; CONTENT-MATTERS True; OVERCONFIDENCE-COSTS False; CRITIC-INFO-LIMITED True

| p | IMP leaf | IMP critic | leaf − critic (same p) |
|---|---|---|---|
| 0 | +0.002 [-0.007, +0.011] | +0.001 [-0.007, +0.009] | +0.001 [-0.008, +0.010] |
| 0.1 | +0.021 [+0.012, +0.031] | +0.004 [-0.005, +0.012] | +0.018 [+0.008, +0.027] |
| 0.2 | +0.032 [+0.023, +0.041] | +0.008 [-0.000, +0.016] | +0.024 [+0.015, +0.034] |
| 0.35 | +0.053 [+0.044, +0.062] | +0.009 [+0.001, +0.017] | +0.044 [+0.035, +0.054] |
| 0.5 | +0.062 [+0.053, +0.071] | +0.010 [+0.002, +0.019] | +0.052 [+0.042, +0.062] |
| 0.75 | +0.071 [+0.061, +0.080] | +0.014 [+0.006, +0.022] | +0.056 [+0.047, +0.066] |
| 1 | +0.072 [+0.062, +0.081] | +0.015 [+0.007, +0.023] | +0.056 [+0.047, +0.066] |

| axis | bucket | n | P-STAR (policy / critic same p / critic today) | TODAY [upper95] | REACH | SHAPE | Π |
|---|---|---|---|---|---|---|---|
| k | 1-2 | 331 | 0.2 / 0.1 / 0.1 | 0.014935064935064931 [0.1646] | WITHIN-REACH | CONCAVE | -0.000 [-0.001, +0.000] |
| k | 3 | 299 | 0.35 / 0.35 / 0.35 | 0.035826771653543296 [0.1775] | INTERMEDIATE | CONCAVE | +0.003 [-0.002, +0.010] |
| k | 4 | 286 | 0.1 / 0.1 / 0.1 | 0.0 [0.0216] | WITHIN-REACH | CONCAVE | +0.032 [+0.019, +0.046] |
| k | 5 | 285 | 0.35 / 0.35 / 0.35 | 0.19999999999999998 [0.3216] | WITHIN-REACH | CONCAVE | +0.011 [+0.003, +0.021] |
| k | 6 | 276 | 0.0 / 0.0 / 0.0 | 0.0 [0.0305] | WITHIN-REACH | CONCAVE | +0.012 [+0.003, +0.024] |
| k | DETERMINIZATION-FIRST | ['4'] | | | | | |
| material | 12 | 257 | 0.35 / 0.35 / 0.35 | 0.13333333333333322 [0.2851] | WITHIN-REACH | CONCAVE | +0.012 [+0.006, +0.021] |
| material | 10-11 | 468 | 0.1 / 0.1 / 0.1 | 0.0 [0.0254] | WITHIN-REACH | CONCAVE | +0.026 [+0.016, +0.036] |
| material | 8-9 | 357 | 0.1 / 0.1 / 0.1 | 0.004278074866310167 [0.1025] | WITHIN-REACH | CONCAVE | +0.022 [+0.012, +0.032] |
| material | <=7 | 395 | 0.1 / 0.1 / 0.1 | 0.02683982683982684 [0.2577] | WITHIN-REACH | CONCAVE | +0.002 [+0.000, +0.006] |
| material | DETERMINIZATION-FIRST | none | | | | | |
| turn | <=7 | 346 | 0.2 / 0.35 / 0.2 | 0.015384615384615389 [0.1719] | INTERMEDIATE | CONCAVE | +0.000 [-0.002, +0.002] |
| turn | 8-14 | 451 | 0.2 / 0.2 / 0.2 | 0.03158458244111349 [0.1594] | WITHIN-REACH | CONCAVE | +0.003 [+0.000, +0.008] |
| turn | 15-21 | 335 | 0.1 / 0.1 / 0.1 | 0.0 [0.0264] | WITHIN-REACH | CONCAVE | +0.028 [+0.016, +0.042] |
| turn | >=22 | 345 | 0.1 / 0.1 / 0.1 | 0.0 [0.2202] | WITHIN-REACH | CONCAVE | +0.006 [+0.001, +0.012] |
| turn | DETERMINIZATION-FIRST | ['15-21'] | | | | | |

## Amendment 3: per-class dose curves (F+X worlds at weight p, F worlds at 1 − p)

- **M** pooled: P-STAR {'vs_critic_same_p': None, 'vs_critic_today': 0.75, 'vs_policy': 0.75}; SHAPE LINEAR; p-half 0.65; TODAY proxy +0.011 [-0.000, +0.021] (hit F +0.508 [+0.501, +0.515], FN +0.513 [+0.507, +0.520]); FN joint on this curve 0.5279569892473118
- **U** pooled: P-STAR {'vs_critic_same_p': 0.35, 'vs_critic_today': 0.2, 'vs_policy': 0.2}; SHAPE CONCAVE; p-half 0.25; TODAY proxy +0.122 [+0.112, +0.132] (hit F +0.058 [+0.054, +0.061], FN +0.172 [+0.163, +0.182]); FN joint on this curve 0.025155925155925175

| axis | bucket | M P-STAR (pol / crit) | U P-STAR (pol / crit) | M p-proxy | U p-proxy | CLASS-FIRST (from proxy; from 0) |
|---|---|---|---|---|---|---|
| k | 1-2 | None / None | 0.35 / 0.35 | +0.048 [+0.016, +0.078] | +0.037 [+0.030, +0.043] | NOT DETECTED; U-FIRST |
| k | 3 | None / None | 1.0 / None | +0.018 [-0.005, +0.040] | +0.092 [+0.080, +0.104] | NOT DETECTED; U-FIRST |
| k | 4 | None / None | 0.1 / 0.1 | +0.005 [-0.019, +0.027] | +0.184 [+0.167, +0.201] | NOT DETECTED; U-FIRST |
| k | 5 | None / None | None / None | +0.002 [-0.020, +0.023] | +0.537 [+0.492, +0.582] | NOT DETECTED; NOT DETECTED |
| k | 6 | 0.0 / 0.0 | 0.0 / 0.0 | +0.004 [-0.013, +0.022] | — | —; — |
| material | 12 | None / None | None / None | +0.007 [-0.022, +0.036] | +0.053 [+0.042, +0.066] | NOT DETECTED; NOT DETECTED |
| material | 10-11 | 0.75 / None | 0.1 / None | -0.003 [-0.022, +0.016] | +0.104 [+0.091, +0.118] | NOT DETECTED; U-FIRST |
| material | 8-9 | None / None | 0.35 / 0.1 | +0.019 [-0.002, +0.039] | +0.175 [+0.153, +0.198] | NOT DETECTED; NOT DETECTED |
| material | <=7 | None / None | None / 0.75 | +0.017 [+0.001, +0.033] | +0.300 [+0.258, +0.345] | NOT DETECTED; NOT DETECTED |
| turn | <=7 | None / None | 0.35 / None | +0.010 [-0.018, +0.036] | +0.046 [+0.039, +0.054] | NOT DETECTED; NOT DETECTED |
| turn | 8-14 | None / None | 0.35 / 0.5 | +0.014 [-0.003, +0.031] | +0.142 [+0.127, +0.158] | NOT DETECTED; NOT DETECTED |
| turn | 15-21 | 0.75 / 1.0 | 0.1 / None | +0.013 [-0.006, +0.033] | +0.199 [+0.170, +0.232] | NOT DETECTED; NOT DETECTED |
| turn | >=22 | None / None | None / None | +0.004 [-0.014, +0.022] | +0.261 [+0.215, +0.311] | NOT DETECTED; NOT DETECTED |

## Validity, exclusions, size rules

```
{
 "F+U = F at k = 6": {
  "states": 276,
  "states_differing": {
   "CRITIC": 0,
   "LEAF-K8": 0,
   "TRUNC-4-K8": 0
  }
 },
 "F_excluded_over_10pct": false,
 "HE@F re-play = the arm's own critic scorers": {
  "CRITIC_differing": 0,
  "TRUNC-4-K8_differing": 0,
  "states": 1097
 },
 "HE@O re-play = the arm's own critic scorers": {
  "CRITIC_differing": 0,
  "TRUNC-4-K8_differing": 0,
  "states": 1200
 },
 "O-U = O (seeds 0-7) at k = 6": {
  "states": 320,
  "states_differing": {
   "CRITIC": 0,
   "LEAF-K8": 0,
   "TRUNC-4-K8": 0
  }
 },
 "exclusions": {
  "D50": {
   "excluded_or_refused": 103,
   "missing": 0,
   "ok": 1097,
   "share_excluded": 0.0858,
   "states": 1200
  },
  "D90": {
   "excluded_or_refused": 103,
   "missing": 0,
   "ok": 1097,
   "share_excluded": 0.0858,
   "states": 1200
  },
  "F": {
   "excluded_or_refused": 143,
   "missing": 0,
   "ok": 1477,
   "share_excluded": 0.0883,
   "states": 1620
  },
  "F+M": {
   "excluded_or_refused": 139,
   "missing": 0,
   "ok": 1481,
   "share_excluded": 0.0858,
   "states": 1620
  },
  "F+U": {
   "excluded_or_refused": 142,
   "missing": 0,
   "ok": 1478,
   "share_excluded": 0.0877,
   "states": 1620
  },
  "FN": {
   "excluded_or_refused": 143,
   "missing": 0,
   "ok": 1477,
   "share_excluded": 0.0883,
   "states": 1620
  },
  "MAP100": {
   "excluded_or_refused": 331,
   "missing": 0,
   "ok": 869,
   "share_excluded": 0.2758,
   "states": 1200
  },
  "O-M": {
   "excluded_or_refused": 0,
   "missing": 0,
   "ok": 1620,
   "share_excluded": 0.0,
   "states": 1620
  },
  "O-U": {
   "excluded_or_refused": 2,
   "missing": 0,
   "ok": 1618,
   "share_excluded": 0.0012,
   "states": 1620
  },
  "POOL": {
   "excluded_or_refused": 0,
   "missing": 0,
   "ok": 1200,
   "share_excluded": 0.0,
   "states": 1200
  },
  "POOLSETS": {
   "excluded_or_refused": 62,
   "missing": 0,
   "ok": 1138,
   "share_excluded": 0.0517,
   "states": 1200
  },
  "W90": {
   "excluded_or_refused": 106,
   "missing": 0,
   "ok": 1094,
   "share_excluded": 0.0883,
   "states": 1200
  }
 }
}
```

Size rules: {"G_F_halfwidth_by_k": {"1-2": 0.021, "3": 0.0228, "4": 0.0243, "5": 0.0183, "6": 0.0161}, "n_M_halfwidth": 0.1134, "rule_12_10_triggered": false, "rule_12_13_2_bins_triggered": [], "s_M_halfwidth": 0.1333}

Natural mix: {"G_all_k_weighted": {"ci": [0.06073, 0.08034], "mean": 0.07044}, "G_all_unweighted": {"ci": [0.05986, 0.07922], "mean": 0.06947}}

Root-choice sensitivity: {"G": {"ci": [0.05952, 0.08122], "mean": 0.07042, "n": 1091}, "shares": {"M": {"n": {"ci": [0.30336, 0.5303], "mean": 0.41486, "n": 1091}, "s": {"ci": [-0.01396, 0.25681], "mean": 0.12596, "n": 1051}}, "U": {"n": {"ci": [0.69812, 0.95989], "mean": 0.82213, "n": 1091}, "s": {"ci": [0.17691, 0.42851], "mean": 0.30992, "n": 1091}}}, "states_dropped": 8}

CONDITION (§10.4, partial): CONDITION-DEPENDENT {"opp:bot": {"G": {"ci": [0.05715, 0.08898], "mean": 0.07303}, "n": 556, "s_M": {"ci": [-0.1399, 0.23934], "mean": 0.07799}}, "opp:sentinel": {"G": {"ci": [0.0476, 0.07829], "mean": 0.0626}, "n": 437, "s_M": {"ci": [-0.01943, 0.44818], "mean": 0.2262}}, "phase:11-25": {"G": {"ci": [0.04944, 0.08001], "mean": 0.06458}, "n": 470, "s_M": {"ci": [0.02484, 0.39804], "mean": 0.22306}}, "phase:2-10": {"G": {"ci": [0.07284, 0.11082], "mean": 0.09236}, "n": 443, "s_M": {"ci": [-0.11053, 0.24241], "mean": 0.08758}}, "phase:26-40": {"G": {"ci": [0.00063, 0.03472], "mean": 0.0162}, "n": 144, "s_M": {"ci": [-6.25991, 1.18909], "mean": -0.33036}}}
