# Value-discrimination pilot, PHASE 2 (determinized worlds): result tables

IMP against the TRUE world's 24 truth seeds; det = honest-information worlds (Smogon-prior samples of the
opponent's hidden team, rejected unless the searched side's observations replay byte-equal).

## end2 (578 states, 402 battles; excluded 22)

Verdicts: R-HONEST **NOT DETECTED**, R-INFO **INFO-COSTS**

| scorer (ARGMAX) | det IMP [95 %] | oracle IMP, same states | det CPU-s / decision |
|---|---|---|---|
| ARGMAX|CRITIC | +0.0129 [+0.0005, +0.0259] | +0.0266 | 2.08 |
| ARGMAX|CV-K1 | -0.0240 [-0.0391, -0.0085] | +0.0250 | 4.86 |
| ARGMAX|CV-K2 | -0.0017 [-0.0170, +0.0140] | +0.0457 | 7.74 |
| ARGMAX|CV-K4 | +0.0052 [-0.0108, +0.0208] | +0.0663 | 13.45 |
| ARGMAX|CV-K8 | +0.0061 [-0.0092, +0.0218] | +0.0760 | 24.97 |
| ARGMAX|LEAF-K1 | -0.0009 [-0.0111, +0.0091] | +0.0338 | 2.77 |
| ARGMAX|LEAF-K2 | +0.0028 [-0.0085, +0.0147] | +0.0520 | 5.66 |
| ARGMAX|LEAF-K4 | +0.0042 [-0.0089, +0.0185] | +0.0674 | 11.37 |
| ARGMAX|LEAF-K8 | +0.0068 [-0.0075, +0.0221] | +0.0788 | 22.89 |
| ARGMAX|MIX-K1 | +0.0045 [-0.0092, +0.0178] | +0.0498 | 4.86 |
| ARGMAX|MIX-K4 | +0.0092 [-0.0057, +0.0249] | +0.0736 | 13.45 |
| ARGMAX|MIX-K8 | +0.0138 [-0.0006, +0.0289] | +0.0865 | 24.97 |
| ARGMAX|TRUNC-1-K1 | +0.0076 [-0.0061, +0.0210] | +0.0260 | 0.11 |
| ARGMAX|TRUNC-1-K4 | +0.0136 [+0.0010, +0.0263] | +0.0242 | 0.45 |
| ARGMAX|TRUNC-1-K8 | +0.0162 [+0.0043, +0.0287] | +0.0283 | 0.90 |
| ARGMAX|TRUNC-2-K1 | +0.0039 [-0.0093, +0.0179] | +0.0238 | 0.22 |
| ARGMAX|TRUNC-2-K4 | +0.0176 [+0.0030, +0.0315] | +0.0350 | 0.88 |
| ARGMAX|TRUNC-2-K8 | +0.0255 [+0.0127, +0.0384] | +0.0399 | 1.75 |
| ARGMAX|TRUNC-4-K1 | +0.0048 [-0.0106, +0.0196] | +0.0264 | 0.42 |
| ARGMAX|TRUNC-4-K4 | +0.0165 [+0.0018, +0.0311] | +0.0516 | 1.69 |
| ARGMAX|TRUNC-4-K8 | +0.0231 [+0.0096, +0.0375] | +0.0504 | 3.38 |

| rule (det) | IMP [95 %] |
|---|---|
| GATE2SE|LEAF-K4 | -0.0006 [-0.0042, +0.0032] |
| GATE2SE|LEAF-K8 | +0.0057 [-0.0010, +0.0129] |
| GUMBEL|LEAF-K1 | +0.0026 [-0.0055, +0.0118] |
| GUMBEL|LEAF-K2 | +0.0042 [-0.0063, +0.0153] |
| GUMBEL|LEAF-K4 | +0.0072 [-0.0034, +0.0193] |
| GUMBEL|LEAF-K8 | +0.0125 [+0.0014, +0.0249] |
| PRIOR2|LEAF-K8 | +0.0092 [-0.0006, +0.0201] |
| SH-4|B=n*1 | +0.0015 [-0.0074, +0.0110] |
| SH-4|B=n*2 | +0.0058 [-0.0042, +0.0161] |
| SH-4|B=n*4 | +0.0155 [+0.0036, +0.0283] |
| SH-4|B=n*8 | +0.0145 [+0.0024, +0.0272] |
| SH-all|B=n*1 | +0.0023 [-0.0092, +0.0144] |
| SH-all|B=n*2 | +0.0006 [-0.0106, +0.0120] |
| SH-all|B=n*4 | +0.0071 [-0.0053, +0.0205] |
| SH-all|B=n*8 | +0.0057 [-0.0073, +0.0198] |

| contrast | Δ [95 %] |
|---|---|
| R-HONEST: det LEAF-K8 - det CRITIC | -0.0061 [-0.0222, +0.0100] |
| R-INFO: oracle LEAF-K8 - det LEAF-K8 | +0.0720 [+0.0567, +0.0868] |
| det LEAF-K8 - oracle CRITIC | -0.0198 [-0.0347, -0.0045] |
| det CRITIC - oracle CRITIC | -0.0137 [-0.0245, -0.0042] |
| det LEAF-K8 - det LEAF-K4 | +0.0026 [-0.0100, +0.0154] |
| det GUMBEL - ARGMAX at LEAF-K4 | +0.0030 [-0.0042, +0.0107] |
| det SH-4 (B=n*2) - ARGMAX LEAF-K2 | +0.0030 [-0.0073, +0.0141] |
| R-INFO at K1: oracle - det LEAF-K1 | +0.0348 [+0.0216, +0.0473] |
| R-INFO at K4: oracle - det LEAF-K4 | +0.0632 [+0.0478, +0.0782] |
| det TRUNC-1-K8 - det CRITIC | +0.0033 [-0.0042, +0.0113] |
| det TRUNC-1-K8 - det LEAF-K8 | +0.0094 [-0.0058, +0.0245] |
| det TRUNC-2-K8 - det CRITIC | +0.0126 [-0.0012, +0.0263] |
| det TRUNC-2-K8 - det LEAF-K8 | +0.0187 [+0.0037, +0.0328] |
| det TRUNC-4-K8 - det CRITIC | +0.0102 [-0.0031, +0.0242] |
| det TRUNC-4-K8 - det LEAF-K8 | +0.0163 [+0.0012, +0.0312] |
| det CV-K8 - det LEAF-K8 | -0.0006 [-0.0102, +0.0082] |
| retained share, LEAF-K1 (det / oracle) | -0.028 [-0.3648, 0.2623] |
| retained share, LEAF-K4 (det / oracle) | 0.062 [-0.1452, 0.2568] |
| retained share, LEAF-K8 (det / oracle) | 0.086 [-0.0996, 0.2618] |

| unrevealed opponent slots | n | det LEAF-K8 | oracle LEAF-K8 | det CRITIC | oracle − det |
|---|---|---|---|---|---|
| unrev_0-1 | 236 | +0.0194 | +0.0724 | +0.0215 | +0.0530 [0.03198, 0.076] |
| unrev_2-3 | 198 | -0.0091 | +0.0838 | +0.0094 | +0.0928 [0.06369, 0.12254] |
| unrev_4-5 | 144 | +0.0078 | +0.0823 | +0.0036 | +0.0745 [0.04695, 0.10254] |

Worlds: {'tries_mean': 2.562, 'rewritten_moves': 179, 'rewritten_switches': 3, 'ambiguous_moves_kept': 68, 'root_choice_kept': 7, 'new_slots_mean': 2.128, 'redrawn_move_slots_mean': 10.128}

## prod (579 states, 408 battles; excluded 21)

Verdicts: R-HONEST **NOT DETECTED**, R-INFO **INFO-COSTS**

| scorer (ARGMAX) | det IMP [95 %] | oracle IMP, same states | det CPU-s / decision |
|---|---|---|---|
| ARGMAX|CRITIC | +0.0019 [-0.0099, +0.0139] | +0.0124 | 1.38 |
| ARGMAX|CV-K1 | -0.0332 [-0.0484, -0.0169] | +0.0041 | 3.15 |
| ARGMAX|CV-K2 | -0.0286 [-0.0436, -0.0129] | +0.0303 | 4.93 |
| ARGMAX|CV-K4 | -0.0165 [-0.0306, -0.0018] | +0.0499 | 8.57 |
| ARGMAX|CV-K8 | -0.0128 [-0.0286, +0.0034] | +0.0683 | 15.84 |
| ARGMAX|LEAF-K1 | -0.0024 [-0.0101, +0.0057] | +0.0270 | 1.76 |
| ARGMAX|LEAF-K2 | +0.0005 [-0.0100, +0.0115] | +0.0396 | 3.55 |
| ARGMAX|LEAF-K4 | -0.0045 [-0.0162, +0.0080] | +0.0573 | 7.19 |
| ARGMAX|LEAF-K8 | +0.0038 [-0.0101, +0.0186] | +0.0703 | 14.46 |
| ARGMAX|MIX-K1 | +0.0032 [-0.0086, +0.0154] | +0.0338 | 3.15 |
| ARGMAX|MIX-K4 | -0.0052 [-0.0189, +0.0093] | +0.0611 | 8.57 |
| ARGMAX|MIX-K8 | -0.0006 [-0.0157, +0.0145] | +0.0749 | 15.84 |
| ARGMAX|TRUNC-1-K1 | -0.0054 [-0.0179, +0.0073] | +0.0070 | 0.08 |
| ARGMAX|TRUNC-1-K4 | -0.0001 [-0.0127, +0.0130] | +0.0114 | 0.30 |
| ARGMAX|TRUNC-1-K8 | -0.0039 [-0.0156, +0.0085] | +0.0138 | 0.61 |
| ARGMAX|TRUNC-2-K1 | -0.0113 [-0.0253, +0.0040] | +0.0076 | 0.15 |
| ARGMAX|TRUNC-2-K4 | -0.0050 [-0.0174, +0.0086] | +0.0130 | 0.59 |
| ARGMAX|TRUNC-2-K8 | -0.0049 [-0.0177, +0.0086] | +0.0142 | 1.19 |
| ARGMAX|TRUNC-4-K1 | -0.0101 [-0.0258, +0.0049] | +0.0274 | 0.28 |
| ARGMAX|TRUNC-4-K4 | +0.0076 [-0.0084, +0.0226] | +0.0359 | 1.14 |
| ARGMAX|TRUNC-4-K8 | -0.0011 [-0.0152, +0.0133] | +0.0348 | 2.29 |

| rule (det) | IMP [95 %] |
|---|---|
| GATE2SE|LEAF-K4 | -0.0008 [-0.0041, +0.0026] |
| GATE2SE|LEAF-K8 | +0.0017 [-0.0048, +0.0098] |
| GUMBEL|LEAF-K1 | -0.0022 [-0.0095, +0.0050] |
| GUMBEL|LEAF-K2 | -0.0016 [-0.0100, +0.0070] |
| GUMBEL|LEAF-K4 | -0.0009 [-0.0112, +0.0086] |
| GUMBEL|LEAF-K8 | -0.0009 [-0.0113, +0.0099] |
| PRIOR2|LEAF-K8 | +0.0037 [-0.0049, +0.0129] |
| SH-4|B=n*1 | -0.0062 [-0.0150, +0.0026] |
| SH-4|B=n*2 | -0.0074 [-0.0185, +0.0031] |
| SH-4|B=n*4 | -0.0018 [-0.0132, +0.0101] |
| SH-4|B=n*8 | +0.0017 [-0.0097, +0.0146] |
| SH-all|B=n*1 | -0.0026 [-0.0127, +0.0080] |
| SH-all|B=n*2 | -0.0054 [-0.0172, +0.0055] |
| SH-all|B=n*4 | +0.0015 [-0.0114, +0.0147] |
| SH-all|B=n*8 | +0.0023 [-0.0103, +0.0162] |

| contrast | Δ [95 %] |
|---|---|
| R-HONEST: det LEAF-K8 - det CRITIC | +0.0019 [-0.0122, +0.0164] |
| R-INFO: oracle LEAF-K8 - det LEAF-K8 | +0.0665 [+0.0527, +0.0800] |
| det LEAF-K8 - oracle CRITIC | -0.0086 [-0.0240, +0.0068] |
| det CRITIC - oracle CRITIC | -0.0105 [-0.0203, -0.0002] |
| det LEAF-K8 - det LEAF-K4 | +0.0083 [-0.0029, +0.0197] |
| det GUMBEL - ARGMAX at LEAF-K4 | +0.0036 [-0.0041, +0.0108] |
| det SH-4 (B=n*2) - ARGMAX LEAF-K2 | -0.0078 [-0.0171, +0.0006] |
| R-INFO at K1: oracle - det LEAF-K1 | +0.0295 [+0.0163, +0.0447] |
| R-INFO at K4: oracle - det LEAF-K4 | +0.0617 [+0.0469, +0.0779] |
| det TRUNC-1-K8 - det CRITIC | -0.0058 [-0.0130, +0.0010] |
| det TRUNC-1-K8 - det LEAF-K8 | -0.0077 [-0.0230, +0.0069] |
| det TRUNC-2-K8 - det CRITIC | -0.0068 [-0.0208, +0.0068] |
| det TRUNC-2-K8 - det LEAF-K8 | -0.0087 [-0.0230, +0.0050] |
| det TRUNC-4-K8 - det CRITIC | -0.0030 [-0.0171, +0.0116] |
| det TRUNC-4-K8 - det LEAF-K8 | -0.0049 [-0.0186, +0.0083] |
| det CV-K8 - det LEAF-K8 | -0.0166 [-0.0254, -0.0077] |
| retained share, LEAF-K1 (det / oracle) | -0.090 [-0.4921, 0.2059] |
| retained share, LEAF-K4 (det / oracle) | -0.079 [-0.3187, 0.1291] |
| retained share, LEAF-K8 (det / oracle) | 0.054 [-0.1697, 0.2306] |

| unrevealed opponent slots | n | det LEAF-K8 | oracle LEAF-K8 | det CRITIC | oracle − det |
|---|---|---|---|---|---|
| unrev_0-1 | 217 | +0.0136 | +0.0616 | +0.0026 | +0.0480 [0.02876, 0.06989] |
| unrev_2-3 | 194 | +0.0020 | +0.0785 | +0.0089 | +0.0765 [0.05113, 0.1027] |
| unrev_4-5 | 168 | -0.0068 | +0.0719 | -0.0071 | +0.0788 [0.05313, 0.1034] |

Worlds: {'tries_mean': 2.632, 'rewritten_moves': 186, 'rewritten_switches': 0, 'ambiguous_moves_kept': 71, 'root_choice_kept': 0, 'new_slots_mean': 2.29, 'redrawn_move_slots_mean': 9.791}

