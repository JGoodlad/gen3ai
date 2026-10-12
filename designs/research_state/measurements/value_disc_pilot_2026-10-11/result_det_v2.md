# Value-discrimination pilot, PHASE 2 (determinized worlds): result tables

IMP against the TRUE world's 24 truth seeds; det = honest-information worlds (Smogon-prior samples of the
opponent's hidden team, rejected unless the searched side's observations replay byte-equal).

## end2 (599 states, 417 battles; excluded 1)

Verdicts: R-HONEST **NOT DETECTED**, R-INFO **INFO-COSTS**

| scorer (ARGMAX) | det IMP [95 %] | oracle IMP, same states | det CPU-s / decision |
|---|---|---|---|
| ARGMAX|CRITIC | +0.0139 [+0.0019, +0.0264] | +0.0261 | 1.94 |
| ARGMAX|CV-K1 | -0.0224 [-0.0376, -0.0080] | +0.0251 | 4.65 |
| ARGMAX|CV-K2 | -0.0036 [-0.0192, +0.0122] | +0.0479 | 7.45 |
| ARGMAX|CV-K4 | +0.0026 [-0.0133, +0.0182] | +0.0679 | 12.78 |
| ARGMAX|CV-K8 | +0.0055 [-0.0101, +0.0211] | +0.0785 | 23.71 |
| ARGMAX|LEAF-K1 | +0.0042 [-0.0056, +0.0129] | +0.0348 | 2.71 |
| ARGMAX|LEAF-K2 | +0.0104 [-0.0008, +0.0224] | +0.0542 | 5.51 |
| ARGMAX|LEAF-K4 | +0.0153 [+0.0027, +0.0290] | +0.0687 | 10.84 |
| ARGMAX|LEAF-K8 | +0.0110 [-0.0015, +0.0244] | +0.0809 | 21.77 |
| ARGMAX|MIX-K1 | +0.0073 [-0.0060, +0.0202] | +0.0495 | 4.65 |
| ARGMAX|MIX-K4 | +0.0138 [-0.0015, +0.0295] | +0.0735 | 12.78 |
| ARGMAX|MIX-K8 | +0.0150 [+0.0010, +0.0300] | +0.0874 | 23.71 |
| ARGMAX|TRUNC-1-K1 | +0.0095 [-0.0039, +0.0231] | +0.0271 | 0.11 |
| ARGMAX|TRUNC-1-K4 | +0.0127 [-0.0002, +0.0263] | +0.0245 | 0.43 |
| ARGMAX|TRUNC-1-K8 | +0.0153 [+0.0035, +0.0274] | +0.0278 | 0.85 |
| ARGMAX|TRUNC-2-K1 | +0.0022 [-0.0111, +0.0157] | +0.0266 | 0.21 |
| ARGMAX|TRUNC-2-K4 | +0.0199 [+0.0067, +0.0334] | +0.0364 | 0.83 |
| ARGMAX|TRUNC-2-K8 | +0.0223 [+0.0093, +0.0356] | +0.0410 | 1.66 |
| ARGMAX|TRUNC-4-K1 | +0.0022 [-0.0128, +0.0166] | +0.0268 | 0.40 |
| ARGMAX|TRUNC-4-K4 | +0.0157 [+0.0020, +0.0291] | +0.0525 | 1.60 |
| ARGMAX|TRUNC-4-K8 | +0.0213 [+0.0084, +0.0346] | +0.0514 | 3.21 |

| rule (det) | IMP [95 %] |
|---|---|
| GATE2SE|LEAF-K4 | -0.0000 [-0.0039, +0.0039] |
| GATE2SE|LEAF-K8 | +0.0045 [-0.0017, +0.0104] |
| GUMBEL|LEAF-K1 | +0.0075 [+0.0001, +0.0148] |
| GUMBEL|LEAF-K2 | +0.0113 [+0.0011, +0.0222] |
| GUMBEL|LEAF-K4 | +0.0171 [+0.0062, +0.0292] |
| GUMBEL|LEAF-K8 | +0.0135 [+0.0021, +0.0249] |
| PRIOR2|LEAF-K8 | +0.0101 [+0.0007, +0.0193] |
| SH-4|B=n*1 | +0.0113 [+0.0026, +0.0206] |
| SH-4|B=n*2 | +0.0149 [+0.0049, +0.0251] |
| SH-4|B=n*4 | +0.0206 [+0.0097, +0.0323] |
| SH-4|B=n*8 | +0.0175 [+0.0056, +0.0285] |
| SH-all|B=n*1 | +0.0096 [-0.0013, +0.0214] |
| SH-all|B=n*2 | +0.0143 [+0.0039, +0.0256] |
| SH-all|B=n*4 | +0.0173 [+0.0062, +0.0291] |
| SH-all|B=n*8 | +0.0137 [+0.0010, +0.0269] |

| contrast | Δ [95 %] |
|---|---|
| R-HONEST: det LEAF-K8 - det CRITIC | -0.0029 [-0.0175, +0.0119] |
| R-INFO: oracle LEAF-K8 - det LEAF-K8 | +0.0699 [+0.0562, +0.0839] |
| det LEAF-K8 - oracle CRITIC | -0.0151 [-0.0290, -0.0015] |
| det CRITIC - oracle CRITIC | -0.0123 [-0.0225, -0.0033] |
| det LEAF-K8 - det LEAF-K4 | -0.0043 [-0.0148, +0.0066] |
| det GUMBEL - ARGMAX at LEAF-K4 | +0.0018 [-0.0043, +0.0088] |
| det SH-4 (B=n*2) - ARGMAX LEAF-K2 | +0.0045 [-0.0056, +0.0142] |
| R-INFO at K1: oracle - det LEAF-K1 | +0.0305 [+0.0193, +0.0427] |
| R-INFO at K4: oracle - det LEAF-K4 | +0.0534 [+0.0392, +0.0685] |
| det TRUNC-1-K8 - det CRITIC | +0.0014 [-0.0072, +0.0105] |
| det TRUNC-1-K8 - det LEAF-K8 | +0.0043 [-0.0099, +0.0179] |
| det TRUNC-2-K8 - det CRITIC | +0.0084 [-0.0056, +0.0220] |
| det TRUNC-2-K8 - det LEAF-K8 | +0.0112 [-0.0018, +0.0248] |
| det TRUNC-4-K8 - det CRITIC | +0.0074 [-0.0069, +0.0214] |
| det TRUNC-4-K8 - det LEAF-K8 | +0.0103 [-0.0034, +0.0237] |
| det CV-K8 - det LEAF-K8 | -0.0056 [-0.0161, +0.0050] |
| retained share, LEAF-K1 (det / oracle) | 0.122 [-0.1797, 0.3636] |
| retained share, LEAF-K4 (det / oracle) | 0.223 [0.0394, 0.3997] |
| retained share, LEAF-K8 (det / oracle) | 0.136 [-0.0213, 0.2855] |

| unrevealed opponent slots | n | det LEAF-K8 | oracle LEAF-K8 | det CRITIC | oracle − det |
|---|---|---|---|---|---|
| unrev_0-1 | 240 | +0.0168 | +0.0717 | +0.0228 | +0.0549 [0.03487, 0.07666] |
| unrev_2-3 | 206 | +0.0055 | +0.0832 | +0.0071 | +0.0778 [0.0531, 0.10268] |
| unrev_4-5 | 153 | +0.0095 | +0.0922 | +0.0090 | +0.0827 [0.05389, 0.11348] |

Worlds: {'tries_mean': 1.812, 'rewritten_moves': 237, 'rewritten_switches': 3, 'ambiguous_moves_kept': 0, 'root_choice_kept': 7, 'new_slots_mean': 2.162, 'redrawn_move_slots_mean': 10.142}

## prod (598 states, 416 battles; excluded 2)

Verdicts: R-HONEST **NOT DETECTED**, R-INFO **INFO-COSTS**

| scorer (ARGMAX) | det IMP [95 %] | oracle IMP, same states | det CPU-s / decision |
|---|---|---|---|
| ARGMAX|CRITIC | +0.0054 [-0.0060, +0.0167] | +0.0128 | 1.25 |
| ARGMAX|CV-K1 | -0.0270 [-0.0421, -0.0117] | +0.0047 | 2.89 |
| ARGMAX|CV-K2 | -0.0145 [-0.0304, +0.0012] | +0.0294 | 4.58 |
| ARGMAX|CV-K4 | -0.0151 [-0.0311, +0.0015] | +0.0517 | 7.88 |
| ARGMAX|CV-K8 | -0.0001 [-0.0142, +0.0151] | +0.0709 | 14.58 |
| ARGMAX|LEAF-K1 | +0.0013 [-0.0074, +0.0106] | +0.0266 | 1.64 |
| ARGMAX|LEAF-K2 | +0.0014 [-0.0102, +0.0130] | +0.0374 | 3.33 |
| ARGMAX|LEAF-K4 | +0.0000 [-0.0121, +0.0128] | +0.0581 | 6.63 |
| ARGMAX|LEAF-K8 | +0.0065 [-0.0069, +0.0206] | +0.0718 | 13.33 |
| ARGMAX|MIX-K1 | +0.0046 [-0.0075, +0.0170] | +0.0344 | 2.89 |
| ARGMAX|MIX-K4 | -0.0004 [-0.0146, +0.0144] | +0.0628 | 7.88 |
| ARGMAX|MIX-K8 | +0.0068 [-0.0073, +0.0213] | +0.0764 | 14.58 |
| ARGMAX|TRUNC-1-K1 | -0.0053 [-0.0177, +0.0069] | +0.0075 | 0.07 |
| ARGMAX|TRUNC-1-K4 | +0.0010 [-0.0105, +0.0131] | +0.0120 | 0.28 |
| ARGMAX|TRUNC-1-K8 | +0.0009 [-0.0107, +0.0132] | +0.0145 | 0.56 |
| ARGMAX|TRUNC-2-K1 | -0.0169 [-0.0313, -0.0023] | +0.0080 | 0.14 |
| ARGMAX|TRUNC-2-K4 | +0.0027 [-0.0100, +0.0155] | +0.0123 | 0.55 |
| ARGMAX|TRUNC-2-K8 | +0.0003 [-0.0127, +0.0138] | +0.0147 | 1.09 |
| ARGMAX|TRUNC-4-K1 | -0.0029 [-0.0172, +0.0113] | +0.0289 | 0.26 |
| ARGMAX|TRUNC-4-K4 | +0.0091 [-0.0053, +0.0235] | +0.0367 | 1.05 |
| ARGMAX|TRUNC-4-K8 | +0.0067 [-0.0066, +0.0207] | +0.0369 | 2.10 |

| rule (det) | IMP [95 %] |
|---|---|
| GATE2SE|LEAF-K4 | +0.0024 [-0.0006, +0.0061] |
| GATE2SE|LEAF-K8 | +0.0016 [-0.0049, +0.0082] |
| GUMBEL|LEAF-K1 | +0.0017 [-0.0068, +0.0106] |
| GUMBEL|LEAF-K2 | +0.0007 [-0.0091, +0.0104] |
| GUMBEL|LEAF-K4 | +0.0038 [-0.0064, +0.0140] |
| GUMBEL|LEAF-K8 | +0.0055 [-0.0049, +0.0166] |
| PRIOR2|LEAF-K8 | +0.0033 [-0.0047, +0.0112] |
| SH-4|B=n*1 | +0.0034 [-0.0062, +0.0135] |
| SH-4|B=n*2 | +0.0004 [-0.0106, +0.0113] |
| SH-4|B=n*4 | -0.0009 [-0.0120, +0.0105] |
| SH-4|B=n*8 | -0.0000 [-0.0110, +0.0113] |
| SH-all|B=n*1 | +0.0031 [-0.0077, +0.0144] |
| SH-all|B=n*2 | +0.0003 [-0.0106, +0.0116] |
| SH-all|B=n*4 | +0.0006 [-0.0111, +0.0132] |
| SH-all|B=n*8 | +0.0011 [-0.0107, +0.0137] |

| contrast | Δ [95 %] |
|---|---|
| R-HONEST: det LEAF-K8 - det CRITIC | +0.0011 [-0.0133, +0.0163] |
| R-INFO: oracle LEAF-K8 - det LEAF-K8 | +0.0653 [+0.0530, +0.0783] |
| det LEAF-K8 - oracle CRITIC | -0.0063 [-0.0212, +0.0094] |
| det CRITIC - oracle CRITIC | -0.0074 [-0.0170, +0.0031] |
| det LEAF-K8 - det LEAF-K4 | +0.0065 [-0.0057, +0.0181] |
| det GUMBEL - ARGMAX at LEAF-K4 | +0.0037 [-0.0040, +0.0118] |
| det SH-4 (B=n*2) - ARGMAX LEAF-K2 | -0.0010 [-0.0110, +0.0089] |
| R-INFO at K1: oracle - det LEAF-K1 | +0.0253 [+0.0137, +0.0373] |
| R-INFO at K4: oracle - det LEAF-K4 | +0.0581 [+0.0451, +0.0713] |
| det TRUNC-1-K8 - det CRITIC | -0.0045 [-0.0114, +0.0023] |
| det TRUNC-1-K8 - det LEAF-K8 | -0.0056 [-0.0219, +0.0097] |
| det TRUNC-2-K8 - det CRITIC | -0.0050 [-0.0190, +0.0095] |
| det TRUNC-2-K8 - det LEAF-K8 | -0.0061 [-0.0197, +0.0073] |
| det TRUNC-4-K8 - det CRITIC | +0.0013 [-0.0129, +0.0168] |
| det TRUNC-4-K8 - det LEAF-K8 | +0.0002 [-0.0135, +0.0133] |
| det CV-K8 - det LEAF-K8 | -0.0066 [-0.0147, +0.0012] |
| retained share, LEAF-K1 (det / oracle) | 0.050 [-0.3815, 0.3511] |
| retained share, LEAF-K4 (det / oracle) | 0.001 [-0.2421, 0.1991] |
| retained share, LEAF-K8 (det / oracle) | 0.090 [-0.1113, 0.2539] |

| unrevealed opponent slots | n | det LEAF-K8 | oracle LEAF-K8 | det CRITIC | oracle − det |
|---|---|---|---|---|---|
| unrev_0-1 | 220 | +0.0266 | +0.0614 | -0.0003 | +0.0348 [0.01773, 0.05321] |
| unrev_2-3 | 199 | +0.0034 | +0.0808 | +0.0195 | +0.0775 [0.05241, 0.10362] |
| unrev_4-5 | 179 | -0.0148 | +0.0745 | -0.0034 | +0.0893 [0.06574, 0.11499] |

Worlds: {'tries_mean': 1.765, 'rewritten_moves': 250, 'rewritten_switches': 1, 'ambiguous_moves_kept': 0, 'root_choice_kept': 0, 'new_slots_mean': 2.328, 'redrawn_move_slots_mean': 9.801}

