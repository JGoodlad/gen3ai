### Headline — sibling-control R, v2 beside v1

| arm | reference | v1 R (n=9) | **v2 R (n=9)** | v2 R (n=3) |
|---|---|---|---|---|
| v8 (3 teachers) | parent = origin | 1.4498 [1.2728, 1.6722] | **1.8316 [1.5334, 2.1744]** | 1.7940 [1.4803, 2.1551] |
| gen unfunded R5F (8) | REF-A fold parent | 1.0723 [0.9803, 1.1634] | **1.0722 [0.9432, 1.1977]** | 1.0956 [0.9208, 1.2971] |
| gen funded R5FUND (8) | REF-A fold parent | 1.1016 [1.0008, 1.1987] | **1.1067 [1.0026, 1.2071]** | 1.1013 [0.9604, 1.2441] |
| gen unfunded R5F (8) | REF-B true origin | — | **1.2542 [1.0318, 1.4663]** | 1.2961 [1.0439, 1.5443] |
| gen funded R5FUND (8) | REF-B true origin | — | **1.1953 [1.0972, 1.2983]** | 1.1859 [1.0794, 1.2859] |

### Absolute levels (n=9)

| half | reference | KL on own taught | KL on untaught 8 | raw L |
|---|---|---|---|---|
| gen unfunded | REF-A | 0.5789 [0.4908, 0.6741] | 0.5536 [0.5030, 0.6034] | 1.0502 [0.8998, 1.2137] |
| gen funded | REF-A | 0.7613 [0.6756, 0.8555] | 0.6957 [0.6400, 0.7546] | 1.0971 [0.9900, 1.1912] |
| gen unfunded | REF-B | 0.3433 [0.2639, 0.4186] | 0.2576 [0.2031, 0.3045] | 1.3306 [1.1904, 1.4944] |
| gen funded | REF-B | 0.5317 [0.4800, 0.5959] | 0.4160 [0.3795, 0.4576] | 1.2808 [1.1950, 1.3697] |
| v8 (all 3) | parent = origin | 0.3969 (own, sibling control) | — | 1.6202 [1.2922, 2.0816] |

### Contrasts (n=9)

| contrast | reference | delta | CI95 | verdict |
|---|---|---|---|---|
| v8 − gen unfunded (R, unpaired) | REF-A | +0.7594 | [0.4312, 1.1236] | SIGNIFICANT |
| v8 − gen funded (R, unpaired) | REF-A | +0.7249 | [0.4111, 1.0829] | SIGNIFICANT |
| gen funded − unfunded (R, paired) | REF-A | +0.0345 | [-0.0795, 0.1636] | NOT DETECTED |
| v8 − gen unfunded (R, unpaired) | REF-B | +0.5774 | [0.2053, 0.9832] | SIGNIFICANT |
| v8 − gen funded (R, unpaired) | REF-B | +0.6363 | [0.3196, 0.9919] | SIGNIFICANT |
| gen funded − unfunded (R, paired) | REF-B | -0.0589 | [-0.2508, 0.1472] | NOT DETECTED |

### Matched-noise floor (n=9)

| era | pair | reference | KL untaught | KL taught | floor L |
|---|---|---|---|---|---|
| gen | `FLOORA_ckptA` | REF-A | 0.0374 | 0.0401 | 1.0725 |
| gen | `FLOORA_ckptB` | REF-A | 0.0654 | 0.0760 | 1.1620 |
| gen | `FLOORB_ckptA` | REF-B | 0.0112 | 0.0133 | 1.1894 |
| gen | `FLOORB_ckptB` | REF-B | 0.0415 | 0.0453 | 1.0923 |
| v8 | `FLOOR_c277178` | parent = origin | 0.0383 | 0.0263 | 0.6878 |
| v8 | `FLOOR_c275758` | parent = origin | 0.0664 | 0.0535 | 0.8053 |

### v8 per-teacher, resolved vs what v1 scored (n=9)

| teacher | n taught | v1 KL untaught | **v2 KL untaught** | v1 L | **v2 L** |
|---|---|---|---|---|---|
| `pool10` | 10 | 0.3223 | **0.3176** | 1.4643 | **1.4869** |
| `semistall3` | 3 | 0.2190 | **0.1036** | 2.0450 | **2.0816** |
| `defensive10` | 10 | 0.2807 | **0.2775** | 1.2571 | **1.2922** |
| **pooled** | 23 cells | 0.2740 | **0.2329** | 1.5220 [1.3551, 1.7205] | **1.6718 [1.4567, 1.9161]** |

### gen per-teacher untaught KL, v1 (final_model) → v2 (best_model), REF-A, n=9

| teacher | v1 | **v2** | Δ |
|---|---|---|---|
| `FUND00` | 0.8239 | **0.8324** | +0.0084 |
| `FUND02` | 0.5944 | **0.5961** | +0.0017 |
| `FUND04` | 0.7063 | **0.7180** | +0.0117 |
| `FUND06` | 0.7810 | **0.7876** | +0.0067 |
| `FUND08` | 0.6374 | **0.6280** | -0.0094 |
| `FUND10` | 0.6157 | **0.6096** | -0.0061 |
| `FUND12` | 0.6636 | **0.6417** | -0.0219 |
| `FUND14` | 0.7527 | **0.7523** | -0.0004 |
| `UNF00` | 0.5759 | **0.5669** | -0.0089 |
| `UNF02` | 0.5379 | **0.4242** | -0.1136 |
| `UNF04` | 0.6921 | **0.6481** | -0.0439 |
| `UNF06` | 0.6474 | **0.5148** | -0.1326 |
| `UNF08` | 0.5063 | **0.4956** | -0.0107 |
| `UNF10` | 0.5776 | **0.5690** | -0.0086 |
| `UNF12` | 0.5678 | **0.5504** | -0.0173 |
| `UNF14` | 0.6868 | **0.6595** | -0.0273 |
