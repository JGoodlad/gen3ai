
### `cond.opp_class_auc.t4_10` — DECISION

| vs 10M | Δ draw1 | CI draw1 | Δ draw2 | CI draw2 | W(s) = |Δ₁−Δ₂| |
|---|---|---|---|---|---|
| 20M | -0.0391 | [-0.0574, -0.0211] | -0.0127 | [-0.0316, +0.0064] | 0.0264 |
| 40M | -0.0529 | [-0.0718, -0.0342] | -0.0277 | [-0.0476, -0.0073] | 0.0251 |
| 73M | -0.0596 | [-0.0787, -0.0394] | -0.0372 | [-0.0575, -0.0172] | 0.0224 |

| consecutive gap | draw1 | draw2 | bar = max(W) over the pair | clears bar (|Δ| > bar) on BOTH draws? |
|---|---|---|---|---|
| 10M → 20M | -0.0391 | -0.0127 | 0.0264 | no |
| 20M → 40M | -0.0138 | -0.0150 | 0.0264 | no |
| 40M → 73M | -0.0067 | -0.0094 | 0.0251 | no |

cond.opp_class_auc.t4_10 ⇒ **UNDECIDED**  (max W = 0.0264; Δ(10M→73M) = -0.0596 / -0.0372)
  · control-level wobble across pairs (quota match): draw1 0.0018, draw2 0.0017

### `gate.resolution.bot` — DECISION

| vs 10M | Δ draw1 | CI draw1 | Δ draw2 | CI draw2 | W(s) = |Δ₁−Δ₂| |
|---|---|---|---|---|---|
| 20M | -0.0058 | [-0.0119, +0.0004] | -0.0032 | [-0.0098, +0.0035] | 0.0026 |
| 40M | -0.0041 | [-0.0094, +0.0015] | +0.0028 | [-0.0048, +0.0099] | 0.0069 |
| 73M | -0.0004 | [-0.0065, +0.0060] | -0.0010 | [-0.0067, +0.0044] | 0.0006 |

| consecutive gap | draw1 | draw2 | bar = max(W) over the pair | clears bar (|Δ| > bar) on BOTH draws? |
|---|---|---|---|---|
| 10M → 20M | -0.0058 | -0.0032 | 0.0026 | **YES — DOWN on both** |
| 20M → 40M | +0.0016 | +0.0060 | 0.0069 | no |
| 40M → 73M | +0.0037 | -0.0038 | 0.0069 | no |

gate.resolution.bot ⇒ **(ii) SUPPORTED** — flat: |Δ(10M→73M)| within max W on both draws and both CIs cover zero  (max W = 0.0069; Δ(10M→73M) = -0.0004 / -0.0010)
  · control-level wobble across pairs (quota match): draw1 0.0000, draw2 0.0000

### `gate.resolution.all` — DECISION

| vs 10M | Δ draw1 | CI draw1 | Δ draw2 | CI draw2 | W(s) = |Δ₁−Δ₂| |
|---|---|---|---|---|---|
| 20M | -0.0077 | [-0.0162, +0.0007] | -0.0099 | [-0.0185, -0.0007] | 0.0022 |
| 40M | -0.0109 | [-0.0189, -0.0043] | -0.0115 | [-0.0201, -0.0035] | 0.0006 |
| 73M | -0.0114 | [-0.0201, -0.0034] | -0.0207 | [-0.0295, -0.0130] | 0.0093 |

| consecutive gap | draw1 | draw2 | bar = max(W) over the pair | clears bar (|Δ| > bar) on BOTH draws? |
|---|---|---|---|---|
| 10M → 20M | -0.0077 | -0.0099 | 0.0022 | **YES — DOWN on both** |
| 20M → 40M | -0.0032 | -0.0016 | 0.0022 | no |
| 40M → 73M | -0.0005 | -0.0092 | 0.0093 | no |

gate.resolution.all ⇒ **UNDECIDED**  (max W = 0.0093; Δ(10M→73M) = -0.0114 / -0.0207)
  · control-level wobble across pairs (quota match): draw1 0.0000, draw2 0.0000

### `cond.opp_class_auc.t1_3` — DECISION

| vs 10M | Δ draw1 | CI draw1 | Δ draw2 | CI draw2 | W(s) = |Δ₁−Δ₂| |
|---|---|---|---|---|---|
| 20M | -0.0233 | [-0.0467, +0.0002] | -0.0200 | [-0.0424, +0.0037] | 0.0033 |
| 40M | -0.0267 | [-0.0521, -0.0020] | -0.0113 | [-0.0342, +0.0131] | 0.0154 |
| 73M | +0.0024 | [-0.0197, +0.0257] | -0.0061 | [-0.0309, +0.0181] | 0.0085 |

| consecutive gap | draw1 | draw2 | bar = max(W) over the pair | clears bar (|Δ| > bar) on BOTH draws? |
|---|---|---|---|---|
| 10M → 20M | -0.0233 | -0.0200 | 0.0033 | **YES — DOWN on both** |
| 20M → 40M | -0.0034 | +0.0087 | 0.0154 | no |
| 40M → 73M | +0.0291 | +0.0051 | 0.0154 | no |

cond.opp_class_auc.t1_3 ⇒ **(ii) SUPPORTED** — flat: |Δ(10M→73M)| within max W on both draws and both CIs cover zero  (max W = 0.0154; Δ(10M→73M) = +0.0024 / -0.0061)
  · control-level wobble across pairs (quota match): draw1 0.0048, draw2 0.0007

### `identity.resolution_cap_share` — REPORTED — resolution as a SHARE of its base-rate cap

| vs 10M | Δ draw1 | CI draw1 | Δ draw2 | CI draw2 | W(s) = |Δ₁−Δ₂| |
|---|---|---|---|---|---|
| 20M | -0.0581 | [-0.1331, +0.0166] | -0.0883 | [-0.1627, -0.0070] | 0.0301 |
| 40M | -0.0301 | [-0.1073, +0.0477] | -0.0122 | [-0.0886, +0.0665] | 0.0180 |
| 73M | -0.0640 | [-0.1345, +0.0093] | -0.0552 | [-0.1337, +0.0228] | 0.0088 |

| consecutive gap | draw1 | draw2 | bar = max(W) over the pair | clears bar (|Δ| > bar) on BOTH draws? |
|---|---|---|---|---|
| 10M → 20M | -0.0581 | -0.0883 | 0.0301 | **YES — DOWN on both** |
| 20M → 40M | +0.0280 | +0.0761 | 0.0301 | no |
| 40M → 73M | -0.0339 | -0.0430 | 0.0180 | **YES — DOWN on both** |

identity.resolution_cap_share ⇒ **UNDECIDED**  (max W = 0.0301; Δ(10M→73M) = -0.0640 / -0.0552)
  · control-level wobble across pairs (quota match): draw1 0.0000, draw2 0.0000

### `gate.skill.bot` — REPORTED

| vs 10M | Δ draw1 | CI draw1 | Δ draw2 | CI draw2 | W(s) = |Δ₁−Δ₂| |
|---|---|---|---|---|---|
| 20M | -0.0641 | [-0.1118, -0.0164] | -0.0492 | [-0.1041, +0.0078] | 0.0149 |
| 40M | -0.0708 | [-0.1188, -0.0186] | -0.0121 | [-0.0761, +0.0490] | 0.0587 |
| 73M | -0.0424 | [-0.1030, +0.0094] | -0.0505 | [-0.1010, -0.0008] | 0.0081 |

| consecutive gap | draw1 | draw2 | bar = max(W) over the pair | clears bar (|Δ| > bar) on BOTH draws? |
|---|---|---|---|---|
| 10M → 20M | -0.0641 | -0.0492 | 0.0149 | **YES — DOWN on both** |
| 20M → 40M | -0.0067 | +0.0371 | 0.0587 | no |
| 40M → 73M | +0.0284 | -0.0384 | 0.0587 | no |

gate.skill.bot ⇒ **UNDECIDED**  (max W = 0.0587; Δ(10M→73M) = -0.0424 / -0.0505)
  · control-level wobble across pairs (quota match): draw1 0.0000, draw2 0.0000

### `cond.own_team_r2.t1` — PROVISIONAL (rule 18)

| vs 10M | Δ draw1 | CI draw1 | Δ draw2 | CI draw2 | W(s) = |Δ₁−Δ₂| |
|---|---|---|---|---|---|
| 20M | +0.0118 | [+0.0017, +0.0217] | -0.0029 | [-0.0134, +0.0083] | 0.0146 |
| 40M | +0.0080 | [-0.0002, +0.0168] | -0.0064 | [-0.0165, +0.0035] | 0.0143 |
| 73M | +0.0081 | [-0.0014, +0.0174] | -0.0055 | [-0.0151, +0.0042] | 0.0136 |

| consecutive gap | draw1 | draw2 | bar = max(W) over the pair | clears bar (|Δ| > bar) on BOTH draws? |
|---|---|---|---|---|
| 10M → 20M | +0.0118 | -0.0029 | 0.0146 | no |
| 20M → 40M | -0.0038 | -0.0035 | 0.0146 | no |
| 40M → 73M | +0.0001 | +0.0009 | 0.0143 | no |

cond.own_team_r2.t1 ⇒ **(ii) SUPPORTED** — flat: |Δ(10M→73M)| within max W on both draws and both CIs cover zero  (max W = 0.0146; Δ(10M→73M) = +0.0081 / -0.0055)
  · control-level wobble across pairs (quota match): draw1 0.0009, draw2 0.0014

### `cond.spread_ratio.t1_3` — REPORTED (run-level floor)

| vs 10M | Δ draw1 | CI draw1 | Δ draw2 | CI draw2 | W(s) = |Δ₁−Δ₂| |
|---|---|---|---|---|---|
| 20M | +0.0082 | [-0.0111, +0.0355] | +0.0206 | [+0.0014, +0.0444] | 0.0124 |
| 40M | +0.0242 | [+0.0025, +0.0565] | +0.0528 | [+0.0291, +0.0829] | 0.0287 |
| 73M | +0.0537 | [+0.0314, +0.0814] | +0.0696 | [+0.0443, +0.1024] | 0.0160 |

| consecutive gap | draw1 | draw2 | bar = max(W) over the pair | clears bar (|Δ| > bar) on BOTH draws? |
|---|---|---|---|---|
| 10M → 20M | +0.0082 | +0.0206 | 0.0124 | no |
| 20M → 40M | +0.0159 | +0.0323 | 0.0287 | no |
| 40M → 73M | +0.0295 | +0.0168 | 0.0287 | no |

cond.spread_ratio.t1_3 ⇒ **UNDECIDED**  (max W = 0.0287; Δ(10M→73M) = +0.0537 / +0.0696)
  · control-level wobble across pairs (quota match): draw1 0.0000, draw2 0.0000

### `cond.spread_ratio.t4_10` — REPORTED (run-level floor)

| vs 10M | Δ draw1 | CI draw1 | Δ draw2 | CI draw2 | W(s) = |Δ₁−Δ₂| |
|---|---|---|---|---|---|
| 20M | +0.0625 | [+0.0209, +0.1029] | +0.0818 | [+0.0433, +0.1217] | 0.0193 |
| 40M | +0.0898 | [+0.0468, +0.1365] | +0.1401 | [+0.0961, +0.1885] | 0.0503 |
| 73M | +0.1151 | [+0.0662, +0.1652] | +0.1843 | [+0.1314, +0.2428] | 0.0692 |

| consecutive gap | draw1 | draw2 | bar = max(W) over the pair | clears bar (|Δ| > bar) on BOTH draws? |
|---|---|---|---|---|
| 10M → 20M | +0.0625 | +0.0818 | 0.0193 | **YES — UP on both** |
| 20M → 40M | +0.0273 | +0.0583 | 0.0503 | no |
| 40M → 73M | +0.0253 | +0.0442 | 0.0692 | no |

cond.spread_ratio.t4_10 ⇒ **UNDECIDED**  (max W = 0.0692; Δ(10M→73M) = +0.1151 / +0.1843)
  · control-level wobble across pairs (quota match): draw1 0.0000, draw2 0.0000

### `identity.bias.ALL` — REPORTED (run-level, rule 19)

| vs 10M | Δ draw1 | CI draw1 | Δ draw2 | CI draw2 | W(s) = |Δ₁−Δ₂| |
|---|---|---|---|---|---|
| 20M | -0.0426 | [-0.0685, -0.0175] | -0.0230 | [-0.0489, +0.0029] | 0.0196 |
| 40M | -0.0745 | [-0.0999, -0.0499] | -0.0680 | [-0.0923, -0.0439] | 0.0065 |
| 73M | -0.1063 | [-0.1307, -0.0827] | -0.0891 | [-0.1132, -0.0654] | 0.0171 |

| consecutive gap | draw1 | draw2 | bar = max(W) over the pair | clears bar (|Δ| > bar) on BOTH draws? |
|---|---|---|---|---|
| 10M → 20M | -0.0426 | -0.0230 | 0.0196 | **YES — DOWN on both** |
| 20M → 40M | -0.0320 | -0.0451 | 0.0196 | **YES — DOWN on both** |
| 40M → 73M | -0.0317 | -0.0211 | 0.0171 | **YES — DOWN on both** |

identity.bias.ALL ⇒ **UNDECIDED**  (max W = 0.0196; Δ(10M→73M) = -0.1063 / -0.0891)
  · control-level wobble across pairs (quota match): draw1 0.0000, draw2 0.0000

### `identity.bias.late (turn>=25)` — REPORTED (run-level, rule 19)

| vs 10M | Δ draw1 | CI draw1 | Δ draw2 | CI draw2 | W(s) = |Δ₁−Δ₂| |
|---|---|---|---|---|---|
| 20M | -0.0316 | [-0.0828, +0.0176] | -0.0137 | [-0.0696, +0.0408] | 0.0179 |
| 40M | -0.0422 | [-0.0924, +0.0074] | -0.0449 | [-0.0951, +0.0039] | 0.0028 |
| 73M | -0.0745 | [-0.1251, -0.0252] | -0.0510 | [-0.1012, -0.0016] | 0.0235 |

| consecutive gap | draw1 | draw2 | bar = max(W) over the pair | clears bar (|Δ| > bar) on BOTH draws? |
|---|---|---|---|---|
| 10M → 20M | -0.0316 | -0.0137 | 0.0179 | no |
| 20M → 40M | -0.0106 | -0.0312 | 0.0179 | no |
| 40M → 73M | -0.0323 | -0.0060 | 0.0235 | no |

identity.bias.late (turn>=25) ⇒ **UNDECIDED**  (max W = 0.0235; Δ(10M→73M) = -0.0745 / -0.0510)
  · control-level wobble across pairs (quota match): draw1 0.0000, draw2 0.0000

### `gate.ece.all` — REPORTED

| vs 10M | Δ draw1 | CI draw1 | Δ draw2 | CI draw2 | W(s) = |Δ₁−Δ₂| |
|---|---|---|---|---|---|
| 20M | -0.0306 | [-0.0487, -0.0136] | -0.0288 | [-0.0466, -0.0091] | 0.0018 |
| 40M | -0.0598 | [-0.0787, -0.0410] | -0.0702 | [-0.0883, -0.0517] | 0.0103 |
| 73M | -0.0853 | [-0.0989, -0.0689] | -0.0909 | [-0.1047, -0.0739] | 0.0056 |

| consecutive gap | draw1 | draw2 | bar = max(W) over the pair | clears bar (|Δ| > bar) on BOTH draws? |
|---|---|---|---|---|
| 10M → 20M | -0.0306 | -0.0288 | 0.0018 | **YES — DOWN on both** |
| 20M → 40M | -0.0293 | -0.0414 | 0.0103 | **YES — DOWN on both** |
| 40M → 73M | -0.0255 | -0.0208 | 0.0103 | **YES — DOWN on both** |

gate.ece.all ⇒ **UNDECIDED**  (max W = 0.0103; Δ(10M→73M) = -0.0853 / -0.0909)
  · control-level wobble across pairs (quota match): draw1 0.0000, draw2 0.0000

### `cond.calibration_slope.all` — REPORTED (run-level)

| vs 10M | Δ draw1 | CI draw1 | Δ draw2 | CI draw2 | W(s) = |Δ₁−Δ₂| |
|---|---|---|---|---|---|
| 20M | -0.1408 | [-0.2248, -0.0586] | -0.1739 | [-0.2571, -0.0883] | 0.0331 |
| 40M | -0.0593 | [-0.1490, +0.0309] | -0.0624 | [-0.1516, +0.0238] | 0.0032 |
| 73M | +0.0637 | [-0.0329, +0.1612] | +0.0303 | [-0.0738, +0.1259] | 0.0334 |

| consecutive gap | draw1 | draw2 | bar = max(W) over the pair | clears bar (|Δ| > bar) on BOTH draws? |
|---|---|---|---|---|
| 10M → 20M | -0.1408 | -0.1739 | 0.0331 | **YES — DOWN on both** |
| 20M → 40M | +0.0815 | +0.1115 | 0.0331 | **YES — UP on both** |
| 40M → 73M | +0.1229 | +0.0927 | 0.0334 | **YES — UP on both** |

cond.calibration_slope.all ⇒ **UNDECIDED**  (max W = 0.0334; Δ(10M→73M) = +0.0637 / +0.0303)
  · control-level wobble across pairs (quota match): draw1 0.0000, draw2 0.0000
wrote /home/goodlad/.claude/jobs/9ab51de6/tmp/stepcurve/stepcurve_table.json

