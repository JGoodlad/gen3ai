# CRITIC GATE — `ai_v12_02_winprob_critic` vs `ai_v9_59_R2ACTION_0827`

The pre-registered read of `designs/ai_v12/design_winprob_only_critic.md` §5.5 (endpoints) / §4.3 (bars). Generated 2026-09-07T13:07:57.

| input | spec | resolved file | rung |
|---|---|---|---|
| run | `ai_v12_02_winprob_critic` | `/home/goodlad/dev/gen3ai/models/ai_v12_02_winprob_critic/checkpoints/checkpoint_34840320_steps.zip` | `latest_txt` |
| parent | `v9_fold_parent` | `/home/goodlad/dev/gen3ai/models/ai_v9_59_R2ACTION_0827/final_model.zip` | `explicit_zip` |
| control | `models/ai_v9_195_G5PLAINA_0906` | `/home/goodlad/dev/gen3ai/models/ai_v9_195_G5PLAINA_0906/final_model.zip` | `latest_txt` |
| control | `models/ai_v9_196_G5PLAINB_0906` | `/home/goodlad/dev/gen3ai/models/ai_v9_196_G5PLAINB_0906/final_model.zip` | `latest_txt` |
| control | `models/ai_v9_197_G5PLAINC_0906` | `/home/goodlad/dev/gen3ai/models/ai_v9_197_G5PLAINC_0906/final_model.zip` | `latest_txt` |

## VERDICT — **KILL**

> G7 breached: ep_len_bots 23.79 is 1.30x the era's 18.28 (> 1.25); ep_len_bots 23.49 is 1.28x the era's 18.28 (> 1.25)

## 1. Anchored ladder, at matched SNAPSHOT COUNT

**rating final — the run wrote a final model**

| # | ai_v12_02_winprob_critic step | elo ±95% | ai_v9_59_R2ACTION_0827 step | elo ±95% |
|---|---|---|---|---|
| 1 | 4,000,032 | 1736 ± 19 | 2,000,016 | 1588 ± 81 |
| 2 | 6,000,000 | 1866 ± 19 | 4,000,032 | 1837 ± 57 |
| 3 | 8,000,016 | 1920 ± 19 | 6,000,000 | 1817 ± 58 |
| 4 | 10,000,032 | 1941 ± 19 | 8,000,016 | 1888 ± 55 |
| 5 | 12,000,000 | 1966 ± 19 | 10,000,032 | 1913 ± 55 |
| 6 | 14,000,016 | 1991 ± 20 | 12,000,000 | 1916 ± 55 |
| 7 | 16,000,032 | 2004 ± 20 | 14,000,016 | 1965 ± 55 |
| 8 | 18,000,000 | 2038 ± 20 | 16,000,032 | 1937 ± 55 |
| 9 | 20,000,016 | 1993 ± 20 | 18,000,000 | 1962 ± 55 |
| 10 | 22,000,032 | 2005 ± 20 | 20,000,016 | 1934 ± 74 |
| 11 | 24,000,000 | 2011 ± 20 | 22,000,032 | 1934 ± 74 |
| 12 | 26,000,016 | 2032 ± 20 | 24,000,000 | 1983 ± 74 |
| 13 | 28,000,032 | 2040 ± 20 | 26,000,016 | 1948 ± 30 |
| 14 | 30,000,000 | 2034 ± 20 | 28,000,032 | 1947 ± 30 |

**Δ at 14 snapshots: +87 ELO [+50, +123]** — matched SNAPSHOT COUNT, never matched step. Both fits are anchored to the same pinned bots, so the delta is meaningful; its SE combines two independent fits and carries NO term for the anchor uncertainty they share.

Fit size: BOTH sides were REFIT from their own games.jsonl on their first n snapshots (strict: every snapshot endpoint inside the prefix) with THIS tree's fit_ladder — because the newest node of a longer fit is inflated, AND because a committed ladder.json written before 2026-09-07 folded the eval cycles' greedy-vs-stochastic SENTINEL edges into the fit (+8.9 pp to the newer snapshot; +21..+29 Elo on the newest nodes) — refit: run=True, parent=True; the committed-ladder delta would have read +91.

## 2. Calibration gate (G1–G4) — RESOLUTION is primary

Bars read from `/home/goodlad/dev/gen3ai-wt-ladderfix/designs/research_state/measurements/winprob_critic_baseline_2026-09-06/selection_reweighted.json` (`ai_v9_59_R2ACTION_0827`, steps [26000016, 28000032]; G1 reduce=`max`, G2/G3 reduce=`last`), selection-reweighted. the committed baseline publishes no interval for `resolution`, so G1 compares the ARM's cluster CI against the baseline as a FIXED bar. G2/G3 inherit the same asymmetry: the baseline value they are compared against is a point, and only the ARM carries an interval.

### G1 (resolution, primary) + G4 (skill)

| step | stratum | gated | **resolution** [95% CI] | baseline | Δ | skill [95% CI] | G1 | G4 |
|---|---|---|---|---|---|---|---|---|
| 2,000,016 | `all` | no | **0.0792** [0.0514, 0.1071] | 0.0618 | +0.0174 | +0.314 [+0.173, +0.420] | ❌ | ✅ |
| 2,000,016 | `bot` | yes | **0.0792** [0.0514, 0.1071] | 0.0337 | +0.0455 | +0.314 [+0.195, +0.425] | ✅ | ✅ |
| 4,000,032 | `all` | no | **0.0314** [0.0176, 0.0537] | 0.0618 | -0.0305 | +0.170 [+0.002, +0.312] | ❌ | ✅ |
| 4,000,032 | `bot` | yes | **0.0314** [0.0176, 0.0537] | 0.0337 | -0.0023 | +0.170 [+0.015, +0.312] | ❌ | ✅ |
| 6,000,000 | `all` | no | **0.0378** [0.0226, 0.0576] | 0.0618 | -0.0240 | +0.267 [+0.192, +0.336] | ❌ | ✅ |
| 6,000,000 | `bot` | yes | **0.0346** [0.0203, 0.0571] | 0.0337 | +0.0009 | +0.245 [+0.171, +0.323] | ❌ | ✅ |
| 6,000,000 | `pool` | yes | **0.0636** [0.0262, 0.1296] | 0.0711 | -0.0075 | +0.390 [+0.156, +0.532] | ❌ | ✅ |
| 8,000,016 | `all` | no | **0.0315** [0.0211, 0.0446] | 0.0618 | -0.0303 | +0.206 [+0.139, +0.262] | ❌ | ✅ |
| 8,000,016 | `bot` | yes | **0.0249** [0.0158, 0.0397] | 0.0337 | -0.0087 | +0.189 [+0.114, +0.255] | ❌ | ✅ |
| 8,000,016 | `pool` | yes | **0.0480** [0.0234, 0.0836] | 0.0711 | -0.0231 | +0.205 [+0.044, +0.331] | ❌ | ✅ |
| 10,000,032 | `all` | no | **0.0375** [0.0267, 0.0544] | 0.0618 | -0.0243 | +0.233 [+0.176, +0.287] | ❌ | ✅ |
| 10,000,032 | `bot` | yes | **0.0303** [0.0193, 0.0482] | 0.0337 | -0.0033 | +0.223 [+0.175, +0.266] | ❌ | ✅ |
| 10,000,032 | `pool` | yes | **0.0465** [0.0285, 0.0771] | 0.0711 | -0.0246 | +0.227 [+0.077, +0.324] | ❌ | ✅ |
| 12,000,000 | `all` | no | **0.0434** [0.0296, 0.0618] | 0.0618 | -0.0185 | +0.236 [+0.180, +0.286] | ❌ | ✅ |
| 12,000,000 | `bot` | yes | **0.0376** [0.0222, 0.0567] | 0.0337 | +0.0040 | +0.258 [+0.187, +0.331] | ❌ | ✅ |
| 12,000,000 | `pool` | yes | **0.0478** [0.0265, 0.0758] | 0.0711 | -0.0233 | +0.204 [+0.092, +0.285] | ❌ | ✅ |
| 14,000,016 | `all` | no | **0.0391** [0.0275, 0.0554] | 0.0618 | -0.0227 | +0.254 [+0.191, +0.307] | ❌ | ✅ |
| 14,000,016 | `bot` | yes | **0.0303** [0.0204, 0.0480] | 0.0337 | -0.0033 | +0.255 [+0.198, +0.299] | ❌ | ✅ |
| 14,000,016 | `pool` | yes | **0.0460** [0.0260, 0.0769] | 0.0711 | -0.0251 | +0.236 [+0.139, +0.318] | ❌ | ✅ |
| 16,000,032 | `all` | no | **0.0316** [0.0181, 0.0469] | 0.0618 | -0.0303 | +0.186 [+0.114, +0.248] | ❌ | ✅ |
| 16,000,032 | `bot` | yes | **0.0162** [0.0087, 0.0284] | 0.0337 | -0.0175 | +0.166 [+0.064, +0.246] | ❌ | ✅ |
| 16,000,032 | `pool` | yes | **0.0391** [0.0215, 0.0659] | 0.0711 | -0.0320 | +0.134 [+0.006, +0.235] | ❌ | ✅ |
| 18,000,000 | `all` | no | **0.0336** [0.0221, 0.0529] | 0.0618 | -0.0282 | +0.213 [+0.128, +0.279] | ❌ | ✅ |
| 18,000,000 | `bot` | yes | **0.0197** [0.0128, 0.0320] | 0.0337 | -0.0139 | +0.169 [+0.076, +0.245] | ❌ | ✅ |
| 18,000,000 | `pool` | yes | **0.0406** [0.0226, 0.0688] | 0.0711 | -0.0305 | +0.196 [+0.072, +0.290] | ❌ | ✅ |
| 20,000,016 | `all` | no | **0.0411** [0.0261, 0.0644] | 0.0618 | -0.0208 | +0.168 [+0.054, +0.244] | ❌ | ✅ |
| 20,000,016 | `bot` | yes | **0.0144** [0.0087, 0.0240] | 0.0337 | -0.0192 | +0.131 [+0.047, +0.205] | ❌ | ✅ |
| 20,000,016 | `pool` | yes | **0.0515** [0.0295, 0.0844] | 0.0711 | -0.0196 | +0.098 [-0.105, +0.234] | ❌ | ❌ |
| 22,000,032 | `all` | no | **0.0383** [0.0246, 0.0598] | 0.0618 | -0.0236 | +0.201 [+0.136, +0.258] | ❌ | ✅ |
| 22,000,032 | `bot` | yes | **0.0095** [0.0040, 0.0238] | 0.0337 | -0.0241 | +0.029 [-0.180, +0.148] | ❌ | ❌ |
| 22,000,032 | `pool` | yes | **0.0691** [0.0432, 0.0991] | 0.0711 | -0.0020 | +0.220 [+0.097, +0.311] | ❌ | ✅ |
| 24,000,000 | `all` | no | **0.0425** [0.0278, 0.0629] | 0.0618 | -0.0193 | +0.218 [+0.148, +0.284] | ❌ | ✅ |
| 24,000,000 | `bot` | yes | **0.0273** [0.0187, 0.0394] | 0.0337 | -0.0063 | +0.192 [+0.119, +0.257] | ❌ | ✅ |
| 24,000,000 | `pool` | yes | **0.0527** [0.0315, 0.0830] | 0.0711 | -0.0184 | +0.185 [+0.065, +0.273] | ❌ | ✅ |
| 26,000,016 | `all` | no | **0.0378** [0.0193, 0.0636] | 0.0618 | -0.0241 | +0.206 [+0.085, +0.293] | ❌ | ✅ |
| 26,000,016 | `bot` | yes | **0.0141** [0.0079, 0.0275] | 0.0337 | -0.0196 | +0.114 [-0.035, +0.217] | ❌ | ❌ |
| 26,000,016 | `pool` | yes | **0.0508** [0.0248, 0.0912] | 0.0711 | -0.0203 | +0.197 [+0.022, +0.310] | ❌ | ✅ |
| 28,000,032 | `all` | no | **0.0294** [0.0169, 0.0467] | 0.0618 | -0.0325 | +0.168 [+0.088, +0.243] | ❌ | ✅ |
| 28,000,032 | `bot` | yes | **0.0115** [0.0064, 0.0228] | 0.0337 | -0.0222 | +0.112 [-0.014, +0.222] | ❌ | ❌ |
| 28,000,032 | `pool` | yes | **0.0343** [0.0179, 0.0594] | 0.0711 | -0.0368 | +0.122 [-0.007, +0.216] | ❌ | ❌ |
| 30,000,000 | `all` | no | **0.0476** [0.0339, 0.0654] | 0.0618 | -0.0143 | +0.274 [+0.199, +0.331] | ❌ | ✅ |
| 30,000,000 | `bot` | yes | **0.0229** [0.0127, 0.0399] | 0.0337 | -0.0108 | +0.197 [+0.067, +0.297] | ❌ | ✅ |
| 30,000,000 | `pool` | yes | **0.0587** [0.0389, 0.0852] | 0.0711 | -0.0124 | +0.241 [+0.139, +0.311] | ❌ | ✅ |
| 32,000,016 | `all` | no | **0.0425** [0.0297, 0.0589] | 0.0618 | -0.0193 | +0.238 [+0.154, +0.316] | ❌ | ✅ |
| 32,000,016 | `bot` | yes | **0.0233** [0.0139, 0.0373] | 0.0337 | -0.0104 | +0.172 [+0.049, +0.272] | ❌ | ✅ |
| 32,000,016 | `pool` | yes | **0.0491** [0.0294, 0.0790] | 0.0711 | -0.0220 | +0.219 [+0.087, +0.319] | ❌ | ✅ |
| 34,000,032 | `all` | no | **0.0265** [0.0181, 0.0397] | 0.0618 | -0.0354 | +0.157 [+0.092, +0.220] | ❌ | ✅ |
| 34,000,032 | `bot` | yes | **0.0120** [0.0066, 0.0198] | 0.0337 | -0.0217 | +0.107 [+0.032, +0.177] | ❌ | ✅ |
| 34,000,032 | `pool` | yes | **0.0313** [0.0184, 0.0543] | 0.0711 | -0.0398 | +0.122 [+0.005, +0.217] | ❌ | ✅ |

### G2 / G3 — PER-STRATUM NON-INFERIORITY vs the same-stratum baseline

> OWNER RULING 2026-09-06: §4.3's absolute G2 (reliability <= 0.005) and G3 (ECE <= 0.05) bars are ALREADY BREACHED by the committed baseline on the POOL stratum (reliability 0.0064 / 0.0103, ECE 0.0667 / 0.0875 at 26M / 28M), while §4.3 called G3 a 'no-regression clause' the reweighted baseline already passes — true pooled and on bot, FALSE on pool, and both gates are registered over 'both classes'. As written the arm had to clear a bar its predecessor never cleared. G2 and G3 are therefore PER-STRATUM RELATIVE bars: no worse than the baseline's SAME-stratum value. The absolute numbers stay printed as aspirational targets.

Rule: PASS if the arm's point estimate is <= the baseline's same-stratum value, OR the arm's cluster-bootstrap CI CONTAINS that value (non-inferiority — never a direction claim). FAIL only when the arm's whole CI sits ABOVE it. The baseline value is its own row **at the matched checkpoint** where the artifact carries that step, else its steps reduced with `last` — the cell says which.

| step | stratum | gated | gate | metric | arm [95% CI] | baseline (step) | Δ | verdict | **decided by** | §4.3 absolute (aspirational) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2,000,016 | `all` | no | G2 | reliability | **0.0014** [0.0010, 0.0165] | 0.0020 (@28,000,032, reduced `last`) | -0.0006 | ✅ | `point <= base` | ≤ 0.005 — arm meets, baseline meets |
| 2,000,016 | `all` | no | G3 | ECE | **0.0337** [0.0263, 0.1007] | 0.0349 (@28,000,032, reduced `last`) | -0.0013 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline meets |
| 2,000,016 | `bot` | yes | G2 | reliability | **0.0014** [0.0010, 0.0165] | 0.0012 (@28,000,032, reduced `last`) | +0.0002 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 2,000,016 | `bot` | yes | G3 | ECE | **0.0337** [0.0263, 0.1007] | 0.0228 (@28,000,032, reduced `last`) | +0.0109 | ❌ | `CI above base` | ≤ 0.05 — arm meets, baseline meets |
| 4,000,032 | `all` | no | G2 | reliability | **0.0069** [0.0019, 0.0201] | 0.0020 (@28,000,032, reduced `last`) | +0.0049 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline meets |
| 4,000,032 | `all` | no | G3 | ECE | **0.0483** [0.0284, 0.0926] | 0.0349 (@28,000,032, reduced `last`) | +0.0134 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 4,000,032 | `bot` | yes | G2 | reliability | **0.0069** [0.0019, 0.0201] | 0.0012 (@28,000,032, reduced `last`) | +0.0057 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 4,000,032 | `bot` | yes | G3 | ECE | **0.0483** [0.0284, 0.0926] | 0.0228 (@28,000,032, reduced `last`) | +0.0255 | ❌ | `CI above base` | ≤ 0.05 — arm meets, baseline meets |
| 6,000,000 | `all` | no | G2 | reliability | **0.0021** [0.0007, 0.0116] | 0.0020 (@28,000,032, reduced `last`) | +0.0001 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 6,000,000 | `all` | no | G3 | ECE | **0.0354** [0.0164, 0.0779] | 0.0349 (@28,000,032, reduced `last`) | +0.0005 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 6,000,000 | `bot` | yes | G2 | reliability | **0.0021** [0.0009, 0.0129] | 0.0012 (@28,000,032, reduced `last`) | +0.0008 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 6,000,000 | `bot` | yes | G3 | ECE | **0.0383** [0.0164, 0.0854] | 0.0228 (@28,000,032, reduced `last`) | +0.0155 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 6,000,000 | `pool` | yes | G2 | reliability | **0.0106** [0.0061, 0.0604] | 0.0103 (@28,000,032, reduced `last`) | +0.0003 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 6,000,000 | `pool` | yes | G3 | ECE | **0.0334** [0.0306, 0.1548] | 0.0875 (@28,000,032, reduced `last`) | -0.0541 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline MISSES |
| 8,000,016 | `all` | no | G2 | reliability | **0.0030** [0.0017, 0.0086] | 0.0020 (@28,000,032, reduced `last`) | +0.0010 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 8,000,016 | `all` | no | G3 | ECE | **0.0435** [0.0280, 0.0749] | 0.0349 (@28,000,032, reduced `last`) | +0.0086 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 8,000,016 | `bot` | yes | G2 | reliability | **0.0021** [0.0014, 0.0079] | 0.0012 (@28,000,032, reduced `last`) | +0.0008 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 8,000,016 | `bot` | yes | G3 | ECE | **0.0380** [0.0258, 0.0703] | 0.0228 (@28,000,032, reduced `last`) | +0.0152 | ❌ | `CI above base` | ≤ 0.05 — arm meets, baseline meets |
| 8,000,016 | `pool` | yes | G2 | reliability | **0.0108** [0.0045, 0.0475] | 0.0103 (@28,000,032, reduced `last`) | +0.0005 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 8,000,016 | `pool` | yes | G3 | ECE | **0.0771** [0.0355, 0.1979] | 0.0875 (@28,000,032, reduced `last`) | -0.0104 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 10,000,032 | `all` | no | G2 | reliability | **0.0069** [0.0025, 0.0189] | 0.0020 (@28,000,032, reduced `last`) | +0.0048 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 10,000,032 | `all` | no | G3 | ECE | **0.0606** [0.0285, 0.1028] | 0.0349 (@28,000,032, reduced `last`) | +0.0257 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline meets |
| 10,000,032 | `bot` | yes | G2 | reliability | **0.0055** [0.0018, 0.0174] | 0.0012 (@28,000,032, reduced `last`) | +0.0043 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 10,000,032 | `bot` | yes | G3 | ECE | **0.0473** [0.0182, 0.0886] | 0.0228 (@28,000,032, reduced `last`) | +0.0245 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 10,000,032 | `pool` | yes | G2 | reliability | **0.0105** [0.0033, 0.0424] | 0.0103 (@28,000,032, reduced `last`) | +0.0002 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 10,000,032 | `pool` | yes | G3 | ECE | **0.0805** [0.0342, 0.1720] | 0.0875 (@28,000,032, reduced `last`) | -0.0070 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 12,000,000 | `all` | no | G2 | reliability | **0.0081** [0.0033, 0.0203] | 0.0020 (@28,000,032, reduced `last`) | +0.0061 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 12,000,000 | `all` | no | G3 | ECE | **0.0635** [0.0326, 0.1124] | 0.0349 (@28,000,032, reduced `last`) | +0.0286 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline meets |
| 12,000,000 | `bot` | yes | G2 | reliability | **0.0051** [0.0014, 0.0178] | 0.0012 (@28,000,032, reduced `last`) | +0.0039 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 12,000,000 | `bot` | yes | G3 | ECE | **0.0504** [0.0174, 0.1043] | 0.0228 (@28,000,032, reduced `last`) | +0.0276 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline meets |
| 12,000,000 | `pool` | yes | G2 | reliability | **0.0124** [0.0042, 0.0390] | 0.0103 (@28,000,032, reduced `last`) | +0.0021 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 12,000,000 | `pool` | yes | G3 | ECE | **0.0781** [0.0406, 0.1751] | 0.0875 (@28,000,032, reduced `last`) | -0.0094 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 14,000,016 | `all` | no | G2 | reliability | **0.0040** [0.0011, 0.0138] | 0.0020 (@28,000,032, reduced `last`) | +0.0020 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 14,000,016 | `all` | no | G3 | ECE | **0.0420** [0.0166, 0.0826] | 0.0349 (@28,000,032, reduced `last`) | +0.0071 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 14,000,016 | `bot` | yes | G2 | reliability | **0.0031** [0.0015, 0.0114] | 0.0012 (@28,000,032, reduced `last`) | +0.0018 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 14,000,016 | `bot` | yes | G3 | ECE | **0.0268** [0.0190, 0.0624] | 0.0228 (@28,000,032, reduced `last`) | +0.0040 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 14,000,016 | `pool` | yes | G2 | reliability | **0.0075** [0.0023, 0.0279] | 0.0103 (@28,000,032, reduced `last`) | -0.0028 | ✅ | `point <= base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 14,000,016 | `pool` | yes | G3 | ECE | **0.0635** [0.0241, 0.1355] | 0.0875 (@28,000,032, reduced `last`) | -0.0240 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 16,000,032 | `all` | no | G2 | reliability | **0.0035** [0.0011, 0.0124] | 0.0020 (@28,000,032, reduced `last`) | +0.0015 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 16,000,032 | `all` | no | G3 | ECE | **0.0535** [0.0244, 0.0972] | 0.0349 (@28,000,032, reduced `last`) | +0.0186 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline meets |
| 16,000,032 | `bot` | yes | G2 | reliability | **0.0009** [0.0006, 0.0061] | 0.0012 (@28,000,032, reduced `last`) | -0.0003 | ✅ | `point <= base` | ≤ 0.005 — arm meets, baseline meets |
| 16,000,032 | `bot` | yes | G3 | ECE | **0.0227** [0.0128, 0.0493] | 0.0228 (@28,000,032, reduced `last`) | -0.0001 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline meets |
| 16,000,032 | `pool` | yes | G2 | reliability | **0.0142** [0.0037, 0.0485] | 0.0103 (@28,000,032, reduced `last`) | +0.0039 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 16,000,032 | `pool` | yes | G3 | ECE | **0.1082** [0.0507, 0.1930] | 0.0875 (@28,000,032, reduced `last`) | +0.0207 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 18,000,000 | `all` | no | G2 | reliability | **0.0014** [0.0007, 0.0079] | 0.0020 (@28,000,032, reduced `last`) | -0.0007 | ✅ | `point <= base` | ≤ 0.005 — arm meets, baseline meets |
| 18,000,000 | `all` | no | G3 | ECE | **0.0317** [0.0159, 0.0730] | 0.0349 (@28,000,032, reduced `last`) | -0.0032 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline meets |
| 18,000,000 | `bot` | yes | G2 | reliability | **0.0030** [0.0019, 0.0076] | 0.0012 (@28,000,032, reduced `last`) | +0.0018 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 18,000,000 | `bot` | yes | G3 | ECE | **0.0340** [0.0209, 0.0597] | 0.0228 (@28,000,032, reduced `last`) | +0.0112 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 18,000,000 | `pool` | yes | G2 | reliability | **0.0046** [0.0010, 0.0275] | 0.0103 (@28,000,032, reduced `last`) | -0.0057 | ✅ | `point <= base` | ≤ 0.005 — arm meets, baseline MISSES |
| 18,000,000 | `pool` | yes | G3 | ECE | **0.0610** [0.0220, 0.1410] | 0.0875 (@28,000,032, reduced `last`) | -0.0265 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 20,000,016 | `all` | no | G2 | reliability | **0.0120** [0.0038, 0.0321] | 0.0020 (@28,000,032, reduced `last`) | +0.0100 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 20,000,016 | `all` | no | G3 | ECE | **0.0957** [0.0474, 0.1561] | 0.0349 (@28,000,032, reduced `last`) | +0.0608 | ❌ | `CI above base` | ≤ 0.05 — arm MISSES, baseline meets |
| 20,000,016 | `bot` | yes | G2 | reliability | **0.0019** [0.0014, 0.0063] | 0.0012 (@28,000,032, reduced `last`) | +0.0006 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 20,000,016 | `bot` | yes | G3 | ECE | **0.0278** [0.0178, 0.0611] | 0.0228 (@28,000,032, reduced `last`) | +0.0050 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 20,000,016 | `pool` | yes | G2 | reliability | **0.0310** [0.0101, 0.0810] | 0.0103 (@28,000,032, reduced `last`) | +0.0206 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 20,000,016 | `pool` | yes | G3 | ECE | **0.1596** [0.0765, 0.2590] | 0.0875 (@28,000,032, reduced `last`) | +0.0721 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 22,000,032 | `all` | no | G2 | reliability | **0.0040** [0.0011, 0.0184] | 0.0020 (@28,000,032, reduced `last`) | +0.0019 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 22,000,032 | `all` | no | G3 | ECE | **0.0527** [0.0235, 0.1130] | 0.0349 (@28,000,032, reduced `last`) | +0.0177 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline meets |
| 22,000,032 | `bot` | yes | G2 | reliability | **0.0072** [0.0023, 0.0201] | 0.0012 (@28,000,032, reduced `last`) | +0.0060 | ❌ | `CI above base` | ≤ 0.005 — arm MISSES, baseline meets |
| 22,000,032 | `bot` | yes | G3 | ECE | **0.0551** [0.0310, 0.0939] | 0.0228 (@28,000,032, reduced `last`) | +0.0323 | ❌ | `CI above base` | ≤ 0.05 — arm MISSES, baseline meets |
| 22,000,032 | `pool` | yes | G2 | reliability | **0.0224** [0.0045, 0.0576] | 0.0103 (@28,000,032, reduced `last`) | +0.0121 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 22,000,032 | `pool` | yes | G3 | ECE | **0.1197** [0.0472, 0.1922] | 0.0875 (@28,000,032, reduced `last`) | +0.0322 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 24,000,000 | `all` | no | G2 | reliability | **0.0061** [0.0019, 0.0183] | 0.0020 (@28,000,032, reduced `last`) | +0.0040 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline meets |
| 24,000,000 | `all` | no | G3 | ECE | **0.0664** [0.0298, 0.1156] | 0.0349 (@28,000,032, reduced `last`) | +0.0315 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline meets |
| 24,000,000 | `bot` | yes | G2 | reliability | **0.0042** [0.0023, 0.0117] | 0.0012 (@28,000,032, reduced `last`) | +0.0029 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 24,000,000 | `bot` | yes | G3 | ECE | **0.0421** [0.0230, 0.0878] | 0.0228 (@28,000,032, reduced `last`) | +0.0193 | ❌ | `CI above base` | ≤ 0.05 — arm meets, baseline meets |
| 24,000,000 | `pool` | yes | G2 | reliability | **0.0159** [0.0045, 0.0503] | 0.0103 (@28,000,032, reduced `last`) | +0.0056 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 24,000,000 | `pool` | yes | G3 | ECE | **0.1056** [0.0496, 0.1939] | 0.0875 (@28,000,032, reduced `last`) | +0.0181 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 26,000,016 | `all` | no | G2 | reliability | **0.0043** [0.0022, 0.0159] | 0.0013 (@26,000,016) | +0.0030 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 26,000,016 | `all` | no | G3 | ECE | **0.0529** [0.0287, 0.1006] | 0.0249 (@26,000,016) | +0.0280 | ❌ | `CI above base` | ≤ 0.05 — arm MISSES, baseline meets |
| 26,000,016 | `bot` | yes | G2 | reliability | **0.0022** [0.0014, 0.0098] | 0.0027 (@26,000,016) | -0.0005 | ✅ | `point <= base` | ≤ 0.005 — arm meets, baseline meets |
| 26,000,016 | `bot` | yes | G3 | ECE | **0.0355** [0.0174, 0.0755] | 0.0395 (@26,000,016) | -0.0040 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline meets |
| 26,000,016 | `pool` | yes | G2 | reliability | **0.0116** [0.0048, 0.0414] | 0.0064 (@26,000,016) | +0.0053 | ✅ | `CI covers base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 26,000,016 | `pool` | yes | G3 | ECE | **0.0861** [0.0461, 0.1717] | 0.0667 (@26,000,016) | +0.0194 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 28,000,032 | `all` | no | G2 | reliability | **0.0031** [0.0015, 0.0121] | 0.0020 (@28,000,032) | +0.0011 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 28,000,032 | `all` | no | G3 | ECE | **0.0474** [0.0308, 0.0971] | 0.0349 (@28,000,032) | +0.0125 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 28,000,032 | `bot` | yes | G2 | reliability | **0.0019** [0.0008, 0.0085] | 0.0012 (@28,000,032) | +0.0007 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 28,000,032 | `bot` | yes | G3 | ECE | **0.0261** [0.0129, 0.0580] | 0.0228 (@28,000,032) | +0.0033 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 28,000,032 | `pool` | yes | G2 | reliability | **0.0095** [0.0030, 0.0382] | 0.0103 (@28,000,032) | -0.0008 | ✅ | `point <= base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 28,000,032 | `pool` | yes | G3 | ECE | **0.0879** [0.0451, 0.1795] | 0.0875 (@28,000,032) | +0.0004 | ✅ | `CI covers base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 30,000,000 | `all` | no | G2 | reliability | **0.0030** [0.0023, 0.0089] | 0.0020 (@28,000,032, reduced `last`) | +0.0010 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 30,000,000 | `all` | no | G3 | ECE | **0.0237** [0.0191, 0.0696] | 0.0349 (@28,000,032, reduced `last`) | -0.0112 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline meets |
| 30,000,000 | `bot` | yes | G2 | reliability | **0.0039** [0.0020, 0.0094] | 0.0012 (@28,000,032, reduced `last`) | +0.0026 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 30,000,000 | `bot` | yes | G3 | ECE | **0.0422** [0.0200, 0.0681] | 0.0228 (@28,000,032, reduced `last`) | +0.0194 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 30,000,000 | `pool` | yes | G2 | reliability | **0.0088** [0.0042, 0.0362] | 0.0103 (@28,000,032, reduced `last`) | -0.0015 | ✅ | `point <= base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 30,000,000 | `pool` | yes | G3 | ECE | **0.0734** [0.0339, 0.1614] | 0.0875 (@28,000,032, reduced `last`) | -0.0140 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |
| 32,000,016 | `all` | no | G2 | reliability | **0.0014** [0.0012, 0.0061] | 0.0020 (@28,000,032, reduced `last`) | -0.0007 | ✅ | `point <= base` | ≤ 0.005 — arm meets, baseline meets |
| 32,000,016 | `all` | no | G3 | ECE | **0.0283** [0.0212, 0.0604] | 0.0349 (@28,000,032, reduced `last`) | -0.0066 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline meets |
| 32,000,016 | `bot` | yes | G2 | reliability | **0.0036** [0.0015, 0.0102] | 0.0012 (@28,000,032, reduced `last`) | +0.0024 | ❌ | `CI above base` | ≤ 0.005 — arm meets, baseline meets |
| 32,000,016 | `bot` | yes | G3 | ECE | **0.0414** [0.0231, 0.0759] | 0.0228 (@28,000,032, reduced `last`) | +0.0186 | ❌ | `CI above base` | ≤ 0.05 — arm meets, baseline meets |
| 32,000,016 | `pool` | yes | G2 | reliability | **0.0021** [0.0022, 0.0185] | 0.0103 (@28,000,032, reduced `last`) | -0.0082 | ✅ | `point <= base` | ≤ 0.005 — arm meets, baseline MISSES |
| 32,000,016 | `pool` | yes | G3 | ECE | **0.0379** [0.0327, 0.1187] | 0.0875 (@28,000,032, reduced `last`) | -0.0496 | ✅ | `point <= base` | ≤ 0.05 — arm meets, baseline MISSES |
| 34,000,032 | `all` | no | G2 | reliability | **0.0030** [0.0017, 0.0083] | 0.0020 (@28,000,032, reduced `last`) | +0.0010 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 34,000,032 | `all` | no | G3 | ECE | **0.0491** [0.0291, 0.0831] | 0.0349 (@28,000,032, reduced `last`) | +0.0142 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 34,000,032 | `bot` | yes | G2 | reliability | **0.0023** [0.0012, 0.0066] | 0.0012 (@28,000,032, reduced `last`) | +0.0011 | ✅ | `CI covers base` | ≤ 0.005 — arm meets, baseline meets |
| 34,000,032 | `bot` | yes | G3 | ECE | **0.0340** [0.0186, 0.0693] | 0.0228 (@28,000,032, reduced `last`) | +0.0112 | ✅ | `CI covers base` | ≤ 0.05 — arm meets, baseline meets |
| 34,000,032 | `pool` | yes | G2 | reliability | **0.0074** [0.0034, 0.0326] | 0.0103 (@28,000,032, reduced `last`) | -0.0030 | ✅ | `point <= base` | ≤ 0.005 — arm MISSES, baseline MISSES |
| 34,000,032 | `pool` | yes | G3 | ECE | **0.0802** [0.0455, 0.1714] | 0.0875 (@28,000,032, reduced `last`) | -0.0073 | ✅ | `point <= base` | ≤ 0.05 — arm MISSES, baseline MISSES |

_`all` is reported for context and NEVER gated: it averages two populations whose measured calibration bias has opposite sign._

## 2b. FAMINE pre-test — does terminal-only learn at the incumbent's rate?

Comparator `famine_comparator` → `/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/final_model.zip`; floor **38 ELO** (from registry `famine_comparator`.floor_elo).

**the arm TRAILS the comparator by 21 ELO at 12 snapshots (floor 38) — inside the floor: starvation is NOT demonstrated on the ladder half** (signed trail +21, positive = behind).

> RULE: at ~5M: trailing the comparator by more than the floor at matched SNAPSHOT COUNT **AND** win_rate_vs_bots not rising ⇒ terminal-only starves ⇒ kill the arm and launch FROZEN-φ.
>
> This computes the LADDER half only — win_rate_vs_bots is the AND-gate's other half.
>
> ⚠️ the incumbent had PBRS AND PopArt AND the shaped critic, so this is a rate comparison ACROSS RECIPES and the floor is the incumbent's own run-to-run noise, not a replicate of this arm. A trail inside the floor is NOT evidence the two recipes are equivalent — only that starvation has not been demonstrated.

## 3. G7 — stall rate + episode length (KILL condition)

Thresholds: stall rate ≤ 0.05 (a battle at ≥ 250 turns is a stall), episode length ≤ 1.25× the era's. (the design registers G7 as 'no increase over the era, pre-registered threshold' and names no number; this threshold is main.critic_gate's default, not the design's.)

| step | stall rate (captured) | mean turns | ep_len bots | ep_len pool | verdict |
|---|---|---|---|---|---|
| 2,000,016 | 0.0000 | 20.7 | 20.70 | — | OK |
| 4,000,032 | 0.0062 | 22.7 | 21.99 | — | OK |
| 6,000,000 | 0.0000 | 21.9 | 21.84 | 25.89 | OK |
| 8,000,016 | 0.0000 | 25.2 | 21.80 | 28.35 | OK |
| 10,000,032 | 0.0000 | 27.9 | 22.57 | 31.05 | OK |
| 12,000,000 | 0.0000 | 28.1 | 22.34 | 31.10 | OK |
| 14,000,016 | 0.0000 | 28.0 | 21.94 | 32.62 | OK |
| 16,000,032 | 0.0000 | 27.5 | 22.32 | 32.03 | OK |
| 18,000,000 | 0.0000 | 26.2 | 21.47 | 31.75 | OK |
| 20,000,016 | 0.0000 | 26.7 | 21.93 | 31.87 | OK |
| 22,000,032 | 0.0000 | 27.4 | 22.78 | 32.07 | OK |
| 24,000,000 | 0.0041 | 27.0 | 21.54 | 32.08 | OK |
| 26,000,016 | 0.0000 | 25.0 | 21.44 | 31.33 | OK |
| 28,000,032 | 0.0046 | 27.3 | 21.61 | 31.50 | OK |
| 30,000,000 | 0.0000 | 26.2 | 21.83 | 31.58 | OK |
| 32,000,016 | 0.0042 | 27.5 | 23.79 | 33.14 | **KILL — ep_len_bots 23.79 is 1.30x the era's 18.28 (> 1.25)** |
| 34,000,032 | 0.0042 | 27.7 | 23.49 | 31.69 | **KILL — ep_len_bots 23.49 is 1.28x the era's 18.28 (> 1.25)** |

_the captured eval traces' per-battle summary meta.turns / meta.result — the CAPTURE QUOTA, which is loss-enriched by design (agents.training.trace_selection), so read it as an upper-ish bound, never as a population rate_

## 4. Untaught meter — with a CONTINUATION control

_skipped (`--skip-meter`)_

## Not runnable here

| # | criterion | why |
|---|---|---|
| G5 | sd_true_excess, floor-subtracted, per population | gap M1 — not runnable from traces |
| G6 | the MIRROR TABLE (no cell crossing 0.50) | gap M2 — not runnable from traces |
| G8 | win_mask coverage >= a pre-registered floor | gap M3 — the run must record it |
| G9 | capacity value_pooled participation ratio | runnable, but by `python -m main.capacity` |

*Falsification clause (design §5.5, verbatim): What would falsify the design, stated before the data: G1 flat (resolution unmoved) with G2–G4 passing means the promotion bought calibration this head already had and nothing else — the wrong-meter trap, and the target/readout diagnosis of §2 would survive intact while *this* remedy for it would not. That must be reported as loudly as a pass.*
