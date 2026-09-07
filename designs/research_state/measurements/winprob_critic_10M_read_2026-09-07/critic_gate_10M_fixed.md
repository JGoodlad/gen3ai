# CRITIC GATE — `ai_v12_02_winprob_critic` vs `ai_v9_59_R2ACTION_0827`

The pre-registered read of `designs/ai_v12/design_winprob_only_critic.md` §5.5 (endpoints) / §4.3 (bars). Generated 2026-09-07T01:00:19.

| input | spec | resolved file | rung |
|---|---|---|---|
| run | `ai_v12_02_winprob_critic` | `/home/goodlad/dev/gen3ai/models/ai_v12_02_winprob_critic/checkpoints/checkpoint_9969408_steps.zip` | `latest_txt` |
| parent | `v9_fold_parent` | `/home/goodlad/dev/gen3ai/models/ai_v9_59_R2ACTION_0827/final_model.zip` | `explicit_zip` |
| control | `models/ai_v9_195_G5PLAINA_0906` | `/home/goodlad/dev/gen3ai/models/ai_v9_195_G5PLAINA_0906/final_model.zip` | `latest_txt` |
| control | `models/ai_v9_196_G5PLAINB_0906` | `/home/goodlad/dev/gen3ai/models/ai_v9_196_G5PLAINB_0906/final_model.zip` | `latest_txt` |
| control | `models/ai_v9_197_G5PLAINC_0906` | `/home/goodlad/dev/gen3ai/models/ai_v9_197_G5PLAINC_0906/final_model.zip` | `latest_txt` |

## VERDICT — **MIXED**

> criteria not met: G1, G2, G3

## 1. Anchored ladder, at matched SNAPSHOT COUNT

**rating final — the run wrote a final model**

| # | ai_v12_02_winprob_critic step | elo ±95% | ai_v9_59_R2ACTION_0827 step | elo ±95% |
|---|---|---|---|---|
| 1 | 4,000,032 | 1778 ± 23 | 2,000,016 | 1554 ± 64 |
| 2 | 6,000,000 | 1914 ± 25 | 4,000,032 | 1837 ± 57 |
| 3 | 8,000,016 | 1975 ± 26 | 6,000,000 | 1818 ± 58 |
| 4 | 10,000,032 | 2022 ± 27 | 8,000,016 | 1863 ± 44 |

**Δ at 4 snapshots: +159 ELO [+107, +211]** — matched SNAPSHOT COUNT, never matched step. Both fits are anchored to the same pinned bots, so the delta is meaningful; its SE combines two independent fits and carries NO term for the anchor uncertainty they share.

Fit size: 🚨 UNMATCHED FIT SIZE — this delta compares the n-th node of a LONGER final fit against a short fit's n-th node, which is the newest-node-inflation error (94 Elo on rev-1's 8M node, 2026-09-07) and is NOT a matched-count reading; quote it only with this label. Why no refit: the first-4 prefix of parent's ladder rates 0 node(s) on its own, not 4: the prefix has no measured edge inside itself (its early nodes are rated only through edges to LATER snapshots), so a matched-size fit does not exist for it.

## 2. Calibration gate (G1–G4) — RESOLUTION is primary

Bars read from `/home/goodlad/dev/gen3ai-wt-gatefix/designs/research_state/measurements/winprob_critic_baseline_2026-09-06/selection_reweighted.json` (`ai_v9_59_R2ACTION_0827`, steps [26000016, 28000032]; G1 reduce=`max`, G2/G3 reduce=`last`), selection-reweighted. the committed baseline publishes no interval for `resolution`, so G1 compares the ARM's cluster CI against the baseline as a FIXED bar. G2/G3 inherit the same asymmetry: the baseline value they are compared against is a point, and only the ARM carries an interval.

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

_`all` is reported for context and NEVER gated: it averages two populations whose measured calibration bias has opposite sign._

## 2b. FAMINE pre-test — does terminal-only learn at the incumbent's rate?

Comparator `famine_comparator` → `/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/final_model.zip`; floor **38 ELO** (from registry `famine_comparator`.floor_elo).

**the arm TRAILS the comparator by 30 ELO at 4 snapshots (floor 38) — inside the floor: starvation is NOT demonstrated on the ladder half** (signed trail +30, positive = behind).

> RULE: at ~5M: trailing the comparator by more than the floor at matched SNAPSHOT COUNT **AND** win_rate_vs_bots not rising ⇒ terminal-only starves ⇒ kill the arm and launch FROZEN-φ.
>
> This computes the LADDER half only — win_rate_vs_bots is the AND-gate's other half.
>
> ⚠️ the incumbent had PBRS AND PopArt AND the shaped critic, so this is a rate comparison ACROSS RECIPES and the floor is the incumbent's own run-to-run noise, not a replicate of this arm. A trail inside the floor is NOT evidence the two recipes are equivalent — only that starvation has not been demonstrated.

## 3. G7 — stall rate + episode length (KILL condition)

Thresholds: stall rate ≤ 0.05 (a battle at ≥ 250 turns is a stall), episode length ≤ 1.25× the era's. (the design registers G7 as 'no increase over the era, pre-registered threshold' and names no number; this threshold is main.critic_gate's default, not the design's.)

| step | stall rate (captured) | mean turns | ep_len bots | ep_len pool | verdict |
|---|---|---|---|---|---|
| 2,000,016 | 0.0000 | 20.7 | 20.70 | 0.00 | OK |
| 4,000,032 | 0.0062 | 22.7 | 21.99 | 0.00 | OK |
| 6,000,000 | 0.0000 | 21.9 | 21.84 | 25.89 | OK |
| 8,000,016 | 0.0000 | 25.2 | 21.80 | 28.35 | OK |
| 10,000,032 | 0.0000 | 27.9 | 22.57 | 31.05 | OK |

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
