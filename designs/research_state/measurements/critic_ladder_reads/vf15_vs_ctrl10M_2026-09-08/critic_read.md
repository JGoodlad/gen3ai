# CRITIC READ — `ai_v12_10_ladder_vf15` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v1 at 2026-09-08T22:18:32. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_10_ladder_vf15` | `step_10000032` | 205 | 0.7% | 141/142 (99.3%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 204 | 0.5% | 140/140 (100.0%) | no |

> 🚨 **NO REPLICATE FLOOR — deltas are NOT DETECTED unless their CI clears zero, and never DETECTED against a floor.** The ladder's replicate floor is the control-vs-control difference and does not exist until a second control replicate does. Every DETECTED below is a detection against ZERO and carries that qualifier.

## SUMMARY — the four headline deltas

| quantity | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|
| resolution · `bot` | +0.0327 | +0.0187 | **+0.0140** | [-0.0080, +0.0389] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | +0.0288 | -0.0198 | **+0.0486** | [-0.0305, +0.1311] | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) · `ALL` | +0.0694 | -0.0940 | **+0.1634** | [-0.0807, +0.3733] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | +0.2591 | +0.1372 | **+0.1219** | [-0.0681, +0.2941] | **NOT DETECTED** (CI covers zero) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0507 | +0.0419 | **+0.0088** | [-0.0207, +0.0384] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0327 | +0.0187 | **+0.0140** | [-0.0080, +0.0389] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.1094 | +0.0720 | **+0.0374** | [-0.0253, +0.0861] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0047 | +0.0022 | **+0.0024** | [-0.0047, +0.0137] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0013 | +0.0033 | **-0.0020** | [-0.0100, +0.0066] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0358 | +0.0025 | **+0.0333** | [+0.0065, +0.0682] | 400 | **DETECTED** (vs ZERO — NO FLOOR) |
| ece | `all` | `capture-rate (gauge)` | +0.0337 | +0.0351 | **-0.0014** | [-0.0399, +0.0370] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `bot` | `capture-rate (gauge)` | +0.0143 | +0.0425 | **-0.0282** | [-0.0598, +0.0175] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `pool` | `capture-rate (gauge)` | +0.1008 | +0.0407 | **+0.0601** | [-0.0224, +0.1156] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.3277 | +0.2668 | **+0.0609** | [-0.0772, +0.2024] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2591 | +0.1372 | **+0.1219** | [-0.0681, +0.2941] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.4133 | +0.3592 | **+0.0541** | [-0.1163, +0.2317] | 400 | **NOT DETECTED** (CI covers zero) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.1761 | +0.0917 | **+0.0845** | [+0.0261, +0.1425] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `ALL` | `pop` | +0.0986 | +0.0306 | **+0.0680** | [+0.0206, +0.1184] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | +0.0065 | -0.0418 | **+0.0484** | [+0.0139, +0.0815] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.1197 | +0.0987 | **+0.0211** | [-0.0407, +0.0853] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `early (turn<=10)` | `pop` | +0.0576 | +0.0134 | **+0.0442** | [-0.0111, +0.0997] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0134 | -0.0703 | **+0.0568** | [-0.0023, +0.1133] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.1954 | +0.0889 | **+0.1064** | [+0.0331, +0.1759] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.1057 | +0.0498 | **+0.0560** | [-0.0030, +0.1158] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | +0.0161 | -0.0286 | **+0.0447** | [+0.0112, +0.0829] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.2314 | +0.0818 | **+0.1496** | [-0.0016, +0.2857] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.1408 | +0.0297 | **+0.1111** | [-0.0120, +0.2287] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | +0.0288 | -0.0198 | **+0.0486** | [-0.0305, +0.1311] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `bot` | `raw` | +0.2178 | +0.1259 | **+0.0918** | [+0.0191, +0.1585] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `bot` | `pop` | +0.1291 | +0.0535 | **+0.0756** | [+0.0147, +0.1340] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat ⭐ | `bot` | `ipw` | +0.0101 | -0.0456 | **+0.0557** | [+0.0178, +0.0943] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| bias V - p_hat | `pool` | `raw` | +0.1037 | +0.0277 | **+0.0760** | [-0.0058, +0.1546] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `pool` | `pop` | +0.0598 | -0.0134 | **+0.0731** | [-0.0043, +0.1506] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `pool` | `ipw` | +0.0047 | -0.0382 | **+0.0428** | [-0.0168, +0.0993] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `raw` | +0.0368 | +0.0364 | **+0.0004** | [-0.0173, +0.0159] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0673 | +0.0650 | **+0.0023** | [-0.0199, +0.0228] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0343 | +0.0300 | **+0.0043** | [-0.0148, +0.0204] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.0045 | +0.1172 | **-0.1126** | [-0.2294, +0.0073] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `pop` | +0.2353 | +0.2892 | **-0.0539** | [-0.1561, +0.0470] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2924 | +0.2296 | **+0.0628** | [-0.0596, +0.1892] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1570 | +0.1606 | **-0.0036** | [-0.0815, +0.0653] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.3021 | +0.2935 | **+0.0086** | [-0.0851, +0.0932] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.3048 | +0.2482 | **+0.0566** | [-0.0588, +0.1524] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0367 | +0.0113 | **+0.0254** | [+0.0064, +0.0445] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| Murphy reliability | `ALL` | `pop` | +0.0153 | +0.0022 | **+0.0131** | [+0.0030, +0.0246] | 4000 | **DETECTED** (vs ZERO — NO FLOOR) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0014 | +0.0038 | **-0.0024** | [-0.0093, +0.0019] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | +0.0694 | -0.0940 | **+0.1634** | [-0.0807, +0.3733] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | +0.0080 | -0.0990 | **+0.1070** | [-0.1035, +0.2964] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | +0.0164 | +0.0004 | **+0.0161** | [-0.2028, +0.2402] | 4000 | **NOT DETECTED** (CI covers zero) |

⭐ = the REGISTERED estimand for that quantity. The others are the same statistic under a different selection correction, printed so the registered number is never the only one on the page.

### the three weightings

| name | what it corrects |
|---|---|
| `raw` | unweighted over the cf_audit draw — the estimand arm A's committed +0.3089 was computed under, kept for comparability |
| `pop` | cf_audit's stratified draw recombined at the frame's own (decile, outcome) mass — corrects the SAMPLER against the trace tree |
| `ipw` | `pop` times 1/capture_rate(opponent, outcome) from the cycle's eval_manifest — RULE OF EVIDENCE 17, correcting the loss-enriched TREE against the eval population |

The gate rows carry no weighting column because the scaffolding gauge applies the capture-rate correction itself, from the same `eval_manifest.json` rates — there is no unweighted variant of a gate row. And the anchor (label-trust) arm is a CENSUS of the bot battles it draws from and its estimand is the REPLAY DRIVER's fidelity, not a property of the eval population — capture-rate reweighting does NOT apply to it and none is applied.

## 3. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_10_ladder_vf15` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0507 [+0.0334, +0.0770] | 0.0618 | -0.0112 | +0.3277 [+0.2500, +0.3918] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0327 [+0.0203, +0.0583] | 0.0337 | -0.0009 | +0.2591 [+0.1340, +0.3623] | ❌ | ✅ | ✅ | ✅ |
| `pool` | yes | 0.1094 [+0.0606, +0.1517] | 0.0711 | +0.0383 | +0.4133 [+0.2852, +0.4998] | ❌ | ❌ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 178 battles / 6400 rollouts · Brier 0.0796 = REL 0.0014 − RES 0.0343 + UNC 0.1125 + WBV 0.0008 (resid -8.50e-04) · base rate 0.8708 · **resolution is 30.5% of the base-rate cap** · corr(turn,V) -0.1905 vs corr(turn,MC) -0.2598

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 185 battles / 6400 rollouts · Brier 0.0930 = REL 0.0038 − RES 0.0300 + UNC 0.1208 + WBV 0.0008 (resid -2.34e-03) · base rate 0.8595 · **resolution is 24.8% of the base-rate cap** · corr(turn,V) -0.1227 vs corr(turn,MC) -0.0286

## 4. THE LEDGER LINE

```
ai_v12_10_ladder_vf15 vs ai_v12_11_ladder_ctrl10M at 10M: G1 bot Δ +0.0140 [-0.0080, +0.0389] NOT DETECTED · identity bias late Δ +0.0486 [-0.0305, +0.1311] NOT DETECTED · turn-contrast Δ +0.1634 [-0.0807, +0.3733] NOT DETECTED
```

## 5. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_10_ladder_vf15 --control ai_v12_11_ladder_ctrl10M --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_vf15

# arm — ai_v12_10_ladder_vf15
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/dev/gen3ai/models/ai_v12_10_ladder_vf15 --step 10000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_vf15/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_10_ladder_vf15 --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/read_vf15/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/read_vf15/gate/critic_gate.md
# control — ai_v12_11_ladder_ctrl10M
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M --step 10000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate/critic_gate.md
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_10_ladder_vf15` |
| arm trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_10_ladder_vf15/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_vf15/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_vf15/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_vf15/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_vf15/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_vf15/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
