# CRITIC READ — `ai_v12_13_ladder_tdaux` vs `ai_v12_11_ladder_ctrl10M`

The critic ladder's registered read (ledger 2026-09-08 *REGISTRATION · THE CRITIC LADDER*), produced by `python -m main.ops.critic_read` v1 at 2026-09-09T03:48:43. Every number is a DELTA, **arm − control**, with the difference of the two runs' independent battle-clustered bootstraps. Strength is NOT read.

| role | run | cycle | battles | draw/timeout share | anchors | live |
|---|---|---|---|---|---|---|
| arm | `ai_v12_13_ladder_tdaux` | `step_10000032` | 210 | 0.8% | 143/143 (100.0%) | no |
| control | `ai_v12_11_ladder_ctrl10M` | `step_10000032` | 204 | 0.5% | 140/140 (100.0%) | no |

> 🚨 **NO REPLICATE FLOOR — deltas are NOT DETECTED unless their CI clears zero, and never DETECTED against a floor.** The ladder's replicate floor is the control-vs-control difference and does not exist until a second control replicate does. Every DETECTED below is a detection against ZERO and carries that qualifier.

## SUMMARY — the four headline deltas

| quantity | arm | control | **Δ (arm − control)** | 95% CI | verdict |
|---|---|---|---|---|---|
| resolution · `bot` | +0.0336 | +0.0187 | **+0.0149** | [-0.0045, +0.0343] | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat · `late (turn>=25)` | -0.0349 | -0.0198 | **-0.0151** | [-0.0977, +0.0654] | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) · `ALL` | -0.0848 | -0.0940 | **+0.0092** | [-0.2117, +0.2030] | **NOT DETECTED** (CI covers zero) |
| skill · `bot` | +0.2513 | +0.1372 | **+0.1141** | [-0.0649, +0.2685] | **NOT DETECTED** (CI covers zero) |

Direction, stated so a sign cannot be misread: **resolution** and **skill** are HIGHER is better; **identity bias** is `V − p̂`, so POSITIVE means the head is OPTIMISTIC against its own Monte-Carlo continuation and a NEGATIVE delta is an improvement; the **turn-contrast** is `corr(turn,V) − corr(turn,MC)` and its target is ZERO, so a negative delta from a positive control moves toward the clock.

## 1. RESOLUTION — the calibration gate's metrics, per stratum

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| resolution ⭐ | `all` | `capture-rate (gauge)` | +0.0502 | +0.0419 | **+0.0083** | [-0.0216, +0.0395] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `bot` | `capture-rate (gauge)` | +0.0336 | +0.0187 | **+0.0149** | [-0.0045, +0.0343] | 400 | **NOT DETECTED** (CI covers zero) |
| resolution ⭐ | `pool` | `capture-rate (gauge)` | +0.0737 | +0.0720 | **+0.0017** | [-0.0492, +0.0632] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `all` | `capture-rate (gauge)` | +0.0063 | +0.0022 | **+0.0040** | [-0.0038, +0.0100] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `bot` | `capture-rate (gauge)` | +0.0063 | +0.0033 | **+0.0030** | [-0.0061, +0.0088] | 400 | **NOT DETECTED** (CI covers zero) |
| reliability | `pool` | `capture-rate (gauge)` | +0.0070 | +0.0025 | **+0.0045** | [-0.0078, +0.0225] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `all` | `capture-rate (gauge)` | +0.0587 | +0.0351 | **+0.0236** | [-0.0246, +0.0449] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `bot` | `capture-rate (gauge)` | +0.0562 | +0.0425 | **+0.0137** | [-0.0341, +0.0439] | 400 | **NOT DETECTED** (CI covers zero) |
| ece | `pool` | `capture-rate (gauge)` | +0.0668 | +0.0407 | **+0.0261** | [-0.0463, +0.0705] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `all` | `capture-rate (gauge)` | +0.3299 | +0.2668 | **+0.0630** | [-0.1001, +0.2171] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `bot` | `capture-rate (gauge)` | +0.2513 | +0.1372 | **+0.1141** | [-0.0649, +0.2685] | 400 | **NOT DETECTED** (CI covers zero) |
| skill ⭐ | `pool` | `capture-rate (gauge)` | +0.3954 | +0.3592 | **+0.0362** | [-0.1604, +0.2327] | 400 | **NOT DETECTED** (CI covers zero) |

## 2. IDENTITY — V against the Monte-Carlo continuation

| quantity | stratum | weighting | arm | control | **Δ** | 95% CI | n draws | verdict |
|---|---|---|---|---|---|---|---|---|
| bias V - p_hat | `ALL` | `raw` | +0.0858 | +0.0917 | **-0.0059** | [-0.0582, +0.0460] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `ALL` | `pop` | +0.0251 | +0.0306 | **-0.0055** | [-0.0497, +0.0410] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `ALL` | `ipw` | -0.0512 | -0.0418 | **-0.0094** | [-0.0466, +0.0294] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `early (turn<=10)` | `raw` | +0.0486 | +0.0987 | **-0.0501** | [-0.1097, +0.0098] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `early (turn<=10)` | `pop` | -0.0149 | +0.0134 | **-0.0283** | [-0.0823, +0.0273] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `early (turn<=10)` | `ipw` | -0.0737 | -0.0703 | **-0.0034** | [-0.0661, +0.0610] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `mid (11-24)` | `raw` | +0.1424 | +0.0889 | **+0.0534** | [-0.0103, +0.1158] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `mid (11-24)` | `pop` | +0.0613 | +0.0498 | **+0.0115** | [-0.0430, +0.0650] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `mid (11-24)` | `ipw` | -0.0416 | -0.0286 | **-0.0130** | [-0.0492, +0.0214] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `late (turn>=25)` | `raw` | +0.0298 | +0.0818 | **-0.0521** | [-0.1898, +0.0815] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `late (turn>=25)` | `pop` | +0.0164 | +0.0297 | **-0.0133** | [-0.1246, +0.0966] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `late (turn>=25)` | `ipw` | -0.0349 | -0.0198 | **-0.0151** | [-0.0977, +0.0654] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `bot` | `raw` | +0.0961 | +0.1259 | **-0.0298** | [-0.0952, +0.0336] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `bot` | `pop` | +0.0306 | +0.0535 | **-0.0229** | [-0.0768, +0.0317] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `bot` | `ipw` | -0.0622 | -0.0456 | **-0.0166** | [-0.0567, +0.0239] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `pool` | `raw` | +0.0633 | +0.0277 | **+0.0357** | [-0.0529, +0.1173] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat | `pool` | `pop` | +0.0134 | -0.0134 | **+0.0268** | [-0.0539, +0.1041] | 4000 | **NOT DETECTED** (CI covers zero) |
| bias V - p_hat ⭐ | `pool` | `ipw` | -0.0185 | -0.0382 | **+0.0197** | [-0.0500, +0.0821] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `raw` | +0.0375 | +0.0364 | **+0.0011** | [-0.0176, +0.0186] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution | `ALL` | `pop` | +0.0676 | +0.0650 | **+0.0025** | [-0.0228, +0.0269] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution ⭐ | `ALL` | `ipw` | +0.0270 | +0.0300 | **-0.0030** | [-0.0215, +0.0140] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `raw` | +0.1322 | +0.1172 | **+0.0151** | [-0.0902, +0.1291] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score | `ALL` | `pop` | +0.3139 | +0.2892 | **+0.0247** | [-0.0813, +0.1330] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy skill_score ⭐ | `ALL` | `ipw` | +0.2190 | +0.2296 | **-0.0106** | [-0.1577, +0.1382] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `raw` | +0.1708 | +0.1606 | **+0.0102** | [-0.0711, +0.0867] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share | `ALL` | `pop` | +0.3161 | +0.2935 | **+0.0226** | [-0.0803, +0.1217] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy resolution_cap_share ⭐ | `ALL` | `ipw` | +0.2530 | +0.2482 | **+0.0048** | [-0.1209, +0.1257] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `raw` | +0.0095 | +0.0113 | **-0.0018** | [-0.0140, +0.0082] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability | `ALL` | `pop` | +0.0016 | +0.0022 | **-0.0006** | [-0.0067, +0.0034] | 4000 | **NOT DETECTED** (CI covers zero) |
| Murphy reliability ⭐ | `ALL` | `ipw` | +0.0040 | +0.0038 | **+0.0001** | [-0.0070, +0.0058] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) ⭐ | `ALL` | `raw` | -0.0848 | -0.0940 | **+0.0092** | [-0.2117, +0.2030] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `pop` | -0.0231 | -0.0990 | **+0.0759** | [-0.1229, +0.2473] | 4000 | **NOT DETECTED** (CI covers zero) |
| corr(turn,V) - corr(turn,MC) | `ALL` | `ipw` | +0.0661 | +0.0004 | **+0.0657** | [-0.1552, +0.2837] | 4000 | **NOT DETECTED** (CI covers zero) |

⭐ = the REGISTERED estimand for that quantity. The others are the same statistic under a different selection correction, printed so the registered number is never the only one on the page.

### the three weightings

| name | what it corrects |
|---|---|
| `raw` | unweighted over the cf_audit draw — the estimand arm A's committed +0.3089 was computed under, kept for comparability |
| `pop` | cf_audit's stratified draw recombined at the frame's own (decile, outcome) mass — corrects the SAMPLER against the trace tree |
| `ipw` | `pop` times 1/capture_rate(opponent, outcome) from the cycle's eval_manifest — RULE OF EVIDENCE 17, correcting the loss-enriched TREE against the eval population |

The gate rows carry no weighting column because the scaffolding gauge applies the capture-rate correction itself, from the same `eval_manifest.json` rates — there is no unweighted variant of a gate row. And the anchor (label-trust) arm is a CENSUS of the bot battles it draws from and its estimand is the REPLAY DRIVER's fidelity, not a property of the eval population — capture-rate reweighting does NOT apply to it and none is applied.

## 3. Each run on its own — the registered G1–G4 rows at the read cycle

**arm — `ai_v12_13_ladder_tdaux` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0502 [+0.0303, +0.0765] | 0.0618 | -0.0116 | +0.3299 [+0.2164, +0.4280] | ❌ | ❌ | ❌ | ✅ |
| `bot` | yes | 0.0336 [+0.0203, +0.0503] | 0.0337 | -0.0000 | +0.2513 [+0.1259, +0.3394] | ❌ | ❌ | ❌ | ✅ |
| `pool` | yes | 0.0737 [+0.0384, +0.1353] | 0.0711 | +0.0026 | +0.3954 [+0.1997, +0.5327] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 181 battles / 6400 rollouts · Brier 0.0834 = REL 0.0040 − RES 0.0270 + UNC 0.1068 + WBV 0.0008 (resid -1.12e-03) · base rate 0.8784 · **resolution is 25.3% of the base-rate cap** · corr(turn,V) -0.2530 vs corr(turn,MC) -0.1681

**control — `ai_v12_11_ladder_ctrl10M` @ step_10000032** (`main.critic_gate` verdict: `{"G1": false, "G2": false, "G3": false, "G4": false, "n_gated_rows": 8}`)

| stratum | gated | resolution [CI] | baseline | Δ vs baseline | skill [CI] | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| `all` | no | 0.0419 [+0.0261, +0.0661] | 0.0618 | -0.0200 | +0.2668 [+0.1406, +0.3576] | ❌ | ✅ | ✅ | ✅ |
| `bot` | yes | 0.0187 [+0.0106, +0.0333] | 0.0337 | -0.0149 | +0.1372 [+0.0059, +0.2536] | ❌ | ❌ | ✅ | ✅ |
| `pool` | yes | 0.0720 [+0.0436, +0.1127] | 0.0711 | +0.0009 | +0.3592 [+0.1599, +0.4649] | ❌ | ✅ | ✅ | ✅ |

Identity, capture-rate reweighted: 800 labels / 185 battles / 6400 rollouts · Brier 0.0930 = REL 0.0038 − RES 0.0300 + UNC 0.1208 + WBV 0.0008 (resid -2.34e-03) · base rate 0.8595 · **resolution is 24.8% of the base-rate cap** · corr(turn,V) -0.1227 vs corr(turn,MC) -0.0286

## 4. THE LEDGER LINE

```
ai_v12_13_ladder_tdaux vs ai_v12_11_ladder_ctrl10M at 10M: G1 bot Δ +0.0149 [-0.0045, +0.0343] NOT DETECTED · identity bias late Δ -0.0151 [-0.0977, +0.0654] NOT DETECTED · turn-contrast Δ +0.0092 [-0.2117, +0.2030] NOT DETECTED
```

## 5. Provenance — every command and every path

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.ops.critic_read ai_v12_13_ladder_tdaux --control ai_v12_11_ladder_ctrl10M --no-cache --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_tdaux2

# arm — ai_v12_13_ladder_tdaux
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/dev/gen3ai/models/ai_v12_13_ladder_tdaux --step 10000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/read_tdaux2/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_13_ladder_tdaux --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/read_tdaux2/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/read_tdaux2/gate/critic_gate.md
# control — ai_v12_11_ladder_ctrl10M
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m agents.training.cf_audit /home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M --step 10000032 --impl rust --rollouts 8 --states 800 --anchors 150 --seed 0 --anchor-tolerance 0.9 --out /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity
  /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.critic_gate /home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M --parent v9_fold_parent --famine-comparator off --skip-meter --boot 400 --seed 0 --reliability-bins 10 --json /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate/critic_gate.json --md /home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate/critic_gate.md
```

| what | path |
|---|---|
| arm run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_13_ladder_tdaux` |
| arm trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_13_ladder_tdaux/eval_traces/step_10000032` |
| arm identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_tdaux2/identity` |
| arm gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_tdaux2/gate` |
| arm identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_tdaux2/identity_payload.json` |
| control run dir (READ-ONLY) | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M` |
| control trace dir | `/home/goodlad/dev/gen3ai/models/ai_v12_11_ladder_ctrl10M/eval_traces/step_10000032` |
| control identity | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity` |
| control gate | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/gate` |
| control identity_payload | `/home/goodlad/.claude/jobs/9ab51de6/tmp/ai_v12_11_ladder_ctrl10M/identity_payload.json` |
| this report | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_tdaux2/critic_read.md` |
| machine-readable | `/home/goodlad/.claude/jobs/9ab51de6/tmp/read_tdaux2/critic_read.json` |

Bootstraps: identity 4000 draws, gate 400 draws, seed 0, unit = BATTLE, deltas = difference of INDEPENDENT bootstraps. Nothing was written under `models/`.
