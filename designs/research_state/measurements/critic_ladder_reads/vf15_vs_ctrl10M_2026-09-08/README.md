# Critic ladder read — arm 1 `ai_v12_10_ladder_vf15` (`--vf-coef 1.5`) vs control `ai_v12_11_ladder_ctrl10M`, both at `step_10000032`

Produced by `python -m main.ops.critic_read ai_v12_10_ladder_vf15 --control ai_v12_11_ladder_ctrl10M --out <job tmp>/read_vf15` on 2026-09-08 22:18 PT (the tool's own provenance block is at the foot of `critic_read.md`). Every number is ARM − CONTROL with the difference of two independent battle-clustered bootstraps; NO replicate floor exists yet (arm 6 `ctrl10M_b` supplies it), so every DETECTED is against ZERO and says so.

| file | holds |
|---|---|
| `critic_read.md` / `critic_read.json` | the registered read: the four headline deltas, every G1–G4 row per stratum, the identity rows under the three weightings, each run's own gate verdict |
| `run_readout.json` | the promoted statistics per run (the readout library's output) |
| `identity_payload.json` | the per-state identity labels both sides were read from (arm's cf_audit draw; the control's sits in the tool cache) |

Verdict: see the ledger entry *2026-09-08 · READ · critic ladder arm 1*.


## `critic_read_v2.*` — the same pair, regenerated with the CONDITIONING section (2026-09-09 ~05:15 PT)

`python -m main.ops.critic_read ai_v12_10_ladder_vf15 --control ai_v12_11_ladder_ctrl10M --step
10000032 --out <job tmp>/read_vf15_v2`. **`critic_read.md/json` (v1) are UNCHANGED and remain the
record of what was reported at the time**; v2 adds `main.ops.conditioning_meters`' ten rows and
changes nothing else — the identity and gate halves were reused from the v1 run's own artifacts on
a matching fingerprint, so every number they carry is byte-identical and the whole re-read cost
**42 s**.

**The CONDITIONING rows** (ARM − CONTROL, battle-clustered difference of independent bootstraps,
2,000 draws, no replicate floor so every label is against zero):

| row | arm | control | Δ | 95% CI | |
|---|---|---|---|---|---|
| spread ratio, turn 1–3 ⭐ | +0.2515 | +0.1152 | +0.1362 | [−0.3152, +0.4826] | NOT DETECTED |
| spread ratio RAW, turn 1–3 | +0.3942 | +0.3333 | +0.0609 | [−0.2650, +0.3167] | NOT DETECTED |
| sd(V) − sd(outcome), turn 1–3 | −0.0389 | −0.0886 | +0.0497 | [−0.0115, +0.0789] | NOT DETECTED |
| spread ratio, all states | +0.5345 | +0.7895 | −0.2549 | [−0.6704, +0.4634] | NOT DETECTED |
| spread ratio RAW, all states | +0.6293 | +0.8053 | −0.1760 | [−0.5588, +0.3676] | NOT DETECTED |
| sd(V) − sd(outcome), all states | −0.0242 | −0.0211 | −0.0031 | [−0.0463, +0.0440] | NOT DETECTED |
| Elo slope, per 100 Elo | +0.0142 | +0.0098 | +0.0044 | [−0.0090, +0.0168] | NOT DETECTED |
| own-team LOO win-rate R², turn 1 ⭐ | −0.0466 | −0.0237 | −0.0229 | [−0.2312, +0.0917] | NOT DETECTED |
| own-team LOO win-rate R², all states | −0.0093 | −0.0098 | +0.0005 | [−0.1135, +0.1285] | NOT DETECTED |
| opponent-class AUC, turn 1 | +0.5963 | +0.4619 | +0.1344 | [−0.0214, +0.2857] | NOT DETECTED |

Frame: arm 6,161 states / 197 battles / 12 opponents / 91 trainee teams, `selection_schema` 2,
`max|values − win_probs|` = 0.0, 8 draw battles excluded; control 6,080 / 198 / 12 / 76, 6 draws
excluded. No row was omitted on either side.

⚠️ The control's clamped turn-1–3 ratio is **+0.1152 with a CI of [0.0000, 0.6098]** — the clamp
hazard again; read the interval and the unclamped +0.3333 beside it.

**`critic_read_v3.md/json` (2026-09-09, tool v3 — QUOTA MATCHING now default).** `main.ops.critic_read` now equalises the two trace frames before it reads any FRAME-SENSITIVE conditioning row (`main.ops.quota_match`). This pair is **SYMMETRIC** — both sides carry the same realized cap, **8/12 traced wins/losses per opponent** (vf15 205 traced battles, ctrl10M 204) — so nothing is subsampled and **every delta row in v3 is bit-for-bit identical to v2**. v3 is kept only because the report now prints the realized capture profiles in its header, which is the evidence that this read was never affected.
