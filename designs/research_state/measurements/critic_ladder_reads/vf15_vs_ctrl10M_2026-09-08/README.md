# Critic ladder read — arm 1 `ai_v12_10_ladder_vf15` (`--vf-coef 1.5`) vs control `ai_v12_11_ladder_ctrl10M`, both at `step_10000032`

Produced by `python -m main.ops.critic_read ai_v12_10_ladder_vf15 --control ai_v12_11_ladder_ctrl10M --out <job tmp>/read_vf15` on 2026-09-08 22:18 PT (the tool's own provenance block is at the foot of `critic_read.md`). Every number is ARM − CONTROL with the difference of two independent battle-clustered bootstraps; NO replicate floor exists yet (arm 6 `ctrl10M_b` supplies it), so every DETECTED is against ZERO and says so.

| file | holds |
|---|---|
| `critic_read.md` / `critic_read.json` | the registered read: the four headline deltas, every G1–G4 row per stratum, the identity rows under the three weightings, each run's own gate verdict |
| `run_readout.json` | the promoted statistics per run (the readout library's output) |
| `identity_payload.json` | the per-state identity labels both sides were read from (arm's cf_audit draw; the control's sits in the tool cache) |

Verdict: see the ledger entry *2026-09-08 · READ · critic ladder arm 1*.
