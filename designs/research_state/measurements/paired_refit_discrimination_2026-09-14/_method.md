## 2. The frame

| | |
|---|---|
| frozen checkpoint (B, C, and A's anchor) | `models/ai_v12_11_ladder_ctrl10M/snapshots/snapshot_000010000032.zip` — step **10,000,032**, `ARCH_SIGNATURE` `gen3_critic_route_wave_v1`, `critic winprob`, `win_prob_mode shaping` |
| the head | `WinProbHead` = LayerNorm(128) → Linear(128,128) → ReLU → Linear(128,1) off `stash.value_pooled`. **Only these four tensors ever move.** |
| Part A arms | `ai_v12_23_ladder_rollout` · `ai_v12_28_ladder_ent05` · `ai_v12_29_ladder_vf025`, each @10000032, same `ARCH_SIGNATURE` |
| Part A anchor | `ai_v12_11_ladder_ctrl10M` @10000032, **in the same window as the three arms** (rule 23) |
| fork source | that checkpoint's OFFLINE full-capture eval tree (seed 20260910, 9,600 battles, `capture: ALL`, `eval_sentinel_greedy: true`) |
| regime | CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice 15`, rust sim bridge for the forks / node search driver for the battery (the validated default), `models/` read-only, nothing written under `models/` |
| the box | a live training arm (`ai_v13_01_flywheel_shaped`, arm S) + its eval workers throughout; load average 22–55 on 16 cores |
