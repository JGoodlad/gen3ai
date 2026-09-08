---
name: project_next_run_plan
description: "The next-run pre-flight list lives at designs/ai_v8/next_run_plan.md — locked-in changes, contingencies, staged experiments"
metadata: 
  node_type: memory
  type: project
  originSessionId: f1a71c92-5431-47c7-a6e2-b2b0c1e359c4
---

**NEXT-RUN PLAN (2026-07-20, owner-requested).** The authoritative pre-flight list for the next
fresh run / retrain-class fork after ai_v8_03 is **`designs/ai_v8/next_run_plan.md`** (in-repo,
uncommitted until a /gen3ai-ship). Locked-in: privileged critic (scouting-safe: public obs AND
privilege) + public-V aux readout; categorical-critic Phase B (value-dist as PRIMARY loss, widen
vmax first); `--damage-refine-rounds 1` (coupled to the belief-grad outcome); skip env-side CPU
obs blocks that `--unified-obs` masks (verify encoder still computes them); belief-grad-mode
decided by the staged in-run flip on ai_v8_03 (shaping if it revives the belief→physics stack,
else detached + strip refine/threat); top-K+tail op candidates (offline fidelity probe first);
booster stack carries (block-64, accum-16, onesided PFSP, zarch heads/32; film-accum 4→6);
ship-before-fork = the fork eval-anchor guard. Staged experiments: search-teacher at low coef,
z_opp FiLM, FiLM centering (fresh-run only), Phase-2 exploiter distill. Rank-0 engineering: Rust
`state_encoder` hot path. Compile is dead. See [[project_damage_op_block_audit]],
[[project_belief_shaping_experiment]].
