---
name: project_defensive_entropy_built
description: State-conditioned defensive-exploration entropy boost BUILT (gen3_defensive_entropy_v1) + the under-healing diagnosis + the PBRS advantage-invariance insight; NOT shipped
metadata: 
  node_type: memory
  type: project
  originSessionId: a6b67bc4-6b85-495f-93e8-837a94b09851
---

> **Archived 2026-09-08** — ai_v6-era exploration lever, built and never adopted; era closed. Preserved verbatim; nothing below is current.

2026-06-21 (worktree bridge-cse, `gen3_defensive_entropy_v1`, training-only — no version/ARCH change, NOT shipped). Built the user's idea: explore defensive moves more WITHOUT a reward bias, so the existing anti-stall reward stays the guardrail.

**DIAGNOSIS first (ai_v6_13 prober dig):** the model UNDER-uses Recover/Soft-Boiled/Wish/Refresh/Heal Bell. Confounded raw signal (heal-rate ~flat 26% across HP bands) — BUT at low HP healing is often correctly skipped (DOOMED: e.g. Blissey 6% vs Swampert active_pko 1.0/exp 75% → chipping correct, NOT a mistake). Threat-CONDITIONED (ProbeSession batch-analyze, 40 survivable-HP decisions): SAFE (opp exp<30%) chose-heal 38%/mean-prob 24%; opp 30-50% → 0%; the user's exact case reproduced (Jolteon doing 16%, pko 0.00 → chose Thunder Wave, heal-prob 3%). So it under-heals even when safe. NOT the no-progress clock (HEAL_FREEZE_GRACE=2 → first heals already FREE → under-use is first-heal = a VALUE problem, not a tax).

**KEY CONCEPTUAL INSIGHT (reusable for reward design):** material PBRS is **advantage-INVARIANT** (A_shaped = A_base — the heal's +ΔΦ reward is exactly cancelled by the −Φ baked into V_shaped=V_base−Φ; the Φ(s′) terms cancel in Q). So you CANNOT make the model heal by tuning the material potential — it provably contributes zero gradient to heal-vs-attack. PBRS only speeds the CRITIC learning values (faster/lower-variance), it does NOT bias behavior or change the optimum. Healing-when-safe IS optimal vs a human, but the model is model-FREE (learns from statistics, can't simulate "heal-or-die-to-the-sub-in") AND optimized for the AGGRESSIVE self-play meta (no opponent punishes over-aggression → no "heal→win" signal). The only reward-side lever that CAN change the heal advantage is a NON-PBRS bias (risks over-healing/stall) — which is why the user (rightly) chose EXPLORATION instead. Real fix = teacher/league (make patience win in-distribution).

**BUILT (the safe exploration lever):** env `gen3_env._defensive_opportunity` → training-only `defensive_opportunity` obs key =1.0 when active has a PRODUCTIVE defensive move legal (is_heal+HP<0.85 / Refresh+statused / Heal Bell+team-statused; forced-switch→0; never raises; public-only, no leak). PPO weights the per-decision entropy bonus: `entropy_loss=-mean((1+(B_eff−1)·flag)·entropy)`, B_eff anneals B→1 over `--defensive-entropy-anneal-frac`. `--defensive-entropy-boost` (1.0=OFF byte-identical) + anneal-frac; training-only (NOT version-locked, resume-settable like ent_coef); emit gated on boost>1. New `defent/*` metrics (flagged_frac, entropy_flagged vs unflagged). **Honest caveat (in docs):** ORTHOGONAL to reward — if the critic learns healing is net-negative the boost won't override it; model already samples heals ~24% so it helps mainly at rare policy-collapse states; complementary to, NOT a substitute for, a teacher/league.

**Verified:** 8 unit tests (`defensive_entropy_test.py`: flag predicate, forced-switch=0, never-raises, anneal schedule, off-byte-identical) + full suite 2939 green + serverless smoke (Round-trip PASSED, flagged_frac 0.12-0.25, **entropy_flagged 1.76-1.84 > entropy_unflagged 1.52-1.67** = boost works) + 4-lens adversarial review ALL 4 ship (29 praise, 1 minor + 1 nit = doc-wording on the contingency, fixed). Use: `--defensive-entropy-boost 3.0 [--defensive-entropy-anneal-frac 0.5]`. Pairs w/ [[project_nature_ev_belief_built]] (the "am I safe to heal" read) + the positional_grind teacher/league conclusion.
