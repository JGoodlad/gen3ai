---
name: project_opponent_distillation_findings
description: "Opponent distillation IS feasible — a faithful distilled self-play opponent at ~3.1x cheaper; the recipe + what does/doesn't work"
metadata: 
  node_type: memory
  type: project
  originSessionId: efe52a87-8b27-4306-a5ee-edd8dbc6796d
---

> **Archived 2026-09-08** — ai_v5-era opponent distillation; superseded by torch.compile'd opponents and the rust bridge. Preserved verbatim; nothing below is current.

Empirical result (2026-06-03/04, multi-agent workflow exploration over the 72M-step `snapshot_000072000000` teacher; goal = cheaper self-play opponent since its forward is ~70% of worker CPU — see [[project_throughput_profile]]). Design + guardrails: `designs/ai_v5/design_opponent_distillation.md`.

**Best deployable = `assembled` (lean cheap encoder + lite-attn head): ~6.40× cheaper (0.30 ms vs teacher ~1.95 ms), h2h 0.428 ± 0.061 (N=250) — statistically TIED with the 4.67× baseline.** Conservative/more-confirmed option = `cheaper_encoder`: 4.67×, h2h 0.443 ± 0.056 (N=300). Recipe = TWO-STAGE: (1) distill a **131K cheap per-slot MLP encoder** onto the teacher's **frozen 12×128 role tokens** by MSE (cosine 0.956) — broke the cheap-encoder ceiling (top-1 0.757→0.858); (2) train a **matchup-aware head** (opp-active role token + per-move type-matchup slices, soft-KL **T=0.7 + label-smoothing 0.02**); the 6.40× variant additionally replaces the head's transformer with a distilled **lite-attn** (the head-shrink was the only surgical shrink that composed; +1.73× e2e). **Component map:** COST lives in the transformer family (team_transformer 53% + cls_pool 9%); FIDELITY lives ONLY in the per-id embeddings (Δtop1 +0.040) — move-MLP/within-mon-attn/role-encoder are ~zero-fidelity (a single Linear matches the teacher). **IRREDUCIBLE FLOOR = `unpack` (0.107 ms ≈ 36% of the student, frozen slicing, not distillable) → practical ceiling ~6–7×; next gain is VECTORIZING unpack (engineering, not distillation).** (`improve_1`, the same head on the REAL frozen encoder, = 3.1× — superseded.)

**HONEST CEILING: ~0.44–0.48 h2h, NOT a true 0.50 draw** — the distilled opponent is *slightly weaker* than the teacher (deploy-safe, stable: 0.443±0.056 @N=300 and 0.475±0.069 @N=200; NOT meaningfully exploitable — a fixed bot beats the student 26% vs the teacher 15%, gap CI spans zero). **The ceiling is a STRUCTURAL decision-rule limit — not data, not encoder, not capacity** (proven across 6 exhausted levers):
- **NOT encoder fidelity:** the token-fidelity→h2h curve is FLAT (cosine 0.956→0.974→1.000 leaves h2h 0.41–0.49, no trend); the EXACT teacher encoder still caps at 0.467. So the cheap 131K encoder costs ~nothing in h2h.
- **NOT compounding error / a data gap (DAgger FALSIFIED it):** DAgger confirmed its premise (on the student's own visited states, teacher-agreement drops 0.858→0.767, a clean 9-pt Ross-Bagnell drift) but its cure FAILED (relabel+retrain moved h2h +0.002 while top-1 rose to 0.882). On-policy data is NOT the lever.
- **The real cause:** the student matches the teacher's argmax 86–88% but is coarser on the ~12–14% **near-tie decisions** — the residual ~6 pts live in **value/near-tie info a greedy, value-blind, logit-only-distilled student can't reproduce.**
- **DEAD ENDS (do not revisit):** more DAgger, head depth/2nd layer, joint e2e unfreeze, encoder-token fidelity, focal reweighting, FitNets, int8 dynamic quant, flat MLP, more width.
- **Value distillation ALSO failed (the last lever, now closed):** 3 value-informed levers (multitask, decisiveness-weighted KL, hard-argmax blend) all stayed ≤0.44. Decisive proof — the student predicts teacher V(s) at **98.7% variance** yet the argmax doesn't sharpen (value was already linearly in the pooled read), and value-extremity is FLAT across error locations while errors concentrate in the near-tie gap quartile (37.9%). So **~0.44–0.48 is the HARD OFFLINE ceiling across 8 independent attacks**; a true 0.50 needs **ONLINE / in-loop fine-tuning** (self-play RL against the teacher), not any offline signal.

**SHIP `cheaper_encoder` at 4.67× behind a fail-closed gate keyed to the LOW edge** (operating point ~0.44, auto-revert on a ≥300-battle live h2h below ~0.40 — asymmetric, not symmetric around 0.50, to avoid false-trips; upper bound 0.55). Integration plan: `designs/ai_v5/distill_integration.md`.

**Hard-won findings (don't re-discover):**
- **The frozen `pokemon_encoder` is load-bearing** — it sets a ~0.86 top-1 ceiling. Every student that reconstructs it cheaply caps at ~0.75 (`cheap_structured`, 6.3× but unfaithful) or ~0.47 (flat MLPs, dead).
- **The transformer is NOT the speed bottleneck** — cutting it 2→1 layer gave only **1.14×**. Per-component profile: frozen unpack 17% + frozen `pokemon_encoder` **46%** = ~63% immovable; the head (transformer+matchup+MLP) is ~35%. So **speed lives in the encoder, never the transformer head** — and the win came from *replacing* the encoder with a distilled cheaper one (3.1×→4.67×), exactly as predicted. int8 dynamic quant moved the WRONG way (overhead on small Linears).
- **h2h is THE gate, top-1/KL are proxies.** `reuse_penc_attn` had great KL 0.029 / top-1 0.858 but **h2h 0.383** (failed live). Gate on live h2h + auto-revert; never trust the proxy.
- **What you pool away, the student re-derives poorly** — feeding the matchup tensor directly moved h2h 0.383→0.467 while top-1 barely moved (0.858→0.863): the fidelity win was in *live play*, not static accuracy.
- FitNets/feature-matching **hurt** (frozen-body head; logit-KL alone is better). Action-factored pointer head hurt. Turn-history was NOT the missing input (it's noise for shallow nets).
- **Frontier hardens as the teacher grows** (deeper body, richer obs, league play) → plan for "~3× faithful, decaying toward 2×"; distill **opportunistically** only snapshots that pass the gate; prefer fidelity over speed (an unfaithful opponent silently corrupts self-play).

Experiment harness was `/tmp/distill/` (ephemeral): cached ~31.5k on-distribution decisions (teacher-vs-teacher/bot via the bridge), a student zoo reusing the teacher's frozen `unpack`+`embeddings`, soft-KL train + `fidelity` + `time_decode` + bridge `head_to_head`. Reconstructable from the design doc.
