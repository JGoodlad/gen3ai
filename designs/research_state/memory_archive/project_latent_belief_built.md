---
name: project_latent_belief_built
description: "The latent-belief head (predict the hidden opp mon's identity in pokemon_encoder role-token space via cosine-to-stop-grad-encoder-token) is BUILT on branch claude/latent-belief (config 18) — the role-probe's designed escalation; LEARNS but UNMEASURED if it helps the policy"
metadata: 
  node_type: memory
  type: project
  originSessionId: 077e20a7-9988-4394-bbfa-fceff55cc1c7
---

> **Archived 2026-09-08** — branch build record, ai_v5 era; superseded by the shipped belief stack. Preserved verbatim; nothing below is current.

2026-06-13 (branch `claude/latent-belief`, off main's config-17 state, `MODEL_CONFIG_VERSION` 18,
`--opp-belief-latent-coef`). BUILT the **latent-belief escalation** the role-probe designed
([[project_belief_latent_role_probe]]): the user asked to implement `design_latent_predictive_representation`
"for ONLY predicting the opponent mons, not moves" — which (after clarifying the doc-name-vs-scope
mismatch) = the LATENT escalation of the existing species head ([[project_hidden_team_belief_built]]),
NOT the doc's outcome-predictor. **User chose the ENCODER-ROLE-TOKEN target over the cheaper
species-embedding** (the validated rich geometry; ~3× more plumbing).

**Architecture (the role-probe's "right v1", as-built).** Each unrevealed opp slot's refined
(post-transformer) token → an **asymmetric SimSiam predictor** on `BeliefHead` (the pre-cut `"latent"`
dict-key seam) → regressed (COSINE, MSE forbidden under PopArt) toward the **STOP-GRAD `pokemon_encoder`
role-token of the TRUE hidden mon**. TARGET = the model's OWN `pokemon_encoder` run over a **privileged
12-slot block `[live our-team, true hidden-opp-team]`** under `no_grad` (SimSiam stop-grad; the encoder
is TASK-ANCHORED → no EMA, no collapse), opp-half taken. The true-mon features ride a NEW training-only
`belief_target_slots` [6,107] obs Dict key = the **fresh per-mon obs encode** (`pokemon_encoder.encode(
mon, battle2, is_own=True)`; a hidden mon is untouched → its current encode IS fresh), placed at its
believed slot by the **SAME `assign_hidden_to_slots` assignment as `belief_species`** (refactored to one
source → no drift), per-battle cached by species. Loss rides the **SAME species-CE Hungarian assignment**
(one mon per slot across both heads). **VICReg variance floor + `belief_latent_std` NO-GO monitor**
(belt-and-braces; the task-anchored stop-grad is the primary collapse defense). AUGMENTS the species CE
(banked fallback); **moves untouched** (per the request). Believed-slot matchups are already neutral
(hidden) → the target is a clean identity encode "for free."

**Robust to RUN.** OFF byte-identical (`BeliefHead(latent_dim=None)`, no predictor); `opp_belief_latent`
version-checked (v18, bool, check_compatible, requires `opp_belief_slots`); `--opp-belief-latent-coef`
read back on flagless resume; **threaded into `arch_toggles_from_model` + `current_model_version`** so a
latent-ON self-play run doesn't FATAL on its own sentinels (all opp-load sites). CLI enforces
`--opp-belief-latent-coef>0` REQUIRES `--opp-belief-aux-coef>0` (rides the species head + Hungarian).
**Leak-safe**: target stashed in `last_belief_target_latent`, NEVER concatenated into pi/vf;
**`is_grad_enabled()`-gated** so the 2nd encoder pass runs ONLY in train()'s evaluate_actions (skipped in
no-grad rollout/eval). Value-neutral `slice_pokemon_categoricals` refactor (ObsUnpack ↔ privileged encode
share one slicer). Obs build untouched (`belief_target_slots` off the encode hot path; benchmark 7253
calls/encode = baseline).

**Verification.** 2403 unit + new latent tests (cosine/VICReg/grad/rides-species-matching + the **no-leak
gate** `test_latent_target_is_no_leak` + no_grad-skip) + **`belief_target_fuzz_test.py`** (2848 real-bridge
decisions: `belief_target_slots` == an INDEPENDENT fresh-encode of the actual hidden mon the species label
names, bit-for-bit; PAD zero; no leak) + belief_labels fuzz regression + **smoke** (roundtrip PASSED;
latent LEARNS cosine 0.20→0.42, std 0.59→0.75 = NO collapse, loss falling, no NaN). **6-agent ultracode
adversarial review** (empirical verify — refactor reviewer ran old-vs-new slicer side-by-side, 2000-trial
assignment equivalence; all 6 dimensions clean; 1 dead-code nit fixed).

**HONESTY GATE (UNMEASURED).** It LEARNS the role-token (cosine climbs immediately) but whether it HELPS
THE POLICY is unmeasured — belief is a MEANS. The role-probe's REAL open gate still stands: **decodable ≠
helps** (revealed-role is trivially in the obs; value is UNREVEALED-only) + the **unrevealed-slot
inference probe** ([[project_belief_latent_role_probe]] "NEXT"). Falsify-after-build = fresh-run A/B
(coef>0 vs 0) where the **surprise-OHKO / hidden-mon crater share falls** AND wr non-regresses. NOTHING
shipped/run. Did NOT touch `designs/research_state/levers/hidden_team_belief.md` (explicit-only; flag for
a ship). Pairs w/ [[project_model_frontier_roadmap]] lever 4, [[feedback_research_state]].
