---
name: project_hidden_team_belief_built
description: The hidden-team belief aux head (in-place belief slots + Hungarian species/moves objective) is BUILT on branch claude/belief-head (config 16) — the un-falsified gap the opp-action falsification named; robust for self-play/stable/distill; UNMEASURED if it helps the policy
metadata: 
  node_type: memory
  type: project
  originSessionId: 5295b510-4f56-4871-9eec-0c9ecfba2aff
---

> **Archived 2026-09-08** — belief-slot build record; the head is production and stated in designs/ARCHITECTURE.md. Preserved verbatim; nothing below is current.

2026-06-13 (branch `claude/belief-head`, off main's config-15 state, `MODEL_CONFIG_VERSION` 16,
`--opp-belief-aux-coef`). BUILT the **hidden-team belief aux head** — the SEPARATE un-falsified gap the
opp-action falsification named ([[project_opp_action_head_falsified]]): Gen3 has NO team preview, so the
~3 unrevealed opp slots are ABSENT from the obs (a probe CAN'T recover them — the exact opposite of the
falsified opp-ACTION head). Pre-build **learnability probe** (`models/saved_work/team_completion_learnability_pool.py`):
conditioning on revealed mons beats the marginal usage prior **+7pp recall / +8–10pp top-1** → real
signal. It is the one "predict-it" lever passing the *unknown ∩ action-changing* test.

**Architecture (user drove these calls).** `BeliefSlots` fills the un-revealed opp team slots with
DISTINCT learned unknown-mon tokens IN-PLACE (not zeros), refined by the same 12-token transformer so
BOTH heads attend over the imagined mons as party members (user: "in latent space, the network attends
over it"). `BeliefHead` aux-supervises the refined slot-tokens → species + moves. Matched
**order-invariantly (HUNGARIAN / DETR — user: "do it right once")**: the k believed-slot preds matched
to the k hidden mons by min-CE-cost (k! perms enumerated, vectorised per distinct k), fixing the
reveal-shifting-target defect a review caught. Returns a logits dict so a **BYOL latent target swaps in
cleanly later** (user wants that path; v1 = species+moves classification). Privileged labels from
`battle2.team`, training-only, NEVER in the forward (cardinal leak rule holds — verified by 2 reviews).

**Robust to RUN (the bulk of the work).** OFF byte-identical; `opp_belief_slots` version-checked (v16);
`--opp-belief-aux-coef` read back from saved config on a flagless launcher resume (else FATAL); **`arch_toggles`
threaded into `current_model_version` at ALL 4 opponent-load sites** (in-process pool+stable via
`_run_arch_toggles`; `eval_worker` via cfg; `distill/worker` via config — `snapshot.arch_toggles_from_model`)
so a belief-ON SELF-PLAY run doesn't FATAL on its own sentinels. **Works for self-play AND stable play**
(belief is internal to the forward + needs no labels for a forward; obs["observation"] interface
unchanged; `check_opponent_compatible` is arch_signature-only and belief doesn't bump it). **Fail-loud**
on out-of-vocab labels + a non-contiguous believed mask (single-sourced from the obs `species_known`).
**Perf** measured + optimized (gather-before-softmax, no_grad diagnostics, moves fast-path). **12 TB
metrics** incl. `species_acc_above_chance`, moves P/R, coverage, k_mean + the shared-trunk grad-balance
probe `grad/belief_share`. **Fuzz** (`belief_labels_fuzz_test.py`, 1792 real-battle decisions: labels ==
actual hidden team + no leak) + gradient-flow + interop tests. 2253 unit tests green. NOTHING shipped.

**The honesty gate (UNMEASURED).** It LEARNS (`species_acc` 0.08–0.16 vs ~0.003 chance immediately), but
whether it HELPS THE POLICY is unmeasured — belief is a MEANS. Falsify-after-build: a fresh-run A/B
(coef>0 vs 0) where `belief_species_acc_above_chance` climbs AND the **surprise-OHKO / hidden-mon crater
share falls** AND wr non-regresses. Risk: "learnable but inconsequential" (surprise-OHKO ~as common in
wins 56% as losses 52% → may be informative-not-pivotal; pairs with the under-switching/commitment lever
[[project_opp_action_head_falsified]]). Subsumes the bench/switch-in half of H3 surprise-OHKO. See
`designs/research_state/levers/hidden_team_belief.md`, [[project_model_frontier_roadmap]] (lever 4
team-completion, now built), [[feedback_research_state]].
