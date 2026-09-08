---
name: project_opp_action_head_falsified
description: "The opponent-ACTION aux head (roadmap lever 2) is COMPREHENSIVELY FALSIFIED by the prober — the trunk+policy-head already model the opponent; under-switching is a policy-COMMITMENT gap, not representation"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5295b510-4f56-4871-9eec-0c9ecfba2aff
---

> **Archived 2026-09-08** — ai_v5-era falsification; era closed and the trunk it probed no longer exists. Preserved verbatim; nothing below is current.

2026-06-12 (branch claude/trusting-cannon-9610be). Gate-0 falsified the **opponent-action /
forward-model aux head** ([[project_model_frontier_roadmap]] lever 2, "shared scaffold") BEFORE
building it — owner asked "did we falsify this first?" The prober's representation probe (linear
probe of head features, 5-fold OOF/auto-L2, ai_v5_11 @53M) shows the trunk AND the policy head
already model the opponent across every dimension:
- opp_switches (hidden simultaneous switch, NO provided feature — the clean load-bearing pillar):
  AUC **0.89 vf / 0.90 pi**
- which move TYPE (macro 1-vs-rest, 14 types): **0.93 vf / 0.96 pi** (prior 0.23)
- which mon they switch TO (8 species): **0.88 / 0.88** (prior 0.15)
- big_hit_incoming 0.75/0.78 (beats provided feat 0.61), faint_soon 0.86 (beats 0.57)
→ opponent modelling COMPREHENSIVELY present incl. high-cardinality IDENTITY. **DO NOT build the
opp-action head** (~0.80 conf). An aux loss re-encodes info already IN the policy head → can't
fix a decision problem; and it is an ARCH_SIGNATURE-bump build (evaluate_actions returns only
values/log_prob/entropy), not a flag flip — high bar, not cleared.

**Adversarial workflow (4 skeptics+synth) OVERTURNED two of my own sub-claims — record the kills:**
1. **opp_status_move (0.82-0.87) is LEAKY** — the obs already encodes each revealed opp move's
   category flag (moves.py) which flows into the trunk → re-decodes an INPUT, not anticipation.
   Caveated in code; NOT valid falsifier evidence. (Same partial leak applies to opp_move_TYPE; the
   switch-target probe has no such leak.) The clean falsifier is opp_switches alone.
2. **"Information-blind switching" is REFUTED** — my switch_vs_info corr(switch, #revealed-opp-mons)
   = -0.025 was a CONFOUNDED NULL (revealed-count is monotone with game progress → Simpson erasure,
   wrong axis). The double-switch data proves the policy IS information-conditioned: switch-rate
   jumps 0.229→**0.573** the turn after the opp switches (2.5×), 0.066 after our own switch.

**The REAL lever (reframe): policy COMMITMENT / temperature, NOT representation.** The policy's SOFT
switch-prob (0.28) already ≈ strong-human (0.30) — so under-switching is an ARGMAX/commitment gap
(the mass is right, the argmax under-commits). Knob = `--switch-bias-weight` / `ent-coef` /
temperature; no arch build. (Note plateau_diagnosis had switch_bias on a DON'T list, but it now has
a target anchor AND a mechanism — argmax sharpness, not valuation.) Validate against a
confound-controlled WITHIN-state target (regress switch ~ revealed+turn+mons_alive, partial coeff;
or P(KO)-threat within fixed cells). **Separate REAL gap = HIDDEN/unrevealed-team belief** (a
DIFFERENT head: surprise-OHKO craters pko<0.3, [[project_incoming_damage_outcome]]; K-CLS belief
queries are "untrained capacity" without an aux objective — design_offense_and_opponent_belief.md).

**Built (branch, not shipped):** prober probes `opp_status_move`(new, leak-caveated) +
`switch-vs-info`(new, confound-caveated) in session.py/query.py + 3 helper tests (134 prober tests
green); opp_move_type/opp_switch_target ran as a one-off script (not committed — could become proper
multiclass targets). Honesty-gate debt flagged by the workflow: human-agreement + switch tooling are
off-tree → no number from them anchors a build until committed+guarded. See [[feedback_research_state]],
[[project_human_agreement_probe]], [[project_representation_probe]], [[project_plateau_diagnosis_2026_06_09]].
