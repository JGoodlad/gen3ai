---
name: project_model_frontier_roadmap
description: "The BIG-lever roadmap (2026-06-09, 11-agent frontier audit) to make the Gen3 model itself smarter — inputs/signals/architecture, NO inference search. Ceiling = 3 structural holes, not capacity. Build order 1→5"
metadata: 
  node_type: memory
  type: project
  originSessionId: b7d3c1d7-0aa2-4875-9183-6e9491acdea9
---

> **Archived 2026-09-08** — ai_v5-era roadmap; superseded by designs/research_state/UNDERSTANDING.md. Preserved verbatim; nothing below is current.

2026-06-09 frontier-audit workflow (5 auditors → adversarial filter → roadmap). User explicitly RULED OUT
inference-time search (MCTS) and rejected the "near-convergence / irreducible floor" framing (3 weeks in,
tiny net — nowhere near a fundamental ceiling). This is the program to raise the DESIGN ceiling.

**THESIS — the ceiling is NOT capacity; it's 3 structural deficits:** (1) the obs is OFFENSE-BLIND +
OPPONENT-BLIND — precise calibrated incoming-KO belief but only raw type-eff×base-power for OUR offense, and
the ~3 unrevealed opp mons are HP==0 and KEY-MASKED out of the transformer (the model reasons about a 3-mon
opponent); (2) the trunk's ONLY teacher is the scalar PPO advantage/value-MSE — no world-model signal forces
it to encode dynamics, despite the GPU ~86% idle; (3) self-play is a TREADMILL — win_rate_vs_pool pinned ~50%
by the promotion gate → a converged agent gets ~0 gradient toward its weaknesses. Path: PERCEPTION →
WORLD-MODEL TRUNK → TRAINING GRADIENT.

**RANKED BUILD ORDER (verified against code):**
1. **Outgoing-KO belief** (transformational, medium, BUILD FIRST). Mirror incoming_damage.py for OUR offense:
   per (our known moves × each opp mon) calibrated [phys/spec pko, phys/spec expdmg, 2HKO, P(outspeed)],
   uncertainty flipped to the DEFENDER (opp Def/SpD/HP/item hidden→priors at a BULKY tail; our Atk/SpA/moves
   EXACT). `compute_team_block`/`gen3_damage_max`/`p_ko`/`p_outspeed` are already side-neutral. Genuine new
   work: an HP-stat formula (gen3_stat is non-HP only) + extend priors.stat_distribution to def/spd/hp. Route
   BOTH as a non_matchup_rest scalar tail AND per-cell into the move slots (so the argmax sees which move is
   lethal). The ONLY lever that fixes the policy ARGMAX directly (under-switching/futile-attack/bad-trade are
   the incoming side of the SAME blindness). Independent; machinery+data+routing site all exist. Designed
   (project_outgoing_damage_design, PRELIMINARY-GO) never built. Bump ARCH_SIGNATURE.
2. **Forward-model + opponent-action aux head** (transformational, medium). A 4th head off the shared trunk
   (conditioned on the action) predicting next-turn TurnDelta (hp deltas, faints, eff, we_moved_first) + the
   opp's action. Forces a world-model into the rep both heads read. INFERENCE-UNCHANGED (head dropped at act).
   Builds the SHARED SCAFFOLD levers 3-4 reuse: thread the realized next-turn delta onto (s_t,a_t) via the info
   dict + a NEW rollout-buffer column (SB3 stores per-step obs NOT next-obs — real plumbing, "free labels" is
   overstated). loss coef ~0.15, use_forward_model_aux bool like use_popart.
3. **Distributional/quantile critic + reward-component head** (large, medium). Fixes the #1 symptom (critic
   tail-craters, td_tail~-10): a scalar critic can't represent the bimodal "winning UNLESS they have the one
   sweeper" (same mean). 3a (cheap, first): predict the 34 RewardBreakdown terms (already decomposed then
   discarded, reward_manager.py:164) as aux MSE — forces the spread the prober found missing (damage r2=0.06).
   3b: replace value_net scalar with a K~21-32 quantile head + quantile-Huber; policy bootstraps the mean
   (GAE unchanged). NOTE: a flat tail-/terminal-weighted value reweight was DOWNGRADED (near "recalibrate a
   scalar"); the distributional/component objective is the real critic lever.
4. **Team-completion aux head** (transformational, heavy). RE-HOME the ALREADY-WRITTEN-BUT-ORPHANED predictor
   (src/agents/model/team_completion_model.py + training/team_completion/{team_dataset,replay_parser}.py +
   main/train_team_completion.py — imported by NOTHING in model/policy/obs) onto lever-2's 4th-head machinery:
   per unseen opp slot predict species/role/moves/item, supervised by the trace's full 6-mon team (free
   labels). Turns the world-model 3-mon→6-mon. AVOID injecting the static Smogon Teammates table as fixed obs
   (cuts against [[feedback_provide_vs_learn]]); the transformational grade is the TRUNK-internalized head.
   Unlocks downstream: hidden-item belief + OUR→OPP KO under hidden defender bulk.
5. **PFSP league + exploiter** (transformational, heavy, ACTIVATE LAST). Replace the recency-weighted pool +
   win-rate promotion (the treadmill) with PFSP sampling (w~(1-p)^beta, hardest-first; snapshot_pool.py:183 is
   already a parameterized sampler) + a persistent exploiter line (reuse eval-worker/SnapshotPool). Restores
   the gradient. AlphaStar mechanism. Activate AFTER 1-4 — an exploiter vs a still-blind model just farms a
   weakness you're about to fix. ELO is the confirm metric (win_rate_vs_pool structurally can't show it).

Pairs with [[project_representation_probe]] (the validator at each rung), [[project_outgoing_damage_design]],
[[project_stable_opponents_design]], [[project_arch_compute_decision]] (transformer is NOT the bottleneck).
NOT committed. Recommended start: Lever 1.
