---
name: project_loss_analysis_run20260531
description: "Forensic loss analysis of run_20260531_182804 (eval 17M/18M/19M) — all-or-nothing policy, move-effect obs gap, clean harness; retrain plan"
metadata: 
  node_type: memory
  type: project
  originSessionId: 719b1a3e-7067-4904-baf3-caa8ae0155d0
---

Deep loss analysis of `models/run_20260531_182804` eval traces (steps 17/18/19M, ~63% vs bots).
Report saved at `models/run_20260531_182804/LOSS_ANALYSIS.md`. Toolkit at `/tmp/gen3ai_analysis/`
(dump_trace.py, aggregate.py, crosscut.py). Key durable findings:

- **Harness is CLEAN**: 0/10,352 decisions had chosen ≠ argmax(masked logits); obs re-runs
  faithfully. The "clicked move A, got move B" class of bug does not exist here — don't re-chase it.
- **Policy is ALL-OR-NOTHING**: every game ends 6-0 (135/135 wins sweep, 247/248 losses are full
  wipes). No close games. Root context for all "tactical" bugs — brittle momentum/snowball policy
  with no risk management. Driven by offense-heavy dense reward + γ=0.9999 + training only vs 8
  fixed bots (no self-play). [[project_training_versions]]
- **Top obs gap (INFO)**: only action-aligned move features are base-power + type-multiplier
  (reactive block, request order). For status/utility moves these are identical (0, 1.0), so the
  head can't tell Toxic/Protect/Spikes/Recovery apart → wasted-status loops (Toxic into Poison-type,
  TWave into PAR'd, etc.). Fix = ADD action-aligned move-effect bits, NOT re-sort (damaging
  power+mult already aligned; "EQ into Levitator" shows mult 0 correctly).
- **Under-switching (REWARD)**: vol-switch only 8-10%; switching is locally dominated (eats a hit
  for +0.5) while matchup/dead-matchup penalties (−0.15/−0.10) too weak; matchup_penalty fires
  equally in wins and losses.
- **futile_attack (−0.05) far too weak** → can't-break-walls / 80-250-turn can't-close loops vs
  stallers and Random (PP→Struggle). repetition_tax totals are inflated by a few stall games, NOT
  pervasive.
- **Value head mostly fine** (corr w/ material +0.42); over-optimism rare (0.5%), only the
  setup-boost-stacking illusion.

Subagent claims I verified and REJECTED: "add toxic/sleep counter" (already in per-mon block),
"unrevealed bench conflated with fainted" (species_known distinguishes), "explosion +2 incentivizes
self-explosion" (that term is OPPONENT-explosion-survival only — `reward_manager.py:841`).

Retrain plan: P0 self-play; P1 add action-aligned move-effect obs (ARCH bump); P2 reward rebalance
(futile_attack→−0.25, escalate pivot penalties + reward the pivot, dense material shaping, setup
discipline); P3 anneal ent_coef, try γ=0.999. [[project_reward_shaping_verification]]

**P1 IMPLEMENTED (uncommitted, on branch off main 27bf853, NOT shipped):** action-aligned per-move
effect block in the reactive obs — 4 request-order slots × 9 feats [is_boost, is_heal, is_protect,
is_phaze, is_hazard, inflicts_status, status_will_land, pp_fraction, status_will_land_known].
ARCH_SIGNATURE gen3_item_num_fix_v1 → **gen3_move_effects_v1**, obs 3321 → **3357**, REACTIVE_DIM
302 → 338 (matchups now at reactive-offset 50/194). status_will_land_known = prior-vs-confirmed bit
routed with the SAME predicate as the per-mon ability block's `known` flag (_ability_revealed: set
ability ≠ "unknownability") OR a type-certain hard block — closes the discrepancy where abilities
had a known bit but status_will_land didn't. Static flags derived in tools/sync.py build_moves from
Showdown's keying fields (flags.heal, volatileStatus, forceSwitch, sideCondition, primary status,
self-positive boosts) + curated Belly Drum callback override; Memento excluded; Curse + status_will_land
resolved LIVE. status_will_land is a **prior-weighted probability** (gen3_mechanics.status_land_probability +
_resolve_ability_distribution — priors-first-then-confirm, like the matchup ability path; Snorlax≈0.14
Toxic). New: ABILITY_STATUS_IMMUNITY map. Verified: golden fixture regen (3353), obs-build benchmark
+0% (gate ok), model roundtrip ok, full non-e2e suite + new tests pass, move_effects_fuzz_test (17k
slots, 0 violations, prior path exercised), data-quality audit (0 source mismatches, 0 curated-set
misses, 0-power invariant). Data-quality is CI-pinned (moves_test cross-checks + extractor_parity
Showdown-source faithfulness).
