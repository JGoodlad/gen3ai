---
name: project_archetype_competence_gradient
description: "Measured per-archetype behavior/value of ai_v6_13 — not ridiculous, but a confidence/value style gradient; the decisive competence-vs-matchup test"
metadata: 
  node_type: memory
  type: project
  originSessionId: 58ec26e7-8e3f-43d8-b15a-0709256c9b1d
---

> **Archived 2026-09-08** — ai_v6_13/ai_v7_02-era archetype probe; era closed per UNDERSTANDING.md §1. Preserved verbatim; nothing below is current.

2026-06-27 exploration pass (NOT shipped, NOT promoted to research_state ledger). Full report:
`designs/research_state/team_archetype_and_exploration_report.md`. Diagnostic script `/tmp/archetype_probe.py`
(model-free ProbeSession + reconstruction movesets; archetype labeler validated 75% vs the 32 tagged sample teams).

**Question answered** (user's "was it being reasonable or ridiculous across team styles?"): on `models/ai_v6_13_outgoing_dmg_0620`,
1030 late-step eval battles. NOT ridiculous, but the honest read is subtler than "plays each style right":
- **Move-mix differentiation is mostly a MOVE-AVAILABILITY artifact** (offense teams have no recovery move → "0% recovery" isn't a choice). So raw "archetype-appropriate play" is overstated.
- **On the availability-FREE axis (switching), the policy does NOT differentiate** — switch rate is FLAT ~30% across offense/balance/stall AND every speed tercile (connects to the under-switch/COMMITMENT lever: the one free knob is archetype-blind).
- **Policy-INTERNAL signals stratify hard** (not availability): confidence (softmax-on-chosen) 0.45 offense vs 0.37 balance/stall; **turn-1 P(win) 0.69 offense / 0.55 stall / 0.46 balance** (the WinProbHead, already on+calibrated in ai_v6_13); turn-1 V monotonic with speed (−13.45 slow → −6.72 fast). => a **style-competence gradient**: crisp/high-value offense, tentative/low-value defensive styles.

**Why it matters / reframe:** this MEASURES the [[project_l3_oracle_grind_l4]] "lost from turn 0" debate. Low turn-1 V on stall/balance = the **policy-conditional-V** effect (V low because THIS policy expects to win less), indistinguishable-at-V from "pilots them worse". The user's "turn-0 win-prob by team comp" idea WORKS as a diagnostic and is the cheap monitor to build first.

**Decisive test (GO-TO-BUILD):** the **archetype-specialist probe** — fine-tune a balance-only/stall-only specialist from the current ckpt; if its balance win-prop climbs >> generalist 0.46 → COMPETENCE gap (coachable → curriculum/specialists-distill); if flat → matchup ceiling (stop chasing). Per-archetype version of can-anyone-win.

**Replay corpus re-exam (2026-06-27, `/replays` = 137k gen3ou rated ladder `.log`):** scans `/tmp/replay_scan.py`+`replay_scan2.py`. Two realities reshape the "group winning teams by archetype→train/exploit" idea: (1) MID-LADDER — winner rating median **1242**, ≥1500=10%, ≥1700=0.3% → a HABIT-CORRECTOR not an expert teacher; explains "limited value before"; BC-from-weak-data can DRAG a decent policy down → rating-filter hard + prefer AWR/offline-RL over pure BC. (2) PARTIAL reveals (gen3 no preview) — 44% reveal 6 mons, ~1.9/4 moves/mon — BUT meta is concentrated (canonical cores recur 100-248×, e.g. Blissey/Gengar/Skarmory/Starmie/Swampert/Tyranitar ×248) → **aggregate-by-species-set + usage priors recovers full teams**. KEY POSITIVE: strong (≥1400) winners are archetype-BALANCED (~27% stall-ish) → the weak styles the model fears ARE present (~1000 stall winning teams). PLAN (robust-first): (1) curated archetype-BALANCED winning-team POOL [uses teams not play → immune to obs-degradation, do first]; (2) targeted offline BC/AWR on WEAK archetypes [pre-test: moves under-switch + balance turn-1 wp?]; (3) EXPLOITER on strong-archetype teams vs current policy [synthesis of "train+exploiter"]. Latent-grouping good for discovery, hand-labeler for curation. Weaker instance of the Metamon ceiling-raiser ([[project_plateau_research_2026_06_25]]).

**BUILT + smoke-verified 2026-06-27 (current main ee8f2b9, config 42): the archetype STYLE-SPECIALIST.** `src/agents/training/team_archetype.py` (pure classifier+filter, no Node/torch; 10 tests incl. a ≥65% quality gate vs the 32 tagged samples) + `--trainee-archetype {offense,balance,stall}` (filters ONLY the trainee's team pool; opponent stays diverse → no mirror; OFF=byte-identical). Pairs with the SHIPPED `--exploiter <main>` (single fixed target, no self-play, no self-promote → "don't self-play against itself"): run `--exploiter <main> --trainee-archetype stall`, run dir `models/exploiter_vs_<main>`, watch `eval/win_rate_vs_ext_<main>` (>~0.6 = exploitable stall hole → fold back; ~0.5 = main defends stall). Serverless smoke PASSED (filter fired 92/719 stall kept, episodes ran, roundtrip OK). LIVE POOL: 92 stall / 194 balance / 433 offense (719). Caveats: high `--ent-coef` ~0.13 (basin-lock — else null uninterpretable), single-target overfit (temp-1.0 mitigates), win-rate IS the verdict (70-80% = strong, not assumed). Phase-2 (vs the snapshot POOL) **BUILT+smoke-verified 2026-06-27**: `--exploiter-pool <target-run>` — sole opponent is a FROZEN pool of the target's snapshots (sampled per worker, rotated every 64 eps via a generation bump), no self-promote, plain recency. REUSES self-play `SnapshotPool`+per-gen `pool_player` swap (memory-safe); copies most-recent `--exploiter-pool-size` (def 15) snaps (or fallback checkpoints/) into `<run>/opponent_pool/` at launch (arch-gated FATAL). Mutually excl. with `--self-play`/`--exploiter`. Run: `--exploiter-pool models/<main> --trainee-archetype stall --ent-coef 0.13` (+`--stable-opponents models/<main>` for `eval/win_rate_vs_ext_<main>` verdict). Files: `wrappers.py` `_exploiter_pool` branch, `train_rl_agent.py` `_resolve_exploiter_pool`+freeze-copy+factory, `wrappers_test.py::test_exploiter_pool_*` (4 tests). Fixes stall DEFENSE/opp-distribution (robustness via fold-back), NOT directly stall-PILOTING competence. Distillation subsystem REMOVED from main → specialist is a league opponent, not weights-to-merge.

**Training headroom found (all UNBUILT in live code):** team sampling is UNIFORM (no per-team weighting); self-play pool is recency-weighted only (no loss-weighting); NO training-time robustness/CVaR; NO curiosity; `gen3_defensive_entropy_v1` is NOT in live code. So "equal robustness across teams" (group-DRO/CVaR-over-teams) + "prioritized replay of struggling teams" (by REGRET not raw loss, winnability-gated) are clean cheap additions. Curiosity/RND = LOW fit (mismatched: problem is commitment-to-a-plan, not state-novelty). Caveat: most levers are policy-conditional → reach the per-archetype optimum, don't break the self-play fixed point ([[project_plateau_research_2026_06_25]]); only exogenous data (BC/human replays) raises the ceiling.

**VALUE archetype-conditioning PROBE (2026-07-02, ai_v7_02, 2-agent workflow, adversarially reproduced).** Q: is
V/win-prob RICHLY archetype-conditional (board-response INTERACTION) vs COARSELY shifted (additive OFFSET)? 233,895
trainee decisions / 6,853 battles from ai_v7_02 eval traces; archetype labeled from the reconstruction.json packed
team. VERDICT = MIXED, leaning rich. Held-out grouped-by-battle CV predicting V: board 0.269 → +arch-OFFSET 0.312
(+0.043) → +arch-INTERACTION 0.341 (+0.029); win-prob 0.390→0.422→0.441. Interaction BULLETPROOF (battle-level
label-permutation null 14σ V / 28σ win-prob; SURVIVES a full quadratic board baseline — gap GREW to +0.046 =
orthogonal to board nonlinearity; PERSISTS under label-free cuts speed-tercile +0.018 / recovery-count +0.020).
Archetype-attributable variance ≈ **60% coarse OFFSET / 40% real INTERACTION** (held-out); interaction SMALL in
absolute magnitude (~3 R²-pts). Matched-state autograd (33 HP/alive bins): value weights the SAME feature differently
by archetype — offense≫stall on opp_spikes (rank-biserial −0.44) + our setup boosts; stall values raw HP least. KEY
SYNTHESIS with the behavioral finding above: VALUE reasons ~40%-conditional YET the POLICY's FREE-axis (switch)
behavior is FLAT by archetype → the bottleneck is EXECUTION, not representation. IMPLICATION for the FiLM/shared+routed
"condition-everything-on-archetype" idea (from the OPD/plateau design chats): real but MODERATE headroom on the value
(already 40% there) → condition the POLICY + advantage (not just value), and pair with the INCENTIVE lever (DRO /
balanced sampling / specialist-league [[project_plateau_research_2026_06_25]]) since flat-behavior-despite-conditional-
value == the generalist-Nash equilibrium representation alone can't move. Caveats: archetype labeler ~50% agreement on
the hand-tagged sample (offense→balance mislabels; conclusion survives label-free defs); --unified-obs masks the
incoming-damage obs so threat dims are structural-0 in the gradient (kept as a board descriptor in the regression).
Bounded by learns≠helps + intransitivity (measured REPRESENTATION, not win-rate). Diagnostic scripts /tmp/probe_archetype/.
