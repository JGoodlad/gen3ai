---
name: project_plateau_diagnosis_2026_06_09
description: "ai_v5_6 @66M plateau diagnosis (13-agent adversarial probe workflow) — GENUINE convergence not measurement artifact; biggest lever = decision-time search (MCTS), then tail-calibrated critic; pool healthy, blind layers refuted"
metadata: 
  node_type: memory
  type: project
  originSessionId: b7d3c1d7-0aa2-4875-9183-6e9491acdea9
---

2026-06-09, run ai_v5_6_stable_N_0608 @66M (ELO ~1924). Question: improvement rate slowing —
what's the biggest lever? Ran a 13-agent workflow (6 probe investigators → adversarial skeptics →
synthesis). The skeptics OVERTURNED the investigators on several points; I verified the load-bearing
numbers directly from TB.

**The plateau is GENUINE convergence, not a measurement artifact or a stale pool.** Over 48M→66M,
EVERY metric is flat/declining: `eval/elo` slope −0.4/Mstep (1920↔1940↔1910), `win_rate_vs_bots`
flat 0.86, `win_rate_vs_pool` declining 0.66→0.59, `td_resid_tail_mean` flat at −10. Optimization
is healthy (entropy −1.2 not collapsed, approx_kl ~0.011, clip ~0.27), aggregate critic well-fit
(explained_variance 0.77, value_loss ~0.19 PopArt units). Self-play pool is LIVE (trainee wins only
~51% vs newest self, pool rotates 26M→64M, sentinel ladder monotonic by recency) — NOT stale.

**Adversarial corrections (don't repeat the naive reads):**
- "Add transformer layers" (naive triage #1) = REFUTED as a blind bet: EV 0.77 flat-healthy, no
  aggregate capacity signal. BUT the real narrower defect is the critic's LOSING-TAIL calibration:
  it sits at V=+15 confidently-winning then craters −50 WITH the OHKO belief present+read
  (max_pko=1.0, salient), and `td_resid_tail` is stuck at −10, not shrinking. Aggregate EV is blind
  to this tail (easy-state-dominated). Much of `critic_blindspot` is IRREDUCIBLE (hidden last mon +
  freeze/confusion/damage-roll RNG); the high-V craters analyzed were a non-representative subsample
  (full bucket median V≈5.3).
- "Refresh self-play pool" = REFUTED (gradient already live at ~51% vs newest self).
- "Turn on switch_bias_weight" = WEAKENED to ~0.5%: the existing stay-tax fires only when
  P(KO)·(1−P(outspeed))≥0.5, but the dominant under-switch cases are outspeed≈1.0 → tax is SILENT
  on exactly them; 11/20 are the critic_blindspot signature, not a reward gap.
- `surprise_ohko` (grew 6.6%→15.7%) = CONFIRMED small-but-real obs gap: skeptic spot-checked 8/8
  killing moves were UNREVEALED (board.opp.moves==[]) → belief defaults to pko≈0.02 on a revealed
  mon whose killing move it hasn't seen. Same root as the hidden-last-mon critic craters.

**Recommended levers (ranked, evidence-grounded):**
1. **Decision-time search (PPO+MCTS / lookahead)** = highest ceiling, the "push ourselves" move. A
   converged reactive policy net + a PopArt-calibrated value net is exactly the MCTS substrate;
   matches [[reference_wang2024_thesis]] (PPO+MCTS broke this same plateau, rank 8 / 1693 Elo). NOT built.
2. **Tail-calibrated critic** (attacks critic_blindspot+positional_grind ≈49% of losses + the stuck
   td-tail): tail-weighted value loss (upweight high-|TD|/near-terminal) + make the value-CLS attend
   to the attrition clock (Toxic-ctr/PP/HP active-slot saliency is only ~0.02× in zero-incoming
   heal-wars → fold Toxic-ctr/PP into the incoming_damage group). Cheap, falsifiable.
3. **Unrevealed-threat belief coverage** — the one provable obs gap; a "revealed mon has unrevealed
   move slots / opp has unrevealed team slot" coverage FLAG (provide-fact-let-it-learn per
   [[feedback_provide_vs_learn]], NOT baked Smogon damage). Pairs with [[project_incoming_damage_outcome]].
4. **Fix the yardstick (do regardless):** promote anchored ELO + a fixed external-anchor LADDER
   (2-3 frozen strong checkpoints, not just ext_ai_v5_5) as the primary metric; win_rate_vs_bots is
   saturated above ~48M. (Measures the levers, doesn't move them.)

DO NOT: blind transformer depth, pool refresh, switch_bias_weight as the headline. Triage recov% is
an UPPER BOUND — cheap obs/reward levers each buy only ~1-2% wr now (they're largely spent), which is
WHY the rate slowed. See [[project_loss_triage_tool]], [[project_throughput_profile]].
