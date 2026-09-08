---
name: project_positional_grind_decomposition
description: "DEAD CLAIM — never cite '~⅔ matchup-lost / uncoachable team-draw': the attribution was CIRCULAR (our own policy/critic judged recoverability) and a strong human beats our bot on the same teams ⇒ HEADROOM, not a floor. What survives: the P(win)≥0.5 re-centering (V zero≠even, self-mirror V≈−6.5), under-switching OUTGROWN, tail-critic falsified; levers = league/exploiter + teacher"
metadata: 
  node_type: memory
  type: project
  originSessionId: a6b67bc4-6b85-495f-93e8-837a94b09851
---

> **⚠️ DEAD CLAIM — DO NOT CITE THE "~⅔ MATCHUP-LOST / TEAM-DRAW" NUMBER (owner-corrected 2026-07-24,
> the THIRD time).** The attribution was **CIRCULAR**: recoverability was judged by our OWN
> policy/critic (and shallow search over that critic), so "unwinnable / uncoachable team-draw" only
> ever meant **"unwinnable by a policy of THIS strength"** — which is precisely the thing we are trying
> to improve. The falsifying test is simple and decisive: **a strong human beats our bot easily on the
> same teams**, so those games are not team-draw. Treat that bucket as **HEADROOM, not a floor**, and
> never use it to argue a lever is not worth building. (Prior corrections:
> [[project_l3_oracle_grind_l4]] 2026-07-03, [[project_exploiter_no_team_advantage]] 2026-07-14 — the
> equal-pilot mirror already showed the exploiter's wins were exploitation, not team advantage.)



2026-06-20 (ai_v6_11_typed_hp_0619 @34M, prober V-trajectory analysis, adversarially verified). Answers the recurring
"is the ~17% bot-loss / ~1880 plateau a real skill ceiling, and can training fix it?"

**METHOD FIX (important):** the triage `positional_grind` label only means "critic V(s)≤0", but this run's critic runs on
a negative-biased shaped scale (γ=0.9999) → turn-1 V is negative in ~97% of WINS too. So "V≤0" is the early-game baseline,
NOT "the critic knew it was losing." The valid separator is **"did V ever cross ABOVE 0?"** — 100% of bot WINS do, only
70% of bot LOSSES do.

**THE SPLIT (stable across 30M/32M/34M, ~32 grind losses sampled, model-free):**
- **~⅔ (66%) MATCHUP-LOST** — V never crosses >0, behind from turn 1. Modest draw disadvantage (BST ~524 vs ~539 thrown
  vs ~550 wins). NOT a play-skill bug; structurally hard to fix in-policy (random teams, NO Gen3 team preview, no team-building).
- **~⅓ (34%) THROWN** — was ahead/even (V>0) then cratered, always LATE (median turn 16, every one ≥ turn 7; NONE thrown
  in the opening). Mechanism = **conversion/closing failure vs RECOVERY-STALLERS** — repeats a no-progress move into a wall
  (HP-Flying into a Recovering Milotic; 4× Aromatherapy while own Blissey chipped to 14%) and fails to close.

**TWO BELIEF UPDATES:**
1. **Under-switching is OUTGROWN.** The old [[project_human_agreement_probe]] "policy switches 16% vs human 30%" no longer
   holds at 34M: thrown-grind switch-rate 0.30 ≈ matchup-lost 0.31 ≈ human 0.30. The agent pivots plenty; grind is NOT an
   under-switch problem. (Under-switch/temperature lever, if anywhere, now targets surprise_ohko, not grind.)
2. **Tail-weighted critic is FALSIFIED for grind** (research_state kill K1): residuals SUB-GAUSSIAN (tail-dom 0.33, exkurt
   −0.89) — no fat tail to re-weight; the "fat tail" was outcome-conditioning + a PP-stall reward artifact. Corrects my prior
   "value-dist shaping = top lever" — see [[project_value_dist_head_status]].

**HUMAN-95% IS CONFOUNDED, not apples-to-apples:** gen3ou, supplied teams, BOTH sides draw a random team per battle from the
~~[RETRACTED — circular, see the banner above]~~ same 770-team pool, NO team preview, agent does no team-building/scouting. So ~⅔ of grind ≈ draw variance + no-preview info
deficit = play-skill-INDEPENDENT. The clean skill gap is well under 17% (~the thrown ⅓ + surprise_ohko ~19%). A human picks/
scouts/builds a known team — advantages the agent structurally lacks. (In the agent's favor: bots draw the SAME pool, so it
isn't handed a worse team than them — the asymmetry is purely vs the human's agency.)

**WHY IT CAN'T BREAK STALLS (2026-06-20 follow-up, grounded in code+traces).** It HAS + USES every tool (98% of teams
carry ≥1 stall-breaker; across 9169 traced decisions: CalmMind 206×, Toxic 188×, TWave 174×, Spikes 131×, Sub 113×, DD 76×)
→ NOT a coverage gap, it's timing/commitment. Real drivers, ranked: (1) LONG-HORIZON CREDIT ASSIGNMENT (Toxic/Spikes/setup
do 0 immediate dmg; the KO lands 4-10 turns later; at γ=0.9999 over a 30-turn grind the break-move's advantage is diluted to
noise; craters are LATE, median t16). (2) NO TEACHING OPPONENT: it already beats the stall BOTS (0.90/0.87, they stall to a
fixed 50%-HP threshold); the long grinds are SYMMETRIC greedy SELF-PLAY (win_rate_vs_pool pinned 0.56, ep_len 35 vs 23 vs bots)
where a frozen copy under-closes identically → neither side punished for failing to close → NO gradient. (3) **REWARD IS SPLIT
(corrected my naive "no-progress tax penalizes all stall-breaks" — WRONG):** under `--all-shaping-pbrs`, landing status /
setting a hazard / ongoing-residual chip RESET the no-progress clock (progress_clock._is_progress clauses ii/iii/v → NOT taxed).
**BUT SETUP (CM/DD/SD/Curse/Sub) falls through to the no_progress_tax: −0.15 vs only +0.06 Φ_boost = NET −0.09/turn (Sub ≈−0.15,
Φ_boost only tracks STAT stages not Sub), flat per repeat** → the reward DISCOURAGES the setup route to break a wall
(reward_manager.py _apply_pbrs_suppression L1496, _compute_phi_boost, BOOST_WEIGHT=0.03, no_progress_penalty=0.15;
progress_clock.py _is_progress L229-266). **REFINEMENT (user-spotted, code-confirmed 2026-06-20):** the real flaw is that
`_is_progress` has 5 clauses (damage / opp-status / hazard-layer / forced-opp-commit / our-owned-residual) and **NO clause for
"our own boost stage rose" or "we created a Substitute"** — so it taxes a PRODUCTIVE first CM/DD/SD/Sub the SAME as a redundant
+6-cap CM or an already-up-Sub. It conflates useful vs redundant setup. So the fix is NOT "remove the setup tax" (that'd also
stop taxing the genuinely-idle +6 spin) but **ADD a progress clause: a non-redundant boost-stage increase OR a newly-created
Substitute → progress (reset, no tax); leave maxed/failed setup as the charged no-op** (+ probably a per-mon setup-turn budget so
"boost into a wall forever" can't reset indefinitely). Feasible/low-risk: `live.ours.active.boosts`/`volatiles` carry the data
(TurnDelta doesn't expose own-boosts yet), and the clock already has the exact `_prev_our_spikes` tracker pattern to mirror
(`_prev_our_boosts`/`_prev_had_sub`). Pairs w/
[[project_markovian_reward_design]]. CONCRETE FAILURE SIGNATURE: loops the single highest-prior move (HP-Flying into a FROZEN
mon 3 turns; Aromatherapy 4× curing nothing) while the winning chip/pivot (Seismic Toss, Curse-Swampert) sits legal at 1-15% prob.
FIX = a league EXPLOITER that stalls well (Recover-wall+status, plays to outlast not a fixed threshold) to manufacture the
gradient + REMOVE the −0.09/turn setup tax (give setup a PBRS potential telescoping to the KO). 3 example losses:
staller/loss_s3_004, heuristic/loss_s0_005 (thrown), heuristic2/loss_s0_002 (matchup-lost). Loss-biased forensic traces → absolute
win-rates deflated, directional signal robust.

**CAN MORE TRAINING FIX IT?** More of the SAME self-play+risk-neutral-PPO = NO (converged; co-plateaued pool can't teach a
skill no member has → no closer-punishing gradient; risk-neutral PPO indifferent to variance-management). Only a DIFFERENT
opponent distribution helps the thrown ⅓. RANKED LEVERS: (1) **PFSP league + EXPLOITER** (an exploiter that punishes
failure-to-close — the named structural treadmill-breaker, not built); (2) **offline teacher / human-ladder BC** (inject
external "how to close"; `log_reader` foundation built, no trainer); (3) **multi-ply upstream falsify** (diagnostic, cheap —
pinpoint when the thrown line turned, ~2-3 turns before the death turn; research_state's named priority). MCTS ruled out by
user → survives only as offline-teacher-distilled. Realistic upside: a few pts of bot win-rate from the thrown ⅓; the rest is
L4 ceiling + draw variance. All shares are UPPER BOUNDS (single-run, model-free). Pairs w/ [[project_plateau_diagnosis_2026_06_09]],
[[project_loss_triage_tool]], [[project_model_frontier_roadmap]], [[feedback_research_state]].

**2026-06-27 RE-CENTERED THE GRIND/THROW THRESHOLD — SHIPPED, and it shrinks "grind" by ~⅓.** The split above leaned on
V (the "V≤0" / "did V cross >0" separators). But V's zero is NOT "even": a measured opening study (32 strong bias teams +
SELF-MIRROR, ai_v6_13 @96M) found a provably-50/50 self-mirror position reads **V ≈ −6.5** (PopArt μ ≈ −3.6) — V is a
shaped/discounted RETURN with a structural negative offset, so even "V crossed >0" UNDER-counts "was winning." The traces
already carry a calibrated **P(win)** (win-prob head), so the prober taxonomy now splits grind-vs-throw on **P(win) ≥ 0.5**,
not V-sign (`engine._was_winning`; `triage(wp_even=,v_even=)` / `query triage --wp-even --v-even`; win-prob primary, V-even
fallback re-centerable for head-less runs). RE-RUN on ai_v6_13 (66–104M, **3213** loss craters, 100% carry P(win)):
positional_grind **35%→27%**, critic_blindspot (was-winning-then-threw, coachable) **21%→29%**. **386/1140 (34%) of old
"grinds" FLIP to throws** — the model rated itself WINNING at the cliff. De-biased for the win-prob head's ~+0.12 optimism
(self-mirror reads 0.65 not 0.5 → true even-point ~0.6): **210/1140 (18%) still flip**; median old-grind P(win)=0.40 (most
grinds ARE real). So the @34M "~⅔ matchup-lost / ~⅓ thrown" figure is **biased UP** by the V=0 threshold — the COACHABLE
bucket (per-turn-fixable throws) is bigger than stated (~18–34% of old grinds were mislabeled), which STRENGTHENS the case
for running the [[project_search_teacher]] coef=0 A/B (it attacks exactly that throw bucket). Opening V is also nearly
opponent-INSENSITIVE (weak-bot ≈ strong-self ≈ pool — no Gen3 team preview → V is "my team vs the prior"). Honest caveats:
win-prob head carries an absolute optimism bias (trust the RELATIVE split, not the absolute P(win)); shares not volumes
(loss-enriched). Method lesson: never threshold "winning/losing" on the sign of a shaped-return critic — use the calibrated
P(win) head or re-center on the self-mirror / PopArt μ.
