# The hidden-info study's Tier 1 under a RED TEAM (2026-10-15)

**Status: the red team of `design_hidden_info_binding.md` §13.2, written into the study as §16.10.** Its job was to BREAK
the Tier-1 conclusions, not to confirm them. Every read here is POST HOC: it was made after the author's read (§16.8) and
the third party's replica comparison (§16.9) had been read, and it changes no registered verdict. The numbers are the red
team's. The author's and the third party's verdicts and confidences stay as they wrote them.

**What was used.** One script, `red_team.py` (this folder), writes `red_team.json`. It is pure numpy and reads:
- the FROZEN blinded table (sha `fc3aced3…`, checked on load);
- the arm key;
- the archived world rows;
- the pilot's Phase 1 truth rows and bank;
- the blind replica's per-state IMPs.

No model, engine or playout was used. Every interval is the registered read's own bootstrap: battle-clustered, 2,000 draws,
seed 20261011, pooled within checkpoint. So each number here is paired with the author's on the same draws.

**Re-derivation.** The script re-derives every LEAF-K8, TRUNC-4-K8 and CRITIC IMP of eight sources from the per-world
outcome arrays. It finds **0 mismatches** with the frozen table, on the registered 600 and on all Tier-1S states.

**Ground rules.** Unless stated, figures are pooled over end2 + prod, on the registered 600 per checkpoint, for LEAF-K8.

## In plain language

**The headline survives.** The opponent's hidden moves are NOT the binding constraint on honest search. The red team found
no reading in which they are. Their reveal-only share never reaches the 0.5 that "binding" needs, under any scorer, on any
checkpoint, or in any subgroup (at most 0.30).

**The second half is weaker than reported.** That half says the never-seen Pokémon carry more than the moves.
- **It survives the attacks it was expected to fail.**
  - Breaking the leaf's ties in three different ways makes it STRONGER.
  - The states the honest builder could not fit do not move it.
  - The world-builder's mis-keyed records never touched a world.
  - The never-seen draws hold the true species exactly as often as an honest draw should.
- **But it does not hold everywhere.**
  - **The opponent.** The registered states come from two kinds of games: against five scripted bots (58 %) and against
    the network's own frozen snapshots (42 %). U's dominance lives entirely in the bot games. On the snapshot games the
    moves are worth as much as the never-seen Pokémon. By sufficiency the moves even lead (point estimate). Hiding the
    moves alone opens more than half the gap there. The difference between the two populations is clear on both sides.
    It survives adjusting for how many Pokémon and move slots are hidden. It appears on both checkpoints.
  - **The architecture.** On the end-state checkpoint alone, the architecture we now run as production, rule R3 is not
    met.
  - **Per hidden fact.** The two classes are worth the same. "The never-seen Pokémon carry more" mostly says that they are
    the bigger bundle of hidden facts: about 2.1 Pokémon of 8 facts each, against 10.4 hidden move slots.

**Two claims around the dose are broken.**
- **"Search needs 0.2–0.35 of weight on the true world"** is an artifact of the 8-rollout leaf's coarse steps. That leaf
  moves only in units of 1/8, so the first weights that can change its pick are 1/5, 1/4 and 1/3. On a smooth 4-turn
  leaf the curve rises from a weight of 0.05, with no step.
- **"The network's belief is no better than Smogon usage"** is false for species. Corrected for the mis-keyed records,
  the network's worlds hold the true never-seen species 41 % of the time against Smogon's 18 %. That bought at most
  +0.017 of value.

**What survives with a qualifier.**
- **No stage has an open search gate.** True by the rule, but all-six-seen misses on every condition by a hair.
- **Foul Play's hand evaluator is no better than our critic.** Not shown either way, and the point estimate favours the
  hand evaluator.

**Fully intact.** The pool posterior's 0.80 recovery is team identification. On the 11 % of states where its worlds never
hold the true team, it recovers 0.14 [−0.27, 0.41].

## 1. The claims, one row each

| claim | red-team verdict | the evidence (red team's, this file's sections) | the strongest remaining doubt |
|---|---|---|---|
| **C1** R-MOVES REFUTED by R3: the never-seen Pokémon (U) dominate the seen mons' hidden moves (M) on both sides | **WEAKENED.** "The moves are not THE binding constraint": SURVIVES (B3 fails in every read). "U dominates M": holds on the registered natural mix and survives ties, exclusions, the records bug, leak checks and the U-arm construction checks; it FAILS on end2 alone, on the snapshot-opponent states, at k = 5, on one half of the referee's seeds and under the strict worst-case imputation; per hidden fact the classes are equal | §2.1, §2.2, §2.5, §2.6 | U's dominance is a property of the bot-game states (the bot − snapshot difference is clear on both sides, k- and volume-adjusted); U is still single-implementation |
| **C2** the honest gap G ≈ +0.070 (replicated) | **SURVIVES**, scoped to the 8-rollout leaf | G > 0 under every scorer: LEAF-K8 +0.070; tie-free LEAF-K8 +0.076–0.077; TRUNC-4-K8 +0.037; CRITIC +0.018 [+0.009, +0.026]; bots +0.074, snapshots +0.065 | **H-REFEREE**, never run: the oracle knows how THIS greedy opponent will play on; a second referee (§3, E3) prices that part. "Items, abilities and spreads add ≈ 0.003" holds only when nothing else is known: once the never-seen mons are known they add +0.019 [+0.010, +0.028] (§2.1) |
| **C3** R-EVAL: information-bound, not evaluator-bound | **WEAKENED** | INFORMATION-BOUND support stands (g_HE / g_CRIT 0.75). "Not evaluator-bound" is a near miss toward the opposite: HE-TRUNC-4 − TRUNC-4 in F **+0.0075 [−0.0008, +0.0155]**; the one-sided tail at the +0.0005 bar is 5.5 %; positive on both checkpoints (+0.0066 / +0.0084) | a hand evaluator ≈ 0.7 points per decision better than our critic in honest worlds is NOT excluded; R-EVAL has no second implementation |
| **C4** the stage frontier is EMPTY | **WEAKENED** | EMPTY by the registered rule, but k = 6 misses each condition narrowly: pooled TRUNC-4 − CRITIC **+0.024 [+0.006, +0.044]**, at 99.6 % [−0.002, +0.055] (holds at 98.8 %); end2 +0.021 [−0.002, +0.047], prod +0.027 [−0.0003, +0.057]; n 145 / 131 < 150 because F's exclusions concentrate late (31 of the 103 registered exclusions are at k = 6) | k = 6 is a probable frontier cell that this sample cannot confirm; "EMPTY" should not be read as "search never pays" |
| **C5** the dose: P-STAR 0.1 is a tie-break artifact; the real dose needs 0.2–0.35; FN ≈ Smogon | **BROKEN** in its two quantitative parts; the tie-break finding itself SURVIVES (it replicated) | "0.2–0.35" is the K = 8 leaf's QUANTISATION: on 1,477 states the truth can move an untied honest pick at p ≤ 0.1 on 3 states only, and the smallest moving weights are 1/2, 1/3, 2/3, 1/4, 1/5, 2/5 (ratios of eighths). On the continuous TRUNC-4-K8 leaf the curve has no step and rises from p = 0.05: +0.002 (0) / +0.009 (0.05) / +0.014 (0.1) / +0.023 (0.2) / +0.029 (0.35) / +0.038 (1). "FN ≈ Smogon" is false for SPECIES: 0.411 vs 0.180 of never-seen slots hold a true never-seen species (counted from the teams); true for moves (0.513 vs 0.508). FN − F: +0.003 [−0.006, +0.012] at LEAF-K8, **+0.008 [+0.0001, +0.017]** at TRUNC-4-K8 (a BOUNDARY result by rule 8) | the value of a better belief is scorer-dependent; a 16- or 32-world leaf (§3, E5) would show how much of the remaining LEAF-K8 null is the leaf's resolution |
| **C6** R-RECOVER: POOL's 0.80 is team identification (does not transfer) | **SURVIVES (strengthened)** | all 8 POOL worlds ARE the true team on 69.6 % of states (r_POOL 1.00 [0.98, 1.02] there); some are on 19.1 % (0.71 [0.53, 0.90]); none on 11.3 % (**0.14 [−0.27, 0.41]**); the median number of distinct candidate teams is 1 | partial identification (19 %) might partly transfer to an archetype-level belief; untested |

## 2. The attacks

### 2.1 The U arms (attack 1): could the never-seen machinery favour U?

**The construction, checked from the stored TEAMS against the bank's true teams** (registered 600; `red_team.json`
`F_u_arm_team_text`):
- **F+U.** Its never-seen slots hold the TRUE entry, at the TRUE slot, on **20,392 / 20,392** never-seen world slots
  (gender field aside).
- **O−U.** Its seen slots hold the true entry on 35,952 / 35,968. The 16 others are one Charizard whose true set spells
  Hidden Power untyped (`hiddenpower`) while the world spells it typed (`hiddenpowerdark`): the same move.
- **Team size.** It is 6 in all 1,200 states, so no team-size leak exists.

**The mis-keyed records** (`8702a56f`, the orchestrator's verdict) moved per-slot provenance, not worlds. Section F counts
the never-seen species from the teams directly:

| arm | share of never-seen slots holding a true never-seen species |
|---|---|
| F | 0.180 |
| F+M | 0.179 |
| O−U | 0.177 |
| FN | **0.411** |
| POOLSETS | 0.186 |
| POOL | 0.757 |
| F+U | 1.000 |

Consequences:
- No IMP, share or R3 reads the records.
- The R-GIGO audit's provenance check never compared records with team text. It also ran **no leak check on the U
  class** (only the seen mons' moves, items and abilities). Its PASS is therefore silent on U.

**A U-leak check the audit never ran.** The builder's own draw, re-simulated without the gate, is sequential, weighted by
the mean Smogon teammate weight of the revealed species, and never repeats a species. It predicts that **0.178** of F's
never-seen slots hold a true species. F's accepted worlds hold **0.180** (4,611 of 25,680). The gate does not condition
the never-seen species toward or away from the truth.

**Slot order is bundled into U.** F+U places the never-seen mons on their true slots. O−U and F permute the drawn mons
among the never-seen slots. Order can matter through engine randomness keyed by slot (Roar / Whirlwind) and through the
opponent policy's own team encoding. It is part of U by the partition. Its size is unmeasured (E7).

**The third class R** = the seen mons' items, abilities and spreads (section I).
- **Telling R is worth almost nothing on its own.** P − F is +0.0015 [−0.009, +0.012].
- **R is worth +0.019 [+0.010, +0.028] once the never-seen mons are known** ((O−M) − (F+U)), and +0.004 [−0.005, +0.013]
  once the moves are.
- **The full 2 × 2 of M and U with R true** (from P = O−M−U):
  - the two-point Shapley values are φ_U 0.048 and φ_M 0.020, difference **+0.029 [+0.018, +0.039]**;
  - the interaction is +0.017 [+0.003, +0.031]: the classes are complements.
- So U's lead is order-robust (first and last), and it does not depend on R being drawn.
- But the "items / abilities / spreads add ≈ 0.003" line of §16.8 is CONTEXT-DEPENDENT. R is a complement of U.

**Verdict on attack 1.**
- The U arms are built as specified.
- They are not biased toward U by the draw, the gate, team size, formes or legality (the F+U entries are the truth
  verbatim; the clause check is honest).
- The records bug touches no outcome.
- What remains: U is a single implementation, and it bundles slot order.

### 2.2 The exclusions (attack 2): not missing at random?

**The key fact: R3 needs no G, so the n-side needs no F.** R3 asks for the sign of two numerators over a positive G. Those
numerators are IMP(F+U) − IMP(F+M) and IMP(O−M) − IMP(O−U). F (and with it the honest builder's exclusions) enters only
the denominator.

**The n-side** does not need F at all. O−M and O−U fail on 0 and 2 states.
- On all 1,198 states: **+0.0273 [+0.0175, +0.0367]**. On the registered read's 1,097 states: +0.0285.
- On the 103 F-excluded states alone: +0.015 [−0.017, +0.048]. Those states are not where hiding the moves costs more:
  the original's ν_M there is 0.022, against 0.029 on the included states.
- The replica's n_M of 1.17 on the 110 states it alone could fit is a between-implementation difference on those states.
  Its ν_M there is 0.068; the original's own O−M gives 0.041. It is not a property of the states.

**The s-side** needs F+U and F+M, which share F's seen-mon screen. It is missing on 143 states.

| imputation on the 143 missing states | s-side IMP(F+U) − IMP(F+M) | R3's s-side |
|---|---|---|
| none (registered read) | +0.0132 [+0.0020, +0.0242] | holds |
| MAR (the included mean) | +0.0132 [+0.0033, +0.0229] | holds |
| the replica's own σ_M there (−0.011 [−0.039, +0.021], 110 states) with σ_U := 0, else MAR | +0.0129 [+0.0031, +0.0227] | holds |
| the same, the other 33 at the worst case | +0.0102 [−0.0001, +0.0202] | BOUNDARY |
| **worst case on all 143: F+M recovers the whole oracle, F+U nothing** | −0.0004 [−0.011, +0.010] | **fails** |
| tipping value (the same imputed contrast on every missing state) | c = −0.014 (included mean +0.013, per-state SD 0.19) | — |

**Verdict.**
- The n-side SURVIVES with every state included.
- The s-side SURVIVES every evidence-based imputation, and falls only if the excluded states carried a sufficiency
  contrast opposite to everything measured there.

### 2.3 The NECESSARY-NOT-SUFFICIENT / SHARED knife edge (attack 3)

Suppose R3 is set aside. The M-only cascade (R1, then NECESSARY-NOT-SUFFICIENT, then SHARED) then reads as follows:

| read | cascade |
|---|---|
| LEAF-K8 | **SHARED**: s_M upper 0.252 |
| LEAF-K8, every tie-free version | **SHARED**: s_M lower > 0 (random tie-break 0.18 [0.08, 0.27]) |
| TRUNC-4-K8 | **SHARED** |
| LEAF-K4 / K1 | NECESSARY-NOT-SUFFICIENT, with B4 HOLDING (n_M 0.52 / 0.58) |
| the snapshot-opponent states | SHARED, with **B4 HOLDING** (n_M 0.58 [0.38, 0.80] on the registered 600; 0.53 [0.37, 0.70] on all Tier-1S) |

**Meaning.**
- The fallback is **MOVES-SHARED**, not a coin flip: the tie-free reads lift s_M's lower bound clear of 0.
- MOVES-BINDING is out of reach in every read, because B3 (s_M ≥ 0.5) fails everywhere.
- The roadmap meaning of C1's first half ("no move-inference programme on this evidence") is the same under every
  fallback.

### 2.4 Common-mode components no replica can catch (attack 4)

| shared component | what it can do to M vs U | size and direction |
|---|---|---|
| **The dice-silent leak** (the true battle seed replays the prefix, so a hidden fact that spends a die without printing must spend the truth's) | only facts that acted before the root: a seen mon's item (King's Rock, Focus Band, Quick Claw), contact ability, speed (S). Never a seen mon's unrevealed move (never used) or a never-seen mon (never acted; the gender draw is padded) | it sits only in arms with R drawn: F, F+M, F+U, the same distribution in each (F+U shares F's very screen draws), and NOT in O−M / O−U. So it cancels from both R3 numerators. It can only shrink G (the denominator, both shares alike), by at most what telling R outright is worth in F's context (P − F = +0.0015 [−0.009, +0.012]). **It cannot flip R3.** |
| **The shared truth rows and referee seeds 16–39** | a referee error moves every arm alike; its NOISE is shared too | split-half (section J): R3 holds on truth seeds 28–39 (s-side +0.25 [+0.06, +0.45]) and NOT on 16–27 (+0.13 [−0.03, +0.29]). The n-side holds on both (+0.36 / +0.46). **R3's s-side rides on the referee's noise** |
| **The referee's continuation (H-REFEREE)** | the oracle and every O−X arm know how the greedy opponent will play on. U holds the opponent's future switch-ins, so U plausibly carries more of this "model of the opponent" than M | UNMEASURED. Direction: toward U. Settled by a second referee (E3) |
| **The shared written spec** | both builders draw never-seen species ∝ the MEAN Smogon teammate weight of the revealed species; a better honest species prior would shrink U | the network's own species belief holds 0.41 vs Smogon's 0.18 and buys ≤ +0.017: a 2.3× better species prior does not shrink U materially at LEAF-K8 |
| **Common random numbers** (world j on dice seed j; the oracle leaf on seeds 0–7) | variance only: the marginal distribution of each arm's leaf, hence its expected IMP, does not depend on the coupling | none in expectation |
| **The registered states** | both implementations read the same 1,200 states | §2.6 (b): the state population is the strongest attack |

### 2.5 The tie-dominated scorer (attack 5): re-read tie-free, from the stored per-world arrays (no re-play)

The honest LEAF-K8 argmax ties on 0.61 of states. The tie shares by arm:

| arm | F | F+M | F+U | O | O−M | O−U | FN | POOL |
|---|---|---|---|---|---|---|---|---|
| ties | 0.61 | 0.61 | 0.57 | 0.58 | 0.58 | 0.62 | 0.60 | 0.59 |

The table below re-reads G, the shares and R3 three ways:
- with ties broken uniformly at random (the exact expectation over the tied actions);
- with ties broken by the same source's one-ply critic;
- with ties broken by its 4-turn leaf.

It adds the continuous scorers beside them.

| scorer | G | s_M | n_M | s_U | n_U | s_U − s_M | n_U − n_M | R3 | s-side one-sided tail |
|---|---|---|---|---|---|---|---|---|---|
| LEAF-K8 (registered) | +0.070 | 0.13 | 0.42 | 0.31 | 0.82 | +0.19 [+0.03, +0.35] | +0.41 [+0.26, +0.55] | holds | 1.05 % |
| LEAF-K8, random tie-break | +0.077 | 0.18 | 0.44 | 0.38 | 0.79 | +0.20 [+0.08, +0.32] | +0.34 [+0.22, +0.46] | holds | 0.0 % |
| LEAF-K8, critic tie-break | +0.077 | 0.14 | 0.43 | 0.38 | 0.82 | +0.25 [+0.09, +0.40] | +0.39 [+0.25, +0.53] | holds | 0.05 % |
| LEAF-K8, 4-turn tie-break | +0.076 | 0.19 | 0.41 | 0.43 | 0.79 | +0.25 [+0.10, +0.40] | +0.38 [+0.24, +0.52] | holds | 0.1 % |
| TRUNC-4-K8 (continuous) | +0.037 | 0.30 | 0.23 | 0.74 | 0.54 | +0.41 [+0.15, +0.70] | +0.31 [+0.04, +0.62] | holds | 0.15 % |
| LEAF-K4 | +0.057 | 0.04 | 0.52 | 0.25 | 0.99 | +0.21 [+0.03, +0.39] | +0.46 [+0.27, +0.65] | holds | 1.2 % |
| LEAF-K1 | +0.031 | −0.13 | 0.58 | 0.32 | 0.89 | +0.45 [+0.13, +0.85] | +0.31 [−0.002, +0.67] | fails (n-side) | — |
| CRITIC (one ply) | +0.018 | 0.27 | 0.37 | 0.53 | 0.57 | +0.27 [−0.17, +0.81] | +0.20 [−0.22, +0.70] | fails (no power) | — |

**Verdict.** The tie attack does NOT break C1: every tie-free LEAF-K8 makes R3 stronger. It does strengthen the
knife-edge reading (§2.3: SHARED, not NECESSARY-NOT-SUFFICIENT).

### 2.6 The red team's own attacks

**(a) Multiplicity.**
- **Within R-MOVES.** R-MOVES refutes on ANY of three routes (R1, R2, R3). At the registered 2.5 % one-sided per bound,
  R3's s-side tail is 1.05 %. Under a Bonferroni correction for the three routes (0.83 %) it would NOT hold at LEAF-K8.
  It would hold under every tie-free LEAF-K8 (tails ≤ 0.1 %). The n-side tail is 0.
- **Study-wide.** The study reads about twenty rules, and none was multiplicity-adjusted except the stage gate's 99.6 %
  level.
- **Verdict.** The headline is marginal at the registered scorer and comfortable at the tie-free ones.

**(b) The state population: the STRONGEST ATTACK.** The registered states come from the checkpoints' own eval traces.
- **The two populations.** 697 states come from games against the five SCRIPTED bots (aggressive, heuristic, random,
  setup_sweep, staller and their v2s). 503 come from games against the network's own frozen snapshots ("sentinels").
- **What differs.** The playout continuation is the checkpoint on both sides either way; only the prefix differs.
- **Registration status.** §10.4 registers the opponent class as a condition axis, for s_M only. The read below is post
  hoc for R3.

| population (registered 600) | n | G | s_M | n_M | s_U | n_U | s_U − s_M | n_U − n_M | R3 |
|---|---|---|---|---|---|---|---|---|---|
| scripted bots | 697 | +0.074 | 0.07 | 0.31 | 0.44 | 0.88 | **+0.38 [+0.19, +0.58]** | **+0.57 [+0.39, +0.76]** | holds |
| own snapshots | 503 | +0.065 | **0.23 [−0.02, 0.45]** | **0.58 [0.38, 0.80]** | 0.12 [−0.10, 0.32] | 0.72 [0.51, 0.94] | **−0.13 [−0.40, +0.13]** | +0.14 [−0.10, +0.39] | **NOT met** |
| own snapshots, all Tier-1S | 708 | +0.068 | 0.23 [0.03, 0.40] | 0.53 [0.37, 0.70] | 0.19 [0.02, 0.35] | 0.74 [0.58, 0.92] | −0.06 [−0.29, +0.17] | +0.21 [+0.01, +0.41] | NOT met |

**The difference is real by every check.**
- **Paired bot − snapshot difference** of the absolute numerators: s-side **+0.036 [+0.015, +0.058]**, n-side
  **+0.032 [+0.011, +0.052]**.
- **Adjusted for k:** +0.028 [+0.005, +0.052] / +0.025 [+0.003, +0.046].
- **Adjusted for the information volume** (the per-state numerators regressed on the never-seen count, the move slots and
  a bot indicator): the bot coefficient is s-side +0.029 [+0.007, +0.050], n-side +0.022 [+0.002, +0.043].
- **Per family**, on the n-side: every non-random bot family is positive (aggressive +0.042, heuristic +0.042,
  setup_sweep +0.056, staller +0.039; each interval above 0). Random is −0.012 (n 55). Snapshots are +0.009
  [−0.006, +0.025].
- **Per checkpoint**, the snapshot s-side is −0.008 on end2 and −0.008 on prod.

**What changes in the snapshot games.** At the same k and the same number of hidden move slots, the moves are worth
1.5–2.4× more per slot there: ν_M per slot at k = 3 / 4 / 5 / 6 is 0.0042 / 0.0033 / 0.0031 / 0.0022 on snapshot states
and 0.0021 / 0.0027 / 0.0013 / 0.0013 on bot states. The mechanism is UNVERIFIED. A plausible one: against a competent
opponent the position is decided by move-level detail, against a scripted bot by what is still in the back.

**Why it matters.** A competent opponent (a ladder human, a strong bot) plays more like our snapshots than like
`random` or `setup_sweep`. On the states closest to competent play, "the never-seen Pokémon dominate the moves" is not
supported. The moves' necessity share there passes B4.

**(c) Not plateaued, and not the current architecture.**
- **Per checkpoint, R3 holds on prod only.** On prod (`rb_es_prod_s2001`, the legacy production network) s-side
  +0.23 [+0.05, +0.43] and n-side +0.51 [+0.33, +0.70].
- **On end2 it is not met.** `rb_es_end2_s2001` is the end-state network, adopted as production on 2026-10-12. There the
  s-side is +0.13 [−0.16, +0.42] and the n-side +0.28 [+0.05, +0.52]. It is not met under any tie-free scorer either
  (random tie-break: s-side +0.12 [−0.09, +0.32]).
- The signs agree, so this is power, not contradiction. But the architecture we run is the one where U's sufficiency
  lead is not detected.
- Both checkpoints still moved about 0.1 nats and flipped 15–22 % of their top actions per 1M steps (§16.3). Every
  reading is PROVISIONAL.

**(d) "More", or "more facts"? Normalised by information quantity** (section E, all Tier-1S).
- **The bundles differ in size.** Per state there are 2.11 never-seen mons (≈ 17 facts at 8 per mon: species, four moves,
  item, ability, spread) against 10.37 unrevealed move slots of seen mons, of which F already holds 5.20.

| unit | value (ν, the hide-only cost) |
|---|---|
| ν_U per never-seen MON | 0.026 [0.022, 0.030] |
| ν_U per never-seen FACT | 0.0032 [0.0028, 0.0037] |
| ν_M per hidden MOVE SLOT | 0.0027 [0.0020, 0.0033] |
| **per-fact ratio U / M** | **1.21 [0.98, 1.58]** |
| per-fact ratio on the reveal-only side | 1.21 [0.68, 3.64] |

- **Per hidden fact, the classes are indistinguishable.**
- **Both show diminishing returns.** ν per never-seen mon rises from 0.020 at k ≤ 2 to 0.047 at k = 5, and ν_M per slot
  falls from 0.0057 at k ≤ 2 to 0.0017 at k = 6. A per-unit ranking therefore swings with the stage.
- **At k = 5** (ONE never-seen mon against 13.4 hidden move slots), σ_U − σ_M is −0.016 [−0.039, +0.006] (the s-side
  reverses) and ν_U − ν_M is +0.017 [−0.0001, +0.035] (a BOUNDARY).
- **U's dominance is a VOLUME result at the stages where U is large** (k ≤ 4).

**(e) Is M's share held down by the move prior's quality?**
- The Smogon move prior holds 0.51 of the hidden moves.
- The network's own move belief holds 0.513, the same, and moves the share nowhere.
- The pool-vs-human read (§16.2) says the moves are intrinsically hard to guess.
- A weaker move prior would RAISE s_M. A better one cannot exist much above 0.51 (the prior-as-truth ceiling, §16.2).
- **The prior's quality is not what holds M down.** The volume and the opponent population are (§2.6 (b), (d)).

**(f) Half the referee.** §2.4: R3's s-side holds on one half of the truth seeds and not the other.

## 3. The experiments that would settle what this read cannot (sized)

Costs use the rows' own measured CPU (§12.13.6):
- Phase 1 truth: 98 s per state (end2) / 58 s (prod);
- a played honest arm: about 24 s / 15 s per state, plus 3–5 s of world building.

Rule 5b's three agent cores give about 2.5 effective cores.

| id | experiment | what it settles | CPU-h | wall on 3 agent cores | GPU |
|---|---|---|---|---|---|
| **E1** | **The independent BLIND U-arm build** (spec below) | R3's never-seen half, by a second implementation | 16 (F+U, O−U only) to 40 (re-running F, F+M, O−M with the learnset fix) | 7–16 h | 0 |
| **E2** | **A competent-opponent state bank**: 600 states per checkpoint from games against the network's own snapshots (more `sentinel` states from the same eval cycles, plus self-play), Phase 1 + F, F+M, F+U, O−M, O−U, with R3 PRE-REGISTERED on it as the primary population | whether U dominates where play is competent (§2.6 (b)) | end2 only: 600 × (98 + 5 × 28) s ≈ **40**; both checkpoints ≈ **65** | 16–26 h | 0 |
| **E3** | **The second referee** (§6.7, H-REFEREE): 300 states, 24 new truth seeds per action with the opponent at temperature 1 (sampled, seeded) | how much of G, and of U's share, is "knowing this greedy opponent". Every arm's picks are re-scored and no arm is re-played | ≈ 3–5, plus ≈ 0.5 agent-day for a pin-side sampled-opponent driver | 2 h | 0 |
| **E4** | **F+Usp and F+M+U** (registered 600) | F+Usp: is U's value the never-seen SPECIES (which FN already predicts at 0.41) or their SETS? F+M+U: closes the R-drawn 2 × 2 (§2.1) | 2 × 1,200 × ≈ 22 s ≈ **15** | 6 h | 0 |
| **E5** | **A 16-world leaf** for F, F+M, F+U, O−M, O−U (8 more worlds each) | how much of the LEAF-K8 nulls (FN, the dose's flat start) is the leaf's 1/8 resolution | 5 × 1,200 × ≈ 22 s ≈ **37** | 15 h | 0 |
| **E6** | **The plateau replication** (§10.5 + the U arms) on X26's plateaued checkpoint | whether any of this survives maturity | ≈ **60** per checkpoint | 24 h | 0 (blocked on X26's plateau) |
| **E7** | **O-perm**: the truth with its never-seen mons permuted among their slots | the slot-order part of U | 1,200 × ≈ 20 s ≈ **7** | 3 h | 0 |

**Order.** E1 and E2 together decide C1's second half: E1 replicates the implementation, E2 replicates the population.
E3 is cheap and is the only check of the strongest common-mode doubt. E4 is the cheapest step toward a payable path
(species vs sets). Then E5. E6 is gated on the plateau. E1 and E2 can share one dispatch: run E1's arms on E2's bank
too.

### E1, ready to dispatch: the BLIND U-arm build

**Who.** A fresh agent (sonnet-xhigh or opus-high), blind to the original. It EXTENDS the blind replica's own code
(`measurements/hidden_info_replica_2026-10-11/replica/`). That code is already independent, and extending it keeps the
replica's F as the paired base: common random numbers within one implementation.

**May read:**
- the replica's own code and README;
- §13.1's hand-over spec, and the U spec text below (copied into the brief);
- the registered states (`value_disc_pilot_2026-10-11/states_<cp>.json`) and the bank;
- the pin checkouts.

**Anti-contamination: forbidden to read** (the brief lists these; the agent's report declares compliance):
- `src/main/value_search/` (all of it);
- `src/agents/info_levers/`;
- `determinize.py`;
- the whole `measurements/hidden_info_tier1_2026-10-13/` folder (scripts, JSON, `red_team.*`, `replica_comparison.*`);
- `~/gen3ai_archive/hidden_info_tier1/`;
- `design_hidden_info_binding.md` §16 (any result);
- `search_ledger.md`;
- `UNDERSTANDING.md`;
- `research_state/ledger.md`.

The brief carries no Tier-1 number. A THIRD party compares the two implementations (as §16.9 did).

**Fix first: the replica's learnset handicap.** Its move prior uses the table before `64a695dc`. It holds 0 % of the
1,500 missing pairs, and those touch a never-seen mon's true set on 54 % of states. Draw from the current table (or with
no learnset filter, as the original does), for EVERY arm it runs here, F included.

**The U spec (hand-over text).** U = every fact of a NEVER-SEEN opponent slot: a mon the searched side has not seen
switch or be dragged in up to the root. That is its species, four moves (Hidden Power's type included), item, ability,
nature, EVs, IVs, gender, and which slot it occupies.

- **F+U.** World j = your F world j, with every never-seen slot replaced by the TRUE packed entry at its TRUE slot. The
  seen mons' draws stay exactly F's. Gate it exactly as F.
- **O−U.** Take the TRUE team. Re-draw every never-seen slot as F does: species ∝ the Smogon teammate prior given the
  revealed species, then a whole set from the Smogon prior, with no species twice and no gen-3 OU clause broken. Place
  the drawn mons on the never-seen slots in a uniformly random order. Every seen mon is verbatim true. Gate as F
  (byte-equal rows AND protocol at every decision of the searched side).
- **Identities to check:** at k = 6 (no never-seen mon), F+U ≡ F and O−U ≡ O outcome for outcome.
- **Seeds and scorer:** world j on dice seed j of `truth.turn_seeds(id, 48)`. LEAF-K1 / K4 / K8. IMP = T(argmax, ties
  toward the policy's order) − T(top1), with the referee on seeds 16–39 as in §13.1.
- **Deliverable:** per state and arm, the 8 world teams, the per-world outcomes and the IMPs; the exclusions with
  reasons.

**States.** The registered 600 per checkpoint (1,200). If E2 is dispatched with it, add E2's bank.

**The comparison** is pre-registered by the dispatcher before the build starts:
- **Primary:** s_U, n_U and R3's two paired numerators on the common set. Agreement means overlapping 95 % intervals AND
  the same R3 conditions, plus |Δ| < 0.005 on σ_U and ν_U (rule 8).
- **Secondary:** R3 per opponent population (bots / snapshots) and per checkpoint.
- **TV:** use a permutation-null TV in place of §12.2's TV < 0.02, which is unmeetable (§16.9).

**Cost.** Minimum (F+U, O−U on 1,200 states, reusing the replica's F): 2 × (600 × 28 + 600 × 19) s ≈ **16 CPU-h**. With F,
F+M and O−M re-run on the fixed prior (recommended: removes the handicap from every share): ≈ **40 CPU-h**. No GPU. At
rule 5b's three cores, 7–16 wall-hours.

## 4. Confidence (the red team's; the author's §16.8 and the third party's §16.9 figures stand as they wrote them)

| statement | author (§16.8) | third party (§16.9) | **red team** | why |
|---|---|---|---|---|
| hidden TEAM information binds honest search (G > 0) | ≈ 90 % | ≈ 95 % | **≈ 90 %** | G > 0 under every scorer and population; H-REFEREE (the oracle's model of this opponent) unmeasured |
| the seen mons' MOVES are THE binding constraint | ≈ 5 % | ≈ 5 % | **≈ 5 %** | B3 fails in every read (s_M ≤ 0.30) |
| the moves are a MATERIAL share (n_M ≥ 0.25) | — | — | **≈ 85 %** | n_M lower ≥ 0.31 pooled; 0.53–0.58 on snapshot states |
| the never-seen Pokémon are the largest single class **on the registered mix** | ≈ 80 % | ≈ 80 % | **≈ 70 %** | robust to ties, exclusions, the records bug and the leak check; single implementation; R3's s-side marginal under multiplicity and on half the referee |
| … **where play is competent** (the snapshot-opponent states) | — | — | **≈ 35 %** | the s-side reverses in point, the n-side is not clear, the bot − snapshot difference is clear |
| per hidden FACT, a never-seen fact is worth more than a hidden move | — | — | **≈ 40 %** | ratio 1.21 [0.98, 1.58]; swings with stage |
| the dose needs ≈ 0.2–0.35 of weight on the true world (a property of belief quality) | stated | — | **≈ 10 %** | it is the 8-rollout leaf's quantisation; a continuous leaf rises from 0.05 |

## 5. FINDINGS (standing rule 7)

1. **U's dominance (R3) depends on the opponent population.** It holds on the states from games against scripted bots and
   is not detected on the states from games against the network's own snapshots. On those, s_U − s_M is −0.13
   [−0.40, +0.13] and n_M is 0.58. The bot − snapshot difference is clear on both sides, k- and volume-adjusted, on both
   checkpoints. It is post hoc, on an axis §10.4 registered for s_M only. E2 settles it.
2. **R3 is not met on end2 alone**, the architecture now in production: s-side +0.13 [−0.16, +0.42].
3. **"The dose needs 0.2–0.35 of weight on the true world" is the K = 8 leaf's quantisation, not a belief threshold.**
   Only 3 of 1,477 states can be moved by the truth at p ≤ 0.1. On the continuous 4-turn leaf the curve rises smoothly
   from p = 0.05. The §16.8 / search-ledger W9 / UNDERSTANDING lines that quote "0.2–0.35" overstate.
4. **The network's species belief is 2.3× Smogon's** (0.411 vs 0.180, from the teams; agrees with the orchestrator's
   corrected 0.411 / 0.180). "FN ≈ Smogon" holds for moves only. FN − F is +0.008 [+0.0001, +0.017] at TRUNC-4-K8, a
   BOUNDARY result.
5. **"Items, abilities and spreads add ≈ 0.003" is context-dependent.** They add +0.019 [+0.010, +0.028] once the
   never-seen mons are known: R is a complement of U.
6. **R-GIGO for the Tier-1 worlds ran no leak check on the U class** and never compared provenance records with team
   text. The red team's U-leak check passes (0.180 observed vs 0.178 expected). The F+U / O−U team-text checks pass.
7. **R3 rides on referee noise and on multiplicity.** It holds on truth seeds 28–39, not on 16–27, and its s-side tail
   (1.05 %) exceeds a three-route Bonferroni level (0.83 %). The tie-free LEAF-K8 reads remove both concerns (tails
   ≤ 0.1 %).
8. **The k = 6 frontier cell misses GATE-OPEN by hairs** (99.6 % lower −0.002; per-checkpoint lowers −0.002 / −0.0003;
   n 145 / 131 against the 150 floor). The n shortfall is F's exclusions concentrating late.
9. **R-EVAL's "not evaluator-bound" is a near miss toward evaluator-bound** (+0.0075 [−0.0008, +0.0155]).
10. **The author's CONDITION read classifies `setup_sweep_v2` as neither bot nor sentinel** (`unblind_t1.condition_robust`:
    `"setup_sweep_v2".split("_")[0]` is `setup`). That drops 72 states from its bot condition. It is descriptive; no
    verdict changes.
11. **Not verified here:**
    - the samplers' corners §13.2 names (Transform, Baton Pass, a choice lock across a re-drawn slot);
    - the replica's code (not read: it must stay a blind implementation);
    - H-REFEREE;
    - the slot-order component of U;
    - anything at plateau.

    No playout, no GPU. CPU work used rule 5b throughout: `nice -n 19`, `taskset -c 5-7,13-15`, `mem_cap`, one job at a
    time, peak 1.9 GB.

## Reproduce

```bash
PYTHONPATH=<worktree>/src nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 10 \
  /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 \
  designs/research_state/measurements/hidden_info_tier1_2026-10-13/red_team.py      # → red_team.json (~2 min)
```

`PYTHONPATH` is needed only for section F's honest-draw re-simulation, which uses `main.value_search.world_draws.Prior`
through the `agents.gen3_data` facade.
