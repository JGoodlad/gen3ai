# S-U — search headroom on unfamiliar teams: does one-ply search on P\* gain MORE on U than on home72?

**Status: REGISTERED (2026-10-27): [`PREREG.md`](../research_state/measurements/su_headroom_2026-10-27/PREREG.md), pin
`bf5f1a8e7`; stage 0 (cost only) next, the full read only on the orchestrator's go. The live searched player, the
per-game lane, G-PREFIX, the stage-0 harness and the rule `search_headroom` are BUILT (B1–B4, §6).** The PREREG
re-quotes §4's cost from the build's smoke and changes no class, bar or size. It is the read that H-MID's FINDING 1 names
([`h_mid_value_test_2026-10-26/README.md`](../research_state/measurements/h_mid_value_test_2026-10-26/README.md),
`f200258f`). Search stays PAUSED ([`search_ledger.md`](search_ledger.md)) until a read like this one says otherwise. This
unit read committed artefacts only; it ran one arithmetic script (the power table, §4.3) and no game.

**Abbreviations.**
- P\* = `rb_x26_s2601` @280,000,170, the plateau policy. Kakuna = Metamon's Kakuna ckpt 34, greedy.
- home72 = 72 pool teams. U = H (`heldout72_hpfix`), F (`forum72`) and P72 at equal panel weight (X51's definition).
- S = L1's anchored one-ply search: 8 honest F worlds (`dice="seed"`), the TRUNC-4-K8 leaf (8 playouts of 4 turns, then
  the critic V, P\* playing both sides), piKL at the fitted λ = 0.015 (argmax Q + λ·log π). It is R0 §4's search,
  unchanged ([`R0_REGISTRATION.md`](../research_state/measurements/x35_breakers_stage_2026-10-17/R0_REGISTRATION.md) §4).
- G_set = WR(P\* + S) − WR(P\*) on a set, paired by deal. **I = G_U − G_home72**, the interaction.
- d = the discordance: the share of paired deals whose two outcomes differ.
- pp = percentage points; LB / UB = the lower / upper bound of a 95 % interval; IMP = the per-decision gain of
  `search_ledger.md` §6.2; GUESS = an unmeasured effect; ESTIMATE = a cost from measured rates; MY JUDGEMENT = an
  inference no read states.

---

## 0. One screen

1. **The question.** P\* converts level mid-game positions about 11 pp worse on U than on home72 against Kakuna, and
   its critic sees only about a third of that (H-MID: Δbias +0.076 [+0.016, +0.139]). V's error is a LEVEL shift
   between the sets. Its ordering of games is intact (AUC 0.667 on U, 0.663 on home72). One-ply search compares the
   actions of ONE node, so a level shift common to all successors cancels. Does search then recover part of the U
   conversion gap?
2. **The read.** P\* against P\* + S, against Kakuna on home72 and on U, on ONE engine and interleaved. Every deal is
   played twice, raw then searched. **S acts only from turn 10**, where the forensics put the gap. Before that both arms
   are the same deterministic play, so each pair shares its first nine turns exactly (§2).
3. **The estimands.** I, which is the H-MID question, and G_U alone, which is goal (b). G_U is LEAK-INCLUSIVE: the
   honest builder replays the true seed (`search_ledger.md` §6.1). So only G_U's negative class can be credited (§3).
4. **It is a new question.** No search read so far had P\* PILOTING an unfamiliar team (§1.1). E1-0 tests the
   decision-level instrument. L1 at P\* is a decision-level null on P\*'s own pool-team states, under the self-referee.
   S-U is a GAME-level read against a real non-self opponent. It is E4 of
   [`design_search_next.md`](design_search_next.md), re-cut around H-MID's interaction and carrying its review's
   fixes; if S-U registers, E4 is not registered separately.
5. **The cost** (ESTIMATE from stage 0's one finished searched game, §4.2; stage 0 decides). The searched decision's
   cost is the honest world screen, and it climbs with the turn. On the LEAN screen (2026-10-27, bit-identical, the
   default) a searched game costs **median ≈ 5 CPU-min, mean ≈ 12–16**, the mean set by the long games. A raw game
   costs 11.8 s. The recommended size is 1,200 deals per set (§4.4): ≈ 2,400 searched games, **≈ 560 CPU-h
   (480–640)**. Phase 1, which reads G_U finally, is ≈ 1,600 of them (≈ 370 CPU-h). The builder stage 0 played would
   have cost ≈ 2,000–3,000 CPU-h.
6. **Its power** (d = 0.4; §4.3).
   - DOES-NOT-CLOSE-B, i.e. G_U's UB < +4.3 pp: 0.65 if G_U is truly 0.
   - U-GAIN at a true G_U of +4.3 pp: 0.65.
   - U-HEADROOM at a true I of +7 pp: 0.77; at +5 pp, 0.49.
   - NO-U-PREMIUM (UB(I) < +5.5 pp) if I is truly 0: 0.57.

   **It resolves large effects only.** An interaction the size of a third of the gap (≈ 3.5 pp) would need ≈ 5,100
   deals a set.
7. **When it can run.** Not on the local agent cores. Their 2 heavy slots are shared with the N1-P, H-CTX and X59-S
   reads, and ≈ 560 CPU-h would take 3–7 weeks there. A rented CPU box (an owner question, strategy §3.1) makes it a
   1–2-day job for ≈ $15–50 (UNVERIFIED). The live player's build and stage 0 (≈ 4–5 CPU-h for the cost re-size) can
   go locally whenever a slot is free (§5).
8. **Verdict expected** (GUESS): DOES-NOT-CLOSE-B is the likeliest informative class (P ≈ 0.55), and I most likely
   reads UNDECIDED. Its value is to close or open "search on V reopens on U" with a real opponent, at the price of one
   rented-box run.
9. **A gate runs first** (the S-U review's "gate test", 7aafa527, BUILT 2026-10-27): on ≈ 1,600 fresh states of raw P\*
   vs Kakuna games (unit seeds 100–109, 2 hashed decisions a game at turns 10–30), S runs offline exactly as the live
   searcher would, and its DEVIATIONS from π are scored by the self-referee. GO iff the pooled-U sign accuracy's LB >
   0.50; on STOP no live S-U read is played (`../research_state/measurements/su_gate_2026-10-27/`; rule
   `deviation_sign`, `main.value_search.sign_gate`). It read GO (2026-10-28, 0.635 [0.578, 0.687] on pooled U), its
   excess in the true-seed-leak-exposed states (review `52f470c21` M1). **The leak-reduced re-gate SUPERSEDES it and
   reads GO: 0.599 [0.543, 0.649]** (S′ = `r0s4_rolls@10`, `dice="rolls"`; home72 0.551; leak-clean 0.564 [0.478,
   0.639]; `../research_state/measurements/su_regate_rolls_2026-10-28/`). Next, the orchestrator's call: a
   deviation-only live read against Kakuna (review M3) or V-RANK. An amendment that cites the re-gate registers S′ and
   its engine (`e00a212a…/3d0d5c9c…`, not this PREREG's `bdc5d2be…`).

---

## 1. The question, and how it differs from the reads already done

### 1.1 What H-MID adds, and what it does not

H-MID read P\*'s critic V on the X51 games, from level positions at turn 10:

| | home72 | U | home72 − U |
|---|---|---|---|
| win rate | 0.609 | 0.501 | +0.108 [+0.046, +0.173] |
| mean V | 0.557 | 0.525 | +0.033 [+0.010, +0.059] |
| AUC of V for the outcome | 0.663 | 0.667 | −0.004 [−0.090, +0.078] |

At equal V, P\* wins 8.6 pp [+2.3, +15.1] less on U.

Its FINDING 1 says the lever map is contested.
- **The case for search.** A one-ply search compares successors within a node, so a between-set level error in V does
  not by itself rule out headroom on U. V's within-set ordering is intact.
- **The case against.** H-MID's reading 2 places the deficit in a per-team skill that applies across the whole game
  (H-CTX), not in specific positions that V misreads.

The direct test is a search read on U, and that is this doc.

**What H-MID did NOT measure.** AUC is a BETWEEN-GAME ordering: which U games are going better. Search needs a
WITHIN-NODE ordering: which ACTION is better at one state, after the 4-turn playout that P\* pilots for both sides. No
read has measured V's action ranking on U. The self-check returns to this (§8, A1).

### 1.2 Against E1-0, L1 at P\* and E4

| read | what it asks | states / games | referee | bears on S-U how |
|---|---|---|---|---|
| **E1-0** (UNDECIDED; `../research_state/measurements/search_e1_0_2026-10-26/`) | Does the decision-level self-referee misread gains against a non-self opponent? | 160 bot-stratum states, X26 225M, pool teams | the self-referee against the in-core bot | **A different question:** it checks the instrument, and S-U uses none (the game result against Kakuna is the measure). Its direction, gains shrinking against the real opponent, is a PRIOR for S-U: expect less than any self-referee read implies |
| **L1 at P\*** (`../research_state/measurements/search_l1_plateau_2026-10-25/`) | Does the piKL cell pay on the plateau bank? | 719 states from P\*'s own games, pool teams | self | +0.0001 [−0.0097, +0.0101] per decision at P\*. A prior for G_home72 ≈ 0. Its interval is ≈ ±29 pp in game units (≈ 30 decisions; `design_search_next.md` §1.6, UNVERIFIED bridge), so it bounds nothing at game scale. It holds no U state |
| **Off-pool bank, stage 1 / 1c, maturity** | the worlds, the judge | P\* on POOL teams; the off-pool bank's opponents on `heldout72_hpfix` ("the traced side on a pool team", its G-OFF guard) | self | Where any team was unfamiliar, it was the opponent's (facing), never P\*'s own (piloting). X37a puts the deficit in piloting |
| **E4** (`design_search_next.md` §3; review F11) | Does one-ply search pay in games against Kakuna? | U and home72 vs Kakuna, paired with raw P\* | the game | **The same player and opponent.** S-U re-cuts it: (a) the H-MID interaction as a co-primary; (b) the review's F11 fixes (sized against the bars, a DOES-NOT-CLOSE class, PAYS read as leak-inclusive, σ̂ = π only); (c) search from turn 10 only, for a shared prefix. E4's gate (an E1 / E2 arm predicting ≥ +4.6 pp) has not opened. S-U enters on H-MID's mechanism, not on that gate (§8, A6) |

"L1 gains vanish at P\*" and S-U therefore ask different things.
- L1 asks whether search improves P\* where P\* is at its best, on its own teams, scored against itself.
- S-U asks whether search helps where P\* is measurably worse at piloting, scored by a real opponent.

---

## 2. The read

**The arms.** Two labels, one checkpoint:
- **P\*** (raw, greedy);
- **P\* + S.** S runs only at our side's decisions from turn 10 onwards that have ≥ 2 legal actions. Every other
  decision is P\*'s greedy choice.

The opponent is Kakuna ckpt 34, greedy, the X51 regime. Each searched decision is seeded from (deal, turn, slot), so a
searched game replays exactly.

**The sets.** home72, H, F and P72: X51's four panels.
- **Pooled U is primary** for G_U and I, because it is goal (b)'s meter and H-MID's population.
- F, H and P72 are each REPORTED.
- F alone is the X60 / X62 panel. Its interaction is a secondary, not a verdict.

**Pairing and interleaving.** One unit is one (unit seed, panel) cell of X51's shape: 20 role-balanced deals, played
raw then searched, deal by deal, in ONE job. So the arms are interleaved at the deal, on one engine, in one process
lifetime. Unit seeds are the outer loop, and the four panels share each seed. A stopped read therefore stays balanced
by seed and set.

**The shared prefix (a GIGO tooth).** Before turn 10 both arms are P\* greedy against Kakuna greedy, on the same seed.
- X51 measured Kakuna's argmax rate at 1.0000.
- The H-MID and `prior_mix_ab` guards reproduced banked games exactly.
- **G-PREFIX:** every pair's protocol through turn 9 must be byte-identical. Any mismatch holds the lane for a root
  cause. If the mismatch is not resolved, the read is VOID. It means a nondeterminism that would corrupt the pairing.

A bonus follows from the prefix. The turn-10 state is a PRE-treatment covariate, so the level-at-10 subgroup (H-MID's
population) is a valid conditional estimand. It is not a post-hoc cut.

**The engine.** The read plays on ONE engine, named at registration (`../tools/reads.md` §4, "The engine"). The raw arm
is re-played, not imported: X51's banked rows are on E0. Its fresh P\* vs Kakuna numbers on that engine are a
by-product, compared with X51 game for game as a REPORTED cross-engine guard, never pooled.

**What it does not do.**
- No search on Kakuna's side.
- No opponent model other than π inside the leaf (σ̂ = π, as R0 §4; Kakuna cannot run in the playout core,
  `design_search_next.md` D-7).
- No training.
- No public server.

The only play is on our own `ws_frontend` on a 9XXX port, against the locally run Metamon peer.

---

## 3. Estimands and the registered rules (rule `search_headroom`, BUILT; not registered)

Per set, the unit statistic is the mean paired difference (searched − raw) over the unit's 20 deals, with a tie as ½.
- G_home72 is the mean over home72 units.
- G_U is the equal-weight mean of the three U panels' means.
- I = G_U − G_home72.
- Intervals: a percentile bootstrap over units, resampled within each panel, B = 2,000, seed 20261027 (H-MID's and the
  forensics' scheme).
- Bars are compared as exact fractions. A bound within 0.0005 of a bar takes UNDECIDED (root rule 8).

**Primary A, the H-MID question (I).**

| class | rule | meaning |
|---|---|---|
| **U-HEADROOM** | LB(I) > 0 and LB(G_U) > 0 | search gains more on U, and gains there. V's within-node use survives its level error |
| **U-LESS** | UB(I) < 0 | search gains LESS on U. The diffuse per-team deficit (H-CTX) leaves one ply less to fix there |
| **NO-U-PREMIUM** | UB(I) < +5.5 pp | search recovers less than half of the U premium (11.1 pp, X51) beyond what it does on home72. It may co-occur with U-LESS |
| UNDECIDED | otherwise | |

**Primary B, goal (b) (G_U).** Every positive class here is LEAK-INCLUSIVE.

| class | rule | meaning |
|---|---|---|
| **DOES-NOT-CLOSE-B** | UB(G_U) < +4.3 pp | even with the builder's leaks, search adds less than goal (b)'s gap (X51: 0.487 vs ≈ 0.53). **This class is creditable**, because the leaks only push G_U up |
| **U-GAIN (leaky)** | LB(G_U) > 0 | a gain that is not credited to honest search until the fresh-dice replay exists (`search_ledger.md` §6.1, Tier B) |
| **GOAL-B-REACHED (leaky)** | the searched arm's pooled U score has LB > 0.50 (`main.x35.d1.newcombe_strat`'s stratified score) | reported, never claimed as goal (b) met |
| UNDECIDED | otherwise | |

**Why I is the leak-robust estimand.** The two residual leaks act the same way on both sets: speed ties on the reused
seed (20.1 % of decisions) and the truth's hidden counters. To the extent their value does not depend on the set, they
cancel in I. That is plausible but UNVERIFIED (FINDING S-4).

**REPORTED, never decided.**
- G_home72 alone, and G per panel.
- I on F alone.
- G and I in the level-at-10 subgroup (valid, §2).
- The deviation rate (the share of searched decisions where S ≠ π's argmax), by set.
- Searched CPU-s per decision and per game.
- The world quality by set, from the true log in process at zero playouts: the exact never-seen species mass and the
  share of the truth's binding facts held. This is the covariate for confound A3.
- The raw arm's score per panel on this engine, against X51's.
- H-MID's Δbias re-read on the raw arm's captures, a free replication on a new engine.

**VOID.**
- Counted units on two engines, or an unknown engine.
- An unresolved G-PREFIX mismatch.
- A raw arm on another checkpoint (sha256).
- Any search record whose config differs from the registered one.

---

## 4. Cost and power

### 4.1 Per decision (measured elsewhere; none of it on Kakuna games)

| figure | value | source |
|---|---|---|
| an honest F world build | ≈ 14 CPU-s | maturity README, "Costs" (MEASURED, one thread, contended) |
| one 8-world arm played | ≈ 15 CPU-s | same |
| an honest decision at 4 worlds | ≈ 28 s | `search_ledger.md` §6.1 |
| an honest decision at 8 worlds | 42–96 s | R0 §4 |
| a full playout | ≈ 0.3 CPU-s | `design_search_next.md` E2's cost (the pilot's LEAF-K8) |

**The planned budget, one decision ≈ 29–60 CPU-s.** It builds 8 worlds, then plays each legal action 8 times 4 turns
deep with the critic at the cut. A live decision rebuilds its worlds each time; no amortisation is designed.

### 4.2 Per game

**Decisions a game.** P\* made 79,799 choices in X51's 1,600 Kakuna games, ≈ 50 a game, forced switches included
(H-MID's reconstruction table). Median length is 32–38 turns (X51). Turns ≥ 10 hold ≈ 74 % of a 35-turn game, so
≈ 37 choices fall from turn 10 on. Of those, an estimated 30–37 have ≥ 2 legal actions; the ≥ 2-legal share is
unmeasured.

**The cost a searched game** (ESTIMATE, corrected by stage 0's first searched game, 2026-10-27; S-1):
- **The cost a decision climbs with the turn, and it is the honest world screen (MEASURED 2026-10-27,
  [`AMENDMENT_1_DRAFT.md`](../research_state/measurements/su_headroom_2026-10-27/AMENDMENT_1_DRAFT.md) §1b).** On
  the builder stage 0 played, a decision went from ≈ 6 search CPU-s at turns 10–19 to ≈ 130 at turns 50–60, with worlds, depth, tries and playouts flat or falling. Re-played offline, turn 59 is
  240.5 CPU-s of world building and 2.3 of playouts. Every screening replay is the candidate world's battle from
  turn 1, and both the replays a decision and their length climb.
- **The LEAN screen** (`worlds.WorldSampler(lean_screen)`, the default; the same worlds, draws and picks, bit for
  bit) stops each replay at its first line that differs from the truth's. That game's 57 searched or excluded
  decisions cost 77.8 search CPU-min on the old builder and **18.0** on the lean one (4.3×; its 9 exclusions 6.4×, turn 59 243 → 54 s).
- **Per game** (that game's per-turn cost over X51's 2,800 game lengths, ESTIMATE): median ≈ 5 CPU-min of whole-unit
  CPU, mean ≈ 12–16, so ≈ 560 CPU-h (480–640) for the full read. The old builder's mean would have been ≈ 50–74
  CPU-min (≈ 2,000–3,000 CPU-h), so the design's central ≈ 25 was low: the long games set the mean, and they are dear.
- The raw twin costs 11.8 s (X51, measured).
- Kakuna's forward is not counted. It waits while we search.

**Wall time:**
- At nice 19 on the agent cores, a game runs ≈ 1–1.5× its CPU time.
- Under SCHED_IDLE it ran 2.5–3× (stage 1c's log). A searched game is therefore ≈ 25–75 min of wall time.
- That is too long for the anchor lane's 600-s short lane and for 20-game units in a heavy slot (FINDING S-2, build
  B3).

### 4.3 Power (two-sided 95 %, normal approximation; d from X51's twins)

**The discordance.** X51's P\* vs X26 @161M twins on the same deals had discordance 0.36–0.41. Those are two different
policies from turn 1. S-U's pairs share nine turns, and a game with no deviation is concordant. So d ≈ 0.3–0.4, with
0.5 as the worst case, where outcomes are independent once the pair diverges (review F11; N4).

**The model.** SE(G) = √(d / n) per set, and SE(I) = √(SE_U² + SE_H²).

| n_U / n_home72 | d | SE G_U | SE I | P(DNC-B) at G_U = 0 | P(U-GAIN) at G_U = +4.3 | P(U-HEADROOM) at I = +5 / +7 | P(NO-U-PREMIUM) at I = 0 | searched games | CPU-h at 15 / 25 / 32 min |
|---|---|---|---|---|---|---|---|---|---|
| 400 / 400 (X51's size) | 0.4 | 3.16 | 4.47 | 0.27 | 0.27 | 0.20 / 0.35 | 0.23 | 800 | 200 / 333 / 427 |
| 1,200 / 400 | 0.4 | 1.83 | 3.65 | 0.65 | 0.65 | 0.28 / 0.48 | 0.33 | 1,600 | 400 / 667 / 853 |
| 1,200 / 800 | 0.4 | 1.83 | 2.89 | 0.65 | 0.65 | 0.41 / 0.68 | 0.48 | 2,000 | 500 / 833 / 1,067 |
| **1,200 / 1,200** | 0.3 | 1.58 | 2.24 | 0.78 | 0.78 | 0.61 / 0.88 | 0.69 | 2,400 | 600 / 1,000 / 1,280 |
| **1,200 / 1,200** | **0.4** | **1.83** | **2.58** | **0.65** | **0.65** | **0.49 / 0.77** | **0.57** | **2,400** | **600 / 1,000 / 1,280** |
| 1,200 / 1,200 | 0.5 | 2.04 | 2.89 | 0.56 | 0.56 | 0.41 / 0.68 | 0.48 | 2,400 | 600 / 1,000 / 1,280 |
| 1,800 / 1,800 | 0.4 | 1.49 | 2.11 | 0.82 | 0.82 | 0.66 / 0.91 | 0.74 | 3,600 | 900 / 1,500 / 1,920 |

- DOES-NOT-CLOSE-B at 80 % power with G_U truly 0 needs n_U ≈ 4,240·d: 1,273 / 1,698 / 2,122 at d = 0.3 / 0.4 / 0.5.
- **An interaction the size of a third of the gap (≈ 3.5 pp), at 80 %, needs ≈ 5,100 deals a set** (≈ 10,200 searched
  games). That is out of reach: S-U sees only large interactions.
- The table is arithmetic from the model above (z = 1.96, 80 % power at z = 0.84), computed in under a second on
  the agent cores. No script is committed.

### 4.4 The size, the looks and the re-size rule

- **Final: 1,200 deals per set.** For U that is unit seeds 0–19 on each of H, F and P72 (400 a panel, X51's). For
  home72 it is unit seeds 0–59. Balanced n minimises SE(I) for a fixed number of searched games, because both sets
  cost the same per game.
- **Stage 0** is the first 5 deals of unit seed 0 on each panel: 20 searched games, ≈ 4–5 CPU-h. It reads COST ONLY:
  CPU per searched game, plus the G-PREFIX and peer-stability checks. Its games are the read's first games, not extra.
- **Phase 1** is unit seeds 0–19 on all four panels: U complete at 1,200, home72 at 400 (the 1,200 / 400 row, ≈ 1,600
  searched games, ≈ 370 CPU-h central on the lean screen). **G_U's classes are READ here, and are final.** I is read futility-only: the
  read STOPS iff NO-U-PREMIUM or U-LESS is already reached. No positive class of I is read at phase 1.
- **Phase 2** is home72 unit seeds 20–59 (+800 searched games, ≈ 333 CPU-h), played only when I is UNDECIDED at phase 1.
  I's classes are then read on all 1,200 / 1,200. Every phase-2 deal is still raw-then-searched in one job on the same
  engine (the pairing is within the deal), so the cross-set interleaving of phase 1 is not needed for validity.
- **Two looks on I.** A negative class read at either look has a false rate ≤ 2 × 2.5 % (MY JUDGEMENT, the Bonferroni
  bound), acceptable for a futility stop. The positive class is read once.
- **The re-size rule** is fixed now, and cost is not an outcome. If stage 0's median searched game costs > 2× the
  central estimate (> 50 CPU-min), phase 2 is dropped and I becomes REPORTED. Nothing else re-sizes.
- **Discordance.** The pooled b + c is reported after stage 0 and again before phase 2's decision, together with the
  achieved power. The sign, b − c, is not shown. It never changes n.

---

## 5. When it can run

**Locally, not now.** The 2 heavy slots on rule 5b's agent cores are shared with:
- N1-P look 1 (≈ 60 h left at its start);
- the H-CTX read (35–70 lane-h);
- X59-S's reads;
- D3 look 2's read;
- PR-1 (165–238 lane-h) and X62's read when they register.

At ≈ 560 CPU-h on about one effective lane, S-U would take **≈ 3–7 weeks** of local wall time, behind those reads.
So:
- **The build (B1–B4, §6) is CPU-light** and can land any time.
- **Stage 0 (≈ 4–5 CPU-h)** runs locally in one free heavy slot. It turns this doc's cost ESTIMATE into a measurement
  before anyone rents anything.
- **The read itself does not run locally** unless the queue empties.

**A rented CPU box changes it.** The approval is the one already asked for PR-1 (strategy §3.1). The box would play
both arms, because the engine rule does not let a cross-host import stand in for either.
- A 32-vCPU box is likely memory-bound: a P\* + search process plus a Kakuna peer is ≈ 4–6 GB (X51's unit peaked at
  3.7 GB; stage 1c's 128-world chunk at 6.3 GB). It might hold 12–24 games at once.
- That gives ≈ 24–48 box-h, **≈ 1–2 days, ≈ $15–50** at the strategy doc's ≈ $0.6–1 a box-h (UNVERIFIED: no rented
  CPU has played a Kakuna game or a search).
- A few units re-played locally game for game are the cross-host guard, REPORTED.
- **Order on the box** (MY JUDGEMENT): after PR-1. PR-1 is ranked by P(GAIN) per resource-day, and S-U's expected class
  is a negative one.

---

## 6. Builds (PROPOSED, each default OFF)

| # | build | status (2026-10-27) |
|---|---|---|
| **B1** | **The live one-ply search player** in the anchors core slot | **BUILT**: `main.anchors --our-search r0s4@10` (`src/main/anchors/search_side.py`, presets in `search_presets.py`). Each eligible decision runs `main.value_search.search_decision` on the battle's TRUE input log, read in process from the front end (`ws_frontend._Battle.choice_log`) and cut honestly: every command of the earlier request updates, ordered (update, p1 before p2), never the opponent's choice of OUR update (so its root decision is open in every world and π answers it), and `root_choice_in_log=False`. Two guards before any world: the log's replay opens our decision with the live tokens, turn and decision index, and its replayed observation row is the live core frame's, byte for byte. The worlds are `WorldSampler(reveal="off", max_tries=60)`, `dice="seed"` (the maturity battery's F builder). Each search is keyed by the DEAL (a digest of the battle's `START`: both teams, names, seed) and the decision (turn, our decision index), so a searched game replays exactly. It runs in a worker thread (§8 A9). A `WorldError` (no consistent world) is an EXCLUSION, played as π's choice and recorded; any other exception fails the game. One record per decision in `<out>/search_decisions.jsonl`: searched or why not, π's argmax, the pick, Q̂ and log π per legal action, the root V, the playouts, CPU-s (the worker thread plus the replay children) and wall-s, the worlds' tries, leak block and species hit rate against the truth (the world-quality covariate, at species level). Tests: `search_side_test.py`, `search_side_integration_test.py` (a live in-process battle: every decision's root replays byte for byte; the search on that live log is deterministic and blind to the order of an update's two choices) |
| B2 | **piKL as an acting rule** | **BUILT**: `SearchConfig(rule="pikl", pikl_lambda=0.015)` (`main.value_search.rules.pikl_scores`, L1's arithmetic). `rules_pikl_test.py` re-scores L1's banked rows with it: E2's and X26's held-out TRUNC-4-K8 IMP and deviation share and the Tier-1 fit point reproduce `result_l1.json` / `fit_l1.json` to their rounding |
| B3 | **A searched label in `main.reads`** | **BUILT**: `models.<label>.load.search` (stamped `our_regime = greedy+search=r0s4@10`) in an `anchor` group with `per_game: true`: each seed's 20-game series is 20 ONE-GAME units (`main.anchors --games 20 --deal-game G`: game G's role, both sides' team draws, its seed), queued DEAL by deal (raw before searched), so a stopped read resumes at game granularity. A searched unit is a long heavy-slot job with its watchdog at 3 h. **G-PREFIX** (`main.reads.prefix`): each pair's protocol through `\|turn\|10` must be byte-identical; a mismatch HOLDS the lane and keeps the group inconclusive. `run --max-units N` and `search-cost` (`../tools/reads.md` §3). The design's G-REPRO (a searched label with the switch OFF reproduces raw P\* game for game) is G-PREFIX's own condition: before turn 10 the searched arm IS the raw arm |
| B4 | **The rule `search_headroom`** (§3), with G-PREFIX as its VOID gate, and a test | **BUILT**: `src/main/reads/rules_search_headroom.py` (`rules_search_headroom_test.py`). ONE `per_game` group; a deal is a pair only when both games counted; the bootstrap draws the U panels before home72 from one `default_rng(20261027)`, so phase 2's extra home72 units leave G_U's interval as phase 1 read it; VOID on G-PREFIX, engines, bytes, the searched units' stamp and played preset, or a raw unit's search stamp (§3). Not in the rule (the cells do not carry them): the level-at-10 subgroup, the world quality, the raw arm against X51, H-MID's Δbias (S-12) |

The PREREG pins the commit that lands B1–B4 (`../tools/reads.md` §1's flow). Stage 0 plays at that pin: `python -m
main.reads run --state DIR --lane anchor --hours H --max-units 40` (with the four panels in ONE `per_game` group, the
queue's first 40 units are the first 5 deals of unit seed 0 on each panel, raw and searched), then `python -m main.reads
search-cost --state DIR`.

---

## 7. Predictions (GUESSES, to be registered)

- **The deviation rate** is 8–15 % of searched decisions on both sets, a few points higher on U. The policy is less
  sure there: the N2 probe's policy shift is 1.19× on unfamiliar sets.
- **G_home72 = +0.5 pp (−3, +4).** L1 at P\* reads ≈ 0, and E1-0's direction shrinks gains against a real opponent.
- **G_U = +1.0 pp (−3, +5).**
- **I = +0.5 pp (−4, +5).** H-MID's ordering argument pushes it up. H-CTX's diffuse per-team skill and the leaf's
  opponent model (A2) push it down.
- **The classes:** DOES-NOT-CLOSE-B 0.55; U-GAIN (leaky) 0.15; U-HEADROOM 0.08; NO-U-PREMIUM 0.45; U-LESS 0.07;
  a phase-1 stop on I ≈ 0.3.
- **The cost:** a searched game at 20–30 CPU-min.

---

## 8. Adversarial self-check

**A1. "V orders positions on U" is the wrong ordering.**
- **The objection.** H-MID's AUC ranks GAMES by outcome; search ranks ACTIONS within a node. Equal AUC does not imply
  equal action ranking, and the action ranking on U is unmeasured.
- **The leaf compounds it.** The TRUNC-4 leaf is 4 turns of P\* on both sides and then V. So Q̂ carries P\*'s U piloting
  for 4 turns, which is the right continuation for one-ply improvement. But it also carries P\* as the model of Kakuna on
  Kakuna's U team.
- **Answer.** That is why the read is a game read and not a re-analysis of H-MID. The hypothesis is the brief's, held
  with low confidence (§7). The cheap mechanism read (the within-node Spearman of Q̂ against a true-world referee on U vs
  home72 states) is decision-level, and it needs a self-referee here, because Kakuna cannot run in the core. E1-0 left
  that instrument UNDECIDED. It was REJECTED as a gate (§10 below).
- **Update (2026-10-27): the referee objection is removed.** [`design_v_rank_on_u.md`](design_v_rank_on_u.md) (V-RANK,
  DRAFT) makes Kakuna the referee without the core: deterministic replay to the position, then one real game per legal
  action. It is a FILTER for this read (its §4.3): a null or worse within-node slope on U turns S-U OFF; an intact one
  makes it eligible, never predicted to pay.

**A2. The leaf's opponent model differs by set.**
- **The objection.** On U the playouts model Kakuna as P\* piloting an unfamiliar team, which P\* does ≈ 11 pp worse
  than Kakuna does. The leaf then expects the opponent to blunder. Search would over-value lines that need the opponent
  to mis-pilot. That is a set-dependent S-factor error (`design_search_next.md` §1.4) pushing G_U down, so it is a
  confound for I.
- **Answer.** It cannot be removed without Kakuna in the core. It is named, and it biases I toward U-LESS. It makes a
  U-HEADROOM, if found, conservative.

**A3. The worlds differ by set.**
- **The objection.** The F worlds draw Kakuna's team from Smogon usage. How close they come to the truth differs between
  a pool team (home72), forum teams (F), ladder teams (H) and procedural teams (P72). So I mixes "V / π headroom on U"
  with "the honest worlds fit U teams better or worse".
- **Answer.** The REPORTED world-quality covariate (§3) measures this per decision at zero playouts, and its set
  difference is reported beside I. The oracle arm that would remove it was rejected (§10).

**A4. The leaks may not cancel in I.** The speed-tie leak's value depends on which species' speed ties occur, and the
species mix differs by set. FINDING S-4. The interaction is leak-ROBUST only to first order. A U-HEADROOM still has to
survive the fresh-dice replay before anything is built on it.

**A5. Power.**
- **The objection.** At the recommended size the read sees I ≥ ≈ 7 pp with 77 % power. Its own prediction (+0.5)
  makes UNDECIDED the likeliest outcome for I.
- **Answer.** This is stated up front (§0.6, §7). The read is worth its cost because G_U's negative class is reachable,
  at 0.65, and because no game-level search read against a non-self opponent exists at all. Buying I at a third-of-gap
  resolution would cost ≈ 4× and is not proposed. If the owner prefers to spend only on the creditable class,
  phase 1 alone (the 1,200 / 400 row) reads DNC-B equally well for ⅔ of the CPU.

**A6. It bypasses E4's gate.**
- **The objection.** `design_search_next.md` gates E4 on an E1 / E2 arm predicting ≥ +4.6 pp, and none has.
- **Answer.** S-U's entry is H-MID's mechanism, a new prior that E4's gate did not have. Its review (F11) said a
  NEGATIVE is the informative class of a live read with leaks. S-U is built around exactly that class. This is MY
  JUDGEMENT; the orchestrator rules whether the gate applies.

**A7. Search from turn 10 changes the question.**
- **The objection.** Searching only from turn 10 leaves turns 1–9 as P\*'s. If search would help in the opening, S-U
  misses it.
- **Answer.** The forensics put none of the U gap in the state at turn 10 (−0.4 pp [−2.6, +1.3]). The prefix buys the
  pre-treatment covariate and ≈ 25 % of the cost. G is "search from the mid game", and the doc says so.

**A8. Determinism of the prefix.**
- **The objection.** If Kakuna or our side is not bit-deterministic across two plays in one process (thread counts, a
  peer's internal state), G-PREFIX fails and the pairing is lost.
- **Answer.** The evidence is X51's argmax rate 1.0000 and the game-for-game guards (H-MID 1,600 / 1,600; prior-mix
  80 / 80). Stage 0 tests it on 20 pairs before any outcome.

**A9. The peer's patience.** Kakuna waits while we search. Read and tested at the build (S-3, CLOSED): our front end
has no battle timer and the Metamon peer waits on its observation queue with no timeout; its ONE clock is poke-env's
websocket keepalive (a ping every 20 s, the connection closed when a pong is 20 s late). The search therefore runs off
the event loop: a 70-s off-loop decision kept Kakuna connected (`src/main/anchors/peer_patience_integration_test.py`),
and the negative control, 70 s ON the loop, dropped the peer and the front end booked a forfeit WIN for us. Such a
game is now VOID in every anchor lane, never an outcome (the forfeit-win guard, `EXTERNAL_ANCHORS_SOP.md` § Counting a
unit; that control is its test). Our own watchdog is sized for a searched game (`main.x35.kakuna.UNIT_SEARCH_S`). A timer firing is a defect of the harness,
never an outcome (root: "a timeout is never a semantic outcome").

**A10. Pooled U carries P72.** P72 is weak and procedural. X51 FINDING 2: it sits below 0.50 and did not move at the
plateau. A G_U carried by P72 alone would be a weak-team effect. The per-panel G is REPORTED, and F's I is a
secondary.

**A11. The scope rule.**
- **The objection.** The owner's one-ply rule caps IN-LOOP search.
- **Answer.** A live, measurement-only player on our own server against a local bot follows the owner's 2026-10-20
  "exploiter + pi search, just to know" (R0 §4; review F13). It never plays in training or on a public server. D-10's
  owner question (`design_search_next.md` §6) covers this read too.

---

## 9. FINDINGS (standing rule 7)

- **S-1.** Every cost in §4–§5 is an ESTIMATE resting on ONE searched game. The build's smoke (12 decisions at turns
  10–19, `~/gen3ai_archive/su_build_smoke_2026-10-27/`) measured 6.4 CPU-s a decision and did not hold past turn 20.
  Stage 0's first searched game (home72 seed 0 game 0, 66 turns, a loss) cost 77.8 search CPU-min on the builder it
  played (78.3 whole-unit, 83 min of wall), 36.5 of them in 9 exclusions. **MEASURED (2026-10-27): the climb is the
  honest world screen** (turn 59: 240.5 of 243.0 CPU-s), and the LEAN screen re-makes all 57 of its decisions
  identically for 18.0 CPU-min
  (`AMENDMENT_1_DRAFT.md` §1b). The per-game figures spread that one game's per-turn cost over X51's game lengths;
  its exclusion rate, which sets the late cost, is unmeasured anywhere else. Stage 0's 20 games measure it before the
  read sizes. It is HELD behind N1-P, H-CTX and X59-S; the cost variants are `AMENDMENT_1_DRAFT.md` (DRAFT, not
  registered).
- **S-2.** CLOSED by B3: one-game units (`per_game`), a searched unit a long heavy-slot job with a 3-h watchdog.
  Residual: a searched unit never takes the short lane, so it queues behind the slots' 40–45-min chunks; if stage 0
  confirms ≈ 5 min of wall a searched game, the short lane with its exit-88 long fallback would fit (a one-line change
  in `main.x35.kakuna`, not made). Stage 0's first searched game ran > 1 h of wall on the old builder: the short lane
  did NOT fit, and a searched long job took a heavy slot ahead of the short units of reads ranked above S-U. On the
  lean screen that game is 18 search CPU-min and a median game ≈ 5 (ESTIMATE): the short lane would fit most games, a
  long one still not. A game that runs past 45 min holds its slot past the chunk policy: it cannot be split.
- **S-3.** CLOSED (A9): the keepalive is the one clock, and the search runs off the event loop.
- **S-4.** The leak-cancellation argument for I is first-order and UNVERIFIED (A4).
- **S-5.** The X51 twins' discordance (0.36–0.41) is between two DIFFERENT policies from turn 1. S-U's d is
  extrapolated from it.
- **S-6.** The ≥ 2-legal share of decisions from turn 10: 12 / 12 in the smoke (too few to size on; stage 0 reports it).
- **S-7.** The rented box's memory per search game, its parallelism and its price are UNVERIFIED. The smoke's searched
  unit peaked at 2.4 GB.
- **S-8.** This doc ran no game and no search before the build; the build ran the smoke above, nothing else.
- **S-9.** One deal is one game of a seed's 20-game series played alone (`main.anchors --deal-game`). The smoke's lone
  forum72 game 13 got exactly the seed, both packed teams and the names that game 13 of a banked whole series got (D1's
  `c0_s3505__forum72__00`), so the deal matches X51's. That its PLAY matches game for game is UNVERIFIED: a peer that
  carries state across battles would play a lone game otherwise (the REPORTED cross-engine guard against X51 reads it).
- **S-10.** The searched log is the battle's TRUE input log cut honestly (§6 B1). Its within-update order is canonical
  (p1 before p2) and a peer's switch is re-spelled to the core's token; the root guard (the replayed row equal to the
  live one) and the integration test's order-swap check hold it, and the smoke replayed all 12 roots byte for byte.
- **S-11.** The builds changed `main.reads`' play path (`ws_frontend`, `main.anchors`, `main.x35.kakuna`): a running read
  can no longer `repin` to a commit after them. N1-P, H-CTX and X59-S play their own pinned commits and need no repin.
- **S-12.** CLOSED by B4: `search_headroom` decides §3's classes. Residual: (a) it reads ONE `per_game` group, whose
  team sets share its unit seeds, and a searched group imports nothing, so phase 2 (home72 seeds 20–59 alone) has no
  spec shape yet; the PREREG must pick one (two groups the rule pools, or a phase-2 group read beside phase 1's) before
  phase 2 can be played; (b) the REPORTED level-at-10 subgroup, world quality, raw-vs-X51 guard and Δbias re-read are
  not in the rule: the cells do not carry their data.

## 10. Alternatives considered

| alternative | why not (now) |
|---|---|
| A decision-level read on X51's banked level states: the honest piKL pick against a self-referee, IMP_U − IMP_home72 | n·SE² ≈ 0.0074 gives ≈ ±0.0075 a decision at 1,000 states a set, ≈ ±23 pp a game. Its referee is P\*, not Kakuna, which is E1-0's open problem. It is no cheaper than games at the game bar (review F12) |
| An ORACLE-worlds arm as a ceiling | It removes A3 but adds a huge information gain on both sets (G_O ≈ +0.05 a decision). That saturates the win rate and compresses I. It doubles the read |
| F alone, not pooled U | Goal (b) and H-MID are pooled U. F's I is REPORTED |
| Search on every turn | +≈ 25–35 % CPU. It loses the shared prefix and the pre-treatment subgroup. The gap is not in the opening |
| The asymmetric cell (we pilot U, Kakuna pilots home72), which isolates piloting | Not goal (b)'s meter, and it changes A2's confound rather than removing it. A possible follow-up on a U-HEADROOM |
| Running E4 as written | Sized against 0, not its bar, and no DOES-NOT-PAY class (review F11) |

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-27 | **This doc CREATED (DRAFT, explicit-only)** (search-research design worker, orchestrator brief; docs only, one arithmetic script) | S-U: P\* vs P\* + L1's piKL search (8 honest worlds, TRUNC-4-K8, λ 0.015) from turn 10, against Kakuna on home72 and pooled U, ONE engine, deal-interleaved and paired with a byte-identical nine-turn prefix. The co-primaries are I = G_U − G_home72 (U-HEADROOM / U-LESS / NO-U-PREMIUM) and G_U (DOES-NOT-CLOSE-B creditable; positives leak-inclusive). 1,200 deals a set: a cost-only stage 0; phase 1 (U complete, home72 400) reads G_U finally and I futility-only; phase 2 tops home72 up only if I is UNDECIDED. ≈ 1,000 CPU-h central: a rented-box read, not a local one. It supersedes E4's registration if registered | a decision-level self-referee screen (too coarse, wrong referee); an oracle ceiling arm (saturates, doubles); F alone; search every turn; E4 as written; buying a third-of-gap interaction (≈ 4× the CPU) | H-MID README (`f200258f`) FINDING 1, reading 2–3; the forensics (`c758fcb8`); X51 README (twins, costs); maturity README "Costs"; R0 §4; `search_ledger.md` §6.1, §6.2; `design_search_next.md` E4 and its review (`9738296f`) F11, F12; `../tools/reads.md` §3, §4 (`20875e96`) |
| 2026-10-27 | **B1–B3 BUILT and smoke-tested** (search engineer, orchestrator brief; no read registered or played) | the live searched player (`main.anchors --our-search r0s4@10`), piKL in `main.value_search.rules`, per-game units with G-PREFIX and the stage-0 harness (`run --max-units`, `search-cost`) in `main.reads`; §6's table, A9, §9 (S-1 measured on the smoke, S-2 and S-3 closed, S-9–S-12 new) | searching ON the event loop (the peer's keepalive drops it, and the forfeit reads as our win); a mid-game checkpoint to split a long searched game into chunks (one game per unit fits once the smoke's cost holds); re-sizing §4 on the smoke's 12 decisions (stage 0 does it, by §4.4's rule) | `src/main/anchors/search_side.py`, `search_presets.py`, `peer_patience_integration_test.py`; `src/main/reads/prefix.py`, `search_cost.py`; `~/gen3ai_archive/su_build_smoke_2026-10-27/` |
| 2026-10-27 | **B4 BUILT, the forfeit-win guard landed** (S-U operator, orchestrator brief) | `search_headroom` (§6 B4; ONE `per_game` group, pairs only where both games counted, a percentile unit bootstrap within panel with the U panels drawn first, exact-fraction bars with rule 8's 0.0005, the phase-1 futility-only rule, VOID on G-PREFIX / engines / bytes / the search stamp and played preset); every anchor game a FORCELOSE decided other than our own turn-limit cap is VOID (`runner.forced_end_verdict`, A9) | computing the REPORTED subgroup / world quality / Δbias inside the rule (the cells do not carry them: separate scripts); counting a forced-end game as a loss instead of VOID (a peer's drop is a harness defect, never an outcome) | `src/main/reads/rules_search_headroom.py`, `src/main/anchors/forced_end_guard_test.py`, `peer_patience_integration_test.py` (the negative control); the banked scan: 0 forced-end games in 972 counted units |
| 2026-10-27 | **Stage 0 held; the cost model corrected** (S-U operator; orchestrator ruling (b)) | stage 0 HELD until N1-P, H-CTX and X59-S finish (a searched game held a heavy slot > 1 h); §4.2's per-game cost corrected from stage 0's first searched game (the world screen's cost climbs with the turn); the smoke's ≈ 150–250 CPU-h withdrawn; the cost-capped variants drafted, not registered (`AMENDMENT_1_DRAFT.md`: an incremental screen first, else search turns 10–30) | continuing stage 0 now (a day of a slot above the brief's priority); a per-decision CPU-time budget (not deterministic); 4 worlds (no longer L1's S) | stage 0's `search_decisions.jsonl` for `pstar_s__home72__00__g00` |
| 2026-10-27 | **A filter proposed: V-RANK** (research designer, orchestrator brief; docs only here) | §8 A1's update: the within-node ranking read now has a non-self referee (live Kakuna branches by deterministic replay), `design_v_rank_on_u.md`; its gate can turn S-U OFF (RANK-NULL-ON-U, RANK-WORSE-ON-U) or make it ELIGIBLE (RANK-HOLDS-ON-U). Nothing in this doc's classes, bars or size changes | treating an intact ranking as evidence that S pays (on P\*'s own states the slope is 0.54 while L1 reads +0.0001) | `design_v_rank_on_u.md` §0, §4.3, §5.1 |
| 2026-10-27 | **The cost's cause MEASURED; the LEAN world screen BUILT (bit-identical)** (search-performance engineer, orchestrator brief) | the climb is the world screen (turn 59: 240.5 of 243.0 CPU-s; every screening replay is the candidate world's battle from turn 1); the lean screen, the default: a screening replay stops at its first line that differs from the truth's and returns only what the gate reads, and a proposal the prior rules out is not replayed; the decision log splits world vs playout CPU; §0, §4.2, §4.4, §5, S-1 re-costed (≈ 560 CPU-h, was ≈ 1,000; the old builder ≈ 2,000–3,000) | replaying the TRUE log's prefix once (the draft's E: not bit-identical, a world leaves the truth at the screened mon's first switch-in, every mon by turn 10 here); dropping `core_events`' cross-checks from screens (2.4× more on the replays, but the other side's checks can refuse a candidate the searched side's cannot: not identical by construction; FINDING) | `AMENDMENT_1_DRAFT.md` §1b; `src/main/value_search/world_screen_parity_test.py` (4 banked decisions both ways, an exclusion included, and a wrong reference caught); all 57 of the game's decisions re-made offline equal to their live records |
| 2026-10-27 | **Ruling 6 of the precommit memo accepted** (doc editor; docs only) | the S-U gate's rule is the review's, verbatim: GO iff LB > 0.50 on pooled U sign accuracy, the GIGO tooth first; any change is in the registration, before any gate state is scored | — | `precommit_reads_2026-10-27.md` §4 |
| 2026-10-27 | **The gate test BUILT** (S-U gate operator, orchestrator brief; the review's "gate test") | §0 item 9: raw cells keep honest input logs (`main.anchors --keep-input-logs`), S re-run offline through the live searcher's `_search` at 2 hashed states a game, deviations scored by the 24-seed self-referee; GO / STOP on the pooled-U sign accuracy's LB (rule `deviation_sign`) | V-RANK first (the review's M6: its ON class is uninformative; ≈ 96 CPU-h); the live read first (≈ 370–560 CPU-h for a likely chance-level S) | `../research_state/measurements/su_headroom_2026-10-27/REVIEW.md` B2, M6, § The gate test |
| 2026-10-28 | **The gate READ: GO** (S-U gate operator) | §0 item 9: pooled-U sign accuracy 0.635 [0.578, 0.687], home72 0.537 [0.458, 0.617]; next V-RANK, then the amendment (B1's four items, M3, M4, and the gate's engine) | treating GO as a gain (self-referee) | `../research_state/measurements/su_gate_2026-10-27/README.md` |
| 2026-10-28 | **The leak-reduced RE-GATE READ: GO; it supersedes the gate's GO** (S-U re-gate operator, orchestrator brief; PREREG `3626ef6e5`) | §0 item 9: S′ = `r0s4_rolls@10` (S with `dice="rolls"`, one lever) on the gate's 1,560 states, rule unchanged: 0.599 [0.543, 0.649] on pooled U, home72 0.551; the amendment cites S′ and the gate's engine | citing the gate's leak-exposed 0.635 for the licence; treating GO as a gain (self-referee) | `../research_state/measurements/su_regate_rolls_2026-10-28/README.md` |
