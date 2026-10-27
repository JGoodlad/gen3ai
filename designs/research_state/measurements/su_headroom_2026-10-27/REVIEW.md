# Adversarial review: S-U, search headroom on unfamiliar teams (2026-10-27)

**Scope.** This review attacks the registered read, its draft amendment and the gate proposed for it:
- the design [`design_search_u_headroom.md`](../../../endstate/design_search_u_headroom.md);
- the [`PREREG.md`](PREREG.md) and [`read_spec.json`](read_spec.json) (`ffd9a443`, pin `bf5f1a8e7`);
- [`AMENDMENT_1_DRAFT.md`](AMENDMENT_1_DRAFT.md) (`54b9f29a`, `b1c88594`, and the lean screen E′ of `2eb65b3d9`, still
  NOT registered);
- [`search_ledger.md`](../../../endstate/search_ledger.md) §6.4;
- the stage-0 record (`~/gen3ai_archive/su_headroom_2026-10-27/`: one finished searched game,
  `pstar_s__home72__00__g00`);
- V-RANK, the gate drafted for S-U ([`design_v_rank_on_u.md`](../../../endstate/design_v_rank_on_u.md), `aae39166`).

It reads them against two sources:
- the H-MID review ([`../h_mid_value_test_2026-10-26/REVIEW.md`](../h_mid_value_test_2026-10-26/REVIEW.md), M4,
  `90708bff`);
- the context-generalisation survey ([`../../literature/context_generalisation_2026-10-27.md`](../../literature/context_generalisation_2026-10-27.md),
  `5d554c23`: §6.3, recommendation 4).

The PREREG is not edited.

**Method.** Two checks, both zero-playout and CPU only, in [`scripts/review_checks.py`](scripts/review_checks.py). Every
number below is in [`review_results.json`](review_results.json). The run took 8 s and 1.0 GB in the short lane, at nice
19 on the agent cores.
- **`noise`.** It applies S's acting rule (piKL λ 0.015 on the TRUNC-4 leaf, 8 honest worlds) to the banked rows of
  the maturity battery's PLATEAU bank. It loads them through L1's registered loader: the table and the common ids are
  sha256-checked, giving 719 states per anchor.
  - The states are each policy's own games on POOL teams.
  - The referee T is 24 truth playouts with the anchor on both sides (the SELF-REFEREE).
  - The check reproduces L1's registered cell exactly: IMP at P\* +0.0001, at 25M +0.0193.
- **`cost`.** It spreads stage 0's one game's cost curve over X51's 1,600 P\* vs Kakuna game lengths (the forensics'
  rows).
  - Two builders: the REFERENCE one (the pin's: 77.8 search CPU-min for that game) and the LEAN screen E′
    (`2eb65b3d9`: 18.0).
  - Two extrapolations past the game's 66 turns: per-decision cost frozen at its turn-66 value (`flat`) or still
    growing (`growth`).
  - It is an ESTIMATE from ONE game.

Abbreviations follow the design:
- S = `r0s4@10`;
- G = WR(P\* + S) − WR(P\*), paired by deal; I = G_U − G_home72;
- DNC-B = DOES-NOT-CLOSE-B;
- pp = percentage points; LB / UB = the lower / upper 95 % bound;
- "sign accuracy" = P(T(S's pick) > T(π's top-1) | S deviates, the two T differ).

## Verdict

- **S-U is a legitimate, independent question** (H-MID review M4). It is not worth its ≈ 560 CPU-h, nor V-RANK's
  ≈ 96, until a ≈ 10–15 CPU-h screen shows that S's DEVIATIONS beat chance on U.
  - The S it tests is, at P\*, a 40 %-rate coin flip around π under its own referee (B2). So its likeliest class,
    DNC-B, is predictable from banked rows.
  - No class it can reach is attributable to V's within-node ranking, the mechanism the design names.
- **As registered at its pin, the read cannot run** (B1). The lean screen (`2eb65b3d9`) has since fixed most of the
  cost. Three items still have to be registered: E′'s parity on U, an attrition rule, and a re-size on the mean.
- **V-RANK is the right referee aimed at the wrong statistic** (M6).
  - Its slope β can hold while search gains nothing: its own probe shows β 0.54 with H ≈ 0.
  - Its guessed P(S-U OFF) ≈ 0.2 makes it roughly break-even as a filter.
  - Run its planned self-referee arm FIRST, cut to S's deviations, as the cheap screen (§ The gate test).
- **Only DNC-B is creditable, and only for this S.** I is not leak-robust in either direction (M1, M2).

## BLOCKER

**B1. At its registered pin the read cannot run. The lean screen fixes the cost, but not the bound, the re-size rule
or the attrition.**

Per searched game, search CPU, from one game's curve over X51's lengths:

| | home72 | H | F | P72 | all | full read (2,400) |
|---|---|---|---|---|---|---|
| median turns (X51) | 31.5 | 38 | 37 | 38 | 36 | |
| **reference** (the pin): median / mean `flat`–`growth`, CPU-min | 6.2 / 29–66 | 12 / 43–100 | 11 / 60–162 | 12 / 49–134 | 9.8 / **45–115** | **≈ 1,800–4,600 CPU-h** |
| reference: share past ≈ 180 CPU-min (the 3-h bound) | 0.04 | 0.06–0.07 | **0.11–0.13** | 0.08–0.09 | **0.07–0.08** | |
| **lean E′**: median / mean `flat`–`growth`, CPU-min | 3.2 / 7–9 | 5.0 / 10–13 | 4.7 / 13–17 | 5.0 / 11–15 | 4.4 / **10–13** | **≈ 410–530** (+ raw units) |
| lean: share past the bound | ≤ 0.005 | ≤ 0.007 | ≤ 0.010 | ≤ 0.013 | ≤ 0.009 | |

- **The pin is the reference builder.** The PREREG plays stage 0 and the read "at the spec's pin". At the pin a
  searched unit is killed at `UNIT_SEARCH_S` (3 h of wall, +10 min) and retried once, and the game is deterministic,
  so the retry dies at the same length.
  - Each such deal burns ≈ 6 h of a heavy slot, then leaves the pairs ("not imputed").
  - That is ≈ 7–8 % of deals: home72 4 %, F 11–13 %.
  - The selection is POST-treatment: the searched game's own length decides it. It is concentrated in long games,
    where the conversion gap lives, and it is uneven across sets. No registered bound applies (root's "> 25 %
    timeouts ⇒ INCONCLUSIVE" is far off).
  - My mean (45–115) and the amendment's independent "≈ 50–74 CPU-min, ≈ 2,000–3,000 CPU-h" bracket each other.
- **The bound is WALL time.** It is 1–1.5× CPU at nice 19 and 2.5–3× under SCHED_IDLE (design §4.2), so whether a
  long deal pairs depends on box contention, which breaks root rule 8. Under E′ the share is ≈ 1 % (P72 and F the highest),
  but it is not zero and still not deterministic.
- **The re-size rule keys on the wrong statistic.** It reads stage 0's MEDIAN game against a bar of 50 CPU-min.
  - The median is ≈ 4–10 CPU-min on either builder, so the rule never fires; the TOTAL is set by the ≈ 6 % of games
    past 100 turns.
  - Twenty stage-0 games hold about one such game, so their mean is unstable too.
- **E′'s parity is shown on ONE home72 game.** It covers the game's 57 decisions and 4 fixtures. U's off-prior sets
  are where the screen works hardest and excludes most: P72's procedural spreads, Hidden Power IV patterns, H's ladder
  sets (M5).

**Required before any searched game counts**, in the amendment that moves the pin:
1. E′'s parity on U decisions, exclusions among them;
2. an attrition rule (M4);
3. a deterministic bound per game: a CPU bound, or the window A;
4. a re-size on projected TOTAL CPU (the mean over X51's lengths), not the median.

**B2. At P\*, S is a chance-level perturbation of π under its own referee. The read's likeliest class is predictable,
and no class can be attributed to V's within-node ranking.** The plateau bank, S's acting rule, 719 states each
(battle-clustered 95 % intervals):

| | P\* (the read's policy) | 25M (control) |
|---|---|---|
| deviation rate (S's pick ≠ π's top-1) | **0.402 [0.365, 0.437]** | 0.415 |
| two 4-world halves deviate to the SAME action, given both deviate | 0.437 [0.367, 0.505] (chance 0.166) | 0.460 |
| within-state Spearman, half against half (reliability) | 0.319 | 0.380 |
| within-state Spearman, S's Q̂ against T (validity) | **0.139 [0.105, 0.174]** | 0.168 |
| **sign accuracy** | **0.513 [0.447, 0.580]**, n 228 | **0.580 [0.518, 0.645]**, n 238 |
| IMP (= L1's registered cell, reproduced) | +0.0001 [−0.0100, +0.0111] | +0.0193 [+0.0083, +0.0305] |

**Reading.**
- S deviates on 40 % of decisions. The design predicted 8–15 % (§7); stage 0's game deviated on 18 of 48.
- V-RANK's probe agrees: its own `pikl_deviation_qhat_gap` counts 289 of 719 deviations. Its §5.1 row "piKL
  deviations 13.8 % of states" is a mislabel. 13.8 % is `imp_pikl_deviation_share`, the share of states whose ONE
  referee seed gives the pick and π's top different results.
- The halves agree on WHICH alternative well above chance, so Q̂ is consistent with itself. It does not order siblings
  the way the truth does (ρ 0.14).
- At P\* the deviations are right about as often as wrong, under the referee that shares S's own model of the
  opponent. The same instrument sees 0.58 at 25M, so 0.51 is not the referee's noise floor.
- V-RANK's probe reads the same thing a different way: Q̂-preferred contenders gain H = +0.0006 [−0.015, +0.016].

**Consequences for S-U.**
- **G_home72 ≈ 0 is expected, not tested.** DNC-B, the design's "likeliest informative class", is what a
  chance-level S produces by construction.
- **A coin-flip S has a set-dependent noise dose.** π is less sure on U (the N2 probe's 1.19× policy shift), so the
  λ = 0.015 anchor binds less there, and S deviates more on U at the same accuracy.
  - If noise costs games against Kakuna, that alone reads as U-LESS.
  - A null on I says nothing about V either.
- **The read is not wasted only if S's deviations beat chance on U where they do not on home72.** That is H-MID's
  mechanism, cut to what one-ply search can use. It is what the gate test below measures.
- **Caveats.**
  - The bank is P\*'s self-play states, loss-enriched and stratified by the top-2 gap (a proportional sample). It holds
    no U state, and it is not the live Kakuna distribution.
  - T's 24 playouts attenuate sign accuracy toward 0.5. The 25M control shows there is still room to detect 0.58.

## MAJOR

**M1. I is not leak-robust, in either direction. Three set-dependent terms of unknown sign sit beside the mechanism.**
The design's argument (§3, A4) is that the leaks act alike on both sets and cancel in I. Each leak's VALUE depends on
the set:
- **Speed ties on the true seed** (20.1 % of decisions on pool states, `search_ledger.md` §6.1). They need equal
  Speeds.
  - Pool and forum teams run standard spreads (max Speed on common bases), so they plausibly tie more.
  - P72's procedural spreads plausibly tie less.
  - Direction: a larger leak on home72 / F. UNVERIFIED.
- **The true-seed screen as a truth filter.** Its information gain grows with the distance between the Smogon prior
  and the truth: P72's procedural sets, H's ladder sets.
  - Direction: a larger leak on U. UNVERIFIED.
- **Exclusions, a dose rather than a leak.** No consistent world means π plays. The screen's support omits IVs
  below 30 and thins off-table spreads, so off-prior sets on P72 and H plausibly exclude more.
  - Stage 0's one home72 game excluded 9 of 57.
  - Direction: less search on U, so I pulled toward 0 or below.
- **A2 (the leaf's opponent model).** Its sign is argued, not identified (m3).
- **B2's noise dose.** It is set-dependent as well.

**Consequence.** A U-HEADROOM cannot mean "V's within-node use survives its level error". NO-U-PREMIUM and U-LESS are
creditable only if U's searched-decision dose, deviation rate and exclusion rate are no lower than home72's. A4's
"leak-robust to first order" should read "leak-confounded, sign unknown".

**Fix.**
- Measure each set's exposure first, in the gate (below): the leak block, the exclusions, the deviation rate, and the
  §6.1 Speed ±1 counterfactual-flip rate per set.
- Register that an I class is read only beside them.

**M2. What each class licenses** (the design's "meaning" column claims more):

| class | licenses | does NOT license |
|---|---|---|
| **DNC-B** | This S (r0s4@10, leak-inclusive, greedy Kakuna, this engine) adds < 4.3 pp on pooled U. If leaks only help S (plausible, UNVERIFIED: no read compares leaky with honest at game level), an honest S adds no more. Search stays paused for goal (b) | That search on V cannot help on U: another λ, more worlds, another leaf or opponent model. A chance-level S (B2) reaches DNC-B by construction. Nothing about V's sibling ranking |
| U-GAIN (leaky) | Building Tier B (fresh dice) and replaying the deals | Any credit to honest search; any goal (b) progress |
| GOAL-B-REACHED (leaky) | Nothing beyond U-GAIN | That goal (b) is met |
| U-HEADROOM (leaky, phase 2) | A follow-up: the fresh-dice replay, M1's covariates, the asymmetric cell | "V's within-node use survives its level error" (M1) |
| NO-U-PREMIUM | This S does not recover ≥ half the premium beyond home72, if M1's dose, deviations and exclusions match by set | That no search recovers it; that H-CTX is right |
| U-LESS | The same, stronger | A mechanism: B2's noise dose and M1's exclusions both produce it |
| UNDECIDED (I's likeliest) | Nothing | — |

DNC-B's 4.3 pp is X51's gap on engine E0. This engine and forum72's new peer pack (`b0441623`, Gyarados) both move the
raw U score (m2).

**M3. I is a co-primary the read cannot afford to resolve. Drop phase 2.**
- **Power.** At I = 0, NO-U-PREMIUM has power 0.33 at phase 1 and 0.57 after phase 2. U-HEADROOM has power 0.49 /
  0.77 at I = +5 / +7 pp, against a registered prediction of +0.5 [−4, +5].
- **Cost.** Phase 2 (+800 searched games, ≈ 140–190 CPU-h under E′) runs whenever phase 1 leaves I UNDECIDED, which
  the design puts at ≈ 0.7.
- **What it buys.** Its likeliest product is UNDECIDED, and its positive class licenses only a leaky follow-up (M2).
- **The discordance d.** S deviates on 40 % of decisions (B2), so nearly every pair diverges within a turn or two of
  turn 10. The d = 0.3 row (pairs kept concordant by "no deviation") does not apply: plan on d ≈ 0.4–0.45.
- **Fix.** Make I REPORTED, plus phase 1's futility stop.

**M4. The forfeit-win guard stops false wins. It turns every failure into a silently dropped pair.**
- **What is covered.** A FORCELOSE that is not our own 250-turn cap is VOID: the peer's disconnect or keepalive
  timeout, the peer's `/forfeit`, our side dropping. The 70-s ON-loop negative control is its slow test. The banked
  scan found 0 such games in 972 units.
- **What is not covered.**
  - (a) Any searched unit that fails for any reason leaves the pairs with no registered bound: the 3-h kill (B1), a
    non-`WorldError` exception, a root-guard mismatch. The guard's own VOIDs leave the same way.
  - (b) All of these are SEARCHED-arm-only and post-treatment. They cluster in long games and in hard-to-fit sets.
  - (c) The patience test covers a 70-s off-loop decision. The reference builder's late decisions ran 243 s, and an
    exclusion 407 s; E′ brings these to ≈ 55 s.
  - (d) About a third of world-build time was a Python decode in the worker thread (`2eb65b3d9`). Under nice-19
    contention it competes with the event loop for the interpreter lock. UNVERIFIED that pongs stay inside 20 s.
- **Fix** (deterministic).
  1. Re-run a failed deal from scratch. It replays exactly, so a timing drop recovers and a deterministic failure
     repeats.
  2. Report each set's permanent-failure rate.
  3. Register worst-case bounds: read every class twice, with each failed searched game scored as a loss and as a
     win. A class stands only if it holds both ways. Above a registered failure share (say 2 % per set), the affected
     primary reads INCONCLUSIVE.
  4. Add one patience run at the longest observed decision under contention.

**M5. The variants E and A each give up more than the draft first said.**
- **E as drafted ("replay the prefix once") cannot be bit-identical.** `2eb65b3d9` reached the same conclusion and
  built E′, the lean early-stopping screen.
  - E′ gives up `core_events`' cross-checks after the first differing line, which is a loss of defect detection.
  - Its parity is ONE home72 game (B1).
  - **E′ parity on H, F and P72 decisions, exclusions included, is owed before a U searched game counts.**
- **A (`r0s4@10-30`) gives up about a third of the in-game gap.**
  - The H-MID review's E[V(t)] gap is +0.032 at turn 10, +0.086 at 30, and +0.112 at the end (m3). Turns 10–30
    hold ≈ 2/3 of the accrual after turn 10.
  - A's "selects nothing by set" is not quite true. U games run longer (H, F and P72 median 37–38 turns against
    home72's 31.5; past turn 60: 0.16–0.22 against 0.105). So a fixed turn window searches a smaller share of a U
    game's decisions.
  - Report the searched decisions per game by set. Alternatively, cap by eligible-decision COUNT (the first K from
    turn 10), which gives the same dose per game.
  - A ends B1's attrition at its source: ≈ 2.5 CPU-min a game under E′, ≈ 100 CPU-h for the full read.

**M6. V-RANK, the gate drafted for S-U, has the right referee and a filter too weak for this decision.**
- **Its strength is real.** Live Kakuna branches by deterministic replay remove the self-referee objection (design
  §10, A1) for ranking.
- **Its ON class is uninformative, by its own probe.** β 0.54 [0.43, 0.64] on P\*'s states, while search gains H ≈ 0
  and L1 +0.0001 there.
  - β is dominated by siblings π would never play. The contrast S acts on is π's top against S's pick, and there B2
    reads chance.
  - So RANK-HOLDS-ON-U, which V-RANK guesses at 0.45, would license an S-U whose S is plausibly still a coin flip.
- **Its value as a filter is about break-even.** It costs ≈ 96 CPU-h (80–115) and guesses P(S-U OFF) ≈ 0.2. Against
  S-U's ≈ 370 (phase 1) to 560 CPU-h, the expected saving is ≈ 75–110 CPU-h, about its own cost.
- **It already plans the cheap half.** Its self-referee arm at the same positions (`oracle_leaf8`, §4.4) is REPORTED.
  That arm, restricted to S's deviations and run FIRST, is the gate test below.
- **Its deviation-facing statistics stay REPORTED** for want of power at one live game an action (V-RANK §5.1
  reading 2). That is right for magnitude. It is not right for a STOP: a coin-flip test needs only a sign, and the
  self-referee gives 24 seeds an action.

## MINOR

- **m1. The deviation and d predictions** (8–15 %, d 0.3–0.4) were set before any deviation was measured.
  - B2 gives 0.40 on the bank, and stage 0 gave 0.375.
  - The smoke's 1 of 12 was the outlier: its forum72 game deviated 0 of 10.
  - The predictions are registered and stay. The power table should be re-quoted at d ≈ 0.4–0.45.
- **m2. DNC-B's bar is engine-relative.** 4.3 pp is X51's gap (0.487 vs ≈ 0.53) on E0, with forum72's old pack
  flattering F slightly (H-MID review m6). Re-quote the bar from this engine's raw arm, or state DNC-B as an absolute
  4.3 pp.
- **m3. A2's sign is not identified.** On home72 the leaf models Kakuna as P\* on a pool team, stronger than Kakuna
  (P\* wins 0.61 at level). On U it models Kakuna as P\* on a U team, ≈ Kakuna's level. So the leaf may be MORE
  accurate about the opponent on U. "A2 biases I toward U-LESS" is a judgement, and its opposite is as plausible.
- **m4. Stage-0 blinding is structurally leaky.**
  - `search_decisions.jsonl`, the cost log stage 0 reads, carries root V and Q̂ per decision, so the cost reader sees
    the game's trend.
  - `AMENDMENT_1_DRAFT.md` §1 (`2eb65b3d9`) now states that searched game's result, against PREREG §4 ("no outcome is
    read or shown"). One game carries almost no information, but the rule exists to keep the habit.
  - Register that cost reports print no V and no result.
- **m5. Protocol-silent opponent choices.** The live true log holds Kakuna's actual choice at every earlier update,
  including choices never executed (full paralysis, flinch, a mon KO'd first). Every world replays them. This is
  plausibly benign, but nothing tests it. The leak block counts `rewritten_moves`, 0 in stage 0's game; report it by
  set.
- **m6. The percentile bootstrap at 20 units a panel** under-covers slightly. It is acceptable for a futility class
  and marginal for a creditable one. A bias-corrected (BCa) check of DNC-B's UB at read time is cheap.

## The gate test (proposed; it needs its own short registration)

**The question.** On fresh U and home72 states from the read's engine, do S's deviations from π beat chance under S's
own model? B2 answers it for home-like states: 0.513 at P\*.

**Why a self-referee is valid for a STOP.**
- The referee shares S's model of the opponent, so it is the best case for S's choices.
- E1-0's direction is that gains shrink against a real opponent.
- So chance-level deviations under it predict chance or worse against Kakuna. This is MY JUDGEMENT.
- A GO licenses only the next stage, never a gain.

**States.**
- Raw P\* against Kakuna greedy, on the read's engine and peer pack.
- Unit seeds 100–109, outside S-U's 0–59, × 20 games × 4 panels: 800 games, ≈ 12 s each.
- Each game keeps its honest input log, as `2eb65b3d9`'s stage-0 replay did.
- From each game, the 2 of P\*'s decisions at turns 10–30 with ≥ 2 legal actions that have the smallest hash (V-RANK's
  `sha256` draw): ≈ 400 states a panel.

**Per state** (offline, on the kept logs):
1. S exactly as registered (`r0s4` through the live searcher's `_search` and keying, E′), so the instrument is the
   read's.
2. Where S deviates, the self-referee on the TWO actions, S's pick and π's top-1:
   `main.value_search.playout.referee`, 24 truth seeds as on the bank, P\* on both sides, common dice across the pair.
3. Per state, recorded:
   - the deviation, the exclusion and the CPU split;
   - the leak block, and the §6.1 Speed ±1 flip rate of the seen mons;
   - optionally a second keying of S, for reproducibility.

**Cost** (ESTIMATE). A search at turns 10–30 is ≈ 9 CPU-s on E′. The referee is 0.4 × 48 playouts × 0.3–1 s. The raw
games add ≈ 6 s a state. That is ≈ 20–35 s a state over ≈ 1,600 states, **≈ 10–15 CPU-h** in short-lane chunks:
≈ 2–4 % of S-U under E′, and ≈ 1/8 of V-RANK.

**The rule** (deterministic; battle-clustered bootstrap, B = 2,000, seed fixed; rule 8's ε):

| outcome | rule | what follows |
|---|---|---|
| **GO** | LB(sign accuracy on pooled U, panels equal-weighted) > 0.50 | V-RANK's live branches, then the amended S-U (B1's four items, M3, M4) if V-RANK does not turn it OFF |
| **STOP** | otherwise | No live read. Recorded: "S at P\* is a chance-level perturbation of π on U (and on home72, B2) under its own model". Search stays paused for goal (b). DNC-B is NOT claimed |

- **Its power.** ≈ 1,200 U states give ≈ 380 countable deviations (40 % deviating, 79 % of them untied, as on the
  bank), so SE ≈ 0.026.
  - P(GO) at a true 0.58 (25M-like) is ≈ 0.88.
  - P(GO) at P\*'s home 0.51 is ≈ 0.06.
- **Expected value** (GUESS: U behaves like home with P ≈ 0.7). It saves ≈ 0.7 × (96 + 370–560) ≈ 330–460 CPU-h, for
  10–15.
- **REPORTED, never decided.**
  - Home72's sign accuracy, which should reproduce B2's 0.51.
  - The U − home72 difference: I's mechanism, sign only.
  - The deviation rate per set: the noise dose.
  - Exclusions, CPU per decision and leak exposure per set: the first cost and confound read on U.
- **GIGO tooth.** Before it reads one U state, the gate's scorer re-scores this review's bank states and must reproduce
  0.513 / 0.580 (the sign test) and L1's +0.0001 / +0.0193 (IMP) exactly.

**Why not something cheaper or richer.**
- **Cheaper.** A zero-playout gate exists only for banked states, and none is on U. The `noise` check above is that
  zero-cost half.
- **Richer.** An all-siblings slope (V-RANK, the survey's rec 4) spends most of its referee on actions S never plays
  (M6).
- **Magnitude.** A decision-level IMP at the game bars needs ≈ 10⁴–10⁵ states (V-RANK §5.1; S-U §10), which is no
  gate.

## Not edited (FINDINGS for the orchestrator)

- `design_v_rank_on_u.md` §5.1's "piKL deviations 13.8 % of states" (40.2 %, B2), and its gate's ON class (M6).
- `design_search_u_headroom.md` §7's deviation prediction and §4.3's d rows (m1); A4's "leak-robust" (M1).
- The stage-0 result stated in `AMENDMENT_1_DRAFT.md` §1 (m4).
- The PREREG: per the brief. Every fix above goes through an amendment registered before a searched game counts.
