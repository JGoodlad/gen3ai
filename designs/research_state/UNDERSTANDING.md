# UNDERSTANDING.md — what we believe about the research, NOW

## 0. What this file is, and how to read a claim

`ARCHITECTURE.md` states what the MODEL is now; `ledger.md` records what was believed at each moment and is never
edited; the `learning/` notes explain concepts. **This file is what we currently believe about the research**:
always-current, present tense, no story of how a belief changed (that is the ledger's job). When the two disagree,
**the later ledger entry wins and this file is a bug: fix it in the same pass.** The one-page dashboard of the same
beliefs (each goal's number, what is running, what waits on the owner) is [`../STATUS.md`](../STATUS.md): STATUS is
the dashboard, this file the reasoning.

**Scope: the Rustboro era** (runs named `rb_…`, from 2026-10-03; `utils/era.py`). Pre-era work appears only as the
few lessons in §2, or with an era caveat where it still steers a decision. The last full text before this rewrite
is `git show cdca64227:designs/research_state/UNDERSTANDING.md`; an older pointer to a section cut on 2026-10-26
(the flywheel §2.x, the pre-era meters §3.3–§3.6, the win-prob critic era §4.1–§4.3 and §4.2b, the search dividend
§4.x) resolves there.

**Words used throughout.** "pp" is percentage points of win rate. A bracket is a 95 % interval unless it says
90 %; LB / UB are its lower / upper bound, and Δ̂ is a point estimate. "Greedy" play takes the top action. A "nat" is
the unit of log loss (lower predicts better).

Every claim carries three things.

**(a) An evidence tag.** A floor is an **operational replicate floor**: what two same-recipe arms differ by, the MAX
pairwise |Δ| over the replicates in hand.

| tag | means |
|---|---|
| **SIGNIFICANT** | magnitude above the replicate floor AND the 95 % interval excludes zero |
| **WITHIN FLOOR** | the interval excludes zero but the magnitude is under the floor |
| **NOT DETECTED** | the interval spans zero |
| **EQUIVALENCE SUPPORTED** | the DELTA's own interval lies inside the floor band (never "the intervals overlap") |
| **INDETERMINATE** | outside the floor AND straddling zero: the instrument answered nothing |
| **REFUTED** | a pre-registered prediction contradicted |
| **UNVERIFIED** | asserted, not checked against code or a measurement |

A registered read uses its own rule's words: **GAIN** (lower bound > 0), **LOSS** (upper bound < 0), **NULL** (the
interval inside the registered band: ±2 pp in-house, ±3 pp against Kakuna), **UNDECIDED** (anything else; at look 1
with a positive point it CONTINUEs to look 2), **NON-INFERIOR** (lower bound above the registered loss bar). Also:
**DESCRIPTIVE** (post hoc, no registered verdict), **PROVISIONAL** (one seed or one checkpoint), **SELF-REFEREE**
(scored by our own network playing both sides). "No effect" is never a label.

**(b) A pointer**: a ledger entry by DATE + *TITLE*, a landing commit (backticked hex, `git show <sha>`), or a path
under `measurements/`. A number with no pointer is not a measurement.

**(c) A regime**: what the number was measured against. The Rustboro regimes:

| regime | what it measures |
|---|---|
| **Kakuna** | our model against Metamon's strongest public model, Kakuna (checkpoint 34, 142.8M parameters; ours has 2.61M), both sides greedy, on CPU: 400 games a panel in 20-game units with paired seeds (`main.anchors`, `main.reads`). The goal reads |
| **the panels** | **home72**: 72 of the 719 teams we train on. Three UNFAMILIAR panels, teams our model never trained on: **H** (`heldout72_hpfix`, 72 ladder-replay teams with Hidden Power repaired), **F** (`forum72`, human-built Smogon-forum teams), **P72** (procedural teams built from Smogon usage). **U** is H, F and P72 at equal weight. Registered in `heldout_panels.json` and `diagnostic_panels.json` |
| **in-house S** | the population score: mirrored-pair games against a frozen 7-member reference population R, 500 pairs a member, greedy, CPU (X35's readout). Every plateau lever's in-pool guard |
| **the fork frame** | an X35 plateau lever is P\* + 20M steps with ONE change, at the production pin, seeds 3501–3503 (3504–3505 at look 2), read against **C0** = P\* + 20M of plain training on the same seeds. Its primary statistic R1 is the lever's S minus C0's |
| **search banks** | banked decision states, every legal action played to the end under shared dice ("common random numbers", CRN), in win probability per decision. **Oracle** worlds know both real teams; **honest** worlds are drawn to agree with everything our side saw |

---

## TL;DR — what we know now

1. **P\* is our strongest model, and it has plateaued.** P\* is X26 (`rb_x26_s2601`) at 280M steps, the named baseline
   `pstar_plateau`. It beats the production baseline 0.694 [0.666, 0.721] in-house. Training it 100M further (380M → 480M,
   X26c) moves neither Kakuna U (+0.08 pp [−3.90, +4.06]) nor in-house S (−0.01 [−1.54, +1.52]). [FLAT · Kakuna and
   in-house S · one seed · `d172c10ed`, `204204bbf`; ledger 2026-10-25 · *X26c CONTINUATION vs KAKUNA*]
2. **Goal (a) is met: P\* beats Kakuna on our own teams**, home72 **0.598 [0.549, 0.644]**. **Goal (b) is not: it is
   level on unfamiliar teams**, U **0.487 [0.459, 0.515]**, about 4 pp short of the ≈ 0.53 a lower bound above 0.50
   needs. The familiarity premium (home72 − U, ≈ 11 pp) has not moved from 161M to the plateau. No panel is shown
   Kakuna-familiar. [MEASURED · Kakuna · `3010c81ba`, `4f555e577`; ledger 2026-10-21 · *X51 PLATEAU vs KAKUNA*]
3. **The deficit is PILOTING an unfamiliar team, not facing one.** Corrected for team strength, piloting costs
   +0.123 [+0.039, +0.206] and facing +0.017 [−0.062, +0.095]. It is about teams unseen BY US (forum and procedural
   teams drop as far as replay teams), and not about the teams being weak. In-house it is invisible: against its own
   25M checkpoint P\* keeps about four fifths of its progress on unseen teams. [MEASURED · Kakuna (X26 @161M) and
   in-house · PROVISIONAL · `8b43eb44a`, `71270538b`, `82858f433`, `fd005a6a4`]
4. **It is lost from level mid-game positions, and it goes with the TEAM, not the set.** From level material at
   turn 10 P\* wins 0.609 on home72 and 0.501 on U; the opening carries none of the gap. Sets we trained on lose
   their edge inside an unfamiliar six (+0.093 [+0.004, +0.176]). [DESCRIPTIVE · banked X51 games, replicated on the
   PastaMittens twins · `c758fcb82`] H-CTX, H-MID and N2 test it (§6).
5. **No lever measured so far closes goal (b).** Against its own control each reads +1 to +3 pp with intervals
   ±3–4 pp wide: practice on 36 teams (X52) +1.3 pp on the unpractised half, UNDECIDED; a wider pool (D1) +2.10
   [−0.92, +5.11], UNDECIDED (final), at −1.61 pp in-pool; set mutation (D3, look 1) +1.27 [−2.63, +5.16], CONTINUE,
   with an engine-mix caveat; the belief prior mix ±1.50, NOT DETECTED; plain training FLAT. [Kakuna · `42bb0858d`,
   `eb99ab362`, `58287c525`, `f299a7243`, `71c76dfbb`, `204204bbf`] Breadth at 10⁴ teams (N1-P) and human data
   (X59-S) are being read.
6. **The belief stack memorises the pool, but that is not where games are lost.** On unfamiliar opponents 63–85 %
   of the hidden slots hold the wrong species guess, yet mixing in the Smogon prior (λ = 0.25: U +1.50 pp [−2.74,
   +5.74]) or using the prior alone (λ = 1: −1.50 [−5.47, +2.47]) moves Kakuna games by nothing detectable at ±5 pp.
   [MEASURED · calibration DESCRIPTIVE, games registered · ledger 2026-10-26 · *H3 … BANKED ON CALIBRATION*, *THE
   BELIEF PRIOR MIX* (both λ)]
7. **Kakuna is the right opponent.** Metamon's ensemble PastaMittens is not stronger for us (U −0.25 pp [−4.23,
   +3.73]), and P\* beats all three of Metamon's human-only models (0.666 / 0.631 / 0.790 on U). Foul Play has never
   played P\*. [MEASURED · Kakuna harness · `82f273eff`, `4d8d9eef8`, `df35d1bb0`]
8. **Population levers: the headroom is there, the cheap levers do not reach it.** A fresh best responder beats
   P\* by +11.1 pp [+8.3, +13.8], saturated by 8M steps; an archive opponent pool (A1, +0.19 pp) and promotion by a sequential
   test (SPRT, −0.25 pp) are NULL in-house. [MEASURED · in-house · `c456b6a09`, `806e50931`, `4290d652a`]
9. **The plateau is a treadmill, not a collapse.** About 30 % of greedy choices change per 20M steps without
   improvement, the self-play pool goes stale and the learning rate halves; no coverage, rank or dormant-unit
   collapse, and no repeat of the pre-era lineage's starvation. [DESCRIPTIVE · one seed · `3e8a41249`]
10. **Search does not pay at P\*, and is not getting closer.** In honest worlds the rollout leaf's edge is gone (it
    was hidden information about the opponent's team); training to the plateau leaves the oracle − honest gap flat
    (−0.003 [−0.020, +0.015]); only identifying the opponent's EXACT team pays; the one cell that replicates (a
    policy-anchored rule at a 4-turn leaf) is nil at P\* (+0.0001). With a real bot as referee the one registered
    read is UNDECIDED (E1-0, −0.024 [−0.054, +0.009]). Search is paused. [SELF-REFEREE banks · `e7048bb1d`,
    `537670f7a`, `f756019d7`, `015f00e20`]
11. **Hidden information is a search lever, not a policy lever.** Telling the network the opponent's hidden facts
    for 3M steps buys nothing (moves −0.38 pp [−1.86, +1.10]) and makes it depend on them; perfect species knowledge
    buys ≈ +2.4 pp, within floor. [NULL · PROVISIONAL at 15M · `ffd2280ee`, `c344ccd1a`]
12. **The architecture is settled for now.** The end state (static per-mon tokens with the recovery and fact
    levers) passed its closing test (+0.60 pp, 90 % [−0.72, +1.92]); the bf16 (16-bit "bfloat") trunk passed its A/B (+1.06 pp, 90 %
    [−1.19, +3.31]) and trains 22.5 % faster. A small entity-and-attention toy beats production at equal wall clock
    at 15M (+6.5 pp), but at the plateaus P\* is ahead of the toy's T\* against Kakuna. [PASS ×2 · `748f90e6f`,
    `89d258af6`, `38d767462`, `3010c81ba`]
13. **Human data carries signal; whether it carries goal (b)'s is open.** Our intent head predicts human opponents
    0.253 nats worse than our own snapshots; a human opponent model learns humans (0.582 nats better than P\*) but
    does not help search. X59-S asks whether human data beats our self-play data. [MEASURED · `f8d58205c`,
    `92af1f121`, `73428c077`]

---

## 1. The mission, the goals and the era

**The mission.** A Generation 3 OverUsed (gen3 OU) generalist that keeps improving: not a specialist, not a search agent. The owner's
permanent side constraint is that it must be able to play the public ladder; no ladder campaign runs without the
owner's approval (`designs/endstate/design_ladder_campaign.md`).

**The current goals (owner, 2026-10-17), in order**, each with its status in STATUS §1:
1. the strongest baseline (P\*, at its plateau);
2. beat Metamon: (a) on our teams (MET), (b) on unfamiliar teams (NOT MET, the programme's focus);
3. know where we are (mapped: the piloting deficit, §3.1);
4. AlphaStar-style improvement: league, prioritised fictitious self-play (PFSP), exploiters (§4.10–§4.11);
5. what search needs, read as a trajectory across checkpoints (§4.7);
6. which collapse we see, and a source of counterfactuals (§4.4).

The plan for goal (b) is `designs/endstate/design_beat_metamon_path.md`; if D3, N1-P and X59-S all miss, the
fallback is `designs/endstate/strategy_if_levers_null_2026-10-26.md`.

**The Rustboro era.** It starts with the X5 belief A/B and X26 (2026-10-03): the fixed bots and the Rustboro anchor
base, the Rust environment core, the END STATE architecture (adopted 2026-10-12, §4.6b) and the bf16 trunk (§4.6c).
Pre-era runs are history and never comparators: their inputs carried garbage-in-garbage-out (GIGO) defects fixed
since, and their environment, opponent regime and team curriculum differ (§2).

**The runs that matter.** Cite each by its NAMED BASELINE (`designs/baselines.json`, resolved through
`agents.training.baselines`; change one only with `python -m main.baselines set … --reason`).

| run | baseline name | what it is |
|---|---|---|
| `rb_x26_s2601` @ 280,000,170 (**P\***) | `pstar_plateau` | the strongest checkpoint; every X35 lever forks from it and every goal read quotes it |
| `rb_es_end2_s2005` | `production` | the production ARCHITECTURE surface: the end state, 15M |
| `rb_es_end2_s200{1,2,3,6,7}` | `outside_es_end2_s…` | the Tier-2 outside panel's members (`designs/endstate/design_evaluation.md` §8.2) |
| X26 @ 380M → `rb_x26c_s2601` @ 480M | — | the continuation that read FLAT (TL;DR 1) |
| X34 @ 290M (**T\***) | — | the entity toy's plateau (§4.9) |

---

## 2. Pre-era lessons that still guide us

Measured before Rustboro on inputs with since-fixed GIGO: directions, never levels. Detail in the cited ledger
entries.

- **A plain continuation is the control.** The pre-era flywheel's one "gift" (v8, +69 anchored Elo) was its parent
  still learning: a plain continuation gained +3.45 pp against the full fold recipe's +4.56 (difference −1.11
  [−3.12, +0.91], EQUIVALENCE SUPPORTED). Hence C0 in every fork read. [ledger 2026-09-06 · *CELL 2*]
- **Exploiters learn counterplay to one target, not skill.** A teacher that beat its parent 0.657 head-to-head was
  level with it on its own teams against a third party, and distilling exploiters robbed off-slice. A head-to-head
  number and a third-party number are different measurements. [ledger 2026-09-21 · *NONE OF THE THREE 5-TEAM
  TEACHERS IS ADMITTED*]
- **In-house meters rank arms, but their pp are not external pp**: a +15.5 pp in-house gain was ≈ +3 pp against an
  outside model. Rustboro shows the same split (TL;DR 3). [ledger 2026-09-20 · *THE CONTINUATION'S +15.50 pp DOES NOT
  APPEAR EXTERNALLY*]
- **One arm is not a result**: of three replicates of a real effect, one cleared zero alone. [ledger 2026-09-05 ·
  *P1*]
- **Learned value heads hit a ceiling as search leaves** (twelve heads at 0.567–0.578 pairwise accuracy; a few
  rollouts beat them). The leaf is bound by information and representation, not by the learner (§4.7). [ledger
  2026-09-19 · *THE LEAF CEILING*]
- **A V-bootstrapped one-ply Q̂ is used only as a lookahead or a diagnostic**, never as a training target (owner
  ruling 2026-09-30, `designs/endstate/design_q_head.md` §5.0, `006156b7`): the critic is blind to the moves the
  policy starves.
- **Learned belief heads memorise a closed pool** (first seen 2026-09-24; re-measured on P\*, §3.1).

---

## 3. Strength against outside opponents

### 3.1 The head-to-head: P\* against Kakuna and Metamon

**Goal (a), and the plateau read (X51).** P\* vs Kakuna, 400 games a panel, the X26 reads' own harness `63f97a57`,
paired seeds: **home72 0.598 [0.549, 0.644]**, MET. Unfamiliar: H 0.497, F 0.530, P72 0.433; **U 0.487 [0.459,
0.515]**, NOT MET, ≈ 4.3 pp short. P\* − X26 @161M on U is +3.0 pp [−1.0, +7.0], not detected; the familiarity
premium stays ≈ 11 pp. Kakuna's checkpoint 28 is not stronger than 34 (0.515 against it), so 34 stays the
opponent. [MEASURED · PROVISIONAL · `3010c81ba`; ledger 2026-10-21 · *X51 PLATEAU vs KAKUNA*;
`measurements/x51_plateau_vs_kakuna_2026-10-20/`]

**The panels are honest.** H can be in Kakuna's data only as its one source replay. Ladder near-copies (≥ 4
shared sets): H 4.2 %, a random ladder team 8.9 %, F 14 %, P72 0 %, home72 47 %. Without H, P\* reads 0.481
[0.447, 0.516], and X51, X26c and D1 keep their verdicts on every subset. Open: whether Kakuna's self-play saw near
copies of H (assumption A-K). [DESCRIPTIVE · `4f555e577`; ledger 2026-10-26 · *GOAL (b)'S PANELS ARE NOT SHOWN
KAKUNA-FAMILIAR*] Metamon's `competitive` set is NOT unfamiliar: 19 of its 20 teams are copies of pool teams;
every read reports its panel's overlap with the pool (`gen3_team_overlap_v1`). [ledger 2026-10-18 · *THE AWAY SET
IS NOT HELD OUT*]

**What the deficit is (X26 @161M, 2026-10-18 / 19; PROVISIONAL, preregistered).**
- **Piloting, not facing (X37a stage 1).** When X26 brings home72 and Kakuna brings H teams, X26 wins in every
  species stratum (the ladder-weighted headline L = 0.630 [0.573, 0.686]); when X26 PILOTS the H teams it reads
  0.380 [0.334, 0.428]. Corrected by a Kakuna-vs-Kakuna cell for team strength, piloting costs **+0.123 [+0.039,
  +0.206]** and facing +0.017 [−0.062, +0.095]; X26 pilots off-meta (FAR-core) teams 0.199 [0.017, 0.385] worse
  than sample-core ones. The correction rests on assumption A-K (Kakuna's own familiarity is symmetric),
  UNVERIFIED. [`8b43eb44a`; ledger 2026-10-18 · *X37a STAGE 1*]
- **Unseen by us, not a different population (X37a stage 2).** Forum72 0.470 and P72 0.453 drop as far as H
  (0.448, Hidden Power repaired) from home72's 0.575. [`71270538b`; ledger 2026-10-18 · *X37a STAGE 2*]
- **Not weak teams (TSR).** Rated by one pilot on both sides, replay teams are ≈ 70 Elo below forum teams, but
  matched on strength the piloting gap stays (+0.28 [+0.12, +0.44] per team). [`82858f433`; ledger 2026-10-19 ·
  *TEAM STRENGTH*]
- **Invisible in-house (X37b).** Against X26 @25M, P\*'s edge is +18.8 pp on home72 and +14.8 pp [+11.6, +17.9]
  on unseen teams; the shortfall (+4.0 [−0.7, +8.4]) is not detected. So in-house meters cannot steer goal (b);
  only Kakuna games can. [`fd005a6a4`; ledger 2026-10-19 · *X37b TRANSFER*]

**Where it is lost (the piloting-gap forensics, 2026-10-26).** [DESCRIPTIVE, post hoc · X51's 1,600 banked games,
replicated on the 1,600 PastaMittens twins · `c758fcb82`; `measurements/piloting_gap_forensics_2026-10-26/`]
- **Not the opening.** Through turn 10 the material state carries none of the 11.2 pp gap (−0.4 pp [−2.6, +1.3]).
  From LEVEL material at turn 10 (≈ 3 games in 4) P\* wins 0.609 on home72 and 0.501 on U (+10.8 pp [+4.8, +17.0]);
  the state carries 2 pp of the gap by turn 20 and 8 by turn 40.
- **Not the kind of decision.** The mix of attacks, switches, setup, status, hazards and Explosions shifts alike
  for both pilots (each pilot-specific shift ≤ ±1 pp).
- **The team, not the set.** P\*'s mons out-kill Kakuna's by +0.081 per entry on home72 and −0.028 on U. Restricted
  to sets in our pool on both sides the drop is **+0.093 [+0.004, +0.176]**: a familiar set loses its edge inside an
  unfamiliar six. Set novelty costs about a third as much again. This bounds what new sets in old teams (D3) can buy
  at ≈ a quarter of the premium.
- Facing stays minor: Kakuna's unseen-set and rare-species mons do not out-kill its familiar ones.

**The levers, each against its own control (Kakuna, U unless stated).**
- **Practice (X52, 2026-10-23).** P\* + 5M piloting only 36 forum teams: practised half +3.2 pp [−0.2, +6.8],
  unpractised half +1.3 pp [−3.5, +6.1]; UNDECIDED. Exposure is worth at most a few pp per team at this dose.
  [`42bb0858d`; ledger 2026-10-23 · *X52 PRACTICE CEILING*]
- **A wider pool (D1, final, 2026-10-24).** Half of each side's teams from 1,981 forum teams: **+2.10 pp [−0.92,
  +5.11]**, UNDECIDED; the gain sits on the forum panel (F +4.0, H +2.1, P72 +0.1); in-house S **−1.61 pp [−2.90,
  −0.32]**, a real in-pool cost. [`eb99ab362`; ledger 2026-10-24 · *X35 D1 LOOK 2*]
- **Plain training (X26c, 2026-10-25).** 380M → 480M: U 0.4875 vs P\* 0.4867, **+0.08 pp [−3.90, +4.06]**, FLAT;
  in-house S −0.01; the 290–380M drift (+2.05 pp / 100M) did not continue. Kakuna-cell play is deterministic per
  (model, seed): P\* re-played at current code reproduced X51 game for game. [`204204bbf`; ledger 2026-10-25 · *X26c
  CONTINUATION vs KAKUNA*]
- **Set mutation (D3 / X41, look 1, 2026-10-26).** 30 % of each side's teams from 2,876 variants of our 719 pool
  teams with 1–2 mons' sets redrawn from Smogon: **+1.27 pp [−2.63, +5.16]** (per seed +2.4 / −2.9 / +4.3),
  CONTINUE to look 2; in-house **−0.24 pp [−1.10, +0.62], NON-INFERIOR**. **Read across two Rust engines** (REVIEW F F-M2; `designs/tools/reads.md` §4): D3's games ran on engine E2,
  C0's banked rows (D1's games) on E1, before the 10-25 observation / label and engine fixes (`d8d674b99`,
  `f9978586b`). The guard's 140 / 140 identical replayed games bound the divergence below about 2.1 % of games
  (95 %), so a bias of Δ̂'s size is unlikely but not excluded [INFERRED]. Look 2 plays every game on E1
  (PREREG AM-D3-ENG). [`58287c525`; ledger 2026-10-26 ·
  *X41 D3 LOOK 1* and its CORRECTION]
- **The belief prior mix (2026-10-26).** P\*'s learned belief deltas are confidently wrong off the pool: 63–85 % of
  an unfamiliar team's hidden slots hold the wrong species (24–29 % on a familiar one), and the true species'
  log-presence is −4.6 to −10.0 against the prior's −2.0 to −3.2 (calibration, 48 games a panel). Our own unseen
  sets are encoded compositionally and are not the problem. In games: **λ = 0.25 (a quarter-weight prior mix) +1.50
  pp [−2.74, +5.74]; λ = 1 (prior only) −1.50 pp [−5.47, +2.47]**, both NOT DETECTED; H3 (the belief-memorisation
  hypothesis) is not killed: its interval is too wide. The memorisation is real and invisible in Kakuna games at ±5 pp; the belief-anchoring levers have no
  game-level case. [`f299a7243`, `71c76dfbb`; ledger 2026-10-26 · *H3 … BANKED ON CALIBRATION*, *THE BELIEF PRIOR
  MIX* ×2; `measurements/belief_prior_mix_2026-10-26/`; `../model/generalisation_review_2026-10-26.md`]
- **Breadth at 10⁴ teams (N1-P / X61P), trained, being read.** Every system that generalises across teams trained on
  10⁴–10⁵ of them; every breadth test here ran at 10³. A fork of P\* with half its teams from 12,000 procedural teams
  (`n1_proc_v1`), 3 seeds, read on F alone by D1's rule (H and P72 overlap the corpus and are reported only). The
  registered guess: Δ_F +1.5 pp, 80 % [−1.5, +4.5]. [`5b84bca62`; `measurements/x61p_n1_production_2026-10-26/PREREG.md`]

**The opponents.**
- **PastaMittens** (Metamon's deployed ensemble over Kakuna checkpoints 28, 34, 30 and Alakazam): P\* 0.484 [0.456,
  0.512] on U and 0.550 [0.501, 0.598] on home72; **Δ-U −0.25 pp [−4.23, +3.73]**, NOT DETECTED, so Kakuna stays the
  opponent at a quarter of the CPU. [`82f273eff`; ledger 2026-10-26 · *P\* vs PastaMittens*]
- **Metamon's human-only models** (trained only on human replays), 1,600 games each, twins of X51's: P\* wins
  **0.666 [0.639, 0.692] vs `SmallRL`** (14M, offline reinforcement learning, RL), **0.631 [0.603, 0.658] vs `LargeRL`** (195M, offline RL),
  **0.790 [0.766, 0.812] vs `LargeIL`** (194M, imitation). Each is weaker than Kakuna on every panel; offline RL beats
  imitation on the same data by ≈ 16 pp. Against all three the unfamiliar panels are level, so P\*'s P72 deficit is
  something Kakuna exploits. These compare recipes, not data (X59-S compares data). [OURS-BEATS ×3 · `4d8d9eef8`,
  `df35d1bb0`; `measurements/human_only_vs_pstar_2026-10-25/`]
- **Foul Play** (Monte Carlo tree search over a hand-written evaluator; a Gen 9 OU champion) has NEVER played P\*.
  By proxy: Kakuna draws it, 0.490 [0.422, 0.559], and P\* beats Kakuna on home72. A pre-era 75M policy lost to it
  0.388 [0.288, 0.497]. [`8cba100f8`, `e00e78da0`; ledger 2026-09-14 · *FOUL PLAY DE-RISKED*]

### 3.2 Reading a rating

1. The in-run headline is `<run>/snapshot_ladder/ladder.json` (dense, ±10), never `eval/elo` (±29).
2. A rating is final only once the run is: Bradley-Terry re-solves every node on every add, and the newest node is
   systematically inflated.
3. Compare runs at matched snapshot COUNT, not matched step.
4. A bot-anchored gap between two models far above the bots needs a direct match before it is quoted.
5. A windowed rating never crosses an opponent-regime boundary; the regime is recorded per run and per eval row.
6. The bot frame is precise and wrong near the frontier (274 Elo of disagreement against a fitted se of 9.5), so a
   strength claim carries an outside opponent's number. [ledger 2026-09-14 · `elo_calibration_external_anchors`]

---

## 4. What the Rustboro era has measured in-house

### 4.4 Collapse: no starvation on X26; a treadmill at the plateau

**The pre-era baseline.** Along the ai_v14 lineage the share of decisive turns where a near-best action got < 1 %
of the policy rose ≈ 0.33 → 0.50 ("starvation"), and the critic was as blind to those moves as the policy.
[ledger 2026-09-29 · *M5 LANE S GATE ④ AT SCALE*; 2026-09-30 · *X4 PRE-READ*]

**X26 does not repeat it (collapse pilot, 2026-10-17).** Starvation is FLAT from 10M to 160M (−0.011 [−0.090,
+0.061] on its own truth) while dominated mass and greedy regret FALL: healthy sharpening, at ≈ 0.3 of decisive
turns. About 3/4 of what starvation there is is MISRANK, not ties; the starved moves are worth +0.35 to +0.39 per
turn in the TRUE world, but an honest one-ply read cannot find them. [DESCRIPTIVE · `393fe716f`;
`measurements/collapse_pilot_2026-10-17/`]

**A cleaner value label does not help (2026-10-06).** Refitting the win-prob readout on 8-rollout CRN labels moves
held-out sibling accuracy −0.014 [−0.032, +0.002] (a gain excluded); an 8-rollout leaf reads +0.077. The bound is
representation or coverage, not the label. [ledger 2026-10-06 · *CRN LABEL REFIT*]

**The plateau (180M–380M) is a TREADMILL (2026-10-20).** ≈ 30 % greedy churn per 20M whose distance barely grows
with lag; the pool goes stale (promotions per 50M 9 → 3, median member age 63M → 135M); the learning rate halves (1.74e-4 →
8.4e-5) under its adaptive controller, which holds the policy change per update (the KL divergence) at
0.017–0.019. Starvation creeps up +0.03 to +0.06. No coverage, rank, dormant-unit or opponent-model collapse. The toy shows the same halving and staleness, but its
rate held for 80M before its plateau, so the step size covaries with the plateau without coinciding. Every X35 fork
starts at P\*'s halved rate, so a NULL there reads "NULL at the halved rate". The learning-rate floor (LR-F) tests
it (§6). [DESCRIPTIVE · PROVISIONAL, one seed · `3e8a41249`; ledger 2026-10-20 · *X26 COLLAPSE BATTERY*]

### 4.5 Beliefs: hypothesis tokens adopted; hidden facts are not a policy lever

**X5 (look 3, final, 2026-10-07).** Hypothesis tokens ("fixed_mass": concrete weighted guesses of the opponent's
hidden mons) against the older belief "blob", 8 seeds a side: matched steps −0.99 pp [−2.92, +0.94], **NON-INFERIOR**;
matched wall time −5.91 pp, INFERIOR (a cost-now finding); opponent-intent log loss −0.151 nats, better. Adopted
under the owner's pre-committed rule; the blob path is deleted. [`c344ccd1a`; ledger 2026-10-07 · *X5 A/B LOOK 3*]

**Perfect information buys little strength.** A network trained TOLD the opponent's species beats blob by +2.44 pp
[+0.31, +4.57], told the whole set +2.18 [+0.53, +3.84]; both under the 4.57 pp floor, and the set adds nothing over
the species. Handing an untrained network the truth costs it 8.1 pp. [WITHIN FLOOR · same entry]

**The knock-in forks (2026-10-15).** The end state forked for 3M steps with the truth in 95 % of rows: told moves
**−0.38 pp [−1.86, +1.10]**, items +0.13, species −0.59, all NULL; spreads NOT DETECTED. Withheld at play, the moves
and species forks lose ≈ 2.4–2.6 pp: the network swaps its own inference for the told facts. Privileged information,
if ever used, belongs in the critic. [NULL · PROVISIONAL (+3M from one 15M parent, no placebo) · `ffd2280ee`;
`designs/endstate/design_hidden_info_binding.md` §8.7]

### 4.6 The static-token screen (closed 2026-10-09)

Static per-mon tokens alone were slightly weaker than the legacy encoder: look 3, 8 seeds a side, **−2.32 pp
[−3.68, −0.95]**, NOT DETECTED as non-inferior at the 3.5 pp margin (t 1.623 against 1.874). Not slow learning: the
deficit is flat over training; it is expressiveness for board facts that arrive only through attention (own-side
Spikes R² 0.33 vs 0.46), worst on stall teams (−10.7 pp). A probe battery found five decision facts poorly decodable
in both encodings (the end-of-turn race, boost size, the opponent's HP at our move seats, the speed margin, their
Spikes). The fix was the end-state bundle (§4.6b); the isolated value of its third trunk round is unknown. [NOT
DETECTED · `6956cccf8`; ledger 2026-10-09 · *STATIC-TOKEN SCREEN LOOK 3*, *WHERE IS STATIC WORSE?*, *THE
REPRESENTATION PROBE BATTERY*]

### 4.6b The end-state closing test: PASS (2026-10-12)

The full end state (static tokens + the recovery levers incl. a third trunk round + the fact levers), 8 seeds × 15M
against production's legacy encoder, the registered 8 × 8 mirrored cross: **Δ̂ +0.596 pp, 90 % [−0.723, +1.915]**.
Both clauses of the owner's rule hold (upper > 0, lower > −2.0): **PASS**, adopted; not a superiority read. Which
levers carry the recovery is unknown (no bisection). [PASS · `748f90e6f`; ledger 2026-10-12 · *THE END-STATE
CLOSING TEST: PASS*]

### 4.6c The bf16 trunk: PASS (2026-10-14)

The trunk rounds under bf16 autocast (heads, losses, decisions and rollouts in fp32), 8 seeds × 15M a side, the same
rule: **Δ̂ +1.06 pp, 90 % [−1.19, +3.31]**, PASS; `train_ms` 54.3 s → 42.1 s (−22.5 %). It covers fresh runs, not an
fp32 lineage switched midway: a fork inherits its parent's precision. [PASS · `89d258af6`; ledger 2026-10-14 · *THE
bf16 STRENGTH A/B: PASS*]

### 4.7 What search needs: the information, not the acting rule

The consolidated search view is [`../endstate/search_ledger.md`](../endstate/search_ledger.md) (always-current;
every search result updates it). The findings that steer it, all on search banks:

- **Oracle rollouts carry value the critic lacks** (2026-10-11): an 8-rollout leaf's pick beats the policy's by
  +0.081 / +0.071 per decision on the end-state and production seeds, the critic by +0.055 / +0.059 (SIGNIFICANT,
  oracle world). Trusting the policy prior hurts. [ledger 2026-10-11 · *VALUE-DISCRIMINATION PILOT PHASE 1*]
- **In honest worlds that edge is gone**: +0.007 / +0.004, not clear of zero; oracle − honest ≈ +0.07. The value was
  hidden information about the opponent's team. Oracle playouts as value labels for a frozen body do not help.
  [SIGNIFICANT (the information cost) · NOT DETECTED (honest gain) · ledger 2026-10-11 / 12 · *PHASE 2 / 2b*, *PHASE
  2c + the v2 RE-READ*, *PHASE 3*]
- **Which hidden facts bind (Tier 1 of the hidden-information study, 2026-10-15, and its replications).** The gap is ≈ 7 points per decision;
  "the opponent's hidden moves are THE binding constraint" is REFUTED. At 15M the never-seen mons carry more
  (replicated by a blind build at the 8-rollout leaf; U-FIRST on the competent-opponent bank E2); at P\* the two
  classes are SHARED. The gap falls with the turn (+0.095 early → +0.033 late), and no stage has honest search
  clearly above the critic. [ledger 2026-10-15 · *HIDDEN-INFO STUDY TIER 1*, *THE RED TEAM*; 2026-10-16 · *THE BLIND
  U-ARM BUILD*; 2026-10-18 · *E2*]
- **Maturity does not bring honest search closer (2026-10-25).** X26 25M → P\* on 719 plateau-drawn states: the gap
  +0.061 → +0.058, change **−0.003 [−0.020, +0.015]**, NOT DETECTED; the network's belief is worth p ≈ 0.014 of
  weight on the true world against the ≈ 0.2–0.5 a search needs; no stage opens. [SELF-REFEREE · `e7048bb1d`;
  `measurements/hidden_info_maturity_x26_2026-10-18/`]
- **The acting rule is not what binds (L1, 2026-10-20; replication 2026-10-25).** A policy-anchored rule (piKL)
  lifts honest search from below the policy to about the policy, never above it on X26. Its one paying cell (the
  4-turn leaf) replicates thinly, **+0.0097 [+0.0025, +0.0169]**, all of it at the 25M anchor (+0.0193) and nil at P\*
  (**+0.0001 [−0.0097, +0.0101]**). [PAYS-SCOPED · REPLICATES thinly · `f756019d7`; `measurements/search_l1_plateau_2026-10-25/`]
- **Only identifying the opponent's whole team pays (world sampler stage 1, 2026-10-21).** The training pool's
  consistent teams beat today's honest worlds by +0.038 [+0.022, +0.055] (0.72 of the oracle gap); every
  open-population sampler (Smogon teammates, a ladder-corpus model, the forum corpus, the network's own species) is
  NOT DETECTED, and calibration buys nothing. [`537670f7a`; `measurements/world_sampler_stage1_2026-10-19/`]
- **The opponent's own choices identify the truth only when the opponent plays like our network (L3, off-pool,
  stage 1c, 2026-10-20 / 21).** At P\* against our snapshots the truth outranks honest worlds 79 % of the time;
  against scripted bots not at all (0.472). Worlds resampled by that likelihood are slightly nearer and do not pay
  (best arm +0.007 [−0.003, +0.017]). A human opponent model as the judge does no better (R-b2, §4.8). [`0d0058408`;
  ledger 2026-10-20 · *L3*, *L3 OFF-POOL*; 2026-10-21 · *stage 1c phase B*]
- **The referee (E1-0, 2026-10-25).** With a real bot scoring the banked bot-stratum picks, oracle-bot − self is
  −0.024 [−0.054, +0.009]: UNDECIDED, no relabel; descriptively the self-referee's gains shrink against the bot
  (oracle gap +0.050 → +0.010 at 225M). Every figure above is a self-referee size. [`015f00e20`;
  `measurements/search_e1_0_2026-10-26/`]

**The view.** Search is paused. Enabling it needs a world sampler that holds the opponent's actual team; policy
training absorbs some of what perfect information would buy, but not enough to make honest search useful.

### 4.8 Human data: signal in the labels; a human opponent model that learns but does not judge

- **The intent head's gap (X50 R1 / R0).** X26's intent head predicts human ladder opponents (rated ≥ 1500) worse
  than its own pool snapshots: through the fixed spectator pipeline **+0.253 nats [+0.232, +0.276]**, most on
  switches (+0.411; on a human switch it is worse than uniform). Human labels carry signal self-play labels lack.
  [SIGNIFICANT · `6888a5ec4`, `f8d58205c`; ledger 2026-10-20 · *X50 R1*, 2026-10-23 · *X50 R0*]
- **The human opponent model (HOM; R-b1 / R-b2).** P\* fine-tuned on 1.76M completed human choices predicts held-out
  humans **0.582 nats [0.565, 0.599] better than P\*** (top-1 57 % vs 46 %); on accounts never seen in training
  0.503 [0.467, 0.542], so mostly not account memory. As search's judge it does not help: the truth's rank against
  scripted bots 0.459 [0.406, 0.510] (P\* 0.472), HOM-NONE. Whether a human's choices identify a human's team
  (L3-H) is unread. [SIGNIFICANT · `92af1f121`, `73428c077`; ledger 2026-10-24 · *X50 R-b1*, *X50 R-b2*]
- **The data card (2026-10-25).** The frozen spectator snapshot `hr_spec_v1`: 376,410 ladder logs, 0 parse errors;
  actors rated ≥ 1500 give 1.80M informative rows; ≈ 6 % of sides are rule-flagged bots; keeping our panels
  unfamiliar removes ≈ 71 % of ≥ 1500 battles. Whether human labels carry the unfamiliar-team skill goal (b) lacks
  has no evidence either way. Human replays may train the policy only as a toggleable lever, off by default (owner).
  [DESCRIPTIVE · `14ae42035`; `measurements/human_replay_quality_2026-10-25/`;
  `../endstate/human_data_architecture.md`, `../endstate/design_human_replay_lever.md`]
- The X50 reads and the HOM's rows used the pre-fix intent masks (opponent Explosion / Self-Destruct and 12–17 % of
  attacks unlabelled); no result is invalidated, but an extraction at HEAD is a different population. [ledger
  2026-10-26 · FIX · *the obs / label GIGO audit*]

### 4.9 The entity toy: ahead at equal compute, behind at the plateau; depth 2 → 4 UNDECIDED

- **At production's 15M budget, equal GPU wall clock** (T-ENT: 1.42M parameters, ≈ 3.3× production's steps per
  second, so ≈ 46–48M steps): **+6.5 pp, 90 % [+4.5, +8.4]** against the 8 bf16 end-state finals; −14.1 pp at matched
  steps. [DESCRIPTIVE (seeds-as-units interval) · `38d767462`; ledger 2026-10-16 · *THE T-ENT TOY AT EQUAL COMPUTE*]
- **At the plateaus the order reverses**: against Kakuna the toy's T\* (290M) is −10.5 pp (home72) and −7.3 pp
  (forum72) below P\*, both detected. [`3010c81ba`] The registered toy-cap vs X26-plateau head-to-head (R-T2) is
  unread.
- **Depth (X47, 2026-10-21).** T\* grown from 2 to 4 attention rounds, +70M, against toy C0 at equal learn time: **Δ
  −0.88 pp [−3.20, +1.44]**, UNDECIDED with Δ̂ ≤ 0, FINAL at look 1. The new rounds did change the policy (zeroing
  them moves the argmax on 25–29 % of decisions) without raising S. X47b (production 3 → 4) is not run. Never "depth
  does not help": one growth recipe, on the toy. [`d6e57f2e9`; ledger 2026-10-21 · *X47 LOOK 1*]

### 4.10 The population at the plateau: no archive hole, a real best-responder gap

- **The archive (X55 / LG-0, 2026-10-20).** X26's archive Nash sits in C0's own recent pool (0.90 of the mass on
  164–268M); P\* scores 0.45–0.61 against every node and 0.514 ± 0.014 against the Nash mixture: a wide flat top with
  no hole to repair. [DESCRIPTIVE · `bb962c132`; ledger 2026-10-20 · *X55 LG-0*]
- **The cycle monitor (2026-10-19).** X26's snapshot ladder is non-transitive at every 10M check, with ≈ 3–5 pp of
  per-edge cyclic spread around P\*: MATERIAL at 28 of 37 checks under the size bar, so P\* is a plateau by the in-run
  meter, with cycling suspected, not confirmed. [DESCRIPTIVE · `6730430b9`; ledger 2026-10-19 · *THE CYCLE MONITOR*, *THE SIZE BAR*]
- **The archive pool (X35 A1, 2026-10-22).** Half the self-play games against 8 frozen X26 nodes, 3 seeds a side:
  **R1 +0.19 pp [−1.07, +1.46]**, NULL (final). The variance gate passed (seed σ̂ 0.41 pp against a 1.2 pp bar), so
  the X35 frame resolves ±2 pp. [`806e50931`; ledger 2026-10-22 · *X35 A1 LOOK 1*]
- **The best responder (R0, 2026-10-21 / 22).** A fresh exploiter trained against the frozen target finds **+11.1 pp
  [+8.3, +13.8]** against P\* (SATURATED, 90 % of it by 8M) and +11.8 pp against the toy's T\*; its edge transfers to
  C0 (+14.5 [+7.6, +21.2]). Both plateau policies leave ≈ 11–12 pp on the table; whether one-ply search defends it is
  deferred with search. [`92f1aa93e`, `c456b6a09`, `1459416fb`]

### 4.11 A fresher pool via SPRT promotion: the lever acts, S does not move (2026-10-24)

SPRT (a sequential probability ratio test) admits each 2M snapshot to the pool only on a fresh test (H0 0.50 vs H1
0.55). It acted as predicted: 10 of 30 accepted, 10 promotions against C0's 3, median pool age 60–68M against 72M.
**R1 −0.25 pp [−1.65, +1.16]**, NULL (final). With A1, neither a fresher pool nor an older one moves S at P\*: pool
freshness is not the binding constraint. It costs 15 % of throughput. [`4290d652a`; ledger 2026-10-24 · *X35 SPRT
LOOK 1*]

---

## 5. Retired hypotheses

| hypothesis | retired by |
|---|---|
| The unseen-team drop is the replay POPULATION (H4) | forum and procedural teams drop as far (X37a stage 2) |
| Unseen teams are simply WEAKER | matched on team strength the piloting gap stays (TSR) |
| The opponent's hidden MOVES are THE binding constraint on honest search | REFUTED by Tier 1, in both builds (§4.7) |
| An archive of past selves holds holes a league would repair | X55's flat top; A1 NULL (§4.10) |
| Pool FRESHNESS binds S at the plateau | SPRT: 3× the promotions, S −0.25 pp (§4.11) |
| The opponent's choices, judged by our network or a human model, make search pay | stage 1c NOT DETECTED on every arm; HOM-NONE (§4.7) |
| Plain training past 380M keeps lifting goal (b) | X26c FLAT (§3.1) |
| Better policy-side beliefs are a strength lever at this maturity | the knock-in forks NULL; the prior mix NOT DETECTED (§4.5, §3.1) |

The flywheel-era retirements (fold dose, length, team count, budget, teacher quality, the architecture account) are
retired for the fold-gift question only; they live in the last full text (§0) and the ledger (2026-09-05 / 06).

**A retired hypothesis stays retired at the power it was retired with**: a NULL at ±2 pp rules out effects above
±2 pp, not above zero.

---

## 6. Open questions

> The ranked queue is [`EXPERIMENT_BACKLOG.md`](EXPERIMENT_BACKLOG.md); builds are in
> [`../ops/TASK_BACKLOG.md`](../ops/TASK_BACKLOG.md). Each question below names the read that answers it.

**In flight**

| question | the read that answers it |
|---|---|
| Does breadth at 10⁴ procedural teams lift unfamiliar-team piloting at the plateau? | **N1-P look 1** (X61P, D1's rule on F; [`PREREG.md`](measurements/x61p_n1_production_2026-10-26/PREREG.md), `5b84bca62`). GAIN funds **X53-N1**, a fresh production run on 10⁴ teams (`../endstate/design_x53_n1_fresh_breadth.md`); NULL rules out only "N1 at the plateau" |
| Is human data better than our own self-play data? | **X59-S**: three matched student arms, five seeds each; S-H − S-S answers it, S-H − S-P picks the lever ([`PREREG.md`](measurements/x59s_matched_students_2026-10-25/PREREG.md), `5cbf46dc0`). It decides whether the human-data switch and X60 get built |
| Is the familiarity premium TEAM context or set knowledge (H-CTX)? | **H-CTX on rc72**: P\* vs Kakuna on 72 new teams of verbatim pool sets, ≤ 2 shared with any pool team; ≥ 0.57 kills H-CTX, ≤ 0.52 supports it ([`PREREG.md`](measurements/hctx_rc72_2026-10-26/PREREG.md), `b712c46d1`) |
| In the level mid-game positions, does P\*'s critic mis-value unfamiliar teams, or does its policy fail to act? | **H-MID**: critic calibration at level turn-10 states, home72 vs U, on banked X51 games; CRITIC-MISCALIBRATED-ON-U vs POLICY-CANNOT-ACT ([`PREREG.md`](measurements/h_mid_value_test_2026-10-26/PREREG.md), `2dd522785`) |
| Does P\* key on the EXACT identity of its own sets (a memorised team → plan)? | **the N2 identity test**: pool-team KL ≥ 1.5 × unfamiliar-team KL under a same-species Smogon redraw; KILL lever N2 if the ratio's interval covers 1.0 and excludes 1.5 ([`PREREG.md`](measurements/n2_identity_test_2026-10-26/PREREG.md), `98bcb1146`) |
| Does the plateau resist new data a young network would absorb (plasticity)? | **the plasticity probe** on X26 @25M / @100M / P\*; its bars (LOSS / NO-LOSS / UNCLEAR) are drafted in `../endstate/strategy_if_levers_null_2026-10-26.md` §2 and fixed in its own registration before any number |

**Queued and planned**

| question | the read that answers it |
|---|---|
| Is set mutation a goal-(b) lever? | **D3 look 2**: seeds 3504–3505 (≈ 8.1 GPU-h), every game on engine E1, final (`measurements/x41_d3_setmut_2026-10-25/PREREG.md`) |
| Is the plateau the learning-rate controller's halving (§4.4)? | **LR-F**: P\* continued with a floor on the learning rate, seeds 3502–3503 (X35 `measurements/x35_breakers_stage_2026-10-17/STAGED_PLAN.md` §4.1.2) |
| Do the breadth levers help by +1–2 pp each, below what one read can see? | **PR-1**, the zero-GPU precision read: D1, D3 and N1-P against C0 pooled on F, 38,000 new Kakuna games at one engine (`../endstate/strategy_if_levers_null_2026-10-26.md` §3.1); the fallback's first branch |

**Open, no read scheduled**

| question | what would settle it |
|---|---|
| Did Kakuna's self-play see near copies of our unfamiliar panels (assumption A-K)? | the sealed `forum216b` panel and the reverse cell; the open part needs downloads not made |
| Does Foul Play beat P\*? | a direct read; only the Kakuna proxy exists (§3.1) |
| Do a human's choices identify a human's team (L3-H)? | the L3 read on ladder states with a human opponent model |
| Does one-ply search add to a best responder, or defend its target? | R0's search cells, DEFERRED while search is paused |
| Does the value pool need an explicit SIDE embedding on the 12 entity tokens? Today it separates ours from theirs only through content asymmetry (a probe reads own-team identity at 0.97 from `value_pooled`); belief machinery that makes opponent tokens look like ours would erode that. UNVERIFIED as a defect | first a test that swapping the two sides' tokens flips the value pool's "who is winning"; then, if it fails, a learned side embedding behind a flag in a fresh arm (an ARCH-surface build) |

---

## 7. Standing rules of evidence

Numbered as first written, so ledger citations still resolve; rules 18, 20, 27 and 28 retired with the pre-era
instruments and boundaries they governed.

1. **PRE-REGISTER before any number exists**: both branches, the bar, the comparator, and the PRIOR the result reads
   against.
2. **One arm is not a result**: replicate seeds, and attach the single-arm caveat to any one-arm reading.
3. **A floor is the MAX pairwise |Δ| over the replicates in hand**, and belongs to its regime and depth.
4. **A continuation control at matched depth** wherever a delta is taken against a frozen parent (C0 in the fork
   frame).
5. **Matched SNAPSHOT COUNT** for any cross-run rating comparison (§3.2).
6. **Equivalence needs the DELTA's own interval inside the bar.** A bar against a point estimate, or "the intervals
   overlap", is vacuous.
7. **Validate by executing**: `python -m main.launcher --dry-run`, not clause-checking.
8. **Check the banked reads before promoting a cell**: the answer may already exist.
9. **Quote a model from the RESOLVED file**, never from an argv: a bare run dir means its last snapshot.
10. **The team is the unit**: cluster-bootstrap over teams; state- and team-weighted statistics have disagreed in
    sign.
11. **Human-readable names**: every code (C0, D3, H-CTX) gets a description where it is used.
12. **A timeout is never a semantic outcome**, nor a draw; above 25 % timeouts a read is INCONCLUSIVE.
13. **A result is not a result until every registered look has landed.**
14. **Write the kills**, here or in the ledger.
15. **A windowed statistic never crosses an OPPONENT-REGIME boundary** (a step in the opponent distribution);
    read each arm's own boundary, never assume a common step. [ledger 2026-09-06 · STANDING RULE 15]
16. **An instrument that reads an artifact another process is still writing asks first whether it is complete.**
17. **A statistic on the eval-trace tree is reweighted by each cycle's recorded capture rate** (the tree is
    loss-enriched by design); a tree without a manifest is SELECTION UNKNOWN.
19. **Find which variance component dominates, then replicate at THAT level**: eval-draw variance needs more draws,
    run-level variance needs more runs.
21. **A post-hoc candidate that replicates at about half its size with intervals covering zero is SELECTION
    INFLATION**: NOT CONFIRMED, and a reason to stop.
22. **A control's replicate floor bounds the control, not the lever**: a single-arm detection is a candidate until
    its own seed replicate agrees.
23. **A meter whose value depends on box throughput needs a contemporaneous control or width matching.**
24. **A committed `ladder.json` carries the recipe it was fitted with**; quote it only if its recipe stamp matches
    the current fitter.
25. **Measure the label ceiling before reading a bar against it**, and read pairwise accuracy on contested siblings
    before any battle.
26. **A mirror battery's null is its contemporaneous unsearched cell**, not 0.500, and its width is set by the arm's
    own spread.
29. **Every outside read reports its panel's overlap with our training pool** (`summary.json["team_overlap"]`): a
    panel described as "theirs" was 19 of 20 our own teams. [ledger 2026-10-18 · *THE AWAY SET IS NOT HELD OUT*]
30. **A banked cell is reused only at the same engine, or after a game-for-game replay guard, and the mix is
    stated** (the D3 engine-mix caveat). [ledger 2026-10-26 · CORRECTION · *THE VIEWS AUDIT*]
31. **A search number names its referee**: a self-referee size is not a gain against a real opponent (E1-0).

---

## 8. Pointers

| what | where |
|---|---|
| the append-only record | [`ledger.md`](ledger.md): cite an entry by DATE + TITLE, or by its landing commit |
| finding an entry in it | [`ledger_index.md`](ledger_index.md), generated (`python -m main.ledger_index --write`); never hand-edit it |
| every number behind a claim | [`measurements/`](measurements/): each record carries its checkpoint, step, state count and date |
| the dashboard · the plans | [`../STATUS.md`](../STATUS.md) · [`../endstate/README.md`](../endstate/README.md) |
| goal (b)'s plan · the fallback if the levers miss | [`../endstate/design_beat_metamon_path.md`](../endstate/design_beat_metamon_path.md) · [`../endstate/strategy_if_levers_null_2026-10-26.md`](../endstate/strategy_if_levers_null_2026-10-26.md) |
| the search view | [`../endstate/search_ledger.md`](../endstate/search_ledger.md) |
| the literature survey | [`literature/sota_survey_2026-10-26.md`](literature/sota_survey_2026-10-26.md) |
| outside opponents | [`../ops/EXTERNAL_ANCHORS_SOP.md`](../ops/EXTERNAL_ANCHORS_SOP.md) |
| what the MODEL is now · how it got here | [`../ARCHITECTURE.md`](../ARCHITECTURE.md) · [`../CHANGELOG.md`](../CHANGELOG.md) |
| the orientation, the defect genres | [`README.md`](README.md) |

**Instruments** (offline; none needs a GPU): `python -m main.reads` (registered reads) · `python -m main.anchors`
(outside opponents) · `python -m main.h2h` · `python -m main.elo` · `python -m main.best_response_gap` ·
`python -m main.dose` · `python -m main.lineage` · `python -m main.capacity` · `python -m main.prober.query` ·
`python -m main.ledger_index`.
