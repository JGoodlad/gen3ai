# The most direct path to a reliable win over Metamon's Kakuna

**Status: DESIGN 2026-10-20 (orchestrator brief; planning input; no game played, nothing built).** ALWAYS-CURRENT
(owner, 2026-09-27): a later read, decision or build that differs updates this doc in the same commit, and the
Decision record (§9) says what changed and why. Every number below is quoted from a banked read with its directory;
every effect size that is not measured is marked **GUESS** or **ESTIMATE**.

Abbreviations: pp = percentage points; CI = 95 % interval; LB = the interval's lower bound; P\* = X26's plateau
checkpoint (`rb_x26_s2601` @ 280,000,170); T\* = the toy's plateau checkpoint (`rb_toy_ent_s4002_deep` @ 290,000,087);
home72 = our 72 sample teams (all in the training pool); H = `heldout72_hpfix`; F = `forum72`; P72 = 72 procedural
Smogon-usage teams; U = the three UNFAMILIAR panels H, F and P72; "pilot" = play a team, "face" = play against it.

---

## 0. One screen

- **Goal (a), our teams: met, once, before the plateau.** X26 @161M beats Kakuna on home72, **0.575 [0.526, 0.623]**
  (n = 400, greedy against greedy). P\* has never been read against Kakuna. Two loose ends: P\* is unread, and Kakuna's
  strongest CHECKPOINT is unread (§2, FINDING 1). Closing both costs **≈ 2 lane-hours of CPU and 0 GPU**.
- **Goal (b), unfamiliar teams: not met, and it is about 6–8 pp away.** X26 @161M scores 0.470 on F, 0.448 on H and
  0.453 on P72, a pooled **≈ 0.457**. A pooled LB > 0.50 needs about **0.53** at 1,200 games (§1).
- **The deficit is PILOTING an unfamiliar team, and only Kakuna sees it.**
  - Against Kakuna, X26 pilots unfamiliar teams 12 pp worse after correcting for team strength (X37a PILOT_c +0.123),
    and 11–28 pp worse at matched strength (TSR).
  - Facing unfamiliar teams costs nothing (FACE_c +0.017).
  - Off-meta teams cost most: X26 pilots held-out FAR teams at 0.263.
  - In-house, P\* shows no piloting deficit beyond team strength (X37b), so **self-play meters cannot steer this
    goal**. Only Kakuna reads can, and they run on CPU at ≈ 10 s per game on one lane.
- **No lever we have measured is expected to close 6–8 pp alone.** Ranked by expected pp per unit of cost:
  1. **X51: read where P\* stands** (CPU only).
  2. **X52: a PRACTICE-CEILING fork** (≈ 2.5 GPU-h). It tests whether exposure to a team fixes piloting it, and
     whether that transfers to the same population's other teams. It sizes or kills the D1 family before ≈ 13–60
     GPU-h are spent on it.
  3. **X38 / D1: widen the pool**, already staged in X35. Currently it sits ≈ 78 GPU-h deep in the queue (FINDING 3).
  4. **X41 / D3: set mutation.** Build it now on CPU; run it only on D1's verdict.
  5. **X53: a fresh wide run** (pool + forum teams from step 0) with a 100M look (≈ 21 GPU-h), if D1 reads NULL from
     plasticity or GAIN but short.
  6. **X54: the goal's confirmation read** on a FRESH panel (FINDING 2): **`forum216b`**, 216 human-built teams,
     carved out of D1's corpus and REGISTERED and SEALED on 2026-10-20, before D1 launched.
- **Low rank for this goal:** the X35 plateau breakers, more capacity, X49, search, a Kakuna-trained exploiter, an
  ensemble of our checkpoints and test-time adaptation as a deployed regime. Each lever's reason is in §5.

---

## 1. The goal, stated so that it can be met or missed

The owner's goal (2026-09-27, `project_goals_metamon_foulplay`): our win rate against Metamon's best model has a
Wilson LB > 0.50 "on our teams, at Metamon's strongest temperature". The orchestrator added a held-out goal
(2026-10-18, `design_diversity_teams_opponents.md` §9 Q3): the same bar on the held-out panel, at the plateau. This
doc reads the brief's two targets as:

| target | cell | statistic | bar | status (X26 @161M) |
|---|---|---|---|---|
| **(a) our teams** | symmetric home72: both sides draw from home72 | Wilson on n games | LB > 0.50 at Kakuna's strongest checkpoint and temperature, at the PLATEAU | **met at 161M**: 0.575 [0.526, 0.623]; P\* unread; checkpoint unread |
| **(b) unfamiliar teams** | symmetric cells on H, F and P72: both sides draw from one unseen panel | the equal-weight mean of the three panel scores, Kish effective n (D1's H statistic, `main.x35.d1`) | LB > 0.50 | **not met**: 0.457 pooled; F 0.470 [0.422, 0.519], H 0.448 [0.400, 0.496], P72 0.453 [0.404, 0.501] |
| L, the external headline (reported) | we pilot home72, Kakuna pilots H by `ladder_weights_v1` stratum | stratified, team-clustered bootstrap | — | 0.630 [0.573, 0.686] |

**Why (b) is the symmetric cell, and not L or the reverse cell:**

- In a symmetric cell both sides draw from the same set, so team strength cancels in expectation. TSR measured H
  ≈ 70 Elo weaker than human-built teams, and that cancels too.
- L is already above 0.50 (§3). If the owner reads the goal on L, the deployment-shaped goal is met. The brief's (b)
  is the harder, team-neutral question.
- The reverse cell (we pilot unfamiliar teams, Kakuna pilots home72) is the cleanest piloting meter. It is reported
  beside (b), never as the goal: it carries the opponent's familiarity.

**How far (b) is.** Wins needed for a Wilson LB > 0.50: 220 / 400 (0.550), 428 / 800 (0.535), 634 / 1,200 (0.528),
1,249 / 2,400 (0.520). From the pooled 0.457 that is **+7 pp at 1,200 games, +6 pp at 2,400**. If the TEAMS are
treated as a sample (a team-clustered interval, X37b FINDING 3), the bar rises further, by an amount nobody has
measured on a Kakuna cell (**UNVERIFIED**). The registration of the goal read (X54) must say which interval it uses.

---

## 2. What Kakuna is (re-checked 2026-10-20)

- **The model.** `metamon:Kakuna`: 142.8M parameters, default checkpoint 34, in Metamon's local checkout `0a00a759`
  (2026-05-22).
  - Its docstring (`metamon/rl/pretrained.py`) says "The current best Metamon policy": Superkazam fine-tuned on +7.8M
    self-play battles collected at a higher sampling temperature.
  - It was trained on `metamon-parsed-replays` (human replays), `pac-base` and `pac-exploratory` (self-play)
    (`README.md:825`). It is ONE model across gen 1–4 and gen 9; the README says Kakuna "compensates for our
    inattention to Gens2-4".
  - Published gen3ou GXE against humans is ~63 %, measured with the `competitive` team set.
- **Upstream (checked 2026-10-20 against GitHub `main`'s `pretrained.py`).** Kakuna is still "the current best". The
  later single models (the V2A family, TaurosV0) are Gen1OU-focused, and none claims a better gen3ou number.
- **What its teams were.**
  - Its self-play team sets are not named anywhere (**UNVERIFIED**).
  - `competitive` is the set Metamon says it overfit to, and 19 of its 20 teams are in OUR pool.
  - In our cells Kakuna plays whatever team directory we install. "Kakuna's teams" is therefore a property of each
    cell, not of the model.
- **Its regime.**
  - Greedy and T = 1.0 read identically against X26 (Δ 0.000 [−0.087, +0.087], n = 240 each). The 2026-09-28 "T = 1.0
    is strongest" ranking was a point-estimate ordering.
  - Our side plays greedy: sampling at T = 1 cost N0 23 pp (`kakuna_goal_read_2026-09-28`).
  - Kakuna costs ≈ 75 ms per decision on CPU, 2 GB per peer, and ≈ 8.5–10 s per game with an end-state checkpoint on
    our side (SOP §2).
- 🚨 **Two things the goal's wording does not pin** (FINDING 1):
  1. **The checkpoint.** Metamon's own ensemble presets (`metamon/rl/experimental/ensemble/ensemble_presets.json`)
     rate Kakuna **ckpt 28 at GXE 0.78, above ckpt 34's 0.75**, with the code comment "these gxe scores are
     incorrect". Every one of our reads used ckpt 34. A Kakuna-28 vs Kakuna-34 pair cell settles which one is the
     strongest in OUR regime.
  2. **The ensemble.** Metamon's strongest deployed agent is a TEST-TIME ENSEMBLE: `KakunaEnsemble` / the
     `PastaMittens` router (Kakuna ckpts 28 / 30 / 34 + Superkazam). It was "the first Metamon agent to reach #1 on the
     Showdown leaderboard" (format not stated). `main.anchors` cannot play it. The goal names Kakuna, and this doc
     keeps that. Beating the ensemble is a separate, later bar, costed in §5.

---

## 3. Where we stand: every Kakuna read, in one table

All cells are greedy against greedy, Kakuna ckpt 34, on CPU, 20-game role-balanced units.

| cell | model | n | win rate [Wilson 95 %] | read |
|---|---|---:|---|---|
| home72, symmetric | X26 @161M | 400 | **0.575 [0.526, 0.623]** BETTER | `k2_x26_kakuna_both_teamsets_2026-10-17` |
| | K2 (previous lineage, plateau) | 400 | 0.487 [0.439, 0.536] | the same |
| | N0 | 200 | 0.460 [0.392, 0.529] | `kakuna_goal_read_2026-09-28` |
| home72, Kakuna at T = 1.0 | X26 @161M | 240 | 0.600 [0.537, 0.660] | `x26_metamon_anchor_2026-10-17` |
| away (`competitive`; 19 / 20 in our pool) | X26 @161M | 400 | 0.497 [0.449, 0.546] | `k2_x26_kakuna_both_teamsets_2026-10-17` |
| **H, symmetric** | X26 @161M | 400 | **0.448 [0.400, 0.496]** WORSE | `heldout72_hpfix_kakuna_2026-10-18` |
| **F, symmetric** | X26 @161M | 400 | **0.470 [0.422, 0.519]** | `x37a_eval_matrix_2026-10-18` stage 2 |
| **P72, symmetric** | X26 @161M | 400 | **0.453 [0.404, 0.501]** | the same |
| DEPLOYMENT: we pilot home72, Kakuna pilots H by stratum | X26 @161M | 600 | **L = 0.630 [0.573, 0.686]** | X37a stage 1 |
| REVERSE: we pilot H, Kakuna pilots home72 | X26 @161M | 400 | **0.380 [0.334, 0.428]** | the same |
| Kakuna pilots home72 vs Kakuna pilots H | Kakuna vs Kakuna | 400 | 0.573 [0.524, 0.620] (strength term) | the same |

**What the reads say together:**

- **Bringing its own teams, X26 beats Kakuna whatever Kakuna brings.** L is 0.630, and every stratum cell is ≥ 0.61.
- **Handed a team it never trained on, X26 loses.**
  - Strength-corrected piloting deficit: PILOT_c **+0.123 [+0.039, +0.206]**. The facing deficit is FACE_c +0.017,
    not detected.
  - At matched team strength the deficit is +0.28 [+0.12, +0.44] (TSR, Kakuna-rated teams); the set-level correction
    gives +0.11.
  - By stratum of the piloted team: CORE 0.463, NEAR 0.389, **FAR 0.263**. Unfamiliar SETS on familiar species
    already cost ≈ 11 pp; off-meta species cost more on top.
- **The drop is about teams unseen BY US.** The two third sources agree (X37a stage 2: UNSEEN-DRIVEN), and the
  population-shift hypothesis is killed.
- **Most in-house progress transfers, and the in-house meter is blind to this deficit** (X37b).
  - P\*'s edge over X26 @25M is +18.8 pp on home72 and +14.8 pp on unseen panels.
  - P\* pilots H against itself at 0.418, about what H's weakness predicts.
  - Almost all of that gain came by 100M. From 100M to 280M: +3.5 pp on home72 and +3.6 pp on unseen panels, neither
    detected.
- **PROVISIONAL throughout:** one checkpoint (161M) of one seed against Kakuna; P\* never.

---

## 4. What the deficit is, mechanically: the familiarity premium

The reads point at one mechanism: X26 plays a team well in proportion to how much it has PRACTISED that team.

- **The premium.** home72 (≈ 2.1× a pool team's piloting exposure) reads 0.575 against Kakuna. Unseen panels read
  0.45–0.47: a premium of **≈ 10–13 pp** (raw; X37a d2).
- **The dose.** These figures assume ≈ 42 of our decisions per game (X22 f3: 8,418 decisions in 200 games) and that a
  training step is one of our decisions (**ESTIMATE**).
  - By P\* every pool team was piloted ≈ **9,300 times** (280M / 42 / 719), and a home72 team ≈ 2× that.
  - D1 at 20M with a 0.5 share on both sides pilots each of its 1,982 forum teams (2,424 before the forum216b carve)
    ≈ **120 times**: ≈ 1.3 % of a pool team's dose.
  - **D1 is therefore a BREADTH lever, not a familiarity lever.** It works only if piloting skill GENERALISES from
    many briefly-seen teams to unseen ones. Nothing has measured whether it does. X52 is built to measure it (§6).
- **Why the critic matters (UNVERIFIED for X26).** `designs/learning/generalist_specialist_amortization_gap.md`
  (ai_v7 era) argues that a team-averaged objective with a tail-blind critic learns a team-agnostic "locally sensible"
  policy. Under that argument per-team win conditions are learned only by repetition, which matches the premium.
  Nobody has re-measured it on the end state.

---

## 5. The levers, with expected pp toward goal (b)

"Expected pp" is the gain in the pooled unfamiliar score vs Kakuna, over P\*. Every entry without a measurement
behind it is a GUESS.

| lever | id | mechanism vs the measured deficit | expected pp (b) | evidence | cost | rank |
|---|---|---|---|---|---|---|
| **Read P\*** | X51 | none: it measures where the plateau stands | **+0 to +3** (ESTIMATE) | X37b in-house 100M → 280M: +3.6 pp on unseen panels, n.d.; 161M is between | 0 GPU; ≈ 9 lane-h | **1** |
| **Practice ceiling** | X52 | sizes the familiarity premium and its within-population transfer | diagnostic; **practised half up to +10** (GUESS: home72's premium) | §4 | ≈ 2.5 GPU-h; ≈ 4.5 lane-h; corpus ≈ 0.25 agent-day | **2** |
| **Widen the pool (D1)** | X38 | breadth on BOTH sides: forum teams at a 0.5 share | **+2 to +6** (GUESS) | targets the 10–13 pp premium; dose ≈ 1 % per team (§4); plasticity at a plateau is a named risk (diversity doc §4.2) | ≈ 13 GPU-h (C0 shared); ≈ 10 lane-h for the readout | **3** |
| **Set mutation (D3)** | X41 | defeats exact-set identification on familiar cores, the CORE-stratum ≈ 11 pp | **+1 to +4** (GUESS) | X37a reverse CORE 0.463 vs home72 0.575 | build 1–2 agent-days (CPU, now) + ≈ 13 GPU-h | **4** (build now, run on D1's verdict) |
| **Fresh wide run** | X53 | breadth from step 0, without plateau plasticity | **+3 to +8** (GUESS) | X26 gained most of its in-house strength by 100M (X37b), so a 100M look is informative | ≈ 21 GPU-h to 100M; ≈ 60 to 280M (ESTIMATE: 1,333 steps/s) | **5** (conditional) |
| Off-meta exploiters (D4) | X42 | exploiters pilot FAR teams | +0 to +3 (GUESS) | FAR 0.263 | ≈ 18 GPU-h | conditional (diversity doc §6.1) |
| X35 plateau breakers (A1 archive, A2 PFSP, A3 exploiters) | X35 | the in-pool plateau and its cycling | **+0 to +2** (GUESS) | ≈ 79 % of in-house gain transfers (X37b), but the plateau's in-house remainder is small; A2's prior is negative | ≈ 65–91 GPU-h for batch 1 | runs anyway (owner goal 4); report U per final (§6) |
| More capacity (depth 3 → 4, kind-wise FFN experts) | X47b, X48 | capacity to condition on the team | +0 to +2 (GUESS) | the toy X47 reads first | ≈ 15–16 GPU-h each | low |
| Opponent-intent shaping | X49 | models the OPPONENT's next action | **+0 to +1** (GUESS) | targets FACING, which costs ≈ 0 (FACE_c +0.017) | ≈ 15 GPU-h | low |
| Search | — | improves decisions at play time | **≈ 0 now** (MEASURED) | anchored honest search reaches the policy, never above it (L1); the worlds bind, not the acting rule; stage 1c registered | CPU-heavy at play time | low for this goal (`search_ledger.md`) |
| An ensemble of our own checkpoints at play time | — | X26's in-run ladder cycles ≈ 3–5 pp | +0 to +2 (GUESS) | cycle monitor (ledger 2026-10-19) | ≈ 1 agent-day build; CPU reads | low |
| Test-time adaptation, deployed | — | practise the team you will bring before the game | large on THAT team (GUESS) | §4 | ≈ 0.6 GPU-h per team at 2M steps | a REGIME, never the generalist's number; X52 measures its feasibility |
| Train against Kakuna | — | fit the meter | — | — | Kakuna is ≈ 75 ms per decision on CPU | **REJECTED**: it trains on the meter, and RL against it is ≈ 2 orders of magnitude too slow |
| Sample our side at T > 0 | — | — | negative | T = 1 cost N0 23 pp; 0 < T < 1 unmeasured | — | no |

**The arithmetic.** Goal (b) needs +6–8 pp. The plausible stack is P\* (+0–3), then D1 (+2–6), then D3 or a fresh
wide run. **No single row reaches the bar on its own expected value.** X52 decides whether this family is the right
one before the GPU is spent.

**Goal (a) needs no lever.** It needs X51's reads: P\* on home72, and Kakuna's strongest checkpoint. If P\* falls
below the bar at the strongest checkpoint, specialisation (more exposure to the teams we bring: the X52 mechanism
pointed at our own deployment set) is the direct lever.

---

## 6. The plan: order, cost, decision rules

Every experiment is preregistered before its first game, as usual. Each Kakuna cell uses the anchors protocol
(`python -m main.anchors`, `designs/ops/EXTERNAL_ANCHORS_SOP.md`): CPU, one lane on rule 5b's cores, ≈ 360 games per
lane-hour, paired seeds 0–19 so that it twins with the X26 @161M cells.

### 6.1 X51 — Kakuna standing at the plateau (CPU, 0 GPU, ≈ 9 lane-h). FIRST.

**Cells:**

| cell | games |
|---|---:|
| P\* on home72, symmetric | 400 |
| P\* on F, symmetric | 400 |
| P\* on H, symmetric | 400 |
| P\* on P72, symmetric | 400 |
| P\* REVERSE: we pilot H, Kakuna pilots home72 | 400 |
| **Kakuna-28 vs Kakuna-34 pair cell** on home72 (FINDING 1) | 400 |
| T\* (the toy, no belief stack) on home72 and F | 400 each |

The toy cells subsume X37c's toy question: is the belief stack the memorisation channel? `main.anchors` loading the
toy is **UNVERIFIED**.

- **Harness.** The `63f97a57` detached checkout every X26 Kakuna read used, so that P\* − X26@161M is a model
  difference only (FINDING 4).
- **Reads:**
  - goal (a) at the plateau;
  - goal (b)'s statistic at the plateau, i.e. the gap left for the levers;
  - P\* − 161M on every panel (paired);
  - PILOT at the plateau;
  - which Kakuna checkpoint is the stronger.
- **Rules:**
  - If Kakuna-28 beats Kakuna-34 (pair cell LB > 0.50), every later goal cell uses ckpt 28, and P\*'s home72 cell is
    re-read against it (+400 games).
  - If P\* reads (b)'s LB > 0.50, goal (b) moves straight to X54's confirmation.

### 6.2 X52 — the practice ceiling (≈ 2.5 GPU-h; CPU ≈ 4.5 lane-h). SECOND.

**Question.** Does practising a team fix our piloting of it against Kakuna? Does practising 36 teams transfer to 36
unpractised teams of the same population?

**Design.**

- **The teams.** 72 tier-A teams from `w_forum_v1`, never a registered panel (the `--team-mix` guard refuses panel
  teams). They are split 36 / 36 by a registered seed, stratified by `ladder_weights_v1` stratum, and frozen as a new
  pin-safe corpus `w_practice36_v1` (≈ 0.25 agent-day, by `build_w_forum`'s machinery).
- **The fork.** P\* + `--team-mix w_practice36_v1:0.5 --team-mix-sides trainee`. Our side practises; the opponent side
  stays on the pool, because facing is not the deficit. The fork runs 5M steps, 2 seeds (≈ 1.2 GPU-h each), with
  checkpoints at 1M and 5M.
  - At 5M each practised team is piloted ≈ 1,650 times (ESTIMATE, §4's dose rule), ≈ 18 % of a pool team's dose.
  - Use the X35 production pin, so the fork's network matches C0's.
- **The read.** REVERSE-cell shape: our side pilots the half, Kakuna pilots home72.
  - P\* on each half: 200 each.
  - Each seed's 5M node on each half: 200 each.
  - Total 1,200 games. The 1M node on the practised half is descriptive: the dose curve.
- **Rules** (a team-clustered interval on fork − P\*, seeds pooled):

| reading | rule | consequence |
|---|---|---|
| **FAMILIARITY WORKS** | practised-half gain LB > 0 | exposure fixes piloting. Its size is the ceiling D1 cannot exceed at its dose |
| **TRANSFERS** | FAMILIARITY WORKS and unpractised-half gain LB > 0 | breadth generalises within the population: D1's mechanism is supported. Move D1 forward (§6.3) |
| **NO TRANSFER** | FAMILIARITY WORKS and the unpractised gain's upper bound < +3 pp | D1 at 20M is unlikely to move U. Its expected value falls to +0–2 pp; D3 and the fresh run carry the hope; X53 waits for D1's read |
| **NO FAMILIARITY** | practised-half gain upper bound < +4 pp | the deficit is not an exposure problem at this dose. Pause the D1 family; re-open the critic / representation question (§4) and capacity (X47b) |
| else | UNDECIDED | D1 proceeds as staged |

**Why it is worth 2.5 GPU-h ahead of D1.** It is the cheapest read that can kill or size the whole widening family
(D1, D3, X53: ≈ 13–90 GPU-h). It also measures test-time adaptation's feasibility as a by-product.

### 6.3 X38 / D1 — as staged, with a scheduling ask

D1's design, readout and rule are the diversity doc's §4.2 (`main.x35.d1`). Its primary H IS goal (b)'s statistic
taken as ON − OFF.

🚨 **The schedule (FINDING 3).** D1 is X35 batch 1's stage P4. It waits behind the toy block T1 (≈ 30 GPU-h, running
now: `rb_x35t_c0_s3601`, `rb_x35t_x47d4_s3601`), P1 (≈ 26), the G1 gate and P2 (≈ 8.6). That puts **D1's finals ≈
78 GPU-h (≈ 3.5 days) out**. D1 reads only C0 s3501–3503, which P1 produces.

**Ask (the orchestrator's call, not decided here):** on X52 = TRANSFERS, run D1 directly after P1. That is an
amendment to the AM-2 G1 gate's position, and it gains ≈ 9 GPU-h of calendar.

### 6.4 X41 / D3 — build now, run on D1's verdict

The build (diversity doc §4.5, an offline pre-generated augmented corpus; 1–2 agent-days) uses CPU and agent time
only, and can land while the GPU works through X35. Run D3 (≈ 13 GPU-h, one lever, its own C0) if:

- D1 reads TRADE (it gains on U but costs in-pool), or
- D1 reads GAIN with the target unreached (P\* + Δ_D1 below ≈ 0.53 pooled).

### 6.5 X53 — a fresh wide run, conditional

- **Trigger:** X52 = TRANSFERS, and D1 reads NULL or GAIN-short. NULL with transfer shown points at plasticity at the
  plateau, which the diversity doc names as D1's risk.
- **Design:** fresh `--arch production` + `--team-mix w_forum_v1@ABC:0.5 --team-mix-sides both`. It is one lever
  against X26's own recipe. Add D3 only if D3 has read GAIN.
- **A 100M look (≈ 21 GPU-h):** U vs Kakuna at 100M against X26 @100M on the same cells (X26's checkpoints exist;
  ≈ 2 × 1,200 games ≈ 7 lane-h).
- **Continue to the plateau (≈ 60 GPU-h in all)** only if the 100M look's LB > 0.

### 6.6 X54 — the goal read, on a fresh panel

The panels F, H and P72 have now steered four reads and will steer D1. A goal claimed on the panels that chose the
levers carries the selection (FINDING 2). The claim is made on:

- **a fresh human-built panel: `forum216b`** (REGISTERED and SEALED 2026-10-20, before D1 launched;
  [`forum216b_x54_2026-10-20/`](../research_state/measurements/forum216b_x54_2026-10-20/README.md)).
  - 216 teams from `w_forum_v1`'s tier AB, drawn with seed 5400 and stratified like forum72 (CORE 12 / NEAR 162 /
    FAR 42).
  - No team copies or near-copies a pool, home72, away, forum72, heldout72(_hpfix) or P72 team.
  - `w_forum_v1` was rebuilt without the panel and its near neighbours (2,424 → 1,982 teams).
  - `main.heldout_panels.SEALED` keeps any meter from playing it until X54's registration unseals it.
- plus home72 at Kakuna's strongest checkpoint;

with the interval named in the registration.

- **The size comes from X54's power** (`power.py` there):
  - Var(mean) = τ² / T + p(1 − p) / G. τ² is the per-team piloting interaction of a symmetric cell, measured on
    X37a's forum72 cell: 0 (method of moments), with a one-sided 95 % upper bound of 0.033.
  - Under that upper bound, T = 216 is the smallest size that reaches 80 % power for LB > 0.50 at p = 0.55: 0.82 at
    G = 1,600 games and 0.88 at G = 2,400. T = 144 reaches 0.79 at 2,400.
  - At p = 0.53 no size reaches 80 %: X54 is then limited by games and by its interval, not by the panel.
- **Cost:** 1,600–2,400 games on forum216b plus 400 on home72, ≈ 5.6–7.9 lane-h per model (at the 1,600-games-per-
  4.5-lane-h rate).

### 6.7 Beside the plan

- **Every X35 FINAL reports U against Kakuna** (`kakuna plan --teamsets unfamiliar`, ≈ 3.5 lane-h per arm, CPU). Each
  breaker then says what it does to goal (b), as a descriptor; no rule reads it.
- **Beating the ensemble.** `PastaMittens` is a later bar. It needs a build: the peer script would have to load an
  ensemble preset, at ≈ 3–6× Kakuna's cost per decision (ESTIMATE). Not before (b) is met.

### 6.8 The cost of the direct path

| step | GPU-h | CPU lane-h | agent-days |
|---|---:|---:|---:|
| X51 | 0 | ≈ 9 | ≈ 0.2 (prereg + queue) |
| X52 | ≈ 2.5 | ≈ 4.5 | ≈ 0.5 (corpus + prereg) |
| X38 / D1 (already in X35) | ≈ 13 (C0 shared) | ≈ 10 | 0 (built) |
| X41 / D3 build; run if triggered | 0; ≈ 13 | ≈ 10 | 1–2 |
| X53 (conditional) | ≈ 21 to the look; ≈ 60 in all | ≈ 7 | ≈ 0.3 |
| X54 | 0 | ≈ 5.6–7.9 per model | ≈ 0.3 (prereg; the carve is DONE) |
| **sure path (X51, X52, D1, X54)** | **≈ 15.5** | **≈ 29–31** | **≈ 1.0** |
| **with D3 and X53 to the plateau** | **≈ 89** | **≈ 46–48** | **≈ 2.8** |

---

## 7. What would change this plan

- **X51 reads (b)'s LB > 0.50 at P\*.** The plateau closed the gap by itself. Go straight to X54. The widening family
  becomes a margin question.
- **X51 reads goal (a) MISSED at P\*, or against Kakuna-28.** Goal (a) needs specialisation on the deployment set:
  X52's mechanism pointed at home72 (or the teams we will bring). That becomes the first GPU unit.
- **X52 reads NO FAMILIARITY.** Exposure is not the lever. D1, D3 and X53 lose their basis. Re-open the critic tail /
  representation question (§4) and capacity (X47b). Tell the owner as a strategic deviation.
- **X52 reads NO TRANSFER.** Per-team practice works and does not generalise. Goal (b) then has no measured route.
  The deployable answer becomes "practise the teams you bring" (goal (a) and L, not (b)). Escalate to the owner.
- **A newer Metamon model appears upstream.** Re-check at X54. The goal names "Metamon's best".

---

## 8. FINDINGS (standing rule 7)

1. **The goal's opponent is under-specified.**
   - Every read used Kakuna ckpt 34 (the default).
   - Metamon's ensemble presets rate ckpt 28 higher (GXE 0.78 vs 0.75, numbers the code itself calls incorrect).
   - Metamon's strongest DEPLOYED agent is an ensemble that `main.anchors` cannot play.
   - "Met" is therefore not shown against Metamon's strongest checkpoint. X51's pair cell settles the checkpoint; the
     ensemble is a later bar (§6.7).
2. **Panel reuse: there will be no fresh human-built panel after D1.**
   - F, H and P72 have steered X37a, TSR, X37b and the D1 ranking, so a goal claim on them carries the selection.
   - `w_forum_v1@ABC` takes EVERY tier-A / B forum team outside the registered panels (A 332, AB 559). Once D1
     trains on it, no unused human-built A / B team is left.
   - The only leftovers are the 1,828 tier-C teams dropped by the curator cap. They come from four curators, and
     their near-neighbour status against the corpus is unchecked.
   - **RESOLVED 2026-10-20 (the orchestrator's ruling): carved as `forum216b`** (216 teams; §6.6), REGISTERED
     and SEALED before D1 launched. D1's corpus was rebuilt to 1,982 teams (AB 559 → 305), and X35's launch sheet
     records it as AMENDMENT D1-CORPUS. Launch pre-check 2d holds a pin without it. The original wording follows.
   - **To confirm on a clean human-built panel, carve forum72b from the A / B tier and REGISTER it before D1
     launches.** That changes `w_forum_v1`'s content and sha256, so X38's validated corpus is re-validated at the
     production pin. Cost ≈ 0.5 agent-day; the orchestrator's call.
3. **D1 sits ≈ 78 GPU-h deep in X35's order**, behind levers whose expected gain on this goal is smaller (§5). Moving
   it is an amendment to the G1 gate's position, and it needs the orchestrator (§6.3).
4. **Harness drift for P\*.** P\* trained at pin `9c033489`. Every X26 Kakuna read ran from a detached `63f97a57`
   checkout. HEAD now carries FP32_ALT (config v160), which changes the τ solve.
   - X51 should run P\* at `63f97a57`, so that P\* − 161M is a model difference only.
   - Running at HEAD would mix in a play-time numerical change (small; K9(b)'s class).
5. **The dose figures assume ≈ 42 of our decisions per game, and that one training step is one of our decisions**
   (§4). They are ESTIMATES that order the levers. No rule reads them.
6. **Every expected-pp entry in §5 except search's is a GUESS.** No Rustboro-era read has measured a widening,
   augmentation or practice effect against Kakuna. X52 is the first.
7. **Goal (b)'s interval is unsettled.** A Kish / Wilson interval treats the panel's teams as fixed. A team-clustered
   interval treats them as a sample and is wider by an unmeasured amount on Kakuna cells. X37b's in-house version
   was about 2× the planning width. X54's registration must choose.
   - **Added 2026-10-20 (the forum216b sizing):** on X37a's forum72 cell, the method-of-moments team component is
     ≈ 0 (upper 95 % ≈ 0.033).
   - Yet a bootstrap that resamples the panel once for both sides, weighting a game by c_i · c_j, reads **2.02×** the
     binomial width there (P72: 1.94×). With Poisson(1) counts, E[(c_i c_j)²] = 4, so that weighting doubles the
     width even when there is no team variance.
   - X37b's "about 2×" may be this artifact rather than team variance (**UNVERIFIED**: X37b's exact resampling was not
     re-read). X54 should register a two-way interval whose noise term is not double-counted.
8. **X51's toy cells assume `main.anchors` loads T\*** (**UNVERIFIED**, as for X37c). If it does not, drop the cells;
   nothing else depends on them.
9. **home72 holds one team twice** (X37b FINDING 1). Every home72 Kakuna cell inherits a 1 / 72 double weight; it is
   immaterial at this resolution.
10. This unit played no game, used no GPU, and wrote nothing under `models/` or `data/`. It read Metamon's checkout
    and upstream's `pretrained.py` read-only.

---

## 9. Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-20 | The goal, stated as two targets (design worker, orchestrator brief) | (a) symmetric home72 at the plateau and at Kakuna's strongest checkpoint and temperature; (b) the equal-weight symmetric unfamiliar panels (H, F, P72), D1's H statistic, LB > 0.50; L and the reverse cell reported beside | (b) on L (already met at 0.630, and it carries the strength gap); (b) on the reverse cell (it carries the opponent's familiarity); one panel alone | §1; X37a; TSR |
| 2026-10-20 | Kakuna re-checked | still upstream's best single model (GitHub `main`, 2026-10-20); the ckpt 28 vs 34 question and the ensemble raised as FINDING 1 | assuming ckpt 34 is the strongest; switching the goal to the ensemble now (no harness, ≈ 3–6× the cost) | §2 |
| 2026-10-20 | The ranked path (PROPOSED; the orchestrator's GO per step) | X51 plateau read (CPU) → X52 practice ceiling (≈ 2.5 GPU-h) → X38 / D1 as staged (with a scheduling ask) → X41 / D3 built now, run on D1's verdict → X53 fresh wide run with a 100M look (conditional) → X54 confirmation on a fresh panel | D1 first with no diagnostic (≈ 13–90 GPU-h of the widening family unsized); a fresh wide run first (≈ 60 GPU-h before the cheap reads); X35 breakers as the path (+0–2 pp, GUESS); search (≈ 0, measured); X49 (targets facing, which costs ≈ 0); training against Kakuna (trains on the meter; too slow) | §4, §5 |
| 2026-10-20 | **FINDING 2 RULED and DONE: X54's confirmation panel carved before D1** (eval-registry worker, orchestrator ruling) | `forum216b`: 216 teams from `w_forum_v1` AB, seed 5400, forum72's strata × 3, curator cap 42; registered `held_out_from: ["training"]` and SEALED until X54 (`main.heldout_panels.SEALED`); no copy or near copy of the pool or any panel. `w_forum_v1` rebuilt: ABC 2,424 → 1,982 (tier sha `7bcdc835…` → `73a76e80…`); X35 AMENDMENT D1-CORPUS, launch pre-check 2d | forum72-sized (72: under the upper τ², 0.56 power at G = 2,400); 144 (0.79, fails the 80 % rule); the 1,828 capped C-tier leftovers (four curators, near-neighbour status unchecked); a successor corpus name (`w_forum_v2`: every registered panel must be excluded from any corpus anyway, and the startup guard would refuse `w_forum_v1` unrebuilt) | §6.6; `forum216b_x54_2026-10-20/` |
| 2026-10-20 | Expected pp per lever | quoted from measured effects where they exist (X37b's 100M → 280M transfer; L1's search null; FACE_c), GUESS elsewhere, with the arithmetic that (b) needs +6–8 pp | single-number promises | §1, §5 |
