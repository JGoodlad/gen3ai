# The most direct path to a reliable win over Metamon's Kakuna

**Status: X51 READ 2026-10-21 (`measurements/x51_plateau_vs_kakuna_2026-10-20/`): goal (a) MET at the plateau P\*
(home72 0.598 [0.549, 0.644]); goal (b) NOT met, 0.487 [0.459, 0.515], ≈ 4.3 pp short; Kakuna ckpt 28 not
detectably stronger than 34. X52 READ 2026-10-23: UNDECIDED (practised +3.2 pp [−0.2, +6.8], unpractised +1.3
[−3.5, +6.1]; §6.2); D1 proceeds as staged. D1 LOOK 2 (FINAL) READ 2026-10-24: UNDECIDED (H +2.10 pp
[−0.92, +5.11], D1 0.492 [0.471, 0.513]; P −1.61 [−2.90, −0.32]; §6.3); D3 / X53's trigger is the orchestrator's
call (FINDING 11).** ALWAYS-CURRENT
(owner, 2026-09-27): a later read, decision or build that differs updates this doc in the same commit, and the
Decision record (§9) says what changed and why. Every number below is quoted from a banked read with its directory;
every effect size that is not measured is marked **GUESS** or **ESTIMATE**.

Abbreviations: pp = percentage points; CI = 95 % interval; LB = the interval's lower bound; P\* = X26's plateau
checkpoint (`rb_x26_s2601` @ 280,000,170); T\* = the toy's plateau checkpoint (`rb_toy_ent_s4002_deep` @ 290,000,087);
home72 = our 72 sample teams (all in the training pool); H = `heldout72_hpfix`; F = `forum72`; P72 = 72 procedural
Smogon-usage teams; U = the three UNFAMILIAR panels H, F and P72; "pilot" = play a team, "face" = play against it.

---

## 0. One screen

- **Goal (a), our teams: MET at the plateau** (X51). P\* beats Kakuna on home72, **0.598 [0.549, 0.644]** (n = 400,
  greedy against greedy; X26 @161M read 0.575). Kakuna's checkpoint 28 is not detectably stronger than the 34 we
  read (0.515 [0.466, 0.564] against it, n = 400), so ckpt 34 stays the opponent (§2, FINDING 1).
- **Goal (b), unfamiliar teams: not met, about 4 pp away.** P\* scores 0.497 on H, 0.530 on F and 0.433 on P72, a
  pooled **0.487 [0.459, 0.515]** (161M: 0.457; +3.0 pp, n.d.). A pooled LB > 0.50 needs about **0.53** at 1,200
  games (§1).
- **More plain training did not move it (X26c, 2026-10-25).** X26 trained on from 380M to 480M at the production pin
  scores **0.4875 [0.459, 0.516]**, against P\*'s 0.4867 in the same harness: **Δ +0.08 pp [−3.90, +4.06]**, FLAT. In-house
  S did not rise either (58.22 vs 58.23 at 380M), so the 2026-10-24 ruling's D3 trigger does not fire (§9).
- **The deficit is PILOTING an unfamiliar team, and only Kakuna sees it.**
  - Against Kakuna, X26 pilots unfamiliar teams 12 pp worse after correcting for team strength (X37a PILOT_c +0.123),
    and 11–28 pp worse at matched strength (TSR).
  - Facing unfamiliar teams costs nothing (FACE_c +0.017).
  - Off-meta teams cost most: X26 pilots held-out FAR teams at 0.263.
  - In-house, P\* shows no piloting deficit beyond team strength (X37b), so **self-play meters cannot steer this
    goal**. Only Kakuna reads can, and they run on CPU at ≈ 10 s per game on one lane.
- **No lever we have measured is expected to close the remaining ≈ 4 pp alone.** Ranked by expected pp per unit of
  cost:
  1. **X51: read where P\* stands** (CPU only). DONE 2026-10-21 (above; §6.1).
  2. **X52: a PRACTICE-CEILING fork.** READ 2026-10-23: **UNDECIDED**. Practising 36 teams for 5M moved our piloting
     of them against Kakuna by +3.2 pp [−0.2, +6.8], and of 36 unpractised teams by +1.3 pp [−3.5, +6.1]. That is too
     small to call, and too large to kill exposure as a lever. D1 proceeds as staged (§6.2).
  3. **X38 / D1: widen the pool**, run in X35; FINAL (look 2) reads UNDECIDED: H +2.10 pp [−0.92, +5.11], and
     it costs in-pool strength (P −1.61 [−2.90, −0.32]); §6.3.
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

| target | cell | statistic | bar | status (P\*, X51) |
|---|---|---|---|---|
| **(a) our teams** | symmetric home72: both sides draw from home72 | Wilson on n games | LB > 0.50 at Kakuna's strongest checkpoint and temperature, at the PLATEAU | **met at the plateau**: P\* 0.598 [0.549, 0.644] vs ckpt 34; ckpt 28 not stronger (0.515 [0.466, 0.564] vs 34) |
| **(b) unfamiliar teams** | symmetric cells on H, F and P72: both sides draw from one unseen panel | the equal-weight mean of the three panel scores, Kish effective n (D1's H statistic, `main.x35.d1`) | LB > 0.50 | **not met**: 0.487 [0.459, 0.515] pooled (161M 0.457); F 0.530 [0.481, 0.578], H 0.497 [0.449, 0.546], P72 0.433 [0.385, 0.481] |
| L, the external headline (reported) | we pilot home72, Kakuna pilots H by `ladder_weights_v1` stratum | stratified, team-clustered bootstrap | — | 0.630 [0.573, 0.686] (X26 @161M; P\* unread) |

**Why (b) is the symmetric cell, and not L or the reverse cell:**

- In a symmetric cell both sides draw from the same set, so team strength cancels in expectation. TSR measured H
  ≈ 70 Elo weaker than human-built teams, and that cancels too.
- L is already above 0.50 (§3). If the owner reads the goal on L, the deployment-shaped goal is met. The brief's (b)
  is the harder, team-neutral question.
- The reverse cell (we pilot unfamiliar teams, Kakuna pilots home72) is the cleanest piloting meter. It is reported
  beside (b), never as the goal: it carries the opponent's familiarity.

**How far (b) is.** Wins needed for a Wilson LB > 0.50: 220 / 400 (0.550), 428 / 800 (0.535), 634 / 1,200 (0.528),
1,249 / 2,400 (0.520). From P\*'s pooled 0.487 that is **+4.1 pp at 1,200 games, +3.3 pp at 2,400**. If the TEAMS are
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
     `PastaMittens` router (Kakuna ckpts 28 / 30 / 34 + Alakazam ckpt 8). It was "the first Metamon agent to reach #1 on the
     Showdown leaderboard" (format not stated). `main.anchors` cannot play it. The goal names Kakuna, and this doc
     keeps that. Beating the ensemble is a separate, later bar, costed in §5.

---

## 3. Where we stand: every Kakuna read, in one table

All cells are greedy against greedy, Kakuna ckpt 34, on CPU, 20-game role-balanced units.

| cell | model | n | win rate [Wilson 95 %] | read |
|---|---|---:|---|---|
| **home72, symmetric** | **P\* (X26 @280M)** | 400 | **0.598 [0.549, 0.644]** BETTER | `x51_plateau_vs_kakuna_2026-10-20` |
| | T\* (toy) | 400 | 0.493 [0.444, 0.541] | the same |
| | X26 @161M | 400 | **0.575 [0.526, 0.623]** BETTER | `k2_x26_kakuna_both_teamsets_2026-10-17` |
| | K2 (previous lineage, plateau) | 400 | 0.487 [0.439, 0.536] | the same |
| | N0 | 200 | 0.460 [0.392, 0.529] | `kakuna_goal_read_2026-09-28` |
| home72, Kakuna at T = 1.0 | X26 @161M | 240 | 0.600 [0.537, 0.660] | `x26_metamon_anchor_2026-10-17` |
| away (`competitive`; 19 / 20 in our pool) | X26 @161M | 400 | 0.497 [0.449, 0.546] | `k2_x26_kakuna_both_teamsets_2026-10-17` |
| **H, symmetric** | **P\*** | 400 | **0.497 [0.449, 0.546]** | `x51_plateau_vs_kakuna_2026-10-20` |
| | X26c @480M | 400 | 0.455 [0.407, 0.504] | `x26c_continuation_2026-10-25` |
| | X26 @161M | 400 | 0.448 [0.400, 0.496] WORSE | `heldout72_hpfix_kakuna_2026-10-18` |
| **F, symmetric** | **P\*** | 400 | **0.530 [0.481, 0.578]** | `x51_plateau_vs_kakuna_2026-10-20` |
| | T\* (toy) | 400 | 0.458 [0.409, 0.506] | the same |
| | X26c @480M | 400 | 0.543 [0.494, 0.591] | `x26c_continuation_2026-10-25` |
| | X26 @161M | 400 | 0.470 [0.422, 0.519] | `x37a_eval_matrix_2026-10-18` stage 2 |
| **P72, symmetric** | **P\*** | 400 | **0.433 [0.385, 0.481]** WORSE | `x51_plateau_vs_kakuna_2026-10-20` |
| | X26c @480M | 400 | 0.465 [0.417, 0.514] | `x26c_continuation_2026-10-25` |
| | X26 @161M | 400 | 0.453 [0.404, 0.501] | `x37a_eval_matrix_2026-10-18` stage 2 |
| **U** (H, F, P72 equal weight; Kish n_eff ≈ 1,200) | **P\*** / X26 @161M | — | **0.487 [0.459, 0.515]** / 0.457 [0.429, 0.485] | X51 |
| **U** | X26c @480M (P\* re-played: 0.4867, identical to X51) | — | 0.4875 [0.459, 0.516] | X26c |
| Kakuna ckpt 28 vs Kakuna ckpt 34, home72 (from ckpt 28's side) | Kakuna vs Kakuna | 400 | 0.515 [0.466, 0.564] | X51 |
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
- **The plateau gained on unfamiliar teams, not detectably** (X51; paired twins, the same harness `63f97a57`).
  - P\* − X26 @161M: home72 +0.023 [−0.046, +0.090], H +0.050 [−0.019, +0.118], F +0.060 [−0.009, +0.128], P72
    −0.020 [−0.088, +0.049]; pooled U **+0.030 [−0.010, +0.070]**. All not detected.
  - The familiarity premium (home72 − U, raw) is unchanged: +11.1 pp at P\*, +11.8 pp at 161M.
  - The toy T\* is weaker than P\* on home72 (−0.105, detected) and F (−0.073, detected); its home72 → F drop is half
    P\*'s on the logit scale (0.14 vs 0.28), a difference not detected. It neither supports nor kills "the belief
    stack is the memorisation channel" (X37c's question).
- **PROVISIONAL throughout:** one seed of one run against Kakuna.

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
    many briefly-seen teams to unseen ones. X52 measured it and could not resolve it: +1.3 pp [−3.5, +6.1] (§6.2).
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
| **Read P\*** | X51 | none: it measures where the plateau stands | **MEASURED: +3.0 [−1.0, +7.0]** (n.d.) | X51: P\* 0.487 vs 161M 0.457 pooled | 0 GPU; 8 lane-h of play (18.5 h wall behind the slot queue) | **DONE** |
| **Practice ceiling** | X52 | sizes the familiarity premium and its within-population transfer | diagnostic; **READ: practised +3.2 [−0.2, +6.8], unpractised +1.3 [−3.5, +6.1], UNDECIDED** (§6.2) | §4; R0's practice curve | 2.2 GPU-h; 8.4 lane-h | **2** (DONE) |
| **Widen the pool (D1)** | X38 | breadth on BOTH sides: forum teams at a 0.5 share | **+2 to +6** (GUESS) | targets the 10–13 pp premium; dose ≈ 1 % per team (§4); plasticity at a plateau is a named risk (diversity doc §4.2) | ≈ 13 GPU-h (C0 shared); ≈ 10 lane-h for the readout | **3** |
| **Set mutation (D3)** | X41 | defeats exact-set identification on familiar cores, the CORE-stratum ≈ 11 pp | **+1 to +4** (GUESS) | X37a reverse CORE 0.463 vs home72 0.575 | build 1–2 agent-days (CPU, now) + ≈ 13 GPU-h | **4** (build now, run on D1's verdict) |
| **Fresh wide run** | X53 | breadth from step 0, without plateau plasticity | **+3 to +8** (GUESS) | X26 gained most of its in-house strength by 100M (X37b), so a 100M look is informative | ≈ 21 GPU-h to 100M; ≈ 60 to 280M (ESTIMATE: 1,333 steps/s) | **5** (conditional) |
| Off-meta exploiters (D4) | X42 | exploiters pilot FAR teams | +0 to +3 (GUESS) | FAR 0.263 | ≈ 18 GPU-h | conditional (diversity doc §6.1) |
| X35 plateau breakers (A1 archive, A2 PFSP, A3 exploiters) | X35 | the in-pool plateau and its cycling | **+0 to +2** (GUESS) | ≈ 79 % of in-house gain transfers (X37b), but the plateau's in-house remainder is small; A2's prior is negative | ≈ 65–91 GPU-h for batch 1 | runs anyway (owner goal 4); report U per final (§6) |
| More capacity (depth 3 → 4, kind-wise FFN experts) | X47b, X48 | capacity to condition on the team | +0 to +2 (GUESS) | the toy X47 reads first | ≈ 15–16 GPU-h each | low |
| **Human replays as a toggleable lever** (owner's grant 2026-10-25) | X59 → X60 | H-BC: a BC loss on ≥ 1500 human rows (the only gradient on states of teams we never trained); H-TEAMS: human ladder teams in the pool (D1's breadth mechanism on Kakuna's population) | **+1 to +4** (GUESS); the offline proxy is ABSENT (P\* predicts humans equally well on unfamiliar and pool-like teams) | data card `human_replay_quality_2026-10-25`; [`design_human_replay_lever.md`](design_human_replay_lever.md) (DRAFT) | X59 0 GPU (≈ 4.5 lane-h); X60 ≈ 16 GPU-h per arm + a build of 1–3.5 agent-days | X59 next (CPU); X60 after the reviews |
| Opponent-intent shaping | X49 | models the OPPONENT's next action | **+0 to +1** (GUESS) | targets FACING, which costs ≈ 0 (FACE_c +0.017) | ≈ 15 GPU-h | low |
| Search | — | improves decisions at play time | **≈ 0 now** (MEASURED) | anchored honest search reaches the policy, never above it (L1); the worlds bind, not the acting rule; stage 1c registered | CPU-heavy at play time | low for this goal (`search_ledger.md`) |
| An ensemble of our own checkpoints at play time | — | X26's in-run ladder cycles ≈ 3–5 pp | +0 to +2 (GUESS) | cycle monitor (ledger 2026-10-19) | ≈ 1 agent-day build; CPU reads | low |
| Test-time adaptation, deployed | — | practise the team you will bring before the game | large on THAT team (GUESS) | §4 | ≈ 0.6 GPU-h per team at 2M steps | a REGIME, never the generalist's number; X52 measures its feasibility |
| Train against Kakuna | — | fit the meter | — | — | Kakuna is ≈ 75 ms per decision on CPU | **REJECTED**: it trains on the meter, and RL against it is ≈ 2 orders of magnitude too slow |
| Sample our side at T > 0 | — | — | negative | T = 1 cost N0 23 pp; 0 < T < 1 unmeasured | — | no |

**The arithmetic.** At P\* goal (b) needs ≈ +4 pp (0.487 → ≈ 0.53; X51). The plausible stack is D1 (+2–6), then
D3 or a fresh wide run. **No single row reaches the bar on its own expected value.** X52 decides whether this family is the right
one before the GPU is spent.

**Goal (a) needs no lever.** X51 read it MET at P\* against ckpt 34, and ckpt 28 is not detectably stronger. Had it
failed, specialisation (more exposure to the teams we bring: the X52 mechanism pointed at our own deployment set)
would have been the direct lever.

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
- **READ 2026-10-21** (`measurements/x51_plateau_vs_kakuna_2026-10-20/`): every cell above at 400 games EXCEPT the
  REVERSE cell, which the unit's brief did not include (it stays open). Neither rule fired: the pair cell's LB is
  0.466, and (b)'s LB is 0.459. The numbers are in §0, §1 and §3.

### 6.2 X52 — the practice ceiling (2.2 GPU-h; CPU 8.4 lane-h). SECOND. REGISTERED 2026-10-22, READ 2026-10-23: UNDECIDED

**Read** ([`x52_practice_ceiling_2026-10-22/read/`](../research_state/measurements/x52_practice_ceiling_2026-10-22/read/README.md),
4,200 games, every unit counted, at the production pin `f4b25238`). [MEASURED · one read, two seeds]

| half | 5M seeds pooled | P\* | Δ, team-clustered 95 % | predicted (80 %) |
|---|---|---|---|---|
| practised | 0.553 (n = 1,200) | 0.522 (n = 600) | **+3.2 pp [−0.2, +6.8]** | +5 [+0.5, +9.5] |
| unpractised | 0.572 (n = 1,200) | 0.558 (n = 600) | **+1.3 pp [−3.5, +6.1]** | 0 [−3.5, +3] |

- **UNDECIDED** (the modal registered class, 0.40). It is not NO FAMILIARITY (UB 6.8 ≥ 4), and the practised LB does not
  clear 0. **D1 proceeds as staged.**
- **Descriptive.**
  - s3501 @1M − P\* on the practised half is −3.5 [−12.1, +4.6] (200 games), so the dose curve is unread.
  - C0's final − P\* on the unpractised half is −2.3 [−7.9, +3.6]: plain continuation does not drift up on unseen
    teams. On the same units, X52 − C0 is +4.0 pp (unregistered, no interval). This is a hint that practice did not
    narrow transfer.
- **What it means.** Exposure buys at most a few pp per practised team at ≈ 3,300 pilots, below R0's +6 pp at ≈ 9,500.
  D1 gives ≈ 120 pilots a team, so it can reach goal (b) only by transfer, and transfer is unresolved here
  (UB +6.1, LB −3.5). The ≈ 4 pp gap to goal (b) stays open, with no lever shown to close it.


**Question.** Does practising a team fix our piloting of it against Kakuna? Does practising 36 teams transfer to 36
unpractised teams of the same population?

**Registered** in the X35 launch sheet's §0 (AM-X52, 2026-10-22) before any argv. That note holds the predictions, the
R0 grounding, the contamination analysis and FINDINGS F-X52-1 to 6. The halves are in
[`x52_practice_ceiling_2026-10-22/`](../research_state/measurements/x52_practice_ceiling_2026-10-22/README.md).

**Design.**

- **The teams.** 72 teams of `w_forum_v1` tier A as committed at the production pin, never a registered panel's team or
  near neighbour, and ≥ 4 mon sets away from every pool team and from each other. They are split 36 / 36 inside each
  `ladder_weights_v1` stratum (CORE 1/1, NEAR 28/28, FAR 7/7; seeds 5200 / 5201).
- **The fork.** P\* + `--trainee-teams <the 36 practised files>`: our side pilots ONLY the practised half, and the
  opponent side stays on the pool, because facing is not the deficit.
  - It runs 5M steps, seeds 3501 and 3502, at the X35 production pin `f4b25238`, so the network and the recipe are
    C0's. Checkpoints come every 1M.
  - Each practised team is piloted ≈ 3,300 times by 5M (ESTIMATE, §4's dose rule), ≈ 36 % of a pool team's dose.
  - It is the kit arm `x52` (`main.x35.kit describe --arm x52`).
  - **Changed from the first draft:** the share is 1.0, not 0.5. The planned `--team-mix w_practice36_v1:0.5` needs a
    corpus in code, which a new pin would have to carry. So the dose doubles, and our side pilots no pool team for 5M.
- **The read.** REVERSE cells: our model pilots one half, and Kakuna ckpt 34 pilots home72, greedy against greedy, on
  X51's harness with the same deals for every model.
  - P\*, s3501 @5M and s3502 @5M play **600 games per half each**: 3,600 primary games.
  - Descriptive only: s3501 @1M on the practised half (200 games, the dose curve), and C0 s3501's final on the
    unpractised half (400 games, the drift a continuation alone gives).
- **The statistic.** Δ = the two seeds' pooled win rate minus P\*'s, per half. Its interval is a team-clustered
  bootstrap (95 %, ≈ ±4.9 pp). At the first draft's 200 games a cell, it was ±8.5 pp.
- **Rules,** in this order (the sheet's AM-X52-g has the exact bounds):

| reading | rule | consequence |
|---|---|---|
| **NO FAMILIARITY** | practised-half gain upper bound < +4 pp | the deficit is not an exposure problem at this dose. Pause the D1 family; re-open the critic / representation question (§4) and capacity (X47b) |
| **TRANSFERS** | practised-half gain LB > 0 and unpractised-half gain LB > 0 | breadth generalises within the population: D1's mechanism is supported. Move D1 forward (§6.3) |
| **NO TRANSFER** | practised-half LB > 0 and the unpractised gain's upper bound < +3 pp | D1 at 20M is unlikely to move U. Its expected value falls to +0–2 pp; D3 and the fresh run carry the hope; X53 waits for D1's read. Weaker evidence than it looks: our side piloted no pool team for 5M |
| **FAMILIARITY ONLY** | practised-half LB > 0, transfer unresolved | familiarity is sized. D1 proceeds as staged |
| else | UNDECIDED | D1 proceeds as staged |

**Predicted (GUESS, registered):**

- practised half +5 pp, 80 % [+0.5, +9.5];
- unpractised half 0 pp, 80 % [−3.5, +3];
- the classes: TRANSFERS 0.05 · NO TRANSFER 0.15 · FAMILIARITY ONLY 0.30 · NO FAMILIARITY 0.10 · UNDECIDED 0.40.

**How R0 grounds this.** R0's best responder practised 5 already-familiar teams against P\* and learned +13.6 pp in
20M. Per team that is +6 pp at ≈ 9,500 pilots (2M), and the gain is general within P\*'s family (+14.5 pp against
C0). X52's ≈ 3,300 pilots per team is a third of that. The guess sits above the linear +2 pp because X52's teams start
unfamiliar, where home72's premium says the early pilots matter most.

**Contamination.** The practised half is the treatment, so its gain is a familiarity ceiling and never a goal (b)
number. The unpractised half and every goal panel stay unseen by the fork. `w_forum_v1` holds all 72 teams, so after D1
these halves are unfamiliar only to P\*'s lineage.

**Why it is worth ≈ 2.2 GPU-h ahead of D1.** It is the cheapest read that can kill or size the whole widening family
(D1, D3, X53: ≈ 13–90 GPU-h). It also measures test-time adaptation's feasibility as a by-product.

### 6.3 X38 / D1 — as staged, with a scheduling ask

D1's design, readout and rule are the diversity doc's §4.2 (`main.x35.d1`). Its primary H IS goal (b)'s statistic
taken as ON − OFF.

**Look 2 (FINAL) READ 2026-10-24: UNDECIDED** (`x35_breakers_stage_2026-10-17/d1_read/`, record
`reads/d1_look2.json`). D1 s3501–s3505 at D1's pin `440b61c3` (AM-D1-FIX), C0 s3501–s3505, all K9(b) PASS.
- **H = +2.10 pp [−0.92, +5.11]** (4,200 games against Kakuna): D1 0.492 [0.471, 0.513] vs C0 0.471 [0.450, 0.492].
  Seeds +4.3 / 0.0 / +4.0 / 0.0 / +2.1 (3 of 5 up). Panels forum72 +4.0, H +2.1, P72 +0.1 (each n.d.): the gain, if
  any, is on D1's own (forum) population. Not GAIN (LB < 0), not NULL (UB > +3): UNDECIDED at the final look.
- **P = −1.61 pp [−2.90, −0.32]** on home72 in-house: a COST, wholly below 0, inferiority vs −2.0 not excluded. Guards
  not read (H non-qualifying).
- **For goal (b):** D1's own score is 0.492, LB 0.471 < 0.50. The point gain is about half the ≈ 4 pp gap, unresolved,
  and paid for with ≈ 1.6 pp in-pool. D1 alone does not reach (b). Look 1 (3 seeds) had read +2.78 [−1.12, +6.67].
- **Next:** none of §6's D1 triggers fired as written (D3 "on D1's verdict"; X53 "if NULL from plasticity or GAIN but
  short"), so whether D3 or X53 runs is the orchestrator's ruling (FINDING 11; RULED 2026-10-24: neither now — the X26 continuation fork runs instead, see the decision record).

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
- **Beating the ensemble.** `PastaMittens` is a later bar. It needs a build (two downloads, an agent entry, a
  router regime instrument) and costs ≈ 4× Kakuna per decision on CPU (a 5-member cost proxy, 0.23–0.28 s median);
  the read is preregistered in [`metamon_strongest_2026-10-25/PREREG.md`](../research_state/measurements/metamon_strongest_2026-10-25/PREREG.md).

### 6.8 The cost of the direct path

| step | GPU-h | CPU lane-h | agent-days |
|---|---:|---:|---:|
| X51 | 0 | ≈ 9 | ≈ 0.2 (prereg + queue) |
| X52 | ≈ 2.2 | ≈ 11.8 | ≈ 0.5 (halves + prereg + kit, DONE) |
| X38 / D1 (already in X35) | ≈ 13 (C0 shared) | ≈ 10 | 0 (built) |
| X41 / D3 build; run if triggered | 0; ≈ 13 | ≈ 10 | 1–2 |
| X53 (conditional) | ≈ 21 to the look; ≈ 60 in all | ≈ 7 | ≈ 0.3 |
| X54 | 0 | ≈ 5.6–7.9 per model | ≈ 0.3 (prereg; the carve is DONE) |
| **sure path (X51, X52, D1, X54)** | **≈ 15.2** | **≈ 36–39** | **≈ 1.0** |
| **with D3 and X53 to the plateau** | **≈ 88** | **≈ 53–56** | **≈ 2.8** |

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
   - **X51's pair cell (2026-10-21): ckpt 28 vs ckpt 34 reads 0.515 [0.466, 0.564] from ckpt 28's side, NOT
     DETECTED.** §6.1's switch rule does not fire and ckpt 34 stays the opponent; the cell does not exclude a ±6 pp
     difference. The ensemble is a later bar (§6.7).
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
8. **RESOLVED (X51): `main.anchors` loads T\*** at `63f97a57` (`bare`, config v158). T\* trained at pin `ae2d0901`,
   so its numbers carry that harness difference (UNVERIFIED identity); P\*'s do not.
9. **home72 holds one team twice** (X37b FINDING 1). Every home72 Kakuna cell inherits a 1 / 72 double weight; it is
   immaterial at this resolution.
10. This unit played no game, used no GPU, and wrote nothing under `models/` or `data/`. It read Metamon's checkout
    and upstream's `pretrained.py` read-only.
11. **D1's final UNDECIDED matches none of §6's written triggers.** D3 was to run "on D1's verdict" and X53 "if D1
    reads NULL from plasticity or GAIN but short". The final is neither: +2.10 pp [−0.92, +5.11] with a −1.61 pp
    in-pool cost (§6.3). Whether D3, X53 or neither runs next is an orchestrator ruling, not this doc's.

---

## 9. Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-20 | The goal, stated as two targets (design worker, orchestrator brief) | (a) symmetric home72 at the plateau and at Kakuna's strongest checkpoint and temperature; (b) the equal-weight symmetric unfamiliar panels (H, F, P72), D1's H statistic, LB > 0.50; L and the reverse cell reported beside | (b) on L (already met at 0.630, and it carries the strength gap); (b) on the reverse cell (it carries the opponent's familiarity); one panel alone | §1; X37a; TSR |
| 2026-10-20 | Kakuna re-checked | still upstream's best single model (GitHub `main`, 2026-10-20); the ckpt 28 vs 34 question and the ensemble raised as FINDING 1 | assuming ckpt 34 is the strongest; switching the goal to the ensemble now (no harness, ≈ 3–6× the cost) | §2 |
| 2026-10-20 | The ranked path (PROPOSED; the orchestrator's GO per step) | X51 plateau read (CPU) → X52 practice ceiling (≈ 2.5 GPU-h) → X38 / D1 as staged (with a scheduling ask) → X41 / D3 built now, run on D1's verdict → X53 fresh wide run with a 100M look (conditional) → X54 confirmation on a fresh panel | D1 first with no diagnostic (≈ 13–90 GPU-h of the widening family unsized); a fresh wide run first (≈ 60 GPU-h before the cheap reads); X35 breakers as the path (+0–2 pp, GUESS); search (≈ 0, measured); X49 (targets facing, which costs ≈ 0); training against Kakuna (trains on the meter; too slow) | §4, §5 |
| 2026-10-20 | **FINDING 2 RULED and DONE: X54's confirmation panel carved before D1** (eval-registry worker, orchestrator ruling) | `forum216b`: 216 teams from `w_forum_v1` AB, seed 5400, forum72's strata × 3, curator cap 42; registered `held_out_from: ["training"]` and SEALED until X54 (`main.heldout_panels.SEALED`); no copy or near copy of the pool or any panel. `w_forum_v1` rebuilt: ABC 2,424 → 1,982 (tier sha `7bcdc835…` → `73a76e80…`); X35 AMENDMENT D1-CORPUS, launch pre-check 2d | forum72-sized (72: under the upper τ², 0.56 power at G = 2,400); 144 (0.79, fails the 80 % rule); the 1,828 capped C-tier leftovers (four curators, near-neighbour status unchecked); a successor corpus name (`w_forum_v2`: every registered panel must be excluded from any corpus anyway, and the startup guard would refuse `w_forum_v1` unrebuilt) | §6.6; `forum216b_x54_2026-10-20/` |
| 2026-10-21 | **X51 READ** (anchor worker; CPU, 2,800 games at the `63f97a57` harness; the stop lifted by the orchestrator so every cell reached 400) | goal (a) MET at P\* (0.598 [0.549, 0.644]); goal (b) NOT met (0.487 [0.459, 0.515], ≈ 4.3 pp short; P\* − 161M +3.0 pp n.d.); ckpt 34 stays the opponent (ckpt 28 0.515 [0.466, 0.564] vs it, n.d.); T\* weaker than P\*. §0, §1, §3, §5 and FINDINGS 1 and 8 now quote it; the path's order is unchanged (X52 next) | switching the goal to ckpt 28 (its rule did not fire); reading (b) as within reach of the plateau alone (+3 pp, n.d.) | `x51_plateau_vs_kakuna_2026-10-20/` |
| 2026-10-22 | **X52 REGISTERED and STAGED** (design worker; CPU only) | 72 forum tier-A teams drawn and split 36 / 36 (`x52_practice_ceiling_2026-10-22/`); the fork is C0's argv + `--trainee-teams <practised 36>` at the production pin `f4b25238`, +5M, s3501–s3502, ≈ 2.2 GPU-h, after G1 and before D1; the read is 600 games per (model, half) against Kakuna; predictions practised +5, unpractised 0 (X35 sheet §0 AM-X52) | the first draft's `--team-mix w_practice36_v1:0.5` (its corpus is code, so it needs a new pin, not C0's); 200 games a cell (±8.5 pp, wider than the predicted effect); practising on goal (b)'s own panels (that trains on the meter) | §6.2; X35 sheet §0 |
| 2026-10-23 | **X52 READ: UNDECIDED** (anchor worker; CPU, 4,200 games, reverse cells at the production pin `f4b25238`) | practised Δ +3.2 pp [−0.2, +6.8], unpractised +1.3 [−3.5, +6.1] (pooled seeds − P\*, team-clustered); no rule held, so D1 proceeds as staged; §0 and §6.2 quote it | a third seed or more games to resolve it (not registered; a new registration if wanted); reading +3.2 as familiarity shown (its LB is below 0) | `x52_practice_ceiling_2026-10-22/read/` |
| 2026-10-23 | **D1 LOOK 1 READ: CONTINUE** (readout worker; CPU, 2,520 Kakuna games + 21,000 in-house pairs) | H +2.78 pp [−1.12, +6.67], P −1.30 [−4.54, +1.93]; guards not read; look 2 (s3504–s3505, both arms) decides; §0 status and §6.3 quote it; §6.3's schedule ask is retired (D1 ran) | reading +2.78 as a breadth gain (its LB is below 0); skipping look 2 for D3 / X53 | `x35_breakers_stage_2026-10-17/d1_read/` |
| 2026-10-24 | **D1 LOOK 2 (FINAL) READ: UNDECIDED** (readout worker; CPU, 4,200 Kakuna games + 35,000 in-house pairs; look 1's rows reused) | H +2.10 pp [−0.92, +5.11] (D1 0.492, LB 0.471), P −1.61 [−2.90, −0.32]; guards not read; goal (b) not met by D1; §0, §6.3 quote it; D3 / X53's trigger left to the orchestrator (FINDING 11) | reading +2.10 as a breadth gain (LB < 0); calling the final UNDECIDED a NULL (its UB is +5.1) | `x35_breakers_stage_2026-10-17/d1_read/` |
| 2026-10-24 | **ORCHESTRATOR RULING on FINDING 11: neither D3 nor X53 now** | D1's final is UNDECIDED, and its gain sits mostly on its own forum population (+4.0 pp) with no transfer shown to unseen populations (heldout +2.1, p72 +0.1), at a resolved in-pool cost (P −1.6 pp). D3 targets exact-set identification on familiar cores, which this read does not implicate, and X53 (≈ 60 GPU-h) needs X52 = TRANSFERS, which X52 did not read. The GPU instead runs the one lever with consistently positive measured value: more training. That is `rb_x26c_s2601`, a continuation fork of X26 @380M to 480M at the production pin (≈ 20 GPU-h), re-read against Kakuna after. D3's build stays queued and can be revisited if the continuation's re-read shows the unfamiliar gap persisting with in-house strength rising | the X26 drift read (dbc49cc1, +2.05 pp / 100M); X51 (161M 0.457 → 280M 0.487 unfamiliar) |
| 2026-10-25 | **X26c READ: goal (b) NOT met; FLAT** (readout worker; CPU, 2,400 Kakuna games + 10,500 in-house mirrored pairs; preregistered `e05c45eeb`) | the continuation to 480M scores U 0.4875 [0.459, 0.516] vs P\* 0.4867 (same harness, seeds 0–19), Δ +0.08 pp [−3.90, +4.06]; S1 58.22 vs 58.23 at 380M (−0.01 [−1.54, +1.52]): in-house strength did NOT rise, so the ruling's D3 trigger does not fire; §0 quotes it; the next lever is the orchestrator's | reading the FLAT as "training has stopped helping" outright (Δ's UB is +4.1 pp); revisiting D3 on the gap alone (the ruling tied it to S rising) | `x26c_continuation_2026-10-25/` |
| 2026-10-25 | **Human replays enter the lever table** (design worker, the owner's grant: human replays may train the policy, toggleable, off by default, after a data-quality analysis) | the data card on `hr_spec_v1` (1.80M informative ≥ 1500 rows; ≥ 15.9 % of ≥ 1500 human sides in the panels' region; no familiarity gradient in P\*'s human-prediction NLL); ranked H-BC > H-TEAMS > a human-only comparator HB0 > H-KL > a BC warm start; **X59 (the HOM vs Kakuna on U and home72, CPU, no build) first**, then one lever in X60 in D1's frame; the always-current architecture `human_data_architecture.md` (DRAFT until two adversarial reviews) | building H-BC before X59 (the offline proxy for its mechanism is absent); H-KL (acts on self-play states, where we are strong); a fresh warm-start run (≥ 42 GPU-h before a comparable read); human outcomes as critic labels (off-policy) | `measurements/human_replay_quality_2026-10-25/`; [`design_human_replay_lever.md`](design_human_replay_lever.md) |
| 2026-10-20 | Expected pp per lever | quoted from measured effects where they exist (X37b's 100M → 280M transfer; L1's search null; FACE_c), GUESS elsewhere, with the arithmetic that (b) needs +6–8 pp | single-number promises | §1, §5 |
