# The most direct path to a reliable win over Metamon's Kakuna

**Status (2026-10-26): goal (a) MET at the plateau P\* (home72 0.598 [0.549, 0.644], X51); goal (b) NOT met, U 0.487
[0.459, 0.515], ≈ 4 pp short, on panels that stand (2026-10-26 validity read). Read since, none detected: X52
UNDECIDED (§6.2); D1 FINAL UNDECIDED, U +2.10 pp [−0.92, +5.11] at an in-pool cost (§6.3); X26c, 100M more training,
FLAT (+0.08 pp [−3.90, +4.06]); D3 FINAL UNDECIDED (look 2, every game on one engine), U +2.33 pp [−0.69, +5.35], in-pool non-inferior (§6.4); the belief prior
mix NOT DETECTED at λ = 0.25 and λ = 1 (§5); Metamon's ensemble no stronger than Kakuna (§6.7). X61P / N1-P (12,000
procedural teams, a fork of P\* against the shared C0) look 1 READ 2026-10-28: CONTINUE, F +0.75 pp [−1.56, +3.06] at a
resolved in-pool cost (P −3.27 pp [−4.20, −2.33]), look 2 licensed (§6.5b); H-CTX
SUPPORTED: the home premium does not go with the sets, which keep at most about a fifth of it (§4, 2026-10-27), and under a neutral pilot rc72b is only about 4 pp weaker (A-K, 2026-10-28); X61's toy
design is NOT-READY and not run. X53-N1 and the all-null fallback are REVISED DRAFTs after their review (`6fca2c06`): X53-N1
gated on a zero-GPU seed-spread read, the fallback opening with PR-1, a zero-GPU pooled precision read (§6.5, §7).** ALWAYS-CURRENT
(owner, 2026-09-27): a later read, decision or build that differs updates this doc in the same commit, and the
Decision record (§9) says what changed and why. Every number below is quoted from a banked read with its directory;
every effect size that is not measured is marked **GUESS** or **ESTIMATE**.

Abbreviations: pp = percentage points; CI = 95 % interval; LB = the interval's lower bound; P\* = X26's plateau
checkpoint (`rb_x26_s2601` @ 280,000,170, the named baseline `pstar_plateau`); T\* = the toy's plateau checkpoint (`rb_toy_ent_s4002_deep` @ 290,000,087);
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
- **The panels stand as goal (b)'s meter (validity read, 2026-10-26).**
  - No panel is shown Kakuna-familiar. H came from Metamon's `hl_05_26`, but Kakuna can have seen an H team at most
    as its one source replay (≲ 2 trajectories). As a population, H is no more ladder-typical than F, and home72 is
    the panel closest to Kakuna's data.
  - Dropping H LOWERS P\*'s score: 0.481 [0.447, 0.516] on F + P72.
  - X51, X26c and D1 look 2 keep their verdicts on every subset of the panels.
  - **OPEN QUESTION:** whether Kakuna's self-play saw near copies of H teams (A-K at the exact-team level). Settling
    it needs downloads that were not made.
  - Read: [`goal_b_panel_validity_2026-10-26/`](../research_state/measurements/goal_b_panel_validity_2026-10-26/README.md).
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
  4. **X41 / D3: set mutation.** FINAL (look 2) READ 2026-10-28: **UNDECIDED** (§6.4). 2,876 pool teams with 1–2
     mons' sets redrawn from Smogon usage, mixed in at 0.3 per side, read by D1's rule against the shared C0. H
     +2.33 pp [−0.69, +5.35] (D3 0.494, LB 0.473; 4 of 5 seeds positive), +1.64 [−2.06, +5.34] without H; P −0.45 pp
     [−1.16, +0.25], NON-INFERIOR. Every game ran on ONE engine (E1) at a dedicated play pin. Look 2 is terminal.
  5. **X61P / N1-P: breadth at 10⁴ teams, at the plateau.** Look 1 READ 2026-10-28: **CONTINUE** (§6.5b). F (the
     primary) +0.75 pp [−1.56, +3.06] (N1-P 0.516 vs C0 0.508, 3,600 games per arm; 2 of 3 seeds positive); P
     **−3.27 pp [−4.20, −2.33]**, INFERIORITY-NOT-EXCLUDED, twice D1's in-pool cost; reported U +1.76 [−1.52, +5.03]
     (M4-hazarded), without H +1.80 [−2.17, +5.77], Δ_H − Δ_F +0.92 (no sign of learning the generator). Look 2
     (s3504–s3505) decides; on an F GAIN the P cost makes it a TRADE unless P recovers. 12,000
     procedural teams drawn from Smogon usage alone, 0.5 of each side's draws, forked from P\* at C0's pin + the corpus,
     read by D1's rule on F alone against the shared C0 (≈ 12.3 GPU-h at look 1, ≈ 20.5 with look 2). GO (F GAIN)
     funds X53-N1, a from-scratch production run with this corpus; NULL rules out only "N1 at the plateau". The
     literature survey (`sota_survey_2026-10-26.md` §5) ranks N1 first: every system that generalises across teams
     trained on 10⁴–10⁵ of them. X61's toy design was reviewed NOT-READY and is not run as written.
  6. **X53: a fresh wide run**, now conditional on N1-P's GAIN, as X53-N1 (this corpus; ≈ 39 GPU-h to a one-seed
     screen). Its design is a REVISED DRAFT, gated on a zero-GPU seed-spread read (§6.5).
  7. **X54: the goal's confirmation read** on a FRESH panel (FINDING 2): **`forum216b`**, 216 human-built teams,
     carved out of D1's corpus and REGISTERED and SEALED on 2026-10-20, before D1 launched.
- **The belief channel moves no Kakuna games detectably.** P\*'s belief deltas memorise the pool (confidently wrong
  off-pool, on calibration), yet mixing them toward the Smogon prior reads U +1.5 pp [−2.7, +5.7] at λ = 0.25 and
  −1.5 pp [−5.5, +2.5] at λ = 1 (prior only), both NOT DETECTED (§5).
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
| **(b) unfamiliar teams** | symmetric cells on H, F and P72: both sides draw from one unseen panel | the equal-weight mean of the three panel scores, Kish effective n (D1's H statistic, `main.x35.d1`) | LB > 0.50 | **not met**: 0.487 [0.459, 0.515] pooled (161M 0.457); F 0.530 [0.481, 0.578], H 0.497 [0.449, 0.546], P72 0.433 [0.385, 0.481]; without H 0.481 [0.447, 0.516] (the panels stand, 2026-10-26) |
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
  - Exposure to our panels (2026-10-26 validity read): at most one source replay per H team; no panel is shown familiar; home72 is the closest to Kakuna's data.
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
     Showdown leaderboard" (format not stated). **Read 2026-10-26: it is NOT detectably stronger than Kakuna for us**
     (P\* scores 0.484 against it on U and 0.487 against Kakuna; Δ-U −0.25 pp [−4.23, +3.73]; §6.7). The goal keeps
     Kakuna ckpt 34.

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
| **U**, 20M forks of P\* against C0 (P\* + 20M plain) | D1 final (5 seeds) / its C0 | — | 0.492 [0.471, 0.513] / 0.471 | `x35_breakers_stage_2026-10-17/d1_read` |
| **U** | D3 look 1 (3 seeds) / its C0 | — | 0.491 [0.464, 0.519] / 0.479 [0.451, 0.506] | `x41_d3_setmut_2026-10-25/read` |
| **U** | P\* with the belief prior mix, λ = 0.25 / λ = 1 | 1,200 each | 0.502 [0.473, 0.530] / 0.472 [0.444, 0.500] | `belief_prior_mix_2026-10-26` |
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
- **No lever read since moves U detectably** (§5): D1 +2.1, D3 +1.3 (look 1), the belief prior mix +1.5 / −1.5,
  X26c +0.1 pp.
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
- **Where it is lost (2026-10-26, DESCRIPTIVE, `measurements/piloting_gap_forensics_2026-10-26/`).** In X51's games
  the premium is not lost in the opening: the material state at turn 10 carries none of it. It is lost converting
  level mid-game positions (0.609 vs 0.501 from level at turn 10). The per-mon KO evidence could not say whether it
  goes with the team or the set (review `measurements/h_mid_value_test_2026-10-26/REVIEW.md` B1).
- **It does not go with the sets (H-CTX SUPPORTED, registered, 2026-10-27,
  `measurements/hctx_rc72_2026-10-26/`).** P\* vs Kakuna on rc72b, 72 new sixes made only of verbatim pool sets (≤ 2
  shared with any pool or corpus team), beside home72 and F on one engine: home72 0.615 [0.581, 0.648], F 0.540
  [0.505, 0.574], rc72b 0.514 [0.486, 0.542]. rc72b recovers −0.34 [−2.05, +0.20] of the home premium (KILL needed ≥
  0.75); rc72b − home72 −10.1 pp [−14.2, −6.0]. Familiar sets in new combinations play like unfamiliar teams.
  - **The bound:** the sets keep at most about a fifth of the premium (a quarter against X51's U, across engines).
    Set-level breadth (D3) alone works that smaller channel.
  - **Not measured:** WHY the six matters. rc72b's sixes are less coherent than home72's and F's (Smogon teammate
    lift ≈ random; pool pair PMI −0.22 against F's +0.05). Memorised sixes, core (pair) knowledge that human teams
    keep, and a plan-less recombined six all fit (review `measurements/hctx_rc72_2026-10-26/REVIEW.md` M1, M2).
  - **The levers:** they are recombined teams (X62, whose first gate this was), procedural teams (N1-P) and their
    union (X63). X62's corpus breaks cores as rc72b does, so only F reads it.
  - **A-K, measured (2026-10-28, `measurements/hctx_kk_2026-10-27/`).** With Kakuna on both sides, rc72b scores
    0.459 [0.428, 0.493] against home72 + F teams: UNDECIDED, detected weaker by about 4 pp (−28 Elo).
    - It is flat across coherence thirds, so the weakness is not what the coherence metrics track.
    - POST HOC, across engines: a full transfer of that weakness into P\*'s symmetric score would make the recovered
      share 0.21 [−0.88, +0.88]. Under first-order cancellation the registered −0.34 stands.
    - Either way the premium is mostly team-level. "No plan" explains at most the ~4 pp part.
    - (a) memorised sixes against (b) cores needs the rc72c core-preserving panel (predictions in that PREREG §2).
    - For X62: its ≤ 2 corpus is about 28 Elo weaker as teams, small next to P72's −128.

---

## 5. The levers, with expected pp toward goal (b)

"Expected pp" is the gain in the pooled unfamiliar score vs Kakuna, over P\*. Every entry without a measurement
behind it is a GUESS.

| lever | id | mechanism vs the measured deficit | expected pp (b) | evidence | cost | rank |
|---|---|---|---|---|---|---|
| **Read P\*** | X51 | none: it measures where the plateau stands | **MEASURED: +3.0 [−1.0, +7.0]** (n.d.) | X51: P\* 0.487 vs 161M 0.457 pooled | 0 GPU; 8 lane-h of play (18.5 h wall behind the slot queue) | **DONE** |
| **Practice ceiling** | X52 | sizes the familiarity premium and its within-population transfer | diagnostic; **READ: practised +3.2 [−0.2, +6.8], unpractised +1.3 [−3.5, +6.1], UNDECIDED** (§6.2) | §4; R0's practice curve | 2.2 GPU-h; 8.4 lane-h | **2** (DONE) |
| **Widen the pool (D1)** | X38 | breadth on BOTH sides: forum teams at a 0.5 share | **MEASURED (final): +2.1 [−0.9, +5.1], UNDECIDED**; in-pool −1.6 [−2.9, −0.3] | §6.3; what gain there is sits on its own forum population | ≈ 13 GPU-h (C0 shared); ≈ 10 lane-h for the readout | **DONE** (UNDECIDED) |
| **Set mutation (D3)** | X41 | defeats exact-set identification on familiar cores, the CORE-stratum ≈ 11 pp | **MEASURED (final): +2.3 [−0.7, +5.3], UNDECIDED**; in-pool −0.5, non-inferior | §6.4; X37a reverse CORE 0.463 vs home72 0.575 | 20.68 GPU-h (MEASURED; C0 shared) | **DONE** (UNDECIDED, terminal) |
| **Breadth at 10⁴ (N1), at the plateau** | X61P (N1-P; the X61 toy NOT-READY) | Procgen-scale breadth: 12,000 procedural Smogon-usage teams at 0.5 per side, a fork of P\*; the field's common factor (Metamon 11k + 107k teams, random-battle bots) | **+3 to +8** on production from step 0 (survey GUESS); N1-P's GUESS **+1.5 [−1.5, +4.5]** on F at the plateau | Procgen's gap closes at ≈ 10⁴ levels; VGC-Bench's monotone team-count gain; against: D1 +2.1 at 10³, procedural teams rate −146 Elo, plasticity at the plateau | TRAINED 2026-10-26 (s3501–s3503); ≈ 12.3 GPU-h at look 1, + 8.2 at look 2 | **look 1 CONTINUE** (2026-10-28): F +0.75 [−1.56, +3.06], P −3.27 [−4.20, −2.33] |
| **Fresh wide run** | X53 | breadth from step 0, without plateau plasticity | **+3 to +8** (GUESS) | X26 gained most of its in-house strength by 100M (X37b), so a 100M look is informative | ≈ 21 GPU-h to 100M; ≈ 60 to 280M (ESTIMATE: 1,333 steps/s); as X53-N1 ≈ 39 to its screen | conditional on N1-P's GAIN (§6.5) |
| Off-meta exploiters (D4) | X42 | exploiters pilot FAR teams | +0 to +3 (GUESS) | FAR 0.263 | ≈ 18 GPU-h | conditional (diversity doc §6.1) |
| X35 plateau breakers (A1 archive, A2 PFSP, A3 exploiters) | X35 | the in-pool plateau and its cycling | **+0 to +2** (GUESS) | ≈ 79 % of in-house gain transfers (X37b), but the plateau's in-house remainder is small; A2's prior is negative | ≈ 65–91 GPU-h for batch 1 | A1 NULL and SPRT NULL (in-house S, FINAL); LR-F (a learning-rate floor) UNDECIDED, FINAL: −1.1 pp [−2.4, +0.2] in-house S, no Kakuna read (owner goal 4) |
| More capacity (depth 3 → 4, kind-wise FFN experts, FFN width) | X47b, X48, N4 | capacity to condition on the team | +0 to +2 (GUESS) | the toy X47 read UNDECIDED, Δ̂ ≤ 0, FINAL (−0.88 pp [−3.20, +1.44], 2026-10-21); **X48 (per-kind FFN experts at P\*) READ NO-GO, weak ("experts not engaged")**: in-house −0.55 pp [−1.32, +0.21], forum72 vs Kakuna +2.06 [−0.06, +4.19] UNDECIDED (1,000 games a final; post-H22 pack, never beside a legacy-pack F) | ≈ 13 GPU-h (X48, MEASURED) | low; N4 (width) next, the orchestrator's choice |
| **Human replays as a toggleable lever** (owner's grant 2026-10-25) | FR-1 / FR-3 → X59-S → X60 | first the reads: P\* vs Metamon's human-only models (FR-1 / FR-3 READ: P\* beats all three, U 0.666 / 0.631 / 0.790) and X59-S, three matched student arms (human states + labels; the same states + P\*'s labels; P\*'s self-play states + labels) that answer "is human data better than our self-play data?"; then H-TEAMS (human ladder teams in the pool) or H-BC (a BC loss on human rows) | **+0 to +4** (GUESS); no offline evidence either way (the familiarity proxy is uninformative) | data card `human_replay_quality_2026-10-25`; [`design_human_replay_lever.md`](design_human_replay_lever.md); X59-S's PREREG | X59-S 8.16 GPU-h of step time (MEASURED, its Amendment 3; ≈ 3–4.5 h wall at 3 at once) + ≈ 60 lane-h (ESTIMATE); X60 NOT-READY (its estimator and power disagree) | X59-S students training on the GPU |
| Plain continuation (X26 380M → 480M) | X26c | more of the same training | **MEASURED: +0.1 [−3.9, +4.1], FLAT**; in-house S −0.01 | ledger 2026-10-25 · *X26c* | ≈ 20 GPU-h | **DONE** |
| Belief anchoring (the prior mix, inference only) | H3 | pulls the memorised belief deltas toward the Smogon prior | **MEASURED: λ 0.25 +1.5 [−2.7, +5.7]; λ 1 (prior only) −1.5 [−5.5, +2.5]**, both n.d. | ledger 2026-10-26 · *THE BELIEF PRIOR MIX*; H3 holds on calibration only | 0 GPU | **DONE**: no game-level case |
| Opponent-intent shaping | X49 | models the OPPONENT's next action | **+0 to +1** (GUESS) | targets FACING, which costs ≈ 0 (FACE_c +0.017) | ≈ 15 GPU-h | low |
| Search | — | improves decisions at play time | **≈ 0 now** (MEASURED) | anchored honest search reaches the policy, never above it (L1); the worlds bind, not the acting rule; stage 1c READ: nearer, not paying | CPU-heavy at play time | low for this goal (`search_ledger.md`) |
| An ensemble of our own checkpoints at play time | — | X26's in-run ladder cycles ≈ 3–5 pp | +0 to +2 (GUESS) | cycle monitor (ledger 2026-10-19) | ≈ 1 agent-day build; CPU reads | low |
| Test-time adaptation, deployed | — | practise the team you will bring before the game | large on THAT team (GUESS) | §4 | ≈ 0.6 GPU-h per team at 2M steps | a REGIME, never the generalist's number; X52 measures its feasibility |
| Train against Kakuna | — | fit the meter | — | — | Kakuna is ≈ 75 ms per decision on CPU | **REJECTED**: it trains on the meter, and RL against it is ≈ 2 orders of magnitude too slow |
| Sample our side at T > 0 | — | — | negative | T = 1 cost N0 23 pp; 0 < T < 1 unmeasured | — | no |

**The arithmetic.** At P\* goal (b) needs ≈ +4 pp (0.487 → ≈ 0.53; X51). Every measured lever reads below that, and
none is detected: D1 +2.1 (final), D3 +2.3 (final), N1-P +0.75 on F (look 1), the belief prior mix +1.5 / −1.5,
plain training +0.1 (X26c). **No single row reaches the bar on its own.** The open routes are N1-P look 2 and X59-S
(D3 closed UNDECIDED, 2026-10-28).

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

### 6.3 X38 / D1 — FINAL READ 2026-10-24: UNDECIDED

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
  short"), so whether D3 or X53 runs was the orchestrator's ruling (FINDING 11; RULED 2026-10-24: neither then). X26c ran instead and read FLAT (2026-10-25); D3 then ran on a fresh GO (§6.4).

### 6.4 X41 / D3 — REGISTERED 2026-10-25; look 1 READ 2026-10-26: CONTINUE; look 2 (FINAL) READ 2026-10-28: UNDECIDED

**Look 1 (s3501–s3503, 12.54 GPU-h; [`read/README.md`](../research_state/measurements/x41_d3_setmut_2026-10-25/read/README.md)):**

- **H, UNDECIDED → CONTINUE:** +1.27 pp [−2.63, +5.16] (D3 0.491 vs C0 0.479, 420 games per panel per arm; φ̂ 0.60,
  so not widened). Per seed +2.4 / −2.9 / +4.3 pp. Per panel H +2.9, P72 +1.0, F +0.0, none detected.
- **Without H** (F + P72): +0.48 pp [−4.29, +5.24].
- **P, NON-INFERIOR:** −0.24 pp [−1.10, +0.62]. Unlike D1 (−1.61 pp), the set mutation costs no detectable in-pool
  strength.
- **Guards:** not read (H is not a GAIN).
- **C0's banked rows:** reused after the game-for-game replay guard passed 7 of 7 units. **Caveat (engine mix):** those
  rows were played before the 10-25 obs / label and engine fixes (`d8d674b99`, `f9978586b`), D3's at `2cc864f2f`,
  which carries them; the guard re-plays one cell (140 games of one seed on one panel), so the mix is bounded, not
  excluded ([REVIEW F](../research_state/measurements/human_replay_quality_2026-10-25/REVIEW_F_x60.md) F-M2).
- **The registered guess:** H +1.5 pp; read +1.27.

**Look 2, FINAL (s3501–s3505, every game on ONE engine; [`read/README_look2.md`](../research_state/measurements/x41_d3_setmut_2026-10-25/read/README_look2.md)):**

- **H, UNDECIDED (final):** +2.33 pp [−0.69, +5.35] (D3 0.494 vs C0 0.471, 700 games per panel per arm; φ̂ 0.60).
  Per seed +2.1 / −3.1 / +4.3 / +2.1 / +6.2. Per panel H +3.7, P72 +2.1, F +1.1, none detected. Without H: +1.64
  [−2.06, +5.34].
- **P, NON-INFERIOR:** −0.45 pp [−1.16, +0.25].
- **The engine mix is gone.** D3 s3501–s3505 were played at a play pin carrying E1's game code (PREREG AM-D3-L2),
  and C0's E1 rows imported. An identity check replayed 80 / 80 banked games choice by choice, and the guard 140 / 140.
  Look 1 recomputed on E1 reads +1.11, still CONTINUE.
- **Terminal.** NULL was unreachable at this size (the pre-data review M1). The interval is the bound: a loss is
  excluded, and a gain up to ≈ 5 pp is not. No further D3 seeds are pooled.
- **Guards:** not read (H is not a GAIN).


**The written triggers did not fire.** D3 was to run if D1 read TRADE, or GAIN with the target unreached. D1's final
was UNDECIDED (FINDING 11), and X26c's revisit rule ("FLAT with S rising") did not fire either. The build was the
orchestrator's standing queue item, so the GO is a fresh ruling.

**What was built** (CPU only; [`x41_d3_setmut_2026-10-25/PREREG.md`](../research_state/measurements/x41_d3_setmut_2026-10-25/PREREG.md)):

- **The lever:** `--team-mix d3_setmut_v1@ALL:0.3 --team-mix-sides both` (kit arm `d3`, C0's argv token for token plus
  this one lever).
- **The corpus** (`utils.team_corpora.build_d3_setmut`, deterministic, an offline pre-generated corpus as §4.5 of the
  diversity doc recommends): 2,876 teams, i.e. 4 variants of each of the 719 pool teams.
  - Each variant keeps its source's six species, the abilities and the untouched mons byte for byte.
  - It redraws k ∈ {1, 2} mons from Smogon usage only: 4 moves by P(move in set), the item by usage, and nature + EVs
    as one usage spread. Hidden Power is typed by `gen3_hp_repair_v1`.
  - Every redrawn set is a new mon set, so a variant shares exactly 6 − k sets with its source.
  - Every variant passes the pinned validator, and none is near a held-out panel.
- **The soak:** PASS, 10 rounds, 28,760 battles (every team 10 times in each seat), 951,264 p1 decisions, 0 core errors, 32 stall forfeits, 121.9 s; every one of the 2,876 teams packed by the training teambuilder with 0 drops (label path included).
- **Its own pin** D3_PIN `9e289ba44eb0015f6fa9c7815fc2e94f296d8f4f` = the production pin `f4b25238` + one commit under `src/utils/team_corpora/`
  (D1's AM-D1-FIX pattern). It carries the pinned-data root.
- **The read:** D1's rule verbatim (H primary, P co-primary, G-F / G-E / G-C, look 2 on CONTINUE). C0 s3501–s3505 are
  shared, and their banked rows are reused after a deterministic replay check.
- **Registered prediction (GUESS):** H +1.5 pp, 80 % [−1.5, +4.5]; P −0.8 [−2.5, +0.9]; the final class UNDECIDED with
  probability 0.67.

### 6.5b X61P / N1-P — breadth at 10⁴ teams, at the plateau: look 1 READ 2026-10-28: CONTINUE (the X61 toy: NOT-READY)

**Why.** The literature survey (`../research_state/literature/sota_survey_2026-10-26.md` §5, N1) ranks breadth first:
every system that generalises across teams trained on 10⁴–10⁵ of them, and every breadth test here (D1, D3, X52) ran
at 10³ teams.

**The toy design (X61, `de269480`) was reviewed NOT-READY** ([`x61_n1_breadth_2026-10-25/REVIEW.md`](../research_state/measurements/x61_n1_breadth_2026-10-25/REVIEW.md)):

- **B1:** H is generator-adjacent. 62 % of its full sets sit verbatim in the corpus (the pool holds 10 %), so its
  primary U_HF carried the contamination it excluded P72 for.
- **B2:** the toy's whole familiarity premium (≈ 3.5 pp) sits below its GO bar, so its KILL / BELOW-GO classes would
  have closed production on the evidence the hypothesis predicts.
- **M1:** under a null it ends UNDECIDED 73–79 % of the time.

**What runs instead: N1-P** ([`x61p_n1_production_2026-10-26/PREREG.md`](../research_state/measurements/x61p_n1_production_2026-10-26/PREREG.md)):

- **The corpus `n1_proc_v1`** (`utils.team_corpora.build_n1_proc`, X61's build, unchanged): 12,000 whole teams from the
  Smogon facade alone, the pinned validator, level 100, no team within 2 mon sets of a held-out panel team; the
  label-path soak PASS (120,000 battles). Not human data.
- **The arm:** kit arm `n1p` = C0's argv + `--team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both` (D1's share), forked
  from P\* for +20M, s3501–s3503, look 2 s3504–s3505 on CONTINUE; ≈ 12.3 GPU-h at look 1, ≈ 20.5 with look 2.
- **The pin:** `batch.N1P_PIN` `5f8f6cfa` (tag `x61p-n1p-pin-5f8f6cfa`) = C0's `f4b25238` + one commit adding only the
  corpus's files and its registry line. It deliberately leaves out the 10-26 GIGO fixes (`d8d674b9`): C0 ran without
  them, so they would be a second lever.
- **The read** (`main.reads` specs `read_spec_look<k>.json`, rule `d1`, label `X61P`): the PRIMARY is Δ_F against the
  shared C0, Kakuna on F alone at 1,200 games per arm per seed, D1's classes and widening. The co-primary is P (S on
  home72). H, P72, home72, U and Δ_H − Δ_F are REPORTED, never decided.
- **The licences:** GAIN (ADOPT-CANDIDATE or TRADE) funds **X53-N1**, a from-scratch production run with this corpus
  (§6.5). NULL rules out only "N1 at the plateau" (plasticity confounds it). LOSS stops the family at the plateau.
- **The M4 hazard:** any N1-trained model's U is inflated on H and P72, so a goal-(b) claim from one needs a set-level
  coverage line and leans on F and the sealed `forum216b` (X54's owner is told first).
- **Registered prediction (GUESS):** Δ_F +1.5 pp, 80 % [−1.5, +4.5]; final GAIN 0.20, NULL 0.30.
- **Look 1 READ 2026-10-28: CONTINUE** ([`read/README.md`](../research_state/measurements/x61p_n1_production_2026-10-26/read/README.md),
  record `x35_breakers_stage_2026-10-17/reads/n1p_look1.json`). F +0.75 pp [−1.56, +3.06] (seeds +1.0 / −0.25 / +1.5;
  φ̂ 0.33, not widened): UNDECIDED with Δ̂ > 0, so look 2 (s3504–s3505) decides. P −3.27 pp [−4.20, −2.33]
  (Welch 97.5 %): the in-pool cost is resolved and below the registered 80 % guess. Reported: H +1.67 [−5.07, +8.38],
  P72 +2.86 [−3.89, +9.56], home72 vs Kakuna −0.95 [−7.63, +5.74], U +1.76 [−1.52, +5.03] (M4: not a goal read), U
  without H +1.80 [−2.17, +5.77], Δ_H − Δ_F +0.92. K9(b) and VOID guards 2–5 PASS on every seed; C0 re-played on one
  engine (Kakuna's legacy pack). A look-2 GAIN is read beside D1's two-look false-GAIN rate, 3.5–7.4 %, and is a TRADE
  unless P's interval clears −2.0.
- **Not built:** N4 (no width flag; `D_MODEL` is a constant) and the human-derived arm `n1_hrep_v1` (X61 PREREG §10).
  The from-step-0 toy screen, if wanted, is the review's Option T by a new registration.

### 6.5 X53 — a fresh wide run, conditional

- **Trigger:** N1-P reads GAIN (§6.5b). The original trigger (X52 = TRANSFERS, and D1 NULL or GAIN-short) did not
  fire: X52 and D1 both read UNDECIDED. **On N1-P's GAIN it is X53-N1, with
  `--team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both` as its one lever** and F as its primary (the M4 hazard).
- **Design:** fresh `--arch production` + `--team-mix w_forum_v1@ABC:0.5 --team-mix-sides both`. It is one lever
  against X26's own recipe. Add D3 only if D3 has read GAIN.
- **A 100M look (≈ 21 GPU-h):** U vs Kakuna at 100M against X26 @100M on the same cells (X26's checkpoints exist;
  ≈ 2 × 1,200 games ≈ 7 lane-h).
- **Continue to the plateau (≈ 60 GPU-h in all)** only if the 100M look's LB > 0.
- **X53-N1's design (REVISED DRAFT 2026-10-26, after review `6fca2c06`):
  [`design_x53_n1_fresh_breadth.md`](design_x53_n1_fresh_breadth.md).** It replaces the X26 @100M control above with
  a fresh same-pin control CF (X26 is 160+ `src/` commits away, so it stays descriptive) and makes F the primary (M4).
  A zero-GPU Stage 0 measures a fresh run's seed spread σ on eight fresh 15M seeds and gates the whole design (GO iff
  σ̂ ≤ 1.5 pp and its 80 % bound ≤ 2.5). Then one seed per arm to 100M (≈ 39 GPU-h; a catastrophic harm stop at 25M
  only) and a screen at 100M; 1 or 3 further pairs by σ̂; N1F and CF to their plateaus on GAIN. A true +3 pp reads
  GAIN 0.30–0.43 at σ 1–1.5 pp; a miss RETIRES the bet with its power printed. ≈ 46–52 GPU-h expected under a null.

### 6.6 X54 — the goal read, on a fresh panel

The panels F, H and P72 have steered every goal-(b) read so far (X37a through D3 look 1) and steer N1-P and D3 look 2. A goal claimed on the panels that chose the
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

- **Every X35 FINAL is to report U against Kakuna** (`kakuna plan --teamsets unfamiliar`, ≈ 3.5 lane-h per arm, CPU), as
  a descriptor no rule reads. A1 and SPRT (both NULL, FINAL) have no Kakuna U read in their records (FINDING 12).
- **The ensemble, READ (2026-10-26).** `main.anchors` plays `metamon:PastaMittens` (`d32ffcee4`), verifying each router call as
  deterministic, at ≈ 4× Kakuna's CPU per decision. P\* vs it, as registered: **U 0.484 [0.456, 0.512]** (n.d.),
  **home72 0.550 [0.501, 0.598]** (BETTER); against Kakuna's X51 twins **Δ-U −0.25 pp [−4.23, +3.73]** and Δ-home
  −4.75 pp [−11.5, +2.1], both NOT DETECTED. Kakuna stays the goal's opponent. The ensemble is not a separate bar
  until a read separates it from Kakuna ([`metamon_strongest_2026-10-25/`](../research_state/measurements/metamon_strongest_2026-10-25/)).

### 6.8 The cost of the direct path

| step | GPU-h | CPU lane-h | agent-days |
|---|---:|---:|---:|
| X51 | 0 | ≈ 9 | ≈ 0.2 (prereg + queue) |
| X52 | ≈ 2.2 | ≈ 8.4 | ≈ 0.5 (halves + prereg + kit, DONE) |
| X38 / D1 (already in X35) | ≈ 13 (C0 shared) | ≈ 10 | 0 (built) |
| X41 / D3: looks 1 and 2 RUN and READ (UNDECIDED, final) | 20.68 (MEASURED) | ≈ 10 (look 1) + ≈ 9.4 Kakuna and 5.5 S lane-h (look 2) | DONE |
| X61P / N1-P: look 1 READ, CONTINUE; look 2 licensed (the X61 toy: NOT-READY, not run) | ≈ 12.3 (look 1); ≈ 20.5 (final) | ≈ 44 (look 1); ≈ 29 (look 2, AM-N1P-1) | DONE |
| X53 (conditional); as X53-N1 (on N1-P's GO), a fresh pair (§6.5) | ≈ 21 to the look; ≈ 60 in all. X53-N1: ≈ 39 to the screen, ≈ 150 in all (CF to its plateau) | ≈ 7. X53-N1: ≈ 110–160 to the screen (with Stage 0), ≈ 205–305 in all, at the measured lane rate | ≈ 0.3. X53-N1: ≈ 2.2 |
| X54 | 0 | ≈ 5.6–7.9 per model | ≈ 0.3 (prereg; the carve is DONE) |
| **sure path (X51, X52, D1, X54)** | **≈ 15.2** | **≈ 36–39** | **≈ 1.0** |
| **with D3 and X53 to the plateau** (D3 at look 1; N1-P outside) | **≈ 88** | **≈ 53–56** | **≈ 2.8** |

---

## 7. What would change this plan

- **N1-P reads GAIN.** X53-N1 (a REVISED DRAFT) is registered, gated on its zero-GPU Stage 0 (§6.5).
- **N1-P and X59-S come back without a GAIN** (D3 already has: UNDECIDED, final). The fallback is
  [`strategy_if_levers_null_2026-10-26.md`](strategy_if_levers_null_2026-10-26.md), a REVISED DRAFT whose first
  branch is PR-1, a zero-GPU pooled precision read of D1 + D3 + N1-P against C0 on F.
- **A newer Metamon model appears upstream.** Re-check at X54. The goal names "Metamon's best".

---

## 8. FINDINGS (standing rule 7)

1. **The goal's opponent is under-specified.**
   - Every read used Kakuna ckpt 34 (the default).
   - Metamon's ensemble presets rate ckpt 28 higher (GXE 0.78 vs 0.75, numbers the code itself calls incorrect).
   - Metamon's strongest DEPLOYED agent is an ensemble (`PastaMittens`); read 2026-10-26, not detectably stronger
     than Kakuna (Δ-U −0.25 pp [−4.23, +3.73]).
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
3. **RESOLVED: D1 ran** (X35's order, after G1); its final reads UNDECIDED (§6.3).
4. **RESOLVED (X51 ran P\* at `63f97a57`): harness drift for P\*.** P\* trained at pin `9c033489`; every X26 Kakuna
   read ran from a detached `63f97a57` checkout, so P\* − 161M is a model difference only. Later reads play at later
   harnesses (X26c re-played P\* there game for game; D3's engine mix: §6.4).
5. **The dose figures assume ≈ 42 of our decisions per game, and that one training step is one of our decisions**
   (§4). They are ESTIMATES that order the levers. No rule reads them.
6. **§5's entries are MEASURED where a Kakuna read exists** (X51, X52, D1, D3 look 1, X26c, the belief prior mix,
   search) **and a GUESS elsewhere** (N1, X53, D4, capacity, X49, the human-replay lever, an ensemble of our
   checkpoints, test-time adaptation).
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
11. **RULED 2026-10-24 (X26c first, then D3 on a fresh GO). D1's final UNDECIDED matched none of §6's written triggers.** D3 was to run "on D1's verdict" and X53 "if D1
    reads NULL from plasticity or GAIN but short". The final is neither: +2.10 pp [−0.92, +5.11] with a −1.61 pp
    in-pool cost (§6.3).
12. **The X35 rule "every FINAL reports U against Kakuna" (§6.7) was not executed for A1 or SPRT.** Neither record
    holds a Kakuna U read; both are NULL in-house, so no goal-(b) descriptor exists for them.

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
| 2026-10-25 | **X41 / D3 BUILT and REGISTERED** (training-lever builder, orchestrator brief; CPU only) | the set-mutation corpus `d3_setmut_v1` (2,876 = 719 pool teams × 4 variants, k ∈ {1, 2} mons redrawn from Smogon usage, HP by `gen3_hp_repair_v1`, the pinned validator, no held-out-panel neighbour); the lever `--team-mix d3_setmut_v1@ALL:0.3 --team-mix-sides both`; kit arm `d3` with its own corpus-only pin D3_PIN (production pin + `src/utils/team_corpora/` only) and the G-E reader `ge_d3`; the label-path soak; D1's readout rule; the PREREG with predictions; checkargs and the launcher dry-run at the pin. Not launched: the GO is the orchestrator's | per-episode resampling inside the trainer (a trainer build, so a pin beyond the corpus package and an A/B whose code differs from C0's); excluding the home72 / away cores (they are the familiar cores the lever targets); a 0.5 share (§4.5 names q = 0.3); species substitution or procedural teams (§4.5 (b) / (c), separate levers) | `x41_d3_setmut_2026-10-25/PREREG.md`; `src/utils/team_corpora/d3_setmut_v1/` |
| 2026-10-25 | **X61 / N1 BUILT and REGISTERED** (training-lever builder, orchestrator brief after the 2026-10-26 literature survey; CPU only) | the procedural corpus `n1_proc_v1` (12,000 teams, Smogon facade only, ≥ 3-fit panel exclusion, the pinned validator, the label-path soak) and the held-out procedural panel `n1p72`; the lever `--team-mix n1_proc_v1@ALL:0.7 --team-mix-sides both`; the fresh toy pair `main.x61.arms` (toy_c0f / toy_n1, s6101, 100M, one pin); the PREREG with U_HF as the primary (P72 excluded as the corpus's own population), a seed-widened Newcombe, GO / KILL / BELOW-GO / UNDECIDED, K0 before launch; §0 item 5, §5, §6.5b and §6.8 quote it | adding breadth at a plateau again (D1's shape); P72 in the primary; reusing X35's toy C0 (another pin, a fork, fp64 τ); N4 as a third arm (no width flag); building the replay-team arm now (human data: its own toggle and snapshot, PREREG §10) | `x61_n1_breadth_2026-10-25/PREREG.md`; `sota_survey_2026-10-26.md` §5 N1 |
| 2026-10-26 | **X61's toy design NOT-READY (review `aed145fd`) ⇒ X61P / N1-P BUILT and REGISTERED** (training-lever builder, orchestrator brief; CPU only) | the review's recommended design: kit arm `n1p` = C0's argv + `--team-mix n1_proc_v1@ALL:0.5 --team-mix-sides both`, s3501–s3503 (+ s3504–s3505 on CONTINUE), 20M from P\*, against the shared C0; its own pin `5f8f6cfa` = `f4b25238` + the corpus's three files and its `CORPORA` line only (tag `x61p-n1p-pin-5f8f6cfa`), WITHOUT the 10-26 GIGO fixes (C0 ran without them); the primary Δ_F on F alone at 1,200 games per arm per seed by D1's rule (`d1` gains per-arm panels and a reported group); H / P72 / home72 / U / Δ_H − Δ_F reported (B1); GO ⇒ X53-N1; NULL ⇒ only "N1 at the plateau"; the M4 hazard named; `main.reads` specs; checkargs exit 0 and a dry-run FORK refused only on CUDA visibility | the toy design as registered (B1, B2, M1); the toy Option T now (the from-step-0 screen stays available by a new registration); D3's whole corpus subtree in the pin (more than the brief's "corpus files only"); the GIGO fixes in the pin (a second lever against C0); importing D1's banked P rows (no regime id) | `x61p_n1_production_2026-10-26/PREREG.md`; `x61_n1_breadth_2026-10-25/REVIEW.md` |
| 2026-10-26 | **Goal (b)'s panels checked for Kakuna familiarity** (measurement-validity worker; CPU, no games; survey F3) | the panels STAND: H, F and P72 at equal weight, with "unfamiliar" meaning unseen by OUR training and Kakuna's exposure reported, not required to be zero. The read found that Kakuna's exposure to an H team is at most its one source replay (≲ 2 trajectories; hl's window UNVERIFIED). H's ladder near-copy rate (4.2 % at ≥ 4) is below a random hl team's (8.9 %), and F's is 14 %. home72 is the most Kakuna-familiar panel: 8 exact `competitive` copies. Dropping H gives P\* 0.481 [0.447, 0.516], lower. X51, X26c and D1 look 2 are unchanged on every subset. U − H is reported beside U from now on; X54 is asked for two descriptive forum216b secondaries (without its 9 hl-≥ 4 teams; the 86 teams posted in 2026) | dropping H from (b) (no evidence of familiarity; it lowers the score and costs a third of the n); downloading Metamon's Kakuna-era team sets now (no go; listed as the open question's cost) | `goal_b_panel_validity_2026-10-26/` |
| 2026-10-26 | **D3 LOOK 1 READ: CONTINUE** (readout worker; CPU, 1,260 D3 Kakuna games + 140 guard re-plays + 10,500 in-house pairs; `main.reads` spec, D1's rule verbatim) | H +1.27 pp [−2.63, +5.16] (D3 0.491, LB 0.464; seeds +2.4 / −2.9 / +4.3), without H +0.48 [−4.29, +5.24]; P −0.24 [−1.10, +0.62] NON-INFERIOR; guards not read; C0's banked rows reused after the determinism guard re-played 7 / 7 units game for game; goal (b) not met by D3; look 2 licensed; §0 item 4 and §6.4 quote it | reading +1.27 as a set-generalisation gain (LB < 0); re-playing C0 (the guard passed; +1,260 Kakuna games + 10,500 pairs); calling the look a NULL (UB +5.2 > +3) | `x41_d3_setmut_2026-10-25/read/`; `x35_breakers_stage_2026-10-17/reads/d3_look1.json` |
| 2026-10-26 | **PastaMittens (Metamon's ensemble) READ: NOT DETECTED vs Kakuna** (anchor worker, the owner's ask "Metamon's largest or most powerful gen3ou model"; CPU, 1,600 games, preregistered `62bb07c00`) | P\* scores U 0.484 [0.456, 0.512] and home72 0.550 [0.501, 0.598] against it; Δ-U −0.25 pp [−4.23, +3.73] and Δ-home −4.75 pp [−11.5, +2.1] against X51's Kakuna twins; Kakuna ckpt 34 stays the goal's opponent (4× cheaper); §2 item 2, §6.7 and §8 FINDING 1 quote it | switching the goal to the ensemble (no separation from Kakuna); the largest model `SyntheticRLV2` (200.9M, rank 7 on upstream's gen3 board) | `metamon_strongest_2026-10-25/` |
| 2026-10-26 | **Views audit: this doc made current** (research-integrity auditor; docs only) | the status line rewritten (D3 look 1 READ, N1-P TRAINED and being read); §5 carries the MEASURED reads (D1 final, D3 look 1, X26c, the belief prior mix, A1 / SPRT, X47, FR-1 / FR-3, stage 1c); §5's arithmetic, §6.5's trigger and §7's triggers restated on the live reads; §3 gains the D1, D3 and prior-mix U rows; D3's engine-mix caveat (§0, §6.4); FINDINGS 3, 4, 6, 11 resolved, FINDING 12 added; the panel-validity row's hl comparison corrected (H 4.2 % is below a random hl team's 8.9 %) | leaving resolved triggers as live text | `research_state/measurements/views_audit_2026-10-26/` |
| 2026-10-20 | Expected pp per lever | quoted from measured effects where they exist (X37b's 100M → 280M transfer; L1's search null; FACE_c), GUESS elsewhere, with the arithmetic that (b) needs +6–8 pp | single-number promises | §1, §5 |
| 2026-10-25 | Status line made true (docs-health pass) | the status still said D3 / X53's trigger "is the orchestrator's call"; it now carries the 2026-10-24 ruling (neither now) and the X26c FLAT read (rows above) | the old line | this Decision record 2026-10-24, 2026-10-25 rows |
| 2026-10-25 | **The human-replay lever row re-stated after REVIEW D** (design worker) | the first reads (FR-1 / FR-3 playing; X59-S built and registered: three matched student arms × 5 seeds, Δ_data = S-H − S-S answers the owner); H-TEAMS then H-BC; H-INIT dropped; X60 NOT-READY | X59 as the first read (withdrawn: its rule rewarded forgetting); H-BC called "the only gradient on unfamiliar states" (struck) | `human_replay_quality_2026-10-25/RESPONSE.md` §D; `x59s_matched_students_2026-10-25/PREREG.md` |
| 2026-10-26 | **Two DRAFT design docs added: X53-N1 and the all-null contingency** (research-strategy worker, orchestrator brief; docs only) | [`design_x53_n1_fresh_breadth.md`](design_x53_n1_fresh_breadth.md): the from-scratch pair N1-P's GO funds, N1F (`n1_proc_v1` at 0.5) vs a fresh same-pin control CF, F primary, staged (≈ 39 GPU-h to a one-seed screen at 100M, ≈ 114 if every stage runs), the human-team corpus as a separate local-only arm, the cloud as its natural home once an approval covers it; §6.5 and §6.8 point to it. [`strategy_if_levers_null_2026-10-26.md`](strategy_if_levers_null_2026-10-26.md): if D3, N1-P and X59-S all miss a GAIN (≈ 0.58 by their own priors), a zero-GPU week (the N2 / N3 / plasticity probes, search's E2) and then, by expected pp per GPU-h, N5 (an offline breadth student), N2 (set-channel noise) and a breadth stack fork, with an owner escalation at ≈ 90 GPU-h | X53 as written (X26 @100M as its control: cross-pin, one seed); reading another +1–2 pp lever alone (a ±4 pp read cannot see it) | the two docs; N1-P PREREG §6, §8; `sota_survey_2026-10-26.md` §5 |
| 2026-10-26 | **X53-N1 and the all-null contingency REVISED after their review** (`6fca2c06`; research-strategy worker, orchestrator brief; docs + CPU Monte Carlos) | X53-N1: a zero-GPU σ gate (Stage 0), power at +2 / +3 pp, no L50 futility stop, the unselected Stage-B estimate, a RETIRE outcome; §6.5 and §6.8 re-stated. The contingency: PR-1, a zero-GPU precision read of D1 + D3 + N1-P vs C0 pooled on F (38,000 games, one engine; 0.76 at +1.5 pp), as its first branch; rank by P(GAIN) per resource-day; the owner asked at the end of its zero-GPU stage | the drafts' L50 futility stop, 1.5 pp fixed floor and ≈ 90 GPU-h exit | [`strategy_revision_2026-10-26/`](../research_state/measurements/strategy_revision_2026-10-26/) |
| 2026-10-27 | **"The premium goes with the TEAM context" softened to OPEN** (doc editor, orchestrator brief; docs only) | the "Where it is lost" bullet: the per-mon KO drop restates the result, the result-fixed comparison reads −0.007 [−0.093, +0.070]; the "bounds D3 at a quarter" claim removed. No measured number changed | keeping the team-context reading | `../research_state/measurements/h_mid_value_test_2026-10-26/REVIEW.md` B1, M3 |
| 2026-10-27 | **H-CTX read: SUPPORTED** (H-CTX worker, orchestrator brief; 2,800 Kakuna games) | §4's "Where it is lost": the premium goes with the TEAM context (rc72b recovers −0.34 [−2.05, +0.20] of it, on one engine with home72 and F controls); the per-team win-condition account is now measured, not a hypothesis; the status line names it | "OPEN until rc72b" (10-27 softening, now resolved); the absolute ≥ 0.57 / ≤ 0.52 bars against X51 (kept only as a cross-engine line, Amendment 1) | `measurements/hctx_rc72_2026-10-26/README.md`, record `read_am2/record.json`; PREREG `b044d214b` + `aaf7ced86`, `905c29f5b` |
| 2026-10-27 | **H-CTX's reading narrowed after its adversarial review** (reviewer, orchestrator brief; docs + one CPU script on banked games) | §4 and the status line: "goes with the TEAM" becomes "does not go with the sets" (≤ about a fifth; a quarter against X51's U); the mechanism is open, since rc72b's sixes are less coherent than both controls (memorised sixes, core knowledge or a plan-less six); "must add TEAM contexts" and "the win-condition account is now measured" withdrawn; X62's corpus read on F only. The verdict is unchanged | "the per-team win-condition account is now measured, not a hypothesis" | `../research_state/measurements/hctx_rc72_2026-10-26/REVIEW.md` M1–M3, `review_results.json` |
| 2026-10-28 | **LR-F look 1 read: UNDECIDED, FINAL** (LR-F read operator, orchestrator brief; 21,000 in-house S games, CPU) | the X35 plateau-breakers row: LR-F (P\* + 20M with the learning rate floored two controller rungs up) reads −1.13 pp [−2.45, +0.18] on in-house S (Welch 97.5 %, 3 seeds). No look 2. It says nothing about goal (b), and no Kakuna game was played (the guards are read only on a GAIN) | — | `x35_breakers_stage_2026-10-17/reads/lrfloor_look1.json`; ledger 2026-10-28 · *X35 LR-F LOOK 1* |
| 2026-10-28 | **H-CTX's A-K check read: rc72b UNDECIDED under Kakuna, detected weaker** (A-K read operator, orchestrator brief; 1,080 Kakuna-vs-Kakuna games, CPU) | §4's H-CTX caveat replaced: under a neutral pilot rc72b is about 4 pp weaker, uniformly (no coherence gradient); the strength-adjusted share (post hoc, across engines) is 0.21 [−0.88, +0.88]; the premium stays mostly team-level, and "no plan" covers at most the ~4 pp part. No lever's order changes; rc72c (core-preserving recombination) is the registered separator of memorised sixes from cores | the A-K cell was the review's cheapest check; it bounds Kakuna's own response, which H-CTX left open |
| 2026-10-28 | **D3 LOOK 2 (FINAL) READ: UNDECIDED** (D3 look-2 read operator, orchestrator brief; CPU, 2,100 D3 Kakuna games + 140 guard re-plays + 17,500 in-house pairs; every game on E1 at the play pin `c4a5c37b6`) | H +2.33 pp [−0.69, +5.35] (D3 0.494 vs C0 0.471; 4 of 5 seeds positive), P −0.45 [−1.16, +0.25] NON-INFERIOR; guards not read; look 2 TERMINAL. §4 item 4, §5's row, §6.4, §6's cost table and §7 updated: D3 is no longer an open route | a look 3 (terminal by registration); deciding from main's newer code (the pin decides, AM-D3-L2 addendum B1) | `x41_d3_setmut_2026-10-25/read/README_look2.md`; record `x35_breakers_stage_2026-10-17/reads/d3_look2.json` |
| 2026-10-28 | **N1-P LOOK 1 READ: CONTINUE** (N1-P look-1 reader, orchestrator brief; CPU, 9,720 Kakuna games + 21,000 in-house pairs; `main.reads` spec `read_spec_look1.json` byte for byte, state at the registered path) | F +0.75 pp [−1.56, +3.06] (N1-P 0.516 vs C0 0.508; 2 of 3 seeds positive), UNDECIDED with Δ̂ > 0 ⇒ look 2; P −3.27 pp [−4.20, −2.33], INFERIORITY-NOT-EXCLUDED; H / P72 / home72 / U reported only (U +1.76, without H +1.80, Δ_H − Δ_F +0.92); guards not read; every VOID guard PASS; §0 item 5, §5's row and arithmetic, §6.5b and §6.8 quote it | re-using C0's banked rows (the registered spec has no `import_off`, and editing it would move look 2's pinned sha); reading +0.75 as breadth transfer (LB < 0); calling the look a NULL (UB +3.06 > +3) | `x61p_n1_production_2026-10-26/read/`; `x35_breakers_stage_2026-10-17/reads/n1p_look1.json` |
| 2026-10-30 | **X48 LOOK 1 READ: NO-GO, "experts not engaged"** (X48 read operator, orchestrator brief; CPU, 168,000 in-house S games + 5,980 Kakuna games on E1 at the shared play pin `86cfdf7ef`) | the capacity row: X48's per-kind FFN experts give no ≥ 2 pp in-house gain within 20M (−0.55 pp [−1.32, +0.21], 90 %); its forum72 K is +2.06 pp [−0.06, +4.19], UNDECIDED, a weak meter that is not goal-(b) progress. The NO-GO is weak because the copied AdamW moments throttled the experts (D_k / null 0.72–0.93). FFN width (N4) and `D_MODEL` stay open | the row's GUESS-only evidence for X48 | ledger 2026-10-30 · *X48 LOOK 1*; `x35_breakers_stage_2026-10-17/reads/x48_look1.json` |
