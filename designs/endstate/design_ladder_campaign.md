# Design — The ladder campaign: stages, their meters, and the training ecology

**Status: DESIGN, being executed stage by stage (updated 2026-09-27).** Authored 2026-09-23 by the
orchestrator at the owner's request, from a design conversation the same day. 🚨 **ALWAYS-CURRENT
(owner, 2026-09-27)**: see [`README.md`](README.md). §2a states the new lineage the stages now run in. It organises the research into **stages** that each own
one artifact, one meter and one plateau test, and it states the **training ecology** (who plays
which team, and why) that the stages need. Nothing here is a commitment; §7's ordering is the
recommendation.

**Companions:** [`design_model_management.md`](design_model_management.md) supplies the registry
ROLES this campaign fills and the `ecology` record block it writes.
[`design_three_tier_environment.md`](design_three_tier_environment.md) with
[`program_rust_core.md`](program_rust_core.md) is stage 0.

---

## 0. The one-paragraph version

The goal is a model that tops the gen3 OU ladder while piloting a handful of iconic Smogon sample
teams. Today one big loop does everything, and when it stalls nobody can say *which part* stalled.
This month is the evidence: weeks went into separating "the fold leaks", "the teacher has nothing
to teach", "the dose is wrong" and "the meter cannot resolve it", which were four different broken
parts behind one number. The campaign splits the work into **stages** (environment → generalist →
main agent → exploiters → value function → search → ladder). Each stage produces one artifact, is
read by one meter, and has a written plateau test **and a written guess at what would un-plateau
it**, so every stall arrives with an address. **An era is what happens when an upstream stage
produces a new artifact**: the stages downstream of it are re-derived. **One stage holds the GPU at
a time; the rest stay frozen.** Every game any stage plays is drawn from a declared **ecology**:
(our team) × (who the opponent is) × (the opponent's team), three draws with three different jobs.

---

## 0a. The goals, stated as numbers (owner, 2026-09-27)

| horizon | goal | what it would mean |
|---|---|---|
| **near term** | **beat Metamon's BEST model on OUR teams, with Metamon at its STRONGEST sampling temperature** — the lower 95% bound of our win rate above 0.50 | we are in the ball park of a state-of-the-art gen3 OU model |
| **long term** | **beat Foul Play** (on shared teams, at a stated search budget) | we are in the ball park of the best known gen3 OU bot |

⚠️ **Metamon's best is `Kakuna`, not `SyntheticRLV2`.** The local Metamon checkout (`~/dev/metamon` @ `0a00a759`, 2026-05-22) documents `Kakuna` as "the current best Metamon policy" (Superkazam finetuned on +7.8M higher-temperature self-play battles; estimated gen3ou GXE vs humans ~63%). Our standing era-gate reference has been `SyntheticRLV2` (the paper's model). The near-term goal is read against `Kakuna`; upstream may have newer models than this checkout — verify before the read. **"Strongest sampling temperature"** is measured, not assumed: hold our side fixed (greedy) and vary Metamon's temperature; the goal read uses Metamon's best cell. The last era's standing on our teams was `SyntheticRLV2` greedy · home **0.500 [0.404, 0.596]** (`ai_v12_02_winprob_critic` @75M) and Foul Play @1 s **0.388** (asymmetric teams, not like-for-like). Tracked as EXPERIMENT_BACKLOG X22.

## 1. Why stages (and why not a waterfall)

A plateau is information only when it has an address. "Round 1's gap did not fall" means little
while the generalist, the exploiters and the meter all move together; it means a lot when the
exploiter recipe and the meter are frozen and only the generalist moved. So each stage FREEZES its
inputs while it runs, and the stage's own meter reads only its own artifact.

The stages form a dependency graph with feedback, not a waterfall. Exploiters need a main agent;
the main agent needs opponents, which the generalist and the exploiters supply; the value function
learns from games the others generate. The rule that keeps this cheap and readable: **a stage
re-runs only when one of its inputs changes, and only one stage trains at a time.**

## 2. The stages

| # | stage | artifact | meter | plateau test | where it stands (2026-09-27) |
|---|---|---|---|---|---|
| 0 | environment | the Rust core | parity + throughput | — (infrastructure) | M1–M4 built; the cutover (M6) DONE `ac0b6469` (training reads core observations); deletion pass part 1 done; M5 Phase A done, Lane 0 scheduled (`program_rust_core.md`) |
| 1 | **generalist** | broad pilot, all teams | untaught meter; SmallRL anchor; best-response gap | an 8M block's untaught Δ inside the 3.69 pp floor | **NEW ERA (§2a):** N0 = `ai_v14_01_base`, fresh 75M on the clean-input boundary `b0a28b5b`, running (end-of-run reads registered, EXPERIMENT_BACKLOG X1); G0′ = its first continuation block inside the floor. *History:* the old lineage PLATEAUED ONCE at `ai_v13_12_plateau` (G0), ledger 2026-09-21 |
| 2 | **main agent** | G-derived, fine-tuned on ~5 iconic sample teams | anchors ON those teams; win rate vs pressure opponents per team | per-team gain inside its floor over a block | one WARNING point: stage A (below); not re-run in the new era |
| 3 | **exploiters** | best responses to the stage-1 or stage-2 artifact | `main.best_response_gap` per archetype | the gap stops falling across rounds | old lineage: rounds 1 and 2 both NOT DETECTED against the 5.0 bar (Δ −10.00 and −8.25 pp), branch N+ FINAL, no round 3 (owner). Carried into the new lineage with more power (§2a) |
| 4 | **value function** | the critic | DISCRIMINATION: pairwise sibling accuracy on rollout-labelled forks | accuracy flat across a lever | **PLATEAUED**: twelve learned heads at 0.567–0.578; a 4-rollout leaf beats them (+0.057 DETECTED). Next levers: N0's leaf read (X1), a hand-eval leaf (X3), and the Q head ([`design_q_head.md`](design_q_head.md), X4), which outranks every other research row (owner, 2026-09-27) |
| 5 | search | search over 2 + 4 | win-rate dividend at fixed compute | — | gated on 4; wound down as an objective |
| 6 | ladder | the deployed agent | ladder Elo | — | — |

### 2a. The new lineage (2026-09-26 →): what the stages run in now

The observation-architecture batch (`b0a28b5b`, v121) folded in every GIGO fix of the week and made
`MIGRATION_FLOOR` 121, so no old checkpoint loads: stage 1 restarted from scratch. Registration:
[`../research_state/new_lineage_2026-09-26.md`](../research_state/new_lineage_2026-09-26.md);
owner GO 2026-09-26.

- **N0 = `ai_v14_01_base`**: a FRESH 75M run, the old fresh root's recipe (`ai_v13_02_flywheel_winprob`)
  token-exact except the run name, the pin (`8d07051a`), `--arch production` and `--obs-source core`.
- **Opponents: the POOL only, first.** The Metamon ladder teams enter only after their Hidden Power
  fields are repaired (TASK_BACKLOG T8), as a SEPARATE **ladder-teams arm** that reproduces N0's
  recipe with them added and is read at matched snapshot count (EXPERIMENT_BACKLOG X9). The repair must
  not change any file a live pinned run reads from `data/`.
- **The population loop carries over**, with both power levers registered before round 1: readers at
  `--eval-games 200` (800 games each; this row wrote `--eval-battles 200`, which sized only the deleted post-training final eval and never a live eval cycle — corrected in P6, 2026-10-02) and a **POOLED read over R = 4 rounds**
  (D = mean_r[gap(RB_r) − gap(RC_r)], one verdict, no per-round verdict, no extension; detection needs
  D ≲ −7.4 pp). The stable set is a WINDOW {A′, A2′, the most recent reader} at share 0.40, so each
  specialist's exposure stays at 0.12. KILL guards: the untaught 8 against `untaught_meter_opponent_v14`
  (N0's 24M snapshot) and the SmallRL anchor.
- **Archetype exploiters from ladder teams** join the loop once the repaired teams exist (X10): a wider
  exploiter set, read by the same pooled rule.
- **Pool memorization is a SUCCESS milestone, then generalize (owner, 2026-09-27).** The belief heads
  measurably memorize the pool (on-pool DiD +4.8 pp; nothing detectable on ladder teams, ledger
  2026-09-24). That is reported on-pool FIRST, off-pool second, and the ladder-teams arm is the
  generalization step. Root-causing it (X8) runs in parallel, below the Q head.
- **The learner battery** (X2; [`../research_state/learner_battery_2026-09-26.md`](../research_state/learner_battery_2026-09-26.md)):
  four +8.06M forks of N0's final at the frozen generalist dose (control, `--n-epochs 5`
  dose-matched, TF32, policy GAE λ 0.95), judged on strength per GPU-hour. It runs BEFORE the G0′
  continuation blocks; its control doubles as the first block, K1.

**Stage 2's warning point.** Stage A (`ai_v13_16_teach5_offense_dist`) was a stage-2 experiment in
all but name: G0 fine-tuned for +8M on five offense teams against a distribution of opponents. It
came out LEVEL with its parent on those same teams against a third party (−1.37 pp [−2.78, +0.27]).
At 1.78× the dose, the same recipe went WORSE (−7.15 pp). So "fine-tune on five teams and they get
piloted better" is not free at G0. In this campaign that is a useful answer, not a failure: if
stage 2 plateaus immediately, the limit is team-specific piloting (capacity, credit assignment over
long horizons, or what the observation exposes), and sampling is not the lever.

**Stage 4 is the one gating search.** Its meter exists. Its open question fits the campaign: **does
a NARROW value function discriminate better?** The Big 5 + Starmie win-prob exploiter queued on
2026-09-17 was meant to test exactly this, and its question was never answered. Critic
discrimination on five teams may be reachable where broad discrimination is not.

**Written before running (the un-plateau guesses).** A stage that plateaus with no written guess
turns into a shrug; one that has a guess turns into the next arm. The guesses to record per stage:
- generalist: capacity; the credit-assignment horizon; observation features;
- main agent: dose, team count, the stall-specific concepts;
- exploiters: budget, and a different archetype;
- value: label quality (rollout labels), narrowness, and architecture.

## 3. Eras, defined

An **era** starts when an upstream stage emits a new artifact that downstream stages adopt. A new
generalist G1 begins a new era for stages 2–5, which are re-derived from G1. A new critic begins a
new era for stage 5 only. An era is named after the artifact that started it and is recorded in
each downstream model's lineage, so "which era did this number come from" is read, never
remembered. Calendar eras (the "flywheel era", the "win-prob era") remain as history; they are not
this definition.

## 4. The ecology: three draws, three jobs

Every game is **(our team) × (who the opponent is) × (the opponent's team)**. Today one 719-team
pool and a set of accreted flags (`--heuristic-floor`, `--stable-opponent-selfplay-share`,
`--team-pfsp` (deleted in deletion pass L4, 2026-10-02; the one PLR sampler below replaces it, a new build), `--team-block-episodes`, `--pool-spread`, `--pfsp-scale`) serve all three draws at
once. That is the root of several confounds this month, the unregistered `--team-block-episodes`
lever among them.

### 4.1 Our team: the deployment target lives here

*(`data/teams/sample/` today holds the 32 Smogon sample teams AND the 40 promoted fleet teams: the
split that separated them was REVERTED, §7 step 0. "The 32" below means the Smogon teams.)*

| source | share (registered per stage, illustrative) | job |
|---|---|---|
| **sample** (the 32 Smogon teams; stage 2 picks ~5) | the majority | what we optimise for; within it, LEARNABILITY-weighted with a uniform floor (§5) |
| **pool** (usage-weighted) | a minority | breadth for the representation, not a skill we value for its own sake |
| **procedural** (§4.3) | a small slice | novel states to pilot through |

🚨 **UNVERIFIED and must be measured, not assumed:** that breadth on our side HELPS sample-team
piloting. Team count dominates conditioning (the N = 20 ceiling), and negative transfer is measured
here. §7's arm E1 prices it.

### 4.2 The opponent: two slices, two jobs

- **Pressure opponents**: *well-piloted* opponents on the teams we will actually face. Teams: the
  ladder distribution, usage-weighted from Smogon statistics, weak teams included, because the
  ladder has them (the owner's rule: priors are Smogon-derived). Pilots, which must play those teams
  WELL or the pressure is fake:
  - **recent self-snapshots**;
  - **a frozen broad league**: G0 and its broad-trained lineage kept permanently as pilots of
    non-sample teams. This breaks a real coupling: if stage 2 narrows the main agent, the main
    agent's own snapshots get worse at piloting other teams and the pressure quietly weakens;
  - **stage-3 specialists** with one-sided PFSP (the right place for it, because in a foreign pool
    hardness is not recency).
- **Coverage opponents**: *unusual* opponents, where the pilot's quality matters little.
  **Procedural teams** piloted by snapshots, which generate novel positions; and **bots**, the
  stationary yardstick and anti-forgetting floor, kept as a DECLARED share and never folded into
  PFSP (at p ≈ 0.92 PFSP would make their share emergent from pool size, or starve them).
- **Held out from every draw:** the untaught 8, so the meter stays honest.

### 4.3 The procedural source already exists

`src/rust_sim/harness/ou_random_teams.js` generates random but ON-DISTRIBUTION gen3 OU teams from
Smogon statistics only (usage, teammate joint priors, move/item/ability/spread priors), and
deliberately not from the pool, so it obeys the Smogon-only rule. Today it serves only the fuzzers.
The campaign promotes it to a first-class team source behind the data facade, so training, fuzzing
and evaluation share one generator.

### 4.4 Self-play weighting: what stays and what changes

- **Self-snapshots:** uniform or `--pool-spread`, not PFSP. In a self-pool hardness tracks recency
  (the June run: 0.87 vs the 6M self, 0.46 vs the 114M self), so PFSP amplifies recency. The one
  clean A/B, gen-17 vs gen-16, measured **+26.5 Elo [+17.9, +35.0]** for turning it OFF on a
  homogeneous pool. The win-prob-era test (`ai_v12_04_pfsp_fork25M`) was paused unread; one
  matched two-arm contrast on G0 would settle it.
- **Foreign opponents (specialists, other lineages):** one-sided PFSP, `1 + s·(1 − p)`. The floor
  keeps coverage, and the extra keeps pressure on whoever beats us.
- **Mastery retirement:** off for the population loop (retiring at 0.80 abandons counterplay the
  loop exists to close).

## 5. Our-team sampling: learnability, not loss rate

A win rate mixes team STRENGTH, MATCHUPS and PILOT COMPETENCE; only the last is trainable. Two tools
separate them:

1. **The strength/piloting split, free and offline.** Over the (snapshot × team) win-rate table the
   pool already records, fit logit(win) ≈ snapshot skill + team strength + snapshot×team
   interaction. Team strength is what every pilot shares; the INTERACTION is "this pilot, relative to
   its general skill, does badly here". A persistently negative, non-shrinking stall interaction is
   a piloting deficit, not a weak team.
2. **Learnability scoring (prioritised level replay, PLR).** Score each of our teams by how much the
   agent can still learn there: the mean positive advantage (critic surprise), with a staleness
   bonus, rank-based weights, and a uniform mixing floor (the owner's one-sided floor, by its
   standard name). With a win-prob critic, a weak team's losses are EXPECTED (small advantages, low
   score), while a strong team piloted badly produces surprise (high score). **Caveat:** the score
   is only as sharp as the critic's discrimination, which is stage 4's open problem.

**Stuck vs. learnable.** A sampler can oversample a team whose obstacle is structural: stall's
long horizons with the sparse terminal reward, slow resource accounting the observation may not
expose, and the forfeit turn limit. The strength/piloting split, tracked over training, says which
case holds. If stall's interaction does not shrink while its share is high, the fix belongs outside
the sampler.

## 6. What gets built once, and what stays a lever

Built once:
1. **One match sampler with a declared `ecology` block**, recorded in the model record (the field
   `design_model_management.md` §3.1 already names). It replaces the accreted flags in §4.
2. **Three team sources behind `agents.gen3_data`:** sample (the 32), pool (usage-weighted), and
   procedural (the generator moved out of the fuzz harness).
3. **Every episode TAGGED with its cell** (our-source × opponent role × opponent-source) in the
   event log, so every meter can read per cell. Most of this month's confounds were two cells
   averaged together unseen.
4. **Meters by cell:** per-archetype piloting (§5.1), the best-response gap per archetype, and
   anchors on the sample teams.

The **mixture weights** are experiment levers, set by registered arms, never by feel.

## 7. Ordering (recommended)

| step | what | cost | gates |
|---|---|---|---|
| 0 | the sample-team split: `sample/` = the 32 Smogon teams; the 40 promoted fleet teams get their own role — **LANDED and REVERTED 2026-09-23** (`33da2cf6`, reverted `e74c0610`): pinned runs read `data/teams/` from the MAIN checkout, so moving files changed a live run's inputs. Not re-landed; any data move now waits for no pinned run to be live or queued | small build | — |
| 1 | the strength/piloting split on existing eval data; is stall a piloting deficit? | free, offline | — |
| 2 | choose the stage-2 teams (~5, one per archetype plus iconic extras) from the 32, with the owner | a conversation | step 1 informs it |
| 3 | finish population-loop round 1 (stage 3 aimed at stage 1): does absorbing exploiter pressure work at all? — **DONE**: the loop ABSORBED its specialists (M = +8.50 pp on the specialists' first team; +9.33 [+3.77, +14.81] over all five, re-measured 2026-09-30 for F-LH-13, `research_state/measurements/ext_first_team_audit/`, no decision changed), but the gap's fall (−10.00 pp) did not clear the bar; round 2 the same (−8.25, N+ FINAL). Underpowered, so the new lineage pools four rounds (§2a) | ~21 GPU-h | — |
| 4 | first stage-2 arm: G0 fine-tuned on the chosen teams against pressure opponents, read by the anchors on those teams | ~1 arm | a plateau is a result |
| 5 | E1: our-side breadth, sample-only vs + pool vs + pool + procedural | 3 arms | sets §4.1's shares |
| 6 | the narrow-critic question: discrimination on the stage-2 teams vs broad | 1 arm + the offline meter | decides whether stage 5 reopens |
| 7 | the match sampler + ecology block + cell tagging | a build; after the Rust core's M2, where team sources and tagging belong | — |

*(Steps 1, 2, 4–7 not started as of 2026-09-27; the new lineage (§2a) and the Q head took the GPU
and the build slots first. EXPERIMENT_BACKLOG is the ranking now.)*

## 8. Open questions this document does not settle

1. **How narrow?** "Great on five teams, robust against everything" is a different target from "a
   gen3 OU generalist", and it sets how much breadth is worth paying for.
2. **Does the generalist need to keep improving?** If stages 2–5 climb on a fixed G0, stage 1's job
   is "a good starting point and a good sparring partner", which is cheaper.
3. **Should procedural teams ever be on our side?** Novel states we pilot through build
   representation; novel states only the opponent creates build defence. They may be worth
   different amounts.
4. **Should the population loop's target move from the generalist to the main agent** once stage 2
   exists? Probably yes. *(Round 1 answered absorption: yes. The new lineage still aims the loop at
   the generalist G0′, since no stage-2 artifact exists.)*

---

## Decision record

Owner decisions are marked **(owner)**. `L…` is the ledger line as `ledger_index.md` lists it.

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-09-23 | How to organize the research **(owner conversation)** | STAGES, each with one artifact, one meter, one plateau test and a written un-plateau guess; one stage trains at a time | One big loop whose stall has no address | this doc §0–§1 |
| 2026-09-23 | The deployment target **(owner)** | A main agent on ~5 iconic SMOGON sample teams (the 32), not the 40 promoted fleet teams | — | this doc §0, §4.1 |
| 2026-09-23 | Team sources | Sample, pool and procedural (Smogon-only generator), three draws with three jobs | One 719-team pool serving all draws through accreted flags | §4 |
| 2026-09-23 | Sample-team split | Landed, then REVERTED the same day | Keeping it (it changed a live pinned run's inputs: pinned runs read `data/` from main) | `33da2cf6`, `e74c0610`; ledger L21269, L21273 |
| 2026-09-23 | Population loop round 1 | B (loop, share 0.40, PFSP on, no distillation) vs C (share 0.0), each read by a fresh exploiter | — | `da0ab0b0`; ledger L21246 |
| 2026-09-24 | Round 1 verdict | NOT DETECTED → branch N+ (Δ −10.00 [−16.60, −3.27] vs bar 5.0); the loop absorbed | — | ledger L21329 |
| 2026-09-25 | After round 2 **(owner)** | No round 3 on the old lineage; carry the loop into the NEW lineage with both power levers registered up front | A round 3 on the old lineage | ledger L21387, L21391 (N+ FINAL) |
| 2026-09-25/26 | A new era **(owner)** | One retrain boundary on clean inputs (`b0a28b5b`), a FRESH lineage N0 on the Rust-core observation | Continuing the `ai_v13` lineage | ledger L21407, L21411 |
| 2026-09-26 | N0's opponents | Pool only first; ladder teams as a separate arm after the Hidden Power repair | Ladder teams in N0 (their HP fields are unrepaired) | `new_lineage_2026-09-26.md` §0, §7 |
| 2026-09-26 | Loop design in the new lineage **(owner)** | R = 4 rounds, pooled read, 200-game readers, WINDOW stable set, `untaught_meter_opponent_v14`, the 3.69 floor carried provisionally, the `ai_v14` prefix | R = 5 (+18 GPU-h for power 0.78 vs 0.69); the accumulating stable set; re-measuring the floor | ledger L21415; registration §8 |
| 2026-09-26 | The learner battery | Four forks of N0's final (C, E5, T32, L95), strength per GPU-hour, BEFORE the continuation blocks; D_g = 2.8e-5 (orchestrator, owner-delegated) | — | `6df5160f`; ledger L21425 |
| 2026-09-27 | Research priority **(owner)** | Q right first; memorization chipped in parallel; pool memorization is a SUCCESS milestone, then generalize (on-pool reported first) | Root-causing memorization before Q | EXPERIMENT_BACKLOG header (`93745a66`) |
| 2026-09-27 | Exploiter breadth | Archetype exploiters from repaired ladder teams inside the pooled loop (proposed, X10) | — | EXPERIMENT_BACKLOG X10 |
| 2026-10-02 | Distillation (the exploiter fold) and the search teacher DELETED (deletion pass L3, owner: "delete all, port none") | The stages no longer have a built distillation route: stage 3 (exploiters) feeds the generalist only through the opponent POOL (the population loop), and stage 5 (search) has no training-side teacher consumer. Distillation was the only BUILT route to X15 (expert iteration); a Rust port is ~1-2 agent-days if X15 is ever scheduled | Porting either onto the Rust core first (nothing is scheduled to use them) | `designs/deleted_flags.md`; `designs/training/exploiter_and_distillation.md` HISTORY |
| 2026-10-02 | The 200-game reader's flag is `--eval-games`, not `--eval-battles` (deletion pass P6) | The power lever of the readers is `--eval-games 200` — the flag that sizes every live eval cycle. `--eval-battles` sized ONLY the post-training final eval, which is deleted (D5), so the flag is gone and an argv carrying it is refused | Keeping `--eval-battles` as a no-op alias (a lever that silently sets nothing is how a registered power lever goes unapplied) | `src/main/train/callbacks.py` read `args.eval_battles` only into `final_eval`; the live callbacks read `args.eval_games` |
