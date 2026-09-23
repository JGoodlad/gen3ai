# Design — The ladder campaign: stages, their meters, and the training ecology

**Status: DESIGN, not scheduled.** Authored 2026-09-23 by the orchestrator at the owner's request,
from a design conversation the same day. It organises the research into **stages** that each own
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

| # | stage | artifact | meter | plateau test | where it stands (2026-09-23) |
|---|---|---|---|---|---|
| 0 | environment | the Rust core | parity + throughput | — (infrastructure) | Phase 0 done; M1 next (`program_rust_core.md`) |
| 1 | **generalist** | broad pilot, all teams | untaught meter; SmallRL anchor; best-response gap | an 8M block's untaught Δ inside the 3.69 pp floor | **PLATEAUED ONCE**: `ai_v13_12_plateau` (G0), ledger 2026-09-21 |
| 2 | **main agent** | G-derived, fine-tuned on ~5 iconic sample teams | anchors ON those teams; win rate vs pressure opponents per team | per-team gain inside its floor over a block | one WARNING point: stage A (below) |
| 3 | **exploiters** | best responses to the stage-1 or stage-2 artifact | `main.best_response_gap` per archetype | the gap stops falling across rounds | round 1 aimed at G0, queued 2026-09-23 |
| 4 | **value function** | the critic | DISCRIMINATION: pairwise sibling accuracy on rollout-labelled forks | accuracy flat across a lever | **PLATEAUED**: twelve learned heads at 0.567–0.578; a 4-rollout leaf beats them (+0.057 DETECTED) |
| 5 | search | search over 2 + 4 | win-rate dividend at fixed compute | — | gated on 4; wound down as an objective |
| 6 | ladder | the deployed agent | ladder Elo | — | — |

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
`--team-pfsp`, `--team-block-episodes`, `--pool-spread`, `--pfsp-scale`) serve all three draws at
once. That is the root of several confounds this month, the unregistered `--team-block-episodes`
lever among them.

### 4.1 Our team: the deployment target lives here

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
| 0 | the sample-team split: `sample/` = the 32 Smogon teams; the 40 promoted fleet teams get their own role (dispatched 2026-09-23) | small build | — |
| 1 | the strength/piloting split on existing eval data; is stall a piloting deficit? | free, offline | — |
| 2 | choose the stage-2 teams (~5, one per archetype plus iconic extras) from the 32, with the owner | a conversation | step 1 informs it |
| 3 | finish population-loop round 1 (stage 3 aimed at stage 1): does absorbing exploiter pressure work at all? | ~21 GPU-h, queued | — |
| 4 | first stage-2 arm: G0 fine-tuned on the chosen teams against pressure opponents, read by the anchors on those teams | ~1 arm | a plateau is a result |
| 5 | E1: our-side breadth, sample-only vs + pool vs + pool + procedural | 3 arms | sets §4.1's shares |
| 6 | the narrow-critic question: discrimination on the stage-2 teams vs broad | 1 arm + the offline meter | decides whether stage 5 reopens |
| 7 | the match sampler + ecology block + cell tagging | a build; after the Rust core's M2, where team sources and tagging belong | — |

## 8. Open questions this document does not settle

1. **How narrow?** "Great on five teams, robust against everything" is a different target from "a
   gen3 OU generalist", and it sets how much breadth is worth paying for.
2. **Does the generalist need to keep improving?** If stages 2–5 climb on a fixed G0, stage 1's job
   is "a good starting point and a good sparring partner", which is cheaper.
3. **Should procedural teams ever be on our side?** Novel states we pilot through build
   representation; novel states only the opponent creates build defence. They may be worth
   different amounts.
4. **Should the population loop's target move from the generalist to the main agent** once stage 2
   exists? Probably yes; round 1's answer about absorption comes first.
