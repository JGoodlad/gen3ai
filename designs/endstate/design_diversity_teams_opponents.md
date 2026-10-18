# Diversity of training: our teams, the opponents' teams, the opponents, and the evaluation sets

**Status: DESIGN 2026-10-18 (owner request; planning input; nothing built, nothing launched).** §5 was REWORKED the
same day into an EVALUATION MATRIX on the owner's input. Two zero-game, post-hoc, descriptive reads were run for it:
D0a (§2.4) and the ladder resemblance with its strata weights (§5.0). ALWAYS-CURRENT (owner, 2026-09-27): a later decision,
build or measurement that differs updates this doc in the same commit, and the Decision record (§11) says what
changed and why.

**Trigger.** Three reads of `rb_x26_s2601` @161,000,169 ("X26") against `metamon:Kakuna` ckpt 34 (Metamon's best
public policy), greedy vs greedy on the CPU, n = 400 each:

| team set | in our 719-team pool | X26 win rate [Wilson 95 %] | K2 (G0′, the previous lineage's plateau) | source |
|---|---|---|---|---|
| home72 (our `data/teams/sample`) | 72 / 72 | **0.575 [0.526, 0.623]**, ahead | 0.487 [0.439, 0.536] | [`k2_x26_kakuna_both_teamsets_2026-10-17/`](../research_state/measurements/k2_x26_kakuna_both_teamsets_2026-10-17/README.md) |
| away (Metamon's `competitive`, the set Metamon's README says it overfit to) | 19 / 20 | 0.497 [0.449, 0.546], level (not detected) | 0.355 [0.310, 0.403] | the same |
| heldout72 (72 replay-derived `hl_05_26` teams) | 0 / 72 | **0.395 [0.348, 0.444]**, behind | 0.357 [0.312, 0.406] | [`heldout_teams_kakuna_2026-10-18/`](../research_state/measurements/heldout_teams_kakuna_2026-10-18/README.md) |

X26 − K2 is +0.087 on home72, +0.143 on away and +0.038 [−0.030, +0.104] (not detected) on heldout72.

The owner's reading (2026-10-18, paraphrased): "roughly tied with state of the art: ahead where they haven't seen
the teams, behind where we haven't, tied on shared ones". The owner's questions:

- Should we widen the training pool?
- Should we fork at the plateau, widen the pool, and see what that does to held-out teams AND to the teams we
  train on now?
- Do we train AGAINST unseen teams, or only PLAY them in evaluation?
- Exploiters could play the off-meta teams we need to handle, but then how do we read the evaluations?

This doc answers them, after challenging the reading itself (§8).

**Binding rules this doc is checked against:**

- **PRIORS come only from Smogon** (owner, 2026-08-15, reaffirmed 2026-10-18). Teams used as TRAINING DATA (the
  pool, replay-derived teams) are experience, not priors. Anything the network READS as a prior must trace to
  Smogon. Every lever in §4 is checked against this.
- **A research-PATH claim needs adversarial treatment, counter-tests and replication on a PLATEAUED model**
  (owner, 2026-10-11). Every read at 161M is PROVISIONAL here.
- **One lever per A/B, the cheapest option that is not a regression** (owner, 2026-10-03).
- **`data/` is FROZEN while X26 is live** (and no `data/` change while any pinned run is live, root rule 6).

---

## 0. One screen

- **What we know.** On teams our models never trained on, X26 loses to Kakuna by about 10 points and is not
  detectably better than K2. On our own teams it wins. **That drop is not yet explained.** "Unseen by us" and
  "a different team population, the kind Kakuna's replay data covers" are confounded in the one held-out read we
  have, and its Hidden Powers are an artifact (HP Dark on 164 of 432 mons; the repaired `heldout72_hpfix` re-read
  is PENDING, a parallel agent).
- **What the ladder looks like (§5.0, new 2026-10-18).** Against the sample teams (home72 ∪ `away`), 85 % of the
  22,862 `hl_05_26` ladder-replay teams share ≥ 4 species with one sample team and 18 % share all six; only 0.2 %
  share ≥ 4 full sets (understated by the archive's usage fills). **The ladder is sample-CORE-heavy.** heldout72's
  strata (22 / 61 / 17 % CORE / NEAR / FAR) MATCH the ladder's (18 / 67 / 15 %). It is not off-meta by species;
  it differs from the sample teams in its SETS and its Hidden Power artifact.
- **D0a and the strata, read with zero games (descriptive, post hoc).**
  - heldout72 is not stall-heavy.
  - Only 9 of its 72 teams carry an entity the pool never had.
  - X26 piloting held-out teams scores 0.43 on sample-like cores and **0.25 on off-meta (FAR) cores**, against
    0.575 on home72.
  - So unfamiliar SETS on familiar cores already cost most of the drop, and off-meta species cost more on top.
- **The evaluation is a MATRIX, honest about bias (§5, reworked on the owner's input).**
  - **(i)** Every team set gets its bias direction: home72 toward us, `away` toward Kakuna, heldout72 toward Kakuna,
    smogon_forum near-neutral, P72 unclear.
  - **(ii)** The PILOT vs FACE cells, with the owner's hypothesis "we don't pilot off-meta teams well" as the named
    test **H-PILOT**.
  - **(iii)** A LADDER-WEIGHTED estimate **L**: we bring our sample teams, and the opponent's teams are weighted
    CORE 0.176 / NEAR 0.675 / FAR 0.149 (`ladder_weights_v1`).
  - **(iv)** **GEN-M**, an in-house meter with no external opponent.
  - **(v)** **THE headline is L**, always beside the sample ground, the unfamiliar panel, GEN-M and L's CORE
    sensitivity.
  - **(vi)** Freeze, then widen.
- **The first experiment: X37a, now two CPU stages, 0 GPU.**
  - **Stage 1** reads the matrix's core (≈ 3.5 lane-hours; the calibration cell needs B-D9):
    - the DEPLOYMENT cell by stratum: we pilot home72, Kakuna pilots heldout72_hpfix's CORE / NEAR / FAR teams,
      600 games;
    - the REVERSE cell: we pilot heldout72_hpfix, Kakuna pilots home72, 400 games;
    - the strength calibration K(P, H): Kakuna vs Kakuna, 400 games.
  - **Stage 2** reads the third-source rows (≈ 2 lane-hours + the P72 build, ≈ 0.25 agent-day): P72 and
    smogon_forum symmetric, 400 games each.
  - The parallel `heldout72_hpfix` read supplies S(H, H).
- **The levers (§4), ranked by expected information per GPU-hour:**
  - **D1 WIDEN fork at the plateau.** The pool plus about 22.7k HP-repaired `hl_05_26` teams, minus every frozen
    panel and its near neighbours, at a 50 % draw share on both sides, against X35's shared control C0
    (**≈ 13 GPU-h**).
    - **Primary:** gain on the UNFAMILIAR cells.
    - **Co-primary:** no loss worse than −2.0 pp on sample-team ground, measured in-house.
  - **D5 LEAVE-OUT** (**≈ 5.6 GPU-h**) is the clean memorisation test.
  - D2 / D3 / D4 / D7 are conditional.
- **Train against unseen teams, or only play them?** Both, on DISJOINT sets (§4.3): freeze the panels first, train
  on a wide slice that excludes them and their near neighbours, and never train on a panel's teams.
- **Where this sits in X35 (§6).** D1 ranks **2nd**, after A1 and before A2, unless X37 points squarely at
  Kakuna's home turf.
- **Where I disagree with the owner (§8).**
  - "Roughly tied with state of the art" is one PROVISIONAL checkpoint read against confounded sets.
  - The held-out replay teams are NOT unfair because "the ladder is mostly sample teams": their species cores match
    the ladder's.
  - "We are matched on sample teams even though we don't bias toward them": we DO bias toward them, at ≈ 2.1×
    piloting exposure on home72.
  - "More diversity favours Kakuna": diversity penalises whoever memorised a closed set, which is us. It favours
    Kakuna only if its data covers that population (UNVERIFIED).

---

## 1. Vocabulary

| term | meaning here |
|---|---|
| **pool** | the 719 teams `TeamLoader` reads from `data/teams/`. It holds the Smogon sample set (`data/teams/sample/`, 72 valid teams = home72) and five community collections in `data/teams/others/` (giraffe 396, yak_attack 185, johnnyg2 100, mcmegan 36, schmuck_nick 21 files; 719 valid after validation and de-duplication) |
| **mon set** | species + sorted moves, with Hidden Power typed (`gen3_team_overlap_v1`, `main.anchors.team_overlap`). The `strict` tier adds item, ability, nature and EVs (effort values) |
| **near neighbour** | a team that shares ≥ 4 of 6 mon sets with one reference team (the draw rule of heldout72) |
| **held-out** | unseen by the model being read, as exact teams AND near neighbours. Held out FROM WHOM is always stated |
| **population** | the process that made a team set: Smogon forum builders (sample), community archives (others), ladder replays with usage-filled gaps (`hl_05_26`), the Smogon-usage generator (procedural) |
| **piloting / facing** | our side playing a team / our side playing against a team |
| **pp** | percentage points |
| **CORE / NEAR / FAR** | a team's best SPECIES overlap with any one sample team (home72 ∪ `away`): all 6 / 4–5 / ≤ 3 (§5.0) |
| **S(X, Y)** | our win rate when WE pilot team set X and the opponent pilots set Y (§5.2) |
| **L** | the ladder-weighted deployment estimate, THE headline (§5.3, §5.5) |
| **GEN-M** | the in-house generalisation meter, no external opponent (§5.4) |

---

## 2. Four axes of diversity, and a fifth that couples them

Every game is (our team) × (who the opponent is) × (the opponent's team): `design_ladder_campaign.md` §4's "three
draws". Evaluation is the fourth axis. The fifth axis is the coupling between the first three under self-play.

### 2.1 Axis A — OUR side's team diversity

**Today:**

- The trainee's builder is `MatchupSpec.from_args` → `TeamSource(kind="default_biased")`
  (`src/agents/training/matchup_spec.py`). It draws the full 719-team pool, plus a 10 % extra draw from the
  sample teams (`DEFAULT_TRAINEE_BIAS_PROB = 0.1`).
- **So a home72 team is piloted about 2.1× as often as any other pool team:** 0.1/72 + 0.9/719 against
  0.9/719 alone.
- Teams are validated and their Hidden Power IVs fixed by `Gen3Teambuilder` (`src/utils/teambuilder.py`).
- **Levers that exist:**
  - `--trainee-team` / `--trainee-teams` (specialist pins);
  - `--team-block-episodes` (default 1 = off);
  - `--team-wr-tracking` (per-team win rates, instrumentation only).
- **Levers that do not exist:**
  - team-PFSP (prioritised fictitious self-play over teams) was Python-only and was deleted in L4;
  - no learnability sampler (prioritised level replay, PLR; `design_ladder_campaign.md` §5) is built;
  - no procedural or replay teams are ever piloted.

**Evidence.**

- Count dominates conditioning: fewer teams at the same budget made a STRONGER exploiter (+7.7 pp, pre-era).
- Fine-tuning on five teams came out level, then worse at 1.78× the dose (stage A, `design_ladder_campaign.md`
  §2a).
- The strength/piloting split of `design_ladder_campaign.md` §5.1 was never run.
- The exposure ordering matches X26's Kakuna scores. Trainee exposure runs home72 (≈ 2.1×) > away (19 teams that
  appear as 23 pool entries, ≈ 1.2×) > heldout72 (0). X26 scores 0.575 > 0.497 > 0.395. This is DESCRIPTIVE:
  the sets also differ in population and in Kakuna's familiarity (§3).

### 2.2 Axis B — the OPPONENT's team diversity

**Today:**

- Every opponent draws the pool uniformly (`TeamSource(kind="pool")`): self-play snapshots, stable opponents,
  exploiters' targets and the bot floor.
- The best-response readers pin archetype sets of POOL sample teams (`exploiter_teamsets_2026-09-22.json`).
- Three team sources exist, but only as TEST sources, never in training (`src/utils/team_sources.py`):
  - the pool;
  - the procedural Smogon-only generator (`src/rust_sim/harness/ou_random_teams.js`, seeded,
    `TeamValidator`-legal);
  - the LADDER corpus (`src/utils/ladder_corpus/`, 22,813 engine-playable `hl_05_26` teams, under `src/` so that
    no pinned run reads it).

**Evidence.**

- **The belief heads memorise the pool** (ledger 2026-09-24, on `ai_v13_22`, PRE-ERA):
  - off the pool, the learned species head is WORSE than the Smogon prior it fuses with (−1.28 nats on ladder
    teams);
  - the learned belief deltas buy +4.8 pp on pool opponents and nothing detectable on ladder opponents.
  - X26's beliefs are X5's hypothesis tokens. They are the same channel: a learned delta on the T0 Smogon
    prior, `HypothesisBuilder.delta`, trained on pool labels. **UNVERIFIED on X26.**
- **On our CLOSED pool, identifying the opponent's TEAM is worth most of the oracle's edge.** The Bayes posterior
  over the 719 teams recovers 0.80 of the oracle − honest gap (`search_ledger.md` W12). The pool's per-species
  set statistics recover nothing (POOLSETS −0.02). That information exists only inside a closed pool. **A policy
  that learned to use it implicitly loses it on any unseen team.** This is the mechanism behind H2 (§3).
- **The pool's sets are as predictable from Smogon usage as real human sets** (`design_hidden_info_binding.md`
  §16.2). So the pool is not un-Smogon-like at the set level. What is special about it is that it is CLOSED.

### 2.3 Axis C — the opponent POLICY's diversity

**Today:**

- A self-play pool of the 20 most recent snapshots (`SnapshotPool.DEFAULT_MAX_SNAPSHOTS`), with `--pool-spread`
  available.
- PFSP (`--pfsp-scale`, default 0) and `--n-sentinels` 5.
- The bot floor: nine pinned bots, Rust ports.
- Stable opponents (`--stable-opponents`, with a share, PFSP and mastery-to-floor), and `--exploiter`.
- X35 stages the AlphaStar-style population levers (`design_compute_efficiency.md` §5.6):
  - A1, an archive pool;
  - A2, PFSP;
  - A3 / A6, exploiters;
  - A4, a trailing magnet;
  - A5, Nash weights.
- External agents (Metamon's policies, Foul Play) are EVALUATION-only.

**Evidence.**

- X26's plateau mechanism so far is a slow transitive climb with a small, real cycle. The Hodge excess width is
  32 Elo, p = 0.005 (`design_collapse_and_counterfactuals.md` §3.4).
- PFSP over a homogeneous self-pool cost 26–33 Elo pre-era.
- The population loop absorbed its specialists, but the exploitability gap's fall was not detected (rounds 1–2).
- **Every policy-diversity lever in X35 still draws the POOL's teams.** None of them can close a team-diversity
  gap by construction.

### 2.4 Axis D — evaluation diversity, and D0a

**Today:**

| instrument | its teams | what it measures |
|---|---|---|
| in-run eval cycle (every 2M steps: 9 bots + 5 greedy sentinels × 100 games) | the trainee's builder (pool + sample bias) | in-pool strength; the trend |
| snapshot ladder (`ladder.json`), the Tier-1 plateau test (`main.plateau`) | pool | in-pool progress |
| `main.h2h` / `cross_stat`, X35's population score R1 | pool (no team-set flag) | in-pool strength against a population |
| untaught meter (`main.untaught_meter`) | any manifest, default the "untaught 8" (POOL teams: "untaught" means no teacher was trained on them; the generalist pilots them). The opponent always draws the pool | piloting a chosen team slice; paired games across checkpoints; teams as the cluster |
| `main.best_response_gap` | pinned pool archetype sets | exploitability |
| external anchors (`main.anchors`) | home / home72 (pool), away (19 / 20 pool), heldout72 (0 / 72), any config | strength against an agent nobody here trained; `team_overlap` is a REQUIRED field since `fda1efda` |

🚨 **No in-run meter, no X35 readout component and no plateau test reads a team our models have not trained
on.** X35's external-anchor guard G-K reads home72 and away, both training data (FINDING F-DV-1).

**D0a — the three anchor sets read with ZERO games** (descriptive, post hoc, our piloted team only; script and
output: [`measurements/diversity_d0a_2026-10-18/`](../research_state/measurements/diversity_d0a_2026-10-18/README.md)):

| | home72 | away | heldout72 | pool (719) |
|---|---|---|---|---|
| archetype mix, rule-based (`team_archetypes.classify_team`): HO / O / B / SS / S | 11 / 19 / 16 / 12 / 14 | 1 / 5 / 5 / 5 / 4 | **7 / 18 / 25 / 18 / 4** | 139 / 155 / 206 / 129 / 90 |
| teams with ANY species / move / item / ability no pool team carries | 0 | 0 | **9 / 72** (7 mons with an unseen species, 7 unseen move slots, 1 ability, 0 items) | — |
| species-move pairs absent from the pool | 0 | 0 | 203, **of which 160 are Hidden Power** | — |
| HP Dark mons | 0 | 0 | 164 / 432 | — |
| median game length, X26 / K2 (turns) | 30 / 33 | 37 / 36 | **40 / 41** | — |

(HO / O / B / SS / S = hyper offense / offense / balance / semi-stall / stall.)

Inside heldout72, splitting each model's 400 games by OUR team's covariates:

- **HP Dark mons on our team** (0–1 / 2 / ≥ 3): X26 0.447 / 0.410 / 0.352; K2 0.426 / 0.389 / 0.290.
- **≥ 1 species the pool never had:** X26 0.25 (32 games) vs 0.41; K2 0.16 vs 0.38.
- **Mon sets shared with the closest pool team** (0 / 1 / 2): X26 0.338 / 0.411 / 0.402, not monotone.

Every interval overlaps (Wilson per game, optimistic because teams repeat). The splits are confounded with team
strength: an incoherent HP-Dark set is a weaker set for whoever pilots it. **Read as: nothing here identifies
familiarity, the artifact is a live candidate, and the set is not stall-heavy.**

### 2.5 The coupling: under self-play, opponent-team pressure is only as good as our piloting

Only the trainee's decisions are trained. The opponent in self-play is a past copy of us, so it pilots the
opponent's team only as well as we pilot that kind of team. Widening ONLY the opponent's teams therefore produces
off-meta opponents piloted by a policy that never practised them. That is weak pressure: "pilots must play those
teams WELL or the pressure is fake" (`design_ladder_campaign.md` §4.2). **This is why D1 widens BOTH sides (one
lever: the team distribution the builders draw from) and leaves the side-attribution to a conditional D2.**

---

## 3. What the held-out gap is: competing hypotheses, each with a kill test

The drop to explain is X26 heldout72 − home72 = **−0.180 [−0.247, −0.111]** (K2: −0.130). X26 away − heldout72 is
−0.102 [−0.170, −0.034]. K2 away − heldout72 is +0.003.

| # | hypothesis | mechanism | what it predicts | KILL TEST (cheapest first) | evidence so far |
|---|---|---|---|---|---|
| H1 | **POLICY memorisation, piloting** | the trunk and policy learned team-specific plans for the 719 teams; on a new team its plans are generic | X26's TRAINING PROGRESS transfers less to piloting unseen teams than pool teams | **D0e-pilot** (§3.2): X26 @25M → latest piloting heldout72 / P72 / home72 against a fixed reference, paired games. KILLED iff the transfer deficit T's 95 % interval lies wholly below +2.0 pp | descriptive only: the exposure ordering 2.1× > 1.2× > 0 matches 0.575 > 0.497 > 0.395 |
| H2 | **FACING memorisation: implicit TEAM IDENTIFICATION** | from 1–3 reveals the policy infers WHICH pool team it faces and plays to its unseen members (W12: worth 0.80 of the oracle gap in-pool) | X26's progress transfers less to FACING unseen teams; the deficit is largest early in the game (few reveals) | **D0e-face**: the reference pilots the set's teams against X26 @25M → latest. Then the same by reveal count, from the eval traces. KILLED iff the facing deficit's interval lies wholly below +2.0 pp | W12 (in-pool information exists); no off-pool read |
| H3 | **BELIEF-HEAD memorisation** (learned deltas over the Smogon priors encode the pool) | species, move and item posteriors are miscalibrated off-pool; the policy acts on a wrong world | zeroing the learned deltas (prior-only) costs a lot on the pool and nothing, or HELPS, on held-out teams | **D0f**: the 2026-09-24 prior-only DiD (difference in differences), re-run on X26 (a build). KILLED as a major cause iff the DiD's 95 % upper bound < +3.0 pp AND prior-only − full on heldout72 is not detected above 0 | pre-era: DiD +4.8 pp [+1.9, +7.8]; harmful calibration off-pool. Its size caps H3 at about a quarter of the drop |
| H4 | **POPULATION shift** (off-meta / out-of-distribution SETS and play styles: replay teams with usage-sampled fills; longer games) | X26 is bad at this KIND of team whether or not it saw the team | the drop is the same on unseen teams of this population (heldout72) and ABSENT on unseen teams of another population | **X37a stage 2** (§3.1): P72 and smogon_forum. KILLED iff X26(P72) − X26(heldout72) is NOT detected AND X26(P72) − X26(home72) is detected below 0 | D0a: not stall-heavy; games 25 % longer; 21 / 72 teams share all six SPECIES with a pool team but no set |
| H5 | **KAKUNA's home turf** (it trained on parsed ladder replays of this population; UNVERIFIED) | Kakuna is relatively STRONGER on heldout72; our "drop" is partly its gain | the in-house transfer reads (no Kakuna) show NO deficit while the Kakuna drop stays | **D0e** (H1/H2's reads are Kakuna-free) + **X37a stage 2** (P72, smogon_forum). KILLED as the SOLE cause iff D0e's deficit (piloting or facing) has a 95 % lower bound ≥ 5.5 pp, half the Kakuna drop's smallest plausible size (11.1 pp) | partial: `away` is the set Kakuna overfit to, and X26 does 10 pp BETTER there than on heldout72. So the drop is not Kakuna-familiarity alone, ASSUMING its familiarity with `competitive` ≥ with `hl_05_26`-like teams (UNVERIFIED) |
| H6 | **the Hidden Power ARTIFACT** (HP Dark from blank IVs on 164 / 432 mons) | (a) our belief stack predicts Smogon's HP types and mis-reads their coverage; (b) the sets are incoherent (special HP Dark on physical attackers), which changes play; (c) Kakuna's own data carries the same artifact, so it is calibrated to it (UNVERIFIED) | the repaired set reads detectably higher for X26 | **`heldout72_hpfix` re-read** (parallel agent, PENDING; not duplicated here). this doc reads H6 as KILLED iff X26 hpfix − heldout72 is not detected and its 95 % upper bound < +5 pp (the parallel read's own registered rule governs its verdict) | D0a: scores fall with HP Dark on our team in BOTH models (confounded with strength) |
| H7 | **team-building QUALITY / strength** (the held-out teams are weaker or stronger) | both sides draw the same set, so pure strength cancels in expectation; only an INTERACTION remains (noisier teams compress toward 0.50) | compression toward 0.50, never a sign flip | **already KILLED as a sole cause:** X26 fell PAST 0.50 (0.575 → 0.395). As a contributor: the PILOT / FACE cells and their strength calibration (§5.2) | logic |
| H8 | **ENTITY coverage** (species / moves the pool never had; their embeddings are barely trained) | the static token reads an id embedding for species and items, and a move network over ids + properties | worse play on teams that carry unseen entities | **D0a** (done): only 9 / 72 teams carry any unseen entity. At most a small share of the drop. Also D5's split | KILLED as a major cause (D0a coverage); the 9 teams may still be locally bad (X26 0.25 on 32 games) |
| H9 | **statistics / one checkpoint / pre-plateau** | a fluctuation of one checkpoint, one seed | the drop vanishes at other checkpoints | **D0d / D0e trajectory** (≥ 3 checkpoints). Already partly killed: K2, another lineage at ITS plateau, drops 13 pp too | K2 replicates the sign |
| H10 | **GIGO off-pool** (a reading or encoding bug that only off-pool content reaches: the Heal Bell crash, the Transform + Hidden Power tracker crash, Imprison, ...) | a mis-encoded fact degrades decisions silently | divergence between the core frame and the live reader on held-out games; anomalies by entity | the existing differential gates at the ladder corpus's FULL tier (`--ladder-tier full` soak), and `ladder_drift_scan`-style per-entity scans on the heldout72 games. KILLED iff 0 divergences over the soak | 800 held-out games ran clean (0 protocol failures, every unit OK) |
| H11 | **the 250-turn forfeit and draws** | longer games hit the forfeit more often | more forfeits on heldout72 | rows: X26 hit it 4 times in 400 (1.0 %) on heldout72, against 1–4 elsewhere | KILLED as a major cause |

H1–H3 are not exclusive: they are three channels of one phenomenon, "the model knows the CLOSED pool". H4–H6 are
the alternatives. **What decides between the two families is the 2 × 2 below, which needs a third source.**

### 3.1 The third source, and the 2 × 2 it completes

| | **set is Kakuna-familiar** (its kind is in Kakuna's data or eval: UNVERIFIED either way) | **set is NOT Kakuna-familiar** |
|---|---|---|
| **seen by us** | away (0.497) | home72 (0.575): Smogon sample teams, 8 / 72 exact copies of `competitive`. Kakuna-familiar in kind? UNVERIFIED |
| **unseen by us** | heldout72 (0.395); heldout72_hpfix PENDING | **THE MISSING CELL: the third sources** |

**The third sources (owner decision Q4, 2026-10-18: no owner or friends' teams, whose quality is too low):**

1. **P72, procedural** (`ou_random_teams.js --emit 72 --seed <registered>`). Smogon-only and validator-legal.
   Unseen by us by construction, checked by `team_overlap` against the pool, `competitive` and `hl_05_26`. It costs
   ≈ 0.25 agent-day to build (B-D4). **Caveats:**
   - not human-built: less coherent teams, so team quality differs;
   - its sets ARE the Smogon usage our belief priors encode;
   - Metamon's `paper_variety` set is also usage-generated, and whether Kakuna trained on it is UNVERIFIED.
2. **smogon_forum**, a Smogon-forum ADV OU team collection, built by another agent into
   `designs/research_state/measurements/smogon_forum_teams_2026-10-18/` (**PENDING**). It is human-built and unseen
   by us. **The condition for using it:** its exact copies and near neighbours of `competitive` (the forum is
   `competitive`'s own source) and of the pool must be excluded, or it reads `away` again. It is never used for
   training (§5.6).

**The cells to read are X37a's (§6.3), in two stages:**

- **Stage 1, the matrix's core (§5.2, §5.3):**
  - the DEPLOYMENT cell by stratum (600 games);
  - the REVERSE cell (400);
  - the strength calibration K(P, H) (400).
- **Stage 2, the third-source rows:** P72 and smogon_forum symmetric, 400 games each.
- All of it on paired seeds, at ≈ 9 s per game, CPU only (rule 5b).

**The third-source decision rule** (pre-registrable; deterministic; Newcombe 95 % on win rate; n = 400 each).
Let H = heldout72_hpfix. For each third source T ∈ {P72, smogon_forum}: d1 = X26(T) − X26(H) and
d2 = X26(T) − X26(home72).

| class | condition | meaning |
|---|---|---|
| **UNSEEN-DRIVEN** | d1 NOT detected AND d2's upper bound < 0 | X26 loses on unseen teams of another population too: H1–H3 lead; widening (D1) is the lever |
| **POPULATION / KAKUNA-DRIVEN** | d1's lower bound > 0 AND d2 NOT detected | unseen teams of another population are fine: H4–H6 lead; D1's expected value is lower |
| **MIXED** | d1's lower bound > 0 AND d2's upper bound < 0 | both |
| **UNDECIDED** | otherwise, or any bound within 1e-9 of 0 (rule 8) | — |

**When the two third sources disagree, smogon_forum governs** (human-built, near-neutral, §5.1). P72's verdict is
reported beside it.

### 3.2 The in-house TRANSFER reads (no Kakuna): D0e

**Instrument:** `main.untaught_meter`, which exists today.

- Each ref pilots a pinned team from a manifest.
- The FIXED opponent plays the flat pool.
- Games are keyed so that every ref plays the same games (common random numbers).
- Teams are the cluster, and one resampling index set is shared by every contrast.

**The design:**

- **D0e-pilot:** refs = X26 @{25M, 100M, 161M, latest 10M-multiple}; the opponent = the registry baseline
  `production` (`rb_es_end2_s2005`, 15M), greedy. Four manifests (home72, heldout72_hpfix, smogon_forum, P72), 10
  games per team; the heldout72_hpfix manifest also split by stratum (CORE / NEAR / FAR, §5.0).
- **D0e-face:** the same refs as OPPONENT and the reference as PILOT. One meter run per checkpoint, because the
  meter has one opponent per run.
- Both run at a commit where every ref loads (the anchors reads used a detached `63f97a57`; X26's own pin is
  `9c033489`), recorded in the read.

**The statistic.** g_S = score(latest) − score(25M) on set S, paired, with the team-cluster bootstrap. The transfer
deficit is T_S = g_home72 − g_S.

| class | condition |
|---|---|
| **TRANSFERS** | T_S's 95 % interval lies wholly below +2.0 pp AND g_S's lower bound > 0 |
| **PARTIAL** | T_S's lower bound > 0 AND g_S's lower bound > 0 |
| **MEMORISATION-DOMINATED** | T_S's lower bound > 0 AND g_S's lower bound ≤ 0 |
| **UNDECIDED** | otherwise; a bound within 1e-9 of a boundary is UNDECIDED |

**Cost.** ≈ 4 refs × 4 sets × ≈ 720 games ≈ 11,500 games per axis, ≈ 2.7 lane-hours per axis at the h2h CPU rate of
≈ 1.2 games/s (`design_x26_readiness.md` §2.6; UNVERIFIED for the meter).

**Why this read matters most.** It is the one read that can say whether X26's continued pool training, the plateau
the programme is waiting for, is buying held-out strength at all (§8.4).

### 3.3 The other cheap reads

| id | read | cost | prerequisite |
|---|---|---|---|
| D0a | the zero-game set description (§2.4) | **DONE** | — |
| D0d | X26 @{60M, 110M, latest} vs Kakuna on home72 and heldout72, 200 games each: the Kakuna-side trajectory of the gap | ≈ 3 lane-h | — |
| D0f | **prior-only DiD on X26.** A copy of the checkpoint (scratch dir, never `models/`) with `hypothesis.delta_out`, the move-belief learned delta and the item-belief delta zeroed: by construction the cold-start function, which equals the Smogon prior. The 2026-09-24 design (paired battles, the opponent's TEAM varies) on home72 vs heldout72 vs P72 | ≈ 0.5 agent-day build (opus-medium: the surgery tool + an equality test that the zeroed model's beliefs equal the prior on fixed rows) + ≈ 2 lane-h | — |
| D0g | **the toy** (T-ENT, entity tokens + attention, NO belief stack, NO op) @100M vs Kakuna on home72 and heldout72, 200 games each. If its drop (logit scale) matches X26's, the belief machinery is not the channel | ≈ 1 lane-h | `main.anchors` loads the toy at HEAD (UNVERIFIED; X34's staging proved the toy loads strictly at HEAD) |
| D0h | `heldout72_hpfix` | parallel agent | PENDING; not duplicated |

---

## 4. The levers

GPU costs use measured throughput:

- X26 late-run ≈ 1,334 steps/s including eval, so a 20M fork ≈ **4.4 GPU-h per seed and ≈ 13 GPU-h per
  3-seed arm** (`design_compute_efficiency.md` §5.6.1);
- production fresh at 30M ≈ **5.6 GPU-h** (X33's budget table);
- the toy ≈ **5.8 GPU-h per 100M**.

CPU reads at ≈ 9 s per Kakuna game on one rule-5b lane.

### 4.1 L1 — WIDEN the training pool (how many, from where, validated how)

**Mechanism.**

- A closed 719-team pool lets three things key on team identity: the policy, the beliefs and the value. W12
  says doing so is worth a lot in-pool.
- A pool of ~23k teams from an open population makes exact identification rare. It forces set- and
  species-level inference, which is what transfers.
- In the literature, procedurally generated RL environments show the train / test gap closing as the number of
  training LEVELS grows into the thousands, and early overfitting on hundreds (Cobbe et al. 2019, CoinRun;
  Cobbe et al. 2020, Procgen; **UNVERIFIED**: from memory, the exact counts not checked). Random-battle formats
  remove team memorisation by definition (Wang 2024 trained on gen4 random battles).

**From where (ranked).**

1. **`hl_05_26` (W_hl):** the ladder corpus already on the box (`src/utils/ladder_corpus`, 22,813 engine-playable,
   validator-legal teams), MINUS the exclusions below. These are real players' revealed sets from the 1400+
   ladder and smogtours replays, with the gaps filled by usage sampling. It is the population a ladder brings.
2. `gl_05_26` (the general ladder): not on the box, UNVERIFIED availability. A later dose step.
3. Procedural teams: they belong to L5 (augmentation), not here. They are synthetic.
4. **NOT smogon_forum or P72.** They are evaluation panels (§5.6), and smogon_forum is the matrix's only
   human-built, near-neutral source.

**How many.** All of W_hl that survives the exclusions (≈ 22.7k, UNVERIFIED until built). It is mixed at a DRAW
SHARE s = 0.5 per side: each draw comes from W_hl with probability 0.5, else from the pool with today's 10 %
sample bias on the trainee side.

- **The share is the lever, not the count.** A 20M fork plays ≈ 570k games (≈ 35 decisions per game), so at
  s = 0.5 each W_hl team is drawn ≈ 13 times per side, and each pool team ≈ 400 times.
- Dose follow-ups: s ∈ {0.25, 0.75}, and the count at a fixed s.

**Validated how.** Every step runs in a corpus BUILD, offline. The output is a versioned, pin-safe corpus with a
manifest and sha256s, like `src/utils/ladder_corpus`.

1. **Hidden Power repair per the OWNER SPEC** (2026-09-25, `TECH_DEBT_BACKLOG` (f) P1, `TASK_BACKLOG` T8):
   - KEEP the guessed EV spread and nature;
   - choose an HP type SYNERGISTIC with the set's physical / special side (in gen 3 the category follows the
     TYPE), sampled from **Smogon's** per-species HP-type usage restricted to that side;
   - write the IVs from the validator's template (`sim/team-validator.ts:1154-1164`, `data/typechart.ts`
     `HPivs`);
   - verify with `dex.getHiddenPower` and the pinned validator;
   - pin a speed-tie case for the 30-Speed-IV types.

   The parallel `heldout72_hpfix` repair is the first implementation (`make_heldout72_hpfix.py`). T8
   generalises it to the corpus, so the two must be one function (FINDING F-DV-4).
2. EVs / natures: kept as filled (owner spec). They are usage fills, not human truth. The belief heads will learn
   the fill distribution on these teams, which pulls the learned deltas toward the prior (compare §4.7).
3. The pinned validator (`gen3ou`, Showdown pin `e0551883f`) on the REPAIRED text; engine-playable (the corpus
   filter).
4. **EXCLUSIONS:**
   - (a) every team of every registered held-out panel (§5.6), and every NEAR NEIGHBOUR of one (≥ 4 shared mon
     sets): `hl_05_26` holds repeat teams from the same players, so an exact-only exclusion leaks;
   - (b) teams that reach OPEN off-pool crash classes: Transform + Hidden Power (TECH_DEBT (a) P1: 20 corpus teams;
     a training crash, not a skipped battle) and Imprison (a possible livelock, UNVERIFIED). They are excluded
     until fixed.
5. A full-tier soak through the Rust env, in the emission self-check build, before any launch: 0 crashes, 0
   divergences.

**Where it lives (FINDING F-DV-2).** NOT under `data/teams/`:

- `TeamLoader` walks `data/teams/` RECURSIVELY for every `teams.json`.
- A pinned run reads `data/` from MAIN.
- So a new directory there WIDENS X26's own pool at its next 6-hour restart, silently. The same holds for X34
  and every X35 arm after it.

**Recommended:** a versioned corpus under `src/` (the ladder corpus's own reason for living there: pinned code
reads `src/` from its pin) behind a new `MatchupSpec` source kind and one flag, e.g. `--team-mix
<corpus>:<share>`. The alternative is TECH_DEBT (f) P1's fix (pinned children resolve `data/` through the pin
tree), landed BEFORE X34 launches. **Owner / orchestrator choice (Q2, §9).**

**Owner rule check.**

- The teams are TRAINING DATA (experience). Nothing the network reads is fitted on them.
- The HP repair samples SMOGON usage.
- 🚨 Forbidden: refitting any `data/` prior (species usage, teammates, moves) on W_hl. The corpus must never
  feed `tools/compute_priors.py`.

**Risks.**

- In-pool regression: the pool's share halves. That is the owner's worry, and D1 measures it.
- Fill artifacts (EVs / natures) taught as truth.
- Crash classes off-pool.
- A pin hazard if the corpus is placed wrong.
- Validation time at startup for ~23k teams: `validate_teams_locally` through Node, UNVERIFIED. Pre-validate in
  the build and assert the stamp.

### 4.2 L2 — the FORK at the plateau that widens the pool: D1

**Design.** D1 is an X35 arm (§6), so it uses X35's frame (`design_compute_efficiency.md` §5.6.1):

- the start is P\*, X26's plateau checkpoint;
- the batch pin is `X35_PIN_k` (which must carry the §6.2 builds);
- the recipe is X26's own record;
- `--behaviour-check warn`;
- 20M steps;
- seeds 3501–3503, then 3504–3505 at look 2.

b-ON adds `--team-mix <W_hl corpus>:0.5`. b-OFF is X35's shared control **C0**, the same P\*, pin and seeds.
Nothing else changes.

**The readout. Its primary is NOT X35's R1**, because R1 scores pool teams only and would call a successful
generalisation lever a small LOSS.

| role | statistic | rule |
|---|---|---|
| **PRIMARY H** (gain on the UNFAMILIAR cells, §5.6) | Δ_H = ON − OFF score vs Kakuna greedy, pooled over the symmetric UNFAMILIAR cells: `heldout72_hpfix` (W_hl's population, unseen teams), smogon_forum and P72 (populations no arm trains on), equal weights per panel. Each seed final plays 7 role-balanced 20-game units per panel (140 games), so 3 seeds give 420 per panel and 1,260 per arm; unit seeds 0–6 are shared across arms. Newcombe 95 % on the pooled score | **GAIN** iff the lower bound > 0 AND ≥ 2 of 3 seeds have Δ > 0; **LOSS** iff the upper bound < 0; **NULL** iff the interval lies inside [−3.0, +3.0]; else **UNDECIDED** → X35's look 2 at 5 seeds iff Δ̂ > 0. A bound within 1e-9 of a boundary is UNDECIDED |
| **CO-PRIMARY P** (the owner's "what it does to the teams we are currently training on": no loss worse than −2.0 pp on SAMPLE-TEAM ground) | Δ_P = ON − OFF of the population score on the home72 panel IN-HOUSE (each final against X35's frozen reference population, mirrored pairs on home72 via B-D7; Welch over seed finals at 97.5 %). X35's R1 on the full pool is reported beside it | **NON-INFERIOR** iff the lower bound > −2.0 pp. A Kakuna read cannot resolve this bar at an affordable n (F-DV-12) |
| reported | Δ on each unfamiliar panel separately; **L** and **L_c1** ON − OFF (§5.3); GEN-M's class on each final (§5.4) | gain on `heldout72_hpfix` alone = "learned the population"; gain on smogon_forum / P72 too = "generalises beyond its population" |
| reported | Kakuna on home72 and away, ON − OFF | in-pool external read; no rule |
| reported | the D0e transfer read on each final | in-house held-out piloting and facing |
| guards | X35's G-E (exploitability), G-C (the collapse battery), G-F (head-to-head frontier) | as registered there |

**Verdicts:**

- **ADOPT-CANDIDATE:** H GAIN and P NON-INFERIOR, with the guards passing.
- **TRADE:** H GAIN and P inferior. Report the EXCHANGE RATE (held-out pp gained per in-pool pp lost); the owner
  decides.
- **NULL / LOSS / UNDECIDED** after look 2.
- Nothing is adopted by this rule alone.

**Power.** At p ≈ 0.40 and 1,260 games per arm (three panels), the Newcombe half-width is ≈ 3.9 pp. That is enough
for a recovery of a quarter of the 18-pp drop. More games cost CPU only.

**Cost.**

- ≈ **13 GPU-h** (3 seeds × 4.4), with C0 shared with the X35 batch. Unshared: +13.
- CPU: the Kakuna unfamiliar panel, ≈ 2 × 1,260 games ≈ 6.3 lane-h; L's three deployment sub-cells on each arm
  (≈ 2 × 600 games ≈ 3 lane-h); the in-house sample ground and GEN-M ride B-D7 (≈ 1 lane-h per final).
- Builds: §6.2.

**Risks.**

- A plateaued network may have lost plasticity, so the fork cannot unlearn the pool's crutches. Then D1 reads
  NULL and the question becomes a fresh wide run (X9's original form, ≈ X26-scale GPU), which is deferred.
- The C0 control drifts on its own: X35's matched-noise control handles that.

### 4.3 L3 — train AGAINST unseen teams, or only PLAY them in evaluation?

**Both, on disjoint sets.**

- **Playing unseen teams in evaluation MEASURES generalisation but never improves it.** It is the only
  instrument that can say whether a lever helps, so it must exist first and stay clean.
- **Training against teams improves play on THEIR population, and it destroys them as a held-out
  instrument.** A team set used in training is held out from nobody.

**So:**

1. Carve and FREEZE the held-out panels and the ladder weights first (§5.6).
2. Then train on a wide slice that EXCLUDES the panels AND their near neighbours.
3. Read gains on the panels.

**Never train on the evaluation teams themselves.** That is teaching to the test: the number would rise and mean
nothing.

**Which population to train on is a choice of target.** The ladder brings the open population (heldout72's kind),
so training on it is training toward deployment. Keeping one panel from a source NO arm trains on (P72, smogon_forum) is
what tells "learned this population" from "learned to generalise".

### 4.4 L4 — exploiters that play OFF-META teams, and how to read evaluations when the opponent distribution is the treatment

**Mechanism.** This is the team-axis version of an AlphaStar main exploiter, or of unsupervised environment design:
an adversary chooses the hardest environments for the learner (PAIRED, Dennis et al. 2020; ACCEL, Parker-Holder et
al. 2022; **UNVERIFIED**: from memory). Here, a best-response policy pilots a set of off-meta teams against P\*,
and P\* then trains with that exploiter in its stable slice (A3's recipe).

**Today it is REFUSED by design (FINDING F-DV-3).** `validate_exploiter_trainee_is_sample` makes a pinned exploiter
team that is not a curated SAMPLE team a startup FATAL. An off-meta exploiter therefore needs a DECLARED off-meta
manifest that the gate accepts by content sha: a build, ≈ 0.25 agent-day.

**How to read the evaluations (the owner's worry).** The rule is the train/test split applied to opponents.

1. **The eval panels are disjoint from every team the treatment trains with or against**, exact and near
   neighbour, enforced at startup by the registry guard (§5.6). The off-meta exploiter's teams are drawn from
   W_hl's remainder or P-generated, never from a panel.
2. **The exploiter's own teams are an ON-TARGET read**, reported but never the headline. "Better against the
   teams it was drilled on" is the expected, uninformative gain. This is the old taught / untaught split.
3. **Exploitability is read by a FRESH best responder on a DIFFERENT archetype set** (X35's R2 design). The arm is
   never scored by the exploiter it trained against.
4. **The external panel is FIXED:** Kakuna on the frozen held-out panel plus in-pool. Its opponent and team
   distribution do not move with the treatment, so ON − OFF is interpretable.
5. **Every number carries its `team_overlap` block and its CELL tag** (our source × opponent role × opponent
   source; `design_ladder_campaign.md` §6.3, not built). Most of September's confounds were two cells averaged
   unseen.

**Cost.** ≈ 18 GPU-h (A3's shape: 3 exploiters × 1.1 + 13 + 1.4). It is CONDITIONAL (§6). It is the existing
backlog row X10, re-scoped.

### 4.5 L5 — team randomisation / augmentation from Smogon usage

**Mechanism.** Domain randomisation (Tobin et al. 2017, **UNVERIFIED**). Perturb pool teams so that exact
identification fails while human team structure survives:

- **(a) set resampling:** keep the six species and resample k mons' moves, item, EVs and nature from Smogon's
  per-species usage. Hidden Power follows the L1 repair rule.
- **(b) species substitution:** replace k mons using Smogon's `Teammates` prior conditional on the others.
- **(c) whole procedural teams** (`ou_random_teams.js`).

It applies per episode with probability q (e.g. 0.3, k ∈ {1, 2}).

**Owner rule.** The generator reads SMOGON statistics only, which is compliant. The network never reads the
generator. It sees battles.

**Risks:**

- incoherent or illegal teams (validator, learnsets, Species / Sleep Clause);
- less realistic matchups;
- a policy that learns "anything can be anywhere" and loses in-pool sharpness.

**Belief labels:** augmented truths are Smogon samples, so the learned deltas learn the prior back. That is harmless
and, by design, kills H3's channel.

**Cost.** A build of ≈ 1–2 agent-days (opus-high: an offline pre-generated augmented corpus is simpler and
pin-safe), plus ≈ 13 GPU-h for the arm.

**Rank.** Below D1. D1 uses real human sets. D3 is the fallback if D1 reads TRADE with a large in-pool cost, or the
alternative if W_hl's repair fails validation at scale.

### 4.6 L6 — our side's team diversity vs the opponent side's

**The argument for asymmetry.** On a ladder WE choose our team (a few sample teams: the ladder campaign's stage 2)
and the opponent brings anything. So the deployment needs OPPONENT-team breadth and tolerates our-side narrowness.

**The argument against** is §2.5's coupling: self-play opponents pilot the opponent's teams only as well as we
pilot them.

**The test is D2: OPPONENT-ONLY widening** (`--team-mix` on the opponent builder only).

- It is CONDITIONAL on D1 reading ADOPT-CANDIDATE or TRADE.
- Read against D1 and C0 with D1's rule. It answers which side carries the held-out gain and which carries the
  in-pool cost.
- The stage-2 main agent (our side narrowed to ~5 sample teams) stays the ladder campaign's question, not this
  doc's.

### 4.7 L7 — anchor the beliefs to the Smogon prior (found here)

If D0f shows the learned deltas HURT off-pool, cheap levers exist:

- **belief-delta dropout:** per episode, with probability d, the learned deltas are zeroed, so the policy
  learns to act on prior-only beliefs part of the time;
- a KL penalty from the posterior to the prior;
- shrinking the delta's output scale.

All three are Smogon-anchored, so they are compliant. This is CONDITIONAL on D0f. It is a learner-side build
(≈ 0.5–1 agent-day) plus ≈ 13 GPU-h, and it is not ranked until D0f reads.

### 4.8 D5 — the LEAVE-OUT run: the clean memorisation test

**Design.**

- A FRESH `--arch production` run to 30M (≈ 5.6 GPU-h).
- The pool MINUS 72 teams drawn at random by a registered seed, stratified by source directory, with their near
  neighbours removed too.
- Read at 10M, 20M and 30M:
  - X(held-in sample of 72) vs X(held-out 72), piloting and facing (D0e's instrument, a fixed reference);
  - Kakuna on both sets.

**Why it matters.** The split is random inside ONE population, so population shift, Kakuna's familiarity and the
HP artifact are ALL absent. A detected deficit is memorisation of the training teams (PROVISIONAL at 30M). The
three checkpoints show how fast it grows.

**Limits:**

- 30M is not a plateau, so the result is PROVISIONAL by the owner's rule;
- one seed;
- the stratification leaves the sample bias untouched (sample teams can be held out too: declared).

**It needs:** a `--team-exclude <manifest>` source filter, a small build, also useful to the registry guard
(§5.6); a GPU window or the rented cloud GPU (`project_cloud_gpu_benchmark`); the orchestrator's GO.

---

## 5. The evaluation MATRIX: honest about bias, not a hunt for one neutral set

**Rewritten 2026-10-18 on the owner's input.** The first version proposed frozen panels and one held-out headline.
The owner's point is that **every team set we can read is biased toward one side**:

- the sample teams favour Kakuna, because Metamon overfit to `competitive`;
- the held-out replay teams favour Kakuna too, because they are more diverse and of the population its replay data
  is built from (UNVERIFIED);
- our own sample folder favours us, because we pilot it about 2.1× as often as any other pool team.

So §5 no longer looks for a neutral set. It states each set's bias (5.1), splits piloting from facing (5.2),
weights the cells by what the ladder actually brings (5.3), adds a meter with no external opponent at all (5.4),
names ONE headline with fixed side numbers (5.5), and freezes everything before any pool is widened (5.6).

### 5.0 What the ladder looks like: the resemblance data

The ladder proxy is Metamon's `hl_05_26/gen3ou`: 22,862 teams from 1400+ ladder and smogtours replays, unrevealed
parts filled by usage sampling. Each team is compared with the SAMPLE reference: home72 ∪ `away` (92 files, 79
distinct six-species cores).

| best overlap with ONE sample team | 6 | 5 | 4 | ≤ 3 |
|---|---|---|---|---|
| **species** (share of the 22,862) | **17.6 %** | 33.2 % | 34.3 % | 14.9 % |
| **mon sets** (species + moves), ≥ 4 | — | — | 0.2 % (all of ≥ 4) | — |

- **The ladder is sample-CORE-heavy, not sample-COPY-heavy.** 85.1 % of ladder teams share ≥ 4 species with one
  sample team and 50.8 % share ≥ 5. Only 0.2 % share ≥ 4 full mon sets. That last figure is UNDERSTATED, because
  replays reveal partial movesets and the archive fills the rest from usage; the true share of literal sample copies
  is UNKNOWN.
- **Reproduced twice.** The orchestrator's quick script (`~/.cache/gen3ai/hl_vs_samples.py`, re-run 2026-10-18:
  6 = 17.6 %, 5 = 33.2 %, 4 = 34.3 %, mon sets ≥ 4 = 0.2 %) and the versioned one,
  [`measurements/ladder_weights_2026-10-18/scripts/ladder_weights.py`](../research_state/measurements/ladder_weights_2026-10-18/README.md)
  (`gen3_team_overlap_v1`: formes and IV-typed Hidden Power), agree to 0.1 pp.
- **heldout72 is ladder-like at the species level.** Its strata are 16 CORE / 44 NEAR / 12 FAR teams (22 / 61 /
  17 %), against the proxy's 18 / 67 / 15 % (CORE = all six species of one sample team; NEAR = 4–5; FAR = ≤ 3).
  What separates it from the sample teams is its SETS (104 / 432 mon sets seen in our pool) and the Hidden Power
  artifact, not off-meta species.
- **Descriptive, post hoc:** X26 piloting heldout72 teams scores 0.425 on CORE (80 games), 0.430 on NEAR (244) and
  **0.250 on FAR** (76); K2 0.350 / 0.381 / 0.289. Kakuna's team is the same random heldout72 mix in every row, so
  the rows differ in our piloted team only, and they are confounded with team strength. Even CORE held-out teams
  read 15 pp below home72 (0.575): **unfamiliar SETS on familiar cores already cost most of the drop**, and
  off-meta species cost more on top.

### 5.1 (i) The familiarity matrix

One row per frozen team set. "Ours" is training exposure; "Kakuna's" is its likely exposure, **UNVERIFIED in every
row**, because Metamon's README names no training team set for Kakuna.

| team set | our PILOTING exposure | our FACING exposure | mon sets in our pool | species strata (CORE / NEAR / FAR) | Kakuna's likely familiarity (UNVERIFIED) | BIAS of a symmetric cell |
|---|---|---|---|---|---|---|
| **home72** (our `data/teams/sample`) | exact teams, ≈ 2.1× a pool team (the 10 % sample bias) | exact teams, 1× | 432 / 432 | 72 / 0 / 0 (it IS the reference) | the cores are common on the ladder (85 % of ladder teams share ≥ 4 species with a sample team); 8 / 72 are exact `competitive` copies | **toward US**: we trained on the exact sets |
| **away** (Metamon `competitive`) | 19 / 20 exact, ≈ 1.2× | ≈ 1.2× | 119 / 120 | 20 / 0 / 0 | its evaluation set; "Metamon has overfit to them" (`metamon/README.md:865`) | **both familiar**; the most balanced cell we have, leaning to Kakuna by its own README |
| **heldout72_hpfix** (`hl_05_26`, HP repaired; PENDING, the parallel agent) | 0 (0 exact, 0 near neighbours) | 0 | 104 / 432 | 16 / 44 / 12 (ladder-like) | the SET postdates Kakuna (dates); its POPULATION is the kind its replay data is built from | **toward KAKUNA** (population), reduced by the repair (Kakuna's data may share the HP-Dark fill) |
| **smogon_forum** (`measurements/smogon_forum_teams_2026-10-18/`, PENDING, another agent) | 0 by construction (overlap-checked) | 0 | measured at build | measured at build | forum teams are `competitive`'s own source ("human-made sample teams, usually from the Smogon forums", `metamon/env/wrappers.py:40`); popular ones get laddered | **near-neutral**: human-built and unseen by us. Its near neighbours of `competitive` must be excluded, or it is `away` again |
| **P72** (procedural, Smogon-only; X37a's build) | 0 as teams | 0 | measured at build | measured at build | Metamon's `paper_variety` is also usage-generated; whether Kakuna trained on it is UNVERIFIED | **unclear, leaning to US**: its sets ARE the Smogon usage our belief PRIORS encode. Low team quality compresses toward 0.50 |

**How to read it.** No row is neutral. Each number is quoted WITH its row's bias, the way RULE 1 makes a number
travel with its regime. The rows bracket the truth from both sides: home72 overstates us, heldout72 understates
us, and `away` and smogon_forum sit between.

### 5.2 (ii) PILOT vs FACE: the asymmetric cells, and the owner's hypothesis as a named test

Write S(X, Y) for our win rate when WE pilot set X and Kakuna pilots set Y, with P = home72 (sample) and
H = heldout72_hpfix.

| | Kakuna pilots P | Kakuna pilots H |
|---|---|---|
| **we pilot P** | S(P, P) = 0.575 [0.526, 0.623] (exists) | **S(P, H), the DEPLOYMENT cell**: we bring our familiar teams, the opponent brings ladder-like ones |
| **we pilot H** | **S(H, P), the REVERSE cell** | S(H, H): the parallel `heldout72_hpfix` read, PENDING |

S(P, H) is read from X37a's three stratum sub-cells (§5.3), reweighted to heldout72_hpfix's OWN strata shares
(16 / 44 / 12 of 72), so that it estimates the plain asymmetric cell. S(H, P) and K(P, H) use the full
heldout72_hpfix set.

**What each difference contains** (an additive model; Δ = how much stronger P teams are than H teams for whoever
pilots them):

- **PILOT drop** = S(P, P) − S(H, P) = our piloting familiarity (P vs H) + Kakuna's facing familiarity + Δ.
- **FACE drop** = S(P, P) − S(P, H) = our facing familiarity + Kakuna's piloting familiarity − Δ.
- Under additivity the two add up to S(P, P) − S(H, H), the total drop. The interaction
  I = S(P, H) + S(H, P) − S(P, P) − S(H, H) should read ≈ 0; it is reported, and a detected I makes the split
  UNDECIDED.
- **The team-strength term Δ does not cancel.** It is calibrated by the cell K(P, H): **Kakuna pilots P against
  Kakuna piloting H** (an anchor-vs-anchor pair cell). K(P, H) − 0.5 = Δ + (Kakuna's piloting − facing familiarity
  with P over H). Read as Δ under assumption **A-K: Kakuna's piloting and facing familiarity are equal in size**
  (it learned from both sides of replays; UNVERIFIED).

**The named test H-PILOT (the owner's hypothesis, 2026-10-18): "we do not PILOT off-meta teams well".** With
Δ̂ = K(P, H) − 0.5, PILOT_c = PILOT drop − Δ̂ and FACE_c = FACE drop + Δ̂. Intervals are 95 % Wald over independent
cells (n = 400 each).

| class | condition | meaning |
|---|---|---|
| **SUPPORTED** | PILOT_c's lower bound > 0 AND (PILOT_c − FACE_c)'s lower bound > 0 | piloting unfamiliar teams carries more of the drop than facing them |
| **KILLED** | PILOT_c's upper bound < +3.0 pp, OR (FACE_c − PILOT_c)'s lower bound > 0 | piloting costs under 3 pp, or facing carries more |
| **UNDECIDED** | otherwise; I detected; or any bound within 1e-9 of a boundary (rule 8) | — |

- **Power.** PILOT_c has a half-width ≈ 8 pp; PILOT_c − FACE_c ≈ 12 pp. That resolves a lopsided split of the
  18-pp drop and not a balanced one. The Kakuna-free second line is D0e (§3.2), which splits piloting from facing
  by the TRAJECTORY of X26's checkpoints, where strength cancels by construction.
- **The OFF-META variant.** heldout72 is ladder-like at the species level (§5.0), so H-PILOT as defined reads
  "unfamiliar teams", mostly unfamiliar SETS. The owner's "off-meta" reading is the stratum split of the REVERSE
  cell by our piloted team (CORE / NEAR / FAR, recorded per row): **H-PILOT-OFFMETA is SUPPORTED iff our score
  piloting FAR teams is below our score piloting CORE teams by a 95 % team-clustered bootstrap bound > 0**
  (12 FAR and 16 CORE teams; descriptive until B-D6 stamps the opponent's team per game). The banked descriptive
  version reads FAR 0.250 vs CORE 0.425 (§5.0).

### 5.3 (iii) The LADDER-WEIGHTED headline L

**The quantity.** Our expected win rate against Kakuna when WE bring our sample teams (home72) and the OPPONENT
brings a team drawn from the ladder's distribution:

L = w_CORE · D_CORE + w_NEAR · D_NEAR + w_FAR · D_FAR

- **D_s** is the DEPLOYMENT cell restricted to stratum s: we pilot home72; Kakuna pilots only the heldout72_hpfix
  teams of stratum s (16 CORE / 44 NEAR / 12 FAR teams, installed for Metamon as three sub-sets).
- **w_s** is stratum s's share of the ladder proxy: **CORE 0.1761, NEAR 0.6747, FAR 0.1493**
  (`ladder_weights_v1`).

**How the weights are computed, frozen and versioned.**

1. **Computed** by `ladder_weights.py`: for each of the 22,862 proxy teams, the best species overlap with any one
   sample-reference team, binned into the three strata; the share is per FILE (a file is one replay team, so a
   popular team weighs by how often it was played).
2. **Frozen** by X37a's registration commit, BEFORE any deployment-cell game. The JSON records the proxy's
   digest (every file's name and sha256), the reference's digest, the matcher (`gen3_team_overlap_v1`), the strata
   boundaries and the counts. Until then it is a DRAFT (`ladder_weights_v1.json`, `status`).
3. **Versioned.** A change of ANY input (a newer ladder proxy, a different sample reference, the matcher, the
   boundaries) is a NEW version `ladder_weights_vN`. Every L is quoted with its version, and two versions are never
   pooled or differenced.

**The CORE stratum's sensitivity (one parameter, declared).** A ladder team with a sample team's six species may BE
that sample team with its published sets (the owner's reading: "the ladder is mostly sample teams"), or the same
core with other sets (what the replay files show). The fills hide which. So:

- **L (the headline)** reads D_CORE on the replay CORE teams, the data as it is (c = 0);
- **L_c1 (always beside it)** reads D_CORE on literal sample teams, i.e. S(P, P) (c = 1).

The truth lies between. L_c1 − L = w_CORE × (S(P, P) − D_CORE): with w_CORE = 0.176, a 15-pp difference between
the two CORE reads moves L by ≈ 2.6 pp.

**Interval.** A stratified bootstrap that resamples TEAMS within each stratum, then games within team, with a
fixed seed and draw count (deterministic, rule 8). The game-only Wald interval is reported beside it.
Allocation (Neyman-like, toward NEAR): 100 CORE / 400 NEAR / 100 FAR games, game-only half-width ≈ 4 pp; the team
clusters (12 FAR teams) widen it.

**Weaknesses (stated with every L):**

- **The proxy is not the ladder:** `hl_05_26` is 1400+ ladder and smogtours over a year, not the ladder at our
  target rating today.
- **The strata are partly fill-driven:** unrevealed team MEMBERS are usage-filled too, which favours common species
  and likely inflates CORE / NEAR (size UNVERIFIED).
- **Species resemblance ignores sets:** the c parameter covers this only at CORE.
- **Few teams per stratum:** 16 / 44 / 12, so team-cluster variance dominates FAR.
- **The opponent is Kakuna, not a human:** L is "ladder-weighted vs Kakuna", never a ladder win rate. Each
  stratum keeps its row's bias (§5.1).
- **Our side pilots home72 uniformly:** a deployed agent would choose its best teams.
- **The reference set defines the strata:** another sample reference changes the weights, which makes it a new
  version.

### 5.4 (iv) GEN-M: an in-house generalisation meter with NO external opponent

**Why.** Every Kakuna number carries Kakuna's training bias. A meter where both sides are OUR models, with the same
pool exposure, has none. It measures one thing: does our training progress transfer to teams we never trained on?

**The design.**

- **Players:** X26 @t against X26 @(t − 50M), the Tier-1 plateau test's own pairing (its W-back node), and against
  the registry `production` baseline.
- **Teams:** mirrored pairs (each pair plays the same two teams twice with the pilots swapped, so team strength
  cancels) on each frozen panel: home72 (the sample ground), heldout72_hpfix, smogon_forum and P72.
- **When:** at every Tier-1 check (every 10M), from the plateau meter's own schedule; retroactively on X26's
  existing nodes every 50M.
- **K2**, the previous lineage's plateau with the same pool, is the other reference that matters: X26 − K2 on
  unseen teams vs on sample ground says whether X26's lead is pool-specific, with no Kakuna. K2 does not load at
  HEAD, so this needs a pinned-peer player over the websocket front end (B-D8, optional; F-DV-13).

**The statistic.** g_P = the pair score on sample ground − 0.5, and g_U the same on an unseen panel.

| class | condition |
|---|---|
| **TRANSFERS** | (g_P − g_U)'s 95 % interval (pentanomial over pairs) lies wholly below +2.0 pp AND g_U's lower bound > 0 |
| **PARTIAL** | (g_P − g_U)'s lower bound > 0 AND g_U's lower bound > 0 |
| **POOL-ONLY** | (g_P − g_U)'s lower bound > 0 AND g_U's lower bound ≤ 0 |
| **UNDECIDED** | otherwise, or a bound within 1e-9 of a boundary |

**Cost.** 500 mirrored pairs per panel × 4 panels ≈ 4,000 games ≈ 1 lane-hour per check on the CPU (≈ 1.16 games/s,
`design_x26_readiness.md` §2.6). It needs B-D7: `main.h2h` gains a team-panel flag (mirrored pairs on a REGISTERED
panel, the panel digest in the regime).

**Relation to D0e (§3.2).** D0e is a one-time split of piloting from facing (the untaught meter). GEN-M is the
standing total, read beside every plateau check, so a plateau on the pool can never again be read without its
transfer.

### 5.5 (v) THE headline, and the numbers that always sit beside it

**THE headline: L.** It is the ladder-weighted deployment estimate vs Kakuna greedy (§5.3), quoted as
"L (ladder_weights_vN) = p [lo, hi]". It is the one number that answers "how would we do on the ladder, bringing our
own teams", and it carries its weights version and its bias statement.

**Always beside it, in this order:**

1. **The sample ground:** S(P, P) on home72 and the `away` symmetric cell. This is where we are matched or ahead,
   with its bias (toward us on home72, toward Kakuna on `away`).
2. **The unfamiliar panel:** the symmetric cells on heldout72_hpfix, smogon_forum and P72. This is where we lose,
   with its bias (toward Kakuna on heldout72).
3. **GEN-M at the latest check:** the transfer class, with no external bias.
4. **L_c1:** the CORE-as-literal-sample sensitivity (§5.3).
5. **When measured, the H-PILOT split:** PILOT_c and FACE_c (§5.2).

No number leaves without its row of the matrix (§5.1), its regime (SOP RULE 1) and its `team_overlap` block.

### 5.6 (vi) Freeze, then widen

**Freeze FIRST.** Before any pool-widening build lands (B-D1), the panels AND the ladder weights are frozen.

- **The registry** `designs/research_state/heldout_panels.json` (PROPOSED, B-D3) holds, per panel: the name and
  source, the draw rule written before the draw, the seed, the manifest sha256, the near-neighbour radius (≥ 4
  shared mon sets), FROZEN-since and "held out from". It also points at `ladder_weights_vN`.
- **Panels:** home72 (in-pool reference), `away` (in-pool, Kakuna's turf), heldout72_hpfix, smogon_forum, P72;
  D5's leave-out 72 is private to D5.
- **The guard.** At training startup, every team source is checked by content against every FROZEN panel, exact
  and near neighbour. A hit is a startup FATAL, like `gen3_untaught_teacher_guard_v1`. A typed consent flag exists
  for a declared contamination, which is then stamped on every read.
- **Why:** `away` was described as "never trained on" for a month while 19 of its 20 teams were pool copies.
- **Every read carries** its `team_overlap` block, the panel name and the registry sha.
- **Declared ASYMMETRIC cells** (the deployment, reverse and calibration cells) count under their DECLARATION. Today
  `team_source_asymmetry` compares file counts only (F-DV-5), and the anchors counting rule DROPS an asymmetric
  unit, so these cells need the per-side content digest instead (B-D6).
- 🚨 Anchor rows carry `our_team` but NOT the peer's team. The peer CSV's `Team File` column does not align
  one-to-one with `team_draws`, so per-game opponent-team splits on banked reads are unsafe (F-DV-6, B-D6).

**Then WIDEN, and read every widening arm (D1, D2, D3, D4) on the matrix:**

- **PRIMARY (gain):** ON − OFF on the UNFAMILIAR cells. That is the pooled symmetric reads on heldout72_hpfix,
  smogon_forum and P72 vs Kakuna, equal weights per panel. The rule is §4.2's.
- **CO-PRIMARY (no loss worse than −2.0 pp on sample-team ground):** measured IN-HOUSE, on GEN-M's sample-ground
  cell (each final against the frozen reference population, mirrored pairs on home72). A Kakuna read cannot resolve
  a −2.0 pp bar at an affordable n (F-DV-12).
- **Reported:** L and L_c1, the Kakuna sample ground (home72, `away`), GEN-M's class, and the H-PILOT split.
- **A widening corpus must exclude every panel's near neighbours.** W_hl and smogon_forum overlap by population:
  forum teams get laddered.

**An in-run held-out meter (B-D5)** is for FUTURE runs only, since X26 is pinned: an eval-cycle cell piloting
held-out-panel teams, 100 games per cycle. It is subsumed by GEN-M once B-D7 exists, and stays optional.

---

## 6. How this fits X35, and the ranked plan

### 6.1 The new arms against A1–A7

X35's arms all draw pool teams. Its primary R1 and its anchor guard G-K read pool teams only. **The diversity arms
target a different deficit:** the largest one measured against the external reference. They share X35's start
(P\*), pin, control C0, seeds and guards, but carry their OWN primary (§4.2).

| rank in X35 | arm | GPU-h (production) | built? | condition |
|---|---|---|---|---|
| 1 | A1 archive pool | ≈ 13 + 1.4 | yes | — |
| **2** | **D1 WIDEN (both sides, s = 0.5)** | **≈ 13** + the Kakuna panel on CPU | **no** (§6.2) | rank 2 UNLESS X37a reads POPULATION / KAKUNA-DRIVEN AND X37b reads TRANSFERS (then below A2, §7); the §6.2 builds in `X35_PIN_k` |
| 3 | A2 PFSP | ≈ 13 + 1.4 | yes | — |
| 4 | A3 main exploiters | ≈ 18 | yes | — |
| 5 | A4 trailing magnet | ≈ 13.4 + 1.4 | T56 | — |
| 6 | A5 Nash archive | ≈ 14.4 | T57 | A1 not LOSS |
| 7 | A6 league exploiters | ≈ 18 | yes | A3 ADOPT or gap ≥ 0.15 |
| cond. | **D2** opponent-only widen | ≈ 13 | with D1's builds | D1 ADOPT-CANDIDATE or TRADE |
| cond. | **D3** augmentation | ≈ 13 | 1–2 agent-days | D1 TRADE with a large in-pool cost, or W_hl unusable |
| cond. | **D4** off-meta exploiters (X10 re-scoped) | ≈ 18 | gate change + manifest | D1 GAIN on the replay panel but NOT on the third source |
| cond. | **D7** belief anchoring | ≈ 13 | 0.5–1 agent-day | D0f says the deltas hurt off-pool |
| — | A7 entropy | — | — | not run (X35's triggers) |

**Why D1 ranks 2nd, behind A1 and ahead of A2:**

- A1 is built, dry-run validated and first in X35's ABBA (A-B-B-A) order. Moving it gains nothing.
- A2 is the transfer calibration's "plausible null". D1 does not displace it from calibration, because D1 is
  PRODUCTION-only: its mechanism reads the belief machinery, and the toy has none (`design_compute_efficiency.md`
  §5.2 rule 4).
- **D1 targets the one deficit we can name against the outside world (−10 pp vs Kakuna on held-out teams).**
  A1–A6 target an in-pool plateau whose size against outsiders is unknown.

**Batch 1 order, if adopted:** A1 s3501, C0 s3501, C0 s3502, A1 s3502, A1 s3503, C0 s3503, then D1 s3501–3503.
The first D1 read lands ≈ 13 GPU-h after A1's (≈ 39 GPU-h into batch 1).

### 6.2 The builds (proposed; to be filed in `TASK_BACKLOG` when the orchestrator adopts each)

| build | what | size | must land before |
|---|---|---|---|
| B-D1 | the W_hl corpus: T8's HP repair (one function with `heldout72_hpfix`'s), validation, the exclusions (every frozen panel and its near neighbours, smogon_forum and P72 included; the crash classes), the manifest, a full-tier soak | opus-high, ≈ 1–1.5 agent-days | `X35_PIN_k` |
| B-D2 | `--team-mix <corpus>:<share>`: a `MatchupSpec` source kind for both builders, recorded in `metadata.json` (spec hash), restored on resume; byte-identical when absent; a pin-safe corpus path | opus-medium, ≈ 0.5 agent-day | `X35_PIN_k` |
| B-D3 | the held-out panel registry (pointing at `ladder_weights_vN`) + the refusing startup guard (exact + near neighbour, by content) + `--team-exclude` | opus-medium, ≈ 0.5 agent-day | B-D1, `X35_PIN_k` (and D5) |
| B-D4 | P72: generate, sanitise, validate, overlap-check (pool, `competitive`, `hl_05_26`, smogon_forum), assign strata, install for Metamon; the X37a configs (per-stratum Metamon sub-sets of heldout72_hpfix; the asymmetric cells' per-side sources) | opus-medium, ≈ 0.25 agent-day | X37a |
| B-D5 | an in-run held-out eval cell (future runs; subsumed by GEN-M) | opus-medium, ≈ 0.5 agent-day | optional |
| B-D6 | `their_team` per anchors row (F-DV-6); a per-side content-digest asymmetry check, so a DECLARED asymmetric cell counts (F-DV-5) | opus-medium, ≈ 0.25 agent-day | X37a's asymmetric cells count only with it, or with a registration that waives the count-based flag for declared cells |
| B-D7 | `main.h2h --panel <registered panel>`: mirrored pairs on a frozen team panel, the panel digest in the regime, a refusal for an unregistered panel (GEN-M's engine) | opus-medium, ≈ 0.5 agent-day | GEN-M |
| B-D8 | a pinned-peer player: a checkpoint that loads only at its own pin (K2) plays over the websocket front end against an in-process HEAD side, regime-stamped (F-DV-13) | opus-high, ≈ 1 agent-day | optional (GEN-M's K2 rows) |
| B-D9 | an ASYMMETRIC Metamon pair cell: each peer of an anchor-vs-anchor cell takes its own team set (today one `team_sets[<teamset>]` serves both peers) | opus-medium, ≈ 0.25 agent-day | X37a stage 1's K(P, H) |

### 6.3 The ranked plan, the first experiment, and its cost

| rank | id | what | GPU-h | CPU | when |
|---|---|---|---|---|---|
| **1** | **X37a stage 1** | **the matrix's core** (§5.2, §5.3): the DEPLOYMENT cell by stratum (we pilot home72; Kakuna pilots heldout72_hpfix's CORE 100 / NEAR 400 / FAR 100 games); the REVERSE cell (we pilot heldout72_hpfix, Kakuna pilots home72, 400); the strength calibration K(P, H) (Kakuna vs Kakuna, 400). It yields **L**, **L_c1** and **H-PILOT** / **H-PILOT-OFFMETA** | **0** | **≈ 3.5 lane-h** (1,400 games); configs ≈ 0.1 agent-day; K(P, H) needs B-D9 | **NOW**, once `heldout72_hpfix` is frozen (the parallel read) |
| 2 | **X37a stage 2** | the third-source rows: P72 symmetric (400) and smogon_forum symmetric (400); the §3.1 rule | 0 | ≈ 2 lane-h + B-D4 ≈ 0.25 agent-day | when smogon_forum lands (PENDING) |
| 3 | X37b | GEN-M (§5.4) on X26's nodes every 50M (home72, heldout72_hpfix, smogon_forum, P72) + D0e's piloting / facing split (§3.2) | 0 | ≈ 5 lane-h + D0e ≈ 4–6 lane-h; B-D7 ≈ 0.5 agent-day | after B-D7 |
| 4 | X37c | D0g toy + D0d Kakuna trajectory | 0 | ≈ 4 lane-h | now |
| 5 | X37d | D0f prior-only DiD (with its surgery build) | 0 | ≈ 2 lane-h + 0.5 agent-day | now |
| 6 | **X38** | **D1 WIDEN fork** (X35 rank 2), read on the matrix (§5.6) | ≈ 13 (+13 if C0 is not shared) | ≈ 4 lane-h panel + GEN-M | the plateau call, batch 1 |
| 7 | X39 | D5 LEAVE-OUT (production, 30M) | ≈ 5.6 | ≈ 3 lane-h | a GPU window or the cloud; conditional on X37 MIXED or UNDECIDED |
| 8 | X40 | D2 opponent-only widen | ≈ 13 | ≈ 4 | conditional on X38 |
| 9 | X41 | D3 augmentation | ≈ 13 + build | ≈ 4 | conditional on X38 |
| 10 | X42 | D4 off-meta exploiters (re-scopes X10) | ≈ 18 | ≈ 4 | conditional on X38 |
| 11 | X43 | D7 belief anchoring | ≈ 13 + build | ≈ 4 | conditional on X37d |

**The first experiment is X37a stage 1.**

- **What:** 1,400 CPU games on one rule-5b lane (≈ 3.5 lane-hours, 0 GPU), plus ≈ 0.1 agent-day of configs: the
  per-stratum Metamon sub-sets of heldout72_hpfix, and the asymmetric per-side sources (supported today by
  `our_team_sources` vs Metamon's `team_sets`).
- **K(P, H)** also needs B-D9 (≈ 0.25 agent-day). Without it, stage 1 still yields L and the raw PILOT / FACE drops,
  and H-PILOT stays UNDECIDED until K lands.
- **The counting rule** must accept DECLARED asymmetric units: B-D6, or a registration that waives the count-based
  flag for the declared cells only.
- **No `data/` change.** The sub-sets live in the measurement directory and Metamon's cache, like heldout72.
- **Why it goes first.** It is the cheapest read that produces THE headline (L), tests the owner's hypothesis
  (H-PILOT), and decides between the hypothesis families with the third source (stage 2). Every GPU decision
  downstream (D1's rank, D5's need) reads it.

---

## 7. What would change this plan

- **`heldout72_hpfix` reads ≥ 0.50 for X26:** the artifact was most of the drop (H6). D1's case weakens to "train
  on the population for its own sake". The headline proposal still stands.
- **X37a stage 2 reads POPULATION / KAKUNA-DRIVEN and D0e / GEN-M read TRANSFERS:** D1 drops below A2; the
  smogon_forum read governs before any claim.
- **D0e reads MEMORISATION-DOMINATED:** continued pool training buys little held-out strength. D1 (or D5, then an
  earlier diagnostic widening fork from a current X26 checkpoint, never inside X26) moves up, and the owner hears
  it as a strategic deviation (§8.4).
- **D0f reads that the deltas hurt off-pool:** D7 joins the ranked list.
- **H-PILOT reads SUPPORTED (§5.2):** piloting carries the drop. OUR-side breadth matters most: D1 keeps both sides,
  D2 (opponent-only) is dropped, and D3's set augmentation on our side rises. **KILLED with FACE carrying the drop:**
  D2 rises and the belief levers (D0f, D7) come first.
- **L reads at or above 0.50:** the deployment question is answered in our favour despite the unfamiliar cells;
  widening is then about the open population's tail, not the headline.

---

## 8. Where I disagree with the owner

### 8.1 "Roughly tied with state of the art" is the wrong summary

- **It is one checkpoint (161M of a 500M cap), one seed, pre-plateau, PROVISIONAL**, read against one opponent at
  one regime. Kakuna's strongest temperature is unresolved, and greedy vs T = 1.0 read identically.
- **The three cells are not three draws from one population.** They differ in our exposure (2.1× / 1.2× / 0), in
  Kakuna's familiarity (UNVERIFIED), and in the HP artifact. Averaging them into "tied" weights a deployment-
  relevant loss against two in-distribution reads.
- **The number a ladder would see is the held-out one, and there X26 is BEHIND:** 0.395 [0.348, 0.444]. The
  honest one-liner is **"ahead on our training teams, behind on the open population, PROVISIONAL"**.

### 8.2 "Ahead where they haven't seen the teams" assumes too much about Kakuna

home72 is the Smogon sample set. 8 of its 72 teams are exact copies of `competitive`, the set Metamon says it
overfit to. Smogon sample teams are widely played, so whether Kakuna's replay data holds them is UNVERIFIED. "Our
home turf" is accurate; "their blind spot" is not shown.

### 8.3 "Tied on shared ones" is not tied

`away` reads 0.497 [0.449, 0.546]. That is NOT DETECTED, not equivalent: our own rule says equivalence needs the
difference's interval inside a declared bar (`feedback_equivalence_needs_delta_ci`). We cannot exclude ±5 pp.

### 8.4 The lead over K2: what shrinks and what does not

- The owner's "lead shrinks to +3.8 on held-out" is right about the held-out cell.
- **The SHRINK itself is not detected against home72.** (X26 − K2 on home72) − (X26 − K2 on heldout72) = **+0.050
  [−0.046, +0.146]**: Wald, my computation from the published counts, independence assumed.
- Against `away` it is detected: +0.105 [+0.010, +0.200]. But `away` has 20 teams, and an interval that ignores
  team clustering is optimistic.
- **So "the K2 → X26 gain is pool-specific" is SUGGESTED, not shown.** D0e is the read that shows it.
- 🚨 **The strategic point.** The plan is X26 to its plateau ON THE POOL, then plateau-breakers ON THE POOL. If D0e
  reads MEMORISATION-DOMINATED, part of that plateau is a memorisation plateau, and the deployment metric does not
  move with it. That is why D0e runs NOW, during X26, and not after.

### 8.5 The near-term goal's wording needs a re-scope

"Beat Kakuna on OUR teams" was set (2026-09-27) when `away` was believed unseen and ours was the natural home set.
Met on training teams only, it says little about strength.

**Proposal (owner decision Q3):**

- keep it as the in-distribution milestone, MET PROVISIONALLY;
- add a held-out goal: the Wilson lower bound > 0.50 on the held-out panel, at the plateau (DECIDED, Q3). With
  the Q1 revision the proposal is that the goal reads on THE headline L (§5.5).

### 8.6 Where I agree

Widening the pool is the right lever FAMILY, and a fork at the plateau against a matched control, measured on BOTH
held-out gain and in-pool cost, is the right design. Three amendments:

- **carve the panels first** (§5.6);
- **repair the HP fields first** (the 09-24 finding: heads trained on the artifact would memorise it);
- **keep the corpus out of `data/teams/`** (F-DV-2).

### 8.7 On the owner's 2026-10-18 inputs

**1. "Held-out replay teams were a bit unfair, because the ladder is mostly sample teams."**

- Half right. The ladder IS sample-CORE-heavy: 85 % of ladder teams share ≥ 4 species with one sample team.
- But heldout72 has the SAME core mix: 22 / 61 / 17 % CORE / NEAR / FAR against the ladder's 18 / 67 / 15 %.
- So as a ladder proxy it is fair at the species level. What it changes is the SETS (and the Hidden Power artifact,
  which the hpfix re-read removes).
- "Mostly sample TEAMS" in the sense of literal copies is not measurable from replays: ≥ 4 shared full sets is
  0.2 %, understated by fills. L's c parameter (§5.3) carries exactly this uncertainty.

**2. "We are matched on sample teams even though we don't bias toward them."**

- We do bias toward them. home72 IS our `data/teams/sample`, which the trainee draws at ≈ 2.1× any other pool team
  (`DEFAULT_TRAINEE_BIAS_PROB = 0.1`).
- 19 of `away`'s 20 teams are in our pool.
- "Matched on `away`" is level WITH an exposure advantage, against a set Metamon says it overfit to.

**3. "Held-out teams are biased in Kakuna's favour because they have more diversity."**

- Diversity by itself favours no one. It penalises whoever MEMORISED a closed set, which is us by construction (719
  teams; W12's team identification).
- It favours Kakuna only if Kakuna's data covers that population (UNVERIFIED).
- The distinction matters, because the remedies differ: memorisation is fixed by widening (D1); a yardstick that
  favours Kakuna is not a defect to fix, only a bias to state (§5.1).

**4. "The key may be that we don't PILOT off-meta teams well."**

- Plausible, and now a named test (H-PILOT, §5.2).
- The descriptive data support a version of it: X26 pilots FAR (off-meta-core) held-out teams at 0.25.
- But held-out teams on SAMPLE-LIKE cores already read 0.43 against home72's 0.575. So the larger part of the drop
  is unfamiliar SETS, not off-meta species. H-PILOT-OFFMETA is the stratum split that separates the two.

**5. "I like sample teams the most."**

- Agreed as the sample GROUND beside the headline (§5.5). It is where a deployed agent would choose to play.
- Not as the headline itself, because the opponent does not choose sample teams for us. L is the deployment
  question with the opponent's side weighted by the ladder.

---

## 9. Owner questions: all DECIDED 2026-10-18

| # | question | decision |
|---|---|---|
| Q1 | The external HEADLINE | **DECIDED (orchestrator, 2026-10-18)** as "the held-out panel, with the sample-team ground beside it"; **REVISED the same day at the orchestrator's request, in light of the matrix (§5):** THE headline is **L**, the ladder-weighted deployment estimate vs Kakuna (`ladder_weights_vN`). Beside it, always: the sample ground (home72, `away`), the unfamiliar panel (heldout72_hpfix, smogon_forum, P72), GEN-M's class and L_c1 (§5.5) |
| Q2 | Where the widening corpus lives | **DECIDED (orchestrator):** a pin-safe `src/` corpus now; `data/` once TECH_DEBT (f) P1's fix (pinned children resolve `data/` through the pin tree) lands |
| Q3 | A held-out near-term goal | **DECIDED (orchestrator):** added; the Wilson lower bound > 0.50 on the held-out panel, at the plateau. **Proposed consequence of the Q1 revision (not decided here):** the goal reads on L's 95 % lower bound > 0.50, so the goal and the headline are the same number |
| Q4 | The third source | **DECIDED (owner):** NO owner or friends' teams (quality too low). The third sources are P72 (X37a stage 2) and smogon_forum (another agent; PENDING) |
| Q5 | D1 at rank 2 in X35 batch 1 | **DECIDED (orchestrator):** yes, conditional on X37 (§6.1) |

---

## 10. FINDINGS (standing rule 7)

- **F-DV-1.** No in-run meter, plateau test or X35 readout component reads a team our models have not trained on.
  X35's G-K reads home72 and away, both training data. Proposed: add the held-out panel to G-K. This doc does
  not change X35; the orchestrator adopts or declines.
- **F-DV-2.** Anything placed under `data/teams/` widens X26's (and X34's, and X35's) pool at the next restart.
  `TeamLoader` walks `data/teams/` recursively, and pinned runs read `data/` from main. This is a silent pool
  change, not a crash.
- **F-DV-3.** `validate_exploiter_trainee_is_sample` refuses any non-sample team for a pinned exploiter, so
  off-meta exploiters (L4) are blocked by design until a declared-manifest exception exists.
- **F-DV-4.** Two HP repairs could diverge: the parallel `heldout72_hpfix` script, and T8 for the training
  corpus. They must be ONE function, or a training corpus and its evaluation panel get different HP rules.
- **F-DV-5.** `team_source_asymmetry` compares file COUNTS, so an asymmetric 72-vs-72 cell, deliberate or
  accidental, reads "false".
- **F-DV-6.** Anchor rows carry no per-game peer team. The peer CSV's `Team File` does not align one-to-one with
  `team_draws` (one unit checked: draws[0] never logged, the last repeated). Per-game facing analysis on banked
  reads is UNSAFE (GIGO).
- **F-DV-7.** The "untaught 8" are POOL teams the generalist pilots. Only teacher pins are guarded. A read that
  calls them unseen for X26-era models is wrong.
- **F-DV-8.** Off-pool crash classes are open and reachable from W_hl (TECH_DEBT (a) P1, Transform + Hidden
  Power: 20 corpus teams) and possibly from Imprison (UNVERIFIED livelock). In training these become run
  crashes, not skipped battles.
- **F-DV-9.** D0a's per-team splits are post hoc, many-way and confounded with team strength. They are cited
  only as description. The HP-Dark gradient appears in BOTH models.
- **F-DV-10.** Not verified here:
  - Kakuna's training team sets (the README names none);
  - whether Kakuna trained on `paper_variety`-style procedural teams;
  - whether `main.anchors` loads the toy;
  - the untaught meter's CPU rate;
  - `validate_teams_locally`'s time on ~23k teams.

  Each is marked where used.
- **F-DV-11.** The literature in §4 (Cobbe 2019 / 2020, Tobin 2017, Dennis 2020, Parker-Holder 2022, the
  AlphaStar league) is cited from memory: **UNVERIFIED** in detail.
- **F-DV-12.** A Kakuna read cannot resolve a −2.0 pp non-inferiority bar at an affordable n. At p ≈ 0.5 a ±2 pp
  Newcombe half-width needs ≈ 4,800 games per arm (≈ 12 lane-hours per arm). So the widening arms' sample-ground
  co-primary is measured IN-HOUSE (§5.6), and the Kakuna sample ground is reported only.
- **F-DV-13.** K2 does not load at HEAD (`gen3_event_record_v2`; read only through its pin `a5b2fdba` on torch
  2.5.1). An in-house X26-vs-K2 row needs a pinned-peer player (B-D8). Until then GEN-M uses X26's own past nodes and
  the `production` baseline.
- **F-DV-14.** The strength calibration K(P, H) assumes A-K: Kakuna's piloting and facing familiarity are equal
  (UNVERIFIED). An anchor-vs-anchor cell also gives both peers ONE team set today (`cfg.team_set(teamset)` in
  `main.anchors.peers`), so K(P, H) needs B-D9.
- **F-DV-15.** The ladder strata are partly FILL-driven. `hl_05_26` fills unrevealed team MEMBERS from usage too,
  which favours common species and likely inflates CORE / NEAR (size UNVERIFIED). The literal-sample-copy share is
  unmeasurable from replays, which is why L carries the c sensitivity (§5.3).
- **F-DV-16.** The banked anchors counting rule drops any unit with `team_source_asymmetry`, so a DECLARED
  asymmetric cell with unequal set sizes (the stratum sub-cells: 72 vs 16 / 44 / 12) would never count. X37a needs
  B-D6 or a registration that waives the flag for declared cells only.

---

## 11. Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-18 | The doc and its taxonomy (design worker, owner request) | four axes (our teams, opponent teams, opponent policies, evaluation) plus the self-play coupling between the first three | one "diversity" knob; treating policy diversity (X35) as covering team diversity | §2; X35's arms all draw pool teams |
| 2026-10-18 | How to explain the held-out gap | 11 competing hypotheses, each with a kill test; H7 killed by logic (the score crossed 0.50); H8 and H11 killed as MAJOR causes by D0a and the rows; H9 partly killed (K2 replicates) | reading the drop as memorisation outright | §3; D0a |
| 2026-10-18 | The third source | P72 (procedural, Smogon-only) now; T2 (human-built, unseen by both) as the confirmation; T2 is never trained on | waiting for a human set before any read; using `gl_05_26` (same population as heldout72) | §3.1 |
| 2026-10-18 | The widening design (PROPOSED) | D1: W_hl (repaired `hl_05_26` minus panels and near neighbours, minus crash classes), both sides, draw share 0.5, a fork of P\* against X35's C0, primary = the held-out panel, co-primary = in-pool non-inferiority (R1 > −2.0 pp) | widening into `data/teams/` (F-DV-2); widening inside X26 (owner's plateau-first order); X35's R1 as D1's verdict (it scores pool teams only); training on heldout72's own teams | §4.1–§4.2 |
| 2026-10-18 | Train against vs play only | both, on DISJOINT sets: frozen panels carved first, a refusing registry guard, training on the remainder | training on the evaluation teams; evaluation only | §4.3, §5.1 |
| 2026-10-18 | Reading evaluations when opponents are the treatment | panels disjoint from every treatment team; on-target reads reported, never the headline; a fresh best responder on a different archetype set; a fixed external panel; overlap + cell tags | scoring an arm by the exploiter it trained against | §4.4 |
| 2026-10-18 | The first experiment | X37a: P72 + the deployment cell + its reverse, ≈ 3 CPU lane-hours, 0 GPU | D1 first (GPU before the cheap discriminator); D5 first (GPU, and not yet needed) | §6.3 |
| 2026-10-18 | D1's rank in X35 (PROPOSED, the orchestrator's GO) | 2nd, after A1, ahead of A2, conditional on X37 | 1st (A1 is built and ordered); after A6 (it targets the larger measured deficit) | §6.1 |
| 2026-10-18 | **§5 REWORKED into an EVALUATION MATRIX** (design worker, owner input, orchestrator's follow-up unit) | no neutral set exists, so the protocol states each set's bias: (i) a familiarity matrix (our piloting / facing exposure, set and species overlap, Kakuna's likely familiarity, UNVERIFIED; the bias direction); (ii) the PILOT / FACE cells with a Kakuna-vs-Kakuna strength calibration and the owner's hypothesis as the named test H-PILOT (and H-PILOT-OFFMETA); (iii) the ladder-weighted L; (iv) GEN-M, in-house; (v) ONE headline; (vi) freeze, then widen | the first version's "one held-out panel as the headline" (it hides the panel's bias toward Kakuna); hunting a single neutral set | owner 2026-10-18; §5 |
| 2026-10-18 | The ladder resemblance and `ladder_weights_v1` (DRAFT; frozen by X37a's registration) | strata by best SPECIES overlap with the sample reference (home72 ∪ `away`): CORE 0.1761 / NEAR 0.6747 / FAR 0.1493 of 22,862 `hl_05_26` teams, per file; versioned by every input's digest; never pooled across versions | mon-set strata (fills make them meaningless: ≥ 4 shared sets 0.2 %); unweighted unique teams (popular teams are what the ladder brings) | `measurements/ladder_weights_2026-10-18/`; the orchestrator's `hl_vs_samples.py` agrees to 0.1 pp |
| 2026-10-18 | **Q1 REVISED: THE headline is L** (ladder-weighted deployment estimate vs Kakuna; our side pilots home72) | beside it always: the sample ground, the unfamiliar panel, GEN-M, L_c1 and (when measured) the H-PILOT split | the held-out panel alone (biased toward Kakuna, and not the deployment shape); the sample ground alone (biased toward us) | §5.5 |
| 2026-10-18 | Q4 (owner): the third sources | P72 + smogon_forum (another agent's collection, PENDING); smogon_forum governs when the two disagree. SUPERSEDES the 2026-10-18 "T2 = the owner's and friends' teams" row | owner and friends' teams (quality too low, owner) | §3.1 |
| 2026-10-18 | X37a REVISED into two CPU stages | stage 1, the matrix's core: the deployment cell by stratum (600), the reverse cell (400), K(P, H) (400), ≈ 3.5 lane-h, 0 GPU; stage 2, the third-source rows (P72, smogon_forum; 800) | the first X37a (P72 + the unstratified deployment and reverse cells: it gave neither L nor the strength calibration) | §6.3 |
| 2026-10-18 | Widening arms' readout (amends §4.2) | PRIMARY = the pooled UNFAMILIAR cells (heldout72_hpfix, smogon_forum, P72); CO-PRIMARY = no loss worse than −2.0 pp on SAMPLE-TEAM ground, measured in-house (home72 panel); L, L_c1 and GEN-M reported | X35's R1 on the full pool as the co-primary (it is not the sample ground); the Kakuna sample ground as the co-primary (it cannot resolve −2.0 pp, F-DV-12) | §4.2, §5.6 |
| 2026-10-18 | Q2, Q3 and Q5 DECIDED (orchestrator) as this doc recommended | Q2: a pin-safe `src/` corpus now, `data/` after TECH_DEBT (f) P1; Q3: a held-out near-term goal (with the Q1 revision, the proposal that it reads on L); Q5: D1 at X35 rank 2, conditional on X37 | — | §9 |
