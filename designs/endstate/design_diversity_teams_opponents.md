# Diversity of training: our teams, the opponents' teams, the opponents, and the evaluation sets

**Status: DESIGN 2026-10-18 (owner request; planning input; nothing built, nothing launched).** One zero-game,
post-hoc, descriptive read was run for it (D0a, §2.4). ALWAYS-CURRENT (owner, 2026-09-27): a later decision,
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
  "a different team population, the kind Kakuna's replay data covers" are confounded in the one held-out read
  we have. Its Hidden Powers are also an artifact: HP Dark on 164 of 432 mons. A parallel agent is repairing
  that and re-reading the set (`heldout72_hpfix`, PENDING).
- **D0a: what the held-out set is, read with zero games (§2.4, descriptive, post hoc).**
  - heldout72 is **not** stall-heavy. Only 4 of its 72 teams are stall, against 90 of the pool's 719. It leans
    to balance and semi-stall.
  - Its games run longer: median 40–41 turns, against 30–37 on the pool sets, for both models.
  - **Entity coverage is small.** Only 9 of 72 teams carry a species, move, item or ability that no pool team
    carries (7 mons with an unseen species). 203 species-move pairs are absent from the pool, and **160 of
    those are Hidden Power**.
  - Inside heldout72 the scores fall as our team carries more HP Dark, in BOTH models. That is consistent with
    the artifact making weaker sets, and this read cannot tell it from familiarity.
- **The cheapest reads that split the hypotheses (§3) use the CPU only (rule 5b) and can run NOW, while X26
  trains.** In order of value:
  1. **The THIRD-SOURCE read: P72, procedural Smogon-only teams.** It separates "unseen by us" from "Kakuna's
     distribution". It sits in the cell where neither model trained on the set.
  2. **The in-house TRANSFER trajectory (no Kakuna).** X26's checkpoints pilot heldout72, P72 and home72 teams
     against a fixed reference of ours, with paired games (common random numbers) across checkpoints. The
     question: does X26's progress since 25M transfer to teams it never saw?
  3. **The DEPLOYMENT cell.** We pilot our teams and Kakuna pilots held-out teams. That is the shape of a
     ladder game.
  4. **The toy (no belief machinery) on both sets.**
  5. **A prior-only belief switch on X26.** It needs a small build.
- **The levers (§4), ranked by expected information per GPU-hour:**
  - **D1 WIDEN fork at the plateau.** The pool plus about 22k HP-repaired `hl_05_26` teams, minus every
    held-out team and its near neighbours, at a 50 % draw share on both sides. It is read against X35's shared
    control C0, so the arm costs **≈ 13 GPU-h**. Its primary is the HELD-OUT panel. In-pool non-regression is
    a co-primary, because the owner asked "what it does to the teams we are currently training on".
  - **D5 LEAVE-OUT.** A fresh production run at 30M with 72 random pool teams carved out, **≈ 5.6 GPU-h**. It
    is the clean memorisation test inside one population.
  - Conditional follow-ups: **D2** widens the opponent side only; **D3** resamples sets from Smogon usage;
    **D4** trains exploiters on off-meta teams; **D7** anchors the beliefs to the prior.
- **Train against unseen teams, or only play them?** Both, on DISJOINT sets (§4.3):
  - train on a wide slice;
  - evaluate on frozen panels carved out BEFORE any training use;
  - never train on a panel's teams or their near neighbours.
- **The first experiment: X37a.** Build P72 (≈ 0.25 agent-day, opus-medium), then read X26 vs Kakuna on P72
  plus the two asymmetric cells (our home72 against their heldout72, and the reverse). That is 1,200 games
  ≈ **3 lane-hours of CPU, 0 GPU-h**.
- **Where this sits in X35 (§6).** D1 ranks **2nd**, after A1 and before A2, IF X37 does not point squarely at
  Kakuna's home turf. It shares C0, adds ≈ 13 GPU-h and needs six builds (§6.2).
- **Where I disagree with the owner (§8).**
  - "Tied with state of the art" reads one PROVISIONAL checkpoint against confounded sets.
  - The deployment-relevant number is the held-out one, and there we are BEHIND.
  - The "shrinking lead over K2" is itself not detected on home72.
  - The near-term goal's "on OUR teams" was written when we thought `away` was unseen. It should be re-scoped
    to a held-out panel.

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
| H4 | **POPULATION shift** (off-meta / out-of-distribution SETS and play styles: replay teams with usage-sampled fills; longer games) | X26 is bad at this KIND of team whether or not it saw the team | the drop is the same on unseen teams of this population (heldout72) and ABSENT on unseen teams of another population | **X37a P72** (§3.1) and, later, a human-built set (T2). KILLED iff X26(P72) − X26(heldout72) is NOT detected AND X26(P72) − X26(home72) is detected below 0 | D0a: not stall-heavy; games 25 % longer; 21 / 72 teams share all six SPECIES with a pool team but no set |
| H5 | **KAKUNA's home turf** (it trained on parsed ladder replays of this population; UNVERIFIED) | Kakuna is relatively STRONGER on heldout72; our "drop" is partly its gain | the in-house transfer reads (no Kakuna) show NO deficit while the Kakuna drop stays | **D0e** (H1/H2's reads are Kakuna-free) + **X37a P72**. KILLED as the SOLE cause iff D0e's deficit (piloting or facing) has a 95 % lower bound ≥ 5.5 pp, half the Kakuna drop's smallest plausible size (11.1 pp) | partial: `away` is the set Kakuna overfit to, and X26 does 10 pp BETTER there than on heldout72. So the drop is not Kakuna-familiarity alone, ASSUMING its familiarity with `competitive` ≥ with `hl_05_26`-like teams (UNVERIFIED) |
| H6 | **the Hidden Power ARTIFACT** (HP Dark from blank IVs on 164 / 432 mons) | (a) our belief stack predicts Smogon's HP types and mis-reads their coverage; (b) the sets are incoherent (special HP Dark on physical attackers), which changes play; (c) Kakuna's own data carries the same artifact, so it is calibrated to it (UNVERIFIED) | the repaired set reads detectably higher for X26 | **`heldout72_hpfix` re-read** (parallel agent, PENDING; not duplicated here). this doc reads H6 as KILLED iff X26 hpfix − heldout72 is not detected and its 95 % upper bound < +5 pp (the parallel read's own registered rule governs its verdict) | D0a: scores fall with HP Dark on our team in BOTH models (confounded with strength) |
| H7 | **team-building QUALITY / strength** (the held-out teams are weaker or stronger) | both sides draw the same set, so pure strength cancels in expectation; only an INTERACTION remains (noisier teams compress toward 0.50) | compression toward 0.50, never a sign flip | **already KILLED as a sole cause:** X26 fell PAST 0.50 (0.575 → 0.395). As a contributor: the asymmetric cells and the deployment cell (§3.1) | logic |
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
| **unseen by us** | heldout72 (0.395) | **THE MISSING CELL: a third source** |

**Candidates for the third source, ranked by cost:**

1. **P72, procedural** (`ou_random_teams.js --emit 72 --seed <registered>`). Smogon-only, validator-legal,
   unseen by us by construction (checked by `team_overlap` against the pool, `competitive` and `hl_05_26`). It
   costs ≈ 0.25 agent-day to build. **Caveats:**
   - not human-built: the teams are less coherent, so team quality differs;
   - Metamon's `paper_variety` set is also generated from usage statistics, and whether Kakuna trained on it is
     UNVERIFIED. P72 is therefore "unseen as TEAMS by both", and its POPULATION may be closer to Kakuna's than
     the pool is.
2. **T2, human-built and unseen by both:**
   - (a) teams from the owner and friends (1400–1600 ladder players), n ≈ 20–50, which needs the owner;
   - (b) Smogon forum or tournament pastes not in the pool, `competitive` or `hl_05_26`, which needs a `tools/`
     downloader (network) and the owner's OK on sources.

   Either set is checked by `team_overlap` (exact = 0, near-neighbours = 0) and validated. **T2 is the
   confirmation set, and it is never used for training** (§5).

**The cells to read (X37a, CPU, rule 5b, one lane):**

- X26 vs Kakuna on P72: 400 games.
- The DEPLOYMENT cell: our side home72, Kakuna heldout72 (asymmetric, unmirrored): 400 games. On a ladder we
  bring OUR teams and the opponent brings theirs. This cell is that shape. It mixes skill with team choice, BY
  DESIGN. Its config sets `our_team_sources.home` = home72 and Metamon's `team_sets.home` = `gen3ai_heldout72`.
- The reverse: our side heldout72, Kakuna home72: 400 games.

Paired seeds 0–19; ≈ 9 s per game; ≈ 3 lane-hours.

**Decision rule (pre-registrable; deterministic; Newcombe 95 % on win rate, n = 400 each).** Let d1 = X26(P72) −
X26(heldout72) and d2 = X26(P72) − X26(home72).

| class | condition | meaning |
|---|---|---|
| **UNSEEN-DRIVEN** | d1 NOT detected AND d2's upper bound < 0 | X26 loses on unseen teams of another population too: H1–H3 lead; widening (D1) is the lever |
| **POPULATION / KAKUNA-DRIVEN** | d1's lower bound > 0 AND d2 NOT detected | unseen teams of another population are fine: H4–H6 lead; D1 still trains on the population, but the expected value is lower and T2 is needed before a claim |
| **MIXED** | d1's lower bound > 0 AND d2's upper bound < 0 | both |
| **UNDECIDED** | otherwise, or any bound within 1e-9 of 0 (rule 8) | — |

The asymmetric cells are REPORTED, with no rule. They are the inputs to the headline proposal (§5.3).

### 3.2 The in-house TRANSFER reads (no Kakuna): D0e

**Instrument:** `main.untaught_meter`, which exists today.

- Each ref pilots a pinned team from a manifest.
- The FIXED opponent plays the flat pool.
- Games are keyed so that every ref plays the same games (common random numbers).
- Teams are the cluster, and one resampling index set is shared by every contrast.

**The design:**

- **D0e-pilot:** refs = X26 @{25M, 100M, 161M, latest 10M-multiple}; the opponent = the registry baseline
  `production` (`rb_es_end2_s2005`, 15M), greedy. Three manifests (home72, heldout72, P72), 10 games per team.
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

**Cost.** ≈ 4 refs × 3 sets × 720 games ≈ 8,600 games per axis, ≈ 2 lane-hours per axis at the h2h CPU rate of
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
4. **NOT human-built unseen sets (T2).** They are scarce, and they are the evaluation panel's last clean source.

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
   - (a) every team of every registered held-out panel (§5.1), and every NEAR NEIGHBOUR of one (≥ 4 shared mon
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
| **PRIMARY H** (held-out gain) | Δ_H = ON − OFF score vs Kakuna greedy on the HELD-OUT PANEL: `heldout72_hpfix` (W_hl's population, unseen teams) and the third source (P72 now, T2 when built), equal weights. Each seed final plays 7 role-balanced 20-game units per set (140 games), so 3 seeds give 420 per set and 840 per arm; unit seeds 0–6 are shared across arms. Newcombe 95 % on the pooled score | **GAIN** iff the lower bound > 0 AND ≥ 2 of 3 seeds have Δ > 0; **LOSS** iff the upper bound < 0; **NULL** iff the interval lies inside [−3.0, +3.0]; else **UNDECIDED** → X35's look 2 at 5 seeds iff Δ̂ > 0. A bound within 1e-9 of a boundary is UNDECIDED |
| **CO-PRIMARY P** (the owner's "what it does to the teams we are currently training on") | Δ_P = ON − OFF of X35's population score R1 (pool teams, `main.h2h`, Welch over seed finals at 97.5 %) | **NON-INFERIOR** iff the lower bound > −2.0 pp (X35's bar) |
| reported | Δ on each held-out set separately | gain on `heldout72_hpfix` alone = "learned the population"; gain on P72 / T2 too = "generalises beyond its population" |
| reported | Kakuna on home72 and away, ON − OFF | in-pool external read; no rule |
| reported | the D0e transfer read on each final | in-house held-out piloting and facing |
| guards | X35's G-E (exploitability), G-C (the collapse battery), G-F (head-to-head frontier) | as registered there |

**Verdicts:**

- **ADOPT-CANDIDATE:** H GAIN and P NON-INFERIOR, with the guards passing.
- **TRADE:** H GAIN and P inferior. Report the EXCHANGE RATE (held-out pp gained per in-pool pp lost); the owner
  decides.
- **NULL / LOSS / UNDECIDED** after look 2.
- Nothing is adopted by this rule alone.

**Power.** At p ≈ 0.40 and 840 games per arm, the Newcombe half-width is ≈ 4.7 pp. That is enough for a recovery
of a third of the 18-pp drop. More games cost CPU only: doubling to 280 per seed per set narrows it to ≈ 3.3 pp at
≈ 8.4 lane-hours for the two arms.

**Cost.**

- ≈ **13 GPU-h** (3 seeds × 4.4), with C0 shared with the X35 batch. Unshared: +13.
- CPU: the Kakuna panel, ≈ 2 × 840 games ≈ 4.2 lane-h; R1 rides X35's read.
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

1. Carve and FREEZE the held-out panels first (§5.1).
2. Then train on a wide slice that EXCLUDES the panels AND their near neighbours.
3. Read gains on the panels.

**Never train on the evaluation teams themselves.** That is teaching to the test: the number would rise and mean
nothing.

**Which population to train on is a choice of target.** The ladder brings the open population (heldout72's kind),
so training on it is training toward deployment. Keeping one panel from a source NO arm trains on (P72 / T2) is
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
   neighbour, enforced at startup by the registry guard (§5.1). The off-meta exploiter's teams are drawn from
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
(§5.1); a GPU window or the rented cloud GPU (`project_cloud_gpu_benchmark`); the orchestrator's GO.

---

## 5. An evaluation protocol that stays interpretable when the training distribution changes

### 5.1 Frozen held-out PANELS, by source, with a registry and a refusing guard

**The registry** is `designs/research_state/heldout_panels.json`, PROPOSED (build B-D3). It holds one entry per
panel:

- the name and source;
- the draw rule, written before the draw;
- the seed;
- the manifest sha256;
- the near-neighbour radius (≥ 4 shared mon sets);
- FROZEN-since;
- "held out from" (our lineages; Kakuna as far as known).

**The panels:**

| panel | source | held out from us | Kakuna | status |
|---|---|---|---|---|
| home72 | our sample teams | NO (2.1× trainee exposure) | 8 / 72 exact `competitive` | in-pool reference |
| away | Metamon `competitive` | NO (19 / 20) | its overfit eval set | in-pool, Kakuna's turf |
| heldout72 / **heldout72_hpfix** | `hl_05_26`, replay population | YES | the set postdates Kakuna; the population may not | the replay-population panel |
| **P72** | the Smogon-usage generator | YES | as teams; the population is UNVERIFIED | third source, now |
| **T2** | human-built, unseen by both | YES | YES (as far as checkable) | third source, confirmation |
| D5's leave-out 72 | pool split | YES for D5 only | — | D5's private panel |

**The guard.**

- At training startup, every team source is checked by content against every FROZEN panel: exact and
  near-neighbour.
- A hit is a startup FATAL, like `gen3_untaught_teacher_guard_v1`.
- A typed consent flag exists for a declared contamination, which is then stamped on every read.

**Why.** `away` was described as "never trained on" for a month (2026-09-14 → 10-18) while 19 of its 20 teams
were pool copies. A registry plus a guard makes that impossible.

### 5.2 Every read carries its overlap and its cell

- `summary.json["team_overlap"]` is already required (`fda1efda`).
- Add the PANEL name and the registry sha.
- An asymmetric cell (the deployment cell) declares its two sources. `team_source_asymmetry` today compares
  file COUNTS only (72 vs 72 reads "false"), which is FINDING F-DV-5.
- 🚨 `main.anchors` rows carry `our_team` but NOT the peer's team. The peer CSV's `Team File` column does not
  align one-to-one with the peer's `team_draws` list: unit `X26_greedy_hl72_00` logs `team_draws[1..9]` with the
  last one repeated, and never `team_draws[0]`. Per-game opponent-team analysis is therefore NOT possible on the
  banked reads (FINDING F-DV-6). The fix stamps `their_team` per row from the peer's own record by battle id.

### 5.3 Which number is THE headline (proposal, owner decision Q1)

- **External headline:** X vs Kakuna greedy on the HELD-OUT PANEL. That is `heldout72_hpfix` + the third source,
  equal weights, with the overlap block. It is the number a ladder would see.
- **Beside it, always:**
  - **in-distribution:** home72, which is the near-term goal as literally worded;
  - **the deployment cell:** our home72 vs their held-out.
- No single number is quoted without its panel.
- **In-house:** the population score (pool) AND the D0e transfer read, never one without the other.

### 5.4 An in-run held-out meter (build B-D5)

For FUTURE runs only. X26 is pinned and its eval cannot change mid-run.

- An eval-cycle cell: greedy self vs a fixed sentinel, piloting held-out-panel teams.
- 100 games per cycle.
- It gives every run a held-out trajectory, which no run has today (F-DV-1).

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

### 6.2 The builds D1 needs (proposed; to be filed in `TASK_BACKLOG` when the orchestrator adopts D1)

| build | what | size | must land before |
|---|---|---|---|
| B-D1 | the W_hl corpus: T8's HP repair (one function with `heldout72_hpfix`'s), validation, the exclusions (panels + near neighbours; the crash classes), the manifest, a full-tier soak | opus-high, ≈ 1–1.5 agent-days | `X35_PIN_k` |
| B-D2 | `--team-mix <corpus>:<share>`: a `MatchupSpec` source kind for both builders, recorded in `metadata.json` (spec hash), restored on resume; byte-identical when absent; a pin-safe corpus path | opus-medium, ≈ 0.5 agent-day | `X35_PIN_k` |
| B-D3 | the held-out panel registry + the refusing startup guard (exact + near neighbour, by content) + `--team-exclude` | opus-medium, ≈ 0.5 agent-day | `X35_PIN_k` (and D5) |
| B-D4 | P72 (and later T2): sanitise, validate, overlap-check, install for Metamon; the anchors configs | opus-medium, ≈ 0.25 agent-day | X37a |
| B-D5 | an in-run held-out eval cell | opus-medium, ≈ 0.5 agent-day | optional; future runs only |
| B-D6 | `their_team` per anchors row (F-DV-6); a content-digest asymmetry flag (F-DV-5) | opus-medium, ≈ 0.25 agent-day | before any per-team facing analysis |

### 6.3 The ranked plan, the first experiment, and its cost

| rank | id | what | GPU-h | CPU | when |
|---|---|---|---|---|---|
| **1** | **X37a** | **B-D4 (P72) + X26 vs Kakuna on P72 + the deployment cell + its reverse** (§3.1) | **0** | **≈ 3 lane-h**; build ≈ 0.25 agent-day | **NOW** (rule 5b, during X26) |
| 2 | X37b | D0e transfer reads (pilot and face; home72, heldout72, P72; 4 checkpoints) | 0 | ≈ 4–6 lane-h | now |
| 3 | X37c | D0g toy + D0d Kakuna trajectory | 0 | ≈ 4 lane-h | now |
| 4 | X37d | D0f prior-only DiD (with its surgery build) | 0 | ≈ 2 lane-h + 0.5 agent-day | now |
| 5 | **X38** | **D1 WIDEN fork** (X35 rank 2) | ≈ 13 (+13 if C0 is not shared) | ≈ 4 lane-h panel | the plateau call, batch 1 |
| 6 | X39 | D5 LEAVE-OUT (production, 30M) | ≈ 5.6 | ≈ 3 lane-h | a GPU window or the cloud; conditional on X37 MIXED or UNDECIDED |
| 7 | X40 | D2 opponent-only widen | ≈ 13 | ≈ 4 | conditional on X38 |
| 8 | X41 | D3 augmentation | ≈ 13 + build | ≈ 4 | conditional on X38 |
| 9 | X42 | D4 off-meta exploiters (re-scopes X10) | ≈ 18 | ≈ 4 | conditional on X38 |
| 10 | X43 | D7 belief anchoring | ≈ 13 + build | ≈ 4 | conditional on X37d |

**The first experiment is X37a.** Build P72, then read three 400-game cells on one rule-5b lane:

- ≈ 0.25 agent-day (opus-medium) + ≈ 3 lane-hours of CPU;
- 0 GPU;
- no `data/` change: P72 lives in its measurement directory and Metamon's cache, like heldout72.

**Why it goes first.** It is the cheapest read that discriminates the two hypothesis FAMILIES (§3.1). It is also the
input to the headline proposal (the deployment cell). And every GPU decision downstream (D1's rank, D5's need)
depends on it.

---

## 7. What would change this plan

- **`heldout72_hpfix` reads ≥ 0.50 for X26:** the artifact was most of the drop (H6). D1's case weakens to "train
  on the population for its own sake". The headline proposal still stands.
- **X37a reads POPULATION / KAKUNA-DRIVEN and D0e reads TRANSFERS:** D1 drops below A2 and T2 is built before any
  claim.
- **D0e reads MEMORISATION-DOMINATED:** continued pool training buys little held-out strength. D1 (or D5, then an
  earlier diagnostic widening fork from a current X26 checkpoint, never inside X26) moves up, and the owner hears
  it as a strategic deviation (§8.4).
- **D0f reads that the deltas hurt off-pool:** D7 joins the ranked list.

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
- add a held-out goal: the Wilson lower bound > 0.50 on the held-out panel (§5.3), at the plateau.

### 8.6 Where I agree

Widening the pool is the right lever FAMILY, and a fork at the plateau against a matched control, measured on BOTH
held-out gain and in-pool cost, is the right design. Three amendments:

- **carve the panels first** (§5.1);
- **repair the HP fields first** (the 09-24 finding: heads trained on the artifact would memorise it);
- **keep the corpus out of `data/teams/`** (F-DV-2).

---

## 9. Owner questions

| # | question | this doc's recommendation |
|---|---|---|
| Q1 | The external HEADLINE becomes the held-out panel (§5.3)? | **yes**, with home72 and the deployment cell always beside it |
| Q2 | The W_hl corpus as a pin-safe `src/` corpus, or under `data/` after TECH_DEBT (f) P1's fix? | **`src/` corpus now** (no freeze conflict); `data/` once P1 is fixed |
| Q3 | Add a held-out near-term goal (§8.5)? | **yes** |
| Q4 | T2: the owner's and friends' teams, or a Smogon-forum downloader, or both? | **ask the owner for 20–50 teams first** (cheapest clean source); the downloader if fewer than 20 |
| Q5 | D1 at rank 2 in X35 batch 1? | **yes, conditional on X37** (§6.1) |

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
