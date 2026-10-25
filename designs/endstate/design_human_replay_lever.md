# Human replays as a TOGGLEABLE training lever: the first reads, the levers, and the pre-registered plan (X59-S, X60)

**Status: REVISED DRAFT — pending re-review (2026-10-25). EXPLICIT-ONLY** (the plan and its pre-registrations; what is
current lives in the always-current [`human_data_architecture.md`](human_data_architecture.md)). This revises the draft
of `14ae4203` after REVIEW A (data / bias / leakage), REVIEW B (experiments / causal) and REVIEW C (reasoning); every
finding's disposition is in
[`RESPONSE.md`](../research_state/measurements/human_replay_quality_2026-10-25/RESPONSE.md). **Nothing is built,
trained or launched; no GPU was used.** The power numbers are
[`plan_power.py`](../research_state/measurements/human_replay_quality_2026-10-25/plan_power.py)'s (seeded, CPU;
output `results/plan_power.json`) unless they cite REVIEW B's `review_b_power.py`.

**The grant (owner, 2026-10-25).** Human replays MAY train the policy and the search, only as a TOGGLEABLE lever, OFF by
default, with data-quality analysis first. The owner also asked five questions: collection, snapshotting, drift, human
vs self-play data, and whether a human-only model beats ours. The last two are this plan's FIRST reads (§2).

Abbreviations: P\* = X26's plateau (`rb_x26_s2601` @ 280,000,170); HOM = the human opponent model (P\* fine-tuned on
human choices, rung 1b); H = heldout72_hpfix, F = forum72, P72 = the procedural panel; U = the unfamiliar panels vs
Kakuna; home72 = our sample teams (in the pool); C-GAME = rung 1b's own-side completion (the log's reveals + Smogon
draws); BC = behaviour cloning; pp = percentage points; lane-h = one CPU lane for an hour.

---

## 0. One screen

- **The data is usable with filters** (the card): `hr_spec_v1` gives 1.80M informative ≥ 1500 rows before the
  revised exclusions; the revised panel exclusion (§3.1) keeps ≈ 29 % of ≥ 1500 battles (REVIEW A's measure).
- **There is no offline evidence either way** that human labels carry unfamiliar-team skill: the familiarity proxy is
  uninformative (REVIEW A M1). So nothing is built before a read says what to build.
- **First reads, in order (§2):** (1) P\* vs Metamon's human-only `SmallRL` (≈ 1.3 lane-h, zero build); (2) the HOM's
  trace twin, the instrument check (≈ 1 lane-h); (3) P\* vs `LargeRL` / `LargeIL` (two downloads for the owner's
  approval); (4) **X59-S**, the matched-student contest: two students on the SAME human states, human labels vs P\*'s
  sampled labels (≈ 25–35 lane-h, or ≈ 2 GPU-h + 20 lane-h).
- **The levers, re-ranked by the stated criterion (§4):** the reads first; then H-TEAMS with the D1 contrast; then H-BC
  or H-INIT, whichever X59-S favours; H-KL last. X59 (the HOM vs Kakuna) is WITHDRAWN as a decision rule (its SUPPORTS
  fired more under forgetting than under skill) and kept only as a descriptive Δ_U read.
- **X60 (§6)** keeps D1's frame with REVIEW B's fixes: the primary moves to F + P72 at 630 games per (seed final,
  panel), so NULL is reachable (0.58 at Δ = 0) and GAIN has 0.77 power at +2.5 pp; H becomes a positive control
  (practised / unpractised halves); the human-data contrasts are H-TEAMS − D1 and H-BC − P-BC.

---

## 1. The architecture this plan builds on

The sources, the inventory of human data already in use, the snapshot → rowset → model chain, the one switch and its
grammar, the questions Q-H1..Q-H15 and the drift policy are [`human_data_architecture.md`](human_data_architecture.md).
This doc does not restate them.

## 2. The first reads (the owner's two questions)

### 2.0 Done in this revision (CPU, on R-b1's cached forwards)

- **The account-disjoint re-read (REVIEW A M2).** On held-out rows whose acting account never played on the train
  dates, the HOM beats P\* by 0.503 nats [0.467, 0.542] (22.5 % of the ≥ 1500 held-out rows; seen accounts: 0.605 [0.586, 0.624]). The difference, 0.102 [0.060, 0.143], is account memory: ≈ 17 % of the gain. Learnability is not account
  memory. The HOM's own early stopping used a battle-id split; that stays a declared caveat until a rowset re-fits it.
- **The familiarity proxy, re-cut (REVIEW A M1).** On full-six sides only, P\*'s NLL on the panels' region minus on
  pool-like teams is +0.064 [−0.016, +0.149] nats (one joint battle-cluster bootstrap). With no bar registered and no A-1
  prediction in the metric, it is reported and steers nothing.

### 2.1 FR-1: P\* vs `metamon:SmallRL` (Q-H10, zero build)

- **Cells:** H, F, P72 and home72; 400 games each; greedy vs greedy; `main.anchors` at the harness X51 used; seeds 0–19.
- **Read:** P\*'s win rate per cell and pooled U, Wilson 95 % (half-width ≈ 4.5–4.9 pp a cell, ≈ 2.6–2.8 pp pooled U).
- **Rule:** HUMAN-ONLY-BEATS-OURS iff pooled U's UB < 0.50; OURS-BEATS iff its LB > 0.50; else UNDECIDED. A bound
  within 1e-9 of 0.50 takes UNDECIDED.
- **Prediction (GUESS):** OURS-BEATS (Metamon's own local table puts its human-only models far below its self-play ones).
- **Cost:** ≈ 1.3 lane-h at ≈ 3 s a game.

### 2.2 FR-2: the HOM's trace twin (REVIEW B M6; registered in rung 1b §8.5, never read)

- **What:** R1's traces read from the opponent's seat, two rows per decision (C-GAME-completed; the TRUE team); the NLL
  of the HOM and of P\* on both; the twin gap = completed − true; report HOM's gap − P\*'s gap.
- **Bar:** if the HOM's gap exceeds P\*'s by > 0.05 nats, every HOM-based Kakuna score is a lower bound and C-GAME rows
  carry a learnable artefact; X59-S's contrast survives (both students see identical completed states), its absolute
  levels do not.
- **Cost:** forwards only, ≈ 1 lane-h; ≈ 0.2 agent-day to point `main.x50_r1.read_r0`'s trace path at the policy NLL.

### 2.3 FR-3: P\* vs `LargeRL` / `LargeIL` (Q-H10 at Metamon's largest human-only scale)

- **Needs two downloads** (Metamon's `LargeRL` and `LargeIL`, 200M each, from its public model hub) and their pins in
  `designs/ops/anchors.json`. Not done: the owner's approval is required.
- **Cells, rule:** as FR-1. **Cost:** ≈ 4.5 lane-h each (ESTIMATE).

### 2.4 FR-4: X59-S, the matched-student contest (Q-H9 and Q-H10 in our architecture)

**The question.** On the SAME human states, does a policy taught by human labels play unfamiliar teams better than one
taught by our own policy's labels? And does a model trained only on humans beat P\*?

- **Rows:** one registered rowset of `hr_spec_v1`'s TRAIN dates: actor ≥ 1300 (volume for a from-scratch student), the
  card's DROP filters (bots, walk-overs, duplicates), §3.1's exclusion over EVERY held-out panel including
  `forum216b`, §3.3's account-disjoint validation, ≥ 2 legal actions, hidden turns with set labels (§3.2), the
  quitter's last 2 decisions dropped (§3.5). Its rows manifest sha256 is registered before training.
- **Two students, ONE difference:** S-H's labels are the human's action; S-P's labels are one action SAMPLED from
  frozen P\*'s masked policy at T = 1 on the same row, keyed by the row id (the matched-noise rule). On a hidden turn
  S-H carries the set label and S-P its sample (a declared asymmetry on ≈ 12 % of rows; the secondary drops those rows
  in both arms).
- **Matched:** our architecture (`--arch production`'s network), one random init, one keyed batch order, the HOM's
  optimiser, a FIXED step budget (no early stopping, which would de-match the arms); one epoch on CPU or three under a
  GPU lease, the same for both.
- **Optional third arm S-S:** P\* labels on P\*'s own SELF-PLAY states, the same row count; S-P − S-S is the value of
  human STATE coverage at a matched teacher (the breadth mechanism of H-TEAMS and D1).
- **Reads:** each student vs Kakuna ckpt 34 (greedy, harness `63f97a57`), home72 400 and H, F, P72 at 800 each
  (2,400 U games per student, ≈ 16 lane-h for both); in-house S-H vs S-P head to head on each U panel and home72,
  4,000 games a panel (half-width 1.55 pp); each student vs P\*.
- **Primary:** Δ_S = S-H − S-P on pooled U vs Kakuna (equal-weight panels; normal interval; one seed per arm).
- **Power** (`plan_power.py`, students at ≈ 0.30 on U): half-width 2.6 pp; P(HUMAN-LABELS-BETTER) = 0.61 / 0.84 / 0.96
  at Δ = +3 / +4 / +5 pp; 0.025 at Δ = 0; MDE80 4.0 pp (worst case p = 0.5). At 3,600 U games per student: 0.78 /
  0.95 at +3 / +4 pp (+8 lane-h).
- **Rule** (a bound within 1e-9 of 0 takes UNCLEAR):
  - **HUMAN-LABELS-BETTER** iff Δ_S's LB > 0 → build H-BC first, the P-BC placebo conditional (§6).
  - **OWN-LABELS-BETTER** iff Δ_S's UB < 0 → drop H-BC; H-INIT remains only if the HOM's descriptive Δ_U ≥ 0.
  - **else UNCLEAR** → H-TEAMS first, with the D1 contrast.
  - **Floor fallback (registered now):** if both students score < 0.15 vs Kakuna on U, the in-house head-to-head on U
    decides with the same rule.
  - **Replication (the strong-claims rule):** a decisive class that steers a GPU build is re-run with a second seed
    pair first; it stands only if the second pair clears the same bound in the same direction (joint power at +4 pp:
    0.71; +50 % cost).
- **Q-H10, reported:** S-H − P\* on U and home72 vs Kakuna, with intervals; "a human-only model beats ours" iff the LB is
  > 0 on U.
- **VOID:** the two label columns differ anywhere but the label (assert row bytes equal); a student fails to load in the
  anchors harness; a 20-game smoke is not reproduced by a seeded repeat; > 1 uncounted unit per cell.
- **What it cannot say:** from-scratch students are not a plateau fork; HUMAN-LABELS-BETTER does not show that H-BC
  inside PPO helps (X60 does).
- **Cost:** P\*'s labels ≈ 2–4 lane-h; two students ≈ 5–10 lane-h each at one epoch (ESTIMATE from the HOM's 373
  rows / s) or ≈ 1 GPU-h each under a lease; reads ≈ 16 lane-h; build ≈ 0.5–1 agent-day (the HOM trainer with a label
  switch and a from-scratch init).

### 2.5 Descriptive, optional: the HOM's Δ_U (REVIEW B B1 fix 2)

The HOM vs Kakuna is kept as a DESCRIPTIVE read, never a decision rule: Δ_U = HOM_U − P\*_U, one-sided (LB > 0 is weak
support; a negative Δ_U refutes nothing, since the HOM may have lost general skill), at 1,600 games per cell per model
with P\* re-played on NEW unit seeds (half-width 2.0 pp, power 0.84 at +3 pp; ≈ 18 lane-h).

---

## 3. Data rules every rowset obeys

### 3.1 The panel exclusion: ONE rule, on REVEALED facts, before the fill (REVIEW A B1, B2)

- **Rule:** drop a battle if either seat's REVEALED facts (moves, item, ability; `memory._fits`) fit ≥ 3 mons of one
  team of ANY held-out panel (`heldout72`, `heldout72_hpfix`, `forum72`, `p72`, `forum216b`), counting only mons with at
  least one move revealed. It runs on the log BEFORE C-GAME completes anything.
- **Why ≥ 3:** the panels are "unfamiliar" because no pool team holds ≥ 3 of their sets; a training rowset with a 3-set
  core of a panel team makes that team familiar by the same yardstick. ≥ 4 left 3-set cores in ≈ 48 % of ≥ 1500 battles.
- **Its dose, stated:** it removes 70.7 % of ≥ 1500 TRAIN battles (REVIEW A's `E_panel_exclusion`, without
  `forum216b`); the rowset manifest records the realised share. ≈ 0.5M ≥ 1500 rows remain (ESTIMATE).
- **Second net:** the exact-set guard (`main.heldout_panels.contamination`) still runs on completed teams.
- **Regression test (with the build):** a log in which a side pilots a panel team verbatim is excluded; it fails on
  revert to the post-fill guard.
- **Proximity report:** per panel team, the most fits any kept human side reaches (and ≥ 5 shared species), reported by
  tercile in every lever read; descriptive (REVIEW B: per-team MDE ≈ 7.5 pp).

### 3.2 Hidden turns keep a SET label (REVIEW A M3)

A `cant` or a faint before acting proves the human chose a move (a gen-3 switch resolves first). Those rows stay, with
the loss −log Σ over the legal MOVE actions of p(a); windows where the foe's Pursuit KO'd a switching mon are excluded.
Dropping them doubled the labelled switch rate of a sleeping mon (0.56 vs 0.26) and a frozen one (0.68 vs 0.29).

### 3.3 Accounts, not battles, are held out (REVIEW A M2)

Validation (2 %) and held-out splits are by a salted hash of the ACCOUNT, never the battle id; the held-out dates'
reads report the unseen-account stratum separately (§2.0).

### 3.4 The Kakuna-family audit (REVIEW A M4; DESIGN, ≈ 1 agent-day + ≈ 6 lane-h, ESTIMATE)

- **Why:** Metamon's public agents can ladder under any name, and every lever read scores against Kakuna; imitating
  Kakuna's own decisions would read as skill. A name / volume rule cannot see a fleet at 40 games a day.
- **Score:** for every ≥ 1500 account with ≥ 50 TRAIN games, the mean log-likelihood of its labelled actions under
  Kakuna (Metamon's own replay parser and observation, CPU) minus that under the HOM.
- **Flag:** accounts above the p99 of that score's distribution over accounts with ≥ 1 chat line a game (a human
  reference); report the flagged share of rows.
- **Use:** a registered SENSITIVITY in X59-S and X60: the primary re-read without the flagged accounts; a sign change
  is reported as the headline.

### 3.5 Smaller rules

- **Forfeit tails (A m3):** drop the quitting side's last 2 decisions; report NLL by decisions-before-quit.
- **The stale fill prior (A m4):** the Smogon window is 2025-05..2026-04; measure the cost with a lag-matched prior on a
  held-out slice before any lever reads.
- **Circular calibration (A m5):** X60 reports the belief / intent heads' calibration on the Kakuna cells, ON vs OFF.
- **Choice Band (A m6):** report the share of rows whose active used one move all stay with no item revealed; mark
  their masks uncertain.
- **The drawn-move signature (B M6):** a drawn move is never the label and a drawn bench mon is never a switch target;
  X59-S matches it across arms; FR-2 measures whether a model learns it.

---

## 4. The levers, ranked for goal (b)

The criterion: expected pp on U vs Kakuna per GPU-hour, at an acceptable risk to home72. Measured anchors: D1 (+2.1 pp
[−0.9, +5.1], breadth via forum teams, P −1.6) and X52 (practice +3.2 [−0.2, +6.8]). Every other number is a GUESS.

| rank | lever | what changes | why | expected pp (b), GUESS | GPU | main risk |
|---|---|---|---|---|---|---|
| 0 | **the first reads** (§2) | nothing | they decide which lever is worth a build | n/a | 0–2 h | none |
| 1 | **H-TEAMS** `teams:<rowset>@<share>` | the POOL: completed ≥ 1500 human ladder teams join the self-play draw on both sides | the only lever with a measured sibling (D1); breadth on the ladder population | +0 to +4 | ≈ 13 per 3-seed arm | panel memory (§3.1); Smogon-drawn fills (Q-H13) |
| 2 | **H-BC** `bc:<rowset>@<coef>` **or H-INIT** (a fork of the HOM itself, plain PPO) | the LOSS (an auxiliary BC term) / the INIT | human LABELS on states P\* never trained; H-INIT is Metamon's recipe class (human pretraining, then self-play) at no extra build | +0 to +4 | ≈ 16 per 3-seed arm | a pull toward 1500-level play; forgetting (H-INIT's value head refits) |
| 3 | H-KL `kl:<model>@<coef>` | the LOSS: KL to a registered HOM on self-play states | piKL-style; acts where we are already strong | +0 to +1 | ≈ +15 % | in-pool play pulled toward the HOM |
| — | HB0 | none: X59-S's S-H IS the human-only comparator in our architecture | answers Q-H10 | n/a | ≈ 1 h | none |
| — | BC warm start of a FRESH run | the INIT of a new run | AlphaStar's supervised init | unknown | ≥ 42 for a pair to 100M | cost |

- **Struck:** "H-BC is the only lever whose gradient lands on unfamiliar-team states" (false: H-TEAMS, D1 and X52 all
  do; what only H-BC adds is human labels). The literature (AlphaStar's KL, Metamon's pretraining) supports the
  init-plus-KL family, not an auxiliary BC term added at a plateau (UNVERIFIED readings, from memory).
- **Rejected:** human game OUTCOMES as critic labels (off-policy); the HOM as a self-play opponent (FACE_c +0.017); an
  advantage-weighted BC variant now (one lever at a time).

---

## 5. The complexity budget

One clause, one parameter, one rowset or model id per run; every other knob lives in the rowset or the
pre-registration; one lever per A/B; one shared control. The grammar and its rules are the architecture doc's §6.

---

## 6. X60: a lever A/B (GPU; D1's frame with REVIEW B's fixes)

| item | registration |
|---|---|
| arms | ON = P\* forked +20M with ONE clause (H-TEAMS, or H-BC / H-INIT per X59-S); OFF = the same fork with `off` |
| matched control | X35's C0 only if bit-identity holds at the arm's pin (below); else a fresh C0h |
| seeds | 3501–3503 (look 1), 3504–3505 (look 2), shared across arms |
| **PRIMARY** | Δ = ON − OFF vs Kakuna, pooled equal-weight over **F + P72**, AM-5's widened Newcombe (`main.x35.d1`, with an upper-quantile φ, REVIEW B m6), **630 games per (seed final, panel)** |
| power (seed SD 0) | NULL 0.58 at Δ = 0; GAIN 0.58 / 0.77 at +2.0 / +2.5 pp; half-width 1.74 pp at look 2. At a seed SD of 2.7 pp NULL falls to 0.05: the seed rule below |
| heterogeneity guard | GAIN also needs both F's and P72's pooled point estimates > 0 |
| **CO-PRIMARY P** | Δ_P = ON − OFF on home72 in-house (`main.h2h`), Welch 97.5 %, non-inferiority bar −2.0 pp |
| **goal (a) guard** | ON's Kakuna home72 at the final (400 games pooled over seeds) keeps its Wilson LB > 0.50 |
| **positive control** | H split in two halves stratified by `ladder_weights_v1`; ON's rows INCLUDE the ≥ 3-fit battles near H-p and EXCLUDE those near H-u (a named exception to §3.1); 420 games per half per seed final. H-p's Δ is the lever's ceiling at this dose (power 0.90 at +5 pp); H-u's is transfer |
| **human-data contrasts** | H-TEAMS − D1 on F + P72 (D1's five finals topped up to 630); H-BC − P-BC, the placebo run only after an H-BC GAIN (the same rows, coefficient and schedule, labels sampled from frozen P\*). Half-width 1.74 pp; P(LB > 0) 0.61 / 0.92 at +2 / +3 pp |
| controls topped up | C0 and D1's ten finals to 630 a panel on F and P72, and C0's five to 420 on each H half (≈ 37 lane-h at ≈ 10 s a Kakuna game, ESTIMATE); the D1 re-read is a new descriptive read |
| seed rule (before look 1) | if the topped-up φ̂'s one-sided upper bound implies a seed SD > 1.5 pp, add 3 seeds per arm, or declare NULL unreachable before launch |
| decision table | ADOPT-CANDIDATE (GAIN, P non-inferior, guards pass); TRADE (GAIN, P inferior): accepted iff Δ̂ ≥ 2 × \|Δ̂_P\| and the goal (a) guard holds; NULL ([−3, +3]) → stop this lever at the plateau, route to H-INIT / the warm start only if H-p moved; LOSS → drop; final UNDECIDED → the next lever in §4, no extension. Nothing is adopted by the rule alone |
| learning curve | the 10M checkpoints read too (descriptive); before any GPU, an offline data-scaling curve: the HOM recipe on 25 / 50 / 100 % of the rowset, held-out human NLL on unseen accounts (≈ 9 lane-h, ESTIMATE), so "too little data" has a number |
| guards | X35's G-E, G-C, G-F; the belief / intent calibration on the Kakuna cells (descriptive) |
| null-dose identity | before launch, the ON build at coefficient or share 0 reproduces OFF's played bits exactly |
| VOID (checked in a pre-launch smoke, ≈ 0.2 GPU-h) | the startup stamp `[HUMAN-DATA] lever=… rowset=… rows_sha=…` missing or not the registered one; the rows' layout hash ≠ the trainee's; the realised human share outside ±10 %; H-BC's loss not falling ≥ 10 % in the smoke; a panel contamination hit; K9(b) FAIL |
| C0 reuse check | the two goldens and the argv as before, plus `data/` content hashes, the Kakuna harness commit and units, the reference population's sha, the sim binary hash, torch / CUDA / driver versions |
| scope | an ADOPT at P\* is re-read at the production checkpoint before it steers production (X26c read FLAT on U vs P\*) |

**What X60 cannot say.** One dose at one plateau checkpoint; a NULL with H-p flat points to dose or plasticity, not to a
useless lever. `forum216b` (sealed, X54) carries the clean claim, with the same proximity report.

### 6.1 Cost

| step | GPU-h | CPU lane-h | agent-days |
|---|---:|---:|---:|
| FR-1 (SmallRL) | 0 | ≈ 1.3 | ≈ 0.1 |
| FR-2 (trace twin) | 0 | ≈ 1 | ≈ 0.2 |
| FR-3 (LargeRL, LargeIL; downloads) | 0 | ≈ 9 | ≈ 0.3 |
| FR-4 X59-S | 0 (or ≈ 2) | ≈ 25–35 (or ≈ 20) | ≈ 0.5–1 |
| the Kakuna-family audit | 0 | ≈ 6 | ≈ 1 |
| the switch + consumer snapshot check | 0 | — | ≈ 1 |
| H-TEAMS build (export on revealed facts, corpus, guards) | 0 | ≈ 5 | ≈ 1.5 |
| X60 one arm, look 1 (+ look 2) | ≈ 16 (+ 11) | ≈ 20 (+ 13) | ≈ 0.5 |
| control top-ups (C0, D1) | 0 | ≈ 37 | — |

---

## 7. Appendix: what to build (all PROPOSED)

- **The switch:** `agents.info_levers.human_data` (the grammar is the architecture doc's §6): parse, canonicalise,
  hash, `record()` {label, canonical, hash, rowset or model id, the chain down to the snapshot's manifest sha256};
  training honours `bc` / `teams` / `kl`; eval and play honour only `off`.
- **The rowset builder:** reads the snapshot through the manifest check, applies §3's rules in order (exclusion on
  revealed facts → completion → labels), writes rows plus a rows manifest, and registers the rowset with its panel set.
- **H-BC in the learner:** a fold in the eager tail after every stash reader, guarded; its loss joins K9; logs
  `human/bc_loss`, `human/bc_coef`, `human/rows_seen`, and a held-out human NLL on the account-disjoint validation.
- **H-TEAMS:** export C-GAME's completed team per kept seat (≈ 5 lane-h), then a `utils/team_corpora` builder with
  D1's filters, a per-account cap of 5 and both panel nets.

---

## 8. Assumptions, each with its falsifier

| id | assumption | falsified by |
|---|---|---|
| A-1 | Human labels carry unfamiliar-team skill P\* lacks | X59-S OWN-LABELS-BETTER (Δ_S's UB < 0, 2,400 U games per student, power 0.84 at a true −4 pp) |
| A-2 | C-GAME rows teach nothing false | FR-2: the HOM's twin gap exceeds P\*'s by > 0.05 nats |
| A-3 | ≥ 1500 humans are good enough teachers that a small dose does not cost home72 | X60's Δ_P 97.5 % UB < −2 pp |
| A-4 | The spectator stream is representative of finished ladder games | the coverage read (Q-H4; needs approval) finds P(spectated) varying by > 20 pp across length or rating terciles |
| A-5 | A BC gradient ≈ 10 % of the policy's moves the trunk without dominating | the pre-launch smoke's BC loss does not fall ≥ 10 % (VOID before GPU), or A-3's falsifier |
| A-6 | The bot rule plus the Kakuna-family audit catch the bots that matter | §3.4's sensitivity flips the sign of a primary |
| A-7 | §3.1's exclusion removes team memory of the panels | the positive control: H-u's Δ CI excludes 0 in the same direction as H-p's while F + P72 is flat (descriptive; per-team MDE 7.5 pp) |
| A-8 | 20M steps from a plateau is enough plasticity | X60's H-p half reads flat (UB < +2 pp) |

---

## 9. FINDINGS (standing rule 7)

1. **The familiarity proxy is uninformative,** not negative (REVIEW A M1); no offline evidence exists either way for A-1.
2. **The ≥ 3-fit exclusion costs ≈ 71 % of ≥ 1500 battles** (REVIEW A, `forum216b` not included): the near-panel
   region where human coverage is best is exactly what must go.
3. **No unfamiliar panel is clean for a human lever** (Q-H13): H is replay-derived, P72 is drawn from the fill's own
   tables, F has human cores near most teams. `forum216b` carries the clean claim.
4. **The HOM was fitted with a battle-id validation split** and its rowset is unregistered (it predates the registry).
5. **Metamon `LargeRL` / `LargeIL` need two downloads** (FR-3), and the spectator coverage read needs proxy pulls and a
   Smogon fetch (Q-H4): all await the owner's approval.
6. **Experiment ids X59-S / X60 are named here only**, not in `EXPERIMENT_BACKLOG.md`; the orchestrator assigns them.
7. **Reviews pending:** this REVISED DRAFT is re-reviewed before any build.
8. **Process:** CPU only; no GPU, launch or download; `models/`, `data/` and the original corpus were only read.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-25 | **CREATED (DRAFT; explicit-only)** on the owner's grant | the data card; H-BC > H-TEAMS > HB0 > H-KL > warm start; X59 (the HOM vs Kakuna) first; X60 in D1's frame | building H-BC before A-1 is tested; a factorial; pooling the archive | the card; R-b1, R-b2; D1 look 2; X52 |
| 2026-10-25 | **REVISED after REVIEW A / B / C** (REVISED DRAFT, pending re-review) | the owner's two questions as FIRST reads (SmallRL, the trace twin, LargeRL / LargeIL, X59-S); X59 withdrawn as a rule (descriptive Δ_U only); X60's primary on F + P72 at 630 games a panel, H as a positive control, the H-TEAMS − D1 and H-BC − P-BC contrasts, a goal (a) guard, a seed rule, an action for every class; ONE panel-exclusion rule on revealed facts at ≥ 3 fits before the fill; set labels on hidden turns; account-disjoint splits; a Kakuna-family audit; H-TEAMS re-ranked above H-BC, H-INIT added; the consent flag and `judge` dropped | keeping X59's premium rule (it rewarded forgetting); 140 games a cell (NULL unreachable); the ≥ 4-fit exclusion with a dose-response read (unpowered at ≈ 10 games a team) | `RESPONSE.md`; `plan_power.py` |
