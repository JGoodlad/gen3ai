# Human replays as a TOGGLEABLE training lever: the first reads, the levers, and the pre-registered plan (X59-S, X60)

**Status: REVISED (2026-10-25, second revision, after REVIEW D); §6 X60 REDESIGNED (2026-10-26), pending review.
EXPLICIT-ONLY** (the plan and its
pre-registrations; what is current lives in the always-current [`human_data_architecture.md`](human_data_architecture.md)).
The first revision answered REVIEW A (data / bias / leakage), B (experiments / causal) and C (reasoning); this one
answers REVIEW D, the re-review (READY-WITH-FIXES). Every disposition is in
[`RESPONSE.md`](../research_state/measurements/human_replay_quality_2026-10-25/RESPONSE.md) (§D for REVIEW D).
**FR-1 and FR-3 are registered and playing; X59-S is BUILT and PRE-REGISTERED
([`PREREG.md`](../research_state/measurements/x59s_matched_students_2026-10-25/PREREG.md)), nothing trained; its
REVIEW E (NOT-READY) is answered by the PREREG's Amendment 2 (§2.4); X60 is REDESIGNED, pending review, conditional
on X59-S's outcome (§6).** No GPU was used. The power numbers are
[`plan_power.py`](../research_state/measurements/human_replay_quality_2026-10-25/plan_power.py)'s (seeded, CPU;
output `results/plan_power.json`) unless they cite REVIEW B's `review_b_power.py`; §6's are
[`x60_redesign_2026-10-26/power.py`](../research_state/measurements/x60_redesign_2026-10-26/power.py)'s.

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
- **No evidence yet, either way, that human data helps goal (b).** Human teams reach the unfamiliar region and human
  choices are learnable; neither says that learning them improves play.
- **What the panel exclusion KEEPS is a different population (REVIEW D M7).** Of the candidate battles (a ≥ 1300
  non-bot seat, ≥ 11 turns, every held-out panel incl. `forum216b`) it keeps 16.8 %; the kept battles are 64 %
  forfeits or timer losses (49 % among the excluded) and 52 % of their sides were seen whole (75 %). Every rowset's
  manifest records this profile, and a result trained on one is about that slice.
- **First reads, in order (§2):** (1) P\* vs Metamon's human-only `SmallRL` and (3) `LargeRL` / `LargeIL`: registered
  and playing; (2) the HOM's trace twin, read before X59-S's Δ_data is reported (an ARTEFACT read stops a
  SELF-PLAY-DATA-BETTER from closing anything; Amendment 2); (4) **X59-S**: three matched student arms
  × 5 seeds, our architecture from one init per seed — S-H (human states, human labels), S-P (the same states, P\*'s
  labels restricted to what the log allows), S-S (P\*'s own self-play states and labels). **Δ_data = S-H − S-S answers
  the owner's "is human data better than our self-play data?"**; Δ_labels = S-H − S-P routes H-BC.
- **The levers, re-ranked by the stated criterion (§4):** the reads first; then H-TEAMS with the D1 contrast; then H-BC
  if X59-S favours human labels; H-KL last. H-INIT is DROPPED (it forked a HOM fitted on rows the exclusion now
  forbids; REVIEW D M5). The existing HOM routes nothing.
- **X60 (§6) is REDESIGNED, pending review**, with both routes written before X59-S is read: X59-S's verdict picks
  X60-B (H-BC against a P\*-label placebo) on HUMAN-LABELS-BETTER, nothing on a closure, X60-T (H-TEAMS against D1)
  otherwise. One estimator, simulated as registered; the primary on F alone; an in-house meter that never saw human
  data; a positive control on the lever's own teams; GAIN 0.80 at +3 pp at the measured noise; ≈ 20–25 GPU-h and
  ≈ 95–140 CPU lane-h a route.

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
- **Bar (an interval, REVIEW D m4):** the gap difference (HOM's − P\*'s) with a battle-cluster bootstrap 95 % CI.
  ARTEFACT iff its LB > 0.05 nats: every HOM-based Kakuna score is a lower bound and C-GAME rows carry a learnable
  artefact. NO-ARTEFACT iff its UB < 0.05. Else UNCLEAR. A bound within 1e-9 of 0.05 takes UNCLEAR.
- **On X59-S's path for Δ_data only** (REVIEW E M3; PREREG Amendment 2 A2.5): S-P's labels never name a drawn
  option, so the signature is matched across S-H and S-P, but S-S's own side is true. FR-2 is read before Δ_data is
  reported; ARTEFACT ⇒ a SELF-PLAY-DATA-BETTER closes nothing.
- **Tool (built, not run):** `python -m main.x50_r1.read_fr2` pairs the two rows by `(battle, seat, k)`, checks each
  row's viewer against its trace's `trainee_side`, and refuses unless every row is the `--viewer` seat (only
  `opponent` is this registration). **R1's `--complete` extraction holds the TRAINEE's seat only** (X26's snapshot
  as viewer and actor; every one of its 151,234 mode 2 / 3 rows), so FR-2 reads its own extraction:
  `python -m main.x50_r1.run traces --complete --seat opponent` (the sentinel or bot as viewer and actor, the same
  guards). The opponent's own stream carries its REAL requests, so the true rows are mode 1 (the true team on real
  requests, labelled with the seat's own action), as registered; mode 3 (synthesised) stays a descriptive
  `--true-rows own_true`.
- **Cost:** forwards only, ≈ 0.2 lane-h at 2 threads for an extraction of R1's size (one part measured; Decision record).

### 2.3 FR-3: P\* vs `LargeRL` / `LargeIL` (Q-H10 at Metamon's largest human-only scale)

- **Downloaded and pinned** in `designs/ops/anchors.json` (hub `ac8e88e7`, checkpoint 40, sha256 recorded), and
  REGISTERED with FR-1 ([`human_only_vs_pstar_2026-10-25/PREREG.md`](../research_state/measurements/human_only_vs_pstar_2026-10-25/PREREG.md)).
- **Cells, rule:** as FR-1. **Cost:** ≈ 4.5 lane-h each (ESTIMATE).

### 2.4 FR-4: X59-S, the matched-student contest (Q-H9 and Q-H10 in our architecture)

**The binding text is the PREREG** ([`x59s_matched_students_2026-10-25/PREREG.md`](../research_state/measurements/x59s_matched_students_2026-10-25/PREREG.md),
committed before any student trains). Its shape, after REVIEW D:

- **The question (the owner's, D-M4):** is human DATA (human states, human labels) better than our self-play DATA (P\*'s
  states, P\*'s labels) for a policy of our architecture, at matched volume and recipe, on unfamiliar teams?
- **Rows:** the registered rowset `hr_spec_v1.r1` (`src/main/human_data/rowset.py`): `hr_spec_v1`'s TRAIN dates
  through the manifest check; rated, a result, ≥ 11 turns (D-M7's information floor), duplicates and bot-flagged
  accounts out, §3.1's exclusion over every held-out panel incl. `forum216b`, actor ≥ 1300, ≥ 2 legal actions, hidden
  turns with set labels (§3.2), the quitter's last 2 decisions dropped, account-disjoint validation (§3.3).
- **Three arms, five seeds each (5901–5905), one difference per contrast:**
  - **S-H**: the rowset, the human's action as the label;
  - **S-P**: the SAME rows, one action from frozen P\* at T = 1 RESTRICTED to the row's CONSISTENT set (D-M2: the
    legal actions the log leaves possible for the human; no drawn move, no drawn bench mon), matched noise keyed by
    the rowset digest and row id. On a hidden-move row both arms train the SAME set label;
  - **S-S**: P\*'s own self-play rows on the pool, both seats, P\*'s sampled action as the label, the same row count.
- **Matched:** `--arch production`'s network (P\*'s architecture) from one init per seed in every arm, one recipe, a
  fixed step budget (no early stopping), the seed-keyed batch order (S-H and S-P see the same batches).
- **Contrasts:** **Δ_data = S-H − S-S** (the owner's read); **Δ_labels = S-H − S-P** (routes H-BC); the states part
  S-P − S-S is reported. Each on pooled U vs Kakuna (H, F, P72 at 400 games each per student), averaged over seeds.
- **The interval carries the seed variance (D-M3):** the binomial part plus 2 × max(1.5 pp, the measured seed SD)² / 5;
  a Welch t across seeds replaces it when the measured seed SD's one-sided 95 % lower bound exceeds 1.5 pp. Sized at
  p = 0.5, so no guessed base rate enters: P(LB > 0) at +4 pp is 0.94 / 0.89 / 0.81 at a seed SD of 0 / 1 / 1.5 pp
  (false positives ≤ 0.02 there, ≤ 0.028 at 2–3 pp; `power.py`).
- **Floor pilot first:** 240 U games per student; if every arm's mean is < 0.15, the in-house head-to-heads decide.
- **After REVIEW E (Amendment 2):** S-S's team draw faces the human panel rule (a simulated human reveal, redrawn
  while near; 16.6 % of pool pairs pass against the 16.8 % of human battles kept); the Kakuna-family audit (§3.4)
  runs first and its flagged accounts leave the rowset before training; the quitter's last 10 decisions are dropped
  (`hr_spec_v1.r1` → a derived rowset); S-P is named "P\*-restricted-to-human-support" and routes H-BC only if it
  learned its teacher; a descriptive unrestricted arm S-P0; floor references (random-legal, max-damage) and an
  UNINFORMATIVE class; NO-DIFFERENCE-AT-4PP; KAKUNA-SPECIFIC when the in-house contrast is ≤ 0; a CUDA pre-flight;
  a conditional ≥ 1500 follow-up; the closure scoped to "1300+ ladder data at this mix and volume".
- **Cost:** training 8.16 GPU-h of step time under a lease (15 primary + 5 S-P0 students × 3 epochs at the MEASURED
  208.6 ms a step, PREREG Amendment 3; ≈ 3–4.5 h wall at 3 students at once, ESTIMATE; ≈ 3.6 h a student on the CPU, so
  not on the CPU); P\* labels and self-play rows on the CPU; games ≈ 80 lane-h (ESTIMATE at
  ≈ 10 s a Kakuna game, S-P0 included).
- **What it cannot say:** from-scratch students are not a plateau fork; a verdict is about the ≥ 1300 forfeit-heavy
  kept slice (mostly 1300–1499 actors); HUMAN-LABELS-BETTER does not show that H-BC inside PPO helps (X60 does).

### 2.5 Descriptive, optional: the HOM's Δ_U (REVIEW B B1 fix 2) — CONTAMINATED, routes nothing (REVIEW D M5)

The existing HOM was fitted on rows that kept 3-set cores of H, F and P72 (its exclusion was heldout72 ≥ 4-fit), so a
Δ_U on F + P72 may read panel memory as skill. It is barred, by name, from every lever and every routing rule. If read
at all, the HOM vs Kakuna is a DESCRIPTIVE read labelled CONTAMINATED: Δ_U = HOM_U − P\*_U, one-sided (LB > 0 is weak
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
  `forum216b`) and **83.2 % of `hr_spec_v1.r1`'s candidate battles** (a ≥ 1300 non-bot seat, ≥ 11 turns) with
  `forum216b` (216 teams) included; the rowset manifest records the realised share.
- **What it keeps (REVIEW D M7), measured on the same candidates:**

  | | KEPT (27,586 battles) | EXCLUDED (136,852) |
  |---|---:|---:|
  | forfeit or timer loss | 64.1 % | 49.3 % |
  | mean turns | 28.4 | 30.4 |
  | mons seen per side | 5.14 | 5.66 |
  | sides seen whole (six mons) | 51.9 % | 74.6 % |

  REVIEW D's ≥ 1500 read (before the ≥ 11-turn floor, `forum216b` not read) found the kept sides ENRICHED in
  pool-unfamiliar teams (31.5 % vs 11.6 %): the exclusion removes the mainstream. Every rowset manifest records this
  profile (`scan.json`, `rows_manifest.json`); a result trained on a rowset is about that slice.
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
- **Score:** for every account of the rowset with ≥ 10 kept games (X59-S: every actor is ≥ 1300; REVIEW E B2),
  on a seeded 10 of its games, the mean log-likelihood of its actions under Kakuna (Metamon's own replay parser and
  observation, CPU) minus that under P\* (REVIEW D m7: the HOM trained on these accounts, so it would flag a
  high-volume fleet LESS often; the HOM's version is descriptive).
- **Flag:** accounts above the p99 of that score's distribution over accounts with ≥ 1 chat line a game (a human
  reference; every scored account if fewer than 100 qualify); report the flagged share of rows. A POWER control:
  Kakuna's own captured seats, pooled like an account, must clear the reference's p99 under Kakuna.
- **Use (X59-S Amendment 2):** the audit runs BEFORE training; flagged rows > 1.0 % of the TRAIN rows ⇒ the flagged
  accounts leave the rowset before any student trains (no re-read, no retrain). X60 applies the same rule to its
  rowset.

### 3.5 Smaller rules

- **Forfeit tails (A m3; REVIEW E M1):** drop the quitting side's last 10 decisions (P\*'s agreement with the
  quitter falls over its last ~8 kept decisions while it rises for every other seat); report NLL by
  decisions-before-quit.
- **The stale fill prior (A m4):** the Smogon window is 2025-05..2026-04; measure the cost with a lag-matched prior on a
  held-out slice before any lever reads.
- **Circular calibration (A m5):** X60 reports the belief / intent heads' calibration on the Kakuna cells, ON vs OFF.
- **Choice Band (A m6):** report the share of rows whose active used one move all stay with no item revealed; mark
  their masks uncertain.
- **The drawn-move signature (B M6, REVIEW D M2):** a drawn move is never the human's label and a drawn bench mon is
  never a switch target. X59-S matches it across the label arms by construction: S-P's label is drawn from the row's
  CONSISTENT set (the legal actions the log leaves possible), so no S-P label names a drawn option either. It is NOT
  matched against S-S, whose own side is true (REVIEW E M3): FR-2 measures whether a model learns the signature, and
  gates X59-S's closure.
- **Forfeits and short games (REVIEW D M7), registered for X59-S:** battles of ≤ 10 turns are FILTERED (the
  information floor; it also removes every walk-over); forfeits beyond that are KEPT, the quitter's last 2 decisions
  dropped; the row share by ending is in the rowset manifest.

---

## 4. The levers, ranked for goal (b)

The criterion: expected pp on U vs Kakuna per GPU-hour, at an acceptable risk to home72. Measured anchors: D1 (+2.1 pp
[−0.9, +5.1], breadth via forum teams, P −1.6) and X52 (practice +3.2 [−0.2, +6.8]). Every other number is a GUESS.

| rank | lever | what changes | why | expected pp (b), GUESS | GPU | main risk |
|---|---|---|---|---|---|---|
| 0 | **the first reads** (§2) | nothing | they decide which lever is worth a build | n/a | 0–2 h | none |
| 1 | **H-TEAMS** `teams:<rowset>@<share>` | the POOL: completed ≥ 1500 human ladder teams join the self-play draw on both sides | the only lever with a measured sibling (D1); breadth on the ladder population | +0 to +4 | ≈ 13 per 3-seed arm | panel memory (§3.1); Smogon-drawn fills (Q-H13) |
| 2 | **H-BC** `bc:<rowset>@<coef>` | the LOSS (an auxiliary BC term) | human LABELS on states P\* never trained; built first only if X59-S reads HUMAN-LABELS-BETTER | +0 to +4 | ≈ 16 per 3-seed arm | a pull toward 1500-level play |
| 3 | H-KL `kl:<model>@<coef>` | the LOSS: KL to a registered `hom_<rowset>` on self-play states | piKL-style; acts where we are already strong | +0 to +1 | ≈ +15 % | in-pool play pulled toward the HOM |
| — | HB0 | none: X59-S's S-H IS the human-only comparator in our architecture | answers Q-H10 | n/a | in X59-S | none |
| — | ~~H-INIT~~ (a fork of the HOM, plain PPO) | DROPPED (REVIEW D M5) | the HOM it forks kept 3-set panel cores in its rows; it returns only as a new registration on a `hom_<rowset>` re-fit on a §3 rowset with account-disjoint early stopping (compute is small; what it loses is VOLUME: the compliant ≥ 1500 rows are `hr_spec_v1.r1`'s `rating_ge1500_rows`, against the HOM's 1.76M) | — | — | — |
| — | BC warm start of a FRESH run | the INIT of a new run | AlphaStar's supervised init | unknown | ≥ 42 for a pair to 100M | cost |

- **Struck:** "H-BC is the only lever whose gradient lands on unfamiliar-team states" (false: H-TEAMS, D1 and X52 all
  do; what only H-BC adds is human labels). The literature (AlphaStar's KL, Metamon's pretraining) supports the
  init-plus-KL family, not an auxiliary BC term added at a plateau (UNVERIFIED readings, from memory).
- **Rejected:** human game OUTCOMES as critic labels (off-policy); the HOM as a self-play opponent (FACE_c +0.017); an
  advantage-weighted BC variant now (one lever at a time).
- **Barred (REVIEW D M5):** the existing HOM (`~/gen3ai_archive/x50_r1b/hom/`) from every lever, every `kl:<model>`
  and every routing rule; any lever model is a `hom_<rowset>` fitted on a registered §3 rowset.

---

## 5. The complexity budget

One clause, one parameter, one rowset or model id per run; every other knob lives in the rowset or the
pre-registration; one lever per A/B; one shared control. The grammar and its rules are the architecture doc's §6.

---

## 6. X60: the first human-data lever A/B, conditional on X59-S — REDESIGNED, pending review

> **REDESIGNED (2026-10-26), pending adversarial review. Not registered, not built, not launched.** It replaces the
> NOT-READY draft (REVIEW D B1 and the other listed items; the fix table is §6.8). Both routes are written now,
> BEFORE X59-S is read, so the route that fires carries no choice made after its outcome. After the review, the
> route X59-S picks is committed as a PREREG (with its read specs) before its arm launches. Every power number is
> [`power.py`](../research_state/measurements/x60_redesign_2026-10-26/power.py)'s (seeded, CPU, under a minute;
> output [`results/power.json`](../research_state/measurements/x60_redesign_2026-10-26/results/power.json)).

### 6.0 X60 in one minute

- **The question:** does ONE human-data clause, added to P\* for 20M steps, raise its win rate against Kakuna on
  human-built teams it never trained on (forum72, F)? And is the gain the human-DATA part, or would a stand-in
  that carries no replay data do as well?
- **X59-S picks the clause** (§6.1): human labels beat P\*'s → **X60-B** (H-BC, contrast: a P\*-label placebo);
  X59-S closes human data → **no X60**; anything else → **X60-T** (H-TEAMS, contrast: D1's forum teams).
- **Five numbers decide**, the same in both routes: **Δ_F** (Kakuna, F; the primary), **I_F** (in-house on F: a
  meter that never saw human data), **Δ_P** (in-house on our own teams: the cost), **R72** (the lever moved the
  teams it trained on: the positive control), and **the contrast** (is it the human part?).
- **Its power, with the noise we measured:** a +3 pp gain reads GAIN 80 % of the time; a true zero reads GAIN ≤ 1 %
  and NO-USEFUL-GAIN 79 % (§6.7).

### 6.1 Which route fires (mechanical, from X59-S's registered verdict)

X59-S's read code returns its actions (`main.human_data.x59s_read.verdict`). X60 routes on them; no judgement enters.

| X59-S's actions contain | X60 | why |
|---|---|---|
| "closes 1300+ ladder data at this mix and volume" | **none** | X59-S's §5 closure: neither H-TEAMS nor H-BC is built |
| "H-BC first (against P\*-restricted-to-human-support)" | **X60-B** | HUMAN-LABELS-BETTER, not KAKUNA-SPECIFIC, not UNINFORMATIVE, and S-P learned its teacher (A2.4) |
| anything else (UNCLEAR, NO-DIFFERENCE-AT-4PP, OWN-LABELS-BETTER, KAKUNA-SPECIFIC, UNINFORMATIVE, or HUMAN-LABELS-BETTER with S-P not learned) | **X60-T** | "H-TEAMS first, with the D1 contrast" |

The closure line wins when both appear. One lever per A/B: the route that did not fire is a later, separate
registration.

### 6.2 Which panels are clean for a human lever (REVIEW B M1; Q-H13 per lever)

"Clean" has two parts: no near-copy of a panel TEAM in training (the §3.1 exclusion settles that for every panel), and
no training on the panel's POPULATION (no rule settles that; it decides a panel's role).

| panel | what it is | for H-TEAMS | for H-BC | role in X60 |
|---|---|---|---|---|
| **F** `forum72` | human-built forum pastes | team-clean; a different population (forum, not ladder) | team-clean; different population | **PRIMARY** (both routes) |
| P72 `p72` | procedural teams from the Smogon usage tables | **adjacent**: the corpus's hidden sets are Smogon draws from the same tables (REVIEW B: "H-TEAMS literally trains on Smogon-drawn sets, the generator of P72"), and its species mix is the ladder's usage | adjacent: the BC rows' own sides carry the same draws | reported |
| H `heldout72_hpfix` | ladder replay teams with usage-filled sets | **its own population and construction** (C-GAME completes ladder sides exactly as `hl_05_26` was built) | its own population | reported; X60-T's contrast arena |
| home72 | our sample teams (in the pool) | in-pool | in-pool | co-primary P (in-house) |
| `forum216b` | sealed forum pastes | team-clean | team-clean | SEALED until X54; X60 never reads it; the clean claim is X54's |
| **R72** (new, PRACTISED) | 72 teams drawn from the lever's OWN rows | trained on, by design | piloted in the BC rows, by design | positive control (§6.3) |

So F alone carries the claim, as in X61P (its review's B1 put a generator-adjacent panel out of the primary). This
departs from the goal-(b) statistic (F + P72, ruling `4f555e57`): an F GAIN is "human data moves human-built teams it
never saw", and U is reported with its contamination named.

### 6.3 The shared frame (both routes)

| item | registration |
|---|---|
| **arms** | ON = P\* (`rb_x26_s2601` @ 280,000,170) forked +20M with ONE `--human-data` clause (§6.4 / §6.5), C0's argv token for token otherwise; OFF = **C0** (`rb_x35_c0_s3501`–`s3505`, production pin `f4b25238`), reused under the C0 reuse check (below), 0 GPU-h |
| **pin** | a side pin = `f4b25238` + the human-data switch commits only (D3's and N1-P's pattern); `kit pin-check` proves the diff |
| **seeds** | **3501–3505**, C0's. Look 1 trains 3501–3503; 3504–3505 only on CONTINUE. Fixed: no extension, no adaptive seed rule (the seed term is inside the interval, below). A final that does not reach step 300,000,170 is re-launched with its seed |
| **PRIMARY Δ_F** | ON − OFF vs Kakuna ckpt 34, greedy v greedy, `forum72`'s committed symmetric config, **60 role-balanced 20-game units per seed final (unit seeds 0–59, 1,200 games)**: X61P's cell, so C0's units are imported from X61P's read state when its regime id matches and D3's determinism guard passes (one imported cell re-played game for game), and played otherwise |
| **the interval** (D-B1) | Δ̂ = ON's pooled F rate − OFF's. V = each arm's binomial variance `p (1 − p) / (1,200 n)` summed, **plus `2 × max(1.0 pp, σ̂)² / n`** for the seed-to-seed spread (n seed finals per arm), σ̂² = W − b (W the mean of the two arms' sample variances of their per-seed F rates, df 2 (n − 1); b the mean per-seed binomial variance). **Guard:** if σ̂²'s one-sided 95 % LOWER bound, `W × df / χ²(df, 0.95) − b`, exceeds (1.0 pp)², a Welch t interval on the per-seed rates decides instead. 95 % two-sided. No sign rule (the seed term does its job). This is the form of X59-S's registered rule (`x59s_read.contrast`), on one panel, with a 1.0 pp floor (X59-S's from-scratch students carry 1.5) and the two arms' spread |
| **looks** | **Look 1 (3 v 3) is a FUTILITY look only:** Δ̂ ≤ 0 (within 1e-9 counts) ENDS the read with look 1's class (never GAIN); otherwise look 2 (5 v 5) decides. No alpha is spent at look 1, so the GAIN rate under a true null stays ≤ the nominal 2.5 % at σ ≤ 1.0 pp and reads 2.9–3.3 % at 1.5–2 pp (§6.7; D1's rule, which can end GAIN at look 1, reads 3.1–7.2 %) |
| **classes** (rule 8: a bound within 1e-9 of 0 or +3.0 takes the undecided class) | **GAIN** iff LB > 0; **LOSS** iff UB < 0; **NO-USEFUL-GAIN** iff 0 ≤ UB < +3.0 pp (a gain of 3 pp is excluded); else **NOT-RESOLVED**. "Equivalent" (the interval inside ±3 pp) is reported beside, deciding nothing |
| **the clean meter I_F** (D-M6) | ON − OFF in-house on `forum72`: X35's population score S of each final against the FROZEN reference population R (`reference_population_production.json`; our own checkpoints, no human decision in any of them), 7 × 500 mirrored pairs a final (`main.h2h play-many --panel forum72`), Welch 95 % across finals. A Δ_F GAIN whose I_F point estimate is ≤ 0 is **KAKUNA-SPECIFIC** (opens nothing) |
| **CO-PRIMARY Δ_P** | D1's rule verbatim: S on `home72` vs R, 7 × 500 pairs, Welch 97.5 %, **NON-INFERIOR** iff LB > −2.0 pp. C0's P rows imported from X61P's read when the regime matches |
| **goal (a) guard** | ON's Kakuna `home72`, 7 units a final (700 games pooled): Wilson LB > 0.50 (P\* reads 0.598) |
| **positive control R72** (REVIEW B M3) | 72 teams drawn from the lever's OWN rows (seeded, ≤ 1 per account, HP-repaired, validator-clean), registered in `heldout_panels.json` as a PRACTISED set (not `held_out_from` training, so no rowset retires). ON − OFF in-house on R72 (S vs R, 7 × 500 pairs a final, Welch 95 %). **DELIVERED** iff LB > 0. It asks: did the lever, at this dose and plasticity, move the teams it trained on? |
| **X35's guards** | G-F, G-E, G-C, read only on a Δ_F GAIN (D1's convention) |
| **reported** | H, P72 at 7 units a final (X60-T: H at 30, §6.4); U and the goal-(b) F + P72 statistic, each with its contamination named; per-team proximity terciles (§3.1); the belief / intent calibration on the Kakuna cells (A m5); the 10M checkpoints are not read |
| **VOID** (checked in a pre-launch smoke, ≈ 0.2 GPU-h, and at each final) | the startup stamp `[HUMAN-DATA] lever=… rowset=… rows_sha=…` missing or not the registered one; the rows' layout hash ≠ the trainee's; the realised human share outside ±10 % of the registered one; a panel contamination hit (`main.heldout_panels.contamination` on the completed teams); K9(b) FAIL; a FATAL naming a human-data row or team |
| **C0 reuse check** (before launch) | at the side pin with `--human-data off`: `learner_golden` and the played-bits sentinel equal `f4b25238`'s; the null-dose ON build (share or coefficient 0) reproduces OFF's played bits; plus `data/` content hashes, the Kakuna harness commit and units, R's sha, the sim binary hash, torch / CUDA / driver versions. Any miss: a fresh C0h (5 × ≈ 4.1 GPU-h) |
| **scope** | an ADOPT-CANDIDATE at P\* is re-read at the production checkpoint before it steers production (X26c read FLAT on U vs P\*), and its clean claim is X54's on `forum216b` |

### 6.4 Route T: X60-T, H-TEAMS against D1

- **The clause:** `--human-data teams:hr_spec_v1.r<n>@0.5`, team sides both: D1's dose (`--team-mix w_forum_v1@ABC:0.5
  --team-mix-sides both`), so the contrast differs in the corpus alone.
- **The rowset `hr_spec_v1.r<n>`** (the next free id at registration; its chain `hr_spec_v1` `e7d491e7…` → r1
  `e9e0df42…` → r2 `2a413c85…` → r<n>): the kept battles of r2 (§3's rules, the Kakuna-family audit's drop applied);
  seats at actor ≥ 1500 whose six mons were all seen (fewer invented species; D-M7: 52 % of kept sides);
  C-GAME-completed, Hidden Power repaired (`gen3_hp_repair_v1`), validator-clean, label-path soaked; ≤ 5 teams per
  account; both panel nets (§3.1 before the fill, the exact-set guard after). Its size is measured at the build
  (≈ 2,000–3,000 teams, ESTIMATE; D1's corpus held 1,981). It is also the corpus `n1_hrep_v1` (X61 PREREG §10;
  X53-N1's N1H), built once under one id; its two entry points (`--team-mix n1_hrep_v1`, `--human-data teams:`)
  become one provenance record before either arm launches (§9 FINDING 10).
- **The contrast T − D1** (REVIEW B M2): H-TEAMS's five finals against D1's (`rb_x35_d1wide_s3501`–`s3505`) on **H +
  F, equal weight**, units 0–29 of each (600 games per final and panel; D1's units 7–29 are new), §6.3's interval, one
  look, on the seeds the primary ended with (3 v 3 if look 1 ended it, else 5 v 5). **The arena is chosen so each arm owns one panel:** F is D1's own population, H is H-TEAMS's.
  Classes: **LADDER-BETTER** iff LB > 0; **FORUM-BETTER** iff UB < 0; **NO-DIFFERENCE-AT-3PP** iff inside ±3 pp;
  else NOT-RESOLVED. Read whatever Δ_F's class (no selection on the primary). Per-panel rows reported. D1's banked
  units 0–6 pool with the new ones only under D3's determinism guard (one banked cell re-played game for game).
- **R72** = 72 of r<n>'s teams: in ON's pool, never in OFF's or D1's.

### 6.5 Route B: X60-B, H-BC against a P\*-label placebo

- **The clause:** `--human-data bc:hr_spec_v1.r2@c`: an auxiliary cross-entropy on the rows X59-S validated (r2,
  1,201,538 TRAIN rows; set labels on hidden-move rows, X59-S §3's loss), added to the PPO loss.
- **The coefficient c, fixed before ON trains:** in the pre-launch smoke, the largest value of {0.01, 0.02, 0.05, 0.1,
  0.2, 0.5} whose median ratio of the BC term's to the PPO policy loss's gradient norm on the shared trunk is ≤ 0.10
  (A-5's "≈ 10 % of the policy's moves"); the human rows per update are fixed in the PREREG. A-5's falsifier stays a
  VOID: the BC loss falls < 10 % over the smoke.
- **The placebo P-BC:** `--human-data bc:hr_spec_v1.r<n>@c`, r<n> = r2's rows with X59-S's registered S-P labels
  (`labels_pstar`, manifest `2512c8a7…`: P\* restricted to what the log allows the human, the same matched noise,
  the same set labels on hidden rows). Same rows, coefficient, schedule and seeds (all five; no futility look); only the labels differ. It
  answers "the human labels, or any extra cross-entropy on human states?" (REVIEW B M2).
- **It trains only after an H-BC GAIN** (≈ 25 GPU-h saved otherwise). **The contrast reads FRESH games**: H-BC's
  finals re-played on unit seeds 60–119 (1,200 F games a final), against P-BC's at 0–59, §6.3's interval, one look,
  so the binomial noise that selected the GAIN is not re-used (§6.7: the selection that remains, through the seed
  effects, gives a false LABELS-MATTER of 0.9 % / 2.3 % / 4.9 % at σ 0 / 1.0 / 1.5 pp). Classes: **LABELS-MATTER**
  iff LB > 0; **PLACEBO-AS-GOOD** iff UB < +3.0 pp; else NOT-SEPARATED.
- **Delivery, two checks:** R72 (72 completed full-six human sides of r2's TRAIN battles, ≤ 1 per account) and the
  label check: ON's NLL of the human label on r2's account-disjoint VALIDATION rows below OFF's (Welch across the 5 v
  5 finals, UB < 0). DELIVERED needs both.

### 6.6 Verdicts and what each licenses (both routes; nothing is adopted by a rule alone)

| verdict | rule | licenses |
|---|---|---|
| **ADOPT-CANDIDATE** | Δ_F GAIN, I_F point > 0, Δ_P NON-INFERIOR, goal (a) holds, G-F / G-E / G-C PASS | the re-read at the production checkpoint, then X54. Reported with its attribution: X60-T LADDER-BETTER = "ladder data beats forum-team breadth", FORUM-BETTER = "D1's corpus does it better: prefer it", else "not separated from breadth"; X60-B LABELS-MATTER = "the human labels", PLACEBO-AS-GOOD = "the human states, not the labels", else "not separated" |
| **TRADE** | Δ_F GAIN, I_F > 0, and Δ_P's inferiority not excluded, or goal (a) fails, or a guard FAILs | the exchange rate Δ̂_F / −Δ̂_P to the owner |
| **KAKUNA-SPECIFIC** | Δ_F GAIN, I_F point ≤ 0 | nothing (a gain only the human-pretrained meter sees) |
| **LOSS** | Δ_F LOSS | drop the lever at the plateau |
| **NO-TRANSFER** | Δ_F NO-USEFUL-GAIN, DELIVERED | stop this lever at the plateau: it moved what it trained on, not unfamiliar teams. A fresh-run or warm-start use stays open |
| **NOT-DELIVERED** | Δ_F NO-USEFUL-GAIN, not DELIVERED | closes nothing about the lever's value: at this dose the plateau did not move even practised teams (REVIEW B M3's (b) / (c)); a dose or fresh-run follow-up is a new registration |
| **NOT-RESOLVED** | otherwise | nothing licensed; the next lever of §4; no extension |

### 6.7 Power with the measured noise

**The noise, measured** (`power.py` part 1, from the committed per-seed counts of C0 and D1's five finals and D3's
three, 140 F games each): the seed-to-seed SD of a P\*+20M final's F rate is **0 at the point** (φ̂ 0.56, df 10),
one-sided 95 % upper bound **2.75 pp** (F alone) and 1.74 pp (F + P72). Pairing does not help (the belief-prior-mix
FINDING: twin games correlate little), so the power assumes independent games. Base F rate 0.49 (C0's). The floor
σ = 1.0 pp is a judgement inside that range; σ̂ is reported at every read.

**PRIMARY Δ_F** (F alone, 1,200 games a final, 5 v 5, the two-look rule of §6.3; 40,000 draws a cell):

| true seed SD | GAIN at +2 / +3 / +4 pp | false GAIN at 0 | NO-USEFUL-GAIN at 0 | half-width (look 2) | Welch guard fires |
|---:|---|---:|---:|---:|---:|
| 0 | 0.41 / **0.80** / 0.97 | 0.008 | 0.79 | 2.18 pp | 0.004 |
| 0.5 pp | 0.41 / 0.78 / 0.96 | 0.010 | 0.76 | 2.18 pp | 0.008 |
| 1.0 pp | 0.38 / 0.71 / 0.91 | 0.020 | 0.66 | 2.18 pp | 0.05 |
| 1.5 pp | 0.32 / 0.57 / 0.80 | 0.029 | 0.52 | 2.47 pp | 0.20 |
| 2.0 pp | 0.24 / 0.43 / 0.64 | 0.033 | 0.37 | 2.93 pp | 0.45 |

- **Against the draft's rule:** D1's AM-5 point φ̂, GAIN possible at look 1, reads a false GAIN of 0.031 at σ 0
  and 0.072 at σ 2 pp. REVIEW D's t-at-df fix keeps the false-GAIN rate near this rule's (0.009 / 0.019 at σ 0 /
  1.0) with a little less power (0.78 / 0.66 at +3 pp) and a two-sided NULL of 0.28 at Δ = 0. Both are in
  `results/power.json`.
- **Larger cells:** 1,600 games a final lift +3 pp to 0.89 (σ 0) / 0.78 (σ 1.0); 2,000 to 0.95 / 0.83; +400 games
  a final costs ≈ 11 lane-h. **Eight seeds a arm** (D-B1 item 3, costed): +3 pp 0.96 / 0.89 / 0.78 at σ 0 / 1.0 /
  1.5, for ≈ +25 GPU-h (three ON seeds and three new C0 seeds). Neither is registered: at the measured noise five
  seeds and C0's own cell already give 0.80.
- **The contrast T − D1** (H + F, 600 a panel): half-width 2.17 pp; LADDER-BETTER at +3 pp 0.80 / 0.70 / 0.58 at σ 0 /
  1.0 / 1.5; false 0.008 / 0.020 / 0.028; NO-DIFFERENCE-AT-3PP at 0: 0.61 / 0.44 / 0.27.
- **The placebo** (after a GAIN, fresh games): LABELS-MATTER when H-BC gains +4 and P-BC +1 pp: 0.80 / 0.71 / 0.61 at
  σ 0 / 1.0 / 1.5.
- **In-house (I_F, Δ_P, R72):** the measured per-final SD of S is 0.37 pp (C0) and 0.86 pp (D1), so a Welch 5 v 5 MDE
  at 80 % is 0.74–1.75 pp.

### 6.8 The NOT-READY items, and where each is fixed

| item | the draft | now |
|---|---|---|
| D-B1.1 one AM-5 multiplier | an upper-quantile φ registered, the point φ̂ powered | ONE rule (§6.3): the binomial part plus a seed term with a 1.0 pp floor and a Welch guard; the same rule is simulated (`power.py`) and coded (§6.9 item 2, ported verbatim) |
| D-B1.2 power from the rule | NULL 0.58 / GAIN 0.77 quoted for an unregistered rule | §6.7, of the registered rule, at 1,200 games a final |
| D-B1.3 the seed rule | fired 67 % at a true SD of 0; the 8-seed path uncosted | no adaptive rule: 5 fixed seeds and the seed term inside the interval; 8 seeds powered and costed (§6.7) |
| D-B1.4 the contrasts' width | 1.74 pp, no seed term | both contrasts use §6.3's interval (§6.4, §6.5) |
| D-M6 a meter not trained on human data | in-house U as a point guard, unpowered | I_F in-house on the primary panel, its interval reported, KAKUNA-SPECIFIC when ≤ 0 |
| panel contamination (B M1) | F + P72 primary | F alone; P72 named fill-adjacent and H the lever's own population (§6.2) |
| contrast arms (B M2) | H-TEAMS − D1 on F + P72 (D1's own panel); P-BC after a GAIN on the same games | T − D1 on H + F, one panel owned by each; P-BC on fresh games |
| positive control (B M3) | H's practised half H-p IN ON's rows | R72 + (X60-B) the label check. H-p was dropped: it changes the treatment (ON would no longer be the lever adopted), adds an unmeasured number of mainstream battles (`forum216b` alone is near 76 % of the candidate battles, so the near-only-H-p share is unknown), and burns H for the lineage |
| NULL / NO-DIFFERENCE | NULL inside ±3 pp: 0.16–0.30 at Δ = 0 under its registered rule (REVIEW D); a two-sided band is rarely reached after a futility stop (0.36 under §6.3) | NO-USEFUL-GAIN (UB < +3 pp: 0.79), split by delivery into NO-TRANSFER and NOT-DELIVERED |
| realistic power | assumed seed SD 0 or 2.7 pp | measured σ (point 0; upper 2.75 pp), pairing credited nothing, swept 0–3 pp |

Dropped as well: the per-seed sign rule (redundant with the seed term), the learning-curve and data-scaling reads (no
rule read them; R72 asks the delivery question directly), the heterogeneity guard (one primary panel).

### 6.9 Build list, cost, ids

**Builds (agent-days, ESTIMATE), in order:**

| # | build | T | B |
|---|---|---:|---:|
| 1 | the switch `agents.info_levers.human_data` (grammar, hash, `record()`, provenance in the config / metadata / lineage, the fork / resume refusal, the consumer snapshot check, the off-means-off gates; arch doc §6) | 1.5 | 1.5 |
| 2 | the read rule: an X60-keyed variant in `main.x35.d1` and a `main.reads` rule (§6.3's interval ported from `power.py`, the futility look, the classes, I_F, R72, the contrasts, the verdict table); D1's, D3's and X61P's reads re-decide unchanged (golden) | 1.5 | 1.5 |
| 3 | R72: the draw, its registration as a PRACTISED set, its `main.h2h` panel | 0.5 | 0.5 |
| 4 | H-TEAMS: the r<n> team corpus (export, completion, HP repair, validator, soak, cap, both nets) and `teams:` resolving to the team-mix path | 1.5 | — |
| 4 | H-BC: the BC term in the learner (after every stash reader; K9; `human/bc_loss`, `human/bc_coef`, `human/rows_seen`, the validation NLL), the coefficient smoke, the r<n> placebo rowset | — | 3.3 |
| 5 | the kit arm(s), side pin, launch sheet, `checkargs` / `--dry-run` / `pin-check`, the read specs | 1 | 1.2 |
| 6 | the PREREG and two adversarial reviews (data / bias; experimental / causal), per the owner's rule | 1.5 | 1.5 |
| | **total** | **≈ 7.5** | **≈ 9.5** |

**GPU-h and CPU lane-h** (ESTIMATE; a fork ≈ 4.1 GPU-h from the 11 siblings measured at this pin and length,
3.84–4.43; Kakuna ≈ 10 s a game a lane and in-house ≈ 2.8 lane-h per 3,500-pair set, X61P §11):

| | X60-T | X60-B |
|---|---|---|
| GPU, look 1 (3 seeds) / look 2 (+2) | 12.3 / +8.2 | ≈ 14.7 / +9.8 (+20 % for the BC term, ESTIMATE; the smoke measures it) |
| GPU, smoke and calibration | 0.2 | 0.5 |
| GPU, conditional | — | P-BC after a GAIN: ≈ 24.5 |
| Kakuna games, ON | F 6,000 + H 3,000 + P72 / home72 1,400 ≈ 29 lane-h | F 6,000 + H / P72 / home72 2,100 ≈ 22 lane-h |
| Kakuna games, controls | D1's H and F units 7–29: 4,600 ≈ 13 lane-h; C0's F: 0 if imported from X61P, else ≤ 5,300 ≈ 15 lane-h | C0's F as X60-T; after a GAIN, H-BC fresh F 6,000 + P-BC F 6,000 ≈ 33 lane-h |
| in-house sets (7 × 500 pairs) | ON P, I_F, R72 + C0 I_F, R72 = 25 sets ≈ 70 lane-h (+14 if C0's P is re-played) | the same ≈ 70 (+14), + the label check ≈ 1 |
| **total** | **≈ 20.7 GPU-h; ≈ 112–141 lane-h** | **≈ 25 GPU-h (+24.5); ≈ 93–122 lane-h (+33)** |

The futility look ends a lever with no effect at look 1 half the time, saving its look-2 GPU and about 40 % of ON's
Kakuna games. **Wall time is set by the CPU queue**, not the GPU: D3's look 1 held units at 9–31 min (≈ 30–90 s
a game) on a contended box against the 10 s estimate.

**Toggle and snapshot ids carried by every result:**

| | ON | contrast arm | control |
|---|---|---|---|
| X60-T | `--human-data teams:hr_spec_v1.r<n>@0.5` (clause hash `sha256("human_data_v1:" + canonical)[:16]`); run `rb_x60t_s<seed>` | D1: `--team-mix w_forum_v1@ABC:0.5 --team-mix-sides both`, no human-data clause | C0, no clause (reads `off`) |
| X60-B | `--human-data bc:hr_spec_v1.r2@c`; run `rb_x60b_s<seed>` | P-BC: `--human-data bc:hr_spec_v1.r<n>@c`; run `rb_x60p_s<seed>` | C0 |

Each result cites the snapshot (`hr_spec_v1`, manifest `e7d491e7…`), the rowset id and its rows-manifest sha256, the
clause hash, the side pin, and R72's content digest. `hr_spec_v1.r<n>` is the next free rowset id when the firing route
registers its one new rowset (only one route fires).

### 6.10 What X60 cannot say

One dose, at one plateau checkpoint, for 20M steps; a NO-TRANSFER is about that, not about a fresh run or a warm
start. F is 72 human-built forum teams: a GAIN says the lever moves THEM. Kakuna is human-pretrained (I_F guards the
direction, not the size). The rows are `hr_spec_v1`'s kept slice (forfeit-heavy; X60-B's mostly 1300–1499 actors);
a verdict speaks for it, and a drift alarm (arch doc §8) sends it to a re-read.

### 6.11 The plan's cost, every step

| step | GPU-h | CPU lane-h | agent-days |
|---|---:|---:|---:|
| FR-1 (SmallRL) | 0 | ≈ 1.3 | ≈ 0.1 |
| FR-2 (trace twin) | 0 | ≈ 1 | ≈ 0.2 |
| FR-3 (LargeRL, LargeIL; downloads) | 0 | ≈ 9 | ≈ 0.3 |
| FR-4 X59-S (15 + 5 S-P0 students; labels and self-play rows on the CPU) | 8.16 of step time (MEASURED rate, Amendment 3) | ≈ 80 games + ≈ 5 rows / labels (ESTIMATE) | built |
| X59-S's conditional ≥ 1500 follow-up (Amendment 2 A2.7) | ≈ 0.7 (ESTIMATE) | ≈ 8 | ≈ 0.3 |
| the Kakuna-family audit (X59-S's rowset, before training) | 0 | ≈ 6 | done |
| X60-T or X60-B (§6.9) | ≈ 20.7 or ≈ 25 (+24.5) | ≈ 112–141 or ≈ 93–122 (+33) | ≈ 7.5 or ≈ 9.5 |

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
  D1's filters, a per-account cap of 5 and both panel nets (§6.4: it is also `n1_hrep_v1`, built once).

---

## 8. Assumptions, each with its falsifier

| id | assumption | falsified by |
|---|---|---|
| A-1 | Human labels carry unfamiliar-team skill P\* lacks | X59-S OWN-LABELS-BETTER: Δ_labels's registered 95 % interval (seed term included) lies wholly below 0 (5 seeds × 1,200 U games per arm; power 0.81 at a true −4 pp and a 1.5 pp seed SD) |
| A-2 | C-GAME rows teach nothing false | FR-2 ARTEFACT: the 95 % CI of (HOM's twin gap − P\*'s) lies wholly above 0.05 nats |
| A-3 | ≥ 1500 humans are good enough teachers that a small dose does not cost home72 | X60's Δ_P 97.5 % UB < −2 pp |
| A-4 | The spectator stream is representative of finished ladder games | the coverage read (Q-H4; needs approval): the 95 % CI of the largest between-tercile difference in P(spectated), by length or by rating, lies wholly above 20 pp |
| A-5 | A BC gradient ≈ 10 % of the policy's moves the trunk without dominating | the pre-launch smoke: the 95 % CI of the BC loss's relative fall (first vs last 10 % of the smoke's steps, by batch) lies wholly below 10 % (VOID before GPU), or A-3's falsifier |
| A-6 | The bot rule plus the Kakuna-family audit catch the bots that matter | §3.4's POWER control fails (Kakuna's own seats do not clear the human reference's p99), or a lever's Kakuna gain is KAKUNA-SPECIFIC (its in-house contrast ≤ 0; X59-S Amendment 2 A2.10) |
| A-7 | §3.1's exclusion removes team memory of the panels | X60's proximity report: Δ_F's top proximity tercile's CI excludes 0 while the bottom tercile's contains it (descriptive; per-team MDE 7.5 pp) |
| A-8 | 20M steps from a plateau is enough plasticity | X60's R72 interval lies wholly below +1.0 pp (the lever did not move the teams it trained on) |

Every falsifier is an interval statement (REVIEW B m7, REVIEW D m4): a bound within 1e-9 of its bar falsifies nothing
(the undecided class).

---

## 9. FINDINGS (standing rule 7)

1. **No offline evidence exists either way for A-1** (the familiarity proxy is uninformative, REVIEW A M1).
2. **The ≥ 3-fit exclusion costs 83.2 % of the candidate battles with `forum216b`** (≈ 71 % of ≥ 1500 battles
   without it), and what it keeps is forfeit-heavy and less fully seen (§3.1).
3. **No unfamiliar panel is fully clean for a human lever** (Q-H13; §6.2): every panel is team-clean after §3.1, but H
   is the levers' own population and P72 is fill-adjacent; F (human-built forum teams) is the cleanest open panel
   and carries X60's primary. `forum216b` (sealed for X54) carries the clean claim.
4. **The existing HOM is barred** from levers and routing (REVIEW D M5); its rowset is unregistered.
5. **X59-S's teacher is mostly 1300–1499 actors** (the ≥ 1300 floor buys volume for a from-scratch student); the
   card ranks that stratum "the human-only comparator only". The realised share is in the rowset manifest.
6. **X59-S's training needs a GPU lease** (≈ 10 lane-h a student on the CPU); no CPU pilot was trained, and the base
   rate vs Kakuna is measured by the registered floor pilot.
7. **X60 is REDESIGNED, pending review** (§6), not registered. Experiment ids X59-S / X60 are named here and in the
   PREREG only, not in `EXPERIMENT_BACKLOG.md`; the orchestrator assigns them.
8. **The Kakuna-family audit runs before X59-S trains** (Amendment 2 A2.2); its result is in the PREREG's Amendment
   2b.
9. **Process:** CPU only; no GPU, launch or download; `models/`, `data/` and the corpus were only read.
10. **X60's redesign rests on four things not yet true** (§6): the seed SD is measured only at 140 games a final
    (point 0, upper bound 2.75 pp on F), so the 1.0 pp floor is a judgement; C0's 60-unit F cells exist only if
    X61P's read plays them (s3504–3505 only on N1-P's CONTINUE); `main.x35.d1` needs an X60-keyed variant that leaves
    D1's, D3's and X61P's reads unchanged; and the CPU queue, not the GPU, sets X60's wall time. Also: H-TEAMS's
    corpus is X53-N1's `n1_hrep_v1` (arm N1H, via `--team-mix`), so one corpus has two entry points until the switch
    makes them one.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-25 | **CREATED (DRAFT; explicit-only)** on the owner's grant | the data card; H-BC > H-TEAMS > HB0 > H-KL > warm start; X59 (the HOM vs Kakuna) first; X60 in D1's frame | building H-BC before A-1 is tested; a factorial; pooling the archive | the card; R-b1, R-b2; D1 look 2; X52 |
| 2026-10-25 | **REVISED after REVIEW A / B / C** (REVISED DRAFT, pending re-review) | the owner's two questions as FIRST reads (SmallRL, the trace twin, LargeRL / LargeIL, X59-S); X59 withdrawn as a rule (descriptive Δ_U only); X60's primary on F + P72 at 630 games a panel, H as a positive control, the H-TEAMS − D1 and H-BC − P-BC contrasts, a goal (a) guard, a seed rule, an action for every class; ONE panel-exclusion rule on revealed facts at ≥ 3 fits before the fill; set labels on hidden turns; account-disjoint splits; a Kakuna-family audit; H-TEAMS re-ranked above H-BC, H-INIT added; the consent flag and `judge` dropped | keeping X59's premium rule (it rewarded forgetting); 140 games a cell (NULL unreachable); the ≥ 4-fit exclusion with a dose-response read (unpowered at ≈ 10 games a team) | `RESPONSE.md`; `plan_power.py` |
| 2026-10-25 | **REVISED after REVIEW D** (READY-WITH-FIXES; X60 NOT-READY) | X59-S rebuilt and pre-registered: THREE arms (S-S mandatory; Δ_data = S-H − S-S answers the owner, Δ_labels = S-H − S-P routes H-BC), FIVE seeds each with a seed term in the interval, sized at p = 0.5 with a floor pilot, S-P's labels restricted to the log-consistent set, ≤ 10-turn games filtered; H-INIT dropped and the existing HOM barred; X60 marked NOT-READY with the four fixes it needs and an in-house U guard; the kept population's profile on page one; every falsifier an interval | one seed per arm (a 2 pp seed SD made a false HUMAN-LABELS-BETTER 0.20 likely); S-S optional (the owner's question unread); re-fitting the HOM now (its compliant ≥ 1500 volume is small, and X59-S answers the question first); fixing X60 now (not on the path) | `RESPONSE.md` §D; `x59s_matched_students_2026-10-25/power.py`; `hr_spec_v1.r1`'s scan |
| 2026-10-26 | **X60 REDESIGNED (pending review), conditional on X59-S** | both routes written before X59-S is read (X60-B on "H-BC first", none on a closure, X60-T otherwise); the primary on F alone (P72 fill-adjacent, H the levers' own population); ONE interval (binomial + a 1.0 pp seed-term floor + a Welch guard, X59-S's rule) simulated as registered; look 1 a futility look only; NO-USEFUL-GAIN split by delivery into NO-TRANSFER / NOT-DELIVERED; the in-house meter I_F on the primary panel; the positive control R72 on the lever's own teams; T − D1 on H + F (one panel owned by each); P-BC after a GAIN on fresh games; 5 fixed seeds, C0 and D1 shared | D1's rule as is (false GAIN 3.1–7.2 %, GAIN at look 1); REVIEW D's t-at-df multiplier (no better at the same cost, NULL 0.28); F + P72 as the primary (P72 fill-adjacent); H's practised half in ON's rows (changes the treatment, burns H, an unmeasured dose); an adaptive seed rule; 8 seeds (+25 GPU-h for +0.16 power at the measured noise) | `x60_redesign_2026-10-26/power.py`, `results/power.json`; REVIEW D B1, M6; REVIEW B M1–M3 |
| 2026-10-26 | **FR-2 TOOL BUILT, NOT RUN** | `main.x50_r1.read_fr2`: pairs by `(battle, seat, k)` (1:1 in R1's extraction: 75,563 keys in both modes, 108 in mode 3 alone, 70,603 usable), R-b1's float64 masked policy NLL, R0's joint battle bootstrap, the sha gates, a resumable chunked cache; a viewer check against `trainee_side` that REFUSES the existing extraction, which holds the TRAINEE's seat only | reading the trainee seat as if it were the registered opponent seat; inventing an opponent-seat pairing | one part (29,902 rows, both models, 2 threads) took 144 s at 2.3 GB peak: 5 parts ≈ 12 min ≈ 0.2 lane-h for an extraction of this size |
