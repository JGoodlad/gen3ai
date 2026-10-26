# Human replays as a TOGGLEABLE training lever: the first reads, the levers, and the pre-registered plan (X59-S, X60)

**Status: REVISED (2026-10-25, second revision, after REVIEW D). EXPLICIT-ONLY** (the plan and its
pre-registrations; what is current lives in the always-current [`human_data_architecture.md`](human_data_architecture.md)).
The first revision answered REVIEW A (data / bias / leakage), B (experiments / causal) and C (reasoning); this one
answers REVIEW D, the re-review (READY-WITH-FIXES). Every disposition is in
[`RESPONSE.md`](../research_state/measurements/human_replay_quality_2026-10-25/RESPONSE.md) (§D for REVIEW D).
**FR-1 and FR-3 are registered and playing; X59-S is BUILT and PRE-REGISTERED
([`PREREG.md`](../research_state/measurements/x59s_matched_students_2026-10-25/PREREG.md)), nothing trained; its
REVIEW E (NOT-READY) is answered by the PREREG's Amendment 2 (§2.4); X60 is NOT-READY (§6).** No GPU was used. The power numbers are
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
- **X60 (§6) is NOT-READY** (REVIEW D B1: its estimator and its power did not match). It is not on the path: it runs
  only after X59-S routes a lever, and needs the four fixes in §6 before it is registered.

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
- **Cost:** forwards only, ≈ 1 lane-h; ≈ 0.2 agent-day to point `main.x50_r1.read_r0`'s trace path at the policy NLL.

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
- **Cost:** training ≈ 3.7 GPU-h under a lease (15 primary + 5 S-P0 students × 3 epochs, ESTIMATE; ≈ 10 lane-h a
  student on the CPU, so not on the CPU); P\* labels and self-play rows on the CPU; games ≈ 80 lane-h (ESTIMATE at
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

## 6. X60: a lever A/B (GPU; D1's frame with REVIEW B's fixes) — NOT-READY (REVIEW D B1)

> **NOT-READY, and not on the path.** REVIEW D B1: the table registers AM-5's widened Newcombe with an UPPER-QUANTILE
> φ, but the quoted power (NULL 0.58, GAIN 0.77) is for the POINT φ̂ that `plan_power.py` and `main.x35.d1` use; under
> the registered rule NULL is 0.16–0.30 and GAIN 0.39–0.63. The seed rule below fires 67 % of the time at a TRUE seed
> SD of 0, so its unpowered, uncosted 8-seed fallback is the expected path. Before X60 is registered:
> 1. register ONE AM-5 multiplier and implement it in `main.x35.d1` (t at df is the milder choice);
> 2. re-run `plan_power` with it and size the cells from it (t needs ≈ 1,000 games per (seed final, panel) for NULL
>    0.59 / GAIN 0.84);
> 3. set a seed bar 10 finals can resolve (the point estimate > 1.5 pp, or the one-sided 80 % bound > 2.5 pp),
>    simulate its firing rate at seed SD 0 and 1.5 pp, and register and cost the 8-seed classifier;
> 4. widen the H-TEAMS − D1 and H-BC − P-BC contrasts by the same multiplier.
>
> X60 runs only after X59-S routes a lever. The rows below are the draft, kept for those fixes, with REVIEW D's other
> X60 findings applied (the in-house U guard, D-M6; the H-p exception, D-m6).

| item | registration |
|---|---|
| arms | ON = P\* forked +20M with ONE clause (H-TEAMS, or H-BC / H-INIT per X59-S); OFF = the same fork with `off` |
| matched control | X35's C0 only if bit-identity holds at the arm's pin (below); else a fresh C0h |
| seeds | 3501–3503 (look 1), 3504–3505 (look 2), shared across arms |
| **PRIMARY** | Δ = ON − OFF vs Kakuna, pooled equal-weight over **F + P72**, AM-5's widened Newcombe (`main.x35.d1`, with an upper-quantile φ, REVIEW B m6), **630 games per (seed final, panel)** |
| power (seed SD 0) | NULL 0.58 at Δ = 0; GAIN 0.58 / 0.77 at +2.0 / +2.5 pp; half-width 1.74 pp at look 2. At a seed SD of 2.7 pp NULL falls to 0.05: the seed rule below |
| heterogeneity guard | GAIN also needs both F's and P72's pooled point estimates > 0 |
| **in-house U guard** (REVIEW D M6: a meter not trained on human data) | ON − OFF on F + P72 against the frozen reference population (or P\*) through `main.h2h`, CPU, thousands of games (half-width ≈ 1.5 pp); GAIN also needs its point estimate > 0, and a GAIN only Kakuna sees is reported "Kakuna-specific" |
| **CO-PRIMARY P** | Δ_P = ON − OFF on home72 in-house (`main.h2h`), Welch 97.5 %, non-inferiority bar −2.0 pp |
| **goal (a) guard** | ON's Kakuna home72 at the final (400 games pooled over seeds) keeps its Wilson LB > 0.50 |
| **positive control** | H split in two halves stratified by `ladder_weights_v1`; ON's rows INCLUDE a ≥ 3-fit battle near H-p only if EVERY one of its ≥ 3-fit matches is to an H-p team (REVIEW D m6: a battle that also fits an F or P72 team stays out), and EXCLUDE those near H-u (a named exception to §3.1); 420 games per half per seed final. H-p's Δ is the lever's ceiling at this dose (power 0.90 at +5 pp); H-u's is transfer |
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
| FR-4 X59-S (15 + 5 S-P0 students; labels and self-play rows on the CPU) | ≈ 3.7 (ESTIMATE) | ≈ 80 games + ≈ 5 rows / labels (ESTIMATE) | built |
| X59-S's conditional ≥ 1500 follow-up (Amendment 2 A2.7) | ≈ 0.7 (ESTIMATE) | ≈ 8 | ≈ 0.3 |
| the Kakuna-family audit (X59-S's rowset, before training) | 0 | ≈ 6 | ≈ 1 |
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
| A-1 | Human labels carry unfamiliar-team skill P\* lacks | X59-S OWN-LABELS-BETTER: Δ_labels's registered 95 % interval (seed term included) lies wholly below 0 (5 seeds × 1,200 U games per arm; power 0.81 at a true −4 pp and a 1.5 pp seed SD) |
| A-2 | C-GAME rows teach nothing false | FR-2 ARTEFACT: the 95 % CI of (HOM's twin gap − P\*'s) lies wholly above 0.05 nats |
| A-3 | ≥ 1500 humans are good enough teachers that a small dose does not cost home72 | X60's Δ_P 97.5 % UB < −2 pp |
| A-4 | The spectator stream is representative of finished ladder games | the coverage read (Q-H4; needs approval): the 95 % CI of the largest between-tercile difference in P(spectated), by length or by rating, lies wholly above 20 pp |
| A-5 | A BC gradient ≈ 10 % of the policy's moves the trunk without dominating | the pre-launch smoke: the 95 % CI of the BC loss's relative fall (first vs last 10 % of the smoke's steps, by batch) lies wholly below 10 % (VOID before GPU), or A-3's falsifier |
| A-6 | The bot rule plus the Kakuna-family audit catch the bots that matter | §3.4's POWER control fails (Kakuna's own seats do not clear the human reference's p99), or a lever's Kakuna gain is KAKUNA-SPECIFIC (its in-house contrast ≤ 0; X59-S Amendment 2 A2.10) |
| A-7 | §3.1's exclusion removes team memory of the panels | the positive control: H-u's Δ CI excludes 0 in the same direction as H-p's while F + P72's CI contains 0 (descriptive; per-team MDE 7.5 pp) |
| A-8 | 20M steps from a plateau is enough plasticity | X60's H-p half reads flat (UB < +2 pp) |

Every falsifier is an interval statement (REVIEW B m7, REVIEW D m4): a bound within 1e-9 of its bar falsifies nothing
(the undecided class).

---

## 9. FINDINGS (standing rule 7)

1. **No offline evidence exists either way for A-1** (the familiarity proxy is uninformative, REVIEW A M1).
2. **The ≥ 3-fit exclusion costs 83.2 % of the candidate battles with `forum216b`** (≈ 71 % of ≥ 1500 battles
   without it), and what it keeps is forfeit-heavy and less fully seen (§3.1).
3. **No unfamiliar panel is clean for a human lever** (Q-H13): H is replay-derived, P72 is drawn from the fill's own
   tables, F has human cores near most teams. `forum216b` carries the clean claim.
4. **The existing HOM is barred** from levers and routing (REVIEW D M5); its rowset is unregistered.
5. **X59-S's teacher is mostly 1300–1499 actors** (the ≥ 1300 floor buys volume for a from-scratch student); the
   card ranks that stratum "the human-only comparator only". The realised share is in the rowset manifest.
6. **X59-S's training needs a GPU lease** (≈ 10 lane-h a student on the CPU); no CPU pilot was trained, and the base
   rate vs Kakuna is measured by the registered floor pilot.
7. **X60 is NOT-READY** (§6). Experiment ids X59-S / X60 are named here and in the PREREG only, not in
   `EXPERIMENT_BACKLOG.md`; the orchestrator assigns them.
8. **The Kakuna-family audit runs before X59-S trains** (Amendment 2 A2.2); its result is in the PREREG's Amendment
   2b.
9. **Process:** CPU only; no GPU, launch or download; `models/`, `data/` and the corpus were only read.

---

## Decision record

| date | decision | chosen | rejected / alternatives | evidence |
|---|---|---|---|---|
| 2026-10-25 | **CREATED (DRAFT; explicit-only)** on the owner's grant | the data card; H-BC > H-TEAMS > HB0 > H-KL > warm start; X59 (the HOM vs Kakuna) first; X60 in D1's frame | building H-BC before A-1 is tested; a factorial; pooling the archive | the card; R-b1, R-b2; D1 look 2; X52 |
| 2026-10-25 | **REVISED after REVIEW A / B / C** (REVISED DRAFT, pending re-review) | the owner's two questions as FIRST reads (SmallRL, the trace twin, LargeRL / LargeIL, X59-S); X59 withdrawn as a rule (descriptive Δ_U only); X60's primary on F + P72 at 630 games a panel, H as a positive control, the H-TEAMS − D1 and H-BC − P-BC contrasts, a goal (a) guard, a seed rule, an action for every class; ONE panel-exclusion rule on revealed facts at ≥ 3 fits before the fill; set labels on hidden turns; account-disjoint splits; a Kakuna-family audit; H-TEAMS re-ranked above H-BC, H-INIT added; the consent flag and `judge` dropped | keeping X59's premium rule (it rewarded forgetting); 140 games a cell (NULL unreachable); the ≥ 4-fit exclusion with a dose-response read (unpowered at ≈ 10 games a team) | `RESPONSE.md`; `plan_power.py` |
| 2026-10-25 | **REVISED after REVIEW D** (READY-WITH-FIXES; X60 NOT-READY) | X59-S rebuilt and pre-registered: THREE arms (S-S mandatory; Δ_data = S-H − S-S answers the owner, Δ_labels = S-H − S-P routes H-BC), FIVE seeds each with a seed term in the interval, sized at p = 0.5 with a floor pilot, S-P's labels restricted to the log-consistent set, ≤ 10-turn games filtered; H-INIT dropped and the existing HOM barred; X60 marked NOT-READY with the four fixes it needs and an in-house U guard; the kept population's profile on page one; every falsifier an interval | one seed per arm (a 2 pp seed SD made a false HUMAN-LABELS-BETTER 0.20 likely); S-S optional (the owner's question unread); re-fitting the HOM now (its compliant ≥ 1500 volume is small, and X59-S answers the question first); fixing X60 now (not on the path) | `RESPONSE.md` §D; `x59s_matched_students_2026-10-25/power.py`; `hr_spec_v1.r1`'s scan |
