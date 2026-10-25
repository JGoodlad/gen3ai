# Goal (b)'s panels: are they unfamiliar to Kakuna too? (2026-10-26)

**Question.** Goal (b) asks us to beat Metamon's Kakuna on teams that are unfamiliar to US: the panels H
(`heldout72_hpfix`), F (`forum72`) and P72. The survey's FINDING F3 (`literature/sota_survey_2026-10-26.md`) notes
that H was drawn from Metamon's own `hl_05_26` team set. If Kakuna trained on H's teams, it knows them and we do not,
and our 0.487 on goal (b) would understate where we stand. This read checks each panel's provenance and what Kakuna
can have seen, then recomputes goal (b) without any doubtful panel. It played no game, used no GPU, and was read-only
on Metamon's checkout and cache, the archives and `models/`.

**Answer.**
- **No panel is shown to be familiar to Kakuna in a way that could move goal (b).**
  - At most, Kakuna saw an H team as one human replay (about 2 trajectories). That holds only for H teams whose
    replay is old enough to be in Kakuna's training data, and the date window that would show how many are is not
    in the checkout.
  - Our familiarity premium (≈ 10–13 pp) comes from about 9,300 pilots per team, which is 3–4 orders of magnitude
    more.
  - At the population level, H is no more ladder-typical than F. home72, where we win, is the panel closest to
    Kakuna's data.
- **Removing H lowers our score; it does not raise it.** P\* scores **0.481 [0.447, 0.516]** on F + P72, against
  0.487 [0.459, 0.515] on H + F + P72, because H is P\*'s second-best panel. Goal (b) is not met on any subset of
  the panels.
- **No past verdict changes.** X51: (b) not met. X26c: FLAT. D1 look 2: UNDECIDED. Each holds on every subset (§D).
- **Kakuna-side assumption A-K stays UNVERIFIED** at the exact-team level, but it is now bounded by dose (see
  FINDING G1). The dose argument is MY JUDGEMENT; no read measured Kakuna's familiarity effect directly.

Reproduce: `PYTHONPATH=src python designs/research_state/measurements/goal_b_panel_validity_2026-10-26/validity.py
--out …/results.json`. It is CPU-only and deterministic: results sha256 `a163f6e0…b237` at this commit. The X51 and
X26c inputs are banked, and the hl_05_26 copy is `~/gen3ai_archive/metamon_cache_2026-09-24/`.

---

## A. Provenance and what Kakuna can have seen

**Kakuna's training data, read from Metamon's checkout `0a00a759` (git log and README).**

- **Date.** Kakuna is dated "Dec 2025" (`README.md:288`). Its class first appears on 2025-12-17.
- **Human replays.** It trained on `metamon-parsed-replays`, `pac-base` and `pac-exploratory` (`README.md:825`).
  - In Dec 2025 the latest parsed-replays revision was **v4, released 2025-08-13** (`2466f16d`). v5 came on
    2026-01-21 (`c9c2f441`) and v6 on 2026-05-21 (`a790b624`).
  - Superkazam, Kakuna's parent, used "Alakazam's (11M self-play + 4M human replay) dataset". Kakuna "reduced
    [the] sampling weight of human replays".
  - **ESTIMATE:** Kakuna's human battles end around Aug 2025.
- **Self-play team sets.** These are never named; the README says only "various team sets".
  - The sets that existed then were `competitive`, `paper_variety`, `paper_replays`, `modern_replays` and
    `modern_replays_v2` (teams revision v4, 2025-09-16; "verified against the PokéAgent Challenge server").
  - `gl_05_26` and `hl_05_26` came in teams v5 on 2026-05-21, after Kakuna.
- **The hl_05_26 window.** `hl_05_26` is "high ladder May 2026: 1400+ ∪ smogtours" (`metamon/env/wrappers.py:46`).
  `gl_05_26` is "last-year revealed teams, NaiveUsagePredictor fill".
  - The window's `min_date` lives in `team_sets_gl_hl_05_26.yaml`, which the checkout does not ship.
  - The team files carry no replay ids or dates: `index.csv` holds only file names.
  - **UNVERIFIED:** how many of H's 72 source replays come before Kakuna's data cut. If the window is May 2025 to
    May 2026 and replays are spread evenly in time, about a quarter of H's sources (May to Aug 2025) could sit in
    parsed-replays v4 (GUESS).

| panel | where its teams come from, and when | what Kakuna can have seen | verdict for Kakuna |
|---|---|---|---|
| **home72** | our 72 curated sample teams (`data/teams/sample`); all in our pool | **8 / 72 are exact copies of `competitive`**, the set Metamon says it "overfit to" and measures Kakuna's GXE on. 34 / 72 have a ≥ 4-set near copy in hl_05_26 (HP-agnostic), and 2 are exact | **FAMILIAR**, the most of any panel. It is goal (a)'s panel, where we win (0.598) |
| **H** (`heldout72_hpfix`) | 72 `hl_05_26/gen3ou` files drawn with seed 20261018 on 2026-10-18. Each is one ladder or smogtours replay of ≥ 1400 within the May '26 window, unrevealed fields filled from usage, and Hidden Power repaired by us | Its exact source battle, if that battle predates Kakuna's data cut: at most ≈ 2 trajectories per team, and the filled fields are not what the human played. Whether `modern_replays_v2` holds near copies is unknown (not downloaded). As a population, an H team has a ≥ 4-set near copy elsewhere in hl in 4.2 % of cases (3 / 72), against 8.9 % for a random hl team (2,000-team sample) | **UNKNOWN at the exact-team level; NOT SHOWN FAMILIAR.** Its exposure is bounded at ≈ 1–2 games a team |
| **F** (`forum72`) | 72 Smogon-forum pastes (v1 independent core), drawn with seed 3720 on 2026-10-18. Posts date from 2014 to 2025 (53 from 2021, 5 from 2025) | 0 exact in `competitive` or hl. 10 / 72 have a ≥ 4-set near copy in hl (HP-agnostic) and 36 / 72 a ≥ 3-set one, so humans ladder versions of forum teams | **UNFAMILIAR as teams.** Its population is ladder-typical, about like H |
| **P72** | 72 procedural Smogon-usage teams (`ou_random_teams.js --seed 3721`), generated 2026-10-18 after Kakuna | 0 at ≥ 4 in hl or `competitive`; 9 / 72 at ≥ 3. **UNKNOWN:** how close it is in style to `paper_variety` (also usage-sampled and procedural, not downloaded) | **UNFAMILIAR** as teams; its style is unknown |
| **forum216b** (SEALED; manifest metadata only) | 216 v2 forum pastes (tier AB), seed 5400, carved 2026-10-20. **86 posted in 2026**, 96 on or after 2025-05 | From the v2 manifest's precomputed overlap: 0 at ≥ 4 with `competitive`; **9 / 216 at ≥ 4 with hl_05_26** (4.2 %, the same as H) | **UNFAMILIAR as teams.** The 86 posted in 2026 postdate Kakuna |

## B. The overlap counts (`results.json` → `A_overlap`, `B_forum_metadata`)

**Method.** The matcher is `gen3_team_overlap_v1`: a mon set is the species plus its sorted moves, with Hidden Power
typed.
- "≥ k" means the panel team shares ≥ k mon sets with ONE reference team.
- **"≥ 3" is review A's "3-fit" radius** (`human_replay_quality_2026-10-25/REVIEW_A_data.md` B2), the radius at
  which a team stops being unfamiliar.
- **HP-agnostic** drops Hidden Power's type. hl_05_26 types about 96 % of its Hidden Powers Dark through a fill
  artifact, so a typed match under-counts.
- An H team's own hl source file is excluded. Check: all 72 heldout72 teams are exact copies of their source.

| panel (72) | vs hl_05_26: exact / ≥ 4 / ≥ 3 (typed) | the same, HP-agnostic | share with any ≥ 4 / ≥ 3 neighbour in hl (HP-agn.) | vs `competitive`: exact / ≥ 4 / ≥ 3 |
|---|---|---|---|---|
| home72 | 0 / 13 / 35 | **2 / 34 / 59** | 47 % / 82 % | **8 / 8 / 11** |
| H (as drawn, self excluded) | 0 / 3 / 41 | 0 / 3 / 41 | 4.2 % / 57 % | 0 / 0 / 0 |
| F | 0 / 0 / 15 | 0 / 10 / 36 | 14 % / 50 % | 0 / 0 / 3 |
| P72 | 0 / 0 / 4 | 0 / 0 / 9 | 0 % / 12.5 % | 0 / 0 / 0 |
| a random hl team (n = 2,000, self excluded) | — | 0 / 177 / 1,090 | 8.9 % / 54.5 % | — |
| forum216b (manifest, typed; n = 216) | 0 / 9 / 47 | — | — | 0 / 0 / 6 |

**What the counts say.**
- **Exact copies of H teams are rare in the ladder population.** Beyond its own source, an H team has a ≥ 4-set near
  copy in 4 % of cases. A 2025-26 ladder team is mostly a one-off once its unrevealed sets are filled.
- **So "Kakuna trained on hl-like data" means familiarity with 3-set cores, not with whole teams.** At the 3-set
  radius, H (57 %), F (50 %) and a random ladder team (54.5 %) look alike. home72 (82 %) looks MORE ladder-like than
  any unfamiliar panel. Our own human-replay card agrees: ≥ 1500 human sides fitting ≥ 4 of a panel team's sets
  number a median 218 per team for home72, 105 for H, 74 for F and 6 for P72 (`human_replay_quality_2026-10-25` §D).

## C. Does Kakuna play ladder-typical teams better? (`C_gradient`, descriptive)

**What it tests.** If Kakuna's ladder data gave it a team-level edge, our score should be lower in games where
KAKUNA's team has an hl near copy (≥ 3, HP-agnostic, own source excluded) than in games where it has none.

| banked cell | Kakuna's team has an hl ≥ 3 neighbour | has none | difference, Newcombe 95 % |
|---|---|---|---|
| P\* on H (X51) | 119 / 238 (0.500) | 80 / 162 (0.494) | +0.6 pp [−9.3, +10.5] |
| P\* on F | 102 / 188 (0.543) | 110 / 212 (0.519) | +2.4 pp [−7.4, +12.0] |
| P\* on P72 | 21 / 50 | 152 / 350 | −1.4 pp [−15.1, +13.3] |
| X26 @161M on F / P72 | 85 / 188, 20 / 50 | 103 / 212, 161 / 350 | −3.4 [−13.0, +6.4], −6.0 [−19.5, +8.7] |

**What it shows.** No gradient is detected, and none is excluded: the interval is ±10 pp. The X26 @161M H rows have
no `their_team` field (400 unmapped). Across panels the score ordering also runs the "wrong" way for the bias
hypothesis:
- home72, the panel closest to Kakuna's data, is our best at 0.598;
- P72, the furthest, is our worst at 0.433.

That ordering is confounded with OUR familiarity (home72) and with P72's procedural play, so it is not evidence
either way.

## D. Goal (b) recomputed from the banked rows (`D_goal_b_recomputed`; `main.x35.d1` statistics)

Each score is the equal-weight panel mean with a Kish-effective Wilson interval. Each Δ is Newcombe
(`newcombe_strat`); X26c also gets the paired-unit interval (seeds 0–19); D1 is `d1_primary` (look 2, AM-5 widened).

| model / read | U = H + F + P72 (registered) | **F + P72 (H dropped)** | F alone | H alone |
|---|---|---|---|---|
| P\* (X51) | 0.487 [0.459, 0.515] | **0.481 [0.447, 0.516]** | 0.530 [0.481, 0.578] | 0.498 [0.449, 0.546] |
| X26 @161M (X51 twin) | 0.457 [0.429, 0.485] | 0.461 [0.427, 0.496] | 0.470 | 0.448 |
| P\* − 161M | +3.0 [−1.0, +7.0] | +2.0 [−2.9, +6.9] | +6.0 [−0.9, +12.8] | +5.0 [−1.9, +11.8] |
| X26c @480M | 0.4875 [0.459, 0.516] | 0.504 [0.469, 0.538] | 0.543 [0.494, 0.591] | 0.455 [0.407, 0.504] |
| X26c − P\*, Newcombe / paired units | +0.1 [−3.9, +4.1] / [−3.6, +3.8] | +2.3 [−2.6, +7.1] / [−2.6, +7.1] | +1.3 [−5.6, +8.1] | −4.3 [−11.1, +2.7] |
| D1 look 2: D1 score | 0.492 [0.471, 0.513] | 0.501 [0.475, 0.528] | 0.529 [0.492, 0.565] | 0.473 |
| D1 look 2: Δ (AM-5), class | +2.10 [−0.92, +5.11] UNDECIDED | +2.07 [−1.63, +5.76] UNDECIDED | +4.0 [−1.2, +9.2] | +2.1 [−3.1, +7.3] |

- **Goal (b): no lower bound clears 0.50 on any subset**, for any model. Without H, P\* stands slightly FURTHER from
  the bar: it needs ≈ +5 pp at this n.
- **The verdicts are unchanged.**
  - X51: not met.
  - X26c: FLAT on every subset; the LB is below 0 in each.
  - D1 look 2: UNDECIDED; F + P72 gives the same class (LB < 0, UB > +3).
- H is the only panel on which X26c read lower than P\* (−4.3 pp, n.d.). Excluding it makes X26c's point estimate
  +2.3 pp, still FLAT.
- P\*'s X26c re-play is game-for-game identical to X51's (deterministic deals: 199 / 212 / 173). It is never pooled
  as a second sample.

## E. Recommendations

1. **Goal (b) keeps H, F and P72 at equal weight.** No panel is shown to be Kakuna-familiar, and dropping H would
   LOWER the score and cost a third of the games.
   - State the definition: "unfamiliar" means unseen by OUR training, at ≤ 3 shared sets.
   - Kakuna's exposure is reported, not required to be zero. The exact-team dose is ≤ ≈ 2 human trajectories per H
     team (≥ 1400 ladder). home72 is Kakuna's most familiar ground.
   - Report U − H (F + P72) beside U as a sensitivity line in every goal-(b) read.
2. **Past verdicts stand:** X51, X26c and D1 look 2 (§D).
3. **The clean panel for the goal read stays `forum216b` (X54).** It shares nothing at ≥ 4 with `competitive`, and
   only 4.2 % of it is ≥ 4 with hl, the same as H. 86 of its teams were posted in 2026, after Kakuna.
   - X54's registration should add two descriptive secondaries, decided before unsealing:
     - forum216b without its 9 hl-≥ 4 teams;
     - its 2026-posted subset, the teams Kakuna cannot have seen as pastes.
   - Neither changes the sealed panel; whether to add them is the orchestrator's call.
4. **To settle A-K at the exact-team level, these downloads would be needed.** None was made; each needs the
   orchestrator's go.
   - `jakegrigsby/metamon-teams`: `modern_replays_v2/gen3ou`, `modern_replays/gen3ou`, `paper_replays/gen3ou` and
     `paper_variety/gen3ou`, to overlap H, F and P72 with Kakuna-era team sets.
   - The metamon-teams README plus `hl_05_26`'s `predictions_meta.json`, or upstream's
     `team_sets_gl_hl_05_26.yaml`, for hl's date window.
   - The file list of `metamon-parsed-replays` v4 gen3ou (replay ids carry dates), to place H's sources.
   - The provenance of `pac-exploratory` gen3ou, if Metamon records which team sets it used.

## FINDINGS

- **G1. The dose bound is a judgement.** "≤ ≈ 2 trajectories per H team" assumes Kakuna saw a source replay at most
  once, from both seats. Our premium needs ≈ 10³–10⁴ pilots (§4 of the beat-Metamon doc, an ESTIMATE). Nobody has
  measured how many games Kakuna needs to gain on a team.
- **G2. hl_05_26's date window is UNVERIFIED** (the yaml is not in the checkout), and so is how many of H's sources
  predate parsed-replays v4. The team files carry no replay id (review A said the same).
- **G3. Kakuna's self-play team sets are still UNVERIFIED** (survey F3). If `modern_replays_v2` (Sep 2025) holds
  replay-predicted twins of H teams, Kakuna's dose rises to O(10–100) games a team (GUESS from 7M pac-exploratory
  battles over five formats and a ≈ 10⁴-team set). That is still about 1 % of our pool dose.
- **G4. P72 against `paper_variety` is unmeasured.** Both are usage-sampled and procedural, so Kakuna may know P72's
  STYLE.
- **G5. The §C gradient is underpowered** (±10 pp). It neither shows nor excludes a Kakuna familiarity effect.
- **G6. forum216b was read only through metadata:** its manifest ids, joined to the forum v2 manifest's precomputed
  overlap and post dates. No team file was opened.
- **G7. Two pre-existing artifacts were found and not used as samples.**
  - The X26 @161M H rows lack `their_team`.
  - P\*'s X26c re-play duplicates X51 exactly, so pooling the two would double-count.
- **G8. hl_05_26's fill makes nonsense spreads** (e.g. a Timid Swampert with Earthquake / Surf / Focus Punch, file
  `team_000000`). H's sets are partly invented. That is a property of the panel, not of Kakuna's exposure.
