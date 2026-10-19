# TSR — the team-strength rating meter's first read (2026-10-18 / 19)

**What this is.** A per-TEAM strength rating with a FIXED pilot on both sides, so that sets of teams can be compared
without the pilot's familiarity standing in for strength. It answers the owner's 2026-10-18 question ("can we get a
team strength? … Kakuna's unseen teams might not be competitive"). The tool is `python -m main.team_strength`
(`src/main/team_strength/`). **The preregistration ([`PREREG.md`](PREREG.md), commit `119313b9`) was committed before
any game**, with every input frozen in [`tsr_registration.json`](tsr_registration.json) (digest `336b8221…`).

Abbreviations: pp = percentage points; CI = 95 % interval; Elo = 400·log10 odds (0 = the reference panel's mean;
−70 Elo ≈ a 40 % expected score against a 0-Elo team; 35 Elo ≈ 5 pp); BT = Bradley-Terry; P = home72 (our sample
teams), H = heldout72_hpfix (72 replay-reconstructed ladder teams, Hidden Power repaired).

## Answer

- **The held-out (H) teams are WEAKER than human-built teams, by about 70 Elo (≈ 10 pp per game at even), and both
  pilots agree on that.** X26 piloting both sides: H − forum72 = **−71 Elo [−147, −2]**; Kakuna piloting both sides:
  **−69 [−188, +43]**. They are also ≈ 60–100 Elo below home72 / away. By the registered 35-Elo rule the verdict is
  **UNDECIDED** for both pilots: X26 DETECTS H as weaker than forum72, but only just (the bound is −1.7 Elo), and
  neither pilot can say whether the gap is above or below 35 Elo.
- **The procedural P72 teams are the weakest set, by far:** P72 − forum72 = **−146 Elo [−236, −84], MATERIALLY
  WEAKER** (X26); −128 [−249, −13] (Kakuna). H sits between: H − P72 = +75 [+6, +164] (X26).
- **Human-built forum teams are about as strong as our sample teams:** forum72 − home72 = −27 [−98, +38] (X26),
  +7 [−98, +110] (Kakuna).
- **The two pilots agree on the SETS, and only weakly on single teams.** Team-level Spearman ρ = **0.33** over 98
  teams. The registered interval [−0.06, +0.44] gives **NOT SHOWN**. That interval is biased toward 0 (FINDING 3); a
  teams-only bootstrap gives [+0.13, +0.50]. Each pilot's per-team ratings are noisy (reliability 0.41 / 0.44), and
  corrected for that noise ρ ≈ 0.77.
- **X37a's piloting gap SURVIVES matching on team strength.** X26 loses more when it pilots an unfamiliar team
  against Kakuna, and team strength explains only a small part of that.
  - Primary read (strength from Kakuna's ratings): **PILOT_m = +0.28 [+0.12, +0.44], DETECTED**. The raw gap on the
    same 266 games is +0.31, so team strength accounts for about 3 pp of it.
  - The per-team match under-corrects, because the ratings are noisy and X26's win rate rises only slowly with them.
    The set-level correction, which assumes the full Kakuna-vs-Kakuna slope, gives +0.107 on all 800 games (X37a's
    PILOT_c was +0.123).
  - Every variant is ≥ +0.10 and detected.
- **For a widened training pool:** draw from human-built forum teams, tier A first. Strength-screen anything else,
  and do NOT screen by X26's own rating alone (§ Recommendation).
- PROVISIONAL throughout: X26 is one checkpoint (161M) of a live, unplateaued run, one seed.

## Per-set strength

Mean team rating in Elo (reference panel = 0) with the two-stage CI (teams resampled, then pairs / matchups; B = 2,000).
Pilot A rates every team (20 games each); pilot B rates a 24-team subset of each set (about 12 games each, ~6
distinct matchups).

| set | teams | **X26 pilots both sides** | sd | teams | **Kakuna pilots both sides** | sd |
|---|---:|---|---:|---:|---|---:|
| away (Metamon `competitive`) | 19 | +1 [−81, +83] | 118 | 19 | +19 [−37, +80] | 66 |
| home72 (our sample) | 71 | 0 [−50, +50] | 131 | 24 | −17 [−84, +52] | 96 |
| forum72 (human-built, unseen) | 72 | −27 [−81, +18] | 120 | 24 | −10 [−94, +70] | 143 |
| **heldout72_hpfix (replay-reconstructed)** | 72 | **−98 [−165, −51]** | 162 | 24 | **−79 [−173, +5]** | 154 |
| p72 (procedural, Smogon usage) | 72 | −173 [−245, −137] | 145 | 24 | −138 [−227, −60] | 131 |

**D1 classes** (registered: MATERIALLY WEAKER iff upper < −35 Elo; COMPETITIVE iff lower > −35; else UNDECIDED):

| difference | X26 | class | Kakuna | class |
|---|---|---|---|---|
| **H − forum72 (PRIMARY: X26)** | **−71 [−147, −2]** | **UNDECIDED, detected weaker** | −69 [−188, +43] | UNDECIDED |
| H − home72 | −98 [−184, −33] | UNDECIDED, detected weaker | −62 [−178, +44] | UNDECIDED |
| H − away | −99 [−208, −16] | UNDECIDED, detected weaker | −97 [−209, −1] | UNDECIDED, detected weaker |
| H − p72 | +75 [+6, +164] | COMPETITIVE | +59 [−60, +177] | UNDECIDED |
| p72 − forum72 | −146 [−236, −84] | **MATERIALLY WEAKER** | −128 [−249, −13] | UNDECIDED, detected weaker |
| forum72 − home72 | −27 [−98, +38] | UNDECIDED | +7 [−98, +110] | UNDECIDED |
| away − home72 | +1 [−94, +95] | UNDECIDED | +35 [−49, +122] | UNDECIDED |

Under X26, every comparison against home72 or away carries X26's familiarity with those sets (it trained on them), so
those rows are descriptive. **The strength term agrees with X37a.** Kakuna's home72 − H on the subsets is +62 Elo, and
X37a's direct cell K(P, H) = 0.573 is +51 Elo.

## Cross-pilot agreement (D2)

| quantity | value |
|---|---|
| teams rated by both (reference panel excluded) | 98 |
| Spearman ρ | **0.33** |
| registered CI (replicate ratings, teams resampled) | [−0.06, +0.44] ⇒ **NOT SHOWN** |
| teams-only CI on the point ratings (descriptive) | [+0.13, +0.50] |
| reliability X26 / Kakuna | 0.41 / 0.44 |
| ρ corrected for both reliabilities | 0.77 |
| Pearson on Elo | 0.25 |

The two pilots rank the SETS in the same order (away ≈ home72 ≈ forum72 > H > P72); at single-team level each rating
is too noisy to say much.

## X37a's piloting gap at matched strength (D3, no new games)

Games where X26 PILOTS a team against Kakuna on home72. Familiar = the banked S(P, P) rows (400), unfamiliar = X37a's
reverse cell (400). The logistic is P(win) = sigmoid(a + g·U + b·z), where U marks an unfamiliar team and z is the
piloted team's strength. PILOT_m is the average of sigmoid(a + b z) − sigmoid(a + g + b z).

| variant | games | PILOT_m | 95 % | class | raw gap, same games | explained by strength |
|---|---:|---:|---|---|---:|---:|
| **PRIMARY: z = Kakuna's rating** (24 + 23 subset teams) | 266 | **+0.282** | [+0.117, +0.443] | **DETECTED** | +0.310 | +0.028 |
| SECONDARY: z = X26's rating (descriptive, circular) | 800 | +0.172 | [+0.075, +0.276] | DETECTED | +0.195 | +0.023 |
| z regression-calibrated, Kakuna (descriptive) | 266 | +0.281 | [+0.087, +0.447] | DETECTED | | |
| z regression-calibrated, X26 (descriptive) | 800 | +0.130 | [+0.020, +0.274] | DETECTED | | |
| set-level, slope 1 (descriptive): raw − (Kakuna home72 − H as a probability, +0.088) | 800 | +0.107 | | | +0.195 | +0.088 |
| X37a's PILOT_c, for reference (K(P, H) cell) | | +0.123 | [+0.039, +0.206] | | | +0.073 |

**Reading.**
- The deficit is about piloting an unfamiliar team, not the team being weak. Within each group, X26's win rate rises
  only slowly with a team's Kakuna-rated strength: b = 0.34 per logit, against 1 by construction in Kakuna-vs-Kakuna
  play. So matching team by team removes ~3 pp. The slope-1 set-level correction removes ~9 pp.
- The truth lies between the two:
  - the slope is attenuated by rating noise (reliability 0.44), so the per-team match under-corrects;
  - X26 may also genuinely draw less from a team's strength than Kakuna does, so the slope-1 correction may
    over-correct.
- Either way the gap is ≥ +0.10 and detected.
- The SECONDARY row matched on X26's own self-play ratings. Those carry X26's familiarity, so it was expected to
  absorb the gap. It did not (+0.17). X26 rates H only ~36 Elo (≈ 5 pp) further below home72 than Kakuna does, which
  is a much smaller self-play deficit than the ~11–28 pp it shows against Kakuna. The same-team comparison (descriptive
  (e)) is −18 Elo [−104, +72], not resolved.
- **If confirmed, a self-play meter would UNDER-measure the piloting deficit** (GEN-M, §5.4 of the diversity doc).
  That is a hypothesis, not a result.
- A-K's analogue (Kakuna's own familiarity with P vs H is equal) is UNVERIFIED.

## Descriptive (D4)

- **H by its frozen `ladder_weights_v1` stratum:**
  - X26: CORE −17 (16 teams), NEAR −89 (44), **FAR −236 (12)**;
  - Kakuna: CORE +67 (4), NEAR −122 (16), FAR −52 (4).
  - X26 rates the off-meta FAR teams far below where Kakuna does. That is X26's off-meta piloting deficit
    (H-PILOT-OFFMETA, X37a) showing up inside its own ratings.
- **forum72 by its manifest:**
  - tier A (67 teams) −17 Elo; tier B (5 teams, all judged teambuilding-competition entries) **−166** (X26). Kakuna:
    tier A −6, its 1 tier-B team −161.
  - By category (X26): resource-listed analyses +11, curated dumps −17, archive history −30, competition winners −166.
- **Tails (X26):** below −170 Elo are 31 / 72 P72 teams, 18 / 72 H, 8 / 72 forum72, 4 / 71 home72, 2 / 19 away.
  The bottom quintile is 29 P72, 17 H, 8 forum72, 4 home72, 2 away.
- **X26 − Kakuna by set** (full sets vs subsets): home72 +17, away −17, forum72 −17, H −19, P72 −36 Elo. On the SAME
  subset teams: home72 −27, away −17, forum72 −26, H −9, P72 −38. Too noisy to read a per-set familiarity premium.

## Recommendation: which teams a widened training pool should draw from

1. **Human-built forum teams first** (`smogon_forum_teams_v2_2026-10-18/`, tier A, then the A/B independent core).
   Forum teams rate as strong as our sample teams, ~70 Elo above replay reconstructions and ~146 Elo above procedural
   teams.
2. **Strength-screen tier B and C before use.** The only tier-B forum72 teams (5 judged competition entries) rated
   lowest of any group (−166 Elo). The meter screens 150 teams in about 30 lane-minutes (X26 vs X26, 10 pairs per
   team, 0.8 games/s).
3. **Do NOT screen by X26's rating alone.** X26 rates off-meta teams low partly because it pilots them badly
   (FAR −236 under X26 vs −52 under Kakuna), and those are the teams X37a says the widening needs. Use a lenient
   threshold (e.g. drop only teams below the P72 mean, ≈ −170 Elo), keep FAR cores deliberately, and confirm the
   drops with the Kakuna pilot when it matters.
4. **Procedural P72-style teams are not a primary source** (MATERIALLY WEAKER than forum teams).
   **Replay reconstructions (W_hl-like)** are ~70 Elo weaker than forum teams. They are usable for diversity, but
   after the forum corpus.

## Integrity

- **Pilot A:** 20 / 20 units, 3,000 mirrored pairs, 6,000 games. Every pair played its planned teams in its planned
  seats (the lane's plan check + `score_games`).
  - Draws: 32. The 250-turn stall cap was hit 14 times; median 33 turns.
  - 13 games (7 pairs) had a decision inside the CPU near-tie margin (2e-5); refitting without them moves no class
    (supplement (f)). 885 games had one inside the GPU-bar margin (2e-3), which matters only for a cross-device
    replay.
  - Model re-hashed before every unit; T28 clear; 0.83 GB peak.
- **Pilot B:** 75 / 75 units counted on the first try (`unit_counts` against the registered per-side digests, every
  capture matched by STRICT identity on both sides, every capture winner = `games.jsonl`). 1,500 games; **6
  self-games dropped** (a reference team drawn on both sides), 1,494 used.
- **Regime stamp:**
  - Pilot A: `main.h2h`'s engine at `119313b9` (CPU eager, 48 envs, 3 + 3 threads, release, `ffi`), regime id
    `503e2d21…`, team set `tsr:b291d9b5…`, torch `2.8.0+cu126`.
  - Pilot B: anchors pair cell, Kakuna ckpt 34 (Metamon `0a00a759`), greedy / greedy, CPU, the Rust front end on 9563.
    The supervisor ran from `f40e6bf3` (FINDING 1); `src/main/anchors` is byte-identical to `cb0f1769`.
- **Cost:** pilot A 1 h 59 m (0.61–1.48 games/s); pilot B 3 h 37 m wall, including the 6-minute stop of FINDING 1
  (~2.2–5 min per unit). One lane on rule 5b's
  cores, beside the live X26 trainer and the maturity battery. 0 GPU.

## FINDINGS

1. **Pilot B's lane stopped after its first unit.** The free-port probe bound without `SO_REUSEADDR`, so the previous
   unit's TIME_WAIT sockets read as a busy port. It was fixed in `f40e6bf3` (and tested), the same pin dir was moved to
   `f40e6bf3` (only `src/main/team_strength` differs from `119313b9`), and the lane resumed at unit 2. No game was
   lost or replayed.
2. **The anchors' two role halves replay ONE team sequence.** With one `--team-seed`, game 10 + i plays game i's two
   teams with the seats swapped, on another battle seed. Each 20-game unit is therefore 10 matchups × 2 seats, and
   pilot B saw ~6 distinct matchups per team, not ~12.
   - The read clusters pilot B by MATCHUP. That deviates from PREREG §3, which treated each game as its own cluster.
   - The PREREG-as-written clustering changes no class: H − forum72 −69 [−191, +41]; agreement CI [−0.06, +0.42]
     (`results.json` `kakuna_prereg_game_clusters`).
   - This also explains X37a's `kcal` showing the same team in both halves.
3. **The registered D2 interval is biased toward 0.** It recomputes ρ on REPLICATE ratings, which carry the game noise
   a second time, so the point ρ (0.33) sits near the top of its own interval. A teams-only bootstrap on the point
   ratings gives [+0.13, +0.50]. The registered verdict stands (NOT SHOWN); a future registration should use the
   teams-only interval, or a disattenuated ρ with its own bootstrap.
4. **The per-team match under-corrects.** The matched-strength estimator uses a noisy covariate (reliability ≈ 0.4),
   and that biases PILOT_m toward the raw gap. The calibrated and set-level variants are reported beside it. A
   preregistration of D3 should have named the set-level correction as a co-read.
5. **A bootstrap replicate failed to converge in a refit** (the descriptive rule-8 run): undamped Newton from a warm
   start cycled. `fit.fit` is now a damped Newton, with a test that fails on revert. Re-running the registered read
   with it changes every set mean and interval by < 1e-10 Elo.
6. **X26's detection of H as weaker than forum72 is marginal.** The upper bound is −1.7 Elo, and −0.005 after the 7
   near-tie pairs are dropped. "Detected weaker" should travel with that caveat.
7. **Early glance:** the first pilot-B unit's log tail (read to confirm the lane worked) printed that unit's win rate
   (0.450, side A on the away subset). No decision was taken on it.
8. **Same-pilot mirror pairs are not two independent games** (PREREG hazard 2). Pilot A's 20 games per team are
   closer to ~10 effective outcomes, hence a reliability of 0.41.
9. No other hazard hit:
   - T28 clear before every unit; servers only on 9563 (ours, stopped by the tool);
   - nothing under `models/` or `data/`; no GPU (`CUDA_VISIBLE_DEVICES=""`);
   - Metamon sets installed as new `gen3ai_tsr_*` dirs only (never overwriting);
   - games started only after X37a stage 2's done marker.

## Files

- `PREREG.md`, `tsr_spec.json`, `tsr_registration.json`, `teams_away/`: the registration.
- `results.json`: both pilots' per-set means, comparisons, per-team tables, agreement, D3, D4, the PREREG-as-written
  pilot-B clustering. `teams.tsv` is the per-team table. `supplement.json` holds the descriptive supplements (a)–(f).
- `rows_x26.jsonl` (6,000 games, from the rated team's side, with its seat) and `rows_kakuna.jsonl` (1,494 games,
  matchup cluster).
- `scripts/analyze.py`: the registered read. `scripts/supplement.py`: the descriptive supplements.
  `scripts/export_rows.py`: the rows.
- **Not committed:** the unit files, captures and logs, in `/home/goodlad/dev/gen3ai-reads/tsr_2026-10-18/`.
