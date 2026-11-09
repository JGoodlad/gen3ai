# S′-HONEST — RECORD (2026-11-09): GO-HONEST, GOAL-B-REACHED

**Verdict: GO-HONEST, with the decided flag GOAL-B-REACHED.** P\* + S′h (`r0s4_honest_inc@10`: a search that reads
only what the live player knows, deciding every eligible decision from turn 10) beats P\* playing alone against the
REAL Kakuna (checkpoint 34, greedy) on unfamiliar teams (U = H, F, P72 at equal weight). On the same 5,097 fresh
deals it wins **+2.00 pp** more of the game, 95 % **[+0.34, +3.51] pp** (paired, pooled U). With every
post-treatment unpaired deal imputed at its worst case for GO, the lower bound is **+0.32 pp**, above 0: GO-HONEST,
not near the boundary. The NULL-imputed upper bound is +3.53 pp (NULL needed < +3), so `go_below_margin` is false.

**The claim, in the PREREG's §1 wording (GOAL-B column):** "**Goal (b) met with search:** P\* + S′h beats Kakuna
(checkpoint 34, greedy) on unfamiliar teams (U = H, F, P72)": the searched arm's pooled-U score is **0.5145
[0.5008, 0.5282]** over 5,100 fresh deals (1,700 a panel, unit seeds 700–784), ties counted as non-wins and the one
missing searched game counted a loss. Its LB clears 0.50 by 0.0008, outside the 0.0005 band (rule 8). The Wilson
interval treats the games as independent: it is a claim about these panels' teams, not a population beyond them.
Never quote "goal (b) met" without that regime.

**Clustering addendum (2026-11-09, REPORTED; the verdict above is unchanged):**
[`addendum_clustering/`](addendum_clustering/README.md) recomputes both numbers with the team as the unit
(`UNDERSTANDING.md` §7, rule 10). The gain survives every dependence model (LB +0.31 pp at worst). The score's 0.50
clearance survives none but independence: [0.4942, 0.5353] clustered on our team, [0.4908, 0.5382] two-way. The gain is
detected on P72 alone (H + F +0.66 [−1.39, +2.67]), post hoc.

**What it licenses (PREREG §1), and only this:** search is a measured path to goal (b). The search program for goal
(b) is un-paused, and its next registrations may be written: **cost** (a cheaper or faster S′h, or a budget study) and
**the deployable player's design**. (The PREREG's third item, a confirmation read, is named for a GO-HONEST that
does not reach GOAL-B.)

**What it does NOT license:** any public-server or ladder play (the owner's approval file and the drift scan, root
§ Playing); any training or league change; a claim against other opponents, home72, other budgets, builders or
presets; a size claim beyond the interval; any per-panel claim (panels are REPORTED, never decided; below).

- PREREG [`PREREG.md`](PREREG.md) (registered `e81f80657`), spec [`read_spec.json`](read_spec.json) (sha256
  `9222393a…8afa`), pin `e46e3ddf8`, engine E6 throughout. No amendment.
- State `~/gen3ai_archive/sprime_honest_play_2026-11-06/`; its `record.json` (9.4 MB, with its inputs) is there, and
  `main.reads check-record` reads it OK. In this folder: [`record/decision.json`](record/decision.json),
  [`record/record_summary.json`](record/record_summary.json) (the record without its inputs, plus the state files'
  sha256), [`record/by_hand.py`](record/by_hand.py) / [`by_hand.json`](record/by_hand.json), the builder report
  [`record/inc_report.json`](record/inc_report.json) and [`record/search_cost.json`](record/search_cost.json).

## The decision (rule `searched_play`)

| panel | G paired | 95 % | pairs | post-treatment drops |
|---|---|---|---|---|
| heldout72_hpfix (H) | +1.32 pp | [−1.39, +4.00] | 1,699 | 1 (raw side) |
| forum72 (F) | −0.005 pp | [−2.98, +2.71] | 1,699 | 1 (searched side) |
| p72 | +4.69 pp | [+2.00, +7.37] | 1,699 | 1 (raw side) |
| **pooled U** | **+2.00 pp** | **[+0.34, +3.51]** | 5,097 | 3 |

| interval | G_U | 95 % | decides |
|---|---|---|---|
| paired only (descriptive) | +2.00 pp | [+0.34, +3.51] | — |
| GO-imputed | +1.98 pp | [**+0.32**, +3.47] | GO-HONEST iff LB > 0: **yes** |
| NULL-imputed | +2.04 pp | [+0.38, **+3.53**] | NULL iff UB < +3 pp: no |

Exact fractions: G_U paired 3877/193800, GO-imputed 101/5100, NULL-imputed 26/1275. GOAL-B score 656/1275.
[`record/by_hand.py`](record/by_hand.py) recomputes every panel fraction, all three G_U, all three intervals, the class
and the GOAL-B point with its own loop from the record's inputs: all equal to the rule's.

## Reported, never decided (PREREG §12)

- **The panels.** P72 carries the pooled gain (+4.69 pp, its interval above 0); H's and F's intervals span 0, F's
  point is 0. No panel claim is made; the PREREG predicted no ordering.
- **Each arm's pooled-U score** over the paired games: searched 0.5146 [0.5009, 0.5283], raw **0.4948** [0.4811,
  0.5085]; searched − raw +1.98 pp [+0.04, +3.92] (stratified Newcombe).
- **Discordance, SE, power.** 1,784 of 5,097 pairs differ (d = 0.350). The achieved bootstrap SE of G_U is 0.80 pp
  (the iid SE at this d 0.83 pp; the PREREG planned 0.71 realised / 0.82 conservative), so the read had ≈ 0.82
  power at +2.30 pp (0.79 on the iid SE).
- **The cross-read difference** (S′-LIVE's G_U − this read's, on different deals): +4.60 − +2.00 = **+2.60 pp
  [+0.12, +5.07]** (√(0.974² + 0.804²) = 1.26 pp). **UNPAIRED and SELECTED** (S′-LIVE is here because it cleared its
  bar). It is the leak's share only if the two deal samples are exchangeable and S′h differs from S′ only by the
  leaks; it also differs by its builder's budget, its exclusions (below), its correlated worlds (PREREG §8) and the HP
  fix. A reading, not a measurement: about half of S′-LIVE's point did not survive the honest builder.
- **The search,** over the 5,097 paired searched games: 188,121 searched decisions, 69,988 departures from P\*'s top
  (0.372), 11,062 exclusions; search CPU **249 CPU-s a game** pooled.

  | panel | departure rate | exclusion share (bound) | exclusions a game | search CPU-s a game | whole-unit CPU-min (mean / median / max) |
  |---|---|---|---|---|---|
  | H | 0.373 | 0.0257 (0.05) | 0.93 | 212 | 4.94 / 3.14 / 78.7 |
  | F | 0.371 | **0.1002** (0.13) | 4.32 | 309 | 9.52 / 4.72 / 102.0 |
  | p72 | 0.372 | 0.0333 (0.09) | 1.26 | 226 | 5.47 / 3.60 / 77.6 |
  | U | 0.372 | 0.0555 | 2.17 | 249 | 6.64 / 3.63 / 102.0 |

  The exclusion shares are the rule's (every counted searched unit: 11,071 / 199,368); the other columns are the
  paired games'. `excess_exclusions` is false (FINDING S′HR-1 below).
- **The builder report** (`inputs/inc_report.py` over the 5,099 counted searched units): of 199,368 search-path
  decisions, 92.6 % warm, 4.8 % stuck, 2.6 % cold first decisions, 0.1 % re-cold. Battles with an exclusion 810 of
  5,099 (F 430, H 172, P72 208); with a stuck stretch 775 (S′H-13). Searched decisions holding a world past the
  correlation bound 7.7 % (F 6.5, H 8.3, P72 8.5; ages to 71; S′H-15). Warm chain fates: fit 156,414, refit 25,957,
  skipped 7,483, failed 2,171, deferred 1,313. Cold first decisions 10.2 CPU-s mean, max 399.
- **Game length:** raw mean 46.5 turns (median 37), searched 45.5 (36); counted games reaching the 250-turn cap raw 23,
  searched 20. Ties: raw 6, searched 8 (non-wins in GOAL-B).
- **The identity guard:** 600 of 12,604 sampled decisions re-scored at the pin in REBUILD mode, 0 mismatches,
  17,734 checkpoints rebuilt, no torn tail. By panel H 190, F 208, p72 202; by turn 10–19 172, 20–29 140, 30–63 199,
  64+ 89; by unit seed 700–716 131, 717–733 138, 734–750 107, 751–767 120, 768–784 104.
- **Distillation fields:** 0 lacking, 0 units without a capture: 188,297 searched and 11,071 excluded honest records,
  the ExIt-0H bank's input.

## Every check

| check | result |
|---|---|
| engine | every counted unit of both arms on E6 |
| bytes | every counted unit and resolved model on P\* `814823c5…91ed` |
| regime | searched `greedy+search=r0s4_honest_inc@10`, raw greedy with no search stamp |
| G-PREFIX | 5,097 deals checked, 0 mismatches, so 0 pre-treatment drops; 3 not checkable (one arm exhausted) |
| identity guard | PASS, 600 / 600, 0 mismatches, REBUILD, the registered order, checkout the pin (clean) |
| drops | post-treatment 1 a panel (0.06 %; cap 8); pre-treatment 0 (cap 34) |
| unfinished deals | 0 |
| re-derived anchor guards | count mode, 10,200 units, 0 hits |
| exclusions | every panel under its bound; `excess_exclusions` false |
| completeness | COMPLETE; VOID gate: no reason; `check-record` OK |
| by hand | every estimate, bound, the class and the GOAL-B point equal the rule's |

**The three drops**, each imputed at its worst case per class, all `peer_long_tail_fault` (Kakuna's "Force resetting
due to long-tail error" at our turn-250 forfeit, SOP H5 / H21), each exhausted on its second attempt:
`pstar_s__forum72__722__g14` (searched side, 11-07 15:11), `pstar__heldout72_hpfix__762__g14` (raw, 11-08 12:39),
`pstar__p72__780__g10` (raw, 11-08 22:08). Nine more units failed one attempt of the same class and counted on their
second. No other error class occurred; no HOLD rule of §6 fired (≤ 1 exhaustion in any 6 h, ≤ 1 drop a panel).

## Against the GUESSES (§13)

| guess | measured |
|---|---|
| G_U ≈ +2.5 pp (−0.5, +5.5) | +2.00 pp |
| P(GO-HONEST) ≈ 0.6, P(NULL) ≈ 0.3, P(STOP) ≈ 0.1; P(GOAL-B-REACHED) ≈ 0.35 | GO-HONEST, GOAL-B-REACHED |
| no panel ordering | P72 +4.7 > H +1.3 > F 0.0, intervals overlapping |
| d ≈ 0.35 | 0.350 |
| exclusions H ≈ 0.5 %, F ≈ 6 %, P72 ≈ 4 %; `excess_exclusions` not set | H 2.6 %, F 10.0 %, P72 3.3 %; not set |
| the mean searched game ≈ 6–8 CPU-min of whole-unit CPU | 6.64 (median 3.63, max 102) |
| G-PREFIX 0; identity PASS, 600; drops < 0.5 % a panel; every distillation field present | 0; PASS, 600; 0.06 %; all present |

## Cost and wall

- **CPU:** the searched units' whole-unit CPU ≈ 565 CPU-h (last attempts; search alone 352), the raw units not metered
  here; the registered re-score 25.8 CPU-h (92,867 CPU-s; S′HP-7's estimate 16.2, the checkpoint's projection 38–75),
  the checkpoint's re-score 16.1 CPU-h (143 decisions; estimate ≈ 4). The ESTIMATE was ≈ 597 CPU-h in all (§11).
- **Wall:** 52 h from the first unit (11-06 23:02 PST) to the record (11-09 03:00), against the ≈ 45 h ETA. Life 0
  (the checkpoint, BOX shape at K = 12 beside X60-T look 1's K = 2 lane): 23:02–00:09. Its reading and re-score to
  02:20 (the §14 re-score command hit its 90-min bound and was resumed for its last 3 decisions). Life 1 (BOX, K = 14,
  the whole box): 11-07 02:20 → 11-09 00:44. The registered re-score (`--jobs 14`, peak 14.9 GB of its 24-GB cap):
  00:45–02:59. Both lives were `systemd-run --user` services and neither died.

## FINDINGs

- **S′HR-1 (live exclusions above stage 1b's projection).** U 5.55 % against the projected 3.35 %: H 2.57 % (projected
  0.29 % [0.05, 0.62]), F 10.02 % (5.75 [1.45, 12.86]), P72 3.33 % (4.02). On H the live share is above twice the
  projection and above its 95 % upper end; only the registered 5 % floor keeps H under its bound. Every panel is under
  its bound, so `excess_exclusions` is false by the rule, and the class does not depend on it. S′h searched ≈ 94 % of
  its search-path decisions where the design projected ≈ 97 %; a quarter of F's battles held an exclusion. The live
  distribution reaches the builder's failure modes (S′H-13) more often than stage 1b's 407 battles did. It bears on a
  cost or builder registration (fewer exclusions is more search), not on this verdict.
- **S′HP-7 (the re-score's REBUILD cost), closed.** Option A (keep REBUILD, the orchestrator 2026-11-07) cost 25.8
  CPU-h and 2.2 h of wall for the registered 600: above the registered 16.2, below the checkpoint's 38–75 projection.
- **S′HP-11 (the standing GPU lease blocked the BOX shape), resolved operationally.** The training session released
  its lease at 11-06 23:01 for this read; both lives ran the BOX shape; the lease was re-taken after the lane ended
  (X60Tb s3504 started 11-09 03:04).
- **S′HR-2 (two "searched U scores").** The decided GOAL-B score (0.5145 [0.5008, 0.5282]) is over all 5,100 deals
  with the one missing searched game a loss; `score_u` (0.5146 [0.5009, 0.5283]) is over the 5,097 pairs only. Quote
  GOAL-B's.
