# S′-LIVE — RECORD (2026-11-05): GO-LEAKY

**Verdict: GO-LEAKY.** P\* + S′ (`r0s4_rolls@10`, deciding every eligible decision from turn 10) beats P\* playing
alone against the REAL Kakuna on unfamiliar teams (U). On the same 2,698 deals it wins **+4.60 pp** more of the game,
with a 95 % interval of **[+2.66, +6.48] pp** (paired, pooled U). With every post-treatment unpaired deal imputed at
its worst case for GO, the lower bound is **+2.57 pp**, above 0, so the registered rule reads GO-LEAKY. It is not
near the boundary. The NULL-imputed upper bound is +6.48 pp, so `go_below_margin` is false.

**What it licenses (PREREG §1), and only this:** S′ WITH the builder's true-seed information (§7) beats P\* on U
against Kakuna. It licenses building and registering a **leak-honest replication on fresh deals**, and that work
alone is un-paused. The searched arm's pooled-U score, 0.545 [0.526, 0.564], clears 0.50, so it is quoted at most as
**"GOAL-B-REACHED (leaky)"**.

**What it does NOT license:** anything about deployable search; "goal (b) met"; any training, league or
public-server change; a size claim beyond the interval. The leaks are in the estimate by construction (§7), and
dlive's strata leaned leak-exposed, so the replication is not a formality.

- PREREG [`PREREG.md`](PREREG.md) (registered `9c2f023e4`), spec [`read_spec.json`](read_spec.json) (sha256
  `57f0ee87…fb7a`), pin `0c76b521e`, engine E4 throughout.
- Amendment AM-K6 (`d93a51124`, its correction and FINDING S′LIVE-OP-6 `96f661e90`): wall-clock only, blind.
- State `~/gen3ai_archive/sprime_live_play_2026-11-01/`. Its record (`record.json`, 4.9 MB, with its inputs) is
  there. In this folder: [`record/decision.json`](record/decision.json),
  [`record/record_summary.json`](record/record_summary.json) (the record without its inputs, plus the state files'
  sha256), and [`record/by_hand.py`](record/by_hand.py) / [`by_hand.json`](record/by_hand.json).

## What was measured

Two lone one-game units per deal, on U unit seeds 300–344 (20 deals a seed, 900 a panel, all fresh): P\* raw
(greedy) and P\* + S′, both against Kakuna (checkpoint 34, greedy) on the same seed and teams. D = y(searched) −
y(raw) for a deal (win 1, tie ½, loss 0). G_U = the mean over H, F and P72 of each panel's mean unit statistic (the
mean D of a panel and unit seed's deals), with a within-panel unit bootstrap (B = 2,000, `default_rng(20261101)`).

## The decision (rule `searched_play`)

| panel | G paired | 95 % | pairs | post-treatment drops |
|---|---|---|---|---|
| heldout72_hpfix (H) | +4.53 pp | [+0.79, +7.98] | 899 | 1 (raw side) |
| forum72 (F) | +5.15 pp | [+1.71, +8.67] | 899 | 1 (searched side) |
| p72 | +4.11 pp | [+1.33, +7.00] | 900 | 0 |
| **pooled U** | **+4.60 pp** | **[+2.66, +6.48]** | 2,698 | 2 |

| interval | G_U | 95 % | decides |
|---|---|---|---|
| paired only (descriptive) | +4.60 pp | [+2.66, +6.48] | — |
| GO-imputed | +4.54 pp | [**+2.57**, +6.44] | GO-LEAKY iff LB > 0: **yes** |
| NULL-imputed | +4.61 pp | [+2.68, **+6.48**] | NULL iff UB < +3 pp: no |

Exact fractions: G_U paired 1573/34200, GO-imputed 49/1080, NULL-imputed 83/1800.
[`record/by_hand.py`](record/by_hand.py) recomputes every panel fraction, all three G_U, all three intervals and the
class with its own loop from the record's inputs: all equal to the rule's.

## Reported, never decided (PREREG §10)

- **Each arm's pooled-U score** over the paired games: searched **0.545 [0.526, 0.564]**, raw 0.498 [0.479, 0.517];
  searched − raw +4.71 pp [+2.05, +7.36] (stratified Newcombe). The searched LB against 0.50 is goal (b)'s meter,
  quoted only with the leak caveat above.
- **Discordance:** 919 of 2,698 pairs differ (0.341).
- **The search,** over 2,698 searched games: 97,157 searched decisions, 35,532 departures from P\*'s top (0.366), and
  5,995 exclusions (no consistent world found, so P\* played). Search CPU 4.6 CPU-min a game.

  | panel | departure rate | exclusions a game | search CPU-s a game |
  |---|---|---|---|
  | H | 0.371 | 0.46 | 232 |
  | F | 0.363 | **5.70** | 351 |
  | p72 | 0.364 | 0.51 | 242 |

  F excludes ≈ 12× as often as H and p72 (FINDING S′LIVE-R1). An exclusion is P\*'s own play, so F's G measures a
  less-searched S′.
- **Game length:** raw mean 46.0 turns (median 37), searched 44.7 (37). Counted games reaching the 250-turn cap: raw
  13, searched 10.
- **The identity guard:** 600 of 6,615 sampled decisions re-scored, 0 mismatches. By panel H 204, F 217, p72 179; by
  turn 10–14 77, 15–19 77, 20–29 186, 30–63 183, 64+ 77; by unit seed 300–314 192, 315–329 201, 330–344 207 (every
  supervisor life is covered).
- **Timer kills:** none. **One-sided deals:** the two drops below.

## Every check

| check | result |
|---|---|
| engine | every counted unit of both arms on E4 |
| bytes | every counted unit and resolved model on P\* `814823c5…91ed` |
| regime | searched `greedy+search=r0s4_rolls@10`, raw greedy with no search stamp |
| G-PREFIX | 2,698 deals checked, 0 mismatches, so 0 pre-treatment drops; 2 not checkable (each has one arm exhausted) |
| identity guard (`live_identity rescore`) | PASS, 600 / 600, 0 mismatches, the registered order |
| drops | post-treatment H 1, F 1, p72 0 of 900 each (0.11 %; cap 0.5 % = 4); pre-treatment 0 (cap 2 %) |
| unfinished deals | 0 |
| re-derived anchor guards | count mode, 5,400 units, 0 hits |
| completeness | COMPLETE; VOID gate: no reason; `check-record` OK |
| by hand | every estimate, bound and the class equal the rule's |

**The two drops** (FINDING S′LIVE-OP-7), each imputed at its worst case per class:
- `pstar__heldout72_hpfix__337__g03` (raw side missing): both attempts `peer_long_tail_fault`.
- `pstar_s__forum72__337__g19` (searched side missing): both attempts `peer_long_tail_fault`.

That class is Kakuna's own "Force resetting due to long-tail error" (SOP H5 / H21), printed when it races our
turn-250 forfeit. It is a peer fault after turn 10, so §5's post-treatment kind.

**Units counted on a third attempt** (a fresh life's retries): `pstar_s__forum72__322__g16` (F), on life 4 after two
`peer_long_tail_fault` failures in life 3. Counted on a second attempt, same class: `pstar_s__forum72__317__g17`,
`pstar_s__heldout72_hpfix__337__g03`. No other unit failed an attempt.

## Against the GUESSES (§11)

| guess | measured |
|---|---|
| G_U ≈ +4 pp (−2, +12) | +4.60 pp |
| P(GO-LEAKY) ≈ 0.5, P(NULL) ≈ 0.2, P(STOP) ≈ 0.3 | GO-LEAKY |
| no panel ordering | F +5.2 > H +4.5 > p72 +4.1, intervals overlapping |
| d ≈ 0.42 | 0.341 |
| G-PREFIX 0 mismatches | 0 |
| identity guard PASS, 600 | PASS, 600 |
| drops < 0.5 % a panel | 0.11 % at most |
| mean searched game ≈ 9 CPU-min (7–13) | 6.2 CPU-min (median 3.4, p99 50, max 198) |

## Cost and wall

- **CPU:** 292 CPU-h for the searched units, 22 for the raw units, 1.1 for the registered re-score: ≈ 315 CPU-h,
  every attempt included. The ESTIMATE was 370–450 (§8), raised to ≈ 600–700 by the smoke (S′L-4).
- **Wall:** 102 h from the cost checkpoint's start (11-01 09:34 PST) to the last unit (11-05 15:37), over four
  supervisor lives, all on this box. Life 4 ran at `--concurrency 6` (AM-K6), at ≈ 25–50 deals an hour against life
  3's 3–30.

## FINDINGs of the operation

- **S′LIVE-OP-1 (the orchestrator's).** A detached `setsid nohup … mem_cap.sh …` launch once died silently.
- **S′LIVE-OP-2 (the reboot).** The box rebooted on 11-03 at ≈ 07:11–07:21 PST and killed resume life 1 (PID
  1902640). Its two units in flight were cut off with no `end` event, so they are not failed attempts; `resume` set
  each cut-off cell aside as `.partial.<epoch>` and replayed it.
- **S′LIVE-OP-3 (the session-restart kill).** Life 2 (PID 42546, started 08:13:17) died ≈ 15 s later, when the
  orchestrator's Claude session restarted during the owner's nvidia-module work. The supervisor lived inside that
  session's process tree; its units then died with it (each wrapper's parent-death signal). Lives 3 and 4 ran as
  transient `systemd-run --user` services (`<state>/ops/life<n>.sh`, the engine checked against E4 first), outside any
  session, and both survived.
- **S′LIVE-OP-4 (a remaining hazard).** `Linger=no` for the box's user: if every login session closes, systemd
  stops `user@1000` and every service in it. Not changed (a box setting).
- **S′LIVE-OP-5 (the pace under a live run).** With a training run live, life 3 fell to 3–5 deals an hour at nice 19
  on the agent cores; it recovered to 12–30 once the run ended.
- **S′LIVE-OP-6 / OP-7 (the turn-cap peer fault).** Above. The orchestrator's standing decision (2026-11-05): an
  exhaustion of this class alone does not hold the lane; another class, a panel's drops reaching 3, or more than 2
  new exhaustions in 6 h would have. None did.
- **S′LIVE-R1 (F's exclusions).** Reported above.
