# The deviation-only live read — RECORD (2026-11-01): STOP

**Verdict: STOP.** Where the search S′ departs from P\*'s own choice on unfamiliar teams, playing its pick and then P\*
greedy against the REAL Kakuna wins **+1.20 pp** more of the game than P\* playing its own choice, with a 95 % interval
of **[−1.55, +3.88] pp** (pooled U, n = 1,793 departures in 1,402 games). The lower bound is below 0, so the
registered rule reads STOP. It is not near the boundary.

Recorded as the PREREG says (§4): **"S′ at P\* is not shown to improve on π against Kakuna on U at ≈ 1,800
deviations."** No live S-U read on S′. Search stays paused for goal (b). DOES-NOT-CLOSE-B is NOT claimed, and "S′ is no
better than π" is NOT claimed: this is a power statement (§6).

- PREREG [`PREREG.md`](PREREG.md) (registered `715e0b482`), spec [`read_spec.json`](read_spec.json) (sha256
  `9821b331…6851`), pin `c01ae8ee5`.
- Amendments: A1 (`f0029cdca`, operations); **AM-DL-PEER** (`e40d9663c`, Kakuna's replay nondeterminism, blind); its
  null clarification (`d25b40c55`); rule 6 built (`276c5b7e6`). Each was registered before any branch outcome was read
  (DL-8 is the one exception, disclosed).
- State `~/gen3ai_archive/dlive_read_2026-10-28/`. Its record (`record.json`, 14.5 MB, with `inputs.json`) is there; in
  this folder: [`record/decision.json`](record/decision.json), [`record/record_summary.json`](record/record_summary.json)
  (the record without its inputs, plus the state files' sha256), [`record/peer_drops_summary.json`](record/peer_drops_summary.json),
  and [`record/by_hand.py`](record/by_hand.py) / [`by_hand.json`](record/by_hand.json).

## What was measured

At 6,398 fresh states of raw P\* vs Kakuna games (turns 10–30, two hashed states a game), S′ = `r0s4_rolls@10` was
run offline in the registered order until each panel had its quota of S′ DEPARTURES from P\*'s top action. At each
departure S′'s pick was FORCED in a one-game replay of that deal against Kakuna, with P\* greedy afterwards. D = the
branch's result minus the raw game's (win 1, tie ½, loss 0).

## The decision (rule `deviation_live`)

| panel | mean D | departures counted | games |
|---|---|---|---|
| heldout72_hpfix (H) | +0.0417 | 599 | 473 |
| forum72 (F) | +0.0051 | 594 | 471 |
| p72 | −0.0108 | 600 | 458 |
| **pooled U (equal weight)** | **+0.0120 [−0.0155, +0.0388]** | 1,793 | 1,402 |
| home72 (REPORTED) | +0.0268 [−0.0396, +0.0956] | 298 | 226 |

The interval is the registered game-clustered percentile bootstrap (B = 2,000, `default_rng(20261028)`). Its width,
SE ≈ 0.0139, is what §6 predicted (0.0135). [`record/by_hand.py`](record/by_hand.py) recomputes the panel means, the
pooled mean, both bounds and the class with its own loop: all equal to the rule's.

## Reported, never decided (PREREG §8)

- **U − home72:** −0.0149 [−0.0854, +0.0598].
- **Live sign accuracy**, P(D > 0 | D ≠ 0): U 0.521 [0.472, 0.569] (507 split departures); home72 0.545 [0.435, 0.663]
  (88). The gate measured the same statistic under T at 0.599 (re-gate).
- **T against Kakuna on the same departures (REVIEW_gate M2):**
  - T's mean improvement (T(pick) − T(top)): U **+0.0099 [+0.0014, +0.0193]**; home72 +0.0178 [−0.0015, +0.0365].
  - Live D − T: U +0.0021 [−0.0263, +0.0292]; home72 +0.0091 [−0.0575, +0.0787]. M2 predicted the gap to be more
    negative on U than on home72. The point estimates lean that way (+0.002 against +0.009), but the intervals overlap
    almost entirely, so it is not resolved.
  - **The agreement of T's sign with the live sign**, where both split: **0.560 [0.513, 0.607]** on U (n 458).
    DESCRIPTIVE ONLY (DL-3): T shares S′'s opponent model.
- **Strata, pooled U:**

  | stratum | mean D | n |
  |---|---|---|
  | Speed-pinned | +0.0266 [−0.0354, +0.0845] | 397 |
  | not Speed-pinned | +0.0087 [−0.0241, +0.0392] | 1,396 |
  | leak-exposed | +0.0259 [−0.0128, +0.0644] | 853 |
  | leak-clean | −0.0006 [−0.0379, +0.0361] | 940 |

  No stratum excludes 0.
- **Per panel:**

  | panel | branches planned / counted / dropped | split | nulls counted | branch CPU-s, mean |
  |---|---|---|---|---|
  | H | 600 / 599 / 1 | 159 | 37 / 37 | 21.3 |
  | F | 600 / 594 / 6 | 193 | 41 / 41 | 23.8 |
  | p72 | 600 / 600 / 0 | 155 | 37 / 37 | 23.2 |
  | home72 | 300 / 298 / 2 | 88 | 20 / 20 | 23.6 |

- **The draw and the quotas:** every panel met its quota exactly. States scored / drawn: H 1,455 / 1,800, F 1,624 /
  1,800, p72 1,450 / 1,799, home72 799 / 999. Deviation rates: 0.41, 0.37, 0.41, 0.38 (the PREREG's estimates 0.43,
  0.39, 0.40, 0.38). Exclusions (S′ found no consistent world): H 6, F 90, p72 13, home72 51.

## Every check

| check | result |
|---|---|
| the pre-play tooth (PREREG §5), at the pin, on the registered engine | **PASS**, 5 / 5 units (2026-10-30 17:27) |
| raw arm: counted units / engine / bytes / regime / search stamp | 160 / 160 counted, 0 exhausted; all 3,200 games on `port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129`; P\* `814823c5…91ed`; greedy; no search stamp |
| the scorer's engine (`live/engine.txt`) | the registered one, re-checked before each of 31 chunks |
| the draw and the quota prefix | PASS (each panel's registered-order prefix to its quota) |
| preset | every searched row `r0s4_rolls` |
| G-BRANCH | **one** unit failed it on both attempts: `pstar__forum72__212__g16__branch__t12n13`, classed a PEER DROP by AM-DL-PEER (below); no other failure |
| the capture prefix (2,091 counted branches) | PASS, 0 differ before their branch turn |
| null controls | **135 / 135** counted, each replays its raw game whole (protocol, result, length); 0 peer drops |
| forced action | PASS: every counted branch played S′'s scored pick at its raw game's series index |
| branch engines | one, the registered one |
| plan | PASS: 2,100 branches for 2,100 scored deviations, plus 135 nulls |
| drops | 9 of 2,100 branches (0.43 %), all PEER DROPS (F 6 = 1.0 %, H 1, home72 2, p72 0): far under the 10 % cap |
| G14 re-derived guards (count mode) | 160 units checked, 0 hits, 0 unavailable |
| record | COMPLETE, `void: null`, `cells_inconclusive: []` |
| by hand | the estimate, both bounds and the class equal the rule's |

**Cost (ESTIMATE was ≈ 43 CPU-h):** 36.6 CPU-h measured. Raw lane 7.5, S′ 7.5, T 6.8, Speed marker 0.5, branches 14.2
(2,244 attempts at ≈ 23 CPU-s), tooth 0.04. Wall: the raw lane 9.6 h (10-30 17:27 → 10-31 03:04), scoring 15.7 h
(→ 18:44), branches 14.4 h (→ 11-01 08:08 PST).

## Against the predictions (PREREG §10, GUESSES)

- Pooled-U mean D ≈ +0.012 (−0.01 to +0.035): **+0.0120**, on the point.
- home72 ≈ +0.008: +0.027, inside its wide interval.
- Live − T on U ≈ −0.02, more negative than home72: +0.002 on U against +0.009 on home72. The direction holds, the size
  does not, and neither is resolved.
- P(GO) ≈ 0.3: STOP.
- G-BRANCH 0 failures, all nulls identical, drops < 2 % a panel: nulls identical (135 / 135); **G-BRANCH 1 (a peer
  drop)**; drops ≤ 1.0 % a panel.

## FINDINGs

- **DL-7 — Kakuna does not always replay a game alone as it played it inside its 20-game unit.** 9 of 2,100 branches
  (0.43 %) left their raw game because Kakuna chose differently on a byte-identical view: 8 at its first choice
  (turn 1), 1 at turn 4. Every one diverged identically on both attempts, so the lone replay is self-consistent; the
  difference is between a fresh process and the same game inside a long one.
  - They cluster by raw game: forum72 unit 212 holds 3 and unit 218 game 4 holds 2. F holds 6 of the 9.
  - None of the 135 nulls diverged (0 / 135, one-sided 95 % UB ≈ 2.2 %), so the branch rate (0.43 %, Wilson [0.22, 0.81] %) is the
    sharper estimate.
  - The root cause is not established. GUESS: a near-tie in Kakuna's action scores flipped by a fresh process's
    numerics.
  - AM-DL-PEER (`e40d9663c`, blind) made such a divergence a DROP instead of a VOID. Without it the registered code
    would have VOIDed this read on the one G-BRANCH failure.
  - It corrects `designs/tools/reads.md` §4 ("Kakuna's choices never differ first"), which measured whole-unit
    replays.
- **DL-8 — one branch result was seen** by the operator while diagnosing DL-7: the dropped unit's, so it enters no
  estimate. Disclosed in AM-DL-PEER.
- **DL-9 — the pipeline's automatic record step was removed.** It would have run the pin's rule, which predates
  AM-DL-PEER rule 6, and written a VOID record. The record was written from `276c5b7e6`
  (`/home/goodlad/dev/gen3ai-reads/dlive_record_276c5b7e`) with the classification
  `live/branches/peer_drops.jsonl` (sha256 `c9631445…0afb`; classifier `be4c61f8…`, its `selftest` PASS).
  - Classification of all 2,244 attempts: 2,091 counted branches `ours_first` at exactly their own branch turn; 135
    nulls `identical`; 18 `peer_first` attempts (9 units).
- **DL-10 — T's own effect did not replicate on fresh states.** On these 1,793 departures T's mean improvement is
  +0.0099 per departure. On the re-gate's 469 it was +0.034, the figure §6's power analysis assumed (H +0.0045 here
  against +0.048 there; F +0.0136 against +0.025; p72 +0.0115 against +0.030).
  - So the read's GO power at the effect T now measures was ≈ 0.11, not 0.71.
  - Live D (+0.012) sits on T's (+0.010): Kakuna behaved like T on average. The headroom S′ showed on the gate's
    frozen states shrinks on fresh states before any opponent-model question arises.
  - GUESS: the re-gate's preset was chosen on those states (a winner's curse).
- **DL-5 (schedule) retired.** The branch lane took 14.4 h of wall, not 75–190 h: one-game units took a slot at once
  for most of the run.
- **DL-6 (attempts per supervisor life):** no restart happened, so no unit had more than two attempts.
- **Operational:** the edit worktree's static tier fails one environmental test (the freshness gate's
  `data/mods/gen3`, an unbootstrapped checkout's missing symlinks) at every ship of this read. It is unrelated to the
  read.
