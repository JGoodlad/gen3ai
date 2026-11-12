# X60-T / H-TEAMS look 2 (final), read as registered: **NO-TRANSFER** (2026-11-11)

X60-T = C0 + `--human-data teams:hr_spec_v1.r3@0.5` (2,295 human-built ladder teams, share 0.50 on both sides), P\*
+ 20M at X60T_PIN `7fe15726`, s3501–s3508. Control: C0 s3501–s3508 (production pin `f4b25238`); s3501–s3505 are
IMPORTED from X61P looks 1 and 2, s3506–s3508 are the X60T-C0b top-up. Rule `x60` ([`../PREREG.md`](../PREREG.md)
§5–§11, AM-X60T-1 §21, AM-X60T-2 §23, AM-X60T-3 §24). Record:
[`../../x35_breakers_stage_2026-10-17/reads/x60t_look2.json`](../../x35_breakers_stage_2026-10-17/reads/x60t_look2.json),
the verbatim `main.reads record --out` output (sha256 `5839533b0aeceeae…`, `check-record` OK, `COMPLETE`).
Run record (the two resumes, §13 / §15 per run, the K9(b) records, the three exhausted units and their retry): PREREG
§25, committed with the K9(b) records at `1f3a3b38a`, before `decide` ran. Look 1: [`../read/README.md`](../read/README.md)
(CONTINUE).

**In one paragraph.** Twenty million more steps with human ladder teams in half the self-play pool made the forks
better on those human teams themselves (the positive control R72, +1.69 pp in-house) and no better on human-built
teams they had not practised: −0.23 pp against Kakuna on forum72 (the primary, inside ±3 pp) and −0.34 pp on the
in-house forum72 meter. They paid 2.1 pp in-house on the pool's own teams. The lever moved what it trained on and
nothing else the registration decides on.

## Result (`main.reads decide`, verbatim; [`decision.json`](decision.json))

| read | statistic | X60-T | C0 | X60-T − C0 | class |
|---|---|---|---|---|---|
| **Δ_F (primary)** | Kakuna ckpt 34, greedy vs greedy, `forum72`, 60 units = 1,200 games per seed final, 8 v 8 | 0.5107 (4,903 / 9,600) | 0.5130 (4,925 / 9,600) | **−0.23 pp [−1.95, +1.49]** (95 %, §5's interval; σ̂ 0.28 pp, so the 1.0 pp seed-SD floor sets it; Welch not used, σ̂²'s LB95 −8.0e-5) | **NO-USEFUL-GAIN** (UB < +3 pp; equivalent within ±3 pp: true) |
| **Δ_P** | S on `home72` vs the frozen R, 7 × 500 mirrored pairs a final | 54.67 | 56.79 | **−2.12 pp [−2.99, −1.25]** (Welch, 97.5 % confidence, df 10.7; bar −2.0 pp) | **INFERIORITY-NOT-EXCLUDED** |
| **I_F** | S on `forum72`, in-house | 55.79 | 56.12 | **−0.34 pp [−1.14, +0.46]** (Welch 95 %); LB90 −0.84 | **not a GAIN** |
| **R72** (positive control) | S on 72 of the corpus's own teams | 57.60 | 55.91 | **+1.69 pp [+1.08, +2.30]** (Welch 95 %); LB90 +1.31 | **DELIVERED** |
| **Goal (a)** | ON's Kakuna rate on `home72`, units 0–6 | 642 / 1,120 = 0.573, Wilson [0.544, 0.602] | | bar 0.50 | **PASS** |
| **T − D1** (the contrast, 5 v 5, H + F at equal weight, 600 games a final and panel) | Kakuna | 0.5070 | D1 0.5010 | **+0.60 pp [−1.58, +2.78]** (σ̂ 0.65 pp → floor; Welch not used) | **NO-DIFFERENCE-AT-3PP** |

- **Verdict (§11): NO-TRANSFER** = Δ_F NO-USEFUL-GAIN at look 2, R72 DELIVERED, I_F not GAIN. `decide`'s reason,
  verbatim: "Δ_F NO-USEFUL-GAIN, R72 DELIVERED, I_F not GAIN".
- **What it licenses (§11, verbatim):** "stop this lever **for this six-seen, loser-enriched slice** at the plateau:
  it moved what it trained on, not unfamiliar teams on either meter". Nothing else: not human teams from step 0, not
  another slice of the corpus, not human replays as behaviour data. No recipe moves.
- **Δ_F per seed (X60-T − C0, pp), s3501–s3508:** −0.42, +0.17, +2.17, −0.75, +1.42, +0.75, −1.25, −3.92 (4 of 8
  positive). ON's finals 0.5017 / 0.5225 / 0.5200 / 0.5008 / 0.5000 / 0.5242 / 0.5092 / 0.5075 (SD 1.01 pp); C0's
  0.5058 / 0.5208 / 0.4983 / 0.5083 / 0.4858 / 0.5167 / 0.5217 / 0.5467 (SD 1.82 pp). C0 s3508 (656 / 1,200) is the
  highest final of either arm.
- **Look 1 recomputed on the same counts** (the rule's own block): +0.64 pp [−2.17, +3.45], NOT-RESOLVED, no stop.
- **Per seed, in-house (pp):** Δ_P −3.61, −3.36, −2.49, −1.46, −1.84, −1.10, −1.51, −1.59 (8 of 8 negative; arm SDs
  0.83 / 0.45); I_F −2.01, +0.01, −2.07, −0.46, +0.06, +0.36, +1.72, −0.29 (SDs 0.69 / 0.80); R72 +1.34, +2.04,
  +0.10, +1.21, +2.49, +2.71, +2.79, +0.86 (8 of 8 positive; SDs 0.49 / 0.64).
- **T − D1 per panel:** forum72 ON 0.5047 vs D1 0.5130 (−0.83 pp); `heldout72_hpfix` ON 0.5093 vs D1 0.4890
  (+2.03 pp). Registered false rates on this path: FORUM-BETTER 0.009, LADDER-BETTER 0.027 at σ 1.0 pp. The class
  says only that the two levers do not differ by 3 pp on H + F; §10's attribution limits stand (source is confounded
  with outcome selection).
- **Guards G-F / G-E / G-C: not read.** `guards_not_read`, verbatim: "Δ_F is NO-USEFUL-GAIN, not a GAIN: G-F / G-E /
  G-C are read only on a GAIN (§11)". The guards artefact stays the registered stub
  [`../read/guards_look2_not_played.json`](../read/guards_look2_not_played.json) (sha256 `282f843c…`, NOT PLAYED); the
  8 × 8 matrix, the `ge_x60t` reader's read and G-C were never needed.
- **Goal (a) passes but routes nothing here:** it separates ADOPT-CANDIDATE from TRADE, on a GAIN only.
- **Against the registered guess (§16):** Δ_F +1.5 [−1.5, +4.5] → −0.23; I_F +0.8 [−0.5, +2.5] → −0.34; Δ_P −1.5
  [−3.0, 0.0] → −2.12; R72 +0.8 [−0.3, +2.5] → +1.69; T − D1 −0.5 [−3.5, +2.5] → +0.60; Δ_H +2.5 [−1.0, +6.0] →
  +2.77. All six inside their 80 % intervals. The final Δ_F class NO-USEFUL-GAIN was given 0.33.

## REPORTED, never decided (§12; Kakuna, units 0–6 = 140 games per seed final and panel, 8 v 8)

| panel | X60-T | C0 | Δ (95 %, Newcombe, binomial only; [`by_hand.json`](by_hand.json)) | its contamination (§12) |
|---|---|---|---|---|
| **Δ_H** `heldout72_hpfix` | 533 / 1,120 = 0.476 | 502 / 1,120 = 0.448 | +2.77 pp [−1.36, +6.88] | H is the corpus's own population |
| **Δ_home72** | 642 / 1,120 = 0.573 | 615 / 1,120 = 0.549 | +2.41 pp [−1.70, +6.51] | the pool's own teams, against Kakuna |
| **Δ_P72** | 579 / 1,120 = 0.517 | 514 / 1,120 = 0.459 | +5.80 pp [+1.66, +9.92] | P72 is fill-adjacent to the corpus |
| **U** (H, F, P72 at equal weight; F at its 9,600 games) | 0.501 | 0.473 | +2.78 pp [+0.78, +4.78] (φ̂ 0.81, not widened) | both of the above |
| **F + P72** (the goal-(b) statistic, ruling `4f555e57`) | 0.514 | 0.486 | +2.79 pp [+0.60, +4.97] (φ̂ 0.77, not widened) | P72's |
| **Δ_H − Δ_F** | | | +3.00 pp | |

- **Every reported Kakuna panel but F reads positive, and P72's interval excludes 0.** By registration these decide
  nothing: they are 140 games a final against F's 1,200, their intervals here carry no seed term, and the two panels
  that move are the ones the corpus is adjacent to (P72 by the fill that completed its teams, H by population). The
  human-built panel the corpus is not adjacent to, F, is flat on both meters. That is the pattern §11's sentence
  names: the lever moved what it trained on.
- **Kakuna's home72 (+2.41) and the in-house Δ_P (−2.12) disagree in sign.** Δ_P is the decided meter (56,000
  mirrored games an arm against the frozen reference pool, 8 of 8 seeds negative); Δ_home72 is 1,120 games an arm
  against one opponent and spans 0.
- **Not produced by this read (FINDING 6):** §12's per-team proximity terciles and the belief / intent calibration on
  the Kakuna cells. No registered module computes them. The registration's static lines stand unchanged: the corpus
  is 39.0 % winners (median 31 turns) against 54.3 % of all ≥ 1500 seats; 58 pack-identity and 7 engine drops at the
  pin; 709 teams touch a 10-26 parity class.

## §13: Δ_F without the three resumed ON seeds (descriptive only; [`by_hand.py`](by_hand.py))

Three of eight ON seeds resumed once each on the T2 parity near-tie (s3501, s3506, s3508; PREREG §22.2, §25.2), which
meets §13's ≥ 3 bar. C0's finals ran 0 restarts.

| line | X60-T − C0 | note |
|---|---|---|
| all eight (the verdict) | −0.23 pp [−1.95, +1.49] | the by-hand formula reproduces the rule's interval exactly |
| **ON s3502, s3503, s3504, s3505, s3507 against C0's eight** | **−0.25 pp [−2.21, +1.71]** | §5's V generalised by hand to 5 v 8 (the rule takes equal arms only); UB < +3 pp |
| the same five seeds, 5 v 5 | +0.35 pp [−1.83, +2.53] | the rule's own `x60_interval`; UB < +3 pp |

The reading does not depend on the resumed seeds.

## Determinism guard (§4, §21.3): PASS

`c0_s3501__forum72__00`, re-played at the play pin, against the unit imported from X61P look 1
(`~/gen3ai_archive/x61p_n1p_read/look1`): tuple-equal, choices `equal`, counted.

## VOID (§13, §15): every one of the 21 runs PASSES ([`void.json`](void.json))

| runs | K9(b) at arm end (`k9b/…json`) | pin (`void.pin`) | dose vs `rb_x35_c0_s3501`, relative 1e-6 |
|---|---|---|---|
| x60t_s3501–s3508 | PASS × 8 (judged max / p99 of the five new: 2.36e-4 / 3.29e-5, 2.55e-4 / 3.40e-5, 3.58e-4 / 4.00e-5, 2.40e-4 / 3.01e-5, 1.79e-4 / 3.38e-5; s3501–s3503: look 1's README) | `7fe15726…` | PASS × 8: dose_rate 2.0440550334496103e-08, lr_median 1.0047e-4, upd/step 2.0345e-4, identical to the reference |
| c0_s3501–s3508 | PASS × 8 (the three new: 2.40e-4 / 3.34e-5, 1.74e-4 / 3.71e-5, 2.88e-4 / 3.38e-5) | `f4b25238d` | PASS × 8, identical |
| d1_s3501–s3505 (`rb_x35_d1wide_*`) | PASS × 5 | `440b61c32` | PASS × 5, identical |

The operator's checks that no module reads (the `[HUMAN-DATA]` stamp and `[TEAM-MIX]` banner in every ON launch log
and in neither control log, the held-out panels line at 0 hits, no `corpus_<n>` line, the T54 pool-seed check,
`sidecar_audit` at one pin span a run) are PREREG §25.3's table, written before `decide`.

## Pre-record checks

| check | result |
|---|---|
| **Floor** (§5: ≥ 1,200 · 8 − 20 = 9,580 counted F games an arm) | 9,600 / 9,600, after the retry below |
| **Completeness** | `COMPLETE`; `cells_inconclusive` empty; `reported_incomplete` empty; no repins |
| **One regime a meter** | F `00c8c0114bdf0836` (own, both C0 imports and the determinism unit); S home72 `d4de0fd7c16a2658` (own and both imports); I_F `0c972bc06eab468f`; R72 `1bfe0b8a101ba8fb`; the REPORTED panels pool at one regime a panel, status OK (H `28e65ac93633f6b5`, home72 `2e756ff1ca5e4f40`, P72 `b7547f904af56acc`); engine E3 on every pooled panel |
| **Recorded seeds** | all 21 runs: the run's recorded seed equals the spec's |
| **Re-derived anchor guards** (`count` mode) | 1,712 units checked, **0 hits**, not held. 112 units `unavailable` for the forced-end half: every `home72` unit (77 played here, 35 imported), FINDING 5 |
| **Team ids** (§9, REVIEW I M5) | no corpus or R72 team id, and no `corpus_<n>` string, in any file of this folder or in the stage record (checked against all 2,295 corpus ids and the 72 R72 file names) |

## The three exhausted units (PREREG §25.5)

The anchor lane's first life ended `done` on 2026-11-11 15:23 with three units `not_counted`:
`x60t_s3505__forum72__33`, `c0_s3506__forum72__34`, `x60t_s3508__forum72__39`. Each had lost both attempts of that
supervisor life to guard G2: the Kakuna peer's log held 986 "Force resetting due to long-tail error" lines a unit,
the Metamon long-tail fault of `designs/ops/EXTERNAL_ANCHORS_SOP.md` §3.1 (re-run at the same seed, never salvage).
ON stood at 9,560 counted F games, under the floor, so a read then was INCONCLUSIVE, whose registered remedy (§11)
is to play the missing piece and re-decide with nothing else changed.

They were re-played by the tool's own resume and nothing else: `python -m main.reads resume --state
~/gen3ai_archive/x60t_read/look2 --lane anchor --hours 6 --concurrency 2`, in the playing checkout at the same pin,
16:40:48 → 16:45:46. A new supervisor life retries every uncounted unit at its registered seed and replays no
counted unit; the operator chose no unit. Each counted on its first attempt; the lane ended `done`,
`not_counted []` ([`anchor_supervisor.jsonl`](anchor_supervisor.jsonl)). In all, 11 attempts over 8 units were
voided by that fault; none of their games enters any cell.

## How it was read

- Spec [`../read_spec_look2.json`](../read_spec_look2.json), byte for byte (sha256 `a5ad7411…8ed7`). State
  **`~/gen3ai_archive/x60t_read/look2`**. [`cells.json`](cells.json), [`inputs.json`](inputs.json),
  [`decision.json`](decision.json), [`void.json`](void.json), [`models.json`](models.json), [`code.json`](code.json),
  [`anchor_supervisor.jsonl`](anchor_supervisor.jsonl) and [`s_supervisor.jsonl`](s_supervisor.jsonl) are the
  state's.
- **Playing code `f564c333f`** (clean tree), checkout `~/dev/gen3ai-reads/x60t_play_de15dfc7`, no repins, no holds.
  CPU only, nice 19 on the agent cores; Kakuna at concurrency 2 on the heavy slots' short lane, S in 40-minute boxes.
- **Kakuna: 1,127 units = 22,540 games played**, all counted: primary 480 (ON s3504–s3508, C0 s3506–s3508), D1 300
  (s3501–s3505 on H and F, units 0–29, re-played at E3), ON's H units for the contrast 150, home72 77, P72 77, H 42,
  the determinism unit 1. Imported, never re-played: ON s3501–s3503's 180 primary units (look 1's state) and C0
  s3501–s3505's primary, reported and S `p` cells (X61P looks 1 and 2).
- **S: 43 nodes of 7 × 500 mirrored pairs:** `p` 11 (ON's eight, C0 s3506–s3508), I_F 16, R72 16.
- **Wall:** both lanes started 2026-11-09 03:36. S done 11-11 13:18. Kakuna's first life done 11-11 15:23; the retry
  life 16:40 → 16:45.
- `assemble`, `decide` and `record` ran once each from the main checkout at `1f3a3b38a` (§4, §23.1), at 16:52.

## Hazards (FINDINGS)

1. **Three units were exhausted and the ON arm sat one unit under the floor** until the tool's resume re-played them
   (above). The retry ran under `--hours 6`, not the registered lane's 150: a stop bound only.
2. **The operator saw the three voided attempts' summaries** (9, 9 and 10 wins of 20) while finding the cause, before
   the retry. They are void by G2 and a clean replay at the registered seed does not depend on them.
3. **The eight new K9(b) arm-end records were not on main when the lanes finished.** The VOID gate reads them from
   the recorder's checkout, so they were produced (`main.x35.kit k9b`, CPU) and landed with PREREG §25 at
   `1f3a3b38a` before `assemble`. All PASS.
4. **Three of eight ON seeds resumed** (s3501, s3506, s3508), so "replicate = same pin + seed" is lost for them. The
   tie row behind s3508's failure (bucket 64) was not re-derived and its fatal step was not read (≤ 1,000,000 steps
   re-trained). §13's without-resumed line reads the same.
5. **The forced-end re-derivation is `unavailable` on all 112 home72 units** (PREREG §19 FINDING 8, registered: the
   play pin `de15dfc7b` takes no capture for home72's `ours_challenge` games). The peer-fault half ran; 0 hits. It
   touches goal (a) and Δ_home72 only, and neither enters NO-TRANSFER.
6. **Two of §12's REPORTED items were not produced:** the per-team proximity terciles and the belief / intent
   calibration on the Kakuna cells. No module of the registered read computes them; building one is a new unit. They
   decide nothing. `EXPERIMENT_BACKLOG.md`'s "every class" line (the belief calibration feeds MF-3 step (a)) is
   therefore still owed.
7. **Δ_P's whole interval is below zero** (−2.12 pp [−2.99, −1.25], 8 of 8 seeds). Human teams at share 0.5 cost
   in-house strength on the pool's own teams, as D1 (−1.61) and N1-P (−3.28) did at the same share.
8. **The REPORTED panels read positive** (P72 +5.80 pp with a binomial interval that excludes 0; F + P72 +2.79).
   They are contaminated as §12 names, small, and decide nothing; they are a reason not to read NO-TRANSFER as
   "human teams do nothing off the pool", only as the registered sentence.
9. **σ̂ is small (0.28 pp),** so the 1.0 pp seed-SD floor sets Δ_F's interval; C0's own per-final SD (1.82 pp) is
   larger than ON's (1.01 pp), driven by s3508 (0.5467) and s3505 (0.4858).
10. **C0 is a shared control (§11, m7)** with X61P and D3, and now X67's columns. A high C0 lowers all of them.
11. **§23.7 names the training session as the read's operator;** this read was run by a delegated read operator of
    the orchestrator. The play pin's `main.reads` has no `pid` / `--by`, so the lanes were watched by PID.
