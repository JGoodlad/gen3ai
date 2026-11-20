# Q6-N — the flat policy tower retired, one fresh cell against BOUNDARY-1's banked cell C: PRE-REGISTRATION (2026-11-20)

**Status: REGISTERED 2026-11-20, FULL tier, blind: before any Q6 training step, any Q6 game, any look and the card
check. NOTHING HAS TRAINED and this file authorises no launch.** The GPU window is the orchestrator's; its first item
is § 7's card check. Nothing below changes after the first Q6-N run launches except by a dated amendment written
before the look it touches, saying why.

**The question.** From scratch, does an agent whose move chooser reads the attention trunk through one learned query
(`--policy-readout trunk`) end stronger, weaker or not detectably different at 100M steps than BOUNDARY-1's (B1's)
banked cell C, which keeps production's flat policy tower?

**The tier is FULL** (`../../README.md` § Registration tiers): a WIN makes an architecture change a candidate for the
next fresh lineage.

**What was read to write this.** B1's 50M look as recorded (`eff47b0df`; `../boundary1_scratch_2026-11-09/record/look_50M/verdict.json`),
which includes cell C's scores against B1's other arms, and B1's two pin pilots. No Q6 run, pilot, game or number
exists. The width (§ 2) was the orchestrator's ruling under a queue rule written before B1's look.

Abbreviations: B1 = BOUNDARY-1; pp = percentage points of win rate; SD = standard deviation; df = degrees of freedom;
σ = the seed SD (the SD across seeds of one arm, game noise removed); ρ = a step-time ratio, arm over control;
GPU-h = hours of the one training card; N = the deciding view, ½ home72 + ⅙ H + ⅙ F + ⅙ P72 (B1 §5.1).

## 1. What is adopted, and what this registration changes from the drafts

**Adopted by reference, unchanged:** [`FRAME.md`](../fresh_cell_frame_2026-11-17/FRAME.md) (its rules as last changed
at `52353b1b7`; this registration's commit edits its status line only) and the lever card
[`CARD_Q6_policy_readout.md`](../fresh_cell_frame_2026-11-17/CARD_Q6_policy_readout.md) §1 to §5, for the arm Q6-N
only. The rule is `b1_fresh_cell` (`main.b1.fresh_rule`, `bc76e9657`) and the kit `python -m main.b1.fresh`
(`52353b1b7`). **No statistic is added or changed.** The drafts stay where they are: `src/main/b1/fresh_test.py` reads
the frame folder's arm files and argvs.

**Changed by the orchestrator's rulings of 2026-11-20** (the brief of this registration; each is a choice made before
any Q6 number exists):

| ruling | what it sets | what it replaces in the drafts |
|---|---|---|
| **R1, width 128** | Q6 is registered at width 128 only (arm Q6-N against cell C). **The 256 arm Q6-W is NOT registered: DROPPED.** Reason: B1's 50M look put width at CONTINUE with −0.37 [−3.32, +2.57] pp, so width is not ahead, and the post-B1 queue's row for that state ("U / 50M harm stop": as Neither, on the narrow side; `~/.cache/gen3ai/handoffs/post_b1_queue_proposal_2026-11-16.md` §3 and Addendum 2026-11-18) keeps the lever cells narrow | the queue's Addendum 2026-11-17 ("Q6 at 256 if width is not ahead") and the card's Q6-W rows; `fresh_arm_q6w.json` and its records remain DRAFT, never registered |
| **R3, the card check** | the ≤ 3 GPU-h check is the FIRST item of Q6's GPU window, run by the Training session under the lease; its criteria are registered here (§ 7) | card §7's order (the check, then the registration) |
| **R4, the order** | Q6 runs BEFORE Q5 ([`../q5n_epochs_fresh_2026-11-20/PREREG.md`](../q5n_epochs_fresh_2026-11-20/PREREG.md)), whatever B1's width factor does. Reason: Q6 has a measured mechanism (the tower's first layer 0.897 saturated, the policy gradient cut 15 to 18×, 43 % of the parameters; the 11-09 architecture diagnosis `a8265f36b`). The order changes neither statistic | card §7 ("Q6 ranks above Q5 if B1's width factor wins") |

## 2. The lever and the arm

- **The lever: `--policy-readout trunk --allow-nonproduction-arch`**, typed on top of cell C's argv; nothing else moves.
  Declared keys: `policy_readout` (`tower` → `trunk`) and `allow_nonproduction_arch` (False → True). 1,547,039
  learner parameters against C's 2,609,183 (−40.7 %; card §3). The value path is untouched.
- **The pin is B1's launch pin, `15136b61634b667bb9da5ccc6647ff9265238829`, and it does not move** (§ 6).
- **The banked control: cell C**, `rb_b1c_s5101`, `rb_b1c_s5102`, `rb_b1c_s5110`, the files B1 read.

| run | seed | phase 1 (fresh, to 50,000,000) | phase 2 (only on CONTINUE) |
|---|---|---|---|
| `rb_q6n_s5211` | 5211 | the argv below | `--model models/rb_q6n_s5211/final_model.zip --steps 100000000` |
| `rb_q6n_s5212` | 5212 | the same, its own `--seed` / `--run-name` | the same form |
| `rb_q6n_s5213` | 5213 | the same | the same form |

```
python -m main.launcher --restart-interval-hours 6 --pin-commit 15136b61634b667bb9da5ccc6647ff9265238829 --arch production --ent-coef 0.0125 --behaviour-check warn --policy-readout trunk --allow-nonproduction-arch --checkpoint-every-steps 1000000 --steps 50000000 --seed 5211 --device cuda --log-level periodic --run-name rb_q6n_s5211
```

The nine commands (three runs, both phases) are [`describe_q6n.txt`](describe_q6n.txt), printed by the kit from the
registered arm file [`fresh_arm_q6n.json`](fresh_arm_q6n.json). FRAME §2 items 3 and 4 (the base, the tail, torch
`2.8.0+cu126` from step 0, one pin entry, the core split recorded at each launch) hold for every run.

## 3. The comparison, the looks, the classes (FRAME §3 to §5, restated; the frame wins any disagreement)

- **The meter:** every fresh run against every banked run, 9 cells a look, 45,000 games; each cell 1,000 mirrored
  pairs on home72 and 500 on each of H, F and P72, both sides greedy, `--seed 0`, the pin's engine, ONE device.
- **The statistic:** `main.b1.cross.simple_effect(design, scores, "<fresh>", "<banked>", view="N")`: the estimate is
  the mean of the 9 cells − 50; its variance is s²_fresh / 3 + s²_banked / 3, each s² the variance of that cell's
  three row means **floored at 1.76²**; 4 df; two-sided 90 %. The half-width is never below 3.06 pp. A bound within
  0.005 pp of a threshold fails the comparison. **N decides; every other view is REPORTED.**
- **The 50M look:** CONTINUE iff the estimate on N > +0.005 pp; else **STOPPED-AT-50M**, reported with its interval
  on every view and NO class (HARM-AT-50M beside it, as a description, if the upper end is below −2.0 pp).
- **The 100M look:** **WIN** (lower end > 0), **LOSS** (upper end < 0), **NO-USEFUL-GAIN** (upper end < +4.0), else
  **UNDECIDED**. **INCONCLUSIVE** by B1 §10's rules as the kit applies them to all six runs (FRAME §3 Validity).
- **What each outcome does** is FRAME §5, unchanged: a WIN is a CANDIDATE for the next fresh lineage, never an
  adoption (in-house until a registered Kakuna read confirms it); a LOSS retires this lever at this width under this
  recipe, never wider; NO-USEFUL-GAIN and STOPPED-AT-50M are never "no effect"; UNDECIDED defaults, after B1
  Amendment 6's 72-hour veto window, to "not adopted, no fourth seed"; production's `tower` does not move on any
  outcome of this cell.
- **The look files.** Each fresh run plays the first `checkpoint_<N>_steps.zip` at or after the look's step. The
  banked runs play the bytes B1 read. For the 50M look those are fixed now: B1's 50M `look.json`, banked here as
  [`b1_look_50M.json`](b1_look_50M.json) (sha256 `925bfb32…dedc868`; `rb_b1c_s5101` at 50,000,412 `f706243d…`,
  `rb_b1c_s5102` at 50,000,041 `c46ddab8…`, `rb_b1c_s5110` at 50,000,220 `cbcccec7…`), the kit's `--b1-look`. For
  the 100M look they are whatever B1's own 100M look reads for C (survivor phase 2, expected ≈ 2026-11-21); that
  `look.json` is banked beside this file by a dated amendment before Q6's 100M look. If B1 never plays C at 100M,
  the kit's `plan` refuses and the look is not played (FRAME §2 item 7; then §7 of the frame).
- **Retention** (FRAME §2 item 6) already holds: B1's twelve run dirs are never thinned, moved or rewritten
  (`../../models_retention_policy.md`, `tiers.BANKED_B1_PREFIXES`, B1 Amendment 7, commit `07fc04c1d`). Nothing is added.

## 4. Seeds, collisions, a lost run

- Seeds 5211, 5212, 5213; spare 5219; the pilot's 5299. The scan of 2026-11-20 (693 sidecars, 4,173 directories under
  the main `models/` and `~/gen3ai_archive`) finds no hit for any of them or their names
  ([`seed_scan_q6n.txt`](seed_scan_q6n.txt)). It is re-run on the day of the first launch (FRAME §6 item 2).
- **A lost run** follows B1 §10's closed list: a run is replaced, from step 0 with seed 5219 (`rb_q6n_s5219`), only
  when its files are lost or unreadable, or it was launched with an argv other than its registered one. ONE
  replacement exists for this cell. The kit has no replacement switch for a fresh run, so the replacement is a dated
  amendment that edits `fresh_arm_q6n.json` (the seed and the run name of the lost row), written with the cause
  before any look that would contain the run. Anything else (a NaN, an out-of-memory failure, a K9(b) VOID) is "lost
  to its lever": not replaced, and that look is INCONCLUSIVE. A second loss is INCONCLUSIVE too.

## 5. Power: the frame's σ amendment, from cell C's own spread at 50M (FRAME §4's obligation)

This is the dated amendment FRAME §4 asks of the first card registered after B1's 50M look. The script is
[`power_c50m.py`](power_c50m.py), its output [`power_c50m.log`](power_c50m.log): B1's own helpers
(`power_matrix.py` `simulate`, `simple_np`, `classes_100`), 50,000 draws a row, numpy seed 20261120, the first 1,000
draws of every row classed again by `main.b1.cross` itself: **0 mismatches in 48,000**. It reproduces the frame's
three rows (σ 1.76: half-width 3.28, P(WIN) 0.16 / 0.61 / 0.93 at +2 / +4 / +6).

**What was measured.** B1's wide 50M intervals came from the WB arm's spread (raw variance of its run means 18.8 to
25.6 on N), not from C. Cell C's own three runs, in each of its three crosses on N:

| C against | C's three run means (pp) | raw variance | floored to 3.0976? |
|---|---|---|---|
| W | 51.20 / 51.91 / 52.02 | 0.2000 | yes |
| Br | 48.67 / 48.57 / 48.70 | 0.0045 | yes |
| WB | 45.28 / 46.47 / 46.80 | 0.6411 | yes |

Mean raw variance 0.2819; a run mean carries cell noise 0.68² / 3 = 0.154; so **σ_C ≈ 0.36 pp at 50M**. It is ONE
estimate on 2 df (the same three runs each time): its 95 % upper bound is 2.31 pp and its lower bound is 0.
**Consequences, stated plainly:**

1. **The floor, not C, sets the banked side of the interval.** Every measured C variance is far under 1.76², so the
   banked side contributes 3.0976 / 3 whatever C does, and the half-width is at least 3.06 pp. The floor stays
   (FRAME §4: a small measured σ must not narrow the interval).
2. **A +2 pp lever is essentially never a WIN.** With a 3.06 pp half-width a WIN needs an estimate above 3.06.
3. **The fresh cell's σ is the unknown that is left**, and the trunk readout is a new architecture: nothing measures
   its seed spread. **UNVERIFIED:** σ of the fresh cell at either step, and σ of C at 100M.

**The 50M rule, P(CONTINUE)** at a true 50M effect (closed form):

| σ fresh / σ banked | SD of the estimate | −4 | −2 | 0 | +2 | +4 | +6 pp |
|---|---|---|---|---|---|---|---|
| 0.36 / 0.36 (both as tight as C measured) | 0.37 | 0.000 | 0.000 | 0.50 | 1.000 | 1.000 | 1.000 |
| 1.76 / 0.36 | 1.06 | 0.000 | 0.03 | 0.50 | 0.97 | 1.000 | 1.000 |
| 2.5 / 0.36 | 1.48 | 0.003 | 0.09 | 0.50 | 0.91 | 0.997 | 1.000 |
| 3.1 / 0.36 | 1.82 | 0.014 | 0.14 | 0.50 | 0.86 | 0.99 | 1.000 |
| 1.76 / 2.31 (C at its upper bound) | 1.69 | 0.009 | 0.12 | 0.50 | 0.88 | 0.99 | 1.000 |
| 1.76 / 1.76 (the frame's) | 1.45 | 0.003 | 0.08 | 0.50 | 0.92 | 0.997 | 1.000 |
| 2.5 / 2.5 (the frame's) | 2.05 | 0.03 | 0.16 | 0.50 | 0.83 | 0.97 | 1.00 |
| 3.1 / 3.1 (the frame's) | 2.54 | 0.06 | 0.22 | 0.50 | 0.78 | 0.94 | 0.99 |

**The 100M classes, P(WIN) / P(LOSS) / P(NO-USEFUL-GAIN) / P(UNDECIDED)** at a true 100M effect, IF the 100M spreads
are as in the row (a true −4 pp mirrors +4: LOSS with the WIN column's probability):

| σ fresh / σ banked | median half-width | 0 | +2 | +4 | +6 | +8 pp |
|---|---|---|---|---|---|---|
| 0.36 / 0.36 | 3.06 | 0.000 / 0.000 / 0.99 / 0.006 | **0.002** / 0 / 0.002 / 0.996 | **0.99** / 0 / 0 / 0.006 | 1.00 | 1.00 |
| 1.76 / 0.36 | 3.06 | 0.002 / 0.002 / 0.73 / 0.27 | **0.13** / 0 / 0.12 / 0.75 | **0.72** / 0 / 0.002 / 0.27 | 0.98 | 1.00 |
| 2.5 / 0.36 | 3.39 | 0.010 / 0.010 / 0.56 / 0.42 | **0.15** / 0 / 0.15 / 0.70 | **0.56** / 0 / 0.01 / 0.43 | 0.89 | 0.99 |
| 3.1 / 0.36 | 3.88 | 0.022 / 0.021 / 0.45 / 0.51 | **0.15** / 0 / 0.15 / 0.70 | **0.47** / 0 / 0.02 / 0.51 | 0.78 | 0.94 |
| 1.76 / 2.31 | 3.61 | 0.018 / 0.018 / 0.51 / 0.45 | **0.16** / 0 / 0.16 / 0.68 | **0.53** / 0 / 0.02 / 0.45 | 0.86 | 0.98 |
| 1.76 / 1.76 (the frame's) | 3.28 | 0.011 / 0.011 / 0.60 / 0.38 | **0.16** / 0 / 0.16 / 0.68 | **0.61** / 0 / 0.01 / 0.38 | 0.93 | 1.00 |
| 2.5 / 2.5 (the frame's) | 4.22 | 0.027 / 0.028 / 0.405 / 0.54 | **0.15** / 0 / 0.15 / 0.70 | **0.43** / 0 / 0.03 / 0.54 | 0.75 | 0.93 |
| 3.1 / 3.1 (the frame's) | 5.09 | 0.036 / 0.037 / 0.30 / 0.62 | **0.14** / 0.01 / 0.13 / 0.73 | **0.34** / 0 / 0.03 / 0.63 | 0.60 | 0.81 |

**In words. The power for a realistic lever is poor, and the rule is not changed for it.** A +2 pp lever (the size we
have detected at forks) reads WIN between never and one time in six (0.00 to 0.16) in EVERY row, and UNDECIDED about
seven times in ten or more. A +4 pp lever reads WIN between one time in three and nearly always (0.34 to 0.99); what
decides which is the fresh cell's own seed spread, which is unknown. A +6 pp lever reads WIN at least six times in
ten, and about nine in ten when both spreads are at or under 1.76. An arm with no effect continues past 50M half the time and then reads NO-USEFUL-GAIN
or UNDECIDED (WIN ≤ 0.04). The three banked runs are already trained: their own luck against the fresh cell is fixed
and unknown, and Q5-N reads against the same three (FRAME §5: the two verdicts are correlated; neither is ranked
against the other).

## 6. The pre-launch record on current main (2026-11-20, main `66edebfff`; FRAME §6 item 2 is re-run on launch day)

All at nice 19 on the agent cores with `CUDA_VISIBLE_DEVICES=""`, no card.

| check | result | record |
|---|---|---|
| the kit's argvs | the first line is the card's recorded argv token for token | [`describe_q6n.txt`](describe_q6n.txt) |
| `python -m main.checkargs --argv …` | exit 0; `parser : the PINNED commit 15136b61's`; 0 unrecognized; ARCH surface `policy_readout 'trunk'` against production's `'tower'` and nothing else; RECIPE surface `ent_coef 0.0125` alone. Identical to the 11-17 draft but for the timing | [`checkargs_q6n.txt`](checkargs_q6n.txt) |
| `python -m main.launcher --dry-run …` | exit 3 with exactly the two refusals an agent shell owes (`cuda devices`, `inherited nice`); every other preflight passes; nothing created | [`dryrun_q6n.txt`](dryrun_q6n.txt) |
| the resolved-config diff, in a clean detached checkout of the pin (`main.b1.fresh lever-check --pin-tree`) | exit 0, `ok true`, no problem: against C at 50M and at 100M, for all three seeds, the keys that differ are exactly `allow_nonproduction_arch`, `policy_readout`, `seed`, `run_name`. **Exactly one lever** | [`lever_check_q6n.json`](lever_check_q6n.json) |
| the seed scan | no hit | [`seed_scan_q6n.txt`](seed_scan_q6n.txt) |

**The pin does not move.** Main moved from `81594609b` (the drafts) to `66edebfff`; main's launcher and kit still
launch and diff the pin's parser with the same output.

## 7. The card check: the FIRST item of Q6's GPU window (≤ 3 GPU-h; criteria registered now, NOT run)

Run by the Training session under the lease, from the main checkout at nice ≤ 10 with the card visible, into an
EMPTY scratch `GEN3AI_MODELS_DIR` (the pilot is never a comparator and never enters `models/`). It is B1 §13 item 4's
pilot in the arm's launch shape at the launch pin. The seed scan for 5299 is re-run first; then the dry run of the
same argv must pass every preflight (FRAME §6 item 3). This file is not a window chain: the session writes its own.

```
python -m main.launcher --restart-interval-hours 6 --pin-commit 15136b61634b667bb9da5ccc6647ff9265238829 --arch production --ent-coef 0.0125 --behaviour-check warn --policy-readout trunk --allow-nonproduction-arch --checkpoint-every-steps 1000000 --steps 4500000 --seed 5299 --device cuda --log-level periodic --run-name q6n_pilot_s5299
```

**The pilot runs to 4,500,000 steps, not the draft's 2,500,000** (the registrar's change, flagged in § 10): both B1
pilots seeded the self-play pool only at their 4,000,000 evaluation, so a 2.5M pilot would end before the first
promotion that criterion 4 reads. If no promotion has happened by 4.5M, that is recorded as "not reached" and the
check is still read on the rest.

**PASS iff every one of 1 to 4 holds:**

1. the bf16 R1 compile completes and its compile-parity check passes (the child log's parity line);
2. the inference T2 CUDA-graph capture succeeds (the child log's `T2 up … backend graph on cuda`);
3. the trainer's own startup fit check passes, and `update_fit.json` records `oom` null and a headroom of at least
   1,024.5 MiB (the 1,024 bar with its 0.5 MiB resolution excluded, root rule 8); headroom and demand are recorded.
   Reference: B1's C pilot, demand 7,060.0 MiB, headroom 3,710.5 MiB;
4. no parity-gate FATAL and no non-finite FATAL in any generation of the child's full log through the end of the
   pilot (the first promotion included, if reached), and the update-10 canary's verdict is PASS.

**FAIL** = any of 1 to 4 does not hold. The card HOLDS: no Q6-N run launches, the failure comes back to the
orchestrator as a build, and a fix that lands after the pin sends the arm to FRAME §7 (it buys its own control).
**INCOMPLETE** = the pilot did not reach 4,500,000 steps for a reason outside 1 to 4 (a box fault, a stop by its
owner, the 3 GPU-h bound): neither PASS nor FAIL; it is reported and re-run or held by the orchestrator. Elapsed
time is never a criterion.

**RECORDED, deciding nothing:**

5. **ρ_Q6**, by ONE formula applied to the pilot and to B1's C pilot ([`pilot_rho.py`](pilot_rho.py)): (median
   `lifecycle/update_wall_s` + median `rollout/collect_ms` / 1000) / median steps a rollout × 10⁶, over the rollout
   rows with 300,000 ≤ step < 4,000,000. The reference on that window is **518.27 s per 1M steps** (38 rows, update
   share 0.825; [`pilot_rho_reference_c.txt`](pilot_rho_reference_c.txt); the run is
   `~/.cache/gen3ai/b1_pin_2026-11-12/models/rb_b1pinpilot_c_s9201`). ρ_Q6 prices the cell (§ 8) and is told to the
   orchestrator with the PASS; whether a price is too high is the orchestrator's, written before the first run;
6. the cold-start time, whether the shared compile cache hit or missed, and the first evaluation's bot score.

`python pilot_rho.py <the pilot's run dir> --reference <B1's C pilot run dir>` prints ρ_Q6. The record of the check
(the six items, the child-log lines, `update_fit.json`) is appended to this file as a dated record before the first
Q6-N launch (FRAME §6 item 4).

**Cost:** 4.5M steps at 518 s per 1M is 0.65 GPU-h × ρ_Q6, plus two evaluation cycles and a cold start (B1's C pilot:
0.33 h cold with a warm compile cache): about 1.0 to 1.3 GPU-h. **UNVERIFIED:** the compile time of the new graph
(the shared cache will miss); the bound is 3 GPU-h.

## 8. The order and the cost

**Order (R4):** Q6-N's window comes first, Q5-N's after it. Within the window: § 7's check; on PASS the three runs to
50M in seed order; the 50M look; on CONTINUE the three resumes to 100M; the 100M look.

| item | GPU-h | provenance |
|---|---|---|
| the card check | ≈ 1.0 to 1.3 (bound 3) | § 7 |
| the cell to 50M | ≈ 28.6 × ρ_Q6 | FRAME §4: B1's measured 9.55 h a narrow run per 50M |
| the cell from 50M to 100M, only on CONTINUE | ≈ 30.1 × ρ_Q6 | the same |
| each look (45,000 games) | ≈ 0.33, plus a cold engine start | FRAME §4; **UNVERIFIED** (B1's look engine came up cold in 1,360.7 s, ≈ 0.38 h; the trunk readout is a new inference graph) |
| **total** | ≈ 30 if it stops at 50M; ≈ 61 to 63 to a 100M class; **≈ 46 expected with no true effect** (ρ_Q6 = 1) | 1.2 + 28.6 + 0.5 × 30.1 + looks |

ρ_Q6 is **UNVERIFIED** until § 7 (the arm holds 40.7 % fewer parameters; the update is 0.825 of a step's time).

## 9. What this cell cannot say (FRAME §8, plus this card's)

- It reads at 100M from scratch, not at the plateau: a slow starter can be STOPPED-AT-50M or UNDECIDED and still win
  later. None of those is a retirement.
- It cannot resolve an effect below about +4 pp (§ 5), and it cannot show "no worse": a cheaper, smaller network
  that reads NO-USEFUL-GAIN or UNDECIDED is NOT adopted for its size or its speed by this cell.
- It reads the lever at width 128 under B1's recipe (`--ent-coef 0.0125`). Nothing is said about width 256 (R1).
- The mechanism in R4 (a saturated tower layer) is why the cell is bought first. It is not a prediction this
  registration scores: no class depends on it, and the saturation is not re-measured here.
- Time is a confound through the box only (FRAME §8): the fresh cell trains weeks after C on the same code and card.

## 10. FINDINGS and the registrar's choices (each flagged for the orchestrator; all amendable while still blind)

- **F1. The pilot is 4.5M steps, not the draft's 2.5M** (§ 7): the draft's length ends before B1's observed first
  promotion (4.0M), which would leave criterion 4's promotion half unread. Cost ≈ +0.3 GPU-h.
- **F2. B1's `rho.json` (527.67 s per 1M for C) cannot be reproduced**: the script that made it is not in the repo,
  and three plausible forms give 518 to 537 on the same rows. ρ is therefore defined by one committed formula applied
  to both runs (§ 7 item 5), never against 527.67.
- **F3. B1's 50M `look.json` existed only under `~/.cache/gen3ai/b1_aside/`**, yet the kit refuses a plan without
  it. A byte copy is banked here with its hash (§ 3); B1's 100M `look.json` is owed the same before Q6's 100M look.
- **F4. Power for a +2 pp lever is 0.00 to 0.16 in every scenario** (§ 5), and the floor, not C's measured spread
  (σ_C ≈ 0.36 pp, 2 df, upper bound 2.31), sets the interval. Not a reason to change the rule; it is what three runs
  against three buy.
- **F5. The kit has no replacement path for a lost fresh run** (§ 4): a replacement is an amendment of the arm file.
- **F6. Nothing in code holds the FRESH cell's look checkpoints** (`tiers.BANKED_B1_PREFIXES` names B1's runs only).
  They are referenced by this registration, so the standing rule (run data is the owner's) covers them; no thinning
  of `rb_q6n_*` before this read closes. **UNVERIFIED:** that a grooming plan would refuse them.
- **F7. UNVERIFIED and priced as such:** ρ_Q6, the compile time of the new graph, the look's cost, σ of the fresh
  cell, σ at 100M.
