# Q5-N — five PPO epochs at matched dose, one fresh cell against BOUNDARY-1's banked cell C: PRE-REGISTRATION (2026-11-20)

**Status: REGISTERED 2026-11-20, FULL tier, blind: before any Q5 training step, any Q5 game, any look and the card
check. NOTHING HAS TRAINED and this file authorises no launch.** The GPU window is the orchestrator's and comes AFTER
Q6-N's ([`../q6n_policy_readout_fresh_2026-11-20/PREREG.md`](../q6n_policy_readout_fresh_2026-11-20/PREREG.md), its
ruling R4); its first item is § 7's card check. Nothing below changes after the first Q5-N run launches except by a
dated amendment written before the look it touches, saying why.

**The question.** From scratch, under the production learning-rate controller, does an agent trained with five
passes over each batch at matched movement (`--n-epochs 5 --lr 6e-4`, X27's form) end stronger, weaker or not
detectably different at 100M steps than BOUNDARY-1's (B1's) ten-pass cell C?

**The tier is FULL** (`../../README.md` § Registration tiers): a WIN makes a recipe change a candidate for the next
fresh lineage.

**What was read to write this.** B1's 50M look as recorded (`eff47b0df`), which includes cell C's scores against
B1's other arms, B1's C pilot, and the card's dose figures for C (`python -m main.dose`). No Q5 run, pilot, game or
number exists. The width (§ 1) was the orchestrator's ruling under a queue rule written before B1's look.

Abbreviations: B1 = BOUNDARY-1; pp = percentage points of win rate; SD = standard deviation; σ = the seed SD; KL =
the Kullback-Leibler divergence between the policy after an update and the policy that collected the batch; LR =
learning rate; ρ = a step-time ratio, arm over control; GPU-h = hours of the one training card; N = the deciding
view, ½ home72 + ⅙ H + ⅙ F + ⅙ P72 (B1 §5.1).

## 1. What is adopted, and what this registration changes from the drafts

**Adopted by reference, unchanged:** [`FRAME.md`](../fresh_cell_frame_2026-11-17/FRAME.md) (its rules as last changed
at `52353b1b7`) and the lever card [`CARD_Q5_epochs.md`](../fresh_cell_frame_2026-11-17/CARD_Q5_epochs.md) §1 to §6,
for the arm Q5-N only. The rule is `b1_fresh_cell` (`main.b1.fresh_rule`, `bc76e9657`) and the kit `python -m
main.b1.fresh` (`52353b1b7`). **No statistic is added or changed.**

**Changed by the orchestrator's rulings of 2026-11-20** (each made before any Q5 number exists):

| ruling | what it sets | what it replaces in the drafts |
|---|---|---|
| **R1, width 128** | Q5 is registered at width 128 only (arm Q5-N against cell C). **The 256 arm Q5-W is NOT registered: DROPPED.** Reason: B1's 50M look put width at CONTINUE with −0.37 [−3.32, +2.57] pp, so width is not ahead, and the post-B1 queue's row for that state ("U / 50M harm stop": as Neither, on the narrow side; `~/.cache/gen3ai/handoffs/post_b1_queue_proposal_2026-11-16.md` §3, Addendum 2026-11-18) keeps the lever cells narrow | the card's Q5-W rows; `fresh_arm_q5w.json` and its records remain DRAFT, never registered. The queue's note that Q5's card carry the 4.4× sensitivity ratio "at 256" has no arm left to apply to: recorded, not used |
| **R3, the card check** | a ≤ 3 GPU-h check is the FIRST item of Q5's GPU window, run by the Training session under the lease; its criteria are registered here (§ 7) | card §7 ("No GPU check is owed") |
| **R4, the order** | Q6-N's window runs before this one. The order changes neither statistic | the queue's conditional order |

## 2. The lever, the dose, the arm

- **The lever: `--n-epochs 5 --lr 6e-4`**, typed on top of cell C's argv; nothing else moves. Declared keys:
  `n_epochs` (10 → 5) and `lr` (0.0003 → 0.0006). The network is C's (2,609,183 parameters).
- **"Matched dose", as card §2 defines it and this file registers it:** (a) the arm STARTS at the control's dose,
  `lr × n_epochs × optimizer steps an epoch` = 6e-4 × 5 × 2 = 3e-4 × 10 × 2; and (b) from there the SAME live
  controller acts on both cells (the EMA of the last epoch's `train/approx_kl`, α 0.20; above 0.02 the LR is divided
  by 1.2, below 0.005 multiplied by 1.2; a 7-rollout cooldown; floor `--min-lr` 1e-5; ceiling 2 × the starting LR,
  so 1.2e-3 for the arm and 6e-4 for C). The controller is held by the SAME TARGET, never by a fixed LR: C's
  controller ran live, so a frozen arm would differ in two levers (card §2's table).
- **The pin is B1's launch pin, `15136b61634b667bb9da5ccc6647ff9265238829`, and it does not move** (§ 6).
- **The banked control: cell C**, `rb_b1c_s5101`, `rb_b1c_s5102`, `rb_b1c_s5110`, the files B1 read.

| run | seed | phase 1 (fresh, to 50,000,000) | phase 2 (only on CONTINUE) |
|---|---|---|---|
| `rb_q5n_s5201` | 5201 | the argv below | `--model models/rb_q5n_s5201/final_model.zip --steps 100000000` |
| `rb_q5n_s5202` | 5202 | the same, its own `--seed` / `--run-name` | the same form |
| `rb_q5n_s5203` | 5203 | the same | the same form |

```
python -m main.launcher --restart-interval-hours 6 --pin-commit 15136b61634b667bb9da5ccc6647ff9265238829 --arch production --ent-coef 0.0125 --behaviour-check warn --n-epochs 5 --lr 6e-4 --checkpoint-every-steps 1000000 --steps 50000000 --seed 5201 --device cuda --log-level periodic --run-name rb_q5n_s5201
```

The nine commands are [`describe_q5n.txt`](describe_q5n.txt), printed by the kit from the registered arm file
[`fresh_arm_q5n.json`](fresh_arm_q5n.json). FRAME §2 items 3 and 4 hold for every run.

## 3. The comparison, the looks, the classes (FRAME §3 to §5; the frame wins any disagreement)

- **Deciding: matched STEPS on view N.** The meter, the statistic, the 0.005 pp boundary rule, the 50M rule
  (CONTINUE iff the estimate on N > +0.005 pp, else STOPPED-AT-50M with no class) and the 100M classes (WIN / LOSS /
  NO-USEFUL-GAIN / UNDECIDED; INCONCLUSIVE by B1 §10's rules) are exactly those of Q6-N's §3, which restates the
  frame: 9 cells a look, the floor 1.76² on each cell's variance, a half-width never below 3.06 pp.
- **What each outcome does** is FRAME §5, unchanged, with card §3's reading: **a tie keeps ten epochs.** A
  NO-USEFUL-GAIN, an UNDECIDED or a STOPPED-AT-50M leaves the recipe at ten passes. This cell CANNOT adopt five
  epochs for their speed alone: showing "no worse within 2 pp" needs about ten runs an arm (FRAME §8). A WIN is a
  CANDIDATE, never an adoption; `designs/production_config.json` does not move on any outcome of this cell.
- **The look files** are Q6-N's §3: the banked bytes of the 50M look are fixed in
  [`../q6n_policy_readout_fresh_2026-11-20/b1_look_50M.json`](../q6n_policy_readout_fresh_2026-11-20/b1_look_50M.json)
  (sha256 `925bfb32…dedc868`), the kit's `--b1-look`; B1's 100M `look.json` is banked the same way before any 100M
  look. Retention of B1's run dirs already holds (`../../models_retention_policy.md`), whole run dirs, so every
  banked checkpoint the matched-GPU-time cross below could name is covered.
- **REPORTED beside each look, never deciding** (card §2): each run's realized dose (`python -m main.dose <runs>
  --md`) and the arm-to-control ratio of the cell means of `dose_rate`; the LR rung histogram (`6e-4 × 1.2^k` for
  the arm, `3e-4 × 1.2^k` for C); the last-epoch KL; clip fraction; entropy by 5M window (an outcome, never a dose:
  FRAME §8); value loss; every touch of the LR floor or ceiling; the step-time ratio of the real runs.
- **The label MOVEMENT-UNMATCHED** is printed beside the outcome if the arm cell's last-epoch KL lies outside the
  controller's band. Made deterministic here (the registrar's wording of card §2's rule): for each fresh run, the
  median of `train/approx_kl` over its update rows with 10,000,000 ≤ step ≤ 50,000,000, rows a resume superseded
  excluded (`main.b1.lr_report.read_series`'s rule); the cell's figure is the median of the three; the label is
  printed unless 0.005 + 1e-6 < figure < 0.02 − 1e-6. **It voids nothing and changes no class.**
- **REPORTED at 100M: the matched-GPU-time cross** (card §3). The arm's 100M files against the banked runs'
  `checkpoint_<M>_steps.zip`, M the first 1,000,000 multiple at or after ρ_run × 100M (ρ_run = the arm cell's GPU
  time a step over C's, from the real runs over 10M to 50M; rounding toward the control is conservative to the arm);
  9 more cells, ≈ 0.33 GPU-h. **It is OWED A BUILD** (§ 10 F2): the kit plays only a look's registered files. The
  exact meter of ρ_run is registered with that build by a dated amendment, before Q5's 100M look and before ρ_run is
  computed. If the build does not exist at the 100M look, the cross is recorded as owed; the class does not wait.

## 4. Seeds, collisions, a lost run

- Seeds 5201, 5202, 5203; spare 5209; the pilot's 5298. The scan of 2026-11-20 (693 sidecars, 4,173 directories)
  finds no hit for any of them or their names ([`seed_scan_q5n.txt`](seed_scan_q5n.txt)); it is re-run on the day of
  the first launch. Seed 5201 is also a TEAM-SPLIT seed in `x35_breakers_stage_2026-10-17`: not a training seed.
- **A lost run** follows B1 §10's closed list (files lost or unreadable; launched with a wrong argv): replaced from
  step 0 with seed 5209 (`rb_q5n_s5209`) by a dated amendment that edits `fresh_arm_q5n.json`, written with the
  cause before any look that would contain the run. ONE replacement exists. Anything else is "lost to its lever":
  not replaced, that look INCONCLUSIVE; a second loss is INCONCLUSIVE too.

## 5. Power (the σ amendment is Q6-N's §5; the same tables apply, the banked cell being the same)

[`power_c50m.py`](../q6n_policy_readout_fresh_2026-11-20/power_c50m.py) and its
[log](../q6n_policy_readout_fresh_2026-11-20/power_c50m.log): cell C's own spread at 50M is **σ_C ≈ 0.36 pp** on 2 df
(95 % upper bound 2.31), every measured variance far under the 1.76² floor. B1's wide 50M intervals came from the
WB arm, not from C. So the floor sets the banked side, the half-width is at least 3.06 pp, and the fresh cell's own
spread is the unknown. For this card the fresh cell is C's network under another update schedule, so C's measured
spread is the nearest evidence of it; **UNVERIFIED** all the same (2 df, another recipe, and 50M is not 100M).

| IF the 100M spreads are (σ fresh / σ banked) | half-width | 0 | +2 | +4 | +6 pp | −4 pp reads LOSS |
|---|---|---|---|---|---|---|
| 0.36 / 0.36 (both as tight as C measured) | 3.06 | NO-USEFUL-GAIN 0.99 | WIN **0.002**, UNDECIDED 0.996 | WIN **0.99** | 1.00 | 0.995 |
| 1.76 / 0.36 | 3.06 | WIN 0.002, NO-USEFUL-GAIN 0.73 | WIN **0.13**, UNDECIDED 0.75 | WIN **0.72** | 0.98 | 0.73 |
| 1.76 / 1.76 (the frame's) | 3.28 | WIN 0.011, NO-USEFUL-GAIN 0.60 | WIN **0.16**, UNDECIDED 0.68 | WIN **0.61** | 0.93 | 0.61 |
| 2.5 / 2.5 (the frame's) | 4.22 | WIN 0.027, NO-USEFUL-GAIN 0.405 | WIN **0.15**, UNDECIDED 0.70 | WIN **0.43** | 0.75 | 0.43 |
| 3.1 / 3.1 (the frame's) | 5.09 | WIN 0.036, NO-USEFUL-GAIN 0.30 | WIN **0.14**, UNDECIDED 0.73 | WIN **0.34** | 0.60 | 0.34 |

The 50M rule: an arm with no effect continues half the time; one that is +2 pp ahead at 50M continues 0.78 to 1.00;
one that is 2 pp behind continues 0.00 to 0.22 (the full table is Q6-N's §5).

**In words. The power for a realistic lever is poor, and the rule is not changed for it.** A +2 pp gain reads WIN
between never and one time in six in every row; it reads UNDECIDED about seven times in ten or more, and UNDECIDED
keeps ten epochs. A +4 pp gain reads WIN between one time in three and nearly always. What this cell is good for is
the LARGE HARM the one earlier fresh read saw: a harm of O8's size (−10.6 pp) reads LOSS more often than a −8 pp
harm does, and a −8 pp harm reads LOSS at least 0.92 of the time in every row of the log but the widest (0.81; the
rule is symmetric, so this is the log's +8 pp WIN column). Q6-N reads
against the same three banked runs: the two verdicts are correlated and neither is ranked against the other.

## 6. The pre-launch record on current main (2026-11-20, main `66edebfff`; FRAME §6 item 2 is re-run on launch day)

All at nice 19 on the agent cores with `CUDA_VISIBLE_DEVICES=""`, no card.

| check | result | record |
|---|---|---|
| the kit's argvs | the first line is the card's recorded argv token for token | [`describe_q5n.txt`](describe_q5n.txt) |
| `python -m main.checkargs --argv …` | exit 0; `parser : the PINNED commit 15136b61's`; 0 unrecognized; the RECIPE surface lists `n_epochs 5`, `lr 0.0006` and `ent_coef 0.0125` as TYPED and nothing else; the ARCH surface is production's. Identical to the 11-17 draft but for the timing | [`checkargs_q5n.txt`](checkargs_q5n.txt) |
| `python -m main.launcher --dry-run …` | exit 3 with exactly the two refusals an agent shell owes (`cuda devices`, `inherited nice`); every other preflight passes; nothing created | [`dryrun_q5n.txt`](dryrun_q5n.txt) |
| the resolved-config diff, in a clean detached checkout of the pin (`main.b1.fresh lever-check --pin-tree`) | exit 0, `ok true`, no problem: against C at 50M and at 100M, for all three seeds, the keys that differ are exactly `lr`, `n_epochs`, `seed`, `run_name`. **Exactly one lever** | [`lever_check_q5n.json`](lever_check_q5n.json) |
| the seed scan | no hit | [`seed_scan_q5n.txt`](seed_scan_q5n.txt) |

**The pin does not move.**

## 7. The card check: the FIRST item of Q5's GPU window (≤ 3 GPU-h; criteria registered now, NOT run)

The draft owed none ("the arm's network is the control's"); R3 asks for one, and the draft itself marked UNVERIFIED
that no five-epoch fresh run has run at this pin. Run by the Training session under the lease, from the main
checkout at nice ≤ 10 with the card visible, into an EMPTY scratch `GEN3AI_MODELS_DIR` (never a comparator, never in
`models/`). The seed scan for 5298 is re-run first; then the dry run of the same argv must pass every preflight.
This file is not a window chain: the session writes its own.

```
python -m main.launcher --restart-interval-hours 6 --pin-commit 15136b61634b667bb9da5ccc6647ff9265238829 --arch production --ent-coef 0.0125 --behaviour-check warn --n-epochs 5 --lr 6e-4 --checkpoint-every-steps 1000000 --steps 2500000 --seed 5298 --device cuda --log-level periodic --run-name q5n_pilot_s5298
```

The pilot runs to 2,500,000 steps: past the update-10 canary (≈ 0.99M in B1's pilot), the first checkpoint (1M), the
first evaluation (2M) and the controller's first possible moves (one a 7-rollout cooldown, ≈ 0.69M steps). It does
not wait for a pool promotion (4M in B1's pilots): the network and its promotion path are C's, which B1's pilot and
runs exercised.

**PASS iff every one of 1 to 5 holds:**

1. the bf16 R1 compile completes and its compile-parity check passes (the child log's parity line);
2. the inference T2 CUDA-graph capture succeeds (the child log's `T2 up … backend graph on cuda`);
3. the trainer's own startup fit check passes, and `update_fit.json` records `oom` null and a headroom of at least
   1,024.5 MiB (the 1,024 bar with its 0.5 MiB resolution excluded, root rule 8). Reference: B1's C pilot, demand
   7,060.0 MiB, headroom 3,710.5 MiB;
4. no parity-gate FATAL and no non-finite FATAL in any generation of the child's full log through the end of the
   pilot, and the update-10 canary's verdict is PASS;
5. the lever is the run's: the pilot's `metadata.json` records `n_epochs` 5 and `lr` 0.0006 at the pin.

**FAIL** = any of 1 to 5 does not hold: the card HOLDS, no Q5-N run launches, and the failure comes back to the
orchestrator as a build (a fix after the pin sends the arm to FRAME §7). **INCOMPLETE** = the pilot did not reach
2,500,000 steps for a reason outside 1 to 5: neither PASS nor FAIL; reported, then re-run or held by the
orchestrator. Elapsed time is never a criterion.

**RECORDED, deciding nothing:**

6. **ρ_Q5**, by the same formula as Q6-N's §7 item 5 (`pilot_rho.py` in Q6-N's folder), over the rollout rows with
   300,000 ≤ step < 2,000,000 (`--hi 2000000`). The reference on that window is **517.86 s per 1M steps** (B1's C
   pilot, 18 rows, update share 0.825; `pilot_rho_reference_c.txt` there). It prices the cell (§ 8);
7. the controller's record: the LR at every row, each move with its step, the last-epoch KL, and whether the shared
   compile cache hit (expected: the graph is C's; **UNVERIFIED**).

The record of the check is appended to this file as a dated record before the first Q5-N launch.

**Cost:** 2.5M steps at 518 s per 1M is 0.36 GPU-h × ρ_Q5, plus one evaluation cycle and a start (0.08 h warm, 0.33 h
cold in B1's pilot): about 0.4 to 0.7 GPU-h. The bound is 3 GPU-h.

## 8. The order and the cost

**Order (R4):** after Q6-N's window. Within the window: § 7's check; on PASS the three runs to 50M in seed order; the
50M look; on CONTINUE the three resumes to 100M; the 100M look; the matched-GPU-time cross if built.

| item | GPU-h at ρ_Q5 = 1 | at the expected ρ_Q5, 0.59 to 0.86 | provenance |
|---|---|---|---|
| the card check | ≈ 0.4 to 0.7 (bound 3) | the same | § 7 |
| the cell to 50M | ≈ 28.6 | ≈ 16.9 to 24.7 | FRAME §4 × ρ_Q5 |
| the cell from 50M to 100M, only on CONTINUE | ≈ 30.1 | ≈ 17.8 to 26.0 | the same |
| each look; the matched-GPU-time cross | ≈ 0.33 each, plus an engine start | the same | FRAME §4; **UNVERIFIED** |
| **total** | ≈ 30 stopped at 50M; ≈ 61 to a 100M class; **≈ 45 expected with no true effect** | ≈ 18 to 26; ≈ 36 to 52; **≈ 27 to 39** | |

**ρ_Q5 is UNVERIFIED** until § 7. The range is two bounds, not a forecast: 0.59 if the update's time halves with the
passes (the update was 0.825 of a step's time in B1's C pilot: 0.825 / 2 + 0.175), and 0.864 from the one measured
figure (ledger `L21451`: 13.59 % less GPU time a step at forks, an earlier era and learner).

## 9. What this cell cannot say (FRAME §8, plus this card's)

- It reads at 100M from scratch, not at the plateau; a STOPPED-AT-50M, NO-USEFUL-GAIN or UNDECIDED retires nothing.
- It cannot adopt five epochs for speed: "no worse" is beyond three runs against three (§ 3, § 5).
- It reads X27's form only (half the passes at twice the starting LR under the live controller) at width 128 under
  B1's recipe. `--n-epochs 5` alone, a frozen LR, X28's controller and width 256 are other experiments.
- The controller may move the two cells' realized doses apart; that is REPORTED (§ 3) and, outside the band,
  labelled. The earlier fresh read (the sizing verdict's O8, −10.6 pp at ≈ 60 % of the control's dose) was confounded
  in exactly this way; a LOSS here with the label printed is a LOSS of the lever AS SPECIFIED, and the record says
  the movement did not match.
- Realized entropy is an outcome (FRAME §8): fewer passes can sharpen the policy less per update; reported, not matched.

## 10. FINDINGS and the registrar's choices (each flagged for the orchestrator; all amendable while still blind)

- **F1. A card check is registered where the draft owed none** (R3): § 7, a 2.5M pilot, ≈ 0.4 to 0.7 GPU-h. Its
  criterion 5 and its recorded items 6 and 7 are the registrar's.
- **F2. Three REPORTED items have no built tool.** `python -m main.dose` reads the realized dose. But
  `main.b1.lr_report` accepts B1's registered run names only, so the KL median, the rung histogram and the
  MOVEMENT-UNMATCHED label have no command for `rb_q5n_*`; and the kit cannot play the matched-GPU-time cross (§ 3).
  Both are builds owed before Q5's 50M look and 100M look respectively. Neither decides; a look is valid without
  them and its record then says OWED.
- **F3. The label's meter is the registrar's wording** (§ 3): the card said "the arm cell's median last-epoch KL";
  the per-run median, the median of three and the 1e-6 edge are fixed here so that no judgment is left.
- **F4. Power for a +2 pp gain is 0.00 to 0.16 in every scenario** (§ 5): this cell can find a large harm or a gain
  of +4 pp and more, and nothing between. Not a reason to change the rule.
- **F5. The kit has no replacement path for a lost fresh run** (§ 4), and **nothing in code holds the fresh cell's
  own look checkpoints** (`tiers.BANKED_B1_PREFIXES` names B1's runs only): no thinning of `rb_q5n_*` before this
  read closes. **UNVERIFIED:** that a grooming plan would refuse them.
- **F6. UNVERIFIED and priced as such:** ρ_Q5, the look's cost, σ of the fresh cell, σ at 100M, and that five
  epochs change nothing in the compiled update but its pass count.
