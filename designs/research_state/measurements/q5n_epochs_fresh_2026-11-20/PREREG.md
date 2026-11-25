# Q5-N — five PPO epochs at matched dose, one fresh cell against BOUNDARY-1's banked cell C: PRE-REGISTRATION (2026-11-20)

**READ 2026-11-25 (§13): STOPPED-AT-50M.** The three fresh runs trained to 50M on the launch pin (each launcher rc 0,
each `check-run --step 50000000` PASS) and the 50M look was played once on the card, 45,000 games. On N, **fresh minus
banked = −1.54 pp, 90 % interval [−4.99, +1.91]** (4 df; the banked side floored at 1.76, the fresh side's own spread
above the floor; no bound near a threshold). CONTINUE needed an estimate above +0.005 pp. It is not HARM-AT-50M (the
upper end is above −2.0 pp). Per §3 and FRAME §5 it is "Not continued. Reported with its interval; no class", and it
is never "no effect": the 100M leg is not run, the matched-GPU-time cross (a 100M item) is not played, and the recipe
stays at ten epochs ("a tie keeps ten epochs"). The status paragraph below is the registration's own, as it stood
before any Q5 event.

**Status: REGISTERED 2026-11-20, FULL tier, blind: before any Q5 training step, any Q5 game, any look and the card
check. NOTHING HAS TRAINED and this file authorises no launch.** The GPU window is the orchestrator's and comes AFTER
Q6-N's ([`../q6n_policy_readout_fresh_2026-11-20/PREREG.md`](../q6n_policy_readout_fresh_2026-11-20/PREREG.md), its
ruling R4); its first item is § 7's card check. Nothing below changes after the first Q5-N run launches except by a
dated amendment written before the look it touches, saying why. **AMENDED 2026-11-22 (Amendment 1, § 11: the banked
cell's 100M files are those of B1's 100M survivor look; ρ_run's meter is registered with its build; no rule
moves), also blind.**

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

## 11. Amendments

### Amendment 1 (2026-11-22): the banked cell's 100M files; ρ_run's meter

Written 2026-11-22 on the orchestrator's instruction and ruling, before both looks and before every other Q5 event.
**Blind: no Q5-N run, pilot, card check, game or look exists** (no `rb_q5n_*` directory, no GPU window), **and
ρ_run has not been computed** (it needs the arm's runs). **It moves no rule**: no bar, class, view, panel, pair
count, seed, run name, argv, pin, guard or deciding statistic. Item 1 says WHICH files the 100M look reads for the
banked cell; item 2 fixes the meter § 3 left to "a dated amendment, before Q5's 100M look and before ρ_run is
computed". **What was read to write it:** the plan of B1's 100M survivor look (its `look.json`, which holds no
score), the `checkpoints` block of B1's committed record of that look, and the build of commit `ab0048861`. Known to
the writer and saying nothing of a Q5 run: B1's 100M outcome (W against C, `NO-CLASS`), and the meter's figures on
B1's own runs from that build's check (C over W, 0.7127).

1. **The banked cell's 100M files** are Q6-N's Amendment 1, item for item
   ([`../q6n_policy_readout_fresh_2026-11-20/PREREG.md`](../q6n_policy_readout_fresh_2026-11-20/PREREG.md) § 11):
   B1's 100M look was the SURVIVOR look of width, and the banked cell's files are that look's files for C:
   - `rb_b1c_s5101` at 100,000,127, sha256 `d833aab47f49a726753b60a35bb7d085fb70909e855b2bf9b46cef54c26db098`;
   - `rb_b1c_s5102` at 100,000,172, sha256 `32e1a8d0587f2463600a41107a1b431990de53b085c320312ae28e2642a43148`;
   - `rb_b1c_s5110` at 100,000,025, sha256 `21868790990e0ded79e710a2d747fa2f303a9047b8d3c41a1557a14848fa159b`.
   The look's `look.json` is banked as
   [`../q6n_policy_readout_fresh_2026-11-20/b1_look_100M.json`](../q6n_policy_readout_fresh_2026-11-20/b1_look_100M.json)
   (sha256 `cd074fb09fd2e9d1ae0f119b64b270ee1721f26945b8f6013667f854e5bd826c`), the kit's `--b1-look` at the 100M
   look. The kit that reads it is `python -m main.b1.fresh` at `d41989b96`, under the three conditions of Q6-N's
   Amendment 1 item 2 (the rows for C exactly the banked cell's three runs; each step and sha256 the same on disk,
   in the look and in B1's record; B1's run guard with K9(b) at the step). This closes § 3's "B1's 100M `look.json`
   is banked the same way before any 100M look". The 50M look is untouched.
2. **ρ_run's meter** (the orchestrator's ruling; the text below is registered as ruled):

   > ρ_run's meter (§3, F2), registered with its build `ab0048861`: per run, `main.b1.lr_report.step_time`. That
   > is §7's card-check formula, (median `lifecycle/update_wall_s` + median `rollout/collect_ms`/1000) / median
   > Δsteps × 10⁶, over the rollout rows with 10,000,000 ≤ step ≤ 50,000,000 after a resume's superseded rows are
   > removed. The cell's figure is the mean of its three runs, and ρ_run = the arm's figure / C's, in exact
   > arithmetic. M = ⌈ρ_run × 100⌉ × 1,000,000. The banked file is each C run's first checkpoint at or after M,
   > within one interval. Not played if M ≥ 100,000,000. Command: `python -m main.b1.fresh plan-gpu-time`. The
   > same meter is the REPORTED step-time ratio at both looks.

3. **Why this meter.** It is the card check's own formula (§ 7 item 6) on the window § 3 already names, so it adds
   no degree of freedom; exact arithmetic leaves no rounding at a 1,000,000 boundary (root rule 8). **The cell's
   figure is the MEAN of three** because that is this registration's convention for an arm-to-control ratio of a
   training quantity: § 3 reports the dose as "the arm-to-control ratio of the cell means of `dose_rate`", and the
   build reports every cell meter as the mean of its runs. § 3's one median of three is the MOVEMENT-UNMATCHED
   label's, which is the card's own wording for that label ("the arm cell's median last-epoch KL"); it is not
   moved, and it is not a ratio. § 3 itself named no aggregate for ρ_run ("the arm cell's GPU time a step over
   C's"): this item fixes it.
4. **What the build does at the edges** (a description of `ab0048861`, registering nothing more). No run is
   dropped: the meter refuses when the rollout rows of any of the six runs do not span the window, and the plan
   then refuses by name; that refusal is recorded beside the look. The plan needs the card's own 100M look's
   `look.json` (`--fresh-look`), so it is made only after that look. The cross stays REPORTED; the class never
   waits on it (§ 3).
5. **What this closes.** § 3's owed amendment and § 10 F2: the rung, KL and label report and every REPORTED meter
   (`python -m main.b1.lr_report --fresh-arm`), the cross's plan (`python -m main.b1.fresh plan-gpu-time`) and its
   meter are built and registered. Nothing of this registration is owed a build.

## 12. The card check's record (2026-11-24, § 7; FRAME §6 item 4), written before the first Q5-N launch

Run by the Training session under the lease, from the main checkout (HEAD `15d96fcdd`) with the card visible, into
the empty scratch `GEN3AI_MODELS_DIR` `~/.cache/gen3ai/q5n_2026-11-24/models` (the pilot is never a comparator and
never entered `models/`). The seed scan for 5201, 5202, 5203, 5209 and 5298 was re-run first (729 sidecars over
`models/`, the archive and the scratch directory): no hit. The dry run of § 7's argv exited 0 with every preflight
passed, 52.79 GiB free against 10.19 GiB required. The pilot `q5n_pilot_s5298` ran 09:31:38 to 10:05:41 PST (34 min,
about 0.57 GPU-h, inside the 3 GPU-h bound) to 2,561,280 steps, launcher rc 0, one child-log generation, launcher end
"Training complete — all steps done".

**Verdict: PASS** (criteria 1 to 5 all hold).

1. Compile parity: `[CompileRegions] parity PASS (bf16 trunk: the bf16 rule) — R1 [fresh weights] bf16 rule: drift loss 1.01e-03 <= 0.014, grad 7.40e-03 <= 0.68`.
2. T2 capture: `[09:47:00] [RUST ENV] T2 up in 895.0s ... backend graph on cuda`.
3. Startup fit (`update_fit.json`, sha256 `479d7728feda9827…`): `oom` null, headroom **3,710.5 MiB** (bar 1,024.5),
   demand 7,060.0 MiB; both equal B1's C pilot's (demand 7,060.0, headroom 3,710.5).
4. No parity-gate FATAL and no non-finite FATAL in the child's full log (one generation); the update-10 canary's
   verdict is pass.
5. The lever is the run's: `n_epochs` 5 and `lr` 0.0006 recorded, one pin entry, `15136b61634b667bb9da5ccc6647ff9265238829`.

**Recorded, deciding nothing:**

6. **ρ_Q5 = 0.5914** (§ 7 item 6's formula, rollout rows with 300,000 ≤ step < 2,000,000): the pilot 306.26 s per 1M
   steps (18 rows, median update 21.359 s, collect 8.748 s, 98,304 steps a rollout, update share 0.709) against the
   registered 517.86 of B1's C pilot `rb_b1pinpilot_c_s9201` (18 rows, 42.0 s, 8.908 s, share 0.825).
7. The controller moved the LR up twice: 6e-4 → 7.2e-4 at 988,928 steps, then → 8.64e-4 at 1,775,104; no other move.
   The last-epoch KL (`train/approx_kl_epoch_4`) went 0.0044 at 202,752 to 0.0070 at 2,561,280 (range 0.0012 to
   0.0080 over the 25 rows before the end). The shared compile cache MISSED: `SHARED-NEW`, key `1bdf57369a43`, a new
   entry (cold), so § 7 item 7's expected hit did not happen.

**FINDINGS** (each recorded; none moves a rule):

- (a) ρ_Q5 sits at the bottom of § 8's expected 0.59 to 0.86: the update's time halved (21.36 s against 42.0 s), as
  five passes against ten predict.
- (b) Step 2,561,280 carries two LR rows and two KL rows (0.0070 and 0.0221), probably the final update logged twice.
  It decides nothing here, but `main.b1.lr_report`'s MOVEMENT-UNMATCHED median at 50M reads these rows.
- (c) The expected compile-cache hit missed (item 7).
- (d) Within a run the pace rises about 40 % from 2M to 50M in this recipe (Q6-N's and B1's C runs both showed it).
  B1's measured 9.55 h a run already carries it, which is why the price below uses 9.55 h and not the pilot's pace.

**The orchestrator's price, written before the first run (§ 7 item 6, § 8):** ρ = 0.591 is ACCEPTED. The cell to
50M is ≈ 3 × 9.55 × 0.591 ≈ 16.9 GPU-h, plus one cold compile at the first 6 h restart (the pin's compile-key issue,
fixed on main in `4d327c5a1` after the pin; ≈ 0.3 h) and the 50M look (≈ 0.6 h): ≈ 17.8 GPU-h. That is not too high
for this card's question. The window's scope at its GO is the three runs `rb_q5n_s5201`–`s5203` to 50M and the 50M
look; any later leg is ruled at that look. The record of the check is `~/.claude/jobs/x35/q5n/card_check.json`
(sha256 `f92bb9603eced552…`), with `pilot.dryrun.txt` and `prelaunch_20261124T093133/seed_scan.txt` beside it. No
rule, bar, class, seed, argv, pin or statistic moves.

## 13. The READ: the 50M look (2026-11-25)

Written after the 50M look was played and read, once (§3). Nothing in §1 to §12 is changed by it. Every number below
is in [`record/look_50M/`](record/look_50M/) ([`SHA256SUMS`](record/look_50M/SHA256SUMS)): byte copies of the look's
work directory `~/.cache/gen3ai/q5n_2026-11-24/look50_20261125T072808/` (`cells_plan.json`, `look.json`, `run.log`,
`result/verdict.json`, `result/inputs.json`, `result/team_bootstrap.json`), the play wrapper's log (`play.out`), the
REPORTED meters' report (`look50_lr_report.out` and `.json`, `python -m main.b1.lr_report --look 50M --fresh-arm`)
and its duplicate-row count (`look50_kl_dups.json`: a JSON object followed by one `SUMMARY:` text line, copied as
written), the window chain's Q5W, Q5P and Q5D status lines (`chain_status_q5w.txt`, from
`~/.claude/jobs/x35/status.txt`), the realised cost and pace (`cost_and_pace.txt`), and the eval ledger's per-batch
row files under `models/_ledger/`, listed by sha256 and size and not copied (`eval_ledger_files.txt`). No file is
renamed: no guard of this kit refuses a file named like a verdict or a GO marker in this directory. **Re-derived:**
`python -m main.b1.fresh recheck` on a copy of the record re-decides STOPPED-AT-50M from `inputs.json` alone, with
the same words, and rewrites `verdict.json` byte for byte; the estimate and the interval by hand (the mean of the
fresh rows − 50; SE √((4.7535 + 3.0976) / 3) = 1.6177; t₀.₉₅,₄ = 2.132) equal `verdict.json`'s; the six look files
re-hashed in `models/` equal the plan's (`af50a4bf…`, `a40f7665…`, `4b12f841…` for s5201 / s5202 / s5203;
`f706243d…`, `c46ddab8…`, `cbcccec7…` for C, as §3 fixed them). Read on the CPU from main `cc426ac42`'s code, no
GPU lease.

### 13.1 The runs (§2, §4)

The Q5-N window chain trained the three runs one at a time on the card under the Training session's lease and the
orchestrator's GO marker (`commit=aa87519ff…`, `card_check=PASS`, `rho=0.591`, `scope=50M`), after its pre-launch
checks PASS (FRAME §6 item 2: describe equal to the registered argvs, `checkargs` × 3 exit 0, `lever-check` ok, the
seed scan no hit). Each argv is §2's, token for token.

| run | launcher | wall | restarts | look file (step) |
|---|---|---:|---|---:|
| `rb_q5n_s5201` | 11-24 11:23:33 → 18:06:28 PST | 6.72 h | one, warm | 50,000,311 |
| `rb_q5n_s5202` | 11-24 18:06:38 → 11-25 00:47:34 | 6.68 h | one, warm | 50,000,144 |
| `rb_q5n_s5203` | 11-25 00:47:44 → 07:28:03 | 6.67 h | one, warm | 50,000,092 |

Each launcher rc 0, the early check PASS at the first checkpoint (1,000,192), and `check-run --step 50000000` PASS at
the end. No run was lost or replaced (§4); the spare 5209 was not used.

### 13.2 The look (§3)

Planned at 07:28:27 PST with `--b1-look b1_look_50M.json` (sha256 `925bfb32c30f…`): 9 cells, the plan PASS. Played
07:28:27 to 07:53:13 PST on the card at the pin, from the detached pin checkout `~/dev/gen3ai-worktrees/b1_look_pin`,
with a fresh inductor cache; 45,000 games (home72 1,000 mirrored pairs a cell, H, F and P72 500 each), both sides
greedy. The `sidecar_audit --strict` and the six `checkargs` exited 0 first (REPORTED). Draws 0.5 % to 1.1 % of a
panel's games (the worst cell 1.7 %, F and P72).

**The decision (N decides; §3):**

| view | fresh − banked (pp) | 90 % interval (4 df) | fresh row means | banked column means |
|---|---:|---|---|---|
| **N** | **−1.54** | **[−4.99, +1.91]** | 50.81 / 48.05 / 46.51 | 47.56 / 48.45 / 49.35 |
| home72 | −2.57 | [−6.79, +1.65] | 50.66 / 46.73 / 44.91 | 46.10 / 47.99 / 48.20 |
| H (`heldout72_hpfix`) | −1.73 | [−5.02, +1.56] | 50.52 / 47.67 / 46.63 | 48.45 / 48.40 / 47.97 |
| F (`forum72`) | −0.29 | [−3.36, +2.77] | 51.20 / 49.73 / 48.18 | 49.00 / 49.45 / 50.67 |
| P72 | +0.47 | [−2.92, +3.86] | 51.18 / 50.72 / 49.52 | 49.63 / 48.90 / 52.88 |
| U | −0.52 | [−3.58, +2.55] | 50.97 / 49.37 / 48.11 | 49.03 / 48.92 / 50.51 |
| HF | −1.43 | [−4.91, +2.05] | 50.93 / 48.23 / 46.55 | 47.55 / 48.72 / 49.43 |

On N the SE is 1.618 pp and the half-width 3.45 pp, above the registered minimum of 3.06: the banked side's variance
(raw 0.800) is floored to 3.0976, and **the fresh side's is NOT floored** (raw 4.754 of the three run means, an SD
of 2.18 pp against the floor's 1.76). The same holds on home72 (raw 8.639, SE 1.978), H (4.045) and HF (4.890); on
F and U both sides are floored (SE 1.437); on P72 the fresh side is floored and the banked side is not (raw 4.495,
SE 1.591). Cells on N (fresh row × banked column): s5201 49.82 / 50.68 / 51.95; s5202 47.13 / 47.99 / 49.03; s5203
45.75 / 46.70 / 47.08.

**Outcome: STOPPED-AT-50M** (`verdict.json`: "STOPPED-AT-50M on N: fresh - banked = -1.54 pp [-4.99, +1.91] (90 %,
4 df)"; `at_boundary` empty, no label, no reason). CONTINUE needed an estimate above +0.005 pp. **HARM-AT-50M is not
written beside it:** the upper end, +1.91, is not below −2.0 pp. **REPORTED, deciding nothing:** the team-cluster
bootstrap (10,000 draws) adds 1.6 % to the variance on N and gives [−5.02, +1.93], also STOPPED-AT-50M (`class_moves`
false); every view's interval spans zero by both methods.

**REPORTED beside the look, never deciding (§3; `look50_lr_report.out`):**

- **The label MOVEMENT-UNMATCHED is NOT printed.** The KL median over 10M ≤ step ≤ 50M (406 rows a run) is 0.016734 /
  0.016520 / 0.015894 for s5201 / s5202 / s5203; the cell's figure, the median of three, is 0.016520, inside
  0.005 + 1e-6 < figure < 0.02 − 1e-6. C's three read 0.017498 / 0.017687 / 0.017274.
- **Dose: matched.** `dose_rate` to each look checkpoint is 4.239e-08 / 3.532e-08 / 3.532e-08 for the fresh runs and
  3.532e-08 / 4.239e-08 / 3.532e-08 for C's; the cell means are both 3.768e-08, **fresh / banked = 1**. At the look
  every fresh run's LR is 3.4722e-04 (rung −3 of `6e-4 × 1.2^k`) and every C run's 1.7361e-04 (rung −3 of
  `3e-4 × 1.2^k`): twice the rate at half the passes.
- **Rungs** (rows to the look, the cell's three runs summed): fresh −3: 750, −2: 280, −1: 168, 0: 104, +1: 84,
  +2: 135, off the ladder 0; C −3: 691, −2: 271, −1: 207, 0: 97, +1: 153, +2: 102, off 0. No run touched the LR
  floor (1e-5) or its ceiling (1.2e-3 fresh, 6e-4 C).
- **Clip fraction** over 10M to 50M (cell mean of the runs' medians): fresh 0.1228, C 0.1445; of the last epoch
  0.1704 against 0.1922. **Value loss:** 0.01643 against 0.01766.
- **Entropy, an outcome and not a dose** (FRAME §8): 0.5121 / 0.5135 / 0.4941 for the fresh runs over 10M to 50M
  (cell 0.5066) against C's 0.5796 / 0.5818 / 0.5728 (cell 0.5781); the fresh cell is the lower in every 5M window
  from 5M on (0.498 against 0.570 in the last). The five-epoch runs ended sharper, not blunter; §9's sentence named
  the other direction as possible. Reported, not matched.
- **The step-time ratio (§11 item 2's meter):** 471.30 / 469.81 / 469.62 s per 1M steps for the fresh runs over
  10M ≤ step ≤ 50M (cell mean 470.2) against C's 676.96 / 673.93 / 680.20 (677.0): **ρ_run = 0.6946**
  (`48918586570298457867130348925/70430423640550054812457869312`, exact). The card check's ρ_Q5, on 0.3M to 2M of
  the pilot, was 0.5914 (§12; FINDING Q5N-G2 below).

### 13.3 What it licenses, and what it does not

In the registration's words (§3, FRAME §5): a STOPPED-AT-50M arm is "Not continued. Reported with its interval; no
class"; "**a tie keeps ten epochs.** A NO-USEFUL-GAIN, an UNDECIDED or a STOPPED-AT-50M leaves the recipe at ten
passes. This cell CANNOT adopt five epochs for their speed alone"; and "`designs/production_config.json` does not
move on any outcome of this cell". FRAME §4: it "carries NO class: it is never a LOSS, never \"no effect\", never
retired". §9: "It reads at 100M from scratch, not at the plateau; a STOPPED-AT-50M, NO-USEFUL-GAIN or UNDECIDED
retires nothing", and "It cannot adopt five epochs for speed: \"no worse\" is beyond three runs against three".

So: five epochs at twice the starting LR is not continued to 100M in this cell, not a candidate for the next fresh
lineage, not retired and not adopted; the recipe stays at ten passes, and no recipe, `designs/production_config.json`
or baseline moves. The interval holds both a 5.0 pp loss and a 1.9 pp gain at 50M; it says nothing about 100M, the
plateau, `--n-epochs 5` alone, a frozen LR, another controller or width 256 (§9). The runs took 0.69 of C's GPU time
a step at the matched dose, and that is not a reason to adopt them: this cell cannot show "no worse" (§3).

**The 100M leg is NOT run, and the matched-GPU-time cross is NOT played.** The window chain ended at the look
("phase 2 (100M) is the orchestrator's ruling; nothing launches from here"), and the phase-2 chain, which waited for
a `50M look: CONTINUE` line, saw none and ended without launching anything (Q5P, 07:54:06 PST; its first copy had
been stopped and re-armed by the Training session at 06:40 with nothing launched). No `GO_Q5N_100M` marker was
written. The matched-GPU-time cross is §3's REPORTED item "at 100M": it reads the arm's 100M files, which do not
exist, so it is neither planned nor owed.

### 13.4 The cost against the price (§8, §12)

**Realised: ≈ 20.07 GPU-h of training + 0.41 GPU-h for the look = ≈ 20.5 GPU-h** (plus §12's card check, about
0.57 GPU-h). Method: each launcher's START → END in the chain's lines, which includes every bring-up and restart
(6.715 + 6.682 + 6.672 = 20.069 h), and the look's play start → DONE (1,486 s of wall; the chain printed 1485 s)
(`cost_and_pace.txt`). **Price** (§12, accepted before the first run): ≈ 16.9 GPU-h to 50M at ρ 0.591, plus one cold
compile at the first restart (≈ 0.3 h) and the look (≈ 0.6 h), ≈ 17.8 in all. **The window ran ≈ 2.7 GPU-h (15 %)
over the accepted price and inside §8's registered range** for the cell to 50M (≈ 16.9 to 24.7 GPU-h at ρ 0.59 to
0.86): training alone is 0.70 of B1's 3 × 9.55 h. The orchestrator ruled on 2026-11-24, while run 1 was training,
that the window is not re-priced because it stays inside that range (the ruling as the orchestrator stated it to
this record's writer; no file carrying it was found). Why the price was low is FINDING Q5N-G2.

### 13.5 FINDINGs

- **Q5N-G1. Every run's last step carries TWO update rows, and `main.b1.lr_report` keeps the last one silently.**
  `look50_kl_dups.json` (the Training session's reporter, Q5D): each fresh run has one duplicated step, its last
  (50,052,231 / 50,055,123 / 50,055,268), and each C run two (its last row of the 50M leg and of the 100M leg). The
  two `train/approx_kl` values differ (s5201 0.0182 then 0.0239; s5202 0.0182 then 0.0212; s5203 0.0194 then 0.0173)
  and the two LR values are equal. This read timed them in the event files: the two rows are 21.3 to 21.4 s apart
  in the fresh runs and 42.1 to 43.5 s apart in C's at 50M, one update's wall in each cell, and 255 of the step's
  331 tags are doubled. So it reads as a second UPDATE on the last rollout at the end of a launch, not one row
  written twice (**UNVERIFIED:** the trainer's code at the pin was not read). **What it touches here: nothing that
  decides or labels.** No duplicated step lies in 10M ≤ step ≤ 50M, so the MOVEMENT-UNMATCHED figure never reads
  one; the rung histogram's last row is the duplicated step, where both LR rows are equal; the look files
  (`checkpoint_50000311_steps.zip` and its like) were saved before it; and it is the same in both cells. It closes
  §12's FINDING (b) as a property of every run at this pin. **Not fixed in this record** (the kit and the trainer
  are outside it): `lr_report.load_tags` should refuse or report a duplicated step, and a `final_model.zip` at this
  pin holds one update more than the run's last logged step count implies, in C's runs as in these.
- **Q5N-G2. The accepted price was 15 % low, because ρ was measured where a step is mostly update.** The update's
  wall is flat within a run (21.3 to 21.4 s a rollout in the pilot and over 10M to 50M; C's 42.0 to 42.1), while the
  collect's wall grows the same way in both cells (≈ 8.7 s a rollout at 0.3M to 2M, ≈ 24.9 s over 10M to 50M; C's
  8.9 → 24.5). Halving the update therefore saves a fixed ≈ 21 s a rollout, and the ratio climbs as the collect
  grows: 0.591 at 0.3M to 2M, 0.695 over 10M to 50M, 0.70 for the whole run against B1's 9.55 h. Checkpoint to
  checkpoint the fresh runs went 315 s per 1M steps at 1M → 2M to 499–511 at 49M → 50M (+60 %); C's 524–549 to
  704–760 (+35 %) (`cost_and_pace.txt`). §12's price multiplied B1's whole-run 9.55 h by the early ratio. A 100M
  leg or a later five-epoch card is priced with ρ_run 0.695, not 0.591. Two causes the orchestrator's brief named
  were checked: "the update share rising faster than C's" is the same fact seen from the other side (the update's
  share FALLS, 0.709 → 0.461, C's 0.825 → 0.632); "overlapping snapshot-ladder promotions at 24M and 26M in run 1"
  does not show in the pace (s5201's intervals ending at 24M and 26M took 475 and 458 s, s5202's 476 and 459,
  s5203's 472 and 456) and was not otherwise re-derived.
- **Q5N-G3. The cold compile the price carried did not happen.** Every launch hit the shared compile cache warm
  (`SHARED-HIT`, key `1bdf57369a43`, the card-check pilot's entry; T2 up in 136.5 to 139.9 s), and every 6 h restart
  too (key `70105b399543`, an entry from 2026-11-12; T2 up in 125.6 to 129.9 s). The restart's interval (46M → 47M)
  cost ≈ 300 s more than its neighbours in each run. ≈ 0.3 GPU-h priced and not spent; the overrun of G2 is net of it.
- **Q5N-G4. The fresh cell's seeds spread wider than the floor.** The three run means on N are 50.81 / 48.05 / 46.51
  (SD 2.18 pp; 2.94 on home72), against C's 0.89 under the same nine cells and Q6-N's fresh cell's 0.52. s5201 is at
  or above 50 against two of C's three runs and s5203 below 47.1 against all three. One estimate on 2 df: it is not
  evidence that five epochs widen the seed spread, and §5's power tables assumed the floor on both sides, so this
  look had less power than they state (half-width 3.45 against 3.06). The rule is unchanged by it.
- **Q5N-G5. The retention hold on `rb_q5n_*` is RELEASED by this record's commit.**
  `models_retention_policy.md`'s rule ("the commit that RECORDS the card's closing read (its 100M class, or
  STOPPED-AT-50M) moves the prefix") applies as written: the 100M leg is not run, so no later look reads these runs
  (§10 F5's "before this read closes"). The commit moves `rb_q5n_` from `tiers.OPEN_FRESH_CELLS` to
  `tiers.RELEASED_FRESH_CELLS` with the ledger heading. No card is open after it, so the hold's tests open a
  synthetic card. Release only returns the runs to the ordinary policies; nothing under `models/` was touched, and
  deleting run data stays the owner's.
- **Q5N-G6. The chain's lr-report line reads "label;" with nothing after the word.** The report's `label` is null,
  which is "no label" (`look50_lr_report.json`; the text report's line ends "yes, no label"). The chain's
  wording is the Training session's; it is not changed by this record.
