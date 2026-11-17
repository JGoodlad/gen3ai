# Lever card Q5: fewer PPO epochs at matched dose, one fresh cell (DRAFT, 2026-11-17)

**Status: DRAFT, not registered, nothing launched.** A card for [`FRAME.md`](FRAME.md): the lever, what "matched dose"
means under the learning-rate controller, the argvs at width 128 and 256 with their checks, and the CPU smokes.
Registering it is a FULL `PREREG.md` that adopts the frame. Queue proposal row Q5; backlog rows 18 (X27) and 19 (X28).

## 1. The lever and why it is still open

- **The lever: `--n-epochs 5 --lr 6e-4`**, X27's form: half the passes over each batch at twice the starting step
  size, so the total movement per update starts matched. Typed on top of the banked arm's argv; nothing else moves.
- **What is known.** At FORKS, five epochs at twice the LR was ADOPTED (ledger `L21451`, 2026-09-29, the learner
  battery's E5): untaught Δ vs control +0.75 pp [−1.21, +2.65] (non-inferior at −3.69) and 13.59 % [13.38, 13.71] less
  GPU time per step. That was at a FROZEN LR (`--fork-lr 5.6e-05`, the controller off), on the since-fixed miscompiled
  learner, in an earlier era. From SCRATCH, the one read (the sizing verdict's O8, 2026-10-02) put five epochs
  −10.6 pp untaught, but the controller had raised that arm's LR only to 5.18e-4 against the control's 4.32e-4, so it
  trained at ≈ 60 % of the control's dose (backlog row 18): fewer passes and less movement, confounded. The owner's
  prior (10-03): "10 is probably too much".
- **The question this cell answers:** from scratch, under the production controller, does an agent trained with
  five passes at matched movement end stronger, weaker or not detectably different at 100M than B1's ten-pass cell?

## 2. "Matched dose" under the controller, and how the controller is held

**The controller at the pin** (`src/agents/training/adaptive_lr_callback.py` `AdaptivePPOCallback`; the sibling
analysis `~/.cache/gen3ai/handoffs/b1_wide_lr_floor_2026-11-17.md`): once a rollout it folds `train/approx_kl` into an
EMA (α 0.20); EMA > 0.02 divides the LR by 1.2, EMA < 0.005 multiplies it by 1.2, else no move; a 7-rollout cooldown
after each move and at every (re)start; bounds `--min-lr` 1e-5 and `--max-lr`, which defaults to 2 × `--lr`. Target
0.01, the band factor 2.0 and the 1.2 step have no flag at the pin.

**What the controller holds.** At the pin `train/approx_kl` is the mean KL over the LAST epoch's micro-batches only
(`src/agents/training/instrumented_ppo/ppo.py`: `approx_kl_divs` is reset at each epoch's start and the logged value
is the last epoch's mean). That is the policy's distance from the behaviour policy at the end of the update: the
update's TOTAL movement. So the controller already holds total movement per update in its band, whatever the epoch
count. This is what makes X27's intent ("the total movement matches") a property the controller enforces.

**Definition, registered by this card.** *Matched dose* means: (a) the arm STARTS at the control's dose,
`lr × n_epochs × optimizer steps per epoch` = 6e-4 × 5 × 2 = 3e-4 × 10 × 2 (both arms take 2 optimizer steps an
epoch: 98,304 rollout rows at an effective batch of 65,536); and (b) from there the SAME controller acts on both
arms, with the same target, band, step, cooldown and floor, and the ceiling by the same rule (2 × the starting LR:
1.2e-3 for the arm, 6e-4 for the control). Both arms then approach the band from the same side. B1's control runs
started above the band and were cut until the EMA sat just under its 0.02 edge (median last-epoch KL 0.017 to 0.018
on all eight B1 runs read, sibling analysis), so the realized setpoint is the band's upper edge in both arms.

**How the controller is held: the SAME TARGET, not a fixed LR.** Three ways were weighed:

| form | what it does | verdict |
|---|---|---|
| **same target, `--lr 6e-4`** (this card) | starts matched; the controller then matches total movement per update in both arms | **chosen**: one lever against the banked cell, whose controller ran live |
| `--n-epochs 5` alone | starts at half the movement; the band is 4× wide (0.005 to 0.02), so the controller may never move it | rejected: the dose ends anywhere in a 4× range, the O8 confound again |
| a fixed LR | no fresh-run freeze at the pin (`--fork-lr` / `--fork-lr-freeze` are refused on a fresh run); `--min-lr 6e-4 --max-lr 6e-4` would clamp it with existing flags | rejected: the banked cell's controller ran live, so a frozen arm differs in two levers and must buy its own control (FRAME § 7, +28.6 GPU-h) |

**What is REPORTED, never deciding:** each run's realized dose (`python -m main.dose <runs> --md`; B1's C runs read
`dose_rate` 3.53e-8 / 4.24e-8 / 3.53e-8 at 50M, medians 1.736e-4 / 2.083e-4 / 1.736e-4; W's two complete runs
2.04e-8, median 1.005e-4), the arm-to-control ratio of the cell means of `dose_rate` (1.0 is "matched" in the X27
sense; the controller may move it either way), the LR rung histogram (`3e-4 × 1.2^k`; the sibling's finding that a
median LR is a rung index), the last-epoch approx-KL median over 10M to 50M, clip fraction, entropy (by 5M window;
FRAME § 8: entropy is an outcome, not a dose), value loss, any touch of the floor or ceiling (0 in all eight B1 runs),
and the speed ratio ρ (GPU-h a step, arm over control, from the runs' launch records over 10M to 50M, and
`train/train_ms`). A label **MOVEMENT-UNMATCHED** is printed beside the class if the arm cell's median last-epoch
KL over 10M to 50M lies outside the band [0.005, 0.02]; it voids nothing.

## 3. The comparison and the look files

- **Deciding: matched STEPS** on view N, FRAME § 3 and § 4: the arm's 50M and 100M look files against the banked
  cell's own look files (the files B1 reads; no extra banked file is needed).
- **REPORTED at 100M: matched GPU time.** The arm is cheaper a step, so the strength-per-GPU-hour question is also
  played: the arm's 100M files against the banked runs' `checkpoint_<N>_steps.zip` with N the first 1M multiple at or
  after ρ × 100M (rounded toward the control: conservative to the arm). 9 more cells, ≈ 0.33 GPU-h. Those banked files
  join FRAME § 2 item 6's never-thin list when the card registers. Why matched steps decides: ρ is measured weeks
  apart under different box contention (**UNVERIFIED** that it is comparable), and at a speed gain of 14 % to 28 %
  (L21451 at forks; the throughput audit's projections, which assume other forms) a 3 × 3 cell cannot see the
  strength that buys.
- **What a tie means.** A NO-USEFUL-GAIN or UNDECIDED keeps ten epochs (FRAME § 5's default). This cell CANNOT adopt
  five epochs for their speed alone: showing "no worse within 2 pp" needs about ten runs an arm (FRAME § 8). It can
  find a large harm (the O8 size, −10.6 pp, reads LOSS essentially always) or a gain of +4 to +6 pp (P(WIN) 0.61 to
  0.93 at σ 1.76).

## 4. The arms, the seeds, the argvs

| arm | banked control | lever typed beyond the control's argv | seeds (spare) | runs |
|---|---|---|---|---|
| **Q5-N** (width 128) | B1 C: `rb_b1c_s5101`, `rb_b1c_s5102`, `rb_b1c_s5110` | `--n-epochs 5 --lr 6e-4` | 5201, 5202, 5203 (5209) | `rb_q5n_s5201` … `_s5203` |
| **Q5-W** (width 256) | B1 W: `rb_b1w_s5103`, `rb_b1w_s5104`, `rb_b1w_s5111` | `--n-epochs 5 --lr 6e-4` | 5204, 5205, 5206 (5209) | `rb_q5w_s5204` … `_s5206` |

Q5-W runs only if B1's W cell reaches 100M (FRAME § 2 item 7). At 256 the control's LR fell to 1.005e-4 and below
with cuts until 38 to 46M (sibling analysis); the arm starts 2 rungs higher (3.8 more cuts at 7 rollouts each, about
2.6M more steps above the band: **UNVERIFIED** that this costs anything).

The first-seed argvs (records `argv_q5n_s5201.txt`, `argv_q5w_s5204.txt`; the other seeds differ in `--seed` and
`--run-name` only; phase 2 is B1's resume form):

```
python -m main.launcher --restart-interval-hours 6 --pin-commit 15136b61634b667bb9da5ccc6647ff9265238829 --arch production --ent-coef 0.0125 --behaviour-check warn --n-epochs 5 --lr 6e-4 --checkpoint-every-steps 1000000 --steps 50000000 --seed 5201 --device cuda --log-level periodic --run-name rb_q5n_s5201
python -m main.launcher --restart-interval-hours 6 --pin-commit 15136b61634b667bb9da5ccc6647ff9265238829 --arch production --ent-coef 0.0125 --behaviour-check warn --d-model 256 --trunk-ffn-dim 512 --allow-nonproduction-arch --n-epochs 5 --lr 6e-4 --checkpoint-every-steps 1000000 --steps 50000000 --seed 5204 --device cuda --log-level periodic --run-name rb_q5w_s5204
```

## 5. The checks, run 2026-11-17 (records in this folder)

- **checkargs** (main `81594609b`'s tool, the pin's parser): both exit 0, `parser : the PINNED commit 15136b61's`,
  11 / 14 flags accepted, 0 unrecognized; the RECIPE surface lists `n_epochs 5`, `lr 0.0006` and `ent_coef 0.0125`
  as TYPED and nothing else. (`checkargs_q5n.txt`, `checkargs_q5w.txt`)
- **The resolved-config diff** (FRAME § 2 item 2, in a clean detached checkout of the pin): against C and against W,
  at 50M and at 100M, exactly `lr` (0.0003 → 0.0006), `n_epochs` (10 → 5), `seed`, `run_name`. `max_lr` resolves to
  None in both (the 2 × `--lr` rule applies at run time). PASS. (`diff_q5n.txt`, `diff_q5w.txt`)
- **The dry run** (`python -m main.launcher --dry-run`, main's launcher, a scratch `GEN3AI_MODELS_DIR`, an agent's
  nice-19 shell with `CUDA_VISIBLE_DEVICES=""`): both exit 3 with exactly B1's two refusals, `cuda devices : ✗
  REFUSED` (the CUDA preflight) and `inherited nice : ✗ REFUSED`; every other preflight passes, `role : FRESH`.
  (`dryrun_q5n.txt`, `dryrun_q5w.txt`)
- **The seed scan** (FRAME § 2 item 5): 688 sidecars and 4,146 directories under the main `models/` and
  `~/gen3ai_archive`, no hit for 5201 to 5219 or the sixteen names of both cards. (`seed_scan.txt`) Seed 5201 also
  appears as a TEAM-SPLIT seed in `x35_breakers_stage_2026-10-17` (not a training seed; no collision).
- **The CPU `--debug` smokes at the pin** (`--steps 10000`, the base and the lever typed, `--n-epochs 5` so the typed
  epoch count wins over `--debug`'s 1): see § 6.

## 6. Smoke results

Both PASS, at the pin, on CPU (`smokes_cpu.txt`; two at a time at nice 19 on the agent cores beside the live run):

| smoke | exit | updates (epochs each) | LR held | markers | wall | peak RSS |
|---|---|---|---|---|---|---|
| Q5-N (128) | 0 | 5 (5: `n_updates` 25) | 6e-4 | round-trip PASSED, LEARNER FREEZE released (10 checks), Training complete | 2,356 s | 3.29 GB |
| Q5-W (256) | 0 | 5 (5: `n_updates` 25) | 6e-4 | the same | 2,514 s | 5.41 GB |

The smoke run records `n_epochs` 5, `lr` 0.0006 and `git_hash` `15136b616` (Q5-N's `metadata.json`). The LR does not
move inside 10,000 steps (the controller's 7-rollout startup cooldown), so the smoke shows the lever is taken and
trains, not how the controller then behaves. `--debug` bypasses `--compile-trainer` and warm start (runbook): the
first minutes of a real launch remain the test of that layer.

## 7. Status: LAUNCH-READY ARGV, NOT LAUNCH-READY AS AN EXPERIMENT

Ready: the argvs, their checks, the dose definition. Owed before the first launch: the FULL `PREREG.md` (adopting the
frame, this card's § 2 and § 3, the σ amendment after B1's 50M look) and its landing on main; the GPU slot (Segment C
of the queue proposal). Owed before its 50M look: the `b1_fresh_cell` rule and kit (FRAME F1). No GPU check is owed:
the arm's network is the control's, so B1's pilots cover its fit and compile; five epochs change only how many
passes the same compiled update makes (**UNVERIFIED** on the card: no five-epoch fresh run at this pin has run).
