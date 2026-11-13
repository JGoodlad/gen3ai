# X70 — the ent-0 control of the trailing-magnet screen: PRE-REGISTRATION (2026-11-13)

**Status: REGISTERED 2026-11-13, blind: before any X70 training step, any X70 game and any read. NO FORK HAS
TRAINED and no launch is authorised by this file.** The launch is the orchestrator's GO through the Training
session's kit (arm `x70`); the hand-off is [`chain_items_x70.json`](chain_items_x70.json) (§7), which carries no GO.
Nothing below changes after the first fork launches except by a dated amendment, written before the deciding cross
is played, that says why.

**The question.** CE-3d's swap forks (the entropy bonus replaced by a pull toward a trailing copy of the policy)
beat the unchanged recipe's forks. Did the trailing anchor do that, or did removing the entropy bonus do it alone?
Put as the cross that decides: **at `--ent-coef 0`, do forks trained WITH the anchor beat forks trained WITHOUT
it?**

**The tier is FULL** (`../../README.md` § Registration tiers): the follow-up that CE-3d's PASS licensed (CE-3d
PREREG §10, §16.4), and the first time a learner with no regulariser at all trains from this plateau.

**In one paragraph.** One new arm, `x70`: P\* + 20M with `--ent-coef 0` and no magnet, four seeds (7331–7334), at
the swap forks' own training pin `9e97cad5`. The deciding cross is the three banked swap forks against the four new
forks, 4,000 mirrored pairs a cell, one look. Its classes are `ANCHOR-ADDS`, `ANCHOR-COSTS`, `NOT-SEPARATED` and
`INCONCLUSIVE` (§5.3). Two more crosses, against the eight C0 finals and against X67's three finals, are REPORTED
and decide nothing (§5.4). Every class licenses at most REGISTERING a next step; **nothing here moves a recipe, and
nothing moves one without the owner** (§10). Cost ≈ 16.4–17 GPU-h and ≈ 53 CPU lane-h (§11). The slot is proposed
in §12 and placed by the orchestrator; no BOUNDARY-1 run is paused for it.

Abbreviations: pp = percentage points. P\* = baseline `pstar_plateau` (`rb_x26_s2601` @280,000,170, sha256
`814823c5…`). C0 = P\* + 20M with no lever at the production side pin `f4b25238` (`rb_x35_c0_s3501`–`s3508`,
`ent_coef` 0.05), the banked plateau control. X67 = the entropy screen's arm (`rb_x35_x67ent_s6701`–`s6703`,
`ent_coef` 0.0125, [`../x67_entropy_screen_2026-11-09/`](../x67_entropy_screen_2026-11-09/README.md)). The swap =
CE-3d's arm (`rb_x35_ce3dswap_s7321`–`s7323`, `--ent-coef 0 --magnet-kl-coef 0.05`,
[`../ce3d_magnet_screen_2026-11-11/PREREG.md`](../ce3d_magnet_screen_2026-11-11/PREREG.md)). `ent_coef` = the
coefficient on the entropy bonus in the PPO loss. KL = the Kullback–Leibler divergence. π = the learner's policy
over a row's legal actions; π_mag = the magnet's (a frozen copy of the policy, refreshed every 2M steps). β =
`--magnet-kl-coef`. H(π) = the policy's entropy, in nats. T2 = the Rust-side inference server self-play runs on;
R1 = the learner's compiled update step. K9(b) = the behaviour-policy check (does the learner's log-probability of
each played action match the one stored at play time). SD = standard deviation; df = degrees of freedom.
B1 = BOUNDARY-1 ([`../boundary1_scratch_2026-11-09/PREREG.md`](../boundary1_scratch_2026-11-09/PREREG.md)), the
live GPU campaign. L = B1's launch, 2026-11-12 19:26 PST.

## 1. Why now, and what the cross can and cannot say

**What CE-3d found** (its §16; ledger entry of 2026-11-12). Three swap forks beat the eight C0 finals head to head,
D = +3.97 pp, 90 % interval [+3.31, +4.63]. They ended at 0.484–0.496 nats of policy entropy, against C0's
1.004–1.018 and X67's 0.659–0.687 (label `ENT-LOW`). Against X67's finals they read +1.01 pp [−0.12, +2.13], so the
PASS is quoted as "the swap's gain over C0 is plausibly the lower uniform push alone". Its licence named the
control it most needs: an `--ent-coef 0` arm with no magnet at the CE-3d pin on new seeds. This file registers that
arm.

**What the two arms' losses are.** The swap adds β · KL(π ‖ π_mag) with β = 0.05 and sets `ent_coef` to 0. Written
out, β · KL(π ‖ π_mag) = −β · H(π) − β · Σ π log π_mag. The first part is an entropy bonus at weight 0.05, which is
C0's own (the pin's banner says so: "the KL's own entropy part adds 0.05 to that weight"). The second part is a
cross-entropy pull toward the trailing copy. So in loss terms the three recipes are:

| recipe | entropy term | pull toward the trailing copy |
|---|---|---|
| C0 | 0.05 · H(π) | none |
| the swap | 0.05 · H(π), inside the KL | 0.05 · cross-entropy to π_mag |
| **X70 (this arm)** | **none** | **none** |

The swap is therefore "C0 plus a cross-entropy pull", and X70 is "C0 minus the entropy bonus". The deciding cross
compares the swap with X70: ONE flag differs (`--magnet-kl-coef 0.05` against its absence, both at `--ent-coef 0`),
and what that flag adds is the WHOLE KL term, both parts.

**What the deciding cross answers.** Whether the full trailing-KL term adds head-to-head strength over having no
regulariser, at this plateau and horizon. Its sign says which way: D > 0, the anchor adds; D < 0, the policy is
stronger with nothing.

**What it cannot answer** (stated before the number):
1. It cannot separate the anchor's DIRECTION (stay near your recent self) from the ENTROPY LEVEL the KL holds up.
   If the X70 forks end sharper than the swap forks, an `ANCHOR-ADDS` could be "0.49 nats is a better operating
   point than the X70 forks' entropy", which a plain small entropy bonus might also reach. §8's entropy labels
   carry that reading; the test of it is a matched-realised-entropy arm, which is NOT registered here (§13).
2. It does not say whether the swap's +3.97 pp over C0 is "the anchor" or "less uniform push". The REPORTED cross
   of X70 against C0 (§5.4) is the evidence on that, and it decides nothing.

**A GUESS, written before any step (it binds nothing).** With no regulariser the policy sharpens further than the
swap's (`ENT-BELOW-SWAP`), and the X70 forks beat C0 by about what X67's did or more. The deciding D is small and
positive, between 0 and +1.5 pp, and most likely `NOT-SEPARATED`.

## 2. The arm

**`x70`**: C0's argv plus two tokens, `--ent-coef 0`. Equivalently the swap's argv minus `--magnet-kl-coef 0.05`.
Everything else is inherited from P\*'s record, as for every X35 production-side fork: `--arch production`'s
architecture, `n_epochs` 10, effective batch 65,536, the adaptive learning-rate controller, the 14-snapshot pool.

| | C0 | X67 | the swap | **X70** |
|---|---|---|---|---|
| `--ent-coef` | (0.05, inherited) | 0.0125 | 0 | **0** |
| `--magnet-kl-coef` | — | — | 0.05 | **untyped ⇒ 0.0, OFF** |
| training pin | `f4b25238` | `f4b25238` | `9e97cad5` | **`9e97cad5`** |
| runs | `rb_x35_c0_s<seed>` | `rb_x35_x67ent_s<seed>` | `rb_x35_ce3dswap_s<seed>` | **`rb_x35_x70ent0_s<seed>`** |

Read from the pin's source, not from prose: an untyped `--magnet-kl-coef` resolves to 0.0 and builds no magnet (no
T2 magnet slot, no tenth argument to R1, no sidecar); the pin's `model_config.json` WRITES `magnet_kl_coef: 0.0`
and `magnet_refresh_steps: 2000000` for every run, so an X70 config carries the field at 0.0 (§5.2 checks exactly
that); the `[MAGNET] --magnet-kl-coef` line prints only when the coefficient is above 0.

In the kit (`src/main/x35/kit.py`): `X70_LEVER = ("--ent-coef", "0")` (the first two tokens of `CE3D_SWAP_LEVER`, so
the two arms cannot drift apart), `ARM_TYPED_RECIPE["x70"] = (("ent_coef", "0.0", "--ent-coef"),)`, and the arm
`Arm("x70", "production", L_PROD, X70_SEEDS, "rb_x35_x70ent0_s{seed}", 0, "warn")`.

## 3. Seeds, pin, stage, dose, one look

**Seeds: 7331, 7332, 7333, 7334** (`kit.X70_SEEDS`). They collide with nothing: not with any seed in `src/`,
`designs/`, `scripts/` or `tools/`, the X35 job's status file, `models/` or the run archive (searched 2026-11-13);
not with the swap's 7321–7323 or CE-3d's six retired seeds 7301–7303 and 7311–7313. No fifth seed is registered:
the kit refuses 7335.

**Why four and not three** is §6.2: the swap arm has three banked forks and no more, and the fourth X70 fork is
the cheapest unit of power that is left (≈ 4.1 GPU-h for +12 points of power at +1 pp).

**The training pin: `9e97cad5709da21a1f9d4d8846096c87eb17b1d0`** (tag `ce3d-pin-9e97cad5`, `batch.CE3D_PIN`), the
pin the three swap forks trained at. `batch.ARM_PINS["x70"] = CE3D_PIN`, and `batch.arm_pin_check` refuses any
other pin for the arm, naming this section. The pin carries `src/main/launcher/pinned_data_test.py`, so its runs
read their own `data/`: queuing X70 freezes no `data/` change on main (root rule 6).

**The play pin: `333a73ddda7829b9ccdf9e17ef04624eb40edeba`** (tag `ce3d-pin-333a73dd`), the pin both CE-3d crosses
were played at. `f4b25238` cannot load an X70 final (its config carries the magnet's two fields) and HEAD's game
and observation are not the ones these finals trained on.

**The stage: `X70`** (`batch.X70_STAGES`): the four forks, in seed order, one at a time. `kit stage-check` allows
a fork only while this file exists at `batch.X70_PREREG`. The forks need not run back to back: each is its own
chain item, so the orchestrator may place them in separate GPU gaps (§12). All four train before any cell of any
cross is played.

**Dose: matched to the swap forks by `main.dose`** ([`dose_swap.txt`](dose_swap.txt)): +20,000,000 steps from
280,000,170 (`--steps 300000170`), effective batch 65,536, 10 epochs. The swap forks read `lr_median` 0.0001005
and a dose rate of 2.044e-08, each. A fork whose epochs or effective batch differ is VOID (§8); a fork whose
`lr_median` moved is labelled, not voided (§8), because the learning-rate controller is part of the recipe.

**One look.** The deciding cross is played once, after the fourth fork's arm-end checks, and read once. No look 2,
no top-up, no extension. A cell that falls short of its registered pairs makes the read `INCONCLUSIVE` (§5.2).

## 4. The confound, and the first fork as a GPU confirmation

### 4.1 What is matched in the deciding cross

CE-3d read forks trained at `9e97cad5` against finals trained at `f4b25238` and had to carry that as a caveat (its
§4.3). **The deciding cross here has no pin difference**: both arms trained at `9e97cad5`, both are forks of the
same P\* checkpoint by sha256, with the same argv but for the lever, and they are played at one pin. The plan
enforces one `torch_version` and one `init_num_threads` across all seven runs.

### 4.2 What still differs, besides the lever

| difference | what it is | evidence that it is not the result | status |
|---|---|---|---|
| the lever | `--magnet-kl-coef 0.05` in the swap, absent in X70 | it IS the treatment | — |
| the day | the swap forks trained 2026-11-12 to 11-13; X70 trains later, possibly in separate gaps | same card, same interpreter pin, same training pin; the plan's `torch_version` / `init_num_threads` guard | CE-3d and X67 carried a larger version of the same difference |
| the seeds | 7321–7323 against 7331–7334 | seed variance is what the statistic's two variance terms estimate | in the rule |
| the shared compile cache | the swap's R1 took the magnet's argument; X70's does not, so it compiles its own program | a compiled program is a function of its source and inputs; the compile gate checks parity at startup | **UNVERIFIED on the card** (§4.3) |
| the box's load | B1 or its reads may share the box | the trainer cores are split from the agent cores (`designs/ops/core_split.md`); throughput is gated (§8) | reported per fork |

For the two REPORTED crosses the columns are C0's and X67's finals, trained at `f4b25238` on other days: CE-3d
§4.3's whole table applies to them, unchanged, and that is one reason they decide nothing.

### 4.3 The first X70 fork is the first β = 0 learner at this pin on the card

CE-3d §4.4 item 6 left it open: "that a β = 0 learner at this pin trains on the card as `f4b25238` does" was shown
on the CPU only (the learner golden is bit-identical in bf16 and fp32 at the first pin with β = 0; the fix's three
paths leave a β = 0 dry update as it was). Fork 7331 is that confirmation. **How a failure is classified**, fixed
now:

| what happens | what it is | what follows |
|---|---|---|
| the launch is held or dies before its first training step, from a cause outside the pin's code (the lease, disk, a kit pre-check, a box fault) | a launch failure, not a fork | the dead directory is moved aside, never deleted; the same seed relaunches under its own name with the orchestrator's word and a dated NOTE here. No guard judges it and nothing is read |
| a FATAL that is the pin's own at β = 0: the compile gate, T2 parity, the startup fit check, the behaviour check | **a FINDING about the pin** (it also falsifies CE-3d §4.3's "nothing changes for a β = 0 learner" row on the card) | **the arm STOPS**: no further fork launches, nothing is read, no class. A fix is a new pin, which breaks §4.1's match; it needs a dated amendment that restates §4, or a new registration. The orchestrator decides which |
| a canary FATAL | not judged here: the arm is in no canary family (§8) | HOLD for the orchestrator; never resumed on a bar the Training session types |
| it trains, below 1,000 steps/s in the first hour | PAGE | HOLD and keep training; the orchestrator decides |
| it trains to the end and a §8 guard fires | a VOID fork | §8 |
| it trains to the end and every guard passes | a valid fork, and the confirmation | REPORTED beside the read: its median `time/fps` next to the swap forks' 1,372–1,398 and its compile and canary lines |

A failure of the second kind is never read as "ent-0 is worse": no cross is played on a stopped arm.

## 5. The meter and the rule

### 5.1 The crosses

| cross | rows (the player) | columns | cells | pairs a cell | games | role |
|---|---|---|---:|---:|---:|---|
| **`anchor`** | the 3 swap forks | the 4 X70 forks | 12 | **4,000** | 96,000 | **DECIDES** |
| `c0` | the 4 X70 forks | the 8 C0 finals | 32 | 1,000 | 64,000 | reported |
| `x67` | the 4 X70 forks | X67's 3 finals | 12 | 1,000 | 24,000 | reported |

Each cell is mirrored pairs of each `final_model.zip`, both sides greedy, batches of 500 pairs, schedule seed 0,
the players' eval team builder (no panel), `--oracle-reveal-mode off`, purpose `ab`, label `x70_ent0_control`, at
the play pin. This is CE-3d's and X67's protocol with one change: the deciding cell holds 4,000 pairs, not 1,000
(§6.2). Families `x70_swap_vs_ent0_h2h`, `x70_ent0_vs_c0_h2h`, `x70_ent0_vs_x67_h2h`; requests `x70_swap_vs_ent0`,
`x70_ent0_vs_c0`, `x70_ent0_vs_x67`. The driver is [`cross/drive_cross.sh`](cross/drive_cross.sh). **The deciding
cross is played first**; no cross waits on another or reads another's cells, and no banked CE-3d or X67 cell is
re-read as a look of this screen.

**Why the swap forks against the X70 forks is the deciding cross, and not X70 against C0.** The question is the
anchor's own contribution. The direct cross has one lever between its arms and one pin (§4.1). The indirect route
(X70 against C0, set beside the swap's +3.97 against C0) would difference two crosses played at different times
with a pin and day confound in both, and X67's lesson was that such a difference has no registered interval.

### 5.2 The plan ([`cross/plan_cross.py`](cross/plan_cross.py))

**Every run of a cross is REQUIRED**; a missing final or a failed guard REFUSES the plan (exit 3). A run must
have: `final_model.zip`; a `pin_history` of exactly one entry at its pin (`9e97cad5` for a swap or X70 fork,
`f4b25238` for C0 and X67) from 280,000,170 to at least 300,000,170; the lineage record of a FORK of P\* by sha256;
its own seed; `n_epochs` 10; its arm's `ent_coef` in `metadata.json` and in its `cli_args` (X70: 0.0); its arm's
magnet record in `model_config.json` (**X70: `magnet_kl_coef` present and 0.0**; the swap: 0.05 with a refresh of
2,000,000; C0 and X67: both keys absent); `final_model.magnet.pt` beside a swap final and beside no other;
K9(b) PASS over its own rows (`main.x35.batch.k9b_read`); one `torch_version` and one `init_num_threads` across
the cross.

At the registration ([`plan_check_banked.txt`](plan_check_banked.txt)): the 14 banked finals pass every guard, and
the `anchor` plan REFUSES (exit 3) because no X70 final exists.

**The read** ([`cross/read_cross.py`](cross/read_cross.py)) needs every cell present once, `OK`, with EXACTLY its
registered count of completed pairs. Fewer or more is an `INCONCLUSIVE` reason and the statistic is not computed.
The driver's resume replays batches the ledger does not hold, so an interrupted cross is finished, not re-read.

### 5.3 The statistic and the classes ([`screen_rule.py`](screen_rule.py))

h[i][j] = swap fork i's score in pp (a draw = 1/2) against X70 fork j. With r = 3 rows and c = 4 columns:

```
D    = mean(h) − 50                                         the swap's lead = the anchor's contribution, pp
s²_R = max(variance of the 3 row means,    CELL_VAR / 4)    ddof 1; the floor is a row mean's binomial part
s²_C = max(variance of the 4 column means, CELL_VAR / 3)
V    = s²_R / 3 + s²_C / 4
df   = V² / ((s²_R / 3)² / 2 + (s²_C / 4)² / 3)             Welch–Satterthwaite
90 % interval = D ± t(0.95, df) · √V                        CELL_VAR = 2500 / (2 · 4000) = 0.3125 pp²
```

This is CE-3d's estimator (itself X67's) with the cell's pair count a parameter. The rule is copied into this
directory, never imported from CE-3d's; [`selfcheck.log`](selfcheck.log) shows it reproducing CE-3d's two banked
results from their recorded matrices to 1e-9 (D 3.9677 [3.3061, 4.6293], df 7.3176; D 1.0083 [−0.1152, 2.1319],
df 4.0).

| class | condition | reading |
|---|---|---|
| **`ANCHOR-ADDS`** | the interval's LOWER end > 0 | with the trailing KL the forks are stronger than with no regulariser, at this plateau and horizon |
| **`ANCHOR-COSTS`** | the interval's UPPER end < 0 | with no regulariser the forks are stronger than with the trailing KL |
| **`NOT-SEPARATED`** | neither | quoted with its upper end: "the anchor adds less than `<hi>` pp at this horizon", and with §6's power line |
| **`INCONCLUSIVE`** | an input is missing or invalid | never interpreted; D is not computed; a re-run is a new registration |

Each directional class is a one-sided 95 % statement; under D = 0 either fires at most 5 % of the time (§6.1).
**Rule 8:** an end within 1e-9 pp of 0 satisfies neither class and is recorded in `near_boundary`; a tie never
classifies. Reported beside the verdict, never deciding: the same interval with both variances floored at the
fresh-run seed variance (1.76 pp)², B1's planning figure.

### 5.4 The two REPORTED crosses. They decide nothing and carry no class

Rows = the four X70 forks. `screen_rule.report` applies the same statistic (CELL_VAR = 1.25 pp² at 1,000 pairs)
and returns one of three READINGS a cross, fixed here; `decide` refuses these matrices' shapes.

| cross | lower end > 0 | upper end < 0 | neither |
|---|---|---|---|
| `c0` (4 × 8) | "removing the entropy bonus alone, with no anchor, beats the C0 finals at this horizon" | "removing the entropy bonus alone, with no anchor, loses to the C0 finals at this horizon" | "the ent-0 forks are not separated from the C0 finals at this horizon" |
| `x67` (4 × 3) | "ent_coef 0 beats ent_coef 0.0125 at this horizon: the gain from a lower uniform push continues below 0.0125" | "ent_coef 0 loses to ent_coef 0.0125 at this horizon: the gain from a lower uniform push turns over above 0" | "ent_coef 0 is not separated from ent_coef 0.0125 at this horizon" |

**Why they are played.** Together with the banked results they draw the entropy coefficient's dose curve at this
plateau: 0.05 (C0), 0.0125 (X67, +1.97 pp [+1.35, +2.59] over C0), 0 (X70), and beside it the swap (+3.97 over
C0). That is the evidence on §1's second limit.

**Why they decide nothing.** Their columns trained at another pin on other days (§4.2); the question registered
here is the anchor's, and a second deciding cross would be a second look at it. What they can see
([`power_reported.log`](power_reported.log)): against C0 a +1 pp effect clears 0 with probability 0.90 at a seed
SD of 0.3 pp and 0.72 at 0.5; against X67's three finals only 0.41 and 0.34 (0.81 and 0.67 at +1.5 pp). A
"not separated" from the `x67` cross is therefore weak evidence and is quoted with those figures.

## 6. Power

### 6.1 The registered design ([`power.log`](power.log): `screen_rule.py simulate`, 50,000 draws a row)

The model: h[i][j] = 50 + D + a_i − b_j + noise, with a_i and b_j normal at the row and column seed SD and cell
noise of 1.05 pp at 1,000 pairs scaled by √(1000 / pairs).

| seed SD, rows / columns (pp) | median half-width | P(`ANCHOR-ADDS`) at D = 0 | +0.5 | **+1** | +1.5 | +2 | +3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 / 0 | 0.47 | 0.001 | 0.46 | 0.99 | 1.00 | 1.00 | 1.00 |
| **0.3 / 0.3** | **0.63** | 0.016 | 0.31 | **0.83** | 0.99 | 1.00 | 1.00 |
| 0.4 / 0.4 | 0.75 | 0.026 | 0.26 | **0.70** | 0.94 | 0.99 | 1.00 |
| 0.5 / 0.5 | 0.87 | 0.032 | 0.21 | **0.58** | 0.87 | 0.98 | 1.00 |
| 0.5 / 1.0 | 1.22 | 0.040 | 0.15 | 0.37 | 0.64 | 0.85 | 0.99 |
| 1.0 / 1.0 | 1.59 | 0.042 | 0.12 | 0.25 | 0.44 | 0.64 | 0.91 |
| 1.76 / 1.76 (fresh runs) | 2.72 | 0.043 | 0.08 | 0.14 | 0.22 | 0.32 | 0.55 |

`ANCHOR-COSTS` is the mirror image (0.83 / 0.70 / 0.58 at −1 pp). The false-positive rate is at most 0.046 a side
in every row.

**Which row is believable.** The only measurements of a P\* + 20M fork's seed spread are CE-3d's two crosses
(`selfcheck.log` prints them). The swap forks' row means varied by LESS than their binomial floor in both (0.146
against 0.156 pp², 0.126 against 0.417), so they show no seed spread above game noise. The eight C0 finals' column
means varied by more than theirs (0.572 against 0.417 pp²), an excess SD of about 0.39 pp; X67's three did not
(0.359 against 0.417). So the plausible range is 0.3 to 0.5 pp, with C0's own estimate near 0.4. A regulariser-free
arm may spread more than any of them; that is unknown until it trains.

**The power line.** The registered cross resolves a half-width of ≈ 0.63–0.87 pp. It sees a **+1.5 pp** anchor
effect with probability **0.87–0.99**, and a **+1 pp** effect with probability **0.83 at a seed SD of 0.3, 0.70 at
0.4 and 0.58 at 0.5**. It is therefore a reliable test of +1.5 pp and an UNRELIABLE test of +1 pp unless the seed
spread is at the low end. A `NOT-SEPARATED` does not show the anchor adds nothing: it bounds it at the interval's
upper end.

### 6.2 The sizes that were weighed ([`power_designs.log`](power_designs.log))

P(`ANCHOR-ADDS`) at D = +1 pp, seed SD 0.3 / 0.5:

| design (swap × X70 @ pairs a cell) | SD 0.3 | SD 0.5 | extra cost over 3 × 3 @ 1,000 |
|---|---:|---:|---|
| 3 × 3 @ 1,000 (CE-3d's cell size) | 0.25 | 0.23 | — |
| 3 × 3 @ 4,000 | 0.71 | 0.47 | games only |
| 3 × 4 @ 1,000 | 0.41 | 0.34 | one fork |
| 3 × 4 @ 2,000 | 0.68 | 0.48 | one fork |
| **3 × 4 @ 4,000 (registered)** | **0.83** | **0.57** | one fork (≈ 4.1 GPU-h), 96,000 games |
| 3 × 4 @ 8,000 | 0.90 | 0.62 | one fork, 192,000 games |
| 3 × 5 @ 4,000 | 0.88 | 0.64 | two forks |
| 3 × 6 @ 4,000 | 0.91 | 0.67 | three forks |
| 6 × 6 @ 2,000 | 0.99 | 0.87 | three X70 forks AND three NEW swap forks (≈ +25 GPU-h) |

- **A 3 × 3 cross at CE-3d's 1,000 pairs cannot separate a +1 pp effect**: it sees one a quarter of the time. That
  is the size the brief asked about, and it is rejected.
- **Pairs first.** At 1,000 pairs a cell's game noise (1.05 pp) is larger than the seed spread it is trying to
  see through. Quadrupling the pairs is CPU or a few GPU minutes; it takes 3 × 3 from 0.25 to 0.71.
- **Then one more fork.** The fourth X70 fork buys +12 points at SD 0.3 for ≈ 4.1 GPU-h. A fifth buys +5 and a
  sixth +3: the limit is no longer the X70 side.
- **The limit that stays: three banked swap forks, 2 df on that factor.** At a seed SD of 0.5 no design with three
  rows reaches 0.8 at +1 pp. Only more swap forks do (6 × 6: 0.87), and those are three new magnet forks (≈ 13
  GPU-h) at a later date. They are NOT registered: the GPU is booked for B1, and whether +1 pp of anchor is worth
  25 GPU-h more is the question a `NOT-SEPARATED` with an upper end near +1.5 would put to the owner, with this
  table.
- 8,000 pairs was rejected: +7 points for twice the games, past the point where seed spread dominates.

### 6.3 What the power figures assume

- **UNVERIFIED: cell noise falls as 1/√pairs out to 4,000 pairs.** No banked cell holds more than 1,000. Each
  500-pair batch draws its own team schedule from its own seed, so batches are independent given the two finals;
  a per-cell team effect that does not average out would show as extra noise. A residual of 0.3 pp a cell costs
  about 3 points of power (`power.log`, the `tau` row). The floors in the rule assume the binomial figure; the
  sample variances pick up any excess, which widens the interval rather than biasing D.
- A normal model for seed effects, and a plateau-fork seed spread the X70 arm shares. Neither is checked here.

## 7. The launch commands, their validation, and the hand-off

For the Training session, from the main checkout at or after this registration's commit, holding the lease, one
run at a time, **only on the orchestrator's GO**:

```bash
STAGE=designs/research_state/measurements/x35_breakers_stage_2026-10-17
PIN=9e97cad5709da21a1f9d4d8846096c87eb17b1d0       # tag ce3d-pin-9e97cad5: the swap forks' training pin
python -m main.x35.kit stage-check --stage $STAGE --arm x70 --seed $s                # ALLOWED: "X70 PREREG present"
bash $STAGE/launch_x35.sh x70 $s --pin $PIN --lease-owner "<owner>" --thinning       # the kit's DRY RUN (default)
bash $STAGE/launch_x35.sh x70 $s --pin $PIN --lease-owner "<owner>" --thinning GO    # only on the orchestrator's GO
```

for `$s` = 7331, 7332, 7333, 7334. The kit resolves each to `python -m main.launcher --restart-interval-hours 24
--pin-commit $PIN <argv>`; the first seed ([`argv_x70_s7331.txt`](argv_x70_s7331.txt)):

```
--model /home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip --behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cuda --ent-coef 0 --seed 7331 --run-name rb_x35_x70ent0_s7331
```

The block was generated from the kit (`kit.build_argv`), not typed. Validated on 2026-11-13 at the training pin,
CPU only, no lease, `CUDA_VISIBLE_DEVICES=""`, nice 19 on the agent cores, nothing created:

- `python -m main.checkargs --argv "<argv>" --pin $PIN` → `checkargs_x70_s<seed>.txt`, all four: **exit 0**; every
  flag accepted by the PINNED parser; every ARCH-surface key the production mirror's; the lever row
  `ent_coef 0.0 (TYPED) production: 0.05`; `✓ fork arch`; no magnet row.
- `python -m main.launcher --dry-run --restart-interval-hours 24 --pin-commit $PIN <argv>` →
  `dryrun_x70_s<seed>.txt`, all four: **role FORK of `rb_x26_s2601`**, the pin (source `pin_commit`), the lever
  row TYPED, 14.27 GiB REQUIRED. **Exit 3, "would NOT launch", on exactly two refusals, both this unit's own
  shell and both expected:** the CUDA-visibility preflight (`CUDA_VISIBLE_DEVICES=''`: no lease) and the
  inherited-nice preflight (the shell ran at nice 19 by rule 5b; a GPU launch is refused above nice 10). The kit's
  own dry run re-runs it with the card visible, at the launch shell's priority, before every GO. The transcripts'
  `showdown deps` line names the registering unit's worktree, which no longer exists.
- `python -m main.x35.kit stage-check` and `pin-check` → [`kit_checks_x70.txt`](kit_checks_x70.txt): the four
  seeds ALLOWED; a fifth seed refused; the training pin passes for the arm; the first CE-3d pin and the plain
  side pin are each refused.
- `python -m main.ops.arm_chain plan <scratch chain> --check-pins --describe` on the four draft items →
  [`arm_chain_plan_x70.txt`](arm_chain_plan_x70.txt): PLAN OK, the kit agrees on all four; an unregistered seed
  REFUSES. The plan does not judge an arm's own pin; that is the launch script's `kit pin-check`.
- `src/main/x35/x70_test.py`: the kit's argv equals the four validated files token for token, equals C0's plus
  the two lever tokens and the swap's minus the magnet's two; `check_validation` passes the eight transcripts (the
  two shell refusals modelled as absent) and fails them when read as another arm; the chain items are the kit's
  forks at the pin with §7's banner; the cross plan's constants are the kit's; the rule's self-check passes.

**The chain items** ([`chain_items_x70.json`](chain_items_x70.json); `go_by` is the orchestrator's to fill, and
the file is a draft of items, not a chain: **there is no GO marker**). Each: the arm, the seed, the pin,
`thin: planner`, `flags: ["--thinning"]`, the throughput gate (1,000 steps/s within 60 min, first line within 90),
`on_throughput_fail: page`, `on_void: stop`, and a banner the child's log must show within 30 min:

```json
{"expect": "ent_coef   | 0 ",
 "void": ["ent_coef   | 0.05 ", "ent_coef   | 0.1 ", "ent_coef   | 0.0125 ", "[MAGNET] --magnet-kl-coef",
          "[ModelVersion] FATAL"], "within_min": 30}
```

The expected string is the hyperparameter table's own row for this arm (`|    ent_coef   | 0        |`, as the
swap forks' logs print it; the trailing space keeps `0.05`, `0.0125` and `0.1` from matching). The VOID lines are
the rows of the three entropy coefficients a mistyped or inherited launch could carry, and the magnet's attach
line, which the pin prints only with β above 0: **an X70 fork that starts with an entropy bonus or with a magnet
is stopped.** VOID wins when both appear. The table prints when `learn()` starts, after the
startup's compile gate and prewarm and before the first rollout: 5.3 and 7.6 min after the child attached on X67's
first two forks, whose banner was this row at its own value, and after a COLD prewarm of 275 s on the swap's first
fork. X70's first fork compiles R1 cold too (§4.2). A start that runs past the 30 minutes is a VOID by deadline:
the chain stops the launch, it goes to the orchestrator, and it is §4.3's launch failure, not a fork.

**Disk.** `disk_need_gib` is **15.0** a fork: the dry run's 14.27 rounded up. The arm writes no magnet sidecar,
so CE-3d's allowance for those is not carried.

**Not executed at this registration:** `launch_x35.sh` (no lease); any cross's play command (it writes eval-ledger
rows); any game. Executed: the kit's argv, `checkargs` and the launcher's dry run for all four seeds; the kit's
stage and pin checks; the arm chain's plan on a scratch chain; the cross plan's guards on the 14 banked finals and
its refusal without the X70 finals; the rule's self-check and its three simulations.

## 8. VOID guards, labels, the canary family

**VOID** (any one ⇒ that fork is VOID). **All four X70 forks are required**: with a VOID fork the plan REFUSES and
no cell is played. A replacement is a new seed by a dated amendment (and a kit change), written before any cell
of any cross is played; the cross is never shrunk to three columns after a fork's training curve has been seen.
- not finished: the arm chain's completeness check (rc 0, `Last step` ≥ 300,000,170, `final_model.zip`, the newest
  checkpoint within 98,304 steps of it);
- **K9(b) not PASS at the arm's end** (`python -m main.x35.kit k9b --run-dir <run>`), as for every X35 fork;
- the launcher log names another pin than `9e97cad5`, or the run is not `FORK of …/rb_x26_s2601`;
- the post-spawn pool-seed check fails;
- **the lever is not in the record**: §7's banner (the chain STOPS the run); any of §5.2's plan guards on
  `ent_coef`, `magnet_kl_coef` or a stray sidecar; `main.dose` reads an epoch count other than 10 or an effective
  batch other than 65,536.

**A FATAL on the first fork, a launch failure, a canary FATAL, low throughput:** §4.3's table.

**Labels, reported beside the class and never changing it.** End entropy = the mean of a run's last 20 points of
−`train/entropy_loss` (X67's and CE-3d's definition; the tag is logged at `ent_coef` 0 as at any other). The swap
forks' values are read from CE-3d's banked record (`../ce3d_magnet_screen_2026-11-11/record/diag9_ce3d.json`:
0.4838, 0.4962, 0.4921), never retyped; `python screen_rule.py labels <e1> <e2> <e3> <e4>` applies:
- **`ENT-BELOW-SWAP`**: the HIGHEST X70 end entropy is below the LOWEST swap one (0.4838). The X70 arm ended
  sharper than every swap fork: the anchor held entropy up, and any verdict is then "the anchor OR its entropy
  level" (§1). With `ANCHOR-ADDS` it names the next control: a matched-realised-entropy arm.
- **`ENT-ABOVE-SWAP`**: the LOWEST X70 end entropy is above the HIGHEST swap one (0.4962). The anchor sharpened
  the policy beyond what removing the bonus does.
- Neither when the ranges overlap: the two arms sit at one entropy level and the cross reads the anchor's
  direction at its cleanest. A difference within 1e-9 nats of a bound counts as overlap (rule 8).
- **`ENT-COLLAPSED`**: ANY X70 fork ends below half the lowest swap end entropy (below 0.2419 nats). **A label,
  not a VOID, and that is deliberate.** A policy that collapses with no regulariser is the lever's own effect and
  the result the arm exists to show; voiding it would condition the sample on its outcome and keep only the forks
  that happened not to collapse. A collapsed fork still plays its cells, and the collapse is quoted with the
  class. What IS guarded is a fork whose data is invalid: K9(b) at the arm's end, above.
- **`LR-MOVED`**, a fork: its `lr_median` (`main.dose`, as printed) is below 0.0000905 or above 0.0001105, more
  than 10 % from the swap forks' and C0's 0.0001005 (X67's label; all three X67 forks carried it, no swap fork
  did). A class with `LR-MOVED` goes to the orchestrator with both numbers.

**Canary family: none.** `rb_x35_x70ent0_s*` is in no family of `canary_decide.FAMILY_NAME_PATTERNS`: no
healthy-pass ratio was ever measured for a β = 0 learner at this pin, and C0's bar (2.64) was measured at the side
pin with the entropy term live. A canary FATAL on an X70 fork therefore HOLDs ("no family bar") and goes to the
orchestrator.

## 9. REPORTED diagnostics (they decide nothing)

- **Entropy, the rate and the KL**: `python -m main.ops.tb_read <run> --tag train/entropy_loss --tag
  train/learning_rate --tag train/approx_kl` for the four forks; the entropy curve in 5M windows beside the swap
  forks' and C0's; whether and when the controller stepped the rate. The reading unit copies CE-3d's
  `record/diag9_ce3d.py` into this directory's record for it; the end-entropy definition is §8's.
- **Where the entropy went**: the step at which each X70 fork's entropy first fell below the swap's end range,
  if it did. A fork that crossed early spent most of its 20M steps sharper than any swap fork.
- **The two reported crosses** (§5.4), each with its D, interval and registered reading, and the dose curve they
  complete. The difference between two crosses' D carries no interval and no class.
- **The first fork's GPU confirmation** (§4.3): throughput, the compile lock, the canary's lines.
- **The fresh-floor interval** from `decide`'s reported block.
- No Kakuna game is planned. One played later needs a dated NOTE here, written before any game.

## 10. What each class licenses

**Every class licenses at most REGISTERING a next step. No class moves a recipe, and nothing moves a recipe
without the owner.**

- **`ANCHOR-ADDS`** ⇒ at β 0.05 and a 2M cadence, the trailing KL makes a P\* + 20M fork stronger head to head
  than no regulariser does. It licenses REGISTERING one of:
  - with `ENT-BELOW-SWAP`: first the matched-realised-entropy arm (a small entropy bonus tuned to end near 0.49
    nats, against the swap), because the result is not yet "the anchor";
  - with overlapping entropies or `ENT-ABOVE-SWAP`: a longer-horizon or fresh-run magnet arm, which the owner
    weighs against B1's queue.
  It is quoted with D's interval, the labels, and §5.4's two readings word for word.
- **`NOT-SEPARATED`** ⇒ the magnet row closes at this spelling as "not shown to add over removing the entropy
  bonus; adds less than `<hi>` pp at +20M", quoted with §6's power line. CE-3d's +3.97 pp is then read as an
  entropy-coefficient result, and the open question becomes the coefficient itself (X67's PASS is already the
  owner's to decide; the `c0` and `x67` readings go to the owner beside it). With an upper end above +1 pp the
  report says so and shows §6.2's table: the cross could not rule out an effect of that size.
- **`ANCHOR-COSTS`** ⇒ the magnet row closes at this spelling as harmful relative to no regulariser; no magnet
  follow-up is queued unless a new case is made for a different form. The entropy coefficient goes to the owner
  as above.
- **`INCONCLUSIVE`** ⇒ never interpreted; a re-run is a new registration.

**This file moves no recipe.** No class puts a magnet or an `ent_coef` into `production_config.json`,
`design_learner_recipe.md` or any B1 arm; the screen never holds a B1 launch or look, and it neither waits on nor
informs the owner's decision on X67's PASS until it is read.

**The limits, stated before the number:**
1. **The anchor is the whole KL term** (§1): its direction and the entropy it holds are not separated; the labels
   are evidence, not a test.
2. **Three swap forks.** The swap side has 2 df; +1 pp is seen with probability 0.58–0.83 (§6).
3. **The meter is not the backlog's question.** The card behind CE-3d asks whether a trailing regulariser damps
   cycling and exploitable tie-breaks. Head to head against a sibling arm measures neither; a sharp, regulariser-
   free policy may win head to head and be MORE exploitable. No best-responder read is registered here.
4. **A late fork, a short horizon**: 20M steps from a policy shaped by 280M steps at `ent_coef` 0.05. Nothing is
   claimed about `ent_coef` 0 or a magnet from step 0, where exploration matters most.
5. **One spelling of each side**: one β, one cadence, entropy coefficient exactly 0.
6. **Head to head against siblings is not strength against others** (X67 PREREG §10).
7. **The reported crosses carry CE-3d §4.3's pin and day confound**, and decide nothing.

## 11. Cost

| item | GPU-h | CPU | note |
|---|---:|---|---|
| X70 s7331–s7334, +20M each | **≈ 16.4–17** | — | 4.10 GPU-h a fork (the mean of the 20 X35 production forks' dose records, X60-T PREREG §5.1); the swap forks measured 4.25 with the magnet's pass (264.8 + 247.7 + 252.7 min); plus one first compile of R1 at β = 0 at this pin |
| the deciding cross, 96,000 games | ≈ 0.67 in a GPU window | or ≈ 9.2 h on the three agent cores (≈ 27.6 lane-h) | CPU: MEASURED rate, CE-3d's deciding cross, 48,000 games in 16,580 s on three threads with B1 live. GPU: an ESTIMATE carried from X67's and CE-3d's drivers (48,000 games ≈ 20 min); **UNVERIFIED, neither cross was played on the card** |
| the `c0` reported cross, 64,000 games | ≈ 0.45 | or ≈ 6.1 h (≈ 18.4 lane-h) | as above |
| the `x67` reported cross, 24,000 games | ≈ 0.17 | or ≈ 2.3 h (≈ 6.9 lane-h) | as above |
| the play checkout | 0 | none if CE-3d's `/home/goodlad/dev/gen3ai-reads/ce3d_play_333a73dd` still stands (it does at the registration: HEAD = the play pin, clean); else a Rust build of minutes under `mem_cap` | |
| §9's reads, the dose read, the guards | 0 | seconds | |

**≈ 16.4–17 GPU-h in all, and ≈ 53 CPU lane-h (≈ 17.6 h on the three agent cores) when all three crosses are
played on the CPU**; or ≈ 1.3 GPU-h more and no CPU lane when they are played on the card. Disk: 15 GiB free a
fork before thinning. **Hard stop:** four forks, the deciding cross and its one read, the two reported crosses and
their one report each; nothing here extends.

## 12. The slot (a proposal; the placement is the orchestrator's)

**No B1 run is paused, stopped early or re-ordered for X70, under any option.**

B1's plan (its §4, §8, §12): phase 1 (12 runs to 50M) ends ≈ L + 6.1 d (≈ 2026-11-18, late); the 50M look that
day; phase 2 ends ≈ L + 12.6 d (≈ 2026-11-25); three looks (≈ 6 GPU-h); then Kakuna, ≈ 60 h on three CPU lanes;
the decision ≈ L + 15.4 d.

**The candidate gap in the brief does not exist under B1's plan.** B1 §8 plays the 50M look ON THE CARD between
the phases (≈ 2 GPU-h), because it gates phase 2; the CPU path (≈ 27 h) is only its registered fallback. While the
look is played the card is busy, and when it has been read phase 2 starts.

| option | when | what it costs | note |
|---|---|---|---|
| **A (recommended)** | after B1's phase 2 and its three looks (≈ L + 12.9 d, ≈ 2026-11-25), while B1's Kakuna read runs on the CPU lanes and the card is otherwise idle for ≈ 60 h | nothing to B1 | the four forks (≈ 17 h) fit the window three times over. The CPU lanes are Kakuna's then, so X70's crosses go on the card right after the fourth fork (≈ 1.3 GPU-h, UNVERIFIED rate) or on the CPU after Kakuna |
| B | between B1's phases, after the 50M look is read and before phase 2's first run | B1's 100M decision slips by ≈ 17–18 h | no run is paused (the phase boundary is a clean stop). X70 reads a week sooner. Only the orchestrator can weigh that slip |
| C | opportunistic: any gap of ≥ 4.5 h in which the card would otherwise idle (a B1 HOLD awaiting a decision, the CPU fallback of the 50M look if B1 invokes it) | nothing, if the gap is real | the forks are separate chain items and need not be contiguous (§3). A fork is never started in a gap that a B1 run is waiting to use |

Option A is recommended because it is the only one that costs B1 nothing by construction. The read then lands
≈ 2026-11-26 to 11-27. Under every option the crosses wait for all four forks.

## 13. What this file does NOT do

It registers no matched-realised-entropy arm, no new swap fork, no second β or cadence, no magnet beside an
entropy bonus, no fresh-run arm, no same-pin C0 control, no class or licence on either reported cross, no
exploitability read and no Kakuna game. It does not adopt or delay X67's PASS, amends no CE-3d or B1 document,
changes no baseline and no production recipe, and holds no GO. X70 has no `main.reads` rule or `reads/` record:
its verdict is `cross/read_cross.py`'s `verdict.json`, written by `screen_rule.decide`; the reported crosses write
`report_c0.json` and `report_x67.json` (`screen_rule.report`) and no verdict.

**Built for this registration** (shipped in the same commit): the kit arm `x70` and its seeds, lever and typed
recipe row (`src/main/x35/kit.py`); its pin, stage and registration path, with `ce3d_pin_check` generalised to
judge either arm that trains at `CE3D_PIN` (`src/main/x35/batch.py`); the launch script's header line; the
no-canary-family cases (`src/main/ops/canary_decide_test.py`, `designs/training/compile_flags.md`); and
`src/main/x35/x70_test.py`.
