# X66 ExIt-0H: the read (2026-11-10) — NOT-DELIVERED

**Class: NOT-DELIVERED** (PREREG §1, §4). The delivery gate's lower bound on Δagree is −7.31 pp, not above 0. On the
panel each student never saw, it agrees with the honest search's pick LESS often than P\* does. No Kakuna game of this
read is played. The registration is [`PREREG.md`](PREREG.md) (`a2dbc9616`) with Amendment 1 (the bank). The roadmap
item is MF-1 (`../../EXPERIMENT_BACKLOG.md` §2.0).

**Terms.** P\* is `rb_x26_s2601` at 280,000,170 steps. S′h is the honest one-ply search `r0s4_honest_inc@10` that
S′-HONEST played (its worlds read only what the live player knows). A student is P\* fine-tuned offline toward S′h's
piKL targets on two panels' banked decisions, with a weak KL anchor to P\*. The panels are H (`heldout72_hpfix`), F
(`forum72`) and P72 (`p72`).

**What it licenses (§1).** "At this recipe and ≈ 113k honest labels a fold, a P\* student does not learn S′h's choices
on unseen teams." The §7 readings below say which pattern.

**What it does NOT license (§1; the pre-commitments of §0 and §8.3 are binding).**
- **Evidence for the scale bet (MF-5).** S′h's labels are only about a third reproducible under another search key,
  and the game gain being distilled is one panel's (P72 +4.69 pp [+2.00, +7.37]; H +1.32 [−1.39, +4.00]; F −0.005
  [−2.98, +2.71]). A failure to distil cannot tell "the network is too small" from "the target is mostly noise, and
  its signal lives on one panel". The readings below are about THIS recipe and THESE labels, never a capacity result.
- **Planning MF-1b (ExIt-1)**, which needs its own case, evidence and registration whatever X66 read.
- Anything about larger networks, more labels, another recipe or in-loop distillation.

**What was not built or written, by the registration.** Amendment 2 (the students in the read specs) and the
dose-matched student are DELIVERED-only; the three read specs keep their PENDING values, and nothing was planned.

## The gate (PREREG §4)

The gate scores every label row of each student's read panel. A label row is one of S′h's verified searched
decisions on the panel the student never trained on.
- a(student) = 1 when the exported student's greedy action equals S′h's pick (`played`);
- a(P\*) = 1 when P\*'s recorded greedy action (`pi_argmax`) equals `played`;
- Δagree is the mean of a(student) − a(P\*).

The interval is a bootstrap by deal within each panel: B = 4,000, `default_rng(20261105)`.

| panel | student | label rows | deals | agree, student | agree, P\* | Δagree |
|---|---|---|---|---|---|---|
| H | S_H | 60,004 | 1,695 | 0.5623 | 0.6271 | **−6.48 pp** |
| F | S_F | 65,996 | 1,681 | 0.5576 | 0.6292 | **−7.16 pp** |
| P72 | S_P | 62,297 | 1,696 | 0.5521 | 0.6271 | **−7.50 pp** |
| **pooled** (mean of the three) | | 188,297 | | | | **−7.05 pp [−7.31, −6.78]** |

- **LB = −0.07311**, so DELIVERED is false. It is not at the boundary: |LB| ≫ 0.0005.
- **By hand.** [`delivery_by_hand.py`](delivery_by_hand.py), with loops of its own, gives every Δagree, the pooled
  Δ, the LB and DELIVERED equal to the gate's file. See [`record/delivery_by_hand.json`](record/delivery_by_hand.json):
  `agrees: true`, no problem listed.
- **The gate's file** is [`record/delivery.json`](record/delivery.json) (sha256 `2cbc21bb…31bc`); it names the bank
  `0812768c…5344` and the three exported zips.
- The 188,297 scored rows are every label row of the bank (Amendment 1's count): no row was dropped by the scorer.

**No VOID condition of §11 occurred.** The bank and the students are on their registered sha256s (the kit's
preconditions PASS, [`record/gpu_driver.log`](record/gpu_driver.log)); `student.split` raised no read-panel row in
training or validation; the picks read `val` alone; all three exports are bit-identical at the play pin; the gate's
file equals the by-hand recompute. The game-level conditions do not apply: no game was played.

### The picks and the exports

The pick is the checkpoint after step 0 with the highest inner-validation Δagree (the training panels' withheld
deals), ties to the earlier. Every candidate's `val` Δagree is negative, so each pick is the least negative one.

| student | seed | pick | `val` Δagree, 95 % | picked policy sha256 | exported zip sha256 | bit-identical at the play pin |
|---|---|---|---|---|---|---|
| S_H | 6611 | checkpoint 4, step 1,120 | −5.60 pp [−6.52, −4.69] | `47ff4192…9494` | `3ddc6955…75fb` | yes |
| S_F | 6612 | checkpoint 14, step 3,210 | −6.47 pp [−7.30, −5.65] | `45386ccb…2c3c` | `25b63f54…8650` | yes |
| S_P | 6613 | checkpoint 5, step 1,323 | −5.12 pp [−6.20, −4.09] | `c802ec92…61d0` | `3db40d28…d73f` | yes |

The export check ran each zip's logits on the 64-row fixture (sha256 `bb4cf83d…a5db`) through the trainer at the side
pin and through the play pin `e46e3ddf8`'s bare loader: equal bit for bit, no problem listed
([`record/export_s_h.json`](record/export_s_h.json), [`_s_f`](record/export_s_f.json),
[`_s_p`](record/export_s_p.json); the picks are [`record/pick_s_h.json`](record/pick_s_h.json),
[`_s_f`](record/pick_s_f.json), [`_s_p`](record/pick_s_p.json)).

## Reported (§7), never decided

### On the gate's rows

| panel | student agrees at S′h's departures (rows) | agreement with S′h where S′h kept P\*'s pick | pre-turn-10 agreement with P\* (rows) | mean \|V_student − V_P\*\| |
|---|---|---|---|---|
| H | 15.7 % (22,376) | 80.3 % | 75.3 % (16,714) | 0.051 |
| F | 16.3 % (24,469) | 79.0 % | 74.2 % (16,990) | 0.045 |
| P72 | 15.6 % (23,229) | 78.8 % | 74.4 % (16,695) | 0.067 |

How to read it:
- S′h leaves P\*'s greedy choice on ≈ 37 % of its searched decisions (a departure). P\* agrees with S′h at none of
  the departures, by definition, and at all of the rest.
- Each student recovers ≈ 16 % of the departures and loses ≈ 20–21 % of the decisions where S′h kept P\*'s choice.
  There are 1.7 times as many of the latter, so the net is ≈ −7 pp.
- The honest bank carries no leak counter, so the gate's exposure split is empty: every row is in "neither".
- The students moved far from P\* everywhere: before turn 10, where no decision was searched or labelled, they pick
  P\*'s action on only 74–75 % of rows (X65's students: 92.5–95.5 %), and the value head's output drifts by
  0.045–0.067 although its own parameters are frozen (X65: 0.002–0.004). See FINDING X66-R4.

### The learning curves ([`record/learnability_report.json`](record/learnability_report.json))

Each student's own dumps at step 0 and at its 24 checkpoints are scored on three sets of label rows:
- **`train`**, the rows it trains on (109k–115k);
- **`val`**, the training panels' withheld deals: familiar teams, unseen deals (≈ 13k);
- **`read`**, the held-out panel: unseen teams (60k–66k).

The meters are X65's probe's:
- **D** is the student's agreement with S′h at S′h's departures;
- **K** is its agreement with S′h on the other rows;
- **dep** is how often the student leaves P\*'s choice;
- **precision** is P(student = S′h's pick | student ≠ P\*'s pick). Δagree > 0 needs a precision above about ½.

**The path check is exact.** At step 0 every student has D = 0 and K = 1 on all three sets.

**At the val-selected checkpoint** (Δagree in pp; 95 % deal-bootstrap interval, B = 2,000):

| student, step | `train` D / K / dep / precision / Δagree | `val` D / K / dep / precision / Δagree | `read` D / K / dep / precision / Δagree |
|---|---|---|---|
| S_H, 1,120 | 0.184 / 0.816 / 0.252 / 0.271 / −4.74 | 0.166 / 0.809 / 0.247 / 0.254 / −5.60 [−6.52, −4.69] | 0.157 / 0.803 / 0.250 / **0.234** / **−6.48 [−6.92, −6.05]** |
| S_F, 3,210 | 0.241 / 0.834 / 0.262 / 0.344 / −1.39 | 0.161 / 0.800 / 0.257 / 0.235 / −6.47 [−7.30, −5.65] | 0.163 / 0.790 / 0.263 / **0.231** / **−7.16 [−7.61, −6.71]** |
| S_P, 1,323 | 0.191 / 0.817 / 0.252 / 0.281 / −4.42 | 0.173 / 0.814 / 0.250 / 0.260 / −5.12 [−6.20, −4.09] | 0.156 / 0.788 / 0.264 / **0.220** / **−7.50 [−7.97, −7.02]** |

**At the last checkpoint** (12 epochs) and at the best-on-`read` checkpoint (OPTIMISTIC: chosen on the read panel):

| student | last: `train` D / precision / Δagree | last: `val` Δagree | last: `read` D / K / precision / Δagree | best on `read` (OPTIMISTIC) |
|---|---|---|---|---|
| S_H | 0.294 / 0.402 / +1.38 (step 5,376) | −6.27 [−7.10, −5.42] | 0.170 / 0.784 / 0.232 / −7.19 [−7.67, −6.75] | step 2,912: −6.30 [−6.78, −5.86] |
| S_F | 0.293 / 0.405 / +1.54 (step 5,136) | −7.14 [−8.07, −6.27] | 0.167 / 0.780 / 0.226 / −7.64 [−8.09, −7.19] | step 1,498: −6.50 [−6.94, −6.04] |
| S_P | 0.294 / 0.413 / +1.75 (step 5,292) | −6.33 [−7.31, −5.29] | 0.167 / 0.775 / 0.221 / −7.87 [−8.34, −7.39] | step 1,764: −6.95 [−7.40, −6.49] |

**Over every checkpoint of the three students:**
- `val` and `read` Δagree are ≤ 0 throughout. The maximum is 0, at step 0.
- The first half epoch does most of the damage: `read` Δagree is −8.19 / −6.92 / −7.73 pp at the first checkpoint,
  and never recovers past −6.30 / −6.50 / −6.95 pp.
- The best `read` precision is 0.241 (S_H), 0.236 (S_F) and 0.231 (S_P); the best `val` precision is 0.257, 0.237
  and 0.260.
- On its own `train` rows a student is negative for the first ≈ 9 epochs and ends at +1.4 to +1.8 pp, with `train` D
  ≈ 0.29 and precision ≈ 0.40–0.41. So the val-selected checkpoints do not fit even the rows they were trained on.

**`read` D by S′h's q margin at the selected checkpoint** (how much better S′h scored its pick than P\*'s choice):

| student | margin < 0.03 (departures) | 0.03–0.10 | ≥ 0.10 |
|---|---|---|---|
| S_H | 0.249 (4,240) | 0.145 (10,399) | 0.122 (7,737) |
| S_F | 0.266 (4,877) | 0.145 (11,806) | 0.127 (7,786) |
| S_P | 0.249 (4,768) | 0.140 (11,012) | 0.119 (7,449) |

The students recover S′h's near-ties most often and its clearest departures least often.

### The probe's readings table, applied (PREREG §7; the table is `../x65_learnability_probe_2026-11-06/README.md`'s)

The readings are descriptive. They are stated for this recipe and these labels and are never a capacity result (§8.3).

- **Reading 2's curve row matches, on all three students.** "Stays at about chance on `val` and `read` at every
  checkpoint: precision ≈ 0.2–0.3 and Δagree ≤ 0 throughout": `read` precision 0.21–0.24, `val` 0.21–0.26, Δagree ≤ 0
  at every checkpoint. Its control row ("a same-rate target transfers") is the probe's `ctrl_h`, not re-run here
  (+11.70 pp [+11.05, +12.32] at this recipe).
- **Reading 2's in-sample row is NOT shown for the honest labels.** It asks for `train` D ≥ ½ and precision > ½. The
  students' `train` D ends at 0.29 with precision 0.40–0.41, under the anchor. On the leaky labels that row rested on
  the probe's capacity arm (no anchor, `train` D 0.688), which the registration did not re-run. So "the network can
  memorise S′h's departures" is unmeasured on honest labels (FINDING X66-R5).
- **Reading 3d's pattern shows.** "D rises on `val` / `read`, but precision stays < ½, so K falls": on `read`, D goes
  0.149 → 0.170 (S_H), 0.145 → 0.167 (S_F) and 0.152 → 0.167 (S_P) from the first to the last checkpoint, while
  precision stays 0.21–0.24 and K ends at 0.78. The students leave P\*'s choice on ≈ 25–26 % of rows (S′h: ≈ 37 %),
  and where they leave it is close to unrelated to where S′h did.
- **Reading 1 does not match:** no student has Δagree > 0 on `val` or `read` at any checkpoint.
- **Reading 3c does not match:** `val` (familiar teams, unseen deals) is as negative as `read`; nothing transfers
  even to new deals of the training panels' teams.
- **Reading 3a** is excluded by the probe's `ctrl_h`, as registered.
- **Reading 3b cannot be applied:** it is defined on a no-anchor capacity arm, which this read has none of.
- **Reading 3e:** no bin transfers. D is highest at the lowest q margin, the opposite of "transfer only at the high
  margin bin". Reported only.

### Honest against leaky, at one recipe (DESCRIPTIVE, unpaired: other deals, ≈ 2× the label rows)

| arm | labels | selected step | `read` Δagree | `read` D | `read` precision | last checkpoint `read` Δagree |
|---|---|---|---|---|---|---|
| S_H (this read) | honest, S′h | 1,120 | −6.48 pp [−6.92, −6.05] | 0.157 | 0.234 | −7.19 pp |
| curve_h (the X65 probe) | leak-inclusive, S′ | 114 | −4.79 pp [−5.21, −4.38] | 0.119 | 0.23 | −6.50 pp |
| S_P (this read) | honest, S′h | 1,323 | −7.50 pp [−7.97, −7.02] | 0.156 | 0.220 | −7.87 pp |
| curve_p (the X65 probe) | leak-inclusive, S′ | 114 | −5.16 pp [−5.60, −4.71] | 0.113 | 0.22 | −8.10 pp |

At the same recipe the honest students' precision equals the leaky arms' (0.22–0.23): on this unpaired comparison,
honest labels were no more predictable from the observation than leak-inclusive ones. The selected Δagree is not
comparable row for row: the probe's arms were selected at their first checkpoint, the honest students later, with
larger D.

### The noise ceiling (§8.2)

| panel | student's `read` agreement with S′h | P\*'s agreement with S′h | S′h's agreement with itself under another key |
|---|---|---|---|
| H | 56.2 % | 62.7 % | 61.8 % [54.0, 69.0] (157 decision pairs, stage 1b; [`noise_ceiling.json`](noise_ceiling.json)) |
| F | 55.8 % | 62.9 % | the same figure |
| P72 | 55.2 % | 62.7 % | the same figure |

P\* already agrees with S′h's pick about as often as a second run of S′h does when its search key changes; the
students fell 6 to 7 points below both. Self-agreement is not a strict upper bound (a student that named each
decision's most likely pick would exceed it), and it is 157 pairs from other deals, so this is a description
(FINDING X66-R6).

### The meters cross-check

The gate scores the exported zip at the side pin; the report scores the trainer's own dumps. For all three students
the gate's Δagree equals the report's `read` Δagree at the selected checkpoint to 1e-16, and the gate's agreement at
departures equals the report's `read` D. Two code paths, one answer.

## What ran

**The GPU step** (the training session's lease; [`record/gpu_driver.log`](record/gpu_driver.log)).
`gpu_students.sh` ran 2026-11-10 12:52–15:39 PST: preconditions PASS (side pin `c0b95914e` on engine E6, the bank,
P\*'s sha256), then S_H, S_F and S_P in order. Each student took one 50-minute chunk that ended on its budget (exit 3,
as registered) and a second chunk that finished (exit 0).

**The CPU gate** ([`record/gate.log`](record/gate.log), sha256 `765c72d1…853a`). [`gate.sh`](gate.sh) ran unedited
(sha256 `2d778cf8…9c8e`) 2026-11-10 15:40–18:00 PST, with no GPU, at nice 19 on the 5+3 split's agent cores under
`mem_cap`: preconditions (both pin worktrees at their pins, both stamping E6), then per fold the pick, the export at
the play pin and the delivery score, then the gate, the by-hand recompute and the learnability report. Every step
exited 0. Its last line: `DONE: {'pooled': -0.0704…, 'lb': -0.0731…, 'delivered': False, 'boundary': False}`.

**Pins.** Training, picking and scoring at the side pin `c0b95914e239` (worktree `x66_pin`); the export's other
side at the play pin `e46e3ddf8`; engine E6 = `port=f0fe71f6c54757b2/95;env=c0b7c649acfb48d8/130` on both.

**What did not run.** No Kakuna game, no plan, no `check-spec`, no `decide`, no dose-matched student, no placebo.

**State** (outside every checkout): `~/gen3ai_archive/x66_exit0h_2026-11-09/` holds `bank/`, `students/s_{h,f,p}/`
(24 checkpoints, 25 dumps, `pick.json`, `model/x66_S_<X>.zip`), `delivery/<panel>.npz`, `delivery.json`,
`delivery_by_hand.json`, `learnability_report.json` and `gate/gate.log`.

## Against the GUESSES (§13)

P(DELIVERED) was guessed at ≈ 0.3; the read is NOT-DELIVERED, and by a wide margin (the guess gave no size for that
branch). The guess's stated reason held: "at this exact recipe the leaky labels read ≤ 0 at every checkpoint", and so
did the honest ones.

## FINDINGs

- **X66-R1 (`decide` was not run).** X65's read also ran `python -m main.reads decide` on its planned, unplayed
  look-1 state, as a second statement of the class. X66's specs keep PENDING students and pin, because Amendment 2
  is DELIVERED-only, so no state can be planned. PREREG §1 classes NOT-DELIVERED from the gate's file alone; the
  second statement here is the by-hand recompute.
- **X66-R2 (the gate's wall time was queueing).** The gate took 2 h 20 min. Its twelve steps each waited 4–15 minutes
  for a heavy slot while a training run and play lanes were live; the compute was seconds to ≈ 11 minutes a step.
- **X66-R3 (a bare interpreter in the gate's last line).** `gate.sh` prints its `DONE` line with a bare `python3`
  (base miniconda), a JSON read only. Nothing scored runs there. The script is a frozen record and was not edited.
- **X66-R4 (the recipe moved the whole policy, not only the searched decisions).** Before turn 10 the students pick
  P\*'s action on 74–75 % of rows, and the value output drifts ≈ 15–20× X65's although only the policy-logit path
  trains (the shared trunk moved). The anchor at β = 0.1 did not hold the greedy choice to P\*'s where no label
  exists. Had a student been delivered, its games would have differed from P\*'s well beyond the searched decisions.
- **X66-R5 (in-sample memorisation is unmeasured on honest labels).** The registration dropped the capacity arm on
  the argument that "the students' own `train` curves report the same thing". They do not: `train` D ends at 0.29
  under the anchor. Whether a network of P\*'s size can memorise S′h's departures, and so whether reading 3b (labels
  that contradict one another across near-identical rows, which key noise predicts) applies, is open.
- **X66-R6 (P\* is already at the labels' self-agreement).** P\*'s 62.7 % agreement with S′h's pick is inside the
  interval of S′h's 61.8 % self-agreement. With labels this noisy, the room above P\* on argmax agreement with a
  single-key S′h pick is unmeasured and may be small, which would make Δagree a weak meter for this target. This is
  a description from 157 pairs; it licenses nothing about scale (pre-commitment 1). A de-noised target (several
  keys a decision) is PREREG FINDING X66-F3's separate lever, not this read's.
- **X66-R7 (nothing here bears on S′h's game gain).** S′-HONEST's +2.00 pp against Kakuna is a game result of the
  search and stands. This read measured agreement with its picks, not their value.

## Files

| file | what |
|---|---|
| [`PREREG.md`](PREREG.md) | the registration: question, classes, bank, recipe, gate, deals, statistic, power, cost, launch, VOID |
| [`record/`](record/) | the read's record: `delivery.json`, `delivery_by_hand.json`, `learnability_report.json`, the three picks and exports, `gate.log`, `gpu_driver.log` |
| [`bank_build.sh`](bank_build.sh) | builds the honest bank at the side pin, outcome-blind, in ≤ 44-min resumable chunks (CPU) |
| [`gpu_students.sh`](gpu_students.sh) | the GPU kit for the lease holder: preconditions, then S_H, S_F, S_P (resumable) |
| [`gate.sh`](gate.sh) | the read operator's CPU steps: pick, export at the play pin, delivery score and gate, by hand, the report |
| [`delivery_by_hand.py`](delivery_by_hand.py) | the gate recomputed with loops of its own (X65's) |
| [`read_spec_look1.json`](read_spec_look1.json), [`read_spec_look1_fresh.json`](read_spec_look1_fresh.json), [`read_spec_look2.json`](read_spec_look2.json) | the Kakuna read's specs, never filled (PENDING: the students, the pin), written by [`make_specs.py`](make_specs.py); unused on NOT-DELIVERED |
| [`power.py`](power.py) → [`power.txt`](power.txt) | the two-look power of the Kakuna read (not played) |
| [`smoke_train.py`](smoke_train.py) | Amendment 1's CPU smoke of the x66 trainer on the registered bank (3 steps) |
| [`noise_ceiling.py`](noise_ceiling.py) → [`noise_ceiling.json`](noise_ceiling.json) | S′h's self-agreement under another key (stage 1b's 157 pairs): 61.8 %; a third of its departures reproduce |
