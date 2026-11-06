# X65 learnability probe (2026-11-06): can a student learn S′'s departures from P\*? DESCRIPTIVE

**Status: DESIGNED. The GPU arms have not run.** This is a descriptive probe. It has no PREREG, no registration and
no verdict. No number from it is creditable as a verdict on X65, S′-LIVE or S′-HONEST. It answers finding X65-R2 of
[`x65_exit0_2026-11-05/`](../x65_exit0_2026-11-05/README.md).

**Abbreviations.**
- P\* = `rb_x26_s2601` @280,000,170, greedy.
- S′ = `r0s4_rolls@10`, S′-LIVE's search.
- H / F / P72 = the panels `heldout72_hpfix` / `forum72` / `p72`.
- A **departure** = a label row where S′'s pick (`played`) differs from P\*'s recorded greedy choice (`pi_argmax`).
  That is 37 % of label rows.
- The bank = `~/gen3ai_archive/x65_exit0_2026-11-05/bank`, manifest sha256 `d13fd55f…92b7a`.

## The question

X65's students agreed with S′ less often than P\* does: Δagree −1.26 pp. Even on rows from their own training panels,
the held-out label KL fell only about 3 %. Two explanations fit that result:

- **(U) unlearnable.** S′'s departures are not a function of what the player sees. They could come from the true-seed
  leaks S′-HONEST closes, from the search's own key noise, or from lookahead a reactive network cannot amortise.
- **(W) recipe too weak.** β = 1, ≤ 916 steps and lr ≤ 3e-5 never moved the student far enough to find out.

The probe separates the two using three pieces:
- A **learning curve** on S′'s labels at a recipe 3–10× stronger, read on three sets of rows: the rows it trains on,
  unseen deals of the same teams, and unseen teams.
- A **capacity** arm that tries to memorise S′'s departures outright.
- A **control** of the same shape whose target is learnable by construction: another policy of P\*'s own lineage.

## The meters (`python -m main.x65.learnability`; its docstring is the spec)

Each arm is scored on three sets of label rows:
- **`train`**: the rows it trains on.
- **`val`**: the training panels' inner-validation deals. These are the same teams on deals it never saw: 6,800 rows
  and 2,494 departures for fold H.
- **`read`**: every label row of the read panel. These are teams it never saw: 32,020 rows and 11,910 departures for
  H.

The `read` rows are loaded for scoring only and never batched.

At step 0 and at every checkpoint, each set reports:
- **D**: the student's agreement with S′ at departures.
- **K**: its agreement at the other rows.
- **Δagree**: the X65 gate's quantity.
- **student departure rate**: how often the student leaves P\*'s choice.
- **precision**: P(student = S′ | student ≠ P\*). A student departure gains a row only when it lands on S′'s pick,
  and costs one when the row was a keep. So **Δagree > 0 needs precision above about ½**.

The checkpoint read is the one with the best `val` Δagree, never one chosen on `read`. The last checkpoint and the
best-on-`read` checkpoint (labelled OPTIMISTIC) are reported too. Intervals bootstrap deals.

Departures are also split three ways, all reported, never decided:
- **The bank's per-row leak counters**: `attribution_miss` ≥ 1, `rewritten_moves` ≥ 1, or neither.
- **S′'s q margin** `q[pick] − q[base]` at departures, in bins < 0.03, 0.03–0.10 and ≥ 0.10. The median departure
  margin on H is 0.069.
- **The per-decision persisted-dice exposure (L2)**: a non-Rest sleep, confusion or a pending Future Sight.
  [`l2_exposure.py`](l2_exposure.py) produces it, and `--l2` reads it.

The step-0 dump is P\*'s own forward on the trainer's path. Its D must be 0 and its K must be 1, up to logit ties: that
is the path check.

## What is known before any GPU (design-time, CPU)

- **X65's students** (the registered recipe) recover 4.0–5.6 % of S′'s departures on `read`. They leave P\*'s choice on
  5.4–8.0 % of rows (X65's `forward_check.json`). Their precision is therefore about 0.26–0.28: barely above the
  chance level below.
- **The control target and the chance level** ([`control_pick.py`](control_pick.py); every 8th H label row, 4,003
  rows, of which S′ departs on 37.6 %):

  | candidate (greedy) | departs from P\* | agrees with S′ | recovers S′'s departures |
  |---|---|---|---|
  | `rb_x65_c0_s6501` @300,000,198 (P\* + 20 M steps) | 28.9 % | 54.5 % | 17.2 % |
  | `rb_x26c_s2601` @489,000,196 | 32.9 % | 53.1 % | 19.7 % |
  | `rb_x26_s2601` @200,000,120 | 34.2 % | 53.6 % | 22.0 % |
  | `rb_x26_s2601` @150,000,100 | 35.2 % | 52.7 % | 21.6 % |
  | **`rb_x26_s2601` @100,000,037 — the control** | **37.1 %** | 50.1 % | 19.8 % |

  The control target is P\*'s ancestor at 100 M steps. It departs from P\* at S′'s own rate, and it is a function of
  the observation by construction. It is a softer target than π′: at step 0 the inner-validation label KL is 0.417
  for the control and 0.716 for π′ (the smoke and X65's S_H record).
  - **Chance level.** Any policy change of this lineage lands on S′'s pick at about 20 % of S′'s departures, a precision
    of about 0.2. That is the level an unrelated policy change reaches.
- **S′'s key noise** (S′-HONEST stage 1, [`agreement.json`](../sprime_honest_stage1_2026-11-05/agreement.json)). On 94
  states, S′ run again under another key keeps its pick 72 % of the time. So part of S′'s departures is not
  reproducible even by S′. No student can learn that part. Its size on departure rows is not measured (FINDING
  X65L-3).
- **The L2 exposure**: 7,110 of S′-LIVE's 103,616 search-path decisions (6.86 %). This matches that study's
  `exposure.json`.
- **R0_H**: X65's S_H scored by this probe's meters (`dump`, CPU, at design time). It reproduces the gate on `read`:
  Δagree −1.06 pp, D 5.4 %.

  | set | rows (departures) | D | K | student departs | precision | Δagree [95 %] |
  |---|---|---|---|---|---|---|
  | `train` (its own rows) | 58,585 (21,241) | 6.9 % | 95.9 % | 7.1 % | 0.35 | −0.10 pp [−0.30, +0.11] |
  | `val` | 6,790 (2,494) | 5.3 % | 95.0 % | 6.9 % | 0.28 | −1.19 pp [−1.74, −0.66] |
  | `read` | 32,020 (11,910) | 5.4 % | 95.1 % | 7.2 % | 0.28 | −1.06 pp [−1.32, −0.79] |

  - **On its own training rows the registered recipe barely moved.** D is 6.9 % and Δagree ≈ 0. X65 is therefore no
    evidence either way on (U) against (W): that is X65-R2, confirmed by count.
  - **On `read`, D splits by S′'s q margin**: 10.6 % below 0.03 (2,332 departures), 4.6 % at 0.03–0.10 (5,815) and
    3.5 % at ≥ 0.10 (3,763). The student's few departures land on S′'s near-ties, not on its confident moves.
  - **The leak splits are flat.** D is 4.7 % with `attribution_miss`, 5.3 % with `rewritten_moves` and 5.9 % with
    neither; it is 5.1 % on L2-exposed rows (594 departures) and 5.5 % on unexposed ones. Every `read` key is covered
    by the L2 scan.

## The arms

All arms share these settings:
- Init P\*; the bank above.
- A batch of 256 label rows plus 256 anchor rows; AdamW; 200-step warm-up; clip 0.5; fp32.
- Only the policy-logit path trains, as in X65's recipe.

| arm | read fold | trains on | target | lr | β | epochs | checkpoints | steps | ≈ GPU min |
|---|---|---|---|---|---|---|---|---|---|
| **curve_h** | H | F + P72 (58,585 label rows) | π′ (S′) | 1e-4 | 0.1 | 12 | 24 | 2,748 | 22 |
| **ctrl_h** | H | F + P72, the same rows | the control's softmax | 1e-4 | 0.1 | 12 | 24 | 2,748 | 24 |
| **cap_h** | H | F alone (29,727 label rows) | π′ (S′) | 3e-4 | 0 | 24 | 24 | 2,808 | 22 |
| **curve_p** | P72 | H + F | π′ (S′) | 1e-4 | 0.1 | 12 | 24 | 2,748 | 22 |
| R0_H, R0_P | H, P72 | — (X65's S_H, S_P as exported) | | | | | | 0 | CPU |

**The GPU cost.** About 11,050 steps at X65's measured 0.33 s a step on this card (916 steps in 298 s of training),
which is 61 min. Add start-up and 24 checkpoint dumps an arm (a forward on about 97 k rows each), roughly 6–8 min an
arm. **That is ≈ 1.5 GPU-h, inside the 2 GPU-h target.**
- **The order** puts the decisive pair first: curve_h, ctrl_h, cap_h, curve_p.
- **If time runs short**, stop after ctrl_h. The first two arms alone carry readings 1 and 2.

**Why these knobs.**
- **curve_h.** lr 1e-4 is 3–10× X65's. β 0.1 is a tenth of the anchor. 12 epochs is 3× the passes. That puts the
  in-sample fit across a wide range while the anchor keeps K from collapsing. The curve's 24 points are the axis the
  reading uses, so no single recipe point carries it.
- **cap_h.** No anchor, a high lr, and half the rows for twice the epochs: the most memorisation one GPU-quarter-hour
  buys.
- **ctrl_h.** Exactly curve_h with only the target changed, so a difference between the two is the target's.

## Commands

**The GPU session** (the lease holder) runs [`gpu_arms.sh`](gpu_arms.sh) from a checkout that carries this commit:

```bash
GEN3AI_GPU_LEASE_TOKEN_FILE=<lease token file> \
  designs/research_state/measurements/x65_learnability_probe_2026-11-06/gpu_arms.sh   # all four, in order; resumable
```

Each arm is one resumable invocation. It is re-run on exit 3, a 50-min chunk:

```bash
env -u CUDA_VISIBLE_DEVICES -u GEN3AI_TEST_ALLOW_GPU GEN3AI_GPU_LEASE_TOKEN_FILE=<token file> \
  taskset -c 0-4,8-12 scripts/ops/gpu_lock.sh "$PY" -m main.x65.student probe \
  --bank ~/gen3ai_archive/x65_exit0_2026-11-05/bank \
  --bank-sha256 d13fd55f10ba649ad8e728ce1585e4203c4200148d772919eb175d3273992b7a \
  --fold heldout72_hpfix --name curve_h --lr 1e-4 --beta 0.1 --epochs 12 --evals-per-epoch 2 \
  --init models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip \
  --out ~/gen3ai_archive/x65_learnability_2026-11-06/curve_h --device cuda --budget-min 50
# ctrl_h:  the same, --name ctrl_h  --target-model models/rb_x26_s2601/checkpoints/checkpoint_100000037_steps.zip
# cap_h:   --name cap_h  --lr 3e-4 --beta 0 --epochs 24 --evals-per-epoch 1 --train-panels forum72
# curve_p: --fold p72 --name curve_p (curve_h's knobs)
```

The outputs live under `~/gen3ai_archive/x65_learnability_2026-11-06/<arm>/lr<lr>/`: `state.pt`, `ckpt_*.pt`,
`rows.npz`, `probe.json` and `eval_s<step>.npz`. Nothing goes under `models/`. `pick` and `export` refuse a probe
arm.

**The CPU scoring** runs afterwards, under rule 5b:

```bash
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=""; O=~/gen3ai_archive/x65_learnability_2026-11-06
X=~/gen3ai_archive/x65_exit0_2026-11-05; BSHA=d13fd55f10ba649ad8e728ce1585e4203c4200148d772919eb175d3273992b7a
C="nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 12"
$C "$PY" -m main.x65.learnability dump --bank $X/bank --bank-sha256 $BSHA --fold heldout72_hpfix \
  --model $X/students/s_h/model/x65_S_H.zip --out $O/r0_h --threads 3        # done at design time (the smoke)
$C "$PY" -m main.x65.learnability dump --bank $X/bank --bank-sha256 $BSHA --fold p72 \
  --model $X/students/s_p/model/x65_S_P.zip --out $O/r0_p --threads 3
$C "$PY" -m main.x65.learnability report --l2 $O/l2_exposure.json.gz --out $O/report.json \
  --arm curve_h=$O/curve_h/lr0.0001 --arm ctrl_h=$O/ctrl_h/lr0.0001 --arm cap_h=$O/cap_h/lr0.0003 \
  --arm curve_p=$O/curve_p/lr0.0001 --arm r0_h=$O/r0_h --arm r0_p=$O/r0_p
```

`$O/l2_exposure.json.gz` was built at design time by `python3 l2_exposure.py
~/gen3ai_archive/sprime_live_play_2026-11-01/anchor/u/pstar_s/cells $O/l2_exposure.json.gz`, which takes seconds.

## The readings (descriptive; the thresholds are guides, not rules)

Every reading below is at the val-selected checkpoint unless it says otherwise. Each one names the arms it reads.

**First, the path check.** In every arm's step-0 dump, D = 0 and K = 1 on all three sets, apart from logit ties. If
that fails, nothing below is read.

| # | what the probe shows | the reading |
|---|---|---|
| 1 | **curve_h / curve_p**: Δagree > 0 on `val` and on `read` (deal interval above 0); precision > ½; D on `val` and `read` rising with D on `train` along the curve | **(W) recipe too weak.** S′'s departures carry learnable signal, and X65's NOT-DELIVERED was the recipe. ExIt-0 at a stronger recipe is worth a new registration. It does not show the signal is leak-free: an observable can correlate with a leak |
| 2 | **ctrl_h** transfers: `read` precision against its own target well above ½, and Δagree > 0. **cap_h / curve_h** fit in-sample: `train` D ≥ ½ and precision > ½. **But curve_h** stays at about chance on `val` and `read` at every checkpoint: precision ≈ ctrl_h's `vs_sprime` precision (≈ 0.2–0.3) and Δagree ≤ 0 throughout | **(U) unlearnable at this scale.** The trainer and the data can learn a target of the same departure rate from the observation, and the network can memorise S′'s departures, yet those departures carry over to nothing it did not see, not even new deals of familiar teams |
| 3a | **ctrl_h** does not transfer either: `read` Δagree ≤ 0 against its own target | **Neither.** About 60 k label rows from 3 teams' panels cannot move a held-out policy toward ANY departure target at this recipe. The probe says nothing about S′'s labels. The lever is more label rows, not honest labels |
| 3b | **cap_h** cannot fit in-sample: `train` D < about ⅓ after 24 epochs with no anchor | an optimisation or capacity limit, or labels that contradict one another across near-identical rows (key noise). ctrl_h's `train` D tells which: if it fits, it is the labels |
| 3c | **curve_h** transfers to `val` (Δagree > 0) but not to `read` | **learnable but team-specific.** The breadth of teams binds, not honesty |
| 3d | **curve_h** D rises on `val` / `read`, but precision stays < ½, so K falls as fast | **X65's pattern at a larger dose.** The student learns *how often* to leave P\*'s choice but not *where* |
| 3e | transfer only at the high q-margin bin, or only on rows with no L4 counter and no L2 exposure | **Partial signal.** Low-margin departures look like key noise, and exposed-row departures look leak-driven. These splits have hundreds of rows, not thousands: reported, never decided |

**What reading 2 does and does not say about S′-LIVE's +4.60 pp.**
- **What it says.** S′'s advantage is not distillable from what the player sees at ~60 k labels. That is
  evidence in favour of the leak explanation.
- **What it does not say.** It does not separate the three sub-causes of (U):
  - **the leak** (the truth);
  - **S′'s key noise**;
  - **genuine lookahead** a reactive network cannot amortise.

  The third can win games for S′ without being learnable. The cleanest test is S′-HONEST's replication itself, and the
  same probe on honest labels (an ExIt-0H bank).
- **Ahead of that replication.** Reading 2 raises the prior that S′-HONEST's gain shrinks. Reading 1 lowers it.

## FINDINGs and limits

- **X65L-1 (the bank's exposure columns).** Per row, the bank records only L4's counters (`attribution_miss`,
  `rewritten_moves`).
  - `leak_resid` (`counters:seen`) and the `partial` list are the preset's static lists, the same on every record. As
    in X65's REVIEW R-m4, they split nothing.
  - Speed-tie exposure (L1a) exists only for 400 sampled decisions (`sprime_honest_2026-11-05/speed_pin.json`), about
    45 of them departures on any one read panel: too few to split, so it is not used.
  - L2 is rebuilt per key by [`l2_exposure.py`](l2_exposure.py), a verbatim copy of that study's `scan` that returns
    the full key. It reads our side's protocol stream to its end, so it is outcome-blind by construction only: it uses
    no result field.
  - L1 (the true seed in the prefix) touches every decision, so **no bucket is leak-free**. The splits compare more
    exposure with less, never exposure with none.
- **X65L-2 (the trainer).** `main.x65.student` gains a `probe` subcommand. The registered `train` / `pick` / `export`
  path is unchanged:
  - The defaults are the registered constants.
  - `pick` refuses a probe state.
  - The step-0 and checkpoint dumps, and the read panel's rows, exist only under `probe`.

  `main.x65.learnability` holds the probe's rows, dumps and report. The tests are in
  `src/main/x65/learnability_test.py`; its slow test trains a probe arm and a control end to end.
- **X65L-3 (not built: the noise ceiling).** A CPU arm could re-run S′ under another key on about 1,000 bank departure
  rows. At 6.4 CPU-s a decision that is about 2 CPU-h. It would put a ceiling on D that no observation-function student
  can pass, and so split key noise from leak plus lookahead. It needs each row's search log rebuilt from the bank
  (the bank builder's `honest_cut`). It is outside this unit.
- **X65L-4.** The control is one policy of P\*'s own lineage, and it shares P\*'s function class. That makes it an
  *easier* target than an arbitrary function of the observation. A pass of ctrl_h is a lower bound on the recipe's
  reach, not a proof that every observable target is learnable.
- **X65L-5.** The GPU-minute figures project X65's own measured step time. The dump cost on the GPU is estimated, not
  measured. The CPU smoke below proves that the commands run, not how long they take.

## The CPU smoke (design time)

The smoke ran from this commit's worktree, CPU only, at nice 19 on cores 5-7,13-15 under `mem_cap`, with outputs in
`~/gen3ai_archive/x65_learnability_2026-11-06/smoke/`:
- **ctrl_h's knobs** with `--epochs 1 --evals-per-epoch 100 --limit-steps 3`, which also exercises the control-target
  path. Exit 3 (the step limit). 21 min CPU, peak 5.2 GB.
- **cap_h's knobs** the same way. Its first chunk was stopped while queued for a heavy slot and then resumed from
  `state.pt`, so the resume path ran too. Exit 3.
- **`report`** on both arms with `--l2`. Exit 0.

**What the smoke showed.**
- **The path check is exact.** At step 0, D = 0 and K = 1 on every set of both arms, with no logit ties on these rows.
- **The sizes match the plan.** cap_h has 29,727 / 3,465 / 32,020 rows (forum72 only), and ctrl_h has
  58,585 / 6,790 / 32,020.
- **The read rows are the read panel's alone**, and checkpoint dumps are written at each checkpoint.
- **The control re-defines the departures.** P\*'s ancestor departs on 36.6 % of `train`, 36.9 % of `val` and 36.4 %
  of `read`.

**R0_H's `dump`** ran in the short lane: 3 min, peak 4.5 GB.

**Disk.** A GPU arm keeps 24 `ckpt_*.pt` of 30 MB, a 49 MB `state.pt` and 25 dumps of 1.4 MB: about 0.8 GB an arm,
or about 3.2 GB for all four.
