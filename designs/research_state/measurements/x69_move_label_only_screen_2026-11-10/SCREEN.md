# X69 — the move belief trained by its labels alone, off the pool: LIGHT SCREEN (2026-11-10)

**Tier: LIGHT SCREEN. Status: REGISTERED by this file's landing commit, before the first step, game or look.** No amendments: a change to anything below is a NEW screen in a new folder that names this one.
**This screen can only PROMOTE its candidate to a full registration or DROP it. Its result is never a verdict and
is never quoted as a claim.**

## 1. Question

A revealed opponent mon's unshown moves score 0.6 to 2.2 nats a unit BELOW the Smogon prior on teams outside the
pool (`../move_leg_offpool_2026-11-10/README.md`): the residual is an over-confident species × move table that its
own labels would shrink, held up by the published posterior's consumers (≈ 350× the label's gradient) and by the
move-latent loss's variance hinge. Does a production fork trained without one or both recover within 20M steps?
If so the repair is a flag on a fork, worth a full registration.

## 2. Eligibility

| check | value |
|---|---|
| GPU cost ≤ ~15 GPU-h | ≈ 12.3: three forks × 4.10 (the mean dose of the 20 X35 production forks, X60-T PREREG §5.1) |
| CPU cost ≤ ~50 lane-h | ≈ 1.1: eight scores × ≈ 8 min (455–467 s measured) |
| adopts nothing | yes |
| supports no goal claim, and no doc will quote its number as a finding | yes |

## 3. Arms and pin

| arm | what differs from the control (ONE lever) | name | start | pin |
|---|---|---|---|---|
| control | nothing: the five BANKED finals `rb_x35_c0_s3501`…`s3505` (sha256 `09a6a2df4850`, `7e9a9890e2b3`, `99096be8eccf`, `26835480f068`, `bb10cd96b4c5`), 0 GPU-h | C0 | P\* @280,000,170 | `f4b25238…` |
| A | `--belief-grad-mode move_label_only --allow-belief-grad-mode-change`: the consumers' gradient stops at the move head | `rb_x35_x69a_s6901` | the same | `65c7374b…` |
| B | `--move-belief-latent-coef 0`: no move-latent loss | `rb_x35_x69b_s6902` | the same | `f4b25238…` |
| AB | both | `rb_x35_x69ab_s6903` | the same | `65c7374b…` |

Pins: production `f4b25238def57900c0305bb2a3a5a8f9edaef285`; X69 `65c7374b8e17f3a14e693c084d2986da84e97e3b` (tag
`x69-pin-65c7374b`) = it + ONE commit, the mode's backport (forward bit-identical under `shaping`). Each argv is
C0's + the lever (`argv_x69*.txt`); the Training session launches it under its lease:
`python -m main.launcher --restart-interval-hours 24 --pin-commit <pin> $(cat <argv file>)`.
**The arms are not in the x35 kit**, so the arm chain cannot queue them as `arm` items.

Launch proof, by `grep -o` over `checkargs_x69*.txt` (rc 0) and `dryrun_x69*.txt` (rc 3: no lease, so the CUDA
preflight refuses; each dry run's two ✗ lines follow):

```
a_s6901: accepted by the PINNED parser : 9, unrecognized : 0, still launches; pin : 65c7374b, +20,000,000 steps, 2 ✗
b_s6902: accepted by the PINNED parser : 8, unrecognized : 0, still launches; pin : f4b25238, +20,000,000 steps, 2 ✗
ab_s6903: accepted by the PINNED parser : 10, unrecognized : 0, still launches; pin : 65c7374b, +20,000,000 steps, 2 ✗
  cuda devices : ✗ REFUSED (FATAL_CONFIG): CUDA_VISIBLE_DEVICES='' is set in the env
  ✗ DRY RUN — this command would NOT launch. Nothing was created or modified.
```

Neither prints the gradient mode: `pin_apply_check.txt` shows the pin's post-load step applying it to P\*'s
checkpoint (CPU).

## 4. Seeds, checked disjoint

Training seeds 6901, 6902, 6903. Before this file existed, `grep -rnoE "\b(6901|6902|6903)\b"
measurements/*/SCREEN.md measurements/*/PREREG*.md EXPERIMENT_BACKLOG.md src/main/x35/{kit,batch}.py` printed nothing
and `ls models | grep -cE "_s69[0-9]{2}$"` printed 0. C0's seeds are 3501–3505; the panels' is 20261109. The start
node is C0's on purpose.

## 5. Meter

**Primary: `moves_ll` on `heldout72_hpfix` (H), `forum72` (F) and `home72`**, from `python -m main.offpool_belief`
(`score --mix 0.25`, then `table --versus C0`): the log-likelihood of a revealed mon's candidate move set (its
unshown moves present, every other candidate absent), nats a unit, over the banked decisions of the seat facing
each panel team (72 teams, 144 games a panel; rows sha256 `21545e87b1fb`, `0caf6037fee3`, `5207d8c100df`), as the
mean over teams of team means. d = arm − C0 (C0: the team mean of its five finals), paired on teams, team bootstrap
(10,000 draws, seed 20261109, 95 %). One score a run: its `final_model.zip`. The bank guards run unchanged
(registry re-resolve, rows sha256, sealed-panel refusal). Reported, never deciding: `p72`; `moves_logp`; the
species and item legs; arm − prior; C0's `checkpoint_range` (its seed spread); each run's end-of-run ladder ELO.
**No strength guard: the screen plays no game.**

Early check, ≤ 30 min after each launch, in `launcher_child.log`: A and AB show `belief_grad_mode MIGRATION
'shaping' -> 'move_label_only'` and `APPLIED at runtime -> 'move_label_only'`; B and AB show no `movelatent_loss`
row. Otherwise, or on `[ModelVersion] FATAL`, the run is VOID: stopped, reported, never scored; its argv may be
relaunched once.

## 6. Decision rule (ONE look, after the last launched arm's final is scored)

1. An arm **passes** iff the lower bound of d is ≥ +0.6 on H, > 0 on F and ≥ −0.2 on `home72`.
2. **PROMOTE** iff an arm passes: the first passing one in the order B, A, AB (fewest changes first).
3. Otherwise **DROP**. A bound within 1e-6 of its threshold is not met; a VOID or unlaunched arm does not pass.
   Which arms launch is fixed before the first launch (if fewer than three: AB, then B, then A).

(+0.6 is a third of C0's 1.81-nat deficit on H, where breadth alone, N1-P s3501, gains 0.41, and where the dose
itself moves nothing: P\* − C0 s3501 is −0.017 [−0.058, +0.023]. −0.2 is half of C0's in-pool lead over the prior.
One run an arm: candidate seed noise is not measured.)

## 7. What each outcome does

- **PROMOTE:** a full registration for the promoted arm: three seeds against C0, this meter, and the game read
  this screen lacks (Kakuna, beside C0: the cut gradient may have helped play). Until it reads, the checkpoint
  plays and seeds nothing and no default changes.
- **DROP:** MF-3's move-leg gradient lever closes at this dose. A bounded residual or a trained-in prior floor
  (`design_offpool_belief.md` §7, §8) stays open for a fresh run. Reopens only with a longer dose argued from the
  DROP's own numbers.

## 8. Cost and stop

≈ 12.3 GPU-h, ≈ 1.1 CPU lane-h; each run ends at 300,000,170 steps. Past 15.4 GPU-h in total the screen stops and
is reported with no outcome. This registration launches nothing.

## 9. Result (filled AFTER the look)

**DROP (rule 3), 2026-11-11.** All three arms launched and ended clean; none is VOID. `moves_ll`, d = arm − C0,
72 teams a panel, in rule 2's order:

| arm | H (bar: lower bound ≥ +0.6) | F (bar: > 0) | `home72` (bar: ≥ −0.2) | rule 1 |
|---|---|---|---|---|
| B (`rb_x35_x69b_s6902`) | +0.046 [+0.001, +0.088]: NOT met | +0.093 [+0.047, +0.140]: met | +0.046 [+0.015, +0.077]: met | does not pass |
| A (`rb_x35_x69a_s6901`) | +0.552 [+0.480, +0.624]: NOT met | +0.571 [+0.497, +0.649]: met | +0.531 [+0.463, +0.599]: met | does not pass |
| AB (`rb_x35_x69ab_s6903`) | +0.494 [+0.427, +0.559]: NOT met | +0.546 [+0.476, +0.621]: met | +0.544 [+0.475, +0.616]: met | does not pass |

No bound is within 1e-6 of a threshold. Every d equals a by-hand recomputation from the eight score files to
1e-12 (`read/decide.py`, `read/decide.json`). Cost 12.55 GPU-h (753 min: AB 261, B 250, A 242; under the 15.4
GPU-h stop) and 11 minutes of scoring on the agent cores. The eight finals were scored at `75abf79d8` on the
registered banks; the five C0 scores of 2026-11-09 predate `moves_ll` and were not reused. Reported columns and
the FINDINGs: ledger `### 2026-11-11 · SCREEN · X69 …`; files `read/` (`table.md`, `table.json` sha256
`f4722cabe84ed150…`, `scores.sha256`, `elo_compare.txt`); the score files are
`~/gen3ai_archive/x69_move_label_only_2026-11-10/scores/`.
