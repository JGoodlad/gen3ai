# CE-3d — the trailing-magnet screen: PRE-REGISTRATION (2026-11-11)

**Status: REGISTERED before any launch, any game and any read. NOT LAUNCHED** (no `rb_x35_ce3dmag_*` or
`rb_x35_ce3dent_*` run exists). The launch is the orchestrator's GO through the Training session's kit (arms
`ce3d_mag`, `ce3d_ent`); the hand-off is [`chain_items_ce3d.json`](chain_items_ce3d.json) (§7). Nothing below changes
after the first fork launches except by a dated amendment, written before the deciding cross is played, that says why.

**The question.** Does pulling the policy toward a trailing copy of itself (a "magnet", refreshed every 2M steps)
make a plateau policy stronger than the same policy trained on without the pull, when both carry the same weight on
the entropy bonus's gradient?

**The tier is FULL** (`../../README.md` § Registration tiers), not a light screen: six forks cost ≈ 26 GPU-h against
the light tier's ≈ 15, the crosses cost ≈ 63 CPU lane-h against its ≈ 50, and the arms run at a NEW pin carrying
code that has never trained on the GPU. Its verdict still adopts nothing (§10).

Abbreviations: pp = percentage points. P\* = baseline `pstar_plateau` (`rb_x26_s2601` @280,000,170, sha256
`814823c5…`). C0 = P\* + 20M with no lever at the production side pin `f4b25238` (`rb_x35_c0_s<seed>`), the banked
plateau control. `ent_coef` = the coefficient on the entropy bonus in the PPO loss. KL = the Kullback–Leibler
divergence. π = the learner's policy over a row's legal actions; π_mag = the magnet's. β = `--magnet-kl-coef`.
H(π) = the policy's entropy. T2 = the Rust-side inference server self-play runs on; R1 = the learner's compiled
update step. K9(b) = the behaviour-policy check (does the learner's log-probability of each played action match the
one stored at play time). bf16 / fp32 = 16-bit / 32-bit floating point. SD = standard deviation; df = degrees of
freedom. X67 = the entropy screen ([`../x67_entropy_screen_2026-11-09/`](../x67_entropy_screen_2026-11-09/README.md)).

## 1. Why now, and what X67's PASS changes

The lever was built on 2026-11-10 (task T56, `251bbfb87`; [`../../../training/magnet_kl.md`](../../../training/magnet_kl.md)):
`loss += β · KL(π ‖ π_mag)`, from Magnetic Mirror Descent (Sokota et al. 2023) and R-NaD (Perolat et al. 2022). It is
off by default and no run has trained with it. The GPU has a gap of ≈ a day after X69 while BOUNDARY-1 waits on its
re-pin.

**The magnet arm is not one lever against C0.** Over a row's legal actions

```
KL(π ‖ π_mag) = −H(π) − Σ_a π(a) · log π_mag(a)
```

so β = 0.05 beside `ent_coef` 0.05 is two things: 0.05 MORE weight on the entropy bonus's own gradient (0.10 in
all), and a cross-entropy pull toward the magnet. The screen therefore trains an **entropy-matched control**
(`--ent-coef 0.1`, no magnet), and the verdict is the magnet arm against THAT arm: the pull alone.

**How the arms sit beside X67's PASS** (read 2026-11-11: three forks at `ent_coef` 0.0125 beat the eight C0 finals,
D = +1.97 pp [+1.35, +2.59], every fork LR-MOVED, end entropy 0.68 against 1.01 nats):
- X67 moved the entropy weight DOWN from 0.05 and gained. Both CE-3d arms sit at an effective weight of **0.10**,
  twice C0's, the other side of 0.05. X67's direction predicts that both arms lose to C0; that is an extrapolation
  through one measured point and is **UNVERIFIED**. The two REPORTED crosses against C0 (§5) measure it.
- So the deciding contrast isolates the pull at an entropy weight no recipe uses or is heading to. A PASS says the
  pull helps THERE; a NOT-SHOWN says nothing about a magnet at 0.0125 or about the swap (`--ent-coef 0
  --magnet-kl-coef 0.05`, the magnet in place of the uniform prior, entropy weight 0.05). Neither is registered here.
- X67's PASS is not adopted (`ent_coef` stays 0.05 until the owner's word), and this screen neither waits on that
  decision nor informs it.

**Prediction** (GUESS; it decides nothing): the deciding D ∈ [−1.5, +1.5] pp; control against C0 between −3 and
0 pp; the control's end entropy above C0's 1.01 nats. Nine refreshes in 20M steps is a short horizon for a
regulariser whose claimed effect is on a slow cycle.

## 2. The arms

| | **MAGNET** (kit arm `ce3d_mag`) | **CONTROL** (kit arm `ce3d_ent`) | C0 (banked; reported columns) |
|---|---|---|---|
| in plain words | P\* trained on, pulled toward a copy of itself that is replaced every 2M steps | P\* trained on with the entropy bonus doubled and no pull | P\* trained on, nothing changed |
| run | `rb_x35_ce3dmag_s<seed>` | `rb_x35_ce3dent_s<seed>` | `rb_x35_c0_s<seed>` |
| start | P\* @280,000,170 | the same file | the same file |
| lever, typed | **`--magnet-kl-coef 0.05`** (`kit.CE3D_MAG_LEVER`) | **`--ent-coef 0.1`** (`kit.CE3D_ENT_LEVER`) | none |
| `ent_coef` | 0.05, inherited from P\*'s record | 0.1 | 0.05 |
| weight on the entropy gradient | 0.05 + 0.05 = **0.10** | **0.10** | 0.05 |
| the pull toward the magnet | β 0.05; magnet = P\*'s weights at the fork, replaced at the first update boundary at or after each 2,000,000 steps of age (`--magnet-refresh-steps`, untyped: the pin's default) | none | none |
| argv | C0's, token for token, + the lever before `--seed` ([`argv_ce3d_mag_s7301.txt`](argv_ce3d_mag_s7301.txt)) | the same + its lever ([`argv_ce3d_ent_s7311.txt`](argv_ce3d_ent_s7311.txt)) | its recorded `original_command` |
| dose | +20,000,000 steps (`--steps 300000170`) | the same | the same |
| pin | **`333a73ddda7829b9ccdf9e17ef04624eb40edeba`** (tag `ce3d-pin-333a73dd`, §4) | the same pin | `f4b25238def57900c0305bb2a3a5a8f9edaef285` |
| seeds | **7301, 7302, 7303** (new) | **7311, 7312, 7313** (new) | 3501–3508 (banked) |
| GPU | ≈ 13 GPU-h (3 × ≈ 4.2–4.4) | ≈ 12.3 GPU-h (3 × ≈ 4.1) | 0 here |

The two new arms' argvs differ in four tokens: the lever's flag and value, the seed and the run name
(`ce3d_test.py`). `checkargs` prints each arm's lever as its ONE typed recipe difference from its parent:
`magnet_kl_coef 0.05 (TYPED) production: 0.0`, `ent_coef 0.1 (TYPED) production: 0.05`.

**Refreshes.** A fork's magnet is set at 280,000,170 with age 0 and falls due at each 2M of age, so **nine** refreshes
land inside the run (≈ 282M … 298M, each up to one rollout late); the tenth falls due at the end, where the last
update has no boundary after it (`magnet_kl.md` §5).

**Not registered:** the swap arm, a second β, a second cadence, a magnet under `ent_coef` 0.0125.

## 3. Seeds, dose, one look

- **Seeds 7301–7303 and 7311–7313** are no registered sibling's: no spec, kit arm, design doc or run directory names
  them (searched `src/`, `designs/`, `scripts/` and `models/` on 2026-11-11; `ce3d_test` pins them against every
  other kit arm and against each other). No magnet seed is paired with a control seed.
- **Dose.** Both arms inherit C0's effective batch 65,536, 10 epochs and learning-rate controller from P\*'s record
  (the dry runs list them). C0's finals read `lr_median` 0.0001005 (`main.dose`). §8 checks the dose at the end and
  labels a moved rate.
- **One look.** The stage is `CE3D` (`batch.CE3D_STAGES`): three seeds an arm, no look 2, no extension, no substitute
  seed. `kit stage-check` is PENDING until this file exists on the kit's checkout.
- **Order.** The chain alternates the arms (magnet 7301, control 7311, magnet 7302, …), so a drift over the ≈ 26
  hours falls on both alike.

## 4. The pin

**Both arms run at `333a73ddda7829b9ccdf9e17ef04624eb40edeba`**: the production side pin `f4b25238` + ONE commit,
annotated tag **`ce3d-pin-333a73dd`** on origin (`batch.CE3D_PIN`; `kit pin-check` requires the parent to be the
side's written pin and the commit to change EXACTLY the 44 paths of `batch.CE3D_PIN_PATHS`, executed below).

**Why not main, and why not the plain side pin.** Main's fixes move P\*'s inputs on 96 % of rows, so a fork of P\* at
main is confounded with a 20M re-adaptation (X67 PREREG §4). `f4b25238` has no magnet. The commit carries:
- **the magnet lever** as main built it (`251bbfb87`): `agents/training/magnet.py`, its seams in the collector, the
  buffer, R1, the checkpoint commit and the parser, and its tests;
- **the T2 near-tie judge** (`3260d7ae0`), because it applied cleanly and changes no training number: it runs only
  after a parity judgement has already failed, and turns a near-tie flip (X60-T s3501's FATAL class) from a crash
  into a pass. Evidence at the pin: its two `dtype=` arguments are no-ops in float32 (a probe over 594 + 270 calls),
  and the learner golden is bit-identical in bf16 and fp32.

**The config version stays 161.** The two fields (`magnet_kl_coef`, `magnet_refresh_steps`) are un-versioned
additions with defaults 0.0 and 2,000,000; every run at this pin records them, the control included (0.0). Two
consequences, both executed: the plain side pin REFUSES to load a config that carries them, so **every cross is
played at the CE-3d pin**, which loads C0's finals as `f4b25238` does; and main loads a CE-3d config keeping the
recorded values.

**Both arms at one pin** so the deciding contrast shares its code. With β = 0 the learner takes R1's unchanged
nine-argument call (the learner golden is bit-identical at the pin), so the control is C0's program with a different
scalar.

**Tested at the pin, CPU only** (2026-11-11, rule 5b, no lease):
- the routine tier: 12,721 passed, 2 failed, neither the commit's: one heavy-slots test that fails only when the
  suite itself runs inside a memory-capped slot (17 passed alone, three times), and one test that walks the live
  `models/` archive and already fails at plain `f4b25238`;
- the lever's own fast tests (178 passed), its slow real-core integration test (6 passed: the slot, the column, the
  refresh, a resume serving the same column bit for bit), and 349 targeted trainer and parity tests;
- a magnet smoke (`--debug`, two refreshes, the sidecar and the recorded fields written) and a control smoke (no
  `[MAGNET]` line, no sidecar, `magnet_kl_coef` 0.0 recorded), and a `main.h2h play` between the two smoke finals.

**UNVERIFIED on the GPU, every one of them** (this registration held no lease):
1. the magnet's cost: one more full-bucket T2 slot of CUDA-graph captures, one more weight copy on the card, one
   inference pass over the buffer per rollout (ESTIMATED ≈ +3 %). Read `magnet/forward_ms` on the first fork;
2. R1's compiled program with the magnet's tenth argument has never been compiled on the card, and the compile gate
   and the canary have never run with it;
3. the size of the KL's floor: T2 serves fp32, R1's trunk is bf16, so the term is a small positive number, not 0,
   right after a refresh;
4. a resume with the sidecar on the card (the CPU integration test passes);
5. that the control at this pin trains as C0 does at `f4b25238` apart from its scalar (CPU golden only).

## 5. The meter and the rule

### 5.1 The three crosses

All head to head, both sides greedy, the players' eval team builder, schedule seed 0, `--oracle-reveal-mode off`,
protocol `gen3_eval_protocol_v1_h2h`, label `ce3d_magnet_screen`, played from a clean detached checkout **at the
CE-3d pin** ([`cross/drive_cross.sh`](cross/drive_cross.sh)). One family and one request a cross.

| cross | rows × columns | pairs a cell | games | family / request | role |
|---|---|---:|---:|---|---|
| **deciding** | 3 magnet × 3 control | 2,000 | 36,000 | `ce3d_magnet_vs_control_h2h` / `ce3d_magnet_vs_control` | **DECIDES** |
| magnet vs C0 | 3 magnet × c C0 (5 ≤ c ≤ 8) | 1,000 | ≤ 48,000 | `ce3d_magnet_vs_c0_h2h` / `ce3d_magnet_vs_c0` | REPORTED |
| control vs C0 | 3 control × the same C0 | 1,000 | ≤ 48,000 | `ce3d_control_vs_c0_h2h` / `ce3d_control_vs_c0` | REPORTED |

**Why magnet against control decides.** It is the only contrast with one difference: same pin, same days, same
entropy weight, the pull on or off. Magnet against C0 mixes the pull with the doubled entropy weight, the pin and
the day (C0 trained 2026-10-21 to 11-10 at `f4b25238`). The reported crosses are X67's own shape and cell size, so
each D sits beside X67's +1.97 pp. The deciding cross is played first.

### 5.2 The plan

[`cross/plan_cross.py`](cross/plan_cross.py) is run ONCE, after the sixth fork's arm-end checks. All six forks are
REQUIRED; the five banked C0 finals (s3501–s3505) are REQUIRED and s3506–s3508 join when they pass. Guards, every
run: `final_model.zip`; one `pin_history` entry at ITS pin from 280,000,170 to ≥ 300,000,170; lineage FORK of P\*'s
sha256; the seed; `n_epochs` 10; `ent_coef` (magnet 0.05, control 0.1, C0 0.05) in `metadata.ent_coef` and
`cli_args.ent_coef`; `model_config.json` `magnet_kl_coef` (magnet 0.05 with `magnet_refresh_steps` 2,000,000;
control 0.0; C0: the key absent); `final_model.magnet.pt` present for a magnet fork and absent otherwise; K9(b)
PASS; one `torch_version` and one `init_num_threads` across all. Executed 2026-11-11: `--columns-only` →
`columns OK`, all eight C0 finals; the full plan REFUSES (no CE-3d final exists).

### 5.3 The statistic and the classes ([`screen_rule.py`](screen_rule.py))

X67's estimator, copied with the cell size as a parameter. For a matrix h of cell scores in pp (r rows, c columns,
n mirrored pairs a cell):

- **D = mean(h) − 50**, the row arm's lead over the column arm.
- s²_R = max(variance of the row means, CELL_VAR / c); s²_C = max(variance of the column means, CELL_VAR / r), with
  CELL_VAR = 2500 / (2n) pp², one cell's binomial variance (0.625 at 2,000 pairs, 1.25 at 1,000).
- V = s²_R / r + s²_C / c, Welch–Satterthwaite df, 90 % interval D ± t(0.95, df) · √V.

**The deciding matrix** (`decide`: exactly 3 × 3 at 2,000 pairs):

| class | condition | meaning |
|---|---|---|
| **PASS** | the interval's lower end > 0 | the magnet forks beat their entropy-matched controls (one-sided 95 %) |
| **WORSE** | its upper end < 0 | the pull loses to the controls |
| **NOT-SHOWN** | otherwise | no difference this screen can see |
| **INCONCLUSIVE** | fewer than three valid forks on either side, a missing or short cell | never interpreted |

Rule 8: an end within 1e-9 pp of 0 satisfies neither PASS nor WORSE. No look 2, no substitute seed, no minimum
effect size.

**The reported matrices** (`report`: 3 × c at 1,000 pairs) return D, the interval and where it lies, and never a
class. `indirect` reports D(magnet vs C0) − D(control vs C0), a second estimate of the deciding contrast through
C0; a sign that disagrees with the direct D is reported as non-transitivity and changes nothing. `decide` also
reports the deciding interval with both variances floored at the fresh-run seed SD (1.76 pp), the pessimistic
reading.

**The self-check** (`screen_rule.py self-check`, [`selfcheck.log`](selfcheck.log): OK): D and SE equal
`main.h2h.cross.cross_stat`'s on a banked 8 × 8; the rule reproduces X67's banked verdict from its record
(D 1.96875, [1.3456, 2.5919], df 6.22); the hand cases for each class, the floors and the tie read as registered.

## 6. What the screen can see ([`power.log`](power.log): `screen_rule.py simulate`, 50,000 draws a row)

The deciding 3 × 3 at 2,000 pairs a cell:

| true fork-seed SD, each arm, pp | half-width, pp | P(PASS) at D = +0.5 | +1 | +1.5 | +2 | +3 | false PASS at 0 | P(WORSE) at −2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.32 | 0.95 | 0.11 | 0.49 | 0.85 | **0.98** | 1.00 | 0.006 | 0.98 |
| 0.5 (the fork evidence, X67 PREREG §5.3) | 1.14 | 0.11 | 0.37 | 0.68 | **0.89** | 1.00 | 0.017 | 0.89 |
| 1.0 | 1.90 | 0.09 | 0.20 | 0.35 | **0.52** | 0.82 | 0.035 | 0.52 |
| 1.76 (seeds spread like fresh runs) | 3.18 | 0.07 | 0.11 | 0.18 | **0.25** | 0.45 | 0.040 | 0.25 |

- If the forks spread like C0's, the screen reads a true +2 pp as PASS about nine times in ten and a true +1 pp
  about one time in three. If they spread like fresh runs it reads +2 pp one time in four. **Three seeds an arm
  carry the variance on 2 df each; more games do not help** (at 1,000 pairs and SD 0.5 the +2 pp row reads 0.78).
- 2,000 pairs a cell were chosen over 1,000 because nine cells are cheap and the binomial part is then below the
  seed part at every row above.
- A NOT-SHOWN means "no gain above the half-width the cross itself reports", not "no effect".

## 7. The launch commands, and their validation

For the Training session, from the main checkout at or after this commit, holding the lease, one run at a time in
the order of [`chain_items_ce3d.json`](chain_items_ce3d.json):

```bash
STAGE=designs/research_state/measurements/x35_breakers_stage_2026-10-17
PIN=333a73ddda7829b9ccdf9e17ef04624eb40edeba       # tag ce3d-pin-333a73dd; NOT the side pin
python -m main.x35.kit stage-check --stage $STAGE --arm <arm> --seed $s        # ALLOWED: "CE-3d PREREG present"
bash $STAGE/launch_x35.sh <arm> $s --pin $PIN --lease-owner "<owner>" --thinning       # the kit's DRY RUN (default)
bash $STAGE/launch_x35.sh <arm> $s --pin $PIN --lease-owner "<owner>" --thinning GO    # only on the orchestrator's GO
```

with `<arm> $s` = `ce3d_mag 7301`, `ce3d_ent 7311`, `ce3d_mag 7302`, `ce3d_ent 7312`, `ce3d_mag 7303`,
`ce3d_ent 7313`. The kit resolves each to `python -m main.launcher --restart-interval-hours 24 --pin-commit $PIN
<argv>`; the first seed of each arm:

```
--model /home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip --behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cuda --magnet-kl-coef 0.05 --seed 7301 --run-name rb_x35_ce3dmag_s7301
--model /home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip --behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cuda --ent-coef 0.1 --seed 7311 --run-name rb_x35_ce3dent_s7311
```

Validated on 2026-11-11, CPU only, no lease, `CUDA_VISIBLE_DEVICES=""`, nice 19 on cores 5-7,13-15, nothing created:

- `python -m main.checkargs --argv "<argv>" --pin $PIN` → `checkargs_ce3d_<arm>_s<seed>.txt`, all six: **exit 0**,
  every flag accepted by the PINNED parser, every ARCH-surface key the production mirror's, the arm's lever its one
  TYPED recipe row, `✓ fork arch`.
- `python -m main.launcher --dry-run --restart-interval-hours 24 --pin-commit $PIN <argv>` →
  `dryrun_ce3d_<arm>_s<seed>.txt`, all six: **role FORK of `rb_x26_s2601`**, +20,000,000 steps, the pin (source
  `pin_commit`), the lever TYPED, 14.33 GiB REQUIRED. **Exit 3, "would NOT launch", on exactly two refusals, both
  this unit's own shell and both expected:** the CUDA-visibility preflight (`CUDA_VISIBLE_DEVICES=''`: no lease) and
  the inherited-nice preflight (the shell ran at nice 19 by rule 5b; a GPU launch is refused above nice 10). The
  kit's own dry run re-runs it with the card visible, at the launch shell's priority, before every GO. The
  transcripts' `showdown deps` line names this registration's worktree, which no longer exists.
- `python -m main.x35.kit pin-check --pin $PIN --side production --arm <arm> --stage $STAGE` → exit 0 for both arms
  ("the CE-3d pin 333a73ddda78 = the side pin f4b25238def5 + one commit, exactly the 44 enumerated path(s)"); with
  the plain side pin it exits 1 ("the CE-3d arms launch only at CE3D_PIN"). `kit required-check` passes at the pin.
  [`kit_checks_ce3d.txt`](kit_checks_ce3d.txt).
- `ce3d_test`: the kit's argv equals the six validated files token for token and is C0's plus the lever;
  `check_validation` PASSes the twelve transcripts as their own arm's (the two shell refusals modelled as absent)
  and FAILs them when the typed lever is absent, another value, or shown by one tool only, and when read as the
  other arm's, C0's or X67's.
- `python -m main.ops.arm_chain plan <scratch chain> --check-pins --describe` on the six draft items →
  [`arm_chain_plan_ce3d.txt`](arm_chain_plan_ce3d.txt): PLAN OK, the kit agrees on all six; a seventh seed REFUSES.
  **The plan does not judge an arm's own pin** (an item at the plain side pin still plans OK): that check is the
  launch script's `kit pin-check`, above.

**The chain items** ([`chain_items_ce3d.json`](chain_items_ce3d.json); `go_by` is the orchestrator's to fill): the
arm, the seed, the pin, `thin: planner`, `flags: ["--thinning"]`, the throughput gate (1,000 steps/s within 60 min,
first line within 90), `on_throughput_fail: page`, `on_void: stop`, and a banner the child's log must show within
30 min:

```json
"ce3d_mag": {"expect": "beside the entropy bonus (ent_coef 0.05; the KL's own entropy part adds 0.05 to that weight)",
             "void": ["ent_coef   | 0.1 ", "ent_coef   | 0.0125 ", "[ModelVersion] FATAL"], "within_min": 30}
"ce3d_ent": {"expect": "ent_coef   | 0.1 ",
             "void": ["ent_coef   | 0.05 ", "[MAGNET]", "[ModelVersion] FATAL"], "within_min": 30}
```

The magnet's line is the one `magnet.attach` prints once the magnet object is built, before the Rust env starts; it
names β and the entropy coefficient beside it, so one string proves both (the pin's smoke printed it with the debug
recipe's values; `ce3d_test` reads the format from the pin's source). The control's is the hyperparameter table's
row, as X67's (C0's logs carry `ent_coef   | 0.05 ` ≈ 5 min after the child attaches; the pin's control smoke
printed `ent_coef   | 0.1 `). A magnet fork that starts without its magnet, or a control that starts on 0.05 or with
a magnet, is stopped within the half hour.

**Disk.** `disk_need_gib` is 14.33 for a control fork (the dry run's figure) and **16.0 for a magnet fork**: the
launcher's arithmetic does not count the magnet's sidecars (one `<checkpoint>.magnet.pt` a checkpoint, each at most
a checkpoint zip's 54.7 MiB; the production size is unmeasured). The thinning planner does not recognise the
sidecars and keeps them, so up to ≈ 1.1 GiB a magnet fork stays behind after thinning.

**Not executed at registration:** `launch_x35.sh` itself (a run is live, so its pre-check 1 HOLDs, and its checks
read the lease) and the crosses' play commands (they write eval-ledger rows). The plan's column check, the rule and
its self-check are executed; the driver's three `family-register` lines and `read_cross.py` are executed with the
pin's code against an empty SCRATCH ledger (the read: INCONCLUSIVE, 9 missing deciding cells, both reported crosses
unavailable), never on a played cell.

## 8. VOID guards, labels, the canary family

**VOID** (any one ⇒ that fork is VOID; fewer than three valid forks in either arm ⇒ INCONCLUSIVE):
- not finished: the arm chain's completeness check (rc 0, `Last step` ≥ 300,000,170, `final_model.zip`, the newest
  checkpoint within 98,304 steps of it);
- K9(b) not PASS at the arm's end;
- the launcher log names another pin than the CE-3d pin, or the run is not `FORK of …/rb_x26_s2601`;
- the post-spawn pool-seed check fails;
- **the lever is not in the record**: §7's banner (the chain STOPS the run); any of §5.2's plan guards on
  `ent_coef`, `magnet_kl_coef`, `magnet_refresh_steps` or the sidecar; `main.dose` reads an epoch count other than
  10 or an effective batch other than 65,536;
- **a magnet that never refreshed**: `python -m main.ops.tb_read <run> --tag magnet/refreshes` shows no point above
  0 for a magnet fork (nine refreshes are due; the tag is emitted only with the magnet on).

**A FATAL that is the lever's own on the first magnet fork** (the magnet's T2 slot, the compile gate or the canary
with the magnet's argument, a sidecar refusal) **STOPS the screen**: no further fork launches, nothing is read, and
a fix is a new pin and a dated amendment or a new registration. The orchestrator decides which.

**Throughput < 1,000 steps/s in the first hour ⇒ PAGE**: HOLD and keep training; the orchestrator decides.

**Labels, reported beside the class and never changing it:**
- `LR-MOVED`, a fork: its `lr_median` (`main.dose`, as printed) is below 0.0000905 or above 0.0001105, more than
  10 % from C0's 0.0001005.
- `LR-SPLIT`: the magnet arm's mean `lr_median` differs from the control arm's by more than 10 % of the control's.
  The controller is part of the recipe, so a rate the pull moved is the lever's own effect, but the two arms then
  differ in dose as well as in the pull.
- `ENT-SPLIT`: the three magnet forks' end entropies and the three control forks' do not overlap (§9), with the
  lower arm named. X67 showed that a sharper policy wins head to head against its own lineage; if the magnet arm
  ends sharper than its control, a PASS may be that effect and not the pull's.

**Canary family.** The CONTROL is **C0's** (`canary_decide.FAMILY_NAME_PATTERNS`, `ce3dent`): C0's argv and graph
with a different scalar in the loss, as X67. The MAGNET arm is in **no family**: R1 takes the magnet's column and
no healthy-pass ratio was ever measured with it, so the C0 bar (2.64) does not transfer. A canary FATAL on a magnet
fork therefore HOLDs (`canary_decide`: "no family bar") and goes to the orchestrator; it is never resumed on a bar
typed by the Training session.

## 9. REPORTED diagnostics (they decide nothing)

- **Entropy, the rate and the KL**: `python -m main.ops.tb_read <run> --tag train/entropy_loss --tag
  train/learning_rate --tag train/approx_kl` for the six forks and the C0 finals; entropy in nats =
  −`entropy_loss`; the end value is the mean of a run's last 20 points (X67's definition).
- **The magnet's own meters**, magnet forks: `train/magnet_kl` (expected: a saw-tooth that rises toward each
  refresh and drops at it; its floor right after a refresh is §4's unverified bf16 number), `magnet/age_steps`,
  `magnet/refreshes`, `magnet/forward_ms` (the cost, against §11's + 3 % estimate).
- **The two crosses against C0** and the indirect line (§5.3), with their caveat: C0 differs from the CE-3d arms by
  pin and by day as well as by lever.
- **The pessimistic interval** (`decide`'s fresh-floor reading).
- No Kakuna game is planned. One that is played later carries its team-clustered interval and a dated NOTE here,
  written before any game, naming the finals, the panel and the count.

## 10. What each class licenses

- **PASS** ⇒ at β 0.05, a 2M cadence and an entropy weight of 0.10, the pull makes a P\* + 20M fork stronger head
  to head than the same weight without it. It licenses REGISTERING, on new seeds, a magnet arm in the entropy
  regime the next recipe actually uses (the swap, or a magnet under whatever `ent_coef` BOUNDARY-1's base carries),
  with the control that regime needs. It licenses nothing else.
  - A PASS whose reported magnet-against-C0 interval does not lie above 0 is a PASS of the pull alone: the arm as
    trained is not shown better than the unchanged recipe.
  - A PASS with `ENT-SPLIT` (magnet lower) or `LR-SPLIT` goes to the orchestrator with the label; the follow-up's
    control must then match the realised entropy or dose, not the coefficient.
- **WORSE** ⇒ this spelling closes: the backlog row records it, and no magnet follow-up is queued unless a new case
  is made for a different form.
- **NOT-SHOWN** ⇒ no gain above the cross's own half-width; the row is closed as "not shown at β 0.05, weight 0.10,
  +20M", and a different form needs its own case.
- **INCONCLUSIVE** ⇒ never interpreted; a re-run is a new registration.

**This file moves no recipe.** No class puts a magnet into `production_config.json`, `design_learner_recipe.md` or
any BOUNDARY-1 arm, and the screen never holds BOUNDARY-1's launch.

**The limits, stated before the number:**
1. **The meter is not the card's question.** The backlog asks whether a trailing regulariser damps the treadmill
   and exploitable tie-breaks. Head to head against a sibling arm measures neither exploitability nor cycling; no
   best-responder read is registered here.
2. **A late fork, a short horizon.** Nine refreshes over 20M steps, from a policy shaped by 280M steps without a
   magnet. Nothing is claimed about a magnet from step 0.
3. **One spelling at an entropy weight nobody plans to use** (§1): one β, one cadence, weight 0.10.
4. **Head to head against siblings is not strength against others** (X67 PREREG §10).

## 11. Cost

| item | GPU-h | CPU | note |
|---|---:|---|---|
| control s7311–s7313, +20M each | **≈ 12.3** | — | ≈ 4.10 GPU-h a fork, the mean of the 20 X35 production forks' dose records (X60-T PREREG §5.1) |
| magnet s7301–s7303, +20M each | **≈ 13** | — | the same + the magnet's pass (≈ +3 %, ESTIMATED, never measured) and a first compile of R1 with the magnet's argument |
| the deciding cross, 36,000 games | ≈ 0.3 in a GPU window | or ≈ 5.8 h on the three agent cores | from X67's cross: 48,000 games in 27,677 s on three threads; the GPU figure is X67's ESTIMATE |
| the two reported crosses, ≤ 96,000 games | ≈ 0.7 | or ≈ 15.4 h | the same rate; 9.6 h at five columns |
| the pin checkout's Rust build | 0 | minutes, under `mem_cap` | once |
| §9's reads, the dose read | 0 | seconds | |

≈ 26 GPU-h and ≈ 21 h of the three agent cores (≈ 63 lane-h) in all. **Hard stop:** six forks, three crosses, one
read; nothing here extends.

## 12. What this file does NOT do

It registers no swap arm, no second β or cadence, no magnet at another entropy weight, no fresh-run magnet arm, no
exploitability read and no Kakuna game. It does not adopt or delay X67's PASS, amends no BOUNDARY-1 document, and
changes no baseline and no production recipe. CE-3d has no `main.reads` rule or `reads/` record: its verdict is
`cross/read_cross.py`'s `verdict.json`, written by `screen_rule.decide`.
