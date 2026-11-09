# X67 — the entropy screen: PRE-REGISTRATION (2026-11-09)

**Status: REGISTERED before any launch, any game and any read. NOT LAUNCHED** (no `rb_x35_x67ent_*` run exists). The
launch is the orchestrator's GO through the Training session's kit (arm `x67`). Nothing below changes after the first
fork launches except by a dated amendment, written before the cross is played, that says why.

**The question.** P\* was trained with an entropy bonus of 0.05, five to fifty times the literature's range, chosen on
2026-09-12 because it restored an entropy level, with no strength comparison. Does a plateau policy trained on with a
quarter of that bonus beat the same policy trained on with all of it?

Abbreviations: pp = percentage points. P\* = baseline `pstar_plateau` (`rb_x26_s2601` @280,000,170, sha256
`814823c5…`). C0 = P\* + 20M with no lever at the production side pin (`rb_x35_c0_s<seed>`), the banked plateau
control. `ent_coef` = the coefficient on the entropy bonus in the PPO loss. Greedy = the deployed policy, the most
probable action; sampled = an action drawn from the policy at temperature 1, which is how self-play trains.
SD = standard deviation; SE = standard error; df = degrees of freedom. BOUNDARY-1 = the next from-scratch retrain
([`../boundary1_scratch_2026-11-09/DRAFT_DESIGN.md`](../boundary1_scratch_2026-11-09/DRAFT_DESIGN.md)).

## 1. Why now

BOUNDARY-1's draft design §9.1 asks for this screen before the fresh runs: `ent_coef` is a property of the recipe, so
it must be screened on its own and, if it passes, moved into the base of BOTH arms, never carried by one. The GPU has
a gap after X60-T's and X66's training (≈ Tue 2026-11-10 afternoon) while E13 is built.

The case for the lever, and what is and is not measured:
- 0.05 against a literature range of 0–0.01 (`../../../endstate/design_learner_recipe.md` §3.9). The 2026-09-12
  choice cleared an ENTROPY bar (ledger L18044, L18067); its strength effect was NOT DETECTED (L18073).
- C0's own plateau entropy under 0.05 is **≈ 1.01 nats** (`rb_x35_c0_s3501`, the last 20 `train/entropy_loss`
  points: median 1.010, range 0.995–1.019; `python -m main.ops.tb_read`).
- The independent review's "0.56 top-action mass" and "greedy is worth 23 pp over sampled" (L21467, L21447) were
  measured on N0, an older lineage, the second on 100 games. **They are not P\*'s numbers**; §9 measures the gap on
  these finals if CPU allows.

**Prediction** (GUESS; it decides nothing): D ∈ [−1, +2] pp, most likely within ± 1 pp. A late fork is the weakest
frame for a recipe knob (§10's limit), and Lane S found sharpening raises action starvation (L21471, PRELIMINARY).

## 2. The arms

| | **X67** (kit arm `x67`) | **C0**, the control (kit arm `c0`) |
|---|---|---|
| run | `rb_x35_x67ent_s<seed>` | `rb_x35_c0_s<seed>` |
| start | P\* @280,000,170 | the same file |
| lever | **`--ent-coef 0.0125`** typed (`kit.X67_LEVER`) | none: `ent_coef` 0.05 inherited from P\*'s record |
| argv | C0's, token for token, + the lever before `--seed` ([`argv_x67_s6701.txt`](argv_x67_s6701.txt)) | its recorded `original_command` |
| dose | +20,000,000 steps (`--steps 300000170`) | the same |
| pin | `f4b25238def57900c0305bb2a3a5a8f9edaef285` | the same (`pin_history`, one entry each) |
| seeds | **6701, 6702, 6703** (new) | 3501–3505 BANKED; 3506–3508 are X60-T's top-up (§5.2) |
| GPU | ≈ 12.3 GPU-h (3 × ≈ 4.1) | **0 GPU-h here** |

A fork applies a typed `--ent-coef`: the resume path sets `model.ent_coef = args.ent_coef` (`model_build.py` line 297
at the pin), and `checkargs` prints it as the arm's ONE typed recipe difference from its parent
(`ent_coef 0.0125 (TYPED) production: 0.05`). Nothing else differs: one lever per A/B.

## 3. Seeds, dose, one look

- **Seeds 6701–6703** are no registered sibling's: no spec, kit arm, design doc or run directory names them
  (searched `src/`, `designs/`, `models/` and the archive on 2026-11-09; `kit_test` pins it against every kit arm).
  A seed draws the snapshot pool's RNG and the team schedule; the seeded pool SET is the side's record, checked after
  the spawn.
- **Dose.** `python -m main.dose` on the five banked C0 finals ([`dose_c0.txt`](dose_c0.txt)): effective batch
  65,536, 10 epochs, `lr_median` 0.0001005, dose rate 2.044e-08, identical on all five. X67 inherits every one of
  those knobs from P\*'s record (the dry run lists them), so the two arms are matched by dose at launch; §8 checks it
  at the end.
- **One look.** The stage is `X67` (`batch.X67_STAGES`): three seeds, no look 2, no extension. `kit stage-check` is
  PENDING until this file exists on the kit's checkout.

## 4. The pin, and why the control is banked

**The pin is `f4b25238`, the production side pin, not main.** Main is config v167 and its fixes move P\*'s inputs on
96 % of rows, so a fork of P\* at main trains on a different observation than P\* did: the lever would be confounded
with a 20M re-adaptation. `f4b25238` is config v161 with obs-semantics 151, the same obs-semantics as P\*'s own commit
(`9c033489`, its ancestor), so P\*'s inputs are unchanged. It reads its own pinned `data/`, and the forks run under
P\*'s recorded torch (`2.8.0+cu126`).

**The control is reused, not retrained.** A fresh same-dose control at this pin would be C0 again: the same parent
file, the same argv less the lever, the same pin, data and torch. Five such finals are banked (s3501–s3505, each
K9(b) PASS, each one `pin_history` entry at the pin) and three more are queued by X60-T (s3506–s3508, the same launch
line). Reusing them costs 0 GPU-h and gives the cross five to eight control columns instead of three.

**What reuse does not hold fixed:** the day. The banked five trained 2026-10-21 to 10-24, the top-up on 11-09/10, X67
on ≈ 11-10. Same pin, same torch, same card; a driver or thermal difference would be confounded with the lever. The
top-up controls train within a day of X67, so §5.3 reports D over the banked and the top-up columns separately.

## 5. The meter and the rule

### 5.1 The cross

Every X67 final plays every admitted C0 final head to head: **1,000 mirrored team pairs a cell (2,000 games), both
sides greedy**, the players' eval team builder, schedule seed 0, `--oracle-reveal-mode off`, protocol
`gen3_eval_protocol_v1_h2h`, family `x67_entropy_screen_h2h`, request `x67_entropy_screen`. It is played from a clean
checkout **at the pin** ([`cross/drive_cross.sh`](cross/drive_cross.sh)), never at main, whose game and observation
are not the ones these finals trained on. No X67 seed is paired with a C0 seed (the seeds differ), so every fork
meets every control.

### 5.2 The columns

[`cross/plan_cross.py`](cross/plan_cross.py) is run ONCE, after the third X67 fork's arm-end checks. Rows: the three
X67 finals. Columns: the C0 finals among s3501–s3508 that exist then. **The five banked are REQUIRED**; a top-up seed
with no `final_model.zip` is left out and named; a top-up final that fails a guard REFUSES the plan. Guards, every
run: `final_model.zip`; one `pin_history` entry at the pin from 280,000,170 to ≥ 300,000,170; lineage FORK of P\*'s
sha256; the seed; `n_epochs` 10; `ent_coef` (0.0125 for a row, 0.05 for a column) in BOTH `metadata.ent_coef` and
`cli_args.ent_coef`; K9(b) PASS; one `torch_version` and one `init_num_threads` across all. Executed on 2026-11-09:
`--columns-only` → `columns OK: [3501, 3502, 3503, 3504, 3505]`; the full plan REFUSES (no X67 final exists).

The column count is fixed by what has finished, never by a look at a score: no cross game exists before the plan.

### 5.3 The statistic and the classes ([`screen_rule.py`](screen_rule.py), `decide`)

h is the 3 × c matrix of cell scores in pp (5 ≤ c ≤ 8).

- **D = mean(h) − 50**, the X67 arm's lead over C0.
- s²_R = max(variance of the 3 row means, 1.25 / c); s²_C = max(variance of the c column means, 1.25 / 3).
  1.25 pp² is the binomial variance of one 2,000-game cell: a mean of k cells cannot vary by less than 1.25 / k
  between seeds, so a smaller sample variance is luck and is not believed.
- V = s²_R / 3 + s²_C / c, with Welch–Satterthwaite df; the 90 % interval is D ± t(0.95, df) · √V.

| class | condition | meaning |
|---|---|---|
| **PASS** | the interval's lower end > 0 | the forks beat their controls (one-sided 95 %) |
| **WORSE** | its upper end < 0 | the smaller bonus loses to the controls |
| **NOT-SHOWN** | otherwise | no difference this screen can see |
| **INCONCLUSIVE** | fewer than three valid rows or five columns, a missing or short cell | never interpreted |

Rule 8: an end within 1e-9 pp of 0 satisfies neither PASS nor WORSE (a tie never passes). There is no look 2, no
substitute seed and no minimum effect size.

**Why the variance is estimated and not floored at 1.76 pp.** The 1.76 pp in
[`../boundary1_scratch_2026-11-09/seed_spread.log`](../boundary1_scratch_2026-11-09/seed_spread.log) pools FRESH 15M
crosses. Forks of one plateau checkpoint sit closer: C0's three G1 finals read an SD of 0.48 pp on S against a
0.55 pp measurement SE (X48 PREREG §6), and fork finals show no Kakuna seed spread beyond binomial (X60-T PREREG
§5.1). A 1.76 floor would fix the half-width near ± 2.6 pp whatever the data said. **A fork's head-to-head seed SD
has never been measured**, and under a quarter of the bonus it may be larger, so the rule estimates it from the
cross, and `decide` REPORTS the fresh-floor interval beside the verdict as the pessimistic reading.

**The self-check** (`screen_rule.py self-check`, run 2026-11-09: OK): D and SE equal `main.h2h.cross.cross_stat`'s on
a banked 8 × 8; the hand cases for each class, the floor, the tie and five INCONCLUSIVE shapes read as registered.

## 6. What the screen can see ([`power.log`](power.log): `screen_rule.py simulate`, 50,000 draws a row)

| columns | true seed SD, X67 / C0, pp | half-width, pp | P(PASS) at D = +1 | +2 | +3 | +4 | false PASS at 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| 5 | 0.5 / 0.5 (the fork evidence) | 1.04 | 0.43 | **0.93** | 1.00 | 1.00 | 0.016 |
| 5 | 1.0 / 0.5 | 1.41 | 0.28 | 0.69 | 0.91 | 0.98 | 0.036 |
| 5 | 1.76 / 0.5 (X67 seeds spread like fresh runs) | 2.35 | 0.18 | **0.39** | 0.62 | 0.80 | 0.050 |
| 5 | 2.5 / 0.5 | 3.35 | 0.14 | 0.27 | 0.42 | 0.59 | 0.054 |
| 8 | 0.5 / 0.5 | 0.84 | 0.57 | **0.97** | 1.00 | 1.00 | 0.024 |
| 8 | 1.0 / 0.5 | 1.34 | 0.33 | 0.71 | 0.92 | 0.99 | 0.045 |
| 8 | 1.76 / 0.5 | 2.38 | 0.18 | **0.39** | 0.62 | 0.80 | 0.053 |
| 8 | 2.5 / 0.5 | 3.40 | 0.14 | 0.27 | 0.42 | 0.58 | 0.057 |

- If X67's seeds spread like C0's, the screen resolves ± 0.8–1.0 pp and reads a true +2 pp as PASS 93–97 % of the
  time. If they spread like fresh runs, it reads +2 pp 39 % and +4 pp 80 % of the time: **three rows carry nearly
  all of the variance on 2 df, and more columns do not help.**
- The false-PASS rate at a true 0 is 0.001–0.057 across the table (nominal 0.05; Welch's df runs slightly
  anti-conservative at a large row spread). P(WORSE) mirrors P(PASS).
- A NOT-SHOWN therefore means "no gain above the half-width the cross itself reports", not "no effect".

## 7. The launch commands, and their validation

For the Training session's kit, from the main checkout at or after this commit, holding the lease, per seed
s = 6701, 6702, 6703 (one run at a time):

```bash
STAGE=designs/research_state/measurements/x35_breakers_stage_2026-10-17
PIN=f4b25238def57900c0305bb2a3a5a8f9edaef285
python -m main.x35.kit stage-check --stage $STAGE --arm x67 --seed $s          # ALLOWED: "X67 PREREG present"
bash $STAGE/launch_x35.sh x67 $s --pin $PIN --lease-owner "<owner>" --thinning       # the kit's DRY RUN (default)
bash $STAGE/launch_x35.sh x67 $s --pin $PIN --lease-owner "<owner>" --thinning GO    # only on the orchestrator's GO
```

The kit resolves that to `python -m main.launcher --restart-interval-hours 24 --pin-commit $PIN <argv>`, with the argv
of [`argv_x67_s6701.txt`](argv_x67_s6701.txt) (and `_s6702`, `_s6703`):

```
--model /home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip --behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cuda --ent-coef 0.0125 --seed 6701 --run-name rb_x35_x67ent_s6701
```

Validated on 2026-11-09, CPU only, no lease, `CUDA_VISIBLE_DEVICES=""`, nice 19 on cores 5-7,13-15, nothing created:

- `python -m main.checkargs --argv "<argv>" --pin $PIN` → [`checkargs_x67_s6701.txt`](checkargs_x67_s6701.txt)
  (and `_s6702`, `_s6703`): **exit 0**, 8 flags accepted by the PINNED parser, 0 unrecognized, 72 inherited from
  P\*'s record, every ARCH-surface key the production mirror's, `ent_coef 0.0125 (TYPED) production: 0.05`, `✓ fork
  arch`. The other five RECIPE differences from `recipe.fork` it lists are C0's too (a fork's recipe is its parent's
  record; `recipe.fork` is never applied).
- `python -m main.launcher --dry-run --restart-interval-hours 24 --pin-commit $PIN <argv>` →
  [`dryrun_x67_s6701.txt`](dryrun_x67_s6701.txt) (and `_s6702`, `_s6703`): **role FORK of `rb_x26_s2601`**,
  +20,000,000 steps, pin `f4b25238` (source `pin_commit`), torch 2.8.0+cu126 as recorded, `ent_coef 0.0125 (TYPED)`,
  `✓ fork arch`, 12.95 GiB REQUIRED vs 212 GiB free. **Its one refusal is the CUDA-visibility preflight**
  (`CUDA_VISIBLE_DEVICES=''`: this unit holds no lease): exit 3, "would NOT launch", as X48's
  registration. The kit's own dry run re-runs it with the card visible before every GO.
- `kit_test`: the kit's argv equals the three validated files token for token and is C0's plus the lever;
  `check_validation` PASSes the six transcripts as X67's (the hidden-card line alone modelled as visible) and FAILs
  them when the typed `ent_coef` is absent, another value, or shown by one tool only, and FAILs them read as C0's
  (`kit.ARM_TYPED_RECIPE`, added with this registration: before it, X67's transcripts passed as C0's).
- `python -m main.ops.arm_chain plan <scratch chain> --check-pins --describe` on the three draft items →
  [`arm_chain_plan_x67.txt`](arm_chain_plan_x67.txt): PLAN OK, the kit agrees on all three; a fourth seed REFUSES.

**The draft chain items** (for a chain file the Training session writes; `go_by` is the orchestrator's to fill): arm
`x67`, the seed, the pin, `thin: planner`, `flags: ["--thinning"]`, `disk_need_gib: 12.95`, the throughput gate
(1,000 steps/s within 60 min, first line within 90), `on_throughput_fail: page`, and **a banner with `on_void: stop`**:

```json
"banner": {"expect": "ent_coef   | 0.0125 ", "void": ["ent_coef   | 0.05 ", "[ModelVersion] FATAL"], "within_min": 30}
```

The child prints its hyperparameter table once at training start (`_HparamLogCallback`, the value the resume path
assigns to the model). All five banked C0 logs carry `|    ent_coef   | 0.05     |`, ≈ 5 min after the child
attaches (4.1–5.6 min); the logger's own formatter renders 0.0125 as `|    ent_coef   | 0.0125   |` (executed
2026-11-09). A fork that starts on 0.05 is therefore stopped within the half hour instead of training 4 GPU-h as a
ninth control.

**Not executed at registration:** `launch_x35.sh` itself (a run is live, so its pre-check 1 HOLDs, and its checks
read the lease) and the cross's play command (it writes eval-ledger rows and needs a Rust build at the pin). The
cross's planner, rule and self-check are executed. The driver's `family-register` line and `read_cross.py` are
executed with the pin's code against an empty SCRATCH ledger (the read: INCONCLUSIVE, 15 missing cells; against the
real ledger it refuses, the family being unregistered until the driver runs), never on a played cell.

## 8. VOID guards, labels, the canary family

**VOID** (any one ⇒ that fork is VOID; fewer than three valid forks ⇒ INCONCLUSIVE, and 0.05 stays):
- not finished: the arm chain's completeness check (rc 0, `Last step` ≥ 300,000,170, `final_model.zip`, the newest
  checkpoint within 98,304 steps of it);
- K9(b) not PASS at the arm's end;
- the launcher log names another pin, or the run is not `FORK of …/rb_x26_s2601`;
- the post-spawn pool-seed check fails;
- **the lever is not in the record**: the child's hyperparameter table does not print `ent_coef 0.0125` within
  30 min (§7's banner; the chain STOPS the run), `metadata.json` `ent_coef` or `cli_args.ent_coef` is not 0.0125 (the
  plan's guard), or `main.dose` reads an epoch count other than 10 or an effective batch other than 65,536.

**Throughput < 1,000 steps/s in the first hour ⇒ PAGE**: HOLD and keep training; the orchestrator decides.

**Labels, reported beside the class and never changing it:**
- `LR-MOVED`: a fork's `lr_median` (`main.dose`, as printed) is below 0.0000905 or above 0.0001105, more than 10 %
  from C0's 0.0001005. The KL controller is part of the recipe, so a rate it moved under the smaller bonus is the
  lever's own effect, but the reader must know the dose differs.
- `ENT-UNMOVED`: the X67 arm's mean end entropy (§9) is not below the lowest C0 seed's. The lever then did not
  change the policy's sharpness, and a NOT-SHOWN is a null on a lever that did nothing, not on sharpness.

**Canary family: C0** (`canary_decide.FAMILY_NAME_PATTERNS`). X67 is the production architecture, C0's argv and
graph; the entropy coefficient is a scalar in the loss and adds no kernel. The compile canary's C0 bar (2.64)
applies; a canary HOLD is the Training session's to resolve as for any C0 arm.

## 9. REPORTED diagnostics (they decide nothing)

- **The entropy trajectory**: `python -m main.ops.tb_read <run> --tag train/entropy_loss --tag train/learning_rate
  --tag train/approx_kl` for the three forks and the C0 finals; entropy in nats = −`entropy_loss`; the end value is
  the mean of a run's last 20 points. This is the check that the lever is live.
- **The pessimistic interval and the day split**: `decide`'s fresh-floor interval, and D over the banked and the
  top-up columns.
- **Greedy minus sampled**, only if Kakuna lanes are free: each side's final at greedy and at
  `main.anchors --our-temperature 1.0` against Kakuna (`main.h2h` cannot play a sampled side). A dated NOTE here
  names the finals, the panel and the game count before any game.
- **Kakuna on U, clustered by team**, only if CPU allows: it confirms or contradicts, it never decides. Its estimator
  is the team-clustered `main.reads` rule BOUNDARY-1's draft §4.3 names, which is not landed today; a dated NOTE
  names the rule and the games before any game. ≈ 6,480 games for the X67 side alone (≈ 45 lane-h at 144 games a
  lane-hour), so it is unlikely to fit beside the live queue.

## 10. What each class licenses

- **PASS** ⇒ `ent_coef` 0.0125 is named for the **base recipe of BOTH arms** of BOUNDARY-1 in that PREREG's §10 fill
  ([`../boundary1_scratch_2026-11-09/DRAFT_PREREG.md`](../boundary1_scratch_2026-11-09/DRAFT_PREREG.md) §2: a
  screened knob "is in the base of EVERY arm"). It is never carried by one arm.
- **WORSE** or **NOT-SHOWN** or **INCONCLUSIVE** ⇒ 0.05 stays in both arms. The screen never holds the launch.
- A PASS with `LR-MOVED` goes to the orchestrator with both numbers before the recipe moves.

**This file moves no recipe.** `production_config.json`'s recipe block and `design_learner_recipe.md` §3.22 move
together, in BOUNDARY-1's registration, and the owner's 2026-09-29 ruling (§3.9: "no direct adoption of a lower
bonus") stands until the owner lifts it there.

**The limits of a PASS, stated before the number:**
1. **A late fork is not a fresh run.** The screen changes the bonus at step 280M of a policy shaped by 0.05. The
   licence puts 0.0125 in from step 0, where early exploration is a different regime; the 2026-09-12 record is the
   only from-scratch evidence, and it compared entropy, not strength. Both BOUNDARY-1 arms share the base, so a bad
   base cannot confound their contrast, but it can lower both. Draft §9.1 names the alternative (an in-loop
   schedule for the plateau continuation).
2. **Head to head against siblings is not strength against others.** A sharper policy may beat its own lineage and
   gain nothing against Kakuna. §9's Kakuna read is the only outside evidence, and it is conditional.
3. **One dose.** +20M at one coefficient. Nothing is claimed about 0.025, 0.005 or a schedule.

## 11. Cost

| item | GPU-h | CPU | note |
|---|---:|---|---|
| X67 s6701–s6703, +20M each | **≈ 12.3** | — | ≈ 4.10 GPU-h a fork, the mean of the 20 X35 production forks' dose records (X60-T PREREG §5.1) |
| the cross, 3 × 8 cells = 48,000 games | ≈ 0.35 in a GPU window | or ≈ 5 h on the three agent cores | ESTIMATE from BOUNDARY-1's draft design §8 (the closing cross: 128,000 games in 52 min under the GPU lock; ≈ 8 h for 80,000 games on the three agent cores) |
| the cross at 5 columns, 30,000 games | ≈ 0.2 | or ≈ 3 h | |
| the pin checkout's Rust build | 0 | minutes, under `mem_cap` | once |
| §9's entropy read, dose read | 0 | seconds | |
| §9's Kakuna reads | 0 | ≈ 45 lane-h and up | conditional; not planned |

## 12. What this file does NOT do

It registers no fresh-run entropy arm, no schedule, no second coefficient, no Lane S read and no Kakuna game. It does
not amend BOUNDARY-1's draft, whose §9.1 says "the ten C0 finals": five are banked and eight are registered. It
changes no baseline and no production recipe. X67 has no `main.reads` rule or `reads/` record: its verdict is
`cross/read_cross.py`'s `verdict.json`, written by `screen_rule.decide`.
