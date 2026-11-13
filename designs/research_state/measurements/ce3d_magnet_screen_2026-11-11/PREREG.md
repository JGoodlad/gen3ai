# CE-3d — the trailing-magnet screen: PRE-REGISTRATION (2026-11-11)

**READ 2026-11-12 (§16): PASS.** The three swap forks trained (every VOID guard executed, none fired) and beat the
eight banked C0 finals head to head: **D = +3.97 pp, 90 % interval [+3.31, +4.63]**. Label `ENT-LOW` (the swap ended
at 0.48–0.50 nats, sharper than every C0 final and than X67's forks); no fork `LR-MOVED`. The reported cross against
X67's finals read D = +1.01 pp [−0.12, +2.13], so the reading quoted with the PASS is "**the swap's gain over C0 is
plausibly the lower uniform push alone**". The PASS licenses REGISTERING a follow-up and nothing else; none is
registered here and no recipe moves. The status paragraph below is the registration's own, as it stood at
Amendment 3 (before any fork trained); §1 to §15 are unchanged.

**Status: REGISTERED 2026-11-11 (`08b998d12`) and AMENDED three times the same day (Amendment 1, §13: ONE arm, the
SWAP, read against the banked C0 finals with X67's cross and rule; the two arms first registered are dropped.
Amendment 2, §14: a REPORTED cross of the swap forks against X67's three finals, which decides nothing. Amendment 3,
§15: a RE-PIN of the forks' training pin, after the first fork's launch died at step 0 in a startup check; the arm,
its seeds, argv and dose, both crosses and the rule are unchanged). Amendments 1 and 2 were written blind, before
any launch, any game and any read; Amendment 3 after ONE launch that took no training step, and before any training
step, any game and any read. NO FORK HAS TRAINED** (the one attempt, `rb_x35_ce3dswap_s7321` at the first pin, wrote
no checkpoint and no metadata, §15; no CE-3d ledger family exists).
The launch is the orchestrator's GO through the Training session's kit (arm `ce3d_swap`); the hand-off is
[`chain_items_ce3d.json`](chain_items_ce3d.json) (§7). Nothing below changes after the first fork launches except by
a dated amendment, written before the cross is played, that says why.

**The question.** Does a plateau policy get stronger when the entropy bonus (a pull toward the uniform policy) is
replaced, at the same weight, by a pull toward a trailing copy of the policy itself (a "magnet", refreshed every 2M
steps)?

**The tier is FULL** (`../../README.md` § Registration tiers). Its cost would fit a light screen (≈ 13 GPU-h, ≈ 32
CPU lane-h), but it is an amendment of a full registration, and its arm runs at a NEW pin carrying code that has
never trained on the GPU. Its verdict adopts nothing (§10).

Abbreviations: pp = percentage points. P\* = baseline `pstar_plateau` (`rb_x26_s2601` @280,000,170, sha256
`814823c5…`). C0 = P\* + 20M with no lever at the production side pin `f4b25238` (`rb_x35_c0_s<seed>`), the banked
plateau control. `ent_coef` = the coefficient on the entropy bonus in the PPO loss. KL = the Kullback–Leibler
divergence. π = the learner's policy over a row's legal actions; π_mag = the magnet's. β = `--magnet-kl-coef`.
H(π) = the policy's entropy. MMD = Magnetic Mirror Descent (Sokota et al. 2023). T2 = the Rust-side inference
server self-play runs on; R1 = the learner's compiled update step. K9(b) = the behaviour-policy check (does the
learner's log-probability of each played action match the one stored at play time). bf16 / fp32 = 16-bit / 32-bit
floating point. SD = standard deviation; df = degrees of freedom. X67 = the entropy screen
([`../x67_entropy_screen_2026-11-09/`](../x67_entropy_screen_2026-11-09/README.md)).

## 1. Why now, and why the swap

The lever was built on 2026-11-10 (task T56, `251bbfb87`; [`../../../training/magnet_kl.md`](../../../training/magnet_kl.md)):
`loss += β · KL(π ‖ π_mag)`, from MMD and R-NaD (Perolat et al. 2022). It is off by default and no run has trained
with it. The GPU has a gap after X69 while BOUNDARY-1 waits on its re-pin.

**The loss identity.** Over a row's legal actions

```
KL(π ‖ π_mag) = −H(π) − Σ_a π(a) · log π_mag(a)
```

so the term carries an entropy part of its own (`magnet_learner_test.py` pins it: against a masked-uniform magnet
its gradient is the entropy bonus's at `ent_coef + β`). The three losses, regularisers only:

| arm | `ent_coef` | β | regulariser in the loss | weight on the entropy gradient |
|---|---:|---:|---|---:|
| C0 | 0.05 | 0 | −0.05 · H(π) | 0.05 |
| **the swap** | **0** | **0.05** | −0.05 · H(π) − 0.05 · Σ π log π_mag | **0.05** |
| (dropped) magnet beside the bonus | 0.05 | 0.05 | −0.10 · H(π) − 0.05 · Σ π log π_mag | 0.10 |

**The swap's loss is C0's plus one term**, the cross-entropy pull toward the magnet, with the entropy weight
matched. It is MMD's own form: the uniform magnet (which is what an entropy bonus is) replaced by a trailing one at
the same weight. Against the eight banked C0 finals that is one lever, at half the GPU of a two-arm screen.

**Why the first registration's design was dropped.** It tested the magnet beside the bonus, at an entropy weight of
0.10, against a control at `--ent-coef 0.1`. X67's PASS (read 2026-11-11: three forks at `ent_coef` 0.0125 beat the
eight C0 finals, D = +1.97 pp [+1.35, +2.59], every fork LR-MOVED, end entropy 0.68 against 1.01 nats) says the
entropy weight should go DOWN from 0.05, so a contrast at 0.10 isolates the pull in a regime no recipe is heading
to, and its control answers a question X67 has made moot.

**One term in the loss is not one effect on the policy** (§10 limit 1 has the consequence). A pull toward the
uniform policy never switches off; a pull toward the magnet is zero whenever π = π_mag, where the entropy part and
the cross-entropy part of the KL cancel. So the swap also REMOVES the standing push toward uniform that C0 has, and
X67 showed that a smaller push wins head to head. The screen therefore reads "the trailing anchor in place of the
uniform one", and cannot by itself split that into "the anchor helps" and "less uniform push helps". §8's entropy
label and §5.4's reported cross against X67's own finals are the registered evidence on which it is.

**Prediction** (GUESS; it decides nothing): D ∈ [0, +2.5] pp; the swap forks end sharper than C0 (`ENT-LOW`, §8),
with an end entropy between X67's 0.68 and C0's 1.01 nats. Nine refreshes in 20M steps is a short horizon for a
regulariser whose claimed effect is on a slow cycle.

## 2. The arm

| | **THE SWAP** (kit arm `ce3d_swap`) | C0 (banked; the columns) |
|---|---|---|
| in plain words | P\* trained on with no entropy bonus, pulled instead toward a copy of itself that is replaced every 2M steps | P\* trained on, nothing changed |
| run | `rb_x35_ce3dswap_s<seed>` | `rb_x35_c0_s<seed>` |
| start | P\* @280,000,170 | the same file |
| lever, typed | **`--ent-coef 0 --magnet-kl-coef 0.05`** (`kit.CE3D_SWAP_LEVER`) | none |
| `ent_coef` | 0 (typed; P\*'s record carries 0.05) | 0.05 |
| weight on the entropy gradient | 0 + 0.05 = **0.05** | 0.05 |
| the pull toward the magnet | β 0.05; magnet = P\*'s weights at the fork, replaced at the first update boundary at or after each 2,000,000 steps of age (`--magnet-refresh-steps`, untyped: the pin's default) | none |
| argv | C0's, token for token, + the four lever tokens before `--seed` ([`argv_ce3d_swap_s7321.txt`](argv_ce3d_swap_s7321.txt)) | its recorded `original_command` |
| dose | +20,000,000 steps (`--steps 300000170`) | the same |
| pin | **`9e97cad5709da21a1f9d4d8846096c87eb17b1d0`** (tag `ce3d-pin-9e97cad5`, §4: the first CE-3d pin `333a73dd` + the startup fit check's fix, Amendment 3) | `f4b25238def57900c0305bb2a3a5a8f9edaef285` |
| seeds | **7321, 7322, 7323** (new) | 3501–3508 (banked) |
| GPU | ≈ 13 GPU-h (3 × ≈ 4.2–4.4) | 0 here |

`checkargs` prints the lever as the arm's TWO typed recipe rows against its parent: `ent_coef 0.0 (TYPED)
production: 0.05` and `magnet_kl_coef 0.05 (TYPED) production: 0.0` (the argv's `0` prints as `0.0`).

**A typed `--ent-coef 0` is applied on a fork.** At the pin, `main/train/model_build.py` sets `model.ent_coef =
args.ent_coef` on every resume and fork, and the swap's CPU fork smoke (§4) recorded 0.0 from a parent whose record
carried 0.1.

**Refreshes.** A fork's magnet is set at 280,000,170 with age 0 and falls due at each 2M of age, so **nine** refreshes
land inside the run (≈ 282M … 298M, each up to one rollout late); the tenth falls due at the end, where the last
update has no boundary after it (`magnet_kl.md` §5).

**Not registered:** the magnet beside an entropy bonus (at 0.05, 0.0125 or any other), an `--ent-coef 0` arm with
no magnet, a second β, a second cadence, any control trained at the CE-3d pin.

## 3. Seeds, dose, one look

- **Seeds 7321, 7322, 7323** are no registered sibling's: no spec, kit arm, design doc or run directory names them
  (searched `src/`, `designs/`, `scripts/` and `models/` on 2026-11-11; `ce3d_test` pins them against every other
  kit arm).
- **Retired seeds.** The first registration's 7301–7303 (`ce3d_mag`) and 7311–7313 (`ce3d_ent`) were never launched
  and are never reused by any arm (`batch.CE3D_RETIRED_SEEDS`; `ce3d_test` pins that no kit arm carries one and
  that the kit refuses them).
- **Dose.** The arm inherits C0's effective batch 65,536, 10 epochs and learning-rate controller from P\*'s record
  (the dry runs list them). C0's finals read `lr_median` 0.0001005 (`main.dose`). §8 checks the dose at the end and
  labels a moved rate.
- **One look.** The stage is `CE3D` (`batch.CE3D_STAGES`): three seeds, no look 2, no extension, no substitute
  seed. `kit stage-check` is PENDING until this file exists on the kit's checkout. A fork whose launch died before
  its first training step is relaunched at its own seed, argv and run name (§15): the same fork, not a substitute.
- **Order.** 7321, 7322, 7323, one at a time.

## 4. The pin, and the confound it brings

**The swap TRAINS at `9e97cad5709da21a1f9d4d8846096c87eb17b1d0`** (Amendment 3, §15), annotated tag
**`ce3d-pin-9e97cad5`** on origin (`batch.CE3D_PIN`): two commits on the production side pin `f4b25238`.

| commit | on | paths | what | tag, kit constant |
|---|---|---:|---|---|
| `333a73ddda7829b9ccdf9e17ef04624eb40edeba`, **the first CE-3d pin** | the side pin `f4b25238` | 44 | the lever (§4.1). The first registration's, unchanged. **Both crosses are PLAYED here** (§5.1) | `ce3d-pin-333a73dd`, `batch.CE3D_MAGNET_PIN` |
| `9e97cad5709da21a1f9d4d8846096c87eb17b1d0`, **the training pin** | the first CE-3d pin | 3 | the startup fit check's fix (§4.1, §15) | `ce3d-pin-9e97cad5`, `batch.CE3D_PIN` |

`kit pin-check` requires both steps (executed in §7): the training pin's one parent is the first CE-3d pin and its
commit changes EXACTLY the 3 paths of `batch.CE3D_PIN_PATHS`; the first pin's one parent is the side's written pin
and its commit changes EXACTLY the 44 paths of `batch.CE3D_MAGNET_PIN_PATHS`. Below, "the CE-3d pin" is the training
pin wherever a fork is launched, trained or checked (§7, §8), and the first pin wherever a game is played (§5).

### 4.1 What the commit carries

**Why not main, and why not the plain side pin.** Main's fixes move P\*'s inputs on 96 % of rows, so a fork of P\* at
main is confounded with a 20M re-adaptation (X67 PREREG §4). `f4b25238` has no magnet. The commit carries:
- **the magnet lever** as main built it (`251bbfb87`): `agents/training/magnet.py`, its seams in the collector, the
  buffer, R1, the checkpoint commit and the parser, and its tests;
- **the T2 near-tie judge** (`3260d7ae0`), because it applied cleanly and changes no number: it runs only after a
  parity judgement has already failed, and turns a near-tie flip (X60-T s3501's FATAL class) from a crash into a
  pass. It also types the dtype of two zero tensors in the damage op, which sits in every forward pass, training
  and play; §4.2's probe found them float32 either way.

**The config version stays 161.** The two fields (`magnet_kl_coef`, `magnet_refresh_steps`) are un-versioned
additions with defaults 0.0 and 2,000,000; every run at this pin records them. Two consequences, both executed: the
plain side pin REFUSES to load a config that carries them, so **both crosses are played at the first CE-3d pin**, which
loads the v161 finals trained at `f4b25238` (C0's and X67's: neither config has a magnet key) as `f4b25238` does
(§5.4's load check, one final of each); and main loads a CE-3d config keeping the recorded values.

**The training pin's commit (Amendment 3)** is main's fix of the startup update-fit check
(`gen3_update_fit_fill_inputs_v1`, main `13c50a5e6`) and nothing else: `src/agents/training/update_fit.py`, its test,
and the magnet's integration test. At `333a73dd` the three files are byte-identical to main's before the fix, and at
the training pin to main's after it, so the commit IS main's change to them. The check runs once at startup, on the
card only: one dry update on a fixture rollout, restored exactly, to refuse a first update that would not fit. No
collector fills the fixture, so at `333a73dd` it had no magnet column and the lever's own `train()` refused it:
every `--magnet-kl-coef > 0` launch on the card died there, before step 0 (§15). At the training pin the dry update
fills the fixture's column as the collector does (`magnet.fill_column` through the run's own T2 magnet slot, from the
magnet built off the loaded weights), so the check measures the update the run will train with; the magnet is
neither skipped nor zeroed. It changes no loss, no collector, no version, no compile key and no play code, and with
the magnet off the dry update is the one `333a73dd` runs.

### 4.2 Tested at the pin, CPU only (2026-11-11, rule 5b, no lease)

- the routine tier: 12,721 passed, 2 failed, neither the commit's: one heavy-slots test that fails only when the
  suite itself runs inside a memory-capped slot (17 passed alone, three times), and one test that walks the live
  `models/` archive and already fails at plain `f4b25238`;
- the lever's own fast tests (178 passed), its slow real-core integration test (6 passed: the slot, the column, the
  refresh, a resume serving the same column bit for bit), and 349 targeted trainer and parity tests;
- **the learner golden is bit-identical at the pin in bf16 and in fp32 with β = 0**, and the near-tie judge's two
  `dtype=` arguments are no-ops in float32 (a probe over 594 + 270 calls);
- **a swap fork smoke** (`--debug`, forked from a smoke final whose record carried `ent_coef` 0.1, `--ent-coef 0
  --magnet-kl-coef 0.05 --magnet-refresh-steps 2000`): exit 0; the `[MAGNET]` line printed §7's banner string word
  for word; the hyperparameter table printed `ent_coef   | 0`; `metadata.ent_coef` and `cli_args.ent_coef` 0.0;
  `model_config.json` `magnet_kl_coef` 0.05; `final_model.magnet.pt` written; `train/magnet_kl` positive
  (2.6e-05 rising to 4.7e-04); the learner-freeze checks released with 6 passes;
- the first registration's magnet smoke and control smoke, and a `main.h2h play` between their finals.

All of the above was run at the first CE-3d pin. **At the training pin** (Amendment 3; the same shell rules):
- **fail-on-revert:** with `update_fit.py` as `333a73dd` has it, the new CPU test of the dry update with the magnet
  on fails with the incident's own error (`MagnetError … no magnet column`, raised in `PPO._magnet_args`);
- with the fix: `update_fit_test` (it drives `check_update_fits`' dry-update path on the CPU with the magnet on, the
  card's verdict stubbed; refuses by name, before `train()`, a lever whose fill input nothing provides; and holds
  the collector's fill to a declared table, so a new collector-filled input fails the build),
  `magnet_integration_test` (slow, real core and real T2: the dry update's column is the one T2's magnet slot
  serves, the learner is restored bit for bit, and the next real fill is unchanged), the magnet's and the learner
  golden's tests: 168 passed, 24 of them slow; the static tier, 261 passed;
- §7's `checkargs` and launcher dry runs, re-run for all three seeds.
- **Not re-run at the training pin:** the routine tier (the commit touches one runtime module, imported only by the
  trainer's startup) and the smokes (a `--debug` run is CPU, where the fit check does not run).

### 4.3 The confound, stated: the swap forks and the C0 finals differ by more than the lever

The decision reads forks trained at `9e97cad5` (the first CE-3d pin `333a73dd` + the startup fit check's fix)
against finals trained at `f4b25238`. **This is a REPORTED caveat, not a guard: nothing in the rule corrects for it,
and the verdict is quoted with it.**

| difference | what it is | evidence that it is not the result | status |
|---|---|---|---|
| the magnet code | the lever itself, live in the swap arm | it IS the treatment | — |
| the near-tie judge | parity judging in the T2 service (training and play), reached only after a parity judgement has failed; two `dtype=` arguments in the damage op's forward | the judge decides only whether a failed check is fatal and serves no logit; the two tensors were float32 with and without the arguments (§4.2's probe) | CPU only |
| the startup fit check's fix (Amendment 3) | before the first rollout, on the card: the dry update serves the fixture's rows through T2's magnet slot once, trains on them and is restored | the learner comes back bit for bit and the next real fill is unchanged (the pin's slow integration test); T2 is a function of its rows, so the pass moves no played bit. It does advance T2's serving counters by the fixture's rows | CPU only; **UNVERIFIED on the card** (§4.4 item 7) |
| everything else in the learner | nothing: 44 + 3 paths, enumerated and checked | **with β = 0 the learner golden is bit-identical in bf16 and fp32 at the first pin**, and the fix's three paths leave a β = 0 dry update as it was | **CPU only; never run on the card** |
| the day | C0 trained 2026-10-21 to 11-10; the swap after 11-11 | none (same card, same interpreter pin, one `torch_version` and `init_num_threads` enforced by the plan) | X67 carried the same difference |
| the play pin | both CE-3d crosses are played at `333a73dd`; X67's was played at `f4b25238` | the row above; no play or inference path builds a magnet (the pin's h2h smoke printed none); a ledger row written at the pin carries the regime id, team set and Rust core source and schema hashes of X67's banked rows ([`x67_loadcheck_at_pin.txt`](x67_loadcheck_at_pin.txt)) | **UNVERIFIED**: no game was played at both pins |

**Why this was judged acceptable for a one-arm read** (and not the fallback, a second arm of `--ent-coef 0.05` with
no magnet at the CE-3d pin, ≈ 26 GPU-h in all): the only code the pin adds to a β = 0 learner changes no number
it computes, and the β = 0 learner reproduces `f4b25238`'s golden to the bit, so a control at the CE-3d pin would be
C0's program again; the day difference is the one X67's PASS already carries. What a same-pin control WOULD buy is
the GPU confirmation of the CPU evidence. It is not bought here, and §10 limits the licence accordingly.

### 4.4 UNVERIFIED on the GPU, every one of them (no lease was held)

1. the magnet's cost: one more full-bucket T2 slot of CUDA-graph captures, one more weight copy on the card, one
   inference pass over the buffer per rollout (ESTIMATED ≈ +3 %). Read `magnet/forward_ms` on the first fork;
2. R1's compiled program with the magnet's tenth argument, the compile gate and the canary with it. The dead attempt
   of §15 is the only evidence: at `333a73dd` its launcher log shows T2's magnet slot declared, R1 compiled, the
   region's parity gate PASS and the compile lock, once. The canary was armed and never ran (its first check is at
   update 10);
3. the size of the KL's floor: T2 serves fp32, R1's trunk is bf16, so the term is a small positive number, not 0,
   right after a refresh;
4. a resume with the sidecar on the card (the CPU integration test passes);
5. `ent_coef` 0 beside the magnet on the card (the CPU smoke only);
6. that a β = 0 learner at this pin trains on the card as `f4b25238` does (§4.3: CPU golden only);
7. the fit check's dry update with the magnet on (Amendment 3): the first flush through T2's magnet slot, before
   any rollout; the size of its `fill_inputs` segment in `update_fit.json`; and the check's verdict. The first
   relaunch is the first time this code meets the card. A FATAL there is §8's lever-own FATAL again.

## 5. The meter and the rule: X67's, unchanged

### 5.1 The crosses

Two crosses, **one that decides and one that is only reported**, with the same settings: head to head, both sides
greedy, the players' eval team builder, schedule seed 0, `--oracle-reveal-mode off`, protocol
`gen3_eval_protocol_v1_h2h`, purpose `ab`, label `ce3d_magnet_screen`, played from a clean detached checkout **at
the first CE-3d pin `333a73dd`** ([`cross/drive_cross.sh`](cross/drive_cross.sh), whose fourth argument picks the
cross). The play pin did not move with Amendment 3: the training pin's three paths are a startup check of the
trainer and two tests, which no play or inference path imports, so the two pins play the same program and
[`x67_loadcheck_at_pin.txt`](x67_loadcheck_at_pin.txt) stands.

| rows × columns | pairs a cell | games | family / request | role |
|---|---:|---:|---|---|
| 3 swap forks × c C0 finals (5 ≤ c ≤ 8; 8 expected) | 1,000 mirrored | ≤ 48,000 | `ce3d_swap_vs_c0_h2h` / `ce3d_swap_vs_c0` | **DECIDES** |
| 3 swap forks × X67's 3 finals (`rb_x35_x67ent_s6701`–`s6703`) | 1,000 mirrored | 18,000 | `ce3d_swap_vs_x67_h2h` / `ce3d_swap_vs_x67` | REPORTED (§5.4); decides nothing |

The deciding cross is X67's cross with the swap forks as its rows: the same columns, cell size, statistic and
classes, so its D sits directly beside X67's +1.97 pp (§9). §5.2 and §5.3 are about it; §5.4 is the reported one.

### 5.2 The plan

[`cross/plan_cross.py`](cross/plan_cross.py) is run ONCE, after the third fork's arm-end checks. The three swap
forks are REQUIRED; the five banked C0 finals (s3501–s3505) are REQUIRED and s3506–s3508 join when they pass (a
top-up final that exists and fails a guard REFUSES the plan). Guards, every run: `final_model.zip`; one
`pin_history` entry at ITS pin (a swap fork: the training pin `9e97cad5`; C0 and X67: the side pin) from
280,000,170 to ≥ 300,000,170; lineage FORK of P\*'s sha256; the seed; `n_epochs`
10; `ent_coef` (swap 0.0, C0 0.05) in `metadata.ent_coef` and `cli_args.ent_coef`; `model_config.json` (swap:
`magnet_kl_coef` 0.05 with `magnet_refresh_steps` 2,000,000; C0: both keys absent); `final_model.magnet.pt` present
for a swap fork and absent for C0; K9(b) PASS; one `torch_version` and one `init_num_threads` across all. Executed
2026-11-11, and again at Amendment 3 with the swap forks' pin moved: `--columns-only` → `columns OK`, all eight C0
finals; the full plan REFUSES (no swap final exists).
`--cols x67` plans the reported cross with the same guards (§5.4).

### 5.3 The statistic and the classes ([`screen_rule.py`](screen_rule.py))

X67's `rect_stat` and `decide`, copied (a measurement's script is a frozen record and is never imported). For the
matrix h of cell scores in pp (r = 3 rows, c columns, 1,000 mirrored pairs a cell):

- **D = mean(h) − 50**, the swap arm's lead over C0.
- s²_R = max(variance of the row means, CELL_VAR / c); s²_C = max(variance of the column means, CELL_VAR / r), with
  CELL_VAR = 2500 / 2000 = 1.25 pp², one cell's binomial variance.
- V = s²_R / r + s²_C / c, Welch–Satterthwaite df, 90 % interval D ± t(0.95, df) · √V.

| class | condition | meaning |
|---|---|---|
| **PASS** | the interval's lower end > 0 | the swap forks beat the C0 finals head to head (one-sided 95 %) |
| **WORSE** | its upper end < 0 | the swap loses to C0 |
| **NOT-SHOWN** | otherwise | no difference this screen can see |
| **INCONCLUSIVE** | fewer than three valid swap forks, fewer than five C0 columns, a missing or short cell | never interpreted |

Rule 8: an end within 1e-9 pp of 0 satisfies neither PASS nor WORSE. No look 2, no substitute seed, no minimum
effect size. `decide` also reports the interval with both variances floored at the fresh-run seed SD (1.76 pp), the
pessimistic reading, and D over the banked columns (3501–3505) and the top-up columns (3506–3508) separately.

**The self-check** (`screen_rule.py self-check`, [`selfcheck.log`](selfcheck.log): OK, run with the pin's code): D
and SE equal `main.h2h.cross.cross_stat`'s on a banked 8 × 8; **`decide` on X67's banked 3 × 8 matrix reproduces
X67's banked verdict** to 1e-9 (PASS, D 1.96875, [1.3456, 2.5919], df 6.22, both floors binding, the fresh-floor
interval); the hand cases for each class, the floor, the tie and the INCONCLUSIVE shapes read as registered.

### 5.4 The REPORTED cross against X67's finals (Amendment 2). It decides nothing

**Why.** §1 and limit 1 (§10): the swap adds the anchor AND removes the standing push toward uniform, and X67
already showed that a smaller push wins. The deciding cross cannot split the two. X67's finals are the nearest
banked policies that have the smaller push and no anchor (P\* + 20M at `ent_coef` 0.0125, no magnet), so the swap
forks are played against them directly.

**Shape.** 3 × 3: rows = the three swap forks, columns = `rb_x35_x67ent_s6701`, `s6702`, `s6703`, all REQUIRED,
1,000 mirrored pairs a cell (18,000 games), every setting of §5.1, its own family and request so that the deciding
family holds only the deciding cells. Played ONCE, after the third fork's arm-end checks, before or after the
deciding cross: neither reads the other's cells. `drive_cross.sh <PIN_DIR> <WORK_DIR> cpu|cuda x67`, in a work
directory of its own.

**The pin it plays at, and why X67's finals load there.** The first CE-3d pin `333a73dd`, as the deciding cross and
for the same reason (the swap forks' configs carry the magnet's two fields, which `f4b25238` refuses). X67's forks trained at
`f4b25238` exactly as C0's did: a version-161 `model_config.json` of the same 107 keys with neither magnet key, and
the pin's two new fields are un-versioned with defaults (0.0 and 2,000,000), so the pin builds the same network
from it. **Executed** ([`x67_loadcheck_at_pin.txt`](x67_loadcheck_at_pin.txt), CPU, no lease, a scratch ledger):
`main.h2h play` at the pin loaded `rb_x35_x67ent_s6701/final_model.zip` (sha256 `94f173f9…`, row 0 of X67's banked
record) and `rb_x35_c0_s3501/final_model.zip`, brought the engine up and played 4 pairs, exit 0; the ledger row it
wrote carries the regime id (`adfdbee824c9eb0a`), team set and Rust core source and schema hashes of the rows X67's
cross banked at `f4b25238`. It is a load check, not a read: 4 pairs, purpose `audit`, quoted nowhere.

**The plan** (`cross/plan_cross.py --cols x67`): §5.2's guards with X67's values. Each X67 final must have one
`pin_history` entry at `f4b25238` from 280,000,170 to ≥ 300,000,170, the lineage of a FORK of P\*, its seed,
`n_epochs` 10, `ent_coef` 0.0125 in both records, NO magnet key and NO sidecar, K9(b) PASS, and the swap forks'
`torch_version` and `init_num_threads`. Executed 2026-11-11: `--columns-only --cols x67` → `columns OK: [6701,
6702, 6703]`; the full plan REFUSES (no swap final exists).

**What is reported** (`screen_rule.report_x67`, written to `report_x67.json`, never to a `verdict.json`): §5.3's
statistic on the 3 × 3, unchanged — **D = mean(h) − 50** (the swap forks' lead over X67's), the same binomial floors
(CELL_VAR / 3 on each side), Welch df, and **the 90 % interval**. No class. `decide` refuses a 3 × 3 (the self-check
pins it), so this matrix cannot produce one by accident.

**The reading rule, written now.** It is quoted only when the registered verdict (§5.3) is PASS; with any other
verdict there is no gain over C0 to attribute, and D and its interval are reported with no reading.

| the reported interval | the reading |
|---|---|
| its lower end > 0 (D > 0 and the interval above 0) | **evidence that the anchor adds something beyond lower entropy** |
| otherwise: it holds 0 (D ≈ 0), or D is negative | **the swap's gain over C0 is plausibly the lower uniform push alone** |

Rule 8: a lower end within 1e-9 pp of 0 is not above 0. An interval wholly below 0 is the second reading, and the
report says the swap forks lost to X67's.

**It decides nothing.** It changes no class, no label and no licence of §10; it is not a second look at the
registered question; a missing or short cell leaves it unread (`"read": false`) and leaves the verdict untouched; it
is not re-played, extended or given a substitute seed.

**How much it can see** ([`power_x67.log`](power_x67.log): `screen_rule.py simulate-x67`, §6's model at 3 × 3,
50,000 draws a row):

| true seed SD of the swap forks / of X67's finals, pp | half-width, pp | P(lower end > 0) at D = +1 | +2 | +3 | at 0 |
|---|---:|---:|---:|---:|---:|
| 0.32 / 0.59 | 1.36 | 0.24 | 0.79 | 0.98 | 0.007 |
| 0.5 / 0.5 | 1.38 | 0.23 | 0.78 | 0.98 | 0.007 |
| 1.0 / 0.5 | 1.71 | 0.20 | 0.58 | 0.87 | 0.021 |
| 1.76 / 0.5 (the swap forks spread like fresh runs) | 2.44 | 0.15 | 0.36 | 0.60 | 0.040 |

Both sides carry their variance on 2 df. **A true +1 pp of the anchor reads as "lower push alone" three times in
four**, and an anchor effect of X67's own size (+2 pp) is seen four times in five when the swap forks spread as
X67's did, one time in three when they spread like fresh runs. X67's forks read a row-mean SD of 0.32 pp in their
own cross, so the first two rows are the likely ones for the columns.

**Its limits, stated before the number:**
- **X67 is not the clean control.** It keeps a quarter of the bonus (`ent_coef` 0.0125); the swap has none. A D
  above 0 is therefore also what "no standing push beats a small one" would give, and "evidence" in the first
  reading means that and no more. The clean control is an `--ent-coef 0` arm with no magnet at the CE-3d pin, which
  §10 names as the follow-up and this file does not register.
- **The dose differs.** All three X67 forks are `LR-MOVED` (`lr_median` 0.00008372 against C0's 0.0001005); whether
  the swap forks are is read at their end (§8) and quoted beside D.
- **Pin and day** (§4.3, limit 2): X67's forks trained at `f4b25238` on 2026-11-10 and 11; the evidence that the
  pin does not matter is CPU only.
- Head to head between siblings (limits 3 and 6).

## 6. What the screen can see ([`power.log`](power.log): `screen_rule.py simulate`, 50,000 draws a row)

At 3 × 8 (the expected shape; the log also holds 3 × 5):

| true seed SD of the swap forks / of the C0 finals, pp | half-width, pp | P(PASS) at D = +0.5 | +1 | +1.5 | +2 | +3 | false PASS at 0 | P(WORSE) at −2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.32 / 0.59 (what X67's cross measured for its forks and for these columns) | 0.77 | 0.21 | 0.67 | 0.94 | **0.99** | 1.00 | 0.016 | 0.99 |
| 0.5 / 0.5 (the fork evidence, X67 PREREG §6) | 0.84 | 0.20 | 0.57 | 0.86 | **0.97** | 1.00 | 0.024 | 0.97 |
| 1.0 / 0.5 | 1.33 | 0.15 | 0.33 | 0.54 | **0.72** | 0.92 | 0.045 | 0.72 |
| 1.76 / 0.5 (the swap forks spread like fresh runs) | 2.35 | 0.11 | 0.19 | 0.29 | **0.40** | 0.62 | 0.055 | 0.40 |

- If the swap forks spread like X67's did, the screen reads a true +2 pp (X67's own size) as PASS nearly always and
  a true +1 pp two times in three. If they spread like fresh runs it reads +2 pp two times in five. **The three rows
  carry the variance on 2 df; more games or more columns do not help.**
- The realised half-width is printed with the verdict. A NOT-SHOWN means "no gain above that half-width", not "no
  effect".
- A magnet could widen or narrow the fork-to-fork spread; that is unknown and is what the last two rows are for.

## 7. The launch commands, and their validation

For the Training session, from the main checkout at or after Amendment 3's commit, holding the lease, one run at
a time in the order of [`chain_items_ce3d.json`](chain_items_ce3d.json). **First, once:** move fork 1's dead attempt
aside (§15; e.g. `mv models/rb_x35_ce3dswap_s7321 models/rb_x35_ce3dswap_s7321.FITCHECK_NO_MAGNET_COLUMN`, never
deleted). While it is in place the kit HOLDs every production-side launch at `pin-check`, and this run's own
"run dir exists" pre-check HOLDs too.

```bash
STAGE=designs/research_state/measurements/x35_breakers_stage_2026-10-17
PIN=9e97cad5709da21a1f9d4d8846096c87eb17b1d0       # tag ce3d-pin-9e97cad5; NOT the side pin, NOT the first CE-3d pin
python -m main.x35.kit stage-check --stage $STAGE --arm ce3d_swap --seed $s    # ALLOWED: "CE-3d PREREG present"
bash $STAGE/launch_x35.sh ce3d_swap $s --pin $PIN --lease-owner "<owner>" --thinning       # the kit's DRY RUN (default)
bash $STAGE/launch_x35.sh ce3d_swap $s --pin $PIN --lease-owner "<owner>" --thinning GO    # only on the orchestrator's GO
```

for `$s` = 7321, 7322, 7323. The kit resolves each to `python -m main.launcher --restart-interval-hours 24
--pin-commit $PIN <argv>`; the first seed:

```
--model /home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip --behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cuda --ent-coef 0 --magnet-kl-coef 0.05 --seed 7321 --run-name rb_x35_ce3dswap_s7321
```

Validated on 2026-11-11 at the training pin (every transcript below was re-made at Amendment 3), CPU only, no lease,
`CUDA_VISIBLE_DEVICES=""`, nice 19 on the agent cores, nothing created:

- `python -m main.checkargs --argv "<argv>" --pin $PIN` → `checkargs_ce3d_swap_s<seed>.txt`, all three: **exit 0**,
  every flag accepted by the PINNED parser, every ARCH-surface key the production mirror's, the two lever rows
  TYPED (§2), `✓ fork arch`.
- `python -m main.launcher --dry-run --restart-interval-hours 24 --pin-commit $PIN <argv>` →
  `dryrun_ce3d_swap_s<seed>.txt`, all three: **role FORK of `rb_x26_s2601`**, +20,000,000 steps, the pin (source
  `pin_commit`), both lever rows TYPED, 10.97 GiB REQUIRED (it read 14.33 at Amendment 1: the shared compile cache
  now holds 22.00 GiB against its 20.00 GiB cap, so its row counts no growth). **s7321's transcript was made with
  the dead attempt's directory still in place**: its run-dir line reads `[EXISTS — a real launch WRITES INTO IT]`
  and it lists that directory's 14-snapshot pool; after the move it reads as the other two do.
  **Exit 3, "would NOT launch", on exactly two refusals,
  both this unit's own shell and both expected:** the CUDA-visibility preflight (`CUDA_VISIBLE_DEVICES=''`: no
  lease) and the inherited-nice preflight (the shell ran at nice 19 by rule 5b; a GPU launch is refused above nice
  10). The kit's own dry run re-runs it with the card visible, at the launch shell's priority, before every GO. The
  transcripts' `showdown deps` line names the amending unit's worktree, which no longer exists.
- `python -m main.x35.kit pin-check --pin $PIN --side production --arm ce3d_swap --stage $STAGE`
  ([`kit_checks_ce3d.txt`](kit_checks_ce3d.txt)): the pin itself passes ("the CE-3d pin 9e97cad5709d = the first
  CE-3d pin 333a73ddda78 + one commit, exactly the 3 enumerated path(s) …; 333a73ddda78 = the side pin f4b25238def5
  + one commit, exactly the 44 enumerated path(s)"). Against `models/` as it stood, the check exits 1 on the dead
  attempt ("no recorded pin … registered as a startup-dead attempt …, but it is still IN PLACE"). Against a scratch
  view of `models/` with that directory under its moved-aside name (symlinks; nothing under `models/` written) it
  exits 0 and lists the attempt as "a STARTUP-DEAD attempt at 333a73ddda78 …, moved aside". The first CE-3d pin and
  the plain side pin each exit 1 ("the CE-3d arm launches only at CE3D_PIN"). `kit required-check` passes at the
  pin; a fourth seed and a retired seed are refused.
- `ce3d_test`: the kit's argv equals the three validated files token for token and is C0's plus the four lever
  tokens; `check_validation` PASSes the six transcripts (the two shell refusals modelled as absent) and FAILs them
  when either typed lever is absent, another value, or shown by one tool only, and when read as C0's or X67's.
- `python -m main.ops.arm_chain plan <scratch chain> --check-pins --describe` on the three draft items →
  [`arm_chain_plan_ce3d.txt`](arm_chain_plan_ce3d.txt): PLAN OK, the kit agrees on all three; a fourth seed and a
  dropped arm REFUSE. **The plan does not judge an arm's own pin** (an item at the plain side pin, or at the first
  CE-3d pin, still plans OK): that check is the launch script's `kit pin-check`, above.

**The chain items** ([`chain_items_ce3d.json`](chain_items_ce3d.json); `go_by` is the orchestrator's to fill): the
arm, the seed, the pin, `thin: planner`, `flags: ["--thinning"]`, the throughput gate (1,000 steps/s within 60 min,
first line within 90), `on_throughput_fail: page`, `on_void: stop`, and a banner the child's log must show within
30 min:

```json
{"expect": "beside the entropy bonus (ent_coef 0; the KL's own entropy part adds 0.05 to that weight)",
 "void": ["ent_coef   | 0.05 ", "ent_coef   | 0.1 ", "ent_coef   | 0.0125 ", "[ModelVersion] FATAL"], "within_min": 30}
```

The expected line is the one `magnet.attach` prints once the magnet object is built, before the Rust env starts. It
names the entropy coefficient (`ent_coef 0;`, which no other value can print) and β (`adds 0.05`), so **one string
guards both halves of the lever**; the pin's swap smoke printed it word for word, and `ce3d_test` reads the format
from the pin's source. The VOID lines are the hyperparameter table's rows for the three entropy coefficients a
mistyped or inherited launch could carry (the swap's own row is `ent_coef   | 0 `, which matches none). A fork that
starts without its magnet, or with an entropy bonus, is stopped within the half hour.

**Disk.** `disk_need_gib` is **16.0** a fork: Amendment 1's dry run's 14.33 plus the magnet's sidecars, which the
launcher's arithmetic does not count (one `<checkpoint>.magnet.pt` a checkpoint, each at most a checkpoint zip's
54.7 MiB; the production size is unmeasured). It is kept at 16.0 although the dry run now reads 10.97: the figure
is a floor on free space, and the higher one is the safe one. The thinning planner does not recognise the sidecars
and keeps them, so up to
≈ 1.1 GiB a fork stays behind after thinning.

**`launch_x35.sh` itself** was not run by any amending unit (no lease). The Training session ran it once, for fork
1 at the first pin: its dry run and its GO passed every pre-check, and the launch died in the trainer's startup
(§15). It has not been run at the training pin. **Not executed at any amendment:** either cross's play command (it
writes eval-ledger rows). Executed: the plan's column
check for both crosses, the rule, the report and their self-check; the driver's `family-register` line for both
families and `read_cross.py` for both crosses, with the pin's code against an empty SCRATCH ledger (the deciding
read: INCONCLUSIVE, 24 missing cells; the reported one: `"read": false`, 9 missing cells), never on a played cell;
and `drive_cross.sh` up to its HOLDs (a copy pointed at this amendment's files: each cross's plan REFUSES with no
swap final, exit 3; a work directory used by the other cross, exit 5; a checkout not at the pin, exit 5).

## 8. VOID guards, labels, the canary family

**VOID** (any one ⇒ that fork is VOID; fewer than three valid forks ⇒ INCONCLUSIVE):
- not finished: the arm chain's completeness check (rc 0, `Last step` ≥ 300,000,170, `final_model.zip`, the newest
  checkpoint within 98,304 steps of it);
- K9(b) not PASS at the arm's end;
- the launcher log names another pin than the training pin `9e97cad5`, or the run is not `FORK of …/rb_x26_s2601`;
- the post-spawn pool-seed check fails;
- **either half of the lever is not in the record**: §7's banner (the chain STOPS the run); any of §5.2's plan
  guards on `ent_coef`, `magnet_kl_coef`, `magnet_refresh_steps` or the sidecar; `main.dose` reads an epoch count
  other than 10 or an effective batch other than 65,536;
- **a magnet that never refreshed**: `python -m main.ops.tb_read <run> --tag magnet/refreshes` shows no point above
  0 (nine refreshes are due; the tag is emitted only with the magnet on).

**A FATAL that is the lever's own on the first fork** (the magnet's T2 slot, the compile gate or the canary with
the magnet's argument, a sidecar refusal) **STOPS the screen**: no further fork launches, nothing is read, and a fix
is a new pin and a dated amendment or a new registration. The orchestrator decides which. **Invoked once** (§15:
the startup fit check, before step 0; the fix is the training pin and Amendment 3). A launch that dies before its
first training step is a launch failure, not a fork: the VOID guards above judge the fork that trains.

**Throughput < 1,000 steps/s in the first hour ⇒ PAGE**: HOLD and keep training; the orchestrator decides.

**Labels, reported beside the class and never changing it** (end entropy as §9 defines it):
- `LR-MOVED`, a fork: its `lr_median` (`main.dose`, as printed) is below 0.0000905 or above 0.0001105, more than
  10 % from C0's 0.0001005 (X67's label; all three X67 forks carried it at 0.00008372). The controller is part of
  the recipe, so a rate the swap moved is the lever's own effect, but the reader must know the dose differs.
- `ENT-LOW`: the HIGHEST of the three swap forks' end entropies is below 1.00406 nats, the lowest of the eight
  banked C0 finals' (`../x67_entropy_screen_2026-11-09/record/diag9_entropy.json`). The swap arm then ended sharper
  than every control, as X67's arm did (0.659–0.687), and a PASS may be X67's effect (§1).
- `ENT-HIGH`: the LOWEST swap fork's end entropy is above 1.01763 nats, the highest banked C0 final's.
- Neither label when the ranges overlap; a difference within 1e-9 nats of a bound counts as overlap (rule 8).

**Canary family: none.** The swap arm is in no family of `canary_decide.FAMILY_NAME_PATTERNS`: R1 takes the magnet's
column and no healthy-pass ratio was ever measured with it, so the C0 bar (2.64) does not transfer. A canary FATAL
on a swap fork therefore HOLDs (`canary_decide`: "no family bar") and goes to the orchestrator; it is never resumed
on a bar typed by the Training session.

## 9. REPORTED diagnostics (they decide nothing)

- **Entropy, the rate and the KL**: `python -m main.ops.tb_read <run> --tag train/entropy_loss --tag
  train/learning_rate --tag train/approx_kl` for the three forks; entropy in nats = −`entropy_loss` (logged with
  `ent_coef` 0 as with any other); the end value is the mean of a run's last 20 points (X67's definition; the C0 and
  X67 values are banked in X67's record).
- **The magnet's own meters**: `train/magnet_kl` (expected: a saw-tooth that rises toward each refresh and drops at
  it; its floor right after a refresh is §4.4's unverified bf16 number), `magnet/age_steps`, `magnet/refreshes`,
  `magnet/forward_ms` (the cost, against §11's + 3 % estimate).
- **Against X67's finals, directly**: §5.4's reported cross, its D, 90 % interval and (with a PASS) its reading.
- **Beside X67**: D and its interval next to X67's +1.97 pp [+1.35, +2.59], and the difference of the two D. No
  interval and no class is registered for that difference: the two crosses share their columns but were played at
  different pins and on different days (§4.3). Where this indirect line and §5.4's direct cross point different
  ways, both are quoted and neither is preferred.
- **The pessimistic interval** and the banked / top-up column split (`decide`'s reported block).
- No Kakuna game is planned. One that is played later carries its team-clustered interval and a dated NOTE here,
  written before any game, naming the finals, the panel and the count.

## 10. What each class licenses

- **PASS** ⇒ at β 0.05 and a 2M cadence, a P\* + 20M fork trained with the trailing anchor in place of the entropy
  bonus beats the unchanged recipe's forks head to head. **It licenses REGISTERING a follow-up, and nothing else**:
  on new seeds, with the control that separates the anchor from the missing uniform push — an `--ent-coef 0` arm
  with no magnet at the CE-3d pin, or the magnet against an arm matched on REALISED entropy — and, if the claim is
  to be quoted, a C0-recipe control at the same pin (§4.3).
  - A PASS with `ENT-LOW` is reported as "the swap won and ended sharper; not separable from X67's effect here".
  - A PASS whose D is not above X67's +1.97 pp shows no gain over simply lowering `ent_coef`, and says so.
  - A PASS is quoted with §5.4's reading of the cross against X67's finals, word for word. Neither reading widens
    or narrows this licence: the follow-up it permits is the same, and the reading says which control it most needs.
  - A PASS with `LR-MOVED` goes to the orchestrator with both numbers.
- **WORSE** ⇒ this spelling closes: the backlog row records it, and no magnet follow-up is queued unless a new case
  is made for a different form.
- **NOT-SHOWN** ⇒ no gain above the cross's own half-width; the row is closed as "not shown at β 0.05 in place of
  the bonus, +20M", and a different form needs its own case.
- **INCONCLUSIVE** ⇒ never interpreted; a re-run is a new registration.

**This file moves no recipe.** No class puts a magnet or an `ent_coef` into `production_config.json`,
`design_learner_recipe.md` or any BOUNDARY-1 arm; the screen never holds BOUNDARY-1's launch, and it neither waits
on nor informs the owner's decision on X67's PASS.

**The limits, stated before the number:**
1. **One loss term, two effects** (§1). The swap adds the pull toward the magnet AND removes the standing push
   toward uniform. No class separates them; `ENT-LOW` and §5.4's reported cross against X67's finals are evidence,
   not a test (X67 keeps a quarter of the bonus, and that cross sees a +1 pp effect one time in four).
2. **The control was trained at another pin and on other days** (§4.3). The evidence that this does not matter is
   CPU only. A verdict from this screen is quoted with that caveat, and a claim that needs it removed needs the
   same-pin control.
3. **The meter is not the card's question.** The backlog asks whether a trailing regulariser damps the treadmill
   and exploitable tie-breaks. Head to head against a sibling arm measures neither exploitability nor cycling; no
   best-responder read is registered here.
4. **A late fork, a short horizon.** Nine refreshes over 20M steps, from a policy shaped by 280M steps without a
   magnet. Nothing is claimed about a magnet from step 0.
5. **One spelling**: one β, one cadence, no entropy bonus beside it.
6. **Head to head against siblings is not strength against others** (X67 PREREG §10).

## 11. Cost

| item | GPU-h | CPU | note |
|---|---:|---|---|
| swap s7321–s7323, +20M each | **≈ 13** | — | ≈ 4.10 GPU-h a fork (the mean of the 20 X35 production forks' dose records, X60-T PREREG §5.1) + the magnet's pass (≈ +3 %, ESTIMATED, never measured) and a first compile of R1 with the magnet's argument |
| the deciding cross, ≤ 48,000 games | ≈ 0.35 in a GPU window | or ≈ 7.7 h on the three agent cores (≈ 23 lane-h) | from X67's cross: 48,000 games in 27,677 s on three threads; the GPU figure is X67's ESTIMATE |
| the reported cross against X67's finals, 18,000 games | ≈ 0.13 in a GPU window | or ≈ 2.9 h on the three agent cores (≈ 8.7 lane-h) | the row above scaled by 18,000 / 48,000; both figures ESTIMATED |
| the pin checkout's Rust build | 0 | minutes, under `mem_cap` | once |
| §9's reads, the dose read | 0 | seconds | |

≈ 13 GPU-h in all, and ≈ 32 CPU lane-h (≈ 10.6 h on the three agent cores) when both crosses are played on the
CPU. **Hard stop:** three forks, the deciding cross and its one read, the reported cross and its one report;
nothing here extends.

## 12. What this file does NOT do

It registers no magnet beside an entropy bonus, no `--ent-coef 0` arm without a magnet, no same-pin control, no
second β or cadence, no fresh-run magnet arm, no class or licence on the cross against X67's finals, no
exploitability read and no Kakuna game. It does not adopt or delay X67's PASS, amends no BOUNDARY-1 document, and
changes no baseline and no production recipe. CE-3d has no `main.reads` rule or `reads/` record: its verdict is
`cross/read_cross.py`'s `verdict.json`, written by `screen_rule.decide`; the reported cross writes
`report_x67.json` (`screen_rule.report_x67`) and no verdict.

## 13. Amendment 1 (2026-11-11, before any launch)

**Blind.** Written the day of the registration (`08b998d12`), with no CE-3d run directory in `models/`, no CE-3d
family or request in the eval ledger, and no cell played. The design change is the orchestrator's, on the
registration's own §1 caveat (the magnet arm sat at an entropy weight of 0.10, the other side of X67's PASS).

What changed against `08b998d12`, and why:

| | the registration | Amendment 1 | why |
|---|---|---|---|
| arms | `ce3d_mag` (`--magnet-kl-coef 0.05` beside `ent_coef` 0.05; seeds 7301–7303) and `ce3d_ent` (`--ent-coef 0.1`; 7311–7313) | ONE arm, `ce3d_swap` (`--ent-coef 0 --magnet-kl-coef 0.05`; seeds 7321–7323). Both old arms are removed from the kit; their seeds are retired, never launched, never reused | the swap keeps the entropy weight at C0's 0.05, so the banked C0 finals are its control; the 0.10 regime is one X67 points away from |
| the decision | magnet × control, 3 × 3 at 2,000 pairs a cell | swap × the banked C0 finals, 3 × c at 1,000 pairs: X67's cross and X67's rule (`decide`, reproduced on X67's record) | one lever in the loss against a banked control; D directly beside X67's +1.97 pp |
| reported crosses | each arm against C0, and the indirect line | none | the deciding cross IS the one against C0 |
| the control's pin and day | shared with the magnet arm (both new forks at the CE-3d pin) | NOT shared: C0 at `f4b25238`, trained earlier | accepted as a REPORTED caveat (§4.3) on the bit-identical β = 0 learner golden, CPU only; the same-pin fallback (≈ 26 GPU-h) was judged not needed for a screen that licenses a follow-up only |
| limits | the meter, the horizon, the spelling, siblings | + limit 1 (one term, two effects) and limit 2 (the pin and the day), with the `ENT-LOW` / `ENT-HIGH` labels replacing `ENT-SPLIT` and `LR-SPLIT` | the swap removes the uniform push as well as adding the anchor; with no second new arm there is nothing to split against |
| canary family | the control in C0's family; the magnet arm in none | the swap in none; `ce3dent` removed from C0's family pattern | no CE-3d arm is C0's graph any more |
| power | 3 × 3 at 2,000 pairs | 3 × 8 (and 3 × 5) at 1,000 pairs, X67's simulation | the new shape |
| cost | ≈ 26 GPU-h, ≈ 63 CPU lane-h | ≈ 13 GPU-h, ≈ 23 CPU lane-h | three forks, one cross |
| files | six argv / checkargs / dry-run transcripts an arm pair, six chain items, a three-cross plan, reader and driver | three of each for `ce3d_swap`, three chain items (16.0 GiB a fork), X67's one-cross plan, reader and driver | the transcripts of arms that no longer exist are deleted, not kept |

Unchanged: the pin and its tag, the 44 paths, the dose, the stage `CE3D` and its one look, the play pin, the VOID
guards other than the lever's, the hard stop's form, and every "moves no recipe" clause.

## 14. Amendment 2 (2026-11-11, before any launch)

**Blind.** Written the day of the registration and of Amendment 1 (`71c20b521`), before the first swap fork
launches: no `rb_x35_ce3d*` directory in `models/`, no CE-3d family or request in the eval ledger, no cell played.
The addition is the orchestrator's, on Amendment 1's own limit 1 and its "Beside X67" line (an indirect comparison
with no interval).

What changed against `71c20b521`, and why:

| | Amendment 1 | Amendment 2 | why |
|---|---|---|---|
| crosses | one, against C0, which decides | the same one, which still decides, + a REPORTED 3 × 3 against X67's finals `rb_x35_x67ent_s6701`–`s6703` at 1,000 pairs a cell (§5.4) | the direct evidence on "the anchor" against "less uniform push"; Amendment 1 had only the difference of two D read at different pins |
| the reported statistic | — | D = mean(h) − 50 and its 90 % interval, §5.3's estimator at r = c = 3; no class | the same meter as the decision, so the two D are comparable |
| the reading | "evidence, not a test" | two sentences fixed now (§5.4), quoted only with a PASS; rule 8 on the bound | a reading chosen after the number is a story |
| what it can change | — | nothing: no class, label or licence | 2 df a side, and X67 is not the clean control (§5.4's limits) |
| the play pin | the CE-3d pin | the same, for both crosses; one X67 final and one C0 final loaded and played there on the CPU ([`x67_loadcheck_at_pin.txt`](x67_loadcheck_at_pin.txt)) | `f4b25238` refuses the swap forks' configs |
| files | one-cross plan, reader and driver | the same three files take the cross as an argument (`--cols x67`, `shas.json`'s `cols_arm`, the driver's fourth argument); `screen_rule.report_x67` and `simulate-x67`; `power_x67.log`; the self-check extended and re-run at the pin (`power.log` reproduced byte for byte) | it fitted the existing plan and driver; `decide` is untouched |
| cost | ≈ 13 GPU-h, ≈ 23 CPU lane-h | ≈ 13 GPU-h, ≈ 32 CPU lane-h (+ 18,000 games) | the second cross |

Unchanged: the arm, its seeds and argv, the pin, the kit (`src/main/x35/`), the chain items, the deciding cross, its
plan, rule, classes and power, every VOID guard and label, the licences, and every "moves no recipe" clause.

## 15. Amendment 3 (2026-11-11, after fork 1's launch died at step 0; before any training step, game or read)

**Blind.** Written the day of the registration, after ONE launch and before any data. On the orchestrator's GO the
Training session launched fork 1, `rb_x35_ce3dswap_s7321`, at the first CE-3d pin
`333a73ddda7829b9ccdf9e17ef04624eb40edeba` (22:23 local, through its chain: every kit pre-check, the pool-seed check
and §7's banner passed). Twenty minutes later, before step 0, the trainer exited `FATAL_CONFIG` in the startup
update-fit check (`update_fit.check_update_fits` → `dry_update` → `ppo.train` → `PPO._magnet_args`; the crash file
reads `last step: 0`):

```
agents.training.magnet.MagnetError: --magnet-kl-coef > 0 but the rollout buffer has no magnet column: the collector's fill writes it (`agents.training.magnet.fill_column`); this buffer was filled without one
```

What the attempt left is all startup: the launcher's logs, a compile cache, `cuda_ledger.json` (four startup rows),
the fork's seeded opponent pool (14 snapshots) and one crash file. It wrote no `metadata.json`, no checkpoint and
no model. It collected no rollout and took no optimizer step. No CE-3d family or request exists in the eval ledger
and no cell was played. Nothing that a rule, a label or a diagnostic of this file reads exists, so there was
nothing to look at. The startup log is evidence about the build only; §4.4 item 2 quotes the part that bears on it.

**The cause is a gap in the build, not in the design.** The fit check runs one dry update on a fixture rollout that
no collector filled, and only the collector's fill writes the magnet's column. The check runs on CUDA only, and
§4.2's tests and smokes were CPU, so no test had reached it with the lever on. Every `--magnet-kl-coef > 0` launch on
the card would have died there. §4.4 already listed the lever's contact with the card as UNVERIFIED.

**What the rules say (checked).**
- §8's lever-own FATAL clause applies: the error is the lever's own (`MagnetError`, raised by the lever's code in
  `train()`), on the first fork. The screen stopped (the chain HELD; forks 2 and 3 never launched), and the
  orchestrator chose "a new pin and a dated amendment". The clause's list names the T2 slot, the compile gate, the
  canary and the sidecar, not the fit check: this amendment reads the clause by its sense, and says so.
- §3 forbids a substitute seed and a second look. The relaunch is neither: seed 7321 again, the same run name,
  argv, parent and dose, and no number was seen between the two launches.
- §8's VOID guard "not finished" judges the fork that trains, the relaunched `rb_x35_ce3dswap_s7321`. The dead
  attempt is recorded as a **build-gap launch failure**: not a fork of the screen, not a VOID fork, never read.
- `TRAINING_RUN_SOP` §4 is the same rule from the operations side: a defect that is not scientific relaunches
  under the same run name, and the dead directory is moved aside with its cause, never deleted.

What changed against `46c6532ff`, and why:

| | Amendment 2 | Amendment 3 | why |
|---|---|---|---|
| the training pin | `333a73dd` (tag `ce3d-pin-333a73dd`) | `9e97cad5709da21a1f9d4d8846096c87eb17b1d0` (tag `ce3d-pin-9e97cad5`) = `333a73dd` + ONE commit of 3 paths, main's fix `13c50a5e6` of the fit check, byte-identical (§4, §4.1) | the fix must be in the code the forks launch with, and nothing else of main may enter |
| the play pin | "the CE-3d pin" | the same commit, `333a73dd`, now named apart from the training pin (§5.1) | the three paths are not play code, so the load check and the scratch reads made at `333a73dd` stand |
| the kit's pin check | one commit on the side pin, 44 paths | two commits, 44 then 3 paths, each parent and path list exact (`batch.ce3d_pin_check`); the first pin alone is refused for a launch | the arm launches only at the pin that carries the fix |
| the dead attempt | — | named in `batch.STARTUP_DEAD_ATTEMPTS`. `kit pin-check` lists it and does not judge it, only once it is MOVED ASIDE (`models/rb_x35_ce3dswap_s7321.<CAUSE>`) and only while it holds no metadata, checkpoint or model; in place it HOLDs | a directory with no recorded pin fails the side's pin check for every production-side launch; one name in a registry is narrower than tolerating any directory without metadata |
| fork 1 | not launched | relaunched at seed 7321 under its own run name, after the move | the rules above |
| the plan's swap guard | one `pin_history` entry at `333a73dd` | at `9e97cad5` (`cross/plan_cross.py`); the driver's rule text names both pins | the guard names the pin the forks train at |
| the confound table, the UNVERIFIED list | five rows; six items | + the fit check's fix (§4.3); + item 7, and item 2 quotes the dead attempt's startup log (§4.4) | the first relaunch is this code's first contact with the card |
| validation | transcripts at `333a73dd`; 14.33 GiB REQUIRED | re-made at `9e97cad5` for the three seeds (10.97 GiB, §7); `kit_checks_ce3d.txt` and `arm_chain_plan_ce3d.txt` regenerated | a transcript must be the pin's |
| chain items | pin `333a73dd`, a GO against Amendment 1 | pin `9e97cad5`; `go_by` is the orchestrator's to fill again, against this amendment | the old GO named the old pin |
| cost | ≈ 13 GPU-h | + the dead attempt: 20 min of the lease, no training | recorded |

Unchanged: the arm, its seeds (7321–7323), argv, parent and dose; the stage `CE3D` and its one look; the first pin,
its tag and its 44 paths; both crosses, the plans' other guards, the rule, the classes, the power and the reported
reading; every VOID guard and label; the banner; the canary family (none); the licences; and every "moves no
recipe" clause.

**Not done by this amendment** (it does not touch `models/`, the GPU or the chain): the move of the dead directory
and the incident's ledger entry, both the Training session's (`TRAINING_RUN_SOP` §4, steps 3 and 6), and the
relaunch. **UNVERIFIED:** all of §4.4, item 7 first; and `launch_x35.sh` at the training pin, whose pin and run-dir
checks were run only against the scratch view of `models/` described in §7.

## 16. The READ (2026-11-12)

Written after both crosses were read, each once (§3). Nothing in §1 to §15 is changed by it. Every number below is
in [`record/`](record/): copies of the two crosses' result directories (the originals are the eval ledger
`models/_ledger` and `~/gen3ai_archive/ce3d_cross_2026-11-12/`), and the guard and diagnostic commands with their
output. The guards and the diagnostics were run from main's code at `4d327c5a1`, on the CPU, with no GPU lease.

### 16.1 The deciding cross

Family `ce3d_swap_vs_c0_h2h`, request `ce3d_swap_vs_c0`, protocol `gen3_eval_protocol_v1_h2h`, eval ledger regime
`adfdbee824c9eb0a`. Played at the play pin `333a73dd` on the CPU (rule 5b's shape: nice 19, three threads on the
agent cores, under `mem_cap` 24 GB, peak 1.52 GB). Planned 2026-11-12 17:21:45 PST, read 21:58:05 PST, once.
**24 cells × 1,000 mirrored pairs = 48,000 games; every cell complete, 0 aborted; all eight columns
admitted** (`left_out: []`). Games: 25,802 won, 21,993 lost, 205 drawn. h = the swap fork's score in pp:

| swap ↓ / C0 → | s3501 | s3502 | s3503 | s3504 | s3505 | s3506 | s3507 | s3508 | row mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s7321 | 53.250 | 54.325 | 53.375 | 54.450 | 54.075 | 53.525 | 52.400 | 53.350 | 53.594 |
| s7322 | 54.100 | 55.175 | 53.500 | 51.125 | 56.400 | 53.425 | 54.225 | 53.675 | 53.953 |
| s7323 | 51.725 | 55.625 | 54.525 | 55.650 | 54.800 | 54.400 | 55.525 | 52.600 | 54.356 |
| column mean | 53.025 | 55.042 | 53.800 | 53.742 | 55.092 | 53.783 | 54.050 | 53.208 | **53.968** |

All 24 cells are above 50 (lowest 51.125, s7322 against s3504; highest 56.400, s7322 against s3505).

### 16.2 The verdict

**Class: PASS** (`record/deciding/verdict.json`; `near_boundary: []`, the lower end is 3.3 pp from the boundary).

| | |
|---|---|
| D = mean(h) − 50 | **+3.9677 pp** |
| 90 % interval | **[+3.3061, +4.6293]** (SE 0.3515, Welch df 7.318, t 1.8823) |
| row (swap fork) variance | sample 0.1455, below the binomial floor: FLOORED to 0.15625 |
| column (C0 final) variance | sample 0.5717; the floor does not bind |
| rule | PASS iff the lower end > 0 (§5.3) |

### 16.3 Labels (§8; they never change the class)

- **`ENT-LOW`: fires.** End entropy (mean of the last 20 points of −`train/entropy_loss`): 0.4838 (s7321), 0.4962
  (s7322), 0.4921 (s7323). The highest, 0.49621, is below 1.00406, the lowest banked C0 final's. Nothing is within
  1e-9 of a bound.
- **`ENT-HIGH`: no.**
- **`LR-MOVED`: no fork.** `lr_median` (`main.dose`, as printed) is 0.0001005 for all three, C0's own value, inside
  0.0000905 to 0.0001105.

### 16.4 What the PASS licenses (§10, quoted)

> **PASS** ⇒ at β 0.05 and a 2M cadence, a P\* + 20M fork trained with the trailing anchor in place of the entropy
> bonus beats the unchanged recipe's forks head to head. **It licenses REGISTERING a follow-up, and nothing else**:
> on new seeds, with the control that separates the anchor from the missing uniform push — an `--ent-coef 0` arm
> with no magnet at the CE-3d pin, or the magnet against an arm matched on REALISED entropy — and, if the claim is
> to be quoted, a C0-recipe control at the same pin (§4.3).

The sub-lines of §10, each applied:

- **`ENT-LOW` fired**, so the PASS is reported as "**the swap won and ended sharper; not separable from X67's effect
  here**".
- **D is above X67's +1.97 pp** (+3.97 against +1.97; the difference of the two D is +2.00 pp, with no interval
  and no class, §9), so the line "shows no gain over simply lowering `ent_coef`" does NOT fire.
- **§5.4's reading, word for word** (the reported interval [−0.1152, +2.1319] holds 0, so the second row of the
  table applies): "**the swap's gain over C0 is plausibly the lower uniform push alone**". As §10 says, the reading
  neither widens nor narrows the licence; it says which control the follow-up most needs: the `--ent-coef 0` arm
  with no magnet.
- **`LR-MOVED` did not fire**; the "goes to the orchestrator with both numbers" line does not apply.

**This read registers nothing and moves no recipe.** No follow-up is registered by this section; no magnet and no
`ent_coef` enters `production_config.json`, `design_learner_recipe.md` or any BOUNDARY-1 arm.

### 16.5 The limits (§10's six, as they bear on this number)

1. **One loss term, two effects.** The PASS is for "the trailing anchor in place of the uniform one". Both
   registered pieces of evidence say the removed push may be all of it: `ENT-LOW` fired (the swap ended sharper
   than X67's forks, 0.49 against 0.68 nats), and the direct cross against X67's finals holds 0. Neither is a
   test: X67 keeps a quarter of the bonus, and that cross reads a true +1 pp as "lower push alone" three times in
   four.
2. **The control was trained at another pin and on other days.** C0 at `f4b25238`, 2026-10-21 to 11-10; the swap
   at `9e97cad5`, 2026-11-12. The evidence that the pin does not matter is CPU only (§4.3), and no same-pin
   control exists. The +3.97 pp is quoted with that caveat.
3. **The meter is not the card's question.** Nothing here measures exploitability or cycling; no best-responder
   read was made.
4. **A late fork, a short horizon.** Nine refreshes over 20M steps from a 280M-step policy; nothing is claimed
   about a magnet from step 0. The entropy was still falling at the end (16.7), so +20M is not a settled state.
5. **One spelling.** β 0.05, a 2M cadence, no entropy bonus beside it.
6. **Head to head against siblings is not strength against others.** No Kakuna game and no external anchor was
   played.

### 16.6 VOID guards (§8), executed per fork: none fired, three valid forks

Commands and output: [`record/void_guards.sh`](record/void_guards.sh),
[`record/void_guards.txt`](record/void_guards.txt) (2026-11-12 23:43 PST).

| guard | s7321 | s7322 | s7323 |
|---|---|---|---|
| finished: launcher rc / `Last step` (≥ 300,000,170) | 0 / 300,061,272 | 0 / 300,060,944 | 0 / 300,061,282 |
| `final_model.zip`; newest checkpoint (within 98,304) | present; 300,000,383 | present; 300,000,375 | present; 300,000,204 |
| K9(b) at the arm's end (`kit k9b`; ceilings 1e-3 max, 1e-4 p99) | PASS: max 2.20e-4, p99 3.16e-5 | PASS: 2.04e-4, 3.32e-5 | PASS: 1.84e-4, 3.39e-5 |
| the launcher log's pin | only `9e97cad5709d…` | the same | the same |
| a FORK of `rb_x26_s2601` | yes: fork step 280,000,170, parent sha256 `814823c5…` | yes | yes |
| `main.sidecar_audit` | one span at `9e97cad5`, 0 PIN-SPLIT | the same | the same |
| the post-spawn pool-seed check | PASS (at launch, and re-run) | PASS | PASS |
| the banner `[MAGNET] … (ent_coef 0 …)` | present | present | present |
| `ent_coef` in metadata and `cli_args` | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| `magnet_kl_coef` / `magnet_refresh_steps` / the sidecar | 0.05 / 2,000,000 / `final_model.magnet.pt` present | the same | the same |
| `main.dose`: epochs / effective batch | 10 / 65,536 | 10 / 65,536 | 10 / 65,536 |
| `tb_read … --tag magnet/refreshes` | 183 of 204 points above 0; last value 9 | the same | the same |

Arm-end records: `31b0b402c`, `4662ebb53`, `60efef08c`. Restarts: 0 in every fork. The lowest `time/fps` point of any
fork's first hour is 1,357 steps/s (no PAGE). The lever-own FATAL clause was invoked once, before any training step
(§15), and not again.

### 16.7 REPORTED diagnostics (§9; they decide nothing)

Script and output: [`record/diag9_ce3d.py`](record/diag9_ce3d.py), [`record/diag9_ce3d.txt`](record/diag9_ce3d.txt),
[`record/tb_read_ce3d.txt`](record/tb_read_ce3d.txt), [`record/fps_compare.txt`](record/fps_compare.txt).

| diagnostic | swap s7321 / s7322 / s7323 | C0 (eight finals) | X67 (three finals) |
|---|---|---|---|
| end entropy, nats | 0.484 / 0.496 / 0.492 (mean 0.491) | 1.004 to 1.018 (mean 1.010) | 0.659 / 0.683 / 0.687 (mean 0.676) |
| entropy by 5M window, nats | 0.767 → 0.598 → 0.536 → 0.495; 0.759 → 0.589 → 0.529 → 0.501; 0.768 → 0.597 → 0.523 → 0.496 | — | — |
| learning rate: median, cuts | 1.005e-4, one cut / 1.005e-4, two cuts (the second to 8.372e-5 at step 297,996,939) / 1.005e-4, one cut | 1.005e-4, one cut each | 8.372e-5 (all `LR-MOVED`) |
| `train/approx_kl`, median | 0.0186 / 0.0189 / 0.0187 | 0.0179 to 0.0183 | 0.0185 to 0.0188 |
| `train/magnet_kl`, nats | median 0.070, mean 0.066; a saw-tooth | — | — |
| … right after a refresh (the first update's value) | 0.0092 to 0.0106 | — | — |
| … right before a refresh | 0.127 to 0.133 before the first; 0.097 to 0.100 before the second; 0.090 to 0.097 after that | — | — |
| `magnet/refreshes` / `magnet/age_steps` max | 9 / ≈ 2.07M in every fork | — | — |
| `magnet/forward_ms`, median per rollout | 3,506 / 3,505 / 3,517 | — | — |
| `time/fps`, median | 1,372 / 1,398 / 1,384 (mean 1,385) | 1,382 to 1,501 (mean 1,420) | 1,335 / 1,344 / 1,412 (mean 1,364) |
| hours between the first and the last logged update | 4.05 / 4.03 / 4.12 | 3.75 to 4.08 | 4.00 to 4.40 |

- **Entropy.** The swap forks ended at half of C0's entropy and below X67's, and were still falling in the last
  5M steps. With no uniform push and an anchor that follows the policy, nothing in the loss holds entropy up.
- **The magnet's meters behaved as §9 expected.** The KL is a saw-tooth that drops at each refresh and rises to
  ≈ 0.09 to 0.10 nats over the next 2M steps. Its value at the first update after a refresh, ≈ 0.010 nats, already
  holds that update's own ten epochs of drift, so it is an UPPER BOUND on §4.4 item 3's bf16 floor, not the floor.
- **The magnet's cost.** ≈ 3.5 s a rollout, ≈ 4.9 % of a ≈ 71 s iteration (98,304 steps at ≈ 1,385 steps/s),
  against §11's ESTIMATE of ≈ +3 %. The `time/fps` medians are ≈ 2.5 % below C0's and ≈ 1.5 % above X67's; that
  comparison is across pins, days and CPU load and is not a controlled one.
- **Beside X67.** D = +3.97 pp [+3.31, +4.63] next to X67's +1.97 pp [+1.35, +2.59]; the difference of the two D
  is **+2.00 pp**, with no interval and no class (§9).
- **Against X67's finals, directly** (§5.4; family `ce3d_swap_vs_x67_h2h`, request `ce3d_swap_vs_x67`; planned
  21:58:11 PST, read 23:40:05 PST, once; 9 cells, 18,000 games, every cell complete, 0 aborted; 9,142 won, 8,779
  lost, 79 drawn): **D = +1.0083 pp, 90 % interval [−0.1152, +2.1319]** (SE 0.5270, df 4.0; both variances
  floored to 0.41667, from samples 0.1259 and 0.3592). `decides: nothing`. All nine cells are above 50:

  | swap ↓ / X67 → | s6701 | s6702 | s6703 | row mean |
  |---|---:|---:|---:|---:|
  | s7321 | 51.100 | 50.975 | 51.975 | 51.350 |
  | s7322 | 51.325 | 50.125 | 51.650 | 51.033 |
  | s7323 | 50.375 | 50.250 | 51.300 | 50.642 |
  | column mean | 50.933 | 50.450 | 51.642 | **51.008** |

- **The indirect and the direct line.** Indirect: +2.00 pp over X67 (through C0, no interval). Direct: +1.01 pp,
  an interval that holds both 0 and +2.00. They do not contradict each other, and the direct point is half the
  indirect one. Both are quoted and neither is preferred (§9).
- **The pessimistic interval** (the fresh-run floor of 1.76 pp on the rows): **[+1.353, +6.583]** (SE 1.192, df
  3.635). It excludes 0; X67's own spanned 0. **Banked columns** (s3501 to s3505): D = +4.14 pp; **top-up columns**
  (s3506 to s3508): +3.68 pp.
- **Cost, against §11.** Training: 264.8 + 247.7 + 252.7 min = 12.75 GPU-h (ESTIMATE ≈ 13). The crosses: 16,579 s
  and 6,114 s on three threads, 6.3 h in all (ESTIMATE ≈ 10.6 h).
- No Kakuna game was played.

### 16.8 Against the prediction (§1, a GUESS)

D ∈ [0, +2.5] pp: **missed**, the swap won by more (+3.97, the whole interval above +2.5). `ENT-LOW`: **held**. An
end entropy between X67's 0.68 and C0's 1.01 nats: **missed**, the swap ended below X67 (0.49).

### 16.9 FINDINGs

- **CE3D-R1 — a canary "unconfirmed" on s7323.** At update 10 (step 280,990,290) the gradient canary's R1 bf16
  rule tripped on one parameter (`features_extractor.move_resolution_cell.switch_proj.bias`: compiled 1.29e-01
  against eager 1.05e-02) and reproduced on neither of the two re-check slices, so the verdict is `unconfirmed`, not FATAL; the
  run continued, and its checks at updates 100 and 200 passed. s7321 and s7322 passed all three checks. §8 has no
  canary family for this arm and only a FATAL HOLDs, so this is no VOID; it is evidence on §4.4 item 2 (R1 with
  the magnet's argument) and belongs to the compile gate's owner. The run keeps `canary_disagreements.jsonl` and
  `canary_u10_unconfirmed_policy.pt`.
- **CE3D-R2 — K9(b) warn notices.** Each swap fork logged 18 single-check `BEHAVIOUR-POLICY MISMATCH` notices
  under `--behaviour-check warn` (largest 1.84e-4 to 2.20e-4, against a 1e-4 per-check bar). The same count in the
  comparators' logs is 25 and 22 (C0 s3501, s3506) and 29 to 34 (X67's three), so the swap is not worse; the
  arm-end K9(b) guard is PASS for all three.
- **CE3D-R3 — "FORK of" is not a literal line of the launcher log.** §8 words the guard as the run being
  `FORK of …/rb_x26_s2601`. The real launcher log carries no such line; the fork was verified from the GO dry-run
  transcript's role line, `metadata.json`'s lineage (parent sha256 and fork step) and `main.sidecar_audit`.
- **CE3D-R4 — §4.4's UNVERIFIED items, after three forks on the GPU.** Items 1 (cost: ≈ 3.5 s a rollout), 3 (the
  floor: at most ≈ 0.010 nats), 5 (`ent_coef` 0 beside the magnet) and 7 (the fit check: `update_fit.json` written with no OOM, three times)
  now have first numbers from these forks. Item 4 (a resume with the sidecar) was NOT exercised: no fork
  restarted. Item 6 (a β = 0 learner at this pin on the GPU) and §4.3's play-pin row remain UNVERIFIED.
- **CE3D-R5 — heavy-slot holds.** The deciding cross held one heavy slot for 16,579 s and the reported cross for
  6,114 s, each as one chunk; `heavy_slots` logged a LONG HOLD notice (its policy is ≤ 45 min a chunk).
- **CE3D-R6 — the guards and diagnostics ran from main's code**, not the pin's (`main.dose`, `main.ops.tb_read`,
  `main.sidecar_audit`, the kit). They read the runs' own files; no number of the rule depends on them.
- **CE3D-R7 — s7322's late learning-rate cut** (to 8.372e-5 at step 297,996,939, the last 2M steps). Its median is
  unmoved, so the label does not fire; it is reported because X67's forks sat at that rate for most of their run.
