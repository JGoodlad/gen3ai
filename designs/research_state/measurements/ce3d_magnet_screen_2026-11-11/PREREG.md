# CE-3d — the trailing-magnet screen: PRE-REGISTRATION (2026-11-11)

**Status: REGISTERED 2026-11-11 (`08b998d12`) and AMENDED the same day (Amendment 1, §13: ONE arm, the SWAP, read
against the banked C0 finals with X67's cross and rule; the two arms first registered are dropped). Written blind,
before any launch, any game and any read. NOT LAUNCHED** (no `rb_x35_ce3d*` run and no CE-3d ledger family exists).
The launch is the orchestrator's GO through the Training session's kit (arm `ce3d_swap`); the hand-off is
[`chain_items_ce3d.json`](chain_items_ce3d.json) (§7). Nothing below changes after the first fork launches except by
a dated amendment, written before the cross is played, that says why.

**The question.** Does a plateau policy get stronger when the entropy bonus (a pull toward the uniform policy) is
replaced, at the same weight, by a pull toward a trailing copy of the policy itself (a "magnet", refreshed every 2M
steps)?

**The tier is FULL** (`../../README.md` § Registration tiers). Its cost would fit a light screen (≈ 13 GPU-h, ≈ 23
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
label and §9's comparison with X67 are the registered evidence on which it is.

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
| pin | **`333a73ddda7829b9ccdf9e17ef04624eb40edeba`** (tag `ce3d-pin-333a73dd`, §4) | `f4b25238def57900c0305bb2a3a5a8f9edaef285` |
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
  seed. `kit stage-check` is PENDING until this file exists on the kit's checkout.
- **Order.** 7321, 7322, 7323, one at a time.

## 4. The pin, and the confound it brings

**The swap runs at `333a73ddda7829b9ccdf9e17ef04624eb40edeba`**: the production side pin `f4b25238` + ONE commit,
annotated tag **`ce3d-pin-333a73dd`** on origin (`batch.CE3D_PIN`; `kit pin-check` requires the parent to be the
side's written pin and the commit to change EXACTLY the 44 paths of `batch.CE3D_PIN_PATHS`, executed in §7). The pin
is the first registration's, unchanged.

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
plain side pin REFUSES to load a config that carries them, so **the cross is played at the CE-3d pin**, which loads
C0's finals as `f4b25238` does; and main loads a CE-3d config keeping the recorded values.

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

### 4.3 The confound, stated: the swap forks and the C0 finals differ by more than the lever

The decision reads forks trained at `333a73dd` against finals trained at `f4b25238`. **This is a REPORTED caveat,
not a guard: nothing in the rule corrects for it, and the verdict is quoted with it.**

| difference | what it is | evidence that it is not the result | status |
|---|---|---|---|
| the magnet code | the lever itself, live in the swap arm | it IS the treatment | — |
| the near-tie judge | parity judging in the T2 service (training and play), reached only after a parity judgement has failed; two `dtype=` arguments in the damage op's forward | the judge decides only whether a failed check is fatal and serves no logit; the two tensors were float32 with and without the arguments (§4.2's probe) | CPU only |
| everything else in the learner | nothing: 44 paths, enumerated and checked | **with β = 0 the learner golden is bit-identical in bf16 and fp32 at the pin** | **CPU only; never run on the card** |
| the day | C0 trained 2026-10-21 to 11-10; the swap after 11-11 | none (same card, same interpreter pin, one `torch_version` and `init_num_threads` enforced by the plan) | X67 carried the same difference |
| the play pin | this cross is played at `333a73dd`; X67's was played at `f4b25238` | the row above; no play or inference path builds a magnet (the pin's h2h smoke printed none) | **UNVERIFIED**: no game was played at both pins |

**Why this was judged acceptable for a one-arm read** (and not the fallback, a second arm of `--ent-coef 0.05` with
no magnet at the CE-3d pin, ≈ 26 GPU-h in all): the only code the pin adds to a β = 0 learner changes no number
it computes, and the β = 0 learner reproduces `f4b25238`'s golden to the bit, so a control at the CE-3d pin would be
C0's program again; the day difference is the one X67's PASS already carries. What a same-pin control WOULD buy is
the GPU confirmation of the CPU evidence. It is not bought here, and §10 limits the licence accordingly.

### 4.4 UNVERIFIED on the GPU, every one of them (no lease was held)

1. the magnet's cost: one more full-bucket T2 slot of CUDA-graph captures, one more weight copy on the card, one
   inference pass over the buffer per rollout (ESTIMATED ≈ +3 %). Read `magnet/forward_ms` on the first fork;
2. R1's compiled program with the magnet's tenth argument has never been compiled on the card, and the compile gate
   and the canary have never run with it;
3. the size of the KL's floor: T2 serves fp32, R1's trunk is bf16, so the term is a small positive number, not 0,
   right after a refresh;
4. a resume with the sidecar on the card (the CPU integration test passes);
5. `ent_coef` 0 beside the magnet on the card (the CPU smoke only);
6. that a β = 0 learner at this pin trains on the card as `f4b25238` does (§4.3: CPU golden only).

## 5. The meter and the rule: X67's, unchanged

### 5.1 The cross

One cross, head to head, both sides greedy, the players' eval team builder, schedule seed 0, `--oracle-reveal-mode
off`, protocol `gen3_eval_protocol_v1_h2h`, purpose `ab`, label `ce3d_magnet_screen`, played from a clean detached
checkout **at the CE-3d pin** ([`cross/drive_cross.sh`](cross/drive_cross.sh)).

| rows × columns | pairs a cell | games | family / request | role |
|---|---:|---:|---|---|
| 3 swap forks × c C0 finals (5 ≤ c ≤ 8; 8 expected) | 1,000 mirrored | ≤ 48,000 | `ce3d_swap_vs_c0_h2h` / `ce3d_swap_vs_c0` | **DECIDES** |

This is X67's cross with the swap forks as its rows: the same columns, cell size, statistic and classes, so its D
sits directly beside X67's +1.97 pp (§9).

### 5.2 The plan

[`cross/plan_cross.py`](cross/plan_cross.py) is run ONCE, after the third fork's arm-end checks. The three swap
forks are REQUIRED; the five banked C0 finals (s3501–s3505) are REQUIRED and s3506–s3508 join when they pass (a
top-up final that exists and fails a guard REFUSES the plan). Guards, every run: `final_model.zip`; one
`pin_history` entry at ITS pin from 280,000,170 to ≥ 300,000,170; lineage FORK of P\*'s sha256; the seed; `n_epochs`
10; `ent_coef` (swap 0.0, C0 0.05) in `metadata.ent_coef` and `cli_args.ent_coef`; `model_config.json` (swap:
`magnet_kl_coef` 0.05 with `magnet_refresh_steps` 2,000,000; C0: both keys absent); `final_model.magnet.pt` present
for a swap fork and absent for C0; K9(b) PASS; one `torch_version` and one `init_num_threads` across all. Executed
2026-11-11: `--columns-only` → `columns OK`, all eight C0 finals; the full plan REFUSES (no swap final exists).

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

For the Training session, from the main checkout at or after this amendment's commit, holding the lease, one run at
a time in the order of [`chain_items_ce3d.json`](chain_items_ce3d.json):

```bash
STAGE=designs/research_state/measurements/x35_breakers_stage_2026-10-17
PIN=333a73ddda7829b9ccdf9e17ef04624eb40edeba       # tag ce3d-pin-333a73dd; NOT the side pin
python -m main.x35.kit stage-check --stage $STAGE --arm ce3d_swap --seed $s    # ALLOWED: "CE-3d PREREG present"
bash $STAGE/launch_x35.sh ce3d_swap $s --pin $PIN --lease-owner "<owner>" --thinning       # the kit's DRY RUN (default)
bash $STAGE/launch_x35.sh ce3d_swap $s --pin $PIN --lease-owner "<owner>" --thinning GO    # only on the orchestrator's GO
```

for `$s` = 7321, 7322, 7323. The kit resolves each to `python -m main.launcher --restart-interval-hours 24
--pin-commit $PIN <argv>`; the first seed:

```
--model /home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip --behaviour-check warn --steps 300000170 --checkpoint-every-steps 1000000 --device cuda --ent-coef 0 --magnet-kl-coef 0.05 --seed 7321 --run-name rb_x35_ce3dswap_s7321
```

Validated on 2026-11-11, CPU only, no lease, `CUDA_VISIBLE_DEVICES=""`, nice 19 on the agent cores, nothing created:

- `python -m main.checkargs --argv "<argv>" --pin $PIN` → `checkargs_ce3d_swap_s<seed>.txt`, all three: **exit 0**,
  every flag accepted by the PINNED parser, every ARCH-surface key the production mirror's, the two lever rows
  TYPED (§2), `✓ fork arch`.
- `python -m main.launcher --dry-run --restart-interval-hours 24 --pin-commit $PIN <argv>` →
  `dryrun_ce3d_swap_s<seed>.txt`, all three: **role FORK of `rb_x26_s2601`**, +20,000,000 steps, the pin (source
  `pin_commit`), both lever rows TYPED, 14.33 GiB REQUIRED. **Exit 3, "would NOT launch", on exactly two refusals,
  both this unit's own shell and both expected:** the CUDA-visibility preflight (`CUDA_VISIBLE_DEVICES=''`: no
  lease) and the inherited-nice preflight (the shell ran at nice 19 by rule 5b; a GPU launch is refused above nice
  10). The kit's own dry run re-runs it with the card visible, at the launch shell's priority, before every GO. The
  transcripts' `showdown deps` line names this amendment's worktree, which no longer exists.
- `python -m main.x35.kit pin-check --pin $PIN --side production --arm ce3d_swap --stage $STAGE` → exit 0 ("the
  CE-3d pin 333a73ddda78 = the side pin f4b25238def5 + one commit, exactly the 44 enumerated path(s)"); with the
  plain side pin it exits 1 ("the CE-3d arm launches only at CE3D_PIN"). `kit required-check` passes at the pin; a
  fourth seed and a retired seed are refused. [`kit_checks_ce3d.txt`](kit_checks_ce3d.txt).
- `ce3d_test`: the kit's argv equals the three validated files token for token and is C0's plus the four lever
  tokens; `check_validation` PASSes the six transcripts (the two shell refusals modelled as absent) and FAILs them
  when either typed lever is absent, another value, or shown by one tool only, and when read as C0's or X67's.
- `python -m main.ops.arm_chain plan <scratch chain> --check-pins --describe` on the three draft items →
  [`arm_chain_plan_ce3d.txt`](arm_chain_plan_ce3d.txt): PLAN OK, the kit agrees on all three; a fourth seed and a
  dropped arm REFUSE. **The plan does not judge an arm's own pin** (an item at the plain side pin still plans OK):
  that check is the launch script's `kit pin-check`, above.

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

**Disk.** `disk_need_gib` is **16.0** a fork: the dry run's 14.33 plus the magnet's sidecars, which the launcher's
arithmetic does not count (one `<checkpoint>.magnet.pt` a checkpoint, each at most a checkpoint zip's 54.7 MiB; the
production size is unmeasured). The thinning planner does not recognise the sidecars and keeps them, so up to
≈ 1.1 GiB a fork stays behind after thinning.

**Not executed at the amendment:** `launch_x35.sh` itself (a run is live, so its pre-check 1 HOLDs, and its checks
read the lease) and the cross's play command (it writes eval-ledger rows). The plan's column check, the rule and
its self-check are executed; the driver's `family-register` line and `read_cross.py` are executed with the pin's
code against an empty SCRATCH ledger (the read: INCONCLUSIVE, 24 missing cells), never on a played cell.

## 8. VOID guards, labels, the canary family

**VOID** (any one ⇒ that fork is VOID; fewer than three valid forks ⇒ INCONCLUSIVE):
- not finished: the arm chain's completeness check (rc 0, `Last step` ≥ 300,000,170, `final_model.zip`, the newest
  checkpoint within 98,304 steps of it);
- K9(b) not PASS at the arm's end;
- the launcher log names another pin than the CE-3d pin, or the run is not `FORK of …/rb_x26_s2601`;
- the post-spawn pool-seed check fails;
- **either half of the lever is not in the record**: §7's banner (the chain STOPS the run); any of §5.2's plan
  guards on `ent_coef`, `magnet_kl_coef`, `magnet_refresh_steps` or the sidecar; `main.dose` reads an epoch count
  other than 10 or an effective batch other than 65,536;
- **a magnet that never refreshed**: `python -m main.ops.tb_read <run> --tag magnet/refreshes` shows no point above
  0 (nine refreshes are due; the tag is emitted only with the magnet on).

**A FATAL that is the lever's own on the first fork** (the magnet's T2 slot, the compile gate or the canary with
the magnet's argument, a sidecar refusal) **STOPS the screen**: no further fork launches, nothing is read, and a fix
is a new pin and a dated amendment or a new registration. The orchestrator decides which.

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
- **Beside X67**: D and its interval next to X67's +1.97 pp [+1.35, +2.59], and the difference of the two D. No
  interval and no class is registered for that difference: the two crosses share their columns but were played at
  different pins and on different days (§4.3).
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
   toward uniform. No class separates them; `ENT-LOW` and the comparison with X67 are evidence, not a test.
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
| the cross, ≤ 48,000 games | ≈ 0.35 in a GPU window | or ≈ 7.7 h on the three agent cores (≈ 23 lane-h) | from X67's cross: 48,000 games in 27,677 s on three threads; the GPU figure is X67's ESTIMATE |
| the pin checkout's Rust build | 0 | minutes, under `mem_cap` | once |
| §9's reads, the dose read | 0 | seconds | |

≈ 13 GPU-h in all. **Hard stop:** three forks, one cross, one read; nothing here extends.

## 12. What this file does NOT do

It registers no magnet beside an entropy bonus, no `--ent-coef 0` arm without a magnet, no same-pin control, no
second β or cadence, no fresh-run magnet arm, no cross against X67's finals, no exploitability read and no Kakuna
game. It does not adopt or delay X67's PASS, amends no BOUNDARY-1 document, and changes no baseline and no
production recipe. CE-3d has no `main.reads` rule or `reads/` record: its verdict is `cross/read_cross.py`'s
`verdict.json`, written by `screen_rule.decide`.

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
