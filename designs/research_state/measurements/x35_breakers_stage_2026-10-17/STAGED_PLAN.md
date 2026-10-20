# X35, plateau-breakers: STAGED ON PAPER (2026-10-17)

**Status: STAGED, not launched; every launch NEEDS THE ORCHESTRATOR'S GO. BOTH PLATEAUS ARE CALLED (2026-10-19): P\* =
X26 @ 280,000,170 and T\* = X34 @ 290,000,087 (sha256 `cfa7f7be…a22965e7`); both reference populations are FROZEN and all
32 batch-1 argvs are validated on the real values at `03b983be` (see the paragraphs below and
[`BATCH1_LAUNCH_SHEET.md`](BATCH1_LAUNCH_SHEET.md) §4.3).** The spec is
[`design_compute_efficiency.md`](../../../endstate/design_compute_efficiency.md) §5.6. Its backlog rows are
`EXPERIMENT_BACKLOG.md` CE-3 to CE-3g, and its builds are `TASK_BACKLOG.md` T54–T59. X35 starts when X26 (`rb_x26_s2601`)
and X34 (`rb_toy_ent_s4002_deep`) have both reached their Tier-1 plateau (they have) and the GPU is free. **The stage kit (T55)
is BUILT (2026-10-17, §6):** the launch script, the R freeze, the readout and the Kakuna runner.

**D1 (X38, the pool-widening arm) is staged too (2026-10-19, §7):** its argv is `main.x35.kit` arm `d1`, dry-run
validated on the REAL P\* at pin `c1b0c1f0`; it NEEDS GO like every arm.

**AMENDED 2026-10-19, before any launch:** the orchestrator ruled every amendment of the adversarial review
[`X35_BATCH1_REVIEW.md`](X35_BATCH1_REVIEW.md) (AM-1 one pin, AM-2 the variance gate G1, AM-3 the conditional toy
calibration, AM-4 X47's look 2 and readout, AM-5 D1's widened H, AM-6 K9(b) at arm end, AM-7 the staged order, the POOL
record, the G-E readers, C1 (a)); each is built in `src/main/x35/` (`batch.py`, `x47.py`, `d1.py`) and enforced by
`launch_x35.sh` (pre-checks 2b, 9, 10, 11). The launch sheet's §0 is the table; its §1 the new order (§4 below now points there
and holds the BATCH-2 CANDIDATES).

**AMENDED AGAIN 2026-10-20 (AM-8 / AM-9), before any launch, after the GPU battery's read** (the T62 gate is DECIDED as
FP32_ALT, which does not exist yet): the orchestrator ruled ONE PIN PER SIDE (AM-1 amended: `x35_pin_batch1_production.txt`
and `x35_pin_batch1_toy.txt`), the TOY block FIRST (AM-9: T1, then T1b on X47's look-1 CONTINUE, then production when FP32_ALT
has landed, the toy calibration T2 last), and the side's own `--t62` token (production `FP32_ALT`, toy `NA_TOY`). Production
is HELD by a placeholder commit in `x35_required_commits_production.txt` (pre-check 2c). The launch sheet's §0 (AM-8) and §3
are the record. **§4 holds the BATCH-2 CANDIDATES (2026-10-20), first the LR FLOOR (LR-F), from the X26 treadmill read.**

**THE LAUNCH SHEET for batch 1 + X47 (2026-10-19): [`BATCH1_LAUNCH_SHEET.md`](BATCH1_LAUNCH_SHEET.md).** The 32 forks in AM-7's GPU order (14 unconditional, 18 on a stage condition; sheet §1) with GPU-h, stop rule and matched control; every argv `checkargs` exit 0, "would launch" and a PASS verdict at `03b983be` (all 32 forks, the toy on the REAL T\*, [`batch1_validation_03b983be/`](batch1_validation_03b983be/); the older validations, on the 200M stand-in for T\*, are history: `d825c6a1` (32 forks), `9c3bf3c4` (24) and `803a13e3` (24, whose toy lines typed K9(b) `fatal`), kept because `kit_test.py` reads the last one's argv files); the T62 / FP32_ALT pin variants (no FP32_ALT commit exists); the pre-launch checks and the failure handling. **X47 is a kit arm** (`toy_x47`, F-LS-1 closed). **Ruled by the orchestrator (2026-10-19): C2** (X47 interleaved with its toy C0, toy A2 last), **C3** (the toy arms run K9(b) `warn` under X34's ceilings) and **C4** (batch 1 launches without R0); C1 and the AM-1..AM-7 rulings have since been made (sheet §0); C5 is ruled and done (the production R re-frozen with T\* as a second X member); C6 is still open. [`validate_batch1.sh`](validate_batch1.sh) rebuilds and re-validates the lines at any pin, with a deterministic verdict per line.

**R (production) is FROZEN (2026-10-19)** at P\* = X26 @ 280,000,170 (Amendment 5's `TIER1_PLATEAU` call, sha256
`814823c5…91ed`): [`reference_population_production.json`](reference_population_production.json), content sha256
`cef0960e…e4519` (**RE-FROZEN the same night under ruling C5**, see below; the first freeze, `39bae18f…5bec0`, is kept as
[`reference_population_production_PRE_C5.json`](reference_population_production_PRE_C5.json), history), `refpop verify` OK.
Seven members: A-block 280 / 270 / 80 / 140 / 210M, and X-block `production` (band score 0.6935 at 500 pairs, KEPT) and
`tstar` (P\* vs T\*, P\* the player: **0.5905** [0.5605, 0.6205] at 500 pairs, KEPT). Write-up:
[`../x35_refpop_freeze_2026-10-19/README.md`](../x35_refpop_freeze_2026-10-19/README.md). **R (toy) is FROZEN
(2026-10-19, after X34 plateaued)** at T\* = X34 @ 290,000,087 (sha256 `cfa7f7be…a22965e7`):
[`reference_population_toy.json`](reference_population_toy.json), content sha256 `856c04cb…55a4d`, `refpop verify` OK. Six
members: A-block 290 / 280 / 80 / 150 / 220M, and X-block `pstar_x26` (T\* vs P\*: **0.4160** [0.3863, 0.4457] at 500
pairs, KEPT); A1's archive 30 / 70 / 100 / 130 / 170 / 200 / 240 / 270M. Write-up:
[`../x35_toy_refpop_freeze_2026-10-19/README.md`](../x35_toy_refpop_freeze_2026-10-19/README.md). **The production R now has
TWO X members** (C5 ruled by the orchestrator, executed 2026-10-19: a MEASURED P\*-player cell, not an assumed symmetry;
sheet §5, production README §6). Nothing was launched.

**How this was staged.** Everything ran at `9ebda190` with `CUDA_VISIBLE_DEVICES=""`, under `nice -n 19 taskset -c
5-7,13-15`, with no GPU lease. It used only `main.checkargs`, `main.launcher --dry-run` and one offline call to
`resolve_stable_opponents`. Nothing was created under `models/`, and nothing was written into X26's directory.

**THE G-E READERS are kit arms (2026-10-20, F27 closed for the readers):** `ge_a1` / `ge_a2` / `ge_d1` / `ge_c0`
(`src/main/x35/ge.py`), launched by `launch_x35.sh` only when the arm's primary reads GAIN; the launch sheet's §4.4 is the
record (argv, trigger, read, validation) and its §5 C7 the open choices.

## 1. The stand-ins, and what replaces them at the plateau call

| stand-in here | at the call |
|---|---|
| P\* = `models/rb_x26_s2601/checkpoints/checkpoint_180000055_steps.zip` (the newest 10M node at staging) | the newest node of X26's second FLAT Tier-1 check, sha256 frozen |
| pin `9ebda1904da7a7782f22de2495b6e4c51368510b` | `X35_PIN_1`: a main commit carrying T54 (and T56 / T57 if they landed) |
| A1's archive = 20, 40, …, 160M (`a1_archive_spec_standin.txt`) | N_k = the 10M-multiple checkpoint nearest k·(P\* − 20M)/8, k = 1..8, ties down |
| `--steps 200000055` | P\* + 20,000,000 (production), T\* + 70,000,000 (toy) |
| toy C0 from `rb_toy_ent_s4002/final_model.zip` (100,081,428) | T\* in `rb_toy_ent_s4002_deep` (**resolved 2026-10-19: the 290M node, step 290,000,087**; the 200M node used as a later stand-in is retired) |

## 2. The validated argvs (launcher flags: `--restart-interval-hours 24 --pin-commit <PIN>`)

| arm | argv file | `checkargs` | `launcher --dry-run` |
|---|---|---|---|
| **A1 s3501 (the FIRST arm)** | [`argv_a1_s3501_standin.txt`](argv_a1_s3501_standin.txt) | exit 0 ([`checkargs_a1_9ebda190.txt`](checkargs_a1_9ebda190.txt)) | "would launch", FORK of `rb_x26_s2601`, +20,000,000 ([`dryrun_a1_9ebda190.txt`](dryrun_a1_9ebda190.txt)) |
| C0 s3501 | [`argv_c0_s3501_standin.txt`](argv_c0_s3501_standin.txt) | exit 0 | "would launch", +20,000,000 |
| A2 s3501 | [`argv_a2_s3501_standin.txt`](argv_a2_s3501_standin.txt) | — | "would launch", +20,000,000 |
| A3 phase 1 (main exploiter) | [`argv_a3_main_exploiter_standin.txt`](argv_a3_main_exploiter_standin.txt) | — | "would launch", +5,000,000. **Without `--no-self-play` it is REFUSED** (F16) |
| A6 phase 1 (league exploiter) | [`argv_a6_league_exploiter_standin.txt`](argv_a6_league_exploiter_standin.txt) | — | "would launch", +5,000,000 |
| toy C0 | [`argv_toy_c0_standin.txt`](argv_toy_c0_standin.txt) | — | "would launch", FORK of `rb_toy_ent_s4002`, +70,000,000 |

**A1's archive resolves.** `resolve_stable_opponents` was called against X26's `ModelVersion`, offline
([`a1_archive_resolution_standin.txt`](a1_archive_resolution_standin.txt)). It returned 8 entries, rung
`explicit_zip`, with 8 unique labels (`ext_x26_020m` … `ext_x26_160m`). The arch gate passed. The dry-run cannot show
this, because the stable-opponent resolver runs in the child.

**What the dry-runs show for every production fork:**
- the 12 recipe knobs RESTORED from X26's record and passed typed (`n_envs=256 … n_epochs=10 … ent_coef=0.05
  self_play=True`);
- `--lr` / `--batch-size` / `--n-steps` INERT;
- the ARCH SURFACE matching the production mirror;
- a bf16 trunk;
- 20.68 GiB of disk REQUIRED against ≈ 318 GiB free;
- the Showdown deps present;
- no display process holding the GPU.

The first arm's block, with its placeholders, is in the design doc's §5.6.8. The literal stand-in line is:

```
# NEEDS GO. Stand-in values. Never run as written.
python -m main.launcher --restart-interval-hours 24 --pin-commit 9ebda1904da7a7782f22de2495b6e4c51368510b \
  $(cat designs/research_state/measurements/x35_breakers_stage_2026-10-17/argv_a1_s3501_standin.txt)
```

## 3. Pre-checks at the call (a failure is a HOLD; [`launch_x35.sh`](launch_x35.sh) runs them, §6)

1. X26 and X34 have stopped, and the GPU lease is the Training session's to grant.
2. `X35_PIN_1` contains T54, the pool seed's step filter (F18).
3. No `models/rb_x35_*` directory for the arm exists.
4. The sha256 of P\* and of every archive node is unchanged since the call.
5. `checkargs --pin X35_PIN_1` exits 0.
6. The dry-run says "would launch", FORK of `rb_x26_s2601`, +20,000,000.
7. The reference population R is frozen and its file is written before the first arm launches (§5.6.4). **Production: DONE 2026-10-19** (`reference_population_production.json`); **toy: DONE 2026-10-19** (`reference_population_toy.json`, on the real T\*).
8. The disk has room for the whole batch, or finished arms are being thinned (F22).

## 4. Batch 1's queue is the launch sheet's §1; this section holds the BATCH-2 CANDIDATES

**Batch 1's order, GPU-h and stop rule are [`BATCH1_LAUNCH_SHEET.md`](BATCH1_LAUNCH_SHEET.md) §1.** The kit's `BATCH1`
is the order of record. The 24-fork "production first" queue that stood here is superseded and is removed; git history
keeps it. R0 is not in batch 1 (ruling C4, 2026-10-19: its argv is unregistered, F27).

**The batch-2 candidates.** Each has ONE lever per A/B and is read by R1 against a C0 at its own side's pin. **AM-RP-f
(the launch sheet's §0, 2026-10-20) REGISTERED two of them, before X47's look 1 and before any production arm: the
SPRT arm (§4.2, both sides) and toy LR-F (§4.1).** Production LR-F stays a designed candidate. None is a kit arm yet
(§4.3: F-AS-8), and every launch NEEDS the orchestrator's GO. The GPU order is AM-RP-e's.

| candidate | in plain words | the lever | GPU-h, look 1 | designed in |
|---|---|---|---|---|
| **LR-F, the LR FLOOR** (§4.1; **toy REGISTERED**, AM-RP-f) | continue the plateau checkpoint with the learning-rate controller forbidden to go below the rate the run used while it was still climbing | `--min-lr` (two controller rungs above the start's rate) | toy 12.9; production 12.9 | here |
| LG-1, main exploiters (A3) + R0 | train 3 exploiters to beat P\*, then fold them into P\*'s opponents | the exploiter slice | ≈ 19 | [`design_alphastar_league_path.md`](../../../endstate/design_alphastar_league_path.md) §6 |
| LG-2, PFSP over the archive (**only on an A1 GAIN**, AM-RP-b) | A1 with the archive nodes the trainee loses to drawn more often | `--stable-opponent-pfsp` | ≈ 14.4 | the same |
| A4, the trailing magnet | a pull toward a recent copy of itself (R-NaD / MMD) | `β · KL(π ‖ π_mag)` (build T56) | ≈ 15 | [`design_compute_efficiency.md`](../../../endstate/design_compute_efficiency.md) §5.6.2 |
| **SPRT, the promotion rule** (§4.2; **REGISTERED**, AM-RP-f) | promote a snapshot by its own sequential test instead of the first noisy crossing of 0.55, so the self-play pool refreshes | `--promotion-sprt` (T6, built) | toy 12.9; production 12.9 | [`REPLAN_2026-10-20.md`](REPLAN_2026-10-20.md) §2 |

### 4.1 LR-F, the LR FLOOR (designed 2026-10-20, from the X26 treadmill read; the TOY arm REGISTERED by AM-RP-f, §4.1.1; production NEEDS a registration and GO)

**In plain words.** Continue the plateau checkpoint, but forbid the learning-rate controller to go below the rate it
used while the run was still climbing. Everything else is C0's.

**Why: the evidence** (MEASURED, read-only, [`batch2_lr_floor/lr_pool_series.md`](batch2_lr_floor/lr_pool_series.md)
by `batch2_lr_floor/lr_pool_series.py`; trailing-10M means).

| | X26 (production) | X34 (the toy) |
|---|---|---|
| the controller's rungs (3e-4 / 1.2^k) | 1.7361e-4 from 138.9M to 218.3M (Tier 1 GAIN throughout); 1.4468e-4 to 247.3M; **1.2056e-4 to 282.0M (P\*)**; 1.0047e-4 to 342.7M; 8.3724e-5 after | 3.0e-4 at 75–105M; 2.0833e-4 from 134.7M; 1.7361e-4 from 168.0M; **1.4468e-4 from 211.4M through T\* (290M)** |
| approx KL per update | 0.017–0.019 throughout | 0.017–0.019 throughout |
| gradient norm | 0.62 (150M) → 1.04 (280M) → 1.28 (380M) | 0.53 (90M) → 1.01 (290M) |
| KL per unit LR² | ×2.2 from 180M to 280M | ×1.5 from 190M to 290M (×4.4 from 90M) |
| promotions per 50M | 18 (90M) → 6 (280M) → 3 (380M) | 20 (90M) → 3 (240M) → 3 (290M) |
| pool median age; oldest member | 33M → 83M (280M) → 135M (380M); 164M at 380M | 29M → 103M (240M) → 128M (290M); 92M at T\* |

- **The toy shows the same pattern:** the rate halved, the gradient norm doubled at a constant KL, and the pool went
  stale.
- **But on the toy the step size did not coincide with the plateau.** The rate held at 1.4468e-4 for 80M, from 211.4M
  through T\*. Tier 1 read GAIN at 210–270M (UNDECIDED at 250M) and FLAT only at 280M and 290M. The pool was already
  at 3–6 promotions per 50M from 240M on.
- On X26 the cuts continued into the plateau (247.3M, 282.0M).
- So the step size COVARIES with the plateau on both lineages, but it COINCIDES with the plateau only on X26. This arm
  is the intervention that tells the two apart. **It is also the direct test of the LR confound every X35 production
  fork inherits** (sheet §0 TM-4).

**The ONE lever: `--min-lr`, two controller rungs (×1.44) above the start's saved rate.**
- **Production:** `--min-lr 1.7361e-4`. P\*'s rate is 1.2056e-4, and 1.7361e-4 is the rate X26 climbed at from 139M
  to 218M.
- **Toy:** `--min-lr 2.0833e-4`. T\*'s rate is 1.4468e-4, and 2.0833e-4 was X34's rate from 134.7M to 168.0M.
- The SAME ×1.44 on both sides, so the toy read calibrates the production arm.
- **How it acts, read in the code:**
  - On a fork, the resume path clamps the saved rate into `[min_lr, max_lr]` (`main/train/model_build.py`: "Clamping
    resume LR …"), so the arm starts AT the floor.
  - The controller's down-step is `max(lr / 1.2, min_lr)`, so it never goes below the floor.
  - Its up-step (KL EMA < 0.005) stays live, but it is not expected to fire.
  - The floor is in the argv, so every launcher restart re-applies it.
  - `max_lr` stays 6e-4, and the inherited `--lr` 3e-4 is ≥ the floor, so the `[min_lr, max_lr]` check passes.
- **Validation:** `main.checkargs` exits 0 on both argvs, with `min_lr … (TYPED)`:
  [`checkargs_lrfloor_s3501.txt`](batch2_lr_floor/checkargs_lrfloor_s3501.txt) on P\*'s pinned parser @`9c033489`,
  and [`checkargs_toy_lrfloor_s3601.txt`](batch2_lr_floor/checkargs_toy_lrfloor_s3601.txt) on T\*'s @`ae2d0901`.
  The launcher dry-run was NOT run (FINDING below).

**Rejected.**
- **`--fork-lr 1.7361e-4` alone.** It pins the rate once. With the KL then above 0.02, the controller walks it back
  two rungs within ≈ 2–4M of a 20M arm. That is a pulse, not a lever.
- **`--fork-lr … --fork-lr-freeze`.** It gives the same level, but it also switches off the controller's raise half: a
  second difference.
- **Back to the declared 3e-4.** The rate would rise 2.5× and the KL ≈ 2.5–6×, beyond anything either lineage climbed at.
- **A warm restart (a cosine re-warm, SGDR).** That is two levers: the level and the shape.
- **A floor at P\*'s own rate.** It only stops C0's next ≈ 20 % cut, which is too small to read at R1's ±2 pp.

**What it tests: a two-sided prior.** The literature is cited from memory, **UNVERIFIED**.
- **Toward a GAIN.**
  1. Self-play's target moves. Tracking a non-stationary target needs a step that does not vanish (Sutton & Barto
     2018 §2.5).
  2. Plasticity. Rising curvature goes with plasticity loss (Lyle et al. 2023). The remedies include resets and
     re-warmed rates:
     - the primacy bias (Nikishin et al. 2022);
     - continual learning's loss of plasticity (Dohare et al. 2024);
     - LR re-warming in continual LLM pre-training (Gupta et al. 2023; Ibrahim et al. 2024);
     - warm restarts (Loshchilov & Hutter 2017).
     X26's KL per unit LR² rose ×2.2 (the toy's ×1.5): the curvature-rise signature. That reading is an
     interpretation.
- **Toward NULL or LOSS.**
  1. The controller held the per-update POLICY step constant (KL 0.017–0.019). The floor raises it about 1.5–2×,
     beyond what either lineage climbed at. PPO's adaptive-KL variant and its LR form exist to hold that step
     (Schulman et al. 2017 §4; Rudin et al. 2021). The LR is the most sensitive knob (Andrychowicz et al. 2021).
  2. In stationary supervised learning, cutting the step at a plateau LOWERS the SGD noise floor (step decay), so a
     higher rate raises it.
  3. No plasticity loss of the dormant-unit kind shows: battery B10 is flat and rank is rising.
  4. The toy climbed for 60M at its halved rate.
  5. Pre-era fold and exploiter evidence: more dose did not buy more (L09015), and at 1.78× it hurt (L21165).

**The predictions, registered with the candidate** (a ruling may revise them before any LR-F launch, never after; FROZEN for the toy by AM-RP-f, §4.1.1).
- **Mechanism.**
  - The LR sits at the floor on ≥ 95 % of the arm's updates, every seed.
  - The arm-mean approx KL lies in [0.025, 0.045]: at ×1.44, KL ∝ LR gives 0.026–0.028 and KL ∝ LR² gives
    0.038–0.040.
  - The clip fraction is 0.20–0.30 (from 0.17–0.18).
  - Production only: greedy churn per 20M on `bank_x26` is 0.33–0.42 (from ≈ 0.30).
  - N_prom (sheet TM-2) is within ±2 of the same side's C0, pooled: the step does not move the pool-relative edge.
- **R1.** Δ_pop (LR-F − C0) = 0.0 pp, 80 % [−2.0, +1.5], on each side. The modal class is look-1 UNDECIDED.
- **G-C.**
  - B1 ON − OFF lies in [−0.05, +0.01] nats: a larger step sharpens a little, so a B1 FAIL is possible but not
    expected.
  - B2's point estimate lies in [−0.02, +0.04].

**What each final class licenses** (toy first).
- **Toy GAIN** (R1 GAIN, guards pass):
  - production LR-F runs;
  - every batch-1 production NULL is flagged "NULL at the halved rate";
  - the controller's floor becomes a recipe question (`design_learner_recipe.md` §3.7).
- **Toy NULL:**
  - the step size is not the binding constraint at the toy's plateau;
  - TM-4's first caveat is downgraded on the toy side;
  - production LR-F is DEPRIORITISED, not dropped: the toy-to-production transfer is uncalibrated (T2).
- **Toy LOSS:** the controller's cuts were protective, and production LR-F is not run on this evidence.
- **UNDECIDED:** the current look's detectable effect is printed (`readout.r1_detectable_effect`, AM-X47c-1).
- **TRADE** (a guard fails): the orchestrator's ruling.

**The frame.**
- **Matched control:** the same side's C0, at the same pin and seeds. That is toy C0 s3601–s3603 (T1) and production C0
  s3501–s3503. Look 2 adds s3604–s3605 / s3504–s3505.
- **Reads:**
  - R1 as registered (§5.6.4: S over the side's frozen R, Welch at 97.5 %, ±2 pp, two looks);
  - G-F, G-C (§5.6.5, pooled by AM-X47b-1) and G-K;
  - G-E only on a GAIN;
  - reported: sheet TM-2's instruments, the KL, the clip fraction and the churn.
- **A VOID guard, new and proposed for the ruling.**
  - A seed whose trailing-2M mean approx KL exceeds 0.08 (4× the band top, ≈ 2× the predicted top) is stopped and
    VOID. A broken trust region is not the lever under test.
  - Rule 8 applies at 1e-9.
  - It is read by hand with `main.ops.tb_read` at every restart and daily, as K9(b) is. No code enforces it.
- **Order: TOY FIRST**, for four reasons.
  1. The toy shows the same pattern.
  2. The toy needs no FP32_ALT, and its pin is written.
  3. Toy C0 s3601–s3603 already exist, so look 1 costs only the 3 arm seeds.
  4. T\*'s rate held for 80M, so toy C0 runs at a constant 1.4468e-4 and the contrast is a clean ×1.44 throughout.
     Production C0 is expected to cut to 1.0047e-4 within ≈ 2–3M, which makes the production contrast ≈ ×1.73 for
     most of the arm.
- **Cost, on the sheet's §1 basis (≈ 4.3 GPU-h per seed on either side):**
  - toy look 1: 3 × 4.3 = **12.9 GPU-h**;
  - toy look 2, on CONTINUE: + 2 × 4.3 = 8.6. Toy C0 s3604–s3605 are T1b's and are shared if they ran;
  - production: the same, **12.9** (+ 8.6);
  - each R1 read: ≈ 0.3 GPU-h in a window; G-C runs on CPU;
  - if the kit cannot admit a new arm on a side whose pin file is already written (**UNVERIFIED**, the league doc's
    F-AS-8), each side adds its own C0: + 12.9;
  - the build: a kit arm (`lrfloor` / `toy_lrfloor`: the C0 argv + `--min-lr`, with `describe`, a stage condition and
    a test), ≈ 0.25 agent-day.
- **The argv (a design block, NOT a launch command; NEEDS GO).**
  - Toy: toy C0's argv + `--min-lr 2.0833e-4`, run name `rb_x35t_lrfloor_s360x`.
  - Production: C0's argv + `--min-lr 1.7361e-4`, run name `rb_x35_lrfloor_s350x`.

**FINDINGS (LR-F).**
- **F-LRF-1.** The launcher dry-run was not run. Its CUDA-visibility preflight refuses a `--device cuda` dry run with no
  visible GPU, and this unit had no lease. `checkargs` validated the argv against the CHECKPOINTS' pinned parsers, not
  an X35 pin.
- **F-LRF-2.** The VOID guard has no code, like K9(b)'s ceilings (F-LS-9).
- **F-LRF-3.** The toy's prior is weakened by its own record: the toy plateaued 70M after its last cut.
- **F-LRF-4.** No X35 run's telemetry was read for this design. Toy C0 s3601 has finished, and its LR path is
  unread.
- **F-LRF-5 (AM-RP-f).** The toy argv also passes `checkargs` against the toy side's WRITTEN pin `4609c155`
  ([`checkargs_toy_lrfloor_s3601_at_toy_pin.txt`](batch2_lr_floor/checkargs_toy_lrfloor_s3601_at_toy_pin.txt), exit 0,
  `min_lr 0.00020833 (TYPED)`). The dry run is still not run (F-LRF-1).

#### 4.1.1 Toy LR-F: the REGISTRATION (AM-RP-f, 2026-10-20, before X47's look 1)

**Registered before X47's look-1 read** (which reads toy C0's finals, F-LRF-4) **and before any LR-F launch.** The
predictions are §4.1's as designed in `6befe873`, unchanged and now FROZEN: no ruling may revise them.

- **Arm.** `toy_lrfloor`, run `rb_x35t_lrfloor_s{seed}`. Toy C0's argv + `--min-lr 2.0833e-4` (one lever).
- **Pin and matched control.** The toy side's written pin `4609c155` (`x35_pin_batch1_toy.txt`). Toy C0 at the same pin
  and seeds: look 1 s3601–s3603 (T1's), look 2 s3604–s3605 (T1b's, shared if they ran).
- **The registered predictions:**
  - **Mechanism:**
    - the LR sits at the floor on ≥ 95 % of updates, every seed;
    - the arm-mean approx KL is in [0.025, 0.045];
    - the clip fraction is 0.20–0.30;
    - N_prom is within ±2 of toy C0's, pooled.
  - **R1:** Δ_pop = 0.0 pp, 80 % [−2.0, +1.5]; the modal class is look-1 UNDECIDED.
  - **G-C:** B1 ON − OFF is in [−0.05, +0.01] nats; B2's point estimate is in [−0.02, +0.04].
- **The read.**
  - R1 as registered: S over the toy's frozen R, Welch at 97.5 %, ±2 pp, two looks.
  - G-F, G-C (pooled, AM-X47b-1) and G-K. G-E only on a GAIN.
  - Reported beside them: TM-2's instruments, the LR path, the KL, the clip fraction.
  - What each class licenses is §4.1's table.
- **The stop rule.**
  - The arm runs to its fixed `--steps` 360,000,087 (T\* + 70M) and ends with `Training complete`.
  - It ends early only on the sheet §1's stops: K9(b) `warn` with X34's ceilings read by hand (AM-6 at arm end), a
    `FATAL_*`, a 4th crash, or the post-spawn pool-seed check.
- **The VOID guard (registered).**
  - A seed whose trailing-2M mean approx KL exceeds **0.08** is stopped and is VOID. Rule 8 applies at 1e-9.
  - It is read by hand with `main.ops.tb_read` at every restart and daily. No code enforces it (F-LRF-2).
  - AM-6's K9(b) VOID applies as for every arm.
  - A VOID seed enters no read, and it is re-run at the same pin and seed after its cause is fixed.
- **Order and cost.** AM-RP-e (11): toy LR-F fills a GPU idle gap. Look 1 costs 12.9 GPU-h, and look 2 adds 8.6 on
  CONTINUE.
- **Production LR-F is NOT registered by AM-RP.** It is AM-RP-e's slot (8). Its predictions are §4.1's, and it needs a
  registration and a GO before its launch.

### 4.2 SPRT, the PROMOTION-RULE arm (REGISTERED 2026-10-20 by AM-RP-f, before X47's look 1 and any production arm)

**In plain words.** Today a snapshot joins the self-play pool the first eval cycle its win rate against the pool crosses
0.55: one noisy crossing. Under SPRT every eval-cycle snapshot is a candidate, and it is admitted only by its own
sequential test on fresh games. At the small edge a plateaued trainee has over its own pool, that admits more snapshots,
so the pool stays fresher (the memo's §2).

**The ONE lever: `--promotion-sprt`** (T6, built, `gen3_sprt_promotion_v1`, config v129). The argv is the side's C0 argv
plus that one flag. Everything else is C0's.
- **How it acts, read in the code** (`agents/training/sprt_promotion.py`):
  - every eval-cycle snapshot (each 2M) is a CANDIDATE once the pool holds a snapshot;
  - each is decided by a pentanomial GSPRT over fresh MIRRORED pairs: H0 0.50 against H1 0.55, α = β = 0.05, checks
    every 40 pairs, no decision before 40, and a cap of 1,680 pairs that counts as a REJECT;
  - the test plays the candidate's own cycle's sentinels on a seed namespace disjoint from every cycle's, and it
    BLOCKS that cycle's collect;
  - `--promote-threshold` is not read;
  - every test writes `start` / `verdict` lines to the run's append-only `sprt_promotion.jsonl`, and a test cut by a
    restart is `abandoned` (not promoted, never re-run).
- **Runs.** Production `sprt`, run `rb_x35_sprt_s{seed}`. Toy `toy_sprt`, run `rb_x35t_sprt_s{seed}`.
- **Validated** (CPU, [`batch2_sprt/`](batch2_sprt/)): `main.checkargs` exits 0 in three places:
  - production C0 + `--promotion-sprt` against P\*'s pinned parser @`9c033489`
    ([`checkargs_sprt_s3501.txt`](batch2_sprt/checkargs_sprt_s3501.txt));
  - toy C0 + `--promotion-sprt` against T\*'s @`ae2d0901` ([`checkargs_toy_sprt_s3601.txt`](batch2_sprt/checkargs_toy_sprt_s3601.txt));
  - the same toy argv against the toy side's written pin @`4609c155`
    ([`checkargs_toy_sprt_s3601_at_toy_pin.txt`](batch2_sprt/checkargs_toy_sprt_s3601_at_toy_pin.txt)).
  These transcripts are REGENERATED here: the memo's own runs were not committed (F-RP-1) and were not found on disk.
  The launcher dry-run was NOT run (no lease, as F-LRF-1).

**The mechanism, simulated** ([`batch2_sprt/sprt_vs_threshold.py`](batch2_sprt/sprt_vs_threshold.py) →
[`.json`](batch2_sprt/sprt_vs_threshold.json); `sprt_promotion/simulate.py`'s `simulate()` with the shipped rule, τ 1,
5,000 runs, seed 20261020, CPU, 8 s). This is the probability that ONE candidate is promoted, at a true edge over the pool:

| true edge | 0.51 | 0.52 | 0.53 | 0.54 |
|---|---|---|---|---|
| SPRT accept | 0.10 | 0.32 | 0.66 | 0.89 |
| today's rule, Binomial(500) > 0.55 (the code's strict `>`) | 0.03 | 0.08 | 0.17 | 0.31 |
| the same with ≥ 0.55 | 0.04 | 0.10 | 0.20 | 0.34 |

X26 measured ≈ 0.24 promotions per cycle at 280M, so its noise is above binomial (F-RP-4). The ratio is therefore
≈ 2–3×, not the binomial 2.7–3.5×.

**The registered predictions** (GUESS, the memo's, frozen now):
- **Mechanism.**
  - The ACCEPT FRACTION over the arm's candidates (≈ 30 over 3 production seeds × 10) lies in **[0.10, 0.90]**.
    Today's rule promotes 0.04–0.33 per candidate (the memo's figures; F-RP-7 has the exact binomial).
  - N_prom ≥ 2 × the same side's C0's, pooled, **iff** the accept fraction is ≥ 0.10.
  - The final median pool age is below C0's (production: TM-3's 55–76M).
  - Reading it: a fraction near 0.03 means an edge ≈ 0.50 over its own pool, which confirms TM-1 ("progress stopped,
    so the pool stopped refreshing"). A high fraction with no S gain says freshness is not the binding constraint.
- **R1.** Δ_pop (SPRT − C0) = **0.0 pp, 80 % [−1.0, +1.5]**. There is a LOSS tail, because near-copies narrow the pool.
- **The same predictions are registered for the toy side.** The memo derived them for production. On the toy they are
  a second lineage's read of the pool edge (the memo §3, step 1), not a calibration.

**The frame (as LR-F's).**
- **Matched control:** the same side's C0 at the same pin and seeds: production C0 s3501–s3503 (look 2 s3504–s3505),
  toy C0 s3601–s3603 (look 2 s3604–s3605).
- **Reads:**
  - R1 as registered;
  - the guards G-F / G-C (pooled) / G-K, and G-E only on a GAIN;
  - reported: TM-2's instruments, N_prom, the accept fraction, pairs per candidate, the pool's median age and each
    seed's LR path (TM-4: the controller is endogenous).
- **The stop rule.**
  - Production runs to `--steps` 300,000,170 (P\* + 20M), the toy to 360,000,087 (T\* + 70M), each ending with
    `Training complete`.
  - They end early only on the sheet §1's stops: K9(b) `warn` (AM-6 at arm end), a `FATAL_*`, a 4th crash, or the
    post-spawn pool-seed check.
- **The VOID guards (registered; read at arm end from the seed's own `sprt_promotion.jsonl`, deterministic integer counts).**
  - **V-S1, the lever did not act.** A seed with no `verdict` line by its final is VOID.
  - **V-S2, restarts ate the test.** A seed whose `abandoned` tests exceed 25 % of its candidates (strictly) is VOID.
  - **AM-6's K9(b) VOID** applies as for every arm.
  - A VOID seed enters no read, and it is re-run at the same pin and seed after its cause is fixed.
  - **Reported, not a guard:** the arm's wall time per step against its C0's. The test blocks a collect (≈ 11 s per 2M
    cycle in the memo's estimate), and the arms stop at equal steps.
- **Order and cost.**
  - Production is AM-RP-e's slot (6): 12.9 GPU-h, + 8.6 for look 2.
  - The toy is slot (11), filling a GPU idle gap: 12.9, + 8.6.
  - The toy goes earlier only if the 2026-10-20 GPU battery FAILS (AM-RP-e (1)).
- **The build.** It is a kit arm on each side, ≈ 0.25 agent-day (§4.3). This is the FIRST GPU use of T6.

### 4.3 F-AS-8: can the kit admit a NEW arm at a side whose pin is already written? (verified 2026-10-20, CPU, read-only)

**Verdict: at the PIN level YES, so SPRT and LR-F share their side's C0 and need no +12.9 GPU-h. At the KIT level NO,
not today: a kit change is needed first.** The transcript is
[`batch2_sprt/f_as_8_kit_dryform.txt`](batch2_sprt/f_as_8_kit_dryform.txt).

- **What already works for a new arm** (read in `batch.py` / `kit.py` / `launch_x35.sh`):
  - The pin file is per SIDE, not per arm. `pin-check` passes any launch whose `--pin` equals the side's written pin
    (shown: toy `4609c155`, exit 0), and `pin-write` then says "already recorded".
  - The POOL record is per side, so the new arm seeds C0's exact snapshot set. The post-spawn `pool-seed-check` is
    arm-generic.
  - `launch_x35.sh` picks the side from the arm's `toy_` prefix.
  - The trainer flags exist at the toy pin: `checkargs --pin 4609c155` accepts both `--promotion-sprt` and `--min-lr`.
  - R1's input is S per seed for ON and OFF, so C0's finals are re-used as they are.
- **What refuses or misses it today:**
  1. `kit describe` / `stage-check` / `pool-check` / `validate-run` refuse the arm: argparse `choices=sorted(ARMS)`
     (shown, exit 2). Without `describe` there is no argv, so `launch_x35.sh` HOLDs.
  2. `batch.pin_check` judges only `BATCH1_ARMS` (+ the G-E readers). A new arm's run is listed "not batch 1, not
     judged". So a later launch does not check that run's recorded pin (shown with a deliberately wrong one): AM-1's
     per-run check silently skips it.
  3. `batch.stage_status` returns ALLOWED for any arm outside `BATCH1_ARMS` BEFORE the AM-8 side gate. A production
     `sprt` reads ALLOWED while `c0` reads PENDING on the FP32_ALT placeholder (shown). Pre-check 2c
     (`required-check`) would still HOLD the launch, so this is a wrong answer, not a launch hole.
  4. `k9b-batch` (pre-check 11) and `disk` count only batch-1 runs, so a finished new arm's K9(b) is never pre-checked.
  5. `readout.decide` refuses an arm label outside {A1…A6}, and `batch.READ_LABEL` has no entry, so no
     `reads/<arm>_look<k>.json` record can be written.
- **The code change that admits them (a separate unit, NOT built here):**
  - `kit.ARMS`: add `sprt` / `toy_sprt` and `lrfloor` / `toy_lrfloor` (phase 0, `warn`, the side's seeds and length).
  - `build_argv`: the levers `--promotion-sprt` and `--min-lr 1.7361e-4` (production) / `--min-lr 2.0833e-4` (toy).
  - `batch`: a `SIDE_PINNED_ARMS` set (batch 1 + batch 2) that `pin_check`, `k9b_batch` and `disk_check` read, and that
    `stage_status` runs through the AM-8 side gate. Add batch 2's own conditions in a `BATCH2_STAGES`, never in the
    closed `STAGES`.
  - `readout.ARMS` / `batch.READ_LABEL`: the labels `SPRT` and `LRF`.
  - The VOID readers: V-S1 / V-S2 from `sprt_promotion.jsonl`, and LR-F's KL guard as a `tb_read` check.
  - Each with a test that fails on revert. ≈ 0.25–0.5 agent-day for both arms. LR-F's cost line already budgets the
    kit arm.
- **Hazard.** The toy pin `4609c155` is fixed, so a toy batch-2 arm trains on that commit's code. Any T6 / `--min-lr` fix
  that lands after it does not reach the toy arms.

## 5. Findings at staging

These are F13–F24 in the design doc. The four that bind a launch:
- **F14.** A fork defaults to `--behaviour-check fatal`, so the argv types `warn`.
- **F16.** An exploiter fork needs `--no-self-play`.
- **F18.** The fork pool seed leaks X26's post-P\* snapshots; T54 fixes it. **T54 shipped as `cdab9cfe`** (registered in
  [`x35_required_commits.txt`](x35_required_commits.txt)). Its seed runs in the child, so `launch_x35.sh` checks it
  AFTER the spawn (§6).
- **F31 (A2 only, a KNOWN BIAS, not fixed).** T54 filters the snapshots and prunes `pfsp_win_rates` to the KEPT
  steps, but the pool's scalar state is the parent's LATEST, not its value at P\*: `win_rate_vs_bots`, the
  `pfsp_win_rates` values of the kept snapshots (measured by X26 after P\*), and `pool_generation`. A2's early PFSP
  weights therefore start from post-P\* win rates. Every other arm reads the pool's scalars only through the
  self-play fraction, and b-ON and b-OFF share the bias. A2's read must declare it.
- **F21.** The home72 config lived outside the repo. **Closed by T55:** it is committed here (§6).

## 6. The stage kit (T55, built 2026-10-17)

Code in `src/main/x35/` (tests beside it); this directory holds the script, the committed config and the records.
Nothing below has been run on a real P\*: P\* is not called yet.

**(a) The launch: [`launch_x35.sh`](launch_x35.sh)** `<arm> <seed> --pin <X35_PIN_k> [--lease-owner NAME]
[--phase1 FILE] [--thinning] [--at-epoch EPOCH] [GO]`.
- ONE arm × seed per call. The default is a DRY RUN: every pre-check of §3, then the launch line printed. Only the
  literal `GO` launches, and a GO needs the orchestrator's go.
- The argv is BUILT by `python -m main.x35.kit describe` from the frozen R file of the arm's side, never typed by
  hand. Every production argv equals its stand-in in §2 flag for flag (`src/main/x35/kit_test.py`). The toy arms
  (`toy_c0`, `toy_a1`, `toy_a2`, `toy_x47`) type K9(b) `warn` (ruling C3, 2026-10-19: X34's regime and ceilings, read by
  hand); `toy_x47` is the toy C0 argv + `--grow trunk_layers=4` (F-LS-1). `describe` carries the toy ceilings.
- The first failed pre-check is a HOLD (exit 7, written to `~/.claude/jobs/x35/status.txt`). Pre-check 2 reads
  [`x35_required_commits.txt`](x35_required_commits.txt): it must name T54's commit, so **until T54 lands and its
  commit is written there, every launch HOLDs.**
- `--at-epoch E` arms an ABSOLUTE epoch in [now, now + 24 h], runs the pre-checks at arming AND again at E. A
  launcher that dies within 180 s is a HOLD.
- **The POST-SPAWN pool-seed check (T54).** A dry run cannot preview the seed, because it runs in the child. So once
  the launcher is up, the script polls `python -m main.x35.kit pool-seed-check` for up to 30 min. It reads the fork's
  `snapshots/pool_seed.json`, which must be schema 2 with a NON-NULL `fork_step`. T54 only WARNS and seeds unfiltered
  when it cannot read the step, so a null is a HOLD here. `fork_step` must equal P\*'s `num_timesteps` and the call
  record's step, every seeded snapshot must be at or below it, and the startup line `[step filter <= N: DROPPED …]`
  must agree with the record. A FAIL, or no record in 30 min, SIGTERMs the launcher BY ITS PID (it saves and stops
  its child) and HOLDs. `a3_mx` (`--no-self-play`) seeds nothing and is exempt. Tested on constructed records
  (`kit_test.py`); never run on a real fork.
- **The two-phase kits.** Phase 1: `a3_mx` (seeds 3511–3513) or `a6_lx` (3521–3523), +5M each. When all three
  exploiters have ENDED, `python -m main.x35.kit phase1-record --arm a3|a6 --call <R> --out phase1_a3.json` writes a
  write-once record of their finals (path, step ≥ P\* + 5M, sha256). Phase 2 (`a3` / `a6`, seeds 3501–3505)
  reads it and verifies its sha256s before launching.
- **Validated** ([`kit_validation_acb99527.txt`](kit_validation_acb99527.txt)): on a stand-in R (P\* = 180,000,055)
  at pin `acb99527`, `a1`, `c0`, `a2`, `a3_mx` and `a6_lx` passed `checkargs` and "would launch", FORK of
  `rb_x26_s2601`, +20,000,000 / +5,000,000. That is the first `checkargs` of A2 and both exploiter argvs. The real
  script HOLDs at pre-check 1 while X26 is live, so checks 2–8 ran on a test copy without check 1. At today's disk
  pre-check 8 HOLDs batch 1 without `--thinning` (F28).

**(b) The R freeze: `python -m main.x35.refpop plan | freeze | verify`.**
- `freeze` takes `--side`, `--pstar`, every `--archive-dir` and the X candidates (`--x tstar=<zip>`,
  `--x production=baseline:production`), plus P\*'s band cells (`--band-ledger <root> --band-family F
  --band-request Q`, or `--band-json`). It writes `reference_population_<side>.json` here, ONCE, with every sha256
  and a content hash. `plan` prints the nodes and the `main.h2h play-many` command for the band cells.
- Tested on synthetic archives only (`refpop_test.py`), including P\* = 180M → archive 20…160M, A-block 180, 170, 50,
  90, 130M. A start node at or below 150M is REFUSED (F25): the toy side may need a rule first.

**(c) The readout: `python -m main.x35.readout decide <read.json> | self-check`.**
- R1, R2, G-F, G-E, G-C, G-K and the verdict, as pure functions; every declared choice is in the module's docstring
  and in F29.
- **The self-check ran BEFORE any real read: 14 / 14 PASS** ([`readout_selfcheck_t55.txt`](readout_selfcheck_t55.txt)).
  Welch was checked by hand (df 4 exactly) and against scipy's own interval (to 6e-14). Its 97.5 % coverage over
  4,000 draws per cell is 0.977–0.984 at 3 v 3 and 0.972–0.984 at 5 v 5, against a floor of 0.96. Newcombe was
  checked against the published 1998 method-10 table (to 5.1e-5, the 4-d.p. rounding). Every row of the class,
  guard and verdict tables was checked, with rule 8 at ±2.0, at 0 and at −0.05. The population score is exact on
  constructed cells.

**(d) The Kakuna runner: `python -m main.x35.kakuna plan | run | status | gk | smoke | manifest`.**
- It runs on the COMMITTED [`anchors_home72_kakuna.json`](anchors_home72_kakuna.json), byte-identical to the X26
  read's (sha256 `e8a064db…`), and [`home72_manifest.json`](home72_manifest.json), which pins the 72 team files on
  both sides. `metamon_team_dir` is Metamon's cache, outside the repo; no `data/` file is used.
- Before every unit it checks the T28 halt, the manifest and the model's sha256. It runs from the checkout root under
  rule 5b, on a free 95xx port it picks itself. A unit counts by §5.6.6's rule, and `gk` feeds G-K.
- **The smoke** ([`kakuna_smoke_2026-10-17.json`](kakuna_smoke_2026-10-17.json)): ONE 2-game unit, X26 @161,000,169's
  read-only copy against `metamon:SmallRL`, greedy vs greedy, on home72, port 9540, in 73 s. It came back `status OK`,
  `n 2`, `regime_verified_decisions`, `peer_clean`, no asymmetry: COUNTED. Its 0–2 is not a measurement.

**(e) The D1 / X38 readout kit (2026-10-19, B-D7 + the D1 rule; `design_diversity_teams_opponents.md` §4.2).**
- `python -m main.x35.kakuna plan --teamsets unfamiliar | panels | d1h --on X38 --off C0`: the Kakuna runner on the
  three unfamiliar panels (heldout72_hpfix, forum72, p72; 7 units × 20 games per panel per seed final), every panel
  checked against `heldout_panels.json` by content before each unit; panel units record per-game teams
  (`--capture-dir`). `home72` / `away` and G-K are unchanged.
- `python -m main.h2h play | play-many --panel home72 …`: mirrored pairs on a REGISTERED panel (B-D7), the panel's
  content digest in the regime's `team_set`; the input of D1's co-primary P (`python -m main.x35.d1 p-input`).
- `python -m main.x35.readout decide-d1 <read.json>` and `self-check --suite d1`: **10 / 10 PASS before any read**
  ([`readout_selfcheck_d1.txt`](readout_selfcheck_d1.txt)).

## 7. The D1 arm (X38, the pool-WIDENING fork), dry-run validated on the REAL P\*, NEEDS THE ORCHESTRATOR'S GO

**Status: LAUNCH-READY ON PAPER (2026-10-19), nothing launched.** D1 is X35 rank 2 (after A1, before A2;
`design_diversity_teams_opponents.md` §4.2, §6.1). Its ONE lever, b-ON vs C0: `--team-mix w_forum_v1@ABC:0.5
--team-mix-sides both`, i.e. half of each side's draws come from the pin-safe forum corpus
(`src/utils/team_corpora/w_forum_v1/`, **1,982 human-built teams since AMENDMENT D1-CORPUS (2026-10-20,
[`BATCH1_LAUNCH_SHEET.md`](BATCH1_LAUNCH_SHEET.md) §0)**, which carved X54's sealed confirmation panel `forum216b` out of
its AB tier before D1 launched; 2,424 after the 2026-10-19 crash-class lift, 2,415 at `fc467f1f` / `c1b0c1f0`; always
minus every registered panel and its near neighbours). The pin must carry the 1,982-team corpus: launch pre-check 2d
(`kit corpus-check`) HOLDs a pin that does not;
the in-loop eval stays on the pool. Its verdict is D1's OWN rule (`main.x35.d1`), not R1.

| item | value |
|---|---|
| P\* | `models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip`, sha256 `814823c5…91ed` (R's frozen call) |
| pin | a main commit carrying `c1b0c1f0` (B-D2, `--team-mix`) and so `fc467f1f` (the corpus) and `cdab9cfe` (T54): both are in [`x35_required_commits.txt`](x35_required_commits.txt), so pre-check 2 HOLDs a pin without them. **Validated at pin `c1b0c1f0`** |
| seeds | 3501–3503 (look 1), 3504–3505 (look 2); shared with C0 |
| argv (s3501) | [`argv_d1_s3501.txt`](argv_d1_s3501.txt), built by `python -m main.x35.kit describe --arm d1 --seed 3501 --call reference_population_production.json` |
| `checkargs --pin c1b0c1f0` | exit 0, 9 flags accepted by the PINNED parser ([`checkargs_d1_c1b0c1f0.txt`](checkargs_d1_c1b0c1f0.txt)) |
| `launcher --dry-run` | "would launch", FORK of `rb_x26_s2601`, +20,000,000 steps (280,000,170 → 300,000,170), 13.53 GiB required vs 296.59 GiB free, Showdown deps present ([`dryrun_d1_c1b0c1f0.txt`](dryrun_d1_c1b0c1f0.txt)) |
| batch-1 order | `main.x35.kit` `BATCH1`: A1 / C0 ABBA (6), then **d1 s3501, s3502, s3503**, then A2 × 3 |

```
# NEEDS GO. A design block is not a launch command. Pin: any main commit carrying c1b0c1f0 (validated AT c1b0c1f0).
launch_x35.sh d1 3501 --pin <X35_PIN_k>          # dry run: every pre-check, then the line below; add GO only on the orchestrator's go
python -m main.launcher --restart-interval-hours 24 --pin-commit <X35_PIN_k> \
  --model models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip --behaviour-check warn \
  --steps 300000170 --checkpoint-every-steps 1000000 --device cuda \
  --team-mix w_forum_v1@ABC:0.5 --team-mix-sides both --seed 3501 --run-name rb_x35_d1wide_s3501
```

The control is C0 s3501 (the same line without the two `--team-mix` flags, `--run-name rb_x35_c0_s3501`), at the SAME
pin. Startup banners to expect beyond §5.6.8's: `🧬 [TEAM-MIX] w_forum_v1@ABC (1982 teams, tier sha 73a76e808431)
share 0.50 on both sides; eval stays on the pool` with **0 panel hits** and 0 pool copies, and a `🧭 [MATCHUP]` hash
that differs from C0's while the eval line still reads "full pool + 10% sample-team bias". (A pin that predates the carve reads `2424 teams, tier sha 7bcdc835c107`, one that predates the lift `2415 teams, tier sha 65465d25252e`; pre-check 2d refuses both. The banner decides which corpus the run trained on, and each rebuild re-selects a DIFFERENT C-tier sample, not a superset.) A dry run cannot show either
(they are child-side): read them in the first minute. A corpus that fails verification, a held-out panel hit or a
validator drop is a startup `FATAL_CONFIG`.

**The readout (§6 (e)), and what it still needs at read time:** H = `kakuna plan --teamsets unfamiliar` + `run` on
each D1 and C0 seed final (≈ 2,520 games per arm, ≈ 6.3 lane-hours); P = an ab family registered, then `main.h2h
play-many --panel home72` each final vs R's 6 members (500 pairs each: ≈ 18k games per arm, ≈ 10 GPU-minutes or
≈ 8.6 CPU-hours), then `main.x35.d1 p-input`; the guards G-F / G-E / G-C from X35's tools, assembled into the read
JSON by hand (no assembler exists). L / L_c1, GEN-M and the D0e transfer read are reported-only and not built.
