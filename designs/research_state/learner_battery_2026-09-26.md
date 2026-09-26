# The LEARNER BATTERY (PPO epochs, TF32, policy GAE λ) — PRE-REGISTRATION (2026-09-26)

**Status: REGISTERED before any arm exists.** Written 2026-09-26 at `origin/main` `2cc83080`, while N0
(`ai_v14_01_base`) was live at ~5.6M of 75M steps. This session launched nothing, created nothing under
`models/` and changed nothing in `data/`. The kit is
[`measurements/learner_battery_2026-09-26/`](measurements/learner_battery_2026-09-26/README.md); its launch
script is PREPARED BUT NEVER EXECUTED. The Training Run session launches each arm after the orchestrator's go.

**Codes, each described once here and again where they recur.**
- **N0** = `ai_v14_01_base`, the new lineage's fresh 75M base run (pin `8d07051a`, v121).
- **D_g** = the lineage's frozen generalist learning rate (its §3.1). **FINDING F-1: its rule is not settled.**
- **C** = `ai_v14_02_lbat_ctrl`, the control: N0's recipe continued unchanged. C *is* the lineage's first
  continuation block K1 (§7).
- **E5** = `ai_v14_03_lbat_e5`: 5 PPO epochs instead of 10, at the same dose.
- **T32** = `ai_v14_04_lbat_t32`: TF32 matmuls in the trainer (`--matmul-precision high`).
- **L95** = `ai_v14_05_lbat_l95`: policy GAE λ 0.95 instead of the hardcoded 0.80.
- **U** = the untaught-8 meter: 8 teams, fixed opponent `untaught_meter_opponent_v14` = N0's 24M snapshot.
- **G-A** = the SmallRL external anchor, greedy vs greedy.
- **S** = the speed gain: the fraction of GPU-time per step an arm saves against C.

---

## 1. Why (owner-directed, 2026-09-26), and what the files say about the premise

The owner's case has three parts:
- N0's PPO update takes 52% of each iteration, so the learner caps throughput. This gets worse once Rust
  core M5 makes rollouts nearly free.
- 10 epochs are largely wasted: `clip_fraction` climbs toward 0.2, and the KL controller has raised the lr
  3e-4 → 4.32e-4.
- The policy GAE λ has never been tested. With γ = 1 and a terminal-only reward, λ = 0.80 gives the outcome
  weight 0.8^d at d decisions from the end. The critic is the weakest component.

**What N0's own files say** (`scripts/speed_read.py`, `main.ops.tb_read`, read 2026-09-26 ~11:35 PT):

| phase | median iteration | PPO update (`train/train_ms`) | update share |
|---|---:|---:|---:|
| bots-only (steps ≤ 4,000,032) | 106.0 s | 55.7 s | **0.525** (the owner's 52%) |
| self-play 0.90 (from 4,000,032 on; every fork runs here) | 182.0 s | 56.2 s | **0.309** |

- **lr** is 4.32e-4 (= 3e-4 × 1.2²).
- **`train/approx_kl`** (the LAST epoch's mean) has a median of 0.0128 over the last 20 rollouts. That is
  inside the controller's no-op band [0.005, 0.02].
- **`train/clip_fraction`** (pooled over every epoch) has a median of 0.179.
- **No per-epoch series exists for N0**, because its pin predates `2cc83080`. The "climbs toward 0.2 across
  the epochs" shape is therefore unmeasured in files. C's curves (§5) are its first measurement.

**FINDING F-2: at the regime the forks run in, the update is 31% of an iteration, not 52%.** E5's gain is
therefore bounded near 15% of GPU time today, not near 26%. The quantity that carries over to the post-M5
world is the **update-time ratio** train_X / train_C, and it is reported beside S (§4.1).

---

## 2. The arms: one lever each against a matched control

All four arms share these settings:
- Each is a FORK of **`models/ai_v14_01_base/final_model.zip` (75,005,952)**, named explicitly.
- Budget: +8,000,000 requested, which lands **+8,060,928** (82 rollouts of 48 × 2,048).
- Seed 1001. Pin **`2cc83080`**.
- `--fork-lr … --fork-lr-freeze`: the KL controller is OFF, so the dose is a constant that the run records.
- `--checkpoint-every-steps 500000`.
- **Both battery levers are stated explicitly on every arm** (`--policy-gae-lambda 0.8`,
  `--matmul-precision highest` unless the arm moves them). Each arm therefore differs from C in exactly
  the registered values; `scripts/build_argvs.py` asserts this.

| arm | differs from C in | question |
|---|---|---|
| **C** | — | the lineage's continuation block at D_g |
| **E5** | `--n-epochs 5` **and** `--fork-lr 2·D_g` (ONE lever: passes per rollout at a MATCHED dose, §3) | same strength per step with half the passes? |
| **T32** | `--matmul-precision high` | is TF32 faster on this card, and harmless? |
| **L95** | `--policy-gae-lambda 0.95` | does leaning the advantages on the outcome instead of the critic learn better per step? |

**The argv construction** (`build_argvs.py`, from the lineage kit's `argv_base.txt`):
- **Six flags deleted at `e3ef16db` are DROPPED:** `--bias-additivity`, `--mat-alive-weight`,
  `--no-progress-penalty`, `--self-ko-hp-penalty`, `--switch-bias-weight`, `--no-hand-shaping`. The reward
  is the terminal alone either way (M3 parity measured unchanged).
- **`--arch production` is DROPPED.** It is a fresh-run flag, and `resolve_config` REFUSES it on a fork
  (checkargs, FINDING F-4). A fork inherits the architecture from N0's `model_config.json`, and checkargs
  still prints `✓ every ARCH-surface key matches the production mirror`.
- **`--max-lr 2·D_g` is added to ALL four arms, and only when 2·D_g > 6e-4** (the default cap, 2 × `--lr`).
  The freeze clamps its pin into [`--min-lr`, `--max-lr`], so E5's 2·D_g would otherwise be silently cut.
  With the controller frozen, it has no other effect.

**What each lever touches.**
- **L95:** `--policy-gae-lambda` changes only the policy's ADVANTAGES (and the frozen scalar head's
  returns). The win-prob critic's BCE target is `--win-prob-lambda`'s and is untouched
  (`designs/training/ppo_step.md`). At N0's current ~34 decisions per episode, an action 20 decisions from
  the end sees the outcome at weight 0.8^20 = 0.012 under C and 0.95^20 = 0.36 under L95.
- **E5:** `--n-epochs` is applied on a resume (`model_build.py`: `model.n_epochs = args.n_epochs`).
- **T32:** `--matmul-precision` is a runtime knob and is never inherited.
- **Inherited values:** `--lr`, `--batch-size` and `--n-steps` stay inert/inherited (2048, 2048, equal on
  every arm). The lr is set by the pin.

---

## 3. The dose, the KL controller, and why E5 is dose-matched

Dose = lr × n_epochs / (batch_size × grad_accum_steps) = lr × n_epochs / 65,536 on every arm. C's dose is
D_g × 10 / 65,536. **E5 runs at 2·D_g × 5: the SAME dose.**

**Why the lr is held by a pin rather than leaving the KL target to act.**
- The controller (`AdaptivePPOCallback`: target 0.01, band [0.005, 0.02], ×/÷1.2 per move, a 7-rollout
  cooldown, EMA α 0.2) reads **`train/approx_kl`, which is the LAST epoch's mean KL.**
- At 5 epochs that is epoch 4's KL. Epoch 4's KL is lower than epoch 9's (between about ½ and about ¼,
  depending on whether KL grows linearly or quadratically with drift).
- An unfrozen E5 would therefore raise the lr ×1.2 at most once every 8 rollouts, and only up to `--max-lr`
  = 6e-4 = 1.39 × N0's current rate. That compensates only partly and slowly, along a path that depends on
  the run. The realised dose would be a per-rollout variable that nobody chose: the era-2 exploiter defect
  (`designs/training/step_size_and_batch.md`).
- The lineage's continuation blocks, where any adopted lever would run, are frozen anyway (its §3.1). So
  every arm pins its lr.

**Why E5 is dose-matched and not lr-matched.** E5 at D_g would halve the dose, which is a second lever. The
ledger's standing lesson (M7) is that the dose, not the lr, predicts what a fold does.
- In the low-lr regime, where clipping is rare, 5 passes at 2·D_g ≈ 10 passes at D_g to first order.
  E5 then asks the pure compute question.
- In the high-lr regime, where the clip binds in late epochs, the owner's hypothesis ("epochs 6–10 are
  wasted") predicts that both variants pass. If the hypothesis is false, the lr-matched variant fails and
  the dose-matched one may still pass.
- Adopting a dose-matched E5 leaves the lineage's recorded dose unchanged. Adopting an lr-matched E5 would
  halve every later generalist's dose without anyone saying so.
- The lr-matched variant is the literal test of the hypothesis, and it is **not run** (§9, O2).

**FINDING F-1 (MAJOR, upstream of this battery and of the lineage's K1): the lineage's D_g rule rests on a
false premise.**
- §3.1 of the lineage fixes D_g as "N0's own final learning rate (`python -m main.dose`) … the old lineage's
  construction (its 2.8e-5 was the root's annealed rate)".
- The old root `ai_v13_02_flywheel_winprob` never annealed there. Its `metadata.json` `dose.lr_now` at the
  final save is **2.5e-4**, and its TB `train/learning_rate` reads 0.0003 (4 decimals) at min and max over
  all 740 rollouts.
- **2.8e-5 is the rev-4 FOLD's median lr** (ledger L8469), reused by `ai_v13_09_wcont` and
  `ai_v13_12_plateau` as `--fork-lr 2.8e-5`.
- `main.dose` prints the MEDIAN, not the final rate. For the old root that median is 3e-4.
- N0 sits at 4.32e-4 today, 15× 2.8e-5, so the lineage's own "> 2× away ⇒ FINDING before K1" trigger will
  fire.
- The two readings give continuation doses of **0.20× v8** (2.8e-5) or **≈ 3× v8** (N0's final, if it
  stays near 4.3e-4). That is a 15× difference in the regime every later generalist trains in.
- **D_g is the owner's decision (§9, O1). The battery is written for either value:** the committed argvs
  carry a `__DG__` placeholder, and `launch_arm.sh` REFUSES until `build_argvs.py --final --dg <D_g>` has
  been run.
- **FINDING F-3:** the "clip → 0.2" premise was observed at 4.32e-4 in N0's adaptive regime. At D_g =
  2.8e-5 the per-epoch policy movement is far smaller and clipping far rarer, so fewer epochs may be
  "wasted". C's per-epoch curves are the direct check (§5).

---

## 4. Endpoints, bars and decision rules (fixed now)

### 4.1 PRIMARY: strength per GPU-hour, as a composite of speed and per-step strength

**This endpoint cannot be read off a strength meter directly.** A 15% speed gain buys C about 1.2M extra
steps. At the old continuation's slope (+15.5 pp untaught over 12M ≈ 1.3 pp per 1M), that is ≈ 1.6 pp of
strength, below this instrument's resolution (§4.5). The per-GPU-hour endpoint is therefore the product of
two parts:
- a precisely measured **speed gain S**, and
- a **per-step non-inferiority** test on U.

**THE SPEED ENDPOINT** (`speed_read.py`):
- Rollout collection is the same code on C, E5 and L95. On T32 it differs only in the trainee's GPU forward,
  which is not credited (conservative).
- The iteration time is therefore PROJECTED from the update time, holding the rollout at C's:
  t_X = med(iter_C) − (med(train_C) − med(train_X)); S_X = 1 − t_X / med(iter_C).
- Medians are taken over every post-fork rollout, dropping the first 3 rollouts after each process start
  and every gap across a restart.
- CI: a 95% percentile bootstrap over rollouts (4,000 draws, seed 20260926).
- The projection makes S **robust to CPU contention**, so the lineage's CPU meters may overlap the arms.
- Two DESCRIPTORS sit beside S:
  - the MEASURED iteration ratio, which carries each arm's own day's contention;
  - the update-time ratio train_X / train_C, which is what carries over to M5.
- Mechanics check on N0 as the stand-in (an identity pair): S = +0.00% [−1.32, +1.33] over 16 self-play
  rollouts. At 79 rollouts the half-width is about 0.6%.

### 4.2 Strength instruments (they load v121+ checkpoints; read at pin `2cc83080`)

- **U, the PRIMARY strength meter:**
  - `main.untaught_meter` with refs C, E5, T32 and L95 finals;
  - `--opponent untaught_meter_opponent_v14 --config auto`, `--seed 0`, concurrency 1;
  - **600 games per team (4,800 per ref)**;
  - the meter's cluster bootstrap over the 8 teams (20,000 draws, seed 20260915).
  - Run incrementally with the round-2 G-U driver (`population_loop_r2_2026-09-24/read/scripts/gu_driver.py`,
    refs changed; units of ref × team × 25 battles, durable per-battle rows).
  - Δ_X = U(X) − U(C). **A cell is a pure function of (ref, team, battle)**, so C's first 200 games per team
    ARE its 200-per-team plateau read for the lineage (§7).
- **G-A, a guard only:**
  - `main.anchors --opponent metamon:SmallRL --regime greedy --teamset {away,home} --games 100
    --server rust --device cpu`, S ∈ {0 … 50}: 1,200 games per model.
  - Read for C and for every arm that would otherwise be ADOPTED.
  - Newcombe 95% on arm − C; floor **11.0 pp** (the SOP's three-seed run-level floor).
- **Descriptors, not rules:**
  - each arm's gain against N0's final, U(X) − U(N0), which puts the arms on C's learning curve;
  - for a speed arm that passes the speed bar, **the matched-GPU-time read** U(X_final) − U(C@s_T), where
    s_T is C's 500k checkpoint nearest below the step at which C's cumulative update+rollout time equals X's
    total. It is underpowered (see above) and is reported, never ruled on.

### 4.3 Bars and rules (`scripts/battery_rule.py rule` mechanises them)

**Bars:**
- **Untaught bar: 3.69 pp.** This is the G-U replicate floor, CARRIED PROVISIONALLY from the old lineage,
  where it was measured at 200 games per team. At 600 per team it is conservative.
  - OUTSIDE ABOVE: Δ > 3.69 and CI low > 3.69.
  - EQUIVALENT: CI inside ±3.69.
  - **NON-INFERIOR: CI low > −3.69.**
- **Speed bar:** S ≥ 5% and CI low > 0.
- **Guards** (a KILL is never adopted):
  - G-A OUTSIDE BELOW its 11.0 floor;
  - `[STALL LOGGED]` per 1M steps > 1.5× C's (the stall rate is a PRIMARY endpoint of a win-prob arm,
    lineage §1).

**Rules per arm:**

| arm | ADOPT iff | otherwise |
|---|---|---|
| **E5, T32** (speed levers) | S passes **AND** Δ_X non-inferior **AND** guards clean | NOT ADOPTED. A speed pass with non-inferiority not shown is a reported per-step deficit; there is no extension |
| **L95** (quality lever) | Δ_L95 OUTSIDE ABOVE **AND** guards clean | NOT ADOPTED; 0.80 stays. EQUIVALENT is a positive null (λ in [0.80, 0.95] does not matter at the bar). OUTSIDE BELOW is a FINDING (0.80 is better) |

**T32 FUTILITY (a speed-only look, so no strength forking path):** after T32's 11th post-fork rollout,
`speed_read.py --futility T32`. Projected S < 2.5% ⇒ **STOP T32** and read it NOT ADOPTED (speed). Costs ≈
0.6 GPU-h. Why TF32 might not help: on GeForce Ampere (3080 Ti) dense TF32 tensor throughput is about equal
to the fp32 CUDA-core peak, so any gain is uncertain.

**The honest limit of the composite:** the rule tolerates a per-step deficit up to the 3.69 floor. That is
larger than the ≈ 1.6–2 pp that E5's projected speed is worth on C's curve. An adopted speed lever can
therefore be, at worst, a small net loss per GPU-hour that this instrument cannot see. The matched-time
descriptor is reported so a sign disagreement is visible.

### 4.4 Descriptors recorded for every arm
- **Per-epoch `train/approx_kl_epoch_k` and `train/clip_fraction_epoch_k`** (k = 0…n−1), the means over
  post-fork rollouts, **for C and E5 as the registered descriptor** (T32 and L95 too, since they are free).
- lr and `train/dose_rate` (constant: the manipulation check), `rollout/ep_len_mean`, `[STALL LOGGED]` per
  1M, the eval `win_rate_vs_bots`, and the greedy sentinels.

### 4.5 Power at the chosen n (`validation/power_table.txt`; Wald on game noise, p ≈ 0.6)

The Wald figures are conservative against the meter's bootstrap, which read half-widths of 2.2–2.9 pp at
200 games per team in the old lineage.

| games/team | Δ SE | hw | P(non-inferior \| true 0 / −1 / −2 / −3.69) | P(OUTSIDE ABOVE \| +6 / +8 / +10) | P(EQUIV \| 0) |
|---:|---:|---:|---|---|---:|
| 200 | 1.73 | 3.39 | 0.57 / 0.34 / 0.16 / 0.03 | 0.27 / 0.70 / 0.95 | 0.14 |
| 400 | 1.22 | 2.40 | 0.85 / 0.59 / 0.28 / 0.03 | 0.47 / 0.94 / 1.00 | 0.71 |
| **600 (REGISTERED)** | **1.00** | **1.96** | **0.96 / 0.77 / 0.39 / 0.03** | **0.64 / 0.99 / 1.00** | **0.92** |

- A harmless speed lever passes 96% of the time, and one at the margin passes 3% of the time.
- L95 is detected at ≥ 0.99 if it adds ≥ 8 pp over the block (about half again of what the old continuation
  gained), and at 0.64 at +6 pp.
- G-A (1,200 games, hw 4.0 pp) fires only on a gross loss (a point estimate below ≈ −15 pp).
- Cost of reads: 3–4 refs × 4,800 games ≈ 14,400–19,200 games ≈ 4.5–6 h CPU at 8 workers (the old G-U ran
  4,800 in ≈ 1.5 h), plus 1,200 SmallRL games per G-A model.

---

## 5. Order, GPU time, and what runs when

**Registered order: C → E5 → T32 → L95**, one arm at a time on the single GPU. C goes first because both speed
endpoints and the T32 futility look read C's rollouts (`launch_arm.sh` refuses the others until C has
finished). E5 runs adjacent to C, so the measured-iteration descriptor sees the same box.

| arm | est. GPU-h | basis |
|---|---:|---|
| C (≡ K1) | ≈ 4.5 | 82 × 182 s + 4 evals + restarts |
| E5 | ≈ 3.9 | update 56 → ~28–33 s ⇒ S ≈ 13–15% |
| T32 | ≈ 0.6 if futile / ≈ 4.4 | futility at rollout 11 |
| L95 | ≈ 4.5 | same speed as C |
| **total** | **≈ 13.5–17.3** | **marginal over the lineage (C is K1): ≈ 9.0–12.8** |

The reads for C (and the lineage's registered N0 base read) may overlap the later arms, because S is
contention-robust.

---

## 6. Watch list (Training Run), per arm

### 6.1 First two minutes: the STOP-list (printed by `launch_arm.sh`)
- Role FORK of `models/ai_v14_01_base`; pin `2cc83080`; obs source core; transport rust.
- `[ForkLR]` PINNED and FROZEN at D_g (2·D_g on E5).
- No `[ModelVersion] FATAL`. N0 is v121 and the pin is v123. MIGRATION_FLOOR is 121, and N0's
  `model_config.json` was verified read-only to migrate 121 → 123 with `policy_gae_lambda` 0.8 and no
  shaped-reward evidence.
- `🧮 [MATMUL PRECISION]` reads `high` only on T32.
- The self-play pool is seeded from N0's, with no empty-pool `FATAL_CONFIG`.
- No `CoreObsMismatch` (one = STOP).
- After rollout 1: `train/approx_kl_epoch_0..9` (only 0..4 on E5), `hparams/gae_lambda` 0.95 only on L95,
  and a constant `train/learning_rate`.

### 6.2 While running

| item | read | rule |
|---|---|---|
| fps | `speed_read.py` / `tb_read --tag time/fps --tag train/selfplay_fraction` | an iteration > 1.25× N0's self-play median (182 s; C/L95/T32) ⇒ NOTIFY |
| stalls | `[STALL LOGGED]` per 1M from the arm's own child log (a FORK's log covers exactly its steps) against N0's rate over its last 8M | > 3× in two consecutive 1M buckets ⇒ NOTIFY |
| per-epoch scalars | `tb_read --tag train/approx_kl_epoch_<last> --tag train/clip_fraction_epoch_<last>` | last-epoch KL > 0.05 or last-epoch clip > 0.35 on any arm ⇒ NOTIFY (E5 at 2·D_g is the likeliest) |
| T32 futility | `speed_read.py --fork-step 75005952 --futility T32 C=… T32=…` after rollout 11 | STOP ⇒ stop T32's launcher (never auto-restart) |

---

## 7. Amendment to the NEW LINEAGE's schedule (this battery runs BEFORE the G0′ blocks)

The orchestrator's intent, adopted: winning levers compound for everything after, so the battery sits between
N0 and the continuation blocks.
- **C ≡ K1.** Its argv is K1's by construction (lineage §3.1), with these three changes:
  1. the lineage's continuation blocks move their pin to `2cc83080`;
  2. the six deleted flags and `--arch` are dropped (F-4: the lineage's K1 argv as written would be refused
     on the new tree for the deleted flags and on any tree for `--arch`);
  3. the two levers are stated at their defaults.

  C's untaught read at its first 200 games per team is K1's plateau test.
- **During the reads** the GPU runs **K2 = a fork of C's final at the unchanged recipe** (never idle). Adopted
  levers enter at the **first launch after the verdict**.
- **Adoption scope: the GENERALISTS** (K_k, B_r, C_r). The exploiters and readers (A′, A2′, RB_r, RC_r) keep
  arm A's recipe token-exact. They are the population loop's measuring instrument, and the battery never ran
  in their regime (frozen 2.5e-4, 1.78× v8).
- **Winners combine** (E5 and T32 act on different mechanisms). **Combination guard:** the first block that
  carries ≥ 2 adopted levers reverts to the single strongest winner for the next block if either holds:
  - its projected S is off the product of the single-lever S values by > 5 pp;
  - its untaught block Δ is OUTSIDE BELOW −3.69.
- **Cost to the lineage:** G0′ moves ≈ 9–13 GPU-h (≈ 0.5 day) later. Every block after the verdict runs
  S-faster if a speed lever is adopted (E5: ≈ −0.6 h per 4.5-h block, ≈ −13 h over the loop's ≈ 100 GPU-h),
  so a passing E5 alone repays the battery within the population loop.

---

## 8. What gets adopted, and how it is recorded

An ADOPT sets that flag on every later generalist argv of the lineage and is appended to the ledger with the
verdict line from `battery_rule.py`. A NOT ADOPTED leaves the default. The per-epoch curves and the
update-time ratios are banked as descriptors whatever the verdicts are.

---

## 9. Decisions (recommendation beside each)

**The owner's:**
- **O1 — D_g (F-1), upstream of both the battery and K1.** Either 2.8e-5 (the rate the old lineage's
  continuation actually ran at: +15.50 pp, replicated +12.94, plateau reached; 0.20× v8), or N0's final rate
  as the rule literally says (≈ 3× v8 if N0 ends near 4.3e-4).
  *Recommend 2.8e-5:* it is the value with evidence, and the rule's stated intent was to copy the old
  construction. Say which in the K1 entry.
- **O2 — E5 dose-matched (registered) vs lr-matched.** *Recommend dose-matched* (§3). The lr-matched variant
  is the literal "epochs are wasted" test, and a 5th arm (+≈ 3.9 GPU-h) if wanted.

**The orchestrator's:**
- **D1:** C ≡ K1, and the continuation pin moves to `2cc83080`. *Recommend yes.*
- **D2:** K2 at the unchanged recipe while the reads run. *Recommend yes.*
- **D3:** adoption scope is generalists only. *Recommend yes.*
- **D4:** combine winners under the §7 guard. *Recommend yes.*
- **D5:** 600 games per team. *Recommend yes.*
- **D6:** the order and T32 futility as registered.

---

## 10. Hazards and findings (each is a finding, not a footnote)

- **F-1 (MAJOR):** the lineage's D_g premise is false (§3). The kit refuses to launch until D_g is set.
- **F-2:** the 52% update share is the bots-only phase; the forks' regime is 31% (§1).
- **F-3:** the clip → 0.2 premise was measured at 4.32e-4, not at the battery's frozen D_g (§3).
- **F-4:** `--arch production` is refused on a fork, and the lineage's K1 construction did not say to drop it
  or the six deleted flags (§2, §7).
- **F-5 — not verified here (child-only):**
  - the full `check_compatible` load of a v121 zip under v123 (the config migration WAS verified read-only);
  - the pool seeding;
  - the per-epoch scalars and the TF32 stamp on a real launch;
  - the 60-s `--debug` smoke (it creates a run directory under `models/`).

  The first two minutes of each arm are the test.
- **F-6:** `speed_read.py` counts EVERY `[STALL LOGGED]` line of the run's child log. That is exact for a fork
  and over-counts a window inside a longer run: the stand-in's "189 per 1M" is that artefact, and N0's
  whole-run rate is 59.7.
- **F-7:** `train/n_epochs` is recorded only on some rollouts under the unfrozen controller (the in-band path
  returns before `logger.record`). N0 has 23 points, all ≤ 2.46M. A small code/doc finding for the training
  owner. The frozen arms record it every rollout.
- **F-8:** the 3.69 floor is carried from the old lineage and the old opponent (as the lineage itself
  carries it).
- **F-9:** the stand-in parent is N0's periodic `checkpoint_4800000_steps.zip` at its current lr (4.3e-4),
  because N0 has not finished. That exercised the `--max-lr` rule (2·D_g = 8.6e-4 > 6e-4). The FINAL argvs
  differ from the validated ones only in `--model`, `--steps` and the D_g value.
