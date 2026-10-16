# Toy crosses: equal compute and toy at cap, against production at 15M (PRE-REGISTRATION)

**Registered 2026-10-16 08:20 PDT, before any game of either cross.** Code at `66e66237` (main). The only games played
before this file was written are a 2 × 20-pair CPU-cost PILOT, described in §3. It wrote to a scratch ledger
(`/tmp/toyx_pilot_ledger_{a,b}`), it is not part of any read, and its score is not quoted.

**Why.** The owner asked: *"did our toy model do any head-to-heads to see its relative performance?"* The X26 road's W4
played the toy at matched steps: R-T1, toy@15M vs EB 15M, ≈ −14.1 pp; R-T1b, toy@5M vs EB 5M, ≈ −25.5 pp
(`../x26_exit_check_2026-10-15/road/w4/`). Two gaps remain, and these two crosses fill them:
1. the toy at the same COMPUTE as production's 15M;
2. the toy at its 100M cap, against the same production reference.

**Names.** The orchestrator's brief called these crosses R-T3 and R-T4. Those two ids already belong to other reads in
[`design_x26_readiness.md`](../../../endstate/design_x26_readiness.md) §3.5 (R-T3 is the external-units read, R-T4 is
cost). To avoid a collision, they are registered here as:
- **R-T5, EQUAL COMPUTE** (the brief's R-T3);
- **R-T6, TOY AT CAP vs EB 15M** (the brief's R-T4).

R-T6 is NOT §3.5's R-T2. R-T2 plays the same toy checkpoints against X26's plateau, and it is untouched by this file.

**The toy and the reference.**
- The toy is T-ENT, "Toy: entity tokens + attention only" (readiness §3.2). Its seeds are `rb_toy_ent_s4001`
  (+ `s4001b`, its fork from 72M to 100M), `s4002` and `s4003`, all pinned at `3899277a`.
- The production reference is the 8 bf16 A/B seeds `rb_bf_eb_s2001`–`s2008`, 15M `final_model.zip`.

## 1. The equal-compute step (how it was computed)

**Compute is measured as the GPU training run's wall clock.** Every run trained alone on the box's one GPU, so wall
clock stands in for GPU time. The clock is SB3's `time/time_elapsed`, read from every `time/` table in the run's
`launcher_child.full.log`. It counts from the process's `learn()` start, so it covers rollouts, learner updates and
the in-learn callbacks (eval included). It excludes process startup and compile before `learn()`.

**Restarts.** A restart starts a new segment, and SB3's clock goes back to zero. The script
[`time_to_step.py`](time_to_step.py) builds the EFFECTIVE time-to-step curve:
- a segment resumed at step s₀ starts at the earlier curve's time at s₀, so work redone after a restart is not counted
  twice;
- `s4001b`'s curve is `s4001`'s curve up to its fork point at 72,000,094, followed by its own segment.

The full curves are in [`time_to_step.json`](time_to_step.json).

**Production's compute at 15M.** T_ref is the mean over the 8 EB seeds of the effective time to 15,000,000 steps:
**9,066.8 s (2.52 h), SD 170.2 s**. Per seed: 9,190 / 8,961 / 8,846 / 8,799 / 9,193 / 9,163 / 9,186 / 9,196 s.

**The toy's step at T_ref.** [`equal_compute.py`](equal_compute.py) linearly interpolates each toy curve. Its output
is in [`equal_compute.json`](equal_compute.json).

| toy seed | toy time to 15M | speed ratio at 15M | step s* at T_ref | s* at T_ref ± 1 SD | **nearest 1M checkpoint (R-T5)** |
|---|---|---|---|---|---|
| s4001 | 2,636 s | 3.44× | 47,709,396 | 46.89M–48.43M | **`rb_toy_ent_s4001/checkpoints/checkpoint_48000139_steps.zip`** |
| s4002 | 2,720 s | 3.33× | 46,263,244 | 45.50M–47.11M | **`rb_toy_ent_s4002/checkpoints/checkpoint_46000198_steps.zip`** |
| s4003 | 2,725 s | 3.33× | 46,783,828 | 45.94M–47.65M | **`rb_toy_ent_s4003/checkpoints/checkpoint_47000187_steps.zip`** |

- **The equal-compute step is ≈ 46–48M, not the brief's ~50–55M.** The toy runs about 3.3–3.4× faster per step at
  15M, but its games lengthen as it trains, so its speed ratio falls with steps.
- **The "nearest" choice is clear of rule 8's band in every seed.** Each s* is ≥ 0.21M from the midpoint between two
  checkpoints (48.0 vs 47.0 for s4001: 0.29M and 0.71M away).
- **s4001's 48M is BEFORE its last passing compile canary (62.83M; readiness F17),** so it trained on the clean graph.

## 2. The cells

| read | rows (toy) | columns (production) | cells | pairs per cell |
|---|---|---|---|---|
| **R-T5, equal compute** | the three checkpoints above | the 8 EB 15M `final_model.zip` | 3 × 8 = 24 | **250** mirrored pairs |
| **R-T6, toy at cap** | `rb_toy_ent_s4001b/final_model.zip` (100,024,189), `rb_toy_ent_s4002/final_model.zip` (100,081,428), `rb_toy_ent_s4003/final_model.zip` (100,080,466) | the 8 EB 15M `final_model.zip` | 24 | **250** |

- **The cells plan** is [`plan/`](plan/), written by [`plan_crosses.py`](plan_crosses.py): `cells_rt5.json`,
  `cells_rt6.json`, and `refs.json` with every sha256.
- **All 14 references LOAD at `66e66237`** (strict loader, historical kwargs, CPU), checked before any cell was
  written. So the crosses run at current main, not at the toy's pin.
- **The h2h path:** `python -m main.h2h play-many --device cpu --torch-threads 1 --threads 2 --purpose ab`, families
  `toyx_rt5_eqcompute` / `toyx_rt6_cap`, requests `toyx_rt5` / `toyx_rt6`, `--oracle-reveal-mode off`, protocol
  `gen3_eval_protocol_v1_h2h` (greedy vs greedy, mirrored). Each cell's schedule key is the default: an
  order-independent digest of the two checkpoints.

## 3. The CPU budget, and why 250 pairs, not 500 (decided HERE, before playing)

**The pilot** was one 20-pair cell (s4002 at 100M vs `rb_bf_eb_s2001` 15M) on the agent cores. That is an R-T6 cell, and
under the default schedule key its 20 pairs are presumably the first 20 of that cell's schedule. The R-T6 read plays
the cell afresh into `models/_ledger`, and the pilot's rows (scratch ledger) are never merged. It ran under rule 5b
(`taskset -c 5-7,13-15`, `nice -n 19`, `mem_cap` 16 GB, `OMP_NUM_THREADS=1`, `--torch-threads 1`, no GPU):
- **0.94 games/s** with 2 Rust threads (148 % CPU, peak RSS 1.0 GB);
- **1.00 games/s** with 4 threads (identical digest: deterministic).

The bound is the single Python forward. A second concurrent process would make 3 heavy jobs on the agent cores beside
the blind-replication runner, which breaks rule 5b's limit of 2.

**The budget:**
- **500 pairs per cell** is 48 cells × 1,000 games = 48,000 games ≈ **14 h**, over the brief's 12 h line.
- **CUT to 250 pairs per cell:** 24,000 games ≈ **7 h** wall on one process.
- A 250-pair cell has an SE of ≈ 2.2 pp (the pilot's pair SD of 0.34). The between-seed spread is the larger term in
  any pooled interval.

**Order and resumption:**
- R-T5's 24 cells are played first, then R-T6's 24.
- Each cell is played in 50-pair batches, recorded in `models/_ledger` as each batch finishes. A re-run of
  [`run_crosses.sh`](run_crosses.sh) resumes from what is recorded.

## 4. The statistics (descriptive), declared before any game

**The primary number** for each cross is **the toy's mean score minus 50 pp**:
- per toy seed: its row mean over the 8 EB columns;
- pooled: the grand mean of the 3 × 8 matrix.

A draw counts ½, and the score is mirrored-pair mean, read from the ledger by `main.h2h.cross.matrix_from_cells` at
`min_pairs` 250.

**The interval: a simple RECTANGULAR-cross statistic, [`rect_stat.py`](rect_stat.py). It is NOT backlog T40.**
- It treats seeds as units over row and column means:
  - V̂ = s²_R / m + s²_C / n, which equals the square `cross_stat`'s V̂ when m = n;
  - the degrees of freedom are Welch–Satterthwaite (here, not the square case's 2(n − 1));
  - the interval is the two-sided **90 %** (the R-T reads' level, readiness §3.5), with the 95 % one alongside.
- Under the additive model it counts cell noise twice, so it is CONSERVATIVE in that term.
- Per seed: the one-sample t interval over that seed's 8 cells (df 7). It holds the EB seeds' spread and cell noise,
  not the toy seed's own training noise.
- Rule 8: a bound within 1e-9 of 0 is reported as touching 0, never as excluding it.
- **Self-check** (`rect_stat.py self-check`, run before this file was committed): 8 / 8 PASS.
  - The square case's Δ̂ and V̂ equal `cross_stat`'s.
  - On a noiseless additive 3 × 8 matrix, Δ̂ is exact and the row means are recovered.
  - Monte-Carlo coverage of the 90 % interval at m = 3, n = 8 is 0.912 (4,000 draws, fixed seed; the floor is 0.88).
  - It refuses 1 × 8 and ragged matrices.
  - A Δ̂ = 0 matrix's interval does not exclude 0.

**For comparison only:** the read also passes W4's R-T1 / R-T1b matrices (`w4/result/w4_read.json`, as committed)
through the same statistic.
- This is RETROSPECTIVE: those point means were already published.
- A dry run of the read script, done to test it before this registration, printed their intervals:
  - R-T1: −14.10 pp, 90 % [−17.16, −11.04];
  - R-T1b: −25.51 pp, 90 % [−28.04, −22.98].
- They are the same estimator, not T40's registered one.

## 5. No verdict rule

**Both crosses are DESCRIPTIVE.** No outcome classifies, gates or stops anything.
- They are not G_m and not G_p(cap), and readiness §3.6's classes (CEILING / SPEED ONLY / NOTHING MEASURABLE) are
  not read from them.
- The R-T6 reference is production at 15M, not at plateau, so R-T6 says only how the toy at 100M scores against
  production at 15M.

**The read:** run [`read_crosses.py`](read_crosses.py) through [`py.sh`](py.sh). It writes `result/read.json`, which
holds each cross's cells, h matrix, per-seed and pooled mean − 50, and `rect_stat`. An incomplete cross is reported
with its missing cells and never interpolated.

## 6. Known hazards (declared)

- **`s4001b` inherits ≈ 9.2M steps (62.83M → 72.0M) trained after s4001's last passing compile canary** (F17: a
  CompileCanary FATAL at 72.66M). Its fork point, `checkpoint_72000094`, is past that canary.
  - The brief names `s4001b` as the cap checkpoint, and readiness §3.5's G_p(cap) amendment does too, so it is used.
  - Its R-T6 row is to be read with this caveat, and the per-seed rows are reported separately so it can be dropped.
- **The wall clock is the runs' own, measured on a shared box.** Agent CPU load at the time was not recorded per
  run. EB trained 10-12 to 10-13; the toys trained 10-15. CPU contention differences between those windows are
  UNMEASURED.
- **Equal compute is GPU-training wall clock, not FLOPs.** The toy is ≈ 17.4 vs 95.7 MFLOP per row (§3.2), so on
  FLOPs its equal step would be far larger. Kernel-launch and Rust-core time keep the wall-clock ratio near 3.3×.
