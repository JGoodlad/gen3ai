# T25 perf phase on the END-STATE graph: results (2026-10-10)

The plan, the lever ranking and the protocol are in [`PLAN.md`](PLAN.md). This file holds what the GPU windows
MEASURED. Box: RTX 3080 Ti, torch 2.8.0+cu126, fp32 'highest'. The training chain was paused, so no other GPU
process ran; load1 was 1.3–2.4 (quiet). Tags are **MEASURED** unless marked otherwise.

## Window A: the COST ATTRIBUTION (09:13–09:49, lease `perf_T25`)

**Method.** `scripts/update_bench.py`, one process per config, with the closing test's ride-along flags
(`scripts/configs.py`). Each process:
- builds the learner for the config's argv at the production shape (98,304 rows, 2,048 × 32, 10 epochs);
- runs the trainer's own `arm_compile_sentinel` (the R1 startup parity gate, the prewarm, the lock);
- times the real `train()` on the update-fit fixture.

A full PLAIN update is estimated as median(1 epoch) + 9 × the marginal epoch, where the marginal epoch is (3-epoch
call − 1-epoch call) / 2. The code is PINNED to an extracted tree of `6e0a1a7a` (E's exact KO = the 16-roll sum).
Every config **passed the R1 startup gate**.

**Calibration against real launches (MEASURED both sides).** R: bench 51.4 s vs the launch's 51.59 s
(`gpu_checks_endstate_2026-10-09`). E: bench 53.5 s vs ~53.8 s (the closing test's precondition-E launch, descriptive).
Both agree within 0.6 %, so the tiled-fixture bench reads the real update.

| step | config | plain update (s) | Δ vs previous | marginal epoch (s) | diagnostic 1-epoch call (s) | compile + gate + prewarm (s) | peak reserved (MiB) |
|---|---|---|---|---|---|---|---|
| P | `--arch production` | **44.95** (real launch; the bench's P ran all-diagnostic, see F-PA-2) | — | (4.44, distorted) | 13.06 | 159.7 | 6,560 |
| S | + `--token-encoding static` | 44.4 | ≈ 0 vs P | 4.31 | 12.70 | 161.2 | 6,098 |
| SF | + the recovery facts, depth 2 (`static_recovery --trunk-layers 2`) | 44.5 | +0.1 | 4.33 | 12.72 | 164.4 | 6,084 |
| R | + the THIRD trunk round (`--arch static_recovery`) | 51.4 | **+6.9 (+15.5 %)** | 5.02 | 14.63 | 162.9 | 6,758 |
| EK | + the rest of the end state, KO = `ramp` | 53.2 | +1.8 (+3.5 %) | 5.20 | 14.71 | 196.7 | 6,624 |
| E | + exact KO (the 16-roll sum) = `--arch endstate` | 53.5 | **+0.3 (+0.6 %)** | 5.22 | 14.81 | 277.5 | 6,696 |

**Read.**
- The third trunk round is **~80 %** of the end state's extra update time over production: 6.9 of 8.6 s.
- The recovery facts and the static encoding cost nothing measurable.
- The rest of the end state costs 1.8 s, and the exact KO 0.3 s. The 16-term KO loop is one fused kernel on the
  GPU, so it was already cheap. The closed form (now THE `exact`) can save at most that 0.3 s; window B's E_base
  reads it.
- A diagnostic update (every 10th: the per-term noise sampler, grad balance, liveness) costs +7.2 to +8.4 s, about
  0.8 s per update amortised.
- A plain update's first-epoch overhead is ~1.2 s.

**E's profile** (one epoch under torch.profiler; this epoch was a DIAGNOSTIC one, F-PA-2):
- GPU busy 93.2 %; 9,092 kernels per micro-batch.
- Kernel time by class: GEMM 35.8 %, Triton pointwise 26.4 %, attention 15.1 %, Triton reductions 14.8 %, aten other
  7.8 % (`results/window_A/E.json` `profile`).
- The single top kernel is the fp32 memory-efficient attention BACKWARD (`fmha_cutlassB_f32`): 1.92 s of 14.1 s.
  Then the fp32 SIMT GEMMs (`ampere_sgemm_128x64_nn` 1.20 s, `_128x128_nt` 1.12 s).
- This is the shape the 2026-10-03 production profile had. The bf16 lever (A) attacks the GEMM and attention
  classes, together ~51 % of kernel time.
- The raw trace is LOCAL (`~/gen3ai_archive/perf_phase_2026-10-10/traces/`).

**Not measured in window A:** T2's per-step cost per architecture. The template deepcopy failed on E (F-PA-3), and T2
was dropped from P and R to fit the deadline. It is window C.

## Window B: the LEVERS + perf item 1's deferred A/B (12:32–13:10, file-based handoff)

**Method.** Same bench (plain updates, phase-advanced, one diagnostic call), at HEAD `e65a31ab` (E's exact KO = the
closed form), except the item-1 pair, which ran on extracted trees of `957d4dbd^` and `957d4dbd`. Every
`--profile` epoch is a PLAIN one.

| job | per-call walls, 1 ep / 3 ep (s) | update (s) | vs its baseline | GPU busy (profiled plain epoch) | kernel time / epoch (s) | gate / canary |
|---|---|---|---|---|---|---|
| **E_base** (HEAD, closed-form KO) | 6.57, 6.57 / 17.00, 17.01 | **53.5** | = window A's 16-roll E (53.5) | 85.6 % | 5.89 | PASS / PASS (grad cosine 1.0000) |
| **E_bf16** (`trunk_precision` bf16) | 5.24, 5.28 / 16.54, 13.26 | 48.7 (the 3-ep pair is noisy) | **−20 % per 1-ep call**; kernel time −21 % | 85.4 % | 4.67 | REFUSED (loss rel 9.7e-4 > 1e-4) / FAIL (1.3e-4) |
| **E_cudagraphs** (`cudagraphs` preset) | — | — | — | — | — | gate PASS, bit-equal (loss rel 0); then DIED on the first update (F-PB-2) |
| **E_coordesc** (`coordesc` preset) | — | — | — | — | — | never finished compiling: killed by its 11-min timeout (F-PB-3) |
| **P_pre_item1** (`957d4dbd^`, production) | 5.69, 5.70 / 14.63, 14.66 | 45.98 | — | 86.0 % | 5.10 | PASS |
| **P_at_item1** (`957d4dbd`, perf item 1) | 5.67, 5.67 / 14.50, 14.50 | **45.38** | **−0.60 s (−1.3 %)** | 86.6 % | 5.11 | PASS |

**Read.**
- **The closed-form KO** (now THE `exact`) is cost-neutral on the GPU: 53.5 vs 53.5 s. Exact KO's whole GPU cost was
  0.3 s (window A), and Inductor already fused the 16-term loop. The CPU gain (KO function 23.9 → 6.1 ms) is real but
  off the training path.
- **bf16 trunk (lever A) is the biggest measured lever.**
  - Per-epoch kernel time −21 %: GEMM 2.11 → 1.51 s, attention 0.70 → 0.34 s (−51 %), pointwise 1.50 → 1.39 s.
  - The 1-epoch calls are −20 % (4 calls, consistent); peak memory −540 MiB.
  - The full-update estimate (48.7 s) rests on a noisy 3-epoch pair; a −15 to −20 % update is the ESTIMATE.
  - The fp32 R1 gate refuses it, as predicted: compiled bf16 vs eager bf16 is a bf16-rounding gap, 10× the fp32 bar.
    Adoption needs PLAN.md §2: the matched-envelope parity rule, K9(b)'s bar, and a strength A/B.
- **Perf item 1** (`gen3_batched_host_reads_v1`), the deferred measurement:
  - marginal epoch 4.48 → 4.41 s; plain update −1.3 %;
  - `cudaStreamSynchronize` per micro-batch 126.3 → 121.2; GPU busy 86.0 → 86.6 %;
  - the scalar-read meter's blocked time moved from many short reads to fewer long ones (0.15 → 1.76 s per call):
    the batched reads now WAIT on a deep queue instead of draining a shallow one many times.
- **THE NEW TOP LEVER: the update is ~14 % GPU-IDLE.** Every plain epoch profiled at 85–87 % GPU busy, with ~121
  `cudaStreamSynchronize`, ~235 `cudaMemcpyAsync` and ~4,500 kernel launches per micro-batch.
  - Item 1's counter (`host_sync_guard_test`, the learner's syncs) does not see these ~121 per micro-batch
    (~58,000 per update).
  - Their source is UNVERIFIED. Candidates: the ride-along heads' eager tail (five ensemble + RND + ADV + OPP heads,
    each with its own optimizer step per micro-batch) and per-micro-batch device-to-host copies.
  - A with-stack profile attributes them (window C). At 14 % idle, closing it is worth up to ~7 s per E update
    (ESTIMATE), and it is also what lets bf16's GPU-side gain reach the wall clock.
  - ⚠️ **CORRECTION (2026-10-10, before window C).** Every profiled call above was ONE epoch, i.e. EPOCH 0. Epoch 0
    carries the once-per-update work: the ride-along accumulator's host reads (`ridealong_terms.py` `acc.observe`),
    the calibration and noise reads, and the label alignment. It costs ~1.3 s more than a later epoch
    (6.57 vs 5.21 s). So the 14 % idle is EPOCH 0's, and later epochs' idle fraction is NOT measured. The
    "~7 s per update" above is an UPPER bound; if later epochs are busy, the lever is nearer epoch 0's ~1 s.
  - A CPU proxy (`--sync-sites`: every host-blocking Tensor call wrapped and attributed to its `src/` frames)
    points at the ride-along tail: per micro-batch, a `bool(isfinite(grad norm))` for the four heads and one per
    RND variant, plus the epoch-0 meters. These are a handful per micro-batch, not ~121. Hypothesis
    (UNVERIFIED): the rest are pageable host-to-device copies, which CUDA stages synchronously.
  - Window C profiles 3 epochs and runs the wrapped meter on the GPU (which also counts pageable H2D copies).

## Window C: who idles the GPU, the SHARED compile cache, T2 per architecture (16:00–16:42)

**E_stack** (`--profile --profile-epochs 3 --sync-sites`, HEAD `15982f6c`): plain update 53.8 s (gate PASS).

| read | epoch 0 alone (window B) | 3 epochs | so, later epochs |
|---|---|---|---|
| GPU busy | 85.6 % | **92.1 %** | ~95 % |
| `cudaStreamSynchronize` per micro-batch | 121 | 40 | ~0 |
| kernel launches per micro-batch | 4,528 | 3,304 | ~2,700 |

- The wrapped meter (every host-blocking Tensor call and every pageable host-to-device copy, on the GPU) counted
  33 calls per micro-batch and 2.22 s blocked over 3 epochs.
  - **1.75 s of it is one site**: `ridealong_heads.py:466` (`.item()`), in the ride-along accumulator's EPOCH-0 read
    (`ridealong_terms.py:395`). It blocks once per micro-batch on the deep queue and drains it.
  - Everything else is < 0.15 s.
  - No pageable H2D copy appeared.
- **Read:** the idle time is ~1 s per update in epoch 0 (that drain) plus ~5 % launch-bound gaps in each later epoch.
  That is ~3 s per update in all (~6 %), not the 7 s first estimated (window B's correction stands).
- The cheap fix is to make the ride-along epoch-0 read DEFERRED, as perf item 1 did for the learner: it rides the
  optimizer step's batched transfer. NEXT.

**The SHARED compile cache** (`gen3_shared_compile_cache_v1`, `62c6165e` + the clean-rule fix `15982f6c`): a FRESH
one-update `--arch endstate` launch through the launcher's normal pinned path, a scratch archive, an empty shared root:

| launch | cache line | T2 up | R1 reset + gate + prewarm | whole launch (start → `Training complete`) |
|---|---|---|---|---|
| cc_cold | `SHARED-NEW` (key `182d764f2a0e`) | **850.7 s** | not finished: killed by the 18-min budget mid-R1 | > 17.8 min |
| cc_warm (a different fresh run) | `SHARED-HIT`, the same entry | **124.9 s (6.8× faster)** | **104.1 s** (vs ~277 s cold in the bench) | **5.6 min** |

- A fresh launch or restart of an unchanged commit + config starts ~15 min faster: T2 −726 s, R1 ~−170 s.
- The pinned worktree read CLEAN under the new rule (ignored importables counted), so pinned runs share.
- Caveats:
  - cc_cold was killed mid-R1, so cc_warm's R1 hit was at most partial and 104 s is an UPPER bound for a warm R1.
  - n = 1 per arm; descriptive.
  - The sharing-off control was skipped: a cold shared start compiles what a per-run cold start compiles.
- The throwaway run dirs and the 3.3 GB shared root were deleted after the read. The launcher output is in
  `results/window_C/logs/cc_{cold,warm}.log`.

**T2 per flush, ONE slot** (graph backend, the service's own startup parity gate; `--t2-only`, code pinned at
`6e0a1a7a`):

| arch | bucket 64 | bucket 256 | T2 startup (2 buckets) |
|---|---|---|---|
| P | 3.17 ms | 6.38 ms | 243 s |
| R (+ the third round) | 3.44 ms (+8.6 %) | 7.04 ms (+10.3 %) | 250 s |

The third trunk round costs T2 about the share it costs the update. This matches `gpu_checks_endstate`'s per-step T2
wait (+11 %). **E's T2 is NOT measured**: E_stack passed its buckets out of order and the service refused them; the
harness now sorts them.

## Window D: the ride-along host-read fix, the bf16 parity envelope (19:36–19:57, file-based handoff)

**Job 1: the ride-along host-read fix (`3545b868`) against its parent (`5e399caf`).**
- Both runs are E at the production shape, pinned to extracted trees, with `--profile --profile-epochs 3 --sync-sites`.
- The comparator is window C's `E_stack` (HEAD `15982f6c`).
- Both pinned runs passed the gate. Load1 was 4.7–6.0; no other GPU process ran.

| read | E_stack (window C) | E_ride_before (`5e399caf`) | E_ride_after (`3545b868`) |
|---|---|---|---|
| plain update (s) | 53.81 | 53.59 | **53.24** (−0.35 s, −0.7 %) |
| 1-epoch calls (s) | 6.69, 6.70 | 6.50, 6.51 | 6.34, 6.35 |
| first-epoch overhead (s) | 1.46 | 1.27 | 1.14 |
| marginal epoch (s) | 5.235 | 5.232 | 5.210 |
| GPU busy, 3 profiled epochs | 92.1 % | 93.2 % | **95.1 %** |
| wrapped host-blocking calls per micro-batch | 33.4 | 33.4 | **2.1** |
| host-blocked time, 3 epochs (s) | 2.22 | 2.36 (1.85 at `ridealong_heads.py:466`) | **0.50** (0.28 at the one batched transfer) |
| memcpy + memset events | 26,539 | 26,539 | 21,644 |

**Read.**
- The fix does what it says: the epoch-0 drain is gone, from ~120 reads to 1 batched transfer per micro-batch, and
  GPU busy is up 1.9 pp.
- The wall gain is **0.35 s per update**, not the ~1 s ESTIMATE. Most of the old drain overlapped queued GPU work, so
  it did not add its whole length to the wall.
- The two 1-epoch reps per arm agree to 0.01 s, so the difference is real.
- **Calibration.** E_stack → E_ride_before (the same code but 2 commits) moved 0.22 s at a lower load1 (8 → 5). Tool
  drift between windows is therefore ~0.2 s. Compare only within a window.

**Job 2: the bf16 parity ENVELOPE** (`scripts/bf16_envelope.py` on the `3545b868` tree; `results/window_D/E_bf16_envelope.json`).
- **Setup.** One compiled-bf16 install of `--arch endstate` at B = 2,048 (compile + prewarm 243 s).
- **Arms.** On the gate's own rows (slices 0 and 1): compiled bf16 ×2, eager bf16 ×2 and eager fp32 ×2.
- **States.** Fresh, the ladder rungs (0.05 × 4 seeds, 0.1 × 2), and 1 / 3 / 6 real bf16 updates.
- **Run-to-run.** Both bf16 arms were bit-identical on their repeat: on this GPU, eager and compiled bf16 are
  deterministic run to run.
- **Recompiles.** R1's dynamo cache held 1 entry throughout, so the fp32 reference arm's class swap never
  recompiled.

| quantity (bf16 vs the fp32 eager reference) | range over the 20 reads |
|---|---|
| eager loss rel | 2.5e-5 – 3.26e-3 |
| eager gradient rel-L2 | **0.010 – 0.169** (trained states 0.02–0.11) |
| compiled gradient rel-L2 | 0.012 – 0.183 |
| decision, legal log-prob \|Δ\| (eval, no grad): p50 / max | 8e-4 – 9e-3 / 5e-3 – 0.12 |
| decision, V \|Δ\| max | 7e-5 – 9.4e-3 |

**The derived bars** (`scripts/derive_bf16_bars.py` → `results/bf16_parity_bars.json`; PLAN.md §2a):
- DRIFT: loss 0.014 and gradient rel-L2 0.68, each 4 × the healthy maxima 3.26e-3 / 0.169.
- ENVELOPE K: loss 5.4, gradient 6.4, per parameter 12, each 4 × the healthy maximum ratios 1.35 / 1.58 / 2.94.
- **The loss floor is bf16's unit roundoff 2^-9.** Under the fp32 floor of 1e-4, one read's eager loss error landed
  at 2.5e-5 by chance while the compiled one read 8.5e-4. That made an 8.5 × "ratio" out of rounding luck alone, and
  K_loss = 35.

**Read.**
- **The bf16 trunk moves the micro-batch GRADIENT by 1–18 % relative L2 from fp32's.** This is a real numerics
  change, which is why its strength A/B is mandatory (PLAN.md §2d).
- The micro-batch figure overstates the optimizer step's, because rounding errors from different rows partly average
  over the 32-micro-batch accumulation (ESTIMATE, ~√32 smaller if they are independent; NOT measured).
- **The per-parameter TEETH are coarse.** The refusal threshold, K × max(own bf16 error, the fp32 bar), has a median
  over judged parameters of 0.2–1.3 relative error by state. So the rule refuses a dropped or sign-flipped path
  gradient (error 1–2) on most parameters, but not a 10 % one.
- **What it still catches at fp32 strength:** non-finite gradients, mis-wired arms and a broken reference.
- **Rule 8.** Every healthy read sits ≤ 25 % of the way to its bar.

## Window E: the levers on their FLAG paths (22:43–23:18, file-based handoff; the last window during the closing test)

**Method.** Every job ran on the extracted tree of `2652864e`, which holds levers A, C and 4. Load1 was 1.2–2.2 and no
other GPU process ran. `E_base_e` is the paired baseline: its 53.21 s equals window D's `E_ride_after` (53.24 s).

| job | argv beyond E | plain update (s) | 1-ep / 3-ep calls (s) | marginal epoch (s) | diagnostic 1-ep call (s) | peak reserved (MiB) | gate / canary |
|---|---|---|---|---|---|---|---|
| **E_base_e** | — | **53.21** | 6.27, 6.29 / 16.69, 16.73 | 5.214 | 14.63 | 6,718 | PASS / PASS (cosine 1.0000) |
| **E_bf16_flag** | `--trunk-precision bf16` | **41.17 (−12.0 s, −22.6 %)** | 5.07, 5.07 / 13.08, 13.10 | 4.011 (−23.1 %) | 11.32 (−22.6 %) | 6,308 (−410) | **PASS under the bf16 rule** / PASS |
| **E_cg** | `--learner-cudagraphs on` | — | — | — | — | — | **DIED in its own startup gate** (F-PE-1) |
| **E_cg_bf16** | both | — | — | — | — | — | died the same way |

**The bf16 rule's first GPU verdict** (startup, the bench's perturbed weights; trained regime):
- drift: loss 1.18e-3 ≤ 0.014, gradient 9.69e-3 ≤ 0.68;
- envelope: loss 2.13e-3 (eager 1.18e-3); gradient 1.18e-2 (eager 9.69e-3);
- the worst per-parameter ratio is 1.94 ≤ 12;
- the canary on trained weights PASSED, with gradient cosine 0.99943 against eager bf16.

**Read.**
- **bf16 on its real flag path cuts the end-state update by 22.6 %** (53.2 → 41.2 s), which is more than window B's
  bench-switch read (−20 % per 1-epoch call). The R1 compile + gate + prewarm was 215 s vs 212 s fp32.
- At 41.2 s the end-state update is now BELOW production's fp32 44.95 s.
- ⚠️ **The bf16 profiled epoch read 51 % GPU-busy** (kernel time 4.66 s, the same 9,000-odd kernels per micro-batch,
  over a 9.18 s profiled span vs a 5.07 s unprofiled call). The profiler roughly doubled the host time of the bf16
  epoch alone. The likely cause is autocast's per-op casts recorded as host events (UNVERIFIED). Trust the
  unprofiled walls. Whether a bf16 update is launch-bound is NOT settled by this profile (F-PE-3).

**T2 lever 4, lead lanes** (`T2_lead`: `lead_lanes_benchmark`, the PRODUCTION policy, fresh-perturbed, production's
slot layout of 25 slots, 8 lanes, buckets 8 / 64 / 256, 212 opponent + 245 trainee rows per step, 6 interleaved
OFF/ON blocks × 50 steps):

| | OFF | ON (`--t2-lead-trainee`) |
|---|---|---|
| host step (flush + wait), median | 29.94 ms | **28.25 ms** |
| flush (host launch) / GPU wait | 1.23 / 28.69 ms | 1.24 / 27.00 ms |
| ON/OFF per block | — | 0.934 – 0.947 (median **0.942**) |
| slots on the trainee's lane | 0, 8, 16 | none |
| decisions, ON vs OFF | — | **bit-identical** (max \|Δ\| 0.0); 0 compiles / captures after the freeze |

**Read.**
- −1.7 ms per host step (−5.8 % of T2's step), consistent in all 6 blocks. At ~400 host steps per production
  rollout that is **≈ −0.7 s per cycle** (ESTIMATE from the per-step read; the end state's T2 is ~10 % heavier, so
  slightly more).
- A pure scheduling change: the same kernels, nothing else moved.

## Findings from window E (standing rule 7)

- **F-PE-1 (lever C fails on the GPU, loudly, at its own startup rung).** `--learner-cudagraphs on` died in the
  graphed route's ACCUMULATION rung (`compile_regions._r1_accum_rule`).
  - The error: "accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run", raised in the
    SECOND micro-batch's `backward()`. It named the embedding table's gradient-addition node (`static_tokens.py:191`,
    `embeddings.move_embedding`, a tensor used more than once).
  - The cause: the first replay's backward output became `.grad` (AccumulateGrad stole the graph-pool tensor). The
    second replay then overwrote that pool slot before the accumulation read it. This is exactly the hazard the rung
    was built to catch (the builder's FINDING 1); it caught it before any update.
  - **Two defects:**
    1. The rung raises a bare torch `RuntimeError`, not a typed `CompileTrainerError`. A real launch would see an
       untyped crash, and the launcher may RESTART into it rather than stop FATAL_CONFIG.
    2. The route itself needs `.grad` never to alias the pool. Candidate fixes, untested: materialise every
       parameter's `.grad` (`zero_grad(set_to_none=False)`) before each accumulation group so AccumulateGrad adds in
       place; or clone the graphed backward's grads out of the pool, at a cost.
  - Not fixed in this phase (no GPU window left during the closing test).
- **F-PE-2 (lever 4 is a small, clean win).** −5.8 % of the T2 step, bit-identical. Adoption is an operational call
  (a runtime knob, not versioned).
- **F-PE-3 (the bf16 profile is distorted).** Under torch.profiler the bf16 epoch's host time doubled while its
  kernel time fell 21 %, so the profiled GPU-busy % (51 %) is an artifact. The CUDA-graphs lever's value under bf16
  stays the open question that lever C's fix would answer.

## Findings (standing rule 7)

- **F-PD-1 (the ride-along fix's gain).** It is −0.35 s per E update (−0.7 %), with host-blocked time 2.36 → 0.50 s
  and GPU busy 93.2 → 95.1 %. The ~1 s ESTIMATE was high because the drain overlapped GPU work.
- **F-PD-2 (bf16 moves the gradient).** On the gate's rows, eager bf16's micro-batch gradient is 1–18 % rel-L2 from
  fp32's (trained 2–11 %). The bf16 parity rule's per-parameter teeth are coarse by construction (median refusal
  threshold 0.2–1.3 relative error). The strength A/B is mandatory before adoption.
- **F-PD-3 (the scalar floor).** A SCALAR's bf16 error can land near zero by chance, so the loss envelope's floor is
  bf16's unit roundoff, not fp32's 1e-4.

- **F-PC-1 (harness).** `--t2 256,64` was refused (`ServiceSpec: buckets must be strictly ascending`), so E's T2 is
  lost. Fixed for later jobs; T2_P / T2_R ran with it.
- **F-PC-2 (cold cache vs budget).** A cold `--arch endstate` launch takes > 17.8 min to its first update, with T2
  alone 14.2 min. This is the per-restart cost the shared cache now removes, but only for a CLEAN tree at an unchanged
  key.
- **F-PC-3 (the closing test stays per-run).** At the orchestrator's request, the closing test's remaining seeds run
  with `GEN3AI_SHARED_COMPILE_CACHE=0`, so their setup stays constant. They pay the cold startup every restart.

- **F-PB-1 (bf16 gate).** The fp32 bar refuses bf16 by design: compiled vs eager loss rel 9.7e-4 at startup, and
  1.3e-4 at the canary on trained weights. Not a bug; it is the missing bf16 parity rule.
- **F-PB-2 (CUDA graphs).** R1 under cudagraph trees passes the startup gate bit-for-bit. The first update then dies
  in the grad-balance probe's `autograd.grad(retain_graph=True)` (`grad_balance._flat_grads`), with a torch
  "TODO: graph recording observed an input tensor deallocate during graph recording that did not occur during
  replay". The probes (grad balance, the per-term noise sampler) need a non-graphed route; a diagnostic update is
  always the process's first.
- **F-PB-3 (coordinate descent).** `coordinate_descent_tuning` did not finish compiling R1 in 11 minutes (one core
  at ~88 %, CPU-bound tuning). As a startup cost it is impractical unless the shared compile cache serves it warm.
  Even then, its first compile is a >11-min cold start. NOT measured.
- **F-PB-4 (the idle 14 %).** See the read above. It is the top unexplained cost in the update.

## Findings from window A (standing rule 7)

- **F-PA-1 (harness, fixed).** The first attempt died in E's first `train()`: the ride-along heads' Adam state was
  still on the CPU. The bench builds the learner on CPU and moves it; a launch builds it on the device. The bench now
  moves every optimizer's state.
- **F-PA-2 (harness, fixed; two readings are distorted).** The diagnostics cadence's phase is the ROLLOUT index
  (`num_timesteps // rows`). The first E and P runs never advanced it, so every call was a diagnostic update. Their
  1-epoch calls carry ~7–11 s of probes, and their marginal epoch is distorted: E read 4.07 there vs 5.22 plain,
  because the diagnostic overhead differs between a 1-epoch and a 3-epoch call. The bench now times plain updates at
  phase 1–9 plus one diagnostic call. E was re-run plain (`E_plain.json`); P's plain bench value is NOT measured, and
  the real launch's 44.95 s stands in.
- **F-PA-3 (harness, fixed).** The trained policy does not `deepcopy`: its per-forward stashes hold non-leaf graph
  tensors. The T2 template is now a fresh build carrying the learner's state dict.
- **F-PA-4.** E's compile + gate + prewarm (277 s) is 1.7× production's (160 s). That is the learner side of every
  launch and restart, on top of T2's build. It feeds the owner's compile-cache decision (PLAN.md §3).

## Files

- `scripts/`: `update_bench.py`, `configs.py`, `window.sh`, `window_watch.sh` (the file-based handoff),
  `cpu_flops.py`, `module_flops.py`, `cpu_flops_all.sh`.
- `results/cpu_flops_*.json`, `results/module_flops_endstate.json`: the CPU FLOP reads (PLAN.md §0).
- `results/window_A/{P,S,SF,R,EK,E,E_plain}.json` + `logs/`: window A. `results/window_A_attempt1_device_bug/` is the
  dead first attempt (F-PA-1).
- `results/window_B/{E_base,E_bf16,E_cudagraphs,E_coordesc,P_pre_item1,P_at_item1}.json` + `logs/` + `window.log`:
  window B. Raw traces are LOCAL (`~/gen3ai_archive/perf_phase_2026-10-10/traces/`).
- `results/window_C/{E_stack,T2_P,T2_R}.json` + `logs/` (incl. the two launches' launcher output) + `window.log`:
  window C.
- `results/window_D/{E_ride_before,E_ride_after,E_bf16_envelope}.json` + `logs/` + `window.log`: window D;
  `results/bf16_parity_bars.json`: the derived bf16 bars (`scripts/derive_bf16_bars.py`); `scripts/bf16_envelope.py`.
- `results/window_E/{E_base_e,E_bf16_flag,E_cg,E_cg_bf16,T2_lead}.json` + `logs/` + `window.log`: window E (the pinned tree
  `~/gen3ai_archive/perf_phase_2026-10-10/trees/at_lever_a` = `2652864e`).
