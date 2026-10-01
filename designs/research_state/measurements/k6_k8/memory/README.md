# K6 — the learner's CUDA memory, calibrated (2026-09-30)

**Verdict.** A healthy production-shape learner on torch 2.8 does not grow its LIVE CUDA memory. The
quiescent allocated floor holds within **2.0 MiB** over 25–57 updates per process. Reserved bytes settle
by the first diagnostics update and then stay flat. So the leak detector
(`src/agents/training/cuda_memory_trend.py`) can use a tight noise bar, and noise cannot make it stop
a run. **`expandable_segments:True` is NOT recommended.** It removes the inactive-split bytes but saves
only 60 MiB of reserved (1%). It grows reserved MORE after the warm-up, and it makes `memory_stats`
report **zero segments**, which blinds every `cuda_segments_after_freeze` counter (T2's included). Its
update-time effect cannot be separated from box contention.

Files:

- `driver.py`: the measurement. One unit is one fresh process.
- `analyze.py`: the description, the detector replay and the false-trip sweep.
- `result.json`: everything `analyze.py` derives.
- `healthy_trace.json`: every measured sample. It is the routine test's fixture.

Raw rows, one JSON line per sample and fsynced, are in `~/gen3ai_archive/k6_k8_memory/units/<unit>/samples.jsonl`.

## Setup

- **What a unit runs.** `driver.py run --arm default|expandable` on `gen3ai_torch28` (torch
  2.8.0+cu126), RTX 3080 Ti (12 GiB). It runs under `scripts/ops/gpu_lock.sh timeout 1200` and
  `mem_cap.sh 40`, and is never a training run.
- **The learner.** A production-surface learner (`testkit.fresh_model` at `production_args()` +
  `recipe.fresh`: 48 envs x 2,048 steps = 98,304 rows, micro-batch 2,048, grad-accum 32, the win-prob
  critic, `--diagnostics-every 10`, fp32). It is compiled exactly as a launch arms it: the real-obs
  parity gate, then `arm_compile_sentinel` (reset, prewarm the four production signatures, and lock
  after the first update).
- **The buffer.** The REAL buffer is arm C's pickled rollout (`learner_bench/20260928_135948_cuda`).
  Its keys and shapes are checked equal to this learner's.
- **Each cycle:**
  1. Restore the buffer.
  2. A ROLLOUT BLOCK: 256 eval/no-grad `policy(obs, action_masks)` calls at 48 rows (the rollout's
     signature), then the behaviour log-probs of all 98,304 rows in train/no-grad chunks of 2,048 (the
     rank probe's declared signature). K9's behaviour gate then holds: max |Δ| 4.8e-7 to 3.8e-6 at
     fp32.
  3. Sample.
  4. `train()`, with a fresh minibatch permutation each time.
  5. Sample.
  6. Every 10 updates, a `model.save`.
- **The rollout index advances per cycle**, so the diagnostics cadence fires on updates 1, 10, 20, …
- **Shortened epochs.** `n_epochs = 2`, not the production 10, so a 17-minute unit holds 50+ updates.
  Memory per update is set by the micro-batch shape, not the epoch count; one 10-epoch unit checks
  this (below).
- **The compile cache** is one root per (torch, commit), declared by the driver and reused only by its
  own units. This is the run-own-restart semantics; a cached kernel allocates exactly what a fresh
  one does. Startup took 150 + 314 s cold and 40–50 + 93–99 s warm.

## The units

| unit | allocator | diag cadence | updates | load1 start/end | allocated floor (post-update) | reserved after warm-up | peak reserved | segments after warm-up |
|---|---|---|---|---|---|---|---|---|
| `211824_default` | default | every update (the rollout index was not advanced; equivalent to `--diagnostics-every 1`) | 25 | 8.0 / 7.7 | 167.6–169.3 MiB (range 1.71) | +4 MiB @ 7 | 5,634 MiB | +2 |
| `220048_expandable` | `expandable_segments:True` | 10 | 50 | 13.5 / 10.3 | 165.9–166.0 (0.03) | +36 @ 10, +120 @ 20 | 5,540 | n/a (reads 0) |
| `223051_default` | default | 10 | 57 | 18.7 / 9.3 | 166.5–168.5 (2.03) | +16 @ 10 | 5,600 | +8 |
| `225409_expandable` | `expandable_segments:True` | 10 | 55 | 7.0 / 10.7 | 165.9–166.0 (0.03) | +36 @ 10, +120 @ 20 | 5,540 | n/a (reads 0) |
| `231653_default_e10` | default, **10 epochs** | 10 | 15 | 7.3 / 6.2 | 166.7–168.2 (1.5) | +16 @ 10 | 5,600 | +8 |

What every unit shows:

- **The startup step.** Allocated goes from 15 MiB at build, to 123–125 MiB after compile + prewarm,
  to 166–169 MiB after update 1. The last step is Adam's moment buffers, created lazily at the first
  optimizer step: the step `WARMUP_UPDATES = 2` absorbs. Reserved goes 3.7–3.9 GiB → 5.4–5.6 GiB at
  update 1 (the update's activations).
- **The post-rollout samples** sit about 91 MiB above post-update: the rollout block's last outputs
  are still referenced. The window FLOOR is the min, so it reads the post-update level.
- **Peak allocated per update** is 3,986–4,021 MiB on a plain update and 5,188–5,251 MiB on a
  diagnostics update. The noise-scale and grad-balance probes add 1.2 GiB.
- **Device free** is 5.4–5.6 GiB. The ceiling (reserved + free − 512 MiB) is about 10.5 GiB against a
  demand of 5.6 GiB, so the headroom is about 4.9 GiB.
- **Fault counters:** `num_alloc_retries` = 0 and `num_ooms` = 0 in every unit.
- **Detector replay:** every closed window reads OK (11 / 24 / 27 / 26 windows) and none reads
  SUSTAINED.

**The 10-epoch unit** (`231653_default_e10`: production epochs, default allocator, 15 updates, load1
6–7) reads the 2-epoch default unit's memory exactly:

- reserved 5,584 → 5,600 MiB, with the same +16 MiB step at update 10 and segments 173 → 181;
- peak allocated 4,013–4,019 MiB plain and 5,219–5,239 MiB on diagnostics updates;
- floor 166.7–168.2 MiB.

So the shortened epochs did not change what was allocated. Its update took 53.1–56.3 s plain and
60.9–62.2 s on diagnostics updates.

## Where the trainer process touches the card between updates (code-read, 2026-09-30)

**Python env core (today's default):**

- the rollout's own forwards: SB3 `collect_rollouts` → `policy(obs)` at `n_envs` rows, eval/no-grad,
  and `predict_values` at the rollout end (`async_vec_env.py:191,283` on the async path);
- the label callbacks' forwards on `model.device`: `win_prob_callback.py:460`,
  `winprob_pbrs.py:256,353`, `frozen_phi.py:201`, `cf_terms.py:125`;
- `fork_buffer.py:425` allocates a device rollout buffer for fork arms (off in production);
- the rank probe runs INSIDE `train()`;
- checkpoint saves (`model.save`) are device→host copies;
- **eval cycles do NOT touch the trainer's card.** Eval runs in `--eval-workers` subprocesses
  (`eval_callback.spawn_eval_workers` → `python -m main.eval_worker`) on `--eval-device` (default
  `cpu`). The trainer only saves the snapshot they load.
- Self-play and pool opponents run in the env-worker subprocesses on CPU.

**Rust env core (`--env-core rust`, the M5 target), which differs:**

- T2's `InferenceService` lives IN the trainer process, on the trainer's card. Its slots × buckets are
  CUDA graphs captured at startup, guarded by its own `cuda_segments_after_freeze`.
- The Rust eval cycle runs IN PROCESS and blocking (Lane H) on declared T2 eval slots. Sentinel
  snapshots load on CPU and are copied into the declared slots.
- So under the Rust core an eval cycle DOES run on the trainer's card. Its memory is declared at
  startup, and the trend detector sees it only as part of the ceiling's "free" reading.
- **This measurement does not include T2** (`--t2probe` below checks only allocator compatibility).

## The detector's constants, and why

| constant | value | from |
|---|---|---|
| `WINDOW_UPDATES` | 2 | 4 samples per window (post-rollout + post-update × 2). The floor = their min |
| `FIT_WINDOWS` / `SEGMENTS` | 9 / 3 | 18 updates of evidence. Odd groups of 3: a single step-up can raise at most ONE group median |
| `PERSIST_WINDOWS` | 3 | STOP needs the condition on 3 consecutive window closes. A ramp of L windows lines the medians up for about L closes (`test_short_fast_ramp_then_plateau_never_stops`) |
| `WARMUP_UPDATES` | 2 | update 1's Adam state, measured +43 MiB on the floor |
| `NOISE_BAR_BYTES` | 8 MiB | 4.4x the largest healthy floor range pooled over all units (1.83 MiB of window-floor residual; 2.03 MiB sample range) |
| `MIN_SLOPE_BYTES_PER_UPDATE` | 1 MiB | the bar spread over a group spacing (8 MiB / 6 updates ≈ 1.3 MiB) is the rule's real floor. Below it a leak reads OK (still logged as `lifecycle/cuda_floor_mib`) |
| `STEP_WARN_BYTES` | 64 MiB | a floor step 30x the healthy range: a retained allocation |
| `RESERVED_STEP_WARN_BYTES` | 256 MiB | 2.1x the largest healthy reserved step (+120 MiB, `expandable_segments`, at a diagnostics update) |
| `HORIZON_UPDATES` | 25 | the default periodic checkpoint (50,000 vec calls x 48 envs = 2.4M steps = 24.4 rollouts) |
| `CEILING_MARGIN_BYTES` | 512 MiB | kept free for the context, cuBLAS workspaces and a late block. It adds `margin / slope` updates of lead |

## False-trip estimate

**A deterministic bound, from the rule.** A STOP needs every consecutive group-median rise over the
span to exceed 8 MiB, on 3 consecutive window closes. A group median is one of the span's window
floors, so any rise is at most the floors' range. With healthy floors inside a 2.03 MiB band, no rise
can reach 8 MiB, and a false STOP (or even a SUSTAINED read) is impossible. That holds as long as the
healthy band stays under 8 MiB: 3.9x more wobble than any measured process.

**A simulation for wobble beyond the measured band.** `analyze.py` → `false_trip.sweep`:

- **Method:** synthetic 60-window (120-update, about 6 h of production) healthy processes. Each window
  floor is a constant plus a residual drawn i.i.d. from the pooled measured residuals, MAGNIFIED. The
  card is set TIGHT (1 GiB headroom).
- **Result** (2,000 processes per magnification): P(any SUSTAINED read) is 0 up to ×8, 0.05% at ×16,
  10.1% at ×32 and 27.7% at ×64 (pooled over all 5 learner units).
- **P(STOP) is 0 at every magnification.** A noise "slope" of a few MiB per update cannot project
  1 GiB of headroom away inside 25 updates.
- **What a false STOP would need:** a monotone floor drift of ≥ 1.3 MiB per update, held for 18+
  updates, on a nearly full card. That is no longer noise; it is the thing the detector exists to see.

**ASSUMED and UNVERIFIED:**

- a live run's floor wobble is like this harness's: same shapes, same compiled program, the same
  per-update data dependence through a real buffer;
- production's callbacks keep no growing device state, which was not measured with the callbacks
  attached;
- a trained checkpoint behaves like a fresh perturbed learner (weights never change what is
  allocated, only branches can).

## Detection lead, against a synthetic leak

Measured steady state (default allocator): floor 167.5 MiB, reserved 5,600 MiB, free 5,414 MiB. One
update is about 3 min in production.

| leak (MiB / update) | first SUSTAINED (from process start / from a mid-run onset at update 30) | first STOP | OOM | lead |
|---|---|---|---|---|
| 8 | 20 / 42 | 594 | 677 | 83 updates |
| 30 | 20 / 42 | 144 | 181 | 37 |
| 60 | 20 / 42 | 62 | 91 | 29 |
| 100 | 20 / 42 | 30 | 55 | 25 |
| 200 | 20 / 42 | 24 / 46 | 28 / 58 | 4 / 12 |
| 400 | — | — | 14 / 44 | **not caught**: OOMs before the 18-update span plus persistence fills |

**A slow leak is visible about 12 updates after it starts** (a SUSTAINED WARN carrying its projection).
It is stopped about `HORIZON + margin/slope` updates before the OOM. **A fast leak** (OOM within about
12–20 updates of onset, ≳ 300 MiB/update here) is out of scope: it crashes before the span fills, and
the launcher's restart policy governs that case.

## `expandable_segments:True` vs the default caching allocator (torch 2.8, same harness)

| | default (2 units) | `expandable_segments:True` (2 units) |
|---|---|---|
| peak reserved, steady | 5,600 / 5,634 MiB | 5,540 / 5,540 MiB (−60 MiB, −1%) |
| reserved growth after the warm-up | +4 MiB, +16 MiB (one step each) | +156 MiB (two steps: +36 @ 10, +120 @ 20; identical in both units) |
| inactive-split bytes (post-update) | 170–316 MiB | 0 |
| `segment.all.current` / `.allocated` | 173–182, stable after its first diagnostics update | **0 always: the counters are BLIND** |
| allocated floor range | 1.7 / 2.0 MiB | 0.03 / 0.03 MiB |
| plain update `train_s`, median [min–max] | 12.20 s [10.27–13.55] (n = 51, load1 18.7→9.3) | 13.48 [12.44–26.74] (n = 44, load1 13.5→10.3); 12.90 [9.94–20.89] (n = 49, load1 7.0→10.7) |
| diagnostics update `train_s` | 19.24 [17.41–20.07] (n = 5) | 21.24 / 21.54 (n = 5 each) |

**Update time: no ratio claimed.** The box carried other agents' work (load1 7–19 on 16 CPUs). Inside
ONE expandable unit the plain update moved from 9.9 s to 20.9 s with load, which is far more than any
between-arm difference. The arms ran as interleaved units (D, E, D, E), but each unit ran under
different contention. The fastest stretches (default 10.3–11.0 s, expandable 9.9–10.2 s) are not
separable either.

**Why not adopt it:**

1. Fragmentation under the default allocator is ALREADY a plateau: reserved stays flat for 47
   updates after one +16 MiB step, while the inactive split wobbles between 170 and 316 MiB with no
   trend.
2. The reserved saving is 1%.
3. It grows reserved more after the freeze, the opposite of a declared lifecycle.
4. Above all, it makes `memory_stats` report no segments, so the lifecycle's
   `cuda_segments_after_freeze` counters (T2's `InferenceService._frozen_guard`, the collector's and
   the eval executor's) read 0 forever. A gate that cannot fire is worse than no gate.

**T2 compatibility** (`driver.py run --t2probe`: T2's graph backend at a trainee + 4-slot pool spec,
buckets 8 / 48, compile + CUDA-graph capture + its startup parity gate, under each allocator config;
`result.json` → `t2probe`):

- **Both start cleanly:** 30 real-parity verdicts each, every after-freeze counter 0.
- **Default allocator:** 108 s cold; afterwards 154 MiB allocated, 416 MiB reserved, 86 segments.
- **`expandable_segments:True`:** 15 s on the now-warm cache, so the startup times are not a
  comparison. Afterwards 153 MiB allocated, 382 MiB reserved, and **0 segments**.
- So T2's graphs capture and replay under `expandable_segments`, but its
  `_frozen_guard`'s `cuda_segments_after_freeze` cannot see a new segment there.
