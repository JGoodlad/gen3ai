# X14 — overlap collection with the update: the build's measurements and the GPU A/B command (2026-10-19)

**What was built:** `--overlap-collect` (default OFF, `gen3_overlap_collect_v1`). The design and every hazard are in
[`../../../training/rust_collector.md`](../../../training/rust_collector.md) "Overlap (X14)". This folder holds what
could be measured WITHOUT the GPU (no lease when it was built; X34 held the card) and the A/B command for a GPU
window. Nothing here was run on the GPU, nothing was written under `models/` or `data/`. CPU work ran under rule 5b
(`nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh`, `CUDA_VISIBLE_DEVICES=""`).

## 1. The evidence the design rests on (read-only TensorBoard reads, `python -m main.ops.tb_read`)

| quantity | X26 (`rb_x26_s2601`, last 30 updates at ~380M) | X34 toy (`rb_toy_ent_s4002_deep`, last 20–30 at ~139M) |
|---|---|---|
| `train/train_ms` (update) | 42,159 | 9,229 |
| `rollout/collect_ms` | 28,812 | 10,501 |
| max hideable share of the cycle, min(update, collect) / (update + collect) | ≤ 41 % | ≤ 47 % |
| `staleness/probe_age_1_approx_kl` (one-version-old rows vs μ, at an update's start) | 0.0163 | — |
| `staleness/probe_age_1_clip_frac` (outside the 0.15 band before any step) | 0.225 | — |
| `staleness/rows_age_1_share` today (serial) | 0.076 | — |
| `train/approx_kl` (last epoch) / KL controller target | 0.0183 / 0.01 (factor 2) | — |
| `behaviour/probe_forward_ms` (the learner's eager fp32 forward, 2,048 rows) | 338 | 16.8 |
| `lifecycle/cuda_update_peak_reserved_mib` | 8,304 | 1,880 |
| policy state dict (from a checkpoint's `policy.pth`) | 11,670,103 values = 44.5 MiB (with the ride-along heads) | 3,121,590 = 11.9 MiB |

What follows from it:

- **Plain PPO on overlapped rows would be confounded.** One-version-old rows already start an update at KL 0.016 vs
  μ (above the controller's 0.01 target) with 22.5 % outside the clip band. With EVERY row that old, the KL→LR
  controller, which reads `train/approx_kl` against μ, would cut the rate for a lag it cannot remove. So the update
  under overlap is DECOUPLED PPO: the clip and the KL are centred on π_prox, the update's starting policy, and a
  truncated importance weight min(π_prox/μ, 2) multiplies the advantage.
- **π_prox comes from T2, not the learner.** The learner's eager forward costs 338 ms per 2,048 rows on X26, i.e.
  ≈ 16 s per 98,304-row update. T2's trainee slot holds exactly the learner's weights at each fill, and it is the
  function μ was served by. Its cost on the card is UNMEASURED (ESTIMATE 2–5 s per update; §2.1).
- **Memory.** Three policy copies (two snapshot slots + a swap scratch): 134 MiB at X26's policy and 36 MiB for the
  toy, against update peaks of 8,304 / 1,880 MiB on the 12 GB card. There is no second row buffer: the buffer under
  training and the arena under collection both exist in a serial run. **It fits.**

## 2. CPU mechanics (`cpu_mechanics.py` → `result/cpu_mechanics.json`)

The production policy (perturbed fresh) on the real env core, T2 eager on the CPU, 8 envs, 256-row updates,
micro-batch 128, 4 epochs, 6 rollouts; one arm per subprocess; arms alternated ABBA. **This measures the
MECHANICS (threads, the replay, the GIL), not the GPU gain**: on the CPU the worker's T2 forwards and the learner
share the same three agent cores, so the overlap mostly trades one CPU job for another.

### 2.1 Results (`read_cpu_mechanics.py` → `result/cpu_mechanics_summary.txt`; load average 17–19 on 16 threads, so CPU-STARVED)

Medians over the updates after the first two:

| arm | decisions / s | update (`train/train_ms`) | exposed collection (`rollout/collect_ms`) | worker play | join wait | π_prox pass (T2, eager CPU) | peak RSS |
|---|---|---|---|---|---|---|---|
| OFF rep 0 / rep 1 | 10.2 / 10.7 | 28.2 / 22.4 s | 14.3 / 15.9 s | — | — | — | 1,485 / 1,478 MiB |
| ON rep 0 / rep 1 | 7.9 / 9.0 | 24.2 / 22.8 s | 11.0 / 8.4 s | 14.2 / 10.6 s | 0.009 ms | 9.6 / 7.5 s | 1,572 / 1,587 MiB |

The ON arm's mechanics:
- importance weights mean 0.9997, normalised ESS 0.977;
- KL(μ → π_prox) 0.0112 per update of lag;
- `train/approx_kl`, now centred on π_prox, 0.0134;
- the snapshot judge's max |Δ log π| on judged rows 6.0e-7;
- the two ON reps are BIT-IDENTICAL in every learning read (deterministic).

What it says:
- **The mechanics work.** The worker finished inside every update (join wait ≈ 0), so the whole play was hidden. The
  replay, the fill and K9(b) under the snapshot were exact.
- **On the CPU, overlap is SLOWER per decision (−15 %).** Two reasons: the "GPU" work (T2's forwards, the update)
  runs on the same three CPU cores, so nothing is really freed; and the π_prox pass is one more trainee forward over
  the whole buffer, which on an eager CPU T2 costs about as much as the rollout's own trainee inference. **This is
  NOT a proxy for the card.**
- **The π_prox pass is the overlap's own exposed cost.** It is one T2 forward over the 98,304 rows, in the exposed
  half of the cycle. On the GPU it is UNMEASURED: ESTIMATE 2–5 s per update (the trainee's share of X26's 16.7 s of
  per-rollout inference wait), against ≤ 29 s hidden. The A/B reads it as `overlap/prox_forward_ms`.
- **Memory.** +90–110 MiB of peak RSS on the CPU (76 MiB of it the three policy copies at this fixture's policy).
  No second row buffer.

## 3. The GPU A/B — WRITTEN, NOT RUN

### 3.1 The throughput read (≈ 1 GPU-h; first)

Two forks of X26's 380M checkpoint at the X14 commit, 15 updates each (≈ 10 min startup + ≈ 18 / 14 min of
updates), eval and ladder games off, K9(b) `fatal`.
The training session (or the agent holding the lease) runs them one after the other, after X34 releases the card.
The runs land in a scratch archive, not `models/`.

```bash
cd /home/goodlad/dev/gen3ai && export PYTHONPATH=$PYTHONPATH:src
export GEN3AI_MODELS_DIR=$HOME/gen3ai_archive/x14_ab_runs              # scratch: not main's models/
X14=<the X14 commit on main>
CKPT=/home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_380000195_steps.zip
STEPS=$((380000195 + 15 * 98304))                                     # ABSOLUTE target: 15 updates past the parent
for ARM in off on; do
  ARGV="--model $CKPT --steps $STEPS --device cuda --run-name rb_x14ab_${ARM}_s2601 --pin-commit $X14 \
        --restart-interval-hours 24 --eval-freq 1000000000 --snapshot-ladder-games 0 --behaviour-check fatal \
        $([ $ARM = on ] && echo --overlap-collect)"
  python -m main.checkargs --argv "$ARGV" && python -m main.launcher $ARGV    # (the launcher takes the GPU lock)
done
```

Read per arm (`python -m main.ops.tb_read <run> --tag …`), on the updates after the 3rd (the ON arm's first
update in the process runs alone by design):

- `time/fps` and the cycle (`train/train_ms` + `rollout/collect_ms`): the gain is fps(on) / fps(off). Bar for
  "worth a learning A/B": ≥ ×1.10.
- On the ON arm: `overlap/play_ms` (did the worker finish inside the update?), `overlap/join_wait_ms` (the exposed
  collection), `overlap/prox_forward_ms` (T2's π_prox pass), `train/train_ms` vs the OFF arm's (how much the
  concurrent inference slowed the update), `overlap/is_ess`, `overlap/is_weight_truncated_frac`,
  `behaviour_snapshot/max_abs_dlogp_judged` (must stay < 1e-4) and `behaviour_snapshot/excluded_frac`.
- Both: `lifecycle/cuda_update_peak_reserved_mib` (the fit), and the child log for `[T2] WARN … grew the CUDA
  reservation` (the learner's allocations seen inside a worker flush — counted, not fatal).

### 3.2 The learning read (the X14 row's registered read; only if 3.1 clears its bar)

Fresh `--arch production` with and without `--overlap-collect`, read at MATCHED GPU WALL, like X44 (OFF to 15M ≈
2.6 h; ON gets the same wall), 2 seeds per arm as a pilot (≈ 10 GPU-h), then 8 per arm if it is not a clear loss.
Readout: strength per GPU-hour against a frozen reference; `overlap/is_ess` and the `staleness/*` lag histogram as
the mechanism reads. The X14 row's own condition still holds: put the measured idle budget (§3.1's
`overlap/join_wait_ms` and the OFF arm's collect / update split) beside the filler-eval alternative before
adopting.

## 4. FINDINGS

1. **The brief's premise held only for the GRADIENT.** PPO's ratio π/μ keeps the policy gradient unbiased for any
   behaviour policy, but production's clip AND its KL→LR controller are centred on μ. At X26's measured one-version
   lag, plain overlap would have cut the learning rate. Decoupled PPO was built for that reason.
2. **`normalize_advantage` normalises w·A per micro-batch, not A** (the rewrite feeds the unchanged micro-step). At
   w ≈ 1 ± 0.1 that is a small rescale plus a baseline shift. Unmeasured.
3. **No V-trace.** The GAE advantages of an overlapped rollout use μ's trajectories and T2's version-k value. That is
   one version of lag, the same as today's 7.6 % age-1 rows, but now it applies to ~100 % of rows. Unmeasured.
4. **CUDA concurrency is UNVERIFIED.** The worker runs T2 on its own stream, and T2's frozen guard reads
   process-wide allocator counters. The first update of each process runs alone, and the launcher sets
   `expandable_segments`. A run WITHOUT `expandable_segments` that grows a segment during a worker flush would
   POISON T2.
5. **The GIL is shared.** The update's Python (R1 calls, the eager tail, the deferred reads) and the host loop
   contend for it. On the card the update mostly waits on the GPU with the GIL released. Unmeasured.
6. **Callback side effects land one rollout later under overlap** (a pool push, an opponent refresh, a team pull).
   The callbacks themselves fire at the same `num_timesteps` with the same learner weights.
7. **`rollout/collect_ms` means the EXPOSED collection on an ON run.** A tool that reads it as the whole collection
   (the cloud bench's `LIVE_CYCLE` calibration, killbar, the iteration split) misreads an ON run.
8. **A fork does not inherit the flag.** A launcher restart replays the typed argv; a fork must type it.
9. **Tiny-fixture K9(b) trips.** On 8 envs × 32 rows, one tie-heavy game made 40 % of a probe's rows ties, under the
   snapshot AND under the learner's weights on the same rows (`excluded_frac` 0.402 vs the 0.15 ceiling). This is a
   fixture property, not the overlap's; production's probe samples 2,048 of 98,304 rows.
10. **The X14 row's precondition was not met by a measurement.** The row asks for the idle-phase budget to be weighed
    against filler eval before either is built. The build went ahead on the orchestrator's brief. §3.1 measures that
    budget.
11. **The GPU-window battery's X14 slot stays EMPTY.** `gpu_window_battery_2026-10-19/` cell 12 expects an
    `x14_cell.sh` hook (30 min ETA, 60 min cap). The two-fork read above needs ≈ 1 h with two cold startups, so it does
    not fit. A battery-shaped cell would be one in-process run with the overlap toggled per iteration (ABBA). It needs
    a small driver seam (an `enabled` switch on the runner) and was NOT built: out of this unit's scope.
12. **Arming T2's reserved-growth poison would make overlap FATAL.** T2's frozen guard reads process-wide allocator
    counters. A learner reservation growth that lands inside a worker flush is booked as T2's own: today it is
    counted and warned (`cuda_reserved_growths_after_freeze`, 847efe34); once `reserved_growth.RESERVED_GROWTH_POISONS`
    is armed (battery cell 2 is its arming test), it is a `LifecycleViolation`. Before arming it, attribute the growth
    per stream or per thread, or keep `--overlap-collect` off.
