# The cloud GPU ladder — 2026-10-18 (T13, the GPU-upgrade decision)

**What this is.** The frozen GPU benchmark suite (`python -m agents.training.gpu_bench_suite`, suite version 1,
schema `gen3_gpu_bench_suite_v1`) run on rented vast.ai cards, one box at a time, at the pinned public commit
`97616f5e7d208703df9ed9c195428d0ac7d02f18`. One JSON per box is in [`suites/`](suites/) (with that box's torch
sanity and its setup step times). The table below is GENERATED from those JSONs:

```bash
export PYTHONPATH=$PWD/src M=designs/research_state/measurements/gpu_bench_2026-10-18/suites
python -m agents.training.gpu_bench_suite compare $M/rtx3090_hu.json $M/rtx3090_ca.json $M/rtx4090_il.json \
    $M/rtx5090_mx.json $M/rtx5090_nl.json $M/rtx5080_us.json $M/rtx5070ti_fr.json $M/a100sxm4_us.json \
    --reference "RTX 3090 / HU (cu126) control" --markdown
```

The runbook, the recommendation and the buying notes are in
[`designs/ops/cloud_gpu_benchmark.md`](../../../ops/cloud_gpu_benchmark.md) §0. This file is the record.

## What each column is

Every update number is the learner's real `train()` at the production shape: 98,304 rows, micro-batch 2,048 x
accumulation 32, 10 epochs, on the production end-state network with the ride-along heads (X26's training
surface). It is compiled with the startup parity gate (PASS on every cell) and run on the K9 golden's real rows,
tiled. `update s` = `train.update_s_est`, the `train/train_ms` equivalent of a plain update, estimated from timed 1-
and 3-epoch calls. `env steps/s` = the median of 3 timed rollouts through the real trainer: 256 envs on the Rust env
core, the trainee plus a 20-snapshot fixture pool at self-play fraction 0.90, all served by T2. `4,096 x 16` /
`8,192 x 8` = the same update at larger micro-batches with the same effective batch (the VRAM question). Every
cycle column is the Amdahl estimate in §"The cycle estimate".

## The table

| card | build | power limit W | peak SM MHz | update bf16 s | update fp32 s | env steps/s | peak bf16 GiB | VRAM GiB | headroom GiB | 4,096 x 16 | 8,192 x 8 | $/h | update speedup vs ref | cycle s (est) | cycle speedup (est) | cycle speedup, update only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX 3090 / HU (cu126) control | 2.8.0+cu126 | 390 | 1995 | 40.02 | 51.19 | 3456 | 6.1 | 24 | 17.4 | 37.8 s @ 12.1 GiB | 37.1 s @ 23.1 GiB | 0.336 | 1.00x | 74.8 | 1.00x | 1.00x |
| RTX 3090 / CA (cu126) | 2.8.0+cu126 | 330 | 1950 | 40.13 | 54.39 | 3228 | 6.1 | 24 | 17.4 | 39.1 s @ 12.1 GiB | 37.8 s @ 23.1 GiB | 0.242 | 1.00x | 75.3 | 0.99x | 1.00x |
| RTX 4090 / IL (cu126) | 2.8.0+cu126 | 500 | 2865 | 27.24 | 33.21 | 5250 | 6.2 | 24 | 17.3 | 26.8 s @ 12.1 GiB | 26.8 s @ 22.9 GiB | 0.656 | 1.47x | 55.5 | 1.35x | 1.22x |
| RTX 5090 / MX (cu128) replicate | 2.8.0+cu128 | 600 | 2947 | 18.64 | 25.04 | 5764 | 6.2 | 31 | 25.2 | 19.3 s @ 12.0 GiB | 20.2 s @ 23.7 GiB | 0.736 | 2.15x | 44.0 | 1.70x | 1.43x |
| RTX 5090 / NL (cu128) | 2.8.0+cu128 | 420 | 2977 | 20.44 | 25.21 | 5838 | 6.1 | 31 | 25.2 | 19.7 s @ 12.0 GiB | 20.1 s @ 23.5 GiB | 0.944 | 1.96x | 45.8 | 1.63x | 1.38x |
| RTX 5080 / US (cu128) | 2.8.0+cu128 | 300 | 2910 | 31.49 | 38.02 | 5510 | 6.1 | 16 | 9.4 | 31.6 s @ 12.0 GiB | OOM | 0.344 | 1.27x | 60.1 | 1.24x | 1.14x |
| RTX 5070 Ti / FR (cu128) | 2.8.0+cu128 | 250 | 2857 | 35.40 | 42.83 | 4599 | 6.1 | 15 | 9.3 | 35.1 s @ 12.1 GiB | OOM | 0.407 | 1.13x | 65.6 | 1.14x | 1.07x |
| A100 SXM4 / US (cu126) | 2.8.0+cu126 | 380 | 1410 | 47.81 | 47.91 | — | 6.2 | 39 | 33.2 | — | — | 0.656 | 0.84x | — | — | — |

**The local RTX 3080 Ti row is PENDING.** X26 holds the GPU lease; the orchestrator runs this same suite locally in
the GPU gap after X26 ends. Until then the RTX 3090 stands in for it as the same-architecture proxy (GA102, sm_86,
936 vs 912 GB/s). The two 3090s agree with the 3080 Ti's last local reading:

| source | bf16 update | fp32 update |
|---|---|---|
| RTX 3080 Ti, local, `cuda_build_2026-10-12` (E config) | 41.2-41.3 s | 53.0-53.5 s |
| X26 live `train/train_ms` (median of the last 20, bf16) | 42.2 s | — |
| RTX 3090, this ladder (two boxes) | 40.0 / 40.1 s | 51.2 / 54.4 s |

## The cycle estimate (Amdahl), with the arithmetic

X26's live cycle on the 3080 Ti (`main.ops.tb_read rb_x26_s2601 --last 20`, at 258.6M steps): train 42.24 s, collect
26.97 s, everything else 5.6 s (98,304 / `time/fps` 1,314 = 74.8 s per cycle). Of the collect, 37.39 ms per host step x
416.5 host steps = 15.57 s waits on T2 (`rust_env/gpu_wait_ms_per_host_step`). The other 11.40 s is the host CPU's
(Rust core, submit, flush) and stays the same on any card. For a card `c` against the reference `r` (the 3090 on the
Zen 4 host, standing in for our 3080 Ti):

    train_c   = 42.24 x U_c / U_r                         U = bf16 update s
    collect_c = 11.40 + 15.57 x W_c / W_r                 W = GPU wait ms per host step (collect phase)
    cycle_c   = train_c + collect_c + 5.6                 speedup = 74.8 / cycle_c

For example, the 5090 at 600 W: train 42.24 x 18.64 / 40.02 = 19.67 s, collect 11.40 + 15.57 x 18.81 / 40.05 = 18.71 s,
cycle 19.67 + 18.71 + 5.6 = 43.98 s, so 74.8 / 43.98 = 1.70x (2,235 env steps/s against today's 1,314). The "update only" column holds collection at 26.97 s, which is the floor. A control isolates the host
CPU's share of `W`: the same card (3090) on a Zen 3 host (Ryzen 7 5700X) and on a Zen 4 host (Ryzen 9 7900X)
waits 41.0 vs 40.0 ms per host step. So `W` is mostly the card. Small-batch inference rewards clocks more than the
update does: the 5070 Ti waits 1.39x less than the 3090 but updates only 1.13x faster.

## The VRAM question: does 16-32 GB unlock anything?

**Not at today's shape.** The production update peaks at 6.1-6.2 GiB reserved on every card (bf16; fp32 6.7-7.1). A
larger micro-batch at the same effective batch saves at most a few percent: 3090 −7 % at 8,192, 4090 −2 %, the
420 W 5090 −4 %, and the 600 W 5090 gets SLOWER (+3.5 % at 4,096, +8 % at 8,192). It needs 12.0-12.4 GiB at 4,096, which does NOT fit the 12 GB 3080 Ti, and
22.9-24.3 GiB at 8,192 (a 24 GB card only). It OOMs on the 16 GB cards. More VRAM matters only if the model, the
batch or the env count grows. The n-envs point (more than 256 envs) was NOT measured: collection's CPU half is the
host's, and a rented host is not ours.

## Sessions and money

Credit **9.759 → 6.217** (read 15:29 PT, after the last destroy settled): **$3.54 spent**, $6.22 left (no auto
top-up; the API key was never printed). Cost per session =
the credit read at this session's create minus the credit at the next create (billing lags a few minutes).

| # | card | instance | host CPU | where | listed $/h | wall | cost | outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | RTX 3090 | 56582634 | Ryzen 7 5700X (Zen 3) | CA | 0.242 | 57 min | $0.256 | OK (setup re-run once: abbreviated SHA, FINDING) |
| 2 | RTX 4090 | 56590769 | Ryzen 9 7950X | IL | 0.656 | 54 min | $0.617 | OK (torch sanity re-run once: the sm_89 check, FINDING) |
| 3 | RTX 5090 | 56597764 | Ryzen 9 9950X3D | NL | 0.944 | 41 min | $0.802 | OK, but power-capped at 420 W |
| 4 | RTX 5070 Ti | 56603234 | EPYC 7K62 | US-CA | 0.406 | 10 min | $0.053 | MISBEHAVED: image pull stuck >10 min, destroyed, machine excluded |
| 5 | RTX 5070 Ti | 56604645 | Threadripper 9970X | FR | 0.407 | 41 min | $0.342 | OK |
| 6 | A100 SXM4 40 GB | 56609402 | EPYC 7532 (Zen 2) | US-MA | 0.656 | 31 min | $0.343 | STOPPED after the two update phases (FINDING) |
| 7 | RTX 5080 | 56613372 | Ryzen 9 9950X | US-FL | 0.344 | 41 min | $0.329 | OK |
| 8 | RTX 3090 (control) | 56617845 | Ryzen 9 7900X (Zen 4) | HU | 0.336 | 48 min | $0.286 | OK — the reference row |
| 9 | RTX 5090 (replicate) | 56623714 | Ryzen 9 9900X | MX | 0.736 | 36 min | $0.514 | OK, at 600 W |

After every session: `destroy`, then `verify-zero` (0 instances), and every deadman exited with "gone already".

## Hazards of this read (each a FINDING in the runbook)

- **Rented cards run at the HOST's power limit, not the board's.** The first 5090 was capped at 420 W (board default
  575-600 W) and was 9 % slower on the bf16 update than a 600 W one (20.44 vs 18.64 s; fp32 equal). The 5080 ran at
  300 W (stock 360 W) and the 5070 Ti at 250 W (stock 300 W), so a bought stock card may be FASTER than its row
  here (**UNVERIFIED** by how much). The suite now also records `power.default_limit`.
- **fp32 differs 6 % between the two 3090s** (54.4 vs 51.2 s; power limits 330 vs 390 W) while bf16 agrees to
  0.3 %. Quote the bf16 ratios.
- **The collect plays UNTRAINED fixture policies.** The games are deterministic: every box played the same rollouts
  (110,949 / 97,701 / 98,112 / 99,253 env steps), so the work is identical across cards. Their mix is not a trained
  run's (405 host steps per rollout vs X26's 416).
- **The A100 is slower than a 3090 on this update** (0.84x). It is consistent with the 2026-09-28 profile (many
  small, memory-latency-bound kernels: clocks matter, the A100 boosts to 1.41 GHz vs ~2.0-3.0 GHz). Its host was a Zen 2
  EPYC taking ~11 min per compile, so the run was stopped after its two production update phases to stay inside the
  90-min deadman. Its headroom and collect phases were not run.
- **T2's cold startup costs 10.7-16.8 min per box** (640-1006 s, the inference service's compile and capture of 32
  slots x 3 buckets). It is the largest fixed cost of a rental and is not part of any number above.
