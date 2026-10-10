# T25 PERFORMANCE PHASE on the END-STATE graph: the plan (2026-10-10)

**Scope (owner 2026-10-09).** "Keeping the GPU busy since it's our bottleneck is a great first prioritization …
performance should only optimise the end state that we desire to keep." Target `--arch endstate`, and production only
where a win is generic. No perf work on the legacy encoding.

**Rule.** Every lever sits behind a switch, default OFF. Each needs a PAIRED before / after benchmark and the
compiled-learner gates green before adoption: the R1 startup parity gate and the compile canary. bf16 keeps every
DECISION in fp32: the policy logits and softmax, the value, and the losses. Authority comes from
`memory/project_perf_levers_authority.md`. The orchestrator decides the levers below. The OWNER decides hand-written
GPU kernels and a shared compile cache.

Tags: **MEASURED** / **ESTIMATE** / **UNVERIFIED**.

## 0. What the cycle costs now

| quantity | value | source |
|---|---|---|
| E (`--arch endstate`) `train_ms` per update | ~53.8 s | the precondition-E launch at `95d014fa` (closing test Decision record), MEASURED, descriptive |
| R (`--arch static_recovery`) / P (`--arch production`) `train_ms` | 51.59 s / 44.95 s | `gpu_checks_endstate_2026-10-09`, MEASURED, quiet updates only |
| rollout per cycle (P, N = 256) | ~13 s (T2 inference ~70 % of the host step, the Rust core ~24 %) | `bottleneck_profile_2026-10-03` (P at `336ddd27`), MEASURED |
| the update's GPU busy share (P) | 92 %: GEMM 35 %, Triton pointwise + reduction 47 %, attention 14 % of kernel time | the same profile, MEASURED; **the end-state graph has never been profiled** (window A) |
| cold startup per process | T2 build 583–847 s + R1 reset / prewarm 184–240 s | `gpu_checks_endstate_2026-10-09`, MEASURED |

So the UPDATE is ~80 % of an end-state cycle, and the levers that move it come first.

**Where the end state's matmul FLOPs live** (CPU, one eager R1 micro-step, forward + backward, 256 rows;
`scripts/cpu_flops.py`, `scripts/module_flops.py`, `results/cpu_flops_*.json`, MEASURED, analytic counts):

| config | GFLOP / 256 rows | vs P |
|---|---|---|
| P `--arch production` | 44.8 | — |
| S + `--token-encoding static` | 44.4 | −0.8 % |
| SF `--arch static_recovery --trunk-layers 2` | 44.5 | −0.8 % |
| R `--arch static_recovery` (the third trunk round) | 59.0 | **+31.6 %** |
| E `--arch endstate` | 63.2 | +41.0 % |
| EK E with `--ko-ramp ramp` | 63.0 | E − 0.3 % |

The three trunk rounds (`team_transformer`: two post-LN `BiasedEncoderLayer`s and one pre-LN `IdentityInitRound`)
carry **~70 %** of E's matmul FLOPs. The FlopCounter's module attribution is approximate for the backward. Exact KO
adds no matmul FLOPs. It is element-wise, and its eager CPU wall was 0.68 s for E vs 0.56 s for EK (one sample,
DESCRIPTIVE). Its GPU cost is what window A's E vs EK reads.

## 1. The levers, ranked by expected wall-clock gain on the end-state cycle

| rank | lever | expected gain | risk | built? | how it is measured |
|---|---|---|---|---|---|
| 1 | **A. bf16 trunk region** (`agents/model/trunk_precision.py`, `gen3_trunk_bf16_region_v1`): the trunk rounds under bf16 autocast, fp32 out. Everything else stays fp32: the damage op and its edges, every head, the pointer head, the masked log-softmax, the value, the losses | GEMMs ~35–40 % of update kernel time, of which the trunk is ~70 % of FLOPs; bf16 tensor cores at 2–3× on those, plus bf16 attention: update **−6 to −12 s** (ESTIMATE). It pays in full only once the host is not the wall (lever C) | **HIGH**, numerics: compiled bf16 vs eager bf16 differs by bf16 rounding, far above R1's fp32 bar; the rollout's T2 forward is fp32, so epoch 0's ratio is no longer 1 to 1e-6 and K9(b)'s bar must be re-derived; needs its own strength A/B | YES (measurement lever, no trainer flag) | window B `E_bf16` vs `E`: update time, the gate's per-parameter error (recorded even when it refuses), the canary |
| 2 | **C. CUDA graphs over R1** (`R1_INDUCTOR_PRESETS["cudagraphs"]`: cudagraph trees over R1's forward + backward) | the update's host bubbles were ~8 % of the stream (P profile) minus perf item 1's sync cut: **−2 to −4 s** (ESTIMATE); the PREREQUISITE for lever A's full gain (the 2026-10-03 profile: the host becomes the wall at ~1.4× faster kernels) | MEDIUM: the graph pool's memory vs E's 2,058 MiB UpdateFit headroom; the retain_graph probes (grad balance, noise scale) through a graphed backward; output lifetime across micro-steps (cudagraph trees raise on a stale read, loud, never silent) | YES (measurement preset) | window B `E_cudagraphs`: time, peak reserved, gate + canary |
| 3 | **B1. Inductor coordinate-descent tuning** (`coordesc`) | Triton pointwise + reduction ~47 % of kernel time; tile search typically buys 3–10 % on those: **−1 to −3 s** (ESTIMATE) | LOW: compile time grows (benchmarks every kernel at startup); an fp32 reduction order may move, within what the gate and canary judge | YES (preset) | window B `E_coordesc` |
| 4 | **Closed-form exact KO** (owner 2026-10-10: "optimise the 16 rolls"; `--ko-ramp exact_closed`) | ranked by window A's E vs EK cost. The op's damage is continuous (no floors), so Σ over 16 rolls of a clamped linear ramp is an EXACT closed form (count + arithmetic series), O(1) per cell instead of O(32). On the GPU the 16-term loop is already ONE fused kernel, so the gain is bytes and ALU, and possibly small | LOW–MEDIUM: a count needs a floor of a score (continuous sum, so tie-safe, but it must be DECLARED in `selection_sites`); the closing test pins `exact`, untouched | IN BUILD (sub-agent, CPU) | window B `E_koclosed` vs `E` vs `EK` |
| 5 | **B2. Horizontal fusion** (`combo`: Inductor combo kernels) | fewer small launches: **≤ −1 s** (ESTIMATE) | LOW | YES (preset) | window B `E_combo` |
| 6 | **T2 fan-out**: one grouped forward or one multi-slot graph per host step instead of ~21 | T2 was ~16 % of a P cycle, latency-bound: **−3 to −5 s / cycle** (ESTIMATE, 2026-10-03) | MEDIUM–HIGH: `vmap` is blocked by the extractor's in-place writes, and a grouped forward is a model-code rewrite | no | after the levers above; window A's T2 bucket-256 times price the per-arch cost |
| 7 | **T26: overlap the next rollout with the update** | up to 18–30 % (UNVERIFIED) | changes on-policyness (rows one version stale): its OWN strength A/B | no (T26) | its own item |
| 8 | **Compile-cache by STAMP** (commit + torch + config hash + interpreter) | startup 13–18 min per process: at the 6 h restart interval, **~4–5 % of wall clock** (ESTIMATE from MEASURED startup times) | relaxes the owner's 09-29 "fresh setup" rule | no | **OWNER decision** (§3) |
| 9 | **Hand-fused damage-op physics kernel** | the op's pointwise share of the Triton time (window A's profile prices it) | a large maintenance surface | no | **OWNER decision** (§3) |

**Not worth it** (2026-10-03 profile, unchanged by the end state): the Rust core (~5.5 %), host glue (~2 %), eval
(~1.8 %), the optimizer (fused, ~0), the staged H2D copies (overlapped).

**Why the levers have no trainer flag yet.** Each is a SWITCH on the learner: `trunk_precision.set_trunk_precision`
and `model._r1_inductor_preset`. The benchmark sets them, and the production path cannot reach them, so nothing is
adoptable by accident. A lever that wins its paired benchmark AND its gates gets its CLI flag, flag-registry row and
recipe surface in its adoption commit. bf16 also needs §2's design.

## 2. bf16 adoption design (for lever A, if window B shows the gain)

1. **R1 parity for a bf16 region: the MATCHED-ENVELOPE rule** (`gpu_checks_endstate` F-GE-3's suggestion). Compiled
   bf16's per-parameter gradient error against a float64 reference must sit within k× of EAGER bf16's own error
   against the same reference. This replaces "compiled vs eager < the fp32 bar". It costs one fp64 arm at startup.
2. **K9(b) under a bf16 learner.** Either (a) T2 serves the trainee slot through the same bf16 region, so behaviour
   and learner share the numerics class (still not bit-equal), with K9(b)'s bar re-derived from the measured
   T2-vs-learner bf16 spread; or (b) the learner stays fp32 for K9(b)'s probe forward only. (a) is the honest one.
3. **The strength A/B**: E fp32 vs E bf16, matched seeds and budget, on the registered meter. A numerics change is
   never adopted on speed alone.

## 3. What needs the OWNER

- The **compile-cache strategy** (lever 8). The evidence is the measured cold startup above and its share of wall
  clock at the 6 h restart. Window A records each config's compile + gate + prewarm time on a quiet box.
- Any **hand-written kernel** (lever 9). It goes to the owner as a proposal only if the end-state profile shows a
  fusable hot spot that Inductor's own levers (B1, B2) do not close.

## 4. Measurement protocol

`scripts/update_bench.py`: the real `train()` of a learner built for a trainer argv at the production shape, compiled
by the trainer's own `arm_compile_sentinel` (gate, prewarm, lock), on the update-fit fixture rollout (the K9 golden's
real rows tiled to 98,304). A fresh fixture is used per call. The schedule is one warm call, then 2 × 1-epoch and
2 × 3-epoch calls. The update estimate is median(1 ep) + 9 × the marginal epoch. Optional extras: a torch.profiler
epoch, the canary on the trained weights, and T2's single-slot flush per bucket. `scripts/configs.py` holds every
argv (with the closing test's ride-along flags). `scripts/window.sh` runs a window in priority order under the lease,
never starting a job it cannot finish.

**Calibration:** the bench's E / R / P update estimates against the real launches' 53.8 / 51.59 / 44.95 s. **Caveat:**
the fixture is 64 real rows tiled. The compiled graph is branch-free, but the eager tail's masked selects scale with
label presence. Compare configurations within the tool; quote absolute seconds only beside the calibration.

- **Window A (approved, opens ~08:55):** E (+ T2 bucket 256 + profile), P (+ T2), R (+ T2), EK, SF, S.
- **Window B (after A is read):**
  - E_bf16, E_coordesc and E_cudagraphs, each with its canary;
  - perf item 1's deferred A/B: `957d4dbd^` vs `957d4dbd` on production, with profiles (busy %, syncs per
    micro-batch, scalar-read time);
  - E_koclosed and E_combo.
