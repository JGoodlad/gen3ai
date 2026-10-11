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
| 1 | **A. bf16 trunk region** (`agents/model/trunk_precision.py`, `gen3_trunk_bf16_region_v1`): the trunk rounds under bf16 autocast, fp32 out. Everything else stays fp32: the damage op and its edges, every head, the pointer head, the masked log-softmax, the value, the losses | GEMMs ~35–40 % of update kernel time, of which the trunk is ~70 % of FLOPs; bf16 tensor cores at 2–3× on those, plus bf16 attention: update **−6 to −12 s** (ESTIMATE). It pays in full only once the host is not the wall (lever C) | **HIGH**, numerics: compiled bf16 vs eager bf16 differs by bf16 rounding, far above R1's fp32 bar; the rollout's T2 forward is fp32, so epoch 0's ratio is no longer 1 to 1e-6 and K9(b)'s bar must be re-derived; needs its own strength A/B | YES: `--trunk-precision bf16` (config v155, default fp32) + the bf16 parity rule with DERIVED bars (§2) | window B `E_bf16` vs `E`: update time, the gate's per-parameter error (recorded even when it refuses), the canary |
| 2 | **C. CUDA graphs over R1** (`R1_INDUCTOR_PRESETS["cudagraphs"]`: cudagraph trees over R1's forward + backward) | the update's host bubbles were ~8 % of the stream (P profile) minus perf item 1's sync cut: **−2 to −4 s** (ESTIMATE); the PREREQUISITE for lever A's full gain (the 2026-10-03 profile: the host becomes the wall at ~1.4× faster kernels) | MEDIUM: the graph pool's memory vs E's 2,058 MiB UpdateFit headroom; the retain_graph probes (grad balance, noise scale) through a graphed backward; output lifetime across micro-steps (cudagraph trees raise on a stale read, loud, never silent) | YES: the trainer flag `--learner-cudagraphs on` (default off, `gen3_r1_graphed_route_v1`): R1's GRAPHED route serves every micro-step no retain_graph probe re-enters, the compiled route the probe micro-steps; both gated (+ an accumulation rung), prewarmed, inventoried, canaried (`designs/training/compile_flags.md` "Lever C"). The bare `cudagraphs` preset stays the F-PB-2 reproduction. GPU NOT measured | window B `E_cudagraphs`: time, peak reserved, gate + canary |
| 3 | **B1. Inductor coordinate-descent tuning** (`coordesc`) | Triton pointwise + reduction ~47 % of kernel time; tile search typically buys 3–10 % on those: **−1 to −3 s** (ESTIMATE) | LOW: compile time grows (benchmarks every kernel at startup); an fp32 reduction order may move, within what the gate and canary judge | YES (preset) | window B `E_coordesc` |
| 4 | **Closed-form exact KO** (owner 2026-10-10: "optimise the 16 rolls"; then "remove the old one") | ranked by window A's E vs EK cost. The op's damage is continuous (no floors), so Σ over 16 rolls of a clamped linear ramp has an EXACT closed form (count + arithmetic series): O(1) per cell instead of O(32). On the GPU the 16-term loop is already ONE fused kernel, so the gain is bytes and ALU, and possibly small | LOW: the counts are a floor / ceil of a score, but the sum is continuous across them, so it is tie-safe and DECLARED (`COUNT_CONTINUOUS`); the closing test is pinned at `b132b099` (the 16-roll sum) | **LANDED as THE `--ko-ramp exact`** (`6e0a1a7a` built it as `exact_closed`; the follow-up made it the only implementation and RETIRED `exact_closed`; the 16-roll sum is the test oracle; CPU KO function 23.9 → 6.1 ms) | window A's `E` is pinned at `6e0a1a7a` (the 16-roll sum) and `EK` (ramp); window B's `E_base` at HEAD is the closed form |
| 5 | **B2. Horizontal fusion** (`combo`: Inductor combo kernels) | fewer small launches: **≤ −1 s** (ESTIMATE) | LOW | YES (preset) | window B `E_combo` |
| 6 | **T2 fan-out**: one grouped forward or one multi-slot graph per host step instead of ~21 | T2 was ~16 % of a P cycle, latency-bound: **−3 to −5 s / cycle** (ESTIMATE, 2026-10-03) | MEDIUM–HIGH: `vmap` is blocked by the extractor's in-place writes, and a grouped forward is a model-code rewrite | **PART: LEAD LANES built** behind `--t2-lead-trainee` (default OFF, `gen3_t2_lead_lanes_v1`: the trainee's replay launched first on its own lane; bit-identical; ESTIMATE −0.4 to −1.6 s; paired read `lead_lanes_benchmark.py`). The grouped forward: NOT built (the model-wide rewrite) | after the levers above; window A's T2 bucket-256 times price the per-arch cost |
| 7 | **T26: overlap the next rollout with the update** | up to 18–30 % (UNVERIFIED) | changes on-policyness (rows one version stale): its OWN strength A/B | no (T26) | its own item |
| 8 | **Compile-cache by STAMP** (commit + torch + config hash + interpreter) | startup 13–18 min per process: at the 6 h restart interval, **~4–5 % of wall clock** (ESTIMATE from MEASURED startup times) | relaxes the owner's 09-29 "fresh setup" rule | no | **OWNER decision** (§3) |
| 9 | **Hand-fused damage-op physics kernel** | the op's pointwise share of the Triton time (window A's profile prices it) | a large maintenance surface | no | **OWNER decision** (§3) |

**Not worth it** (2026-10-03 profile, unchanged by the end state): the Rust core (~5.5 %), host glue (~2 %), eval
(~1.8 %), the optimizer (fused, ~0), the staged H2D copies (overlapped).

**Which levers have a trainer flag.** The Inductor presets are SWITCHES on the learner (`model._r1_inductor_preset`);
the benchmark sets them, and the production path cannot reach them. `--trunk-precision` (lever A) and
`--learner-cudagraphs` (lever C) are flags, default OFF and byte-identical when off, because each one's gate
needs a real run path to be measured on. Neither enters the recipe surface (`--arch endstate` /
`recipe.fresh`) before its adoption read: bf16 needs the strength A/B (§2d); CUDA graphs need their paired
GPU benchmark (window E).

## 2. bf16 adoption design (lever A; phase 2, 2026-10-10)

**The split (design (b)): bf16 lives in the TRAINING micro-step only.** `--trunk-precision bf16`
(`gen3_trunk_precision_flag_v1`, config v155, default fp32, recorded and inherited on a flagless resume) sets the
LEARNER's live trunk rounds to bf16 autocast (fp32 out). Everything that must agree with the rollout stays fp32: T2, the
eval core and every snapshot build from a state dict (base classes), and the K9(b) probe's forward and its violation
scan run under `trunk_precision.fp32_scope`. The decisions inside R1 are fp32 by construction: the region ends at each
round's output; the pointer head, the masked log-softmax, the value, every loss and the AdamW state are fp32.
PLAN's earlier wording called design (a) — T2 also bf16 — "the honest one". It is not adopted, for three reasons:
1. K9(b) checks the DATA plumbing (stale weights, wrong action, misaligned rows). Under (b) it keeps its fp32 bars,
   its tie rule and its measured teeth unchanged.
2. Under (a) its healthy noise would grow to bf16 rounding scale: this is what `k9_bf16_2026-10-10/` prices. Its
   rounding scale R sets the tie epsilon (10 × R), which sets the excluded share against the 0.15 ceiling.
3. Under (b), the distance between the bf16 TRAINING function and the fp32 served function is bounded where it
   arises, in R1's own gate (the DRIFT bound below), at startup and at every canary.

What (b) costs: PPO's epoch-0 ratio reads the bf16 learner against the fp32 behaviour. That is importance-ratio
noise of |Δ log π| p50 3e-5 to 5e-3 by state, with clip fraction 0 at clip 0.15 (§2b, CPU). It is reported, and the
strength A/B (d) judges it.

### 2a. The R1 parity rule for a bf16 trunk (`gen3_r1_bf16_parity_v1`, `compile_regions._r1_verdict_bf16`)

**Why the fp32 rule cannot judge it.** Compiled bf16 and eager bf16 round at different points. Eager rounds every
op's output to bf16; Inductor fuses and upcasts. So they differ by bf16 ROUNDING, not by a miscompile. The unit
roundoff is 2^-8 relative, about 32,768 × fp32's. The fp32 rule refuses this on the loss alone: window B read loss
rel 9.7e-4 against the 1e-4 bar.

**The rule: three arms.** The gate runs three arms on the same weights and rows:
- compiled bf16 (C);
- eager bf16 (E);
- the fp32 EAGER REFERENCE (ref): R1's eager body under `fp32_scope`, i.e. the production function. It is proved
  eager by the region counters: the body runs once and the compiled route zero times.

The rule then asks two questions:

| question | quantity | condition | bar |
|---|---|---|---|
| **DRIFT**: is the bf16 function still near the fp32 one? | E vs ref: loss rel; global gradient rel-L2 | ≤ `BF16_DRIFT_BARS` | R1_BAR_K (4) × the healthy GPU max |
| **ENVELOPE**: is the compiled graph no farther from the truth than eager bf16? | C vs ref against E vs ref: loss; global gradient; EVERY judged parameter | d(C, ref) ≤ K · max(d(E, ref), floor) | K = R1_BAR_K × the healthy GPU max of the ratio (`BF16_ENVELOPE_K`, ≥ 1) |

The floor is the fp32 rule's own bar for that quantity, except the scalar loss:
- loss: bf16's unit roundoff 2^-9 = 1.95e-3 (F-PD-3; max with fp32's 1e-4);
- global gradient: √(2(1 − 0.9999)) = 0.0141, the cosine bar as a relative L2;
- per parameter: `R1_PARAM_BAR[regime]`, or the FRESH bar for an unmoved parameter, as in fp32.

So a bf16 graph is never held tighter than an fp32 one, and a quantity bf16 leaves exact still gets fp32's
allowance. The fp32 rule's rungs are unchanged: a fresh launch also runs on the seeded perturbation at the trained
regime's floors, and the unmoved-parameter rung still applies. They run at startup and at the update-10 canary.

**Rule 8 (deterministic).** Every bar is 4 × its healthy maximum, so every healthy read sits ≤ 25 % of the way to
its boundary. The comparisons are `not x <= bar`, so a NaN fails.

**The derivation** (`scripts/derive_bf16_bars.py`; `compile_regions_bf16_test` re-runs it on the committed reads and
fails on any constant that differs). Its inputs are window D's `E_bf16_envelope` reads:
- one compiled-bf16 install of `--arch endstate` at the production micro-batch (2,048 rows);
- the gate's own rows, slices 0 and 1;
- fresh, six ladder-perturbed states (0.05 × 4 seeds, 0.1 × 2) and 1 / 3 / 6 real bf16 updates;
- each arm twice, so the drift and ratio maxima include the run-to-run spread.

**The derived bars (window D, 20 reads):** DRIFT loss 0.014 / gradient rel-L2 0.68 (4 × 3.26e-3 / 0.169);
ENVELOPE K loss 5.4 / gradient 6.4 / per parameter 12 (4 × the maximum ratios 1.35 / 1.58 / 2.94). The loss
floor is bf16's unit roundoff 2^-9, not fp32's 1e-4 (F-PD-3: a scalar's bf16 error lands near zero by chance; under
1e-4 one read made a ratio of 8.5 and K_loss 35). Both bf16 arms were run-to-run bit-identical on the GPU, and R1's
cache held 1 entry through every fp32 reference arm. A `None` table would REFUSE a bf16 R1 (never a guess).

**The teeth, stated.** A backward miscompile on a parameter is refused only when it exceeds K × that parameter's own
bf16 error (`teeth` in `results/bf16_parity_bars.json`: the refusal threshold's median / p90 / max per state).
bf16's gradient noise is the resolution limit, so the rule is coarser than the fp32 one by the ratio of the two
errors. On the GPU (window D), the refusal threshold's median over the judged parameters is 0.2–1.3 relative error by
state (p90 0.5–3.1). So a dropped or sign-flipped path gradient (error 1–2) is refused on most parameters, a
10 % one is not. The global gradient (K 6.4 × max(eager's 0.01–0.17, 0.014)) and the loss are judged alongside. What the gate cannot resolve under bf16 is, by the same token, inside the noise the strength A/B judges.

### 2b. K9(b) under a bf16 learner

**Bar: unchanged** (design (b)). The probe and T2 are both fp32: max |Δ log π| < 1e-4 over the judged rows,
`FP32_TIE_EPS` 2e-4, the 0.15 ceiling and the flip-judge all keep their fp32 derivation. Two places make it so:
- `trunk_precision_flag_test` pins the probe: on a bf16-classed learner `probe_forward` returns BIT-IDENTICAL
  log-probs to the fp32 learner's, and without the scope they differ (revert-checked).
- `slots.served_replica` resets every T2 slot replica to the fp32 base classes. T2 deep-copies the learner policy
  into its slots, so without the reset a bf16 learner's classes would ride into T2 (`k9_bf16_2026-10-10` FINDING 7).

**The re-derivation for design (a)** (T2 and the probe both bf16; [`../k9_bf16_2026-10-10/`](../k9_bf16_2026-10-10/README.md)):
- **Setup.** CPU, eager T2 and probe, `--arch endstate`, 4,096 real Rust-collector rows per state. The states are
  ladder-perturbed (0.05 / 0.1), 6 fp32 updates in, and a real 9M production checkpoint.
- **Tie margins: no change.** They are bit-identical at a bf16 and an fp32 trunk on 100 % of rows (no declared
  selection site reads a trunk output). R_bf16 = R_fp32 = 3.66e-5, so the tie rule would not move.
- **The |Δ log π| bar: no workable value.** The healthy bf16-vs-bf16 max is 1.8e-4 to 2.8e-2 by state (pooled
  2.57e-2), so a 4 × H bar is ~0.2.
- **The one-step-stale fault (lr 2.8e-5) is lost.** It reads only 1.16–1.74 × its own state's healthy max: no bar
  separates it, even per state. At bar 0.2 the real `behaviour_probe` passed healthy rows and MISSED both stale
  faults, where the fp32 control caught both.
- **Wrong-action rows: 44 % caught at 0.2**, against 95–99 % at 1e-4.
- **So (a) cannot keep K9(b)'s teeth:** design (b) is the adopted shape.

**What (b) costs: the epoch-0 ratio noise** (bf16 learner vs fp32 behaviour, all rows):

| state | p50 / p99 / max \|Δ log π\| | approx-KL |
|---|---|---|
| perturbed 0.05 | 1.0e-3 / 4.9e-3 / 8.6e-3 | 1.5e-6 |
| perturbed 0.1 | 5.2e-3 / 3.1e-2 / 6.3e-2 | 4.8e-5 |
| 6 updates in | 3.3e-5 / 1.5e-4 / 2.5e-4 | 1.4e-9 |
| production 9M | 2.0e-3 / 1.7e-2 / 3.5e-2 | 1.1e-5 |

- **Clip fraction from numerics alone: 0** at clip 0.15 in every state. The worst |log r| is 0.063, 0.45 of the
  clip edge.
- **fp32 is a valid reference for bf16 error:** |fp32 − fp64| max 8.6e-6 vs |bf16 − fp32| max 5.5e-2, a ~6,000 ×
  separation.
- **Limit:** all of it is CPU. The CUDA re-run is in that README, ~10 min under a lease.

### 2c. The flag

`--trunk-precision {fp32,bf16}`:
- `ModelVersion.trunk_precision` (config v155; a pre-v155 config migrates to fp32), `_resolve`-inherited;
- never compared by `check_compatible`: the served forward and every shape are unchanged;
- in the shared compile cache's key (`COMPILE_ARG_KEYS`; R1 traces the classes);
- applied by `apply_training_hparams` on both build paths before grad checkpointing and the R1 install.

At fp32 everything is byte-identical: the production / end-state graphs, the K9 golden and the fp32 gate's line.

### 2d. The STRENGTH A/B for adopting it (a PROPOSAL: not registered, not run)

**Status: MANDATORY before adoption** (orchestrator, 2026-10-10). The bf16 trunk is a numerics change of the
TRAINING signal, not a rounding detail: window D measured the micro-batch gradient 1–18 % rel-L2 from fp32's
(F-PD-2), and the parity rule's per-parameter resolution is coarse. Speed never adopts it.

**Question.** Is the end state trained with a bf16 trunk not detectably worse than the same end state trained fp32,
at the closing test's own meter and rule?

**Arms.** EB = `--arch endstate --trunk-precision bf16` vs E = `--arch endstate` (fp32), the closing test's
recipe, ride-along heads and budget (15M steps). E's eight seeds are the closing test's end-state arm (ids 2001–2008,
already banked), REUSED as the comparator under an identity precondition at the bf16 pin:
- the fp32 end-state graph sha, the K9 golden and the production mirror are byte-identical to P_end's;
- a classified src/ + data/ diff shows no fp32 training-path change, as §2.3 of the closing test requires.

If identity fails, E is re-run at the bf16 pin: 8 more seeds.

**Stage 1, the SCREEN** (futility only, never adoption):
- EB seeds 2001–2004 to 5M steps;
- a 4 × 4 mirrored cross against E seeds 2001–2004's 5M checkpoints (matched STEP; **UNVERIFIED** that the closing
  runs keep a 5M checkpoint — else the nearest kept one, both arms at the same step);
- 1,000 mirrored pairs per cell, both greedy.
- STOP for futility iff the 90 % interval's upper end < −2.0 pp. Rule 8: an end within 1e-9 pp of −2.0 does not
  stop.

**Stage 2, the READ:**
- EB seeds 2001–2004 continue to 15M, plus 2005–2008;
- the 8 × 8 cross of the 15M finals, `main.h2h play-many` at the bf16 pin;
- the closing test's statistic (`cross_stat`: Δ̂, V̂ = (s²_R + s²_C)/8, df 14) and rule: PASS iff the upper end > 0
  AND the lower end > −2.0 pp; INCONCLUSIVE / FAIL as there.
- K9(b) strict at its fp32 bars (design (b) leaves them unchanged); the bf16 R1 gate and canary at the derived bars
  (a canary FATAL is INCONCLUSIVE, never a stop on strength).

Speed (the pooled quiet `train_ms` ratio) is REPORTED, never decides.

**Cost.** EB ≈ 8 × ~2.6 GPU-h (the end state's ~3 h per seed less the bf16 update saving), stage 1 ≈ 4 × 0.9 GPU-h
of that, plus the crosses (16 + 64 cells, ≈ 0.3 + 1.2 GPU-h).

**Order.** After the closing test's verdict, on whichever architecture is adopted. A PASS adopts bf16 into
`--arch endstate`'s recipe surface (`recipe.fresh`), with the design_learner_recipe §3.22 row in the same commit.

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
  - E_base (the closed-form KO at HEAD, the levers' paired baseline) and E_combo.

## 5. Status after windows A–D (2026-10-10; numbers in [`README.md`](README.md))

| lever | status |
|---|---|
| the attribution | MEASURED. Production 44.95 → the end state 53.5 s per update. The third trunk round is +6.9 s, ~80 % of the gap; the rest of the end state is +1.8 s; exact KO is +0.3 s |
| A. bf16 trunk region | MEASURED (window B): −20 % per 1-epoch call; per-epoch kernel time −21 % (GEMM −28 %, attention −51 %); peak memory −540 MiB. BUILT (phase 2): `--trunk-precision {fp32,bf16}` (config v155, default fp32, design (b): bf16 in R1 only, T2 and the K9(b) probe fp32); THE bf16 PARITY RULE with bars DERIVED from window D's 20 GPU reads (§2a). F-PD-2: bf16 moves the micro-batch gradient 1–18 % rel-L2 from fp32's, so adoption needs the strength A/B (§2d, a proposal). NEXT: window E's `E_bf16_flag` (the rule + canary on the flag path, timing) |
| C. CUDA graphs over R1 | UNBLOCKED, BUILT (CPU only). The retain_graph probes (grad balance, the per-term noise sampler) die under cudagraph trees on the first, diagnostic, update (F-PB-2); `--learner-cudagraphs on` now routes each probe micro-step to the un-graphed compiled route and every other micro-step (all of a plain update, ~448 of a diagnostic update's 480) to the graphed route — a second declared R1 signature, gated with an accumulation rung, prewarmed, locked, canaried. NEXT: window E, the paired `E` vs `E --learner-cudagraphs on` (time, peak reserved incl. the graph pool, gate + canary, GPU busy) |
| B1. coordinate-descent tuning | IMPRACTICAL as a startup cost: R1 did not finish compiling in 11 min. Revisit only behind the shared compile cache |
| B2. combo kernels | not measured (window C dropped it; low expected gain) |
| closed-form exact KO | LANDED as THE `exact`. Cost-neutral on the GPU (53.5 vs 53.5 s); CPU KO function 23.9 → 6.1 ms |
| perf item 1 (batched host reads) | MEASURED. −1.3 % per plain update; syncs 126 → 121 per micro-batch |
| the ride-along host-read fix (`3545b868`) | MEASURED (window D): plain E update 53.59 → 53.24 s (−0.7 %); host-blocked 2.36 → 0.50 s; GPU busy 93.2 → 95.1 % |
| **the update's GPU-idle time** | SETTLED (window C). It is 92 % busy over 3 epochs: epoch 0 85.6 %, later epochs ~95 %. About 1 s per update is the ride-along accumulator's epoch-0 `.item()` drain (`ridealong_heads.py:466`: 1.75 s of the 2.2 s blocked). The rest is ~5 % launch-bound gaps per later epoch, ~3 s per update (~6 %) in all. NEXT: defer the ride-along epoch-0 read onto the optimizer step's batched transfer, as perf item 1 did; CUDA graphs remain the fix for the launch-bound part |
| 8. the shared compile cache | the OWNER APPROVED it 2026-10-10 ("shared cache is fine as long as we check the hash and the workspace is clean"). BUILT and LANDED: `62c6165e`, then `15982f6c` (ignored importables count as dirt), then `ef6f49aa` (the disk preflight counts the root). MEASURED in window C on a fresh `--arch endstate` launch: T2 850.7 s cold → 124.9 s warm-shared; R1 ~277 → 104 s; the whole one-update launch 5.6 min warm vs > 17.8 min cold |
| T2 per architecture | MEASURED, one slot: P 3.17 / 6.38 ms at buckets 64 / 256; R +8.6 % / +10.3 %. E NOT measured (a harness bug, now fixed) |
