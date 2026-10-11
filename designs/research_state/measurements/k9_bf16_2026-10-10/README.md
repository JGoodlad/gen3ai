# K9(b) under the bf16 trunk region: the numbers for its re-derivation (2026-10-10)

**Question** (T25 perf lever A, `agents/model/trunk_precision.py`; `../perf_phase_2026-10-10/PLAN.md` §2.2). How does
the K9(b) behaviour check work when the trunk rounds run under bf16 autocast? Two candidate designs:

- **(a)** T2 and the learner's probe both run the bf16 trunk. K9(b) then compares two bf16 forwards.
- **(b)** T2 and the probe stay fp32 and only R1 runs bf16. K9(b) is unchanged, but PPO's epoch-0 ratio reads a bf16
  learner against an fp32 behaviour log-prob.

**Verdict (CPU only).**
- **(a) has no teeth.** The one-optimizer-step-stale fault sits INSIDE the healthy bf16 noise. On the same state its
  smallest micro-batch statistic is only **1.2–1.7x** the healthy max at lr 2.8e-5, and 3.1–4.9x at lr 3e-4. A bar
  ≥ 4x the healthy max therefore misses the fault, at every state.
- **(b) is clean.** The fp32 gate is bit-for-bit today's. The numerics move PPO's epoch-0 ratio by at most |r − 1|
  0.066, against the clip band's 0.15, with a clip fraction of **0** at every state.
- **Recommend (b).**

Every number below comes from [`result.json`](result.json) (`derive.py`) unless another file is named. **CPU, torch
2.8.0+cu126, at `3545b868`.** CPU bf16 kernels are not CUDA's; see the limits section.

## Method

- **Learner.** `--arch endstate` resolved as a launch resolves it (`build_parser` + `resolve_config`, via
  `update_bench.resolved_args`).
- **Rows.** Each CELL is a seeded real complete-game Rust-collector rollout of **4,096 rows** (16 envs x 256 steps,
  self-check build, run seed 1001, p2 a seeded random policy). That is two 2,048-row micro-batches, the endstate
  `batch_size`.
- **T2 precision.** T2 serves the trunk at the cell's BEHAVIOUR precision. Its replicas are deep copies of the
  policy, taken after `set_trunk_precision`, so the stored log μ is that precision's. T2 is EAGER on the CPU.
- **States.**
  - `fresh`: the seed-1001 init. It is SELECTION-FREE: the scorers are zero, so bf16 cannot reach log π, and
    `derive` sets it aside.
  - `perturb05` / `perturb10`: `parity_probe.perturb_`, keyed, seed 1234, at 0.05 / 0.1, the ladder's two scales.
  - `trained`: `train_state.py`. From the fresh init it ran **6 real PPO updates** on the CPU at fp32. Each update
    used a 2,048-row seeded rollout (run seed 2002) and the resolved recipe: lr 3e-4, n_epochs 10, clip 0.15. Each
    optimizer step was one micro-batch, because a CPU rollout cannot fill 65,536 rows. That makes 60 Adam steps.
    Entropy fell from 1.84 to 1.78; approx-KL per update ran 8e-4 to 6e-3 (`train_state_log.json`). Weights
    sha256[:16] `3ea0a61f9d06f0a2`, kept at `~/gen3ai_archive/k9_bf16_2026-10-10/trained_u6.pt`.
  - `prod9m`: a REAL long-trained policy, read only: `rb_es_prod_s2001` checkpoint 9,000,119, at `--arch
    production`, which has two trunk rounds. It is reported on its own and never pooled.
- **Forwards on every row** (`measure.py`).
  - The probe forward (`consistency.probe_forward`: train mode, grad on, 2,048-row chunks) at the bf16 and at the
    fp32 trunk, under `TieMargins` with per-site margins and the flip-judge's `flippable` at each epsilon of a grid.
  - T2-like bf16 variants, each with its margins: eval / no grad at chunk 2,048, 48, 16 (this harness's T2 bucket)
    and 8; train / no grad at 2,048; a 3e-7 weight jitter; and each round under `torch.compile` (inductor, eval,
    chunk 48), as a proxy for T2's compiled fusion.
  - The stale-weight fault, at the bf16 and at the fp32 trunk: `lr x sign(N(0,1))` on every weight, seed 5.
  - A float64 reference (`Fp64Mode` on a float64 copy) on 512 rows.
- **Archive.** The per-row arrays are `~/gen3ai_archive/k9_bf16_2026-10-10/cells/*.npz`. `cells/*.json` here are their
  summaries.

## 1. The rounding scale of the tie margins

| quantity | value |
|---|---|
| R_bf16: the largest \|m_learner − m_variant\| over 24,576 rows x every declared site x every bf16 variant | **3.66e-5**, from the weight jitter at `hypothesis_set.py:214 argsort` (perturb10). **Every** eval / batch / grad-mode / compiled-round variant reads **0** |
| R_fp32 on the same rows, same device | **3.66e-5**, the same value at the same site and row |
| eps_bf16 = round_up_125(10 x R_bf16) | **5e-4** (CPU), identical to this device's fp32 eps |
| learner margins, bf16 trunk vs fp32 trunk | **bit-identical on 100 % of rows**, in every cell (prod9m too) |
| excluded share at 2e-4: pooled before / after the flip-judge; largest 2,048 block | 6.05 % / 5.33 %; 9.5 % |
| excluded share at 5e-4: pooled before / after; largest block | 6.71 % / 5.35 %; 9.6 % |
| ceiling | under 0.15 at both epsilons. The rule (≥ 3x pooled, ≥ 1.5x the largest block) gives 0.2 here; 0.15 holds |

**Why R does not move.** On the end-state graph no declared MARGIN site reads a trunk-round output. Every margin is
the same number at bf16 and at fp32. So **bf16 changes nothing about tie exclusion, and the flip-judge is
untouched.** It changes only the continuous |Δ log π| that judged rows are held to. This is a property of TODAY's
graph; see FINDING 3.

## 2. Design (a): bf16 behaviour (T2) vs the bf16 learner

**HEALTHY** = \|log π_bf16 learner − log μ_bf16 T2\| over the JUDGED rows at eps 5e-4 (a flip-judged row counted
as-is), plus the learner vs every bf16 T2-like variant.

| cell | healthy per-row p99.9 / max | micro-batch max | stale lr 2.8e-5: smallest block (÷ this cell's healthy max) | stale lr 3e-4: smallest block (÷) |
|---|---|---|---|---|
| perturb05 | 4.7e-3 / 5.7e-3 | 5.7e-3, 4.2e-3 | 6.7e-3 (**1.16x**) | 1.77e-2 (3.1x) |
| perturb10 | 2.1e-2 / 2.4e-2 | 2.2e-2, 2.4e-2 | 4.2e-2 (**1.74x**) | 7.6e-2 (3.2x) |
| trained | 1.7e-4 / 1.8e-4 | 1.8e-4, 1.8e-4 | 2.3e-4 (**1.32x**) | 8.6e-4 (4.9x) |
| prod9m (production arch) | 2.2e-2 / 2.8e-2 | — | 4.6e-2 (1.67x) | 1.14e-1 (4.1x) |

- **Pooled.** H (the healthy max, including the T2-like variants) = **2.57e-2**.
  - B₄ = round_up_125(4H) = **0.2**; B₁₀ = **0.5**.
  - The smallest stale block over all states is **2.3e-4** at lr 2.8e-5 and 8.6e-4 at lr 3e-4. Both sit far UNDER
    either bar (S/B₄ = 0.0012 and 0.0043).
  - **No bar ≥ 4x the healthy max can be ≥ 10x under the stale fault.** Not pooled, and not per state, since the
    per-state ratio of fault to healthy is ≤ 1.74 at lr 2.8e-5 and ≤ 4.9 at lr 3e-4.
- **Today's fp32 gate on the same harness.** Healthy max 3.1e-6, a headroom of 32x under 1e-4.
  - At lr 2.8e-5 the stale fault is caught at perturb05 (4.8e-3) and perturb10 (3.7e-2). At `trained` it reads
    **7.3e-5, under the bar** (FINDING 4).
  - At lr 3e-4 it is caught in every cell: 7.5e-4 at the smallest, 7.5x the bar.
- **Wrong action** (judged rows with ≥ 2 legal actions, n = 11,178).
  - Taking the first other legal action, the median |Δ| is 0.127. At B₄ = 0.2 only **44 %** of such rows would be
    caught; at B₁₀, 31 %.
  - Taking the nearest other action, p1 = 4.6e-5 and the median 9.3e-3, so 3 % would be caught. Today's 1e-4 bar
    catches 95–99 % even by the nearest action (min-alt p1 4.6e-5, p5 2.6e-4).
- **End to end with the REAL probe** (`real_probe.py`: `consistency.behaviour_probe`, `fatal`, perturb05, one
  2,048-row micro-batch).
  - Design (a) at bar 0.2 / eps 5e-4 (`real_probe_a_perturb05.json`): healthy passes (judged max 5.6e-3). Both
    stale faults **PASS, which means they are MISSED** (6.9e-3, 1.77e-2). The wrong-action row at margin 0.34 is
    FATAL (0.824).
  - Today's fp32 control at 1e-4 / 2e-4 (`real_probe_fp32_perturb05.json`): healthy 8.3e-7; stale lr 2.8e-5 FATAL
    (6.0e-3), lr 3e-4 FATAL, wrong action FATAL.
- **Where the CPU noise comes from.** The bf16 cast AMPLIFIES any fp32-level upstream difference to bf16 rounding.
  - A 3e-7 weight jitter moves bf16 log π by up to 4.7e-3 at perturb05, against 2.4e-6 at fp32.
  - The grad-on probe vs every no-grad forward differs by the cell's whole healthy max. Inside a round at the same
    input, grad mode changes nothing (`diag_grad_mode_cpu.json`: 0.0). The SDPA kernel choice moves a round's
    output by up to 0.025 (default vs MATH).
  - Inductor-compiled rounds vs eager rounds: up to 2.9e-3 (perturb05).
  - On the GPU, T2 is compiled and the probe eager, so every row differs upstream. Expect at least this.

## 3. Design (b): bf16 learner vs fp32 behaviour, over ALL rows (no exclusion)

| cell | \|Δ log π\| p50 / p99 / p99.9 / max | clip fraction at 0.15 | max \|r − 1\| | approx-KL | fp32 learner today: max |
|---|---|---|---|---|---|
| perturb05 | 1.0e-3 / 4.9e-3 / 6.8e-3 / 8.6e-3 | **0** | 0.0086 | 1.5e-6 | 1.1e-6 |
| perturb10 | 5.2e-3 / 3.1e-2 / 5.1e-2 / 6.3e-2 | **0** | 0.066 | 4.8e-5 | 3.1e-6 |
| trained | 3.3e-5 / 1.5e-4 / 1.9e-4 / 2.5e-4 | **0** | 0.00025 | 1.4e-9 | 4.8e-7 |
| prod9m (9M steps, production) | 2.0e-3 / 1.7e-2 / 2.4e-2 / 3.5e-2 | **0** | 0.036 | 1.1e-5 | 3.3e-6 |

- **Pooled endstate.** Max \|Δ\| 0.063 is 0.45 of the clip edge, log(1.15) = 0.140. The clip fraction from numerics
  alone is 0.
- **The fp32 K9(b) gate is untouched** (`scope_test.py` → `scope_result_eager.json`). Both mechanisms make the probe
  forward **bit-identical** to a pure fp32 policy in log π, the full masked log-probs and the tie margins, and the
  bf16 forward afterwards is bit-identical to before:
  - **A**, a class-swap scope: `set_trunk_precision(policy, "fp32")` around the probe, then back to `"bf16"`;
  - **B**, a measurement-local flag read by a bf16 subclass.
- **Under dynamo** (one round compiled bf16, `fullgraph=True`, backend `eager`):
  - an EAGER scope between two compiled calls adds **no** graph;
  - a compiled call made INSIDE the scope **recompiles**, a second graph whose output is the fp32 result. That is
    loud under the K6 lock, never a silent stale bf16 graph. This holds for both mechanisms.
- **Which is safe.** **A is the safe one.** It needs no new class and reuses the production classes. B's foreign
  subclass is invisible to `set_trunk_precision`, which SKIPS it, and to `trunk_precision_of`, which reads `"fp32"`
  with B installed (FINDING 6).

## 4. The reference: is fp32 a valid yardstick for bf16 error?

**Yes.** This covers 2,832 fp64 rows of the informative endstate cells, judged at the fp32 eps.

- The two rows where fp64 resolved a discrete op differently (\|fp32 − fp64\| ≥ 1e-5: 3.1e-4 and 3.9e-3) are counted,
  not judged.
- Over the other 2,830 rows:

| | fp32 − fp64 | bf16 − fp32 |
|---|---|---|
| p50 / p99 / max | 1.3e-7 / 2.3e-6 / 8.6e-6 | 7.8e-4 / 2.5e-2 / 5.5e-2 |

- The separation is **6,351x** at the max and **6,073x** at the median, ≥ 100x as required.
- prod9m: fp32 − fp64 max 3.8e-6, bf16 − fp32 max 2.4e-2.

## Recommendation: (b)

1. **(a) cannot keep the fault table.** K9(b)'s job is to see served weights one optimizer step stale, a misaligned
   observation, or a wrong action. Under (a) the stale-weight class sits inside the healthy bf16 spread at every state
   measured. That includes a real 9M-step policy and the recipe's own lr 3e-4. A bar that never false-fatals must
   clear the healthy max, which sits at 25–85 % of the fault's (lr 2.8e-5). A bar 4x above it misses the fault class
   by 200–800x (S/B₄ 0.0012 pooled), and most single wrong-action rows too. On the GPU the spread can only grow,
   because compiled T2 vs an eager probe differ on every row upstream of the cast.
2. **(b) keeps today's gate byte for byte** and costs PPO nothing measurable. Epoch-0 |r − 1| ≤ 0.066 against the
   0.15 clip, a clip fraction of 0, approx-KL ≤ 5e-5.
3. **What (b) does not see** is a defect inside the bf16 region itself. That is R1's matched-envelope parity gate's job
   (PLAN §2.1), against the fp64 arm this file shows is a valid reference.
4. **What (b) costs.** The probe runs the eager fp32 trunk while R1 runs compiled bf16. That is one class-swap scope
   around an eager forward the probe already makes, with no new compiled signature.

## FINDINGS

1. **CPU only.** Every number here is the CPU's bf16, eager T2, at torch 2.8.0. CUDA bf16 GEMM / SDPA kernels and
   compiled T2 (the graph backend) differ. The design-(a) spread on the GPU is UNVERIFIED and expected larger. The
   design-(b) clip margin on the GPU is UNVERIFIED. `measure.py --device cuda` is written but **never run**.
2. **The GPU graph backend under bf16 is UNTESTED.** It is unknown whether T2's graph backend (AOT / export) carries the
   bf16 class's autocast region at all, and what it costs to compile cold. The `K9BF16_T2_BACKEND=graph` path is
   written, not run.
3. **The "no margin site downstream of the trunk" property is load-bearing for (a) and (b).** It is measured on
   today's endstate and production graphs (100 % identical margins). A future selection on a trunk output would put
   bf16 rounding into the tie margins, and both epsilon and the exclusion must then be re-measured.
4. **Today's fp32 K9(b) on the `trained` state misses the lr 2.8e-5 stale fault.** Both micro-batches read
   **7.3e-5 < 1e-4**. The 2026-09-30 table's 200x margin at that lr was a fresh perturbed learner's. On this
   near-uniform early policy (small logits) the fault signal is small. At the endstate recipe's lr 3e-4 it is
   caught (7.5e-4).
5. **The endstate's fp32 tie-margin rounding scale exceeds the banked one under the jitter variant.** `hypothesis_set.py:214
   argsort` moves by 3.66e-5 at perturb10 (9.8e-6 perturb05, 1.6e-6 trained, 8.8e-6 prod9m), against the banked CUDA
   R 1.44e-5. By the banked rule that gives eps 5e-4, not 2e-4. CPU, at a stress state; it is not about bf16.
6. **`set_trunk_precision` silently skips a foreign subclass** of a round class, and `trunk_precision_of` then reads
   `"fp32"`. A flag-subclass mechanism would need `trunk_precision` to know it, which is another reason for scope A.
7. **T2 replicas take the policy's CLASS at deep-copy time.** For (b), T2 must be built before the learner is set to
   bf16, or its replicas set back to fp32. Slot loads (`svc.load`, a state dict) do not change classes. That coupling
   is silent today.
8. **Fault teeth were measured on 4,096 rows per cell** (2 micro-batches x 3 endstate states, plus prod9m). The
   wrong-action and stale reads are per micro-batch maxima, not distributions over many updates.
9. **`train_state`'s own fp32 K9(b)** (`warn`) read an excluded share of **0.147** at its update 5. That is 0.3 pp
   under the 0.15 ceiling (CPU, 2,048 rows): `train_state_log.json`.
10. **The inductor check covers only the dynamo guards.** The compiled-guard check ran with backend `eager`. The guards
    are dynamo's and do not depend on the backend, but an inductor / CUDA-graph R1 was not exercised.

## Re-run on the GPU (a window, lease held)

The essential cells are perturb05 and trained, at both behaviours:

    export GEN3AI_GPU_LEASE_TOKEN=<token>
    scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh 40 env K9BF16_TRAINED=$HOME/gen3ai_archive/k9_bf16_2026-10-10/trained_u6.pt \
      designs/research_state/measurements/k9_bf16_2026-10-10/run_cells.sh $HOME/gen3ai_archive/k9_bf16_2026-10-10/cells_cuda cuda \
      perturb05:bf16 perturb05:fp32 trained:bf16 trained:fp32
    python designs/research_state/measurements/k9_bf16_2026-10-10/derive.py --cells $HOME/gen3ai_archive/k9_bf16_2026-10-10/cells_cuda --out result_cuda.json

- **Cost (ESTIMATE).** About **8–10 min** in all. On the CPU each cell took 3–8 min under contention, most of it
  forwards and the eager-T2 collection. On an RTX 3080 Ti, expect about 1–2 min per cell, plus about 1 min of
  inductor compile for the compiled-round variant.
- **The real compiled T2.** Add `K9BF16_T2_BACKEND=graph`. It costs a cold T2 compile per cell; its time and whether
  it works are UNVERIFIED (FINDING 2), so give it its own slot.

## Files

| file | what |
|---|---|
| `common.py` | the build (endstate, resolved like a launch), forwards, margins, the faults, the fp64 copy |
| `measure.py` | one (state, behaviour) cell → `<out>.json` + `<out>.npz` |
| `train_state.py` | the 6-update trained state |
| `run_cells.sh` | runs cells (CPU: `nice -n 19`, `mem_cap 24`, 4 threads; CUDA: under the lease) |
| `derive.py` | the declared rules → `result.json` |
| `real_probe.py` | the real probe, end to end, at a candidate bar |
| `scope_test.py` | design (b)'s fp32-probe mechanisms, and their dynamo interaction |
| `diag_grad_mode.py` | where the grad-on vs no-grad bf16 difference does NOT come from (a round at a fixed input) |
| `result.json`, `cells/*.json`, `real_probe_*.json`, `scope_result_eager.json`, `diag_grad_mode_cpu.json`, `train_state_log.json` | the outputs |
