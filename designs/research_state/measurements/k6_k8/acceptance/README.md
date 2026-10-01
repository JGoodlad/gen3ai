# K8 acceptance read — the declared regions vs the pre-regions learner (2026-10-01)

**Verdict: ACCEPTED on the two timing criteria.** On torch 2.8 with TF32, one production PPO update
takes **36.3 s with the K8 regions, against 46.7 s before** (ratio **0.778**; the bar is ≤ 0.80).
The compiled share of the update's wall time is **90.0%, against 63.4%** (the bar is ≥ 80%).
The lock held with 0 compiles after it.

Tag: MEASURED, n = 2 per arm, interleaved A1 B1 A2 B2. The B arm is tight (36.27 / 36.37 s). The
A arm spreads 2.7 s (45.28 / 48.03 s). The ratio's margin under the bar (0.022) is smaller than
A's spread relative to its mean (0.058), so with n = 2 the verdict is "met", not "met with room".

## What was compared

| | A (pre-regions) | B (K8) |
|---|---|---|
| tree | main `990851de` (K6's compile half; the extractor-only compile) | `aa693925` (K8.3 + the regions R0 / R1 + the stash rank probe + the device-resident batch). The same code as the shipped K8 commits, apart from one idempotent `ensure_hermetic_cache` call |
| learner | `--compile-trainer` gate → prewarm → lock (legacy) | the region gate → prewarm of 2 signatures → lock |

Both arms share the rest of the setup:

- **Tool:** `python -m main.compile_inventory run --stage time --device cuda --matmul-precision high --keep-prewarm --unbracketed`.
- **Inputs:** arm C's `final_model.zip` (`models/ai_v14_02_lbat_ctrl`) and the learner benchmark's pinned buffer (`~/gen3ai_archive/learner_bench/20260928_135948_cuda/rollout_buffer.pkl`).
- **Shape:** 98,304 rows, 48 micro-batches of 2,048, accumulation 32, 10 epochs, C's own flags.
- **Box:** torch 2.8.0+cu126, RTX 3080 Ti. The GPU was serialised by `gpu_lock`. No gate of this lane ran during a timed unit, and load average was 1.1 / 2.1 / 3.7 at the end of B1 (the units record no load of their own: **UNVERIFIED** per unit).

The driver (`run.sh`) and the reader (`summarize.py`) are here. `results/` holds each unit's
`time_result.json` and `time_analysis.json`.

## Numbers

| unit | unbracketed update | bracketed | host scalar reads | batch phase | compiled share of train wall (profiled, diag-skipped) | compiled share of kernel time | GPU idle (2 profiled epochs) |
|---|---|---|---|---|---|---|---|
| A1 | 45.28 s | 50.74 s | 66,784 | 2.36 s | 63.6% | 85.2% | 2.35 s |
| A2 | 48.03 s | 45.50 s | 66,784 | 2.27 s | 63.1% | 85.2% | 2.44 s |
| B1 | 36.37 s | 36.32 s | 9,942 | 0.31 s | 89.9% | 98.1% | 0.58 s |
| B2 | 36.27 s | 36.34 s | 9,942 | 0.31 s | 90.2% | 98.1% | 0.56 s |

Where the time went:

- **Bracketed phases, A2 → B2.** The fold's eager `loss` phase (6.5 s) disappeared into R1. That phase is now inside `forward` (10.8 → 12.1 s), because R1 holds `evaluate_actions` together with fold steps 1–3a. `backward` went from 25.3 to 23.7 s, and `batch` from 2.27 to 0.31 s.
- **Eager kernels.** The profiled epochs' eager kernel time fell from 1,113 to 131 ms.

The update's results agree between the arms: `train/loss` 0.38423 in both, `approx_kl` 0.00871 in both. The other tags match within TF32 run-to-run noise.

## The other acceptance criteria

1. **Graphs equal regions × signatures.** After the lock the cache holds 2 code objects with 1 entry each. The routine `compile_regions_test` pins this as inventory == the table.
2. **0 undeclared breaks.** Every region is compiled `fullgraph=True` on 2.8, so a break would be a startup FATAL. 2.5.1 is legacy; its extractor-only path is in the deletion pass.
3. **Compiled share ≥ 80%.** Met (above).
4. **0 recompiles after the freeze.** The time stage ran under the K6 lock, where any compile is a FATAL, and both B units finished `ok`.
5. **Update time ≤ 0.80×.** Met (above). The idle re-baseline is 46.7 s; the K8 inventory's 53.2 s was measured under contention.
6. **K6 parity and the K9 golden.**
   - **K6 parity.** The startup region gate passed at TF32 in both B units: R1's loss and the gradient's 1 − cos against the fp32 reference were within 4× eager's own TF32 error.
   - **R1 on CUDA is non-vacuous.** The compiled-vs-eager gradients differ in 2.06M of 3.07M elements; see `designs/training/compile_flags.md` "K8 — DECLARED COMPILE REGIONS".
   - **K9 golden.** It was re-recorded once, at K8.3, as a pure refactor with its declared bar (`../fold_equivalence/`). K8.4–K8.6 leave it unchanged.

## Two tool defects this read found (fixed in the same branch)

- **K9(b) killed every TF32 read.** The behaviour check FATAL'd at the first update of every torch-2.8 TF32 read of the pinned buffer (p99 |Δ| 0.0058 > 0.0036). The buffer's stored log-probs come from another program, so the tools now pass `--behaviour-check warn` explicitly. A launch's default is still `fatal`.
- **The time stage refused the regions learner.** It took a regions learner for "not compiled" (it read only an instance-compiled extractor forward), and it now accepts the regions.

`A1_behaviour_fatal/` and `B1_tool_refused/` under `~/gen3ai_archive/k6_k8/accept/` are those two failed first attempts. They were rerun as A1 and B1.

## A FINDING the numbers do not cover

On 2.8 the trainer still runs the legacy `--compile-trainer` extractor gate before `arm_compile_sentinel` replaces it with the regions. The B log reads "6 graphs compiled in this process (incl. the gate's)". That is minutes of startup compile the steady state never uses, and it is a candidate cleanup for the deletion pass.
