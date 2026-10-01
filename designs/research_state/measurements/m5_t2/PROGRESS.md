# M5 Lane T2 — the unified inference service: PROGRESS (resume point)

Lane T2 of `designs/endstate/program_rust_core.md` §2 M5 (the lane table row, and the **"T2 DESIGN"**
paragraph under "### T2 — The inference tier", which is the design of record and carries the
measured backend decision). Owns `src/agents/inference/service/`. Consumers that will call it:
Lane E (opponent routing), Lane H (eval), Lane G (the trainee's rollout, wiring). Nothing in training
calls it yet — it is standalone and tested standalone.

Owner decisions it implements (ledger L21431 (2), 2026-09-27): unified inference (trainee, opponents,
eval) through fixed GPU weight SLOTS, stacked same-arch weights, slot-tagged requests, fixed-width
buckets, priority classes (rollout > eval filler), eval as background filler, GPU-over-CPU bias; float
rounding accepted, every compiled path parity-gated. Principle: the M5 DECLARED LIFECYCLE
(`*_after_freeze` counters stay 0).

## How to test

```bash
export PYTHONPATH=$PYTHONPATH:src
python3 -m pytest src/agents/inference/service/service_test.py -q          # routine, CPU, ~5 s
GEN3AI_TEST_ALLOW_GPU=1 python3 -m pytest src/agents/inference/service/service_cuda_test.py -q  # slow, idle GPU, ~2.5 min
python3 src/agents/inference/service/service_benchmark.py --backend graph --buckets 8,48,128 \
    --slots 2 --ckpt /home/goodlad/dev/gen3ai/models/ai_v14_06_lbat_ctrl_fix/final_model.zip
```

Before any GPU use: `pgrep -af '[t]rain_rl_agent'` and `nvidia-smi`; with a trainer live keep to
small forwards (< 3 GB) and never within 10 min of a trainer launch. Fresh
`TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` per run (the tests and the benchmark do it).

## Units

| # | unit | status | commit |
|---|---|---|---|
| 1 | design note (program doc "T2 DESIGN" + a Decision-record row) + backend measurements (`probes/`) | LANDED | see git log (`M5 T2 unit 1+2`) |
| 2 | service skeleton: `decision.py` (bit-identical decision forward), `spec.py`, `slots.py` (stacked storage, two load identities), `parity.py` (the compile gate's bars + mask contract + greedy), `service.py` (eager + graph backends, lifecycle, packing, priority, arena), tests (CPU routine + CUDA slow), `service_benchmark.py`; `team_transformer._event_reference_cells` built on-device (the one host sync in the compiled graph) | LANDED | see git log (`M5 T2 unit 1+2`) |
| 3 | multi-slot throughput: vmap over stacked weights NO-GO (in-place writes into forward-created tensors); LANES instead — per-lane stream + graph pool + static inputs, graphs captured on their lane's stream (the shared-capture-stream cuBLAS-workspace defect found + regression test that fails on revert), CONCURRENT parity gate at startup + canary; Lane E's shape 9.89 → 3.95 ms per flush at 4 lanes | LANDED | see git log (`M5 T2 units 3+4 + AOT`) |
| 4 | staging: `engine.py` — double-buffered pinned host input arenas + event waits, ONE H2D and ONE D2H per flush, device-side chunk staging; `submit` copies rows; `ticket.host()` | LANDED | same |
| 4b | backend `aot` (torch ≥ 2.8, weights as inputs; `aot.py`) + CUDA 12.6 headers in `gen3ai_torch28` (`environment_torch28.yml`) + the C++ loader proof (`probes/cpp/`) | LANDED | same |
| 5 | Lane E's adapter (opponent rows from the core's columns → `submit` per slot, greedy/sampled actions back) — BUILT by Lane E as `agents.training.rust_env_opponents.PolicyOpponentServer` (gated: `m5_laneE/PROGRESS.md`) | LANDED (Lane E) | Lane E's commit |
| 6 | K3 hermetic per-run compile cache for the service's buckets (restart cost); a torch-2.8 run of `service_cuda_test` in `gen3ai_torch28` | LATER | |

## Units 3/4/4b measurements (RTX 3080 Ti, idle GPU via a coordinator training pause, 2026-09-29; `window_2026-09-29/`)

| read | result |
|---|---|
| 2.8 `graph` vs `aot`, real ckpt, 1 slot, per flush B = 8/48/128 | 1.25/1.81/2.76 vs **1.91/2.31/3.31 ms** |
| 2.8 `aot` parity (real / perturbed fresh through the SAME package) | 1.9e-5 / 7.7e-7 ; 8.3e-7 / 4.5e-7 |
| 2.8 `aot` build / size / load | ~77 s per bucket (graph ~51 s) / ~15 MB / 0.01–0.09 s |
| `aot` from C++ (libtorch only, no libpython) | loads in 131 ms, equals eager at 6e-6 / 1.8e-7 |
| embedded-weight package + `load_constants` swap | WRONG: 0.076 / 0.026 (folded constants) — hence weights as inputs |
| `aot` inside a CUDA graph | fails: "operation not permitted when stream is capturing" |
| lanes, 8 slots x B = 8, raw replay (1/2/4/8 lanes) | 9.81 / 6.79 / 4.26 / 2.58 ms; parity 7.2e-7; 212 → 399 MiB |
| service, Lane E shape (48 rows over 8 slots, B = 8) | **9.89 ms (1 lane) → 3.95 ms (4 lanes)** |
| lanes before the capture-stream fix | concurrent replays wrong: max|dlogp| 0.048 |

Rebased over `23f4f85c` (gen3_fresh_parity_probe_v1, another agent): the per-slot gate's vacuity
fallback is theirs; the CONCURRENT gate got the same treatment (every slot perturbed in place with
seed PERTURB_SEED + slot, then the real weights waived). Verified on the GPU after the rebase: the
lane regression + graph tests pass, and a FRESH 2-slot/2-lane service starts with the perturbed
concurrent reports, weights restored bit-exact, canary clean (run beside another agent's 770 MiB
probe — correctness only, no timing).

`aot28_weights_as_inputs_probe.jsonl`'s LATENCIES are void (a trainer was starting on the card);
its parity columns stand.

## Measurements, units 1+2 (RTX 3080 Ti, idle GPU, 2026-09-29; `probes/`)

Checkpoint `ai_v14_06_lbat_ctrl_fix/final_model.zip` (read-only), the 64-row real-obs fixture.

| arm (torch 2.5.1 unless noted) | B = 8 | B = 48 | parity (max\|Δ logp\| / \|ΔV\|) |
|---|---|---|---|
| eager | 30.5 ms | 32.4 ms | reference |
| Inductor (no graph) | 3.93 ms | — | 3.3e-6 / 2.4e-7 |
| Inductor `reduce-overhead` | 2,091 ms (cudagraphs skipped: host-built constant) | — | 3.3e-6 / 2.4e-7 |
| **Inductor + manual CUDA graph** (T2's `graph`) | **1.22 ms** | **1.97 ms** | 3.8e-6 / 1.8e-7 |
| AOTInductor (fill rewrite + CUDA_HOME shim) | 1.86 ms | — | **0.68 / 0.012 — MISCOMPILE** |
| 2.8.0+cu126 Inductor + CUDA graph | 1.22 ms | 1.77 ms | 8.6e-6 / 7.2e-7 |
| 2.8.0+cu126 AOTInductor | BLOCKED: no `<nv/target>` (no CUDA toolkit) | | |

Per-bucket static compile + capture (`static_buckets_graph.jsonl`): B = 2 / 8 / 48 / 128 →
1.06 / 1.23 / 1.97 / 2.94 ms per replay (23 µs/row at 128), ~75 s compile each (2.8: ~42 s), capture
0.6–2.2 s, 369 MiB peak with one shared graph pool. Dynamic-shape compile fails on 2.5.1. Two module
instances of the same class share the compiled code (no recompile). The policy is 3.07M params
(12.3 MB) + 0.97M buffer elements: a slot is ~16 MB.

**The service end to end** (`service_benchmark_graph_2.5.1.json`, `service_benchmark.py`, the real
checkpoint, 2 slots, buckets 8/48/128, no busy-box warning): startup 249 s (build = 3 compiles +
6 captures 248 s, parity 0.5 s); worst startup parity 1.7e-5 legal log-prob / 7.2e-7 V; one flush
filling a bucket costs **1.25 / 1.82 / 2.78 ms** (156 / 38 / 22 µs per row) — staging, H2D, replay
and arena copy included; a mixed flush (64 rows on slot 0 + 42 on slot 1) 4.6 ms; 397 MiB peak;
all three `*_after_freeze` counters 0.

## Design decisions (mirrored in the program doc's T2 DESIGN paragraph)

- Backend `graph` = `torch.compile` per bucket + one CUDA graph per slot × bucket in ONE shared pool;
  after the freeze dynamo is never entered (structural K6 lock). AOT rejected on measurement.
- Slots: per-group replicas whose state-dict tensors are views into stacked `[n_slots, …]` storage;
  a load is an in-place copy (captured graphs serve it), refused on a state-dict SIGNATURE or a
  FORWARD-FINGERPRINT mismatch, and parity-verified at the verify bucket (failure poisons).
- Buckets ≥ 2, default `(8, 48, 128)`; packing: largest buckets then the smallest that holds the
  remainder; pad rows repeat a real row.
- Priority: every ROLLOUT row per flush + `filler_batches_per_flush` batches of EVAL/FILLER, FIFO.
- Results are views into a declared arena, valid until the next flush (a stale read raises).
- The gate: `compile_trainer.decision_verdicts`'s bars (legal log-prob 1e-3, V 1e-4 at fp32; TF32
  rule otherwise) + mask contract + greedy on decisive rows (near-ties reported) — at startup for
  every slot × bucket (full and padded), on every load, and as `canary()`.

## Findings for the orchestrator

- **Lanes default to 1 in the spec (declared, never inferred).** Lane E should declare
  `lanes = min(n_slots, 8)` — 8 lanes read 2.58 ms for 8 slots vs 4.26 at 4 (raw replay).
- **CLOSED (2026-09-29): the double-buffer event wait has a revert-must-fail proof** —
  `service_cuda_test.test_the_double_buffer_wait_stops_a_repack_before_the_queued_copy_reads_it`
  holds the GPU with a sleep kernel so every flush's copy is still queued while the host issues the
  next flushes: passes with the wait, FAILS with it removed ("flush 0: served another flush's
  rows", twice in two runs). It needs backend `graph`: on `eager` the forward's 24 host syncs block
  the host inside each flush and the reverted wait PASSED (measured) — so the eager backend never
  exercises this race at all.
- **vmap over stacked weights needs a model-wide out-of-place rewrite** (first blocker
  `t0_species.py:82` `zeros(...).scatter_(1, ids, src)`). Owner-level call if the lanes' 2.5x is not
  enough.

- **The eager forward makes 24 host syncs per call** (Python-scalar `index_put`s, host-built
  constants: `extractor_ctx.py` 337/348, `damage_op.py` 743/875/879, `damage_op_blocks.py` ×12,
  `damage_op_pairwise.py` ×4, `team_transformer.py`, `extractor_forward.py` 393). Harmless for the
  compiled paths; they make eager uncapturable and cost eager GPU inference (prober, anchors on GPU).
  Hygiene, not T2's.
- **AOT for inference is blocked on both torches on this box** (see the table). Revisit only with a
  CUDA toolkit installed (a box change) and a reason AOT beats compile + graphs (artifact reuse
  across restarts is the candidate; speed is not).
- **A fresh production-arch policy is a vacuous log-prob probe** (zero-init pointer head ⇒ constant
  logits): the AOT miscompile read 0.0 on it and 0.68 on a real checkpoint. Any parity test on
  fresh weights must perturb them (`fixtures.perturbed_fresh_policy`).

## Flat weights — the perturbation LADDER (gen3_parity_perturb_ladder_v1, 2026-09-30)

**The refusal.** The cutover-prep CPU launcher runs (`~/gen3ai_archive/cutover_prep/fresh2`,
`fresh3`: `--arch production --critic winprob --env-core rust`, 4 envs, 64-step rollouts) died at
step ~14–16k in `collector.after_update` → `svc.load(trainee)` with `VacuousParity`. Every restart
then refused the SAME weights at T2 STARTUP (the resume loads `final_model_exception.zip`), three
crashes in a row until the circuit breaker stopped it. The trainee had lost ~97 % of its games, so
its win-prob critic COLLAPSED: the head saturated at logit −8…−10, V spread 6e-6 across the fixture.
The gate's single perturbation (0.05) only lifted V's spread to **9.58e-5**, under the 1e-4 bar.
It is a collapsed critic, NOT fresh weights.

**Does a FRESH production run hit it? No, at startup.** Measured on CPU, on the 8/7-row fixture that
the smallest bucket judges. Fresh production policies (seeds 0/1/2) have V spread **0.06–0.68**
with only the log-probs vacuous (0.0), and the 0.05 rung makes both informative (log-prob spread
≥ 0.41, V ≥ 0.19). The existing fallback passes them, so a fresh `--arch production` launch starts.
The exposure was a run whose critic collapses, at its next trainee load, at a pool / eval load of
such a snapshot, and at every restart onto it. The eval core loads through the same `svc.load`.

**Spreads, CPU, min over the 8/7/48/47/128/127-row fixtures (V spread; bar 1e-4):**

| weights | real | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|
| fresh3 `final_model_exception` | 5.9e-6 | **9.6e-5** | 2.4e-3 | 2.9e-3 | 1.2e-2 | 1.0 |
| fresh3 `checkpoint_1024_steps` | 5.0e-6 | 3.7e-3 | 0.19 | 0.25 | 1.0 | 1.0 |
| fresh seed 3, win head bias −12, weight ×0.01 | 8.7e-8 | 5.0e-6 | 1.1e-5 | 2.0e-3 | 1.0 | 1.0 |
| the same, bias +9 | 1.7e-6 | 1.5e-5 | 1.1e-5 | 2.4e-6 | 1.6e-5 | 1.0 |
| fresh seed 1 (unmodified) | 6.0e-2 | 0.21 | 0.13 | 0.94 | 0.68 | **0.0** |

Informativeness is NOT monotone in the scale: at 1.0 a fresh policy saturates V the other way.

**Why the scale is CAPPED at 0.1 (the first design walked 0.05 → 1.0; it was refused on the GPU).**
Compiled `decide` (the graph backend's callable) vs the eager reference. Setup: RTX 3080 Ti,
torch 2.5.1, 48 fixture rows, seeds `PERTURB_SEED + 1000·k` for k = 0, 1, 2, max over five weight
sets (fresh, collapsed −12 / +9, fresh3's `final_model_exception`, `ai_v14_06_lbat_ctrl_fix`
final). Raw rows: `~/gen3ai_archive/vacuous_parity/gpu_rungs.jsonl` (2026-09-30).

| scale | fp32 max \|Δ log π\| | fp32 max \|ΔV\| |
|---|---|---|
| 0 (real weights) | 1.0e-5 (trained) | 6.0e-7 |
| 0.05 | 9.5e-7 (fresh-based); 1.9e-5 (trained) | 3.9e-7 |
| 0.1 | 4.5e-6 (fresh-based); 3.1e-5 (trained) | 1.6e-6 |
| 0.2 | 9e-5 … **1.7e-3** | 6.9e-5 |
| 0.3 | up to 2.8e-2 | 2.0e-4 |
| 0.5 | up to **0.66** | 6.8e-4 |

The bars are absolute (log π 1e-3, V 1e-4). At 0.2 a CORRECT graph already crosses the log-prob
bar (fresh base, seed +1). The end-to-end T2 run refused the collapsed critic on its concurrent
gate at the 0.5 rung (8.6e-2). So the ladder is now `PERTURB_LADDER` = (scale, seed offset) rungs:
scales 0.05 then 0.1, eight seeds each, and `PERTURB_MAX_SCALE` = 0.1 is enforced by the spec.

**First informative rung of the capped ladder** (CPU, the gate's 8/7/2/1-row fills):

| weights | first rung |
|---|---|
| fresh seeds 1 / 3 | (0.05, +0) |
| fresh3 `final_model_exception` | (0.05, +2) |
| collapsed +9 (seed 0) | (0.05, +2) |
| collapsed −9 (seed 0) | (0.1, +3) |
| collapsed −12 (seed 3), collapsed +9 (seed 3) | **none — REFUSED** (`FATAL_CONFIG`) |

That last row is the declared limit. A critic saturated beyond what scale 0.1 moves is refused,
never passed and never judged on an ill-conditioned rung. The real collapse that started this
(fresh3) resolves at the third rung.

**TF32 (`--matmul-precision high`): the precision-keyed tie band and ladder cap
(gen3_precision_keyed_parity_v1, 2026-09-30).** With the same setup, a FRESH production policy was
REFUSED at T2 startup under TF32. The concurrent gate reported "greedy action differs on 1 decisive
row" at margin 3.5e-3. The greedy rule's band was the fp32 log-prob bar (1e-3) at every precision,
and was the bar itself, not 2x it (the TECH_DEBT row). The default `highest` always passed.

Calibration: T2's served decision, compiled `decide` at TF32 vs eager at TF32, per row, on the
gate's own fixture rows. Setup: RTX 3080 Ti, buckets 8 / 48; raw rows in
`~/gen3ai_archive/vacuous_parity/gpu_tf32.jsonl`.

| weights | rows | max \|Δ log π\| | p99 |
|---|---|---|---|
| 4 trained checkpoints (`ai_v14_01/05/06/07` finals), real weights | 224 | **1.4e-2** | 1.3e-2 |
| every 0.05 rung (fresh 0 / 3, collapsed −9, fresh3's checkpoint) | 1,792 | 9.5e-4 | 6.4e-4 |
| every 0.1 rung | 1,792 | **0.11** | 7.5e-3 |
| Lane K, K9: the learner forward at TF32, 147,456 rollout rows | — | **0.040** | p99.9 1.17e-3 |

- **The 0.1 rungs are a second TF32 defect.** In 5 of 138 groups the compiled graph's error against
  fp32 is 6–24x eager's own TF32 error. The TF32 log-prob rule (≤ 4x) refuses that, so a CORRECT
  graph is refused there.
- **The fix is ONE table, `parity_probe.PRECISION_BARS`, keyed by
  `torch.get_float32_matmul_precision()`.** The tie band is 2x the bar.

| precision | log-prob bar | tie band | ladder scale cap |
|---|---|---|---|
| `highest` (fp32) | 1e-3 (the compile gate's, read from the table) | 2e-3 | 0.1 |
| `high` (TF32) | 0.071 = 1.75 x 0.040 (the larger healthy max; K9's multiple) | 0.142 | 0.05 |

- An unmeasured precision (`medium`) is REFUSED.
- The service, the learner compile gate and the judge all read this table.
- At TF32, a collapsed critic that needs a 0.1 rung is REFUSED, naming the skipped rungs. Example:
  the −9 fixture, which fp32 judges at (0.1, +3).
- The greedy check stays secondary at TF32. The TF32 log-prob rule (≤ 4x eager's own error) is
  unchanged and still judges every row.

**Confirmed end to end (2026-09-30, GPU).** A fresh `--arch production --env-core rust
--matmul-precision high` launcher run (`~/gen3ai_archive/vacuous_parity/gpu_smoke_tf32`, 16 envs,
throwaway):
- T2 came up in 271.8 s: 28 slots × buckets (8, 16), graph backend, every slot × bucket and the
  concurrent gate judged under TF32.
- The eval core came up.
- The learner compile gate PASSED under TF32 (features e_comp 6.1e-3 ≤ 4 x 2.3e-2 + 1e-4).
Before this change, the same TF32 startup was refused by the greedy check. The run was bounded at
1150 s inside the GPU lock, so no update was reached.

**The fix** (program doc T2 section, "FLAT weights"; Decision record 2026-09-30):
- The DECLARED ladder `ServiceSpec.perturb_ladder` (default `parity_probe.PERTURB_LADDER`). The
  service judges at the first informative rung, then the real weights with the guard waived. No
  rung, or an empty ladder, ⇒ `VacuousParity`.
- Every verdict records `ParityReport.path` (seed and scale), and `stats()["parity_paths"]` counts
  them.
- `NonFiniteWeights` refuses NaN / Inf before any slot is touched. Before this, a NaN load came out
  as a `VacuousParity`, and then as a `RuntimeError` from the bit-exact restore check (NaN ≠ NaN).
  That is not a `ServiceError`, so the service was NOT poisoned, and the slot held the NaN weights.
- `main.exit_codes` maps `ParityFailure` → `FATAL_CONFIG` and `NonFiniteWeights` →
  `FATAL_NONFINITE`, so the launcher stops.
- The learner compile gate and the opponent compile gate walk the same ladder.

**Teeth.** A served path that DOUBLES each row's win-logit deviation from the batch mean (a
batch-coupled value miscompile) leaves the collapsed −9 critic's real V unmoved, so the waived
real-weights check passes it (asserted). The ladder's informative rung REFUSES it as a real
`ParityFailure`, not a vacuity refusal. That holds on T2 (CPU eager), and the learner gate refuses
a doubled win logit. The fresh-weights pointer teeth (`_temperature_bug`,
`_pointer_cell_miscompile`) are unchanged. Every new test fails on revert of the part it names:
- ladder → its first rung alone: 6 tests;
- the guard waived on the real pass: 7;
- the finite check: 2;
- the exit mapping: 3.
