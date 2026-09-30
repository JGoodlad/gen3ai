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
| 5 | Lane E's adapter (opponent rows from the core's columns → `submit` per slot, greedy/sampled actions back) — with Lane E | LATER | |
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
- **The double-buffer event wait has NO proven teeth.** The back-to-back-flush check in
  `service_cuda_test` passes with it; a revert-must-fail run of that wait was not done (only the
  capture-stream revert was). It may be unable to catch the race it guards.
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
