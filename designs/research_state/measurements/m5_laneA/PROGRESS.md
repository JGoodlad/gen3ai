# M5 Lane A — the FFI front end: PROGRESS (resume point)

Lane A of `designs/endstate/program_rust_core.md` §2 M5. Owns `src/rust_env/src/ffi.rs` and
`src/utils/rust_env/ffi.py` (+ their tests `ffi_test.py`, `ffi_integration_test.py`,
`src/rust_env/tests/ffi_reference_test.rs`, and the descriptor `src/utils/rust_env/ffi_benchmark.py`).
Hand-off: ONE line in Lane 0's `src/rust_env/src/lib.rs` (`pub mod ffi;`). Seed: the Phase-A
prototype's `proto/src/ffi.rs` + `m5_loader.py` (`../rust_core_m5_transport_2026-09-26/`).

## How to build / test (worktree-local target only)

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m utils.rust_env.ffi --write               # after any change to FUNCTIONS (ffi.py)
python3 -m pytest src/utils/rust_env/ffi_test.py src/utils/rust_env/ffi_integration_test.py -q
python3 src/utils/rust_env/ffi_benchmark.py        # gate ④, a descriptor (release build)
```

The integration test builds `src/rust_env/target/selfcheck/libpokesim_env.so` itself
(`cargo build --lib --profile selfcheck --features emission-selfcheck`, `CARGO_TARGET_DIR` = the
checkout's own `src/rust_env/target`).

## What is built

- **The API** (`FfiCore`): `FfiCore(spec_json)` = `rust_env_new` (→ `Core::new(Spec::from_json)`)
  + every column allocated from `columns.allocate(n, obs_dim)` (n and obs_dim read FROM THE LIBRARY)
  + `rust_env_freeze` with the address array taken ONCE; then `reset()` / `step()` /
  `dispatch(op)`; `counters()`, `after_freeze()`, `bank()`, `close()`. A non-OK status raises the
  `protocol.py` class decoded from `rust_env_last_error()` (this thread's `DispatchError::json`).
  Output columns are read-only NumPy views on the Python side (the core writes through its pointer).
- **Generated signatures:** `FUNCTIONS` in `ffi.py` is the one table. `render()` writes the
  `extern "C"` wrappers into the marked region of `ffi.rs`; each wrapper calls `imp::<name>` with
  the same arguments (a drifted Rust type does not compile) and `load()` sets every ctypes
  `argtypes` / `restype` from the same rows. The table's id (`sig_id()`, which also covers the column
  schema id and column count) is compiled in and compared at load. `ffi_test.py` (routine) pins
  the region.
- **Load:** ABSOLUTE path only, `RTLD_LOCAL`; before anything is returned: every symbol present, the
  FFI table id, the column schema id + `N_COLUMNS`, then `stamp.check_stamp` (sources, schema,
  data dir, and `nan_poison` when demanded). Nothing is dispatched on a refused build.
- **Panics:** every export runs inside `guard` (`catch_unwind`); a panic is status `PANIC` + a
  `CorePanic`. A panic inside a handle's locked section POISONS the handle (later calls →
  `LifecycleViolation`). A handle is single-caller: a concurrent call is refused (`try_lock` →
  `LIFECYCLE`), never a race. `rust_env_panic_probe(h, kind)` is the test hook (exported in every
  build; it only panics when called).

## Gates

| gate | test | result |
|---|---|---|
| ① FFI == in-Rust, byte for byte | `ffi_integration_test.py` — Lane 0's gate-① runs RECORDED in Rust (`tests/ffi_reference_test.rs`, the same staging RNG + random policy as `sim_bridge_parity_test.rs`, T = 1), REPLAYED through ctypes at T = 3; every output column after every op, the deterministic counters and the bank | **PASS**: bridge corpus 6 × 400 (401 ops, 4,543 rows, 28 finished episodes); ladder COMMIT tier 8 × 600 (601 ops); F-M5-1's banked refusal + 60 steps (1 quarantine; bank byte-equal); teeth: one flipped obs byte fails the replay |
| ② panic → typed error, no crash | `test_gate_2_…` (in a SUBPROCESS) + `ffi::tests::a_panic_never_crosses_the_boundary` (cargo, routine via `core_cargo_test.py`) | **PASS**: string and non-string payloads → `CorePanic`; a panic in the locked section → `CorePanic`, then `LifecycleViolation` ("POISONED"); the process exits 0 |
| ③ declared lifecycle through FFI | gate-① runs assert every `*_AFTER_FREEZE` == 0 read through `rust_env_counters`; `test_gate_3_…`: STEP before RESET, a second freeze and a dispatch naming another column set → `LifecycleViolation`, the rebind COUNTED (`COLUMN_REBINDS_AFTER_FREEZE` = 1), an output column is read-only, a closed core refuses | **PASS** |
| ④ throughput (descriptor) | `ffi_benchmark.py` | see below |
| stamp at load | `test_the_stamp_is_checked_at_load_before_any_op` (a self-check build demanded as release → `StampMismatch`); `ffi_test.py`: a relative path and a missing file refused | **PASS** |

## Gate ④ — throughput (DESCRIPTOR)

Measured 2026-09-29 13:23–13:25 PT, release build, N = 48, both sides' rows, bridge corpus;
2 rounds interleaved in-Rust / FFI / FFI / in-Rust; load1 2.7–5.0 on 16 cores (no training run live;
`utils.contention`: "box looks idle"). Warn, never stretch: these numbers are labelled, not corrected.

| shape | in-Rust (`bench_test`, 8 reads) | FFI wall (4 reads) | FFI transport |
|---|---|---|---|
| T = 8 | 15.9–17.8 µs/row | 17.1–17.9 µs/row | 4.7–5.0 µs per dispatch |
| T = 1 | 74.9–82.0 µs/row | 78.0–90.4 µs/row | 5.3–7.7 µs per dispatch |

- **Transport** (wall minus `CORE_NS_LAST`) is ~5 µs per dispatch against ~1.5 ms of core work per
  dispatch at T = 8 (~91 rows per STEP): **≈ 0.3 %**. Phase A's prototype read 6.9 µs.
- FFI wall ≈ FFI core per row to 0.1 µs, so the FFI-vs-in-Rust row difference (≈ +4 % at T = 8)
  sits INSIDE the core, not in the boundary: the two runs play DIFFERENT battles (a NumPy random
  policy vs `bench_test`'s Rust one) and the ranges overlap. No ratio is claimed; no CI computed.
- Lane 0's F-L0-7 (the core's cost only read on a loaded box) now has a quiet-box read: **16–18
  µs/row at T = 8, 75–82 at T = 1** (Phase-A prototype: 12.1 / 69.0, no labels, no refusal policy).

## Open findings

- **F-LA-1 (for Lanes B / E / G): which failures poison the pool.** `Core::dispatch` does NOT
  poison on a `LIFECYCLE` status (STEP-before-RESET, a second freeze, a rebind are refused and the
  pool stays usable); every OTHER failure (FAULT / CALLER / PANIC / BUDGET) poisons it. A front end must therefore build
  a NEW core after any non-lifecycle error — including a `CallerError` for one illegal action.
- **F-LA-2: `CORE_NS_*` counters are wall-clock** and are the only nondeterministic cells in the
  output columns; any byte-equality gate over `counters` (Lane B's FFI == process gate) must mask
  indices 5–6, as `ffi_integration_test._replay` does.
- **F-LA-3: `dlopen` caches by path** — a rebuild in place is NOT re-read in a process that already
  loaded the library; a long-lived host that rebuilds must restart (or load a copy at a new path).
  The stamp check runs on the image actually loaded, so a stale image is still REFUSED if its
  sources differ — but only at the first `load()`; `load()` is not re-run per op.
- **F-LA-4: a panic INSIDE a worker env** (Lane 0's per-env `catch_unwind` in `pool.rs`) is NOT
  exercised through the FFI — there is no injection hook in the core (Lane 0's files), so gate ②
  covers the boundary guard (calling thread + locked section) and the `PANIC` status → `CorePanic`
  mapping, not a real worker-thread panic end to end. A test-only panic op in the core would close
  it (Lane 0 / J).
- **F-LA-5: `rust_env_panic_probe` ships in every build** (release included). It panics only when
  called and is in the FFI table (so the stamp and `sig_id` cover it); if a production consumer must
  not carry it, gate it behind a cargo feature — which would then need a second stamp field.
- **F-LA-6: one handle is single-caller** (`try_lock` → `LIFECYCLE` on a concurrent call). A
  consumer that wants two Python threads stepping must hold two cores. `FfiCore` also serialises
  with its own lock, so the Rust refusal is a second line only.
- **F-LA-7: the process front end (Lane B) can reuse the SAME recorded traces**
  (`tests/ffi_reference_test.rs` → `spec.json` / `trace.bin` / `bank.json`) for its byte-equality
  gate, and `ffi_integration_test._replay`'s loop is front-end-agnostic apart from `FfiCore`.
- **F-LA-8: the error JSON is the only error channel, and `rust_env_last_error` is THREAD-LOCAL** — a caller must read it on the thread that got the status
  (`FfiCore` does).
