# M5 Lane B — the process front end: PROGRESS (resume point)

Lane B of `designs/endstate/program_rust_core.md` §2 M5. Owns `src/rust_env/src/bin/rust_env_proc.rs`,
`src/rust_env/src/shm.rs` and `src/utils/rust_env/proc.py` (+ their tests `proc_test.py`,
`proc_integration_test.py`, and the descriptor `src/utils/rust_env/proc_benchmark.py`). Hand-off: ONE
line in Lane 0's `src/rust_env/src/lib.rs` (`pub mod shm;`). Seed: the Phase-A prototype's
`proto/src/shm.rs` + `proto/src/bin/m5_envproc.rs` + `m5_loader.ProcPool`
(`../rust_core_m5_transport_2026-09-26/`). Nothing of Lane A's (`ffi.rs` / `ffi.py`) was edited; the
gate reads them.

## How to build / test (worktree-local target only)

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m utils.rust_env.proc --write              # after any change to the wire table (proc.py)
python3 -m pytest src/utils/rust_env/proc_test.py src/utils/rust_env/proc_integration_test.py -q
python3 src/utils/rust_env/proc_benchmark.py --out <f.json>   # gate ⑥, a descriptor (release builds)
```

The integration test builds `src/rust_env/target/selfcheck/{libpokesim_env.so, rust_env_proc}` itself
(`cargo build --lib --bin rust_env_proc --profile selfcheck --features emission-selfcheck`,
`CARGO_TARGET_DIR` = the checkout's own `src/rust_env/target`). Routine tier (`sim`, `integration`,
not `slow`): ~7 s warm.

## What is built

- **The API** (`ProcCore`), the same surface as `FfiCore`: `ProcCore(spec_json)` → spawn + handshake +
  INIT (`Core::new` + the mapped columns + `freeze`); then `reset()` / `step()` / `dispatch(op)`,
  `counters()`, `after_freeze()`, `bank()`, `close()`, plus `respawn()` and `pid`. `cols` are NumPy
  views of the shared mapping (outputs read-only) and STAY VALID across a respawn.
- **The transport:** ONE anonymous `memfd` (`os.memfd_create`, `MFD_CLOEXEC`), handed to the child by
  fd (`pass_fds`); layout = a 4 KiB header + every column of `columns.COLUMNS` in table order at
  64-byte-aligned offsets (`proc.layout` / `shm::layout`). The child writes the header (magic, wire id,
  n, obs_dim, column count, total, every offset); the host compares it with its own layout before
  the first op. One request byte per op on the child's stdin; one reply frame per op on its stdout,
  `[status u8][len u32 LE][payload]` (a failure's payload = `DispatchError::json`, `BANK`'s = the bank).
  Every non-control byte is forwarded to `Core::dispatch` unread, so the core, not the front end,
  decides what an opcode means (an unknown one is the core's `LIFECYCLE`).
- **The wire is GENERATED** (Lane A's pattern): the control bytes (`INIT`, `BANK`, `QUIT`,
  `PANIC_PROBE`, `ABORT_PROBE`), the header words, `HEADER_BYTES` / `ALIGN` / `MAGIC` are one table in
  `proc.py`, rendered into the marked region of `shm.rs`; `wire_id()` (covering the column schema id
  and column count) is compiled in and compared at the handshake; `proc_test.py` (routine) pins the
  region and asserts no control byte shadows a core opcode.
- **The handshake, before any op:** the child's FIRST output, printed before it reads a byte:
  `rust_env_proc\twire=…\tobs_dim=…\tn_columns=…\tschema=…\tstamp=…`. The host checks wire, schema,
  column count, then `stamp.check_stamp` (sources, schema, data dir, `nan_poison` when demanded); a
  refusal closes the child's stdin (it exits having run nothing) and reaps it.
- **Death:** EOF on the reply pipe (or `EPIPE` on the request pipe) IS the death signal → the child
  is reaped (returncode + signal name in the message) → `CoreProcessDied`; with `auto_respawn`
  (default) a fresh child is spawned on the SAME mapping, re-stamped, the output columns zeroed,
  BEFORE the error is raised (`respawned=True`; the caller RESETs). A failed respawn is still a
  `CoreProcessDied` (`respawned=False`, `__cause__` = the refusal). `op_timeout` (default None =
  wait) bounds one op: past it the child is SIGKILLed by its PID and handled as a death
  (`CoreProcessTimeout`); `startup_timeout` (default 600 s) bounds handshake + INIT.
- **Panics in the child:** every op runs inside `catch_unwind`; a panic is status `PANIC` →
  `CorePanic`, and POISONS the child's core (later ops `LIFECYCLE`), as a panic in the FFI's locked
  section poisons that handle. `respawn()` recovers.
- **Parent death:** the child exits on stdin EOF (no `PR_SET_PDEATHSIG` — see F-LB-4).

## Gates

| gate | test (`proc_integration_test.py` unless named) | result |
|---|---|---|
| ① FFI == PROCESS, byte for byte | Lane A's in-Rust recordings (`tests/ffi_reference_test.rs`, T = 1) REPLAYED through BOTH front ends side by side at T = 3 (quarantine run T = 1): after every op every output column of the process equals the FFI's AND the recording; counters too (`CORE_NS_*` masked, F-LA-2); the bank. Plus the ERROR channel: a bad spec, STEP before RESET, an illegal action (and the poisoned op after it) → the same class, message, kind, env and input log through both | **PASS**: bridge corpus 6 × 400 (401 ops), ladder COMMIT tier 8 × 600 (601 ops), F-M5-1's quarantine + 60 steps (bank byte-equal, 1 refusal); teeth: one flipped recorded byte fails |
| ② SIGKILL → typed error + respawn, no hang | SIGKILL between ops and DURING an op (timer thread), `ABORT_PROBE` (SIGABRT), a SIGSTOPped child past `op_timeout` = 2 s, `PANIC_PROBE` + `respawn()`, `auto_respawn=False`, and a binary SWAPPED for a forged one before the respawn | **PASS**: `CoreProcessDied` / `CoreProcessTimeout` with `returncode` −9 / −6, seen at once (EOF; the between-ops case in ~ms); a new pid, re-stamped, outputs zeroed, `STEP` before `RESET` refused (a FRESH core), then plays; the swapped binary → `CoreProcessDied(respawned=False)` caused by `StampMismatch`, and the forged child was sent 0 bytes |
| ③ no leaked segment | after a normal close, a startup error, a poisoned core, a SIGKILLed-then-respawned child, and a SIGKILLed PARENT (a subprocess): no new `/dev/shm` entry naming `rust_env`/`gen3ai`, no `memfd:gen3ai_rust_env` fd or mapping in this process — or, for the parent kill, in ANY process of this user (by inode) — every child exited | **PASS**; the census SEES the live memfd first (no vacuous pass) |
| ④ the stamp refuses a mismatched child before any op | a self-check child demanded as release (`nan_poison=False`) → `StampMismatch`; a forged child (right wire, wrong `src`) → `StampMismatch`, a forged wire → `ProcLoadError`; each forged child logs every byte it is sent: **0**; `proc_test.py`: each handshake field refused | **PASS** |
| ⑤ the declared lifecycle through the process | every `*_AFTER_FREEZE` (the core's 4 + the front end's `PROC_SPAWNS_AFTER_FREEZE`) = 0 after each gate-① run; STEP before RESET and an unknown opcode → the core's `LifecycleViolation`, which does NOT poison (F-LA-1: a RESET then plays); outputs read-only; a closed core refuses. A rebind is not EXPRESSIBLE through this front end (the child binds the mapping once) | **PASS** |
| ⑥ throughput vs FFI (descriptor) | `proc_benchmark.py` | below |

**Teeth (mutation-checked, 2026-09-29):** the handshake stamp check removed → all 3 stamp tests FAIL
(the forged-child tests carry `startup_timeout=20`, so a missing refusal fails in seconds rather
than waiting on a child that will never answer INIT); the respawn's output zeroing removed → the
SIGKILL test FAILS.
The census also caught a real leak in the TESTS on its first `-n 2` run: a replay that failed (the teeth test) left its `ProcCore` alive in the traceback — `_replay_both` now closes both cores on failure, and gate ③ compares against a baseline census rather than assuming a clean process.

## Gate ⑥ — throughput (DESCRIPTOR)

Measured 2026-09-29 ~15:00 PT, release builds of both front ends, N = 48, both sides' rows, the bridge
corpus, 3 rounds interleaved FFI PROC PROC FFI × 5 blocks × 200 STEPs; load1 2.2–3.0 on 16 cores (no
training run live; `utils.contention`: "box looks idle"). The two front ends play the SAME battles
(same spec and seeds; a CRC over every obs column after every STEP is asserted equal), so blocks are
PAIRED. Raw: [`throughput_2026-09-29.json`](throughput_2026-09-29.json).

| shape | FFI wall | PROC wall | transport FFI / PROC (µs per dispatch) | proc / FFI (paired, 95 % bootstrap over 30 blocks) |
|---|---|---|---|---|
| T = 8 | 15.21 µs/row | 15.51 µs/row | 4.9 / 15.4 | **1.020 [1.007, 1.034]** |
| T = 1 | 72.15 µs/row | 70.16 µs/row | 4.7 / 21.2 | **0.967 [0.958, 0.976]** |

- **Transport:** the process costs ~10 µs (T = 8) to ~16 µs (T = 1) more per dispatch than the FFI —
  a pipe round trip plus two context switches — against ~1.4 ms of core work per T = 8 STEP (~91
  rows): **≈ 0.7 %**. Phase A read 14.1 µs (process) and 6.9 µs (FFI).
- **The ratios are CORE-side differences, not transport:** at T = 1 the process is FASTER (its
  `CORE_NS` per row is lower too), at T = 8 ~2 % slower, both well outside what the transport
  explains. Plausible sources (UNVERIFIED): the child's own address space (allocator, no Python heap
  in the caches), scheduler placement of the child's workers. Phase A's training-shape ratio was 0.994
  [0.943, 1.026].
- ⚠️ The CI is over blocks, which are autocorrelated within a run (only 6 runs per front end per
  shape): read it as optimistic. **Verdict for M5: the transport does not decide the front end
  choice** (Phase A's conclusion holds), and PROC at N = 48 is within ±3.5 % of FFI either way.

## Open findings

- **F-LB-1 (for Lane G): a respawn is COUNTED, not hidden.** `after_freeze()` carries
  `PROC_SPAWNS_AFTER_FREEZE`, so a trainer that asserts every `*_AFTER_FREEZE` == 0 FAILS after a
  child death, by design (a new process in steady state). Lane G must decide the policy: treat a
  respawn as a typed, logged, BUDGETED event (e.g. a respawn budget declared at startup, like the
  refusal budget) and discard the in-flight rollout segment of those envs — the fresh core's episodes
  restart from RESET, so every env's episode is cut at the death (truncation, not termination).
- **F-LB-2 (for Lane G): after a respawn the caller MUST RESET before STEP**, and the staged
  `ep_team` / `ep_seed` inputs survive in the mapping (the respawned core reads them at RESET). The
  outputs are zeroed, so a stale row can never be read as fresh.
- **F-LB-3 (for Lanes G / E): a non-lifecycle core failure through the process is the same as
  through the FFI (F-LA-1) — it poisons the core.** The process adds a cheap recovery: `respawn()`
  (~tens of ms at N = 2; `Core::new` validates every team of the table by use, so it scales with the
  team table — UNMEASURED at 800 teams). `auto_respawn` fires on DEATH only, never on a typed error.
- **F-LB-4: parent death is detected by stdin EOF, not `PR_SET_PDEATHSIG`.** A child busy inside a
  long op finishes it, then sees EOF / EPIPE and exits; PDEATHSIG was not used because it fires on
  the death of the spawning THREAD (a trainer spawning from a worker thread would lose its core
  when that thread ended). Gate ③ proves the orphan exits.
- **F-LB-5: `op_timeout` defaults to None (wait forever).** A deadlocked core hangs the caller exactly
  as it would through the FFI; the bound exists (tested) but choosing it is Lane G's (a contention-
  scaled bound, `utils.contention`; a timeout is never a semantic outcome — it is a typed death).
- **F-LB-6: the probes ship** (`PANIC_PROBE`, `ABORT_PROBE`, as Lane A's `rust_env_panic_probe`,
  F-LA-5): harmless unless the host sends the byte; the wire id covers them.
- **F-LB-7: GIL.** Every blocking read/write is `os.read` / `os.write` on raw fds (the GIL is released
  while the core runs); the view writes (`action`, `ep_*`) are plain NumPy stores into the mapping —
  the pipe syscalls order them before the child's read.
- **F-LB-8: `/dev/shm` is never used.** The lane was specified as "/dev/shm hygiene"; the memfd makes
  a leak unrepresentable instead of cleaned up (Decision record, 2026-09-29). A debugger can still
  inspect the mapping through `/proc/<pid>/fd/<n>` of either process.
