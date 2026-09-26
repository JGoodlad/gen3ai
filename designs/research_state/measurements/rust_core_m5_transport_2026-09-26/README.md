# Rust core M5, Phase A — the transport benchmark (2026-09-26)

**What was asked:** program §2 M5 registers a benchmark to choose M5's transport: in-process FFI
(EnvPool-style, where Rust steps N envs on its own threads with the GIL released and fills NumPy
arrays) against a separate Rust env process (shared memory, one signal per batch). **Owner direction
(2026-09-26, via the orchestrator):** do not pick one winner. Carry BOTH front ends over ONE Rust env
core, measure each per CONSUMER SHAPE, and recommend a default per consumer. The plan this record
feeds is `designs/endstate/program_rust_core.md` §2 M5.

**Verdict.** The transport does not decide M5. With Python out of the per-decision loop, a batch of
48 envs costs **~1.05 ms of core work**. The front end adds **6.9 µs (FFI) or 14.1 µs (process)** to
that. Per-decision cost and throughput are **equal within noise at N = 48** (proc/FFI **0.994
[0.943, 1.026]**, not detected). The two front ends are **byte-identical** on the same seeds. The
default front end per consumer is therefore decided by **crash isolation, start-up, debugging and
the stamp hazard**, not by speed (§4).

## 1. The prototype (measurement scaffolding, not production)

`proto/` is a std-only Rust crate (no PyO3, no libc crate), path-dependent on the port and built
into its own `proto/target/`:

| part | file | holds |
|---|---|---|
| **the ONE env core** | `proto/src/core.rs` | N battles on T persistent worker threads. Each env holds a live `BridgeSession` and one PARSE chain per side (`parse_root_unrecorded` → `parse_advance_lean` → `encode`, the exact chain `sim_bridge`'s `core_obs` ships to training today, program §6c). The mapper is `present::choice_tokens`; the mask is `present::mask`. The core auto-resets (EnvPool semantics), quarantines a refused battle (§3), and runs `successors(k)` (a step-built root from a snapshot, k children, each encoded). |
| front end 1: **FFI** | `proto/src/ffi.rs` + `m5_loader.FfiPool` | a `cdylib` with a C ABI. Python loads it with `ctypes.CDLL`, which releases the GIL for every call. |
| front end 2: **process** | `proto/src/bin/m5_envproc.rs` + `proto/src/shm.rs` + `m5_loader.ProcPool` | a child process mapping a `/dev/shm` file (a hand-declared `mmap`). One opcode byte in and one status byte out per batch. |
| the **build stamp** | `proto/build.rs` + `m5_loader.check_stamp` | commit + FNV-1a-64 over the `(path, git-blob-id)` listing of every `.rs` source that reaches the binary. Both front ends REFUSE a mismatch at load. |

**The column contract** (caller-allocated, row-major): `obs f32[N][2][OBS_DIM]`, `mask u8[N][2][11]`,
`need u8[N][2]` (the caller must act for that side), `reward f32[N]` (win indicator), `done u8[N]`,
and the input `action i32[N][2]`. In the T2 shape BOTH sides' rows go to the caller, so the Rust env
never runs a network. The alternative is a seeded random opponent INSIDE Rust (the scripted-bot shape).

**Policies:** a seeded NumPy random policy over the mask, for every side with `need = 1`.
**Teams:** the port's bridge corpus, 46 real gen3ou teams, with a random pair per episode.
**Clock config:** the default. **Not in the prototype:** training labels, the stall forfeit
(a turn-300 cap stands in), T2 inference, bots, and the info dict.

## 2. Parity — the two front ends are ONE implementation (`parity.py`, `results/parity.txt`)

Same pool seed + same seeded actions → **every column byte-identical at every step**:

- FFI T=1 == FFI T=4 == proc T=1 == proc T=4 == a FFI rerun.
- Checked over 300 steps × 12 envs in both opponent modes: 6,841 T2-shape decisions and 36 episode
  ends, plus 3,600 bot-shape decisions and 35 episode ends.
- `successors(k = 48)` gives identical bytes on 60 roots.
- Teeth: a different seed differs.

So thread count, front end and rerun are all invisible in the output. That makes the rerun/replay
consumer a determinism property of the core, not a front-end choice.

## 3. The numbers

**The box:** 16 cores. `nice` 15 (the shell's 5 + the script's 10). The contention factor
(`utils.contention`) was **1.0** before and after every shape, and load1 was 0.4–3.9, most of it this
benchmark's own threads. **The live arm was NOT running.** `ai_v14_01_base` had only reached its
pre-launch smoke (`models/smoke_ai_v14_01_base_0926`), GPU 0 %. These are therefore quiet-box
numbers. The between-front-end ratios are the registered quantity, and interleaving controls for
drift; the absolute µs will rise under the live arm's contention.

**Method:** both pools are built once and stay alive. The blocks alternate A B / B A. Each shape
records the per-pair ratio of block medians, with a 95 % bootstrap CI (10,000 resamples). The timed
quantity is `step()` alone: the core plus the transport. "Transport" is the caller's wall minus the
core's own wall. The rule is WARN, never stretch.

| consumer shape | N / threads | FFI | process | proc / FFI | transport per call (FFI / proc) |
|---|---|---|---|---|---|
| **training** (T2 shape: both sides' rows out) | 48 / 8, 10 × 1000 batches, 91.1 decisions/batch | **12.14 µs/decision** [11.60, 12.41] = 82.4 k/s | **11.96** [11.61, 12.31] = 83.6 k/s | **0.994 [0.943, 1.026]** n.d. | 6.9 / 14.1 µs of a 1.05 ms batch (0.65 % / 1.35 %) |
| training, **bot** opponent in Rust | 48 / 8, 48 decisions/batch | 22.8 [22.4, 23.5] | 23.7 [22.7, 23.9] | 1.022 [0.994, 1.048] n.d. | 7.1 / 14.4 µs |
| **eval** (fixed opponent, rows out) | 16 / 4, 30.3 decisions/batch | 19.92 [19.82, 19.95] | 20.10 [20.02, 20.21] | **1.012 [1.006, 1.014]** (proc +1.2 %, detected) | 5.9 / 11.9 µs of 567 µs |
| **search** (`successors(64)` from a live root) | 1 / 1, 113 roots | 8.72 ms/call [8.46, 9.20] = **136 µs/successor** | 8.46 [8.13, 8.71] | **0.952 [0.932, 0.965]** (proc −4.8 %, detected) | 12.7 / 26.6 µs per call (0.15 % / 0.3 %) |
| **one-off**, start-up to the first row (fresh interpreter, after `import numpy`) | 1 / 1, 10 reps | **24.1 ms** [24.0, 24.4] | 26.1 [26.0, 26.3] | +2.0 ms (spawn + shm + stamp line) | stamp check 2.7 ms (both) |
| **one-off**, per-decision latency | 1 / 1 | 64.5 µs [62.0, 65.2] | 62.3 [60.4, 63.8] | 0.972 [0.962, 0.985] (proc −2.8 %) | 2.3 / 5.0 µs |
| **rerun / replay** | — | byte-identical to a first run and across front ends and thread counts (§2) | | | |

**Thread scaling of the core** (FFI, N = 48, T2 shape, `results/threads.json`):

| threads | µs per decision | decisions/s | speed-up |
|---|---|---|---|
| T = 1 | 69.0 | 14.5 k | 1.0× |
| T = 2 | 35.0 | 28.6 k | 2.0× |
| T = 4 | 19.4 | 51.6 k | 3.6× |
| T = 8 | 11.9 | 84.3 k | 5.8× |

**GIL** (`results/gil.json`). A second Python thread spinning in the host kept its full rate with
either front end: 24.77 M it/s (FFI) and 24.65 M (proc), against 24.78 M with nothing stepping. Both
release the GIL while the core runs. But the stepping thread itself fell to ~48 batches/s, because
it must re-take the GIL after every call and loses up to the 5 ms switch interval each time. That
cost is **identical for both front ends** (FINDING F-M5-3 below).

**Crash isolation** (`results/crash_isolation.txt`). A SIGKILL to the env process mid-run surfaced
in the parent as a typed `m5_envproc DIED (rc=-9) — the parent survives`, and a respawned child
served rows. Under FFI the core CATCHES a panic (per env: the batch fails with the env's input log,
or the battle is quarantined). What no in-process code can catch — an abort, a stack overflow, OOM,
a fault in `unsafe` — ends the Python host.

**The stamp** (`results/stamp_teeth.txt`). Appending a comment to one source file without
rebuilding made BOTH front ends refuse at load (`StampMismatch`, naming both hashes). The process
front end checks its OWN stamp line, independent of the `.so`. Restoring the file restored both.
Recomputing the hash at import costs 2.7 ms.

### Against today's path (a cross-record comparison, NOT an A/B)

The cutover's throughput A/B (`rust_core_cutover_2026-09-24/README.md` §6) put today's full cycle
per trainee decision at **1.07–1.19 ms** of serialized CPU. That is our Python plus the `sim_bridge`
child, with the tracker fold paid twice, and it excludes the opponent's own Python encode and B = 1
forward in the same worker. The core here spends **69 µs of single-thread CPU per decision**, with
both sides encoded (T = 1 above): roughly **15–17× less CPU per decision**.

That dividend comes from taking Python OUT of the per-decision loop. The transport choice is ≤ 1.4 %
of it. The prototype lacks the labels, the stall bookkeeping and the info dict (§1), so the ratio is
an upper bound until Lane C / D land.

## 4. Default front end per consumer (the recommendation)

**Rule:** use the **process front end when the Python host holds state a core fault must not
destroy**, and **FFI when the host is itself disposable**. A worker, a probe or a meter process can
die and be restarted without loss, so FFI's crash exposure is already paid for by that process
boundary.

| consumer | default | why |
|---|---|---|
| **training rollout** (N = 48, the learner) | **process** | Speed is equal: 0.994 [0.943, 1.026]. The learner holds the optimizer, the rollout buffer and the GPU context, and a core abort or segfault under FFI would take all three. A dead child is a typed error plus a respawn, which is today's per-env child semantics, so the launcher's crash handling carries over. The child can be attached with gdb / py-spy separately from the learner. Start-up (+2 ms) is irrelevant. |
| **eval** (moderate N, fixed opponent) | **process** in-trainer; **FFI** inside a disposable eval worker | The process costs +1.2 %, which is immaterial. Run inside the trainer, eval inherits the learner's isolation need. A sharded eval worker (`eval_sharding`) is already a disposable process. |
| **search** (many short forks from a root) | **FFI** for offline search (prober, `search_dividend`, meters); **process** only if search runs inside the learner | Transport is 0.15–0.3 % of a call. The in-process tree keeps version HANDLES as plain pointers, and a future `Leaf` hook calls T2 in the same address space. The measured −4.8 % favours the process but lives inside the core's own wall, not the transport (cause UNVERIFIED, F-M5-4), and does not decide. |
| **one-off probes / meters** (N = 1–4) | **FFI** | 2 ms faster to the first row. No child to manage or reap, no `/dev/shm` file to clean. One process under pdb / py-spy. The host is disposable. |
| **reruns / replays** | **FFI** (either is legal) | Byte-identical by construction (§2), so the choice is convenience. Every rerun stamps the core's source hash. |

**The build-stamp hazard applies to BOTH front ends.** The 09-09 incident was a stale *binary*
(`POKESIM_SIM_BRIDGE_BIN`), and a stale `.so` first on `sys.path` is the same class. So the M5 build
ships ONE stamp that both front ends refuse on. FFI additionally needs the module name pinned to an
absolute path (never `sys.path` resolution).

**The cost of carrying two front ends instead of one.**
- In the prototype the second front end is ~140 lines of Rust (`m5_envproc.rs` + `shm.rs`), ~110 lines
  of Python (`ProcPool`), and the parity test.
- In the build that is **+1.5–2 agent-days**: Lane B, plus the FFI == proc parity gate in the routine
  tier. That is ≈ +6–8 % of M5's build.
- The running cost is that every new op touches both front ends. The build removes it by routing both
  through ONE `core::dispatch(op, cols)` entry, so a front end forwards an opcode and a column set and
  never names an op's semantics.

## 5. Findings

- **F-M5-1 — a shared READING-class refusal the core surfaced, NOT a core-vs-Python divergence**
  (classified by execution; `repro/hp_tracker_env46.script`).
  - Both the Python `HiddenPowerTracker` and the Rust `hp_belief` refuse ("all candidates
    eliminated").
  - The sequence:
    1. A transformed Smeargle is KO'd by a Hidden Power.
    2. Its switch-out clears its temporary types (`poke_env/battle/pokemon.py:665` / `present/mon.rs:917`).
    3. At the next decision both trackers re-read the target's types from the LIVE board
       (`episode_tracker.py:44-72` / `hp_belief.rs:167-176`) and see base Normal.
    4. Normal contradicts the Grass-consistent log.
  - A second, silent effect: a transformed mon's Hidden Power is filed under its OWN species (Transform
    copies the HP type).
  - **Exposure:**
    - `data/teams/` holds no Transform user (824 files), so training cannot reach it.
    - The ladder-usage corpus holds 20 Transform Smeargles, all outside the 600 `MILESTONE_LADDER_KEYS`
      teams — so M7 / ladder play would.
  - The benchmark's own 5 quarantines (3 in the T2 shape, 2 in the bot shape, over ~50 k env-steps)
    come from the corpus's `25_transform_foe…` pair. Only env 46's was banked and classified; the
    other four are UNVERIFIED as the same class.
  - Proposed fix (NOT applied — a TRAINING-INPUT change on both paths): snapshot the target's types
    and ability into `DamagingMoveEvent` when the move fires, as `target_status` already is. Pin it
    with a test built on this script that fails on revert.
- **F-M5-2 — the M5 estimate is ~4× short.** Program §2 sizes M5 at 5–8 agent-days on the premise
  that the Python path is already only an oracle. The deletion pass (`56837827`) found it is not:
  labels, reward bookkeeping, policy opponents, bots and eval all still run in Python. The lane plan
  (program §2 M5) totals **24–33 agent-days**, T2 included.
- **F-M5-3 — the GIL re-take, not the release, bounds a Python rollout thread.** With one CPU-busy
  Python thread in the host, the stepping thread fell from ~950 to ~48 batches/s on BOTH front ends.
  The learner's rollout loop must not share its interpreter with a CPU-busy Python thread (or must
  shorten `sys.setswitchinterval`). This is a Lane G constraint; today's `SubprocVecEnv` loop has the
  same exposure.
- **F-M5-4 — the core ran 2.8–4.8 % faster OUT of the Python process** in the single-threaded shapes
  (N = 1, search); it was not detected at N = 48 / T = 8. The difference is inside the core's own
  wall, not the transport. Cause UNVERIFIED (hypothesis: allocator / heap state of a Python host).
  The build should re-measure it before any claim.
- **F-M5-5 — the prototype's rows are the production chain's, but were not byte-diffed against
  `sim_bridge`'s `__OBS__`** for the same battle. Lane 0's first gate is exactly that diff.
- **F-M5-6 — `ctypes`, not PyO3, keeps the crate std-only.** A C ABI plus `ctypes.CDLL` gives
  GIL-free calls with no new dependency, which the port's "std-only" rule favours. The GIL release
  and the zero-copy columns were measured. The cost is hand-written signatures, so Lane A generates
  them from one table.

## 6. Reproduce

```bash
cd designs/research_state/measurements/rust_core_m5_transport_2026-09-26
(cd proto && CARGO_TARGET_DIR=$PWD/target nice -n 10 cargo build --release)
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
nice -n 10 $PY parity.py 300 12                       # exit 0 = byte-identical
for s in "train --pairs 10 --steps 1000" "trainbot --pairs 10 --steps 1000" \
         "eval --pairs 10 --steps 1000" "threads --pairs 3 --steps 500" \
         "search --pairs 3" "oneoff --pairs 10 --steps 300" gil; do
  nice -n 10 $PY bench.py $s; done                    # -> results/<shape>.json
```

The stamp is taken at the commit the build ran at (`663d11bc`) plus the working tree's sources. Every
`results/*.json` carries it.
