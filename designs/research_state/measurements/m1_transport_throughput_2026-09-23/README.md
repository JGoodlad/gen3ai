# Did Rust Core M1 make the PRODUCTION battle transport slower? — the A/B

<!-- A MEASUREMENT record (2026-09-23). Question from the owner before any arm launches from a
post-M1 commit. Nothing in production code was changed; every script here is measurement-only. -->

**Verdict.** The change has a **real but small cost**, and **most of it is not M1's typed builder**.
The Rust `sim_bridge` child does **+13 % CPU per decision** (the CI excludes zero: a FINDING), which
is **about +5 µs per `CHOOSE`, or about +10 µs per trainee decision**. At the env-step level:

- The pinned single-env step reads **+1.6 % wall**. Its CI of +0.4 to +2.2 % excludes zero, so under
  the bands this is formally a FINDING, although the whole CI lies inside ±2.2 %.
- The production-shape 8-worker throughput reads **+0.2 %**. Its CI of −0.7 to +3.6 % straddles zero:
  NO MEASURABLE COST at this resolution.

**Where the time goes.** ≈ 60 % of the Rust delta (+18.8 ± 2.8 ms of +31.1 ms per replay) is the
**one-sided view fold** (`SideObservation::observe`), which roughly doubled in cost with the truth
audit's reading rules (`db2e6515`). `sim_bridge` runs it on every shipped line, but only
`search_driver` / `core_events` ever read it. **M1's typed-`Line` build + render is ≈ 24 %**
(+7.4 ± 1.2 ms). The `attrLastMove` retro-edit re-parse draws **0 samples out of 1,150**.

| | hash |
|---|---|
| **A** (before M1) | `dfab2558` — main, the last commit before `1ffa13bc` |
| **B** (after) | `7c511161` — `origin/main` HEAD when the task started |

⚠️ **`origin/main` moved during the task** (`65f22334` … `afc6807b`, the poke-env R1–R3 reading fixes).
Its only Rust change is a 3-line `set_pstatus` counter reset in `view.rs`, which is not in B's binary.
It sits inside the same fold (`SideObservation::observe`) and adds a string compare per `-status`
line, which is negligible next to the measured delta. It is not measured here.

## Method

- **Two worktrees, two builds, one binary per process.** A was checked out at
  `.claude/worktrees/m1bench-A` (detached), and B is this worktree. `cargo build --release --bin
  sim_bridge` ran into each worktree's OWN `src/rust_sim/target`: there is no symlink and main's
  target was not touched. `verify_bins.sh` prints each binary's sha256 and the
  `CARGO_MANIFEST_DIR` baked into it (the dex path). Each names only its own worktree:
  - A `sim_bridge` sha256 `1fc6263c9b394560…`, baked `…/worktrees/m1bench-A/src/rust_sim`
  - B `sim_bridge` sha256 `7348bbfe9f2bafea…`, baked `…/worktrees/agent-a35e14fb…/src/rust_sim`

  Every run is its own subprocess, and `POKESIM_SIM_BRIDGE_BIN` names exactly one side's binary.
- **One Python tree for both arms (B's).** The training-path Python is identical at A and B:
  `git diff dfab2558 7c511161 -- '*.py'` touches only the search-side `ViewEventFolder` / `view_adapter`,
  the parity harness, and one new resolver in `sim_bridge_bin.py`. Neither benchmark script
  changed. So the only thing varied is the Rust binary.
- **Interleaved, order-alternated.** Pair *k* runs A→B for odd *k* and B→A for even *k*. The
  1-minute load average is recorded at each run's start and end. Every process runs at `nice -n 10`,
  CPU only. There is no port, no `models/` write, and no `data/` change.
- **Statistic.** The per-pair ratio is always read as "cost of B": B/A for times, A/B for a rate. The
  report gives its median and a paired bootstrap 95 % CI on that median (20,000 resamples,
  `analyze.py`).
- **Byte identity was checked on this workload too.** Both arms play the same pinned battles
  (`measured` = 10,141 / 3,036 decisions and `battles` = 128 / 41 are equal in every pair). A and B
  also emit **byte-identical** stdout (20.2 MB, 41 `__END__`, 0 `__ERR__`) on the recorded
  transcript.

## The measurements

| # | bench | what it isolates | pairs | load1 | median cost of B | 95 % CI | B slower |
|---|---|---|---|---|---|---|---|
| 1 | `bridge_impl_throughput_benchmark.py --impl rust --workers 8 --seconds 45` | the production transport shape: 8 worker processes, each with a Gen3Env and a **persistent** child, random legal actions through the real obs/reward stack; aggregate env-steps/s | 8 | 23.6–40.7 | **+0.2 %** (fps A/B 1.0023) | 0.9925 – 1.0360 | 4/8 |
| 2a | `env_step_rust.py` = `trainer_turn_benchmark --pin-battles`, rust bridge, 10,000 decisions | one env's pinned decision loop (wall) | 12 | 1.1–30.6 | +0.1 % | 0.928 – 1.049 | 7/12 |
| 2b | same, 3,000 decisions (more, shorter pairs) | ″ (wall) | 30 | 9.3–40.6 | **+1.6 %** | **1.0037 – 1.0223** | 22/30 |
| 2b | ″ | Rust child CPU (rusage of reaped children) | 30 | ″ | +2.8 % | 1.0157 – 1.0372 | 23/30 |
| 2b | ″ | Python parent CPU (identical code, so a control) | 30 | ″ | +0.6 % | 0.9907 – 1.0171 | 17/30 |
| 3 | `run_pairs.py --bench replay`: ONE persistent `sim_bridge` fed the recorded 41-battle transcript (6,078 `CHOOSE`s), 5 reps per sample | **the Rust transport alone**, no Python | 30 | 20.2–34.8 | **+12.9 %** | **1.1055 – 1.2620** | 26/30 |
| 4 | `run_pairs.py --bench startup`: bare spawn + eager `Dex::for_gen(3)` + exit, 100 reps | per-battle fixed cost | 10 | 9.4–11.5 | +0.4 % | 0.9951 – 1.0166 | 6/10 |

Absolute values for context (never the claim): replay median CPU **A 191.9 ms → B 223.0 ms** per
41 battles. Spawn ≈ 7.0 ms per child on both sides. The throughput bench ran at ~2,060–2,310 steps/s
aggregate.

**Reading.**

- (3) is the sensitive, pure measurement. The Rust child really does ~31 ms more work per 41
  battles, which is +5.1 µs per `CHOOSE` and ≈ +10 µs per trainee decision (two `CHOOSE`s per
  decision).
- (2b) is the same effect diluted by the Python env. The single-env benchmark's decision cycle
  is ~0.8–1.7 ms here, so +10 µs is ≈ +0.6–1.3 %, consistent with the measured +1.6 % [+0.4, +2.2].
  In the M1 record (`rust_core_m1_2026-09-23/`, loads 19–21) a random-action production
  `Gen3Env.step` is ~3–4 ms. There, +10 µs is ≈ +0.3 %.
- (1) cannot resolve an effect of that size: its CI half-width is ~2–3 %. It reads NO MEASURABLE
  COST.
- (2a) had too few pairs to say anything (CI ±5–7 %). It is reported, not dropped.
- ⚠️ **Order of decisions, for honesty.** 2b was run *after* 2a came back unresolved, and 3 was
  designed after 2b showed a child-CPU signal. Neither was pre-registered. All four series are
  reported in full.

## Where the time goes (profile)

`perf` is unavailable (`perf_event_paranoid = 4`) and there is no valgrind, so `gdb_sampler.py` is
a gdb stack sampler. gdb runs the binary as its own child, which is legal at `ptrace_scope = 1`. An
external ticker SIGINTs it every ~5 ms, and each stop records the full inlined backtrace. It ran on
debuginfo builds of each side (`CARGO_PROFILE_RELEASE_DEBUG=true`, built into `/tmp/m1bench_prof_{A,B}`,
never a worktree's target) over 25× the transcript, giving 1,114 (A) and 1,150 (B) samples.
`profile_attrib.py` converts inclusive shares to ms per replay using bench 3's medians
(`profile_attrib.txt`, raw samples in `logs/`):

| component | A ms | B ms | B − A | ±1 SE |
|---|---|---|---|---|
| whole replay | 191.5 | 222.6 | **+31.1** | 0.4 |
| **one-sided view fold** (`SideObservation::observe`, via `BridgeChunks::push_chunk_lines`) | 15.2 | 33.9 | **+18.8** | 2.8 |
| **typed `Line`**: `ProtocolBuilder::emit` (build + `render`) | 0 | 7.4 | **+7.4** | 1.2 |
|  ↳ of which `Line::render` | 0 | 3.7 | +3.7 | 0.8 |
| retro-edit re-parse (`retro_edit` / `Line::parse` from `attrLastMove`) | 0 | 0 | **0** (0 of 1,150 samples) | — |
| `|request|` build (`build_request`) | 77.3 | 83.0 | +5.7 | 4.2 (noise) |
| stdout frame write (`flush_new_chunks`) | 30.1 | 35.7 | +5.5 | 3.2 (noise) |
| battle engine (`FullBattleDriver::feed`) | 37.7 | 37.4 | −0.3 | 3.4 |
| per-side log fold (`emit_log_batch_chunk`) | 33.4 | 34.7 | +1.3 | 3.2 |

**What this says.**

1. **The largest item is not M1.** `BridgeChunks::push_chunk_lines` folds EVERY line shipped to
   either side into `observed[side]` (`bridge.rs:379`). That is the one-sided view (`view.rs`), and
   its only readers are `view::one_sided_view`, called from `search_driver` and `core_events`.
   Training's `sim_bridge` never reads it, but it pays the fold on every line. The fold existed at A
   (15.2 ms, 7.9 %), and `db2e6515` (the truth audit's reading rules V3–V11, same range) made it ≈ 2.2×
   dearer.
2. **M1's own cost is ≈ 7 ms per 41 battles**, about 1.2 µs per `CHOOSE`: typed-`Line` construction
   plus render replacing `push_raw`. That is ≈ 3 % of the Rust child, and far below 0.1 % of an env
   step.
3. **The `attrLastMove` retro-edit re-parse is unmeasurable.** It is rare, and no sample landed in
   it.

This is an input to the owner, **not an optimisation**. The obvious lever is to gate the
observed-view fold to sessions that will read it (search / core), which would remove more than the
whole M1 cost. Nothing was changed.

## Hazards and findings from running it

- **`trainer_turn_benchmark.py` runs on the NODE bridge.** `run_local_battles`'s default is
  `impl="node"`, so, as shipped, it cannot see any Rust change. `env_step_rust.py` rebinds that one
  name to `impl="rust"` and changes nothing else. Anyone A/B-ing the Rust port with that script
  as-is measures Node.
- **`bridge_replay` is not a transport benchmark.** It replays through the genesis REFERENCE
  ORACLE (O(turns²)), not the production incremental `BridgeSession`. 28 of the 30 capture-golden
  battles also diverge early on the pre-existing gender/unmodeled-move scope, on BOTH A and B. So
  bench 3 replays a recorded `sim_bridge` stdin transcript (`transcript_env_seed0.txt`, built by
  `tee_sim_bridge.sh` + `build_transcript.py`) through the real binary instead.
- **gdb's `handle SIGINT … noprint` implies `nostop`.** The first sampler collected 0 samples because
  of it. A Python thread inside gdb never runs while `continue` holds the GIL, so the ticker has to
  be an external process.
- **Env 2b's child CPU (≈ 3 s per run) is far above the transcript replay's (0.19 s + 41 × 7 ms
  spawn).** The excess is some other child the harness reaps. It is identical in both arms, so it
  only dilutes that ratio, but it is unexplained.
- **The training run relaunched** (`train_rl_agent` PID 298087 → 333789, launcher-driven, not by
  this task) during series 2a, so the load swung 1 → 40. The interleaving is what makes the ratios
  usable, and absolute numbers are not comparable across series.

## Commands

```bash
export PYTHONPATH=$PYTHONPATH:src
PY=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3
M=designs/research_state/measurements/m1_transport_throughput_2026-09-23
git worktree add --detach .claude/worktrees/m1bench-A dfab2558
cargo build --release --bin sim_bridge --manifest-path .claude/worktrees/m1bench-A/src/rust_sim/Cargo.toml
cargo build --release --bin sim_bridge --manifest-path src/rust_sim/Cargo.toml
bash $M/verify_bins.sh
$PY $M/run_pairs.py --bench thru    --pairs 8  --workers 8 --seconds 45 --out $M/thru_rows.jsonl
$PY $M/run_pairs.py --bench env     --pairs 12 --decisions 10000       --out $M/env_rows.jsonl
$PY $M/run_pairs.py --bench env     --pairs 30 --decisions 3000        --out $M/env_rows_s2.jsonl
$PY $M/run_pairs.py --bench startup --pairs 10 --reps 100              --out $M/startup_rows.jsonl
# transcript: one capture run through the tee shim, then fold to ONE persistent stream
M1_TRANSCRIPT_DIR=/tmp/caps M1_REAL_SIM_BRIDGE=$PWD/src/rust_sim/target/release/sim_bridge \
  POKESIM_SIM_BRIDGE_BIN=$PWD/$M/tee_sim_bridge.sh \
  $PY $M/env_step_rust.py --decisions 3000 --arm capture --pair 0 --out /tmp/capture.jsonl
$PY $M/build_transcript.py /tmp/caps $M/transcript_env_seed0.txt
$PY $M/run_pairs.py --bench replay  --pairs 30 --reps 5                --out $M/replay_rows.jsonl
$PY $M/analyze.py $M/thru_rows.jsonl $M/env_rows_s2.jsonl $M/env_rows.jsonl $M/replay_rows.jsonl $M/startup_rows.jsonl
# profile: debuginfo builds OUTSIDE any worktree target, 25x transcript, gdb sampler
CARGO_PROFILE_RELEASE_DEBUG=true CARGO_TARGET_DIR=/tmp/m1bench_prof_A cargo build --release --bin sim_bridge \
  --manifest-path .claude/worktrees/m1bench-A/src/rust_sim/Cargo.toml     # and _B from this worktree
M1_SAMPLES_OUT=/tmp/samples_B.jsonl M1_STDIN=/tmp/long.txt M1_INTERVAL=0.005 \
  nice -n 10 gdb -q -batch -x $M/gdb_sampler.py --args /tmp/m1bench_prof_B/release/sim_bridge
$PY $M/profile_attrib.py /tmp/samples_A.jsonl /tmp/samples_B.jsonl $M/replay_rows.jsonl
```

Files: `*_rows.jsonl` are the raw rows (every run: arm, pair, binary, the measure, load at start and
end). `analysis.txt` has every metric's per-pair ratios, median and CI. `profile_attrib.txt` and
`logs/gdb_samples_{A,B}.jsonl.gz` hold the profile.

## Raw per-pair numbers

### 1 — throughput (`thru_rows.jsonl`), fps ratio shown as A/B (> 1 = B slower)

| pair | order | A fps | B fps | A battles | B battles | A/B | load1 A | load1 B |
|---|---|---|---|---|---|---|---|---|
| 1 | A,B | 2073.3 | 2061.3 | 1073 | 1078 | 1.006 | 23.6-30.3 | 30.3-36.3 |
| 2 | B,A | 2305.1 | 2240.8 | 1242 | 1204 | 1.029 | 36.0-34.4 | 36.3-36.0 |
| 3 | A,B | 2301.4 | 2221.4 | 1183 | 1177 | 1.036 | 34.4-35.6 | 35.6-36.6 |
| 4 | B,A | 2297.5 | 2308.5 | 1220 | 1199 | 0.995 | 38.6-34.9 | 36.6-38.6 |
| 5 | A,B | 2283.1 | 2285.7 | 1213 | 1244 | 0.999 | 34.9-40.0 | 40.0-32.5 |
| 6 | B,A | 2159.8 | 2192.7 | 1164 | 1173 | 0.985 | 31.0-30.3 | 32.5-31.0 |
| 7 | A,B | 2206.6 | 2223.2 | 1148 | 1210 | 0.993 | 30.3-38.4 | 38.4-35.9 |
| 8 | B,A | 2297.1 | 2203.6 | 1228 | 1144 | 1.042 | 39.2-40.7 | 35.9-39.2 |

### 2b — pinned env step, 3,000 decisions (`env_rows_s2.jsonl`)

| pair | order | A wall s | B wall s | A child CPU s | B child CPU s | B/A wall | load1 A | load1 B |
|---|---|---|---|---|---|---|---|---|
| 1 | A,B | 6.247 | 6.721 | 2.543 | 2.409 | 1.076 | 9.3-9.4 | 9.4-9.5 |
| 2 | B,A | 6.908 | 6.213 | 2.700 | 2.689 | 0.899 | 9.5-9.6 | 9.5-9.5 |
| 3 | A,B | 5.893 | 6.477 | 2.384 | 2.609 | 1.099 | 9.6-9.6 | 9.6-9.7 |
| 4 | B,A | 7.470 | 6.354 | 2.877 | 2.545 | 0.851 | 9.7-13.3 | 9.7-9.7 |
| 5 | A,B | 8.144 | 8.149 | 3.101 | 3.196 | 1.001 | 13.3-20.0 | 20.0-24.4 |
| 6 | B,A | 8.302 | 8.449 | 3.206 | 3.228 | 1.018 | 26.7-31.3 | 24.4-26.7 |
| 7 | A,B | 7.576 | 7.782 | 2.794 | 2.976 | 1.027 | 31.3-28.1 | 28.1-26.7 |
| 8 | B,A | 7.563 | 7.328 | 2.807 | 2.906 | 0.969 | 27.3-28.8 | 28.7-27.3 |
| 9 | A,B | 7.696 | 7.820 | 3.188 | 3.240 | 1.016 | 28.8-31.3 | 31.3-27.9 |
| 10 | B,A | 7.481 | 7.151 | 3.117 | 2.814 | 0.956 | 27.0-25.7 | 27.9-27.0 |
| 11 | A,B | 7.696 | 8.192 | 3.148 | 3.202 | 1.064 | 25.7-27.8 | 27.8-28.8 |
| 12 | B,A | 8.411 | 8.295 | 3.227 | 3.243 | 0.986 | 29.3-30.2 | 28.8-29.3 |
| 13 | A,B | 8.258 | 8.285 | 3.209 | 3.291 | 1.003 | 30.2-34.1 | 34.1-30.4 |
| 14 | B,A | 8.164 | 8.262 | 3.172 | 3.221 | 1.012 | 31.0-31.2 | 30.4-31.0 |
| 15 | A,B | 8.070 | 8.380 | 3.156 | 3.252 | 1.038 | 31.2-31.2 | 31.2-32.4 |
| 16 | B,A | 6.648 | 6.998 | 2.611 | 3.095 | 1.053 | 30.7-27.5 | 32.4-30.7 |
| 17 | A,B | 6.790 | 7.156 | 2.581 | 3.051 | 1.054 | 27.5-26.1 | 26.1-23.8 |
| 18 | B,A | 6.564 | 7.186 | 2.701 | 3.134 | 1.095 | 22.7-20.7 | 23.8-22.7 |
| 19 | A,B | 6.761 | 6.883 | 2.670 | 2.971 | 1.018 | 20.7-19.9 | 19.9-18.3 |
| 20 | B,A | 7.868 | 6.975 | 2.726 | 3.011 | 0.887 | 17.7-17.4 | 18.3-17.7 |
| 21 | A,B | 8.435 | 8.510 | 3.194 | 3.317 | 1.009 | 17.4-20.0 | 20.0-23.0 |
| 22 | B,A | 8.468 | 8.617 | 3.255 | 3.351 | 1.018 | 25.9-27.0 | 23.0-25.9 |
| 23 | A,B | 8.453 | 8.552 | 3.194 | 3.308 | 1.012 | 27.0-28.2 | 28.2-28.0 |
| 24 | B,A | 8.514 | 8.736 | 3.291 | 3.434 | 1.026 | 32.1-34.0 | 28.0-32.1 |
| 25 | A,B | 8.489 | 8.647 | 3.375 | 3.287 | 1.019 | 34.0-35.4 | 35.4-38.6 |
| 26 | B,A | 8.460 | 8.593 | 3.303 | 3.363 | 1.016 | 38.0-40.6 | 38.6-38.0 |
| 27 | A,B | 8.392 | 8.427 | 3.235 | 3.317 | 1.004 | 40.6-36.0 | 36.0-34.0 |
| 28 | B,A | 9.536 | 9.450 | 3.481 | 3.248 | 0.991 | 35.5-33.6 | 34.0-33.2 |
| 29 | A,B | 9.606 | 7.982 | 3.185 | 3.145 | 0.831 | 33.6-33.9 | 33.9-33.5 |
| 30 | B,A | 6.029 | 8.020 | 2.509 | 3.274 | 1.330 | 33.3-31.4 | 33.5-33.3 |

### 2a — pinned env step, 10,000 decisions (`env_rows.jsonl`)

| pair | order | A wall s | B wall s | A child CPU s | B child CPU s | B/A wall | load1 A | load1 B |
|---|---|---|---|---|---|---|---|---|
| 1 | A,B | 10.154 | 10.689 | 2.396 | 2.552 | 1.053 | 1.1-1.1 | 1.1-1.2 |
| 2 | B,A | 13.821 | 10.472 | 2.976 | 2.565 | 0.758 | 1.5-2.5 | 1.2-1.5 |
| 3 | A,B | 22.805 | 23.852 | 5.198 | 5.526 | 1.046 | 4.1-12.7 | 12.7-24.2 |
| 4 | B,A | 20.123 | 23.699 | 4.965 | 5.560 | 1.178 | 30.2-28.4 | 24.2-30.2 |
| 5 | A,B | 10.586 | 10.608 | 2.491 | 2.532 | 1.002 | 28.4-24.3 | 24.3-19.5 |
| 6 | B,A | 10.577 | 10.581 | 2.503 | 2.575 | 1.000 | 16.8-14.4 | 19.5-16.8 |
| 7 | A,B | 13.781 | 12.014 | 3.305 | 2.740 | 0.872 | 14.4-14.7 | 14.7-12.8 |
| 8 | B,A | 18.125 | 16.583 | 4.294 | 4.114 | 0.915 | 20.1-21.2 | 16.0-20.1 |
| 9 | A,B | 18.029 | 18.330 | 4.323 | 4.409 | 1.017 | 21.2-26.8 | 28.5-26.1 |
| 10 | B,A | 15.172 | 18.525 | 3.908 | 4.483 | 1.221 | 30.5-30.6 | 26.1-30.5 |
| 11 | A,B | 10.775 | 10.639 | 2.605 | 2.574 | 0.987 | 29.5-25.3 | 25.3-21.8 |
| 12 | B,A | 11.458 | 10.787 | 2.686 | 2.596 | 0.942 | 18.8-15.5 | 21.8-18.8 |

### 3 — pure-Rust transcript replay, CPU s per replay (`replay_rows.jsonl`)

| pair | order | A | B | B/A | load1 A | load1 B |
|---|---|---|---|---|---|---|
| 1 | A,B | 0.2579 | 0.2695 | 1.045 | 30.3-32.6 | 32.6-32.6 |
| 2 | B,A | 0.2585 | 0.2530 | 0.979 | 32.6-32.6 | 32.6-32.6 |
| 3 | A,B | 0.2287 | 0.2688 | 1.175 | 32.6-34.8 | 34.8-34.8 |
| 4 | B,A | 0.1928 | 0.2762 | 1.433 | 34.8-34.8 | 34.8-34.8 |
| 5 | A,B | 0.1781 | 0.2016 | 1.132 | 34.8-34.8 | 34.8-32.8 |
| 6 | B,A | 0.1987 | 0.3015 | 1.517 | 32.8-32.8 | 32.8-32.8 |
| 7 | A,B | 0.1839 | 0.2033 | 1.106 | 32.8-32.8 | 32.8-31.1 |
| 8 | B,A | 0.1772 | 0.2213 | 1.249 | 31.1-31.1 | 31.1-31.1 |
| 9 | A,B | 0.1782 | 0.2364 | 1.327 | 31.1-31.1 | 31.1-31.1 |
| 10 | B,A | 0.2067 | 0.2206 | 1.067 | 29.4-29.4 | 31.1-29.4 |
| 11 | A,B | 0.1872 | 0.2533 | 1.353 | 29.4-29.4 | 29.4-29.4 |
| 12 | B,A | 0.1886 | 0.2190 | 1.161 | 29.4-27.9 | 29.4-29.4 |
| 13 | A,B | 0.1839 | 0.3018 | 1.641 | 27.9-27.9 | 27.9-27.9 |
| 14 | B,A | 0.1736 | 0.2057 | 1.185 | 27.9-26.6 | 27.9-27.9 |
| 15 | A,B | 0.1762 | 0.2247 | 1.275 | 26.6-26.6 | 26.6-26.6 |
| 16 | B,A | 0.1967 | 0.2781 | 1.413 | 26.6-26.6 | 26.6-26.6 |
| 17 | A,B | 0.2322 | 0.3073 | 1.323 | 26.6-25.4 | 25.4-25.4 |
| 18 | B,A | 0.2122 | 0.2882 | 1.358 | 25.4-24.8 | 25.4-25.4 |
| 19 | A,B | 0.2486 | 0.1985 | 0.799 | 24.8-24.8 | 24.8-24.8 |
| 20 | B,A | 0.1771 | 0.1971 | 1.113 | 24.8-23.8 | 24.8-24.8 |
| 21 | A,B | 0.2541 | 0.1978 | 0.778 | 23.8-23.8 | 23.8-23.8 |
| 22 | B,A | 0.1967 | 0.2003 | 1.018 | 23.8-23.8 | 23.8-23.8 |
| 23 | A,B | 0.2760 | 0.2486 | 0.901 | 23.8-22.8 | 22.8-22.8 |
| 24 | B,A | 0.1779 | 0.1962 | 1.102 | 22.8-22.8 | 22.8-22.8 |
| 25 | A,B | 0.2108 | 0.2339 | 1.109 | 22.8-22.8 | 22.8-22.1 |
| 26 | B,A | 0.1910 | 0.2441 | 1.278 | 22.1-22.1 | 22.1-22.1 |
| 27 | A,B | 0.1829 | 0.2022 | 1.105 | 22.1-22.1 | 22.1-21.1 |
| 28 | B,A | 0.1790 | 0.2016 | 1.126 | 21.1-21.1 | 21.1-21.1 |
| 29 | A,B | 0.1787 | 0.2012 | 1.126 | 21.1-21.1 | 21.1-21.1 |
| 30 | B,A | 0.2021 | 0.2024 | 1.001 | 20.2-20.2 | 21.1-20.2 |

### 4 — bare spawn, CPU ms per spawn (`startup_rows.jsonl`)

| pair | order | A | B | B/A | load1 |
|---|---|---|---|---|---|
| 1 | A,B | 7.000 | 6.998 | 1.000 | 11.5-10.7 |
| 2 | B,A | 6.990 | 6.914 | 0.989 | 10.7 |
| 3 | A,B | 6.931 | 6.987 | 1.008 | 10.7 |
| 4 | B,A | 6.959 | 6.926 | 0.995 | 10.7 |
| 5 | A,B | 6.937 | 7.135 | 1.029 | 10.7-10.0 |
| 6 | B,A | 7.115 | 7.120 | 1.001 | 10.0 |
| 7 | A,B | 7.165 | 7.220 | 1.008 | 10.0 |
| 8 | B,A | 7.094 | 7.271 | 1.025 | 10.0-9.4 |
| 9 | A,B | 7.079 | 7.143 | 1.009 | 9.4 |
| 10 | B,A | 7.111 | 7.042 | 0.990 | 9.4 |

## Ledger paragraph (ready to append — NOT appended by this task)

> **2026-09-23 · MEASUREMENT · Rust Core M1 transport cost: a real but SMALL cost, and mostly
> NOT M1.** A/B `dfab2558` (pre-M1) vs `7c511161`, with `sim_bridge` built in separate worktrees,
> one binary per process, interleaved order-alternated pairs at nice 10, load 1–41, and byte-identical
> output confirmed on the workload. Results as the median cost of B with a paired bootstrap 95 % CI:
> (1) production-shape `bridge_impl_throughput_benchmark` at 8 workers **+0.2 % [−0.7, +3.6]**,
> 8 pairs: NO MEASURABLE COST at this resolution; (2) pinned rust env step (`trainer_turn_benchmark
> --pin-battles` rebound to `impl="rust"`) **+1.6 % wall [+0.4, +2.2]**, 30 pairs: formally a
> finding, inside ±2.2 %; (3) pure-Rust `sim_bridge` replay of a recorded 41-battle transcript
> **+12.9 % CPU [+10.6, +26.2]**, 30 pairs, = **+5.1 µs per CHOOSE (≈ +10 µs per trainee
> decision)**. A gdb stack profile puts ≈ 60 % of the delta (+18.8 ± 2.8 of +31.1 ms) in the
> one-sided view fold (`SideObservation::observe`, made ≈ 2.2× dearer by the truth audit
> `db2e6515`). `sim_bridge` runs it on every shipped line for a view only `search_driver` /
> `core_events` read. M1's typed-`Line` build+render is ≈ 24 % (+7.4 ± 1.2 ms), and the
> `attrLastMove` re-parse draws 0 of 1,150 samples. Not a blocker for a post-M1 arm (≈ +0.3 % of a
> production ~3–4 ms env step). Gating the observed-view fold to sessions that read it would remove
> more than M1's whole cost. Hazard: `trainer_turn_benchmark.py` runs on the NODE bridge by default,
> so as shipped it cannot see a Rust change.
> `designs/research_state/measurements/m1_transport_throughput_2026-09-23/`.
