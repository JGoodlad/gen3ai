# M5 Lane J — the M5 gate harness: PROGRESS (resume point)

Lane J of `designs/endstate/program_rust_core.md` §2 M5 (lane-table row J, order constraints 4–6, the
"Lane J BUILT" paragraph). Owns `src/main/rust_core_m5/` and this directory. It IS the M5 gate: the
single place the milestone is judged, composed from every lane's OWN gate (delegated, never
re-implemented) plus the three components the program names — slice N at the env level, the
depth-3 successor slice, the throughput A/B at `--n-envs 48`.

## How to run

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.rust_core_m5 gates --from-status                 # the banked MILESTONE verdicts; runs nothing (~1 s)
python -m main.rust_core_m5 gates --tier commit                 # every lane's routine gate, one pytest session
python -m main.rust_core_m5 gates --tier milestone [--gpu]      # + the slow tests (and the GPU ones, under the lock)
python -m main.rust_core_m5 slice-n --tier commit|milestone     # slice N at the env level
python -m main.rust_core_m5 depth3 --tier commit|milestone      # the depth-3 successor slice (Lane I's harness)
python -m main.rust_core_m5 throughput --n-envs 48 --out <f>    # the throughput A/B (a DESCRIPTOR)
python -m main.rust_core_m5 verdict                             # the M5 gate, composed from results/*.json
python3 -m pytest src/main/rust_core_m5 -q -m "not slow"        # routine: registry + verdict logic + slice N COMMIT + A/B units
python3 -m pytest src/main/rust_core_m5/slice_n_test.py -q -m slow   # slice N MILESTONE (banked in slow_tier_status.json)
```

Every component writes `results/<component>_<tier>.json` here (`--results` moves it; `models/` is
refused). `verdict` reads what is there: a component never run reads NOT RUN, never PASS.

## Units

| # | unit | status |
|---|---|---|
| 1 | the lane REGISTRY (`lanes.py`, one declared row per lane) + the verdict table (`gates.py`, `outcomes_plugin.py`: RUN at a tier, or RECORDED from the slow-tier bank) + the routine completeness test (`lanes_test.py`) | BUILT |
| 2 | SLICE N at the env level (`slice_n.py` + `slice_n_test.py`: COMMIT routine, MILESTONE slow) | BUILT — PASS |
| 3 | the depth-3 slice (`depth3.py`, delegating to Lane I's `successors_parity`) | BUILT — PASS (F-LI-1 open) |
| 4 | the THROUGHPUT A/B scaffold (`throughput.py` + `hooks.py` + `throughput_test.py`) | see below |
| 5 | the composed M5 verdict (`__main__.py verdict`) | BUILT |
| 6 | the SIZING study (order constraint 5) | DONE 2026-10-02 — N\* = 256 (Part L guard NO LOSS DETECTED, n = 1 flagged), E10 stays, `recipe.sizing` 256 × 384; `measurements/m5_sizing/` (REGISTRATION, PROGRESS "THE VERDICT"), ledger 2026-10-02 *THE M5 SIZING VERDICT* |

## The lane registry — how a lane joins the gate

One `LaneGate` row in `lanes.py`: the lane id (as the program doc's lane table names it), a human
description, what its gate proves, the pytest files / node ids that ARE its gate, its GPU node ids,
and for a NOT BUILT lane the gate it will have. Tier = MARKERS, as the repo's tiers are (COMMIT =
`not slow and not e2e`, MILESTONE = `not e2e`). `lanes_test.py` (routine) FAILS when:

- a lane the program doc marks BUILT (a lane-table cell saying BUILT, or a `**Lane X … BUILT`
  paragraph; `K1`/`K2` mark `K`) has no row, or its row says NOT BUILT, or declares no tests;
- a lane of the doc's lane table has no row at all (J itself excepted: it is the harness);
- a declared file does not exist, or a declared `file::test` names no function in it (a renamed
  test cannot silently collect nothing);
- a row claims BUILT where the doc does not, without a `doc_note` saying why (today: none — 0, C
  and T2 were, until the doc was flipped; F-LJ-3).

The fold (`gates.fold`): NOT BUILT never runs and never passes; any fail ⇒ FAIL; a timeout
(`slow_tier_status.classify`'s markers) ⇒ INCONCLUSIVE; nothing passed (skip-only or empty) ⇒
INCONCLUSIVE; a GPU part not run ⇒ `GPU part NOT RUN` beside the row. A node two rows declare (the
Rust cargo suite carries 0, D, E and I) runs ONCE and counts for each.

## Slice N at the env level — what it is (the reading)

The program's gate sentence: "obs rows, masks, labels, rewards and dones equal the Python `Gen3Env`
on recorded battles with both sides scripted from the recording". Lane C (labels) and Lane D
(reward / ends) each gate their columns at N = 1, T = 1. Slice N JOINS their record-in-Rust /
replay-in-Python machinery and runs it at N:

- ONE core of N envs on T threads, the production declaration (every label family, the production
  terminal, the stall threshold), a seeded random policy on both sides; episodes chained through
  the AUTO-RESET (each env's next episode staged, re-staged whenever its `episode` column moves —
  F-LE-4's rule), so every env plays several back to back;
- recorded through the FFI front end at T and the PROCESS front end at a different T′: the
  recordings must be identical (front ends + thread-count invariance at N);
- replayed through the production-surface `Gen3Env` (rust bridge, `--obs-source core`): per trainee
  decision `dec_n`, row bytes, BOTH masks the learner reads (`obs["action_mask"]` for the async
  rollout, `env.action_masks()` for the stock one), every label key (dtype / shape / bytes); every
  non-final reward 0; the final `(reward, terminated, truncated)`; exact decision counts both sides.

Declared out of scope (other lanes' gates): the host-filled keys (`win_target` / `win_mask` /
`opp_class`, Lane G), in-core opponents (their p2 decisions are never exposed, so they cannot be
replayed from the columns — Lanes E / F), the ROLLOUT level (the learner's buffer, Lane G).

## The depth-3 slice — the reading chosen

The program: "a depth-3 successor slice (search's default depth) equals `search_driver`'s rows".
Three things were open; the reading:

1. "search's default depth" = `SearchConfig().max_depth` = 3 — ASSERTED by `depth3.run_tier` and a
   routine test, so a new default re-opens this reading instead of silently gating another depth.
2. "equals `search_driver`'s rows" = Lane I's gate ① (the in-process tree and the binary in
   lockstep; every root and arm field + ROW BYTES at plies 1–3; no allowlist; each depth
   non-vacuous), at Lane I's registered MILESTONE sources (ladder milestone 12, pool 6, procedural 6
   battles, 3 turns, seed 17). J does not re-implement it.
3. A batch BOTH roads refuse with the SAME error is equality, not a difference — and not coverage.
   Those are F-LI-1 (a follow-up choice fed into the next turn, in both roads). The slice reports
   them as `refused_both` beside the verdict: **PASS means "equal wherever either road serves an
   arm", NOT "depth 3 is correct past a replacement switch that ends the turn"**.

## Results (2026-09-29, this worktree at `d58b12de` + Lane J; selfcheck builds for the gates, release for the A/B)

### The lane-gate table (`gates --tier milestone`, one pytest session, `-n 2`, 7 m 32 s; `results/gates_milestone.json`)

| lane | what it is | verdict | tests pass/fail/skip | GPU part |
|---|---|---|---|---|
| 0 | the shared core boundary (columns, pool, refusal policy, stamp) | PASS | 22/0/0 | — |
| A | the FFI front end | PASS | 13/0/0 | — |
| B | the process front end | PASS | 24/0/0 | — |
| C | training labels (18 production keys) | PASS | 22/0/0 | — |
| D | episodes and reward | PASS | 9/0/0 | — |
| E | opponent routing | **FAIL** (one near-tie, F-LJ-6) | 20/1/0 | NOT RUN here; PASS in the GPU read below |
| F | scripted bots | PASS | 23/0/0 | — |
| I | search on successors() in process | PASS | 10/0/0 | — |
| T2 | the inference service | PASS | 27/0/0 | NOT RUN here; PASS in the GPU read below |
| G | training integration | NOT BUILT *at this read* — BUILT since (2026-09-30); re-run `gates` for its verdict | — | — |
| H | eval on the core | NOT BUILT | — | — |
| S *(beside M5)* | the policy-spectrum instrument | PASS | 49/0/0 | — |
| K *(beside M5)* | the learner pipeline (K1, K2) | PASS | 53/0/5 | — |

GPU-inclusive read of E and T2 (`gates --tier milestone --gpu --lanes E,T2`, under the GPU lock):
| lane | verdict | GPU part | tests pass/fail/skip |
|---|---|---|---|
| E | **FAIL** — the same one `greedy_neartie` in `test_slow_compiled_per_env_path` (F-LJ-6) | **PASS** (GPU milestone, greedy and sampled) | 22/1/0 |
| T2 | PASS | **PASS** (`service_cuda_test`; its AOT test skips on torch 2.5.1) | 31/0/1 |

(`results/gates_milestone_lanes_E_T2.json`, overlaid on the full table by `verdict`.)

### The components

| component | tier | verdict | evidence |
|---|---|---|---|
| slice N (env level) | COMMIT (routine) | PASS | N = 4, T = 3 / 2: 8 pool episodes at the production threshold + 8 at threshold 6 (8 / 8 forfeits); 0 divergences; front ends identical |
| slice N (env level) | **MILESTONE** (`slow`, banked) | **PASS** | N = 48, T = 8 / 5: pool 240 + ladder 240 + procedural 96 at the production threshold + ladder 48 at threshold 12 — **46,353 decisions, 834,354 label + 92,706 mask compares, 0 divergences**, 847 auto-resets, 11 stall forfeits at the PRODUCTION threshold (250), 7 ties; front ends identical |
| depth-3 slice | MILESTONE | **PASS** (F-LI-1 open) | 47,856 arms, 53,736 rows byte-equal, 0 differences, 4,422 D10 leaves, rows at every depth; **25 batches refused by BOTH roads** |
| throughput A/B | N = 48, CPU | MEASURED | below |
| **M5 verdict** (`verdict`) | — | **NOT MET** | Lanes G and H NOT BUILT, and Lane E's compiled milestone test FAILS (F-LJ-6) |

### The throughput A/B (DESCRIPTORS; `results/throughput_*_n48.json`)

Regime (both reads): 16 cores, `nice` 15, N = 48, the Rust arm T = 8 through the PROCESS front end (training's default), RELEASE builds of both arms' Rust binaries (`rust_env` core; the Python arm's `sim_bridge`), a uniform-random opponent on both arms (`RandomPlayer` / the in-core bot `random`), the step collector (window 2048, counts only), 30 warm-up steps, then 4 interleaved pairs (A B, B A, …) of 25 s blocks; CPU = the whole process tree from `/proc`; the ratio = the geometric mean of per-pair ratios with a percentile bootstrap 95 % CI over pairs (4 pairs: coarse). Load1 0.8 / 4.4 at the start; the arms themselves drive it to ~24 (the Python arm burns ~11 cores, the Rust arm ~6), so the end-of-run "busy box" warnings are self-inflicted; no bystander process inside the measured tree.

| read | arm | trainee decisions / s | ms / vec step | CPU µs / decision | startup |
|---|---|---|---|---|---|
| CPU, random trainee | today's path (`SubprocVecEnv` forkserver, `Gen3Env`, `MaskableAgentWrapper`) | 3,194 | 15.1 | 3,519 | 17.1 s |
| CPU, random trainee | the Rust env core | **34,154** | 1.41 | **176** | 1.0 s |
| | **ratio rust / python** | **10.7× [10.2, 11.3]** | | **0.050× [0.0496, 0.0504]** | |
| T2 `graph` CUDA trainee (ai_v14_06_lbat_ctrl_fix final, buckets 8 / 48), GPU lock | today's path | 2,154 | 22.5 (2.6 inference) | 4,151 | 19.3 s + T2 163 s |
| same | the Rust env core | **11,611** | 4.13 (2.1 inference) | **299** | 1.1 s |
| | **ratio rust / python** | **5.4× [4.9, 5.8]** | | **0.072× [0.071, 0.074]** | |

Every core `*_AFTER_FREEZE` and T2 `*_after_freeze` counter 0 in both reads; 0 quarantines.

## What Lanes G and E plug in (the hooks)

- **`hooks.TraineeInference`** — `(obs [n, OBS_DIM] f32, masks [n, 11]) -> actions [n] int` + `describe()`.
  Built: `RandomLegal(seed)`, `T2Inference(ckpt, backend, device, buckets)`, and (Lane G, 2026-09-30)
  `LearnerSampling(ckpt, device, compile)` — the LEARNER's sampling forward (`--inference learner`;
  `--compile-trainee` compiles it as `--compile-trainer` would). On the production Rust arms the trainee
  is served by T2 inside the collector (one flush with the opponents); the Python arm calls the learner.
- **`hooks.OpponentMix`** — `python_opponent(idx, tag, teambuilder)`, `rust_routes()`, `ep_opp(env)`.
  Built: `UniformRandom`, and (Lane G) `ProductionMix(pool, pool_size, self_play_fraction,
  compile_opponents)` — ONE resolved args namespace (`rust_core_cutover.envs.production_args()`, self-play
  on) and ONE pool (the last `pool_size` snapshots of `--pool`, symlinked into a temp dir). It selects the
  PRODUCTION arms (`production.py`): `ProductionPythonArm` (`create_training_env_random`'s workers with the
  production kwargs, per-worker compiled `RLPlayer` opponents) and `CollectorRustArm` /
  `OverlappedRustArm` (Lane G's `build_collector` + Lane E's `RustEnvOpponents`, one T2 service shared by
  every rust arm of one A/B). Arm names: `rust_<serial|overlap>_<keyed|generator>[_p<K>]` — `_p<K>` routes
  to only the K newest pool snapshots (the fewer-active-snapshots candidate). Every row is stamped with
  the mix, the pool size, the active snapshots, T2's slots / buckets / lanes / backend and p2's sampling.
- **`hooks.Collector`** — `observe(dones)`, `ready()`, `describe()`. Built: `StepCollector(n_steps)`, and
  (Lane G) `CompleteGameCollector(target)` — the rust arms step `RustCollector` (`--collector
  complete_game`, the fill counted); its gate is Lane G's slice N at the ROLLOUT level
  (`rust_rollout/parity.py`), on the registry's G row.
- **The registry rows:** G is BUILT (2026-09-30) with its `tests=` / `gpu_tests=`; H flips the day its
  gate lands; `lanes_test.py` fails the day this doc marks a lane BUILT and the row does not.

### The PRODUCTION throughput A/B (Lane G, 2026-09-30; `results/throughput_production_*.json`)

The owner's registered comparison: 48 envs, the production opponent mix (95 % self-play against a
20-snapshot pool of `ai_v14_06_lbat_ctrl_fix`, served through T2 on the Rust arms and by per-worker
compiled `RLPlayer`s on CPU on the Python arm; 5 % the 8 training-floor bots — in the core on Rust), the
trainee = that run's final checkpoint (the learner's compiled sampling forward on Python; T2 `graph` on
Rust), release builds, fp32 `highest` (the run's own precision), GPU under the lock, fresh per-run
Inductor / Triton caches, 50 warm-up steps, then 6 rounds of interleaved 20 s blocks (rotating order);
95 % CIs are percentile bootstraps over blocks (per-arm) and over rounds (ratios). Load1 ~2 at the
start, self-driven by the Python arm's 48 workers; no bystander in the measured trees; no warning. See
Lane G's PROGRESS for the component split, the fan-out read and the decisions.

| read (N = 48, 95 % self-play / 5 % bots) | arm | trainee decisions / s [95 % CI] | ms / step | CPU µs / decision | GPU util |
|---|---|---|---|---|---|
| `throughput_production_sp95_n48.json` | Python (today's path) | 743 [674, 783] | 65.4 | 12,993 | 6 % |
| | Rust serial, keyed | 3,780 [3,650, 3,941] | 11.7 | 496 | 67 % |
| | Rust serial, generator | 3,769 [3,593, 3,921] | 11.7 | 493 | 68 % |
| | Rust overlapped, keyed | 2,690 [2,643, 2,732] | 16.3 | 723 | 86 % |
| | Rust overlapped, generator | 2,662 [2,586, 2,726] | 16.5 | 731 | 85 % |
| | **ratios** | **serial / Python 5.11× [4.72, 5.63]; CPU 0.038× [0.037, 0.039]; generator / keyed 0.997× [0.957, 1.030]; overlapped / serial 0.712× [0.690, 0.734]** | | | |
| `…_active_snapshots.json` (quiet box, Rust only) | serial keyed, 20 / 8 / 4 / 1 active snapshots | 4,116 / 6,530 / 5,913 / 7,151 | 10.7 / 6.8 / 7.5 / 6.2 | 451 / 359 / 370 / 349 | 74 / 62 / 67 / 58 % |
| | **ratios vs 20** | **8: 1.59× [1.58, 1.60]; 4: 1.44×; 1: 1.74×** | | | |
| `…_null_identical_arms.json` | four IDENTICAL serial arms (a void fewer-snapshots run) | ratios 0.97–1.00 | | | |

The 90 % self-play read (`throughput_production_sp90_n48.json`, the day before, 2 arms): 5.10× [4.93, 5.26].
## Next (resume point)

1. When E's / T2's GPU-inclusive read is in, fold it here and in the program doc's Lane J paragraph.
2. Lane E: make `test_slow_compiled_per_env_path` reproducible (F-LJ-6) — Lane E's file, not J's.
3. ~~On Lane G's landing: its rows + `ProductionMix` + `CompleteGameCollector`, then the A/B at
   `--n-envs 48` with the learner's forward and the production mix~~ DONE by Lane G (2026-09-30, above).
   Next: a recorded `gates` read with G's row (not yet re-run), then the SIZING study (order constraint 5: N ∈ {48, 256, 1024, 2048} at a fixed rollout size;
   the A/B takes `--n-envs` / `--threads` for it).
4. A quiet-box re-read of the A/B (these ran beside nothing else, but at `nice` 15 with the arms'
   own load) is optional; the ratio's CI excludes 1 by an order of magnitude.

## Findings

- **F-LJ-1 (bootstrap, a HAZARD to every lane):** `scripts/bootstrap.sh` keys its "the conda env is
  current" stamp on the PER-WORKTREE git dir (`$(git rev-parse --git-dir)/gen3ai-bootstrap`), so
  EVERY fresh worktree runs `conda env update -n gen3ai_stable -f environment.yml --prune` on the
  SHARED env — here while Lane S's truth job (PID 1831130) was running on it. Caught at "Installing
  pip dependencies" and killed by PID; conda history (last change 09-24) and site-packages (no
  entry newer than the run) show nothing changed. Worked around by stamping this worktree only. Fix
  proposed (not J's file): key the env stamp on `git rev-parse --git-common-dir`, or make step 2
  report drift and update only with an explicit flag.
- **F-LJ-2 (closes F-LD-6's UNVERIFIED):** slice N's milestone reached the stall forfeit at the
  PRODUCTION threshold (250 turns) — 11 episodes, equal on both paths.
- **F-LJ-3 (this program doc's BUILT markers):** Lane 0's and Lane C's paragraphs say BUILDING and
  T2's lane-table row carries no BUILT marker, while all three have every gate built, routine and
  green. The registry records each as BUILT with a `doc_note`; their owners should flip them.
  **CLOSED 2026-09-30:** the program doc now marks all three BUILT with their landing commits, and
  the three `doc_note`s are removed (the registry and the doc agree; `lanes_test.py` green).
- **F-LJ-4 (F-LI-1 stays open):** the one thing the depth-3 PASS does not cover — 25 batches both
  roads refuse at the milestone. The verdict prints the count every time.
- **F-LJ-5 (slow-tier bank, FIXED in J's path):** a milestone run without
  `GEN3AI_TEST_ALLOW_GPU` banked SKIP over 7 banked GPU PASS rows in `slow_tier_status.json` (the
  merge replaces rows). Restored from HEAD; `gates.run` now banks through a scratch copy and never
  lets a skip replace a verdict (`bank_without_skip_clobber`, tested). Anyone else running `-m slow`
  without the GPU flag still clobbers them — the general fix is `slow_tier_status.record_results`'s.
- **F-LJ-6 — CLOSED (hand-off note by Lane E, 2026-09-30).** **FIXED by Lane E (2026-09-30):** the leaked state was the intra-op THREAD COUNT — the harness's own `record()` set `torch.set_num_threads(4)` and never restored it, so the next test built its FRESH snapshots at 4 threads instead of 8, and a fresh policy's orthogonal init (a QR) is thread-sensitive: 94 of 721 tensors differ by up to 7.5e-6. Same battles, weights moved by rounding, and at the gate's one EXACT tie (margin 0.0, two switch targets scored identically) the argmax fell the other way. Fix: `declared_torch_state` pins every global the gate depends on (snapshot build at 1 thread, recording at 4, replay at 1), RESTORES the count it found, and raises `GateStateError` on an undeclared global; the tie rule is declared in the ASSERTION (`judge_flips`: a flip below 2 x the tier's |Δ| bar is a counted TIE, any other flip FATAL). Verified: the compiled configuration run after the file's other tests and run alone records byte-identical battles; a full-order session of the whole file (incl. both GPU milestones) passes, banked. **Re-verified 2026-09-30 (cutover prep, `6b5df359` + the new test):** the whole file in its natural order, slow included, 12 passed / 3 GPU skips — the compiled test 1,881 decisions, 0 tie flips (1 decision inside the tie band, not flipped), banked; the ASSERT PATH itself is now pinned by `test_the_assert_path_counts_a_near_tie_and_stays_fatal_above_the_margin` (a tie inside 2 × bar is counted and returned, a flip at / above it or with no margin is fatal, a `div` stays fatal) — it fails on revert to an assertion that treats any flip as fatal. As found:
  `rust_env_opponents_parity_test.py::test_slow_compiled_per_env_path` FAILS with one
  `greedy_neartie` (compiled-CPU vs eager argmax where the top two legal log-probs are within 2 × the
  bar) in EVERY session that runs it after other tests — 3 of 3 (the full milestone session; the E + T2
  session; its OWN file alone, after its COMMIT tests) — and PASSED once run completely alone. So it
  is ORDER-DEPENDENT (global torch / compile state left by an earlier test — not isolated further), and
  as the slow tier runs it, it is red. Its harness NAMES the near-tie class but the assertion
  (`div == {}`) treats it as fatal. The banked row reads PASS (`e5478fcb`, likely an isolated run), so
  the routine gate does not see it. **NOT banked here:** banking the FAIL turns every routine gate
  on main red, and J's commit is briefed to ship green — the row is left at HEAD's value and the
  decision is the orchestrator's (bank it, or Lane E fixes the isolation / near-tie rule first).
- **F-LJ-7 (for Lane G / SIZING):** with the trainee's T2 forward in the loop the Rust path's
  step is half inference (2.1 of 4.1 ms at N = 48), serial; overlapping it with the env step
  (two half-batches in flight) is the next 2× and a SIZING input. The env step itself read 1.4 ms
  alone and 2.0 ms beside the T2 process (cause UNVERIFIED).
- **F-LJ-8:** the throughput subagent died twice on an API error (a service-side refusal, not the
  work); its files were complete and are reviewed and finished here (one change: action columns
  reset to −1 each step).
