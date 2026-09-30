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
| 6 | the SIZING study (order constraint 5) | NOT STARTED — after M5's gate (parity at N = 48); the A/B takes N and T as parameters for it |

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
| G | training integration | NOT BUILT | — | — |
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
  Built: `RandomLegal(seed)`, `T2Inference(ckpt, backend, device, buckets)`. Lane G's rollout host
  replaces it with the learner's SAMPLING forward (log-probs, values and the behaviour μ(a|s) out,
  not only actions) — the A/B needs only the actions.
- **`hooks.OpponentMix`** — `python_opponent(idx, tag, teambuilder)`, `rust_routes()`, `ep_opp(env)`.
  Built: `UniformRandom`. DECLARED, NOT BUILT: `ProductionMix` (raises `HookNotBuilt`) — Lane G builds
  BOTH arms from ONE resolved args namespace and ONE pool: the args, the snapshot pool dir, a started
  T2 service with `OpponentPlan.n_policy_slots` slots, Lane E's `RustEnvOpponents` (its
  `after_op(cols)` after EVERY core op — F-LE-4 — and `PolicyOpponentServer.serve(cols)` before
  every step), and on the Python arm `create_training_env_random`'s kwargs from the same args. Lane
  E's `ep_opp(env)` then comes from the sampler's staged route instead of the constant 0.
- **`hooks.Collector`** — `observe(dones)`, `ready()`, `describe()`. Built: `StepCollector(n_steps)`.
  DECLARED, NOT BUILT: `CompleteGameCollector` (Lane G, order constraint 6). Lane G's unit 1 LANDED
  while this lane was finishing (`11073fe1`: `agents.training.rust_rollout.build.build_collector` →
  `RustCollector`, not yet wired into training): plugging it in = the A/B's rust arm steps that
  collector (which already serves p2 + trainee rows in one T2 flush) instead of the bare core. What
  the hook must carry: the sample-count trigger
  (a startup input from the SIZING study), a buffer keyed by (env, episode) that back-fills each row's
  outcome at game end, games carried over at the trigger, and its gate — slice N at the ROLLOUT level
  with the trigger set to reproduce today's window.
- **The registry rows:** G and H flip `built=True` with their `tests=` (and `gpu_tests=`) the day
  their gates land; `lanes_test.py` fails the day this doc marks them BUILT and the row does not.

## Next (resume point)

1. When E's / T2's GPU-inclusive read is in, fold it here and in the program doc's Lane J paragraph.
2. Lane E: make `test_slow_compiled_per_env_path` reproducible (F-LJ-6) — Lane E's file, not J's.
3. On Lane G's landing: its rows + `ProductionMix` + `CompleteGameCollector`, then the A/B at
   `--n-envs 48` with the learner's forward and the production mix (the program's registered shape),
   then the SIZING study (order constraint 5: N ∈ {48, 256, 1024, 2048} at a fixed rollout size;
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
- **F-LJ-6 — CLOSED (hand-off note by Lane E, 2026-09-30).** **FIXED by Lane E (2026-09-30):** the leaked state was the intra-op THREAD COUNT — the harness's own `record()` set `torch.set_num_threads(4)` and never restored it, so the next test built its FRESH snapshots at 4 threads instead of 8, and a fresh policy's orthogonal init (a QR) is thread-sensitive: 94 of 721 tensors differ by up to 7.5e-6. Same battles, weights moved by rounding, and at the gate's one EXACT tie (margin 0.0, two switch targets scored identically) the argmax fell the other way. Fix: `declared_torch_state` pins every global the gate depends on (snapshot build at 1 thread, recording at 4, replay at 1), RESTORES the count it found, and raises `GateStateError` on an undeclared global; the tie rule is declared in the ASSERTION (`judge_flips`: a flip below 2 x the tier's |Δ| bar is a counted TIE, any other flip FATAL). Verified: the compiled configuration run after the file's other tests and run alone records byte-identical battles; a full-order session of the whole file (incl. both GPU milestones) passes, banked. As found:
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
