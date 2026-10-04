# X5 U2 GPU checks + eval U2 real-launch gate (2026-10-04)

The GPU checks that X5 build unit U2 deferred (`designs/endstate/design_x5_belief_tokens.md` §3.2, §3.6,
§8.3 U2 hand-off, F-X5-22), run under the GPU lease, plus the eval U2 real-launch gate the orchestrator added.

- **X5 commit measured: `e78884c4`** (`origin/main` when the agent started). It contains U3 **part 1**
  (`be6ba590`: hypothesis tokens in the trunk, the log-π key bias in the four class-E pools, OTHER's trunk seat).
  U3 part 2 (`5697c762`, the move axis) and part 3 (the op) landed or are still to land AFTER it, so every cost
  figure below is an EARLY read and will move.
- **Eval U2 commit gated: `ecf9eeca`** (in-loop eval + SPRT write ledger v2 rows). `e0252dc0` (the F-ED-22 index fix)
  landed after it and was NOT exercised.
- **Box.** RTX 3080 Ti (12 GiB), torch 2.8.0+cu126, fp32 only, the production recipe at N = 256
  (98,304 rows per rollout, batch 2,048 × accumulation 32, 10 epochs = 480 micro-batches per update), Rust env
  core, T2 graph backend, buckets (8, 64, 256). Tags: **MEASURED** unless marked.

## Verdicts

| check | verdict |
|---|---|
| **1. F-X5-22: the unrolled 64-step bisection's GPU compile cost** | **Does NOT bind.** Whole-launch startup +35 s on ~410 s (two launches per arm), inside the 245–299 s launch-to-launch spread of the T2 build alone. Isolated, the 64 steps cost ≈ 0 to compile (fixed_mass 94.3 / 97.0 s vs the same graph at ZERO steps 98.7 s; blob 83.2 / 81.5 s). Same graph count, no recompile after the lock, canary PASS. The alternative F-X5-22 names (τ outside the graph) was measured as a measurement only: it gains no compile time and is SLOWER at run time (+4.2 ms per 2,048-row fwd+bwd, +2.8 ms per 256-row fwd). |
| **2. CUDA eager vs compiled parity of the hypothesis set** | **PASS, 27 of 27 comparisons** (plus the tie test), every difference ≤ 4.6e-6 against bars of 1e-5 / 1e-4. The selection (hypotheses, slots, move seats) is identical on every non-near-tie row; stable argsort tie order identical on CPU, CUDA eager and CUDA compiled. |
| **3. Early cost under `fixed_mass` (U3 part 1)** | **Over the pre-registered budget.** `train_ms` **+8.7 %** (43.44 s vs 39.97 s; budget +5 %). Trainee T2 GPU wait +11.7 % (budget wording "T2 flush ≤ +3 %", see F-G-3). Memory headroom 2,143 → 1,963 MiB (−180; floor 1,024) WITHOUT the X26 heads. The cost is in the token path (GEMM +7.4 of +11.3 ms per micro-batch), not in the bisection (≈ 0.4 ms). |
| **eval U2 gate** | **PASS.** Startup line, `9 cycle row(s) appended` per cycle, `audit` OK (51 rows, 9 requests, 2 decisions), 48 of 48 cycle rows' `[w, w+l+d]` equal `eval_results.jsonl`, the cycle wall tracks trainee decisions at the profile's rate, one decision row per SPRT candidate. |

## 1. F-X5-22 — compile cost on the GPU

**Four measured launches** on `e78884c4`, one fresh `--arch production` run each, into a scratch
`GEN3AI_MODELS_DIR` on disk (a hermetic per-run compile cache, cold), through the real startup and the compile
lock. Order fm_a, blob_a, blob_b, fm_b. fixed_mass = `--belief-tokens fixed_mass --allow-nonproduction-arch`
(`checkargs`: the only ARCH-surface drift is `belief_tokens`; every RECIPE knob matches). Eval OFF
(`--eval-freq 50000000`), so the trainee plays the in-core bots (regime B: no pool, T2 serves the trainee only).

| launch | T2 build | R1 reset + prewarm | start → first rollout | startup windows contended (≥ 1.05) | max factor |
|---|---|---|---|---|---|
| blob_a | 293.1 s | 119.8 s | 433.5 s | 35 / 215 | 2.04 |
| blob_b | 245.1 s | 119.0 s | 382.7 s | 9 / 190 | 1.81 |
| fm_a | 299.0 s | 125.5 s | 444.9 s | 19 / 221 | 1.76 |
| fm_b | 290.9 s | 130.5 s | 441.5 s | 7 / 219 | 1.25 |
| **fm − blob (means)** | +25.9 s (blob_a alone: +5.9 s) | **+8.6 s (+7 %)** | **+35.1 s** | | |

- **Read.** The same arm's T2 build moves 245–293 s with the box's load (compile is CPU-parallel, and other
  agents' gates ran), so the X5 delta is inside that spread; R1's prewarm, the steadier figure, reads +6 to +11 s.
  The whole startup delta is under a minute, not "a couple of minutes". Contended windows are reported, not
  hidden: the compile is not a quiet-window measurement, so the figures carry the spread above.
- **Graphs.** R1: `4 graphs in process (incl. the gate's)` in all four launches; T2: 3 (`decide ×3`, one per
  bucket) in all four. `compile/graphs_total` 4, `recompiles_after_lock` **0**, `cache_limit_hits` 0 over every
  update of every launch. The compile canary at update 10 (blob_b, fm_b): `compiled == eager`, loss rel 0.00, grad
  cosine 1.000000; the startup R1 parity gate PASSED on fresh and perturbed weights in both fixed_mass launches.
- **Isolated compile A/B** (`scripts/compile_ab.py`; one fresh process per arm, private cold cache,
  `torch.compile(fullgraph=True, dynamic=False)` of the extractor forward on real rows, one sample per row
  unless two are listed):

  | arm | B = 2,048 fwd+bwd: compile | steady compiled | B = 256 fwd: compile | steady compiled |
  |---|---|---|---|---|
  | blob | 83.2 s, 81.5 s | 71.97 ms, 71.73 ms | 52.5 s | 4.54 ms |
  | fixed_mass (64 steps) | 94.3 s, 97.0 s | 83.31 ms, 83.05 ms | 59.9 s | 5.22 ms |
  | fixed_mass, **0 steps** (what the unrolled steps cost, by subtraction; meaningless τ) | 98.7 s | 82.88 ms | — | — |
  | fixed_mass, 28 steps ("fewer fp32 steps") — MEASUREMENT ONLY | 90.5 s | 83.16 ms | 58.5 s | 5.18 ms |
  | fixed_mass, τ in an opaque custom op ("τ outside the graph") — MEASUREMENT ONLY | 87.6 s | 87.49 ms | 54.8 s | 8.01 ms |

  The 64 unrolled steps cost nothing measurable: 94–97 s with them against 98.7 s without. Why identical
  Triton kernels would dedupe across the 64 iterations is an **UNVERIFIED** explanation (the CPU
  measurement, ≈ 9 s per construction, did not carry over). The +11 to +15 s over blob is the rest of X5.
  The opaque-op alternative runs the bisection in eager (~450 small launches per call) and is slower, so
  F-X5-22's "run τ outside the compiled region" is a **loss on the GPU**: no compile gain, +4.2 ms per
  micro-batch (≈ +2 s per update) and +2.8 ms per T2 forward. The design stays as built.

## 2. CUDA eager vs compiled parity (`scripts/cuda_parity.py`, `results/parity.json`)

- **What.** The production-surface `fixed_mass` learner (one build thread, no test perturbation), the extractor
  forward on **4,096 real rows** (`scripts/gen_real_obs.py`: the Rust collector, a seeded perturbed learner against a
  random opponent, complete games), reading `fe.last_hypothesis`. CPU eager fp32 is the reference; CUDA eager;
  CUDA compiled (`fullgraph=True`, `dynamic=False`, T2's flags, one graph per static shape): B = 8, 64, 256
  (T2's buckets, no_grad) and B = 2,048 in grad mode (the learner's micro-batch). Three weight sets: **cold**
  (δ_θ's last layer zero, so π is the Smogon prior's fixed-size marginal), **pert** (`parity_probe.perturb_`, the
  gates' own perturbation, scale 0.05) and **stress** (pert plus δ_θ's output layer at 0.5, π max 0.999).
- **Bars, registered in the script header before the run** (the code's own tolerances): probabilities and masses
  1e-5 (the fp32 Σπ = k tolerance), log-presences 1e-4, embeddings 1e-4 (the compile gate's feature bar),
  |Σπ − k| ≤ 1e-5, integers and structure exact, and the **selection identical on every row outside rule 8's
  near-tie set** (`near_tie_rows`, ε = 1e-6, the union over both computations), the excluded count reported.
- **Result: 27 of 27 comparisons PASS** (3 weight sets × [CUDA eager vs CPU, plus 4 compiled shapes × (vs CUDA
  eager, vs CPU)] = 3 × 9), and the tie test PASSES. Worst case over all:

  | quantity | CUDA eager vs CPU | compiled vs CUDA eager | compiled vs CPU | bar |
  |---|---|---|---|---|
  | π, hypothesis π, move π | 1.6e-6 | 1.4e-6 | 1.8e-6 | 1e-5 |
  | OTHER mass / P(any tail) | 2.5e-6 / 6.9e-7 | 2.6e-6 / 7.2e-7 | 2.5e-6 / 8.9e-7 | 1e-5 |
  | slot log π, OTHER log-mass | 3.9e-6, 2.2e-6 | 3.5e-6, 2.3e-6 | 4.3e-6, 3.6e-6 | 1e-4 |
  | OTHER token, tail mean | 2.7e-6, 4.0e-6 | 2.5e-6, 3.9e-6 | 3.1e-6, 4.6e-6 | 1e-4 |
  | \|Σπ − k\| (fp32, live rows) | ≤ 1.4e-6 | ≤ 1.4e-6 | ≤ 1.4e-6 | 1e-5 |

  Integers (k, n, live, full, OTHER live, move group r / k) equal in every comparison. Selection mismatches on
  non-excluded rows: **0** for `hyp_species`, `slot_species` and `mv_seat_nums`, in every comparison.
- **Rule-8 exclusions.** 67 of 4,096 rows (1.6 %; 5.2 % of the 1,284 live rows) are near-ties at the **cold start**
  (exact prior ties, δ_θ = 0); **0** under pert and stress. All 67 still agree across devices (selection
  identical), so the exclusion did not hide a disagreement here.
- **Stable argsort tie order** (`ties`): 2,048 synthetic rows × 400 candidates with only 12 distinct scores, so ~33
  candidates share each π exactly and **the seat boundary falls inside an exact tie block on 1,763 of 1,763 live
  rows**. `stable_order` and `fixed_mass_presence` give the **same full 400-long permutation** on CPU eager, CUDA eager
  and CUDA compiled; equal scores give bit-equal π on all three (0 violations); π differs by ≤ 6.7e-8 between devices;
  ties resolve to the lower num on CPU, and a raw `argsort(stable=True)` of 5-level keys is identical on all three.
- **First run, one harness bug.** Its first run failed the CPU "ties resolve to the lower num" line because the
  checker indexed a permutation with a species mask (the code was right: CUDA's permutation equalled CPU's). Fixed
  and re-run; every other number was identical. The first result is kept as `results/parity_run1_tie_check_bug.json`.
- **Coverage limit (F-G-6).** The rows come from a random-policy rollout: 2,812 of 4,096 have all six opponent mons
  revealed (k = 0, not live), r = 1…5 has 130 / 168 / 219 / 286 / 481 rows, and r = 0 never occurs (the lead is
  always revealed). The synthetic tie test covers the structure the real bank under-samples.
- **Compile wall of the whole extractor forward** under `fixed_mass` (cold, in one process): 57.7 s (B = 8),
  59.1 s (64), 64.1 s (256), 66.1 s (2,048, grad).

## 3. Early cost read (the same four launches)

Steady updates = the QUIET ones only: warm-up (the dry update and the first real one), the compile-canary update
and any update with a windowed contention factor ≥ 1.05 are excluded and listed in `results/<launch>.json`
(standing rule 8). blob: 16 updates; fixed_mass: 11 (fm_a lost five to other agents' CPU gates).

| | blob | fixed_mass | Δ |
|---|---|---|---|
| **update wall (`train_ms`), median** | **39.97 s** [39.93, 40.09] | **43.44 s** [43.40, 43.50] | **+3.47 s = +8.7 %** (CI of the difference +8.4 to +8.9 %); per launch +8.4 %, +8.8 % |
| per-launch medians | 40.09, 39.95 | 43.44, 43.46 | |
| trainee T2 GPU wait per host step (B = 256) | 4.315, 4.346 ms | 4.784, 4.893 ms | +0.51 ms = **+11.7 %** |
| T2 host flush per host step | 0.694, 0.544 ms | 0.550, 0.556 ms | none |
| flush + wait | 5.01, 4.89 ms | 5.33, 5.45 ms | +8.9 % |
| rollout (play) wall, quiet | 5.88 s | 5.81 s | −1 % (noise: the Rust core dominates) |
| `UpdateFit` headroom (card free after the dry update, declared floor 1,024) | 2,144, 2,142 MiB | 1,964, 1,962 MiB | **−180 MiB** |
| peak reserved in the update | 8,058, 8,060 MiB | 8,188, 8,190 MiB | +130 MiB |
| peak allocated in the dry update (98,304 staged rows) | 6,388 MiB | 6,605, 6,607 MiB | +217 MiB (+3.4 %) |
| startup allocation, T2 slots + arena / compiled regions (reserved in brackets) | +863 [2,166] / +200 [4,642] MiB | +930 [2,302] / +249 [4,804] MiB | +67 [+136] / +49 [+162] MiB |

- **Where the +8.7 % is** (`results/compile_ab_*_prof.json`: torch.profiler of 3 compiled fwd+bwd steps at B = 2,048,
  one process per arm): the extractor's forward+backward goes 71.7 → 83.1 ms (+15.7 %) and 1,491 → 1,719 kernel
  launches (+15 %). **GEMM +7.45 ms** (26.5 → 34.0; the 128×128 weight-gradient GEMM alone 5.1 → 8.8 ms), Triton
  pointwise +2.44, Triton reductions +1.14, **attention −0.01 ms**. Reading: the second `PokemonEncoder` pass over
  the hypothesis context and the extra trunk token cost GEMMs; the float log-π key bias costs nothing in SDPA; the
  bisection is ≈ 0.4 ms of it (the zero-step arm). This is **inference from the profile, not a causal ablation of
  each U3 piece.**
- **Not measured.** (a) The X26 ride-along heads: `--arch production` at this commit has all `--ridealong-*` OFF
  (`checkargs` lists them as not applied), so the budget's "with the X26 heads" headroom is **UNVERIFIED**; the
  heads' memory comes out of the 938 MiB of slack. (b) Regime A (a self-play pool): the opponents' graphs would also
  run the X5 forward, and a pool of X5 snapshots does not exist yet; the T2 figures are trainee-only. (c) U3 part 2
  and part 3. (d) The budget's "T2 flush" is not defined in the note beyond "13.12 ms (trainee 4.48 ms)"; the 4.48 ms
  matches the GPU-wait column above, so +11.7 % is the like-for-like figure.
- **Cross-launch noise** on this box is bounded near 1 % (bottleneck profile README); here the four launches agree
  within 0.4 % per arm, an order of magnitude under the +8.7 % delta.

## 4. Eval U2 real-launch gate (`ecf9eeca`)

Two fresh production launches into one scratch archive (`GEN3AI_MODELS_DIR` on disk), stopped by SIGTERM to the
trainer's PID after the cycles (clean `[ABORT] Checkpoint saved`). The brief's literal `python src/main/train_rl_agent.py`
was run through `phase_probe_run.py` (the bottleneck profile's wrapper: it calls the trainer's own `main()` unchanged;
standing rule 3 keeps the script's name out of a backgrounded argv).

- **`u2_gate`**: `--arch production --device cuda --steps 3000000 --eval-freq 500000`, 2 cycles.
  1. `📒 [EVAL LEDGER] the in-loop eval writes COUNT rows (protocol gen3_eval_protocol_v1_inloop) to …/_ledger` at
     startup; `📒 [EVAL LEDGER] step 500,224: 9 cycle row(s) appended` and `step 1,000,192: 9 cycle row(s) appended`.
  2. `python -m main.eval_ledger audit` → **OK** (18 rows, 2 requests closed, 18 claims, 0 live).
  3. `scripts/check_ledger_vs_results.py`: **18 of 18** rows' `[w, w+l+d]` equal `counts[<opp>]`; every row
     `aborted` 0; the per-team counters sum to the counts (games 100, wins = w, the opponent teams' wins = l).
  4. Cycle wall: `Rust eval core played 900 games in 10.3 s` and `10.2 s`; TB `rust_eval/cycle_wall_s` 10.255, 10.248.
     Against the bottleneck profile's cycles (1,400 games: 17.2 s, 37.4 s (nsys), 21.8 s, 18.8 s) these are **different
     games** (900 vs 1,400, no pool); per trainee decision the rate is 2.70k and 2.65k decisions/s here against 2.69k
     in the profile's first cycle (1.9–2.2k in its later ones). **Within the profile's range; the ≤ 0.3 s
     `GameSink` cost is below this resolution (UNVERIFIED)**, since there is no same-shape pre-U2 control.
- **`u2_sprt`**: `--promotion-sprt --eval-freq 200000 --self-play-start-wr 0.0`, a FRESH run (a fork's pool is
  seeded from its parent's; this run seeds its own at the first eval, which `--self-play-start-wr 0.0` allows), 3 cycles.
  1. Cycle rows 9 / 10 / 11 (the pool's sentinels add a row each); the pool seeded at 200,192, candidates 400,026 and
     600,179 tested.
  2. Both SPRT tests ended at their first look: `ACCEPT (bound) after 40 pairs in 1 batch(es)`, LLR +3.332 and +3.199.
  3. **One decision row per candidate** (`u2_sprt:sprt:400026`, `:600179`; `decisions` 2) and the batch rows: 1 for the
     first (one sentinel), 2 for the second (two). `scripts/check_sprt_ledger.py`: each candidate's batch rows hold
     80 games = 2 × 40 pairs, their score (0.875, 0.80625) equals the `sprt_promotion.jsonl` verdict line, the decision's
     verdict and consumed-row count agree. The archive's `audit` → **OK** (51 rows, 9 requests, 2 decisions, 0 live claims).
  4. `scripts/check_ledger_vs_results.py` on this run: **30 of 30** cycle rows match, **sentinel rows included**
     (80 / 100, 60 / 100, 93 / 100 equal `sentinels[].counts`).
- **Limit.** Both SPRT tests ACCEPTED at the minimum sample (a fresh trainee beats its own 200k-step self), so only
  the accept path was exercised; the reject / truncated-at-cap / abandoned-on-restart paths were not.

## Findings (standing rule 7)

- **F-G-1 (budget; decision for the orchestrator).** `fixed_mass` at U3 part 1 costs **+8.7 % `train_ms`** against the
  +5 % pre-registered budget (§3.6): the U8 rule says STOP before any A/B GPU if this holds at the end of U3. U3 parts
  2 and 3 and the X26 heads will add to it. The cost is the GEMMs of the token path (+7.4 of +11.3 ms per
  micro-batch), not the construction. Possible levers are an orchestrator call; none was tried here.
- **F-G-2 (F-X5-22 closed on the GPU).** The unrolled 64-step bisection costs no measurable compile time or run
  time (≈ 0.4 ms per micro-batch); "τ outside the graph" is slower and "28 steps" gains nothing, so neither is worth
  the §3.2 change. Evidence above; the CPU-measured 9 s per construction did not carry to Triton.
- **F-G-3 (T2).** Trainee GPU wait +11.7 % at B = 256 (regime B). The budget's "T2 flush ≤ +3 %" is ambiguous (the
  note's 13.12 ms is regime A with a pool); in that regime only the trainee's 4.48 ms share is X5-sensitive today, +0.5
  ms = about +3.9 % of 13.12 ms if the opponents did not change, which is also over. Regime A is unmeasured.
- **F-G-4 (memory).** Headroom 1,962 to 1,964 MiB without the X26 heads (floor 1,024). With the heads: UNVERIFIED.
- **F-G-5 (cold-start ties).** 5.2 % of live real rows are rule-8 near-ties while δ_θ is zero (exact prior ties).
  They agree across devices anyway; the exclusion shrinks to 0 once δ_θ moves.
- **F-G-6 (coverage).** The parity bank is 69 % late-game rows from a random-policy rollout (see §2). A bank from a
  trained policy would sample r = 1…3 more. The tie structure is covered synthetically.
- **F-G-7 (startup noise).** Two identical blob launches differ by 48 s in the T2 build (245.1 vs 293.1 s) under
  contention. Any future compile-cost A/B on this box needs the same replicates or a quiet box.
- **F-G-8 (commits).** X5 numbers are at `e78884c4` (U3 part 1). `origin/main` already has U3 part 2.
- **F-G-9 (eval gate scope).** Gated at `ecf9eeca`, before the F-ED-22 index fix `e0252dc0`; SPRT accept path only;
  the SPRT leg is a fresh run with `--self-play-start-wr 0.0`, not a fork of a seeded parent.
- **F-G-10 (orphan children).** The trainer's `[LADDER] spawned round-robin update` children
  (`agents.training.snapshot_ladder --promote …`, ~90 % CPU each, with `sim_bridge` grandchildren) kept running
  after the trainer's SIGTERM exit. They are detached by design; this agent killed its own by explicit PID.
  Anyone stopping a real run should expect them.

## Reproduce

All commands run from a worktree with `export PYTHONPATH=$PWD/src` and the `gen3ai_torch28` interpreter; every GPU
command ran under the lease (`GEN3AI_GPU_LEASE_TOKEN_FILE=…`, `scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh <GB>
timeout <s> …`). Scratch (not committed) sat on disk under the job directory.

```bash
# real rows (CPU)
python scripts/gen_real_obs.py --out real_obs.npz --envs 64 --steps 64
# check 2
GEN3AI_TEST_ALLOW_GPU=1 python scripts/cuda_parity.py --obs real_obs.npz --out parity.json
# check 1 + 3: one launch each (drive.sh = samplers + the phase-probe trainer + SIGTERM by PID at N update rows)
#   blob:       phase_probe_run.py --arch production --run-name x5blob_a --steps 6000000 --device cuda --eval-freq 50000000
#   fixed_mass: the same + --belief-tokens fixed_mass --allow-nonproduction-arch
python scripts/read_run.py --tag x5fm_a --dir <scratch> --out results/x5fm_a.json   # per launch
python scripts/summarize.py --dir <scratch> --results results --out results/numbers.json
# the compile / runtime decomposition (one process per arm)
GEN3AI_TEST_ALLOW_GPU=1 python scripts/compile_ab.py --arm {blob,fixed_mass,fixed_mass_iters0,fixed_mass_iters28,fixed_mass_tau_op} \
    --obs real_obs.npz --batch 2048 --bwd [--profile] --out results/compile_ab_<arm>_B2048_bwd.json
# eval gate (worktree at ecf9eeca)
phase_probe_run.py --arch production --run-name u2_gate --device cuda --steps 3000000 --eval-freq 500000
phase_probe_run.py --arch production --run-name u2_sprt --device cuda --steps 3000000 --eval-freq 200000 --promotion-sprt --self-play-start-wr 0.0
python -m main.eval_ledger audit --root $GEN3AI_MODELS_DIR/_ledger ; python -m main.eval_ledger show --root … --json
python scripts/check_ledger_vs_results.py --ledger …/_ledger --run-dir …/models/<run>
python scripts/check_sprt_ledger.py --ledger …/_ledger --run-dir …/models/u2_sprt
```

`phase_probe_run.py` is `../bottleneck_profile_2026-10-03/scripts/phase_probe_run.py`, unchanged. The scripts here
import the code at `e78884c4`; a later commit may need small edits to re-run them.

## Files

- `scripts/`: `gen_real_obs.py`, `cuda_parity.py`, `compile_ab.py`, `read_run.py`, `summarize.py`,
  `check_ledger_vs_results.py`, `check_sprt_ledger.py`.
- `results/`: `numbers.json` (the four launches joined), `x5{blob,fm}_{a,b}.json` (+ `.drive.log`), `parity.json`
  (+ `parity.log`, `parity_run1_tie_check_bug.json`), `compile_ab_*.json` (+ `_prof.json`),
  `u2_gate_ledger_check.json`, `u2_sprt_cycle_check.json`, `u2_sprt_ledger_check.json`, `u2_archive_ledger_audit.txt`,
  `u2_archive_ledger_show.json`, `u2_gate_ledger_audit.txt`, `u2_gate_ledger_show.json`, `eval_gate_log_excerpts.txt`,
  `eval_gate_tb_rust_eval.json`, the two runs' `eval_results.jsonl` and `u2_sprt_sprt_promotion.jsonl`.
