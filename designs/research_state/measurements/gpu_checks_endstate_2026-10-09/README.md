# The deferred GPU checks on the END-STATE graph: P, R, E (2026-10-09/10)

The GPU checks that four CPU-only builds deferred: the static port (`static_port_2026-10-09/`, v147), F6b's
principled operator reductions (`op_reduction_f6b_2026-10-08/`, v146), F7b's speed physics
(`speed_physics_f7b_2026-10-07/`, v143) and the static-recovery levers (`static_recovery_2026-10-09/`, v150).
They ran under the GPU lease, on three configurations at **`43a59bbd`** (main's HEAD when the work started):

| cfg | argv (on top of `--compile-trainer --device cuda`) |
|---|---|
| **P** | `--arch production` |
| **R** | `--arch static_recovery` |
| **E** (end state) | `--arch static_recovery --move-resolution on --speed-physics on --no-value-threat-inject --op-reduction principled --obs-facts v1 --allow-nonproduction-arch` |

`checkargs` gives P and R an empty ARCH diff against their declared surface. For E it lists exactly the five
levers, and the recipe matches `recipe.fresh` in all three. The box: RTX 3080 Ti (12 GiB), torch 2.8.0+cu126,
fp32 'highest', the production recipe at N = 256. Tags are **MEASURED** unless marked otherwise.

## Verdicts

| check | P | R | E |
|---|---|---|---|
| 1. compiled == eager, forward + backward: the R1 startup gate (fresh + perturbed rungs), then the update-10 canary | **PASS** (worst per-param 1.48e-5, canary 2.46e-5) | **PASS** (1.53e-5; canary 1.88e-5) | **FAIL at startup**: `[CompileSentinel] FATAL`, `obs_facts_inject.choice_proj.weight` 1.13e-2 > the trained bar 9.88e-3 (perturbed rung) |
| 2. T2 inference service CUDA-graph build; graph and recompile counts | **PASS**: T2 up (the service's parity gate runs every slot × bucket), 3 `decide` graphs (one per bucket), R1 4 graphs incl. the gate's, `recompiles_after_lock` 0, `cache_limit_hits` 0 | **PASS**, same counts | T2 **PASS** (up in 710.5 s); R1 never locked |
| 3. a real launcher launch: startup lines, the learner freeze, first updates, no FATAL | **PASS**, 13 updates, clean SIGTERM stop (`[ABORT] Checkpoint saved`) | **PASS**, 13 updates, clean stop | **FAIL** (check 1's FATAL; exit FATAL_CONFIG, no restart) |
| 4. cost read | below | below | none (it never reached an update) |

**E at `c0f528b4` (P_prod, the fact-completion build): the same offline reproduction reads CLEAN** (worst
1.17e-5; §2.5). The defect is a property of E's graph at `43a59bbd`, and the next build's graph no longer forms it.
Only ONE offline weight state was read, so a real E launch at the closing-test commit, run to its update-10
canary, is still the proof (**UNVERIFIED** until then).

## 1. Cost: R vs P (`results/numbers.json`, `results/{P,R}.json`)

Each launch was stopped after 13 update tables, which includes the update-10 canary. A **steady** update is
quiet (no windowed contention factor ≥ 1.05 over its rollout and train window) and is neither the first update,
the canary update, nor the stop's interrupt-time row. Standing rule 8 excludes the rest, and they are listed in
the JSONs. The box carried a CPU probe battery (about 10 busy cores) during P's startup and part of its run,
so P kept 5 steady updates and R kept 2. The median over every non-warm-up update (contended ones included) is
given beside each as DESCRIPTIVE.

| quantity | P | R | R vs P |
|---|---|---|---|
| `train_ms` per update | 44.95 s (n 5; all 44.95) | 51.59 s (n 2; all 51.23) | **+14.8 %** (all +14.0 %) |
| cycle rows/s (rollout + train, per update) | 1,843 (all 1,779) | 1,632 (all 1,556) | **−11.4 %** (all −12.5 %) |
| T2 GPU wait per host step (B = 256) | 6.27 ms | 6.96 ms | **+11.0 %** |
| T2 host flush per host step | 0.77 ms (all 1.09) | 0.81 ms (all 0.93) | +4.6 % (all −14.9 %): within noise |
| update peak reserved | 8,258 MiB | 8,810 MiB | +552 MiB (+6.7 %) |
| `UpdateFit` headroom (card free after the dry update; floor 1,024) | 2,666 MiB | 2,110 MiB | **−556 MiB** |
| dry first update: peak allocated | 6,416 MiB | 6,873 MiB | +457 MiB |
| T2 build (CPU-bound compile) | 846.9 s | 582.9 s | **not comparable**: P's startup was contended (95 of 121 windows ≥ 1.05, max 2.19), R's was not (7 of 88) |
| R1 reset + prewarm | 240.0 s | 183.7 s | not comparable (same reason) |

- `train_ms` is GPU-bound and stable (P 44.89–45.10 s over 11 updates, contended or not), so the +14.8 % holds
  against the contention. R's trunk round and levers cost it. The CPU read (`static_recovery_2026-10-09` §3)
  priced the arm at +31 % matmul FLOPs and about +10 % eager CPU forward.
- **Not measured:** the isolated cost of `--trunk-layers 3` (R with `--trunk-layers 2`). That launch was
  stopped in its startup to free the GPU for E's diagnosis, so its run dir held nothing. E's cost: E never
  reached an update. Compile time vs P: needs a quiet-box replicate of P (F-GE-7).

## 2. E's startup FATAL, diagnosed

### 2.1 It reproduces exactly, offline

`scripts/init_match.py`: the golden learner built for R's namespace at the trainer's `--seed` 42, unperturbed,
is **bit-equal to the R launch's own init record** (276 of 276 `param_init_sha256`). So the same build of E's
namespace is the E launch's starting weights. On it, the R1 gate's own perturbed rung
(`parity_probe.perturbed_parameters(seed=rung_seed(0), scale 0.05)`) and the gate's batch reproduce the
launch's reading to the digit: `choice_proj.weight` 1.1269e-2. That holds in `noise_cfg.py`, in `order_probe.py`
(two independent compiles in one process, with and without the eager arm's argument tuple) and in
`run_ablation_sites.sh`'s control.

### 2.2 It is a COMPILED error, not fp32 noise (`results/noise_E_s42.json`)

This is the matched-noise method of `k6_k8/r1_noise/` (the evidence the R1 bar rests on), with a float64 CPU
reference:

| | compiled vs fp64 | CUDA eager vs fp64 | CPU eager vs fp64 | `aot_eager` vs eager |
|---|---|---|---|---|
| `obs_facts_inject.choice_proj.weight` | **1.13e-2** | 2.2e-6 | 1.7e-6 | 0 (bit-equal) |
| about 15 others (`hp_type_belief_head.reinject_*`, `op_content.outgoing_*`, `edge_bias.d2_map`, the belief norms, …) | 2–5e-3 | ≤ 1.6e-6 | ≤ 7e-6 | 0 |

Both correct fp32 implementations sit at about 1e-6. The compiled gradient is about 1,000× off on a whole set
of parameters, and 5,000× off on the one that trips the bar. The traced and decomposed graph run eagerly
(`aot_eager`) is bit-equal to eager, so the error is in Inductor's code generation. The startup gate did its job.

Two other things were ruled out:
- `noisy_or`'s `torch.prod` backward, compiled vs eager, alone (`scripts/prod_backward_probe.py`): about 1e-7,
  with exact zeros (a certain KO) and with values near 1.
- ObsFactsInject's compiled backward alone, with the real upstream gradient captured from the full step
  (`scripts/isolate_obs_facts.py`): exact.

The error therefore comes from the rest of the graph.

### 2.3 Which lever: a CONJUNCTION, not one flag (`results/ablation_worktree_code_final.log`)

These runs used one process per variant, one job at a time, on this checkout's own code (imports printed). Each
row reverts ONE lever of E. The value is the gate's worst per-parameter relative error, compiled vs eager CUDA:

| variant | worst |
|---|---|
| E (control) | **1.13e-2** (`choice_proj`) |
| E with `--obs-facts off` | 1.6e-5 |
| E with `--move-resolution off` | 1.1e-5 |
| E with `--op-reduction max` | 2.5e-5 |
| E with `--value-threat-inject` (production's value) | 8.6e-6 |
| E with `--speed-physics off` | 6.1e-4 (about 50× healthy, under the bar) |

The defect needs obs_facts, move_resolution, the principled reduction AND the value-threat injection off,
together. It depends on the graph as a whole, not on one lever's code. That fits an Inductor rewrite that forms
only in that exact graph.

### 2.4 Which Inductor transformation: SDPA via the pattern matcher (**UNVERIFIED** on `43a59bbd`)

`scripts/inductor_toggle_probe.py` and `scripts/sdpa_nodes_probe.py` (`results/*_maincode.log`) found the
following. The error vanishes with `torch._inductor.config.pattern_matcher=False`, and with SDPA restricted to
the MATH backend (eager and compiled alike). It survives `split_reductions=False` and deterministic algorithms;
`persistent_reductions=False` moves it to other parameters. E's compiled graph holds 5 memory-efficient SDPA
forward/backward pairs: the two trunk rounds (`team_transformer.py:65`), the extra round (`trunk_depth.py:70`),
and the hidden-opponent belief decoder's self- and cross-attention (`pools.py:177`, the X5 float key mask).

**Caveat:** these scripts ran BY PATH, so `agents` resolved through the editable install to the MAIN checkout,
whose code moved during the hunt (`c0f528b4` landed at 22:49). They are probably right, but they are not proven
on `43a59bbd`. The scripts now put this checkout's `src/` first (`DIAG_SRC` overrides it). The per-site MATH
restriction (`scripts/site_probe.py`) was not run on pure `43a59bbd` code (cap), so WHICH site is open.

### 2.5 At `c0f528b4` the same probe reads clean (`results/E_control_on_c0f528b4.log`)

`git archive c0f528b4` (P_prod, the closing test's commit) was run with the same script, seed and rung:
E's argv reads **worst 1.17e-5**. The fact-completion build changed E's graph (its new levers are off in this
argv), and the conjunction no longer forms. This is one weight state, offline. The proof is a real E launch at
the closing-test commit to its update-10 canary.

**No fix shipped.** Both known global switches (pattern matcher off, MATH SDPA everywhere) would change
production's compiled path, which the closing-test registration forbids. A fix gated to the end-state arm (the
MATH backend at the offending site only when the end-state flags are on) is the fallback, if a launch at the
closing-test commit FATALs again.

## Findings (standing rule 7)

- **F-GE-1 (E, at `43a59bbd`).** E cannot launch with `--compile-trainer`: the startup gate refuses it with a
  real compiled-gradient error (§2), not noise. On the closing-test commit `c0f528b4`, the same offline probe
  reads clean. One real E launch there, to its update-10 canary, is required before E's seeds (**UNVERIFIED**
  until run).
- **F-GE-2 (a defect that depends on the whole graph).** The error forms only for one conjunction of levers.
  Any future lever can make or unmake it, so a CPU smoke cannot stand in for it, and neither can a per-lever
  compile test. The R1 startup gate and the update-10 canary are the only guards, and they worked.
- **F-GE-3 (wide sub-bar compiled error).** In E's failing graph about 15 parameters sat at 2–5e-3 against
  eager's 1e-6, all UNDER the trained bar 9.88e-3. Only `choice_proj` tripped it. A graph whose whole gradient
  is that far off can pass the gate when its worst parameter stays under the bar. The bar was calibrated as 4×
  the largest HEALTHY reading. A matched-envelope rule (compiled vs fp64 within 2× of eager's own) would catch
  it, but costs a float64 arm. Flagged; nothing was changed.
- **F-GE-4 (launcher pin from an un-bootstrapped agent worktree).** The launcher's pinned worktree links
  `deps/pokemon-showdown` to the LAUNCHING checkout's submodule (`main/launcher/worktree.py`,
  `get_repo_root()`). From an agent worktree whose submodule was never initialised, that is an empty
  placeholder, and the child dies in team validation ("Cannot find module …/deps/pokemon-showdown"). The first
  P attempt died this way. `git submodule update --init` plus the dist / node_modules links fixed it here. The
  launcher could link main's checkout (git common dir) instead.
- **F-GE-5 (diagnostic scripts import MAIN's code).** A script run by path gets its own dir as `sys.path[0]`,
  and the editable install then resolves `agents` to the MAIN checkout. The "worktree pytest puts src first"
  rule covers pytest only. Main's code changed mid-diagnosis and silently turned two reproductions into
  non-reproductions. The scripts here now insert this checkout's `src/` first.
- **F-GE-6 (contention).** A CPU probe battery held about 10 cores through P's startup. P's T2 build (846.9 s)
  and prewarm are not comparable to R's, and only 5 (P) and 2 (R) updates read quiet.
- **F-GE-7 (not measured).** E's cost; `--trunk-layers 3`'s isolated cost; the compile-time delta on a quiet
  box; the per-site MATH restriction.
- **F-GE-8 (orchestration).** Three diagnostic jobs run in parallel OOM'd the card (12 GiB). One job at a time.

## Reproduce

Every GPU command ran under the lease: `GEN3AI_GPU_LEASE_TOKEN_FILE=… scripts/ops/gpu_lock.sh
scripts/ops/mem_cap.sh <GB> …`. Diagnostic scripts run from `<checkout>/src`.

```bash
# the real launches (drive.sh = launcher under gpu_lock + sampler + SIGTERM to the launcher's pid at N updates)
scripts/drive.sh rb_gpucheck_<cfg>_<ts> <scratch> 13 --restart-interval-hours 0 --pin-commit 43a59bbd <argv> \
    --compile-trainer --device cuda --steps 3000000 --log-level periodic --run-name rb_gpucheck_<cfg>_<ts>
python scripts/read_run.py --tag <run> --run-dir models/<run> --scratch <scratch> --out results/<cfg>.json
python scripts/summarize.py --results results --cfgs P R --out results/numbers.json
# E's diagnosis (launch-equivalent weights: seed 42, the gate's perturbed rung 0)
python scripts/init_match.py --cfg R --ckpt models/<R run>/final_model_interrupted.zip
python scripts/noise_cfg.py <out.jsonl> E 2048 0 --model-seed 42 --focus obs_facts_inject
python scripts/order_probe.py --cfg E --steps cuda_eager,compile_reuse,compile
bash scripts/run_ablation_sites.sh          # the per-lever reversions (+ the per-site MATH probe, not reached)
python scripts/inductor_toggle_probe.py --toggles base,no_pattern_matcher,sdpa_math,...
DIAG_SRC=<an extracted tree>/src python scripts/order_probe.py --cfg E --steps cuda_eager,compile
```

The throwaway run dirs (`rb_gpucheck_P_20261009` (the dead first attempt), `_P_20261009a`, `_R_20261009a`,
`_E_20261009a`, `_R2_20261009a`) held only this check and were deleted. Their reads are in `results/`, and E's
crash log is `results/E_launch_crash_log.txt`.
