# Training — compile flags

The owner of the detail for `--compile-trainer`, the learner's compile region, the compile sentinel, the startup parity
gate and the in-run canary, the compile caches, and the offline CPU opponent compile. `src/agents/training/CLAUDE.md`
keeps the hazards and points here. The dated incidents, measurements and superseded designs behind these rules are in
`designs/research_state/claude_md_archive/training_compile_flags_history.md`, under the same headings; a section's
"History:" line says which of its parts moved there. A figure in that archive is history, not a claim about the present.

---

## Compiled CPU opponents — the trainer-side path was DELETED (deletion pass U3); the offline compile and its gates remain

Under the Rust env core every policy opponent forwards through the T2 inference service, so the trainer compiles no opponent:
the trainer-side opponent-compile flags, the forkserver preload, the cache prewarm and the cross-process revert quorum are
deleted (`designs/deleted_flags.md`), and **`--compile-trainer` is the only compile flag**. What survives is
`agents.model.compile_opponents.maybe_compile_extractor(model, enabled, label, hide_cuda)`: it `torch.compile`s ONE frozen
model's extractor for an OFFLINE reader. Its callers are the prober's counterfactual view (`main/prober/session/counterfactual.py`;
the `--compile` flag of `main.prober.query`, off by default because a one-off `summary` / `list` never amortizes the ~10-20 s
compile; turn it on for the search-shaped commands `better-line`, `falsify`, `falsify-scan`, `replay-counterfactual`,
`lookahead`) and `compile_cache_benchmark`. `play.py` and the snapshot ladder stay eager.

The contract (pinned by `compile_extractor_test.py`, `extractor_compiles_test.py`, `opponent_parity_test.py`):

- **It patches the BOUND `fe.forward`, never the module** (`torch.compile(module)` prefixes every `state_dict` key with
  `_orig_mod.`). Parameters are graph inputs, so a different checkpoint does not recompile.
- **`hide_cuda` is the caller's explicit declaration**: compiling even a CPU model in a CUDA-visible process takes a CUDA
  context (~252 MiB of card), so a worker passes `hide_cuda=True` (it sets `CUDA_VISIBLE_DEVICES=""`), and a `hide_cuda=True`
  caller in a process that already holds a context is REFUSED, not compiled.
- **Speed floor `_MIN_COMPILE_SPEEDUP = 1.05`**: timed against eager at load and REVERTED below the floor. The measurement is
  the fix, never the floor (lowering it widens a broken instrument; the 2026-08 launches died on one noisy timing pair, the
  same checkpoint reading 0.78x and 6.3x minutes apart): both arms are warmed identically (`_warm_arm`) before either is
  timed, then timed ALTERNATED sample by sample with the round order flipping, and the verdict is the median of
  `_TIMING_SAMPLES` = 5 samples, each a min over `_TIMING_REPS` = 4 forwards. A timing ratio cannot tell a fragmented graph from
  a busy box, so the message prints both series, both medians and the floor and asserts no cause.
- **The warm-up observation carries `observation` AND a float32 `action_mask`** (plus `zero_extra_obs`): dynamo guards on a
  dict's key set and dtype as hard as on shape.
- **Failures are loud, never fatal to the caller**: `DISABLED`, `REVERTED` and a mid-run `FELL BACK to eager` (a ~6x slower
  forward) go through `_compile_warn` (stderr and the launcher event stream). `_eager_fallback_on_error` scopes the fallback to
  the one model and routes every grad-ENABLED call to eager (the prober backprops through this extractor for saliency). There is
  no strict variant.
- **Validation is paid once per process** (`_COMPILE_VALIDATED`): torch keys on the code object, so a second model in the process
  cannot get a different answer; the reuse path still logs `ON (reused this process's validated compile)`.
- **A decision-level parity check runs before the compile is installed** (`opponent_parity.check_opponent_parity`, once per
  DISTINCT WEIGHTS per process, cached under `weights_fingerprint`; a failure is never cached). It compares eager with the
  compiled callable on features, MASKED legal log-probs and V at the learner gate's bars (`compile_trainer._FP32_TOL` via
  `decision_verdicts`) on 16 rows of the committed real-obs fixture at B = 1 with the float `action_mask` and every declared
  extra key (the warm-up's key set, so no recompile), and on fresh weights also on the seeded perturbation ladder. **A miss
  RAISES the learner gate's typed error** (`CompileTrainerError` / `VacuousCompileParityError`) after uninstalling the compile,
  whatever else is set: a timing miss costs throughput, a parity miss is a WRONG opponent. Any other exception from the check
  takes the warn / fall-back path; a policy without the Gen3 heads is refused.
- **Its cache is hermetic** (`ensure_hermetic_cache`, below). **BLAS threads are pinned** for any process that runs CPU forwards
  beside others: `train_rl_agent.py` sets `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` to 1 at import
  (`setdefault`, before torch loads) and `launcher/child.py` exports them; unpinned, N workers spawn N x cores threads (a ~38x
  cliff). `src/main/thread_pinning_test.py` pins it.

The extractor's device x grad compile matrix is owned by `extractor_compiles_test.py` (Inductor's CPU backend emits C++, its CUDA
backend Triton, so a green CPU cell says nothing about a CUDA cell): CPU forward compiles (the offline opponent); CPU
forward+backward lowers on torch 2.8 (`test_cpu_backward_still_does_not_compile` is keyed per torch, `_CPU_BACKWARD_LOWERS`), but
no CPU learner has ever been gated; CUDA forward (eval / inference) and forward+backward (R1 below) compile. Each compile cell
runs by default (`GEN3AI_SKIP_COMPILE_TESTS=1` opts out). **The CUDA cells SKIP under a normal `pytest` run** (the root
`conftest.py` hard-sets `CUDA_VISIBLE_DEVICES=""`, and an Inductor CUDA compile cannot run with the device hidden) and refuse a
busy card; they run only with a lease: `GEN3AI_TEST_ALLOW_GPU=1 pytest src/agents/model/extractor_compiles_test.py -q`.

History: archive, same heading and its subsections (the saturated per-forward win, the `suppress_errors` removal and the
`species_posterior` spelling, the June 0.70x attempt, the forkserver preload and its wedge, the startup and promotion costs, the
parity check's cost and revert proofs, the 2026-08-24 floor failures with the retracted beliefs, the quorum), and "Every
non-training model can use it" (the consumer table).

## The HERMETIC per-run compile cache (K3, `gen3_hermetic_compile_cache_v1`, 2026-09-30)

**The rule (owner 2026-09-29: "always initialize from a fresh setup").** A cache KEY can omit a setting that changes
the artifact (the K1b fault below), so no process trusts a box-wide cache. `agents.model.compile_cache` is the one
module and every compile site calls it. **A new `torch.compile` site must call `ensure_hermetic_cache()` first**
(`compile_cache_test` fails a production module that does not, and fails any string literal that names a box-wide dir).

| process | its compile cache | reused when |
|---|---|---|
| a TRAINING RUN whose code checkout is CLEAN (the default) | the SHARED entry for its key, `~/.cache/gen3ai/compile_shared/<key>/{inductor,triton}`; T2's AOT packages in `<run>/compile_cache/t2_aot/` | by ANY run with the same key (next section) |
| a TRAINING RUN on a DIRTY tree, an unnameable commit, or opted out (`GEN3AI_SHARED_COMPILE_CACHE=0`) | `<run>/compile_cache/{inductor,triton,t2_aot}`: `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `GEN3AI_COMPILE_CACHE_DIR`, inherited by every child (env core, eval subprocess, T2) | only by the run's OWN restart (`--model` = a checkpoint this run wrote, `fork_lr.is_same_run_checkpoint`) with a matching `stamp.json`; a fresh launch or a fork WIPES it to EMPTY |
| a pytest process (controller and each xdist worker) | a fresh `gen3ai_pytest_compile_<pid>_*` dir under the SCRATCH root (root `conftest.py`) | within that process's session only; deleted at unconfigure |
| anything else that compiles (a benchmark, the prober, an offline meter, a standalone eval) | a fresh private `gen3ai_compile_<pid>_*` dir under the SCRATCH root (`ensure_hermetic_cache`) | never; deleted when its creator exits |

- **The SCRATCH root is the REAL DISK**: `$GEN3AI_SCRATCH`, else `~/.cache/gen3ai/tmp`, never `/tmp` (tmpfs is RAM with a
  1,048,576-inode cap; a compile cache is thousands of files and up to 1.5 GB). `utils.scratch.scratch_root` is the one
  helper and REFUSES a tmpfs / ramfs root. Disk vs tmpfs cost no detectable startup time.
- **THE STAMP** (`<run>/compile_cache/stamp.json`): the code commit (`utils.git.get_git_hash`, a pinned child's PIN),
  whether `src/` is dirty, `torch.__version__`, the interpreter, and `compile_config.config_row_hash()` (the
  `COMPILE_CONFIG` row, data only so stamping never imports the sentinel). Any differing field WIPES; a dirty or unnamed
  tree NEVER reuses (a sha cannot name it); an unreadable stamp WIPES. The decision prints as
  `🧊 [CompileCache] CREATED | WIPED | REUSED <root> — <reason>`.
- **Where it is declared**: `main.train.lifecycle._declare_compile_cache`, called in `train_rl_agent.main` the moment
  `model_dir` exists, before the vec env, T2 or `build_and_train`. A cache value already in force then (torch exports its
  default on a process's FIRST compile) means something compiled first: a typed `CompileCacheError`, `FATAL_CONFIG`.
  Torch's default `/tmp/torchinductor_<user>` and the retired `/tmp/gen3ai_inductor_cache` are never honoured.
- **The parity gates still run at EVERY start**: the cache decides only what may be loaded, never whether outputs are
  verified (the region gate, T2's startup gate).
- **Temp caches never outlive their owner**: the creator deletes its dir (`atexit`; pytest at `pytest_unconfigure`), and every
  new declaration SWEEPS dirs whose creator PID is dead, so a SIGKILL leaks only until the next compile anywhere on the box.
- **A crash loop on a corrupt cache**: delete `<run>/compile_cache/`; the next start is cold and correct (it is regenerable,
  so `models/` retention may delete it freely).
- **Cost**: a cold learner start (gate + prewarm) is minutes and T2 about three; a warm restart is ~0.3x of that (a 2.8 run's
  cache is ~2 GB). A fresh per-run launch is always cold by design; the shared entry below removes that for a clean checkout.
- 🚨 **A compile-path change runs the compile files' `slow` tier under torch 2.8 before shipping**: the first CPU
  codegen of the production extractor in each pytest process is cold (70-90 s), so its codegen-numerics tests are `slow`.
  Run `python3 -m pytest src/agents/model/extractor_compiles_test.py src/agents/model/compile_control_test.py
  src/agents/inference/service/service_cuda_test.py -m slow` under `gen3ai_torch28` (the CUDA file under
  `scripts/ops/gpu_lock.sh`); the rows land in `designs/ops/slow_tier_status.json`, which the routine gate reads. The
  both-torches half of this rule is retired (torch 2.5.1 is legacy-resume only).

History: archive, same heading (the tmpfs inode exhaustion, the ext4-vs-tmpfs startup table, the cold / warm benchmark
tables on torch 2.5.1 and 2.8).

### The SHARED compile cache, by KEY (`gen3_shared_compile_cache_v1`, T25, 2026-10-10)

**Owner, 2026-10-10: "Shared cache is fine as long as we check the git hash and the workspace is clean."** For TRAINING
RUNS only, a clean checkout compiles into the SHARED entry for its key (`agents.model.compile_cache_shared`, called from
`compile_cache.prepare_run_cache`); everything else keeps the per-run rule above. Pinned by `compile_cache_shared_test.py`;
the root `conftest.py` sets `GEN3AI_SHARED_COMPILE_CACHE=0` so no test touches the operator's shared root.

**THE KEY** is the first 32 hex of the SHA256 of the fields below; the full fields are written into the entry's
`stamp.json` and compared on every open, so a mismatch is a MISS (the per-run cache), never a hit:

- K3's per-run stamp, verbatim (code commit, `code_dirty`, torch, interpreter, `config_row_hash()`: the CONFIG ROW is the K1b
  protection, the settings the adapter pins that Inductor's own key omits);
- the commit again, `git rev-parse HEAD` in the checkout the code was IMPORTED from (`utils.paths.repo_root()`), which must
  agree with `$LAUNCHER_GIT_HASH` when set (a pinned child imports its pin's worktree while standing in main);
- python's full version, triton's version, torch's CUDA version, each GPU's name + compute capability + driver (`nvidia-smi`, no
  CUDA context) and `$CUDA_VISIBLE_DEVICES` (the artifacts are kernels for one toolchain and one card; an unreadable GPU does
  not share);
- every `TORCHINDUCTOR_*` / `TORCHDYNAMO_*` / `TORCH_COMPILE*` / `TRITON_*` / `PYTORCH_*` / `TORCH_CUDA_ARCH_LIST` /
  `INDUCTOR_*` variable except the two cache-dir ones (an environment knob can change codegen without changing the graph);
- the run's compile surface: `extractor_arch.arch_toggles_from_args(args)` plus `COMPILE_ARG_KEYS` (batch size,
  `compile_trainer`, `device`, `device_batch`, `grad_checkpointing`, `learner_cudagraphs`, the `t2_*` keys, `trunk_precision`).
  A new trainer flag whose name says compile / precision / device / batch / T2 FAILS
  `test_every_compile_flag_is_keyed_or_declared_irrelevant` until it is keyed or declared irrelevant; a key added later enters
  only when SET (`COMPILE_ARG_KEYS_WHEN_SET`), so existing runs keep their key. The compile REGIONS are code, named by the commit.

**THE CLEAN RULE** (`workspace_state`, run in the CHILD's checkout, i.e. `repo_root()` of the imported code):

- No tracked change anywhere (staged or not) and no untracked file under `src/`. Both reads are `git status` runs with
  the submodule ignored and the second restricted to `src`.
- Untracked files OUTSIDE `src/` do not matter: every process imports from `src/` alone, and whatever data reaches the
  model reaches it as tensor values or shapes, which Inductor's own key covers.
- No IGNORED or untracked IMPORTABLE file under `src/` (`untracked_importables`: every `.py` / `.pyc` / `.pyo` / `.pyd` /
  `.so` / `.pth` / `.pyx` / `.pxd` git does not track), because `git status` never reports an IGNORED file. Two directory
  names are skipped, each for a stated reason: `__pycache__` (CPython validates each `.pyc` against its tracked source;
  a sourceless `.pyc` outside it IS counted) and a Cargo `target/` (built from tracked crate sources; it feeds
  observations, never code, to a compile).
- The `deps/pokemon-showdown` submodule is EXCLUDED: no torch compile reads it, and the launcher's pinned worktree
  replaces it with a symlink, which makes a plain `git status` there exit 128.
- A dirty tree, an unknown commit, a `$LAUNCHER_GIT_HASH` disagreement, an unreadable GPU, a missing compile surface or
  the opt-out falls back to the per-run cache with ONE line: `🧊 [CompileCache] NOT SHARED — <why>; this run compiles
  into its own hermetic cache`.

**Safety.**

- **Entry creation is atomic**: a `.new-<key>-<pid>-*` dir gets its lock file, stamp and subdirs, then ONE `rename` publishes
  it, so an entry is never visible without its complete stamp. Torch's and triton's own writers are atomic renames too
  (`test_torchs_own_cache_writes_are_atomic_renames`), except Inductor's C++ link: **a process SIGKILLed mid-link can leave a
  truncated `.so`**, which the next process FAILS TO LOAD (loud, never a silent wrong kernel). Recovery: delete that entry, or
  start with `GEN3AI_SHARED_COMPILE_CACHE=0`.
- **Live use is a LEASE**: every process that compiles into an entry holds a SHARED `flock` on its `.inuse.lock`; the kernel
  releases it at death. **Pruning** runs at every shared start, least-recently-used first (`.last_used`) until the root is under
  its cap (`GEN3AI_COMPILE_SHARED_CAP_GB`, default 20 GiB); an entry is removed only under an EXCLUSIVE non-blocking `flock`
  and by renaming it to `.trash-*` first, one pruner at a time (`.prune.lock`); an over-cap entry in use stays, reported.
- **The gates still run on a hit; a hit only skips compile time.** R1's startup gate, the prewarm, the lock and the
  update-10 canary, and T2's startup parity gate at every slot x bucket run exactly as on a cold start (pinned at the call
  level under a real shared HIT).

**Where things land.** The run's own `<run>/compile_cache/` keeps `stamp.json` (`"mode": "shared"`, the entry, the key) and
`t2_aot/` (T2's packages are `mkdtemp`-unique and never reused: `$GEN3AI_T2_AOT_DIR`). A same-run restart whose previous start
was shared and whose tree is now dirty WIPES the run-local dir and starts the per-run cache cold. The startup line is
`🧊 [CompileCache] SHARED-HIT | SHARED-NEW <entry> — clean tree, key <12 hex>; …`. Knobs:
`GEN3AI_SHARED_COMPILE_CACHE` (`0` / `off` / `false` / `no` = off; unset / `1` / `on` = on; anything else = off, said in
the line), `GEN3AI_COMPILE_SHARED_DIR` (the root; tmpfs refused), `GEN3AI_COMPILE_SHARED_CAP_GB`.

**UNVERIFIED:** the end-to-end saving of a shared hit (a cold vs warm-shared startup on `--arch endstate`) was specified but
never run; the parts it would sum are measured separately (warm ~0.3x cold).

History: archive, same heading (the planned three-launch measurement and its commands, the K1b argument, the
submodule-symlink exit-128 measurement).

## FP32 matmul precision — the ONLY precision (TF32 retired; `--matmul-precision` DELETED)

Every fp32 matmul in the trainer process runs at PyTorch's own default, `torch.get_float32_matmul_precision() ==
"highest"` (full FP32, no TF32). Nothing in the trainer sets it and there is no flag. TF32 was retired by the owner
(2026-10-01; deletion pass K2): its ~1e-2 rounding cannot be checked deterministically, it had cost a startup crash and a
second gate graph, and the T32b read found it futile (+2.48 % GPU time per step, ledger 2026-10-01). What stays:

- **The provenance stamp.** `metadata.json` records `matmul_precision`, read back from torch (always `highest` now;
  `main.train.run_io._matmul_precision`).
- **Every parity gate refuses any other precision.** The region gate (`compile_regions.gate_regions`), K9(b)
  (`consistency.behaviour_gate`, `UndeclaredPrecision`) and T2's parity judge each call
  `parity_probe.unmeasured_precision()` and raise their own typed error, so a process set to `high` or `medium` is
  refused instead of judged at bars measured elsewhere. The inference service's freeze check and `torch_state_guard`
  (a test that leaks the precision or `allow_tf32` FAILS) are unchanged.
- **A run that recorded TF32 resumes pinned or not at all.** A resume or fork whose `metadata.json` recorded
  `matmul_precision != "highest"` (one run on record) is refused `FATAL_CONFIG` naming the pin
  (`model_version.retired_levers.RETIRED`, `LAST_COMMIT_K2`: any commit at or before it still has the flag). A recorded
  argv that types the flag fails argparse on HEAD and must run pinned (`designs/deleted_flags.md`);
  `learner_benchmark.build_trainer_argv` drops flags the parser no longer knows.

## Compiled GPU trainer (`--compile-trainer`, DEFAULT ON for cuda)

`torch.compile`s the LEARNER, the CUDA forward **and backward** the PPO step runs. **The learner compiles ONLY as its
DECLARED REGION, R1 the micro-step** (K8 below), installed, gated, prewarmed and locked by
`compile_trainer.arm_compile_sentinel`. The trainer's own compile step (`preflight_compile_trainer`) compiles nothing: it
refuses a learner the region cannot serve (no features extractor, a non-CUDA device, no `_micro_static`) and says what
will be compiled. **HEAD runs torch >= 2.8 only** (`utils/torch_floor.py`; Lane K1 below). The extractor-only compile, its
gate, the CUDA trunk split and the compiled rollout region R0 are deleted; their history is in the archive.

🚨 **The default is AUTO, not a flat `True`**, because the flag REFUSES a non-cuda device and `default=True` would turn
every working CPU invocation (the `--debug` smoke, a laptop, CI) into `FATAL_CONFIG`. It is resolved by
`main.train.compile_flags.resolve_compile_trainer_default` (pure, injectable, unit-tested without a card; re-exported
through the `train_rl_agent` hub):

| resolved device | `--debug` | default |
|---|---|---|
| `cuda` / `cuda:N`, or `auto` on a box with a card | no | **ON** |
| `auto` with no card, `cpu`, anything else explicit | no | OFF |
| any device, including an explicit `--device cuda` | **yes** | **OFF** (a smoke proves the pipeline in ~1 minute; a multi-minute Inductor compile defeats that) |

An explicit `--compile-trainer --device cpu` still exits `FATAL_CONFIG`; `--no-compile-trainer` is the opt-out. The device
is only half the default: `resolve_compile_trainer_auto` also runs `check_shape_stability`, and on a refusal **leaves
the default OFF and says why** (`⚡ --compile-trainer would be ON by default here, but this config cannot take it —
leaving it OFF rather than refusing to launch`) instead of refusing to launch. The rule, for any future default: **a
default yields to the config the user typed and announces it; an explicit flag refuses.** When the default resolves ON it
is announced (`⚡ --compile-trainer ON by default (device=cuda)`), launcher attached or not. Pinned by
`src/main/compile_defaults_test.py`.

**Runtime perf knob**: never versioned, never in `check_compatible`, NOT inherited on resume. With the AUTO default a
flagless cuda resume gets it ON, so it is `--no-compile-trainer` you re-pass each launch, not the flag.

**Fail-loud by design.** A silent fallback to eager trains correctly and just runs ~1.75x slower forever, so every failure
is fatal (`CompileTrainerError` -> `TrainExitCode.FATAL_CONFIG`, so the launcher gives up instead of restart-looping),
and the flag has no strict variant. The refusals:

| refusal | where | why |
|---|---|---|
| `--device cpu` | `preflight_compile_trainer` | the compiled learner is gated and measured on CUDA only; no CPU compiled learner has ever been gated |
| a learner without the micro-step (`_micro_static`) | `preflight_compile_trainer`, `arm_compile_sentinel` | R1 cannot be built, and there is no smaller fallback compile |
| a graph break inside a region, a compile error | the region gate | `fullgraph=True`; bisect the op (the whole "torch cannot compile our model" story was ONE op: `src/agents/model/CLAUDE.md`, the `species_posterior` precedent) |
| a region disagrees with eager on REAL rows | the region gate | a faster wrong model is not a win |
| an undeclared signature, a cache-limit hit, a region run eager | the sentinel | a silent stall or a silent ~2x slowdown |
| an UPDATE that does not divide by `--batch-size` | `check_shape_stability` | below |

**`check_shape_stability`** (pure, unit-tested) refuses an UPDATE size (`--rollout-target-samples` under the
complete-game trigger, `n_steps * n_envs` when it is 0: `rust_rollout.trigger.update_rows`,
`main.train.compile_flags.update_rows_for`) not divisible by `batch_size`: the remainder micro-batch is a SECOND shape the
one compiled R1 graph does not declare, replayed every epoch. The error names a concrete divisor. `_maybe_compile_trainer`
reads the MODEL's own `n_steps` / `batch_size` / `n_envs` when it has them (a resume restores the checkpoint's; the argv's
are inert) with the argv's trigger and target. It is backed by two runtime refusals, `RustCollector._ensure_buffer` (every
buffer build) and R1's dispatcher (one declared row count), so a ragged micro-batch can neither reach the compiled graph
nor run eager (`gen3_r1_no_ragged_v1`). Production is safe by arithmetic (98,304 = 48 x 2048, one micro-batch shape).

**Mechanics.** Never compiles the module: `torch.compile(module)` returns an `OptimizedModule` and prefixes every
`state_dict` key with `_orig_mod.`, which would land in every checkpoint. The regions compile FUNCTIONS over the module and
are stored off it (`model._compiled_micro_step`). Tests: `agents/model/compile_trainer_test.py` (the verdicts and the
preflight are pure functions, so every refusal is testable without a GPU), `compile_regions_test.py` (the regions) and
`extractor_compiles_test.py` (the extractor's own compilability).

**Speed (provenance).** The extractor-only compile measured 155.1 -> 88.5 ms for forward + backward at batch 4096 (**1.75x**;
v76, RTX 3080 Ti, 3 interleaved pairs). The "~+62 % end-to-end FPS" inherits an EXTRAPOLATED ~89 % train share (measured 61 % at
`n_envs=8, n_steps=128, 2 epochs`): **UNVERIFIED** at production shape. R1's own update times are in the K8 section.

History: archive, same heading (the measurement tables, the small-batch / zero-observation validation story, the
per-shape agreement reads).

### The single-graph CUDA miscompile, and the trunk split (`gen3_inductor_trunk_split_v1`, 2026-09-28) — HISTORY

This is why torch 2.5.1 must never run HEAD code. On torch 2.5.1+cu121 the whole extractor as ONE CUDA Inductor graph
disagreed with eager on REAL rows (argmax agreement 70.9 %, extractor-gradient cosine 0.778, features off by up to 7.65)
while an all-zero observation agreed to 4.8e-7, because zeros exercise no masking, no top-K seat selection and no edge
family. Every default-cuda run from `28eaef29` (2026-08-17) to 2026-09-28 therefore rolled out, took its PPO ratio and
its gradients on the diverged function, while eval and opponents used the correct one. A deliberate graph break
(the trunk split) fixed it; it was DELETED on 2026-10-02 (deletion pass K1) with HEAD's 2.5.1 support, and torch 2.8 does
not show the fault. **UNVERIFIED:** the root-cause op was never named. The standing lesson: a parity probe runs on real
rows, never zeros. `utils/torch_floor.py` refuses < 2.8 and a 2.5.1 run resumes pinned to its own commit.

History: archive, same heading (the break-position table, the knobs tried, the speed cost of the split).

### Lane K1 — torch 2.8 (`gen3ai_torch28`), THE DEFAULT; the split OFF there (2026-09-28; default 2026-09-30)

**torch 2.8 is the DEFAULT interpreter for every new run, gate and tool (owner, 2026-09-30); torch 2.5.1 is LEGACY.**
`gen3ai_torch28` (torch 2.8.0+cu126, triton 3.4.0, `environment_torch28.yml`) is the env `scripts/bootstrap.sh` manages.
`gen3ai_stable` (torch 2.5.1+cu121) is kept FROZEN only so an old run resumes on the torch it trained on. **New code
targets 2.8 only.** The 2.5.1-keyed branches (the split, `compile_control`'s 2.5.1 rows, the extractor-only compile,
`aot.py`'s < 2.8 refusal) were deleted on 2026-10-02 (K1, `designs/ops/deletion_pass_manifest.md` R7 / R8) behind ONE
HEAD-side refusal: `src/utils/torch_floor.py`, called first thing in the trainer's `main()`, exits `FATAL_CONFIG` on
torch < 2.8 (so a 2.5.1 checkpoint under `--no-pin`, `--sync-to-main` or a bare `train_rl_agent.py --model` stops with the
reason).

- **Which torch a run uses is RECORDED and ENFORCED.** Every save writes `metadata.json`'s `torch_version` (and a `torch`
  key per `pin_history` span); a run without one trained on 2.5.1. The launcher (`src/main/launcher/torch_runtime.py`)
  resumes or forks a run under its recorded torch, SELECTS `gen3ai_stable` for a 2.5.1 run and refuses a mismatch
  `FATAL_CONFIG` unless `--allow-torch-switch`; `--dry-run` prints the interpreter, its torch and the run's record. A
  bare `train_rl_agent.py --model` is not launcher-checked; the torch floor refuses it on 2.5.1.
- **torch 2.8.0+cu128** (a Blackwell card, sm_120, needs the CUDA 12.8 build; the +cu126 arch list stops at sm_90): the
  sentinel keys on the EXACT version string, so `compile_control` / `compile_config` carry a +cu128 row that COPIES the
  +cu126 one (all hashed internals were equal on the rented box). The shared env stays +cu126; a remote installs +cu128
  (`tools/cloud/vast/requirements.py`; runbook `designs/ops/cloud_gpu_benchmark.md`).
- **The extractor traces `fullgraph=True` on 2.8 in all three modes** (train/grad, train/no-grad, eval/no-grad); 2.5.1
  REFUSED at `forward_guard`'s `WeakKeyDictionary.get`, which `torch._dynamo.explain` does not count as a break. So
  K8's per-region rule (ONE declared `fullgraph=True` region) is a 2.8-only property. `explain` reads 1 graph / 0 breaks
  at every `--log-level`.
- 2.8 with the split off was measured parity-clean on real rows (features 2.4e-5, legal log-prob 9.5e-6, gradient cosine
  1.000000) and 1-2 % FASTER than 2.5.1 with the split (`designs/research_state/measurements/m5_k1/`).

History: archive, same heading (the 2.8 sweep before the flip, the parity tables, the speed A/B, the `fullgraph`
inventory of the rollout-shaped break).

### CUDA-graph capturability of the compiled extractor (M5 T2, 2026-09-29)

The compiled decision forward is captured as a `torch.cuda.CUDAGraph` by the T2 inference service
(`src/agents/inference/service/`), which needs a graph with NO host synchronisation. The compiled forward had exactly ONE
(`team_transformer._event_reference_cells` built a side-sign constant on the CPU and copied it every call); it is built on
the device now. The EAGER forward still makes ~24 host syncs, so only the compiled graph can be captured. Detail and the AOT
verdict: `designs/endstate/program_rust_core.md` §2, "T2 DESIGN".

### Lane K1b — two backwards through one compiled graph (`gen3_donated_buffer_off_v1`, 2026-09-29)

The first real update puts TWO backwards through one compiled graph (`grad_balance._flat_grads` and the per-term
noise-scale probe call `autograd.grad(retain_graph=True)` before `loss.backward()`). On torch 2.8 that failed twice:
`torch._functorch.config.donated_buffer` defaults True from torch 2.6, so a donating backward reuses the forward's saved
activations in place (legal only for single-use backwards); and the Inductor FX-graph cache key is blind to donation, so a
donation-OFF compile was served a donating backward cached by another process.

**The rule: one row of `agents.model.compile_config.COMPILE_CONFIG`**, applied by `compile_control.install()` before the
gate's first compile and restored at `uninstall()`: `donated_buffer = False`, and `torch.compiler.config.cache_key_tag =
"gen3_donated_buffer_off_v1"`, a public knob that IS in the Inductor and AOTAutograd cache keys, so our graphs are never
served an artifact compiled under the default config. Donation OFF runs the same kernels on the same inputs
(`compile_control_test` pins the gradient bit-for-bit on vs off) and costs nothing measurable. `compile_control_test`
(routine) also pins: a donating backward REFUSES a retain-graph probe and the pinned row accepts it, install / uninstall
restore the row, and the cache key carries the tag. `compiled_first_update_test.py` (`slow`, CPU and CUDA cells) drives a full
real first `train()` through the compiled R1 region with `donated_buffer` forced back to torch's default True and asserts the
adapter pinned it False; reverting the pin fails it. **UNVERIFIED** whether any graph of ours has a donating Triton kernel that
overwrites a saved activation without an ATen op (a version counter would not catch it), which is why the tag is not optional.

History: archive, same heading (the two failure traces, the batch-1 eager route and its deletion).

### The startup parity gate (`gen3_compile_parity_real_obs_v1`)

The extractor-only gate was DELETED with the extractor-only compile (K1, 2026-10-02), with its probe loss, its zero-gradient
coverage guard and its speed refusal. What still applies lives in the REGION gate (`compile_regions.gate_regions`, K8
below) and the in-run canary:

- **Real rows, never zeros.** The rows are the committed fixture `src/agents/model/compile_parity_obs.npz` (64 decision
  rows from 8 seeded Rust-core battles, `utils.rust_env.fixture_battles`); regenerate it for any layout with
  `python -m agents.model.compile_parity_fixture --write`. A missing fixture or one of the wrong width REFUSES
  (`FATAL_CONFIG`) rather than falling back to zeros, and
  `compile_trainer_test::test_the_committed_real_obs_fixture_matches_the_live_layout` fails the routine gate the moment the
  layout changes. R1 itself is judged on the K9 golden's labelled buffer (K8 below).
- **The bars** (`compile_trainer._FP32_TOL`, `decision_verdicts` / `train_verdict`; fp32 `highest`, the only precision):
  (pi || vf) features 1e-4, MASKED legal log-probs 1e-3, V 1e-4; for the train graph the loss, the gradient cosine
  >= 0.9999 computed in float64 (fp32 read 1.0005 over 7M entries), and the per-parameter rule: every parameter whose
  eager gradient norm exceeds 1e-3 x the largest is also held to a relative error `||c-e|| / ||e||` at the bar the weight
  regime picks (a global cosine is dominated by the largest gradients: dropping a whole move-cell gradient left it at
  0.9999998). `train_verdict` takes NO default bar. `compile_gate_probe_test` pins the per-parameter rule.
- **Fresh weights run a seeded perturbation too** (`gen3_fresh_parity_probe_v1`). A fresh launch's pointer head is
  zero-init, so every legal log-prob is `-log(n_legal)` whatever the extractor computed and that check cannot fail. The
  gate counts weights as fresh when a quantity is not above its own bar (`parity_probe.spread` / `fresh_reason`), and then
  runs both arms again on every policy parameter plus seeded Gaussian noise (`PERTURB_SEED`), through the SAME installed
  graph (parameters are graph inputs, so nothing recompiles), from a private CPU generator (training's RNG untouched),
  restoring the parameters bit-exactly and re-checking.
- **The LADDER** (`gen3_parity_perturb_ladder_v1`): `parity_probe.PERTURB_LADDER` is (scale, seed) rungs, 0.05 then 0.1,
  8 seeds each, capped at `PERTURB_MAX_SCALE` = 0.1. The gate judges at the FIRST rung whose EAGER arm is informative on
  every rule (tested by running the rules on the eager arm against itself, where only a vacuity guard can fire); no
  informative rung is `VacuousCompileParityError` (`FATAL_CONFIG`) naming each rung. Capped, with seeds instead of larger
  scales, because the bars are absolute and the perturbed network's compiled-vs-eager noise grows steeply (|Δ log π| 4.5e-6
  at 0.1, up to 1.7e-3 at 0.2, 0.66 at 0.5): a critic saturated beyond what 0.1 moves is refused. `parity_probe` also holds
  `LOGPROB_BAR` (1e-3) and T2's greedy near-tie band `NEAR_TIE_BAND` (2x that); a process at any other matmul precision is
  refused (`unmeasured_precision`).
- **Fail-closed.** `decision_verdicts` / `train_verdict` refuse a vacuous comparison (`VacuousCompileParityError`, a
  `CompileTrainerError`, so `FATAL_CONFIG`). Only a real-weights pass waives the check (`allow_vacuous=True`), and only
  after the perturbed pass passed with the check ON.
- **T2's gate does the same per slot** (`agents.inference.service`): a `VacuousParity` re-runs the slot on perturbed weights up
  the same ladder (`ServiceSpec.perturb_ladder`), then runs the real weights with the check waived; each verdict records its
  path (`ParityReport.path`). A refusal exits the trainer `FATAL_CONFIG`, because a restart would replay it.
- **Recompiles are normal; dynamo's `cache_size_limit` (8) is the hazard** (past it dynamo falls back to eager SILENTLY): the
  sentinel below owns that; `error_on_recompile = True` would crash a healthy two-shape run on its second call.

History: archive, same heading (the healthy-vs-miscompile tables, the coverage-guard numbers and the probe loss, the
end-to-end fresh-policy reads, the small-batch validation story).

### The compile sentinel (`gen3_compile_sentinel_v1`, 2026-09-28)

**`src/agents/model/compile_control.py` is the ONLY runtime module that touches `torch._dynamo`** (and the only one that
sets `torch._functorch.config`: the `COMPILE_CONFIG` row of Lane K1b). Add nothing that does elsewhere. It closes two
silent failures: the CACHE-LIMIT FALLBACK (dynamo keeps at most `cache_size_limit` = 8 entries per CODE OBJECT; the ninth
frame runs EAGER with one warning, ~1.75x slower, no error) and LATE RECOMPILES (a stall each, spending the cache toward
the fallback).

**The phases** (`compile_control.control()`, one per process):

| phase | where | what |
|---|---|---|
| 1 `reset()` | `lifecycle._arm_compile_sentinel`, after `_apply_grad_checkpointing` | `torch._dynamo.reset()` drops any graph compiled before the region is installed (it spent production's cache slots); the log line names how many entries it dropped |
| (the startup gate) | `compile_regions.gate_regions`, after the install | R1 compiled vs eager on real rows; its graph is the one the prewarm then hits; a cache-limit hit here is fatal too |
| 2 `prewarm(calls)` | same, after the install + gate | runs every DECLARED signature now (`compile_regions.prewarm_calls`: R1's, once per route), RNG-neutral (`fork_rng`), gradients zeroed |
| 3 `lock(where)` | **the end of startup**, right after the prewarm, BEFORE the first real iteration | the `fail_on_recompile` stance (`torch.compiler.set_stance`) plus a compile-START callback counting every frame compile. `🧊 [COMPILE LOCK] …` states the per-code-object entry count and the headroom to the limit. A rejection is a typed FATAL naming the failing guard(s) (`UNDECLARED SIGNATURE — the failing guard(s): …`); the stance's own rejection says nothing, so the sentinel replays the call ONCE under `error_on_recompile` to read them |
| `release(why)` | when `learn()` returns or raises | the lock's scope is TRAINING: a forward after `learn()` is a new signature by design. Prints `🧊 [COMPILE LOCK] released …` with the post-lock compile count (must be 0) |
| 4 `stats()` | every update | TB `compile/graphs_total`, `compile/recompiles_after_lock` (**must stay 0**), `compile/cache_limit_hits`, `compile/max_cache_entries_per_code`, `compile/locked`, `compile/regression_flag`, `compile/train_ms_vs_lock_baseline` |

The checks run at every rollout end and every update end: `CompileControl.attach` registers the `compile_sentinel` owner on
the loop's DECLARED HOOK TABLE (`agents/training/loop_hooks.py`, `gen3_declared_loop_hooks_v1`) at the `collect`, `update`
and `learn` points, inner to the freeze guard (the update hook runs the canary inside its guard, then `record`). Neither
`_compile_control` nor the table is saved (`_excluded_save_params`). A `RecompileError` at a call site, a cache-limit hit,
or any compile that STARTS after the lock is `CompileSentinelError` -> **`[CompileSentinel] FATAL …` and
`os._exit(FATAL_CONFIG)`** (an exception inside `learn()` would become a restartable CRASH, and a restart would replay it).
The launcher matches `[CompileSentinel] FATAL` / `[CompileTrainer] FATAL` in `_FATAL_CONFIG_SIGNATURES`. There is no
checkpoint save on this exit (the last periodic checkpoint stands); the weights are not wrong, the throughput is.
`CompileControl.wrap_compiled` installs the compiled region and records any lock rejection (sticky,
`rejected_after_lock`) before it propagates, because 2.8's callback runs AFTER the recompile check and a caller's
`except Exception` could otherwise hide it.

**The late-shape rule** (dynamo guards batch, grad mode, `module.training`, the obs dict's KEY SET, dtype/device and read
attributes such as `grad_checkpointing`): every learner-process caller of the compiled forward is either R1 (prewarmed) or EAGER.
`evaluate_actions` is R1; the rollout `policy(obs)`, `predict_values` (any batch, including batch 1), frozen-phi / PBRS, the
capacity cosine and velocity, the cf terms and the win-prob episode-start read run eager and take no region route; eval, the
snapshot ladder and opponents are other processes with a fresh dynamo. `rank_probe` reads R1's own stashes (`last_trunk_tokens`
/ `last_value_cls`) and is HOOK-FREE: a forward hook on a compiled module is a guard, and on 2.8 the hook was silently skipped
inside the compiled frame. A launcher restart is a fresh process (a fresh dynamo cache, gate and lock; the on-disk cache is the
run's own, reused only under a matching stamp).

**The behavioural backstop, independent of dynamo** (`TrainMsWatch`): the baseline is the median `train/train_ms` of the
first 5 post-lock updates after the process's first update (which runs every diagnostic probe), then FROZEN;
`train_ms > 1.4x` it for 3 consecutive updates prints `⚠️ [COMPILE REGRESSION?] …` once and sets
`compile/regression_flag` = 1. WARN, never fatal: contention has the same signature.

**The torch guardrails.** (1) An exact version table (`compile_control._SUPPORTED`, today `2.8.0+cu126` and its
`+cu128` copy, mode `stance`); an unknown torch REFUSES. (2) **A source-hash drift tripwire** (`_SOURCE_HASHES`, the
`instrumented_ppo._verify_upstream_unchanged` pattern): {torch version -> {qualname -> SHA256 of `inspect.getsource`}} over
every dynamo / inductor internal the adapter rests on; a mismatch raises naming each drifted qualname with both hashes.
Re-record with `python -m agents.model.compile_control --record` AFTER re-reading the functions and re-running (3).
(3) The behavioural CONTRACT tests in `compile_control_test.py` (the warning's text and logger, the stance's rejection, the
start callback, the counter / entry list / reset, the per-code-object cache). Hash = the code changed; contract = the
behaviour changed; both are kept. The check runs WHERE A LEARNER COMPILES (`compile_control.require_supported_torch` at the
compile preflight and at every `CompileControl` construction, `FATAL_CONFIG` with the reason), **never at import**: an
import-time check ran in EVERY trainer through the cache stamp and killed non-compile runs (`compile_sentinel_scope_test`).
The 2.8 differences that matter: `config.cache_size_limit` is an alias of `config.recompile_limit` (the warning reads
`hit config.recompile_limit (8)`), the stance's rejection is a plain `RuntimeError` (`_STANCE_REJECT_TEXT`,
`find_recompile_error`), and `config.fail_on_recompile_limit_hit` (a HARD cache-limit failure) is not used.

**Tests** (`compile_control_test.py`, CPU with dynamo's `eager` backend, since the cache, guards, limit and recompile path are
dynamo's): >8 signatures trip the detector; after the lock a new shape raises the typed FATAL at the call site; a SWALLOWED
`RecompileError` and a late FIRST compile are caught; the healthy two-shape alternation locks clean; **the sentinel changes no
numerics** (bit-identical on vs off); the version refusal and the hash tripwire.

History: archive, same heading (why the reset, with its entry-count table; the end-to-end smoke; the full late-shape table
with the iteration-1 signature found 2026-09-30; the per-torch rows and the 2.5.1 mode).

### FUNCTIONAL MASKING — the K8 prerequisite (`gen3_functional_masking_v1`, 2026-09-30)

sb3's `MaskableCategorical.apply_masking` pops its cached `probs` out of `__dict__` and re-runs `Categorical.__init__`, an
object mutation dynamo cannot trace (17-19 graph breaks per update in the K8 inventory), and every `torch.distributions`
object validates its logits and samples with host reads. `agents/model/masked_categorical.py` is the same arithmetic as plain
tensor ops, op for op (the unmasked path's double normalisation, masking from the single-normalised `_original_logits`, the
`-1e8` floor, the masked entropy's `where`, the same `multinomial` call), and the policy's two hot paths use it: `forward` and
`evaluate_actions` (the ride-along readers get a `MaskedPi` answering `.distribution.logits` / `.probs`).
`get_distribution` keeps sb3's object for `predict`. Pinned BIT-IDENTICAL to sb3 (`masked_categorical_test`); the K9
learner golden is unchanged.

### K8 — DECLARED COMPILE REGIONS (`gen3_declared_regions_v1`, torch 2.8, 2026-10-01)

`--compile-trainer` compiles ONLY the learner's DECLARED REGION (`agents/model/compile_regions.py`, the table `REGIONS`),
`fullgraph=True, dynamic=False`, so a graph break INSIDE a region is a startup `CompileTrainerError` naming dynamo's reason
(`FATAL_CONFIG`), never a silent split:

| region | callable | declared signature | why the boundary is here |
|---|---|---|---|
| R1 `learner_micro_step` | `instrumented_ppo.micro_step.micro_step` (`evaluate_actions` + fold steps 1-3a; its backward is AOTAutograd's from the same graph) | train / grad / `batch_size`: ONE row count (a ragged micro-batch is REFUSED by the dispatcher, `gen3_r1_no_ragged_v1`); a second `graphed` route under `--learner-cudagraphs on` (Lever C) | the accumulation group and the optimizer step are per GROUP; the host read follows it |
| R2 rank probe | no forward: reads R1's own stashes on the first micro-batch, spectra on the device (`rank_metrics.rank_probe_from_stash`) | none | one host read |
| R3 optimizer step | `clip_grad_norm_` + AdamW + `zero_grad`, EAGER by decision | none | 0.04 s per update; the KL->LR controller moves lr every update, so a compiled step keyed on it is a new signature per update |
| *eager, declared* | the batch build (micro-batches staged to the device by a prefetch thread, or one resident copy per update: `--device-batch {staged (default), resident, host}`, `instrumented_ppo/device_batches.py`, bit-identical batches in every mode), label alignment, the per-micro packed read (`micro_step.pack`, one deferred transfer at the optimizer step), the fold's eager tail, every cadence diagnostic, logging | none | host-side or data-dependent by nature |

**R0, the compiled rollout region, was DELETED** (P10-E, 2026-10-03, owner-approved): the Rust collector serves every rollout
forward through T2 (`agents/inference/service`, gated on every served bucket at load), so the learner process never called R0,
yet every launch compiled, gated, prewarmed and canaried it (a vacuous check, standing rule 8). `Gen3DualHeadMaskablePolicy.forward`
and `rollout_core` stay, EAGER, for the toy-collector test kit, `masked_categorical_test`, the compile inventory's
`fullgraph_check` and `compile_regions.weights_regime`. A resume onto a checkpoint with a saturated win-prob head is therefore
judged by R1's loss and gradients (`test_a_COLLAPSED_critic_is_judged_by_R1_like_any_other_weights`).

**The sequence.** reset -> `install` (removes any instance-level compiled extractor forward, so every other caller runs eager)
-> `gate_regions` holds R1 to eager on REAL rows -> prewarm exactly the declared signature -> LOCK -> the canary at update 10
and every 100 updates (R1's loss and gradient on the golden rows, confirmed in the same update before it FATALs:
`designs/training/learner_lifecycle.md`; `compile_canary._regions`). ONE startup gate per region
(`gen3_one_gate_per_region_v1`). The gate's bars are R1's loss rel <= 1e-4 and gradient cosine >= 0.9999 plus the
regime-selected per-parameter bar below, with a perturbed pass on fresh weights; K9(b) (the learner's log pi against the
rollout's stored log-prob) and the learner golden (test time) cover the rest. There is no speed refusal at startup. The rows:

- **R1 ALWAYS uses the K9 golden's labelled buffer** (`gen3_r1_golden_rows_always_v1`). A key the run declares that the golden
  lacks takes its DECLARED placeholder (`compile_regions.fill_value`: the label inventory's `host_const`, e.g. the fork arm's
  `fork_pg_m` = 1.0, or the extra-obs-keys registry's zeros); a golden key the run does not declare is dropped; anything else
  (a shape mismatch, an undeclared key, an unreadable buffer) is `FATAL_CONFIG` naming it. The old zero-label fallback
  blanked the critic / intent / belief losses and left ~43 parameters unjudged on every fork run.
- **When a key was filled, R1's judged parameter set is held to the golden rows' own** (`gen3_r1_judged_set_v1`, one extra
  eager micro-step without the key): a parameter the golden rows judge CLEARLY (gradient-norm ratio > `JUDGED_SET_BAND` = 2x
  the floor) that the run's rows do not is `FATAL_CONFIG`. The gate line reads `R1 judged set >= the golden rows' own: all N
  parameters …`.

**No silent fall-back to eager** (`gen3_no_silent_eager_v1`; owner: "I just most care that we don't have silent performance
regressions from it not being fully compiled"). The run-time defences are the PRIMARY ones:

- **Every region call goes through its dispatcher** (`compile_regions.install`), which takes the COMPILED route. There is no
  declared eager route (a future one counts under `<region>_eager_<why>`, `agents/model/region_calls.py`).
- **A compiled route that silently ran eager is a typed FATAL**: each region body calls `note_eager_body`, a no-op inside a
  dynamo trace, and a compiled graph never re-enters the Python body, so a body that runs during a compiled-route call means
  the region ran EAGER (dynamo disabled, a swallowed compile error, a skipped frame, the `force_eager` stance).
- **The lock refuses the three switches** that make dynamo run eager without an error (`CompileControl.violation`):
  `torch._dynamo.config.disable`, `suppress_errors`, and a stance other than `fail_on_recompile`.
- **The run asserts its compiled inventory** after the prewarm (`assert_inventory`: one cache entry on R1's code object); a
  learner without the micro-step is refused (`CompileTrainerError`) instead of compiling less silently.
- **Every update records** `lifecycle/compiled_region_calls`, `lifecycle/eager_fallback_calls`, `lifecycle/eager_share` and
  `lifecycle/update_wall_s`; the `TrainMsWatch` drift warning (above) is never fatal.
- A ragged micro-batch, `torch.compiler.disable` or a graph break inside a region, a cache-size-limit fallback and a guard
  failure after the lock are all refused at startup or typed FATALs.

**Performance is checked by STRUCTURE plus a milestone benchmark; there is no wall-clock gate** (owner, 2026-10-02): a timing
test is noisy, needs an idle box and goes stale (`compiled_perf_guard_test` was retired the day K2 retired TF32, its only
baseline). Every routine gate runs the deterministic tests of what makes the update fast: inventory == declaration, 0
compiles after the lock, the no-silent-eager FATALs, **every micro-batch of a real update on R1's compiled route (route
counts)**, **a pinned host-scalar-read count per update**, **attention on the fused SDPA kernel, not MATH**, the micro-batch
never re-copied whole per micro-step, no CUDA stream or optimizer built after startup (`designs/ops/testing.md`
"Performance-shape tests"; `src/agents/model/update_performance_shape_test.py`). At milestones (a torch upgrade, before a
baseline run, on suspicion) run the **update benchmark** under `gpu_lock` (about 6.5 min):
`python -m main.compile_inventory run --stage time --device cuda --keep-prewarm --unbracketed`. Its 2026-10-02 read (fp32, the
pinned 98,304 rows) was 40.17 s per update, n = 5, compiled share 0.892: a read, not a baseline.

**R1's per-parameter gradient bar is chosen by WEIGHT REGIME** (`gen3_r1_param_bar_by_regime_v1`; `compile_regions.R1_PARAM_BAR`,
`weights_regime`). The extractor gate's fresh bar (1e-3) failed every healthy R1 start, because fp32 CUDA EAGER is itself the
inaccurate side against a float64 reference on some parameters (matched-noise control:
`designs/research_state/measurements/k6_k8/r1_noise/`).

- `weights_regime` reads "fresh" when the legal log-probs on the fixture are constant within every row (the zero-init pointer
  head); every resume, fork, restart and canary reads "trained". Each bar is `R1_BAR_K` = 4x the largest healthy gate reading
  in its regime (`R1_HEALTHY_MAX`): fresh 0.100, trained 9.9e-3. A 10 % backward error on a path reads about 0.1, which the
  trained bar refuses and the fresh bar would not, so a fresh launch ALSO runs R1 on the declared ladder's first seeded
  perturbation, judged at the trained bar. The canary uses the same selection.
- **An UNMOVED parameter is judged by its OWN regime** (`gen3_r1_unmoved_param_v1` + `gen3_r1_unmoved_init_v1`;
  `compile_regions.unmoved_parameters`, `r1_rungs`): one whose every element is bit-identical to its value at the run's fresh
  build, or exactly 0.0 (categorical tests, no tolerance). It is judged at the FRESH bar on the live weights; R1 then runs
  again with ONLY the unmoved parameters moved off their init (seeded noise, restored bit-exactly) and EVERY parameter is
  judged at the TRAINED bar there. Why: under `--oracle-reveal` the species belief labels are all PAD, so the zero-init
  `belief_head.species_head` gets no gradient while the golden rows still supervise it, and CPU eager fp32 errs 1.5e-2 on it
  against float64 (compiled 4.4e-7): fp32 noise that FATAL'd a run at update 10. TB `compile/canary_unmoved_params` counts them.
  - **The INIT RECORD**: `compile_regions.record_param_init` (a `@startup_builder` in the trainer's fresh construction,
    `model_build.construct_fresh_learner`) stores `{parameter name: sha256(dtype, shape, bytes)}` as the model attribute
    `param_init_sha256`, plain JSON that SB3 writes into every checkpoint (~46 KB). A restart, resume or fork reads the
    lineage's own record from its checkpoint and nothing recomputes it after a load; a checkpoint with NO record (the K9
    golden, `main.fresh_checkpoint`) gets the zero rule alone. **Limits**: the rule is per PARAMETER, so a parameter only
    PART of which never moves is judged at the model's regime, and a record-less resume does not see a dead non-zero-init
    parameter. Neither has tripped.
- **Every comparison PROVES its two arms independent** (`gen3_gate_independent_arms_v1`; `compile_regions._r1_pair`, every
  rung): `region_calls`' counters must show the compiled arm dispatching the compiled route ONCE while running R1's Python
  body ZERO times, and the eager arm the reverse; anything else is a typed refusal (a gate whose two sides share their
  arithmetic would read 0 everywhere and PASS).
- **A non-finite gradient is NAMED, never read as a cosine** (`gen3_gate_nonfinite_named_v1`): `train_verdict` refuses a NaN /
  inf gradient on either arm FIRST (`NonFiniteGateArmError`, naming the parameters), and `_cos` returns NaN, not 0.0.
- **A max over candidates on a gradient path uses `index_max.max_by_index`** (`gen3_fm_index_max_v1`; the class is CLOSED on
  the gradient path): `amax`'s backward is `grad * (x == amax) / sum(x == amax)`, and when the partitioner recomputes `x` in
  the backward kernel Triton's FMA contraction can differ from the forward's, so on some rows NO element equals the saved max
  (tie count 0, gradient 0/0 = NaN: a 41-parameter NaN compiled backward under fixed_mass, eager finite). `max_by_index`
  selects BY INDEX (value bit-identical, backward a scatter at the saved index; the eager tie rule gives the whole gradient to
  the first maximal element). `amax` / `amin` remain only off any gradient path (`designs/model/op_contracts.md` "The op's
  MAXIMA"). A NaN FATAL of this shape elsewhere gets the same cure, never a looser bar.
- `compile_regions_trained_cuda_test` (GPU tier) runs real TRAINED weights through the real gate: the `production` baseline,
  read by name through `agents.training.baselines`, at fp32 and under the bf16 rule, with a CPU test that loads it STRICT into
  the learner golden so a stale fixture is caught without a GPU. An fp64-REFERENCED gate is deferred to
  `designs/ops/TASK_BACKLOG.md` T16 (the model is an fp32 program with ~300 `.float()` casts, and CUDA eager alone is no
  envelope: CPU and CUDA eager differ 100x on a correct implementation).

**R1's DECLARATION (`gen3_r1_declared_levers_v1`).** R1's signature is more than the batch's shapes: dynamo specialises on its
static flags (`MicroStatic`, resolved by `TrainSetup._micro_static`) and guards the observation dict's KEYS. Both are DECLARED
at startup from the resolved config (`compile_regions.install` records `R1Declaration`) and held to that declaration every
update: `check_r1_declared(model, st)` runs at every compiled `train()` and raises a typed `CompileSentinelError`
(`FATAL_CONFIG`) naming the field or key that moved. R1 takes NO per-update tensor dict: every value it multiplies by is a
static flag or coefficient. The gate, the prewarm and the canary build R1's arguments through the same `_r1_static`.
**A new R1 lever is declared in that static, never by relaxing the lock.** The lever inventory of `r1_declared_levers_test`
is the production argv plus the fork mask (`--fork-fraction > 0`: the obs key `fork_pg_m`, present in the env's obs space at
startup). The test (routine, CPU) builds the learner per inventory row, runs the real startup lifecycle and two real updates (0
compiles after the lock), and holds an undeclared lever or key after the lock to the sentinel's UNDECLARED SIGNATURE FATAL.

History: archive, same heading (the R0 deletion's measurement table, the K8 acceptance read, the R0 mask-dtype fix, the
unmoved-init case on `rb_x5ab_oracle_sp_s1001`, the F-XC-4 gate readings and the per-kernel NaN scan).

---

## T25 perf levers on R1 (2026-10-10)

The perf phase (`designs/research_state/measurements/perf_phase_2026-10-10/PLAN.md`) adds the switches below and lever C's
flag (next section). Each is OFF by default; off, the production and end-state compiles are byte-for-byte unchanged.

- **R1's Inductor option presets** (`compile_regions.R1_INDUCTOR_PRESETS`, `gen3_r1_inductor_presets_v1`): `default` (no
  options), `coordesc` (`coordinate_descent_tuning`), `combo` (`combo_kernels`) and `cudagraphs` (`triton.cudagraphs`).
  Presets join with `+`. Set `model._r1_inductor_preset` BEFORE `compile_regions.install`, which passes them as
  `torch.compile(..., options=)`: R1 only, never T2's graphs, never a global config. They are MEASUREMENT switches with no
  CLI flag; `compile_regions_presets_test.py` pins that the default passes NO `options`, every key is a real Inductor config of
  this torch, and an unknown name refuses. A preset gets its CLI flag, flag-registry row and recipe surface in its ADOPTION
  commit, after its paired GPU benchmark and the R1 gate and canary are green. ⚠️ The `cudagraphs` PRESET alone graphs R1's
  ONE compile and dies on the first (diagnostic) update (F-PB-2); lever C's route is `--learner-cudagraphs`, and `install`
  refuses the preset combined with it.
- **The trunk's compute precision: `--trunk-precision {fp32,bf16}`** (`gen3_trunk_precision_flag_v1`, config v155;
  `agents/model/trunk_precision.py`). **ADOPTED 2026-10-14**: the bf16 strength A/B PASSED (`designs/endstate/design_bf16_strength_ab.md`;
  8 seeds x 15M per arm, delta +1.06 pp, 90 % interval [-1.19, +3.31]; `train_ms` 54.3 s -> 42.1 s, -22.5 %), so `bf16` is
  `recipe.fresh.trunk_precision`. The flag's own default is still `fp32` (an UNSET flag resolves to `fp32`); `--arch production`
  writes `bf16` and the recipe surface REFUSES a fresh argv that leaves it untyped. A flagless fork or restart INHERITS its
  parent's recorded value (an fp32 parent's fork stays fp32 unless it types `bf16`; the A/B did not measure a mid-lineage
  switch). The K9 learner golden is the production (bf16) update and pins the fp32 opt-out beside it.
  - **The switch.** `bf16` class-swaps the LEARNER's live trunk rounds (`BiasedEncoderLayer`, `IdentityInitRound`) to
    subclasses whose forward runs under bf16 autocast and returns fp32; `apply_run_precision`, from `apply_training_hparams`,
    does it on both build paths, before grad checkpointing and the R1 install. **Decisions stay fp32**: the heads, the pointer
    head, the masked log-softmax, the value, the losses and the AdamW state.
  - **Recorded.** A `ModelVersion` field, `_resolve`-inherited by a flagless resume; a pre-v155 config migrates to fp32;
    `check_compatible` never compares it; it is in the shared compile cache's `COMPILE_ARG_KEYS`.
  - **bf16 lives in R1 only**: T2, eval and every snapshot build from a state dict, so they get base classes. The K9(b)
    probe's forward and violation scan run under `trunk_precision.fp32_scope`, so K9(b) compares fp32 with fp32 at its
    unchanged bars (`trunk_precision_flag_test`: bit-identical probe log-probs on a bf16 learner).
  - **THE bf16 PARITY RULE** (`gen3_r1_bf16_parity_v1`, `compile_regions._r1_verdict_bf16`) applies when the policy's rounds
    are bf16, at the startup gate and at every canary, through the same rungs as fp32. Three arms run on the same weights and
    rows: compiled bf16, eager bf16, and the fp32 EAGER REFERENCE (`_r1_reference`: R1's eager body under `fp32_scope`,
    proved eager by the region counters). (1) **DRIFT**: eager bf16 vs the reference, loss rel and global gradient rel-L2,
    each <= `BF16_DRIFT_BARS` (loss 0.014, gradient 0.68). (2) **ENVELOPE**: compiled bf16 vs the reference <= K x
    max(eager bf16 vs the reference, a floor), for the loss, the global gradient and EVERY judged parameter
    (`BF16_ENVELOPE_K`: loss 5.4, gradient 6.4, per parameter 12); the floor is the fp32 bar of that quantity, except the
    scalar loss's, which is bf16's unit roundoff 2^-9 (`BF16_LOSS_FLOOR`). With `--learner-cudagraphs on`, the graphed
    route's accumulation rung is judged by the same rule. The bars are `R1_BAR_K` x the healthy GPU maxima of window D's 20
    reads, derived into
    `designs/research_state/measurements/perf_phase_2026-10-10/results/bf16_parity_bars.json` (`compile_regions_bf16_test`
    re-derives them and fails on a differing constant); a `None` table refuses a bf16 R1 ("no measured bars").
  - ⚠️ **The bf16 gate's resolution is a KNOWN LIMIT** (`gen3_r1_bf16_resolution_v1`): a parameter's backward miscompile is
    refused only above K = 12 x max(that parameter's own bf16 error, the fp32 bar), and bf16 moves the micro-batch gradient
    1-18 % rel-L2 from fp32's. On window D's GPU reads (B = 2,048) the MEDIAN parameter is refused above 0.22-1.29 relative
    error and the worst above 1.27-10.2 (on FRESH weights the live rung's floor makes it 1.20, so the seeded-perturbation rung
    judges). So a 10 % backward error is NOT refused (a planted 10 % on `pokemon_encoder.move_network.0.weight` passed on
    CUDA); a DROPPED (error 1.0) or sign-flipped (2.0) gradient is refused on MOST parameters, not all; a NaN is always named.
    The resolution is COMPUTED exactly from the gate's arms and confirmed by plants at 0.5 m* (passes) and 2 m* (refused):
    `compile_regions_resolution_test.py` (`test_a_10pc_…` pins the limit, so a sharper gate fails it and must update this
    bullet); the CUDA numbers come from `scripts/ops/verify_compile_gate_cuda.sh` (`summary.json`, "resolution"; PASS 4/4 at
    `87e493c3`). The fp32 gate keeps its teeth: a 10 % plant is FATAL on an fp32 learner (`compile_regions_independence_test.py`
    on CPU, `compile_regions_fixed_mass_cuda_test.py` on CUDA).
  - **Decision (2026-10-15): keep the fp32 eager reference; do NOT add a same-precision (compiled bf16 vs EAGER bf16) rule.**
    Under autocast Inductor's fused kernels keep fp32 intermediates where eager rounds every op to bf16, so the two bf16 arms
    also differ by bf16 rounding and the derived K is 11.8, the envelope's 12: the same teeth for a second bar
    (`test_a_SAME_PRECISION_rule_would_not_be_sharper_on_the_committed_GPU_reads`). Not taken: Inductor's
    `emulate_precision_casts` (changes the production graph) and a SHADOW fp32 compile of R1 (a second compile per launch that
    judges a graph that does not train).
  - 🚨 **A false-FATAL mode**: the rule's per-parameter statistic crosses the PPO CLIP BOUNDARY of any fixture row whose ratio
    sits within rounding of it (the next sections: "The bf16 canary FATAL on trained weights" and its T36 fix).
  - **Pinned by** `trunk_precision_test.py`, `trunk_precision_flag_test.py`, `compile_regions_bf16_test.py` (refused without bars,
    a healthy pass, a planted compiled error or drift, a mis-wired reference, an fp32 learner untouched),
    `compile_regions_resolution_test.py` and `compile_regions_clip_margin_test.py` + `compile_regions_clip_margin_fatal_test.py`.

History: archive, same heading (the adoption A/B, window D's resolution reads and the same-precision derivation table, the
bar-derivation and update-benchmark measurement scripts).

### 🚨 The bf16 canary FATAL on trained weights (`rb_toy_ent_s4001`, 2026-10-15) — diagnosed on CPU; the FIXTURE FIXED (T36), CONFIRMED on CUDA at `87e493c3`

**The finding: the canary's per-parameter statistic contains a DECISION BOUNDARY, the PPO clip.** The canary judges R1 on the
K9 golden's 64 committed rows (tiled to B = 2,048), whose old log-probs come from a near-uniform policy, so on trained weights
their PPO ratio spreads over 0.002 ... 6.9. A row whose ratio sits within rounding of the clip bound that SWITCHES ITS gradient
(A > 0 at 1 + 0.15, A < 0 at 1 - 0.15) carries its whole policy gradient in one arm and none in the other. The toy control
`rb_toy_ent_s4001` (bf16 trunk, `--compile-trainer`) ended FATAL at 72.66M steps on three pointer-head parameters this way
(compiled 4.03e-01 vs eager 2.82e-02 from the fp32 reference; the six canaries before it passed).

- It is a per-STATE lottery, not "trained weights" as such: eager bf16's error on the pointer head is BIMODAL (median 0.020
  where no row's clip branch differs from fp32's, 0.294 where one does), and two HEALTHY eager arms read a canary ratio above K
  = 12 in 18-21 % of dithered pairs. The pointer head is the exposed one (only the policy and entropy terms reach it and a few
  rows carry its gradient).
- **It reaches the production surface**: a row sat within 0.01 of its switching boundary at 2 of 10 end-state states, and a
  flip's SIZE scales with the row's |A| x ratio. The startup gate is SINGLE-SHOT (no confirmation step), so a boundary row at a
  6-hourly restart ends the run there. The old confirmation could not catch it: slice 1 is a SUBSET of slice 0.
- **Diagnosis**: (iv) the FIXTURE was CONFIRMED by the GPU read; "the yardstick is unusually small" is the same bimodality; bf16
  numerics in the compiled graph is unlikely by construction (the heads, the masked log-softmax and the losses run outside the
  bf16 region); a real Inductor miscompile would be deterministic and read in the no-flip mode too. Rejected: fp32 for the
  pointer head (MOOT, already fp32), `emulate_precision_casts` (eager flips too), recalibrating the bar (a bimodal statistic has
  no bar with teeth), an fp32 trunk (a stopgap, `train_ms` +29 %), an eager learner (removes the canary's subject, not the defect).
- **The diagnostic**: `scripts/ops/diag_bf16_canary.sh <run>`, run by the lease holder (`--plan` prints the command; module
  `src/main/ops/diag_bf16_canary.py`; CPU reads: `python -m main.ops.diag_bf16_canary cpu --run <run> --ckpts … --out D`). It
  reads the run's own checkpoints on the canary's own rows through the arms E32 / Ebf / Cbf / C32 / Cbf_emul / Cbf_aot /
  Cbf_pheager, tells a backward that disagrees with its own forward from a forward whose logits moved, and writes
  `gpu_<label>.json` per checkpoint plus `VERDICT.md` under `~/.cache/gen3ai/diag_bf16_canary/<sha>/`.

**THE FIX (T36, `gen3_r1_clip_margin_v1`).** Standing rule 8 (exclude inputs within a rounding error of a decision boundary
rather than tolerate them by chance) applied to the gate's own fixture, in `compile_regions` / `compile_canary`, for the STARTUP
GATE and the CANARY alike (they share `r1_rungs` -> `_r1_judge`), under the bf16 rule and the fp32 rule alike:

1. **Relocate the boundary rows, on every rung's own weights** (`relocate_clip_boundary` -> `clip_relocation`). One fp32 eager
   forward of R1 (`fp32_scope`) reads every row's log-prob and normalised advantage; a row whose reference log-ratio is within
   `R1_CLIP_MARGIN` of (or AT) the bound that switches ITS gradient (A = 0 rows carry no policy gradient and stay) gets its OLD
   log-prob moved so the reference sits 2 x the margin inside the ACTIVE side. **Moved, not excluded**: the batch keeps its
   declared shape (no new signature) and every row, and excluding a row would change the advantage normalisation and drop its
   policy gradient from the check. Deterministic: a function of the reference's own forward on the same rows.
2. **The margin's precondition is CHECKED** (`_clip_margin_rule`, every rung): every judged arm's largest |Δ log π| from the
   reference is <= the margin and no arm switches any row's clip branch. A COMPILED arm that breaks it is refused BY NAME as a
   forward this far from the reference, never read through a gradient ratio; a YARDSTICK arm (eager, the fp32 reference) that
   breaks it is a recorded event (below). Every R1 rule line prints `clip margin: <n>/<B> boundary rows relocated (M …; nearest
   unmoved …); max |dlogp| compiled …, eager …, fp32 reference … <= M, 0 clip-branch switches`.
   **The margin**: `R1_CLIP_MARGIN = R1_BAR_K x R1_LOGPROB_SHIFT_MAX = 4 x 0.18 = 0.72` (log-prob). 0.18 is the largest
   |Δ log π| of a judged arm measured so far, rounded UP (eager bf16 vs eager fp32 reads 0.161 over the toy's 62 checkpoints and
   0.055 over 10 production states; the compiled bf16 arm on CUDA peaks at 0.110). On a LIVE run the constant is raised with
   `--canary-shift-max` (next section).
3. **The canary confirms on two DISJOINT halves** (`compile_trainer.FIXTURE_HALVES`: rows 0-31 and rows 32-63), so no single row
   can confirm a disagreement. The scheduled check and the startup gate keep ALL 64 rows (a half can leave a parameter below the
   per-parameter floor); a miscompile only one half judges is FATAL at the next scheduled canary (two consecutive disagreements).

**Proof.** On the raw rows 212 of 992 healthy dithered eager pairs read the canary ratio above 12 (max 41.5x); on the RELOCATED
rows 0 of 992 (max 3.65x), and a planted 10 % backward error on `pointer_head.move_proj.weight` is still refused at the fp32 rule
(`compile_regions_clip_margin_fatal_test`, on the FATAL's rows committed as
`src/agents/model/testdata/k9_golden_buffer_rb_toy_ent_s4001_pin.npz`; `compile_regions_clip_margin_test`). **CUDA confirmation
(2026-10-15, `87e493c3`):** `verify_compile_gate_cuda.sh` PASS 4/4; the diagnostic's 8 checkpoints met every criterion (the
canary's replay passes at the FATAL's weights, relocated `branch_flips` 0 for every arm, the relocated null P(ratio > K) = 0,
`fp32_rule.over_bar` empty, C32 within 4e-6 of E32, Cbf's shift peaking at 0.110 <= 0.18); the UNRELOCATED canary would have
refused at 15M too. The compiled shift GROWS with training (0.050 at 15M, 0.110 at the FATAL) and eager bf16 grows alike, so
the growth is the weights', not the compiler's (`designs/research_state/measurements/x26_exit_check_2026-10-15/canary_shift_87e493c3.json`).
**Watch it in-run**: `canary_verdicts.jsonl` `.watch.scheduled.clip_margin.max_abs_dlogp.compiled` (`gen3_canary_watch_v1`), with
bands and actions in `designs/endstate/design_x26_readiness.md` §4.2. If Cbf is ever off the reference with NO branch flip, or
C32 is off at the fp32 bar, the fixture was not the whole story: take the fp32 trunk and hunt the miscompile with the diagnostic.

History: archive, same heading (the event and its numbers, the checkpoint census and growth curve, the ranked-hypotheses table
with each read that kills it, the GPU-window plan, the decision-options cost table).

### The check-only margin override (`--canary-shift-max`, `gen3_canary_shift_max_v1`, config v158, T37, 2026-10-15)

**Why.** The bf16 HEADROOM WATCH (`design_x26_readiness.md` §4.2) raises `R1_LOGPROB_SHIFT_MAX` (0.18) when the compiled bf16
arm's taken-action shift passes it. A code constant reaches a live run only through its pin, and **the launcher REFUSES to move
a live run's pin at a restart** (`worktree.resolve_pin`), while `--sync-to-main` would walk the run onto all of HEAD. So the
delivery is a flag a same-run RESUME may change.

**What it is.** `--canary-shift-max V` (argparse default None; `type` = `compile_regions.resolve_shift_max`):

- **Check-only.** It is the margin's INPUT and nothing else: `compile_regions.clip_margin_of(model)` = `R1_BAR_K` x the shift,
  read by `relocate_clip_boundary` and `_clip_margin_rule`. No loss, optimizer, seed or parity bar reads it. **Proven**:
  `main/train/canary_shift_max_test.py::test_one_update_is_BIT_IDENTICAL_with_the_flag_at_two_values` runs the K9 golden's
  update with the flag at None, 0.72 and 25.0 and compares every parameter's bytes and pinned loss scalar.
- **Raise-only.** A value below `R1_LOGPROB_SHIFT_MAX` (0.18), or a non-number / NaN / inf, is REFUSED, typed
  (`CanaryShiftMaxError`; an argparse error at parse, a `parser.error` in `resolve_config` for an INHERITED value, and the
  gate itself refuses a model carrying one): 0.18 is the largest healthy shift measured, and a lower margin has no basis.
- **Ceiling-bounded (T52, 2026-10-20).** A value ABOVE `compile_regions.SHIFT_MAX_CEILING` (0.5, the orchestrator-approval cap
  of `TRAINING_RUN_SOP.md` §3) is refused the same way, because a typo of 18 for 0.18 (a margin of 72 log-prob) silently
  disarms every canary refusal. 0.3 (the ruled X34 / toy resume value) and 0.5 itself parse. A typed value is refused at parse
  (exit 2); an INHERITED one (the ceiling fell after the run recorded it) exits `FATAL_CONFIG` (3) from `resolve_config`, so
  the launcher stops instead of crash-looping. Raising the ceiling is a code change with a measured basis, not a flag; no
  acknowledgement flag exists.
- **A typed value needs the gate it moves.** `combination_checks.canary_shift_max_needs_the_compiled_cuda_learner` refuses a
  TYPED value on `--no-compile-trainer` / `--device cpu` / `--debug` (silently inert otherwise); an INHERITED value is carried,
  never refused (a CPU smoke of an X26 fork still launches).
- **Recorded and inherited like `--trunk-precision`** (config v158, `ModelVersion.canary_shift_max`, Optional[float], default
  None = the code's constant): a TYPED value wins and is what the next save records; an untyped restart, flagless resume or FORK
  inherits the recorded one. Never compared by `check_compatible`; the model attribute `_canary_shift_max` is never pickled.
- **Printed.** The launch prints `🐤 [CANARY SHIFT MAX] V …`; every R1 rule line names the effective margin and its source
  (`[--canary-shift-max]` or `[default]`); every canary record carries `canary_shift_max` {value, source, margin}.

**The headroom watch's RED action uses it**: when the compiled bf16 shift passes 0.18 but stays under the margin, **stop the
run at its next planned restart and resume the SAME run** (`--model <its latest checkpoint>`, the SAME pin verified with
`--dry-run`, never `--sync-to-main`) with `--canary-shift-max <v>`, v the largest `compiled` value read rounded UP to 2
significant digits with >= 1.25x headroom. The flag must exist in the run's pinned tree (any pin at or after the T37 commit;
a pinned parser that lacks it refuses the argv: `pinned_argv.pinned_parser_check`, `✗ NOT IN PINNED TREE`). The raise survives
the launcher's later restarts through the recorded field. It does not replace `scripts/ops/diag_bf16_canary.sh <run>`, which
says whether the shift is shared by eager bf16 and C32 is still ~0.

### The yardstick is not the compiled arm (`gen3_r1_yardstick_margin_v1`, 2026-10-19)

`_clip_margin_rule` used to refuse ANY arm past the margin and the canary logged it as "compiled DISAGREES with eager". At X34
update 1700 (`../research_state/measurements/x34_canary_u1700_2026-10-19/README.md`) the arm that broke M = 0.72 was EAGER
bf16 at 0.856, the canary's own yardstick, while compiled read 0.52 with 0 clip switches: a bf16 CONDITIONING tail, not a
compile bug (compiled vs eager at the SAME precision reads 0.017-0.028 on CPU Inductor, 4-6x tighter than either arm's
distance from fp32).

**The rule** (`compile_regions.YARDSTICK_ARMS` = eager, fp32 reference; every other arm, "compiled" or "graphed", is the
judged one):

1. **A yardstick out of margin is an EVENT, not a refusal.** It is recorded (`watch.*.yardstick_out_of_margin`: the arms past
   M with shift, batch row and fixture row; M; M'; each arm's first-read shift; a `📏 [CompileCanary]` log line) and never
   counts toward confirm / FATAL (`compile/canary_yardstick_events`).
2. **The rung re-runs ONCE at M' = max(M, `YARDSTICK_HEADROOM` 1.25 x the largest YARDSTICK shift)** and is judged there (a
   new relocation, new arms, the full verdict; the first read's verdict is discarded). **The compiled arm's shift never sets
   M'**, so a miscompile cannot widen its own bar. The startup gate's rungs (`r1_rungs`, the graphed accumulation rung) take
   the same path.
3. **Deterministic** (rule 8): the arms' forwards do not read the old log-probs, so the re-run reads the same shifts (CUDA arms
   are run-to-run bit-identical, window D). A yardstick STILL past M' did not reproduce its own forward: `YardstickRefusal`,
   which refuses and counts on the ladder.
4. **The log names the failing arm**: "compiled DISAGREES with eager" only when the compiled arm failed; a `YardstickRefusal`
   (and the bf16 DRIFT rule, eager bf16 vs fp32) is worded "the canary's YARDSTICK refused (eager / the fp32 reference, NOT the
   compiled arm)".
5. **Every record names each arm's worst row** (`clip_margin.worst_row`, `worst_fixture_row`) and carries `same_precision`
   (max |log pi_compiled - log pi_eager| and its row), RECORDED with no bar until a CUDA census reads it. An UNCONFIRMED or
   FATAL verdict writes the policy `state_dict` beside its record (`canary_u<update>_s<steps>_<verdict>_policy.pt`, never over
   an existing file: `gen3_canary_dump_no_clobber_v1`), at most `compile_canary.DUMP_WEIGHTS_MAX` = 3 per process.

**UNVERIFIED:** the 1.25 headroom and the same-precision gap's healthy range have no CUDA census; the gradient rule's ratio on
the re-relocated u1700 rows (the weights are gone). A pin before this change keeps the old rule; its only remedy for a
yardstick tail is the `--canary-shift-max` raise on a resume (`TRAINING_RUN_SOP.md` §3 rule 5).

### A PASS near its bar keeps its weights (`gen3_canary_near_margin_dump_v1`, 2026-10-25)

The X26 continuation's u1000 canary PASSED with the compiled arm 0.712 from the fp32 reference against M 0.72, but a PASS
dumped nothing, so the 478.31M weights were lost and the CUDA read cannot be replayed
(`../research_state/measurements/x26c_canary_margin_2026-10-25/` F1). Now any judged quantity at >= 0.9 of its bar
(`NEAR_MARGIN_FRACTION`) writes the same `state_dict` an UNCONFIRMED does, rotated to the newest 3 per run
(`NEAR_MARGIN_KEEP`, independent of `DUMP_WEIGHTS_MAX`); every record carries `bar_fractions`. The record and the rotation are
documented in `learner_lifecycle.md` (the canary). A run picks it up only at its next PIN.

**The per-row dither-null margin is DESIGNED; its null runs as a RECORDED-ONLY SHADOW**
(`../research_state/measurements/canary_row_null_design_2026-10-25/README.md`; `gen3_canary_row_null_shadow_v1`,
`canary_row_null`). The margin would replace the global M with a per-row bar of k x that row's own fp32 half-ulp dither null.
On CPU healthy bf16 arms read at most 1.1x their row's null (704 row-reads), so k ~ 4.4; the one CUDA compiled point (u1000)
reads 2.7x a CPU null, so k is unsettled. Every canary records each row's 16-seed null and each arm's per-row shift / null as
`row_null` and judges nothing (`learner_lifecycle.md`), so each production canary from its next pin on is one CUDA census row;
the global M stays binding. The lease-gated census `gpu_census.sh` is written there, not run.

**Replaying a canary on the GPU (`python -m main.ops.canary_dump_replay`).** `--dump <x>_policy.pt --run-dir <run>` runs a
dumped `state_dict` through the canary's own `_check` and through the arms E32 / Ebf / Cbf, using the canary's rows (the K9
golden buffer, tiled to the run's B). Its validity is the dump's own record. `--anchor <zip> --run-dir <run>` (the X-CHECK)
puts a different run's checkpoint through the same rows, B, compile (the default R1 route) and rule. That separates a
weight-specific event from a graph-wide one (X48's kffn u200 event against the production graph: C0 / X26c). Its
validity is the anchor's own eager-bf16 numbers, which must be self-consistent across repeats and between the canary and
direct arms, because no record exists for that state. Its module docstring holds the pre-declared rule (REPRODUCES / DOES
NOT / MIXED / INVALID-REPLAY).

**The X-CHECK's finding (2026-10-28, 5 repeats each, CUDA, all VALID; `~/.claude/jobs/x35/w0_2026-10-28/xcheck/`).** The
edge_bias event belongs to the **X48 s3502 u200 STATE, not to a graph class**:
- the dump REPRODUCES (c2_map 26.6 x eager);
- X48 s3502's own final checkpoint (kffn graph, about 4 updates later) DOES NOT (worst edge_bias 8.0);
- neither does X26c @480M (2.4) on the production graph;
- C0 s3501 final reads edge_bias-clean (worst 8.67 < K 12), but is MIXED: slice 1 refuses the bf16 LOSS rule in 5 of 5
  repeats (compiled 1.09e-2 > 5.4 x max(eager, floor) = 1.05e-2, about 4 % over), which a scheduled canary (slice 0)
  would not see.

That marginal loss-rule refusal is a TECH_DEBT row (`../ops/TECH_DEBT_BACKLOG.md`, "C0 s3501 final refuses the bf16 loss
rule"), to be weighed in the canary-rule recalibration from W0's census.

### The canary-FATAL resume rule and its family bars (`python -m main.ops.canary_decide <run>`, 2026-10-26)

A FATAL canary on an X35 arm resumes (`--canary-shift-max 0.3`, the orchestrator's operator step) iff 0 PPO-clip
branches switched AND every judged arm's compiled / eager `max_abs_dlogp` is at or below its FAMILY's bar; anything else
HOLDs (exit 0 / 3). The ratio is judged only at a FATAL: a PASS above the bar is descriptive. The bars are fixed, not drifting:

| Family | Bar | Derivation | Ruled |
|---|---|---|---|
| `c0` | 2.64 | 1.25 x 2.108, the healthy-PASS maximum over three seeds (toy C0 s3603 u500); first 1.30, re-ruled up | 2026-10-21 01:55 |
| `x47` | 2.40 | 1.25 x 1.893, the healthy-PASS maximum (toy X47 s3601 u700) | 2026-10-20 11:00 |

Source: Training Oct's X35 job log (`~/.claude/jobs/x35/FINDINGS.md`, outside the repo) and the orchestrator's rulings
there. The run-name map is `canary_decide.FAMILY_NAME_PATTERNS`: every production arm `rb_x35_{c0,a1*,sprt,lrfloor*,
d1wide,x52*,d3mut,n1p,x48kffn}_s*` is a fork of P* (rb_x26_s2601 @280M) at production pins, the same family as C0, so it
takes the C0 bar (X48, per-token-kind FFN experts grown at the fork, by its PREREG's ruling:
`designs/research_state/measurements/x48_kffn_production_2026-10-27/PREREG.md` §7); the toy `rb_x35t_c0_*` takes C0 and `rb_x35t_x47*_*` takes X47. **UNVERIFIED:** the 2.64 was measured on the TOY
C0, not on a production arm; the production reads so far sit under it (C0 s3504 u10 2.38) except one PASS above it
(rb_x26c_s2601 u1000, 3.18, which is not an X35 arm). Any other name (`rb_x35_r0_*`) HOLDs until `--family` or a typed
`--bar` (logged with `--log`, this run only).

### bf16 drift on a sharpening policy: the loss is read on the golden's own rows (`gen3_r1_loss_on_golden_rows_v1`, 2026-10-17)

**The rule.** The bf16 rule's LOSS comparisons (the DRIFT and the ENVELOPE) are read on the golden's OWN rows, not the
relocated ones: `compile_regions.golden_row_loss` re-evaluates each arm's policy term on the golden's raw old log-probs
(`ClipRelocation.raw_old_log_prob`), as L - pg(relocated old) + pg(raw old), from the arm's own log-probs and the normalised
advantages; `micro_step`'s formula in float64 with the fork mask and the accumulation rung's per-micro sum. `_r1_judge` and the
graphed accumulation rung hand the bf16 rule those losses (`_r1_verdict_bf16(..., losses=)`). The GRADIENT comparisons stay on
the relocated rows (rule 8 needs them there). The bars, `BF16_LOSS_FLOOR` and the fp32 rule are unchanged and training is
untouched. The R1 line prints `[loss on the golden's own rows; relocated rows: drift …, envelope …]` and every canary
record's `global` carries `drift_loss_relocated` / `envelope_loss_relocated` beside the judged values.

**Why.** The loss is CONTINUOUS in the ratio (the clip switches the gradient, not the value), so it needs no relocation, but
T36's move multiplies a relocated A < 0 row's loss weight by ~4.2 (ratio 0.85 -> 3.59, and by e^(2 dM) more under a raised
`--canary-shift-max`), while `BF16_DRIFT_BARS` were derived on the golden's own rows BEFORE the relocation existed. On a
sharpening policy ONE heavy row (X26's golden row 9, |A| = 3.23) then carried 88-100 % of the drift: its fp32 log-prob moves
0.04-0.11 under a half-ulp weight dither, so bf16's shift on it is the fp32 function's own conditioning, not a near-tie. X26's
startup gate read 6.68e-3 (48 % of 0.014) at one restart; a per-state lottery with ~1-2 % per-restart odds above the bar
(UNVERIFIED on CUDA). Rejected: normalising the policy part by its sensitivity (a new unmeasured bar), a raise-only override of
the loss bars (loosens the teeth with no basis), fp32 for one trunk round (a training change).

**Proof** (`compile_regions_loss_rows_test`, each test fails on revert): the reconstruction EQUALS a forward on the golden's
rows (K9 learner, bf16 and fp32 trunk, within 1e-5 relative, one micro-batch and the accumulation rung's two); a relocation-
amplified healthy one-row shift passes on the golden's rows and is refused on the relocated ones; THE TEETH still refuse a
broad 0.2 log-prob shift (DRIFT), a non-policy loss fault in the eager arm (DRIFT) and a compiled-only loss fault (ENVELOPE).
**Not covered by a revert test:** the graphed accumulation rung's wiring (CUDA with `--learner-cudagraphs on` only; its
reconstruction function is the tested one).

**A pin BEFORE this change** (X26's `9c033489`) keeps the relocated statistic and cannot take new code. If its startup gate
refuses ONLY on the bf16 loss (the `DRIFTED … loss` or `COMPILED … loss:` lines, never a gradient, per-parameter or margin
refusal), resume the SAME run at the SAME pin from the newest periodic checkpoint whose CPU pre-flight reads at most half the
bar (`designs/research_state/measurements/x26_bf16_drift_2026-10-17/scripts/newrule.py <ckpts> 64`; its "relocated-row drift"
column is what the pinned gate judges; ~10 s per checkpoint on the agent cores; adjacent checkpoints differ by up to 10x, so
this costs <= 1-2M steps). Weigh a staged T37 raise against its cost: it moves the per-restart refusal odds from ~1-2 % to ~4 %.

History: archive, same heading (the CPU finding with the 2M-123M drift series, the forecast, the ranked-options table, the
X26 reads).

### The gate releases its memory (`gen3_gate_releases_memory_v1`, `gen3_gate_memory_v1`, `gen3_canary_memory_v1`, 2026-10-15)

**The rule.** The gate and the canary must not leave their autograd graphs or cache behind. T36's relocation forward (one
fp32 grad-mode forward of R1's eager body at B = 2,048) STASHED its outputs on the policy and through them its whole autograd
graph (4,486 MiB of saved fp32 activations), which stayed alive under the COMPILED arm until the arm's forward replaced the
stashes; the allocator then kept the 8.2 GiB peak CACHED and the update-fit check read that cache as the update's demand
(`rb_xs_s2701`: demand 9,790 MiB, headroom 978 < 1,024, `UpdateWontFit`, with the dry update itself at 6,579 MiB). Three parts,
none changes a number:

- **The relocation keeps nothing for a backward**: it runs under `saved_tensors_hooks` that drop every saved tensor
  (`consistency._drop_saved`; hooks change what autograd keeps, never a value; `compile_regions_memory_test` pins the
  reference log-probs bit-identical and that the forward saves 0 bytes). `release_gate_memory` then detaches the module-level
  stashes, and `release_autograd_stashes` walks INTO the typed stash containers (`ExtractorStashes`, `OpStashes` and the dicts
  / tuples / dataclasses they hold: `gen3_release_walks_stashes_v1`; `consistency_test` pins the walk), because it used to
  reach only tensors held DIRECTLY on a module.
- **The gate hands its cache back**: `arm_compile_sentinel` calls `release_gate_memory` + `empty_cache()` after the LOCK.
- **The update-fit check measures what training will need** (`gen3_update_fit_v2`, `learner_lifecycle.md` "The update fit
  check"): from an emptied cache, the dry update and then the canary's check once (`CompileCanary.measure`); demand = the
  peak reserved over both.

**The record.** Every gate prints a `🧮 [CompileRegions] gate memory` line (MiB allocated after / high-water per step); every
canary record carries `memory` {start_alloc_mib, transient_mib, peak_alloc_mib, reserved_mib}; TB gets
`compile/canary_peak_alloc_mib` and `compile/canary_reserved_mib` on a canary update. A smoke moves the FIRST canary with
`GEN3AI_CANARY_FIRST=<n>` (a test hook; unset in production). With the fix, the X26 argv's gate peaked at 5,480 MiB allocated
(was 8,244) and the update-fit headroom was 2,444 MiB (was 978); the K9 learner golden is identical. Full table:
`learner_lifecycle.md`; `designs/endstate/design_x26_readiness.md` I31.

History: archive, same heading (the per-step attribution table and the verification run).

### Lever C: R1's GRAPHED route (`--learner-cudagraphs {off,on}`, `gen3_r1_graphed_route_v1`, 2026-10-10)

⚠️ **ON FAILS ON THE GPU. Do not launch with `--learner-cudagraphs on`** (window E, 2026-10-10, F-PE-1): the accumulation
rung dies at startup with "accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run" (a bare
`RuntimeError`, not a typed `CompileTrainerError`), because AccumulateGrad took the first replay's pool output as `.grad` and
the second replay overwrote it. Until a follow-up materialises `.grad` before each accumulation group (or clones the graphed
grads out of the pool) and types the error, the flag stays a MEASUREMENT lever, default off.

**Why a second route.** R1 under Inductor's cudagraph trees passes the startup gate bit-equal, then dies on the first update in
the grad-balance probe's `autograd.grad(retain_graph=True)` (F-PB-2): cudagraph trees record ONE backward per forward. Two
consumers re-enter R1's graph with `retain_graph`: the per-term noise sampler (`noise_scale_terms.PerTermNoiseSampler.flush_micro`)
and the grad-balance probe (`grad_balance._flat_grads`), both on diagnostic updates only.

**The route** (`compile_regions.r1_route_for` <- `diagnostics_cadence.micro_retains_graph`, decided BEFORE each R1 call,
deterministic, no try/except): a micro-step whose per-term sampler is collecting, or whose grad-balance probe is still
PENDING, takes the un-graphed COMPILED route; every other micro-step takes the GRAPHED one. At the production shape (accumulation
32, 48 micro-batches x 10 epochs) a diagnostic update runs ~32 of its 480 micro-steps compiled; a plain update (9 in 10) runs
all 480 graphed.

**What `on` builds.** `compile_regions.install` compiles R1 a SECOND time (`model._compiled_micro_step_graphed`): the same
function, kwargs and preset options plus `triton.cudagraphs`, a second DECLARED signature in `REGIONS`
(`Signature(route="graphed", gate="learner_cudagraphs")`), one more dynamo entry on R1's code object. Both routes are **gated**
(`gate_regions` runs `r1_rungs` on each, plus the graphed route's ACCUMULATION rung `_r1_accum_rule`: two micro-batches, ONE
`.grad` accumulating across two replays, against eager), **prewarmed** (`prewarm_calls`, one per route), **inventoried**
(`assert_inventory`: 2 entries), **locked** by the same stance and **canaried** (the graphed one first; its cosine is
`compile/canary_grad_cosine`). **No silent un-graphed route**: Inductor's `cudagraph_skips` counter moving after the install is
FATAL at `assert_inventory` and at every sentinel `check` (`compile_control.cudagraph_skips`, `CompileControl.graphed_check`);
`lifecycle/graphed_region_calls` counts the graphed calls per update.

**Output lifetime.** Cudagraph trees overwrite a replay's outputs at the next replay, so nothing may read R1's outputs or stashes
across micro-steps (readers copy or run inside the same micro-step); the remaining risk is the AccumulateGrad steal above.

**OFF is byte-identical** (one `torch.compile` with the production kwargs, one cache entry, the same rules; pinned by
`r1_graphed_route_test.py`). The flag is a RUNTIME perf knob like `--compile-trainer`: not versioned, not in
`model_config.json`, never pickled (`hparams._excluded_save_params`), not fresh-only (the launcher's restarts forward it),
keyed in the shared compile cache (`COMPILE_ARG_KEYS`). Refused (`FATAL_CONFIG`) on cpu, under `--debug`, with
`--no-compile-trainer` (`combination_checks`) and when an AUTO `--compile-trainer` resolved off (`lifecycle._maybe_compile_trainer`).

### 🚨 The CPU Inductor tile2d codegen bug (`gen3_cpu_inductor_scalar_v1`, 2026-10-15) — torch 2.8.0; every CPU compile carries `cpp.simdlen = 1` (`inductor_compile.compile_on`)

**Read this before you compile anything on CPU.** Production compiles on CUDA only (Triton); `--compile-trainer` REFUSES a CPU
device. A CPU compile of R1 exists as a TEST SEAM (`compiled_first_update_test`'s CPU cell, the `compile_regions_*` CPU cells,
`diag_bf16_canary --device cpu`).

- **Symptom.** `torch._inductor.exc.InductorError: CppCompileError`, g++: `error: 'tmp8' was not declared in this scope` on a
  `transpose_mxn<float,16,16,false>(…)` line. The typed wrapper (`compile_regions._fatal_compile_errors`) used to report it as
  "a DECLARED REGION does not compile as one graph"; it now says the BACKEND's code generation failed and points here.
- **Cause.** Inductor's CPU backend picks a 2-D TILED kernel (`TilingSelect._select_tiling_indices`) when a fusion has a
  gather at a per-row index (`x[arange, idx]`) beside a reduction over strided columns, and `CppTile2DKernel.load` then emits
  a TRANSPOSED tile load for any index whose outer stride is 1 (`need_vec_transpose`, `torch/_inductor/codegen/cpp.py`). That
  check reads only the strides of the two tile variables and does not exclude an index containing an INDIRECT variable
  (`tmp8`, defined later in the inner loop), so the hoisted load names `tmp8` before it exists. It surfaced when the end state
  became production (derived move resolution: ~178 lines of `agents/model` use the `x[ar, ...]` idiom).
- **The smallest repro** (`cpu_inductor_tile2d_test.py::_repro`, no model; B = 16, K = 18):

  ```python
  def repro(x, idx, q):               # x [B,6,K] · idx [B] long · q [B,K,6]
      g = x[torch.arange(B), idx]     # a row gathered at each batch row's own slot   (arange INSIDE the graph)
      return (g * q[..., 5]).sum(-1), g   # a [B] reduction over K of a strided column; the gather stored
  torch.compile(repro, backend="inductor", fullgraph=True)(x, idx, q)   # CPU, torch 2.8.0 -> CppCompileError
  ```

  Each ingredient is needed (`designs/research_state/measurements/cpu_inductor_tile2d_2026-10-15/`). **Shape threshold** (AVX-512,
  16 fp32 lanes): fails for B >= 16 AND K >= 16; compiles for B 8 / 15 or K 8. **UNVERIFIED:** the threshold on an AVX2 box
  (expected 8). The K9 learner golden's micro-batch is 16, which is why the CPU cell hit it.
- **A known CLASS upstream** (pytorch/pytorch#193581 open, #181621 closed; both newer than 2.8.0, summaries not independently
  checked): a torch bump is not a known fix, and it is **UNVERIFIED** whether either covers this repro.
- **The decision: a CPU-only compile option.** `CPU_INDUCTOR_OPTIONS = {"cpp.simdlen": 1}` turns the CPP backend's
  vectorisation, and so the tiling, off: scalar C++ kernels of the same graph (only fp32 summation order inside a kernel may
  move). **A CUDA learner's compile kwargs are unchanged byte for byte** (`cpu_inductor_tile2d_test::test_install_hands_R1_the_cpu_option_and_a_cuda_learner_nothing`).
  Not taken: RESPELLING (`x.gather` lowers to the same indirect load; only a one-hot sum compiles, but it rewrites ~178 sites of
  the PRODUCTION CUDA graph and needs the gate and canary re-judged on the GPU), and declaring CPU unsupported (it would drop
  the only CPU coverage of the donated-buffer pin). A TRIPWIRE test fails when torch 2.8.0 stops failing the repro, so the
  workaround is retired with the bug.
- **Every other CPU compile site carries the same option** through ONE seam (next subsection).
- **The CUDA twin** (`compiled_first_update_test.py -m slow -k on_cuda`) is banked green at `51c95abc` (2026-10-25,
  `designs/ops/slow_tier_status.json`). Re-run it under a lease after a graph change:
  `GEN3AI_GPU_LEASE_TOKEN=<token> GEN3AI_TEST_ALLOW_GPU=1 scripts/ops/gpu_lock.sh timeout 1200 "$PY" -m pytest
  src/agents/model/compiled_first_update_test.py -m slow -k on_cuda -q -p no:cacheprovider` (~3 min).

History: archive, same heading (the bisect, the kernel read from `TORCH_COMPILE_DEBUG`, the evidence timings).

#### Every CPU compile site goes through ONE seam (`gen3_cpu_compile_sites_v1`, 2026-10-19)

**The rule.** `agents.model.inductor_compile.compile_on(fn, device=..., **torch_compile_kwargs)` is `torch.compile` plus
`CPU_INDUCTOR_OPTIONS` when, and only when, the compile reaches Inductor's C++ codegen: the target device is CPU and the
backend is Inductor (`None` / `"inductor"`). A CUDA compile (Triton) and a stand-in backend (`"eager"`, `"aot_eager"`, a
callable) get their kwargs back unchanged, byte for byte. The workaround is merged LAST, so a caller's option cannot
re-enable the tiled kernel; `cpu_workaround=False` exists for the bug's own repro and is refused anywhere else. A backend that
calls Inductor itself takes `inductor_options(device)` as its `config_patches` / `inductor_configs`.
`compile_regions.CPU_INDUCTOR_OPTIONS` / `cpu_inductor_options` are re-exports of the seam's.

**The guard** (`inductor_compile_test.py`, routine tier): an AST scan of `src/` FAILS a `torch.compile` call outside the seam
(a TEST may name a non-Inductor stand-in backend), a `compile_fx` / `aoti_compile_and_package` call in a module that does not
read `inductor_options`, and a `cpu_workaround=` outside the repro's two files; it asserts it SEES every production site below
and catches planted offenders. `compile_cache_test::test_every_production_compile_site_declares_the_hermetic_cache` counts a
`compile_on` call as a compile site too. **Not seen by the scan:** `torch.compile` inside a child-process SCRIPT STRING
(one-line elementwise lambdas that cannot form the pattern) and an alias (`f = torch.compile; f(...)`).

**The inventory** (every compile call in `src/`, 2026-10-19). Routes through `compile_on(device=…)`: R1 and its graphed twin
(`compile_regions.install`; the CPU test seams reach C++ at B = 16, production is CUDA only), `maybe_compile_extractor` (warmed
at B = 1, but a reader's larger batch compiles a NEW dynamic graph at call time, and `_eager_fallback_on_error` turned that g++
failure into a silent ~6x eager fallback), `main/ops/diag_bf16_canary` on `--device cpu`, `main/compile_inventory`
(`fullgraph_check.run`; the capture backend calls `compile_fx(config_patches=inductor_options(device_of_inputs(…)))` itself),
T2's `inference/service/engine._build_graphs` (CUDA graphs; `ServiceSpec` refuses backend `graph` / `aot` off CUDA, so the
backlog row "T2 `decide` on CPU" is not reachable), `inference/service/aot.build_package` (`inductor_options(static_obs.device)`
merged into its `inductor_config.patch`) and the CPU compile tests.

**The proof on the failing shapes** (B = 16 / 32 / 64; all `slow`): `inductor_compile_test` compiles the 4-line repro at each B
through `compile_on` and through the capture's own `compile_fx`; `cpu_compile_sites_slow_test` compiles the PRODUCTION
extractor through `compile_on` at a static B on the committed real parity rows and drives `maybe_compile_extractor` to install
at B = 1 and then serve B = 16 / 32 / 64 COMPILED (it fails on the fallback warning, not only on an exception). Without the
option the production extractor's CPU compile FAILS in g++ at B = 16 and B = 64 no-grad and at B = 16 with grad, so
`maybe_compile_extractor`'s first B >= 16 call had been FALLING BACK to eager since the end state became production.

### Triton RemoveLayoutConversions at XBLOCK 16 (triton 3.4.0, 2026-10-25): the CUDA inference compile at B = 2 only

`extractor_compiles_test.py::test_production_arch_compiles_on_cuda` (the CUDA eval / no-grad cell) failed at B = 2 in the
2026-10-25 window (`InductorError … make_ttgir … PassManager::run failed`); its last pass was 2026-10-04 and it was skipped
(busy card) until 10-21, so no window saw it go red.

**The cause** (found on the CPU: Inductor's Triton codegen driven with fake device properties, every kernel compiled for sm_86
by `triton.compile`, which needs no GPU; `designs/research_state/measurements/triton_xblock16_2026-10-25/`): at B = 2, 2 of the
forward's 656 kernels fail (0 at B = 4, 8, 64, 256), both `triton_red_fused__to_copy_add_clamp_cumsum_index_mul_ones_like_reciprocal_sub_where`:
the speed mixture's fp64 cumsum (`damage_op_speed._opp_speed_mix`) with the `[B,6]` Beat-Up / max-HP pointwise epilogue
(`damage_kinds.beatup_base_def`) fused in, which makes that epilogue `[XBLOCK, 1]`-shaped. Every autotune config with `XBLOCK =
16` fails Triton's `TritonGPURemoveLayoutConversions` ("operand #1 does not dominate this use"); `XBLOCK = 16` is chosen only
when the `6B` rows round to 16, i.e. B = 2. The epilogue alone fails (`triton_xblock16_remat_test.py` pins it, routine tier) and
the fp64 is not the cause; triton 3.1.0 compiles the pinned kernel: a Triton regression, reaching this graph with the discrete
speed mixture (the op's only speed rule since legacy deletion U1, config v156). Confirmed on the card
(`~/gen3ai_archive/extractor_crash_2026-10-25/`).

**No launch is at risk.** The served forward is compiled only at T2's buckets, `>= MIN_BUCKET = 8` by default
(`rust_rollout/build.py`; a smaller `--t2-buckets` would fail LOUD at T2 startup, `suppress_errors` off), and R1 compiles a
train-mode forward + backward. **The fix** is the cell's shape: it runs at `MIN_BUCKET`, the smallest shape a default launch
compiles; production kernels and numerics are untouched. When the pin reports the bug lifted, the cell may return to `_BATCH`.
**B = 1 fails differently** (UNVERIFIED on the card): 33 kernels with `xnumel = 1` die in Triton's frontend
(`tl.broadcast_to` of a constant index: `'constexpr_type' object has no attribute 'is_block'`). Nothing compiles the extractor
at B = 1 on CUDA today; a new CUDA B = 1 compile (a `--t2-buckets 1`) must be proven on the card first.
