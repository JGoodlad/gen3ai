# Running Gen3AI

The operational guide: environment setup, training, evaluation, and the test tiers.
The architecture itself is documented in [`designs/ARCHITECTURE.md`](../designs/ARCHITECTURE.md)
(the only doc that describes the model *as it is now*), and the exhaustive operational detail —
every flag, benchmark, and failure mode — lives in the `CLAUDE.md` files beside the code
(start at the [root one](../CLAUDE.md)).

## Environment

Python via a conda environment:

```bash
conda env create -f environment.yml        # creates gen3ai_stable
conda env update -f environment.yml        # after environment.yml changes
conda activate gen3ai_stable
export PYTHONPATH=$PYTHONPATH:src          # every command below assumes this
```

The battle simulator is a git submodule plus build artifacts:

```bash
git submodule update --init                             # deps/pokemon-showdown source
cd deps/pokemon-showdown && npm install && npm run build  # node_modules + dist/
cargo build --release --bin sim_bridge --bin search_driver \
    --manifest-path src/rust_sim/Cargo.toml             # the Rust simulator (recommended)
```

Build the Rust binaries before your first test run — a fresh checkout otherwise pays for
`cargo build` inside the first Rust-backed test, which can saturate the box and cascade into
spurious timeouts.

## Training — no server required

Training is **serverless by default**: `--use-bridge` defaults to `rust`, which runs battles
through an in-process reimplementation of the Gen 3 battle engine. Quick smoke (~1 minute, CPU):

```bash
python src/main/train_rl_agent.py --debug --steps 10000
```

Look for `[ModelVersion] Round-trip smoke test PASSED` early and `Training complete` at the end.

For real runs, use the **launcher** — it wraps the trainer with periodic restarts (memory
hygiene), crash auto-restart from the last checkpoint, a terminal dashboard, and git-worktree
isolation (the run is pinned to its launch commit, so pushes to `main` never disturb it):

```bash
python -m main.launcher \
  --restart-interval-hours 3 \
  --steps 15000000 --n-envs 64 --batch-size 16384 \
  --n-epochs 10 --ent-coef 0.02 --n-steps 2048 --lr 0.0003 \
  --device cuda --log-level periodic
```

Resume from a checkpoint (the launcher pins to the checkpoint's recorded commit):

```bash
python -m main.launcher --model models/<run>/checkpoints/checkpoint_NNNN_steps.zip \
  --restart-interval-hours 3 --steps 15000000 --device cuda
```

Checkpoints land in `models/run_<timestamp>/checkpoints/`; TensorBoard logs beside them
(`tensorboard --logdir models/`).

**`torch.compile` is on by default** — the CPU env-worker opponents (`--compile-opponents`, plus its
forkserver preload) and, when the resolved device is `cuda`, the GPU learner (`--compile-trainer`,
~1.75x on the PPO train step). The flags exist as fallbacks: `--no-compile-opponents` /
`--no-compile-trainer` return to eager if the compiler is ever the suspect. Two things worth knowing
before a launch: `--compile-trainer` drops the per-forward ObservationDebugger (it says so at
startup; `--no-compile-trainer` keeps it), and a compile failure there is FATAL by design rather
than a silent fall back to a ~1.75x slower run. The CPU smoke above is unaffected — the trainer
compile is off on `cpu` and off under `--debug`.

## Tests

Two orthogonal marker axes: capability (*what a test needs* — `integration`, `sim`, `browser`,
`e2e`) and cost (`slow`). Cut on **`slow`**, not on capability:

| When | Command | ~Time |
|---|---|---|
| Fast inner loop | `pytest src/ -m "not slow and not e2e and not sim and not integration" -q -n 2` | ~1.5 min |
| **The routine gate** (before any commit) | `pytest src/ -m "not slow and not e2e" -q -n 2` | ~4 min |
| Everything (before a release/ship) | `pytest src/ -q` | ~31 min |

Two static gates run inside the suite (and independently):

```bash
python -m mypy src/agents/model                                   # typed packages, zero errors
ruff check src/agents src/main src/utils --select F,E9            # real-bug lint classes
```

**Fuzz tests** run real battles through the in-process bridge (no server) and validate
observations against the actual protocol stream — run them directly as scripts, e.g.:

```bash
python src/agents/action/fuzz_test.py 50
python src/agents/training/poke_env_gaps/transition_fuzz_test.py 50
```

## The Showdown server (optional)

Only the live-server paths need it — ladder play, `--use-bridge off`, and `*_e2e_test.py`
scripts:

```bash
npm run showdown            # port 8000 (the port is a positional arg; there is no --port flag)
npm run showdown -- 8001
npm run stop                # stop :8000 (Ctrl+C orphans subprocesses — use this)
npm run stop -- 8001
```

Convention on a shared box: 8000 = development, 8001 = training (`main.launcher` defaults to
8001), anything ephemeral on 9XXX.

## Evaluation and forensics

```bash
python src/main/play.py                          # play/evaluate against the server
python -m main.elo models/<run>                  # offline ELO ladder + Elo-vs-step curve
python -m main.prober models/<run>               # forensic replay inspector (web UI, :6008)
```

The prober reads the eval traces a run writes: per-decision analysis, luck-vs-mistake dice
attribution (`falsify`), counterfactual replays, and a beam search for better lines — see
`src/main/prober/CLAUDE.md`.

## Working in git worktrees

A fresh worktree gets an empty submodule dir. Two steps (do **not** symlink the whole
`deps/pokemon-showdown` directory — it breaks `git status`):

```bash
git submodule update --init
ln -s <main-checkout>/deps/pokemon-showdown/dist deps/pokemon-showdown/dist
ln -s <main-checkout>/deps/pokemon-showdown/node_modules deps/pokemon-showdown/node_modules
```
