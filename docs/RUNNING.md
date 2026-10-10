# Running Gen3AI

The operational guide: environment setup, training, evaluation, and the test tiers.
The architecture itself is documented in [`designs/ARCHITECTURE.md`](../designs/ARCHITECTURE.md)
(the only doc that describes the model *as it is now*), and the exhaustive operational detail —
every flag, benchmark, and failure mode — lives in the `CLAUDE.md` files beside the code
(start at the [root one](../CLAUDE.md)).

## Environment

**One command does all of it**, idempotently, and verifies itself at the end:

```bash
./scripts/bootstrap.sh          # --dry-run to see the plan; --with-rust to skip the prompt
conda activate gen3ai_torch28   # torch 2.8 — the default (gen3ai_stable = legacy 2.5.1, old resumes only)
export PYTHONPATH=$PYTHONPATH:src          # OPTIONAL here; MANDATORY in a git worktree
```

The bootstrap's step 3 is `pip install -e .`, which puts `src/` on the import path for good — so
every command below works without the export **in the main checkout**. It is still load-bearing
where the install does not reach: a **git worktree** (the install names the main checkout's `src/`,
so a worktree without the export runs its own tests against main's code — the one silent failure
in this area), a container, a machine that skipped the bootstrap. Both spellings are correct;
neither is being retired.

Setup mechanics for contributors — flags, the CPU-only variant, the worktree case — are in
[`DEVELOPING.md`](DEVELOPING.md). What the script actually does, should you need to run a
step by hand:

```bash
conda env create -f environment_torch28.yml   # creates gen3ai_torch28
conda env update -f environment_torch28.yml   # after environment_torch28.yml changes
# environment.yml (gen3ai_stable, torch 2.5.1) is LEGACY and frozen: only to resume an old run

git submodule update --init                             # deps/pokemon-showdown source
cd deps/pokemon-showdown && npm ci && npm run build     # node_modules + dist/
cargo build --release --bin sim_bridge --bin search_driver \
    --manifest-path src/rust_sim/Cargo.toml             # the Rust simulator (recommended)
```

Build the Rust binaries before your first test run — a fresh checkout otherwise pays for
`cargo build` inside the first Rust-backed test, which can saturate the box and cascade into
spurious timeouts.

Two `environment_torch28.yml` details that are load-bearing: the pip block opens with
`--extra-index-url https://download.pytorch.org/whl/cu126` (the `torch==2.8.0+cu126` pins are
local-version builds that PyPI does not carry, so without it `conda env create` fails on a fresh
machine), and `poke-env` is **deliberately not installed** — poke-env is retired (the vendored fork was deleted in
T27 P6) and nothing here imports it. `src/poke_env_absent_gate_test.py` guards both.

## Training — no server required

Training is **serverless**: the in-process Rust bridge is the only transport, and runs battles
through an in-process reimplementation of the Gen 3 battle engine. Quick smoke (~2 minutes, CPU;
set `GEN3AI_MODELS_DIR` to a scratch directory to keep a throwaway run out of `models/`):

```bash
export GEN3AI_MODELS_DIR=$(mktemp -d)
python src/main/train_rl_agent.py --debug --steps 10000
```

Look for `🦀 [ENV CORE] rust` and `[ModelVersion] Round-trip smoke test PASSED` early and
`Training complete` at the end. `--debug` skips evaluation; add `--debug-eval --eval-freq 4000` to
run two in-process eval cycles as well (`--debug-eval` alone fires none: the default interval is
2,000,000 steps).

For real runs, use the **launcher** — it wraps the trainer with periodic restarts (memory
hygiene), crash auto-restart from the last checkpoint, a terminal dashboard, and git-worktree
isolation (the run is pinned to its launch commit, so pushes to `main` never disturb it):

```bash
python -m main.launcher \
  --restart-interval-hours 6 \
  --steps 15000000 \
  --device cuda --log-level periodic --arch production
```

🚨 **`--arch production` is not optional on a fresh run.** Every architecture toggle defaults to
OFF, so an argv that omits it trains a near-bare network that launches cleanly and looks healthy.
It applies the whole surface in `designs/production_config.json` as if each flag had been typed —
the architecture AND the production training recipe (the file's `recipe.fresh` block — the
lineage's measured fresh launch: env count, batch, epochs, learning rate, clip, entropy, self-play;
the critic and its win-indicator terminal are constants of every trainer namespace) — and an
explicitly-typed flag still wins. Validate any argv offline first with
`python -m main.checkargs --argv "…"`, which prints an architecture-surface and a recipe-surface
diff and refuses a fresh argv that differs from production (a recipe knob only when it was not
typed). Resolve what a launch would actually do, changing nothing,
with `python -m main.launcher --dry-run …` — never by launching the real command and killing it,
which is destructive on a restart.

Resume from a checkpoint (the launcher pins to the checkpoint's recorded commit):

```bash
python -m main.launcher --model models/<run>/checkpoints/checkpoint_NNNN_steps.zip \
  --restart-interval-hours 6 --steps 15000000 --device cuda
```

On a resume an argv is not a config: every flag you do not name is inherited from the checkpoint's
`model_config.json`, and `--lr`, `--batch-size` and `--n-steps` are inert because SB3
restores the checkpoint's own values. A bare run directory anywhere a model is expected means that
run's **last snapshot** — name the `.zip` to pin a file.

Checkpoints land in `models/rb_run_<timestamp>/checkpoints/` (the `rb_` prefix is the current
era's code, `src/utils/era.py`); TensorBoard logs beside them (`tensorboard --logdir models/`). From
a git worktree the run still lands in the MAIN checkout's `models/`.

**`torch.compile` is on by default for the GPU learner** (`--compile-trainer`, auto-on when the
resolved device is `cuda`; ~1.75x on the PPO train step). `--no-compile-trainer` returns to eager if
the compiler is ever the suspect. It announces itself at startup, and a compile failure there is
FATAL by design rather than a silent fall back to a slower run. There is no opponent compile flag:
opponents are served by the inference service (the old `--compile-opponents` flags are deleted,
`designs/deleted_flags.md`). The CPU smoke above is unaffected — the trainer compile is off on `cpu`
and under `--debug`.

## Tests

Two orthogonal marker axes: capability (*what a test needs* — `integration`, `sim`, `browser`,
`e2e`) and cost (`slow`). Cut on **`slow`**, not on capability:

| When | Command | ~Time |
|---|---|---|
| Fast inner loop | `pytest src/ -m "not slow and not e2e and not sim and not integration" -q -n 2` | ~1.5 min |
| **The routine gate** (before any commit) | `pytest src/ -m "not slow and not e2e" -q -n 6` | ~4.3 min (`-n 4` beside a training run) |
| Everything (before a release/ship) | `pytest src/ -q` | ~47 min |

Seventeen static gates run inside the suite, all declaring the `static` tier so they run in every tier:
mypy, ruff, file size, `CLAUDE.md` freshness, stub vacuity, slow-tier status, the `ARCHITECTURE.md`
mode-flag mirror, the training-recipe mirror, the ledger index, the eval-trace summary readers, the
enum-string comparison, the learner lifecycle, the global-RNG seeding, the strict checkpoint load, the
eval-ledger readers, the poke-env import ratchet and the poke-env absence. (`src/packaging_gate_test.py`
guards the import path and is not one of them.) The full table with each gate's opt-out is in
[`DEVELOPING.md`](DEVELOPING.md#the-static-gates); `python -m utils.static_gates` lists them from the
`pytest.mark.static` declarations, and `src/utils/static_gates_test.py` holds every count to that list.

```bash
python -m mypy                                                    # scope from mypy.ini's files =
ruff check src/agents src/main src/utils --select F,E9            # real-bug lint classes
```

**Fuzz tests** run real battles and validate against ground truth (the protocol stream or the omniscient
engine) — on the Rust core since the poke-env retirement; the rust_sim fuzzers are listed in
`src/rust_sim/CLAUDE.md`, e.g.:

```bash
node src/rust_sim/harness/ab_fuzz.js --battles 200
```

## The Showdown server (optional)

Only the live-server paths need it — `play.py` against a local server and the `*_e2e_test.py`
scripts:

```bash
npm run showdown            # port 8000 (the port is a positional arg; there is no --port flag)
npm run showdown -- 8001
npm run stop                # stop :8000 (Ctrl+C orphans subprocesses — use this)
npm run stop -- 8001
```

Convention on a shared box: 8000 = development, 8001 = the standing training server (other tools
use it; training itself connects to no server), anything ephemeral on 9XXX.

## Evaluation and forensics

```bash
python src/main/play.py --mode selfplay --port 9017   # Rust-stack websocket client: selfplay/challenge/accept (ladder is REFUSED: we never play humans)
python src/main/ladder_drift_scan.py --n 200      # pre-flight protocol-drift gate (public replays)
python -m main.elo models/<run>                  # offline ELO ladder + Elo-vs-step curve
python -m main.prober models/<run>               # forensic replay inspector (web UI, :6008)
```

The prober reads the eval traces a run writes: per-decision analysis, luck-vs-mistake dice
attribution (`falsify`), counterfactual replays, and a beam search for better lines — see
`src/main/prober/CLAUDE.md`.

## Working in git worktrees

A fresh worktree gets an empty submodule dir. `./scripts/bootstrap.sh` detects the worktree and
does this for you; by hand it is two steps (do **not** symlink the whole `deps/pokemon-showdown`
directory — it breaks `git status`):

```bash
git submodule update --init
ln -s <main-checkout>/deps/pokemon-showdown/dist deps/pokemon-showdown/dist
ln -s <main-checkout>/deps/pokemon-showdown/node_modules deps/pokemon-showdown/node_modules
```

⚠️ Guard those with `[ -e ]` and run them **only** in a worktree. In the main checkout `dist`
already exists as a real directory, so `ln -s TARGET dist` creates `dist/dist` pointing at its
own parent, `node build` dies with `ELOOP`, and every websocket-server path stops working.
