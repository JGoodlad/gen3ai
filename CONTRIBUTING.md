# Contributing to Gen3AI

Contributions of any size are genuinely wanted — an idea, a question, a single test, or a
subsystem. **What to work on** (the good-first list, drawn from real open items) is in the
[README's Contributing section](README.md#contributing). This page is the mechanical part: how a
contribution lands, how to get a working checkout, what to run before you open a pull request, and
the handful of local conventions that are not obvious from the code.

## How a contribution lands

- **Outside contributors: fork, branch, open a pull request against `main`.** The maintainer works
  differently — edits in git worktrees and pushes straight to `main` (see
  [Git workflow](#git-workflow--never-commit-on-main)) — but that path is the maintainer's;
  everything from outside comes in as a PR and is reviewed there.
- **Open an issue first for anything bigger than a fix** — a new observation fact, a model change,
  an experiment. A change to what the network reads is run as ONE lever with its own screen, so it
  is scheduled, not merged blind.
- **Proposing an experiment:** an issue stating the ONE lever, the question, the meter you would
  read and the decision rule (what result keeps it, what kills it). Experiments here are
  pre-registered before their first game; the queue is
  [`designs/research_state/EXPERIMENT_BACKLOG.md`](designs/research_state/EXPERIMENT_BACKLOG.md)
  and the evidence vocabulary is
  [`designs/research_state/UNDERSTANDING.md`](designs/research_state/UNDERSTANDING.md) §0.
- **"The model got this wrong" reports are contributions.** Link the battle in the prober's `/game`
  view ([prober.g5d.io](https://prober.g5d.io)), name the turn, and say what you would have clicked
  and why.

### What a pull request needs

1. **A test that fails if your fix is reverted** — for a mechanics bug, a named, deterministic pin
   of the exact case. A test that passes either way is not a test of the fix.
2. **A green test run you name in the PR.** The routine gate (below) is the default; a TARGETED set
   you choose plus the static gates is fine when the blast radius is clearly contained — say which
   tests you ran and why that scope is enough. Shared infrastructure, or any doubt, means the
   routine gate.
3. **Docs in the same PR.** If your change makes a `CLAUDE.md`, a `README.md` or
   `designs/ARCHITECTURE.md` stale, fix it there (see
   [Conventions](#conventions-worth-knowing-before-you-write-code)).
4. **Mechanics claims verified at the source.** Anything about how Gen 3 works is checked against
   `deps/pokemon-showdown/data/mods/gen3/` (and the generations it inherits from), with the file
   named in the PR.

### Conduct

Be kind and direct, and argue with ideas, not people. **Respect the people on Pokémon Showdown**
(the README's [Respect for players](README.md#respect-for-players)). They're real people playing a
game they love. Our own work plays the agent against itself and against bots, and we haven't taken
it onto the public ladder. If you run this code on Showdown, follow the server's rules and its
policy on bots, and treat every opponent with respect.

The research context lives elsewhere and is worth reading if you want to work on the model:
[`designs/ARCHITECTURE.md`](designs/ARCHITECTURE.md) is the only document that describes the
architecture **as it is now**, and [`docs/RUNNING.md`](docs/RUNNING.md) covers training and
evaluation in depth.

---

## Setup — one command

```bash
git clone git@github.com:JGoodlad/gen3ai.git
cd gen3ai
./scripts/bootstrap.sh
```

That is the whole story. The script is idempotent (re-run it any time; completed steps are
skipped), fail-loud, and it announces what each step costs before spending it:

| Step | What | Cost |
|---|---|---|
| 1 | prerequisite check — `git`, `conda`, `node`, `npm` | instant |
| 2 | create/update the `gen3ai_torch28` conda env (torch 2.8) from `environment_torch28.yml` — **a worktree never updates it silently** (see below) | ~5-15 min fresh (≈2 GB of wheels) |
| 3 | `pip install -e .` — puts `src/` on the import path for good | ~2 s |
| 4 | `git submodule update --init` — the Pokémon Showdown reference engine | ~30 s |
| 5 | the Showdown build artifacts (`npm ci` + `node build`) — **or** worktree symlinks | ~3-6 min fresh |
| 6 | *(optional)* `cargo build --release` for the Rust simulator | ~3-10 min cold |
| 7 | verify — the ruff and mypy gates, the two import-precedence gates, a ~10 s unit smoke | ~30-60 s |

Useful flags: `--dry-run` (print the plan, change nothing), `--with-rust` / `--no-rust`,
`--force` (redo the conda step), `--update-shared-env`, `--skip-env`, `--no-check`, `--help`.

**The conda env is SHARED; a checkout is not.** Every new run, gate and agent on the box uses
the one `gen3ai_torch28`, and a `conda env update --prune` under a live process can swap a package
out from under it. So step 2's "env is current" stamp lives in the git **common** dir (one per
clone, keyed by the sha256 of `environment_torch28.yml`, holding a copy of the file it was written for):

| run from | `environment_torch28.yml` matches the stamp | it differs (or `--force`) |
|---|---|---|
| the main checkout | nothing to do | updates the env in place, and warns that it is shared |
| a linked worktree | nothing to do | prints the diff and **REFUSES (exit 3)** |

A worktree refusal names both ways forward: `--update-shared-env` (update it anyway — only when no
live or pinned run uses it), or `--skip-env` (leave the env alone and finish the rest of the
worktree's setup). A missing env is simply created, from anywhere — nothing can be running on it.
The refusal exists because a worktree bootstrap cannot see which runs are live, queued or pinned;
before it (2026-09-29) the stamp was per-worktree, so every fresh worktree ran a prune-update.

**The LEGACY env.** `gen3ai_stable` (torch 2.5.1, `environment.yml`) was the env until 2026-09-30
(owner: "go ahead with moving to 2.8"). It is kept, FROZEN, only so a run trained on it resumes on
it — the launcher reads the run's recorded torch (`metadata.json` `torch_version`; none = 2.5.1) and
selects that env, or refuses (`src/main/launcher/torch_runtime.py`). Bootstrap never touches it;
never edit `environment.yml`. On a fresh machine you do not need it unless you resume such a run.

**Say yes to the Rust build if you have ten minutes.** Training runs on the in-process Rust bridge (the only transport),
and the first Rust-backed test builds those binaries *anyway* — mid-test, saturating every core, a
documented cause of spurious timeout failures on a fresh checkout.

### After bootstrap: activate, and that is it

```bash
conda activate gen3ai_torch28
python -c "import agents, main, utils; print('ok')"
```

**No `export PYTHONPATH` needed — in the main checkout.** Step 3 of the bootstrap runs
`pip install -e .`, which puts this checkout's `src/` on the import path permanently, so
`import agents` works from any directory, shell, IDE and debugger. Without the install —
a container, a machine that skipped the bootstrap — `export PYTHONPATH=$PYTHONPATH:src` is
exactly equivalent, and harmless when the install does exist.

**In a git worktree the export is not a fallback but a requirement**, because the install names
ONE absolute path: the main checkout's `src/`. A worktree with no `PYTHONPATH` runs its own tests
against *main's* code. See [Git workflow](#git-workflow--never-commit-on-main).

> **Install from the MAIN CHECKOUT, never from a git worktree.** An editable install records one
> absolute path in a `.pth`. Install from a worktree, delete the worktree, and Python skips the
> missing entry *in silence* — imports start failing for a reason the install never reports.
> `src/packaging_gate_test.py` catches a stale one and prints the fix.

**`pyproject.toml` declares no dependencies, on purpose.** The env file (`environment_torch28.yml`) is the single owner
of what is installed — including a CUDA-local-version torch that PyPI does not carry — so
`pip install -e .` writes a `.pth` and a `dist-info` and cannot resolve, upgrade or replace anything
in your environment. One owner per question.

### CPU-only machines

`environment_torch28.yml` installs CUDA builds of torch, but nothing is CUDA-gated at import time — a
CPU-only box runs the entire test suite, the prober, and `--device cpu` training; it just downloads
~2 GB of wheels it will never use. A CPU variant is derived the usual way (extra index at
`.../whl/cpu`, drop the `+cu126` suffixes, the `nvidia-*-cu12` / `triton` pins and the `cuda-*`
header packages). We ship no such file because nobody here runs one; a good one is a welcome PR.

### Do not install `poke-env` from PyPI

poke-env is **retired** (T27, finished at P6, 2026-10-08): the vendored fork `src/poke_env/` is deleted, and
nothing in `src/` / `tools/` / `scripts/` imports `poke_env`. Every battle, row, eval game and live game is
the Rust core's. Installing the PyPI package would only add a package nothing here uses (the one process
that still runs upstream poke-env is the Metamon peer script, in Metamon's own interpreter).
`src/poke_env_absent_gate_test.py` keeps it out for good — no `poke_env` package under the scanned
roots, none in site-packages, neither env file installing one. To check by hand:

```bash
python -c "import poke_env"   # must raise ModuleNotFoundError
```

---

## Tests — what to run, and when

Two **orthogonal** marker axes. A *capability* marker says what a test **needs**; the single
*cost* marker `slow` says what it **costs**. Cut on cost, never on capability:

| When | Command | ~Time |
|---|---|---|
| Inner loop — fastest true/false | `pytest src/ -m "not slow and not e2e and not sim and not integration" -q -n 2` | ~1.5 min |
| **The routine gate — before any commit** | `pytest src/ -m "not slow and not e2e" -q -n 6` | ~4.3 min (one at a time; `-n 4` beside a training run) |
| Everything — before a release | `pytest src/ -q` | ~47 min |

The routine gate's ~4.3 min at `-n 6` was measured on a quiet box on 2026-10-01
([`designs/ops/testing.md`](designs/ops/testing.md)); the other two are older — treat them as an
order of magnitude, not a budget. Counts collected 2026-10-09: 11,193 tests in all, 11,144 in the
routine gate, 10,633 in the inner loop.

| Marker | Means |
|---|---|
| *(unmarked)* | pure in-process; needs nothing |
| `integration` | an out-of-process dependency, but no battles and no browser |
| `sim` | plays real battles in-process through the bridge (no server) |
| `browser` | needs headless chrome |
| `e2e` | needs a live Showdown server |
| **`slow`** | minutes, not seconds — **the one marker that decides routine cost** |
| `benchmark` | a measurement, not a pass/fail gate — its bounds are never scaled |

⚠️ **Do not use the old `-m "not integration and not e2e"` gate.** `integration` now spans a
~100× cost range, so excluding it throws away cheap, high-value coverage — that exact gate is how
the six-battle obs-golden test rode `main` red three separate times. `-n 2` is `pytest-xdist` and
is ~1.8× faster; use plain serial when you need `-s` or a debugger.

**Seventeen static gates run inside the suite** (there is no CI on this box, so a check outside the
suite is a check that rots). All declare the `static` tier, so they run in every tier, and all are ~free warm:

| Gate | Checks | Opt-out |
|---|---|---|
| `src/agents/model/mypy_gate_test.py` | `python -m mypy`, scope from `mypy.ini`'s `files =` (`src/agents/model`, `src/agents/observation`) | `GEN3AI_SKIP_MYPY_GATE=1` |
| `src/ruff_gate_test.py` | pyflakes over `src/agents`, `src/main`, `src/utils` — `--select F,E9`, wrongness only, never style | `GEN3AI_SKIP_RUFF_GATE=1` |
| `src/file_size_gate_test.py` | a file over 2,000 lines fails; 1,000-2,000 is reported. The allowlist is empty — decompose instead | `GEN3AI_SKIP_SIZE_GATE=1` |
| `src/claude_md_freshness_gate_test.py` | every repo-relative path and every `--flag` a `CLAUDE.md` names still resolves | `GEN3AI_SKIP_CLAUDE_MD_GATE=1` |
| `src/test_stub_vacuity_gate_test.py` | every `monkeypatch.setattr` / `patch` target is a symbol the code under test actually reads — a stub that stubs nothing fails | `GEN3AI_SKIP_STUB_GATE=1` |
| `src/slow_tier_status_gate_test.py` | the last recorded verdict of every `slow` test — a recorded failure fails the routine gate that deselected it | `GEN3AI_SKIP_SLOW_STATUS_GATE=1` |
| `src/mode_flag_doc_gate_test.py` | every mode-flag value `designs/ARCHITECTURE.md` states in prose equals `designs/production_config.json` | `GEN3AI_SKIP_MODE_FLAG_DOC_GATE=1` |
| `src/recipe_doc_gate_test.py` | every training-recipe value `designs/endstate/design_learner_recipe.md` states equals the `recipe.fresh` / `recipe.fork` blocks of `designs/production_config.json` | `GEN3AI_SKIP_RECIPE_DOC_GATE=1` |
| `src/trace_summary_reader_gate_test.py` | no module but `main/prober/core_trace.py` opens an eval-trace `*_summary.json` — every reader goes through its loaders | `GEN3AI_SKIP_SUMMARY_READER_GATE=1` |
| `src/ledger_index_gate_test.py` | `designs/research_state/ledger_index.md` matches what `python -m main.ledger_index` renders | `GEN3AI_SKIP_LEDGER_INDEX_GATE=1` |
| `src/enum_str_compare_gate_test.py` | no enum `agents.enums` defines is compared to a string (mypy `--strict-equality`, typed) — the bug that kept four bots from ever setting up | `GEN3AI_SKIP_ENUM_STR_GATE=1` |
| `src/poke_env_import_gate_test.py` | no file imports `poke_env` unless it is on the GENERATED, shrink-only `designs/ops/poke_env_import_allowlist.txt` (EMPTY since T27 P6: frozen counts 0 / 0) | `GEN3AI_SKIP_POKE_ENV_IMPORT_GATE=1` |
| `src/poke_env_absent_gate_test.py` | there is NO `poke_env` package in our tree, in this interpreter's site-packages, or in either env file (the vendored fork was deleted in T27 P6) | none (static, ~free) |
| `src/learner_lifecycle_gate_test.py` | no training-STEP path constructs an optimizer, an `nn.Parameter` or an `nn.Module` — nor a CUDA stream / graph / pool — outside a declared startup builder or a class's `__init__` / `_build` / `_setup_model`; the allowlist is empty | `GEN3AI_SKIP_LIFECYCLE_GATE=1` |
| `src/global_rng_seed_gate_test.py` | no non-test module binds a process-global SEEDING function by name or takes one as a value (the runtime reseed guard cannot see a pre-bound name); the allowlist is empty | `GEN3AI_SKIP_GLOBAL_RNG_GATE=1` |
| `src/strict_checkpoint_load_gate_test.py` | no non-test module loads a PPO checkpoint through sb3's NON-STRICT retry (`MaskablePPO.load` / `PPO.load`, `set_parameters(…, exact_match=False)`); every reader uses `agents.model.snapshot.load_checkpoint_strict` | `GEN3AI_SKIP_STRICT_LOAD_GATE=1` |
| `src/eval_ledger_reader_gate_test.py` | every reader of the eval COUNT ledger goes through `eval_ledger.read` / `read_by_regime` with a fully spelled-out `ReaderDecl`, and the ledger's closed lists equal `design_evaluation.md` §0b.2's table | `GEN3AI_SKIP_LEDGER_READER_GATE=1` |

One more guard the import path itself, and it is NOT a static-tier gate: `src/packaging_gate_test.py`
(`PYTHONPATH` still outranks the editable install). The count above is held to the `pytest.mark.static`
declarations by `src/utils/static_gates_test.py` (list them with `python -m utils.static_gates`). A missing linter **fails** rather than skips: a linter that silently opts out
reads exactly like a linter that found nothing.

**Fuzz tests are not parametrized unit tests.** In this repo a fuzzer plays *real battles* and
checks them against ground truth. The main one today is the Rust engine's A/B differential fuzzer:
it drives the real Showdown sim and the Rust port side by side and saves a standalone repro for
every divergence (`src/rust_sim/CLAUDE.md`, "A/B fuzzer"):

```bash
node src/rust_sim/harness/ab_fuzz.js --mode random --battles 200
```

Every divergence it finds becomes a named deterministic pin
([`designs/rust_sim/regression_pins.md`](designs/rust_sim/regression_pins.md)). A pytest-collected
test, by contrast, takes SEEDED battles from the Rust core, so it plays the same battle every run.

**Benchmarks warn instead of scaling.** Wall-clock *bounds* scale by measured CPU contention
(`src/utils/contention.py`), because this box usually carries a live training run and a timeout is
never a semantic outcome. Benchmarks get the opposite treatment — their output *is* the measurement,
so they print a "THE BOX IS BUSY" banner rather than stretching. Report whether the box was quiet.

---

## Ports — the one rule that is not negotiable

| Port | Owner |
|---|---|
| **8000** | development |
| **8001** | **the training server — NEVER stop, restart, or kill it** |
| **9XXX** | pick one for anything ephemeral you start |

Killing :8001 drops every websocket client at once and crashes a training run that may have been
going for days. `npm run stop` with no argument kills :8000 — never run a blanket `node`/`showdown`
kill, and only ever stop the port you personally started. Most work needs no server at all:
training and evaluation run on the in-process Rust bridge (the only transport), an in-process reimplementation of the Gen 3
engine. Prefer it for throwaway work.

```bash
npm run showdown -- 9001     # your own server, if you really need one
npm run stop -- 9001         # and only ever this one
```

---

## Git workflow — never commit on `main`

*This section is the MAINTAINER's workflow. Outside contributors fork and open a pull request
([How a contribution lands](#how-a-contribution-lands)); the worktree notes below still help if you
keep several branches checked out.* The maintainer works without pull requests, but **`main` must
never be dirty**: all edits and commits happen in a branch or a git worktree, and land on `main` by
push:

```bash
git worktree add ../gen3ai-myfeature -b myfeature
cd ../gen3ai-myfeature
./scripts/bootstrap.sh          # detects the worktree; symlinks instead of rebuilding; never updates the shared env
export PYTHONPATH=$PYTHONPATH:src   # MANDATORY here — see below
# ... work, test ...
bash scripts/land.sh myfeature ../gen3ai-myfeature
```

`scripts/land.sh` is the landing path, and it is four steps in a fixed order: run the static
gates **inside the worktree**, `git push origin myfeature:main` from the main checkout,
`git pull --ff-only` there so main is not left behind its own remote, then remove the worktree and
the branch — unless the run-data guard (`python -m utils.worktree_guard`) finds a main-checkout
`models/` symlink into the worktree, or more than 50 MiB of untracked/ignored non-build content in
it, in which case the code is landed but the worktree and branch are KEPT and it exits 3 (a
2026-09-23 forced removal destroyed eight run directories that lived inside worktrees). Any gate
failure exits without pushing. It never force-pushes — a rejected
non-fast-forward means someone landed first, so rebase the worktree on `main`, resolve, and run it
again. Run the routine gate yourself before you call it; the script gates statics, not the suite.

🚨 **The export is mandatory in a worktree for anything but pytest, and it is the one thing that
fails silently.** A `.pth` entry cannot know which worktree you are standing in, so a
`python <script>` run here with no `PYTHONPATH` imports *main's* code — every result is then about a
tree you did not edit. `pytest` is covered for you: since 2026-10-07 the root `conftest.py` puts
THIS checkout's `src/` first on `sys.path` and in `PYTHONPATH` for the session, its workers and
every subprocess a test spawns. `bootstrap.sh` **skips** the install step in a worktree rather than
pointing the `.pth` at a directory that will later be deleted, and `src/packaging_gate_test.py`
fails loudly on a stale one. Export first anyway.

A fresh worktree gets an empty submodule directory and no build artifacts. `bootstrap.sh` detects
a linked worktree and symlinks `dist/` and `node_modules/` from the main checkout rather than
spending six minutes rebuilding them.

> 🚨 If you do it by hand, **guard the symlink with `[ -e ]`**. In the *main* checkout
> `deps/pokemon-showdown/dist` already exists as a real directory, so `ln -s TARGET dist` puts the
> link *inside* it as `dist/dist` → pointing at its own parent, `node build` dies with `ELOOP`, and
> every websocket-server path stops working. That happened here and went unnoticed for four weeks.
> Never symlink the whole `deps/pokemon-showdown` directory either — git then treats the submodule
> path as a symlink and `git status` breaks.

Long training runs use `python -m main.launcher`, which creates its **own** git worktree pinned
to the launch commit — so pushing to `main` never disturbs a run in flight.

---

## Conventions worth knowing before you write code

- **Documentation is part of the change, not a follow-up.** Every `CLAUDE.md`, every
  `README.md`, and `designs/ARCHITECTURE.md` are always-current: if your change makes one stale,
  fix it in the same commit. So are the `designs/` trees that hold detail lifted out of a
  `CLAUDE.md` — `designs/ops/`, `designs/training/`, `designs/rust_sim/port_build_log.md` — and the
  `designs/research_state/UNDERSTANDING.md` view. `designs/CHANGELOG.md` and
  `designs/research_state/ledger.md` are append-only history — add to them, never edit them.
  Everything else under `designs/` (`impl_step*.md`, `design_*.md`, `todo.md`, the historical
  `ai_vN/` chapters, and `research_state/claude_md_archive/`) is explicit-only; leave it alone
  unless asked.
- **The `CLAUDE.md` beside the code is the real documentation.** The root one is a
  constitution, a command card and a map, and nothing else; the leaf in `src/agents/model/`,
  `src/agents/gen3_data/`, `src/agents/observation/`, `src/agents/battle/`,
  `src/agents/training/`, `src/rust_sim/`, `src/main/launcher/`, `src/main/prober/`,
  `src/main/prober/web/`, `src/main/tui/` and `designs/` carries the detail. Read the leaf for the
  area you are touching.
- **Never hardcode an observation index.** Every offset comes from named constants in
  `agents/observation/constants.py`; read `Gen3ObservationEncoder.get_layout()`.
- **Never hardcode a path, and never hand-roll `Path(__file__).parents[N]`.** Use
  `src/utils/paths.py`: `repo_path(...)` / `src_path(...)` for anything in the tree, and
  `main_models_dir()` for the `models/` run archive — **not committed**, so on your clone it is
  absent and the tests that need it *skip*, which is expected (set `$GEN3AI_MODELS_DIR` if you have
  an archive elsewhere). `src/utils/paths_test.py` AST-scans the tree and fails any absolute
  `/home/…` used as a value.
- **Architecture constants live in exactly one file**: `src/agents/model/arch_constants.py`.
- **`data/` is the source of truth.** The runtime reads only `data/`, through the
  `agents.gen3_data` facade — never live from an upstream. `tools/` is the only layer that knows
  the upstreams.
- **Priors are Smogon-derived, only.** Anything the network reads as a prior traces to Smogon
  usage data, ground-truth labels or ladder replays; the training team pool may MEASURE structure
  but never ships as a prior.
- **Hand the network FACTS, not judgments.** A hand-computed feature is the probability or size of
  a game event in its own unit (an HP fraction, a probability, turns, hazard layers). A weighted mix
  of units, an assumption about our own later plan, or a good/bad threshold is a judgment, and the
  network's job. The test and the inventory:
  [`designs/endstate/design_hand_computed_features.md`](designs/endstate/design_hand_computed_features.md) §1.
- **Any change under `src/agents/observation/` must run the encoder benchmark** before and
  after (`python -m agents.observation.rust_encoder_benchmark`) — the gate is in that
  package's `CLAUDE.md`.
- **An edge case you fixed gets a named regression test** that fails if the fix is reverted.
- **Claims carry their measurements.** "This is faster" is not a result; "1.41× at `--n-envs 48`,
  measured on an idle box" is. Retractions are recorded as retractions rather than quietly
  deleted — see `designs/research_state/`.

---

## Before you push

```bash
pytest src/ -m "not slow and not e2e" -q -n 6     # the routine gate
```

If something unrelated is red, re-run it on a clean checkout of `main` before blaming your change,
and say so in the PR. A timeout on a busy machine is not a semantic failure: bounds scale with
measured contention, but a duration measured under starvation is not a measurement.

---

## Getting help

Open an issue. Questions, ADV theory arguments, "the bot's play is wrong here" reports, and "I don't
understand this subsystem" are all welcome and all useful — the last one especially, because it
usually means the documentation is wrong.
