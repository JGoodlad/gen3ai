# Contributing to Gen3AI

Contributions of any size are genuinely wanted — an idea, a question, a single test, or a
subsystem. This page is the mechanical part: how to get a working checkout, what to run before
you push, and the handful of local conventions that are not obvious from the code.

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
| 2 | create/update the `gen3ai_stable` conda env from `environment.yml` | ~5-15 min fresh (≈2 GB of wheels) |
| 3 | `pip install -e .` — puts `src/` on the import path for good | ~2 s |
| 4 | `git submodule update --init` — the Pokémon Showdown reference engine | ~30 s |
| 5 | the Showdown build artifacts (`npm ci` + `node build`) — **or** worktree symlinks | ~3-6 min fresh |
| 6 | *(optional)* `cargo build --release` for the Rust simulator | ~3-10 min cold |
| 7 | verify — the ruff and mypy gates, the two import-precedence gates, a ~10 s unit smoke | ~30-60 s |

Useful flags: `--dry-run` (print the plan, change nothing), `--with-rust` / `--no-rust`,
`--force` (redo the conda step), `--no-check`, `--help`.

**Say yes to the Rust build if you have ten minutes.** Training defaults to `--use-bridge rust`,
and the first Rust-backed test builds those binaries *anyway* — mid-test, saturating every core, a
documented cause of spurious timeout failures on a fresh checkout.

### After bootstrap: activate, and that is it

```bash
conda activate gen3ai_stable
python -c "import agents, poke_env; print('ok')"
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

**`pyproject.toml` declares no dependencies, on purpose.** `environment.yml` is the single owner
of what is installed — including a CUDA-local-version torch that PyPI does not carry — so
`pip install -e .` writes a `.pth` and a `dist-info` and cannot resolve, upgrade or replace anything
in your environment. One owner per question.

### CPU-only machines

`environment.yml` installs CUDA builds of torch, but nothing is CUDA-gated at import time — a
CPU-only box runs the entire test suite, the prober, and `--device cpu` training; it just downloads
~2 GB of wheels it will never use. The `environment.yml` comment block explains how to derive a CPU
variant (extra index at `.../whl/cpu`, drop the `+cu121` suffixes and the `nvidia-*-cu12` / `triton`
pins). We ship no such file because nobody here runs one; a good one is a welcome PR.

### Do not install `poke-env` from PyPI

This repo **vendors a fork** of poke-env at `src/poke_env/` and that fork is authoritative — it
carries modules upstream does not, and the battle layer depends on them. Installing the PyPI package
alongside it creates two importable `poke_env` packages whose winner is decided by `sys.path` order,
and **the failure is silent**: upstream imports cleanly and behaves subtly differently.
`src/poke_env_fork_gate_test.py` guards this permanently. To check by hand:

```bash
python -c "import poke_env; print(poke_env.__file__)"   # must print a path under this repo's src/
```

---

## Tests — what to run, and when

Two **orthogonal** marker axes. A *capability* marker says what a test **needs**; the single
*cost* marker `slow` says what it **costs**. Cut on cost, never on capability:

| When | Command | ~Time |
|---|---|---|
| Inner loop — fastest true/false | `pytest src/ -m "not slow and not e2e and not sim and not integration" -q -n 2` | ~1.5 min |
| **The routine gate — before any commit** | `pytest src/ -m "not slow and not e2e" -q -n 2` | ~4 min |
| Everything — before a release | `pytest src/ -q` | ~31 min |

Those durations were measured on an idle box on 2026-08-14 and the tree has grown since; treat them
as an order of magnitude, not a budget. The counts are current: 10,186 tests collected in all,
10,105 in the routine gate, 9,979 in the inner loop.

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

**Eight static gates run inside the suite** (there is no CI on this box, so a check outside the
suite is a check that rots). All are unmarked, so they run in every tier, and all are ~free:

| Gate | Checks | Opt-out |
|---|---|---|
| `src/agents/model/mypy_gate_test.py` | `python -m mypy`, scope from `mypy.ini`'s `files =` (`src/agents/model`, `src/agents/observation`) | `GEN3AI_SKIP_MYPY_GATE=1` |
| `src/ruff_gate_test.py` | pyflakes over `src/agents`, `src/main`, `src/utils` — `--select F,E9`, wrongness only, never style | `GEN3AI_SKIP_RUFF_GATE=1` |
| `src/file_size_gate_test.py` | a file over 2,000 lines fails; 1,000-2,000 is reported. The allowlist is empty — decompose instead | `GEN3AI_SKIP_SIZE_GATE=1` |
| `src/claude_md_freshness_gate_test.py` | every repo-relative path and every `--flag` a `CLAUDE.md` names still resolves | `GEN3AI_SKIP_CLAUDE_MD_GATE=1` |
| `src/test_stub_vacuity_gate_test.py` | every `monkeypatch.setattr` / `patch` target is a symbol the code under test actually reads — a stub that stubs nothing fails | `GEN3AI_SKIP_STUB_GATE=1` |
| `src/slow_tier_status_gate_test.py` | the last recorded verdict of every `slow` test — a recorded failure fails the routine gate that deselected it | `GEN3AI_SKIP_SLOW_STATUS_GATE=1` |
| `src/mode_flag_doc_gate_test.py` | every mode-flag value `designs/ARCHITECTURE.md` states in prose equals `designs/production_config.json` | `GEN3AI_SKIP_MODE_FLAG_DOC_GATE=1` |
| `src/ledger_index_gate_test.py` | `designs/research_state/ledger_index.md` matches what `python -m main.ledger_index` renders | `GEN3AI_SKIP_LEDGER_INDEX_GATE=1` |

Two more guard the import path itself — `src/packaging_gate_test.py` (`PYTHONPATH` still outranks
the editable install) and `src/poke_env_fork_gate_test.py` (the vendored fork still wins
`import poke_env`). A missing linter **fails** rather than skips: a linter that silently opts out
reads exactly like a linter that found nothing.

**Fuzz tests are not parametrized unit tests.** In this repo a `*_fuzz_test.py` plays *real
battles* through the in-process bridge and validates observations against the actual protocol
stream. Run them directly as scripts:

```bash
python src/agents/action/fuzz_test.py 50
```

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

Killing :8001 drops every poke-env websocket at once and crashes a training run that may have been
going for days. `npm run stop` with no argument kills :8000 — never run a blanket `node`/`showdown`
kill, and only ever stop the port you personally started. Most work needs no server at all:
training and evaluation default to `--use-bridge rust`, an in-process reimplementation of the Gen 3
engine. Prefer it for throwaway work.

```bash
npm run showdown -- 9001     # your own server, if you really need one
npm run stop -- 9001         # and only ever this one
```

---

## Git workflow — never commit on `main`

There are no pull-request gates on this repo, but **`main` must never be dirty**. All edits and
commits happen in a branch or a git worktree, and land on `main` by push:

```bash
git worktree add ../gen3ai-myfeature -b myfeature
cd ../gen3ai-myfeature
./scripts/bootstrap.sh          # detects the worktree; symlinks instead of rebuilding
export PYTHONPATH=$PYTHONPATH:src   # MANDATORY here — see below
# ... work, test ...
bash scripts/land.sh myfeature ../gen3ai-myfeature
```

`scripts/land.sh` is the landing path, and it is four steps in a fixed order: run the static
gates **inside the worktree**, `git push origin myfeature:main` from the main checkout,
`git pull --ff-only` there so main is not left behind its own remote, then remove the worktree and
the branch. Any gate failure exits without pushing. It never force-pushes — a rejected
non-fast-forward means someone landed first, so rebase the worktree on `main`, resolve, and run it
again. Run the routine gate yourself before you call it; the script gates statics, not the suite.

🚨 **The export is mandatory in a worktree, and it is the one thing that fails silently.** A `.pth`
entry cannot know which worktree you are standing in, so `pytest` run here with no `PYTHONPATH`
collects *this* tree's test files while importing *main's* code — every result is then about a tree
you did not edit. `bootstrap.sh` therefore **skips** the install step in a worktree rather than
pointing the `.pth` at a directory that will later be deleted, and `src/packaging_gate_test.py`
fails loudly with exactly this diagnosis. Export first anyway.

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
  `agents.gen3_data` facade — never live from poke-env. `tools/` is the only layer that knows
  the upstreams.
- **Any change under `src/agents/observation/` must run the obs-build benchmark** before and
  after (`src/agents/training/obs_build_benchmark.py`) — the gate and its baseline are in that
  package's `CLAUDE.md`.
- **An edge case you fixed gets a named regression test** that fails if the fix is reverted.
- **Claims carry their measurements.** "This is faster" is not a result; "1.41× at `--n-envs 48`,
  measured on an idle box" is. Retractions are recorded as retractions rather than quietly
  deleted — see `designs/research_state/`.

---

## Before you push

```bash
pytest src/ -m "not slow and not e2e" -q -n 2     # the routine gate
```

If something unrelated is red, `git stash` and re-run before blaming your change — this box
often has a training run on it, and a duration measured under starvation is not a measurement.

---

## Getting help

Open an issue. Questions, ADV theory arguments, "the bot's play is wrong here" reports, and "I don't
understand this subsystem" are all welcome and all useful — the last one especially, because it
usually means the documentation is wrong.
