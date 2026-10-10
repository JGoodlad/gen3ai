# Developing Gen3AI: the detailed reference

This is the reference behind [`CONTRIBUTING.md`](../CONTRIBUTING.md). You don't need to read it
front to back: come here when something in setup, the tests or the repo layout surprises you.
Training and evaluation have their own guide, [`RUNNING.md`](RUNNING.md).

- [Setup in depth](#setup-in-depth)
- [Tests](#tests)
- [Ports and the Showdown server](#ports-and-the-showdown-server)
- [Git worktrees and the maintainer's workflow](#git-worktrees-and-the-maintainers-workflow)
- [Conventions worth knowing before you write code](#conventions-worth-knowing-before-you-write-code)

---

## Setup in depth

`./scripts/bootstrap.sh` does everything. It is idempotent (re-run it any time; completed steps are
skipped), it fails loudly, and it announces what each step costs before spending it:

| Step | What | Cost |
|---|---|---|
| 1 | prerequisite check: `git`, `conda`, `node`, `npm` | instant |
| 2 | create/update the `gen3ai_torch28` conda env (torch 2.8) from `environment_torch28.yml`; a worktree never updates it silently (see below) | ~5-15 min fresh (about 2 GB of wheels) |
| 3 | `pip install -e .`, which puts `src/` on the import path for good | ~2 s |
| 4 | `git submodule update --init`: the Pokémon Showdown reference engine | ~30 s |
| 5 | the Showdown build artifacts (`npm ci` + `node build`), or worktree symlinks | ~3-6 min fresh |
| 6 | *(optional)* `cargo build --release` for the Rust simulator | ~3-10 min cold |
| 7 | verify: the ruff and mypy gates, the two import-precedence gates, a ~10 s unit smoke | ~30-60 s |

Useful flags: `--dry-run` (print the plan, change nothing), `--with-rust` / `--no-rust`,
`--force` (redo the conda step), `--update-shared-env`, `--skip-env`, `--no-check`, `--help`.

**Say yes to the Rust build if you have ten minutes.** Training runs on the in-process Rust bridge,
and the first Rust-backed test builds those binaries anyway: mid-test, saturating every core, which
is a known cause of spurious timeout failures on a fresh checkout.

### After bootstrap

```bash
conda activate gen3ai_torch28
python -c "import agents, main, utils; print('ok')"
```

In the main checkout no `export PYTHONPATH` is needed: step 3's editable install puts this
checkout's `src/` on the import path from any directory, shell, IDE or debugger. Without the
install (a container, a machine that skipped the bootstrap) `export PYTHONPATH=$PYTHONPATH:src` is
exactly equivalent, and harmless when the install exists. **In a git worktree the export is
required**; see [below](#git-worktrees-and-the-maintainers-workflow).

**Install from the main checkout, never from a git worktree.** An editable install records one
absolute path in a `.pth`. Install from a worktree, delete the worktree, and Python skips the
missing entry in silence, so imports start failing for a reason the install never reports.
`src/packaging_gate_test.py` catches a stale one and prints the fix.

**`pyproject.toml` declares no dependencies, on purpose.** The env file (`environment_torch28.yml`)
is the single owner of what is installed, including a CUDA-local-version torch that PyPI does not
carry, so `pip install -e .` cannot resolve, upgrade or replace anything in your environment.

### The conda env is shared; a checkout is not

Every run, gate and agent on a box uses the one `gen3ai_torch28` env, and a `conda env update
--prune` under a live process can swap a package out from under it. So step 2's "env is current"
stamp lives in the git **common** dir (one per clone, keyed by the sha256 of
`environment_torch28.yml`):

| run from | `environment_torch28.yml` matches the stamp | it differs (or `--force`) |
|---|---|---|
| the main checkout | nothing to do | updates the env in place, and warns that it is shared |
| a linked worktree | nothing to do | prints the diff and refuses (exit 3) |

A worktree refusal names both ways forward: `--update-shared-env` (update it anyway, only when no
live or pinned run uses it), or `--skip-env` (leave the env alone and finish the rest of the setup).
A missing env is simply created, from anywhere.

**The legacy env.** `gen3ai_stable` (torch 2.5.1, `environment.yml`) is kept frozen only so a run
trained on it resumes on it: the launcher reads the run's recorded torch (`metadata.json`
`torch_version`; none = 2.5.1) and selects that env, or refuses (`src/main/launcher/torch_runtime.py`).
Bootstrap never touches it; never edit `environment.yml`. On a fresh machine you don't need it.

### CPU-only machines

`environment_torch28.yml` installs CUDA builds of torch, but nothing is CUDA-gated at import time: a
CPU-only box runs the entire test suite, the prober, and `--device cpu` training; it just downloads
~2 GB of wheels it never uses. A CPU variant is derived the usual way (extra index at
`.../whl/cpu`, drop the `+cu126` suffixes, the `nvidia-*-cu12` / `triton` pins and the `cuda-*`
header packages). We ship no such file because nobody here runs one; a good one is a welcome PR.

### Don't install `poke-env`

poke-env is retired (finished 2026-10-08): every battle, row, eval game and live game is the Rust
core's, and nothing in `src/` / `tools/` / `scripts/` imports `poke_env`.
`src/poke_env_absent_gate_test.py` keeps it out (no package in our tree, none in site-packages,
neither env file installing one). `python -c "import poke_env"` must raise `ModuleNotFoundError`.

---

## Tests

Two independent marker axes. A *capability* marker says what a test **needs**; the single *cost*
marker `slow` says what it **costs**. Cut on cost, never on capability:

| When | Command | ~Time |
|---|---|---|
| Inner loop: fastest true/false | `pytest src/ -m "not slow and not e2e and not sim and not integration" -q -n 2` | ~1.5 min |
| **The routine gate: before any commit** | `pytest src/ -m "not slow and not e2e" -q -n 6` | ~4.3 min (one at a time; `-n 4` beside a training run) |
| Everything: before a release | `pytest src/ -q` | ~47 min |

The routine gate's ~4.3 min at `-n 6` was measured on a quiet box on 2026-10-01
([`designs/ops/testing.md`](../designs/ops/testing.md)); treat the other two as an order of
magnitude, not a budget. Counts collected 2026-10-09: 11,193 tests in all, 11,144 in the routine
gate, 10,633 in the inner loop. `-n` is `pytest-xdist`; run serial when you need `-s` or a debugger.

| Marker | Means |
|---|---|
| *(unmarked)* | pure in-process; needs nothing |
| `integration` | an out-of-process dependency, but no battles and no browser |
| `sim` | plays real battles in-process through the bridge (no server) |
| `browser` | needs headless chrome |
| `e2e` | needs a live Showdown server |
| **`slow`** | minutes, not seconds: the one marker that decides routine cost |
| `benchmark` | a measurement, not a pass/fail gate; its bounds are never scaled |

Don't use the old `-m "not integration and not e2e"` cut. `integration` now spans a ~100x cost
range, so excluding it throws away cheap, high-value coverage; that exact cut is how a six-battle
golden test rode `main` red three separate times.

### The static gates

**Seventeen static gates run inside the suite** (there is no CI on this box, so a check outside the
suite is a check that rots). All declare the `static` tier, so they run in every tier, and all are
~free warm. A missing linter **fails** rather than skips: a linter that silently opts out reads
exactly like one that found nothing.

| Gate | Checks | Opt-out |
|---|---|---|
| `src/agents/model/mypy_gate_test.py` | `python -m mypy`, scope from `mypy.ini`'s `files =` (`src/agents/model`, `src/agents/observation`) | `GEN3AI_SKIP_MYPY_GATE=1` |
| `src/ruff_gate_test.py` | pyflakes over `src/agents`, `src/main`, `src/utils`: `--select F,E9`, wrongness only, never style | `GEN3AI_SKIP_RUFF_GATE=1` |
| `src/file_size_gate_test.py` | a file over 2,000 lines fails; 1,000-2,000 is reported. The allowlist is empty: decompose instead | `GEN3AI_SKIP_SIZE_GATE=1` |
| `src/claude_md_freshness_gate_test.py` | every repo-relative path and every `--flag` a `CLAUDE.md` names still resolves | `GEN3AI_SKIP_CLAUDE_MD_GATE=1` |
| `src/test_stub_vacuity_gate_test.py` | every `monkeypatch.setattr` / `patch` target is a symbol the code under test actually reads; a stub that stubs nothing fails | `GEN3AI_SKIP_STUB_GATE=1` |
| `src/slow_tier_status_gate_test.py` | the last recorded verdict of every `slow` test; a recorded failure fails the routine gate that deselected it | `GEN3AI_SKIP_SLOW_STATUS_GATE=1` |
| `src/mode_flag_doc_gate_test.py` | every mode-flag value `designs/ARCHITECTURE.md` states in prose equals `designs/production_config.json` | `GEN3AI_SKIP_MODE_FLAG_DOC_GATE=1` |
| `src/recipe_doc_gate_test.py` | every training-recipe value `designs/endstate/design_learner_recipe.md` states equals the `recipe.fresh` / `recipe.fork` blocks of `designs/production_config.json` | `GEN3AI_SKIP_RECIPE_DOC_GATE=1` |
| `src/trace_summary_reader_gate_test.py` | no module but `main/prober/core_trace.py` opens an eval-trace `*_summary.json`; every reader goes through its loaders | `GEN3AI_SKIP_SUMMARY_READER_GATE=1` |
| `src/ledger_index_gate_test.py` | `designs/research_state/ledger_index.md` matches what `python -m main.ledger_index` renders | `GEN3AI_SKIP_LEDGER_INDEX_GATE=1` |
| `src/enum_str_compare_gate_test.py` | no enum `agents.enums` defines is compared to a string (mypy `--strict-equality`, typed): the bug that kept four bots from ever setting up | `GEN3AI_SKIP_ENUM_STR_GATE=1` |
| `src/poke_env_import_gate_test.py` | no file imports `poke_env` unless it is on the generated, shrink-only `designs/ops/poke_env_import_allowlist.txt` (empty since 2026-10-08) | `GEN3AI_SKIP_POKE_ENV_IMPORT_GATE=1` |
| `src/poke_env_absent_gate_test.py` | there is no `poke_env` package in our tree, in this interpreter's site-packages, or in either env file | none (static, ~free) |
| `src/learner_lifecycle_gate_test.py` | no training-step path constructs an optimizer, an `nn.Parameter` or an `nn.Module` (nor a CUDA stream / graph / pool) outside a declared startup builder or a class's `__init__` / `_build` / `_setup_model`; the allowlist is empty | `GEN3AI_SKIP_LIFECYCLE_GATE=1` |
| `src/global_rng_seed_gate_test.py` | no non-test module binds a process-global seeding function by name or takes one as a value (the runtime reseed guard cannot see a pre-bound name); the allowlist is empty | `GEN3AI_SKIP_GLOBAL_RNG_GATE=1` |
| `src/strict_checkpoint_load_gate_test.py` | no non-test module loads a PPO checkpoint through sb3's non-strict retry (`MaskablePPO.load` / `PPO.load`, `set_parameters(…, exact_match=False)`); every reader uses `agents.model.snapshot.load_checkpoint_strict` | `GEN3AI_SKIP_STRICT_LOAD_GATE=1` |
| `src/eval_ledger_reader_gate_test.py` | every reader of the eval count ledger goes through `eval_ledger.read` / `read_by_regime` with a fully spelled-out `ReaderDecl`, and the ledger's closed lists equal `design_evaluation.md` §0b.2's table | `GEN3AI_SKIP_LEDGER_READER_GATE=1` |

One more test guards the import path itself and is not a static-tier gate:
`src/packaging_gate_test.py` (`PYTHONPATH` still outranks the editable install). The count above is
held to the `pytest.mark.static` declarations by `src/utils/static_gates_test.py`; list them with
`python -m utils.static_gates`.

### Fuzzers and benchmarks

**Fuzz tests are not parametrized unit tests.** Here a fuzzer plays *real battles* and checks them
against ground truth. The main one is the Rust engine's A/B differential fuzzer: it drives the real
Showdown sim and the Rust port side by side and saves a standalone repro for every divergence
(`src/rust_sim/CLAUDE.md`, "A/B fuzzer"):

```bash
node src/rust_sim/harness/ab_fuzz.js --mode random --battles 200
```

Every divergence it finds becomes a named deterministic pin
([`designs/rust_sim/regression_pins.md`](../designs/rust_sim/regression_pins.md)). A
pytest-collected test, by contrast, takes seeded battles from the Rust core, so it plays the same
battle every run.

**Benchmarks warn instead of scaling.** Wall-clock bounds in tests scale by measured CPU contention
(`src/utils/contention.py`), because this box usually carries a live training run and a timeout is
never a semantic outcome. A benchmark's output *is* the measurement, so it prints a "THE BOX IS
BUSY" banner rather than stretching. Say whether the box was quiet when you report one.

If something unrelated is red, re-run it on a clean checkout of `main` before blaming your change,
and say so in the PR.

---

## Ports and the Showdown server

Most work needs no server: training and evaluation run on the in-process Rust bridge. If you do
start one, use your own port:

| Port | Owner |
|---|---|
| 8000 | development |
| 8001 | the standing training server on the maintainer's box: never stop, restart or kill it |
| 9XXX | pick one for anything ephemeral you start |

```bash
npm run showdown -- 9001     # your own server, if you really need one
npm run stop -- 9001         # and only ever stop the one you started
```

`npm run stop` with no argument stops :8000. Never run a blanket `node` / `showdown` kill.

---

## Git worktrees and the maintainer's workflow

*Outside contributors fork and open a pull request ([`CONTRIBUTING.md`](../CONTRIBUTING.md)); the
worktree notes still help if you keep several branches checked out.* The maintainer works without
pull requests, but `main` is never dirty: all edits and commits happen in a git worktree, and land
on `main` by push:

```bash
git worktree add ../gen3ai-myfeature -b myfeature
cd ../gen3ai-myfeature
./scripts/bootstrap.sh              # detects the worktree; symlinks instead of rebuilding
export PYTHONPATH=$PYTHONPATH:src   # required here, see below
# ... work, test ...
bash scripts/land.sh myfeature ../gen3ai-myfeature
```

`scripts/land.sh` runs four steps in a fixed order: the static gates **inside the worktree**,
`git push origin myfeature:main` from the main checkout, `git pull --ff-only` there, then removal of
the worktree and branch. The removal is skipped (exit 3, code still landed) when the run-data guard
(`python -m utils.worktree_guard`) finds a main-checkout `models/` symlink into the worktree or more
than 50 MiB of untracked content in it. Any gate failure exits without pushing, and it never
force-pushes: a rejected non-fast-forward means someone landed first, so rebase and run it again.
It gates statics, not the suite, so run the routine gate yourself first.

**Why the export matters in a worktree.** The editable install names one absolute path, the main
checkout's `src/`, so a `python <script>` run in a worktree with no `PYTHONPATH` imports *main's*
code, and every result is about a tree you didn't edit. `pytest` is covered for you: the root
`conftest.py` puts this checkout's `src/` first for the session, its workers and every subprocess a
test spawns. `bootstrap.sh` skips the install step in a worktree.

**Worktree build artifacts.** A fresh worktree gets an empty submodule directory and no build
artifacts; `bootstrap.sh` symlinks `dist/` and `node_modules/` from the main checkout. If you do it
by hand, guard each link with `[ -e ]` and run it only in a worktree: in the main checkout
`deps/pokemon-showdown/dist` is a real directory, so `ln -s TARGET dist` creates `dist/dist`
pointing at its own parent and `node build` dies with `ELOOP`. Never symlink the whole
`deps/pokemon-showdown` directory either; git then treats the submodule path as a symlink.

Long training runs use `python -m main.launcher`, which creates its own git worktree pinned to the
launch commit, so pushing to `main` never disturbs a run in flight.

---

## Conventions worth knowing before you write code

- **Documentation is part of the change.** Every `CLAUDE.md`, every `README.md`, and
  `designs/ARCHITECTURE.md` are always-current: if your change makes one stale, fix it in the same
  commit. So are the `designs/` trees that hold detail lifted out of a `CLAUDE.md` (`designs/ops/`,
  `designs/training/`, `designs/model/`, `designs/rust_sim/`, `designs/prober/`), the
  `designs/endstate/` specs and `designs/research_state/UNDERSTANDING.md`.
  `designs/CHANGELOG.md` and `designs/research_state/ledger.md` are append-only history: add to
  them, never edit them. Everything else under `designs/` is explicit-only; leave it alone unless
  asked.
- **The `CLAUDE.md` beside the code is the detailed documentation** for that area, written for
  humans and AI agents alike. The root one is the agents' rulebook and map.
- **Never hardcode an observation index.** Every offset comes from named constants in
  `agents/observation/constants.py`; read `Gen3ObservationEncoder.get_layout()`.
- **Never hardcode a path, or hand-roll `Path(__file__).parents[N]`.** Use `src/utils/paths.py`:
  `repo_path(...)` / `src_path(...)` for anything in the tree, and `main_models_dir()` for the
  `models/` run archive. `models/` is not committed, so on your clone it is absent and the tests
  that need it skip, which is expected (set `$GEN3AI_MODELS_DIR` if you have an archive elsewhere).
- **Architecture constants live in exactly one file**: `src/agents/model/arch_constants.py`.
- **`data/` is the source of truth.** The runtime reads only `data/`, through the
  `agents.gen3_data` facade; `tools/` is the only layer that knows the upstream sources.
- **Priors are Smogon-derived, only.** Anything the network reads as a prior traces to Smogon
  usage data, ground-truth labels or ladder replays; the training team pool may measure structure
  but never ships as a prior.
- **Hand the network facts, not judgments.** A hand-computed feature is the probability or size of
  a game event in its own unit (an HP fraction, a probability, turns, hazard layers). A weighted mix
  of units or a good/bad threshold is a judgment, and the network's job
  ([`design_hand_computed_features.md`](../designs/endstate/design_hand_computed_features.md) §1).
- **Any change under `src/agents/observation/` runs the encoder benchmark** before and after
  (`python -m agents.observation.rust_encoder_benchmark`).
- **Claims carry their measurements.** "This is faster" is not a result; "1.41x at `--n-envs 48`,
  measured on an idle box" is. Retractions are recorded as retractions, not quietly deleted.
