# CLAUDE.md — Gen3AI Project Guide

## Development Stage

**Rapid iteration — checkpoint compatibility is not a concern.** Breaking changes to the observation space, network architecture, or action space are fine. Do not add backwards-compatibility shims or hesitate to change dims, layer sizes, or layouts.

Architecture constants (embedding dims, layer sizes, etc.) are defined as module-level constants in `src/agents/model/arch_constants.py` — that is the single source of truth (`features_extractor.py` re-exports the whole block, so historical import paths still resolve). When you change one, change it there and nowhere else. See [Model Versioning](#model-versioning).

**Before reasoning about the model, read [`designs/ARCHITECTURE.md`](designs/ARCHITECTURE.md).** It is the only document that states the architecture *as it is now* — obs layout, phase chain, what each head consumes, the `DamageOperator` block, the edge families, and a flag table marking which nominally-set flags are actually `INERT`. Version-numbered narrative lives in [`designs/CHANGELOG.md`](designs/CHANGELOG.md) and describes the past, not the present.

## 🚨 CRITICAL — running subagents / Workflows on this box

**Two rules. Both are required; either alone fails.** Measured 2026-08-09: two workflows returned
**0 results out of 5 agents and 0 out of 4** — 5.6M subagent tokens across three attempts, most of
it wasted — before this was understood.

**1. Pass `stallMs: 900_000` on EVERY `agent()` call in a Workflow script.**
Workflow subagents have their own stall watchdog **hardcoded** in the Claude Code binary
(`K2b=180000, B1p=5` — 3 minutes, 5 retries, verified by string-inspecting v2.1.226). There is **no
env var and no `settings.json` key**; the runner reads only a per-call override
(`ye = _e?.stallMs != null ? Number(_e.stallMs) : K2b`). `stallMs` is Workflow-only — it appears in
no tool input schema, so the `Agent` tool rejects it.
⚠️ **`CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS` does NOT cover Workflow** — it feeds the Agent-tool path
(`env || 600000`) and IS live (1800000). Two different watchdogs; that is why a `180000` in a
workflow error looks impossible.

**2. Cap concurrency to 1–2 agents. This is the actual cause.**
`stallMs` only raises the kill threshold; it creates no stream capacity. The stalls are **API
stream starvation** when ≥3 LLM streams are live on the account (interactive sessions + agents).
Signature: an agent transcript ending on a **`user`-role tool result with no assistant turn after**
— the tool returned, the next model stream produced nothing. Long/quiet tools are innocent, and
"emit periodic output" instructions do not help. Even with `stallMs`, fanning out 6 verifiers at
once still cost ~3.5 attempts per agent, and **a workflow retry restarts the agent from scratch**
(4 attempts = 4× tokens). Plain `Agent` calls beat Workflow fan-out on a busy account, because a
stalled `Agent` can be **resumed** with `SendMessage` to its agentId rather than redone.

**3. Never let a script report agent ERRORS as "no findings".** `parallel()` returns `null` for a
failed agent, so `findings.length === 0` is ambiguous — track failures and return a distinct
status. Diagnose from `journal.jsonl` in the transcript dir; the field is **`result`**, not
`value`, and many `started` lines with zero `result` lines is this failure's signature.

---

## Documentation Maintenance

Keep docs in sync **automatically, as part of the same change** — no need to be asked:

- **Every `CLAUDE.md`** (root, and the directory leaves — `src/agents/model/`, `src/agents/observation/`, `src/agents/battle/`, `src/agents/training/`, `src/main/launcher/`, `designs/`, anywhere): always current. If a change makes one stale, fix it in the same pass.
- **`designs/ARCHITECTURE.md`**: always current, and it is the **first** thing an architecture change updates. It states only what is true now — no version numbers in prose, every measured figure carrying its provenance, every unverifiable claim marked `**UNVERIFIED:**`. Never narrate a change inline; state the new truth and delete the old.
- **`designs/CHANGELOG.md`**: append-only history. Add the new version entry here in the same pass. Never edit or "correct" an existing entry — its job is to record what was believed at the time.
- **Every `README.md`**: always current. When you change dims, layout, obs/architecture, or anything a README documents, update it without being prompted. **Exception:** `designs/ai_v3/README.md` is a **frozen ai_v3 historical** digraph — do NOT update it for current-arch changes; the live architecture is `designs/ARCHITECTURE.md` + `src/agents/model/CLAUDE.md`.

**Do NOT auto-update other docs under `designs/`** — `impl_step*.md`, `design_*.md`, `todo.md`, etc. are explicit-only. Touch them only when the user asks (directly or via `/gen3ai-update-design-docs`). The lone exception is `CLAUDE.md` files inside `designs/`, which follow the always-current rule above.

**Leaf CLAUDE.md map** — when working in one of these areas, read its leaf for the detail this root only summarises:

| Directory | Leaf covers |
|---|---|
| `src/agents/model/` | Feature-extractor phase contract, dual-head policy, architecture-constant rules, model versioning |
| `src/agents/gen3_data/` | The data facade: concept modules over `data/`, the acquisition-vs-access split, how it threads into the encoders |
| `src/agents/observation/` | Obs-build performance gate (mandatory benchmark) + the full per-block obs-vector layout |
| `src/agents/battle/` | Event-sourced battle layer (Gen3Battle, BattleEvent log, LiveView/TurnView/LegalActions, StrictBattleView, TurnDelta fold) |
| `src/agents/training/` | Bot-eval subprocess architecture + Showdown-port (`server_config`) threading |
| `src/main/launcher/` | Launcher internals: restarts, crash reporting, exit codes, flags, port default |
| `src/main/prober/` | Forensic-replay inspector: the analysis ENGINE + the `ProbeSession` facade, the JSON CLI, and trace discovery. The human surface is `web/` |
| `src/main/prober/web/` | The prober's browser front end — FastAPI over `ProbeSession`, Jinja2+HTMX, vendored JS, the committed `openapi.json` contract |
| `src/main/tui/` | Thin shared Textual base (`Gen3App`, theme, `gradient_color`) — the LAUNCHER's UI (the prober's TUI is retired) |
| `designs/` | Which `ai_vN` folder is relevant; version map |

**Two non-`CLAUDE.md` docs carry the same always-current obligation:**

| File | Holds |
|---|---|
| `designs/ARCHITECTURE.md` | **What is true NOW** — obs layout, phase chain, per-head inputs, op block, edge families, the production flag table with `INERT` markings, training-only obs keys |
| `designs/CHANGELOG.md` | **How it got here** — the verbatim v16→v59 version narrative. History; do not quote as current. |

## Git Workflow

This is a personal project — no pull requests needed. Work is pushed directly to `main`, but **all edits and commits must happen in a worktree or branch, never on the main checkout itself.**

Main must never be in a dirty state. The `/gen3ai-ship` skill is the only mechanism that lands code on main — it commits in the worktree branch and pushes:

```bash
git push origin <worktree-branch>:main
```

Never `git add` or `git commit` from `/home/goodlad/dev/gen3ai` directly.

**NEVER run `git add`, `git commit`, or `git push` unless the user's current message explicitly contains `/gen3ai-ship`.** Completing a task, writing tests, or any other finishing signal is NOT permission to commit. This applies even when the task feels "done". Only `/gen3ai-ship` authorises a commit.

---

## Python Environment

The project uses a dedicated conda environment, **not** `deps/venv`. Always prefix commands with the correct interpreter and `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 <script>
```

The conda env is `gen3ai_stable`. `deps/venv` exists but is outdated — ignore it.

---

## Git Worktree Setup

When opening a new git worktree (e.g. via Claude Code), the `deps/pokemon-showdown` submodule directory is created but left empty. Two steps are required before training or running tests:

**Step 1 — initialize the submodule** (gets source files, fixes VS Code git integration):
```bash
git submodule update --init
```

**Step 2 — symlink the build artifacts** (the submodule checkout has no compiled `dist/` or installed `node_modules/`, but the main repo already has them):
```bash
ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown/dist \
      deps/pokemon-showdown/dist
ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown/node_modules \
      deps/pokemon-showdown/node_modules
```

Both `dist/` and `node_modules/` are gitignored in pokemon-showdown, so these symlinks don't affect `git status`. Without step 2, training fails with `Cannot find module '.../dist/sim/index.js'`.

Do **not** symlink the entire `deps/pokemon-showdown` directory — git treats the submodule path as a symlink rather than a real checkout, which breaks `git status` and VS Code's git integration.

---

## Running Tests

### Running beside a live training run (`gen3_contention_robust_timeouts_v1`)

**The box normally carries a production run**, so any wall-clock timeout measures the spare
capacity as much as the code. This voided three separate investigations (the same tests passed in
isolation and passed on the main checkout; the only difference was a load average of 35 on 16
cores), and it produced one genuinely dangerous artifact: `bridge_impl_parity_test` counted a
per-battle TIMEOUT as an "unmodeled move" SKIP, so a starved run reported 39/40 bogus skips as a
**clean pass** that blamed the Rust port for the load average.

The bounds are now **scaled by measured CPU contention** (`src/utils/contention.py`): the
per-battle bridge timeout, the bridge-session battle-end timeout, and the poke-env `race_get`
silent-stall watchdog all read `scale_timeout(...)` at CALL time, where the factor is
`max(1, loadavg / len(sched_getaffinity))`, clamped to 12x. **On an idle box the factor is exactly
1.0, so nothing changes** — this is a no-op until the box is actually busy.

- **A timeout is never a semantic outcome.** It gets its own bucket, and a run whose timeouts
  exceed 25% of attempted battles is declared INCONCLUSIVE rather than reported.
- **Every timeout message self-diagnoses** (`describe_contention()`): it prints the load average
  and the `ps -eo pcpu,pid,args --sort=-pcpu | head` command, so a starved failure ends the
  investigation instead of starting one.
- **`GEN3AI_TIMEOUT_SCALE=N`** forces the factor when you know the regime and don't want to wait
  out the load average's one-minute lag. **Run the suite under `GEN3AI_TIMEOUT_SCALE=6` after
  touching any of this** — it proves no test depends on a raw constant that the helper scales.
  That check caught two real ones: the hung-eval-cycle tests built a past timestamp from
  `_EVAL_CYCLE_TIMEOUT_SEC` directly, so they passed idle and failed loaded. (Both suites green:
  3978 passed at factor 1 and at factor 6.)
- **The eval cycle is the path MOST exposed to contention** — it deliberately runs concurrently
  with training, so it is under load 100% of the time, and the 30-min `_EVAL_CYCLE_TIMEOUT_SEC`
  hung-cycle bound now scales (`eval_cycle_timeout()`, shared by both callbacks). Firing early
  does not merely lose a cycle: `_abort_pending_cycle` kills the workers and collects **partial**
  results, which flow into `win_rate_vs_bots` (curriculum ramp), `win_rate_vs_pool` (promotion
  gate) and the ELO fit — and a truncated sample is whichever shards got scheduled, not a random
  subsample. The partial-coverage warning also no longer asserts "worker crash" as the cause,
  since an overrun-kill produces an identical shortfall.
- Prefer `ProgressDeadline` (bound the IDLE gap) over a total-duration cap wherever incremental
  progress is observable — contention stretches duration, but only a real wedge stops progress.
  A total-duration cap conflates the two by construction. Keep the total as an opt-in
  **livelock** backstop: `node_reject_bound_integration_test`'s pre-fix wedge emits `|error|`
  frames *forever*, so an idle-only bound would never expire there.
  **The per-battle bridge bound now works this way** (`gen3_battle_progress_deadline_v1`):
  `BattleStreamClient.feed` counts protocol chunks as the sign of life and
  `local_battle_runner._await_battle` bounds the gap between them, with `_PER_BATTLE_TIMEOUT`
  demoted to the livelock backstop. It was a duration cap until 2026-08-15, and the failure it
  produced is the canonical one: on a box saturated by a `cargo build --release`, the parity test
  scored **8 of 12 battles as timeouts plus a transport error and FAILED**, none of them wedged,
  all still emitting chunks — and passed warm with no code change. **Scaling does not rescue a
  cap**: the factor is `loadavg / cpus`, and the real slowdown of a starved subprocess is a
  multiple of it. *(Only the SEQUENTIAL path — under concurrency several battles share one client,
  so a lively neighbour would mask a wedged battle's silence.)*
- ⚠️ **A test that PRINTS a diagnostic must not print it in the format of a measurement.**
  `bridge_impl_parity_test`'s threshold unit test fed `timed_out=4, attempted=12` into the real
  warning function, so under `-s` it emitted `⚠️ small run: 4/12 battles TIMED OUT (33%)` into the
  same stream as the live series lines — indistinguishable from a starvation reading, and taken
  for one during this very investigation while every real series that run reported 0. Its label is
  now `SYNTHETIC unit-test sample`. Same family as the benchmark rule above: a number that cannot
  be told apart from a measurement will eventually be read as one.
- **Benchmarks get the OPPOSITE treatment — warn, never stretch.** A benchmark's output IS the
  measurement, so scaling its bounds just buys a confidently-reported wrong number. All five
  (`obs_build`, `trainer_turn`, `bridge_impl_throughput`, `bridge_heap_growth`,
  `bridge_vs_websocket_latency`) now call `warn_if_contended()` at entry and print a loud
  "THE BOX IS BUSY" banner. This is a recorded failure, not a hypothetical: a node-vs-rust
  throughput result (node 798 vs rust 427 fps) was measured on a saturated box and had to be
  superseded — **with the conclusion reversed** — and nothing in its output said so.

**Measured** (a since-deleted scratch script, 40 CPU burners → load ~47 on 16 cores, factor ~3.7, same
battles both arms): at a 2.0 s baseline, scaling OFF = **0 completed / 6 timed out**; scaling ON =
**6 completed / 0 timed out**. At a 4.0 s baseline both arms completed — the scaling matters
exactly when the bound sits within ~2x of the real battle duration, which is where a loaded box
puts you.

### Test tiers — TWO AXES, and keeping them apart is the point

A marker says **what a test NEEDS** (capability). A separate marker says **what it COSTS** (`slow`).
Collapsing those into one axis is what the old single `integration` marker did, and it failed in
**both** directions at once — read this before adding a marker to anything.

| Marker | Answers | Values |
|---|---|---|
| capability | *can this run here?* | *(unmarked)* · `integration` · `sim` · `browser` · `e2e` |
| **cost** | *should this run routinely?* | **`slow`** |

| Tier | Needs | Measured 2026-08-14 |
|---|---|---|
| *(unmarked)* | nothing — pure in-process | 4536 tests, **127 s** serial (~56 s at `-n 4`) |
| `integration` | an out-of-process dep, no battles, no browser | ~16 s total |
| `sim` | real battles in-process via the bridge, no server | 45 tests, ~100 s total |
| `browser` | headless chrome | 53 tests, **1426 s** — ALL of it also `slow` |
| `e2e` | a live Showdown server | run directly as scripts |
| `slow` | *(orthogonal)* minutes, not seconds | 71 tests |

**MEASURE BEFORE YOU TIER — the intuitive answer was wrong.** The assumption going in was that the
rust/node bridge parity was the expensive thing. It is not: the **browser** tests are **88% of the
entire integration tier** (1426 s of 1623 s), because `_probe` launches a FRESH headless chrome per
test at ~25 s of cold start, 53 times. Every bridge test *combined* is ~170 s. Tier by the profile
(`pytest -m <tier> --durations=0`), not by which subsystem feels riskiest.

**Cost tracks battle COUNT, not "does it battle".** The 6-battle obs-golden linchpin is 4.2 s; the
12-battle rust/node parity is 72.6 s and is the most contention-fragile test in the tree. So `sim`
cannot be the marker that decides routine cost — `slow` is. That distinction is not academic:
`gen3_data_obs_parity` is battle-backed and CHEAP, and putting it behind the old
`-m "not integration"` gate is precisely how it rode main RED **three separate times**.

**A tier is DECLARED, never inferred.** Cost arrives transitively — the obs-golden test reaches
`run_local_battles` through a helper it imports — so neither a filename nor an import graph can
classify it, and both were measured wrong here: 30 collected test files transitively reach the
battle runner, nearly all millisecond unit tests. The root `conftest.py` enforces the other
direction instead, on the only signal that is actually cost: **an unmarked test that overruns a 30 s
budget is reported and told which marker to take** (`GEN3AI_SKIP_TIER_BUDGET=1` opts out). Only the
COST markers exempt — a `sim` test is not excused for being slow. The 30 s is sized from the
measured distribution with clearance: the tier's slowest legitimate members sit at 12-20 s on a
quiet box, and a 20 s budget flapped on a **0.1 s** overshoot.

⚠️ **It FAILS the run only on a QUIET box; on a busy one it is ADVISORY, and scaling the budget was
NOT enough on its own.** The contention factor is `loadavg / cpus` (~1.4 at load 22), but a
compile-heavy test competing for every core slows by multiples of that — **measured 2026-08-14:
12.3 s idle → 65.9 s at load 22, a 5.4× slowdown against a 1.2× scaled budget.** A scaled-only guard
therefore goes red whenever a training run is live, which is most of the time, and this tree has
already eaten that failure mode twice over (the starved parity run that reported 39/40 timeouts as a
clean PASS; three investigations voided by wall-clock bounds measured beside a trainer). A duration
measured under starvation is not a measurement of the test, so it cannot be a verdict on it.

**"Quiet" means factor < 1.05, NOT `contention.py`'s 1.25 "looks idle" wording** — and that
distinction is itself a shipped-then-fixed bug. Reusing 1.25 put the threshold exactly where this
box lives: a `--nice 10` trainer parks the load at ~16-25 on 16 cpus, so a gate run **failed at
factor 1.24** on two tests whose idle cost is a third of what they showed, while printing "box looks
idle" beside a load average of 19.9. On this machine the guard therefore REPORTS and enforces on a
genuinely quiet box (CI, a deliberate idle run) — the honest asymmetry, since you cannot take a
trustworthy duration measurement on a machine that is always training.
`tier_budget_guard_test.py` pins all of it — the two halves, the 1.24 band reading as contended, and
that the guard may only ever ADD a failure, never clear one.

### Test file naming conventions

| Pattern | Requires | Marker |
|---|---|---|
| `*_test.py` | Nothing — pure unit tests with mocks | — |
| `*_integration_test.py` | An out-of-process dependency, no live server. **The name is historical and no longer implies the tier** — these split across `integration` (light), `sim` (bridge battles) and `browser` (headless chrome, always `slow`). Read the file's `pytestmark`, not its name | `integration` and/or `sim` / `browser` / `slow` |
| `*_fuzz_test.py` | `deps/pokemon-showdown` — runs **real battles in-process via the local BattleStream bridge** (`utils/bridge/local_battle_runner.py`); **no live server**. The default for fuzzing. | none — run directly as scripts (no `test_*` funcs, so `pytest` imports but collects nothing) |
| `*_fuzz_e2e_test.py` | A **live Showdown server** — fuzz whose checks need real async-server timing (e.g. `effectiveness_fuzz_e2e_test`, whose TurnDelta-vs-BattleContext effectiveness window is decision-timing-sensitive) | run directly as scripts |
| `*_e2e_test.py` | A **live Showdown server** on localhost:8000 | `@pytest.mark.e2e` (scripts only, run directly) |
| `*_benchmark.py` | `deps/pokemon-showdown` bridge (no live server) — **performance profiling, not pass/fail**: plays a real battle in-process, then `cProfile`s a hot path | none — run directly as scripts (no `test_*` funcs → `pytest` collects nothing). Place in a dir with no stdlib-shadowing names (e.g. `training/`, not `observation/`) |

### Which command to run

| When | Command | Measured |
|---|---|---|
| **inner loop** — you want the fastest true/false | `-m "not slow and not e2e and not sim and not integration"` | 4536 tests, **127 s** (~56 s at `-n 4`) |
| **THE ROUTINE GATE** — before a commit | `-m "not slow and not e2e"` | 4634 tests, **4 m 36 s** |
| **before a `/gen3ai-ship`, and in CI** | `pytest src/` (everything) | **31 m** |
| just the bridge | `-m sim` | ~100 s |
| just the browser views | `-m browser` | ~24 m |

```bash
# THE ROUTINE GATE — everything cheap, whatever it needs. Add -n 2 (~1.8x, two cores).
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -m "not slow and not e2e" -q
```

**Do not use the old `-m "not integration and not e2e"`.** It is what let the obs-golden linchpin
rot on main three times: `integration` now spans a ~100x cost range, so excluding it throws away
cheap, high-value coverage (bridge battles, data parity, mechanics) to avoid the browser suite. Cut
on **`slow`** instead — that is the marker that means "expensive".

### The two STATIC gates (mypy + ruff) — default-on, in every tier

Static checking is enforced by **tests**, not by habit, because there is no CI on this box: the
routine suite is the only thing that runs on every change, so a check outside it is advisory and
rots. Both are unmarked (they run even in the fast inner loop) and both are ~free:

| Gate | Runs | Scope | Measured |
|---|---|---|---|
| `src/agents/model/mypy_gate_test.py` | `python -m mypy src/agents/model` | the model package only, per `mypy.ini` | **0.28 s warm**, 19.6 s cold |
| `src/ruff_gate_test.py` | `ruff check src/agents src/main src/utils --select F,E9 --exclude src/poke_env --exclude src/rust_sim` | `agents/` + `main/` + `utils/` | **0.10 s** |

They are complementary, not overlapping: mypy is deep over ONE package (`mypy.ini` sets `files =
src/agents/model` with `follow_imports = silent`, so the rest of the tree is read for types but not
reported — widening is a `mypy.ini` edit and the test follows it), while ruff is shallow over
everything. `--select F,E9` is pyflakes + syntax errors only — findings that mean the code is
**wrong**, never a style opinion, so the gate cannot degrade into a formatting argument.

**A missing tool FAILS, it does not skip** (both are pinned in `environment.yml`) — a linter that
silently opts out reads exactly like a linter that found nothing. Opt out explicitly with
`GEN3AI_SKIP_MYPY_GATE=1` / `GEN3AI_SKIP_RUFF_GATE=1`.

**Known ruff findings are per-file entries in `ruff.toml`, never a blanket exclude**, and that file
keeps two categories apart: a PERMANENT one (the model package's declared re-export hubs, which
import names solely so other modules can import them back out) and a TEMPORARY handoff list of
ordinary dead code. The second is meant to shrink to nothing. `/gen3ai-ship` runs both gates before
staging (step 1c), so the ship path does not depend on whether the suite was run.

### Unit tests only (the fast inner loop)
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -m "not slow and not e2e and not sim and not integration" -q
```

**Add `-n 2` — it is ~1.8x faster** and costs the box only two cores, which matters because a
training run normally shares this machine. `pytest-xdist` is in `environment.yml`.

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -m "not integration and not e2e" -q -n 2
```

Measured 2026-08-14, 16-core box, whole unit suite, same 4527 passed. **The two columns are not
comparable to each other** — a benchmark on this box is meaningless while a production run shares
it, so the idle and under-load figures are reported separately rather than blended:

| | idle box | beside a live training run |
|---|---|---|
| serial | 147 s | 160-168 s |
| **`-n 2`** | — | **90.4 / 90.9 s** |
| `-n 4` | 56 s | 72.1 / 73.9 s |
| `-n 8` / `-n 12` / `-n 16` | 59 / 57 / 59 s | — |

`-n 4` is the floor: past it the wall is one long-pole file (the `torch.compile` gates), so more
workers buy nothing. Under real conditions `-n 4` saves only ~17 s over `-n 2` while taking twice
the cores from the run — hence `-n 2` as the default and `-n 4` when the box is yours. Use plain
serial when you need `-s`, a debugger, or a readable single failure.

> ⚠️ **The parallel speedup depends entirely on BLAS thread pinning, and the root `conftest.py` now
> does it for you** (`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`; escape hatch
> `GEN3AI_TEST_ALLOW_THREADS=1`). Unpinned, `-n 8` measured **389 s — 6.5x SLOWER than pinned
> serial**, with `user` time at 68 min vs 3 min: N workers x 16 BLAS threads on 16 cores thrashes the
> box. It is the same ~38x cliff `src/main/thread_pinning_test.py` defends for env workers. Without
> the conftest pin, anyone trying `-n auto` would measure a slowdown and conclude parallelism does
> not work here.

### Everything, including the slow tiers (requires symlinked deps/pokemon-showdown + chrome)
**Run this before a `/gen3ai-ship`, and in CI.** ~31 minutes, ~24 of them the browser suite.
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -q
```

⚠️ **A FRESH WORKTREE pays for a `cargo build --release` on its first rust-backed test**, which
saturates every core and can turn the contention-scaled per-battle timeouts into a wall of
TIMEOUTs. Observed twice, both times misreading as a rust defect: `bridge_impl_parity` reported 8
of 12 battles timed out and one transport error, and `better_line[rust]` reported a candidate
divergence — **both passed on the warm tree with no code change.** Build the binaries first, or
discount the first run in a new worktree:
```bash
cargo build --release --bin sim_bridge --bin search_driver --manifest-path src/rust_sim/Cargo.toml
```

### Fuzz tests (`*_fuzz_test.py`, run directly as scripts)
Run battles **in-process via the local BattleStream bridge — no `npm run showdown`
needed** (`utils/bridge/local_battle_runner.py`):
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/fuzz_test.py [n_battles]
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/poke_env_gaps/transition_fuzz_test.py [n_battles]
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/battle/event_log_fuzz_test.py [n_battles]
# also bridge-backed (no server): poke_env_gaps/{abilities,item_consumption,move_outcome,snatch,incoming_damage}_fuzz_test.py,
#                                  poke_env_gaps/move_alignment_fuzz_test.py (per-move obs features ↔ legal.move_slots[k] ↔ action 6+k, forces Choice-lock/Disable),
#                                  poke_env_gaps/belief_labels_fuzz_test.py (hidden-opp belief labels == actual opp team + no-leak),
#                                  poke_env_gaps/damage_op_probe_fuzz_test.py (AUTHORITATIVE DamageOperator physics gate — CONSTRUCTED single-turn
#                                      scenarios via the OMNISCIENT BattleStream `utils/bridge/damage_probe.js`: exact both-side HP + the sim's OWN
#                                      stats, zero measurement confounds; one modifier per scenario [type/STAB/SE/resist/4×/immunity/Thick Fat/
#                                      Choice Band/item/boosts/burn/screens/weather]) + poke_env_gaps/damage_op_fuzz_test.py (looser random-game net),
#                                  training/hidden_power_tracker_fuzz_test.py,
#                                  utils/bridge/reconstruction_fuzz_test.py (battle replay/re-roll invariants),
#                                  utils/bridge/reroll_many_parity_fuzz_test.py (batched reroll_many == per-call reroll_turn, bit-for-bit obs),
#                                  utils/bridge/search_clone_parity_fuzz_test.py (serializeBattle clone == reroll_many, bit-for-bit obs + value_crn anchor + depth-2),
#                                  and training/obs_roundtrip_fuzz_test.py (offline obs == live obs, bit-for-bit)
```

### E2E tests (`*_e2e_test.py` / `*_fuzz_e2e_test.py`, require a live server)
```bash
# Start server first: npm run showdown
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/telemetry_e2e_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/poke_env_gaps/effectiveness_fuzz_e2e_test.py [n_battles]
```

### Benchmarks (`*_benchmark.py`, run directly as scripts)
Profile a hot path on a real bridge battle (no server). `obs_build_benchmark.py` plays until a
representative late-game decision, then reports a component wall-clock breakdown
(`state_encoder.encode` vs deque-cached turn-history vs `live_view()`) plus a `cProfile`
`tottime` ranking — use it to catch obs-pipeline regressions and confirm an optimization moved
the bottleneck:
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/obs_build_benchmark.py [--turn 25] [--reps 400] [--top 22] [--battles 200] [--seed 0]
```
Absolute ms scale with machine load; the component **ratios** and the cProfile ranking are the
load-stable signal — run on an otherwise-idle box for a clean baseline. **You no longer have to
remember**: every benchmark calls `warn_if_contended()` at entry and prints a loud "THE BOX IS
BUSY" banner with the load average when the box is not idle (see [Running beside a live training
run](#running-beside-a-live-training-run-gen3_contention_robust_timeouts_v1)).

**Every change under `src/agents/observation/` must run this benchmark before/after and
confirm no meaningful regression** — that gate, the canonical baseline, and the
load-stable regression criteria live in `src/agents/observation/CLAUDE.md`.

For the **top-down** view — where a whole trainer turn's CPU goes (parse + obs + reward +
mask + map + tracker), GPU-excluded and server-free — use `trainer_turn_benchmark.py`. It
walks a real bridge battle, times every per-decision CPU stage `Gen3Env` runs (a random legal
action stands in for the policy forward), and prints a stacked breakdown. Baseline: obs build
≈ 88% of our CPU (`state_encoder.encode` ≈ 80%), parse ≈ 7%, reward ≈ 4%, everything else <1%.
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/trainer_turn_benchmark.py [--decisions 150] [--warmup 3] [--seed 0]
```

### What "fuzz test" means in this project

**Fuzz tests run real battles — by default in-process via the local BattleStream bridge (no server), or against a live server — and validate observations or behaviour against the actual protocol stream.** They are NOT deterministic scenario tests with fixed inputs.

The canonical pattern (see `src/agents/training/poke_env_gaps/`):

1. Subclass `Player` and override `_handle_battle_message` to intercept raw Showdown protocol lines mid-battle.
2. Archive per-turn snapshots of the state you care about (e.g. which items were consumed, which moves were used).
3. In `choose_move()`, validate that the encoded observation vector matches what the archived protocol events say should be there.
4. Run N random battles; any validation failure raises immediately with a detailed error.

This catches poke-env parsing bugs and encoder gaps that unit tests with mocks cannot — the test exercises the Showdown sim → poke-env → encoder pipeline end to end (the bridge feeds the identical protocol stream the live server would). When asked to write a fuzz test, always follow this pattern rather than writing parametrized unit tests with hand-crafted mock objects.

---

## Smoke Test

Before a full training run, verify the core pipeline (env, reward, replay callback, stall
detection) with a quick debug run (~1 min):

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py --debug --steps 10000
```

`--debug` defaults to **CPU** (so a smoke never contends with a live GPU training run — an
explicit `--device cuda` still wins) and **skips all eval** by default — both the periodic eval
callback and the final win-rate eval. It is also **serverless out of the box**: `--use-bridge`
defaults to `rust`, so the in-process sim runs the training battles and no Showdown server is
involved at all. To also exercise the eval pipeline (final win-rate eval, and the self-play
seed → pool eval → promotion path under `--self-play`), add `--debug-eval` — still serverless
on the default; only an explicit `--use-bridge off` needs a server (default `:8000`).

What to look for:
- `[ModelVersion] Round-trip smoke test PASSED` — serialization and reload are healthy (printed early, before training begins)
- `🏁 Episode Finished` lines appearing throughout — episodes completing and resetting
- `[STALL LOGGED]` may appear if a 250-turn game occurs — should be followed by another `🏁 Episode Finished`, not a hang
- `Win rate vs Random` / `Win rate vs Heuristic` at the end — **only with `--debug-eval`**; the default smoke ends at `Training complete` with no eval

A hang after `[STALL LOGGED]` or a crash before "Training complete" indicates a regression in the env/stall/forfeit pipeline. A `[ModelVersion] FATAL` error at startup means the checkpoint was saved with a different architecture than the current code.

---

## Launcher (preferred for long runs)

`src/main/launcher/` wraps `train_rl_agent.py` with **periodic restarts** (reclaim pymalloc
fragmentation; child saves a checkpoint on SIGTERM, launcher relaunches), **crash auto-restart**
(a self-crash relaunches from the last checkpoint after dumping a per-crash
`<run_dir>/crashes/restart_err_<token>.txt`, with a `--max-crash-restarts` circuit-breaker), **git-worktree
isolation** (agent pushes to `main` never affect a running session), a **Textual TUI** (metrics,
FPS, restart countdown, a `↻ N restarts (M crash)` badge; `l` logs, `e` events, `d` dashboard,
`r` restart, `c` checkpoint, `p` plots, `s` status, `f` force eval (confirm-gated; rejected if an
eval cycle is already running), `q`/ctrl-c quit, `v` copy mode
[freeze + native terminal select-and-copy — the portable copy path, works on Terminal.app]),
and **live crash-log
streaming** to `<run_dir>/launcher_child.log`.

The UI is **Textual** (built on the shared `src/main/tui/` base), launched with
`python -m main.launcher …` (or the back-compat alias `python -m main.launcher.tui …`). A closed
terminal (SIGHUP) or external `kill` (SIGTERM) is caught and turned into a clean,
checkpoint-saving shutdown rather than a lost checkpoint.

**A detached launch (`nohup … < /dev/null &`, systemd, cron) runs HEADLESS automatically** — with
no TTY on stdin, Textual's input thread would otherwise busy-loop a whole core forever (measured
on a live run: 96% of a core for 13 h, plus a 982 MB log of full-screen ANSI repaints growing at
17 KB/s into a redirected file). Headless drops the input thread and the repaints, and events are
echoed as plain `[HH:MM:SS] …` lines instead, so `tail -f` on the redirect target still follows
the run. A TTY keeps the full interactive TUI. Detail: `src/main/launcher/CLAUDE.md`.

**Internals — how the UI reconciles the restart loop with Textual's event loop, the
quit/ctrl-c/SIGHUP teardown, crash reporting + auto-restart, exit codes
(`COMPLETE`/`INTERRUPTED`/`CRASH`/`FATAL_CONFIG` — the last gives up without restarting on an
arch/config mismatch instead of looping), the full flag table (`--restart-interval-hours`,
`--restart-grace-minutes`, `--max-crash-restarts`, `--nice`, `--no-pin`, `--sync-to-main`), the
resume contract, and the `:8001` Showdown-port default — live in `src/main/launcher/CLAUDE.md`.**

**The launcher and everything it spawns run at `--nice 10` by default** (`0` disables). A run holds
~940 processes; at nice 0 it competes on equal terms with interactive work sharing the box.
Niceness is inherited across fork/exec, so one call before the first child covers the trainer, its
SubprocVecEnv workers and every eval worker, across periodic and crash restarts alike. **On an idle
box this changes nothing** — niceness only arbitrates under contention.

### Will this command still launch? — `python -m main.checkargs`

A run's recorded `launcher_command` outlives the flags in it, and **argparse reports only the FIRST
unrecognized flag** — so relaunching an old argv after a deletion is a launch-crash-fix loop, ~40 s
and a stray run dir per stale flag. `checkargs` answers offline, in one pass, without importing
torch or touching `models/`:

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.checkargs models/<run>                 # validate that run's recorded command
python -m main.checkargs --argv "--steps 1 --device cuda"
```

Exit 0 = every flag is accepted; exit 1 names each one that would fail (with its value, so you can
see whether it mattered). It checks **two** ways a command fails: a flag the parser no longer knows,
and a COMBINATION the extractor refuses — `agents.model.flag_registry`'s `requires` graph, e.g.
`--intent-conditional` without `--damage-outgoing`. That second crash is later and dearer than an
argparse error (the run dir exists, the child starts, the traceback comes out of
`Gen3FeaturesExtractor.__init__`). The dependency half is deliberately conservative: it fires only
when the argv enables a flag AND explicitly names a dependency with a disabled value, because a
resume inherits every unspecified flag from the checkpoint's config, so absence carries no
information. **It reports; it does not repair** — a deleted flag may have a
replacement, so dropping one silently could change the run. Launcher-owned flags
(`--restart-interval-hours`, `--nice`, `--sync-to-main`, …) are recognised as not-forwarded rather
than reported as stale. **Run it after deleting flags**, over the recorded commands of any run you
might still relaunch or fork — that is what it is for.

It reads the parser's own `_actions` via **`train_rl_agent.build_parser()`** (extracted from
`main()` so the parser can be inspected without running a training job), not scraped `--help` text.
That distinction is load-bearing: `--help` was itself broken by one unescaped `%` — `"~0.6% of"`
renders as a space-flag `%o` conversion and raises `TypeError: %o format: an integer is required,
not dict` — and nothing rendered the help strings, so nothing caught it.
`checkargs_test.py::test_every_help_string_renders` is now that guard.

### Starting a fresh run via launcher
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.launcher \
  --restart-interval-hours 3 \
  --steps 15000000 \
  --n-envs 64 \
  --batch-size 16384 \
  --n-epochs 10 \
  --ent-coef 0.02 \
  --n-steps 2048 \
  --lr 0.0003 \
  --device cuda \
  --log-level periodic
```

### Resuming from a checkpoint
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.launcher \
  --restart-interval-hours 3 \
  --model models/<run>/checkpoints/checkpoint_NNNN_steps.zip \
  --steps 15000000 \
  --device cuda
```

Periodic + forced checkpoints live in `models/<run>/checkpoints/` (each `.zip` beside its
`.json` sidecar); legacy runs kept them at the run root and still resume. The checkpoint must
carry a `metadata.json` with a `git_hash`; the launcher pins the isolated
worktree to that commit so the resumed run uses the original code (override with
`--sync-to-main`). All non-launcher flags are forwarded verbatim to `train_rl_agent.py`.
`python -m main.launcher.tui …` is an alias for the same command.

---

## Training

Run directly (no restart loop, no worktree isolation):

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py \
  --model <path/to/checkpoint.zip> \
  --steps 15000000 \
  --n-envs 64 \
  --batch-size 16384 \
  --n-epochs 10 \
  --ent-coef 0.02 \
  --n-steps 2048 \
  --lr 0.0003 \
  --device cuda \
  --log-level periodic
```

Omit `--model` to start a fresh run. Use `--debug` for a single env (DummyVecEnv). Use `--device cpu` on machines without a GPU.

**GPU OOM lever — `--grad-accum-steps K`:** keep a large effective batch when the full minibatch
won't fit. It runs K `--batch-size` micro-batches and steps the optimizer once per group of K,
giving the **exact** gradient of a `batch_size·K` batch at the activation-memory cost of one
micro-batch (stock SB3 steps per-minibatch, so `--batch-size` alone couples effective batch to the
memory peak). E.g. `--batch-size 4096 --grad-accum-steps 4` trains like `--batch-size 16384` at ~¼
the peak. `K=1` (default) is byte-identical to stock; it's a train-loop knob (not version-locked) —
forward it on every resume like `--batch-size`. With `K>=2` it also emits a **`train/noise_scale`**
diagnostic (McCandlish critical batch size) that tells you, as a number, whether your effective batch
is too small / about right / bigger than needed. Details: `src/agents/training/CLAUDE.md` → Gradient
accumulation.

Checkpoints are saved automatically into `models/run_<timestamp>/checkpoints/` (each `.zip`
beside its per-checkpoint `.json` sidecar); the run-level `model_config.json` / `metadata.json`
/ `latest.txt` and the `final_model*.zip` / `best_model/` stay at the run root.

### In-process bridge transport (`--use-bridge {off,node,rust}`, **default `rust`**)

`--use-bridge` swaps **both training and eval** between a websocket Showdown server and an
in-process `BattleStream` subprocess — no server, no port, no `/challenge` connection storm,
deterministic delivery (poke-env issue #907). **THE DEFAULT IS `rust`, so a run needs no Showdown
server at all.** It reuses the *entire* obs/reward/mask/wrapper stack unchanged.

**The three values:**

| value | transport | when |
|---|---|---|
| **`rust`** | the std-only pokesim `src/rust_sim/src/bin/sim_bridge.rs` | **the DEFAULT** — fastest, smallest child, serverless |
| `node` | the Node `local_sim_bridge.js` | the explicit A/B arm; the parity harness and `gen_sim_bridge_diff.js` need it |
| `off` | websocket to a Showdown server on `--showdown-port` | the ladder / live-server path |

`rust` is a byte-for-byte protocol-compatible drop-in for `node` (validated by
`src/rust_sim/harness/gen_sim_bridge_diff.js`), so nothing above the transport changes. The
DEPRECATED `--use-showdown-bridge` boolean alias is **DELETED**: it meant `--use-bridge=node`,
which is no longer the default, so keeping it would have made the legacy spelling silently select
the slower impl. The flag resolves into one internal `bridge_enabled: bool` + `bridge_impl:
"node"|"rust"`; `bridge_impl` is threaded to `attach_bridge_transport(env, …, impl=…)` (training),
`run_local_battles(…, impl=…)` (eval driver), and the eval-worker shard config
(`bridge_impl` alongside `use_showdown_bridge`). The binary is resolved by
`src/utils/bridge/sim_bridge_bin.py::resolve_sim_bridge_bin()`: `$POKESIM_SIM_BRIDGE_BIN`
(absolute override) first, else `cargo build --release --bin sim_bridge` in `src/rust_sim`
(cached; a clear error, never a silent node fall-back, if cargo/crate/binary is missing).

**`rust` deferrals + coverage limit (honest, warned at startup; re-audited 2026-08-03).** The old
"the Rust bridge emits **no `__RECON__`**" claim is **STALE** — `2b826d4` shipped both
`gen3_bridge_recon_record_v1` and `gen3_bridge_resume_reseed_v1`, and a **seeded** rust battle
passes the whole forensic stack: `reconstruction_fuzz_test` (replay reproduced the rust-recorded
winner + turn), `reroll_many_parity_fuzz_test`, and `search_clone_parity_fuzz_test` (the clone's
successor obs equals the rust-recorded `states.npz` next obs **bit-for-bit**) all PASS when the
record is fed to the Node `replay_driver.js` / `search_driver.js` — a strong cross-impl parity
result, since the offline replay/clone layer is Node either way. What is **actually** still broken:

**⚠️ THE THREE SEED DEFECTS BELOW ARE FIXED — do not re-derive plans from them** (`bc00d4d`,
`gen3_bridge_seedless_fixed_seed_v1` + `gen3_bridge_seed_forms_v1`; re-verified against the live
tree + binary 2026-08-04). Kept as history because the *coverage-hole lesson* is the durable part.
- ~~The SEEDLESS path produces nothing / replays one dice stream.~~ **FIXED.** A seedless `START`
  now MINTS a fresh `sodium,<hex>` (`Prng::generate_seed`) instead of falling through to
  `DEFAULT_CONSTRUCT_SEED = "0,0,0,0"`, and still emits `__RECON__` (the resolved seed is the
  minted one). So eval traces DO get their `*_reconstruction.json` sibling and every training
  episode draws its own dice. Gate: `bridge_impl_parity_test::
  test_seedless_rust_battles_are_distinct_and_recorded` (three seedless battles must hash
  DIFFERENTLY and each carry a `__RECON__`).
- ~~A STRING `seed` is SILENTLY IGNORED.~~ **FIXED.** One `parse_seed_field` accepts every form
  Node accepts — `[a,b,c,d]`, `"m,n,o,p"`, `"gen5,<hex16>"`, `"sodium,<hex>"` — and a
  present-but-unparseable seed is a LOUD `__ERR__`, never a silent fall-through. Gate:
  `test_seed_forms_reproduce_the_same_battle_on_rust_and_node` (parametrized over all forms).
- ~~`resumeReseed` accepts ONLY the array form.~~ **FIXED** — same shared parser, so the
  counterfactual Monte-Carlo works on rust.
- ~~The clone-and-branch SEARCH server has no rust path.~~ **CLOSED — search runs on rust.** Three
  layers landed together:
  - **The snapshot primitive** (`gen3_bridge_clone_branch_v1`) — `BridgeSession::snapshot()`, a
    derived deep `Clone`, plus `clear_chunks` / `request_kind` / `is_choice_done` /
    `active_request_json` / `battle_state` / `winner`; gated by `tests/bridge_clone_branch_test.rs`.
    The `Battle::serialize`/`deserialize` stubs are **DELETED**: Showdown needs a byte format
    because its battle graph is cyclic, but the port's state is plain owned data and the snapshot
    never crosses a process boundary, so an in-process clone is the whole requirement.
  - **The drivers** — ONE binary, `src/rust_sim/src/bin/search_driver.rs`, serves BOTH offline verb
    families: `open_root` / `expand_many` / `close` over a persistent stdin loop
    (`gen3_rust_search_driver_v1`) and the one-shot `replay` / `reroll` / `reroll_many`
    (`gen3_rust_replay_driver_v1`; node splits these across `search_driver.js` +
    `replay_driver.js`). `src/rust_sim/src/search.rs` ports `replay_kernels.js` — the mulberry32
    aux-RNG, `random_choice`, `resolve_turn`/`resolve_turn_exact`, `recorded_queues`, `outcome_of`.
  - **The Python seam** (`gen3_search_driver_impl_seam_v1`) — `resolve_search_driver_bin()` /
    `search_driver_spawn_argv(impl)` mirror the `sim_bridge` pair (env override
    `$POKESIM_SEARCH_DRIVER_BIN`, else `cargo build --release --bin search_driver`, cached, clear
    error, **never** a node fall-back). `impl="node"|"rust"` threads through `SearchSession(impl=)`,
    `reconstruction.{replay_battle,reroll_turn,reroll_many}(impl=)`,
    `obs_materializer.{materialize_from_record,infer_action_indices}(impl=)`, the prober
    (`ProbeSession(impl=)` + `python -m main.prober.query --impl {node,rust}`), both search-teacher
    workers, `SearchTeacherCallback(impl=args.bridge_impl)` and `teacher/generate.py`'s
    `run_local_battles` (which used to take node silently). **Every default is `"node"`**, so this
    is byte-identical until someone asks for rust.

  **Gates** (all run): `src/rust_sim/harness/search_impl_parity.py` 6 cases / 60 arms / 18873 leaf fields and
  `src/rust_sim/harness/replay_impl_parity.py` 76 cases / 136 arms / 30689 leaf fields, both node-vs-rust with only
  `|t:|` normalized; `search_clone_parity_fuzz_test --impl rust [--record-impl rust]` (clone ≡
  `reroll_many` at the OBS, bit-for-bit); `counterfactual_fuzz_test --impl rust`;
  `better_line_integration_test`, parametrized over both impls plus a **cross-impl** test asserting
  node and rust produce identical candidate V — and since that fake model's `V = obs.sum()`, an
  exact match is an obs-level bit-identity claim at every ply of the beam.

  **The `--search-teacher` + rust guard is GONE.** For the record, the reason it used to give was
  **WRONG** — not `input_log` byte-identity: *nothing reads the record's committed-choice lines*
  (`replay_kernels.js::writeStart` and `ReconstructionRecord.start_options()`/`players()` read only
  `>start`/`>player`, which the rust record renders exactly). The real blocker was always the
  missing driver. **Do not re-derive a plan from the old reason.** Not yet gated: a full multi-cycle
  teacher run end-to-end on rust — every leg is gated, the composition is not.

  **The CHOOSE path is `__ERR__`-parity gated** (`gen3_bridge_choose_path_parity_v1`). An `__ERR__`
  is NOT an in-band error — it retires `BridgeSession`'s reader, trips `_signal_transport_dead()`,
  and raises `ShowdownException` in every in-flight `step()` — so **any CHOOSE node tolerates, rust
  must tolerate too**; a stricter parser there is a whole-run crash. Two such divergences killed
  `--use-bridge=rust --n-envs 48` at ~8 min, twice, at load 31 and at load 5 alike (a RATE, not a
  load effect): `CHOOSE <side> default`/`pass` (node passes every token to `Side.choose` verbatim;
  `parse_choice` took only `move `/`switch ` — and `/choose default` is routine, from
  `singles_env.py`'s `action == -2`, an inference player's `None` predict, its redecide exhaustion,
  and `DEFAULT_CHOICE_CHANCE`), and a **stray CHOOSE after `__END__`** on a persistent child (the
  child resets at `__END__` while `_dispatch` fires poke-env's feeds as un-awaited tasks). Gates:
  `bridge_impl_parity_test.py::test_poke_env_fallback_choice_tokens_never_produce_a_fatal_err` /
  `::test_stray_choose_after_battle_end_is_ignored_on_a_persistent_child`, both over node AND rust.
  **The existing fuzz gate could not catch either** — it drives only masked-legal tokens and never
  lands a post-`__END__` CHOOSE (22k episodes × 16 workers pass clean pre-fix). A related
  DIAGNOSTICS fix ships with it (`gen3_bridge_fatal_report_now_v1`): the `__ERR__` text used to be
  latched into `_child_error` and printed only by the NEXT `reset()`, which never runs, so the sole
  surviving evidence was poke-env's generic "websocket dropped" — `_report_fatal` now prints the
  reason + child stderr tail immediately. (And `race_trace.dump_recent()` is a no-op unless
  `GEN3_RACE_TRACE=1`; an empty dump means the buffer was off, not that nothing happened.)

  **⚠️ The old CHOICE-REJECT allowlist entry was FALSE ON BOTH HALVES and is DELETED — do not
  re-derive a plan from it.** It claimed rust "emits no `|error|` frame and re-opens the boundary to
  BOTH sides ... on a path poke-env never takes". The framing half was closed by
  `gen3_choice_reject_framing_v1` (the entry survived its own fix); the "never takes" half was
  falsified by poke-env taking it and killing **two production launches** at ~8 minutes. The real
  defect was `gen3_locked_choice_never_rejected_v1`: a MOVE-LOCKED mon (two-turn charging, or
  `must_recharge`) gets a single-entry request with `trapped:true`, and `classify_reject` — which
  never consulted `move_locked()` — could then REFUSE the only move that request offered, because
  `move_usable` models Choice-lock/Disable/Encore/Taunt/PP and knows nothing of lock-in. Rust
  contradicted ITSELF; it was not a stricter parser. Fixed with a `move_locked()` early-out beside
  the existing `must_struggle` one (same principle: **a forced choice is not a refusable one**), and
  one predicate covers the charge family and the recharge mirror, so fly/dig/bounce and hyperbeam go
  with it. Gated by `bridge_choice_reject_test::a_move_locked_mon_is_never_rejected_for_its_only_offered_move`
  — VERIFIED failing on revert. **The existing fuzz cannot catch this class by construction**: it
  drives masked-LEGAL tokens, and here the token IS masked-legal (the mask is built FROM the
  request), so 22k episodes passed clean while it was live. **The durable lesson: an allowlist entry
  can outlive its own fix and then mislead every reader after — including a subagent briefed from
  it.** One honest gap remains, allowlisted and printed by both harnesses, never silent: (2) `pre_state` volatile NAMES are reconstructed from the port's typed
  fields, and the golden verifies exactly one fact about them (duration-1 volatiles must not leak
  into a move boundary — that one really did diverge and was fixed). `pre_state` has no consumer.

**THE DURABLE LESSON from the seed defects** (why they shipped at all): every gate on that path —
`sim_bridge_bin_test`, `gen_sim_bridge_diff.js`, and the parity test's own check 2 — was inherently
SEEDED or compared only aggregates, so the **production SEEDLESS branch was never exercised**. When
a code path has a "default" branch nothing tests, that branch is untested no matter how green the
suite looks.

**Coverage (MEASURED, not asserted).** The port fail-louds (`__ERR__` → `RuntimeError` → env crash
→ launcher restart; the child survives via `catch_unwind`) rather than desync. On the **training
pool it is a non-issue**: 719/719 `data/teams/` teams construct (719 = the LOADED/deduped count
from 773 raw `.txt` files — both figures appear in this doc and mean different things), and 1500 random-play rust battles
hit **zero** coverage errors (the only 4 failures were the 1000-turn runaway cap). On
`gen3randombattle` **the construction blockers are CLOSED**: the Deoxys/Unown forme DATA gap is
fixed (`gen3_species_formes_v1`, ROUND 38), `transform` is modeled (`gen3_transform_v1`, ROUND 33),
the wrap family too (`gen3_partial_trap_v1`, ROUND 32), and **Forecast/Castform — the last blocker
(~2.6% of teams) — is modeled (`gen3_forecast_v1`, ROUND 35, with the hail/sandstorm weather-set
moves and the expiry-draw fix it flushed out)**. Arbitrary ladder gen3ou is ~5% of battles
(INFERRED from Smogon usage weights). The old inverse hazard — unmodeled items/moves running as
SILENT no-ops / generic hits — is CLOSED by the ROUND 39/40 silent-no-op audits: the 5
genuinely-effectful unmodeled items (`gen3_unmodeled_item_failloud_v1`) and the 16 silent-desync
moves (`fakeout`, `rollout`, the lock-in family, `eruption`, … —
`gen3_unmodeled_move_failloud_v2`; full-universe census: 369 gen3-legal moves → 281 modeled, 88
fail-loud, 0 silent) now FAIL LOUD at construction, and every gen3 ability is modeled or
verified-no-op (none fail-loud). Measured exposure of the guarded sets is ZERO on both surfaces
(0 pool carriers; 0 in the entire curated randbats movepool) — latent-hazard guards. (The former
**seeded speed-tied-lead** / unspecified-gender divergence is FIXED —
`gen3_turn0_construction_v1` models the turn-0 construction window, so a seeded rust battle is
byte-for-byte with node.) See `src/utils/bridge/README.md`. Transport parity (poke-env sends
move-ids/species names, e.g. `move hiddenpowerice`) is guarded by
`src/utils/bridge/bridge_impl_parity_test.py`.

**`rust` is FASTER than node, and its child is ~25× smaller** (`bridge_impl_throughput_benchmark.py`,
a same-invocation A/B of N parallel env workers on an idle 16-core box): at 8 workers **1.18×**
(1852 → 2182 steps/s), at the production `--n-envs 48` **1.41×** (1942 → 2729 steps/s), with the
bridge child at **9 MB RSS vs node's ~224 MB** — so the children cost ~0.4 GB under rust vs ~10.7 GB
under node at `n_envs=48`. (An older note recording node 798 vs rust 427 fps at 8 envs was measured
on a CPU-saturated box and is superseded.) **`gen3_bridge_forfeit_win_v1` (2026-07-31) was the
blocker that made rust unusable for training**: `FORCELOSE` emitted a bare `__END__` with no `|win|`
line, so poke-env never marked the battle finished and the next `reset()` hung forever. Since the
training seam forfeits whenever `reset()` lands mid-battle, every episode boundary could wedge —
the multi-env soaks logged finished episodes yet completed ZERO PPO iterations. Fixed in
`BridgeSession::forfeit`; the durable gate is `bridge_session_fuzz_test.py --impl rust` (its
every-9th-episode forfeit-reset is the reproducer, and `--impl` exists because that fuzz previously
only ever tested node — the coverage hole that let this ship).

It reuses the *entire* obs/reward/mask/wrapper stack unchanged:

- **Training** — `attach_bridge_transport` (`src/utils/bridge/bridge_session.py`) swaps the two
  `_EnvPlayer` agents' transport for a background-pumped bridge subprocess per env. The child is
  **persistent by default** — one long-lived Node process reused across every episode (a fresh
  `START` rebuilds a clean `BattleStream`), which kills the per-episode Node-spawn cost. A
  single-env latency A/B (`bridge_vs_websocket_latency_benchmark.py`) measured ~13 ms/step
  websocket → ~6 ms/step persistent bridge (~2.1×); spawn-per-battle was only ~11 ms/step, so the
  reuse is the win. **But the 2.1× is single-env transport-only — at production `n_envs=64` the
  measured end-to-end training-FPS gain is just ~5%** (bridge 1192 vs websocket 1140 fps, vs-bots,
  CUDA), because oversubscription hides the per-step transport latency behind the SubprocVecEnv
  barrier (the box is CPU-saturated; n_envs is not the FPS lever — see
  `src/agents/training/CLAUDE.md` throughput notes). The bridge also started ~17% faster (no
  `/challenge` connection storm) and ran steadier. **The case for the bridge is operational, not
  FPS:** no server at all → no RAM-growth leak, no connection storm, no port tuning, deterministic.
- **Eval / self-play eval / final eval** — the eval players (built `start_listening=False`) play
  in-process via `run_local_battles` (the synchronous driver) instead of `battle_against`. Threaded
  as a `use_showdown_bridge` config key through `PerOpponentEvalCallback` / `SelfPlayCallback` →
  `eval_worker`. Each eval worker plays its opponents **one game at a time by default**
  (`--eval-concurrency-per-worker`, default `1`; threaded to `run_local_battles(concurrency=…)` /
  `max_concurrent_battles`). Overlapping battles within an opponent is single-thread asyncio
  latency-hiding, **not multi-core** — it nets negative under training contention (the old default's
  "measured slower"), but ~2× decisions/sec on spare cores (idle box / cycle tail); see
  `src/agents/training/CLAUDE.md` → intra-worker concurrency. Cross-opponent parallelism comes from the
  `--eval-workers` (5) subprocesses work-stealing the pool; this takes all eval load off the server.

**Persistent-child lifecycle (measured + optimized):** a child's RSS is **flat** — ~189 MB fresh
→ one-time ~+36 MB V8 warmup → ~229 MB with ~0 growth over thousands of battles
(`bridge_heap_growth_benchmark.py`). A child plays only ~2150 battles in the launcher's 3h restart
window, so the bridge does **not** reintroduce the server's RAM-creep and needs **no recycle within
3h** — the 3h restart owns the lifecycle. `recycle_every` (default 5000) is a backstop that never
fires under the launcher; it only caps marathon / no-launcher runs. A child that **dies** mid-run
**crashes** the env (no in-place recovery → launcher restart; resuming risks a corrupted PPO
transition).

**The default is `rust` (changed 2026-08-14; it was `off` = websocket).** The case was never
throughput — at production `n_envs` the end-to-end FPS gain over websocket is ~5% — it is
**operational**: no server to start, no port to tune, no connection storm, no RAM-creep, and
deterministic delivery. Those hold on every run, whereas the websocket path's costs land exactly
when a run is long. `rust` over `node` is the measured half (1.41x at `--n-envs 48`, a ~25x smaller
child); `node` stays a first-class explicit value because the parity harness and the A/B arm need
it, and `off` stays for the ladder. **The launcher agrees**: `child_uses_bridge` treats an ABSENT
`--use-bridge` as a bridge run, so it no longer injects a phantom `--showdown-port` (pinned by
`default_port_test.py` — a drift between the two defaults is what that file now catches). See
`src/utils/bridge/README.md` and `designs/ai_v5/design_local_sim_bridge_transport.md`.

### The two compile flags — CPU opponents vs GPU trainer

`torch.compile` is applied at two independent sites, split by WHO and WHERE. They were one flag
(`--compile-extractor`) until 2026-08-14, which named neither half:

| flag | what | device | on failure |
|---|---|---|---|
| `--compile-opponents` | each frozen self-play OPPONENT's extractor, in the env workers | **CPU** | warn + fall back to eager; `--compile-opponents-strict` opts into raising |
| `--compile-trainer` | the LEARNER's extractor — the fwd **and bwd** of the PPO step | **CUDA** | **always FATAL** |

Orthogonal — a run can take either, both, or neither. Both are runtime perf knobs: never versioned,
never in `check_compatible`, NOT inherited on resume, so re-pass them each launch.

### Compiled CPU opponents (`--compile-opponents`, opt-in)

`torch.compile`s each frozen self-play OPPONENT's feature extractor in the env workers — the measured
**68% of rollout-worker time**, run on CPU at B=1 where the graph is dispatch-bound. A **runtime perf
knob**: never versioned, never in `check_compatible`, NOT inherited on resume — re-pass it each launch
like `--grad-checkpointing`.

**Measured: B=1 CPU forward 6.371 → 0.976 ms (6.53×)** on the literal production arch (1 graph, 0
graph breaks, max|Δ| vs eager 5.07e-07), and **+33.3% marginal training FPS at `--n-envs 48`**
(406.5 → 541.8, disjoint ranges, 48/48 workers compiled) — the first throughput lever here the
`SubprocVecEnv` barrier does NOT absorb. But the per-forward win has **saturated**: doubling it
(3.6× → 6.53×) moved end-to-end only ~31% → ~33%, so the opponent forward is no longer the rollout
bottleneck and further compiler work on this path is spent effort.

Startup: `agents.model.compile_prewarm` warms the shared on-disk Inductor cache in the trainer before
any worker exists, halving worker startup (**59.6 s -> 30.1 s** wall for 16 workers). Going further,
**`--compile-opponents-preload`** (`gen3_forkserver_preload_v1`, 2026-08-16) compiles ONCE in the
forkserver so every worker inherits the traced graph (~0.12 s each) — possible since the LAZY
poke_env package inits made the extractor import single-threaded (the 2026-08 attempt hung a 48-env
run on exactly that thread), and fail-loud: a preload that cannot prove the forkserver
single-threaded RAISES instead of forking. Detail: the training leaf.
Failure is loud on stderr + the launcher event stream, and `--compile-opponents-strict` promotes it to
a hard error (falling back to eager is an invisible ~6.5× regression). "The model still compiles" is a
**default-on test** (`extractor_compiles_test.py`; `GEN3AI_SKIP_COMPILE_TESTS=1` opts out).

Full detail — the four guards, the Inductor crash root-caused to one op, and the startup-cost table —
is in `src/agents/training/CLAUDE.md` → Compiled CPU opponents.

### Compiled GPU trainer (`--compile-trainer`, opt-in)

The other half, and the bigger one. **Measured on v76 at the production shape** (batch 4096, PopArt
on, the real `MaskablePPO` path, arms interleaved on an idle box): `policy.evaluate_actions`
forward+backward **155.1 → 88.5 ms = 1.75×**, i.e. **~+62% end-to-end FPS** at the ~89% train share.
Compiling the whole policy instead of just the extractor measured the same to within 0.004×, so the
extractor is what ships — same win, less graph.

**CUDA only, and fail-loud by design.** A silent fall back to eager would be a 1.75× regression that
no metric surfaces (the run trains correctly, just ~38% fewer steps/hour, forever), so a failed,
slower, or numerically-divergent compile is a hard `FATAL_CONFIG` exit rather than a warning — and
`--device cpu` is refused up front, because the CPU backward provably does not lower (Inductor's C++
backend refuses the damage op's `atomic_add` scatter).

**⚠️ It drops the ObservationDebugger**, which dynamo cannot trace; that debugger is on in
production, so this is a real trade and it announces itself at startup. Full detail — the four
refusals, the `state_dict` hazard, the measurement table — is in `src/agents/training/CLAUDE.md` →
Compiled GPU trainer.

### Non-barrier async rollout (`--async-rollout`, opt-in)

Stock `SubprocVecEnv.step()` is a **barrier** — each step waits for the *slowest* of N env workers,
so the latency-bound rollout (py-spy: ~86% wall, GPU ~86% idle) is straggler-gated and the policy
forward never overlaps env stepping. `--async-rollout` swaps in `AsyncSubprocVecEnv` + an on-policy
`collect_rollouts_async` (`src/agents/training/async_vec_env.py`) that keeps every worker
continuously in-flight and forwards whichever envs are **ready**, filling each env's own buffer
column. It stays **exactly on-policy** (PPO freezes the policy during collection — a scheduling
change, not an APPO-style algorithm change). Masks ride natively in the Dict obs (`obs["action_mask"]`);
`env_method` is drain-safe so the eval callback's mid-collection pushes don't desync.
**Measured (bridge, GPU forward, steady-state FPS): +20% at n_envs=16 (=logical cores); +14% at the
production `--n-envs 64` (1489→1695); `--async-rollout --n-envs 32` matches production `sync@64` FPS
with half the envs** (≈half the RAM). Off by default (= stock `SubprocVecEnv`), ignored under
`--debug`. Full design: `designs/ai_v5/design_async_rollout.md`.

### Bot evaluation

Bot eval runs in **frozen-snapshot subprocesses** (`--eval-workers`, default 5) that
**work-steal at battle granularity** from a shared pool and play the live server (or the bridge)
**without pausing training**; results merge into TensorBoard + TUI + best-model and land in
`metadata.json` as a top-level `latest_eval` block. Each opponent's `EVAL_GAMES` are split into
**shard units** (`--eval-shard-games`, default 25 → 4 shards/opponent; per-opponent game count
overridable with `--eval-games`) so any idle worker drains a
straggler's remaining games instead of one worker grinding a whole opponent — the long eval tail
collapses to one shard. The mechanism lives in the well-encapsulated **`eval_sharding/` package**
(deep `ShardedEvalPool` interface; aggregation is **exact** — Σwon/Σfinished etc., raw δ pooled then
one CVaR), with a documented **`rating.py` seam** (`MatchRecord` / `RatingModel` / `BradleyTerryRating`)
ready for a future Glicko-2/TrueSkill without touching the live ELO path. **`--self-play` eval shares
this exact non-blocking pipeline** (with the worker pool doubled to 10, since sentinel matchups infer
for both players) — the workers additionally work-steal the pool sentinels' shards, and a winning
cycle promotes its frozen snapshot into the pool by file-copy (`SnapshotPool.add_from_path`). The full
design (battle-level work-stealing, exact aggregation, graceful-shutdown drain, resume re-publish,
sentinels + promotion, `--eval-workers` / `--eval-shard-games` / `--eval-device`) is in
`src/agents/training/CLAUDE.md`.

### ELO / skill rating

Under self-play pool play, `win_rate_vs_pool` is pinned near 50% by the promotion gate (a
sliding window of recent selves) and `win_rate_vs_bots` saturates — so neither tracks real
progress. The **ELO subsystem** fits an **anchored Bradley-Terry** rating over the win-records
every eval cycle already produces (trainee vs the 9 fixed bots + pool sentinels — no new
battles), giving one absolute number that rises with skill. Each cycle appends a row to an
append-only `<run>/eval_results.jsonl` and records a live `eval/elo` (+CI) to TensorBoard + a
`🏅 ELO` TUI badge. The fixed bots are the anchor; `python -m agents.training.bot_elo_calibration`
plays a one-time bot-vs-bot round-robin (bridge, no server) → `data/gen3_bot_elo_anchors.json`,
making snapshot ELOs **comparable across runs**. Offline: `python -m main.elo <run_dir>` prints a
ladder and plots an Elo-vs-step curve (and can backfill a running run from TensorBoard with
`--source tb`). Full design: `src/agents/training/CLAUDE.md` → ELO / skill rating.

🚨 **Reporting an ELO has three rules — read them before quoting a number.** The headline is
`<run>/snapshot_ladder/ladder.json` (dense, ±10) rather than `eval/elo` (±29); a rating is only
final once the run is, because BT re-solves every node on every add and the newest one is
**systematically inflated** (gen-10's 12M fell 2089 → 2021 over 12 refits); and a cross-run
comparison must be at matched snapshot COUNT, not matched step. The measured drift table and the
worked example are in `src/agents/training/CLAUDE.md` → *Reading an ELO*.

---

## Playing / Evaluation

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/play.py
```

Requires the Showdown server to be running (see below).

---

## Prober (forensic-replay inspector)

Browses the `eval_traces` a run writes and analyzes each saved decision point (faithfulness,
beliefs, threat tables, an intervention sweep, gradient saliency). No Showdown server needed — it
reads saved traces and a checkpoint, auto-discovering the trace tree and resolving the checkpoint
per battle (exact → nearest → recent).

**The interface is the browser.** The Textual TUI was retired 2026-08-13 — one analysis engine
deserves one renderer, and two meant every new signal had to be drawn twice for a single reader.
`python -m main.prober` starts the web app; its two TUI-only flags (`--ckpt`, `--inv`) explain what
replaced them rather than erroring.

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.prober models/run_<timestamp>
```

⚠ **A model-loading view only works on a run at the CURRENT architecture.** Measured over
`models/`: 79 of 79 archived runs cannot be re-loaded, so `analyze` / `lookahead` / `better-line` /
`replay-counterfactual` / `probe` return an `ArchDriftError` diagnosis there (naming the obs-dim and
`arch_signature` drift and the `git checkout` to re-probe from). Everything model-free — `scan`,
`triage`, `turns`, `falsify`, `calibration` — works on every run regardless. See
`src/main/prober/CLAUDE.md` → Architecture drift.

The same analysis is available headless for one invocation via the
`probe_replay.py` CLI (`python -m main.probe_replay <ckpt> <summary.json>
<states.npz> <inv>`); both share the pure engine in `src/main/prober/engine.py`.

**For agents/scripts**, a JSON API + CLI (`ProbeSession` / `python -m
main.prober.query summary|list|scan|awareness|overview|turns|find|analyze|lookahead|better-line|replay-counterfactual|falsify|falsify-scan|calibration`)
exposes the same probing infrastructure programmatically — list/filter battles, **`scan` the worst turn in
every loss across an opponent (model-free, ranked)**, digest one battle, **`turns` READ one battle as a
GAME — decisions grouped by game turn, each with the board it was made on, an ordered battle log of what
then happened (HP loss attributed to the move that dealt it), and V·ΔV·TD δ (model-free, ~20 ms)**, find
decisions the model disagrees with, deeply analyze one decision, **`lookahead`
a decision: one-ply VALUE-DELTA — re-roll each legal action one turn (opp plays its recorded move),
materialize the successor, and read the critic's V(s′) per action** (the model-scored counterpart to
`falsify`), **`better-line` a decision: SEARCH for a better trajectory — a shallow CRN-anchored BEAM
over the critic that branches a tree by CLONING mid-battle states (the warm `serializeBattle`
search-server, `utils/bridge/search_session.py`), returning ONE contrastive line ("at turn T do X
instead" + per-ply ΔV/ΔP(win)); opponent RECORDED at the divergence ply, reloaded policy at interior
plies; `--depth N`, `--confirm-rollouts N` for a rollout-to-end win-% confirmation; the depth-≥2
generalization of `lookahead`, bridge-eval traces only**, **`replay-counterfactual` a decision: substitute a move and play the rest LIVE vs the
RELOADED real opponent to a win/loss — "could it have won if it hadn't choked this turn?"** (a scripted
prefix over `run_local_battles`; `--rollouts N` resamples the post-divergence dice for a Monte-Carlo
win-prob ± Wilson CI; bridge-eval traces only), **`falsify`
a battle's worst decisions: luck-vs-mistake dice attribution by RE-ROLLING the
real turns** (fix-both luck percentile + paired alternative-action sweep via the
battle-reconstruction layer; bridge-eval traces with a `*_reconstruction.json`
sibling only), or **`falsify-scan` a whole run: aggregate that split
across every loss into a CRATER-FRACTION BRACKET** (|δ|-weighted — `aleatoric`=LUCK /
`unattributed`=NEUTRAL residual / proven `policy_reducible`=MISTAKE; an input to the
distributional-critic decision where `critic_headroom_upper_bound` = LUCK+NEUTRAL is an
explicit UPPER BOUND with `caveats`, not a measurement), or **`calibration` to split that
`unattributed` bucket** into `critic_overvalued` vs `lost_position` via recorded V(s) vs realized
return G(s) — a selection-aware reliability curve that self-diagnoses the eval-quota confound
(`bias_on_wins`/`bias_on_losses`). Internals
— engine/app split, the model-resolution ladder, Outcome panel, flags, and the
agent API — are in `src/main/prober/CLAUDE.md`.

### Web front end (`src/main/prober/web/`)

One of TWO surfaces over the engine (the JSON CLI is the other — neither is a layer on the other). Read-only browser views for the analyses a terminal renders worst, adapting
`ProbeSession` and nothing else: run summary · battles · `scan` · `triage` · the **turn-by-turn
battle replay** (`/battle` — board, what the model EXPECTED the opponent to do (the v67 α/β heads),
battle log and critic per game turn) · **`/analyze`** (one decision all the way down —
faithfulness, beliefs, threat tables, intervention, saliency; the ONE view that loads a checkpoint,
so it works on a current-architecture run and DIAGNOSES the drift on every other; reached from a `scan` row's
**turns** link, which lands on the losing turn) · the `falsify_scan` crater bracket · the
`calibration` reliability curve, and the counterfactual tier (`lookahead` / `better-line` /
`replay-counterfactual`) as password-gated background jobs off `/analyze`. **The Textual TUI is
retired** — one engine deserves one renderer, and two were costing every new signal twice.

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.prober.web models/   # :6008, pick any run
python -m main.prober.web --check-openapi        # the committed-contract drift gate
```

Point it at **`models/`** and the header carries a run picker. A request names a run by NAME, and
the name must be a member of the server's own listing — **no client string is ever joined to a
path**, so traversal is unrepresentable rather than filtered. A direct child of `models/` may be a
symlink and is followed (that is how the launcher's worktree runs appear); a symlink *inside* a run
refuses the run.

**Reading is anonymous; the two minutes-long probes (`falsify_scan`, `calibration`) need a shared
password** (`GEN3AI_PROBER_PASSWORD_FILE`, set at boot on this box). It fails closed — no password
configured means the probes are off, not open. `--open` is the laptop opt-out.

**Local only — it is NOT on g5d.io.** Unlike tensorboard (`:6006`) and the model viewer
(`:6007`), there is no systemd unit and no tunnel entry; it binds loopback on **6008** and you
start it when you want it. From another machine:
`ssh -p 2222 -L 6008:localhost:6008 goodlad@workstation.g5d.io`. What a `prober.g5d.io` deploy
would take is in `scripts/workstation/GCP_INFRASTRUCTURE.md` → *Prober web views*.

FastAPI + uvicorn (the minutes-long `falsify_scan`/`calibration` probes run as background jobs the
page polls, so they never block the event loop), server-rendered Jinja2 + HTMX (no build step, no
`node_modules` — the repo's root `package.json` still has zero dependencies), charts as
**Vega-Lite specs emitted from Python** (dicts: diffable, snapshot-testable). **All JS is
vendored**, never CDN-linked — the arch viewer's render test *skips* when its CDN is unreachable,
which makes the strongest gate in that suite a no-op offline; here headless chrome runs with every
non-loopback host mapped to a dead address, so a remote asset FAILS the test instead.

**Usable at desktop and phone widths**, gated as a MEASUREMENT rather than a CSS review: the page
publishes what it measured of its own layout (viewport width, whether the narrow breakpoint
matched, header height, control font size, and **which element** overflows if any) and the render
test reads that back at 1280px and at the 500px floor headless chrome allows. The rule the layout
serves is that the page never scrolls sideways — wide tables and charts scroll inside their own
box. **`/battle` is the one view that REFLOWS instead** (a turn is a narrative, not a row whose
columns carry the meaning): its boards stack on a phone and sit side by side on a desktop, and that
is measured too (`monstack`), not asserted. Detail — what the render record proves, plus the two
non-obvious traps (Vega does not wrap title text; a media query can lose on specificity) — is in
`src/main/prober/web/CLAUDE.md`.

---

## Showdown Server

> **⚠️ Never stop or restart the training Showdown server on port 8001.** A server started
> manually from a bash shell on **port 8001** (`npm run showdown -- 8001`) is the dedicated
> **training** server consumed by `main.launcher --showdown-port=8001`. Claude must NOT stop,
> restart, SIGTERM/SIGKILL, or `npm run stop -- 8001` it — and must not bounce it as a side
> effect of any other task — **unless the user explicitly asks in their current message.**
> Killing it mid-run drops every poke-env websocket at once (`ConnectionClosedError: no close
> frame received or sent` in `ps_client.listen`) and crashes training.
>
> **If Claude needs its own server, bind it to a unique port in the `9XXX` range** (e.g. 9001)
> that no other agent/session uses — never 8000 (dev) or 8001 (training). Pass it through
> (`npm run showdown -- 9001`, `--showdown-port=9001`). **Only ever kill the process on the
> port you started** — never `npm run stop` (no arg → kills :8000), never `npm run stop --
> 8001`, never a blanket node/showdown kill. Prefer the in-process bridge (no server) for
> throwaway work entirely.

Start the local Pokémon Showdown instance:

```bash
npm run showdown            # port 8000 (default)
npm run showdown -- 8001    # explicit port
```

Stop it:

```bash
npm run stop                # stops the :8000 instance
npm run stop -- 8001        # stops the :8001 instance
```

The server must be running for any battles (training or play). It binds to port 8000 by
default. Pokémon Showdown has **no `--port` flag** — the port is a positional argument
(`pokemon-showdown start [PORT]`); `npm run showdown -- 8001` forwards it (npm appends
`-- <args>` to the script).

### Separate training port

To run training without clashing with a development server on 8000, start the server on a
separate port and point the trainer at it with `--showdown-port`:

```bash
npm run showdown -- 8001                            # server on 8001 (dev stays on 8000)
... -m main.launcher ...                            # launcher DEFAULTS to :8001
... -m main.launcher --showdown-port 8123 ...       # explicit port still wins
npm run stop -- 8001                                # kills the 8001 instance
```

**The launcher defaults `--showdown-port` to 8001** (see `src/main/launcher/CLAUDE.md`) so a long
session never rides on the shared dev server; an explicit `--showdown-port` always wins.
`train_rl_agent.py` run directly still defaults to 8000. How a single `ServerConfiguration` is
built once and threaded to **every** Showdown client (env spawn-workers, eval, self-play) — and
the `server_port_threading_test.py` regression guard — is documented in
`src/agents/training/CLAUDE.md`.

---

## Repository Structure

```
src/
  agents/
    model/           # Gen3FeaturesExtractor (PyTorch feature extractor) — has CLAUDE.md
                     #   arch_constants.py — the weight-shape dims (single source of truth)
                     #   one file per phase group (extractor_ctx/encoders/team_transformer/pools/
                     #   belief_heads/aux_value_heads/pointer_head/value_readouts) + damage_op.py
                     #   (the physics) + damage_op_layout.py (the _DMG_* shape contract) — all
                     #   re-exported by features_extractor, so old imports still work
    gen3_data/       # The data facade: concept modules (moves/species/items/abilities/natures/
                     #   type_chart/priors) over data/ — single interface, poke-env-free — has CLAUDE.md
    observation/     # Observation encoders (state_encoder, pokemon, moves, etc.) — has CLAUDE.md
    action/          # Action mask + mapping via LegalActions: pure action_to_choice →
                     #   Choice → serialize.choice_to_order (the one poke-env order touch)
    training/        # Callbacks, reward manager, eval pipeline — has CLAUDE.md
                     #   elo.py (Bradley-Terry skill rating), bot_elo_calibration.py (anchor round-robin)
                     #   eval_sharding/ (battle-level work-stealing pkg), rating.py (Glicko-ready seam)
    battle/          # Event-sourced battle layer (Gen3Battle, BattleEvent log, TurnView,
                     #   LiveView/LegalActions read-models, StrictBattleView) — has CLAUDE.md
  main/
    launcher/          # Restart loop + Textual TUI (preferred for long runs) — has CLAUDE.md
                     #   core: checkpoint.py, worktree.py, child.py, input.py, state.py, ipc.py
                     #   UI: app.py + launcher.tcss · run loop: run.py · format.py · tui.py (alias)
    prober/            # Forensic-replay inspector (engine + session + JSON CLI) — has CLAUDE.md
                     #   web/ — browser front end (FastAPI + Jinja2/HTMX over ProbeSession) — has CLAUDE.md
                     #   engine.py (pure analysis), model.py, discovery.py, app.py
    tui/               # Shared Textual base (Gen3App, theme, colors) — launcher UI — has CLAUDE.md
    exit_codes.py      # TrainExitCode enum (COMPLETE=0, INTERRUPTED=15, CRASH=1, FATAL_CONFIG=3)
    train_rl_agent.py  # Training entry point (also callable directly)
    eval_worker.py     # Subprocess eval worker (frozen snapshot, CPU) — work-steals shard units
    probe_replay.py    # Forensic-replay CLI (thin wrapper over main.prober.engine)
    elo.py             # Offline ELO analyzer CLI (ladder + Elo-vs-step curve)
    play.py            # Battle / evaluation entry point
  poke_env/          # Forked poke-env library
  utils/
    git.py           # get_git_hash(), get_repo_root()
    (other utils)    # Hidden Power, teambuilder, team loader, logging
data/                # Source of truth — derived by tools/, read via agents.gen3_data
  pokemon/           # species/moves/items/abilities/type_chart/natures + smogon stats & priors
  teams/             # Downloaded sample teams (gen3ou pool)
  gen3_bot_elo_anchors.json  # Fixed bot ELO anchors (bot_elo_calibration.py round-robin); optional
models/              # Saved PPO checkpoints (run_<timestamp>/ subdirs)
                     #   <run>/checkpoints/  — periodic + forced checkpoints (.zip + .json sidecars)
                     #   <run>/best_model/   — best-by-eval export; <run>/snapshots/ — self-play pool
                     #   <run>/model_config.json, metadata.json, latest.txt, final_model*.zip — run root
                     #   latest.txt holds a run-RELATIVE path (e.g. checkpoints/checkpoint_123_steps.zip)
deps/
  pokemon-showdown/  # Git submodule — local Showdown server
tools/               # Acquisition layer (knows the 3 upstreams) — has CLAUDE.md
  pokemon_data_extractor/  # pokedex/Showdown/GenData -> data/pokemon/ reference files
  smogon_stats_downloader/ # Smogon usage stats -> data/pokemon/ priors
  sample_team_downloader/, others_team_downloader/  # -> data/teams/
```

---

## Observation Vector

**The full current layout — every block, offset and constant — is
[`designs/ARCHITECTURE.md`](designs/ARCHITECTURE.md) § Observation.** That file is derived from the
code and the live run config; this summary is the orientation only.

The observation is a flat **2501-dim float32 vector** (`Gen3ObservationEncoder.dimension`) plus an
11-dim `action_mask`, delivered as a Dict obs.

| Block | Dims | Offset |
|---|---|---|
| Our team (6 × `POKEMON_FULL_DIM` 122) | 732 | 0 |
| Opp team (6 × 122) | 732 | 732 |
| Active context ×2 (boosts 14 + volatiles `VOLATILE_DIM` 44) | 116 | 1464 |
| Global env (`GLOBAL_ENV_DIM`) | 20 | 1580 |
| Board (5 raw scalars + 12 active-req-moves) | 17 | 1600 |
| Pair history (6×6×5, Tier H-A2) | 180 | 1617 |
| Event window (32 × 22 typed event records, Tier H-B) | 704 | 1797 |
| **Total** | **2501** | |

⚠️ **This table was stale by two generations before 2026-08-17** — it still described a 2669-dim
vector after H-A (`gen3_pair_history_v1`) and H-B (`gen3_event_window_v1`) had added 852 dims
between them. Read [`designs/ARCHITECTURE.md`](designs/ARCHITECTURE.md) or
`Gen3ObservationEncoder.get_layout()` when the number matters; this summary is orientation.

`gen3_frame_deletion_v1`: the **7 × 159 TurnDelta lag frames and the 11-dim prev-turn action mask
are DELETED** — the event window is the last block, so `total_dim == base_dim` and the encoder's
output IS the observation (`Gen3Env.embed_battle` appends nothing). The licence was gen-13.5 §4:
the H-B event seats measured dV 2.7714 against the frames' 1.3015, so the frames were a second,
weaker copy of a job the seats already do. `TurnDelta` itself SURVIVES — it is the reward
manager's per-decision input and the α/β label source; only its obs encoding died. Four facts had no
substitute, found by a per-fact COVERAGE audit that a dV ablation cannot perform. Three are now
CLOSED — `cant_reason` (`EVENT_T_CANT` + col 19 `cant_id`), the eight faint causes (col 20
`faint_cause_id`), and the item-GONE family (col 21 `item_transition`, an enum: gen3 has three
such routes and a bare flag would leave the conflation half-alive). `our_attempted_switch_spec`
is knowingly ACCEPTED on value grounds — the rejection fact and trappedness both survive.

Every offset is computed from named constants in `agents/observation/constants.py` — **never
hardcode an index**; read `Gen3ObservationEncoder.get_layout()`.

`gen3_deadline_clock_v1`: the global block's CLOCK group is **3** scalars (`CLOCK_DIM`) —
log-ELAPSED plus remaining-LINEAR and log-REMAINING. `MAX_TURNS` (250) is also the forfeit
deadline (`StallConfig.threshold` imports it). The old single log-elapsed scalar had ~1.5% of its
range across the last 20 turns, so the critic had almost no resolution on the cap it actually
loses on — measured: a POSITIVE V on the final decision in 13 of 14 timeout losses. See
[`designs/ARCHITECTURE.md`](designs/ARCHITECTURE.md) § 1.4.

Per-block detail (the per-mon slot, the move slot, the TurnDelta fold, the embedded-ID manifest) and
the **mandatory obs-build performance gate** live in `src/agents/observation/CLAUDE.md`. Historical
prose about deleted blocks is preserved in [`designs/CHANGELOG.md`](designs/CHANGELOG.md).
---

## Feature Extractor Architecture

**The current phase chain, the dims that flow between phases, what each head consumes, the
`DamageOperator` block layout, the edge-family grid, and the production flag table are in
[`designs/ARCHITECTURE.md`](designs/ARCHITECTURE.md).** Read it before reasoning about the model —
it is the only document that describes the model as it is now rather than as it was at some version.

Orientation only:

- `Gen3FeaturesExtractor` (`src/agents/model/features_extractor.py`) is decomposed into named phase
  `nn.Module`s chained by a thin orchestrator — one file per phase group since 2026-08-16
  (`extractor_ctx` / `encoders` / `team_transformer` / `pools` / `belief_heads` / `aux_value_heads`
  / `pointer_head` / `value_readouts`, all re-exported by `features_extractor`). Most phases are
  flag-gated; which ones exist depends on the run config.
- It returns a **`(pi_features, vf_features)` tuple** and therefore MUST be paired with
  `Gen3DualHeadMaskablePolicy` (`policy.py`). A stock SB3 policy will not work.
- The action head is the **pointer head** — there is no flat `action_net`, and no flag to restore
  one. Each action is scored from the token of the entity it selects.
- Architecture constants (`D_MODEL`, `PROJECTION_DIM`, `N_HISTORY_TURNS`, …) live in
  `src/agents/model/arch_constants.py` and **nowhere else**.

The phase-by-phase contract (`ExtractorContext` / `Embeddings` ownership rules, what a phase may
own) is in `src/agents/model/CLAUDE.md`.

---

## Model Versioning

Every model save writes two **run-level** files at the run root:

- `model_config.json` — all weight-shape-relevant architecture params (embedding dims, layer sizes, obs dim, etc.)
- `metadata.json` — git hash, timestamp, SB3/Python versions (+ `snapshot_history`, `latest_eval`,
  and run provenance: `cli_args` [the latest process's full argparse namespace], `launcher_command`,
  and `original_command` — the **immutable** original invocation that created the model, written once
  at creation and preserved verbatim across every restart/checkpoint, unlike `cli_args` which is
  overwritten by the resuming process. Provenance lives ONLY here, never in `model_config.json`, which
  is the weight-shape/arch record used for `check_compatible`)

These are run-level (one per run), NOT per-checkpoint: a periodic checkpoint `.zip` lives one
level down in `checkpoints/` beside its own per-checkpoint `.json` sidecar, so
`load_model_snapshot()` searches the zip's dir **and its parent** (the run root) for
`model_config.json`. `load_model_snapshot()` in `src/agents/model/snapshot.py` checks it before
calling `MaskablePPO.load()`. A mismatch causes a hard `[ModelVersion] FATAL` error at startup,
not a silent wrong-output bug later. `model_config.json` additionally records `vf_coef` (`--vf-coef`,
the PPO value-loss coefficient): it is **fixed for a run's lifetime**, so resuming with a
different value is a FATAL error — enforced resume-only (frozen eval/pool/distill opponents are
exempt, since vf_coef doesn't affect a forward pass). See `src/agents/model/CLAUDE.md` →
resume-immutable training hparams. `_run_roundtrip_test()` in `train_rl_agent.py` runs automatically
before every `model.learn()` (save → reload → zero forward pass), so serialization breakage
crashes in seconds rather than hours.

The architecture-constant single source of truth is `src/agents/model/arch_constants.py`
(`ROLE_TOKEN_SIZE`, `PROJECTION_DIM`, `MOVE_NET_HIDDEN`, `ROLE_ENCODER_HIDDEN`, `ACTIVE_CTX_HIDDEN`,
`N_HISTORY_TURNS`, …); `ARCH_SIGNATURE` and `MODEL_CONFIG_VERSION` live in
`src/agents/model/model_version.py`. **Read the live values there, not from prose** — a version
number quoted in a document is stale the moment the next one lands.

- **Current architecture and flag state:** [`designs/ARCHITECTURE.md`](designs/ARCHITECTURE.md).
- **How every version got here (v16 → v59, verbatim):** [`designs/CHANGELOG.md`](designs/CHANGELOG.md).
  That file is HISTORY — its dims, defaults and measured percentages describe the config current at
  the time each entry was written, not today's. Do not quote it as current.
- **The versioning playbook** — what to do when you change a dim vs add an optional feature vs make
  a structural change, plus the resume-immutable-hparam rules — is in `src/agents/model/CLAUDE.md`.

When you land an architecture change: update `ARCHITECTURE.md` in the same pass, and append the
entry to `CHANGELOG.md`. Never narrate the new version inline here.

---

## Data Dependencies

**`data/` is the single source of truth — the runtime reads only `data/`, never live from
poke-env.** The split is **acquisition vs. access**: `tools/` (the only layer that knows the three
upstreams — poke-env static data, the Showdown source tree, Smogon usage stats) *derives* and
normalizes each file into `data/pokemon/`; the runtime reaches all of it through the
**`agents.gen3_data` facade**, blind to provenance (see `src/agents/gen3_data/CLAUDE.md`).

Reference data (deterministic) under `data/pokemon/`, all regenerable via
`tools/pokemon_data_extractor/sync.py`:
- `gen3_species.json` — species id → `{num, baseStats, name, types}` (`types` UPPERCASED to the TypeEncoder
  axis — `gen3_bidir_threat_trunk_v1`, for the op's expected-latent read; the obs still reads revealed types live).
  **419 rows** = the 386 base forms + the 33 gen-3 ALTERNATE/COSMETIC FORMES (`gen3_species_formes_v1`:
  Deoxys-Attack/-Defense/-Speed with their own base stats, the 27 Unown letters, Castform's 3 weather
  formes), each carrying `baseSpecies`. Formes were missing before and cost the `src/rust_sim` port
  **6.6% of gen3 random-battle teams / ~14% of battles** at construction. A forme SHARES its base's
  `num`, and the obs species channel + every `table[species.num]` model buffer are num-keyed — so
  num-indexed consumers MUST iterate `gen3_data.species.base_form_ids()` (see
  `src/agents/gen3_data/CLAUDE.md`)
- `gen3_moves.json` — move id → `{num, basePower, type, accuracy, never_miss, hasSecondary, hasRecoil,
  priority, secondaryEffects {col: percent}, drainFraction, recoilFraction, …}` (the structured
  secondary/priority/drain fields are `gen3_unified_move_system_v1` — GPU-side only, NOT in the obs vector).
  **Typed Hidden Power has distinct nums** (`gen3_typed_hidden_power_ids_v1`): bare `hiddenpower`=237,
  the 16 typed variants=355-370 (Showdown ships them all at 237; the extractor tool overrides — see
  `tools/CLAUDE.md`). OUR known HP uses the distinct num; the opponent's unrevealed HP is the bare 237.
- `gen3_items.json` — item id → `{num, name}` (`num` is the item-dex number; cross-gen aliases share one num)
- `gen3_abilities.json` — ability id → `{num, name}`
- `gen3_type_chart.json` — `{DEF: {ATT: multiplier}}` effectiveness chart (was live `GenData`)
- `gen3_natures.json` — nature → `{num, stat multipliers}` (was live `poke_env/.../natures.json`)
- `gen3_learnset.json` — species id → `[move_id, ...]` gen3 legal movepool (the hard legality gate the
  move-belief prior uses to prune impossible candidate moves; via `gen3_data.learnset`)
- `gen3_move_aliases.json` — `{alias_id: canonical_move_id}` from Showdown's `aliases.ts`
  (`wisp`→`willowisp`, `sd`→`swordsdance`, …). **Consumed ONLY by the `src/rust_sim` port** (its dex
  resolves a packed-team move alias like the RL runtime never touches); the `agents.gen3_data` facade
  does NOT load it, so it is obs-neutral. `gen3_move_alias_resolution_v1`.

Smogon-derived priors (probabilistic), via `tools/smogon_stats_downloader/` (`sync.py` merges 12
months of chaos JSONs → `compute_priors.py` derives six committed artifacts). **ALL priors must be
Smogon-derived; only the MODEL gets bias against the pool** (owner rule 2026-08-15): anything the
network READS must trace to Smogon (or ground-truth labels / ladder replays) — pool structure may
enter only implicitly, through training against pool opponents (team sampling / league targeting
are the sanctioned pool consumers). The 719-team pool may MEASURE structure (it is the only
set-level joint we own) but never ships as a prior:
- `gen3_smogon_stats.json` (raw aggregated chaos stats; per-species `Moves`/`Items`/`Spreads`/
  `Teammates` are 12-month summed counts) → `gen3_ability_priors.json`,
  `gen3_hidden_power_priors.json`, `gen3_move_priors.json`, `gen3_item_priors.json`,
  `gen3_spread_priors.json`, and `gen3_teammate_priors.json` — the chaos `Teammates` field
  normalized per species: the ONE species×species JOINT Smogon publishes (the hidden-team
  belief's coupling prior; `gen3_data.priors.teammates`). Note chaos `Moves` are per-species
  MARGINALS — within-species move-pair couplings exist in the data we can measure (pool) but
  have no Smogon source, so they stay with in-battle evidence + learning.

Pool-derived (a committed calibration artifact, same pattern):
- `data/teams/gen3_team_archetypes.json` — every pool team labeled by PACE class
  (hyper_offense/offense/balance/semi_stall/stall via a transparent composition rubric) + style
  TAGS (sand/spikes/spin/spinblock/phaze/**trap**/**trap_core**/wish/boom/choice/…), keyed by
  `sha1(team_str)[:10]` (the MatchupSpec `pin_sha` convention, so labels join every provenance
  record). Derived by `python -m agents.training.team_archetypes` (a k-means cross-tab prints as
  the unsupervised sanity check); consumed by league targeting (the `trap_core` exploiter
  shortlist) and future archetype-aware team sampling. Loader:
  `agents.training.team_archetypes.load_team_archetypes`.

All are loaded once (lazy singletons) and raise `FileNotFoundError` / `ValueError` if missing or
empty. The data layer is poke-env-free; the only poke-env touches left in the battle layer are a
parser sentinel (`GenData.UNKNOWN_ITEM`) and the `to_id_str` string util — neither is static data.

---

## Event-Sourced Battle Layer (`src/agents/battle/`, ai_v4)

poke-env is a **state tracker** (each `|...|` line overwrites "current board"); RL/reward/replay
need *what happened, in order*. The event-sourced layer (`Gen3Battle` + `BattleEvent` log) gives
that without reimplementing poke-env, and our non-`battle/` code reads battle state **only**
through the read-models it exposes — `LiveView` (current board), `TurnView` (history fold),
`LegalActions` (server-authoritative legality) — via the `StrictBattleView` boundary
(`battle.strict_view()`). The boundary is enforced by `src/agents/strict_api_lock_test.py` (the
lock) + the `src/agents/enums.py` seam. The per-decision `TurnDelta` history block folds entirely
from the event log (`build_from_events`) and feeds the obs turn-history.

**This is live and consumed across training; ai_v4 is closed out.** The full design — every
read-model, the injection seam, the per-decision event window, the TurnDelta fold's
value-identity findings, and the verification harnesses — is documented in
**`src/agents/battle/CLAUDE.md`** (as-built record: `designs/ai_v4/impl_step8_strict_battle_api_and_turndelta_fold.md`).
