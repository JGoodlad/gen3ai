# CLAUDE.md — Gen3AI Project Guide

> **This file is a CONSTITUTION, a COMMAND CARD and a MAP — nothing else.** It is loaded into every
> session, including every subagent's, so a line earns its place only if an agent that has NOT read
> it would do the work **wrong**. Detail lives in the leaf `CLAUDE.md` files and the `designs/` docs
> named throughout; **follow the pointer when you touch the subject.**

## Where the detail is

| I am about to… | Read first |
|---|---|
| run / resume / fork a training run | `designs/ops/training_runbook.md`, then `src/main/launcher/CLAUDE.md` |
| operate a live run (watch, kill, read) | `designs/ops/TRAINING_RUN_SOP.md` |
| write or tier a test, or run a benchmark | `designs/ops/testing.md` |
| reason about the model | `designs/ARCHITECTURE.md` **first**, then `src/agents/model/CLAUDE.md` |
| reason about the research | `designs/research_state/UNDERSTANDING.md`; `ledger.md` wins any disagreement |
| touch a training flag | `src/agents/training/CLAUDE.md` → its `designs/training/<topic>.md` |
| touch the rust port | `src/rust_sim/CLAUDE.md`, then its `designs/rust_sim/<topic>.md` |
| probe a run's traces / the prober | `src/main/prober/CLAUDE.md`, then its `designs/prober/<topic>.md` |

---

## Development Stage

**Rapid iteration — checkpoint compatibility is not a concern.** Breaking changes to the observation space, network architecture, or action space are fine. Do not add backwards-compatibility shims or hesitate to change dims, layer sizes, or layouts.

Architecture constants (embedding dims, layer sizes, etc.) are defined as module-level constants in `src/agents/model/arch_constants.py` — that is the single source of truth (`features_extractor.py` re-exports the whole block, so historical import paths still resolve). When you change one, change it there and nowhere else.

**Before reasoning about the model, read [`designs/ARCHITECTURE.md`](designs/ARCHITECTURE.md).** It is the only document that states the architecture *as it is now*. Version-numbered narrative lives in [`designs/CHANGELOG.md`](designs/CHANGELOG.md) and describes the past, not the present.

---

## 🚨 CRITICAL — running subagents / Workflows on this box

**Two rules. Both are required; either alone fails.** Measured 2026-08-09: two workflows returned
**0 results out of 5 agents and 0 out of 4** — 5.6M subagent tokens wasted — before this was understood.

1. **Pass `stallMs: 900_000` on EVERY `agent()` call in a Workflow script.** Workflow subagents have
   their own stall watchdog **hardcoded** in the Claude Code binary (3 min, 5 retries). There is **no
   env var and no `settings.json` key**. `stallMs` is Workflow-only — the `Agent` tool rejects it.
   ⚠️ `CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS` does **NOT** cover Workflow; it feeds the Agent-tool path.
2. **Cap concurrency to 1–2 agents — for TOKEN COST, not to prevent stalls.** A retry restarts the
   agent from scratch, so a four-attempt agent costs 4× tokens and one workflow phase burned ~3.5
   attempts per agent. ⚠️ The old "≥3 live streams starve the account" cause is **DOWNGRADED**
   (2026-08-11, 19,286 remote turns) — capping buys no stream capacity, and neither does `stallMs`,
   which only raises the kill threshold. Plain `Agent` calls still beat Workflow fan-out: **a
   stalled `Agent` is RESUMED with `SendMessage` to its agentId rather than redone.** Evidence and
   the stall mechanics: [`designs/ops/ORCHESTRATOR_SOP.md`](designs/ops/ORCHESTRATOR_SOP.md) §7
   (the measurement is in §2).
3. **Never let a script report agent ERRORS as "no findings".** `parallel()` returns `null` for a
   failed agent, so `findings.length === 0` is ambiguous — track failures and return a distinct
   status. Diagnose from `journal.jsonl`; the field is **`result`**, not `value`.

---

## Documentation Maintenance

Keep docs in sync **automatically, as part of the same change** — no need to be asked:

- **Every `CLAUDE.md`** (root and every directory leaf): always current. If a change makes one stale, fix it in the same pass.
- **`designs/ARCHITECTURE.md`**: always current, and the **first** thing an architecture change updates. It states only what is true now — no version numbers in prose, every measured figure carrying its provenance, every unverifiable claim marked `**UNVERIFIED:**`. Never narrate a change inline; state the new truth and delete the old.
- **`designs/CHANGELOG.md`**: append-only history. Add the new version entry in the same pass. Never edit or "correct" an existing entry — its job is to record what was believed at the time.
- **Every `README.md`**: always current. **Exception:** `designs/ai_v3/README.md` is a **frozen ai_v3 historical** digraph — do NOT update it for current-arch changes.

**Five `designs/` trees hold detail lifted OUT of a `CLAUDE.md`** (2026-09-07) and each OWNS what it holds — update it in the same pass as the code, exactly like a leaf: `designs/ops/` (the testing + training-runbook chapters, and the SOPs), `designs/training/` (the training leaf's topics), `designs/rust_sim/` (the port's topic docs — the module map, the gate-ladder rungs, the e2e capstone, the regression pins, the fuzzer findings, the mechanic classes, protocol emission — plus `port_build_log.md`, its closed coverage rounds), `designs/prober/` (the prober's topic docs — the analyze panels, the result timeline, the belief/threat views, the per-method session reference, the counterfactual probes, arch drift, the sim impl, the tests, and `/battle`'s field map). `designs/research_state/claude_md_archive/` is the exception — it is HISTORY, do not update it.

**Do NOT auto-update other docs under `designs/`** — `impl_step*.md`, `design_*.md`, `todo.md` are explicit-only (directly, or via `/gen3ai-update-design-docs`). The lone exception is `CLAUDE.md` files inside `designs/`, which follow the always-current rule.

**Leaf `CLAUDE.md` map** — read the leaf for the detail this root only summarises:

| Directory | Leaf covers |
|---|---|
| `src/agents/model/` | Feature-extractor phase contract, dual-head policy, architecture-constant rules, model versioning |
| `src/agents/gen3_data/` | The data facade over `data/`, the acquisition-vs-access split, and every per-file schema |
| `src/agents/observation/` | Obs-build performance gate (mandatory benchmark) + the full per-block obs layout |
| `src/agents/battle/` | Event-sourced battle layer (Gen3Battle, BattleEvent log, LiveView/TurnView/LegalActions, StrictBattleView, TurnDelta fold) |
| `src/agents/training/` | The training hub — each topic keeps its heading + summary there and its detail in `designs/training/<topic>.md` |
| `src/rust_sim/` | The Rust Showdown port: the module map, the conventions, the differential-gate ladder and how to run each rung, the four A/B fuzzers and their green-gate allowlists, the search/replay drivers, and the standing lessons — detail in `designs/rust_sim/` |
| `src/main/launcher/` | Launcher internals: restarts, crash reporting, exit codes, flags, port default |
| `src/main/prober/` (+ `web/`) | Forensic-replay inspector: the analysis ENGINE, the `ProbeSession` facade, the JSON CLI, and the hazards that have cost a wrong reading; the detail is its `designs/prober/<topic>.md`. `web/` is the browser front end |
| `src/main/tui/` | Thin shared Textual base — the LAUNCHER's UI (the prober's TUI is retired) |
| `designs/` | Which `ai_vN` folder is relevant; the version map |

**Non-`CLAUDE.md` docs under the same always-current obligation:**

| File | Holds |
|---|---|
| `designs/ARCHITECTURE.md` | **What is true NOW about the MODEL** — obs layout, phase chain, per-head inputs, op block, edge families, the production flag table with `INERT` markings |
| `designs/research_state/UNDERSTANDING.md` | **What we BELIEVE NOW about the RESEARCH** — era map, meters, open questions with their tests, standing rules of evidence. Every claim carries an evidence tag and a pointer. Its append-only counterpart `designs/research_state/ledger.md` **wins any disagreement** — fix the view, never the ledger |
| `designs/ops/TRAINING_RUN_SOP.md` | **How a run is OPERATED** — pre-launch checks, the four watch layers, the 55-min fallback cron, kill/relaunch, the read. `designs/ops/ORCHESTRATOR_SOP.md` is the orchestrator session's |
| `designs/CHANGELOG.md` | **How it got here.** History; do not quote as current |

---

## Git Workflow

Personal project — no pull requests. Work is pushed directly to `main`, but **all edits and commits must happen in a worktree or branch, never on the main checkout itself.** Main must never be dirty. The `/gen3ai-ship` skill is the only mechanism that lands code on main:

```bash
git push origin <worktree-branch>:main
```

Never `git add` or `git commit` from `/home/goodlad/dev/gen3ai` directly.

🚨 **NEVER run `git add`, `git commit`, or `git push` unless the user's current message explicitly contains `/gen3ai-ship`.** Completing a task, writing tests, or any other finishing signal is NOT permission to commit. This applies even when the task feels "done".

---

## Python Environment

`./scripts/bootstrap.sh` does all setup idempotently — conda env, submodule, Showdown build, worktree symlinks, the optional cargo build — and verifies with the two static gates plus a ~10 s smoke. `--dry-run` prints the plan. `CONTRIBUTING.md` is the human version.

The env is **`gen3ai_stable`** (not `deps/venv`, which is outdated — ignore it):

```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 <script>
```

🚨 **IN A GIT WORKTREE THE `export` IS MANDATORY.** `pip install -e .` names ONE absolute path — the main checkout's `src/` — so a worktree that runs `pytest` with no `PYTHONPATH` collects *its own* test files and imports *main's* code. Every result is then about a tree you did not edit. `src/packaging_gate_test.py` catches it, but the fix is to export, every time. Optional means optional **in the main checkout**.

**The ORDER between the two mechanisms is load-bearing.** `PYTHONPATH` lands in `sys.path` *before* site-packages; an editable install's `.pth` lands *after*. That is what lets the launcher pin a resumed run to its checkpoint's commit. Install from the **main checkout only** — a `.pth` made in a worktree points at a directory that later gets deleted, and Python skips it in silence.

🚨 **`poke-env` is deliberately ABSENT from `environment.yml`.** We vendor the fork at `src/poke_env/`. A second installed copy makes `import poke_env` depend on `sys.path` ORDER — and the failure is **silent**: upstream imports cleanly and behaves subtly differently. `src/poke_env_fork_gate_test.py` guards it. (`environment.yml`'s `--extra-index-url .../cu121` is load-bearing too: the torch pins are local-version builds not published on PyPI.)

**That absolute path is THIS box's env, not a requirement.** Every process the project spawns runs under `sys.executable`, so elsewhere `conda activate gen3ai_stable && python <script>` is enough. The launcher takes **`$GEN3AI_PYTHON`** to pin its child's interpreter.

Detail: `designs/research_state/claude_md_archive/pythonpath_archaeology.md`.

---

## Git Worktree Setup

**`./scripts/bootstrap.sh` does this automatically.** The manual recipe is what you need when the script is not what went wrong:

```bash
git submodule update --init                    # 1. source files (+ fixes VS Code git integration)
for n in dist node_modules; do                 # 2. borrow main's build artifacts
  # GUARD: only link when the NAME does not already exist. See the warning below.
  [ -e "deps/pokemon-showdown/$n" ] || \
    ln -s "/home/goodlad/dev/gen3ai/deps/pokemon-showdown/$n" "deps/pokemon-showdown/$n"
done
```

> 🚨 **Run these ONLY from a fresh worktree, and keep the `[ -e ]` guard.** From the MAIN checkout `dist` already exists as a real directory, so `ln -s TARGET dist` puts the link *inside* it as `dist/dist` → pointing at its own parent. `node build` then dies with `ELOOP` and **every websocket-server path stops working**. That happened on 2026-07-23 and went unnoticed for four weeks, because training runs on the in-process bridge and nothing else starts a server. Fix: `rm deps/pokemon-showdown/dist/dist`, then `node build`.

Without step 2, training fails with `Cannot find module '.../dist/sim/index.js'`. The root `conftest.py` probes for both at session start and REFUSES the run with one message naming the fix (`GEN3AI_SKIP_DEPS_GUARD=1` opts a pure-unit CI out). Do **not** symlink the entire `deps/pokemon-showdown` directory — git then treats the submodule path as a symlink and `git status` breaks.

---

## Running Tests

**Full chapter — tiers, the four gates, contention, fuzz, benchmarks: [`designs/ops/testing.md`](designs/ops/testing.md).** The card:

| When | Command (prefix each with `export PYTHONPATH=$PYTHONPATH:src &&`) |
|---|---|
| **inner loop** — fastest true/false | `python3 -m pytest src/ -m "not slow and not e2e and not sim and not integration" -q -n 2` |
| **THE ROUTINE GATE — before a commit** | `python3 -m pytest src/ -m "not slow and not e2e" -q -n 2` |
| **before `/gen3ai-ship`, and in CI** | `python3 -m pytest src/ -q` *(~31 min, ~24 of it browser)* |
| just the bridge / just the browser | `-m sim` *(~100 s)* / `-m browser` *(~24 min)* |

Use `-n 2` (~1.8x, two cores) — a training run normally shares this box; `-n 4` when the box is yours. Serial when you need `-s` or a debugger.

🚨 **Do NOT use the old `-m "not integration and not e2e"`.** `integration` now spans a ~100x cost range, so excluding it throws away cheap high-value coverage — that is how the obs-golden linchpin rode main RED three separate times. **Cut on `slow`**, the marker that means "expensive".

**Two axes, and keeping them apart is the point.** A marker says what a test NEEDS (*(unmarked)* · `integration` · `sim` · `browser` · `e2e`); a separate marker says what it COSTS (`slow`). **A tier is DECLARED, never inferred** — cost arrives transitively, so no filename or import graph can classify a test. `conftest.py` reports an unmarked test that overruns 30 s, and **enforces only on a quiet box** (factor < 1.05); on a busy one it is advisory, because a duration measured under starvation is not a measurement.

**Seven static gates, all unmarked (they run in every tier), all ~free.** A missing tool FAILS rather than skips — a linter that silently opts out reads exactly like one that found nothing.

| Gate | Checks | Opt-out |
|---|---|---|
| `src/agents/model/mypy_gate_test.py` | `python -m mypy`, scope from `mypy.ini`'s `files =` | `GEN3AI_SKIP_MYPY_GATE=1` |
| `src/ruff_gate_test.py` | `ruff check … --select F,E9` — wrongness only, never style | `GEN3AI_SKIP_RUFF_GATE=1` |
| `src/file_size_gate_test.py` | >2,000 lines FAILS; 1,000–2,000 reported. **The allowlist is EMPTY — a new entry is not a legal move; decompose it** | `GEN3AI_SKIP_SIZE_GATE=1` |
| `src/claude_md_freshness_gate_test.py` | every repo-relative path and every `--flag` in a `CLAUDE.md` resolves | `GEN3AI_SKIP_CLAUDE_MD_GATE=1` |
| `src/test_stub_vacuity_gate_test.py` | every `monkeypatch.setattr` / `patch` / `mod.x = stub` target under `src/**/*_test.py` is a symbol the code under test actually READS — **a stub that stubs nothing FAILS**. **The allowlist is EMPTY**; fix at the source | `GEN3AI_SKIP_STUB_GATE=1` |
| `src/slow_tier_status_gate_test.py` | the last recorded verdict of every `slow` test (`designs/ops/slow_tier_status.json`, written by the slow tier itself). **A recorded FAIL fails the ROUTINE gate**, naming the test and the commit it failed at; inconclusive (a timeout), unrecorded and stale are REPORTED, never fatal | `GEN3AI_SKIP_SLOW_STATUS_GATE=1` |
| `src/mode_flag_doc_gate_test.py` | every MODE-flag value `designs/ARCHITECTURE.md`'s PROSE states equals `designs/production_config.json` (read via `agents.training.baselines.production_config()`), and every key the mirror marks INERT is called INERT. The (doc pattern → key) table is DECLARED, so a renamed key FAILS instead of going quiet | `GEN3AI_SKIP_MODE_FLAG_DOC_GATE=1` |
| `src/ledger_index_gate_test.py` | `designs/research_state/ledger_index.md` (the generated date · line · title index over the 13.8k-line ledger) matches what `python -m main.ledger_index` renders — an entry appended without a regeneration FAILS here. **On a rebase conflict take either side and re-run the generator; never hand-merge it, and never edit the ledger** | `GEN3AI_SKIP_LEDGER_INDEX_GATE=1` |

A path or flag named deliberately as HISTORY goes in `designs/deleted_flags.md` with its citation.

🚨 **A `slow` TEST IS DESELECTED BY THE ROUTINE GATE, AND A DESELECTED TEST CANNOT FAIL.** That is how `tb_relevance_test`'s winprob smoke rode main RED for a day (2026-09-07) — the same shape as the obs-golden linchpin, one marker further out. So the slow tier now WRITES its verdict and the routine gate READS it: any run in which a `slow` test executes merges that test's row into **`designs/ops/slow_tier_status.json`** (a COMMITTED artifact — a gitignored one would be per-worktree, i.e. absent exactly where the work happens), and the gate above turns a recorded FAIL into a routine-gate failure. Refresh the whole file with `python3 -m pytest src/ -m slow -q -n 2`. ⚠️ **The honest limit: a slow test that broke SINCE the last recorded run still reads green** — nothing but running the tier closes that, which is what the staleness report is for. Detail: `src/utils/slow_tier_status.py`.

🚨 **A DECOMPOSITION MOVES SYMBOLS OUT FROM UNDER THE STUBS THAT NAME THEM.** A patch on `mod.func` reaches the code under test only if `mod` still READS `func` at call time — a consumer that wrote `from mod import func` holds its own copy and the stub reaches nothing, so the test asserts about the real path and passes for the wrong reason. `ccd08003` created four such sites in one commit; two were `mod.name = stub` assignments, which raise NOTHING at runtime. The stub gate above is standing for exactly this, and it is blind to a target named through a LOOP VARIABLE — spell the module and attribute out.

🚨 **A TIMEOUT IS NEVER A SEMANTIC OUTCOME.** The box normally carries a production run, so bounds scale by measured contention (`src/utils/contention.py`; the factor is exactly 1.0 on an idle box). A run whose timeouts exceed 25% of attempted battles is INCONCLUSIVE, not reported. **Benchmarks get the opposite treatment — warn, never stretch**: a benchmark's output IS the measurement. `GEN3AI_TIMEOUT_SCALE=6` forces the factor; run the suite under it after touching any of this.

🚨 **A fuzz SCRIPT wants a new battle every run; a pytest-collected TEST wants the same battle every run.** A collected test takes its battle from `obs_roundtrip_fuzz_test.record_fixture_battle(...)` and **asserts its precondition rather than branching on it** (`if x is not None:` around the decisive gate fails green). `random.seed(k)` is NOT enough — reproducibility needs fixed teams, a per-player RNG, a fixed sim seed, **and `concurrency=1`** (at concurrency 3, two runs of the same measurement differed by up to +0.043). Prefer REFUSING an unreproducible configuration over emitting a quietly-wandering number.

**File naming:** `*_test.py` (pure unit) · `*_integration_test.py` (out-of-process dep; **the name no longer implies the tier — read the `pytestmark`**) · `*_fuzz_test.py` (real battles in-process via the bridge, run as scripts) · `*_e2e_test.py` / `*_fuzz_e2e_test.py` (live server) · `*_benchmark.py` (profiling, run as scripts).

```bash
export PYTHONPATH=$PYTHONPATH:src
# a fresh worktree pays for a cargo build on its first rust test — build first or discount that run
cargo build --release --bin sim_bridge --bin search_driver --manifest-path src/rust_sim/Cargo.toml
python3 src/agents/action/fuzz_test.py [n_battles]            # + many more: see designs/ops/testing.md
python3 src/agents/training/obs_build_benchmark.py            # 🚨 MANDATORY before/after any obs change
python3 src/agents/training/trainer_turn_benchmark.py --pin-battles  # 🚨 --pin-battles for ANY A/B claim
python3 src/agents/training/live_view_build_benchmark.py      # the only one that can A/B LiveView.from_battle
```

---

## Smoke Test

Verify the core pipeline before a full run (~1 min). `--debug` defaults to **CPU**, skips all eval, and is **serverless** (`--use-bridge` defaults to `rust`); add `--debug-eval` to exercise the eval path.

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py --debug --steps 10000
```

Look for `[ModelVersion] Round-trip smoke test PASSED`, `🏁 Episode Finished` throughout, and `Training complete`. A hang after `[STALL LOGGED]` or a crash before completion is a regression in the env/stall/forfeit pipeline; `[ModelVersion] FATAL` means the checkpoint's architecture ≠ current code.

---

## Training + the Launcher

**Full chapter — every flag, the bridge transport, the compile flags, `--critic`, eval, the meters: [`designs/ops/training_runbook.md`](designs/ops/training_runbook.md).** Per-flag semantics: `src/agents/training/CLAUDE.md` and `src/main/launcher/CLAUDE.md`.

`src/main/launcher/` wraps `train_rl_agent.py` with periodic restarts, crash auto-restart, git-worktree isolation, a Textual TUI and live crash-log streaming. **Prefer it for long runs.** A detached launch (`nohup … < /dev/null &`) runs HEADLESS automatically. Everything runs at `--nice 10` by default.

```bash
export PYTHONPATH=$PYTHONPATH:src
# validate an argv OFFLINE — does it launch, AND is it the architecture you meant?
python -m main.checkargs models/<run>          # or --argv "…"
# resolve the ACTUAL launch on this box without creating anything (SAFE on a restart)
python -m main.launcher --dry-run …
# fresh run
python -m main.launcher --restart-interval-hours 3 --steps 15000000 --n-envs 64 \
  --batch-size 16384 --n-epochs 10 --ent-coef 0.02 --n-steps 2048 --lr 0.0003 \
  --device cuda --log-level periodic --arch production
# resume / fork (all non-launcher flags forwarded verbatim)
python -m main.launcher --restart-interval-hours 3 --model models/<run>/checkpoints/<ckpt>.zip \
  --steps 15000000 --device cuda
```

**The hazards that have actually cost runs — none of these are theoretical:**

- 🚨 **"It launches" and "it is the experiment" are INDEPENDENT checks.** On 2026-09-06 an arm launched from a design-doc command block with every architecture flag at its OFF default and trained a near-bare network for 24.4M steps; three gates passed and all three were right. **`--arch production`** applies the production surface as if typed. `checkargs` prints an ARCH SURFACE diff and refuses a fresh argv that differs.
- 🚨 **A BARE RUN DIRECTORY MEANS THE RUN'S LAST SNAPSHOT** (`resolve_model_ref`) — for `--distill-teacher`, `--stable-opponents`, `--exploiter`, `--win-prob-pbrs-source` and friends. Name the `.zip` or `@step` to pin a file.
- 🚨 **AN ARGV IS NOT A CONFIG.** With `--model`, every flag you do not name is INHERITED from the checkpoint's `model_config.json`.
- 🚨 **`--lr`, `--batch-size`, `--n-steps` and `--gamma` are INERT on a resume** — SB3 restores the checkpoint's own values, so a FORK inherits whatever the parent's KL controller had annealed to. **`--fork-lr`** pins it; the quantity that predicts a fold's collateral is the **DOSE** (`lr × n_epochs / (batch_size × grad_accum_steps)`), read with `python -m main.dose <run>`.
- 🚨 **A FORK starts with an EMPTY self-play pool, and an empty pool does not disable `--self-play` — it falls back to the BOT pool.** A genuine fork now auto-seeds its parent's pool and exits `FATAL_CONFIG` if it still has none.
- 🚨 **A PINNED argv is judged by the PINNED commit's parser** — a flag whose ARITY changed is invisible to a presence check.
- ⚠️ Do not "launch the real command and kill it" to validate — harmless on a fork, **DESTRUCTIVE on a restart**. Use `--dry-run`.

**Defaults worth knowing:** `--use-bridge` is **`rust`** (serverless — no Showdown server needed for training or eval); `--compile-opponents` and `--compile-trainer` are **ON** (the latter auto-on for cuda, and it **drops the ObservationDebugger**); `--critic` is `shaped`. Checkpoints land in `models/run_<ts>/checkpoints/`.

**Offline meters** (no training, nothing written under `models/`): `main.elo` · `main.untaught_meter` · `main.critic_gate` · `main.exploitability` · `main.scaffolding_gauge` · `main.capacity` · `main.lineage` · `main.dose` · `main.sidecar_audit` · `main.baselines` · `main.tb_curate`.

**LIVE-run instruments are a different tier — `main.ops.*` and `scripts/ops/`**, and reading a live
arm through an offline meter is not the same operation: they read TensorBoard events, the launcher
child log and checkpoint mtimes while all of those are still being appended to, and REFUSE when a
precondition of the live read is unmet. `designs/ops/TRAINING_RUN_SOP.md` §2 names which layer uses
which; `scripts/ops/README.md` lists every one.

🚨 **Reporting an ELO has three rules:** the headline is `<run>/snapshot_ladder/ladder.json` (dense, ±10), not `eval/elo` (±29); a rating is only final once the run is (the newest BT node is systematically inflated); and a cross-run comparison must be at matched snapshot **COUNT**, not matched step.

🚨 **`win_rate_vs_pool` / `eval/elo` carry an OPPONENT-REGIME BOUNDARY at 2026-09-07** and are not comparable across it. Eval pool sentinels are now **GREEDY and draw the trainee's own teams** by default (`--no-eval-sentinel-greedy` opts out; `--promote-threshold` follows, 0.55 / 0.65). The old asymmetry was worth **+8.9 pp** to the trainee, so equal skill reads ~9 pp lower now. **The regime is RECORDED and INHERITED on a flagless resume** — read it (`model_config.json`'s `eval_sentinel_greedy`, or the launch's `⚖️  [EVAL REGIME]` line), never assume it. `ladder.json` and every bot edge are UNAFFECTED. Detail: `designs/training/eval_and_rating.md`.

---

## Playing / the LADDER

`src/main/play.py` is the only entry point that talks to a Showdown server **as a client**, so it is the exact path a rated ladder game uses — modes `selfplay` / `challenge` / `accept` / `ladder`.

```bash
export PYTHONPATH=$PYTHONPATH:src
python3 src/main/play.py --mode selfplay --port 9017        # 8000/8001 are REFUSED in code
PS_PASSWORD=… python3 src/main/play.py --mode ladder --server official \
  --model models/<run>/final_model.zip --username <acct> --n-battles 20
python3 src/main/ladder_drift_scan.py --n 200               # 🚨 RUN BEFORE ANY LIVE SESSION
```

🚨 **NO `--proxy` for laddering** — Showdown auto-locks accounts on datacenter/VPS/proxy IPs (`#hostfilter`), and the GCP tunnel's egress is exactly that class. 🚨 **Run the drift gate first**: `deps/pokemon-showdown` is pinned, the public server runs master, and `battle_event.classify` raises on an unknown keyword **by design** — on a live battle that kills the parse task, sends no choice, and loses on the timer. Full audit: [`designs/research_state/ladder_readiness.md`](designs/research_state/ladder_readiness.md).

---

## Prober (forensic-replay inspector)

Reads a run's `eval_traces` + a checkpoint and analyzes saved decision points. No server. **Two surfaces over one engine** — the browser app and the JSON CLI (`python -m main.prober.query summary|scan|turns|analyze|lookahead|better-line|replay-counterfactual|falsify|falsify-scan|calibration|loops`). The Textual TUI is retired.

```bash
export PYTHONPATH=$PYTHONPATH:src
python3 -m main.prober models/run_<timestamp>     # browser app
python3 -m main.prober.web models/                # :6008, run picker; also at prober.g5d.io
```

⚠️ **Model-loading views only work on a run at the CURRENT architecture** — they return an `ArchDriftError` diagnosis elsewhere. Model-free commands (`scan`, `triage`, `turns`, `falsify`, `calibration`) work on every run. 🚨 **The trace quota PREFERS LOSSES**, and each cycle's `eval_manifest.json` records the selection — a tree that records none is SELECTION UNKNOWN, never uniform. Detail: `src/main/prober/CLAUDE.md`, `src/main/prober/web/CLAUDE.md`.

---

## Showdown Server

> 🚨 **Never stop or restart the training Showdown server on port 8001.** A server on **8001** is the dedicated **training** server. Claude must NOT stop, restart, SIGTERM/SIGKILL or `npm run stop -- 8001` it — and must not bounce it as a side effect of any other task — **unless the user explicitly asks in their current message.** Killing it mid-run drops every poke-env websocket at once and crashes training.
>
> **If Claude needs its own server, bind it to a unique port in the `9XXX` range** — never 8000 (dev) or 8001 (training). **Only ever kill the process on the port you started**: never bare `npm run stop` (kills :8000), never `npm run stop -- 8001`, never a blanket node kill. Prefer the in-process bridge (no server) for throwaway work.

```bash
npm run showdown            # port 8000 (default) — Showdown has no --port flag; it is positional
npm run showdown -- 8001    # explicit port
npm run stop -- 8001        # stops that instance
```

The **launcher defaults `--showdown-port` to 8001**; `train_rl_agent.py` run directly defaults to 8000. With the default `--use-bridge rust` no server is involved at all.

---

## Repository Structure

Top two levels only — each directory's own `CLAUDE.md` (see the leaf map) is the detail.

```
src/
  agents/
    model/           # Gen3FeaturesExtractor + the phase modules, damage_op, belief/dex tables,
                     #   critic_mode, capacity_probes, model_version/ — has CLAUDE.md
    gen3_data/       # The data facade over data/ (poke-env-free) — has CLAUDE.md
    observation/     # Observation encoders — has CLAUDE.md
    action/          # Action mask + mapping via LegalActions
    battle/          # Event-sourced battle layer (Gen3Battle, TurnView, LiveView) — has CLAUDE.md
    training/        # Callbacks, reward, eval, distillation, cf grounding, meters — has CLAUDE.md
  main/
    launcher/        # Restart loop + Textual TUI — has CLAUDE.md
    ops/             # LIVE-run instruments: tb_read, killbar, g7_report, plateau_signal…
                     #   the shell half is scripts/ops/ — see scripts/ops/README.md
    prober/          # Forensic-replay inspector (+ web/) — has CLAUDE.md
    train/           # The training entry point's phases (parser/, config, combination_checks)
    search_dividend/ # Search-around-the-policy probe + its 4-arm battery
    tui/             # Shared Textual base — has CLAUDE.md
    *.py             # The offline CLIs: elo, dose, lineage, baselines, critic_gate,
                     #   untaught_meter, exploitability, scaffolding_gauge, capacity,
                     #   checkargs, sidecar_audit, tb_curate, tb_inherit, play, promote_teams,
                     #   ledger_index
  poke_env/          # Forked poke-env library (vendored — see the Python Environment warning)
  rust_sim/          # The Rust Showdown port — has CLAUDE.md
  utils/             # paths.py (path discovery), git.py, bridge/, teambuilder, logging
designs/             # ARCHITECTURE.md, CHANGELOG.md, baselines.json, production_config.json,
                     #   ops/, training/, rust_sim/, research_state/, ai_vN/ — has CLAUDE.md
data/                # Source of truth — derived by tools/, read via agents.gen3_data
models/              # Saved runs (NOT committed; exists only in the MAIN checkout)
deps/                # pokemon-showdown git submodule
tools/               # Acquisition layer (knows the 3 upstreams) — has CLAUDE.md
```

### Path discovery — `src/utils/paths.py`

🚨 **Never hand-write `Path(__file__).resolve().parents[N]` to find the repo root.** That arithmetic lives in **one** module, cross-checked against `git rev-parse`.

| You want | Use | Mechanism |
|---|---|---|
| the checkout this code came from (`data/`, `designs/`, `deps/`) | `repo_root()` / `repo_path(*parts)` | `__file__` |
| `src/` (the import root) | `src_root()` / `src_path(*parts)` | `__file__` |
| the **run archive** `models/` | `main_models_dir()` → `Path` **or `None`** | `git` |
| git's opinion / the HEAD hash | `utils.git` | `git` |

🚨 **`models/` is the one that bites.** It is not committed and exists only in the **MAIN checkout** — `repo_root()` inside a worktree is the *worktree*, which has none. `main_models_dir()` reaches across via git's shared common dir and returns `None` when there is no archive, which every caller must turn into a **skip**. `$GEN3AI_MODELS_DIR` overrides and is authoritative (set-but-missing ⇒ `None`, never a quiet fall-back).

`paths_test.py` AST-scans `src/agents`, `src/main`, `src/utils` and fails any module using a `/home/…` literal as a value. **When `__file__`-relative is still right:** a module locating a file that ships *beside it* is not doing repo-root discovery.

---

## The Model — obs, extractor, versioning, baselines

**[`designs/ARCHITECTURE.md`](designs/ARCHITECTURE.md) is the only document that states the model as it is NOW.** Read it before reasoning about the model. Orientation:

- The observation is a flat **2501-dim float32 vector** + an 11-dim `action_mask`, as a Dict obs. 🚨 **NEVER hardcode an index** — read `Gen3ObservationEncoder.get_layout()`; every offset comes from `agents/observation/constants.py`. (This root's dim table was stale by two generations before 2026-08-17. Ask the code.)
- `Gen3FeaturesExtractor` returns a **`(pi_features, vf_features)` tuple** and MUST be paired with `Gen3DualHeadMaskablePolicy`. A stock SB3 policy will not work.
- The action head is the **pointer head** — there is no flat `action_net` and no flag to restore one.
- Architecture constants live in `src/agents/model/arch_constants.py` and **nowhere else**; `ARCH_SIGNATURE` / `MODEL_CONFIG_VERSION` in `src/agents/model/model_version/`. **Read the live values from the code — a version number quoted in prose is stale the moment the next one lands.**

**Every save writes two run-level files:** `model_config.json` (the weight-shape/arch record, checked by `check_compatible` — a mismatch is a hard `[ModelVersion] FATAL` at startup) and `metadata.json` (provenance). Three **IMMUTABLE** blocks in the latter, each written once and preserved verbatim: `original_command`, `lineage` (who forked whom, with the file every reference RESOLVED to), and `pin_history` (which commit ran which steps). Read them through `agents.training.lineage` / `main.lineage` / `main.sidecar_audit`, never by re-deriving. `vf_coef` is fixed for a run's lifetime.

🚨 **BASELINES are NAMED, and read by name** — `designs/baselines.json` + `agents.training.baselines` (`production`, `v9_long_baseline`, `v9_fold_parent`, `famine_comparator`, `untaught_meter_opponent`, …). Never copy a path. Every entry pins an EXPLICIT checkpoint so the last-snapshot rule cannot move it; a registry name survives every retention tier; `python -m main.baselines set … --reason` is the only way to change one, and it prints the ledger line to append.

**`models/` retention** is policy, not habit: [`designs/research_state/models_retention_policy.md`](designs/research_state/models_retention_policy.md) plus its dry-run tool. Nothing has been deleted.

When you land an architecture change: update `ARCHITECTURE.md` in the same pass, and append to `CHANGELOG.md`.

---

## Data Dependencies

**`data/` is the single source of truth — the runtime reads only `data/`, never live from poke-env.** The split is **acquisition vs. access**: `tools/` (the only layer that knows the three upstreams) derives each file into `data/pokemon/`; the runtime reaches it through the **`agents.gen3_data` facade**, blind to provenance.

🚨 **ALL priors must be Smogon-derived; only the MODEL gets bias against the pool** (owner rule 2026-08-15). Anything the network READS must trace to Smogon, ground-truth labels or ladder replays. Pool structure may enter only *implicitly*, through training against pool opponents. The 719-team pool may MEASURE structure but **never ships as a prior**.

🚨 **A forme SHARES its base species' `num`**, and the obs species channel and every `table[species.num]` buffer are num-keyed — num-indexed consumers MUST iterate `gen3_data.species.base_form_ids()`.

Per-file schemas: `src/agents/gen3_data/CLAUDE.md`. Acquisition: `tools/CLAUDE.md`.

---

## Event-Sourced Battle Layer (`src/agents/battle/`)

poke-env is a **state tracker**; RL/reward/replay need *what happened, in order*. `Gen3Battle` + the `BattleEvent` log give that. **Non-`battle/` code reads battle state ONLY through the read-models** — `LiveView` (current board), `TurnView` (history fold), `LegalActions` (server-authoritative legality) — via `battle.strict_view()`, enforced by `src/agents/strict_api_lock_test.py`. The per-decision `TurnDelta` folds entirely from the event log. Detail: `src/agents/battle/CLAUDE.md`.
