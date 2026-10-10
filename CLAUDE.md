# CLAUDE.md — Gen3AI Project Guide

> **This file is a CONSTITUTION, a COMMAND CARD and a MAP — nothing else.** It is loaded into every
> session, including every subagent's, so a line earns its place only if an agent that has NOT read
> it would do the work **wrong**. Detail lives in the leaf `CLAUDE.md` files and the `designs/` docs
> named throughout; **follow the pointer when you touch the subject.** (The previous, longer root is
> frozen at `designs/research_state/claude_md_archive/root_CLAUDE_2026-10-09.md`.)

## Where the detail is

| I am about to… | Read first |
|---|---|
| run / resume / fork a training run, or run the smoke | `designs/ops/training_runbook.md`, then `src/main/launcher/CLAUDE.md` |
| operate a live run (watch, kill, read) | `designs/ops/TRAINING_RUN_SOP.md` |
| dispatch, wait on or resume agents | `designs/ops/ORCHESTRATOR_SOP.md` §2, §7 |
| write or tier a test, run a benchmark, check a static gate | `designs/ops/testing.md`; the gate table is `docs/DEVELOPING.md` |
| reason about the model | `designs/ARCHITECTURE.md` **first**, then `src/agents/model/CLAUDE.md` → `designs/model/` |
| reason about the research | `designs/research_state/UNDERSTANDING.md`; `ledger.md` wins any disagreement |
| touch a training flag, or quote an offline meter / ELO | `src/agents/training/CLAUDE.md` → `designs/training/<topic>.md`; the meter index + ELO rules: `designs/ops/training_runbook.md` |
| touch the rust port | `src/rust_sim/CLAUDE.md`, then its `designs/rust_sim/<topic>.md` |
| probe a run's traces / the prober | `src/main/prober/CLAUDE.md`, then its `designs/prober/<topic>.md` |
| play over a websocket, or read an external anchor | § Playing below, then `designs/ops/EXTERNAL_ANCHORS_SOP.md` |
| see where the system is heading | `designs/endstate/README.md` (index + reading order) |

---

## Development Stage

**Rapid iteration — checkpoint compatibility is not a concern.** Breaking changes to the observation space, network architecture or action space are fine; add no backwards-compatibility shims. Architecture constants live in `src/agents/model/arch_constants.py` and **nowhere else** (`features_extractor.py` re-exports them). **Before reasoning about the model, read [`designs/ARCHITECTURE.md`](designs/ARCHITECTURE.md)** — the only document that states it *as it is now*; [`designs/CHANGELOG.md`](designs/CHANGELOG.md) is the past.

---

## 🧱 Standing rules for EVERY session and agent — the ONE list

Briefs do not repeat these; an agent that breaks one has broken its brief. Detail lives where each points.

1. **Never stop, restart or signal a server or process you did not start.** Ports 8000 / 8001 are reserved and REFUSED in code; a server of your own binds a `9XXX` port and is stopped by its PID (§ Showdown Server).
2. **Kill only by explicit PID or process group** — never `pkill -f` / a blanket kill. **No unbounded foreground wait loop**: wrap it in `timeout N` (N below the tool timeout) or run it in the background. (A hook refuses both.)
3. **Never put the literal trainer script name in a backgrounded argv** — it trips the live-run watchers.
4. **Edits and commits only in a worktree; main is never dirty.** Land code only via `/gen3ai-ship` (delegated or typed): routine gate green, `python -m utils.push_guard`, never a soft reset onto a moved ref (§ Git Workflow). In a worktree, `export PYTHONPATH=$PWD/src` (§ Python Environment).
5. **The GPU is LEASED for an agent's lifetime, and nobody waits** (owner, 2026-10-03). The orchestrator grants ONE lease at a time and names the agent in its brief; that agent runs `scripts/ops/gpu_lease.sh acquire --owner <name>` (immediate typed refusal if held), puts the printed token in front of each GPU call, and `release`s at the end. Every GPU command goes through `scripts/ops/gpu_lock.sh`, which passes the owner's token through and REFUSES everyone else AT ONCE (exit 6 leased, 5 one-off holder) — `--wait` is for the orchestrator or a training launch only. **An agent that is not briefed with a lease never touches the GPU and never sets `GEN3AI_TEST_ALLOW_GPU`.** A heavy one-off job goes under `scripts/ops/mem_cap.sh` (§ Running Tests).
6. **`models/` is read-only** except your own new run dirs; **no `data/` change while a pinned run is live** (pins isolate code, not data).
7. **Report every hazard, skip, or thing you could not verify as an explicit FINDING** — a reported hazard is a finding, not a footnote.
8. **Checks pass or fail DETERMINISTICALLY**: exclude inputs within a rounding error of a decision boundary rather than tolerating them by chance (owner, 2026-10-01).
9. **Out of scope ⇒ STOP and report**; never widen a unit or a closed list on your own.

---

## Subagents, Workflows and waiting

Mechanics and evidence: [`designs/ops/ORCHESTRATOR_SOP.md`](designs/ops/ORCHESTRATOR_SOP.md) §2 (dispatch) and §7 (stalls, waiting).

- **Dispatch a named agent type** (`~/.claude/agents/`: `opus-*`, `sonnet-*`) — the cheapest tier that fits the risk, never above your own effort; never a fork or a generic type, whose effort you do not control.
- **Never an unbounded wait on a sub-agent** — its completion notification can fail to arrive. Right after dispatching, arm ONE background `timeout 3300 sleep 3300`; on a timer wake make ONE cheap status read per agent and re-arm silently; let it lapse when the last one reports. No wait of any kind runs past 55 min (the prompt cache lives 1 h).
- **A stalled `Agent` is RESUMED with `SendMessage` to its agentId, not redone**; after two resumes that change nothing on disk, do the work inline.
- **In a Workflow script, pass `stallMs: 900_000` on EVERY `agent()` call** — Workflow subagents carry a separate 180 s watchdog hardcoded in the binary (no env var, no settings key; the `Agent` tool rejects `stallMs`). Cap concurrency at 1–2, for token cost.
- **Never report agent ERRORS as "no findings"**: `parallel()` returns `null` for a failed agent — track failures and return a distinct status; diagnose from `journal.jsonl`'s `result` field.
- The 2026-08…10 stall epidemic was Claude Code's keep-alive latch, patched 2026-10-07 (131 → 0 watchdog aborts in an afternoon): a session runs patched when launched via `cc_keepalive_patch.py claude …`, and `cc_keepalive_patch.py self` checks the running one. Don't re-chase stalls by tuning timeouts.

---

## Documentation Maintenance

Keep docs in sync **automatically, as part of the same change** — no need to be asked:

- **Every `CLAUDE.md`** (root and every leaf) and **every `README.md`**: always current; fix a stale one in the same pass. Exception: `designs/ai_v3/README.md` is a frozen historical digraph.
- **`designs/ARCHITECTURE.md`**: always current, the **first** thing an architecture change updates. Only what is true now — no version numbers in prose, every measured figure with its provenance, every unverifiable claim marked `**UNVERIFIED:**`; state the new truth and delete the old.
- **`designs/CHANGELOG.md`**: append-only; add the new version entry in the same pass, never edit an old one.
- **Five `designs/` trees OWN detail lifted out of a leaf** and are updated with the code exactly like one: `designs/ops/`, `designs/training/`, `designs/model/`, `designs/rust_sim/`, `designs/prober/`. **Every doc in `designs/endstate/` is ALWAYS-CURRENT** (owner, 2026-09-27): a decision or build that differs updates it in the same commit, saying what changed and why. `designs/research_state/claude_md_archive/` is HISTORY — add, never update.
- **Other `designs/` docs** (`impl_step*.md`, `design_*.md`, `todo.md`) are explicit-only (directly, or via `/gen3ai-update-design-docs`); `CLAUDE.md` files inside `designs/` follow the always-current rule.
- A path or flag a `CLAUDE.md` names deliberately as HISTORY goes in `designs/deleted_flags.md` with its citation (the freshness gate reads it).

**Leaf `CLAUDE.md` map:**

| Directory | Leaf covers |
|---|---|
| `src/agents/model/` | Feature-extractor phase contract, dual-head policy, architecture-constant rules, model versioning |
| `src/agents/gen3_data/` | The data facade over `data/`, the acquisition-vs-access split, every per-file schema |
| `src/agents/observation/` | The observation LAYOUT (constants, `get_layout()`, vocabularies), the Rust-encoder benchmark mandate, the per-block meaning table |
| `src/agents/battle/` | The battle read-models (LiveView / LegalActions, built from the Rust core by `core_view`), core-obs frames, corpus replay |
| `src/agents/training/` | The training hub — each topic's summary, its detail in `designs/training/<topic>.md` |
| `src/rust_sim/` | The Rust Showdown port: module map, conventions, the differential-gate ladder, the A/B fuzzers, the search/replay drivers |
| `src/main/launcher/` | Launcher internals: restarts, crash reporting, exit codes, flags, which interpreter the child runs |
| `src/main/prober/` (+ `web/`) | The forensic-replay inspector's engine, `ProbeSession`, the JSON CLI, the browser front end and its hazards |
| `src/main/tui/` | The thin shared Textual base (the launcher's UI) |
| `tools/` | The acquisition layer — the only code that knows the three upstreams |
| `designs/` | The `designs/` folder map and the always-current docs in it |

**Non-`CLAUDE.md` docs under the same always-current obligation:** `designs/ARCHITECTURE.md` (the MODEL now) · `designs/research_state/UNDERSTANDING.md` (what we BELIEVE now; its append-only `ledger.md` wins any disagreement — fix the view, never the ledger) · `designs/ops/TRAINING_RUN_SOP.md` and `designs/ops/ORCHESTRATOR_SOP.md` (how runs and the orchestrator operate).

---

## Git Workflow

Personal project — no pull requests. Work lands directly on `main`, but **all edits and commits happen in a worktree or branch, never on the main checkout** (`/home/goodlad/dev/gen3ai`); main is never dirty. `/gen3ai-ship` (`.claude/commands/gen3ai-ship.md`) is the only mechanism that lands code: it rebases onto `origin/main`, regenerates the ledger index, runs `python -m utils.push_guard` (exit 1 = a stale file would revert others' work — do NOT push), then pushes `<branch>:main`.

🚨 **Who may commit and push (owner, 2026-09-29):** the ORCHESTRATOR session (named in `~/.claude/projects/-home-goodlad-dev-gen3ai/ORCHESTRATOR`) holds STANDING `/gen3ai-ship` permission and may DELEGATE it per logical unit in a brief; everyone else commits or pushes only when the user's current message contains `/gen3ai-ship`. Finishing a task is NOT by itself a reason to commit — ship a complete unit with green gates. **No AI attribution lines** (`Co-Authored-By`, `Claude-Session`) in commits; the skill overrides the harness reminder.

**Rebases:** the ledger and `CHANGELOG.md` merge by union and `ledger_index.md` keeps our side (`.gitattributes`) — never hand-merge the index, re-run `python -m main.ledger_index --write` after a rebase, and never edit the ledger. When parallel agents edited the same doc, keep both sides' facts.

---

## Python Environment

The env is **`gen3ai_torch28`** (torch 2.8, `environment_torch28.yml`) for every run, gate and tool; new code targets torch 2.8 only. A bare `python3` outside an activated env is BASE miniconda, and `deps/venv` is stale — ignore both.

```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 <script>     # elsewhere: conda activate gen3ai_torch28
```

- **`./scripts/bootstrap.sh`** does all setup idempotently (env, submodule, Showdown build, worktree symlinks, optional cargo build) and verifies with the static gates and a smoke; `--dry-run` prints the plan. 🚨 **The conda env is SHARED:** a WORKTREE bootstrap whose env file differs prints the diff and REFUSES (exit 3) — `--update-shared-env` opts in, `--skip-env` finishes without touching it. Human version: `docs/DEVELOPING.md`.
- 🚨 **In a worktree, a direct `python <script>` / `python -m …` imports MAIN's code unless you export `PYTHONPATH`** (the editable install names main's `src/`); `pytest` puts this checkout's `src/` first by itself (root `conftest.py`). Detail: `designs/ops/testing.md` "Worktree imports". `PYTHONPATH` outranks the editable install, which is how the launcher pins a resumed run to its commit — so install editable from the **main checkout only**.
- **`gen3ai_stable` (torch 2.5.1, `environment.yml`) is LEGACY and FROZEN** — never edit `environment.yml` or update that env. The launcher selects it only to resume a pre-era run on the torch it recorded (`--allow-torch-switch` consents to switching); HEAD runs torch ≥ 2.8 only (`src/utils/torch_floor.py`). Detail: `src/main/launcher/CLAUDE.md` "Which interpreter the child runs".
- 🚨 **poke-env is RETIRED — one Rust stack** (T27, done 2026-10-08): no file of ours imports it (`src/poke_env_import_gate_test.py`, a shrink-only allowlist, now EMPTY) and no package exists (`src/poke_env_absent_gate_test.py`). Fix an import with `agents.gen3_data` (data tables), `agents.enums` / `utils.showdown_id` / `utils.team_packing` (owned seams) or the Rust path. The one process still on poke-env is the Metamon peer script (`src/main/anchors/peer_scripts/metamon_side.py`), upstream poke-env in Metamon's own interpreter.
- Every process the project spawns runs under `sys.executable`; the launcher takes **`$GEN3AI_PYTHON`** to pin its child's interpreter.

---

## Git Worktree Setup

**`./scripts/bootstrap.sh` does this.** The manual recipe, from a FRESH worktree only:

```bash
git submodule update --init
for n in dist node_modules; do
  [ -e "deps/pokemon-showdown/$n" ] || \
    ln -s "/home/goodlad/dev/gen3ai/deps/pokemon-showdown/$n" "deps/pokemon-showdown/$n"
done
```

> 🚨 **Keep the `[ -e ]` guard, and never run this from the MAIN checkout**: there `dist` is a real directory, `ln -s` puts a `dist/dist` loop inside it, and every Node path then dies with `ELOOP` (2026-07-23, unnoticed for four weeks). Fix: `rm deps/pokemon-showdown/dist/dist`, then `node build`. Never symlink the whole `deps/pokemon-showdown` (git then sees a symlink and `git status` breaks).

Without step 2 the root `conftest.py` REFUSES the test session with one message naming the fix (`GEN3AI_SKIP_DEPS_GUARD=1` opts a pure-unit CI out), and every Node-backed path (the team validator, `--server node`, the Node differential gates) fails with `Cannot find module '…/dist/sim/…'`.

---

## Running Tests

**Full chapter — tiers, markers, contention, fuzz, benchmarks: [`designs/ops/testing.md`](designs/ops/testing.md).** Prefix each command with `export PYTHONPATH=$PYTHONPATH:src PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3} &&`:

| When | Command |
|---|---|
| **inner loop** | `"$PY" -m pytest src/ -m "not slow and not e2e and not sim and not integration" -q -n 2` |
| **THE ROUTINE GATE** | `scripts/ops/gate_lock.sh "$PY" -m pytest src/ -m "not slow and not e2e" -q -n 6` — ONE gate at a time (`utils.gate_lock`), ~4.3 min quiet; `-n 4` beside a live training run |
| **before `/gen3ai-ship`** | the routine gate, or a TARGETED set you choose + the static gates, scope and reason named in the commit body (owner, 2026-10-05). Shared infrastructure or any doubt ⇒ the routine gate. Never an automated test-selection framework |
| everything / the slow tier | `"$PY" -m pytest src/ -q` (~47 min serial) / `"$PY" -m pytest src/ -m slow -q -n 2` |
| anything on the GPU | only with a LEASE (standing rule 5): `scripts/ops/gpu_lock.sh <cmd>` with `GEN3AI_GPU_LEASE_TOKEN` set (Python: `utils.gpu_lock.gpu_lock()`) — **never a bare `flock …/gpu.lock`** |
| a one-off HEAVY job | `scripts/ops/mem_cap.sh <GB> <cmd>` (Python: `utils.mem_cap`) — its own capped scope, so an overrun kills only that job, not the session |

The rules an agent would otherwise break (each detailed in `designs/ops/testing.md`):

- 🚨 **Cut on `slow`, never the old `-m "not integration and not e2e"`** — `integration` spans a ~100x cost range; that cut let the obs golden ride main RED three times. A tier is DECLARED by marker (what a test NEEDS: unmarked · `integration` · `sim` · `browser` · `e2e`; what it COSTS: `slow`), never inferred from a filename.
- **Seventeen static gates** run in every tier (the `static` marker: mypy, ruff, file size, `CLAUDE.md` freshness, stub vacuity, slow-tier status, the doc gates, the ledger index, the poke-env gates, …); a missing tool FAILS. The table with every opt-out is `docs/DEVELOPING.md`; `python -m utils.static_gates` lists them. **Their allowlists are EMPTY — decompose a file over 2,000 lines, fix a vacuous stub at the source; never add an entry.**
- 🚨 **A deselected `slow` test cannot fail the routine gate — so the slow tier records its verdicts** in `designs/ops/slow_tier_status.json` (committed) and the routine gate fails on a recorded FAIL. A test that broke since the last slow run still reads green.
- 🚨 **A decomposition moves symbols out from under the stubs that name them**: a patch on `mod.func` reaches the code only if `mod` still reads `func` at call time. Spell module and attribute out (the stub gate is blind to loop variables).
- 🚨 **A test that leaks process-global torch state FAILS** (threads, matmul precision, dtype, dynamo/inductor config) — restore with `try/finally`, `torch_globals(...)` or the `restore_torch_globals` fixture.
- 🚨 **A TIMEOUT IS NEVER A SEMANTIC OUTCOME.** Bounds scale by measured contention (`src/utils/contention.py`); >25 % timeouts ⇒ INCONCLUSIVE. Benchmarks are the opposite: warn, never stretch.
- 🚨 **A collected test wants the SAME battle every run** (the Rust core, seeded end to end: `utils.rust_env.fixture_battles.play_rows(...)`, fixed teams, `concurrency=1`) and **asserts its precondition** rather than branching on it; a fuzz SCRIPT wants a new battle every run.
- Rust tests and fuzz scripts run the emission self-check build; a fresh worktree pays one cargo build first. 🚨 **Any obs change: run `python -m agents.observation.rust_encoder_benchmark` before and after, and `python -m agents.training.golden_obs_core --check`.**

```bash
cargo build --profile selfcheck --features emission-selfcheck --bin sim_bridge --bin search_driver --bin core_events --manifest-path src/rust_sim/Cargo.toml
```

---

## Smoke Test

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 src/main/train_rl_agent.py --debug --steps 10000
```

~2 min on CPU, serverless, on the production critic and the Rust env core; exit 0 with `Training complete. Model saved to …`. Add `--debug-eval --eval-freq 4000` to exercise eval (`--debug-eval` alone fires no cycle); `--arch production --debug` works (it auto-shrinks the update and prints `🧪 [DEBUG SHAPE]`). For a throwaway smoke set `GEN3AI_MODELS_DIR` to a scratch dir — otherwise the run lands in main's `models/`. What to look for, and what each failure means: `designs/ops/training_runbook.md` "The smoke test". 🚨 **`--debug` exercises a STRICTLY SMALLER surface** (one env, CPU, eager inference, no `--compile-trainer`, no warm start): the first two minutes of a real launch are the only test of that layer.

---

## Training + the Launcher

**Full chapter: [`designs/ops/training_runbook.md`](designs/ops/training_runbook.md)**; per-flag semantics in `src/agents/training/CLAUDE.md` and `src/main/launcher/CLAUDE.md`. The launcher (restarts, crash auto-restart, worktree pinning, a TUI; headless when detached) is the way to run anything long.

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.checkargs models/<run>          # or --argv "…": does it launch, AND is it the architecture you meant?
python -m main.launcher --dry-run …            # resolve the ACTUAL launch without creating anything (safe on a restart)
python -m main.launcher --restart-interval-hours 6 --steps 15000000 \
  --device cuda --log-level periodic --arch production                     # fresh run
python -m main.launcher --restart-interval-hours 6 --model models/<run>/checkpoints/<ckpt>.zip \
  --steps 15000000 --device cuda                                           # resume / fork
```

The hazards that have actually cost runs (full text: the runbook's "Launch hazards in full"):

- 🚨 **"It launches" and "it is the experiment" are INDEPENDENT checks.** `--arch production` supplies the architecture AND the training recipe; `checkargs` refuses a fresh argv whose surface differs. Type a recipe knob only when it IS the arm's lever. A design-doc command block is not a launch command (2026-09-06: 24.4M steps on a near-bare network).
- 🚨 **Restart interval: 6 h** (owner, 2026-10-04) — the launcher default since 2026-10-10 (`run.DEFAULT_RESTART_INTERVAL_HOURS`; was 3), so a bare launch restarts every 6 h; the examples above still type it, and a typed value wins.
- 🚨 **AN ARGV IS NOT A CONFIG.** With `--model`, every flag you do not name is INHERITED from `model_config.json`; `--lr` / `--batch-size` / `--n-steps` are INERT on a resume — a fork pins its rate with `--fork-lr`, and the quantity to match is the DOSE (`python -m main.dose <run>`).
- 🚨 **A BARE RUN DIRECTORY MEANS THE RUN'S LAST SNAPSHOT** (`--stable-opponents`, `--exploiter`, …); name the `.zip` or `@step` to pin a file.
- 🚨 **A restart RESUMES; a FRESH argv never lands on a run** — a fresh launch into a dir holding a checkpoint is refused; pass `--model` or a new `--run-name`. A PINNED argv is judged by the PINNED commit's parser. Never "launch and kill" to validate — use `--dry-run`.
- 🚨 **A fork's self-play pool starts EMPTY**, and an empty pool silently falls back to the bot pool; the trainer auto-seeds the parent's pool and refuses (`FATAL_CONFIG` / `FATAL_SUPPLY`) otherwise.
- **Refusals you will meet:** a CUDA launch while the desktop holds the GPU (stop `gdm.service` first; `--allow-desktop-gpu`), and one the disk cannot fit (`--allow-low-disk`; in flight, `FATAL_DISK`) — `designs/ops/TRAINING_RUN_SOP.md` §1.
- **ERAS:** runs are named for Hoenn towns; the Rustboro era (`rb_`) is current (`utils/era.py`). Pre-era runs are history, never comparators.

**Constants, not flags:** the Rust env core and the in-process Rust bridge are the only env core and transport (no Showdown server); the win-prob critic, its win-indicator terminal and `gamma` 1.0 are constants — `--critic`, `--env-core`, `--gamma` and friends are DELETED and refused with the reason (`designs/deleted_flags.md`). `--compile-trainer` is on for cuda.

**Meters:** `python -m main.<meter>` — the index (what each reads, which four write under `models/<run>/`) and the rules for quoting an ELO (`ladder.json` not `eval/elo`, final only at run end, matched snapshot COUNT, a `recipe` block that matches the fitter; the transport and eval-regime boundaries) are the runbook's last sections. `main.ops.*` and `scripts/ops/` are the LIVE-run tier (`scripts/ops/README.md`).

---

## Playing / the LADDER

🚨 **NO LADDER CAMPAIGN, and an agent NEVER plays, challenges or chats with a human** (owner, 2026-10-07: "be respectful that they are real people"). We play the agent against itself and against bots. The only public-server use is a low-volume CUSTOM gen3ou series between OUR OWN accounts, through the owner's SOCKS5 proxy, with the orchestrator's explicit go (`play.py --server official --public-acceptance --proxy …`, both accounts in `$PS_OWN_ACCOUNTS`). Decision record: `designs/endstate/design_ladder_campaign.md`.

🚨 **`--mode ladder` (owner, 2026-10-09): allowed for USERS at concurrency 1 (fixed); NEVER by an agent without the owner's explicit approval token.** In an agent session (Claude Code's environment markers, own or any ancestor's) it REFUSES unless the OWNER has written `~/.local/state/gen3ai/ladder_owner_approval.json` (a purpose and an expiry ≤ 24 h) — **an agent never creates, edits or requests it be created for itself.** The other guards (`src/main/ladder_guard.py`): no T28 halt marker, and a green `ladder_drift_scan` from the last 2 days (the scan records it). Never queue a ladder game from an agent on your own initiative.

`src/main/play.py` is a websocket CLIENT on the Rust stack (`main.live`: the live reader reads the server's stream through the same chain training's rows come from — `designs/rust_sim/live_reader.md`). It dials a server YOU started on a 9XXX port:

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m utils.bridge.ws_frontend --port 9017 &       # our Rust websocket front end (no Node); stop it by its PID
python3 src/main/play.py --mode selfplay --port 9017 --model models/<run>/final_model.zip --n-battles 4
python3 src/main/ladder_drift_scan.py --n 200          # 🚨 the DRIFT GATE — before any public-server session
python -m main.anchors --model models/<run> --opponent metamon:SmallRL \
  --regime greedy --teamset away --games 100 --out <dir>   # an EXTERNAL-ANCHOR read
```

- 🚨 **A LIVE PARSE PANIC HALTS ALL PLAY (T28).** An unparseable / unclassified line, an encoder raise or an unsendable choice exits `FATAL_LIVE_PARSE` (7) and writes a durable HALT marker; `main.play` and `main.anchors` refuse to start while it exists (`python -m main.live.halt status`). Root-cause it with a regression test built from the captured lines that fails on revert, land it, then `python -m main.live.halt clear --fixed-by <commit>`, and PUSH the owner. Never skip the line.
- 🚨 **Run the drift gate first**: the public server runs Showdown master while `deps/pokemon-showdown` is pinned, and the reader refuses an unknown keyword BY DESIGN — on a live game that is a T28 halt. `--mode ladder` enforces it (a green record ≤ 2 days old, written by a FULL scan). Audit: `designs/research_state/ladder_readiness.md`.
- 🚨 **An external-anchor read is `python -m main.anchors`, procedure [`designs/ops/EXTERNAL_ANCHORS_SOP.md`](designs/ops/EXTERNAL_ANCHORS_SOP.md)** — greedy-vs-greedy, and a number never leaves it without its regime. It starts and stops its OWN server (the Rust front end by default, 9500–9599).
- **A websocket game ends where a training episode ends**: `--forfeit-turn-limit` defaults to the trainer's stall threshold (`MAX_TURNS`); lower it if you must, never raise it.

---

## Prober (forensic-replay inspector)

Reads a run's `eval_traces` + a checkpoint; no server. Two surfaces over one engine — the browser app and the JSON CLI (`python -m main.prober.query summary|scan|turns|analyze|…`).

```bash
python3 -m main.prober models/<run>          # browser app
python3 -m main.prober.web models/           # :6008 run picker; public at prober.g5d.io (model routes behind the shared password)
```

⚠️ Model-loading views work only at the CURRENT architecture (`ArchDriftError` elsewhere); model-free commands work on every run. 🚨 **The trace quota PREFERS LOSSES** — read each cycle's `eval_manifest.json`; none recorded = SELECTION UNKNOWN. Detail: `src/main/prober/CLAUDE.md`.

---

## Showdown Server

**No standing Showdown server exists on this box, and none is needed**: training and eval run in-process on the Rust core. Ports **8000 / 8001 are reserved and refused in code** (`play.py`, `main.anchors`, `utils.bridge.ws_frontend`). Need a server? Bind a **9XXX** port, prefer our Rust front end (`python -m utils.bridge.ws_frontend --port 9XXX`), use Node (`npm run showdown -- 9XXX`) only when the test is about Node, and **stop only what you started, by its PID** — never a bare `npm run stop` (it `kill -9`s whatever holds :8000), never a blanket node kill.

---

## Repository Structure

```
src/
  agents/       model/ (extractor, heads, versioning) · observation/ (the obs layout) · gen3_data/ (the data facade)
                battle/ (read-models) · action/ (the 11-dim action space) · inference/ (the inference service, belief decode)
                training/ (callbacks, eval, meters) · enums.py
  main/         train_rl_agent.py + train/ (its phases) · launcher/ · live/ (websocket play, the T28 halt) · anchors/
                h2h/ · prober/ (+ web/) · ops/ (live-run instruments) · tui/ · the offline meters (*.py and packages)
  rust_sim/     the Rust Showdown port          rust_env/   the Rust ENV core (the only env core)
  utils/        paths.py, git.py, era.py, bridge/ (ws_frontend, the sim bridge), rust_env/ (its Python side), …
designs/        ARCHITECTURE.md, CHANGELOG.md, production_config.json, baselines.json, endstate/, ops/,
                training/, model/, rust_sim/, prober/, research_state/, ai_vN/ (history)
docs/           RUNNING.md, DEVELOPING.md (the human references)
data/           the source of truth (derived by tools/, read via agents.gen3_data)
tools/          the acquisition layer          scripts/   bootstrap.sh, land.sh, ops/
deps/           the pokemon-showdown submodule   models/   saved runs (NOT committed; MAIN checkout only)
```

### Path discovery — `src/utils/paths.py`

🚨 **Never hand-write `Path(__file__).resolve().parents[N]`.** Use `repo_root()` / `repo_path(...)` (this checkout), `src_root()` / `src_path(...)`, `main_models_dir()` (the run archive to READ — `Path` **or `None`**, which every caller turns into a skip) and `run_archive_dir()` / `checked_run_dir(path)` (where a run is CREATED — a typed `RunArchiveError` when there is none); `utils.git` for git's opinion. 🚨 **`models/` exists only in the MAIN checkout**, and a run ALWAYS lands there, even from a worktree (`$GEN3AI_MODELS_DIR` overrides, authoritatively). A test that creates a run dir takes the `run_archive` fixture. `src/utils/paths_test.py` fails any `/home/…` literal used as a value in `src/agents`, `src/main`, `src/utils`.

---

## The Model — obs, extractor, versioning, baselines

- The observation is a flat **2845-dim float32 vector** + an 11-dim `action_mask` (a Dict obs). 🚨 **NEVER hardcode an index** — read `Gen3ObservationEncoder.get_layout()`; offsets live in `agents/observation/constants.py`.
- `Gen3FeaturesExtractor` returns a **`(pi_features, vf_features)` tuple** and MUST be paired with `Gen3DualHeadMaskablePolicy`; the action head is the **pointer head** (no flat `action_net`).
- `ARCH_SIGNATURE` / `MODEL_CONFIG_VERSION` live in `src/agents/model/model_version/`. **Read live values from the code — a version quoted in prose is stale the moment the next lands.** Experimental arms (e.g. the static-token encoder, `--arch static_recovery`) are built and OFF; `designs/ARCHITECTURE.md` and `designs/endstate/design_static_tokens.md` say which. Every quantity the model reads that our code derives by hand is ledgered in `designs/endstate/design_hand_computed_features.md`.
- **Every save writes `model_config.json`** (the arch record; a mismatch is a hard `[ModelVersion] FATAL`) **and `metadata.json`**, whose `original_command`, `lineage` and `pin_history` blocks are IMMUTABLE — read them through `main.lineage` / `main.sidecar_audit`, never by re-deriving.
- 🚨 **BASELINES are NAMED and read by name** (`designs/baselines.json` + `agents.training.baselines`); never copy a path. `python -m main.baselines set … --reason` is the only way to change one.
- 🚨 **`models/` retention is policy:** the pre-Rustboro SKELETON was applied 2026-10-09 — every pre-`rb_` run keeps only its records, TensorBoard, jsonl, final model and registry-named files (intermediate checkpoints, eval traces and their sidecar JSONs are GONE). What survives where: `designs/research_state/models_retention_policy.md` §0.

---

## Data Dependencies

**`data/` is the single source of truth**: `tools/` (the only layer that knows the three upstreams) derives each file into `data/pokemon/`; the runtime reads it only through the **`agents.gen3_data` facade**.

- 🚨 **ALL priors must be Smogon-derived; only the MODEL gets bias against the pool** (owner, 2026-08-15). Anything the network READS traces to Smogon, ground-truth labels or ladder replays; the team pool may MEASURE structure but **never ships as a prior**.
- 🚨 **A forme SHARES its base species' `num`**: num-indexed consumers MUST iterate `gen3_data.species.base_form_ids()`.

Per-file schemas: `src/agents/gen3_data/CLAUDE.md`. Acquisition: `tools/CLAUDE.md`.
