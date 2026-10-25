# CLAUDE.md — Gen3AI Project Guide

Loaded into every session and subagent: the standing rules, the commands, and a map. A line stays only if an agent
without it would do the work wrong ([`designs/ops/instruction_docs_principles.md`](designs/ops/instruction_docs_principles.md)).
Follow the pointer when you touch the subject.

## Where the detail is

| I am about to… | Read first |
|---|---|
| run / resume / fork a training run, or run the smoke | `designs/ops/training_runbook.md`, then `src/main/launcher/CLAUDE.md` |
| operate a live run (watch, kill, read) | `designs/ops/TRAINING_RUN_SOP.md` |
| dispatch, wait on or resume agents | `designs/ops/ORCHESTRATOR_SOP.md` §2, §7 |
| write or tier a test, run a benchmark, check a static gate | `designs/ops/testing.md`; the gate table is `docs/DEVELOPING.md` |
| reason about the model | `designs/ARCHITECTURE.md` **first**, then `src/agents/model/CLAUDE.md` → `designs/model/` |
| touch the observation | `src/agents/observation/CLAUDE.md` |
| touch `data/` or the data facade | `src/agents/gen3_data/CLAUDE.md`, `tools/CLAUDE.md` |
| reason about the research | `designs/research_state/UNDERSTANDING.md`; `ledger.md` wins any disagreement |
| reason about search | `designs/endstate/search_ledger.md` (always-current; update it with any search result) |
| give training, search or the model's inputs EXTRA information, or HUMAN replays | `designs/endstate/design_information_levers.md` (`agents.info_levers`); human data: `designs/endstate/human_data_architecture.md` |
| touch a training flag, or quote an offline meter / ELO | `src/agents/training/CLAUDE.md` → `designs/training/<topic>.md`; meters + ELO rules: the runbook |
| touch the rust port | `src/rust_sim/CLAUDE.md` → `designs/rust_sim/<topic>.md` |
| probe a run's traces / the prober | `src/main/prober/CLAUDE.md` → `designs/prober/<topic>.md` |
| play over a websocket, or read an external anchor | § Playing below, then `designs/ops/EXTERNAL_ANCHORS_SOP.md` |
| edit anything under `designs/` | `designs/CLAUDE.md` |
| see where the system is heading | `designs/endstate/README.md` |

**Rapid iteration:** breaking changes to the observation, network or action space are fine; add no compatibility
shims. Architecture constants live only in `src/agents/model/arch_constants.py`.

---

## Standing rules for EVERY session and agent — the ONE list

Briefs do not repeat these; an agent that breaks one has broken its brief.

1. **Never stop, restart or signal a server or process you did not start.** Ports 8000 / 8001 are reserved and refused
   in code. Training and eval need no server; if you need one, bind a `9XXX` port (prefer
   `python -m utils.bridge.ws_frontend --port 9XXX`; Node only when the test is about Node) and stop it by its PID.
2. **Kill only by explicit PID or process group**, never `pkill -f` or a blanket kill. **No unbounded foreground wait
   loop**: wrap it in `timeout N` or run it in the background. (A hook refuses both.)
3. **Never put the literal trainer script name in a backgrounded argv**: it trips the live-run watchers.
4. **Edits and commits only in a worktree; main is never dirty. Never `git stash`**: the stash is shared by every
   worktree. Land code only via `/gen3ai-ship` (§ Git Workflow). In a worktree, `export PYTHONPATH=$PWD/src`.
5. **The GPU is LEASED for an agent's lifetime, and nobody waits.** The orchestrator grants ONE lease and names the
   agent in its brief; that agent runs `scripts/ops/gpu_lease.sh acquire --owner <name>`, puts the token in front of
   each GPU call, and `release`s at the end. Every GPU command goes through `scripts/ops/gpu_lock.sh`, which refuses
   everyone else at once (`--wait` is for the orchestrator or a training launch). **An agent without a lease never
   touches the GPU, never sets `GEN3AI_TEST_ALLOW_GPU`, and runs with `CUDA_VISIBLE_DEVICES=""`** (an import can
   open a CUDA context).
5b. **While a training run is live, agent CPU work runs at low priority on the agent cores**: `nice -n 19` and
   `taskset -c 5-7,13-15` (3 whole physical cores; CPU n and n+8 are SMT siblings), because nice alone did not protect
   a live run. A heavy job goes under `scripts/ops/mem_cap.sh`, whose 2 heavy slots hold the box to ≤ 2 heavy jobs
   (memory cannot be shared by priority); **a job of ≤ 10 min declares `--max-s <S>`** (short lane: jumps queued
   chunks, killed past S + 30 s with exit 88). A unit the orchestrator marks BLOCKING the GPU's next work sets
   `GEN3AI_BLOCKING=<why>` (it skips the slots) and may run at nice 0 on the agent cores — launched that way, since
   renicing to a lower nice is refused.
6. **`models/` is read-only** except your own new run dirs; **no `data/` change while a run pinned BEFORE
   `gen3_pinned_data_root_v1` is live or queued** (it reads main's `data/teams` CWD-relative; a pin carrying
   `src/main/launcher/pinned_data_test.py` reads its own `data/` — `designs/launcher/pinning_and_worktrees.md`).
7. **Report every hazard, skip, or thing you could not verify as an explicit FINDING.**
8. **Checks pass or fail DETERMINISTICALLY**: exclude inputs within a rounding error of a decision boundary rather
   than tolerating them by chance.
9. **Out of scope ⇒ STOP and report**; never widen a unit or a closed list on your own.

## Subagents and waiting

Mechanics: `designs/ops/ORCHESTRATOR_SOP.md` §2 (dispatch), §7 (stalls, waiting).

- Dispatch a named type (`opus-*`, `sonnet-*`), the cheapest that fits the risk and never above your own effort;
  never a fork or a generic type, whose effort you do not control.
- Never an unbounded wait on a sub-agent (its notification can fail to arrive). A main session is covered by the
  idle-keepalive plugin (it wakes the session after 55 min idle); a sub-agent arms ONE bounded watcher, preferably on
  the job's PID, and ends a turn only with a result, a blocker or a timer wake (each turn end notifies its caller).
  No wait runs past 55 min (the prompt cache lives 1 h).
- Resume a stalled `Agent` with `SendMessage` to its agentId; after two resumes that change nothing on disk, do the
  work inline.
- In a Workflow script pass `stallMs: 900_000` on every `agent()` call (a separate 180 s watchdog), and track
  `parallel()`'s `null`s as failures, never as "no findings".

## Documentation Maintenance

Keep docs current **in the same change**, unasked. Apply the deletion test and replace, never append
(`designs/ops/instruction_docs_principles.md`; the freshness gate enforces each file's byte cap).

- **Always current:** every `CLAUDE.md` and `README.md` (except the frozen `designs/ai_v3/README.md`);
  `designs/ARCHITECTURE.md` (the model as it is now, the first thing an architecture change updates; no version
  numbers in prose, every figure with its provenance, `**UNVERIFIED:**` where it cannot be checked); every doc in
  `designs/endstate/`; `designs/research_state/UNDERSTANDING.md`; the SOPs; and the eight trees that own a leaf's
  detail: `designs/ops/`, `designs/training/`, `designs/model/`, `designs/rust_sim/`, `designs/prober/`,
  `designs/launcher/`, `designs/observation/`, `designs/tools/`.
- **Append-only:** `designs/CHANGELOG.md` (a new entry per version), the research ledger (never edit it; fix the
  view), `designs/research_state/claude_md_archive/`.
- **Explicit-only:** other `designs/` docs (`impl_step*.md`, `design_*.md`, `todo.md`).
- A flag or path a `CLAUDE.md` names deliberately as history goes in `designs/deleted_flags.md` with its citation.

## Git Workflow

No pull requests: work lands on `main` only through `/gen3ai-ship` (`.claude/commands/gen3ai-ship.md`: rebase,
ledger index, `python -m utils.push_guard`, push). **Who may ship:** the orchestrator session (named in
`~/.claude/projects/-home-goodlad-dev-gen3ai/ORCHESTRATOR`) holds standing permission and may delegate it per unit in
a brief; anyone else only when the user's current message contains `/gen3ai-ship`. Ship a complete unit with green
gates. **No AI attribution lines** in commits (the skill overrides the harness reminder). After a rebase, re-run
`python -m main.ledger_index --write`; never hand-merge the index.

## Python Environment

Every run, gate and tool uses **`gen3ai_torch28`** (torch 2.8); a bare `python3` is base miniconda and `deps/venv` is
stale.

```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 <script>
```

- **`./scripts/bootstrap.sh`** sets up a checkout or worktree idempotently (env, submodule, Showdown symlinks, cargo);
  `--dry-run` prints the plan. The conda env is SHARED: a worktree whose env file differs is refused; never
  `conda env update` it to get past that (`docs/DEVELOPING.md`). The cu128 twin and the frozen `gen3ai_stable` are
  `designs/launcher/interpreter.md`'s.
- **In a worktree, a direct `python` imports MAIN's code unless you export `PYTHONPATH`** (the editable install names
  main's `src/`; `pytest` fixes this itself). Install editable from the main checkout only.
- The launcher takes **`$GEN3AI_PYTHON`** to pin its child's interpreter.
- Node's only runtime job is validating teams with Showdown's own validator, so `deps/pokemon-showdown` stays a launch
  requirement; everything else Node does is a dev / test oracle, and search, replay and the prober are Rust only.

## Running Tests

Full chapter: [`designs/ops/testing.md`](designs/ops/testing.md). Prefix each command with
`export PYTHONPATH=$PYTHONPATH:src PY=${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3} &&`:

| When | Command |
|---|---|
| **inner loop** | `"$PY" -m pytest src/ -m "not slow and not e2e and not sim and not integration" -q -n 2` |
| **the static tier** | `"$PY" -m pytest $("$PY" -m utils.static_gates --paths) -q -n 2` (no gate lock) |
| **THE ROUTINE GATE, before every `/gen3ai-ship`** | `scripts/ops/ship_gate.sh run`: `-m "not slow and not e2e"` under `gate_lock.sh` (one gate at a time; `--status`) and `mem_cap.sh`, in the shape rule 5b allows; plus the `slow` tests of code you touched. A targeted set only with high confidence, its scope and reason in the commit body |
| everything / the slow tier | `"$PY" -m pytest src/ -q` / `"$PY" -m pytest src/ -m slow -q -n 2` |
| anything on the GPU | only with a lease (rule 5): `scripts/ops/gpu_lock.sh <cmd>` with `GEN3AI_GPU_LEASE_TOKEN` set |
| a one-off HEAVY job | `scripts/ops/mem_cap.sh <GB> <cmd>`: its own capped scope, so an overrun kills only that job |

- **Cut tiers on `slow`, never `-m "not integration and not e2e"`.** A tier is declared by marker, never inferred
  from a filename.
- **Seventeen static gates** run in every tier (`python -m utils.static_gates` lists them; the table with every
  opt-out is `docs/DEVELOPING.md`); a missing tool FAILS. Their allowlists are empty: fix the code (decompose a file
  over 2,000 lines, fix a vacuous stub at the source), never add an entry.
- A deselected `slow` test cannot fail the routine gate, so the slow tier records its verdicts in
  `designs/ops/slow_tier_status.json` and the routine gate fails on a recorded FAIL.
- **A timeout is never a semantic outcome**: bounds scale by measured contention; > 25 % timeouts ⇒ INCONCLUSIVE.
- A collected test plays the SAME battle every run (`utils.rust_env.fixture_battles.play_rows(...)`) and asserts its
  precondition; a test restores any process-global torch state it changes.
- **Any obs change:** run `python -m agents.observation.rust_encoder_benchmark` before and after, and
  `python -m agents.training.golden_obs_core --check`.

## Training and the Launcher

Full chapter: [`designs/ops/training_runbook.md`](designs/ops/training_runbook.md) (its "Launch hazards in full").
Run these with the env's interpreter and `PYTHONPATH` (§ Python Environment).

```bash
python3 src/main/train_rl_agent.py --debug --steps 10000   # the SMOKE: ~2 min, CPU; set GEN3AI_MODELS_DIR to a scratch dir
python -m main.checkargs models/<run>                     # or --argv "…": does it launch, AND is it the architecture you meant?
python -m main.launcher --dry-run …                       # the ACTUAL launch, creating nothing
python -m main.launcher --steps 15000000 --device cuda --log-level periodic --arch production        # fresh
python -m main.launcher --model models/<run>/checkpoints/<ckpt>.zip --steps 15000000 --device cuda   # resume / fork
```

- **"It launches" and "it is the experiment" are independent checks**: `--arch production` supplies the architecture
  AND the recipe; type a recipe knob only when it is the arm's lever. A design-doc command block is not a launch
  command.
- **An argv is not a config:** with `--model`, every flag you do not type is inherited from the run (a fork: its
  parent's recipe); `--lr` / `--batch-size` / `--n-steps` are inert on a resume. Match forks by DOSE
  (`python -m main.dose <run>`).
- Never "launch and kill" to validate; use `--dry-run`. `--debug` tests a strictly smaller surface than a real launch.
- Runs are named by era (`utils/era.py`); pre-era runs are history, never comparators.

## Playing

**No ladder campaign, and an agent never plays, challenges or chats with a human.** We play against ourselves and
bots. `--mode ladder` refuses in an agent session unless the OWNER wrote the approval file; an agent never creates it
or asks for it (`src/main/ladder_guard.py`, `designs/endstate/design_ladder_campaign.md`). Any public-server session
needs the orchestrator's go and a green `src/main/ladder_drift_scan.py` first.

- `src/main/play.py` dials a server YOU started on a 9XXX port; an external-anchor read is `python -m main.anchors`
  (`designs/ops/EXTERNAL_ANCHORS_SOP.md`; a number never leaves it without its regime).
- **A live parse panic halts all play** (`FATAL_LIVE_PARSE`, a durable halt marker that `main.play` and
  `main.anchors` refuse on): root-cause it with a regression test from the captured lines, land it, then
  `python -m main.live.halt clear --fixed-by <commit>`, and push the owner. Never skip the line.

## Paths, the model, data

- **Paths:** never hand-write `Path(__file__).parents[N]`; use `src/utils/paths.py` (`repo_path`, `src_path`,
  `main_models_dir`, `run_archive_dir`). `models/` exists only in the MAIN checkout and a run always lands there
  (`$GEN3AI_MODELS_DIR` overrides); a test that creates a run dir takes the `run_archive` fixture.
- **The observation:** never hardcode an index; read `Gen3ObservationEncoder.get_layout()`.
- **Read versions from the code** (`src/agents/model/model_version/`), never from prose. `model_config.json` and
  `metadata.json` are written at every save; their `original_command`, `lineage` and `pin_history` blocks are
  immutable: read them through `main.lineage` / `main.sidecar_audit`.
- **Baselines are named** (`designs/baselines.json`, `agents.training.baselines`); never copy a path; change one only
  with `python -m main.baselines set … --reason`.
- **`data/` is the source of truth**, derived by `tools/` and read only through the `agents.gen3_data` facade.
  **Everything the network reads traces to Smogon, ground-truth labels or ladder replays**; the team pool may measure
  structure but never ships as a prior (only the model gets bias against the pool).
