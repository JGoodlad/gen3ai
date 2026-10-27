# designs/ — the folder map

A map, not a reference. What the model IS: [`ARCHITECTURE.md`](ARCHITECTURE.md). How it got there:
[`CHANGELOG.md`](CHANGELOG.md) (history; never quote it as current). What we believe about the research:
[`research_state/UNDERSTANDING.md`](research_state/UNDERSTANDING.md) (it owns the claim-tag vocabulary). Where we stand right now, on one page: [`STATUS.md`](STATUS.md) (`src/status_doc_gate_test.py`). Where the system is heading, and all
current design work: [`endstate/README.md`](endstate/README.md) (the index and reading order).

## Always-current docs: the rules a reader would miss

Which docs are always-current is root § Documentation Maintenance. Here, the ones with a mechanism:

- **`ARCHITECTURE.md` §6's tables are GENERATED** (`python -m agents.model.arch_tables`, pinned by
  `arch_tables_test.py`); the prose around them is pinned against `production_config.json` by
  `src/mode_flag_doc_gate_test.py`.
- **Find a ledger entry through the GENERATED `research_state/ledger_index.md`** (`python -m
  main.ledger_index --write`, gated by `src/ledger_index_gate_test.py`); never edit it or the ledger by
  hand. `research_state/README.md` holds the heading convention and the rebase rule. When
  `UNDERSTANDING.md` and the ledger disagree, the later ledger entry wins: fix the view.
- **Every `endstate/` doc ends with a Decision record**: a decision or build that differs updates the doc
  in the same commit, saying what changed and why.

## A run is not the code

Read a run from the RUN — its `model_config.json` + `metadata.json` (`python -m main.ops.run_ref`,
`main.lineage`) — and the code's version from `src/agents/model/model_version/constants.py` +
`migrations.py`, never from prose; a pinned run trains on its pin's code. Production is what `--arch
production` resolves (`production_config.json`) plus the named baselines in `baselines.json`, never a run
named in prose. The ranked queues: `research_state/EXPERIMENT_BACKLOG.md` (experiments),
`ops/TASK_BACKLOG.md` (builds), `ops/TECH_DEBT_BACKLOG.md` (debt; nothing on it is dispatched without the
owner's word).

## The leaf-detail trees (always-current; each owns its topic)

| tree | lifted out of | holds |
|---|---|---|
| [`model/`](model/) | `src/agents/model/CLAUDE.md` | the phase pipeline, op contracts, flag-registry rules, versioning, opponent intent; it loses any disagreement with `ARCHITECTURE.md` |
| [`training/`](training/) | `src/agents/training/CLAUDE.md` | one doc per training topic |
| [`rust_sim/`](rust_sim/) | `src/rust_sim/CLAUDE.md` | the module map, census, gate ladder, fuzzers, emission, the websocket front end, the live reader |
| [`prober/`](prober/) | `src/main/prober/CLAUDE.md` (+ `web/`) | the engine, panels, session reference, counterfactuals, arch drift, the web service and pages |
| [`launcher/`](launcher/) | `src/main/launcher/CLAUDE.md` | index: [`launcher/README.md`](launcher/README.md) |
| [`observation/`](observation/) | `src/agents/observation/CLAUDE.md` | index: [`observation/README.md`](observation/README.md) |
| [`tools/`](tools/) | `tools/CLAUDE.md`, `src/agents/gen3_data/CLAUDE.md`, `src/main/reads/` | index: [`tools/README.md`](tools/README.md) |
| [`ops/`](ops/) | the root `CLAUDE.md` | the procedures below |

The freshness gate scans the seven topic trees for dead paths, flags and `python -m` modules and caps each doc at
120 KB (target 40 KB; `ops/` is scanned by name, uncapped). A log of CLOSED rounds is history: it goes to
`research_state/claude_md_archive/` (the Rust port's lives there), not into a tree.

**`ops/`:** `TRAINING_RUN_SOP.md` and `ORCHESTRATOR_SOP.md` are the procedures of record for the two
long-lived sessions; an owner ruling on how a session operates goes into the owning section there, not
into a memory file. `EXTERNAL_ANCHORS_SOP.md` (strength against agents we did not train; box paths in
`anchors.json`), `testing.md` and `training_runbook.md` (the root's testing and training chapters),
`cloud_gpu_benchmark.md` (the vast.ai rental runbook), `cloud_gpu_plan.md` (training on a rented card: the card rule, the mechanics, the cost), `deletion_pass_manifest.md` (the post-M5 deletion
pass), `history_rewrite.md` (the one-shot account-name purge of git history: runbook; `python -m utils.git_sha`
resolves an OLD sha an immutable record cites), and `instruction_docs_principles.md` (how every `CLAUDE.md`,
`MEMORY.md`, slash command and brief is written).

## Cross-version docs (`designs/` root)

- **`flag_registry.md`** — GENERATED from `agents/model/flag_registry.py` (`python -m
  agents.model.flag_registry`, `--check` is the gate): every model toggle's tier, class, default and
  meaning. Read it before adding or demoting a toggle; the rules are `model/flag_registry_rules.md`.
- **`deleted_flags.md`** — every deleted flag and path with its citation; the freshness gate reads it for
  names a doc cites as history.
- **`production_config.json`** + **`production_config.README.md`** (provenance). Its `recipe` block is
  hand-edited together with `endstate/design_learner_recipe.md` §3.22 (`src/recipe_doc_gate_test.py`
  fails when either moves alone). **`baselines.json`** changes only by `python -m main.baselines set …
  --reason`.
- **`design_pathologies.md`** — a HISTORICAL register of one May-June 2026 pass (banner-superseded); open questions
  live in `research_state/UNDERSTANDING.md` §6 and the ledger.

## Version folders `ai_vN` (history)

Each holds `todo.md`, `impl_step*.md` (post-implementation records; match the folder's style) and
`design_*.md` (forward designs). All are explicit-only, never auto-updated, and every closed chapter folder has a
one-paragraph `README.md` saying so (`ai_v3`'s is the frozen one). Read one for what was designed and why, then
check the code before believing it still holds.

| folder | the chapter |
|---|---|
| `ai_v1` | the initial end-to-end PPO pipeline |
| `aI_v2` *(mixed case on disk)* | the feature-extractor redesign |
| `ai_v3` | stability and signal hardening; its README is a frozen historical digraph |
| `ai_v4` | data quality and encapsulation (own-team spreads, Hidden Power inference, damage attribution) |
| `ai_v5` | self-play and the league (snapshot pool, exploiters, PFSP) |
| `ai_v6` | anticipation (the superseded MCTS route; the latent-predictive auxiliary) |
| `ai_v7` | specialisation and ladder play (forward designs) |
| `ai_v8` | the conditioning / credit-assignment epoch |
| `ai_v9` | the entity-graph generation (`design_generation_roadmap.md`); `design_frame_deletion_coverage_gaps.md` holds the standing rule on reading a dV ablation |
| `ai_v10` | exploiter scaling (forward docs, never built) |
| `ai_v11` | human ladder replays (forward doc, never built) |
| `ai_v12` | the win-probability-critic chapter, BUILT and CLOSED (the critic and its terminal are constants, `ARCHITECTURE.md`); `launch_runbook.md` is a bannered history record (its arms never ran as written); `promotion_exclusions.json` is still read by `main.promote_teams` (rebuild: `--regenerate-exclusions`) |

## `learning/`

Concept explainers, always-current, maintained by `/gen3ai-learning`; the index is
[`learning/README.md`](learning/README.md).
