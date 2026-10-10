# CLAUDE.md — `src/main/prober/` (forensic-replay inspector)

The **prober**: browse the `eval_traces` a training run writes and inspect *why the policy chose
what it did* at any saved decision point. No server, no poke-env: battles are read from the RUST
CORE.

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
python3 -m main.prober <models_dir | run_dir>        # the browser front end -> :6008
python3 -m main.prober.query <cmd> ...               # the JSON CLI, for agents and scripts
python3 -m pytest src/main/prober -q                 # the suite
```

**The DETAIL lives in `designs/prober/`** — an always-current tree that OWNS what it holds and is
updated in the same pass as the code, exactly like this leaf. This file is the rules, the
invocations, the module maps and the hazards that have already cost a wrong reading. The leaf as it
stood before the 2026-10-10 cut is frozen at
`designs/research_state/claude_md_archive/src_main_prober_CLAUDE_2026-10-10.md` (history).

| I am about to… | Read |
|---|---|
| read or change `/game`, THE battle viewer (`/battle` redirects to it) | `designs/prober/battle_view_v2.md` (field map) + `designs/prober/battle_viewer_ux_2026-10-09.md` (why it is shaped this way) |
| read a decision's panels field by field; the per-invocation flags; the label/op-order gotchas | `designs/prober/analyze_panels.md` |
| change what the RESULT timeline says happened | `designs/prober/result_timeline.md` |
| read `win` / `loss` / `draw` on a trace, or an old tree's `draw: 0` | `designs/prober/result_vocabulary.md` |
| read or change the belief / threat views, the species-clause reading | `designs/prober/beliefs_and_threats.md` |
| call `loops` / `triage` / `probe`; the investigation recipe | `designs/prober/session_scans.md` |
| call any model-free reading method; `critic_currency` | `designs/prober/session_reading.md` |
| read `analyze`'s JSON block by block; β name provenance | `designs/prober/analyze_output.md` |
| run a re-roll / search / rollout probe; the trace-quota selection; `overvalue_tau` | `designs/prober/counterfactual_probes.md` |
| touch `ProbeModel`, trace discovery, core traces, the per-battle model ladder, obs offsets | `designs/prober/engine_and_model.md` |
| diagnose a checkpoint that will not load | `designs/prober/arch_drift.md` |
| choose `--impl node` vs `rust`; `--compile` | `designs/prober/sim_impl.md` |
| groom `eval_traces/` | `designs/prober/retention.md` |
| add or find a test | `designs/prober/tests.md` |
| ask what a checkpoint REPRESENTS (`python -m main.probe_battery`, its own package) | `designs/prober/probe_battery.md` |

The Textual TUI is RETIRED: `python -m main.prober` starts the WEB app, and its two TUI-only flags
(`--ckpt`, `--inv`) print what replaced them instead of failing. History:
`designs/research_state/claude_md_archive/prober_leaf_history.md`.

## Engine / app split (the important seam)

**TWO surfaces over one engine.** `engine/` + `session/` + `model.py` are the analysis — pure and
framework-agnostic; `web/` renders it for a human, `query.py` prints it for an agent, and
`src/main/probe_replay.py` is a one-shot caller. Change the analysis once and every surface follows; neither
surface is a layer on the other. **Every torch call goes through the injected `model`**, so the engine
is unit-tested with a `FakeProbeModel` (`engine_test.py`).

`engine/` and `session/` are PACKAGES whose `__init__.py` is a re-export hub: `from main.prober.engine
import <anything>` resolves as before the split; `hub_contract_test.py` pins the export set, the
no-cycle rule and `ProbeSession`'s mixin base list. `engine/` is a strict DAG, leaves first:

| `engine/` module | holds |
|---|---|
| `views.py` | every frozen dataclass an analysis returns — the DATA MODEL. No numpy, no IO |
| `util.py` | shared leaves: percent parsing, npz access, species keys, move predicates |
| `opponents.py` | opponent-NAME ordering (sentinels first, strongest first) |
| `flags.py` | `summary_flags` + the cure-option ("heal ≠ cure") helpers |
| `protocol.py` | reading the RAW Showdown protocol out of a trace's `*_replay.html` |
| `board.py` | the BOARD read-model (`build_board`) |
| `timeline.py` | the RESULT timeline — HP loss re-attributed, one line per action |
| `beliefs.py` | species / move / exclusive-species beliefs + the refinement trajectory |
| `intent.py` | the α/β opponent-intent read + `awareness_text` |
| `spread.py` | believed vs TRUE derived spreads (the DamageOperator's stat input) |
| `switch_in.py` | forced-switch OUTGOING damage per bench candidate |
| `decode.py` | `_faithfulness` / `_matchups` / `_intervention_sweep` / `_saliency` / `_threats` / `decode_incoming_belief` |
| `analyze.py` | `analyze_invocation` — the top-level entry — plus `build_meta` |
| `taxonomy.py` | loss attribution: the turning-point category table |
| `probes.py` | representation probing (`fit_probe`) |
| `turn_events.py` | `/game`'s turn story: every state-bearing protocol line → a typed event, and the board after each turn as a fold over the same lines |
| `turn_beats.py` | the turn's events grouped into ordered BEATS (phase, execution order, consequences under their cause, faint causes, the rail's summary) |
| `perspective.py` | INFORMATION PERSPECTIVE: every board fact tagged public / ours / hidden, and `shown(vis, view)` — the ONE rule `/game`'s `fv` macro applies |
| `readout.py` | `/game`'s MODEL panels from one battle capture (opponent pointer, hypothesis tokens, attention, operator facts) |
| `scouting.py` | `/game`'s SCOUTING NOTES: per revealed opponent mon the believed moves / item / spread / HP type vs the truth |

The plain-text battle log is `engine.timeline_entry_text`, and its vocabulary (`engine.CANT_PHRASE` /
`NO_EFFECT_TEXT` / `surprise_phrase`) lives in the ENGINE so a reason one surface learns cannot go
missing on another.

The other top-level modules (full text: `designs/prober/engine_and_model.md`):

- **`model.py`** — `ProbeModel`, the torch boundary and the ONLY place a forward or backward runs.
  `load(ckpt)` is a STRICT load (`snapshot.load_checkpoint_strict`; sb3's non-strict retry is refused),
  checked against the RECORDED identity first (`arch_status`). **`capture_battle`** (`model_capture.py`)
  is `/game`'s one batched eager forward with read-only hooks, every hook removed in a `finally`.
  🚨 **A turn's events land in the NEXT decision's obs** — `describe_turn_outcome` is read from decision *T+1*.
- **`arch_status.py`** — can the MODEL views run here, from RECORDS (no torch, no checkpoint opened)?
  One typed `ArchVerdict` (`current` · `incompatible` · `unrecorded`, a `kind` from `arch_status.KINDS`,
  one plain sentence), used by the run picker (`web/runs.py`), `ProbeSession.model_status` and `ProbeModel.load`.
- **`discovery.py`** — pure filesystem, never opens a JSON or npz: `build_trace_tree` groups step →
  opponent → battle by parsing PATH STRINGS, so a 1000-battle run opens instantly; `resolve_model_for_step`
  picks the model per battle. 🚨 **The `<outcome>` alternation is BUILT from
  `agents.training.trace_result.OUTCOMES`**, never retyped.
- **`core_trace.py`** — a Rust-eval CORE TRACE ships a META-ONLY `*_summary.json`; `ProbeSession._summary`
  EXPANDS it on read from the Rust core (`core_walk.py` runs `core_events --walk`; `core_recorder.py`
  builds the summary with the live recorder's rules, labels from `agents.training.trace_labels`).
  🚨 **The stored `<prefix>.p1.jsonl.gz` record is the AUTHORITY**: a walk whose protocol or decisions
  differ RAISES `CoreTraceMismatch` — never repaired. Nothing is written into the run dir. A core trace has
  NO `*_replay.html` (stand-in: `core_trace.protocol_log`) and no auxiliary heads (`win_probs` is NaN;
  `analyze` re-runs the model). 🚨 **`core_trace` IS THE ONE `*_summary.json` READER under `src/`** —
  `load_summary` / `load_summary_meta` / `refuse_core_trace` (`CoreTraceUnsupported` by name);
  `src/trace_summary_reader_gate_test.py` fails any other opener (F-LH-5: one read ZERO decisions, silently).
- **`core_walk.py`** — the transport + alignment: `walk(record, side)`, `decision_choices` (legal action →
  choice string), `read_streams` (`core_events --obs-stream`: one side's text through the same parse chain
  training's rows are encoded on — the successor rows lookahead / better-line score), `replay_log`.
- 🚨 **No prober command needs poke-env**: every CLI command and the web app RUN with poke-env blocked
  (`src/poke_env_free_entry_points_test.py`, whose `PROBER_POKE_ENV_COMMANDS` is a closed list, EMPTY).
  `replay-counterfactual` plays out on the Rust core in process (`replay.py` → `utils/rust_env/counterfactual.py`).
- **`web/`** — FastAPI + Jinja2/HTMX over `ProbeSession`, all JS vendored. The ONLY human-facing surface.
  Its own leaf: `src/main/prober/web/CLAUDE.md`.

## Hazards that have cost a wrong reading

- 🚨 **Model-loading views work ONLY at the CURRENT architecture** (owner, 2026-10-08: no pinned-checkout
  worker). Model-FREE (`scan` · `triage` · `turns` · `battle_story` · `awareness` · `loops` · `overview` ·
  `find` bar `disagree` · `falsify` · `falsify_scan` · `calibration` · `decision_table`) works on every run;
  model-LOADING (`analyze` · `battle_readout` · `decision_attention` · `probe` · `lookahead` · `better_line` ·
  `replay_counterfactual` · `history_saliency` · `find disagree`) only on a current-arch run.
- 🚨 **Every incompatibility is ONE typed diagnosis** (`ArchDriftError.kind`, 2026-10-09), said BEFORE
  trying, including the SILENT class (weights fit, the observation changed meaning — only
  `OBS_SEMANTICS_VERSION` in `agents/model/model_version/constants.py` can see it). `ArchDriftError` is the
  EXPECTED outcome on an archived checkpoint: render its `plain` sentence, fold `exc.detail` under it, never
  collapse it to "analysis failed", and branch on `kind`, never on message text. Kinds and walls:
  `designs/prober/arch_drift.md`.
- 🚨 **Per-battle model resolution is exact → nearest → most recent**, and "most recent" is the run's LAST
  SNAPSHOT through the one choke point `agents.training.fixed_opponent_pool.resolve_model_ref`; `best_model`
  (the bot-win-rate export) is the last-resort rung and is NAMED as such (`ModelChoice.detail`) — five of five
  archived runs once loaded it while labelled "most recent". Faithfulness is exact only on the `exact` tier.
- 🚨 **A TIMEOUT ARRIVES WEARING A LOSS'S FLAGS**, and `classify_result` tests the turn cap BEFORE the loss
  (parity with the training reward pinned by `trace_result_test`). **A pre-2026-09-07 tree's `draw: 0` is NOT
  a measurement** (no `meta.result_vocabulary` ⇒ ties were dropped, timeouts written as losses); **an UNKNOWN
  result is REFUSED** (`UnknownTraceResult`), never rendered. Draws have their OWN capture quota; `calibration`
  excludes them and reports `n_draw_excluded`. Detail: `designs/prober/result_vocabulary.md`.
- 🚨 **"— no effect" IS A CLAIM** (`engine._no_effect_supported`): only a decoded fate, a sim tag
  (`|-fail|` / `|-immune|` / `|-miss|`) or a located, EMPTY window supports it; anything else renders
  `— outcome unrecorded`, and a window carrying effect tags beats every recorded outcome. "Never moved
  (fainted first)" and "couldn't move (frozen)" are distinct lines. Detail: `designs/prober/result_timeline.md`.
- 🚨 **A DETECTOR READS THE RAW PROTOCOL; ONLY A HUMAN SURFACE READS THE RENDERED TIMELINE.** `loops.py`
  keys on protocol lines, never on `engine.timeline_entry_text`.
- ⚠️ **The model's belief is `a.belief`, the raw marginals; `a.exclusive_belief` (the species-clause
  reading) is a reading aid.** Always render both — showing only the adjusted view substitutes the prober's
  arithmetic for the model's state (`app_test.py` pins the copy). Detail: `designs/prober/beliefs_and_threats.md`.
- 🚨 **`critic_currency` says what every V MEANS** (`ProbeSession.critic_mode()`): an ABSENT `critic` key in
  `model_config.json` means `shaped` (a fact about the archive), `winprob` means V ∈ [0,1] with 0.5 even.
  Never read a V without it. Detail: `designs/prober/session_reading.md`.
- 🚨 **`overvalue_tau` is in the CRITIC'S OWN UNITS** — the shaped 5.0 carried onto a win-prob critic makes
  `critic_overvalued` read a confident, false 0. `None` resolves it per run; an explicit value is honoured
  and a tau no gap can reach raises `threshold_warning`.
- 🚨 **β name PROVENANCE**: a β candidate not `revealed` is named by the species POSTERIOR, which was a mon
  not on the team at all in 73.3% of pivots — one such label produced a wrong research conclusion. Read time
  attaches `engine.BELIEF_NAME_CAVEAT` and NEVER re-derives a name. Detail: `designs/prober/analyze_output.md`.
- 🚨 **THE TRACE QUOTA PREFERS LOSSES — read `selection` FIRST** (`ProbeSession.trace_selection`,
  declared in `agents/training/trace_selection.py`). A tree that records none is `known: false`, SELECTION
  UNKNOWN — never read as uniform. Detail: `designs/prober/counterfactual_probes.md`.
- 🚨 **Move-action labels are ALREADY in action-index order — do NOT re-sort them**
  (`engine_test::test_recorded_actions_are_action_index_aligned`); the op's OUR-move blocks are
  action-ordered too, and the old "op move order ≠ action order" caveat must never come back (`app_test.py`).
  Detail: `designs/prober/analyze_panels.md`.
- **Obs offsets resolve at runtime** from `Gen3ObservationEncoder.get_layout()`; deleted regions resolve to
  **0 = absent** and every consumer no-ops on 0; `engine_test.py::test_offsets_resolve_matches_layout` pins
  them. A trace whose obs length ≠ today's `total_dim` gets the `a.obs_mismatch` banner. A missing
  `_states.npz` or `has_state=0` yields `warnings` and no panels — handled, not a crash.
- **Blocking work**: the checkpoint load and every forward/backward block. The web runs `def` handlers on a
  worker thread, loads `/analyze` via an HTMX fragment and puts minutes-long probes in the job registry
  (`web/CLAUDE.md`); `query.py` and `src/main/probe_replay.py` are one-shot and simply block.
- `models/` lives only in the MAIN checkout — from a worktree point the prober at an absolute `models/...` path.

## Agent API & JSON CLI (`session/`, `query.py`)

`ProbeSession` is the framework-agnostic facade: every method returns a JSON-serializable dict, model
loading uses the exact → nearest → recent ladder (cached per process), and a `battle_id` is the trace's
`*_summary.json` path **or** a short `step_<N>/<Opponent>/<outcome>_<idx>` id. `ProbeSession(...,
model_loader=fn)` injects a fake model in tests (no torch). It is assembled from one MIXIN per family:

| `session/` module | holds |
|---|---|
| `core.py` | `ProbeSession` itself — construction, shared internals, the resolution ladder, `critic_mode` |
| `reading.py` | MODEL-FREE orientation: `run_summary` · `battles` · `decision_table` · `battle_overview` · `battle_turns` |
| `scans.py` | run-level model-free folds: `scan` · `awareness_scan` · `loops` · `triage` |
| `trace_io.py` | the trace's sibling files (protocol log, privileged teams, our HP types) — file IO kept OUT of the engine |
| `analysis.py` | the per-decision deep read: `analyze` (loads the model) · `find` |
| `counterfactual.py` | `falsify` · `lookahead` · `better_line` · `replay_counterfactual` |
| `aggregate.py` | `falsify_scan` · `calibration` — the two run-level counterfactual folds |
| `probes.py` | `probe` · `switch_vs_info` · `history_saliency` |
| `story.py` | `/game`'s MODEL-FREE half: `battle_story` · `battle_board` (protocol fold cached per battle, dropped by `close()`) |
| `game.py` | `/game`'s model half: `battle_readout` · `decision_attention` (the capture cached per (checkpoint, battle), bounded, dropped by `close()`) |
| `serialize.py` | the JSON-shaping leaves |
| `stats.py` | pure statistics (loop aggregation, discounted returns, reliability bins) |
| `probe_targets.py` | the representation-probe target table |

The per-method reference is one doc per family: `designs/prober/session_reading.md` (model-free reading),
`designs/prober/session_scans.md` (`loops` · `triage` · `probe`), `designs/prober/analyze_output.md`
(`analyze`), `designs/prober/counterfactual_probes.md` (`lookahead` · `better_line` · `replay_counterfactual`
· `falsify` · `falsify_scan` · `calibration`).

The CLI prints JSON to stdout (`{"error": …}` + exit 1 on failure, so an agent always gets parseable
output); `--help` carries a worked example sequence:

```bash
python -m main.prober.query triage   <run_dir> [--step N] [--opponent X]          # start here for "what next"
python -m main.prober.query probe    <run_dir> <is_faster|damage_taken|faint_soon|faint_healthy|big_hit_incoming|opp_switches|opp_status_move> [--which vf|pi] [--step N] [--max-decisions K]
python -m main.prober.query switch-vs-info <run_dir> [--step N] [--opponent X] [--outcome win|loss] [--max-battles K]
python -m main.prober.query summary  <run_dir>
python -m main.prober.query list     <run_dir> --outcome loss --step 8000000
python -m main.prober.query scan     <run_dir> --outcome loss --opponent X [--metric td_residual] [--limit K]
python -m main.prober.query awareness <run_dir> [--outcome loss] [--opponent X] [--step N] [--lead-bar 5] [--cap-turn 240] [--stall-bar 0.25]
python -m main.prober.query overview <battle_id>
python -m main.prober.query turns    <battle_id>          # MODEL-FREE turn-by-turn replay of the game
python -m main.prober.query find     <battle_id> value_drop --limit 5
python -m main.prober.query analyze  <battle_id> <inv> [--ckpt PATH] [--tier auto|nearest|recent]
python -m main.prober.query lookahead <battle_id> [--inv N] [--worst K] [--seeds N] [--followup random|default]
python -m main.prober.query better-line <battle_id> <inv> [--depth 2] [--beam 3] [--top-k 4] [--interior-opponent self|ckpt|none] [--opponent-ckpt PATH] [--confirm-rollouts N]
python -m main.prober.query replay-counterfactual <battle_id> <inv> <action> [--rollouts N] [--opponent-ckpt PATH] [--opponent-source auto|bot|self|ckpt] [--opponent-regime recorded|stochastic|greedy] [--narrate]
python -m main.prober.query falsify  <battle_id> [--inv N]... [--worst K] [--seeds N] [--alts K] [--followup random|default]
python -m main.prober.query falsify-scan <run_dir> [--outcome loss|win] [--opponent X] [--step N] [--limit K] [--worst K] [--seeds N] [--alts K] [--concurrency N]
python -m main.prober.query calibration  <run_dir> [--step N] [--opponent X] [--limit K] [--worst K] [--seeds N] [--concurrency N] [--bins N] [--overvalue-tau F]
python -m main.prober.query decision-table <run_dir> [--step N]... [--opponent X]... [--outcome loss] [--cat selfko]... [--out t.jsonl] [--limit K]
# also: loops, history-saliency. GLOBAL flags go BEFORE the subcommand:
python -m main.prober.query --impl rust --compile better-line <battle_id> <inv>
```

**Investigation recipe** (full text: `designs/prober/session_scans.md`): `triage` → `summary` → `scan
--outcome loss` → `overview` the top battles → `find disagree` / `find value_drop` → `analyze` the worst
turn → `falsify` it → `falsify-scan` the run (the crater-fraction bracket; its
`critic_headroom_upper_bound` is an UPPER BOUND — read its `caveats`) → `calibration` to split the
`unattributed` bucket (selection-aware; knowingly confounded on a quota-captured sample).

## The counterfactual tier and the two global flags

`lookahead` (one-ply, common random numbers), `better_line` (a CRN-anchored beam) and
`replay_counterfactual` (play to the end on the Rust core, seeded) need the `*_reconstruction.json`
sibling and launch from the bottom of `/analyze` as password-gated background jobs. Two rules every
surface must repeat: at depth ≥ 2 `better_line`'s interior opponent is the trainee standing in for the
real one — FLAG it; and `replay_counterfactual` at `n_rollouts == 1` is a single realized-dice line, **not**
a probability (its own `caveats` say so). Detail: `designs/prober/counterfactual_probes.md`.

- **`--compile`** — `torch.compile`s the no-grad rollout models (~6.5× per B=1 CPU forward, ~10-20 s once).
  Off by default; it pays only on the SEARCH-shaped commands. Grad-enabled calls route to eager, so it
  cannot change a saliency result.
- **`--impl {node,rust}`** (default `node`) — which child the re-roll probes exec (`better-line` /
  `lookahead` / `falsify` / `falsify-scan` / `calibration`); inert elsewhere; `replay-counterfactual` runs on
  the Rust core whatever it says. 🚨 **It NEVER falls back to node** — an unbuildable binary is an error. The
  default lives on the SESSION (`ProbeSession(root, impl=…)`), and `better_line` REFUSES a warm
  `SearchSession` of the other impl, so one probe is never half on each engine.

Both in full, with the pinned node-vs-rust equivalence: `designs/prober/sim_impl.md`.

## Retention (`groom.py`)

```bash
python -m main.prober.groom <run_dir> [--keep-trace-steps 10] [--keep-snapshots 10] [--apply]   # dry-run by default
```

Scoped strictly to `eval_traces/`; a MANUAL fallback — the trainer prunes its own snapshots
(`--keep-eval-snapshots`, default 10), and `--keep-eval-trace-steps` defaults to `0` = KEEP ALL traces.
The prober itself is read-only and never grooms. Detail: `designs/prober/retention.md`.

## Tests

Everything under `src/main/prober/` is unmarked (pure, no torch, no bridge) except the
`*_integration_test.py` files that carry `@sim` (`core_trace_integration_test.py` also builds the rust env
cdylib) and `web/`'s headless-chrome render test (`@integration @browser`). What each file pins: `designs/prober/tests.md`.

`fastapi`, `uvicorn`, `jinja2` and `httpx` (starlette's `TestClient` runs on it) are pinned in
`environment_torch28.yml`; `textual` stays for the LAUNCHER's UI (`src/main/tui/`).

## Web front end (`web/`)

```bash
python3 -m main.prober.web models/                 # :6008, run picker (runs without traces hidden behind "show all")
python3 -m main.prober.web --check-openapi         # contract drift gate
```

A run is selected by NAME from the server's own listing, so no client string reaches a path join.
Reading is anonymous; the background probes and every route that loads a checkpoint or runs the model are
password-gated (`web/gate.py`, class guard `web/gate_guard_test.py`). `--impl` is a STARTUP flag there.
Deployed at prober.g5d.io; from elsewhere: `ssh -p 2222 -L 6008:localhost:6008 goodlad@workstation.g5d.io`.
Everything else: `src/main/prober/web/CLAUDE.md`.
