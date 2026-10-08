# poke-env retirement + Rust hot-path survey (2026-10-06)

Read-only survey for two owner questions (2026-10-06):

- **(A)** "How close are we to deprecating the legacy poke_env, and what do we need it for?" — widened
  the same day by owner direction: *"I would really like to not maintain the dual stack — getting off
  poke_env so we don't have change amplification."* (§A4 is the retirement plan.)
- **(B)** "How optimized is our Rust hot path in training, and is there anything else we want?"

Audited at `1220fd8a`. CPU static analysis only (an AST import scan, one CPU import trace of
`main.train_rl_agent`, code and doc reads). **No benchmark was run** — the box carried a live X5 seed
and ~7 build agents (load ~13 on 16 threads), so every timing below is a BANKED measurement, cited.
No code changed.

---

## Verdicts

- **(A)** We are **not close** to deleting poke-env, but the hard part is already built. Training's
  observation is ALREADY made by a Rust protocol reader: every training decision parses its side's
  protocol text in Rust (`BattleVersion::parse_root_unrecorded` → `parse_advance_lean` →
  `encode`, `src/rust_env/src/core/pool.rs:180,225`; program §6c). What still needs poke-env is
  everything that plays a game through the PYTHON agent (`RLPlayer`): live ladder play, the anchor
  reads, **the ELO headline (`snapshot_ladder`)** and the untaught meter, plus the prober, the Python
  parity oracle, and a handful of data-table / enum / generator uses. Retiring it from our side is
  **≈ 10–13 agent-days** in six phases (§A4). Metamon / Foul Play keep upstream poke-env in their
  own processes permanently, and that is fine.
- **(B)** The Rust hot path is **not the bottleneck**. In the X5 arms it is ~6 % of a ~50 s PPO cycle
  (3.06 s per update); the PPO update is ~82 % and GPU-bound; GPU inference (T2) is next. Making the
  Rust core infinitely fast buys ≤ ~6 %. After the already-queued GPU work (host-launch cut, bf16,
  T2 fusion) lands, the GPU **still** caps throughput (~80 % of the cycle, estimate); the CPU binds
  only if we also build rollout/update overlap, and even then not at N = 256 on this box.

---

## (A) poke-env — what we need it for

### A1. The import map (exact, AST scan of `src/`, `tools/`, `scripts/`, excluding `src/poke_env/`)

| | files |
|---|---|
| non-test importers | **75** (47 top-level, 27 lazy-only, 1 `TYPE_CHECKING`-only: `agents/action/mapper.py`) |
| test importers | **97** (16 lazy-only) |
| `tools/` | 1 (`tools/pokemon_data_extractor/sync.py`, lazy `GenData`) · `scripts/`: 0 |
| the vendored fork | 15,563 non-test lines + 13 test files in `src/poke_env/` |

The brief's ~104 / ~110 do not reproduce; 307 files merely MENTION the string (docstrings, comments).

**By role** (counts are non-test files):

| Role | Files | What poke-env provides | Load-bearing? | To replace |
|---|---|---|---|---|
| **Training hot path** | 0 at runtime; **36 poke-env submodules load at import** | nothing at runtime (the Rust env core builds rows, `deletion_pass_manifest.md:44`) | no — but see Finding A-F1 | P1 below |
| **Python eval / meters** | ~8: `snapshot_ladder.py` (721–744), `untaught_meter.py` (604–692), `main/eval_worker.py` Python branch, `eval_roster.py`, `churn_probe.py`, `cf_producer_snapshot.py`, `bot_elo_calibration.py` | `Player`, `RandomPlayer`, `SimpleHeuristicsPlayer`, `LocalhostServerConfiguration` → `RLPlayer` over the in-process bridge | **yes** — `ladder.json` (the ELO headline) is played by `RLPlayer` + the Python encoder (VERIFIED, `snapshot_ladder.py:721-744`); spawned DETACHED during training on every promotion (`selfplay_callback.py:756,820`) | port onto the `main.h2h` / Rust eval core engine (P2) |
| **Websocket live play** | 7: `main/play.py`, `agents/inference/player.py` (`RLPlayer`), `main/anchors/{session,runner,mirrored}.py`, `main/anchors/peer_scripts/metamon_side.py`, `main/ladder_usage_smoke.py` | login, `/search` ladder, challenge/accept, `wss://` config, team packing, the protocol parse + state tracker feeding `Gen3Battle` → `LiveView` → Python encoder | **yes** | P3 (anchors) + P4 (ladder) |
| **Python battle layer + encoder** | ~25: `agents/battle/{gen3_battle,live_view,battle_event,offline_feed}.py`, `agents/action/serialize.py`, `agents/gen3_mechanics.py`, `agents/training/{turn_delta,episode_tracker,battle_snapshot}.py`, `agents/opponents.py` (8 scripted Python bots), `agents/bc/log_reader.py`, the 10 `agents/observation/*` encoders | `Battle`, `Pokemon`, `Move`, `AbstractBattle`, `Effect`, `SideCondition`, `to_id_str`, `GenData` | **yes**, twice over: it is what `RLPlayer` encodes with, AND it is the Rust encoder's **parity oracle** (`design_three_tier_environment.md` §6; `program_rust_core.md:44`; gates: `rust_core_parity{,_obs,_trackers,_views}.py`, `core_row_parity_fuzz_test.py`, `rust_core_present_test.py`, `rust_core_trackers_fuzz_test.py`, the obs golden `golden_obs_capture.py`, `poke_env_findings.py`) | P4 gate replaces the oracle (§A4.2), P6 deletes |
| **Generators of Rust source** | 3: `agents/battle/rust_core_present_tables.py` → `src/rust_sim/src/present/tables.rs`; `utils/rust_env/bot_tables.py` → `src/rust_env/src/bots/tables.rs`; `agents/battle/rust_core_schema.py` → `src/rust_sim/src/core_events/schema.rs` (imports `poke_env.player.player.Player` + Python `battle_event`'s keyword tables) | private effect sets, pokedex, `GenData`, heuristic-bot constants, the **protocol keyword schema** | **yes** — the Rust reader's keyword schema is GENERATED from the Python classifier (VERIFIED), so Python is still the schema's source of truth | P1: freeze the generated files as Rust-owned source, or regenerate from `data/` |
| **Teambuilder / data / enums** | ~6: `utils/teambuilder.py`, `utils/team_sources.py`, `agents/training/team_archetypes.py`, `species_priors.py`, `agents/enums.py`, `agents/training/clone_pins.py` | `Teambuilder` packing, `to_id_str`, `GenData`, the enums (`MoveCategory`, `PokemonType`, `Status`, `Weather`) re-exported BY IDENTITY | yes, but trivially portable | P1 |
| **Bridge / replay plumbing** | 7: `utils/bridge/{battle_stream_client,local_battle_runner,counterfactual,reconstruction}.py`, `agents/training/obs_materializer.py`, `main/collect_replays.py`, `agents/battle/rust_core_schema.py` | `PSClient`, `POKE_LOOP`, `SingleBattleOrder`, `BattleSpectator` | only as the seam `RLPlayer` attaches to | dies with P2–P4 |
| **Prober** | 4, all lazy: `main/prober/{core_trace.py, replay.py, session/counterfactual.py, engine/switch_in.py}` | `RLPlayer` re-parse, `PokemonType` | yes for model-loading views | P5: walk core versions |
| **Benchmarks / goldens** | 7: `obs_build_benchmark`, `trainer_turn_benchmark`, `live_view_build_benchmark`, `utils/bridge/{search_impl_throughput,ws_frontend_throughput}_benchmark.py`, `rust_sim/harness/{search_golden,gen_search_golden,better_line_bench}.py` | the Python obs path | the first three measure a path training no longer runs (Finding A-F5) | delete with P6; the Rust encoder needs its own benchmark |
| **Search (wound down)** | 4: `main/search_dividend/{battery,playoff,determinize}.py`, `agents/training/replay_imputation_probe.py` | the Python battle layer | no | delete (owner yes) |
| **Tests** | 97 (agents/training 42, utils/bridge 11, agents/battle 11, agents/observation 7, agents/action 5, utils/rust_env 3, ~15 singles incl. the three poke-env gates) | — | follow their code | migrate or delete in P6 |

### A2. Dead or removable now (each needs an owner yes; ≈ 0.5–1 agent-day together)

`main/collect_replays.py` (only ai_v6 docs + `scripts/workstation/GCP_INFRASTRUCTURE.md` reference it) ·
`agents/training/replay_imputation_probe.py` (+ its `strict_api_lock_test.py:99-110` allowlist rows) ·
`main/ladder_usage_smoke.py` (one-off, measurement banked 2026-09-24) · `main/search_dividend/*` if search is
formally closed · the Python eval branch of `main/eval_worker.py` (already a P-row in
`deletion_pass_manifest.md:477`). No module has literally zero references. The three Python-obs benchmarks
**cannot** go before P4 — `play.py` still runs that path.

### A3. Existing plan of record

- **`design_three_tier_environment.md`** ("…and the retirement of poke-env") §6: poke-env leaves the
  production path and **survives as a parity oracle**.
- **`program_rust_core.md` §M7** (line ~846): `play.py` becomes parse → version → encode → T2 → act,
  gated by `ladder_drift_scan` and a Metamon / Foul Play cell with byte-identical actions; the prober walks
  versions; "removes the last production use of poke-env", **3–4 agent-days**. Its **Rust-only readiness**
  (owner 2026-09-28) requires (1) a GENERATED importer inventory, (2) a shrinking import gate, (3) per-consumer
  parity on the same recorded battles, (4) the test tiers re-run with poke-env UNINSTALLABLE. **None of the four
  is built.**
- **Backlog:** T9 (M7, size L) and T10 (deletion pass part 2, size M), `designs/ops/TASK_BACKLOG.md`;
  neither scheduled.
- **Already deleted** (U3/D2/D4, 2026-10-02): the Python env core (`gen3_env.py`, a poke-env `SinglesEnv`),
  wrappers, env factory, async vec env, `bridge_session.py`, `--env-core` / `--use-bridge`. **Explicitly kept**
  (`deletion_pass_manifest.md:44`): the trackers, `TurnDelta`, `episode_tracker`, the assembler — because
  `RLPlayer` still runs them.
- **The fork is still maintained**: reading fixes R1–R3 (`gen3_pe_reading_fixes_v1`), F-LF-1, the
  Metronome/Assist `[from]` fix; `TECH_DEBT_BACKLOG.md` §(a) lines 73–87 lists open poke-env READING defects
  (Mimic'd Hidden Power, Yawn, Sleep Talk into Pressure, Skill Swap, Focus Band/Spite, and a **P2 Solar Beam
  assertion that loses a live game on the timer**). Each is a second copy of a fact the Rust reader already
  reads — the change amplification the owner names.

---

## A4. The retirement plan — ONE stack (Rust everywhere; poke-env gone from OUR side)

### A4.1 What already exists, and what a live-play reader still needs

**Exists (VERIFIED in code):**

- **The reader.** `core_events::parse` / `Line::parse` (`src/rust_sim/src/core_events/`) reads one side's
  text into typed events; `BattleVersion::parse_root(viewer, username, packed_team)` /
  `parse_root_unrecorded` + `parse_advance_lean` is a PARSE-BUILT partial-information chain with trackers;
  `BattleVersion::encode` writes the 2761-dim row; `present::mask` / `present::choice_tokens` give the mask
  and the choice strings. **Training runs exactly this on every decision.**
- **Its keyword table already covers the live-server lines** — `|inactive|`, `|raw|`, `|t:|`, `|c|`,
  `|j|`, `|html|`, `|-hint|` (`core_events/schema.rs`) — and an unknown keyword is REFUSED
  (`LineError::UnknownKeyword`, `line.rs:326`), the same tripwire `battle_event.classify` is.
- **Correctness chain** (program §6b): our emitted text == Node Showdown's (protocol-emission phases, e2e
  capstone, differential fuzzers) and `parse(emit) == step` (M1's gate) — so parsing Showdown's text yields what
  our simulator would have typed. **The `--server rust` websocket front end** (`utils.bridge.ws_frontend`) is
  the reverse direction (our sim served to a poke-env-style client) and is already the default for anchors.

**What a live reader still needs:**

1. **A session entry point** — an FFI (or stdio) "reader session": open(viewer, username, packed team) →
   feed(lines) → on a decision, (row, mask, tokens, rqid). Every piece exists; the wrapper does not (`core_events`
   re-runs an engine, `sim_bridge` reads a battle it runs; neither reads a foreign stream). ≈ 0.5 agent-day.
2. **Real-server framing.** A real server sends `|request|` (with `rqid`) as its own `sideupdate`, and the
   battle chunk separately; the core's rule is "the decision's request is the LAST line of the write"
   (`encoder.md` §5a). The client must assemble a write exactly as the bridge does, or the row is built at the
   wrong boundary. **GIGO risk #1.** `rqid` must be echoed (`ws_frontend.md` "The rqid").
3. **A thin Python websocket client** replacing `PSClient`/`Player`: official login (`action.php` assertion),
   `/trn`, `/search gen3ou`, challenge / accept, `/choose … |rqid`, the timer, `|error|[Invalid choice]`
   re-choose. Reconnect stays out (as today: `ladder_readiness.md:24`). Inference stays Python (CPU or T2).
4. **The drift gate re-pointed at Rust.** `ladder_drift_scan` today feeds public replays through Python's
   `battle_event.classify` + `Gen3Battle` + `gen3_effects`. It must feed them through `Line::parse` and a
   spectator-side parse chain (a viewer with no team: `parse_root(0, "me", None)` is already exercised in
   `core_error.rs:215`) plus the encoder's effect classification. Otherwise it guards a client nobody runs.
5. **Schema ownership.** `schema.rs` is generated from Python's `battle_event` (and imports poke-env); after
   retirement the Rust file becomes the source, and a new keyword is added there (with a drift-scan row).

### A4.2 What replaces the Python encoder as the parity reference

The Python encoder is a **second implementation**, not a truth; `present.md` §3 already says "parity with
poke-env is not the goal; truth is". Its replacement is four checks, none of which needs Python:

| Check | Proves | Status |
|---|---|---|
| **Engine-truth audit** — the core reading's sim-fact fields vs the omniscient engine at every decision (`present/audit.rs`, `check_view`) | the reader reads what HAPPENED | exists |
| **Round-trip chain** — emitted text == Node Showdown's; `parse(emit) == step` | the reader reads Showdown's text as our sim typed it | exists |
| **Two roads, one row** — replay a battle's per-side protocol (as a websocket client would receive it, real-server framing included) through the NEW reader session and require its row, mask and tokens BYTE-equal to the `__OBS__` frame training's chain produced for the same battle and decision | the live reader = the training reader | **to build (P4)** |
| **Frozen goldens** — the obs golden and the parity corpus re-recorded ONCE while both stacks still agree, then owned by Rust; a later change re-records from Rust with a reviewed cell diff | regressions on banked battles | re-record in P6 |

Losing the Python oracle loses one thing: an independent second reading of facts the engine CANNOT audit
(beliefs, inferred items, HP-percentage folds). Those are already checked against the engine where a truth exists,
and the poke-env disagreements were registered as findings rather than rules (`poke_env_findings.py`), so this
loss is small — but it is a loss, and it is named as a risk below.

### A4.3 What gets deleted, and what depends on each

| Delete | Depends on it today | Unblocked by |
|---|---|---|
| `agents/battle/` Python layer (`Gen3Battle`, `LiveView`, `TurnView`, `LegalActions`, `battle_event`, `offline_feed`) + trackers, `TurnDelta`, `episode_tracker`, `battle_snapshot` | `RLPlayer` (play, anchors, snapshot ladder, untaught meter, eval worker), the prober's replay, the drift scan, the parity harness, the schema generator | P2–P5 |
| `agents/observation/*` (the Python encoder, ~10 modules) + `assembler` + the `live_view` memo | `RLPlayer`; slice O parity; the obs golden capture; 3 benchmarks | P2–P4, goldens re-owned in P6 |
| The **strict-view lock** (`src/agents/strict_api_lock_test.py` and its allowlist) | exists only to police reads of the Python read-models | dies with the layer |
| `agents/inference/player.py` (`RLPlayer`), `agents/opponents.py` (8 Python bots; the Rust core already has its bot ports, `src/rust_env/src/bots/`) | play, anchors, meters, `eval_roster` | P2–P4 |
| `utils/bridge/{battle_stream_client,local_battle_runner,counterfactual,reconstruction}.py`, `obs_materializer.py` | `RLPlayer` over the in-process bridge | P2–P4 |
| Generators `rust_core_present_tables.py`, `bot_tables.py`, `rust_core_schema.py` | their `.rs` outputs | P1 (freeze outputs as source) |
| `agents/enums.py` re-exports, `to_id_str`, `Teambuilder` packing, `GenData` uses (teambuilder, team sources, archetypes, species priors, clone pins, `tools/…/sync.py`) | **`agents.gen3_data`** (via `moves.py:21`), the enum-compare gate | P1 |
| Benchmarks `obs_build_benchmark`, `trainer_turn_benchmark`, `live_view_build_benchmark` (root CLAUDE.md calls the first MANDATORY) | nothing in training | P6 (+ a Rust encoder benchmark replaces the mandate) |
| ~97 test files (incl. the three poke-env gates `poke_env_fork_gate_test`, `poke_env_enum_str_compare_gate_test`, `reading_fixes_test`) | their code | P6 |
| **`src/poke_env/`** (the vendored fork, 15.6k lines) | all of the above | P6 |

### A4.4 Metamon / Foul Play stay poke-env — isolated

They subclass **upstream** poke-env 0.8.3.3, and one process resolves `import poke_env` to exactly one copy
(`ws_frontend.md` verdict (d)). They therefore already run in **their own process and their own env**
(`main/anchors/peer_scripts/metamon_side.py`, a separate interpreter) and reach us over a SOCKET (the `--server
rust` front end). After retirement that is unchanged, and it becomes the ONLY poke-env on the box: it lives in
the Metamon env, never in `gen3ai_torch28`. What must hold: the front end keeps serving the poke-env client
dialect (its byte-differential gate vs Node stays), and the anchors' peer script is the one place in our tree
that may name poke-env (an allowlisted path in the P0 import gate, run by the peer's interpreter only). Known
cost carried over: the Metamon Baton Pass bias (H18, TECH_DEBT P2).

**A simplification for anchors (proposal, UNVERIFIED feasibility):** with `--server rust` the battle runs in
OUR `sim_bridge`. Our side does not need to be a websocket client at all: the front end can drive our slot
in-process from the bridge's core-obs `__OBS__` frames (the training reader, unchanged), with only the
opponent on the socket. That retires poke-env from anchors **without** the foreign-stream reader, so P3 is
smaller than the ladder work. `--server node` anchor reads would still need P4.

### A4.5 Phased plan

1 agent-day ≈ one agent-day of build with gates. Each phase ships alone.

| Phase | Work | Gate that proves it before anything is deleted | Agent-days |
|---|---|---|---|
| **P0** | Generated importer inventory + routine-tier import gate with a shrinking allowlist (M7 readiness 1–2); the scan behind §A1 is the prototype | the gate fails on a new importer | 0.5 |
| **P1** | Off the import path: own enums (keep the enum-compare gate's teeth), `to_id_str`, team packing → `utils.teambuilder`, `GenData` → `agents.gen3_data`; freeze `schema.rs` / `present/tables.rs` / `bots/tables.rs` as Rust-owned source (or regenerate from `data/`). Trainer's import closure poke-env-free | import gate: `main.train_rl_agent`'s closure has 0 `poke_env` modules; byte-equal generated files | 1.5 |
| | ↳ **P1 STATUS (2026-10-07): DONE.** Enums, `to_id_str` and team packing are owned (`agents/enums.py`, `utils/showdown_id.py`, `utils/team_packing.py`; the fork re-exports them by IDENTITY); the eval facade, the training roster and the reward imports were cut; the three generators are frozen and deleted (`.rs` bodies byte-equal, header comment only); the trainer, h2h and plateau run with poke-env BLOCKED (`utils/poke_env_blocker.py`, `src/poke_env_free_entry_points_test.py`); allowlist 165 → 138. Two plan deviations: (1) the plan's "None of P0–P2 touches obs code" is true of values but P1 edited 9 files in `agents/observation/` (annotation-only imports moved under `TYPE_CHECKING`, `global_env`'s screen ids as strings) — value-neutral, obs golden unchanged; (2) `GenData` → `agents.gen3_data` was NOT done for `clone_pins.py` / `tools/…/sync.py` (acquisition and the prober's clone sharing need the fork's `GenData` object itself). | | |
| **P2** | Meters onto the Rust eval core (`main.h2h` engine): `snapshot_ladder`, `untaught_meter`, `bot_elo_calibration`, `churn_probe`, `cf_producer_snapshot`; delete the eval-worker Python branch | a banked ladder's edges re-played on the Rust engine agree within the delta's CI (equivalence rule) at matched regime | 1.5–2 |
| **P3** | Anchors: our side as an in-process core slot of the `--server rust` front end (A4.4), Metamon / Foul Play over the socket | byte-identical ACTIONS vs today's `RLPlayer` client on the same seeded battles (M7 gate), both still present | 1–1.5 |
| | ↳ **P3 STATUS (2026-10-07): DONE.** A4.4's simplification held: the front end runs IN the anchors process (`main.anchors.server.InProcessFrontEnd`) and our side is a slot of it (`main.anchors.core_side`) that decides on `sim_bridge`'s `__OBS__` frame and answers with its choice token; only the opponent is on the socket. `--our-transport auto` (default) → `core`, the legacy `RLPlayer` client kept for `--server node` / `--server-uri` / a `bot:` our-side; `our_transport` stamped per row. Gate: **100 / 100 seeded battle pairs byte-identical** (both sides' choices + every per-side chunk; greedy away 40, greedy home 40, seeded T = 1.0 20), 47/53 both paths, 0 protocol failures (`measurements/pokeenv_p3_anchors_2026-10-07/`). `main.anchors` imports AND runs with poke-env blocked; allowlist 136 → 135 (after P5) (`main/anchors/runner.py`; `session.py` keeps the legacy client). | | |
| **P4** | Ladder / challenge on the public server: the reader session (A4.1 item 1), real-server framing, the thin websocket client, the Rust drift scan | (a) **two roads, one row** on every decision of ≥ 1,000 captured battles; (b) ≥ 200 public master replays parse with 0 refusals (Rust drift scan); (c) a challenge series vs Metamon on a local **Node Showdown at master** (not the pin) — real framing, real `rqid`; (d) **shadow mode**: the reader runs beside the poke-env client on those games, row / mask / action compared decision by decision, 0 diffs (or each diff a registered finding) | 2.5–3 |
| | ↳ **P4 STATUS (2026-10-07): DONE.** The reader session is `live_reader` around `pokesim::side_reader::SideReader` (the chain `sim_bridge`'s core_obs keeps, now one type); the thin client is `main.live` (`main.play`'s default); T28's halt is built. Gates PASS ([`../pokeenv_p4_live_2026-10-07/`](../pokeenv_p4_live_2026-10-07/)): (a) 2,224 battles (2,000 random-policy + 224 model, both sides of the v144 version break; the plan's floor was 1,000) / 318,465 decisions, 0 differences; (b) 376,410 public replays (not 200), both seats, 0 unclassified refusals, 1 spectator-only refusal root-caused; (c) 315 games on a LOCAL master-built Node (bots, Metamon SmallRL, self-vs-self, the shadow peer), 0 T28 halts; (d) 16,523 shadow decisions, 0 differences. Deviations from the plan: the Rust drift scan is `main.live.replay_scan` beside, not inside, `ladder_drift_scan.py`; no reconnect inside a battle (by design); the public login is built, never run. | | |
| **P5** | Prober walks core versions instead of re-parsing traces | the prober's views on core-walked traces equal today's on a fixed trace set (M7 readiness 3) | 1 |
| | ↳ **P5 STATUS (2026-10-07): DONE, bar one command.** A core trace's expansion is the core's walk (`core_events --walk` → `main/prober/core_walk.py` + `core_recorder.py`); falsify / lookahead / better-line read choice maps and successor rows from the core (`core_events --obs-stream`); every JSON-CLI command except `replay-counterfactual` and the web app RUN with poke-env blocked; allowlist 138 → 136. Identity on 21 banked battles (3 runs): `measurements/pokeenv_p5_prober_2026-10-07/`. **Deviation:** `replay-counterfactual` stays on poke-env — it plays the rest of a battle LIVE with `RLPlayer` + the Python bots, so its port is a Rust play-out (`utils.rust_env.successors.play_out` + the in-core bots), P6-blocking and not costed in this phase's 1 day. | | |
| **P6** | Deletion pass part 2 (T10 widened): the battle layer, the encoder, trackers, `TurnDelta`, strict-view lock, bridge JSON road, `RLPlayer`, Python bots, three benchmarks, ~97 tests migrated or deleted, the goldens re-owned by Rust, **`src/poke_env/` deleted**; test tiers run with poke-env UNINSTALLABLE (M7 readiness 4); root + leaf CLAUDE.md updated (the "MANDATORY obs benchmark" becomes a Rust encoder benchmark) | the full tier set green with poke-env absent from the env | 2–3 |
| | ↳ **P6 slice 1 STATUS (2026-10-07): DONE.** `replay-counterfactual` (P5's one deviation) plays out on the Rust core (`gen3_cf_core_playout_v1`: `utils/rust_env/counterfactual.py`; the playout took a divergence-turn root, an in-core bot side, several stall sides and a side's text); every prober command and the web app RUN with poke-env blocked (`PROBER_POKE_ENV_COMMANDS` empty); identity old vs new: `measurements/pokeenv_p6_replay_cf_2026-10-07/`; allowlist 135 → 133. `utils/bridge/counterfactual.py` stays for `cf_producer` and the search-dividend playoff (their port is a later slice). | | |
| | **Total** | | **≈ 10–13** |

**Ordering against the pre-plateau architecture work.** The static-token rebuild (owner, 2026-10-05; the entity
audit, 2026-10-06) is where the dual stack doubles the cost: its build stage 1 (`0ea14560`, `--token-encoding`)
is model-side only and touches no encoder, but any obs fact the rebuild ADDS (the entity audit's board facts, an
amount the switch cell lacks) has to be written twice today. Recommended order:

1. **P0 + P1 + P2 now** (≈ 3.5–4 days). None of them touches obs code, so they run beside the arch design
   without a file conflict, and P2 is a hard prerequisite: the ELO headline is played through the Python encoder,
   so a Rust-only obs change would break `ladder.json` for the new-arch runs (it is spawned during training).
2. **Then make the rebuild Rust-only**: freeze the Python encoder at the current layout, re-record the obs golden
   from Rust, and drop slice O for the new layout. From that point every obs fact is built once.
3. **P3 and P5** alongside the rebuild (before the first anchor read / prober session on a new-arch checkpoint);
   **P4 before the ladder campaign's first live stage**; **P6** after P4's shadow-mode gate is green.

### A4.6 Risks

- **GIGO in the reader on a live stream** (framing order, `rqid`, a master-only argument shape). Mitigation: P4's
  four gates, shadow mode before cut-over, the reader REFUSING unknown keywords (a lost game on the timer beats a
  silently wrong row).
- **Server drift** — the public server runs Showdown master, we pin `deps/pokemon-showdown`. The Rust drift scan
  must exist before the Python one is deleted; the Showdown version stays stamped on every online record.
- **The independent oracle disappears** (A4.2's named loss) for facts the engine cannot audit.
- **Ladder readiness is not improved by retirement alone**: no reconnect at any layer (`ladder_readiness.md:24`);
  the open live crashes (Transform + Hidden Power belief T11, Imprison T12) must be re-checked on the Rust reader
  (the Solar Beam assertion is a poke-env bug and disappears with it).
- **A reading fix is a training-input change** (`design_three_tier_environment.md:179`): the live reader must be
  the SAME chain training runs, never a fork of it — that is what "two roads, one row" enforces.
- **Old traces**: a meter that replays a pre-retirement trace through the Python path loses that ability (pinned
  legacy runs are unaffected — they run at their own commit).

---

## (B) The Rust hot path

Sources: ledger 2026-10-04 MEASUREMENT "PRE-X26 BOTTLENECK PROFILE" and
[`../bottleneck_profile_2026-10-03/README.md`](../bottleneck_profile_2026-10-03/README.md) (production shape,
pool pre-seeded to 20 snapshots); [`../x5_paired_speed_2026-10-05/README.md`](../x5_paired_speed_2026-10-05/README.md);
[`../m5_sizing/PROGRESS.md`](../m5_sizing/PROGRESS.md); and the TensorBoard scalars + child logs of the FINISHED
`models/rb_x5ab_*` runs (2026-10-04 → 06; read-only). Shape: N = 256 envs × n_steps 384 = **98,304 rows per
update**, 10 epochs, RTX 3080 Ti, fp32. X5 columns = medians over updates ≥ 6 of blob s1001–1005 and fixed_mass
s1001–1005 + s1006b.

### B1. Wall-clock breakdown

| Row | Profile (20-snapshot pool), s / update | X5 blob, s / update (s per 1M steps) | X5 fixed_mass, s / update (s per 1M) |
|---|---|---|---|
| **Rust env stepping** (engine + protocol emit + parse + encode) | 3.0 (7.5 ms / host step) | **3.06** (31); 7.3–7.6 ms / step | **3.05** (31) |
| obs encoding, alone | **GAP** — no in-core split | GAP | GAP |
| shm / column transfer | inside "glue" | GAP (arena write 0.47 ms / step ≈ 0.2 s) | GAP |
| **GPU inference, T2** (trainee + opponents) | 8.6 (6.7 ms launch + 14.8 ms wait / step) | **4.17** (42) | **5.85** (60) |
| host glue (submit, draws, write, post) + fill | 1.1 | 1.03 (10.5) | 1.03 (10.5) |
| unattributed collect residual | ≈ 0 | 0.59 (6) | 0.77 (8) |
| **PPO update (`train_ms`)** | **40.0** | **41.1** (418) | **46.4** (472) |
| eval cycles (blocking, every 2M) | 0.96 amortised (19.5 s / cycle) | 0.79 (8) | 0.88 (9) |
| startup / restarts (amortised) | — | ≈ 2.4 (≈ 25); T2 up in ~245 s, ~370 s launch → first update | ≈ 4.3 (≈ 44); T2 up ~495 s |
| idle / sync waits | ~8 % of the update GPU-idle; ~11 s / update of host spin-block in ~764 D2H scalar reads | GAP (no in-run instrument) | GAP |
| **median cycle** | **54.4** | **49.9–50.6** | **57.2–58.8** |

The X5 runs had no restarts (15M < the 6 h interval). `eval/duration_sec` (~330 s) is SUMMED game time, not wall
(`eval/wall_sec` is wall) — see Finding B-F6.

### B2. Effective throughput (trainee steps / s)

| | steady state | end to end (launch → `final_model.zip`) |
|---|---|---|
| X5 blob | **~1,966** (`time/fps` median 1,894–1,992) | **1,776–1,833** (~2.3 h per 15M) |
| X5 fixed_mass | **~1,718** (`time/fps` 1,729–1,758) | **1,491–1,569** (~2.7 h per 15M) |
| oracle arms | — | species-only 1,828–1,832; full reveal 1,792–1,815 |
| production profile, 20-snapshot pool | ~1,807 | — |

fixed_mass's cost over blob: +16.7 % in the paired benchmark (54.43 vs 63.53 s / cycle), +14.3 % on steady-state
cycles inside the arms, +16.9 % end to end (its T2 startup is twice as long). See Finding B-F2.

### B3. The bottleneck is the GPU

- The update is **74–82 %** of the cycle and **GPU-bound**: 92 % busy; fp32 GEMMs at 11.4 TFLOP/s (46 % of the
  measured 24.9 peak); memory-bound Triton kernels at 75 % of the measured DRAM roof; the host main thread is
  0.92 CPU busy and becomes the next wall at ~1.4× faster GPU (profile estimate).
- T2 inference is latency-bound (~21 CUDA graphs × ~590 small kernels per host step at 20 snapshots) and is
  larger than the Rust core inside the rollout.
- **The Rust core is CPU-bound but small**: 6.7–7.0 of 8 worker CPUs busy during its step; 16 threads buy only
  +4.0 % [3.5, 4.5] at N = 1,024 (`m5_sizing`); the box uses 2.4–2.9 of 16 CPUs over a whole cycle. It is sensitive
  to box load (a live seed on today's loaded box reads 10.0 ms / step vs 7.3 quiet — contaminated, not a
  measurement).

### B4. CPU-side optimizations already in place

- **Persistent worker pool**: T threads over contiguous env blocks, spawned once and frozen; one mpsc message
  per worker per step; results folded in env order for determinism (`src/rust_env/src/core/pool.rs`). Default
  8 threads (`src/main/train/rust_env_setup.py:25`), capped at N.
- **Observation encoded in Rust** straight into the caller's column when a decision opens.
- **Column arena, two front ends**: a separate process over shared memory (`shm.rs`, `utils/rust_env/proc.py`,
  the default) or in-process FFI (`ffi.rs`); equal at N = 48 (12 µs / decision, 82k decisions / s, ledger
  2026-09-26).
- **Scripted bots inside the core** (`src/rust_env/src/bots/`): bot opponents cost no GPU.
- **T2**: CUDA-graph backend with bucketed shapes, 31 slots over 8 lanes; the AOT package backend works again
  (`174f5e62`).
- **Learner**: fused optimizer (760 launches, 0.01 s / update); pinned non-blocking staged H2D (11.9 GB / update,
  overlapped); `--compile-trainer` on.
- **Net**: M5 = 5.11× [4.72, 5.63] the trainee decisions / s of the old Python path at 0.038× the CPU per
  decision (ledger 2026-09-30 M5 GATE MET, N = 48).

### B5. Already queued GPU-side — not re-proposed

CUDA-graph / host-launch cut for the R1 update micro-step (removes ~764 D2H reads + ~950 syncs, fuses ~177k
tiny grad-accumulate kernels; ~2–3 s / update) · **bf16 with decisions in fp32** (update 40 → ~20 s ESTIMATE;
pays only after the launch cut) · **T2 fan-out / grouped opponent forward** (trigger FIRED 2026-10-02: opponent
inference 38.6 % [38.3, 38.9] of the serial host step at N = 256; up to ~5 s / cycle at a 20-snapshot pool) ·
a torch upgrade if it carries these (T14 → 2.8 is done; nothing past 2.8 is written down) · opponent bf16 =
EXPERIMENT_BACKLOG X19 · restart interval = T24. Status: none built; the list lives in the owner's 2026-10-06
bounded performance list before the deep run (see Finding B-F1).

**After those land, the GPU still caps throughput.** In a blob screen: update ~20–22 s vs collect ~9 s
(~4 s of it T2 latency, ~4 s CPU) → GPU ~80 % of the cycle, CPU ~13 % (ESTIMATE from the profile's
component times). At the X26 20-snapshot pool, T2 grows to ~8.6 s and becomes the second limit.

### B6. What else we want, ranked by expected wall-time gain

| # | Item | Expected gain | Rests on |
|---|---|---|---|
| 1 | **Overlap the next rollout with the current update** (async collection; one-policy-version-stale rows) | up to the whole collect share: ~18 % of today's cycle, ~30 % after bf16 | **UNVERIFIED** — not measured anywhere. Not the same as within-rollout step/inference overlap, which was measured SLOWER (0.855× [0.851, 0.860] at N = 256, sizing O5). Costs: staleness (an algorithmic change, needs its own A/B) and T2 contending with a 92 %-busy update on one GPU |
| 2 | **Startup**: cache T2's compile / package; break startup down | T2 up 245 s (blob) / 495 s (fixed_mass) ≈ 4.5 % / 6.8 % of a 15M screen; ~1.7 % per 6 h restart (UNVERIFIED) | `x5_paired_speed` logs; T24 open |
| 3 | **Close the profiling gaps first** (before any CPU work): an in-core split (engine / protocol emit / parse / encode), a shm-transfer timer, the 0.6–0.8 s unattributed collect, an in-run idle/sync meter, a startup breakdown | enables the rest; tells us what the parse-through-text design (program §6c) costs | gaps listed in B1 |
| 4 | Rust release profile tuning (`lto`, `codegen-units = 1`, `target-cpu=native`) — today no `[profile.release]` overrides in either `Cargo.toml` | ~5–15 % of core time ≈ 0.2–0.45 s / cycle (< 1 %) | **UNVERIFIED** (typical range) |
| 5 | Work-stealing instead of static env blocks + a per-step barrier (6.7 of 8 CPUs busy) | ≤ ~16 % of core time ≈ 0.5 s / cycle (~1 %) | **UNVERIFIED** |
| 6 | More threads | +4 % of core step ≈ 0.1 s | measured (`m5_sizing` threads addendum) — skip |
| 7 | Allocation churn (a report vector per worker per step + a sort) | negligible | **UNVERIFIED** |
| 8 | Eval (1.6–1.7 %), host glue (~2 %) | small | B1 |

**Rust-side ceiling: ~6 % of the cycle in total.** Items 4–7 together are worth ~1–2 % (UNVERIFIED). The Rust
core binds only if item 1 is built AND the GPU work lands, and even then the cycle is max(update ~20 s,
collect ~9–13 s) with the core at 3 s of the collect — so it does not bind at N = 256 on this 8C/16T box.

---

## Findings (rule 7)

**(A) poke-env**

- **A-F1 — the trainer loads poke-env at import** (36 poke-env submodules through 25 repo modules) although
  training never calls it. Root: `agents/gen3_data/moves.py:21` → `agents/enums.py:28-31` re-exports poke-env's
  enums by identity, so the data facade root CLAUDE.md calls "poke-env-free" is not.
- **A-F2 — the ELO headline (`ladder.json`) and the untaught meter are played through poke-env + the PYTHON
  encoder** (`snapshot_ladder.py:721-744`, VERIFIED; `untaught_meter.py:604-692`), while the trainee trained on Rust
  rows. Any Python/Rust encoder gap the parity gates miss reaches the headline meter. Neither has a Rust path, and
  a Rust-only obs change would break both for new runs.
- **A-F3 — the Rust reader's keyword schema is generated from Python** (`agents/battle/rust_core_schema.py`,
  which imports `poke_env.player.player.Player` + `battle_event`): the Python classifier is still the source of
  truth for what the Rust reader accepts.
- **A-F4 — an open P2 live-play crash in the fork**: a Solar Beam assertion in `available_moves_from_request`
  kills the parse task and loses on the timer (`TECH_DEBT_BACKLOG.md:82`); undiagnosed.
- **A-F5 — stale mandate**: root CLAUDE.md calls `obs_build_benchmark.py` MANDATORY for any obs change, but it
  measures the Python encoder, which training no longer runs; the Rust encoder has no equivalent mandated
  benchmark. `deletion_pass_manifest.md:47`'s 18,045 lines for `poke_env/` vs 15,563 non-test lines now (the old
  figure likely included the fork's tests).
- **A-F6 — M7's four Rust-only readiness items are unbuilt** and absent from the backlog beyond T9's single row.
- **A-F1 / A-F3 — P1 (2026-10-07):** A-F1 FIXED (the trainer, h2h and plateau load 0 poke-env modules at import AND run time); A-F3 FIXED as to ownership (`schema.rs` is frozen Rust-owned source; the Python classifier is held equal to it by `rust_core_schema_test.py` until P6).
- **A-F7 — not verified**: whether `eval_roster.EvalRLPlayer` is still INSTANTIATED by the Rust eval path or only
  imported; whether `main.h2h` / `main.plateau` load poke-env indirectly; the feasibility of A4.4's in-process
  anchor slot.

**(B) hot path**

- **B-F1 — the performance work list has no backlog row**: CUDA-graph R1, bf16, T2 fusion and "a torch upgrade"
  live in the ledger / profile and a memory note only; "a torch upgrade beyond 2.8" is undefined in the repo.
- **B-F2 — the X5 speed benchmark ran in a different regime from the arms** (a seeded 20-snapshot pool vs a pool
  growing 0 → 5). Steady-state cycles inside the arms read +14.3 %, which by the benchmark README's own threshold
  (s ≤ 15.4 %) would floor the matched-wall checkpoint to **13M, not 12M**; end to end (startup included) reads
  +16.9 %, consistent with 12M. The amendment should state which wall definition it uses and the regime gap.
- **B-F3 — 15M screens never reach the X26 steady state**: `eval/pool_snapshot_count` is 5 at 14M (blob s1002,
  fm s1002), so screens see ~8 % T2 share, not the profile's 16 %. Screen throughput / cost does not transfer to
  the deep run without that correction.
- **B-F4 — a one-off collect spike near 6.0M in both s1002 runs** (25.7 s and 29.2 s vs ~9 s); cause UNVERIFIED
  (perhaps a promotion slot load).
- **B-F5 — the Rust release build is untuned** (no `[profile.release]` overrides); cheap to test, < 1 % of wall.
- **B-F6 — missing instruments**: no in-core timing split (so the cost of the parse-through-text observation is
  unknown), no in-run startup breakdown (T2 startup only in benchmark logs), and `eval/duration_sec` still reads
  like wall time (≈ 330 s vs ≈ 19 s wall; F-SZ-5, not renamed).
- **B-F7 — profiler access**: `ptrace_scope` read 1 during the profile (F-BP-7; memory says it stays 0) and GPU
  counters need admin, so host-side attribution is indirect.
- **B-F8 — contaminated data excluded**: today's live seed's core time (10.03 ms / step) is inflated by box load;
  not cited as a measurement.
