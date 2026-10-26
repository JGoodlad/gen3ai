# CLAUDE.md — `src/rust_sim/` (pokesim)

A Rust port of the Pokémon Showdown simulator for Gen 3 singles (OU first; the full-gen3 coverage table is
`designs/rust_sim/gen3_full_parity.md`). Its hard requirement is
**bit-for-bit identical** output to upstream Showdown for the same seed, teams and choices. It is also
the RUST CORE: the typed event layer, the one-sided reading, the trackers and the encoder that every
training row comes from. `sim_bridge` (in-process) is the only training / eval transport; the Node bridge
(`src/utils/bridge/local_sim_bridge.js`) is the differential harnesses' reference arm. Detail: `designs/rust_sim/`.

## What bit-for-bit means

- **Match Showdown's control flow, not only its formulas.** Consume the one `PRNG` in Showdown's exact
  order and count, including the Fisher-Yates speed-tie shuffle (which itself draws): mirror the
  `runEvent` / `singleEvent` dispatch order (order → priority → speed). The draw-count subtleties no
  formula tells you are pinned and listed in `differential_gates.md`.
- **Protocol lines are byte-identical**, because the side reader and every outside client parse them;
  the one exception is the `|t:|` wall-clock line.
- **Model a condition's counters as the sim holds them, never a fused "turns left"** (`lockedmove`: `duration` AND
  `trueDuration`; `mustrecharge`: `duration: 2`): a cancelled action is where the two part ways. `--mode lockin`.

## The move census is RECOUNTED, never quoted

Never copy a move count: the tools state it (`scan_move_probe`, the engine as oracle; `harness/scan_move_coverage.js`,
the pool report and the 0-MISMODELED gate, whose universe verdict is a hand-mirrored copy of `turn.rs` and goes
stale). Commands: `move_census.md`.

- **The invariant is 0 MISMODELED**: an unmodeled move fails loud at construction, never desyncs silently.
  Re-run the scan after admitting any move class.
- **MODELED means "does not fail loud"**, not "gated" and not "draws right". Before touching a move, check
  whether a committed battle golden has ever executed it.
- The engine-playable pool and the TRAINING pool are different filters; never reconcile their counts.
- Before admitting a blocked move (`gen3_full_parity.md`'s gap table), read its findings in `move_census.md` and
  `designs/research_state/claude_md_archive/rust_sim_port_build_log.md`, and probe its current status.

## Rust core rules

Contracts: `core_events.md`, `present.md`, `trackers.md`, `encoder.md`, `emission_selfcheck.md`,
`live_reader.md`.

- Every `ProtocolBuilder` method builds a typed `core_events::Line`, and the text is its `render()`. A
  new emit form is a typed method and must be canonical: `Line::parse(render(l)) == l` (a `|` inside one
  field is two fields).
- A move of the OTHER side's mon run inside another action (Pursuit, Snatch) goes through
  `BattleState::in_nested_move_scope`, or `parse != step`.
- `core_events/schema.rs`, `present/tables.rs` and `encoder/layout.rs` are Rust-owned source with no
  generator: edit them directly. `layout.rs` changes together with `agents/observation/constants.py`
  (`src/agents/observation/rust_core_obs_layout_test.py` fails on any shared value that differs).
- `present()` takes no board: the view is built from one side's stream alone, and the omniscient board
  is only a referee (`check_view`).
- A training session builds no core recording unless asked (`new_core`). START's opt-in `core_obs` key
  is the one exception, and without it not one byte changes.
- The encoder reproduces, it never fixes: fix a wrong value in its tracker. Its gate is bytes (f64,
  one round to f32 at the write; a `-0.0` fails the golden).
- Emission is a side output: it draws no PRNG and mutates no asserted state.
- A `MonRef`'s ident is the on-field NICKNAME (`display_name` in `src/turn/helpers.rs`), never the
  species; the species appears only in the `|switch|` / `|drag|` details.
- std-only, zero dependencies (determinism, an offline `cargo test`). Keep it generation-generic (Gen-3
  constants behind `Dex::for_gen(gen)`), and read `data/pokemon/*.json` the way `agents.gen3_data` reads it.

## Builds

🚨 **Never build into main's `target/` from a worktree** (main would exec the worktree's code), and give a
fuzzer or an investigation its own `CARGO_TARGET_DIR`, never the shared `target/` (it holds binaries a live
run execs). The emission self-check build lives in `target/selfcheck/`, never `target/release/`; Python picks
it with `POKESIM_EMISSION_SELFCHECK=1` (the root `conftest.py` sets it). The data directory is resolved at RUN
time (`$POKESIM_DATA_DIR`, else the repo holding the binary, else the compile-time default;
`src/dex/data_dir.rs`): read data only through `dex::default_data_dir()`, never `env!("CARGO_MANIFEST_DIR")`.

🚨 **A STALE RUST BUILD IS REFUSED, never run.** `build.rs` embeds a hash of the sources a non-test build
compiles (`build_support.rs`); every binary prints it on `<bin> --build-stamp`, and Python recomputes it from
the checkout it runs in (a pinned run's own pin, never main's) and raises `StampMismatch` with the rebuild
command (`utils.rust_env.stamp`; the stamp's scope is in `module_map.md`). After a Rust commit, rebuild every
target you run: `target/` is per checkout, and MAIN's is rebuilt by `/gen3ai-ship` via
`scripts/ops/rebuild_main_rust.sh`. A `#[cfg(test)] mod x;` file is named `tests.rs` / `*_tests.rs` /
`*_test.rs` so an edit to it never stales a build (`stamp_test.py`). `POKESIM_ALLOW_STALE_BUILD=1` runs a stale
build on purpose and logs loudly; never use it to get past a refusal.

```bash
cargo build --profile selfcheck --features emission-selfcheck     # self-check ON, optimized -> target/selfcheck/
python3 -m pytest src/agents/battle/core_corpus_test.py -q        # the corpus through core_events
python -m agents.training.golden_obs_core --check                 # the obs golden
```

## Search and replay

`src/bin/search_driver.rs` is the only offline driver; dispatch is on the KEY (`mode` ⇒ the one-shot
replay family, else the persistent `{id, cmd}` search loop). `BridgeSession::snapshot()` is the
clone-and-branch primitive. Contract and gates: `search_and_replay_drivers.md`.

- Run each parity gate on at least two fresh seeds before calling it green (a bug class appeared only
  on the seventh golden).
- Verify an allowlist claim against the harness's `ALLOWLIST` and the code, never against prose.
- Replay a gen3ou repro with `{format:'gen3ou', allowHiddenPower:true}`; the probe default is customgame.
- Never feed `search::volatile_names` to the obs layer: it includes conditions the sim never announces.
- Every gen-3 damage roll (`random(16)`) goes through `BattleState::damage_roll`: the DAMAGE-ROLL
  CONTROL (`src/rolls.rs`) changes a roll's VALUE only when armed. A new damage site that draws directly
  breaks a matched world's replay. Contract: `search_and_replay_drivers.md` "The damage-roll control".
- The gender DRAW PAD (`M*` / `F*` / `N*` in the packed gender field) is ours and offline only.
  Showdown reads `M*` as unnamed, so a padded team never goes to Node or a live server.

## Gates, pins and fuzzers

The gate ladder (PRNG → dex → team → stats → state → turn → E2E capstone): `differential_gates.md`,
`e2e_capstone.md`. The A/B fuzzers, their team modes and blind spots: `fuzzers.md`; closed findings:
`ab_fuzzer_findings.md`.

- 🚨 **The E2E capstone is STRICT: `filtered_diverged == 0`, no escape hatch.** The golden regenerates
  byte for byte from its PINNED team list (`tests/vectors/e2e_fuzz_teams.sha256`;
  `src/rust_sim/e2e_fuzz_golden_test.py` fails on drift). A widened modeled universe or a re-pinned list is a
  deliberate RE-RECORD: review it against the Rust gate and write the provenance in `e2e_capstone.md`. The
  golden's last battle has a Quick Claw LEAD and the only 5th INIT field (the turn-0 `quickClawRoll`); keep it
  when re-recording.
- 🚨 **Every edge case or engine bug a fuzz surfaces becomes a NAMED deterministic pin** in
  `tests/regression_test.rs` (or a `# regression:` golden scenario), and the pin is real only once you have
  reverted the fix and watched it fail. Assert the sim's observable (the line, the seed), not the port's own
  representation. After a PRNG / draw-order change, regenerate the ground-truth probes (`regression_pins.md`).
- 🚨 **Allowlist clauses define when a fuzz gate may pass: edit one only with an injection proof** (the
  negative cases are the load-bearing half), then run that fuzzer's `--selftest`.
- Fix bugs on the surface we care about (`--mode ourandom` / `ladder`); root-cause a repro with
  `harness/probe_repro_simtrace.js`.

## Data-driven mechanics

Items and abilities are mechanic classes whose parameters come from the RESOLVED `Dex.mod('gen3')`; read the
class before adding a member (`data_driven_mechanics.md`).

🚨 **The mod-chain law: never regex a single data file for a mechanic.** Later mods replace and delete
handlers (gen3 Light Ball is SpA-only ×2); the resolved dist and the probe are the oracle. The drift gates:

```bash
node src/rust_sim/harness/dump_gen3_learnset.js --check    # gen3_learnset.json == Showdown's validator
node src/rust_sim/harness/dump_gen3_mechanics.js --check   # gen3_items / gen3_abilities; run whenever either regenerates
node src/rust_sim/harness/dump_gen3_handlers.js --audit    # handler completeness (no flag = regenerate)
```

## Where the detail is (`designs/rust_sim/`)

| I am about to… | Read |
|---|---|
| edit a module | `module_map.md` (its row first) |
| count or admit a move, item or format | `gen3_full_parity.md` (the current table), `move_census.md` |
| change a gate-ladder rung, or the e2e capstone | `differential_gates.md`, `e2e_capstone.md` |
| touch `search_driver`, replay or clone-and-branch | `search_and_replay_drivers.md` |
| add or read a regression pin | `regression_pins.md` |
| run a fuzzer, edit an allowlist, triage a divergence | `fuzzers.md`, `ab_fuzzer_findings.md` |
| add an item or ability to a mechanic class | `data_driven_mechanics.md` |
| add a protocol line | `protocol_emission.md` |
| touch the event layer, reading, trackers or encoder | `core_events.md`, `present.md`, `trackers.md`, `encoder.md` |
| touch the side reader / live play, or the websocket front end | `live_reader.md`, `ws_frontend.md` |
| debug one gen-3 mechanic, or read a standing lesson | `designs/research_state/claude_md_archive/rust_sim_port_build_log.md` |
