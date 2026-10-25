# pokesim — Rust Pokémon Showdown sim (Gen 3 OU singles)

A from-scratch, **bit-for-bit-faithful** Rust reimplementation of the Pokémon Showdown battle simulator: the
same seed, teams and choices give byte-identical protocol output to upstream Showdown. It is the RUST CORE
(typed event layer, one-sided reading, trackers, encoder) and the only training / eval transport
(`sim_bridge`); the Node bridge survives as the reference arm of the differential harnesses. std-only, zero
dependencies.

This README is the human entry point: how to build, run the fuzzer, and regenerate the golden vectors. Each
other fact has one home:

| For | Read |
|---|---|
| the rules, the bit-for-bit contract, build hazards (stale-build refusal, target dirs, data dir) | `CLAUDE.md` (this directory) |
| the module map (what each file is and its state) | `designs/rust_sim/module_map.md` |
| the gate ladder (PRNG, dex, team, stats, state, turn, E2E) and every golden generator | `designs/rust_sim/differential_gates.md`, `designs/rust_sim/e2e_capstone.md` |
| the move census (always recounted: `scan_move_probe`, `harness/scan_move_coverage.js`) | `designs/rust_sim/move_census.md` |
| the four A/B fuzzers, their modes and allowlists; closed findings | `designs/rust_sim/fuzzers.md`, `designs/rust_sim/ab_fuzzer_findings.md` |
| regression pins, edge cases, the protocol design | `designs/rust_sim/regression_pins.md`, `EDGE_CASES.md`, `PROTOCOL_EMISSION_DESIGN.md` (this directory) |
| the closed port rounds and the status blockquote that used to open this file | `designs/research_state/claude_md_archive/rust_sim_port_build_log.md`, `rust_sim_readme_history.md` |

Layout: `src/` (the crate; `src/bin/` holds `sim_bridge`, `ab_replay`, `bridge_replay`, `search_driver`,
`scan_move_probe`, `live_reader`, `core_events`), `tests/` (`*_test.rs` plus `tests/vectors/`, the committed
goldens), `harness/` (the Node and Python generators and fuzzers that talk to the real Showdown in
`deps/pokemon-showdown`).

## Build & test

Needs a Rust toolchain (`rustup`, stable). From a worktree build into your own `CARGO_TARGET_DIR`, never
main's `target/` (`src/rust_sim/CLAUDE.md` § Builds):

```bash
cd src/rust_sim
cargo test            # unit smokes + every differential golden test
```

The Node generators below need the `deps/pokemon-showdown` submodule's `dist/` and `node_modules` symlinks
(`./scripts/bootstrap.sh` makes them; manual steps in `docs/DEVELOPING.md`). A Python generator in a linked
worktree first needs `export PYTHONPATH=$PYTHONPATH:src`.

## A/B differential fuzzer — the overnight parity hunter (runbook)

`harness/ab_fuzz.js` runs for **hours unattended**, driving the REAL Showdown sim and this port side by side
over generated teams and saving a minimized, standalone-replayable repro for each divergence. It reuses the
e2e capstone's recorder and modeled-universe predicates (required from `gen_e2e_fuzz.js`), and the Rust side
replays chunks through `src/bin/ab_replay.rs` (per-battle JSON verdicts, never panicking on a divergence).
No Showdown server is needed (in-process `BattleStream`). The driver builds the EMISSION SELF-CHECK
`ab_replay` once at startup (`cargo build --profile selfcheck --features emission-selfcheck --bin ab_replay`
into `target/selfcheck/`; a failed check is a `panic` verdict, `designs/rust_sim/emission_selfcheck.md`).
The authoritative usage header is the top of `harness/ab_fuzz.js`.

### Launch overnight

```bash
cd src/rust_sim
# Showdown's OWN gen3 randbats generator (default mode), 12 hours:
nohup node harness/ab_fuzz.js --mode randbats --hours 12 > /dev/null 2>&1 &
# The modeled-universe generator (widest coverage of the modeled surface):
nohup node harness/ab_fuzz.js --mode random --hours 12 > /dev/null 2>&1 &
# A fixed battle count over the e2e capstone's filter-clean real teams:
node harness/ab_fuzz.js --mode pool --battles 500
# Smogon-derived RANDOM gen3ou teams (the surface training + ladder actually use):
node harness/ab_fuzz.js --mode ourandom --battles 500
# The OMNISCIENT BYTE differential (the emission-form hunter behind byte_fuzz_corpus/):
node harness/ab_fuzz.js --mode pool --protocol --format gen3ou --battles 200
```

Flags: `--mode randbats|random|pool|ourandom|ladder` (default randbats; what each hunts:
`designs/rust_sim/fuzzers.md`) · `--battles N` / `--hours H` (default: run until killed) · `--master-seed S`
(default from time, ALWAYS printed, so any run is reproducible) · `--chunk N` (default 25) · `--out DIR`
(default `harness/ab_fuzz_out/`, gitignored: never commit run output) · `--keep-chunks` (default: clean chunk
files are deleted after replay; divergent battles are always saved standalone) · `--protocol` (the
OMNISCIENT BYTE differential: adds the `protocol` divergence kind and the known-residual allowlist GREEN GATE,
which exits non-zero on any non-allowlisted divergence) · `--format gen3customgame|gen3ou` (default
gen3customgame) · `--ladder-tier` (with `--mode ladder`). `POKESIM_AB_REPLAY_BIN` points the driver at an
isolated `ab_replay` build so a byte run never touches the shared `target/`.

### Where results land / how to read them

- `<out>/ab_fuzz.log`: ONE stats line per chunk (append-only): cumulative battles / ok / diverged / panic, a
  `kinds=` first-divergence taxonomy, `bph` (battles per hour) and, in randbats mode, `adj_rate` (sets touched
  by the item/ability adapter over sets fielded).
- `<out>/divergences/<runid>_<battleid>/`: one SELF-CONTAINED repro per divergence: `battle.txt` (a
  single-battle chunk in the e2e TAB format, replayable standalone forever, independent of generator drift)
  and `summary.json` (mode, master seed, init/choose/battle seeds, packed teams, choice tokens, and the first
  divergence: kind, decision index, expected-vs-got, detail). Allowlisted residuals land in `allowlisted/`.
- `<out>/summary_<runid>.json`: the end-of-run cumulative summary (tallies, divergence kinds, coverage counts,
  randbats adjustment stats, repro dirs).

### Replaying a repro

```bash
./target/selfcheck/ab_replay harness/ab_fuzz_out/divergences/<runid>_<battleid>/
# → {"battle":"ab_3_7","verdict":"diverged","kind":"seed","decision":12,
#    "expected":"…","got":"…","detail":"…"}
```

The replayer accepts a repro dir (reads its `battle.txt`) or any chunk file. After an engine fix the same
command must print `"verdict":"ok"`: that flip is the fix's acceptance test. Root-cause with
`harness/probe_repro_simtrace.js`, under the repro's own `FMT` (a gen3ou repro replayed as `gen3customgame`
is a different battle).

### Repro → regression pin (the project's edge-case → pin law)

Every REAL engine bug a repro isolates gets a NAMED, DETERMINISTIC pin in `tests/regression_test.rs` when it
is fixed: reduce the repro (its `summary.json` carries the exact packed teams, init seed and choice tokens;
`battle.txt`'s INIT/DEC lines are the script), reconstruct the minimal scenario as a constructed-team test
(mirror the existing pins), assert the exact post-turn STATE **and the post-turn SEED** for draw-order bugs,
and verify the pin FAILS with the fix reverted. The repro dir stays as the provenance record; the pin is the
permanent guard (`designs/rust_sim/regression_pins.md`).

### SIGINT / reproducibility

Ctrl-C (SIGINT) finishes the current chunk, prints and writes the cumulative summary, and exits 0 (a second
SIGINT aborts immediately), so an overnight run can always be stopped cleanly. Stop it by its PID. The same
mode + master seed + code replays the same battles; repro dirs additionally replay after the generator changes.

## Regenerating the golden vectors

Each golden under `tests/vectors/` is captured from the REAL Showdown (omniscient `BattleStream`, no server)
and pinned by one `cargo test` file; the full ladder with what each rung proves is
`designs/rust_sim/differential_gates.md`. Regenerate after a change to what the golden covers, then run
`cargo test`. A regenerated golden is a deliberate re-record: review the diff against the Rust gate. The E2E
capstone's golden regenerates from its pinned team list (`designs/rust_sim/e2e_capstone.md`).

### PRNG (`tests/vectors/prng_golden.txt`)

```bash
node src/rust_sim/harness/gen_prng_vectors.js
```

It first cross-checks the dependency-free JS reference against the real `prng.js` and **aborts if they ever
differ**, so a regenerated `prng_golden.txt` is a faithful capture. `tests/prng_golden.rs` re-pins the Rust.

### Dex (`tests/vectors/dex_golden.txt`)

Dumped from the `agents.gen3_data` facade (the Python runtime's source of truth); regenerate after any data
or category-derivation change. `tests/dex_test.rs` then asserts the Rust dex reproduces every line.

```bash
/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 src/rust_sim/harness/gen_dex_golden.py
```

### Team (`tests/vectors/team_golden.txt`)

Captured from the real `Teams.pack` / `unpack`; regenerate after any team-codec change. It aborts if
`Teams.pack` is not idempotent on its own output, and emits `(IN, UNPACK, PACK)` triples (Showdown-canonical,
poke-env lowercase, raw edge-case fixtures) that `tests/team_test.rs` pins the codec against.

```bash
node src/rust_sim/harness/gen_team_golden.js
```

### Switch-in (`tests/vectors/switchin_golden.txt`)

A real battle's POST-switch-in state (lead Atk stage + `field.weather` for the Intimidate / Sand Stream /
Drizzle / Drought scenarios), pinned by `tests/switchin_test.rs`. Regenerate after any change to the switch-in
event dispatch.

```bash
node src/rust_sim/harness/gen_switchin_golden.js
```

### Turn, multi-turn and full-battle (`turn_golden.txt`, `battle_golden.txt`, `fullbattle_golden.txt`)

Each records, per decision boundary, the post-state and the PRNG seed before and after, so the Rust must
consume the PRNG in Showdown's exact order and count. Regenerate after any change to `turn.rs` (the single
turn, the multi-turn cycle and residuals, or the switch phase and draw order).

```bash
node src/rust_sim/harness/gen_turn_golden.js        # tests/turn_test.rs
node src/rust_sim/harness/gen_battle_golden.js      # tests/battle_test.rs
node src/rust_sim/harness/gen_fullbattle_golden.js  # tests/fullbattle_test.rs
```

`gen_turn_golden.js` guards that each "distinct-speed" scenario's actives do not silently tie on action speed.
`harness/trace_turn_rng.js`, `trace_multiturn_rng.js` and `trace_switch_rng.js` print the live sim's per-draw
order for an investigation; they are not generators.

### Protocol capture (`tests/vectors/protocol_capture_golden.txt`)

The RAW OMNISCIENT `|...|` stream Showdown emits per battle, the byte set `ProtocolBuilder` reproduces (the
`|t:|` wall-clock line is normalized to `|t:|<NORMALIZED>`). Regenerate after any change to the scenarios or
the sim's line ordering. `tests/protocol_test.rs` replays each battle through `run_full_battle_logged` and
asserts byte-equality per line, in order, with a first-divergence panic. Emission is observation-only: it
draws no PRNG, so the seed suites must stay green with identical assertion counts. The line grammar is
`tests/vectors/protocol_inventory.md`; the design is `PROTOCOL_EMISSION_DESIGN.md`.

```bash
node src/rust_sim/harness/gen_protocol_capture.js
```

### Item-modifier class sweep (`tests/vectors/item_mods_golden.txt`)

Covers every wired item-class member (type-boost, species-stat, Choice Band, each with matching and
wrong-type / wrong-species controls) as full battles to game-end, per-decision STATE+HP+SEED. Replayed by
`tests/item_mods_test.rs`, which enforces a minimum number of boosted-hit rows per member. The exact fold math
is additionally pinned by the damage golden (`node src/rust_sim/harness/gen_damage_golden.js` →
`tests/damage_test.rs`).

```bash
node src/rust_sim/harness/gen_item_mods_golden.js
```

### The gen3 mechanics inventory and the item / ability mechanics data

The data-driven mechanics framework's class map (`tests/vectors/gen3_mechanics_inventory.md`) and the
committed item / ability mechanics fields derive from the RESOLVED `Dex.mod('gen3')`, never a raw data file
(the mod-chain law, `CLAUDE.md`). After a Showdown submodule bump or an extractor change:

```bash
node src/rust_sim/harness/dump_gen3_mechanics.js            # rewrites the inventory .md
node src/rust_sim/harness/dump_gen3_mechanics.js --check    # DRIFT GATE: committed gen3_items / gen3_abilities
                                                            # json vs the resolved dist
/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 \
    tools/pokemon_data_extractor/sync.py --datasets items abilities
```

Run `--check` after every regeneration; the Python side is separately pinned by
`src/agents/gen3_data/extractor_parity_test.py`.

### The handler-completeness audit (manifest + gate)

The STATIC completeness guarantee for the at-site dispatch model (`designs/rust_sim/data_driven_mechanics.md`,
the handler-completeness audit): every resolved handler key on every REACHABLE effect has an explicit
disposition in `tests/vectors/gen3_handler_audit.json` (human census: `gen3_handler_audit.md`),
body-fingerprint-pinned against the resolved dist.

```bash
node src/rust_sim/harness/dump_gen3_handlers.js --audit     # THE GATE (also inside cargo test via
                                                            # tests/handler_audit_test.rs): fails on a NEW
                                                            # handler, a stale row, a body-fingerprint drift,
                                                            # or a dead `implemented` anchor
node src/rust_sim/harness/dump_gen3_handlers.js             # regenerate manifest + census (after triaging in
                                                            # harness/handler_audit_dispositions.js)
node src/rust_sim/harness/dump_gen3_handlers.js --enumerate # raw row dump (triage / debug)
```

Triage rule: a fingerprint drift or a new key means the dist's semantics moved. PROBE first (the sim is the
only oracle), then disposition in `handler_audit_dispositions.js` and regenerate. Never wave a gap through: a
real miss is a latent bug (see `EDGE_CASES.md`).
