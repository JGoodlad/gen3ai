# CLAUDE.md — `src/rust_sim/` (pokesim)

A from-scratch Rust reimplementation of the Pokémon Showdown battle simulator, scoped to **Gen 3 OU
singles**, whose hard requirement is **bit-for-bit identical** output to upstream Showdown given the
same seed + teams + choices. It is also the RUST CORE: the typed event layer, the one-sided reading,
the trackers and the encoder every training row comes from.

**`sim_bridge` (in-process, serverless) is the ONLY training/eval transport**; a typed `--use-bridge`
is refused at parse time (`designs/deleted_flags.md`). The Node bridge (`local_sim_bridge.js`)
survives only as the REFERENCE arm of the A/B harnesses and gates below; an outside websocket client
is served by the in-repo front end `python -m utils.bridge.ws_frontend` (`--impl rust`, the default,
spawns this crate's `sim_bridge` per battle — [`designs/rust_sim/ws_frontend.md`](../../designs/rust_sim/ws_frontend.md)).

**This leaf is the RULES, the COMMANDS, the MAP and the HAZARDS.** Every section's full text —
including what used to live here verbatim — is in its `designs/rust_sim/` topic doc (each holds a
"Moved from the leaf (2026-10-10)" section). The frozen pre-cleanup leaf:
`designs/research_state/claude_md_archive/src_rust_sim_CLAUDE_2026-10-10.md` (history, not current).

### 🚨 The move census is RECOUNTED, never quoted

Every dated count anywhere (the build log's ROUND entries, census snapshots) was true when written.
The ONLY current answer is the tool:

```bash
cd src/rust_sim && cargo build --release --bin scan_move_probe
PROBE_KIND=move|species|item|ability ./target/release/scan_move_probe < ids.txt   # the ENGINE is the oracle
cd src/rust_sim/harness && node scan_move_coverage.js                            # pool report + the 0-MISMODELED gate
```

- 🚨 **The universe count is the PROBE's to state, not the JS scan's** — `scan_move_coverage.js`
  mirrors modeled-move sets BY HAND from `turn.rs` and was three moves stale on 2026-09-08.
- 🚨 **"MODELED" means "does not fail loud" — not "gated", not "the draw count is right"**
  (`confuseray` never rolled accuracy for 13 rounds). Before touching a move, check whether any
  committed battle golden has ever EXECUTED it.
- **The invariant is the load-bearing claim: 0 MISMODELED** (an unmodeled move is a loud
  construction failure, never a silent desync). Re-run after admitting any move class, and update
  the copy in `src/utils/bridge/README.md` in the same pass.
- The engine-playable pool count (762 today) is NOT the 719-team TRAINING pool; different filters,
  both current — never "reconcile" them.
- Before admitting Swagger / Flatter or Attract read their findings first: as of ROUND 59 both
  were fail-loud, Attract blocked by the unmodelled construction-gender draw (which would move every
  committed golden's seed). Probe their current status; do not quote this line.

Detail, the dated measurements and the blocked-move findings: [`designs/rust_sim/move_census.md`](../../designs/rust_sim/move_census.md);
the ranked gap snapshot: [`designs/rust_sim/gen3_coverage_census_2026-09-08.md`](../../designs/rust_sim/gen3_coverage_census_2026-09-08.md).

## Why "bit-for-bit" is the hard part (read this first)

Reproducing Gen 3's *mechanics* is the easy half. The cost is matching Showdown's **control flow**:

1. **RNG-consumption-order equivalence.** All battle randomness funnels through one `PRNG`.
   Bit-identity requires consuming it in the *exact same order and count* — including the
   Fisher-Yates speed-tie shuffle, which itself draws. Mirror Showdown's `runEvent`/`singleEvent`
   dispatch order (order → priority → speed), not just the observable formula.
2. **Byte-identical protocol output.** The Rust side reader (`side_reader.rs`), the websocket front
   end and any outside client parse the `|...|` lines; they must match exactly. The one documented
   exception is `|t:|` wall-clock lines.

The answer to both is **differential testing against the real Showdown, layer by layer**.

## Module map

One line per module. **Read the row in [`designs/rust_sim/module_map.md`](../../designs/rust_sim/module_map.md) before you edit the module.**

| Module | In one line |
|---|---|
| `prng/` | Bit-for-bit port of `sim/prng.ts` (`SodiumRng` ChaCha20 default + `Gen5Rng`). |
| `json.rs` | std-only JSON reader for the dex; load-time only. |
| `dex/` | Static data over `data/pokemon/*.json` (the source `agents.gen3_data` reads) + the mechanic PARAMETERS the data-driven classes fold on. |
| `team.rs` | Bit-faithful `Teams.unpack`/`pack`. |
| `stats.rs` | Gen-3 in-battle stats (floor placement, integer nature math, Shedinja). |
| `state.rs` | `BattleState` and every modelled volatile / side condition, incl. `cached_speed`. |
| `event.rs` | Dispatch core + `speed_sort` with the Fisher-Yates speed-tie shuffle (the RNG crux). |
| `damage.rs` | Gen-3 single-hit damage from EXPLICIT inputs. |
| `turn.rs` (+ `src/turn/`) | The turn cycle in Showdown's exact draw order; `run_turn` / `run_battle` / `run_full_battle`. |
| `protocol.rs` | The append-only, PRNG-free `ProtocolBuilder`; all `\|...\|` formatting in ONE place. |
| `battle.rs` | `Battle::start*` + `BattleStream::write_line` (the streaming drop-in for `local_sim_bridge.js`). |
| `bridge.rs` | The per-side streams + `\|request\|` JSON + the HP-privacy fold. |
| `bin/sim_bridge.rs` | THE training/eval transport (incremental); opt-in `core_obs` ships `__OBS__` rows. |
| `side_reader.rs` + `bin/live_reader.rs` | THE one side reader (parse-built observation chain) — `sim_bridge`'s `core_obs`, live play (`main.live`), `core_events --obs-stream`. |
| `search.rs` + `bin/search_driver.rs` | Search + replay kernels; the drop-in for both Node offline drivers. |
| `driver_timing.rs` | Opt-in `expand_many` timing (`POKESIM_SEARCH_TIMING=1`); off = byte-identical. |
| `core_events/` (+ `bin/core_events.rs`) | Typed event layer; the binary is also the prober's battle reader. |
| `emission_check.rs` | The emission self-check (compiled OUT of `--release`). |
| `present/` | The TRUE one-sided reading: `present()`, `legal_actions()`, `check_view()`. |
| `version.rs` | `BattleVersion`: the persistent battle state; the `Engine` as referee. |
| `trackers/` | Per-decision trackers + the native window record. |
| `encoder/` | THE observation row (`OBS_DIM`), the wire frame, the oracle reveal. |
| `core_error.rs` | `CoreError` — never a bare `String` inside the core. |
| `engine.rs` | The ENGINE half of a bridge session; `BridgeSession` is the transport around it. |
| `bin/ab_replay.rs`, `bin/bridge_replay.rs` | The A/B fuzzers' replayers + their known-residual classifiers. |
| `bin/scan_move_probe.rs` | The census oracle (above). |

## The Rust core — rules that bind every change

Contracts and gate tables: [`core_events.md`](../../designs/rust_sim/core_events.md),
[`present.md`](../../designs/rust_sim/present.md), [`trackers.md`](../../designs/rust_sim/trackers.md),
[`encoder.md`](../../designs/rust_sim/encoder.md), [`emission_selfcheck.md`](../../designs/rust_sim/emission_selfcheck.md),
[`live_reader.md`](../../designs/rust_sim/live_reader.md).

- 🚨 **Every `ProtocolBuilder` method builds a typed `core_events::Line`; the text is its `render()`.**
  A new emit form is a typed method and must be CANONICAL: `Line::parse(render(l)) == l` (a `|`
  inside one field is two fields). `tests/core_events_test.rs` checks every corpus battle.
- 🚨 **A move of the OTHER side's mon run inside another action goes through
  `BattleState::in_nested_move_scope`** (Pursuit, Snatch) — else `parse != step` (core_events.md §3, §5).
- 🚨 **`core_events/schema.rs`, `present/tables.rs` and `encoder/layout.rs` are Rust-OWNED source** —
  their Python generators are deleted; edit them directly. `layout.rs` changes together with
  `agents/observation/constants.py`; `src/agents/observation/rust_core_obs_layout_test.py` parses it
  and fails on any shared value that differs.
- 🚨 **`present()` takes NO BOARD** — the view is built from one side's stream alone; the omniscient
  board is only a REFEREE (`check_view`). It is the TRUE reading: never add a rule whose only
  purpose is to reproduce a poke-env mistake. Every version folds `parse(render)`.
- 🚨 **A training session builds NO core recording unless asked** (`new_core`); START's opt-in
  `core_obs` key is the one exception, and absent it not one byte changes.
- 🚨 **The encoder reproduces, it never fixes** — a wrong tracker value is fixed in the tracker. **The
  gate is BYTES** (f64, one round to f32 at the write; a `-0.0` fails the golden). Test/fuzz builds
  NaN-prefill the row (release zero-fills).
- 🚨 **Emission self-check: every call site carries `#[cfg(any(debug_assertions, feature = "emission-selfcheck"))]`**
  (`tests/emission_check_test.rs` fails one without it). The self-check build lives in
  `target/selfcheck/`, NEVER `target/release/` (a live run execs the release binary). Python picks it
  with `POKESIM_EMISSION_SELFCHECK=1` (set by the root `conftest.py` and for fuzz scripts). A
  regression the check finds gets its own unit test that FAILS on revert.

```bash
cargo test                                                         # self-check ON (debug_assertions)
cargo build --profile selfcheck --features emission-selfcheck      # ON, optimized -> target/selfcheck/
cargo build --release                                              # OFF: every call compiled out
python3 -m pytest src/agents/battle/core_corpus_test.py -q         # the COMMIT corpus through core_events
python -m agents.training.golden_obs_core --check                  # the obs golden
```

🚨 **Never build into main's `target/` from a worktree** (a compile-time dex path leaks); a fuzzer
or investigation gets its own `CARGO_TARGET_DIR` — never the shared `target/`, which holds the live
`ab_replay`.

## The offline drivers (search + replay) and the clone-and-branch primitive

`src/bin/search_driver.rs` is the byte-compatible drop-in for BOTH Node offline drivers; **dispatch
is on the KEY** (`mode` ⇒ one-shot replay family, else the persistent `{id, cmd}` search loop).
`impl="node"` is still the DEFAULT in `utils/bridge/search_session.py`, `reconstruction.py` and the
prober's `better_line`; `impl="rust"` selects the binary. `BridgeSession::snapshot()` (a derived
`Clone`, PRNG continuity automatic) is the clone-and-branch primitive. The callable-surface table,
the snapshot API contract and every gate: [`search_and_replay_drivers.md`](../../designs/rust_sim/search_and_replay_drivers.md).

- 🚨 **Run each parity gate on at least TWO fresh seeds before calling it green** — one bug class
  appeared only on the seventh golden.
- 🚨 **Verify an allowlist claim against the harness's `ALLOWLIST` and the code, never prose** — a
  stale entry hid the path that killed two production launches (`gen3_locked_choice_never_rejected_v1`).
- 🚨 **`pre_state` volatile names are UNVERIFIED** (only `substitutebroken` is); a green parity run
  is not evidence for the rest.
- 🚨 **A gen3ou repro MUST be replayed with `{format:'gen3ou', allowHiddenPower:true}`** — the probe
  default is customgame.
- 🚨 **Do NOT feed `search::volatile_names` to the obs layer** — it includes conditions the sim never
  announces. The one-sided `view.rs` projection is DELETED ([`one_sided_view.md`](../../designs/rust_sim/one_sided_view.md)).
- `expand_many`'s optional `side` elides the half a search does not read (`gen3_expand_many_side_elision_v1`);
  Python's elided slot is a sentinel that RAISES on read, never an empty dict.
- ⚠️ `harness/search_impl_parity.py` has NO obtainable golden (its generators were deleted in P6).

## The differential-gate ladder

Each rung's construction and honest scope: [`differential_gates.md`](../../designs/rust_sim/differential_gates.md).

| rung | harness (regenerates the vectors) | `cargo test` gate |
|---|---|---|
| PRNG | `harness/gen_prng_vectors.js` (aborts on any mismatch vs the real `prng.js`) | `tests/prng_golden.rs` |
| Dex | `harness/gen_dex_golden.py` | `tests/dex_test.rs` |
| Team | `harness/gen_team_golden.js` + hand-crafted edge fixtures (keep them: they caught four real bugs) | `tests/team_test.rs` |
| Stats | `harness/gen_stats_golden.js` | `tests/stats_test.rs` |
| State | `harness/gen_state_golden.js` | `tests/state_test.rs` |
| Turn | `harness/gen_turn_golden.js` (the draw-ORDER+COUNT proof) | `tests/turn_test.rs` |
| E2E capstone | `harness/gen_e2e_fuzz.js` | `tests/e2e_fuzz_test.rs` |

🚨 **Three draw-count subtleties no formula tells you** (each pinned): an IMMUNE move draws ONLY
accuracy; a first-mover KO cancels the second mover's move (draws nothing); the end-of-turn Quick
Claw draw is unconditional of possession but deferred by a faint.

🚨 **E2E capstone: STRICT `filtered_diverged == 0`, no escape hatch; the tallies are CLEAN-ONLY;
coverage is GATED by floors, not reported.** ⚠️ A plain regen of `gen_e2e_fuzz.js` no longer
reproduces the committed 220-battle golden (the pool grew) — a regen is a NEW golden to review, not
a check. Detail: [`e2e_capstone.md`](../../designs/rust_sim/e2e_capstone.md).

## Regression pins — THE LAW

**Every edge case / engine bug a fuzz surfaces becomes a NAMED deterministic pin in
`tests/regression_test.rs`** (or a `# regression:`-named golden scenario when the board is
irreducibly complex), **and the pin is only real once you have REVERTED the fix and watched it fail.**
Assert the sim's OBSERVABLE (the emitted line, the seed), not the port's own representation. After
any PRNG/draw-order change regenerate the ground truth (`node src/rust_sim/harness/probe_regression_rng.js`,
`probe_residual_order_rng.js`, `probe_phaze_regression_rng.js`) and update the constants. The
bug → pin map: [`regression_pins.md`](../../designs/rust_sim/regression_pins.md).

## The four A/B fuzzers (continuous parity hunters)

Runbooks, verdict taxonomies, the known-residual allowlists clause by clause, team modes and the
picker's blind spots: [`designs/rust_sim/fuzzers.md`](../../designs/rust_sim/fuzzers.md). Closed
findings: [`ab_fuzzer_findings.md`](../../designs/rust_sim/ab_fuzzer_findings.md).

| fuzzer | compares | green gate / self-test |
|---|---|---|
| `harness/ab_fuzz.js` (+ `ab_replay`) | STATE + seed + winner on the omniscient stream | repro → `ab_replay <dir>` flips to `ok` → pin it |
| `ab_fuzz.js --protocol` | the omniscient `\|...\|` BYTES | E1 / A1 allowlist (`classify_known_residual`); `tests/byte_fuzz_corpus_test.rs` |
| `harness/bridge_ab_fuzz.js` (+ `bridge_replay --ab`) | per-side streams + `\|request\|` JSON | `node harness/bridge_ab_fuzz.js --selftest` after ANY per-side allowlist change |
| `harness/gen_sim_bridge_diff.js` | both real bridges, boundaries discovered LIVE — the strongest gate | `--selftest` after any `ALLOWLIST_TRANSFORMS` / probe change; `drain_timeouts` must stay 0; `--persistent` for a soak |

Team modes (`--mode`): `randbats`, `random`, `pool`, `ourandom` (gen3ou-native, Smogon-derived —
the surface we care about) and `ladder` (22,813 public-ladder teams).

- 🚨 **Allowlist clauses are the live definition of when a gate may pass — edit one only with an
  injection proof; the NEGATIVE cases are the load-bearing half.** Every escape is a named reason
  backed by a tagged repro fixture.
- 🚨 **B1 and the per-side mirror key do NOT check the rest of the battle** (first-divergence
  verdict); retrofitting the block-swap key's clause (3) onto them is OPEN.
- 🚨 **A sloppy seed anchor makes the omniscient gate VACUOUS** (`align_seed_subsequence`;
  `POKESIM_DUMP_SEEDS=1` checks its superset precondition on any repro).
- 🚨 **The layering principle:** per-side + request = the CONTRACT; the omniscient log = a LOCALIZER;
  the seed = a LEADING INDICATOR. An outer-layer mismatch is always a bug; an inner-layer one with a
  clean outer layer is not necessarily one.
- 🚨 **No fuzzer submits an imprisoned pick** (the pickers mirror the hidden disable) — that path is
  covered by pins only. **The recorder never re-writes a HELD side**; a repro recorded before that
  fix cannot flip to `ok`.
- 🚨 **`ab_fuzz_out*` run dirs are gitignored — never commit run output.**
- Root-cause a repro with `harness/probe_repro_simtrace.js` (per-draw call sites, under the repro's
  own format). Probe-settled specs: read the probe's SETTLED header before implementing; do not
  re-derive from source.

## Data-driven mechanics and the handler audit

Items / abilities are implemented as mechanic CLASSES with parameters extracted from the RESOLVED
`Dex.mod('gen3')` — **read the class before adding a member** ([`data_driven_mechanics.md`](../../designs/rust_sim/data_driven_mechanics.md)).

- 🚨 **THE MOD-CHAIN LAW: never regex a single data file for a mechanic** — later mods replace and
  delete handlers (gen3 Light Ball is SpA-only ×2). The resolved dist and the probe are the oracle.
- `node src/rust_sim/harness/dump_gen3_mechanics.js --check` — the drift gate on the committed
  `data/pokemon/gen3_items.json` / `gen3_abilities.json`; run it whenever either regenerates.
- **The handler-completeness audit** (`gen3_handler_audit_v1`, `tests/handler_audit_test.rs`) FAILS
  on a new handler, a stale row, a body-fingerprint drift or a dead anchor, and **fails loudly when
  node/dist are missing**. Admitting a deferred effect to a MODELED set pulls its handlers in.

```bash
node src/rust_sim/harness/dump_gen3_handlers.js          # regenerate
node src/rust_sim/harness/dump_gen3_handlers.js --audit   # the gate
```

## Protocol emission

Emission is a **side output** of events that already happened: it draws NO PRNG and mutates no
asserted state (the whole seed suite stays byte-identical). Gates: `tests/protocol_test.rs` (capture
golden, byte-equal per line), `tests/writeline_test.rs` (per-write chunks vs the real Node
`BattleStream`), `tests/turn_limit_test.rs` (the 1000-turn tie). Line inventory and formatters:
[`protocol_emission.md`](../../designs/rust_sim/protocol_emission.md), `PROTOCOL_EMISSION_DESIGN.md`,
`tests/vectors/protocol_inventory.md`.

- 🚨 **A `MonRef`'s IDENT is the on-field NICKNAME, never the species** (`turn.rs::display_name`); the
  species lives only in `|switch|`/`|drag|` DETAILS. Rendering the species crashed the reader on a
  nicknamed team (`gen3_nickname_ident_v1`).

## Standing lessons — each cost a coverage round, each binds the next change

(Full text with evidence: [`port_build_log.md`](../../designs/rust_sim/port_build_log.md) § Moved from the leaf.)

- A new mechanic can FALSIFY an old "no draw here" proof — re-read every "can never tie" argument
  when a class gains a second member.
- A determinism-oriented suite UNDER-TESTS the nondeterministic default: at least one gate must run
  with the reproducibility knob OFF and assert a distributional property.
- A gate that exempts a file by BASENAME exempts every file with that name — compare the relative path.
- A picker predicate that gates a test silently SHRINKS that test.
- A diagnostic that names an innocent bystander is worse than one that names nothing.
- An allowlist entry (or a "STILL NEEDED" note) can outlive its own fix — verify against the code.
- A count comparison whose two sides cover different windows is not evidence — compare draw
  POSITIONS within one decision.
- A pin that reads the port's own representation certifies its bugs — assert the sim's observable,
  and check a probe's script actually runs the row its header claims.
- Fix bugs found on the SURFACE YOU CARE ABOUT (`--mode ourandom` / `ladder`).
- A gate nobody can start is indistinguishable from a gate that passes.

## Conventions

- **std-only, zero dependencies.** Determinism + a no-network `cargo test` are the point; never a
  dep in the deterministic battle path.
- **Generation-generic.** Put Gen-3 constants behind the generation parameter (`Dex::for_gen(gen)`),
  mirroring Showdown's mod-delta layering.
- **Data source of truth** is `data/pokemon/*.json`, read the way `agents.gen3_data` reads it.

## Where the rest of the detail lives

| I am about to… | Read |
|---|---|
| edit a module | [`module_map.md`](../../designs/rust_sim/module_map.md) |
| count / admit a move | [`move_census.md`](../../designs/rust_sim/move_census.md) |
| change a rung of the gate ladder | [`differential_gates.md`](../../designs/rust_sim/differential_gates.md) |
| touch `search_driver` / replay / clone-branch | [`search_and_replay_drivers.md`](../../designs/rust_sim/search_and_replay_drivers.md) |
| regenerate or widen the e2e capstone | [`e2e_capstone.md`](../../designs/rust_sim/e2e_capstone.md) |
| add or read a regression pin | [`regression_pins.md`](../../designs/rust_sim/regression_pins.md) |
| run a fuzzer / edit an allowlist clause | [`fuzzers.md`](../../designs/rust_sim/fuzzers.md) |
| triage a fuzz divergence, or ask whether a class was closed | [`ab_fuzzer_findings.md`](../../designs/rust_sim/ab_fuzzer_findings.md) |
| add an item/ability to a mechanic class | [`data_driven_mechanics.md`](../../designs/rust_sim/data_driven_mechanics.md) |
| add a protocol line | [`protocol_emission.md`](../../designs/rust_sim/protocol_emission.md) |
| touch the event layer / reading / trackers / encoder | `core_events.md` · `present.md` · `trackers.md` · `encoder.md` (all in `designs/rust_sim/`) |
| touch the side reader / live play | [`live_reader.md`](../../designs/rust_sim/live_reader.md) |
| touch the websocket front end | [`ws_frontend.md`](../../designs/rust_sim/ws_frontend.md) |
| debug ONE gen-3 mechanic | [`port_build_log.md`](../../designs/rust_sim/port_build_log.md) — the closed round that modelled it |
