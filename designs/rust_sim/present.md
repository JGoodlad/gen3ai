# The Rust core's VERSION, READING and LEGALITY — `BattleVersion`, `present()`, `legal_actions()` (M2)

<!-- ALWAYS-CURRENT (this tree OWNS what it holds; see the root CLAUDE.md). State the truth, never
narrate a change. The code is `src/rust_sim/src/version.rs` + `src/rust_sim/src/present/`; the
program is `designs/endstate/program_rust_core.md` §2 M2. -->

The Rust Core Program's M2 (`gen3_core_version_v1`, `gen3_core_present_v1`, `gen3_core_search_v1`).
Built alongside the Python path: **training reads none of it** (§6 of the program — the cutover is
M6); SEARCH adopts it (`materializer=core`, §4 below).

| | |
|---|---|
| **Version** | `src/rust_sim/src/version.rs` — `BattleVersion`, `SideStream`, `Origin`, `parse_matches_step`, `streams_equal`; its engine is `src/rust_sim/src/engine.rs` |
| **Reading** | `src/rust_sim/src/present/` — `BoardReading` (poke-env's `Battle` + `Pokemon` for one side), `present()`, `legal_actions()` / `mask()`, `check_view()` (the board audit), `tables.rs` (GENERATED from poke-env) |
| **Python transport** | `src/agents/battle/core_view.py` (a `LiveView` / `LegalActions` from the core's JSON — no rule applied), `src/agents/training/core_successor.py` (the search successor) |
| **Gates** | `cargo test` (`present::tests` one pin per rule, `tests/version_test.rs`, `tests/engine_split_test.rs`, `tests/view_fold_opt_in_test.rs`); `src/agents/battle/rust_core_present_test.py` (`sim`: every rule against poke-env ITSELF); slice V (`rust_core_parity_views.py`, COMMIT + MILESTONE); the three search gates with `core` as a road (§5) |

---

## 1. The interface rule — the view is built from ONE SIDE'S STREAM, the board is a REFEREE

`present(reading: &BoardReading) -> OneSidedView` has **no board parameter**. A
`BoardReading` is built only from one side's lines (its protocol + its own `|request|`s, each
`Line::parse` of the shipped text), so a board-derived fact cannot reach the view by construction —
a training-only leak does not compile. The omniscient board is used in exactly two places, both
outside the view:

* **the audit** — `check_view(view, board, side, dex, pp_synced) -> Audit` asserts every SIM-FACT
  field of a view against the engine (`BattleVersion::audit`); slice V runs it at every decision;
* **the step path's owner truth** — on a CORE session (source recording on: the parity replay) each
  shipped line carries its source record's action SCOPE (`BridgeSession::side_scopes`), which the
  native window record keeps and the parse-reproduces-step gate holds the line-order owner to. It
  carries no LINE: every version folds `parse(render)`, the one observation path (program §6c) —
  the typed-at-source shortcut and its integrity mode are DELETED (program §4 M4 row).

## 2. `BattleVersion`

A battle is a chain of immutable versions, one per decision boundary: `parent: Option<Arc<…>>`
(history; a fork is an `Arc` clone), per side a `SideStream` (the `BoardReading` + M1's event `Reader`),
the transition's per-side `CoreEvent`s, a memoized per-side view (`OnceLock`), its `Origin`, and — on
a step-built version only — the ENGINE (`engine::Engine`, the referee and the stepper).

**The engine is not the transport** (`gen3_core_engine_split_v1`, the pre-M3 hand-off). A
`BridgeSession` is an `Engine` (the `Battle`, the `FullBattleDriver` turn loop, the open boundary, the
requests as TYPED values — `engine::Request`) wrapped in a TRANSPORT (the per-side chunks with their
reframe + HP-privacy fold, the `|request|` JSON as shipped, the command queue, the committed `script`
and the seed anchors). The engine advances over a caller-owned command queue and reports every
emission to an `EngineSink`; the transport is that sink for `sim_bridge` (byte-identical — the
41-battle transcript gate). A version owns the ENGINE only: a fork clones it and wraps the clone in a
FRESH transport (`BridgeSession::resume` — no chunks, no script, no seed anchors; the outstanding
requests' issued bytes SHARED with the engine, which rendered each once at issue) for as long as it is being driven.

| constructor | origin | what |
|---|---|---|
| `root(sess, names, teams, want)` / `root_with(…, trackers)` | `Step` | a fork tree's root: every line `sess` shipped is folded from its TEXT, then the transport is dropped; `want` names the sides that carry a stream (a search reads one) |
| `fork_session()` / `step(cmds, dex)` / `step_with(f)` / `child(sess)` | `Step` | a FORK: an engine clone in a fresh transport, driven, its lines folded from their TEXT; the child keeps the engine |
| `observe_root(&sess, …)` / `observe(&sess)` | `Observed` | a LINEAR replay OBSERVING a session the caller drives (the parity harness): no engine, no copy; the caller's session is the referee (`audit_on(side, board, dex)`) |
| `parse_root` / `parse_step` / `parse_advance` | `Parse` | ONE side's text, no engine — what a real server sends |

**The gate** — `parse_matches_step(step_built, parse_built, side)`: the whole board reading (the
`BoardReading`), the transition's events and the view equal, version by version. `core_events` runs it at
every step of every corpus battle (both sides), and `tests/version_test.rs` pins the properties one
battle can show: the chain equals its parse twin; a fork leaves its parent untouched; a fork chain
equals the observed linear replay; a fork's transport carries the engine and NO wire history, its
requests rendered to the parent wire's bytes; a NON-recording fork (a search tree's: no scopes, owners
by line order) folds the same stream as a recording one (with teeth); the audit
passes the truth and catches a tampered view; a parse-built version refuses to step or audit.
`tests/engine_split_test.rs` pins the split itself on the trapping golden and a disabled-move reject:
at every boundary the typed request renders the shipped bytes, and a resumed engine clone emits the
same chunks and commits as the full session it was cloned from.

**Errors are typed** (`gen3_core_error_v1`, `src/rust_sim/src/core_error.rs`). Every fallible core
call returns `CoreResult<T>`; a `CoreError` is one of three things, never an undifferentiated string:

| variant | means | example |
|---|---|---|
| `Refusal { exc: PyExc, msg }` | the input is one poke-env REFUSES, and the core refuses it with the SAME Python exception class (`KeyError`, `ValueError`, `IndexError`, `AssertionError`, `RuntimeError`, `NotImplementedError`, `UnknownMessageType`, `UnsupportedMessageType`) | `\|gen\|4` → `RuntimeError`; an unknown keyword → `UnknownMessageType`; `\|turn\|two` → `ValueError` |
| `Malformed(msg)` | input no protocol can carry (an undecodable request, a record that does not parse, a choice naming nothing, a client re-sending a refused choice) | `\|request\|{"side":` |
| `Fault(msg)` | the core contradicting itself (a gate disagreement, a conservation miss, a version asked for what its origin cannot do, an upstream engine desync) — a BUG | `parse_matches_step` failing |

The message is the pre-`CoreError` string byte for byte (`From<CoreError> for String` at the
transport boundary), so `sim_bridge`'s `__ERR__` frames are unchanged. The engine's fatal condition is
typed too (`Engine::fatal_error`). **The parity gate compares the CLASS**:
`rust_core_present_test.py::test_refusals_raise_the_same_class` feeds each refusing line to poke-env and
to the core (`core_events --present-stream` reports `core_error {kind, class, message}`) and requires
the same class; the Rust pins are `core_error::tests`.

## 3. `present()` — the TRUE reading; poke-env's mistakes are FINDINGS, never rules

🚨 **Parity with poke-env is not the goal; truth is** (owner directive, 2026-09-23). `present()`
applies every poke-env PRESENTATION rule (slot order, the sighting-count PP, the volatile lifecycle,
item / ability disclosure, the opponent's hidden spread, weather and screen turns, the legality
parse — V1–V9, V11–V15, V17, each NAMED, applied at one site, pinned by a Rust unit test AND by a
scenario replayed through poke-env itself, `rust_core_present_test.py`; the table is in
`src/rust_sim/src/present/mod.rs`). Where poke-env is WRONG about a sim fact the stream can
establish, `present()` carries the TRUTH and **never a rule whose only purpose is to reproduce the
mistake**; the disagreement is a FINDING in `agents/battle/poke_env_findings.py`: one field, a
VALUE-AWARE predicate, the reproduction, the truth's source, whether it reaches the obs. Every
comparison of the core against poke-env (slice V's core column, the present pins, the one-sided fuzz
gate's core road) routes a field difference through it — a difference no finding explains is a
divergence, and a finding that stops firing fails the pin that expects it. **The fork is not fixed
from M2**: fixing it moves the training-input boundary, the owner's call; when it lands, the entry is
deleted and the check tightens.

**The registry is EMPTY.** M2 registered three findings; all three were FIXED in the fork as
`gen3_pe_reading_fixes_v1` (2026-09-24, a TRAINING-INPUT change, `designs/CHANGELOG.md`) and their
entries deleted, so every comparison of the core against poke-env is now exact:

| was | field | upstream poke-env read | the truth (source) — what both read now | fork fix |
|---|---|---|---|---|
| **PE-V10** | `boosts` | a fainted mon kept its stages until switched out | none — the faint's `clearVolatile` zeroes them (`sim/pokemon.ts`; the Rust board) | `Pokemon.faint` clears them |
| **PE-R1b** | `status_counter` | +1 per `\|turn\|` while active — ONE AHEAD for a mon that entered after the residual, ONE BEHIND at a decision between the residual and the next `\|turn\|`; frozen at a faint | the toxic STAGE: residual `[from] psn` chips since the switch-in, cap 15 (`data/conditions.ts` `tox`; the Rust board's `Toxic(stage)`) | `Pokemon.note_residual_chip` from the `-damage` handler; `Battle.switch` resets it; `end_turn` no longer ticks |
| **PE-V16** | `volatiles` | a damaging Fire move of the holder's own ENDED Flash Fire (`Pokemon.moved`, any gen) | `flashfire` lasts until the holder leaves the field (`data/abilities.ts`; the Rust board, probe-verified) | the branch in `Pokemon.moved` removed |

All three reached the obs (the fainted active's stages in `active_context`, the per-mon toxic slot,
the `flashfire` volatile slot). Pins: `src/poke_env/battle/reading_fixes_test.py` (each FAILS on
upstream), `agents/training/poke_env_gaps/pe_reading_fixes_obs_integration_test.py` (three
constructed real-Showdown battles read at the obs), and this crate's `present::tests` +
`rust_core_present_test.py`, which now assert the two readings EQUAL at the truth. The view road
(`view.rs` / `view_adapter` / `event_fold`) reproduces poke-env and followed: V10 and its boost
ledger deleted, V5's toxic half on the residual chip.

**The CALLED-MOVE class — fixed in the fork AND mirrored here (`gen3_called_move_reading_v1`).**
gen3's `useMoveInner` announces a move another move called in the BARE form
`|move|<user>|<called>|<target>|[from] <Caller>` and `attrLastMove` may append `[still]` (which blanks
the target), `[miss]` (twice, seen) or `[notarget]` after it. Upstream poke-env knew only the
modern `[from]move: Metronome`, so a Metronome / Assist / Nature Power call RAISED `Unhandled move
message format` (a live ladder game lost on the timer), and a `[still]`-blanked one was silently
added to the actor's OWN moveset. The truth, from the sim: the called move is not the actor's
(Metronome draws from the dex, Assist from a teammate, Nature Power is always Swift in gen 3), is
not revealed, and costs no PP anywhere — gen 3 charges no Pressure for a sourced move (measured:
Metronome's PP is the same against a Pressure foe). `GEN3_BARE_MOVE_CALLERS` joins the random
callers to the Magic Coat / Mirror Move / Snatch branch, and `_canonical_from_tail` puts a
multi-flag tail in the order the single-pass strip consumes; `BoardReading::move_line` mirrors both
(`is_gen3_bare_move_caller`, `canonical_from_tail`). Every shape is one the real sim emitted
(`harness/probe_called_move_shapes.js`: 1,500 battles, 25,419 sourced `|move|` lines, 31 shapes);
pins: `poke_env/battle/called_move_reading_test.py` (23 fail on upstream), `present::called_move_tests`,
`rust_core_present_test.py`, and the node-bridge battle `agents/battle/called_move_bridge_integration_test.py`
(both players encode every decision; p2's reading of p1's caller PP equals p1's own `|request|`).
Not a training-input change: every line it changes used to raise (or, for the `[still]` shape, only
Metronome / Assist / Nature Power reach it — none is on a pool team or engine-playable).

**Not findings — INFORMATION LIMITS**, read as poke-env reads them because no client can know
better: V15 (an own mon the current request did not re-sync holds the sighting count of its PP,
which lags an un-announced Pressure deduction — the audit checks it can only lag) and V14 / R4 (a
Transformed own mon's copied ability; a gen-3 request states neither).

**A BENCHED badly-poisoned mon's counter — RESOLVED, both sides right** (was UNRESOLVED). The sim
keeps the stage the mon left with in `effectState.stage`; poke-env and the view read 0. The stored
value is a DEAD STORE: in the pinned Showdown its only reader is `tox.onResidual`, which runs for
ACTIVE mons only (`fieldEvent('Residual')`), and every entry runs `tox.onSwitchIn` (stage = 0)
before any residual (gen 3 inherits `tox` unchanged through gen 4's mod, whose `runSwitch` fires
`runEvent('SwitchIn')`); no other code reads `.stage`. So 0 is the benched mon's stage in every
respect that can act — the next chip after its re-entry is 1/16 — and it is what the obs should
hold (the damage op prices the next tick as `(stage + 1) / 16` on EVERY slot, benched included). It
is not an information limit (the chips are on the stream); the audit now CHECKS it
(`status_counter[tox-benched]`: a benched badly-poisoned mon reads 0).

The board audit (`check_view`) checks every SIM-FACT field against the engine at the truth — the
three fixed findings' fields and the benched toxic stage included (Flash Fire joined the audited
volatiles) — and names what it cannot check (V9, V14, V15) in `rules_fired`.

`legal_actions()` is `LegalActions.from_battle` over the raw `|request|` (the `BoardReading` keeps its
text) and `mask()` the 11-dim mask; slice V compares both to `Gen3ActionMasker` on every decision.

**The poke-env data the reading consults is GENERATED**: `tables.rs` from poke-env's pokedex, move
table, `Effect` lifecycle sets, `SideCondition`, type names and ignore set
(`python -m agents.battle.rust_core_present_tables --write`; `rust_core_present_tables_test.py`
fails the day it is stale).

## 4. Search on the core — `materializer=core`

`search_driver`'s `open_root` takes `core: "text"` (the one path; `"typed"` is REFUSED — the typed
shortcut is deleted, program §4 M4) and `side` (the one stream a tree folds); a core root is a
`BattleVersion`, and every `expand_many` arm's successor IS a version (`NodeState::Core`). Per arm
and wanted side it returns, with `rows` (what SEARCH sends, `gen3_core_encoder_v1`),
`core_pN = {mid, row, mask, tokens}` — the leaf's observation ENCODED on its version (the tree folds
the trackers: `open_root`'s `trackers`), its mask and its choice tokens (`present::choice_tokens`),
`row: null` where the side does not decide — and search wraps the row with `np.frombuffer`
(`designs/rust_sim/encoder.md` §5); no Python tracker, view or encoder touches the successor.
Without `rows` it returns the M2 payload `core_pN = {view, legal, request, events (the readings),
mid}`, which Python's `CoreSuccessorFactory` encodes (kept, unused by search, until the deletion
pass).

* **D10 — a leaf AT the intermediate decision.** An arm whose ply opens a second decision (a KO's
  replacement, a refused trapped switch) returns the version AT that decision (`mid`), built from
  the engine snapshot `resolve_turn_capturing(Capture { sessions: true })` takes there; the child
  node IS that leaf. So a forced switch is a NON-BRANCHABLE node on the core road, where the old
  roads expanded a D10 node from end-of-turn with the intermediate tokens — a semantic difference at
  depth ≥ 2, recorded as a finding (§6).
* **`recorded_exact` is REFUSED on core** (`resolve_turn_exact` has no production consumer on this
  path and no capture rule).
* **The INTEGRITY mode is DELETED** with the typed shortcut (its only job was typed == text):
  `expand_many` REFUSES an `integrity` key, and `--search-integrity` / `--core-path` are gone
  (`designs/deleted_flags.md`). Every battery row is stamped `materializer`, and `core_path: "text"`
  on the core road (so a row stays comparable to the rows written before the deletion).
* **Defaults**: `SearchConfig.materializer = "core"`; `--materializer {core,protocol,view}`
  (default `core`). The protocol and view roads are NOT deleted — the cutover's deletion manifest
  names them (program §4).

## 5. The gates, and what each proves

| gate | proves |
|---|---|
| slice V — COMMIT / MILESTONE (`rust_core_parity_views.py`, via `core_events --views`) | at every decision, both viewers: `present()` + `legal_actions()` + the mask == the `LiveView` / `LegalActions` training builds, type-strict, no allowlist but the registered poke-env FINDINGS (§3, value-aware, counted per decision — none registered today); the board audit; parse == step at every version |
| `materializer_parity_integration_test.py` | `protocol`, `view` and `core` give identical decisions, values and obs bytes |
| `fork_sharing_parity_integration_test.py` | the VIEW road: one root fork per DECISION, shared across the K worlds, gives the same successor obs bytes as one fork per world; the CORE road builds NO Python fork (`test_the_core_road_builds_no_python_fork`) |
| `one_sided_view_parity_fuzz_test.py` (`sim`) | on real bridge battles, the core's root and arm views == the protocol road's `LiveView`, and the CORE ROW road's encoded rows == the protocol road's successor rows, byte for byte |
| `core_successor_test.py` | the event transport; the driver REFUSES the deleted typed path and integrity mode |

## 6. Measurements and findings

Record: [`designs/research_state/measurements/rust_core_m2_2026-09-23/`](../research_state/measurements/rust_core_m2_2026-09-23/README.md)
(interleaved, one binary or road per process, the load stated with every table).

* **Slice V with the core column** (at `969c4e30` — the code of `9f77695d`, before its rebase onto main): MILESTONE 142,360 decisions, 81.7 M field
  comparisons, 26.2 M board-audit checks, **0 divergences**; COMMIT 1,838 decisions, 0. The
  findings' rates per 1,000 decisions then (pool · `production` policy · procedural): PE-V10 4.64 ·
  12.27 · 6.45; PE-R1b 2.69 · 0.58 · 6.04; PE-V16 0 · 0 · 0.
* **Slice V with the registry EMPTY** (at `c97358e8`, `gen3_pe_reading_fixes_v1`,
  [`../research_state/measurements/pe_reading_fixes_2026-09-24/`](../research_state/measurements/pe_reading_fixes_2026-09-24/README.md)):
  MILESTONE 142,410 decisions, 81.7 M field comparisons, 26.3 M board-audit checks (the benched
  toxic stage now among them), **0 divergences, 0 findings**; COMMIT 1,838 decisions, 0; procedural
  (seed 23, 29,783 decisions) the core column clean, findings `{}`.
* **The core road's cost**: 1.41× the view road's Rust `expand_many` per successor [1.36, 1.48], and
  the whole searched decision 0.926× [0.920, 0.944] at wide B.
* **The typed shortcut saves nothing measurable** (fold typed/text 1.019× [0.983, 1.316]; −0.1 % of
  the decision wall) — the program's §6 records the recommendation to delete it.
* **Findings** — the D10 leaf is the version AT the intermediate decision, so a forced switch is a
  non-branchable node on the core road where the old roads expanded it from end-of-turn with the
  intermediate tokens (a depth ≥ 2 difference, not a parity break: the gates' depth-1 decisions
  agree); `recorded_exact` is refused on core.
