# The Rust core's VERSION, READING and LEGALITY — `BattleVersion`, `present()`, `legal_actions()` (M2)

<!-- ALWAYS-CURRENT (this tree OWNS what it holds; see the root CLAUDE.md). State the truth, never
narrate a change. The code is `src/rust_sim/src/version.rs` + `src/rust_sim/src/present/`; the
program is `designs/endstate/program_rust_core.md` §2 M2. -->

The Rust Core Program's M2 (`gen3_core_version_v1`, `gen3_core_present_v1`, `gen3_core_search_v1`).
Built alongside the Python path: **training reads none of it** (§6 of the program — the cutover is
M6); SEARCH adopts it (`materializer=core`, §4 below).

| | |
|---|---|
| **Version** | `src/rust_sim/src/version.rs` — `BattleVersion`, `SideStream`, `parse_matches_step`, `streams_equal` |
| **Reading** | `src/rust_sim/src/present/` — `BoardReading` (poke-env's `Battle` + `Pokemon` for one side), `present()`, `legal_actions()` / `mask()`, `check_view()` (the board audit), `tables.rs` (GENERATED from poke-env) |
| **Python transport** | `src/agents/battle/core_view.py` (a `LiveView` / `LegalActions` from the core's JSON — no rule applied), `src/agents/training/core_successor.py` (the search successor) |
| **Gates** | `cargo test` (`present::tests` one pin per rule, `tests/version_test.rs`, `tests/view_fold_opt_in_test.rs`); `src/agents/battle/rust_core_present_test.py` (`sim`: every rule against poke-env ITSELF); slice V (`rust_core_parity_views.py`, COMMIT + MILESTONE); the three search gates with `core` as a road (§5) |

---

## 1. The interface rule — the view is built from ONE SIDE'S STREAM, the board is a REFEREE

`present(reading: &BoardReading) -> OneSidedView` has **no board parameter**. A
`BoardReading` is built only from one side's typed lines (its protocol + its own `|request|`s), so a
board-derived fact cannot reach the view by construction — a training-only leak does not compile.
The omniscient board is used in exactly two places, both outside the view:

* **the audit** — `check_view(view, board, side, dex, pp_synced) -> Audit` asserts every SIM-FACT
  field of a view against the engine (`BattleVersion::audit`); slice V runs it at every decision;
* **search's typed shortcut** — the step path folds each shipped line from its TYPED source record
  (`BridgeSession::typed_side_lines`) instead of re-parsing the text. The typed `Line` is the same
  value `Line::parse` of its rendering gives (M1's canonical-form gate), and §4's integrity mode
  checks the two roads' VERSIONS equal on search's own arms.

## 2. `BattleVersion`

A battle is a chain of immutable versions, one per decision boundary: `parent: Option<Arc<…>>`
(history; a fork is an `Arc` clone), per side a `SideStream` (the `BoardReading` + M1's event `Reader`),
the transition's per-side `CoreEvent`s, a memoized per-side view (`OnceLock`), and the ENGINE (a
`BridgeSession`, the referee and the stepper; absent on a parse-built version).

| constructor | what |
|---|---|
| `root(engine, names, teams, compact, want)` | the step-path root; `want` names the sides that carry a stream (a search reads one), `compact` drops the engine's chunk history once folded |
| `step(cmds)` / `step_with(f)` / `child(engine)` | a FORK: the engine cloned, stepped, its new lines folded TYPED; the child is compacted |
| `advance_with(f)` | a LINEAR replay step (keeps the history, for the parse gate) |
| `root_text` / `child_text` | the same, folding the TEXT (render → `Line::parse`) — the integrity twin |
| `parse_root` / `parse_step` / `parse_advance` | ONE side's text, no engine — what a real server sends |

**The gate** — `parse_matches_step(step_built, parse_built, side)`: the whole board reading (the
`BoardReading`), the transition's events and the view equal, version by version. `core_events` runs it at
every step of every corpus battle (both sides), and `tests/version_test.rs` pins the properties one
battle can show: the chain equals its parse twin; a fork leaves its parent untouched; a compacted
fork chain equals the linear replay; typed == text (`streams_equal`, with teeth); the audit passes
the truth and catches a tampered view; a parse-built version refuses to step or audit.

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

| finding | field | poke-env reads | the truth (source) | reaches the obs |
|---|---|---|---|---|
| **PE-V10** | `boosts` | a fainted mon keeps its stages until switched out | none — the faint's `clearVolatile` zeroes them (the Rust board) | YES — the fainted active's stages in `active_context` at the forced-switch decision |
| **PE-R1b** | `status_counter` | +1 per `\|turn\|` while active — ONE AHEAD for a mon that entered after the residual, ONE BEHIND at a decision between the residual and the next `\|turn\|` (an end-of-turn forced replacement); frozen at a faint | the toxic STAGE: residual `[from] psn` chips since its switch-in (the Rust board's `Toxic(stage)`) | YES — the per-mon toxic slot |
| **PE-V16** | `volatiles` | a damaging Fire move of the holder's own ENDS Flash Fire (`Pokemon.moved`, any gen) | `flashfire` lasts until the holder leaves the field (the Rust board's `flash_fire`, probe-verified; `abilities.ts`) | YES — the `flashfire` binary volatile slot |

**Not findings — INFORMATION LIMITS**, read as poke-env reads them because no client can know
better: V15 (an own mon the current request did not re-sync holds the sighting count of its PP,
which lags an un-announced Pressure deduction — the audit checks it can only lag) and V14 / R4 (a
Transformed own mon's copied ability; a gen-3 request states neither). **UNRESOLVED**: a BENCHED
badly-poisoned mon's counter — the sim stores the stage it left with and resets it on switch-in,
poke-env and the view read 0; the stored stage never acts again, so neither side is clearly wrong
(counted by the audit as `UNRESOLVED:tox-stage-benched`, never checked).

The board audit (`check_view`) checks every SIM-FACT field against the engine at the truth — the
three findings' fields included (Flash Fire joined the audited volatiles) — and names what it
cannot check (V9, V14, V15) in `rules_fired`.

`legal_actions()` is `LegalActions.from_battle` over the raw `|request|` (the `BoardReading` keeps its
text) and `mask()` the 11-dim mask; slice V compares both to `Gen3ActionMasker` on every decision.

**The poke-env data the reading consults is GENERATED**: `tables.rs` from poke-env's pokedex, move
table, `Effect` lifecycle sets, `SideCondition`, type names and ignore set
(`python -m agents.battle.rust_core_present_tables --write`; `rust_core_present_tables_test.py`
fails the day it is stale).

## 4. Search on the core — `materializer=core`

`search_driver`'s `open_root` takes `core: "typed" | "text"` (and `side`: the one stream a tree
folds); a core root is a `BattleVersion`, and every `expand_many` arm's successor IS a version
(`NodeState::Core`). Per arm and wanted side it returns `core_pN = {view, legal, request, events
(the readings), mid, text_view?}`, and Python's `CoreSuccessorFactory` (a `ViewSuccessorFactory`
with `view_adapter`'s rules and `ViewEventFolder`'s re-parse deleted) encodes it with the unchanged
tracker cadence and encoder.

* **D10 — a leaf AT the intermediate decision.** An arm whose ply opens a second decision (a KO's
  replacement, a refused trapped switch) returns the version AT that decision (`mid`), built from
  the engine snapshot `resolve_turn_capturing(Capture { sessions: true })` takes there; the child
  node IS that leaf. So a forced switch is a NON-BRANCHABLE node on the core road, where the old
  roads expanded a D10 node from end-of-turn with the intermediate tokens — a semantic difference at
  depth ≥ 2, recorded as a finding (§6).
* **`recorded_exact` is REFUSED on core** (`resolve_turn_exact` has no production consumer on this
  path and no capture rule).
* **INTEGRITY** (`SearchConfig.integrity = N`, `--search-integrity N`, `expand_many`'s
  `integrity`): every Nth core arm is folded BOTH ways — typed and from the text — the driver
  asserts the two VERSIONS equal (`streams_equal`: board reading, events, view), and Python encodes both
  views and asserts the obs bytes and mask equal (`CoreIntegrityError`, naming the decision, the
  depth and the first differing obs block). **ON (N = 1) in every search test, fuzzer and parity
  gate; OFF (0) by default in production**; a sampled N stamps a number "integrity-sampled at 1/N".
  Every battery row is stamped `materializer` / `core_path` / `integrity`.
* **Defaults**: `SearchConfig.materializer = "core"`, `core_path = "typed"`, `integrity = 0`;
  `--materializer {core,protocol,view}` (default `core`). The protocol and view roads are NOT
  deleted — the cutover's deletion manifest names them (program §4).

## 5. The gates, and what each proves

| gate | proves |
|---|---|
| slice V — COMMIT / MILESTONE (`rust_core_parity_views.py`, via `core_events --views`) | at every decision, both viewers: `present()` + `legal_actions()` + the mask == the `LiveView` / `LegalActions` training builds, type-strict, no allowlist but the registered poke-env FINDINGS (§3, value-aware, counted per decision); the board audit; parse == step at every version |
| `materializer_parity_integration_test.py` | `protocol`, `view` and `core` give identical decisions, values and obs bytes, integrity on |
| `fork_sharing_parity_integration_test.py` (parametrized `view` / `core`) | one root fork per DECISION, shared across the K worlds, gives the same successor obs bytes as one fork per world — a reused factory leaks nothing between arms |
| `one_sided_view_parity_fuzz_test.py` (`sim`) | on real bridge battles, the core's root and arm views == the protocol road's `LiveView` |
| `core_successor_test.py` | the transport and the integrity check's teeth (a tampered `text_view` raises) |

## 6. Measurements and findings

Record: `designs/research_state/measurements/rust_core_m2_2026-09-23/`.
