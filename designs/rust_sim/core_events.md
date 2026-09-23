# The core's event layer — typed lines, the reading, `parse`, the record (`gen3_core_events_v1`)

<!-- ALWAYS-CURRENT (a `designs/rust_sim/` topic doc): state the truth, never narrate a change. The
leaf `src/rust_sim/CLAUDE.md` keeps the rule, the command and the hazard; this file keeps the
detail. Program context: `designs/endstate/program_rust_core.md` (milestone M1). -->

The Rust Core Program's M1 (`designs/endstate/program_rust_core.md`): every omniscient line the
port emits is TYPED at the source; one side's stream becomes typed [`CoreEvent`]s carrying the
reading `Gen3Battle` would record; `parse(lines)` reads a side's text into the same events; the
events persist as a versioned RECORD; and a parity harness holds all of it to `Gen3Battle`, per
viewer, per event, with no allowlist. **Nothing in training uses any of it** (§1 of the program:
build alongside, one cutover).

Code: `src/rust_sim/src/core_events/` (`line.rs`, `reading.rs`, `side.rs`, `parse.rs`,
`record.rs`, `jsonval.rs`, the generated `schema.rs`), the typed `ProtocolBuilder`
(`src/rust_sim/src/protocol.rs`), the bridge's core tracking (`src/rust_sim/src/bridge.rs`,
`BridgeSession::new_core` / `new_construct_turn0_core` / `core_events`), and the replay/record
binary `src/rust_sim/src/bin/core_events.rs`. Python: `src/agents/battle/rust_core_parity.py` (+
`_test.py`), `src/agents/battle/offline_feed.py`, `src/agents/battle/rust_core_schema.py`.

---

## 1. Always compiled, used by nothing in production — the choice and why

The Phase-0 spike was a cargo feature (`event_spike`) whose hooks compiled to NOTHING in the
default build. M1 **deleted the feature** and made typed emission the ONE emission path:

* **Every `ProtocolBuilder` method builds a [`Line`] and the text is `Line::render()` of it.** There
  is no raw-string escape hatch (`push_raw` is gone; its 17 call sites are typed methods). So the
  typed value and the bytes CANNOT disagree — one representation, not a typed record staged
  beside a separately formatted string (the spike's shape, where only a gate could notice a
  disagreement).
* **What production still does not USE is the RECORDING**: `ProtocolBuilder::record_core()` (the
  per-line `SourceRec`s) and the bridge's per-side source tracking are runtime-off unless a session
  is built with `BridgeSession::new_core` / `new_construct_turn0_core`. `sim_bridge` never does.
* **Why not keep the feature gate (the build-alongside rule reads both ways).** The rule is "two
  paths cost only when both are IN USE". A cfg-gated path is a second build configuration that is
  not type-checked in the default build (the spike's macro argument was not even compiled), so it
  rots between the milestones that would need it; and a typed-beside-text design IS two
  representations in use. Rendering the text from the typed value is one path, compiled and tested
  on every commit, and production's bytes are proven unchanged by the port's byte gates (below).
* **What it costs production**: building a small typed value per line before rendering (a
  disabled builder builds nothing — the seed suite's zero-cost invariant holds), and a retro-edit
  (`attrLastMove`) re-parses the one line it edits. Not separately timed; the differential gates
  below are the correctness claim.

**Byte identity of production output** — every rung ran on the typed builder: `cargo test` (the
132-battle / 19,348-line protocol capture gate, the 44-battle / 2,377-write `write_line` gate, the
bridge goldens, the 77-fixture byte-fuzz corpus, the e2e capstone, every seed pin), plus
`core_events_test::recording_changes_no_byte_of_the_production_stream` (a core session and a plain
one ship identical chunks and end on the same PRNG seed).

## 2. The typed line (`line.rs`)

`Line { kw: Kw, fields: Vec<Field> }`. `Kw` is one variant per protocol keyword (the 120 of
`MESSAGE_POLICY` + the player-intercepted ones), GENERATED with its routing into `schema.rs` from
`agents/battle/battle_event.py` + `Player.MESSAGES_TO_IGNORE` by `python -m
agents.battle.rust_core_schema --write`, and pinned by `rust_core_schema_test.py` (routine) — the
Rust core cannot classify a keyword differently from poke-env without that test failing.

A `Field` is typed only where the keyword's grammar puts that type AND it re-renders to the same
bytes: `Ident` (`p1a: Nick` / slot-less `p1: Nick`), `SideRef` (a side condition's first field),
`Hp` (`0 fnt` / `x/y` / `x/y st`, at the keyword's HP position), `From(Cause)` (`[from] item:` /
`ability:` / `move:` / bare), `Of(Ident)`, `Wisher`, `Tag` (`[still]`, `[miss]`, …), `Text`,
`Empty`. Everything else is text, so `Line::parse` is LOSSLESS for any line and
`render(parse(s)) == s`; an unknown keyword is REFUSED (`LineError::UnknownKeyword`). The builder
constructs every line CANONICALLY (`Line::parse(render(l)) == l`) — gated on every source record
of every corpus battle (`core_events_test`, and the binary refuses otherwise).

## 3. Source records and the per-side stream (`side.rs`)

With recording on, `ProtocolBuilder` keeps one `SourceRec { line, turn, scope }` per committed
line (conservation by construction). `scope` is the ENGINE's action — `Move(side)` / `Switch(side)`
/ `Residual` / `Other` / `Start` — set by the turn loop (`turn/driver.rs`) and around Pursuit's
nested strike (`turn/switch.rs`): the sim's own answer to "whose move owns this line".

Each side's shipped lines carry the index of the source record they derive from
(`BridgeChunks::core`). `side::step_events` rebuilds each line from its TYPED record (the typed
HP-privacy fold `side_view`) and **refuses** unless it renders exactly the bytes the bridge
shipped, and unless the side received every source record once, in order, except the owner-only
lines the privacy split withholds (gen-3 Pressure's `-ability …|[silent]`, the Intimidate-vs-Sub
`-hint`) and the per-format tier/rule reframe. The bridge's own frames — `|request|`, `|error|`,
the reframed tier/rules and the forced-Struggle `|-activate|…|move: Struggle` — are the only
unsourced lines it accepts.

## 4. The reading (`reading.rs`) and its named rules

`Reader` folds one side's typed lines into `Reading`s — exactly `Gen3Battle._capture_pre` +
`_build_event` + `_move_suffix_events` + `record_choice_rejected`, over the five facts of poke-env's
state those read (species, HP fraction, status, active, the resolving move's side) and the turn,
each transition mirroring poke-env's own line (requests included — they write our own team).
Every reading is checked against both halves of `gen3_event_value_schema_v1`.

Where the reading is not the simulator's truth, the rule is NAMED and pinned by a unit test to the
poke-env line it mirrors:

| rule | the reading | mirrors |
|---|---|---|
| R1 | an outcome line's side = the last `\|move\|` line's, reset only at `\|turn\|` | `abstract_battle.py:711`, `:1634`; `gen3_battle.py:716,733` |
| R2 | an effectiveness line with no open move → the side opposite the defender | `gen3_battle.py:735-738` |
| R3 | a MISS/FAIL/CRIT's target = the mon NAMED at index 2 (for `-miss`, the USER) | `gen3_battle.py:717,727` |
| R4 | a `[still]` / empty-target move targets the OTHER side's active | `gen3_battle.py:532-539` |
| R5 | `\|move\|…\|[miss]` / `[notarget]` adds a synthetic MISS / FAIL (`from="move-suffix"`) | `gen3_battle.py:272-274,488-502` |
| R6 | HP in the viewer's rendering (own exact, foe `ceil%`) | `Pokemon.current_hp_fraction`; `bridge.rs::hp_percent` |
| R7 | an effectiveness event carries only its multiplier (`[from] ability:` dropped) | `gen3_battle.py:744-749` |
| R8 | `-cureteam`'s `status` = field 3 verbatim (the `[from] move: Aromatherapy` clause) | `gen3_battle.py:687-691` |
| R9 | DAMAGE/HEAL/SETHP drop the line's `[of]` (Recoil, Leech Seed, drain) | `gen3_battle.py:642-652` |
| R10 | a mon named but never introduced is created with `species = to_id(name)` (a NICKNAME) | `abstract_battle.py:421-424` |
| R11 | `-formechange` / `detailschange` never rename the species later events name | `pokemon.py:469-471` |

**The truth sits beside the reading, and the Python schema is unchanged** (Phase-0 finding F3).
The dropped `[of]` (R9) and the dropped `[from] ability:` (R7) are fields of the typed LINE every
`CoreEvent` carries; the engine scope and exact foe HP are in the `SourceRec`. Changing
`gen3_event_value_schema_v1` would change what training reads, so it is NOT changed: a retrain that
wants the truth reads the typed line, with no re-derivation.

## 5. `parse` (`parse.rs`) and the M1 gate

`parse(lines, viewer) -> Vec<CoreEvent>` reads ONE side's text (all a real server sends): each
line typed, the same `Reader`, and the one attribution the protocol never prints — an outcome
line's OWNER — recovered from line order (`OwnerScan`: `|move|X` opens X's move; `|switch|`,
`|turn|`, `|upkeep` and the bare `|` close it; `|drag|` does not). `CoreEvent.owner` is defined on
outcome lines only; the step path takes it from the engine scope. `parse_matches_step` (typed line,
owner, every reading) is `parse(emit(step)) == step`.

## 6. The persisted record (`record.rs`, `event_schema` = `gen3_core_event_v1`)

One side's stream as JSON lines: a header (`record`, `event_schema`, `core_commit`, `path` step |
parse, `viewer`, `format`, `showdown_version`, `battle`, `lines`) then one line per protocol line —
`{"i","text","owner","readings"}`. The TEXT is the authority; the typed line is `Line::parse` of it
(so it is not stored twice). Floats are written in Rust's shortest round-trip form, so a JSON
reader keeps int vs float and the exact value. `read` REFUSES an unknown record kind or
`event_schema`, a bad line count, and a text that no longer parses; `write(read(bytes)) == bytes`;
`reparse` (the migration) re-derives the stream from the text; `check_reparse` is the golden gate.

**Golden corpus**: `src/agents/battle/rust_core_parity_fixtures/records/*.jsonl.gz` — the COMMIT
tier's 8 battles + the first battle of each of the protocol capture scenarios, both viewers (62
records; gzip'd because the text carries every `|request|` frame, ~8 MB plain / 600 KB gzip'd).
Regenerate: `python -m agents.battle.rust_core_parity write-records`. Gate:
`rust_core_parity_test.py::test_the_golden_records_round_trip_and_reparse` (`core_events
--check-records`).

## 7. The parity harness — slice E, two tiers

`python3 -m pytest src/agents/battle/rust_core_parity_test.py -q` (COMMIT, `sim`, ~2 s) and
`… -m slow -q -n 2` (MILESTONE; verdicts into `designs/ops/slow_tier_status.json`). Both compare,
per viewer and per event, `seq · turn · kind · side · actor · target · value · raw` of the core's
readings against the log of a `Gen3Battle` fed the same per-side text through `offline_feed`
(TYPE-strict: an int is not a float). No allowlist.

| tier | corpus |
|---|---|
| COMMIT | `commit_tier.json.gz` (6 seeded-random battles over 12 pool teams + 2 `production`-policy battles, recorded input logs, the chunk bytes pinned by digest) + the six byte-fuzz fixtures that carry the four shapes random battles never reach + the first battle of each protocol capture scenario |
| MILESTONE | 2 × 200 seeded-random (keys 0-199, 5000-5199) and 2 × 50 `production`-policy battles PLAYED live — the live `Gen3Battle` logs must ALSO equal the offline feed's — + the protocol corpus × 2 seeds + every byte-fuzz fixture; the pool hash and the checkpoint sha256 are pinned in `rust_core_parity_fixtures/manifest.json` and the tier refuses on a mismatch (`… write-manifest` in the same commit) |

🚨 **The four ambiguity-prone shapes live in the BYTE-FUZZ corpus, not the protocol capture
golden.** Damp's `[of]` cant, the slot-less future-move `-miss`, a bench `-curestatus` and
`[from] lockedmove` occur 0 times in `protocol_capture_golden.txt`; they are in
`tests/vectors/byte_fuzz_corpus/` (`42_…`, `29_…`/`30_…`/`31_…`, `70_…`, `52_…`), which is what
both tiers and `core_events_test` run. A capture's scripted choices are replayed per decision with
`CHOOSEIF` (a side whose choice this boundary already accepted is skipped); where a capture re-sends
a rejected choice past the bridge's no-progress cap (8 consecutive rejects) the battle is TRUNCATED
there and everything before it is still checked.

## 8. Cost of `parse` for a text-driven observation path

`designs/research_state/measurements/rust_core_m1_2026-09-23/` (`core_events --bench-parse`):
incremental `Line::parse` + the reading fold of one side's stream, per decision (≈ 12.5 lines, the
`|request|` included), in-process Rust, no IPC — against a `Gen3Env.step` wall on the rust bridge
with random legal actions (no policy forward).
