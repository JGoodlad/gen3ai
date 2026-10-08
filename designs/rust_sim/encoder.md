# The Rust core's ENCODER — the observation row on the version (M4)

<!-- ALWAYS-CURRENT (this tree OWNS what it holds; see the root CLAUDE.md). State the truth, never
narrate a change. The code is `src/rust_sim/src/encoder/`; the program is
`designs/endstate/program_rust_core.md` §2 M4. -->

The Rust Core Program's M4 (`gen3_core_encoder_v1`, `gen3_core_obs_layout_v1`,
`gen3_core_obs_wire_v1`, `gen3_core_parity_obs_v1`) and the M6 cutover's Rust half
(`gen3_bridge_core_obs_v1`, `gen3_core_parse_obs_gate_v1`). Built alongside the Python path: **training
reads it only through `sim_bridge`'s OPT-IN core observation mode (§5a), which the Python env turns on
under its own flag, default OFF** — with the flag off, no training byte changes.

| | |
|---|---|
| **Encoder** | `src/rust_sim/src/encoder/mod.rs` (`encode`, `encode_slice`, `prefill`, the active context, global env, board, pair history and event-window writers), `slot.rs` (the 122-dim per-mon slot; `populated_slot` is the ONE writer of a populated slot), `hypothesis.rs` (X5's hypothesis row, §10), `oracle.rs` (the diagnostic ORACLE REVEAL, §11), `data.rs` (the dex / prior tables, read from `data/pokemon/`), `layout.rs` (GENERATED), `wire.rs` (the row on the wire) |
| **Version** | `BattleVersion::encode(side, &mut [f32; OBS_DIM])` — the side's reading, its view, its legality, its TRACKERS (required) |
| **Python** | `agents/observation/rust_core_obs_layout.py` (the generator), `agents/battle/core_obs.py` (`wrap_row` / `check_row`), `agents/battle/rust_core_parity_obs.py` (slice O) |
| **Bridge** | `src/rust_sim/src/bin/sim_bridge.rs` — the `core_obs` START key and the `__OBS__` frame (§5a) |
| **Gates** | slice O (COMMIT + MILESTONE, `rust_core_parity_test.py`; through `core_events --obs` it also gates the PARSE chain's row, §6), the obs golden (`test_the_obs_golden_is_reproduced_by_the_core`), `rust_core_obs_layout_test.py`, `core_obs_test.py`, `cargo test` (`encoder::tests`, `tests/encoder_test.rs`, `tests/sim_bridge_core_obs_test.rs`) |

---

## 1. What it is

`encode(inputs, out)` is `Gen3ObservationEncoder.encode(battle, hp_tracker, legal, progress_clock,
recency, pair_history, event_window)` — the call the deleted Python `Gen3Env.embed_battle` made for the trainee — over
the core's reading of ONE side's stream. It reads the same two surfaces the Python encoder reads:

* the **view** (`present()`, the `LiveView`): species / base stats, status, HP fraction, counters,
  spread, protect streak, boosts, volatiles, weather, side conditions, fainted counts;
* the **raw reading** (`BoardReading`'s `PMon`, poke-env's `Pokemon`): the item and consumed item,
  the types (`type_1` / `type_2`, temporary types included), the ability, the moveset (sorted by
  `Move.id`, the category from poke-env's pre-split table), the active mon's request-slot moves;

plus the legality (`LegalActions`: the request-order move block, the trapping bits) and the side's
**trackers** (M3: the HP belief, the progress clock, recency, pair history, the event window, the
wish and sleep folds). A stream built without trackers REFUSES to encode (a structurally-zero
tracker block is never written silently).

🚨 **The encoder never fixes semantics.** Every value is what the reading and the trackers hold. A
fact that is wrong there (the M3 loss catalogue's GIGO: stat drops recorded as rises, a Protect
block as a hit, …) is fixed THERE, in both languages, the same day — the encoder reproduces it.

## 2. Float exactness — the gate is BYTES

Each cell is the Python float expression evaluated in f64 in the SAME order, rounded ONCE to f32 at
the write (numpy's float32 assignment is round-to-nearest; so is `as f32`). Where the Python code
compares against an already-stored float32 (the volatile block's `max(vec[idx], value)`), the Rust
code compares against the f32 cell too. The clamps keep Python's argument order (`max(-1.0,
min(1.0, x))`: the first extreme wins). Logs (`ln` of the clock scalars) call the same libm the
Python `math` module calls on this box. The committed constants — `SAT_LUT`, `LOG_MAX_TURNS`, the
sleep tables — are written by the generator with `repr`, so each literal parses to the exact f64.
Slice O compares the float32 rows by BYTES (`tobytes()`), stricter than `np.array_equal`: a `-0.0`
for a `0.0` is a divergence.

## 3. Every cell is WRITTEN — the NaN poison

Test and fuzz builds (`debug_assertions` or the `emission-selfcheck` feature — `cargo test`, the
`selfcheck` profile every pytest session and fuzzer runs) fill the row with NaN before encoding;
release builds zero-fill (`encoder::prefill`, `NAN_POISON`). A slot, block or appended tail cell no
branch wrote reads NaN and fails slice O (a NaN equals nothing) and
`tests/encoder_test.rs::every_cell_is_written_at_every_decision_under_the_nan_poison` (teeth: removing
the active-flag write fails it at cells 121 / 243 / 853). **The honest limit:** each sub-encoder zeroes
its own sub-block first (Python's `np.zeros` per sub-encoder), so a skipped cell INSIDE a sub-block
reads 0, not NaN — that class is caught by the byte gate against the Python value, not by the poison.

## 4. The layout is GENERATED

`src/rust_sim/src/encoder/layout.rs` is rendered by `python -m agents.observation.rust_core_obs_layout
--write` from `agents/observation/constants.py` (every offset and dim, `EventCol`, the event-type and
item-transition ids), `gen3_effects` (`VOLATILE_SLOTS`, `GEN3_VOLATILE_TO_SLOT`, `NOT_A_VOLATILE`,
`CANT_REASONS_LIVE`), `turn_view.FAINT_CAUSE_VOCAB`, `TypeEncoder.TYPE_TO_IDX`, the status / weather /
screen maps, `assembler.SAT_LUT`, the sleep-wake tables, the protect floor and poke-env's
`_MOVE_CATEGORY_PER_TYPE_PRE_SPLIT`. `rust_core_obs_layout_test.py` (routine) fails the day it is
stale; a dim change is a one-place edit in `constants.py` plus a regeneration. The dex and prior
tables are NOT generated: `data.rs` reads `data/pokemon/{species,items,abilities,moves,ability_priors,
natures}.json` at runtime exactly as the facade does, so a `tools/` regeneration reaches both encoders.
A species / item / ability / move row with no numeric `num` is a LOAD ERROR on both sides (`data.rs::num_of`; `gen3_data._base.load_dex_json`) — neither reads a missing `num` as 0 any more (F-X5-5; no shipped row trips it).

## 5. The row on the wire

`wire::frame(row)` = `{"dtype":"<f4","shape":[OBS_DIM],"b64":…}` (the head is GENERATED into `layout.rs` as `FRAME_HEAD`; 2845 since `gen3_obs_facts_v1`, the X5 version break) — the row's little-endian float32 bytes,
in the reply of a pipe that already exists (`core_events --obs`; the process and the pipe protocol
are kept, program M4 "Transport"). Python wraps it with `np.frombuffer` (`core_obs.wrap_row`, a
read-only view, no copy) and REFUSES — never converts — a wrong dtype, shape or byte length;
`check_row` refuses an array that is not float32, `(OBS_DIM,)` and C-contiguous. On the Rust side
`encode` takes `&mut [f32; OBS_DIM]` (the shape is the type) and `encode_slice` refuses a slice of any
other length before touching it. Pins: `core_obs_test.py`, `tests/encoder_test.rs`.

## 5a. The row on the TRAINING wire — `sim_bridge`'s core observation mode

The training env talks to the `sim_bridge` / env core over its stdin/stdout pipe (the Python session that used to do this, `bridge_session.py`, was deleted in U3). The mode is OPT-IN per battle: START's `core_obs` key

```text
"core_obs": {"sides": ["p1"] | ["p2"] | ["p1","p2"]}
```

(`sides` REQUIRED when the key is present; an unknown key — the two clock booleans the key once carried,
`decision_tense` / `switch_freeze`, were deleted with training's `--progress-decision-tense` /
`--progress-switch-freeze` in the flag census, P11d, and are refused like any unknown key — or an empty /
repeated / unknown side is a loud `__ERR__`). The chains are built with `ClockConfig::default()`; the one
exception is a TEETH hook of a test / self-check build, `POKESIM_SIM_BRIDGE_TEETH=clock_start` (every chain's
clock starts at `n` = 1; compiled out of `--release`), which a gate that compares the bridge's rows with another
route's uses to prove it can SEE a clock divergence. **Absent or `null`, the child's stdout is BYTE-identical to the pre-mode binary.**

**The observation comes through the parser** (program §6c). Per requested side the child keeps a
PARSE-built version chain with trackers and NO native record (`BattleVersion::parse_root_unrecorded(side,
name, packed team, cfg)` — the row never reads the record, trackers.md §2), advanced by
`parse_advance_lean` over exactly the lines that side was newly shipped in the write (incremental —
never the whole stream again; LEAN — the chain keeps no per-transition events and MOVES each line's
readings to the trackers, every other read equal to `parse_advance`'s). The pins below hold this
chain's rows to `core_events --obs`'s recording, event-keeping chains byte for byte. Every `CHOOSE` of a requested side is noted on its
chain (`note_choice`, the raw token) BEFORE the command is fed, the order `core_events` uses. For
each write (START's first emission, every `CHOOSE` / `FORCELOSE`), each requested side whose chain
took a DECISION at the write's boundary gets ONE frame, written **BEFORE that write's chunk frames**
(p1's first): the parent fires each chunk as an un-awaited task, so the row must already be stashed
when the request chunk is dispatched.

```text
__OBS__ p1 {"frame":{"dtype":"<f4","shape":[2845],"b64":…},"mask":[11 ints],
            "tokens":{"<idx>":"<choice>",…},"turn":<int>,"line":<int>,"rqid":<int>|null,"n":<int>}
```

| field | is |
|---|---|
| `frame` | `wire::frame` of `BattleVersion::encode` (§5; NaN-prefilled in test / self-check builds, zero-filled in release); the whole object is `wire::obs_json_into` |
| `mask` | `present::mask` — the 11-dim action mask |
| `tokens` | `present::choice_tokens` — the real mapper's choice string per legal action (§7) |
| `turn` | the reading's turn at the decision |
| `line` | the side's stream index of the `\|request\|` it decided on |
| `rqid` | the request JSON's `rqid` when it carries one — the engine never writes one, so `null` on the bridge |
| `n` | the frame's 0-based index among THIS side's frames in the current battle (0 at every START) — the alignment key a consumer counts its own decisions against |

**The per-side state is ONE type, `pokesim::side_reader::SideReader`** (`src/rust_sim/src/side_reader.rs`):
the chain, the folded-line cursor, the decision count, the alignment checks and the sticky failure. `sim_bridge` holds
one per requested side; the `live_reader` session of live websocket play holds one for its side
([`live_reader.md`](live_reader.md)) — so a live game's rows are built by the code that builds training's.

A decision's request must be the LAST line the side was shipped in the write, and a write opens at
most one decision per side — either violation, and every parse / fold / encode failure, is a FATAL
`__ERR__` written IN PLACE of the write's chunks, and the mode stays failed for the rest of the
battle (never a skipped frame, never a fall-back). A battle that ends ships no frame for its terminal
board (no decision). The chains are dropped at every battle reset, and each START builds its own. The
mode builds the core's reading on its own parse chain: the core source recording stays OFF.

**Pins** (`tests/sim_bridge_core_obs_test.rs`, real binaries, the bridge corpus's real teams under a
seeded random policy that also sends rejected choices and forfeits): every `__OBS__` row equals
`core_events --obs`'s row for the same battle BYTE for byte, mask and tokens equal, one per decision,
NaN-free, before its request chunk; a recycled persistent child equals a fresh one, and a one-side
request ships that side only; OFF (absent / `null`) is byte-identical and ON adds only the `__OBS__`
lines; the two clock booleans reach the rows; a malformed key is refused.

**Cost** (2026-09-25, release, the box carrying a production run, everything at `nice 19`, so the
absolute figures are load-inflated and move with the load; **UNVERIFIED:** the figures on an idle
box). Process level, `bench_core_obs_cost`'s scripts (`CORE_OBS_BENCH_BATTLES=34`: 34 battles /
6,085 commands) replayed through ONE persistent child, OFF vs ON, the pre-change and the current
binary alternating one process per run, 10 rounds: ON adds **+55.0 µs per frame** for `["p1"]`
(2,755 frames; per-round 54–56; load 18), against +125.0 for the pre-change binary in the same run
(OFF ≈ 20.3 µs per command for both binaries); a load-26 repeat read +105.0 vs +209.7, both sides +75.7 vs
+176.5 per frame (5,482 frames, 8 rounds, load 31). A fresh child also pays a ONE-TIME **≈ 3.7 ms**
at its first frame (the encoder tables and the HP priors — the trackers share the bridge's dex,
`trackers::dex`; it was ≈ 8.9 ms), so ONE ~100-frame battle reads ≈ 90 µs per frame (219 before);
a training child is recycled every 5,000 battles, so the steady state is the figure.

In-process, per frame (`bench_core_obs_stages` in the same file: 17 battles, 1,396 frames, median
of 8 interleaved rounds against the pre-change code, load 18–28; its attribution re-runs between
the stages inflate the production stages against the process figure): **the fold 36.7 µs** (80.5
before) and **the frame 12.7 µs** (33.9 before). Attribution, each piece re-run standalone on the
same input: a TRACKERLESS parse chain 24.7 (the line parse 1.8; the request JSON, parsed once for
both reading folds, 7.4; the M1 reading fold ≈ 4.2 beyond the JSON; the board reading ≈ 5.9 beyond
the JSON on request lines and 5.0 on the others), then `present()` 6.3, the tracker decide 3.2,
legality 0.5 and ≈ 2 of decision bookkeeping by difference; the frame = the encode (view memoized)
6.0 + the base64 frame and tokens JSON 3.6 + legality and `choice_tokens` 0.9 + the rest of the JSON.

What the mode no longer pays (each change byte-identical: `tests/sim_bridge_core_obs_test.rs`, slice
O, the parse-path gate): the request JSON parsed twice and every line split two or three times (once
per reading fold); a char-at-a-time JSON string reader allocating every key (the request vocabulary
is interned, `jsonval::JStr`); the pokemon array, every request record and the teambuilder cloned
per request, and the spent backfill scan; a moveset COPY at every move read; a binary search of the
move table at every move row read (an id index now) and SipHash on the encoder's tables; allocating
`to_id` / item-key folds on ASCII; 36 × 5 owned-key map lookups in the pair block (one pass per map
now); the HP parse's filtered copies; a `Reading` clone per line into an event nobody reads
(`parse_advance_lean`); the native window record, built per line and read by nothing on this
path (`parse_root_unrecorded`); a char-at-a-time base64 of the 10 KB row and the frame JSON copied three
times (the line is now built once as bytes and written with one `write_all`); and a SECOND gen-3
dex load at the first decision.

## 6. Slice O — the gate

`agents/battle/rust_core_parity_obs.py`, inside slice T's decision loop (the tracker fold runs once
for both): at every decision of both viewers, the row the Python `Gen3ObservationEncoder` path encodes (formerly via the deleted `Gen3Env.embed_battle`; the real
`EpisodeTracker`, the legality snapshot, the incremental assembler) against the core's, BYTE-equal,
plus the 11-dim mask. **No allowlist.** `core_events --obs` also folds each side's PARSE chain (one
side's text, trackers on, the same `note_choice` tokens — the chain §5a ships to training) and REFUSES
a battle unless it decides at exactly the step chain's decisions and encodes a BYTE-identical row
with an equal mask and equal tokens (`version::parse_encode_matches_step`,
`gen3_core_parse_obs_gate_v1`; the first differing cell named by `encoder::cell_name`) — so every
slice-O run also gates the encode path the bridge ships. A divergence is classed by FIELD, slot-independent
(`our_team moves+7`, `event_window MAGNITUDE`); the census also counts value-differing, byte-only and
NaN cells, and which blocks were ever nonzero (a tier that never saw a block nonzero fails).

| tier | runs |
|---|---|
| COMMIT | `python3 -m pytest src/agents/battle/rust_core_parity_test.py -q` (unmarked: slice O on the recorded corpus, the obs GOLDEN reproduced by the core, the teeth) |
| MILESTONE | `… -m slow -q -n 2` — slice O on every played battle (pool, policy, LADDER), the verdict in `designs/ops/slow_tier_status.json` |
| FRESH | `python src/agents/battle/rust_core_trackers_fuzz_test.py [--minutes N] [--procedural P]` — slice O (with E / V / T) on new battles every run, pool + mechanic-dense + PROCEDURAL teams |

**The obs golden.** `test_the_obs_golden_is_reproduced_by_the_core` plays `golden_obs_capture`'s
fixed battle set, replays the recorded input logs through `core_events --obs`, and requires the
trainee's rows to hash EXACTLY to `training/golden_obs_fixture.json` and to equal the Python capture's
vectors byte for byte.

**Teeth** (`rust_core_parity_test.py`): a Python encoder change the core does not mirror (the move PP
normaliser) fails on `our_team moves+` / `opp_team moves+`; a NaN cell and a `-0.0` in the core row
each fail.

## 7. Search takes rows

Search (the core, its one road) opens its tree with the TRACKERS on and expands with
`rows` (`search_driver`'s `expand_many`): each wanted side's leaf version is ENCODED in the driver and
comes back as `core_pN = {mid, row, mask, tokens}` — the `<f4` frame, the 11-dim mask and the choice
string per legal action (`present::choice_tokens`, the real mapper's `action_to_order` mirrored: a
switch is `switch <Pokemon.name>`, a move the first available `Move.id` matching the request slot —
Hidden Power by prefix, `recharge` as `move 1` — the round trip back to the index enforced). `row:
null` where the side does not decide (a `wait` request, the battle over, no legal action). Python
(`SearchEngine._materialize_core`) wraps the row with `np.frombuffer` and scores it: no view JSON, no
event fold, no Python tracker, no Python encoder and no Python prefix fork run on a core successor.
Slice O compares the tokens against the real mapper at every decision (12,677 at COMMIT);
`core_row_parity_fuzz_test` compares each arm's row to the poke-env replay's, byte for byte.

## 8. The benchmark's core row

`src/agents/training/obs_build_benchmark.py` prints a CORE row: the profiled battle's recorded input
log is replayed through `core_events --obs --obs-bench SIDE K REPS` (the release build unless
`POKESIM_EMISSION_SELFCHECK=1`), which times the Rust encoder at the SAME decision — the version's
encode with its view memoized (the production shape) and `present()` + encode (cold) — and its row is
asserted byte-equal to the Python row before the time is printed (`--no-core` skips it).

## 9. Measurements

Recorded in [`../research_state/measurements/rust_core_m4_2026-09-24/`](../research_state/measurements/rust_core_m4_2026-09-24/README.md).

## 10. X5's hypothesis row — the one SYNTHETIC input (`gen3_x5_dex_rows_v1`)

X5 (`designs/endstate/design_x5_belief_tokens.md` §3.4, build unit U1) gives each unrevealed opponent
seat a CONCRETE species hypothesis, encoded from the per-mon row THIS encoder writes for "species s
present, unrevealed set, full HP, no status". The encoder never meets that state on its own (an
opponent mon enters the reading only by appearing), so `encoder::hypothesis::hypothesis_slot(species)`
builds the smallest synthetic input and hands it to the SAME slot writer the team loop uses
(`slot::populated_slot` — the real path and this one cannot render a mon differently):

| input | built as |
|---|---|
| the raw reading | `PMon::from_species(s)` (poke-env's `Pokemon(species=…)`: the pokedex's stats, types and the one-ability inference, as a real switch-in builds it) + `set_hp_status("100/100")` |
| the view | `present::view::mon_view(mon, active = false, own = false)` — what `present()` builds for an opponent mon |
| the trackers | `SideTrackers::new` — a side that has seen nothing (recency saturated, no Hidden-Power evidence, no sleep source, no last action) |
| the tail | not trapped, not active |

Every cell is written (a test / self-check build NaN-prefills the slot; a leftover NaN is a FAULT).
`core_events --dex-rows` prints the rows for the species ids on stdin; `python -m
agents.model.hypothesis_dex_rows --write` renders them into the COMMITTED table
`src/agents/model/hypothesis_dex_rows.json` (one row per base-form species, keyed by num; 386 rows).

**The declared cell classes** (`hypothesis::CELLS`, tiling the 122 cells; the artifact carries them):
`species` (species, types — depend on the species) · `default` (species_known, the opponent's empty
spread, the Hidden-Power block, protect odds, the trapping bits — a fresh mon carries the same) ·
`revealed_on_field` (item, ability, moves — zero / the Smogon prior here; the FIELD can reveal them,
and the real row's own flag cell says when) · `on_field` (status, HP fraction, the status counters,
the sleep belief, recency, last action, the active flag — the mon's state on the field).

**Gates.**
* `hypothesis_dex_rows_sim_test.py` (`sim`): the table regenerated through the encoder is BYTE-equal
  to the committed file; and it runs the cross-check below in the self-check build.
* `tests/hypothesis_dex_rows_test.rs` — the REAL-STATE cross-check: over a fixed seeded set of real
  Rust battles, at each opponent mon's FIRST appearance (the first decision of the viewing side with
  the mon in its reading, on the PARSE chain), the encoder's real slot equals `hypothesis_slot` of that
  species byte for byte on every cell outside the `on_field` blocks and outside a `revealed_on_field`
  block whose flag the real row set. COVERAGE: every one of the 386 base-form species (constructed teams
  of six in num order, first pokedex ability, Leftovers, Roar / Toxic / Seismic Toss / Protect, a
  switch-preferring seeded policy) — 33 battles, 388 first appearances, 38,301 cells compared, 0
  differing; the field revealed 21 items and 12 abilities. REALISM: the bridge corpus's 21 real team
  pairs — 252 first appearances of 52 species, 24,689 cells compared, 0 differing (85 items, 1 ability
  revealed) (2026-10-03).
* `encoder::hypothesis::tests` (`cargo test`): the cells tile the slot, a pristine row's fixed cells,
  an unknown species refused, and the comparator's teeth (each compared class, a signed zero, an item
  cell that differs without its reveal flag).

## 11. The ORACLE REVEAL — a diagnostic observation mode, never production (`gen3_oracle_reveal_v1`)

`--oracle-reveal {off,species,full}` (`designs/endstate/design_x5_belief_tokens.md` §7.6 "As built"; backlog X32) writes
the opponent's TRUE species (`species`) or whole SET (`full`) into the opponent team block of the observation from turn 1,
as if the game had a team preview. It enters the shared trunk through the row; it is not a side input to any head. The code is
`src/rust_sim/src/encoder/oracle.rs`; the mode is the Rust env core's `Spec.oracle_reveal` (a REQUIRED spec key,
`protocol.SPEC_KEYS`), and `off` leaves every row byte-identical to the encoder that had no reveal. The key takes ONE level
(a string: both sides, the run's recorded mode) or a PER-SIDE pair `[p1, p2]` (`core::spec::Reveal`; each side's chain told
the other side's team at its own side's level), which only the head-to-head engine declares (`main.h2h`'s one-sided /
both-sided oracle cells, X5 A/B §7.7(a)); a symmetric spec is always written as the string, so its text is unchanged.
`oracle_reveal_test.rs::a_per_side_reveal_writes_each_side_exactly_as_the_symmetric_core_of_its_level` pins that a split
core's side rows (obs, mask, every label column) are bit-identical to the symmetric core of that side's level.

| piece | is |
|---|---|
| `Oracle` | the OTHER side's packed team (`team::unpack`, species ids by `to_id`), built per episode and side by `rust_env`'s `Env::start` and held on that side's `SideStream` (`BattleVersion::with_oracle`; an `Arc`, so a fork shares it); `Inputs.oracle` carries it into `encode` |
| the tail | `slot::team`, for the OPPONENT block only: after the seen mons (reveal order, `off`'s bytes) one slot per unseen mon in dex-num order, each `hypothesis_slot(species)` — the row the encoder writes for a never-seen mon, §10 — copied from the `OracleMon` built at episode start |
| `Oracle::tail(revealed)` | the unseen mons: one oracle entry consumed per revealed mon, matched by DEX NUM (a forme shares its base species' num), a revealed mon the oracle team does not hold is a FAULT. The encoder AND the label writers call it, so the row and the labels share one slot order |
| `SPECIES_SLOT_CELLS` / `FULL_SLOT_CELLS` + `check_slot` | the producer's THROWING guard, per level: all 122 cells declared (`SpeciesDerived` / `SetFact` / `Zero` / `One`; each declaration tiles the slot) and every built tail slot checked bit for bit |
| `full_slot` / `Oracle::overlay` | the `full` level: an UNSEEN mon's slot is the row of an OWN mon of that set (the same `populated_slot`, a `PMon` built like an own mon's reading, a view with `spread_known`); a SEEN mon's slot is the reading's with only the facts play has not revealed written on top |

**What a tail slot carries.** The species' dex row (num, base stats, types) and the ability block (the Smogon prior,
or the one ability of a one-ability species) — derived from the species alone — plus the pristine state of a mon that
has never been on the field: `species_known` 1, HP fraction 1.0, recency "never seen", full Protect odds. Item, status,
counters, moves, spread, the Hidden-Power block, the sleep belief, the last action and the trapped / active flags are
exactly 0.0. A mon is never marked seen, acted or active by being listed. The pair-history block needs no change: an
unseen slot's pair cells are the never-interacted values, bit-for-bit an absent slot's.

**The `full` level.** Item, ability, the four moves (a bare Hidden Power typed from the set's declared type or its IVs, as the owner's request spells it) and the spread block (`PMon::backfill_spread`, the own mon's own backfill: nature lower-cased, `serious` if none) are written for every opponent mon, `hp_revealed` is 1 (probs 0, as for an own mon), and every other cell is the never-seen mon's. The slot writer's spread block and `hp_revealed` follow the view's `spread_known` (own: true; an opponent under `off` / `species`: false). A seen mon keeps what play revealed: the item only while the reading's is the unknown sentinel (`None` is a consumed / removed item, not an unknown one), the ability only while none is revealed, an observed move keeps its slot and tracked PP, a bare `hiddenpower` the reading learned stands for the typed one, a transformed mon gains no move. The independent oracle is the OPPOSING chain's own-team slot of the same mon in the same battle.

**Scope.** The reveal exists on the Rust env core's chains (training, the in-loop eval core). `sim_bridge`'s `core_obs`
mode, `core_events --obs` (slice O) and the search chains (`search_driver`, the env core's search, the fork arm) build
`off` rows: the Python `Gen3ObservationEncoder` has no reveal, which is why slice O gates `off` only.

**Gates.** `src/rust_env/tests/oracle_reveal_test.rs`: `off_is_inert` (the off obs / mask / label bytes pinned to a
digest recorded on `e0d56693`, the commit before the build), `species_bytes_are_pinned`, `full_bytes_are_pinned`, the
DIFFERENTIALS `species_differs_from_off_only_in_the_declared_cells` and `full_differs_from_off_only_in_the_opponent_block_and_tells_the_true_set`
(real battles, `off` against the level, every decision, both sides; `full` also against the opposing chain's own row), the
edge cases (formes, Species-Clause duplicates, a reveal in play, a revealed mon off the oracle team, a real Forecast battle
through a forme change at both levels);
`encoder::oracle::tests` (the cells tile the slot, the guard's teeth per block); `label_lookup_guard_test.rs` (the
labels' consumer guard); `src/utils/rust_env/oracle_reveal_integration_test.py` (the real core from Python).

## 12. The OBS-FACTS block — the row's LAST block (`gen3_obs_facts_v1`)

`encoder::facts::obs_facts(inputs, tables, out)` is `agents/observation/obs_facts.py`'s
`encode_obs_facts`, cell for cell (f64, one round at the write): what the opponent has seen of our team,
the opponent active's Choice-lock evidence, the actives' Encore / Taunt / Disable / Uproar / partial-trap
turns and each side's screen turns (`ARCHITECTURE`-level detail: `src/agents/observation/CLAUDE.md`).
`encode_into` writes it at `OFFSET_OBS_FACTS` (2761) as step 8, the row's last block (appended at the X5
version break, config v144, part 3: `OBS_DIM` 2761 → 2845, the prefix byte-identical; `cell_name` names a
cell `obs_facts+k`). `BattleVersion::encode_facts(side, &mut [f32; OBS_FACTS_DIM])` computes the block
alone from the same inputs (the engine-truth test reads it). Slice O covers it through the whole-row byte
comparison (`obs_facts` is one of its non-vacuity blocks); the separate `"facts"` field `core_events
--obs` shipped beside the row while the block was outside it, and slice O's `[FACTS]` comparison of it,
are DELETED as redundant with the row compare. The reading half is `present/` (V18 / V19) and the fold is
`trackers::facts` (`trackers.md`). The ENGINE truth test is `tests/obs_facts_truth_test.rs`: 60 seeded
random-legal battles, every decision, the encoded screens equal the engine's remaining duration, the
engine's volatile durations inside the encoded bounds, no NOT-locked proof while the engine holds
`choicelock` and the first move IS the locked move.

