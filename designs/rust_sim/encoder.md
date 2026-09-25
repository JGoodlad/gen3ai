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
| **Encoder** | `src/rust_sim/src/encoder/mod.rs` (`encode`, `encode_slice`, `prefill`, the active context, global env, board, pair history and event-window writers), `slot.rs` (the 122-dim per-mon slot), `data.rs` (the dex / prior tables, read from `data/pokemon/`), `layout.rs` (GENERATED), `wire.rs` (the row on the wire) |
| **Version** | `BattleVersion::encode(side, &mut [f32; OBS_DIM])` — the side's reading, its view, its legality, its TRACKERS (required) |
| **Python** | `agents/observation/rust_core_obs_layout.py` (the generator), `agents/battle/core_obs.py` (`wrap_row` / `check_row`), `agents/battle/rust_core_parity_obs.py` (slice O) |
| **Bridge** | `src/rust_sim/src/bin/sim_bridge.rs` — the `core_obs` START key and the `__OBS__` frame (§5a) |
| **Gates** | slice O (COMMIT + MILESTONE, `rust_core_parity_test.py`; through `core_events --obs` it also gates the PARSE chain's row, §6), the obs golden (`test_the_obs_golden_is_reproduced_by_the_core`), `rust_core_obs_layout_test.py`, `core_obs_test.py`, `cargo test` (`encoder::tests`, `tests/encoder_test.rs`, `tests/sim_bridge_core_obs_test.rs`) |

---

## 1. What it is

`encode(inputs, out)` is `Gen3ObservationEncoder.encode(battle, hp_tracker, legal, progress_clock,
recency, pair_history, event_window)` — the call `Gen3Env.embed_battle` makes for the trainee — over
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

## 5. The row on the wire

`wire::frame(row)` = `{"dtype":"<f4","shape":[2501],"b64":…}` — the row's little-endian float32 bytes,
in the reply of a pipe that already exists (`core_events --obs`; the process and the pipe protocol
are kept, program M4 "Transport"). Python wraps it with `np.frombuffer` (`core_obs.wrap_row`, a
read-only view, no copy) and REFUSES — never converts — a wrong dtype, shape or byte length;
`check_row` refuses an array that is not float32, `(2501,)` and C-contiguous. On the Rust side
`encode` takes `&mut [f32; OBS_DIM]` (the shape is the type) and `encode_slice` refuses a slice of any
other length before touching it. Pins: `core_obs_test.py`, `tests/encoder_test.rs`.

## 5a. The row on the TRAINING wire — `sim_bridge`'s core observation mode

The training env already talks to one `sim_bridge` child per env over its stdin/stdout pipe
(`src/utils/bridge/bridge_session.py`). The mode is OPT-IN per battle: START's `core_obs` key

```text
"core_obs": {"sides": ["p1"] | ["p2"] | ["p1","p2"], "decision_tense": <bool>, "switch_freeze": <bool>}
```

(all three keys REQUIRED when the key is present — the two booleans are the progress clock's
`ClockConfig`, training's `--progress-decision-tense` / `--progress-switch-freeze`, read and never
defaulted; an unknown key, an empty / repeated / unknown side, a non-boolean flag is a loud
`__ERR__`). **Absent or `null`, the child's stdout is BYTE-identical to the pre-mode binary.**

**The observation comes through the parser** (program §6c). Per requested side the child keeps a
PARSE-built version chain with trackers (`BattleVersion::parse_root_with(side, name, packed team,
cfg)`), advanced by `parse_advance` over exactly the lines that side was newly shipped in the write
(incremental — never the whole stream again). Every `CHOOSE` of a requested side is noted on its
chain (`note_choice`, the raw token) BEFORE the command is fed, the order `core_events` uses. For
each write (START's first emission, every `CHOOSE` / `FORCELOSE`), each requested side whose chain
took a DECISION at the write's boundary gets ONE frame, written **BEFORE that write's chunk frames**
(p1's first): the parent fires each chunk as an un-awaited task, so the row must already be stashed
when the request chunk is dispatched.

```text
__OBS__ p1 {"frame":{"dtype":"<f4","shape":[2501],"b64":…},"mask":[11 ints],
            "tokens":{"<idx>":"<choice>",…},"turn":<int>,"line":<int>,"rqid":<int>|null,"n":<int>}
```

| field | is |
|---|---|
| `frame` | `wire::frame` of `BattleVersion::encode` (§5; NaN-prefilled in test / self-check builds, zero-filled in release) |
| `mask` | `present::mask` — the 11-dim action mask |
| `tokens` | `present::choice_tokens` — the real mapper's choice string per legal action (§7) |
| `turn` | the reading's turn at the decision |
| `line` | the side's stream index of the `\|request\|` it decided on |
| `rqid` | the request JSON's `rqid` when it carries one — the engine never writes one, so `null` on the bridge |
| `n` | the frame's 0-based index among THIS side's frames in the current battle (0 at every START) — the alignment key a consumer counts its own decisions against |

A decision's request must be the LAST line the side was shipped in the write, and a write opens at
most one decision per side — either violation, and every parse / fold / encode failure, is a FATAL
`__ERR__` written IN PLACE of the write's chunks, and the mode stays failed for the rest of the
battle (never a skipped frame, never a fall-back). A battle that ends ships no frame for its terminal
board (no decision). The chains are dropped at every battle reset, and each START builds its own. The
mode builds the core's reading on its own parse chain: the transport's reveal fold and the core source
recording stay OFF (`tests/view_fold_opt_in_test.rs`).

**Pins** (`tests/sim_bridge_core_obs_test.rs`, real binaries, the bridge corpus's real teams under a
seeded random policy that also sends rejected choices and forfeits): every `__OBS__` row equals
`core_events --obs`'s row for the same battle BYTE for byte, mask and tokens equal, one per decision,
NaN-free, before its request chunk; a recycled persistent child equals a fresh one, and a one-side
request ships that side only; OFF (absent / `null`) is byte-identical and ON adds only the `__OBS__`
lines; the two clock booleans reach the rows; a malformed key is refused.

**Cost** (2026-09-24, release, `bench_core_obs_cost` in that file: 34 battles, 6,085 commands, median
of 9 interleaved runs; the box carried a production run, load ≈ 27 on 16 cores, and the bench ran at
`nice 19`, so the absolute figures are inflated): OFF 23.2 µs per command; ON adds **+138.5 µs per
frame** for `["p1"]` (2,755 frames) and +134.1 µs for both sides (5,482). An in-process breakdown of
the same path (13,775 frames, same box) puts **≈ 137 µs in the parse + tracker fold**, ≈ 30 µs in the
encode (§8's figure), ≈ 19 µs in the frame JSON and ≈ 2 µs in legality, tokens and mask: the TRACKER
FOLD dominates, not the encoder. **UNVERIFIED:** the figure on an idle box.

## 6. Slice O — the gate

`agents/battle/rust_core_parity_obs.py`, inside slice T's decision loop (the tracker fold runs once
for both): at every decision of both viewers, the row `Gen3Env.embed_battle` encodes (the real
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

On `materializer=core` (the default) search opens its tree with the TRACKERS on and expands with
`rows` (`search_driver`'s `expand_many`): each wanted side's leaf version is ENCODED in the driver and
comes back as `core_pN = {mid, row, mask, tokens}` — the `<f4` frame, the 11-dim mask and the choice
string per legal action (`present::choice_tokens`, the real mapper's `action_to_order` mirrored: a
switch is `switch <Pokemon.name>`, a move the first available `Move.id` matching the request slot —
Hidden Power by prefix, `recharge` as `move 1` — the round trip back to the index enforced). `row:
null` where the side does not decide (a `wait` request, the battle over, no legal action). Python
(`SearchEngine._materialize_core`) wraps the row with `np.frombuffer` and scores it: no view JSON, no
event fold, no Python tracker, no Python encoder and no Python prefix fork run on a core successor.
Slice O compares the tokens against the real mapper at every decision (12,677 at COMMIT);
`one_sided_view_parity_fuzz_test`'s CORE ROW road compares each arm's row to the protocol road's,
byte for byte; `fork_sharing_parity_integration_test::test_the_core_road_builds_no_python_fork` pins
that no Python fork is built.

## 8. The benchmark's core row

`src/agents/training/obs_build_benchmark.py` prints a CORE row: the profiled battle's recorded input
log is replayed through `core_events --obs --obs-bench SIDE K REPS` (the release build unless
`POKESIM_EMISSION_SELFCHECK=1`), which times the Rust encoder at the SAME decision — the version's
encode with its view memoized (the production shape) and `present()` + encode (cold) — and its row is
asserted byte-equal to the Python row before the time is printed (`--no-core` skips it).

## 9. Measurements

Recorded in [`../research_state/measurements/rust_core_m4_2026-09-24/`](../research_state/measurements/rust_core_m4_2026-09-24/README.md).
