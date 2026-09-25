# The Rust core's ENCODER — the observation row on the version (M4)

<!-- ALWAYS-CURRENT (this tree OWNS what it holds; see the root CLAUDE.md). State the truth, never
narrate a change. The code is `src/rust_sim/src/encoder/`; the program is
`designs/endstate/program_rust_core.md` §2 M4. -->

The Rust Core Program's M4 (`gen3_core_encoder_v1`, `gen3_core_obs_layout_v1`,
`gen3_core_obs_wire_v1`, `gen3_core_parity_obs_v1`). Built alongside the Python path: **training
reads none of it** (the cutover is M6).

| | |
|---|---|
| **Encoder** | `src/rust_sim/src/encoder/mod.rs` (`encode`, `encode_slice`, `prefill`, the active context, global env, board, pair history and event-window writers), `slot.rs` (the 122-dim per-mon slot), `data.rs` (the dex / prior tables, read from `data/pokemon/`), `layout.rs` (GENERATED), `wire.rs` (the row on the wire) |
| **Version** | `BattleVersion::encode(side, &mut [f32; OBS_DIM])` — the side's reading, its view, its legality, its TRACKERS (required) |
| **Python** | `agents/observation/rust_core_obs_layout.py` (the generator), `agents/battle/core_obs.py` (`wrap_row` / `check_row`), `agents/battle/rust_core_parity_obs.py` (slice O) |
| **Gates** | slice O (COMMIT + MILESTONE, `rust_core_parity_test.py`), the obs golden (`test_the_obs_golden_is_reproduced_by_the_core`), `rust_core_obs_layout_test.py`, `core_obs_test.py`, `cargo test` (`encoder::tests`, `tests/encoder_test.rs`) |

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

## 6. Slice O — the gate

`agents/battle/rust_core_parity_obs.py`, inside slice T's decision loop (the tracker fold runs once
for both): at every decision of both viewers, the row `Gen3Env.embed_battle` encodes (the real
`EpisodeTracker`, the legality snapshot, the incremental assembler) against the core's, BYTE-equal,
plus the 11-dim mask. **No allowlist.** A divergence is classed by FIELD, slot-independent
(`our_team moves+7`, `event_window MAGNITUDE`); the census also counts value-differing, byte-only and
NaN cells, and which blocks were ever nonzero (a tier that never saw a block nonzero fails).

| tier | runs |
|---|---|
| COMMIT | `python3 -m pytest src/agents/battle/rust_core_parity_test.py -q` (unmarked: slice O on the recorded corpus, the obs GOLDEN reproduced by the core, the teeth) |
| MILESTONE | `… -m slow -q -n 2` — slice O on every played battle (pool, policy, LADDER), the verdict in `designs/ops/slow_tier_status.json` |

**The obs golden.** `test_the_obs_golden_is_reproduced_by_the_core` plays `golden_obs_capture`'s
fixed battle set, replays the recorded input logs through `core_events --obs`, and requires the
trainee's rows to hash EXACTLY to `training/golden_obs_fixture.json` and to equal the Python capture's
vectors byte for byte.

**Teeth** (`rust_core_parity_test.py`): a Python encoder change the core does not mirror (the move PP
normaliser) fails on `our_team moves+` / `opp_team moves+`; a NaN cell and a `-0.0` in the core row
each fail.

## 7. The benchmark's core row

`src/agents/training/obs_build_benchmark.py` prints a CORE row: the profiled battle's recorded input
log is replayed through `core_events --obs --obs-bench SIDE K REPS` (the release build unless
`POKESIM_EMISSION_SELFCHECK=1`), which times the Rust encoder at the SAME decision — the version's
encode with its view memoized (the production shape) and `present()` + encode (cold) — and its row is
asserted byte-equal to the Python row before the time is printed (`--no-core` skips it).

## 8. Measurements

Recorded in [`../research_state/measurements/rust_core_m4_2026-09-24/`](../research_state/measurements/rust_core_m4_2026-09-24/README.md).
