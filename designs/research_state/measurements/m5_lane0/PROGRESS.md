# M5 Lane 0 — the shared core boundary: PROGRESS (resume point)

Lane 0 of `designs/endstate/program_rust_core.md` §2 M5. Owns `src/rust_env/{Cargo.toml, build.rs,
src/lib.rs, src/core/{mod, spec, columns, pool, dispatch, refusal}.rs}` and
`src/utils/rust_env/{protocol.py, columns.py, stamp.py}`. Seed: the Phase-A prototype in
`../rust_core_m5_transport_2026-09-26/proto/`.

Gates: ① rows byte-equal to `sim_bridge`'s `__OBS__` on the same input log (F-M5-5); ② determinism
(seed → bytes, thread-count-invariant); ③ the column schema GENERATED from one table + a routine pin;
④ the refusal policy (quarantine + banked input log + typed class); ⑤ the stamp's teeth. Principle:
the DECLARED LIFECYCLE (`*_after_freeze` counters stay 0).

## How to build / test (worktree-local target only)

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m utils.rust_env.columns --write          # after any table change (columns.py / protocol.py)
python3 -m pytest src/utils/rust_env -q           # gate ③ (routine)
(cd src/rust_env && CARGO_TARGET_DIR=$PWD/target cargo test --profile selfcheck --features emission-selfcheck)
python3 -m pytest src/utils/rust_env/core_cargo_test.py -q   # the same, as the routine tier runs it
```

## Units

| # | unit | status | commit |
|---|---|---|---|
| 1 | crate skeleton + the GENERATED column / op / status / counter contract + its routine pin (gate ③) | LANDED | `32341863` |
| 2 | `spec.rs` + `pool.rs` + `dispatch.rs` + `refusal.rs`; gate ② (`tests/determinism_test.rs`), gate ④ + lifecycle (`tests/lifecycle_refusal_test.rs`), run by the routine `core_cargo_test.py`; `protocol.spec_json` / `error_from_json` | LANDED | `2be7e4e3` |
| 3+4 | `build.rs` stamp + `stamp.py` + teeth (gate ⑤); gate ① `tests/sim_bridge_parity_test.rs` (4,545 frames byte-equal over 34 logs; teeth: `decision_tense` moves 14/97 rows) | LANDED | `c803dd9f` |
| 7 | gate ① over the TRAINING POOL (719 teams, 90,556 frames) and PROCEDURAL teams (200, 90,844 frames), both `slow` | LANDED | see git log (`M5 Lane 0 unit 7`) |
| 5+6 | gate ① over the LADDER corpus (commit tier routine: 9,198 frames; milestone tier `slow`: 121,463 frames over 717 logs, 800 teams); quarantined battles compared (sim_bridge must refuse the same one); gate ② with quarantines (bank identical at 1/3/6 threads); `tests/bench_test.rs` (ignored) | LANDED | see git log (`M5 Lane 0 units 5+6`) |

## Design decisions so far (mirrored in the program doc's Lane-0 paragraph)

- Episode inputs are CALLER-STAGED columns (`ep_team` indices into a team table declared at
  startup, `ep_seed`), read at RESET and at every auto-reset.
- Outputs beyond the prototype: `refused`, `episode`, `dec_n` (= `__OBS__` `n`), `turn`, `counters`.
- No in-Rust opponent in Lane 0 (Lane F); both sides' rows go to the caller.
- `OBS_DIM` in Rust is `pokesim::encoder::OBS_DIM` (never a literal); a front end reads it from the core.
- The spec's keys are part of the one table (`protocol.SPEC_KEYS` → `columns::SPEC_KEYS`).
- Refusal policy: every PORT error for one battle quarantines (kind `refusal`/`malformed`/`fault`,
  `engine` for an engine fatal); the env core's own invariants, caller errors and panics fail the
  batch and POISON the pool. Teams are validated by use at startup.
- API for the front ends: `Core::new(Spec)` → `freeze(ColAddrs)` → `dispatch(op, ColAddrs) -> status`,
  `last_error().json()`, `bank()`, `counters()`; `OwnedCols` for Rust-side harnesses;
  `inline_env(i)` (threads <= 1) exposes `log` / `prev_log` / `open_tokens` for gate ①.

## Next

- Lane 0's five gates are built and routine; gate ① covers all three team sources (ladder commit
  tier routine; ladder milestone, training pool and procedural as `slow` tests, verdicts in
  `designs/ops/slow_tier_status.json`). What is left of the lane's 2.5–3.5 agent-days:
  (b) a quiet-box benchmark of the core (the only reading was on a loaded box);
  (c) review of the dispatch API by Lanes A/B once they start (their first use is the real test of
  the contract). Lanes A (FFI) and B (process) can start NOW: they forward to
  `Core::new(Spec::from_json)` → `freeze(ColAddrs)` → `dispatch(op, ColAddrs)`, allocate / map
  the columns from `columns.py` (`allocate`, `nbytes`, `shapes`), decode failures with
  `protocol.error_from_json`, compare `columns::SCHEMA_ID` with `columns.schema_id()` and call
  `stamp.check_stamp(core STAMP, …, nan_poison=…)` BEFORE the first op.

## (history) the unit-3/4 plan as written before they landed

- Unit 3: `build.rs` writes `STAMP` (commit informational; the refusal key = FNV-1a-64 over the
  `(path, git-blob-id)` listing of every `.rs` under `src/rust_sim/src` + `src/rust_env/src`, both
  `Cargo.toml`s, `build.rs`, and `src/utils/rust_env/{columns,protocol}.py`; plus profile + the
  selfcheck feature). `stamp.py` recomputes it and raises `StampMismatch`; teeth test on a temp copy.
- Unit 4: gate ① — a Rust test running the core (T = 1, both sides' rows) over corpus battles, then
  replaying each finished episode's `prev_log` through the worktree's selfcheck `sim_bridge` with
  `core_obs` both sides, comparing every `__OBS__` frame (b64 → bytes), mask and tokens to the
  core's row at the same `(episode, side, n)`. Needs the port's `sim_bridge` selfcheck binary built
  in THIS checkout (`src/rust_sim/target/selfcheck/sim_bridge`).

## Open findings

- **F-L0-1:** the port types the HP tracker's "all candidates eliminated" (F-M5-1) as
  `CoreError::Fault`, while Python raises `ValueError` on the same input — so it carries no Python
  class. Not changed here (a port file). The env core quarantines it anyway (see the policy).
- **F-L0-2:** an episode that ENDS naturally and whose NEXT start is refused reads `refused = 1`,
  `reward = 0` — the ended episode's reward is lost in that (start-refusal) case. Starts are
  validated by use at startup, so only a seed-specific construction refusal can reach it. Lane D
  owns end-of-episode semantics. **CLOSED by Lane D** (`m5_laneD/PROGRESS.md`).
- **F-L0-4 (closed, unit 5):** gate ① now runs the ladder corpus (commit tier routine, milestone
  tier slow), the training pool (slow) and procedural teams (slow).
- **F-L0-5 (closed, unit 6):** gate ② now exercises quarantines across thread counts.
- **F-L0-6:** the placeholder turn limit is checked AFTER the op's feeds, so the decisions that
  write opened are encoded and then discarded by the forfeit (10 frames in the milestone run) —
  wasted encode work and a Lane-D decision (check at the decision boundary instead, and which side
  forfeits: `StallConfig().threshold` semantics). **CLOSED by Lane D** (`m5_laneD/PROGRESS.md`).
- **F-L0-7:** the only cost reading (23.5–23.9 µs per row at N = 48, T = 8, release) was taken at
  load1 19–32 with a production run on the box; T = 1 read 126 vs 252 µs in two consecutive
  blocks. UNVERIFIED as a cost; F-M5-4 (in- vs out-of-process) needs the front ends.
- **F-L0-3:** heap ALLOCATIONS after freeze are not counted: the engine and the parse chains
  build protocol lines and persistent versions every step (the count is NOT measured), so an allocation counter cannot
  be a 0-gate. The `*_AFTER_FREEZE` counters cover threads, envs, column rebinds and bank growth.
