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
(cd src/rust_env && CARGO_TARGET_DIR=$PWD/target cargo test)
```

## Units

| # | unit | status | commit |
|---|---|---|---|
| 1 | crate skeleton + the GENERATED column / op / status / counter contract + its routine pin (gate ③) | LANDED | see git log (`M5 Lane 0 unit 1`) |
| 2 | `spec.rs` (startup declaration, JSON-parsable for Lane B) + `pool.rs` (N envs, T workers, bind → freeze) + `dispatch.rs`; determinism tests (gate ②) | NEXT | |
| 3 | `refusal.rs` (quarantine, banked input log as a sim_bridge script, typed class, budget) + `build.rs` stamp + `stamp.py` (gates ④ ⑤) | | |
| 4 | gate ①: the core's rows vs the worktree's `sim_bridge` `__OBS__` (core_obs both sides) on the banked input logs | | |

## Design decisions so far (mirrored in the program doc's Lane-0 paragraph)

- Episode inputs are CALLER-STAGED columns (`ep_team` indices into a team table declared at
  startup, `ep_seed`), read at RESET and at every auto-reset.
- Outputs beyond the prototype: `refused`, `episode`, `dec_n` (= `__OBS__` `n`), `turn`, `counters`.
- No in-Rust opponent in Lane 0 (Lane F); both sides' rows go to the caller.
- `OBS_DIM` in Rust is `pokesim::encoder::OBS_DIM` (never a literal); a front end reads it from the core.

## Open findings

- (none yet)
