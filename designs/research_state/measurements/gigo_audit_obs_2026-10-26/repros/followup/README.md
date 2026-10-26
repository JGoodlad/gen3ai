# repros/followup: the audit's open items (2026-10-26)

Run from the repo root of a checkout or worktree (`export PYTHONPATH=$PWD/src`), CPU only.

| file | what |
|---|---|
| `run_targeted_set.sh`, `targeted_selfcheck.py`, `showdown_replay_selfcheck.py`, `showdown_replay_paced.py` | the B-C1 targeted lock / trap game set against Showdown on the self-check build; the PACED re-feed that tells a replay artifact (Node's reject-streak guard on a bulk feed of refused hidden-trap switches) from an engine divergence |
| `step_node.py` | feed a script to the Node bridge one command at a time and print what each produced |
| `targeted_summary_seed*.txt` | the fixed engine, seeds 1 and 2 |
| `smoke_stall.py`, `stall_smoke_before.out`, `stall_smoke_after.out` | B-C2 by execution: the `--debug` smoke at a 6-turn stall limit, collector rows against the `CHOOSE p1` lines the core fed (297 of 297 games one row too many before, 347 of 347 equal after) |
| `probe_bp_taunt_lockon.js` | A-F: a taunted mon cannot Baton Pass; a Baton-Passed `lockon` expires at that turn's residual |
| `oracle_census_dump.rs.txt`, `oracle_census_diff.py` | the `oracle_reveal_test` census: drop the `.rs.txt` into a scratch copy of `src/rust_env/tests/` as `census_dump.rs`, run `CENSUS_OUT=<dir> cargo test --profile selfcheck --features emission-selfcheck --test census_dump` at two trees, then `oracle_census_diff.py <dirA> <dirB>` |

The Showdown probes that pin the engine fixes are `src/rust_sim/harness/probe_lockin_cancelled_repeat.js` and
`probe_recharge_expiry.js`; the standing guards are `gen_sim_bridge_diff.js --mode lockin` and
`src/utils/rust_env/lockin_parity_integration_test.py`.
