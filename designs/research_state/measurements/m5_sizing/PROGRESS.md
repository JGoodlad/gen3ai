# M5 SIZING STUDY — PROGRESS (resume point)

Registration: [`REGISTRATION.md`](REGISTRATION.md) (order constraint 5 of
`designs/endstate/program_rust_core.md`). Worktree `/home/goodlad/dev/gen3ai-wt/m5-sizing`, branch
`m5-sizing`. Lane J owns it (`src/main/rust_core_m5/` + this directory).

## Units

| # | unit | status |
|---|---|---|
| 1 | harness extension: `fanout.py --real-flushes` (the trainee / opponent split on REAL flushes) + `--t2-buckets` | SHIPPED `a73a8feb` |
| 1b | F-SZ-1 fix: the battery's untaught unit driver on main + its test | SHIPPED `bfb8e7e2` |
| 2 | the registration + its scripts (`bucket_rule.py`, `part_t.sh`, `pps_check.py`) | REGISTERED (this commit) |
| 3 | Part T: (T-c) update cost; (T-b)/(T-a) at N = 48, 128, 256, 512, 1024, 2048; (T-a′) at N\* | NOT STARTED |
| 4 | Part L: pre-flight at each arm's N; arms A, B, C, A′; U + G-A reads | NOT STARTED |
| 5 | verdict: Decision records, endstate docs, `production_config.json` iff a production value changes, ledger | NOT STARTED |

## How to run

```bash
K=designs/research_state/measurements/m5_sizing
bash $K/scripts/part_t.sh update          # (T-c), once
bash $K/scripts/part_t.sh n 256           # (T-b) then (T-a) at one N; resumable via ~/.cache/gen3ai/tmp/sizing/part_t/status
bash $K/scripts/part_t.sh nstar <N*>      # (T-a') fewer active snapshots at N*
python3 $K/scripts/bucket_rule.py rule $K/results/split_n256.json
python3 $K/scripts/pps_check.py --out $K/results/pps_check.json
```

## Pre-registration reads (declared in the registration's header)

- Untaught-8 operating range (run with this study's then-copy of the driver, the same change as
  `bfb8e7e2`), 25 games / team, opponent N0 @24M: N0 @4.8M **24.0 %** (48/200);
  N0 @9.48M **45.0 %** (90/200). Rows in `~/.cache/gen3ai/tmp/sizing/pilot/rows/` (scratch).
- SmallRL guard operating range: N0 @9.48M, away, greedy, 100 games: **38 / 100** (regime verified).
- N = 256 instrument smoke of the throughput harness. It ran end to end; T2 startup at 4 buckets was
  327 s. Its throughput number is discarded.
- Noise-scale history of N0 / C_fix / K2 (REGISTRATION §6).

## Findings

- **F-SZ-1 (FIXED on main, `bfb8e7e2`):** the battery's `n0_endofrun_2026-09-27/scripts/gu_unit.py`
  called `untaught_meter._strip_debugger`, which `ecd2be00` deleted along with the
  ObservationDebugger, so every untaught unit died at load. The driver now uses the models as loaded,
  and `src/main/untaught_unit_script_test.py` pins it (routine: a static symbol check with teeth;
  `sim`: one real battle). Ten older measurement scripts under `designs/research_state/measurements/`
  (`arch_transfer_2026-09-05/*`, `teacher_content_2x2_2026-09-04/*`, `ext_first_team_audit`) still
  call it. They are era-pinned history, not live drivers, and are left as they are.
- **F-SZ-2 (a trap for any fork at N > 48, UNVERIFIED as a failure):** on a resume, SB3 restores the
  checkpoint's n_steps (2,048). `load(…, env=…)` then builds a rollout buffer of n_steps × N rows
  before `RustCollector._ensure_buffer` resizes it to target / N: 4.2M rows (≈ 46 GB of obs) at
  N = 2048. `np.zeros` is lazily paged, so it may never be resident. Not exercised here (the arms are
  FRESH), and not tested.
- **F-SZ-3:** on the Rust core one SB3 "vec-env call" is N trainee decisions. So the default
  checkpoint cadence (50,000 vec calls) is 50,000 × N steps: 102M at N = 2048, i.e. never in an 8M arm.
  The arms type `--checkpoint-every-steps 2000000`. The production-default fix is routed to another
  agent by the orchestrator.
