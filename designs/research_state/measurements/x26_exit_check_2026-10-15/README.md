# X26 exit check, 2026-10-15 (at `87e493c3`)

This folder holds the evidence behind `designs/endstate/design_x26_readiness.md` §4 (the status column), §4.1 (the
launch plan) and §4.2 (the bf16 headroom watch). Everything here was read on CPU under rule 5b, with no GPU: `nice 19`,
CPUs 5–7 and 13–15, `OMP_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=""`. The one exception is `gpu_t36_87e493c3/`, which
COPIES the Training session's GPU window outputs from `~/.cache/gen3ai/`.

| file | what it is |
|---|---|
| `e10.sh`, `e10_checkargs_87e493c3.log`, `e10_dryrun_87e493c3.log` | E10: `main.checkargs --argv` and `main.launcher --dry-run` on X26's argv (Amendment 2 §A2.2) at `87e493c3`, run from an agent worktree with `GEN3AI_MODELS_DIR` a scratch dir. checkargs exits 0; the dry run says "would launch". Re-run both at P_X26 from the MAIN checkout |
| `e9.sh`, `e9_tests_87e493c3.log` | E9: Amendment 2's two launch-gate tests (the bf16 heads ON/OFF bit-identity test and the K9(b) every-update record). 2 passed |
| `sanity_rule.py` | **E7's coded decision rule, committed BEFORE any `rb_xs_*` seed.** `self-check` passes on 7 constructed cases; run `decide <in.json>` on the 2 × 8 matrix plus the run gates |
| `canary_shift_extract.py`, `canary_shift_87e493c3.json` | the compiled bf16 arm's log-prob shift per toy checkpoint (taken action vs all legal actions), extracted from the T36 GPU diagnostic's `gpu_*.json` (pure read) |
| `gpu_t36_87e493c3/` | the T36 CUDA confirmation: `verify_compile_gate_cuda_summary.json` (PASS 4/4), and the diag's `VERDICT.md` (its checklist UNFILLED; the criteria are re-read in readiness I30), `gpu_meta.json` and `gpu_status.json`. The full `gpu_*.json` (~5 MB) stay in `~/.cache/gen3ai/diag_bf16_canary/87e493c35e6a/` |

The canary reader `jq` line in §4.2 was validated on records written by the real writer (`CompileCanary._record`,
`GradWatch.record`, `clip_relocation`). The synthetic script stayed in scratch and is not committed.

## The road to the X26 launch, as run (2026-10-15 / 16; the Training session's records, copied unedited into `road/`)

- **P_X26 = `9c033489`** (the memfix). The first P_X26, `71351ff5`, refused the first sanity seed at startup: UpdateWontFit, headroom 978 < 1,024 MiB. Its dir is `models/rb_xs_s2701_void_updatefit`. `x26-memfix` fixed it, and the headroom at `9c033489` is 2,444 to 2,468 MiB.
- **W1 at `9c033489`:**
  - `verify_compile_gate_cuda.sh` PASS 4/4 (`road/w1/verify_compile_gate_summary_9c033489.json`).
  - `compiled_first_update_test … on_cuda` PASS.
  - The five slow-tier rows recorded at `9c033489` are in `road/w1/slow_tier_rows_9c033489.json`.
- **W2 / W3:** `rb_xs_s2701` (untyped) and `rb_xs_s2702` (`--canary-shift-max 0.18` typed) both ran fresh `--arch production` to 5M, rc 0.
  - **T37 (a):** s2702's startup check and canary records carry `[--canary-shift-max]`.
  - **T37 (b):** a resume of s2701 with `--canary-shift-max 0.25` passes checkargs and the dry-run (`road/t37b_*.txt`).
- **E6:**
  - The scripted read was OUTSIDE: fps +13.6 % against the registered 1,631, train_s −0.2 %.
  - **ORCHESTRATOR RULING: PASS** on the like-for-like read. Over updates 10–50, from `launcher_child.full.log`, s2701 runs 42.00 s at 1,882 fps. `rb_bf_eb_s2001` runs 42.10 s at 1,865 fps, and `s2005` 42.10 s at 1,841 fps.
  - The registered 1,631 is EB's late-window figure. The original OUTSIDE line and the corrected line are both in `road/status.txt`.
  - **FINDING:** `e6_read.py` read the trimmed log.
- **W4** (cells at `9c033489`; every reference loaded first; `road/w4/`):
  - **E7 (`sanity_rule.decide`): PASS.** Pooled −0.44 pp; s2701 −0.32, s2702 −0.56. The run gates are clean.
  - **R-T1** (toy @15M vs EB 15M, 500 pairs; descriptive here): the per-seed means are 33.89, 35.84 and 37.98.
  - **R-T1b** (toy @5M vs EB 5M, 250 pairs): the per-seed means are 24.88, 23.14 and 25.45.
  - The matrices are in `road/w4/result/w4_read.json`.
  - **FINDING:** the first read used `matrix_from_cells`' default `min_pairs` of 1,000 against the registered 500 / 250. That made E7 INCONCLUSIVE and HELD the chain. The read was corrected per cross, with no new games. The bugged read is kept: `road/read_w4_v1_minpairs_bug.py` and `road/w4/result_v1_minpairs_bug/`.
- **E10:** checkargs and the dry-run of the X26 argv at `9c033489` both pass (`road/e10_*.txt`). The `data/` freeze (E11) was announced at launch.
- **X26 `rb_x26_s2601` launched 2026-10-16 03:08:59.** Startup parity PASS. UpdateFit headroom 2,468 MiB. canary@10 PASS: reserved 8,326 MiB, compiled max |Δ log π| 0.0033 (GREEN). The 1M config matches the production mirror.
- `road/FINDINGS.md` lists every finding. `road/alert_watch.sh` is the race-free alert watch, a class fix for a HOLD line that was read 13 minutes late.
