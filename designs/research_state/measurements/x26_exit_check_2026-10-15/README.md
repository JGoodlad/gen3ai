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
