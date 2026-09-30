# Battery reruns after the compile fix (2026-09-28)

These are the Training Run's launch records for the two arms added after the `--compile-trainer` miscompile was found. The fix landed at `6521f420`; the ledger entry is `bd44d222`.

| Run | Argv | Pin | Control | Outcome |
|---|---|---|---|---|
| **C_fix** (`ai_v14_06_lbat_ctrl_fix`) | `argv_Cfix.txt` | `6521f420` | plain C | Complete at 83,066,880. Its speed vs C: S = −1.50 % [−1.60, −1.35], the cost of the graph break. |
| **T32b** (`ai_v14_04b_lbat_t32`) | `argv_T32b_sentinel.txt` | `8fc297a2` | C_fix | STOPPED by the §4.3 futility look. It was stopped by SIGTERM at 20:29:50 PT and left `final_model_interrupted.zip` at 76,283,904. |

- **C_fix's argv** is `argv_C.txt` with only two tokens changed: `--pin-commit` and `--run-name`.
- **T32b's argv** is `argv_T32.txt` with the same two tokens changed.
- **The superseded T32b argv:** `argv_T32b_pre_sentinel_SUPERSEDED.txt` is the first T32b argv. It was never launched, because it was replaced by the compile-sentinel pin.
- **T32b's futility read:** `T32b_futility_read.txt`.
  - The live read used 12 rollouts: S = +2.48 % [+2.24, +2.91].
  - The exactly-11 recompute gives S = +2.49 % [+2.23, +2.93]. It used `speed_read_cap_for_T32b_recompute.py`, which is `speed_read.py` with the arm's rows truncated.
  - Both give **STOP**, and the orchestrator accepted it. Production matmul precision stays `highest`.
- **Other files:** the pre-launch checkargs and dry-run logs, and `rss_log_battery.txt` (steady-state RSS per life for every battery arm).
