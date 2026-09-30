# G0′ continuation blocks 2 and 3, on the fixed learner (2026-09-28/29)

Every block is a +8,060,928-step frozen fork at `--fork-lr 2.8e-05`, with 10 epochs, fp32 precision and pin `a5b2fdba`.

**Block 1 is C_fix** (`ai_v14_06_lbat_ctrl_fix`), which replaces the buggy-compile C. This is a deviation, and the orchestrator recorded it in the ledger.

| Block | Run | Parent | Final step |
|---|---|---|---|
| K2 | `ai_v14_07_g0p_k2` | C_fix final | 91,127,808 |
| K3 | `ai_v14_08_g0p_k3` | K2 final | 99,188,736 |

- **K2 is G0′**, the plateau parent the population loop forks. K3 is a spare.
- **K3 was launched unattended by `chain_k3.sh`.** The script checks, in order: K2 finished, a HOLD file, a `data/` diff, the precise tenant check (waiting up to 90 min), and a dry-run that must resolve to a FORK at +8,000,000. `chain_status.txt` is its log.
- **Both blocks ended with the COMPILE LOCK released line**: `released … 0 compile(s) after the lock`.
- **The compile counters stayed at zero**: `compile/recompiles_after_lock` = 0 and `cache_limit_hits` = 0, in every life of both blocks.
- **`rss_log.txt`** holds steady-state RSS at each interval restart.
