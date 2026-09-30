# Population-loop round 0 on the new lineage — PREPARED, then DEFERRED (2026-09-29)

Round 0 is two offense exploiters of G0′ (`ai_v14_07_g0p_k2/final_model.zip` @ 91,127,808). They use arm A's recorded `original_command` (from `ai_v13_18_teach5_offense_hidose`), changed only as registered (`new_lineage_2026-09-26.md` §5.1).

**The registered changes:**
- `--run-name`
- `--model` and `--exploiter`, both set to K2's final zip, named explicitly
- `--steps 99127808`
- `--pin-commit 13cc85ec`
- `--eval-battles 200`

**One forced change:** the six flags deleted at `e3ef16db` are dropped. All six are inert under the winprob critic, and the battery's argvs dropped them the same way.

| Arm | Run | Seed | Status |
|---|---|---|---|
| A′ | `ai_v14_09_r0_offense_a` | 1001 | Launched 15:53:36 PT. PAUSED by SIGTERM at 16:36:54 for an M5 GPU window, at step 92,110,848; `final_model_interrupted.zip` was kept. |
| A2′ | `ai_v14_10_r0_offense_a2` | 1002 | Never launched. |

**Deferred.** At about 18:10 PT the owner decided: no GPU training until M5 fully lands and training moves to the new infrastructure.

**`argv_<run>_RESUME.txt`** is the launch argv with `--model models/<run>/final_model_interrupted.zip`. A′'s resume dry-run returned 0: RESTART at 92,110,848 → +7,016,960.

🚨 **These argvs pin `13cc85ec` on `gen3ai_stable` / torch 2.5.1.** If M5 moves training to a new env or torch, rebuild them rather than resuming as-is.

**Other files:**
- `launch_one.sh` — the gap-waiting launcher: not-before time, empty GPU, precise tenant check, `data/` diff, dry-run.
- `chain_status.txt` — its log.
