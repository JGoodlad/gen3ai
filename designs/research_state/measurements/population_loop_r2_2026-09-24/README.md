# Population loop round 2 — launch kit (2026-09-24)

The registration is [`../../population_loop_round2_2026-09-24.md`](../../population_loop_round2_2026-09-24.md).
**Nothing here was executed for real by the session that wrote it: no arm was launched, and nothing
was written under `models/`.**

## Contents

| path | what |
|---|---|
| `generalists/` | **VERBATIM copies** of the Training Run session's B2 / C2 kit from `/home/goodlad/.claude/jobs/popr2_2026-09-24/`: `argv_B2.txt` (sha256 `9014d843…`), `argv_C2.txt` (sha256 `9dce840c…`), `launch_B2.sh`, `launch_C2.sh`, `chain.sh`, and their `--dry-run` outputs. **B2** = `ai_v13_27_popr2_loop` (round-2 loop generalist; LAUNCHED 17:43 PT 09-24), **C2** = `ai_v13_28_popr2_ctrl` (round-2 no-exploiter control; queued behind B2). Copied so the registration does not depend on a scratch path (finding P-5) |

The reader kit (RB2 / RC2) and the convergence side-check kit (RB+ / RC+) follow in the next commit.
