# E1-0: the bot referee on banked bot-stratum picks (2026-10-25)

**Status: REGISTERED** ([`PREREGISTRATION.md`](PREREGISTRATION.md); file hashes in
[`e10_registration.json`](e10_registration.json)); the play is running. No result yet.

| file | what |
|---|---|
| `PREREGISTRATION.md` | the question, states, referees, statistic, rules, VOID guards, power, predictions |
| `play_e10.py` | the play side at X26's pin: every legal action × the 24 referee seeds, bot or self referee (`main.value_search.playout.referee`, loaded by file path from this checkout) |
| `run_e10.sh` | the CPU-rule wrapper (one thread, SCHED_IDLE, nice 19, agent cores, `mem_cap`, bounded chunks) |
| `score_e10.py` | the read: each experiment's own scorer with the referee's T substituted; `--selftest` on planted data |

Archive: `~/gen3ai_archive/search_e1_0/` (`rows/bot/`, `rows/self/`, `smoke/`).
