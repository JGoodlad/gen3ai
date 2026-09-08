# Benchmark baselines — the measurement narrative

Lifted verbatim from the root `CLAUDE.md` § *Benchmarks* on **2026-09-07**, when that section was
cut from 85 lines to ~44. The RULES (run the obs gate; `--pin-battles` for any A/B; a memoized
value is billed to whoever asks first) stayed in `CLAUDE.md`. What follows is the evidence: the
per-stage share tables, the "reward is NOT negligible" correction to a baseline this file carried
for generations, the shaped-vs-terminal composition cells, and their load conditions.

Every number here carries a date. Re-measure rather than quote.

---

### Benchmarks (`*_benchmark.py`, run directly as scripts)
Profile a hot path on a real bridge battle (no server). `obs_build_benchmark.py` plays until a
representative late-game decision, then reports a component wall-clock breakdown
(`state_encoder.encode` vs deque-cached turn-history vs `live_view()`) plus a `cProfile`
`tottime` ranking — use it to catch obs-pipeline regressions and confirm an optimization moved
the bottleneck:
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/obs_build_benchmark.py [--turn 25] [--reps 400] [--top 22] [--battles 200] [--seed 0]
```
Absolute ms scale with machine load; the component **ratios** and the cProfile ranking are the
load-stable signal — run on an otherwise-idle box for a clean baseline. **You no longer have to
remember**: every benchmark calls `warn_if_contended()` at entry and prints a loud "THE BOX IS
BUSY" banner with the load average when the box is not idle (see [Running beside a live training
run](#running-beside-a-live-training-run-gen3_contention_robust_timeouts_v1)).

**Every change under `src/agents/observation/` must run this benchmark before/after and
confirm no meaningful regression** — that gate, the canonical baseline, and the
load-stable regression criteria live in `src/agents/observation/CLAUDE.md`.

For the **top-down** view — where a whole trainer turn's CPU goes (parse + obs + reward +
mask + map + tracker), GPU-excluded and server-free — use `trainer_turn_benchmark.py`. It
walks a real bridge battle, times every per-decision CPU stage `Gen3Env` runs (a random legal
action stands in for the policy forward), and prints a stacked breakdown. Baseline (re-measured
2026-08-23, idle box, no training run): **our controllable CPU ≈ 0.90 ms/decision — obs build 63%
(`state_encoder.encode` 33%), reward 27%, parse 9%**, everything else ≈1%.

⚠️ **The reward stage is NOT negligible, and this doc said it was for generations.** The old
baseline here read "obs ≈ 88%, parse ≈ 7%, reward ≈ 4%" and matched no arm measured in 2026-08:
reward has been **23–27%** of our CPU since at least the pre-frame-deletion tree, and
`process_turn_reward` alone is **0.21 ms — the second-largest per-decision consumer**. Anyone
optimizing this path off the old line would have skipped it. (The frame deletion is *not* the
explanation: reward was already 23% before it. At reward's measured 0.23 ms absolute, a 4% share
would need our-CPU near 5.75 ms — six times anything observed.) Record:
`designs/research_state/measurements/post_paydown_baselines_2026-08-23.{json,md}`.

⚠️ **A memoized value is billed to whichever stage asks for it FIRST — and that made one line of
this breakdown mean the opposite of what it said.** `battle.live_view()` memoizes per state-epoch
and FIVE stages read it, so the whole 12-mon board build was charged to `obs: legal + mask`, which
consequently reported **22% of worker CPU** and was named as the next optimization target on that
basis. Its own work (parse the request, set 11 bits, run two integrity checks) measures **0.028 ms
— 2.8%**; the other 88% was the shared build. `trainer_turn_benchmark` now times
`obs: live_view (shared build)` on its own line, and post-fix the obs shares read **live_view 17% ·
`state_encoder.encode` 26% · progress-clock 10% · `tracker.record` 6% · legal+mask 2.8%**. *When a
stage looks expensive, check whether it is merely first.* Detail — and the sized, deliberately
un-built next item (per-mon reuse, ~13–16% of worker CPU) — is in `src/agents/battle/CLAUDE.md`.

**The reward stage's cost depends on the REWARD COMPOSITION, so the baseline above is the SHAPED
one and there is now a second.** `--reward-argv '<train_rl_agent flags>'` times any composition,
built through the launcher's own `build_parser` + `RewardConfig.from_args`, and the resolved
census is printed with the run header — so an arm cannot be timed under a composition no launch
produces. Under **TERMINAL-only** (`--no-hand-shaping --terminal-indicator --victory-value 1.0`,
the win-prob arm) `process_turn_reward` is **0.021–0.025 ms — 2.2–2.7% of our CPU**, against the
shaped composition's **0.169 ms / 16%** in the same session; the `reward` GROUP falls to **8%**
and is then mostly `build_delta`, the TurnDelta fold, which every composition keeps. Provenance:
2026-09-06, `--decisions 400 --seed 0 --pin-battles`, load average **41–45 on 16 cores** (a live
arm was on the box — treat the ABSOLUTES as inflated and read the same-session shaped-vs-terminal
ratio, which is what the pinning makes comparable). Before the terminal-only short circuit
(`gen3_terminal_only_short_circuit_v1`) the same cell read **0.150–0.164 ms**, i.e. ~64% of the
shaped cost where it is now ~14%.

🚨 **`--pin-battles` is REQUIRED for any before/after or arm-vs-arm claim, and that is a measured
requirement rather than a caution.** Unpinned, each invocation walks a fresh RANDOM battle — the
`--seed` fixes the team draw and the action picks but NOT the sim dice — so two runs profile two
different boards. Measured 2026-09-06, three back-to-back 400-decision runs: the SHAPED arm read
**0.145 ms and 0.102 ms** while the TERMINAL-only arm between them read **0.111 ms** — the
run-to-run spread was LARGER than the effect and carried the wrong SIGN. Pinned, all four cells
land on exactly 468 decisions / 6 battles and the comparison is paired. Same trap
`live_view_build_benchmark` exists to avoid; the flag is off by default so the headline share
table still samples the board distribution.
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/trainer_turn_benchmark.py [--decisions 150] [--warmup 3] [--seed 0] [--pin-battles] [--reward-argv '…']
```

**Neither benchmark above can measure a CHANGE to `LiveView.from_battle`** — the single largest
item in that breakdown at 17% — because both walk a *fresh random battle* per invocation, so two
consecutive runs profile two different boards (a 12-mon turn-40 board reads faster than a 12-mon
turn-65 one on a strictly faster tree; that mistake was made and briefly believed). Use
`live_view_build_benchmark.py`: it captures ONE seeded board, freezes it, and A/Bs the live
implementation against a verbatim copy of the previous one with the arm order alternated, after
asserting the two build field-identical mons. It reports both the wall ratio and a **load-free**
`sys.setprofile` call count per build.
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/live_view_build_benchmark.py [--reps 2500] [--rounds 6] [--turn 12] [--profile]
```
