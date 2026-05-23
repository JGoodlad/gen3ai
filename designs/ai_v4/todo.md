# AI v4 — Todo

---

## Step 1 — Self-Play ✓ DONE

Train the agent against frozen copies of itself rather than against fixed heuristics.
Introduces a snapshot pool, win-rate gating, heuristic fraction curriculum, and sentinel
monotonicity to prevent strategy collapse. See `designs/ai_v4/impl_step1_self_play.md`.

### Deferred / Deliberately Not Done

**Hot-swap opponents mid-run** (low priority)
Currently, pool opponents refresh only at launcher restarts (~every 2.5h). A
`_staged_opponent_path` mechanism in `Gen3Env.reset()` would allow the self-play callback
to swap opponents between episodes without a restart, giving the agent fresher competition
sooner after a new snapshot is promoted. Deferred until pool diversity becomes the
bottleneck.

**ELO tracking** (replaced — won't do in current form)
The original design specified per-snapshot ELO ratings, `elo_state.json`, and Glicko-style
updates. This was replaced by `win_rate_vs_bots` as the curriculum signal and
`eval/win_rate_vs_pool` for promotion gating. Win rate is simpler, requires no per-snapshot
state, and is directly interpretable. The ELO "plateau as completion signal" is replaced by
watching `eval/win_rate_vs_bots` flatten near 85%.

**Reward annealing** (deferred — needed before league play)
The design specified `--reward-anneal-start` / `--reward-anneal-end` flags to gradually
reduce shaping signals (switch subsidies, pivot bonuses, matchup penalties, etc.) toward
zero as the agent matures. Two reasons this matters: (1) prevents reward hacking of shaping
signals at the expense of winning; (2) the value head must estimate win probability, not
shaped reward, for MCTS in v5. Must be implemented before league play begins. Trigger:
`eval/win_rate_vs_bots` flat for ≥ 10M steps.

**Demotion threshold** (replaced by regression guard)
The original design had a hard stop if `win_rate_vs_pool` dropped below 40%. This was
replaced by the regression guard in `_check_bot_regression()`, which is softer
(edge-triggered warning rather than hard abort) and tracks per-opponent regression rather
than a single aggregate. Hard abort on pool regression was deemed too aggressive — short
dips during exploration are normal.

**Win-rate oscillation tracking** (low priority)
The design described tracking σ of `eval/win_rate_vs_pool` over the last 5 snapshot cycles
as a cycling signal. The `eval/sentinel_monotonicity` metric (Kendall's τ) covers the same
ground more directly. Add oscillation tracking only if monotonicity proves insufficient.

**`eval/win_rate_vs_oldest_sentinel`** (low priority)
The design described a dedicated "forgetting signal" tracking the oldest sentinel
specifically. Currently, all 5 sentinels contribute to `sentinel_monotonicity`, and a drop
in the oldest's win rate would show as reduced monotonicity. Add explicit oldest-sentinel
tracking if forgetting becomes a real failure mode.

---

## Step 2 — League Play

Extend self-play into a structured league with dedicated exploiter agents and prioritised
opponent sampling. Exploiters find weaknesses in the Main Agent; the Main Agent must then
generalise past those exploits. See `designs/ai_v4/impl_step2_league_play.md`.

**Prerequisites before starting Step 2:**
- `eval/win_rate_vs_bots` flat for ≥ 10M steps (self-play curriculum saturated)
- `eval/sentinel_monotonicity` ≥ 0.6 (pool is not cycling)
- Reward annealing at least 50% complete (value head needs to learn win probability,
  not shaped reward, for MCTS in v5)
