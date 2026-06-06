# AI v5 — Todo

Self-play / league play. The agent trains against frozen copies of itself (Step 1, **landed**),
then against a structured league with exploiters (Step 2, **designed**). Reward annealing,
league tooling, and a self-play pre-flight checklist are the supporting design docs.

| Doc | State |
|-----|-------|
| `impl_step1_self_play.md` | ✓ built (code landed, not yet run — gated behind ai_v4 pathology hunting) |
| `design_selfplay_preflight.md` | exit criterion + flip checklist for turning `--self-play` on |
| `design_reward_annealing.md` | league prerequisite — three-tier shaping anneal |
| `impl_step2_league_play.md` | forward design — two-pool stable, PFSP, Nash/RPP progress |
| `design_league_tooling.md` | payoff-matrix runner, Nash/RPP metrics, inspector, descriptors |
| `impl_step3_elo_skill_rating.md` | ✓ built & shipped — anchored Bradley-Terry skill rating (the absolute progress signal pool win-rate can't give) |
| `impl_step4_incoming_damage_obs.md` | ✓ built & shipped — incoming-damage / OHKO belief obs block (`gen3_incoming_damage_v1`); the critic-tail-blindness pathology fix (Gate-2 efficacy pending a retrain). See `todo_pathologies.md` row F + `design_pathologies.md`. |

---

## Step 1 — Self-Play ✓ DONE (code) — not yet run

Train the agent against frozen copies of itself rather than against fixed heuristics.
Introduces a snapshot pool, win-rate gating, heuristic fraction curriculum, and sentinel
monotonicity to prevent strategy collapse. See `designs/ai_v5/impl_step1_self_play.md`.
**Flipping it on** is gated by the pathology-hunting phase — exit criterion and checklist in
`designs/ai_v5/design_selfplay_preflight.md`.

### Deferred / Deliberately Not Done

**Hot-swap opponents mid-run** (now a Step 2 prerequisite)
Currently, pool opponents refresh only at launcher restarts (~every 2.5h). A
`_staged_opponent_path` mechanism in `Gen3Env.reset()` would allow swapping opponents between
episodes without a restart. Optional for Step 1; **required** for league play Option A (the
exploiter/main alternation needs per-episode opponent swaps) — see
`designs/ai_v5/impl_step2_league_play.md` → Training Coordination.

**ELO tracking** (resolved — Nash + Glicko-2, not plain ELO)
For Step 1, `win_rate_vs_bots` (a *fixed* reference) is the progress signal — no per-snapshot
state needed. For the league era, the opponent pool is non-stationary, so progress is measured
by Nash **relative population performance** + `win_rate_vs_bots`, with **Glicko-2** (not plain
ELO) as optional non-gating human-readable sugar. Plain ELO is rejected: it misleads under the
non-transitive dynamics league play creates. See `designs/ai_v5/impl_step2_league_play.md`
(Progress & diversity measurement) and `design_league_tooling.md` (metrics module).

**Reward annealing** (designed — needed before league play)
Now specified in `designs/ai_v5/design_reward_annealing.md`: `--reward-anneal-start` /
`--reward-anneal-end` flags drive a three-tier anneal — strategic priors (attack/switch/field
shaping) → 0, outcome proxies (HP/faint/win_loss) kept, anti-degenerate taxes floored. Two
reasons it matters: (1) prevents reward-hacking shaping at the expense of winning; (2) the value
head must estimate win probability, not shaped reward, for MCTS in v6. Build before league play.
Trigger: `eval/win_rate_vs_bots` flat for ≥ 10M steps.

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
generalise past those exploits. See `designs/ai_v5/impl_step2_league_play.md`.

**Prerequisites before starting Step 2:**
- `eval/win_rate_vs_bots` flat for ≥ 10M steps (self-play curriculum saturated)
- `eval/sentinel_monotonicity` ≥ 0.6 (pool is not cycling)
- Reward annealing at least 50% complete — `design_reward_annealing.md` (value head must
  learn win probability, not shaped reward, for MCTS in v6)
- Payoff-matrix runner + Nash/RPP metrics built — `design_league_tooling.md` (the league's
  diversity-alarm thresholds are meaningless without the matrix)
- Hot-swap opponent path in `Gen3Env` (Option A needs per-episode opponent swaps)

**New code (see the two design docs):**
- Two-pool `Stable` (recency + permanent), hardened promotion, PFSP sampler, exploiter
  manager, `league_callback`, `train_league.py`
- `payoff_matrix.py`, `league_metrics.py` (Nash/RPP/non-transitivity/Glicko-2),
  `behavioral_descriptors.py`, `league_report.py`
