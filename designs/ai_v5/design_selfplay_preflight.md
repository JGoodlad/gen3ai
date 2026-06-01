# Design: Self-Play Pre-Flight

Step 1 self-play code has **landed** (`snapshot_pool.py`, `selfplay_callback.py`, `--self-play`)
but is gated behind the ai_v4 pathology-hunting phase. This document defines the concrete
**exit criterion** for that gate and the **flip checklist** for turning `--self-play` on — the
two operational questions that are currently written down nowhere.

## Exit criterion — when is pathology hunting "done"?

The current fixed-bot run exists to find and fix pathologies via eval-replay analysis before
the agent starts shaping its own curriculum. Flip `--self-play` on only when **all three** hold:

1. **No recurring pathology.** Eval-replay analysis surfaces no repeated degenerate behaviour
   (e.g. needless switch loops, set-up into a wall, sleep-talk misuse) across the last N eval
   cycles. This is the qualitative gate and the reason the phase exists — self-play will
   *amplify* any pathology it inherits, because the agent trains against its own habits.
2. **Fixed-bot ceiling reached.** `eval/win_rate_vs_bots` has plateaued (no improvement over
   ≥ 10M steps) around the heuristic ceiling (~80–85% vs Heuristic). Past this point the fixed
   bots provide no further gradient — exactly the condition self-play is designed to break.
3. **Entropy healthy.** `train/entropy` / `ent_coef` is stable, not climbing to fight collapse
   (the v3 failure mode: ent_coef drifting 0.029 → 0.055 while win rate stalled). Entering
   self-play with a collapsing policy bakes the collapse into the snapshot pool.

Conditions 2 and 3 are readable straight off TensorBoard; condition 1 is the judgement call
that gates the flip.

---

## Flip checklist

Once the exit criterion is met:

- [ ] **Smoke test self-play wiring.**
      `--debug --self-play --steps 20000`. Confirm: `snapshot_000000000000.zip` is seeded;
      one eval cycle runs; `eval/win_rate_vs_pool` and `eval/sentinel_monotonicity` are logged;
      `train/selfplay_fraction` is present.
- [ ] **Seed from the best fixed-bot checkpoint**, not a random or latest one. The step-0 seed
      is pinned forever (`SnapshotPool.seed`, never evicted) — it should be the strongest
      fixed-bot policy so the permanent floor of the pool is competent.
- [ ] **Set pool knobs.** Decide `--promote-threshold` (default 0.65), `recency_weight`
      (default 0.3), `max_snapshots` (default 20). If adopting the two-pool stable
      (`impl_step2_league_play.md`), decide whether it lands before the first self-play run or
      after the pool has churned once.
- [ ] **Decide reward-annealing timing.** Recommended: start late-self-play annealing in the
      same run (`--reward-anneal-start` when `win_rate_vs_bots` first plateaus, `--reward-anneal-end`
      before league). See `design_reward_annealing.md`. If deferring, record that the value head
      is still shaped and league must wait.
- [ ] **Confirm `--self-play` survives restarts.** The launcher forwards it verbatim; verify the
      resumed child re-reads the pool directory (`_scan`) and `win_rate_vs_bots.txt` so the
      heuristic-fraction curriculum resumes at the right point.
- [ ] **Curriculum sanity (first ~5M steps).** `train/selfplay_fraction` should start low
      (heavy heuristics) and rise as `eval/win_rate_vs_bots` climbs above 0.50. If it jumps to
      full self-play immediately, the persisted win rate was stale — check `win_rate_vs_bots.txt`.
- [ ] **Monotonicity watch.** `eval/sentinel_monotonicity` should stay ≥ 0.6. A sustained drop
      is the cycling signal — lower `recency_weight` toward 0 (more uniform sampling).

---

## What this is *not* waiting on

- **League tooling** (`design_league_tooling.md`) is **not** a prerequisite for self-play —
  Step 1 only needs the pool, sentinel monotonicity, and the bot-regression guard, all of which
  are built. The payoff-matrix runner and Nash/RPP metrics are Step 2 (league) prerequisites.
- **The two-pool stable** and **hardened promotion** (`impl_step2_league_play.md`) are
  improvements, not blockers — Step 1's single recency pool is sufficient to start. Adopt them
  before the pool diversity becomes the limiting factor.

---

## First-run watchlist (post-flip)

| Signal | Healthy | Action if not |
|--------|---------|---------------|
| `train/selfplay_fraction` | rises with `win_rate_vs_bots` | check persisted win rate |
| `eval/sentinel_monotonicity` | ≥ 0.6 | lower `recency_weight` |
| `eval/win_rate_vs_bots` | holds ≥ ~0.80 (no regression) | `⚠️ BOT_REGRESSION` fires — inspect replays |
| `eval/win_rate_vs_pool` | hovers near `promote_threshold` | persistent ≪ 0.5 ⇒ pool too hard, lower threshold |
| `train/entropy` | stable | collapsing ⇒ raise `ent_coef` |
