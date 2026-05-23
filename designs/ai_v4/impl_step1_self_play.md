# Implementation: Step 1 — Self-Play Training

This step replaces the fixed heuristic opponent with a pool of frozen snapshots of the
agent itself, with win-rate gating to control when a snapshot is promoted and ELO
tracking to measure real improvement over time.

## Motivation

Training against `MaxDamagePlayer` and `SimpleHeuristicsPlayer` has a hard ceiling.
Both opponents are deterministic in structure — the agent can learn to exploit their
specific patterns without developing general strategic understanding. Once it reaches
~80% win rate against heuristics, continued training against them produces diminishing
returns at best and policy collapse at worst.

Self-play addresses this by making the agent's current policy the curriculum: the frozen
opponent always reflects a snapshot of where the agent actually was, so every skill the
agent develops immediately raises the bar its frozen copy sets. The risk is the agent
specialising against its own biases — naive self-play can cycle rather than improve.
A frozen snapshot pool with win-rate gating prevents this by ensuring the opponent pool
retains diversity across the full training history.

---

## Design

### Snapshot Pool

A `SnapshotPool` manages a directory of `.zip` checkpoint files written during training.
Each file is named `snapshot_{global_step}.zip` and accompanied by its `model_config.json`
(already written by `save_model_snapshot()`).

```
models/<run_id>/snapshots/
  snapshot_000000000.zip     # initial checkpoint (from v3)
  snapshot_002000000.zip
  snapshot_005500000.zip
  ...
```

Pool behaviour:
- **Max size**: keep the last `max_snapshots` files (default 20); oldest are deleted when
  the pool overflows. The initial v3 checkpoint is pinned and never deleted.
- **Sampling distribution**: uniform over all pool members by default. An optional
  `recency_weight` (default 0.0, range 0–1) adds a linear tilt toward recent snapshots
  without fully excluding older ones.
- **Load on demand**: each snapshot is loaded into a `MaskablePPO` object only when
  selected as the opponent for the next episode batch. A simple LRU cache (size 3) avoids
  reloading the same checkpoint repeatedly.

### Win-Rate Gating and Snapshot Promotion

Every `eval_interval` training steps (default 500K), a `SelfPlayCallback` runs N
evaluation battles (default 200) between the live agent and a uniform sample from the
snapshot pool. If the live agent's win rate exceeds `promote_threshold` (default 65%),
the current checkpoint is written to the pool. If win rate is below `demotion_threshold`
(default 40%), training stops with a warning — the agent has regressed below the level
of its own historical copies.

The 65% threshold is deliberately conservative. Promoting too aggressively means the
pool fills with nearly-identical snapshots that provide no new signal. Promoting at 65%
ensures there is genuine divergence from the pool before a new snapshot is added.

### ELO Tracking

Each snapshot in the pool carries an ELO rating, updated after each evaluation batch
using the standard Glicko-style update:

```python
expected = 1 / (1 + 10 ** ((elo_opponent - elo_live) / 400))
elo_live += K * (actual_win_rate - expected)
```

K = 32 during the rapid-improvement phase, reduced to 16 once the live agent's ELO
exceeds 1800. ELOs are persisted in `snapshots/elo_state.json` alongside the zip files
so they survive restarts.

TensorBoard logs: `eval/elo_live`, `eval/win_rate_vs_pool`, `eval/pool_size`.

### Hot-Swapping the Frozen Opponent

The current training setup creates the opponent `Gen3EnvPlayer` once at startup and does
not replace it mid-run. To support self-play, the opponent player needs to be
hot-swappable between episode batches.

The cleanest insertion point is `Gen3Env.reset()`. Before calling `super().reset()`, the
env checks whether a new opponent checkpoint has been staged by the callback:

```python
def reset(self, **kwargs):
    if self._staged_opponent_path is not None:
        self._load_frozen_opponent(self._staged_opponent_path)
        self._staged_opponent_path = None
    return super().reset(**kwargs)
```

`_load_frozen_opponent()` creates a new `RLPlayer` from the checkpoint path and replaces
`self._opponent` in place. The old player's WebSocket connection is closed before the new
one connects. This swap happens between episodes so the env state machine is never
interrupted mid-turn.

The callback stages a new path by writing to `env._staged_opponent_path` via a shared
ref, once per `opponent_swap_interval` episodes (default: every episode batch, i.e., each
call to `model.learn()`'s inner rollout collection).

### Warm-Start: Transitioning from Heuristics

At step 0, the snapshot pool is seeded with the v3 checkpoint (the best checkpoint from
heuristic training). The live agent begins self-play immediately against this seed. No
transition period is needed — the v3 agent beats heuristics reliably, so the seed snapshot
is already a meaningful opponent.

If the v3 checkpoint is unavailable or below 60% vs. MaxDamage, fall back to 50K warmup
steps against `SimpleHeuristicsPlayer` before seeding. This is controlled by
`--warmup-steps` (default 0).

### Entropy and Diversity

Self-play can reduce entropy as the agent converges to a narrow set of winning strategies
against its own distribution. Two mitigations:

1. **Entropy bonus**: Keep `ent_coef` at 0.02 (same as heuristic training). Monitor
   `train/entropy_loss` in TensorBoard — if it collapses below −0.5 nats, increase
   `ent_coef` to 0.05 for the next training segment.

2. **Team diversity**: Both sides draw from the full 770-team pool. The live agent's team
   is sampled fresh each episode; the frozen opponent's team is also sampled fresh (not
   pinned to a fixed team). This prevents the agent from specialising against specific team
   matchups.

### Cycling and Forgetting Detection

Early in self-play, the pool contains only the seed snapshot (v3 checkpoint) and behaves
as a single fixed opponent. As the pool grows, two failure modes can emerge: **cycling**
(the agent learns a strategy that beats the current snapshot but is exploited by older
ones, producing an A→B→C→A loop) and **forgetting** (the agent loses the ability to beat
older styles as it specialises against recent ones). Both are detectable from periodic
evaluation data before they become severe.

**Grandparent test** — keep 5 sentinel snapshots evenly spaced in training time and eval
against all of them every 3M steps. Healthy training produces monotonically decreasing
win rates as snapshots get older (old snapshots should be easier). Non-monotone ordering
signals a cycle.

**Monotonicity score** (Kendall's τ over the 5 win rates):

```python
def monotonicity_score(win_rates: list[float]) -> float:
    """win_rates[0] = most recent snapshot, win_rates[-1] = oldest.
    +1.0 = perfectly monotone (expected), −1.0 = fully inverted."""
    n = len(win_rates)
    concordant = sum(
        win_rates[i] <= win_rates[j]
        for i in range(n) for j in range(i + 1, n)
    )
    return 2 * concordant / (n * (n - 1)) - 1
```

Log as `eval/sentinel_monotonicity`. Below 0.6, the pool needs more diverse coverage
and `recency_weight` should be reduced toward 0 (uniform sampling) to force the agent to
keep solving older styles.

**Win-rate oscillation** — track the standard deviation of `eval/win_rate_vs_pool` over
the last 5 snapshot cycles. σ > 0.12 consistently indicates cycling; increase
`max_snapshots` and reduce `recency_weight`.

**Forgetting signal** — `eval/win_rate_vs_oldest_sentinel` should be stable or slowly
rising. A drop > 10% over 10M steps means the agent is forgetting. Reduce snapshot
deletion rate (raise `max_snapshots`) or pin the oldest sentinel as a permanent pool
member.

The pool size (default 20, covering roughly the last 10–20M steps at one snapshot per
500K steps) is calibrated to break 2–3-step cycles. If the monotonicity score indicates
longer cycles, raise `max_snapshots` proportionally.

### Reward Annealing

The reward function has two categories of signals:

- **Outcome signals** — HP delta, faints, win/loss: keep at full strength throughout.
- **Guidance/shaping signals** — switch subsidies, pivot bonuses, matchup penalty, spikes,
  status, sleep rotation, roar, repetition tax, etc.: useful early for sample efficiency,
  but should be annealed toward zero as the agent matures.

Two reasons to anneal:

1. **Reward hacking**: a mature agent may optimise for shaping signals at the expense of
   actual winning — the +0.5 switch subsidy can incentivise unnecessary switches if the
   subsidy outweighs the positional cost.

2. **V_θ calibration for MCTS** (v5): the value head must estimate expected win
   *probability* from a given state. If trained with heavy shaping, V_θ outputs "expected
   shaped reward," not probability — this degrades MCTS leaf evaluation quality.

**Schedule**: start annealing when `eval/elo_live` has been flat for ~10M steps
(indicating the agent has internalised the heuristics). Complete the anneal over the
following 20M steps. For a typical 75M-step self-play run this is roughly
`--reward-anneal-start 50000000 --reward-anneal-end 70000000`. Adjust based on when ELO
actually plateaus — the flat-ELO trigger is the correct signal, not a fixed step count.

If the self-play run ends before annealing completes, pass the same anneal range to the
league play run so it completes naturally.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/agents/training/snapshot_pool.py` | Pool management: save, sample, LRU load, ELO state |
| `src/agents/training/elo_tracker.py` | ELO rating updates, persistence, TensorBoard logging |
| `src/agents/training/selfplay_callback.py` | SB3 BaseCallback: eval loop, promotion gating, opponent staging |

## Files to Modify

| File | Change |
|------|--------|
| `src/agents/training/gen3_env.py` | `_staged_opponent_path`, `_load_frozen_opponent()`, reset hook |
| `src/main/train_rl_agent.py` | `--self-play` flag, `--snapshot-dir`, `--eval-interval`, `--promote-threshold`; wire `SelfPlayCallback` |

---

## CLI Example

```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py \
  --model models/v3_best.zip \
  --steps 75000000 \
  --n-envs 64 \
  --self-play \
  --snapshot-dir models/v4_selfplay/snapshots \
  --eval-interval 500000 \
  --promote-threshold 0.65 \
  --reward-anneal-start 50000000 \
  --reward-anneal-end 70000000 \
  --device cuda \
  --log-level periodic
```

Without `--self-play`, training behaves exactly as before (heuristic opponent). The flag
is additive — existing hyperparameters are unchanged.

---

## Verification

1. **Pool smoke test**: Run `--debug --steps 20000 --self-play`; confirm the initial
   snapshot is written to `snapshots/`, the opponent loads without error, and at least one
   evaluation cycle runs.

2. **ELO monotonicity**: Over the first 20M steps, `eval/elo_live` should trend upward
   (not necessarily monotonically — short plateaus are fine). A flat ELO over 10M+ steps
   indicates self-play has converged and is the trigger for reward annealing (and
   eventually league play in Step 2).

3. **Win-rate calibration**: Spot-check that `eval/win_rate_vs_pool` is in [0.4, 0.8]
   for healthy training. Values persistently near 1.0 mean the pool is stale (promote
   threshold too high or eval interval too long); values near 0.0 mean the agent has
   regressed.

4. **No hot-swap hangs**: With `--n-envs 64`, the opponent swap in `reset()` should
   complete in < 2s (checkpoint load from SSD). Confirm no episode exceeds the stall
   threshold due to the swap.

---

## Final State

Step 1 is complete when **all three gates are green**:

| Gate | Metric | Threshold |
|------|--------|-----------|
| **Strength** | `eval/elo_live` flat for ≥ 10M steps | No improvement plateau |
| **Diversity** | `eval/sentinel_monotonicity` ≥ 0.6 | Pool is not cycling |
| **Regression guard** | `eval/win_rate_vs_heuristic` ≥ 80% | No forgetting of basics |

The ELO plateau is the primary trigger — it means the self-play distribution is saturated
and additional snapshots produce no new learning signal. The diversity gate confirms the
plateau is strategic maturity rather than pool collapse (a collapsed pool also plateaus
ELO, but with a falling monotonicity score and rising win-rate oscillation σ).

Reward annealing (`--reward-anneal-start / --reward-anneal-end`) should be at least 50%
complete before league play starts. If the run ends before annealing finishes, pass the
same anneal range to the league training command so it completes naturally.

**If diversity gate fails before strength gate:**
The pool is cycling before ELO has plateaued — this is recoverable. Increase
`max_snapshots`, reduce `recency_weight` to 0, and continue. Do not start league play
until both gates are green.

**Ready for Step 2: League Play**

- `SnapshotPool` and `SelfPlayCallback` are the direct extension points
- PFSP sampling replaces uniform sampling in `SnapshotPool.sample()`
- Exploiter agents reuse `SnapshotPool` and `SelfPlayCallback` with a different
  `opponent_filter` (Main Agent snapshots only, not the full pool)
- The final self-play checkpoint (at ELO plateau) becomes the Main Agent seed for league
