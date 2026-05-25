# Implementation: Step 1 — Self-Play Training

This step replaces the fixed heuristic opponent with a pool of frozen snapshots of the
agent itself, with win-rate gating to control when a snapshot is promoted and sentinel
evaluation to detect cycling. The heuristic opponents are retained as a floor — their
fraction of training envs decreases as the agent improves.

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

### Snapshot Pool (`snapshot_pool.py`)

A `SnapshotPool` manages a directory of `.zip` checkpoint files written during training.
Each file is named `snapshot_{step:012d}.zip`. Pool state is reconstructed entirely from
the directory on every startup — there is no JSON manifest.

```
models/<run_id>/snapshots/
  snapshot_000000000000.zip   # step-0 seed (pinned, never evicted)
  snapshot_002000000000.zip
  snapshot_005500000000.zip
  ...
```

Pool behaviour:
- **Max size**: keep the last `max_snapshots` files (default 20); oldest non-pinned entries
  are evicted when the pool overflows.
- **Seed pinning**: the step-0 entry is always pinned and never evicted. It is written once
  on the first `--self-play` run and skipped on subsequent restarts if the file exists.
- **Sampling**: weighted toward recent snapshots via a linear `recency_weight` (default 0.3);
  `recency_weight=0` → uniform.
- **LRU model cache**: loaded `MaskablePPO` objects are cached (default size 3) to avoid
  reloading the same checkpoint on consecutive samples.
- **Win-rate persistence**: `win_rate_vs_bots.txt` stores the last bot win rate across
  launcher restarts so `heuristic_fraction()` resumes from the correct curriculum position.

### Heuristic Fraction Curriculum

Rather than a hard switch from heuristics to self-play, the fraction of training envs
using heuristic opponents decreases smoothly as the agent improves:

```python
def heuristic_fraction(win_rate_vs_bots: float) -> float:
    t = (win_rate_vs_bots - 0.50) / (0.85 - 0.50)
    t = max(0.0, min(1.0, t))
    t_smooth = t * t * (3.0 - 2.0 * t)   # Hermite smoothstep
    return 0.80 * (1.0 - t_smooth) + 0.10 * t_smooth
```

Below 50% vs bots: 80% of envs use heuristics (agent is weak, needs guidance).
Above 85% vs bots: 10% of envs use heuristics (floor to prevent forgetting basics).
The ramp uses Hermite smoothstep to avoid abrupt transitions.

### Eval Callback (`selfplay_callback.py`)

`SelfPlayCallback` replaces `PerOpponentEvalCallback` when `--self-play` is active.
On each eval cycle (shared adaptive schedule: 1M/100 games → 2M/200 → 3M/300):

**1. Bot eval** — battles against Random, Heuristic, Staller, Aggressive, SetupSweep.
  Per-opponent win rate, mean reward, and mean episode length logged. Aggregate metrics
  exclude Random (too weak to be a meaningful signal):

  - `eval/win_rate_vs_bots` — mean win rate vs non-Random opponents
  - `eval/mean_reward_vs_bots`
  - `eval/mean_ep_len_vs_bots`
  - `win_rate_vs_bots` persisted to disk for next restart

**2. Pool / sentinel eval** — battles against up to 5 evenly-spaced snapshots from the
  pool (newest → oldest). Computes:

  - `eval/win_rate_vs_pool` — mean win rate across all sentinels
  - `eval/sentinel_monotonicity` — Kendall's τ over sentinel win rates; healthy training
    should produce monotonically decreasing win rates as snapshots get older. Below 0.6
    emits a `⚠️ [SELFPLAY] Cycling signal` event.

**3. Promotion** — if `win_rate_vs_pool > promote_threshold` (default 0.65), the current
  model is added to the pool as a new snapshot.

**4. Best-model saving** — if `win_rate_vs_bots` is a new high, `best_model.zip` is saved.

### Bot Regression Guard

`_check_bot_regression()` monitors each non-Random opponent for regression. It is
**edge-triggered**: the warning fires once when the bot first drops below the threshold,
is silenced while the regression persists, and re-arms when the win rate recovers above
the threshold. This prevents alert storms during a genuine slump.

Threshold: if a bot win rate that previously reached 60%+ drops below 60%, emits
`⚠️ [SELFPLAY] BOT_REGRESSION`. State is tracked in `_regression_active: set[str]` and
resets each launcher run (TensorBoard records the full history for post-hoc review).

### Canonical Abort Path

`abort_training(reason)` is a closure defined in `_setup_signal_handlers()` and injected
into both `SelfPlayCallback` and `PerOpponentEvalCallback` as `abort_fn`. It:

1. Sets `shutdown_event`
2. Saves a full checkpoint (model + `latest.txt` + metadata)
3. Calls `os._exit(int(TrainExitCode.INTERRUPTED))` — works from any thread, exits with
   code 15 so the launcher restarts

Using `os._exit()` rather than `sys.exit()` is required because eval runs on a background
thread (`threading.Thread` + `thread.join()`); `sys.exit()` from a thread only kills that
thread, leaving the main training loop running.

### Utility Functions (`eval_callback.py`)

- `opponent_name(cls) -> str` — maps player class to display name via `_OPPONENT_NAMES`
  dict; falls back to `cls.__name__`. Source of truth for TensorBoard metric keys and TUI
  labels. `RANDOM_OPPONENT_NAME` is derived from this, not hardcoded.
- `bot_mean(d: dict[str, float]) -> float` — filters out the Random opponent and averages
  the rest. Used by both callbacks for win rate, reward, and episode length aggregates.

### TUI Integration

The launcher TUI displays:

```
eval
  all          53.8%    1.754    ← aggregate over all opponents
  vs Bots      44.4%   -4.2      ← excludes Random; reward shown
  vs Pool      61.2%    ——       ← self-play only; appears when --self-play active
  ─────────────────────────────
  vs Random    91.3%   31.75
  vs Heuristic 40.0%   -8.24
  vs Staller   46.3%   -4.81
  vs Aggressive 50.7%  -1.05
  vs SetupSweep 40.7%  -8.88
```

Random is always first in the per-opponent section. "vs Pool" only renders when the
metric is present (i.e. `--self-play` is active).

---

## Files Created

| File | Purpose |
|------|---------|
| `src/agents/training/snapshot_pool.py` | Pool management: save, evict, sample, LRU load, win-rate persistence |
| `src/agents/training/selfplay_callback.py` | SB3 callback: bot eval, sentinel eval, promotion, regression guard |
| `src/agents/training/snapshot_pool_test.py` | Unit tests for SnapshotPool and heuristic_fraction |
| `src/agents/training/selfplay_callback_test.py` | Unit tests for SelfPlayCallback pure functions and regression guard |

## Files Modified

| File | Change |
|------|--------|
| `src/main/train_rl_agent.py` | `--self-play` flag; pool/env wiring; `abort_training` closure; `opponent_name()` for tuple keys |
| `src/agents/training/eval_callback.py` | `eval_schedule()` shared function; `opponent_name()`, `bot_mean()`, `RANDOM_OPPONENT_NAME`; bot aggregate metrics; `abort_fn` pattern |
| `src/agents/training/eval_callback_test.py` | Tests for `bot_mean`, `opponent_name`, `RANDOM_OPPONENT_NAME` |
| `src/main/launcher/ui.py` | "vs Bots" and "vs Pool" summary rows; Random pinned first |

---

## CLI

```bash
# Fresh self-play run
python src/main/train_rl_agent.py \
  --model models/v3_best.zip \
  --steps 75000000 \
  --n-envs 64 \
  --self-play \
  --promote-threshold 0.65 \
  --device cuda

# Resume from checkpoint (launcher handles --self-play flag forwarding)
python -m main.launcher \
  --restart-interval-hours 3 \
  --model models/v4_selfplay/checkpoint_10000000_steps.zip \
  --steps 75000000 \
  --self-play \
  --device cuda
```

Without `--self-play`, training uses `PerOpponentEvalCallback` (heuristic opponents only)
and all self-play code paths are completely dormant.

---

## Verification

1. **Pool smoke test**: Run `--debug --steps 20000 --self-play`; confirm `snapshot_000000000000.zip`
   written, eval cycle runs, `eval/win_rate_vs_pool` logged.

2. **Curriculum check**: Early in training `train/selfplay_fraction` should be low (heavy
   heuristics); it should rise as `eval/win_rate_vs_bots` climbs above 50%.

3. **Monotonicity check**: `eval/sentinel_monotonicity` should generally stay above 0.6.
   A sustained drop means the pool is cycling — reduce `recency_weight` toward 0.

4. **Regression guard**: Force a regression in dev (`--debug`) and confirm the warning fires
   once on entry, stays silent, re-fires after recovery.

---

## Deliberate Omissions

See `todo.md` for a full list of features from the original design that were deferred or
replaced. Key decisions:

- **No ELO tracking**: win_rate_vs_bots serves as the curriculum signal and is simpler,
  more interpretable, and does not require per-snapshot state. ELO adds complexity without
  clear benefit at this stage.
- **No demotion threshold**: replaced by the regression guard, which is softer (warning,
  not hard stop) and more actionable.
- **No reward annealing**: deferred; will be addressed before league play begins.
- **No hot-swap**: opponents swap at launcher restart (~2.5h). The `_staged_opponent_path`
  mechanism is documented as a future extension in `todo.md`.
