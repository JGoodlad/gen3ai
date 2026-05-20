# Implementation: Training Infrastructure

This doc covers three training-loop improvements and two logging fixes that landed on
main between the step7 doc and the current HEAD. None of these change the observation
space or model architecture — they are training-loop mechanics that affect learning
stability, throughput, and observability.

Primary themes: closing a feedback loop on KL divergence to replace fixed LR schedules,
recovering FPS from long-running process drift, simplifying the stall penalty, and
making tensorboard output survive worktree cleanup.

---

## Adaptive Learning Rate (approx_kl feedback control)

### Motivation

A fixed learning rate (or a manually-tuned schedule) cannot respond to the actual
training dynamics. When the policy updates are small (low KL), training is too cautious
and could be faster. When updates are large (high KL), the policy is diverging and the
LR should be pulled back. PPO's clipping ratio partially limits divergence but does not
zero it — the LR still matters.

### Mechanism

`src/agents/training/adaptive_lr_callback.py` implements a proportional controller
around `approx_kl`:

| Parameter | Default | Meaning |
|---|---|---|
| `target_kl` | 0.015 | Desired divergence per policy update |
| `kl_tolerance` | 0.30 | ±30% band around target before adjusting |
| `lr_factor` | 1.2 | Multiply/divide LR by this on each adjustment |
| `min_lr` | 1e-5 | Hard lower bound |
| `max_lr` | 2 × initial_lr | Hard upper bound |

**Adjustment rule (fires at end of each rollout):**

```python
kl = model.logger.name_to_value.get("train/approx_kl")
if kl > target_kl * (1 + tolerance):   # KL too high → slow down
    new_lr = max(current_lr / lr_factor, min_lr)
elif kl < target_kl * (1 - tolerance):  # KL too low → speed up
    new_lr = min(current_lr * lr_factor, max_lr)
# else: in band, no change
```

The callback directly modifies the optimizer's `param_groups[0]["lr"]` and calls
`model.policy.optimizer.zero_grad()` to avoid stale gradients after a large LR step.

### LR persistence across restarts

On SIGTERM the launcher's SIGTERM handler writes `current_lr` to `metadata.json`
alongside the checkpoint (see launcher doc). On resume, `train_rl_agent.py` reads
`current_lr` from the checkpoint's metadata and passes it to the callback's `__init__`
as `initial_lr`. This means the adaptive controller continues from wherever it left off
rather than resetting to the `--lr` CLI argument. An IPC event (`"▶️  Resuming at LR
{lr:.2e}"`) is emitted so the user can confirm the resumed value in the TUI dashboard.

The default `--lr` argument was raised to `3e-4` (from `1.5e-4`) because the adaptive
controller naturally brings the LR down as training stabilises — starting higher gives
it room to find the right operating point faster.

### What the model sees

No change to observations or architecture. The callback is entirely a training-loop
concern — the policy gradient step uses `new_lr` but the model's weights update in
exactly the same way otherwise.

---

## Periodic SubprocVecEnv Worker Restart

### Motivation

Each SubprocVecEnv worker is a long-running subprocess that holds a Showdown WebSocket
connection, a poke-env battle state, and a Python interpreter. Over 3+ hours these
accumulate memory fragmentation, stale connection state, and polling overhead. FPS
degrades measurably over a multi-hour run.

The launcher already restarts the entire training child every N hours to reclaim
pymalloc fragmentation at the process level. The worker restart callback addresses a
finer-grained version of the same problem: recycling just the environment workers
without interrupting the training loop.

### Mechanism

`src/agents/training/env_restart_callback.py` fires at rollout boundaries (when the
in-flight rollout buffer is full and before the policy update begins):

1. Check elapsed time since last worker restart against the configured interval.
2. If the threshold is crossed: call the factory functions to create a new set of
   workers, instantiate a fresh `SubprocVecEnv` from them.
3. Attach the new env to the model via `model.set_env(new_env)`.
4. Reset `model._last_obs` and `model._last_episode_starts` to match the new env's
   initial state.
5. Close the old env (after the swap, so no race condition on the active rollout).
6. Re-attach the subprocess watchdog to the new env. Reset the timer.

The model weights, optimizer state, and rollout buffer contents are untouched.
Training continues from the same step count with fresh workers.

### Interaction with the launcher restart

The launcher sends SIGTERM to the child at the end of its interval (default 3 hours).
The worker restart callback fires at rollout boundaries within that interval. They are
independent: the worker restart is a soft recycle of the environment layer; the
launcher restart is a hard recycle of the entire child process (which also saves a
checkpoint). The two timers are separate.

---

## Stall Tax: Ramping → Flat

### Change

`src/agents/training/reward_manager.py`:

**Before (ramping):**
```python
stall_tax = -1.0 * (battle.turn - STALL_TAX_START_TURN) / 30.0
```
Started at turn 125. At turn 125: 0. At turn 250: −4.2 per turn.
The penalty grew increasingly harsh as the battle dragged on.

**After (flat):**
```python
stall_tax = -0.1  # per turn, starting at turn 125
```
Constant −0.1 per turn from turn 125 onward, regardless of how many turns have elapsed.

### Rationale

The ramping penalty created instability late in long games: the policy received an
extreme gradient signal for stall-like play that happened to extend past the 125-turn
window for legitimate reasons (walls, status wars). The flat penalty applies steady
pressure to end games without creating a catastrophic late-game reward cliff that could
destabilise training. −0.1/turn over 125 turns = −12.5 total, a meaningful incentive
to avoid stalls without overwhelming the win/loss signal.

---

## Tensorboard: Log Path and Run Naming

### Problem

When the launcher pins a training child to an isolated git worktree (a temporary
directory under `.claude/worktrees/`), tensorboard logs written to a relative
`tensorboard/` path inside that worktree were deleted when the worktree was cleaned
up after the run. Logs were lost.

### Fix: Write to repo root (`828fd2a`)

`train_rl_agent.py` now calls `get_repo_root()` (from `src/utils/git.py`) at startup
and writes tensorboard logs to `{repo_root}/tensorboard/`. This path is inside the
main repository, not the ephemeral worktree, and persists across worktree cleanup.

### Fix: Run naming convention (`f0947df`)

Before: tensorboard runs were named by SB3's default (timestamp-based).
After: `tb_log_name=f"MPPO_{run_id}"` where `run_id` is the model directory name
extracted at run start (e.g. `run_20260518_172316` → `MPPO_run_20260518_172316`).

This correlates tensorboard runs directly with checkpoint directories, making it easy
to identify which checkpoint corresponds to which training curve without timestamp
arithmetic.

---

## Files Changed

| File | Change |
|---|---|
| `src/agents/training/adaptive_lr_callback.py` | New — KL-driven LR controller |
| `src/agents/training/env_restart_callback.py` | New — periodic worker recycler |
| `src/agents/training/reward_manager.py` | Stall tax: ramping → flat −0.1/turn |
| `src/main/train_rl_agent.py` | Wire adaptive LR + worker restart callbacks; LR persistence on SIGTERM; tensorboard log path fix; run naming |
| `src/agents/model/snapshot.py` | `save_model_snapshot()` writes `current_lr` to metadata.json |
| `src/utils/git.py` | `get_repo_root()` used for tensorboard path |

## Commits

| Hash | Summary |
|---|---|
| `43ddf5d` | feat(launcher): persist LR across restarts via child→launcher IPC |
| `c1ac7d4` | fix(reward): replace ramping stall tax with flat −0.1/turn after turn 125 |
| `1d91dc9` | feat(training): adaptive LR callback based on approx_kl |
| `80f56de` | feat(training): periodic SubprocVecEnv worker restart to recover FPS |
| `828fd2a` | fix(training): write tensorboard logs to main repo root, not tmp worktree |
| `f0947df` | feat(training): use model dir ID as tensorboard run name |
