# Implementation: Step 10 — Adaptive Training Infrastructure

This step hardens the training loop for long-horizon runs (200M+ steps). Three
independent but related pieces of work were shipped together:

1. **Decoupled `AdaptivePPOCallback`** — the old `AdaptiveLRCallback` used a single
   signal (approx_kl) to drive a single lever (LR), causing the LR to be suppressed
   far below useful levels because n_epochs was fixed at 10. The replacement uses two
   independent signals driving two independent levers: `clip_fraction → LR` and
   `approx_kl → n_epochs`.

2. **Checkpoint resume-state persistence** — `n_epochs` is not stored in SB3's
   optimizer state, so it was lost on every launcher restart. A per-checkpoint
   `.json` metadata file and a run-level `snapshot_history` dict now preserve both LR
   and n_epochs, with a three-tier fallback on resume.

3. **TensorBoard observability** — `train/n_epochs` is now emitted every rollout,
   giving a continuous chart alongside `train/approx_kl` and `train/clip_fraction` so
   all three adaptive signals are visible in a single TensorBoard view.

---

## 1. Decoupled AdaptivePPOCallback

### Motivation

The original `AdaptiveLRCallback` mapped `approx_kl → LR`: if KL was too high, LR
was reduced. With `n_epochs = 10` fixed, each rollout produced 80 gradient steps
(`n_envs × n_steps / batch_size × n_epochs`). Under normal training, 80 steps drive
KL well above the 0.02 target, so the LR was driven down 9× over the first millions
of steps until the gradient magnitude itself was too small to move KL. The result was
a policy that barely updated despite having plenty of rollout data.

The root cause is that KL is a product of *both* step size (LR) and number of update
passes (n_epochs). Using KL to control only LR while holding n_epochs fixed creates a
degenerate single-equation, two-unknown system with no stable equilibrium.

The fix decouples the two control variables:
- `clip_fraction` measures whether individual gradient *steps* are too large (hitting
  the PPO clip boundary) — it drives LR.
- `approx_kl` measures whether the policy changed *too much* across the full epoch
  pass — it drives n_epochs.

`gae_lambda` was also updated from 0.85 to 0.95. Gen 3 battles typically run 10–30
turns; the old value discounted rewards too aggressively across turn sequences, making
long-game strategies (hazard stacking, weather set-up, sweep positioning) look lower
value than they are.

### Design

Two independent levers, both evaluated at the end of each rollout from the *previous*
rollout's logged metrics (one-rollout lag is fine at this timescale):

**clip_fraction → LR**

| Param | Value | Rationale |
|---|---|---|
| `clip_lo` | 0.07 | Below this, steps are too cautious — room to increase LR |
| `clip_hi` | 0.15 | Above this, too many steps hit the clip boundary — decrease LR |
| `lr_factor` | 1.2 | Gradual ÷/× per rollout; avoids oscillation |
| `min_lr` | 1e-5 | Hard floor |
| `max_lr` | `initial_lr × 2` | Hard ceiling; based on the LR the run was *started* with, not the checkpoint LR |

**approx_kl → n_epochs**

| Param | Value | Rationale |
|---|---|---|
| `kl_lo` | 0.010 | Below this, the policy barely changed — add an epoch |
| `kl_hi` | 0.020 | Above this, the policy changed too much — drop an epoch |
| `epochs_step` | 1 | ±1 per rollout |
| `min_epochs` | 2 | Hard floor |
| `max_epochs` | 12 | Hard ceiling; headroom above the starting value of 10 |

Both levers are silent on the first rollout (signals are `None` before the first
`model.train()` call). The console logs only when something changes:

```
[AdaptivePPO] clip=0.18 → LR ↓ 3.25e-05 → 2.71e-05 | kl=0.021 → epochs ↓ 10 → 9
```

`train/n_epochs` is recorded to TensorBoard unconditionally every rollout — even
when the value is stable — so the chart has no gaps. Placement is at the end of
`_on_rollout_end`, after any adjustment, so the logged value is always the one that
will be used in the immediately following `model.train()` call.

The old `AdaptiveLRCallback` name is kept as a module-level alias for backwards
compatibility with any external references.

---

## 2. Checkpoint Resume-State Persistence

### Motivation

When the launcher restarts the training child (every ~3 hours by default), SB3 loads
the most recent checkpoint. The optimizer state (including the current LR as stored by
Adam) survives this reload. The current `n_epochs` does **not** — it exists only as a
live attribute on the `MaskablePPO` instance and is reset to `args.n_epochs` on every
restart. After the adaptive callback had driven n_epochs to, say, 7, a restart would
silently reset it to 10, pushing KL high and triggering the LR decay cycle again.

### Per-Checkpoint Metadata Files

Every time `_TrackingCheckpointCallback` saves a `.zip` checkpoint it also writes a
sibling `.json` file at the same path (e.g. `checkpoint_50000000_steps.json`):

```json
{"current_lr": 2.5e-5, "current_epochs": 8}
```

This records the adaptive state at exactly the moment of that checkpoint save.
`write_checkpoint_metadata()` / `read_checkpoint_metadata()` in `snapshot.py` handle
serialisation and a missing-file fallback (`{}` when not found).

### Snapshot History in metadata.json

Each checkpoint save also appends an entry to `snapshot_history` inside the run-level
`metadata.json` via `record_snapshot_in_history()`:

```json
{
  "snapshot_history": {
    "checkpoint_50000000_steps.zip":  {"lr": 2.5e-5, "n_epochs": 8},
    "checkpoint_100000000_steps.zip": {"lr": 2.1e-5, "n_epochs": 7}
  }
}
```

`save_model_snapshot()` — which is also called by the SIGTERM handler on shutdown —
reads any existing `snapshot_history` before overwriting `metadata.json` and writes it
back, so the full run history is never lost.

### Three-Tier Resume Priority

When the training script resumes from a checkpoint, n_epochs is resolved in order:

1. **Per-checkpoint metadata** (`checkpoint_NNNN_steps.json`) — the most accurate
   source; reflects the adaptive state at exactly that checkpoint step.
2. **Run-level `metadata.json` `current_epochs`** — written by the SIGTERM handler;
   reflects the state at the moment of the last graceful shutdown, which may be later
   than any checkpoint step.
3. **`args.n_epochs`** — fallback for the first run or checkpoints saved before this
   infrastructure existed.

---

## Files Changed

| File | Change |
|---|---|
| `src/agents/training/adaptive_lr_callback.py` | Replaced `AdaptiveLRCallback` (KL→LR) with `AdaptivePPOCallback` (clip_fraction→LR + KL→n_epochs); `train/n_epochs` logged every rollout; `AdaptiveLRCallback` alias preserved |
| `src/agents/training/adaptive_lr_callback_test.py` | 18 unit tests: LR increase/decrease/clamp, epochs increase/decrease/clamp, both levers together, first-rollout silence, max_lr default |
| `src/agents/model/snapshot.py` | Added `write_checkpoint_metadata`, `read_checkpoint_metadata`, `_checkpoint_metadata_path`; added `record_snapshot_in_history`; `save_model_snapshot` gains `current_lr`/`current_epochs` params and preserves existing `snapshot_history` on overwrite |
| `src/agents/model/snapshot_test.py` | Tests for metadata path stripping, write/read round-trip, missing-file fallback, overwrite, per-checkpoint independence, history accumulation, history preservation on `save_model_snapshot` overwrite |
| `src/main/train_rl_agent.py` | `_TrackingCheckpointCallback` extended with `_current_lr_fn`/`_current_epochs_fn` lambdas; writes per-checkpoint metadata and history on each save; signal handler passes `current_epochs`; resume section reads three-tier priority and seeds `adaptive_ppo_callback._current_epochs` and `model.n_epochs`; `gae_lambda=0.95` |
