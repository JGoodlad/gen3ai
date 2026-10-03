"""Run-wide constants for the training entry point.

Lives in its own module because several phase modules read them (`parser` renders two of them
into help text), and a constant imported from a phase module would make the phase order
load-bearing.

It also holds the two pure CHECKPOINT-CADENCE conversions, for exactly that reason: phase 1
(`config`) and phase 4 (`callbacks`, which builds the checkpointer) must agree on the arithmetic to
the step, and phase 1 importing phase 4 would make the phase order load-bearing.
"""
from typing import Optional

BATTLE_FORMAT = "gen3ou"
CLIP_RANGE_DEFAULT = 0.15

# --- CHECKPOINT CADENCE ---------------------------------------------------------------------
#
# 🚨 THE PERIODIC CHECKPOINT FIRES ON A TOTAL-ENV-STEP BOUNDARY, NOT ON A CALL COUNT. SB3's stock
# `CheckpointCallback` saves on `n_calls % save_freq`, and a call is NOT a fixed number of env steps:
# a sync `vec_env.step()` advances N envs, the Rust collector fires once per N trainee decisions, and
# the `--async-rollout` collector fires once per WAVE of however many envs were ready (< N). Two
# defects came from counting calls. (1) A hardcoded 50 000 calls was read as "50k steps" for the
# whole R1 counterfactual work while at `--n-envs 48` it was **2 400 000** env steps
# (`ai_v9_29_rev1_0823`). (2) The same 50 000 calls was ~102M env steps at N = 2048 — a
# checkpointer that never fires at the sizes the M5 SIZING study sweeps — and checkpointed EARLY
# under async waves (F-SZ-3, 2026-10-01). So `_TrackingCheckpointCallback` (`main.train.run_io`)
# saves at the first call whose `num_timesteps` reaches the next multiple of the interval
# (`checkpoint_due`), the rule the eval callbacks already use.
#
#: The DEFAULT checkpoint interval, in TOTAL ENV STEPS (summed over every env) — independent of
#: `--n-envs` and of the collector's call shape. 2 400 000 = the historical N = 48 value.
DEFAULT_CHECKPOINT_EVERY_ENV_STEPS = 2_400_000

def checkpoint_interval_env_steps(checkpoint_every_steps: Optional[int]) -> int:
    """The env-step spacing of the periodic checkpoint BOUNDARIES: `--checkpoint-every-steps`, or
    `DEFAULT_CHECKPOINT_EVERY_ENV_STEPS` when unset. Floored at 1 (`checkpoint_due` divides by it).

    A save lands at the first callback call at or past each boundary, so one gap between saves is
    this ± less than one call's advance (≤ N env steps); the mean is exactly this.
    """
    if checkpoint_every_steps is None:
        return DEFAULT_CHECKPOINT_EVERY_ENV_STEPS
    return max(1, int(checkpoint_every_steps))


def checkpoint_due(last_step: int, now_step: int, interval_env_steps: int) -> bool:
    """Has `num_timesteps` crossed a checkpoint boundary since `last_step`? Boundaries are the
    multiples of the interval in TOTAL env steps, so the answer does not depend on how many env steps
    one callback call advanced (N, a wave, a ragged Rust host step) or on where this process started."""
    interval = max(1, int(interval_env_steps))
    return int(now_step) // interval > int(last_step) // interval


# Wait for an in-flight subprocess eval to FINISH on a graceful restart so its
# results land before exit. A scheduled restart is self-initiated by
# GracefulRestartCallback at a rollout boundary, and the launcher won't force-kill
# until the child overruns the deadline by --restart-grace-minutes (20 min default),
# so a 10-min drain fits. The checkpoint is saved first either way, so even the
# pathological forced-SIGTERM case (child already overran → ~90s SIGKILL) is safe —
# it only risks losing the in-flight eval, never the checkpoint.
_ABORT_EVAL_DRAIN_SEC = 600.0

