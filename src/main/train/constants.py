"""Run-wide constants for the training entry point.

Lives in its own module because several phase modules read them (`parser` renders two of them
into help text, `env_factory` and `final_eval` both build players in `BATTLE_FORMAT`), and a
constant imported from a phase module would make the phase order load-bearing.

It also holds the two pure CHECKPOINT-CADENCE conversions, for exactly that reason: phase 1
(`config`, which refuses a starved counterfactual duty cycle) and phase 4 (`callbacks`, which
builds the checkpointer) must agree on the arithmetic to the step, and phase 1 importing phase 4
would make the phase order load-bearing.
"""
import math
from typing import Optional

BATTLE_FORMAT = "gen3ou"
CLIP_RANGE_DEFAULT = 0.15


# --- CHECKPOINT CADENCE ---------------------------------------------------------------------
#
# 🚨 SB3's `CheckpointCallback.save_freq` COUNTS VEC-ENV CALLS, NOT ENV STEPS. One `_on_step` fires
# per `vec_env.step()`, which advances `n_envs` environments at once — so the ENV-step interval is
# `save_freq * n_envs`, and the multiplier is invisible at the call site. That is not a hypothetical
# readability point: this value was hardcoded at 50 000 and read as "50k steps" for the whole of the
# R1 counterfactual work, while at `--n-envs 48` it was **2 400 000** env steps. The label producer
# can only reload the newest `checkpoints/` zip and the consumer expires a label older than
# `--cf-label-lag-steps` (150 000), so the labels were fresh for 6.25% of each interval — measured
# on `ai_v9_29_rev1_0823` as 6 labels ingested against 255 expired in two hours.
#
#: The historical hardcoded `save_freq`, in VEC-ENV CALLS. It is the value a run with no
#: `--checkpoint-every-steps` still gets, byte for byte, so today's behaviour is preserved exactly.
DEFAULT_CHECKPOINT_SAVE_FREQ_VEC_CALLS = 50_000

#: Below this fraction the counterfactual label path is starved by construction (see
#: `cf_label_duty_cycle`). 0.25 = a label stays fresh for at least a quarter of the interval
#: between the checkpoints the producer stamps its labels with.
CF_DUTY_CYCLE_FLOOR = 0.25


def checkpoint_save_freq_vec_calls(checkpoint_every_steps: Optional[int], n_envs: int) -> int:
    """`--checkpoint-every-steps` (ENV steps) → SB3's `save_freq` (VEC-ENV CALLS).

    `None` — the flagless default — returns the historical hardcoded constant UNCHANGED, so a run
    that names no new flag constructs a byte-identical checkpointer. A value is converted by
    CEIL-division: rounding down would checkpoint more often than asked and rounding to zero would
    make `n_calls % save_freq` a ZeroDivisionError, so the floor is 1 vec-call.
    """
    if checkpoint_every_steps is None:
        return DEFAULT_CHECKPOINT_SAVE_FREQ_VEC_CALLS
    n = max(1, int(n_envs))
    return max(1, math.ceil(int(checkpoint_every_steps) / n))


def checkpoint_interval_env_steps(checkpoint_every_steps: Optional[int], n_envs: int) -> int:
    """The EFFECTIVE env-step interval between periodic checkpoints — post-rounding.

    Reported rather than the requested value, because the ceil above can only make the real
    interval LONGER than asked, and a duty cycle computed on the request would flatter the config
    it is meant to refuse.
    """
    n = max(1, int(n_envs))
    return checkpoint_save_freq_vec_calls(checkpoint_every_steps, n) * n


def cf_label_duty_cycle(cf_label_lag_steps: Optional[int], interval_env_steps: int) -> float:
    """The fraction of a checkpoint interval during which a produced label is still ACCEPTED.

    The producer stamps every label with the step of the newest checkpoint it could load, and the
    consumer drops a row whose `policy_step` is more than `--cf-label-lag-steps` behind the live
    policy. So the two flags define a duty cycle, and nobody was computing it: everything about
    both halves reads healthy while ~94% of the produced labels expire in the buffer.

    `--cf-label-lag-steps 0` means "never expire" and therefore has no duty cycle at all — infinity,
    not a divide-by-zero and not a silent 0.
    """
    if not cf_label_lag_steps:
        return float("inf")
    if interval_env_steps <= 0:
        return float("inf")
    return float(cf_label_lag_steps) / float(interval_env_steps)


# Wait for an in-flight subprocess eval to FINISH on a graceful restart so its
# results land before exit. A scheduled restart is self-initiated by
# GracefulRestartCallback at a rollout boundary, and the launcher won't force-kill
# until the child overruns the deadline by --restart-grace-minutes (20 min default),
# so a 10-min drain fits. The checkpoint is saved first either way, so even the
# pathological forced-SIGTERM case (child already overran → ~90s SIGKILL) is safe —
# it only risks losing the in-flight eval, never the checkpoint.
_ABORT_EVAL_DRAIN_SEC = 600.0


# gen3_smoke_eval_scale_v1: a short run is a SMOKE, and a smoke's final eval is a formality.
# Measured: the final eval is 9 opponents x --eval-battles games; at 100 that is ~900 battles and
# ran past a 300s timeout on a loaded box, printing "Training complete" BEFORE it started — which
# reads exactly like a hang and cost real debugging time. Scaling it for short runs removes the
# tax; the honest banner below removes the confusion.
SMOKE_STEPS = 100_000
SMOKE_EVAL_BATTLES = 5
DEFAULT_EVAL_BATTLES = 100
