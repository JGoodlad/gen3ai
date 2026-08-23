"""Run-wide constants for the training entry point.

Lives in its own module because several phase modules read them (`parser` renders two of them
into help text, `env_factory` and `final_eval` both build players in `BATTLE_FORMAT`), and a
constant imported from a phase module would make the phase order load-bearing.
"""

BATTLE_FORMAT = "gen3ou"
CLIP_RANGE_DEFAULT = 0.15


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
