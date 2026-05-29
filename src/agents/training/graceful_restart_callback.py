"""Graceful, rollout-boundary restart for launcher-managed training runs."""

import os
import time

from stable_baselines3.common.callbacks import BaseCallback

from main.launcher.ipc import send_event

_INTERVAL_ENV = "LAUNCHER_RESTART_INTERVAL_SEC"


class GracefulRestartCallback(BaseCallback):
    """Stops training cleanly at the next rollout boundary once the launcher's
    restart interval has elapsed.

    The launcher (``src/main/launcher``) forwards its ``--restart-interval-hours``
    to the child as ``LAUNCHER_RESTART_INTERVAL_SEC``. This callback measures
    wall-clock time from training start and, at the first ``on_rollout_end``
    after the interval passes, calls the canonical ``abort_fn`` — the same
    ``abort_training`` closure used by the SIGTERM handler — which saves a full
    checkpoint and exits with ``TrainExitCode.INTERRUPTED`` (15) so the launcher
    restarts the run.

    Stopping at a rollout boundary rather than at an arbitrary instant via
    SIGTERM avoids discarding a partially-collected rollout, and — because evals
    run synchronously inside rollout collection — guarantees no eval is ever
    interrupted mid-flight.

    Inert when ``LAUNCHER_RESTART_INTERVAL_SEC`` is absent or <= 0 (``--debug``
    runs, direct ``train_rl_agent.py`` invocations, launcher runs with
    ``--restart-interval-hours 0``).
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        raw = os.environ.get(_INTERVAL_ENV, "")
        try:
            self._interval = float(raw)
        except ValueError:
            self._interval = 0.0
        # Wired by train_rl_agent after the signal handlers are set up — the
        # single canonical "save a checkpoint and exit 15" path.
        self.abort_fn = None
        self._start: float | None = None
        self._fired = False

    @property
    def armed(self) -> bool:
        return self._interval > 0

    def _on_training_start(self) -> None:
        if self.armed:
            self._start = time.monotonic()

    def _on_rollout_end(self) -> None:
        if not self.armed or self._fired or self.abort_fn is None or self._start is None:
            return
        if time.monotonic() - self._start >= self._interval:
            self._fired = True
            send_event(
                f"♻️  Restart interval ({self._interval / 60.0:.1f}m) elapsed "
                f"— restarting at rollout boundary"
            )
            # Does not return — abort_fn saves a checkpoint and os._exit(15)s.
            self.abort_fn(
                f"restart interval ({self._interval:.0f}s) elapsed "
                f"— clean rollout-boundary restart"
            )

    def _on_step(self) -> bool:
        return True
