"""The run's CLEAN STOPS: the graceful rollout-boundary restart, and the SAFE POINTS at which a
signal-requested abort runs (`main.train.deferred_abort`, gen3_deferred_abort_v1)."""

import os
import time

from agents.training.loop_callbacks import BaseCallback

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

    The interval restart is inert when ``LAUNCHER_RESTART_INTERVAL_SEC`` is absent or <= 0
    (``--debug`` runs, direct ``train_rl_agent.py`` invocations, launcher runs with
    ``--restart-interval-hours 0``).

    **The safe points (always on, gen3_deferred_abort_v1).** A SIGINT / SIGTERM / SIGHUP only
    RECORDS a stop request; ``safe_point_fn`` (the run's ``DeferredAbort.safe_point``) runs it here,
    at every event the loop fires — ``training_start``, ``rollout_start`` (the first boundary after an
    update), every collector ``step``, ``rollout_end`` and ``training_end`` (a request made during the
    last update). None of them is inside an update or a logger dump, so the abort's dump cannot
    deadlock on TensorBoard's lock and its save cannot tear a checkpoint mid-update. Each is checked
    FIRST, before this callback's own work.
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
        # Wired with abort_fn: the run's `DeferredAbort.safe_point` (a signal-requested abort, run here).
        self.safe_point_fn = None
        # Wired with abort_fn: the run's `DeferredAbort.disk_stop` (the checkpoint callback's disk guard
        # stops through it; `utils.disk_guard`).
        self.disk_stop_fn = None
        self._start: float | None = None
        self._fired = False

    @property
    def armed(self) -> bool:
        return self._interval > 0

    def _safe_point(self, where: str) -> None:
        if self.safe_point_fn is not None:
            self.safe_point_fn(where)       # does not return when an abort was requested

    def _on_training_start(self) -> None:
        self._safe_point("training_start")
        if self.armed:
            self._start = time.monotonic()

    def _on_rollout_start(self) -> None:
        self._safe_point("rollout_start")

    def _on_training_end(self) -> None:
        self._safe_point("training_end")

    def _on_rollout_end(self) -> None:
        self._safe_point("rollout_end")
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
        self._safe_point("step")
        return True
