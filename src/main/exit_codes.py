import signal
from enum import IntEnum
from typing import Optional


class TrainExitCode(IntEnum):
    COMPLETE     = 0              # all steps done — launcher should stop restarting
    INTERRUPTED  = signal.SIGTERM # == 15, caught SIGTERM: checkpoint saved, please restart
    CRASH        = 1              # unhandled exception — launcher auto-restarts from last checkpoint
    FATAL_CONFIG = 3             # non-recoverable config/arch error (e.g. checkpoint arch-family or
                                 # vf_coef/reward-config mismatch) — restarting would hit the SAME
                                 # error every time, so the launcher must NOT restart; it gives up
                                 # immediately and surfaces the reason instead of looping.
    FATAL_NONFINITE = 4          # the LEARNER went non-finite (a NaN / Inf loss or gradient —
                                 # `NonFiniteLearnerError`, Lane K's K9 fail-closed guard). Same
                                 # class as FATAL_CONFIG: a restart resumes the checkpoint that
                                 # produced it and replays the same update, so a persistent NaN
                                 # would crash-loop until the budget runs out. The launcher STOPS.
    FATAL_SUPPLY = 5             # a LIVE coefficient's EXTERNAL SUPPLY is dead or starved in flight
                                 # (`SupplyStarvedError` — e.g. `--cf-winprob-coef` > 0 and no cf
                                 # label accepted for N cycles, or the label producer exited). A
                                 # restart would train the same run on the same missing supply, so
                                 # the launcher STOPS and names the supplier.
    FATAL_CUDA_LEAK = 6          # the learner's CUDA memory TREND stopped the run (`CudaMemoryLeakError`,
                                 # K6's memory half): a SUSTAINED growth of live CUDA memory projected
                                 # an OOM inside the declared horizon, and the trainer checkpointed
                                 # (`final_model_exception.zip`) before exiting. A fresh process clears
                                 # a leak by definition, so the launcher RESTARTS from that checkpoint —
                                 # at most `CUDA_LEAK_RESTART_CAP` times per launcher session, each one
                                 # logged loudly; the next one STOPS for good (a reproducible leak that
                                 # wants a human).


#: How many CUDA-leak stops the launcher restarts per session before it gives up (orchestrator,
#: 2026-10-01: "a small cap per run").
CUDA_LEAK_RESTART_CAP = 2


class NonFiniteLearnerError(FloatingPointError):
    """A non-finite loss or gradient in the learner — fail CLOSED (Lane K, K9).

    Raised by the learner's guard; the trainer's fail-fast handlers map it to
    ``TrainExitCode.FATAL_NONFINITE`` (:func:`exit_code_for`) and the launcher gives up on that code
    instead of restarting into the same update. A guard that defines its own class must SUBCLASS
    this one (the name is also matched along the MRO, so a same-named class maps too)."""


class FatalConfigError(RuntimeError):
    """A deterministic CONFIGURATION defect found after argument parsing — fail CLOSED.

    Maps to ``TrainExitCode.FATAL_CONFIG`` through :func:`exit_code_for`, so it can be RAISED from
    anywhere on the startup path (or from a callback that discovers a mis-wired lever at its first
    use) and the trainer's fail-fast handlers still exit 3, which the launcher does not restart. The
    class it closes (`gen3_supply_guard_v2`, 2026-09-30): `--bot-weights` typos and consensus
    warm-start failures exited 1 (CRASH), so the launcher restarted them into the identical error —
    bounded by the rapid-crash breaker for a fast failure, and UNBOUNDED for a warm-start that ran
    longer than the breaker's 10-minute window before failing."""


class SupplyStarvedError(RuntimeError):
    """A live coefficient's EXTERNAL SUPPLY delivered nothing — fail CLOSED (`gen3_supply_guard_v1`).

    The class this closes: `ai_v12_12_ladder_cflabels` trained 10M steps at `--cf-winprob-coef 0.5`
    and received ZERO labels, because the producer is a separate program nobody started and every
    counter that could have said so was a scalar nobody thresholded. The in-flight guard raises this
    (or a subclass); the trainer's handlers map it to ``TrainExitCode.FATAL_SUPPLY`` and the
    launcher gives up instead of restarting into the same starvation."""


#: Exception class NAMES mapped to a fatal exit code — matched along each exception's MRO and its
#: ``__cause__`` / ``__context__`` chain, so the mapping holds however the error was wrapped. Matched
#: by NAME so this module imports nothing from ``agents``.
#:
#: * ``NonFiniteLearnerError`` — K9's learner guard (a restart replays the update).
#: * ``CudaMemoryLeakError`` — K6's memory trend STOP (`agents.training.learner_lifecycle`): NOT a
#:   configuration error, and RESTARTED (capped) by the launcher — see ``FATAL_CUDA_LEAK``.
#: * ``NonFiniteWeights`` — the T2 inference service refused a NaN / Inf weight set (the trainee after
#:   an update, a snapshot, a checkpoint template). A restart resumes the same weights.
#: * ``ParityFailure`` (and its ``VacuousParity``) — the T2 inference service's parity gate refused a
#:   slot. The verdict is a deterministic function of (code, weights, the committed fixture), and a
#:   restart resumes the SAME weights (a crash saves ``final_model_exception.zip``; the next startup
#:   gates it as the slot template), so it replays at every restart. Observed 2026-09-30:
#:   ``~/gen3ai_archive/cutover_prep/fresh3`` crash-looped three times on one `VacuousParity`
#:   before the circuit breaker stopped it.
_FATAL_BY_NAME = {"NonFiniteLearnerError": TrainExitCode.FATAL_NONFINITE,
                  "UpdateWontFit": TrainExitCode.FATAL_CONFIG,
                  "CudaMemoryLeakError": TrainExitCode.FATAL_CUDA_LEAK,
                  "NonFiniteWeights": TrainExitCode.FATAL_NONFINITE,
                  "SupplyStarvedError": TrainExitCode.FATAL_SUPPLY,
                  "FatalConfigError": TrainExitCode.FATAL_CONFIG,
                  # `utils.paths.RunArchiveError`: a run dir the archive cannot hold (no archive, or
                  # inside a linked worktree's own models/) — a restart meets the same refusal.
                  "RunArchiveError": TrainExitCode.FATAL_CONFIG,
                  "ParityFailure": TrainExitCode.FATAL_CONFIG}


def fatal_exit_code_for(exc: Optional[BaseException]) -> Optional[int]:
    """The dedicated FATAL exit code for ``exc`` (or anything in its cause chain), else ``None``."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        for cls in type(exc).__mro__:
            code = _FATAL_BY_NAME.get(cls.__name__)
            if code is not None:
                return int(code)
        exc = exc.__cause__ or exc.__context__
    return None


def exit_code_for(exc: Optional[BaseException]) -> int:
    """The trainer's exit code for an UNCAUGHT exception: a mapped FATAL code, else ``CRASH`` (1)."""
    code = fatal_exit_code_for(exc)
    return int(TrainExitCode.CRASH) if code is None else code
