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


class NonFiniteLearnerError(FloatingPointError):
    """A non-finite loss or gradient in the learner — fail CLOSED (Lane K, K9).

    Raised by the learner's guard; the trainer's fail-fast handlers map it to
    ``TrainExitCode.FATAL_NONFINITE`` (:func:`exit_code_for`) and the launcher gives up on that code
    instead of restarting into the same update. A guard that defines its own class must SUBCLASS
    this one (the name is also matched along the MRO, so a same-named class maps too)."""


class SupplyStarvedError(RuntimeError):
    """A live coefficient's EXTERNAL SUPPLY delivered nothing — fail CLOSED (`gen3_supply_guard_v1`).

    The class this closes: `ai_v12_12_ladder_cflabels` trained 10M steps at `--cf-winprob-coef 0.5`
    and received ZERO labels, because the producer is a separate program nobody started and every
    counter that could have said so was a scalar nobody thresholded. The in-flight guard raises this
    (or a subclass); the trainer's handlers map it to ``TrainExitCode.FATAL_SUPPLY`` and the
    launcher gives up instead of restarting into the same starvation."""


#: Exception class NAMES mapped to a fatal exit code — matched along each exception's MRO and its
#: ``__cause__`` / ``__context__`` chain, so the mapping holds however the error was wrapped.
_FATAL_BY_NAME = {"NonFiniteLearnerError": TrainExitCode.FATAL_NONFINITE,
                  "SupplyStarvedError": TrainExitCode.FATAL_SUPPLY}


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
