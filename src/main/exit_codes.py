import signal
from enum import IntEnum


class TrainExitCode(IntEnum):
    COMPLETE     = 0              # all steps done — launcher should stop restarting
    INTERRUPTED  = signal.SIGTERM # == 15, caught SIGTERM: checkpoint saved, please restart
    CRASH        = 1              # unhandled exception — launcher auto-restarts from last checkpoint
    FATAL_CONFIG = 3             # non-recoverable config/arch error (e.g. checkpoint arch-family or
                                 # vf_coef/reward-config mismatch) — restarting would hit the SAME
                                 # error every time, so the launcher must NOT restart; it gives up
                                 # immediately and surfaces the reason instead of looping.
