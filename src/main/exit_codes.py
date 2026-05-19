import signal
from enum import IntEnum


class TrainExitCode(IntEnum):
    COMPLETE    = 0              # all steps done — launcher should stop restarting
    INTERRUPTED = signal.SIGTERM # == 15, caught SIGTERM: checkpoint saved, please restart
    CRASH       = 1              # unhandled exception — launcher should not restart
