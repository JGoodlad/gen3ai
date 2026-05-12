from enum import IntEnum

class LogLevel(IntEnum):
    """
    Cumulative logging levels for the RL training pipeline.
    Higher levels include all output from lower levels.
    """
    QUIET = 0       # Minimal output (Errors only)
    PERIODIC = 1    # Episode summaries
    DETAILED = 2    # Turn-by-turn rewards and subsidies
    DEBUG = 3       # Deep traces and full diagnostic data
