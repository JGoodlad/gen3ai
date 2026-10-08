from dataclasses import dataclass
from typing import Optional

from agents.observation.constants import MAX_TURNS


@dataclass
class StallConfig:
    """Configuration for stall detection and logging."""
    # gen3_deadline_clock_v1: the forfeit deadline and the OBS clock's normaliser are ONE number.
    # They used to be independently-written 250s in two files, so moving the stall threshold would
    # have silently mis-scaled the obs `turns_remaining` scalars the critic prices the deadline
    # with — the classic producer/consumer drift. Pinned by
    # `observation/global_env_test.py::test_max_turns_is_the_forfeit_deadline`.
    threshold: int = MAX_TURNS
    output_dir: Optional[str] = None
