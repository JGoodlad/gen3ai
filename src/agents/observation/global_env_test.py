"""GlobalEnvEncoder tests — LiveView API (event-sourced weather + hazards + screens)."""



from agents.observation.global_env import GlobalEnvEncoder
from agents.observation.constants import GLOBAL_ENV_DIM, MAX_TURNS


def test_dimension_matches_layout():
    g = GlobalEnvEncoder()
    L = g.get_layout()
    assert g.dimension == GLOBAL_ENV_DIM
    assert sum(p["dim"] for p in L.values()) == GLOBAL_ENV_DIM


# --------------------------------------------------------------------------- #
# gen3_deadline_clock_v1 — the CLOCK group is [log_elapsed, remaining_linear, log_remaining]
# --------------------------------------------------------------------------- #
def test_max_turns_is_the_forfeit_deadline():
    """The obs clock normaliser and the turn the trainee actually FORFEITS on are ONE number.
    They were independently-written 250s in two files; moving the stall threshold would have
    silently mis-scaled the `turns_remaining` scalars the critic prices the deadline with."""
    from agents.training.stall import StallConfig
    assert StallConfig().threshold == MAX_TURNS
