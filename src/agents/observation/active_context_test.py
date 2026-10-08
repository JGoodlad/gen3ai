"""ActiveContextEncoder tests — LiveView API (boosts + full gen3 volatile set)."""


from agents.observation.active_context import ActiveContextEncoder
from agents.observation.constants import ACTIVE_CONTEXT_DIM, BOOSTS_DIM
from agents.observation.gen3_effects import VOLATILE_DIM


def test_dimension():
    assert ActiveContextEncoder().dimension == ACTIVE_CONTEXT_DIM == BOOSTS_DIM + VOLATILE_DIM
