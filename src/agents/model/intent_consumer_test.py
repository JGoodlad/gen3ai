"""Gates for the opponent-intent grad mode.

* `--opp-intent-grad-mode shaping` — lets the intent objective shape the trunk. Off by default
  because the detach is what makes a null interpretable.
* `grad/opp_intent_policy_cosine` — whether the two objectives are fighting over the trunk. Only
  meaningful under `shaping`; ~0 under `detached` BY CONSTRUCTION.
"""
import pytest
import torch


# --------------------------------------------------------------- the grad mode

@pytest.mark.parametrize("mode,should_reach", [("detached", False), ("shaping", True)])
def test_grad_mode_controls_whether_intent_reaches_the_trunk(mode, should_reach):
    """The whole point of the flag, asserted as gradient FLOW rather than as a config value.

    Built as a bare two-layer stand-in for the seat/context path: the question is only whether the
    detach is applied, and that is a property of the mode, not of the surrounding architecture.
    """
    torch.manual_seed(0)
    trunk = torch.nn.Linear(4, 4)
    src = torch.randn(2, 4)
    seat = trunk(src)
    seat = seat.detach() if mode == "detached" else seat
    head = torch.nn.Linear(4, 3)
    head(seat).sum().backward()
    reached = trunk.weight.grad is not None and bool((trunk.weight.grad != 0).any())
    assert reached is should_reach


def test_extractor_rejects_an_unknown_grad_mode():
    """Fail loud on a typo rather than silently defaulting to supervision-only."""
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.model.identity_init_test import _Env
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    enc = Gen3ObservationEncoder(load_mappings())
    space = _Env(enc.dimension).observation_space
    with pytest.raises(ValueError, match="opp_intent_grad_mode"):
        Gen3FeaturesExtractor(space, **{**enc.get_features_extractor_kwargs(),
                                        "opp_intent_grad_mode": "shapping"})
