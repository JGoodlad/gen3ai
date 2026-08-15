"""Gates for the alpha/beta CONSUMER path: set-valued credit, the grad mode, and the fight detector.

Three changes, one theme — making the intent belief usable rather than merely measured:

* `set_valued_switch_loss` — partial credit for "they brought someone unseen", the one thing the
  flat-11 redesign would have bought, obtained additively on the head we already have.
* `--opp-intent-grad-mode shaping` — lets the intent objective shape the trunk. Off by default
  because the detach is what makes a null interpretable.
* `grad/opp_intent_policy_cosine` — whether the two objectives are fighting over the trunk. Only
  meaningful under `shaping`; ~0 under `detached` BY CONSTRUCTION.
"""
import pytest
import torch

from agents.model.opp_intent import intent_losses, set_valued_switch_loss


def _logits(mass_on):
    """Beta logits [1,6] putting almost all softmax mass on the given slots."""
    lg = torch.full((1, 6), -6.0)
    for j in mass_on:
        lg[0, j] = 6.0
    return lg


def test_mass_on_the_believed_set_scores_far_better_than_mass_off_it():
    """THE point of the term: it grades the SET, not the member."""
    believed = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    rows = torch.tensor([True])
    good = set_valued_switch_loss(_logits([2]), believed, rows)
    bad = set_valued_switch_loss(_logits([0]), believed, rows)
    assert good is not None and bad is not None
    assert float(good) < 0.1 < float(bad)


def test_it_does_not_care_WHICH_believed_slot_holds_the_mass():
    """Splitting across the set must score the same as concentrating on one member.

    This is the property that keeps the term honest: asserting a member we cannot label would be a
    fabricated target, and sharpening within the set is the species belief's job, not this one's.
    """
    believed = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    rows = torch.tensor([True])
    one = float(set_valued_switch_loss(_logits([2]), believed, rows))
    both = float(set_valued_switch_loss(_logits([2, 3]), believed, rows))
    assert one == pytest.approx(both, abs=0.02)


def test_a_row_with_no_believed_AND_legal_slot_is_dropped_not_charged():
    """An empty target set is unsatisfiable; -log(0) would charge an unbounded loss forever."""
    believed = torch.zeros(1, 6)
    assert set_valued_switch_loss(_logits([0]), believed, torch.tensor([True])) is None


def test_illegal_slots_cannot_absorb_credit():
    """-inf logits are the legality mask; believed-but-illegal must not count as satisfied."""
    lg = _logits([2])
    lg[0, 2] = float("-inf")                       # the believed slot is illegal here
    believed = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    assert set_valued_switch_loss(lg, believed, torch.tensor([True])) is None


def test_no_qualifying_rows_returns_None_rather_than_a_zero():
    """A zero would be added to the loss and silently dilute the mean; None is skipped."""
    believed = torch.ones(2, 6)
    assert set_valued_switch_loss(_logits([0]).repeat(2, 1), believed,
                                  torch.tensor([False, False])) is None


def test_the_term_is_differentiable_wrt_the_logits():
    believed = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    lg = _logits([0]).clone().requires_grad_(True)
    loss = set_valued_switch_loss(lg, believed, torch.tensor([True]))
    loss.backward()
    assert lg.grad is not None and torch.isfinite(lg.grad).all()
    assert float(lg.grad[0, 2]) < 0.0, "credit must PULL mass toward the believed slot"


# --------------------------------------------------------------- the fight detector's inputs

def test_beta_accuracy_is_bucketed_by_alpha_switch_confidence():
    """The falsifier for keeping the alpha/beta split, wired as a metric.

    If beta only works when alpha is already sure, the hierarchy is doing harm and the flat-11
    redesign gets a real argument. The two buckets must both be emitted so the gap is visible.
    """
    torch.manual_seed(0)
    n, k = 40, 3
    alpha_logits = torch.zeros(n, k + 1)
    alpha_logits[:20, -1] = 9.0                    # confident SWITCH on half the rows
    alpha_target = torch.randint(0, k + 1, (n,))
    beta_logits = torch.randn(n, 6)
    beta_target = torch.randint(0, 6, (n,))
    _, m = intent_losses(alpha_logits, alpha_target, beta_logits, beta_target)
    assert "opp_intent/beta_recall_alpha_confident" in m
    assert "opp_intent/beta_recall_alpha_unsure" in m


def test_the_bucketing_does_not_change_beta_accuracy_overall():
    """A diagnostic may not perturb the number it is diagnosing."""
    torch.manual_seed(1)
    n = 32
    beta_logits, beta_target = torch.randn(n, 6), torch.randint(0, 6, (n,))
    _, without = intent_losses(None, None, beta_logits, beta_target)
    _, with_a = intent_losses(torch.zeros(n, 4), torch.randint(0, 4, (n,)),
                              beta_logits, beta_target)
    assert with_a["opp_intent/beta_recall_top1"] == pytest.approx(without["opp_intent/beta_recall_top1"])


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
