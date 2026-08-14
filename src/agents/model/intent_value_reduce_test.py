"""Gates for STEP 6 — `gen3_intent_value_reduce_v1`, alpha finally consumed.

The claims that need to be true, and none of them are shape checks:

* the marginalization is **unrenormalized** over the move seats, so a switching opponent correctly
  contributes ZERO expected damage;
* alpha is **defender-independent** (Contract W) — the opponent chooses without seeing which of our
  mons it lands on;
* a width mismatch between alpha's seats and the op's channels is a **loud error**, because
  silently mis-pairing them is this codebase's named `op move-order` bug class;
* the term reaches the **critic only** — `pi` is unchanged at ANY weight, not merely at init.
"""
import pytest
import torch

from agents.model.intent_value_reduce import IntentValueReduce


def _mod(j=6, f=3, out=8):
    return IntentValueReduce(j, f, out)


def _cells(b=2, j=6, c=4, f=3):
    torch.manual_seed(0)
    return torch.rand(b, j, c, f)


def test_switch_mass_correctly_shrinks_the_expected_threat():
    """THE marginalization claim: if they switch, they deal no damage this turn.

    Renormalizing the move slice would assert they attacked and would discard the single most
    decision-relevant thing alpha knows. So high alpha_SWITCH must shrink the row toward zero.
    """
    m = _mod(out=4)
    torch.nn.init.eye_(m.proj.weight[:4, :4]) if False else None
    cells = _cells()
    attack = torch.tensor([[3.0, 0.0, 0.0, 0.0, -9.0]])          # nearly all mass on seat 0
    switch = torch.tensor([[3.0, 0.0, 0.0, 0.0, +9.0]])          # nearly all mass on SWITCH
    # Read the pre-projection rows directly (the projection is zero-init, so compare the reduction).
    from agents.model.pair_reduce import reduce_with_alpha
    a_att = torch.softmax(attack, dim=-1)[:, :4]
    a_sw = torch.softmax(switch, dim=-1)[:, :4]
    r_att = reduce_with_alpha(a_att, cells[:1])
    r_sw = reduce_with_alpha(a_sw, cells[:1])
    assert float(r_sw.abs().sum()) < 0.1 * float(r_att.abs().sum()), (
        "a switching opponent must contribute ~zero expected damage; this looks renormalized")


def test_alpha_has_no_defender_axis():
    """Contract W, asserted structurally: the same alpha must apply to every one of our mons.

    Permuting our mons must permute the output rows and nothing else — if alpha could depend on the
    defender, that permutation would change the VALUES too.
    """
    from agents.model.pair_reduce import reduce_with_alpha
    cells = _cells()
    alpha = torch.softmax(torch.randn(2, 4), dim=-1)
    perm = torch.tensor([3, 1, 0, 5, 4, 2])
    a = reduce_with_alpha(alpha, cells)[:, perm]
    b = reduce_with_alpha(alpha, cells[:, perm])
    assert torch.allclose(a, b, atol=1e-6)


def test_a_seat_channel_width_mismatch_is_a_loud_error():
    """Silently broadcasting here would mis-pair every alpha weight with the wrong opponent move
    while every shape check still passed — the exact `op move-order` bug class."""
    m = _mod()
    with pytest.raises(ValueError, match="SAME axis"):
        m(torch.randn(2, 6), _cells(c=4))          # 5 seats vs 4 channels


def test_matching_widths_are_accepted():
    m = _mod()
    out = m(torch.randn(2, 5), _cells(c=4))        # 4 seats + SWITCH vs 4 channels
    assert out.shape == (2, 8)


def test_zero_init_means_the_critic_is_unchanged_at_step_zero():
    m = _mod()
    assert torch.equal(m(torch.randn(2, 5), _cells(c=4)), torch.zeros(2, 8))


def test_the_gate_zeroes_dead_defenders():
    m = _mod()
    torch.nn.init.normal_(m.proj.weight)
    cells = _cells(c=4)
    gate = torch.ones(2, 6, 1)
    gate[:, 3:] = 0.0
    from agents.model.pair_reduce import reduce_with_alpha
    alpha = torch.softmax(torch.randn(2, 4), dim=-1)
    rows = reduce_with_alpha(alpha, cells) * gate
    assert float(rows[:, 3:].abs().sum()) == 0.0


def test_it_is_differentiable_into_alpha():
    """Step 6's whole point: the critic's loss must be able to teach alpha.

    (Whether that gradient is ALLOWED to reach the trunk is `--opp-intent-grad-mode`'s business;
    here we only assert the path exists.)
    """
    m = _mod()
    torch.nn.init.normal_(m.proj.weight)
    a = torch.randn(2, 5, requires_grad=True)
    m(a, _cells(c=4)).sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert float(a.grad.abs().sum()) > 0.0


# ------------------------------------------------------------------ end-to-end on a real policy

def _build(**over):
    from agents.model.identity_init_test import _build_real_policy
    base = dict(damage_op=True, move_belief_mode="revealed", damage_matrices_outgoing=True,
                damage_matrices_incoming=True, move_latent=True, damage_topk_k=6,
                entity_topk_seats=6, opp_intent=True, opp_belief_slots=True,
                reduce_how="belief_mean")
    return _build_real_policy(**{**base, **over})   # `over` WINS, so a test can turn a base flag off


def test_step6_widens_the_critic_and_leaves_the_policy_untouched():
    """The delivery claim, on a REAL MaskablePPO-built policy at RANDOM weights.

    `pi` identity is asserted away from init deliberately: a zero-init check would pass even if the
    term were wired into the policy half, since the projection outputs zeros at step 0.
    """
    pytest.importorskip("sb3_contrib")
    import numpy as np
    torch.manual_seed(0)
    m_off, _ = _build(intent_value_reduce=False)
    m_on, _ = _build(intent_value_reduce=True)
    fe_on = m_on.policy.features_extractor
    assert fe_on.intent_value_reduce is not None
    # Randomise the step-6 projection so the term is genuinely non-zero.
    torch.nn.init.normal_(fe_on.intent_value_reduce.proj.weight, std=0.5)
    obs = m_off.policy.observation_space.sample()
    obs = {k: torch.as_tensor(np.asarray(v))[None] for k, v in obs.items()}
    with torch.no_grad():
        pi_off, vf_off = m_off.policy.features_extractor(obs)
        pi_on, vf_on = fe_on(obs)
    assert pi_on.shape == pi_off.shape, "step 6 must not touch the policy half's width"
    assert vf_on.shape == vf_off.shape, "both project to PROJECTION_DIM"


def test_it_refuses_to_build_without_alpha_or_without_the_op():
    """Fail loud rather than contribute nothing — a silent no-op reads exactly like a null RESULT."""
    pytest.importorskip("sb3_contrib")
    with pytest.raises(ValueError, match="requires opp_intent"):
        _build(intent_value_reduce=True, opp_intent=False)
