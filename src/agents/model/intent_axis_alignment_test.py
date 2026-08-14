"""α's seat axis and the op's candidate axis must be THE SAME axis, in the same order.

This is the prerequisite for step 6 and it is a load-bearing assumption nobody had checked.
`opp_intent.match_seats_to_move_num`'s docstring says `seat_nums` are "the move nums the op's
top-K actually holds (`op.last_topk_idx`)" — and the extractor in fact passes
`entity_seats.last_cand[0]`. Those are two different objects that are BELIEVED to agree.

Why it matters more than it looks. Step 6 weights the op's believed-move axis by α:

    reduced[b, j] = Σ_k α[b, k] · outcome[b, j, k]

That expression is only meaningful if `α[b,k]` and `outcome[b,j,k]` index the same opponent move.
If the two axes are permutations of each other, every term is silently mis-paired: the arithmetic
runs, the shapes agree, no gate fires, and the model is trained on a fluent lie. This project has a
NAMED bug class for exactly this (`project_op_move_order_bugclass`: action-aligned consumers must
source their ordering from one place), and it has bitten before.

So the invariant is asserted on a REAL forward rather than argued from docstrings.
"""
import numpy as np
import pytest
import torch


def _forward_with_intent():
    """One real forward with the op + E4 seats + α all live. Returns the extractor."""
    from agents.model.identity_init_test import _build_real_policy
    model, _enc = _build_real_policy(
        damage_op=True, move_belief_mode="revealed", damage_matrices_outgoing=True,
        entity_topk_seats=6, opp_intent=True, opp_belief_slots=True,
        # Production regime: K is the INCOMING matrix's width, and the op only populates
        # last_topk_idx when it actually truncates. Both companions are required by the
        # extractor's own fail-loud guards.
        damage_topk_k=6, damage_matrices_incoming=True, move_latent=True,
    )
    fe = model.policy.features_extractor
    obs = model.policy.observation_space.sample()
    obs = {k: torch.as_tensor(np.asarray(v))[None] for k, v in obs.items()}
    with torch.no_grad():
        fe(obs)
    return fe


def test_alpha_seat_nums_are_the_ops_topk_in_the_same_order():
    """THE gate. Same move nums, same positions — not merely the same SET.

    A set-equality check would pass under a permutation, which is the failure mode that matters.
    """
    pytest.importorskip("sb3_contrib")
    fe = _forward_with_intent()
    seat_nums = getattr(fe, "last_alpha_seat_nums", None)
    topk = getattr(getattr(fe, "damage_op", None), "last_topk_idx", None)
    if seat_nums is None or topk is None:
        pytest.skip("this config built no alpha head or no op top-K")
    assert seat_nums.shape == topk.shape, (seat_nums.shape, topk.shape)
    assert torch.equal(seat_nums.long(), topk.long()), (
        "alpha's seat axis is NOT the op's candidate axis in the same order — an alpha-weighted "
        "reduction over the op's move axis would pair every term with the wrong opponent move.\n"
        f"seats: {seat_nums[0].tolist()}\ntopk : {topk[0].tolist()}"
    )


def test_the_check_would_catch_a_permutation():
    """Prove the assertion is falsifiable rather than trivially true on this data."""
    a = torch.tensor([[10, 20, 30, 40, 50, 60]])
    assert torch.equal(a, a.clone())
    assert not torch.equal(a, a[:, torch.tensor([1, 0, 2, 3, 4, 5])]), \
        "a permuted axis must compare UNEQUAL, or the gate is decorative"
