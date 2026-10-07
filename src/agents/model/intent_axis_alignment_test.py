"""The opponent-intent pointer's seat axis and the op's candidate axis must be THE SAME axis, in the same order.

This is the prerequisite for every α consumer and it is a load-bearing assumption. Since the X5 version break
(v144) the intent readout is X5's FLAT pointer: its K move seats are the opponent active's move group
(`fe.last_flat_intent.seat_nums`), and the op's top-K (`op.last_topk_idx`) is built from the SAME move group.
Those are two different objects that are BELIEVED to agree. (The blob path's α read `entity_seats.last_cand[0]`
— deleted with it.)

Why it matters more than it looks. Every α consumer weights the op's believed-move axis by the pointer:

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
    """One real forward with the op + E4 seats + the intent pointer all live (the shared toggle set has
    the X5 opponent-belief family on). Returns the extractor."""
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


def test_the_intent_pointers_seat_nums_are_the_ops_topk_in_the_same_order():
    """THE gate. Same move nums, same positions — not merely the same SET.

    A set-equality check would pass under a permutation, which is the failure mode that matters.
    """
    pytest.importorskip("sb3_contrib")
    fe = _forward_with_intent()
    fi = fe.last_flat_intent
    seat_nums = fi.seat_nums if fi is not None else None
    topk = fe.damage_op.last_topk_idx if fe.damage_op is not None else None
    # the blob α's stash is never written since the X5 break: a consumer reading it would read None
    assert not hasattr(fe, "last_alpha_seat_nums")              # the blob α stash is DELETED (part 2)
    # NOT a skip. `_forward_with_intent` ASKS for `opp_intent` + `damage_topk_k=6` explicitly, so a
    # build without the intent pointer or an op top-K is a broken build, not an inapplicable one — and a
    # skip here would silently retire THE gate for a named bug class that has bitten before
    # (`project_op_move_order_bugclass`).
    assert seat_nums is not None and topk is not None, (
        f"the config this test constructs did not produce what it asked for "
        f"(flat pointer seat_nums={'present' if seat_nums is not None else 'MISSING'}, "
        f"op top-K={'present' if topk is not None else 'MISSING'}) — the axis-alignment gate "
        f"cannot run, so fix the build rather than skipping past it")
    assert seat_nums.shape == topk.shape, (seat_nums.shape, topk.shape)
    assert torch.equal(seat_nums.long(), topk.long()), (
        "the intent pointer's seat axis is NOT the op's candidate axis in the same order — an α-weighted "
        "reduction over the op's move axis would pair every term with the wrong opponent move.\n"
        f"seats: {seat_nums[0].tolist()}\ntopk : {topk[0].tolist()}"
    )


def test_the_check_would_catch_a_permutation():
    """Prove the assertion is falsifiable rather than trivially true on this data."""
    a = torch.tensor([[10, 20, 30, 40, 50, 60]])
    assert torch.equal(a, a.clone())
    assert not torch.equal(a, a[:, torch.tensor([1, 0, 2, 3, 4, 5])]), \
        "a permuted axis must compare UNEQUAL, or the gate is decorative"
