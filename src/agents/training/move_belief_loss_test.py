"""Unit tests for InstrumentedMaskablePPO._move_belief_loss (the MoveBelief reinjection-head loss).

Two disjoint slot populations, by mode:
- REVEALED slots (mode revealed|both): seen mons → DIRECT multi-label BCE on `known_moves` (slot
  identity is the revealed species; no matching).
- UNREVEALED slots (mode unrevealed|both): hidden mons → order-invariant (Hungarian) BCE on
  `belief_moves` (the believed slots are anonymous, so the k predictions are min-cost matched to the k
  movesets).

Tests cover: the None/empty guards, mode gating (a mode ignores the other population's labels),
the direct-BCE path, order-invariance + min-cost matching for the Hungarian path, 'both' scoring both
populations, grad flow, and FAIL-LOUD on out-of-vocab move ids.
"""
import pytest
import torch

from agents.training.instrumented_ppo import InstrumentedMaskablePPO

M = 40  # small move vocab
_loss = InstrumentedMaskablePPO._move_belief_loss


def _ml(B, requires_grad=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, 6, M, generator=g, requires_grad=requires_grad)


def test_returns_none_when_off_or_nothing_scorable():
    assert _loss(None, torch.zeros(2, 6, 4), torch.zeros(2, 6, 4), "both") is None
    ml = _ml(2)
    # both labels all-PAD → nothing scorable
    assert _loss(ml, torch.full((2, 6, 4), -1), torch.full((2, 6, 4), -1), "both") is None
    # off mode (never built) is handled upstream, but a missing-label both-None also → None
    assert _loss(ml, None, None, "both") is None


def test_revealed_mode_direct_bce_metrics_in_range():
    ml = _ml(2)
    km = torch.full((2, 6, 4), -1)
    km[0, 0, :2] = torch.tensor([3, 9])     # revealed slot 0 has 2 known moves
    km[1, 1, 0] = 5
    loss, m = _loss(ml, km, None, "revealed")
    assert torch.isfinite(loss) and m["bce"] >= 0
    assert 0.0 <= m["precision"] <= 1.0 and 0.0 <= m["recall"] <= 1.0
    assert m["revealed_slots"] == 2.0 and m["unrevealed_slots"] == 0.0


def test_revealed_mode_ignores_belief_moves():
    """revealed mode must NOT score the believed-slot (belief_moves) population."""
    ml = _ml(1)
    km = torch.full((1, 6, 4), -1); km[0, 0, 0] = 3
    bm = torch.full((1, 6, 4), -1); bm[0, 5, 0] = 7     # would be a hidden slot — must be ignored
    _, m = _loss(ml, km, bm, "revealed")
    assert m["unrevealed_slots"] == 0.0 and m["revealed_slots"] == 1.0


def test_unrevealed_mode_ignores_known_moves():
    ml = _ml(1)
    km = torch.full((1, 6, 4), -1); km[0, 0, 0] = 3     # revealed — must be ignored in unrevealed mode
    bm = torch.full((1, 6, 4), -1); bm[0, 5, 0] = 7
    _, m = _loss(ml, km, bm, "unrevealed")
    assert m["revealed_slots"] == 0.0 and m["unrevealed_slots"] == 1.0


def test_unrevealed_mode_order_invariant_under_slot_permutation():
    """Swapping which believed slot holds which moveset must NOT change the loss — the Hungarian
    re-assigns. This is the property the matched loss exists to provide (fixes the reveal-shift)."""
    ml = _ml(1, seed=3)
    bm1 = torch.full((1, 6, 4), -1); bm1[0, 4, 0] = 3; bm1[0, 5, 0] = 9
    bm2 = torch.full((1, 6, 4), -1); bm2[0, 4, 0] = 9; bm2[0, 5, 0] = 3   # same SET, swapped slots
    l1, _ = _loss(ml, None, bm1, "unrevealed")
    l2, _ = _loss(ml, None, bm2, "unrevealed")
    assert torch.allclose(l1, l2, atol=1e-6)


def test_unrevealed_mode_matching_picks_min_cost_assignment():
    """If slot 4's logits favour moveset B and slot 5's favour moveset A, the matched BCE must be the
    LOW cross-assignment cost, strictly below the naive same-order assignment."""
    ml = torch.full((1, 6, M), -5.0)
    A, B = 10, 22
    ml[0, 4, B] = 5.0   # slot 4 confidently predicts move B present
    ml[0, 5, A] = 5.0   # slot 5 confidently predicts move A present
    bm = torch.full((1, 6, 4), -1); bm[0, 4, 0] = A; bm[0, 5, 0] = B   # labels in the "wrong" slot order
    loss_matched, _ = _loss(ml, None, bm, "unrevealed")

    # naive same-slot-order BCE (slot4→{A}, slot5→{B}) — what the matching should beat
    def _bce(slot, present):
        t = torch.zeros(M); t[present] = 1.0
        return torch.nn.functional.binary_cross_entropy_with_logits(ml[0, slot], t, reduction="none").mean()
    naive = (_bce(4, A) + _bce(5, B)) / 2
    # matched picks the cross-assignment (slot4→{B}, slot5→{A}, both correct) → near-zero BCE, well
    # below the naive same-order cost (BCE magnitudes are small, so the margin is ~naive, not the CE 0.5).
    assert loss_matched.item() < naive.item() - 0.1


def test_both_mode_scores_disjoint_populations():
    ml = _ml(1, seed=2)
    km = torch.full((1, 6, 4), -1); km[0, 0, 0] = 3; km[0, 1, 0] = 4      # 2 revealed slots
    bm = torch.full((1, 6, 4), -1); bm[0, 5, 0] = 7                       # 1 hidden slot
    loss, m = _loss(ml, km, bm, "both")
    assert torch.isfinite(loss)
    assert m["revealed_slots"] == 2.0 and m["unrevealed_slots"] == 1.0


def test_grad_flows_to_logits():
    ml = _ml(2, requires_grad=True)
    km = torch.full((2, 6, 4), -1); km[0, 0, 0] = 3
    bm = torch.full((2, 6, 4), -1); bm[1, 5, 0] = 7
    loss = _loss(ml, km, bm, "both")[0]
    loss.backward()
    assert ml.grad.abs().sum() > 0


def test_all_pad_revealed_slot_not_supervised():
    """A revealed slot whose moveset is entirely unknown (all-PAD) must NOT be scored (would push the
    head toward predicting no moves)."""
    ml = _ml(1)
    km = torch.full((1, 6, 4), -1)   # every slot all-PAD
    assert _loss(ml, km, None, "revealed") is None


def test_fails_loud_on_out_of_vocab_revealed_move():
    ml = _ml(1)
    km = torch.full((1, 6, 4), -1); km[0, 0, 0] = M + 9
    with pytest.raises(ValueError, match="n_moves"):
        _loss(ml, km, None, "revealed")


def test_fails_loud_on_out_of_vocab_unrevealed_move():
    ml = _ml(1)
    bm = torch.full((1, 6, 4), -1); bm[0, 5, 0] = M + 3
    with pytest.raises(ValueError, match="n_moves"):
        _loss(ml, None, bm, "unrevealed")


def test_varied_k_across_batch_unrevealed():
    """Samples with different believed-slot counts k are all matched in one call."""
    ml = _ml(3, seed=1)
    bm = torch.full((3, 6, 4), -1)
    bm[0, 5, 0] = 4                              # k=1
    bm[1, 3, 0] = 7; bm[1, 4, 0] = 11; bm[1, 5, 0] = 2   # k=3
    bm[2, 2, 0] = 1; bm[2, 3, 0] = 9            # k=2
    loss, m = _loss(ml, None, bm, "unrevealed")
    assert torch.isfinite(loss)
    assert m["unrevealed_slots"] == 1 + 3 + 2


# ------------------------------------------------ gen3_unified_move_system_v1: latent-space grading
_latent_loss = InstrumentedMaskablePPO._move_belief_latent_loss


def _table(M, D=4):
    """Synthetic context-free latent table: move 0 = [1,0,0,0] (the 'truth'); move 1 = a NEAR move
    (cosine ≈ 1 to move 0); move 2 = a FAR/orthogonal move; the rest random-ish."""
    t = torch.zeros(M, D)
    t[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])     # e.g. Rock Slide
    t[1] = torch.tensor([0.96, 0.28, 0.0, 0.0])   # e.g. Hidden Power Rock (near)
    t[2] = torch.tensor([0.0, 1.0, 0.0, 0.0])     # e.g. Surf (far)
    for m in range(3, M):
        t[m, m % D] = 0.5
    return t


def _belief_on(M, move, hi=12.0, lo=-12.0):
    """ml [1,6,M] putting ~all softmax mass on `move` at slot 0 (else low)."""
    ml = torch.full((1, 6, M), lo)
    ml[0, 0, move] = hi
    return ml


def test_latent_loss_none_guards():
    assert _latent_loss(None, _table(8), torch.zeros(1, 6, 4)) is None
    assert _latent_loss(_belief_on(8, 0), None, torch.zeros(1, 6, 4)) is None
    assert _latent_loss(_belief_on(8, 0), _table(8), None) is None
    # all-PAD known_moves → nothing scorable
    assert _latent_loss(_belief_on(8, 0), _table(8), torch.full((1, 6, 4), -1)) is None


def test_latent_true_belief_high_cosine():
    """Believing the TRUE move (0) → the predicted expected latent matches the target → cosine ≈ 1."""
    km = torch.full((1, 6, 4), -1); km[0, 0, 0] = 0     # slot 0's true move is move 0
    _, m = _latent_loss(_belief_on(8, 0), _table(8), km)
    assert m["cosine"] > 0.99 and m["slots"] == 1.0


def test_latent_near_move_grades_better_than_far():
    """THE point (Rock Slide ≈ HP Rock): when the truth is move 0, a belief on the NEAR move (1) grades
    MUCH closer than a belief on the FAR move (2) — the soft credit the per-ID BCE can't give."""
    km = torch.full((1, 6, 4), -1); km[0, 0, 0] = 0
    _, near = _latent_loss(_belief_on(8, 1), _table(8), km)
    _, far = _latent_loss(_belief_on(8, 2), _table(8), km)
    assert near["cosine"] > 0.9          # near move ≈ the truth
    assert far["cosine"] < 0.2           # far move ≈ orthogonal
    assert near["cosine"] - far["cosine"] > 0.6


def test_latent_grad_flows_to_logits_and_table():
    """The loss is differentiable in BOTH the belief logits (sharpens the head) and the latent table
    (shapes the MoveLatentEncoder)."""
    km = torch.full((1, 6, 4), -1); km[0, 0, 0] = 0
    ml = _belief_on(8, 2).requires_grad_(True)    # believe the FAR move → non-zero gradient pressure
    table = _table(8).requires_grad_(True)
    loss, _ = _latent_loss(ml, table, km)
    loss.backward()
    assert ml.grad is not None and ml.grad.abs().sum() > 0
    assert table.grad is not None and table.grad.abs().sum() > 0
