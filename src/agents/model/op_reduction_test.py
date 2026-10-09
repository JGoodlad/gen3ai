"""`op_reduction.py` — the PRINCIPLED reductions (`--op-reduction principled`, architecture audit F6b) on
CONSTRUCTED rows, every expected number written out by hand.

Each test names what a revert breaks:

* the EXPECTATION is the hand-computed α-weighted sum with α = presence / total presence (fails if α is
  renormalised over a top-K slice, if a channel uses a different weight, if the max comes back);
* the NOISY-OR is ``1 − Π(1 − p)``, exact at presence 0 (no contribution) and presence 1 (a certain KO reads
  exactly 1) with a finite gradient there (fails on a max, a sum, a log-space form that NaNs at p = 1);
* COHERENCE: on a planted row where the old per-channel maxima pick DIFFERENT moves for high, KO and accuracy,
  every principled channel is ONE α's contraction of the same candidates (Contract W's
  `pair_reduce.reduce_with_alpha`) — fails if any channel is re-selected per channel;
* PERMUTATION invariance over the candidate axis, and SMOOTHNESS at a tie: two tied candidates swapped leave
  every output unchanged and split the gradient evenly (the max hands the whole gradient to the first);
* ``believed_reduce`` with no total IS ``max_by_index`` (the `max` identity at the per-attacker kernels).
"""
from __future__ import annotations

import pytest
import torch

from agents.model.index_max import max_by_index
from agents.model.op_reduction import (OP_REDUCTION_MODES, believed_reduce, expectation, incoming_principled,
                                       noisy_or, presence_alpha)
from agents.model.pair_reduce import reduce_with_alpha

_NAMES = ("phys_low", "phys_high", "phys_crit", "phys_pko", "phys_acc", "spec_low", "spec_high", "spec_crit",
          "spec_pko", "spec_acc", "provenance", "phys_high_cb", "phys_pko_cb")


def _row(c: int = 3, seed: int = 0):
    """One battle, six of our mons, ``c`` believed candidates; values in [0, 1] (rolls) as the op emits them."""
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(1, c, generator=g, dtype=torch.float64)
    high = torch.rand(1, 6, c, generator=g, dtype=torch.float64)
    low = 0.85 * high
    crit = (2.0 * high).clamp(max=3.0)
    ko = torch.rand(1, 6, c, generator=g, dtype=torch.float64) * 0.5
    acc = torch.tensor([[1.0, 0.85, 0.9, 0.7, 1.0][:c]], dtype=torch.float64)
    phys = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0][:c]], dtype=torch.float64)
    high_cb, ko_cb = 1.5 * high, (ko * 1.5).clamp(max=1.0)
    return dict(w=w, w_total=None, low=low, high=high, crit=crit, ko=ko, acc=acc, phys=phys,
                high_cb=high_cb, ko_cb=ko_cb)


def _out(r):
    o = incoming_principled(**r)
    return dict(zip(_NAMES, o[:13])), o[13]


def test_the_modes_are_max_then_principled():
    assert OP_REDUCTION_MODES == ("max", "principled")


def test_the_expectation_is_the_hand_computed_alpha_weighted_sum():
    # three candidates: Earthquake (phys), Surf (spec), Rock Slide (phys), presences 1.0 / 0.6 / 0.4.
    w = torch.tensor([[1.0, 0.6, 0.4]], dtype=torch.float64)
    high = torch.zeros(1, 6, 3, dtype=torch.float64)
    high[0, 0] = torch.tensor([0.50, 0.30, 0.20])
    ko = torch.zeros(1, 6, 3, dtype=torch.float64)
    ko[0, 0] = torch.tensor([0.10, 0.0, 0.0])
    acc = torch.tensor([[1.0, 1.0, 0.9]], dtype=torch.float64)
    phys = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)
    r = dict(w=w, w_total=None, low=0.85 * high, high=high, crit=2 * high, ko=ko, acc=acc, phys=phys,
             high_cb=1.5 * high, ko_cb=ko)
    o, worst = _out(r)
    a = [1.0 / 2.0, 0.6 / 2.0, 0.4 / 2.0]                       # α = w / Σw, Σw = 2.0
    assert o["phys_high"][0, 0].item() == pytest.approx(a[0] * 0.50 + a[2] * 0.20)
    assert o["spec_high"][0, 0].item() == pytest.approx(a[1] * 0.30)
    assert o["phys_low"][0, 0].item() == pytest.approx(0.85 * (a[0] * 0.50 + a[2] * 0.20))
    assert o["phys_pko"][0, 0].item() == pytest.approx(a[0] * 0.10)
    # accuracy: P(their α-click is a damaging physical move that hits mon 0)
    assert o["phys_acc"][0, 0].item() == pytest.approx(a[0] * 1.0 + a[2] * 0.9)
    assert o["spec_acc"][0, 0].item() == pytest.approx(a[1] * 1.0)
    # provenance: Σ α_c w_c over the damaging candidates
    assert o["provenance"][0, 0].item() == pytest.approx(a[0] * 1.0 + a[1] * 0.6 + a[2] * 0.4)
    assert o["phys_high_cb"][0, 0].item() == pytest.approx(1.5 * (a[0] * 0.50 + a[2] * 0.20))
    # mon 1 takes no damage from anything (immune / no candidate lands): every fact is 0, never NaN
    for k, v in o.items():
        assert v[0, 1].item() == 0.0, k
    # worst case: P(Earthquake KOs) through the physical channel only
    assert worst[0, 0, 0].item() == pytest.approx(1.0 * 0.10)
    assert worst[0, 0, 1].item() == 0.0


def test_alpha_is_normalised_by_the_whole_axis_total_not_the_slice():
    """A top-K slice of a 4-move set: the cut mass is unpriced, never renormalised away."""
    w = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    total = torch.tensor([[4.0]], dtype=torch.float64)
    a = presence_alpha(w, total)
    assert torch.equal(a, torch.tensor([[0.25, 0.25]], dtype=torch.float64))
    v = torch.tensor([[[0.4, 0.8]]], dtype=torch.float64)                       # [B,1,K]
    assert believed_reduce(w[:, None, :] * v, total).item() == pytest.approx(0.3)                 # (0.4 + 0.8) / 4


def test_an_all_zero_presence_row_reads_zero_alpha_not_nan():
    a = presence_alpha(torch.zeros(2, 5))
    assert torch.equal(a, torch.zeros(2, 5))


def test_the_noisy_or_is_one_minus_the_product_of_misses():
    p = torch.tensor([0.2, 0.5], dtype=torch.float64)
    assert noisy_or(p).item() == pytest.approx(1.0 - 0.8 * 0.5)
    assert noisy_or(torch.zeros(7)).item() == 0.0                               # nothing can KO
    assert noisy_or(torch.tensor([0.3, 1.0, 0.1])).item() == 1.0               # a certain KO is certain


def test_the_noisy_or_presence_edges_and_a_finite_gradient_at_certainty():
    """presence 0 contributes nothing whatever its KO; presence 1 with KO 1 reads exactly 1 — through the
    incoming row's worst case — and the gradient at p = 1 is finite (no log 0)."""
    r = _row(c=3)
    r["ko"] = torch.zeros_like(r["ko"])
    r["ko"][0, 2, 0] = 1.0                                     # candidate 0 surely KOs mon 2 (phys)
    r["w"] = torch.tensor([[0.0, 0.5, 0.5]], dtype=torch.float64)
    _, worst = _out(r)
    assert worst[0, 2, 0].item() == 0.0, "a move believed ABSENT cannot KO"
    r["w"] = torch.tensor([[1.0, 0.5, 0.5]], dtype=torch.float64)
    _, worst = _out(r)
    assert worst[0, 2, 0].item() == 1.0, "a REVEALED move that surely KOs is a certain KO"
    p = torch.tensor([1.0, 0.4, 0.2], requires_grad=True)
    noisy_or(p).backward()
    assert torch.isfinite(p.grad).all()
    assert p.grad.tolist() == pytest.approx([0.6 * 0.8, 0.0, 0.0])


def test_coherence_every_channel_is_one_alphas_contraction_where_max_picked_different_moves():
    """PLANT: candidate 0 has the biggest high roll, candidate 2 the biggest KO chance (a low-power but
    accurate finisher into a low-HP mon), candidate 1 the best accuracy — so the legacy maxima describe
    mon 0 with THREE different moves."""
    w = torch.tensor([[0.9, 0.9, 0.9]], dtype=torch.float64)
    high = torch.zeros(1, 6, 3, dtype=torch.float64)
    ko = torch.zeros(1, 6, 3, dtype=torch.float64)
    high[0, 0] = torch.tensor([0.9, 0.5, 0.6])
    ko[0, 0] = torch.tensor([0.30, 0.20, 0.55])
    acc = torch.tensor([[0.7, 1.0, 0.9]], dtype=torch.float64)
    phys = torch.ones(1, 3, dtype=torch.float64)
    r = dict(w=w, w_total=None, low=0.85 * high, high=high, crit=2 * high, ko=ko, acc=acc, phys=phys,
             high_cb=1.5 * high, ko_cb=ko)
    # PRECONDITION (the defect): the max path's argmaxes disagree
    wh, wk = w[:, None, :] * high, w[:, None, :] * ko
    assert int(wh[0, 0].argmax()) == 0 and int(wk[0, 0].argmax()) == 2 and int(acc[0].argmax()) == 1
    o, _ = _out(r)
    alpha = presence_alpha(w)                                                    # ONE distribution, no J / F axis
    cells = torch.stack([0.85 * high, high, 2 * high, ko, acc[:, None, :].expand(1, 6, 3)], dim=-1)  # [B,J,C,F]
    want = reduce_with_alpha(alpha, cells)                                      # Contract W: Σ_c α_c cell[j,c,:]
    got = torch.stack([o["phys_low"], o["phys_high"], o["phys_crit"], o["phys_pko"], o["phys_acc"]], dim=-1)
    torch.testing.assert_close(got[0, 0], want[0, 0], rtol=0, atol=1e-12)
    # and it is NOT any single move's row (the max path's composite): every channel mixes all three
    assert all(abs(got[0, 0, f].item() - cells[0, 0, c, f].item()) > 1e-3 for f in range(5) for c in range(3))


def test_permutation_invariance_over_the_candidate_axis():
    r = _row(c=5, seed=3)
    perm = torch.tensor([3, 0, 4, 1, 2])
    q = dict(r)
    for k in ("w", "acc", "phys"):
        q[k] = r[k][:, perm]
    for k in ("low", "high", "crit", "ko", "high_cb", "ko_cb"):
        q[k] = r[k][:, :, perm]
    a, wa = _out(r)
    b, wb = _out(q)
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=1e-12)
    torch.testing.assert_close(wa, wb, rtol=0, atol=1e-12)


def test_a_tie_is_no_tie_swapped_tied_candidates_leave_everything_unchanged_and_split_the_gradient():
    """Two candidates with the SAME presence and the SAME value: swapping them leaves every output unchanged
    and the gradient is symmetric. The max hands the whole gradient to the first (`max_by_index`'s
    declared convention) — the discontinuity the principled reduction removes."""
    r = _row(c=4, seed=5)
    r["w"][0, 1] = 1.0                                      # the tied pair is the top of every channel's max
    r["high"][:, :, 1] = 1.0
    r["w"][0, 2] = r["w"][0, 1]
    for k in ("low", "high", "crit", "ko", "high_cb", "ko_cb"):
        r[k][:, :, 2] = r[k][:, :, 1]
    r["acc"][0, 2], r["phys"][0, 2] = r["acc"][0, 1], r["phys"][0, 1]
    sw = torch.tensor([0, 2, 1, 3])
    q = {k: (v[:, sw] if k in ("w", "acc", "phys") else (v[:, :, sw] if v is not None and v.dim() == 3 else v))
         for k, v in r.items()}
    a, wa = _out(r)
    b, wb = _out(q)
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=1e-12)
    torch.testing.assert_close(wa, wb, rtol=0, atol=1e-12)
    high = r["high"].clone().requires_grad_(True)
    rr = dict(r, high=high)
    o, _ = _out(rr)
    chan = "phys_high" if bool(r["phys"][0, 1] == 1.0) else "spec_high"
    o[chan].sum().backward()
    g = high.grad[0]
    torch.testing.assert_close(g[:, 1], g[:, 2], rtol=0, atol=0)
    hm = r["high"].clone().requires_grad_(True)
    max_by_index(r["w"][:, None, :] * hm).sum().backward()
    gm = hm.grad[0]
    assert bool((gm[:, 1] != gm[:, 2]).any()), "PRECONDITION: the max splits a tie unevenly (first wins)"


def test_believed_reduce_without_a_total_is_the_legacy_max_bit_for_bit():
    g = torch.Generator().manual_seed(11)
    wv = torch.rand(3, 6, 7, generator=g)
    assert torch.equal(believed_reduce(wv, None), max_by_index(wv))
    total = torch.rand(3, 6, generator=g) + 1.0
    torch.testing.assert_close(believed_reduce(wv, total), wv.sum(-1) / total)


def test_expectation_reduces_the_named_dim():
    a = torch.tensor([[0.5, 0.5]])
    v = torch.tensor([[[1.0, 3.0]]])
    assert expectation(a[:, None, :], v).item() == 2.0
