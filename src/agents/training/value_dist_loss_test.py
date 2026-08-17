"""Distributional value head HL-Gauss CE loss (v29) — the pure `_value_dist_loss` math.

Mirrors win_prob_test.py (the aux-loss test precedent). The head/version live in
agents/model/value_dist_head_test.py; this covers the loss + its interpretability diagnostics.
"""

import math

import torch

from agents.training.instrumented_ppo import InstrumentedMaskablePPO

BINS, VMIN, VMAX = 32, -5.0, 5.0
_loss = InstrumentedMaskablePPO._value_dist_loss


def _atoms(bins=BINS, vmin=VMIN, vmax=VMAX):
    return torch.linspace(vmin, vmax, bins)


def _nearest(atoms, x):
    return int(torch.argmin((atoms - x).abs()))


def test_none_inputs_return_none():
    assert _loss(None, torch.zeros(2), _atoms()) is None
    assert _loss(torch.zeros(2, BINS), None, _atoms()) is None
    assert _loss(torch.zeros(2, BINS), torch.zeros(2), None) is None


def test_shape_metrics_and_grad():
    logits = torch.zeros(4, BINS, requires_grad=True)
    out = _loss(logits, torch.zeros(4), _atoms())
    assert out is not None
    loss, m = out
    assert set(m) >= {"ce", "entropy", "std", "pit_mean", "mean_abs_err"}
    assert all(math.isfinite(v) for v in m.values())
    loss.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0


_SIGMA_G = 0.75 * (VMAX - VMIN) / (BINS - 1)   # the loss's HL-Gauss smoothing width


def _soft_logits(atoms, center, batch=2, scale=_SIGMA_G):
    """A soft-Gaussian logit row centered at `center` (≈ the HL-Gauss target shape when scale=σ_g) — NOT
    a one-hot, so CE against the *smoothed* target is well-behaved (a near-one-hot is penalised on the
    target's tails — the whole point of HL-Gauss vs two-hot)."""
    row = -((atoms - center) ** 2) / (2.0 * scale ** 2)
    return row.unsqueeze(0).repeat(batch, 1)


def test_ce_lower_when_centered_on_target():
    """CE is lower when the predicted distribution is centered ON the target than at the far edge
    (same width). The smoothed target rewards shape-matching, not a spike."""
    atoms = _atoms()
    target = torch.zeros(2)
    near = _soft_logits(atoms, 0.0)        # centered on target
    far = _soft_logits(atoms, VMAX)        # centered at the opposite edge
    assert float(_loss(near, target, atoms)[0]) < float(_loss(far, target, atoms)[0])


def test_mean_recovers_target():
    """E[Z] of a peaked head tracks the return target (within ~one bin width)."""
    atoms = _atoms()
    target = torch.tensor([2.0, -2.0])
    logits = torch.full((2, BINS), -10.0)
    for r in range(2):
        logits[r, _nearest(atoms, float(target[r]))] = 10.0
    _, m = _loss(logits, target, atoms)
    assert m["mean_abs_err"] < 0.5   # delta = 10/31 ≈ 0.32; one-hot mean is within ~delta/2


def test_out_of_support_is_finite():
    """Edge-bin tail absorption: a return far outside [vmin, vmax] reads as 'near the edge', not NaN."""
    atoms = _atoms()
    out = _loss(torch.zeros(2, BINS), torch.tensor([1000.0, -1000.0]), atoms)
    assert out is not None
    loss, m = out
    assert torch.isfinite(loss) and all(math.isfinite(v) for v in m.values())


def test_pit_mean_half_for_symmetric_centered():
    """Uniform logits + target at the support midpoint ⇒ PIT ≈ 0.5 (the calibration anchor)."""
    _, m = _loss(torch.zeros(8, BINS), torch.zeros(8), _atoms())
    assert abs(m["pit_mean"] - 0.5) < 0.1


def test_popart_normalized_target_space():
    """The caller passes the (PopArt-normalized) target; the loss is agnostic to the space — a
    normalized support [-5,5] with normalized targets ~N(0,1) trains the same as raw. Sanity: a
    distribution centered on each row's normalized target beats one centered at the far edge."""
    atoms = _atoms()
    norm_target = torch.tensor([0.3, -0.3])  # normalized returns ~ N(0,1)
    near = torch.stack([_soft_logits(atoms, float(t), batch=1)[0] for t in norm_target])
    far = _soft_logits(atoms, VMAX)
    assert float(_loss(near, norm_target, atoms)[0]) < float(_loss(far, norm_target, atoms)[0])
