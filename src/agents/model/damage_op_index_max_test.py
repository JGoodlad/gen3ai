"""`gen3_fm_index_max_v1` — fixed_mass's incoming channel maxima are selected BY INDEX; blob's are not (F-XC-4).

THE DEFECT (`designs/research_state/measurements/x5_fxc4_nanfix_2026-10-05/`): `amax`'s backward is
``grad · (x == amax) / Σ(x == amax)``. Inductor RECOMPUTED fixed_mass's 400-wide incoming sweep in the
backward kernel, Triton's FMA contraction rounded it differently from the forward kernel that took the
max, no element equalled the saved max on some rows, and the compiled R1 gradient was 0/0 = NaN on 41
parameters (CUDA; finite with FMA contraction off). `damage_op.max_by_index` takes the value at the
detached argmax: bit-identical to `amax`, and its backward is a scatter at a saved index.

These CPU tests pin the SCOPE and the ARITHMETIC; the CUDA fact (the compiled gradient is finite, the
real gate passes) is `compile_regions_fixed_mass_cuda_test.py` (GPU tier).

* the blob forward NEVER selects by index (its compiled and eager arithmetic are the X5 A/B's blob
  seeds', §7.5) — and the fixed_mass forward does, ten times per forward (fails on revert);
* `max_by_index` is `amax` bit for bit in value, its gradient equals `amax`'s off ties, and on a tie it
  goes whole to the FIRST maximal element (`argmax`'s documented tie rule; `amax` splits it);
* the fixed_mass FORWARD is bit-identical to the `amax` spelling on the K9 golden's real rows."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest
import torch

from agents.model import damage_op as DO


def _forward_rows(model: Any) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    from stable_baselines3.common.utils import obs_as_tensor
    rb = model.rollout_buffer
    obs = obs_as_tensor({k: v.reshape((-1,) + v.shape[2:])[:32] for k, v in rb.observations.items()}, "cpu")
    acts = torch.as_tensor(rb.actions.reshape(-1)[:32]).long()
    masks = torch.as_tensor(rb.action_masks.reshape(rb.actions.size, -1)[:32])
    return obs, acts, masks


def _evaluate(model: Any, rows: Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]) -> List[torch.Tensor]:
    obs, acts, masks = rows
    model.policy.set_training_mode(False)
    with torch.no_grad():
        values, logp, ent = model.policy.evaluate_actions(obs, acts, action_masks=masks)
    return [values, logp, ent]


@pytest.fixture(scope="module")
def blob() -> Any:
    from agents.training import learner_golden as LG
    m = LG.build_learner()
    LG.load_buffer_into(m)
    return m


@pytest.fixture(scope="module")
def fixed_mass() -> Any:
    from agents.training import learner_golden as LG
    m = LG.build_arm_learner("fixed_mass")
    LG.load_buffer_into(m, LG.arm_buffer("fixed_mass"))
    assert m.policy.features_extractor.hypothesis_builder is not None        # PRECONDITION: the X5 arm
    return m


def _counting(monkeypatch: Any) -> List[int]:
    calls = [0]
    real = DO.max_by_index

    def counted(x: torch.Tensor) -> torch.Tensor:
        calls[0] += 1
        return real(x)
    monkeypatch.setattr(DO, "max_by_index", counted)
    return calls


def test_the_blob_forward_never_selects_by_index(blob, monkeypatch):
    """§7.5: blob's arithmetic is the trained blob seeds'. A blob forward that reached `max_by_index`
    would change its compiled backward (the K9 blob golden and the code hashes would move too)."""
    calls = _counting(monkeypatch)
    _evaluate(blob, _forward_rows(blob))
    assert calls[0] == 0, f"the blob forward called max_by_index {calls[0]}x"


def test_the_fixed_mass_forward_selects_its_ten_incoming_channel_maxima_by_index(fixed_mass, monkeypatch):
    """Fails on revert (the `amax` spelling: 0 calls)."""
    calls = _counting(monkeypatch)
    _evaluate(fixed_mass, _forward_rows(fixed_mass))
    assert calls[0] == 10, f"max_by_index ran {calls[0]}x in one fixed_mass forward (want the 10 channel maxima)"


def test_the_fixed_mass_forward_is_bit_identical_to_the_amax_spelling(fixed_mass, monkeypatch):
    rows = _forward_rows(fixed_mass)
    by_index = _evaluate(fixed_mass, rows)
    monkeypatch.setattr(DO, "max_by_index", lambda x: x.amax(dim=-1))
    by_amax = _evaluate(fixed_mass, rows)
    for a, b in zip(by_index, by_amax):
        assert torch.equal(a, b)


def test_max_by_index_is_amax_in_value_and_off_ties_in_gradient_and_FIRST_on_a_tie():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(5, 6, 400, generator=g)
    x[0, 0, 7] = x[0, 0, 3] = 2.0                         # an exact tie at the max: positions 3 and 7
    x[1, 2, :] = 0.0                                      # an all-zero row (every element ties)
    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    va, vb = DO.max_by_index(a), b.amax(dim=-1)
    assert torch.equal(va, vb)                            # the VALUE: bit-identical
    w = torch.rand(va.shape, generator=g)
    (va * w).sum().backward()
    (vb * w).sum().backward()
    ga, gb = a.grad, b.grad
    assert ga is not None and gb is not None
    tie_rows = torch.zeros(5, 6, dtype=torch.bool)
    tie_rows[0, 0] = tie_rows[1, 2] = True
    assert torch.equal(ga[~tie_rows], gb[~tie_rows])      # off ties: the same gradient, bit for bit
    assert float(ga[0, 0, 3]) == float(w[0, 0]) and float(ga[0, 0, 7]) == 0.0        # FIRST maximal element
    assert float(gb[0, 0, 3]) == float(gb[0, 0, 7]) == pytest.approx(float(w[0, 0]) / 2)  # amax splits
    assert float(ga[1, 2, 0]) == float(w[1, 2]) and float(ga[1, 2, 1:].abs().sum()) == 0.0
