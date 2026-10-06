"""`gen3_reference_state_released_v1` — the parity gate's EAGER reference leaves no per-forward state on a
served slot's replica (F-XC-3, `designs/research_state/measurements/x5_launchable_2026-10-05/`).

THE DEFECT. `policy_reference` ran the replica's own forward, and a forward REPLACES the extractor's and
the op's per-forward stashes (and `PokemonEncoder.last_move_tokens`, `EntitySeats.last_cand`) at entry.
Those tensors outlived the gate, sized by its LAST row count, so a T2 slot load — which gates every bucket
— left them allocated: the declared-load check (`rust_rollout.build.checked_slot_load`, 1 MiB) read them
as a NEW acquisition. Under fixed_mass the stashes hold ≈ 0.38 MB per row, so a few rows' difference
between the startup gate's last forward and a load's refused the load.

The first test fails on revert of `policy_reference`'s use of `forward_state_released`; the second pins the
helper itself against an unreleased forward of the same rows.
"""
from __future__ import annotations

import gc
import weakref
from typing import Any, List, Tuple

import pytest
import torch

from agents.inference.service.decision import _is_forward_state, forward_state_released, policy_reference


def _rows(model: Any, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rb = model.rollout_buffer
    o = rb.observations["observation"]
    obs = torch.as_tensor(o.reshape(-1, o.shape[-1])[:n])
    mask = torch.as_tensor(rb.action_masks.reshape(rb.actions.size, -1)[:n]).bool()
    return obs, mask


@pytest.fixture(scope="module", params=["blob", "fixed_mass"])
def arm(request) -> Tuple[str, Any]:
    from agents.training import learner_golden as LG
    m = LG.build_learner() if request.param == "blob" else LG.build_arm_learner(request.param)
    LG.load_buffer_into(m, LG.arm_buffer(request.param))
    m.policy.eval()
    return request.param, m


def _stash_tensors(policy: Any) -> List[torch.Tensor]:
    """Every tensor held by a per-forward attribute of any module of ``policy`` (stash fields walked)."""
    out: List[torch.Tensor] = []

    def walk(x: Any) -> None:
        if torch.is_tensor(x):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for y in x.values():
                walk(y)
        elif hasattr(x, "__dataclass_fields__"):
            for k in x.__dataclass_fields__:
                walk(getattr(x, k))
    for mod in policy.modules():
        for v in vars(mod).values():
            if v is not None and _is_forward_state(v):
                walk(v)
    return out


def test_the_eager_reference_leaves_no_per_forward_tensor_on_the_policy(arm):
    """After `policy_reference`, no per-forward attribute holds a tensor — neither the reference's own nor a
    stale one an earlier forward (another row count) left there."""
    name, m = arm
    obs, mask = _rows(m, 16)
    with torch.no_grad():
        m.policy.features_extractor({"observation": obs[:9]})      # a stale stash from another forward
    assert _stash_tensors(m.policy), "PRECONDITION: a forward leaves per-forward tensors on the policy"
    policy_reference(m.policy, obs, mask)
    left = _stash_tensors(m.policy)
    assert not left, (f"{name}: {len(left)} per-forward tensors remain after the reference forward "
                      f"(e.g. {[tuple(t.shape) for t in left[:4]]})")


def test_no_tensor_a_released_forward_produces_outlives_it(arm):
    """The same forward WITHOUT the release leaves its row-sized stash tensors alive; through
    `forward_state_released` none of its own survives, and the earlier forward's are released too."""
    name, m = arm
    obs, _mask = _rows(m, 16)
    fe = m.policy.features_extractor
    with torch.no_grad():
        fe({"observation": obs})                                    # an unreleased forward: its stash stays
    stale = _stash_tensors(m.policy)
    assert any(t.dim() and t.shape[0] == 16 for t in stale), \
        "PRECONDITION: an unreleased forward leaves row-sized stash tensors behind"
    left = [weakref.ref(t) for t in stale]
    del stale
    with torch.no_grad(), forward_state_released(m.policy):
        fe({"observation": obs[:5]})
        produced = [weakref.ref(t) for t in _stash_tensors(m.policy)]
    gc.collect()
    alive = [r for r in produced if r() is not None]
    assert not alive, f"{name}: {len(alive)} tensors from the released forward are still alive"
    assert not _stash_tensors(m.policy), f"{name}: per-forward state remains after the release"
    assert all(w() is None for w in left), f"{name}: the earlier forward's stash tensors are not released"
