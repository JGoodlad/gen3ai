"""The compile gates' per-parameter gradient helpers (gen3_gate_grad_coverage_v1).

* `grad_parameters` — the parameters a train-graph gradient comparison reads over (the whole
  policy, extractor AND heads; never the detached ride-along heads).
* `per_param_grad_errors` — the per-parameter relative gradient error. The GLOBAL cosine is
  dominated by the largest gradients, so a defect in a small path (the pointer head's inputs) can
  move it by less than its bar: on the perturbed production policy (CPU, 64 fixture rows) DROPPING
  the whole gradient through the pointer head's move cells left the cosine at 1.0000 while the
  per-parameter error read 0.10 on `damage_op.out_gain`. Its bar and floor live in `compile_trainer`.
* `has_policy_heads` — is this the Gen3 dual-head policy (the readout seam)?

The extractor-only gate's probe LOSS and its zero-gradient COVERAGE guard lived here until the
torch-2.5.1 extractor compile was deleted (2026-10-02): region R1 is judged on the production
micro-step's own loss, which reaches every path the production loss reads by construction.
"""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import torch


def has_policy_heads(policy: Any) -> bool:
    """Is this the Gen3 dual-head policy (the pointer-head seam and the critic read)?"""
    return all(hasattr(policy, a) for a in ("mlp_extractor", "_critic_value",
                                            "_get_action_dist_from_latent"))


def grad_parameters(model: Any, fe: Any) -> List[Tuple[str, torch.nn.Parameter]]:
    """The parameters the gate's gradient is read over: the whole policy (extractor AND heads) when
    it is a module, else the extractor."""
    policy = getattr(model, "policy", None)
    mod = policy if isinstance(policy, torch.nn.Module) else fe
    # gen3_ridealong_heads_v1: the DETACHED ride-along heads (`policy.ridealong.*`) are in no
    # compiled graph and no production loss — they run eager on stop-grad inputs, with their own
    # optimizer — so no gate loss reaches them and there is nothing to compare there.
    return [(n, p) for n, p in mod.named_parameters() if not n.startswith("ridealong.")]


def per_param_grad_errors(compiled: torch.Tensor, eager: torch.Tensor, sizes: Sequence[int], *,
                          floor_frac: float) -> List[Tuple[int, float]]:
    """``[(param index, relative error)]`` per parameter, for every parameter whose eager gradient
    norm is above ``floor_frac`` x the largest per-parameter norm — the relative error
    ``||c_p - e_p|| / ||e_p||``. Parameters under the floor are not judged (their gradient is
    reduction-order noise next to the rest, and the global cosine already weighs them)."""
    out: List[Tuple[int, float]] = []
    cs, es = torch.split(compiled.float(), list(sizes)), torch.split(eager.float(), list(sizes))
    norms = [float(e.norm()) for e in es]
    top = max(norms) if norms else 0.0
    for i, (c, e, ne) in enumerate(zip(cs, es, norms)):
        if top > 0.0 and ne > floor_frac * top:
            out.append((i, float((c - e).norm()) / ne))
    return out
