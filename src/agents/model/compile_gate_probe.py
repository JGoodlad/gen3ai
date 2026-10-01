"""The compile gate's TRAIN-step probe loss and its gradient-COVERAGE guard (gen3_gate_grad_coverage_v1).

WHY THIS EXISTS. Until 2026-09-29 the learner compile gate's train-graph check backpropagated
``pi.square().mean() + vf.square().mean()`` — a loss that reaches only the extractor's own two
feature outputs. The pointer action head does not read those alone: it reads the extractor's
STASH (``stash.pointer_inputs`` — the request-ordered move seats, the team tokens and the per-action
move/switch cells), and the aux heads read further stash fields (the belief bank, the win-prob
logits). Every parameter that feeds ONLY those paths got a ZERO gradient from the gate's loss —
measured on the fresh production policy (CPU, the committed 64-row real-obs fixture): 52 of 232
extractor parameters (74 of 254 policy parameters); on the seeded perturbation 43 of 232 (65 of 254).
So a BACKWARD miscompile in any of those paths was unchecked on any weights: the forward readout
cannot see a backward-only defect, and the gradient cosine had nothing there to compare.

THE FIX has two halves, both here:

* `gate_loss` — the probe loss. The old features term, PLUS what the production PPO loss reads:
  the MASKED legal log-probabilities of the real pointer head (the rollout's sampling distribution
  and PPO's ratio), the critic value ``V`` (`policy._critic_value`, the win-prob sigmoid under
  ``critic='winprob'``) and the scalar ``value_net`` head, PLUS every graph-carrying tensor the
  extractor stashes for an aux loss (`_stash_tensors`). Each term is weighted by fixed,
  deterministic, sign-varying weights (`probe_weights` — no RNG, identical on every device and arm),
  so no term's gradient cancels by symmetry.
* `coverage_verdict` — the fail-closed GUARD: the fraction of the policy's parameters (extractor
  AND heads) that get an exactly-zero gradient from `gate_loss` must not exceed
  `MAX_ZERO_GRAD_FRACTION`, else `GradCoverageError` — a gradient check that compares nothing on a
  parameter cannot detect a defect there. Applied to the INFORMATIVE arm only (a fresh policy's
  zero-init heads legitimately zero many upstream gradients; the gate judges that arm on the
  seeded perturbation, `parity_probe.perturbed_parameters`).

THE BAR, measured (CPU, fresh production policy + seed-20260929 perturbation, 64 fixture rows):
the new loss leaves **1 of 254** policy parameters with zero gradient (0.39%) —
``features_extractor.edge_bias.c5_map.weight``, the Baton-Pass receiver edge family, whose input
cells are all zero because no fixture row has a Baton-Pass seat (a FIXTURE coverage limit, not a
loss one). The old loss left 65 of 254 (25.6%). ``MAX_ZERO_GRAD_FRACTION = 0.02`` (5 of 254) sits
5x above the healthy count and 13x below the defect, and names every zero parameter when it trips.

The per-parameter gradient rule (`per_param_grad_errors`) is the other half of the train-graph
check's sensitivity: the GLOBAL cosine is dominated by the largest gradients, so a defect in a
small path (the pointer head's inputs) can move it by less than its bar. Its bar and floor live in
`compile_trainer` beside the other tolerances.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import torch

#: See the module docstring for the measurement behind the bar.
MAX_ZERO_GRAD_FRACTION = 0.02


class GradCoverageError(RuntimeError):
    """Too many parameters get no gradient from the gate's probe loss to judge a backward."""


def probe_weights(shape: Sequence[int], salt: int, device: Any = None) -> torch.Tensor:
    """Fixed, deterministic, sign-varying weights of ``shape`` — ``sin(i * golden_angle + salt)``.

    No RNG (so training's stream is untouched and both arms and every device get the SAME weights)
    and never constant (so a weighted sum of log-probs, whose unweighted sum over a softmax has a
    structured gradient, cannot cancel)."""
    n = 1
    for s in shape:
        n *= int(s)
    i = torch.arange(n, dtype=torch.float64)
    w = torch.sin(i * 2.399963229728653 + float(salt) * 1.7).to(torch.float32)
    return w.reshape(tuple(int(s) for s in shape)).to(device)


def _walk(name: str, obj: Any) -> Iterator[Tuple[str, torch.Tensor]]:
    if isinstance(obj, torch.Tensor):
        if obj.requires_grad and obj.is_floating_point():
            yield name, obj
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            yield from _walk(f"{name}.{k}", obj[k])
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        for k in obj._fields:
            yield from _walk(f"{name}.{k}", getattr(obj, k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk(f"{name}[{i}]", v)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from _walk(f"{name}.{f.name}", getattr(obj, f.name))


#: Stash fields the sweep skips. ``pointer_inputs`` is reached by the REAL pointer head (below). The
#: rank probe's READOUT references (K6 ``trunk_tokens`` / ``value_cls``; K8 ``features_out``) are no
#: aux loss's input — the trunk feeds every path already, value_cls is the CLS pool's own output, the
#: features are the loss's own first term — and sweeping them DILUTES the per-parameter rule: their
#: large gradient on every upstream parameter shrank a backward-only pointer defect below the trained
#: bar (`compile_gate_probe_test`'s informative-weights refusal stopped raising, 2026-10-01).
_NOT_SWEPT = frozenset({"pointer_inputs", "trunk_tokens", "value_cls", "features_out"})


def _stash_tensors(fe: Any) -> List[Tuple[str, torch.Tensor]]:
    """Every graph-carrying float tensor in ``fe.stash`` (the per-forward side readouts the aux
    losses, the critic and the pointer head read), in a fixed order. ``pointer_inputs`` is left to
    the REAL pointer head (`gate_loss` reaches it through the masked legal log-probs, the path
    production trains), and the rank probe's readout references are not swept (`_NOT_SWEPT`)."""
    stash = getattr(fe, "stash", None)
    if stash is None or not dataclasses.is_dataclass(stash):
        return []
    out: List[Tuple[str, torch.Tensor]] = []
    for f in dataclasses.fields(stash):
        if f.name in _NOT_SWEPT:
            continue
        out.extend(_walk(f.name, getattr(stash, f.name)))
    return out


def has_policy_heads(policy: Any) -> bool:
    """Is this the Gen3 dual-head policy (the pointer-head seam and the critic read)?"""
    return all(hasattr(policy, a) for a in ("mlp_extractor", "_critic_value",
                                            "_get_action_dist_from_latent"))


def gate_loss(model: Any, fe: Any, obs: Any, legal_mask: Any
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward through whatever ``fe.forward`` is installed, and the probe loss over everything
    the production loss reads. Returns ``(loss, pi_features, vf_features)``. A stand-in policy (no
    heads) gets the features term and the stash sweep only."""
    pi, vf = fe(obs)
    loss = pi.float().square().mean() + vf.float().square().mean()
    policy: Any = getattr(model, "policy", None)
    if has_policy_heads(policy):
        rows = pi.shape[0]
        lp = policy.mlp_extractor.forward_actor(pi)
        lv = policy.mlp_extractor.forward_critic(vf)
        v = policy._critic_value(lv).float().flatten()
        loss = loss + (probe_weights(v.shape, 1, v.device) * v).sum() / rows
        value_net = getattr(policy, "value_net", None)
        if isinstance(value_net, torch.nn.Module):
            # The scalar head: in NO production loss graph under critic='winprob' (the win-prob
            # head is the critic there), included so the coverage guard needs no exclusion list.
            s = value_net(lv).float().flatten()
            loss = loss + (probe_weights(s.shape, 2, s.device) * s).sum() / rows
        dist = policy._get_action_dist_from_latent(lp)
        dist.apply_masking(legal_mask)
        logp = dist.distribution.logits.float()
        legal = torch.as_tensor(legal_mask, device=logp.device, dtype=torch.bool)
        w = probe_weights(logp.shape, 3, logp.device)
        loss = loss + torch.where(legal, logp * w, torch.zeros_like(logp)).sum() / rows
    for k, (_name, t) in enumerate(_stash_tensors(fe)):
        x = t.float()
        loss = loss + (probe_weights(x.shape, 10 + k, x.device) * x).mean()
    return loss, pi, vf


def grad_parameters(model: Any, fe: Any) -> List[Tuple[str, torch.nn.Parameter]]:
    """The parameters the gate's gradient is read over: the whole policy (extractor AND heads) when
    it is a module, else the extractor."""
    policy = getattr(model, "policy", None)
    mod = policy if isinstance(policy, torch.nn.Module) else fe
    # gen3_ridealong_heads_v1: the DETACHED ride-along heads (`policy.ridealong.*`) are in no
    # compiled graph and no production loss — they run eager on stop-grad inputs, with their own
    # optimizer — so the probe loss can never reach them and there is nothing to compare there.
    # Counting them would read as a coverage hole the gate cannot close.
    return [(n, p) for n, p in mod.named_parameters() if not n.startswith("ridealong.")]


def zero_grad_names(per_param_absmax: torch.Tensor, names: Sequence[str]) -> List[str]:
    """The parameters whose gradient is exactly zero (``not (absmax > 0)``, so NaN counts)."""
    return [n for n, a in zip(names, per_param_absmax.tolist()) if not (a > 0.0)]


def coverage_verdict(per_param_absmax: torch.Tensor, names: Optional[Sequence[str]] = None, *,
                     bar: float = MAX_ZERO_GRAD_FRACTION, where: str = "gate") -> str:
    """Raise `GradCoverageError` if more than ``bar`` of the parameters get a zero gradient from the
    probe loss; return the report line otherwise."""
    names = list(names) if names is not None else [f"p{i}" for i in range(len(per_param_absmax))]
    zero = zero_grad_names(per_param_absmax, names)
    n = max(len(names), 1)
    frac = len(zero) / n
    line = f"grad coverage: {len(zero)}/{n} parameters zero-grad ({frac:.2%} <= {bar:.0%})"
    if not (frac <= bar):
        raise GradCoverageError(
            f"{where}: GRAD COVERAGE — {len(zero)} of {n} parameters ({frac:.1%}) get an exactly "
            f"zero gradient from the gate's probe loss (bar {bar:.0%}), so a backward miscompile "
            f"in their paths cannot be detected. Zero: {zero[:12]}{' …' if len(zero) > 12 else ''}. "
            f"The probe loss must reach every path the production loss reads "
            f"(`agents.model.compile_gate_probe.gate_loss`).")
    return line + (f" [{', '.join(zero)}]" if zero else "")


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
