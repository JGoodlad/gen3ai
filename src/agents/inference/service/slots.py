"""SLOT GROUPS: one architecture, a fixed number of weight slots in STACKED storage.

A slot is a private replica of the group's template policy whose persistent parameters and buffers
are VIEWS into one ``[n_slots, ...]`` tensor per unique state-dict tensor, allocated once at startup.
Loading a weight set is an in-place ``copy_`` into those views, so it allocates nothing, never
recompiles, and a CUDA graph captured on the slot (which holds the storage pointers) serves the new
weights on its next replay.

TWO IDENTITIES guard a load, because weight SHAPES alone do not fix the function a slot computes:
a checkpoint built with the same weight shapes but a different forward-affecting flag (a critic
mode, ``--attend-unrevealed-opponents``, a belief grad mode, ...) would run the TEMPLATE's forward
with foreign weights — a silently different function. So a load must match both the state-dict
SIGNATURE (keys x shapes x dtypes) and the FORWARD FINGERPRINT (the policy's constructor
parameters, the extractor kwargs included, plus the critic runtime attributes).
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from agents.inference.service.decision import DecisionModule
from agents.inference.service.spec import SlotArchMismatch, SlotGroupSpec

#: Constructor parameters that do not change the forward (the optimizer and the schedule).
_NOT_FORWARD = frozenset({"lr_schedule", "optimizer_class", "optimizer_kwargs"})
#: Policy attributes set outside the constructor that DO change the forward's value read.
_RUNTIME_ATTRS = ("_critic_mode", "_value_from_dist")
_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")


def state_signature(sd: Dict[str, torch.Tensor]) -> Tuple[Tuple[str, Tuple[int, ...], str], ...]:
    return tuple((k, tuple(v.shape), str(v.dtype)) for k, v in sorted(sd.items()))


def forward_fingerprint(policy: Any) -> str:
    """sha256 over what fixes the forward: constructor parameters (minus the optimizer and the
    schedule) + the critic runtime attributes + whether PopArt is present."""
    params = {k: v for k, v in policy._get_constructor_parameters().items() if k not in _NOT_FORWARD}
    for a in _RUNTIME_ATTRS:
        params[a] = getattr(policy, a, None)
    params["popart"] = getattr(policy, "popart", None) is not None
    text = json.dumps(params, sort_keys=True, default=lambda o: _ADDR.sub("", repr(o)))
    return hashlib.sha256(text.encode()).hexdigest()


def _drop_debugger(policy: Any) -> None:
    # The ObservationDebugger holds host-side state and is never part of an inference forward.
    for m in policy.modules():
        if hasattr(m, "_debugger"):
            m._debugger = None


def _tensor_at(module: torch.nn.Module, key: str) -> torch.Tensor:
    prefix, _, name = key.rpartition(".")
    owner = module.get_submodule(prefix) if prefix else module
    t: Optional[torch.Tensor] = owner._parameters.get(name)
    if t is None:
        t = owner._buffers.get(name)
    if t is None:
        raise KeyError(f"state-dict key {key!r} is neither a parameter nor a buffer")
    return t


class SlotGroup:
    """``n_slots`` replicas of one architecture, weights in stacked storage on ``device``."""

    def __init__(self, spec: SlotGroupSpec, device: torch.device):
        template = spec.template
        self.name = spec.name
        self.n_slots = int(spec.n_slots)
        self.device = device
        tsd = template.state_dict()
        self.signature = state_signature(tsd)
        self.fingerprint = forward_fingerprint(template)
        replicas: List[Any] = []
        for _ in range(self.n_slots):
            r = copy.deepcopy(template)
            _drop_debugger(r)
            r.optimizer = None            # a served replica never steps
            replicas.append(r.to(device).eval())
        # sb3 registers the shared extractor under three names, so one tensor appears under up to
        # three keys: stack each UNIQUE tensor once and remember every key that aliases it.
        first = replicas[0]
        groups: Dict[int, List[str]] = {}
        for key in tsd:
            groups.setdefault(id(_tensor_at(first, key)), []).append(key)
        self._alias_groups: List[List[str]] = list(groups.values())
        self.stacked: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for keys in self._alias_groups:
                head = keys[0]
                stack = torch.stack([_tensor_at(r, head).detach() for r in replicas]).contiguous()
                self.stacked[head] = stack
                for i, r in enumerate(replicas):
                    _tensor_at(r, head).data = stack[i]
        self.policies = replicas
        self.modules = [DecisionModule(r).eval() for r in replicas]
        self.model_ids: List[str] = ["<template>"] * self.n_slots
        self.template_weights = {k: v.detach().to("cpu").clone() for k, v in tsd.items()}

    def storage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.stacked.values())

    def check_loadable(self, policy: Any) -> Dict[str, torch.Tensor]:
        """The policy's state dict, or `SlotArchMismatch` naming which identity differs."""
        sd: Dict[str, torch.Tensor] = policy.state_dict()
        sig = state_signature(sd)
        if sig != self.signature:
            mine, theirs = dict((k, (s, d)) for k, s, d in self.signature), \
                dict((k, (s, d)) for k, s, d in sig)
            diff = sorted(set(mine) ^ set(theirs)) or \
                sorted(k for k in mine if mine[k] != theirs.get(k))
            raise SlotArchMismatch(
                f"slot group {self.name!r}: the weight set's state-dict signature differs from the "
                f"group's ({len(diff)} keys, e.g. {diff[:3]})")
        fp = forward_fingerprint(policy)
        if fp != self.fingerprint:
            raise SlotArchMismatch(
                f"slot group {self.name!r}: same weight shapes but a different FORWARD fingerprint "
                f"({fp[:12]} vs the group's {self.fingerprint[:12]}) — a constructor kwarg or critic "
                "mode that changes the function differs; serving it would run the group's forward "
                "with foreign weights")
        return sd

    def copy_in(self, slot: int, sd: Dict[str, torch.Tensor]) -> None:
        """In-place copy into slot ``slot``'s views: no allocation, graph pointers unchanged."""
        with torch.no_grad():
            for keys in self._alias_groups:
                self.stacked[keys[0]][slot].copy_(sd[keys[0]], non_blocking=False)
