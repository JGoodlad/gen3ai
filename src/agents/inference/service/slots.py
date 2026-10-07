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
parameters, the extractor kwargs included, plus the critic runtime attributes). The extractor kwargs
are CANONICAL: one recorded at exactly the value its ABSENCE means at load (the extractor
constructor's signature default) is dropped, so a checkpoint that records a defaulted kwarg and an
older one that lacks it share a fingerprint, while any non-default difference still splits
(`canonical_extractor_kwargs`; X5 look 3, FINDING 1).

THE RIDE-ALONG HEADS ARE NOT SERVED (F-MEM, `gen3_opponent_inference_load_v1`): they are detached and
no forward reads them, so both identities leave them out — the signature skips every `ridealong.*`
tensor (`served_state_dict`) and the fingerprint skips the declared `ridealong_*` extractor kwargs
(`RIDEALONG_FLAGS`). A snapshot with heads and one without serve the same function and load into the
same slot, in either direction; a replica drops the heads (no slot stores, copies or compiles them).
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from agents.inference.service.decision import DecisionModule
from agents.inference.service.spec import (CopyParityFailure, NonFiniteWeights, SlotArchMismatch,
                                           SlotGroupSpec)

#: Constructor parameters that do not change the forward (the optimizer and the schedule).
_NOT_FORWARD = frozenset({"lr_schedule", "optimizer_class", "optimizer_kwargs"})
#: Policy attributes set outside the constructor that DO change the forward's value read.
_RUNTIME_ATTRS = ("_critic_mode",)
_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")


def served_state_dict(policy: Any) -> Dict[str, torch.Tensor]:
    """The weights a slot SERVES: the policy's state dict minus the ride-along heads (module docs)."""
    from agents.model.ridealong_heads import without_ridealong_state

    return without_ridealong_state(policy.state_dict())


def served_replica(template: Any) -> Any:
    """A deep copy of ``template`` WITHOUT its detached ride-along heads (``policy.ridealong``): the
    heads are memoised to None, so they are never copied — not even transiently onto the template's
    device while the slots are built (31 slots x 18.2 MiB at the X26 surface with every RND variant)."""
    memo: Dict[int, Any] = {}
    heads = getattr(template, "ridealong", None)
    if isinstance(heads, torch.nn.Module):
        memo[id(heads)] = None
    return copy.deepcopy(template, memo)


def state_signature(sd: Dict[str, torch.Tensor]) -> Tuple[Tuple[str, Tuple[int, ...], str], ...]:
    return tuple((k, tuple(v.shape), str(v.dtype)) for k, v in sorted(sd.items()))


def _absent_means(extractor_class: Any) -> Dict[str, Any]:
    """What an ABSENT ``features_extractor_kwargs`` key MEANS at load, per parameter: the extractor
    class's own constructor-signature default. SB3 rebuilds a checkpoint's extractor by splatting the
    zip's saved kwargs into ``features_extractor_class(observation_space, **kwargs)`` (neither
    ``load_checkpoint_strict`` nor ``historical_load_kwargs`` fills a key in), so the signature is the
    ONE source of what a missing key builds — NOT the flag registry's ``default`` (the CLI default:
    ``attend_unrevealed_opponents`` is True there and False in the signature) and NOT a
    ``ModelVersion`` migration default. ``{}`` (nothing canonicalised) for anything whose signature
    cannot be read; a ``*args`` / ``**kwargs`` parameter has no default and is never listed."""
    import inspect

    try:
        sig = inspect.signature(extractor_class)
    except (TypeError, ValueError):
        return {}
    return {n: p.default for n, p in sig.parameters.items()
            if p.default is not inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}


def _is_default(value: Any, default: Any) -> bool:
    """``value`` is EXACTLY ``default`` — same type, equal and the same repr — so the constructor provably
    builds the same thing whether it is passed or not. Deliberately strict: ``1`` vs ``True``, ``0`` vs
    ``0.0``, ``[]`` vs ``()``, and two OFF spellings (``'off'`` vs ``'none'``) are NOT merged."""
    try:
        return type(value) is type(default) and bool(value == default) and repr(value) == repr(default)
    except Exception:                                   # noqa: BLE001 - an uncomparable value is not a default
        return False


def canonical_extractor_kwargs(extractor_class: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """``kwargs`` minus every key recorded AT the value its absence means (:func:`_absent_means`).

    A checkpoint written at a later commit RECORDS constructor kwargs at their defaults that an older
    checkpoint of the same architecture lacks (X5 look 3, FINDING 1: ``policy_readout: 'tower'``,
    ``oracle_reveal: 'off'``); both build the identical extractor, so the forward fingerprint must not
    tell them apart. DROPPING (rather than filling absent keys in) keeps an old checkpoint's fingerprint
    unchanged when a new kwarg is added with a default, and a new checkpoint that records it at that
    default lands on the same hash. Any non-default value — recorded or not in the other — still splits,
    as does a key whose absence means something else than the value recorded."""
    defaults = _absent_means(extractor_class)
    return {k: v for k, v in kwargs.items() if not (k in defaults and _is_default(v, defaults[k]))}


def forward_fingerprint(policy: Any) -> str:
    """sha256 over what fixes the forward: constructor parameters (minus the optimizer, the schedule
    and the ride-along declarations) + the critic runtime attributes. The extractor kwargs are
    CANONICAL (:func:`canonical_extractor_kwargs`): a kwarg recorded at the value its absence means
    is dropped, so "absent" and "present at its default" share a fingerprint."""
    from agents.model.ridealong_heads import RIDEALONG_FLAGS

    params = {k: v for k, v in policy._get_constructor_parameters().items() if k not in _NOT_FORWARD}
    fek = params.get("features_extractor_kwargs")
    if isinstance(fek, dict):
        fek = canonical_extractor_kwargs(params.get("features_extractor_class"), fek)
        params["features_extractor_kwargs"] = {k: v for k, v in fek.items() if k not in RIDEALONG_FLAGS}
    for a in _RUNTIME_ATTRS:
        params[a] = getattr(policy, a, None)
    text = json.dumps(params, sort_keys=True, default=lambda o: _ADDR.sub("", repr(o)))
    return hashlib.sha256(text.encode()).hexdigest()


def require_finite(sd: Dict[str, torch.Tensor], where: str) -> None:
    """`NonFiniteWeights` naming the keys if any floating tensor of ``sd`` holds a NaN / Inf.
    Broken weights are REFUSED up front: the parity gate would also fail them (a NaN breaks the
    mask contract), but only after they were in a slot, and with a message that names the symptom
    instead of the cause."""
    bad = [k for k, v in sd.items() if v.is_floating_point() and not bool(torch.isfinite(v).all())]
    if bad:
        raise NonFiniteWeights(f"{where}: {len(bad)} weight tensor(s) hold NaN / Inf (e.g. {bad[:3]}) "
                               "— refused before any slot was touched")


def _host_bytes(t: torch.Tensor) -> torch.Tensor:
    """``t``'s bytes, flat, on the HOST (a device-to-host copy allocates nothing on the card)."""
    return t.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)


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
        tsd = served_state_dict(template)
        require_finite(tsd, f"slot group {spec.name!r} template")
        self.signature = state_signature(tsd)
        self.fingerprint = forward_fingerprint(template)
        replicas: List[Any] = []
        for _ in range(self.n_slots):
            r = served_replica(template)
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

    def storage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.stacked.values())

    def check_loadable(self, policy: Any) -> Dict[str, torch.Tensor]:
        """The policy's state dict, or `SlotArchMismatch` naming which identity differs."""
        sd: Dict[str, torch.Tensor] = served_state_dict(policy)
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
        require_finite(sd, f"slot group {self.name!r} load")
        return sd

    def copy_in(self, slot: int, sd: Dict[str, torch.Tensor]) -> None:
        """In-place copy into slot ``slot``'s views: no allocation, graph pointers unchanged."""
        with torch.no_grad():
            for keys in self._alias_groups:
                self.stacked[keys[0]][slot].copy_(sd[keys[0]], non_blocking=False)

    def verify_copy(self, slot: int, sd: Dict[str, torch.Tensor], where: str) -> None:
        """BIT-EXACT: slot ``slot``'s storage equals ``sd`` on EVERY key, each alias included, byte for
        byte (P10 F3, `gen3_slot_copy_verify_v1`). A copy has no tolerance, so neither does this. It is
        the only check that ties a slot to the weight set a ``load`` was given: the parity gate's eager
        reference is the slot's own replica, whose parameters are views into this same storage, so a
        copy that missed (or copied the wrong source) passes parity with |dV| 0. `CopyParityFailure`
        names the keys. Compared on the host: it allocates nothing on the card."""
        bad: List[str] = []
        with torch.no_grad():
            for keys in self._alias_groups:
                held = _host_bytes(self.stacked[keys[0]][slot])
                seen: Dict[Tuple[Any, ...], bool] = {}      # a source alias shares its tensor: copy it once
                for k in keys:
                    v = sd[k]
                    ck = (v.data_ptr(), tuple(v.shape), tuple(v.stride()), str(v.dtype), str(v.device))
                    if ck not in seen:
                        seen[ck] = torch.equal(held, _host_bytes(v))
                    if not seen[ck]:
                        bad.append(k)
        if bad:
            raise CopyParityFailure(
                f"{where}: slot group {self.name!r} slot {slot} is NOT bit-exact with the weight set it was "
                f"loaded from on {len(bad)} key(s) (e.g. {bad[:3]}) — the slot would serve weights other "
                "than the ones named; refused (the parity gate's reference reads the slot itself and "
                "cannot see this)")
