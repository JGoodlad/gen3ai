"""THE GROUPED FORWARD (X45, ``gen3_t2_grouped_forward_v1``; ``ServiceSpec.grouped_slots``, default EMPTY = OFF):
ONE decision forward over a contiguous RANGE of a slot group's slots, each slot on its own weights.

WHY. A collection host step serves the trainee plus up to ~20 opponent snapshots, ~11 rows each: ~21 small
forwards, each latency-bound (T24: +1.0 ms of GPU wait per host step per live snapshot; the 2026-10-19 GPU
gap read 5,453 env steps/s at pool 5 against 3,507 at pool 20). The grouped forward runs every slot of the
range in ONE call — ``torch.func.vmap`` over the group's STACKED weight storage (``SlotGroup.stacked``, a
``[n_slots, ...]`` tensor per weight, so a range is a zero-copy view) through ``torch.func.functional_call``
— so the pool size and the opponent mix stay exactly as they are; only the number of forwards changes.

WHAT BLOCKED vmap BEFORE, and how this module gets past it WITHOUT touching the model (measured
2026-09-29, ``m5_t2/PROGRESS.md``; re-read 2026-10-19 at HEAD): the extractor writes IN PLACE into tensors
it creates inside the forward (``torch.zeros(...).scatter_(src)``, ``out[ar, idx] = x`` — 34 sites in 15
files at HEAD), and vmap refuses an in-place write of a slot-batched value into a tensor that is not
batched. :class:`_BatchFresh` is a ``TorchFunctionMode`` active only inside the vmapped body: every tensor a
FACTORY creates there (``zeros`` / ``ones`` / ``full`` / ``empty``, their ``*_like`` and ``Tensor.new_*``)
is made slot-batched by adding a slot-batched ZERO of its dtype — ``-0.0`` for floats, so ``x + (-0.0)`` is
``x`` BIT FOR BIT for every ``x`` (``+0.0`` and ``-0.0`` included), ``0`` for integers, ``False`` (``|``) for
bools. The values are unchanged; the in-place writes then land in batched tensors. Everything is public
API, so ``torch.compile`` traces it (``fullgraph``) for the graph backend. A NEW model site that writes a
slot-batched value in place into a tensor made by any OTHER constructor (``arange``, ``tensor``, ``eye``, …)
raises vmap's "use out-of-place operators" error at the grouped service's startup — fail-closed, and
``grouped_test.py`` (a production-arch model) fails on it in the routine gate: add the constructor to
:class:`_BatchFresh`, or write that site out of place.

THREE MORE THINGS the body does, each a measured blocker:

* the forward's per-forward STASHES (``ExtractorStashes``, ``OpStashes``, ``last_*`` tensors) are module
  attributes; a vmapped forward would leave slot-batched tensors on them (and under ``torch.compile`` they
  become graph OUTPUTS that AOT autograd cannot read: "Cannot access storage of BatchedTensorImpl"). The
  attributes the forward rewrites are FOUND at build by diffing an eager forward (:func:`_forward_state_keys`)
  and RESET to their resting values at the end of every call, so no side effect escapes the body;
* ``prims.inductor_force_stride_order`` (``dense_attn_bias``, the SDPA bias under compile) has no batching
  rule; :func:`_register_force_op_vmap` gives it one (move the slot dim first, pin the PHYSICAL tensor
  row-major — the same layout claim per slot), once per process;
* the non-persistent buffers (data-derived tables: not in the state dict, never loaded, identical in every
  slot) are passed UNBATCHED, so no slot holds a copy; :class:`GroupedForward` checks they are equal across
  the range at build.

PARITY. A grouped slot's decision equals its per-slot decision at fp32-REASSOCIATION tolerance, not bit for
bit: a batched-weight matmul sums in another order (measured CPU, 2 perturbed slots: max|dlogp| 4.8e-7,
max|dV| 3.3e-7). The service judges every grouped slot against its OWN eager reference with the same
declared bars as every other path (``parity.judge``: legal log-prob 1e-3, V 1e-4, greedy on decisive rows,
the mask contract) at startup, at every bucket, on every ``load``; the test pins it at a much tighter bar.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
from torch.overrides import TorchFunctionMode

from agents.inference.service.decision import DecisionModule, _is_forward_state
from agents.inference.service.slots import SlotGroup, served_replica
from agents.inference.service.spec import ServiceError


class GroupedForwardError(ServiceError):
    """The grouped forward cannot serve this slot range (a declaration or build-time check failed)."""


# ------------------------------------------------------------------------------- the factory mode
class _BatchFresh(TorchFunctionMode):
    """Inside the vmapped body: every FACTORY output is made slot-batched by adding ``z`` (a slot-batched
    zero), so the model's in-place writes into fresh tensors are legal under vmap (module docs). The checks
    are IDENTITY checks (``func is torch.zeros``): dynamo cannot trace ``set.__contains__`` here."""

    def __init__(self, z: torch.Tensor):
        super().__init__()
        self.z = z

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        if (func is torch.zeros or func is torch.ones or func is torch.full or func is torch.empty
                or func is torch.zeros_like or func is torch.ones_like or func is torch.full_like
                or func is torch.empty_like or func is torch.Tensor.new_zeros
                or func is torch.Tensor.new_ones or func is torch.Tensor.new_full
                or func is torch.Tensor.new_empty):
            if out.dtype == torch.bool:
                out = torch.logical_or(out, self.z.to(torch.bool))
            elif out.is_floating_point():
                out = out + self.z.to(out.dtype)           # x + (-0.0) == x, bit for bit
            else:
                out = out + self.z.to(torch.int64).to(out.dtype)
        return out


_FORCE_VMAP_REGISTERED = False


def _register_force_op_vmap() -> None:
    """A vmap rule for ``prims.inductor_force_stride_order`` (``dense_attn_bias``): the batched bias is moved
    slot-first and the PHYSICAL tensor pinned row-major — each slot's slice is then row-major too, which is
    the claim the prim makes for one slot. Once per process (``torch.library`` refuses a second)."""
    global _FORCE_VMAP_REGISTERED
    if _FORCE_VMAP_REGISTERED:
        return
    from agents.model import dense_attn_bias as dab

    def rule(info: Any, in_dims: Tuple[Any, ...], x: torch.Tensor, stride: Any) -> Any:
        bd = in_dims[0]
        if bd is None:
            return dab._FORCE_OP(x, stride), None
        x = x.movedim(bd, 0)
        return dab._FORCE_OP(x, dab._row_major_strides(x.shape)), 0

    torch.library.register_vmap("prims::inductor_force_stride_order", rule)
    _FORCE_VMAP_REGISTERED = True


# ------------------------------------------------------------------------------- forward state
_MODULE_INTERNALS = frozenset({"training", "_parameters", "_buffers", "_modules", "_non_persistent_buffers_set",
                               "_backward_pre_hooks", "_backward_hooks", "_is_full_backward_hook",
                               "_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
                               "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs", "_state_dict_hooks",
                               "_state_dict_pre_hooks", "_load_state_dict_pre_hooks",
                               "_load_state_dict_post_hooks"})


def _owners(module: DecisionModule) -> List[Any]:
    """Every object whose attributes a decision forward may rewrite: the policy and each of its modules."""
    pol = module.policy
    return [pol] + [m for m in pol.modules() if m is not pol]


def _forward_state_keys(module: DecisionModule, run: Callable[[], Any]) -> List[Tuple[Any, str]]:
    """``(owner, attribute)`` for every attribute ``run()`` (one eager forward) REPLACED with forward state —
    a tensor, a tuple / list of tensors or a ``*Stashes`` dataclass (``decision._is_forward_state``)."""
    before = [(o, {k: id(v) for k, v in vars(o).items() if k not in _MODULE_INTERNALS}) for o in _owners(module)]
    run()
    keys: List[Tuple[Any, str]] = []
    for o, ids in before:
        for k, v in vars(o).items():
            if k in _MODULE_INTERNALS or ids.get(k) == id(v):
                continue
            if _is_forward_state(v):
                keys.append((o, k))
    return keys


def _resting(v: Any) -> Any:
    """The EMPTY value of a forward-state attribute (``decision.forward_state_released``'s rule)."""
    return type(v)() if type(v).__name__.endswith("Stashes") else None


# ------------------------------------------------------------------------------- the forward
class GroupedForward:
    """One decision forward over slots ``lo .. hi - 1`` of ``group`` (in-group indices), batched over the
    slots. ``__call__(obs [S, b, D], mask [S, b, A], z [S]) -> (logp [S, b, A], value [S, b],
    greedy [S, b])``; ``z`` is :attr:`zeros` (a ``-0.0`` per slot — the factory mode's batched zero).

    The weights it reads are the group's stacked storage, sliced ``[lo:hi]`` — VIEWS, so a ``load`` (an
    in-place copy into a slot) is what the next call reads, eager or captured. The functional base is a
    private replica (its own weights are never read; its stashes are reset after every call)."""

    def __init__(self, group: SlotGroup, lo: int, hi: int, *, obs: torch.Tensor, mask: torch.Tensor):
        if not 0 <= lo < hi <= group.n_slots:
            raise GroupedForwardError(f"slot group {group.name!r}: grouped range [{lo}, {hi}) outside 0..{group.n_slots}")
        _register_force_op_vmap()
        self.group, self.lo, self.hi, self.n = group, int(lo), int(hi), int(hi) - int(lo)
        device = group.device
        tpl = group.policies[lo]
        replica = served_replica(tpl)
        replica.optimizer = None
        self.base = DecisionModule(replica.to(device).eval()).eval()
        self.batched, self.consts = self._weights(group)
        self.zeros = torch.full((self.n,), -0.0, dtype=torch.float32, device=device)
        # the attributes a forward rewrites, found on an eager forward of the BASE, then emptied
        with torch.no_grad():
            self._state = _forward_state_keys(self.base, lambda: self.base(obs[:2], mask[:2]))
        self._rest: List[Tuple[Any, str, Any]] = [(o, k, _resting(getattr(o, k))) for o, k in self._state]
        self._reset()
        self._vmapped = torch.func.vmap(self._body, in_dims=(0, None, 0, 0, 0))

    def _weights(self, group: SlotGroup) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """``(batched, consts)`` keyed by the BASE's parameter / buffer names: every tensor of a slot that is a
        view into the group's stacked storage maps to that storage's ``[lo:hi]`` slice (each slot's tensor is
        proven to BE its row of it); every other buffer (non-persistent, data-derived) is a constant, proven
        equal across the range."""
        lo, hi = self.lo, self.hi
        rows: Dict[Tuple[int, Tuple[int, ...]], str] = {}
        for head, st in group.stacked.items():
            rows[(st[lo].data_ptr(), tuple(st[lo].shape))] = head
        batched: Dict[str, torch.Tensor] = {}
        consts: Dict[str, torch.Tensor] = {}
        mods = group.modules

        def named(m: DecisionModule) -> Dict[str, torch.Tensor]:
            out = dict(m.named_parameters(remove_duplicate=False))
            out.update(dict(m.named_buffers(remove_duplicate=False)))
            return out

        per_slot = [named(mods[j]) for j in range(lo, hi)]
        base_names = named(self.base)
        if set(base_names) != set(per_slot[0]):
            raise GroupedForwardError(f"slot group {group.name!r}: the base replica's tensor names differ from slot {lo}'s")
        for name, t in per_slot[0].items():
            head = rows.get((t.data_ptr(), tuple(t.shape)))
            if head is None:
                if name in dict(mods[lo].named_parameters(remove_duplicate=False)):
                    raise GroupedForwardError(f"slot group {group.name!r}: parameter {name!r} of slot {lo} is not a view "
                                              "into the group's stacked storage — a grouped forward would not read the slot's weights")
                for j, d in enumerate(per_slot[1:], start=lo + 1):
                    if not torch.equal(d[name], t):
                        raise GroupedForwardError(f"slot group {group.name!r}: buffer {name!r} is not in the state dict and "
                                                  f"differs between slots {lo} and {j}; it cannot be served unbatched")
                consts[name] = t
                continue
            st = group.stacked[head]
            for j, d in enumerate(per_slot, start=lo):
                if d[name].data_ptr() != st[j].data_ptr():
                    raise GroupedForwardError(f"slot group {group.name!r}: {name!r} of slot {j} is not row {j} of the "
                                              f"stacked {head!r}")
            batched[name] = st[lo:hi]
        return batched, consts

    def _reset(self) -> None:
        for o, k, v in self._rest:
            setattr(o, k, v)

    def _body(self, w: Dict[str, torch.Tensor], c: Dict[str, torch.Tensor], obs: torch.Tensor,
              mask: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with _BatchFresh(z):
            logp, value = torch.func.functional_call(self.base, (w, c), (obs, mask))
        for o, k, v in self._rest:              # no forward state escapes the body (module docs)
            setattr(o, k, v)
        return logp, value, logp.argmax(-1)

    def __call__(self, obs: torch.Tensor, mask: torch.Tensor, z: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._vmapped(self.batched, self.consts, obs, mask, z)

    def call_with(self, w: Dict[str, torch.Tensor], c: Dict[str, torch.Tensor], obs: torch.Tensor,
                  mask: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The same forward with the weights as ARGUMENTS (what ``torch.compile`` traces for the graph backend:
        the weight views become graph inputs, so a captured graph reads the stacked storage itself)."""
        return self._vmapped(w, c, obs, mask, z)


def grouped_ranges(slots: Sequence[Tuple[int, int]], grouped: Sequence[int]) -> List[Tuple[int, int, int, List[int]]]:
    """``[(group index, lo, hi, global slot ids)]`` — the declared grouped slots split per slot group, each a
    CONTIGUOUS in-group range (else `GroupedForwardError`). ``slots`` is the service's ``[(gi, i)]``."""
    by_group: Dict[int, List[Tuple[int, int]]] = {}
    for s in sorted({int(x) for x in grouped}):
        if not 0 <= s < len(slots):
            raise GroupedForwardError(f"grouped slot {s} is not declared (0..{len(slots) - 1})")
        gi, i = slots[s]
        by_group.setdefault(gi, []).append((i, s))
    out = []
    for gi, members in sorted(by_group.items()):
        idx = [i for i, _ in members]
        if idx != list(range(idx[0], idx[0] + len(idx))):
            raise GroupedForwardError(f"slot group {gi}: grouped slots {idx} are not one contiguous range — a "
                                      "grouped forward reads a zero-copy view of the stacked storage")
        out.append((gi, idx[0], idx[-1] + 1, [s for _, s in members]))
    return out
