"""HOST-SYNC TRACE — count the learner update's host<->device synchronisation points ON CPU (T25 item 1).

On CUDA every one of these blocks the host main thread until the GPU queue drains to it: a scalar read
(`.item()`, `float(t)`, `bool(t)`, `int(t)`, `.tolist()`, `.numpy()` / `np.asarray(t)`, an f-string of a
tensor), a copy to the host (`.cpu()`, `.to("cpu")`), and the ops whose OUTPUT SHAPE depends on the data
(`nonzero`, boolean-mask indexing, `masked_select`, `unique`, ...), which read a count back before they
can allocate. `torch.cuda.set_sync_debug_mode` sees them only on a GPU; this sees the SAME call sites on
CPU, through a `TorchFunctionMode` that every Tensor method and `torch.*` function dispatches through,
so a CPU unit test can bound them (`host_sync_guard_test.py`) without the GPU lease.

**THE RESIDENCY MODEL.** On CPU every tensor is host memory, so the trace keeps its own record of which
tensors WOULD be host-resident on a CUDA learner, and a read of one is booked as ``host`` (free), not as
a sync:

* the result of a copy to the host (`.cpu()`, `.to("cpu")` with the literal string, `.numpy()`) is host;
* a factory (`torch.tensor`, `zeros`, `as_tensor`, ...) called WITHOUT a ``device=`` is host (torch's
  default device); WITH one it is device-resident (production passes ``device=`` only to reach the
  learner's device — on this CPU run that device is also the CPU, which is exactly why the call site,
  not the tensor, decides);
* ``.to(<torch.device>)`` is device-resident (``.to(self.device)``); a dtype-only ``.to`` propagates;
* any other op's result is host iff EVERY tensor it read is host;
* tensors that exist before the trace starts are device-resident unless passed as ``host=`` (the
  optimizer's ``step`` counters, which torch keeps on the CPU for a non-fused, non-capturable Adam).

What it CANNOT see (each is a stated limit, never a silent zero): a sync INSIDE a torch function that
itself dispatches through ``__torch_function__`` (the mode is popped while the handler runs — the
learner's code calls none that sync); and CUDA-only calls (`torch.cuda.synchronize`,
``Stream.synchronize``, a blocking H2D copy) — `host_sync_guard_test` checks those statically.

    with HostSyncTrace(host=[...]) as tr:
        model.train()
    tr.total, tr.by_site()      # {("file.py:123", "fn", "item"): n, ...}
"""
from __future__ import annotations

import collections
import os
import sys
from typing import Any, Counter, Dict, Iterable, List, Optional, Tuple

import torch
from torch.overrides import TorchFunctionMode

#: Tensor methods / torch functions that READ a value to the host.
_READS = frozenset({
    "item", "tolist", "numpy", "__bool__", "__float__", "__int__", "__index__", "__array__",
    "__format__", "__complex__",
})
#: Copies to the host.
_TO_HOST = frozenset({"cpu"})
#: Ops whose output SHAPE depends on the data (a count read back before the allocation).
_SHAPE_DEPENDENT = frozenset({
    "nonzero", "argwhere", "masked_select", "unique", "unique_consecutive", "bincount",
})
#: Indexing ops that sync when an index is a BOOL tensor (a hidden `nonzero`).
_INDEXING = frozenset({"__getitem__", "__setitem__", "index_put_", "index_put"})
#: Factories: host unless given a ``device=``.
_FACTORIES = frozenset({
    "tensor", "as_tensor", "from_numpy", "zeros", "ones", "empty", "full", "arange", "scalar_tensor",
    "linspace", "rand", "randn", "randint", "eye", "asarray",
})

_TORCH_DIR = os.path.dirname(torch.__file__)
_THIS = os.path.abspath(__file__)
_HOST_ATTR = "_gen3_host_resident"


def _is_host(t: Any) -> bool:
    return bool(getattr(t, _HOST_ATTR, False))


def _mark_host(obj: Any) -> None:
    if isinstance(obj, torch.Tensor):
        try:
            setattr(obj, _HOST_ATTR, True)
        except (AttributeError, RuntimeError):      # pragma: no cover - a tensor that takes no attrs
            pass
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            _mark_host(o)


def _tensors(args: Iterable[Any]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for a in args:
        if isinstance(a, torch.Tensor):
            out.append(a)
        elif isinstance(a, (tuple, list)):
            out.extend(x for x in a if isinstance(x, torch.Tensor))
    return out


def _has_bool_index(args: Tuple[Any, ...]) -> bool:
    return any(t.dtype == torch.bool for t in _tensors(args[1:]))


def _to_target(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[str]:
    """``"host"`` for ``.to("cpu")``, ``"device"`` for ``.to(<torch.device>)`` / ``.to(device=...)`` / a
    tensor, None for a dtype-only ``.to``."""
    cand = [kwargs["device"]] if "device" in kwargs else list(args[1:2])
    for c in cand:
        if isinstance(c, str):
            return "host" if c == "cpu" else "device"
        if isinstance(c, torch.device):
            return "device"
        if isinstance(c, torch.Tensor):
            return "host" if _is_host(c) else "device"
    return None


def _site(depth_limit: int = 60) -> Tuple[str, str, Optional[str], bool]:
    """(innermost repo ``file:line``, its function, the ``train()``-level ``ppo.py:line`` if any, and
    whether the read was made from inside torch's own code)."""
    f = sys._getframe(2)
    inner: Optional[Tuple[str, str]] = None
    train_line: Optional[str] = None
    via_torch = False
    first = True
    n = 0
    while f is not None and n < depth_limit:
        fn = os.path.abspath(f.f_code.co_filename)
        if fn != _THIS:
            if first:
                via_torch = fn.startswith(_TORCH_DIR)
                first = False
            if not fn.startswith(_TORCH_DIR):
                if inner is None:
                    inner = (_short(fn) + f":{f.f_lineno}", f.f_code.co_name)
                if f.f_code.co_name == "train" and fn.endswith(os.path.join("instrumented_ppo", "ppo.py")):
                    train_line = f"ppo.py:{f.f_lineno}"
                    break
        f = f.f_back
        n += 1
    if inner is None:
        inner = ("<torch>", "?")
    return inner[0], inner[1], train_line, via_torch


def _short(path: str) -> str:
    marker = os.sep + "src" + os.sep
    i = path.rfind(marker)
    return path[i + len(marker):] if i >= 0 else os.path.basename(path)


class SyncEvent(collections.namedtuple("SyncEvent", "site fn op train_site via_torch")):
    """One host sync: where (innermost repo frame), which op, the `train()` line it hangs off."""


class HostSyncTrace(TorchFunctionMode):
    """Count every host-sync call while active (module docstring). ``events`` holds the SYNCS (reads
    of device-resident tensors), in order; ``host_reads`` counts reads of host-resident ones."""

    def __init__(self, host: Iterable[torch.Tensor] = ()) -> None:
        super().__init__()
        self.events: List[SyncEvent] = []
        self.host_reads = 0
        for t in host:
            _mark_host(t)

    def __torch_function__(self, func: Any, types: Any, args: Tuple[Any, ...] = (),
                           kwargs: Optional[Dict[str, Any]] = None) -> Any:
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        ins = _tensors(args) + _tensors(kwargs.values())
        self_t = args[0] if args and isinstance(args[0], torch.Tensor) else None
        op: Optional[str] = None
        to_host = False
        if name in _READS:
            op = name
            to_host = name == "numpy"
        elif name in _TO_HOST:
            op, to_host = name, True
        elif name == "to" and _to_target(args, kwargs) == "host":
            op, to_host = "to(cpu)", True
        elif name in _SHAPE_DEPENDENT:
            op = name
        elif name in _INDEXING and _has_bool_index(args):
            op = f"{name}[bool]"
        elif name == "where" and len(args) == 1 and not kwargs:
            op = "where(cond)"
        elif name == "repeat_interleave" and "output_size" not in kwargs and \
                len(args) >= 2 and isinstance(args[1], torch.Tensor):
            op = "repeat_interleave"
        if op is not None:
            source_host = (_is_host(self_t) if self_t is not None
                           else bool(ins) and all(_is_host(t) for t in ins))
            if source_host:
                self.host_reads += 1
            else:
                site, fn, train_site, via_torch = _site()
                self.events.append(SyncEvent(site, fn, op, train_site, via_torch))
        out = func(*args, **kwargs)
        # residency of the result
        if to_host:
            if out is self_t:
                # on CPU `.cpu()` / `.to("cpu")` hand back the SAME tensor; on CUDA it is a copy. Copy
                # here too, or tagging the result would tag the device-resident source as host.
                out = out.clone()
            _mark_host(out)
        elif name in _FACTORIES:
            if "device" not in kwargs:
                _mark_host(out)
        elif name in ("to", "cuda"):
            tgt = _to_target(args, kwargs) if name == "to" else "device"
            if tgt is None and self_t is not None and _is_host(self_t):
                _mark_host(out)
        elif ins and all(_is_host(t) for t in ins):
            _mark_host(out)
        return out

    # ---------------------------------------------------------------- reads
    @property
    def total(self) -> int:
        return len(self.events)

    def by_site(self) -> Counter[Tuple[str, str, str]]:
        return collections.Counter((e.site, e.fn, e.op) for e in self.events)

    def by_train_site(self) -> Counter[Tuple[Optional[str], str, str]]:
        return collections.Counter((e.train_site, e.site, e.op) for e in self.events)

    def table(self) -> str:
        rows = sorted(self.by_site().items(), key=lambda kv: (-kv[1], kv[0]))
        return "\n".join(f"{n:6d}  {op:<18s} {site}  ({fn})" for (site, fn, op), n in rows)


def optimizer_host_tensors(optimizer: Any) -> List[torch.Tensor]:
    """The optimizer state torch keeps on the HOST for a non-fused, non-capturable Adam / AdamW: the
    per-parameter ``step`` counters (``torch.tensor(0.0)``, no device). Pass to ``HostSyncTrace(host=)``
    so the optimizer's own ``step.item()`` reads are not booked as device syncs."""
    out: List[torch.Tensor] = []
    for group in getattr(optimizer, "param_groups", ()):
        if group.get("fused") or group.get("capturable"):
            continue
        for p in group["params"]:
            st = optimizer.state.get(p, {})
            s = st.get("step") if isinstance(st, dict) else None
            if isinstance(s, torch.Tensor):
                out.append(s)
    return out
