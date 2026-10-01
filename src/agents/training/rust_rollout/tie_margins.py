"""Which rows sit at a TIE of a discrete SELECTION inside the policy forward — K9(b)'s rule (iii)
(`gen3_behaviour_tie_rule_v1`, 2026-10-01).

WHY. The policy forward is PIECEWISE-discontinuous: it SELECTS with topk (the threat-seat candidates,
`pointer_head`; the damage op's candidate moves, `damage_op_pairwise`) and argmax (`damage_op`'s dominant
move). Where two candidates are within fp32 noise of each other, the rollout's forward (T2: compiled, at
the flush's bucket shape) and the learner's (eager, at the probe's batch) can select DIFFERENTLY, and
log pi jumps on that row. The K9(b) tail sweep (`designs/research_state/measurements/k9_behaviour_tail/`)
caught three such rows in 3.54M: T2 reproduced its own stored value exactly, a few-ulp weight jitter of the
eager forward landed on it too, and the flipped selections' fp64 gaps were 5e-8 .. 1.7e-7.

A real FAULT (stale weights, a misaligned row, a mask mismatch, a corrupted output) moves a row whether or
not it sits at a tie. So a violating row NOT at a tie is a fault — K9(b) FATALs at once on it.

`selection_gaps` runs the learner's forward (train mode, no grad, the run's own precision) on the given
rows under a TorchFunctionMode that records, for every SELECTION op (topk / sort / argsort / argmax /
argmin / max / min / kthvalue with a dim), the per-row gap between the selected candidate and the next one
(for a sort: the smallest adjacent gap). An exact tie (gap 0) is a tie. Called only on a violation."""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch as th
from torch.overrides import TorchFunctionMode

#: The ops recorded. Which gap counts as "at a tie" is the gate table's (``consistency.FP32_TIE_EPS``).
SELECTION_OPS = frozenset({"topk", "sort", "argsort", "argmax", "argmin", "max", "min", "kthvalue", "msort"})


def _site() -> str:
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if "/src/agents/" in fn and not fn.endswith("tie_margins.py"):
            return f"{fn.split('/src/', 1)[1]}:{f.f_lineno}"
        f = f.f_back
    return "?"


def _gap(name: str, x: th.Tensor, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> th.Tensor:
    """The per-element gap of one selection along its dim (keepdim-shaped)."""
    def arg(i: int, key: str, default: Any) -> Any:
        return args[i] if len(args) > i else kwargs.get(key, default)

    if name == "topk":
        k, dim, largest = int(arg(1, "k", 1)), int(arg(2, "dim", -1)), bool(arg(3, "largest", True))
        s = th.sort(x.double(), dim=dim, descending=largest).values
        if s.shape[dim] <= k:
            return th.full_like(s.narrow(dim, 0, 1), float("inf"))
        return (s.narrow(dim, k - 1, 1) - s.narrow(dim, k, 1)).abs()
    if name in ("sort", "argsort", "msort"):
        dim = int(arg(1, "dim", -1)) if name != "msort" else 0
        s = th.sort(x.double(), dim=dim).values
        if s.shape[dim] < 2:
            return th.full_like(s.narrow(dim, 0, 1), float("inf"))
        return s.diff(dim=dim).abs().amin(dim=dim, keepdim=True)
    dim = int(arg(1, "dim", -1))
    if name == "kthvalue":
        k, dim = int(arg(1, "k", 1)), int(arg(2, "dim", -1))
        s = th.sort(x.double(), dim=dim).values
        lo = s.narrow(dim, k - 1, 1)
        nb = [(lo - s.narrow(dim, k - 2, 1)).abs()] if k >= 2 else []
        nb += [(s.narrow(dim, k, 1) - lo).abs()] if s.shape[dim] > k else []
        return th.minimum(*nb) if len(nb) == 2 else (nb[0] if nb else th.full_like(lo, float("inf")))
    s = th.sort(x.double(), dim=dim, descending=name in ("argmax", "max")).values
    if s.shape[dim] < 2:
        return th.full_like(s.narrow(dim, 0, 1), float("inf"))
    return (s.narrow(dim, 0, 1) - s.narrow(dim, 1, 1)).abs()


class SelectionGaps(TorchFunctionMode):
    """Per row (the leading ``rows`` of every selection's input): the smallest selection gap and its site."""

    def __init__(self, rows: int) -> None:
        super().__init__()
        self.rows = int(rows)
        self.min_gap = np.full(self.rows, np.inf)
        self.site: List[str] = ["" for _ in range(self.rows)]
        self.ops = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = getattr(func, "__name__", "")
        if name not in SELECTION_OPS or not args or not isinstance(args[0], th.Tensor):
            return out
        x = args[0]
        if name in ("max", "min") and not isinstance(out, tuple):
            return out                         # a global max / min, or the elementwise form: no selection
        if not x.is_floating_point() or x.dim() == 0 or x.shape[0] % self.rows:
            return out
        try:
            g = _gap(name, x.detach(), args, kwargs)
        except (RuntimeError, IndexError, TypeError, ValueError):
            return out
        per = g.reshape(self.rows, -1).amin(dim=1).cpu().numpy()
        self.ops += 1
        site = _site()
        better = per < self.min_gap
        for i in np.flatnonzero(better):
            self.site[int(i)] = f"{name}@{site}"
        self.min_gap = np.minimum(self.min_gap, per)
        return out


def selection_gaps(policy: Any, obs: Dict[str, np.ndarray], actions: np.ndarray, masks: np.ndarray,
                   device: Any) -> Tuple[np.ndarray, List[str]]:
    """``(min_gap [n], site [n])`` of the learner forward on these ``n`` rows (module docs)."""
    from stable_baselines3.common.utils import obs_as_tensor

    n = int(np.asarray(actions).reshape(-1).shape[0])
    mode = SelectionGaps(n)
    was = policy.training
    policy.set_training_mode(True)
    try:
        with th.no_grad(), mode:
            policy.evaluate_actions(obs_as_tensor(obs, device), th.as_tensor(np.asarray(actions).reshape(-1)).long().to(device),
                                    action_masks=th.as_tensor(np.asarray(masks)).to(device))
    finally:
        policy.set_training_mode(was)
    return mode.min_gap, mode.site
