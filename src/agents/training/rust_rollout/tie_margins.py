"""Each row's TIE MARGIN — how far every discrete operation of the policy forward sits from its cutoff
(K9(b), `gen3_behaviour_tie_exclusion_v1`, owner 2026-10-01).

WHY. The policy forward is piecewise-discontinuous: it SELECTS (topk, argmax) and THRESHOLDS. Where one
of those sits within a rounding error of its cutoff, T2's forward (compiled, at its bucket) and the
learner's (eager, at the probe's batch) can resolve it differently and log pi(a|s) jumps on that row.
K9(b) EXCLUDES such rows from the behaviour judgement and judges every other row deterministically
(`consistency.py`). The owner: "toss out ones where the cutoff would be sensitive to a rounding error
… and then deterministically pass or fail".

WHAT. `TieMargins` is a ``TorchFunctionMode`` run around a forward. At every op executed from a line
of a forward module (`agents/model/selection_sites.FORWARD_MODULES`) it looks the line up in the
DECLARED inventory (`selection_sites`): an EXACT site is skipped, a MARGIN site's per-row margin is
computed by its `Rule` (RELATIVE: |a - b| / max(|a|, |b|); an exact tie is 0), and a discrete op on a
float operand at an UNDECLARED line is recorded (`undeclared`) — the caller refuses it. The row's
margin is the minimum over every MARGIN site, with the site that attained it.

It never changes what the forward computes: it reads each op's arguments after the op ran, on
detached float64 copies. Training forwards never run under it — only the K9(b) probe's own forward
(Rust core) and one no-grad forward of the judged micro-batch (python core).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th
from torch.overrides import TorchFunctionMode

from agents.model import selection_sites as SS

_TORCH_DIR = os.path.dirname(os.path.abspath(th.__file__))
_MODEL_DIR = os.path.dirname(os.path.abspath(SS.__file__))


class TieMarginError(RuntimeError):
    """K9(b): a row's tie margin cannot be computed — an UNDECLARED discrete op on a float operand ran in
    the forward (declare it in `agents/model/selection_sites.py`), a MARGIN site's tensor is not
    row-major, or the forward ran opaque to the recorder (compiled)."""


def _rel(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    """|a - b| / max(|a|, |b|) elementwise (float64); 0 where both are 0 (an exact tie)."""
    d = (a - b).abs()
    s = th.maximum(a.abs(), b.abs())
    return th.where(s > 0, d / th.where(s > 0, s, th.ones_like(s)), th.zeros_like(d))


def _arg(args: Tuple[Any, ...], kwargs: Dict[str, Any], i: int, key: str, default: Any) -> Any:
    return args[i] if len(args) > i else kwargs.get(key, default)


def site_margin(rule: SS.Rule, name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[th.Tensor]:
    """The elementwise margin of one MARGIN op (float64, any shape whose leading dim is the op's rows), or
    None when the op cannot cross a cutoff (fewer candidates than the selection keeps)."""
    x = args[0].detach().double()
    if rule.kind == "topk":
        k = int(_arg(args, kwargs, 1, "k", 1))
        dim = int(_arg(args, kwargs, 2, "dim", -1))
        largest = bool(_arg(args, kwargs, 3, "largest", True))
        if x.shape[dim] <= k:
            return None
        v = th.topk(x, k + 1, dim=dim, largest=largest).values
        return _rel(v.narrow(dim, k - 1, 1), v.narrow(dim, k, 1))
    if rule.kind == "sort_head":           # X5's one order: adjacent genuine -π keys among the first `head`
        dim = int(_arg(args, kwargs, 1, "dim", -1))
        if x.shape[dim] < 2:
            return None
        v = th.sort(x, dim=dim).values.narrow(dim, 0, min(int(rule.head), x.shape[dim]))
        a, b = v.narrow(dim, 0, v.shape[dim] - 1), v.narrow(dim, 1, v.shape[dim] - 1)
        genuine = (a >= -1.0) & (a <= 0.0) & (b >= -1.0) & (b <= 0.0)
        if rule.zero_exact:
            genuine = genuine & ~((a == 0) & (b == 0))
        g = _rel(a, b)
        return th.where(genuine, g, th.full_like(g, float("inf")))
    if rule.kind == "argmax":
        dim = _arg(args, kwargs, 1, "dim", None)
        if dim is None:
            raise TieMarginError(f"an argmax without a dim cannot be attributed to rows ({name})")
        dim = int(dim)
        if x.shape[dim] < 2:
            return None
        v = th.topk(x, 2, dim=dim, largest=name in ("argmax", "max")).values
        top, second = v.narrow(dim, 0, 1), v.narrow(dim, 1, 1)
        g = _rel(top, second)
        if rule.gate > 0:          # a slot whose max is at or below the gate is masked by the gate site
            g = th.where(top > rule.gate, g, th.full_like(g, float("inf")))
        return g
    other = args[1] if len(args) > 1 else kwargs.get("other")
    t = (other.detach().double() if isinstance(other, th.Tensor)
         else th.tensor(float(other), dtype=th.float64, device=x.device))
    x, t = th.broadcast_tensors(x, t.to(x.device))
    g = _rel(x, t)
    if rule.kind == "threshold":
        if rule.zero_exact:
            g = th.where((x == 0) & (t == 0), th.full_like(g, float("inf")), g)
        return g
    if rule.kind == "threshold_self":        # t is one of x's own values along the last dim (a top-k cutoff)
        is_t = x == t
        mult = is_t.sum(dim=-1, keepdim=True)
        g = th.where(is_t, th.full_like(g, float("inf")), g)
        return th.where(mult > 1, th.zeros_like(g), g)
    raise TieMarginError(f"unknown margin rule {rule.kind!r}")


def _caller() -> Optional[Tuple[str, int]]:
    """(module, line) of the forward-module line that issued the op — the first frame outside torch —
    or None when that frame is not in ``agents/model`` (a library-internal op)."""
    f = sys._getframe(2)
    while f is not None and f.f_code.co_filename.startswith(_TORCH_DIR):
        f = f.f_back
    if f is None:
        return None
    fn = f.f_code.co_filename
    if os.path.dirname(fn) != _MODEL_DIR:
        return None
    return os.path.basename(fn)[:-3], int(f.f_lineno)


class TieMargins(TorchFunctionMode):
    """Per row: the smallest margin over every MARGIN site the forward executed (module docs)."""

    def __init__(self, rows: int, keep_calls: bool = False) -> None:
        super().__init__()
        self.rows = int(rows)
        #: a measurement driver's switch: every MARGIN call's per-row margin, in call order (site, array)
        self.calls: Optional[List[Tuple[str, np.ndarray]]] = [] if keep_calls else None
        self.margin = np.full(self.rows, np.inf)
        self.site = ["" for _ in range(self.rows)]
        self.ops = 0
        self.sites_seen: Dict[str, int] = {}
        self.undeclared: Dict[str, int] = {}

    def __torch_function__(self, func: Any, types: Any, args: Tuple[Any, ...] = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        self.ops += 1
        r = SS.runtime_op(getattr(func, "__name__", ""))
        if r is None or not args or not isinstance(args[0], th.Tensor):
            return out
        kind, op = r
        name = getattr(func, "__name__", "")
        if kind == "sel":
            if not args[0].is_floating_point() or (name in ("max", "min") and not isinstance(out, tuple)):
                return out            # an integer selection, or an elementwise / global max (continuous)
        elif kind == "cmp":
            if not (isinstance(out, th.Tensor) and out.dtype == th.bool):
                return out
            other = args[1] if len(args) > 1 else kwargs.get("other")
            if not (args[0].is_floating_point() or (isinstance(other, th.Tensor) and other.is_floating_point())):
                return out            # integer / bool operands: exact by type
        else:
            if not (args[0].is_floating_point() and isinstance(out, th.Tensor)
                    and not out.is_floating_point() and out.dtype != th.bool):
                return out            # not a float -> int cast
        where = _caller()
        if where is None:
            return out
        res = SS.resolve(where[0], where[1], kind)
        if res is None or res.declared is None:
            key = f"{where[0]}.py:{where[1]} {name}"
            self.undeclared[key] = self.undeclared.get(key, 0) + 1
            return out
        rule = res.declared.rule
        if rule is None:
            return out
        g = site_margin(rule, name, args, kwargs)
        site = f"{where[0]}.py:{where[1]} {name}"
        self.sites_seen[site] = self.sites_seen.get(site, 0) + 1
        if g is None:
            return out
        if g.dim() == 0 or g.shape[0] != self.rows:
            raise TieMarginError(f"[K9(b)] the MARGIN site {site} ({res.src!r}) is not row-major: its operand has "
                                 f"shape {tuple(args[0].shape)} for {self.rows} rows — its margin cannot be attributed")
        per = g.reshape(self.rows, -1).amin(dim=1)
        per = th.where(th.isnan(per), th.zeros_like(per), per).cpu().numpy()   # a NaN operand is no margin
        if self.calls is not None:
            self.calls.append((site, per))
        better = np.flatnonzero(per < self.margin)
        for i in better:
            self.site[int(i)] = site
        self.margin = np.minimum(self.margin, per)
        return out

    def check(self) -> None:
        """Raise `TieMarginError` when the forward ran an undeclared discrete op, or saw nothing at all."""
        if self.undeclared:
            raise TieMarginError(
                "[K9(b)] the policy forward ran discrete op(s) on a float operand at line(s) the inventory does "
                f"not declare: {sorted(self.undeclared)} — declare each in agents/model/selection_sites.py "
                "(a MARGIN rule for a score, an EXACT reason otherwise; `selection_sites_test` shows how)")
        if self.ops == 0:
            raise TieMarginError(
                "[K9(b)] the tie-margin recorder saw NO torch op in the forward: it ran opaque to it (compiled?) "
                "— the margins would read 'no tie' on every row")


def selection_gaps(policy: Any, obs: Dict[str, np.ndarray], actions: Any, masks: Any, device: Any
                   ) -> Tuple[np.ndarray, List[str]]:
    """``(margin [n], site [n])``: one no-grad, train-mode forward of the learner on these ``n`` rows under
    `TieMargins` (the python core's judged micro-batch; offline drivers)."""
    from stable_baselines3.common.utils import obs_as_tensor

    acts = actions if isinstance(actions, th.Tensor) else th.as_tensor(np.asarray(actions))
    acts = acts.reshape(-1).long().to(device)
    msk = masks if isinstance(masks, th.Tensor) else th.as_tensor(np.asarray(masks))
    o = {k: (v.to(device) if isinstance(v, th.Tensor) else v) for k, v in obs.items()}
    o = o if all(isinstance(v, th.Tensor) for v in o.values()) else obs_as_tensor(obs, device)
    mode = TieMargins(int(acts.shape[0]))
    was = policy.training
    policy.set_training_mode(True)
    try:
        with th.no_grad(), mode:
            policy.evaluate_actions(o, acts, action_masks=msk.to(device))
    finally:
        policy.set_training_mode(was)
    mode.check()
    return mode.margin, mode.site
