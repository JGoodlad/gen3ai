"""K8 — NO SILENT FALL-BACK TO EAGER (`gen3_no_silent_eager_v1`, owner 2026-10-01: "I just most care
that we don't have silent performance regressions from it not being fully compiled").

Every call into the learner's compiled region (R1, the micro-step) goes through its dispatcher
(`compile_regions.install`), which takes exactly one route:

  * the COMPILED route — counted in ``compiled`` (``"R1_compiled"``);

and there is NO declared eager route: a micro-batch of any other row count is refused by the
dispatcher (`gen3_r1_no_ragged_v1`, the collector guarantees full micro-batches), and the compiled
rollout region R0 and its eager batch-1 route are deleted (P10-E, owner-approved 2026-10-03: the
Rust collector serves the rollout forward through the T2 inference service, so R0 was never called).
A FUTURE declared eager route counts under ``"<region>_eager_<why>"`` — `CompileControl.record` logs
every such route as ``lifecycle/eager_fallback_calls``.

A region whose compiled route silently ran its Python body EAGER (dynamo disabled, a swallowed
compile error, a frame dynamo decided to skip) is caught HERE: the region's body calls
`note_eager_body` first thing; under a dynamo trace that is a constant no-op (`is_compiling()`), and
a compiled graph never re-enters the Python body — so the body counter moving during a compiled-route
call means the region did not run compiled. The dispatcher turns that into a typed FATAL.

Process-local, no torch state beyond reading `torch.compiler.is_compiling()`. Deliberately its own
tiny module: the region body (`micro_step`) imports it, and `compile_regions` imports that.
"""
from __future__ import annotations

from typing import Dict

import torch

#: Executions of each region's Python body OUTSIDE a dynamo trace (process lifetime).
EAGER_BODY: Dict[str, int] = {"R1": 0}

#: Per-update route counters, read and reset by `take()` (the compile control's per-update record).
_ROUTES: Dict[str, int] = {}


def note_eager_body(region: str) -> None:
    """Called first thing in a region's body. A no-op inside a dynamo trace (and absent from the
    compiled graph); counts an eager execution otherwise."""
    if not torch.compiler.is_compiling():
        EAGER_BODY[region] = EAGER_BODY.get(region, 0) + 1


def count(route: str) -> int:
    """Count one dispatcher route (``"R1_compiled"``); returns this update's count so far for that
    route."""
    _ROUTES[route] = _ROUTES.get(route, 0) + 1
    return _ROUTES[route]


def take() -> Dict[str, int]:
    """This update's route counts, then reset (the per-update window)."""
    out = dict(_ROUTES)
    _ROUTES.clear()
    return out


def peek() -> Dict[str, int]:
    return dict(_ROUTES)
