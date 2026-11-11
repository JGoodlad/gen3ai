"""The launcher's INHERITED-NICE PREFLIGHT: a real GPU training launch from a de-prioritised shell is refused.

THE INCIDENT (2026-11-10). A waiter armed under the agent-CPU prefix (``nice -n 19``) would have passed nice 19 down to
every GPU training launch it fired. ``run._apply_nice`` only moves the launcher's own nice TOWARD ``DEFAULT_NICE`` (10),
and an unprivileged process cannot LOWER its nice, so those runs would have trained at nice 19 beside agent work that
was meant to yield to them. It was caught by hand. The launcher can read the inherited nice from the OS alone, so it
refuses here, naming the number and the fix.

Scope: only a REAL training launch, i.e. a device that RESOLVES to CUDA, without ``--debug``. An explicit ``--device
cuda`` / ``cuda:N`` resolves to itself; ``--device auto`` (or none: the default) resolves the way the trainer's own
desktop-GPU check does (``desktop_gpu.resolve_device``): CUDA iff an NVIDIA driver answers (``nvidia-smi`` on PATH) AND the
launcher's env does not hide every GPU (``CUDA_VISIBLE_DEVICES=""``: the agent's no-lease shell, whose child trains on
CPU). The first version read only the typed string, so an ``auto`` launch on a GPU box slipped through (2026-11-10).
``--dry-run`` reports the same verdict as a refusal line (the waiters' dry-run checks catch it before the real launch). A
cpu launch, a ``--debug`` smoke, an ``auto`` launch that resolves to CPU and an agent's own CPU work are never refused.
The ceiling is ``max(DEFAULT_NICE, --nice)``: a typed ``--nice 19`` is a deliberate request and its own inherited 19 is
allowed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional

from main.launcher.cuda_visibility_gate import ENV_VAR as _CUDA_ENV, visible_tokens
from utils import desktop_gpu

DEFAULT_NICE = 10   # asserted equal to ``run.DEFAULT_NICE`` by nice_gate_test (this module must stay import-light)


def inherited_nice() -> Optional[int]:
    """This process's current nice, read-only (``getpriority``), or None where the OS has no such call."""
    try:
        return int(os.getpriority(os.PRIO_PROCESS, 0))
    except (AttributeError, OSError):
        return None


def gpu_usable(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether ``--device auto`` would train on CUDA from this launcher: an NVIDIA driver answers and ``env`` (the
    launcher's own, which is the child's: ``child._build_child_env``) does not hide every device. No CUDA is touched."""
    value = (os.environ if env is None else env).get(_CUDA_ENV)
    hidden = value is not None and not visible_tokens(value)
    return desktop_gpu.nvidia_gpu_visible() and not hidden


@dataclass(frozen=True)
class NiceVerdict:
    device: str            # the RESOLVED device ("cpu", "cuda", "cuda:1", ...)
    debug: bool
    inherited: Optional[int]
    ceiling: int
    applies: bool          # a real GPU training launch
    refused: bool
    requested: str = "auto"  # the typed --device ("auto" when none)

    @property
    def resolved_note(self) -> str:
        return f" (--device {self.requested} resolved to {self.device})" if self.requested != self.device else ""

    @property
    def reason(self) -> str:
        return (f"this shell runs at nice {self.inherited}, which a GPU training launch{self.resolved_note} "
                f"would inherit: the launcher can only raise its nice toward {DEFAULT_NICE}, never lower it, so the run and "
                f"its ~940 processes would train at nice {self.inherited}, starved by agent work. Fix: launch from a nice-0 (or <= "
                f"{self.ceiling}) shell, not one under `nice -n {self.inherited}` or the agent-CPU prefix")

    def lines(self) -> List[str]:
        if self.refused:
            return [f"inherited nice : ✗ REFUSED (FATAL_CONFIG): {self.reason}"]
        if not self.applies:
            return [f"inherited nice : n/a ({self.inherited}; not a real GPU training launch: resolves to "
                    f"{self.device}{', --debug' if self.debug else ''})"]
        return [f"inherited nice : ✓ {self.inherited} (<= {self.ceiling}){self.resolved_note}"]


def verdict_for_launch(device: Optional[str], *, debug: bool, nice_arg: int,
                       inherited: Optional[int] = None,
                       cuda_available: Optional[Callable[[], bool]] = None) -> NiceVerdict:
    """The verdict for a launch with ``--device device`` / ``--debug`` / ``--nice nice_arg``.

    ``device`` is resolved first (an ``auto`` / absent device by ``cuda_available``, default :func:`gpu_usable`), and the
    gate applies to the RESOLVED device. ``inherited`` defaults to the OS's reading (``inherited_nice()``); tests pass
    both."""
    requested = str(device or "auto").strip().lower()
    resolved = desktop_gpu.resolve_device(requested, debug, cuda_available if cuda_available is not None else gpu_usable)
    cur = inherited_nice() if inherited is None else inherited
    ceiling = max(DEFAULT_NICE, int(nice_arg))
    applies = resolved.startswith("cuda") and not debug
    return NiceVerdict(resolved, bool(debug), cur, ceiling, applies,
                       refused=applies and cur is not None and cur > ceiling, requested=requested)
