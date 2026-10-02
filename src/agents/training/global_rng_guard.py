"""K6's RNG half — NO GLOBAL RESEED AFTER THE FREEZE (`gen3_no_global_reseed_v1`).

THE BUG IT CLOSES (2026-10-02, the deletion-pass manifest's P1). sb3's `_setup_model` calls
`set_random_seed(self.seed)`, so EVERY model load re-seeded Python `random`, NumPy and torch to the
LOADED checkpoint's saved seed — and a training process loads models mid-run: every self-play pool
refresh (`SnapshotPool.load_model`), every eval sentinel (`rust_eval/launch.load_sentinels`), every
exploiter rung. Every snapshot of a run carries the run's own `--seed`, so each load REWOUND the
trainee's global streams to the state they had at startup. Measured on the production path
(`--env-core rust --arch production`, CPU `--debug`): the PPO minibatch permutation
(`RolloutBuffer.get` → `np.random.permutation`, one per epoch) is the one global-RNG consumer of the
steady state, and after every load the update sequence replayed the run's first updates' permutation
stream exactly (`designs/training/learner_lifecycle.md` "No global reseed after the freeze").

THE TWO HALVES.

* `isolated_global_rng()` — a scope inside which a model may be BUILT (an inference-only load,
  `InferenceMaskablePPO`) without touching the process's global streams: torch's CPU generator is
  restored on exit (a module's construction draws its init from it), and any stream SEEDED inside the
  scope (the ride-along heads' constructors call `torch.manual_seed`) is restored too, including the
  CUDA generators when CUDA is initialized. A seeding call inside the scope is therefore LOCAL by
  construction, and the guard below lets it through. Streams that are only DRAWN from inside the
  scope (not seeded) are left advanced: advancing is not replaying, and restoring them would rewind
  another thread's draws.

* the GUARD — `arm(sink)` wraps the global SEEDING functions (`random.seed`, `numpy.random.seed`,
  `torch.manual_seed` / `torch.random.manual_seed` / `torch.seed`, `torch.cuda.manual_seed[_all]` /
  `torch.cuda.seed[_all]`). While armed, a call from the armed process, on a thread not inside an
  isolated scope, is `GlobalReseedError` (a `FatalConfigError`: exit FATAL_CONFIG, not restarted)
  naming its call site, and is ALSO appended to ``sink`` — the freeze guard's sticky violation list —
  so a caller's `except Exception` cannot hide it from the next `check()`. `LearnerFreeze.freeze()`
  arms it at the first rollout of `learn()`; `release()` disarms. There is no allowlist: a healthy
  steady state never seeds a global stream (a stream that must be reproducible owns a generator —
  `keyed_draw`, a builder's `_rng`, a `torch.Generator`).

  SCOPE LIMITS (deterministic, documented rather than hidden): the guard sees calls made through the
  module ATTRIBUTE; a name bound before arming (`from random import seed`) bypasses it — the static
  twin `src/global_rng_seed_gate_test.py` refuses such a binding anywhere in `src/`. A state RESTORE
  (`random.setstate`, `np.random.set_state`, `torch.set_rng_state`) is not a seed and is not guarded.
  A forked child inherits the wrappers but not the arming (the process id is checked).
"""
from __future__ import annotations

import contextlib
import os
import random
import sys
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from main.exit_codes import FatalConfigError

FATAL_TAG = "[LearnerLifecycle] FATAL — GLOBAL RESEED"

#: Path fragments of frames that are never the interesting call site.
_SKIP_FRAME_FRAGMENTS = ("/torch/", "/site-packages/", "global_rng_guard.py", "<frozen ")
_SITE_FRAMES = 6


class GlobalReseedError(FatalConfigError):
    """A process-global RNG (Python `random`, NumPy's global `RandomState`, torch's generators) was
    SEEDED after the learner froze. Deterministic, so single-shot: exit `FATAL_CONFIG` (3)."""


def _site(skip: int = 2) -> str:
    try:
        f: Any = sys._getframe(skip)
    except Exception:                                      # pragma: no cover — introspection only
        return "<site unavailable>"
    frames: List[Tuple[str, int, str]] = []
    while f is not None and len(frames) < 60:
        frames.append((f.f_code.co_filename, f.f_lineno, f.f_code.co_name))
        f = f.f_back
    keep = [fr for fr in frames if not any(s in fr[0] for s in _SKIP_FRAME_FRAGMENTS)]
    keep = keep[:_SITE_FRAMES] or frames[:_SITE_FRAMES]
    return " <- ".join(f"{fn.split('/src/', 1)[-1]}:{ln} {name}" for fn, ln, name in keep)


# ------------------------------------------------------------------------------ the seeding surface
#: (module, attribute, the STREAM it seeds). `torch.manual_seed` IS `torch.random.manual_seed` (the
#: same function bound under two names); both names are wrapped.
def _targets() -> List[Tuple[Any, str, str]]:
    import torch.cuda
    import torch.random
    return [
        (random, "seed", "python"),
        (np.random, "seed", "numpy"),
        (torch, "manual_seed", "torch"),
        (torch.random, "manual_seed", "torch"),
        (torch, "seed", "torch"),
        (torch.random, "seed", "torch"),
        (torch.cuda, "manual_seed", "torch"),
        (torch.cuda, "manual_seed_all", "torch"),
        (torch.cuda, "seed", "torch"),
        (torch.cuda, "seed_all", "torch"),
    ]


_tls = threading.local()
_lock = threading.Lock()
#: The armed sinks (one per armed owner; normally exactly one — the learner's freeze).
_sinks: List[List[str]] = []
_armed_pid: Optional[int] = None
#: (module, attr) -> the original function, while wrapped.
_originals: Dict[Tuple[int, str], Tuple[Any, str, Callable[..., Any]]] = {}
#: Holders of the wrappers: one per armed sink + one per open isolated scope (any thread). The
#: wrappers are installed while this is > 0 and the originals restored when it returns to 0.
_install_refs = 0


def _scope_stack() -> List[Dict[str, bool]]:
    st = getattr(_tls, "scopes", None)
    if st is None:
        st = []
        _tls.scopes = st
    return st


def _wrap(mod: Any, attr: str, stream: str, orig: Callable[..., Any]) -> Callable[..., Any]:
    def guarded(*args: Any, **kwargs: Any) -> Any:
        scopes = _scope_stack()
        if scopes:                                   # LOCAL: the isolated scope restores it
            scopes[-1][stream] = True
            return orig(*args, **kwargs)
        if _sinks and os.getpid() == _armed_pid:
            name = f"{getattr(mod, '__name__', mod)}.{attr}"
            msg = (f"the global {stream} RNG was SEEDED after the learner froze: {name}"
                   f"({', '.join(repr(a) for a in args)}) at {_site(2)}")
            for sink in list(_sinks):
                sink.append(msg)
            raise GlobalReseedError(f"{FATAL_TAG}: {msg}\n{_HOWTO}")
        return orig(*args, **kwargs)

    guarded._gen3_global_rng_guard = True            # type: ignore[attr-defined]
    guarded.__wrapped__ = orig                       # type: ignore[attr-defined]
    guarded.__name__ = getattr(orig, "__name__", attr)
    guarded.__doc__ = getattr(orig, "__doc__", None)
    return guarded


_HOWTO = ("A training process's steady state never SEEDS a process-global RNG: a reseed rewinds every "
          "later draw of that stream (the PPO minibatch permutation replayed after every opponent load, "
          "2026-10-02). Build a model inside `global_rng_guard.isolated_global_rng()` (an inference load "
          "does), and give a stream that must be reproducible its OWN generator (`keyed_draw`, a "
          "`random.Random`, a `torch.Generator`). Fatal by design; the launcher does NOT restart it "
          "(FATAL_CONFIG). See designs/training/learner_lifecycle.md \"No global reseed after the freeze\".")


def _acquire() -> None:
    global _install_refs
    _install_refs += 1
    if _install_refs == 1:
        _install()


def _release() -> None:
    global _install_refs
    _install_refs -= 1
    if _install_refs <= 0:
        _install_refs = 0
        _uninstall()


def _install() -> None:
    for mod, attr, stream in _targets():
        key = (id(mod), attr)
        if key in _originals:
            continue
        orig = getattr(mod, attr)
        if getattr(orig, "_gen3_global_rng_guard", False):   # pragma: no cover — defensive
            continue
        _originals[key] = (mod, attr, orig)
        setattr(mod, attr, _wrap(mod, attr, stream, orig))


def _uninstall() -> None:
    for _key, (mod, attr, orig) in list(_originals.items()):
        setattr(mod, attr, orig)
    _originals.clear()


def arm(sink: List[str]) -> None:
    """Arm the guard for this process; violations are appended to ``sink`` (sticky) and raised."""
    global _armed_pid
    with _lock:
        if any(s is sink for s in _sinks):
            return
        _sinks.append(sink)
        _armed_pid = os.getpid()
        _acquire()


def disarm(sink: List[str]) -> None:
    """Remove ``sink``; the last disarm restores the original functions."""
    global _armed_pid
    with _lock:
        if not any(s is sink for s in _sinks):
            return
        _sinks[:] = [s for s in _sinks if s is not sink]
        if not _sinks:
            _armed_pid = None
        _release()


def armed() -> bool:
    return bool(_sinks) and os.getpid() == _armed_pid


# ------------------------------------------------------------------------------ the isolated scope
@contextlib.contextmanager
def isolated_global_rng() -> Iterator[None]:
    """Build a model without touching the process's global RNG streams (module docs): torch's CPU
    generator is restored on exit; a stream SEEDED inside is restored too (CUDA's when initialized).
    Nested scopes compose. Works armed or not: the seeding wrappers are installed for the scope's
    duration, and they are what MARK a stream seeded."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_cpu = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    marks: Dict[str, bool] = {}
    stack = _scope_stack()
    stack.append(marks)
    with _lock:                                      # the wrappers MARK a seed inside the scope
        _acquire()
    try:
        yield
    finally:
        stack.pop()
        with _lock:
            _release()
        if stack:                                    # propagate the marks to the enclosing scope
            for k, v in marks.items():
                stack[-1][k] = stack[-1].get(k, False) or v
        torch.set_rng_state(torch_cpu)
        if marks.get("torch") and cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if marks.get("python"):
            random.setstate(py_state)
        if marks.get("numpy"):
            np.random.set_state(np_state)


__all__ = ["GlobalReseedError", "arm", "armed", "disarm", "isolated_global_rng", "FATAL_TAG"]
