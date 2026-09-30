"""BENCHMARK-ONLY phase marks inside `train()` (gen3_learner_phase_hook_v1).

`train()` calls ``PHASE_HOOK(name)`` at the boundary AFTER each phase of the update, and only when
the hook is installed. Production never installs one, so every site is a single ``is not None``
test on a local read once per call — no sync, no clock, no allocation, and the numbers the fold
computes are byte-identical whether a hook is installed or not (pinned by
`src/agents/training/learner_benchmark_test.py`, which also pins that every call site is guarded).

The one consumer is `agents.training.learner_benchmark`, which installs a SEGMENT TIMER: each mark
(optionally after a ``torch.cuda.synchronize()``) closes the segment that began at the previous
mark and books its wall time to the mark's name. So a name means "the work between the previous
mark and this one":

    start        opens the call (resets the timer; books nothing)
    setup        label alignment, fold flags, probe setup, PopArt, advantage-density read
    batch        `rollout_buffer.get()` — the shuffle/index + host->device copy of one minibatch
    forward      `policy.evaluate_actions` (the extractor + every head it stashes) + capacity snapshot
    ridealong    the DETACHED ride-along heads' own step (`gen3_ridealong_heads_v1`; ~0 when off)
    loss         the whole FOLD: PPO loss, belief/intent/win-prob/... terms and their per-minibatch
                 `.item()`/`float()` diagnostics
    probes       the once-per-call grad-balance, rank, edge- and cell-liveness probes
    kl           the approx-KL read (`.cpu().numpy()`, a host sync) + the early-stop check
    noise_probe  the per-term noise-scale sampler's per-micro flush (`autograd.grad` per group)
    backward     `(loss / accum).backward()` + the distill grad-projector seams
    noise_base   the two-point noise-scale `‖g_small‖²` read (once per call)
    optim        grad clip (a host sync) + optimizer step + zero_grad, when the group is full
    capacity     `--capacity-telemetry`'s per-minibatch observe
    epoch_end    the trailing partial-group flush + the per-epoch KL/clip bookkeeping
    logging      capacity finish, explained variance and every `self.logger.record`

``probes`` is marked twice per minibatch (both once-per-call probe sites book to it); every other
name once. A mark is never placed inside a compiled region.
"""
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

# THE switch. `None` in every production process; read ONCE per `train()` call into a local.
PHASE_HOOK: Optional[Callable[[str], None]] = None

# Every name `train()` marks, in source order (``probes`` appears at two sites).
PHASES = ("start", "setup", "batch", "forward", "ridealong", "loss", "probes", "kl", "noise_probe",
          "backward", "noise_base", "optim", "capacity", "epoch_end", "logging")


def current() -> Optional[Callable[[str], None]]:
    """The installed hook, or None. `train()` calls this ONCE per call — a function rather than an
    imported name, because `from phase_hook import PHASE_HOOK` would copy the binding at import
    time and never see an install."""
    return PHASE_HOOK


@contextmanager
def installed(hook: Callable[[str], None]) -> Iterator[None]:
    """Install ``hook`` for the duration of the block, restoring whatever was there before."""
    global PHASE_HOOK
    prev = PHASE_HOOK
    PHASE_HOOK = hook
    try:
        yield
    finally:
        PHASE_HOOK = prev
