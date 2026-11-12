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
    setup        label alignment, fold flags, probe setup, advantage-density read
    batch        `rollout_buffer.get()` — the shuffle/index + host->device copy of one minibatch
    forward      `policy.evaluate_actions` (the extractor + every head it stashes) + capacity snapshot
    ridealong    the DETACHED ride-along heads' own step (`gen3_ridealong_heads_v1`; ~0 when off)
    loss         the whole FOLD: PPO loss, belief/intent/win-prob/... terms and their per-minibatch
                 `.item()`/`float()` diagnostics
    kl           the approx-KL read (`.cpu().numpy()`, a host sync) + the early-stop check
    backward     `(loss / accum).backward()`
    probes       FIRST site: the once-per-call edge- and cell-liveness reads (parameter norms)
    noise_base   the two-point noise-scale `‖g_small‖²` read (once per call)
    probes       SECOND site: the once-per-call grad-balance probe — one R1 forward and one
                 `autograd.grad` per term, each on its own graph (`probe_forward.py`)
    noise_probe  the per-term noise-scale sampler's per-micro flush — one R1 forward and one
                 `autograd.grad` per group, each on its own graph
    optim        grad clip (a host sync) + optimizer step + zero_grad, when the group is full
    capacity     `--capacity-telemetry`'s per-minibatch observe
    epoch_end    the trailing partial-group flush + the per-epoch KL/clip bookkeeping
    logging      capacity finish, explained variance and every `self.logger.record`

``probes`` is marked twice per minibatch (both once-per-call probe sites book to it); every other
name once. A mark is never placed inside a compiled region. (The rank probe reads R1's stashes
right after the forward and books to ``forward``.) Both gradient probes run AFTER the backward
(gen3_probe_own_graph_v1): the update-fit check's per-phase peaks (`update_fit._PhasePeaks`) read
each one's memory off its own mark.
"""
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

# THE switch. `None` in every production process; read ONCE per `train()` call into a local.
PHASE_HOOK: Optional[Callable[[str], None]] = None

# Every name `train()` marks, in source order (``probes`` appears at two sites).
PHASES = ("start", "setup", "batch", "forward", "ridealong", "loss", "kl", "backward", "probes",
          "noise_base", "noise_probe", "optim", "capacity", "epoch_end", "logging")


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
