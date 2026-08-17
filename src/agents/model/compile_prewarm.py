"""Warm the shared Inductor cache in the trainer BEFORE any env worker exists.

WHY. Each env worker `torch.compile`s its own frozen opponent. The codegen half of that is cached on
disk (`TORCHINDUCTOR_CACHE_DIR`), but N workers spawning at once into a COLD cache all miss and
several pay the full cold compile. Measured (`tmp/compile_spawn_cost.py`, 16 workers, 16 cores), wall
clock until every worker is ready:

    private cache per worker   163.4 s
    cold shared cache           59.6 s
    warm shared cache           30.1 s

So paying one compile here, in the parent, roughly halves worker startup. The residual ~30 s is
dynamo tracing + guard construction per process, which no on-disk cache can remove.

THE FORKSERVER PRELOAD IS LIVE (`gen3_forkserver_preload_v1`, 2026-08-16) — see
`agents.model.compile_preload` and `--compile-opponents-preload`. The hazard that killed the first
attempt (fork() from a multi-threaded process; poke-env's loop thread started at extractor import)
was fixed at the root by the LAZY poke_env package inits, and `extractor_import_is_fork_safe()`
below is the executable statement of the invariant, pinned by `compile_prewarm_test.py`. When the
preload is armed this in-parent prewarm is SKIPPED (the forkserver compile populates the same
on-disk cache, which the Popen'd eval workers still hit). The full story — the 2026-08 silent
48-env wedge, the three loud guards, the honest ~50 s-per-restart sizing — is in
`src/agents/training/CLAUDE.md` → Compiled CPU opponents.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional


def extractor_import_is_fork_safe() -> Optional[str]:
    """None if importing the extractor leaves this process single-threaded, else why not.

    This is the precondition a forkserver preload would need. It is exported (and tested) so the
    hazard is a checkable fact rather than a comment someone can talk themselves out of."""
    import threading

    import agents.model.features_extractor  # noqa: F401  (imported for its side effects)

    extra = [t.name for t in threading.enumerate() if t is not threading.main_thread()]
    if extra:
        return f"importing the extractor leaves {len(extra)} non-main thread(s): {', '.join(sorted(extra))}"
    return None


def prewarm_extractor_compile(arch_kwargs: Dict[str, Any], mappings, quiet: bool = False) -> float:
    """Build the extractor this run will use and compile it once, populating the shared cache.

    Returns seconds spent (0.0 if it was skipped or failed). Never raises: a pre-warm is an
    optimization of an optimization, and if it fails the workers simply compile cold.

    `arch_kwargs` comes from `build_extractor_arch_kwargs(args)` — the same table the real model is
    built from — so the cached codegen is keyed to the graph the workers will actually run. Weights
    are irrelevant (they are graph INPUTS, not baked constants), so a fresh random extractor warms
    the cache for every opponent checkpoint.
    """
    t0 = time.perf_counter()
    try:
        import inspect

        import gymnasium as gym
        import numpy as np
        import torch

        from agents.model.features_extractor import Gen3FeaturesExtractor
        from agents.model.compile_opponents import _compile_warmup_obs, _inductor_cache_dir
        from agents.observation.state_encoder import Gen3ObservationEncoder

        cache_dir = _inductor_cache_dir()
        layout = Gen3ObservationEncoder(mappings).get_layout()
        space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
        sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
        kw = {k: v for k, v in arch_kwargs.items() if k in sig}
        fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kw).eval()
        if hasattr(fe, "disable_observation_debugger"):
            fe.disable_observation_debugger()
        with torch.no_grad():
            torch.compile(fe.forward)(_compile_warmup_obs(fe))
    except Exception as e:                       # noqa: BLE001 — never block a run for a pre-warm
        if not quiet:
            print(f"[CompilePrewarm] skipped — {type(e).__name__}: {str(e)[:200]}", flush=True)
        return 0.0
    took = time.perf_counter() - t0
    if not quiet:
        print(f"[CompilePrewarm] warmed the Inductor cache in {took:.1f}s at {cache_dir} — env "
              f"workers will hit it warm instead of racing on a cold cache.", flush=True)
    return took
