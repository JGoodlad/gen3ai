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

⚠️ WHY NOT GO FURTHER — the forkserver preload, and why it must not be retried naively.
SB3's `SubprocVecEnv` uses `mp.get_context("forkserver")`, and a forkserver child inherits the
forkserver's memory copy-on-write — so `set_forkserver_preload` on a module that compiles at import
would give every worker a ready-made compiled graph for free (measured 0.12 s per worker in a
standalone probe). It does not work here, and the failure is a HANG rather than an error:

  `fork()` gives the child only the calling thread but copies every mutex, including ones held by
  threads that do not exist in the child. Forking is therefore only safe from a SINGLE-THREADED
  process, and the forkserver ends up with at least two extra threads:
    1. Inductor's parallel-codegen pool (`compile_threads`, 16 on this box) survives the compile;
       `shutdown_compile_workers()` does clear it, but
    2. poke-env's GLOBAL asyncio loop thread (`poke_env.concurrency.__run_loop`) is started at
       IMPORT time, and `agents.model.features_extractor` pulls in 37 poke-env modules
       transitively — so merely importing the extractor makes the process multi-threaded, and there
       is nothing the preload can do about it from the outside.

  Observed on a real 48-env run: the forkserver logged a successful compile, then the trainer forked
  2 workers instead of 48 and hung indefinitely — parent blocked in `unix_stream_data_wait` on the
  forkserver socket, every worker idle in `anon_pipe_read`, whole box at 0.2 load average. No error,
  no traceback, no progress.

  Reviving it requires removing poke-env from the extractor's import graph (the model layer has no
  business importing a battle client), and then re-checking `threading.active_count() == 1` in the
  forkserver before arming. `compile_prewarm_test.py` pins the hazard so this is discovered by a test
  rather than by a wedged run.
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
        from agents.model.snapshot import _compile_warmup_obs, _inductor_cache_dir
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
