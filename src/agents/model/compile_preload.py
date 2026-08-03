"""Compile the feature extractor ONCE, in the forkserver, so every env worker inherits it free.

THE PROBLEM. `--compile-extractor` gives a measured 6.5x on the B=1 CPU opponent forward, but each
env worker used to pay for its own compile. Measured on a 16-core box, wall clock until all 16 workers
are ready (`tmp/compile_spawn_cost.py`):

    private Inductor cache per worker   163.4 s
    cold shared cache                    59.6 s
    warm shared cache                    30.1 s

The shared on-disk cache removes CODEGEN, but each process still re-traces the graph and rebuilds
guards, and that residual is O(number of workers).

THE MECHANISM. `torch.compile` returns a live Python object, so it cannot be pickled into a child —
"compile in the parent and pass it down" is not expressible. But SB3's `SubprocVecEnv` does not use
`spawn`: it calls `mp.get_context("forkserver")` explicitly, ignoring the process-wide
`set_start_method('spawn', force=True)` in `train_rl_agent`. A forkserver child is FORKED from the
forkserver process and inherits its address space copy-on-write — including dynamo's traced-graph and
guard state.

`multiprocessing.set_forkserver_preload([...])` makes the forkserver import chosen modules once,
before forking anyone. So importing THIS module in the forkserver compiles once and every worker
inherits it. Measured (`tmp/forkserver_preload_probe.py`, 16 workers): one 19.4 s compile in the
forkserver, then each worker's first call is 3.95 ms and steady-state is 1.06 ms — 20.4 s wall for all
16, and the cost is O(1) in worker count instead of O(N).

WHY THE IMPORT-TIME SIDE EFFECT IS OK HERE. Import side effects are normally a smell, but the
forkserver preload contract IS "import this module for its side effect" — that is the only hook the
API offers. The blast radius is contained: this module is imported by nothing else, and CPython's
forkserver ignores a preload module that fails to import, so the worst case is that every worker
compiles for itself exactly as before.

THE ARCH MUST MATCH. `torch.compile` guards on the traced code path, and this extractor's forward
branches on architecture toggles. If the preload builds a different arch than the workers use, the
workers' guards miss and they recompile — correct, but the benefit is lost silently. So the arch is
handed over explicitly through `GEN3AI_COMPILE_PRELOAD_ARCH` (JSON), written by
`install_forkserver_preload` from the SAME `build_extractor_arch_kwargs` table the real model uses.
`preload_status()` reports what actually happened so the caller can warn loudly.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

ARCH_ENV_VAR = "GEN3AI_COMPILE_PRELOAD_ARCH"
STATUS_ENV_VAR = "GEN3AI_COMPILE_PRELOAD_STATUS"

# Filled in at import time (i.e. inside the forkserver). Workers inherit these values via fork, which
# is how a worker can tell whether it is running on an inherited compile or on its own.
COMPILED: Optional[Any] = None
COMPILE_SECONDS: float = 0.0
FAILURE: Optional[str] = None


def _build_and_compile() -> None:
    global COMPILED, COMPILE_SECONDS, FAILURE

    raw = os.environ.get(ARCH_ENV_VAR)
    if not raw:
        FAILURE = f"{ARCH_ENV_VAR} not set — nothing to pre-compile"
        return

    # A worker must never take a CUDA context (~252 MiB each). The forkserver is exec'd fresh, so
    # this runs before torch initialises CUDA, and every forked worker inherits the hidden device.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    t0 = time.perf_counter()
    try:
        import gymnasium as gym
        import numpy as np
        import torch

        torch.set_num_threads(1)
        from agents.model.features_extractor import Gen3FeaturesExtractor
        from agents.model.snapshot import _compile_warmup_obs, _inductor_cache_dir
        from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

        _inductor_cache_dir()
        arch: Dict[str, Any] = json.loads(raw)
        mappings = load_mappings()
        layout = Gen3ObservationEncoder(mappings).get_layout()
        space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
        import inspect
        sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
        kw = {k: v for k, v in arch.items() if k in sig}
        fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kw).eval()
        fe.disable_observation_debugger()
        fn = torch.compile(fe.forward)
        with torch.no_grad():
            fn(_compile_warmup_obs(fe))
        COMPILED = fn
        COMPILE_SECONDS = time.perf_counter() - t0
        # Announce from the forkserver so the run log shows the one-time cost was actually paid HERE
        # and not N times in the workers. Silence would make a broken preload look identical to a
        # working one.
        print(f"[CompilePreload] forkserver compiled the extractor in {COMPILE_SECONDS:.1f}s — "
              f"every env worker will inherit it.", flush=True)
    except Exception as e:                      # a preload failure must never block worker startup
        FAILURE = f"{type(e).__name__}: {e}"
        COMPILE_SECONDS = time.perf_counter() - t0
        # Loud: the run still works, but every worker now pays its own ~20-30s compile.
        print(f"⚠️ [CompilePreload] FAILED after {COMPILE_SECONDS:.1f}s — {FAILURE}. Each env worker "
              f"will compile for itself (correct, but much slower to start).",
              file=sys.stderr, flush=True)


def preload_status() -> str:
    """One line describing what the preload did in THIS process (inherited by forked workers)."""
    if COMPILED is not None:
        return f"compiled in {COMPILE_SECONDS:.1f}s"
    return f"NOT compiled ({FAILURE})"


def install_forkserver_preload(arch_kwargs: Dict[str, Any], enabled: bool) -> bool:
    """Arrange for the forkserver to compile this arch once. Call BEFORE creating the vec env.

    Returns True if the preload was installed. Must run before the first `forkserver` process is
    started, because the preload list is baked into the forkserver at launch.
    """
    if not enabled:
        return False
    import multiprocessing as mp

    from agents.model.extractor_arch import arch_kwargs_to_plain

    if "forkserver" not in mp.get_all_start_methods():
        print("[CompilePreload] unavailable — this platform has no forkserver start method; "
              "each worker will compile for itself.", flush=True)
        return False
    os.environ[ARCH_ENV_VAR] = json.dumps(arch_kwargs_to_plain(arch_kwargs), sort_keys=True)
    try:
        mp.set_forkserver_preload(["agents.model.compile_preload"])
    except Exception as e:
        print(f"[CompilePreload] could not install: {type(e).__name__}: {e}", flush=True)
        return False
    print("[CompilePreload] installed — the forkserver will compile the extractor once and every "
          "env worker will inherit it.", flush=True)
    return True


_build_and_compile()
