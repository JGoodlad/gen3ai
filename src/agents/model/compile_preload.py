"""The forkserver compile preload — compile ONCE, every env worker inherits it by fork.

`gen3_forkserver_preload_v1`. The per-worker `--compile-opponents` cost is ~30 s against a warm
on-disk Inductor cache (dynamo tracing + guard construction, which no disk cache can remove).
A forkserver child inherits memory copy-on-write, so a compile performed in the FORKSERVER
process is ~free in every worker (a standalone probe measured 0.12 s per worker). This module
is that compile: `train_rl_agent` passes it to
`mp.get_context("forkserver").set_forkserver_preload(...)` and the import side effect runs in
the forkserver before any worker exists.

## Why this was impossible until 2026-08-16, and what made it possible

`fork()` copies every mutex but only the calling thread, so forking is safe ONLY from a
single-threaded process — and importing the extractor used to start poke-env's global asyncio
loop thread (any `poke_env.x` import executed the eager package `__init__`, which imported
`player` → `ps_client` → `concurrency`). A 48-env run with the earlier preload forked 2 workers
and hung forever, parent blocked in `unix_stream_data_wait`, box at 0.2 load, no error anywhere.
The fix was at the ROOT: `poke_env/__init__.py` and `poke_env/player/__init__.py` are now LAZY
(PEP 562), so the extractor's import graph (battle enums, data, normalize) is thread-free —
`compile_prewarm.extractor_import_is_fork_safe()` is the executable statement, pinned by
`compile_prewarm_test.py`.

## The two guards, both LOUD

* Inductor's parallel-codegen pool survives a compile;
  `torch._inductor.async_compile.shutdown_compile_workers()` clears it.
* After the compile + shutdown, `threading.active_count()` MUST be 1. If it is not, this module
  RAISES — which kills the forkserver bootstrap and fails `SubprocVecEnv` construction loudly in
  the parent. That inversion is the point: the failure mode this replaces was a SILENT 13-hour
  wedge, and a preload that cannot prove single-threadedness must refuse to let anything fork.

Config arrives as DATA via `GEN3AI_PRELOAD_ARCH` (JSON of `arch_kwargs_to_plain(...)`) — the
forkserver is a fresh interpreter that never parsed argv, which is exactly why
`extractor_arch.ARCH_ARG_KEYS` exists as a table. No env var ⇒ importing this module is a no-op
(safe against an accidental import outside the forkserver).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time


def _preload() -> None:
    cfg_json = os.environ.get("GEN3AI_PRELOAD_ARCH")
    if not cfg_json:
        return                                          # not a forkserver preload context: no-op
    t0 = time.perf_counter()
    # The forkserver's children are env WORKERS — CPU-only by design. Hide the card BEFORE torch
    # loads so neither the forkserver nor any forked worker can ever hold a CUDA context (the
    # per-worker ~300 MB context class of bug), and pin the thread pools the same way the
    # workers themselves do.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    import torch
    torch.set_num_threads(1)
    # Inductor's parallel codegen pool is the FIRST of the two thread sources the original hang
    # diagnosis named; a single compile thread never creates it.
    torch._inductor.config.compile_threads = 1

    import gymnasium as gym
    import numpy as np

    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    cfg = json.loads(cfg_json)
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    import inspect
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kwargs = {k: v for k, v in cfg.items() if k in sig}
    # `Gen3FeaturesExtractor` never READS `observation_space` (its annotation says so —
    # `spaces.Space`, deliberately unread); every serverless probe path passes the flat
    # `Box` the encoder describes.
    fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kwargs)
    fe.eval()
    fe.disable_observation_debugger()                   # dynamo cannot trace its numpy asserts
    compiled = torch.compile(fe.forward)
    with torch.no_grad():
        obs = {"observation": torch.zeros(1, layout["total_dim"])}
        compiled(obs)                                   # trace + codegen happen HERE, once
    # keep the compiled artifact alive for the fork: workers' torch.compile of the SAME code
    # object hits the in-memory caches this populated.
    globals()["_preloaded"] = (fe, compiled)

    try:
        torch._inductor.async_compile.shutdown_compile_workers()
    except Exception:                                   # noqa: BLE001 — pool may not exist at 1 thread
        pass

    extra = [t.name for t in threading.enumerate() if t is not threading.main_thread()]
    if extra:
        # LOUD BY DESIGN: raising here kills the forkserver bootstrap, so SubprocVecEnv
        # construction fails with a traceback in the parent — the silent 2-of-48-workers wedge
        # this preload's predecessor caused is unrepresentable.
        raise RuntimeError(
            f"[CompilePreload] REFUSING to let the forkserver fork: {len(extra)} non-main "
            f"thread(s) survive the preload compile ({', '.join(sorted(extra))}). fork() from a "
            "multi-threaded process copies held mutexes without their owners — the exact "
            "mechanism of the 2026-08 preload hang. Drop --compile-opponents-preload, or find "
            "and kill the thread source.")
    print(f"[CompilePreload] forkserver compiled the extractor in {time.perf_counter() - t0:.1f}s "
          f"(single-threaded — workers inherit the traced graph)", file=sys.stderr)


_preload()
