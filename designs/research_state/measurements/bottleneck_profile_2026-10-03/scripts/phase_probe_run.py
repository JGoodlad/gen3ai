"""Run the trainer UNCHANGED, with phase markers added from outside (bottleneck profile, 2026-10-03).

    PROF_PHASE_LOG=<file.jsonl> python phase_probe_run.py <trainer argv...>

It imports the trainer's entry module, wraps a handful of CLASS attributes with a thin timer that
(a) pushes / pops an NVTX range (so an `nsys` timeline names the phase) and (b) appends one JSON line
per call with its wall-clock start / end, then runs the trainer's own ``main()``. Nothing inside a
compiled region is wrapped; the wrappers keep ``__wrapped__`` (``functools.wraps``), so every
``inspect.getsource`` pin still reads the original source.

Wrapped (phase name -> attribute):
  collect  InstrumentedMaskablePPO.collect_rollouts   (the whole rollout incl. any eval fired inside)
  update   InstrumentedMaskablePPO.train              (one PPO update; the startup dry update too)
  eval     SelfPlayCallback._launch_eval              (one blocking eval cycle, incl. snapshot save)
  eval_play  selfplay_callback.launch_rust_eval_cycle (the games only)
  dump     OwnedLoop.dump_logs
  NVTX only (too frequent for the JSONL): RustCollector.prepare / step_core / finish, service flush.

The process argv deliberately does not carry the trainer script's name (the live-run watchers match it).
"""
from __future__ import annotations

import multiprocessing
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import asyncio  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

_LOG_PATH = os.environ.get("PROF_PHASE_LOG")
_LOCK = threading.Lock()
_FH = open(_LOG_PATH, "a", buffering=1) if _LOG_PATH else None


def _emit(row: dict) -> None:
    if _FH is None:
        return
    with _LOCK:
        _FH.write(json.dumps(row) + "\n")


def _nvtx():
    try:
        import torch
        return torch.cuda.nvtx
    except Exception:  # pragma: no cover
        return None


def _wrap(owner, attr: str, phase: str, *, log: bool = True) -> None:
    orig = getattr(owner, attr)
    nv = _nvtx()

    @functools.wraps(orig)
    def wrapper(*a, **k):
        t0 = time.time()
        if nv is not None:
            nv.range_push(phase)
        ok = False
        try:
            out = orig(*a, **k)
            ok = True
            return out
        finally:
            if nv is not None:
                nv.range_pop()
            if log:
                _emit({"phase": phase, "t0": t0, "t1": time.time(), "ok": ok,
                       "tid": threading.get_ident()})

    setattr(owner, attr, wrapper)


def _torch_profile_update(owner, k_target: int, out_prefix: str) -> None:
    """PROF_TORCH_UPDATE=K: run the K-th call of ``train`` (1 = the startup dry update) under
    ``torch.profiler`` (CPU + CUDA activities, record_shapes, with_flops, NO with_stack — the 2026-09-30
    OOM was with_stack) and write the op/kernel tables. Every other call is untouched."""
    orig = getattr(owner, "train")
    state = {"n": 0}

    @functools.wraps(orig)
    def wrapper(*a, **k):
        state["n"] += 1
        if state["n"] != k_target:
            return orig(*a, **k)
        import torch
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                     with_flops=True, with_stack=False) as prof:
            out = orig(*a, **k)
            torch.cuda.synchronize()
        ka = prof.key_averages()
        with open(out_prefix + "_ops.txt", "w") as f:
            f.write(ka.table(sort_by="self_cuda_time_total", row_limit=80))
        with open(out_prefix + "_shapes.txt", "w") as f:
            f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total",
                                                                       row_limit=120))
        rows = []
        for e in prof.key_averages(group_by_input_shape=True):
            rows.append({"name": e.key, "shapes": str(e.input_shapes), "count": e.count,
                         "self_cuda_us": getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)),
                         "cuda_us": getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)),
                         "cpu_us": e.cpu_time_total, "self_cpu_us": e.self_cpu_time_total, "flops": e.flops})
        with open(out_prefix + "_rows.json", "w") as f:
            json.dump(rows, f)
        _emit({"phase": "torch_profile_written", "t0": time.time(), "k": k_target})
        return out

    setattr(owner, "train", wrapper)


def main() -> None:
    import main.train_rl_agent as T  # the trainer's entry module (all phases import through it)
    from agents.training import selfplay_callback as SP
    from agents.training.instrumented_ppo.loop import OwnedLoop
    from agents.training.instrumented_ppo.ppo import InstrumentedMaskablePPO
    from agents.training.rust_rollout.collector import RustCollector
    from agents.inference.service import service as SVC

    _wrap(InstrumentedMaskablePPO, "collect_rollouts", "collect")
    if os.environ.get("PROF_TORCH_UPDATE"):
        _torch_profile_update(InstrumentedMaskablePPO, int(os.environ["PROF_TORCH_UPDATE"]),
                              os.environ.get("PROF_TORCH_OUT", "/tmp/torch_update"))
    _wrap(InstrumentedMaskablePPO, "train", "update")
    _wrap(OwnedLoop, "dump_logs", "dump")
    _wrap(SP.SelfPlayCallback, "_launch_eval", "eval")
    _wrap(SP, "launch_rust_eval_cycle", "eval_play")
    _wrap(RustCollector, "prepare", "c_prepare", log=False)
    _wrap(RustCollector, "step_core", "c_core", log=False)
    _wrap(RustCollector, "finish", "c_post", log=False)
    svc_cls = next((getattr(SVC, n) for n in dir(SVC)
                    if isinstance(getattr(SVC, n), type) and hasattr(getattr(SVC, n), "flush")
                    and getattr(getattr(SVC, n), "__module__", "") == SVC.__name__), None)
    if svc_cls is not None:
        _wrap(svc_cls, "flush", "t2_flush", log=False)
    _emit({"phase": "start", "t0": time.time(), "pid": os.getpid(), "argv": sys.argv[1:],
           "svc_cls": getattr(svc_cls, "__name__", None)})
    sys.argv = [T.__file__] + sys.argv[1:]
    asyncio.run(T.main())


if __name__ == "__main__":
    main()
