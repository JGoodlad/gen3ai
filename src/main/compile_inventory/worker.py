"""The TRAINER-HOSTED half of the inventory: a subprocess that runs the real trainer on a real
checkpoint, restores the learner benchmark's pinned rollout buffer in place of the first rollout,
and replaces the first `train()` with one of two stages.

  trace   (CPU or CUDA, never timed) — two VIEWS of the same update under the capture engine:
            ``today``      only `features_extractor.forward` is compiled, as `--compile-trainer`
                           compiles it (the ObservationDebugger dropped, as that path drops it);
            ``whole_step`` `train()` itself is handed to dynamo, so every frame it reaches is
                           compiled or reported — the extractor, heads, masking, every loss-term
                           helper, the optimizer, the buffer, logging.
          Each view runs: update #1 (the process's first update: every diagnostic probe), update #2
          (a `--diagnostics-every` SKIPPED update), a rollout-shaped EVAL-mode no-grad forward at
          n_envs rows, then update #3 (skipped again, profiled on CPU with Python stacks: the
          compiled-vs-eager OP count per phase and component). Recompiles are labelled by the call
          that triggered them. Plus: the ObservationDebugger micro-check and the static scan of the
          fold loop.
  time    (CUDA only) — the PRODUCTION learner exactly as the trainer built it (the real
          `--compile-trainer` gate, reset, prewarm), at fp32 matmul precision: one warm-up,
          one bracketed full update (`PhaseTimer`, synchronised — the per-phase wall), then one
          update of ``profile_epochs`` epochs under `torch.profiler` (CPU + CUDA) with phase
          segments as `record_function` ranges; the trace is exported for `trace_classify`.

The worker EXITS through `learner_benchmark._exit_worker` (terminates its own descendants by
explicit PID) — the trainer's teardown, saves and final eval are never reached.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import gzip
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from agents.training import learner_benchmark as lb

#: The worker's config (the parent writes it as JSON) — module state, like the benchmark's.
_W: Dict[str, Any] = {}

#: The record_function prefix of a phase segment. `k8.seg#<i>`; the segment's NAME is the phase
#: mark that closed it, stored beside the trace (a range's name must be known when it OPENS).
SEG_MARK = "k8.seg#"


class RssSampler:
    """Samples this process's RSS every ``period`` s into ``<out>/rss.jsonl`` (flushed per row, so
    an OOM kill leaves the trail) with the current STAGE label, and keeps the peak per stage.
    Added 2026-09-30 after three host OOM kills during trace runs (74–82 GB anonymous RSS)."""

    def __init__(self, path: Path, period: float = 0.5) -> None:
        import threading
        self.path, self.period = path, period
        self.stage = "startup"
        self.peak: Dict[str, int] = {}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="k8-rss", daemon=True)

    @staticmethod
    def rss_kb() -> int:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
        return 0

    def _run(self) -> None:
        with open(self.path, "a") as f:
            while not self._stop.wait(self.period):
                kb = self.rss_kb()
                self.peak[self.stage] = max(self.peak.get(self.stage, 0), kb)
                f.write(json.dumps({"t": round(time.time(), 2), "stage": self.stage,
                                    "rss_mb": kb // 1024}) + "\n")
                f.flush()

    def set(self, stage: str) -> None:
        self.stage = stage
        print(f"[compile_inventory]   stage {stage} (rss {self.rss_kb() // 1024} MB)", flush=True)

    def start(self) -> "RssSampler":
        self._t.start()
        return self

    def peaks_mb(self) -> Dict[str, int]:
        return {k: v // 1024 for k, v in self.peak.items()}


def _stage(name: str) -> None:
    rs = _W.get("_rss")
    if rs is not None:
        rs.set(name)


class SegmentMarker:
    """A `phase_hook` that turns every phase of `train()` into a `record_function` range."""

    def __init__(self) -> None:
        self.names: List[str] = []
        self._open: Optional[Any] = None

    def __call__(self, name: str) -> None:
        import torch
        if self._open is not None:
            self._open.__exit__(None, None, None)
            self.names.append(name)
        rf = torch.profiler.record_function(f"{SEG_MARK}{len(self.names)}")
        rf.__enter__()
        self._open = rf

    def close(self) -> None:
        if self._open is not None:
            self._open.__exit__(None, None, None)
            self._open = None
            self.names.append("tail")


def slice_buffer_state(state: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """Keep the first ``steps`` rollout steps of a captured buffer state (every [n_steps, n_envs,
    ...] array, the obs dict included), so a CPU trace runs a few micro-batches of REAL rows with
    the [n_steps, n_envs] structure `_align_opp_intent_labels` needs."""
    n0 = int(state["buffer_size"])
    if steps >= n0:
        return state
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n0:
            out[k] = v[:steps].copy()
        elif isinstance(v, dict) and v and all(isinstance(x, np.ndarray) for x in v.values()):
            out[k] = {kk: (x[:steps].copy() if x.shape[0] == n0 else x) for kk, x in v.items()}
        else:
            out[k] = v
    out["buffer_size"] = steps
    out["pos"] = steps
    out["full"] = True
    out["generator_ready"] = False
    return out


def _write(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def _gzip(raw: Path) -> Path:
    gz = raw.with_suffix(raw.suffix + ".gz")
    with open(raw, "rb") as fi, gzip.open(gz, "wb", compresslevel=3) as fo:
        shutil.copyfileobj(fi, fo)
    raw.unlink()
    return gz


def _rollout_batch(model: Any, pristine: Dict[str, Any]) -> tuple:
    """Step 0 of every env, shaped as `collect_rollouts` hands it to `policy.forward`: an obs dict
    of [n_envs, ...] tensors and the NUMPY action masks."""
    import torch
    obs = {k: torch.as_tensor(v[0]).to(model.device) for k, v in pristine["observations"].items()}
    masks = np.asarray(pristine["action_masks"][0]).astype(bool)
    return obs, masks


def _restore(model: Any, pristine: Dict[str, Any], state0: Dict[str, Any], seed: int) -> None:
    lb.restore_buffer_state(model.rollout_buffer, pristine)
    lb.restore_model_state(model, state0)
    lb.seed_all(seed)
    model.logger.name_to_value.clear()


def _update(model: Any, fn: Callable[[Any], None], *, cfg_name: Optional[str], base_epochs: int,
            epochs: Optional[int] = None) -> Dict[str, Any]:
    """One `train()` through ``fn`` under an optional learner-benchmark ablation config."""
    print(f"[compile_inventory]   update ({cfg_name or 'as configured'}) "
          f"{_dt.datetime.now():%H:%M:%S}", flush=True)
    t0 = time.perf_counter()
    ctx = lb._config(model, cfg_name, base_epochs) if cfg_name else _null()
    with ctx:
        if epochs is not None:
            model.n_epochs = int(epochs)
        fn(model)
    ntv = model.logger.name_to_value
    return {"wall_s": time.perf_counter() - t0,
            "loss": float(ntv["train/loss"]) if "train/loss" in ntv else None}


class _null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: Any) -> None:
        return None


# ------------------------------------------------------------------------------------------------
# stage: trace
# ------------------------------------------------------------------------------------------------

def _drop_debugger(fe: Any) -> Dict[str, Any]:
    has = hasattr(fe, "disable_observation_debugger")
    _W["_debugger_obj"] = getattr(fe, "_debugger", None)   # kept for the micro-check only
    dropped = bool(fe.disable_observation_debugger()) if has else False
    return {"debugger_api_present": has, "was_attached": dropped}


def _debugger_microcheck(model: Any, obs: Dict[str, Any]) -> Dict[str, Any]:
    """`torch._dynamo.explain` of the extractor forward WITH the debugger attached, when this tree
    still has one (the owner removed it 2026-09-30; its commit may not have landed yet)."""
    import torch
    fe = model.policy.features_extractor
    out: Dict[str, Any] = {"present": hasattr(fe, "disable_observation_debugger")}
    if not out["present"]:
        return out
    try:
        from agents.model.observation_debugger import ObservationDebugger  # noqa: F401
    except Exception as e:  # noqa: BLE001
        out["error"] = f"cannot import the debugger: {e!r}"
        return out
    obj = _W.get("_debugger_obj")
    if obj is None:
        out["note"] = "the run's extractor had no debugger attached (log level below PERIODIC?)"
        return out
    try:
        fe._debugger = obj                  # re-attach the run's own debugger for this check only
        torch._dynamo.reset()
        ex = torch._dynamo.explain(fe.forward)(obs)
        out.update(graphs=ex.graph_count, breaks=ex.graph_break_count,
                   reasons=[str(r.reason)[:200] for r in ex.break_reasons],
                   sites=[str(r.user_stack[-1]) if r.user_stack else None
                          for r in ex.break_reasons])
    except Exception as e:  # noqa: BLE001 - the check reports, it never fails the stage
        out["error"] = repr(e)
    finally:
        fe.disable_observation_debugger()
        torch._dynamo.reset()
    return out


def _profile_cpu(model: Any, fn: Callable[[Any], None], out: Path, tag: str, *, cfg_name: str,
                 base_epochs: int, epochs: int) -> Dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile
    from agents.training.instrumented_ppo import phase_hook
    seg = SegmentMarker()
    raw = out / f"trace_{tag}.json"
    stacks = bool(_W.get("allow_stack_profile", False))
    if stacks:
        from main.compile_inventory.memcap import (memory_cap_bytes,
                                                   require_cap_for_stack_profile)
        require_cap_for_stack_profile(True, memory_cap_bytes())
    _stage(f"{tag}:profiled_update")
    with profile(activities=[ProfilerActivity.CPU], with_stack=stacks) as prof:
        with phase_hook.installed(seg):
            rec = _update(model, fn, cfg_name=cfg_name, base_epochs=base_epochs, epochs=epochs)
        seg.close()
    _stage(f"{tag}:profile_export")
    prof.export_chrome_trace(str(raw))
    del prof
    rec["trace_raw_bytes"] = raw.stat().st_size
    _stage(f"{tag}:gzip")
    gz = _gzip(raw)
    rec.update(trace=str(gz), segments=seg.names, with_stack=stacks)
    if _W.get("classify_in_worker", False):
        _stage(f"{tag}:classify")
        from main.compile_inventory.trace_classify import classify_trace
        rec["classified"] = classify_trace(gz, segment_names=seg.names)
    return rec


def _view(model: Any, name: str, orig_train: Callable[[Any], None], pristine: Dict[str, Any],
          state0: Dict[str, Any], out: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from main.compile_inventory.capture import InventoryCapture
    fe = model.policy.features_extractor
    seed = int(cfg.get("seed", 1234))
    base_epochs = int(model.n_epochs)
    obs, masks = _rollout_batch(model, pristine)
    calls: Dict[str, Any] = {}
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache(f"compile_inventory {name}")   # K3: never torch's shared default dir
    torch._dynamo.reset()
    with InventoryCapture(backend=cfg.get("backend", "aot_eager")) as cap:
        if name == "today":
            fe.forward = torch.compile(fe.forward, backend=cap.backend)
            step: Callable[[Any], None] = orig_train

            def rollout(pol: Any, o: Any, m: Any) -> Any:
                return pol(o, action_masks=m)
        else:
            step = torch.compile(orig_train, backend=cap.backend)

            def _rollout(pol: Any, o: Any, m: Any) -> Any:
                return pol(o, action_masks=m)
            rollout = torch.compile(_rollout, backend=cap.backend)
        try:
            with cap.label("update1:first(diagnostics)"):
                _stage(f"{name}:update1")
                _restore(model, pristine, state0, seed)
                calls["update1"] = _update(model, step, cfg_name=None, base_epochs=base_epochs)
            with cap.label("update2:diag_skipped"):
                _stage(f"{name}:update2")
                _restore(model, pristine, state0, seed)
                calls["update2"] = _update(model, step, cfg_name="diag_skipped",
                                           base_epochs=base_epochs)
            with cap.label("rollout:eval_no_grad"):
                _stage(f"{name}:rollout")
                model.policy.set_training_mode(False)
                t0 = time.perf_counter()
                with torch.no_grad():
                    rollout(model.policy, obs, masks)
                calls["rollout"] = {"wall_s": time.perf_counter() - t0, "rows": int(masks.shape[0])}
                model.policy.set_training_mode(True)
            with cap.label("update3:diag_skipped(steady)"):
                _stage(f"{name}:update3")
                _restore(model, pristine, state0, seed)
                if cfg.get("profile_trace", True):
                    calls["update3"] = _profile_cpu(model, step, out, f"{name}_update3",
                                                    cfg_name="diag_skipped",
                                                    base_epochs=base_epochs,
                                                    epochs=int(cfg.get("profile_epochs", 1)))
                else:
                    calls["update3"] = _update(model, step, cfg_name="diag_skipped",
                                               base_epochs=base_epochs)
        finally:
            if "forward" in vars(fe):
                del fe.forward
            model.policy.set_training_mode(True)
            for k, v in calls.items():
                print(f"[compile_inventory]   {name}/{k}: {v.get('wall_s', 0):.0f} s", flush=True)
    _stage(f"{name}:summary")
    s = cap.summary()
    s["calls"] = calls
    return s


def _stage_trace(model: Any, cfg: Dict[str, Any], out: Path) -> Dict[str, Any]:
    import torch
    orig_train = _W["orig_train"]
    buf = model.rollout_buffer
    full = lb.capture_buffer_state(buf)
    pristine = slice_buffer_state(full, int(cfg["trace_steps"]))
    del full
    lb.restore_buffer_state(buf, pristine)
    model.grad_accum_steps = int(cfg["accum"])
    model.n_epochs = int(cfg["epochs"])
    if cfg.get("micro"):
        model.batch_size = int(cfg["micro"])
    fe = model.policy.features_extractor
    if "forward" in vars(fe):           # a compiled learner (never expected: trace runs without)
        del fe.forward
    res: Dict[str, Any] = {"debugger": _drop_debugger(fe)}
    state0 = lb.capture_model_state(model)
    rows = int(pristine["buffer_size"]) * int(pristine["n_envs"])
    res["geometry"] = {"rows": rows, "micro_batch": int(model.batch_size),
                       "micro_per_epoch": -(-rows // int(model.batch_size)),
                       "grad_accum_steps": int(model.grad_accum_steps),
                       "n_epochs": int(model.n_epochs), "device": str(model.device),
                       "torch": torch.__version__,
                       "matmul_precision": torch.get_float32_matmul_precision(),
                       "diagnostics_every": getattr(model, "diagnostics_every", None)}
    from main.compile_inventory.attribution import scan_fold_host_syncs
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    res["fold_static_scan"] = scan_fold_host_syncs(_W["orig_train"])
    res["train_source_is"] = f"{InstrumentedMaskablePPO.__module__}.InstrumentedMaskablePPO.train"
    views = [v for v in ("today", "whole_step") if v in cfg.get("views", ["today", "whole_step"])]
    for v in views:
        print(f"[compile_inventory] view {v} ...", flush=True)
        t0 = time.perf_counter()
        res[v] = _view(model, v, orig_train, pristine, state0, out, cfg)
        res[v]["view_wall_s"] = time.perf_counter() - t0
        _write(out / "trace_result.partial.json", res)
        print(f"[compile_inventory] view {v}: {json.dumps(res[v]['totals'])}", flush=True)
    obs, _ = _rollout_batch(model, pristine)
    res["debugger_microcheck"] = _debugger_microcheck(model, obs)
    return res


# ------------------------------------------------------------------------------------------------
# stage: time (CUDA, the production compiled learner)
# ------------------------------------------------------------------------------------------------

def _stage_time(model: Any, cfg: Dict[str, Any], out: Path) -> Dict[str, Any]:
    import torch
    from torch.profiler import ProfilerActivity, profile
    from agents.training.instrumented_ppo import phase_hook
    orig_train = _W["orig_train"]
    fe = model.policy.features_extractor
    from agents.model import compile_regions as _cr
    res: Dict[str, Any] = {"compiled_extractor": "forward" in vars(fe),
                           "buffer_restored": bool(lb._WORKER.get("buffer_restored")),
                           "compiled_regions": _cr.installed(model),     # K8: R0 + R1 on torch 2.8
                           "torch": torch.__version__,
                           "matmul_precision": torch.get_float32_matmul_precision()}
    if not (res["compiled_extractor"] or res["compiled_regions"]):
        raise RuntimeError("the time stage measures the PRODUCTION compiled learner, and this "
                           "worker's learner is not compiled — neither the extractor nor the K8 "
                           "regions (was --compile-trainer stripped?)")
    if not lb._WORKER.get("buffer_restored"):
        # unreachable through `learn_loop_only` (the stage runs only after `_bench_collect`); a typed
        # refusal anyway, because the failure it guards is a WRONG NUMBER, not a crash (K2: 0.32 s on a
        # 4,096-row fixture read as the update)
        raise RuntimeError("the time stage measures the update on the PINNED buffer, and it was reached "
                           "before `collect_rollouts` restored that buffer — the update it would time "
                           "is not the production update")
    buf = model.rollout_buffer
    pristine = lb.capture_buffer_state(buf)
    state0 = lb.capture_model_state(model)
    seed = int(cfg.get("seed", 1234))
    base_epochs = int(model.n_epochs)
    rows = int(pristine["buffer_size"]) * int(pristine["n_envs"])
    res["geometry"] = {"rows": rows, "micro_batch": int(model.batch_size),
                       "micro_per_epoch": -(-rows // int(model.batch_size)),
                       "grad_accum_steps": int(getattr(model, "grad_accum_steps", 1)),
                       "n_epochs": base_epochs, "diagnostics_every": getattr(model,
                                                                             "diagnostics_every",
                                                                             None)}
    common = dict(pristine=pristine, state0=state0, seed=seed, base_epochs=base_epochs)
    cfgname = cfg.get("time_config", "diag_skipped")
    _stage("time:warmup")
    _restore(model, pristine, state0, seed)
    res["warmup"] = _update(model, orig_train, cfg_name=cfgname, base_epochs=base_epochs,
                            epochs=1)
    _stage("time:bracketed")
    brk = lb._one_call(model, orig_train, name=cfgname, bracketed=True, warmup=False, **common)
    res["bracketed"] = {k: brk.get(k) for k in ("train_ms", "wall_s", "phases_s", "phase_counts",
                                                  "scalar_read_s", "scalar_reads", "n_epochs",
                                                  "work")}
    if cfg.get("unbracketed", False):
        _stage("time:unbracketed")
        unb = lb._one_call(model, orig_train, name=cfgname, bracketed=False, warmup=False,
                           **common)
        res["unbracketed"] = {k: unb.get(k) for k in ("train_ms", "wall_s", "work")}
    print(f"[compile_inventory] time: bracketed {brk.get('train_ms')} ms", flush=True)
    profiled: List[Dict[str, Any]] = []
    for tag, name, epochs in cfg.get("profiles", [["skipped", cfgname, 2]]):
        _stage(f"time:profile_{tag}")
        _restore(model, pristine, state0, seed)
        seg = SegmentMarker()
        raw = out / f"trace_time_{tag}.json"
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with phase_hook.installed(seg):
                rec = _update(model, orig_train, cfg_name=name, base_epochs=base_epochs,
                              epochs=int(epochs))
                torch.cuda.synchronize()
            seg.close()
        prof.export_chrome_trace(str(raw))
        gz = _gzip(raw)
        rec.update(tag=tag, config=name, epochs=int(epochs), trace=str(gz), segments=seg.names)
        profiled.append(rec)
        _write(out / "time_result.partial.json", {**res, "profiled": profiled})
    res["profiled"] = profiled
    return res


# ------------------------------------------------------------------------------------------------
# the worker entry
# ------------------------------------------------------------------------------------------------

def _inventory_train(self: Any) -> None:
    cfg = _W
    out = Path(cfg["out_dir"])
    stage = cfg["stage"]
    res: Dict[str, Any] = {"stage": stage, "started": _dt.datetime.now().isoformat(),
                           "argv": cfg["trainer_argv"], "checkpoint": cfg["model_zip"],
                           "checkpoint_sha256": cfg["model_sha256"], "git_head": cfg["git_head"],
                           "buffer": cfg["buffer_in"]}
    rss = RssSampler(out / "rss.jsonl").start()
    _W["_rss"] = rss
    try:
        # inside `isolated_global_rng()`: each unit re-SEEDS the global streams (`lb.seed_all`, so every
        # repeat replays the same minibatch permutations) and the learner has FROZEN (K6: a global reseed
        # after the freeze is a FATAL_CONFIG); a seed inside the scope is local, restored on exit.
        from agents.training.global_rng_guard import isolated_global_rng
        with isolated_global_rng():
            body = _stage_trace(self, cfg, out) if stage == "trace" else _stage_time(self, cfg, out)
        res["rss_peak_mb_by_stage"] = rss.peaks_mb()
        res.update(body)
        res["status"] = "ok"
        res["finished"] = _dt.datetime.now().isoformat()
        _write(out / f"{stage}_result.json", res)
        lb._exit_worker(0)
    except BaseException as e:  # noqa: BLE001 - report, then leave by explicit PID
        res["status"] = "error"
        res["error"] = repr(e)
        res["traceback"] = traceback.format_exc()
        try:
            _write(out / f"{stage}_result.json", res)
        finally:
            print(res["traceback"], flush=True)
            lb._exit_worker(3)


def worker_main(cfg_path: str) -> None:
    cfg = json.loads(Path(cfg_path).read_text())
    _W.update(cfg)
    lb._WORKER.update(cfg)                   # `_bench_collect` reads `buffer_in` from here
    os.environ.pop("GEN3AI_NOISE_SCALE_PER_TERM", None)
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    # The measurement is the learn loop's update on the PINNED buffer: `train` is the stage only
    # after the loop's first collection (which restores that buffer), and the trainer's STARTUP update
    # (the CUDA fit check's dry `train()` on a fixture) runs production's own — `lb.learn_loop_only`.
    lb.install_worker_hooks(InstrumentedMaskablePPO, tool_train=_inventory_train,
                            buffer_collect=lb._bench_collect)
    _W["orig_train"] = lb._WORKER["_orig_train"]
    from main.train import model_build as _mb
    _orig_wd = _mb.start_subprocess_watchdog

    def _wd(env: Any, label: str = "env", shutdown_event: Any = None) -> Any:
        lb._WORKER["_watchdog_event"] = shutdown_event
        return _orig_wd(env, label=label, shutdown_event=shutdown_event)
    _mb.start_subprocess_watchdog = _wd
    trainer = importlib.import_module("main." + lb._TRAINER_LITERAL)
    sys.argv = ["compile_inventory[worker]"] + list(cfg["trainer_argv"])
    asyncio.run(trainer.main())
    _write(Path(cfg["out_dir"]) / f"{cfg['stage']}_result.json",
           {"status": "error", "error": "the trainer returned without reaching train()"})
    lb._exit_worker(4)

