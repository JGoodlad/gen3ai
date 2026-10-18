"""THE UPDATE CELL of the frozen GPU benchmark suite (`gpu_bench_suite`, T13) — the learner's real `train()` at the
production shape on a FIXED fixture rollout, compiled exactly as a launch compiles it, timed.

This is the perf phase's `update_bench.py` (`designs/research_state/measurements/perf_phase_2026-10-10/scripts/`,
the instrument behind the local cu126 / cu128 baseline of `cuda_build_2026-10-12/`) FROZEN into the package, with
its optional extras (profiler, sync sites, canary, T2) left out and the geometry made a parameter so the suite can
also run one LARGER-MICRO-BATCH point (the VRAM-headroom question) and a CPU `--quick` code path.

WHAT ONE CELL DOES (one process; the suite runs each cell in its own subprocess):

1. resolves the trainer ARGV exactly like a launch (`main.train.config.resolve_config`), builds the seeded
   production-surface learner on it (`learner_golden.build_learner`: fixed seeds, the K9 golden's recipe overrides —
   the learning rate is the golden's test constant, which no timing reads) with the cell's trunk precision, moves it
   and every optimizer's state to the device, and sets the cell's micro-batch x accumulation;
2. on CUDA: the trainer's own `_apply_grad_checkpointing`, then `compile_trainer.arm_compile_sentinel` (R1
   `fullgraph`, the STARTUP PARITY GATE on the K9 golden rows, the prewarm, the lock). A REFUSED gate is recorded
   and the cell CONTINUES with install + prewarm only, so a geometry's speed is still read (the perf phase's rule);
3. `update_fit.fixture_buffer` (the golden's REAL labelled rows tiled to the declared rows — deterministic, no run
   data), then the perf phase's schedule: ``warm`` untimed 1-epoch calls, ``reps`` timed calls per epoch count, and
   one 1-epoch DIAGNOSTIC update (the `--diagnostics-every` cadence's heavy update), each bracketed by
   `torch.cuda.synchronize()`;
4. the full update is estimated from two epoch counts (``update_s_est = median(E_lo) + (n_epochs - E_lo) x
   marginal``), the production `train/train_ms` equivalent of a PLAIN update (9 of 10 production updates).

An out-of-memory error is a RESULT, not a crash: the cell records where it ran out (`oom`) and exits 0, so the
headroom point on a small card reads "does not fit" instead of failing the suite.

A BENCHMARK WARNS, IT NEVER STRETCHES: the load average and every other GPU process are recorded, never used to
rescale. The fixture rows are the golden's 64 real rows tiled: the compiled graph is branch-free, so its cost is
data-independent; the eager tail's masked selects are not (UNVERIFIED how much that moves) — compare this cell with
ITSELF across cards, and with a live run's `train/train_ms` only as a calibration.
"""
from __future__ import annotations

import contextlib
import io
import os
import shlex
import statistics
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def resolved_args(argv: Sequence[str], device: str) -> Any:
    """The trainer namespace for ``argv`` on ``device``, resolved like a launch (CUDA adds `--compile-trainer`)."""
    from main.train.config import resolve_config
    from main.train.parser import build_parser

    p = build_parser()
    a = p.parse_args(list(argv) + ["--device", device] + (["--compile-trainer"] if device == "cuda" else []))
    with contextlib.redirect_stdout(io.StringIO()):
        resolve_config(a, p)
    return a


def gpu_others(own_pid: int) -> List[str]:
    """Every OTHER process holding a CUDA context (`nvidia-smi --query-compute-apps`), or ``["unknown: …"]``."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        return [ln.strip() for ln in q.stdout.splitlines() if ln.strip() and not ln.startswith(str(own_pid))]
    except Exception as e:  # noqa: BLE001 — recorded, never fatal
        return [f"unknown: {e!r}"]


def schedule(warm: int, epochs: Sequence[int], reps: int) -> List[Tuple[int, Any]]:
    """The call schedule ``[(epochs, timed), …]``: ``warm`` untimed 1-epoch calls (the first is the process's own
    diagnostics update), ``reps`` timed calls of each epoch count, then ONE 1-epoch diagnostic call (``"diag"``)."""
    return [(1, False)] * int(warm) + [(int(e), True) for e in epochs for _ in range(int(reps))] + [(1, "diag")]


def phase_for_call(i: int, timed: Any, every: int) -> int:
    """The ROLLOUT INDEX a call runs at (`diagnostics_cadence`: a call at phase ≡ 0 mod ``every``, or the process's
    first, is a DIAGNOSTIC update). Timed calls sit at phases 1..every-1 (PLAIN updates); the ``"diag"`` call at
    ``every`` (a diagnostic one). With ``every`` <= 1 every update is diagnostic and the index is ``i``."""
    if every <= 1:
        return i
    if timed == "diag":
        return every
    return 1 + i % (every - 1)


def update_estimate(medians: Dict[int, float], n_epochs_full: int) -> Dict[str, float]:
    """The full-update estimate from per-epoch-count medians (pure).

    Two or more counts: the MARGINAL epoch is ``(m[hi] - m[lo]) / (hi - lo)`` and the update is ``m[lo] +
    (n_epochs_full - lo) x marginal`` (a call's first epoch carries the update's one-time work). One count: ``m / E x
    n_epochs_full`` (an upper bound — the one-time work is scaled with it)."""
    if not medians:
        raise ValueError("no timed calls")
    ks = sorted(medians)
    lo, hi = ks[0], ks[-1]
    if hi > lo:
        marginal = (medians[hi] - medians[lo]) / (hi - lo)
        est = medians[lo] + (n_epochs_full - lo) * marginal
    else:
        marginal = medians[lo] / lo
        est = marginal * n_epochs_full
    return {"marginal_epoch_s": marginal, "update_s_est": est,
            "first_epoch_overhead_s": medians[lo] - lo * marginal}


def is_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory error (torch's own type, or its message from a wrapped compile error)."""
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001
        pass
    return "out of memory" in str(exc).lower()


def _move_optimizers(model: Any, dev: Any) -> int:
    """A launch builds the learner ON its device; this one was built on the CPU, so every optimizer's declared state
    (the policy's AdamW, the ride-along heads' own Adams) still sits there: move it (the perf phase's window A died in
    the ride-along step without this: "cuda:0 and cpu")."""
    import torch

    seen: set = set()

    def _move(o: Any) -> None:
        if isinstance(o, torch.optim.Optimizer) and id(o) not in seen:
            seen.add(id(o))
            for st in o.state.values():
                for k, v in list(st.items()):
                    if torch.is_tensor(v) and (k != "step" or v.device.type != "cpu" or
                                               any(g.get("fused") or g.get("capturable") for g in o.param_groups)):
                        st[k] = v.to(dev)      # torch keeps a non-fused, non-capturable Adam's `step` on the CPU
    for holder in [model, model.policy] + list(model.policy.modules()):
        for v in list(vars(holder).values()):
            for o in (v if isinstance(v, (list, tuple)) else list(v.values()) if isinstance(v, dict) else [v]):
                _move(o)
    return len(seen)


def run_cell(cfg: Dict[str, Any], res: Dict[str, Any], save: Callable[[], None]) -> None:
    """Run one update cell into ``res`` (saved incrementally through ``save``). ``cfg`` keys: ``argv`` (the trainer
    argv string), ``device`` (cuda | cpu), ``trunk_precision`` (fp32 | bf16), ``n_steps`` x ``n_envs`` (the rollout's
    declared shape), ``batch_size`` / ``grad_accum`` (None = the resolved recipe's), ``epochs`` (list),
    ``warm``, ``reps``, ``gate`` (on | off)."""
    try:
        _run_cell(cfg, res, save)
    except BaseException as exc:  # noqa: BLE001 — an OOM is a result; anything else is recorded and re-raised
        if is_oom(exc):
            res["oom"] = {"where": res.get("_stage", "?"), "error": f"{type(exc).__name__}: {str(exc)[:1500]}"}
            res["status"] = "oom"
            try:
                import torch
                if torch.cuda.is_available():
                    res["oom"]["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
            except Exception:  # noqa: BLE001
                pass
            save()
            return
        raise


def _run_cell(cfg: Dict[str, Any], res: Dict[str, Any], save: Callable[[], None]) -> None:
    import torch

    from agents.model.compile_cache import ensure_hermetic_cache
    from agents.training import learner_golden as LG
    from agents.training import update_fit as UF
    from agents.training.learner_benchmark import _scalar_read_meter

    device = str(cfg["device"])
    cpu = device == "cpu"
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError("an update cell on cuda needs a visible CUDA device")
    _sync: Callable[[], None] = (lambda: None) if cpu else torch.cuda.synchronize
    n_steps, n_envs = int(cfg["n_steps"]), int(cfg["n_envs"])
    res.update({"_stage": "build", "load1_start": os.getloadavg()[0],
                "gpu_others_start": [] if cpu else gpu_others(os.getpid()), "phases_s": {}})
    ensure_hermetic_cache(f"gpu bench suite update cell {cfg.get('name', '')}")
    torch.set_float32_matmul_precision("highest")
    if not cpu:
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    args = resolved_args(shlex.split(str(cfg["argv"])), device)
    LG.N_STEPS, LG.N_ENVS = n_steps, n_envs
    model = LG.build_learner(args=args, trunk_precision=str(cfg["trunk_precision"]))
    dev = torch.device(device)
    model.policy.to(dev)
    model.device = dev
    res["optimizers_moved"] = _move_optimizers(model, dev)
    model.rollout_buffer.device = dev
    model.batch_size = int(cfg.get("batch_size") or args.batch_size)
    model.grad_accum_steps = int(cfg.get("grad_accum") or getattr(args, "grad_accum_steps", 1) or 1)
    n_epochs_full = int(args.n_epochs)
    rows = n_steps * n_envs
    if rows % model.batch_size:
        raise ValueError(f"rows {rows} do not divide by the micro-batch {model.batch_size}")
    res["shape"] = {"rows": rows, "n_steps": n_steps, "n_envs": n_envs, "batch_size": model.batch_size,
                    "grad_accum": model.grad_accum_steps, "effective_batch": model.batch_size * model.grad_accum_steps,
                    "micro_batches_per_epoch": rows // model.batch_size, "n_epochs_full": n_epochs_full,
                    "params": sum(p.numel() for p in model.policy.parameters()),
                    "diagnostics_every": getattr(model, "diagnostics_every", None)}
    from main.train.lifecycle import _apply_grad_checkpointing
    _apply_grad_checkpointing(model, bool(getattr(args, "grad_checkpointing", False)))
    res["grad_checkpointing"] = bool(getattr(args, "grad_checkpointing", False))
    res["phases_s"]["build"] = time.perf_counter() - t0
    if not cpu:
        res["peak_after_build_mib"] = torch.cuda.max_memory_reserved() / 2**20
    save()

    # ---- compile: the launch's own sentinel (gate + prewarm + lock), or install + prewarm without the gate
    res["_stage"] = "compile"
    t0 = time.perf_counter()
    lines: List[str] = []
    if cpu:
        res["gate"] = {"verdict": "SKIPPED", "error": "cpu: eager, nothing compiled"}
    else:
        from agents.model import compile_regions as cr
        from agents.model import compile_trainer as ct
        from agents.model.compile_control import control
        try:
            if str(cfg.get("gate", "on")) == "off":
                raise ct.CompileTrainerError("gate skipped by the cell's config")
            ct.arm_compile_sentinel(model, batch_size=model.batch_size, emit=lines.append)
            res["gate"] = {"verdict": "PASS", "lines": [ln for ln in lines if "parity" in ln.lower() or "R1" in ln][-20:]}
        except ct.CompileTrainerError as exc:
            if is_oom(exc):
                raise
            res["gate"] = {"verdict": "SKIPPED" if str(cfg.get("gate", "on")) == "off" else "REFUSED",
                           "error": str(exc)[:4000], "lines": lines[-20:]}
            save()
            cr.uninstall(model)
            ctl = control(lines.append).install()
            ctl.reset()
            cr.install(model)
            ctl.prewarm(cr.prewarm_calls(model, batch_size=model.batch_size))
    _sync()
    res["phases_s"]["compile_gate_prewarm"] = time.perf_counter() - t0
    if not cpu:
        res["peak_after_compile_mib"] = torch.cuda.max_memory_reserved() / 2**20
    save()

    # ---- the timed updates on the fixture rollout
    res["_stage"] = "train"
    live_buf = model.rollout_buffer          # the template (its declared shape); never trained on

    def fresh_buffer() -> Any:
        # A FRESH fixture per train(): `get()` flattens the buffer IN PLACE and train() re-aligns the opponent-intent
        # labels in it, so a buffer is single-use, exactly like a real rollout's.
        model.rollout_buffer = live_buf
        b, src = UF.fixture_buffer(model)
        model.rollout_buffer = b
        return src

    res["fixture_source"] = fresh_buffer()
    model.behaviour_check = "off"
    every = int(getattr(model, "diagnostics_every", 1) or 1)
    epochs_list = [int(x) for x in cfg["epochs"]]
    sched = schedule(int(cfg["warm"]), epochs_list, int(cfg["reps"]))
    walls: Dict[int, List[float]] = {e: [] for e in epochs_list}
    calls: List[Dict[str, Any]] = []
    for i, (ep, timed) in enumerate(sched):
        if i:
            fresh_buffer()
        model.num_timesteps = rows * phase_for_call(i, timed, every)
        model.n_epochs = ep
        acc = {"s": 0.0, "n": 0}
        _sync()
        if not cpu:
            torch.cuda.reset_peak_memory_stats()
        with _scalar_read_meter(acc):
            t = time.perf_counter()
            model.train()
            _sync()
            w = time.perf_counter() - t
        peak_r = 0.0 if cpu else torch.cuda.max_memory_reserved() / 2**20
        peak_a = 0.0 if cpu else torch.cuda.max_memory_allocated() / 2**20
        calls.append({"call": i, "epochs": ep, "kind": "diag" if timed == "diag" else ("timed" if timed else "warm"),
                      "wall_s": w, "scalar_reads": acc["n"], "scalar_read_s": acc["s"],
                      "peak_reserved_mib": peak_r, "peak_alloc_mib": peak_a})
        if timed == "diag":
            res["diag_1ep_s"] = w
        elif timed:
            walls[ep].append(w)
        res["peak_reserved_mib"] = max(float(res.get("peak_reserved_mib", 0.0)), peak_r)
        res["peak_alloc_mib"] = max(float(res.get("peak_alloc_mib", 0.0)), peak_a)
        res["calls"] = calls
        print(f"[gpu_bench update {cfg.get('name', '')}] train {i} ({calls[-1]['kind']}, {ep} ep): {w:.2f}s, "
              f"peak {peak_r:.0f} MiB", flush=True)
        save()
    meds = {e: statistics.median(v) for e, v in walls.items() if v}
    est = update_estimate(meds, n_epochs_full)
    spread = {str(e): (max(v) - min(v)) / statistics.median(v) for e, v in walls.items() if len(v) > 1}
    res["train"] = {"walls_s": {str(k): v for k, v in walls.items()}, "median_s": {str(k): v for k, v in meds.items()},
                    **est, "per_micro_ms_marginal": 1e3 * est["marginal_epoch_s"] / (rows // model.batch_size),
                    "spread_rel": spread}
    res["load1_end"] = os.getloadavg()[0]
    res["gpu_others_end"] = [] if cpu else gpu_others(os.getpid())
    res["status"] = "ok"
    res.pop("_stage", None)
    save()


def headroom_points(base_batch: int, base_accum: int, factors: Sequence[int]) -> List[Tuple[int, int]]:
    """The larger-micro-batch points at the SAME effective batch: ``(base_batch x f, base_accum / f)`` for each
    factor that divides the accumulation (pure). Production 2,048 x 32 -> (8,192 x 8), (4,096 x 16) for (4, 2)."""
    out = []
    for f in factors:
        f = int(f)
        if f > 1 and base_accum % f == 0:
            out.append((base_batch * f, base_accum // f))
    return out


def optional_float(x: Any) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None
