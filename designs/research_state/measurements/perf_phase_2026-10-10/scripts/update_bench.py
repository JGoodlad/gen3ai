"""THE UPDATE BENCHMARK on any architecture (T25 perf phase, 2026-10-10) — the real `train()` of a learner
built for a trainer ARGV, at the production shape, compiled exactly as a launch compiles it, timed.

    # under the GPU lease (gpu_lock + mem_cap), from any cwd; this checkout's src/ is put first (DIAG_SRC overrides)
    python update_bench.py --tag E --argv "--arch endstate <ride-along flags>" --epochs 1,3 --reps 2 --out ../results/E.json

WHAT IT RUNS. `learner_golden.build_learner(args=<the argv, resolved like a launch>)` with the rollout buffer at the
production shape (n_steps 384 x n_envs 256 = 98,304 rows) and the recipe's batch / accumulation (2,048 x 32), on
CUDA; the trainer's own `_apply_grad_checkpointing` then `compile_trainer.arm_compile_sentinel` (R1 installed
`fullgraph=True`, the STARTUP PARITY GATE on the K9 golden rows, the prewarm, the lock) — the gate's verdict is
recorded, and a refusal is recorded and the run CONTINUES uninstalled-gate (install + prewarm only) so a lever's
speed is still read. Then `update_fit.fixture_buffer` (the golden's real labelled rows tiled to 98,304) and
`--reps` timed `train()` calls of `--epochs` epochs each after `--warm` untimed ones, each bracketed by
`torch.cuda.synchronize()`. A rep's wall is the trainer's own `train/train_ms`-equivalent; a full update is
`n_epochs` (10) epochs, so `update_s_est` = per-epoch median x 10 (every epoch runs the same micro-batches).

Optional: `--profile` one more epoch under torch.profiler (GPU busy %, kernel classes, launches + syncs per
micro-batch — `learner_benchmark.summarise_profile` + a kernel-class table); `--canary` the compile canary's own
compiled-vs-eager check on the trained weights after the reps; `--t2 256[,64]` the T2 inference service (graph
backend, ONE slot of this policy, fp32) per-flush time at each bucket. Levers: `--trunk-precision bf16`
(`agents.model.trunk_precision`), `--r1-preset <name>[+<name>]` (`compile_regions.R1_INDUCTOR_PRESETS`).

A BENCHMARK WARNS, IT NEVER STRETCHES: the load average and every other GPU process are RECORDED; nothing is
rescaled. The fixture rows are the golden's 64 real rows tiled: the compiled graph is branch-free, so its cost is
data-independent; the eager tail's masked selects are not (UNVERIFIED how much that moves) — compare this tool
with ITSELF across configurations, and against a real launch's train_ms only as a calibration (PLAN.md).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict

SRC = os.environ.get("DIAG_SRC") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                   "..", "..", "..", "..", "..", "src"))
sys.path.insert(0, SRC)

N_STEPS, N_ENVS = 384, 256


def resolved_args(argv, device="cuda"):
    from main.train.config import resolve_config
    from main.train.parser import build_parser
    p = build_parser()
    a = p.parse_args(list(argv) + ["--device", device] + (["--compile-trainer"] if device == "cuda" else []))
    with contextlib.redirect_stdout(io.StringIO()):
        resolve_config(a, p)
    return a


def gpu_others():
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        return [ln.strip() for ln in q.stdout.splitlines() if ln.strip() and not ln.startswith(str(os.getpid()))]
    except Exception as e:  # noqa: BLE001
        return [f"unknown: {e!r}"]


def kernel_classes(prof):
    """Device kernel time by class (GEMM / attention / Triton pointwise / reduction / persistent / other)."""
    from torch.autograd import DeviceType
    cls = defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        if e.device_type != DeviceType.CUDA or e.name.startswith(("Memcpy", "Memset")):
            continue
        n = e.name
        if "fmha" in n or "attention" in n or "flash" in n:
            k = "attention"
        elif "gemm" in n or "sgemm" in n or "cutlass" in n or "Kernel2" in n:
            k = "gemm"
        elif n.startswith("triton_poi"):
            k = "triton_pointwise"
        elif n.startswith("triton_red"):
            k = "triton_reduction"
        elif n.startswith("triton_per"):
            k = "triton_persistent_reduction"
        elif n.startswith("triton"):
            k = "triton_other"
        else:
            k = "aten_other"
        cls[k][0] += 1
        cls[k][1] += e.time_range.end - e.time_range.start
    tot = sum(v[1] for v in cls.values()) or 1.0
    return {k: {"count": c, "s": t / 1e6, "pct": 100.0 * t / tot} for k, (c, t) in
            sorted(cls.items(), key=lambda kv: -kv[1][1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--argv", required=True, help="the trainer argv of the configuration (one string)")
    ap.add_argument("--trunk-precision", default="fp32", choices=("fp32", "bf16"))
    ap.add_argument("--r1-preset", default="default")
    ap.add_argument("--gate", default="on", choices=("on", "off"))
    ap.add_argument("--epochs", default="1,3", help="epoch counts of the timed train() calls (comma list)")
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--reps", type=int, default=2, help="timed calls per epoch count")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--t2", default="", help="comma-separated T2 buckets to time (empty = skip)")
    ap.add_argument("--t2-reps", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu-smoke", action="store_true",
                    help="the CODE PATH on CPU: 2,048 rows, batch 256 x 4, eager (no compile, no gate, no T2)")
    a = ap.parse_args()

    import numpy as np
    import torch

    res = {"tag": a.tag, "argv": a.argv, "trunk_precision": a.trunk_precision, "r1_preset": a.r1_preset,
           "src": SRC, "torch": torch.__version__, "load1_start": os.getloadavg()[0],
           "gpu_others_start": [] if a.cpu_smoke else gpu_others(), "phases_s": {}}

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f, indent=1, default=str)
        os.replace(tmp, a.out)

    global N_STEPS, N_ENVS
    cpu = a.cpu_smoke
    if cpu:
        N_STEPS, N_ENVS = 32, 64
        a.gate, a.t2 = "off", ""
        _sync = lambda: None  # noqa: E731
    else:
        assert torch.cuda.is_available(), "needs the GPU (lease + gpu_lock)"
        _sync = torch.cuda.synchronize
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache(f"perf phase update bench {a.tag}")
    torch.set_float32_matmul_precision("highest")
    if not cpu:
        torch.cuda.set_device(0)
    from agents.model import compile_regions as cr
    from agents.model import compile_trainer as ct
    from agents.training import learner_golden as LG
    from agents.training import update_fit as UF
    import agents.training.instrumented_ppo.micro_step as MS
    res["imports_from"] = MS.__file__
    print(f"[bench {a.tag}] imports from {MS.__file__}", flush=True)

    t0 = time.perf_counter()
    args = resolved_args(shlex.split(a.argv), "cpu" if cpu else "cuda")
    LG.N_STEPS, LG.N_ENVS = N_STEPS, N_ENVS
    model = LG.build_learner(args=args)
    dev = torch.device("cpu" if cpu else "cuda")
    model.policy.to(dev)
    model.device = dev
    model.rollout_buffer.device = dev
    model.batch_size = 256 if cpu else int(args.batch_size)
    model.grad_accum_steps = 4 if cpu else int(getattr(args, "grad_accum_steps", 1) or 1)
    n_epochs_full = int(args.n_epochs)
    res["shape"] = {"rows": N_STEPS * N_ENVS, "batch_size": model.batch_size, "accum": model.grad_accum_steps,
                    "n_epochs_full": n_epochs_full,
                    "params": sum(p.numel() for p in model.policy.parameters()),
                    "diagnostics_every": getattr(model, "diagnostics_every", None)}
    if a.trunk_precision != "fp32":
        from agents.model.trunk_precision import set_trunk_precision
        res["trunk_rounds_set"] = set_trunk_precision(model.policy, a.trunk_precision)
    model._r1_inductor_preset = a.r1_preset
    from main.train.lifecycle import _apply_grad_checkpointing
    _apply_grad_checkpointing(model, bool(getattr(args, "grad_checkpointing", False)))
    res["grad_checkpointing"] = bool(getattr(args, "grad_checkpointing", False))
    res["phases_s"]["build"] = time.perf_counter() - t0
    save()

    # ---- compile: the launch's own sentinel (gate + prewarm + lock), or install + prewarm without the gate
    t0 = time.perf_counter()
    lines = []
    from agents.model.compile_control import control
    try:
        if cpu:
            raise ct.CompileTrainerError("cpu smoke: eager, nothing compiled")
        if a.gate == "off":
            raise ct.CompileTrainerError("gate skipped by --gate off")
        ct.arm_compile_sentinel(model, batch_size=model.batch_size, emit=lines.append)
        res["gate"] = {"verdict": "PASS", "lines": [ln for ln in lines if "parity" in ln.lower() or "R1" in ln]}
    except ct.CompileTrainerError as exc:
        res["gate"] = {"verdict": "SKIPPED" if a.gate == "off" else "REFUSED", "error": str(exc)[:4000],
                       "lines": lines[-20:]}
        save()
        cr.uninstall(model)
        if not cpu:
            ctl = control(lines.append).install()
            print(ctl.reset(), flush=True)
            cr.install(model)
            ctl.prewarm(cr.prewarm_calls(model, batch_size=model.batch_size))
    _sync()
    res["phases_s"]["compile_gate_prewarm"] = time.perf_counter() - t0
    save()

    # ---- the timed updates on the fixture rollout
    live_buf = model.rollout_buffer          # the template (its declared shape); never trained on

    def fresh_buffer():
        # A FRESH fixture per train(): `get()` flattens the buffer IN PLACE and train() re-aligns the
        # opponent-intent labels in it, so a buffer is single-use, exactly like a real rollout's.
        model.rollout_buffer = live_buf
        b, src = UF.fixture_buffer(model)
        model.rollout_buffer = b
        return b, src

    buf, source = fresh_buffer()
    res["fixture_source"] = source
    model.behaviour_check = "off"
    from agents.training.learner_benchmark import _scalar_read_meter
    # The schedule: `--warm` untimed 1-epoch calls (the first is the process's diagnostics update), then for each
    # epoch count E in `--epochs` (e.g. "1,3"), `--reps` timed calls of E epochs. A call's FIRST epoch carries the
    # update's one-time work (label alignment, epoch-0 calibration / noise reads, the device-batch build), so the
    # MARGINAL epoch is (median(E_hi) - median(E_lo)) / (E_hi - E_lo) and a full update is estimated as
    # median(E_lo) + (n_epochs - E_lo) x marginal. With one E only, update = median / E x n_epochs (an upper bound).
    epochs_list = [int(x) for x in str(a.epochs).split(",")]
    sched = [(1, False)] * a.warm + [(e, True) for e in epochs_list for _ in range(a.reps)]
    walls = {e: [] for e in epochs_list}
    reads = []
    for i, (ep, timed) in enumerate(sched):
        if i:
            buf, _ = fresh_buffer()
        model.n_epochs = ep
        acc = {"s": 0.0, "n": 0}
        _sync()
        if not cpu:
            torch.cuda.reset_peak_memory_stats()
        with (contextlib.nullcontext() if cpu else torch.cuda.nvtx.range(f"{a.tag}_train_{i}_e{ep}")), \
                _scalar_read_meter(acc):
            t = time.perf_counter()
            model.train()
            _sync()
            w = time.perf_counter() - t
        if timed:
            walls[ep].append(w)
            reads.append({"epochs": ep, **acc})
        peak = 0.0 if cpu else torch.cuda.max_memory_reserved() / 2**20
        print(f"[bench {a.tag}] train {i} ({'timed' if timed else 'warm'}, {ep} ep): {w:.2f}s, "
              f"scalar reads {acc['n']} / {acc['s']:.2f}s, peak {peak:.0f} MiB", flush=True)
        res["peak_reserved_mib"] = max(res.get("peak_reserved_mib", 0.0), peak)
    meds = {e: statistics.median(v) for e, v in walls.items()}
    n_micro = N_STEPS * N_ENVS // model.batch_size
    if len(epochs_list) >= 2:
        lo, hi = min(epochs_list), max(epochs_list)
        marginal = (meds[hi] - meds[lo]) / (hi - lo)
        update_est = meds[lo] + (n_epochs_full - lo) * marginal
    else:
        lo = epochs_list[0]
        marginal = meds[lo] / lo
        update_est = marginal * n_epochs_full
    res["train"] = {"walls_s": {str(k): v for k, v in walls.items()}, "median_s": {str(k): v for k, v in meds.items()},
                    "marginal_epoch_s": marginal, "update_s_est": update_est,
                    "first_epoch_overhead_s": meds[lo] - lo * marginal,
                    "per_micro_ms_marginal": 1e3 * marginal / n_micro, "scalar_reads": reads}
    res["load1_end"] = os.getloadavg()[0]
    res["gpu_others_end"] = [] if cpu else gpu_others()
    save()

    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        from agents.training.learner_benchmark import summarise_profile
        buf, _ = fresh_buffer()
        model.n_epochs = 1
        acts = [ProfilerActivity.CPU] + ([] if cpu else [ProfilerActivity.CUDA])
        with profile(activities=acts) as prof:
            t = time.perf_counter()
            model.train()
            _sync()
            pw = time.perf_counter() - t
        n_micro = N_STEPS * N_ENVS // model.batch_size
        tp = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"profile_{a.tag}.trace.json.gz")
        s = summarise_profile(prof, n_micro, None if os.environ.get("BENCH_NO_TRACE") else
                              __import__("pathlib").Path(tp))
        s["wall_s_profiled"] = pw
        s["kernel_classes"] = kernel_classes(prof)
        res["profile"] = s
        save()

    if a.canary:
        try:
            ctl = control()
            can = getattr(ctl, "canary", None)
            if can is None:
                from agents.model.compile_canary import CompileCanary
                can = CompileCanary(model, batch_size=model.batch_size)
            with ctl.guard("bench canary") if ctl.locked else contextlib.nullcontext():
                res["canary"] = {"verdict": "PASS", "out": can.run()}
        except BaseException as exc:  # noqa: BLE001 — the canary raises on a disagreement; record it
            res["canary"] = {"verdict": "FAIL", "error": f"{type(exc).__name__}: {str(exc)[:3000]}"}
        save()

    if a.t2:
        import copy
        model.rollout_buffer = live_buf
        del buf
        torch.cuda.empty_cache()
        from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
        from agents.inference.service.parity import fixture_rows
        from agents.model.compile_control import control as _c
        try:                                    # the learner's lock forbids any later compile: release it
            _c().release("bench: the T2 measurement compiles after the learner's lock")
        except Exception as exc:  # noqa: BLE001
            res["t2_release_note"] = repr(exc)
        tmpl = copy.deepcopy(model.policy).eval()
        if a.trunk_precision != "fp32":
            from agents.model.trunk_precision import set_trunk_precision
            set_trunk_precision(tmpl, "fp32")
        buckets = tuple(int(b) for b in a.t2.split(","))
        t0 = time.perf_counter()
        spec = ServiceSpec(groups=(SlotGroupSpec("bench", 1, tmpl),), device="cuda", backend="graph",
                           buckets=buckets, max_rows_per_flush=max(1024, max(buckets)), lanes=1)
        try:
            svc = InferenceService(spec).startup()
            res["t2"] = {"startup_s": time.perf_counter() - t0, "per_bucket": {}}
            obs, mask = fixture_rows(svc.obs_dim, max(buckets))
            for b in buckets:
                def one():
                    svc.submit(0, obs[:b], mask[:b])
                    svc.flush()
                for _ in range(5):
                    one()
                torch.cuda.synchronize()
                ts = []
                for _ in range(a.t2_reps):
                    t = time.perf_counter()
                    one()
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t)
                res["t2"]["per_bucket"][b] = {"median_ms": 1e3 * statistics.median(ts),
                                              "p10_ms": 1e3 * float(np.percentile(ts, 10)),
                                              "p90_ms": 1e3 * float(np.percentile(ts, 90))}
                print(f"[bench {a.tag}] T2 bucket {b}: {res['t2']['per_bucket'][b]}", flush=True)
        except Exception as exc:  # noqa: BLE001
            res["t2"] = {"error": f"{type(exc).__name__}: {str(exc)[:3000]}", "tb": traceback.format_exc()[-3000:]}
        save()
    print(f"[bench {a.tag}] DONE: marginal epoch {res['train']['marginal_epoch_s']:.2f}s, update est "
          f"{res['train']['update_s_est']:.1f}s, gate {res['gate']['verdict']}", flush=True)
    save()
    sys.stdout.flush()
    os._exit(0)          # no teardown of compile workers / T2 threads


if __name__ == "__main__":
    main()
