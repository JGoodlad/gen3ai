"""CLI for the K8 compile inventory. See ``main/compile_inventory/__init__.py``.

🚨 EVERY heavy command runs under a memory cap (scripts/ops/mem_cap.sh; 2026-09-30: an uncapped
stack-profiled trace OOM-killed the host three times — `memcap.py`), at nice 19:

  scripts/ops/mem_cap.sh 12 nice -n 19 python -m main.compile_inventory run --stage trace \
        --device cpu --micro 64 --trace-steps 4                       # (a)(b)(c), both views
  scripts/ops/gpu_lock.sh timeout 1200 scripts/ops/mem_cap.sh 16 python -m main.compile_inventory \
        run --stage time --device cuda --time-epochs 1 --keep-prewarm        # (d)
  scripts/ops/mem_cap.sh 8 python -m main.compile_inventory fullgraph  # fullgraph=True verdicts
  python -m main.compile_inventory analyze <out dir>                  # re-classify (streaming)
  python -m main.compile_inventory report <out dir> [...] --out <tables.md>

Run the parent under the interpreter you want measured — the worker is spawned with
``sys.executable`` (``gen3ai_stable`` = torch 2.5.1, ``gen3ai_torch28`` = torch 2.8).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_OUT_ROOT = Path.home() / "gen3ai_archive" / "k8_inventory"
DEFAULT_RUN = "ai_v14_02_lbat_ctrl"
DEFAULT_CHECKPOINT = "final_model.zip"
#: The learner benchmark's pinned full rollout of arm C's final checkpoint (98,304 rows).
DEFAULT_BUFFER = (Path.home() / "gen3ai_archive" / "learner_bench" / "20260928_135948_cuda"
                  / "rollout_buffer.pkl")
#: Env workers the trainer builds: the buffer replaces the rollout, so 2 is enough.
WORKER_N_ENVS = "2"


def _write(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def adjust_argv(argv: List[str], *, compile_trainer: bool) -> List[str]:
    """The benchmark's re-pointed trainer argv, with the inventory's two changes: few env workers,
    no opponent compile (neither is exercised — the buffer replaces the rollout), and the learner
    compile ON (time) or OFF (trace: the capture compiles, from a clean dynamo)."""
    from agents.training.learner_benchmark import split_flags
    drop = {"--n-envs", "--n_envs", "--compile-opponents", "--no-compile-opponents",
            "--compile-opponents-strict", "--compile-opponents-preload",
            "--no-compile-opponents-preload", "--compile-trainer", "--no-compile-trainer"}
    out: List[str] = []
    for flag, vals in split_flags(argv):
        if flag in drop:
            continue
        out.append(flag)
        out.extend(vals)
    out += ["--n-envs", WORKER_N_ENVS, "--no-compile-opponents",
            "--compile-trainer" if compile_trainer else "--no-compile-trainer"]
    return out


def _descendants(pid: int) -> List[int]:
    from agents.training.learner_benchmark import descendants
    return descendants(pid)


def cmd_run(a: argparse.Namespace) -> int:
    import torch

    from agents.training import learner_benchmark as lb
    from main.checkargs import LAUNCHER_ONLY
    from utils.git import get_git_hash
    from utils.paths import main_models_dir, repo_root

    if a.stage == "time" and a.device != "cuda":
        print("REFUSED: the time stage measures the compiled GPU learner; pass --device cuda.")
        return 2
    if a.stage == "time" and not a.keep_prewarm:
        # The K8 regions are installed, gated, prewarmed and locked by ONE call (the compile sentinel), so
        # there is no "skip the prewarm" variant: the old one replaced the sentinel wholesale, never
        # installed the regions, and the stage raised "not compiled" after the whole startup.
        print("REFUSED: the time stage runs the production compile sentinel (region install + gate + "
              "prewarm + lock, ~8 min of cold compile); pass --keep-prewarm.")
        return 2
    if getattr(a, "allow_stack_profile", False):
        from main.compile_inventory.memcap import (StackProfileRefused, memory_cap_bytes,
                                                   require_cap_for_stack_profile)
        try:
            require_cap_for_stack_profile(True, memory_cap_bytes())
        except StackProfileRefused as e:
            print(f"REFUSED: {e}")
            return 2
    models = main_models_dir()
    model_zip = Path(a.model).resolve() if a.model else (
        (models / a.run / DEFAULT_CHECKPOINT).resolve() if models else None)
    if model_zip is None or not model_zip.is_file():
        print(f"REFUSED: checkpoint not found: {model_zip}")
        return 2
    buffer = Path(a.buffer).resolve()
    if not buffer.is_file():
        print(f"REFUSED: pinned buffer not found: {buffer}")
        return 2
    sha = lb._sha256(model_zip)
    meta_json = buffer.with_suffix(".json")
    if meta_json.exists():
        bsha = json.loads(meta_json.read_text()).get("checkpoint_sha256")
        if bsha and bsha != sha:
            print(f"REFUSED: the buffer was collected by checkpoint {bsha[:12]}, not {sha[:12]} — "
                  f"its old log-probs and values would belong to a different policy.")
            return 2
    if a.device == "cuda" and a.stage == "time":
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=30).stdout
        why = lb.gpu_busy_reason(apps, lb.read_cmdlines(), os.getpid())
        if why:
            print(f"REFUSED: the GPU is not idle — {why}. The time stage is a measurement.")
            return 2
    run_src = model_zip.parent if model_zip.parent.name != "checkpoints" else model_zip.parent.parent
    original = json.loads((run_src / "metadata.json").read_text()).get("original_command")
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tv = torch.__version__.split("+")[0]
    out = Path(a.out_root).expanduser().resolve() / f"{stamp}_{a.stage}_{a.device}_torch{tv}"
    if models is not None and str(out).startswith(str(models.resolve())):
        print("REFUSED: --out-root is under models/.")
        return 2
    out.mkdir(parents=True, exist_ok=False)
    steps = (lb.recorded_steps(original) or 0) + 10 * 48 * 2048
    argv = lb.build_trainer_argv(original, model_zip=str(model_zip), run_dir=str(out / "run"),
                                 steps=steps, device=a.device, launcher_only=LAUNCHER_ONLY)
    argv = adjust_argv(argv, compile_trainer=(a.stage == "time"))
    cfg: Dict[str, Any] = {
        "stage": a.stage, "model_zip": str(model_zip), "model_sha256": sha,
        "git_head": get_git_hash(), "buffer_in": str(buffer), "out_dir": str(out),
        "trainer_argv": argv, "seed": a.seed, "backend": a.backend,
        "trace_steps": a.trace_steps, "accum": a.accum, "epochs": a.epochs, "micro": a.micro,
        "profile_epochs": a.profile_epochs, "views": a.views.split(","),
        "profile_trace": not a.no_profile, "time_config": "diag_skipped",
        "classify_in_worker": bool(a.classify_in_worker),
        "allow_stack_profile": bool(a.allow_stack_profile),
        "unbracketed": bool(a.unbracketed),
        "profiles": [["skipped", "diag_skipped", a.time_epochs]]
                    + ([["diag", "baseline", 1]] if a.profile_diag else []),
        "cwd": str(models.parent if models is not None else repo_root()),
        "torch": torch.__version__, "python": sys.executable,
    }
    env = dict(os.environ)
    src = str(repo_root() / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # K3: a hermetic, on-disk compile cache per invocation (never /tmp's shared tmpfs one).
    env["TORCHINDUCTOR_CACHE_DIR"] = str(out / "inductor_cache")
    env["TRITON_CACHE_DIR"] = str(out / "triton_cache")
    if a.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    if a.stage == "trace" and a.backend != "inductor":
        env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"   # no idle 16-process compile pool
    cfg_path = out / "worker.json"
    _write(cfg_path, cfg)
    log = out / "worker.log"
    cmd = [sys.executable, "-u", "-m", "main.compile_inventory", "_worker", str(cfg_path)]
    print(f"[compile_inventory] {a.stage}/{a.device}/torch {torch.__version__}: log {log}",
          flush=True)
    t0 = time.monotonic()
    with open(log, "wb") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=cfg["cwd"])
        try:
            rc = proc.wait(timeout=a.worker_timeout_min * 60.0)
        except subprocess.TimeoutExpired:
            from agents.training.learner_benchmark import terminate_pids
            pids = _descendants(proc.pid) + [proc.pid]
            terminate_pids(pids)
            rc = -9
            print(f"[compile_inventory] worker exceeded {a.worker_timeout_min} min — killed {pids}")
    res_path = out / f"{a.stage}_result.json"
    status = json.loads(res_path.read_text()).get("status") if res_path.exists() else "missing"
    print(f"[compile_inventory] worker exit {rc}, status {status}, "
          f"{time.monotonic() - t0:.0f} s; out {out}", flush=True)
    if status == "ok" and not a.defer_analyze:
        return cmd_analyze(argparse.Namespace(out_dir=str(out)))
    return 0 if (rc == 0 and status == "ok") else 1


def _peak_rss_mb() -> int:
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def cmd_analyze(a: argparse.Namespace) -> int:
    """Classify a finished worker's traces — in THIS process, after the worker (and its model and
    buffer) are gone, with the streaming loader. Trace stage: writes each view's steady-state
    classification back into ``trace_result.json``. Time stage: writes ``time_analysis.json``."""
    from main.compile_inventory.trace_classify import classify_trace
    out = Path(a.out_dir)
    tr = out / "trace_result.json"
    if tr.exists():
        res = json.loads(tr.read_text())
        for v in ("today", "whole_step"):
            u3 = ((res.get(v) or {}).get("calls") or {}).get("update3") or {}
            if u3.get("trace") and not u3.get("classified"):
                print(f"[compile_inventory] classifying {u3['trace']} ...", flush=True)
                u3["classified"] = classify_trace(u3["trace"], segment_names=u3.get("segments"))
        res["analyze_peak_rss_mb"] = _peak_rss_mb()
        _write(tr, res)
        print(f"[compile_inventory] classified into {tr} (analyze peak RSS "
              f"{res['analyze_peak_rss_mb']} MB)")
        return 0
    res = json.loads((out / "time_result.json").read_text())
    analysis: Dict[str, Any] = {"source": str(out / "time_result.json"),
                                "torch": res.get("torch"),
                                "matmul_precision": res.get("matmul_precision"),
                                "geometry": res.get("geometry"),
                                "bracketed": res.get("bracketed"),
                                "unbracketed": res.get("unbracketed"),
                                "rss_peak_mb_by_stage": res.get("rss_peak_mb_by_stage"),
                                "profiles": []}
    for p in res.get("profiled", []):
        print(f"[compile_inventory] classifying {p['trace']} ...", flush=True)
        c = classify_trace(p["trace"], segment_names=p.get("segments"))
        analysis["profiles"].append({"tag": p.get("tag"), "config": p.get("config"),
                                     "epochs": p.get("epochs"), "wall_s": p.get("wall_s"),
                                     **c})
    analysis["analyze_peak_rss_mb"] = _peak_rss_mb()
    _write(out / "time_analysis.json", analysis)
    print(f"[compile_inventory] wrote {out / 'time_analysis.json'}")
    return 0


def cmd_fullgraph(a: argparse.Namespace) -> int:
    from main.compile_inventory.fullgraph_check import run
    res = run(batch=a.batch, backend=a.backend)
    for k, v in res["results"].items():
        print(f"{k:58s} {v}")
    if a.out:
        _write(Path(a.out), res)
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    from main.compile_inventory.report import render
    md = render([Path(d) for d in a.out_dirs])
    if a.out:
        Path(a.out).write_text(md)
        print(f"[compile_inventory] wrote {a.out}")
    else:
        print(md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m main.compile_inventory",
                                description=(__doc__ or "").split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="spawn the trainer-hosted worker for one stage")
    r.add_argument("--stage", choices=("trace", "time"), required=True)
    r.add_argument("--device", choices=("cpu", "cuda"), required=True)
    r.add_argument("--run", default=DEFAULT_RUN)
    r.add_argument("--model", default=None, help="explicit checkpoint .zip")
    r.add_argument("--buffer", default=str(DEFAULT_BUFFER),
                   help="the pinned rollout buffer (learner_benchmark's rollout_buffer.pkl)")
    r.add_argument("--backend", choices=("aot_eager", "eager", "inductor"), default="aot_eager",
                   help="trace stage: the backend the capture compiles with")
    r.add_argument("--trace-steps", type=int, default=128,
                   help="trace stage: rollout steps kept (x n_envs rows; 128 x 48 = 3 micro-batches)")
    r.add_argument("--micro", type=int, default=None,
                   help="trace stage: micro-batch size (default: the run's own)")
    r.add_argument("--accum", type=int, default=2, help="trace stage: grad-accum group size")
    r.add_argument("--epochs", type=int, default=2, help="trace stage: epochs per update")
    r.add_argument("--profile-epochs", type=int, default=1,
                   help="trace stage: epochs of the profiled steady-state update")
    r.add_argument("--no-profile", action="store_true", help="trace stage: skip the op profile")
    r.add_argument("--allow-stack-profile", action="store_true",
                   help="trace stage: record Python stacks in the CPU profile (per-component eager "
                        "attribution). REFUSED unless this process runs under a cgroup memory cap "
                        "(see memcap.py: three host OOM kills on 2026-09-30)")
    r.add_argument("--classify-in-worker", action="store_true",
                   help="classify the profile INSIDE the worker (the pre-2026-09-30 path; it held "
                        "the model, the buffer and the whole JSON at once — diagnostic only)")
    r.add_argument("--views", default="today,whole_step")
    r.add_argument("--time-epochs", type=int, default=2,
                   help="time stage: epochs of the profiled update")
    r.add_argument("--keep-prewarm", action="store_true",
                   help="time stage: REQUIRED — the production compile sentinel (region install + "
                        "gate + prewarm + lock; ~8 min cold). The sentinel is the only thing that "
                        "installs the K8 regions, so a time stage cannot skip it")
    r.add_argument("--unbracketed", action="store_true",
                   help="time stage: also time one un-bracketed full update")
    r.add_argument("--profile-diag", action="store_true",
                   help="time stage: also profile one epoch of a DIAGNOSTIC update")
    r.add_argument("--defer-analyze", action="store_true")
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    r.add_argument("--worker-timeout-min", type=float, default=120.0)
    an = sub.add_parser("analyze", help="classify a time stage's traces")
    an.add_argument("out_dir")
    fg = sub.add_parser("fullgraph", help="compile each candidate region fullgraph=True (CPU, "
                                           "production surface) and report the refusals")
    fg.add_argument("--batch", type=int, default=16)
    fg.add_argument("--backend", default="eager", choices=("eager", "aot_eager"))
    fg.add_argument("--out", default=None, help="also write the JSON here")
    rp = sub.add_parser("report", help="render the markdown readout tables")
    rp.add_argument("out_dirs", nargs="+")
    rp.add_argument("--out", default=None)
    w = sub.add_parser("_worker", help=argparse.SUPPRESS)
    w.add_argument("cfg")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "_worker":
        from main.compile_inventory.worker import worker_main
        worker_main(a.cfg)
        return 0
    return {"run": cmd_run, "analyze": cmd_analyze, "report": cmd_report,
            "fullgraph": cmd_fullgraph}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
