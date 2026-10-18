"""THE FROZEN GPU BENCHMARK SUITE (T13, the GPU-upgrade decision) — ONE committed entry point that measures a card on
the production END-STATE network and writes ONE JSON (schema `gen3_gpu_bench_suite_v1`).

    python -m agents.training.gpu_bench_suite plan                      # the phases, their configs; touches nothing
    python -m agents.training.gpu_bench_suite run --out DIR             # the full suite on the visible GPU
    python -m agents.training.gpu_bench_suite run --out DIR --quick     # the CODE PATH on CPU (tiny shapes; numbers mean nothing)
    python -m agents.training.gpu_bench_suite validate DIR/gpu_bench_suite.json
    python -m agents.training.gpu_bench_suite compare A.json B.json … --reference <tag>   # the comparison table + Amdahl

THE PHASES (each its own subprocess, so a phase's compile cache, allocator state or OOM never leaks into the next;
each writes `DIR/phases/<phase>.json`, merged into `DIR/gpu_bench_suite.json` as it lands — INCREMENTAL and RESUMABLE:
a phase whose result is `ok` / `oom` is skipped on a re-run unless `--force`):

* ``host`` — the box: CPU model and thread counts, RAM, python / torch / CUDA / cuDNN build, the arch list, the GPU's
  name, capability, VRAM, driver, PCIe link, power limit, the code's commit;
* ``update_bf16`` / ``update_fp32`` — the learner's real `train()` at the production shape (98,304 rows = 384 x 256,
  micro-batch 2,048 x accumulation 32, 10 epochs estimated from timed 1- and 3-epoch calls), compiled exactly as a
  launch compiles it, at the production bf16 trunk and the fp32 opt-out (`gpu_bench_update`). The `train_ms`
  equivalent is ``train.update_s_est``;
* ``headroom_b4096`` / ``headroom_b8192`` — the SAME update at larger micro-batches with the SAME effective batch
  (4,096 x 16 and 8,192 x 8; bf16): the "does more VRAM unlock something" question — a point that does not fit
  records ``oom`` and the suite goes on;
* ``collect`` — rollout throughput through the real trainer at the production n-envs (256) against a 20-snapshot
  fixture pool at X26's self-play fraction (0.90), the trainee and the opponents on T2 (`gpu_bench_collect`): env
  steps/s per rollout and the collector's per-phase breakdown (GPU wait vs the Rust core).

Every phase is bracketed by an `nvidia-smi` snapshot (before / after) and sampled at 1 Hz while it runs (GPU
utilisation, memory, power, SM clock), and records the torch peak memory.

THE FROZEN INPUTS: the trainer argv is X26's training surface (`--arch production` + the closing test's ride-along
flags; `PRODUCTION_ARGV`), the learner is the seeded K9 build, the rollout is `update_fit.fixture_buffer` (the golden's
real labelled rows, tiled), the collect pool is seeded fresh-init copies. Nothing reads `models/`, a credential or run
data — the rented box clones the PUBLIC repo at a pinned commit and runs exactly this. A suite JSON is comparable with
another only at the same `suite_version` (bump it whenever a phase's work changes).

A BENCHMARK WARNS, IT NEVER STRETCHES (root `CLAUDE.md`): load and other GPU processes are recorded, never used to
rescale. Runbook, cost model and the comparison table: `designs/ops/cloud_gpu_benchmark.md`.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA = "gen3_gpu_bench_suite_v1"
#: Bump when any phase's WORK changes (shapes, schedule, argv, pool): two suite JSONs compare only at one version.
SUITE_VERSION = 1

#: X26's training surface (its recorded `original_command` minus the launcher / run identity): the production
#: architecture + recipe, and the closing test's ride-along heads (they train in every production update).
RIDE_ALONG = "--ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all"
PRODUCTION_ARGV = f"--arch production {RIDE_ALONG}"

#: The live calibration the Amdahl estimate starts from: X26 (rb_x26_s2601) on the local RTX 3080 Ti, read with
#: `main.ops.tb_read` over its last 20 rollouts at 258.6M steps (2026-10-18): train/train_ms median 42,241;
#: rollout/collect_ms median 26,974; time/fps median 1,314 (=> 74.8 s per 98,304-step cycle, so ~5.6 s of the cycle is
#: neither); rust_env/gpu_wait_ms_per_host_step 37.39 and host_steps 416.5 (=> 15.57 s of the collect waits on T2).
LIVE_CYCLE = {"train_s": 42.24, "collect_s": 26.97, "other_s": 5.6, "gpu_wait_s": 37.39 * 416.5 / 1000.0,
              "source": "rb_x26_s2601 tb_read last 20 rollouts @258.6M, 2026-10-18 (RTX 3080 Ti, torch 2.8.0+cu126)"}

FULL = {"n_steps": 384, "n_envs": 256, "epochs": [1, 3], "warm": 1, "reps": 2,
        "headroom": [(4096, 16), (8192, 8)], "collect_warm": 1, "collect_timed": 3,
        "pool_snapshots": 20, "pool_win_rate": 0.94, "seed": 2601}
QUICK = {"n_steps": 8, "n_envs": 4, "batch_size": 16, "grad_accum": 2, "epochs": [1, 2], "warm": 1, "reps": 1,
         "headroom": [(32, 1)], "collect_warm": 1, "collect_timed": 1, "collect_n_envs": 2,
         "collect_extra": ["--n-envs", "2", "--n-steps", "16", "--rollout-target-samples", "32", "--batch-size", "16",
                           "--grad-accum-steps", "1"],
         "pool_snapshots": 2, "pool_win_rate": 0.94, "seed": 2601}

PHASE_ORDER = ("host", "update_bf16", "update_fp32", "headroom", "collect")
#: Wall-clock bound per phase (minutes): a hung phase is killed by PID and recorded as `timeout` — NEVER a result.
PHASE_TIMEOUT_MIN = {"host": 5, "update": 40, "headroom": 40, "collect": 45}
DONE_STATUSES = ("ok", "oom")

NVSMI_FIELDS = ("name", "driver_version", "memory.total", "memory.used", "utilization.gpu", "power.draw",
                "power.limit", "clocks.sm", "clocks.max.sm", "clocks.mem", "temperature.gpu", "pstate",
                "pcie.link.gen.current", "pcie.link.width.current")
SAMPLER_FIELDS = ("utilization.gpu", "memory.used", "power.draw", "clocks.sm", "temperature.gpu")


# ------------------------------------------------------------------------------------------------- planning (pure)

def phase_names(quick: bool) -> List[str]:
    """The concrete phase list in run order (headroom expands into one phase per point)."""
    geo = QUICK if quick else FULL
    out = ["host", "update_bf16", "update_fp32"]
    out += [f"headroom_b{b}" for b, _ in geo["headroom"]]
    out.append("collect")
    return out


def phase_config(name: str, quick: bool, device: str, out_dir: str) -> Dict[str, Any]:
    """The config a phase subprocess runs (pure)."""
    geo = QUICK if quick else FULL
    base: Dict[str, Any] = {"name": name, "device": device, "quick": quick, "argv": PRODUCTION_ARGV,
                            "out": str(Path(out_dir) / "phases" / f"{name}.json")}
    if name == "host":
        return base
    if name.startswith("update_") or name.startswith("headroom_"):
        cell = dict(base, n_steps=geo["n_steps"], n_envs=geo["n_envs"], epochs=list(geo["epochs"]),
                    warm=geo["warm"], reps=geo["reps"], gate="on", batch_size=geo.get("batch_size"),
                    grad_accum=geo.get("grad_accum"))
        if name.startswith("update_"):
            cell["trunk_precision"] = name.split("_", 1)[1]
        else:
            b = int(name.split("_b", 1)[1])
            accum = dict(geo["headroom"])[b]
            cell.update(trunk_precision="bf16", batch_size=b, grad_accum=accum)
        return cell
    if name == "collect":
        rows = (geo.get("collect_n_envs") or geo["n_envs"]) * (16 if quick else geo["n_steps"])
        steps = rows * (geo["collect_warm"] + geo["collect_timed"] + 2)
        return dict(base, warm=geo["collect_warm"], timed=geo["collect_timed"], steps=steps, seed=geo["seed"],
                    pool_snapshots=geo["pool_snapshots"], pool_win_rate=geo["pool_win_rate"],
                    extra=list(geo.get("collect_extra", [])),
                    scratch=str(Path(out_dir) / "scratch" / "collect_run"))
    raise ValueError(f"unknown phase {name!r}")


def phase_timeout_min(name: str) -> float:
    key = "update" if name.startswith("update_") else "headroom" if name.startswith("headroom_") else name
    return float(PHASE_TIMEOUT_MIN[key])


# ------------------------------------------------------------------------------------------------- probes

def nvidia_smi_snapshot() -> Dict[str, Any]:
    """One `nvidia-smi` read of the card (+ every compute process), or ``{"error": …}`` without one."""
    try:
        q = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(NVSMI_FIELDS)}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": repr(e)}
    rows = [ln for ln in q.stdout.splitlines() if ln.strip()]
    if q.returncode != 0 or not rows:
        return {"error": (q.stderr or q.stdout).strip()[:500] or f"rc {q.returncode}"}
    gpu = dict(zip(NVSMI_FIELDS, [v.strip() for v in rows[0].split(",")]))
    gpu["n_gpus"] = len(rows)
    gpu["compute_apps"] = [ln.strip() for ln in apps.stdout.splitlines() if ln.strip()]
    gpu["t"] = time.time()
    return gpu


def _num(x: str) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_sampler(text: str) -> Dict[str, Any]:
    """`nvidia-smi -lms` CSV lines (``SAMPLER_FIELDS``, no header / units) -> per-field mean / max and the count (pure).
    Unparseable fields ("[N/A]") are skipped, never zero."""
    cols: Dict[str, List[float]] = {f: [] for f in SAMPLER_FIELDS}
    n = 0
    for ln in text.splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(SAMPLER_FIELDS):
            continue
        n += 1
        for f, v in zip(SAMPLER_FIELDS, parts):
            x = _num(v)
            if x is not None:
                cols[f].append(x)
    out: Dict[str, Any] = {"samples": n}
    for f, xs in cols.items():
        if xs:
            out[f] = {"mean": sum(xs) / len(xs), "max": max(xs)}
    return out


class GpuSampler:
    """`nvidia-smi -lms 1000` in the background for one phase; stopped by its own PID."""

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path, self.proc, self.enabled = path, None, enabled  # type: Path, Optional[subprocess.Popen], bool

    def __enter__(self) -> "GpuSampler":
        if self.enabled and shutil.which("nvidia-smi"):
            self.fh = open(self.path, "w")
            self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={','.join(SAMPLER_FIELDS)}",
                                          "--format=csv,noheader,nounits", "-lms", "1000"],
                                         stdout=self.fh, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.fh.close()

    def read(self) -> Dict[str, Any]:
        if self.proc is None or not self.path.exists():
            return {"samples": 0, "note": "disabled (cpu run)" if not self.enabled else "no nvidia-smi"}
        return parse_sampler(self.path.read_text())


def _cpu_model() -> str:
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.lower().startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _physical_cores() -> Optional[int]:
    try:
        seen = set()
        cur: Dict[str, str] = {}
        for ln in Path("/proc/cpuinfo").read_text().splitlines() + [""]:
            if not ln.strip():
                if "physical id" in cur and "core id" in cur:
                    seen.add((cur["physical id"], cur["core id"]))
                cur = {}
                continue
            if ":" in ln:
                k, v = ln.split(":", 1)
                cur[k.strip()] = v.strip()
        return len(seen) or None
    except OSError:
        return None


def _mem_total_gib() -> Optional[float]:
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            if ln.startswith("MemTotal:"):
                return int(ln.split()[1]) / 2**20
    except (OSError, ValueError):
        pass
    return None


def _git_commit() -> str:
    env = os.environ.get("GEN3AI_BENCH_COMMIT")
    if env:
        return env
    try:
        from utils.paths import repo_root
        r = subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "HEAD"], capture_output=True, text=True,
                           timeout=10)
        dirty = subprocess.run(["git", "-C", str(repo_root()), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return r.stdout.strip() + ("-dirty" if dirty else "") if r.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def run_host(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    res: Dict[str, Any] = {
        "cpu_model": _cpu_model(), "cpu_logical": os.cpu_count(), "cpu_physical": _physical_cores(),
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "ram_gib": _mem_total_gib(), "kernel": platform.release(), "python": platform.python_version(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(), "commit": _git_commit(), "hostname_hash": None,
        "load1": os.getloadavg()[0]}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cap = torch.cuda.get_device_capability(0)
        res.update(gpu_name=torch.cuda.get_device_name(0), capability=f"sm_{cap[0]}{cap[1]}",
                   vram_gib=props.total_memory / 2**30, sm_count=props.multi_processor_count,
                   arch_list=torch.cuda.get_arch_list(),
                   arch_ok=f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list())
    res["status"] = "ok"
    return res


# ------------------------------------------------------------------------------------------------- the phase worker

def phase_main(name: str, cfg_path: str) -> int:
    """`_phase <name> <cfg.json>`: run ONE phase in this process, its result to ``cfg["out"]``."""
    cfg = json.loads(Path(cfg_path).read_text())
    out = Path(cfg["out"])
    res: Dict[str, Any] = {"phase": name, "status": "running", "config": {k: v for k, v in cfg.items() if k != "out"}}

    def save() -> None:
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(res, indent=1, default=str))
        tmp.replace(out)
    save()
    if name == "host":
        res.update(run_host(cfg))
        save()
        return 0
    if name.startswith("update_") or name.startswith("headroom_"):
        from agents.training.gpu_bench_update import run_cell
        try:
            run_cell(cfg, res, save)
        except BaseException as e:  # noqa: BLE001 — recorded
            import traceback
            res["status"] = "error"
            res["error"] = f"{type(e).__name__}: {str(e)[:3000]}"
            res["traceback"] = traceback.format_exc()[-6000:]
            save()
            print(res["traceback"], flush=True)
            return 1
        return 0
    if name == "collect":
        return _collect_phase(cfg, res, save)
    raise SystemExit(f"unknown phase {name!r}")


def _collect_phase(cfg: Dict[str, Any], res: Dict[str, Any], save: Any) -> int:
    """Build the fixture pool, then run the collect WORKER (the trainer in process) as a child and merge its result."""
    from agents.training import gpu_bench_collect as GC
    from agents.training.gpu_bench_update import resolved_args

    scratch = Path(cfg["scratch"])
    if scratch.exists():
        shutil.rmtree(scratch)            # a fresh run dir each time: a fresh argv never lands on a run
    scratch.mkdir(parents=True)
    argv = shlex.split(cfg["argv"]) + list(cfg.get("extra", []))
    args = resolved_args(argv, "cpu")     # the namespace the pool's models are built on (an arch, not a device)
    res["pool"] = GC.build_pool(scratch, args, n_snapshots=int(cfg["pool_snapshots"]),
                                win_rate=float(cfg["pool_win_rate"]), seed=0)
    save()
    worker_out = Path(cfg["out"]).with_name("collect_worker.json")
    wcfg = {"out": str(worker_out), "warm": cfg["warm"], "timed": cfg["timed"], "pool": res["pool"],
            "trainer_argv": GC.trainer_argv(shlex.split(cfg["argv"]), run_dir=str(scratch), steps=int(cfg["steps"]),
                                            device=cfg["device"], seed=int(cfg["seed"]), extra=cfg.get("extra", []))}
    wpath = Path(cfg["out"]).with_name("collect_worker_cfg.json")
    wpath.write_text(json.dumps(wcfg, indent=1))
    env = dict(os.environ, GEN3AI_MODELS_DIR=str(scratch.parent))
    cmd = [sys.executable, "-u", "-m", "agents.training.gpu_bench_suite", "_collect_worker", str(wpath)]
    print(f"[gpu_bench collect] worker: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    w = json.loads(worker_out.read_text()) if worker_out.exists() else {"status": "missing"}
    res.update({k: v for k, v in w.items() if k != "pool"})
    res["worker_rc"] = rc
    if rc != 0 and res.get("status") == "ok":
        res["status"] = f"exit {rc}"
    try:
        shutil.rmtree(scratch)             # ~1 GB of fixture snapshots; the result holds what was measured
        res["scratch_removed"] = True
    except OSError as e:
        res["scratch_removed"] = repr(e)
    save()
    return 0 if res.get("status") == "ok" else 1


# ------------------------------------------------------------------------------------------------- the driver

def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def _descendants(pid: int) -> List[int]:
    from agents.training.learner_benchmark import descendants
    return descendants(pid)


def _run_phase(name: str, cfg: Dict[str, Any], out_dir: Path, timeout_min: float) -> Dict[str, Any]:
    from agents.training.learner_benchmark import terminate_pids

    cfg_path = out_dir / "phases" / f"{name}.cfg.json"
    _write_json(cfg_path, cfg)
    log = out_dir / "logs" / f"{name}.log"
    gpu = cfg["device"] == "cuda"         # a CPU run never queries the card (nvidia-smi ignores CUDA_VISIBLE_DEVICES)
    no_gpu = {"skipped": "cpu run"}
    rec: Dict[str, Any] = {"nvidia_smi_before": nvidia_smi_snapshot() if gpu else no_gpu}
    t0 = time.time()
    cmd = [sys.executable, "-u", "-m", "agents.training.gpu_bench_suite", "_phase", name, str(cfg_path)]
    with GpuSampler(out_dir / "logs" / f"{name}.nvsmi.csv", enabled=gpu) as smp, open(log, "wb") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_min * 60.0)
        except subprocess.TimeoutExpired:
            pids = _descendants(proc.pid) + [proc.pid]
            terminate_pids(pids)
            rc = None
            rec["timeout_killed_pids"] = pids
    rec["gpu_sampler"] = smp.read()
    rec["nvidia_smi_after"] = nvidia_smi_snapshot() if gpu else no_gpu
    rec["wall_s"] = time.time() - t0
    rec["rc"] = rc
    out = Path(cfg["out"])
    body = json.loads(out.read_text()) if out.exists() else {"status": "missing"}
    body.update(rec)
    if rc is None:
        body["status"] = "timeout"
    elif rc != 0 and body.get("status") in ("ok", "running"):
        body["status"] = f"exit {rc}"
    _write_json(out, body)
    return body


def run_main(a: argparse.Namespace) -> int:
    out_dir = Path(a.out).expanduser().resolve()
    from utils.paths import main_models_dir
    mm = main_models_dir()
    if mm is not None and (out_dir == Path(mm).resolve() or Path(mm).resolve() in out_dir.parents):
        print(f"REFUSED: --out {out_dir} is under the run archive {mm} (models/ is read-only to tools)")
        return 2
    quick = bool(a.quick)
    device = a.device
    if device == "auto":
        device = "cpu" if quick else "cuda"
    if quick and device != "cpu":
        print("REFUSED: --quick is the CPU code path; it runs on --device cpu only")
        return 2
    if not quick and device != "cuda":
        print("REFUSED: the full suite measures a GPU; use --quick for the CPU code path")
        return 2
    (out_dir / "phases").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    names = phase_names(quick)
    if a.phases:
        want = [p.strip() for p in a.phases.split(",") if p.strip()]
        names = [n for n in names if any(n == w or n.startswith(w + "_") or (w == "headroom" and
                                                                            n.startswith("headroom_")) for w in want)]
    suite_path = out_dir / "gpu_bench_suite.json"
    suite: Dict[str, Any] = json.loads(suite_path.read_text()) if suite_path.exists() else {}
    suite.update({"schema": SCHEMA, "suite_version": SUITE_VERSION, "quick": quick, "device": device,
                  "argv": PRODUCTION_ARGV, "live_cycle": LIVE_CYCLE, "geometry": QUICK if quick else FULL,
                  "tag": a.tag or suite.get("tag") or "", "meta": {**suite.get("meta", {}), **dict(_kv(a.meta))}})
    suite.setdefault("phases", {})
    suite.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    for name in names:
        prev = suite["phases"].get(name, {})
        if prev.get("status") in DONE_STATUSES and not a.force:
            print(f"[gpu_bench] {name}: {prev['status']} already — skipped (--force redoes)", flush=True)
            continue
        cfg = phase_config(name, quick, device, str(out_dir))
        print(f"[gpu_bench] {name}: START {time.strftime('%H:%M:%S')} (log {out_dir / 'logs' / (name + '.log')})",
              flush=True)
        body = _run_phase(name, cfg, out_dir, phase_timeout_min(name) * (a.timeout_scale or 1.0))
        suite["phases"][name] = body
        suite["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_json(suite_path, suite)
        print(f"[gpu_bench] {name}: {body.get('status')} in {body.get('wall_s', 0):.0f}s — {headline(name, body)}",
              flush=True)
    problems = validate(suite)
    suite["validation"] = problems
    _write_json(suite_path, suite)
    print(f"[gpu_bench] suite -> {suite_path}; validation: {'OK' if not problems else problems}", flush=True)
    return 0 if not problems else 1


def _kv(items: Optional[Sequence[str]]) -> List[Tuple[str, Any]]:
    out = []
    for it in items or []:
        k, _, v = it.partition("=")
        x: Any = v
        try:
            x = float(v)
        except ValueError:
            pass
        out.append((k.strip(), x))
    return out


def headline(name: str, body: Dict[str, Any]) -> str:
    if name == "host":
        return f"{body.get('gpu_name', 'no gpu')} / {body.get('cpu_model')} / torch {body.get('torch')}"
    if "train" in body:
        t = body["train"]
        return (f"update est {t.get('update_s_est', float('nan')):.2f}s, marginal epoch "
                f"{t.get('marginal_epoch_s', float('nan')):.3f}s, peak {body.get('peak_reserved_mib', 0):.0f} MiB, "
                f"gate {body.get('gate', {}).get('verdict')}")
    if body.get("oom"):
        return f"OOM at {body['oom'].get('where')}"
    if body.get("summary"):
        s = body["summary"]
        return (f"{s.get('env_steps_per_s_median', 0):.0f} env steps/s, {s.get('wall_s_median', 0):.1f}s per rollout, "
                f"gpu_wait {s.get('gpu_wait_ms_per_host_step_median', 0):.2f} ms/host step")
    return str(body.get("error", ""))[:200]


# ------------------------------------------------------------------------------------------------- schema

#: Required keys per phase family (dotted paths) when that phase's status is `ok`.
REQUIRED_TOP = ("schema", "suite_version", "quick", "device", "argv", "phases", "geometry", "live_cycle")
REQUIRED_OK = {
    "host": ("cpu_model", "cpu_logical", "ram_gib", "torch", "commit"),
    "update": ("shape.rows", "shape.batch_size", "shape.grad_accum", "gate.verdict", "train.update_s_est",
               "train.marginal_epoch_s", "train.median_s", "phases_s.compile_gate_prewarm", "calls",
               "nvidia_smi_before", "nvidia_smi_after", "gpu_sampler", "wall_s"),
    "collect": ("pool.n_snapshots", "buffer_shape.rows", "rollouts", "summary.env_steps_per_s_median",
                "summary.wall_s_median", "summary.gpu_wait_ms_per_host_step_median", "summary.core_ms_per_host_step_median",
                "nvidia_smi_before", "nvidia_smi_after", "wall_s"),
}
REQUIRED_GPU_HOST = ("gpu_name", "capability", "vram_gib", "arch_list")


def _get(d: Any, path: str) -> Any:
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def validate(suite: Dict[str, Any]) -> List[str]:
    """Every problem with a suite JSON (``[]`` = valid; pure). A phase not `ok` / `oom` is a problem: a missing,
    timed-out or failed phase never reads as a number."""
    probs: List[str] = []
    for k in REQUIRED_TOP:
        if k not in suite:
            probs.append(f"missing top-level {k!r}")
    if suite.get("schema") != SCHEMA:
        probs.append(f"schema {suite.get('schema')!r} != {SCHEMA!r}")
    phases = suite.get("phases") or {}
    for name in phase_names(bool(suite.get("quick"))):
        body = phases.get(name)
        if body is None:
            probs.append(f"phase {name}: missing")
            continue
        st = body.get("status")
        if st == "oom" and name.startswith("headroom_"):
            continue
        if st != "ok":
            probs.append(f"phase {name}: status {st!r}")
            continue
        fam = "update" if name.startswith(("update_", "headroom_")) else name
        req = list(REQUIRED_OK.get(fam, ()))
        if fam == "host" and suite.get("device") == "cuda":
            req += list(REQUIRED_GPU_HOST)
        for path in req:
            try:
                _get(body, path)
            except KeyError:
                probs.append(f"phase {name}: missing {path}")
        if fam == "update" and not suite.get("quick"):
            v = (body.get("gate") or {}).get("verdict")
            if v != "PASS":
                probs.append(f"phase {name}: compile parity gate {v!r} (a REFUSED gate is recorded, not passed)")
        if fam == "collect" and not suite.get("quick"):
            shape = body.get("buffer_shape") or {}
            geo = suite.get("geometry") or {}
            want = int(geo.get("n_steps", 0)) * int(geo.get("n_envs", 0))
            if int(shape.get("rows", -1)) != want:
                probs.append(f"phase collect: rollout rows {shape.get('rows')} != the production {want}")
    return probs


# ------------------------------------------------------------------------------------------------- comparison + Amdahl

def card_row(suite: Dict[str, Any]) -> Dict[str, Any]:
    """The comparison-table row of one suite JSON (pure). Missing / failed phases read None, never a number."""
    ph = suite.get("phases", {})
    host = ph.get("host", {})

    def upd(name: str) -> Optional[float]:
        b = ph.get(name, {})
        return b.get("train", {}).get("update_s_est") if b.get("status") == "ok" else None

    def peak(name: str) -> Optional[float]:
        b = ph.get(name, {})
        return b.get("peak_reserved_mib") if b.get("status") == "ok" else None
    col = ph.get("collect", {})
    cs = col.get("summary", {}) if col.get("status") == "ok" else {}
    vram_mib = (host.get("vram_gib") or 0) * 1024 or None
    pk = peak("update_bf16")
    headroom = {}
    for name, b in ph.items():
        if name.startswith("headroom_"):
            headroom[name] = ("OOM" if b.get("status") == "oom" else
                              {"update_s_est": b.get("train", {}).get("update_s_est"),
                               "peak_reserved_mib": b.get("peak_reserved_mib")} if b.get("status") == "ok"
                              else b.get("status"))
    meta = suite.get("meta", {})
    return {"tag": suite.get("tag") or host.get("gpu_name"), "gpu": host.get("gpu_name"), "torch": host.get("torch"),
            "cpu": host.get("cpu_model"), "update_bf16_s": upd("update_bf16"), "update_fp32_s": upd("update_fp32"),
            "peak_bf16_mib": pk, "peak_fp32_mib": peak("update_fp32"), "vram_mib": vram_mib,
            "headroom_mib": (vram_mib - pk) if (vram_mib and pk) else None, "headroom_points": headroom,
            "env_steps_per_s": cs.get("env_steps_per_s_median"), "collect_s": cs.get("wall_s_median"),
            "gpu_wait_ms_per_host_step": cs.get("gpu_wait_ms_per_host_step_median"),
            "core_ms_per_host_step": cs.get("core_ms_per_host_step_median"),
            "dph": meta.get("dph"), "suite_version": suite.get("suite_version"), "commit": host.get("commit")}


def amdahl(row: Dict[str, Any], ref: Dict[str, Any], live: Dict[str, Any] = LIVE_CYCLE) -> Optional[Dict[str, Any]]:
    """The full training cycle on ``row``'s card, from the LIVE local cycle and the card/reference RATIOS (pure).

        train_c   = live.train_s   x  U_c / U_ref                       (U = the bf16 production update estimate)
        collect_c = (live.collect_s - live.gpu_wait_s)  +  live.gpu_wait_s x W_c / W_ref
                                                                         (W = GPU wait per host step; the rest of the
                                                                          collect is the host CPU's and stays OURS)
        cycle_c   = train_c + collect_c + live.other_s
        speedup   = cycle_ref / cycle_c   with cycle_ref = live train + collect + other

    The reference stands in for the local 3080 Ti (the live numbers are the 3080 Ti's), so a card's ratios are taken
    against it. ``None`` when a needed number is missing."""
    u, ur = row.get("update_bf16_s"), ref.get("update_bf16_s")
    w, wr = row.get("gpu_wait_ms_per_host_step"), ref.get("gpu_wait_ms_per_host_step")
    if not all(isinstance(x, (int, float)) and x > 0 for x in (u, ur, w, wr)):
        return None
    train_c = live["train_s"] * u / ur
    cpu_part = live["collect_s"] - live["gpu_wait_s"]
    collect_c = cpu_part + live["gpu_wait_s"] * w / wr
    cycle_c = train_c + collect_c + live["other_s"]
    cycle_ref = live["train_s"] + live["collect_s"] + live["other_s"]
    return {"train_s": train_c, "collect_s": collect_c, "other_s": live["other_s"], "cycle_s": cycle_c,
            "cycle_ref_s": cycle_ref, "speedup": cycle_ref / cycle_c, "update_ratio": ur / u, "gpu_wait_ratio": wr / w,
            "steps_per_s": 98304.0 / cycle_c}


def compare_main(a: argparse.Namespace) -> int:
    suites = [json.loads(Path(p).read_text()) for p in a.files]
    rows = [card_row(s) for s in suites]
    ref = next((r for r in rows if r["tag"] == a.reference), rows[0] if rows else None)
    out = {"rows": rows, "reference": ref["tag"] if ref else None, "live_cycle": LIVE_CYCLE,
           "amdahl": {r["tag"]: amdahl(r, ref) for r in rows} if ref else {}}
    print(json.dumps(out, indent=1, default=str))
    return 0


def plan_main(a: argparse.Namespace) -> int:
    for name in phase_names(bool(a.quick)):
        cfg = phase_config(name, bool(a.quick), "cpu" if a.quick else "cuda", "<out>")
        print(f"{name:16s} timeout {phase_timeout_min(name):4.0f} min  {json.dumps(cfg, default=str)}")
    return 0


def validate_main(a: argparse.Namespace) -> int:
    probs = validate(json.loads(Path(a.file).read_text()))
    print("OK" if not probs else "\n".join(probs))
    return 0 if not probs else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents.training.gpu_bench_suite", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--quick", action="store_true", help="the CPU code path at tiny shapes (numbers mean nothing)")
    r.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    r.add_argument("--phases", default="", help="comma list of phases (host,update_bf16,update_fp32,headroom,collect)")
    r.add_argument("--force", action="store_true", help="redo phases already ok / oom")
    r.add_argument("--tag", default="", help="the row's name in a comparison (e.g. 'RTX 3090 / KR')")
    r.add_argument("--meta", action="append", default=[], metavar="K=V", help="recorded as-is (e.g. dph=0.264)")
    r.add_argument("--timeout-scale", type=float, default=1.0, help="multiply every phase's wall bound")
    r.set_defaults(fn=run_main)
    pl = sub.add_parser("plan")
    pl.add_argument("--quick", action="store_true")
    pl.set_defaults(fn=plan_main)
    v = sub.add_parser("validate")
    v.add_argument("file")
    v.set_defaults(fn=validate_main)
    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+")
    c.add_argument("--reference", default="")
    c.set_defaults(fn=compare_main)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_phase"]:
        return phase_main(argv[1], argv[2])
    if argv[:1] == ["_collect_worker"]:
        from agents.training.gpu_bench_collect import worker_main
        worker_main(argv[1])
        return 0
    a = build_parser().parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit(main())
