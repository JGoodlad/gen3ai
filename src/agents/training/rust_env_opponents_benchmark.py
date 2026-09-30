"""M5 Lane E — opponent routing THROUGHPUT at the production shape (a measurement, run as a script).

N envs (48) in the Rust env core (FFI, release build), the PRODUCTION opponent mix drawn per episode
by ``RustEnvOpponents`` (self-play fraction, the training bot roster played IN THE CORE by Lane F's
bots, a pool of K real snapshots in T2 slots), POLICY rows served by a started T2 service (graph
backend, lanes = min(slots, 8)), p1 a seeded random policy (the trainee's forward is Lane G's and is
NOT in these numbers). Per env step it reports the core STEP, the opponent SERVE (gather + T2 flush
+ sampling), the staging, the rows served per flush and their spread over slots — the input to the
bucket choice. A benchmark's output IS the measurement: it WARNS on a busy box and never stretches.

    export PYTHONPATH=$PYTHONPATH:src
    scripts/ops/gpu_lock.sh python -m agents.training.rust_env_opponents_benchmark \\
        --snapshots /home/goodlad/dev/gen3ai/models/ai_v14_06_lbat_ctrl_fix/snapshots --n-envs 48
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _busy(device: str) -> List[str]:
    out = []
    load1 = os.getloadavg()[0]
    if load1 > (os.cpu_count() or 1) * 0.5:
        out.append(f"load average {load1:.1f} on {os.cpu_count()} cores")
    if device.startswith("cuda"):
        try:
            q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10)
            others = [p for p in q.stdout.split() if p.strip() and int(p) != os.getpid()]
            if others:
                out.append(f"other GPU processes: {others}")
        except Exception as exc:
            out.append(f"GPU occupancy unknown ({type(exc).__name__})")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshots", required=True, help="a pool directory (read-only: symlinked into a temp dir)")
    ap.add_argument("--pool-size", type=int, default=20)
    ap.add_argument("--n-envs", type=int, default=48)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--self-play-fraction", type=float, default=0.9)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="graph")
    ap.add_argument("--buckets", default="2,4,8,16")
    ap.add_argument("--lanes", type=int, default=0)
    ap.add_argument("--profile", default="release")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--mode", default="sampled", choices=("sampled", "greedy"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("rust_env_opponents_benchmark")   # K3: fresh, private, deleted at exit
    import torch

    from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
    from agents.model.model_version import ModelVersion
    from agents.training import rust_env_opponents as E
    from agents.training.snapshot_pool import SnapshotPool
    from main.rust_core_cutover.envs import packed_teams
    from utils.rust_env import bot_inventory as BI
    from utils.rust_env import episode as EP
    from utils.rust_env import ffi
    from utils.rust_env import protocol as P

    warnings = _busy(a.device)
    torch.set_num_threads(4)
    src = Path(a.snapshots)
    wd = Path(tempfile.mkdtemp(prefix="laneE_bench_pool_"))
    zips = sorted(src.glob("snapshot_*.zip"))[-a.pool_size:]
    for z in zips:
        (wd / z.name).symlink_to(z)
    (wd / "model_config.json").symlink_to(src / "model_config.json")
    ver = ModelVersion.from_json_file(str(wd / "model_config.json"))   # the TRAINEE's version (the pool's own)
    pool = SnapshotPool(wd, current_version=ver, device="cpu", lru_cache_size=a.pool_size + 4)
    bots = tuple(r.name for r in BI.ROWS if "train" in r.used_by)
    plan = E.OpponentPlan(pool_slots=len(zips) + E.DEFAULT_POOL_SPARE, bots=bots, bot_seed=20260929)
    buckets = tuple(int(x) for x in a.buckets.split(","))
    lanes = a.lanes or min(plan.n_policy_slots, 8)
    template = pool.load_model(pool._entries[-1]).policy.eval()
    t0 = time.perf_counter()
    svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("pool", plan.n_policy_slots, template),),
                                       device=a.device, backend=a.backend, buckets=buckets, lanes=lanes,
                                       max_rows_per_flush=1024)).startup()
    startup_s = time.perf_counter() - t0
    routes = plan.routes()
    load_s: List[float] = []

    def load(route: int, mid: str) -> None:
        step = int(mid.split(":")[1])
        entry = next(e for e in pool._entries if e.step == step)
        t = time.perf_counter()
        svc.load(int(routes[route].slot), pool.load_model(entry).policy.eval(), mid)
        load_s.append(time.perf_counter() - t)

    n = a.n_envs
    t1 = time.perf_counter()
    host = E.RustEnvOpponents(plan, n, load=load, pool=pool, self_play_fraction=a.self_play_fraction,
                              pool_rng_seeds=[5000 + i for i in range(n)], rng_seeds=list(range(n)))
    host.set_self_play_target(a.self_play_fraction, 0)
    pool_load_s = time.perf_counter() - t1
    server = E.PolicyOpponentServer(plan, svc, n, policy_seed=17, seed_stride=1, force_greedy=a.mode == "greedy")
    teams = packed_teams("pool")
    spec = P.spec_json(n=n, threads=a.threads, teams=list(teams), names=("benone", "bentwo"), decision_tense=False,
                       switch_freeze=False, turn_limit=EP.stall_threshold(), refusal_budget=64, bank_dir=None,
                       opponents=plan.spec_rows(bots="core"))
    lib = ffi.load(ffi.default_path(a.profile), nan_poison=a.profile == "selfcheck")
    rng = np.random.default_rng(3)
    t_core, t_serve, t_stage, t_p1 = [], [], [], []
    rows_per_flush: List[int] = []
    slots_per_flush: List[int] = []
    rows_per_slot: List[int] = []
    p1_rows = 0
    svc0 = dict(svc.counters)
    with ffi.FfiCore(spec, lib=lib) as core:
        c = core.cols

        def stage_teams(ix: Any) -> None:
            for i in ix:
                x = int(rng.integers(len(teams)))
                y = int(rng.integers(len(teams) - 1))
                c["ep_team"][i] = [x, y + (y >= x)]
                c["ep_seed"][i] = rng.integers(0, 65536, 4)

        host.stage_all(c)
        stage_teams(range(n))
        core.reset()
        stage_teams(host.after_op(c))
        for k in range(a.warmup + a.steps):
            timed = k >= a.warmup
            ts = time.perf_counter()
            m = c["need"][:, 1] == 1
            sl = c["opp_slot"][m]
            served = server.serve(c)
            t_a = time.perf_counter()
            need1 = np.flatnonzero(c["need"][:, 0] == 1)
            for i in need1:
                c["action"][i, 0] = int(rng.choice(np.flatnonzero(c["mask"][i, 0])))
            t_b = time.perf_counter()
            core.step()
            t_c = time.perf_counter()
            stage_teams(host.after_op(c))
            t_d = time.perf_counter()
            if timed:
                t_serve.append(t_a - ts)
                t_p1.append(t_b - t_a)
                t_core.append(t_c - t_b)
                t_stage.append(t_d - t_c)
                p1_rows += len(need1)
                if served.size:
                    rows_per_flush.append(int(served.size))
                    u, cnt = np.unique(sl[sl >= 0], return_counts=True)
                    slots_per_flush.append(int(u.size))
                    rows_per_slot.extend(int(x) for x in cnt)
        counters = core.counters()
        after = core.after_freeze()

    def ms(x: List[float]) -> Dict[str, float]:
        v = np.asarray(x) * 1e3
        return {"mean": round(float(v.mean()), 3), "p50": round(float(np.median(v)), 3), "p95": round(float(np.percentile(v, 95)), 3)}

    rc = host.route_counts
    klass = np.array([r.klass for r in routes])
    res: Dict[str, Any] = {
        "torch": torch.__version__, "device": a.device, "backend": a.backend, "buckets": buckets, "lanes": lanes,
        "n_envs": n, "threads": a.threads, "profile": a.profile, "pool": [z.name for z in zips],
        "slots": plan.n_policy_slots, "bots": bots, "self_play_fraction": a.self_play_fraction, "mode": a.mode,
        "busy_box_warnings": warnings, "startup_s": round(startup_s, 1), "startup_phases_s":
        {k: round(v, 1) for k, v in svc.startup_seconds.items()}, "pool_admit_s": round(pool_load_s, 2),
        "load_s_per_snapshot": ms(load_s), "steps_timed": a.steps,
        "per_step_ms": {"serve": ms(t_serve), "core_step": ms(t_core), "p1_random": ms(t_p1), "stage": ms(t_stage)},
        "serve_split_s": {"gather": server.stats.gather_s, "flush_launch": server.stats.flush_s,
                          "gpu_wait": server.stats.wait_s, "draw": server.stats.draw_s},
        "rows_per_flush": ms([x / 1e3 for x in rows_per_flush]), "slots_per_flush_mean": float(np.mean(slots_per_flush)),
        "rows_per_slot_hist": {int(k): int(v) for k, v in zip(*np.unique(rows_per_slot, return_counts=True))},
        "p1_decisions_per_s": round(p1_rows / (sum(t_core) + sum(t_serve) + sum(t_stage) + sum(t_p1)), 1),
        "episodes_by_class": {c_: int(rc[klass == c_].sum()) for c_ in sorted(set(klass.tolist()))},
        "svc_after_freeze_delta": {k: svc.counters[k] - svc0[k] for k in ("compiles_after_freeze", "captures_after_freeze", "cuda_segments_after_freeze")},
        "core_after_freeze": after, "core_counters": counters,
    }
    print(json.dumps(res, indent=1, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
