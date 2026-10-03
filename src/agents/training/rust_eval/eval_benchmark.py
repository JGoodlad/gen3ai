"""EVAL WALL-CLOCK PER CYCLE on both eval paths at the production eval shape (M5 Lane H) — a DESCRIPTOR.

One arm per invocation (a driver interleaves them: ``rust python python rust``; the Rust arm runs under
the GPU lock, the Python arm does not touch the GPU):

* ``--arm python`` — TODAY's live eval cycle: the self-play callback's plan (the nine roster bots + the
  sentinels, ``--eval-games`` each, ``--eval-shard-games`` units) played by ``--workers`` work-stealing
  ``main.eval_worker`` processes (the production count WAS 5 x 2 = 10 under
  ``--self-play``), CPU, ``--compile-opponents``' compiled extractor, the rust bridge, no seed — exactly
  what ``SelfPlayCallback._launch_eval`` spawns. Wall = spawn → every worker exited.
* ``--arm rust`` — the same plan on the Rust eval core: T2 (``graph`` on CUDA, the trainee's eval slot
  + the sentinel slots, buckets (8, 48)), the eval core (``--n-envs``, process front end, release),
  ``--cycles`` cycles in one process; wall per cycle = ``run_cycle`` (loads + play + traces). T2's
  startup is reported apart (paid once per run, at startup).

Every row carries its regime: load averages, the measured contention factor, torch, the device, the
commit. Benchmarks WARN, never stretch (``utils.contention.warn_if_contended``).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _regime(device: str) -> Dict[str, Any]:
    from utils.contention import cpu_contention_factor, warn_if_contended

    reg: Dict[str, Any] = {"load_avg": list(os.getloadavg()), "contention": cpu_contention_factor(refresh=True),
                           "cpus": os.cpu_count(), "device": device,
                           "contended_warning": bool(warn_if_contended("rust eval benchmark"))}
    try:
        import torch

        reg["torch"] = torch.__version__
        if device.startswith("cuda") and torch.cuda.is_available():
            reg["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        from utils.git import get_git_hash

        reg["commit"] = get_git_hash()
    except Exception:
        pass
    return reg


def _items(games: int, sentinels: List[str]) -> List[Any]:
    from agents.training.eval_callback import eval_opponent_names
    from agents.training.eval_sharding import BOT, SENTINEL, EvalItem

    items = [EvalItem(b, BOT, games) for b in eval_opponent_names()]
    items += [EvalItem(f"sentinel_{i}", SENTINEL, games, path=p, step=i) for i, p in enumerate(sentinels)]
    return items


def _model_dir(root: Path, trainee: str) -> Path:
    md = root / "run"
    md.mkdir(parents=True, exist_ok=True)
    src = Path(trainee).parent
    for cand in (src / "model_config.json", src.parent / "model_config.json"):
        if cand.exists():
            shutil.copy(cand, md / "model_config.json")
            break
    return md


def arm_python(a: argparse.Namespace, root: Path) -> Dict[str, Any]:
    from sb3_contrib import MaskablePPO

    from agents.model.snapshot import arch_toggles_from_model
    from agents.training.eval_callback import ForensicQuota, kill_eval_workers, spawn_eval_workers
    from agents.training.eval_sharding import ShardedEvalPool

    md = _model_dir(root, a.trainee)
    toggles = arch_toggles_from_model(MaskablePPO.load(a.trainee, env=None, device="cpu"))
    run_dir = md / ".eval_runs" / "step_0"
    shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "claims").mkdir(parents=True)
    pool = ShardedEvalPool(_items(a.games, a.sentinels), a.shard_games, step=0)
    pool.write_plan(str(run_dir))
    cfg = {"snapshot": a.trainee, "port": None, "use_showdown_bridge": True, "compile_extractor": True,
           "bridge_impl": "rust", "model_dir": str(md), "step": 0, "self_play_temp": 1.0,
           "eval_sentinel_greedy": True, "claim_dir": str(run_dir / "claims"), "result_dir": str(run_dir),
           "concurrency": 1, "device": "cpu", "cycle_tag": "lhb", "gamma": 1.0,
           "forensic_quota": ForensicQuota()._asdict(), "arch_toggles": toggles, "trainee_team_str": None}
    reg = _regime("cpu")
    t0 = time.perf_counter()
    procs = spawn_eval_workers(str(run_dir), cfg, a.workers)
    try:
        for w in procs:
            w["proc"].wait()
    finally:
        kill_eval_workers(procs)
        for w in procs:
            w["log"].close()
    wall = time.perf_counter() - t0
    merged, missing = ShardedEvalPool.from_plan(str(run_dir)).collect(str(run_dir))
    games = sum(n for (_w, n) in merged.get("counts", {}).values())
    return {"arm": "python", "wall_s": wall, "games": games, "missing": missing, "workers": a.workers,
            "rc": [w["proc"].returncode for w in procs], "regime": reg, "regime_end": _regime("cpu")}


def arm_rust(a: argparse.Namespace, root: Path) -> Dict[str, Any]:
    from sb3_contrib import MaskablePPO

    from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
    from agents.training.eval_callback import ForensicQuota
    from agents.training.eval_sharding import ShardedEvalPool
    from agents.training.reward_config import RewardConfig
    from agents.training.rust_eval.build import EvalDecl, build_eval_core, eval_builders
    from agents.training.rust_eval.launch import load_sentinels
    from agents.training.rust_rollout.build import RustEnvDecl
    from utils.rust_env import episode as EP

    md = _model_dir(root, a.trainee)
    model = MaskablePPO.load(a.trainee, env=None, device=a.device)
    model.policy.eval()
    n_slots = 1 + len(a.sentinels)
    t0 = time.perf_counter()
    svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("eval", n_slots, model.policy),), device=a.device,
                                       backend=a.backend, buckets=(8, 48), lanes=min(n_slots, 8) if a.device.startswith("cuda") else 1,
                                       max_rows_per_flush=max(1024, n_slots * 48, 4 * a.n_envs))).startup()
    t_svc = time.perf_counter() - t0
    terminal = EP.terminal_from_reward_config(RewardConfig.from_dict(json.loads((md / "model_config.json").read_text())))
    tb, flat, fixed = eval_builders(None, [])
    t1 = time.perf_counter()
    ev = build_eval_core(EvalDecl(n_envs=a.n_envs, n_sentinels=len(a.sentinels)),
                         collector_decl=RustEnvDecl(n_envs=a.n_envs, threads=a.threads, front="proc", profile="release"),
                         svc=svc, extra_ids=list(range(n_slots)), trainee_builder=tb, opp_builder=flat,
                         fixed_builders=fixed, turn_limit=EP.stall_threshold(), terminal=terminal, emit=lambda _m: None)
    t_core = time.perf_counter() - t1
    cycles = []
    for k in range(a.cycles):
        run_dir = md / ".eval_runs" / f"step_{k}"
        shutil.rmtree(run_dir, ignore_errors=True)
        pool = ShardedEvalPool(_items(a.games, a.sentinels), a.shard_games, step=k)
        run_dir.mkdir(parents=True)
        pool.write_plan(str(run_dir))
        reg = _regime(a.device)
        t2 = time.perf_counter()
        st = ev.run_cycle(pool, str(run_dir), step=k, trainee_policy=model.policy,
                          sentinel_policies=load_sentinels(pool, model),
                          forensic_root=str(md / "eval_traces" / f"step_{k}"), quota=ForensicQuota(),
                          gamma=float(model.gamma), sentinel_greedy=True, self_play_temp=1.0, cycle_seed=7000 + k)
        wall = time.perf_counter() - t2
        _m, missing = ShardedEvalPool.from_plan(str(run_dir)).collect(str(run_dir))
        cycles.append({"wall_s": wall, "stats": st.as_dict(), "missing": missing, "regime": reg})
    ev.close()
    return {"arm": "rust", "t2_startup_s": t_svc, "eval_core_startup_s": t_core, "cycles": cycles,
            "n_envs": a.n_envs, "threads": a.threads, "backend": a.backend, "svc_counters": dict(svc.counters)}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=("rust", "python"), required=True)
    ap.add_argument("--trainee", required=True)
    ap.add_argument("--sentinels", default="", help="comma-separated snapshot zips")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--shard-games", type=int, default=25)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="graph")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.sentinels = [p for p in a.sentinels.split(",") if p]
    root = Path(a.workdir) / a.arm
    root.mkdir(parents=True, exist_ok=True)
    rep = arm_python(a, root) if a.arm == "python" else arm_rust(a, root)
    rep["shape"] = {"games": a.games, "shard_games": a.shard_games, "bots": 9, "sentinels": len(a.sentinels),
                    "trainee": a.trainee}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    if a.arm == "python":
        print(f"python arm: {rep['wall_s']:.1f}s for {rep['games']} games ({a.workers} workers)")
    else:
        print("rust arm: " + ", ".join(f"{c['wall_s']:.1f}s" for c in rep["cycles"]) + f" per cycle; T2 startup {rep['t2_startup_s']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
