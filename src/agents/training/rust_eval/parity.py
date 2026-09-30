"""THE LANE H GATE — the same seed set played on BOTH eval paths (M5 Lane H).

One cycle PLAN (``ShardedEvalPool``: roster bots + pool sentinels, split into shard units) and ONE
cycle seed are played twice:

* the RUST path — ``RustEvalCore.run_cycle`` on the eval core with the trainee and the sentinels in T2
  slots (the production executor, in this process);
* the PYTHON path — today's eval worker (``python -m main.eval_worker``, a CUDA-less child) on the
  in-process rust bridge, with the per-GAME seed rule (``seed_rule = "per_game"``): each game's teams,
  battle seed and bot streams from ``rust_eval.seeds`` — the table the Rust path reads.

Neither path reads the other's actions. Compared:

1. PER GAME: winner, end turn and EVERY trainee action. A first differing action is judged by the
   DECLARED tie rule (Lane E's ``judge_flips``): a TIE iff both sides' top-2 legal log-prob margins at
   that decision are below ``NEAR_TIE_FACTOR x`` the tier's |Δ log-prob| bar — counted, reported, and
   the game (which then legitimately diverges) left out of the equality; any other difference FATAL.
2. THE METRICS: both cycles' shard results pooled by the unchanged collect (``ShardedEvalPool.collect``)
   — win rates, reward means, episode lengths, exact W/L counts, draws and the trace-selection tuples
   EQUAL; the critic-residual tails within ``TD_BAR`` (the two forwards differ by float rounding).
3. THE TRACES: the same trace files kept on both sides (same games, same names); each Rust core trace
   EXPANDS in the prober (``main.prober.core_trace``) to the SAME decisions as the Python
   trace of that game (turn, phase, chosen action, both actives, per decision).

Bars (the tier's |Δ legal log-prob|, Lane E's measured ones): eager CPU vs eager CPU 1e-5; T2 graph
on CUDA vs the Python path on CPU 1e-3.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from agents.training.rust_env_opponents_parity import NEAR_TIE_FACTOR, declared_torch_state

#: The critic-residual tail tolerance (V through T2 vs the Python forward).
TD_BAR = 1e-4
BAR_CPU = 1e-5
BAR_GPU = 1e-3
#: Torch threads of the Rust path's forward (declared, F-LJ-6).
RUST_THREADS = 4


def build_models(dst: Path, *, n_sentinels: int, trainee: Optional[str] = None,
                 sentinels: Optional[List[str]] = None) -> Tuple[str, List[str], str]:
    """``(trainee zip, [sentinel zips], model_config.json)``. Default: seeded PERTURBED fresh
    production-arch policies (``build_fresh_model``, built at 1 thread — F-LJ-6); else the given files."""
    dst.mkdir(parents=True, exist_ok=True)
    cfg = dst / "model_config.json"
    if trainee is not None:
        if not cfg.exists():
            src = Path(trainee).parent
            for cand in (src / "model_config.json", src.parent / "model_config.json"):
                if cand.exists():
                    cfg.write_text(cand.read_text())
                    break
        return trainee, list(sentinels or []), str(cfg)
    from agents.model.parity_probe import PERTURB_SCALE, perturb_
    from agents.model.snapshot import current_model_version
    from agents.observation.state_encoder import load_mappings
    from main.fresh_checkpoint import build_fresh_model

    with declared_torch_state(1):
        out = []
        for i in range(n_sentinels + 1):
            model, _, _ = build_fresh_model(i)
            perturb_(model.policy, seed=2000 + i, scale=PERTURB_SCALE)
            p = dst / f"snapshot_{(i + 1) * 1000:012d}.zip"
            model.save(str(p))
            out.append(str(p))
    from agents.model.snapshot import arch_toggles_from_model

    cfg.write_text(current_model_version(load_mappings(), **arch_toggles_from_model(model)).to_json())
    return out[-1], out[:-1], str(cfg)


def _items(bots: List[str], sentinels: List[str], games: int) -> List[Any]:
    from agents.training.eval_sharding import BOT, SENTINEL, EvalItem

    items = [EvalItem(b, BOT, games) for b in bots]
    items += [EvalItem(f"sentinel_{i}", SENTINEL, games, path=p, step=(i + 1) * 1000) for i, p in enumerate(sentinels)]
    return items


def run_rust(*, run_dir: Path, model_dir: Path, trainee: str, sentinels: List[str], items: List[Any],
             shard_games: int, step: int, cycle_seed: int, quota: Dict[str, int], device: str, backend: str,
             n_envs: int, buckets: Tuple[int, ...] = (8, 48), front: str = "ffi", profile: str = "selfcheck",
             lanes: int = 0) -> Dict[str, Any]:
    """The Rust path (module docs) — the production executor over a T2 service declared here."""
    import torch

    # `MaskablePPO.load` re-seeds the loaded model (sb3's `set_random_seed`), which on CUDA sets
    # `torch.backends.cudnn.deterministic` — a process global the gate must hand back (the root
    # conftest's torch global-state guard, `1afd2590`).
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        return _run_rust(run_dir=run_dir, model_dir=model_dir, trainee=trainee, sentinels=sentinels, items=items,
                         shard_games=shard_games, step=step, cycle_seed=cycle_seed, quota=quota, device=device,
                         backend=backend, n_envs=n_envs, buckets=buckets, front=front, profile=profile, lanes=lanes)
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn


def _run_rust(*, run_dir: Path, model_dir: Path, trainee: str, sentinels: List[str], items: List[Any],
              shard_games: int, step: int, cycle_seed: int, quota: Dict[str, int], device: str, backend: str,
              n_envs: int, buckets: Tuple[int, ...], front: str, profile: str, lanes: int) -> Dict[str, Any]:
    from sb3_contrib import MaskablePPO

    from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
    from agents.training.eval_sharding import ShardedEvalPool
    from agents.training.rust_eval.build import EvalDecl, build_eval_core, eval_builders
    from agents.training.rust_eval.launch import load_sentinels
    from agents.training.rust_rollout.build import RustEnvDecl
    from utils.rust_env import episode as EP

    with declared_torch_state(RUST_THREADS):
        t0 = time.perf_counter()
        model = MaskablePPO.load(trainee, env=None, device=device)
        model.policy.eval()
        n_slots = 1 + len(sentinels)
        ln = lanes or (min(n_slots, 8) if device.startswith("cuda") else 1)
        svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("eval", n_slots, model.policy),), device=device,
                                           backend=backend, buckets=tuple(buckets), lanes=ln,
                                           max_rows_per_flush=max(1024, n_slots * max(buckets), 4 * n_envs))).startup()
        t_svc = time.perf_counter() - t0
        decl = EvalDecl(n_envs=n_envs, n_sentinels=len(sentinels))
        cdecl = RustEnvDecl(n_envs=n_envs, threads=4, front=front, profile=profile)
        tb, flat, fixed = eval_builders(None, [])
        # The terminal from the run's model_config.json — exactly what the Python worker builds its
        # reward from (`RewardConfig.from_dict`).
        from agents.training.reward_config import RewardConfig

        terminal = EP.terminal_from_reward_config(
            RewardConfig.from_dict(json.loads((model_dir / "model_config.json").read_text())))
        ev = build_eval_core(decl, collector_decl=cdecl, svc=svc, extra_ids=list(range(n_slots)),
                             trainee_builder=tb, opp_builder=flat, fixed_builders=fixed,
                             turn_limit=EP.stall_threshold(), terminal=terminal, emit=lambda _m: None)
        pool = ShardedEvalPool(items, shard_games, step=step)
        run_dir.mkdir(parents=True, exist_ok=True)
        pool.write_plan(str(run_dir))

        from agents.model.snapshot import arch_toggles_from_model

        sent = load_sentinels(pool, model)
        glog: List[Dict[str, Any]] = []
        from agents.training.eval_callback import ForensicQuota

        st = ev.run_cycle(pool, str(run_dir), step=step, trainee_policy=model.policy, sentinel_policies=sent,
                          forensic_root=str(model_dir / "eval_traces" / f"step_{step}"),
                          quota=ForensicQuota(**quota), gamma=float(model.gamma), sentinel_greedy=True,
                          self_play_temp=1.0, cycle_seed=cycle_seed, game_log=glog)
        ev.close()
        toggles = arch_toggles_from_model(model)
        return {"stats": st.as_dict(), "games": glog, "svc_startup_s": t_svc, "arch_toggles": toggles,
                "gamma": float(model.gamma)}


def run_python(*, run_dir: Path, model_dir: Path, trainee: str, items: List[Any], shard_games: int, step: int,
               cycle_seed: int, quota: Dict[str, int], arch_toggles: Dict[str, Any], gamma: float,
               compile_extractor: bool = False, timeout: float = 7200.0) -> Dict[str, Any]:
    """The Python path (module docs): today's eval worker in a CUDA-less child, per-game seeded."""
    from agents.training.eval_sharding import ShardedEvalPool

    pool = ShardedEvalPool(items, shard_games, step=step)
    run_dir.mkdir(parents=True, exist_ok=True)
    pool.write_plan(str(run_dir))
    claim = run_dir / "claims"
    claim.mkdir(exist_ok=True)
    glog = run_dir / "games.jsonl"
    cfg = {"snapshot": trainee, "port": None, "use_showdown_bridge": True, "compile_extractor": bool(compile_extractor),
           "bridge_impl": "rust", "model_dir": str(model_dir), "step": step, "claim_dir": str(claim),
           "result_dir": str(run_dir), "concurrency": 1, "device": "cpu", "cycle_tag": "lh", "gamma": gamma,
           "forensic_quota": dict(quota), "arch_toggles": arch_toggles, "trainee_team_str": None,
           "eval_sentinel_greedy": True, "self_play_temp": 1.0, "seed_base": int(cycle_seed),
           "seed_rule": "per_game", "game_log_path": str(glog), "worker_id": 0}
    cfg_path = run_dir / "config_0.json"
    cfg_path.write_text(json.dumps(cfg))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "main.eval_worker", str(cfg_path)], env=env, capture_output=True,
                       text=True, timeout=timeout)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"the Python eval worker failed (rc {r.returncode}):\n{(r.stdout + r.stderr)[-4000:]}")
    games = [json.loads(line) for line in glog.read_text().splitlines() if line.strip()] if glog.exists() else []
    return {"games": games, "wall_s": wall}


def compare_games(rust: List[Dict[str, Any]], py: List[Dict[str, Any]], bar: float) -> Dict[str, Any]:
    """Per-game equality under the declared tie rule (module docs)."""
    rk = {(g["item"], int(g["game"])): g for g in rust}
    pk = {(g["item"], int(g["game"])): g for g in py}
    out: Dict[str, Any] = {"games": len(rk), "python_games": len(pk), "equal": 0, "ties": [], "fatal": [],
                           "decisions": 0, "missing": sorted(set(rk) ^ set(pk)), "max_dlogp": 0.0}
    eps = NEAR_TIE_FACTOR * float(bar)
    for k in sorted(set(rk) & set(pk)):
        a, b = rk[k], pk[k]
        ra, pa = list(a["actions"]), list(b["actions"])
        j = next((i for i, (x, y) in enumerate(zip(ra, pa)) if int(x) != int(y)), None)
        for x, y in list(zip(a.get("logp", []), b.get("logp", [])))[: j if j is not None else None]:
            out["max_dlogp"] = max(out["max_dlogp"], abs(float(x) - float(y)))
        if j is None and len(ra) == len(pa) and int(a["winner"]) == int(b["winner"]) \
                and int(a["end_turn"]) == int(b["end_turn"]):
            out["equal"] += 1
            out["decisions"] += len(ra)
            continue
        row = {"item": k[0], "game": k[1], "first_diff": j, "rust": [a["winner"], a["end_turn"], len(ra)],
               "python": [b["winner"], b["end_turn"], len(pa)]}
        if j is not None:
            mr, mp = float(a["margins"][j]), float(b["margins"][j])
            row.update(margin_rust=mr, margin_python=mp)
            if max(mr, mp) < eps:
                out["ties"].append(row)
                continue
        out["fatal"].append(row)
    return out


def compare_metrics(rust_dir: Path, py_dir: Path) -> Dict[str, Any]:
    """Both cycles pooled by the unchanged collect; every metric equal, the TD tails within ``TD_BAR``."""
    from agents.training.eval_sharding import ShardedEvalPool

    mr, miss_r = ShardedEvalPool.from_plan(str(rust_dir)).collect(str(rust_dir))
    mp, miss_p = ShardedEvalPool.from_plan(str(py_dir)).collect(str(py_dir))
    diffs = []
    for block in sorted(set(mr) | set(mp)):
        a, b = mr.get(block, {}), mp.get(block, {})
        for key in sorted(set(a) | set(b)):
            x, y = a.get(key), b.get(key)
            if block == "durations_sec":
                continue
            if block == "td_resid_tails":
                ok = (x is None and y is None) or (x is not None and y is not None and abs(float(x) - float(y)) <= TD_BAR)
            else:
                ok = json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)
            if not ok:
                diffs.append({"block": block, "key": key, "rust": x, "python": y})
    return {"missing": [miss_r, miss_p], "diffs": diffs, "rust": {k: v for k, v in mr.items() if k != "durations_sec"},
            "python": {k: v for k, v in mp.items() if k != "durations_sec"}}


def compare_traces(rust_model_dir: Path, py_model_dir: Path, step: int, *, limit: int = 0) -> Dict[str, Any]:
    """The same trace files on both sides; every (or the first ``limit``) Rust core trace EXPANDS in the
    prober to the Python trace's decisions (turn, phase, chosen, both actives, per decision)."""
    def names(root: Path) -> Dict[str, Path]:
        d = root / "eval_traces" / f"step_{step}"
        return {str(p.relative_to(d)): p for p in d.rglob("*_summary.json")} if d.exists() else {}

    rn, pn = names(rust_model_dir), names(py_model_dir)
    out: Dict[str, Any] = {"rust": len(rn), "python": len(pn), "only_rust": sorted(set(rn) - set(pn)),
                           "only_python": sorted(set(pn) - set(rn)), "checked": 0, "decision_diffs": []}
    try:
        from main.prober import core_trace
    except ImportError as e:                 # the expander is part of this lane; its absence is a failure
        out["error"] = f"main.prober.core_trace unavailable: {e}"
        return out
    keys = ("turn", "phase", "chosen", "our", "opp")
    for rel in sorted(set(rn) & set(pn))[: limit or None]:
        rs = core_trace.load_summary(str(rn[rel]))
        ps = json.loads(pn[rel].read_text())
        ri, pi = rs.get("invocations", []), ps.get("invocations", [])
        out["checked"] += 1
        if len(ri) != len(pi):
            out["decision_diffs"].append({"trace": rel, "n": [len(ri), len(pi)]})
            continue
        for i, (x, y) in enumerate(zip(ri, pi)):
            bad = [k for k in keys if x.get(k) != y.get(k)]
            if bad:
                out["decision_diffs"].append({"trace": rel, "decision": i, "keys": bad,
                                              "rust": {k: x.get(k) for k in bad}, "python": {k: y.get(k) for k in bad}})
                break
    return out


def run(cfg: Dict[str, Any], workdir: Optional[str] = None) -> Dict[str, Any]:
    """The whole gate (module docs). ``cfg``: bots, sentinels (count), games, shard_games, n_envs,
    device, backend, seed, quota, trainee/sentinel_paths (optional real files), bar, trace_limit."""
    from agents.training.eval_callback import eval_opponent_names

    wd = Path(workdir or tempfile.mkdtemp(prefix="laneH_gate_"))
    step = int(cfg.get("step", 1000))
    bots = list(cfg.get("bots") or eval_opponent_names())
    trainee, sentinels, mcfg = build_models(wd / "models", n_sentinels=int(cfg.get("sentinels", 2)),
                                            trainee=cfg.get("trainee"), sentinels=cfg.get("sentinel_paths"))
    items = _items(bots, sentinels, int(cfg.get("games", 4)))
    quota = dict(cfg.get("quota") or {"win": 5, "loss": 10, "draw": 5})
    seed = int(cfg.get("seed", 20260930))
    dirs = {}
    for side in ("rust", "python"):
        md = wd / f"run_{side}"
        md.mkdir(parents=True, exist_ok=True)
        (md / "model_config.json").write_text(Path(mcfg).read_text())
        dirs[side] = md
    t0 = time.perf_counter()
    rust = run_rust(run_dir=dirs["rust"] / ".eval_runs" / f"step_{step}", model_dir=dirs["rust"], trainee=trainee,
                    sentinels=sentinels, items=items, shard_games=int(cfg.get("shard_games", 25)), step=step,
                    cycle_seed=seed, quota=quota, device=str(cfg.get("device", "cpu")),
                    backend=str(cfg.get("backend", "eager")), n_envs=int(cfg.get("n_envs", 16)),
                    front=str(cfg.get("front", "ffi")), profile=str(cfg.get("profile", "selfcheck")))
    t_rust = time.perf_counter() - t0
    py = run_python(run_dir=dirs["python"] / ".eval_runs" / f"step_{step}", model_dir=dirs["python"],
                    trainee=trainee, items=items, shard_games=int(cfg.get("shard_games", 25)), step=step,
                    cycle_seed=int(cfg.get("python_seed", seed)),   # a different one only in the teeth test
                    quota=quota, arch_toggles=rust["arch_toggles"], gamma=rust["gamma"],
                    compile_extractor=bool(cfg.get("compile_python", False)))
    bar = float(cfg.get("bar", BAR_CPU))
    games = compare_games(rust["games"], py["games"], bar)
    metrics = compare_metrics(dirs["rust"] / ".eval_runs" / f"step_{step}", dirs["python"] / ".eval_runs" / f"step_{step}")
    traces = compare_traces(dirs["rust"], dirs["python"], step, limit=int(cfg.get("trace_limit", 0)))
    ok = (not games["fatal"] and not games["missing"] and games["max_dlogp"] <= bar
          and (bool(games["ties"]) or not metrics["diffs"])
          and not traces["only_rust"] and not traces["only_python"] and not traces["decision_diffs"]
          and "error" not in traces and not rust["stats"]["lifecycle"].get("bad"))
    return {"pass": bool(ok), "workdir": str(wd), "cfg": {**cfg, "bots": bots, "bar": bar, "seed": seed},
            "games": {k: v for k, v in games.items()}, "metrics_diffs": metrics["diffs"],
            "metrics_equal": not metrics["diffs"], "metrics": metrics["rust"], "traces": traces,
            "rust_stats": rust["stats"], "rust_wall_s": t_rust, "python_wall_s": py["wall_s"],
            "svc_startup_s": rust["svc_startup_s"]}


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--shard-games", type=int, default=25)
    ap.add_argument("--sentinels", type=int, default=2)
    ap.add_argument("--bots", default="")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", default="eager")
    ap.add_argument("--front", default="ffi")
    ap.add_argument("--profile", default="selfcheck")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--trainee", default=None)
    ap.add_argument("--sentinel-paths", default="")
    ap.add_argument("--bar", type=float, default=None)
    ap.add_argument("--compile-python", action="store_true")
    ap.add_argument("--trace-limit", type=int, default=0)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = {"games": a.games, "shard_games": a.shard_games, "sentinels": a.sentinels, "n_envs": a.n_envs,
           "device": a.device, "backend": a.backend, "front": a.front, "profile": a.profile, "seed": a.seed,
           "bots": [b for b in a.bots.split(",") if b], "trainee": a.trainee,
           "sentinel_paths": [p for p in a.sentinel_paths.split(",") if p] or None,
           "bar": a.bar if a.bar is not None else (BAR_GPU if a.device.startswith("cuda") else BAR_CPU),
           "compile_python": a.compile_python, "trace_limit": a.trace_limit}
    rep = run(cfg, a.workdir)
    text = json.dumps(rep, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text)
    g = rep["games"]
    print(f"LANE H GATE {'PASS' if rep['pass'] else 'FAIL'}: {g['equal']}/{g['games']} games equal "
          f"({g['decisions']} trainee decisions, max |dlogp(chosen)| {g['max_dlogp']:.2e}), {len(g['ties'])} ties, {len(g['fatal'])} fatal; metrics "
          f"{'EQUAL' if rep['metrics_equal'] else 'DIFFER'}; traces {rep['traces'].get('checked')} checked, "
          f"{len(rep['traces'].get('decision_diffs', []))} diffs; rust {rep['rust_wall_s']:.1f}s, python "
          f"{rep['python_wall_s']:.1f}s")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
