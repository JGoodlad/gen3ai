"""T2 FAN-OUT read (M5 Lane G, the F-LE-8 follow-up): what ONE host step's T2 flush costs as the
policy-opponent rows spread over more DISTINCT slots, and so what each candidate lever could buy.

WHY. Lane E billed 5.1 ms / step to "sampling"; it was the host WAITING for the forward (F-LE-8,
corrected 2026-09-30: the draws themselves cost 0.13 / 0.25 / 0.38 ms at 8 / 40 / 48 rows). At a
production mix (95 % self-play, a pool of 20) a step's ~40 opponent rows land on ~18 slots, and every
distinct slot is its own graph replay. The two candidate levers:

* ONE GROUPED FORWARD across slots (stacked slot weights, one batched kernel set). Its UPPER BOUND is
  measured here without building it: the same rows on ONE slot (``opp_S1``) — a grouped forward still
  reads every slot's weights, so it cannot beat one slot's forward over the same rows.
* FEWER DISTINCT SNAPSHOTS ACTIVE per step (a smaller pool, or a rotating active subset of the same
  pool): ``opp_S{K}`` is its per-flush cost, the throughput A/B's ``rust_serial_keyed_p{K}`` arm its
  end-to-end cost.

WHAT IS TIMED. One flush = the submits, ``svc.flush()`` (the launch), and every ticket's ``host()``
(the wait) — the collector's ``submit`` + ``flush`` + ``gpu_wait``. Configs (interleaved in a seeded
random order every round, so drift hits all alike): ``trainee48`` (the trainee's rows alone),
``opp_S{S}`` (``--opp-rows`` rows round-robin over S distinct pool slots), ``both_S{S}`` (the real
step: both in one flush). The rows are REAL observations harvested from the production collector
after its warm-up steps. The per-step distinct-slot count the collector actually meets is read over
the same warm-up (``slots_per_step``). A benchmark's output IS the measurement: busy box => WARN, never
stretch.

    export PYTHONPATH=$PYTHONPATH:src
    python -m main.rust_core_m5.fanout --pool <run>/snapshots --ckpt <run>/final_model.zip \\
        --out /tmp/fanout.json            # re-executes itself under the GPU lock
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from main.rust_core_m5.gates import GPU_LOCK

_MARK = "GEN3AI_M5G_FANOUT_LOCKED"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m main.rust_core_m5.fanout", description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", required=True, help="a run's snapshots/ dir (read-only)")
    ap.add_argument("--ckpt", required=True, help="the trainee .zip (read-only)")
    ap.add_argument("--pool-size", type=int, default=20)
    ap.add_argument("--self-play-fraction", type=float, default=0.95)
    ap.add_argument("--n-envs", type=int, default=48)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--opp-rows", type=int, default=40, help="opponent rows per flush (Lane E's read: ~40)")
    ap.add_argument("--slots", default="1,2,4,8,12,18", help="distinct opponent slots per flush")
    ap.add_argument("--lanes", default="0,1", help="T2 lanes per service (0 = the collector's default)")
    ap.add_argument("--warmup-steps", type=int, default=300, help="collector steps before harvesting")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--reps", type=int, default=50, help="flushes per config per round")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260930)
    return ap


def plan_rows(slots: Sequence[int], n_rows: int) -> List[List[int]]:
    """Row indices ``0..n_rows-1`` dealt round-robin over ``slots`` (the per-slot lists)."""
    out: List[List[int]] = [[] for _ in slots]
    for r in range(int(n_rows)):
        out[r % len(slots)].append(r)
    return out


def _flush(svc: Any, parts: Sequence[Any]) -> tuple:
    from agents.inference.service.spec import Priority

    t0 = time.perf_counter()
    tickets = [svc.submit(int(s), o, m, Priority.ROLLOUT) for s, o, m in parts]
    t1 = time.perf_counter()
    svc.flush()
    t2 = time.perf_counter()
    for t in tickets:
        t.host()
    t3 = time.perf_counter()
    return t1 - t0, t2 - t1, t3 - t2, t3 - t0


def _mean_ci(xs: Sequence[float], seed: int) -> Dict[str, Any]:
    x = np.asarray(xs, dtype=np.float64)
    boot = np.random.default_rng(seed).integers(0, x.size, (10_000, x.size))
    lo, hi = np.percentile(x[boot].mean(1), [2.5, 97.5])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(x.size)}


def measure_service(a: argparse.Namespace, lanes: int) -> Dict[str, Any]:
    """Build one production collector (T2 at ``lanes``), warm it, harvest real rows, time the configs."""
    from main.rust_core_m5 import hooks as H
    from main.rust_core_m5.production import CollectorRustArm

    inference = H.LearnerSampling(a.ckpt, device=a.device, compile=False)
    mix = H.ProductionMix(a.pool, a.pool_size, a.self_play_fraction, compile_opponents=False)
    arm = CollectorRustArm(a.n_envs, a.threads, "proc", "release", inference, mix,
                           H.CompleteGameCollector(98_304), a.seed, sampling="keyed", device=a.device,
                           backend="graph", name=f"fanout_l{lanes}")
    arm.lanes = int(lanes)
    t0 = time.perf_counter()
    arm.build()
    startup = time.perf_counter() - t0
    col, svc = arm.col, arm.col.svc
    per_step: List[int] = []
    rows_per_slot: List[int] = []
    for _ in range(int(a.warmup_steps)):
        c = col.cols
        live = (c["need"][:, 1] == 1) & (c["opp_slot"] >= 0)
        if live.any():
            u, k = np.unique(c["opp_slot"][live], return_counts=True)
            per_step.append(int(u.size))
            rows_per_slot.extend(int(x) for x in k)
        arm.step()
    c = col.cols
    ok = np.flatnonzero((c["need"][:, 0] == 1) & (c["mask"][:, 0].sum(1) > 0))
    if ok.size == 0:
        raise RuntimeError("no trainee row with a legal action to harvest after the warm-up")
    take = np.resize(ok, max(a.n_envs, a.opp_rows))
    obs = np.ascontiguousarray(c["obs"][take, 0]).copy()
    mask = np.ascontiguousarray(c["mask"][take, 0]).copy()
    trainee = int(col.current_slot)
    ids = {}
    for s in range(len(svc._slots)):
        mid = svc.model_id(s)
        if s != trainee and mid and mid not in ids.values():
            ids[s] = mid
    pool_slots = sorted(ids)
    wanted = [int(x) for x in a.slots.split(",")]
    if max(wanted) > len(pool_slots):
        raise RuntimeError(f"asked for {max(wanted)} distinct slots; the service holds {len(pool_slots)} distinct "
                           f"non-trainee models ({len(svc._slots)} slots)")
    configs: Dict[str, List[Any]] = {"trainee48": [(trainee, obs[:a.n_envs], mask[:a.n_envs])]}
    for S in wanted:
        opp = [(pool_slots[j], obs[idx], mask[idx]) for j, idx in enumerate(plan_rows(pool_slots[:S], a.opp_rows))]
        configs[f"opp_S{S}"] = opp
        configs[f"both_S{S}"] = configs["trainee48"] + opp
    rng = np.random.default_rng(a.seed + lanes)
    for parts in configs.values():                      # untimed: first touch of every slot x bucket
        for _ in range(3):
            _flush(svc, parts)
    per_round: Dict[str, List[float]] = {k: [] for k in configs}
    split: Dict[str, List[np.ndarray]] = {k: [] for k in configs}
    for _r in range(int(a.rounds)):
        for name in rng.permutation(list(configs)):
            ts = np.array([_flush(svc, configs[name]) for _ in range(int(a.reps))])
            per_round[name].append(float(1e3 * ts[:, 3].mean()))
            split[name].append(1e3 * ts[:, :3].mean(0))
    stats = svc.stats()
    after = {k: v for k, v in stats.items() if "after_freeze" in k}
    out = {"lanes_requested": int(lanes), "lanes": getattr(svc.engine, "n_lanes", None),
           "buckets": list(getattr(svc.engine, "buckets", []) or []), "slots_declared": len(svc._slots),
           "startup_s": startup, "trainee_slot": trainee, "distinct_pool_models": len(pool_slots),
           "slots_per_step": {"n_steps": len(per_step),
                              "mean": float(np.mean(per_step)) if per_step else None,
                              "p10_p50_p90": ([float(x) for x in np.percentile(per_step, [10, 50, 90])]
                                              if per_step else None)},
           "rows_per_slot_hist": {str(k): int(v) for k, v in zip(*np.unique(rows_per_slot, return_counts=True))},
           "ms_per_flush": {k: _mean_ci(v, a.seed) for k, v in per_round.items()},
           "ms_split_submit_flush_wait": {k: [float(x) for x in np.mean(v, 0)] for k, v in split.items()},
           "after_freeze": after}
    arm.close()
    return out


def derive(res: Dict[str, Any], slots: Sequence[int]) -> Dict[str, Any]:
    """The reads the decision needs, from one service's table."""
    t = {k: v["mean"] for k, v in res["ms_per_flush"].items()}
    top = max(slots)
    return {"trainee_forward_ms": t["trainee48"],
            f"opponent_fanout_ms_at_S{top}": t[f"both_S{top}"] - t["trainee48"],
            "grouped_forward_upper_bound_saving_ms": t[f"both_S{top}"] - t["both_S1"],
            "fewer_snapshots_ms_by_S": {S: t[f"both_S{S}"] for S in slots},
            "marginal_ms_per_extra_slot": ((t[f"opp_S{top}"] - t["opp_S1"]) / (top - 1)) if top > 1 else None}


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    if str(a.device).startswith("cuda") and os.environ.get(_MARK) != "1":
        env = dict(os.environ, **{_MARK: "1"})
        print(f"[fanout] CUDA requested — re-exec under {GPU_LOCK}", file=sys.stderr, flush=True)
        os.execvpe("flock", ["flock", GPU_LOCK, sys.executable, "-m", "main.rust_core_m5.fanout", *argv], env)
    from main.rust_core_m5.throughput import _load, refuse_models_out
    from utils.contention import warn_if_contended

    out_path = refuse_models_out(a.out)
    warnings = []
    load0 = _load()
    if warn_if_contended("m5 T2 fan-out read"):
        warnings.append(f"BUSY BOX at start: contention factor {load0['contention_factor']:.2f}")
    slots = [int(x) for x in a.slots.split(",")]
    per_lanes = {}
    for lanes in (int(x) for x in a.lanes.split(",")):
        res = measure_service(a, lanes)
        res["derived"] = derive(res, slots)
        per_lanes[str(lanes)] = res
        print(f"[fanout] lanes {lanes} (resolved {res['lanes']}): " + json.dumps(res["derived"]), flush=True)
    load1 = _load()
    if load1["contention_factor"] > 1.05:
        warnings.append(f"BUSY BOX at end: contention factor {load1['contention_factor']:.2f}")
    doc = {"schema": "m5_laneG_t2_fanout_v1", "regime": {**{k: v for k, v in vars(a).items() if k != "out"},
                                                          "load_start": load0, "load_end": load1},
           "per_lanes": per_lanes, "warnings": warnings}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1))
    print(f"[fanout] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
