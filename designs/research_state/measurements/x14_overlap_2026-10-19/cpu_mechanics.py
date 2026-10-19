"""X14 on the CPU: the overlap MECHANICS (not the GPU gain) — a serial and an overlapped `learn()` on the real env
core with tiny fixtures (`rust_rollout/testkit`: the production policy, perturbed; T2 eager on CPU; p2 a seeded
random policy), one arm per subprocess (a clean peak RSS), arms alternated.

    python designs/research_state/measurements/x14_overlap_2026-10-19/cpu_mechanics.py --out result/cpu_mechanics.json

Reads, per iteration after the warm-up: `train/train_ms`, `rollout/collect_ms` (the EXPOSED collection: on the
overlapped arm only the join wait + replay + fill), `overlap/play_ms`, `overlap/join_wait_ms`,
`overlap/prox_forward_ms`; per arm: the learn() wall, decisions, the peak RSS, the snapshot bytes.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time


def arm(overlap: bool, n: int, n_steps: int, batch: int, epochs: int, rollouts: int) -> dict:
    import numpy as np
    import torch

    from agents.training.rust_rollout import testkit as TK
    from agents.training.rust_rollout.build import RustEnvDecl
    from agents.training.rust_vec_env import RustVecEnv

    torch.set_num_threads(1)
    TK.build_selfcheck()
    _a, obs, act = TK.production_spaces()
    decl = RustEnvDecl(n_envs=n, threads=2, front="ffi", profile="selfcheck", n_steps=n_steps, micro_batch=batch,
                       device="cpu", backend="eager", run_seed=11, gamma=1.0, gae_lambda=0.8,
                       **({"overlap": True} if overlap else {}))
    env = RustVecEnv(n_envs=n, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl))
    model = TK.fresh_model(env, n_steps=n_steps, batch_size=batch, n_epochs=epochs)
    col = env.startup(model)
    # `warn`, not `fatal`: on this tiny fixture one tie-heavy game can dominate a probe's rows and trip K9(b)'s
    # EXCLUDED-share ceiling (0.15) under ANY weights (measured: 0.402 under the snapshot AND under the learner on the
    # same 92 rows) — a fixture property, not the overlap's; the judged |d log pi| is still recorded below
    model.behaviour_check = "warn"
    dumps = TK.record_dumps(model)
    t0 = time.perf_counter()
    try:
        model.learn(total_timesteps=rollouts * n * n_steps)
    finally:
        wall = time.perf_counter() - t0
        env.close()
    keys = ("train/train_ms", "rollout/collect_ms", "overlap/play_ms", "overlap/join_wait_ms",
            "overlap/prox_forward_ms", "overlap/is_ess", "overlap/is_weight_mean", "overlap/is_weight_truncated_frac",
            "overlap/behaviour_kl", "behaviour_snapshot/rows_current", "behaviour_snapshot/max_abs_dlogp_judged",
            "behaviour/max_abs_dlogp_judged", "staleness/age_mean", "train/approx_kl", "train/clip_fraction")
    rows = [{k: float(d[k]) for k in keys if k in d} | {"_step": d["_step"]} for d in dumps]
    return {"overlap": overlap, "wall_s": wall, "decisions": int(model.num_timesteps),
            "updates": int(model._n_updates) // epochs,
            "snapshot_bytes": int(col.overlap.snapshots.nbytes) if overlap else 0,
            "overlapped_rollouts": int(col.overlap.rollouts) if overlap else 0,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, "dumps": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=32)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--rollouts", type=int, default=6)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--arm", choices=("on", "off"))
    a = ap.parse_args()
    if a.arm:
        r = arm(a.arm == "on", a.n, a.n_steps, a.batch, a.epochs, a.rollouts)
        with open(a.out, "w") as f:
            json.dump(r, f)
        return
    runs = []
    for rep in range(a.reps):
        for name in ("off", "on") if rep % 2 == 0 else ("on", "off"):
            tmp = f"{a.out}.{name}{rep}.json"
            cmd = [sys.executable, __file__, "--out", tmp, "--arm", name, "--n", str(a.n), "--n-steps", str(a.n_steps),
                   "--batch", str(a.batch), "--epochs", str(a.epochs), "--rollouts", str(a.rollouts)]
            subprocess.run(cmd, check=True)
            with open(tmp) as f:
                runs.append(json.load(f) | {"rep": rep})
            os.remove(tmp)
            print(name, rep, f"{runs[-1]['wall_s']:.1f}s", flush=True)
    load = os.getloadavg()
    out = {"config": vars(a), "loadavg_at_end": load, "runs": runs}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
