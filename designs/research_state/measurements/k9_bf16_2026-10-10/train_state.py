"""The FEW-UPDATES-TRAINED state for the K9(b)-under-bf16 measurement (`README.md`).

From the seeded FRESH ``--arch endstate`` init (seed 1001, no perturbation), ``--updates`` REAL PPO updates on
the CPU, fp32 throughout (today's learner): each collects a seeded complete-game Rust-collector rollout
(``--n-envs`` x ``--n-steps`` rows, T2 eager, p2 a seeded random policy), runs
``InstrumentedMaskablePPO.train()`` with the resolved endstate training hparams (`apply_training_hparams`:
lr 3e-4 and its KL controller, clip 0.15, n_epochs 10, the loss terms) except ONE micro-batch per optimizer
step (``grad_accum_steps`` 1, ``batch_size`` = the rollout) — a CPU rollout cannot fill the recipe's
65,536-row step — then loads the new weights into T2 (`collector.after_update`). K9(b) runs ``warn`` at fp32
each update (its metrics are logged here: today's fp32 gate on this state). Writes the policy state dict
(outside ``models/``).

    python train_state.py --out ~/gen3ai_archive/k9_bf16_2026-10-10/trained_u4.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--updates", type=int, default=4)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--run-seed", type=int, default=2002)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if os.path.abspath(os.path.expanduser(a.out)).find("/models/") >= 0:
        raise SystemExit("refusing to write under models/")
    th.set_num_threads(a.threads)
    model, env, args = C.build(state="fresh", weights_from=None, seed=a.seed, n_envs=a.n_envs, n_steps=a.n_steps,
                               run_seed=a.run_seed, device=a.device, t2_backend="eager", trunk="fp32",
                               train_hparams=True)
    model.behaviour_check = "warn"
    model.behaviour_dump_dir = None
    from agents.training.rust_rollout import testkit as TK
    log = []
    try:
        col = env.startup(model)
        for u in range(a.updates):
            t0 = time.time()
            if not col.collect(model, TK.NullCallback(), model.rollout_buffer):
                raise RuntimeError("collector stopped")
            t1 = time.time()
            th.manual_seed(a.seed + u)
            np.random.seed(a.seed + u)
            model._current_progress_remaining = 1.0
            model.train()
            t2 = time.time()
            col.after_update(model)
            vals = {k: float(v) for k, v in model.logger.name_to_value.items()
                    if isinstance(v, (int, float, np.floating)) and (k.startswith("behaviour/") or k.startswith("train/"))}
            row = {"update": u, "collect_s": round(t1 - t0, 1), "train_s": round(t2 - t1, 1),
                   "lr": float(model.policy.optimizer.param_groups[0]["lr"]), **vals}
            log.append(row)
            print(json.dumps({k: row[k] for k in row if k in ("update", "collect_s", "train_s", "lr")
                              or k in ("behaviour/excluded_frac", "behaviour/max_abs_dlogp_judged",
                                       "behaviour/selection_free", "train/approx_kl", "train/clip_fraction",
                                       "train/loss")}), flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(a.out))), exist_ok=True)
        th.save(model.policy.state_dict(), os.path.expanduser(a.out))
        with open(os.path.expanduser(a.out) + ".log.json", "w") as f:
            json.dump({"argv": sys.argv, "updates": log}, f, indent=1)
        print(f"wrote {a.out}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
