"""END-TO-END: the REAL K9(b) probe (`consistency.behaviour_probe`, ``fatal``) on a design-(a) cell — the bf16
learner judged against a bf16 T2's stored log mu — with the gate's table REPLACED in this process only by a
candidate bf16 bar ``--bar`` at epsilon ``--eps`` (ceiling 0.15). Then the planted faults the 2026-09-30 table
and `k9_behaviour_exclusion/sweep.py` used: one-optimizer-step-stale weights (``lr * sign(N(0,1))`` on every
weight, seed 5) at each lr of ``common.STALE_LRS``, and ONE wrong-action row (the judged row at the largest
margin with >= 2 legal actions, its action replaced by the first other legal one). Each probe is one update's
FIRST micro-batch (``--chunk`` rows, the probe's own row choice); the unmodified buffer is probed before and
after. Records FATAL or not and the probe's metrics. Nothing in ``src/`` is changed.

    python real_probe.py --state perturb05 --bar 5e-2 --eps 2e-4 --out real_probe_perturb05.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, choices=("fresh", "perturb05", "perturb10", "weights"))
    ap.add_argument("--weights-from", default=None)
    ap.add_argument("--behaviour", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--learner", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--bar", type=float, required=True)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--run-seed", type=int, default=1001)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.device == "cpu":
        th.set_num_threads(a.threads)
    th.set_float32_matmul_precision("highest")
    from agents.model.trunk_precision import set_trunk_precision
    from agents.training.rust_rollout import consistency as K
    from agents.training.rust_rollout.tie_margins import selection_gaps

    model, env, _ = C.build(state=a.state, weights_from=a.weights_from, seed=a.seed, n_envs=a.n_envs,
                            n_steps=a.n_steps, run_seed=a.run_seed, device=a.device, t2_backend="eager",
                            trunk=a.behaviour)
    K.BEHAVIOUR_GATE = (K.GateCondition("max", a.bar, a.eps), K.GateCondition("excluded_frac", C.CEILING, a.eps))
    out: Dict[str, Any] = {"state": a.state, "behaviour": a.behaviour, "learner": a.learner, "bar": a.bar,
                           "eps": a.eps, "chunk": a.chunk}
    try:
        C.collect(model, env)
        set_trunk_precision(model.policy, a.learner)   # T2's replicas were copied at startup: they keep a.behaviour
        model.batch_size = a.chunk
        model.behaviour_check = "fatal"
        model.behaviour_scan_all = False
        model.behaviour_dump_dir = None
        model._current_progress_remaining = 1.0
        buf = model.rollout_buffer

        def probe(name: str) -> None:
            try:
                m = K.behaviour_probe(model)
                out[name] = {"fatal": False, **{k: m.get(k) for k in (
                    "behaviour/max_abs_dlogp_judged", "behaviour/excluded_frac", "behaviour/excluded_frac_before_flip",
                    "behaviour/rows_judged", "behaviour/rows_judged_by_flip", "behaviour/max_abs_dlogp_current")}}
            except K.BehaviourMismatch as exc:
                out[name] = {"fatal": True, "message": str(exc)[:300]}
            print(f"[real-probe] {name}: {out[name]}", flush=True)

        probe("healthy")
        for lr in C.STALE_LRS:
            with C.stale_step(model.policy, lr):
                probe(f"stale_one_adam_step_lr{lr:g}")
        # ONE wrong-action row among the probe's own rows, at the largest margin
        rng = np.random.default_rng([int(getattr(model, "seed", 0) or 0), int(model.num_timesteps)])
        flat = K.choose_rows(np.zeros(buf.log_probs.shape, np.int64), int(model.batch_size), rng)
        t, e = flat // buf.n_envs, flat % buf.n_envs
        gm, _s = selection_gaps(model.policy, {k: v[t, e] for k, v in buf.observations.items()}, buf.actions[t, e],
                                buf.action_masks[t, e], model.device)
        cand = [i for i in np.argsort(-gm) if (buf.action_masks[t[i], e[i]] > 0.5).sum() >= 2]
        i = int(cand[0])
        a0 = int(buf.actions[t[i], e[i]].reshape(-1)[0])
        legal = np.flatnonzero(buf.action_masks[t[i], e[i]] > 0.5)
        buf.actions[t[i], e[i]] = int(legal[legal != a0][0])
        probe("one_wrong_action_row")
        out["one_wrong_action_row"]["row_margin"] = float(gm[i])
        buf.actions[t[i], e[i]] = a0
        probe("healthy_after")
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, default=float)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
