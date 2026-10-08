"""K9(b) FLIP-JUDGE replay of the static-screen u1480 stop (gen3_behaviour_tie_flip_judge_v1, 2026-10-08).

The `k9_static_tie_2026-10-08` harness (the k9_early_probe build: a seeded 2,048-row complete-game rollout on
the Rust collector, 16 envs x 128 steps, T2 eager on CPU, p2 a seeded random policy, at the weights given by
``--weights-from``, arm ``static_fm`` = fixed_mass x token_encoding static), then the REAL probe
(`consistency.behaviour_probe`, ``warn``) on EVERY row as one micro-batch. It reports the excluded share
before and after the flip-judge, how many rows the flip judged, how many needed the flipped resolution, and
the probe's cost. A FORCED read then resolves EVERY flippable row the other way (one `TieFlip` forward) and
reports how far the other resolution sits from the stored behaviour log-prob: had T2 taken it, would the
row pass?

Run with cwd = the checkout under test (its designs/ and src/ on PYTHONPATH), CPU only:
    python <this file> --weights-from <pt> --out <json>
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch as th

ROOT = os.getcwd()
p = os.path.join(ROOT, "designs/research_state/measurements/k9_early_probe_2026-10-06/measure.py")
spec = importlib.util.spec_from_file_location("k9_early", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.old.ARMS["static_fm"] = {"belief_tokens": "fixed_mass", "token_encoding": "static"}

from agents.training.rust_rollout import consistency as K  # noqa: E402
from agents.training.rust_rollout.tie_margins import TieFlip, TieMargins  # noqa: E402


def main() -> int:
    import argparse
    import subprocess

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--run-seed", type=int, default=1001)
    a = ap.parse_args()
    m._WEIGHTS_FROM = a.weights_from
    model, env = m.build("static_fm", None, 1001, a.n_envs, a.n_steps, a.run_seed)
    try:
        buf = model.rollout_buffer
        n = int(buf.log_probs.size)
        model.batch_size = n                       # every row, unpermuted (choose_rows keeps a full current set)
        model.behaviour_check = "warn"
        model._current_progress_remaining = 1.0
        t0 = time.perf_counter()
        out = K.behaviour_probe(model)
        wall = time.perf_counter() - t0
        res = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
               "weights": a.weights_from, "rows": n, "eps": K.tie_eps(), "bar": K.BEHAVIOUR_BAR,
               "ceiling": K.FP32_EXCLUDED_CEILING, "probe_wall_s": wall,
               "metrics": {k: v for k, v in out.items() if k.startswith("behaviour/")}}
        # FORCED: every flippable row resolved the other way, against the stored behaviour log-prob
        n_envs = int(buf.n_envs)
        f = np.arange(n)
        tt, ee = f // n_envs, f % n_envs
        obs = {k: v[tt, ee] for k, v in buf.observations.items()}
        acts = th.as_tensor(buf.actions[tt, ee].reshape(-1)).long()
        masks = th.as_tensor(buf.action_masks[tt, ee])
        mu = buf.log_probs[tt, ee].astype(np.float64)
        cost = {}                                   # the forward alone, the recorder before, the recorder now
        for label, mk in (("bare", lambda: None), ("recorder_plain", lambda: TieMargins(n)),
                          ("recorder_near", lambda: TieMargins(n, near_eps=K.tie_eps()))):
            ts = []
            for _rep in range(3):
                md = mk()
                t1 = time.perf_counter()
                K.probe_forward(model, obs, acts, masks, md)
                ts.append(1e3 * (time.perf_counter() - t1))
            cost[label + "_ms_median3"] = float(np.median(ts))
        res["cost"] = cost
        rec = TieMargins(n, near_eps=K.tie_eps())
        t1 = time.perf_counter()
        new, _ = K.probe_forward(model, obs, acts, masks, rec)
        rec_ms = 1e3 * (time.perf_counter() - t1)
        fl = np.flatnonzero(rec.flippable())
        ex = rec.margin < K.tie_eps()
        sites = {}
        for r in np.flatnonzero(ex):
            st = rec._steps.get(int(r)) if rec.flippable()[r] else None
            key = (rec.site[int(r)], "flippable" if st is not None else
                   ("multi" if rec.near_calls[r] > 1 or rec.near_elems[r] > 1 else "not_expressed"))
            sites[key] = sites.get(key, 0) + 1
        forced = {}
        if fl.size:
            mode = TieFlip(n, rec.flip_plan(fl), rec.call_sites)
            t2 = time.perf_counter()
            alt, _ = K.probe_forward(model, obs, acts, masks, mode)
            flip_ms = 1e3 * (time.perf_counter() - t2)
            mode.check_applied()
            d_new, d_alt, moved = np.abs(new - mu)[fl], np.abs(alt - mu)[fl], np.abs(alt - new)[fl]
            bar = K.BEHAVIOUR_BAR
            forced = {"rows": int(fl.size), "flip_forward_ms": flip_ms, "recording_forward_ms": rec_ms,
                      "flip_moves_logpi": int((moved > 0).sum()), "max_abs_logpi_move": float(moved.max()),
                      "median_abs_logpi_move": float(np.median(moved)),
                      "pass_as_is": int((d_new < bar).sum()), "pass_flipped_only": int(((d_alt < bar) & ~(d_new < bar)).sum()),
                      "fail_both": int((~(d_new < bar) & ~(d_alt < bar)).sum()),
                      "max_abs_dlogp_as_is": float(d_new.max()),
                      "untouched_rows_max_move": float(np.abs(alt - new)[~ex].max()) if (~ex).any() else 0.0}
        res["excluded_by_site"] = [{"site": k[0], "class": k[1], "rows": v} for k, v in
                                   sorted(sites.items(), key=lambda kv: -kv[1])]
        res["forced"] = forced
        json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps(res, indent=1))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
