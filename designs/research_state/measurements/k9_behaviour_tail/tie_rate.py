"""How often does a HEALTHY row sit at a selection tie (`tie_margins.selection_gaps`)? Rule (iii) has
teeth only if a random row rarely does. Collects real rows on the CPU (rust core, the same checkpoint and
pool as the sweep), and reports the fraction of rows whose smallest selection gap is <= eps for several eps;
then the sweep's violators' own gaps (they must sit at a tie).

    python tie_rate.py --ckpt <ckpt.zip> --pool <snapshots> --sweep <sweep out> --fills 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep import build  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--fills", type=int, default=4)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--target", type=int, default=2048)
    a = ap.parse_args()
    from agents.training.rust_rollout import store as S
    from agents.training.rust_rollout.tie_margins import selection_gaps

    ns = argparse.Namespace(ckpt=a.ckpt, pool=a.pool, pool_size=2, self_play_fraction=0.9, n_envs=a.n_envs,
                            target=a.target, seed=777, device="cpu")
    arm, learner, _ = build(ns)
    col, buf = arm.col, arm._buf
    gaps, sites = [], {}
    for _f in range(a.fills):
        while not col.ready():
            col.host_step()
        S.fill_complete(buf, col.log, int(arm._target), current_version=col.version)
        n = buf.log_probs.size
        for s in range(0, n, 256):
            f = np.arange(s, min(n, s + 256))
            t, e = f // buf.n_envs, f % buf.n_envs
            g, st = selection_gaps(learner.policy, {k: v[t, e] for k, v in buf.observations.items()},
                                   buf.actions[t, e], buf.action_masks[t, e], learner.device)
            gaps.append(g)
            for x in st:
                sites[x.split("@")[1] if "@" in x else x] = sites.get(x.split("@")[1] if "@" in x else x, 0) + 1
    arm.close()
    g = np.concatenate(gaps)
    out = {"rows": int(g.size), "frac_at_or_below": {f"{eps:g}": float((g <= eps).mean())
                                                    for eps in (0.0, 1e-7, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3)},
           "quantiles": {q: float(np.quantile(g, q)) for q in (0.001, 0.01, 0.05, 0.5)},
           "argmin_sites": dict(sorted(sites.items(), key=lambda kv: -kv[1])[:8])}
    rows = [json.loads(x) for x in open(Path(a.sweep) / "violators.jsonl")]
    z = np.load(Path(a.sweep) / "violators.npz")
    vg, vs = selection_gaps(learner.policy, {k: z[k] for k in z.files}, np.array([r["action"] for r in rows]),
                            np.array([[c == "1" for c in r["mask"]] for r in rows], dtype=np.float32),
                            learner.device)
    out["violators"] = [{"fill": r["fill"], "min_gap": float(x), "site": s} for r, x, s in zip(rows, vg, vs)]
    print(json.dumps(out, indent=1))
    (Path(a.sweep) / "tie_rate.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
