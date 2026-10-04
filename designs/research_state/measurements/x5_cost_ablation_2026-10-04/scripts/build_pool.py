"""A 20-snapshot self-play pool SERVABLE beside a fresh trainee, for regime A (X5 cost ablation, 2026-10-04).

    python build_pool.py --template <run>/snapshots/snapshot_000000000000.zip --out <new run>/snapshots [--n 20]

The template is the step-0 pool seed a fresh launch at the SAME flags writes, so the T2 FORWARD
fingerprint equals the trainee's. Each snapshot = the template with the gates' seeded perturbation
(`parity_probe.perturb_`, seed 1000 + i, scale 0.05), so the 20 weight sets are distinct and the T2
fan-out (one slot per snapshot) is the steady state's. The pool's metadata: `summary.json` with
`win_rate_vs_bots` 0.94 (the bottleneck profile's seeded value: the ramp opens to its full self-play
share from the first rollout) and the template pool's `model_config.json` (the pool's arch record).
Nothing trains on these weights that is read as a result; they exist so the measured launch runs
the pool's forward cost. (Same purpose as the bottleneck profile's `transplant_pool.py`; no archived
X5 checkpoint exists to transplant.)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from agents.inference.service.slots import forward_fingerprint
from agents.model.parity_probe import perturb_
from agents.model.snapshot import load_checkpoint_strict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src_dir = os.path.dirname(a.template)
    fp0 = None
    for i in range(a.n):
        model = load_checkpoint_strict(a.template, device="cpu")
        fp = forward_fingerprint(model.policy)
        fp0 = fp0 or fp
        assert fp == fp0
        perturb_(model.policy, seed=1000 + i, scale=0.05)
        step = 10_000 * (i + 1)
        dst = os.path.join(a.out, f"snapshot_{step:012d}")
        model.save(dst)
        print("wrote", dst + ".zip")
    cfg = os.path.join(src_dir, "model_config.json")
    if os.path.isfile(cfg):
        shutil.copy2(cfg, os.path.join(a.out, "model_config.json"))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"win_rate_vs_bots": 0.94, "self_play_fraction": 0.9, "seeded": True}, f, indent=2)
    with open(os.path.join(a.out, "win_rate_vs_bots.txt"), "w") as f:
        f.write("0.940000\n")
    print("fingerprint", (fp0 or "")[:12], "model_config copied:", os.path.isfile(cfg))


if __name__ == "__main__":
    main()
