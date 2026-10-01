"""K9(b) — the LOCALIZED-corruption signal (the orchestrator's TF32 max bar, 2026-09-30).

The TF32 gate is TWO conditions on the first micro-batch: the p99 (catches GLOBAL faults — stale
weights, mode, sampling) and the max (catches LOCALIZED gross faults the p99 cannot see when they touch
< 1 % of rows). This measures the second class on the same real rows as ``measure.py``: corrupt 0.5 % of
the rows (10 of each 2,048-row micro-batch) and read, per micro-batch, the p99 and the max of
|log π_learner − log μ_rollout|, plus the corrupted rows' own |Δ| distribution:

* ``action_index`` — the row's action points at a DIFFERENT legal action (a wrong action index);
* ``obs_swap``     — two corrupted rows exchange observations + masks (rows misaligned after a shuffle);
  a stored action that is illegal in the swapped-in state is COUNTED (its log π is the mask floor);
* ``mask_mismatch`` — the learner's action mask on the row allows every action (a mask mismatch).

Faults are evaluated at update 0 and after 2 real updates, per seed, eager, at ``highest`` and
``high``. The learner is a FRESH perturbed production model (nearly uniform policy): a trained policy is
sharper, so these signals are a LOWER bound on what a trained run would show (UNVERIFIED on a trained
checkpoint).

One ``scripts/ops/gpu_lock.sh timeout 600 ...`` acquisition per (precision, seed) unit (a few minutes each); durable
``parts/corrupt_<precision>_s<seed>.json``, so a re-run resumes.

    python designs/research_state/measurements/k9_behaviour_bar_2026-09-30/corrupt.py \\
        --out designs/research_state/measurements/k9_behaviour_bar_2026-09-30/corrupt_result.json
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).parent))
import measure as M  # noqa: E402  (the same rows, learner and forwards)

RATE = 0.005
KINDS = ("action_index", "obs_swap", "mask_mismatch")


def corrupt(rows, kind: str, rng: np.random.Generator):
    """A deep copy of ``rows`` with RATE of the rows corrupted by ``kind``; returns (rows', flat idx)."""
    r = copy.deepcopy(rows)
    n = M.N_ENVS * M.N_STEPS
    k = max(2, int(round(RATE * n)))
    k += k % 2
    idx = rng.choice(n, size=k, replace=False)
    t, e = idx // M.N_ENVS, idx % M.N_ENVS
    if kind == "action_index":
        for ti, ei in zip(t, e):
            legal = np.flatnonzero(r["action_masks"][ti, ei] > 0.5)
            a = int(r["actions"][ti, ei, 0])
            others = legal[legal != a]
            if others.size:
                r["actions"][ti, ei, 0] = rng.choice(others)
    elif kind == "obs_swap":
        half = k // 2
        for (t1, e1), (t2, e2) in zip(zip(t[:half], e[:half]), zip(t[half:], e[half:])):
            for key in r["obs"]:
                a, b = r["obs"][key][t1, e1].copy(), r["obs"][key][t2, e2].copy()
                r["obs"][key][t1, e1], r["obs"][key][t2, e2] = b, a
            a, b = r["action_masks"][t1, e1].copy(), r["action_masks"][t2, e2].copy()
            r["action_masks"][t1, e1], r["action_masks"][t2, e2] = b, a
    elif kind == "mask_mismatch":
        for ti, ei in zip(t, e):
            r["action_masks"][ti, ei] = 1.0
    return r, idx


def run_unit(prec: str, seed: int) -> dict:
    """One (precision, seed) unit — a few minutes; its own lock acquisition and durable part."""
    th.set_float32_matmul_precision(prec)
    out = {"healthy_micro": [], **{k: {"micro": [], "rows": [], "illegal": 0} for k in KINDS}}
    rows = M.collect_rows(seed)
    m = M.learner(seed, rows, False)
    rng = np.random.default_rng(1000 + seed)
    for k in range(3):
        if k in (0, 2):
            mu = M.logp(m, rows, batch=M.N_ENVS, train_mode=False)
            out["healthy_micro"] += M.micro_stats(M.logp(m, rows, batch=M.MICRO, train_mode=True) - mu)
            for kind in KINDS:
                bad, idx = corrupt(rows, kind, rng)
                d = M.logp(m, bad, batch=M.MICRO, train_mode=True) - mu
                out[kind]["micro"] += M.micro_stats(d)
                a = np.abs(d[idx])
                out[kind]["illegal"] += int((a > 10.0).sum())
                out[kind]["rows"] += [float(x) for x in a]
        if k < 2:
            M.update(m, rows, M.logp(m, rows, batch=M.N_ENVS, train_mode=False), n_epochs=2,
                     batch=M.MICRO, lr=M.LR_RECIPE, seed=200 + k)
    return out


def merge(parts: list) -> dict:
    out = {"healthy_micro": [], **{k: {"micro": [], "rows": [], "illegal": 0} for k in KINDS}}
    for p in parts:
        out["healthy_micro"] += p["healthy_micro"]
        for k in KINDS:
            out[k]["micro"] += p[k]["micro"]
            out[k]["rows"] += p[k]["rows"]
            out[k]["illegal"] += p[k]["illegal"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit", default=None, help="internal: 'precision:seed' in this process")
    ap.add_argument("--smoke", action="store_true", help="CPU, one seed, fp32 (a code check)")
    ap.add_argument("--cpu-seed", default=None, help="one CPU fp32 unit for this seed (durable part, no lock)")
    a = ap.parse_args()
    if a.smoke:
        M.DEVICE = "cpu"
        a.unit = "highest:0"
    if a.cpu_seed is not None:      # the fault SIGNAL on CPU fp32 (no lock): parts/corrupt_cpu_s<seed>.json
        M.DEVICE = "cpu"
        part = Path(a.out).parent / "parts" / f"corrupt_cpu_s{a.cpu_seed}.json"
        if not part.exists():
            part.parent.mkdir(exist_ok=True)
            tmp = part.with_suffix(".partial")
            tmp.write_text(json.dumps(run_unit("highest", int(a.cpu_seed))) + "\n")
            tmp.rename(part)
        return 0
    if a.unit:
        prec, seed = a.unit.split(":")
        Path(a.out).write_text(json.dumps(run_unit(prec, int(seed))) + "\n")
        return 0
    t0 = time.time()
    parts = Path(a.out).parent / "parts"
    parts.mkdir(exist_ok=True)
    lock = str(Path(__file__).resolve().parents[4] / "scripts" / "ops" / "gpu_lock.sh")
    res = {"setup": {"rate": RATE, "kinds": list(KINDS), "seeds": list(M.SEEDS), "micro": M.MICRO,
                     "rows_per_seed": M.N_ENVS * M.N_STEPS, "states": "update 0 and after 2 real updates",
                     "arm": "eager", "micro_stats_columns": ["max", "p99", "mean"],
                     "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                              text=True).stdout.strip()}, "units": {}}
    for prec in ("high", "highest"):
        got = []
        for seed in M.SEEDS:
            part = parts / f"corrupt_{prec}_s{seed}.json"
            if not part.exists():
                tmp = part.with_suffix(".partial")
                r = subprocess.run([lock, "timeout", "600", sys.executable, __file__, "--out", str(tmp),
                                    "--unit", f"{prec}:{seed}"])
                if r.returncode != 0:
                    raise SystemExit(f"unit {prec}:{seed} failed ({r.returncode}); written units are kept")
                tmp.rename(part)
            got.append(json.loads(part.read_text()))
        res["units"][prec] = merge(got)
    res["wall_s"] = time.time() - t0
    Path(a.out).write_text(json.dumps(res) + "\n")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
