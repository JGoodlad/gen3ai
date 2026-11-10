"""BOUNDARY-1's Kakuna meter at THREE runs an arm: Amendment 1's twin of ``kakuna_oc.py`` (PREREG §11.4, §18).

``kakuna_oc.py`` is the registration's script: it simulates the read on the anchor lane's own deals and classes every
experiment through ``main.b1.kakuna_rule``'s own functions, at two runs an arm (8 runs). Amendment 1 registers three
runs an arm (12 runs), which the rule takes unchanged (its seed variance is pooled within arms on ``sum(n_a - 1)``
degrees of freedom and its run weights are ``c_a / n_a``). This script changes ONE thing, the arms table, and runs
the registration's ``main`` as it stands (imported, never copied): the same deals, team model, levels, sensitivity
models, seeds and report.

Deterministic (the registration's ``SeedSequence``); CPU only:
    PYTHONPATH=<checkout>/src python kakuna_oc_n3.py > kakuna_oc_n3.log
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kakuna_oc as K  # noqa: E402  (the registration's script, from this folder)

N = 3
K.ARMS = {"C": [f"c{i}" for i in range(1, N + 1)], "W": [f"w{i}" for i in range(1, N + 1)],
          "Br": [f"b{i}" for i in range(1, N + 1)], "WB": [f"wb{i}" for i in range(1, N + 1)]}

if __name__ == "__main__":
    # The workers must inherit the three-run table above: a spawned worker would re-import kakuna_oc with two.
    multiprocessing.set_start_method("fork")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--boot-reps", type=int, default=1000)
    ap.add_argument("--boot-draws", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    print(f"n {N} runs an arm ({sum(len(v) for v in K.ARMS.values())} runs): {K.ARMS}")
    sys.exit(K.main(args.reps, args.boot_reps, args.boot_draws, args.procs))
