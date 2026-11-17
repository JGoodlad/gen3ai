"""The plateau-candidate-against-P* read's operating characteristics: Amendment 6's twin of ``kakuna_oc.py``
(PREREG §18 Amendment 6, item 4).

``kakuna_oc.py`` is the registration's script: it simulates a Kakuna read on the anchor lane's own deals and classes
every experiment through ``main.b1.kakuna_rule``'s own functions. Amendment 6 registers a read with TWO arms under the
same rule, unchanged: ``cand``, the continuing cell's three runs at their plateau, and ``pstar``, the ONE file of the
named baseline ``pstar_plateau``; its contrast is ``cand - pstar``. The rule pools the seed variance within arms on
``sum(n_a - 1)`` = 2 degrees of freedom (all from ``cand``) and charges it to both arms through the run weights
``c_a / n_a`` (1/3 each and -1), so P* is read as one draw of its own lineage. This script changes the arms table, the
contrast and the arm the effect is added to, and runs the registration's ``main`` as it stands (imported, never
copied): the same deals, team model, levels, seed-SD grid, effect grid, sensitivity models, seeds and report.

Deterministic (the registration's ``SeedSequence``); CPU only:
    PYTHONPATH=<checkout>/src python kakuna_oc_pstar.py > kakuna_oc_pstar.log
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kakuna_oc as K  # noqa: E402  (the registration's script, from this folder)

N = 3
K.ARMS = {"cand": [f"k{i}" for i in range(1, N + 1)], "pstar": ["p1"]}
K.MAIN = {"cand": 1.0, "pstar": -1.0}
K.ON = ("cand",)
K.CONFIRMING = {"candidate_U": "U", "candidate_F": "F"}

if __name__ == "__main__":
    # The workers must inherit the two-arm table above: a spawned worker would re-import kakuna_oc with four arms.
    multiprocessing.set_start_method("fork")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--boot-reps", type=int, default=1000)
    ap.add_argument("--boot-draws", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    print(f"arms {K.ARMS} ({sum(len(v) for v in K.ARMS.values())} runs); contrast {K.MAIN}; the effect is added to "
          f"{K.ON}")
    sys.exit(K.main(args.reps, args.boot_reps, args.boot_draws, args.procs))
