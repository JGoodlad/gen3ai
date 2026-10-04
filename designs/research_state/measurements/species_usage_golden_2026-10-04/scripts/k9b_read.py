"""K9(b) on an arm's COMMITTED buffer under this tree (`learner_golden.behaviour_read`: one real `train()` with
`--behaviour-check fatal`): does the stored behaviour log-prob still match this tree's seeded learner? Prints
the probe's read, or the `BehaviourMismatch` it raised — the reason a buffer must be rebuilt.

    PYTHONPATH=<tree>/src python k9b_read.py --arm blob [--hold out/parent_species_usage.json]
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="blob")
ap.add_argument("--hold", default=None, help="hold the species-usage marginal at THIS file's values")
a = ap.parse_args()

from agents.gen3_data import priors as P  # noqa: E402
from agents.training import learner_golden as LG  # noqa: E402

if a.hold:
    table = json.loads(Path(a.hold).read_text())
    P.species_usage = lambda: table
try:
    read = LG.behaviour_read(a.arm)
    print(json.dumps({"arm": a.arm, "hold": a.hold, "k9b": "PASS", **read}, indent=1))
except Exception as e:  # noqa: BLE001 — the verdict IS the exception
    print(json.dumps({"arm": a.arm, "hold": a.hold, "k9b": "FAIL", "error": f"{type(e).__name__}: {e}"[:2000]},
                     indent=1))
