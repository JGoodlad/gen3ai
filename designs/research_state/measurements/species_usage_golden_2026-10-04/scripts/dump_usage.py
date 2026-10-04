"""Dump the facade's species-usage marginal (`gen3_data.priors.species_usage()`) of whichever tree is on
PYTHONPATH, as canonical JSON — the PARENT's (`7bed4347`, Raw count) is the `--hold` file of every proof step.

    PYTHONPATH=<tree>/src python dump_usage.py --out out/parent_species_usage.json
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()

from agents.gen3_data import priors as P  # noqa: E402

Path(a.out).write_text(json.dumps(P.species_usage(), indent=1, sort_keys=True) + "\n")
print(f"wrote {a.out}: {len(P.species_usage())} species")
