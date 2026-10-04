"""Attribute a routine-gate failure to the marginal: run the named pytest node ids IN-PROCESS with the species-usage
marginal optionally HELD at a file's values (`gen3_data.priors.species_usage` replaced before anything builds), and
print each node's outcome.

    PYTHONPATH=<tree>/src python precondition_attrib.py [--hold out/parent_species_usage.json] NODE [NODE ...]
"""
import argparse
import json
from pathlib import Path

import pytest

ap = argparse.ArgumentParser()
ap.add_argument("--hold", default=None)
ap.add_argument("nodes", nargs="+")
a = ap.parse_args()

from agents.gen3_data import priors as P  # noqa: E402

if a.hold:
    table = json.loads(Path(a.hold).read_text())
    P.species_usage = lambda: table


class _Rec:
    def __init__(self):
        self.out = {}

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.outcome != "passed":
            self.out[report.nodeid] = report.outcome


rec = _Rec()
pytest.main(["-q", "-p", "no:cacheprovider", "-p", "no:randomly", *a.nodes], plugins=[rec])
print(json.dumps({"hold": a.hold or "tree", "outcomes": rec.out}, indent=1))
