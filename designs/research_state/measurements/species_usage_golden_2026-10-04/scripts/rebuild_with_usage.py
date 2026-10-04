"""Rebuild an arm's golden buffer (`learner_golden.rebuild_buffer`, the real Rust-collector rollout) with the
species-usage marginal HELD at a given file's values, into a scratch path. With the PARENT's marginal it must
replay the parent's buffer: the same games (obs / actions / masks / rewards / episode starts byte-identical), the
behaviour columns within F-X5-45's rebuild rounding — so the rebuilt buffer's new games are the marginal's doing.

    PYTHONPATH=<tree>/src python rebuild_with_usage.py --arm blob --hold <json> --out <npz>
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="blob")
ap.add_argument("--hold", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from agents.gen3_data import priors as P  # noqa: E402
from agents.training import learner_golden as LG  # noqa: E402

table = json.loads(Path(a.hold).read_text())
P.species_usage = lambda: table
LG.arm_buffer = lambda arm: Path(a.out)
LG.rebuild_buffer("proof: rebuild with the species-usage marginal held at " + a.hold, arm=a.arm)
