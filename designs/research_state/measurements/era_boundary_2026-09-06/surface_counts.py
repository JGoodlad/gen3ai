"""Recount the two flag SURFACES the census quotes, from the live modules.

Run:  PYTHONPATH=src python designs/research_state/measurements/era_boundary_2026-09-06/surface_counts.py
"""
from agents.model.flag_registry import REGISTRY
from main.train_rl_agent import build_parser

p = build_parser()
actions = [a for a in p._actions]
print(f"registry entries (agents.model.flag_registry.REGISTRY) : {len(REGISTRY)}")
print(f"argparse options (build_parser()._actions)             : {len(actions)}")
print(f"  ... excluding the -h/--help action                   : {len(actions) - 1}")
