"""Which runs LOAD today, and which stop loading if MIGRATION_FLOOR rises to 110.

Today a run loads iff config_version >= MIGRATION_FLOOR (96) AND its arch_signature
matches the live one. After a v110 floor bump every config_version <= 109 is refused,
so the runs that CHANGE STATUS are exactly those loadable today.

Run:  PYTHONPATH=src python designs/research_state/measurements/era_boundary_2026-09-06/run_loadability.py
"""
import json
import pathlib
import os

from agents.model.model_version import ARCH_SIGNATURE, MIGRATION_FLOOR
from utils.paths import main_models_dir

NEW_FLOOR = 110

models = main_models_dir()
rows = []
for d in sorted(os.listdir(models)):
    cfg = os.path.join(models, d, "model_config.json")
    if not os.path.exists(cfg):
        continue
    with open(cfg) as fh:
        data = json.load(fh)
    rows.append((d, int(data.get("config_version", 0)), data.get("arch_signature", "?")))

live_sig = [r for r in rows if r[2] == ARCH_SIGNATURE]
loadable = [r for r in rows if r[1] >= MIGRATION_FLOOR and r[2] == ARCH_SIGNATURE]
sig_ok_below_floor = [r for r in rows if r[1] < MIGRATION_FLOOR and r[2] == ARCH_SIGNATURE]
gen9 = [r for r in rows if r[1] >= 69]

print(f"live ARCH_SIGNATURE                 : {ARCH_SIGNATURE}")
print(f"MIGRATION_FLOOR today               : {MIGRATION_FLOOR}   (proposed: {NEW_FLOOR})")
print(f"runs with a model_config.json       : {len(rows)}")
print(f"  ... arch_signature == live        : {len(live_sig)}")
print(f"  ... gen-9+ (config_version >= 69) : {len(gen9)}")
print(f"LOADABLE TODAY (>= floor AND sig)   : {len(loadable)}")
print(f"  live sig but BELOW today's floor  : {len(sig_ok_below_floor)}")
print()
print(f"=== the {len(loadable)} runs that become UNLOADABLE at floor {NEW_FLOOR} ===")
print(f"{'config_version':>14}  run")
for d, v, _ in sorted(loadable, key=lambda r: (-r[1], r[0])):
    print(f"{v:>14}  {d}")

by_v = {}
for _, v, _ in loadable:
    by_v[v] = by_v.get(v, 0) + 1
print("\nby config_version:", dict(sorted(by_v.items())))

with open("designs/research_state/measurements/era_boundary_2026-09-06/loadability.json", "w") as fh:
    json.dump(
        {
            "arch_signature": ARCH_SIGNATURE,
            "floor_today": MIGRATION_FLOOR,
            "floor_proposed": NEW_FLOOR,
            "n_runs_with_config": len(rows),
            "n_live_signature": len(live_sig),
            "n_gen9": len(gen9),
            "loadable_today": [{"run": d, "config_version": v} for d, v, _ in loadable],
        },
        fh,
        indent=2,
    )
print("\nwrote designs/research_state/measurements/era_boundary_2026-09-06/loadability.json")
