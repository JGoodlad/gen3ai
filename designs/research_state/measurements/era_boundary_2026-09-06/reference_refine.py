"""Refine the reference scan: a run named in a committed ARTIFACT is a record; a run
named in a SCRIPT that loads a checkpoint is a dependency. Separate them.

Run:  PYTHONPATH=src python designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.py
"""
import json
import pathlib
import os
import subprocess

from utils.paths import repo_root

ROOT = repo_root()
# This census's own artifacts name every run by construction; excluding them is
# what stops the scan from scoring itself as a dependency.
SELF_DIR = "era_boundary_2026-09-06"
MEAS = os.path.join(ROOT, "designs/research_state/measurements")
MAIN = os.path.join(ROOT, "src/main")
AGENTS = os.path.join(ROOT, "src/agents")
LEDGER = os.path.join(ROOT, "designs/research_state/ledger.md")

# a script that LOADS a checkpoint imports one of these
LOADER_TOKENS = ("load_model_snapshot", "MaskablePPO.load", "resolve_model_ref",
                 "SnapshotPool", "load_snapshot")

with open("designs/research_state/measurements/era_boundary_2026-09-06/loadability.json") as fh:
    runs = [r["run"] for r in json.load(fh)["loadable_today"]]

with open(LEDGER) as fh:
    tail = fh.readlines()[-1000:]


def hits(pat, path, include):
    p = subprocess.run(["grep", "-rIl", f"--exclude-dir={SELF_DIR}", f"--include={include}", "-F", pat, path],
                       capture_output=True, text=True)
    return [os.path.relpath(f, ROOT) for f in p.stdout.splitlines() if f]


def is_loader(path):
    try:
        with open(os.path.join(ROOT, path)) as fh:
            txt = fh.read()
    except OSError:
        return False
    return any(t in txt for t in LOADER_TOKENS)


rows = {}
for run in runs:
    meas_py = hits(run, MEAS, "*.py")
    meas_json = hits(run, MEAS, "*.json")
    meas_md = hits(run, MEAS, "*.md")
    main_py = [f for f in hits(run, MAIN, "*.py")]
    agents_py = [f for f in hits(run, AGENTS, "*.py")]
    loaders = [f for f in (meas_py + main_py + agents_py) if is_loader(f)]
    tail_n = sum(1 for line in tail if run in line)
    rows[run] = {
        "meas_scripts": meas_py, "meas_artifacts": len(meas_json) + len(meas_md),
        "src_py": main_py + agents_py, "loader_scripts": loaders,
        "ledger_tail_lines": tail_n,
    }

n_loader = sum(1 for v in rows.values() if v["loader_scripts"])
n_script = sum(1 for v in rows.values() if v["meas_scripts"] or v["src_py"])
n_artifact_only = sum(1 for v in rows.values()
                      if v["meas_artifacts"] and not (v["meas_scripts"] or v["src_py"]))
print(f"newly-unloadable runs                                     : {len(runs)}")
print(f"named in a CHECKPOINT-LOADING script (meas/*.py or src/*) : {n_loader}")
print(f"named in ANY script (meas/*.py or src/*)                  : {n_script}")
print(f"named ONLY in committed artifacts (*.json/*.md)           : {n_artifact_only}")
print()
print("=== runs named in a CHECKPOINT-LOADING script ===")
for run in sorted(rows):
    v = rows[run]
    if v["loader_scripts"]:
        print(f"  {run}")
        for f in sorted(v["loader_scripts"]):
            print(f"      {f}")
print()
print("=== runs named in src/ (main or agents) at all ===")
for run in sorted(rows):
    v = rows[run]
    if v["src_py"]:
        print(f"  {run:36s} {', '.join(sorted(v['src_py']))}")
print()
print("=== runs named in the ledger's last 1000 lines ===")
for run in sorted(rows, key=lambda r: -rows[r]["ledger_tail_lines"]):
    v = rows[run]
    if v["ledger_tail_lines"]:
        print(f"  {run:36s} {v['ledger_tail_lines']} line(s)")

with open("designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json", "w") as fh:
    json.dump(rows, fh, indent=2)
print("\nwrote designs/research_state/measurements/era_boundary_2026-09-06/reference_refine.json")
