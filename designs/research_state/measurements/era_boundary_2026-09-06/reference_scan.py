"""Which of the newly-unloadable runs are still REFERENCED by tooling that loads checkpoints?

Three surfaces, each scanned literally for the run's directory name:
  1. designs/research_state/measurements/**  (scripts + committed artifacts)
  2. src/main/*.py  and src/main/**/*.py     (entry points)
  3. the ledger's LAST 1000 lines            (the live campaign)

For the ledger surface the hit is additionally classified by whether the run name appears
on a line carrying a MODEL-LOADING role token: --baseline / --control / --model /
--distill-teacher / --stable-opponents / --exploiter / --ref / teacher / opponent.

Run:  PYTHONPATH=src python designs/research_state/measurements/era_boundary_2026-09-06/reference_scan.py
"""
import json
import pathlib
import os
import re
import subprocess

from utils.paths import repo_root

ROOT = repo_root()
# This census's own artifacts name every run by construction; excluding them is
# what stops the scan from scoring itself as a dependency.
SELF_DIR = "era_boundary_2026-09-06"
LOADABILITY = "designs/research_state/measurements/era_boundary_2026-09-06/loadability.json"

ROLE_TOKENS = (
    "--baseline", "--control", "--model", "--distill-teacher", "--stable-opponents",
    "--exploiter", "--ref", "--refs", "--teacher", "teacher", "opponent", "--parent",
    "--distill-anchor-parent", "--warmstart-consensus", "--win-prob-pbrs-source",
)

with open(LOADABILITY) as fh:
    runs = [r["run"] for r in json.load(fh)["loadable_today"]]

ledger = os.path.join(ROOT, "designs/research_state/ledger.md")
with open(ledger) as fh:
    ledger_tail = fh.readlines()[-1000:]

meas_dir = os.path.join(ROOT, "designs/research_state/measurements")
main_dir = os.path.join(ROOT, "src/main")


def grep_count(pattern, path, include=None):
    cmd = ["grep", "-rIl", f"--exclude-dir={SELF_DIR}", "-F", pattern, path]
    if include:
        cmd = ["grep", "-rIl", f"--exclude-dir={SELF_DIR}", f"--include={include}", "-F", pattern, path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return [l for l in p.stdout.splitlines() if l]


out = {}
for run in runs:
    meas = grep_count(run, meas_dir)
    main_py = grep_count(run, main_dir, include="*.py")
    led = [l for l in ledger_tail if run in l]
    led_role = [l for l in led if any(t in l for t in ROLE_TOKENS)]
    out[run] = {
        "measurements_files": len(meas),
        "measurements_sample": [os.path.relpath(p, ROOT) for p in meas[:4]],
        "src_main_py_files": len(main_py),
        "src_main_py": [os.path.relpath(p, ROOT) for p in main_py],
        "ledger_tail_lines": len(led),
        "ledger_tail_role_lines": len(led_role),
    }

referenced = {k: v for k, v in out.items() if v["measurements_files"] or v["src_main_py_files"] or v["ledger_tail_lines"]}
role_referenced = {k: v for k, v in out.items() if v["src_main_py_files"] or v["ledger_tail_role_lines"] or v["measurements_files"]}

print(f"newly-unloadable runs scanned            : {len(runs)}")
print(f"referenced ANYWHERE (meas|src/main|tail) : {len(referenced)}")
print(f"referenced in measurements/              : {sum(1 for v in out.values() if v['measurements_files'])}")
print(f"referenced in src/main/**.py             : {sum(1 for v in out.values() if v['src_main_py_files'])}")
print(f"named in the ledger's last 1000 lines    : {sum(1 for v in out.values() if v['ledger_tail_lines'])}")
print(f"  ... on a MODEL-ROLE line               : {sum(1 for v in out.values() if v['ledger_tail_role_lines'])}")
print()
hdr = f"{'run':38s} {'meas':>5s} {'main':>5s} {'tail':>5s} {'role':>5s}"
print(hdr)
print("-" * len(hdr))
for run in sorted(referenced, key=lambda r: (-out[r]["measurements_files"], r)):
    v = out[run]
    print(f"{run:38s} {v['measurements_files']:5d} {v['src_main_py_files']:5d} "
          f"{v['ledger_tail_lines']:5d} {v['ledger_tail_role_lines']:5d}")

unreferenced = [r for r in runs if r not in referenced]
print(f"\nUNREFERENCED on all three surfaces: {len(unreferenced)}")
for r in sorted(unreferenced):
    print("   ", r)

with open("designs/research_state/measurements/era_boundary_2026-09-06/reference_scan.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote designs/research_state/measurements/era_boundary_2026-09-06/reference_scan.json")
