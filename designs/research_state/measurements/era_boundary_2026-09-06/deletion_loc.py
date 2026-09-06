"""Line-mention footprint of the v110 deletion family, split source vs test.

A "line mention" is a grep -c hit, not a deletable line: a `for` loop over the flag
name is one mention and zero deletable lines, while a whole module deleted for one
flag is hundreds. The report pairs these counts with whole-module sizes and says so.

Run:  PYTHONPATH=src python designs/research_state/measurements/era_boundary_2026-09-06/deletion_loc.py
"""
import json
import pathlib
import os
import subprocess

from utils.paths import repo_root

ROOT = repo_root()
SRC = os.path.join(ROOT, "src")

FAMILY = {
    "value_dist (head + 5 flags)": ["value_dist"],
    "value_from_dist": ["value_from_dist", "allow_value_from_dist_change"],
    "value_tail_weight": ["value_tail_weight"],
    "win_prob_coef": ["win_prob_coef"],
    "win_prob_pbrs_*": ["win_prob_pbrs"],
    "use_popart (KEPT, flipped)": ["use_popart", "popart"],
    "draw_penalty (REFUSED, kept)": ["draw_penalty"],
    "score auto (search leaf)": ["\"auto\""],
}

MODULES = [
    "src/agents/training/winprob_pbrs.py",
    "src/agents/model/aux_value_heads.py",
    "src/agents/training/instrumented_ppo/value_terms.py",
    "src/agents/model/value_readouts.py",
    "src/agents/model/model_version/migrations.py",
]


def counts(pat):
    p = subprocess.run(
        ["grep", "-rI", "-c", "--include=*.py", "-F", pat, SRC],
        capture_output=True, text=True)
    src_n = test_n = 0
    for line in p.stdout.splitlines():
        f, _, n = line.rpartition(":")
        n = int(n)
        if not n:
            continue
        if f.endswith("_test.py"):
            test_n += n
        else:
            src_n += n
    return src_n, test_n


print("=== line MENTIONS of each deletion-family token (grep -c, *.py under src/) ===")
hdr = f"{'family':32s} {'source':>8s} {'tests':>8s} {'total':>8s}"
print(hdr)
print("-" * len(hdr))
out = {}
for label, pats in FAMILY.items():
    s = t = 0
    for p in pats:
        a, b = counts(p)
        s += a
        t += b
    print(f"{label:32s} {s:8d} {t:8d} {s + t:8d}")
    out[label] = {"source": s, "tests": t}

print("\n=== whole-module sizes (a module deleted outright) ===")
for m in MODULES:
    path = os.path.join(ROOT, m)
    with open(path) as fh:
        n = sum(1 for _ in fh)
    print(f"{n:6d}  {m}")

with open("designs/research_state/measurements/era_boundary_2026-09-06/deletion_loc.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote designs/research_state/measurements/era_boundary_2026-09-06/deletion_loc.json")
