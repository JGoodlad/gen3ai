"""Driver for the three-step proof (README). Each step is a child process under `scripts/ops/mem_cap.sh`, with
PYTHONPATH set to the tree it runs (NEW = this checkout's src/, OLD = the parent commit's src/ extracted with
`git archive 7bed4347 src data designs/production_config.json designs/baselines.json`).

    python run_proof.py --phase {k9b,A,rebuild,diff,B,C} --old <old tree root> --parent <dir with the parent's
        learner_golden.json + both buffers> --scratch <scratch dir>

Run from the NEW tree's root. Outputs land in ../out/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
PY = sys.executable
HOLD = OUT / "parent_species_usage.json"
ARMS = {"blob": "learner_golden_buffer.npz", "fixed_mass": "learner_golden_buffer_fixed_mass.npz"}
REASON = ("F-X5-47 gen3_smogon_species_usage_weighted_v1: the Smogon species-usage marginal is the rating-"
          "weighted set total W (sum Abilities), not the unweighted Raw count - every moved element is the "
          "marginal's (proof designs/research_state/measurements/species_usage_golden_2026-10-04/)")


def run(argv: list, tree: Path, out: Path | None = None, ok=(0,)) -> int:
    env = dict(os.environ, PYTHONPATH=str(tree / "src"))
    cmd = ["scripts/ops/mem_cap.sh", "16", PY, *map(str, argv)]
    print("$", " ".join(cmd), f"(PYTHONPATH={env['PYTHONPATH']})", flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    body = "\n".join(ln for ln in r.stdout.splitlines() if not ln.startswith("[mem_cap]"))
    if out is not None:
        out.write_text(body + "\n")
    print(body[-3000:], r.stderr[-1500:] if r.returncode not in ok else "", flush=True)
    if r.returncode not in ok:
        raise SystemExit(f"step failed ({r.returncode}): {argv}")
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("k9b", "A", "rebuild", "diff", "B", "C"))
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--arms", default="blob,fixed_mass")
    a = ap.parse_args()
    new = Path.cwd()
    train = new / "src/agents/training"
    arms = a.arms.split(",")
    OUT.mkdir(exist_ok=True)
    for arm in arms:
        pbuf = a.parent / ARMS[arm]
        if a.phase == "k9b":       # on the COMMITTED (parent) buffers, before any rebuild
            run([HERE / "k9b_read.py", "--arm", arm], new, OUT / f"k9b_{arm}_committed.txt")
            run([HERE / "k9b_read.py", "--arm", arm, "--hold", HOLD], new,
                OUT / f"k9b_{arm}_committed_parentusage.txt")
        elif a.phase == "A":
            res = OUT / f"A_{arm}_new_parentusage.json"
            run([HERE / "proof.py", "--label", f"A_{arm}_new_parentusage", "--arm", arm, "--buffer", pbuf,
                 "--hold", HOLD, "--out", res], new)
            run([HERE / "compare.py", res, a.parent / "learner_golden.json", "--golden-arm", arm], new,
                OUT / f"A_{arm}_vs_recorded.txt", ok=(0, 1))
        elif a.phase == "rebuild":
            run(["-m", "agents.training.learner_golden", "rebuild-buffer", "--arm", arm, "--reason", REASON], new,
                OUT / f"rebuild_{arm}.txt")
        elif a.phase == "diff":
            run([HERE / "buffer_diff.py", pbuf, train / ARMS[arm]], new, OUT / f"buffer_diff_{arm}.json")
        elif a.phase == "B":
            buf = train / ARMS[arm]
            for label, tree, hold in ((f"B_{arm}_old_tree", a.old, None),
                                      (f"B_{arm}_new_parentusage", new, HOLD),
                                      (f"B_{arm}_new", new, None)):
                run([HERE / "proof.py", "--label", label, "--arm", arm, "--buffer", buf, "--out",
                     OUT / f"{label}.json", *(["--hold", hold] if hold else [])], tree)
            run([HERE / "compare.py", OUT / f"B_{arm}_old_tree.json", OUT / f"B_{arm}_new_parentusage.json"], new,
                OUT / f"B_{arm}_oldtree_vs_new_parentusage.txt", ok=(0, 1))
            run([HERE / "compare.py", OUT / f"B_{arm}_old_tree.json", OUT / f"B_{arm}_new.json"], new,
                OUT / f"B_{arm}_oldtree_vs_new.txt", ok=(0, 1))
        elif a.phase == "C":
            cbuf = a.scratch / f"C_{arm}_parentusage_rebuild.npz"
            run([HERE / "rebuild_with_usage.py", "--arm", arm, "--hold", HOLD, "--out", cbuf], new)
            run([HERE / "buffer_diff.py", pbuf, cbuf], new, OUT / f"C_{arm}_parentusage_rebuild_vs_parent.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
