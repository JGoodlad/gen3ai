"""END-STATE closing test (design_endstate_closing_test.md §3.1): the PLAN of the 8 x 8 cross.

Rows = end-state seeds rb_es_end2_s2001..s2008 (P_end 62085750), columns = production seeds rb_es_prod_s2001..s2008
(P_prod 95d014fa); each run's 15M final_model.zip. REFUSES (exit 3) unless every one of the sixteen runs has a
final_model.zip, a pin_history of exactly ONE entry at its arm's pin covering >= 15,000,000 steps, and equal
init_num_threads. Writes --out (the [player .zip, opponent .zip] list `main.h2h play-many --cells` reads) and
--shas (row / column sha256 lists for the reader). Stdout: the cell count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MODELS = Path("/home/goodlad/dev/gen3ai/models")
SEEDS = (2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008)
ARMS = {"end": ("rb_es_end2_s{}", "62085750baab1ca275d810fbe0bb5a960904e183"),
        "prod": ("rb_es_prod_s{}", "95d014faeeb4d699703264a19094afee6f4eafc2")}
MIN_STEPS = 15_000_000


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shas", required=True)
    a = ap.parse_args()
    fails, finals, threads = [], {"end": [], "prod": []}, set()
    for arm, (fmt, pin) in ARMS.items():
        for s in SEEDS:
            run = MODELS / fmt.format(s)
            fz = run / "final_model.zip"
            if not fz.is_file():
                fails.append(f"{run.name}: no final_model.zip")
                continue
            md = json.loads((run / "metadata.json").read_text())
            ph = md.get("pin_history") or []
            if len(ph) != 1 or ph[0].get("git_hash") != pin:
                fails.append(f"{run.name}: pin_history {[(p.get('git_hash'), p.get('first_step'), p.get('last_step')) for p in ph]} != [{pin[:8]}]")
            elif int(ph[0].get("last_step") or 0) < MIN_STEPS:
                fails.append(f"{run.name}: last_step {ph[0].get('last_step')} < {MIN_STEPS}")
            threads.add(md.get("init_num_threads"))
            finals[arm].append(str(fz))
    if len(threads) != 1:
        fails.append(f"init_num_threads differs across the sixteen: {sorted(map(str, threads))}")
    if fails:
        print("REFUSED: " + "; ".join(fails), file=sys.stderr)
        return 3
    cells = [[r, c] for r in finals["end"] for c in finals["prod"]]
    Path(a.out).write_text(json.dumps(cells, indent=1))
    Path(a.shas).write_text(json.dumps({"rows": [sha(Path(p)) for p in finals["end"]],
                                        "cols": [sha(Path(p)) for p in finals["prod"]],
                                        "row_runs": finals["end"], "col_runs": finals["prod"]}, indent=1))
    print(len(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
