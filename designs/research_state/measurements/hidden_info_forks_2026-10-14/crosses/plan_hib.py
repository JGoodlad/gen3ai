"""HIB knock-in forks: the PLAN of the §8.4 crosses (primary per lever + v8). REFUSES (exit 3) unless every fork has a
final_model.zip and a pin_history whose every entry is at P_hib reaching >= 18,047,835, and equal init_num_threads.
Writes cells_<lever>.json / cells_v8.json and shas.json. Stdout: the cell counts."""
import hashlib, json, sys
from pathlib import Path
M = Path("/home/goodlad/dev/gen3ai/models"); OUT = Path(sys.argv[1])
PIN = "64a695dc23c946eaaf4e81cae04c4e18dfff787d"; SEEDS = (3101, 3102, 3103); LEVERS = ("moves", "items", "spreads", "species")
PARENT = M / "rb_es_end2_s2001" / "final_model.zip"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
fails, threads, runs = [], set(), {}
for arm in LEVERS + ("ctrl",):
    for s in SEEDS:
        r = M / f"rb_hib_{arm}_s{s}"; fz = r / "final_model.zip"
        if not fz.is_file(): fails.append(f"{r.name}: no final_model.zip"); continue
        md = json.loads((r / "metadata.json").read_text()); ph = md.get("pin_history") or []
        if not ph or any(p.get("git_hash") != PIN for p in ph): fails.append(f"{r.name}: pin_history not all at {PIN[:8]}")
        elif max(int(p.get("last_step") or 0) for p in ph) < 18_047_835: fails.append(f"{r.name}: short of 18,047,835")
        threads.add(md.get("init_num_threads")); runs[(arm, s)] = str(fz)
if not PARENT.is_file(): fails.append("parent final missing")
if len(threads) != 1: fails.append(f"init_num_threads differ: {sorted(map(str, threads))}")
if fails: print("REFUSED: " + "; ".join(fails), file=sys.stderr); sys.exit(3)
shas = {k: sha(v) for k, v in {**{f"{a}_s{s}": p for (a, s), p in runs.items()}, "parent": str(PARENT)}.items()}
counts = {}
for lv in LEVERS:
    cells = [[runs[(lv, i)], runs[("ctrl", j)]] for i in SEEDS for j in SEEDS]
    (OUT / f"cells_{lv}.json").write_text(json.dumps(cells, indent=1)); counts[lv] = len(cells)
for lv in LEVERS:  # HONEST (withheld): seed-paired, lever seed i vs ctrl seed i
    cells = [[runs[(lv, i)], runs[("ctrl", i)]] for i in SEEDS]
    (OUT / f"cells_honest_{lv}.json").write_text(json.dumps(cells, indent=1)); counts[f"honest_{lv}"] = len(cells)
cells = [[runs[("ctrl", j)], str(PARENT)] for j in SEEDS]
(OUT / "cells_v8.json").write_text(json.dumps(cells, indent=1)); counts["v8"] = len(cells)
(OUT / "shas.json").write_text(json.dumps({"shas": shas, "runs": {f"{a}_s{s}": p for (a, s), p in runs.items()}, "parent": str(PARENT)}, indent=1))
print(json.dumps(counts))
