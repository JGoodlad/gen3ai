"""CELLS PLAN for the X26-vs-toy INTERIM head-to-head (RESULTS.md). DESCRIPTIVE and INTERIM, NOT R-T2.
Players = production (the X26 70M checkpoint), opponents = the three toy 100M-cap finals (so a cell's score is PRODUCTION's).
Every reference is LOADED (strict, historical kwargs, CPU) before any cell is written; REFUSES (exit 3) on a missing or
unloadable reference. Writes <out>/cells.json ([player, opponent] zips) and <out>/refs.json. Usage: py.sh plan_cells.py <out>"""
import hashlib, json, sys
from pathlib import Path

M = Path("/home/goodlad/dev/gen3ai/models")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
refs = {"rb_x26_s2601@70M": M / "rb_x26_s2601" / "checkpoints" / "checkpoint_70000008_steps.zip",
        "rb_toy_ent_s4001b@cap": M / "rb_toy_ent_s4001b" / "final_model.zip",
        "rb_toy_ent_s4002@cap": M / "rb_toy_ent_s4002" / "final_model.zip",
        "rb_toy_ent_s4003@cap": M / "rb_toy_ent_s4003" / "final_model.zip"}
fails = [f"{k}: missing ({p})" for k, p in refs.items() if not p.is_file()]
if fails: print("REFUSED: " + "; ".join(fails), file=sys.stderr); sys.exit(3)
from agents.model.snapshot import historical_load_kwargs, load_checkpoint_strict
for k, p in refs.items():
    try: load_checkpoint_strict(str(p), device="cpu", **historical_load_kwargs(str(p))); print("LOADS", k)
    except Exception as e: fails.append(f"{k}: LOAD FAILED {type(e).__name__}: {str(e)[:300]}")
if fails: print("REFUSED: " + "; ".join(fails), file=sys.stderr); sys.exit(3)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


R = {k: {"zip": str(p), "sha256": sha(p)} for k, p in refs.items()}
rows = ["rb_x26_s2601@70M"]; cols = [k for k in refs if k.startswith("rb_toy")]
(OUT / "cells.json").write_text(json.dumps([[R[a]["zip"], R[b]["zip"]] for a in rows for b in cols], indent=1))
(OUT / "refs.json").write_text(json.dumps({"refs": R, "rows": rows, "cols": cols}, indent=1))
print(len(rows) * len(cols), "cells")
