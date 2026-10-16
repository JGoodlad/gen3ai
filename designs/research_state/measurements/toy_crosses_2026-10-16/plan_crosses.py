"""The CELLS PLAN for the two toy crosses (PREREG.md), modelled on road/plan_w4.py.
  R-T5 (equal compute): T-ENT s4001 / s4002 / s4003 at the EQUAL-COMPUTE checkpoint (equal_compute.py) x the 8 EB 15M finals
  R-T6 (toy at cap)   : rb_toy_ent_s4001b / s4002 / s4003 final_model.zip (100M) x the 8 EB 15M finals
Every reference is LOADED (strict, historical kwargs, CPU) before any cell is written; REFUSES (exit 3) on any missing or
unloadable reference. Writes <out>/cells_{rt5,rt6}.json ([player, opponent] lists) and <out>/refs.json
({name: {zip, sha256}}, row / col orders per cross). Usage: plan_crosses.py <out_dir> (via py.sh: the checkout's src)."""
import hashlib, json, sys
from pathlib import Path

M = Path("/home/goodlad/dev/gen3ai/models")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
EB = [f"rb_bf_eb_s{s}" for s in range(2001, 2009)]
# the equal-compute checkpoints, as computed by equal_compute.py and written into PREREG.md before any game
EQ = {"rb_toy_ent_s4001": "checkpoint_48000139_steps.zip", "rb_toy_ent_s4002": "checkpoint_46000198_steps.zip",
      "rb_toy_ent_s4003": "checkpoint_47000187_steps.zip"}
CAP = {"rb_toy_ent_s4001": M / "rb_toy_ent_s4001b" / "final_model.zip", "rb_toy_ent_s4002": M / "rb_toy_ent_s4002" / "final_model.zip",
       "rb_toy_ent_s4003": M / "rb_toy_ent_s4003" / "final_model.zip"}
refs, fails = {}, []
for r in EB: refs[f"{r}@15M"] = M / r / "final_model.zip"
for r, f in EQ.items(): refs[f"{r}@eq"] = M / r / "checkpoints" / f
for r, p in CAP.items(): refs[f"{r}@cap"] = p
for k, p in refs.items():
    if not p.is_file(): fails.append(f"{k}: missing ({p})")
if fails: print("REFUSED: " + "; ".join(fails), file=sys.stderr); sys.exit(3)
from agents.model.snapshot import historical_load_kwargs, load_checkpoint_strict
for k, p in refs.items():
    try: load_checkpoint_strict(str(p), device="cpu", **historical_load_kwargs(str(p)))
    except Exception as e: fails.append(f"{k}: LOAD FAILED {type(e).__name__}: {str(e)[:200]}")
if fails: print("REFUSED: " + "; ".join(fails), file=sys.stderr); sys.exit(3)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


R = {k: {"zip": str(p), "sha256": sha(p)} for k, p in refs.items()}
crosses = {"rt5": ([f"{r}@eq" for r in EQ], [f"{r}@15M" for r in EB]),
           "rt6": ([f"{r}@cap" for r in CAP], [f"{r}@15M" for r in EB])}
for c, (rows, cols) in crosses.items():
    (OUT / f"cells_{c}.json").write_text(json.dumps([[R[a]["zip"], R[b]["zip"]] for a in rows for b in cols], indent=1))
(OUT / "refs.json").write_text(json.dumps({"refs": R, "crosses": {c: {"rows": r, "cols": cl} for c, (r, cl) in crosses.items()}}, indent=1))
print(json.dumps({c: len(r) * len(cl) for c, (r, cl) in crosses.items()}))
