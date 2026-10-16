"""W4 (the ONE h2h window before X26; design_x26_readiness.md §4.1): the CELLS PLAN for E7's cross + the toy crosses.
  E7    : rb_xs_s2701 / s2702 (final_model.zip, 5M) x the 8 EB 5M models          2 x 8, 500 pairs
  R-T1  : T-ENT s4001-s4003 at 15M (first checkpoint >= 15,000,000) x EB 15M finals 3 x 8, 500 pairs
  R-T1b : T-ENT at 5M (first checkpoint >= 5,000,000) x EB 5M                       3 x 8, 250 pairs (descriptive)
EB 5M (F30): s2001-s2004 final_model_stage1_5M.zip; s2005-s2008 checkpoints/checkpoint_5000*_steps.zip. EB 15M = final_model.zip.
Every reference is LOADED (strict, historical kwargs, CPU) before any cell is written; REFUSES (exit 3) on any missing
or unloadable reference. Writes <out>/cells_{e7,rt1,rt1b}.json ([player, opponent] lists) and <out>/refs.json
({name: {zip, sha256}}, row / col orders per cross). Run with PYTHONPATH=<P_X26 checkout>/src."""
import glob, hashlib, json, re, sys
from pathlib import Path
M = Path("/home/goodlad/dev/gen3ai/models")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
EB = [f"rb_bf_eb_s{s}" for s in range(2001, 2009)]
XS = ["rb_xs_s2701", "rb_xs_s2702"]
TOY = ["rb_toy_ent_s4001", "rb_toy_ent_s4002", "rb_toy_ent_s4003"]
fails = []
def first_ckpt_at(run, n):
    c = []
    for p in glob.glob(str(M / run / "checkpoints" / "checkpoint_*_steps.zip")):
        m = re.search(r"checkpoint_(\d+)_steps\.zip$", p)
        if m and int(m.group(1)) >= n: c.append((int(m.group(1)), p))
    return min(c)[1] if c else None
def eb5(run):
    p = M / run / "final_model_stage1_5M.zip"
    if p.is_file(): return str(p)
    c = sorted(glob.glob(str(M / run / "checkpoints" / "checkpoint_5000*_steps.zip")))
    return c[0] if c else None
def need(name, p):
    if not p or not Path(p).is_file(): fails.append(f"{name}: missing ({p})"); return None
    return p
refs = {}
for r in EB:
    refs[f"{r}@5M"] = need(f"{r}@5M", eb5(r)); refs[f"{r}@15M"] = need(f"{r}@15M", str(M / r / "final_model.zip"))
for r in XS: refs[f"{r}@5M"] = need(f"{r}@5M", str(M / r / "final_model.zip"))
for r in TOY:
    refs[f"{r}@15M"] = need(f"{r}@15M", first_ckpt_at(r, 15_000_000)); refs[f"{r}@5M"] = need(f"{r}@5M", first_ckpt_at(r, 5_000_000))
if fails: print("REFUSED: " + "; ".join(fails), file=sys.stderr); sys.exit(3)
from agents.model.snapshot import historical_load_kwargs, load_checkpoint_strict
for k, p in refs.items():
    try: load_checkpoint_strict(p, device="cpu", **historical_load_kwargs(p))
    except Exception as e: fails.append(f"{k}: LOAD FAILED {type(e).__name__}: {str(e)[:200]}")
if fails: print("REFUSED: " + "; ".join(fails), file=sys.stderr); sys.exit(3)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
R = {k: {"zip": p, "sha256": sha(p)} for k, p in refs.items()}
crosses = {"e7": ([f"{r}@5M" for r in XS], [f"{r}@5M" for r in EB]),
           "rt1": ([f"{r}@15M" for r in TOY], [f"{r}@15M" for r in EB]),
           "rt1b": ([f"{r}@5M" for r in TOY], [f"{r}@5M" for r in EB])}
for c, (rows, cols) in crosses.items():
    (OUT / f"cells_{c}.json").write_text(json.dumps([[R[a]["zip"], R[b]["zip"]] for a in rows for b in cols], indent=1))
(OUT / "refs.json").write_text(json.dumps({"refs": R, "crosses": {c: {"rows": r, "cols": cl} for c, (r, cl) in crosses.items()}}, indent=1))
print(json.dumps({c: len(r) * len(cl) for c, (r, cl) in crosses.items()}))
