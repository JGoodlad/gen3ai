"""BANK one HIB knock-in fork (read-only). Usage: python3 bank_hib.py <run>
STRICT-BY-READ audit (DECISIONS (f)): every LIVE K9(b) probe's max |d log pi| < 1e-4. A probe is one K9(b) message in
launcher_child.full.log; the UpdateFit dry fixture probe (its dump under /tmp/update_fit_dump_*) is excluded. Also the
per-update table's max_abs_dlogp_judged (every update), the tie stops (crash files) and warnings."""
import json, re, statistics as st, sys, pathlib
run = sys.argv[1]; R = pathlib.Path("/home/goodlad/dev/gen3ai/models") / run
L = (R / "launcher_child.full.log").read_text(errors="replace").splitlines()
md = json.loads((R / "metadata.json").read_text())
print("pin_history", [(p.get("git_hash", "")[:8], p.get("first_step"), p.get("last_step")) for p in md.get("pin_history", [])])
print("final_model.zip", (R / "final_model.zip").is_file())
cr = sorted(p.name for p in (R / "crashes").glob("*.txt")) if (R / "crashes").is_dir() else []
print("crash files", len(cr), cr)
probes, live = [], []
for i, ln in enumerate(L):
    m = re.search(r"K9\(b\)\] BEHAVIOUR-POLICY MISMATCH.*?fp32: max ([0-9.e+-]+) (NOT )?< 0\.0001, excluded_frac ([0-9.]+)", ln)
    if not m or ln.startswith("agents.") or "Training interrupted" in ln: continue
    dry = any("update_fit_dump" in L[j] for j in range(i, min(i + 15, len(L))))
    probes.append((float(m.group(1)), float(m.group(3)), dry))
live = [p for p in probes if not p[2]]
print(f"K9(b) messages: {len(probes)} ({len(probes)-len(live)} dry fixture, excluded) — live max |dlogp| = "
      f"{max([p[0] for p in live], default=0):.3g}; live dlogp violations (>=1e-4): {sum(p[0] >= 1e-4 for p in live)}; "
      f"tie-share > 0.15 messages: {sum(p[1] >= 0.15 for p in live)}")
def col(k):
    return [float(x.split("|")[2]) for x in L if re.match(rf"^\|\s+{k}\s+\|", x)]
for k in ("excluded_frac", "max_abs_dlogp_judged", "train_ms"):
    v = col(k)
    if v: print(f"{k}: n={len(v)} first={v[0]:.4g} median={st.median(v):.4g} max={max(v):.4g}")
print("STRICT-BY-READ:", "PASS" if all(p[0] < 1e-4 for p in live) and all(x < 1e-4 for x in col("max_abs_dlogp_judged")) else "FAIL ⇒ INCONCLUSIVE")
for ln in L:
    if "SELFPLAY EVAL] step" in ln and "Bots" in ln: print(ln[ln.index("step"):ln.index("SelfPlay")].strip())
for ln in L:
    if "ORACLE REVEAL" in ln: print(ln[:170]); break
ck = sorted(p.name for p in (R / "checkpoints").glob("checkpoint_*_steps.zip"))
print("checkpoints", ck[-4:])
