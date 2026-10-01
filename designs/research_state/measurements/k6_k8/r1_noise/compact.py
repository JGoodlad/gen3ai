"""noise_cuda.jsonl -> a compact results file (summary + the top-10 parameters by the gate's reading,
per state) and the README's markdown table."""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
out = []
for r in rows:
    top = sorted(r["per_param"].items(), key=lambda kv: -(kv[1].get("gate") or 0))[:10]
    out.append({k: r[k] for k in ("state", "batch", "B", "device", "torch", "note", "loss", "summary",
                                   "c64_over_e64", "secs")} | {"top10_by_gate": dict(top)})
json.dump(out, open(sys.argv[2], "w"), indent=1)
def short(s):
    return s.replace("ai_v14_", "").replace("/checkpoints/checkpoint_", "@").replace("_steps.zip", "").replace("/final_model.zip", " final")
print("| weight state | regime | gate (compiled vs CUDA eager) | worst parameter | compiled vs fp64 | CUDA eager vs fp64 | CPU eager vs fp64 | params outside 2x the eager envelope |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    s, pp = r["summary"], r["per_param"]
    out_n = sum(1 for v in pp.values() if v.get("c64") is not None and (v.get("gate") or 0) > 1e-4
                and v["c64"] > 2 * max(v.get("e64") or 0, v.get("x64") or 0))
    reg = "fresh" if r["state"].startswith("fresh") else "trained"
    w = s["gate"]["argmax"].replace("features_extractor.", "")
    wv = pp[s["gate"]["argmax"]]
    print(f"| {short(r['state'])} | {reg} | {s['gate']['max']:.2e} | `{w}` | {wv['c64']:.1e} | {wv['e64']:.1e} | {wv['x64']:.1e} | {out_n} |")
