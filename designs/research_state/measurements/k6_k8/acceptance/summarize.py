"""Summarize the K8 acceptance units (A = main pre-regions, B = K8): unbracketed update, bracketed
phases, host scalar reads, and the profiled diag-skipped epochs' compiled share."""
import glob, json, statistics
rows = {}
for u in ("A1", "B1", "A2", "B2"):
    fs = sorted(glob.glob(f"/home/goodlad/gen3ai_archive/k6_k8/accept/{u}/*/time_result.json"))
    an = sorted(glob.glob(f"/home/goodlad/gen3ai_archive/k6_k8/accept/{u}/*/time_analysis.json"))
    if not fs:
        continue
    d = json.load(open(fs[-1]))
    t = json.load(open(an[-1]))["profiles"][0]["totals"] if an else {}
    rows[u] = dict(status=d.get("status"), regions=d.get("compiled_regions"),
                   unbr=d["unbracketed"]["train_ms"] / 1000, br=d["bracketed"]["train_ms"] / 1000,
                   reads=d["bracketed"].get("scalar_reads"),
                   phases={k: round(v, 2) for k, v in d["bracketed"]["phases_s"].items() if v > 0.05},
                   share=t.get("compiled_share_of_train_wall"), kshare=t.get("compiled_kernel_share"),
                   idle=t.get("gpu_idle_ms"), span=t.get("train_span_ms"),
                   loss=d["unbracketed"]["work"]["train/loss"])
for u, r in rows.items():
    print(u, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
A = [r["unbr"] for u, r in rows.items() if u.startswith("A")]
B = [r["unbr"] for u, r in rows.items() if u.startswith("B")]
if A and B:
    print(f"unbracketed update: A {statistics.mean(A):.2f} s (n={len(A)}) -> B {statistics.mean(B):.2f} s "
          f"(n={len(B)}): ratio {statistics.mean(B) / statistics.mean(A):.3f}")
