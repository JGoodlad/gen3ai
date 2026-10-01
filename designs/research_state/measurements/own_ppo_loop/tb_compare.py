import sys, glob, hashlib, zipfile
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
def load(d):
    out = {}
    for f in sorted(glob.glob(d + "/tb/events.*")):
        for ev in EventFileLoader(f).Load():
            if not ev.HasField("summary"): continue
            for v in ev.summary.value:
                val = v.simple_value if v.HasField("simple_value") else (v.tensor.float_val[0] if v.tensor.float_val else None)
                out.setdefault(v.tag, []).append((ev.step, val))
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
WALL = ("time/fps", "time/time_elapsed")
def wall(t): return t in WALL or t.endswith("_ms") or t.endswith("_s") or "duration" in t or "/sec" in t or "per_sec" in t
print("tags a/b:", len(a), len(b), "only a:", sorted(set(a)-set(b))[:8], "only b:", sorted(set(b)-set(a))[:8])
same = diff = 0; first = []
for t in sorted(set(a) & set(b)):
    if wall(t): continue
    if a[t] == b[t]: same += 1
    else:
        diff += 1
        if len(first) < 12:
            k = next((i for i,(x,y) in enumerate(zip(a[t], b[t])) if x != y), None)
            first.append((t, len(a[t]), len(b[t]), k, a[t][k] if k is not None else None, b[t][k] if k is not None else None))
print("non-wall tags identical:", same, "differing:", diff)
for r in first: print("  ", r)
for d in sys.argv[1:3]:
    for z in sorted(glob.glob(d + "/checkpoints/*.zip")) + glob.glob(d + "/final_model.zip"):
        with zipfile.ZipFile(z) as zf:
            print(d.split("/")[-1], z.split("/")[-1], hashlib.sha256(zf.read("policy.pth")).hexdigest()[:16])
