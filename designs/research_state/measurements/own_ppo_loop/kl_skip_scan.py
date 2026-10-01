import sys, glob, json, os
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
TAGS = {"time/fps", "train/approx_kl", "train/learning_rate", "eval/win_rate_mean"}
def scan(run):
    steps = {t: [] for t in TAGS}
    files = [f for f in sorted(glob.glob(run + "/tb/events.*")) if ".inherited." not in f]
    for f in files:
        try:
            for ev in EventFileLoader(f).Load():
                if not ev.HasField("summary"): continue
                for v in ev.summary.value:
                    if v.tag in steps:
                        val = v.simple_value if v.HasField("simple_value") else (v.tensor.float_val[0] if v.tensor.float_val else None)
                        steps[v.tag].append((ev.step, val))
        except Exception as e:
            print("  read error", f, e)
    fps = {s for s, _ in steps["time/fps"]}
    kl = steps["train/approx_kl"]; ev = {s for s, _ in steps["eval/win_rate_mean"]}
    off = [s for s, _ in kl if s not in fps]
    at_eval = [s for s in off if s in ev]
    cfg = {}
    for p in (run + "/model_config.json",):
        try: cfg = json.load(open(p))
        except Exception: pass
    meta = {}
    try: meta = json.load(open(run + "/metadata.json"))
    except Exception: pass
    cmd = json.dumps(meta.get("original_command", ""))
    flags = {k: (k in cmd) for k in ("--fork-lr-freeze", "--eval-freq", "--env-core rust", "--no-adaptive")}
    n = len(kl)
    print(f"{os.path.basename(run):40s} updates={n:5d} skipped={len(off):4d} ({100*len(off)/max(n,1):5.2f}%) at_eval_step={len(at_eval):4d} eval_cycles={len(ev):4d} dumps={len(fps):5d} files={len(files)} {flags}")
for r in sys.argv[1:]:
    scan(r)

def lr_check(run):
    pts = {}
    for f in [f for f in sorted(glob.glob(run + "/tb/events.*")) if ".inherited." not in f]:
        for ev in EventFileLoader(f).Load():
            if not ev.HasField("summary"): continue
            for v in ev.summary.value:
                if v.tag in ("time/fps", "train/learning_rate", "eval/win_rate_mean"):
                    val = v.simple_value if v.HasField("simple_value") else (v.tensor.float_val[0] if v.tensor.float_val else None)
                    pts.setdefault(v.tag, []).append((ev.step, val))
    fps = {s for s, _ in pts.get("time/fps", [])}
    lr = pts.get("train/learning_rate", [])
    # order of writes is the order of updates; pair update k with k+1
    sk_same = sk_n = ok_same = ok_n = 0
    for (s0, v0), (s1, v1) in zip(lr, lr[1:]):
        skipped = s0 not in fps   # update k's stats were flushed by an eval => the controller skipped after update k
        same = (v0 == v1)
        if skipped: sk_n += 1; sk_same += same
        else: ok_n += 1; ok_same += same
    print(f"  LR unchanged after a skipped update: {sk_same}/{sk_n}; after a normal update: {ok_same}/{ok_n}")
if os.environ.get("LRCHECK"):
    for r in sys.argv[1:]: lr_check(r)
