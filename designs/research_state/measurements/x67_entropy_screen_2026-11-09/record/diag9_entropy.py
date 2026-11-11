"""X67 PREREG §9 REPORTED diagnostics (decide nothing): entropy / LR / approx_kl trajectories per run.

End entropy = mean of the run's last 20 points of -train/entropy_loss (arm's own points, via run_ref.scalar_events).
A resumed run's tb holds the abandoned attempt's points past the resume step: kept by wall time, the abandoned
attempt's points at steps >= the resumed attempt's first step are dropped (reported as 'superseded').
"""
import json
import statistics as st
import sys
from pathlib import Path

from main.ops.run_ref import scalar_events

MODELS = Path("/home/goodlad/dev/gen3ai/models")
RUNS = [f"rb_x35_x67ent_s{s}" for s in (6701, 6702, 6703)] + [f"rb_x35_c0_s{s}" for s in range(3501, 3509)]
TAGS = ["train/entropy_loss", "train/learning_rate", "train/approx_kl"]
FORK = 280_000_170


def clean(evs):
    evs = sorted(evs, key=lambda e: e.wall_time)
    out, dropped = [], 0
    for e in evs:
        while out and out[-1].step >= e.step:
            out.pop()
            dropped += 1
        out.append(e)
    return out, dropped


def windows(pts, f):
    rows = []
    for lo in range(0, 20, 5):
        a, b = FORK + lo * 1_000_000, FORK + (lo + 5) * 1_000_000
        v = [f(e.value) for e in pts if a < e.step <= b]
        rows.append(round(st.mean(v), 5) if v else None)
    return rows


res = {}
for r in RUNS:
    ev = scalar_events(MODELS / r, TAGS)
    rec = {}
    for t in TAGS:
        pts, dropped = clean(ev.get(t, []))
        rec[t] = {"n": len(pts), "superseded": dropped, "first_step": pts[0].step, "last_step": pts[-1].step}
        if t == "train/entropy_loss":
            last = [-e.value for e in pts[-20:]]
            rec[t].update(end_mean=st.mean(last), end_min=min(last), end_max=max(last),
                          first20_mean=st.mean([-e.value for e in pts[:20]]),
                          win5M=windows(pts, lambda x: -x))
        elif t == "train/learning_rate":
            ch, prev = [], None
            for e in pts:
                if prev is None or abs(e.value - prev) > 1e-12:
                    ch.append([e.step, e.value])
                    prev = e.value
            rec[t].update(median=st.median([e.value for e in pts]), changes=ch[:10], n_changes=len(ch) - 1)
        else:
            rec[t].update(median=st.median([e.value for e in pts]), win5M=windows(pts, lambda x: x),
                          last20_mean=st.mean([e.value for e in pts[-20:]]))
    res[r] = rec

x67 = [res[r]["train/entropy_loss"]["end_mean"] for r in RUNS[:3]]
c0 = {r: res[r]["train/entropy_loss"]["end_mean"] for r in RUNS[3:]}
summary = {"x67_end_entropy_mean": st.mean(x67), "x67_end_entropy": x67, "c0_end_entropy": c0,
           "c0_end_entropy_mean": st.mean(c0.values()), "c0_lowest": min(c0.values()),
           "ENT_UNMOVED": not (st.mean(x67) < min(c0.values()))}
json.dump({"runs": res, "summary": summary}, open(sys.argv[1], "w"), indent=1)
for r in RUNS:
    e, l, k = res[r]["train/entropy_loss"], res[r]["train/learning_rate"], res[r]["train/approx_kl"]
    print(f"{r:22s} H_end {e['end_mean']:.4f} [{e['end_min']:.3f},{e['end_max']:.3f}] H_first20 {e['first20_mean']:.4f} "
          f"H_5M {e['win5M']} | lr med {l['median']:.4e} changes {l['n_changes']} {[(s, f'{v:.3e}') for s, v in l['changes']]} "
          f"| kl med {k['median']:.4f} 5M {k['win5M']} | n {e['n']} sup {e['superseded']}")
print(json.dumps(summary, indent=1))
