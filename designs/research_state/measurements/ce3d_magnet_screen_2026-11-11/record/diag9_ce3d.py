"""CE-3d PREREG §8 labels (ENT-LOW / ENT-HIGH) and §9 REPORTED diagnostics (they decide nothing).

Per swap fork: entropy / learning-rate / approx_kl trajectories and the magnet's own meters, read from the run's
TensorBoard through main.ops.run_ref.scalar_events (the arm's own points; a fork's inherited parent points are
excluded there). End entropy = the mean of the run's last 20 points of -train/entropy_loss (X67's definition,
PREREG §9). The C0 bounds are read from X67's banked record, never retyped.

usage: diag9_ce3d.py <checkout> <out.json>      (read-only on models/)
"""
import json
import statistics as st
import sys
from pathlib import Path

from main.ops.run_ref import scalar_events

CHECKOUT = Path(sys.argv[1])
MODELS = Path("/home/goodlad/dev/gen3ai/models")
RUNS = [f"rb_x35_ce3dswap_s{s}" for s in (7321, 7322, 7323)]
TAGS = ["train/entropy_loss", "train/learning_rate", "train/approx_kl", "train/magnet_kl",
        "magnet/age_steps", "magnet/refreshes", "magnet/forward_ms", "time/fps"]
FORK = 280_000_170
REFRESH = 2_000_000
EPS = 1e-9  # rule 8: a difference within 1e-9 nats of a bound counts as overlap

X67_REC = json.load(open(
    CHECKOUT / "designs/research_state/measurements/x67_entropy_screen_2026-11-09/record/diag9_entropy.json"))
C0_END = X67_REC["summary"]["c0_end_entropy"]
C0_LOW, C0_HIGH = min(C0_END.values()), max(C0_END.values())


def clean(evs):
    """Drop an abandoned attempt's points past a resume step (kept by wall time). Returns (points, n dropped)."""
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
        if not pts:
            rec[t] = {"n": 0}
            continue
        vals = [e.value for e in pts]
        rec[t] = {"n": len(pts), "superseded": dropped, "first_step": pts[0].step, "last_step": pts[-1].step}
        if t == "train/entropy_loss":
            last = [-e.value for e in pts[-20:]]
            rec[t].update(end_mean=st.mean(last), end_min=min(last), end_max=max(last),
                          first20_mean=st.mean([-e.value for e in pts[:20]]), win5M=windows(pts, lambda x: -x),
                          lowest=min(-v for v in vals), highest=max(-v for v in vals))
        elif t == "train/learning_rate":
            ch, prev = [], None
            for e in pts:
                if prev is None or abs(e.value - prev) > 1e-12:
                    ch.append([e.step, e.value])
                    prev = e.value
            rec[t].update(median=st.median(vals), changes=ch[:10], n_changes=len(ch) - 1, last=vals[-1])
        elif t == "train/magnet_kl":
            # the saw-tooth, per refresh cycle. A fill logs magnet/* at the step its rollout ended on, and the
            # update that trains on that rollout logs train/magnet_kl at the NEXT fill's step; the refresh lands
            # between them. So the update logged at step s trained against the magnet of the last magnet/* point
            # STRICTLY before s (checked on the raw series: the counter's first k and the peak share a step, and
            # the floor is the next train point).
            ref, _ = clean(ev.get("magnet/refreshes", []))
            cyc = {}
            for e in pts:
                k = max((x.value for x in ref if x.step < e.step), default=0.0)
                cyc.setdefault(int(k), []).append(e.value)
            rec[t].update(median=st.median(vals), mean=st.mean(vals), min=min(vals), max=max(vals),
                          last20_mean=st.mean(vals[-20:]), win5M=windows(pts, lambda x: x),
                          per_refresh_cycle={k: {"n": len(v), "first": round(v[0], 5), "last": round(v[-1], 5),
                                                 "max": round(max(v), 5), "mean": round(st.mean(v), 5),
                                                 "monotone_rise": all(b > a for a, b in zip(v, v[1:]))}
                                             for k, v in sorted(cyc.items())},
                          floor_after_refresh=[round(v[0], 5) for k, v in sorted(cyc.items()) if k > 0],
                          peak_before_refresh=[round(v[-1], 5) for k, v in sorted(cyc.items())
                                               if k < max(cyc)])
        elif t == "magnet/refreshes":
            rec[t].update(max=max(vals), last=vals[-1], points_above_0=sum(v > 0 for v in vals),
                          first_step_at_each=[[int(v), next(e.step for e in pts if e.value == v)]
                                              for v in sorted(set(vals))])
        else:
            rec[t].update(median=st.median(vals), mean=st.mean(vals), min=min(vals), max=max(vals),
                          last20_mean=st.mean(vals[-20:]), win5M=windows(pts, lambda x: x))
    res[r] = rec

end = {r: res[r]["train/entropy_loss"]["end_mean"] for r in RUNS}
hi, lo = max(end.values()), min(end.values())
ent_low = hi < C0_LOW - EPS
ent_high = lo > C0_HIGH + EPS
near = [name for name, d in (("highest swap vs lowest C0", hi - C0_LOW), ("lowest swap vs highest C0", lo - C0_HIGH))
        if abs(d) <= EPS]
summary = {
    "swap_end_entropy": end, "swap_end_entropy_mean": st.mean(end.values()),
    "swap_highest": hi, "swap_lowest": lo,
    "c0_lowest": C0_LOW, "c0_highest": C0_HIGH, "c0_end_entropy_mean": X67_REC["summary"]["c0_end_entropy_mean"],
    "x67_end_entropy": X67_REC["summary"]["x67_end_entropy"],
    "x67_end_entropy_mean": X67_REC["summary"]["x67_end_entropy_mean"],
    "ENT_LOW": ent_low, "ENT_HIGH": ent_high, "within_1e-9_of_a_bound": near,
    "label": "ENT-LOW" if ent_low else "ENT-HIGH" if ent_high else "neither (the ranges overlap)",
}
json.dump({"runs": res, "summary": summary}, open(sys.argv[2], "w"), indent=1)
for r in RUNS:
    e, l, k = res[r]["train/entropy_loss"], res[r]["train/learning_rate"], res[r]["train/approx_kl"]
    m, a, f, n = (res[r]["train/magnet_kl"], res[r]["magnet/age_steps"], res[r]["magnet/forward_ms"],
                  res[r]["magnet/refreshes"])
    print(f"{r}\n  H_end {e['end_mean']:.4f} [{e['end_min']:.3f},{e['end_max']:.3f}] H_first20 {e['first20_mean']:.4f} "
          f"H_5M {e['win5M']} H_range [{e['lowest']:.3f},{e['highest']:.3f}] n {e['n']} sup {e['superseded']}\n"
          f"  lr med {l['median']:.4e} last {l['last']:.4e} changes {l['n_changes']} "
          f"{[(s, f'{v:.3e}') for s, v in l['changes']]}\n"
          f"  approx_kl med {k['median']:.4f} 5M {k['win5M']} last20 {k['last20_mean']:.4f}\n"
          f"  magnet_kl med {m['median']:.4f} mean {m['mean']:.4f} [{m['min']:.4f},{m['max']:.4f}] 5M {m['win5M']} "
          f"last20 {m['last20_mean']:.4f}\n"
          f"  magnet_kl per magnet (refreshes so far: n first last max rose-every-update): "
          f"{[(c, v['n'], v['first'], v['last'], v['max'], v['monotone_rise']) for c, v in m['per_refresh_cycle'].items()]}\n"
          f"  magnet_kl floor (first update after each refresh) {m['floor_after_refresh']}\n"
          f"  magnet_kl peak (last update before each refresh) {m['peak_before_refresh']}\n"
          f"  magnet/refreshes max {n['max']:.0f} last {n['last']:.0f} points>0 {n['points_above_0']}/{n['n']} "
          f"first step at each {n['first_step_at_each']}\n"
          f"  magnet/age_steps max {a['max']:.0f} med {a['median']:.0f} | magnet/forward_ms med {f['median']:.1f} "
          f"mean {f['mean']:.1f} [{f['min']:.1f},{f['max']:.1f}]"
          + (f" | time/fps med {res[r]['time/fps']['median']:.0f}" if res[r]['time/fps'].get('n') else " | time/fps: no tag"))
print(json.dumps(summary, indent=1))
