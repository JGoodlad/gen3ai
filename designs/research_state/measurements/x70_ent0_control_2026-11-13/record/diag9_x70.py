"""X70 PREREG §8 labels and §9 REPORTED diagnostics (they decide nothing).

CE-3d's ``record/diag9_ce3d.py`` copied into this record and pointed at the four X70 forks, as §9 registers ("the
reading unit copies CE-3d's record/diag9_ce3d.py into this directory's record for it"). Per fork: the entropy, the
learning rate, approx_kl and time/fps, read from the run's TensorBoard through main.ops.run_ref.scalar_events (the
arm's own points; a fork's inherited parent points are excluded there). End entropy = the mean of the run's last 20
points of -train/entropy_loss (X67's and CE-3d's definition, PREREG §8). The swap forks' and C0's and X67's values are
read from the banked records, never retyped. The entropy labels are the registered rule's (``screen_rule.labels``);
LR-MOVED compares main.dose's printed lr_median (passed on the command line as printed) with §8's two bounds.

usage: diag9_x70.py <checkout> <out.json> <lr_median s7331> <s7332> <s7333> <s7334>     (read-only on models/)
"""
import json
import statistics as st
import sys
from pathlib import Path

from main.ops.run_ref import scalar_events

CHECKOUT = Path(sys.argv[1])
MEAS = CHECKOUT / "designs/research_state/measurements"
sys.path.insert(0, str(MEAS / "x70_ent0_control_2026-11-13"))
import screen_rule  # noqa: E402  (the REGISTERED rule: labels())

MODELS = Path("/home/goodlad/dev/gen3ai/models")
SEEDS = (7331, 7332, 7333, 7334)
RUNS = [f"rb_x35_x70ent0_s{s}" for s in SEEDS]
TAGS = ["train/entropy_loss", "train/learning_rate", "train/approx_kl", "train/magnet_kl", "magnet/refreshes",
        "time/fps"]
FORK = 280_000_170
LR_LOW, LR_HIGH = 0.0000905, 0.0001105  # PREREG §8 LR-MOVED: more than 10 % from the swap forks' and C0's 0.0001005
LR_DOSE = [float(x) for x in sys.argv[3:7]]

CE3D = json.load(open(MEAS / "ce3d_magnet_screen_2026-11-11/record/diag9_ce3d.json"))
X67_REC = json.load(open(MEAS / "x67_entropy_screen_2026-11-09/record/diag9_entropy.json"))
SWAP_END = CE3D["summary"]["swap_end_entropy"]
SWAP_LOW, SWAP_HIGH = min(SWAP_END.values()), max(SWAP_END.values())
C0_END = X67_REC["summary"]["c0_end_entropy"]


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
            below_low = [e.step for e in pts if -e.value < SWAP_LOW]
            below_high = [e.step for e in pts if -e.value < SWAP_HIGH]
            rec[t].update(end_mean=st.mean(last), end_min=min(last), end_max=max(last),
                          first20_mean=st.mean([-e.value for e in pts[:20]]), win5M=windows(pts, lambda x: -x),
                          lowest=min(-v for v in vals), highest=max(-v for v in vals),
                          # §9 "where the entropy went": the first logged point below the swap's END RANGE (its lowest
                          # end entropy), and below the range's top; how many of the run's points sit below each.
                          first_step_below_swap_lowest=below_low[0] if below_low else None,
                          n_points_below_swap_lowest=len(below_low),
                          first_step_below_swap_highest=below_high[0] if below_high else None,
                          n_points_below_swap_highest=len(below_high))
        elif t == "train/learning_rate":
            ch, prev = [], None
            for e in pts:
                if prev is None or abs(e.value - prev) > 1e-12:
                    ch.append([e.step, e.value])
                    prev = e.value
            rec[t].update(median=st.median(vals), changes=ch[:10], n_changes=len(ch) - 1, last=vals[-1])
        else:
            rec[t].update(median=st.median(vals), mean=st.mean(vals), min=min(vals), max=max(vals),
                          last20_mean=st.mean(vals[-20:]), win5M=windows(pts, lambda x: x))
            if t == "time/fps":
                rec[t]["wall_h_first_to_last_point"] = round((max(e.wall_time for e in pts)
                                                              - min(e.wall_time for e in pts)) / 3600, 2)
                hour = [e.value for e in pts if e.wall_time - pts[0].wall_time <= 3600]
                rec[t]["lowest_in_first_hour"] = min(hour)
    res[r] = rec

end = [res[r]["train/entropy_loss"]["end_mean"] for r in RUNS]
lab = screen_rule.labels(end)
lr_moved = {r: {"lr_median_dose_as_printed": v, "LR_MOVED": bool(v < LR_LOW or v > LR_HIGH),
                "lr_median_tensorboard": res[r]["train/learning_rate"]["median"]} for r, v in zip(RUNS, LR_DOSE)}
summary = {
    "x70_end_entropy": dict(zip(RUNS, end)), "x70_end_entropy_mean": st.mean(end),
    "x70_highest": max(end), "x70_lowest": min(end),
    "swap_end_entropy": SWAP_END, "swap_lowest": SWAP_LOW, "swap_highest": SWAP_HIGH,
    "swap_end_entropy_mean": CE3D["summary"]["swap_end_entropy_mean"],
    "swap_win5M": {r: CE3D["runs"][r]["train/entropy_loss"]["win5M"] for r in SWAP_END},
    "c0_end_entropy": C0_END, "c0_end_entropy_mean": X67_REC["summary"]["c0_end_entropy_mean"],
    "c0_win5M": {r: X67_REC["runs"][r]["train/entropy_loss"]["win5M"] for r in C0_END},
    "x67_end_entropy": X67_REC["summary"]["x67_end_entropy"],
    "x67_end_entropy_mean": X67_REC["summary"]["x67_end_entropy_mean"],
    "labels_by_screen_rule": lab,
    "lr_moved_bounds": [LR_LOW, LR_HIGH], "lr_moved": lr_moved,
    "magnet_tags_logged": {r: {t: res[r][t]["n"] for t in ("train/magnet_kl", "magnet/refreshes")} for r in RUNS},
}
json.dump({"runs": res, "summary": summary}, open(sys.argv[2], "w"), indent=1)
for r in RUNS:
    e, l, k, f = (res[r]["train/entropy_loss"], res[r]["train/learning_rate"], res[r]["train/approx_kl"],
                  res[r]["time/fps"])
    print(f"{r}\n  H_end {e['end_mean']:.4f} [{e['end_min']:.3f},{e['end_max']:.3f}] H_first20 {e['first20_mean']:.4f} "
          f"H_5M {e['win5M']} H_range [{e['lowest']:.3f},{e['highest']:.3f}] n {e['n']} sup {e['superseded']}\n"
          f"  first point below the swap's lowest end entropy ({SWAP_LOW:.4f}): step {e['first_step_below_swap_lowest']} "
          f"({e['n_points_below_swap_lowest']} of {e['n']} points below it); below its highest ({SWAP_HIGH:.4f}): "
          f"step {e['first_step_below_swap_highest']} ({e['n_points_below_swap_highest']} of {e['n']})\n"
          f"  lr med {l['median']:.4e} last {l['last']:.4e} changes {l['n_changes']} "
          f"{[(s, f'{v:.3e}') for s, v in l['changes']]}\n"
          f"  approx_kl med {k['median']:.4f} 5M {k['win5M']} last20 {k['last20_mean']:.4f}\n"
          f"  time/fps med {f['median']:.0f} mean {f['mean']:.0f} lowest in the first hour {f['lowest_in_first_hour']:.0f} "
          f"wall h {f['wall_h_first_to_last_point']} | magnet tags logged: "
          f"{summary['magnet_tags_logged'][r]}")
print(json.dumps(summary, indent=1))
