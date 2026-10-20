"""Read-only: X26's behaviour/max_abs_dlogp_judged and p99 up to the fork step (what an X35 fork INHERITS)."""
import glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
FORK = 280_000_170
rows = {}
for f in sorted(glob.glob("/home/goodlad/dev/gen3ai/models/rb_x26_s2601/tb/events.out.tfevents.*")):
    ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
    for tag in ("behaviour/max_abs_dlogp_judged", "behaviour/p99_abs_dlogp_current"):
        if tag in ea.Tags().get("scalars", []):
            for e in ea.Scalars(tag):
                rows.setdefault(tag, []).append((e.step, e.value))
for tag, r in rows.items():
    pre = [(s, v) for s, v in r if s <= FORK]
    bar = 1e-3 if "max" in tag else 1e-4
    over = [(s, v) for s, v in pre if not (v < bar)]
    print(tag, "rows<=fork", len(pre), "max", max(v for _, v in pre) if pre else None, "rows>=bar", len(over),
          "first over", over[:3])
