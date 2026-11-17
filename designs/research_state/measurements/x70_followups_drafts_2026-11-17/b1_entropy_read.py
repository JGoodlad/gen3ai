"""The B1 runs' entropy by 5M window and their realised training span, read from their own TensorBoard.

A frozen record of what DRAFTS (ii) and (iii) quote, not a tool. Read-only on models/ (the main checkout's).
    PYTHONPATH=src python b1_entropy_read.py <models dir>  ->  b1_entropy_read.log
"""
import statistics as st
import sys
from pathlib import Path

from main.ops.run_ref import scalar_events

RUNS = ("rb_b1c_s5101", "rb_b1c_s5102", "rb_b1c_s5110", "rb_b1br_s5105", "rb_b1br_s5106", "rb_b1w_s5103",
        "rb_b1wb_s5107")


def main(models: Path) -> int:
    print("run | last step | end entropy (mean of the last 20 points) | first-to-last update wall span, h | "
          "mean entropy by 5M window (window start, M: nats)")
    for r in RUNS:
        ev = scalar_events(models / r, ["train/entropy_loss"]).get("train/entropy_loss", [])
        if not ev:
            print(f"{r} | no points")
            continue
        win = {}
        for e in ev:
            win.setdefault(int(e.step // 5_000_000), []).append(-e.value)
        end = st.mean([-e.value for e in ev[-20:]])
        span = (ev[-1].wall_time - ev[0].wall_time) / 3600.0
        print(f"{r} | {ev[-1].step:,} | {end:.4f} | {span:.2f} | "
              + " ".join(f"{k * 5}:{st.mean(v):.3f}" for k, v in sorted(win.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
