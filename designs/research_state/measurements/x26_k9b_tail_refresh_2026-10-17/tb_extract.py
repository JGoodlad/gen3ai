"""Extract rb_x26_s2601's K9(b) series from its TensorBoard events at full precision (read only).

    python3 tb_extract.py --tb models/rb_x26_s2601/tb --out tb_series.json

Writes, per tag, the [step, value] pairs (the last write wins on a duplicated step, as after a resume) for
the behaviour/* tags and the sharpness context (rollout/policy_entropy, train/entropy_loss).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict

KEEP = ("rollout/policy_entropy", "train/entropy_loss", "train/approx_kl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    series: Dict[str, Dict[int, float]] = {}
    files = sorted(glob.glob(os.path.join(a.tb, "events.out.tfevents.*")), key=os.path.getmtime)
    for f in files:
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            if tag.startswith("behaviour/") or tag in KEEP:
                d = series.setdefault(tag, {})
                for ev in ea.Scalars(tag):
                    d[int(ev.step)] = float(ev.value)
    out = {"files": [os.path.basename(f) for f in files],
           "series": {k: sorted(v.items()) for k, v in sorted(series.items())}}
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print({k: (len(v), v[-1] if v else None) for k, v in out["series"].items()})


if __name__ == "__main__":
    main()
