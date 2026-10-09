"""List one run's TensorBoard scalar tags (helper for curves.py). DESCRIPTIVE diagnostic."""
import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
tags = set()
for f in sorted(glob.glob(f"{run}/tb/*")):
    e = EventAccumulator(f, size_guidance={"scalars": 0})
    e.Reload()
    tags |= set(e.Tags()["scalars"])
for t in sorted(tags):
    print(t)
