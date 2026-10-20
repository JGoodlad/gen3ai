import sys, json, statistics as st
sys.path.insert(0, "/home/goodlad/dev/gen3ai/src")
from main.x35.tbrows import load_segments, fork_step, own_event_files
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for run in sys.argv[1:]:
    f = own_event_files(run)
    print("==", run, f, "fork", fork_step(run))
    for ef in f:
        ea = EventAccumulator(ef, size_guidance={"scalars": 0}); ea.Reload()
        tags = ea.Tags()["scalars"]
        out = {}
        for t in tags:
            evs = ea.Scalars(t)
            out[t] = [(e.step, e.value, e.wall_time) for e in evs]
        json.dump(out, open("/tmp/x47rev/" + run.rstrip("/").split("/")[-1] + ".json", "w"))
        print(len(tags), "tags")
        for t in sorted(tags):
            if any(k in t for k in ("collect", "update_wall", "fps", "time", "wall", "lifecycle")):
                print("  ", t, len(out[t]))
