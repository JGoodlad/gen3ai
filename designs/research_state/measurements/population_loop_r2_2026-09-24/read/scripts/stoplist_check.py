#!/usr/bin/env python3
"""Round 2 §5 branch-V STOP-list facts, read from files (the launch banners are not in the child logs).

B2's TB ``train/stable_fraction`` must be 0.36 and C2's 0.00 after the first eval (post-fork points
only: the ``*.inherited.*`` event file is the parent's history and is skipped); every B2 / C2 eval row's
``matchup_hash`` must be ``ef5242cffd``; RB2 and RC2 must land at 119,341,056.
Usage (PYTHONPATH containing src): stoplist_check.py <out.json>
"""
import glob
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

MODELS = Path("/home/goodlad/dev/gen3ai/models")
GEN = {"B2": ("ai_v13_27_popr2_loop", 0.36), "C2": ("ai_v13_28_popr2_ctrl", 0.0)}
READERS = {"RB2": "ai_v13_29_popr2_read_loop", "RC2": "ai_v13_30_popr2_read_ctrl"}
FORK, HASH, LAND = 103_219_200, "ef5242cffd", 119_341_056


def main():
    out, ok = {}, True
    for k, (run, want) in GEN.items():
        pts = []
        for f in sorted(glob.glob(str(MODELS / run / "tb" / "events.out.tfevents.*"))):
            if "inherited" in f:
                continue
            ea = EventAccumulator(f, size_guidance={"scalars": 0})
            ea.Reload()
            if "train/stable_fraction" in ea.Tags()["scalars"]:
                pts += [(e.step, round(e.value, 4)) for e in ea.Scalars("train/stable_fraction")]
        pts = sorted(p for p in pts if p[0] > FORK)
        hashes = sorted({json.loads(line).get("matchup_hash")
                         for line in open(MODELS / run / "eval_results.jsonl")})
        good = bool(pts) and all(abs(v - want) < 1e-6 for _, v in pts) and hashes == [HASH]
        ok &= good
        out[k] = {"run": run, "stable_fraction_post_fork": pts, "want": want,
                  "matchup_hashes": hashes, "ok": good}
    # the readers' landing step, as the meter itself read it (brgap_r2.json beside this script's output)
    r2 = json.loads((Path(sys.argv[1]).parent / "brgap_r2.json").read_text())
    got = {r["run"]: r["num_timesteps"] for r in r2["runs"]}
    for k, run in READERS.items():
        good = got.get(run) == LAND
        ok &= good
        out[k] = {"run": run, "num_timesteps": got.get(run), "want": LAND, "ok": good}
    out["ALL_OK"] = ok
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
