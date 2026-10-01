#!/usr/bin/env python3
"""PROGRESS D-6: the MEMORY rule on a candidate N*, from its pre-flight run dir and the N = 48 control.

    mem_rule.py <preflight run dir> <control run dir>      # exit 0 = PASS, 1 = FAIL, 2 = no reading

PASS iff, at the pre-flight's last lifecycle sample: ceiling - demand >= 512 MiB (K6's ceiling already
subtracts its own 512 MiB margin, so this is >= 1 GiB of device headroom at peak — D-6 as CORRECTED before
any candidate was read), AND demand <= the control's max demand + 512 MiB.
"""
from __future__ import annotations

import glob
import json
import sys


def lifecycle(run: str) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out: dict = {}
    for f in sorted(glob.glob(f"{run}/tb/**/events.out*", recursive=True)):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for t in ea.Tags()["scalars"]:
            if t.startswith("lifecycle/cuda_"):
                out.setdefault(t[len("lifecycle/"):], []).extend((e.step, e.value) for e in ea.Scalars(t))
    return {k: sorted(v) for k, v in out.items()}


def main(argv) -> int:
    pre, ctl = lifecycle(argv[0]), lifecycle(argv[1])
    if not pre.get("cuda_demand_mib") or not ctl.get("cuda_demand_mib"):
        print(json.dumps({"verdict": "NO READING", "pre_tags": sorted(pre), "ctl_tags": sorted(ctl)}))
        return 2
    last = {k: v[-1][1] for k, v in pre.items()}
    ctl_demand = max(v for _s, v in ctl["cuda_demand_mib"])
    headroom = last["cuda_ceiling_mib"] - last["cuda_demand_mib"]
    rel = last["cuda_demand_mib"] - ctl_demand
    ok = headroom >= 512 and rel <= 512
    print(json.dumps({"verdict": "PASS" if ok else "FAIL", "demand_mib": last["cuda_demand_mib"],
                      "reserved_mib": last.get("cuda_reserved_mib"), "device_free_mib": last.get("cuda_device_free_mib"),
                      "floor_mib": last.get("cuda_floor_mib"), "ceiling_mib": last["cuda_ceiling_mib"],
                      "headroom_mib": headroom, "control_demand_mib": ctl_demand, "demand_over_control_mib": rel,
                      "ooms": last.get("cuda_ooms"), "samples": len(pre["cuda_demand_mib"])}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
