"""Windowed contention + GPU memory sampler for one measured launch (gpu_checks_endstate, 2026-10-09).

    python sampler.py --root <launcher pid> --out <tag>.cpu.jsonl [--interval 10]

One JSON row per window until the root pid exits: the `utils.cpu_meter` windowed factor with the
launch's own process tree subtracted as SELF, load1, PSI, and nvidia-smi's memory.used / util.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

from utils.cpu_meter import reading_between, take_sample, topology


def smi() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        return {"gpu_mem_used_mib": float(out[0]), "gpu_util": float(out[1])}
    except Exception as exc:  # noqa: BLE001 — a missed sample is recorded, never fatal
        return {"smi_error": repr(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=10.0)
    a = ap.parse_args()
    topo = topology()
    prev = take_sample([a.root], topo)
    with open(a.out, "a") as fh:
        while os.path.exists(f"/proc/{a.root}"):
            time.sleep(a.interval)
            cur = take_sample([a.root], topo)
            r = reading_between(prev, cur, self_root=a.root, topo=topo)
            row = {"t": cur.wall, "dt": cur.mono - prev.mono, "factor": round(r.factor, 4), "busy": r.busy_cpus,
                   "self": r.self_cpus, "psi": r.psi_some, "load1": cur.load1, "source": r.source}
            row.update(smi())
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            prev = cur


if __name__ == "__main__":
    main()
