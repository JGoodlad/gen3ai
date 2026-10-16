"""Where a training ITERATION's wall time goes, per run: the medians of the trainer's own per-iteration rows
(`collect_ms`, `update_wall_s`, `gpu_wait_ms_per_host_step`, `fps`, `collect_decisions`) in launcher_child.full.log.

    python iteration_split.py <run_dir> [<run_dir> ...]   -> JSON on stdout. Read-only.
"""
import json
import re
import statistics
import sys
from pathlib import Path

KEYS = ("collect_ms", "update_wall_s", "gpu_wait_ms_per_host_step", "fps", "collect_decisions")
PAT = re.compile(r"\|\s+([a-z_]+)\s+\|\s+([-0-9.e+]+)\s*\|")


def split(run: Path) -> dict:
    vals = {k: [] for k in KEYS}
    with (run / "launcher_child.full.log").open(errors="replace") as f:
        for line in f:
            m = PAT.match(line)
            if m and m.group(1) in vals:
                vals[m.group(1)].append(float(m.group(2)))
    out = {k: {"n": len(xs), "median": statistics.median(xs) if xs else None} for k, xs in vals.items()}
    c, u = out["collect_ms"]["median"], out["update_wall_s"]["median"]
    if c is not None and u is not None:
        it = c / 1000.0 + u
        out["iteration_s"] = it
        out["update_share"] = u / it
        out["collect_share"] = (c / 1000.0) / it
    return out


if __name__ == "__main__":
    print(json.dumps({Path(p).name: split(Path(p)) for p in sys.argv[1:]}, indent=1))
