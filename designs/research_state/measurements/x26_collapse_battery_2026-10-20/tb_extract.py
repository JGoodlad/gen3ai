"""Extract every TensorBoard SCALAR of X26 (read-only) into one compact archive file.

    run.sh tb_extract.py <run_dir/tb> <out.json.gz>

Output: {"files": [...], "tags": {tag: [[step, value, wall_time], ...]}} with duplicate steps (a restart re-logs a
step) resolved to the LAST write by wall time. Reads only; writes only <out>.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util


def main(tb_dir: str, out: str) -> int:
    files = sorted(Path(tb_dir).glob("events.out.tfevents.*"))
    tags: dict[str, dict[int, tuple[float, float]]] = {}
    for f in files:
        for ev in EventFileLoader(str(f)).Load():
            if not ev.HasField("summary"):
                continue
            for v in ev.summary.value:
                if v.HasField("simple_value"):
                    val = float(v.simple_value)
                elif v.HasField("tensor"):
                    try:
                        arr = tensor_util.make_ndarray(v.tensor)
                    except Exception:  # noqa: BLE001 - a non-numeric tensor (text) is not a scalar
                        continue
                    if arr.size != 1 or arr.dtype.kind not in "fiub":
                        continue
                    val = float(arr.reshape(()))
                else:
                    continue
                d = tags.setdefault(v.tag, {})
                prev = d.get(ev.step)
                if prev is None or ev.wall_time >= prev[1]:
                    d[ev.step] = (val, ev.wall_time)
    blob = {"files": [f.name for f in files],
            "tags": {t: [[s, v, w] for s, (v, w) in sorted(d.items())] for t, d in sorted(tags.items())}}
    with gzip.open(out, "wt") as fh:
        json.dump(blob, fh)
    print(f"{len(files)} files, {len(tags)} scalar tags -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
