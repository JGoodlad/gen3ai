"""Pack the durable rows into the committed deliverable: per checkpoint and arm, one gzipped JSONL with, per state,
the 8 world teams (opponent packed teams), every action's 8 outcomes, and the build's bookkeeping (draws, rejections,
screen summary). The full rows stay in ``~/gen3ai_archive/hidden_info_replica/rows``.

    python -m replica.pack --rows <rows dir> --out <dir>/rows
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

from replica.score import load_rows

KEEP_BUILD = ("tries", "rejects", "reason", "pending_dropped", "notes", "cpu_s")
KEEP_SCREEN = ("rounds", "score", "n_lines", "context_ok", "context_only", "empty_pool", "replays", "pool_sizes",
               "accept", "cpu_s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows_dir = Path(a.rows).expanduser()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in sorted(glob.glob(str(rows_dir / "*" / "*"))):
        cp, arm = Path(d).parts[-2], Path(d).parts[-1]
        rows = load_rows(rows_dir, cp, arm)
        with gzip.open(out / f"{cp}_{arm}.jsonl.gz", "wt") as f:
            for sid in sorted(rows):
                r = rows[sid]
                b = r.get("build", {})
                keep = {"id": sid, "ok": r.get("ok"), "reason": r.get("reason"), "outcomes": r.get("outcomes"),
                        "other_open": r.get("other_open"), "seen": r.get("seen"),
                        "worlds": [{"team": w["team"], "rendered": w.get("rendered")} for w in r.get("worlds", [])],
                        "build": {k: b[k] for k in KEEP_BUILD if k in b},
                        "screen": {k: b.get("screen", {})[k] for k in KEEP_SCREEN if k in b.get("screen", {})}}
                f.write(json.dumps(keep, sort_keys=True) + "\n")
        print(cp, arm, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
