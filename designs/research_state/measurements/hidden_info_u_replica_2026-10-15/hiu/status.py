"""A cheap status read of the runner (no engine, no model): done states, rows and failures per arm, the guard's
raises, the median state time, and a crude ETA.

    python -m hiu.status [--root ~/gen3ai_archive/hidden_info_u_replica]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter
from pathlib import Path

ARMS = ("O", "F", "F+U", "F+M", "O-M", "O-U")
N_STATES = 600


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/gen3ai_archive/hidden_info_u_replica")
    a = ap.parse_args(argv)
    root = Path(os.path.expanduser(a.root))
    total_left = 0
    secs = []
    for cp in ("end2", "prod"):
        dd = root / "done" / cp
        done_ids = {f.read_text().strip() for f in dd.glob("*")} if dd.exists() else set()
        done = len(done_ids)
        total_left += N_STATES - done
        print(f"{cp}: {done} / {N_STATES} states done")
        per_state: Counter = Counter()
        for arm in ARMS:
            n = ok = 0
            reasons: Counter = Counter()
            guard = 0
            for p in sorted((root / "rows" / cp / arm).glob("*.jsonl")):
                for line in p.read_text().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    n += 1
                    ok += bool(r.get("ok"))
                    per_state[r["id"]] += float(r.get("wall_s") or 0)
                    if not r.get("ok"):
                        why = str(r.get("reason"))
                        guard += why.startswith("GUARD")
                        reasons[why.split(":")[0][:40]] += 1
            print(f"  {arm:4s} rows {n:4d}  ok {ok:4d}  failed {n - ok:3d}  guard-raised {guard}  {dict(reasons)}")
        secs += [v for k, v in per_state.items() if k in done_ids]
    if secs:
        med = statistics.median(secs)
        workers = 2
        print(f"done states: median wall {med:.0f} s; {total_left} states left; ETA at {workers} workers ~ "
              f"{total_left * statistics.mean(secs) / workers / 3600:.1f} h (mean {statistics.mean(secs):.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
