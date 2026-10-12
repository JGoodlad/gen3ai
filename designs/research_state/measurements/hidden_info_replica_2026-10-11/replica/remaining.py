"""List the registered states of (cp, arm) with no row yet (to re-shard a partly-run arm over more workers).

    python -m replica.remaining --cp end2 --arm F --rows <rows dir> --out ids.txt
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, default=0, help="only the first N registered states")
    ap.add_argument("--only", default="", help="only the states listed in this file (e.g. the interim subset)")
    a = ap.parse_args(argv)
    here = Path(__file__).resolve().parent.parent
    states = json.loads((here.parent / "value_disc_pilot_2026-10-11" / f"states_{a.cp}.json").read_text())["states"]
    done = set()
    for f in glob.glob(str(Path(a.rows) / a.cp / a.arm / "*.jsonl")):
        for line in open(f):
            if line.strip():
                done.add(json.loads(line)["id"])
    order = list(states)[:a.first] if a.first else list(states)
    if a.only:
        keep = set(Path(a.only).read_text().split())
        order = [s for s in order if s in keep]
    todo = [s for s in order if s not in done]
    Path(a.out).write_text("".join(s + "\n" for s in todo))
    print(f"{a.cp} {a.arm}: {len(done)} done, {len(todo)} remaining -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
