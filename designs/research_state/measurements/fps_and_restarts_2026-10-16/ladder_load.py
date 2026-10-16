"""The detached SNAPSHOT-LADDER updater's wall per promotion, and the rollout slowdown while it runs.

    python .../ladder_load.py <run-dir> <cache.json> <run-name>

Each promotion spawns ``agents.training.snapshot_ladder --promote <step>`` (CPU, its own session),
which plays the new snapshot against EVERY other snapshot in the pool (``promotion_plan``: 5 fresh
baseline pairs at 200 games + the rest round-robin at 100). Its cost therefore grows with the pool
until the pool's cap. The updater shares the CPU with the Rust env core, so a rollout collected
while it runs is slower: this splits the steady cycles by whether a ladder game batch of the
current promotion was still being played.

Promotion wall = the ``eval/pool_snapshot_count`` event of that cycle (the eval that promoted);
the updater's end = the last ``at`` stamp of its games in ``snapshot_ladder/games.jsonl``.
"""
from __future__ import annotations

import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import cycles  # noqa: E402


def main(argv):
    run_dir, cache, run = Path(argv[0]), json.loads(Path(argv[1]).read_text()), argv[2]
    games = [json.loads(x) for x in (run_dir / "snapshot_ladder" / "games.jsonl").read_text().splitlines() if x.strip()]
    procs = [p for p in cache[run]["procs"] if "inherited" not in p["file"]]
    prom = {}
    for p in procs:
        for s, w, _ in p["series"].get("eval/pool_snapshot_count", []):
            prom[s] = w
    jobs = []
    for a in sorted({g["a"] for g in games}):
        ats = [dt.datetime.fromisoformat(g["at"]).timestamp() for g in games if g["a"] == a]
        start = min((w for s, w in prom.items() if abs(s - a) < 50_000), default=None)
        pairs = len([g for g in games if g["a"] == a])
        n = sum(g["games"] for g in games if g["a"] == a)
        jobs.append({"promote_step": a, "pairs": pairs, "games": n, "start": start, "end": max(ats),
                     "wall_s": None if start is None else round(max(ats) - start)})
    cyc = [c for p in procs for c in cycles(p) if not c["eval"]]
    busy = lambda c: any(j["start"] and j["start"] <= c["wall"] - c["dt"] and c["wall"] <= j["end"] + 60 for j in jobs)  # noqa: E731
    on = [c["collect_s"] / c["ds"] * 1e5 for c in cyc if busy(c) and c["step"] > 4.2e6]
    off = [c["collect_s"] / c["ds"] * 1e5 for c in cyc if not busy(c) and c["step"] > 4.2e6]
    gaps = [round(prom[b] - prom[a]) for a, b in zip(sorted(prom), sorted(prom)[1:])]
    out = {"jobs": jobs, "promotion_interval_s": gaps,
           "collect_100k_ladder_running": {"n": len(on), "median": round(st.median(on), 2) if on else None},
           "collect_100k_ladder_idle": {"n": len(off), "median": round(st.median(off), 2) if off else None}}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
