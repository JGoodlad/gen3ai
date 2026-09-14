"""Join our per-battle records with Foul Play's timestamped log into games.jsonl
and print the campaign summary.

Foul Play's own formatter emits no timestamps (`fp/config.py::CustomFormatter`), so its
per-move think time is read off the wall clock added by `ts.py`: the gap between
`Searching for a move using MCTS...` and the matching `Choice: <x>`.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys

ROOT = "/home/goodlad/.claude/jobs/9ab51de6/tmp/foul_play"

TS = re.compile(r"^(\d+\.\d+)\s+(\w+)\s+(.*)$")
INIT = re.compile(r"Initialized (battle-\S+) against")
ITER = re.compile(r"Iterations \d+: (\d+)")
SAMPLING = re.compile(r"Sampling (\d+) battles at (\d+)ms each")


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def parse_fp_logs() -> dict:
    """battle_tag -> foul-play side facts."""
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "logs", "fp_s*.log"))):
        tag = None
        cur = None
        t_search = None
        pending_iters = []
        pending_states = None
        errors = 0
        session = os.path.basename(path)
        with open(path, errors="replace") as f:
            for line in f:
                m = TS.match(line.rstrip("\n"))
                if not m:
                    if "Traceback" in line:
                        errors += 1
                    continue
                ts, level, msg = float(m.group(1)), m.group(2), m.group(3)
                if level in ("ERROR", "CRITICAL"):
                    errors += 1
                mi = INIT.search(msg)
                if mi:
                    tag = mi.group(1)
                    cur = out.setdefault(tag, {
                        "fp_think_s": [], "fp_iterations": [], "fp_states_per_decision": [],
                        "fp_errors": 0, "fp_log": session, "fp_winner": None,
                    })
                    continue
                if cur is None:
                    continue
                ms = SAMPLING.search(msg)
                if ms:
                    pending_states = int(ms.group(1))
                if msg.startswith("Searching for a move using MCTS"):
                    t_search = ts
                    pending_iters = []
                mit = ITER.search(msg)
                if mit:
                    pending_iters.append(int(mit.group(1)))
                if msg.startswith("Choice:") and t_search is not None:
                    cur["fp_think_s"].append(ts - t_search)
                    if pending_iters:
                        # the log prints each state's visit count twice; halve the list
                        half = pending_iters[: max(1, len(pending_iters) // 2)]
                        cur["fp_iterations"].append(sum(half))
                    if pending_states:
                        cur["fp_states_per_decision"].append(pending_states)
                    t_search = None
                if msg.startswith("Winner:"):
                    cur["fp_winner"] = msg.split("Winner:")[-1].strip()
                    cur = None
                    tag = None
        # errors are per-log, not per-battle; attribute to every battle in that log
        for v in out.values():
            if v["fp_log"] == session:
                v["fp_errors"] = errors
    return out


def main() -> int:
    ours = [json.loads(l) for l in open(os.path.join(ROOT, "games_raw.jsonl"))]
    fp = parse_fp_logs()
    games = []
    for r in ours:
        f = fp.get(r["battle_tag"], {})
        think = f.get("fp_think_s") or []
        iters = f.get("fp_iterations") or []
        games.append({
            **r,
            "our_win": bool(r["won"]),
            "result": "win" if r["won"] else ("loss" if r["won"] is False else "tie/unknown"),
            "fp_winner": f.get("fp_winner"),
            "fp_decisions": len(think),
            "fp_think_s_mean": (sum(think) / len(think)) if think else None,
            "fp_think_s_max": max(think) if think else None,
            "fp_mcts_visits_mean": (sum(iters) / len(iters)) if iters else None,
            "fp_states_per_decision_max": max(f.get("fp_states_per_decision") or [0]) or None,
            "fp_errors_in_session_log": f.get("fp_errors", 0),
            "fp_log": f.get("fp_log"),
        })
    with open(os.path.join(ROOT, "games.jsonl"), "w") as g:
        for rec in games:
            g.write(json.dumps(rec) + "\n")

    n = len(games)
    w = sum(1 for g in games if g["our_win"])
    lo, hi = wilson(w, n)
    turns = [g["turns"] for g in games]
    forf = sum(1 for g in games if g["our_forfeit_fired"])
    our_think = [g["our_think_ms_mean"] for g in games if g["our_think_ms_mean"]]
    fpt = [g["fp_think_s_mean"] for g in games if g["fp_think_s_mean"]]
    fpmax = [g["fp_think_s_max"] for g in games if g["fp_think_s_max"]]
    vis = [g["fp_mcts_visits_mean"] for g in games if g["fp_mcts_visits_mean"]]
    fpdec = [g["fp_decisions"] for g in games]
    unmatched = sum(1 for g in games if not g["fp_log"])
    print(json.dumps({
        "games": n,
        "our_wins": w,
        "our_win_rate": w / n if n else None,
        "wilson95": [round(lo, 4), round(hi, 4)],
        "mean_turns": sum(turns) / n if n else None,
        "median_turns": sorted(turns)[n // 2] if n else None,
        "max_turns": max(turns) if n else None,
        "our_forfeit_fired_n": forf,
        "our_think_ms_mean": sum(our_think) / len(our_think) if our_think else None,
        "our_think_ms_max": max(g["our_think_ms_max"] for g in games if g["our_think_ms_max"]),
        "fp_think_s_mean": sum(fpt) / len(fpt) if fpt else None,
        "fp_think_s_max": max(fpmax) if fpmax else None,
        "fp_mcts_visits_per_decision_mean": sum(vis) / len(vis) if vis else None,
        "fp_decisions_total": sum(fpdec),
        "fp_error_lines_total": sum(g["fp_errors_in_session_log"] for g in games),
        "games_without_a_matched_fp_log": unmatched,
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
