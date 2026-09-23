"""Aggregate the Phase-0 searched-decision battery (`logs/decision_*_r*.log`) into the table the
README reports: per-round view÷protocol ratios with the load beside each, the medians, and the
view road's EXCLUSIVE wall grouped into the program's buckets.

    python3 designs/research_state/measurements/rust_core_phase0_2026-09-23/aggregate_baseline.py

Reads only the logs beside it. The `expand_many` span is split into rust / transport+JSON by the
driver-side timer's own shares (`logs/expand_many_split_wide_r*.log`, `POKESIM_SEARCH_TIMING=1`);
`open_root` has no such split and is reported as its own row.
"""

from __future__ import annotations

import glob
import os
import re
import statistics
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")

BUCKETS = {
    "rust engine (expand_many child share)": ["__em_rust__"],
    "rust open_root (engine + its JSON, unsplit)": ["port open_root (rust)"],
    "transport / JSON (pipe + dumps + json.loads + reply GC)": ["__em_py__"],
    "Python parse (prefix replay through poke-env, event folds)": [
        "prefix replay (open_view_fork)", "replay player build", "poke-env protocol feed",
        "event fold (ply protocol)", "event-fold branch"],
    "trackers / fork": ["tracker fork (thaw)", "decision encode/track", "successor context"],
    "read-model folds (view_adapter, mask, choices)": [
        "view_adapter read-models", "action mask", "action_choices / map_actions_at"],
    "encode (obs)": ["encode (obs)"],
    "Python glue (uninstrumented)": ["python glue (uninstrumented)"],
}


def _load(path: str) -> float:
    with open(path) as fh:
        head = fh.readline()
    return float(head.split()[4])


def _parse(path: str) -> Dict:
    txt = open(path).read()
    m = re.search(r"^\s+(view|protocol)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", txt, re.M)
    road, arms, ms_dec, ms_arm = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
    phases: Dict[str, float] = {}
    sec = txt.split("PHASE BREAKDOWN", 1)[1].split("SPANS", 1)[0]
    for line in sec.splitlines():
        mm = re.match(r"^\s{2}(.+?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s*$", line)
        if mm and not mm.group(1).startswith("phase"):
            phases[mm.group(1).strip()] = float(mm.group(2))
    return {"road": road, "arms": arms, "ms_dec": ms_dec, "ms_arm": ms_arm, "phases": phases,
            "load": _load(path)}


def _em_split() -> Dict[str, float]:
    rust = py = 0.0
    for p in sorted(glob.glob(os.path.join(LOGS, "expand_many_split_wide_r*.log"))):
        for line in open(p):
            mm = re.match(r"^(rust: .+?|transport .+?|py: .+?)\s{2,}([\d.]+)\s", line)
            if mm:
                if mm.group(1).startswith("rust"):
                    rust += float(mm.group(2))
                else:
                    py += float(mm.group(2))
    tot = rust + py
    return {"rust": rust / tot, "py": py / tot}


def main() -> None:
    split = _em_split()
    print(f"expand_many split (driver timer, 2 runs pooled): rust {100 * split['rust']:.1f}% · "
          f"transport+JSON+GC {100 * split['py']:.1f}%\n")
    for shape in ("wide", "b1"):
        rounds: List[Dict] = []
        r = 1
        while os.path.exists(os.path.join(LOGS, f"decision_{shape}_view_r{r}.log")):
            v = _parse(os.path.join(LOGS, f"decision_{shape}_view_r{r}.log"))
            p = _parse(os.path.join(LOGS, f"decision_{shape}_protocol_r{r}.log"))
            rounds.append({"r": r, "v": v, "p": p})
            r += 1
        print(f"== {shape}: {len(rounds)} interleaved rounds, one road per process")
        print(f"   {'round':>5} {'load(v)':>8} {'view ms/dec':>12} {'load(p)':>8} "
              f"{'protocol ms/dec':>16} {'view/protocol':>14}")
        ratios = []
        for x in rounds:
            ratio = x["v"]["ms_dec"] / x["p"]["ms_dec"]
            ratios.append(ratio)
            print(f"   {x['r']:>5} {x['v']['load']:>8.2f} {x['v']['ms_dec']:>12.1f} "
                  f"{x['p']['load']:>8.2f} {x['p']['ms_dec']:>16.1f} {ratio:>14.3f}")
        vm = statistics.median(x["v"]["ms_dec"] for x in rounds)
        pm = statistics.median(x["p"]["ms_dec"] for x in rounds)
        print(f"   median view {vm:.1f} ms/decision ({rounds[0]['v']['arms']} arms / 10 decisions), "
              f"protocol {pm:.1f}; median paired view/protocol {statistics.median(ratios):.3f} "
              f"(range {min(ratios):.3f}-{max(ratios):.3f}); loads "
              f"{min(min(x['v']['load'], x['p']['load']) for x in rounds):.1f}-"
              f"{max(max(x['v']['load'], x['p']['load']) for x in rounds):.1f} on 16 cpus")
        # pooled exclusive phase shares, view road
        tot_wall = sum(x["v"]["ms_dec"] * 10 for x in rounds)   # ms_dec is a median; use phases
        pooled: Dict[str, float] = {}
        for x in rounds:
            for k, val in x["v"]["phases"].items():
                pooled[k] = pooled.get(k, 0.0) + val
        pooled["__em_rust__"] = pooled.get("port expand_many (rust)", 0.0) * split["rust"]
        pooled["__em_py__"] = pooled.get("port expand_many (rust)", 0.0) * split["py"]
        denom = sum(v for k, v in pooled.items()
                    if k not in ("__em_rust__", "__em_py__"))
        del tot_wall
        print(f"   VIEW road exclusive wall by bucket (pooled over {len(rounds)} runs):")
        shares = {}
        for b, keys in BUCKETS.items():
            s = sum(pooled.get(k, 0.0) for k in keys)
            shares[b] = 100 * s / denom
            print(f"     {b:<62} {shares[b]:>5.1f}%")
        narrow = (shares["Python glue (uninstrumented)"] + shares["trackers / fork"]
                  + shares["read-model folds (view_adapter, mask, choices)"])
        broad = narrow + shares["Python parse (prefix replay through poke-env, event folds)"]
        print(f"   §7 'Python glue + trackers + folds': {narrow:.1f}% narrow · {broad:.1f}% with the "
              f"protocol/event folds · +transport/JSON {broad + shares['transport / JSON (pipe + dumps + json.loads + reply GC)']:.1f}%")
        print("   (unbucketed rows, for audit):",
              sorted(k for k in pooled if not any(k in ks for ks in BUCKETS.values())
                     and not k.startswith("__") and k != "port expand_many (rust)"))
        print()


if __name__ == "__main__":
    main()
