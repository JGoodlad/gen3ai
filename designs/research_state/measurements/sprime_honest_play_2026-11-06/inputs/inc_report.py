"""S'-HONEST play: the incremental builder's REPORT over a searched label's cells (OUTCOME-BLIND; PREREG §6, §8, §12).

It reads each complete attempt dir's ``search_decisions.jsonl`` (never a ``.partial.<epoch>`` dir, a summary, a
capture or ``games.jsonl``) and prints, per panel and pooled: the search-path decisions by the builder's mode
(``honest.inc.mode``: cold, warm, stuck, recold), exclusions, the battles that held an exclusion or a stuck stretch
(S'H-13), the searched decisions that held a world past the correlation bound and the age distribution (S'H-15), the
warm chain's fates, and the CPU of each battle's cold first decision.

    python3 inc_report.py CELLS_DIR
"""
import collections
import json
import os
import sys
from pathlib import Path


def report(cells):
    cells = Path(cells)
    acc = collections.defaultdict(lambda: {"cells": 0, "decisions": 0, "searched": 0, "excluded": 0,
                                           "modes": collections.Counter(), "chain": collections.Counter(),
                                           "battles_with_exclusion": 0, "battles_stuck": 0,
                                           "searched_with_overage": 0, "max_age": collections.Counter(),
                                           "cold_cpu_s": []})
    for d in sorted(os.listdir(cells)):
        p = cells / d / "search_decisions.jsonl"
        if ".partial." in d or not p.is_file():
            continue
        panel = d.split("__")[1] if d.count("__") >= 3 else "?"
        excl = stuck = False
        for k in (panel, "U"):
            acc[k]["cells"] += 1
        for x in open(p):
            r = json.loads(x)
            if "key" not in r or not (r.get("searched") or r.get("why") == "excluded"):
                continue
            inc = (r.get("honest") or {}).get("inc") or {}
            mode = inc.get("mode", "none")
            for k in (panel, "U"):
                a = acc[k]
                a["decisions"] += 1
                a["modes"][mode] += 1
                a["chain"][str(inc.get("chain"))] += 1
                if r.get("searched"):
                    a["searched"] += 1
                    ages = inc.get("ages") or []
                    a["max_age"][max(ages) if ages else -1] += 1
                    a["searched_with_overage"] += int(bool(inc.get("overage")))
                else:
                    a["excluded"] += 1
                if mode == "cold":
                    a["cold_cpu_s"].append(float(r.get("cpu_s") or 0.0))
            excl = excl or r.get("why") == "excluded"
            stuck = stuck or mode == "stuck"
        for k in (panel, "U"):
            acc[k]["battles_with_exclusion"] += int(excl)
            acc[k]["battles_stuck"] += int(stuck)
    out = {}
    for k, a in acc.items():
        cold = sorted(a.pop("cold_cpu_s"))
        n = a["decisions"]
        out[k] = dict(a, modes=dict(a["modes"]), chain=dict(a["chain"]),
                      max_age={str(x): c for x, c in sorted(a["max_age"].items())},
                      excluded_share=round(a["excluded"] / n, 4) if n else None,
                      stuck_share=round(a["modes"].get("stuck", 0) / n, 4) if n else None,
                      overage_share=round(a["searched_with_overage"] / a["searched"], 4) if a["searched"] else None,
                      cold_cpu_s={"n": len(cold), "mean": round(sum(cold) / len(cold), 1) if cold else None,
                                  "max": round(cold[-1], 1) if cold else None})
    return {"schema": "sprime_honest_inc_report_v1", "cells": str(cells), "by_panel": out}


if __name__ == "__main__":
    print(json.dumps(report(sys.argv[1]), indent=1, sort_keys=True))
