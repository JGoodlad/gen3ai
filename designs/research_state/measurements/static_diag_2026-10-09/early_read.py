"""H1's early diagonal, read: static seed i vs legacy seed i at ~5M and ~10M (this study's CPU play, the
`play-many` result line of h2h_early.log) beside the 15M diagonal of the registered look-1/2 cross
(static_screen_look2_2026-10-09/result.json). DESCRIPTIVE; writes early_h2h.json beside this file."""
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path("/home/goodlad/gen3ai_archive/static_diag_2026-10-09/h2h_early.log")


def stats(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return {"mean_pp": round(m, 2), "delta_pp": round(m - 50, 2), "sd_pp": round(sd, 2), "n": n,
            "se_pp": round(sd / math.sqrt(n), 2)}


def main():
    res = None
    for line in LOG.read_text().splitlines():
        if line.startswith('{"cells"'):
            res = json.loads(line)
    assert res is not None, "no play-many result line"
    by = {}
    for c in res["cells"]:
        seed = int(re.search(r"_s100(\d)@", c["player"]).group(1))
        step = int(c["player"].split("@")[1])
        stage = "5M" if step < 7_500_000 else "10M"
        by.setdefault(stage, {})[seed] = round(100 * c["score"], 2)
    look2 = json.loads((HERE.parent / "static_screen_look2_2026-10-09" / "result.json").read_text())
    h = look2["stat"]["h_pp"] if "h_pp" in look2["stat"] else None
    by["15M"] = {i + 1: round(h[i][i], 2) for i in range(5)}
    out = {"schema": "static_diag_early_h2h_v1", "tag": "DESCRIPTIVE",
           "note": "score = static seed i's score vs legacy seed i (pp, a draw ½); 5M/10M: 400 mirrored pairs per cell, "
                   "CPU eager at P_st, regime " + str(res["cells"][0]["regime_id"]) + "; 15M: the registered cross's "
                   "diagonal (1,000 pairs per cell, CUDA graph, same regime id)",
           "cells": by, "summary": {st: stats(list(v.values())) for st, v in by.items()},
           "summary_without_seed5": {st: stats([x for s, x in v.items() if s != 5]) for st, v in by.items()},
           "paired_change_pp": {f"{a}->{b}": {k: v for k, v in stats([by[b][s] - by[a][s] for s in range(1, 6)]).items()
                                               if k != "delta_pp"}
                                for a, b in (("5M", "15M"), ("10M", "15M"), ("5M", "10M"))}}
    (HERE / "early_h2h.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
