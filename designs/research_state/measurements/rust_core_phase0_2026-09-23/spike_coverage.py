"""Coverage census for the event spike: how often each ambiguity-prone SHAPE actually occurred in
the corpus the diff ran on — a rule that never fired was not tested, only not contradicted.

    python3 designs/research_state/measurements/rust_core_phase0_2026-09-23/spike_coverage.py \
        /tmp/p0/spike_key0 /tmp/p0/spike_key5000
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys


def main() -> int:
    c = collections.Counter()
    battles = 0
    for d in sys.argv[1:]:
        for f in sorted(glob.glob(os.path.join(d, "key*_spike.json"))):
            battles += 1
            spike = json.load(open(f))
            for r in spike["recs"]:
                t = r["typed"]
                line = r["line"]
                if t is None:
                    kw = line.split("|")[1] if "|" in line else ""
                    c[f"untyped line: {kw}"] += 0          # keep the key space honest
                    continue
                k = t["k"]
                c[f"typed {k}"] += 1
                if k == "move":
                    if t.get("from"):
                        c[f"move [from] {t['from']}"] += 1
                    if t.get("miss") or r["miss_suffix"]:
                        c["move [miss] (announce or retro)"] += 1
                    if r["still_suffix"] or t["target"] is None:
                        c["move [still] / empty target"] += 1
                    if "[from] lockedmove" in line:
                        c["move [from] lockedmove (retro)"] += 1
                if k == "switch" and t.get("from"):
                    c[f"switch [from] {t['from']}"] += 1
                if k == "cant":
                    c[f"cant reason={t['reason']}"] += 1
                    if t.get("of"):
                        c["cant with [of] (ability-sourced, e.g. Damp)"] += 1
                if k in ("damage", "heal") and t.get("of"):
                    c[f"{k} with [of] ({t['cause']})"] += 1
                if k in ("damage", "heal") and t.get("cause"):
                    c[f"{k} cause class: {t['cause'].split(':')[0]}"] += 1
                if k == "miss" and t.get("user_raw"):
                    c["miss with SLOT-LESS user (future move)"] += 1
                if k == "curestatus" and t.get("ident_raw"):
                    c["curestatus via side/bench ident (Heal Bell)"] += 1
                if k in ("fail",) and t.get("cause"):
                    c[f"fail [from] {t['cause']}"] += 1
                if k in ("crit", "immune", "resisted", "supereffective", "miss", "fail"):
                    sc = r["scope"]
                    c[f"outcome scope: {sc.split(':')[0]}"] += 1
    print(f"battles: {battles}")
    for k, v in sorted(c.items()):
        if v:
            print(f"  {v:>7}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
