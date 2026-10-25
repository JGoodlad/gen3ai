"""How much does dropping hidden choices (``cant`` / fainted-before-acting) bias the switch rate? A text pass over a
seeded sample of hr_spec_v1 train-date logs; each turn decision of a >= 1500 side is classed by its first own event.

    python -I review_a_mnar.py 6000      # ~2 min, text only, read-only on the corpus
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
import sys
from pathlib import Path

CORPUS = Path(os.environ.get("HRQ_CORPUS", "/home/goodlad/dev/gen3ai/replays/showdown/gen3ou"))
paths = sorted((CORPUS / "SPECTATOR_MANIFEST_2026-09-23.txt").read_text().split())
paths = [p for p in paths if p[:10] <= "2026-09-20"]
random.Random(20261025).shuffle(paths)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
PL = re.compile(r"^\|player\|(p[12])\|([^|]*)\|([^|]*)\|(\d*)")
SUBJ = re.compile(r"^(p[12])a: (.*)$")

C = collections.Counter()
read = 0
for rel in paths:
    if read >= N:
        break
    lines = (CORPUS / rel).read_text(errors="replace").split("\n")
    rating = {}
    for ln in lines[:40]:
        m = PL.match(ln)
        if m and m.group(4):
            rating.setdefault(m.group(1), int(m.group(4)))
    hi = {s for s, r in rating.items() if r >= 1500}
    if not hi:
        continue
    read += 1
    active = {}
    status = collections.defaultdict(dict)       # side -> nick -> status
    win = None                                   # per side: (start status, first event, cant reason)
    first = {}
    start = {}

    def close(hi, first, start):
        for s in hi:
            if s not in start:
                continue
            ev = first.get(s, ("none", ""))
            st = start[s]
            C[("all", ev[0])] += 1
            C[(st or "healthy", ev[0])] += 1
            if ev[0] == "cant":
                C[("cant_reason", ev[1])] += 1

    for ln in lines:
        p = ln.split("|")
        if len(p) < 2:
            continue
        kw = p[1]
        if kw == "turn":
            if win is not None:
                close(hi, first, start)
            win = int(p[2])
            first, start = {}, {}
            for s in ("p1", "p2"):
                if s in active:
                    start[s] = status[s].get(active[s])
            continue
        if len(p) < 3:
            continue
        m = SUBJ.match(p[2])
        if not m:
            if kw == "-cureteam":
                pass
            continue
        s, nick = m.group(1), m.group(2)
        if kw in ("switch", "drag", "replace"):
            if win is not None and kw == "switch" and s not in first:
                first[s] = ("switch", "")
            active[s] = nick
            if len(p) > 4 and p[4].endswith(" slp"):
                status[s][nick] = "slp"
            elif len(p) > 4 and p[4].endswith(" par"):
                status[s][nick] = "par"
            continue
        if kw == "-status" and len(p) > 3:
            status[s][nick] = p[3]
        elif kw == "-curestatus":
            status[s][nick] = None
        elif kw == "-cureteam":
            status[s] = {}
        if win is None or s in first:
            continue
        if kw == "move" and not any(t.startswith("[from]") for t in p[4:]):
            first[s] = ("move", "")
        elif kw == "cant":
            first[s] = ("cant", p[3] if len(p) > 3 else "")
        elif kw == "faint" and nick == active.get(s):
            first[s] = ("faint", "")
    if win is not None:
        close(hi, first, start)

out = {"logs": read}
for st in ("all", "healthy", "slp", "par", "frz"):
    sw, mv, ct, fa, no = (C[(st, k)] for k in ("switch", "move", "cant", "faint", "none"))
    lab = sw + mv
    out[st] = {"switch": sw, "move": mv, "cant": ct, "faint_first": fa, "none": no,
               "P_switch_labelled": sw / lab if lab else None,
               "P_switch_with_hidden_as_move": sw / (lab + ct + fa) if lab + ct + fa else None,
               "hidden_share_of_stay_in": (ct + fa) / (mv + ct + fa) if mv + ct + fa else None}
out["cant_reasons"] = {k[1]: v for k, v in C.most_common() if k[0] == "cant_reason"}
print(json.dumps(out, indent=1))
