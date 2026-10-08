"""Classify the shadow's `race` records by the keywords of the lines that arrived while a decision was open.

A `race` is the shadow harness noting that a websocket frame reached the client while poke-env had not yet
answered the open decision; it is a hazard for the SHADOW's choice note only when that frame carries battle
protocol the reader folds (a non-ROOM_SKIP line). Usage: race_kinds.py <read dir>"""
import collections
import json
import os
import sys

sys.path.insert(0, os.environ.get("GEN3AI_SRC", "src"))
from main.live.reader import ROOM_SKIP, keyword  # noqa: E402

c = collections.Counter()
battle_lines = []
for half in ("ours_challenge", "peer_challenge"):
    p = os.path.join(sys.argv[1], half, "shadow.jsonl")
    for ln in open(p):
        r = json.loads(ln)
        if r.get("kind") != "race":
            continue
        kws = tuple(sorted({keyword(x) for x in r["lines"] if x}))
        c[kws] += 1
        folded = [x for x in r["lines"] if x and keyword(x) not in ROOM_SKIP]
        if folded:
            battle_lines.append({"half": half, "battle": r["battle"], "lines": folded[:5]})
print(json.dumps({"by_keywords": {"|".join(k): n for k, n in c.items()},
                  "races_carrying_battle_protocol": len(battle_lines), "examples": battle_lines[:5]}, indent=1))
