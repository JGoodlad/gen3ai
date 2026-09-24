"""EXPLORATORY (NOT pre-registered) arm "five": the opponent pilots one of the 5 sample offense teams that
BOTH stable opponents of ai_v13_22_popr1_loop pin (train/stable_fraction 0.36), cycled i % 5. Same
trainee teams, seeds and pilot as the registered arms. Writes <work>/manifest_five.json.

    python prep_five_arm.py <work_dir>
"""
import hashlib
import json
import os
import sys

from utils.teambuilder import Gen3Teambuilder

FIVE = ["data/teams/sample/9eb3abdc52876a63.txt", "data/teams/sample/ac17a9dde5.txt",
        "data/teams/sample/a185b2d193.txt", "data/teams/sample/9ba039ba8a.txt",
        "data/teams/sample/b904dbe059.txt"]

work = sys.argv[1]
man = json.load(open(os.path.join(work, "manifest_full.json")))
texts = [open(f).read() for f in FIVE]
tb = Gen3Teambuilder(texts)
assert len(tb.packed_teams) == 5
shas = [hashlib.sha1(t.strip().encode()).hexdigest()[:10] for t in texts]
assert shas == tb._pool_keys
for b in man["battles"]:
    b["opp"]["five"] = shas[b["i"] % 5]
for s, p in zip(shas, tb.packed_teams):
    man["packed"][s] = p
json.dump(man, open(os.path.join(work, "manifest_five.json"), "w"))
print(dict(zip(FIVE, shas)))
