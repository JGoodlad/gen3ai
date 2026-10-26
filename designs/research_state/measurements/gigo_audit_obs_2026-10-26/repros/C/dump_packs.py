"""Dump {src, id, paste, packed} for every pool + corpus team, packed by OUR training path
(parse_showdown_team -> fix_gen3_hp_ivs -> join_team), no validation (Node does it in the oracle)."""
import json, gzip, sys
from utils.team_loader import TeamLoader
from utils.team_packing import Teambuilder
from utils.gen3_utils import fix_gen3_hp_ivs
from utils.team_corpora import CORPORA, DATA_NAME
out = []
tl = TeamLoader()
for k, (f, t) in enumerate(tl._loaded):
    out.append({"src": "pool", "id": f, "paste": t})
for name, d in CORPORA.items():
    for line in gzip.open(d / DATA_NAME, "rt"):
        r = json.loads(line)
        out.append({"src": name, "id": r["id"], "paste": r["text"]})
for r in out:
    try:
        r["packed"] = Teambuilder.join_team(fix_gen3_hp_ivs(Teambuilder.parse_showdown_team(r["paste"])))
    except Exception as e:
        r["packed"] = None; r["pack_error"] = repr(e)
json.dump(out, open(sys.argv[1], "w"))
print(len(out), sum(1 for r in out if r["packed"] is None), "pack errors")
