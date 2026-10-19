import copy, gzip, json, sys
from pathlib import Path
from utils import gen3_utils as g
from utils.teambuilder import Teambuilder
new = copy.deepcopy(g.GEN3_HP_IVS)
old = copy.deepcopy(new); old["Ghost"] = {"hp": 31, "atk": 30, "def": 31, "spa": 31, "spd": 30, "spe": 31}
def run(text, table):
    g.GEN3_HP_IVS.clear(); g.GEN3_HP_IVS.update(table)
    mons = Teambuilder.parse_showdown_team(text.strip())
    return [(m.nickname, tuple(m.ivs)) for m in g.fix_gen3_hp_ivs(mons)]
def diff(text):
    a = run(text, old); b = run(text, new)
    assert len(a) == len(b)
    return [(x[0], x[1], y[1]) for x, y in zip(a, b) if x != y]
tot = {}
for label, items in (
    ("data/teams", [(str(p), p.read_text()) for p in sorted(Path("data/teams").rglob("*.txt"))]),
    ("w_forum_v1", [(r["id"], r["text"]) for r in map(json.loads, gzip.open("src/utils/team_corpora/w_forum_v1/teams.jsonl.gz", "rt"))]),
):
    teams = 0; mons = 0; hit = []
    for name, text in items:
        teams += 1
        try: d = diff(text)
        except Exception as e: print("ERR", name, e); continue
        if d: hit.append((name, d)); mons += len(d)
    print(f"{label}: {len(hit)} of {teams} teams affected, {mons} mons")
    for name, d in hit: print("  ", name, [(n, o, w) for n, o, w in d][:1], len(d))
