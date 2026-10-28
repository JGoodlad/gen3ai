"""FROZEN record (X61P look 1, PREREG §7): C0's re-played forum72 units 0-6 against D1's banked C0 units, game for
game (half, index, our_team, result, turns), and the one uncounted unit's replay against its first play.
Reads the look-1 state and D1's archive read-only; prints JSON. Not a tool: never re-run as one."""
import json, os, sys

S = os.path.expanduser("~/gen3ai_archive/x61p_n1p_read/look1/anchor")
D1 = os.path.expanduser("~/gen3ai_archive/x35_d1_read")


def games(d):
    p = os.path.join(d, "games.jsonl")
    rows = [json.loads(l) for l in open(p)]
    return sorted(((g.get("half"), g.get("index"), g.get("our_team"), g.get("result"), g.get("turns")) for g in rows),
                  key=lambda x: (str(x[0]), x[1]))


out = {"units": [], "drift": {}, "p72_replay": {}}
same = tot = wm = wb = 0
for s in (3501, 3502, 3503):
    for u in range(7):
        name = f"c0_s{s}__forum72__{u:02d}"
        mine = games(f"{S}/primary/c0_s{s}/cells/{name}")
        banked = games(f"{D1}/h_a/cells/{name}")
        eq = sum(a == b for a, b in zip(mine, banked))
        w1 = sum(g[3] == "win" for g in mine); w2 = sum(g[3] == "win" for g in banked)
        out["units"].append({"unit": name, "identical": eq, "games": len(mine), "wins_now": w1, "wins_banked": w2})
        same += eq; tot += len(mine); wm += w1; wb += w2
out["drift"] = {"identical": same, "games": tot, "wins_now": wm, "wins_banked": wb, "banked_source": "x35_d1_read/h_a"}
base = f"{S}/reported/n1p_s3503/cells/n1p_s3503__p72__03"
a, b = games(base + ".partial.1793091066"), games(base)
out["p72_replay"] = {"unit": "n1p_s3503__p72__03", "identical": sum(x == y for x, y in zip(a, b)), "games": len(a),
                     "differ": [{"first": x, "counted_replay": y} for x, y in zip(a, b) if x != y]}
json.dump(out, sys.stdout, indent=1)
