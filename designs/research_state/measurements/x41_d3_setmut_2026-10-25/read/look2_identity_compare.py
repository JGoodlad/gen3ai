"""D3 look-2 pin IDENTITY check (scratch tool, read-only on both cells): each re-played unit against its banked E1 unit.

Per game (matched by battle tag): the 5-tuple the read's guard compares (half, index, our_team, result, turns); the
capture's seed and START command (both packed teams); OUR choices and THEIR choices in order (every CHOOSE of each
player, the choice-level comparison); every per-side protocol chunk with the `|t:|` timestamps removed (byte identity).
Also each unit's anchor regime id (main.reads.assemble.anchor_regime_id, the pin's code) and engine stamp.

usage: python identity_compare.py <bank cell> <replay cell> [...pairs]  (PYTHONPATH = the pin's src)
"""
import json
import re
import sys
from pathlib import Path

from main.reads.assemble import anchor_regime_id
from main.reads.engine import unit_engine

T = re.compile(r"^\|t:\|\d+\n?", re.M)


def games(cell: Path):
    return {g["battle_tag"]: g for g in map(json.loads, open(cell / "games.jsonl"))}


def five(g):
    return (g["half"], g["index"], g["our_team"], g["result"], g["turns"])


def cap(cell: Path, tag: str):
    return json.loads((cell / "capture" / f"{tag}.json").read_text())


def ours(c):
    st = json.loads(c["commands"][0][6:])
    return "p1" if st["p1"]["name"].startswith("Gen3AI") else "p2"


def choices(c):
    d = {"p1": [], "p2": []}
    for x in c["commands"]:
        p = x.split(" ", 2)
        if p[0] == "CHOOSE":
            d[p[1]].append(p[2])
    return d


def chunks(c):
    return [(w, T.sub("", t)) for w, t in c["chunks"]]


def compare(a: Path, b: Path):
    ga, gb = games(a), games(b)
    sa, sb = json.loads((a / "summary.json").read_text()), json.loads((b / "summary.json").read_text())
    out = {"bank": str(a), "replay": str(b), "games": len(ga), "replay_games": len(gb),
           "regime_bank": anchor_regime_id(sa), "regime_replay": anchor_regime_id(sb),
           "engine_bank": unit_engine(sa.get("cell", sa)), "engine_replay": unit_engine(sb.get("cell", sb)),
           "five_equal": 0, "start_equal": 0, "our_choices_equal": 0, "their_choices_equal": 0,
           "protocol_bytes_equal": 0, "diffs": []}
    for tag in sorted(ga, key=lambda t: int(t.rsplit("-", 1)[1])):
        if tag not in gb:
            out["diffs"].append({"tag": tag, "why": "missing in replay"})
            continue
        ca, cb = cap(a, tag), cap(b, tag)
        o = ours(ca)
        th = "p2" if o == "p1" else "p1"
        pa, pb = choices(ca), choices(cb)
        row = {"five": five(ga[tag]) == five(gb[tag]),
               "start": ca["seed"] == cb["seed"] and ca["commands"][0] == cb["commands"][0],
               "ours": pa[o] == pb[o], "theirs": pa[th] == pb[th], "bytes": chunks(ca) == chunks(cb)}
        for k, n in (("five", "five_equal"), ("start", "start_equal"), ("ours", "our_choices_equal"),
                     ("theirs", "their_choices_equal"), ("bytes", "protocol_bytes_equal")):
            out[n] += int(row[k])
        if not all(row.values()):
            first = next((i for i in range(min(len(pa[o]), len(pb[o]))) if pa[o][i] != pb[o][i]), None)
            out["diffs"].append({"tag": tag, **row, "bank5": five(ga[tag]), "replay5": five(gb[tag]),
                                 "first_our_choice_diff": None if first is None else
                                 [first, pa[o][first], pb[o][first]]})
    return out


def main(argv):
    pairs = list(zip(argv[0::2], argv[1::2]))
    res = [compare(Path(a), Path(b)) for a, b in pairs]
    tot = {k: sum(r[k] for r in res) for k in ("games", "five_equal", "start_equal", "our_choices_equal",
                                                "their_choices_equal", "protocol_bytes_equal")}
    print(json.dumps({"units": res, "total": tot}, indent=1, default=str))
    return 0 if all(not r["diffs"] and r["regime_bank"] == r["regime_replay"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
