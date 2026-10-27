"""X35 shared play pin IDENTITY check (read-only on both cells): each re-played unit against a reference unit (a banked
E1 unit, or the same unit played at the D3 look-2 pin c4a5c37b6), game for game.

Per game (matched by battle tag): the 5-tuple the read's guard compares (half, index, our_team, result, turns); OUR
choices and THEIR choices in order (every CHOOSE of each player); every per-side protocol chunk with the `|t:|`
timestamps removed; the seed; and whether either side's START team carries Frustration (forum72's sft431). START
strings are compared but EXPECTED to differ: since ce50227a5 the front end plays the validator's re-pack (field for
field the same sets for our side; `repack_check.json`). Choice / protocol comparisons need a capture on both sides; a
reference without one is compared on the 5-tuple only. Also each unit's anchor regime id (main.reads.assemble, the
pin's code), engine, model_loader and peer pack state.

usage: python identity_compare.py <ref cell> <replay cell> [...pairs]   (PYTHONPATH = the pin's src)
"""
import json
import re
import sys
from pathlib import Path

from main.reads.assemble import anchor_regime_id
from main.reads.engine import unit_engine

T = re.compile(r"^\|t:\|\d+\n?", re.MULTILINE)


def games(cell):
    return {g["battle_tag"]: g for g in map(json.loads, open(cell / "games.jsonl"))}


def five(g):
    return (g["half"], g["index"], g["our_team"], g["result"], g["turns"])


def cap(cell, tag):
    p = cell / "capture" / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def start(c):
    return json.loads(c["commands"][0][6:])


def ours(c):
    return "p1" if start(c)["p1"]["name"].startswith("Gen3AI") else "p2"


def choices(c):
    d = {"p1": [], "p2": []}
    for x in c["commands"]:
        p = x.split(" ", 2)
        if p[0] == "CHOOSE":
            d[p[1]].append(p[2])
    return d


def chunks(c):
    return [(w, T.sub("", t)) for w, t in c["chunks"]]


def meta(s):
    c = s.get("cell", s)
    return {"regime": anchor_regime_id(s), "engine": unit_engine(c), "model_loader": c.get("model_loader"),
            "peer_pack_changed": c.get("peer_pack_changed"), "server_version": c.get("server_version"),
            "peer_clean": s.get("peer_clean"), "n": s.get("n"), "wins": s.get("wins")}


def compare(a, b):
    ga, gb = games(a), games(b)
    sa, sb = json.loads((a / "summary.json").read_text()), json.loads((b / "summary.json").read_text())
    out = {"ref": str(a), "replay": str(b), "games": len(ga), "replay_games": len(gb), "ref_meta": meta(sa),
           "replay_meta": meta(sb), "five_equal": 0, "captured_pairs": 0, "seed_equal": 0, "start_string_equal": 0,
           "our_choices_equal": 0, "their_choices_equal": 0, "protocol_bytes_equal": 0, "frustration_games": [],
           "diffs": []}
    for tag in sorted(ga, key=lambda t: int(t.rsplit("-", 1)[1])):
        if tag not in gb:
            out["diffs"].append({"tag": tag, "why": "missing in replay"})
            continue
        row = {"five": five(ga[tag]) == five(gb[tag])}
        out["five_equal"] += int(row["five"])
        ca, cb = cap(a, tag), cap(b, tag)
        if cb is not None and any("frustration" in start(cb)[s]["team"].lower() for s in ("p1", "p2")):
            out["frustration_games"].append(tag)
        if ca is not None and cb is not None:
            out["captured_pairs"] += 1
            o = ours(ca)
            th = "p2" if o == "p1" else "p1"
            pa, pb = choices(ca), choices(cb)
            row.update({"seed": ca["seed"] == cb["seed"], "start_string": ca["commands"][0] == cb["commands"][0],
                        "ours": pa[o] == pb[o], "theirs": pa[th] == pb[th], "bytes": chunks(ca) == chunks(cb)})
            for k, n in (("seed", "seed_equal"), ("start_string", "start_string_equal"), ("ours", "our_choices_equal"),
                         ("theirs", "their_choices_equal"), ("bytes", "protocol_bytes_equal")):
                out[n] += int(row[k])
            if not all(v for k, v in row.items() if k != "start_string"):
                first = next((i for i in range(min(len(pa[o]), len(pb[o]))) if pa[o][i] != pb[o][i]), None)
                firstt = next((i for i in range(min(len(pa[th]), len(pb[th]))) if pa[th][i] != pb[th][i]), None)
                out["diffs"].append({"tag": tag, **row, "ref5": five(ga[tag]), "replay5": five(gb[tag]),
                                     "first_our_choice_diff": None if first is None else [first, pa[o][first], pb[o][first]],
                                     "first_their_choice_diff": None if firstt is None else
                                     [firstt, pa[th][firstt], pb[th][firstt]]})
        elif not row["five"]:
            out["diffs"].append({"tag": tag, **row, "ref5": five(ga[tag]), "replay5": five(gb[tag])})
    return out


def main(argv):
    res = [compare(Path(a), Path(b)) for a, b in zip(argv[0::2], argv[1::2])]
    tot = {k: sum(r[k] for r in res) for k in ("games", "five_equal", "captured_pairs", "seed_equal",
                                                "start_string_equal", "our_choices_equal", "their_choices_equal",
                                                "protocol_bytes_equal")}
    print(json.dumps({"units": res, "total": tot}, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
