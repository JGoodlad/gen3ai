"""GIGO audit B: an INDEPENDENT mask from each p1 |request| JSON (Showdown semantics, written here from
sim/side.ts + sim/pokemon.ts, not from legal.rs) vs the mask the core wrote at that decision; plus a
coverage census of the request shapes the battles reached (locks, disables, traps, Struggle, forced).

Usage: mask_vs_request.py <games json written by mask_vs_engine.py> [...]
"""
import json
import sys
from collections import Counter

from utils.rust_env import ffi

lib = ffi.load(ffi.default_path("release"))
cov = Counter()
dis = Counter()
examples = {}
n_rows = n_req = 0
for path in sys.argv[1:]:
    games = json.load(open(path))["games"]
    for gi, g in enumerate(games):
        t = ffi.trace(lib, g["fin"]["script"], commit="a", label="x")
        reqs = []
        for ln in t["records"]["p1"].splitlines()[1:]:
            text = json.loads(ln)["text"]
            if text.startswith("|request|"):
                reqs.append(json.loads(text[len("|request|"):]))
        team_order = [p["ident"] for p in reqs[0]["side"]["pokemon"]]
        dreqs = [r for r in reqs if not r.get("wait")]
        rows = g["rows"]
        n_req += len(dreqs)
        n_rows += len(rows)
        # the stall forfeit: p1's decision at turn >= limit opens a request but is never fed
        if len(dreqs) != len(rows):
            cov[f"ALIGN rows {len(rows)} != decision requests {len(dreqs)} (forfeit {g['fin']['forfeit']})"] += 1
            if len(dreqs) < len(rows):
                continue
        for k, (req, row) in enumerate(zip(dreqs, rows)):
            exp = [0] * 11
            side = {p["ident"]: p for p in req["side"]["pokemon"]}
            fs = bool(req.get("forceSwitch", [False])[0])
            act = (req.get("active") or [None])[0]
            trapped = bool(act and act.get("trapped"))
            if not trapped:
                for i, ident in enumerate(team_order):
                    p = side[ident]
                    if not p["active"] and not p["condition"].endswith(" fnt") and p["condition"] != "0 fnt":
                        exp[i] = 1
            if act and not fs:
                moves = act["moves"]
                ids = [m["id"] for m in moves]
                if ids == ["struggle"]:
                    exp[10] = 1
                    cov["struggle-only"] += 1
                else:
                    for i, m in enumerate(moves[:4]):
                        if not m.get("disabled"):
                            exp[6 + i] = 1
                if len(moves) == 1 and ids != ["struggle"]:
                    cov[f"single-entry request ({'recharge' if ids == ['recharge'] else 'locked'})"] += 1
                if any(m.get("disabled") for m in moves):
                    cov["some move disabled"] += 1
                if any(m.get("pp") == 0 for m in moves):
                    cov["some move at 0 PP"] += 1
                if trapped:
                    cov["trapped"] += 1
                if act.get("maybeTrapped"):
                    cov["maybeTrapped"] += 1
                if req.get("update"):
                    cov["update re-request"] += 1
            if fs:
                cov["forceSwitch"] += 1
            if exp != row["mask"]:
                key = ("force" if fs else "move") + f" exp={exp} got={row['mask']}"
                dis[key] += 1
                examples.setdefault(key, (path, gi, k, row["turn"], json.dumps(req)[:600]))
print(f"rows {n_rows}  decision requests {n_req}")
print("coverage:")
for k, v in sorted(cov.items()):
    print("  ", v, k)
print("DISAGREEMENTS:", sum(dis.values()))
for k, v in dis.most_common(20):
    print("  ", v, k)
    print("     ", examples[k])
