"""Outcome-blind: searched decisions per panel in S'-LIVE's final searched attempts (reads search_decisions.jsonl only)."""
import json, os, sys, collections
cells = sys.argv[1]
c = collections.Counter(); dev = collections.Counter(); games = collections.Counter(); caps = collections.Counter()
for name in sorted(os.listdir(cells)):
    if ".partial." in name:
        continue
    panel = name.split("__")[1]
    p = os.path.join(cells, name, "search_decisions.jsonl")
    if not os.path.exists(p):
        continue
    games[panel] += 1
    if os.path.isdir(os.path.join(cells, name, "capture")) and os.listdir(os.path.join(cells, name, "capture")):
        caps[panel] += 1
    for line in open(p):
        d = json.loads(line)
        if d.get("searched"):
            c[panel] += 1
            dev[panel] += bool(d.get("deviation"))
print(json.dumps({"games": games, "captures": caps, "searched": c, "departures": dev,
                  "total_searched": sum(c.values()), "total_departures": sum(dev.values())}, indent=1))
