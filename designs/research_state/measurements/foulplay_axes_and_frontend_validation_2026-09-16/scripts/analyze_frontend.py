#!/usr/bin/env python3
"""JOB 2 — the front-end vs Node comparison for the SOP tier-B read."""
from __future__ import annotations
import json, math, re
from pathlib import Path
ROOT = Path("/home/goodlad/.claude/jobs/9ab51de6/tmp/fp_axes/frontend")

def wilson(k, n, z=1.959963985):
    if n == 0: return (float("nan"), float("nan"))
    p = k/n; d = 1+z*z/n; c = p+z*z/(2*n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)

def newcombe(k1,n1,k2,n2):
    p1,p2 = k1/n1, k2/n2
    l1,u1 = wilson(k1,n1); l2,u2 = wilson(k2,n2)
    d = p1-p2
    return d, d-math.sqrt((p1-l1)**2+(u2-p2)**2), d+math.sqrt((u1-p1)**2+(p2-l2)**2)

cells = {}
for name in ("fehome","feaway","ndhome","ndaway"):
    d = ROOT/name
    s = json.loads((d/"summary.json").read_text())
    rows = [json.loads(x) for x in (d/"games.jsonl").read_text().splitlines() if x.strip()]
    meta = (ROOT/f"{name}.meta").read_text()
    wall = re.search(r"wall_s=(\d+)", meta)
    cells[name] = {"s": s, "rows": rows, "wall": int(wall.group(1)) if wall else None}

print("| cell | transport | team set | status | n | W/L/T | win rate | Wilson 95% | their argmax | our forfeits | defaults/redecides | distinct our teams | wall s | games/s |")
print("|---|---|---|---|---:|---|---:|---|---|---:|---|---:|---:|---:|")
for name in ("fehome","ndhome","feaway","ndaway"):
    c = cells[name]; s = c["s"]
    lo,hi = wilson(s["wins"], s["n"])
    tr = "ws_frontend" if name.startswith("fe") else "Node showdown"
    ts = "home" if name.endswith("home") else "away"
    print(f"| `{name}` | {tr} | {ts} | {s['status']} | {s['n']} | "
          f"{s['wins']}/{s['losses']}/{s.get('ties',0)} | **{s['wins']/s['n']:.3f}** | "
          f"[{lo:.3f}, {hi:.3f}] | {s.get('their_argmax_match_rates')} | "
          f"{s['hit_forfeit_limit']} | {s['n_defaults']}/{s['n_redecides']} | "
          f"{s['distinct_our_teams']} | {c['wall']} | {s['n']/c['wall']:.3f} |")

print()
for ts, a, b in (("home","fehome","ndhome"), ("away","feaway","ndaway")):
    sa, sb = cells[a]["s"], cells[b]["s"]
    d, lo, hi = newcombe(sa["wins"], sa["n"], sb["wins"], sb["n"])
    v = "AGREE (CI covers 0)" if lo <= 0 <= hi else "DISAGREE"
    print(f"{ts}: front end - Node = {d:+.3f}  Newcombe 95% [{lo:+.3f}, {hi:+.3f}] -> {v}")

print("\npooled:")
fa = sum(cells[n]["s"]["wins"] for n in ("fehome","feaway"))
fn = sum(cells[n]["s"]["n"] for n in ("fehome","feaway"))
na = sum(cells[n]["s"]["wins"] for n in ("ndhome","ndaway"))
nn = sum(cells[n]["s"]["n"] for n in ("ndhome","ndaway"))
d, lo, hi = newcombe(fa, fn, na, nn)
print(f"  front end {fa}/{fn} = {fa/fn:.3f}  vs  Node {na}/{nn} = {na/nn:.3f}; "
      f"delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

print("\nper-game comparability (same half+index, same our_team):")
for ts, a, b in (("home","fehome","ndhome"), ("away","feaway","ndaway")):
    A = {(r["half"], r["index"]): r for r in cells[a]["rows"]}
    B = {(r["half"], r["index"]): r for r in cells[b]["rows"]}
    shared = [k for k in A if k in B]
    same_team = [k for k in shared if A[k]["our_team"] == B[k]["our_team"]]
    agree = sum(1 for k in same_team if A[k]["won"] == B[k]["won"])
    print(f"  {ts}: {len(shared)} shared slots, {len(same_team)} with the SAME our-team, "
          f"outcome agreement on those {agree}/{len(same_team)}"
          + (f" ({agree/len(same_team):.2f})" if same_team else ""))

print("\nturns:")
for name in ("fehome","ndhome","feaway","ndaway"):
    s = cells[name]["s"]
    print(f"  {name}: mean {s['mean_turns']:.1f} median {s['median_turns']} max {s['max_turns']}")

print("\nprovenance:")
for name in ("fehome","ndhome","feaway","ndaway"):
    s = cells[name]["s"]; c = s["cell"]
    print(f"  {name}: server={c['server_uri']} opponent={c['opponent']}:{c['opponent_agent']} "
          f"commit={c['opponent_commit']} pin={c['showdown_pin']} step={c['model_step']} "
          f"teams ours={c['our_team_count']} theirs={c['their_team_count']} "
          f"asym={s['team_source_asymmetry']}")
