"""S'-HONEST design input 2 (2026-11-05, OUTCOME-BLIND): the EXPOSURE of S'-LIVE's search-path decisions to the
persisted-dice leak (L2 of design_sprime_honest.md): at each decision, which hidden random counters or pre-rolled dice
were live, read from the searched side's OWN protocol stream (the capture's chunks) and the START teams (Quick Claw).
A decision is aligned to the n-th non-wait request of our stream (n is 0-based; checked against the records' turns).
Sleep from Rest (fixed duration) is excluded. Reads no result (never games.jsonl or a summary).

    python3 exposure.py <S'-LIVE state>/anchor/u/pstar_s/cells exposure.json
"""
import json, glob, collections, sys, re, os

TRAP = ("Wrap", "Bind", "Fire Spin", "Whirlpool", "Sand Tomb", "Clamp")
LOCK = ("Outrage", "Thrash", "Petal Dance")

def toid(s): return re.sub(r"[^a-z0-9]", "", s.lower())

def ident(tok):
    side, _, name = tok.partition(": ")
    return side[:2], name

def team_items(packed):
    out = {}
    for e in packed.split("]"):
        f = e.split("|")
        if len(f) > 2:
            out[f[0] or f[1]] = toid(f[2])
    return out

def scan(cell):
    caps = glob.glob(cell + "/capture/*.json")
    recf = cell + "/search_decisions.jsonl"
    if not caps or not os.path.exists(recf):
        return None
    d = json.load(open(caps[0]))
    recs = [json.loads(l) for l in open(recf)]
    side = recs[0]["side"]; opp = "p2" if side == "p1" else "p1"
    start = json.loads(d["commands"][0][len("START "):])
    items = {s: team_items(start[s]["team"]) for s in ("p1", "p2")}
    slp = {}          # (side, name) -> "rand" | "rest"
    vol = {"p1": set(), "p2": set()}
    active = {"p1": None, "p2": None}
    fs = {"p1": 0, "p2": 0}
    snaps = []
    txt = "\n".join(t for s, t in d["chunks"] if s == side)
    for l in txt.split("\n"):
        p = l.split("|")
        if len(p) < 2:
            continue
        k = p[1]
        if k == "request" and len(p) > 2 and p[2].strip():
            r = json.loads("|".join(p[2:]))
            if r.get("wait") or r.get("teamPreview"):
                continue
            sl = {s: [n for (ss, n), v in slp.items() if ss == s and v == "rand"] for s in ("p1", "p2")}
            snaps.append({
                "slp_ours_active": active[side] in sl[side],
                "slp_ours_any": bool(sl[side]),
                "slp_opp_active": active[opp] in sl[opp],
                "slp_opp_any": bool(sl[opp]),
                "conf": bool({"confusion"} & (vol[side] | vol[opp])),
                "encore_disable": bool({"encore", "disable"} & (vol[side] | vol[opp])),
                "trap_lock_uproar": bool({"trap", "lock", "uproar", "bide"} & (vol[side] | vol[opp])),
                "fs_pending": bool(fs["p1"] or fs["p2"]),
                "qc_ours_active": items[side].get(active[side] or "", "") == "quickclaw",
                "qc_opp_true_active": items[opp].get(active[opp] or "", "") == "quickclaw",
            })
            continue
        if k in ("switch", "drag") and len(p) > 4:
            s, n = ident(p[2]); active[s] = n; vol[s] = set()
            hp = p[4]
            if hp.endswith(" slp") and (s, n) not in slp:
                slp[(s, n)] = "rand"
            if not hp.endswith(" slp"):
                slp.pop((s, n), None)
        elif k == "faint" and len(p) > 2:
            s, n = ident(p[2]); vol[s] = set(); slp.pop((s, n), None)
        elif k == "-status" and len(p) > 3 and p[3] == "slp":
            s, n = ident(p[2]); slp[(s, n)] = "rest" if "move: Rest" in l else "rand"
        elif k == "-curestatus" and len(p) > 3 and p[3] == "slp":
            s, n = ident(p[2]); slp.pop((s, n), None)
        elif k == "-start" and len(p) > 3:
            s, n = ident(p[2]); what = p[3].replace("move: ", "")
            if what == "confusion": vol[s].add("confusion"); vol[s].discard("lock")
            elif what in ("Encore", "Disable", "Uproar", "Bide"): vol[s].add(toid(what))
            elif what in ("Future Sight", "Doom Desire"): fs[s] = 3
        elif k == "-end" and len(p) > 3:
            s, n = ident(p[2]); what = p[3].replace("move: ", "")
            if what == "confusion": vol[s].discard("confusion")
            elif what in ("Encore", "Disable", "Uproar", "Bide"): vol[s].discard(toid(what))
            elif what in TRAP: vol[s].discard("trap")
            elif what in ("Future Sight", "Doom Desire"): fs[s] = 0
        elif k == "-activate" and len(p) > 3 and p[3].replace("move: ", "") in TRAP:
            s, n = ident(p[2]); vol[s].add("trap")
        elif k == "move" and len(p) > 3 and p[3] in LOCK:
            s, n = ident(p[2]); vol[s].add("lock")
        elif k == "turn":
            for s in ("p1", "p2"):
                if fs[s]: fs[s] -= 1
    out = []
    for r in recs:
        if r.get("why") in ("before_turn", "one_legal"):
            continue
        sn = snaps[r["n"]] if r["n"] < len(snaps) else None
        if sn is None:
            continue
        out.append((r["key"].split("|")[1], r.get("searched", False), r["turn"], sn))
    return out

root, out_path = sys.argv[1], sys.argv[2]
tot = collections.Counter(); n = 0; games_with = collections.Counter(); g = 0
bypanel = collections.defaultdict(collections.Counter); npanel = collections.Counter()
for cell in sorted(c for c in glob.glob(root + "/pstar_s__*") if ".partial." not in c):
    res = scan(cell)
    if res is None:
        continue
    g += 1
    panel = cell.split("__")[1]
    seen = collections.Counter()
    for deal, srch, turn, sn in res:
        n += 1; npanel[panel] += 1
        anyc = False
        for k, v in sn.items():
            if v:
                tot[k] += 1; bypanel[panel][k] += 1; seen[k] = 1
                if k not in ("qc_opp_true_active",):
                    anyc = True
        if anyc:
            tot["ANY"] += 1; bypanel[panel]["ANY"] += 1; seen["ANY"] = 1
    for k in seen: games_with[k] += 1
rec = {"games": g, "decisions": n,
       "share": {k: round(v / n, 4) for k, v in sorted(tot.items())},
       "games_share": {k: round(v / g, 4) for k, v in sorted(games_with.items())},
       "by_panel": {p: {k: round(v / npanel[p], 4) for k, v in sorted(c.items())} for p, c in bypanel.items()}}
json.dump(rec, open(out_path, "w"), indent=1, sort_keys=True)
print(json.dumps(rec, indent=1, sort_keys=True))
