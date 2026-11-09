"""Builds src/main/reads/timer_fixtures/banked_rows.json.gz: the clock rows (turn, searched, excluded, wall) of a
fixed outcome-blind subset of S'-HONEST's banked searched games (every 40th cell name per panel in sorted order, 30 a
panel), and the PROTOTYPE's timer block on them (timer_replay.py's simulate). Reads decision logs only (symlinked
into a scratch dir so the prototype's loader reads nothing else).

    python3 make_timer_fixture.py   # run from anywhere; WT below is the checkout"""
import collections, gzip, importlib.util, json, os
WT = os.environ.get("WT", os.getcwd())
spec = importlib.util.spec_from_file_location("proto", WT + "/designs/research_state/measurements/searched_cost_2026-11-09/inputs/timer_replay.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
cells = "/home/goodlad/gen3ai_archive/sprime_honest_play_2026-11-06/anchor/u/pstar_s/cells"
names = sorted(n for n in os.listdir(cells) if P.PAT.match(n))
by = collections.defaultdict(list)
for n in names:
    by[P.PAT.match(n).group(1)].append(n)
pick = []
for panel, ns in sorted(by.items()):
    pick += ns[::40][:30]
tmp = os.environ.get("FX_TMP", "/tmp/timer_fixture_cells")
os.makedirs(tmp, exist_ok=True)
for n in pick:
    d = os.path.join(tmp, n); os.makedirs(d, exist_ok=True)
    src = os.path.join(cells, n, "search_decisions.jsonl")
    if not os.path.exists(os.path.join(d, "search_decisions.jsonl")):
        os.symlink(src, os.path.join(d, "search_decisions.jsonl"))
games = P.load(tmp)
out = {"source": "S'-HONEST banked searched cells (sprime_honest_play_2026-11-06), outcome-blind: decision logs only",
       "games": [{"cell": n, "panel": g["panel"], "rows": [[r["turn"], int(r["searched"]), int(r["excluded"]), r["wall"]] for r in g["rows"]]}
                 for n, g in zip(sorted(pick), games)]}
# the prototype's own timer block on these games
timer = {}
for name, kw in P.POLICIES.items():
    for bg in (False, True):
        if (name == "as_played" or kw.get("off_after_cut")) and bg:
            continue
        acc = collections.defaultdict(collections.Counter)
        nsearch = collections.Counter()
        for g in games:
            s = P.simulate(g["rows"], background=bg, **kw)
            for p in (g["panel"], "U"):
                acc[p]["games"] += 1
                for k, v in s.items():
                    acc[p][k] += v
        for g in games:
            for p in (g["panel"], "U"):
                nsearch[p] += sum(1 for r in g["rows"] if r["searched"])
        timer[name + ("+bg" if bg else "")] = {p: {"forfeit_share": round(a["forfeit"] / a["games"], 4),
            "cut_share_of_search_path": round(a["cut"] / a["path"], 4),
            "cut_share_of_searched": round(a["cut_searched"] / nsearch[p], 4),
            "battles_with_a_cut": round(a["battle_cut"] / a["games"], 4)} for p, a in sorted(acc.items())}
out["prototype_timer"] = timer
with gzip.open(WT + "/src/main/reads/timer_fixtures/banked_rows.json.gz", "wt") as f:
    json.dump(out, f, sort_keys=True, separators=(",", ":"))
print(len(out["games"]), {p: timer["as_played"][p]["forfeit_share"] for p in timer["as_played"]}, timer["bank_r10+bg"]["U"])
