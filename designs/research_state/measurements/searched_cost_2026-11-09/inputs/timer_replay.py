"""Q19 (searched-player cost and deployability): S'-HONEST's searched games replayed through Showdown's battle timer,
and its exclusions classified. OUTCOME-BLIND: it reads only each counted searched unit's ``search_decisions.jsonl``
(cost and build fields; never ``games.jsonl``, a summary, a capture or a result), never a ``.partial.<epoch>`` dir.

    python3 timer_replay.py CELLS_DIR OUT.json
    # CELLS_DIR = ~/gen3ai_archive/sprime_honest_play_2026-11-06/anchor/u/pstar_s/cells

THE TIMER MODEL (``deps/pokemon-showdown/server/room-battle.ts`` at the submodule's e0551883f, ``RoomBattleTimer``,
a NON-challenge singles battle and a format with no ``timer`` rule, as ``[Gen 3] OU``): the bank starts at
STARTING_TIME 150 + STARTING_GRACE_TIME 60; each NEW turn adds ``addPerTurn`` 10 s (``accelerate``: 5 s past turn 100,
and past turn 200 0 s on alternate requests, modelled as odd turns), capped at 150; a mid-turn request (only one
player asked) adds min(addPerTurn, TICK_TIME) = 5 s to both, uncapped; a request's own limit is min(bank,
MAX_TURN_TIME 150); time is charged in TICK_TIME = 5-s ticks (a 4.9-s answer costs 0); a request that reaches its limit
forfeits. A challenge battle (300 / 300, no accelerate) is looser.

APPROXIMATIONS (each conservative or neutral, stated in ``designs/endstate/design_searched_player.md`` §2): our
response time = the decision's search wall (searched or excluded) + OVERHEAD_S for every decision; the opponent's own
mid-turn requests (which also add 5 s to us) are unseen; a decision cut by a deadline is assumed to leave the rest
of the game's costs unchanged (after a cut the game is counterfactual). Walls are S'-HONEST's: one CPU a unit on the
whole box at K = 14 with both SMT siblings busy.

POLICIES: ``as_played`` (no deadline); ``fixed_D`` (fall back to the policy's pick at D s); ``bank_r10`` (fall back at
min(bank, 150) - 10 - OVERHEAD_S: the bank-aware deadline); each also ``+bg``: the cut decision's build runs on (so
the carry stays a function of I) and delays the next search-path decision; ``bank_r10_off``: the bank-aware deadline,
and the battle's search switched OFF (the policy plays) from its first cut on.
"""
import collections
import json
import math
import os
import re
import sys

try:
    import orjson
    _loads = orjson.loads
except ImportError:  # pragma: no cover
    _loads = json.loads

PAT = re.compile(r"^pstar_s__(forum72|heldout72_hpfix|p72)__(\d+)__g(\d+)$")
OVERHEAD_S = 1.0
STARTING, GRACE, MAX_TURN, ADD, TICK = 150, 60, 150, 10, 5
POLICIES = {"as_played": {}, **{f"fixed_{d}": {"deadline": d} for d in (150, 60, 30, 20, 15, 10, 5)},
            "bank_r10": {"reserve": 10}, "bank_r10_fixed_15": {"reserve": 10, "deadline": 15},
            "bank_r10_off": {"reserve": 10, "off_after_cut": True}}


def add_for_turn(t):
    if t <= 100:
        return ADD
    if t <= 200:
        return ADD - TICK
    return 0 if t % 2 else ADD - TICK


def excl_class(err):
    if "stuck" in err and "probe" in err:
        fails = err[err.find("{"):]
        return "stuck_probe_revised_only" if "'extend':" not in fails and "extend_revised" in fails else "stuck_probe_other"
    if "cold build waits" in err:
        return "cold_waits"
    if "no honest world fits" in err:
        return "cold_failed"
    m = re.search(r"(\d+) of (\d+) (?:incremental )?worlds not built", err)
    if m:
        built = int(m.group(2)) - int(m.group(1))
        return "warm_none_built" if built == 0 else "warm_some_built"
    return "other"


def load(cells):
    games = []
    for name in sorted(os.listdir(cells)):
        m = PAT.match(name)
        p = os.path.join(cells, name, "search_decisions.jsonl")
        if not m or not os.path.isfile(p):
            continue
        rows = []
        with open(p, "rb") as f:
            for ln in f:
                if not ln.strip():
                    continue
                r = _loads(ln)
                excluded = r.get("why") == "excluded"
                rows.append({"turn": int(r["turn"]), "searched": bool(r.get("searched")), "excluded": excluded,
                             "wall": float(r.get("wall_s") or 0.0), "cpu": float(r.get("cpu_s") or 0.0),
                             "world": float(r.get("world_cpu_s") or 0.0),
                             "playout": float(r.get("playout_cpu_s") or 0.0),
                             "xclass": excl_class(r.get("error") or "") if excluded else None})
        games.append({"panel": m.group(1), "rows": rows})
    return games


def simulate(rows, deadline=None, reserve=None, background=False, off_after_cut=False):
    bank, prev, cut, cut_searched, path, backlog = float(STARTING + GRACE), None, 0, 0, 0, 0.0
    off = False
    for r in rows:
        if prev is not None:
            bank = min(bank + add_for_turn(r["turn"]), float(STARTING)) if r["turn"] > prev else bank + TICK
        prev = r["turn"]
        limit = min(bank, float(MAX_TURN))
        if (r["searched"] or r["excluded"]) and off:
            path += 1
            cut += 1
            cut_searched += int(r["searched"])
            resp = OVERHEAD_S
        elif r["searched"] or r["excluded"]:
            path += 1
            need = backlog + r["wall"]
            d = math.inf if deadline is None else float(deadline)
            if reserve is not None:
                d = min(d, max(0.0, limit - reserve - OVERHEAD_S))
            if need > d:
                cut += 1
                cut_searched += int(r["searched"])
                resp, backlog = d + OVERHEAD_S, ((need - d) if background else 0.0)
                off = off_after_cut
            else:
                resp, backlog = need + OVERHEAD_S, 0.0
        else:
            resp, backlog = OVERHEAD_S, max(0.0, backlog - OVERHEAD_S)
        if resp >= limit:
            return {"forfeit": 1, "cut": cut, "cut_searched": cut_searched, "path": path, "battle_cut": int(cut > 0)}
        bank -= TICK * math.floor(resp / TICK)
    return {"forfeit": 0, "cut": cut, "cut_searched": cut_searched, "path": path, "battle_cut": int(cut > 0)}


def quant(xs, p):
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main(cells, out_path):
    games = load(cells)
    panels = sorted({g["panel"] for g in games}) + ["U"]
    out = {"schema": "q19_timer_replay_v1", "games": {P: sum(1 for g in games if P in (g["panel"], "U"))
                                                      for P in panels}}
    walls = collections.defaultdict(list)
    cpu = collections.defaultdict(lambda: collections.Counter())
    for g in games:
        for r in g["rows"]:
            if not (r["searched"] or r["excluded"]):
                continue
            kind = "searched" if r["searched"] else "excluded"
            for P in (g["panel"], "U"):
                walls[(P, kind)].append(r["wall"])
                walls[(P, "path")].append(r["wall"])
                cpu[P][kind + "_cpu_s"] += r["cpu"]
                cpu[P][kind + "_world_cpu_s"] += r["world"]
                cpu[P][kind + "_playout_cpu_s"] += r["playout"]
                cpu[P][kind + "_n"] += 1
    out["wall_s"] = {f"{P}/{k}": {"n": len(xs), "mean": round(sum(xs) / len(xs), 2), "p50": round(quant(xs, .5), 2),
                                  "p90": round(quant(xs, .9), 2), "p99": round(quant(xs, .99), 1),
                                  "p999": round(quant(xs, .999), 1), "max": round(max(xs), 1),
                                  **{f"share_over_{t}s": round(sum(1 for x in xs if x > t) / len(xs), 4)
                                     for t in (5, 10, 30, 60, 150)}}
                     for (P, k), xs in sorted(walls.items())}
    out["cpu_s_per_game"] = {P: {k: (v if k.endswith("_n") else round(v / out["games"][P], 1)) for k, v in c.items()}
                             for P, c in cpu.items()}
    timer = {}
    for name, kw in POLICIES.items():
        for bg in (False, True):
            if (name == "as_played" or kw.get("off_after_cut")) and bg:
                continue
            acc = collections.defaultdict(collections.Counter)
            for g in games:
                s = simulate(g["rows"], background=bg, **kw)
                for P in (g["panel"], "U"):
                    acc[P]["games"] += 1
                    for k, v in s.items():
                        acc[P][k] += v
            timer[name + ("+bg" if bg else "")] = {
                P: {"forfeit_share": round(a["forfeit"] / a["games"], 4),
                    "cut_share_of_search_path": round(a["cut"] / a["path"], 4),
                    "cut_share_of_searched": round(a["cut_searched"] / cpu[P]["searched_n"], 4),
                    "battles_with_a_cut": round(a["battle_cut"] / a["games"], 4)}
                for P, a in sorted(acc.items())}
    out["timer"] = timer
    first = collections.Counter()
    recovered = collections.Counter()
    per_battle = collections.defaultdict(list)
    xclass = collections.Counter()
    for g in games:
        xs = [i for i, r in enumerate(g["rows"]) if r["excluded"]]
        for i in xs:
            xclass[(g["panel"], g["rows"][i]["xclass"])] += 1
        if not xs:
            continue
        first[(g["panel"], g["rows"][xs[0]]["xclass"])] += 1
        recovered[(g["panel"], any(r["searched"] for r in g["rows"][xs[0]:]))] += 1
        per_battle[g["panel"]].append(len(xs))
    out["exclusions"] = {
        "by_class": {f"{P}/{c}": n for (P, c), n in sorted(xclass.items())},
        "first_of_battle": {f"{P}/{c}": n for (P, c), n in sorted(first.items())},
        "battle_searched_again": {f"{P}/{v}": n for (P, v), n in sorted(recovered.items())},
        "per_battle": {P: {"battles": len(v), "median": quant(v, .5), "p90": round(quant(v, .9), 1), "sum": sum(v)}
                       for P, v in sorted(per_battle.items())}}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
