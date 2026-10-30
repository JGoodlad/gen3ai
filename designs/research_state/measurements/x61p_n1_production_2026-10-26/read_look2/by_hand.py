"""FROZEN record (X61P look 2, PREREG §7's by-hand lines): (1) C0 s3504/s3505's forum72 units 0-6 played here against
D1's banked C0 units of the same seeds, game for game (half, index, our_team, result, turns); (2) U without H (F + P72,
the panel-validity ruling 4f555e57) with AM-5's widening; (3) N1-P - D1 on F (descriptive, two corpora). Reads the
look-2 state, its inputs.json and D1's archive and record read-only; prints JSON. Not a tool: never re-run as one."""
import json, os, sys

from main.x35 import d1 as D

S = os.path.expanduser("~/gen3ai_archive/x61p_n1p_read/look2/anchor")
D1 = os.path.expanduser("~/gen3ai_archive/x35_d1_read")
HERE = os.path.dirname(os.path.abspath(__file__))


def games(d):
    rows = [json.loads(line) for line in open(os.path.join(d, "games.jsonl"))]
    return sorted(((g.get("half"), g.get("index"), g.get("our_team"), g.get("result"), g.get("turns")) for g in rows),
                  key=lambda x: (str(x[0]), x[1]))


out = {"drift_units": []}
same = tot = wm = wb = 0
for s, bank in ((3504, "h_d"), (3505, "h_e")):
    for u in range(7):
        name = f"c0_s{s}__forum72__{u:02d}"
        mine, banked = games(f"{S}/primary/c0_s{s}/cells/{name}"), games(f"{D1}/{bank}/cells/{name}")
        eq = sum(a == b for a, b in zip(mine, banked))
        w1, w2 = sum(g[3] == "win" for g in mine), sum(g[3] == "win" for g in banked)
        out["drift_units"].append({"unit": name, "identical": eq, "games": len(mine), "wins_now": w1, "wins_banked": w2})
        same += eq; tot += len(mine); wm += w1; wb += w2
out["drift"] = {"identical": same, "games": tot, "wins_now": wm, "wins_banked": wb, "banked_source": "x35_d1_read/h_d, h_e"}

inp = json.load(open(os.path.join(HERE, "inputs.json")))
rep = json.load(open(os.path.join(HERE, "decision.json")))["reported"]["reported_panels"]["per_panel"]
P = ("forum72", "p72")


def arm(side):
    per_seed = {}
    for seed, panels in inp["h"][side].items():
        per_seed[seed] = {"forum72": tuple(panels["forum72"])}
    for seed in per_seed:
        per_seed[seed]["p72"] = tuple(inp["reported_panels"][side][seed]["p72"])
    pooled = {k: (sum(v[k][0] for v in per_seed.values()), sum(v[k][1] for v in per_seed.values())) for k in P}
    return per_seed, pooled


on_s, on_p = arm("on"); off_s, off_p = arm("off")
a, b = D.strat_score(on_p, P), D.strat_score(off_p, P)
d, lo, hi = D.newcombe_strat(a, b)
od = D.overdispersion(on_s, off_s, P)
wlo, whi = D.widen(d, lo, hi, od.factor)
out["u_without_h"] = {"on": a.mean, "off": b.mean, "delta_pp": 100 * d, "ci95_pp": [100 * lo, 100 * hi],
                      "phi_hat": od.phi, "factor": od.factor, "ci95_widened_pp": [100 * wlo, 100 * whi]}

d1 = json.load(open(os.path.expanduser(
    "~/dev/gen3ai/designs/research_state/measurements/x35_breakers_stage_2026-10-17/reads/d1_look2.json")))
d1f = d1["h"]["on"]["per_panel"]["forum72"]
n1f = (sum(v["forum72"][0] for v in on_s.values()), sum(v["forum72"][1] for v in on_s.values()))
a, b = D.strat_score({"forum72": n1f}, ("forum72",)), D.strat_score({"forum72": tuple(d1f)}, ("forum72",))
d, lo, hi = D.newcombe_strat(a, b)
out["n1p_minus_d1_on_f"] = {"n1p": list(n1f), "d1": d1f, "delta_pp": 100 * d, "ci95_pp": [100 * lo, 100 * hi],
                           "note": "N1-P look 2's 5 seeds x 60 units vs D1 look 2's 5 seeds x 7 units; two corpora, two engines (E3 vs E1)"}
json.dump(out, sys.stdout, indent=1)
