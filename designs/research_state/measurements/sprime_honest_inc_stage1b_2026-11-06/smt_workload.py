"""S'-HONEST stage 1b: the incremental builder's steps over one battle's decisions up to a turn (builder only, no network
forward), one JSON line a step with its CPU by phase: the SMT probe's workload and the late-turn profile.

    PYTHONPATH=src python3 smt_workload.py <group prefix> <stop turn>
"""
import json, sys
sys.path.insert(0, "designs/research_state/measurements/sprime_honest_inc_stage1b_2026-11-06")
import seq_probe as SQ
from main.value_search import honest_inv as INV
from main.value_search import info_set as IS
from main.value_search.search import cpu_all
from main.value_search.world_draws import WorldError
gpre, stop_turn = sys.argv[1], int(sys.argv[2])
g = next(v for k, v in SQ.groups(SQ.targets(["late", "stage1"])) if k[1].startswith(gpre))
last = g[-1][1]
print("group", gpre, [int(r["turn"]) for _p, r in g], flush=True)
s = INV.searcher("r0s4_honest_inc@10")
inner = s.sampler.inner
for x in INV.sequence(INV.decision_of(last), 10, s.engine):
    if x.turn > stop_turn:
        break
    c0 = cpu_all()
    info, frame, pts = IS.extract_points(x.log, x.at, x.side)
    c1 = cpu_all()
    err, fails = None, None
    try:
        w = inner.sample_info(info, 8, frame, points=pts); inc = w.inc; ok = True; fails = w.fails
    except WorldError as e:
        inc = getattr(e, "inc", None); ok = False; err = str(e)[-300:]
    c = cpu_all() - c0
    print(json.dumps({"t": x.turn, "ok": ok, "cpu": round(c, 2), "pts": round(c1 - c0, 2), "ph": inner.last_phases,
                      "ext": (inc or {}).get("extended"), "ch": (inc or {}).get("chain"), "seg": len(info.segments()),
                      "mode": (inc or {}).get("mode"), "fails": fails, "err": err}), flush=True)
