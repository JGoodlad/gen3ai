import json, statistics as st, sys
for name in ["rb_x35t_c0_s3601", "rb_x35t_x47d4_s3601"]:
    d = json.load(open(f"/tmp/x47rev/{name}.json"))
    c = d["rollout/collect_ms"]; u = d["lifecycle/update_wall_s"]; f = d["time/fps"]
    print("==", name, len(c), len(u))
    print(" first collect rows", [(s, round(v/1000,2)) for s, v, w in c[:5]])
    print(" first update rows ", [(s, round(v,2)) for s, v, w in u[:5]])
    print(" collect steps vs update steps first 3", [x[0] for x in c[:3]], [x[0] for x in u[:3]])
    cs = [v/1000 for s, v, w in c]; us = [v for s, v, w in u]
    print(" collect s: median %.2f mean %.2f min %.2f max %.2f" % (st.median(cs), st.mean(cs), min(cs), max(cs)))
    print(" update  s: median %.2f mean %.2f min %.2f max %.2f" % (st.median(us), st.mean(us), min(us), max(us)))
    steps = [s for s, v, w in c]
    dsteps = [b - a for a, b in zip(steps, steps[1:])]
    print(" steps/row median", st.median(dsteps), "min", min(dsteps), "max", max(dsteps))
    walls = [w for s, v, w in c]
    dw = [b - a for a, b in zip(walls, walls[1:])]
    print(" wall/row median %.2f mean %.2f" % (st.median(dw), st.mean(dw)))
    tot_wall = walls[-1] - walls[0]
    print(" total wall first->last collect row %.0f s, sum collect %.0f, sum update %.0f, ratio %.3f" % (tot_wall, sum(cs[1:]), sum(us), (sum(cs[1:]) + sum(us[:-1]))/tot_wall))
    print(" last step", steps[-1], "sigma total %.0f" % (sum(cs)+sum(us)), "per step %.3e" % ((sum(cs)+sum(us))/(max(steps[-1], u[-1][0]) - 290000087)))
    if "eval/wall_sec" in d:
        print(" eval rows", [(s, round(v)) for s, v, w in d["eval/wall_sec"][:3]], "sum", sum(v for s,v,w in d["eval/wall_sec"]))
