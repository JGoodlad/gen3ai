import json, statistics as st, datetime as dt
d = json.load(open("/tmp/x47rev/rb_x35t_c0_s3601.json"))
c = d["rollout/collect_ms"]; u = d["lifecycle/update_wall_s"]
ev = d["eval/wall_sec"]; ce = d.get("rust_eval/cycle_wall_s", [])
print("eval rows (step, wall_sec, walltime)", [(s, round(v,1), dt.datetime.utcfromtimestamp(w).strftime("%H:%M")) for s,v,w in ev[:4]])
print("rust_eval cycle rows", [(s, round(v,1)) for s,v,w in ce[:4]])
# collect around eval walltimes
cw = [(s, v/1000, w) for s, v, w in c]
for s, v, w in ev[:4]:
    near = [(x[0], round(x[1],2), round(x[2]-w)) for x in cw if abs(x[2]-w) < 60]
    print("eval at", s, "collect rows near:", near)
# hourly binned collect median
bins = {}
for s, v, w in cw:
    h = dt.datetime.utcfromtimestamp(w).strftime("%m-%d %H")
    bins.setdefault(h, []).append(v)
for h in sorted(bins): print(h, "n", len(bins[h]), "collect median %.2f max %.2f" % (st.median(bins[h]), max(bins[h])))
x = json.load(open("/tmp/x47rev/rb_x35t_x47d4_s3601.json"))
print("x47 start", dt.datetime.utcfromtimestamp(x["rollout/collect_ms"][0][2]), "last", dt.datetime.utcfromtimestamp(x["rollout/collect_ms"][-1][2]))
print("c0 start", dt.datetime.utcfromtimestamp(cw[0][2]), "end", dt.datetime.utcfromtimestamp(cw[-1][2]))
# collect decisions per row
cd = d["rollout/collect_decisions"]; print("c0 decisions/row median", st.median(v for s,v,w in cd))
xd = x["rollout/collect_decisions"]; print("x47 decisions/row median", st.median(v for s,v,w in xd))
