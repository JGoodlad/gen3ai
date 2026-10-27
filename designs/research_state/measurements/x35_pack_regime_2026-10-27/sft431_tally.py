import json, sys, os, collections
roots = sys.argv[1:]
agg = collections.defaultdict(lambda: [0, 0, 0, 0])  # f72 games, sft431 ours games, sft431 wins, f72 wins
for root in roots:
    for dp, dn, fn in os.walk(root):
        if "games.jsonl" not in fn: continue
        for ln in open(os.path.join(dp, "games.jsonl")):
            try: g = json.loads(ln)
            except Exception: continue
            if "forum72" not in str(g.get("our_team_source", "")): continue
            parts = os.path.relpath(dp, root).split(os.sep)
            key = os.path.basename(root) + "/" + "/".join(parts[:-2] if len(parts) > 2 else parts[:1])
            a = agg[key]; a[0] += 1; a[3] += bool(g.get("won"))
            if g.get("our_team") == "sft431.txt":
                a[1] += 1; a[2] += bool(g.get("won"))
tot = [0, 0, 0, 0]
for k, a in sorted(agg.items()):
    print(f"{k:70s} f72={a[0]:6d} ours_sft431={a[1]:4d} ({a[1]/max(a[0],1):.4f}) won={a[2]}  f72_wr={a[3]/max(a[0],1):.3f}")
    for i in range(4): tot[i] += a[i]
print("TOTAL", tot, tot[1]/max(tot[0],1), tot[2]/max(tot[1],1))
