import json,glob,math,collections,numpy as np
zs=collections.defaultdict(list); zsame=[]
for p in glob.glob('/home/goodlad/dev/gen3ai/models/*/snapshot_ladder/games.jsonl'):
    run=p.split('/')[-3]
    by=collections.defaultdict(list)
    for l in open(p):
        r=json.loads(l); a,b=r['a'],r['b']; w=r['wins_a']
        if a>b: a,b=b,a; w=r['games']-w
        by[(a,b)].append((r.get('source','ladder'),w,r['games']))
    for k,v in by.items():
        if len(v)<2: continue
        (s1,w1,g1),(s2,w2,g2)=v[0],v[1]
        pbar=(w1+w2)/(g1+g2)
        if pbar in (0,1): continue
        z=(w1/g1-w2/g2)/math.sqrt(pbar*(1-pbar)*(1/g1+1/g2))
        zs[(s1,s2)].append(z)
for k,v in zs.items():
    v=np.array(v); print(k, "n",len(v),"var(z)",round(v.var(),3), "mean z", round(v.mean(),3))
