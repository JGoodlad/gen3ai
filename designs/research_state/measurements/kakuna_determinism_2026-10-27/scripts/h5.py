"""H5 (turn-250 forfeit -> Metamon's long-tail force-reset): how often a forfeit faults the peer, and what the next game looks like."""
import json, glob, os, re, collections
ROOT = '/home/goodlad/gen3ai_archive'
st = collections.Counter(); ex = []
for sj in glob.glob(ROOT + '/**/summary.json', recursive=True):
    cell = os.path.dirname(sj)
    try: s = json.load(open(sj))
    except Exception: continue
    if not isinstance(s, dict) or 'Kakuna' not in str((s.get('cell') or {}).get('opponent', '')): continue
    if not os.path.exists(cell + '/games.jsonl'): continue
    g = [json.loads(l) for l in open(cell + '/games.jsonl')]
    st['units'] += 1; st['games'] += len(g)
    ff = [x for x in g if x.get('hit_forfeit_limit')]
    if not ff: continue
    st['units_with_forfeit'] += 1; st['forfeits'] += len(ff)
    logs = glob.glob(cell + '/*/peer_*.log')
    lt = {os.path.basename(os.path.dirname(l)): sum('long-tail' in line for line in open(l, errors='replace')) for l in logs}
    for x in ff:
        half = x['half']; faulted = lt.get(half, 0) > 0
        st['forfeit_faulted' if faulted else 'forfeit_clean'] += 1
        nxt = [y for y in g if y['half'] == half and y['index'] == x['index'] + 1]
        if faulted:
            st['faulted_last_in_half' if not nxt else 'faulted_mid_half'] += 1
            if nxt: ex.append((cell.replace(ROOT + '/', ''), x['index'], nxt[0]['result'], nxt[0]['turns']))
print(dict(st))
for e in ex: print('   ', e)
