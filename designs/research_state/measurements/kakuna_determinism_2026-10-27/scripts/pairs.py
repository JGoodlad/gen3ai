import json, sys, itertools, collections
sys.path.insert(0, '/home/goodlad/dev/gen3ai/src')
from main.reads import engine as EN
NAMES = {EN.E0: 'E0', EN.E0B: 'E0B', EN.E1: 'E1', EN.E2: 'E2', EN.E3: 'E3',
         'port=67a272e7796bd85e/93;env=d79c3a74b241a75b/128': 'E-main1026', 'port=bdc5d2be3af5e1fb/94;env=95b45f104119bdb2/129': 'E-su'}
_cache = {}
def eng(r):
    e = r['engine'] or EN.legacy_engine(r['commit'])
    if not e:
        if r['commit'] not in _cache:
            try: _cache[r['commit']] = EN.engine_at(r['commit'])
            except Exception as x: _cache[r['commit']] = 'unknown:' + r['commit']
        e = _cache[r['commit']]
    return NAMES.get(e, e)
groups = json.load(open(sys.argv[1]))
agg = collections.defaultdict(lambda: [0, 0, 0, []])   # (engA, engB) -> units, games, games differing, details
for g in groups:
    for a, b in itertools.combinations(g, 2):
        if a['n'] != b['n'] or a['n'] < 20: continue      # whole 20-game units only (no per-game / searched smoke cells)
        if any(x in c for c in (a['cell'], b['cell']) for x in ('su_build_smoke', 'su_headroom')): continue  # S-U: searched players, live read
        ea, eb = sorted([eng(a), eng(b)])
        k = (ea, eb) if ea != eb else (ea, 'same')
        diff = [i for i, (x, y) in enumerate(zip(a['games'], b['games'])) if x != y]
        panel = next((p for p in ('forum72','heldout72_hpfix','p72','home72','rc72b','rc72','away') if p in a['cell']), '?')
        k = k + (panel,)
        A = agg[k]; A[0] += 1; A[1] += a['n']; A[2] += len(diff)
        if diff: A[3].append((a['cell'].replace('/home/goodlad/gen3ai_archive/', ''), b['cell'].replace('/home/goodlad/gen3ai_archive/', ''),
                              [a['games'][i][1] for i in diff], a['peer_clean'], b['peer_clean'], a['capture'] and b['capture']))
for k, (u, n, d, det) in sorted(agg.items()):
    print(f'{k}: {u} unit pairs, {n} games, {d} differ')
    for x in det: print('     ', x)
