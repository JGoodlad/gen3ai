"""Every completed Kakuna unit in the archive keyed by what fixes its play: model bytes, opponent, regimes, both team
digests, team seed, seed base, games, our load. Pairs that share a key and were played twice are replays."""
import json, os, glob, hashlib, collections, re, sys
ROOT = '/home/goodlad/gen3ai_archive'
_sha = {}
def sha(p):
    if p not in _sha:
        h = hashlib.sha256()
        try:
            with open(p, 'rb') as f:
                for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
            _sha[p] = h.hexdigest()
        except OSError: _sha[p] = None
    return _sha[p]
argvs = {}
for uj in glob.glob(ROOT + '/**/units.jsonl', recursive=True):
    for line in open(uj, errors='replace'):
        try: d = json.loads(line)
        except Exception: continue
        a = d.get('argv')
        if d.get('event') == 'start' and a and '--out' in a:
            argvs[os.path.realpath(a[a.index('--out') + 1])] = a
def opt(a, k):
    return a[a.index(k) + 1] if a and k in a else None
rows = []
for sj in glob.glob(ROOT + '/**/summary.json', recursive=True):
    cell = os.path.dirname(sj)
    if re.search(r'\.partial|\.void|void_', cell): continue
    try: s = json.load(open(sj))
    except Exception: continue
    if not isinstance(s, dict): continue
    c = s.get("cell") or {}
    if 'Kakuna' not in str(c.get('opponent', '')): continue
    if not os.path.exists(cell + '/games.jsonl'): continue
    a = argvs.get(os.path.realpath(cell))
    g = [json.loads(l) for l in open(cell + '/games.jsonl')]
    caps = sorted(glob.glob(cell + '/capture/*.json'))
    seed0 = None
    if a is None and caps: seed0 = json.load(open(caps[0]))['seed']
    key = (sha(c.get('model_zip')), c.get('opponent'), c.get('our_regime'), c.get('their_regime'), c.get('our_team_digest'),
           c.get('their_team_digest'), opt(a, '--team-seed'), opt(a, '--seed-base'), opt(a, '--games') or len(g),
           opt(a, '--our-belief-prior-mix'), opt(a, '--deal-game'), str(seed0) if seed0 else None, c.get('forfeit_turn_limit'))
    if a is None and not seed0: continue
    sv = c.get('server_version', '')
    rows.append({'cell': cell, 'key': key, 'commit': sv.split('@')[1].split('+')[0] if '@' in sv else sv,
                 'engine': c.get('engine_stamp'), 'capture': bool(caps), 'n': len(g),
                 'peer_clean': s.get('peer_clean'), 'status': s.get('status'),
                 'games': [(x['half'], x['index'], x['our_team'], x['result'], x['turns']) for x in g]})
groups = collections.defaultdict(list)
for r in rows: groups[r['key']].append(r)
out = [v for v in groups.values() if len(v) > 1]
json.dump(out, open(sys.argv[1], 'w'))
print(len(rows), 'kakuna cells;', len(out), 'keys played more than once;', sum(len(v) for v in out), 'cells in them')
