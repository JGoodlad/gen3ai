"""Game-for-game survey of two plays of the same Kakuna units: per game, is it identical, and if not, who chose first
differently (ours / kakuna) from an identical protocol, and was a silently ending volatile (Destiny Bond / Endure) live."""
import json, os, re, sys
SILENT = re.compile(r'\|-singlemove\|[^|]*\|Destiny Bond|\|-singleturn\|[^|]*\|move: Endure|\|-singlemove\|[^|]*\|(Grudge|Rage)|\|-singleturn\|[^|]*\|(Snatch|move: Snatch|Helping Hand)')
def games(cell):
    return {g['battle_tag']: g for g in map(json.loads, open(os.path.join(cell, 'games.jsonl')))}
def key(g): return (g['half'], g['index'], g['our_team'], g['result'], g['turns'])
def per_player(cmds):
    d = {'p1': [], 'p2': []}
    for c in cmds:
        p = c.split(' ', 2)
        if p[0] == 'CHOOSE': d[p[1]].append(p[2])
    return d
def ours(cap):
    st = json.loads(cap['commands'][0][6:])
    return 'p1' if st['p1']['name'].startswith('Gen3AI') else 'p2'
def classify(ca, cb):
    if ca['seed'] != cb['seed'] or ca['commands'][0] != cb['commands'][0]: return 'start', None
    ka = [t for w, t in ca['chunks'] if w == 'p1']; kb = [t for w, t in cb['chunks'] if w == 'p1']
    j = next((i for i in range(min(len(ka), len(kb))) if ka[i] != kb[i]), None)
    if j is None and len(ka) != len(kb): j = min(len(ka), len(kb))
    o = ours(ca); pa, pb = per_player(ca['commands']), per_player(cb['commands'])
    if j is None:
        same = pa == pb
        return ('order-only' if same else 'choice-unexecuted'), (None if same else o)
    prev = [t for t in ka[:j] if not t.startswith('|request|')]
    silent = bool(SILENT.search('\n'.join(prev[-6:])))
    def acts(t):
        d = {}
        for m in re.finditer(r'^\|(move|switch|cant)\|(p[12])a: [^|]*\|([^|\n]*)', t, re.M):
            d.setdefault(m.group(2), (m.group(1), m.group(3)))
        return d
    xa, xb = acts(ka[j]), acts(kb[j])
    who = [('ours' if side == o else 'kakuna') for side in ('p1', 'p2') if xa.get(side) != xb.get(side)]
    if '|win|' in ka[j] and 'FORCELOSE' in ' '.join(ca['commands'] + cb['commands']): return 'forcelose', who
    return 'protocol', (who or ['rng/engine']), silent, (xa, xb)
def survey(pairs):
    tot = dict(games=0, equal=0, bytes_identical=0, order_only=0)
    rows = []
    for A, B in pairs:
        ga, gb = games(A), games(B)
        for tag in ga:
            tot['games'] += 1
            eq = key(ga[tag]) == key(gb[tag]); tot['equal'] += eq
            ca = json.load(open(f'{A}/capture/{tag}.json')); cb = json.load(open(f'{B}/capture/{tag}.json'))
            if ca == cb: tot['bytes_identical'] += 1; continue
            c = classify(ca, cb)
            if c[0] == 'order-only': tot['order_only'] += 1; continue
            rows.append((os.path.basename(A), tag, eq, key(ga[tag])[3:], key(gb[tag])[3:], c))
    return tot, rows
def exposure(cells):
    n = e = 0
    for A in cells:
        for f in os.listdir(f'{A}/capture'):
            cap = json.load(open(f'{A}/capture/{f}')); n += 1
            e += bool(SILENT.search('\n'.join(t for w, t in cap['chunks'] if w == 'p1')))
    return n, e
if __name__ == '__main__':
    spec = json.load(open(sys.argv[1]))
    for name, pairs in spec.items():
        tot, rows = survey(pairs)
        n, e = exposure([a for a, _ in pairs])
        print(f'### {name}: {tot}; games with a Destiny Bond / Endure / Grudge / Rage / Snatch / Helping Hand: {e}/{n}')
        for r in rows: print('   ', r)
