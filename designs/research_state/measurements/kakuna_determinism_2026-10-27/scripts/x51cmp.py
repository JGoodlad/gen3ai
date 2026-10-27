import json, glob, os, re, collections, sys
X51 = '/home/goodlad/dev/gen3ai/designs/research_state/measurements/x51_plateau_vs_kakuna_2026-10-20/rows.jsonl'
bank = collections.defaultdict(list)
for l in open(X51):
    r = json.loads(l)
    if not r['model_zip'].endswith('checkpoint_280000170_steps.zip'): continue
    ts = re.search(r'teams_(\w+)$', r['our_team_source']).group(1)
    bank[(ts, int(r['unit_seed']))].append((r['half'], r['index'], r['our_team'], r['result'], r['turns'], r['server_version'].split('@')[1].split('+')[0]))
ROOT = '/home/goodlad/gen3ai_archive'
reads = {'x26c (E1)': ROOT + '/x26c_read/kakuna_pstar/cells', 'hctx am1 (E-main1026)': ROOT + '/hctx_rc72_am1_read/anchor/*/pstar/cells',
         'hctx am2 (E-su)': ROOT + '/hctx_rc72_am2_read/anchor/*/pstar/cells', 'belief mix guard (E1)': ROOT + '/belief_prior_mix_2026-10-26/*/anchor/guard/pstar/cells',
         'hctx guard': ROOT + '/hctx_rc72_read/anchor/guard/pstar/cells', 'x59s identity': ROOT + '/x59s/read/identity*/*/cells'}
for name, pat in reads.items():
    tot = collections.Counter(); diffs = []
    for cell in sorted(glob.glob(pat + '/pstar__*')):
        m = re.search(r'pstar__(\w+?)__(\d+)$', cell)
        if not m or not os.path.exists(cell + '/games.jsonl'): continue
        ts, u = m.group(1), int(m.group(2))
        b = sorted(bank.get((ts, u), []), key=lambda x: x[1])
        if not b: continue
        g = sorted(((x['half'], x['index'], x['our_team'], x['result'], x['turns']) for x in map(json.loads, open(cell + '/games.jsonl'))), key=lambda x: x[1])
        if len(g) != len(b): tot['n_mismatch'] += 1; continue
        tot['units'] += 1; tot['games'] += len(g); tot[ts] += len(g)
        for x, y in zip(g, b):
            if x != y[:5]: diffs.append((ts, u, x, y))
        tot['x51_commit:' + b[0][5]] += 1
    print(name, dict(tot), 'differ:', len(diffs))
    for d in diffs: print('    ', d)
