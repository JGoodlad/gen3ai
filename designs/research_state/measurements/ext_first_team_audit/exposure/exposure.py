"""Re-derive F-LH-13 exposure: every run whose stable/exploiter target recorded >1 trainee team."""
import json, os, sys
sys.path.insert(0, '/home/goodlad/dev/gen3ai-wt/ext-audit/src')
from agents.training.matchup_spec import read_recorded_trainee_teams
M = '/home/goodlad/dev/gen3ai/models'
rows = []
def resolve(ref):
    ref = ref.strip()
    if not ref: return None
    p = ref.split('@')[0]
    if not os.path.isabs(p): p = os.path.join('/home/goodlad/dev/gen3ai', p)
    return p
for d in sorted(os.listdir(M)):
    md = os.path.join(M, d, 'metadata.json')
    if not os.path.isfile(md): continue
    try: m = json.load(open(md))
    except Exception as e: print('BADMETA', d, e); continue
    ca = m.get('cli_args', {}) or {}
    tg = []
    so = ca.get('stable_opponents')
    if so:
        tg += [('stable', r) for r in (so if isinstance(so, list) else str(so).split(','))]
    if ca.get('exploiter'): tg.append(('exploiter', ca['exploiter']))
    for kind, r in tg:
        p = resolve(r)
        try:
            teams = read_recorded_trainee_teams(p)
            n = len(teams)
        except Exception as e:
            n = f'ERR {type(e).__name__}: {str(e)[:80]}'
        pins = [ph.get('git_hash','')[:8] for ph in (m.get('pin_history') or [])]
        rows.append(dict(run=d, kind=kind, target=r, n_teams=n, pins=pins))
json.dump(rows, open(os.path.expanduser('~/gen3ai_archive/ext_audit/exposure_rows.json'), 'w'), indent=1)
multi = [r for r in rows if isinstance(r['n_teams'], int) and r['n_teams'] > 1]
errs = [r for r in rows if not isinstance(r['n_teams'], int)]
print('runs with any target:', len({r['run'] for r in rows}), ' multi-team-target runs:', len({r['run'] for r in multi}))
for r in multi: print('MULTI', r['run'], r['kind'], r['target'], r['n_teams'], r['pins'])
for r in errs: print('ERR', r['run'], r['target'], r['n_teams'])
