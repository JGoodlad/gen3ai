import os
os.chdir('/home/goodlad/dev/gen3ai')
from agents.training.matchup_spec import read_recorded_trainee_teams
from agents.training.team_archetypes import team_sha, lookup_team
opps = ['ai_v8_09_pool10_exploiter_0723','ai_v8_06_semistall_3team_exploiter_0722','ai_v8_13_defensive10_exploiter_0725',
        'ai_v9_31_tock1_k4_0824','ai_v9_32_tock1b_rain_0824','ai_v13_13_exploit5_offense','ai_v13_18_teach5_offense_hidose','ai_v13_24_popr1_read_loop']
for o in opps:
    try:
        teams = read_recorded_trainee_teams(f'models/{o}')
    except Exception as e:
        print(o, 'ERR', e); continue
    print('==', o, len(teams))
    for i,t in enumerate(teams):
        p = t if os.path.isabs(t) else t
        txt = None
        for cand in [p, os.path.join('/home/goodlad/dev/gen3ai', p)]:
            if os.path.exists(cand): txt = open(cand).read(); break
        if txt is None: print('  ', i, t, 'MISSING'); continue
        rec = lookup_team(txt) or {}
        mons = [l.split('@')[0].strip() for l in txt.strip().split('\n\n') for l in [l.strip().split('\n')[0]]]
        print('  ', i, team_sha(txt), os.path.basename(t), rec.get('archetype'), '|', ', '.join(mons)[:120])
