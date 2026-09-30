import os, hashlib
os.chdir('/home/goodlad/dev/gen3ai')
from agents.training.matchup_spec import read_recorded_trainee_teams
targets = {'c3e2671c0f','4e99dc34b3','015cb84e8e','c5676d264e','564b9be3ae','9278913bce'}
opps = ['ai_v8_09_pool10_exploiter_0723','ai_v8_06_semistall_3team_exploiter_0722','ai_v8_13_defensive10_exploiter_0725',
        'ai_v9_31_tock1_k4_0824','ai_v9_32_tock1b_rain_0824','ai_v13_13_exploit5_offense','ai_v13_18_teach5_offense_hidose','ai_v13_24_popr1_read_loop']
for o in opps:
    teams = read_recorded_trainee_teams(f'models/{o}')
    for i,t in enumerate(teams):
        txt = open(t).read()
        raw = hashlib.sha1(txt.encode()).hexdigest()[:10]
        rs = hashlib.sha1(txt.rstrip().encode()).hexdigest()[:10]
        st = hashlib.sha1(txt.strip().encode()).hexdigest()[:10]
        print(o, i, 'raw', raw, 'strip', st, 'rstrip', rs, 'MATCH' if targets & {raw,rs,st} else '')
