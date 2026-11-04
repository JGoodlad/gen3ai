import json, math, glob, sys
from scipy import stats
OPS='/home/goodlad/gen3ai_archive/n4_capacity_look1_2026-10-31_ops'
WT='/home/goodlad/dev/gen3ai-wt/n4-record'
out={}
# pin drift (Welch 90 %, descriptive, cross-engine)
m1,v1,m2,v2=51.479166666666664,0.03543402777777701,52.427777777777784,0.5378703703703673
se=math.sqrt(v1/3+v2/3); df=se**4/((v1/3)**2/2+(v2/3)**2/2); t=stats.t.ppf(0.95,df)
d=m1-m2; out['pin_drift']={'c0p_mean':m1,'banked_c0_mean':m2,'delta':d,'ci90':[d-t*se,d+t*se],'df':df}
# G-F
rows=json.load(open(f'{OPS}/gf/read.json'))
gf={}; tot=[]
for r in rows:
    p=r['player'].split('@')[0]; o=r['opponent'].split('@')[0]
    gf.setdefault(p,{})[o]={'score_pp':100*r['score'],'se_pp':r['se_pp'],'pairs':r['pairs'],'regime':r['regime_id']}
    tot.append(100*r['score'])
out['gf']=gf; out['gf_mean_pp']=sum(tot)/len(tot); out['gf_n_cells']=len(tot)
out['gf_regimes']=sorted({r['regime_id'] for r in rows})
# rho from dose records
dose={}
for f in sorted(glob.glob(f'{WT}/designs/research_state/measurements/x35_breakers_stage_2026-10-17/dose/rb_x35t_*n4*.json')):
    j=json.load(open(f)); dose[f.split('/')[-1][:-5]]=j
out['dose_keys']={k:list(v.keys())[:30] for k,v in list(dose.items())[:1]}
out['dose']={k:{kk:v.get(kk) for kk in ('wall_min','steps_per_s','wall_s','minutes','wall_minutes') if kk in v} for k,v in dose.items()}
# mech
M=f'{OPS}/mech'
e1={};e2={};nul={};bat={}
for s in ('3641','3642','3643'):
    j=json.load(open(f'{M}/e1_n4ffn_s{s}.json')); fin=j['series'][-1]
    e1[s]={'step':fin['step'],'ratios':{k:v['ratio'] for k,v in fin['norms']['features_extractor.team_transformer.']['rounds'].items()}, 'engaged':fin.get('engaged', None)}
    j=json.load(open(f'{M}/e2_n4ffn_s{s}.json')); fin=j['series'][-1]
    e2[s]={'step':fin['step'],'kl_mean':fin['kl_mean'],'argmax':fin['argmax_moved_share'],'series_kl':[round(x['kl_mean'],4) for x in j['series']]}
    j=json.load(open(f'{M}/null_c0n4_s{s}.json'))
    nul[s]={'e1_null':{k:v['e1_null'] for k,v in j['e1_null']['rounds']['features_extractor.team_transformer.'].items()},'e2_null':{k:v for k,v in j['e2_null'].items() if not isinstance(v,(dict,list))}}
for lab in ['tstar']+[f'c0n4_s{s}' for s in ('3641','3642','3643')]+[f'n4ffn_s{s}' for s in ('3641','3642','3643')]:
    j=json.load(open(f'{M}/battery_{lab}.json'))
    rk=j['rank']['trained']; tr=j['trainability']['taps']; ts=j['tanh_saturation']['layers']
    bat[lab]={'tt_pr':rk['team_tokens']['pr'],'tt_srank99':rk['team_tokens']['srank99'],'pi_pr':rk['pi_features']['pr'],'pi_srank99':rk['pi_features']['srank99'],
              'train_ratio_tt':tr['team_tokens']['capacity_ratio'],
              'l1_sat_acts':ts['mlp_extractor.policy_net.1']['mean_sat_frac'],'l1_sat_units':ts['mlp_extractor.policy_net.1']['saturated_units'],
              'l2_sat_acts':ts['mlp_extractor.policy_net.3']['mean_sat_frac'],'l2_sat_units':ts['mlp_extractor.policy_net.3']['saturated_units'],
              'meta_states':j['meta'].get('n_states', j['meta'].get('states'))}
out.update(e1=e1,e2=e2,nulls=nul,battery=bat)
json.dump(out,open(f'{OPS}/by_hand_lines.json','w'),indent=1)
print(json.dumps(out,indent=1))
