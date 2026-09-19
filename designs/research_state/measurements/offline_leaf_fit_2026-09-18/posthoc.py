"""POST-HOC (declared as post-hoc, registered nowhere): where does the bound live?

Two questions the registered bars cannot answer:

 1. Do the fitted heads rank better on the population they were TRAINED on (branched, uniform
    depth, held-out BATTLES) than on the contested read set? A large gap = a TRANSFER failure; no
    gap = the bound is in the label, not in the populations coming apart.
 2. Is the contested set's decidability really lower than the branched set's? The two are not
    comparable as printed — the contested set has three branches and therefore three pairs per
    fork, of which `top1` vs `top2` is almost always tied. Restricted to the SAME (top1, rand)
    pair, what are the two non-tied rates?
"""
import sys, json, numpy as np
B='/home/goodlad/dev/gen3ai/designs/research_state/measurements'
sys.path.insert(0, B+'/paired_refit_discrimination_2026-09-14')
sys.path.insert(0, B+'/fork_arm_read_2026-09-16')
sys.path.insert(0, B+'/offline_leaf_fit_2026-09-18')
import torch; torch.set_num_threads(4)
import refit
from refit import read_head
from forks import load_model
from fit_heads import load_rows, train_table, forward_feats, head_V, wide_head, TRAIN_BRANCHES
from agents.model.aux_value_heads import WinProbHead

L='/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_fit'
model = load_model(L+'/src_run/ai_v13_02_flywheel_winprob/eval_traces/step_75005952/snapshot.zip')
out = {}

# ---- 1. the fits on their OWN held-out branched battles -------------------------------
rows, succ, smask = load_rows(L+'/forks_branch', TRAIN_BRANCHES)
tab, pairs = train_table(rows, TRAIN_BRANCHES)
pooled, prepool, V0 = forward_feats(model, succ, smask)
X_pool = pooled[tab['idx']]
X_pre  = np.concatenate([prepool[tab['idx']].reshape(len(tab['idx']),-1), X_pool],1).astype(np.float32)
rng = np.random.default_rng(20260918)
battles = np.unique(tab['battle']); rng.shuffle(battles)
n_val = max(1, int(round(0.20*len(battles))))
val_b = set(battles[:n_val].tolist())
is_val = np.array([b in val_b for b in tab['battle']])
va_set = set(np.flatnonzero(is_val).tolist())
p_va = np.flatnonzero([a in va_set and b in va_set for a,b in zip(pairs['a'],pairs['b'])])
print(f"held-out branched: {len(p_va)} pairs, {int((pairs['y'][p_va]!=0.5).sum())} non-tied")

specs = {'a_winprob_rand_warm':(WinProbHead,False),'b_winprob_paired_warm':(WinProbHead,False),
         'c_wide_pooled':(lambda: wide_head(128),False),
         'd_wide_prepool':(lambda: wide_head(X_pre.shape[1]),True),
         'e_wide_pooled_rank':(lambda: wide_head(128),False),
         'a0_winprob_rand_fresh':(WinProbHead,False),'c0_winprob_paired_fresh':(WinProbHead,False)}
V = {'original': V0[tab['idx']]}
for name,(mk,pre) in specs.items():
    h = mk(); h.load_state_dict(torch.load(f"{L}/fit/head_{name}.pt")['win_head']); h.eval()
    V[name] = head_V(h, X_pre if pre else X_pool)
out['branched_heldout'] = {}
for name in ['original']+list(specs):
    r = read_head(V[name], tab, pairs, p_va, name)
    out['branched_heldout'][name] = {k:r[k] for k in
        ('pairwise_acc','pairwise_acc_ci','n_nontied','sep_ratio','ece','brier','mean_V','base_rate')}
    print(f"  BRANCHED-HELDOUT {name:24s} {r['pairwise_acc']:.4f} "
          f"[{r['pairwise_acc_ci'][0]:.4f},{r['pairwise_acc_ci'][1]:.4f}] n={r['n_nontied']} "
          f"ECE {r['ece']:.4f} Brier {r['brier']:.4f}")

# ---- 2. the matched (top1, rand) decidability ------------------------------------------
def t1_rand_rate(rows, names):
    n = k = 0
    for r in rows:
        a, b = r['branches']['top1'], r['branches']['rand']
        if any(x['succ'] is None or x['capped'] or x['outcome'] not in ('win','loss') for x in (a,b)):
            continue
        n += 1
        k += int(a['outcome'] != b['outcome'])
    return n, k, k/max(1,n)
e_rows, e_succ, e_sm = refit.load_forks(L+'/forks_eval')
out['matched_top1_rand_decidability'] = {
    'branched': dict(zip(('n','nontied','rate'), t1_rand_rate(rows, TRAIN_BRANCHES))),
    'contested': dict(zip(('n','nontied','rate'), t1_rand_rate(e_rows, ('top1','top2','rand')))),
}
print('matched (top1,rand) decidability:', json.dumps(out['matched_top1_rand_decidability']))

# ---- 3. the policy descriptors of the branched set --------------------------------------
out['branched_gap_percentiles'] = {str(q): float(np.percentile([r['gap'] for r in rows], q))
                                   for q in (10,25,40,50,75,90)}
out['branched_depth_frac_percentiles'] = {str(q): float(np.percentile(
    [r.get('depth_frac',0.0) for r in rows], q)) for q in (10,25,50,75,90)}
json.dump(out, open(L+'/posthoc.json','w'), indent=1)
print('-> posthoc.json')
