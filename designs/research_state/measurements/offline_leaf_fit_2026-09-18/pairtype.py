"""POST-HOC: the contested set's pairwise accuracy DECOMPOSED BY PAIR TYPE.

The published 0.568-0.578 level is measured over all three pairs of a three-branch fork —
(top1,top2), (top1,rand), (top2,rand) — pooled. The branched set has only (top1,rand). If the
two populations differ mainly through the (top1,top2) pair, then "the head cannot rank siblings"
is the wrong statement of the ceiling and the right one is narrower.
"""
import sys, json, numpy as np
B='/home/goodlad/dev/gen3ai/designs/research_state/measurements'
sys.path.insert(0, B+'/paired_refit_discrimination_2026-09-14')
sys.path.insert(0, B+'/fork_arm_read_2026-09-16')
sys.path.insert(0, B+'/offline_leaf_fit_2026-09-18')
import torch; torch.set_num_threads(4)
import refit
from refit import build_table, read_head
from score_forks import branch_index
from forks import load_model
from fit_heads import forward_feats, head_V, wide_head
from agents.model.aux_value_heads import WinProbHead
L='/home/goodlad/.claude/jobs/9ab51de6/tmp/leaf_fit'
model = load_model(L+'/src_run/ai_v13_02_flywheel_winprob/eval_traces/step_75005952/snapshot.zip')
rows, succ, sm = refit.load_forks(L+'/forks_eval')
pooled, prepool, V0 = forward_feats(model, succ, sm)
tab, pairs = build_table(rows, pooled, V0)
b = branch_index(rows); assert np.allclose(tab['V0'], V0[b])
Xp = pooled[b]; Xpre = np.concatenate([prepool[b].reshape(len(b),-1), Xp],1).astype(np.float32)
name_of = tab['name']
ptype = np.array(["|".join(sorted((name_of[a], name_of[bb]))) for a, bb in zip(pairs['a'], pairs['b'])])
specs = {'a_winprob_rand_warm':(WinProbHead,False),'b_winprob_paired_warm':(WinProbHead,False),
         'c_wide_pooled':(lambda: wide_head(128),False),
         'd_wide_prepool':(lambda: wide_head(Xpre.shape[1]),True),
         'e_wide_pooled_rank':(lambda: wide_head(128),False)}
V = {'original': V0[b]}
for n,(mk,pre) in specs.items():
    h = mk(); h.load_state_dict(torch.load(f"{L}/fit/head_{n}.pt", weights_only=False)['win_head']); h.eval()
    V[n] = head_V(h, Xpre if pre else Xp)
out = {}
for t in sorted(set(ptype)):
    sub = np.flatnonzero(ptype == t)
    nt = int((pairs['y'][sub] != 0.5).sum())
    out[t] = {'n_pairs': int(len(sub)), 'n_nontied': nt,
              'nontied_rate': float(nt/len(sub)), 'heads': {}}
    for n in ['original'] + list(specs):
        r = read_head(V[n], tab, pairs, sub, n)
        out[t]['heads'][n] = {'pairwise_acc': r['pairwise_acc'], 'ci': r['pairwise_acc_ci']}
    print(f"{t:12s} pairs {len(sub):6d} non-tied {nt:5d} ({nt/len(sub):.3f})  " +
          "  ".join(f"{n.split('_')[0]}={out[t]['heads'][n]['pairwise_acc']:.4f}"
                    f"[{out[t]['heads'][n]['ci'][0]:.4f},{out[t]['heads'][n]['ci'][1]:.4f}]"
                    for n in ['original','a_winprob_rand_warm','d_wide_prepool']))
json.dump(out, open(L+'/pairtype.json','w'), indent=1)
print('-> pairtype.json')
