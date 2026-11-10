"""Which part of the move LATENT loss pushes the residual larger? (move-leg diagnosis, 2026-11-10; CPU)

``move_belief_latent_loss`` = (1 - cosine) + 1.0 x a VARIANCE HINGE. The cosine part grades
``softmax(move logits) @ latent_table`` against the MEAN latent of the slot's true moveset; the hinge is
``relu(1 - std)`` of that predicted latent over the batch's slots, per latent dimension. Two candidate mechanisms
for a loss whose descent GROWS ``move_head``'s residual:

* the reveal pin: a revealed move's logit is pinned at ``_REVEAL_LOGIT`` (10.0), so on a slot with r revealed
  moves the softmax can put 1/4 of its mass on each true UNSHOWN move only if that logit is near the pin too;
* the hinge: predictions that differ more from slot to slot have a larger batch std, whatever the truth is.

On in-pool rows at P* this prints d(0.05 x part)/d(residual scale) (NEGATIVE = its descent grows the residual)
for the cosine part split by r, with the label BCE's derivative on the same slots beside it, and for the hinge
with the std it reads (the hinge depends on the batch: argv 3 sets the rows a batch). Reads the P* checkpoint and the archive's scoring bank; writes nothing.
"""
import sys
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F

from agents.model.belief_heads import _REVEAL_LOGIT
from agents.model.extra_obs_keys import zero_extra_obs
from main.offpool_belief import finetune as FT
from main.offpool_belief import score as SC

D = Path.home() / "gen3ai_archive/move_leg_diag_2026-11-10"
PANEL = sys.argv[1] if len(sys.argv) > 1 else "home72"
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 8
B = Path.home() / f"gen3ai_archive/offpool_belief_2026-11-09/bank/{PANEL}/rows.npz"
th.manual_seed(0); th.set_num_threads(4)
model, _ck = SC.load_checkpoint(str(np.load(D / "dump" / "pstar__tables.npz")["ckpt_path"]))
fe = model.policy.features_extractor
FT.trained_parameters(model)
head = fe.move_belief.move_head
P = (head.weight, head.bias)
z = np.load(B); rows = {k: z[k] for k in z.files}
rng = np.random.default_rng(1)
N = int(sys.argv[3]) if len(sys.argv) > 3 else 128                  # rows a batch; above 256 the split by r is skipped
hinge, cosall, bceall, stds = [], [], [], []
out = {r: {"slots": 0, "free": 0.0, "cos": 0.0, "d_lat": [], "d_bce": []} for r in range(5)}


def dscale(term):
    g = th.autograd.grad(term, P, retain_graph=True, allow_unused=True)
    return float(sum((x * p).sum() for x, p in zip(g, P) if x is not None))


for rep in range(NB):
    sel = np.sort(rng.choice(len(rows["obs"]), N, replace=False))
    ob = {"observation": th.as_tensor(rows["obs"][sel], dtype=th.float32),
          "action_mask": th.as_tensor(rows["mask"][sel], dtype=th.float32)}
    ob.update(zero_extra_obs(fe, batch=N, device="cpu"))
    model.policy.evaluate_actions(ob, th.zeros(N, dtype=th.long), action_masks=rows["mask"][sel].astype(bool))
    ml = fe.stash.belief_supervision["move_belief_logits"]                     # [B,6,M], the LIVE view
    table = fe.last_move_latent_table
    km = th.as_tensor(rows["known_moves"][sel]).long()
    valid = km >= 0
    has = valid.any(-1)
    pinned = ml.detach() == _REVEAL_LOGIT
    r = pinned.sum(-1)                                                         # [B,6] revealed moves on the slot
    sm = F.softmax(ml, dim=-1)
    pred = sm @ table
    lat = table[km.clamp(min=0)] * valid.unsqueeze(-1).float()
    tgt = (lat.sum(2) / valid.sum(-1, keepdim=True).clamp(min=1).float()).detach()
    cos = F.cosine_similarity(pred, tgt, dim=-1)                               # [B,6]
    target = th.zeros_like(ml).scatter_(-1, km.clamp(min=0), valid.float())
    bce = F.binary_cross_entropy_with_logits(ml, target, reduction="none").mean(-1)   # [B,6]
    n_all = float(has.sum())
    std = th.sqrt(pred[has].var(dim=0, unbiased=False) + 1e-4)                 # [D], the trainer's own expression
    hinge.append(dscale(0.05 * F.relu(1.0 - std).mean()))
    cosall.append(dscale(0.05 * (1.0 - cos[has]).mean()))
    bceall.append(dscale(0.05 * bce[has].mean()))
    stds.append((float(std.mean()), float((std < 1.0).float().mean()), float(F.relu(1.0 - std).mean()),
                 float((1.0 - cos[has]).mean())))
    for k in range(5 if N <= 256 else 0):
        m = has & (r == k)
        if not bool(m.any()):
            out[k]["d_lat"].append(0.0); out[k]["d_bce"].append(0.0)
            continue
        out[k]["slots"] += int(m.sum())
        out[k]["free"] += float((sm.detach() * (~pinned).float()).sum(-1)[m].sum())
        out[k]["cos"] += float(cos.detach()[m].sum())
        out[k]["d_lat"].append(dscale(0.05 * (1.0 - cos)[m].sum() / n_all))
        out[k]["d_bce"].append(dscale(0.05 * bce[m].sum() / n_all))
    print(f"batch {rep}: hinge {hinge[-1]:+.3e} cosine {cosall[-1]:+.3e} sum {hinge[-1] + cosall[-1]:+.3e} | std "
          f"{stds[-1][0]:.3f}, dims below 1: {stds[-1][1]:.2f}, hinge value {stds[-1][2]:.4f}, 1-cos {stds[-1][3]:.4f}",
          flush=True)
    if N <= 256:
        print(f"batch {rep}: " + "; ".join(f"r={k} lat {out[k]['d_lat'][-1]:+.2e} bce {out[k]['d_bce'][-1]:+.2e}" for k in range(5)),
              flush=True)

print(f"\n{PANEL}: {NB} batches x {N} rows; revealed pin {_REVEAL_LOGIT}; each part is its share of the batch mean")
if N <= 256:
    print("| revealed moves on the slot | slots | softmax mass off the pinned moves | cosine | d(0.05 x latent part)/d(scale): mean [min, max] | negative | d(0.05 x label BCE part)/d(scale): mean | negative |")
    print("|---|---|---|---|---|---|---|---|")
    for k in range(5):
        o = out[k]; dl = np.array(o["d_lat"]); db = np.array(o["d_bce"]); n = max(o["slots"], 1)
        print(f"| {k} | {o['slots']} | {o['free'] / n:.4f} | {o['cos'] / n:.3f} | {dl.mean():+.3e} [{dl.min():+.3e}, {dl.max():+.3e}] | "
              f"{int((dl < 0).sum())} / {NB} | {db.mean():+.3e} | {int((db < 0).sum())} / {NB} |")
    tl = sum(np.array(out[k]["d_lat"]) for k in range(5)); tb = sum(np.array(out[k]["d_bce"]) for k in range(5))
    print(f"| all | {sum(out[k]['slots'] for k in range(5))} | | | {tl.mean():+.3e} [{tl.min():+.3e}, {tl.max():+.3e}] | "
          f"{int((tl < 0).sum())} / {NB} | {tb.mean():+.3e} | {int((tb < 0).sum())} / {NB} |")
h = np.array(hinge); c = np.array(cosall); sd = np.array(stds)
print(f"\nthe two parts of 0.05 x the latent loss, d/d(scale): mean [min, max]; batches < 0")
print(f"| cosine part | {c.mean():+.3e} [{c.min():+.3e}, {c.max():+.3e}] | {int((c < 0).sum())} / {NB} |")
print(f"| variance hinge | {h.mean():+.3e} [{h.min():+.3e}, {h.max():+.3e}] | {int((h < 0).sum())} / {NB} |")
print(f"| the loss | {(h + c).mean():+.3e} [{(h + c).min():+.3e}, {(h + c).max():+.3e}] | {int(((h + c) < 0).sum())} / {NB} |")
b = np.array(bceall)
print(f"| label BCE x 0.05 on the same slots (all 400 channels) | {b.mean():+.3e} [{b.min():+.3e}, {b.max():+.3e}] | {int((b < 0).sum())} / {NB} |")
print(f"predicted-latent std over a batch's slots: mean {sd[:, 0].mean():.3f}; share of latent dims below the target 1.0: "
      f"{sd[:, 1].mean():.2f}; loss values: hinge {sd[:, 2].mean():.4f}, 1 - cosine {sd[:, 3].mean():.4f}")
