"""Which loss holds the move head's over-confidence? (move-leg diagnosis, 2026-11-10; CPU)

The move belief is ``move_head(read) + prior[species]``, so scaling the head's weight and bias by a factor
``a`` scales the learned residual exactly. For each loss term of the production objective, at its production
coefficient, this prints d(term)/d(a) at a = 1 = <grad_W, W> + <grad_b, b> on in-pool rows (the ``home72``
bank): POSITIVE means that term's descent SHRINKS the residual, NEGATIVE that it grows it. At a stationary
point of training the terms sum to zero, so the terms on the other side of the label loss are what hold the
head away from its label optimum. Also: the norm of each term's gradient averaged over the batches (the
systematic part; a single batch's norm is mostly noise) and the cosines between the averaged gradients.

The value target is the game's outcome (the bank's ``winner``); the policy term is a PROXY: the bank's
actions are greedy and it stores no advantage, so it uses the minibatch-normalised Monte-Carlo advantage
(outcome minus value) on the greedy action at ratio 1. Reads the P* checkpoint named by the dump's tables
file and the archive's scoring bank; writes nothing.
"""
import sys
from pathlib import Path

import numpy as np
import torch as th

from agents.model.extra_obs_keys import zero_extra_obs
from agents.training import belief_bank as BB
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
N = 128
COEF = {"opp_belief_aux_coef": 0.05, "move_belief_coef": 0.05, "item_belief_coef": 0.05, "move_belief_latent_coef": 0.05}
GATES = {"hidden_team_set": True, "move_belief": True, "item": True, "move_latent": True}
NAMES = ("move label BCE x0.05", "move latent x0.05", "species set x0.05 (moves_weight 1)", "item x0.05",
         "value 0.5*MSE", "entropy -0.05*H", "policy (MC-advantage proxy)")
acc = {n: [th.zeros_like(p) for p in P] for n in NAMES}
dal = {n: [] for n in NAMES}
nrm = {n: [] for n in NAMES}


def take(name, term):
    g = th.autograd.grad(term, P, retain_graph=True, allow_unused=True)
    g = [th.zeros_like(p) if x is None else x for x, p in zip(g, P)]
    for a, x in zip(acc[name], g):
        a += x
    dal[name].append(float(sum((x * p).sum() for x, p in zip(g, P))))
    nrm[name].append(float(th.sqrt(sum((x ** 2).sum() for x in g))))


for rep in range(NB):
    sel = np.sort(rng.choice(len(rows["obs"]), N, replace=False))
    mb = th.as_tensor(rows["mask"][sel], dtype=th.float32)
    ob = {"observation": th.as_tensor(rows["obs"][sel], dtype=th.float32), "action_mask": mb}
    ob.update(zero_extra_obs(fe, batch=N, device="cpu"))
    masks = rows["mask"][sel].astype(bool)
    labels = {k: th.as_tensor(rows[k][sel]) for k in FT.LABEL_KEYS}
    won = th.as_tensor((rows["winner"][sel] == 2).astype(np.float32)).reshape(-1, 1)
    with th.no_grad():
        model.policy.evaluate_actions(ob, th.zeros(N, dtype=th.long), action_masks=masks)
        act = model.policy._last_pi_distribution.distribution.probs.argmax(-1)
    values, logp, ent = model.policy.evaluate_actions(ob, act, action_masks=masks)
    got = {}
    for site in ("hidden_move", "revealed", "latent"):
        for row, term, _m in BB.compute(fe, labels, COEF, GATES, site, params={"moves_weight": 1.0}):
            got[row.name] = got[row.name] + term if row.name in got else term
    take(NAMES[0], got["move_belief"]); take(NAMES[1], got["move_latent"])
    take(NAMES[2], got["hidden_team_set"]); take(NAMES[3], got["item"])
    take(NAMES[4], 0.5 * th.nn.functional.mse_loss(values, won))
    take(NAMES[5], -0.05 * ent.mean())
    adv = (won - values).detach().reshape(-1)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    take(NAMES[6], -(adv * logp.reshape(-1)).mean())
    print(f"batch {rep}: " + "; ".join(f"{n.split(' x')[0].split(' (')[0]} {dal[n][-1]:+.2e}" for n in NAMES), flush=True)

print(f"\n{PANEL}: {NB} batches x {N} rows; |W| {float(head.weight.norm()):.3f} |b| {float(head.bias.norm()):.3f}")
print("| term | d(term)/d(scale): mean [min, max] over batches | batches with a negative sign | mean batch |g| | |mean g| |")
print("|---|---|---|---|---|")
flat = {}
for n in NAMES:
    d = np.array(dal[n]); m = [a / NB for a in acc[n]]
    flat[n] = th.cat([x.reshape(-1) for x in m])
    print(f"| {n} | {d.mean():+.3e} [{d.min():+.3e}, {d.max():+.3e}] | {int((d < 0).sum())} / {NB} | {np.mean(nrm[n]):.3e} | {float(flat[n].norm()):.3e} |")
tot = sum(np.array(dal[n]) for n in NAMES)
print(f"| SUM of the seven | {tot.mean():+.3e} [{tot.min():+.3e}, {tot.max():+.3e}] | {int((tot < 0).sum())} / {NB} | | {float(sum(flat.values()).norm()):.3e} |")
cons = sum(flat[n] for n in NAMES[2:])
print("\ncosine(mean label gradient, mean gradient of ...):")
for n in NAMES[1:]:
    print(f"  {n}: {float((flat[NAMES[0]] * flat[n]).sum() / (flat[NAMES[0]].norm() * flat[n].norm() + 1e-30)):+.3f}")
print(f"  every consumer of the publication (species + item + value + entropy + policy proxy): "
      f"{float((flat[NAMES[0]] * cons).sum() / (flat[NAMES[0]].norm() * cons.norm() + 1e-30)):+.3f}; "
      f"|mean consumer g| / |mean label g| = {float(cons.norm() / flat[NAMES[0]].norm()):.2f}")
