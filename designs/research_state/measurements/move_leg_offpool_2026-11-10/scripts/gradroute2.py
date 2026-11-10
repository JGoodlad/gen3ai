"""Part 2: at P*, on IN-POOL rows (the home72 scoring bank; nothing is trained), the gradient each term of the
training objective sends into `move_head.weight`: the move label loss at its production coefficient (route A),
the value loss and the policy's log-probability / entropy (route C, through the head's publications); then the
same under the BUILT mode `--belief-grad-mode move_label_only`, applied by the runtime setter a fork's load calls
(`set_belief_grad_mode`). It ends with the per-leg read of the fine-tune's objective on X68's `train_0` shard.
A first version emulated the cut by patching `_publish_belief`; the built mode routes the move posterior
through `_publish_move_belief`, so that emulation cut nothing and its output was discarded."""
import numpy as np, torch as th
from pathlib import Path
from main.offpool_belief import score as SC, finetune as FT
from agents.model.extra_obs_keys import zero_extra_obs
D = Path.home() / "gen3ai_archive/move_leg_diag_2026-11-10"
B = Path.home() / "gen3ai_archive/offpool_belief_2026-11-09/bank/home72/rows.npz"
th.manual_seed(0); th.set_num_threads(4)
model, _ck = SC.load_checkpoint(str(np.load(D / "dump" / "pstar__tables.npz")["ckpt_path"]))
fe = model.policy.features_extractor
FT.trained_parameters(model)
W = fe.move_belief.move_head.weight
z = np.load(B); rows = {k: z[k] for k in z.files}
rng = np.random.default_rng(1)
N = 128
A = Path.home() / "gen3ai_archive/offpool_belief_2026-11-09/x68"


def set_cut(cut):
    """The built lever itself: the runtime setter a fork's load calls."""
    fe.set_belief_grad_mode("move_label_only" if cut else "shaping")


def rowstat(gw):
    if gw is None:
        return "NO PATH"
    rn = gw.norm(dim=1)
    return (f"|g| {float(gw.norm()):.3e}  median row {float(rn.median()):.2e}  row237 {float(rn[237]):.2e}  "
            f"rows>0: {int((rn > 0).sum())}  top {[(int(i), float(f'{float(rn[i]):.2e}')) for i in th.argsort(-rn)[:4]]}")


for rep in range(2):
    sel = np.sort(rng.choice(len(rows["obs"]), N, replace=False))
    won = th.as_tensor((rows["winner"][sel] == 2).astype(np.float32)).reshape(-1, 1)
    for cut in (False, True):
        set_cut(cut)
        terms = FT.belief_terms(model, rows, sel, "cpu")
        lab = 0.05 * terms["move_belief"]
        (gA,) = th.autograd.grad(lab, W, retain_graph=True, allow_unused=True)
        mb = th.as_tensor(rows["mask"][sel], dtype=th.float32)
        ob = {"observation": th.as_tensor(rows["obs"][sel], dtype=th.float32), "action_mask": mb}
        ob.update(zero_extra_obs(fe, batch=N, device="cpu"))
        masks = rows["mask"][sel].astype(bool)
        with th.no_grad():
            model.policy.evaluate_actions(ob, th.zeros(N, dtype=th.long), action_masks=masks)
            act = model.policy._last_pi_distribution.distribution.probs.argmax(-1)
        values, logp, ent = model.policy.evaluate_actions(ob, act, action_masks=masks)
        vl = 0.5 * th.nn.functional.mse_loss(values, won)
        (gV,) = th.autograd.grad(vl, W, retain_graph=True, allow_unused=True)
        (gP,) = th.autograd.grad(logp.mean(), W, retain_graph=True, allow_unused=True)
        (gE,) = th.autograd.grad(0.05 * ent.mean(), W, retain_graph=True, allow_unused=True)
        tag = "CUT " if cut else "live"
        if not cut:
            print(f"batch {rep}: values {float(values.min()):.3f}..{float(values.max()):.3f} mean {float(values.mean()):.3f}; "
                  f"won share {float(won.mean()):.3f}; move BCE {float(terms['move_belief']):.5f}; value MSE {float(2 * vl):.4f}", flush=True)
        print(f"  [{tag}] label 0.05*BCE      : {rowstat(gA)}")
        print(f"  [{tag}] value 0.5*MSE       : {rowstat(gV)}")
        print(f"  [{tag}] policy mean logp(a*) : {rowstat(gP)}   (sensitivity; PPO weights it by the advantage)")
        print(f"  [{tag}] entropy 0.05*H      : {rowstat(gE)}", flush=True)
        if not cut and gA is not None and gV is not None:
            cos = float((gA * gV).sum() / (gA.norm() * gV.norm() + 1e-30))
            print(f"  [live] |value| / |label| = {float(gV.norm() / gA.norm()):.2f}; cosine(label, value) = {cos:+.3f}", flush=True)
    set_cut(False)

# the fine-tune's own objective (X68's training rows), per leg, live and under the built mode
zt = np.load(A / "train_0" / "rows.npz"); rt = {k: zt[k] for k in zt.files}
sel = np.sort(np.random.default_rng(0).choice(len(rt["obs"]), 128, replace=False))
for cut in (False, True):
    set_cut(cut)
    terms = FT.belief_terms(model, rt, sel, "cpu")
    for g, t in terms.items():
        (gw,) = th.autograd.grad(t, W, retain_graph=True, allow_unused=True)
        print(f"  finetune objective [{'CUT ' if cut else 'live'}] {g:16s}: {rowstat(gw)}", flush=True)
set_cut(False)
