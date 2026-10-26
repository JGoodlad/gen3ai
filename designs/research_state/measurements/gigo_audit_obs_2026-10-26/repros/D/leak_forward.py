"""(5) Leak test: the production learner's forward with the training-only label keys REAL vs POISONED vs ABSENT.
Compares values, log_prob, entropy and the full masked log-prob row via evaluate_actions_functional / evaluate_actions,
in eval mode, with grad enabled (the train path computes stashes only under grad)."""
import sys, numpy as np, torch as th
from agents.training import learner_golden as L

model = L.build_learner(trunk_precision="fp32")
data = L.load_buffer_into(model)
rb = model.rollout_buffer
pol = model.policy
pol.eval()
keys = list(rb.observations.keys())
print("obs keys:", keys)
B = 64
def take(k):
    a = rb.observations[k].reshape((-1,) + rb.observations[k].shape[2:])[:B]
    return th.as_tensor(np.ascontiguousarray(a))
real = {k: take(k) for k in keys}
acts = th.as_tensor(rb.actions.reshape(-1)[:B]).long()
masks = th.as_tensor(rb.action_masks.reshape(-1, rb.action_masks.shape[-1])[:B])
g = th.Generator().manual_seed(7)
poison = {}
for k, v in real.items():
    if k in ("observation", "action_mask"):
        poison[k] = v.clone(); continue
    if v.dtype in (th.int64, th.int32):
        poison[k] = th.randint(-1, 30, v.shape, generator=g).to(v.dtype)
    else:
        poison[k] = (th.rand(v.shape, generator=g) * 300).to(v.dtype)
absent = {k: real[k] for k in ("observation", "action_mask") if k in real}
def run(o):
    th.manual_seed(0)
    with th.enable_grad():
        if hasattr(pol, "evaluate_actions_functional"):
            out = pol.evaluate_actions_functional(o, acts, action_masks=masks)
        else:
            out = pol.evaluate_actions(o, acts, action_masks=masks)
    stash = {}
    fe = pol.features_extractor
    for name in ("move_belief_logits", "spread_belief", "spread_nature_logits", "spread_ev", "hp_type_logits", "item_logits"):
        try:
            t = fe.belief_supervision(name)
        except Exception as e:
            t = None
        if t is not None:
            stash[name] = t.detach().clone()
    return [x.detach().clone() for x in out if isinstance(x, th.Tensor)], stash
r, rs = run(real); p, ps = run(poison)
try:
    a, as_ = run(absent); have_absent = True
except Exception as e:
    print("ABSENT-keys forward raised:", type(e).__name__, str(e)[:200]); have_absent = False
ok = True
for i, (x, y) in enumerate(zip(r, p)):
    eq = th.equal(x, y); ok &= eq
    print(f"output[{i}] shape {tuple(x.shape)} real==poison: {eq}  maxdiff {float((x.float()-y.float()).abs().max()):.3g}")
    if have_absent:
        print(f"output[{i}] real==absent: {th.equal(x, a[i])}")
for k in rs:
    eq = th.equal(rs[k], ps[k]); ok &= eq
    print(f"stash {k}: real==poison {eq}")
print("LEAK-FREE" if ok else "LEAK DETECTED")
