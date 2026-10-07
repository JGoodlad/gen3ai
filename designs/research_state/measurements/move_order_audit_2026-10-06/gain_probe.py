"""Is Source 2 (op move-cell non-equivariance with all four legal) the per-slot learned out_gain?
Compare the PRE-gain cells (op.stash.raw_block) with the post-gain ones under a request-block permutation."""
import os
import numpy as np
import torch as th

from agents.training import learner_golden as LG
from main.train.production_args import production_args
from agents.observation.schema import build_schema

z = np.load("src/agents/training/learner_golden_buffer.npz")
obs = th.as_tensor(z["obs:observation"].reshape(-1, 2761))
m = LG.build_learner(args=production_args(), perturb_keyed=True)
fe = m.policy.eval().features_extractor
sl = build_schema(fe.layout).slices()["reactive.active_req_moves"]
per = (sl.stop - sl.start) // 3
if os.environ.get("ALL_LEGAL"):
    obs[:, sl.start + 2 * per: sl.start + 3 * per] = 1.0
perm = [2, 0, 3, 1]
ob2 = obs.clone()
for blk in range(3):
    s = sl.start + blk * per
    ob2[:, s:s + 4] = obs[:, s:s + 4][:, perm]
op = [mod for mod in fe.modules() if hasattr(mod, "out_gain") and hasattr(mod, "pointer_cells")][0]


def cells(o):
    with th.no_grad():
        fe({"observation": o})
        post = fe.last_pointer_inputs.move_cells.clone()
        pre, _ = op.pointer_cells(op.stash.raw_block)
        return post, pre.clone()


(a, ap), (b, bp) = cells(obs), cells(ob2)
print("post-gain move-cell dev", (a[:, perm] - b).abs().max().item())
print("PRE-gain  move-cell dev", (ap[:, perm] - bp).abs().max().item())
g = op.out_gain.data
o0 = op.incoming_dim
print("outgoing per-slot gains (perturbed learner):", g[o0:o0 + 16].view(4, 4).tolist())
