"""PROBE F — standing ACID TEST for the load path + timing probe.

The eval_traces ship the exact snapshot.zip that produced the recorded logits.
Forwarding that snapshot on the recorded obs in THIS worktree must reproduce the
recorded logits; if it does, the auxiliary obs-dict channels the traces do not
record are being filled correctly and every downstream gradient is on the real
network's real forward.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO

M = "/home/goodlad/dev/gen3ai/models"
OUT = "/tmp/probeF"


def make_obs(policy, obs_vec, action_mask):
    space = policy.observation_space
    n = obs_vec.shape[0]
    out = {}
    for k, sp in space.spaces.items():
        if k == "observation":
            out[k] = torch.as_tensor(obs_vec, dtype=torch.float32)
            continue
        if k == "action_mask":
            out[k] = torch.as_tensor(action_mask.astype(np.int8))
            continue
        shape = (n,) + tuple(sp.shape)
        if np.issubdtype(sp.dtype, np.integer):
            fill = -1 if float(np.min(sp.low)) <= -1 else 0
            out[k] = torch.full(shape, fill, dtype=torch.long)
        else:
            out[k] = torch.zeros(shape, dtype=torch.float32)
    return out


def main():
    torch.set_num_threads(2)
    run = "ai_v9_53_R2F5a_0826"
    step = "step_28000032"
    snap = f"{M}/{run}/eval_traces/{step}/snapshot.zip"
    files = sorted(glob.glob(f"{M}/{run}/eval_traces/{step}/*/*_states.npz"))
    obs, lg, mk = [], [], []
    for f in files:
        d = np.load(f)
        k = d["has_state"].astype(bool)
        if k.sum() == 0:
            continue
        obs.append(d["obs"][k]); lg.append(d["logits"][k]); mk.append(d["action_mask"][k])
        if sum(x.shape[0] for x in obs) >= 256:
            break
    obs = np.concatenate(obs)[:256]
    lg = np.concatenate(lg)[:256]
    mk = np.concatenate(mk)[:256]

    t0 = time.time()
    model = MaskablePPO.load(snap, env=None, device="cpu")
    p = model.policy
    p.set_training_mode(False)
    tload = time.time() - t0

    got = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, obs.shape[0], 64):
            ob = make_obs(p, obs[i:i + 64], mk[i:i + 64])
            got.append(p.get_distribution(ob).distribution.logits.float())
    got = torch.cat(got).numpy()
    tfwd = time.time() - t0

    m = mk.astype(bool)
    d = np.abs(got - lg)
    res = {
        "worktree": os.getcwd(),
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "snapshot": snap, "n": int(obs.shape[0]),
        "max_abs_diff_all": float(d.max()),
        "mean_abs_diff_all": float(d.mean()),
        "max_abs_diff_legal": float(d[m].max()),
        "corr_all": float(np.corrcoef(got.ravel(), lg.ravel())[0, 1]),
        "top1_agreement_legal": float(
            (np.where(m, got, -1e9).argmax(1) == np.where(m, lg, -1e9).argmax(1)).mean()),
        "load_seconds": round(tload, 2),
        "fwd_seconds_256_nograd": round(tfwd, 2),
    }
    # param census
    tr = [(n_, q) for n_, q in p.named_parameters() if q.requires_grad]
    res["n_trainable_tensors"] = len(tr)
    res["n_trainable_params"] = int(sum(q.numel() for _, q in tr))
    print(json.dumps(res, indent=1))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/acid_test.json", "w"), indent=1)

    if "--timegrad" in sys.argv:
        p.set_training_mode(True)
        ob = make_obs(p, obs[:64], mk[:64])
        t0 = time.time()
        dist = p.get_distribution(ob)
        loss = -dist.log_prob(torch.zeros(64, dtype=torch.long)).mean()
        loss.backward()
        print("bwd_seconds_64:", round(time.time() - t0, 2))


if __name__ == "__main__":
    main()
