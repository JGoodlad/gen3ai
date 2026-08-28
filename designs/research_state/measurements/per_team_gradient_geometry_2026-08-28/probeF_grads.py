"""PROBE F step 2 — per-team LOSS GRADIENTS at the rev-2 fold student.

For each of the 9 R2 slice teams, and for 6 BATTLE-DISJOINT batches of 256
recorded decisions each, flatten dL/dtheta over every shared trainable parameter
of models/ai_v9_59_R2ACTION_0827/final_model.zip into one vector.

Two loss rows:
  distill  (PRIMARY)  — the fold's own target form: --distill-target action
                        --distill-topk 1 --distill-gate none, i.e. the masked
                        mean of -log pi_S(a_teacher) over legal actions, with
                        a_teacher = that team's slice teacher's argmax.
                        DEVIATION: the AWR weight w = clamp(exp(|A|/beta), 20)
                        is dropped (A is not recoverable offline), so every row
                        carries w = 1.  gate="none" fires on every on-pin row,
                        which is exactly what this batch is.
  pg       (SECONDARY, DIRECTIONAL ONLY) — -(A_hat * log pi_S(a_recorded)).mean()
                        with A_hat = per-batch-standardized within-battle TD
                        residual of the RECORDED critic (see build script).

Writes /tmp/probeF/grads.dat (float32 memmap) + /tmp/probeF/grads_index.json
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO

M = "/home/goodlad/dev/gen3ai/models"
OUT = "/tmp/probeF"
STUDENT = f"{M}/ai_v9_59_R2ACTION_0827/final_model.zip"
TEACHERS = {
    "F5a": f"{M}/ai_v9_53_R2F5a_0826/final_model.zip",
    "F5b": f"{M}/ai_v9_54_R2F5b_0826/final_model.zip",
    "F5c": f"{M}/ai_v9_55_R2F5c_0826/final_model.zip",
    "F5d": f"{M}/ai_v9_56_R2F5d_0826/final_model.zip",
    "F5e": f"{M}/ai_v9_57_R2F5e_0826/final_model.zip",
}
# --at init  measures the SAME geometry at the fold's INITIALIZATION (the parent
# checkpoint every fork and the student were forked from), which is where PCGrad
# would actually start operating.
PARENT = f"{M}/ai_v9_29_rev1_0823/final_model.zip"
N_FOLDS = 6
BATCH = 256
MICRO = 64
SEED = 0


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


def logits_of(policy, obs_vec, mask, grad=False):
    ctx = torch.enable_grad if grad else torch.no_grad
    outs = []
    with ctx():
        for i in range(0, obs_vec.shape[0], MICRO):
            ob = make_obs(policy, obs_vec[i:i + MICRO], mask[i:i + MICRO])
            outs.append(policy.get_distribution(ob).distribution.logits)
    return torch.cat(outs)


def battle_folds(fileid, rng, n_folds, per_fold):
    """Assign whole BATTLES to folds, then take up to `per_fold` states each.
    Disjoint batches must not share a game — decisions inside one battle are
    strongly correlated, so a within-team noise floor built from state-level
    splits would be optimistically high."""
    battles = np.unique(fileid)
    rng.shuffle(battles)
    folds = [[] for _ in range(n_folds)]
    for k, b in enumerate(battles):
        folds[k % n_folds].append(b)
    out = []
    for f in folds:
        idx = np.flatnonzero(np.isin(fileid, f))
        rng.shuffle(idx)
        out.append(np.sort(idx[:per_fold]))
    return out


def main():
    global STUDENT, LOSS_KINDS, TAG
    LOSS_KINDS = ("distill", "pg", "bc", "pgmc")
    TAG = ""
    if "--at-init" in sys.argv:
        STUDENT = PARENT
        LOSS_KINDS = ("distill", "bc")
        TAG = "_init"
    torch.set_num_threads(2)
    os.makedirs(OUT, exist_ok=True)
    S = np.load(f"{OUT}/states_per_team.npz")
    meta = json.load(open(f"{OUT}/states_per_team_meta.json"))
    teams = sorted(meta["teams"])
    rng = np.random.default_rng(SEED)

    # ---------- fold assignment ----------
    folds = {}
    for t in teams:
        folds[t] = battle_folds(S[f"{t}__fileid"], rng, N_FOLDS, BATCH)
        sizes = [len(f) for f in folds[t]]
        print(f"  folds[{t[:8]}] sizes={sizes}")

    # ---------- teacher argmax ----------
    tmode = {}
    for fk, path in TEACHERS.items():
        mine = [t for t in teams if meta["teams"][t]["teacher"] == fk]
        if not mine:
            continue
        tm = MaskablePPO.load(path, env=None, device="cpu")
        tp = tm.policy
        tp.set_training_mode(False)
        for t in mine:
            obs, mk = S[f"{t}__obs"], S[f"{t}__mask"]
            lg = logits_of(tp, obs, mk, grad=False).float().numpy()
            neg = (mk.astype(np.float32) - 1.0) * 1e9
            tmode[t] = (lg + neg).argmax(1).astype(np.int64)
            agree = float((tmode[t] == S[f"{t}__actions"]).mean())
            print(f"  teacher {fk} -> {t[:8]}  argmax==recorded_action {agree:.3f}")
        del tm
    del tp

    # ---------- student gradients ----------
    sm = MaskablePPO.load(STUDENT, env=None, device="cpu")
    sp = sm.policy
    sp.set_training_mode(True)
    params = [(n, p) for n, p in sp.named_parameters() if p.requires_grad]
    names = [n for n, _ in params]
    sizes = [int(p.numel()) for _, p in params]
    P = int(sum(sizes))
    print(f"student: {len(params)} tensors, {P} params")

    rows = []
    for loss_kind in LOSS_KINDS:
        for t in teams:
            for b in range(N_FOLDS):
                rows.append({"loss": loss_kind, "team": t, "fold": b})
    G = np.memmap(f"{OUT}/grads{TAG}.dat", dtype=np.float32, mode="w+",
                  shape=(len(rows), P))

    diag = []
    t_start = time.time()
    for r, spec in enumerate(rows):
        t, b, kind = spec["team"], spec["fold"], spec["loss"]
        idx = folds[t][b]
        obs = S[f"{t}__obs"][idx]
        mk = S[f"{t}__mask"][idx]
        neg_np = (mk.astype(np.float32) - 1.0) * 1e9
        if kind == "distill":
            tgt = torch.as_tensor(tmode[t][idx])
            w = None
        elif kind == "bc":
            tgt = torch.as_tensor(S[f"{t}__actions"][idx].astype(np.int64))
            w = None
        else:
            a = torch.as_tensor(S[f"{t}__actions"][idx].astype(np.int64))
            key = "adv_mc" if kind == "pgmc" else "adv_proxy"
            adv = S[f"{t}__{key}"][idx].astype(np.float64)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # PPO-style per-batch norm
            tgt, w = a, torch.as_tensor(adv, dtype=torch.float32)

        sp.zero_grad(set_to_none=True)
        n = len(idx)
        tot = 0.0
        for i in range(0, n, MICRO):
            sl = slice(i, min(i + MICRO, n))
            ob = make_obs(sp, obs[sl], mk[sl])
            lg = sp.get_distribution(ob).distribution.logits
            neg = torch.as_tensor(neg_np[sl])
            logp = F.log_softmax(lg + neg, dim=-1)
            picked = logp.gather(1, tgt[sl].reshape(-1, 1)).reshape(-1)
            if w is None:
                loss = -picked.sum() / n           # masked mean over the batch
            else:
                loss = -(w[sl] * picked).sum() / n
            loss.backward()
            tot += float(loss) * 1.0
        vec = torch.cat([(p.grad if p.grad is not None
                          else torch.zeros_like(p)).reshape(-1)
                         for _, p in params]).numpy()
        G[r] = vec
        diag.append({"row": r, **spec, "n": int(n), "loss_value": tot,
                     "grad_norm": float(np.linalg.norm(vec)),
                     "nonzero_params": int((vec != 0).sum())})
        if r % 12 == 0 or r == len(rows) - 1:
            el = time.time() - t_start
            print(f"  row {r+1}/{len(rows)} {kind} {t[:8]} f{b} "
                  f"|g|={diag[-1]['grad_norm']:.4g} nz={diag[-1]['nonzero_params']} "
                  f"[{el:.0f}s]")
    G.flush()

    json.dump({"rows": rows, "diag": diag, "param_names": names,
               "param_sizes": sizes, "P": P, "n_rows": len(rows),
               "student": STUDENT, "teachers": TEACHERS,
               "n_folds": N_FOLDS, "batch": BATCH, "seed": SEED,
               "fold_index": {t: [f.tolist() for f in folds[t]] for t in teams}},
              open(f"{OUT}/grads_index{TAG}.json", "w"))
    print("wrote", f"{OUT}/grads{TAG}.dat", G.shape)


if __name__ == "__main__":
    main()
