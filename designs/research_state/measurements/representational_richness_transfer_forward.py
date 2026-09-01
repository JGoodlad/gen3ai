"""M2 / part 1 + 4 — RICHNESS forwards: per-phase feature dumps for the participation ratio.

Reuses the plasticity audit's own machinery verbatim (`tmp/plast_forward.py`, 2026-08-28) so the
numbers land on the same scale as its published `pi_features` 50.24 (v8) / 20.59 (gen). Same shared
per-era state set (`/tmp/plast/states_<era>.npz`, n=3000 from each PARENT's own eval_traces), same
six hooks, same batch size, same "unknown"-default auxiliary obs channels.

What is NEW here is only the MODEL LIST: the gen-era folds the audit never forwarded (R3ACTION,
R4ACTION, COMPFOLD, the two no-fold controls) and a within-run rev-1 checkpoint ladder for the
maturity question.

Run (gen era, current code):
  nice -n 15 python designs/research_state/measurements/representational_richness_transfer_forward.py v9
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

The v8-era arm needs the era worktree instead, because current code cannot load a 2992-dim obs:
  PYTHONPATH=/tmp/probeP_v8era/src python ... v8

Reads models/ READ-ONLY. Writes .npy under /tmp/m2rich/fwd/<era>/.
"""
import json
import os
import sys

import numpy as np
import torch
from sb3_contrib import MaskablePPO

M = os.environ.get("GEN3AI_MODELS_DIR", "/home/goodlad/dev/gen3ai/models")
OUT = "/tmp/m2rich/fwd"
STATES = "/tmp/plast"          # the audit's shared state sets, reused as-is
BATCH = 128

FIN, FINI = "final_model.zip", "final_model_interrupted.zip"


def _ck(run, step):
    return f"{M}/{run}/checkpoints/checkpoint_{step}_steps.zip"


# rev-1's own ladder, ~evenly spaced over its 25M — the within-run maturity curve.
REV1_LADDER = [2400000, 4800000, 8045088, 12101808, 16001808, 20073792, 24988992]

MODELS = {
    "v9": {
        # folds the audit never forwarded
        "R3ACTION": f"{M}/ai_v9_70_R3ACTION_0828/{FIN}",
        "R4ACTION": f"{M}/ai_v9_76_R4ACTION_0830/{FIN}",
        "COMPFOLD": f"{M}/ai_v9_91_COMPFOLD_0831/{FIN}",
        # no-fold controls: what +3M of ordinary training does to richness
        "R2CTRL": f"{M}/ai_v9_58_R2CTRL_0827/{FIN}",
        "R2PLAIN": f"{M}/ai_v9_62_R2PLAIN_0827/{FIN}",
        # maturity ladder
        **{f"rev1_step{s}": _ck("ai_v9_29_rev1_0823", s) for s in REV1_LADDER},
    },
    # v8 parent + fold + forks are already dumped by the audit at /tmp/plast/fwd/v8. What is new
    # is a v8-LINEAGE maturity ladder: the era's own archive keeps nothing below ~149M, so this
    # cannot reach back to 25M, but it does answer "was v8 already rich at 149M, or did the last
    # 128M of training buy the gap?" — the only handle on the maturity confound on the v8 side.
    "v8": {
        "v8_03_step149598621": _ck("ai_v8_03_zarch_control_0718", 149598621),
        "v8_03_step200364858": _ck("ai_v8_03_zarch_control_0718", 200364858),
        "v8_03_step267612744": _ck("ai_v8_03_zarch_control_0718", 267612744),
        "v8_04_step269716291": _ck("ai_v8_04_distill_4teacher_0722", 269716291),
    },
}

HOOKS = ["features_extractor.pokemon_encoder",
         "features_extractor.team_transformer",
         "features_extractor.projection",
         "features_extractor.cls_pool",
         "mlp_extractor.policy_net",
         "mlp_extractor.value_net"]


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


def flat(t):
    if isinstance(t, (tuple, list)):
        t = t[0]
    if not torch.is_tensor(t):
        return None
    return t.detach().reshape(t.shape[0], -1).float()


def run_model(path, obs_vec, action_mask):
    model = MaskablePPO.load(path, env=None, device="cpu")
    p = model.policy
    p.set_training_mode(False)
    named = dict(p.named_modules())
    store, handles = {}, []
    for h in HOOKS:
        if h in named:
            def mk(name):
                def fn(_m, _i, o):
                    f = flat(o)
                    if f is not None:
                        store.setdefault(name, []).append(f)
                return fn
            handles.append(named[h].register_forward_hook(mk(h)))
    logits, values, pif, vff = [], [], [], []
    with torch.no_grad():
        for i in range(0, obs_vec.shape[0], BATCH):
            ob = make_obs(p, obs_vec[i:i + BATCH], action_mask[i:i + BATCH])
            feats = p.extract_features(ob)
            pf, vf = feats if isinstance(feats, tuple) else (feats, feats)
            pif.append(pf.detach().float())
            vff.append(vf.detach().float())
            logits.append(p.get_distribution(ob).distribution.logits.detach().float())
            values.append(p.predict_values(ob).detach().float().reshape(-1))
    for h in handles:
        h.remove()
    res = {"logits": torch.cat(logits).numpy(),
           "values": torch.cat(values).numpy(),
           "pi_features": torch.cat(pif).numpy(),
           "vf_features": torch.cat(vff).numpy()}
    for k, v in store.items():
        res[k] = torch.cat(v).numpy()
    del model
    return res


def main():
    era = sys.argv[1]
    torch.set_num_threads(2)
    os.makedirs(f"{OUT}/{era}", exist_ok=True)
    S = np.load(f"{STATES}/states_{era}.npz")
    obs, am = S["obs"], S["action_mask"]
    print(f"[{era}] shared state set n={obs.shape[0]} dim={obs.shape[1]}")
    manifest = {}
    for name, path in MODELS[era].items():
        if not os.path.exists(path):
            print(f"  MISSING {name}: {path}")
            manifest[name] = {"MISSING": path}
            continue
        if os.path.exists(f"{OUT}/{era}/{name}__pi_features.npy"):
            print(f"  cached {name}")
            manifest[name] = {"path": path, "cached": True}
            continue
        r = run_model(path, obs, am)
        for k, v in r.items():
            np.save(f"{OUT}/{era}/{name}__{k.replace('.', '_')}.npy", v)
        manifest[name] = {"path": path,
                          "shapes": {k: list(v.shape) for k, v in r.items()}}
        print(f"  {name:18s} pi{list(r['pi_features'].shape)} done")
    json.dump(manifest, open(f"{OUT}/{era}/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
