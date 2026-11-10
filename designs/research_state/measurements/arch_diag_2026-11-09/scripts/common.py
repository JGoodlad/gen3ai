"""Shared loaders for the architecture diagnostic (2026-11-09). CPU only, read-only on models/ and the archive.

Run with PYTHONPATH=<a checkout at P*'s pin 9c0334890>/src so P*'s own code reads the rows."""
from __future__ import annotations

import glob
import os

import numpy as np

REPO = "/home/goodlad/dev/gen3ai"
PSTAR = f"{REPO}/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip"
PSTAR_SHA256 = "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"
ARCHGEN = "/home/goodlad/gen3ai_archive/arch_generalisation_2026-10-26"
PANELS = ("home72", "H", "F", "P72")


def load_model():
    import torch
    torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "1")))
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings
    model, _ = load_foreign_opponent(PSTAR, current_model_version(load_mappings()), device="cpu")
    model.policy.set_training_mode(False)
    return model


def obs_dict(model, obs, mask):
    import torch as th

    from agents.model.extra_obs_keys import zero_extra_obs
    fe = model.policy.features_extractor
    mb = th.as_tensor(mask, dtype=th.float32)
    ob = {"observation": th.as_tensor(obs, dtype=th.float32), "action_mask": mb}
    ob.update(zero_extra_obs(fe, batch=len(mb), device="cpu"))
    return ob


def archgen_rows(panel):
    """P* greedy self-play rows (2026-10-25): p1 (side 0) pilots a PANEL team, p2 (side 1) a home72 team."""
    z = np.load(os.path.join(ARCHGEN, f"rows_{panel}.npz"))
    d = {k: z[k] for k in z.files}
    d["y"] = (d["winner"] == d["side"] + 1).astype(np.float64)     # winner is 1 / 2 (0 = tie -> a loss, as trained)
    return d


def tie_rate_rows():
    """The 2026-11-06 tie-rate population (16,785 rows): fixture + rb_x60t_s3502 eval traces + its rollout scans."""
    rows, masks, src = [], [], []
    fx = np.load(os.path.join(REPO, "src/agents/model/compile_parity_obs.npz"))
    rows.append(fx["obs"]); masks.append(fx["action_mask"]); src += ["fixture"] * len(fx["obs"])
    for step in ("step_282000232", "step_284000065"):
        for f in sorted(glob.glob(os.path.join(REPO, "models/rb_x60t_s3502/eval_traces", step, "*", "*_states.npz"))):
            d = np.load(f, allow_pickle=True)
            if "action_mask" not in d.files or len(d["obs"]) == 0:
                continue
            rows.append(d["obs"].astype(np.float32)); masks.append(d["action_mask"].astype(bool))
            src += [f"eval:{step}:{f.split('/')[-2]}"] * len(d["obs"])
    for f in sorted(glob.glob(os.path.join(REPO, "models/rb_x60t_s3502/behaviour_violation_*_rows.npz"))):
        d = np.load(f, allow_pickle=True)
        rows.append(d["scan_obs__observation"].astype(np.float32)); masks.append(d["scan_obs__action_mask"].astype(bool))
        src += ["rollout_scan"] * len(d["scan_obs__observation"])
    obs = np.concatenate(rows); mask = np.concatenate(masks); src = np.array(src)
    ok = mask.any(1)
    return obs[ok], mask[ok], src[ok]
