"""Shared helpers for the architecture-generalisation probes (2026-10-26). CPU only, read-only on models/."""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List

import numpy as np

PSTAR = "/home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip"
PANEL_ROOT = "/home/goodlad/dev/gen3ai-reads/x51_2026-10-20"
PANELS = {"home72": "teams_home72", "H": "teams_heldout72_hpfix", "F": "teams_forum72", "P72": "teams_p72"}
OUT = Path(__file__).resolve().parent.parent


def panel_texts(panel: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(PANEL_ROOT, PANELS[panel], "*.txt")))
    assert len(files) == 72, (panel, len(files))
    return [open(f).read() for f in files]


def pack(texts: List[str]) -> List[str]:
    from utils.teambuilder import Gen3Teambuilder
    packed = Gen3Teambuilder(texts).packed_teams
    assert len(packed) == len(texts), "a team failed validation"
    return packed


def load_model():
    import torch
    torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "3")))
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings
    model, _ = load_foreign_opponent(PSTAR, current_model_version(load_mappings()), device="cpu")
    model.policy.set_training_mode(False)
    return model


def forward(model, obs: np.ndarray, mask: np.ndarray):
    """One no-grad forward: (masked probs [B,11], P(win) [B]) plus the extractor (for its stashes)."""
    import torch as th
    from agents.model.extra_obs_keys import zero_extra_obs
    fe = model.policy.features_extractor
    mb = th.tensor(mask, dtype=th.float32)
    ob = {"observation": th.tensor(obs, dtype=th.float32), "action_mask": mb}
    ob.update(zero_extra_obs(fe, batch=len(mb), device="cpu"))
    with th.no_grad():
        logits = model.policy.get_distribution(ob).distribution.logits
        probs = th.softmax(logits + (mb - 1.0) * 1e9, dim=-1).numpy()
        win = th.sigmoid(fe.last_win_prob_logits.reshape(-1)).numpy()
    return probs, win, fe


def lib():
    from utils.rust_env import ffi
    return ffi.load(ffi.default_path("release"))
