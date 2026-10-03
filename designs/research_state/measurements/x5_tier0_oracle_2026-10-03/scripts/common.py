"""X5 Tier 0 oracle counterfactual: shared setup.

- The decision source is the M5 Lane S bank (`m5_laneS/bank_v1`, 20,712 decisions in 580 battles). Each battle
  record carries BOTH packed teams, so the opponent's true team is known at every banked decision (G-4 VERIFIED for
  this bank). The observation rows are RE-ENCODED through the Rust core (`main.policy_spectrum.reader.reencode`),
  which byte-checks every row against the recorded sha (gate 1).
- The op runs on a FRESH production-surface learner (`agents.training.learner_golden.build_learner`, seeded, one
  thread). Every learned belief delta is zero-initialised, so every belief the op reads is the cold-start Smogon one;
  the op's hidden-species input (`T0SpeciesPrior`) is parameter-free, so R0 is EXACTLY what production computes,
  checkpoint or not. Nothing under `models/` is read.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from utils.paths import repo_path

BANK = repo_path("designs", "research_state", "measurements", "m5_laneS", "bank_v1")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
CACHE = Path("/tmp/x5tier0_cache")          # re-encoded rows (229 MB); a cache only, never an input of record
THREADS = 4


def species_num(name: str):
    from agents import gen3_data
    sp = gen3_data.species.get(re.sub(r"[^a-z0-9]", "", name.lower()))
    return sp.num if sp else None


def packed_species(packed: str) -> List[str]:
    """Species of a packed team: field 1 (species) or field 0 (nickname == species when field 1 is empty)."""
    out = []
    for mon in packed.split("]"):
        f = mon.split("|")
        out.append(f[1] if len(f) > 1 and f[1] else f[0])
    return out


def _as_dict(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def load_rows():
    """(bank, rows [N, obs_dim] float32, masks [N, 11] bool, gate). Cached by the bank's content sha."""
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import reencode
    bank = load_bank(BANK)
    sha = bank.manifest["content_sha256"]
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"rows_{sha[:16]}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        gate = json.loads(str(z["gate"]))
        return bank, z["rows"], z["masks"], gate
    rows, masks, gate = reencode(bank, workers=2)
    np.savez(f, rows=rows, masks=masks, gate=json.dumps(gate, sort_keys=True))
    return bank, rows, masks, gate


def opp_true_team(bank) -> Dict[str, Dict[str, List[int]]]:
    """battle_id -> {side: the OPPONENT's true species nums (as seen by `side`)}."""
    out = {}
    for b in bank.battles:
        teams = {}
        for side, other in (("p1", "p2"), ("p2", "p1")):
            packed = _as_dict(getattr(b, other))["team"]
            nums = [species_num(s) for s in packed_species(packed)]
            if any(n is None for n in nums):
                raise RuntimeError(f"{b.battle_id}: unparsed species in {packed_species(packed)}")
            teams[side] = nums
        out[b.battle_id] = teams
    return out


def build_extractor():
    """The fresh production-surface features extractor (eval mode) and its damage op."""
    from agents.training.learner_golden import build_learner
    model = build_learner()
    fx = model.policy.features_extractor
    fx.eval()
    return fx, fx.damage_op


def unpack(fx, rows: np.ndarray, masks: np.ndarray):
    from agents.model.extra_obs_keys import zero_extra_obs
    ob = {"observation": torch.as_tensor(rows), "action_mask": torch.as_tensor(masks.astype(np.float32))}
    ob.update(zero_extra_obs(fx, batch=len(rows)))
    return fx.unpack(ob)


def select_ctx(ctx, idx: torch.Tensor):
    """Row-select every batch-leading tensor of an ExtractorContext (for the parity drivers)."""
    import dataclasses
    B = ctx.batch_size
    kw = {}
    for f in dataclasses.fields(ctx):
        v = getattr(ctx, f.name)
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B:
            kw[f.name] = v[idx]
        else:
            kw[f.name] = v
    kw["batch_size"] = int(idx.numel())
    return dataclasses.replace(ctx, **kw)


def sha_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
