"""Same-weights output comparison of the END-STATE policy across two checkouts.

    cd <tree>/src && PYTHONPATH=<tree>/src python outdiff.py save    <pt>   # build (seed 0) + perturb, save sd/inputs/outputs
    cd <tree>/src && PYTHONPATH=<tree>/src python outdiff.py compare <pt>   # build, LOAD the saved sd, compare outputs
"""
import json
import sys

import numpy as np
import torch

mode, path = sys.argv[1], sys.argv[2]
torch.set_num_threads(1)

from agents.model.endstate_facts_test import _build_real_policy, _toggles  # noqa: E402
from main.train.arch_arms import arm_overlay  # noqa: E402
import agents.model.extractor_build as EB  # noqa: E402

END = arm_overlay("endstate")
pol = _build_real_policy(**_toggles(**END))[0].policy
pol.eval()


def outputs(pol, x, m):
    fe = pol.features_extractor
    with torch.no_grad():
        pi, vf = fe({"observation": x})
        p = fe.last_pointer_inputs
        lat = pol.mlp_extractor.forward_actor(pi)
        logits = pol.pointer_head(lat, p.move_tokens, p.move_valid, p.team_tokens, p.move_cells, p.switch_cells)
        wp = fe.last_win_prob_logits
        val = pol.predict_values({"observation": x})
        masked = torch.where(m, logits, torch.full_like(logits, -1e9))
        logp = torch.log_softmax(masked, -1)
    return {"pi": pi, "vf": vf, "move_cells": p.move_cells, "switch_cells": p.switch_cells, "win_prob_logits": wp,
            "logits": logits, "value": val, "legal_logp": torch.where(m, logp, torch.zeros_like(logp))}


if mode == "save":
    from agents.model.parity_probe import perturb_
    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    perturb_(pol)
    dim = Gen3ObservationEncoder(load_mappings()).dimension
    o1, m1 = load_parity_rows(dim)
    g = np.load("agents/training/learner_golden_buffer.npz")
    o2 = g["obs:observation"].reshape(-1, dim)
    m2 = g["action_masks"].reshape(-1, g["action_masks"].shape[-1]) > 0.5
    o3 = torch.rand(8, dim, generator=torch.Generator().manual_seed(7)).numpy()
    m3 = np.ones((8, m1.shape[1]), bool)
    sets = {"parity_rows": (o1, m1), "golden_buffer": (o2, m2), "rand8": (o3, m3)}
    rec = {"sd": {k: v.clone() for k, v in pol.state_dict().items()}, "inputs": {}, "outputs": {}}
    for name, (o, m) in sets.items():
        x, mm = torch.as_tensor(o, dtype=torch.float32), torch.as_tensor(m, dtype=torch.bool)
        rec["inputs"][name] = (x, mm)
        rec["outputs"][name] = outputs(pol, x, mm)
    torch.save(rec, path)
    print(json.dumps({"saved": path, "src": EB.__file__, "rows": {k: int(v[0].shape[0]) for k, v in sets.items()}}))
else:
    rec = torch.load(path, weights_only=False)
    missing, unexpected = pol.load_state_dict(rec["sd"], strict=True), None
    sd_equal = all(torch.equal(pol.state_dict()[k], v) for k, v in rec["sd"].items())
    res = {"src": EB.__file__, "sd_loaded_equal": sd_equal}
    for name, (x, mm) in rec["inputs"].items():
        out = outputs(pol, x, mm)
        row = {}
        for k, a in out.items():
            b = rec["outputs"][name][k]
            a64, b64 = a.double(), b.double()
            d = (a64 - b64).abs()
            scale = b64.abs().max().item()
            row[k] = {"max_abs": d.max().item(), "max_rel_to_maxabs": d.max().item() / max(scale, 1e-30),
                      "bit_equal": bool(torch.equal(a, b)), "n_diff": int((a != b).sum())}
        la, lb = out["logits"], rec["outputs"][name]["logits"]
        ma = torch.where(mm, la, torch.full_like(la, -1e9)).argmax(-1)
        mb = torch.where(mm, lb, torch.full_like(lb, -1e9)).argmax(-1)
        row["greedy_agree"] = f"{int((ma == mb).sum())}/{len(ma)}"
        res[name] = row
    print(json.dumps(res, indent=1))
