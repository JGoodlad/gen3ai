"""Capture the PRE-BREAK reference for the version break's weight-mapping identity proof.

Run ONCE at the base commit (26131c0c, before any break edit), CPU, one thread:

    python designs/research_state/measurements/version_break_identity_2026-10-07/capture_reference.py OUT_DIR

For the K9 learner golden's ``fixed_mass`` arm (the adopted production belief representation) it writes:

* ``init_state.pt``   — the seeded, name-keyed-perturbed learner's full ``policy.state_dict()``
                        (what the K9 update starts from);
* ``forward.pt``      — ``evaluate_actions_functional`` on the arm's committed 64-row buffer at that state
                        (values, log_prob, entropy, masked log pi) plus every tensor the extractor stashes
                        as a ``last_*`` attribute after the forward (every head's output);
* ``post_state.pt``   — the ``policy.state_dict()`` after the K9 update (one ``train()``, the golden's own
                        seeding), and ``losses.json`` with the update's pinned loss scalars;
* ``meta.json``       — the commit, torch version, the buffer sha256, and the sha256 of every file above.

The weights are NOT committed (12 MB of random floats); ``meta.json`` and the comparison script's result are.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _stash(fe) -> dict:
    import torch as th

    out = {}
    for k, v in sorted(vars(fe).items()):
        if k.startswith("last_") and isinstance(v, th.Tensor):
            out[k] = v.detach().clone()
        elif k.startswith("last_") and isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, th.Tensor):
                    out[f"{k}.{kk}"] = vv.detach().clone()
    return out


def main(out_dir: str) -> int:
    import numpy as np
    import torch as th
    from stable_baselines3.common.utils import obs_as_tensor

    from agents.training import learner_golden as LG

    arm = "fixed_mass"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with LG._one_thread():
        model = LG.build_arm_learner(arm)
        data = LG.load_buffer_into(model, LG.arm_buffer(arm))
        th.save(model.policy.state_dict(), out / "init_state.pt")
        rb = model.rollout_buffer
        obs = {k: th.as_tensor(v.reshape(-1, *v.shape[2:])) for k, v in rb.observations.items()}
        acts = th.as_tensor(rb.actions.reshape(-1, *rb.actions.shape[2:])).long().flatten()
        masks = th.as_tensor(rb.action_masks.reshape(-1, rb.action_masks.shape[-1]))
        model.policy.set_training_mode(True)
        with th.no_grad():
            values, log_prob, entropy, logp, masks_bool = model.policy.evaluate_actions_functional(
                obs, acts, masks)
        fwd = {"values": values, "log_prob": log_prob, "entropy": entropy, "masked_logp": logp,
               "masks_bool": masks_bool}
        fwd.update({f"fe.{k}": v for k, v in _stash(model.policy.features_extractor).items()})
        th.save(fwd, out / "forward.pt")
        res = LG.compute(model=model, buffer=LG.arm_buffer(arm))
        th.save(model.policy.state_dict(), out / "post_state.pt")
    (out / "losses.json").write_text(json.dumps(res["losses"], indent=1, sort_keys=True))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"commit": commit, "torch": th.__version__, "arm": arm,
            "buffer": str(LG.arm_buffer(arm).name), "buffer_sha256": LG.buffer_sha256(LG.arm_buffer(arm)),
            "init_params_sha256": res["init_params_sha256"], "post_params_sha256": res["post_params_sha256"],
            "group_sha256": res["group_sha256"], "init_group_sha256": res["init_group_sha256"],
            "forward_keys": sorted(fwd), "files": {}}
    for f in ("init_state.pt", "forward.pt", "post_state.pt", "losses.json"):
        meta["files"][f] = _sha(out / f)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    print(json.dumps({k: meta[k] for k in ("commit", "torch", "init_params_sha256", "post_params_sha256")},
                     indent=1))
    print("forward keys:", len(fwd), "  rows:", int(values.shape[0]))
    _ = np
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
