"""Lift the ONLINE win head out of a snapshot, and PROVE it is the function under test.

The whole measurement rests on one identity:  ``V = sigmoid(win_head(value_pooled))``.
If that does not hold to float precision then "refit only the head on frozen features" is
not the same experiment as "replace the online head", and every number downstream is about
a different object. So this script does two things and refuses if the second fails:

  1. writes ``head.npz`` — the four parameter tensors of ``features_extractor.win_head.net``
     (LayerNorm(128) -> Linear(128,128) -> ReLU -> Linear(128,1)), for the FINE-TUNE arm's
     initialisation;
  2. re-applies that head to the ALREADY-EXTRACTED ``pooled.npy`` and checks the result
     against ``meta['V_fwd']`` (the frozen forward's own V, which the probe read already
     checked against the RECORDED win prob to 8.9e-07 on the exact cycle). A mismatch above
     ``--tol`` is a REFUSAL, not a warning.

Read-only over models/. CPU only.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

HEAD_KEYS = ("net.0.weight", "net.0.bias", "net.1.weight", "net.1.bias",
             "net.3.weight", "net.3.bias")


def head_state_dict(snapshot):
    import torch
    from main.prober.model import ProbeModel
    m = ProbeModel.load(snapshot, device="cpu")
    fe = m._policy.features_extractor
    if getattr(fe, "win_head", None) is None:
        raise SystemExit("REFUSED: this checkpoint has no win_head — `V` is not the win-prob head "
                         "here and the refit would be replacing a different function.")
    sd = {k: v.detach().numpy().astype(np.float64)
          for k, v in fe.win_head.state_dict().items()}
    missing = [k for k in HEAD_KEYS if k not in sd]
    if missing:
        raise SystemExit(f"REFUSED: win_head is not the expected LayerNorm->Linear->ReLU->Linear "
                         f"stack (missing {missing}, has {sorted(sd)}). The refit arm's "
                         f"architecture claim would be false.")
    with torch.no_grad():
        eps = float(fe.win_head.net[0].eps)
    return sd, eps


def apply_head(pooled, sd, eps):
    """The head in numpy — so the refit's forward and the checkpoint's are provably the same map."""
    x = np.asarray(pooled, np.float64)
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    z = (x - mu) / np.sqrt(var + eps) * sd["net.0.weight"] + sd["net.0.bias"]
    h = z @ sd["net.1.weight"].T + sd["net.1.bias"]
    h = np.maximum(h, 0.0)
    return (h @ sd["net.3.weight"].T + sd["net.3.bias"]).reshape(-1)


def main(a):
    sd, eps = head_state_dict(a.snapshot)
    pooled = np.load(f"{a.dir}/pooled.npy")
    meta = np.load(f"{a.dir}/meta.npy")
    logit = apply_head(pooled, sd, eps)
    v = 1.0 / (1.0 + np.exp(-logit))
    err = float(np.max(np.abs(v - meta["V_fwd"])))
    out = {"snapshot": a.snapshot, "dir": a.dir, "eps": eps,
           "max_abs_head_of_pooled_minus_Vfwd": err,
           "n": int(len(meta)),
           "param_shapes": {k: list(np.shape(sd[k])) for k in HEAD_KEYS}}
    if err > a.tol:
        raise SystemExit(f"REFUSED: sigmoid(win_head(value_pooled)) does not reproduce V_fwd "
                         f"(max |Δ| = {err:.3e} > {a.tol:.0e}). The identity this measurement "
                         f"rests on does not hold for this checkpoint.")
    np.savez(a.out, eps=np.array(eps), **{k: sd[k] for k in HEAD_KEYS})
    with open(a.out.replace(".npz", "_qc.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--dir", required=True, help="the probe read's extract dir (pooled.npy, meta.npy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-5)
    main(ap.parse_args())
