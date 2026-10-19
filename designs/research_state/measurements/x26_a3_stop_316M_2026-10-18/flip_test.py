"""Would a FLIP of the violator's nearest selection element reproduce the observed mismatch?

For each event's violator, under each weight set (the neighbour checkpoints, the interpolation, and for the
control dumps their TRUE weights), the learner forward runs under `tie_margins.TieMargins` with ``near_eps``
just above the row's smallest declared margin, so exactly that ONE element (its nearest sort pair / threshold)
is "near". `TieFlip` then re-runs the forward with that element resolved the other way -- the probe's own
flip-judge machinery (`consistency.flip_judge`), pointed at the nearest element instead of only those under
2e-4. The flip's effect on the row's full masked log-probs (``alt - as_is``) is compared with the observed
mismatch ``delta = log mu (behaviour) - log pi (learner)`` over the legal actions, from the K9(b) record:

* size: max |alt - as_is| against max |delta|;
* direction: the cosine of the two centred vectors (log-prob vectors are defined up to the normaliser, so
  each is centred by its probability-weighted mean before the cosine).

A flip that reproduces delta in size and direction would mean the behaviour forward took the other
resolution of that element. A flip far larger than delta, or orthogonal to it, says the observed mismatch is
not that element's flip. CPU only, pinned code:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pin_9c033489>/src CUDA_VISIBLE_DEVICES= \
        python3 flip_test.py --in <copies> --ctrl <control copies> --out flip_test.json
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch as th

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("x26_a3_replay_neighbours", os.path.join(_HERE, "replay_neighbours.py"))
RN = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(RN)  # type: ignore[union-attr]
RD = RN.RD


def _vec(xs: Any) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in xs], dtype=np.float64)


def _centred(v: np.ndarray, lp: np.ndarray, ok: np.ndarray) -> np.ndarray:
    p = np.exp(lp[ok])
    p = p / p.sum()
    return v[ok] - float((p * v[ok]).sum())


def forward(policy: Any, obs: Dict[str, np.ndarray], act: np.ndarray, mask: np.ndarray, mode: Any) -> np.ndarray:
    from agents.training.rust_rollout.consistency import _stashed_logp
    o = {k: th.as_tensor(v) for k, v in obs.items()}
    policy.set_training_mode(True)
    with th.enable_grad(), mode:
        policy.evaluate_actions(o, th.as_tensor(act).long(), action_masks=th.as_tensor(mask))
    policy.set_training_mode(False)
    out = _stashed_logp(policy)
    assert out is not None
    return out[0]


def flip_one(template: Any, sd: Dict[str, th.Tensor], obs: Dict[str, np.ndarray], act: np.ndarray,
             mask: np.ndarray, delta: np.ndarray) -> Dict[str, Any]:
    from agents.training.rust_rollout.tie_margins import TieFlip, TieMargins
    p = copy.deepcopy(template)
    p.load_state_dict(sd)
    rec0 = TieMargins(1, keep_calls=True)
    base = forward(p, obs, act, mask, rec0)
    m = float(rec0.margin[0])
    res: Dict[str, Any] = {"min_margin": m, "site": rec0.site[0], "undeclared": dict(rec0.undeclared)}
    if not np.isfinite(m):
        res["flippable"] = False
        return res
    rec = TieMargins(1, near_eps=m * 1.0001 if m > 0 else 1e-12)
    as_is = forward(p, obs, act, mask, rec)
    res["as_is_equals_plain"] = bool(np.array_equal(as_is, base))
    res["flippable"] = bool(rec.flippable()[0])
    if not res["flippable"]:
        res["near_calls"], res["near_elems"] = int(rec.near_calls[0]), int(rec.near_elems[0])
        return res
    st = rec._steps[0]
    res["flip_step"] = {"call": st.call, "site": st.site, "kind": st.kind, "pos": list(st.pos), "witness": list(st.witness)}
    mode = TieFlip(1, rec.flip_plan([0]), rec.call_sites)
    alt = forward(p, obs, act, mask, mode)
    mode.check_applied()
    ok = mask[0].astype(bool) & np.isfinite(delta)
    fv = alt - as_is
    a, b = _centred(fv, as_is, ok), _centred(delta, as_is, ok)
    res.update({"flip_maxabs": float(np.abs(fv[ok]).max()), "delta_maxabs": float(np.abs(delta[ok]).max()),
                "cosine_flip_vs_delta": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300)),
                "flip_vec_legal": fv[ok].tolist(), "delta_legal": delta[ok].tolist()})
    return res


def run_dump(template: Any, weights: Dict[str, Dict[str, th.Tensor]], z: Any, rec: Dict[str, Any],
             delta: Optional[np.ndarray]) -> Dict[str, Any]:
    d = RD.rows_of(z)
    obs = {k: v[:1] for k, v in d["obs"].items()}
    act = d["acts"][:1]
    mask = obs["action_mask"].astype(bool)
    if delta is None:
        r0 = rec["rows"][0]
        delta = _vec(r0["dist_stored"]) - _vec(r0["dist_recomputed"])
    return {name: flip_one(template, sd, obs, act, mask, delta) for name, sd in weights.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    th.set_float32_matmul_precision("highest")
    th.set_num_threads(int(os.environ.get("K9B_THREADS", "3")))
    th.manual_seed(0)
    from main.capacity import load_policy
    t0 = time.time()
    model, _cfg, _ = load_policy(os.path.join(a.inp, "checkpoint_316000231_steps.zip"), a.inp, "cpu")
    template = copy.deepcopy(model.policy)
    del model
    out: Dict[str, Any] = {"events": {}, "controls": {}}
    recs = {json.loads(line)["n_updates"]: json.loads(line) for line in open(os.path.join(a.inp, "behaviour_violations.jsonl"))}
    sds = {k: RN.policy_sd(os.path.join(a.inp, f"checkpoint_{k}_steps.zip")) for k in RN.CKPTS}
    for u, (lo, hi) in RN.BRACKET.items():
        rec = recs[u]
        alpha = (rec["num_timesteps"] - RN.CKPTS[lo]) / (RN.CKPTS[hi] - RN.CKPTS[lo])
        w = {f"ckpt_{lo}": sds[lo], f"ckpt_{hi}": sds[hi], f"interp_{alpha:.3f}": RN.interp(sds[lo], sds[hi], alpha)}
        out["events"][str(u)] = run_dump(template, w, np.load(os.path.join(a.inp, f"behaviour_violation_u{u}_rows.npz")), rec, None)
        for name, r in out["events"][str(u)].items():
            print(f"[event u{u} {name}] {time.time() - t0:.0f}s " + json.dumps({k: v for k, v in r.items() if not k.endswith("_legal")}), flush=True)
    crecs = {json.loads(line)["n_updates"]: json.loads(line) for line in open(os.path.join(a.ctrl, "behaviour_violations.jsonl"))}
    for u in (19300, 20490, 20630):
        rec = crecs[u]
        sd = th.load(os.path.join(a.ctrl, f"behaviour_violation_u{u}_policy.pt"), map_location="cpu")
        z = np.load(os.path.join(a.ctrl, f"behaviour_violation_u{u}_rows.npz"))
        # the control's delta at its TRUE weights: stored mu - the GPU learner (the dump's two values), on the
        # taken action only (the K9(b) record of these early dumps holds full distributions too)
        out["controls"][str(u)] = run_dump(template, {"true": sd}, z, rec, None)
        for name, r in out["controls"][str(u)].items():
            print(f"[control u{u} {name}] {time.time() - t0:.0f}s " + json.dumps({k: v for k, v in r.items() if not k.endswith("_legal")}), flush=True)
    out["seconds"] = time.time() - t0
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
