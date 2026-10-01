"""K9(b) tail sweep, the OFFLINE half (CPU): each banked violator (`sweep.py`'s violators.jsonl + .npz)
re-evaluated in fp64 on the same checkpoint, with every discrete op's MARGIN recorded (topk: the k-th
minus the (k+1)-th value; argmax / max: top-1 minus top-2; a comparison: |a - b|; a float->int cast: the
distance to the nearest integer). Which side is right (fp64 sits nearer the stored mu or the learner's
pi?), and how close to its tie is each op the GPU jitter showed flipping?

    python analyze.py --ckpt <ckpt.zip> --sweep <sweep out dir>      -> <sweep out dir>/analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch as th
from torch.overrides import TorchFunctionMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep import CAST, CMP, _site  # noqa: E402


class MarginTrace(TorchFunctionMode):
    def __init__(self) -> None:
        super().__init__()
        self.log: List[Dict[str, Any]] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = getattr(func, "__name__", "")
        m = None
        try:
            x = args[0] if args else None
            if name == "topk" and isinstance(x, th.Tensor):
                k = int(args[1] if len(args) > 1 else kwargs["k"])
                dim = int(args[2] if len(args) > 2 else kwargs.get("dim", -1))
                s = th.sort(x, dim=dim, descending=True).values
                if s.shape[dim] > k:
                    m = (s.narrow(dim, k - 1, 1) - s.narrow(dim, k, 1)).abs()
            elif name in ("argmax", "argmin") or (name in ("max", "min") and isinstance(out, tuple)):
                dim = args[1] if len(args) > 1 else kwargs.get("dim", None)
                if isinstance(x, th.Tensor) and x.is_floating_point() and dim is not None:
                    s = th.sort(x, dim=int(dim), descending=name in ("argmax", "max")).values
                    if s.shape[int(dim)] > 1:
                        m = (s.narrow(int(dim), 0, 1) - s.narrow(int(dim), 1, 1)).abs()
            elif name in CMP and isinstance(out, th.Tensor) and out.dtype == th.bool and len(args) >= 2:
                a, b = args[0], args[1]
                if any(isinstance(v, th.Tensor) and v.is_floating_point() for v in (a, b)):
                    m = (th.as_tensor(a, dtype=th.float64) - th.as_tensor(b, dtype=th.float64)).abs()
            elif (name in CAST and isinstance(out, th.Tensor) and not out.is_floating_point()
                  and out.dtype != th.bool and isinstance(x, th.Tensor) and x.is_floating_point()):
                m = (x.double() - x.double().round()).abs()
        except Exception:                     # a margin we cannot form is reported as None
            m = None
        if m is not None or name in ("topk", "sort", "argmax", "argmin", "argsort"):
            self.log.append({"op": name, "site": _site(),
                             "min_margin": None if m is None or m.numel() == 0 else float(m.min())})
        return out


def fp64_policy(ckpt: str) -> Any:
    from main.rust_core_m5.production import load_trainee

    pol = load_trainee(ckpt, "cpu").policy.double()
    pol.set_training_mode(True)
    return pol


def forward(pol: Any, obs: Dict[str, np.ndarray], a: int, mask: np.ndarray, f64: bool) -> Any:
    import stable_baselines3.common.policies as _sbp
    from stable_baselines3.common.utils import obs_as_tensor

    from agents.training.rust_rollout.consistency import _stashed_logp

    rp, rf = _sbp.preprocess_obs, th.Tensor.float
    if f64:
        _sbp.preprocess_obs = lambda o, s, normalize_images=True: {
            k: (v.double() if v.is_floating_point() else v) for k, v in rp(o, s, normalize_images).items()}
        th.Tensor.float = lambda self, *x, **k: self.to(th.get_default_dtype())  # type: ignore[method-assign]
        th.set_default_dtype(th.float64)
    mode = MarginTrace()
    try:
        with th.no_grad(), mode:
            pol.evaluate_actions(obs_as_tensor({k: v[None] for k, v in obs.items()}, th.device("cpu")),
                                 th.as_tensor([a]).long(), action_masks=th.as_tensor(mask[None]))
    finally:
        th.Tensor.float = rf  # type: ignore[method-assign]
        _sbp.preprocess_obs = rp
        th.set_default_dtype(th.float32)
    return _stashed_logp(pol)[0], mode.log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sweep", required=True)
    a = ap.parse_args()
    d = Path(a.sweep)
    rows = [json.loads(x) for x in open(d / "violators.jsonl")]
    z = np.load(d / "violators.npz")
    pol = fp64_policy(a.ckpt)
    out = []
    for i, r in enumerate(rows):
        obs = {k: z[k][i] for k in z.files}
        mask = np.array([c == "1" for c in r["mask"]])
        act = int(r["action"])
        lp64, log64 = forward(pol, obs, act, mask, True)
        sites = set(r.get("jitter", {}).get("changed_sites", {}))
        flips = [e for e in log64 if f"{e['op']}@{e['site']}" in sites]
        mu, pi = r["logp_stored"], r["logp_recomputed"]
        out.append({"i": i, "fill": r["fill"], "abs_dlogp": r["abs_dlogp"], "action": act, "mask": r["mask"],
                    "p_action": r.get("p_action_recomputed"), "mu_stored": mu, "pi_learner": pi,
                    "t2_alone": r.get("t2_alone"), "t2_in_48": r.get("t2_in_48"), "eager_b1": r.get("eager_b1"),
                    "fp64": float(lp64[act]), "fp64_nearer": "stored mu" if abs(lp64[act] - mu) < abs(lp64[act] - pi)
                    else "learner pi", "jitter_spread": r.get("jitter", {}).get("spread"),
                    "jitter_values": sorted(set(round(v, 5) for v in r.get("jitter", {}).get("values", []))),
                    "changed_sites": r.get("jitter", {}).get("changed_sites"),
                    "fp64_margins_at_changed_sites": flips,
                    "fp64_exact_ties": sum(1 for e in log64 if e["min_margin"] == 0.0),
                    "fp64_smallest_margins": sorted([e for e in log64 if e["min_margin"]],
                                                    key=lambda e: e["min_margin"])[:5]})
        print(json.dumps(out[-1])[:600], flush=True)
    (d / "analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
