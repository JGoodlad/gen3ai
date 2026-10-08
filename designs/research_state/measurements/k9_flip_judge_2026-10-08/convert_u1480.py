"""The HEAD reproduction's weights: the u1480 dump (pin 6c6d2e09, config v143) mapped onto HEAD's static x
fixed_mass policy (v145). It is a CONSTRUCTION, not the dumped policy: HEAD's version break ties the op's
`out_gain` (one gain per (region, channel) and mon, `gen3_mon_tied_gain_v1`) where the pin had one per flat
position, so each tied gain is the MEAN of the dumped positions it now covers; the keys HEAD no longer builds
(the dead SB3 value tower, the ride-along heads, a deleted bias) are dropped. Every other tensor is copied
bit for bit. The dump is only READ; the output goes where ``--out`` says (never under models/).

    python convert_u1480.py --dump models/rb_st_static_s1001/behaviour_violation_u1480_policy.pt --out <pt>
"""
import argparse
import json

import gymnasium as gym
import numpy as np
import torch as th

from agents.model.policy import Gen3DualHeadMaskablePolicy
from main.fresh_checkpoint import _production_policy_kwargs
from main.train.production_args import production_args


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if "/models/" in a.out or a.out.startswith("models/"):
        raise SystemExit("refusing to write under models/")
    args = production_args()
    args.belief_tokens, args.token_encoding = "fixed_mass", "static"
    _a, layout, pk = _production_policy_kwargs(args)
    dim = layout["total_dim"]
    space = gym.spaces.Dict({"observation": gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32),
                             "action_mask": gym.spaces.MultiBinary(11)})
    pol = Gen3DualHeadMaskablePolicy(space, gym.spaces.Discrete(11), lambda _: 3e-4, **pk)
    own = pol.state_dict()
    sd = th.load(a.dump, map_location="cpu", weights_only=True)
    out, report = {}, {"copied": 0, "dropped": sorted(set(sd) - set(own)), "mapped": {}}
    missing = sorted(set(own) - set(sd))
    if missing:
        raise SystemExit(f"HEAD needs keys the dump lacks: {missing[:8]}")
    for k, v in own.items():
        src = sd[k]
        if tuple(src.shape) == tuple(v.shape):
            out[k] = src.clone()
            report["copied"] += 1
            continue
        if not k.endswith("damage_op.out_gain"):
            raise SystemExit(f"unexpected shape change at {k}: {tuple(src.shape)} -> {tuple(v.shape)}")
        mod = pol.get_submodule(k[: -len(".out_gain")])
        tie = mod._out_gain_tie.double()                       # [n_gain, out_dim] one-hot
        if tie.shape[1] != src.shape[0]:
            raise SystemExit(f"{k}: the tie covers {tie.shape[1]} positions, the dump has {src.shape[0]}")
        g = (tie @ src.double()) / tie.sum(1)
        spread = max(float((src.double()[tie[j] > 0] - g[j]).abs().max()) for j in range(tie.shape[0]))
        out[k] = g.to(v.dtype)
        report["mapped"][k] = {"from": int(src.shape[0]), "to": int(v.shape[0]), "max_within_tie_spread": spread}
    pol.load_state_dict(out, strict=True)
    th.save(out, a.out)
    print(json.dumps({"copied": report["copied"], "dropped": len(report["dropped"]),
                      "dropped_prefixes": sorted({k.split(".")[0] + "." + k.split(".")[1] for k in report["dropped"]}),
                      "mapped": report["mapped"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
