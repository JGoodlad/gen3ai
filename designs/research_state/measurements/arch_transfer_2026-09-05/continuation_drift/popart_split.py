"""FOLLOW-UP (not pre-registered): how much of the CRITIC group's displacement is PopArt BOOKKEEPING?

`PopArtNormalizer.update` rescales EXACTLY ONE module — `policy.value_net` — by
`W *= sigma_old/sigma_new`, `b = (sigma_old*b + mu_old - mu_new)/sigma_new`, to preserve the
de-normalized output across a statistics change. That is a deterministic transform, not learning,
and it is the one mechanism that can move critic weights with no gradient behind it. The P1 critic
row (v8 b=0.167 vs gen b=0.418) is therefore exactly the row this can fake, so it is measured
rather than argued about: the critic group is split into `value_net.*` and everything else, and the
exponent refitted on the remainder.

Run — GEN:  PYTHONPATH=<worktree>/src nice -n 10 python popart_split.py --era gen
Run — V8 :  cd /tmp/v8rep_era && PYTHONPATH=/tmp/v8rep_era/src PYTHONDONTWRITEBYTECODE=1 \
              nice -n 10 python <this> --era v8
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import torch as th  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sharing_kernel"))
from kernel import group_of  # noqa: E402

sys.path.insert(0, HERE)
from drift import ERAS, MD, resolve_depths, strip_debugger  # noqa: E402

from agents.model.snapshot import current_model_version, load_foreign_opponent  # noqa: E402
from agents.observation.state_encoder import load_mappings  # noqa: E402

POPART_PREFIX = "value_net."          # the ONLY module PopArtNormalizer.update touches


def ols_slope(t, y):
    lt, ly = np.log(np.asarray(t, float)), np.log(np.asarray(y, float))
    lt = lt - lt.mean()
    return float((lt @ (ly - ly.mean())) / (lt @ lt))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", required=True, choices=sorted(ERAS))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    th.set_num_threads(a.threads)
    E = ERAS[a.era]
    cv = current_model_version(load_mappings())

    parent, _ = load_foreign_opponent(E["parent"], current_version=cv, device="cpu",
                                      config_path=E["parent_cfg"])
    strip_debugger(parent)
    pp = {nm: p.detach().numpy().copy() for nm, p in parent.policy.named_parameters()
          if group_of(nm) == "critic"}
    pop = sorted(nm for nm in pp if nm.startswith(POPART_PREFIX))
    rest = sorted(nm for nm in pp if not nm.startswith(POPART_PREFIX))
    if not pop:
        raise SystemExit(f"[p] no parameter under '{POPART_PREFIX}' in the critic group")
    n_pop = sum(pp[nm].size for nm in pop)
    n_rest = sum(pp[nm].size for nm in rest)
    print(f"[p] era={a.era}  critic group: PopArt-rescaled {pop} = {n_pop:,} params; "
          f"remainder {n_rest:,} params", flush=True)
    del parent

    out = {"era": a.era, "popart_params": pop, "n_popart": n_pop, "n_rest": n_rest, "arms": {}}
    for arm, run in E["arms"]:
        for lab, path in resolve_depths(a.era, run):
            m, _ = load_foreign_opponent(path, current_version=cv, device="cpu",
                                         config_path=f"{MD}/{run}/model_config.json")
            strip_debugger(m)
            sd = dict(m.policy.named_parameters())
            sq_pop = sum(float(((sd[nm].detach().numpy() - pp[nm]).astype(np.float64) ** 2).sum())
                         for nm in pop)
            sq_rest = sum(float(((sd[nm].detach().numpy() - pp[nm]).astype(np.float64) ** 2).sum())
                          for nm in rest)
            out["arms"][f"{arm}@{lab}"] = dict(
                arm=arm, depth=lab, t=int(m.num_timesteps) - E["parent_steps"],
                fit_point=(lab != "d3ck"),
                l2_popart=float(np.sqrt(sq_pop)), l2_rest=float(np.sqrt(sq_rest)),
                popart_share_of_critic_sq=sq_pop / (sq_pop + sq_rest))
            r = out["arms"][f"{arm}@{lab}"]
            print(f"[p] {arm}@{lab:5s} t={r['t']:>9,}  |d|_popart={r['l2_popart']:.4f} "
                  f"|d|_rest={r['l2_rest']:.4f}  popart share of critic sq-disp="
                  f"{r['popart_share_of_critic_sq']*100:.2f}%", flush=True)
            del m

    deps = {"d1", "d3"}
    out["b_critic_excluding_popart"] = {}
    out["b_critic_popart_only"] = {}
    for arm, _ in E["arms"]:
        pts = sorted((r["t"], r) for k, r in out["arms"].items()
                     if r["arm"] == arm and r["fit_point"] and r["depth"] in deps)
        out["b_critic_excluding_popart"][arm] = ols_slope([t for t, _ in pts],
                                                          [r["l2_rest"] for _, r in pts])
        out["b_critic_popart_only"][arm] = ols_slope([t for t, _ in pts],
                                                     [r["l2_popart"] for _, r in pts])
    print(f"[p] b(critic, PopArt layer EXCLUDED) = {out['b_critic_excluding_popart']}")
    print(f"[p] b(PopArt layer only)             = {out['b_critic_popart_only']}")

    dst = os.path.join(HERE, f"popart_split_{a.era}.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[p] wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
