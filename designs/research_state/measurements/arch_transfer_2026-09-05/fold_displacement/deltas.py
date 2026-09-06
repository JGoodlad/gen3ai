"""P1 + P3 — the ACTUAL displacement a fold produced, per parameter group, and its direction.

Delta_g = theta_arm,g - theta_parent,g over the POLICY's named parameters, for the teacher-content
2x2 at three depths and the five reuse-batch arms at their end. Reported as |Delta_g|/|theta_g|
(relative) and |Delta_g|^2/|Delta|^2 (share). Buffers (PopArt and the constant data tables) are
reported SEPARATELY and never inside a group.

P3 is the per-group cosine between two arms' displacements; the two REPLICATE pairs
(TCFUNDA.TCFUNDB, TCUNFA.TCUNFB) are the floor a funded-vs-unfunded difference must clear.

The parameter GROUPING is IMPORTED from ../sharing_kernel/kernel.py (GROUP_RULES / group_of /
GROUPS), never copied, so the two probes cannot drift apart.

This script also caches, for the projection stage:
  deltas.npy   [P, K] float32 -- the displacement columns, in the group-ordered layout
  logits.npy   [K+1, N, 11] float32 -- masked logits of parent (row 0) and every arm, on the
               456 frozen states; illegal entries zeroed (masked_kl_rows re-masks them anyway).

Run:
  PYTHONPATH=<worktree>/src nice -n 10 python deltas.py --scratch <dir> --out displacement.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import torch as th  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "sharing_kernel"))
from kernel import GROUPS, group_of  # noqa: E402  -- IMPORTED, not copied

from agents.model.snapshot import current_model_version, load_foreign_opponent  # noqa: E402
from agents.observation.state_encoder import load_mappings  # noqa: E402

MD = "/home/goodlad/dev/gen3ai/models"
PARENT = f"{MD}/ai_v9_59_R2ACTION_0827/final_model.zip"
# The predecessor's config path for the parent, reused verbatim so the two probes build the same
# module tree from the same declaration.
PARENT_CFG = f"{MD}/ai_v9_29_rev1_0823/snapshots/model_config.json"

_2x2 = [("TCFUNDA", "ai_v9_160_TCFUNDA_0903", "funded A"),
        ("TCFUNDB", "ai_v9_161_TCFUNDB_0903", "funded B"),
        ("TCUNFA", "ai_v9_162_TCUNFA_0903", "unfunded A"),
        ("TCUNFB", "ai_v9_163_TCUNFB_0903", "unfunded B")]
# The two interior depths are the checkpoints COMMON to all four 2x2 arms.
DEPTHS = [("p1M", "checkpoints/checkpoint_29115216_steps.zip"),
          ("mid", "checkpoints/checkpoint_30115248_steps.zip"),
          ("end", "final_model.zip")]
# The reuse batch, END only -- run dirs read from offline_collateral_kl.py::ARMS.
REUSE = [("R4DOSE12", "ai_v9_150_R4DOSE12_0901", "dose 0.53x v8"),
         ("R4DOSE6", "ai_v9_151_R4DOSE6_0901", "dose 1.06x v8"),
         ("R4DOSE3", "ai_v9_152_R4DOSE3_0901", "dose 2.12x v8"),
         ("B2", "ai_v9_140_B2_0901", "the fold, coef 0.1761"),
         ("C1", "ai_v9_141_C1_0901", "the LOSS-OFF control, coef 0")]


def arm_specs():
    out = []
    for tag, run, desc in _2x2:
        for dep, rel in DEPTHS:
            out.append(dict(key=f"{tag}@{dep}", tag=tag, depth=dep, run=run, desc=desc,
                            path=f"{MD}/{run}/{rel}", cfg=f"{MD}/{run}/model_config.json",
                            batch="2x2"))
    for tag, run, desc in REUSE:
        out.append(dict(key=f"{tag}@end", tag=tag, depth="end", run=run, desc=desc,
                        path=f"{MD}/{run}/final_model.zip",
                        cfg=f"{MD}/{run}/model_config.json", batch="reuse"))
    return out


def strip_debugger(m):
    obj = getattr(m, "policy", m)
    for mod in obj.modules():
        if getattr(mod, "_debugger", None) is not None:
            mod._debugger = None
    fe = getattr(obj, "features_extractor", None)
    if fe is not None and hasattr(fe, "_debugger"):
        fe._debugger = None
    return m


def build_layout(policy):
    """Group-ordered flat layout: a group is a CONTIGUOUS slice, so no statistic ever gathers."""
    by_group = {g: [] for g in GROUPS}
    for nm, p in policy.named_parameters():
        by_group[group_of(nm)].append((nm, p))
    layout, order, cur = {}, [], 0
    for g in GROUPS:
        start = cur
        for nm, p in by_group[g]:
            order.append((nm, cur, cur + p.numel(), tuple(p.shape)))
            cur += p.numel()
        layout[g] = (start, cur)
    return layout, order, cur


def flatten(policy, order, P):
    sd = dict(policy.named_parameters())
    v = np.zeros(P, dtype=np.float32)
    for nm, s, e, shp in order:
        t = sd[nm]
        if tuple(t.shape) != shp:
            raise SystemExit(f"[d] SHAPE MISMATCH on {nm}: {tuple(t.shape)} vs parent {shp}")
        v[s:e] = t.detach().reshape(-1).numpy()
    return v


def masked_logits(policy, obs_all, mask_all, chunk=64):
    n = obs_all.shape[0]
    out = np.zeros((n, mask_all.shape[1]), dtype=np.float32)
    space = policy.observation_space.spaces
    with th.no_grad():
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            ob = {"observation": th.as_tensor(obs_all[i:j]),
                  "action_mask": th.as_tensor(mask_all[i:j])}
            ob = {k: v for k, v in ob.items() if k in space}
            dist = policy.get_distribution(ob, action_masks=(mask_all[i:j] > 0.5))
            lg = dist.distribution.logits.detach().numpy()
            out[i:j] = np.where(mask_all[i:j] > 0.5, lg, 0.0)   # illegal -> 0; re-masked downstream
    if not np.isfinite(out).all():
        raise SystemExit("[d] non-finite logit on a LEGAL action")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=os.path.join(HERE, os.pardir, "sharing_kernel",
                                                     "states_gen.npz"))
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "displacement.json"))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    th.set_num_threads(a.threads)
    os.makedirs(a.scratch, exist_ok=True)
    t0 = time.time()

    d = np.load(a.states, allow_pickle=False)
    obs_all = d["observation"].astype(np.float32)
    mask_all = d["action_mask"].astype(np.float32)
    team_all = [str(x) for x in d["team"]]
    grp_all = [str(x) for x in d["group"]]
    N = obs_all.shape[0]
    print(f"[d] {N} states, {len(set(team_all))} teams "
          f"({sum(g == 'taught' for g in grp_all)} taught / "
          f"{sum(g == 'untaught' for g in grp_all)} untaught rows)", flush=True)

    cv = current_model_version(load_mappings())
    parent, _ = load_foreign_opponent(PARENT, current_version=cv, device="cpu",
                                      config_path=PARENT_CFG)
    strip_debugger(parent)
    parent.policy.set_training_mode(False)
    layout, order, P = build_layout(parent.policy)
    theta_p = flatten(parent.policy, order, P)
    pbuf = {k: v.detach().numpy().copy() for k, v in parent.policy.named_buffers()}
    print(f"[d] parent step {parent.num_timesteps:,}  P={P:,} in "
          + ", ".join(f"{g}={layout[g][1]-layout[g][0]:,}" for g in GROUPS), flush=True)

    specs = arm_specs()
    K = len(specs)
    D = np.zeros((P, K), dtype=np.float32)
    L = np.zeros((K + 1, N, mask_all.shape[1]), dtype=np.float32)
    L[0] = masked_logits(parent.policy, obs_all, mask_all)
    print(f"[d] parent logits done ({time.time()-t0:.0f}s)", flush=True)
    del parent

    theta_grp_norm = {g: float(np.linalg.norm(theta_p[layout[g][0]:layout[g][1]].astype(np.float64)))
                      for g in GROUPS}
    theta_norm = float(np.linalg.norm(theta_p.astype(np.float64)))

    per_arm = {}
    for k, sp in enumerate(specs):
        m, _ = load_foreign_opponent(sp["path"], current_version=cv, device="cpu",
                                     config_path=sp["cfg"])
        strip_debugger(m)
        m.policy.set_training_mode(False)
        got = set(dict(m.policy.named_parameters()))
        want = set(nm for nm, _, _, _ in order)
        if got != want:
            raise SystemExit(f"[d] KEY MISMATCH on {sp['key']}: "
                             f"+{sorted(got-want)[:5]} -{sorted(want-got)[:5]}")
        D[:, k] = flatten(m.policy, order, P) - theta_p
        L[k + 1] = masked_logits(m.policy, obs_all, mask_all)
        abuf = {kk: v.detach().numpy() for kk, v in m.policy.named_buffers()}
        bufs = {}
        for bn in sorted(set(pbuf) & set(abuf)):
            dv = np.abs(abuf[bn].astype(np.float64) - pbuf[bn].astype(np.float64)).max()
            if dv > 0:
                bufs[bn] = dict(max_abs_change=float(dv),
                                parent=np.asarray(pbuf[bn]).ravel()[:4].tolist(),
                                arm=np.asarray(abuf[bn]).ravel()[:4].tolist())
        col = D[:, k].astype(np.float64)
        tot_sq = float(col @ col)
        g_stats = {}
        for g in GROUPS:
            s, e = layout[g]
            cg = col[s:e]
            sq = float(cg @ cg)
            g_stats[g] = dict(n_params=e - s, l2=float(np.sqrt(sq)),
                              rel_l2=float(np.sqrt(sq) / theta_grp_norm[g]),
                              sq_share=sq / tot_sq)
        per_arm[sp["key"]] = dict(
            tag=sp["tag"], depth=sp["depth"], run=sp["run"], desc=sp["desc"], batch=sp["batch"],
            path=sp["path"], steps=int(m.num_timesteps),
            steps_since_parent=int(m.num_timesteps) - 28_115_184,
            l2=float(np.sqrt(tot_sq)), rel_l2=float(np.sqrt(tot_sq) / theta_norm),
            groups=g_stats, buffers_changed=bufs)
        print(f"[d] {sp['key']:16s} step {m.num_timesteps:,}  |d|={np.sqrt(tot_sq):.4f}  "
              f"rel={np.sqrt(tot_sq)/theta_norm:.5f}  "
              + " ".join(f"{g[:5]}={g_stats[g]['sq_share']*100:.1f}%" for g in GROUPS)
              + f"  ({time.time()-t0:.0f}s)", flush=True)
        del m

    # --- P3: per-group cosine matrices over the K displacement columns -------------------------
    cos = {}
    for g in ["ALL"] + GROUPS:
        s, e = (0, P) if g == "ALL" else layout[g]
        B = D[s:e, :].astype(np.float32)
        Gm = (B.T @ B).astype(np.float64)
        nn = np.sqrt(np.diag(Gm))
        C = Gm / np.outer(nn, nn)
        cos[g] = C.tolist()

    np.save(os.path.join(a.scratch, "deltas.npy"), D)
    np.save(os.path.join(a.scratch, "logits.npy"), L)
    out = dict(_meta=dict(
        parent=PARENT, parent_cfg=PARENT_CFG, parent_steps=28_115_184, n_params=P,
        group_layout={g: list(layout[g]) for g in GROUPS},
        parent_group_l2={g: theta_grp_norm[g] for g in GROUPS}, parent_l2=theta_norm,
        arm_keys=[s["key"] for s in specs], states=os.path.abspath(a.states), n_states=N,
        grouping_source="../sharing_kernel/kernel.py (imported)",
        threads=a.threads, wall_s=round(time.time() - t0, 1)),
        arms=per_arm, cosine=cos)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[d] wrote {a.out} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
