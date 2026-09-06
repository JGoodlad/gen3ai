"""H9 — IS A PLAIN CONTINUATION A RANDOM WALK OR A DIRECTED DRIFT?

Measures, for the two PLAIN-CONTINUATION cells that exist (v8's 277M parent and our 28M parent),
along each cell's three replicate arms:

  (i)   |theta_t - theta_0| per parameter GROUP and for ALL parameters, at every checkpoint depth
        (theta_0 = that cell's parent's final weights). The exponent b in |Delta theta| ~ t^b is
        fitted downstream in analyze.py.
  (ii)  the REPLICATE COSINE cos(Delta theta_A, Delta theta_B) at matched depth, per group.
  (iii) the OUTPUT-side twin: KL(parent || arm) over LEGAL actions on that era's FROZEN state
        batch, per row, aggregated all / taught / untaught and per team (for a cluster bootstrap).

The parameter GROUPING is IMPORTED from ../sharing_kernel/kernel.py (GROUP_RULES / group_of /
GROUPS), never copied, so this probe and its two predecessors cannot drift apart. That file's
rules already cover BOTH eras by ROLE (v8's `action_net` and the gen `pointer_head` both map to
`action_head`), which is what the brief asks for.

The forward KL is IMPORTED too: the gen era gets the real
`agents.training.instrumented_ppo.distill_anchor.masked_kl_rows`; the v8 era, whose pin predates
that module, gets `../content_locality/era_kl.masked_kl_rows_era` — the sanctioned era copy that
`../content_locality/kl_unit_test.py` pins against the real one.

Buffers (PopArt statistics, the constant data tables) are NEVER inside a group; every buffer whose
value differs from the parent's is reported separately.

Run — GEN half, from this worktree:
  PYTHONPATH=<worktree>/src nice -n 10 python drift.py --era gen --out drift_gen.json

Run — V8 half, from the era-pinned read-only checkout (obs 2992 / config_version 45; current code
REFUSES those checkpoints):
  cd /tmp/v8rep_era && PYTHONPATH=/tmp/v8rep_era/src PYTHONDONTWRITEBYTECODE=1 nice -n 10 \
    python <this> --era v8 --out drift_v8.json
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
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
UP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(UP, "sharing_kernel"))
sys.path.insert(0, os.path.join(UP, "content_locality"))
from kernel import GROUPS, group_of  # noqa: E402  -- IMPORTED, not copied

from agents.model.snapshot import current_model_version, load_foreign_opponent  # noqa: E402
from agents.observation.state_encoder import load_mappings  # noqa: E402

MD = "/home/goodlad/dev/gen3ai/models"

# --- THE TWO CELLS -----------------------------------------------------------------------------
# Every depth is a REAL file on disk; its `t` is verified at load time against the model's own
# num_timesteps (the v8-era metadata has no `num_timesteps` key, so the zip's own `data` blob is
# the only authority there) and the script DIES on a mismatch rather than mislabelling an axis.
ERAS = {
    # OUR parent: ai_v9_59_R2ACTION_0827 @ 28,115,184.  G5 = the plain-continuation control.
    "gen": dict(
        parent=f"{MD}/ai_v9_59_R2ACTION_0827/final_model.zip",
        parent_cfg=f"{MD}/ai_v9_29_rev1_0823/snapshots/model_config.json",
        parent_steps=28_115_184,
        states=os.path.join(UP, "sharing_kernel", "states_gen.npz"),
        arms=[("A", "ai_v9_195_G5PLAINA_0906"),
              ("B", "ai_v9_196_G5PLAINB_0906"),
              ("C", "ai_v9_197_G5PLAINC_0906")],
        depths=[("d1", "checkpoints/checkpoint_28615200_steps.zip"),
                ("d2", "checkpoints/checkpoint_29115216_steps.zip"),
                ("d3", "final_model.zip")],
    ),
    # V8's parent: ai_v8_04_distill_4teacher_0722 @ 277,583,267.  Cell 2 = v8rep_p2self_*.
    # Only TWO depths exist per arm: the self-play snapshot the arm promoted at ~+417k (identical
    # file to best_model/ and eval_traces/step_*/snapshot.zip) and the end. The checkpoint at
    # ~+1.08M and final_model_interrupted.zip are the SAME depth to within 30-5019 steps, so the
    # final is used and the checkpoint is measured as a same-depth agreement check, never as a
    # second point in the fit.
    "v8": dict(
        parent=f"{MD}/ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip",
        parent_cfg=f"{MD}/ai_v8_04_distill_4teacher_0722/model_config.json",
        parent_steps=277_583_267,
        states=os.path.join(UP, "sharing_kernel", "states_v8.npz"),
        arms=[("A", "v8rep_p2self_A_0905"),
              ("B", "v8rep_p2self_B_0905"),
              ("C", "v8rep_p2self_C_0905")],
        depths=[("d1", "snapshots/SNAPSHOT"),          # resolved per arm below
                ("d3", "final_model_interrupted.zip"),
                ("d3ck", "checkpoints/CKPT")],         # same-depth agreement check, not a fit point
    ),
}


def resolve_depths(era: str, run: str):
    """Return [(label, abs_path)] for one arm, expanding the two v8 wildcards."""
    out = []
    for lab, rel in ERAS[era]["depths"]:
        if rel.endswith("SNAPSHOT"):
            d = f"{MD}/{run}/snapshots"
            zs = sorted(f for f in os.listdir(d) if f.startswith("snapshot_") and f.endswith(".zip"))
            if len(zs) != 1:
                raise SystemExit(f"[c] {run}: expected exactly 1 own snapshot, got {zs}")
            out.append((lab, os.path.join(d, zs[0])))
        elif rel.endswith("CKPT"):
            d = f"{MD}/{run}/checkpoints"
            zs = sorted(f for f in os.listdir(d) if f.endswith(".zip"))
            if len(zs) != 1:
                raise SystemExit(f"[c] {run}: expected exactly 1 checkpoint, got {zs}")
            out.append((lab, os.path.join(d, zs[0])))
        else:
            out.append((lab, f"{MD}/{run}/{rel}"))
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
            raise SystemExit(f"[c] SHAPE MISMATCH on {nm}: {tuple(t.shape)} vs parent {shp}")
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
            out[i:j] = np.where(mask_all[i:j] > 0.5,
                                dist.distribution.logits.detach().numpy(), 0.0)
    if not np.isfinite(out).all():
        raise SystemExit("[c] non-finite logit on a LEGAL action")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", required=True, choices=sorted(ERAS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    th.set_num_threads(a.threads)
    E = ERAS[a.era]
    t0 = time.time()

    if a.era == "gen":
        from agents.training.instrumented_ppo.distill_anchor import masked_kl_rows
    else:
        from era_kl import masked_kl_rows_era as masked_kl_rows

    d = np.load(E["states"], allow_pickle=False)
    obs_all = d["observation"].astype(np.float32)
    mask_all = d["action_mask"].astype(np.float32)
    team = np.array([str(x) for x in d["team"]])
    grp = np.array([str(x) for x in d["group"]])
    N = obs_all.shape[0]
    teams = sorted(set(team.tolist()))
    print(f"[c] era={a.era}  {N} states / {len(teams)} teams "
          f"({int((grp == 'taught').sum())} taught / {int((grp == 'untaught').sum())} untaught rows)",
          flush=True)

    cv = current_model_version(load_mappings())
    parent, _ = load_foreign_opponent(E["parent"], current_version=cv, device="cpu",
                                      config_path=E["parent_cfg"])
    strip_debugger(parent)
    parent.policy.set_training_mode(False)
    if int(parent.num_timesteps) != E["parent_steps"]:
        raise SystemExit(f"[c] parent step {parent.num_timesteps} != declared {E['parent_steps']}")
    layout, order, P = build_layout(parent.policy)
    theta_p = flatten(parent.policy, order, P)
    pbuf = {k: v.detach().numpy().copy() for k, v in parent.policy.named_buffers()}
    Lp = masked_logits(parent.policy, obs_all, mask_all)
    print(f"[c] parent {E['parent']} step {parent.num_timesteps:,}  P={P:,} in "
          + ", ".join(f"{g}={layout[g][1]-layout[g][0]:,}" for g in GROUPS)
          + f"  ({time.time()-t0:.0f}s)", flush=True)
    del parent

    theta_g_norm = {g: float(np.linalg.norm(theta_p[layout[g][0]:layout[g][1]].astype(np.float64)))
                    for g in GROUPS}
    theta_norm = float(np.linalg.norm(theta_p.astype(np.float64)))

    cols, keys, per = [], [], {}
    Lp_t = th.as_tensor(Lp)
    mk_t = th.as_tensor(mask_all)
    for arm, run in E["arms"]:
        for lab, path in resolve_depths(a.era, run):
            key = f"{arm}@{lab}"
            m, _ = load_foreign_opponent(path, current_version=cv, device="cpu",
                                         config_path=f"{MD}/{run}/model_config.json")
            strip_debugger(m)
            m.policy.set_training_mode(False)
            got = set(dict(m.policy.named_parameters()))
            want = {nm for nm, _, _, _ in order}
            if got != want:
                raise SystemExit(f"[c] KEY MISMATCH on {key}: "
                                 f"+{sorted(got-want)[:5]} -{sorted(want-got)[:5]}")
            col = (flatten(m.policy, order, P) - theta_p).astype(np.float64)
            steps = int(m.num_timesteps)
            t = steps - E["parent_steps"]
            if t <= 0:
                raise SystemExit(f"[c] {key}: t={t} is not positive (steps {steps})")

            # --- (iii) the output-side twin, on the era's frozen batch --------------------------
            Lq = th.as_tensor(masked_logits(m.policy, obs_all, mask_all))
            kl = masked_kl_rows(Lp_t, Lq, mk_t).numpy().astype(np.float64)
            if not np.isfinite(kl).all() or (kl < -1e-6).any():
                raise SystemExit(f"[c] {key}: bad KL rows")
            per_team_kl = {tm: float(kl[team == tm].mean()) for tm in teams}

            abuf = {kk: v.detach().numpy() for kk, v in m.policy.named_buffers()}
            bufs = {}
            for bn in sorted(set(pbuf) & set(abuf)):
                dv = float(np.abs(abuf[bn].astype(np.float64)
                                  - pbuf[bn].astype(np.float64)).max())
                if dv > 0:
                    bufs[bn] = dict(max_abs_change=dv,
                                    parent=np.asarray(pbuf[bn]).ravel()[:4].tolist(),
                                    arm=np.asarray(abuf[bn]).ravel()[:4].tolist())

            tot_sq = float(col @ col)
            g_stats = {}
            for g in GROUPS:
                s, e = layout[g]
                cg = col[s:e]
                sq = float(cg @ cg)
                g_stats[g] = dict(n_params=e - s, l2=float(np.sqrt(sq)),
                                  rel_l2=float(np.sqrt(sq) / theta_g_norm[g]),
                                  sq_share=sq / tot_sq)
            per[key] = dict(
                arm=arm, depth=lab, run=run, path=path, steps=steps, t=t,
                fit_point=(lab != "d3ck"),
                l2=float(np.sqrt(tot_sq)), rel_l2=float(np.sqrt(tot_sq) / theta_norm),
                groups=g_stats, buffers_changed=bufs,
                kl_all=float(kl.mean()),
                kl_taught=float(kl[grp == "taught"].mean()),
                kl_untaught=float(kl[grp == "untaught"].mean()),
                kl_per_team=per_team_kl)
            cols.append(col.astype(np.float32))
            keys.append(key)
            print(f"[c] {key:8s} t={t:>9,}  |d|={np.sqrt(tot_sq):.4f} rel={np.sqrt(tot_sq)/theta_norm:.5f}"
                  f"  KL={kl.mean():.4f}  "
                  + " ".join(f"{g[:5]}={g_stats[g]['sq_share']*100:.1f}%" for g in GROUPS)
                  + f"  ({time.time()-t0:.0f}s)", flush=True)
            del m

    # --- (ii) replicate cosines, per group, over every (arm, depth) column ----------------------
    D = np.stack(cols, axis=1)                       # [P, K]
    cos = {}
    for g in ["ALL"] + GROUPS:
        s, e = (0, P) if g == "ALL" else layout[g]
        B = D[s:e, :]
        Gm = (B.T @ B).astype(np.float64)
        nn = np.sqrt(np.diag(Gm))
        cos[g] = (Gm / np.outer(nn, nn)).tolist()

    out = dict(_meta=dict(
        era=a.era, parent=E["parent"], parent_cfg=E["parent_cfg"],
        parent_steps=E["parent_steps"], states=os.path.abspath(E["states"]), n_states=N,
        n_params=P, groups=GROUPS,
        group_layout={g: list(layout[g]) for g in GROUPS},
        parent_group_l2=theta_g_norm, parent_l2=theta_norm,
        grouping_source="../sharing_kernel/kernel.py (imported)",
        kl_source=("agents.training.instrumented_ppo.distill_anchor.masked_kl_rows"
                   if a.era == "gen" else "../content_locality/era_kl.masked_kl_rows_era"),
        keys=keys, threads=a.threads, wall_s=round(time.time() - t0, 1)),
        arms=per, cosine=cos)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[c] wrote {a.out} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
