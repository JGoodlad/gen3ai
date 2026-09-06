"""P2 -- does the FIRST-ORDER projection of a fold's displacement predict its off-slice movement?

At the PARENT's parameters, for every state s and every LEGAL action a:

    g_a(s) = grad_theta log pi_parent(a|s)
    u_a    = <g_a, Delta_theta>                     (exactly additive over groups)
    KL1(s) = 1/2 Delta^T F(s) Delta = 1/2 sum_a p_a u_a^2      (F = sum_a p_a g_a g_a^T exactly,
                                                                since sum_a p_a g_a = 0)
    delta(s) = u_{a*}                               (the argmax-action score projection)

against the ACTUAL KL(parent||arm)(s), computed with `masked_kl_rows` IMPORTED from
agents.training.instrumented_ppo.distill_anchor -- the identical statistic the live anchor monitor
and offline_collateral_kl.py use.

GROUP DECOMPOSITION. u_a = sum_g u_a^(g), so the quadratic decomposes EXACTLY as
KL1 = sum_g [1/2 sum_a p_a u_a u_a^(g)]. That additive share is primary. The group-ALONE quadratic
1/2 sum_a p_a (u_a^(g))^2 is reported beside it and does NOT sum to KL1.

Inference: cluster bootstrap over TEAMS (8 untaught / 16 taught), one FIXED resampling index set
shared by every arm so an arm-vs-arm difference is paired on the same team draws.

Run:
  PYTHONPATH=<worktree>/src nice -n 10 python project.py --scratch <dir> --out projection.json
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
from kernel import GROUPS, group_of  # noqa: E402

from agents.model.snapshot import current_model_version, load_foreign_opponent  # noqa: E402
from agents.observation.state_encoder import load_mappings  # noqa: E402
from agents.training.instrumented_ppo.distill_anchor import masked_kl_rows  # noqa: E402

from deltas import PARENT, PARENT_CFG, build_layout, strip_debugger  # noqa: E402

#: the canonical seeded CLUSTERED off-slice KL of ../../reuse_batch_2026-09-03/offline_collateral_kl
PUBLISHED_OFFSLICE = {"R4DOSE12": 0.3062, "R4DOSE6": 0.3502, "C1": 0.3702,
                      "B2": 0.3938, "R4DOSE3": 0.4416}


def ranks(x):
    x = np.asarray(x, dtype=np.float64)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    r[o] = np.arange(len(x), dtype=np.float64)
    # average ties
    i = 0
    xs = x[o]
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def cluster_boot(per_team_by_arm, team_ids, nboot, rng):
    """per_team_by_arm: {arm: np.array over teams}. ONE shared index set across arms."""
    T = len(team_ids)
    idx = rng.integers(0, T, size=(nboot, T))
    out = {}
    for arm, v in per_team_by_arm.items():
        draws = v[idx].mean(axis=1)
        out[arm] = dict(mean=float(v.mean()),
                        ci95=[float(np.percentile(draws, 2.5)),
                              float(np.percentile(draws, 97.5))])
    return out, idx


def paired_diff(per_team_by_arm, idx, a, b):
    d = per_team_by_arm[a] - per_team_by_arm[b]
    draws = d[idx].mean(axis=1)
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return dict(delta=float(d.mean()), ci95=[lo, hi],
                verdict="SEPARATES" if (lo > 0 or hi < 0) else "NOT DETECTED")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=os.path.join(HERE, os.pardir, "sharing_kernel",
                                                     "states_gen.npz"))
    ap.add_argument("--disp", default=os.path.join(HERE, "displacement.json"))
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "projection.json"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--nboot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260905)
    a = ap.parse_args(argv)
    th.set_num_threads(a.threads)
    t0 = time.time()

    disp = json.load(open(a.disp))
    arm_keys = disp["_meta"]["arm_keys"]
    K = len(arm_keys)
    layout = {g: tuple(v) for g, v in disp["_meta"]["group_layout"].items()}
    P = disp["_meta"]["n_params"]

    d = np.load(a.states, allow_pickle=False)
    obs_all = d["observation"].astype(np.float32)
    mask_all = d["action_mask"].astype(np.float32)
    team_all = np.array([str(x) for x in d["team"]])
    grp_all = np.array([str(x) for x in d["group"]])
    N = obs_all.shape[0]
    A = mask_all.shape[1]

    D = np.load(os.path.join(a.scratch, "deltas.npy"), mmap_mode="r")
    L = np.load(os.path.join(a.scratch, "logits.npy"))
    if D.shape != (P, K) or L.shape != (K + 1, N, A):
        raise SystemExit(f"[p] scratch shape mismatch: {D.shape} {L.shape}")
    Dt = th.from_numpy(np.ascontiguousarray(D))          # [P, K] float32

    cv = current_model_version(load_mappings())
    parent, _ = load_foreign_opponent(PARENT, current_version=cv, device="cpu",
                                      config_path=PARENT_CFG)
    strip_debugger(parent)
    parent.policy.set_training_mode(False)
    policy = parent.policy
    lay2, order, P2 = build_layout(policy)
    if P2 != P or {g: tuple(v) for g, v in lay2.items()} != layout:
        raise SystemExit("[p] layout drift between deltas.py and project.py")
    params = []
    slots = []
    for nm, s, e, shp in order:
        slots.append((s, e))
    named = dict(policy.named_parameters())
    params = [named[nm] for nm, _, _, _ in order]
    for nm, _, _, _ in order:
        if group_of(nm) not in GROUPS:
            raise SystemExit(f"[p] ungrouped {nm}")

    space = policy.observation_space.spaces
    gflat = th.zeros(P, dtype=th.float32)
    # u_g[s, a, g, k]; u[s,a,k] = sum_g u_g
    U = np.zeros((N, A, len(GROUPS), K), dtype=np.float64)
    Pr = np.zeros((N, A), dtype=np.float64)
    astar = np.zeros(N, dtype=np.int64)
    nlegal = np.zeros(N, dtype=np.int64)
    ngrad = 0
    for i in range(N):
        m_i = mask_all[i] > 0.5
        legal = np.flatnonzero(m_i)
        nlegal[i] = len(legal)
        ob = {"observation": th.as_tensor(obs_all[i:i + 1]),
              "action_mask": th.as_tensor(mask_all[i:i + 1])}
        ob = {k: v for k, v in ob.items() if k in space}
        dist = policy.get_distribution(ob, action_masks=m_i[None, :])
        logits = dist.distribution.logits
        pr = th.softmax(logits, dim=-1).detach().numpy()[0]
        Pr[i] = np.where(m_i, pr, 0.0)
        Pr[i] /= Pr[i].sum()
        astar[i] = int(np.argmax(np.where(m_i, logits.detach().numpy()[0], -np.inf)))
        for j, act in enumerate(legal):
            lp = dist.log_prob(th.as_tensor([int(act)]))
            gs = th.autograd.grad(lp.sum(), params, allow_unused=True,
                                  retain_graph=(j < len(legal) - 1))
            gflat.zero_()
            for (s, e), gv in zip(slots, gs):
                if gv is not None:
                    gflat[s:e] = gv.detach().reshape(-1)
            for gi, g in enumerate(GROUPS):
                s, e = layout[g]
                U[i, act, gi, :] = (gflat[s:e] @ Dt[s:e, :]).numpy()
            ngrad += 1
        if (i + 1) % 50 == 0:
            print(f"  grad {i+1}/{N}  ({ngrad} action-grads, {time.time()-t0:.0f}s)", flush=True)
    print(f"[p] {ngrad} action-gradients in {time.time()-t0:.0f}s "
          f"(mean legal {nlegal.mean():.2f})", flush=True)

    Usum = U.sum(axis=2)                                  # [N, A, K]  = <g_a, Delta_k>
    legal_f = (mask_all > 0.5).astype(np.float64)         # [N, A]
    w = Pr * legal_f                                      # [N, A]

    # first-order KL and its EXACT additive group decomposition
    KL1 = 0.5 * np.einsum("na,nak->nk", w, Usum ** 2)                    # [N, K]
    KL1_g = 0.5 * np.einsum("na,nak,nagk->ngk", w, Usum, U)              # [N, G, K] sums to KL1
    KL1_alone = 0.5 * np.einsum("na,nagk->ngk", w, U ** 2)               # [N, G, K] does NOT
    delta_lin = np.take_along_axis(Usum, astar[:, None, None], axis=1)[:, 0, :]   # [N, K]

    # ACTUAL forward KL(parent || arm), imported statistic
    mk = th.as_tensor(mask_all)
    pl = th.as_tensor(L[0])
    KLact = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        KLact[:, k] = masked_kl_rows(pl, th.as_tensor(L[k + 1]), mk).numpy()
    if not np.isfinite(KLact).all():
        raise SystemExit("[p] non-finite actual KL")

    teams = sorted(set(team_all.tolist()))
    tgrp = {t: grp_all[team_all == t][0] for t in teams}
    rng = np.random.default_rng(a.seed)

    out = {"_meta": dict(parent=PARENT, parent_steps=28_115_184, n_states=N, n_params=P,
                         arm_keys=arm_keys, n_action_grads=int(ngrad),
                         mean_n_legal=float(nlegal.mean()), nboot=a.nboot, seed=a.seed,
                         threads=a.threads,
                         statistic="KL1 = 1/2 sum_a p_a <g_a,Delta>^2 at the PARENT's params; "
                                   "actual = masked_kl_rows(parent, arm) [imported]",
                         published_offslice_clustered=PUBLISHED_OFFSLICE),
           "slices": {}}

    # exact-decomposition check
    resid = float(np.abs(KL1_g.sum(axis=1) - KL1).max())
    out["_meta"]["group_decomposition_max_residual"] = resid
    print(f"[p] group decomposition exact to {resid:.3e}", flush=True)

    for slice_name in ("untaught", "taught"):
        sel_teams = [t for t in teams if tgrp[t] == slice_name]
        rows = {t: np.flatnonzero(team_all == t) for t in sel_teams}
        sel = np.flatnonzero(grp_all == slice_name)

        per_team = {}
        for k, key in enumerate(arm_keys):
            per_team[key] = dict(
                actual=np.array([KLact[rows[t], k].mean() for t in sel_teams]),
                kl1=np.array([KL1[rows[t], k].mean() for t in sel_teams]),
                absdelta=np.array([np.abs(delta_lin[rows[t], k]).mean() for t in sel_teams]))
        bs_actual, idx = cluster_boot({k: v["actual"] for k, v in per_team.items()},
                                      sel_teams, a.nboot, np.random.default_rng(a.seed))
        bs_kl1, _ = cluster_boot({k: v["kl1"] for k, v in per_team.items()},
                                 sel_teams, a.nboot, np.random.default_rng(a.seed))

        arms = {}
        for k, key in enumerate(arm_keys):
            gshare = {}
            for gi, g in enumerate(GROUPS):
                gshare[g] = dict(
                    additive_share=float(KL1_g[sel, gi, k].mean() / KL1[sel, k].mean()),
                    additive_mean=float(KL1_g[sel, gi, k].mean()),
                    alone_mean=float(KL1_alone[sel, gi, k].mean()),
                    alone_share_of_KL1=float(KL1_alone[sel, gi, k].mean()
                                             / KL1[sel, k].mean()))
            arms[key] = dict(
                actual_kl_state_mean=float(KLact[sel, k].mean()),
                actual_kl_cluster=bs_actual[key],
                kl1_state_mean=float(KL1[sel, k].mean()),
                kl1_cluster=bs_kl1[key],
                delta_lin_mean=float(delta_lin[sel, k].mean()),
                abs_delta_lin_mean=float(np.abs(delta_lin[sel, k]).mean()),
                pearson_kl1_vs_actual=pearson(KL1[sel, k], KLact[sel, k]),
                spearman_kl1_vs_actual=spearman(KL1[sel, k], KLact[sel, k]),
                pearson_absdelta_vs_actual=pearson(np.abs(delta_lin[sel, k]), KLact[sel, k]),
                spearman_absdelta_vs_actual=spearman(np.abs(delta_lin[sel, k]), KLact[sel, k]),
                ratio_kl1_over_actual=float(KL1[sel, k].mean() / KLact[sel, k].mean()),
                groups=gshare)

        # arm ORDERING agreement, on the five reuse arms that have a published number
        reuse = [k for k in arm_keys if k.endswith("@end")
                 and k.split("@")[0] in PUBLISHED_OFFSLICE]
        pub = np.array([PUBLISHED_OFFSLICE[k.split("@")[0]] for k in reuse])
        ord_stats = dict(
            arms=reuse, published=pub.tolist(),
            kl1=[arms[k]["kl1_state_mean"] for k in reuse],
            actual=[arms[k]["actual_kl_state_mean"] for k in reuse],
            spearman_kl1_vs_published=spearman([arms[k]["kl1_state_mean"] for k in reuse], pub),
            spearman_actual_vs_published=spearman([arms[k]["actual_kl_state_mean"] for k in reuse],
                                                  pub),
            spearman_kl1_vs_actual_armlevel=spearman([arms[k]["kl1_state_mean"] for k in reuse],
                                                     [arms[k]["actual_kl_state_mean"]
                                                      for k in reuse]))

        contrasts = {}
        pt_actual = {k: v["actual"] for k, v in per_team.items()}
        pt_kl1 = {k: v["kl1"] for k, v in per_team.items()}
        pairs = [("TCFUNDA@end", "TCUNFA@end"), ("TCFUNDB@end", "TCUNFB@end"),
                 ("TCFUNDA@end", "TCFUNDB@end"), ("TCUNFA@end", "TCUNFB@end"),
                 ("C1@end", "B2@end"), ("R4DOSE3@end", "R4DOSE12@end"),
                 ("B2@end", "R4DOSE3@end")]
        for x, y in pairs:
            contrasts[f"{x} - {y}"] = dict(actual=paired_diff(pt_actual, idx, x, y),
                                           kl1=paired_diff(pt_kl1, idx, x, y))

        out["slices"][slice_name] = dict(n_states=int(len(sel)), n_teams=len(sel_teams),
                                         teams=sel_teams, arms=arms, ordering=ord_stats,
                                         contrasts=contrasts)
        print(f"[p] --- {slice_name}: {len(sel)} states / {len(sel_teams)} teams", flush=True)
        for key in arm_keys:
            v = arms[key]
            print(f"    {key:16s} actual {v['actual_kl_state_mean']:.5f}  "
                  f"KL1 {v['kl1_state_mean']:.5f}  ratio {v['ratio_kl1_over_actual']:6.3f}  "
                  f"r={v['pearson_kl1_vs_actual']:+.3f} rho={v['spearman_kl1_vs_actual']:+.3f}",
                  flush=True)

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(a.scratch, "per_state.npz"),
                        KL1=KL1, KLact=KLact, delta_lin=delta_lin, team=team_all, group=grp_all,
                        KL1_g=KL1_g, KL1_alone=KL1_alone, nlegal=nlegal)
    print(f"[p] wrote {a.out} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
