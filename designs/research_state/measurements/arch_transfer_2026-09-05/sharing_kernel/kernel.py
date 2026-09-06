"""SHARING KERNEL — the cosine gradient kernel between TAUGHT-team and UNTAUGHT-team states.

g_s = grad_theta log pi_theta(a*|s), a* = argmax over LEGAL actions, taken at ONE parameter point
(the era parent's own weights) through the policy's OWN funnel: `policy.get_distribution(obs,
action_masks)`, which in the gen era routes through `_get_action_dist_from_latent` -> pointer_head
and in the v8 era through SB3's flat `action_net`.

K(s,s') = <g_s,g_s'> / (|g_s| |g_s'|), reported for ALL parameters and per PARAMETER GROUP.

THE PRIMARY STATISTIC EXCLUDES SAME-TEAM PAIRS. Two states from one battle on one team are
near-duplicates; counting them inside `within` would inflate it for reasons that have nothing to do
with the taught/untaught split. Every reported mean therefore runs over pairs of states from two
DIFFERENT teams, and `within_same_team` is carried separately as a scale reference.

BECAUSE EVERY TEAM CONTRIBUTES EXACTLY THE SAME NUMBER OF STATES, every state-level mean over
distinct-team pairs equals the corresponding unweighted mean over the 24x24 matrix of team-block
mean cosines. The whole inference (2000 permutations + 2000 cluster bootstraps) therefore runs off
that 24x24 matrix, exactly, at no approximation.

Run: python kernel.py --era gen --states states_gen.npz --out kernel_gen.json
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

from agents.model.snapshot import current_model_version, load_foreign_opponent  # noqa: E402
from agents.observation.state_encoder import load_mappings  # noqa: E402

MD = "/home/goodlad/dev/gen3ai/models"
ERAS = {
    "gen": dict(parent=f"{MD}/ai_v9_59_R2ACTION_0827/final_model.zip",
                cfg=f"{MD}/ai_v9_29_rev1_0823/snapshots/model_config.json",
                steps=28_115_184, head="pointer_head (pointer-native entity head)"),
    "v8": dict(parent=f"{MD}/ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip",
               cfg=f"{MD}/ai_v8_04_distill_4teacher_0722/model_config.json",
               steps=277_583_267, head="action_net (SB3 flat positional Linear(latent,11))"),
}

# --- PARAMETER GROUPS -------------------------------------------------------------------------
# Keyed on the parameter name's leading module path. The list is ORDERED: first match wins, and an
# unmatched parameter RAISES (a silently-ungrouped module would quietly leave norm out of every
# share). Both eras' full module censuses are covered; a name present in only one era is harmless.
GROUP_RULES = [
    # THE ACTION HEAD -- the whole point of the comparison.
    ("pointer_head", "action_head"),            # gen era
    ("action_net", "action_head"),              # v8 era
    # CRITIC PATH -- receives no gradient from log pi. Reported separately, never pooled.
    ("value_net", "critic"),
    ("mlp_extractor.value_net", "critic"),
    ("features_extractor.value_", "critic"),    # value_projection/value_pre_norm/value_dist_head/
                                                # value_entity_pool
    ("features_extractor.win_head", "critic"),
    ("features_extractor.cf_", "critic"),
    ("features_extractor.film_vf", "critic"),
    # THE TEAM TRANSFORMER (+ edge bias).
    ("features_extractor.team_transformer", "team_transformer"),
    ("features_extractor.edge_bias", "team_transformer"),
    # PROJECTIONS / POLICY MLP (incl. the CLS pooling that feeds them, and the pi-side FiLM).
    ("features_extractor.projection", "projection_mlp"),
    ("features_extractor.cls_pool", "projection_mlp"),
    ("features_extractor.film_pi", "projection_mlp"),
    ("mlp_extractor.policy_net", "projection_mlp"),
    # INPUT ENCODERS.
    ("features_extractor.pokemon_encoder", "encoders"),
    ("features_extractor.embeddings", "encoders"),
    ("features_extractor.history_events", "encoders"),
    ("features_extractor.entity_seats", "encoders"),
    ("features_extractor.assembler", "encoders"),
    ("features_extractor.zarch_encoder", "encoders"),
    ("features_extractor.pre_proj_norm", "encoders"),
    ("features_extractor.prefuse_proj", "encoders"),
    # BELIEF HEADS + THE DAMAGE-OP BLOCK AND ITS CELLS.
    ("features_extractor.belief_", "belief_op"),
    ("features_extractor.hidden_opp_belief", "belief_op"),
    ("features_extractor.item_belief_head", "belief_op"),
    ("features_extractor.move_belief", "belief_op"),
    ("features_extractor.spread_belief", "belief_op"),
    ("features_extractor.hp_type_belief_head", "belief_op"),
    ("features_extractor.alpha_head", "belief_op"),
    ("features_extractor.beta_head", "belief_op"),
    ("features_extractor.damage_op", "belief_op"),
    ("features_extractor.intent_", "belief_op"),
    ("features_extractor.pair_outcome_", "belief_op"),
    ("features_extractor.switch_branch", "belief_op"),
    ("features_extractor.conditional_threat", "belief_op"),
    ("features_extractor.outgoing_proj", "belief_op"),
    ("features_extractor.refine_proj", "belief_op"),
    ("features_extractor.status_in_proj", "belief_op"),
    ("features_extractor.status_out_proj", "belief_op"),
]
# Order matters for reporting only.
GROUPS = ["action_head", "encoders", "team_transformer", "projection_mlp", "belief_op", "critic"]


def group_of(name: str) -> str:
    best = None
    for pref, g in GROUP_RULES:
        if name == pref or name.startswith(pref + ".") or name.startswith(pref):
            # longest matching prefix wins, so mlp_extractor.value_net beats nothing and
            # features_extractor.value_projection is not caught by a shorter rule
            if best is None or len(pref) > len(best[0]):
                best = (pref, g)
    if best is None:
        raise SystemExit(f"[k] UNGROUPED PARAMETER: {name} -- add a rule; a silently-ungrouped "
                         "module leaves its norm out of every share")
    return best[1]


def _strip_debugger(m):
    obj = getattr(m, "policy", m)
    for mod in (obj.modules() if hasattr(obj, "modules") else []):
        if getattr(mod, "_debugger", None) is not None:
            mod._debugger = None
    fe = getattr(getattr(m, "policy", m), "features_extractor", None)
    if fe is not None and hasattr(fe, "_debugger"):
        fe._debugger = None
    return m


# --- the pair statistics, all off the 24x24 team-block matrix -----------------------------------

def stats_from_M(M, diag, taught_idx, untaught_idx):
    """M[i,j] = mean cosine over the 19x19 state pairs between team i and team j (i != j).
    diag[i] = mean cosine over the within-team-i pairs (i's own 19 states, off-diagonal).
    Equal states per team => these unweighted team-pair means ARE the state-level means."""
    def pm(rows, cols, same_group):
        vals = []
        for a in rows:
            for b in cols:
                if a == b:
                    continue                      # a team is never paired with itself
                if same_group and b <= a:
                    continue                      # unordered pairs within one label group
                vals.append(M[a, b])
        return float(np.mean(vals)) if vals else float("nan"), len(vals)

    wt, nwt = pm(taught_idx, taught_idx, True)
    wu, nwu = pm(untaught_idx, untaught_idx, True)
    cr, ncr = pm(taught_idx, untaught_idx, False)
    within_pooled = (wt * nwt + wu * nwu) / (nwt + nwu) if (nwt + nwu) else float("nan")
    within_halves = float(np.nanmean([wt, wu]))
    return {"within_taught": wt, "n_pairs_within_taught": nwt,
            "within_untaught": wu, "n_pairs_within_untaught": nwu,
            "cross": cr, "n_pairs_cross": ncr,
            "within_pooled": within_pooled, "within_halves_mean": within_halves,
            "ratio": cr / within_pooled if within_pooled else float("nan"),
            "ratio_halves": cr / within_halves if within_halves else float("nan"),
            "within_same_team": float(np.mean([diag[i] for i in
                                               list(taught_idx) + list(untaught_idx)]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", required=True, choices=sorted(ERAS))
    ap.add_argument("--states", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4,
                    help="torch/BLAS threads for the gradient pass and the Gram matmul")
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260905)
    a = ap.parse_args(argv)
    th.set_num_threads(a.threads)
    E = ERAS[a.era]
    t0 = time.time()

    d = np.load(a.states, allow_pickle=False)
    obs_all = d["observation"].astype(np.float32)
    mask_all = d["action_mask"].astype(np.float32)
    team_all = [str(x) for x in d["team"]]
    grp_all = [str(x) for x in d["group"]]
    n = obs_all.shape[0]
    teams = sorted(set(team_all))
    tidx = {t: i for i, t in enumerate(teams)}
    team_i = np.array([tidx[t] for t in team_all])
    per_team = np.bincount(team_i, minlength=len(teams))
    if len(set(per_team.tolist())) != 1:
        raise SystemExit(f"[k] states per team are not equal ({per_team.tolist()}) -- the "
                         "team-level permutation would not be exchangeable")
    team_group = {}
    for t, g in zip(team_all, grp_all):
        team_group.setdefault(t, g)
    taught_idx = [tidx[t] for t in teams if team_group[t] == "taught"]
    untaught_idx = [tidx[t] for t in teams if team_group[t] == "untaught"]
    print(f"[k] {n} states, {len(teams)} teams ({len(taught_idx)} taught / "
          f"{len(untaught_idx)} untaught), {per_team[0]} states each", flush=True)

    maps = load_mappings()
    cv = current_model_version(maps)
    model, _ = load_foreign_opponent(E["parent"], current_version=cv, device="cpu",
                                     config_path=E["cfg"])
    _strip_debugger(model)
    policy = model.policy
    policy.set_training_mode(False)

    # --- LAYOUT: build the flat gradient vector in GROUP ORDER, so a group is a contiguous
    # column view and no slice ever copies.
    named = list(policy.named_parameters())
    by_group = {g: [] for g in GROUPS}
    for nm, p in named:
        by_group[group_of(nm)].append((nm, p))
    layout, offs, cur = {}, [], 0
    params, pnames = [], []
    for g in GROUPS:
        start = cur
        for nm, p in by_group[g]:
            params.append(p)
            pnames.append(nm)
            offs.append((cur, cur + p.numel()))
            cur += p.numel()
        layout[g] = (start, cur)
    P = cur
    print(f"[k] {P:,} parameters in {len(GROUPS)} groups: "
          + ", ".join(f"{g}={layout[g][1]-layout[g][0]:,}" for g in GROUPS), flush=True)

    obs_space = policy.observation_space.spaces
    G = np.zeros((n, P), dtype=np.float32)
    argmax_actions, n_legal = [], []
    for i in range(n):
        ob = {"observation": th.as_tensor(obs_all[i:i + 1]),
              "action_mask": th.as_tensor(mask_all[i:i + 1])}
        ob = {k: v for k, v in ob.items() if k in obs_space}
        mk = th.as_tensor(mask_all[i:i + 1] > 0.5)
        dist = policy.get_distribution(ob, action_masks=mk.numpy())
        logits = dist.distribution.logits            # masking already applied by get_distribution
        act = int(th.argmax(logits, dim=-1).item())
        lp = dist.log_prob(th.as_tensor([act]))
        gs = th.autograd.grad(lp.sum(), params, allow_unused=True, retain_graph=False)
        for (s, e), gv in zip(offs, gs):
            if gv is not None:
                G[i, s:e] = gv.detach().reshape(-1).numpy()
        argmax_actions.append(act)
        n_legal.append(int(mask_all[i].sum()))
        if (i + 1) % 50 == 0:
            print(f"  grad {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[k] gradients done in {time.time()-t0:.0f}s", flush=True)

    # --- NORM SHARES: "the head is where sharing lives" is meaningless if the head carries no
    # norm. Accumulated in float64 but CHUNKED over rows -- an `.astype(np.float64)` on the whole
    # 456 x 3.1M block would materialise a second 11 GB copy beside G.
    def sumsq_rows(block):
        out = np.zeros(block.shape[0], dtype=np.float64)
        for r0 in range(0, block.shape[0], 32):
            r1 = min(r0 + 32, block.shape[0])
            out[r0:r1] = np.square(block[r0:r1].astype(np.float64)).sum(axis=1)
        return out

    per_state_sq = {g: sumsq_rows(G[:, layout[g][0]:layout[g][1]]) for g in GROUPS}
    tot_check = sum(per_state_sq.values())
    n_zero = int((tot_check <= 0).sum())
    if n_zero:
        raise SystemExit(
            f"[k] {n_zero} state(s) have an IDENTICALLY ZERO gradient. This is what a state with a "
            "single legal action looks like: the masked policy puts all mass on it, log pi(a*|s) "
            "== 0, and the score function vanishes. Such a state has no direction and its cosine "
            "is undefined. Regenerate the batch with the >= 2-legal-actions filter in "
            "gen_states.py rather than dropping rows here -- dropping would break the EQUAL "
            "per-team count the permutation null needs.")
    sq = {g: float(per_state_sq[g].sum()) for g in GROUPS}
    tot_sq = sum(sq.values())
    norm_share = {g: sq[g] / tot_sq for g in GROUPS}
    tot_ps = sum(per_state_sq.values())
    norm_share_mean_per_state = {g: float(np.mean(per_state_sq[g] / tot_ps)) for g in GROUPS}
    print("[k] gradient-norm share (pooled sum-of-squares): "
          + ", ".join(f"{g}={norm_share[g]:.4f}" for g in GROUPS), flush=True)

    rng = np.random.default_rng(a.seed)
    results = {}
    for gname in ["ALL"] + GROUPS:
        if gname == "ALL":
            Gb = G
        else:
            s, e = layout[gname]
            Gb = G[:, s:e]
        nrm = np.sqrt(sumsq_rows(Gb))     # chunked float64; never a whole-block f64 copy
        dead = int((nrm <= 0).sum())
        if dead == n:
            results[gname] = {"status": "ZERO GRADIENT — no state puts any gradient in this "
                                        "group (it is not on the log-pi path)",
                              "dead_states": dead, "n_params": int(Gb.shape[1]),
                              "norm_share_pooled": norm_share.get(gname),
                              "norm_share_mean_per_state": norm_share_mean_per_state.get(gname)}
            print(f"[k] {gname:16s} ZERO GRADIENT (off the log-pi path)", flush=True)
            continue
        safe = np.where(nrm > 0, nrm, 1.0)
        Gn = (Gb.astype(np.float32) / safe[:, None].astype(np.float32))
        K = (Gn @ Gn.T).astype(np.float64)
        if dead:
            K[nrm <= 0, :] = np.nan
            K[:, nrm <= 0] = np.nan

        T = len(teams)
        M = np.full((T, T), np.nan)
        diag = np.full(T, np.nan)
        rows = [np.where(team_i == t)[0] for t in range(T)]
        for i in range(T):
            bi = K[np.ix_(rows[i], rows[i])]
            iu = np.triu_indices(len(rows[i]), 1)
            diag[i] = np.nanmean(bi[iu])
            for j in range(i + 1, T):
                v = float(np.nanmean(K[np.ix_(rows[i], rows[j])]))
                M[i, j] = M[j, i] = v

        obs_stats = stats_from_M(M, diag, taught_idx, untaught_idx)

        # --- PERMUTATION NULL: relabel the 24 teams 16/8, blocks intact.
        allt = np.arange(T)
        nt = len(taught_idx)
        perm_ratio = np.empty(a.nperm)
        perm_gap = np.empty(a.nperm)
        for b in range(a.nperm):
            p = rng.permutation(allt)
            st = stats_from_M(M, diag, p[:nt], p[nt:])
            perm_ratio[b] = st["ratio"]
            perm_gap[b] = st["cross"] - st["within_pooled"]
        obs_gap = obs_stats["cross"] - obs_stats["within_pooled"]
        p_ratio = float((np.abs(perm_ratio - 1.0) >= abs(obs_stats["ratio"] - 1.0)).mean())
        p_gap = float((np.abs(perm_gap) >= abs(obs_gap)).mean())

        # --- CLUSTER BOOTSTRAP over teams, within strata.
        bt = np.empty(a.nboot)
        bg = np.empty(a.nboot)
        ti = np.array(taught_idx)
        ui = np.array(untaught_idx)
        for b in range(a.nboot):
            rt = rng.choice(ti, size=len(ti), replace=True)
            ru = rng.choice(ui, size=len(ui), replace=True)
            st = stats_from_M(M, diag, rt, ru)
            bt[b] = st["ratio"]
            bg[b] = st["cross"] - st["within_pooled"]

        results[gname] = dict(obs_stats)
        results[gname].update({
            "n_params": int(Gb.shape[1]), "dead_states": dead,
            "norm_share_pooled": norm_share.get(gname, 1.0 if gname == "ALL" else None),
            "norm_share_mean_per_state": norm_share_mean_per_state.get(
                gname, 1.0 if gname == "ALL" else None),
            "gap_cross_minus_within": obs_gap,
            "perm_p_ratio": p_ratio, "perm_p_gap": p_gap,
            "perm_ratio_mean": float(perm_ratio.mean()),
            "perm_ratio_sd": float(perm_ratio.std(ddof=1)),
            "perm_gap_sd": float(perm_gap.std(ddof=1)),
            "perm_gap_ci95": [float(np.percentile(perm_gap, 2.5)),
                              float(np.percentile(perm_gap, 97.5))],
            "boot_ratio_ci95": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
            "boot_gap_ci95": [float(np.percentile(bg, 2.5)), float(np.percentile(bg, 97.5))],
            "team_matrix": M.tolist(), "team_diag": diag.tolist(),
        })
        r = results[gname]
        print(f"[k] {gname:16s} within {r['within_pooled']:+.4f}  cross {r['cross']:+.4f}  "
              f"ratio {r['ratio']:.4f}  gap {obs_gap:+.5f}  perm_p(gap) {p_gap:.4f}  "
              f"normshare {r['norm_share_pooled']:.4f}", flush=True)

    out = {"_meta": {"era": a.era, "parent": E["parent"], "parent_steps": E["steps"],
                     "head": E["head"], "states": os.path.abspath(a.states),
                     "n_states": n, "n_teams": len(teams), "states_per_team": int(per_team[0]),
                     "n_params": P, "group_layout": {g: list(layout[g]) for g in GROUPS},
                     "teams_order": teams,
                     "taught_teams": [teams[i] for i in taught_idx],
                     "untaught_teams": [teams[i] for i in untaught_idx],
                     "statistic": "cos(grad log pi(a*|s), grad log pi(a*|s')) at ONE parameter "
                                  "point; a* = argmax over LEGAL actions; SAME-TEAM PAIRS "
                                  "EXCLUDED from within/cross",
                     "nperm": a.nperm, "nboot": a.nboot, "seed": a.seed,
                     "threads": a.threads,
                     "mean_n_legal": float(np.mean(n_legal)),
                     "argmax_action_hist": np.bincount(np.array(argmax_actions),
                                                       minlength=11).tolist(),
                     "wall_s": round(time.time() - t0, 1)},
           "groups": results}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[k] wrote {a.out} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
