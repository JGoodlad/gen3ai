"""PROBE F step 3 — geometry.  Everything derives from GRAM MATRICES.

Every quantity the mission asks for (pairwise cosines, the within-team noise
ceiling, mean/PC1 energy fractions, the PCGrad projection arithmetic) is a
function of inner products between the 108 gradient rows, so we compute one
108x108 Gram per parameter GROUP (chunked over parameters, never holding the
1.4 GB matrix) and read every number off those.

NOISE HANDLING.  A batch gradient is (true team gradient + batch noise).  Two
disjoint batches of the SAME team therefore give the noise ceiling directly, and
a cross-half construction (folds {0,2,4} vs {1,3,5}) gives an UNBIASED estimate
of the noise-free Gram: E<g_i^A, g_j^B> = <mu_i, mu_j> for every i,j including
i==j, because the two halves' noises are independent.
"""
import itertools
import json
import os
import sys

import numpy as np

OUT = "/tmp/probeF"
CHUNK = 400_000
TAG = "_init" if "--at-init" in sys.argv else ""
LOSSES = ("distill", "bc") if TAG else ("distill", "bc", "pgmc", "pg")

GROUPS = {
    "trunk_encoder": ("embeddings", "pokemon_encoder", "entity_seats",
                      "history_events", "edge_bias", "belief_slots", "damage_op",
                      "intent_conditional", "intent_move_cell",
                      "intent_threshold_move", "switch_branch",
                      "conditional_threat", "pair_outcome_switch",
                      "pair_outcome_move", "pre_proj_norm", "prefuse_proj"),
    "team_transformer": ("team_transformer",),
    "projection_pool": ("projection", "cls_pool"),
    "policy_head": ("POLICY_NET", "pointer_head"),
    "value_path": ("VALUE_NET", "value_net", "value_projection",
                   "value_entity_pool", "value_pre_norm", "value_dist_head",
                   "win_head"),
    "aux_belief": ("belief_head", "hidden_opp_belief", "move_belief",
                   "item_belief_head", "spread_belief", "hp_type_belief_head",
                   "alpha_head", "beta_head", "cf_evid_head", "cf_twin_head_b",
                   "cf_twin_head_c", "cf_shadow_head"),
}


def group_of(name):
    if name.startswith("mlp_extractor.policy_net"):
        key = "POLICY_NET"
    elif name.startswith("mlp_extractor.value_net"):
        key = "VALUE_NET"
    elif name.startswith("features_extractor."):
        key = name.split(".")[1]
    else:
        key = name.split(".")[0]
    for g, members in GROUPS.items():
        if key in members:
            return g
    return "UNGROUPED:" + key


def cos_from_gram(G):
    d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    return G / np.outer(d, d)


def main():
    idx = json.load(open(f"{OUT}/grads_index{TAG}.json"))
    meta = json.load(open(f"{OUT}/states_per_team_meta.json"))
    rows, names, sizes, P = idx["rows"], idx["param_names"], idx["param_sizes"], idx["P"]
    n = len(rows)
    G = np.memmap(f"{OUT}/grads{TAG}.dat", dtype=np.float32, mode="r", shape=(n, P))

    # per-parameter group id over the flat index
    gid = np.empty(P, np.int16)
    glist = sorted({group_of(nm) for nm in names})
    off = 0
    census = {g: 0 for g in glist}
    for nm, sz in zip(names, sizes):
        g = group_of(nm)
        gid[off:off + sz] = glist.index(g)
        census[g] += sz
        off += sz
    assert off == P
    print("param census:", json.dumps(census, indent=1))

    grams = {g: np.zeros((n, n), np.float64) for g in glist}
    for s in range(0, P, CHUNK):
        e = min(s + CHUNK, P)
        block = np.asarray(G[:, s:e], dtype=np.float64)
        sub = gid[s:e]
        for k, g in enumerate(glist):
            m = sub == k
            if not m.any():
                continue
            grams[g] += block[:, m] @ block[:, m].T
        del block
    gram_full = sum(grams.values())
    grams["ALL"] = gram_full

    teams = sorted(meta["teams"])
    nf = idx["n_folds"]

    def rows_of(loss, team):
        return [i for i, r in enumerate(rows)
                if r["loss"] == loss and r["team"] == team]

    report = {"teams": {t: meta["teams"][t] for t in teams},
              "param_census": census, "n_folds": nf, "batch": idx["batch"],
              "student": idx["student"], "groups": {}}

    for gname, GR in grams.items():
        C = cos_from_gram(GR)
        gout = {}
        for loss in LOSSES:
            R = {t: rows_of(loss, t) for t in teams}

            # ---- within-team noise ceiling (distinct folds, same team) ----
            within = {t: [C[a, b] for ii, a in enumerate(R[t])
                          for b in R[t][ii + 1:]] for t in teams}
            within_all = [v for t in teams for v in within[t]]

            # ---- between-team, batch level (directly comparable) ----
            pair_batch = {}
            for i, ti in enumerate(teams):
                for tj in teams[i + 1:]:
                    vals = [C[a, b] for a in R[ti] for b in R[tj]]
                    pair_batch[f"{ti[:8]}|{tj[:8]}"] = float(np.mean(vals))

            # ---- noise-free (cross-half) team Gram ----
            # half A = folds 0,2,4 ; half B = folds 1,3,5 (independent noise)
            A = np.zeros((len(teams), n)); B = np.zeros((len(teams), n))
            for k, t in enumerate(teams):
                for f in range(nf):
                    (A if f % 2 == 0 else B)[k, R[t][f]] = 1.0 / (nf / 2)
            # <A_i, B_j> is unbiased for <mu_i, mu_j>; symmetrize
            M = A @ GR @ B.T
            M = 0.5 * (M + M.T)
            dd = np.diag(M).copy()
            # A team is RESOLVED only if its own cross-half inner product is
            # positive -- <g_i^A, g_i^B> <= 0 means the batch gradient carries no
            # detectable team-consistent signal at this sample size, so every
            # cosine involving it is undefined, not zero.  MISSING, not imputed.
            ok = dd > 0
            keep = np.flatnonzero(ok)
            Mk = M[np.ix_(keep, keep)]
            K = len(keep)
            dsr = np.sqrt(np.clip(np.diag(Mk), 1e-30, None))
            Ctk = Mk / np.outer(dsr, dsr)
            Ct = np.full((len(teams), len(teams)), np.nan)
            if K:
                Ct[np.ix_(keep, keep)] = Ctk

            # ---- energy fractions on the NOISE-FREE team Gram ----
            def energies(Msym):
                if Msym.size == 0:
                    return {"UNRESOLVED": "no team survived the cross-half test"}
                tot = float(np.trace(Msym))
                w, _ = np.linalg.eigh(Msym)
                w = np.sort(w)[::-1]
                one = np.ones(Msym.shape[0])
                den = float(one @ Msym @ one)
                num = float(one @ Msym @ Msym @ one) / max(den, 1e-30)
                return {"n_teams": int(Msym.shape[0]),
                        "total_energy": tot,
                        "mean_direction_energy_fraction":
                            (num / tot) if (tot > 0 and den > 0) else None,
                        "pc1_energy_fraction": float(w[0] / tot) if tot > 0 else None,
                        "pc2_energy_fraction":
                            float(w[1] / tot) if (tot > 0 and len(w) > 1) else None,
                        "eigen_spectrum": [float(x) for x in w],
                        "mean_grad_norm_sq": den / (Msym.shape[0] ** 2)}

            e_raw, e_unit = energies(Mk), energies(Ctk if K else np.zeros((0, 0)))

            # ---- UNCERTAINTY on every noise-free cosine ----
            # There are C(6,3)/2 = 10 balanced ways to split the folds into two
            # halves; each gives an independent-noise estimate of the same
            # quantity.  Their spread IS the estimator's uncertainty, so a
            # "negative pair" is only a finding if the whole spread is negative.
            splits = [h for h in itertools.combinations(range(nf), nf // 2)
                      if 0 in h]
            stack = []
            for h in splits:
                other = [f for f in range(nf) if f not in h]
                A2 = np.zeros((len(teams), n)); B2 = np.zeros((len(teams), n))
                for k2, t2 in enumerate(teams):
                    for f in h:
                        A2[k2, R[t2][f]] = 1.0 / len(h)
                    for f in other:
                        B2[k2, R[t2][f]] = 1.0 / len(other)
                M2 = A2 @ GR @ B2.T
                M2 = 0.5 * (M2 + M2.T)
                d2 = np.diag(M2)
                with np.errstate(invalid="ignore", divide="ignore"):
                    C2 = M2 / np.outer(np.sqrt(np.where(d2 > 0, d2, np.nan)),
                                       np.sqrt(np.where(d2 > 0, d2, np.nan)))
                stack.append(C2)
            ST = np.stack(stack)                       # [n_splits, T, T]
            with np.errstate(invalid="ignore"):
                sp_mean = np.nanmean(ST, 0)
                sp_sd = np.nanstd(ST, 0)
                sp_min = np.nanmin(ST, 0)
                sp_max = np.nanmax(ST, 0)
                sp_cnt = np.sum(np.isfinite(ST), 0)
            iu_all = np.triu_indices(len(teams), 1)
            robust_neg = [(f"{teams[a][:8]}|{teams[b][:8]}",
                           float(sp_mean[a, b]), float(sp_sd[a, b]),
                           float(sp_max[a, b]), int(sp_cnt[a, b]))
                          for a, b in zip(*iu_all)
                          if sp_cnt[a, b] >= 5 and sp_max[a, b] < 0]
            split_stats = {
                "n_splits": len(splits),
                "pair_mean": [[None if not np.isfinite(v) else float(v)
                               for v in r] for r in sp_mean],
                "pair_sd": [[None if not np.isfinite(v) else float(v)
                             for v in r] for r in sp_sd],
                "pair_min": [[None if not np.isfinite(v) else float(v)
                              for v in r] for r in sp_min],
                "pair_n_finite": [[int(v) for v in r] for r in sp_cnt],
                "pairs_negative_across_EVERY_split": robust_neg,
                "median_pair_sd": float(np.nanmedian(sp_sd[iu_all])),
            }

            # ---- PCGrad arithmetic ----
            iu = np.triu_indices(K, 1)
            offs = Ctk[iu] if K > 1 else np.zeros(0)
            neg = offs[offs < 0]
            # PCGrad projects g_i off g_j for every conflicting j; each such
            # projection removes ||g_i||*|cos_ij| of NORM (|cos_ij|^2 of ENERGY).
            # Summing over j is an UPPER BOUND (the projections are sequential
            # and interfere), which is the right side to err on for a verdict.
            per_team_removed, per_team_removed_energy = [], []
            for k in range(K):
                c = Ctk[k].copy(); c[k] = 0
                per_team_removed.append(float(np.sum(np.abs(c[c < 0]))))
                per_team_removed_energy.append(float(np.sum(c[c < 0] ** 2)))
            unresolved = [teams[i] for i in range(len(teams)) if not ok[i]]
            # distinguish STRUCTURALLY ZERO (this loss cannot reach these
            # parameters at all -- e.g. a policy CE never touches the value
            # head) from UNRESOLVED-BY-NOISE.  Both print as MISSING, but only
            # one of them is a sample-size problem.
            allrows = [i for t in teams for i in R[t]]
            struct_zero = bool(np.max(np.diag(GR)[allrows]) <= 0.0)
            gout[loss] = {
                "structurally_zero": struct_zero,
                "unresolved_teams": [] if struct_zero else unresolved,
                "unresolved_reason":
                    ("this loss has EXACTLY ZERO gradient on this parameter "
                     "group by construction" if struct_zero else
                     ("cross-half <g^A,g^B> <= 0: no team-consistent signal "
                      "above batch noise at this sample size")
                     if unresolved else None),
                "isotropic_null_pc1_fraction": 1.0 / max(K, 1),
                "within_team_cosine_ceiling": {
                    "per_team": {t: {"mean": float(np.mean(within[t])),
                                     "min": float(np.min(within[t])),
                                     "max": float(np.max(within[t]))}
                                 for t in teams},
                    "pooled_mean": float(np.mean(within_all)),
                    "pooled_min": float(np.min(within_all)),
                    "pooled_max": float(np.max(within_all)),
                    "n_pairs": len(within_all)},
                "between_team_batch_level": {
                    "pairs": pair_batch,
                    "min": float(min(pair_batch.values())),
                    "median": float(np.median(list(pair_batch.values()))),
                    "max": float(max(pair_batch.values())),
                    "n_pairs": len(pair_batch)},
                "between_team_noise_free": {
                    "matrix": [[None if np.isnan(v) else float(v) for v in r]
                               for r in Ct],
                    "team_order": teams,
                    "resolved_teams": [teams[i] for i in keep],
                    "min": float(offs.min()) if offs.size else None,
                    "median": float(np.median(offs)) if offs.size else None,
                    "max": float(offs.max()) if offs.size else None,
                    "n_negative": int((offs < 0).sum()),
                    "n_pairs": int(offs.size),
                    "n_pairs_possible": len(teams) * (len(teams) - 1) // 2,
                    "most_negative_pairs": sorted(
                        [(f"{teams[keep[a]][:8]}|{teams[keep[b]][:8]}",
                          float(Ctk[a, b])) for a, b in zip(*iu)],
                        key=lambda x: x[1])[:5],
                    "exceeds_unity": int((np.abs(offs) > 1.0).sum())},
                "split_uncertainty": split_stats,
                "energy_raw": e_raw,
                "energy_unit_normalized": e_unit,
                "pcgrad": {
                    "n_negative_pairs": int(neg.size),
                    "n_pairs_examined": int(offs.size),
                    "sum_negative_cosine": float(np.sum(np.abs(neg))) if neg.size else 0.0,
                    "mean_frac_of_own_norm_removed_upper_bound":
                        float(np.mean(per_team_removed)) if K else None,
                    "max_frac_of_own_norm_removed_upper_bound":
                        float(np.max(per_team_removed)) if K else None,
                    "mean_frac_of_own_energy_removed_upper_bound":
                        float(np.mean(per_team_removed_energy)) if K else None,
                    "per_team_frac_removed": {teams[keep[i]]: per_team_removed[i]
                                              for i in range(K)}},
            }
        report["groups"][gname] = gout

    json.dump(report, open(f"{OUT}/geometry{TAG}.json", "w"), indent=1)

    # ---- console summary ----
    def fm(x, w=7, p=4):
        return "  MISSING" if x is None else f"{x:+{w}.{p}f}"

    for gname in ["ALL"] + [g for g in glist]:
        for loss in LOSSES:
            d = report["groups"][gname][loss]
            print(f"\n== {gname} / {loss}")
            w = d["within_team_cosine_ceiling"]
            b = d["between_team_batch_level"]
            f = d["between_team_noise_free"]
            print(f"  within-team ceiling  mean {fm(w['pooled_mean'])}  "
                  f"[{fm(w['pooled_min'])},{fm(w['pooled_max'])}]  n={w['n_pairs']}")
            print(f"  between batch-level  min {fm(b['min'])}  med {fm(b['median'])}  "
                  f"max {fm(b['max'])}")
            print(f"  between NOISE-FREE   min {fm(f['min'])}  med {fm(f['median'])}  "
                  f"max {fm(f['max'])}   negatives {f['n_negative']}/{f['n_pairs']}"
                  f" (of {f['n_pairs_possible']} possible)")
            if d["structurally_zero"]:
                print("  STRUCTURALLY ZERO -- this loss cannot reach this group")
            elif d["unresolved_teams"]:
                print(f"  UNRESOLVED teams: {[t[:8] for t in d['unresolved_teams']]}"
                      f" -- {d['unresolved_reason']}")
            print(f"  energy raw:  mean-dir {fm(d['energy_raw'].get('mean_direction_energy_fraction'))}"
                  f"  PC1 {fm(d['energy_raw'].get('pc1_energy_fraction'))}")
            print(f"  energy unit: mean-dir {fm(d['energy_unit_normalized'].get('mean_direction_energy_fraction'))}"
                  f"  PC1 {fm(d['energy_unit_normalized'].get('pc1_energy_fraction'))}")
            su = d["split_uncertainty"]
            print(f"  split-jackknife: median per-pair sd "
                  f"{su['median_pair_sd']:.4f} over {su['n_splits']} splits; "
                  f"pairs negative in EVERY split: "
                  f"{len(su['pairs_negative_across_EVERY_split'])}")
            for nm, mu, sd, mx, c in su["pairs_negative_across_EVERY_split"][:6]:
                print(f"      {nm}  mean {mu:+.3f} sd {sd:.3f} worst-case-max "
                      f"{mx:+.3f} (n_finite {c}/{su['n_splits']})")
            print(f"  PCGrad would remove (upper bound): "
                  f"{fm(d['pcgrad']['mean_frac_of_own_norm_removed_upper_bound'])} "
                  f"of a team gradient's norm "
                  f"({d['pcgrad']['n_negative_pairs']}/{d['pcgrad']['n_pairs_examined']}"
                  f" negative pairs)")
    print("\n=== FULL noise-free cosine matrix, ALL params ===")
    for loss in LOSSES:
        d = report["groups"]["ALL"][loss]["between_team_noise_free"]
        print(f"\n-- {loss}")
        hdr = "            " + " ".join(f"{t[:8]:>8s}" for t in d["team_order"])
        print(hdr)
        for i, t in enumerate(d["team_order"]):
            cells = " ".join("    MISS" if v is None else f"{v:+8.3f}"
                             for v in d["matrix"][i])
            print(f"{t[:8]:>10s}  {cells}")
    print("\nwrote", f"{OUT}/geometry{TAG}.json")


if __name__ == "__main__":
    main()
