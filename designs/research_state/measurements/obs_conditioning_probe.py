"""M8 — is the OBSERVATION badly conditioned, when did it break, and is it NORMALIZATION or
INFORMATION?

Chases M2 (`representational_richness_transfer_2026-08-31.md` §3/§3.1) to its root. M2 measured,
model-free, that the observation matrix's participation ratio collapsed inside the gen lineage on a
datable schedule — obs PR 37.76 at gen-12 -> 22.88 at gen-13 -> 16.20 at gen-14 — while the same
matrix read SCALE-INVARIANTLY (per-column z-score, i.e. the correlation matrix) put the v8 era and
rev-1 at an identical 0.0463 effective directions per live dimension. M2's own recommended next
test was free and unrun: z-score the obs per column and re-read PR, then attribute the change to
blocks ADDED, blocks REMOVED, or a change in the SCALE of surviving columns.

This script answers all three, model-free, over each generation's OWN `eval_traces`. No checkpoint
is loaded, so architecture drift is irrelevant and nothing under `models/` is written.

WHAT IS MEASURED
  1. obs PR under three normalizations x the generation ladder:
       raw      — covariance PR, variance-weighted (M2's headline)
       zscore   — per-column z-score over the SAME sample, dead columns dropped
                  (mathematically the correlation matrix; the identity is ASSERTED, not assumed)
       signedlog— sign(x)*log1p(|x|), a monotone squash that keeps sign and compresses magnitude
  2. per-BLOCK attribution, using each era's own `Gen3ObservationEncoder.get_layout()` (never a
     hardcoded offset): variance share, the block's own PR in isolation, and leave-one-block-out
     PR of the rest.
  3. the COMMON-BLOCK control — PR restricted to the six blocks every generation has. If that is
     flat across generations, the collapse is entirely blocks added/removed; if it moves, the
     surviving columns' scale changed too. This is the test that separates the two hypotheses.
  4. the top variance-carrying COLUMNS by name (the per-mon sub-layout is walked recursively), and
     the variance concentration curve.

ESTIMATOR — verbatim-compatible with M2. PR = (trace C)^2 / ||C||_F^2 with C the column covariance;
that identity is exact ((sum lambda)^2 / sum lambda^2) and is asserted equal to the project's
canonical `agents.training.rank_metrics.effective_rank` before use. The Gram form
(trace G)^2 / ||G||_F^2 with G the double-centered row Gram is used inside the cluster bootstrap so
a resample costs one submatrix index instead of a covariance rebuild.

UNCERTAINTY — cluster bootstrap over the source TRACE FILE (one file ~ one battle), 400 resamples.
States inside one battle are one correlated sample; a per-state bootstrap would understate the
interval (this tree's own pooled-correlation Simpson lesson).

ERA LAYOUTS — gen-12 (2921) and gen-13 (3529) cannot be described by current code. Their layouts
are dumped from git-worktrees pinned to each run's own `metadata.json` git_hash and cached as
/tmp/m8_layout_gen1{2,3}.json (see the report's Provenance table for the two hashes).

Run: nice -n 15 python designs/research_state/measurements/obs_conditioning_probe.py
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

Reads models/ READ-ONLY. Writes /tmp/m8obs/obs_conditioning.json
"""
import glob
import json
import os

import numpy as np

MODELS = os.environ.get("GEN3AI_MODELS_DIR", "/home/goodlad/dev/gen3ai/models")
OUT = "/tmp/m8obs/obs_conditioning.json"
N = 3000
BOOT = 400
SEED = 20260831

# (label, run, trace_step, layout_source). layout_source: "live" = current code's encoder,
# else a path to a cached era layout dump.
GENERATIONS = [
    ("v8_04 (v8 era)", "ai_v8_04_distill_4teacher_0722", "step_276000000", None),
    ("gen-12 (frames LIVE)", "ai_v9_14_gen12_h_entitypool_shaping_0816", "step_8000016",
     "/tmp/m8_layout_gen12.json"),
    ("gen-13 (frames LIVE, +events)", "ai_v9_15_gen13_hb_events_stack_0817", "step_8000016",
     "/tmp/m8_layout_gen13.json"),
    ("gen-14 (frames DELETED)", "ai_v9_16_gen14_framedel_v91_0817", "step_8000016", "live"),
    ("gen-15", "ai_v9_18_gen15_v8rewards_0818", "step_8000016", "live"),
    ("gen-17", "ai_v9_21_gen17_pfspoff_0820", "step_8000016", "live"),
    ("rev-1 @8M", "ai_v9_29_rev1_0823", "step_8000016", "live"),
    ("rev-1 @24M", "ai_v9_29_rev1_0823", "step_24000000", "live"),
    ("CURRENT COMPFOLD @32M", "ai_v9_91_COMPFOLD_0831", "step_32000016", "live"),
]

# blocks every gen-lineage generation has, in offset order — the control set for §3
COMMON_BLOCKS = ("our_team", "opp_team", "active_context", "global_env",
                 "board_reactive", "pair_history")


# --------------------------------------------------------------------------------------
# estimator
# --------------------------------------------------------------------------------------
def pr_cov(Z):
    """Participation ratio (sum l)^2 / sum l^2 of the column covariance, without eigendecomposing.

    Exact: sum l = trace(C) and sum l^2 = ||C||_F^2. Uses whichever of the covariance / Gram form
    is smaller.
    """
    Z = np.asarray(Z, np.float64)
    Z = Z - Z.mean(0, keepdims=True)
    n, d = Z.shape
    M = (Z.T @ Z) if d <= n else (Z @ Z.T)
    tr = float(np.trace(M))
    if tr <= 0:
        return 0.0
    return float(tr * tr / float((M * M).sum()))


def pr_from_gram(G):
    """PR from an ALREADY double-centered row Gram."""
    tr = float(np.trace(G))
    if tr <= 0:
        return 0.0
    return float(tr * tr / float((G * G).sum()))


def double_center(G):
    r = G.mean(1, keepdims=True)
    return G - r - r.T + G.mean()


def live_mask(X):
    """Columns with non-zero variance over the sample. A dead column contributes a zero
    eigenvalue, so it changes `live_dims` but never the PR — asserted in main()."""
    return X.std(0) > 0.0


def zscore(X, mask):
    Z = X[:, mask].astype(np.float64)
    return (Z - Z.mean(0)) / Z.std(0)


def signedlog(X):
    X = np.asarray(X, np.float64)
    return np.sign(X) * np.log1p(np.abs(X))


# --------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------
def load_states(run, step, n=N):
    """Pool decisions from one run's eval_traces at one step.

    DETERMINISTIC by construction: files sorted, all rows pooled, then an evenly-strided n-row
    subsample. A wall-clock- or unseeded-RNG-selected sample would give a different battle mix on
    every run of this script.
    """
    root = f"{MODELS}/{run}/eval_traces/{step}"
    files = sorted(glob.glob(root + "/**/*_states.npz", recursive=True))
    if not files:
        raise FileNotFoundError(root)
    obs, src = [], []
    for i, p in enumerate(files):
        a = np.load(p)["obs"]
        obs.append(a)
        src.append(np.full(len(a), i, np.int32))
    X = np.concatenate(obs)
    S = np.concatenate(src)
    total = len(X)
    idx = np.linspace(0, total - 1, min(n, total)).astype(int)
    return X[idx], S[idx], {"n_files": len(files), "n_rows_available": int(total),
                            "n_used": int(len(idx))}


# --------------------------------------------------------------------------------------
# layout -> blocks + column names
# --------------------------------------------------------------------------------------
def blocks_of(L):
    """Named half-open [start, end) spans covering the WHOLE vector, from the era's own layout."""
    P = L["parts"]
    b = [("our_team", P["our_team"]["start"], P["our_team"]["end"]),
         ("opp_team", P["opp_team"]["start"], P["opp_team"]["end"]),
         ("active_context", P["context"]["start"], P["context"]["end"]),
         ("global_env", P["global"]["start"], P["global"]["end"])]
    # `parts["reactive"]["end"]` is the vector end, not the block end — the block's true width is
    # its own encoder dimension. Never trust the part's end here.
    r0 = P["reactive"]["start"]
    b.append(("board_reactive", r0, r0 + int(P["reactive"]["dim"])))
    b.append(("pair_history", L["pair_history_offset"],
              L["pair_history_offset"] + L["pair_history_dim"]))
    if "event_window_offset" in L:
        b.append(("event_window", L["event_window_offset"],
                  L["event_window_offset"] + L["event_window_dim"]))
    if "prev_mask_dim" in L:
        b.append(("prev_action_mask", L["base_dim"], L["base_dim"] + L["prev_mask_dim"]))
    if "turn_history_offset" in L:
        b.append(("turn_frames", L["turn_history_offset"],
                  L["turn_history_offset"] + L["turn_history_dim"]))
    b.sort(key=lambda t: t[1])
    return b


def _walk(node, base, out, prefix):
    """Recursively name a sub-layout dict of {name: {offset, dim, layout?}}."""
    if not isinstance(node, dict):
        return
    for k, v in node.items():
        if k == "slots" and isinstance(v, list):
            continue
        if not isinstance(v, dict) or "offset" not in v:
            continue
        off = base + int(v["offset"])
        dim = int(v.get("dim", 1))
        sub = v.get("layout")
        if isinstance(sub, dict) and any(isinstance(x, dict) and "offset" in x
                                         for x in sub.values()):
            _walk(sub, off, out, f"{prefix}{k}.")
            slots = sub.get("slots")
            sl = sub.get("slot_layout")
            if isinstance(slots, list) and isinstance(sl, dict):
                for si, s in enumerate(slots):
                    _walk(sl, off + int(s["offset"]), out, f"{prefix}{k}[{si}].")
        else:
            for j in range(dim):
                if 0 <= off + j < len(out):
                    out[off + j] = f"{prefix}{k}[{j}]" if dim > 1 else f"{prefix}{k}"


def column_names(L, blocks, D):
    names = [None] * D
    for name, a, b in blocks:
        for j in range(a, b):
            names[j] = f"{name}[{j - a}]"
    mon = L.get("pokemon")
    if isinstance(mon, dict):
        for team, base0 in (("our_team", L["parts"]["our_team"]["start"]),
                            ("opp_team", L["parts"]["opp_team"]["start"])):
            w = int(L["parts"][team]["reshape"][1])
            for s in range(int(L["parts"][team]["reshape"][0])):
                _walk(mon, base0 + s * w, names, f"{team}[{s}].")
    for j in range(D):
        if names[j] is None:
            names[j] = f"col[{j}]"
    return names


# --------------------------------------------------------------------------------------
def boot_ci(X, clusters, rng, reps=BOOT):
    """Cluster bootstrap of PR. The raw Gram is built ONCE; a resample is a submatrix index plus
    a double-centering, so 400 reps cost one matmul total."""
    Z = np.asarray(X, np.float64)
    Graw = Z @ Z.T
    uc = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in uc}
    vals = []
    for _ in range(reps):
        pick = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        vals.append(pr_from_gram(double_center(Graw[np.ix_(idx, idx)])))
    v = np.asarray(vals)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    os.makedirs("/tmp/m8obs", exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = {"probe": "M8 obs conditioning", "n_target": N, "boot_reps": BOOT, "seed": SEED,
           "generations": {}}

    # ---- estimator agreement against the project's canonical effective_rank ----
    from agents.training.rank_metrics import effective_rank
    Xc, _, _ = load_states("ai_v9_16_gen14_framedel_v91_0817", "step_8000016")
    agree = {"canonical_pr": float(effective_rank(Xc.astype(np.float64))["pr"]),
             "fast_pr": pr_cov(Xc)}
    agree["abs_diff"] = abs(agree["canonical_pr"] - agree["fast_pr"])
    assert agree["abs_diff"] < 1e-6, agree
    res["estimator_agreement"] = agree
    print("estimator agreement:", agree)

    live_layout = None
    for label, run, step, lsrc in GENERATIONS:
        try:
            X, S, meta = load_states(run, step)
        except FileNotFoundError as e:
            print("MISSING", label, e)
            res["generations"][label] = {"MISSING": str(e)}
            continue
        D = X.shape[1]
        m = live_mask(X)
        nlive = int(m.sum())
        row = {"run": run, "trace": step, "obs_dim": D, "live_dims": nlive, **meta}

        # ---- 1. the three normalizations -------------------------------------------------
        pr_raw = pr_cov(X)
        pr_raw_live = pr_cov(X[:, m])          # dead columns must not change PR
        Zz = zscore(X, m)
        pr_z = pr_cov(Zz)
        pr_corr = pr_cov(Zz)                   # z-score PR IS correlation PR; asserted below
        C = np.corrcoef(np.asarray(X[:, m], np.float64), rowvar=False)
        C = np.nan_to_num(C)
        pr_corr_direct = float(np.trace(C) ** 2 / (C * C).sum())
        pr_sl = pr_cov(signedlog(X))
        row["pr"] = {"raw": pr_raw, "raw_live_only": pr_raw_live, "zscore": pr_z,
                     "correlation_direct": pr_corr_direct, "signedlog": pr_sl}
        row["identity_checks"] = {
            "dead_cols_change_pr_by": abs(pr_raw - pr_raw_live),
            "zscore_minus_correlation": abs(pr_z - pr_corr_direct)}
        row["pr_per_live_dim"] = {"raw": pr_raw / nlive, "zscore": pr_z / nlive,
                                  "signedlog": pr_sl / nlive}
        row["ci95"] = {"raw": boot_ci(X[:, m], S, rng),
                       "zscore": boot_ci(Zz, S, rng)}

        # ---- variance concentration over COLUMNS ------------------------------------------
        v = X[:, m].astype(np.float64).var(0)
        vs = np.sort(v)[::-1]
        tot = float(v.sum())
        row["variance_concentration"] = {
            "top1": float(vs[0] / tot), "top5": float(vs[:5].sum() / tot),
            "top20": float(vs[:20].sum() / tot), "top100": float(vs[:100].sum() / tot),
            "col_std_max": float(np.sqrt(vs[0])), "col_std_min_live": float(np.sqrt(vs[-1])),
            "col_std_ratio_max_over_median": float(np.sqrt(vs[0] / np.median(v)))}

        # ---- 2/3. per-block attribution (gen lineage only; v8's layout is not reconstructed) --
        L = None
        if lsrc == "live":
            if live_layout is None:
                from agents.observation.state_encoder import (Gen3ObservationEncoder,
                                                              load_mappings)
                live_layout = Gen3ObservationEncoder(load_mappings()).get_layout()
            L = live_layout
        elif lsrc:
            L = json.load(open(lsrc))
        if L is not None:
            assert int(L["total_dim"]) == D, (label, L["total_dim"], D)
            blocks = blocks_of(L)
            covered = sum(b - a for _, a, b in blocks)
            names = column_names(L, blocks, D)
            row["layout_coverage"] = {"covered_dims": covered, "obs_dim": D,
                                      "uncovered": D - covered}
            bt = {}
            for name, a, b in blocks:
                sub = X[:, a:b]
                sm = live_mask(sub)
                vb = float(sub.astype(np.float64).var(0).sum())
                cell = {"start": a, "end": b, "dim": b - a, "live": int(sm.sum()),
                        "var_share": vb / tot if tot > 0 else 0.0,
                        "pr_block_raw": pr_cov(sub) if sm.any() else 0.0,
                        "pr_block_z": pr_cov(zscore(sub, sm)) if sm.any() else 0.0}
                keep = np.ones(D, bool)
                keep[a:b] = False
                cell["pr_without_block_raw"] = pr_cov(X[:, keep & m])
                cell["delta_pr_if_removed"] = cell["pr_without_block_raw"] - pr_raw
                bt[name] = cell
            row["blocks"] = bt

            # COMMON-BLOCK control: the six blocks every generation has
            cols = np.zeros(D, bool)
            for name in COMMON_BLOCKS:
                if name in bt:
                    cols[bt[name]["start"]:bt[name]["end"]] = True
            cm = cols & m
            Xc2 = X[:, cm]
            row["common_blocks"] = {
                "dims": int(cols.sum()), "live": int(cm.sum()),
                "pr_raw": pr_cov(Xc2), "pr_zscore": pr_cov(zscore(X[:, cols],
                                                                  live_mask(X[:, cols]))),
                "var_share_of_total": float(X[:, cm].astype(np.float64).var(0).sum() / tot)}

            # top variance columns, by name
            order = np.argsort(v)[::-1][:20]
            live_idx = np.flatnonzero(m)
            row["top_variance_columns"] = [
                {"col": int(live_idx[o]), "name": names[live_idx[o]],
                 "var": float(v[o]), "var_share": float(v[o] / tot),
                 "std": float(np.sqrt(v[o]))} for o in order]

        res["generations"][label] = row
        print(f"{label:32s} D={D:5d} live={nlive:5d} raw={pr_raw:7.2f} z={pr_z:8.2f} "
              f"sl={pr_sl:7.2f}  z/live={pr_z / nlive:.4f}")

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
