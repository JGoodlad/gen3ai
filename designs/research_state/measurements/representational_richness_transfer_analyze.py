"""M2 — assemble the richness table, the locus table and the ordering against untaught externality.

Inputs (all produced read-only, no training, no battles):
  /tmp/plast/fwd/{v8,v9}/*.npy       the 2026-08-28 plasticity audit's own forward dumps (reused)
  /tmp/m2rich/fwd/v9/*.npy           this probe's extra gen-era forwards
  /tmp/m2rich/locus.json             this probe's fold-level weight-delta locus

Estimator: `agents.training.rank_metrics.effective_rank` — the project's canonical participation
ratio, i.e. the SAME function that produced the audit's 50.24 / 20.59. A fast covariance-eigenvalue
PR is used for the bootstrap only, and is asserted equal to the canonical one first.

Uncertainty: CLUSTER bootstrap over the state set's `src_file` (one trace file = one battle-ish
cluster). A per-state bootstrap would understate the interval — states inside one battle are one
correlated sample (this tree's own pooled-correlation Simpson lesson).

Run: nice -n 15 python designs/research_state/measurements/representational_richness_transfer_analyze.py
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
import json
import os

import numpy as np

from agents.training.rank_metrics import effective_rank

AUDIT = "/tmp/plast/fwd"
NEW = "/tmp/m2rich/fwd"
STATES = "/tmp/plast"
OUT = "/tmp/m2rich/richness.json"
N = 3000
BOOT = 400
SEED = 20260831

TAPS = [
    ("features_extractor_pokemon_encoder", "encoder (per-mon)", 1536),
    ("features_extractor_team_transformer", "trunk tokens", 768),
    ("features_extractor_projection", "trunk projection", 512),
    ("features_extractor_cls_pool", "pooled cls", 128),
    ("pi_features", "policy head input", 512),
    ("vf_features", "value head input", 512),
    ("mlp_extractor_policy_net", "policy head (mlp out)", 512),
    ("mlp_extractor_value_net", "value head (mlp out)", 512),
]

# where each model's dump lives
SOURCES = {
    "v8": {"v8 PARENT ai_v8_04 @277.2M": (AUDIT, "v8", "parent"),
           "v8 FOLD ai_v8_14 @292.1M": (AUDIT, "v8", "product"),
           "v8 fork semistall3 @matched": (AUDIT, "v8", "semistall3_m"),
           "v8 fork pool10 @matched": (AUDIT, "v8", "pool10_m"),
           "v8 fork defensive10 @matched": (AUDIT, "v8", "defensive10_m")},
    "v9": {"gen PARENT rev-1 @25.1M": (AUDIT, "v9", "parent"),
           "gen FOLD rev-2 R2ACTION @28.1M": (AUDIT, "v9", "product"),
           "gen FOLD rev-3 R3ACTION @32.6M": (NEW, "v9", "R3ACTION"),
           "gen FOLD rev-4 R4ACTION": (NEW, "v9", "R4ACTION"),
           "gen FOLD COMPFOLD": (NEW, "v9", "COMPFOLD"),
           "gen CTRL R2CTRL (no fork)": (NEW, "v9", "R2CTRL"),
           "gen CTRL R2PLAIN": (NEW, "v9", "R2PLAIN"),
           "gen fork F5a": (AUDIT, "v9", "F5a"),
           "gen fork F5c": (AUDIT, "v9", "F5c")},
}
LADDER = [2400000, 4800000, 8045088, 12101808, 16001808, 20073792, 24988992]
# v8-lineage maturity points (era code); the archive keeps nothing below ~149M in this lineage.
LADDER_V8 = [("v8_03_step149598621", 149598621), ("v8_03_step200364858", 200364858),
             ("v8_03_step267612744", 267612744), ("v8_04_step269716291", 269716291)]

# measured untaught-team externality of each FOLD (percentage points, fold - its own parent)
EXTERNALITY = {
    "v8 FOLD ai_v8_14 @292.1M": (+5.42, "[+3.44,+7.42]",
                                 "v8_redistribution_pfsp_2026-08-30 (P1, 7680 untaught battles)"),
    "gen FOLD rev-2 R2ACTION @28.1M": (-7.06, "[-10.56,-3.50]",
                                       "rev3_untaught_pulldown_2026-08-30 (B-C, 8 teams x 200)"),
    "gen FOLD rev-3 R3ACTION @32.6M": (-0.75, "[-4.56,+3.00]",
                                       "rev3_untaught_pulldown_2026-08-30 (A-B, 8 teams x 200)"),
}


def dedupe(Z, n=N):
    """Hooks fired 3x per batch; the repeats are byte-identical. Recover one copy + VERIFY."""
    if Z is None or Z.shape[0] == n:
        return Z, 0.0
    if Z.shape[0] != 3 * n:
        raise ValueError(f"unexpected row count {Z.shape[0]} for n={n}")
    out, chk, i, rem = [], 0.0, 0, n
    while rem > 0:
        bs = min(128, rem)
        a, b, c = Z[i:i + bs], Z[i + bs:i + 2 * bs], Z[i + 2 * bs:i + 3 * bs]
        out.append(a)
        chk = max(chk, float(np.abs(a - b).max()), float(np.abs(a - c).max()))
        i += 3 * bs
        rem -= bs
    return np.concatenate(out), chk


def load(root, era, name, tap):
    f = f"{root}/{era}/{name}__{tap}.npy"
    if not os.path.exists(f):
        return None, None
    return dedupe(np.load(f))


def pr_fast(Z):
    """Participation ratio from the covariance spectrum (identical to effective_rank's `pr`)."""
    Zc = np.asarray(Z, np.float64)
    Zc = Zc - Zc.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(Zc.T @ Zc)
    ev = np.clip(ev, 0.0, None)
    tot = ev.sum()
    if tot <= 0:
        return 0.0
    p = ev / tot
    return float(1.0 / np.square(p).sum())


def boot_pr(Z, clusters, rng, reps=BOOT):
    uc = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in uc}
    vals = []
    for _ in range(reps):
        pick = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        vals.append(pr_fast(Z[idx]))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    os.makedirs("/tmp/m2rich", exist_ok=True)
    rng = np.random.default_rng(SEED)
    S = {e: np.load(f"{STATES}/states_{e}.npz") for e in ("v8", "v9")}
    clusters = {e: S[e]["src_file"] for e in S}

    # --- estimator agreement check: fast PR must equal the canonical one ---
    Z, _ = load(AUDIT, "v9", "parent", "pi_features")
    agree = {"canonical_pr": float(effective_rank(Z.astype(np.float64))["pr"]),
             "fast_pr": pr_fast(Z)}
    agree["abs_diff"] = abs(agree["canonical_pr"] - agree["fast_pr"])
    assert agree["abs_diff"] < 1e-6, agree
    print("estimator agreement:", agree)

    res = {"estimator_agreement": agree, "n_states": N, "boot_reps": BOOT,
           "richness": {}, "dedupe_max_repeat_diff": {}, "ladder": {},
           "externality": {k: {"pp": v[0], "ci": v[1], "source": v[2]}
                           for k, v in EXTERNALITY.items()}}

    boot_targets = {"v8 PARENT ai_v8_04 @277.2M", "v8 FOLD ai_v8_14 @292.1M",
                    "gen PARENT rev-1 @25.1M", "gen FOLD rev-2 R2ACTION @28.1M",
                    "gen FOLD rev-3 R3ACTION @32.6M", "gen FOLD rev-4 R4ACTION",
                    "gen FOLD COMPFOLD", "gen CTRL R2CTRL (no fork)",
                    "gen CTRL R2PLAIN"}
    boot_taps = {"features_extractor_projection", "pi_features",
                 "features_extractor_cls_pool"}

    for era in ("v8", "v9"):
        for label, (root, e, name) in SOURCES[era].items():
            row = {}
            for tap, _pretty, _w in TAPS:
                Z, chk = load(root, e, name, tap)
                if Z is None:
                    continue
                # record the VALUE, always — a check only logged when it is non-zero is a
                # check nobody can tell apart from one that never ran.
                res["dedupe_max_repeat_diff"][f"{label}/{tap}"] = chk
                r = effective_rank(Z.astype(np.float64))
                cell = {"pr": float(r["pr"]), "n99": int(r["n99"]),
                        "effrank": float(r["effrank"]), "dim": int(Z.shape[1])}
                if label in boot_targets and tap in boot_taps:
                    lo, hi = boot_pr(Z, clusters[era], rng)
                    cell["ci95"] = [lo, hi]
                row[tap] = cell
            if row:
                res["richness"][label] = {"era": era, "taps": row}
                print(f"{label:34s} " + " ".join(
                    f"{t.split('_')[-1][:6]}={row[t]['pr']:.1f}" for t in row))

    # --- maturity ladder (within rev-1's own run, one FIXED state set) ---
    for s in LADDER:
        row = {}
        for tap, _p, _w in TAPS:
            Z, _ = load(NEW, "v9", f"rev1_step{s}", tap)
            if Z is None:
                continue
            row[tap] = float(effective_rank(Z.astype(np.float64))["pr"])
        res["ladder"][str(s)] = row
        print(f"  rev1 @{s/1e6:6.2f}M  proj={row['features_extractor_projection']:.2f} "
              f"pi={row['pi_features']:.2f} trunk={row['features_extractor_team_transformer']:.2f}")
    res["ladder_v8"] = {}
    for name, s in LADDER_V8:
        row = {}
        for tap, _p, _w in TAPS:
            Z, _ = load(NEW, "v8", name, tap)
            if Z is None:
                continue
            row[tap] = float(effective_rank(Z.astype(np.float64))["pr"])
        if not row:
            continue
        res["ladder_v8"][name] = {"steps": s, **row}
        print(f"  {name:22s} @{s/1e6:7.2f}M  proj={row['features_extractor_projection']:.2f} "
              f"pi={row['pi_features']:.2f} trunk={row['features_extractor_team_transformer']:.2f}")

    # --- PAIRED cluster bootstrap of PR DIFFERENCES ---------------------------------------
    # The unpaired CIs above are wide because resampling battles moves BOTH arms together. Two
    # models forwarded on the SAME states share that movement, so the difference is the statistic
    # with power. Cross-ERA contrasts are deliberately absent: the two eras have different state
    # sets (different obs, different parents) and cannot be paired at all.
    PAIRED = [
        ("v8 FOLD ai_v8_14 @292.1M", "v8 PARENT ai_v8_04 @277.2M", "v8"),
        ("gen FOLD rev-2 R2ACTION @28.1M", "gen PARENT rev-1 @25.1M", "v9"),
        ("gen FOLD rev-2 R2ACTION @28.1M", "gen CTRL R2CTRL (no fork)", "v9"),
        ("gen FOLD rev-2 R2ACTION @28.1M", "gen CTRL R2PLAIN", "v9"),
        ("gen FOLD rev-3 R3ACTION @32.6M", "gen CTRL R2PLAIN", "v9"),
        ("gen FOLD COMPFOLD", "gen CTRL R2PLAIN", "v9"),
        ("gen CTRL R2CTRL (no fork)", "gen PARENT rev-1 @25.1M", "v9"),
    ]
    res["paired_pr_diff"] = {}
    print("\n--- paired cluster bootstrap of PR differences (A - B) ---")
    for a, b, era in PAIRED:
        if a not in res["richness"] or b not in res["richness"]:
            continue
        rootA, _, nameA = SOURCES[era][a]
        rootB, _, nameB = SOURCES[era][b]
        cl = clusters[era]
        uc = np.unique(cl)
        idx_by = {c: np.flatnonzero(cl == c) for c in uc}
        row = {}
        for tap in ("features_extractor_projection", "features_extractor_cls_pool",
                    "pi_features"):
            ZA, _ = load(rootA, era, nameA, tap)
            ZB, _ = load(rootB, era, nameB, tap)
            if ZA is None or ZB is None:
                continue
            point = pr_fast(ZA) - pr_fast(ZB)
            rr = np.random.default_rng(SEED + 1)
            d = []
            for _ in range(BOOT):
                pick = rr.choice(uc, size=len(uc), replace=True)
                idx = np.concatenate([idx_by[c] for c in pick])
                d.append(pr_fast(ZA[idx]) - pr_fast(ZB[idx]))
            d = np.array(d)
            row[tap] = {"delta_pr": point,
                        "ci95": [float(np.percentile(d, 2.5)),
                                 float(np.percentile(d, 97.5))],
                        "frac_below_zero": float((d < 0).mean())}
            print(f"  {a[:30]:30s} - {b[:26]:26s} {tap[-18:]:18s} "
                  f"{point:+6.2f} [{row[tap]['ci95'][0]:+.2f},{row[tap]['ci95'][1]:+.2f}]")
        res["paired_pr_diff"][f"{a} - {b}"] = row

    LOCUS_KEY = {
        "v8 FOLD ai_v8_14 @292.1M": "FOLD v8_14 (3 teachers / 22 taught teams)",
        "gen FOLD rev-2 R2ACTION @28.1M": "FOLD rev-2 R2ACTION (5 teachers / 9 taught)",
        "gen FOLD rev-3 R3ACTION @32.6M": "FOLD rev-3 R3ACTION (6 teachers / 12 taught)",
        "gen FOLD rev-4 R4ACTION": "FOLD rev-4 R4ACTION (3 teachers)",
        "gen FOLD COMPFOLD": "FOLD COMPFOLD (composite)",
    }
    PARENT_OF = {
        "v8 FOLD ai_v8_14 @292.1M": "v8 PARENT ai_v8_04 @277.2M",
        "gen FOLD rev-2 R2ACTION @28.1M": "gen PARENT rev-1 @25.1M",
        "gen FOLD rev-3 R3ACTION @32.6M": "gen FOLD rev-2 R2ACTION @28.1M",
        "gen FOLD rev-4 R4ACTION": "gen FOLD rev-2 R2ACTION @28.1M",
        "gen FOLD COMPFOLD": "gen FOLD rev-2 R2ACTION @28.1M",
        # nulls: same +3M of training with NO fold
        "gen CTRL R2CTRL (no fork)": "gen PARENT rev-1 @25.1M",
        "gen CTRL R2PLAIN": "gen PARENT rev-1 @25.1M",
        # anchor: what a v8 FORK (not fold) does, for era comparability
        "v8 fork pool10 @matched": "v8 PARENT ai_v8_04 @277.2M",
        "gen fork F5a": "gen PARENT rev-1 @25.1M",
    }

    # --- representation-level locus + the "does the delta fit in existing directions" test ---
    # The parameter-level locus is era-ASYMMETRIC at the head (v8's 5.6k action_net vs the gen
    # era's 55k pointer_head are different objects — the audit's own warning). These two
    # statistics are computed on FEATURES over the same states, so they compare behaviour.
    #
    #   cka_distance  : 1 - linear CKA(parent tap, fold tap). Small at the trunk + large at the
    #                   head = a head-local edit; the reverse = a representation edit.
    #   energy_in_parent_topk : of the feature delta's squared norm, how much lies inside the
    #                   parent's OWN top-k principal directions (k = ceil(parent PR)). This is the
    #                   hypothesis stated as a number: content a rich parent can express as a
    #                   combination of directions it already has should land INSIDE; content that
    #                   has to overwrite should not. Reported against the isotropic null k/D.
    res["representation"] = {}
    for fold, par in PARENT_OF.items():
        if fold not in res["richness"]:
            continue
        era = res["richness"][fold]["era"]
        rootF, _, nameF = SOURCES[era][fold]
        rootP, _, nameP = SOURCES[era][par]
        row = {}
        for tap, _p, _w in TAPS:
            A, _ = load(rootP, era, nameP, tap)
            B, _ = load(rootF, era, nameF, tap)
            if A is None or B is None:
                continue
            A = np.asarray(A, np.float64) - np.asarray(A, np.float64).mean(0)
            B = np.asarray(B, np.float64) - np.asarray(B, np.float64).mean(0)
            hs = np.linalg.norm(A.T @ B, "fro") ** 2
            nx = np.linalg.norm(A.T @ A, "fro")
            ny = np.linalg.norm(B.T @ B, "fro")
            cell = {"cka_distance": float(1.0 - hs / (nx * ny + 1e-30))}
            D = A.shape[1]
            k = max(1, min(D, int(np.ceil(pr_fast(A)))))
            _u, _s, Vt = np.linalg.svd(A, full_matrices=False)
            V = Vt[:k].T                       # parent's top-k principal directions
            Delta = B - A
            e_tot = float((Delta * Delta).sum())
            e_in = float(((Delta @ V) ** 2).sum())
            dpr = pr_fast(Delta)
            cell.update(k_parent_topk=k, dim=D,
                        energy_in_parent_topk=e_in / (e_tot + 1e-30),
                        isotropic_null=k / D,
                        delta_pr=dpr,
                        parent_pr=pr_fast(A),
                        # k-FREE: how many more effective directions the change uses than the
                        # representation it is written into. 1.0 = the delta lives in exactly as
                        # many directions as the parent already uses.
                        delta_pr_over_parent_pr=dpr / (pr_fast(A) + 1e-30))
            cell["lift_over_null"] = cell["energy_in_parent_topk"] / (cell["isotropic_null"] + 1e-30)
            # MATCHED-k: the raw fraction above is confounded by k differing across eras
            # (v8 k=51 vs gen k=21 at pi_features). Fixed k removes that confound.
            for kk in (21, 51):
                if kk <= D:
                    Vk = Vt[:kk].T
                    cell[f"energy_in_parent_top{kk}"] = float(
                        ((Delta @ Vk) ** 2).sum()) / (e_tot + 1e-30)
                    cell[f"parent_var_in_top{kk}"] = float(
                        (_s[:kk] ** 2).sum() / ((_s ** 2).sum() + 1e-30))
            row[tap] = cell
        res["representation"][fold] = row
    print("\n--- representation locus (fold vs its own parent) ---")
    for fold, row in res["representation"].items():
        for tap in ("features_extractor_projection", "pi_features", "mlp_extractor_policy_net"):
            if tap not in row:
                continue
            c = row[tap]
            print(f"{fold:34s} {tap[:24]:24s} ckaD={c['cka_distance']:.4f} "
                  f"inTopK={c['energy_in_parent_topk']:.3f}(k={c['k_parent_topk']},lift "
                  f"{c['lift_over_null']:.2f}) k21={c.get('energy_in_parent_top21', float('nan')):.3f} "
                  f"k51={c.get('energy_in_parent_top51', float('nan')):.3f} "
                  f"dPR/pPR={c['delta_pr_over_parent_pr']:.2f}")

    # --- join richness x locus x externality ---
    locus = json.load(open("/tmp/m2rich/locus.json"))
    joined = {}
    for fold, lk in LOCUS_KEY.items():
        if fold not in res["richness"] or lk not in locus or "MISSING" in locus[lk]:
            continue
        par = PARENT_OF[fold]
        ext = EXTERNALITY.get(fold)
        joined[fold] = {
            "parent": par,
            "parent_pr_pi": res["richness"][par]["taps"]["pi_features"]["pr"],
            "parent_pr_projection":
                res["richness"][par]["taps"]["features_extractor_projection"]["pr"],
            "fold_pr_pi": res["richness"][fold]["taps"]["pi_features"]["pr"],
            "trunk_share": locus[lk]["aggregate"]["trunk_share"],
            "shared_share": locus[lk]["aggregate"]["shared_share"],
            "head_share": locus[lk]["aggregate"]["head_share"],
            "global_rel_frob": locus[lk]["global_rel_frob"],
            "untaught_externality_pp": ext[0] if ext else None,
            "untaught_externality_ci": ext[1] if ext else None,
        }
    res["joined"] = joined
    print("\n--- ordering (parent richness / trunk share / untaught externality) ---")
    for k, v in sorted(joined.items(),
                       key=lambda kv: -(kv[1]["untaught_externality_pp"]
                                        if kv[1]["untaught_externality_pp"] is not None else -99)):
        print(f"{k:34s} parentPR(pi)={v['parent_pr_pi']:6.2f} "
              f"trunk={v['trunk_share']:.3f} head={v['head_share']:.3f} "
              f"ext={v['untaught_externality_pp']}")

    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
