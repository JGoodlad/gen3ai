#!/usr/bin/env python3
"""M3 — THE SUBSTRATE HYPOTHESIS.

Does a fold's UNTAUGHT gift require PRIOR COMPETENCE on the untaught team to attach to?

Prediction under the hypothesis: WITHIN a fold, the gift on an untaught team is LARGER where
the parent was already more competent.

Pure re-analysis of two committed artifacts — no battles, no models, no traces:

  * `v8_redistribution_pfsp_2026-08-30_cells.jsonl.gz`  (probe P; 16 untaught + 6 taught teams,
    8 opponent cells x 30 paired battles x 2 arms)
  * `rev3_untaught_pulldown_2026-08-30_cells.jsonl.gz`  (probe Q; 8 untaught teams,
    200 CRN-paired battles x 3 arms, per-battle win vectors)

TWO nuisances both push the naive correlation NEGATIVE, which makes this test
one-sided-informative:

  (1) CEILING HEADROOM.  A team the parent already wins 0.51 on has less room to gain than one
      at 0.22, so gift shrinks with prior competence for purely mechanical reasons.  Controlled
      by working on the LOGIT scale (an ability-additive scale on which a constant gain is
      constant) and by headroom normalisation at four ceilings C, exactly as the exploitability
      decomposition proved its result invariant to C.

  (2) SHARED MEASUREMENT NOISE.  The parent's win rate appears in BOTH axes
      (x = p_hat, y = f_hat - p_hat), so binomial noise in p_hat enters x positively and y
      negatively: cov_noise(x, y) = cov(p_hat, f_hat) - var(p_hat) < 0.  This is regression to
      the mean and it is NOT fixed by any transform.  Controlled by a SPLIT-HALF instrument:
      x is measured on one half of the parent's battles and y on the disjoint other half, so the
      two carry independent noise.  Reported alongside the naive value with the shift priced.

Because both nuisances are negative, a POSITIVE controlled correlation is conservative evidence
FOR the substrate hypothesis, while a null/negative one is ambiguous.  Stated, not smuggled.

Run:
    nice -n 15 python designs/research_state/measurements/substrate_hypothesis.py \
        --out designs/research_state/measurements/substrate_hypothesis_2026-08-31

(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

CEILINGS = [0.60, 0.6881, 0.80, 1.00]
N_SPLITS = 400
N_BOOT = 20000
BOOT_SPLIT_SUB = 40  # splits averaged inside each bootstrap replicate
SEED = 20260831


# --------------------------------------------------------------------------- helpers
def logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def _pearson_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Row-wise Pearson r for two (S, T) matrices."""
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    num = (Xc * Yc).sum(axis=1)
    den = np.sqrt((Xc**2).sum(axis=1) * (Yc**2).sum(axis=1))
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return out


def _ols_slope_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    num = (Xc * Yc).sum(axis=1)
    den = (Xc**2).sum(axis=1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared) — used only for the descriptive Spearman row."""
    a = np.asarray(a, dtype=float)
    r = np.empty(len(a), dtype=float)
    srt = np.argsort(a, kind="mergesort")
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[srt[j + 1]] == a[srt[i]]:
            j += 1
        r[srt[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def cluster_boot(
    X: np.ndarray, Y: np.ndarray, rng: np.random.Generator, stat=_pearson_rows
) -> dict:
    """X, Y are (S, T): S split replicates x T teams.  Bootstrap over TEAMS (the cluster)."""
    S, T = X.shape
    point = float(np.mean(stat(X, Y)))
    sub = min(BOOT_SPLIT_SUB, S)
    reps = np.empty(N_BOOT, dtype=float)
    rows = rng.integers(0, S, size=(N_BOOT, sub)) if S > 1 else np.zeros((N_BOOT, 1), int)
    idx = rng.integers(0, T, size=(N_BOOT, T))
    for b in range(N_BOOT):
        Xb = X[np.ix_(rows[b], idx[b])]
        Yb = Y[np.ix_(rows[b], idx[b])]
        reps[b] = np.mean(stat(Xb, Yb))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "p_gt_0": float(np.mean(reps > 0)),
        "n_teams": int(T),
    }


def single_boot(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, stat=_pearson_rows) -> dict:
    return cluster_boot(x[None, :], y[None, :], rng, stat)


# --------------------------------------------------------------------------- data
def load_v8():
    """probe P cells -> per (team, arm) list of 8 (wins, n) opponent cells."""
    path = HERE / "v8_redistribution_pfsp_2026-08-30_cells.jsonl.gz"
    per = defaultdict(list)
    meta = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            per[(r["team"], r["arm"])].append((int(r["wins"]), int(r["finished"])))
            meta[r["team"]] = {"kind": r["kind"], "arch": r["arch"]}
    teams = sorted({t for (t, _a) in per})
    for t in teams:
        for a in ("parent", "fold"):
            assert len(per[(t, a)]) == 8, (t, a, len(per[(t, a)]))
    return per, meta, teams


def load_rev3():
    """probe Q cells -> per (team, arm) per-battle win vector (index-aligned across arms)."""
    path = HERE / "rev3_untaught_pulldown_2026-08-30_cells.jsonl.gz"
    per = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            per[(r["team"], r["arm"])] = np.asarray(r["wins"], dtype=np.int8)
    teams = sorted({t for (t, _a) in per})
    return per, teams


def archetypes():
    a = json.loads((REPO / "data" / "teams" / "gen3_team_archetypes.json").read_text())
    return a["teams"]


def sha10(path: Path) -> str:
    return hashlib.sha1(path.read_text().strip().encode()).hexdigest()[:10]


# --------------------------------------------------------------------------- split-half machinery
def v8_split_matrices(per, teams, rng, n_splits=N_SPLITS, mode="hyper"):
    """Return dict of (S, T) matrices for the split-half estimators.

    mode="hyper"  : split the 30 battles of EVERY opponent cell 15/15 by a hypergeometric draw
                    on the win count.  Opponent-balanced.  Residual CRN leak priced in the .md.
    mode="opp"    : split the 8 OPPONENT cells 4/4 (identical opponent split for both arms), so
                    x and y draw on disjoint battles exactly.  Opponent-heterogeneous.
    """
    T = len(teams)
    out = {k: np.empty((n_splits, T)) for k in ("pH1", "pH2", "fH2", "fH1")}
    for s in range(n_splits):
        for ti, t in enumerate(teams):
            pc = per[(t, "parent")]
            fc = per[(t, "fold")]
            if mode == "hyper":
                pw1 = sum(rng.hypergeometric(w, n - w, n // 2) for (w, n) in pc)
                fw1 = sum(rng.hypergeometric(w, n - w, n // 2) for (w, n) in fc)
                ptot, ftot = sum(w for w, _ in pc), sum(w for w, _ in fc)
                half = sum(n for _, n in pc) // 2
                out["pH1"][s, ti] = pw1 / half
                out["pH2"][s, ti] = (ptot - pw1) / half
                out["fH1"][s, ti] = fw1 / half
                out["fH2"][s, ti] = (ftot - fw1) / half
            else:
                sel = rng.permutation(len(pc))
                h1, h2 = sel[: len(pc) // 2], sel[len(pc) // 2 :]
                for nm, arm, cells in (("p", "parent", pc), ("f", "fold", fc)):
                    for tag, hh in (("H1", h1), ("H2", h2)):
                        w = sum(cells[i][0] for i in hh)
                        n = sum(cells[i][1] for i in hh)
                        out[nm + tag][s, ti] = w / n
                del arm
    return out


def rev3_split_matrices(per, teams, arm_f, arm_p, rng, n_splits=N_SPLITS):
    """Exact aligned battle-index split: x from index set S, y from its complement."""
    T = len(teams)
    out = {k: np.empty((n_splits, T)) for k in ("pH1", "pH2", "fH2", "fH1")}
    for s in range(n_splits):
        for ti, t in enumerate(teams):
            p = per[(t, arm_p)].astype(float)
            f = per[(t, arm_f)].astype(float)
            n = len(p)
            perm = rng.permutation(n)
            h1, h2 = perm[: n // 2], perm[n // 2 :]
            out["pH1"][s, ti] = p[h1].mean()
            out["pH2"][s, ti] = p[h2].mean()
            out["fH1"][s, ti] = f[h1].mean()
            out["fH2"][s, ti] = f[h2].mean()
    return out


# --------------------------------------------------------------------------- the estimator suite
def analyse_fold(name, teams, p_full, f_full, splits, rng, extra_x=None):
    """p_full/f_full: (T,) full-sample win rates.  splits: dict of (S,T) half matrices."""
    res = {"fold": name, "n_teams": len(teams), "teams": list(teams)}
    p = np.asarray(p_full, float)
    f = np.asarray(f_full, float)
    gift = f - p
    res["parent_wr"] = p.round(4).tolist()
    res["fold_wr"] = f.round(4).tolist()
    res["gift_pp"] = (100 * gift).round(2).tolist()
    res["parent_wr_mean"] = float(p.mean())
    res["parent_wr_sd"] = float(p.std(ddof=1))
    res["gift_pp_mean"] = float(100 * gift.mean())
    res["gift_pp_sd"] = float(100 * gift.std(ddof=1))

    # ---- NAIVE (uncontrolled for shared noise) -----------------------------------------
    naive = {}
    naive["raw_pp_vs_wr"] = single_boot(p, 100 * gift, rng)
    naive["logit"] = single_boot(logit(p), logit(f) - logit(p), rng)
    naive["logit_spearman"] = single_boot(rank(logit(p)), rank(logit(f) - logit(p)), rng)
    naive["logit_slope"] = single_boot(logit(p), logit(f) - logit(p), rng, stat=_ols_slope_rows)
    naive["headroom"] = {}
    for C in CEILINGS:
        den = np.maximum(C - p, 0.05)
        naive["headroom"][f"{C:.4f}"] = single_boot(p, gift / den, rng)
    res["naive"] = naive

    # ---- SPLIT-HALF (shared noise removed) ---------------------------------------------
    pH1, pH2, fH2 = splits["pH1"], splits["pH2"], splits["fH2"]
    sh = {}
    sh["raw_pp_vs_wr"] = cluster_boot(pH1, 100 * (fH2 - pH2), rng)
    sh["logit"] = cluster_boot(logit(pH1), logit(fH2) - logit(pH2), rng)
    sh["logit_slope"] = cluster_boot(
        logit(pH1), logit(fH2) - logit(pH2), rng, stat=_ols_slope_rows
    )
    sh["headroom"] = {}
    sh["headroom_floor_rate"] = {}
    for C in CEILINGS:
        den = C - pH2
        floored = float(np.mean(den < 0.05))
        den = np.maximum(den, 0.05)
        sh["headroom"][f"{C:.4f}"] = cluster_boot(pH1, (fH2 - pH2) / den, rng)
        sh["headroom_floor_rate"][f"{C:.4f}"] = floored
    res["split_half"] = sh

    # ---- the shared-noise bias, priced ---------------------------------------------------
    # cov_noise(x, y) = cov(p_hat, f_hat) - var(p_hat).  var(p_hat) is estimated from the
    # observed spread of the two independent halves of the parent's own battles.
    # E[(pH1 - pH2)^2] = var(pH1) + var(pH2) = 2*var_half = 4*var_full, so var_full = E[.]/4.
    var_noise_full = float(np.mean((pH1 - pH2) ** 2) / 4.0)
    sd_x = float(np.std(p, ddof=1))
    sd_y = float(np.std(gift, ddof=1))
    res["noise"] = {
        "var_parent_wr_measurement_full": var_noise_full,
        "sd_parent_wr_measurement_full_pp": float(100 * np.sqrt(var_noise_full)),
        "sd_parent_wr_across_teams_pp": float(100 * sd_x),
        "measurement_share_of_x_variance": float(var_noise_full / (sd_x**2)),
        "predicted_naive_r_bias_if_uncorrelated_arms": float(-var_noise_full / (sd_x * sd_y)),
        "observed_naive_minus_splithalf_raw": float(
            naive["raw_pp_vs_wr"]["point"] - sh["raw_pp_vs_wr"]["point"]
        ),
        "reliability_of_x": float(max(0.0, 1.0 - var_noise_full / (sd_x**2))),
        "disattenuated_naive_logit_r": None,
    }
    rel = res["noise"]["reliability_of_x"]
    if rel > 0:
        res["noise"]["disattenuated_naive_logit_r"] = float(
            naive["logit"]["point"] / np.sqrt(rel)
        )

    res["null_calibration"] = null_calibration(p, float(gift.mean()))

    if extra_x is not None:
        res["rivals"] = extra_x
    return res


def null_calibration(p: np.ndarray, mean_gift: float) -> dict:
    """The two ceiling controls disagree because each is NEUTRAL under a different null.

    * Under a LOG-ODDS-ADDITIVE null (the fold adds a constant Delta to every team's log-odds --
      the scale ELO/Bradley-Terry already assumes) the logit correlation is 0 BY CONSTRUCTION,
      but headroom-normalised gift RISES with prior competence, so headroom normalisation
      MANUFACTURES a positive.
    * Under a HEADROOM-ADDITIVE null (the fold captures a constant fraction of the linear
      distance to a ceiling C) the headroom correlation at that C is 0 by construction, but the
      logit correlation goes NEGATIVE, so the logit scale manufactures a negative.

    So an observed value must be read against its own null, not against zero.  Both nulls are
    computed noise-free on the OBSERVED prior-competence vector, with the free parameter pinned
    so the null reproduces the fold's observed MEAN gift.
    """
    out = {"note": null_calibration.__doc__.strip().splitlines()[0]}

    # --- log-odds-additive null: solve for Delta reproducing the mean gift
    lo, hi = -5.0, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        g = float((1.0 / (1.0 + np.exp(-(logit(p) + mid))) - p).mean())
        if g < mean_gift:
            lo = mid
        else:
            hi = mid
    delta = 0.5 * (lo + hi)
    f_lo = 1.0 / (1.0 + np.exp(-(logit(p) + delta)))
    out["logodds_additive_null"] = {
        "delta_logit": float(delta),
        "r_logit": 0.0,
        "r_headroom": {
            f"{C:.4f}": float(
                np.corrcoef(p, (f_lo - p) / np.maximum(C - p, 0.05))[0, 1]
            )
            for C in CEILINGS
        },
        "r_raw_pp": float(np.corrcoef(p, f_lo - p)[0, 1]),
    }

    # --- headroom-additive nulls, one per ceiling
    out["headroom_additive_null"] = {}
    for C in CEILINGS:
        den = np.maximum(C - p, 0.05)
        k = mean_gift / float(den.mean())
        f_hr = np.clip(p + k * den, 1e-4, 1 - 1e-4)
        out["headroom_additive_null"][f"{C:.4f}"] = {
            "k": float(k),
            "r_logit": float(np.corrcoef(logit(p), logit(f_hr) - logit(p))[0, 1]),
            "r_raw_pp": float(np.corrcoef(p, f_hr - p)[0, 1]),
        }
    return out


# --------------------------------------------------------------------------- archetype rival
def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def proximity(team_shas, taught_shas, arch):
    """Per-team archetype/tag proximity to the taught set."""
    taught_arch = [arch[s]["archetype"] for s in taught_shas if s in arch]
    taught_tags = [arch[s]["tags"] for s in taught_shas if s in arch]
    out = {}
    for s in team_shas:
        if s not in arch:
            out[s] = None
            continue
        a, tg = arch[s]["archetype"], arch[s]["tags"]
        share = sum(1 for x in taught_arch if x == a) / len(taught_arch)
        js = [jaccard(tg, tt) for tt in taught_tags]
        out[s] = {
            "archetype": a,
            "archetype_share_of_taught": share,
            "archetype_in_taught_set": float(a in set(taught_arch)),
            "tag_jaccard_mean": float(np.mean(js)),
            "tag_jaccard_max": float(np.max(js)),
        }
    return out


def rival_table(teams, prox, p, f, rng):
    gift_l = logit(f) - logit(p)
    x_comp = logit(p)
    cols = {
        "archetype_share_of_taught": np.array(
            [prox[t]["archetype_share_of_taught"] for t in teams]
        ),
        "tag_jaccard_mean": np.array([prox[t]["tag_jaccard_mean"] for t in teams]),
        "tag_jaccard_max": np.array([prox[t]["tag_jaccard_max"] for t in teams]),
    }
    out = {"univariate": {}, "joint_ols": {}}
    for k, v in cols.items():
        out["univariate"][k] = single_boot(v, gift_l, rng)
    out["univariate"]["prior_competence_logit"] = single_boot(x_comp, gift_l, rng)

    # two-predictor OLS: standardised betas + R^2 decomposition
    for k, v in cols.items():
        A = np.column_stack(
            [np.ones(len(teams)), (x_comp - x_comp.mean()) / x_comp.std(), (v - v.mean()) / (v.std() or 1)]
        )
        beta, *_ = np.linalg.lstsq(A, gift_l, rcond=None)
        pred = A @ beta
        ss_res = float(((gift_l - pred) ** 2).sum())
        ss_tot = float(((gift_l - gift_l.mean()) ** 2).sum())
        out["joint_ols"][k] = {
            "beta_prior_competence": float(beta[1]),
            "beta_proximity": float(beta[2]),
            "r2": float(1 - ss_res / ss_tot) if ss_tot else None,
        }
    return out


def _exposure_bound() -> dict:
    """The hypothesis's ANTECEDENT — did v8's parent actually see the pool more? — read from
    the run archive.  This is the one half of the between-era claim that IS decidable from
    committed artifacts, because step counts need no common opponent."""
    try:
        from utils.paths import main_models_dir  # noqa: PLC0415

        root = main_models_dir()
    except Exception:
        root = None
    if root is None:
        return {"status": "SKIPPED — no models/ archive reachable"}
    import re  # noqa: PLC0415

    def max_ckpt(run):
        d = root / run / "checkpoints"
        if not d.is_dir():
            return None
        n = [int(m.group(1)) for f in d.iterdir() if (m := re.search(r"(\d+)_steps", f.name))]
        return max(n) if n else None

    runs = {
        "v8_parent  ai_v8_04_distill_4teacher_0722": max_ckpt("ai_v8_04_distill_4teacher_0722"),
        "v8_fold    ai_v8_14_distill3_0725": max_ckpt("ai_v8_14_distill3_0725"),
        "rev1       ai_v9_29_rev1_0823": max_ckpt("ai_v9_29_rev1_0823"),
        "rev2_parent ai_v9_59_R2ACTION_0827": max_ckpt("ai_v9_59_R2ACTION_0827"),
        "rev3_fold  ai_v9_70_R3ACTION_0828": max_ckpt("ai_v9_70_R3ACTION_0828"),
    }
    p8 = runs["v8_parent  ai_v8_04_distill_4teacher_0722"]
    pg = runs["rev2_parent ai_v9_59_R2ACTION_0827"]
    out = {"max_checkpoint_steps": runs}
    if p8 and pg:
        out["v8_over_gen_step_ratio"] = round(p8 / pg, 2)
        # both folds ran --distill-team-bias 0.4, so 60% of episodes draw over the 719-team pool
        for tag, steps in (("v8_parent", p8), ("gen_parent", pg)):
            out[f"{tag}_episodes_per_pool_team_estimate"] = round(
                0.6 * (steps / 50.0) / 719.0
            )
        out["estimate_caveats"] = (
            "assumes ~50 env steps per episode and uniform 60% pool draw (bias 0.4); v8_04 also "
            "ran --team-pfsp onesided, measured near-inert (TV distance from uniform <= 0.049). "
            "Lineage before each run's own first step is NOT summed."
        )
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "substrate_hypothesis_2026-08-31"))
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    arch = archetypes()
    report = {
        "probe": "M3 substrate hypothesis",
        "date": "2026-08-31",
        "sources": [
            "v8_redistribution_pfsp_2026-08-30_cells.jsonl.gz",
            "rev3_untaught_pulldown_2026-08-30_cells.jsonl.gz",
            "data/teams/gen3_team_archetypes.json",
        ],
        "design": {
            "hypothesis": "within a fold, the untaught gift is LARGER where the parent was already competent",
            "nuisance_1": "ceiling headroom (pushes r negative)",
            "nuisance_2": "shared measurement noise: parent WR in both axes (pushes r negative)",
            "controls": "logit scale + headroom normalisation at C in %s; split-half instrument for the noise"
            % CEILINGS,
            "n_splits": args.splits,
            "n_boot": N_BOOT,
            "seed": SEED,
        },
        "folds": {},
    }
    split_store: dict[str, dict] = {}

    # ------------------------------------------------------------------ v8 (probe P)
    perP, metaP, teamsP = load_v8()
    untaughtP = [t for t in teamsP if metaP[t]["kind"] == "untaught"]
    taughtP = [t for t in teamsP if metaP[t]["kind"] == "taught"]

    def wr(t, a):
        c = perP[(t, a)]
        return sum(w for w, _ in c) / sum(n for _, n in c)

    for label, tset in (("v8_untaught", untaughtP), ("v8_taught_exploratory", taughtP)):
        p = np.array([wr(t, "parent") for t in tset])
        f = np.array([wr(t, "fold") for t in tset])
        sub = {(t, a): perP[(t, a)] for t in tset for a in ("parent", "fold")}
        sp = v8_split_matrices(sub, tset, rng, args.splits, mode="hyper")
        r = analyse_fold(label, tset, p, f, sp, rng)
        r["arms"] = {
            "parent": "ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip",
            "fold": "ai_v8_14_distill3_0725/final_model_interrupted.zip",
            "reference_opponent": "ai_v8_03/final_model_interrupted.zip",
        }
        r["archetype"] = {t: metaP[t]["arch"] for t in tset}
        # robustness: opponent-disjoint split (exactly independent, opponent-heterogeneous)
        sp_opp = v8_split_matrices(sub, tset, rng, args.splits, mode="opp")
        r["split_half_opponent_disjoint"] = {
            "raw_pp_vs_wr": cluster_boot(
                sp_opp["pH1"], 100 * (sp_opp["fH2"] - sp_opp["pH2"]), rng
            ),
            "logit": cluster_boot(
                logit(sp_opp["pH1"]), logit(sp_opp["fH2"]) - logit(sp_opp["pH2"]), rng
            ),
        }
        report["folds"][label] = r
        split_store[label] = sp

    # v8 archetype rival
    taught_paths = json.loads(Path("/tmp/probeP_taught_paths.json").read_text())
    taught_shas, taught_src = [], []
    for pth in taught_paths:
        fp = Path(pth) if os.path.isabs(pth) else REPO / pth
        if fp.exists():
            taught_shas.append(sha10(fp))
            taught_src.append(str(fp))
    taught_shas = sorted(set(taught_shas))
    report["v8_taught_set"] = {
        "n_distinct": len(taught_shas),
        "shas": taught_shas,
        "archetype_counts": {
            a: sum(1 for s in taught_shas if arch.get(s, {}).get("archetype") == a)
            for a in sorted({arch[s]["archetype"] for s in taught_shas if s in arch})
        },
        "provenance": "/tmp/probeP_taught_paths.json (probe P's recorded --trainee-teams resolution)",
    }
    proxP = proximity(untaughtP, taught_shas, arch)
    pU = np.array([wr(t, "parent") for t in untaughtP])
    fU = np.array([wr(t, "fold") for t in untaughtP])
    report["folds"]["v8_untaught"]["rivals"] = rival_table(untaughtP, proxP, pU, fU, rng)
    report["folds"]["v8_untaught"]["proximity"] = proxP

    # ------------------------------------------------------------------ rev era (probe Q)
    perQ, teamsQ = load_rev3()
    qmeta = json.loads((HERE / "rev3_untaught_pulldown_2026-08-30.json").read_text())
    base2sha = {t["basename"]: t["sha"] for t in qmeta["selection"]["teams"]}
    taughtQ = []
    for b in qmeta["taught_union"]:
        fp = REPO / "data" / "teams" / "sample" / f"{b}.txt"
        if fp.exists():
            taughtQ.append(sha10(fp))
    report["rev_taught_union"] = {
        "n_distinct": len(set(taughtQ)),
        "shas": sorted(set(taughtQ)),
        "archetype_counts": {
            a: sum(1 for s in set(taughtQ) if arch.get(s, {}).get("archetype") == a)
            for a in sorted({arch[s]["archetype"] for s in set(taughtQ) if s in arch})
        },
    }

    for label, arm_f, arm_p in (
        ("rev3_untaught", "R3ACTION", "R2ACTION"),
        ("rev2_untaught_mirrorcaveat", "R2ACTION", "REV1"),
    ):
        p = np.array([perQ[(t, arm_p)].mean() for t in teamsQ])
        f = np.array([perQ[(t, arm_f)].mean() for t in teamsQ])
        sp = rev3_split_matrices(perQ, teamsQ, arm_f, arm_p, rng, args.splits)
        r = analyse_fold(label, teamsQ, p, f, sp, rng)
        r["arms"] = {"parent": arm_p, "fold": arm_f, "reference_opponent": "ai_v9_29_rev1 final"}
        r["archetype"] = {t: arch[base2sha[t]]["archetype"] for t in teamsQ}
        prox = proximity([base2sha[t] for t in teamsQ], sorted(set(taughtQ)), arch)
        prox = {t: prox[base2sha[t]] for t in teamsQ}
        r["proximity"] = prox
        r["rivals"] = rival_table(teamsQ, prox, p, f, rng)
        if arm_p == "REV1":
            r["caveat"] = (
                "REV1 is ALSO the fixed reference opponent, so the x-axis arm is a near-mirror; "
                "probe P's method note says exactly this biases the delta. Secondary only."
            )
        report["folds"][label] = r
        split_store[label] = sp

    # ------------------- pooled across the TWO CLEAN folds (v8 + rev-3) -------------------
    # Within-fold z-scoring puts the two eras on one scale without asserting their LEVELS are
    # comparable; the cluster bootstrap resamples teams WITHIN fold, so the pooled estimate
    # generalises over teams, not over eras (n_folds = 2 is not a sample).
    a, b = split_store["v8_untaught"], split_store["rev3_untaught"]

    def _z(m):
        return (m - m.mean(axis=1, keepdims=True)) / m.std(axis=1, keepdims=True)

    def _xy(sp):
        x = logit(sp["pH1"])
        y = logit(sp["fH2"]) - logit(sp["pH2"])
        return _z(x), _z(y)

    xa, ya = _xy(a)
    xb, yb = _xy(b)
    S = min(xa.shape[0], xb.shape[0])
    Xp = np.concatenate([xa[:S], xb[:S]], axis=1)
    Yp = np.concatenate([ya[:S], yb[:S]], axis=1)
    na = xa.shape[1]
    reps = np.empty(N_BOOT)
    rows = rng.integers(0, S, size=(N_BOOT, min(BOOT_SPLIT_SUB, S)))
    for i in range(N_BOOT):
        ia = rng.integers(0, na, size=na)
        ib = na + rng.integers(0, Xp.shape[1] - na, size=Xp.shape[1] - na)
        idx = np.concatenate([ia, ib])
        reps[i] = float(np.mean(_pearson_rows(Xp[np.ix_(rows[i], idx)], Yp[np.ix_(rows[i], idx)])))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    report["pooled_clean_folds"] = {
        "folds": ["v8_untaught", "rev3_untaught"],
        "n_teams": int(Xp.shape[1]),
        "estimator": "split-half logit r, within-fold z-scored, cluster bootstrap over teams within fold",
        "point": float(np.mean(_pearson_rows(Xp, Yp))),
        "lo": float(lo),
        "hi": float(hi),
        "p_gt_0": float(np.mean(reps > 0)),
    }

    # ------------------------------------------------------------------ between-era
    v8p = report["folds"]["v8_untaught"]
    r3p = report["folds"]["rev3_untaught"]
    r2p = report["folds"]["rev2_untaught_mirrorcaveat"]
    report["between_era"] = {
        "v8_parent_untaught_wr": {
            "mean": v8p["parent_wr_mean"],
            "sd": v8p["parent_wr_sd"],
            "min": min(v8p["parent_wr"]),
            "max": max(v8p["parent_wr"]),
            "reference_opponent": "ai_v8_03 final_model_interrupted (an ancestor of both arms)",
            "opponent_teams": "8 FIXED opponent teams",
            "n_games_per_team": 240,
        },
        "rev_parent_untaught_wr": {
            "mean": r3p["parent_wr_mean"],
            "sd": r3p["parent_wr_sd"],
            "min": min(r3p["parent_wr"]),
            "max": max(r3p["parent_wr"]),
            "reference_opponent": "ai_v9_29_rev1 final",
            "opponent_teams": "719-team pool draw",
            "n_games_per_team": 200,
        },
        "rev1_untaught_wr": {
            "mean": r2p["parent_wr_mean"],
            "sd": r2p["parent_wr_sd"],
            "note": "REV1 vs REV1-as-opponent: a near-mirror, inflated by construction",
        },
        "exposure": _exposure_bound(),
        "comparability": "NOT COMPARABLE AS LEVELS: different reference opponents (different "
        "architectures, so no common model reference is even constructible), different opponent "
        "team-draw regimes, different eras.",
    }

    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(f"wrote {out.with_suffix('.json')}")
    _print_summary(report)
    return report


def _fmt(d, scale=1.0, unit=""):
    return f"{d['point']*scale:+.4f}{unit} [{d['lo']*scale:+.4f}, {d['hi']*scale:+.4f}]"


def _print_summary(rep):
    for name, r in rep["folds"].items():
        print(f"\n=== {name}  (n={r['n_teams']} teams) ===")
        print(
            f"  parent WR {r['parent_wr_mean']:.4f} (sd {r['parent_wr_sd']:.4f})  "
            f"gift {r['gift_pp_mean']:+.2f}pp (sd {r['gift_pp_sd']:.2f})"
        )
        print(f"  NAIVE   raw   r = {_fmt(r['naive']['raw_pp_vs_wr'])}")
        print(f"  NAIVE   logit r = {_fmt(r['naive']['logit'])}")
        print(f"  SPLIT   raw   r = {_fmt(r['split_half']['raw_pp_vs_wr'])}")
        print(f"  SPLIT   logit r = {_fmt(r['split_half']['logit'])}")
        print(f"  SPLIT   slope   = {_fmt(r['split_half']['logit_slope'])}")
        for C, d in r["split_half"]["headroom"].items():
            print(
                f"  SPLIT   headroom C={C} r = {_fmt(d)}  "
                f"(floor rate {r['split_half']['headroom_floor_rate'][C]:.3f})"
            )
        n = r["noise"]
        print(
            f"  noise: parent-WR measurement sd {n['sd_parent_wr_measurement_full_pp']:.2f}pp vs "
            f"across-team sd {n['sd_parent_wr_across_teams_pp']:.2f}pp "
            f"({100*n['measurement_share_of_x_variance']:.1f}% of x variance); "
            f"naive-minus-split raw shift {n['observed_naive_minus_splithalf_raw']:+.4f}"
        )
        nc = r["null_calibration"]
        print(
            "  NULL(log-odds-additive, Delta=%.3f): r_headroom = %s"
            % (
                nc["logodds_additive_null"]["delta_logit"],
                " ".join(
                    f"C{k}:{v:+.3f}" for k, v in nc["logodds_additive_null"]["r_headroom"].items()
                ),
            )
        )
        print(
            "  NULL(headroom-additive):            r_logit    = %s"
            % " ".join(
                f"C{k}:{v['r_logit']:+.3f}" for k, v in nc["headroom_additive_null"].items()
            )
        )
        if "rivals" in r:
            for k, d in r["rivals"]["univariate"].items():
                print(f"  RIVAL  {k:34s} r = {_fmt(d)}")
            for k, d in r["rivals"]["joint_ols"].items():
                print(
                    f"  JOINT  {k:28s} beta_comp {d['beta_prior_competence']:+.3f} "
                    f"beta_prox {d['beta_proximity']:+.3f}  R2 {d['r2']:.3f}"
                )
    print("\n=== between-era ===")
    print(json.dumps(rep["between_era"], indent=1))


if __name__ == "__main__":
    main()
