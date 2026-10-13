"""Shared loader and bootstrap helpers of the by-reveal-count / belief-dose reads (this folder's README).

PURE numpy over the pilot's archive (`~/gen3ai_archive/value_disc_pilot/<cp>/rows` = Phase 1, the TRUE world; `det2*` =
the FIXED (v2) builder's honest worlds), through the pilot's own read-only scorers (`score.state_arrays`,
`score_det.det_arrays`, `score.pick`, `score.Boot`). No model, no playout. `load_bundle` caches each checkpoint's
per-state, per-action, per-world outcome matrices (a few MB) under `~/gen3ai_archive/hidden_info_by_reveal_count/`
so a second read does not re-parse the 400 MB of rows.

A BUNDLE per state (all on the pilot's win scale [0, 1]):
  acts, order (the policy's order over action INDICES), top1, T (the 24-seed referee truth per action)
  WT   [A, 16]  true-world leaf outcomes (dice seeds 0-15; seeds 16-39 are the referee's and never enter a scorer)
  X4T  [A, 16]  true-world value at the searched side's first decision >= 4 turns on (TRUNC-4), same seeds
  critT [A]     true-world one-ply critic (the mean of 8 successors);  critTs [A, 8] the 8 successors' own values
  arms[a] for a in A (Smogon prior), B (belief species), C (network move belief), CT (C + the seen mons' TRUE sets):
    W [A, 8]  honest world j's outcome under dice seed j (ONE playout per world),  X4 [A, 8],  crit [A, 8] per world
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent
PILOT = MEAS / "value_disc_pilot_2026-10-11"
HIB = MEAS / "hidden_info_binding_2026-10-11"
sys.path.insert(0, str(PILOT))
import score as S1  # noqa: E402
import score_det as SD  # noqa: E402

CPS = ("end2", "prod")
ARMS = (("A", "det2"), ("B", "det2_b"), ("C", "det2_c"), ("CT", "det2_ct"))
CACHE = Path.home() / "gen3ai_archive" / "hidden_info_by_reveal_count"
NAMES = ("LEAF-K8", "CRITIC", "TRUNC-4-K8", "LEAF-K4", "TRUNC-4-K4")
TIE = S1.TIE
w = S1.w


def build_bundle(cp):
    st, r1, _missing, _ = S1.load(cp, True)
    E = st["states"]
    arms = {}
    for a, d in ARMS:
        SD.DET_DIR = d
        arms[a] = SD.det_rows(cp)
    facts = json.loads((HIB / f"root_facts_{cp}.json").read_text())["states"]
    stage = json.loads((HERE / f"stage_facts_{cp}.json").read_text())["states"]
    ids = sorted(i for i in E if r1.get(i, {}).get("ok") and all(arms[a].get(i, {}).get("ok") for a, _ in ARMS))
    out = {}
    for did in ids:
        e = E[did]
        So = S1.state_arrays(r1[did], e)
        b = {"id": did, "cp": cp, "battle": f"{cp}:{e['battle']}", "cycle_opp": e["cycle_opp"], "outcome": e["outcome"],
             "k": 6 - int(e["unrevealed"]), "k_check": int(facts[did]["n_seen"]), "turn": int(e["turn"]),
             "gap": float(e["gap"]), "quintile": int(e["quintile"]), "n_legal": len(So["acts"]),
             "alive": int(stage[did]["alive_ours"] + stage[did]["alive_theirs"]),
             "alive_ours": int(stage[did]["alive_ours"]), "alive_theirs": int(stage[did]["alive_theirs"]),
             "hp_mat": float((stage[did]["hp_frac_ours"] + stage[did]["hp_frac_theirs"]) / 2),
             "acts": [int(x) for x in So["acts"]], "order": [int(x) for x in So["order"]], "top1": int(So["top1"]),
             "T": So["truth"].astype(np.float64), "WT": So["W"][:, : S1.N_LEAF].astype(np.float64),
             "X4T": So["Xd"][4][:, : S1.N_LEAF].astype(np.float64), "critT": So["crit"].astype(np.float64),
             "critTs": w(np.array([r1[did]["crit"][str(act)]["q"] for act in So["acts"]], dtype=np.float64)),
             "arms": {}}
        for a, _d in ARMS:
            rd = arms[a][did]
            _S, Sd = SD.det_arrays(r1[did], rd, e)
            qs = np.array([[x for x in rd["crit"][str(act)]["q"]] for act in b["acts"]], dtype=np.float64)
            b["arms"][a] = {"W": Sd["W"].astype(np.float64), "X4": Sd["Xd"][4].astype(np.float64), "crit": w(qs)}
            if abs(float(b["arms"][a]["crit"].mean(1)[0]) - float(Sd["crit"][0])) > 1e-9:
                raise RuntimeError(f"{did}: per-world critic mean != the pilot's honest critic")
        out[did] = b
    return st, out


def load_bundle(cp, rebuild=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"bundle_{cp}.pkl"
    if f.exists() and not rebuild:
        with f.open("rb") as fh:
            return pickle.load(fh)
    st, out = build_bundle(cp)
    with f.open("wb") as fh:
        pickle.dump((st, out), fh, protocol=4)
    return st, out


def pick(score, order):
    return S1.pick(score, order)


def scorer_vectors(b, who):
    """{name: per-action score} for 'O' (the true world) or an honest arm code, from a bundle."""
    if who == "O":
        W, X4, crit = b["WT"], b["X4T"], b["critT"]
        return {"LEAF-K8": W[:, :8].mean(1), "LEAF-K4": W[:, :4].mean(1), "TRUNC-4-K8": X4[:, :8].mean(1),
                "TRUNC-4-K4": X4[:, :4].mean(1), "CRITIC": crit}
    g = b["arms"][who]
    return {"LEAF-K8": g["W"][:, :8].mean(1), "LEAF-K4": g["W"][:, :4].mean(1), "TRUNC-4-K8": g["X4"][:, :8].mean(1),
            "TRUNC-4-K4": g["X4"][:, :4].mean(1), "CRITIC": g["crit"].mean(1)}


def load_records(cp):
    """The per-state records of by_reveal_count.py: IMP and the picked action index of every world kind x scorer."""
    st, bund = load_bundle(cp)
    recs = []
    for did in sorted(bund):
        b = bund[did]
        T, t1, order = b["T"], b["top1"], b["order"]
        rec = {k: b[k] for k in ("id", "cp", "battle", "cycle_opp", "outcome", "k", "k_check", "turn", "gap",
                                 "quintile", "n_legal", "alive", "alive_ours", "alive_theirs", "hp_mat")}
        rec.update({"upper": float(T.max() - T[t1]), "spread": float(T.max() - T.mean()), "top1": t1,
                    "pick": {}, "imp": {}})
        for who in ("O", "A", "B", "C", "CT"):
            sc = scorer_vectors(b, who)
            for nme in NAMES:
                p = int(pick(sc[nme], order))
                rec["pick"][(who, nme)] = p
                rec["imp"][(who, nme)] = float(T[p] - T[t1])
        rec["regret_A"] = float(T.max() - T[rec["pick"][("A", "LEAF-K8")]])
        rec["gain_OA"] = float(T[rec["pick"][("O", "LEAF-K8")]] - T[rec["pick"][("A", "LEAF-K8")]])
        recs.append(rec)
    return st, recs


# ------------------------------------------------------------------------------------------------
# bootstrap helpers over the pilot's Boot (same draws, so every contrast is paired)
# ------------------------------------------------------------------------------------------------
class Bt:
    def __init__(self, battles, weights):
        self.b = S1.Boot(battles, weights)
        self.nb = len(self.b.ub)

    def _sum(self, x, m, wt=None):
        x = np.asarray(x, dtype=np.float64)
        wt = np.ones(len(x)) if wt is None else wt
        return np.bincount(self.b.bi[m], weights=(x * wt)[m], minlength=self.nb)

    @staticmethod
    def _ci(v):
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        if len(v) < 20:
            return [None, None]
        lo, hi = np.percentile(v, [2.5, 97.5])
        return [round(float(lo), 5), round(float(hi), 5)]

    def mean(self, x, sel=None, weighted=False):
        m = np.ones(len(x), bool) if sel is None else np.asarray(sel, bool)
        wt = self.b.wt if weighted else None
        S, N = self._sum(x, m, wt), self._sum(np.ones(len(x)), m, wt)
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = (self.b.C @ S) / (self.b.C @ N)
        return float(S.sum() / N.sum()), vec

    def stat(self, x, sel=None, weighted=False):
        est, vec = self.mean(x, sel, weighted)
        n = int(len(x) if sel is None else np.sum(sel))
        return {"mean": round(est, 5), "ci": self._ci(vec), "n": n}

    def ratio(self, num, den, sel=None):
        """ratio of sums on the same draws: (estimate, bootstrap vector)."""
        m = np.ones(len(num), bool) if sel is None else np.asarray(sel, bool)
        Sn, Sd = self._sum(num, m), self._sum(den, m)
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = (self.b.C @ Sn) / (self.b.C @ Sd)
        return float(Sn.sum() / Sd.sum()) if Sd.sum() != 0 else float("nan"), vec

    def diff(self, x, sel_a, sel_b):
        ea, va = self.mean(x, sel_a)
        eb, vb = self.mean(x, sel_b)
        return {"mean": round(ea - eb, 5), "ci": self._ci(va - vb), "n": [int(np.sum(sel_a)), int(np.sum(sel_b))]}

    def curve(self, X, sel=None):
        """X [n_states, n_grid] -> (means [n_grid], bootstrap matrix [B, n_grid]) of the per-grid means on one draw set."""
        m = np.ones(len(X), bool) if sel is None else np.asarray(sel, bool)
        S = np.stack([self._sum(X[:, g], m) for g in range(X.shape[1])], axis=1)      # [nb, G]
        N = self._sum(np.ones(len(X)), m)
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = (self.b.C @ S) / (self.b.C @ N)[:, None]
        return S.sum(0) / N.sum(), vec

    def ols(self, y, Z, sel=None):
        """OLS coefficients (intercept first) of y on the columns of Z, battle-clustered bootstrap CIs."""
        m = np.ones(len(y), bool) if sel is None else np.asarray(sel, bool)
        X = np.column_stack([np.ones(m.sum()), np.asarray(Z, dtype=np.float64)[m]])
        yy = np.asarray(y, dtype=np.float64)[m]
        bi = self.b.bi[m]
        p = X.shape[1]
        XtX = np.zeros((self.nb, p, p))
        Xty = np.zeros((self.nb, p))
        np.add.at(XtX, bi, X[:, :, None] * X[:, None, :])
        np.add.at(Xty, bi, X * yy[:, None])
        beta = np.linalg.solve(XtX.sum(0), Xty.sum(0))
        A = (self.b.C @ XtX.reshape(self.nb, p * p)).reshape(-1, p, p)
        bvec = self.b.C @ Xty
        ok = np.array([np.linalg.matrix_rank(a) == p for a in A])
        bs = np.linalg.solve(A[ok], bvec[ok][:, :, None])[:, :, 0]
        r2 = 1.0 - ((yy - X @ beta) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
        return {"coef": [round(float(v), 5) for v in beta],
                "ci": [[round(float(a), 5), round(float(b), 5)] for a, b in
                       (np.percentile(bs[:, i], [2.5, 97.5]) for i in range(p))],
                "r2": round(float(r2), 4), "n": int(m.sum())}
