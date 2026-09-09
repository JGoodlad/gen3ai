"""THE PROBE READ — can a LINEAR decoder recover WHO we are playing, and WHOSE TEAM we hold,
from the raw observation / the shared trunk / the value path — and does V rank opponents at all?

The mixture diagnostic (`winprob_mixture_diagnostic_2026-09-09`) established that arm A's
win-prob critic barely separates opponents (turn-1–3 between-opponent spread of V is 0.334× the
outcome's) and that the trainee's OWN team explains ~9× more of the critic's residual than the
opponent does. This measurement decides the MECHANISM, and therefore the treatment:

  (i)  RAW decodes it, TRUNK/VALUE do not  -> the network DISCARDS the signal
                                              => an auxiliary decoding loss on the value path
  (ii) NOTHING decodes it early, all improve with the turn
                                           -> the signal is not observable early
                                              => a conditioning INPUT (Elo scalar + class bit)
  (iii) The VALUE path DOES decode it and V still does not separate
                                           -> the head has it and does not use it
                                              => target / capacity levers

FEATURE SETS (all from ONE frozen forward — see extract.py)
  raw     [2501]  the observation vector as recorded
  pi      [512]   the policy/shared-trunk head input  (extract_features[0])
  vf      [512]   the value-path head input           (extract_features[1])
  pooled  [128]   `stash.value_pooled` — the win head's LITERAL input; `V = sigmoid(head(pooled))`
  V       [1]     the critic's own output — "does V rank opponents?"

RULES OF EVIDENCE, enforced here rather than described:
  * CV folds are GROUPED BY BATTLE. States inside a battle share the team, the opponent and the
    outcome, so a battle split across folds would decode itself.
  * Every interval is a BATTLE-CLUSTERED bootstrap over the held-out predictions.
  * Every cell carries a PERMUTATION NULL run through the identical pipeline (penalty selection
    included), so selection optimism appears in the null too and a chance-level decode is labelled
    as such. Battle-level permutation for the opponent targets; for the OWN-TEAM targets a
    battle-level shuffle destroys the label's within-team constancy and is therefore too easy to
    beat, so a TEAM-level shuffle is also run and the STRICTER of the two is what a DETECTED must
    clear.
  * Every DETECTED comparison is on the DELTA's own CI (paired on the same battle draw), never on
    two overlapping bars.
  * Every fit and every score is Horvitz-Thompson weighted by the manifest capture rates, so the
    number is about the eval POPULATION and not about the loss-enriched captured slice.

The decoder is a WEIGHTED RIDGE throughout; a binary target is scored by weighted AUC of the
ridge prediction (monotone-equivalent to the linear discriminant, and it shares the closed form
that makes the permutation nulls affordable). `mlp_probe.py` carries the non-linear check.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

BUCKETS = {"t1": (1, 1), "t1_3": (1, 3), "t4_10": (4, 10),
           "t11_24": (11, 24), "t25p": (25, 10 ** 9)}
FSETS = ("raw", "pi", "vf", "pooled", "V")
LAMBDAS = np.logspace(-1, 9, 21)
MIN_TEAM_BATTLES_WR = 4
MIN_TEAM_BATTLES_ID = 8


# ──────────────────────────────────────────────────────────────────────────────
# Weighted ridge with a reusable spectral factorisation.
#
# The factorisation depends on X and the weights ONLY — never on y — so ONE decomposition per
# (feature set, fold) serves every target and every permutation repeat. That is what makes a
# 40-repeat null over five feature sets affordable on a CPU.
# ──────────────────────────────────────────────────────────────────────────────
class _FoldSolver:
    # 🚨 A NEAR-CONSTANT COLUMN, STANDARDISED, IS PURE AMPLIFIED NOISE. Dividing a column whose
    # training sd is 1e-7 by that sd turns rounding noise into a unit-variance feature, and 2,501
    # of them turn a ridge into an overfitting machine no penalty can rescue: the first revision
    # of this script read raw-obs R² of **-1.05** on `own_team_wr` and vf R² of **-0.82** on
    # `opp_elo` — negative out-of-fold R², i.e. worse than predicting the mean, from a feature
    # set that visibly contains the answer. Such columns are DROPPED (zeroed after centring), not
    # rescaled. The rule is relative so it is scale-free across obs blocks (embedding ids are
    # O(100), HP fractions O(1)).
    SD_FLOOR = 1e-4
    # 🚨 AND A RARE-VALUE COLUMN IS AN OUTLIER FACTORY. A column that is 0.999 constant and
    # varies on two rows survives the sd floor, and standardising it puts those two rows at ±30;
    # in the d>n dual regime the ridge then extrapolates off them and ONE test row destroys the
    # score. The second revision of this script read raw-obs R² of **-1221** on `own_team_wr`
    # (turns 4–10) and -0.11 on `opp_elo` (turns 1–3) for exactly that reason, with the inner CV
    # unable to see it because the offending row was not in any inner fold. Standardised values
    # are therefore CLIPPED to ±CLIP on both sides of the split. This bounds a single column's
    # leverage; it does not remove the information (the rank order inside the clip is intact),
    # and it is applied identically to every feature set so no comparison is tilted by it.
    CLIP = 8.0
    #: A column must vary on at least this many TRAINING rows to be used at all. This is the
    #: primary fix and the clip is the belt behind it: an obs channel that is all-zero in the
    #: training fold and non-zero in the test fold (a one-hot for a species this fold never saw)
    #: has sd 0 in train, and a channel that fires on two rows has a sd so small that those two
    #: rows standardise to ±30. Both are decoder poison in the d>n dual regime, and a variance
    #: floor alone does not catch the second. Counting the rows that differ from the column's
    #: median does.
    MIN_VARYING_ROWS = 5

    def __init__(self, Xtr, wtr, Xte):
        mu = np.average(Xtr, axis=0, weights=wtr)
        sd = np.sqrt(np.average((Xtr - mu) ** 2, axis=0, weights=wtr))
        med = np.median(Xtr, axis=0)
        n_vary = (np.abs(Xtr - med) > 1e-9).sum(axis=0)
        live = (sd > self.SD_FLOOR * (1.0 + np.abs(mu))) & (n_vary >= self.MIN_VARYING_ROWS)
        sd = np.where(live, sd, 1.0)
        Z = np.clip(np.where(live, (Xtr - mu) / sd, 0.0), -self.CLIP, self.CLIP)
        self.Zte = np.clip(np.where(live, (Xte - mu) / sd, 0.0), -self.CLIP, self.CLIP)
        self.n_live = int(live.sum())
        self.w = wtr / wtr.mean()
        s = np.sqrt(self.w)[:, None]
        Zs = Z * s                                   # weighted design
        n, d = Zs.shape
        self.dual = d > n
        if self.dual:
            K = Zs @ Zs.T
            ev, Q = np.linalg.eigh(K)
            self.ev, self.Q = np.clip(ev, 0, None), Q
            self.C = self.Zte @ Zs.T                 # [n_te, n_tr]
        else:
            A = Zs.T @ Zs
            ev, Q = np.linalg.eigh(A)
            self.ev, self.Q = np.clip(ev, 0, None), Q
            self.Zs = Zs

    def predict(self, ytr, lams):
        """Held-out predictions for every lambda in `lams` -> [len(lams), n_te]."""
        wm = np.average(ytr, weights=self.w)
        yc = (ytr - wm) * np.sqrt(self.w)
        if self.dual:
            c = self.Q.T @ yc
            out = np.stack([self.C @ (self.Q @ (c / (self.ev + lam))) for lam in lams])
        else:
            b = self.Zs.T @ yc
            c = self.Q.T @ b
            out = np.stack([self.Zte @ (self.Q @ (c / (self.ev + lam))) for lam in lams])
        return out + wm


def _fold_map(groups, k, seed):
    """Assign each GROUP (battle) to one of k folds — a seeded permutation, never a hash."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    gf = {g: int(perm[i] % k) for i, g in enumerate(uniq)}
    return np.array([gf[g] for g in groups])


class GroupedRidgeCV:
    """Outer grouped K-fold with an INNER grouped K-fold that picks the penalty. Both split on
    battles, so no battle is ever in train and test at once."""

    def __init__(self, X, w, groups, seed=0, k=5, k_inner=4):
        self.n = len(X)
        self.folds = _fold_map(groups, k, seed)
        self.k = k
        self.outer, self.inner = [], []
        for f in range(k):
            te = np.where(self.folds == f)[0]
            tr = np.where(self.folds != f)[0]
            if len(te) == 0 or len(tr) < 10:
                self.outer.append(None)
                self.inner.append([])
                continue
            self.outer.append((tr, te, _FoldSolver(X[tr], w[tr], X[te])))
            gin = _fold_map(groups[tr], k_inner, seed + 101 + f)
            ins = []
            for g in range(k_inner):
                ite = np.where(gin == g)[0]
                itr = np.where(gin != g)[0]
                if len(ite) == 0 or len(itr) < 10:
                    continue
                ins.append((tr[itr], tr[ite], _FoldSolver(X[tr][itr], w[tr][itr], X[tr][ite])))
            self.inner.append(ins)

    def oof(self, y, w, task):
        """Out-of-fold predictions; the penalty is chosen per OUTER fold on its own inner CV."""
        pred = np.full(self.n, np.nan)
        lams_used = []
        for f in range(self.k):
            if self.outer[f] is None:
                continue
            best, best_lam = -np.inf, LAMBDAS[len(LAMBDAS) // 2]
            if self.inner[f]:
                scores = np.zeros(len(LAMBDAS))
                for itr, ite, sol in self.inner[f]:
                    P = sol.predict(y[itr], LAMBDAS)
                    for li in range(len(LAMBDAS)):
                        scores[li] += _score(y[ite], P[li], w[ite], task) * len(ite)
                if np.isfinite(scores).any():
                    best = np.nanmax(scores)
                    best_lam = LAMBDAS[int(np.nanargmax(scores))]
            tr, te, sol = self.outer[f]
            pred[te] = sol.predict(y[tr], np.array([best_lam]))[0]
            lams_used.append(float(best_lam))
        self.last_lams = lams_used
        return pred, lams_used


# ──────────────────────────────────────────────────────────────────────────────
# Weighted scores
# ──────────────────────────────────────────────────────────────────────────────
def _w_r2(y, p, w):
    m = np.average(y, weights=w)
    den = float(np.sum(w * (y - m) ** 2))
    return float(1.0 - np.sum(w * (y - p) ** 2) / den) if den > 0 else np.nan


def _w_auc(y, p, w):
    """Weighted Mann-Whitney AUC, ties at 0.5. Fully vectorised: the one-vs-rest bootstrap
    evaluates this ~70,000 times, and a per-row python tie loop made that intractable."""
    pos = y > 0.5
    if not pos.any() or pos.all():
        return np.nan
    _, grp = np.unique(p, return_inverse=True)
    ng = grp.max() + 1
    neg_w = np.bincount(grp, weights=np.where(pos, 0.0, w), minlength=ng)
    pos_w = np.bincount(grp, weights=np.where(pos, w, 0.0), minlength=ng)
    below = np.concatenate([[0.0], np.cumsum(neg_w)])[:-1]     # groups are unique-sorted by p
    tot = float(np.sum(pos_w * (below + 0.5 * neg_w)))
    W_pos, W_neg = float(pos_w.sum()), float(neg_w.sum())
    return float(tot / (W_pos * W_neg)) if W_pos > 0 and W_neg > 0 else np.nan


def _score(y, p, w, task):
    ok = ~np.isnan(p)
    if ok.sum() < 5:
        return np.nan
    return _w_auc(y[ok], p[ok], w[ok]) if task == "auc" else _w_r2(y[ok], p[ok], w[ok])


# ──────────────────────────────────────────────────────────────────────────────
# Targets
# ──────────────────────────────────────────────────────────────────────────────
def build_targets(meta):
    """name -> dict(kind, task, y | ovr_labels, mask, note)."""
    battles, binv = np.unique(meta["battle"], return_inverse=True)
    b_y = np.zeros(len(battles))
    b_w = np.zeros(len(battles))
    b_team = np.empty(len(battles), dtype=meta["team"].dtype)
    for i, b in enumerate(battles):
        s = np.where(binv == i)[0][0]
        b_y[i], b_w[i], b_team[i] = meta["y"][s], meta["w"][s], meta["team"][s]

    out = {}
    out["opp_elo"] = {"task": "r2", "y": meta["strength"].astype(float),
                      "mask": np.ones(len(meta), bool), "null_kind": "label",
                      "group": meta["opponent"],
                      "note": "opponent rating on the bot-anchored scale (battle-level constant)"}
    out["opp_class"] = {"task": "auc", "y": (meta["opp_class"] == "sentinel").astype(float),
                        "mask": np.ones(len(meta), bool), "null_kind": "label",
                        "group": meta["opponent"],
                        "note": "opponent CLASS: self-play snapshot (1) vs scripted bot (0)"}

    # own team's expected win rate, LEAVE-ONE-BATTLE-OUT and IPW-weighted, so the label cannot
    # contain the outcome of the battle it labels.
    teams, tinv = np.unique(b_team, return_inverse=True)
    tw = np.bincount(tinv, weights=b_w, minlength=len(teams))
    twy = np.bincount(tinv, weights=b_w * b_y, minlength=len(teams))
    tn = np.bincount(tinv, minlength=len(teams))
    loo = np.full(len(battles), np.nan)
    for i in range(len(battles)):
        t = tinv[i]
        den = tw[t] - b_w[i]
        if tn[t] >= MIN_TEAM_BATTLES_WR and den > 0:
            loo[i] = (twy[t] - b_w[i] * b_y[i]) / den
    y_wr = loo[binv]
    out["own_team_wr"] = {"task": "r2", "y": np.nan_to_num(y_wr), "mask": ~np.isnan(y_wr),
                          "null_kind": "team_assign", "group": meta["team"],
                          "note": (f"the trainee team's LEAVE-ONE-BATTLE-OUT IPW win rate, teams "
                                   f"with >={MIN_TEAM_BATTLES_WR} battles")}

    # own team IDENTITY: one-vs-rest over every team with enough battles, scored macro-AUC.
    keep_t = [teams[i] for i in range(len(teams)) if tn[i] >= MIN_TEAM_BATTLES_ID]
    out["own_team_id"] = {"task": "auc", "ovr": keep_t, "mask": np.ones(len(meta), bool),
                          "null_kind": "team_assign", "group": None,
                          "note": (f"own-team IDENTITY, one-vs-rest macro AUC over the "
                                   f"{len(keep_t)} teams with >={MIN_TEAM_BATTLES_ID} battles")}
    return out, battles, binv, b_team


def first_of_group(inv, n_groups):
    """Index of the first member of each group — vectorised (this runs inside every
    permutation repeat; a per-group `np.where` made the null loop the bottleneck)."""
    first = np.zeros(n_groups, dtype=int)
    order = np.arange(len(inv))[::-1]
    first[inv[::-1]] = order
    return first


def _loo_team_wr(b_team, b_y, b_w, min_battles):
    """Per-battle LEAVE-ONE-OUT IPW win rate of that battle's team; NaN under `min_battles`."""
    teams, tinv = np.unique(b_team, return_inverse=True)
    tw = np.bincount(tinv, weights=b_w, minlength=len(teams))
    twy = np.bincount(tinv, weights=b_w * b_y, minlength=len(teams))
    tn = np.bincount(tinv, minlength=len(teams)).astype(float)
    den = tw[tinv] - b_w
    ok = (tn[tinv] >= min_battles) & (den > 0)
    return np.where(ok, (twy[tinv] - b_w * b_y) / np.where(den > 0, den, 1.0), np.nan)


def shuffle_team_assignment(b_team, rng):
    """Permute WHICH BATTLE HOLDS WHICH TEAM. Every other column of every state is untouched, and
    the multiset of team sizes is preserved exactly, so the permuted problem is the same shape as
    the real one. This is the null for both own-team targets: `own_team_id`'s one-vs-rest labels
    are rebuilt from the shuffled assignment, and `own_team_wr`'s leave-one-out IPW win rates are
    RECOMPUTED from it.

    🚨 The first revision got both wrong, in opposite directions. For `own_team_id` it permuted the
    LIST of one-vs-rest label vectors, which is a re-ordering of the same set — so the macro AUC was
    identical to the observed score and the null was vacuous by construction. For `own_team_wr` it
    permuted the TEAM->win-rate map while keeping the label constant within the real team; a decoder
    that identifies the team then recovers any per-team labelling, so the "null" read 0.98 against a
    real score of 0.83 and convicted a genuine decode of being chance. That second quantity is worth
    having, but as the IDENTITY-MEDIATION reference below — never as the chance level.
    """
    return b_team[rng.permutation(len(b_team))]


def _perm_labels(y_state, binv, kind, grp_of_battle, rng):
    """Two DIFFERENT nulls, and conflating them was a real defect in this script's first run.

    ``kind="battle"`` — shuffle the label across BATTLES. This is the CHANCE level: it destroys
    every association between the state and the label, so a score above it is a real decode. Every
    DETECTED in this measurement is against this null.

    ``kind="group"`` — shuffle the label across the GROUPS the label is constant within (the
    opponent for `opp_elo`/`opp_class`, the trainee team for `own_team_wr`), keeping it constant
    within a group. This is NOT a chance level and must never be read as one: a decoder that
    identifies the group can recover ANY per-group labelling, so a perfect group-identifier scores
    ~1.0 here. It answers the separate, sharper question *does the representation carry the LABEL
    beyond bare GROUP IDENTITY* — and a score BELOW it says the decode is entirely identity-mediated.
    The first revision used this as the null for `own_team_wr` and read a "null" of 0.98 against a
    score of 0.83, i.e. it convicted a real decode of being chance.
    """
    nb = int(binv.max()) + 1
    b_lab = y_state[first_of_group(binv, nb)]
    if kind == "battle" or grp_of_battle is None:
        return b_lab[rng.permutation(nb)][binv]
    grps, ginv = np.unique(grp_of_battle, return_inverse=True)
    g_lab = b_lab[first_of_group(ginv, len(grps))]
    return g_lab[rng.permutation(len(grps))][ginv][binv]


# ──────────────────────────────────────────────────────────────────────────────
def cap_per_battle(meta, mask, cap, seed):
    """At most `cap` states per battle inside the bucket — keeps every battle represented while
    holding n (and therefore the eigendecompositions) affordable, and limits how much of the
    within-battle correlation rides into the fit."""
    rng = np.random.default_rng(seed)
    idx = np.where(mask)[0]
    out = []
    for b in np.unique(meta["battle"][idx]):
        sel = idx[meta["battle"][idx] == b]
        out.append(sel if len(sel) <= cap else rng.choice(sel, cap, replace=False))
    return np.sort(np.concatenate(out)) if out else np.array([], int)


def boot_ci(y, preds, w, groups, task, B, seed, extra=None):
    """Battle-clustered percentile CI for each feature set's score AND for the paired deltas.
    The bootstrap resamples the EVALUATION sample (the fitted decoders are held fixed) — a
    standard paired-OOF bootstrap; it prices sampling noise in the held-out score, not the
    variability of refitting."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    keys = list(preds)
    draws = {k: [] for k in keys}
    dkeys = extra or []
    ddraws = {f"{a}-{b}": [] for a, b in dkeys}
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([members[i] for i in pick])
        for k in keys:
            draws[k].append(_score(y[sel], preds[k][sel], w[sel], task))
        for a, b in dkeys:
            ddraws[f"{a}-{b}"].append(_score(y[sel], preds[a][sel], w[sel], task)
                                      - _score(y[sel], preds[b][sel], w[sel], task))

    def q(v):
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        return [None, None] if len(v) < 10 else [round(float(np.percentile(v, 2.5)), 4),
                                                 round(float(np.percentile(v, 97.5)), 4)]
    return {k: q(v) for k, v in draws.items()}, {k: q(v) for k, v in ddraws.items()}


def run_cell(feats, meta, idx, tgt, name, seed, n_perm, n_boot):
    y_full, task = tgt.get("y"), tgt["task"]
    w = meta["w"][idx].astype(float)
    groups = meta["battle"][idx]
    cv = {k: GroupedRidgeCV(feats[k][idx].astype(np.float64), w, groups, seed=seed)
          for k in FSETS}
    _, binv = np.unique(groups, return_inverse=True)
    b_team = meta["team"][idx][first_of_group(binv, int(binv.max()) + 1)]

    def fit_all(yv):
        return {k: cv[k].oof(yv, w, task)[0] for k in FSETS}

    if "ovr" in tgt:                                   # macro one-vs-rest
        labs = [(meta["team"][idx] == t).astype(float) for t in tgt["ovr"]]
        labs = [ly for ly in labs if 0 < ly.sum() < len(ly)]
        if not labs:
            return None
        preds = {k: [] for k in FSETS}
        for ly in labs:
            p = fit_all(ly)
            for k in FSETS:
                preds[k].append(p[k])
        pt = {k: float(np.nanmean([_score(ly, preds[k][i], w, task)
                                   for i, ly in enumerate(labs)])) for k in FSETS}
        # bootstrap the MACRO score over battles
        rng = np.random.default_rng(seed + 7)
        uq, inv2 = np.unique(groups, return_inverse=True)
        members = [np.where(inv2 == i)[0] for i in range(len(uq))]
        cis = {k: [] for k in FSETS}
        dd = {f"{a}-{b}": [] for a, b in (("pi", "raw"), ("vf", "raw"), ("vf", "pi"),
                                          ("pooled", "raw"), ("pooled", "vf"), ("V", "pooled"))}
        for _ in range(min(n_boot, 600)):
            pick = rng.integers(0, len(uq), len(uq))
            sel = np.concatenate([members[i] for i in pick])
            sc = {k: float(np.nanmean([_score(ly[sel], preds[k][i][sel], w[sel], task)
                                       for i, ly in enumerate(labs)])) for k in FSETS}
            for k in FSETS:
                cis[k].append(sc[k])
            for a, b in (("pi", "raw"), ("vf", "raw"), ("vf", "pi"),
                         ("pooled", "raw"), ("pooled", "vf"), ("V", "pooled")):
                dd[f"{a}-{b}"].append(sc[a] - sc[b])

        def q(v):
            v = np.asarray(v, float)
            v = v[~np.isnan(v)]
            return [None, None] if len(v) < 10 else [round(float(np.percentile(v, 2.5)), 4),
                                                     round(float(np.percentile(v, 97.5)), 4)]
        ci, dci = {k: q(v) for k, v in cis.items()}, {k: q(v) for k, v in dd.items()}
        nulls = {k: [] for k in FSETS}
        rngp = np.random.default_rng(seed + 991)
        for _ in range(max(4, n_perm // 5)):
            pteam = shuffle_team_assignment(b_team, rngp)[binv]
            plabs = [(pteam == t).astype(float) for t in tgt["ovr"]]
            plabs = [ly for ly in plabs if 0 < ly.sum() < len(ly)]
            pp = [fit_all(ly) for ly in plabs]
            for k in FSETS:
                nulls[k].append(float(np.nanmean(
                    [_score(ly, pp[i][k], w, task) for i, ly in enumerate(plabs)])))
        nulls2 = None
    else:
        yv = y_full[idx].astype(float)
        preds = fit_all(yv)
        pt = {k: _score(yv, preds[k], w, task) for k in FSETS}
        ci, dci = boot_ci(yv, preds, w, groups, task, n_boot, seed + 7,
                          extra=[("pi", "raw"), ("vf", "raw"), ("vf", "pi"),
                                 ("pooled", "raw"), ("pooled", "vf"), ("V", "pooled")])
        nulls = {k: [] for k in FSETS}
        rngp = np.random.default_rng(seed + 991)
        b_y = meta["y"][idx][first_of_group(binv, len(b_team))]
        b_w = meta["w"][idx][first_of_group(binv, len(b_team))].astype(float)
        for _ in range(n_perm):
            if tgt["null_kind"] == "team_assign":
                pt_team = shuffle_team_assignment(b_team, rngp)
                lab_b = _loo_team_wr(pt_team, b_y, b_w, MIN_TEAM_BATTLES_WR)
                yp = np.nan_to_num(lab_b, nan=float(np.nanmean(lab_b)))[binv]
            else:
                yp = _perm_labels(yv, binv, "battle", None, rngp)
            pn = fit_all(yp)
            for k in FSETS:
                nulls[k].append(_score(yp, pn[k], w, task))
        # the IDENTITY-MEDIATION reference (not a chance level — see shuffle_team_assignment)
        nulls2 = {k: [] for k in FSETS}
        rng2 = np.random.default_rng(seed + 1993)
        for _ in range(max(4, n_perm // 4)):
            yp = _perm_labels(yv, binv, "group", tgt["group"][idx][first_of_group(
                binv, len(b_team))] if tgt["group"] is not None else None, rng2)
            pn = fit_all(yp)
            for k in FSETS:
                nulls2[k].append(_score(yp, pn[k], w, task))

    null_stat = {k: {"mean": round(float(np.nanmean(v)), 4),
                     "p95": round(float(np.nanpercentile(v, 95)), 4),
                     "n": len(v)} for k, v in nulls.items()}
    return {"target": name, "task": task, "note": tgt["note"],
            "n_states": int(len(idx)), "n_battles": int(len(np.unique(groups))),
            "score": {k: (None if pt[k] is None or np.isnan(pt[k]) else round(float(pt[k]), 4))
                      for k in FSETS},
            "ci": ci, "null": null_stat, "delta_ci": dci,
            "null_identity_mediated": (None if nulls2 is None else
                                       {k: {"mean": round(float(np.nanmean(v)), 4),
                                            "p95": round(float(np.nanpercentile(v, 95)), 4),
                                            "n": len(v)} for k, v in nulls2.items()}),
            "detected": {k: bool(pt[k] is not None and not np.isnan(pt[k])
                                 and ci[k][0] is not None
                                 and ci[k][0] > null_stat[k]["p95"]) for k in FSETS}}


def main(a):
    feats = {"raw": np.load(f"{a.dir}/raw.npy", mmap_mode="r"),
             "pi": np.load(f"{a.dir}/pi.npy", mmap_mode="r"),
             "vf": np.load(f"{a.dir}/vf.npy", mmap_mode="r"),
             "pooled": np.load(f"{a.dir}/pooled.npy", mmap_mode="r")}
    meta = np.load(f"{a.dir}/meta.npy")
    feats["V"] = meta["V_fwd"].astype(np.float32)[:, None]
    tgts, _b, _bi, _bt = build_targets(meta)
    out = {"dir": a.dir, "seed": a.seed, "cap_per_battle": a.cap,
           "n_states_total": int(len(meta)),
           "n_battles_total": int(len(np.unique(meta["battle"]))), "cells": []}
    for bname, (lo, hi) in BUCKETS.items():
        inb = (meta["turn"] >= lo) & (meta["turn"] <= hi)
        for tname, tgt in tgts.items():
            idx = cap_per_battle(meta, inb & tgt["mask"], a.cap, a.seed)
            if len(idx) < 60 or len(np.unique(meta["battle"][idx])) < 40:
                out["cells"].append({"bucket": bname, "target": tname,
                                     "skipped": "too few battles",
                                     "n_states": int(len(idx))})
                continue
            X = {k: np.asarray(feats[k]) for k in FSETS}
            cell = run_cell(X, meta, idx, tgt, tname, a.seed, a.perm, a.boot)
            if cell is None:
                continue
            cell["bucket"] = bname
            out["cells"].append(cell)
            print(f"{bname:>6} {tname:<13} n={cell['n_states']:>5} b={cell['n_battles']:>4} "
                  + " ".join(f"{k}={cell['score'][k]}" for k in FSETS), flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="an extract.py output directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--cap", type=int, default=2, help="max states kept per battle per bucket")
    ap.add_argument("--perm", type=int, default=40)
    ap.add_argument("--boot", type=int, default=2000)
    main(ap.parse_args())
