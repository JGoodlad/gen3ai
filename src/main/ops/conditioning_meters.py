"""``main.ops.conditioning_meters`` — does the win-prob critic know WHO it is playing, and WHOSE
TEAM it is holding?

Promoted 2026-09-09 out of two committed measurement directories, where the same arithmetic lived
as session scripts:

* ``designs/research_state/measurements/winprob_mixture_diagnostic_2026-09-09/`` — the extraction
  (``extract.py``) and the between-opponent SPREAD IDENTITY and the bias-on-Elo slope
  (``analyze.py`` §5 / §4);
* ``designs/research_state/measurements/winprob_probe_read_2026-09-09/`` — the own-team
  LEAVE-ONE-BATTLE-OUT win-rate target, the battle-grouped folds and the weighted scores
  (``decode.py``).

Those copies STAY where they are — a committed measurement must remain reproducible from the
artifacts beside it — and this module is the version :mod:`main.ops.critic_read` imports, so every
ladder arm is read by identical code.

**THE IDENTITY THIS MEASURES.** For ANY calibrated critic ``E[V | opponent] == E[y | opponent]``
exactly, so the BETWEEN-OPPONENT spread of ``V`` must EQUAL the between-opponent spread of the
outcome. A head that emits one marginal win probability regardless of opponent has a ratio near
ZERO; a calibrated one has 1.0. The identity needs no strength axis, which is why it is the
primary row and the Elo slope is the secondary one.

**ONE CYCLE, RECORDED ``V``.** Every statistic here is computed on a SINGLE eval cycle, from the
``win_probs`` column the trace npz recorded. Within one cycle that column IS the cycle's own model,
so no model forward is needed and none is done. 🚨 The distinction that matters — and it cost the
head refit a wrong reading (``winprob_head_refit_2026-09-09`` §12 hazard 3) — is across CYCLES: a
RECORDED ``V`` pooled over several cycles carries the trainee's own improvement between cells and
reads as separation that the head does not have (CTRL's pooled recorded ratio is 1.079 against
0.066 for a frozen forward). Pooling cycles is exactly what this module does not do.

**SELECTION.** The eval trace quota PREFERS LOSSES, so every statistic is Horvitz-Thompson
reweighted by the cycle's own ``capture_rate_win`` / ``capture_rate_loss`` (RULE OF EVIDENCE 17).
A cycle whose manifest carries no ``selection`` block is SELECTION UNKNOWN and is REFUSED, never
silently read raw.

**INTERVALS.** Every interval is a battle-clustered bootstrap that resamples battles WITHIN each
opponent cell — the opponent roster is a fixed pinned set, not a sample — and the outcome side is
redrawn from its own Binomial(``battles_played``, true win rate) so the comparison prices the
100-game noise in the thing ``V`` is being compared against.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: turn windows the meters are reported on. ``t1`` is the sharpest (nothing has happened, every
#: battle contributes exactly one state, and the two sides of the identity condition on the same
#: event); ``t1_3`` is the mixture diagnostic's own registered window.
BUCKETS = ("t1", "t1_3", "all")
#: at most this many states per battle per bucket enter a SCORE meter (the probe read's cap) —
#: it keeps every battle represented while bounding how much within-battle correlation rides in.
STATES_PER_BATTLE_CAP = 2
#: a team needs this many battles before its leave-one-battle-out win rate is a label at all.
MIN_TEAM_BATTLES = 4
N_BOOT = 2000
BOOT_SEED = 20260909
RIDGE_ALPHAS = (0.0, 1e-3, 1e-2, 1e-1, 1.0)


class ConditioningRefusal(RuntimeError):
    """A precondition of the conditioning read is unmet. Never downgraded to a silent NaN."""


def recorded_v_note() -> str:
    return ("every conditioning row is computed on ONE eval cycle from the `win_probs` the trace "
            "npz RECORDED, which within a cycle IS that cycle's own model — no model forward is "
            "done. Pooling a recorded V across cycles would import the trainee's own improvement "
            "as between-opponent spread (head-refit hazard 3: CTRL reads 1.079 pooled against "
            "0.066 frozen); this module never pools cycles.")


# --------------------------------------------------------------------------- extraction

def _load_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def team_id(summary: dict) -> str:
    """A stable 10-hex id for the TRAINEE's team — the sorted species multiset, hashed."""
    sp = sorted(p.get("species", "?") for p in (summary.get("teams") or {}).get("ours") or [])
    return hashlib.sha1("|".join(sp).encode()).hexdigest()[:10] if sp else "unknown"


STATE_DTYPE = np.dtype([("opponent", "U32"), ("opp_class", "U8"), ("battle", "U80"),
                        ("y", "f8"), ("turn", "i8"), ("V", "f8"), ("w", "f8"),
                        ("team", "U12"), ("true_wr", "f8"), ("n_games", "f8")])


def extract_cycle(trace_dir: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """One eval cycle -> a tidy per-STATE table, plus what was refused and why.

    One row per traced decision point that has a state and a real turn:
    ``opponent, opp_class, battle, y, turn, V, w, team, true_wr, n_games``.

    ``V`` is the npz's ``win_probs`` (the QC below asserts it is the same column as ``values`` and
    reports the largest disagreement); ``w`` is ``1 / capture_rate`` for that battle's outcome
    class; ``true_wr`` is the manifest's own ``battles_won / battles_played`` — the cycle's TRUE
    100-game win rate, not the loss-enriched traced one.
    """
    from main.scaffolding_gauge import opponent_class

    man_path = os.path.join(trace_dir, "eval_manifest.json")
    if not os.path.exists(man_path):
        raise ConditioningRefusal(f"{trace_dir} has no eval_manifest.json — SELECTION UNKNOWN "
                                  "(rule 17); no conditioning statistic may be computed on it.")
    man = _load_json(man_path)
    sel = (man.get("selection") or {}).get("opponents") or {}
    if not sel:
        raise ConditioningRefusal(
            f"{trace_dir}'s eval_manifest.json records no `selection.opponents` block — the "
            "capture rates are unknown, so every statistic here would be an uncorrected read of a "
            "LOSS-ENRICHED tree (rule 17). REFUSING.")

    rows: List[tuple] = []
    refusals: List[str] = []
    per_opp: Dict[str, Any] = {}
    vmax = 0.0
    n_draw_battles = 0
    for opp in sorted(sel):
        odir = os.path.join(trace_dir, opp)
        rec = sel[opp]
        played = int(rec.get("battles_played", 0))
        won = int(rec.get("battles_won", 0))
        if played <= 0:
            refusals.append(f"{opp}: battles_played=0")
            continue
        crw, crl = rec.get("capture_rate_win"), rec.get("capture_rate_loss")
        true_wr = won / played
        cls = opponent_class(opp)
        n_b = 0
        if os.path.isdir(odir):
            for fn in sorted(os.listdir(odir)):
                if not fn.endswith("_states.npz"):
                    continue
                base = fn[: -len("_states.npz")]
                spath = os.path.join(odir, base + "_summary.json")
                if not os.path.exists(spath):
                    refusals.append(f"{opp}/{base}: no summary.json")
                    continue
                summ = _load_json(spath)
                res = (summ.get("meta") or {}).get("result")
                if res not in ("WIN", "LOSS"):
                    n_draw_battles += 1
                    continue
                y = 1.0 if res == "WIN" else 0.0
                cr = crw if y == 1.0 else crl
                if not cr:
                    refusals.append(f"{opp}/{base}: no capture rate for a {res} — outcome class "
                                    "never played; row DROPPED, never weighted 1.0")
                    continue
                try:
                    with np.load(os.path.join(odir, fn)) as z:
                        if "win_probs" not in z:
                            refusals.append(f"{opp}/{base}: npz carries no win_probs column")
                            continue
                        wp = np.asarray(z["win_probs"], dtype=float)
                        hs = np.asarray(z["has_state"])
                        vals = np.asarray(z["values"], dtype=float) if "values" in z else wp
                except (OSError, ValueError) as exc:
                    refusals.append(f"{opp}/{base}: npz unreadable ({exc})")
                    continue
                invs = summ.get("invocations") or []
                if len(invs) != wp.size:
                    refusals.append(f"{opp}/{base}: npz/invocation length mismatch "
                                    f"({wp.size} vs {len(invs)})")
                    continue
                if wp.size:
                    vmax = max(vmax, float(np.max(np.abs(vals - wp))))
                turns = np.array([int(i.get("turn", -1)) for i in invs])
                keep = (np.asarray(hs) == 1) & (turns > 0)
                tid = team_id(summ)
                w = 1.0 / float(cr)
                for v, t in zip(wp[keep].tolist(), turns[keep].tolist()):
                    rows.append((opp, cls, f"{opp}/{base}", y, int(t), float(v), w, tid,
                                 true_wr, float(played)))
                n_b += 1
        else:
            refusals.append(f"{opp}: no trace directory")
        per_opp[opp] = {"class": cls, "battles_played": played, "battles_won": won,
                        "true_win_rate": true_wr, "capture_rate_win": crw,
                        "capture_rate_loss": crl, "battles_loaded": n_b}

    arr = np.array(rows, dtype=STATE_DTYPE)
    if arr.size == 0:
        raise ConditioningRefusal(f"{trace_dir} yielded 0 usable states for the conditioning "
                                  f"meters. Refusals: {'; '.join(refusals[:6]) or '(none)'}")
    meta = {"trace_dir": os.path.abspath(trace_dir),
            "step": man.get("step"), "selection_schema": man.get("selection_schema"),
            "n_states": int(arr.size),
            "n_battles": int(np.unique(arr["battle"]).size),
            "n_opponents": int(np.unique(arr["opponent"]).size),
            "n_teams": int(np.unique(arr["team"]).size),
            "n_draw_battles_excluded": n_draw_battles,
            "max_abs_values_minus_winprobs": vmax,
            "opponents": per_opp, "refusals": refusals}
    return arr, meta


# --------------------------------------------------------------------------- roll-up

def rollup(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-BATTLE sums, so a cluster bootstrap is a gather over this table and nothing is
    recomputed from the states. One row per battle; ``sV_<bucket>`` / ``n_<bucket>`` carry the
    bucket's sum of ``V`` and its state count."""
    battles, first, inv = np.unique(arr["battle"], return_index=True, return_inverse=True)
    n_b = battles.size
    turn, V = arr["turn"], arr["V"]
    masks = {"all": np.ones(arr.size, bool), "t1_3": turn <= 3, "t1": turn == 1}
    b: Dict[str, np.ndarray] = {
        "battle": arr["battle"][first], "opponent": arr["opponent"][first],
        "opp_class": arr["opp_class"][first], "y": arr["y"][first],
        "team": arr["team"][first], "w": arr["w"][first],
        "true_wr": arr["true_wr"][first], "n_games": arr["n_games"][first]}
    for name, m in masks.items():
        b[f"n_{name}"] = np.bincount(inv[m], minlength=n_b).astype(float)
        b[f"sV_{name}"] = np.bincount(inv[m], weights=V[m], minlength=n_b)
    b["_state_battle_inv"] = inv
    return b


def cell_index(b: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
    """The cells are the OPPONENTS of this one cycle, sorted; plus each battle's cell code."""
    opps = sorted(set(b["opponent"].tolist()))
    oidx = {o: i for i, o in enumerate(opps)}
    return opps, np.array([oidx[o] for o in b["opponent"].tolist()], dtype=int)


def cell_stats(b: Dict[str, np.ndarray], sel: np.ndarray, cid_of_sel: np.ndarray,
               n_cells: int, bucket: str, wr: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-cell IPW battle-level mean ``V``, its squared standard error, and the bias against the
    cell's TRUE win rate.

    🚨 ``cid_of_sel`` is the cell code of each SELECTED battle, aligned 1:1 with ``sel`` — never a
    per-battle table to be indexed here. A cluster bootstrap draw has exactly as many entries as
    the original, so a size test cannot tell a resampled selection from an unresampled one; the
    mixture diagnostic's first revision paired resampled battles with the ORIGINAL cell codes and
    every interval was wrong (its hazard 2). The tell was percentile CIs that did not contain
    their own point estimates.
    """
    w = b["w"][sel]
    ns = b[f"n_{bucket}"][sel]
    sV = b[f"sV_{bucket}"][sel]
    y = b["y"][sel]
    c = cid_of_sel
    has = ns > 0
    Wb = np.bincount(c[has], weights=w[has], minlength=n_cells)
    Vb = np.bincount(c[has], weights=(w * sV / np.where(ns > 0, ns, 1))[has], minlength=n_cells)
    Yb = np.bincount(c[has], weights=(w * y)[has], minlength=n_cells)
    ok = Wb > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        mV = np.where(ok, Vb / np.where(ok, Wb, 1), np.nan)
        vb = np.where(ns > 0, sV / np.where(ns > 0, ns, 1), np.nan)
        dev = np.where(has, (w * (vb - mV[c])) ** 2, 0.0)
        S2 = np.bincount(c[has], weights=dev[has], minlength=n_cells)
        nb_ = np.bincount(c[has], minlength=n_cells).astype(float)
        # squared SE of the cell's IPW mean V — the noise the spread correction subtracts. Without
        # it a between-group variance is a statement about cell SIZE, not about opponents.
        seV2 = np.where(ok & (nb_ > 1),
                        S2 / np.where(ok, Wb, 1) ** 2 * np.where(nb_ > 1, nb_ / (nb_ - 1), 1.0),
                        np.nan)
    out = {"n_battles": np.bincount(c[has], minlength=n_cells).astype(float),
           "W_b": Wb, "meanV_b": mV, "seV2_b": seV2,
           "out_b": np.where(ok, Yb / np.where(ok, Wb, 1), np.nan)}
    out["bias"] = (mV - wr) if wr is not None else np.full(n_cells, np.nan)
    return out


# --------------------------------------------------------------------------- the identity

def spread_corrected(st: Dict[str, np.ndarray], keep: np.ndarray, true_wr: np.ndarray,
                     n_games: np.ndarray) -> Dict[str, float]:
    """BETWEEN-OPPONENT spread of ``V`` against that of the OUTCOME, each corrected for its OWN
    sampling noise, plus the UNCORRECTED companion.

    For ANY calibrated critic ``E[V | opponent] == E[y | opponent]``, so the two spreads must be
    EQUAL and ``ratio`` must be 1.0. Cells are weighted EQUALLY: the question is how far apart the
    opponents are, not how much traffic each got.

    🚨 ``ratio`` is CLAMPED — each corrected variance is floored at zero — which makes it a
    BIASED, NON-MONOTONE operator, and a point estimate of 0.000 routinely carries an interval
    like [0.21, 0.53] (head-refit hazard 1). **The interval is the read; a 0.000 is not "no
    spread".** ``ratio_raw`` is the same quantity with no correction and no clamp, reported beside
    it as the monotone companion.
    """
    m = (keep & np.isfinite(st["meanV_b"]) & np.isfinite(true_wr) & np.isfinite(st["seV2_b"])
         & np.isfinite(n_games) & (n_games > 0))
    nan = float("nan")
    if int(m.sum()) < 3:
        return {k: nan for k in ("sd_V", "sd_y", "ratio", "delta", "sd_V_raw", "sd_y_raw",
                                 "ratio_raw", "noise_V", "noise_y", "n_cells")}
    V, Y, G = st["meanV_b"][m], true_wr[m], n_games[m]
    nV = float(np.mean(st["seV2_b"][m]))
    nY = float(np.mean(Y * (1 - Y) / G))
    vV, vY = float(np.var(V, ddof=1)), float(np.var(Y, ddof=1))
    sV, sY = np.sqrt(max(vV - nV, 0.0)), np.sqrt(max(vY - nY, 0.0))
    rV, rY = np.sqrt(vV), np.sqrt(vY)
    return {"sd_V": sV, "sd_y": sY, "ratio": (sV / sY) if sY > 0 else nan,
            "delta": sV - sY, "sd_V_raw": rV, "sd_y_raw": rY,
            "ratio_raw": (rV / rY) if rY > 0 else nan,
            "noise_V": np.sqrt(nV), "noise_y": np.sqrt(nY), "n_cells": float(m.sum())}


def ols_slope(bias: np.ndarray, strength: np.ndarray, keep: np.ndarray) -> float:
    """OLS of the per-cell bias on strength/100 — the slope per 100 Elo, with an intercept.

    One cycle, so there are no cycle fixed effects to absorb (the mixture diagnostic needed them
    because it pooled four). Sign convention: bias := ``V`` − true win rate, so the MIXTURE
    hypothesis predicts a POSITIVE slope (too low against a weak opponent, too high against a
    strong one) and a separating head predicts zero.
    """
    m = keep & np.isfinite(bias) & np.isfinite(strength)
    if int(m.sum()) < 3:
        return float("nan")
    X = np.stack([strength[m] / 100.0, np.ones(int(m.sum()))], axis=1)
    try:
        beta, *_ = np.linalg.lstsq(X, bias[m], rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(beta[0])


# --------------------------------------------------------------------------- decoding from V

def w_r2(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    m = float(np.average(y, weights=w))
    den = float(np.sum(w * (y - m) ** 2))
    return float(1.0 - np.sum(w * (y - p) ** 2) / den) if den > 0 else float("nan")


def w_auc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """Weighted Mann-Whitney AUC, ties at 0.5. Vectorised — the bootstrap evaluates it thousands
    of times and a per-row tie loop makes that intractable."""
    pos = y > 0.5
    if not pos.any() or pos.all():
        return float("nan")
    _, grp = np.unique(p, return_inverse=True)
    ng = int(grp.max()) + 1
    neg_w = np.bincount(grp, weights=np.where(pos, 0.0, w), minlength=ng)
    pos_w = np.bincount(grp, weights=np.where(pos, w, 0.0), minlength=ng)
    below = np.concatenate([[0.0], np.cumsum(neg_w)])[:-1]
    tot = float(np.sum(pos_w * (below + 0.5 * neg_w)))
    Wp, Wn = float(pos_w.sum()), float(neg_w.sum())
    return float(tot / (Wp * Wn)) if Wp > 0 and Wn > 0 else float("nan")


def loo_team_wr(b_team: np.ndarray, b_y: np.ndarray, b_w: np.ndarray,
                min_battles: int = MIN_TEAM_BATTLES) -> np.ndarray:
    """Per-battle LEAVE-ONE-BATTLE-OUT IPW win rate of that battle's team; NaN under
    ``min_battles``. The leave-one-out is what stops the label carrying the outcome of the very
    battle it labels — without it the target is partly the answer."""
    teams, tinv = np.unique(b_team, return_inverse=True)
    tw = np.bincount(tinv, weights=b_w, minlength=teams.size)
    twy = np.bincount(tinv, weights=b_w * b_y, minlength=teams.size)
    tn = np.bincount(tinv, minlength=teams.size).astype(float)
    den = tw[tinv] - b_w
    ok = (tn[tinv] >= min_battles) & (den > 0)
    return np.where(ok, (twy[tinv] - b_w * b_y) / np.where(den > 0, den, 1.0), np.nan)


def _fold_of(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Assign each ROW a fold by hashing its GROUP, so no battle is ever in train and test at
    once — the single thing that separates an honest held-out score from a memorised one."""
    uniq, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    fold_of_group = rng.permutation(uniq.size) % k
    return fold_of_group[inv]


def _ridge_1d(x: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Weighted 1-feature ridge with an unpenalised intercept -> (intercept, slope)."""
    sw = float(w.sum())
    if sw <= 0:
        return float("nan"), 0.0
    xb, yb = float(np.average(x, weights=w)), float(np.average(y, weights=w))
    num = float(np.sum(w * (x - xb) * (y - yb)))
    den = float(np.sum(w * (x - xb) ** 2)) + alpha * sw
    slope = num / den if den > 0 else 0.0
    return yb - slope * xb, slope


def grouped_oof_scalar(x: np.ndarray, y: np.ndarray, w: np.ndarray, groups: np.ndarray, *,
                       task: str, k: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions of ``y`` from the SINGLE scalar ``x``, folds grouped by battle.

    The penalty is chosen by an INNER grouped split inside each outer fold, exactly as the probe
    read does — for one feature the choice can only shrink the slope toward zero, and it is kept
    so that the selection optimism priced into the probe read's grid is priced here too. A fold
    whose training column is constant predicts the training mean rather than dividing by zero.
    """
    n = x.size
    pred = np.full(n, np.nan)
    outer = _fold_of(groups, k, seed)
    for f in range(k):
        te = outer == f
        tr = ~te
        if not te.any() or not tr.any() or w[tr].sum() <= 0:
            continue
        xt, yt, wt = x[tr], y[tr], w[tr]
        sd = float(np.sqrt(np.average((xt - np.average(xt, weights=wt)) ** 2, weights=wt)))
        if not np.isfinite(sd) or sd <= 0:
            pred[te] = float(np.average(yt, weights=wt))
            continue
        inner = _fold_of(groups[tr], 3, seed + 101 + f)
        best, best_score = RIDGE_ALPHAS[0], -np.inf
        for a in RIDGE_ALPHAS:
            ip = np.full(int(tr.sum()), np.nan)
            for g in range(3):
                ite, itr = inner == g, inner != g
                if not ite.any() or not itr.any() or wt[itr].sum() <= 0:
                    continue
                b0, b1 = _ridge_1d(xt[itr], yt[itr], wt[itr], a)
                ip[ite] = b0 + b1 * xt[ite]
            ok = np.isfinite(ip)
            if ok.sum() < 5:
                continue
            s = (w_auc(yt[ok], ip[ok], wt[ok]) if task == "auc"
                 else w_r2(yt[ok], ip[ok], wt[ok]))
            if np.isfinite(s) and s > best_score:
                best, best_score = a, s
        b0, b1 = _ridge_1d(xt, yt, wt, best)
        pred[te] = b0 + b1 * x[te]
    return pred


def score(y: np.ndarray, p: np.ndarray, w: np.ndarray, task: str) -> float:
    ok = np.isfinite(p)
    if int(ok.sum()) < 5:
        return float("nan")
    return w_auc(y[ok], p[ok], w[ok]) if task == "auc" else w_r2(y[ok], p[ok], w[ok])


# --------------------------------------------------------------------------- strength axis

def strength_axis(run_dir: str, step: int, per_opp: Dict[str, Any], *,
                  say: Callable[[str], None] = lambda _m: None
                  ) -> Tuple[Optional[Dict[str, float]], str]:
    """``({opponent: Elo}, note)`` on ONE bot-anchored scale, or ``(None, reason)``.

    Bots come from ``data/gen3_bot_elo_anchors.json``. Sentinels come from an ALL-STEPS refit of
    the run's own snapshot ladder, because the committed ``ladder.json`` is sliced to the
    snapshots pool grooming left behind and would leave a cycle's sentinel unrated. The positional
    ``sentinel_k`` -> ``eval_results.jsonl`` map is VERIFIED against the manifest's own win counts
    and a mismatch REFUSES rather than guesses.

    🚨 ``fit_ladder`` calls ``elo.load_bot_anchors()`` with a RELATIVE default path, so a fit run
    from any cwd but a checkout root silently loses the bot pins and returns an UNANCHORED ladder
    — ratings near 1000 instead of near 2000, ``anchored_to_bots: false``, and no error. Mixing
    those with the bot anchors would put one strength axis on two scales. This chdirs to the repo
    root and REFUSES an unanchored fit (mixture-diagnostic hazard 1).
    """
    from utils.paths import repo_path, repo_root

    cwd = os.getcwd()
    try:
        anchors = _load_json(str(repo_path("data", "gen3_bot_elo_anchors.json")))["ratings"]
    except (OSError, ValueError, KeyError) as exc:
        return None, f"no bot anchors ({exc}) — the strength axis has no pins"
    out: Dict[str, float] = {}
    sentinels = [o for o in per_opp if o not in anchors]
    for o in per_opp:
        if o in anchors:
            out[o] = float(anchors[o])
    if not sentinels:
        return (out or None), ("bot anchors only — this cycle has no sentinel opponents"
                               if out else "no opponent could be rated")

    ev = os.path.join(run_dir, "eval_results.jsonl")
    if not os.path.exists(ev):
        return None, (f"{len(sentinels)} sentinel opponents and no eval_results.jsonl — the "
                      "sentinel->snapshot map cannot be built, so a bot-only axis would compare "
                      "two different populations. Row OMITTED.")
    try:
        os.chdir(repo_root())
        from agents.training.snapshot_ladder import fit_ladder, load_games
        steps = sorted({s for pair in load_games(run_dir) for s in pair})
        if not steps:
            return None, "the run's games.jsonl carries no frozen-vs-frozen edges to fit. OMITTED."
        fit = fit_ladder(run_dir, steps=steps, write=False)
    except Exception as exc:                                   # noqa: BLE001 - reported, not raised
        return None, f"snapshot-ladder refit failed ({type(exc).__name__}: {exc}). Row OMITTED."
    finally:
        os.chdir(cwd)
    if not fit.get("anchored_to_bots"):
        return None, ("REFUSED: the snapshot-ladder refit is NOT anchored to the bot pins "
                      "(`anchored_to_bots` false) — its ratings would not share a scale with the "
                      "bot anchors. Row OMITTED rather than reported on two scales.")
    ratings = fit.get("ratings") or {}

    row = None
    try:
        with open(ev) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if int(r.get("step", -1)) == int(step):
                    row = r
    except (OSError, ValueError) as exc:
        return None, f"eval_results.jsonl unreadable ({exc}). Row OMITTED."
    if row is None:
        return None, (f"eval_results.jsonl has no row at step {step} — the positional "
                      "sentinel->snapshot map cannot be verified. Row OMITTED.")
    for k, s in enumerate(row.get("sentinels") or []):
        name = f"sentinel_{k}"
        if name not in per_opp:
            continue
        ev_wr = float(s.get("win_rate", float("nan")))
        man_wr = float(per_opp[name]["true_win_rate"])
        if not np.isfinite(ev_wr) or abs(ev_wr - man_wr) > 1e-9:
            return None, (f"the positional sentinel map MISMATCHES at {name}: eval row win rate "
                          f"{ev_wr} vs manifest {man_wr}. Row OMITTED rather than guessed.")
        rating = ratings.get(str(s.get("step")))
        if rating is None:
            return None, (f"{name}'s snapshot step {s.get('step')} is not in the refit ladder. "
                          "Row OMITTED.")
        out[name] = float(rating)
    missing = [o for o in per_opp if o not in out]
    if missing:
        return None, (f"{len(missing)} opponent(s) unrated ({', '.join(sorted(missing)[:4])}) — "
                      "a partial axis would compare a different population. Row OMITTED.")
    say(f"strength axis: {len(out)} opponents on the bot-anchored refit "
        f"({fit.get('eval_sentinel_edges_dropped', 0)} greedy-vs-stochastic edges dropped)")
    return out, (f"bot anchors + an ALL-STEPS bot-anchored refit of the run's snapshot ladder "
                 f"({len(ratings)} nodes rated, anchored_to_bots=true), the positional "
                 f"sentinel map verified against the manifest's own win counts")


# --------------------------------------------------------------------------- the block

#: every conditioning meter: key -> (human quantity, stratum, "higher is" direction note).
METERS: Tuple[Tuple[str, str, str], ...] = (
    ("cond.spread_ratio.t1_3", "between-opponent spread ratio sd(V)/sd(outcome), noise-corrected",
     "turn 1-3"),
    ("cond.spread_ratio_raw.t1_3", "the same ratio UNCORRECTED and unclamped", "turn 1-3"),
    ("cond.spread_delta.t1_3", "sd(V) - sd(outcome), noise-corrected", "turn 1-3"),
    ("cond.spread_ratio.all", "between-opponent spread ratio sd(V)/sd(outcome), noise-corrected",
     "all states"),
    ("cond.spread_ratio_raw.all", "the same ratio UNCORRECTED and unclamped", "all states"),
    ("cond.spread_delta.all", "sd(V) - sd(outcome), noise-corrected", "all states"),
    ("cond.elo_slope", "slope of bias (V - true win rate) on opponent Elo, per 100 Elo",
     "all states"),
    ("cond.own_team_r2.t1", "own-team leave-one-battle-out win-rate R^2 of V", "turn 1"),
    ("cond.own_team_r2.all", "own-team leave-one-battle-out win-rate R^2 of V", "all states"),
    ("cond.opp_class_auc.t1", "opponent-CLASS (pool vs bot) AUC of V", "turn 1"),
)
METER_KEYS = tuple(k for k, _q, _s in METERS)


def _cap_states(arr: np.ndarray, mask: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """At most ``cap`` states per battle inside the bucket — every battle stays represented and
    the within-battle correlation that rides into a score is bounded."""
    rng = np.random.default_rng(seed)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return idx
    battles = arr["battle"][idx]
    order = np.argsort(battles, kind="stable")
    idx, battles = idx[order], battles[order]
    bounds = np.flatnonzero(np.concatenate([[True], battles[1:] != battles[:-1], [True]]))
    out = [g if g.size <= cap else rng.choice(g, cap, replace=False)
           for g in (idx[bounds[i]:bounds[i + 1]] for i in range(bounds.size - 1))]
    return np.sort(np.concatenate(out))


def conditioning_block(run_dir: str, step: int, *, boot: int = N_BOOT, seed: int = BOOT_SEED,
                       ladder: str = "refit",
                       say: Callable[[str], None] = lambda _m: None) -> Dict[str, Any]:
    """Every conditioning meter for ONE run at ONE cycle, with its raw bootstrap draws.

    Returns ``{"points": {key: value}, "_draws": {key: ndarray}, "frame": …, "omitted": …}``. A
    meter this cycle cannot support (no sentinel, no strength axis, one opponent class) is listed
    in ``omitted`` with the REASON — never emitted as a NaN row that reads like a measurement.
    """
    trace_dir = os.path.join(run_dir, "eval_traces", f"step_{int(step)}")
    arr, meta = extract_cycle(trace_dir)
    b = rollup(arr)
    opps, cid = cell_index(b)
    n_cells = len(opps)
    true_wr = np.full(n_cells, np.nan)
    n_games = np.full(n_cells, np.nan)
    for o, wr_, g in zip(b["opponent"].tolist(), b["true_wr"].tolist(), b["n_games"].tolist()):
        k = opps.index(o)
        true_wr[k], n_games[k] = wr_, g

    omitted: Dict[str, str] = {}
    strength_map, strength_note = ((None, "strength axis DISABLED (--cond-ladder off)")
                                   if ladder == "off"
                                   else strength_axis(run_dir, int(step), meta["opponents"],
                                                      say=say))
    strength = (np.array([strength_map.get(o, np.nan) for o in opps], dtype=float)
                if strength_map else np.full(n_cells, np.nan))
    if strength_map is None:
        omitted["cond.elo_slope"] = strength_note

    # ---- score meters: fixed OOF decoders, then a battle-clustered bootstrap of the EVALUATION
    # sample (the probe read's paired-OOF bootstrap — it prices sampling noise in the held-out
    # score, not the variability of refitting).
    binv = b["_state_battle_inv"]
    b_wr = loo_team_wr(b["team"], b["y"], b["w"])
    y_wr_state = b_wr[binv]
    y_cls_state = (b["opp_class"][binv] == "pool").astype(float)
    scores: Dict[str, Dict[str, Any]] = {}
    for key, task, y_state, mask in (
            ("cond.own_team_r2.t1", "r2", y_wr_state,
             (arr["turn"] == 1) & np.isfinite(y_wr_state)),
            ("cond.own_team_r2.all", "r2", y_wr_state, np.isfinite(y_wr_state)),
            ("cond.opp_class_auc.t1", "auc", y_cls_state, arr["turn"] == 1)):
        idx = _cap_states(arr, mask, STATES_PER_BATTLE_CAP, seed)
        if idx.size < 20 or np.unique(arr["battle"][idx]).size < 5:
            omitted[key] = (f"only {idx.size} usable states / "
                            f"{np.unique(arr['battle'][idx]).size if idx.size else 0} battles at "
                            "this cycle — too few for a grouped out-of-fold decode.")
            continue
        yv, wv = y_state[idx], arr["w"][idx]
        if task == "auc" and (yv.max() == yv.min()):
            omitted[key] = ("this cycle's roster carries only ONE opponent class, so the class "
                            "AUC is undefined. Row OMITTED.")
            continue
        pred = grouped_oof_scalar(arr["V"][idx], yv, wv, arr["battle"][idx], task=task, seed=seed)
        scores[key] = {"idx": idx, "y": yv, "w": wv, "pred": pred, "task": task,
                       "n_states": int(idx.size),
                       "n_battles": int(np.unique(arr["battle"][idx]).size)}

    # ---- point estimates
    sel0 = np.arange(b["y"].size)
    st_all0 = cell_stats(b, sel0, cid, n_cells, "all", true_wr)
    st_t13_0 = cell_stats(b, sel0, cid, n_cells, "t1_3", true_wr)
    keep0 = st_all0["n_battles"] > 0
    sc_all0 = spread_corrected(st_all0, keep0, true_wr, n_games)
    sc_t13_0 = spread_corrected(st_t13_0, keep0, true_wr, n_games)
    points: Dict[str, float] = {
        "cond.spread_ratio.t1_3": sc_t13_0["ratio"],
        "cond.spread_ratio_raw.t1_3": sc_t13_0["ratio_raw"],
        "cond.spread_delta.t1_3": sc_t13_0["delta"],
        "cond.spread_ratio.all": sc_all0["ratio"],
        "cond.spread_ratio_raw.all": sc_all0["ratio_raw"],
        "cond.spread_delta.all": sc_all0["delta"],
        "cond.elo_slope": ols_slope(st_all0["bias"], strength, keep0),
    }
    for key, s in scores.items():
        points[key] = score(s["y"], s["pred"], s["w"], s["task"])

    # ---- battle-clustered bootstrap: battles are resampled WITHIN their opponent cell (the
    # roster is a fixed pinned set), and the outcome side is redrawn from its own Binomial so the
    # 100-game noise in the thing V is compared against is priced.
    order = np.argsort(cid, kind="stable")
    cid_s = cid[order]
    starts = np.searchsorted(cid_s, np.arange(n_cells), "left")
    sizes = np.searchsorted(cid_s, np.arange(n_cells), "right") - starts
    slots = [order[starts[k]:starts[k] + sizes[k]] for k in range(n_cells)]
    live = [k for k in range(n_cells) if slots[k].size]
    all_slots = np.concatenate([slots[k] for k in live])
    code_per_slot = np.concatenate([np.full(slots[k].size, k) for k in live])
    size_per_slot = np.concatenate([np.full(slots[k].size, slots[k].size) for k in live]
                                   ).astype(float)
    start_per_slot = np.concatenate(
        [np.full(slots[k].size, off) for k, off in
         zip(live, np.cumsum([0] + [slots[k].size for k in live[:-1]]))])
    # State rows of each battle, as a flat CSR-shaped (start, count) pair per score meter. The
    # bootstrap gathers ~2,000 x 950 battles three times over; a python loop over `sel` made that
    # the whole cost of the read, and the repeat/cumsum form below is the same gather vectorised.
    gathers: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, s in scores.items():
        bi = binv[s["idx"]]
        o = np.argsort(bi, kind="stable")
        cnt = np.bincount(bi, minlength=b["y"].size).astype(np.int64)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        gathers[key] = (o, start, cnt)

    rng = np.random.default_rng(seed)
    draws: Dict[str, List[float]] = {k: [] for k in METER_KEYS}
    for _ in range(int(boot)):
        j = start_per_slot + (rng.random(all_slots.size) * size_per_slot).astype(int)
        sel, cid_sel = all_slots[j], code_per_slot
        wr = np.where(np.isfinite(true_wr) & np.isfinite(n_games),
                      rng.binomial(np.nan_to_num(n_games, nan=1).astype(int),
                                   np.clip(np.nan_to_num(true_wr, nan=0.5), 0, 1))
                      / np.where(np.isfinite(n_games) & (n_games > 0), n_games, 1), np.nan)
        st_a = cell_stats(b, sel, cid_sel, n_cells, "all", wr)
        st_t = cell_stats(b, sel, cid_sel, n_cells, "t1_3", wr)
        kp = st_a["n_battles"] > 0
        sa, stt = (spread_corrected(st_a, kp, wr, n_games),
                   spread_corrected(st_t, kp, wr, n_games))
        draws["cond.spread_ratio.t1_3"].append(stt["ratio"])
        draws["cond.spread_ratio_raw.t1_3"].append(stt["ratio_raw"])
        draws["cond.spread_delta.t1_3"].append(stt["delta"])
        draws["cond.spread_ratio.all"].append(sa["ratio"])
        draws["cond.spread_ratio_raw.all"].append(sa["ratio_raw"])
        draws["cond.spread_delta.all"].append(sa["delta"])
        draws["cond.elo_slope"].append(ols_slope(st_a["bias"], strength, kp))
        for key, s in scores.items():
            flat, start, cnt = gathers[key]
            c = cnt[sel]
            tot = int(c.sum())
            if tot == 0:
                continue
            base = np.repeat(start[sel], c)
            within = np.arange(tot) - np.repeat(np.cumsum(c) - c, c)
            g = flat[base + within]
            draws[key].append(score(s["y"][g], s["pred"][g], s["w"][g], s["task"]))

    out_draws = {k: np.asarray([d for d in v if np.isfinite(d)], dtype=float)
                 for k, v in draws.items() if k in points and np.isfinite(points.get(k, np.nan))}
    for k in METER_KEYS:
        if k not in points or not np.isfinite(points[k]):
            omitted.setdefault(k, "the point estimate is undefined on this cycle "
                                  "(too few cells or a degenerate column).")
    return {
        "points": {k: float(v) for k, v in points.items() if np.isfinite(v)},
        "_draws": out_draws,
        "omitted": omitted,
        "frame": {**{k: v for k, v in meta.items() if k != "opponents"},
                  "n_cells": int(keep0.sum()), "boot": int(boot), "seed": int(seed),
                  "states_per_battle_cap": STATES_PER_BATTLE_CAP,
                  "min_team_battles": MIN_TEAM_BATTLES,
                  "score_frames": {k: {"n_states": s["n_states"], "n_battles": s["n_battles"]}
                                   for k, s in scores.items()},
                  "recorded_v_note": recorded_v_note()},
        "spread": {"t1_3": sc_t13_0, "all": sc_all0},
        "strength": {"note": strength_note,
                     "ratings": ({o: float(strength_map[o]) for o in sorted(strength_map)}
                                 if strength_map else None)},
        "opponents": {o: {**meta["opponents"][o],
                          "mean_V": (float(st_all0["meanV_b"][opps.index(o)])
                                     if o in opps and np.isfinite(
                                         st_all0["meanV_b"][opps.index(o)]) else None),
                          "mean_V_t1_3": (float(st_t13_0["meanV_b"][opps.index(o)])
                                          if o in opps and np.isfinite(
                                              st_t13_0["meanV_b"][opps.index(o)]) else None),
                          "elo": (float(strength_map[o])
                                  if strength_map and o in strength_map else None)}
                      for o in sorted(meta["opponents"])},
    }
