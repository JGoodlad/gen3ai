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
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from main.ops import calibration_slope as CS
from main.ops import team_conditioning as TC

#: turn windows the meters are reported on. ``t1`` is the sharpest (nothing has happened, every
#: battle contributes exactly one state, and the two sides of the identity condition on the same
#: event); ``t1_3`` is the mixture diagnostic's own registered window.
#:
#: 🚨 ``t4_10`` and ``t11_24`` were added 2026-09-10 because the N-curve
#: (``measurements/winprob_refit_ncurve_2026-09-10/`` §3) established that **on a matched-team
#: frame the opponent is UNOBSERVABLE at turn 1** — Gen 3 has no team preview, and the trainee's
#: own team is the only thing in the observation at that point. Measured there, `value_pooled`
#: decodes the opponent CLASS at AUC 0.502 / 0.508 at turn 1, 0.67 over turns 1-3 and 0.82 over
#: turns 4-10 against a permutation null of ~0.52. **A turn-1 spread ratio of 0 is therefore
#: BAYES-OPTIMAL on such a frame, not a defect**, and the window in which "does the head condition
#: on the opponent" HAS an answer is turns >= 4. The registered `t1` / `t1_3` rows are kept
#: unchanged and bit-identical — they are what three landed reads were registered on — and the
#: late windows are added BESIDE them.
BUCKETS = ("t1", "t1_3", "t4_10", "t11_24", "all")
#: the turn window the opponent first becomes observable in (N-curve §3).
OBSERVABLE_TURNS = (4, 10)
#: the MID window — the N-curve's own headline window for the spread ratio, where both the head
#: and the outcome have had time to separate.
MID_TURNS = (11, 24)
#: how many quantile bins the OPTIMAL-spread reference estimates its posterior `q(o | V)` over.
#: Enough that a 12-opponent roster is separable, few enough that each training bin holds many
#: battles at the frame sizes this ladder reads (~800-24,000).
OPT_BINS = 20
#: folds of the OUT-OF-FOLD posterior behind the optimal-spread reference, grouped by battle —
#: the same k the decode rows use, for the same reason.
OPT_FOLDS = 5
#: at most this many states per battle per bucket enter a SCORE meter (the probe read's cap) —
#: it keeps every battle represented while bounding how much within-battle correlation rides in.
STATES_PER_BATTLE_CAP = 2
#: a team needs this many battles before its leave-one-battle-out win rate is a label at all,
#: and before it is a CELL of the within-team rows.
MIN_TEAM_BATTLES = 4
#: the turn from which a state is LATE. The identity's own `late` bucket, so `own_team_r2.late`
#: and `identity.bias.late` are cut at the same clock.
LATE_TURN = 25
#: the own-team decode at turn 1 MINUS the same decode late. Registered as its own row because it
#: is the contrast that separates a critic CONDITIONING on its team from one SUBSTITUTING team
#: identity for the board: board information should take over as the game unfolds, so a
#: conditioning critic's team R^2 FALLS from turn 1 to late and a substituting one's does not.
OWN_TEAM_R2_DIFF = "cond.own_team_r2.t1_minus_late"
#: the CALIBRATION SLOPE rows — the sharp test of SHRINKAGE. A target that blends a bootstrapped
#: V into the label is compressed toward the base rate, and fitting a compressed target is a
#: shrinkage estimator: better rank ORDER, smaller AMPLITUDE. The out-of-fold decodes above are
#: scale-invariant and cannot see that; the spread rows see it mixed with everything else; the
#: slope sees it directly, in the units it happens in. >1 is UNDER-dispersed (shrunk). Arithmetic
#: and the lever-arm hazard: :mod:`main.ops.calibration_slope`.
#: the LATE-WINDOW rows, added 2026-09-10 (N-curve §3 — the opponent is unobservable at turn 1).
#: `t4_10` is the window the opponent BECOMES observable in; `t11_24` is the window the N-curve
#: reports the head's own spread ratio at (0.759 on `ctrl10M`, 0.464 on `ctrl10M_b`).
SPREAD_RATIO_T4_10 = "cond.spread_ratio.t4_10"
SPREAD_RATIO_T11_24 = "cond.spread_ratio.t11_24"
OPP_CLASS_AUC_T1_3 = "cond.opp_class_auc.t1_3"
OPP_CLASS_AUC_T4_10 = "cond.opp_class_auc.t4_10"
#: the OPPONENT-DECODABLE component of the spread — the between-opponent spread ratio a head that
#: conditioned ONLY on what its own output reveals about WHICH opponent it faces would exhibit.
#: 🚨 **NOT an upper bound on `cond.spread_ratio.*`** — see :func:`oof_opt_value`. DESCRIPTIVE.
OPT_RATIO = {"t1_3": "cond.spread_ratio_optimal.t1_3",
             "t4_10": "cond.spread_ratio_optimal.t4_10",
             "t11_24": "cond.spread_ratio_optimal.t11_24"}
#: the windows the optimal reference is computed on, and the row each is compared against.
OPT_WINDOWS = (("t1_3", "cond.spread_ratio.t1_3"),
               ("t4_10", SPREAD_RATIO_T4_10),
               ("t11_24", SPREAD_RATIO_T11_24))
OPT_PROVISIONAL_WHY = (
    "a DESCRIPTIVE DECOMPOSITION TERM, never a pass/fail bar and never a treatment effect. It is "
    "the between-opponent spread ratio a head conditioning ONLY on WHICH OPPONENT its own `V` "
    "reveals would exhibit — `sum_o q(o|V)*p_o` — and it is read BESIDE "
    "`cond.spread_ratio.<window>` to split that row into an OPPONENT-IDENTITY part and a "
    "remainder. 🚨 IT IS NOT AN UPPER BOUND ON THAT ROW AND `V` ROUTINELY EXCEEDS IT: measured "
    "on the hp800 lambda09/ctrl10M pair, V reads 0.647 / 0.704 at turns 11-24 against this row's "
    "0.214 / 0.187. A per-state conditional mean is an ATTENUATING transform, so the cell-mean "
    "spread of `E[p_o | V]` is smaller than the cell-mean spread of `V` itself; the excess is "
    "between-opponent spread that rides on BOARD STATE correlated with the opponent rather than "
    "on opponent identity. A delta between two sides is a difference in how much opponent "
    "IDENTITY each side's output carries; it is reported, and it is never DETECTED")
CALIB_SLOPE_ALL = "cond.calibration_slope.all"
CALIB_SLOPE_T13 = "cond.calibration_slope.t1_3"
CALIB_SLOPE_WITHIN = "cond.calibration_slope.within_stratum"
CALIB_SLOPE_COMMON = "cond.calibration_slope.common_support"
CALIB_INTERCEPT_ALL = "cond.calibration_intercept.all"
CALIB_INTERCEPT_T13 = "cond.calibration_intercept.t1_3"
CALIB_INTERCEPT_COMMON = "cond.calibration_intercept.common_support"
#: how :mod:`main.ops.calibration_slope` names each fit -> the meter key it lands under. The
#: AS-TRACED rows and the COMMON-SUPPORT companion are the same estimator under two name maps.
CALIB_NAMES: Dict[str, str] = {"slope.all": CALIB_SLOPE_ALL,
                               "intercept.all": CALIB_INTERCEPT_ALL,
                               "slope.t1_3": CALIB_SLOPE_T13,
                               "intercept.t1_3": CALIB_INTERCEPT_T13,
                               "slope.within_stratum": CALIB_SLOPE_WITHIN}
CALIB_COMMON_NAMES: Dict[str, str] = {"slope.all": CALIB_SLOPE_COMMON,
                                      "intercept.all": CALIB_INTERCEPT_COMMON}
#: the label a PROVISIONAL row carries INSTEAD of a registered verdict.
PROVISIONAL_LABEL = "PROVISIONAL — no floor; never DETECTED"
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
                        "battles_drawn": int(rec.get("battles_drawn", 0)),
                        "true_win_rate": true_wr, "capture_rate_win": crw,
                        "capture_rate_loss": crl, "battles_loaded": n_b,
                        # the manifest's OWN realized capture profile, carried through so a
                        # subsample can recompute the rates against the same denominators the
                        # trainer used (main.ops.quota_match)
                        "traces_written": int(rec.get("traces_written", 0)),
                        "traces_won": int(rec.get("traces_won", 0)),
                        "traces_drawn": int(rec.get("traces_drawn", 0))}

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
    masks = {"all": np.ones(arr.size, bool), "t1_3": turn <= 3, "t1": turn == 1,
             "t4_10": (turn >= OBSERVABLE_TURNS[0]) & (turn <= OBSERVABLE_TURNS[1]),
             "t11_24": (turn >= MID_TURNS[0]) & (turn <= MID_TURNS[1])}
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
               n_cells: int, bucket: str, wr: Optional[np.ndarray],
               cols: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """Per-cell IPW battle-level mean ``V``, its squared standard error, and the bias against the
    cell's TRUE win rate.

    ``cols`` substitutes a DIFFERENT per-battle ``(n, sum)`` pair for the bucket's own — the one
    hook the OPTIMAL-spread reference needs, since its ``V_opt`` is a second per-state column over
    the same battles. Absent (every registered row), the bucket's own columns are read and the
    arithmetic is bit-for-bit what it was.

    🚨 ``cid_of_sel`` is the cell code of each SELECTED battle, aligned 1:1 with ``sel`` — never a
    per-battle table to be indexed here. A cluster bootstrap draw has exactly as many entries as
    the original, so a size test cannot tell a resampled selection from an unresampled one; the
    mixture diagnostic's first revision paired resampled battles with the ORIGINAL cell codes and
    every interval was wrong (its hazard 2). The tell was percentile CIs that did not contain
    their own point estimates.
    """
    w = b["w"][sel]
    n_col, sV_col = cols if cols is not None else (b[f"n_{bucket}"], b[f"sV_{bucket}"])
    ns = n_col[sel]
    sV = sV_col[sel]
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


def oof_opt_value(v: np.ndarray, z: np.ndarray, w: np.ndarray, groups: np.ndarray, *,
                  bins: int = OPT_BINS, k: int = OPT_FOLDS, seed: int = 0) -> np.ndarray:
    """``V_opt(s) = sum_o q(o | V(s)) * p_o`` — the OPPONENT-DECODABLE component of ``V``.

    🚨 **THIS IS NOT A CEILING, AND THE FIRST READ OF IT PROVED THAT.** It was commissioned as
    "the spread an optimally-conditioned `V` would exhibit", and on the very first pair `V`'s own
    ratio EXCEEDED it at every window (hp800, turns 11-24: 0.647 / 0.704 against 0.214 / 0.187).
    The mechanism is not a defect: ``E[p_o | V]`` is a per-state CONDITIONAL MEAN, which attenuates
    — the between-cell spread of a shrinking transform of ``V`` is smaller than the between-cell
    spread of ``V``. So the row is a DECOMPOSITION TERM, not a bound: it is the part of
    ``cond.spread_ratio.<window>`` attributable to the head's output REVEALING WHICH OPPONENT it
    faces, and the excess is spread riding on BOARD STATE that happens to differ by opponent (you
    are ahead by turn 12 against a weak bot, and `V` says so without recognising the bot). Read
    the two together; never quote this one as "the maximum".

    ``z`` is each state's ``p_o``: the empirical win rate of the opponent that state's battle was
    played against. Because ``p_o`` is a per-opponent scalar, the sum over opponents collapses
    exactly — ``sum_o q(o | v) * p_o`` IS the conditional expectation of ``z`` given ``v`` — so the
    estimator is a conditional mean of ``z`` on ``V`` and needs no explicit 12-way posterior. It
    is computed NON-PARAMETRICALLY, by quantile bins of ``V`` fitted on the training folds only:
    a bin's weighted mean of ``z`` is *literally* ``sum_o q_hat(o | bin) * p_o``, the weighted
    opponent shares inside the bin being ``q_hat``. Bin EDGES come from the unweighted training
    ``V`` (a partition, not an estimate); the mean INSIDE a bin is HT-weighted, because that is
    the quantity being estimated. A test row landing in a bin with no training mass takes the
    training grand mean rather than a NaN.

    🚨 **DECODED FROM THE HEAD'S OWN OUTPUT, NOT FROM ITS INFORMATION SET.** The conditioning
    block does NO model forward by construction (see the module docstring) — the only per-state
    signal it holds is the recorded scalar ``V``. So the opponent posterior is ``q(o | V)`` and
    not ``q(o | value_pooled)``: this measures how much opponent identity the head's EMITTED
    NUMBER carries, which is a LOWER bound on how much its features carry. A version at the true
    information set needs a forward over ``value_pooled`` and is a different, more expensive
    instrument.

    Folds are grouped by BATTLE for the same reason every other decode here is: a posterior fitted
    on the same battle it scores is partly the answer.
    """
    n = v.size
    out = np.full(n, np.nan)
    if n == 0:
        return out
    outer = _fold_of(groups, k, seed)
    for f in range(k):
        te, tr = outer == f, outer != f
        if not te.any() or not tr.any() or w[tr].sum() <= 0:
            continue
        vt, zt, wt = v[tr], z[tr], w[tr]
        grand = float(np.average(zt, weights=wt))
        edges = np.unique(np.quantile(vt, np.linspace(0.0, 1.0, int(bins) + 1)[1:-1]))
        if edges.size == 0:
            out[te] = grand
            continue
        n_bin = edges.size + 1
        bt = np.searchsorted(edges, vt, side="right")
        Wb = np.bincount(bt, weights=wt, minlength=n_bin)
        Zb = np.bincount(bt, weights=wt * zt, minlength=n_bin)
        with np.errstate(invalid="ignore", divide="ignore"):
            mb = np.where(Wb > 0, Zb / np.where(Wb > 0, Wb, 1.0), grand)
        out[te] = mb[np.searchsorted(edges, v[te], side="right")]
    return out


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

class Meter(NamedTuple):
    """One conditioning meter's DEFINITION, including whether its expectation depends on the SIZE
    of the frame it is computed on.

    🚨 ``frame_sensitive`` is the flag :mod:`main.ops.quota_match` dispatches on, and it is
    DECLARED here rather than inferred from the key. Two ladder arms traced at different outcome
    quotas hold frames of different sizes, and a statistic whose expectation moves with frame size
    is then not comparable between them however the selection is reweighted — Horvitz-Thompson
    reweighting corrects the loss-ENRICHMENT, never the frame's SIZE. That is the defect the
    2026-09-09 RETRACTION convicted: ``cond.own_team_r2.t1`` read +0.084 DETECTED between an arm
    at 40/40/10 and a control at 5/10/5, and −0.001 NOT DETECTED once the arm was cut to the
    control's realized profile.

    Two mechanisms make a row frame-sensitive, and both are marked:

    * an out-of-fold score of a model **FIT** on the frame (``own_team_r2.*``,
      ``opp_class_auc.t1`` — every row that goes through :func:`grouped_oof_scalar`): its
      expectation rises with the number of battles the decoder is fit on;
    * an **UNCORRECTED** second moment (``spread_ratio_raw.*``): the sampling noise it does not
      subtract grows as the cells shrink.

    A noise-CORRECTED spread row, the spread delta and the Elo slope are weighted means and
    regressions on cell means whose expectation does not move with frame size given correct
    weights, so they are read as traced.

    A THIRD mechanism arrived with the own-team rows (2026-09-10) and is marked the same way: a
    row whose CELLS are chosen by a battle-count threshold (``MIN_TEAM_BATTLES``) has a cell SET
    that is itself a function of frame size — a smaller frame keeps only the busiest teams, which
    is a different population of cells and not merely a noisier estimate of the same ones. The
    opponent rows have no such mechanism, their roster being a pinned set.

    ``provisional`` marks a row that must NEVER be labelled DETECTED, whatever its interval does.
    It is not a weaker version of the label — it is the absence of one: the row has no replicate
    floor and cannot be given one from the controls (see ``provisional_why``), so a large move is
    informative, a small one is not, and the report says exactly that in place of a verdict.

    ``pair_level`` marks a row that a SINGLE run's block cannot compute at all, because its
    definition names both sides — the COMMON-SUPPORT calibration slope, whose window is the
    intersection of the two sides' central 95% of ``V``. :mod:`main.ops.critic_read` fits it from
    each side's cached per-state columns once both readouts exist. It is DECLARED rather than
    matched on a name for the same reason the other two flags are, and it is why the block's
    "every meter is reported or omitted with a reason" invariant excludes it: a single-run block
    listing it as omitted would put a false reason on a row that the pair then fills in.
    """

    key: str
    quantity: str
    stratum: str
    frame_sensitive: bool
    why: str
    provisional: bool = False
    provisional_why: str = ""
    pair_level: bool = False


#: every conditioning meter, with its frame-size sensitivity DECLARED (see :class:`Meter`).
METER_SPECS: Tuple[Meter, ...] = (
    Meter("cond.spread_ratio.t1_3",
          "between-opponent spread ratio sd(V)/sd(outcome), noise-corrected", "turn 1-3",
          False, "a ratio of NOISE-CORRECTED between-cell variances: each side's sampling "
                 "variance is subtracted, so a smaller frame inflates the correction rather "
                 "than the estimate"),
    Meter("cond.spread_ratio_raw.t1_3", "the same ratio UNCORRECTED and unclamped", "turn 1-3",
          True, "UNCORRECTED: the noise it does not subtract grows as the cells shrink — the "
                "cflabels arm's own value moved +0.1177 [+0.0131, +0.3546] between its full and "
                "its matched frame (2026-09-09 matched-quota read)"),
    Meter("cond.spread_delta.t1_3", "sd(V) - sd(outcome), noise-corrected", "turn 1-3",
          False, "a difference of noise-corrected spreads; stable in point and label across "
                 "every frame size measured"),
    Meter("cond.spread_ratio.all",
          "between-opponent spread ratio sd(V)/sd(outcome), noise-corrected", "all states",
          False, "as `cond.spread_ratio.t1_3`"),
    Meter("cond.spread_ratio_raw.all", "the same ratio UNCORRECTED and unclamped", "all states",
          True, "as `cond.spread_ratio_raw.t1_3`"),
    Meter("cond.spread_delta.all", "sd(V) - sd(outcome), noise-corrected", "all states",
          False, "as `cond.spread_delta.t1_3`"),
    # ---- the LATE WINDOWS, added 2026-09-10 (tool v6). The N-curve established that the
    # opponent is UNOBSERVABLE at turn 1 on a matched-team frame, so a turn-1(-3) ratio is a
    # weak question; these are the windows where the question has an answer.
    Meter(SPREAD_RATIO_T4_10,
          "between-opponent spread ratio sd(V)/sd(outcome), noise-corrected",
          f"turn {OBSERVABLE_TURNS[0]}-{OBSERVABLE_TURNS[1]}", False,
          "as `cond.spread_ratio.t1_3` — the same noise-corrected estimator on the window the "
          "opponent first becomes OBSERVABLE in (N-curve §3: pooled -> class AUC 0.50 at t1, "
          "0.67 over t1-3, 0.82 over t4-10)"),
    Meter(SPREAD_RATIO_T11_24,
          "between-opponent spread ratio sd(V)/sd(outcome), noise-corrected",
          f"turn {MID_TURNS[0]}-{MID_TURNS[1]}", False,
          "as `cond.spread_ratio.t1_3` — the same noise-corrected estimator on the N-curve's own "
          "headline window"),
    Meter("cond.elo_slope", "slope of bias (V - true win rate) on opponent Elo, per 100 Elo",
          "all states", False,
          "an OLS over the CELLS, of which there are as many as the pinned roster has "
          "opponents — a number no trace quota changes (it moved by 0.0001 across every rung of "
          "the 2026-09-09 frame-size curve)"),
    Meter("cond.own_team_r2.t1", "own-team leave-one-battle-out win-rate R^2 of V", "turn 1",
          True, "an out-of-fold score of a ridge FIT on the frame: measured at 62 / 102 / 176 / "
                "353 / 429 decoder battles on ONE arm it reads -0.025 / +0.004 / +0.037 / "
                "+0.068 / +0.060"),
    Meter("cond.own_team_r2.all", "own-team leave-one-battle-out win-rate R^2 of V", "all states",
          True, "as `cond.own_team_r2.t1`"),
    Meter("cond.opp_class_auc.t1", "opponent-CLASS (pool vs bot) AUC of V", "turn 1",
          True, "also an out-of-fold FIT (`grouped_oof_scalar`, task=auc). It was STABLE across "
                "every rung of the 2026-09-09 curve — a 1-D monotone decoder's AUC is nearly the "
                "AUC of V itself — but it is a fit on the frame, so it is matched by the same "
                "rule rather than exempted by an observation"),
    Meter(OPP_CLASS_AUC_T1_3, "opponent-CLASS (pool vs bot) AUC of V",
          "turn 1-3", True, "as `cond.opp_class_auc.t1`"),
    Meter(OPP_CLASS_AUC_T4_10, "opponent-CLASS (pool vs bot) AUC of V",
          f"turn {OBSERVABLE_TURNS[0]}-{OBSERVABLE_TURNS[1]}", True,
          "as `cond.opp_class_auc.t1`. 🚨 THIS IS THE ROW THE QUESTION HAS AN ANSWER ON: the "
          "N-curve measured the opponent to be unobservable at turn 1 (class AUC 0.502 / 0.508 "
          "from `value_pooled` itself on a matched-team frame) and observable by turns 4-10 "
          "(0.82), where the ONLINE head already reads 0.723 / 0.700 against a null of 0.52"),
    # ---- the OPTIMAL-SPREAD reference family, added 2026-09-10 (tool v6). Descriptive.
    Meter(OPT_RATIO["t1_3"],
          "OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above)",
          "turn 1-3", True,
          "an out-of-fold binned posterior FIT on the frame (`oof_opt_value`), so its expectation "
          "moves with the number of battles the bins are estimated from — the first mechanism, "
          "exactly as for `cond.opp_class_auc.t1`",
          True, OPT_PROVISIONAL_WHY),
    Meter(OPT_RATIO["t4_10"],
          "OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above)",
          f"turn {OBSERVABLE_TURNS[0]}-{OBSERVABLE_TURNS[1]}", True,
          f"as `{OPT_RATIO['t1_3']}`", True, OPT_PROVISIONAL_WHY),
    Meter(OPT_RATIO["t11_24"],
          "OPPONENT-DECODABLE between-opponent spread ratio — sum_o q(o|V)*p_o, noise-corrected (NOT a bound on the row above)",
          f"turn {MID_TURNS[0]}-{MID_TURNS[1]}", True,
          f"as `{OPT_RATIO['t1_3']}`", True, OPT_PROVISIONAL_WHY),
    # ---- the (A)/(B) separation, added 2026-09-10 (arm 8's own-team decode). Detail and the
    # sign table: `main.ops.team_conditioning`.
    Meter("cond.within_team_resolution.all",
          "WITHIN-own-team Murphy resolution of V (cells >= MIN_TEAM_BATTLES battles, "
          "battle-weighted over teams)", "all states, <=2 per battle", True,
          "a BINNED second moment computed INSIDE cells of a handful of episodes: each bin's "
          "observed rate departs from its cell's base rate by chance alone, and that inflation "
          "grows as the cells shrink — the same mechanism that makes an UNCORRECTED spread "
          "frame-sensitive, plus the threshold-selected cell set"),
    Meter("cond.within_stratum_resolution.all",
          "the same resolution WITHIN team-STRENGTH strata (quantiles of the team's LOO win rate)",
          "all states, <=2 per battle", True,
          "the COARSE companion: hundreds of episodes per cell make the small-cell inflation far "
          "smaller, but it is the same estimator on the same threshold-selected teams and is "
          "matched by the same rule rather than exempted by an argument"),
    Meter("cond.team_spread_ratio.t1_3",
          "BETWEEN-team spread ratio sd(mean V) / sd(team win rate), noise-corrected",
          "turn 1-3", True,
          "its CELLS are chosen by a battle-count threshold (MIN_TEAM_BATTLES), so frame size "
          "decides WHICH teams are cells at all — the third mechanism, which the pinned opponent "
          "roster does not have"),
    Meter("cond.team_spread_ratio_raw.t1_3", "the same ratio UNCORRECTED and unclamped",
          "turn 1-3", True,
          "UNCORRECTED *and* on a threshold-selected cell set — both mechanisms at once"),
    Meter("cond.own_team_r2.late", "own-team leave-one-battle-out win-rate R^2 of V",
          f"turn >= {LATE_TURN}", True,
          "as `cond.own_team_r2.t1` — an out-of-fold score of a ridge FIT on the frame"),
    Meter(OWN_TEAM_R2_DIFF, "own-team R^2 at turn 1 MINUS own-team R^2 late",
          f"turn 1 - turn >= {LATE_TURN}", True,
          "a difference of two out-of-fold FITS on the same frame; both move with frame size, so "
          "the difference is matched rather than assumed to cancel",
          True,
          "NO FLOOR EXISTS FOR THIS ROW AND THE CONTROLS CANNOT SUPPLY ONE — every control reads "
          "own-team R^2 ~= 0 at turn 1, so there is nothing for it to FALL from and the two-draw "
          "replicate floor cannot be formed. The first replicate arm supplies it. A LARGE move "
          "either way is informative; a small one is not; and this row is never DETECTED"),
    # ---- the CALIBRATION SLOPE pair, added 2026-09-10 (the SHRINKAGE hypothesis for arm 8's
    # "alignment up, amplitude down"). Arithmetic and the lever-arm hazard: `main.ops.
    # calibration_slope`.
    Meter(CALIB_SLOPE_ALL,
          "calibration SLOPE — weighted logistic regression of the outcome on logit(V) "
          "(1 = correctly dispersed, >1 = SHRUNK, <1 = over-dispersed)",
          f"all states, <={STATES_PER_BATTLE_CAP} per battle", False,
          "an M-estimator — the root of a weighted score equation over the states, whose "
          "expectation is the population coefficient at every frame size given correct weights. "
          "It is not an out-of-fold score whose optimism grows with the fitting set (there is no "
          "held-out evaluation), not an uncorrected second moment (a coefficient is a ratio of "
          "moments, consistent, not a variance with unsubtracted noise), and its states are not a "
          "threshold-selected cell set (every state with a finite V qualifies). The MLE's own "
          "O(1/n) small-sample bias is the one mechanism that touches it, and it is two orders "
          "below the frame differences on this ladder — CHECKED on a real pair, arm 8 vs "
          "ctrl10M_b, matched against unmatched (see the ledger entry)"),
    Meter(CALIB_INTERCEPT_ALL,
          "calibration-in-the-large — the INTERCEPT of that regression (0 = calibrated)",
          f"all states, <={STATES_PER_BATTLE_CAP} per battle", False,
          f"as `{CALIB_SLOPE_ALL}` — the other coefficient of the same fit"),
    Meter(CALIB_SLOPE_T13,
          "calibration SLOPE — weighted logistic regression of the outcome on logit(V)",
          f"turn 1-3, <={STATES_PER_BATTLE_CAP} per battle", False,
          f"as `{CALIB_SLOPE_ALL}`"),
    Meter(CALIB_INTERCEPT_T13,
          "calibration-in-the-large — the INTERCEPT of that regression",
          f"turn 1-3, <={STATES_PER_BATTLE_CAP} per battle", False,
          f"as `{CALIB_SLOPE_ALL}`"),
    Meter(CALIB_SLOPE_WITHIN,
          "calibration SLOPE with a free intercept PER own-team STRENGTH STRATUM — the "
          "dispersion reading INSIDE a stratum rather than across teams",
          f"all states, <={STATES_PER_BATTLE_CAP} per battle, stratum fixed effects", True,
          "the estimator itself is frame-size neutral, but its STATES are the ones sitting in a "
          f"stratum, and a stratum exists only over teams clearing MIN_TEAM_BATTLES="
          f"{MIN_TEAM_BATTLES} — the third mechanism (a threshold-selected cell set, so frame "
          "size decides WHICH teams contribute at all), exactly as for the within-stratum "
          "resolution row"),
    Meter(CALIB_SLOPE_COMMON,
          "calibration SLOPE on the COMMON SUPPORT — both sides restricted to the intersection "
          "of their central 95% of V, so the lever arm sd(logit V) cannot differ between them",
          f"all states, <={STATES_PER_BATTLE_CAP} per battle, common V window", False,
          f"as `{CALIB_SLOPE_ALL}`. It is additionally a PAIR-level row: the window is a function "
          "of BOTH sides, so it is fitted by `main.ops.critic_read` from the cached per-state "
          "columns after both readouts exist, and there is no single-run frame for a quota match "
          "to subsample", pair_level=True),
    Meter(CALIB_INTERCEPT_COMMON,
          "calibration-in-the-large on the COMMON SUPPORT",
          f"all states, <={STATES_PER_BATTLE_CAP} per battle, common V window", False,
          f"as `{CALIB_SLOPE_COMMON}` — the other coefficient of the same fit", pair_level=True),
)
#: the legacy 3-tuple view, kept so the committed measurement scripts that import ``METERS``
#: (``measurements/critic_ladder_reads/.../matched_quota/analyze.py``) keep working unchanged — a
#: committed measurement must stay reproducible from the artifacts beside it.
METERS: Tuple[Tuple[str, str, str], ...] = tuple(
    (m.key, m.quantity, m.stratum) for m in METER_SPECS)
METER_KEYS = tuple(m.key for m in METER_SPECS)
#: the rows a quota-matched read must recompute on an equalised frame.
FRAME_SENSITIVE_KEYS: Tuple[str, ...] = tuple(
    m.key for m in METER_SPECS if m.frame_sensitive)
#: the rows that may never carry a registered verdict (see :class:`Meter`).
PROVISIONAL_KEYS: Tuple[str, ...] = tuple(m.key for m in METER_SPECS if m.provisional)
#: the rows only a PAIR can compute (see :class:`Meter`) — a single-run block never emits them.
PAIR_LEVEL_KEYS: Tuple[str, ...] = tuple(m.key for m in METER_SPECS if m.pair_level)
METER_BY_KEY: Dict[str, Meter] = {m.key: m for m in METER_SPECS}


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
                       frame: Optional[Tuple[np.ndarray, Dict[str, Any]]] = None,
                       say: Callable[[str], None] = lambda _m: None) -> Dict[str, Any]:
    """Every conditioning meter for ONE run at ONE cycle, with its raw bootstrap draws.

    Returns ``{"points": {key: value}, "_draws": {key: ndarray}, "frame": …, "omitted": …}``. A
    meter this cycle cannot support (no sentinel, no strength axis, one opponent class) is listed
    in ``omitted`` with the REASON — never emitted as a NaN row that reads like a measurement.

    ``frame`` injects an ALREADY-EXTRACTED ``(arr, meta)`` instead of reading the trace tree —
    which is how :mod:`main.ops.quota_match` reads a SUBSAMPLE of this cycle without copying or
    symlinking a single file. ``run_dir`` still names the real run, so the strength axis (a
    function of the snapshot ladder and the manifest's true win rates, neither of which a
    subsample touches) is unaffected.
    """
    trace_dir = os.path.join(run_dir, "eval_traces", f"step_{int(step)}")
    arr, meta = extract_cycle(trace_dir) if frame is None else frame
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

    # ---- OWN-TEAM CELLS. Which teams are cells at all (>= MIN_TEAM_BATTLES battles), their
    # strength strata, and the per-battle code every within-cell row is computed on. The cell SET
    # is fixed HERE, on the full frame, and the bootstrap below resamples the states inside it —
    # the same convention the out-of-fold decoders follow, which price the sampling noise in a
    # held-out score rather than the variability of refitting.
    teams_u, tinv_b = np.unique(b["team"], return_inverse=True)
    n_teams_all = int(teams_u.size)
    tW = np.bincount(tinv_b, weights=b["w"], minlength=n_teams_all)
    tWy = np.bincount(tinv_b, weights=b["w"] * b["y"], minlength=n_teams_all)
    tN = np.bincount(tinv_b, minlength=n_teams_all).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        team_rate = np.where(tW > 0, tWy / np.where(tW > 0, tW, 1.0), np.nan)
    team_ok = (tN >= MIN_TEAM_BATTLES) & np.isfinite(team_rate)
    team_code_b = np.where(team_ok[tinv_b], tinv_b, -1)
    strat_of_team = TC.strata_of(team_rate, tN, team_ok, TC.TEAM_STRATA)
    strat_code_b = np.where(team_ok[tinv_b], strat_of_team[tinv_b], -1)

    scores: Dict[str, Dict[str, Any]] = {}
    for key, task, y_state, mask in (
            ("cond.own_team_r2.t1", "r2", y_wr_state,
             (arr["turn"] == 1) & np.isfinite(y_wr_state)),
            ("cond.own_team_r2.late", "r2", y_wr_state,
             (arr["turn"] >= LATE_TURN) & np.isfinite(y_wr_state)),
            ("cond.own_team_r2.all", "r2", y_wr_state, np.isfinite(y_wr_state)),
            ("cond.opp_class_auc.t1", "auc", y_cls_state, arr["turn"] == 1),
            (OPP_CLASS_AUC_T1_3, "auc", y_cls_state, arr["turn"] <= 3),
            (OPP_CLASS_AUC_T4_10, "auc", y_cls_state,
             (arr["turn"] >= OBSERVABLE_TURNS[0]) & (arr["turn"] <= OBSERVABLE_TURNS[1]))):
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

    # ---- WITHIN-CELL RESOLUTION frames: the states of the qualifying cells, capped per battle.
    resolutions: Dict[str, Dict[str, Any]] = {}
    for key, code_b, n_c, what in (
            ("cond.within_team_resolution.all", team_code_b, n_teams_all, "own-team"),
            ("cond.within_stratum_resolution.all", strat_code_b, TC.TEAM_STRATA,
             "team-strength stratum")):
        mask = code_b[binv] >= 0
        idx = _cap_states(arr, mask, STATES_PER_BATTLE_CAP, seed)
        n_b_cells = int(np.unique(arr["battle"][idx]).size) if idx.size else 0
        if idx.size < 20 or n_b_cells < 5:
            omitted[key] = (f"only {idx.size} states / {n_b_cells} battles sit in a {what} cell "
                            f"clearing MIN_TEAM_BATTLES={MIN_TEAM_BATTLES} at this cycle — too "
                            "few for a within-cell resolution.")
            continue
        resolutions[key] = {"idx": idx, "code_b": code_b, "n_cells": n_c,
                            "v": arr["V"][idx], "y": arr["y"][idx], "w": arr["w"][idx],
                            "cell": code_b[binv[idx]],
                            "census": TC.cell_census(code_b[binv[idx]], arr["battle"][idx], n_c)}

    # ---- THE OPTIMAL-SPREAD REFERENCE. One out-of-fold posterior per window, FIT ONCE on the
    # full frame exactly as the decoders above are (the bootstrap then resamples the battles
    # inside the fitted columns, pricing the sampling noise in a held-out quantity rather than
    # the variability of refitting). `p_o` is the cell's own manifest TRUE win rate — the same
    # quantity the DENOMINATOR of every spread ratio is built from, so the reference and the row
    # it is read beside sit on one footing.
    opt_cols: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    n_b_all = int(b["y"].size)
    for window, _row in OPT_WINDOWS:
        wmask = np.isfinite(arr["V"]) & np.isfinite(arr["true_wr"])
        wmask &= {"t1_3": arr["turn"] <= 3,
                  "t4_10": ((arr["turn"] >= OBSERVABLE_TURNS[0])
                            & (arr["turn"] <= OBSERVABLE_TURNS[1])),
                  "t11_24": ((arr["turn"] >= MID_TURNS[0])
                             & (arr["turn"] <= MID_TURNS[1]))}[window]
        widx = np.where(wmask)[0]
        if widx.size < 20 or np.unique(arr["battle"][widx]).size < 5:
            omitted[OPT_RATIO[window]] = (
                f"only {widx.size} states / "
                f"{np.unique(arr['battle'][widx]).size if widx.size else 0} battles sit in turns "
                f"{window.replace('t', '').replace('_', '-')} at this cycle — too few for a "
                "grouped out-of-fold posterior.")
            continue
        vopt = oof_opt_value(arr["V"][widx], arr["true_wr"][widx], arr["w"][widx],
                             arr["battle"][widx], seed=seed)
        ok = np.isfinite(vopt)
        if int(ok.sum()) < 20:
            omitted[OPT_RATIO[window]] = ("the out-of-fold posterior produced no usable column "
                                          "on this cycle.")
            continue
        keep_idx = widx[ok]
        opt_cols[window] = (
            np.bincount(binv[keep_idx], minlength=n_b_all).astype(float),
            np.bincount(binv[keep_idx], weights=vopt[ok], minlength=n_b_all))

    # ---- THE CALIBRATION SLOPE frames. Per-state columns for the two turn windows, capped per
    # battle exactly as the within-cell rows are, plus each state's own-team STRENGTH STRATUM (for
    # the within-stratum companion) and each battle's OPPONENT cell (the bootstrap resamples
    # battles WITHIN the pinned roster, as every other interval here does). The payload is
    # returned on the block so `main.ops.critic_read` can re-fit the COMMON-SUPPORT companion for
    # a PAIR without re-reading either trace tree — that window is a function of both sides.
    calib_buckets: Dict[str, Dict[str, np.ndarray]] = {}
    for bucket, bmask in (("all", np.ones(arr.size, bool)), ("t1_3", arr["turn"] <= 3)):
        idx = _cap_states(arr, bmask & np.isfinite(arr["V"]), STATES_PER_BATTLE_CAP, seed)
        n_b_cal = int(np.unique(arr["battle"][idx]).size) if idx.size else 0
        if idx.size < 20 or n_b_cal < 5:
            continue
        calib_buckets[bucket] = {"v": arr["V"][idx], "y": arr["y"][idx], "w": arr["w"][idx],
                                 "battle": binv[idx], "stratum": strat_code_b[binv[idx]]}
    calib_payload = (CS.payload(n_battles=int(b["y"].size), opp_of_battle=cid,
                                buckets=calib_buckets) if calib_buckets else None)
    calib = CS.block(calib_payload, names=CALIB_NAMES, boot=boot, seed=seed)
    for key in (CALIB_SLOPE_ALL, CALIB_SLOPE_T13, CALIB_SLOPE_WITHIN,
                CALIB_INTERCEPT_ALL, CALIB_INTERCEPT_T13):
        if key not in calib["points"]:
            omitted.setdefault(key, (
                "the weighted logistic fit of the outcome on logit(V) is degenerate on this cycle "
                "— one outcome class, too few states, or a separated fit whose coefficient "
                "diverges. A diverging slope is never reported as a number."))

    def _cell_weight(res: Dict[str, Any], battles: np.ndarray) -> np.ndarray:
        """Per-cell BATTLE count over ``battles`` (battle row indices, duplicates counted) — the
        weight the mean over cells uses, so a team with more episodes counts for more."""
        codes = res["code_b"][battles]
        codes = codes[codes >= 0]
        return np.bincount(codes, minlength=res["n_cells"]).astype(float)

    # ---- point estimates
    sel0 = np.arange(b["y"].size)
    st_all0 = cell_stats(b, sel0, cid, n_cells, "all", true_wr)
    st_t13_0 = cell_stats(b, sel0, cid, n_cells, "t1_3", true_wr)
    keep0 = st_all0["n_battles"] > 0
    sc_all0 = spread_corrected(st_all0, keep0, true_wr, n_games)
    sc_t13_0 = spread_corrected(st_t13_0, keep0, true_wr, n_games)
    sc_late0 = {w: spread_corrected(cell_stats(b, sel0, cid, n_cells, w, true_wr),
                                    keep0, true_wr, n_games) for w in ("t4_10", "t11_24")}
    sc_opt0 = {w: spread_corrected(
        cell_stats(b, sel0, cid, n_cells, w, true_wr, cols=opt_cols[w]),
        keep0, true_wr, n_games) for w in opt_cols}
    points: Dict[str, float] = {
        "cond.spread_ratio.t1_3": sc_t13_0["ratio"],
        "cond.spread_ratio_raw.t1_3": sc_t13_0["ratio_raw"],
        "cond.spread_delta.t1_3": sc_t13_0["delta"],
        "cond.spread_ratio.all": sc_all0["ratio"],
        "cond.spread_ratio_raw.all": sc_all0["ratio_raw"],
        "cond.spread_delta.all": sc_all0["delta"],
        "cond.elo_slope": ols_slope(st_all0["bias"], strength, keep0),
        SPREAD_RATIO_T4_10: sc_late0["t4_10"]["ratio"],
        SPREAD_RATIO_T11_24: sc_late0["t11_24"]["ratio"],
    }
    for w in opt_cols:
        points[OPT_RATIO[w]] = sc_opt0[w]["ratio"]
    for key, s in scores.items():
        points[key] = score(s["y"], s["pred"], s["w"], s["task"])
    if all(np.isfinite(points.get(k, np.nan))
           for k in ("cond.own_team_r2.t1", "cond.own_team_r2.late")):
        points[OWN_TEAM_R2_DIFF] = (points["cond.own_team_r2.t1"]
                                    - points["cond.own_team_r2.late"])
    else:
        omitted.setdefault(OWN_TEAM_R2_DIFF,
                           "one side of the contrast is missing at this cycle — the turn-1 and "
                           "the late own-team decode must BOTH exist for their difference to "
                           "mean anything.")
    for key, res in resolutions.items():
        points[key] = TC.cellwise_resolution(res["v"], res["y"], res["w"], res["cell"],
                                             _cell_weight(res, np.unique(binv[res["idx"]])),
                                             res["n_cells"])
    ts0 = TC.team_spread(w=b["w"][sel0], n_states=b["n_t1_3"][sel0], sum_v=b["sV_t1_3"][sel0],
                         y=b["y"][sel0], cell=team_code_b[sel0], n_cells=n_teams_all,
                         keep_cell=team_ok)
    points["cond.team_spread_ratio.t1_3"] = ts0["ratio"]
    points["cond.team_spread_ratio_raw.t1_3"] = ts0["ratio_raw"]
    points.update(calib["points"])

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
    for key, s in list(scores.items()) + list(resolutions.items()):
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
        for w in ("t4_10", "t11_24"):
            draws[f"cond.spread_ratio.{w}"].append(
                spread_corrected(cell_stats(b, sel, cid_sel, n_cells, w, wr),
                                 kp, wr, n_games)["ratio"])
        for w in opt_cols:
            draws[OPT_RATIO[w]].append(spread_corrected(
                cell_stats(b, sel, cid_sel, n_cells, w, wr, cols=opt_cols[w]),
                kp, wr, n_games)["ratio"])
        draws["cond.spread_ratio_raw.all"].append(sa["ratio_raw"])
        draws["cond.spread_delta.all"].append(sa["delta"])
        draws["cond.elo_slope"].append(ols_slope(st_a["bias"], strength, kp))
        ts = TC.team_spread(w=b["w"][sel], n_states=b["n_t1_3"][sel], sum_v=b["sV_t1_3"][sel],
                            y=b["y"][sel], cell=team_code_b[sel], n_cells=n_teams_all,
                            keep_cell=team_ok)
        draws["cond.team_spread_ratio.t1_3"].append(ts["ratio"])
        draws["cond.team_spread_ratio_raw.t1_3"].append(ts["ratio_raw"])
        cur: Dict[str, float] = {}
        for key, s in list(scores.items()) + list(resolutions.items()):
            flat, start, cnt = gathers[key]
            c = cnt[sel]
            tot = int(c.sum())
            if tot == 0:
                continue
            base = np.repeat(start[sel], c)
            within = np.arange(tot) - np.repeat(np.cumsum(c) - c, c)
            g = flat[base + within]
            if key in scores:
                cur[key] = score(s["y"][g], s["pred"][g], s["w"][g], s["task"])
            else:
                # the cell weights are the DRAWN battle instances per cell, so a team drawn twice
                # counts twice — the resample's own composition, not the original frame's.
                cur[key] = TC.cellwise_resolution(
                    s["v"][g], s["y"][g], s["w"][g], s["cell"][g],
                    _cell_weight(s, sel[c > 0]), s["n_cells"])
            draws[key].append(cur[key])
        if ("cond.own_team_r2.t1" in cur) and ("cond.own_team_r2.late" in cur):
            # PAIRED: both scores come from the SAME resampled battles, so the difference carries
            # their covariance instead of pretending the two are independent draws.
            draws[OWN_TEAM_R2_DIFF].append(cur["cond.own_team_r2.t1"]
                                           - cur["cond.own_team_r2.late"])

    out_draws = {k: np.asarray([d for d in v if np.isfinite(d)], dtype=float)
                 for k, v in draws.items() if k in points and np.isfinite(points.get(k, np.nan))}
    out_draws.update(calib["draws"])
    for k in METER_KEYS:
        # 🚨 the COMMON-SUPPORT rows are PAIR-level and are fitted by `main.ops.critic_read` after
        # both sides exist. Listing them as omitted here would put a false reason in every
        # single-run block, and the reason would still be there after the pair filled them in.
        if k in PAIR_LEVEL_KEYS:
            continue
        if k not in points or not np.isfinite(points[k]):
            omitted.setdefault(k, "the point estimate is undefined on this cycle "
                                  "(too few cells or a degenerate column).")
    return {
        "points": {k: float(v) for k, v in points.items() if np.isfinite(v)},
        "_draws": out_draws,
        "_calib": calib_payload,
        "omitted": omitted,
        "frame": {**{k: v for k, v in meta.items() if k != "opponents"},
                  "n_cells": int(keep0.sum()), "boot": int(boot), "seed": int(seed),
                  "states_per_battle_cap": STATES_PER_BATTLE_CAP,
                  "min_team_battles": MIN_TEAM_BATTLES,
                  "late_turn": LATE_TURN,
                  "n_team_cells": int(team_ok.sum()),
                  "n_teams_seen": n_teams_all,
                  "team_strata": TC.TEAM_STRATA,
                  "score_frames": {k: {"n_states": s["n_states"], "n_battles": s["n_battles"]}
                                   for k, s in scores.items()},
                  # 🚨 the CENSUS every within-cell row is read WITH: cells of a handful of
                  # episodes make a binned resolution mostly its own noise, so the row is not
                  # quotable without these counts beside it.
                  "cell_frames": {k: dict(r["census"], n_cells_declared=int(r["n_cells"]))
                                  for k, r in resolutions.items()},
                  # 🚨 the LEVER ARM every calibration-slope row is read WITH: the slope's SE
                  # scales as 1/sd(logit V), so a compressed head buys a wider interval from the
                  # very effect under test. Never printed without it.
                  "calibration_support": calib["support"],
                  # 🚨 the OPTIMAL reference is read WITH its own ratio's partner: V's ratio in
                  # the same window. `optimal` carries both, so "0.05 optimal" is never quoted
                  # without the number it is the ceiling for.
                  "optimal_reference": {
                      w: {"ratio_V": (float(points[row]) if np.isfinite(points.get(row, np.nan))
                                      else None),
                          "ratio_optimal": (float(points[OPT_RATIO[w]])
                                            if np.isfinite(points.get(OPT_RATIO[w], np.nan))
                                            else None),
                          "n_bins": OPT_BINS, "n_folds": OPT_FOLDS,
                          "sd_V": (float(sc_opt0[w]["sd_V"]) if w in sc_opt0 else None),
                          "sd_y": (float(sc_opt0[w]["sd_y"]) if w in sc_opt0 else None)}
                      for w, row in OPT_WINDOWS},
                  "team_spread_frame": {k: v for k, v in ts0.items()
                                        if k in ("n_cells", "n_battles",
                                                 "median_battles_per_cell", "sd_V", "sd_y",
                                                 "noise_V", "noise_y")},
                  "recorded_v_note": recorded_v_note()},
        "spread": {"t1_3": sc_t13_0, "all": sc_all0, "team_t1_3": ts0,
                   **{w: sc_late0[w] for w in sc_late0},
                   "optimal": {w: sc_opt0[w] for w in sc_opt0}},
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
