"""The ride-along reader's METERS (`gen3_ridealong_read_v1`) — pure NumPy over per-row columns.

Every function here reads arrays the reader already computed (V, the policy's probabilities and
logits, the heads' readout) plus the bank's stamps or the Lane S ground truth. Nothing loads a model.

(i)   :func:`stratified_uncertainty` — does a disagreement / novelty score predict V's ACTUAL error
      |V − z| on the bank (z = 1 win, 0 loss; DRAWS EXCLUDED, see :data:`DRAW_POLICY`)? AUROC for
      |V − z| > 0.5 ("V's rounded prediction was the wrong outcome"), Spearman, the mean error in each
      of the 10 score deciles; all rows, and split by opponent class and by game phase.
(ii)  :func:`within_turn_reads` — a per-action score (A's member mean, or the policy's own logits as
      the reference row) against the ground truth's per-action values on the truth turns.
(iii) :func:`adv_std_flags` — is A's member spread HIGHER on the near-best actions the policy starves?
(iv)  :func:`truth_value_error` — disagreement / novelty against |V − V_truth| on the truth turns.

95 % intervals are percentile bootstraps that resample BATTLES (turns of one battle are correlated),
with a fixed seed, so a read is deterministic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from agents.training.instrumented_ppo.ridealong_terms import (
    ERR_THRESHOLD,
    STARVED_PI,
    error_by_decile,
    rank_auroc,
    spearman,
)

#: A drawn battle has no win bit: its rows are EXCLUDED from every |V − z| read. The win-prob critic
#: trained a tie as a loss (0), but a draw in the bank is mostly a stall / turn-limit truncation,
#: where "V should have said 0" is not what the critic was asked to learn about the position. The
#: all-rows stratum also carries a draws-as-loss sensitivity row (``draws_as_loss``).
DRAW_POLICY = "exclude"
#: |V − V_truth| threshold for (iv)'s AUROC. V_truth is a 64-seed mean, continuous in [0, 1], so the
#: bank's 0.5 ("the wrong outcome") is almost never reached there; 0.25 is a quarter of the range.
TRUTH_ERR_THRESHOLD = 0.25
#: Ground-truth near-best tolerance on the ±1 scale (Lane S's ε; see ``main.policy_spectrum.truth``).
EPS = 0.1
Z = 1.96
N_BOOT = 1000
BOOT_SEED = 20260930
TOPK = 2
#: Quantile bins of V's own entropy inside which a score's AUROC is re-read (the de-confounded read:
#: a std of member PROBABILITIES is mechanically largest where the members' logits sit near 0, i.e.
#: where V itself is near 0.5 — so the raw AUROC can ride V's own uncertainty).
REF_BINS = 5


# ---------------------------------------------------------------------------------------------
# bank-level helpers
# ---------------------------------------------------------------------------------------------

def outcome_target(outcomes: Sequence[str], draws: str = DRAW_POLICY) -> tuple:
    """``(z [N] float, keep [N] bool)``: win → 1, loss → 0; a draw is dropped (``exclude``) or
    scored as a loss (``loss``, the critic's own training convention)."""
    o = np.asarray(outcomes)
    z = (o == "win").astype(np.float64)
    if draws == "exclude":
        keep = o != "draw"
    elif draws == "loss":
        keep = np.ones(len(o), dtype=bool)
    else:
        raise ValueError(f"draws must be 'exclude' or 'loss', got {draws!r}")
    bad = ~np.isin(o, ("win", "loss", "draw"))
    if bad.any():
        raise ValueError(f"unknown outcome(s) {sorted(set(o[bad].tolist()))}")
    return z, keep


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """V's OWN uncertainty (nats) — the reference any epistemic score is read beside."""
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


def cluster_boot_auroc(score: np.ndarray, label: np.ndarray, clusters: Any,
                       n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> Optional[List[float]]:
    """95 % percentile interval of :func:`rank_auroc`, resampling CLUSTERS (battles), B = ``n_boot``."""
    from main.ridealong_read.boot import boot_rank_stats, ci, n_clusters

    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    if len(score) == 0 or label.all() or (~label).all() or n_clusters(clusters) < 2:
        return None
    return ci(boot_rank_stats(score, clusters, label=label, n_boot=n_boot, seed=seed)["auroc"])


def _r(x: Optional[float], nd: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), nd)


def conditional_auroc(score: np.ndarray, label: np.ndarray, ref: np.ndarray,
                      bins: int = REF_BINS) -> Optional[float]:
    """AUROC of ``score`` for ``label`` WITHIN each quantile bin of ``ref``, averaged weighted by the
    bin's row count (bins with one class only are skipped). 0.5 = ``score`` carries nothing about
    ``label`` beyond what ``ref`` already sorts."""
    score, ref = np.asarray(score, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    if len(score) < bins * 2:
        return None
    order = np.argsort(ref, kind="stable")
    tot, w = 0.0, 0
    for chunk in np.array_split(order, bins):
        a = rank_auroc(score[chunk], label[chunk])
        if a is not None:
            tot += a * len(chunk)
            w += len(chunk)
    return tot / w if w else None


def cluster_boot_conditional_auroc(score: np.ndarray, label: np.ndarray, ref: np.ndarray,
                                   clusters: Any, n_boot: int = N_BOOT,
                                   seed: int = BOOT_SEED) -> Optional[List[float]]:
    """95 % percentile interval of :func:`conditional_auroc`, resampling CLUSTERS (battles) with
    replacement, B = ``n_boot`` (the quintiles are re-cut inside each resample). X26's R1 decides on
    this interval's lower bound."""
    cl = np.asarray(clusters)
    uniq, inv = np.unique(cl, return_inverse=True)
    if len(uniq) < 2:
        return None
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    score, label, ref = (np.asarray(score, dtype=np.float64), np.asarray(label, dtype=bool),
                         np.asarray(ref, dtype=np.float64))
    vals: List[float] = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        a = conditional_auroc(score[idx], label[idx], ref[idx])
        if a is not None:
            vals.append(a)
    if len(vals) < 0.5 * n_boot:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [_r(lo), _r(hi)]


def score_vs_error(score: np.ndarray, err: np.ndarray, clusters: Any,
                   threshold: float = ERR_THRESHOLD, boot: bool = True,
                   ref: Optional[np.ndarray] = None, cond_ci: bool = True) -> dict:
    """Does ``score`` predict ``err``? Point estimates with 95 % battle-clustered bootstrap
    intervals (B = 1000, :mod:`main.ridealong_read.boot`): AUROC for ``err > threshold``, Spearman,
    and the top-decile ÷ bottom-decile mean error; plus the mean error in each of the 10 score
    deciles (lowest score first). With ``ref`` (V's own entropy), also the AUROC WITHIN ``ref``'s
    quintiles (:func:`conditional_auroc`) with its own clustered interval — X26's R1 meter."""
    from main.ridealong_read.boot import boot_rank_stats, ci, n_clusters

    score = np.asarray(score, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    lab = err > threshold
    dec = error_by_decile(score, err) if len(err) >= 10 else None
    ratio = (dec[-1] / dec[0]) if dec and dec[0] > 0 else None
    out: dict = {"n": int(len(err)), "n_err": int(lab.sum()), "threshold": threshold,
                 "mean_err": _r(err.mean()) if len(err) else None,
                 "auroc": _r(rank_auroc(score, lab)) if len(err) else None,
                 "spearman": _r(spearman(score, err)),
                 "decile_ratio_top_over_bottom": _r(ratio),
                 "err_by_decile": [_r(x) for x in dec] if dec else None,
                 "auroc_ci": None, "spearman_ci": None, "decile_ratio_ci": None}
    if boot and len(err) >= 10 and n_clusters(clusters) >= 2:
        d = boot_rank_stats(score, clusters, label=lab, err=err)
        out["auroc_ci"] = ci(d["auroc"])
        out["spearman_ci"] = ci(d["spearman"])
        out["decile_ratio_ci"] = ci(d["decile_ratio"])
    if ref is not None:
        out["auroc_within_ref_quintiles"] = _r(conditional_auroc(score, lab, ref))
        out["auroc_within_ref_quintiles_ci"] = (
            cluster_boot_conditional_auroc(score, lab, ref, clusters)
            if boot and cond_ci and len(err) >= 10 else None)
    return out


def stratified_uncertainty(scores: Mapping[str, np.ndarray], err: np.ndarray, clusters: Any,
                           strata: Mapping[str, Sequence[str]],
                           threshold: float = ERR_THRESHOLD,
                           ref: Optional[np.ndarray] = None) -> dict:
    """Meter (i): every score against ``err`` on all rows, then within each value of each stratum
    (``ref``, when given, adds the within-``ref``-quintile AUROC to every cell)."""
    cl_arr = np.asarray(clusters)
    out: dict = {"all": {k: score_vs_error(s, err, cl_arr, threshold, ref=ref)
                         for k, s in scores.items()}}
    for name, labels in strata.items():
        lab = np.asarray(labels)
        out[name] = {}
        for val in sorted(set(lab.tolist())):
            sel = lab == val
            out[name][val] = {k: score_vs_error(s[sel], err[sel], cl_arr[sel], threshold,
                                                ref=None if ref is None else ref[sel],
                                                cond_ci=False)
                              for k, s in scores.items()}
    return out


# ---------------------------------------------------------------------------------------------
# truth turns
# ---------------------------------------------------------------------------------------------

@dataclass
class TruthTurn:
    """One ground-truth turn: its legal actions, their values (±1 scale, the mean over the truth's
    seeds), the gap to the best, and Lane S's classes (``main.policy_spectrum.truth.turn_truth``)."""
    id: str
    battle: str
    row: int                  # index into the bank's decision order (and every per-row column)
    acts: np.ndarray          # [A] int
    v: np.ndarray             # [A] truth value, ±1
    gap: np.ndarray
    near: np.ndarray          # gap ≤ ε
    dom: np.ndarray           # gap − 1.96·SE > ε

    @property
    def decisive(self) -> bool:
        return bool(self.dom.any())

    @property
    def vstar(self) -> float:
        return float(self.v.max())


def truth_turns(truth_rows: Sequence[dict], id_to_row: Mapping[str, int],
                battle_of: Mapping[str, str], eps: float = EPS) -> List[TruthTurn]:
    """Every OK truth row whose decision is in ``id_to_row`` (a ``--limit-battles`` read keeps only
    its own battles' turns), in the truth file's order."""
    out = []
    for r in truth_rows:
        if not r.get("ok") or r["id"] not in id_to_row:
            continue
        acts = sorted(int(a) for a in r["outcomes"])
        o = np.array([r["outcomes"][str(a)] for a in acts], dtype=np.float64)
        v = o.mean(axis=1)
        diff = o[int(np.argmax(v))][None, :] - o
        gap = diff.mean(axis=1)
        s = o.shape[1]
        se = diff.std(axis=1, ddof=1) / np.sqrt(s) if s > 1 else np.zeros(len(acts))
        out.append(TruthTurn(id=r["id"], battle=battle_of[r["id"]], row=int(id_to_row[r["id"]]),
                             acts=np.array(acts), v=v, gap=gap, near=gap <= eps,
                             dom=gap - Z * se > eps))
    return out


def tie_jitter(tid: str, acts: np.ndarray) -> np.ndarray:
    """A fixed per-(turn, action) jitter of < 1e-9 so an exact tie breaks the same way every read."""
    return np.array([int(hashlib.sha256(f"{tid}:{int(a)}".encode()).hexdigest()[:8], 16)
                     / 2 ** 32 * 1e-9 for a in acts])


def _topk(score: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros(len(score), dtype=bool)
    m[np.argsort(-score, kind="stable")[:k]] = True
    return m


def _boot_stat(boot: Any, vals: Sequence[float], cl: Sequence[str]) -> dict:
    return dict(boot.stat(np.asarray(vals, dtype=np.float64), list(cl)))


def within_turn_reads(turns: Sequence[TruthTurn], score: np.ndarray, pi: np.ndarray, boot: Any) -> dict:
    """Meter (ii) for one per-action ``score`` [N, 11] (higher = better) against the truth.

    - ``spearman``: the mean within-turn Spearman ρ of score vs truth value over the legal actions
      (turns with ≥ 3 actions and a non-constant score and truth; all turns);
    - ``argmax_regret`` (all / decisive turns): V* − v(argmax score), ±1 scale;
    - ``argmax_near``: the share of turns whose argmax is near-best;
    - ``starved_top2`` / ``fed_top2`` (decisive turns): per near-best action the policy STARVES
      (π < 1 %) / FEEDS, is it in the score's top 2? ``chance`` = the mean of min(2, n) / n."""
    rho, rho_cl = [], []
    reg, reg_cl, reg_d, reg_d_cl, near1, near1_cl = [], [], [], [], [], []
    st, st_ch, st_cl, fd, fd_ch, fd_cl = [], [], [], [], [], []
    for T in turns:
        s = score[T.row, T.acts].astype(np.float64) + tie_jitter(T.id, T.acts)
        p = pi[T.row, T.acts]
        r = spearman(s, T.v) if len(T.acts) >= 3 and np.ptp(T.v) > 0 else None
        if r is not None:
            rho.append(r)
            rho_cl.append(T.battle)
        a = int(np.argmax(s))
        reg.append(T.vstar - float(T.v[a]))
        reg_cl.append(T.battle)
        near1.append(float(T.near[a]))
        near1_cl.append(T.battle)
        if not T.decisive:
            continue
        reg_d.append(T.vstar - float(T.v[a]))
        reg_d_cl.append(T.battle)
        top = _topk(s, TOPK)
        chance = min(TOPK, len(T.acts)) / len(T.acts)
        for j in np.flatnonzero(T.near):
            if p[j] < STARVED_PI:
                st.append(float(top[j]))
                st_ch.append(chance)
                st_cl.append(T.battle)
            else:
                fd.append(float(top[j]))
                fd_ch.append(chance)
                fd_cl.append(T.battle)
    return {
        "turns": len(turns), "decisive": int(sum(T.decisive for T in turns)),
        "spearman": _boot_stat(boot, rho, rho_cl),
        "argmax_regret_all": _boot_stat(boot, reg, reg_cl),
        "argmax_regret_decisive": _boot_stat(boot, reg_d, reg_d_cl),
        "argmax_near": _boot_stat(boot, near1, near1_cl),
        "starved_top2": {**_boot_stat(boot, st, st_cl),
                         "chance": _r(float(np.mean(st_ch))) if st_ch else None},
        "fed_top2": {**_boot_stat(boot, fd, fd_cl),
                     "chance": _r(float(np.mean(fd_ch))) if fd_ch else None},
    }


def paired_spearman_diff(turns: Sequence[TruthTurn], score_a: np.ndarray, score_b: np.ndarray,
                         boot: Any) -> dict:
    """The PAIRED within-turn Spearman difference ρ(a) − ρ(b) on the turns where both are defined,
    with the battle-clustered interval on the SAME cluster draws (``Boot.diff``)."""
    ra, rb, cl = [], [], []
    for T in turns:
        if len(T.acts) < 3 or np.ptp(T.v) == 0:
            continue
        j = tie_jitter(T.id, T.acts)
        a = spearman(score_a[T.row, T.acts].astype(np.float64) + j, T.v)
        b = spearman(score_b[T.row, T.acts].astype(np.float64) + j, T.v)
        if a is None or b is None:
            continue
        ra.append(a)
        rb.append(b)
        cl.append(T.battle)
    return {"n": len(ra), **boot.diff((np.array(ra), cl), (np.array(rb), cl))}


def adv_std_flags(turns: Sequence[TruthTurn], adv_std: np.ndarray, pi: np.ndarray, boot: Any) -> dict:
    """Meter (iii) on the DECISIVE truth turns: A's member spread on each legal action, grouped as
    starved near-best / fed near-best / starved not-near / fed not-near. AUROCs of the spread for
    starved-near vs every other legal action, and — the control that separates "good and unplayed"
    from merely "unplayed" — starved-near vs starved-not-near."""
    groups: Dict[str, List[float]] = {k: [] for k in ("starved_near", "fed_near", "starved_other",
                                                      "fed_other")}
    cl: Dict[str, List[str]] = {k: [] for k in groups}
    for T in turns:
        if not T.decisive:
            continue
        sd = adv_std[T.row, T.acts]
        p = pi[T.row, T.acts]
        for j in range(len(T.acts)):
            starved = p[j] < STARVED_PI
            key = ("starved_" if starved else "fed_") + ("near" if T.near[j] else "other")
            groups[key].append(float(sd[j]))
            cl[key].append(T.battle)
    out: dict = {k: _boot_stat(boot, groups[k], cl[k]) for k in groups}

    def _au(pos: str, negs: Sequence[str]) -> dict:
        sc = np.array(groups[pos] + [x for n in negs for x in groups[n]])
        lab = np.array([True] * len(groups[pos]) + [False] * (len(sc) - len(groups[pos])),
                       dtype=bool)
        cls = cl[pos] + [c for n in negs for c in cl[n]]
        return {"auroc": _r(rank_auroc(sc, lab)) if len(sc) else None,
                "ci": cluster_boot_auroc(sc, lab, cls) if len(sc) else None,
                "n_pos": int(lab.sum()), "n_neg": int((~lab).sum())}

    out["auroc_starved_near_vs_other_legal"] = _au("starved_near",
                                                   ("fed_near", "starved_other", "fed_other"))
    out["auroc_starved_near_vs_starved_other"] = _au("starved_near", ("starved_other",))
    return out


def bank_starved_spread(adv_std: np.ndarray, pi: np.ndarray, masks: np.ndarray) -> dict:
    """The truth-free complement of (iii) on every bank row: A's spread on legal actions the policy
    starves (π < 1 %) vs feeds (the training-time ``ridealong/adv_std_*`` pair, offline)."""
    legal = np.asarray(masks, dtype=bool)
    starved = legal & (pi < STARVED_PI)
    fed = legal & (pi >= STARVED_PI)
    return {"starved_mean": _r(adv_std[starved].mean()) if starved.any() else None,
            "fed_mean": _r(adv_std[fed].mean()) if fed.any() else None,
            "n_starved": int(starved.sum()), "n_fed": int(fed.sum()),
            "auroc_starved_vs_fed": _r(rank_auroc(np.concatenate([adv_std[starved], adv_std[fed]]),
                                                  np.r_[np.ones(starved.sum(), bool),
                                                        np.zeros(fed.sum(), bool)]))}


def greedy_truth_value(turns: Sequence[TruthTurn], logits: np.ndarray) -> np.ndarray:
    """Per turn, the truth's STATE value under its own greedy continuation, on [0, 1]: the truth
    value of the continuation policy's GREEDY root action, (v + 1) / 2 (a tie scores 0.5). Only
    valid when ``logits`` ARE the continuation policy's (the checkpoint read against its OWN
    continuation) — the truth rows carry per-action values, never a state value."""
    out = np.empty(len(turns))
    for k, T in enumerate(turns):
        s = logits[T.row, T.acts].astype(np.float64) + tie_jitter(T.id, T.acts)
        out[k] = (float(T.v[int(np.argmax(s))]) + 1.0) / 2.0
    return out


def truth_value_error(turns: Sequence[TruthTurn], v: np.ndarray, logits: np.ndarray,
                      scores: Mapping[str, np.ndarray]) -> dict:
    """Meter (iv): |V − V_truth| on the truth turns (V_truth from :func:`greedy_truth_value`) and
    whether each per-row ``score`` predicts it."""
    if not turns:
        return {"turns": 0}
    vt = greedy_truth_value(turns, logits)
    rows = np.array([T.row for T in turns])
    cl = np.array([T.battle for T in turns])
    vv = v[rows].astype(np.float64)
    err = np.abs(vv - vt)
    ref = scores.get("ref_v_entropy")
    return {"turns": len(turns), "v_truth_definition": "greedy continuation's root action value",
            "mean_signed_v_minus_truth": _r(float((vv - vt).mean())),
            **{k: score_vs_error(s[rows], err, cl, TRUTH_ERR_THRESHOLD,
                                 ref=None if ref is None or k == "ref_v_entropy" else ref[rows])
               for k, s in scores.items()}}


__all__ = ["DRAW_POLICY", "TRUTH_ERR_THRESHOLD", "EPS", "TruthTurn", "outcome_target",
           "binary_entropy", "cluster_boot_auroc", "conditional_auroc", "score_vs_error", "stratified_uncertainty",
           "truth_turns", "tie_jitter", "within_turn_reads", "paired_spearman_diff", "adv_std_flags",
           "bank_starved_spread", "greedy_truth_value", "truth_value_error"]
