"""The X4 PRE-READ readout: does the critic's one-ply Q̂_V see the moves the policy starves?

Inputs: the Q̂ rows (:mod:`main.policy_spectrum.qhat`, per checkpoint, 64 seeds that ARE Lane S
truth's), the ground truth (``truth_v2`` compact rows, three greedy continuations, 64 seeds), and the
policies' probabilities on the bank (the baseline read's ``<label>.probs.npz``).

⚠️ **Held-out truth.** Q̂'s seed k rolls the same first-turn dice as truth playout k, so comparing a
Q̂ built on seeds 0..S−1 against a truth that INCLUDES those seeds shares their noise (a lucky seed
lifts both) and flatters agreement. Every headline read therefore uses truth on seeds the Q̂ did not
use: Q̂ on seeds 0..7 (S = 8) against truth on seeds 8..63 (56 seeds); the S-curve uses truth on
seeds 32..63 for every S ≤ 32. The Lane-S-exact sets (all 64 seeds) are reported as a leak check.

Definitions (Lane S's, on the ±1 scale, ε = 0.1): NEAR-BEST = truth gap to the best ≤ ε; DOMINATED
= gap − 1.96·SE > ε; DECISIVE turn = ≥ 1 dominated action; STARVED = near-best with policy mass
< 1 %. For Q̂: "Q̂-near" = Q̂ best − Q̂(a) ≤ ε; top-k by Q̂ (ties broken by a fixed per-(turn, action)
hash jitter of 1e-9).

Reads per (checkpoint, continuation, opponent variant): (a) within-turn Spearman ρ and top-1;
(b) Q̂-near rate of starved near-best vs non-starved near-best vs dominated actions, with the
within-turn CHANCE rate; (c) the dominated false-positive rate; (d) by move category; (e) the
teacher gain on starved turns; the owner's regret-weighted reads — top-k recall, the regret of
V-derived choice rules (argmax, uniform top-2 / top-3, softmax(Q̂/τ)) against the policy's own
argmax and distribution, false positives by truth margin, paired-vs-absolute error and the
within-turn correlation of V's errors, and terminal/V mixing; plus per-branch CALIBRATION of V on
the exact successor states the truth playouts continued from (matched checkpoint, variant M).
95 % intervals: percentile bootstrap resampling BATTLES (≤ 4 turns per battle) — clusters of turns.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from main.policy_spectrum.categories import CATEGORIES
from main.policy_spectrum.truth import EPS, STARVE, Z, load_rows

N_BOOT = 1000
BOOT_SEED = 20260930
TAUS = (0.01, 0.03, 0.1, 0.3)
GAP_BINS = ((0.1, 0.2), (0.2, 0.4), (0.4, 9.0))
CKPTS = {  # Q̂ label -> (policy-probs label, truth continuation)
    "K2final": ("K2@91.1M", "K2final greedy (both sides)"),
    "N0final75M": ("N0@75.0M", "N0final75M greedy (both sides)"),
    "C_fixfinal": ("C_fix@83.1M", "C_fixfinal greedy (both sides)"),
}


def _jit(tid: str, a: int) -> float:
    return int(hashlib.sha256(f"{tid}:{a}".encode()).hexdigest()[:8], 16) / 2 ** 32 * 1e-9


# ---------------------------------------------------------------------------------------------
# turn assembly
# ---------------------------------------------------------------------------------------------

def load_qhat_rows(path: Path) -> Dict[Tuple[str, str], dict]:
    """(id, variant) -> the LAST ok row (a repaired turn is appended after its refusal)."""
    from main.policy_spectrum.qhat import load_qhat

    out: Dict[Tuple[str, str], dict] = {}
    refused: Dict[Tuple[str, str], dict] = {}
    for r in load_qhat(path):
        k = (r["id"], r["variant"])
        if r["ok"]:
            out[k] = r
            refused.pop(k, None)
        elif k not in out:
            refused[k] = r
    out.update({("REFUSED", f"{k[0]}|{k[1]}"): v for k, v in refused.items()})
    return out


def q_matrix(row: dict, acts: Sequence[int]) -> Tuple[np.ndarray, List[str]]:
    q = np.array([row["actions"][str(a)]["q"] for a in acts], dtype=np.float64)
    src = [row["actions"][str(a)]["src"] for a in acts]
    return q, src


def assemble(bank, truth_rows: Sequence[dict], qrows: Dict[Tuple[str, str], dict], probs: np.ndarray,
             variant: str, q_seeds: Sequence[int], t_seeds: Sequence[int]) -> List[dict]:
    """One record per turn with both an ok truth row and an ok Q̂ row of ``variant`` (M falls back
    to R where the root had no open opponent decision — there the two are identical)."""
    idx = {d["id"]: i for i, d in enumerate(bank.decisions)}
    out = []
    qs, ts = np.array(q_seeds), np.array(t_seeds)
    for tr in truth_rows:
        if not tr.get("ok"):
            continue
        tid = tr["id"]
        qr = qrows.get((tid, variant)) or (qrows.get((tid, "R")) if variant == "M" else None)
        if variant == "M" and qr is not None and qr["variant"] == "R" and qr.get("other_open"):
            qr = None                                    # the M row is missing: never substitute R
        if qr is None:
            continue
        acts = sorted(int(a) for a in tr["outcomes"])
        if sorted(int(a) for a in qr["actions"]) != acts:
            raise ValueError(f"{tid}: Q̂ actions {sorted(qr['actions'])} != truth {acts}")
        O = np.array([tr["outcomes"][str(a)] for a in acts], dtype=np.float64)[:, ts]
        Qm, src = q_matrix(qr, acts)
        Q = Qm[:, qs]
        v = O.mean(axis=1)
        best = int(np.argmax(v))
        diff = O[best][None, :] - O
        gap = diff.mean(axis=1)
        se = diff.std(axis=1, ddof=1) / np.sqrt(O.shape[1])
        i = idx[tid]
        d = bank.decisions[i]
        q = Q.mean(axis=1)
        jit = np.array([_jit(tid, a) for a in acts])
        out.append({
            "id": tid, "battle": d["battle"], "kind": d["kind"], "acts": np.array(acts),
            "cats": np.array([d["cats"][str(a)] for a in acts]),
            "pi": probs[i, acts], "v": v, "vstar": float(v[best]), "gap": gap, "se": se,
            "near": gap <= EPS, "strict": gap + Z * se <= EPS, "dom": gap - Z * se > EPS,
            "q": q, "qj": q + jit, "Qs": Q, "Os": O, "src": src,
            "other_open": bool(qr.get("other_open")), "qrow": qr,
        })
        out[-1]["decisive"] = bool(out[-1]["dom"].any())
    return out


# ---------------------------------------------------------------------------------------------
# bootstrap over battle clusters
# ---------------------------------------------------------------------------------------------

class Boot:
    """Shared cluster draws so differences of two means get a joint interval."""

    def __init__(self, battles: Sequence[str]):
        self.uniq = sorted(set(battles))
        self.pos = {b: i for i, b in enumerate(self.uniq)}
        rng = np.random.default_rng(BOOT_SEED)
        self.draws = rng.integers(0, len(self.uniq), size=(N_BOOT, len(self.uniq)))
        self.counts = np.stack([np.bincount(d, minlength=len(self.uniq)) for d in self.draws]).astype(np.float64)

    def _sums(self, vals: np.ndarray, cl: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        inv = np.array([self.pos[c] for c in cl], dtype=np.int64)
        s = np.bincount(inv, weights=vals, minlength=len(self.uniq))
        n = np.bincount(inv, minlength=len(self.uniq)).astype(np.float64)
        return s, n

    def dist(self, vals, cl) -> np.ndarray:
        vals = np.asarray(vals, dtype=np.float64)
        s, n = self._sums(vals, cl)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.counts @ s) / (self.counts @ n)

    def stat(self, vals, cl) -> dict:
        vals = np.asarray(vals, dtype=np.float64)
        if len(vals) == 0:
            return {"n": 0, "mean": None, "ci": None}
        b = self.dist(vals, cl)
        lo, hi = np.nanpercentile(b, [2.5, 97.5])
        return {"n": int(len(vals)), "mean": round(float(vals.mean()), 4), "ci": [round(float(lo), 4), round(float(hi), 4)]}

    def diff(self, a: Tuple, b: Tuple) -> dict:
        """mean(a) − mean(b), each (vals, clusters), on the SAME cluster draws."""
        if len(a[0]) == 0 or len(b[0]) == 0:
            return {"delta": None, "ci": None}
        d = self.dist(*a) - self.dist(*b)
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return {"delta": round(float(np.mean(a[0]) - np.mean(b[0])), 4), "ci": [round(float(lo), 4), round(float(hi), 4)]}


# ---------------------------------------------------------------------------------------------
# per-turn primitives
# ---------------------------------------------------------------------------------------------

def topk_mask(score: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-score, kind="stable")
    m = np.zeros(len(score), dtype=bool)
    m[order[:min(k, len(score))]] = True
    return m


def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    z = (x - x.max()) / tau
    e = np.exp(z)
    return e / e.sum()


def rule_regrets(T: dict) -> Dict[str, float]:
    """V* − E[truth value] of each choice rule (truth on held-out seeds)."""
    v, q, pi = T["v"], T["qj"], T["pi"]
    out = {"q_argmax": T["vstar"] - v[int(np.argmax(q))]}
    for k in (2, 3):
        m = topk_mask(q, k)
        out[f"q_top{k}_uniform"] = T["vstar"] - float(v[m].mean())
    for tau in TAUS:
        out[f"q_softmax_{tau}"] = T["vstar"] - float((softmax(T["q"], tau) * v).sum())
    pj = pi + np.array([_jit(T["id"], int(a)) for a in T["acts"]])
    out["pi_argmax"] = T["vstar"] - v[int(np.argmax(pj))]
    pn = pi / pi.sum()
    out["pi_dist"] = T["vstar"] - float((pn * v).sum())
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    from scipy.stats import spearmanr

    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


# ---------------------------------------------------------------------------------------------
# the reads
# ---------------------------------------------------------------------------------------------

def read(turns: Sequence[dict], B: Boot, cats: bool = True) -> dict:
    out: dict = {"turns": len(turns), "decisive": int(sum(T["decisive"] for T in turns))}
    # (a) rank agreement + top-1
    rho, rho_b, top1, top1n, top1_b = [], [], [], [], []
    for T in turns:
        r = spearman(T["q"], T["v"])
        if r is not None:
            rho.append(r)
            rho_b.append(T["battle"])
        am = int(np.argmax(T["qj"]))
        top1.append(float(T["v"][am] == T["vstar"]))
        top1n.append(float(T["near"][am]))
        top1_b.append(T["battle"])
    out["a_spearman"] = B.stat(rho, rho_b)
    out["a_top1_is_truth_best"] = B.stat(top1, top1_b)
    out["a_top1_is_near_best"] = B.stat(top1n, top1_b)
    # (1) top-k recall
    for k in (1, 2, 3):
        rb, rn, bb = [], [], []
        for T in turns:
            m = topk_mask(T["qj"], k)
            rb.append(float((m & (T["v"] == T["vstar"])).any()))
            rn.append(float((m & T["near"]).any()))
            bb.append(T["battle"])
        out[f"recall_top{k}_truth_best"] = B.stat(rb, bb)
        out[f"recall_top{k}_any_near"] = B.stat(rn, bb)
    # action-level groups on DECISIVE turns: (b), (c), per category
    groups = {"starved_near": [], "fed_near": [], "dominated": []}
    for T in turns:
        if not T["decisive"]:
            continue
        qbest = T["q"].max()
        qnear = (qbest - T["q"]) <= EPS
        chance = float(qnear.mean())
        n = len(T["q"])
        tk = {k: topk_mask(T["qj"], k) for k in (1, 2, 3)}
        for j in range(n):
            if T["near"][j]:
                g = "starved_near" if T["pi"][j] < STARVE else "fed_near"
            elif T["dom"][j]:
                g = "dominated"
            else:
                continue
            groups[g].append({"b": T["battle"], "cat": T["cats"][j], "qnear": float(qnear[j]), "chance": chance,
                              "top1": float(tk[1][j]), "top2": float(tk[2][j]), "top3": float(tk[3][j]),
                              "ch2": min(2, n) / n, "ch3": min(3, n) / n, "gap": float(T["gap"][j])})
    for g, items in groups.items():
        cl = [x["b"] for x in items]
        out[f"b_{g}"] = {m: B.stat([x[m] for x in items], cl)
                         for m in ("qnear", "chance", "top1", "top2", "ch2", "top3", "ch3")}
    sn, dm, fn = groups["starved_near"], groups["dominated"], groups["fed_near"]
    out["b_starved_minus_dominated_qnear"] = B.diff(([x["qnear"] for x in sn], [x["b"] for x in sn]),
                                                    ([x["qnear"] for x in dm], [x["b"] for x in dm]))
    out["b_starved_minus_chance_qnear"] = B.diff(([x["qnear"] for x in sn], [x["b"] for x in sn]),
                                                 ([x["chance"] for x in sn], [x["b"] for x in sn]))
    out["b_starved_minus_fed_qnear"] = B.diff(([x["qnear"] for x in sn], [x["b"] for x in sn]),
                                              ([x["qnear"] for x in fn], [x["b"] for x in fn]))
    out["b_starved_minus_dominated_top2"] = B.diff(([x["top2"] for x in sn], [x["b"] for x in sn]),
                                                   ([x["top2"] for x in dm], [x["b"] for x in dm]))
    # (3) false positives by truth margin: EVERY action worse than the best by more than ε (not only
    # the separable ones — "dominated" needs gap > ε + 1.96·SE ≈ 0.3 at 56 seeds, which would leave
    # the 0.1–0.2 bin empty), on every turn; the separably-dominated subset is reported beside it
    fp_items = []
    for T in turns:
        n = len(T["q"])
        tk = {k: topk_mask(T["qj"], k) for k in (2, 3)}
        qnear = (T["q"].max() - T["q"]) <= EPS
        for j in range(n):
            if T["gap"][j] > EPS:
                fp_items.append({"b": T["battle"], "gap": float(T["gap"][j]), "dom": bool(T["dom"][j]),
                                 "top2": float(tk[2][j]), "top3": float(tk[3][j]), "qnear": float(qnear[j]),
                                 "ch2": min(2, n) / n, "ch3": min(3, n) / n})
    for lo, hi in GAP_BINS:
        for sub, it in (("", [x for x in fp_items if lo < x["gap"] <= hi]),
                        ("_separable", [x for x in fp_items if lo < x["gap"] <= hi and x["dom"]])):
            cl = [x["b"] for x in it]
            out[f"fp_gap_{lo}_{hi if hi < 9 else 'inf'}{sub}"] = {
                m: B.stat([x[m] for x in it], cl) for m in ("top2", "ch2", "top3", "ch3", "qnear")}
    # (d) categories
    if cats:
        out["by_category"] = {}
        for c in CATEGORIES:
            row = {}
            for g, items in groups.items():
                it = [x for x in items if x["cat"] == c]
                cl = [x["b"] for x in it]
                row[g] = {m: B.stat([x[m] for x in it], cl) for m in ("qnear", "chance", "top2", "ch2", "top3")}
            # regret of the rules on decisive turns where THIS category holds a starved near-best action
            st = [T for T in turns if T["decisive"] and any(
                T["near"][j] and T["pi"][j] < STARVE and T["cats"][j] == c for j in range(len(T["q"])))]
            rr = [rule_regrets(T) for T in st]
            row["starved_turn_regret"] = {k: B.stat([r[k] for r in rr], [T["battle"] for T in st])
                                          for k in ("q_argmax", "q_top2_uniform", "q_softmax_0.1", "pi_argmax", "pi_dist")}
            out["by_category"][c] = row
    # would a Q̂-derived target FEED the starved actions? On starved turns: the mass each target puts
    # on the turn's STARVED near-best actions (the policy puts < 1 % on each), against uniform
    feed: Dict[str, list] = {"pi": [], "uniform": [], "q_top2_uniform": [], "q_top3_uniform": [],
                             **{f"q_softmax_{t}": [] for t in TAUS}}
    fb = []
    for T in turns:
        st = T["near"] & (T["pi"] < STARVE)
        if not (T["decisive"] and st.any()):
            continue
        n = len(T["q"])
        feed["pi"].append(float(T["pi"][st].sum() / T["pi"].sum()))
        feed["uniform"].append(float(st.sum() / n))
        for k in (2, 3):
            m = topk_mask(T["qj"], k)
            feed[f"q_top{k}_uniform"].append(float((m & st).sum() / m.sum()))
        for t in TAUS:
            feed[f"q_softmax_{t}"].append(float(softmax(T["q"], t)[st].sum()))
        fb.append(T["battle"])
    out["starved_mass_under_target"] = {k: B.stat(v, fb) for k, v in feed.items()}
    # (2) regret of the rules, on every turn / decisive / starved turns; (e) teacher gain
    sets = {"all": list(turns), "decisive": [T for T in turns if T["decisive"]],
            "starved": [T for T in turns if T["decisive"] and (T["near"] & (T["pi"] < STARVE)).any()]}
    out["regret"] = {}
    for name, ts in sets.items():
        rr = [rule_regrets(T) for T in ts]
        cl = [T["battle"] for T in ts]
        out["regret"][name] = {k: B.stat([r[k] for r in rr], cl) for k in (rr[0] if rr else {})}
        if rr:
            out["regret"][name]["gain_q_argmax_vs_pi_argmax"] = B.diff(
                ([r["pi_argmax"] for r in rr], cl), ([r["q_argmax"] for r in rr], cl))
            out["regret"][name]["gain_q_argmax_vs_pi_dist"] = B.diff(
                ([r["pi_dist"] for r in rr], cl), ([r["q_argmax"] for r in rr], cl))
            for tau in TAUS:
                out["regret"][name][f"gain_q_softmax_{tau}_vs_pi_dist"] = B.diff(
                    ([r["pi_dist"] for r in rr], cl), ([r[f"q_softmax_{tau}"] for r in rr], cl))
    return out


def errors(turns: Sequence[dict], B: Boot) -> dict:
    """(4) absolute vs paired error, the within-turn correlation of V's errors, the noise floor."""
    e_all, e_b, e_abs = [], [], []
    between, within = [], []
    pair_err, pair_b, pair_floor, pair_x = [], [], [], []
    for T in turns:
        e = T["q"] - T["v"]
        e_all += list(e)
        e_abs += list(np.abs(e))
        e_b += [T["battle"]] * len(e)
        n = len(e)
        if n >= 2:
            between.append(e.mean())
            within += list(e - e.mean())
        Qd, Od = T["Qs"], T["Os"]
        for a in range(n):
            for b in range(a + 1, n):
                pair_err.append(abs((T["q"][a] - T["q"][b]) - (T["v"][a] - T["v"][b])))
                pair_b.append(T["battle"])
                sq = np.var(Qd[a] - Qd[b], ddof=1) / Qd.shape[1] if Qd.shape[1] > 1 else 0.0
                so = np.var(Od[a] - Od[b], ddof=1) / Od.shape[1]
                pair_floor.append(sq + so)
                pair_x.append((e[a], e[b]))
    e_all_a = np.array(e_all)
    px = np.array(pair_x)
    corr = float(np.corrcoef(np.r_[px[:, 0], px[:, 1]], np.r_[px[:, 1], px[:, 0]])[0, 1]) if len(px) > 2 else None
    icc = float(np.var(between) / max(np.var(between) + np.var(within), 1e-12)) if between else None
    return {"abs_error": B.stat(e_abs, e_b), "signed_error": B.stat(e_all, e_b),
            "rms_abs_error": round(float(np.sqrt((e_all_a ** 2).mean())), 4),
            "pair_abs_error": B.stat(pair_err, pair_b),
            "rms_pair_error": round(float(np.sqrt(np.mean(np.square(pair_err)))), 4),
            # what the pair error would be if V's errors were INDEPENDENT across a turn's actions
            # (√2 × the error's spread about its global mean) vs FULLY common-mode (only the
            # within-turn residual survives the difference)
            "rms_pair_error_if_independent": round(float(np.sqrt(2 * np.var(e_all_a))), 4),
            "rms_pair_error_if_common_mode": round(float(np.sqrt(2 * np.mean(np.square(within)))), 4)
            if within else None,
            "rms_pair_noise_floor": round(float(np.sqrt(np.mean(pair_floor))), 4),
            "within_turn_error_corr": round(corr, 4) if corr is not None else None,
            "error_icc_turn_share": round(icc, 4) if icc is not None else None,
            "pairs": len(pair_err)}


def mixing(turns: Sequence[dict], B: Boot) -> dict:
    """(5) turns where some branches END (terminal reward) and others are scored by V: the signed
    error of V-scored actions against the truth, split by whether the action has terminal branches;
    and the per-branch signed error of V-scored branches (paired with the SAME seed's truth outcome —
    valid only when the successor IS the truth playout's, i.e. matched checkpoint, variant M)."""
    act_term, act_v, bt, bv = [], [], [], []
    for T in turns:
        has_term = [any(ch != "V" for ch in s) for s in T["qrow_src"]]
        if not any(has_term) or all(all(ch != "V" for ch in s) for s in T["qrow_src"]):
            continue
        e = T["q"] - T["v"]
        for j, h in enumerate(has_term):
            (act_term if h else act_v).append(e[j])
            (bt if h else bv).append(T["battle"])
    # per branch, paired with the same seed's truth outcome of that successor (all 64 seeds)
    bm, bmb, bu, bub = [], [], [], []
    for T in turns:
        srcs = T["qrow_src"]
        allsrc = "".join(srcs)
        mixed = "V" in allsrc and any(ch != "V" for ch in allsrc)
        for j, s_ in enumerate(srcs):
            for k, ch in enumerate(s_):
                if ch == "V":
                    (bm if mixed else bu).append(T["Qfull"][j, k] - T["Ofull"][j, k])
                    (bmb if mixed else bub).append(T["battle"])
    return {"branch_signed_error_V_on_mixed_turns": B.stat(bm, bmb),
            "branch_signed_error_V_on_unmixed_turns": B.stat(bu, bub),
            "mixed_turn_actions_with_terminal_branches": B.stat(act_term, bt),
            "mixed_turn_actions_all_V": B.stat(act_v, bv),
            "terminal_minus_V_signed_error": B.diff((act_term, bt), (act_v, bv))}


def calibration(turns: Sequence[dict], seeds: Sequence[int]) -> dict:
    """Per branch: V (as P(win)) against the SAME seed's truth outcome of that exact successor (win
    = 1, tie = ½, loss = 0). Only meaningful for the matched checkpoint under variant M."""
    p, y = [], []
    for T in turns:
        for j, s in enumerate(T["qrow_src"]):
            for k in seeds:
                if s[k] == "V":
                    p.append((T["Qfull"][j, k] + 1) / 2)
                    y.append((T["Ofull"][j, k] + 1) / 2)
    if not p:
        return {}
    p, y = np.array(p), np.array(y)
    bins = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(p, bins) - 1, 0, 9)
    table = []
    ece = 0.0
    for b in range(10):
        m = which == b
        if m.any():
            table.append({"bin": f"{bins[b]:.1f}-{bins[b + 1]:.1f}", "n": int(m.sum()),
                          "mean_V": round(float(p[m].mean()), 4), "win_rate": round(float(y[m].mean()), 4)})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"branches": int(len(p)), "mean_V": round(float(p.mean()), 4), "mean_outcome": round(float(y.mean()), 4),
            "brier": round(float(((p - y) ** 2).mean()), 4),
            "brier_const": round(float(((y.mean() - y) ** 2).mean()), 4),
            "ece": round(float(ece), 4), "table": table}


def attach_full(turns: List[dict], truth_rows: Sequence[dict]) -> None:
    tr = {r["id"]: r for r in truth_rows if r.get("ok")}
    for T in turns:
        acts = list(T["acts"])
        T["Ofull"] = np.array([tr[T["id"]]["outcomes"][str(a)] for a in acts], dtype=np.float64)
        T["Qfull"] = np.array([T["qrow"]["actions"][str(a)]["q"] for a in acts], dtype=np.float64)
        T["qrow_src"] = [T["qrow"]["actions"][str(a)]["src"] for a in acts]


def branch_facts(qrows: Dict[Tuple[str, str], dict]) -> dict:
    """Counts: turns, variants, refusals, other-open roots, terminal branches, opponent decisions
    answered before capture, switch-only next decisions."""
    f = {"R_ok": 0, "M_ok": 0, "refused": [], "other_open": 0, "branches": 0, "terminal": 0,
         "opp_answered_ge1_R": 0, "opp_answered_ge2_M": 0, "next_switch_only": 0, "mixed_turns_R": 0}
    for k, r in qrows.items():
        if k[0] == "REFUSED":
            f["refused"].append({"id": r["id"], "variant": r["variant"], "error": r.get("error", "")[:160]})
            continue
        f[f"{r['variant']}_ok"] += 1
        if r["variant"] == "R":
            f["other_open"] += int(bool(r.get("other_open")))
            srcs = []
            for a in r["actions"].values():
                f["branches"] += len(a["src"])
                f["terminal"] += sum(ch != "V" for ch in a["src"])
                f["opp_answered_ge1_R"] += sum(ch != "0" for ch in a["opp"])
                f["next_switch_only"] += a["sw"].count("1")
                srcs.append(a["src"])
            allsrc = "".join(srcs)
            f["mixed_turns_R"] += int("V" in allsrc and any(ch != "V" for ch in allsrc))
        else:
            f["opp_answered_ge2_M"] += sum(int(ch) >= 2 for a in r["actions"].values() for ch in a["opp"])
    return f


def timing_summary(qrows: Dict[Tuple[str, str], dict]) -> dict:
    """Per labelled decision (one turn: every legal action × S seeds), from the run's own stamps.
    ⚠️ The run had 3 workers sharing one batched model, so v_fwd includes waiting for a merged
    forward; the dedicated single-thread benchmark is the cost-model number."""
    ws = [r["timing"] for k, r in qrows.items() if k[0] != "REFUSED" and r["variant"] == "R" and "timing" in r]
    if not ws:
        return {}
    keys = ("sim_s", "opp_fwd_s", "v_fwd_s", "v_rows", "wall_s")
    return {k: round(float(np.mean([w.get(k, 0) for w in ws])), 5) for k in keys}


def sanity_summary(bank, path: Path, qrows: Dict[Tuple[str, str], dict]) -> dict:
    """The sanity check (``qhat.sanity_turn``): the PLAYED action under the battle's own dice, the
    opponent on its recorded answers, must reproduce the recorded next observation byte for byte
    (then V there is identical by construction); and Q̂_8 of the played action (variant R, seeds
    0..7) against V on that recorded next observation — equal in expectation over dice, not per turn."""
    rows: Dict[str, dict] = {}
    for x in Path(path).read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            rows[r["id"]] = r                            # the last row per turn (a repair appends)
    status: Dict[str, int] = {}
    for r in rows.values():
        k = r["status"] + ("" if r["status"] != "captured" else ("_byte_equal" if r["byte_equal"] else "_DIFFERS"))
        status[k] = status.get(k, 0) + 1
    cap = [r for r in rows.values() if r["status"] == "captured"]
    dd = {d["id"]: d for d in bank.decisions}
    qd, vr = [], []
    for r in cap:
        qr = qrows.get((r["id"], "R"))
        if qr is None:
            continue
        d = dd[r["id"]]
        a = next(k for k, t in d["tokens"].items() if t == d["played"])
        qd.append(float(np.mean(qr["actions"][str(a)]["q"][:8])))
        vr.append(2 * r["v_recorded"] - 1)
    qd_a, vr_a = np.array(qd), np.array(vr)
    return {"status": status,
            "max_abs_dV_branch_vs_recorded": max((abs(r["v_branch"] - r["v_recorded"]) for r in cap), default=None),
            "q8_played_minus_V_recorded_mean": round(float((qd_a - vr_a).mean()), 4) if len(qd) else None,
            "q8_played_vs_V_recorded_corr": round(float(np.corrcoef(qd_a, vr_a)[0, 1]), 4) if len(qd) > 2 else None,
            "q8_played_minus_V_recorded_mean_abs": round(float(np.abs(qd_a - vr_a).mean()), 4) if len(qd) else None,
            "turns": len(qd)}


def build(bank, truth_paths: Dict[str, Path], qhat_paths: Dict[str, Path], probs_dir: Path,
          sanity_paths: Optional[Dict[str, Path]] = None) -> dict:
    from main.policy_spectrum.reader import load_probs

    truth = {c: load_rows(p) for c, p in truth_paths.items()}
    q = {k: load_qhat_rows(p) for k, p in qhat_paths.items()}
    out: dict = {"schema": "gen3_x4_preread_readout_v1", "eps": EPS, "starve": STARVE, "n_boot": N_BOOT,
                 "headline": {}, "cross": {}, "s_curve": {}, "leak_check": {}, "errors": {}, "mixing": {},
                 "calibration": {}, "facts": {}, "timing_run": {}}
    held = list(range(8, 64))
    S8 = list(range(8))
    for ck, (plab, cont) in CKPTS.items():
        if ck not in q:
            continue
        probs = load_probs(probs_dir, plab, bank)
        out["facts"][ck] = branch_facts(q[ck])
        out["timing_run"][ck] = timing_summary(q[ck])
        if sanity_paths and ck in sanity_paths and sanity_paths[ck].exists():
            out.setdefault("sanity", {})[ck] = sanity_summary(bank, sanity_paths[ck], q[ck])
        for ct_ck, (_, ct) in CKPTS.items():
            rows = truth[ct]
            for var in ("R", "M"):
                turns = assemble(bank, rows, q[ck], probs, var, S8, held)
                B = Boot([T["battle"] for T in turns])
                key = f"{ck}|truth:{ct_ck}|{var}"
                if ct == cont:
                    out["headline"][key] = read(turns, B)
                    out["errors"][key] = errors(turns, B)
                    attach_full(turns, rows)
                    out["mixing"][key] = mixing(turns, B)
                    if var == "M":
                        out["calibration"][ck] = calibration(turns, range(64))
                        # the same, restricted to turns with no open opponent root (R ≡ M there)
                    # leak check: Lane S's 64-seed sets with Q̂'s own seeds INSIDE the truth
                    lt = assemble(bank, rows, q[ck], probs, var, S8, list(range(64)))
                    out["leak_check"][key] = read(lt, Boot([T["battle"] for T in lt]), cats=False)
                    # the S-curve: Q̂ on seeds 0..S−1 vs truth on 32..63
                    sc = out["s_curve"].setdefault(f"{ck}|{var}", {})
                    for s in (1, 2, 4, 8, 16, 32):
                        st = assemble(bank, rows, q[ck], probs, var, list(range(s)), list(range(32, 64)))
                        r = read(st, Boot([T["battle"] for T in st]), cats=False)
                        sc[s] = {k: r[k] for k in ("a_spearman", "a_top1_is_near_best", "recall_top2_any_near", "regret",
                                                   "b_starved_near", "b_dominated")}
                else:
                    r = read(turns, B, cats=False)
                    out["cross"][key] = {k: r[k] for k in (
                        "a_spearman", "a_top1_is_near_best", "recall_top2_any_near", "b_starved_near",
                        "b_dominated", "regret")}
    return out


def _cli(argv=None) -> int:
    import argparse

    from main.policy_spectrum.bank import load_bank

    ap = argparse.ArgumentParser(prog="python -m main.policy_spectrum.qhat_report")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--truth-dir", required=True, help="truth_v2/ (rows_<label>_S64.compact.jsonl.gz)")
    ap.add_argument("--qhat-dir", required=True, help="qhat_<label>_S64.jsonl[.gz]")
    ap.add_argument("--probs-dir", required=True, help="the baseline read's <label>.probs.npz")
    ap.add_argument("--out", required=True, help="the readout JSON")
    ap.add_argument("--md", default=None, help="also render the markdown tables here")
    a = ap.parse_args(argv)
    bank = load_bank(Path(a.bank))
    td, qd = Path(a.truth_dir), Path(a.qhat_dir)
    truth_paths = {c: td / f"rows_{k}_S64.compact.jsonl.gz" for k, (_, c) in CKPTS.items()}
    qhat_paths = {}
    for k in CKPTS:
        for suf in (".jsonl.gz", ".jsonl"):
            p = qd / f"qhat_{k}_S64{suf}"
            if p.exists():
                qhat_paths[k] = p
                break
    sanity_paths = {k: qd / f"sanity_{k}.jsonl" for k in CKPTS}
    res = build(bank, truth_paths, qhat_paths, Path(a.probs_dir), sanity_paths)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")
    if a.md:
        Path(a.md).write_text(render(res))
    print(f"[qhat_report] wrote {a.out}")
    return 0



# ---------------------------------------------------------------------------------------------
# the markdown readout
# ---------------------------------------------------------------------------------------------

def _f(s: Optional[dict], key: str = "mean") -> str:
    if not s or s.get(key) is None:
        return "—"
    ci = s.get("ci")
    return f"{s[key]:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{s[key]:.3f}"


def render(res: dict) -> str:
    L: List[str] = []
    ck = [k for k in CKPTS if any(h.startswith(k + "|") for h in res["headline"])]
    for var in ("R", "M"):
        L += [f"### Variant {var} — " + ("the opponent held at its RECORDED root action" if var == "R" else
                                         "the opponent's open root decision answered by the checkpoint's GREEDY "
                                         "choice (truth's own construction)"), ""]
        L += ["| checkpoint (matched truth) | turns / decisive | Spearman ρ | top-1 is near-best | top-2 holds a near-best |"
              " top-3 holds a near-best | top-2 holds the truth-best |", "|---|---|---|---|---|---|---|"]
        for c in ck:
            h = res["headline"].get(f"{c}|truth:{c}|{var}")
            if h:
                L.append(f"| {c} | {h['turns']} / {h['decisive']} | {_f(h['a_spearman'])} | {_f(h['a_top1_is_near_best'])} | "
                         f"{_f(h['recall_top2_any_near'])} | {_f(h['recall_top3_any_near'])} | {_f(h['recall_top2_truth_best'])} |")
        L += ["", "**(b)/(c) — does Q̂ see the action?** Decisive turns; share of actions Q̂ puts within ε of its "
              "own best (chance = the turn's share of Q̂-near actions), and inside Q̂'s top-2 (chance = 2/n):", "",
              "| checkpoint | group | n | Q̂-near | chance | in Q̂ top-2 | chance | in Q̂ top-3 |", "|---|---|---|---|---|---|---|---|"]
        for c in ck:
            h = res["headline"].get(f"{c}|truth:{c}|{var}")
            if not h:
                continue
            for g, name in (("starved_near", "STARVED near-best"), ("fed_near", "fed near-best"), ("dominated", "dominated")):
                b = h[f"b_{g}"]
                L.append(f"| {c} | {name} | {b['qnear']['n']} | {_f(b['qnear'])} | {_f(b['chance'])} | {_f(b['top2'])} | "
                         f"{_f(b['ch2'])} | {_f(b['top3'])} |")
            L.append(f"| {c} | starved − dominated (Q̂-near) | | {_f(h['b_starved_minus_dominated_qnear'], 'delta')} | | "
                     f"{_f(h['b_starved_minus_dominated_top2'], 'delta')} | | |")
        L += ["", "**Regret of each choice rule** (truth V* − truth value of the pick; lower is better) — on turns with a "
              "STARVED near-best action / on every DECISIVE turn:", "",
              "| checkpoint | set | Q̂ argmax | uniform Q̂ top-2 | uniform Q̂ top-3 | softmax τ=0.01 | τ=0.03 | τ=0.1 | τ=0.3 |"
              " policy argmax | policy distribution |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in ck:
            h = res["headline"].get(f"{c}|truth:{c}|{var}")
            if not h:
                continue
            for st in ("starved", "decisive"):
                r = h["regret"][st]
                L.append(f"| {c} | {st} (n={r['q_argmax']['n']}) | " + " | ".join(_f(r[k]) for k in (
                    "q_argmax", "q_top2_uniform", "q_top3_uniform", "q_softmax_0.01", "q_softmax_0.03", "q_softmax_0.1",
                    "q_softmax_0.3", "pi_argmax", "pi_dist")) + " |")
        L += ["", "**(e) the one-ply teacher's gain on STARVED turns** (truth value gained, positive = the Q̂ rule beats the policy):", "",
              "| checkpoint | Q̂ argmax vs policy argmax | Q̂ argmax vs policy dist | softmax τ=0.03 vs policy dist | τ=0.1 vs dist | τ=0.3 vs dist |",
              "|---|---|---|---|---|---|"]
        for c in ck:
            h = res["headline"].get(f"{c}|truth:{c}|{var}")
            if h:
                r = h["regret"]["starved"]
                L.append(f"| {c} | {_f(r['gain_q_argmax_vs_pi_argmax'], 'delta')} | {_f(r['gain_q_argmax_vs_pi_dist'], 'delta')} | "
                         f"{_f(r['gain_q_softmax_0.03_vs_pi_dist'], 'delta')} | {_f(r['gain_q_softmax_0.1_vs_pi_dist'], 'delta')} | "
                         f"{_f(r['gain_q_softmax_0.3_vs_pi_dist'], 'delta')} |")
        L += ["", "**Would a Q̂ target FEED the starved actions?** On starved turns, the mass on the turn's STARVED "
              "near-best actions under each target (the policy's own, uniform, and the Q̂-derived ones):", "",
              "| checkpoint | policy | uniform | Q̂ top-2 uniform | Q̂ top-3 uniform | softmax τ=0.01 | τ=0.03 | τ=0.1 | τ=0.3 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c in ck:
            h = res["headline"].get(f"{c}|truth:{c}|{var}")
            if h:
                m = h["starved_mass_under_target"]
                L.append(f"| {c} | " + " | ".join(_f(m[k]) for k in (
                    "pi", "uniform", "q_top2_uniform", "q_top3_uniform", "q_softmax_0.01", "q_softmax_0.03",
                    "q_softmax_0.1", "q_softmax_0.3")) + " |")
        L += ["", "**(3) false positives by truth margin** — actions worse than the best by more than ε, share inside Q̂'s top-2 "
              "/ top-3 (chance beside it):", "",
              "| checkpoint | truth gap | n | in top-2 | chance | in top-3 | chance |", "|---|---|---|---|---|---|---|"]
        for c in ck:
            h = res["headline"].get(f"{c}|truth:{c}|{var}")
            if not h:
                continue
            for lo, hi in GAP_BINS:
                b = h[f"fp_gap_{lo}_{hi if hi < 9 else 'inf'}"]
                L.append(f"| {c} | {lo}–{hi if hi < 9 else '∞'} | {b['top2']['n']} | {_f(b['top2'])} | {_f(b['ch2'])} | "
                         f"{_f(b['top3'])} | {_f(b['ch3'])} |")
        L.append("")
    # categories (variant R)
    for var in ("M", "R"):
        L += _render_var_tail(res, ck, var)
    L += render_facts(res, ck)
    return "\n".join(L) + "\n"


def _render_var_tail(res: dict, ck: List[str], var: str) -> List[str]:
    L: List[str] = []
    L += [f"### (d) By move category (variant {var}, decisive turns)", "",
          "Near-best actions of the category: share Q̂-near / in Q̂ top-2, STARVED vs fed; dominated actions: share Q̂-near. "
          "Regret on turns where the category holds a starved near-best action: Q̂ argmax vs the policy's argmax.", "",
          "| checkpoint | category | starved n | starved Q̂-near | starved in top-2 | fed Q̂-near | dominated Q̂-near | "
          "regret Q̂ argmax | regret policy argmax |", "|---|---|---|---|---|---|---|---|---|"]
    for c in ck:
        h = res["headline"].get(f"{c}|truth:{c}|{var}")
        if not h:
            continue
        for cat in CATEGORIES:
            row = h["by_category"][cat]
            rg = row["starved_turn_regret"]
            L.append(f"| {c} | {cat} | {row['starved_near']['qnear']['n']} | {_f(row['starved_near']['qnear'])} | "
                     f"{_f(row['starved_near']['top2'])} | {_f(row['fed_near']['qnear'])} | {_f(row['dominated']['qnear'])} | "
                     f"{_f(rg['q_argmax'])} | {_f(rg['pi_argmax'])} |")
    L += ["", f"### (4) Paired vs absolute error (variant {var}, matched truth)", "",
          "| checkpoint | mean abs error | mean signed error | mean abs PAIR error | RMS abs | RMS pair | RMS pair if independent |"
          " RMS pair if fully common-mode | RMS pair noise floor | within-turn error corr | turn share of error variance |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in ck:
        e = res["errors"].get(f"{c}|truth:{c}|{var}")
        if e:
            L.append(f"| {c} | {_f(e['abs_error'])} | {_f(e['signed_error'])} | {_f(e['pair_abs_error'])} | {e['rms_abs_error']} | "
                     f"{e['rms_pair_error']} | {e['rms_pair_error_if_independent']} | {e['rms_pair_error_if_common_mode']} | "
                     f"{e['rms_pair_noise_floor']} | {e['within_turn_error_corr']} | {e['error_icc_turn_share']} |")
    L += ["", "### (5) Terminal vs V mixing, and V's calibration on the exact successors", "",
          "| checkpoint | variant | signed error, actions WITH game-ending branches | signed error, all-V actions (same turns) | difference |"
          " V-scored branch vs its own seed's outcome, mixed turns | same, turns with no game-ending branch |",
          "|---|---|---|---|---|---|---|"]
    for c in ck:
        for v_ in (var,):
            m = res["mixing"].get(f"{c}|truth:{c}|{v_}")
            if m:
                L.append(f"| {c} | {v_} | {_f(m['mixed_turn_actions_with_terminal_branches'])} | {_f(m['mixed_turn_actions_all_V'])} | "
                         f"{_f(m['terminal_minus_V_signed_error'], 'delta')} | {_f(m['branch_signed_error_V_on_mixed_turns'])} | "
                         f"{_f(m['branch_signed_error_V_on_unmixed_turns'])} |")
    if var == "M":   # per-branch pairing is exact only where the successor IS the truth playout's
        L += ["", "Per-branch calibration of V (as P(win)) against the SAME seed's truth outcome of that exact successor "
              "(matched checkpoint, variant M, all 64 seeds):", "",
              "| checkpoint | V-scored branches | mean V | mean outcome | Brier | Brier (constant) | ECE |", "|---|---|---|---|---|---|---|"]
    for c in (ck if var == "M" else []):
        cal = res["calibration"].get(c)
        if cal:
            L.append(f"| {c} | {cal['branches']} | {cal['mean_V']} | {cal['mean_outcome']} | {cal['brier']} | {cal['brier_const']} | {cal['ece']} |")
    L += ["", f"### Does S matter? (variant {var}; Q̂ on seeds 0..S−1 vs truth on held-out seeds 32..63)", "",
          "| checkpoint | S | Spearman ρ | top-2 holds a near-best | starved Q̂-near | dominated Q̂-near | regret Q̂ argmax (decisive) |",
          "|---|---|---|---|---|---|---|"]
    for c in ck:
        for s, r in sorted(res["s_curve"].get(f"{c}|{var}", {}).items(), key=lambda x: int(x[0])):
            L.append(f"| {c} | {s} | {_f(r['a_spearman'])} | {_f(r['recall_top2_any_near'])} | {_f(r['b_starved_near']['qnear'])} | "
                     f"{_f(r['b_dominated']['qnear'])} | {_f(r['regret']['decisive']['q_argmax'])} |")
    L += ["", f"### Cross-continuation (variant {var}): each checkpoint's Q̂ against every continuation's truth", "",
          "| Q̂ of | truth under | Spearman ρ | top-2 holds a near-best | starved Q̂-near | dominated Q̂-near | regret Q̂ argmax (starved) | regret policy argmax (starved) |",
          "|---|---|---|---|---|---|---|---|"]
    for c in ck:
        for t in CKPTS:
            r = res["headline"].get(f"{c}|truth:{t}|{var}") if t == c else res["cross"].get(f"{c}|truth:{t}|{var}")
            if r:
                L.append(f"| {c} | {t} | {_f(r['a_spearman'])} | {_f(r['recall_top2_any_near'])} | {_f(r['b_starved_near']['qnear'])} | "
                         f"{_f(r['b_dominated']['qnear'])} | {_f(r['regret']['starved']['q_argmax'])} | {_f(r['regret']['starved']['pi_argmax'])} |")
    L += ["", f"### Leak check (variant {var}): Lane S's 64-seed sets, Q̂'s own 8 seeds INSIDE the truth", "",
          "| checkpoint | starved Q̂-near | dominated Q̂-near | top-2 holds a near-best |", "|---|---|---|---|"]
    for c in ck:
        r = res["leak_check"].get(f"{c}|truth:{c}|{var}")
        if r:
            L.append(f"| {c} | {_f(r['b_starved_near']['qnear'])} | {_f(r['b_dominated']['qnear'])} | {_f(r['recall_top2_any_near'])} |")
    return L + [""]


def render_facts(res: dict, ck: List[str]) -> List[str]:
    L = ["### Sanity: the played action under the recorded dice reproduces the recorded next observation", "",
         "| checkpoint | turn outcomes | max abs ΔV (branch vs recorded) | mean Q̂_8(played) − V(recorded next) | mean abs | corr |",
         "|---|---|---|---|---|---|"]
    for c in ck:
        sv = res.get("sanity", {}).get(c)
        if sv:
            L.append(f"| {c} | {sv['status']} | {sv['max_abs_dV_branch_vs_recorded']} | {sv['q8_played_minus_V_recorded_mean']} | "
                     f"{sv['q8_played_minus_V_recorded_mean_abs']} | {sv['q8_played_vs_V_recorded_corr']} |")
    L += ["", "### Branch facts", ""]
    for c in ck:
        f = res["facts"][c]
        L.append(f"- **{c}**: R rows {f['R_ok']}, M rows {f['M_ok']}, refused {len(f['refused'])}; roots with an OPEN "
                 f"opponent decision {f['other_open']}; branches {f['branches']}, of which ended before our next decision "
                 f"{f['terminal']}, had ≥ 1 opponent decision answered by greedy before capture (R) {f['opp_answered_ge1_R']}, "
                 f"captured at a SWITCH-ONLY decision {f['next_switch_only']}; turns mixing terminal and V branches (R) {f['mixed_turns_R']}.")
    return L


if __name__ == "__main__":
    raise SystemExit(_cli())
