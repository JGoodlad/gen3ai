"""The RANK-MASS SPECTRUM and its strata (`gen3_policy_spectrum_v1`, M5 Lane S gate ③). Pure NumPy.

For one policy on one decision: sort the legal actions by the policy's OWN probability, highest
first. The spectrum is the mean over decisions of the probability at rank 1, 2, 3, … — how much
mass the policy puts on its own 1st, 2nd, … nth choice. Beside it: entropy (nats), entropy
normalised by ``log(n_legal)``, the effective number of choices ``exp(H)``, and the share of
decisions where the top choice holds ≥ 0.9 / ≥ 0.99.

**This is a DESCRIPTOR of sharpness, not of quality.** Whether mass that left rank 2 left a
near-best action (starvation) or a dominated one (healthy sharpening) needs each action's TRUE
value — gate ④ (branch ground truth, Lane I). The category readouts below (mass on status / setup
moves, the share of turns where a legal category gets < 1 %) say WHERE mass went, never whether it
should have.

Intervals: a percentile bootstrap that resamples BATTLES (decisions of one battle are correlated),
fixed seed, so a read is deterministic.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from main.policy_spectrum.categories import CATEGORIES

SPECTRUM_SCHEMA = "gen3_policy_spectrum_v1"
K_SHOW = 6                 # ranks reported individually; the rest is summed into `tail`
N_BOOT = 200
BOOT_SEED = 20260929
STARVE_MASS = 0.01         # "a legal category holds < 1 % of the mass" (the owner's starvation bar)


def masked_probs(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """float64 masked softmax of float32 logits (illegal actions exactly 0)."""
    lg = np.asarray(logits, dtype=np.float64)
    m = np.asarray(masks, dtype=bool)
    lg = np.where(m, lg, -np.inf)
    lg = lg - lg.max(axis=1, keepdims=True)
    e = np.where(m, np.exp(lg), 0.0)
    return e / e.sum(axis=1, keepdims=True)


class Derived:
    """Per-decision quantities every stratum reads."""

    def __init__(self, probs: np.ndarray, decisions: Sequence[dict]):
        p = np.asarray(probs, dtype=np.float64)
        self.p = p
        self.order = np.argsort(-p, axis=1, kind="stable")
        self.ranked = np.take_along_axis(p, self.order, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.H = -np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(axis=1)
        self.n_legal = np.array([d["n_legal"] for d in decisions])
        self.Hn = self.H / np.log(np.maximum(self.n_legal, 2))
        battles = [d["battle"] for d in decisions]
        uniq = {b: i for i, b in enumerate(dict.fromkeys(battles))}
        self.battle = np.array([uniq[b] for b in battles])
        n = len(decisions)
        # per-decision category of each action (index -> cat), "" if illegal
        self.cat = np.full((n, p.shape[1]), "", dtype=object)
        for i, d in enumerate(decisions):
            for k, c in d["cats"].items():
                self.cat[i, int(k)] = c
        self.cat_mass = {c: (p * (self.cat == c)).sum(axis=1) for c in CATEGORIES}
        self.cat_legal = {c: (self.cat == c).any(axis=1) for c in CATEGORIES}
        self.rank_cat = np.take_along_axis(self.cat, self.order, axis=1)   # category at rank r


def _boot_ci(values: np.ndarray, clusters: np.ndarray) -> Optional[List[float]]:
    """95 % percentile interval of the mean, resampling clusters (battles)."""
    if len(values) == 0:
        return None
    uniq, inv = np.unique(clusters, return_inverse=True)
    if len(uniq) < 2:
        return None
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    means = sums[draws].sum(axis=1) / cnts[draws].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 6), round(float(hi), 6)]


def _r(x) -> float:
    return round(float(x), 6)


def aggregate(D: Derived, idx: np.ndarray, ci: bool = True) -> dict:
    """The spectrum block for the decisions ``idx``."""
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 0:
        return {"n": 0}
    R = D.ranked[idx]
    out = {
        "n": int(len(idx)),
        "battles": int(len(np.unique(D.battle[idx]))),
        "mean_n_legal": _r(D.n_legal[idx].mean()),
        "spectrum": [_r(v) for v in R[:, :K_SHOW].mean(axis=0)],
        "tail": _r(R[:, K_SHOW:].sum(axis=1).mean()),
        "entropy": _r(D.H[idx].mean()),
        "entropy_norm": _r(D.Hn[idx].mean()),
        "eff_choices": _r(np.exp(D.H[idx]).mean()),
        "top1_ge_0.9": _r((R[:, 0] >= 0.9).mean()),
        "top1_ge_0.99": _r((R[:, 0] >= 0.99).mean()),
    }
    if ci:
        cl = D.battle[idx]
        out["ci"] = {"rank1": _boot_ci(R[:, 0], cl), "rank2": _boot_ci(R[:, 1], cl),
                     "rank3": _boot_ci(R[:, 2], cl), "entropy": _boot_ci(D.H[idx], cl)}
    return out


def category_block(D: Derived, idx: np.ndarray) -> Dict[str, dict]:
    """Per category, over the decisions in ``idx`` where it is LEGAL: the mass on it, the share of
    decisions where it holds < 1 %, the share where the policy's top choice is in it, and the rank
    and probability of its best action."""
    idx = np.asarray(idx, dtype=int)
    out = {}
    for c in CATEGORIES:
        sel = idx[D.cat_legal[c][idx]]
        if len(sel) == 0:
            out[c] = {"n": 0}
            continue
        mass = D.cat_mass[c][sel]
        rc = D.rank_cat[sel]
        best_rank = np.argmax(rc == c, axis=1) + 1          # first rank holding category c
        best_prob = np.take_along_axis(D.ranked[sel], (best_rank - 1)[:, None], axis=1)[:, 0]
        out[c] = {
            "n": int(len(sel)),
            "mass": _r(mass.mean()),
            "mass_ci": _boot_ci(mass, D.battle[sel]),
            "share_lt_1pct": _r((mass < STARVE_MASS).mean()),
            "share_top1": _r((rc[:, 0] == c).mean()),
            "best_rank": _r(best_rank.mean()),
            "best_prob": _r(best_prob.mean()),
        }
    return out


def rank_by_category(D: Derived, idx: np.ndarray, ranks: int = 3) -> Dict[str, dict]:
    """For the policy's 1st / 2nd / 3rd choice: which category it is (share) and the mean mass it
    holds when it is that category. Decisions with fewer legal actions than the rank are left out."""
    idx = np.asarray(idx, dtype=int)
    out = {}
    for r in range(ranks):
        sel = idx[D.n_legal[idx] > r]
        row = {"n": int(len(sel))}
        for c in CATEGORIES:
            hit = sel[D.rank_cat[sel, r] == c]
            row[c] = {"share": _r(len(hit) / len(sel)) if len(sel) else 0.0,
                      "mass": _r(D.ranked[hit, r].mean()) if len(hit) else None}
        out[f"rank{r + 1}"] = row
    return out


#: The strata a read reports: name -> key(decision) (None = not in the stratum).
STRATA: Dict[str, Callable[[dict], Optional[str]]] = {
    "all": lambda d: "all",
    "kind": lambda d: d["kind"],
    "phase": lambda d: d["phase"],
    "free_by_phase": lambda d: d["phase"] if d["kind"] == "free" else None,
    "legal_bucket": lambda d: d["legal_bucket"],
    "opp_class": lambda d: d["opp_class"],
    "free_by_opp_class": lambda d: d["opp_class"] if d["kind"] == "free" else None,
    "opp_name": lambda d: d["opp_name"],
    "source": lambda d: d["source"],
    "outcome": lambda d: d["outcome"],
}


def strata_index(decisions: Sequence[dict]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, key in STRATA.items():
        groups: Dict[str, List[int]] = {}
        for i, d in enumerate(decisions):
            k = key(d)
            if k is not None:
                groups.setdefault(k, []).append(i)
        out[name] = {k: np.asarray(v, dtype=int) for k, v in sorted(groups.items())}
    # free decisions where a category is legal: the spectrum there
    free = [i for i, d in enumerate(decisions) if d["kind"] == "free"]
    out["free_cat_legal"] = {}
    for c in CATEGORIES:
        if c == "attack":
            continue
        sel = [i for i in free if c in decisions[i]["cats"].values()]
        out["free_cat_legal"][c] = np.asarray(sel, dtype=int)
    return out


def read(probs: np.ndarray, decisions: Sequence[dict], ci: bool = True) -> dict:
    """Every block of a read, for one policy's probabilities on the bank."""
    D = Derived(probs, decisions)
    S = strata_index(decisions)
    free = S["kind"].get("free", np.zeros(0, dtype=int))
    return {
        "strata": {name: {k: aggregate(D, v, ci=ci) for k, v in groups.items()}
                   for name, groups in S.items()},
        "categories_free": category_block(D, free),
        "categories_free_by_phase": {ph: category_block(D, v) for ph, v in S["free_by_phase"].items()},
        "rank_category_free": rank_by_category(D, free),
    }


def paired_delta(p_a: np.ndarray, p_b: np.ndarray, decisions: Sequence[dict],
                 idx: Optional[np.ndarray] = None) -> dict:
    """``b − a`` on the SAME decisions for rank-1/2/3 mass and entropy, with a battle-clustered
    bootstrap interval (the bank's point: two policies read on identical turns)."""
    Da, Db = Derived(p_a, decisions), Derived(p_b, decisions)
    idx = np.arange(len(decisions)) if idx is None else np.asarray(idx, dtype=int)
    out = {"n": int(len(idx))}
    for name, fa, fb in (("rank1", Da.ranked[:, 0], Db.ranked[:, 0]),
                         ("rank2", Da.ranked[:, 1], Db.ranked[:, 1]),
                         ("rank3", Da.ranked[:, 2], Db.ranked[:, 2]),
                         ("entropy", Da.H, Db.H)):
        d = (fb - fa)[idx]
        out[name] = {"delta": _r(d.mean()), "ci": _boot_ci(d, Da.battle[idx])}
    return out
