"""The meter's per-unit numbers, in plain numpy (no model, no torch): what one decision row says about each belief
leg, for every VARIANT of the belief handed in (the posterior, the prior, a λ blend of the two).

A UNIT is the thing a number is the mean of, and each family has its own:

* species — one decision ROW with at least one hidden opponent mon;
* moves — one (row, REVEALED opponent mon) with at least one true move the observation has not shown;
* items — one (row, revealed opponent mon) holding an item the observation does not yet state.

Every function returns :class:`Units`: the chunk row of each unit, ``values[metric][variant]`` per unit, and
``excluded[reason]`` — the chunk rows of the units it could not score, one entry each, so an exclusion is counted per
team like everything else and never silently dropped. Higher is better for every metric here (log probabilities,
hit rates).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping

import numpy as np

POST, PRIOR = "post", "prior"
#: A presence / probability gap below which two candidates at a top-k boundary are counted as a near tie (reported,
#: not excluded: ``hit`` and ``prec`` are descriptive columns, the log columns are the meter's verdict numbers).
TIE_EPS = 1e-6


def mix_name(lam: float) -> str:
    return f"mix{lam:g}"


def _logs(lam: float) -> "tuple[float, float]":
    if not 0.0 < lam < 1.0:
        raise ValueError(f"a blend needs 0 < λ < 1, got {lam!r} (λ = 0 is the posterior, λ = 1 the prior)")
    return math.log1p(-lam), math.log(lam)


def mix_log_prob(post: np.ndarray, prior: np.ndarray, lam: float) -> np.ndarray:
    """log((1 − λ)·e^post + λ·e^prior), elementwise: the species presence blend, and (on normalised log
    probabilities) the item blend — ``agents.model.belief_prior_mix``'s ``mix_presence`` / ``mix_categorical_logits``
    in numpy."""
    a, b = _logs(lam)
    return np.logaddexp(a + post, b + prior)


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def mix_bernoulli_logits(post: np.ndarray, prior: np.ndarray, lam: float) -> np.ndarray:
    """logit((1 − λ)·σ(post) + λ·σ(prior)), in log space: ``belief_prior_mix.mix_bernoulli_logits`` in numpy."""
    a, b = _logs(lam)
    log_p = np.logaddexp(a + log_sigmoid(post), b + log_sigmoid(prior))
    log_not = np.logaddexp(a + log_sigmoid(-post), b + log_sigmoid(-prior))
    return log_p - log_not


@dataclass
class Units:
    row: np.ndarray                                   # [U] chunk row of each scored unit
    values: Dict[str, Dict[str, np.ndarray]]          # metric -> variant -> [U]
    excluded: Dict[str, np.ndarray] = field(default_factory=dict)   # reason -> chunk rows, one per excluded unit


def _pack(rows: List[int], cols: Dict[str, Dict[str, List[float]]], excluded: Dict[str, List[int]]) -> Units:
    return Units(row=np.asarray(rows, dtype=np.int64),
                 values={m: {v: np.asarray(x, dtype=np.float64) for v, x in by.items()} for m, by in cols.items()},
                 excluded={k: np.asarray(v, dtype=np.int64) for k, v in excluded.items()})


def _top_hits(score: np.ndarray, truth: np.ndarray, k: int) -> "tuple[float, bool]":
    """The share of ``truth`` (indices into ``score``) among the ``k`` highest scores (stable: the lower index wins
    an exact tie), and whether the k-th and (k+1)-th scores are within :data:`TIE_EPS` in probability."""
    order = np.argsort(-score, kind="stable")
    top = order[:k]
    hit = float(np.isin(truth, top).sum()) / float(len(truth))
    tie = False
    if k < len(score) and np.isfinite(score[order[k]]):
        tie = bool(abs(math.exp(float(score[order[k - 1]])) - math.exp(float(score[order[k]]))) < TIE_EPS)
    return hit, tie


def species_units(log_pi: Mapping[str, np.ndarray], cand: np.ndarray, belief_species: np.ndarray) -> Units:
    """``log_pi[variant]`` [B,S] log presences, ``cand`` [B,S] the candidate set V, ``belief_species`` [B,6] the true
    hidden species nums (< 0 = none).

    ``species_logp`` is the mean log presence over the row's true hidden species that are IN V; a true species
    outside V (no dex row in the table) has no presence to read and is counted in ``excluded['species_off_support']``
    once per species, and a row whose every hidden species is outside V is not a unit. ``species_hit`` is the share
    of ALL the row's true hidden species (an off-V one is a miss for every variant) among the top-k presences, k =
    the true hidden count."""
    rows: List[int] = []
    cols: Dict[str, Dict[str, List[float]]] = {m: {v: [] for v in log_pi} for m in ("species_logp", "species_hit")}
    exc: Dict[str, List[int]] = {"species_off_support": [], "species_hit_near_tie": []}
    for r in range(len(belief_species)):
        hidden = np.unique(belief_species[r][belief_species[r] >= 0]).astype(np.int64)
        if hidden.size == 0:
            continue
        inside = hidden[cand[r, hidden]]
        exc["species_off_support"] += [r] * int(hidden.size - inside.size)
        if inside.size == 0:
            continue
        rows.append(r)
        tied = False
        for v, lp in log_pi.items():
            got = lp[r, inside].astype(np.float64)
            if not np.isfinite(got).all():
                raise ValueError(f"species variant {v!r}: a candidate's log presence is not finite at chunk row {r}")
            cols["species_logp"][v].append(float(got.mean()))
            hit, tie = _top_hits(np.where(cand[r], lp[r].astype(np.float64), -np.inf), hidden, int(hidden.size))
            cols["species_hit"][v].append(hit)
            tied = tied or tie
        if tied:
            exc["species_hit_near_tie"].append(r)
    return _pack(rows, cols, exc)


def move_units(logits: Mapping[str, np.ndarray], legal: np.ndarray, known_moves: np.ndarray,
               revealed_ids: np.ndarray, revealed_flag: np.ndarray, opp_species: np.ndarray,
               hp_family: FrozenSet[int]) -> Units:
    """``logits[variant]`` [B,6,M] per-move log-odds; ``legal`` [B,6,M] bool, the moves the species' prior table
    calls learnable; ``known_moves`` [B,6,4] the true move nums at REVEALED slots (-1 pad); ``revealed_ids`` /
    ``revealed_flag`` [B,6,4] the observation's move cells; ``opp_species`` [B,6].

    For each revealed mon: U = its true moves, minus the Hidden Power family, minus the moves the observation
    already shows. ``moves_logp`` is the mean log σ(logit) over U; ``moves_prec`` the share of U among the |U|
    highest logits of the candidate moves (legal, not revealed, not Hidden Power). A unit whose U holds a move the
    table calls unlearnable is excluded (``moves_truth_illegal``): no variant can score it.

    ``moves_ll`` is the PROPER score of the same unit: the Bernoulli log-likelihood of the truth over the whole
    candidate set, Σ over U of log σ(logit) + Σ over the other candidates of log σ(−logit), in nats a unit (a SUM,
    not a mean over moves). ``moves_logp`` reads only U, so a belief that raises a move the mon does not carry costs
    it nothing there; ``moves_ll`` charges it (the 2026-11-10 move-leg diagnosis: a head within 0.1 nats of another
    on ``moves_logp`` was 9 to 12 nats a unit below the prior on this)."""
    rows: List[int] = []
    cols: Dict[str, Dict[str, List[float]]] = {m: {v: [] for v in logits}
                                               for m in ("moves_logp", "moves_ll", "moves_prec")}
    exc: Dict[str, List[int]] = {"moves_truth_illegal": [], "moves_prec_near_tie": []}
    n_moves = legal.shape[-1]
    hp = np.zeros(n_moves, dtype=bool)
    hp[[m for m in hp_family if 0 <= m < n_moves]] = True
    for r in range(len(known_moves)):
        for i in range(known_moves.shape[1]):
            km = known_moves[r, i]
            if km[0] < 0 or opp_species[r, i] <= 0:
                continue
            shown = np.zeros(n_moves, dtype=bool)
            ids = revealed_ids[r, i][(revealed_flag[r, i] > 0.5) & (revealed_ids[r, i] > 0)].astype(np.int64)
            shown[ids[ids < n_moves]] = True
            truth = np.unique(km[km >= 0]).astype(np.int64)
            truth = truth[~hp[truth] & ~shown[truth]]
            if truth.size == 0:
                continue
            candm = legal[r, i] & ~hp & ~shown
            if not candm[truth].all():
                exc["moves_truth_illegal"].append(r)
                continue
            rows.append(r)
            ci = np.flatnonzero(candm)
            pos = np.searchsorted(ci, truth)
            sign = -np.ones(ci.size)                       # +1 on a true move, −1 on a candidate the mon lacks
            sign[pos] = 1.0
            tied = False
            for v, lg in logits.items():
                x = lg[r, i].astype(np.float64)
                cols["moves_logp"][v].append(float(log_sigmoid(x[truth]).mean()))
                cols["moves_ll"][v].append(float(log_sigmoid(sign * x[ci]).sum()))
                hit, tie = _top_hits(log_sigmoid(x[ci]), pos, int(truth.size))
                cols["moves_prec"][v].append(hit)
                tied = tied or tie
            if tied:
                exc["moves_prec_near_tie"].append(r)
    return _pack(rows, cols, exc)


def item_units(logp: Mapping[str, np.ndarray], item_label: np.ndarray, item_mask: np.ndarray,
               item_known: np.ndarray) -> Units:
    """``logp[variant]`` [B,6,I] normalised log probabilities over item nums; ``item_label`` / ``item_mask`` [B,6] the
    core's label; ``item_known`` [B,6] the observation's item-known cell.

    A unit is a labelled slot holding a real item (label > 0) that the observation does not state. ``items_logp`` is
    the log probability of the true item, ``items_top1`` whether it is the argmax."""
    rows: List[int] = []
    cols: Dict[str, Dict[str, List[float]]] = {m: {v: [] for v in logp} for m in ("items_logp", "items_top1")}
    rr, ii = np.nonzero((item_mask > 0.5) & (item_label > 0) & ~(item_known > 0.5))
    for r, i in zip(rr.tolist(), ii.tolist()):
        lab = int(item_label[r, i])
        rows.append(r)
        for v, lp in logp.items():
            cols["items_logp"][v].append(float(lp[r, i, lab]))
            cols["items_top1"][v].append(float(int(np.argmax(lp[r, i])) == lab))
    return _pack(rows, cols, {})


#: metric -> the family whose unit it is a mean over (one count column per family).
FAMILY_OF = {"species_logp": "species", "species_hit": "species", "moves_logp": "moves", "moves_ll": "moves",
             "moves_prec": "moves", "items_logp": "items", "items_top1": "items"}
METRICS = tuple(FAMILY_OF)
FAMILIES = ("species", "moves", "items")
