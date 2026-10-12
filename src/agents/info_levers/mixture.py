"""THE MIXTURE: what a lever of mass p and a source does to one fact, shared by every Python consumer.

A lever replaces the consumer's honest belief about a fact by p·δ(target) + (1 − p)·(the honest belief), where the
TARGET is (``spec`` docstring):

* ``truth``  — the true value;
* ``placebo`` — an equally confident WRONG but plausible value: drawn ∝ the honest belief over the candidates
  EXCLUDING the truth (and everything play revealed), so it is a value the honest belief finds plausible;
* ``map``    — the honest belief's own mode (its top candidates): an honest sharpening, no truth read.

Two shapes of fact:

* a CATEGORICAL fact (an item, an ability, a spread, a never-seen species): :func:`categorical_target` picks the target
  value and :func:`mix_draw` draws a world's value from the mixture (the search builder);
* a SET fact (the 4 − r unrevealed move slots of a mon): the target is a SET of k values (:func:`set_target`), and the
  mixture is over each candidate's PRESENCE, π'_m = p·1[m ∈ target] + (1 − p)·π_m with π renormalised to mass k over
  the candidates (:func:`mix_presence`); a world draws its k moves by :func:`cp_sample`, the conditional-Poisson
  fixed-size design (independent presences conditioned on the count), so p = 0 is the honest draw exactly and p = 1
  the target exactly. (The hidden-info study's §6.4 mixture; its injection writes the same π' into the network.)

Every random choice takes an explicit ``numpy`` Generator (deterministic per decision / world).
"""
from __future__ import annotations

from typing import Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, TypeVar

import numpy as np

K = TypeVar("K", bound=Hashable)

#: The source of a lever (``spec.SOURCES``).
SOURCES = ("truth", "placebo", "map")


def _ordered(weights: Mapping[K, float], exclude: Iterable[K]) -> List[K]:
    ex = set(exclude)
    return [k for k in weights if k not in ex and weights[k] > 0]


def draw(rng: np.random.Generator, weights: Mapping[K, float], exclude: Iterable[K] = ()) -> Optional[K]:
    """One value ∝ ``weights``, or None. The candidates are taken in ``weights``' own (insertion) order, so a
    deterministically built mapping and a seed reproduce a draw."""
    keys = _ordered(weights, exclude)
    if not keys:
        return None
    w = np.array([weights[k] for k in keys], dtype=np.float64)
    return keys[int(rng.choice(len(keys), p=w / w.sum()))]


def mode(weights: Mapping[K, float], exclude: Iterable[K] = ()) -> Optional[K]:
    """The top candidate (ties broken by the candidate's text: deterministic)."""
    keys = _ordered(weights, exclude)
    if not keys:
        return None
    return min(keys, key=lambda k: (-float(weights[k]), str(k)))


def categorical_target(source: str, truth: Optional[K], weights: Mapping[K, float], rng: np.random.Generator,
                       exclude: Iterable[K] = ()) -> Optional[K]:
    """The target value of a categorical lever: the truth, a placebo (∝ ``weights`` excluding the truth and
    ``exclude``) or the map (the mode of ``weights`` excluding ``exclude``). None when no candidate exists."""
    if source == "truth":
        return truth
    if source == "placebo":
        return draw(rng, weights, list(exclude) + ([truth] if truth is not None else []))
    if source == "map":
        return mode(weights, exclude)
    raise ValueError(f"lever source {source!r} is not one of {SOURCES}")


def mix_draw(rng: np.random.Generator, p: float, target: Optional[K], weights: Mapping[K, float],
             exclude: Iterable[K] = ()) -> Optional[K]:
    """A draw from p·δ(target) + (1 − p)·(``weights`` normalised): the target with probability p (p ≥ 1: always,
    no random number spent), else an honest draw (which may land on the target too)."""
    if target is not None and (p >= 1.0 or (p > 0.0 and rng.random() < p)):
        return target
    return draw(rng, weights, exclude)


def set_target(source: str, truth: Sequence[K], weights: Mapping[K, float], k: int, rng: np.random.Generator,
               exclude: Iterable[K] = (), at_most_one: Optional[Callable[[K], bool]] = None) -> List[K]:
    """The target SET of a set lever (k values): the truth's unrevealed values (``truth`` minus ``exclude``), a
    placebo (k values drawn without replacement ∝ ``weights`` excluding the truth and ``exclude``) or the map (the k
    top candidates). ``at_most_one`` marks a family of which a set holds at most one (Hidden Power)."""
    ex = set(exclude)
    if source == "truth":
        return [t for t in truth if t not in ex]
    pool = {c: w for c, w in weights.items() if c not in ex and (source != "placebo" or c not in set(truth))}
    out: List[K] = []
    while len(out) < k:
        c = draw(rng, pool) if source == "placebo" else mode(pool) if source == "map" else None
        if source not in ("placebo", "map"):
            raise ValueError(f"lever source {source!r} is not one of {SOURCES}")
        if c is None:
            break
        out.append(c)
        pool.pop(c)
        if at_most_one is not None and at_most_one(c):
            pool = {x: w for x, w in pool.items() if not at_most_one(x)}
    return out


def mix_presence(weights: Mapping[K, float], target: Iterable[K], p: float, k: int) -> Dict[K, float]:
    """π'_m = p·1[m ∈ target] + (1 − p)·π_m over the candidates, π = ``weights`` renormalised to mass ``k``
    (each clipped to ≤ 1, so the result is a valid presence vector). Candidates = the keys of ``weights``."""
    tgt = set(target)
    keys = [c for c in weights if weights[c] > 0 or c in tgt] + [t for t in target if t not in weights]
    tot = float(sum(max(0.0, weights.get(c, 0.0)) for c in keys))
    out: Dict[K, float] = {}
    for c in keys:
        pi = min(1.0, k * max(0.0, weights.get(c, 0.0)) / tot) if tot > 0 else 0.0
        out[c] = p * (1.0 if c in tgt else 0.0) + (1.0 - p) * pi
    return out


def cp_inclusion(pis: Sequence[float], m: int) -> List[float]:
    """Each key's EXACT inclusion probability under :func:`cp_sample` (P(i ∈ S) = w_i·e_{m−1}(w_{−i}) / e_m(w),
    w = π / (1 − π); a presence ≥ 1 is included surely, ≤ 0 never): the honest expectation the leak audit compares
    a world's true-move hits with."""
    sure = [p >= 1.0 for p in pis]
    need = m - sum(sure)
    out = [1.0 if s else 0.0 for s in sure]
    idx = [i for i, p in enumerate(pis) if 0.0 < p < 1.0]
    if need <= 0 or len(idx) < need:
        return out
    w = np.array([pis[i] / (1.0 - pis[i]) for i in idx], dtype=np.float64)
    w = w / w.max()

    def esp(v: np.ndarray, k: int) -> np.ndarray:
        e = np.zeros(k + 1)
        e[0] = 1.0
        for x in v:
            e[1:] = e[1:] + x * e[:-1].copy()
        return e

    total = esp(w, need)[need]
    for j, i in enumerate(idx):
        rest = esp(np.delete(w, j), need - 1)[need - 1]
        out[i] = float(w[j] * rest / total) if total > 0 else 0.0
    return out


def cp_sample(rng: np.random.Generator, keys: Sequence[K], pis: Sequence[float], m: int) -> Optional[List[K]]:
    """A FIXED-SIZE subset of ``m`` keys drawn as independent presences π conditioned on the count = m (the
    conditional-Poisson design: P(S) ∝ Π_{i∈S} π_i / (1 − π_i)); an exact sequential draw from the elementary
    symmetric sums (the value-discrimination pilot's Phase 2c sampler, `determinize.cp_sample`). A presence of
    exactly 1 is drawn surely, one of 0 never. Returns the keys in a random order, or None if fewer than m."""
    if m <= 0:
        return []
    sure = [k for k, p in zip(keys, pis) if p >= 1.0]
    rest = [(k, p) for k, p in zip(keys, pis) if 0.0 < p < 1.0]
    if len(sure) > m:
        return None
    need = m - len(sure)
    n = len(rest)
    if n < need:
        return None
    out = list(sure)
    if need:
        w = np.array([p / (1.0 - p) for _, p in rest], dtype=np.float64)
        w = w / w.max()
        E = np.zeros((n + 1, need + 1))
        E[n, 0] = 1.0
        for i in range(n - 1, -1, -1):
            E[i, 0] = 1.0
            for j in range(1, need + 1):
                E[i, j] = E[i + 1, j] + w[i] * E[i + 1, j - 1]
        left = need
        for i in range(n):
            if left == 0:
                break
            if E[i, left] <= 0:
                return None
            if rng.random() < w[i] * E[i + 1, left - 1] / E[i, left]:
                out.append(rest[i][0])
                left -= 1
        if left:
            return None
    return [out[k] for k in rng.permutation(len(out))]
