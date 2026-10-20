"""AMENDMENT A1 (registered 2026-10-19 22:02 PDT, before any outcome was read; ``PREREGISTRATION.md`` §9): the
literature review's L2 (`research_state/literature/hidden_info_search_review_2026-10-20.md`). Does an arm's per-state
GAIN track the EXACT-WORLD MASS of its worlds, or how CALIBRATED they are (how much of the arm's mass sits on the true
hidden facts), or both? SECONDARY; it changes no primary rule or verdict. Pure numpy; the rules are code, tested on
planted data by ``read_s1_test.py``.

THE TWO PREDICTORS, per arm X and state s (both on the arm's OWN 8 built worlds, the worlds it played):
  * q_SM,X(s)  the share of the 8 worlds exact on U (the never-seen species SET) AND M (every seen mon's whole move set,
               a Hidden Power one move whatever its type): ``score_s1.world_meters``;
  * LL8,X(s)   the truth's log-likelihood under the arm's world distribution, FACTORISED over the hidden facts and
               SMOOTHED (the closest computable proxy: the 8-world empirical distribution gives the whole truth
               probability 0 unless a world is exact). The hidden facts of s are each TRUE never-seen species u ∈ U*
               and, for each seen mon with unrevealed slots, each TRUE unrevealed move (canonical; Hidden Power one
               move). c(f) = the number of the 8 worlds holding fact f (u among the world's never-seen species; the
               move in that mon's world set); p̂(f) = (c(f) + 0.5) / (8 + 1) (the Krichevsky–Trofimov estimator);
               LL8 = the MEAN over the state's hidden facts of log p̂(f) (a state with no hidden fact: undefined).
  Both are taken RELATIVE to the baseline on the same state: Δq_X(s) = q_SM,X(s) − q_SM,F_legal(s),
  ΔLL_X(s) = LL8_X(s) − LL8_F_legal(s). THE GAIN: Δ_X(s) = IMP_X(s) − IMP_F_legal(s) at LEAF-K8 (registered;
  TRUNC-4-K8 descriptive).
  A DESCRIPTIVE companion: LLU, stage 0's level-1 analytic log p(U*) of the arm's never-seen SPECIES proposal (the
  review's "proper score"), between arms only.

THE READS (rule 8: a bound within ±0.0005 of its boundary is not clear; the read's own bootstrap draws):
  WITHIN ARMS (the registered decision): per primary arm, Spearman ρ_q = ρ(Δ_X, Δq_X) and ρ_LL = ρ(Δ_X, ΔLL_X) over its
    valid states; POOLED = the state-count-weighted mean over the five primary arms (an arm whose predictor is
    constant in a draw drops out of that draw). TRACKS-EXACT iff lower95[ρ̄_q] > +0.0005; TRACKS-LL iff
    lower95[ρ̄_LL] > +0.0005 -> BOTH / EXACT-ONLY / LL-ONLY / NEITHER. SEPARATION: LL-AHEAD iff lower95[ρ̄_LL − ρ̄_q] >
    +0.0005, EXACT-AHEAD iff upper95 < −0.0005, else NOT SEPARATED.
  BETWEEN ARMS: over the seven non-baseline arms (B, C_ladder, C_forum_strict, C_pool, NS, F_off, F_rolls), Spearman
    of the arm means (mean Δ_X against mean Δq_X, and against mean ΔLL_X), each arm on its own states valid with
    F_legal; the same flags and labels; and the same WITHOUT C_pool (six arms: where exactness is ≈ 0, the
    discriminating comparison). LLU's between-arm ρ is descriptive.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

KT = 0.5
J = 8
L2_ARMS_WITHIN = ("B", "C_ladder", "C_forum_strict", "C_pool", "NS")
L2_ARMS_BETWEEN = ("B", "C_ladder", "C_forum_strict", "C_pool", "NS", "F_off", "F_rolls")


def canon(m: str) -> str:
    return "hiddenpower" if m.startswith("hiddenpower") else m


def _mons(team: str) -> Dict[str, frozenset]:
    from agents.info_levers.packed import parse_set, species_id, typed_moves

    out = {}
    for e in team.split("]"):
        f = parse_set(e)
        out[species_id(f)] = frozenset(canon(m) for m in typed_moves(f))
    return out


def ll8(world_teams: Sequence[str], true_team: str, revealed: Dict[str, Sequence[str]]) -> float:
    """The KT-smoothed factorised log-likelihood of the true hidden facts under the worlds (module docstring)."""
    tru = _mons(true_team)
    seen = set(revealed)
    u_true = [s for s in tru if s not in seen]
    rev = {s: {canon(m) for m in ms} for s, ms in revealed.items()}
    hidden_moves = [(s, m) for s in sorted(seen) if s in tru for m in sorted(tru[s] - rev.get(s, set()))]
    if not u_true and not hidden_moves:
        return float("nan")
    worlds = [_mons(t) for t in world_teams]
    n = len(worlds)
    logs = []
    for u in u_true:
        c = sum(1 for w in worlds if u in w and u not in seen)
        logs.append(np.log((c + KT) / (n + 2 * KT)))
    for s, m in hidden_moves:
        c = sum(1 for w in worlds if m in w.get(s, frozenset()))
        logs.append(np.log((c + KT) / (n + 2 * KT)))
    return float(np.mean(logs))


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    return (start + (cnt - 1) / 2.0)[inv]


def spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def draws_index(bt, battles: Sequence[str]) -> List[np.ndarray]:
    """Per bootstrap draw, the state indices with their multiplicity (``bt.C[r, battle]`` copies of each state)."""
    bi = np.array([bt.ub.index(b) for b in battles]) if battles else np.array([], dtype=int)
    out = []
    for r in range(bt.C.shape[0]):
        w = bt.C[r, bi].astype(int) if len(bi) else np.array([], dtype=int)
        out.append(np.repeat(np.arange(len(bi)), w))
    return out


def _flag(lo: Optional[float]) -> bool:
    return lo is not None and np.isfinite(lo) and lo > 0.0005


def _sep(lo: Optional[float], hi: Optional[float]) -> str:
    if lo is not None and np.isfinite(lo) and lo > 0.0005:
        return "LL-AHEAD"
    if hi is not None and np.isfinite(hi) and hi < -0.0005:
        return "EXACT-AHEAD"
    return "NOT SEPARATED"


def label(tq: bool, tl: bool) -> str:
    return {(True, True): "BOTH", (True, False): "EXACT-ONLY", (False, True): "LL-ONLY",
            (False, False): "NEITHER"}[(tq, tl)]


def within(per_arm: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]], bt, ci) -> dict:
    """``per_arm[X] = (gain, dq, dll, battles)`` over X's valid states. Per arm ρ's, and the pooled n-weighted means
    with their per-draw vectors and the registered flags."""
    out: dict = {"per_arm": {}}
    B = bt.C.shape[0]
    pq, pl = np.zeros(B), np.zeros(B)
    wq, wl = np.zeros(B), np.zeros(B)
    for x, (g, dq, dl, bats) in per_arm.items():
        idx = draws_index(bt, bats)
        rq = np.array([spearman(g[i], dq[i]) for i in idx])
        rl = np.array([spearman(g[i], dl[i]) for i in idx])
        n = float(len(g))
        okq, okl = np.isfinite(rq), np.isfinite(rl)
        pq[okq] += n * rq[okq]
        wq[okq] += n
        pl[okl] += n * rl[okl]
        wl[okl] += n
        out["per_arm"][x] = {"n": int(n), "rho_q": spearman(g, dq), "rho_q_ci": ci(rq), "rho_LL": spearman(g, dl),
                             "rho_LL_ci": ci(rl), "rho_LL_minus_q_ci": ci(rl - rq)}
    with np.errstate(invalid="ignore", divide="ignore"):
        mq, ml = pq / wq, pl / wl
    cq, cl, cd = ci(mq), ci(ml), ci(ml - mq)
    tq, tl = _flag(cq[0]), _flag(cl[0])
    out["pooled"] = {"rho_q_ci": cq, "rho_LL_ci": cl, "rho_LL_minus_q_ci": cd, "TRACKS-EXACT": tq, "TRACKS-LL": tl,
                     "verdict": label(tq, tl), "separation": _sep(*cd)}
    return out


def between(means: Dict[str, Tuple[float, np.ndarray]], gains: Dict[str, Tuple[float, np.ndarray]], ci) -> dict:
    """Spearman over arms of (mean gain, mean predictor), point and per draw; the flags and the label."""
    arms = sorted(set(means) & set(gains))
    if len(arms) < 3:
        return {"verdict": "NOT READ (fewer than 3 arms)"}
    pt = spearman([gains[a][0] for a in arms], [means[a][0] for a in arms])
    G = np.stack([gains[a][1] for a in arms], 1)
    M = np.stack([means[a][1] for a in arms], 1)
    v = np.array([spearman(G[r], M[r]) for r in range(G.shape[0])])
    c = ci(v)
    return {"arms": arms, "rho": pt, "rho_ci": c, "tracks": _flag(c[0])}


class LLU:
    """The DESCRIPTIVE companion: stage 0's level-1 analytic log p(U*) of each arm's never-seen SPECIES proposal
    (``screen_samplers``' exact set probabilities, imported), at stage 1's own parameters: F_legal / F_off / F_rolls
    rule (a) (F's weights); B rule (b); C_ladder / C_forum_strict λ·f_U + (1 − λ)·P_b with λ = n / (n + ALPHA) over the
    arm's own consistent (strict: minus near copies) corpus teams; NS the network's π (FN's rule, F's weights when π has
    no mass); C_pool the share of the consistent pool teams (the pin's packing) whose never-seen set is U* (no
    backoff). A probability below 1e-12 is floored there (stage 0's floor). A state with no never-seen mon: undefined."""

    def __init__(self, alpha: float, beliefs: Dict[str, dict]):
        import json
        import sys

        import s1_common as E

        for p in (E.STAGE0, E.T1):
            if str(p) not in sys.path:
                sys.path.insert(1, str(p))
        import corpus_s1 as CS
        import screen_samplers as S0
        from main.value_search import world_legality as WL
        from main.value_search import worlds as W
        from main.value_search.world_draws import Prior

        self.S0, self.CS, self.W = S0, CS, W
        self.S = S0.Smogon()
        P = Prior()
        legal = WL.Legality(pad=W._gender_pad())
        self.corp = {"C_ladder": CS.load("ladder", legal, P), "C_forum_strict": CS.load("forum", legal, P)}
        self.alpha = float(alpha)
        self.beliefs = beliefs
        self.pool = json.loads((E.ARCH / "pool" / "pool_pin.json").read_text())["teams"]

    def __call__(self, arm: str, did: str, true_team: str, revealed: Dict[str, Sequence[str]]) -> float:
        S, S0, CS = self.S, self.S0, self.CS
        tru = _mons(true_team)
        seen = sorted(revealed)
        unseen = [s for s in tru if s not in revealed]
        if not unseen:
            return float("nan")
        seen_ix = [S.ix[s] for s in seen if s in S.ix]
        tgt = [S.ix[s] for s in unseen if s in S.ix]
        ok_pool = len(tgt) == len(unseen)

        def pb() -> float:
            return S0.p_set_poe(S, seen_ix, tgt) if ok_pool else 0.0

        if arm in ("F_legal", "F_off", "F_rolls"):
            p = S0.p_set_fixed(S.f_weights([s for s in seen if s in S.ix]), seen_ix, tgt) if ok_pool else 0.0
        elif arm == "B":
            p = pb()
        elif arm in self.corp:
            C = self.corp[arm]
            kept = {s: set(ms) for s, ms in revealed.items()}
            ids = C.consistent(kept)
            if arm == "C_forum_strict":
                near = C.near_copies({(s, mv) for s, mv in tru.items()})
                ids = [i for i in ids if i not in near]
            n = len(ids)
            f = (sum(1 for i in ids if frozenset(C.sets[i]) - set(revealed) == frozenset(unseen)) / n) if n else 0.0
            lam = n / (n + self.alpha) if n else 0.0
            p = lam * f + (1 - lam) * pb()
        elif arm == "NS":
            row = self.beliefs.get(did)
            if row is None or not ok_pool:
                return float("nan")
            pi = np.array([float(row["species"].get(s, 0.0)) for s in S.pool])
            w = pi if np.delete(pi, seen_ix).sum() > 0 else S.f_weights([s for s in seen if s in S.ix])
            p = S0.p_set_fixed(w, seen_ix, tgt)
        elif arm == "C_pool":
            want = {s: {canon(m) for m in ms} for s, ms in revealed.items()}
            ids = []
            for t in self.pool:
                m = _mons(t)
                if all(s in m and want[s] <= m[s] for s in want):
                    ids.append(frozenset(m) - set(revealed))
            p = (sum(1 for u in ids if u == frozenset(unseen)) / len(ids)) if ids else 0.0
        else:
            raise ValueError(arm)
        return float(np.log(max(p, 1e-12)))
