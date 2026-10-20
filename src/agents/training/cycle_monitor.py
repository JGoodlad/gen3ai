"""THE CYCLE MONITOR — the rule (``design_evaluation.md`` §2.5; unit U5, offline slice, 2026-10-19).

A pure function of mirrored-pair counts: no I/O, no engine. The driver and reader are ``main.cycle_monitor``; the
plateau's Tier 2 reads this module's verdict (``agents.training.plateau_t2``).

**What it answers.** "Is the run CYCLING (beating its recent past while holes open against older selves), or is
its progress transitive?" — at every check, from the accumulated matrix of THINNED nodes.

**The matrix.** Nodes are the run's periodic checkpoints on the Δ grid (the plateau's node rule,
``plateau_t1.node_at``). At the check of grid step g the newest node plays one ROW: every older node at a lag in
the HYBRID lag set L = {1..8} ∪ {16, 32, 64} × Δ, :data:`MonitorRule.pairs_per_edge` mirrored pairs per edge.
Snapshots are frozen, so old edges never change and the rows accumulate into the matrix; the check at g reads every
monitor row among nodes ≤ g.

**The edge statistic (pair level, §3).** Each edge's pentanomial counts ``c`` (pair scores 0, ¼, ½, ¾, 1 from the
newer node's side) give the Dirichlet(c + ½) posterior mean shape ``q``, the score ``m = Σ q_k s_k``, the pair
variance ``σ² = Σ q_k (s_k − m)²``, the logit ``y = logit m`` and its weight ``w = n (m(1−m))² / σ²``.

**The four signals** (each flags at α = 0.05; the null of (a) and (d) is a PARAMETRIC BOOTSTRAP of
:data:`MonitorRule.n_boot` replicate matrices drawn on the SAME graph and pair counts from the transitive fit — each
edge's own shape TILTED (``sprt.mle_lambda``, the pentanomial MLE at a fixed mean) to the fitted score — and the
replicate goes through the whole pipeline):

* **(a) Hodge cyclic share** — the Pearson χ² of the logit-WLS Bradley–Terry residual (``Σ w (y − (r_i − r_j))²``,
  the curl + harmonic energy). Flags iff its bootstrap p < α.
* **(b) lag-curve non-monotonicity** — on the newest row, ordered by lag, the max over adjacent lags of
  ``z = (m_short − m_long) / SE``. Flags iff max z > Φ⁻¹(1 − α / (k − 1)) (one-sided Bonferroni over the k − 1
  adjacent pairs).
* **(c) intransitive triangles** — every triangle through the newest node with all three edges played: the curl
  ``y_na + y_ab + y_bn`` over its SE, two-sided, Holm at α. Flags iff any triangle is rejected.
* **(d) Nash support** — the support size (members with weight > :data:`MonitorRule.support_w`) of the Nash of the
  symmetric zero-sum meta-game ``A_ij = m_ij − ½`` (missing entries from the BT spine). Flags iff its bootstrap
  p (P_null(size ≥ observed)) < α. REPORTED beside it: the posterior mean support size and P(w_newest > 0.01) over
  :data:`MonitorRule.posterior_draws` Dirichlet draws.

  **Deviation from §2.5, stated (this unit):** the flag statistic is the support of the Nash of the POINT matrix,
  not the posterior-mean support (the latter needs a posterior per bootstrap replicate: 2·10⁶ LPs a check), and the
  Nash is the LP's (HiGHS, deterministic) — on a generic noisy matrix the symmetric game's Nash is unique, so it IS
  the maximum-entropy Nash there; on a degenerate one the LP's vertex can have a SMALLER support than the max-entropy
  point, which errs toward NOT flagging (d). (a)–(c) are as registered.

**The verdict per check** (§2.5): SUSPECTED iff any signal flags; CONFIRMED iff a flagged signal flags again on a
FRESH top-up (:data:`MonitorRule.topup_pairs` more pairs on the edges carrying it, tested on the top-up games ONLY at
α); NONE otherwise. Rule 8: a bootstrap p within one Monte Carlo SE (``√(α(1−α)/R)``) of α, or a z / p within 1e-9
of its bound, is NOT a crossing — and a check with no flag but a signal in its band is INCONCLUSIVE (never NONE);
so is a row whose aborted games exceed 25 % of a cell's attempted ones. A check whose row is not complete is
CONTINUE.

**Determinism.** The bootstrap seed is the sha256 of the rule version and the sorted consumed ids (§9.1); the LP is
HiGHS's dual simplex, deterministic for a given input.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agents.training import sprt

RULE_NAME = "cycle_monitor"
RULE_VERSION = "gen3_cycle_monitor_v1"
PURPOSE = "monitor"
REQUEST_KIND_ROW = "monitor_row"
REQUEST_KIND_TOPUP = "monitor_topup"
DECISION_KIND = "cycle_flag"

NONE, SUSPECTED, CONFIRMED, INCONCLUSIVE = "NONE", "SUSPECTED", "CONFIRMED", "INCONCLUSIVE"
VERDICTS = (NONE, SUSPECTED, CONFIRMED, INCONCLUSIVE)
CONTINUE = "CONTINUE"                 # the check's row (or its top-up) is not complete
NOT_APPLICABLE = "NOT_APPLICABLE"     # the check has fewer than two edges in its row (the first checks)

SIGNALS = ("hodge", "lag", "triangles", "nash")
FLAG, CLEAR, BAND, NA = "FLAG", "CLEAR", "BAND", "NA"

SCORES = np.asarray(sprt.SCORES, dtype=np.float64)       # 0, ¼, ½, ¾, 1
ELO_PER_LOGIT = 400.0 / math.log(10.0)
Z_BAND = 1e-9
BOOT_CHUNK = 250


class MonitorError(ValueError):
    """Rows that are not one monitor's (a regime mix, an unknown node, a disconnected matrix)."""


# ------------------------------------------------------------------------------------------------ the rule + plan
@dataclass(frozen=True)
class MonitorRule:
    alpha: float = 0.05
    n_boot: int = 2000
    pairs_per_edge: int = 500
    batch_pairs: int = 50
    topup_pairs: int = 500
    topup_max_edges: int = 3
    support_w: float = 0.01
    posterior_draws: int = 500
    abort_frac: float = 0.25

    def rule_string(self) -> str:
        return (f"{RULE_NAME}(alpha={self.alpha},n_boot={self.n_boot},pairs={self.pairs_per_edge},"
                f"batch={self.batch_pairs},topup={self.topup_pairs}x{self.topup_max_edges},"
                f"support_w={self.support_w},posterior={self.posterior_draws},abort={self.abort_frac},"
                f"null=parametric_bootstrap_tilted_pentanomial,lag=bonferroni_one_sided,tri=holm_two_sided,"
                f"nash=lp_point_support,band=mc_se|1e-09)")

    @property
    def mc_band(self) -> float:
        return math.sqrt(self.alpha * (1.0 - self.alpha) / self.n_boot)

    def to_json(self) -> Dict[str, Any]:
        return {**asdict(self), "rule": self.rule_string(), "rule_version": RULE_VERSION,
                "registered": self == REGISTERED}


REGISTERED = MonitorRule()


@dataclass(frozen=True)
class MonitorPlan:
    """WHICH edges a check's row holds: the newest node against the node ``lag × Δ`` back for every lag in the
    HYBRID set (§2.5: dense over the last ``dense_lags`` checks, geometric beyond)."""

    delta_steps: int = 10_000_000
    dense_lags: int = 8
    geo_lags: Tuple[int, ...] = (16, 32, 64)
    slack_steps: int = 500_000

    def __post_init__(self) -> None:
        if self.delta_steps < 1 or self.dense_lags < 1:
            raise ValueError("a monitor plan needs delta_steps >= 1 and dense_lags >= 1")
        if not 0 < self.slack_steps < self.delta_steps:
            raise ValueError("a node's slack must be positive and below the check spacing")

    @property
    def lags(self) -> Tuple[int, ...]:
        return tuple(range(1, self.dense_lags + 1)) + tuple(g for g in self.geo_lags if g > self.dense_lags)

    def to_json(self) -> Dict[str, Any]:
        return {"delta_steps": self.delta_steps, "dense_lags": self.dense_lags, "geo_lags": list(self.geo_lags),
                "slack_steps": self.slack_steps, "lags": list(self.lags)}


DEFAULT_PLAN = MonitorPlan()


@dataclass(frozen=True)
class RowPlan:
    grid: int
    newest: Optional[int]                                 # node(grid)'s step
    opponents: Tuple[Tuple[int, Optional[int]], ...]      # (lag, node step or None) per lag reaching ≥ Δ

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """(lag, older step) of every playable edge."""
        return [(lag, s) for lag, s in self.opponents if s is not None] if self.newest is not None else []


def node_at(grid: int, steps: Sequence[int], slack: int) -> Optional[int]:
    cands = [s for s in steps if grid <= s < grid + slack]
    return min(cands) if cands else None


def plan_rows(steps: Iterable[int], plan: MonitorPlan = DEFAULT_PLAN) -> List[RowPlan]:
    """Every check a run's checkpoints reach, oldest first: grid steps g = k·Δ (k ≥ 2) up to the newest checkpoint.
    A lag whose node is missing is listed with ``None`` (never substituted)."""
    ss = sorted(set(int(s) for s in steps))
    out: List[RowPlan] = []
    if not ss:
        return out
    k = 2
    while k * plan.delta_steps <= ss[-1]:
        g = k * plan.delta_steps
        opp = tuple((lag, node_at(g - lag * plan.delta_steps, ss, plan.slack_steps))
                    for lag in plan.lags if g - lag * plan.delta_steps >= plan.delta_steps)
        out.append(RowPlan(grid=g, newest=node_at(g, ss, plan.slack_steps), opponents=opp))
        k += 1
    return out


def row_request_id(run: str, grid: int, plan: MonitorPlan = DEFAULT_PLAN) -> str:
    return f"{REQUEST_KIND_ROW}:{run}:{int(grid)}:d{plan.delta_steps}"


def topup_request_id(run: str, grid: int, plan: MonitorPlan = DEFAULT_PLAN) -> str:
    return f"{REQUEST_KIND_TOPUP}:{run}:{int(grid)}:d{plan.delta_steps}"


# ------------------------------------------------------------------------------------------------ the matrix
@dataclass
class Matrix:
    """Nodes ordered by step (keys are opaque ids: a sha256, or ``snap:<step>``) and edges oriented from the NEWER
    node (index i > j), with the pentanomial counts from the newer node's side."""

    keys: List[str]
    steps: List[int]
    ei: np.ndarray            # newer node index
    ej: np.ndarray            # older node index
    counts: np.ndarray        # (E, 5) float

    @property
    def n(self) -> int:
        return len(self.keys)


def build_matrix(edges: Iterable[Tuple[str, int, str, int, Sequence[float]]]) -> Matrix:
    """``[(key_a, step_a, key_b, step_b, counts5 from a's side), ...]`` → a :class:`Matrix` (an edge seen more than
    once is pooled; orientation flipped to newer-first by reversing the counts)."""
    nodes: Dict[str, int] = {}
    agg: Dict[Tuple[str, str], np.ndarray] = {}
    for ka, sa, kb, sb, c in edges:
        if ka == kb:
            raise MonitorError(f"a self edge on node {ka}")
        for k, s in ((ka, sa), (kb, sb)):
            if nodes.setdefault(k, int(s)) != int(s):
                raise MonitorError(f"node {k} has two steps {nodes[k]} and {s}")
        v = np.asarray(c, dtype=np.float64)
        if v.shape != (5,) or (v < 0).any():
            raise MonitorError(f"edge {ka} vs {kb}: counts {c} are not a pentanomial")
        if (int(sa), ka) < (int(sb), kb):
            ka, kb, v = kb, ka, v[::-1]
        agg[(ka, kb)] = agg.get((ka, kb), np.zeros(5)) + v
    order = sorted(nodes, key=lambda k: (nodes[k], k))
    idx = {k: i for i, k in enumerate(order)}
    pairs = sorted(agg, key=lambda e: (idx[e[0]], idx[e[1]]))
    pairs = [e for e in pairs if agg[e].sum() > 0]
    return Matrix(keys=order, steps=[nodes[k] for k in order],
                  ei=np.asarray([idx[a] for a, _ in pairs], dtype=np.int64),
                  ej=np.asarray([idx[b] for _, b in pairs], dtype=np.int64),
                  counts=np.asarray([agg[e] for e in pairs], dtype=np.float64).reshape(-1, 5))


def edge_stats(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(n, m, var_m, y, w)`` per edge (vectorised over any leading axes of ``counts[..., 5]``): the Dirichlet(c + ½)
    posterior-mean shape's score, the variance of the edge's mean score, its logit and the logit's weight."""
    n = counts.sum(-1)
    q = (counts + 0.5) / (n[..., None] + 2.5)
    m = (q * SCORES).sum(-1)
    var_pair = (q * (SCORES - m[..., None]) ** 2).sum(-1)
    var_m = var_pair / np.maximum(n, 1.0)
    y = np.log(m / (1.0 - m))
    w = (m * (1.0 - m)) ** 2 / var_m
    return n, m, var_m, y, w


def _connected(n: int, ei: np.ndarray, ej: np.ndarray) -> bool:
    parent = list(range(n))

    def f(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(ei.tolist(), ej.tolist()):
        parent[f(a)] = f(b)
    return len({f(i) for i in range(n)}) == 1


def wls_fit(n_nodes: int, ei: np.ndarray, ej: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The logit-WLS Bradley–Terry spine (HodgeRank's gradient part), batched over leading axes of ``y`` / ``w``:
    ratings with node 0 pinned at 0 (a connected graph has one gauge)."""
    lead = y.shape[:-1]
    lap = np.zeros(lead + (n_nodes, n_nodes))
    b = np.zeros(lead + (n_nodes,))
    for e, (i, j) in enumerate(zip(ei.tolist(), ej.tolist())):
        we, ye = w[..., e], y[..., e]
        lap[..., i, i] += we
        lap[..., j, j] += we
        lap[..., i, j] -= we
        lap[..., j, i] -= we
        b[..., i] += we * ye
        b[..., j] -= we * ye
    r = np.zeros(lead + (n_nodes,))
    if n_nodes > 1:
        r[..., 1:] = np.linalg.solve(lap[..., 1:, 1:], b[..., 1:, None])[..., 0]
    return r


def chi2_resid(ei: np.ndarray, ej: np.ndarray, y: np.ndarray, w: np.ndarray, r: np.ndarray) -> np.ndarray:
    res = y - (r[..., ei] - r[..., ej])
    return (w * res ** 2).sum(-1)


def tilt(counts: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Each edge's Dirichlet(c + ½) shape tilted to mean ``mu`` (the pentanomial MLE at a fixed mean)."""
    out = np.zeros_like(counts)
    for e in range(counts.shape[0]):
        q = (counts[e] + 0.5) / (counts[e].sum() + 2.5)
        mu_e = float(min(max(mu[e], 1e-6), 1 - 1e-6))
        lam = sprt.mle_lambda(q.tolist(), mu_e)
        t = q / (1.0 + lam * (SCORES - mu_e))
        out[e] = t / t.sum()
    return out


def nash_lp(a: np.ndarray) -> np.ndarray:
    """A Nash strategy of the symmetric zero-sum game with antisymmetric payoff ``a`` (row's payoff): max v s.t.
    ``aᵀx ≥ v``, ``Σx = 1``, ``x ≥ 0`` — HiGHS dual simplex, deterministic."""
    from scipy.optimize import linprog

    k = a.shape[0]
    c = np.zeros(k + 1)
    c[-1] = -1.0
    a_ub = np.hstack([-a.T, np.ones((k, 1))])
    a_eq = np.hstack([np.ones((1, k)), np.zeros((1, 1))])
    res = linprog(c, A_ub=a_ub, b_ub=np.zeros(k), A_eq=a_eq, b_eq=np.ones(1),
                  bounds=[(0, None)] * k + [(None, None)], method="highs-ds")
    if res.status != 0:
        raise MonitorError(f"the Nash LP failed: {res.message}")
    x = np.clip(res.x[:k], 0.0, None)
    return x / x.sum()


def payoff(n_nodes: int, ei: np.ndarray, ej: np.ndarray, m: np.ndarray, r: np.ndarray) -> np.ndarray:
    """``A_ij = P(i beats j) − ½`` with played edges from ``m`` and the rest from the spine ``r``."""
    d = r[:, None] - r[None, :]
    a = 1.0 / (1.0 + np.exp(-d)) - 0.5
    a[ei, ej] = m - 0.5
    a[ej, ei] = 0.5 - m
    np.fill_diagonal(a, 0.0)
    return a


def _phi_inv(p: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(p))


def _norm_sf(z: float) -> float:
    from scipy.stats import norm

    return float(norm.sf(z))


def _boot_state(p: float, rule: MonitorRule) -> str:
    if abs(p - rule.alpha) <= rule.mc_band:
        return BAND
    return FLAG if p < rule.alpha else CLEAR


# ------------------------------------------------------------------------------------------------ the evaluation
@dataclass
class MonitorResult:
    verdict: str
    signals: Dict[str, str]
    newest: str
    n_nodes: int
    n_edges: int
    detail: Dict[str, Any] = field(default_factory=dict)
    flagged_items: List[Dict[str, Any]] = field(default_factory=list)
    confirmation: Optional[Dict[str, Any]] = None
    reason: str = ""
    spine: Dict[str, float] = field(default_factory=dict)      # the WLS ratings in logits (unrounded), node 0 = 0

    def to_json(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "signals": dict(self.signals), "newest": self.newest,
                "n_nodes": self.n_nodes, "n_edges": self.n_edges, "detail": self.detail,
                "flagged_items": self.flagged_items, "confirmation": self.confirmation, "reason": self.reason}


def seed_of(ids: Iterable[str]) -> int:
    h = hashlib.sha256((RULE_VERSION + "\n" + "\n".join(sorted(ids))).encode()).digest()
    return int.from_bytes(h[:8], "big")


def signals(mx: Matrix, newest: str, rule: MonitorRule = REGISTERED, seed: int = 0) -> MonitorResult:
    """The four signals on matrix ``mx`` at the check whose newest node is ``newest`` (a key of ``mx``)."""
    if newest not in mx.keys:
        raise MonitorError(f"the newest node {newest} has no edge in the matrix")
    k = mx.n
    if not _connected(k, mx.ei, mx.ej):
        raise MonitorError(f"the matrix of {k} nodes is not connected")
    nw = mx.keys.index(newest)
    n, m, var_m, y, w = edge_stats(mx.counts)
    r = wls_fit(k, mx.ei, mx.ej, y, w)
    chi2 = float(chi2_resid(mx.ei, mx.ej, y, w, r))
    pred = r[mx.ei] - r[mx.ej]
    energy = float((w * y ** 2).sum())
    rng = np.random.Generator(np.random.PCG64(seed))
    shapes = tilt(mx.counts, 1.0 / (1.0 + np.exp(-pred)))
    nb = rule.n_boot
    draws = np.stack([rng.multinomial(int(n[e]), shapes[e], size=nb) for e in range(len(n))], axis=1).astype(float)
    _bn, bm, _bv, by, bw = edge_stats(draws)
    br = np.concatenate([wls_fit(k, mx.ei, mx.ej, by[c:c + BOOT_CHUNK], bw[c:c + BOOT_CHUNK])
                         for c in range(0, nb, BOOT_CHUNK)])          # chunked: a large graph's Laplacians
    bchi = chi2_resid(mx.ei, mx.ej, by, bw, br)
    p_a = float((1 + (bchi >= chi2 - 1e-12).sum()) / (nb + 1))
    sig: Dict[str, str] = {"hodge": _boot_state(p_a, rule)}
    detail: Dict[str, Any] = {"hodge": {"chi2": round(chi2, 4), "df": int(len(n) - (k - 1)), "p_boot": round(p_a, 5),
                                        "null_q95": round(float(np.quantile(bchi, 0.95)), 4),
                                        "cyclic_share": round(chi2 / energy, 5) if energy > 0 else None,
                                        # REPORTED effect size (never a verdict input): the w-weighted RMS of the
                                        # residual above the null's mean, in pp of score at ½ (25 pp per logit)
                                        "excess_rms_pp": round(25.0 * math.sqrt(max(0.0, chi2 - float(bchi.mean()))
                                                                                / float(w.sum())), 3)},
                              "ratings_elo": {mx.keys[i]: round(float(r[i] - r[nw]) * ELO_PER_LOGIT, 1)
                                              for i in range(k)}}
    items: List[Dict[str, Any]] = []
    resid_z = (y - pred) * np.sqrt(w)

    # (b) the lag curve of the newest row: its edges ordered by the opponent's step, newest-first = shortest lag
    row = [e for e in range(len(n)) if mx.ei[e] == nw]
    row.sort(key=lambda e: -mx.steps[mx.ej[e]])
    curve = [{"opponent": mx.keys[mx.ej[e]], "opponent_step": mx.steps[mx.ej[e]], "lag_steps":
              mx.steps[nw] - mx.steps[mx.ej[e]], "score": round(float(m[e]), 5),
              "se": round(float(math.sqrt(var_m[e])), 5), "pairs": int(n[e])} for e in row]
    if len(row) >= 2:
        zcrit = _phi_inv(1.0 - rule.alpha / (len(row) - 1))
        zs = [float((m[a] - m[b]) / math.sqrt(var_m[a] + var_m[b])) for a, b in zip(row, row[1:])]
        zi = int(np.argmax(zs))
        zmax = zs[zi]
        sig["lag"] = BAND if abs(zmax - zcrit) <= Z_BAND else (FLAG if zmax > zcrit else CLEAR)
        detail["lag"] = {"curve": curve, "adjacent_z": [round(z, 4) for z in zs], "z_max": round(zmax, 4),
                         "z_crit": round(zcrit, 4)}
        if sig["lag"] == FLAG:
            items.append({"signal": "lag", "edges": [[mx.keys[nw], mx.keys[mx.ej[row[zi]]]],
                                                     [mx.keys[nw], mx.keys[mx.ej[row[zi + 1]]]]],
                          "short": mx.keys[mx.ej[row[zi]]], "long": mx.keys[mx.ej[row[zi + 1]]], "z": round(zmax, 4)})
    else:
        sig["lag"] = NA
        detail["lag"] = {"curve": curve}

    # (c) triangles through the newest node, Holm two-sided
    eid = {(int(a), int(b)): e for e, (a, b) in enumerate(zip(mx.ei.tolist(), mx.ej.tolist()))}
    opp = [int(mx.ej[e]) for e in row]
    tris: List[Tuple[float, float, int, int]] = []
    for ia, a in enumerate(opp):
        for b in opp[ia + 1:]:
            hi, lo = (a, b) if mx.steps[a] > mx.steps[b] or (mx.steps[a] == mx.steps[b] and a > b) else (b, a)
            e_ab = eid.get((hi, lo))
            if e_ab is None:
                continue
            e_nh, e_nl = eid[(nw, hi)], eid[(nw, lo)]
            curl = float(y[e_nh] + y[e_ab] - y[e_nl])         # n > hi > lo > n
            se = float(math.sqrt(1 / w[e_nh] + 1 / w[e_ab] + 1 / w[e_nl]))
            tris.append((curl / se, curl, hi, lo))
    if tris:
        ps = sorted(((2 * _norm_sf(abs(z)), z, c, hi, lo) for z, c, hi, lo in tris), key=lambda t: t[0])
        rejected: List[Tuple[float, float, float, int, int]] = []
        band = False
        for i_t, t in enumerate(ps):
            thr = rule.alpha / (len(ps) - i_t)
            if abs(t[0] - thr) <= Z_BAND:
                band = True
                break
            if t[0] < thr:
                rejected.append(t)
            else:
                break
        sig["triangles"] = FLAG if rejected else (BAND if band else CLEAR)
        detail["triangles"] = {"n": len(ps), "min_p": round(ps[0][0], 6), "rejected": [
            {"nodes": [mx.keys[nw], mx.keys[hi], mx.keys[lo]], "curl": round(c, 4), "z": round(z, 3),
             "p": round(p, 6)} for p, z, c, hi, lo in rejected]}
        if rejected:
            p, z, c, hi, lo = rejected[0]
            items.append({"signal": "triangles", "edges": [[mx.keys[nw], mx.keys[hi]], [mx.keys[hi], mx.keys[lo]],
                                                           [mx.keys[nw], mx.keys[lo]]],
                          "nodes": [mx.keys[nw], mx.keys[hi], mx.keys[lo]], "z": round(z, 4)})
    else:
        sig["triangles"] = NA
        detail["triangles"] = {"n": 0}

    # (d) Nash support of the point matrix vs its bootstrap null; the posterior reported
    x = nash_lp(payoff(k, mx.ei, mx.ej, m, r))
    supp = int((x > rule.support_w).sum())
    bsupp = np.asarray([int((nash_lp(payoff(k, mx.ei, mx.ej, bm[i], br[i])) > rule.support_w).sum())
                        for i in range(nb)])
    p_d = float((1 + (bsupp >= supp).sum()) / (nb + 1))
    sig["nash"] = _boot_state(p_d, rule)
    post = np.stack([rng.dirichlet(mx.counts[e] + 0.5, size=rule.posterior_draws) for e in range(len(n))], axis=1)
    pm = (post * SCORES).sum(-1)
    pxs = [nash_lp(payoff(k, mx.ei, mx.ej, pm[i], r)) for i in range(rule.posterior_draws)]
    detail["nash"] = {"support": supp, "p_boot": round(p_d, 5), "null_q95": float(np.quantile(bsupp, 0.95)),
                      "weights": {mx.keys[i]: round(float(x[i]), 4) for i in range(k) if x[i] > 1e-6},
                      "posterior_mean_support": round(float(np.mean([(v > rule.support_w).sum() for v in pxs])), 3),
                      "p_newest_in_support": round(float(np.mean([v[nw] > rule.support_w for v in pxs])), 4),
                      "entropy": round(float(-(x[x > 0] * np.log(x[x > 0])).sum()), 4)}
    if sig["hodge"] == FLAG:
        top = np.argsort(-np.abs(resid_z), kind="stable")[: rule.topup_max_edges]
        items.append({"signal": "hodge", "edges": [[mx.keys[mx.ei[e]], mx.keys[mx.ej[e]]] for e in top],
                      "sign": [int(np.sign(resid_z[e])) for e in top]})
    if sig["nash"] == FLAG:
        sup = set(np.nonzero(x > rule.support_w)[0].tolist())
        cand = [e for e in range(len(n)) if int(mx.ei[e]) in sup and int(mx.ej[e]) in sup]
        cand.sort(key=lambda e: (-abs(float(resid_z[e])), e))
        top = cand[: rule.topup_max_edges]
        if top:
            items.append({"signal": "nash", "edges": [[mx.keys[mx.ei[e]], mx.keys[mx.ej[e]]] for e in top],
                          "sign": [int(np.sign(resid_z[e])) for e in top]})
    detail["spine"] = {"resid_z_max": round(float(np.max(np.abs(resid_z))), 3)}
    states = set(sig.values())
    verdict = SUSPECTED if FLAG in states else (INCONCLUSIVE if BAND in states else NONE)
    reason = ("flags: " + ", ".join(s for s in SIGNALS if sig.get(s) == FLAG) if verdict == SUSPECTED else
              "a signal within its rounding band: " + ", ".join(s for s in SIGNALS if sig.get(s) == BAND)
              if verdict == INCONCLUSIVE else "no signal flags")
    return MonitorResult(verdict=verdict, signals=sig, newest=newest, n_nodes=k, n_edges=int(len(n)), detail=detail,
                         flagged_items=items, reason=reason, spine={mx.keys[i]: float(r[i]) for i in range(k)})


def topup_edges(res: MonitorResult, rule: MonitorRule = REGISTERED) -> List[Tuple[str, str]]:
    """The (newer, older) edges a SUSPECTED check tops up: each flagged item's edges, deduplicated in order, at most
    ``topup_max_edges`` per item."""
    out: List[Tuple[str, str]] = []
    for it in res.flagged_items:
        for a, b in it["edges"]:
            if (a, b) not in out:
                out.append((a, b))
    return out


def confirm(res: MonitorResult, topup: Mapping[Tuple[str, str], Sequence[float]],
            rule: MonitorRule = REGISTERED) -> Dict[str, Any]:
    """Test each flagged item again on the TOP-UP games only (independent of the selection), one-sided in the
    flagged direction at α: the lag pair's z; the triangle's curl z; for (a) / (d), the residual edges' signed
    z-sum against the ORIGINAL spine. ``topup``: (newer, older) → counts. CONFIRMED iff any item confirms."""
    zc = _phi_inv(1.0 - rule.alpha)
    out: List[Dict[str, Any]] = []

    def st(a: str, b: str) -> Tuple[float, float, float, float]:
        c = np.asarray(topup[(a, b)], dtype=float)[None, :]
        _n, m, v, y, w = edge_stats(c)
        return float(m[0]), float(v[0]), float(y[0]), float(w[0])

    for it in res.flagged_items:
        edges = [tuple(e) for e in it["edges"]]
        if any(e not in topup or sum(topup[e]) < rule.topup_pairs for e in edges):     # type: ignore[index]
            return {"complete": False}
        if it["signal"] == "lag":
            (m1, v1, _, _), (m2, v2, _, _) = st(*edges[0]), st(*edges[1])
            z = (m1 - m2) / math.sqrt(v1 + v2)
        elif it["signal"] == "triangles":
            s = [st(*e) for e in edges]
            curl = s[0][2] + s[1][2] - s[2][2]
            z = math.copysign(1.0, it["z"]) * curl / math.sqrt(sum(1 / x[3] for x in s))
        else:
            zs = []
            for (a, b), sg in zip(edges, it["sign"]):
                _m, _v, yv, wv = st(a, b)
                spine = res.spine[a] - res.spine[b]
                zs.append(sg * (yv - spine) * math.sqrt(wv))
            z = sum(zs) / math.sqrt(len(zs))
        state = BAND if abs(z - zc) <= Z_BAND else (FLAG if z > zc else CLEAR)
        out.append({"signal": it["signal"], "z": round(z, 4), "z_crit": round(zc, 4), "state": state})
    return {"complete": True, "items": out, "confirmed": any(o["state"] == FLAG for o in out)}


def final_verdict(res: MonitorResult, conf: Optional[Mapping[str, Any]]) -> Tuple[str, bool]:
    """``(verdict, final)``: a SUSPECTED check is final once its top-up is complete (CONFIRMED or SUSPECTED)."""
    if res.verdict != SUSPECTED:
        return res.verdict, True
    if conf is None or not conf.get("complete"):
        return SUSPECTED, False
    return (CONFIRMED if conf["confirmed"] else SUSPECTED), True


# ------------------------------------------------------------------------------------------------ ledger rows
def _side_key(p: Mapping[str, Any]) -> str:
    return str(p.get("sha256") or f"id:{p['id']}")


def edges_from_rows(rows: Iterable[Mapping[str, Any]]) -> List[Tuple[str, int, str, int, List[float]]]:
    out = []
    for r in rows:
        out.append((_side_key(r["player"]), int(r["player"]["step"]), _side_key(r["opponent"]),
                    int(r["opponent"]["step"]), [float(c) for c in r["pairs"]["counts"]]))
    return out


def aborted_share(rows: Iterable[Mapping[str, Any]]) -> float:
    """The largest per-cell share of attempted games aborted (§9.1's 25 % rule)."""
    by: Dict[Tuple[str, str], List[int]] = {}
    for r in rows:
        c = r["counts"]
        k = (_side_key(r["player"]), _side_key(r["opponent"]))
        t = by.setdefault(k, [0, 0])
        t[0] += int(c.get("aborted", 0))
        t[1] += int(c["w"]) + int(c["l"]) + int(c["d"]) + int(c.get("aborted", 0))
    return max((a / t for a, t in by.values() if t > 0), default=0.0)


def evaluate_rows(rows: Sequence[Mapping[str, Any]], newest: str, topup_rid: str,
                  rule: MonitorRule = REGISTERED) -> Tuple[MonitorResult, Optional[Dict[str, Any]], str, bool]:
    """The check's verdict from its consumed rows: the matrix = every ``monitor_row`` row (nodes ≤ the newest),
    the confirmation = the rows of ``topup_rid`` only. Returns (result, confirmation, verdict, final)."""
    mat = [r for r in rows if (r.get("request") or {}).get("kind") == REQUEST_KIND_ROW]
    top = [r for r in rows if (r.get("request") or {}).get("id") == topup_rid]
    if not mat:
        raise MonitorError("no monitor_row rows")
    regimes = {r["regime"]["regime_id"] for r in rows}
    if len(regimes) > 1:
        raise MonitorError(f"rows of {len(regimes)} regimes")
    res = signals(build_matrix(edges_from_rows(mat)), newest, rule, seed_of(r["row_id"] for r in mat))
    newest_rows = [r for r in mat if _side_key(r["player"]) == newest]
    if aborted_share(newest_rows) > rule.abort_frac or (top and aborted_share(top) > rule.abort_frac):
        res.verdict, res.reason = INCONCLUSIVE, f"aborted games exceed {rule.abort_frac:.0%} of a cell"
        return res, None, INCONCLUSIVE, True
    conf = None
    if res.verdict == SUSPECTED:
        tc: Dict[Tuple[str, str], np.ndarray] = {}
        for r in top:
            k = (_side_key(r["player"]), _side_key(r["opponent"]))
            tc[k] = tc.get(k, np.zeros(5)) + np.asarray(r["pairs"]["counts"], dtype=float)
        conf = confirm(res, {k: v.tolist() for k, v in tc.items()}, rule)
        res.confirmation = conf
    verdict, final = final_verdict(res, conf)
    return res, conf, verdict, final


def rederive(decision: Mapping[str, Any], rows: List[Dict[str, Any]]) -> Optional[str]:
    """``eval_ledger verify``'s hook: re-derive a ``cycle_flag`` decision of THIS rule from its consumed rows."""
    if decision.get("rule") != REGISTERED.rule_string() or decision.get("rule_version") != RULE_VERSION:
        return None
    rid = str(decision["request_id"])
    topup = rid.replace(REQUEST_KIND_ROW + ":", REQUEST_KIND_TOPUP + ":", 1)
    _res, _c, verdict, _final = evaluate_rows(rows, str(decision["subject"]), topup, REGISTERED)
    return verdict
