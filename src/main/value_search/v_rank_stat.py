"""V-RANK's estimator (``designs/endstate/design_v_rank_on_u.md`` §4; pure numpy, no game, no model): the within-node
slope of the real outcome on the search's value estimate, on unfamiliar teams (U) against home teams.

A POSITION is one decision of ours in a raw P* vs Kakuna game with every legal action played out as a real game: π's
own action by the raw game, every other one by a forced branch of it (:mod:`main.value_search.v_rank`). With Q̂(a) the
search's value of action a and y(a) = 1 / ½ / 0 the real game's result after a:

    q_c = Q̂(a) - mean_a Q̂        y_c = y(a) - mean_a y        (centred WITHIN the position)
    β_panel = Σ_positions Σ_a q_c · y_c  /  Σ_positions Σ_a q_c²
    β_U = the mean of the U panels' β (equal weight)      β_home = the home panel's      Δβ = β_U - β_home

A level bias of Q̂ (its mean over a position's actions) cancels in q_c, so β reads only how Q̂ ORDERS AND SPACES the
siblings a one-ply search chooses between: β = 1 is a calibrated spread, β = 0 no ranking information.

INTERVALS: TEAM-CLUSTERED (``designs/research_state/README.md``: any Kakuna number carries its team-clustered
interval). A percentile bootstrap over the TEAMS WE PILOT (a team with every deal it played, each deal with all its
positions and branches), within panel, B replicates from ONE ``numpy.random.default_rng(seed)``; each replicate draws
the U panels in their registered order, then the home panel; bounds 2.5 / 97.5. A 20-game unit deals each of its ten
teams TWICE, so a deal is not an independent draw: the design's deal bootstrap is REPORTED (``beta_deal_clustered``),
never decided on.

CLASSES (:func:`classes`), from the bounds of β_U and Δβ:

* ``RANK-NULL-ON-U``  UB(β_U) < ``null_bar``. It DOMINATES (Δβ's own class is recorded beside it).
* ``RANK-HOLDS-ON-U`` LB(Δβ) > -``margin`` and LB(β_U) > 0 (a Δβ whose UB < 0 inside the margin reads HOLDS, noted).
* ``RANK-WORSE-ON-U`` UB(Δβ) < 0 and not HOLDS.
* ``UNDECIDED``       otherwise.

Look 1 reads NULL only (NULL stops the read; anything else is ``CONTINUE``); look 2 reads every class. RULE 8: every
inequality is evaluated twice, once needing to clear its bar by more than ``eps`` and once allowed to miss it by less
than ``eps``; when the two classes differ a bound sits within ``eps`` of a bar that matters, and the read is
``UNDECIDED`` (``CONTINUE`` at look 1), flagged ``boundary``.

REPORTED, never decided (:func:`reported`; the design's §4.4): β by stratum, β inside the contenders, the partial slope
given π's logits, α_top, H, the piKL pick's IMP, the self-referee decomposition, Q̂'s Monte Carlo reliability and the
slope corrected for it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "gen3_v_rank_read_v1"
NULL, WORSE, HOLDS, UNDECIDED, CONTINUE = ("RANK-NULL-ON-U", "RANK-WORSE-ON-U", "RANK-HOLDS-ON-U", "UNDECIDED",
                                           "CONTINUE")
OUTCOME = {"+": 1.0, "0": 0.5, "-": 0.0}
#: the contenders (§4.4): π's top ``C_PI`` by logit and Q̂'s top ``C_Q``
C_PI, C_Q = 3, 2


class VRankError(RuntimeError):
    """An input the estimator cannot read honestly: named, never skipped."""


# ------------------------------------------------------------------------------------------- per-position parts
def within(x: np.ndarray) -> np.ndarray:
    return x - x.mean()


def slope_parts(q: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """``(Σ q_c y_c, Σ q_c²)`` of one position."""
    qc = within(q)
    return float(qc @ within(y)), float(qc @ qc)


def partial_parts(q: np.ndarray, lg: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """The within-position partial slope of y on q given the logits: q's residual on the logits, against y."""
    qc, lc = within(q), within(lg)
    r = qc if lc @ lc < 1e-12 else qc - (qc @ lc) / (lc @ lc) * lc
    return float(r @ within(y)), float(r @ r)


def contenders(lg: np.ndarray, q: np.ndarray) -> List[int]:
    o = list(np.argsort(-lg, kind="stable")[:C_PI]) + list(np.argsort(-q, kind="stable")[:C_Q])
    return sorted({int(i) for i in o})


def mean_noise(samples: np.ndarray) -> float:
    """The noise Σ_a Var(centred mean of a) of a position's [A, S] samples whose column s is ONE draw shared by the
    actions (a world, a referee seed): centre each column within the position, then the sample variance over columns
    of each action's centred value, over S. It is what Σ q_c² holds of pure sampling noise (common draws included)."""
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] < 2:
        raise VRankError(f"noise needs [A, S >= 2] samples, got {s.shape}")
    c = s - s.mean(0, keepdims=True)
    return float(c.var(1, ddof=1).sum() / s.shape[1])


# ---------------------------------------------------------------------------------------------- the positions
def build_positions(scored: Sequence[Mapping[str, Any]], branches: Sequence[Mapping[str, Any]]
                    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """``(positions, counts)``. One position per SEARCHED scored state whose every branch (and so its raw game)
    counted: its legal actions in ascending order with Q̂, log π, y (π's action from the raw game), the self-referee's
    Q_self and both noises. ``counts[panel]`` = scored / excluded (no consistent world: no Q̂, no branch) / searched /
    complete / incomplete (a branch never counted) and why. A pending branch, or a branch set that is not exactly the
    position's other legal actions, raises."""
    by: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for r in branches:
        if r["kind"] != "branch":
            continue
        if int(r["action"]) in by.setdefault(r["key"], {}):
            raise VRankError(f"{r['key']}: action {r['action']} branched twice")
        by[r["key"]][int(r["action"])] = r
    out: List[Dict[str, Any]] = []
    counts: Dict[str, Dict[str, int]] = {}
    seen = set()
    for s in scored:
        if s["key"] in seen:
            raise VRankError(f"{s['key']}: scored twice")
        seen.add(s["key"])
        c = counts.setdefault(s["panel"], {"scored": 0, "excluded": 0, "searched": 0, "complete": 0, "incomplete": 0,
                                           "peer_dropped": 0})
        c["scored"] += 1
        se = s["search"]
        if not se.get("searched"):
            c["excluded"] += 1
            if s["key"] in by:
                raise VRankError(f"{s['key']}: branches of a state the search excluded")
            continue
        c["searched"] += 1
        acts = sorted(int(a) for a in se["q"])
        top = int(s["pi_argmax"])
        br = by.get(s["key"], {})
        if top not in acts or sorted(br) != [a for a in acts if a != top]:
            raise VRankError(f"{s['key']}: its branches {sorted(br)} are not its legal actions {acts} less π's {top}")
        pend = [a for a, r in br.items() if r["status"] == "pending"]
        if pend:
            raise VRankError(f"{s['key']}: {len(pend)} branch(es) not yet played to a verdict")
        if any(r["status"] != "counted" for r in br.values()):
            c["incomplete"] += 1
            c["peer_dropped"] += int(any(r.get("peer_drop") for r in br.values()))
            continue
        c["complete"] += 1
        y_raw = {float(r["y_raw"]) for r in br.values()}
        if len(y_raw) != 1:
            raise VRankError(f"{s['key']}: its branches disagree on the raw game's outcome")
        y = [y_raw.copy().pop() if a == top else float(br[a]["y_branch"]) for a in acts]
        if not s.get("team"):
            raise VRankError(f"{s['key']}: no team (the cluster of every interval)")
        pos: Dict[str, Any] = {"key": s["key"], "panel": s["panel"], "deal": s["deal"], "team": str(s["team"]),
                               "stratum": s["stratum"],
                               "turn": int(s["turn"]), "actions": acts, "top": acts.index(top),
                               "pick": acts.index(int(se["played"])),
                               "q": np.asarray([float(se["q"][str(a)]) for a in acts]),
                               "lg": np.asarray([float(se["log_pi"][str(a)]) for a in acts]),
                               "y": np.asarray(y)}
        ref = s.get("referee")
        if ref is not None:
            samples = np.asarray([[OUTCOME[ch] for ch in ref["out"][str(a)]] for a in acts])
            pos["q_self"], pos["self_noise"] = samples.mean(1), mean_noise(samples)
        if s.get("worlds") is not None:
            pos["q_noise"] = mean_noise(np.asarray([s["worlds"][str(a)] for a in acts], dtype=np.float64))
        out.append(pos)
    stray = sorted(set(by) - seen)
    if stray:
        raise VRankError(f"{len(stray)} branched state(s) never scored (first {stray[0]})")
    return out, counts


# ---------------------------------------------------------------------------------------------- the bootstrap
def _ratio(v: np.ndarray) -> float:
    return float(v[0] / v[1]) if v[1] > 0 else float("nan")


def boot(vecs: Mapping[str, Mapping[str, np.ndarray]], u: Sequence[str], home: str, *, B: int, seed: int,
         f: Callable[[np.ndarray], float] = _ratio) -> Dict[str, Any]:
    """The cluster bootstrap of ``f(Σ_clusters vec)`` (module docstring). ``vecs[panel][cluster]`` = the cluster's
    summed vector (:func:`cluster_vecs`). Returns ``{"per_panel": {p: {est, lo, hi, clusters}}, "u", "home",
    "delta"}`` (each ``{est, lo, hi}``). A panel with no cluster, or a replicate ``f`` cannot evaluate (NaN: an empty
    denominator), raises."""
    panels = [*u, home]
    arr = {}
    for p in panels:
        ds = sorted(vecs.get(p, {}))
        if not ds:
            raise VRankError(f"panel {p}: no cluster enters this estimate")
        arr[p] = np.stack([np.asarray(vecs[p][d], dtype=np.float64) for d in ds])
    point = np.array([f(arr[p].sum(0)) for p in panels])
    rng = np.random.default_rng(int(seed))
    reps = np.empty((int(B), len(panels)))
    for b in range(int(B)):
        for j, p in enumerate(panels):
            k = len(arr[p])
            reps[b, j] = f(arr[p][rng.integers(0, k, k)].sum(0))
    if np.isnan(point).any() or np.isnan(reps).any():
        raise VRankError("a bootstrap replicate has an empty denominator (too few clusters to resample)")

    def tri(est: float, x: np.ndarray) -> Dict[str, float]:
        return {"est": float(est), "lo": float(np.percentile(x, 2.5)), "hi": float(np.percentile(x, 97.5))}

    nu = len(u)
    ur, hr = reps[:, :nu].mean(1), reps[:, nu]
    return {"per_panel": {p: {**tri(point[j], reps[:, j]), "clusters": int(len(arr[p]))} for j, p in enumerate(panels)},
            "u": tri(point[:nu].mean(), ur), "home": tri(point[nu], hr), "delta": tri(point[:nu].mean() - point[nu],
                                                                                    ur - hr)}


def cluster_vecs(positions: Sequence[Mapping[str, Any]],
                 part: Callable[[Mapping[str, Any]], Optional[Sequence[float]]], by: str = "team"
                 ) -> Dict[str, Dict[str, np.ndarray]]:
    """``{panel: {cluster: Σ part(position)}}``, the cluster a position's ``by`` (``team``: the registered one;
    ``deal``: the design's, REPORTED). A position whose ``part`` is None adds nothing (its cluster still appears only
    if another of its positions does)."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for x in positions:
        v = part(x)
        if v is None:
            continue
        d = out.setdefault(x["panel"], {})
        d[x[by]] = d.get(x[by], 0.0) + np.asarray(v, dtype=np.float64)
    return out


# ------------------------------------------------------------------------------------------------ the classes
def classes(beta_u: Mapping[str, float], delta: Mapping[str, float], *, null_bar: float, margin: float, eps: float,
            look: int) -> Dict[str, Any]:
    """The registered class (module docstring): ``{"class", "boundary", "delta_class", "note"}``."""
    if look not in (1, 2):
        raise VRankError(f"look {look!r}: 1 or 2")

    def at(e: float, *, with_null: bool) -> str:
        null = beta_u["hi"] < null_bar - e
        holds = delta["lo"] > -margin + e and beta_u["lo"] > e
        worse = delta["hi"] < -e
        if with_null and null:
            return NULL
        if look == 1 and with_null:
            return CONTINUE
        return HOLDS if holds else WORSE if worse else UNDECIDED

    strict, loose = at(eps, with_null=True), at(-eps, with_null=True)
    boundary = strict != loose
    cls = strict if not boundary else (CONTINUE if look == 1 else UNDECIDED)
    ds, dl = at(eps, with_null=False), at(-eps, with_null=False)
    note = None
    if cls == HOLDS and delta["hi"] < -eps:
        note = "worse, within the margin"
    # look 1 reads NULL only: unless it stops the read, Δβ's class is not looked at (it is look 2's)
    dcls = None if cls == CONTINUE else ds if ds == dl else UNDECIDED
    return {"class": cls, "boundary": boundary, "delta_class": dcls, "note": note}


# ------------------------------------------------------------------------------------------------- reported
def _top_anchored(x: Mapping[str, Any]) -> Optional[List[float]]:
    d = np.delete(x["q"] - x["q"][x["top"]], x["top"])
    t = np.delete(x["y"] - x["y"][x["top"]], x["top"])
    return [float(len(d)), float(d.sum()), float(t.sum()), float(d @ d), float(d @ t)]


def _alpha_top(v: np.ndarray) -> float:
    """π's top over an alternative Q̂ rates EQUAL: minus the intercept of (y_alt - y_top) on (Q̂_alt - Q̂_top)."""
    n, sd, st, sdd, sdt = v
    den = n * sdd - sd * sd
    if n <= 0 or den <= 0:
        return float("nan")
    beta = (n * sdt - sd * st) / den
    return float(-(st - beta * sd) / n)


def _h(x: Mapping[str, Any]) -> Optional[List[float]]:
    top = x["top"]
    pref = [j for j in contenders(x["lg"], x["q"]) if j != top and x["q"][j] > x["q"][top]]
    return [float(np.mean([x["y"][j] - x["y"][top] for j in pref])), 1.0] if pref else None


def _contender_parts(x: Mapping[str, Any]) -> Optional[List[float]]:
    c = contenders(x["lg"], x["q"])
    return list(slope_parts(x["q"][c], x["y"][c])) if len(c) >= 2 else None


def reported(positions: Sequence[Mapping[str, Any]], u: Sequence[str], home: str, *, B: int, seed: int
             ) -> Dict[str, Any]:
    """The design's §4.4 rows, each the same team bootstrap (its own stream from ``seed``). A row that cannot be
    estimated (a panel with no position for it) is ``{"error": ...}``: a REPORTED row never stops the read."""
    def row(part: Callable[[Mapping[str, Any]], Optional[Sequence[float]]], f: Callable[[np.ndarray], float] = _ratio,
            pos: Optional[Sequence[Mapping[str, Any]]] = None, by: str = "team") -> Dict[str, Any]:
        try:
            return boot(cluster_vecs(positions if pos is None else pos, part, by), u, home, B=B, seed=seed, f=f)
        except VRankError as exc:
            return {"error": str(exc)}

    rep: Dict[str, Any] = {"beta_deal_clustered": row(lambda x: slope_parts(x["q"], x["y"]), by="deal")}
    for name in ("A", "B"):
        rep[f"beta_stratum_{name}"] = row(lambda x: slope_parts(x["q"], x["y"]),
                                          pos=[x for x in positions if x["stratum"] == name])
    rep["beta_contenders"] = row(_contender_parts)
    rep["beta_partial_given_pi"] = row(lambda x: partial_parts(x["q"], x["lg"], x["y"]))
    rep["alpha_top"] = row(_top_anchored, _alpha_top)
    rep["H"] = row(_h)
    rep["imp_pikl_pick"] = row(lambda x: [float(x["y"][x["pick"]] - x["y"][x["top"]]), 1.0])
    rep["pikl_deviation_rate"] = row(lambda x: [float(x["pick"] != x["top"]), 1.0])
    rep["sum_qc2_per_position"] = row(lambda x: [slope_parts(x["q"], x["y"])[1], 1.0])
    rep["legal_actions"] = row(lambda x: [float(len(x["actions"])), 1.0])
    ref = [x for x in positions if "q_self" in x]
    rep["self_referee"] = {
        "positions": len(ref),
        "beta_self_on_qhat": row(lambda x: slope_parts(x["q"], x["q_self"]), pos=ref),
        "y_on_q_self": row(lambda x: slope_parts(x["q_self"], x["y"]), pos=ref),
        "y_on_q_self_corrected": row(lambda x: [slope_parts(x["q_self"], x["y"])[0],
                                                slope_parts(x["q_self"], x["y"])[1] - x["self_noise"]], pos=ref),
        "q_self_reliability": row(lambda x: [slope_parts(x["q_self"], x["y"])[1] - x["self_noise"],
                                             slope_parts(x["q_self"], x["y"])[1]], pos=ref)}
    wn = [x for x in positions if "q_noise" in x]
    rep["qhat_monte_carlo"] = {
        "positions": len(wn),
        "reliability": row(lambda x: [slope_parts(x["q"], x["y"])[1] - x["q_noise"], slope_parts(x["q"], x["y"])[1]],
                           pos=wn),
        "beta_corrected": row(lambda x: [slope_parts(x["q"], x["y"])[0],
                                         slope_parts(x["q"], x["y"])[1] - x["q_noise"]], pos=wn)}
    return rep


# ----------------------------------------------------------------------------------------------------- read
def read(positions: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> Dict[str, Any]:
    """The registered decision and the REPORTED rows. ``params``: ``u_panels``, ``home_panel``, ``look``,
    ``null_bar``, ``margin``, ``rule8_eps``, ``bootstrap`` {B, seed}."""
    u, home = list(params["u_panels"]), str(params["home_panel"])
    B, seed = int(params["bootstrap"]["B"]), int(params["bootstrap"]["seed"])
    primary = boot(cluster_vecs(positions, lambda x: slope_parts(x["q"], x["y"])), u, home, B=B, seed=seed)
    cls = classes(primary["u"], primary["delta"], null_bar=float(params["null_bar"]), margin=float(params["margin"]),
                  eps=float(params["rule8_eps"]), look=int(params["look"]))
    n = {p: sum(1 for x in positions if x["panel"] == p) for p in [*u, home]}
    return {"schema": SCHEMA, "look": int(params["look"]), **cls,
            "rule": (f"β = Σ q_c·y_c / Σ q_c² within position, teams bootstrapped; NULL iff UB(β_U) < "
                     f"{params['null_bar']}; HOLDS iff LB(Δβ) > -{params['margin']} and LB(β_U) > 0; WORSE iff "
                     f"UB(Δβ) < 0 and not HOLDS; look 1 reads NULL only; rule 8 eps {params['rule8_eps']}"),
            "primary": {"beta_u": primary["u"], "beta_home": primary["home"], "delta": primary["delta"],
                        "per_panel": primary["per_panel"]},
            "positions": n, "reported": reported(positions, u, home, B=B, seed=seed)}
