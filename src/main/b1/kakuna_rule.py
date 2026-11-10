r"""BOUNDARY-1's Kakuna meter as a ``main.reads`` rule ``b1_kakuna``: registered CONTRASTS of arm means on Kakuna
panels, each with an ANALYTIC two-way (seeds x teams) interval. The estimators are the clustering addendum's
(``measurements/sprime_honest_play_2026-11-06/addendum_clustering/clustering.py``, landed 96adb152c): the same clusters,
the same G / (G - 1) correction, panels as fixed strata, the same dyadic formula; the test reproduces its ``our_team``
and ``dyadic`` standard errors from its ``deals.csv``.

THE STATISTIC. A contrast k registers arm coefficients c_a (summing to 0) and a panel list P_k. A run r of arm a has
weight w_r = c_a / n_a (n_a = the arm's runs in the group), panel rate R_{r,p} = wins / games (a tie a non-win) and
score S_r = sum over p in P_k of R_{r,p} / \|P_k\|. psi = sum_r w_r S_r.

THE VARIANCE, once per team scheme (:data:`SCHEMES`):

* V_S (seeds): the pooled within-arm sample variance of S_r, s2 = sum_a (n_a - 1) s_a^2 / df_S, times sum_r w_r^2;
  df_S = sum_a (n_a - 1) over the contrast's arms (a contrast with df_S < 1 is refused at spec load);
* V_T (teams, runs fixed): the cluster-robust sandwich of the per-game linear term psi_g = w_r (1 / \|P_k\|)
  (y_g - R_{r,p}) / n_{r,p}, clusters taken ACROSS runs. ``our_team``: a cluster is (panel, the team we pilot).
  ``dyadic``: Aronow-Samii-Assenova, a team is one cluster in either role within its panel (every pair of games
  sharing a team in any role). A panel is a stratum, each with G / (G - 1) on its OBSERVED clusters;
* V_TS: the same estimator with every cluster split by run, summed over runs;
* V = max(V_S, V_TS) + max(0, V_T - V_TS); df = Satterthwaite V^2 / (A^2 / df_S + B^2 / df_T) on those two parts,
  df_T = sum over the contrast's panels of (observed clusters - 1), kept inside [df_S, df_S + df_T];
* the interval is psi -+ t(level, df) sqrt(V), the Student-t quantile at a non-integer df by ``scipy.stats.t.ppf``.

THE CLASS. The DECIDING interval is the WIDER of the two schemes'. ``POSITIVE`` iff its lower end > 0, ``NEGATIVE`` iff
its upper end < 0, else ``SPANS-ZERO``; an end within :data:`EPS_PP` of 0 is AT-BOUNDARY and takes ``SPANS-ZERO`` (root
rule 8, exact fractions). A registered NET names two contrasts with the same arm coefficients: their sum and its sign
class, ``>= 0`` iff the sum is above :data:`EPS_PP`, else ``< 0`` (AT-BOUNDARY within it). The verdict line holds
each contrast's class and each net's sign, less the ``reported_contrasts`` (classified the same way, REPORTED). The
rule knows no bar and no head-to-head class: the PREREG reads the classes.

VALIDITY. Refused with a typed :class:`main.reads.rules.RuleInputError`, never a fallback: a counted unit with no
per-game rows, no ``hit_forfeit_limit`` per game, or NO OPPONENT TEAM per game (Kakuna's own draw log, ``team_draws``
of ``<cell>/<half>/peer_*_report.json``, one draw a game in game order); a deal (panel, unit seed, half, index) whose
two teams differ between runs. INCONCLUSIVE with its reasons: the group's assembled reasons, unplayed units, no ONE
known engine; a run-panel with fewer counted games than ``min_games_frac`` of ``games_per_panel`` (or more than it);
more than :data:`TIMEOUT_FRAC` of a run-panel's games ended at OUR turn cap (the lane's timeout, booked a loss). A unit
is dropped only whole and only by ``main.anchors.results.unit_counts``: no game is dropped for its outcome. The lane
deals teams by a seeded uniform draw WITH replacement, so a panel is NOT balanced by team: the design block reports
each panel's observed teams and uses.

REPORTED, never deciding: both schemes' parts and intervals; per-panel psi; per-run, per-panel rates; each net's
analytic interval; two bootstraps on ONE shared index set (``bootstrap.draws`` draws at ``bootstrap.seed``, our teams
resampled within each panel, percentile at ``level``): ``literal`` (the draft PREREG's: seeds n-of-n within an arm,
ANTI-CONSERVATIVE at 2 seeds an arm) and ``rao_wu`` (n_a - 1 seed draws an arm, rescaled).
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "gen3_read_b1_kakuna_v1"
KEYS = ("group", "contrasts", "nets", "level", "games_per_panel", "min_games_frac", "bootstrap")
OPTIONAL = ("team_aliases", "reported_contrasts")
#: The two team-dependence schemes; the deciding interval is the wider one's.
SCHEMES = ("our_team", "dyadic")
#: A deciding bound (or a net) this close to 0, in pp, is AT-BOUNDARY (root rule 8).
EPS_PP = "0.005"
#: More than this share of a run-panel's games at our turn cap is INCONCLUSIVE (root CLAUDE.md: a timeout is never a
#: semantic outcome).
TIMEOUT_FRAC = Fraction(1, 4)
#: Kakuna's draw log of each challenge half (``main.anchors``: the acceptor's report in the half we challenge).
HALF_PEER_REPORT = {"ours_challenge": "peer_acceptor_report.json", "peer_challenge": "peer_challenger_report.json"}
POSITIVE, NEGATIVE, SPANS, INCONCLUSIVE, CLASSIFIED = "POSITIVE", "NEGATIVE", "SPANS-ZERO", "INCONCLUSIVE", "CLASSIFIED"
NET_GE, NET_LT = ">= 0", "< 0"
RULE_DOCS = {"b1_kakuna": "BOUNDARY-1's Kakuna meter: registered contrasts of arm means on Kakuna panels, an analytic "
                          "two-way interval (pooled within-arm seed variance; team clusters across runs, our_team and "
                          "dyadic, the wider decides); POSITIVE / NEGATIVE / SPANS-ZERO per contrast, a sign per net"}

#: One run-panel's deals: (our team, their team, games, wins), one row per deal or per (our team, their team) cell.
Cell = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _err(msg: str) -> Exception:
    from main.reads.rules import RuleInputError

    return RuleInputError(msg)


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ------------------------------------------------------------------------------------------------ the arithmetic
def run_weights(arms: Mapping[str, Sequence[str]], coef: Mapping[str, float]) -> Dict[str, float]:
    """w_r = c_a / n_a for every run of every arm the contrast names."""
    return {r: float(coef[a]) / len(arms[a]) for a in coef for r in arms[a]}


def rate(cell: Cell) -> float:
    n = float(cell[2].sum())
    if n <= 0:
        raise _err("a run-panel with no game has no rate")
    return float(cell[3].sum()) / n


def run_scores(data: Mapping[str, Mapping[str, Cell]], runs: Sequence[str], pw: Mapping[str, float]) -> Dict[str, float]:
    """S_r = sum_p pw_p R_{r,p}."""
    return {r: math.fsum(pw[p] * rate(data[r][p]) for p in pw) for r in runs}


def linear_terms(data: Mapping[str, Mapping[str, Cell]], w_run: Mapping[str, float],
                 pw: Mapping[str, float]) -> Dict[str, Dict[str, np.ndarray]]:
    """``{panel: {run: psi}}``, psi per deal row = w_r pw_p (wins - R_{r,p} games) / n_{r,p} (the addendum's
    ``influence``, weighted)."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for p in pw:
        out[p] = {}
        for r, w in w_run.items():
            _a, _b, n, wins = data[r][p]
            out[p][r] = w * pw[p] * (wins - rate(data[r][p]) * n) / float(n.sum())
    return out


def q_our_team(psi: np.ndarray, a: np.ndarray, b: Optional[np.ndarray] = None) -> Tuple[float, int]:
    """``(variance, G)``: the cluster sandwich on our team with G / (G - 1), G the teams present (``v_cluster``)."""
    ids, inv = np.unique(a, return_inverse=True)
    g = int(ids.size)
    if g < 2:
        raise _err("a stratum with one team cluster has no cluster variance")
    sums = np.bincount(inv, weights=psi, minlength=g)
    return float((sums ** 2).sum() * g / (g - 1)), g


def q_dyadic(psi: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
    """``(variance, teams)``: sum of psi_g psi_h over every pair of rows sharing a team in ANY role, times
    teams / (teams - 1) (``v_dyadic``). Computed without the pair matrix: sum_t T_t^2 counts a pair once per shared
    team, so the pairs on the same two distinct teams, counted twice, are taken off once."""
    ids, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    k = int(ids.size)
    if k < 2:
        raise _err("a stratum with one team has no dyadic variance")
    ia, ib = inv[:len(a)], inv[len(a):]
    off = np.where(ia == ib, 0.0, psi)                       # a mirror row touches ONE team
    t = np.bincount(ia, weights=psi, minlength=k) + np.bincount(ib, weights=off, minlength=k)
    pair = np.unique(np.minimum(ia, ib) * k + np.maximum(ia, ib), return_inverse=True)[1]
    both = np.bincount(pair, weights=off)
    return float(((t ** 2).sum() - (both ** 2).sum()) * k / (k - 1)), k


_Q = {"our_team": q_our_team, "dyadic": q_dyadic}


def team_variance(psi: Mapping[str, Mapping[str, np.ndarray]], data: Mapping[str, Mapping[str, Cell]],
                  scheme: str) -> Dict[str, Any]:
    """V_T (clusters across runs), V_TS (clusters split by run) and df_T of one scheme; panels are strata."""
    q = _Q[scheme]
    v_t = v_ts = 0.0
    clusters: Dict[str, int] = {}
    for p, by_run in psi.items():
        runs = list(by_run)
        a = np.concatenate([data[r][p][0] for r in runs])
        b = np.concatenate([data[r][p][1] for r in runs])
        v, g = q(np.concatenate([by_run[r] for r in runs]), a, b)
        v_t += v
        clusters[p] = g
        v_ts += math.fsum(q(by_run[r], data[r][p][0], data[r][p][1])[0] for r in runs)
    return {"v_t": v_t, "v_ts": v_ts, "df_t": sum(g - 1 for g in clusters.values()), "clusters": clusters}


def seed_variance(scores: Mapping[str, float], arms: Mapping[str, Sequence[str]],
                  w_run: Mapping[str, float]) -> Dict[str, Any]:
    """V_S = s2 sum_r w_r^2, s2 the pooled within-arm sample variance of S_r on df_S = sum_a (n_a - 1)."""
    df_s = sum(len(v) - 1 for v in arms.values())
    if df_s < 1:
        raise _err("no arm of the contrast holds two runs: the seed variance has no degree of freedom")
    ss = 0.0
    for labs in arms.values():
        mean = math.fsum(scores[r] for r in labs) / len(labs)
        ss += math.fsum((scores[r] - mean) ** 2 for r in labs)
    s2 = ss / df_s
    return {"s2": s2, "df_s": df_s, "v_s": s2 * math.fsum(w * w for w in w_run.values())}


def compose(v_s: float, v_ts: float, v_t: float, df_s: int, df_t: int) -> Dict[str, float]:
    """V = max(V_S, V_TS) + max(0, V_T - V_TS) and its Satterthwaite df, kept inside [df_S, df_S + df_T]."""
    a, b = max(v_s, v_ts), max(0.0, v_t - v_ts)
    v = a + b
    df = v * v / (a * a / df_s + b * b / df_t) if v > 0 and df_t > 0 else float(df_s)
    return {"a": a, "b": b, "v": v, "df": min(max(df, float(df_s)), float(df_s + df_t))}


def t_quantile(level: float, df: float) -> float:
    """The two-sided Student-t quantile at ``level`` on a (non-integer) ``df`` (``scipy.stats.t.ppf``)."""
    from scipy.stats import t as student_t

    return float(student_t.ppf(0.5 + level / 2.0, df))


def contrast(data: Mapping[str, Mapping[str, Cell]], arms: Mapping[str, Sequence[str]], coef: Mapping[str, float],
             pw: Mapping[str, float], level: float) -> Dict[str, Any]:
    """One contrast (arm coefficients ``coef``, panel weights ``pw``) on ``data[run][panel]``: psi, both schemes'
    variance parts and intervals, and the deciding (wider) one. Proportions, not pp."""
    named = {a: list(arms[a]) for a in coef}
    w_run = run_weights(named, coef)
    scores = run_scores(data, list(w_run), pw)
    psi_hat = math.fsum(w_run[r] * scores[r] for r in w_run)
    sv = seed_variance(scores, named, w_run)
    psi = linear_terms(data, w_run, pw)
    schemes: Dict[str, Dict[str, Any]] = {}
    for s in SCHEMES:
        tv = team_variance(psi, data, s)
        c = compose(sv["v_s"], tv["v_ts"], tv["v_t"], sv["df_s"], tv["df_t"])
        half = t_quantile(level, c["df"]) * math.sqrt(c["v"])
        schemes[s] = {"v_s": sv["v_s"], "v_t": tv["v_t"], "v_ts": tv["v_ts"], "v": c["v"], "df": c["df"],
                      "df_t": tv["df_t"], "clusters": tv["clusters"], "half": half, "lo": psi_hat - half,
                      "hi": psi_hat + half}
    wide = wider(schemes)
    return {"psi": psi_hat, "scores": scores, "run_weights": w_run, "s2": sv["s2"], "df_s": sv["df_s"],
            "schemes": schemes, "deciding": wide, "lo": schemes[wide]["lo"], "hi": schemes[wide]["hi"]}


def wider(schemes: Mapping[str, Mapping[str, Any]]) -> str:
    """The scheme with the larger half-width (both intervals sit on the same psi); a tie takes the first."""
    return max(SCHEMES, key=lambda s: schemes[s]["half"])


def classify(lo_pp: float, hi_pp: float) -> Dict[str, Any]:
    """The class of a deciding interval in pp, by exact fractions: an end within :data:`EPS_PP` of 0 crosses nothing."""
    from main.reads.rules_search_headroom import cmp

    pos, near_lo = cmp(lo_pp, 0, EPS_PP, above=True)
    neg, near_hi = cmp(hi_pp, 0, EPS_PP, above=False)
    return {"class": POSITIVE if pos else NEGATIVE if neg else SPANS, "at_boundary": bool(near_lo or near_hi)}


def net_sign(sum_pp: float) -> Dict[str, Any]:
    """``>= 0`` iff the net is above :data:`EPS_PP`; within it AT-BOUNDARY and ``< 0``."""
    from main.reads.rules_search_headroom import cmp

    up, near = cmp(sum_pp, 0, EPS_PP, above=True)
    return {"sign": NET_GE if up else NET_LT, "at_boundary": bool(near)}


# ------------------------------------------------------------------------------------------------ the bootstraps
def bootstrap_rates(data: Mapping[str, Mapping[str, Cell]], arms: Mapping[str, Sequence[str]],
                    panels: Sequence[str], draws: int, seed: int) -> Dict[str, Any]:
    """ONE index set for every contrast: per panel (in ``panels`` order) a resample of the teams WE pilot, shared by
    every run, giving each run's resampled rate; then per arm (sorted) the seed picks, n_a of n_a (``literal``) and
    n_a - 1 of n_a (``rao_wu``)."""
    rng = np.random.default_rng(seed)
    runs = [r for a in sorted(arms) for r in arms[a]]
    rates: Dict[str, np.ndarray] = {}
    for p in panels:
        ids = np.unique(np.concatenate([data[r][p][0] for r in runs]))
        pos = {int(t): i for i, t in enumerate(ids)}
        w = np.zeros((len(runs), len(ids)))
        n = np.zeros((len(runs), len(ids)))
        for i, r in enumerate(runs):
            a, _b, games, wins = data[r][p]
            idx = np.array([pos[int(t)] for t in a], dtype=int)
            np.add.at(w[i], idx, wins)
            np.add.at(n[i], idx, games)
        pick = rng.integers(0, len(ids), size=(draws, len(ids)))
        counts = np.stack([np.bincount(row, minlength=len(ids)) for row in pick]).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            rates[p] = (counts @ w.T) / (counts @ n.T)                                  # draws x runs
    col = {r: i for i, r in enumerate(runs)}
    picks = {a: {"literal": rng.integers(0, len(arms[a]), size=(draws, len(arms[a]))),
                 "rao_wu": rng.integers(0, len(arms[a]), size=(draws, max(len(arms[a]) - 1, 1)))}
             for a in sorted(arms)}
    return {"rates": rates, "col": col, "picks": picks, "draws": draws, "seed": seed}


def bootstrap_interval(boot: Mapping[str, Any], arms: Mapping[str, Sequence[str]], coef: Mapping[str, float],
                       pw: Mapping[str, float], level: float) -> Dict[str, Any]:
    """Both bootstraps' percentile interval of one contrast on the shared index set. An arm of ONE run has no seed
    draw: its run stands in both."""
    out: Dict[str, Any] = {}
    for kind in ("literal", "rao_wu"):
        psi = np.zeros(boot["draws"])
        for a, c in coef.items():
            cols = np.array([boot["col"][r] for r in arms[a]])
            s = sum(pw[p] * boot["rates"][p][:, cols] for p in pw)                      # draws x n_a
            pick = boot["picks"][a][kind] if len(cols) > 1 else np.zeros((boot["draws"], 1), dtype=int)
            psi += float(c) * np.take_along_axis(s, pick, axis=1).mean(axis=1)
        ok = psi[np.isfinite(psi)]
        if ok.size < 2:
            out[kind] = {"computed": False, "why": "fewer than two finite draws"}
            continue
        lo, hi = np.percentile(ok, [50.0 * (1 - level), 50.0 * (1 + level)])
        out[kind] = {"computed": True, "used_draws": int(ok.size), "lo": float(lo), "hi": float(hi),
                     "sd": float(ok.std(ddof=1))}
    return out


# ------------------------------------------------------------------------------------------------ check (spec load)
def _arms(group: Any, models: Any) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for lab in group.models:
        out.setdefault(models[lab].arm, []).append(lab)
    return out


def _contrast_check(name: Any, c: Any, arms: Mapping[str, Sequence[str]], teamsets: Sequence[str]) -> List[str]:
    w = f"contrasts.{name}"
    if not (isinstance(name, str) and name):
        return ["contrasts: a name is a non-empty string"]
    if not (isinstance(c, dict) and set(c) == {"arms", "panels"}):
        return [f"{w}: exactly {{arms, panels}}"]
    probs: List[str] = []
    co = c["arms"]
    if not (isinstance(co, dict) and co and all(isinstance(a, str) for a in co)
            and all(_num(v) and v != 0 for v in co.values())):
        probs.append(f"{w}.arms: {{arm: a non-zero coefficient}}")
    else:
        missing = sorted(a for a in co if a not in arms)
        if missing:
            probs.append(f"{w}.arms: {missing} are not arms of the group's models (it holds {sorted(arms)})")
        elif sum(Fraction(str(v)) for v in co.values()) != 0:
            probs.append(f"{w}.arms: the coefficients sum to {float(sum(Fraction(str(v)) for v in co.values()))}, "
                         "not 0 (a contrast)")
        elif sum(len(arms[a]) - 1 for a in co) < 1:
            probs.append(f"{w}.arms: no arm holds two runs, so the seed variance has no degree of freedom")
    ps = c["panels"]
    if not (isinstance(ps, list) and ps and all(isinstance(x, str) for x in ps) and len(set(ps)) == len(ps)):
        probs.append(f"{w}.panels: a list of >= 1 distinct team sets")
    elif [x for x in ps if x not in teamsets]:
        probs.append(f"{w}.panels: {[x for x in ps if x not in teamsets]} are not team sets of the group")
    return probs


def check(p: Mapping[str, Any], *, models: Any, anchor: Any, s: Any) -> List[str]:
    from main.reads.rules import _check_keys
    from main.x35.kakuna import UNIT_GAMES

    probs = _check_keys(p, KEYS, KEYS + OPTIONAL)
    g = anchor.get(p.get("group")) if isinstance(p.get("group"), str) else None
    if g is None:
        return probs + [f"group: {p.get('group')!r} is not an anchor group"]
    if g.per_game:
        probs.append(f"group {g.name!r} is per_game (the meter plays whole {UNIT_GAMES}-game units)")
    if g.imports is not None:
        probs.append(f"group {g.name!r} imports cells (every game is played at the read's pin)")
    loaded = [lab for lab in g.models if models[lab].load]
    if loaded:
        probs.append(f"group {g.name!r}: {loaded} carry load options (the meter plays greedy finals)")
    arms = _arms(g, models)
    for a, labs in sorted(arms.items()):
        seeds = [int(models[lab].seed) for lab in labs]
        if len(set(seeds)) != len(seeds):
            probs.append(f"arm {a!r}: two of its runs share a seed")
    if not (_num(p.get("level")) and 0.5 <= p["level"] < 1):
        probs.append("level: the two-sided confidence level, a number in [0.5, 1)")
    gp = p.get("games_per_panel")
    if not (isinstance(gp, int) and not isinstance(gp, bool) and gp > 0):
        probs.append("games_per_panel: an int > 0 (the registered games of ONE run on ONE panel)")
    elif gp != len(g.unit_seeds) * UNIT_GAMES:
        probs.append(f"games_per_panel: {gp} is not the group's {len(g.unit_seeds)} unit seeds x {UNIT_GAMES} games")
    if not (_num(p.get("min_games_frac")) and 0 < p["min_games_frac"] <= 1):
        probs.append("min_games_frac: a number in (0, 1] (a run-panel below it is INCONCLUSIVE)")
    b = p.get("bootstrap")
    if not (isinstance(b, dict) and set(b) == {"draws", "seed"}
            and all(isinstance(b[k], int) and not isinstance(b[k], bool) for k in b) and b["draws"] >= 100
            and b["seed"] >= 0):
        probs.append("bootstrap: {draws: an int >= 100, seed: an int >= 0} (the REPORTED bootstraps)")
    cs = p.get("contrasts")
    if not (isinstance(cs, dict) and cs):
        return probs + ["contrasts: {name: {arms: {arm: coefficient}, panels: [team set, ...]}}, at least one"]
    for name, c in cs.items():
        probs += _contrast_check(name, c, arms, g.teamsets)
    read = {x for c in cs.values() if isinstance(c, dict) and isinstance(c.get("panels"), list)
            for x in c["panels"] if isinstance(x, str)}
    idle = [ts for ts in g.teamsets if ts not in read]
    if idle:
        probs.append(f"group {g.name!r}: team set(s) {idle} are read by no contrast")
    nets = p.get("nets")
    if not isinstance(nets, dict):
        probs.append("nets: {name: [contrast, contrast]} (may be empty)")
    else:
        for name, pair in nets.items():
            if not (isinstance(name, str) and name and name not in cs and isinstance(pair, list) and len(pair) == 2
                    and all(isinstance(x, str) and x in cs for x in pair) and pair[0] != pair[1]):
                probs.append(f"nets.{name}: two distinct registered contrasts, under a name no contrast holds")
            elif all(isinstance(cs[x], dict) for x in pair) and cs[pair[0]].get("arms") != cs[pair[1]].get("arms"):
                probs.append(f"nets.{name}: its two contrasts differ in arm coefficients (a net sums ONE contrast "
                             "over two panel lists)")
    rc = p.get("reported_contrasts", [])
    if not (isinstance(rc, list) and all(isinstance(x, str) and x in cs for x in rc) and len(set(rc)) == len(rc)
            and len(rc) < len(cs)):
        probs.append("reported_contrasts: a list of distinct registered contrasts, not all of them (a contrast named "
                     "here is classified but REPORTED: it stays out of the verdict line)")
    al = p.get("team_aliases", {})
    if not (isinstance(al, dict) and all(
            ts in g.teamsets and isinstance(m, dict) and m
            and all(isinstance(k, str) and isinstance(v, str) and k and v and k != v and v not in m for k, v in m.items())
            for ts, m in al.items())):
        probs.append("team_aliases: {team set of the group: {alias team id: canonical team id}}, ids without their "
                     "extension, no canonical id itself an alias")
    return probs


# ------------------------------------------------------------------------------------------------ inputs
def team_id(name: Any, aliases: Mapping[str, str]) -> str:
    """A team's id: its file name without the extension (the addendum's), through the panel's registered aliases."""
    if not (isinstance(name, str) and name):
        raise _err(f"a game with no team name ({name!r})")
    stem = name.rsplit(".", 1)[0]
    return aliases.get(stem, stem)


def their_draws(cell_dir: Path, half: str, n_games: int, unit: str) -> List[str]:
    """Kakuna's team of each game of one half, in game order: its own draw log. Refused when it is not there or does
    not hold exactly one draw a game (the dyadic clusters need the opponent's team: never a fallback)."""
    if half not in HALF_PEER_REPORT:
        raise _err(f"unit {unit}: half {half!r} has no known peer report (the opponent's team per game is unknown)")
    path = cell_dir / half / HALF_PEER_REPORT[half]
    try:
        draws = json.loads(path.read_text()).get("team_draws")
    except (OSError, ValueError, AttributeError):
        raise _err(f"unit {unit}: no readable peer report {path} (the opponent's team per game is unknown; the "
                   "dyadic clusters cannot be formed)") from None
    if not (isinstance(draws, list) and len(draws) == n_games and all(isinstance(d, str) and d for d in draws)):
        raise _err(f"unit {unit}: {path} holds {len(draws) if isinstance(draws, list) else 'no'} team draw(s) for "
                   f"{n_games} game(s) of {half} (one draw a game, in game order)")
    return list(draws)


def unit_deals(u: Mapping[str, Any], aliases: Mapping[str, str]) -> List[Tuple[str, int, str, str, bool, bool, bool]]:
    """``[(half, index, our team, their team, win, tie, at our turn cap), ...]`` of one COUNTED unit."""
    unit = str(u.get("unit"))
    games = u.get("games")
    if not games or len(games) != int(u.get("n") or 0):
        raise _err(f"counted unit {unit}: {len(games or ())} per-game row(s), n {u.get('n')}")
    if not u.get("cell_dir"):
        raise _err(f"counted unit {unit} names no cell dir (the opponent's team per game is unknown)")
    cell = Path(str(u["cell_dir"]))
    try:
        rows = [json.loads(ln) for ln in (cell / "games.jsonl").read_text().splitlines() if ln.strip()]
    except (OSError, ValueError):
        raise _err(f"counted unit {unit}: no readable {cell / 'games.jsonl'}") from None
    cap = {(str(r.get("half")), r.get("index")): r for r in rows}
    by_half: Dict[str, List[Sequence[Any]]] = {}
    for g in games:
        by_half.setdefault(str(g[0]), []).append(g)
    out: List[Tuple[str, int, str, str, bool, bool, bool]] = []
    for half, gs in by_half.items():
        gs = sorted(gs, key=lambda g: int(g[1]))
        draws = their_draws(cell, half, len(gs), unit)
        for g, their in zip(gs, draws):
            row = cap.get((half, g[1]))
            if row is None or row.get("our_team") != g[2] or row.get("result") != g[3]:
                raise _err(f"counted unit {unit}: game {half}/{g[1]} of games.jsonl is not the assembled row")
            if not isinstance(row.get("hit_forfeit_limit"), bool):
                raise _err(f"counted unit {unit}: game {half}/{g[1]} carries no hit_forfeit_limit (the timeout guard "
                           "cannot be read)")
            if g[3] not in ("win", "loss", "tie"):
                raise _err(f"counted unit {unit}: game {half}/{g[1]} result {g[3]!r}")
            out.append((half, int(g[1]), team_id(g[2], aliases), team_id(their, aliases), g[3] == "win",
                        g[3] == "tie", row["hit_forfeit_limit"]))
    return out


def inputs(cells: Mapping[str, Any], p: Mapping[str, Any], spec: Any) -> Dict[str, Any]:
    grp = spec.anchor[p["group"]]
    labels = list(grp.models)
    arms = _arms(grp, spec.models)
    base: Dict[str, Any] = {"schema": SCHEMA, "group": p["group"], "arms": arms,
                            "seeds": {lab: int(spec.models[lab].seed) for lab in labels}}
    g = (cells.get("anchor") or {}).get(p["group"])
    if g is None:
        return dict(base, missing=True, inconclusive=[f"no assembled anchor group {p['group']!r}"], unplayed=0,
                    engine=None, regime_id=None, teamsets=[], teams={}, cells={}, design={})
    teamsets = list(g.get("teamsets") or [])
    aliases = p.get("team_aliases") or {}
    deals: Dict[Tuple[str, int, str, int], Tuple[str, str]] = {}
    agg: Dict[str, Dict[str, Dict[Tuple[str, str], List[int]]]] = {lab: {ts: {} for ts in teamsets} for lab in labels}
    for u in g.get("units") or ():
        if u.get("label") not in agg or not u.get("counted"):
            continue
        ts = u["panel"]
        for half, index, our, their, win, tie, cap in unit_deals(u, aliases.get(ts) or {}):
            key = (ts, int(u["unit_seed"]), half, index)
            if deals.setdefault(key, (our, their)) != (our, their):
                raise _err(f"deal {key}: {u['label']} played {(our, their)}, another run {deals[key]} (one deal is "
                           "ONE pair of teams in every run)")
            c = agg[u["label"]][ts].setdefault((our, their), [0, 0, 0, 0])
            c[0] += 1
            c[1] += int(win)
            c[2] += int(tie)
            c[3] += int(cap)
    teams = {ts: sorted({t for (pt, *_), pair in deals.items() if pt == ts for t in pair}) for ts in teamsets}
    out_cells: Dict[str, Dict[str, Any]] = {}
    for lab in labels:
        out_cells[lab] = {}
        for ts in teamsets:
            pos = {t: i for i, t in enumerate(teams[ts])}
            rows = sorted([pos[o], pos[t], *c] for (o, t), c in agg[lab][ts].items())
            asm = ((g.get("cells") or {}).get(lab) or {}).get(ts) or {}
            n, wins = sum(r[2] for r in rows), sum(r[3] for r in rows)
            if (n, wins) != (int(asm.get("n") or 0), int(asm.get("wins") or 0)):
                raise _err(f"{lab} on {ts}: the per-game rows hold {wins} / {n}, the assembled cell "
                           f"{asm.get('wins')} / {asm.get('n')}")
            out_cells[lab][ts] = {"n": n, "wins": wins, "ties": sum(r[4] for r in rows),
                                  "caps": sum(r[5] for r in rows), "units": asm.get("units"),
                                  "planned": asm.get("planned"), "skipped": len(asm.get("skipped") or ()),
                                  "rows": rows}
    design = {}
    for ts in teamsets:
        mine = [k for k in deals if k[0] == ts]
        ours: Dict[str, int] = {}
        theirs: Dict[str, int] = {}
        for k in mine:
            ours[deals[k][0]] = ours.get(deals[k][0], 0) + 1
            theirs[deals[k][1]] = theirs.get(deals[k][1], 0) + 1
        design[ts] = {"deals": len(mine), "teams": len(teams[ts]), "teams_ours": len(ours), "teams_theirs": len(theirs),
                      "our_use_min": min(ours.values()) if ours else 0, "our_use_max": max(ours.values()) if ours else 0,
                      "mirror_deals": sum(deals[k][0] == deals[k][1] for k in mine)}
    return dict(base, missing=False, inconclusive=list(g.get("inconclusive") or []), unplayed=int(g.get("unplayed") or 0),
                engine=g.get("engine"), regime_id=g.get("regime_id"), teamsets=teamsets, teams=teams, cells=out_cells,
                design=design)


# ------------------------------------------------------------------------------------------------ decide
def _reasons(doc: Mapping[str, Any], p: Mapping[str, Any]) -> List[str]:
    why = [f"{doc['group']}: {w}" for w in doc.get("inconclusive") or ()]
    if doc.get("unplayed"):
        why.append(f"{doc['group']}: {doc['unplayed']} unit(s) unplayed")
    if doc.get("missing"):
        return why
    if not (isinstance(doc.get("engine"), str) and doc["engine"]):
        why.append(f"{doc['group']}: its counted units are not on ONE known engine")
    full = int(p["games_per_panel"])
    floor = Fraction(str(p["min_games_frac"])) * full
    panels = sorted({ts for c in p["contrasts"].values() for ts in c["panels"]})
    for lab in sorted(doc["cells"]):
        for ts in panels:
            c = doc["cells"][lab].get(ts)
            n = int(c["n"]) if c else 0
            if n < floor or n > full:
                why.append(f"{lab} on {ts}: {n} counted game(s), the registered {full} (at least {float(floor):g})")
            elif Fraction(int(c["caps"]), n) > TIMEOUT_FRAC:
                why.append(f"{lab} on {ts}: {c['caps']} of {n} games ended at our turn cap (> {TIMEOUT_FRAC})")
    return why


def _data(doc: Mapping[str, Any]) -> Dict[str, Dict[str, Cell]]:
    out: Dict[str, Dict[str, Cell]] = {}
    for lab, by in doc["cells"].items():
        out[lab] = {}
        for ts, c in by.items():
            r = np.array(c["rows"], dtype=float).reshape(-1, 6)
            out[lab][ts] = (r[:, 0].astype(int), r[:, 1].astype(int), r[:, 2], r[:, 3])
    return out


def _pp(x: float) -> float:
    return 100.0 * x


def _block(st: Mapping[str, Any], boot: Mapping[str, Any]) -> Dict[str, Any]:
    """One contrast's (or net's) statistic in pp (variances in pp^2)."""
    return {"psi_pp": _pp(st["psi"]), "lo_pp": _pp(st["lo"]), "hi_pp": _pp(st["hi"]), "deciding_scheme": st["deciding"],
            "s2_pp2": 1e4 * st["s2"], "df_s": st["df_s"], "run_scores": dict(st["scores"]),
            "run_weights": dict(st["run_weights"]),
            "schemes": {s: {"v_s_pp2": 1e4 * x["v_s"], "v_t_pp2": 1e4 * x["v_t"], "v_ts_pp2": 1e4 * x["v_ts"],
                            "v_pp2": 1e4 * x["v"], "se_pp": _pp(math.sqrt(x["v"])), "df": x["df"], "df_t": x["df_t"],
                            "clusters": dict(x["clusters"]), "lo_pp": _pp(x["lo"]), "hi_pp": _pp(x["hi"])}
                        for s, x in st["schemes"].items()},
            "bootstrap": {k: (dict(b, lo_pp=_pp(b.pop("lo")), hi_pp=_pp(b.pop("hi")), sd_pp=_pp(b.pop("sd")))
                              if b.get("computed") else b) for k, b in boot.items()}}


def decide(doc: Mapping[str, Any], p: Mapping[str, Any]) -> Dict[str, Any]:
    level = float(p["level"])
    per_run = {lab: {ts: {k: c[k] for k in ("n", "wins", "ties", "caps", "units", "planned", "skipped")}
                     | {"rate": (c["wins"] / c["n"]) if c["n"] else None} for ts, c in by.items()}
               for lab, by in (doc.get("cells") or {}).items()}
    base: Dict[str, Any] = {"schema": SCHEMA, "group": doc["group"], "level": level, "arms": doc["arms"],
                            "seeds": doc["seeds"], "engine": doc.get("engine"), "per_run": per_run,
                            "design": doc.get("design") or {}}
    why = _reasons(doc, p)
    if why:
        return dict(base, contrasts={}, nets={}, decision={
            "verdict": INCONCLUSIVE, "class": INCONCLUSIVE, "contrasts": {}, "nets": {}, "reasons": why,
            "inconclusive_kind": "inputs"})
    data, arms = _data(doc), doc["arms"]
    boot = bootstrap_rates(data, arms, sorted(doc["teamsets"]), int(p["bootstrap"]["draws"]), int(p["bootstrap"]["seed"]))
    contrasts: Dict[str, Any] = {}
    pws: Dict[str, Dict[str, float]] = {}
    for name, c in p["contrasts"].items():
        pw = pws[name] = {ts: 1.0 / len(c["panels"]) for ts in c["panels"]}
        st = contrast(data, arms, c["arms"], pw, level)
        blk = _block(st, bootstrap_interval(boot, arms, c["arms"], pw, level))
        blk.update(arms=dict(c["arms"]), panels=list(c["panels"]), **classify(blk["lo_pp"], blk["hi_pp"]),
                   per_panel_pp={ts: _pp(math.fsum(st["run_weights"][r] * rate(data[r][ts]) for r in st["run_weights"]))
                                 for ts in c["panels"]})
        contrasts[name] = blk
    nets: Dict[str, Any] = {}
    for name, (x, y) in p["nets"].items():
        pw = dict(pws[x])
        for ts, w in pws[y].items():
            pw[ts] = pw.get(ts, 0.0) + w
        coef = p["contrasts"][x]["arms"]
        blk = _block(contrast(data, arms, coef, pw, level), bootstrap_interval(boot, arms, coef, pw, level))
        total = contrasts[x]["psi_pp"] + contrasts[y]["psi_pp"]
        nets[name] = dict(net_sign(total), pair=[x, y], sum_pp=total, reported_interval=blk)
    rep = set(p.get("reported_contrasts") or ())
    classes = {k: v["class"] for k, v in contrasts.items() if k not in rep}
    signs = {k: v["sign"] for k, v in nets.items()}
    verdict = "; ".join([f"{k} {v}" for k, v in classes.items()] + [f"{k} {v}" for k, v in signs.items()])
    return dict(base, contrasts=contrasts, nets=nets, bootstrap={"draws": boot["draws"], "seed": boot["seed"]},
                decision={"verdict": verdict, "class": CLASSIFIED, "contrasts": classes, "nets": signs,
                          "reported_contrasts": {k: v["class"] for k, v in contrasts.items() if k in rep},
                          "reasons": []})
