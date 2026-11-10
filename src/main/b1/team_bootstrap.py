"""BOUNDARY-1's TEAM-CLUSTER BOOTSTRAP: how far each factor's statistic moves when the panel's TEAMS are resampled.

    python -m main.b1.team_bootstrap --work <W>          # <W>/result/team_bootstrap.json (and stdout); exit 3 = no read

**REPORTED, never deciding** (PREREG §5.5, Amendment 1). The registered interval (:func:`main.b1.cross.factor_stat`)
treats the RUNS as the sample and the panel as fixed: it answers "would other seeds agree?". This tool answers the
other question, "would another 72 teams from the same population agree?", and writes nothing a class is read from:
``verdict.json`` is :mod:`main.b1.look`'s alone.

**What is resampled.** The count ledger banks no per-game row. Each row (one batch of one cell) banks a per-TEAM
tally (``main.h2h.play.score_games``): for team ``t``, the games the PLAYER piloted it and the player's wins there,
and the games the OPPONENT piloted it and the opponent's wins there. A mirrored cell's score is
``1/2 + (W - L) / 2G`` (a draw is half a point to each side and drops out of ``W - L``), and
``W - L = sum_t d_t`` with ``d_t`` = (player's wins piloting ``t``) - (opponent's wins piloting ``t``): every
decisive game is carried by the team its WINNER piloted, a draw by none. So with ``g_t`` the games the player
piloted ``t`` (``sum_t g_t = G``),

    score = 50 + 50 * sum_t d_t / sum_t g_t   (pp),

exactly, and :func:`cell_tallies` refuses a cell whose tallies do not reproduce the ledger cell's ``W``, ``L``,
``G`` and score. A resample draws the panel's teams with replacement (multiplicities ``m_t``, ONE draw shared by
every cell of the panel, because a team's quirk moves all of them together; the panels are drawn independently) and
recomputes every cell as ``50 + 50 * sum_t m_t d_t / sum_t m_t g_t``. The factor's statistic is linear in the cell
scores; its weights are read off :func:`main.b1.cross.factor_stat` itself (:func:`cell_weights`) and the tallies
must reproduce the registered estimate before a single draw is taken.

**What it reports**, per factor on the view the look reads it on (and the spread on every view): the team
bootstrap's SE and percentile interval; the registered SE, interval and class; and a COMBINED interval,
``estimate +- t(df) * sqrt(V_registered + V_team)`` at the registered df, with the class that interval would give.
The combined variance double-counts the game noise (both terms hold it), so it is an UPPER bound on "seeds and
teams both resampled", never a second verdict. For a factor under the net rule, the share of resamples whose net
sum stays above zero.

**What it is not.** One-way: a game is clustered by the winning pilot's team, never by the pair of teams it was
played on. A team id that sits on two panels is resampled on each independently (``shared_team_ids`` counts them).
The runs are held fixed, so this is no seed interval. A survivor look (two arms) is refused: not built.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from main.b1 import look as K
from main.b1 import cross as X

DRAWS = 10_000
SEED = 20261110
REPORTED = "REPORTED"
#: The tallies must reproduce a cell's score and a factor's estimate to this many pp (float rounding only).
TOL_PP = 1e-9

Tally = Dict[str, List[int]]  # team id -> [g_t, d_t]


class TeamBootstrapError(ValueError):
    """Rows whose team tallies are missing or do not reproduce the cell, or a look this tool does not read."""


def cell_tallies(rows: Sequence[Mapping[str, Any]], run_of: Mapping[str, str]) -> Dict[X.Cell, Tally]:
    """One panel's rows -> ``{(player run, opponent run): {team: [g_t, d_t]}}``, summed over the cell's batches."""
    out: Dict[X.Cell, Tally] = {}
    for r in rows:
        key = (run_of.get(str((r.get("player") or {}).get("sha256"))),
               run_of.get(str((r.get("opponent") or {}).get("sha256"))))
        if key[0] is None or key[1] is None:
            raise TeamBootstrapError(f"row {r.get('row_id')}: a checkpoint outside the look")
        teams = r.get("teams")
        if not isinstance(teams, Mapping) or not teams:
            raise TeamBootstrapError(f"row {r.get('row_id')}: no per-team tally")
        cell = out.setdefault((str(key[0]), str(key[1])), {})
        for tid, t in teams.items():
            (pg, pw), (og, ow) = t["p"], t["o"]
            if int(pg) != int(og):
                raise TeamBootstrapError(f"row {r.get('row_id')}: team {tid} was piloted {pg} times by the player "
                                         f"and {og} by the opponent (a mirrored batch gives each side every team)")
            acc = cell.setdefault(str(tid), [0, 0])
            acc[0] += int(pg)
            acc[1] += int(pw) - int(ow)
    return out


def tally_score(tally: Tally) -> float:
    """A cell's score in pp from its team tallies (module docstring)."""
    g = sum(v[0] for v in tally.values())
    if g <= 0:
        raise TeamBootstrapError("a cell with no game")
    return 50.0 + 50.0 * sum(v[1] for v in tally.values()) / g


def check_tallies(panel: str, tallies: Mapping[X.Cell, Tally], cells: Sequence[Any], run_of: Mapping[str, str],
                  design: X.Design) -> None:
    """Refuse unless the tallies hold exactly the design's cells and reproduce each ledger cell's W - L, G and
    score."""
    by = {(run_of.get(c.player), run_of.get(c.opponent)): c for c in cells}
    want = design.cells()
    if set(tallies) != set(want):
        raise TeamBootstrapError(f"{panel}: the rows hold {len(tallies)} cells, the design {len(want)}")
    for k in want:
        c, t = by.get(k), tallies[k]
        if c is None:
            raise TeamBootstrapError(f"{panel}: no ledger cell for {k[0]} vs {k[1]}")
        g, d = sum(v[0] for v in t.values()), sum(v[1] for v in t.values())
        if g != int(c.games) or d != int(c.w) - int(c.l):
            raise TeamBootstrapError(f"{panel}: {k[0]} vs {k[1]}: the team tallies sum to {g} games and W - L = {d}, "
                                     f"the cell holds {c.games} and {int(c.w) - int(c.l)}")
        if abs(tally_score(t) - 100.0 * float(c.score)) > TOL_PP:
            raise TeamBootstrapError(f"{panel}: {k[0]} vs {k[1]}: the team tallies give {tally_score(t)!r} pp, the "
                                     f"cell's score is {100.0 * float(c.score)!r} pp")


def cell_weights(design: X.Design, factor: str) -> Dict[X.Cell, float]:
    """``a_c`` with ``estimate = sum_c a_c (score_c - 50)``, read off :func:`main.b1.cross.factor_stat` itself: its
    estimate on all-50 scores (0) against its estimate with one cell at 51."""
    cells = design.cells()
    flat = {c: 50.0 for c in cells}
    base = X.factor_stat(design, flat, factor).estimate
    return {c: X.factor_stat(design, {**flat, c: 51.0}, factor).estimate - base for c in cells}


def resample(tallies: Mapping[X.Cell, Tally], weights: Mapping[str, Mapping[X.Cell, float]], draws: int,
             rng: np.random.Generator) -> Tuple[Dict[str, np.ndarray], int]:
    """One panel: ``({factor: its estimate on each resample of the panel's teams, pp}, the number of teams)``."""
    teams = sorted({t for tally in tallies.values() for t in tally})
    k = len(teams)
    if k < 2:
        raise TeamBootstrapError(f"a panel with {k} team(s) cannot be resampled")
    idx = {t: i for i, t in enumerate(teams)}
    m = rng.multinomial(k, np.full(k, 1.0 / k), size=draws).astype(np.float64)
    out = {f: np.zeros(draws) for f in weights}
    for cell in sorted(tallies):
        g, d = np.zeros(k), np.zeros(k)
        for t, (gt, dt) in tallies[cell].items():
            g[idx[t]], d[idx[t]] = gt, dt
        den = m @ g
        if not bool((den > 0).all()):
            raise TeamBootstrapError(f"cell {cell[0]} vs {cell[1]}: a resample of the {k} teams left it no game "
                                     f"(its games sit on too few teams to resample)")
        dev = 50.0 * (m @ d) / den
        for f, w in weights.items():
            if w[cell]:
                out[f] += w[cell] * dev
    return out, k


def _combined(stat: X.Stat, team_var: float) -> X.Stat:
    v = stat.v_hat + team_var
    se = math.sqrt(v)
    q = X.t_ppf(0.5 + stat.level / 2.0, stat.df)
    return dataclasses.replace(stat, v_hat=v, se=se, lo=stat.estimate - q * se, hi=stat.estimate + q * se)


def bootstrap(look_doc: Mapping[str, Any], cells_by_panel: Mapping[str, Sequence[Any]],
              rows_by_panel: Mapping[str, Sequence[Mapping[str, Any]]], *, draws: int = DRAWS, seed: int = SEED,
              row_reasons: Sequence[str] = ()) -> Dict[str, Any]:
    """The report (JSON-ready) of a played FULL look, or ``{"outcome": INCONCLUSIVE, ...}`` when the look itself
    reads INCONCLUSIVE (nothing is resampled then). Deterministic in ``(draws, seed)``."""
    if look_doc.get("survivor"):
        raise TeamBootstrapError("a survivor look (two arms) has no factor statistic: not built")
    if draws < 100:
        raise TeamBootstrapError(f"--draws {draws}: at least 100")
    look = str(look_doc["look"])
    verdict = K.read(look_doc, cells_by_panel, None, row_reasons)
    head = {"what": "team-cluster bootstrap of each factor's statistic", "status": REPORTED, "decides": None,
            "look": look, "pin": look_doc["pin"], "draws": draws, "seed": seed}
    if verdict.get("outcome") == K.INCONCLUSIVE:
        return {**head, "outcome": K.INCONCLUSIVE, "reasons": list(verdict["reasons"])}
    design = X.design_of({a: tuple(f["run"] for f in look_doc["runs"][a]) for a in X.ARMS})
    run_of = {f["sha256"]: f["run"] for a in X.ARMS for f in look_doc["runs"][a]}
    weights = {f: cell_weights(design, f) for f in X.FACTORS}
    rng = np.random.default_rng(seed)
    by_panel: Dict[str, Dict[X.Cell, float]] = {}
    star: Dict[str, Dict[str, np.ndarray]] = {}
    n_teams: Dict[str, int] = {}
    ids: Dict[str, set] = {}
    for panel in X.PANELS:  # one generator, the panels in their registered order: the same answer every time
        tallies = cell_tallies(rows_by_panel.get(panel, ()), run_of)
        check_tallies(panel, tallies, cells_by_panel.get(panel, ()), run_of, design)
        by_panel[panel] = {c: tally_score(t) for c, t in tallies.items()}
        star[panel], n_teams[panel] = resample(tallies, weights, draws, rng)
        ids[panel] = {t for tally in tallies.values() for t in tally}
    shared = {f"{p}&{q}": len(ids[p] & ids[q]) for i, p in enumerate(X.PANELS) for q in X.PANELS[i + 1:]
              if ids[p] & ids[q]}
    out: Dict[str, Any] = {**head, "n_per_arm": design.n, "teams": n_teams, "shared_team_ids": shared, "factors": {}}
    lo_q, hi_q = 50.0 * (1.0 - X.LEVEL), 100.0 - 50.0 * (1.0 - X.LEVEL)
    for f in X.FACTORS:
        read_on = X.HARM_VIEW[f] if look == "50M" else X.DECIDING[f]
        views: Dict[str, Any] = {}
        for view, pw in X.VIEWS.items():
            stat = X.factor_stat(design, X.blend(design, by_panel, view), f, view=view)
            point = math.fsum(w * math.fsum(weights[f][c] * (by_panel[p][c] - 50.0) for c in weights[f])
                              for p, w in pw.items())
            if abs(point - stat.estimate) > TOL_PP:
                raise TeamBootstrapError(f"{f} on {view}: the tallies give {point!r} pp, the registered statistic "
                                         f"{stat.estimate!r} pp")
            d = sum(w * star[p][f] for p, w in pw.items())
            tv = float(np.var(d, ddof=1))
            comb = _combined(stat, tv)
            views[view] = {
                "estimate_pp": stat.estimate,
                "registered": {"se_pp": stat.se, "df": stat.df, "interval_pp": [stat.lo, stat.hi],
                               "class": X.classify(look, stat).outcome},
                "team": {"se_pp": math.sqrt(tv), "bias_pp": float(np.mean(d)) - stat.estimate,
                         "percentile_interval_pp": [float(np.percentile(d, lo_q)), float(np.percentile(d, hi_q))]},
                "combined": {"se_pp": comb.se, "interval_pp": [comb.lo, comb.hi],
                             "class": X.classify(look, comb).outcome, "team_share_of_variance": tv / comb.v_hat}}
        v = views[read_on]
        entry: Dict[str, Any] = {
            "read_on": read_on, "outcome_registered": verdict["factors"][f]["outcome"],
            "class_before_net_rule": v["registered"]["class"], "class_with_team_variance": v["combined"]["class"],
            "class_moves": v["registered"]["class"] != v["combined"]["class"], **v, "views": views}
        if f in X.NET:
            cost, gain = X.NET[f]
            net = star[cost][f] + star[gain][f]
            entry["net"] = {"cost_view": cost, "gain_view": gain,
                            "net_pp": views[cost]["estimate_pp"] + views[gain]["estimate_pp"],
                            "team_se_pp": float(np.std(net, ddof=1)),
                            "share_of_resamples_above_zero": float(np.mean(net > X.EPS_PP))}
        out["factors"][f] = entry
    out["notes"] = [
        "REPORTED: no class is read from this file; the look's verdict is result/verdict.json",
        "the combined interval adds the team variance to the registered one at the registered df; both hold the "
        "game noise, so it is an upper bound on resampling seeds and teams together",
        "one-way clusters (the winning pilot's team); the panels are resampled independently; the runs are fixed"]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.b1.team_bootstrap", description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--work", required=True, help="the look's work directory (holds look.json)")
    ap.add_argument("--ledger", help="the ledger root (default: <models>/_ledger)")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    try:
        work = Path(a.work)
        doc = json.loads((work / "look.json").read_text())
        from main.h2h import panel as PN
        from utils.paths import main_models_dir

        models = main_models_dir()
        if a.ledger is None and models is None:
            raise TeamBootstrapError("no models/ archive here (set GEN3AI_MODELS_DIR, or pass --ledger)")
        root = Path(a.ledger) if a.ledger else Path(str(models)) / "_ledger"
        cells, rows = K.ledger_rows(root, doc)
        reasons = K.row_problems(rows, str(doc["pin"]), {p: PN.panel_team_set_id(p) for p in X.PANELS})
        rep = bootstrap(doc, cells, rows, draws=a.draws, seed=a.seed, row_reasons=reasons)
        (work / "result").mkdir(parents=True, exist_ok=True)
        (work / "result" / "team_bootstrap.json").write_text(json.dumps(rep, indent=1, default=str))
        print(json.dumps(rep, default=str))
        return 3 if rep.get("outcome") == K.INCONCLUSIVE else 0
    except (TeamBootstrapError, K.LookError, X.CrossInputError, OSError, KeyError) as e:
        print(f"REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
