"""BOUNDARY-1's REPORTED team-cluster bootstrap (``main.b1.team_bootstrap``): the tallies ARE the cell, the point
value IS the registered statistic, and a team effect the seed interval cannot see is what it measures."""
import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pytest

from main.b1 import cross as X
from main.b1 import look as K
from main.b1 import team_bootstrap as TB

PIN = "a" * 40
TEAMS = 10
#: An arm's strength in pp: a cell's score is 50 + (its player's arm) - (its opponent's arm). Width +4, breadth 0.
STRENGTH = {"C": 0.0, "W": 4.0, "Br": 0.0, "WB": 4.0}


def _sha(run: str) -> str:
    return hashlib.sha256(run.encode()).hexdigest()


def make_doc(n: int = 3, look: str = "100M", survivor: Optional[str] = None) -> Dict[str, Any]:
    """A planned look as ``main.b1.look.plan`` writes it, with no run folder behind it."""
    arms = K.look_arms(look, survivor)
    runs = {a: [{"run": f"{a}{k}", "sha256": _sha(f"{a}{k}"), "steps": 100_000_000,
                 "checkpoint": f"/models/{a}{k}/checkpoints/checkpoint_100000000_steps.zip"} for k in range(n)]
            for a in arms}
    design = (X.pair_design_of if survivor else X.design_of)({a: tuple(f["run"] for f in runs[a]) for a in arms})
    zip_of = {f["run"]: f["checkpoint"] for fs in runs.values() for f in fs}
    return {"look": look, "pin": PIN, "steps": {a: 100_000_000 for a in arms}, "eq_step": None,
            "replaced_seed": None, "survivor": survivor, "runs": runs,
            "cells": [[zip_of[p], zip_of[o]] for p, o in design.cells()],
            "names": K.names(look, PIN, survivor=survivor)}


def _arm(run: str) -> str:
    return run.rstrip("0123456789")


def even(panel: str, cell: Tuple[str, str], t: int, total: int) -> int:
    """The cell's W - L spread evenly over the panel's teams."""
    return total // TEAMS


def one_team(panel: str, cell: Tuple[str, str], t: int, total: int) -> int:
    """The cell's whole W - L carried by team 0: the same team in every cell of the panel."""
    return total if t == 0 else 0


def make_look(doc: Mapping[str, Any], spread: Callable[[str, Tuple[str, str], int, int], int] = even,
              strength: Mapping[str, float] = STRENGTH, draws_a_team: int = 2
              ) -> Tuple[Dict[str, List[Any]], Dict[str, List[Dict[str, Any]]]]:
    """``(cells, rows)`` of a played look, a panel: every cell's score is its arms' strengths; ``spread`` says which
    teams carry the cell's W - L. Each team is piloted ``G / TEAMS`` times by each side; ``draws_a_team`` drawn
    games sit on every team (they move no score)."""
    design = X.design_of({a: tuple(f["run"] for f in doc["runs"][a]) for a in X.ARMS})
    sha = {f["run"]: f["sha256"] for a in X.ARMS for f in doc["runs"][a]}
    cells: Dict[str, List[Any]] = {}
    rows: Dict[str, List[Dict[str, Any]]] = {}
    for panel in X.PANELS:
        games = 2 * X.PAIRS[panel]
        g_t = games // TEAMS
        cells[panel], rows[panel] = [], []
        for i, (p, o) in enumerate(design.cells()):
            total = round(games * (strength[_arm(p)] - strength[_arm(o)]) / 50.0)  # W - L
            teams: Dict[str, Dict[str, List[int]]] = {}
            w = l = 0
            for t in range(TEAMS):
                d = spread(panel, (p, o), t, total)
                assert d % 2 == 0 and abs(d) <= g_t - 2 * draws_a_team
                pw, ow = g_t // 2 + d // 2 - draws_a_team, g_t // 2 - d // 2 - draws_a_team
                teams[f"{panel}-t{t}"] = {"p": [g_t, pw], "o": [g_t, ow]}
                w, l = w + pw, l + ow
            assert w - l == total
            dr = games - w - l
            rows[panel].append({"row_id": f"{panel}-{i}", "player": {"sha256": sha[p]}, "opponent": {"sha256": sha[o]},
                                "teams": teams, "counts": {"w": w, "l": l, "d": dr, "aborted": 0}})
            cells[panel].append(SimpleNamespace(player=sha[p], opponent=sha[o], w=w, l=l, d=dr, games=games,
                                                n_pairs=X.PAIRS[panel], verdict="OK", reasons=(),
                                                score=(w + 0.5 * dr) / games))
    return cells, rows


# ------------------------------------------------------------------------------------------- the tallies
def test_the_team_tallies_reproduce_a_scored_batch_exactly_drawn_games_included():
    """``score_games``' own tallies: W - L is the sum over teams of (player's wins with it - opponent's wins with
    it), whatever the draws, so the cell's score needs no per-game row."""
    from agents.training.trace_result import DRAW, LOSS, WIN
    from main.h2h import play as PL

    rng = random.Random(7)
    packed = [f"packed team {i}" for i in range(6)]
    games = []
    for k in range(40):
        a, b = rng.sample(range(6), 2)
        for swapped, teams in ((False, [a, b]), (True, [b, a])):
            games.append({"game": len(games), "result": rng.choice([WIN, WIN, LOSS, DRAW]), "swapped": swapped,
                          "teams": teams, "near_ties": 0, "near_ties_wide": 0})
    s = PL.score_games(games, packed, 40)
    assert s.d > 0 and s.w != s.l
    row = {"row_id": "r", "player": {"sha256": "P"}, "opponent": {"sha256": "O"}, "teams": s.teams}
    tally = TB.cell_tallies([row, row], {"P": "p", "O": "o"})[("p", "o")]  # two batches of one cell add up
    assert sum(v[0] for v in tally.values()) == 2 * (s.w + s.l + s.d)
    assert sum(v[1] for v in tally.values()) == 2 * (s.w - s.l)
    assert TB.tally_score(tally) == pytest.approx(100.0 * (s.w + 0.5 * s.d) / (s.w + s.l + s.d), abs=1e-12)


def test_a_row_without_tallies_outside_the_look_or_unmirrored_is_refused():
    run_of = {"P": "p", "O": "o"}
    ok = {"row_id": "r", "player": {"sha256": "P"}, "opponent": {"sha256": "O"},
          "teams": {"t": {"p": [4, 3], "o": [4, 1]}}}
    assert TB.cell_tallies([ok], run_of) == {("p", "o"): {"t": [4, 2]}}
    with pytest.raises(TB.TeamBootstrapError, match="no per-team tally"):
        TB.cell_tallies([{**ok, "teams": None}], run_of)
    with pytest.raises(TB.TeamBootstrapError, match="outside the look"):
        TB.cell_tallies([{**ok, "opponent": {"sha256": "Z"}}], run_of)
    with pytest.raises(TB.TeamBootstrapError, match="piloted 4 times by the player and 3 by the opponent"):
        TB.cell_tallies([{**ok, "teams": {"t": {"p": [4, 3], "o": [3, 1]}}}], run_of)


@pytest.mark.parametrize("n", [2, 3])
def test_the_cell_weights_are_the_registered_statistics_own(n):
    design = X.design_of({a: tuple(f"{a}{k}" for k in range(n)) for a in X.ARMS})
    rng = random.Random(n)
    scores = {c: rng.uniform(30.0, 70.0) for c in design.cells()}
    for f in X.FACTORS:
        w = TB.cell_weights(design, f)
        assert math.fsum(w[c] * (scores[c] - 50.0) for c in w) == pytest.approx(
            X.factor_stat(design, scores, f).estimate, abs=1e-9)
        assert sum(1 for v in w.values() if v) == (2 * n) ** 2  # the factor's ON runs against its OFF runs


# ----------------------------------------------------------------------------------------- the bootstrap
def test_the_report_is_reported_carries_the_registered_statistic_and_is_deterministic():
    doc = make_doc()
    cells, rows = make_look(doc, spread=one_team)
    a = TB.bootstrap(doc, cells, rows, draws=2000)
    assert a == TB.bootstrap(doc, cells, rows, draws=2000)
    assert a["status"] == TB.REPORTED and a["decides"] is None and a["n_per_arm"] == 3
    assert a["teams"] == {p: TEAMS for p in X.PANELS} and a["shared_team_ids"] == {}
    verdict = K.read(doc, cells)
    for f in X.FACTORS:
        got, want = a["factors"][f], verdict["factors"][f]
        assert got["read_on"] == want["read_on"] and got["outcome_registered"] == want["outcome"]
        view = want["views"][got["read_on"]]
        assert got["estimate_pp"] == view["estimate_pp"] and got["registered"]["se_pp"] == view["se_pp"]
        assert got["registered"]["interval_pp"] == view["interval_pp"]
        assert set(got["views"]) == set(X.VIEWS)
    b = TB.bootstrap(doc, cells, rows, draws=2000, seed=TB.SEED + 1)
    assert b["factors"]["width"]["team"]["se_pp"] != a["factors"]["width"]["team"]["se_pp"]
    assert b["factors"]["width"]["registered"] == a["factors"]["width"]["registered"]


def test_a_team_effect_shared_by_every_cell_is_seen_by_the_bootstrap_and_not_by_the_seed_interval():
    """Two looks with the SAME cell scores, so the same registered statistic and class. In one every team carries
    its share of each cell's W - L; in the other ONE team carries all of it, in every cell. The seed interval cannot
    tell them apart; the team bootstrap's SE is 0 in the first and the binomial's in the second."""
    doc = make_doc()
    flat = TB.bootstrap(doc, *make_look(doc, spread=even))["factors"]["width"]
    lump = TB.bootstrap(doc, *make_look(doc, spread=one_team))["factors"]["width"]
    assert flat["registered"] == lump["registered"] and flat["estimate_pp"] == lump["estimate_pp"] == pytest.approx(4.0)
    assert flat["registered"]["class"] == X.WIN and flat["outcome_registered"] == X.WIN
    assert flat["team"]["se_pp"] < 1e-9 and flat["class_with_team_variance"] == X.WIN and not flat["class_moves"]
    # team 0's multiplicity is Binomial(TEAMS, 1/TEAMS); a panel's statistic is 4 pp times it; N weights the panels
    panel_sd = 4.0 * math.sqrt(TEAMS * (1.0 / TEAMS) * (1.0 - 1.0 / TEAMS))
    want = panel_sd * math.sqrt(sum(w * w for w in X.VIEWS["N"].values()))
    assert lump["team"]["se_pp"] == pytest.approx(want, rel=0.05)
    assert lump["views"]["forum72"]["team"]["se_pp"] == pytest.approx(panel_sd, rel=0.05)
    assert lump["combined"]["se_pp"] == pytest.approx(math.hypot(lump["registered"]["se_pp"], lump["team"]["se_pp"]))
    assert lump["combined"]["interval_pp"][0] < 0.0 < lump["registered"]["interval_pp"][0]
    assert lump["class_with_team_variance"] == X.UNDECIDED and lump["class_moves"]
    assert 0.5 < lump["combined"]["team_share_of_variance"] < 1.0
    assert lump["outcome_registered"] == X.WIN  # the look's own class is untouched


def test_the_net_rule_factor_reports_how_often_its_sum_stays_above_zero():
    doc = make_doc()
    strength = {"C": 0.0, "W": 0.0, "Br": 2.0, "WB": 2.0}
    even_net = TB.bootstrap(doc, *make_look(doc, even, strength))["factors"]["breadth"]["net"]
    assert even_net["cost_view"] == "home72" and even_net["gain_view"] == "forum72"
    assert even_net["net_pp"] == pytest.approx(4.0) and even_net["share_of_resamples_above_zero"] == 1.0
    lump_net = TB.bootstrap(doc, *make_look(doc, one_team, strength))["factors"]["breadth"]["net"]
    # the sum is 0 whenever team 0 is drawn on neither panel: (1 - 1/TEAMS) ** (2 * TEAMS) of the resamples
    assert lump_net["share_of_resamples_above_zero"] == pytest.approx(1.0 - 0.9 ** 20, abs=0.02)
    assert "net" not in TB.bootstrap(doc, *make_look(doc))["factors"]["width"]


def test_the_50m_look_is_resampled_on_its_harm_views():
    doc = make_doc(look="50M")
    got = TB.bootstrap(doc, *make_look(doc, one_team, {"C": 0.0, "W": -4.0, "Br": 0.0, "WB": -4.0}), draws=2000)
    assert {f: got["factors"][f]["read_on"] for f in X.FACTORS} == X.HARM_VIEW
    w = got["factors"]["width"]
    assert w["outcome_registered"] == w["class_before_net_rule"] == X.STOP_HARM
    assert w["class_with_team_variance"] == X.CONTINUE and w["class_moves"]


def test_an_inconclusive_look_is_not_resampled_and_a_survivor_look_is_refused():
    doc = make_doc()
    cells, rows = make_look(doc)
    short = dict(cells, forum72=cells["forum72"][:-1])
    got = TB.bootstrap(doc, short, rows)
    assert got["outcome"] == K.INCONCLUSIVE and "factors" not in got and got["reasons"]
    assert TB.bootstrap(doc, cells, rows, row_reasons=["p72: rows played at ['bbbb'], not the pin"])["outcome"] \
        == K.INCONCLUSIVE
    with pytest.raises(TB.TeamBootstrapError, match="survivor look"):
        TB.bootstrap(make_doc(survivor="breadth"), cells, rows)
    with pytest.raises(TB.TeamBootstrapError, match="at least 100"):
        TB.bootstrap(doc, cells, rows, draws=10)


def test_tallies_that_are_not_the_ledger_cell_are_refused():
    doc = make_doc()
    cells, rows = make_look(doc)
    bad = {p: [dict(r) for r in rs] for p, rs in rows.items()}
    t = dict(bad["p72"][0]["teams"])
    t["p72-t0"] = {"p": list(t["p72-t0"]["p"]), "o": [t["p72-t0"]["o"][0], t["p72-t0"]["o"][1] + 1]}
    bad["p72"][0]["teams"] = t
    with pytest.raises(TB.TeamBootstrapError, match="p72: .* the team tallies sum to 1000 games and W - L = "):
        TB.bootstrap(doc, cells, bad)
    with pytest.raises(TB.TeamBootstrapError, match="home72: the rows hold 53 cells, the design 54"):
        TB.bootstrap(doc, cells, dict(rows, home72=rows["home72"][1:]))


# --------------------------------------------------------------------------------------- a real ledger
def _write_ledger(root: Path, doc: Dict[str, Any]) -> None:
    """A FULL look's rows in a real count ledger, as ``main.h2h`` builds them: one batch of the panel's registered
    pairs a cell. A pair is split (each side wins once) except: the cell's W - L sits on ``|edge|`` pairs the
    stronger side wins twice, all of them with ONE first team (the cell's own, so the teams differ in what they
    carry), and the last pair is drawn twice."""
    import types

    from agents.training import eval_ledger as L
    from agents.training.trace_result import DRAW, LOSS, WIN
    from main.h2h import panel as PN
    from main.h2h import play as PL
    from utils.rust_env import episode as EP

    packed = [f"packed team {i}" for i in range(8)]
    ref = {f["checkpoint"]: PL.PlayerRef(f["sha256"][:8], f["checkpoint"], str(root / "cfg.json"), "/r", f["run"],
                                         "explicit_zip", f["steps"], f["sha256"])
           for a in doc["runs"] for f in doc["runs"][a]}
    run_at = {f["checkpoint"]: f["run"] for a in doc["runs"] for f in doc["runs"][a]}
    root.mkdir(parents=True, exist_ok=True)
    (root / "cfg.json").write_text(json.dumps({"progress_decision_tense": False, "progress_switch_freeze": False}))
    w = L.LedgerWriter(root, producer="h2h")
    for panel in X.PANELS:
        nm = doc["names"][panel]
        pairs = X.PAIRS[panel]
        w.register_family(nm["family"], decision_kind="ab_verdict", rule=nm["rule"], protocol=K.PROTOCOL)
        req = w.open_request(nm["request"], kind="adhoc", purpose="ab", family=nm["family"])
        eng = types.SimpleNamespace(
            compute=PL.Compute(device="cpu"), regime=PL.regime_for(EP.stall_threshold(), PN.panel_team_set_id(panel)),
            torch_version="t", core_stamp="s", team_check={"a": "x"}, historical={"a": []}, mode="off",
            levels=("off", "off"), oracle=None)
        for i, (player, opponent) in enumerate(doc["cells"]):
            a, b = ref[player], ref[opponent]
            edge = round(pairs * (STRENGTH[_arm(run_at[player])] - STRENGTH[_arm(run_at[opponent])]) / 50.0)
            games: List[Dict[str, Any]] = []
            carried = 0
            for k in range(pairs):
                ta = k % 8
                tb = (ta + 1 + (k // 8) % 7) % 8
                if ta == i % 8 and carried < abs(edge):
                    carried += 1
                    first = second = WIN if edge > 0 else LOSS
                elif k == pairs - 1:
                    first = second = DRAW
                else:
                    first, second = WIN, LOSS
                for swapped, teams, res in ((False, [ta, tb], first), (True, [tb, ta], second)):
                    games.append({"game": len(games), "result": res, "swapped": swapped, "teams": teams,
                                  "near_ties": 0, "near_ties_wide": 0})
            assert carried == abs(edge)
            key = f"h2h:{a.sha256}:{b.sha256}"
            row = PL.build_row(writer=w, run_label=nm["request"], commit=PIN, a=a, b=b, eng=eng, purpose="ab",
                               key=key, sched_seed=0, batch=0, cseed=PL.cycle_seed(0, key, 0),
                               score=PL.score_games(games, packed, pairs), t_start=L.utc_now(), t_end=L.utc_now(),
                               wall_s=2.0, load={"start": [0, 0, 0], "end": [0, 0, 0], "contention": 1.0},
                               executor={}, request=req)
            assert L.validate_row(row) == []
            claim = w.claim(req["request_id"], batch=0, player=a.sha256, opponent=b.sha256,
                            regime_id=row["regime"]["regime_id"], expected_wall_s=10)
            w.append_row(row, claim)
    w.close()


@pytest.mark.integration
def test_the_cli_reads_a_real_ledger_through_the_looks_own_read_and_writes_a_reported_file(tmp_path, capsys):
    doc = make_doc(n=2)
    work, root = tmp_path / "w", tmp_path / "ledger"
    work.mkdir()
    (work / "look.json").write_text(json.dumps(doc))
    _write_ledger(root, doc)
    assert K.main(["read", "--work", str(work), "--ledger", str(root)]) == 0
    verdict = json.loads((work / "result" / "verdict.json").read_text())
    assert verdict["factors"]["width"]["views"]["N"]["estimate_pp"] == pytest.approx(4.0)
    capsys.readouterr()
    assert TB.main(["--work", str(work), "--ledger", str(root), "--draws", "500"]) == 0
    rep = json.loads((work / "result" / "team_bootstrap.json").read_text())
    assert json.loads(capsys.readouterr().out) == rep
    assert json.loads((work / "result" / "verdict.json").read_text()) == verdict  # the verdict file is not touched
    assert rep["status"] == TB.REPORTED and rep["teams"] == {p: 8 for p in X.PANELS}
    for f in X.FACTORS:
        got = rep["factors"][f]
        assert got["estimate_pp"] == verdict["factors"][f]["views"][got["read_on"]]["estimate_pp"]
        assert got["outcome_registered"] == verdict["factors"][f]["outcome"]
        assert got["team"]["se_pp"] > 0.0


@pytest.mark.integration
def test_the_cli_exits_three_on_an_unplayed_look_and_two_on_a_missing_plan(tmp_path, capsys):
    doc = make_doc(n=2)
    work = tmp_path / "w"
    work.mkdir()
    assert TB.main(["--work", str(work), "--ledger", str(tmp_path / "ledger")]) == 2
    assert "REFUSED" in capsys.readouterr().err
    (work / "look.json").write_text(json.dumps(doc))
    (tmp_path / "ledger").mkdir()
    assert TB.main(["--work", str(work), "--ledger", str(tmp_path / "ledger")]) == 3
    assert json.loads((work / "result" / "team_bootstrap.json").read_text())["outcome"] == K.INCONCLUSIVE
