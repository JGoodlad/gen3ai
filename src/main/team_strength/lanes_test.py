"""The scheduled panel builder (every planned draw lands, an unplanned one is refused) and pilot B's capture reader
(which team each Kakuna played, who won, and the unit's own record must agree). Node packs / exports the teams."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.training.rust_eval import seeds as SD
from main.team_strength import kakuna_lane as KL
from main.team_strength import panel as PN
from main.team_strength.scheduled import ScheduleMiss, ScheduledTeams
from utils.paths import repo_path

pytestmark = pytest.mark.integration

HOME72 = repo_path("designs/research_state/measurements/n0_endofrun_2026-09-27/teams_home72")
SIM = repo_path("deps", "pokemon-showdown", "dist", "sim")


@pytest.fixture(scope="module")
def panel():
    texts = [f.read_text() for f in sorted(HOME72.iterdir())[:4]]
    teams = {PN.team_id(t): t for t in texts}
    return teams, ScheduledTeams(teams)


def test_every_planned_draw_lands_on_its_team_in_both_games_of_a_pair(panel):
    teams, b = panel
    ids = sorted(teams)
    plan = [(ids[0], ids[1]), (ids[2], ids[3]), (ids[1], ids[3])]
    seed = 123456789
    b.schedule(seed, plan)
    for k, (x, y) in enumerate(plan):
        for g in (2 * k, 2 * k + 1):
            key, _swapped = SD.pair_game(seed, "h2h", g, True)
            assert b.id_by_packed[SD.draw_team(b, key, SD.TRAINEE)] == x
            assert b.id_by_packed[SD.draw_team(b, key, SD.OPPONENT)] == y
    stray, _ = SD.pair_game(seed + 1, "h2h", 0, True)
    with pytest.raises(ScheduleMiss):
        SD.draw_team(b, stray, SD.TRAINEE)
    with pytest.raises(ScheduleMiss):
        b.schedule(seed, [(ids[0], "tsr:not-a-panel-team")])


def test_a_team_the_validator_rejects_is_refused():
    t = sorted(HOME72.iterdir())[0].read_text()
    bad = t.replace("Earthquake", "Spore", 1)            # an illegal move: the validator drops the team
    assert bad != t
    with pytest.raises(ScheduleMiss):
        ScheduledTeams({PN.team_id(t): t, PN.team_id(bad): bad})


def _unit(tmp: Path, panel, records) -> Path:
    """A unit dir with one capture per record ``(a id, b id, winner side 'A'/'B'/None, games.jsonl result)``."""
    teams, b = panel
    u = tmp / "kak_x_000"
    (u / "capture").mkdir(parents=True)
    rows = []
    for i, (ida, idb, win, res) in enumerate(records, start=1):
        a_first = i % 2 == 1
        na, nb = "MetaKakunaAway1", "MetaKakunaHome1"
        p1 = {"name": na if a_first else nb, "team": b.packed_by_id[ida if a_first else idb]}
        p2 = {"name": nb if a_first else na, "team": b.packed_by_id[idb if a_first else ida]}
        start = {"formatid": "gen3ou", "p1": p1, "p2": p2, "seed": [1, 2, 3, 4]}
        chunks = [f"|turn|1\n|win|{na if win == 'A' else nb}\n"] if win else ["|turn|1\n|tie\n"]
        (u / "capture" / f"battle-gen3ou-{i}.json").write_text(json.dumps(
            {"tag": f"battle-gen3ou-{i}", "seed": [1, 2, 3, 4], "commands": ["START " + json.dumps(start)],
             "chunks": chunks}))
        rows.append({"index": i, "result": res})
    (u / "games.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return u


def test_the_capture_names_each_sides_team_and_the_winner(tmp_path, panel, monkeypatch):
    monkeypatch.setattr(KL, "GAMES", 3)
    teams, _b = panel
    ids = sorted(teams)
    side_a = {ids[0]: ids[0], ids[1]: ids[1]}
    side_b = {ids[2]: ids[2], ids[3]: ids[3]}
    u = _unit(tmp_path, panel, [(ids[0], ids[2], "A", "win"), (ids[1], ids[3], "B", "loss"),
                                (ids[0], ids[3], None, "loss")])
    rows, why = KL.parse_unit(u, side_a=side_a, side_b=side_b, texts=teams, showdown_sim=SIM)
    assert why == []
    assert [(r["a"], r["b"], r["s"]) for r in rows] == [(ids[0], ids[2], 1.0), (ids[1], ids[3], 0.0),
                                                       (ids[0], ids[3], 0.5)]
    assert {r["tier_a"] for r in rows} == {"strict"}


def test_a_capture_that_disagrees_with_the_record_or_its_declared_set_voids_the_unit(tmp_path, panel, monkeypatch):
    monkeypatch.setattr(KL, "GAMES", 2)
    teams, _b = panel
    ids = sorted(teams)
    side_a, side_b = {ids[0]: ids[0]}, {ids[2]: ids[2]}
    u = _unit(tmp_path, panel, [(ids[0], ids[2], "A", "loss"), (ids[0], ids[2], "B", "loss")])
    _rows, why = KL.parse_unit(u, side_a=side_a, side_b=side_b, texts=teams, showdown_sim=SIM)
    assert any("games.jsonl" in w for w in why)
    u2 = tmp_path / "other"
    u2.mkdir()
    u3 = _unit(u2, panel, [(ids[1], ids[2], "A", "win"), (ids[0], ids[2], "A", "win")])
    _rows, why = KL.parse_unit(u3, side_a=side_a, side_b=side_b, texts=teams, showdown_sim=SIM)
    assert any("side A" in w for w in why)


def test_install_never_overwrites_a_different_set(tmp_path):
    src = sorted(HOME72.iterdir())[:2]
    files = {f"s__{p.stem}.gen3ou_team": p for p in src}
    d = tmp_path / "mm" / "gen3ou"
    dig = KL.install_one(d, files)
    assert KL.install_one(d, files) == dig                 # idempotent on identical bytes
    (d / next(iter(files))).write_text("x")
    with pytest.raises(KL.KakunaLaneError):
        KL.install_one(d, files)
