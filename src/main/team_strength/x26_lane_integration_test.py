"""PILOT A's lane on the REAL engine (CPU, tiny): every planned pair is played on its planned teams from both seats,
units are written durably and a re-run resumes instead of re-playing; a unit from another registration is refused.
A seeded PERTURBED-fresh checkpoint pilots both sides (``main.h2h``'s fixture)."""
from __future__ import annotations

import json

import pytest

from main.h2h import play as PL
from main.team_strength import fit as F
from main.team_strength import register as RG
from main.team_strength import x26_lane as XL
from utils.paths import repo_path

pytestmark = [pytest.mark.sim, pytest.mark.integration]

HOME72 = repo_path("designs/research_state/measurements/n0_endofrun_2026-09-27/teams_home72")
COMPUTE = dict(device="cpu", backend="eager", n_envs=4, threads=2, torch_threads=1, front="ffi", profile="selfcheck")


@pytest.fixture(scope="module")
def setup(built, checkpoints, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tsr")
    files = sorted(HOME72.iterdir())
    # 4 rated teams x 1 pair, units of 2 pairs: a perturbed-fresh checkpoint plays every game to the 250-turn stall
    # threshold (~110 s for one round of 4 envs on one torch thread), so each unit is ONE round
    for name, chunk in (("a", files[:2]), ("b", files[2:4])):
        d = tmp / name
        d.mkdir()
        for f in chunk:
            (d / f.name).write_text(f.read_text())
    zip_a, _zip_b = checkpoints
    spec = {"name": "t", "sets": {"a": str(tmp / "a"), "b": str(tmp / "b")},
            "reference": {"quota": {"a": 1, "b": 1}, "seed": 1},
            "pilots": {"x26": {"model": zip_a, "sha256": PL.file_sha256(zip_a), "pairs_per_team": 1,
                               "unit_pairs": 2, "seed": 9}},
            "fit": {"prior_sd": 2.0, "B": 20, "seed": 5, "competitive_margin_elo": 35.0, "comparisons": [["a", "b"]]}}
    reg = RG.build(spec)
    p = tmp / "reg.json"
    p.write_text(json.dumps(reg))
    state = tmp / "state"
    comp = PL.Compute(**COMPUTE)
    first = XL.run(RG.load(p), state, compute=comp, max_units=1, emit=lambda _m: None)
    second = XL.run(RG.load(p), state, compute=comp, emit=lambda _m: None)
    third = XL.run(RG.load(p), state, compute=comp, emit=lambda _m: None)
    return RG.load(p), state, first, second, third


def test_units_are_durable_and_a_rerun_resumes(setup):
    reg, state, first, second, third = setup
    units = XL.x26_units(reg)
    assert len(units) == 2                        # 4 rated teams x 1 pair = 4 pairs, units of 2
    assert first == {"played": 1, "remaining": 1}
    assert second == {"played": 1, "remaining": 0}
    assert third == {"played": 0, "remaining": 0}
    assert sorted(p.name for p in (state / "x26" / "units").iterdir()) == [f"{u.name}.json" for u in units]


def test_every_pair_played_its_planned_teams_and_reads_as_games(setup):
    reg, state, *_ = setup
    plan = {u.name: u for u in XL.x26_units(reg)}
    for f in sorted((state / "x26" / "units").iterdir()):
        rec = json.loads(f.read_text())
        u = plan[rec["unit"]]
        assert rec["seed"] == u.seed and rec["registration_digest"] == reg["digest"]
        assert [(r["subject"], r["ref"]) for r in rec["pairs"]] == [(p.subject, p.ref) for p in u.pairs]
        for r in rec["pairs"]:
            assert [g["swapped"] for g in r["games"]] == [False, True]
    games = XL.load_games(reg, state)
    assert len(games) == 2 * 4
    assert all(isinstance(g, F.Game) and g.a != g.b for g in games)


def test_a_unit_from_another_registration_is_refused(setup, tmp_path):
    reg, state, *_ = setup
    other = dict(reg, digest="0" * 64)
    with pytest.raises(XL.LaneError):
        XL.load_games(other, state)


def test_the_fit_reads_the_lane_into_per_team_and_per_set_ratings(setup):
    from main.team_strength import cli

    reg, state, *_ = setup
    out, ratings = cli.compute(reg, state)
    assert set(ratings) == {"x26"} and out["pilots"]["x26"]["n_games"] == 8
    assert set(out["pilots"]["x26"]["sets"]) == {"a", "b"}
    assert "a-b" in out["pilots"]["x26"]["comparisons"]
    assert sorted(r["team_id"] for r in out["pilots"]["x26"]["teams"]) == sorted(ratings["x26"].ids)
