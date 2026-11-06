"""An anchor group's ``games`` subset (X65's G-BANK: one banked deal a panel and seed re-played) and the spec's
``PENDING`` values (a registration whose students do not exist before its pin amendment): the spec parses and refuses
them, ``plan`` plans only the registered games, ``assemble`` counts only them (``unplayed`` and the row check), an
import must have played them, and ``check-spec`` takes PENDING only with ``--pin-pending``. No game is played: a stub
unit runner writes each unit's summary."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.reads import assemble as AS
from main.reads import imports as IM
from main.reads import runner as RN
from main.reads import speccheck as SC
from main.reads.spec import PENDING, SpecError, parse
from main.x35 import kakuna as KA

CODE = {"commit": "e05c45eeb" + "0" * 31, "dirty": False, "root": "/tmp/pin"}
GAMES = {"0": [0], "1": [1], "2": [2]}


def _doc(path: str, sha: str, **anchor):
    g = {"opponent": "metamon:Kakuna", "teamsets": ["home72", "forum72"], "unit_seeds": "0-2", "per_game": True,
         "models": ["pstar"], "games": GAMES}
    g.update(anchor)
    return {"schema": "gen3_read_spec_v1", "read": "gbank_demo_2026-11-05",
            "models": {"pstar": {"arm": "pstar", "seed": 2601, "ref": {"kind": "pinned", "path": path, "sha256": sha}}},
            "anchor": {"gb": g},
            "decision": {"rule": "anchor_trajectory", "params": {"on": "pstar", "off": "pstar", "anchor_group": "gb",
                                                                 "goal_bar": 0.5, "min_unit_frac": 0.9}}}


@pytest.fixture
def model(tmp_path):
    from utils.hashing import sha256_file

    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "model_config.json").write_text("{}")
    z = d / "checkpoint_280000170_steps.zip"
    z.write_bytes(b"pstar")
    return str(z), sha256_file(z)


def test_the_spec_takes_a_games_subset_of_a_per_game_group(model):
    a = parse(_doc(*model)).anchor["gb"]
    assert a.games == {0: (0,), 1: (1,), 2: (2,)} and a.n_planned() == 3
    assert a.planned(1, 1) and not a.planned(1, 0) and not a.planned(3, 3)
    full = parse(_doc(*model, games=None)).anchor["gb"]
    assert full.games is None and full.n_planned() == 60 and full.planned(1, 0)


@pytest.mark.parametrize("over,needle", [
    ({"per_game": False}, "per_game"),
    ({"games": {"0": [0], "1": [1]}}, "exactly the group's unit_seeds"),
    ({"games": {"0": [0], "1": [1], "2": [2], "3": [3]}}, "exactly the group's unit_seeds"),
    ({"games": {"0": [20], "1": [1], "2": [2]}}, r"\[0, 20\)"),
    ({"games": {"0": [0, 0], "1": [1], "2": [2]}}, "distinct"),
    ({"games": {"0": [], "1": [1], "2": [2]}}, "non-empty"),
    ({"games": {"00": [0], "1": [1], "2": [2]}}, "decimal"),
    ({"games": {}}, "an object"),
])
def test_the_spec_refuses_a_bad_games_subset(model, over, needle):
    with pytest.raises(SpecError, match=needle):
        parse(_doc(*model, **over))


def _fake_unit(played):
    def run(st, u):
        played.append(u.name)
        d = st / "cells" / u.name
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({
            "status": "OK", "n": 1, "wins": 1, "ties": 0, "regime_verified_decisions": True, "peer_clean": True,
            "team_source_asymmetry": False}))
        return 0
    return run


def test_plan_plays_and_assemble_counts_only_the_registered_games(tmp_path, model, monkeypatch):
    spec_p = tmp_path / "read_spec.json"
    spec_p.write_text(json.dumps(_doc(*model)))
    state = tmp_path / "state"
    RN.plan(spec_p, state, code=CODE, step_of=lambda p: 280000170)
    st = state / "anchor" / "gb" / "pstar"
    names = sorted(u.name for u in KA.load_plan(st))
    assert names == sorted(f"pstar__{ts}__{s:02d}__g{s:02d}" for ts in ("home72", "forum72") for s in (0, 1, 2))
    spec = RN.load_state(state)
    grp = AS.assemble(spec, state)["anchor"]["gb"]
    assert grp["unplayed"] == 6 and grp["cells"]["pstar"]["forum72"]["planned"] == 3
    played: list = []
    for u in KA.load_plan(st)[:4]:
        _fake_unit(played)(st, u)
    rows = AS.anchor_units(st)
    grp = AS.anchor_group(rows, labels=["pstar"], teamsets=["home72", "forum72"], unit_seeds=[0, 1, 2], per_game=True,
                          games={0: (0,), 1: (1,), 2: (2,)})
    assert grp["unplayed"] == 2
    bad = [dict(rows[0], game=5, unit="pstar__home72__00__g05")]
    with pytest.raises(AS.AssembleError, match="registered games"):
        AS.anchor_group(bad, labels=["pstar"], teamsets=["home72", "forum72"], unit_seeds=[0, 1, 2], per_game=True,
                        games={0: (0,), 1: (1,), 2: (2,)})
    # without the subset the same group plans every game: 2 sets x 3 seeds x 20, less the 4 played
    grp = AS.anchor_group(rows, labels=["pstar"], teamsets=["home72", "forum72"], unit_seeds=[0, 1, 2], per_game=True)
    assert grp["unplayed"] == 116


def test_an_import_must_have_played_every_registered_game(tmp_path, model):
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    (earlier / "spec.json").write_text(json.dumps(_doc(*model)))
    from utils.hashing import sha256_file

    sha = sha256_file(earlier / "spec.json")
    doc = _doc(*model, models=["pstar"], **{"import": {"state": str(earlier), "spec_sha256": sha, "group": "gb",
                                                       "models": ["pstar"]}})
    IM.check(parse(doc), "anchor", "gb")                            # the same subset: the same cell
    doc["anchor"]["gb"]["games"] = {"0": [0], "1": [1], "2": [3]}
    with pytest.raises(IM.ReadImportError, match="were not played there"):
        IM.check(parse(doc), "anchor", "gb")
    doc["anchor"]["gb"].pop("games")                                 # every game: the subset never played them
    with pytest.raises(IM.ReadImportError, match="were not played there"):
        IM.check(parse(doc), "anchor", "gb")


# ------------------------------------------------------------------------------------------------ PENDING
def test_a_pending_value_parses_only_when_allowed_and_is_listed(model):
    doc = _doc(*model)
    doc["models"]["pstar"]["ref"].update(sha256=PENDING, path=PENDING)
    with pytest.raises(SpecError, match="PENDING"):
        parse(doc)
    sp = parse(doc, allow_pending=True)
    assert sp.pending == ("spec.models.pstar.ref.path", "spec.models.pstar.ref.sha256")
    assert parse(_doc(*model)).pending == ()
    doc["models"]["pstar"]["ref"]["sha256"] = "pending"                 # only the exact literal
    with pytest.raises(SpecError, match="64 lowercase hex"):
        parse(doc, allow_pending=True)


def test_check_spec_takes_pending_only_with_pin_pending_and_plan_refuses_it(tmp_path, model):
    doc = _doc(*model)
    doc["models"]["pstar"]["ref"]["sha256"] = PENDING
    p = tmp_path / "read_spec.json"
    p.write_text(json.dumps(doc))
    out = SC.check_spec(p, pin_pending=True)
    assert out["pending"] == ["spec.models.pstar.ref.sha256"] and out["pin"]["state"] == "PENDING"
    with pytest.raises(SpecError, match="PENDING"):
        SC.check_spec(p)
    with pytest.raises(SpecError, match="PENDING"):
        RN.plan(p, tmp_path / "state", code=CODE, step_of=lambda q: 280000170)
    assert not (tmp_path / "state" / "spec.json").exists()


def test_an_import_of_a_pending_spec_refuses(tmp_path, model):
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    doc = _doc(*model)
    doc["models"]["pstar"]["ref"]["sha256"] = PENDING
    (earlier / "spec.json").write_text(json.dumps(doc))
    from utils.hashing import sha256_file

    imp = {"state": str(earlier), "spec_sha256": sha256_file(earlier / "spec.json"), "group": "gb", "models": ["pstar"]}
    with pytest.raises(IM.ReadImportError, match="does not load"):
        IM.check(parse(_doc(*model, **{"import": imp})), "anchor", "gb")
    assert Path(earlier / "spec.json").exists()
