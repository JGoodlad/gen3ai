"""The bank's schedule (pure) and its refusals; the games themselves are ``bank_integration_test.py``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from main.offpool_belief import BANK_SCHEMA, OffPoolBeliefError
from main.offpool_belief import bank as B


def test_every_team_gets_its_games_and_a_game_is_a_function_of_seed_and_index() -> None:
    sch = B.game_schedule(5, 7, 3, seed=11, same_panel=False)
    assert [g.g for g in sch] == list(range(15))
    assert [g.team for g in sch] == [0, 1, 2, 3, 4] * 3
    assert all(0 <= g.home < 7 and len(g.seed) == 4 and all(0 <= s < 65536 for s in g.seed) for g in sch)
    # a longer schedule only APPENDS: game g never depends on how many games there are
    longer = B.game_schedule(5, 7, 4, seed=11, same_panel=False)
    assert longer[:15] == sch
    assert B.game_schedule(5, 7, 3, seed=12, same_panel=False) != sch
    assert len({g.seed for g in sch}) == 15


def test_a_same_panel_schedule_has_no_mirror_game() -> None:
    plain = B.game_schedule(6, 6, 40, seed=3, same_panel=False)
    assert any(g.home == g.team for g in plain), "the fixture never draws a mirror: the test below is vacuous"
    same = B.game_schedule(6, 6, 40, seed=3, same_panel=True)
    assert all(g.home != g.team for g in same)
    # only the mirror draws moved, each to the next team
    for a, b in zip(plain, same):
        assert b.home == (a.home if a.home != a.team else (a.home + 1) % 6) and a.seed == b.seed
    with pytest.raises(ValueError):
        B.game_schedule(1, 1, 1, seed=0, same_panel=True)
    with pytest.raises(ValueError):
        B.game_schedule(0, 4, 1, seed=0, same_panel=False)


def _write_bank(d: Path, **over: object) -> None:
    from utils.hashing import sha256_file

    from main.offpool_belief import panels as PN

    ref = PN.panel_ref("p72")
    n = 3
    rows = {"obs": np.zeros((n, 4), np.float32), "mask": np.ones((n, 2), bool), "game": np.arange(n, dtype=np.int32),
            "team": np.arange(n, dtype=np.int32), "home": np.zeros(n, np.int32), "turn": np.ones(n, np.int32),
            "winner": np.ones(n, np.int8), "belief_species": -np.ones((n, 6), np.int32),
            "belief_moves": -np.ones((n, 6, 4), np.int32), "known_moves": -np.ones((n, 6, 4), np.int32),
            "item_label": -np.ones((n, 6), np.int32), "item_mask": np.zeros((n, 6), np.float32)}
    d.mkdir(parents=True)
    np.savez_compressed(d / B.ROWS_FILE, **rows)
    man = {"schema": BANK_SCHEMA, "panel": "p72", "panel_content_digest": ref.content_digest,
           "panel_held_out": ref.held_out, "team_labels": [l for l, _t in ref.teams], "n_rows": n,
           "rows_sha256": sha256_file(d / B.ROWS_FILE)}
    man.update(over)
    (d / B.MANIFEST_FILE).write_text(json.dumps(man))


def test_load_bank_reads_what_build_wrote_and_refuses_what_moved(tmp_path: Path) -> None:
    _write_bank(tmp_path / "ok")
    bank = B.load_bank(tmp_path / "ok")
    assert bank.panel == "p72" and bank.n_teams == 72 and set(bank.rows) == set(B.ROW_KEYS)

    _write_bank(tmp_path / "rows", rows_sha256="0" * 64)
    with pytest.raises(OffPoolBeliefError, match="sha256"):
        B.load_bank(tmp_path / "rows")
    _write_bank(tmp_path / "digest", panel_content_digest="0" * 64)
    with pytest.raises(OffPoolBeliefError, match="no longer what the bank recorded"):
        B.load_bank(tmp_path / "digest")
    _write_bank(tmp_path / "status", panel_held_out=False)          # a bank that claims the panel is in-pool
    with pytest.raises(OffPoolBeliefError, match="no longer what the bank recorded"):
        B.load_bank(tmp_path / "status")
    _write_bank(tmp_path / "schema", schema="something_else")
    with pytest.raises(OffPoolBeliefError, match="schema"):
        B.load_bank(tmp_path / "schema")
    _write_bank(tmp_path / "count", n_rows=4)
    with pytest.raises(OffPoolBeliefError, match="rows"):
        B.load_bank(tmp_path / "count")
    with pytest.raises(OffPoolBeliefError, match="not a bank"):
        B.load_bank(tmp_path / "nothing_here")


def test_build_refuses_a_checkout_models_and_an_existing_bank(tmp_path: Path) -> None:
    from utils.paths import repo_root

    with pytest.raises(OffPoolBeliefError, match="inside the checkout"):
        B.build("p72", "x.zip", repo_root() / "bank_here")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / B.MANIFEST_FILE).write_text("{}")
    with pytest.raises(OffPoolBeliefError, match="never overwritten"):
        B.build("p72", "x.zip", tmp_path / "b")
    with pytest.raises(OffPoolBeliefError, match="SEALED"):
        B.build("forum216b", "x.zip", tmp_path / "sealed")
    assert not (tmp_path / "sealed").exists()
