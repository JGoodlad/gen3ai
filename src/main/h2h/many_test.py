"""``main.h2h.many``'s pure parts: the cell file, the cross, and the CLI's refusals (no engine)."""
from __future__ import annotations

import json

import pytest

from main.h2h import cli as CLI
from main.h2h import many as MANY
from main.h2h import play as PL


def test_a_cell_file_takes_pairs_and_objects_in_order(tmp_path):
    f = tmp_path / "cells.json"
    f.write_text(json.dumps([["a", "b"], {"player": "c", "opponent": "d", "note": "x"}]))
    assert MANY.cells_from_file(str(f)) == [("a", "b"), ("c", "d")]


@pytest.mark.parametrize("raw", [[], {"player": "a"}, [["a"]], [{"player": "a"}], [["a", "b", "c"]]])
def test_a_malformed_cell_file_is_refused(tmp_path, raw):
    f = tmp_path / "cells.json"
    f.write_text(json.dumps(raw))
    with pytest.raises(PL.H2HError):
        MANY.cells_from_file(str(f))


def test_the_cross_is_every_player_against_every_opponent_players_outer():
    assert MANY.cross(["x1", "x2"], ["b1", "b2", "b3"]) == [
        ("x1", "b1"), ("x1", "b2"), ("x1", "b3"), ("x2", "b1"), ("x2", "b2"), ("x2", "b3")]


def test_the_cli_refuses_players_without_opponents_and_opponents_with_a_cell_file(tmp_path):
    f = tmp_path / "cells.json"
    f.write_text(json.dumps([["a", "b"]]))
    with pytest.raises(SystemExit, match="--opponents"):
        CLI.main(["play-many", "--players", "a", "--pairs", "2", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="not --cells"):
        CLI.main(["play-many", "--cells", str(f), "--opponents", "b", "--pairs", "2", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):                                   # argparse: exactly one of --cells / --players
        CLI.main(["play-many", "--pairs", "2"])


def test_play_cells_and_the_preflight_refuse_an_empty_cell_list(tmp_path):
    with pytest.raises(PL.H2HError, match="no cells"):
        MANY.play_cells(str(tmp_path / "o"), [], pairs=2, run_label="s", compute=PL.Compute())
    with pytest.raises(PL.H2HError, match="no cell to check"):
        MANY.preflight([])
