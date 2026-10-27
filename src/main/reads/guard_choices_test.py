"""The stricter game-for-game guard: OUR choices from the captures, beside the 5-tuple.

``measurements/kakuna_determinism_2026-10-27/`` §6: the 5-tuple ``(half, index, our_team, result, turns)`` saw 11
divergences across E1 vs E3 where the captures show 15 games whose choices differ. A choice can differ and leave the
5-tuple equal (another move to the same result and turn count, or one that never ran). Each fixture pair here differs
ONLY in a choice, so its 5-tuples are equal and only the choice comparison can fail it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from main.reads import rules as RU
from main.reads.assemble import cell_choices, compare_choices, game_choices
from main.reads.rules_test import GAMES, _banked, _d3_cells, _d3_spec

START = 'START {"formatid": "gen3ou", "p1": {"name": "Gen3AIAnchor1"}, "p2": {"name": "MetaKakuna1"}}'


def _capture(tag: str, ours: List[str], theirs: List[str], *, interleave: bool = True) -> Dict:
    cmds = [START]
    if interleave:
        for a, b in zip(ours, theirs):
            cmds += [f"CHOOSE p1 {a}", f"CHOOSE p2 {b}"]
    else:                                                      # the same choices, the other side first (order-only)
        for a, b in zip(ours, theirs):
            cmds += [f"CHOOSE p2 {b}", f"CHOOSE p1 {a}"]
    return {"tag": tag, "seed": [1, 2, 3, 4], "commands": cmds, "chunks": [["p1", "|turn|1"]]}


def _cell(root: Path, games: List[List], flip: Optional[int] = None, order_only: Optional[int] = None) -> Path:
    """A played cell dir: ``games.jsonl`` + one capture per game; game ``flip`` has OUR second choice changed."""
    (root / "capture").mkdir(parents=True)
    rows = []
    for i, g in enumerate(games):
        tag = f"battle-gen3ou-{i + 1}"
        rows.append({"half": g[0], "index": g[1], "our_team": g[2], "result": g[3], "turns": g[4], "battle_tag": tag})
        ours = ["move surf", "move icebeam" if i == flip else "move substitute", "switch blissey"]
        cap = _capture(tag, ours, ["move spikes", "switch skarmory", "move toxic"], interleave=i != order_only)
        (root / "capture" / f"{tag}.json").write_text(json.dumps(cap))
    (root / "games.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


# ------------------------------------------------------------------------------------------- the digest
def test_a_changed_choice_changes_the_digest_and_an_order_only_game_does_not():
    base = _capture("t", ["move surf", "move substitute"], ["move spikes", "move toxic"])
    order = _capture("t", ["move surf", "move substitute"], ["move spikes", "move toxic"], interleave=False)
    other = _capture("t", ["move surf", "move icebeam"], ["move spikes", "move toxic"])
    assert game_choices(base) == game_choices(order)
    assert game_choices(base) != game_choices(other)


def test_a_cell_missing_a_capture_compares_nothing(tmp_path):
    cell = _cell(tmp_path / "c", GAMES[0])
    (cell / "capture" / "battle-gen3ou-2.json").unlink()
    games, choices, why = cell_choices(cell)
    assert games == [list(g) for g in GAMES[0]] and choices is None and "no capture" in why


def test_compare_choices_names_the_differing_game_and_never_passes_a_foreign_source(tmp_path):
    games = GAMES[0]
    here = _cell(tmp_path / "here", games, flip=1)
    there = _cell(tmp_path / "there", games, order_only=0)
    got = compare_choices(here, there, games)
    assert got["choices"] == "differ" and got["differing"] == [list(games[1][:2])]
    assert compare_choices(_cell(tmp_path / "same", games, order_only=1), there, games)["choices"] == "equal"
    # a capture source that does not hold the banked games is never compared (and never a pass)
    wrong = [games[0], ["peer_challenge", 1, "t2.txt", "win", 40]]
    assert compare_choices(here, there, wrong)["choices"] == "unavailable"
    assert compare_choices(here, None, games, unavailable="no source")["why"] == "no source"


# ------------------------------------------------------------------------------- the determinism guard
def _guard_fixture(tmp_path, monkeypatch, flip: Optional[int]):
    """D3's guard (rules_test's fixture) with captures on BOTH sides: the bank's rows name their source unit under
    the archive, and the re-played units carry their cell dirs."""
    archive = tmp_path / "archive"
    monkeypatch.setenv("GEN3AI_ARCHIVE_DIR", str(archive))
    rows = _banked(GAMES)
    for r in rows:
        r["source_unit"] = f"d1_read/h_a/c0_s3501__heldout72_hpfix__{int(r['unit_seed']):02d}"
    sp = _d3_spec(tmp_path, rows)
    cells = _d3_cells(GAMES)
    for u in cells["anchor"]["guard"]["units"]:
        s = int(u["unit_seed"])
        _cell(archive / "d1_read" / "h_a" / "cells" / f"c0_s3501__heldout72_hpfix__{s:02d}", GAMES[s])
        u["cell_dir"] = str(_cell(tmp_path / "play" / "cells" / u["unit"], GAMES[s], flip=flip if s == 0 else None))
    return sp, cells


def test_the_guard_fails_on_a_pair_that_differs_only_in_one_choice(tmp_path, monkeypatch):
    """THE STRICTER GUARD: equal 5-tuples, one choice different in unit 0's game 2. Before this the guard passed
    and the banked side was pooled."""
    sp, cells = _guard_fixture(tmp_path, monkeypatch, flip=1)
    doc = RU.inputs("d1", cells, sp.params, sp)
    guard = doc["import_off"]["guard"]
    u0 = next(c for c in guard["units"] if c["unit_seed"] == 0)
    assert u0["tuple_equal"] is True and u0["choices"] == "differ" and u0["games_equal"] is False
    assert u0["choices_differing"] == [["peer_challenge", 1]]
    assert guard["passed"] is False and doc["import_off"]["h_pooled"] is False
    v = RU.decide("d1", doc, sp.params)
    assert any("determinism guard failed" in r for r in v["reasons"])


def test_the_guard_passes_when_every_choice_matches(tmp_path, monkeypatch):
    sp, cells = _guard_fixture(tmp_path, monkeypatch, flip=None)
    doc = RU.inputs("d1", cells, sp.params, sp)
    guard = doc["import_off"]["guard"]
    assert guard["passed"] is True and {c["choices"] for c in guard["units"]} == {"equal"}


def test_a_bank_without_captures_is_recorded_unavailable_and_the_5_tuple_decides(tmp_path):
    """The bank names no source unit (as X51's rows): the choices are ``unavailable``, said so per unit, and the
    5-tuple decides alone, exactly as before."""
    sp = _d3_spec(tmp_path, _banked(GAMES))
    doc = RU.inputs("d1", _d3_cells(GAMES), sp.params, sp)
    guard = doc["import_off"]["guard"]
    assert guard["passed"] is True
    assert {c["choices"] for c in guard["units"]} == {"unavailable"}
    assert all(c["choices_why"] for c in guard["units"])


def test_x59s_repeat_unit_fails_void6_on_a_pair_that_differs_only_in_one_choice(tmp_path):
    """X59-S's repeat unit (VOID 6) against the student unit it repeats, both in this read: equal 5-tuples, one
    choice different."""
    from main.reads.rules_x59s import repeat_checks

    games = GAMES[0]

    def cells(flip):
        orig = {"unit": "s_h__forum72__00", "label": "s_h", "panel": "forum72", "unit_seed": 0, "counted": True,
                "games": [list(g) for g in games], "cell_dir": str(_cell(tmp_path / f"o{flip}", games))}
        rep = dict(orig, unit="rep__forum72__00", label="rep",
                   cell_dir=str(_cell(tmp_path / f"r{flip}", games, flip=flip)))
        return {"anchor": {"students": {"units": [orig]}, "repeat": {"units": [rep]}}}

    bad = repeat_checks(cells(1), {"group": "repeat", "of": "s_h"}, "students")
    assert bad[0]["tuple_equal"] is True and bad[0]["choices"] == "differ" and bad[0]["games_equal"] is False
    good = repeat_checks(cells(None), {"group": "repeat", "of": "s_h"}, "students")
    assert good[0]["games_equal"] is True and good[0]["choices"] == "equal"


# ------------------------------------------------------------------- REVIEW 2026-10-27 (anchor_guards_review)
@pytest.mark.xfail(strict=True, reason=(
    "REVIEW 2026-10-27 H2 (MAJOR, designs/ops/reviews/anchor_guards_review_2026-10-27.md): a bank that NAMES its "
    "capture source degrades to the 5-tuple alone when that capture is not under $GEN3AI_ARCHIVE_DIR (moved, pruned, "
    "another box), so one registered decision PASSES or VOIDs by the environment. Flips RED when a named-but-missing "
    "capture fails the guard."))
def test_a_bank_that_names_its_capture_never_passes_on_the_5_tuple_when_the_capture_is_gone(tmp_path, monkeypatch):
    sp, cells = _guard_fixture(tmp_path, monkeypatch, flip=1)
    monkeypatch.setenv("GEN3AI_ARCHIVE_DIR", str(tmp_path / "pruned"))
    doc = RU.inputs("d1", cells, sp.params, sp)
    assert doc["import_off"]["guard"]["passed"] is False


@pytest.mark.xfail(strict=True, reason=(
    "REVIEW 2026-10-27 H2 (MAJOR): the RE-PLAYED guard unit's own capture missing (capture off, a cell dir lost) "
    "also degrades to the 5-tuple and passes. Flips RED when it fails the guard."))
def test_a_replayed_guard_unit_without_its_capture_never_passes_on_the_5_tuple(tmp_path, monkeypatch):
    import shutil

    sp, cells = _guard_fixture(tmp_path, monkeypatch, flip=1)
    for u in cells["anchor"]["guard"]["units"]:
        shutil.rmtree(Path(u["cell_dir"]) / "capture")
    doc = RU.inputs("d1", cells, sp.params, sp)
    assert doc["import_off"]["guard"]["passed"] is False
