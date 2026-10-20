"""``main.best_response_gap``'s output location: the report JSON must never land in the current directory.

The old default was ``./best_response_gap.json``, so running the meter from the main checkout DIRTIED main
(round-2 registration K-1, 2026-09-24). Synthetic exploiter runs (the engine tests' ``make_run``), no battles."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from agents.training.best_response_gap_test import TEAMSETS, make_run
from main import best_response_gap as cli


@pytest.fixture
def argv(tmp_path, monkeypatch):
    """Two matched rounds of two archetypes each, a teamsets file, and a scratch cwd + temp dir."""
    work, scratch, tmp = tmp_path / "work", tmp_path / "cwd", tmp_path / "tmp"
    for d in (work, scratch, tmp):
        d.mkdir()
    teamsets = tmp_path / "teamsets.json"
    teamsets.write_text(json.dumps(TEAMSETS))
    runs = []
    for step, end, tag, cyc in ((75_000_000, 83_000_000, "gen1", {"offense": (74, 100), "stall": (72, 100)}),
                                (95_000_000, 103_000_000, "gen2", {"offense": (66, 100), "stall": (46, 100)})):
        for arch, (w, g) in cyc.items():
            runs.append(make_run(work, f"{tag}_{arch}", target_run=tag, target_step=step, fork_step=step,
                                 num_timesteps=end, cycles=[(end - 1_000_000, w, g)],
                                 team_stems=["aaaa1111" if arch == "offense" else "bbbb1111"]))
    monkeypatch.chdir(scratch)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    return [*runs, "--teamsets", str(teamsets), "--quiet", "--draws", "200"], scratch, tmp


def test_the_default_report_goes_to_the_temp_dir_never_the_cwd(argv, capsys):
    args, cwd, tmp = argv
    assert cli.main(args) == 0
    assert os.listdir(cwd) == [], "the report was written into the current directory"
    [written] = os.listdir(tmp)
    assert written.startswith("best_response_gap_") and written.endswith(".json")
    assert json.loads((tmp / written).read_text())["_meta"]["tool"] == "main.best_response_gap"


def test_an_explicit_json_path_is_honoured_and_the_temp_dir_stays_empty(argv, tmp_path):
    args, cwd, tmp = argv
    out = tmp_path / "chosen.json"
    assert cli.main([*args, "--json", str(out)]) == 0
    assert out.is_file() and os.listdir(cwd) == [] and os.listdir(tmp) == []


def test_no_json_writes_nothing_anywhere(argv):
    args, cwd, tmp = argv
    assert cli.main([*args, "--no-json"]) == 0
    assert os.listdir(cwd) == [] and os.listdir(tmp) == []


def test_the_default_path_is_in_the_temp_dir_and_unique_per_time():
    a, b = cli.default_json_path(1.0e9), cli.default_json_path(1.0e9 + 5)
    assert os.path.dirname(a) == tempfile.gettempdir() and a != b
