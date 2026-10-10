"""The launcher's periodic-restart default is 6 hours (owner rule 2026-10-04, in code 2026-10-10).

The code default stayed `3.0` for six days after the owner's rule, so every launch had to TYPE
`--restart-interval-hours 6` (and one that did not restarted twice as often, each restart paying the
compile warm-up and a resume seam). The rule is a number in ONE place — `run.DEFAULT_RESTART_INTERVAL_HOURS`
— and this file pins the number itself (a literal, so editing the constant alone cannot move the pin),
the parser's reading of an absent flag, that a typed value still wins, and the docs that state it.
"""

import re

from main.launcher.run import DEFAULT_RESTART_INTERVAL_HOURS, build_launcher_parser

from utils.paths import repo_path


def test_the_constant_is_six_hours():
    assert DEFAULT_RESTART_INTERVAL_HOURS == 6.0


def test_an_absent_flag_parses_to_six_hours():
    known, _ = build_launcher_parser().parse_known_args(["--steps", "100"])
    assert known.restart_interval_hours == 6.0


def test_a_typed_value_wins_and_zero_still_means_no_restart():
    parser = build_launcher_parser()
    assert parser.parse_known_args(["--restart-interval-hours", "3"])[0].restart_interval_hours == 3.0
    assert parser.parse_known_args(["--restart-interval-hours=0"])[0].restart_interval_hours == 0.0


def test_the_flag_table_and_the_census_state_the_same_default():
    """The two docs that carry the default as a number say what the parser says."""
    leaf = repo_path("src", "main", "launcher", "CLAUDE.md").read_text(encoding="utf-8")
    row = next(ln for ln in leaf.splitlines() if ln.startswith("| `--restart-interval-hours` |"))
    assert row.split("|")[2].strip() == "`6.0`", row
    census = repo_path("designs", "ops", "flag_census.md").read_text(encoding="utf-8")
    crow = next(ln for ln in census.splitlines() if ln.startswith("| `--restart-interval-hours` |"))
    assert re.match(r"\| `--restart-interval-hours` \| 6\.0 \|", crow), crow
