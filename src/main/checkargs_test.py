"""Gates for `main.checkargs` — and for the parser being INSPECTABLE at all.

The load-bearing test here is `test_every_help_string_renders`. `--help` was broken on main by a
single unescaped `%` ("~0.6% of the static table" → argparse reads `% o` as a space-flag `%o`
conversion and raises `TypeError: %o format: an integer is required, not dict`). Nothing rendered
the help text, so nothing caught it — and with `--help` down there was no offline way to ask what
the parser accepts, which is exactly when you need one.
"""
from __future__ import annotations

import json

import pytest

from main.checkargs import (argv_from_run, check, known_option_strings,
                            split_argv)


# ------------------------------------------------------------------ the parser is inspectable

def test_every_help_string_renders():
    """THE regression. argparse formats help lazily, so a bad `%` only explodes when something
    actually renders it — which nothing did. Rendering the whole parser is the guard."""
    import argparse as _ap
    from main.train_rl_agent import build_parser
    parser = build_parser()
    fmt = _ap.HelpFormatter(prog="train_rl_agent.py")
    for action in parser._actions:
        if action.help:
            fmt._expand_help(action)          # raises TypeError/ValueError on a bad conversion
    assert parser.format_help()               # and the whole document composes


def test_build_parser_is_importable_without_running_main():
    """`build_parser()` must not need an event loop, a GPU, or argv — the whole point of pulling
    it out of `main()` is that a tool can ask the parser questions cheaply."""
    from main.train_rl_agent import build_parser
    assert len(build_parser()._actions) > 100


def test_known_option_strings_covers_a_live_flag():
    known = known_option_strings()
    assert "--steps" in known and known["--steps"] == "steps"
    assert "--value-clock" in known, "a v87 flag should be present on a v89 tree"


# ------------------------------------------------------------------ the checker itself

def test_clean_argv_reports_nothing():
    res = check(["--steps", "100", "--device", "cuda"])
    assert res["unknown"] == []
    assert set(res["accepted"]) == {"--steps", "--device"}


def test_deleted_flag_is_reported_with_its_value():
    """The motivating case: `--pubval-*` survived in gen-12's argv after v88 deleted it."""
    res = check(["--steps", "100", "--pubval-mode", "none", "--pubval-coef", "0.1"])
    assert [f for f, _ in res["unknown"]] == ["--pubval-mode", "--pubval-coef"]
    assert dict(res["unknown"])["--pubval-mode"] == ["none"]
def test_split_argv_attaches_values_to_their_flag():
    assert split_argv(["--a", "1", "2", "--b"]) == [("--a", ["1", "2"]), ("--b", [])]


def test_split_argv_tolerates_a_leading_positional():
    assert split_argv(["run_dir", "--a"])[0] == ("run_dir", [])


# ------------------------------------------------------------------ the run-dir front door

def test_argv_from_run_reads_the_recorded_command(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"launcher_command": "/path/to/__main__.py --steps 25000000 --device cuda"}))
    assert argv_from_run(str(tmp_path)) == ["--steps", "25000000", "--device", "cuda"]


def test_argv_from_run_falls_back_to_original_command(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"original_command": "/p/__main__.py --steps 1"}))
    assert argv_from_run(str(tmp_path)) == ["--steps", "1"]


def test_argv_from_run_fails_loud_when_nothing_is_recorded(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"git_hash": "abc"}))
    with pytest.raises(SystemExit):
        argv_from_run(str(tmp_path))


def test_missing_metadata_is_a_clean_message_not_a_traceback(tmp_path):
    """A run writes metadata.json at its first save, so a just-launched run legitimately has none.
    Asking about one is reasonable; a raw FileNotFoundError at the reader is not."""
    with pytest.raises(SystemExit) as e:
        argv_from_run(str(tmp_path))
    assert "no metadata.json yet" in str(e.value)


def test_missing_run_dir_says_so(tmp_path):
    with pytest.raises(SystemExit) as e:
        argv_from_run(str(tmp_path / "nope"))
    assert "no such run dir" in str(e.value)
