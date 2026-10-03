"""The FLAG CENSUS gate: every trainer / launcher flag names a LIVE user, and the census says who.

`designs/ops/flag_census.md` (deletion pass P11) has one row per flag of `build_parser()` (the trainer)
and `build_launcher_parser()` (the launcher), each with a verdict and the live user that justifies it.
This test is what keeps that true after the pass: a NEW flag arrives with a row that names its live user,
or the gate fails naming it; a row whose flag no parser declares any more fails too (the flag was deleted
and its row left behind, or the row was mistyped). It judges NOTHING about whether the named user is
real — that is review's job — only that one was named, in the shape the census defines.
"""
from __future__ import annotations

import re
from collections import Counter

from main.launcher.run import build_launcher_parser
from main.train.parser import build_parser
from utils.paths import repo_path

VERDICTS = ("KEEP", "DELETE", "ONE-VALUED", "NEEDS-OWNER")
# | `--flag` | default | typed | live user | **VERDICT** |
_ROW = re.compile(r"^\| `(--[a-z0-9][a-z0-9-]*)` \|(?P<mid>.*)\| \*\*(?P<verdict>[A-Z-]+)\*\* \|$")


def _census_rows():
    text = repo_path("designs", "ops", "flag_census.md").read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        m = _ROW.match(line)
        if m:
            cells = [c.strip() for c in m.group("mid").split("|")]
            rows.append((m.group(1), m.group("verdict"), cells))
    return rows


def _parser_flags() -> dict:
    """primary `--flag` -> (family, default), for the trainer and the launcher."""
    out = {}
    for fam, parser in (("trainer", build_parser()), ("launcher", build_launcher_parser())):
        for a in parser._actions:
            primary = next((o for o in a.option_strings if o.startswith("--")), None)
            if primary is None or primary == "--help" or a.dest == "help":
                continue
            out[primary] = (fam, a.default)
    return out


def test_every_parser_flag_has_a_census_row():
    have = {f for f, _, _ in _census_rows()}
    missing = {f: v for f, v in _parser_flags().items() if f not in have}
    assert not missing, (
        "flags with NO row in designs/ops/flag_census.md (add one naming its LIVE user — a production "
        f"default / --arch production row, a documented operator knob, a registered experiment, a meter, "
        f"the launcher — or do not add the flag): {sorted(missing.items())}")


def test_every_census_row_names_a_parser_flag():
    live = set(_parser_flags())
    stale = sorted(f for f, _, _ in _census_rows() if f not in live)
    assert not stale, (
        f"census rows whose flag no parser declares (a deleted flag's row belongs in designs/deleted_flags.md, "
        f"not here): {stale}")


def test_every_row_has_a_valid_verdict_and_names_its_live_user():
    bad = []
    for flag, verdict, cells in _census_rows():
        user = cells[-1] if cells else ""
        if verdict not in VERDICTS:
            bad.append((flag, f"verdict {verdict!r}"))
        elif len(user) < 12 or user.lower() in ("none", "n/a", "tbd"):
            bad.append((flag, f"live-user cell {user!r}"))
    assert not bad, f"census rows that do not name a live user or carry an unknown verdict: {bad}"


def test_no_flag_has_two_rows():
    dup = [f for f, n in Counter(f for f, _, _ in _census_rows()).items() if n > 1]
    assert not dup, f"flags with more than one census row: {dup}"
