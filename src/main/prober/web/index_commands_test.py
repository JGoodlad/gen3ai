"""The run picker's example commands (`templates/index.html`) must still parse against the tools they name.

The page prints `python -m main.<meter> <run> --flag ...` rows for a run's meters. The freshness gate reads only
`CLAUDE.md` files, so nothing else notices when a meter is renamed or loses a flag and the page keeps teaching a
command that fails. Each command is resolved here against its real parser: the module's own `main` is run with
`ArgumentParser.parse_args` / `parse_known_args` intercepted, so the REAL parser (not a copy) judges the argv and
nothing else executes.
"""
from __future__ import annotations

import argparse
import html
import importlib
import inspect
import re
import shlex
import sys
from pathlib import Path

import pytest

_TEMPLATE = Path(__file__).parent / "templates" / "index.html"
_CELL_RE = re.compile(r'<td class="mono">\s*(python -m [^<]*?)\s*</td>')
_JINJA_RE = re.compile(r"\{\{.*?\}\}")


class _Captured(Exception):
    pass


def printed_commands() -> list:
    """Every `python -m ...` command the template prints, with the run and the `<placeholders>` filled in."""
    out = []
    for raw in _CELL_RE.findall(_TEMPLATE.read_text()):
        text = html.unescape(_JINJA_RE.sub("models/rb_example_run", raw))   # `{{ run or "<run>" }}` -> a path
        text = re.sub(r"<[^>]*>", "x", text)                                # `<ref>`, `<refs…>`: one argument
        out.append(text)
    return out


def resolve_against_its_parser(command: str, monkeypatch) -> argparse.ArgumentParser:
    """Parse `command` with the named module's real parser; raise if the module or a flag is gone."""
    tokens = shlex.split(command)
    assert tokens[:2] == ["python", "-m"], command
    module, argv = tokens[2], tokens[3:]
    mod = importlib.import_module(module)
    seen = []

    def spy(self, args=None, namespace=None):
        seen.append((self, list(args) if args is not None else None))
        raise _Captured

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", spy)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_known_args", spy)
    monkeypatch.setattr(sys, "argv", [module] + argv)
    try:
        if inspect.signature(mod.main).parameters:
            mod.main(argv)
        else:
            mod.main()
    except _Captured:
        pass
    assert seen, f"{module}.main() never reached ArgumentParser.parse_args, so `{command}` cannot be checked"
    parser, given = seen[0]
    monkeypatch.undo()
    try:
        parser.parse_args(argv if given is None else given)
    except SystemExit as exc:   # argparse reports an unknown flag / missing argument by exiting
        pytest.fail(f"`{command}` does not parse against {module}'s parser (exit {exc.code})")
    return parser


def test_the_page_prints_the_meter_commands_this_test_covers():
    """A new command on the page is covered automatically; one that vanished from the table fails here."""
    modules = {c.split()[2] for c in printed_commands()}
    assert {"main.critic_gate", "main.scaffolding_gauge", "main.untaught_meter", "main.elo"} <= modules, modules


def test_every_printed_command_parses_against_its_parser(monkeypatch):
    commands = printed_commands()
    assert len(commands) >= 4, commands
    for command in commands:
        resolve_against_its_parser(command, monkeypatch)
        monkeypatch.undo()


def test_meta_a_removed_flag_and_a_removed_module_are_caught(monkeypatch):
    """Revert-proof: the check FAILS on a flag the parser does not have and on a module that does not exist."""
    resolve_against_its_parser("python -m main.elo models/rb_x --no-plot", monkeypatch)
    monkeypatch.undo()
    with pytest.raises(pytest.fail.Exception):
        resolve_against_its_parser("python -m main.elo models/rb_x --no-such-flag-anywhere", monkeypatch)
    monkeypatch.undo()
    with pytest.raises(ImportError):
        resolve_against_its_parser("python -m main.gone_meter models/rb_x", monkeypatch)
