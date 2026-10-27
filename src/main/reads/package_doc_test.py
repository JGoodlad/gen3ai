"""The package docstring (``main/reads/__init__.py``) is the map a new reader opens first. It once listed eight modules
and none of ``lanectl``, ``supervisors``, ``playpath``, ``repin``, ``anchor_pool``, ``imports``, ``layout``, ``speccheck``
nor the verbs ``hold``, ``pid``, ``repin``, ``check-spec`` (backlog row, ``main.reads`` open item 3). This holds it to the
code: every module and every operator verb is named, so a module or a verb added without a line there fails here."""
from __future__ import annotations

import pkgutil
import re

import main.reads as pkg
from main.reads import cli

#: A verb that is the S lane's own child process, not an operator verb; the docstring says so.
INTERNAL_VERBS = {"s-unit"}


def _verbs() -> set:
    sub = next(a for a in cli._parser()._actions if getattr(a, "choices", None) and hasattr(a, "_name_parser_map"))
    return set(sub.choices)


def test_the_package_docstring_names_every_module():
    doc = pkg.__doc__ or ""
    mods = sorted(m.name for m in pkgutil.iter_modules(pkg.__path__)
                  if not m.name.endswith("_test") and m.name not in ("conftest", "__main__"))
    assert len(mods) >= 20, mods
    # a whole name: `main.reads.rules` is not met by `main.reads.rules_x59s`
    assert [m for m in mods if not re.search(rf"main\.reads\.{re.escape(m)}(?![A-Za-z0-9_])", doc)] == []


def test_the_package_docstring_names_every_operator_verb_and_the_internal_one_by_name():
    doc = pkg.__doc__ or ""
    verbs = _verbs()
    assert {"check-spec", "plan", "run", "resume", "hold", "pid", "status", "repin", "assemble", "decide", "record",
            "check-record", "search-cost"} <= verbs, verbs
    assert [v for v in sorted(verbs - INTERNAL_VERBS) if v not in doc] == []
    assert all(v in doc for v in INTERNAL_VERBS), "the internal verb is named so a reader does not take it for a gap"
