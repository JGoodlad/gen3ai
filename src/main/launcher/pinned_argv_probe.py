"""The subprocess half of ``pinned_argv`` — runs INSIDE a materialised pinned checkout.

This file is COPIED into the temp directory that holds the pinned ``src/`` and executed as a
plain script (``python <tmp>/_pinned_argv_probe.py <tmp>/src <argv.json> <out.json>``). It is
kept as a real module rather than an embedded string so that ruff, mypy and the file-size gate
can see it, and so it can be unit-tested directly.

It answers ONE question: *would the argv the launcher is about to forward parse against the
parser of the commit that will actually run it?* Two modes, in order:

* **build_parser** — ``main.train_rl_agent.build_parser()``. Authoritative: it IS the parser the
  child constructs. Available from 26b28509 (2026-08-16) onward.
* **ast_scan** — a best-effort STATIC read of every ``…add_argument(…)`` call in the pinned
  ``train_rl_agent.py`` (+ ``main/train/parser/*.py`` when that package exists), replayed into a
  synthetic ``argparse`` parser that carries only each option's SPELLING and ARITY. Older commits
  build their parser inline inside ``main()``, which cannot be called without starting a training
  job, so this is the only cheap way to ask them anything. Because the reconstruction can be
  incomplete, the caller treats its findings as a WARNING, never a refusal.

Arity is the whole point. A flag whose NAME survived but whose arity changed is invisible to any
presence check — ``--hp-type-belief`` took a value at b13b30b2 and is gone today, so the current
parser abbreviation-matches it onto ``--hp-type-belief-coef`` and dies on
``invalid float value: 'learned'``. Replaying the argv through a parser that knows the pinned
arity is what catches that.

The result is written as JSON so nothing about the pinned tree (objects, exceptions, torch) ever
crosses back into the launcher's process.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import glob
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

#: ``add_argument(action=…)`` values that consume NO following token. Everything else consumes
#: one unless ``nargs`` says otherwise. Only arity matters here — the synthetic parser never
#: needs to produce a correct VALUE, only to consume the right number of tokens.
ZERO_ARG_ACTIONS = frozenset({
    "store_true", "store_false", "store_const", "count", "help", "version", "append_const",
})


def _load_build_parser(src_dir: str) -> "Tuple[Optional[argparse.ArgumentParser], str]":
    """``(parser, why_not)`` for the pinned tree's own ``build_parser()``."""
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        import main.train_rl_agent as entry
    except BaseException as e:                      # noqa: BLE001 — a pinned tree may fail any way
        return None, (f"the pinned tree's main.train_rl_agent will not import here "
                      f"({type(e).__name__}: {e})")
    build = getattr(entry, "build_parser", None)
    if build is None:
        return None, ("this commit predates build_parser() — its parser is built inline inside "
                      "main(), which cannot be called without starting a training job")
    try:
        return build(), ""
    except BaseException as e:                      # noqa: BLE001
        return None, f"the pinned build_parser() raised {type(e).__name__}: {e}"


def _spec_from_call(node: ast.Call) -> "Optional[Tuple[List[str], Optional[str], Any]]":
    """``(option_strings, action, nargs)`` from one ``add_argument(...)`` call, or None."""
    opts = [a.value for a in node.args
            if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("-")]
    if not opts:
        return None
    action: Optional[str] = None
    nargs: Any = None
    for kw in node.keywords:
        if kw.arg == "action" and isinstance(kw.value, ast.Constant):
            action = kw.value.value
        elif kw.arg == "nargs" and isinstance(kw.value, ast.Constant):
            nargs = kw.value.value
    return opts, action, nargs


def parser_files(src_dir: str) -> List[str]:
    """The files that can declare a trainer flag, in any generation of this tree."""
    files = [os.path.join(src_dir, "main", "train_rl_agent.py")]
    files += sorted(glob.glob(os.path.join(src_dir, "main", "train", "parser", "*.py")))
    return [p for p in files if os.path.exists(p)]


def synthesise_parser(src_dir: str) -> "Tuple[Optional[argparse.ArgumentParser], int]":
    """A parser carrying every statically-readable option's SPELLING and ARITY, and its count.

    Deliberately permissive: no ``choices``, no ``type``, no ``required`` — a value we cannot
    reconstruct must never turn into a refusal. Duplicate option strings (the same flag declared
    in two files across a refactor) are skipped rather than raised on.
    """
    parser = argparse.ArgumentParser(add_help=False)
    n = 0
    for path in parser_files(src_dir):
        try:
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                continue
            spec = _spec_from_call(node)
            if spec is None:
                continue
            opts, action, nargs = spec
            opts = [o for o in opts if o not in ("-h", "--help")]
            if not opts:
                continue
            kw: Dict[str, Any] = {}
            if action in ZERO_ARG_ACTIONS:
                kw["action"] = "store_true"
            elif nargs is not None:
                kw["nargs"] = nargs
            try:
                parser.add_argument(*opts, **kw)
            except (argparse.ArgumentError, ValueError, TypeError):
                continue
            n += 1
    return (parser, n) if n else (None, 0)


def _error_lines(captured: str) -> List[str]:
    """The `…: error: …` line(s) out of argparse's exit output, never the 200-line usage block.

    argparse prints the whole usage before its one-line diagnosis, and that block is what names
    every flag — dumping it into a launcher's event stream would bury the token that actually
    failed under the entire flag table.
    """
    lines = [ln.strip() for ln in captured.splitlines() if ln.strip()]
    errors = [ln for ln in lines if ": error:" in ln]
    return errors or lines[-1:]


def parse_with(parser: argparse.ArgumentParser, argv: List[str]) -> Dict[str, Any]:
    """Replay ``argv`` through ``parser``; report what did not fit.

    ``unknown`` = leftover tokens that look like flags. ``stray`` = leftover tokens that do NOT —
    the signature of an ARITY change, where a value the pinned parser would have consumed is left
    dangling (or, read the other way, a value the current parser would swallow into the wrong
    flag). Neither parser here has positionals, so a stray token is always a defect.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            ns, rest = parser.parse_known_args(argv)
    except SystemExit:
        return {"unknown": [], "stray": [], "errors": _error_lines(buf.getvalue()), "values": {}}
    return {
        "unknown": [t for t in rest if t.startswith("-")],
        "stray": [t for t in rest if not t.startswith("-")],
        "errors": [],
        "values": {k: repr(v) for k, v in sorted(vars(ns).items())},
    }


def main(raw: "Optional[List[str]]" = None) -> int:
    src_dir, argv_path, out_path = (raw if raw is not None else sys.argv[1:])[:3]
    with open(argv_path) as f:
        child_argv = json.load(f)

    res: Dict[str, Any] = {"mode": "unavailable", "reason": "", "n_options": 0,
                           "unknown": [], "stray": [], "errors": [], "values": {}}
    parser, why = _load_build_parser(src_dir)
    if parser is not None:
        res["mode"] = "build_parser"
        res["n_options"] = sum(len(a.option_strings) for a in parser._actions)
    else:
        parser, n = synthesise_parser(src_dir)
        if parser is None:
            res["reason"] = (f"{why}; and no add_argument() call could be read statically from "
                             f"{src_dir}")
            with open(out_path, "w") as f:
                json.dump(res, f)
            return 0
        res["mode"] = "ast_scan"
        res["reason"] = why
        res["n_options"] = n
    # argparse names the PROG in its error line; left alone that is this probe's filename, which
    # would read as though the launcher's own tooling rejected the flag.
    parser.prog = "train_rl_agent.py"
    res.update(parse_with(parser, child_argv))
    with open(out_path, "w") as f:
        json.dump(res, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
