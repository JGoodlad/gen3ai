"""The subprocess half of ``pinned_argv`` — runs INSIDE a materialised pinned checkout.

This file is COPIED into the temp directory that holds the pinned ``src/`` and executed as a
plain script (``python <tmp>/_pinned_argv_probe.py <tmp>/src <argv.json> <out.json>``). It is
kept as a real module rather than an embedded string so that ruff, mypy and the file-size gate
can see it, and so it can be unit-tested directly.

It answers ONE question: *would the argv the launcher is about to forward parse against the
parser of the commit that will actually run it?* Three modes, in order — and the ORDER is the
design: two of them are the pinned tree's REAL parser answering, and only the last is a guess.

* **build_parser** — ``main.train_rl_agent.build_parser()``. Authoritative: it IS the parser the
  child constructs. Available from 26b28509 (2026-08-16) onward.
* **parse_args_hook** — for every commit BEFORE that, whose parser is built inline inside
  ``main()``. Run the pinned ``main/train_rl_agent.py`` as ``__main__`` in a child process with
  ``argparse.ArgumentParser.parse_args`` MONKEYPATCHED: the first call hands us the fully-built
  real parser, we replay the argv through it, write the report, and ``os._exit(0)`` — before the
  entry point has done one line of work. Also AUTHORITATIVE, because the parser answering is the
  parser the child builds; the only static thing about it is that we never let it run. Time-boxed
  (a pre-2026-08 tree imports torch on the way to its parser), and a timeout or an import failure
  falls back to:
* **ast_scan** — a best-effort STATIC read of every ``…add_argument(…)`` call in the pinned
  ``train_rl_agent.py`` (+ ``main/train/parser/*.py`` when that package exists), replayed into a
  synthetic ``argparse`` parser that carries only each option's SPELLING and ARITY. Because the
  reconstruction can be incomplete, the caller treats its findings as a WARNING, never a refusal.

Arity is the whole point. A flag whose NAME survived but whose arity changed is invisible to any
presence check — ``--hp-type-belief`` took a value at b13b30b2 and is gone today, so the current
parser abbreviation-matches it onto ``--hp-type-belief-coef`` and dies on
``invalid float value: 'learned'``. Replaying the argv through a parser that knows the pinned
arity is what catches that.

⚠️ **A CUSTOM ACTION CLASS IS NOT A STRING, and reading only string actions produced a FALSE
REFUSAL.** ``ast_scan`` used to read ``action=`` only when it was an ``ast.Constant``, so
``action=BoolFlag`` (38 flags at b13b30b2) was read as "no action ⇒ takes one value" and the real
v8 argv came back as ``argument --self-play: expected one argument`` — for a flag that is a bare
boolean there. A non-string action is now resolved from the pinned tree's OWN ``class`` body: the
``nargs`` it passes to ``super().__init__`` and whether it generates ``--no-`` negatives are both
read out of the AST, so the reconstruction follows the pinned commit rather than a class name
frozen into this file.

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
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

#: ``add_argument(action=…)`` values that consume NO following token. Everything else consumes
#: one unless ``nargs`` says otherwise. Only arity matters here — the synthetic parser never
#: needs to produce a correct VALUE, only to consume the right number of tokens.
ZERO_ARG_ACTIONS = frozenset({
    "store_true", "store_false", "store_const", "count", "help", "version", "append_const",
})

#: Action CLASSES whose shape we know without reading a class body: ``(nargs, makes --no-)``.
#: Only stdlib ones belong here — anything this tree defines is read from its own source below,
#: so the scan tracks the pinned commit instead of a name frozen into this file.
_BUILTIN_ACTION_SHAPES: Dict[str, Tuple[Any, bool]] = {
    "BooleanOptionalAction": (0, True),
}

#: What an UNREADABLE custom action is assumed to be. ``"?"`` (an OPTIONAL value) is the only
#: choice that cannot invent a failure: it accepts ``--flag`` bare AND ``--flag value``. A static
#: scan that guesses wrong must under-report, never refuse.
_UNKNOWN_ACTION_NARGS = "?"

#: The parse_args-hook child's own time box. A pre-2026-08 tree imports torch on the way to its
#: parser, and that import is the cost being bounded — past it we fall back to the static scan.
HOOK_TIMEOUT_S = 120.0

#: A parser small enough that it is probably NOT the trainer's — some dependency parsing its own
#: argv at import time. Answering with that parser would be worse than not answering at all.
MIN_HOOK_OPTIONS = 20

_HOOK_FLAG = "--hook"


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
                      "main()")
    try:
        return build(), ""
    except BaseException as e:                      # noqa: BLE001
        return None, f"the pinned build_parser() raised {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------------------------
# ast_scan — the fallback, and the one that must never invent a refusal
# ---------------------------------------------------------------------------------------------

def _action_name(node: ast.expr) -> Optional[str]:
    """The NAME behind a non-string ``action=`` — ``BoolFlag``, ``argparse.BooleanOptionalAction``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _spec_from_call(
    node: ast.Call,
) -> "Optional[Tuple[List[str], Optional[str], Optional[str], Any]]":
    """``(option_strings, action_string, action_class_name, nargs)`` from one ``add_argument``."""
    opts = [a.value for a in node.args
            if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("-")]
    if not opts:
        return None
    action: Optional[str] = None
    action_cls: Optional[str] = None
    nargs: Any = None
    for kw in node.keywords:
        if kw.arg == "action":
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                action = kw.value.value
            else:
                action_cls = _action_name(kw.value)
        elif kw.arg == "nargs" and isinstance(kw.value, ast.Constant):
            nargs = kw.value.value
    return opts, action, action_cls, nargs


def custom_action_shapes(files: List[str]) -> Dict[str, Tuple[Any, bool]]:
    """``class Foo(argparse.Action)`` → ``(the nargs it passes up, does it generate --no- forms?)``.

    Both facts are read from the PINNED tree's own source rather than assumed, because both change
    the answer for the v8 argv: ``BoolFlag`` passes ``nargs="?"`` to ``super().__init__`` (so
    ``--self-play`` is legal bare) and builds ``"--no-" + opt[2:]`` for every option string (so
    ``--no-stall-pbrs`` is a real flag at that commit, declared nowhere).
    """
    out: Dict[str, Tuple[Any, bool]] = dict(_BUILTIN_ACTION_SHAPES)
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            nargs: Any = None
            negates = False
            for sub in ast.walk(node):
                if (isinstance(sub, ast.keyword) and sub.arg == "nargs"
                        and isinstance(sub.value, ast.Constant)):
                    nargs = sub.value.value
                elif (isinstance(sub, ast.Constant) and isinstance(sub.value, str)
                        and sub.value.startswith("--no-")):
                    negates = True
            out[node.name] = (nargs, negates)
    return out


def parser_files(src_dir: str) -> List[str]:
    """The files that can declare a trainer flag, in any generation of this tree."""
    files = [os.path.join(src_dir, "main", "train_rl_agent.py")]
    files += sorted(glob.glob(os.path.join(src_dir, "main", "train", "parser", "*.py")))
    return [p for p in files if os.path.exists(p)]


def synthesise_parser(src_dir: str) -> "Tuple[Optional[argparse.ArgumentParser], int]":
    """A parser carrying every statically-readable option's SPELLING and ARITY, and its count.

    Deliberately permissive: no ``choices``, no ``type``, no ``required`` — a value we cannot
    reconstruct must never turn into a refusal. Duplicate option strings (the same flag declared
    in two files across a refactor) are skipped rather than raised on. Generated ``--no-`` forms
    are added LAST, so an explicitly-declared flag of the same name always wins.
    """
    files = parser_files(src_dir)
    shapes = custom_action_shapes(files)
    parser = argparse.ArgumentParser(add_help=False)
    n = 0
    negatives: List[str] = []
    for path in files:
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
            opts, action, action_cls, nargs = spec
            opts = [o for o in opts if o not in ("-h", "--help")]
            if not opts:
                continue
            if action_cls is not None:
                cls_nargs, makes_negatives = shapes.get(
                    action_cls, (_UNKNOWN_ACTION_NARGS, False))
                nargs = cls_nargs if nargs is None else nargs
                action = "store_true" if nargs == 0 else None
                if makes_negatives:
                    negatives += ["--no-" + o[2:] for o in opts if o.startswith("--")]
            kw: Dict[str, Any] = {}
            if action in ZERO_ARG_ACTIONS:
                kw["action"] = "store_true"
            elif nargs is not None and nargs != 0:
                kw["nargs"] = nargs
            try:
                parser.add_argument(*opts, **kw)
            except (argparse.ArgumentError, ValueError, TypeError):
                continue
            n += 1
    for neg in negatives:
        try:
            parser.add_argument(neg, nargs="?")
        except (argparse.ArgumentError, ValueError, TypeError):
            continue
    return (parser, n) if n else (None, 0)


# ---------------------------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------------------------

def option_strings(parser: argparse.ArgumentParser) -> List[str]:
    """Every spelling this parser accepts — THE PINNED OPTION SET (see ``pinned_argv``).

    This is what stops "the current parser accepts it" from being the verdict: a flag ADDED after
    the pin parses fine today and does not exist at the commit that will run it.
    """
    out: List[str] = []
    for action in parser._actions:                                  # noqa: SLF001 — the contract
        out += list(action.option_strings)
    return sorted(set(out))


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


def _write(out_path: str, res: Dict[str, Any]) -> None:
    with open(out_path, "w") as f:
        json.dump(res, f)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------------------------
# parse_args_hook — the pinned tree's REAL parser, without letting the entry point run
# ---------------------------------------------------------------------------------------------

def _run_hook(src_dir: str, child_argv: List[str], out_path: str) -> int:
    """Execute the pinned entry point as ``__main__``, answering at its first ``parse_args``.

    🚨 THE MONKEYPATCH IS INSTALLED BEFORE THE MODULE IS EXECUTED, and it ``os._exit``s from
    inside the call. Nothing after ``parse_args`` in the entry point runs — no run dir, no
    ``models/`` touch, no torch device grab, no atexit hook — which is what makes it safe to point
    at a real training entry point on a box carrying a live run. (At b13b30b2 the first ~1080
    lines of ``main()`` are ``add_argument`` calls and ``parse_args`` is the next statement, so
    "before any work" is not a hope but the shape of the file.)

    ``os._exit`` rather than ``sys.exit`` on purpose: a ``SystemExit`` can be caught, retried or
    wrapped by whatever the entry point had running (at b13b30b2 the call sits inside
    ``asyncio.run``), and "probably unwinds cleanly" is not a guarantee worth having here.
    """
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    entry_path = os.path.join(src_dir, "main", "train_rl_agent.py")
    if not os.path.exists(entry_path):
        _write(out_path, {"mode": "unavailable", "reason": f"no {entry_path} at this commit"})
        return 0

    real_parse_known = argparse.ArgumentParser.parse_known_args

    def _hooked(self, args=None, namespace=None):    # noqa: ANN001 — matches argparse's signature
        opts = option_strings(self)
        if len(opts) < MIN_HOOK_OPTIONS:
            # Not the trainer's parser — some dependency reading its own argv at import time.
            # Let it through rather than answering the launcher with the wrong parser.
            return real_parse_known(self, args, namespace)[0]
        res: Dict[str, Any] = {"mode": "parse_args_hook", "reason": "",
                               "n_options": len(opts), "options": opts}
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
                ns, rest = real_parse_known(self, list(child_argv), None)
            res.update(unknown=[t for t in rest if t.startswith("-")],
                       stray=[t for t in rest if not t.startswith("-")],
                       errors=[], values={k: repr(v) for k, v in sorted(vars(ns).items())})
        except SystemExit:
            res.update(unknown=[], stray=[], errors=_error_lines(buf.getvalue()), values={})
        except BaseException as e:                   # noqa: BLE001 — a custom action may raise
            res.update(unknown=[], stray=[],
                       errors=[f"train_rl_agent.py: error: {type(e).__name__}: {e}"], values={})
        _write(out_path, res)
        os._exit(0)                                  # noqa: SLF001 — see the docstring

    argparse.ArgumentParser.parse_args = _hooked     # type: ignore[method-assign]
    sys.argv = ["train_rl_agent.py", *child_argv]
    try:
        import runpy
        runpy.run_path(entry_path, run_name="__main__")
    except BaseException as e:                       # noqa: BLE001 — any failure ⇒ fall back
        _write(out_path, {"mode": "unavailable",
                          "reason": (f"running the pinned entry point reached no parse_args() "
                                     f"({type(e).__name__}: {e})")})
        return 0
    _write(out_path, {"mode": "unavailable",
                      "reason": "the pinned entry point returned without calling parse_args()"})
    return 0


def _try_hook(src_dir: str, argv_path: str, tmp_dir: str,
              timeout: float = HOOK_TIMEOUT_S) -> "Optional[Dict[str, Any]]":
    """Run the hook in a CHILD process; ``None``/``unavailable`` when it could not answer.

    A child rather than this process because the pinned tree's IMPORT is the risk being bounded:
    a hang, a hard crash, or an incompatibility with today's site-packages must degrade to the
    static scan, not take the probe down with it.
    """
    hook_out = os.path.join(tmp_dir, "hook_report.json")
    with contextlib.suppress(OSError):
        os.remove(hook_out)
    cmd = [sys.executable, os.path.abspath(__file__), _HOOK_FLAG, src_dir, argv_path, hook_out]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=tmp_dir)
    except subprocess.TimeoutExpired:
        return {"mode": "unavailable",
                "reason": (f"the pinned entry point did not reach parse_args() within "
                           f"{timeout:.0f}s")}
    except Exception as e:                           # noqa: BLE001
        return {"mode": "unavailable", "reason": f"the parse_args hook failed: {type(e).__name__}: {e}"}
    if not os.path.exists(hook_out):
        return None
    try:
        with open(hook_out) as f:
            return dict(json.load(f))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------

def main(raw: "Optional[List[str]]" = None) -> int:
    args = list(raw if raw is not None else sys.argv[1:])
    if args and args[0] == _HOOK_FLAG:
        src_dir, argv_path, out_path = args[1:4]
        with open(argv_path) as f:
            return _run_hook(src_dir, list(json.load(f)), out_path)

    src_dir, argv_path, out_path = args[:3]
    with open(argv_path) as f:
        child_argv = json.load(f)

    res: Dict[str, Any] = {"mode": "unavailable", "reason": "", "n_options": 0, "options": [],
                           "unknown": [], "stray": [], "errors": [], "values": {}}
    parser, why = _load_build_parser(src_dir)
    if parser is not None:
        res["mode"] = "build_parser"
        res["options"] = option_strings(parser)
        res["n_options"] = len(res["options"])
    else:
        # No build_parser() at this commit. Ask its REAL parser anyway, by running the entry point
        # only as far as its first parse_args() — authoritative, and time-boxed.
        hooked = _try_hook(src_dir, argv_path, os.path.dirname(os.path.abspath(out_path)))
        if hooked is not None and hooked.get("mode") == "parse_args_hook":
            hooked["reason"] = why
            _write(out_path, hooked)
            return 0
        hook_why = (hooked or {}).get("reason", "the parse_args hook produced no report")
        parser, n = synthesise_parser(src_dir)
        if parser is None:
            res["reason"] = (f"{why}; {hook_why}; and no add_argument() call could be read "
                             f"statically from {src_dir}")
            _write(out_path, res)
            return 0
        res["mode"] = "ast_scan"
        res["reason"] = f"{why}; {hook_why}"
        res["options"] = option_strings(parser)
        res["n_options"] = n
    # argparse names the PROG in its error line; left alone that is this probe's filename, which
    # would read as though the launcher's own tooling rejected the flag.
    parser.prog = "train_rl_agent.py"
    res.update(parse_with(parser, child_argv))
    _write(out_path, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
