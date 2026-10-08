"""Does every reader of an eval-trace ``*_summary.json`` go through THE LOADER? The static gate.

**The bug class (F-LH-5, M5 Lane H).** A Rust-eval trace (``gen3_core_trace_v1``, written by
``agents.training.rust_eval.traces``) stores ``meta`` ONLY in its ``*_summary.json`` — no ``teams``, no
``invocations``; the prober's ``main.prober.core_trace`` rebuilds them on read. A module that does
``json.load(open(<prefix>_summary.json))`` itself and then walks ``summary.get("invocations", [])``
sees ZERO decisions on such a run and reports the empty read as a result. Nothing raises.

**The rule.** Outside the loader module (:data:`LOADER_MODULE`), no module under ``src/`` opens a
``*_summary.json`` for reading. A reader calls ``main.prober.core_trace.load_summary`` (the full,
EXPANDED summary), ``load_summary_meta`` (the stored ``meta`` block — sound on either trace kind), or
``refuse_core_trace`` (a reader that cannot be sound on a core trace refuses it by name).
**The allowlist is EMPTY** — a new direct reader is fixed at the source, never listed.

**What the scan sees** (one AST parse per module, ~1 s). A path expression CARRIES a summary path when
it holds a string constant ending in ``_summary.json`` (a literal, an f-string's tail, a glob
pattern), a name assigned from one (to a fixpoint, through ``os.path.join`` / ``Path`` / ``str`` /
``glob`` / ``sorted`` / ``list`` / a ``for`` or comprehension target), a parameter named in
:data:`SUMMARY_PARAMS`, or an attribute named in :data:`SUMMARY_ATTRS`. Stripping the suffix (a slice,
``.replace``, ``len("_summary.json")``) or appending another suffix ends the carry — the sibling
``_states.npz`` / ``_reconstruction.json`` is not the summary. A READ is ``open(e)`` /
``gzip.open(e)`` in a non-write mode, or ``e.read_text()`` / ``e.read_bytes()`` / ``e.open()``, on a
carrying ``e``.

**Blind spots, stated:** a path passed through an unrelated-looking parameter name or built in
another module and handed in. That is why the parameter-name list exists; widen it rather than
narrow the rule. Pool / anchor ``summary.json`` files (no ``_`` prefix) are a different file and are
not matched.

Opt out explicitly (never silently): ``GEN3AI_SKIP_SUMMARY_READER_GATE=1``.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import pytest

from utils.paths import src_root

pytestmark = pytest.mark.static   # the `static` budget tier (conftest._STATIC_BUDGET_BASE_S)

#: The ONE module allowed to open a trace summary: it IS the loader.
LOADER_MODULE = "main/prober/core_trace.py"
#: EMPTY by rule. A direct reader is migrated or refuses; it is never listed here.
ALLOWLIST: Tuple[str, ...] = ()
SUFFIX = "_summary.json"
SUMMARY_PARAMS = frozenset({"summary_path", "summ_path", "summary_file", "summary_paths", "spath", "smf",
                            "summ_file"})
SUMMARY_ATTRS = frozenset({"summary_path"})
#: Calls through which a carried path stays a summary path.
_PASS_THROUGH = frozenset({"join", "Path", "PurePath", "str", "abspath", "realpath", "expanduser", "normpath",
                           "fspath", "glob", "iglob", "rglob", "sorted", "list", "tuple", "set", "resolve",
                           "absolute", "reversed", "filter"})
_OPENERS = frozenset({"open"})
_PATH_READERS = frozenset({"read_text", "read_bytes", "open"})


def _str_carries(s: str) -> bool:
    return s.endswith(SUFFIX)


def _call_name(fn: ast.expr) -> str:
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return ""


class _Carry:
    def __init__(self, tainted: Set[str], returns: Optional[Set[str]] = None):
        self.tainted = tainted
        self.returns = returns if returns is not None else set()

    def __call__(self, e: Optional[ast.expr]) -> bool:  # noqa: C901 — one dispatch over node kinds
        if e is None:
            return False
        if isinstance(e, ast.Constant):
            return isinstance(e.value, str) and _str_carries(e.value)
        if isinstance(e, ast.JoinedStr):
            tail = e.values[-1] if e.values else None
            return isinstance(tail, ast.Constant) and isinstance(tail.value, str) and _str_carries(tail.value)
        if isinstance(e, ast.Name):
            return e.id in self.tainted
        if isinstance(e, ast.Attribute):
            return e.attr in SUMMARY_ATTRS
        if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Div)):
            if isinstance(e.right, (ast.Constant, ast.JoinedStr)):
                return self(e.right)                     # the LAST suffix decides
            return self(e.right) or self(e.left)
        if isinstance(e, ast.Call):
            name = _call_name(e.func)
            if name in self.returns:                     # a same-module function RETURNING a summary path
                return True
            if name in ("values", "items") and isinstance(e.func, ast.Attribute):
                return self(e.func.value)                # a dict of summary paths
            if name in _PASS_THROUGH:
                args = list(e.args)
                if name == "join" and args:
                    return self(args[-1]) or (len(args) == 1 and self(args[0]))
                recv = e.func.value if isinstance(e.func, ast.Attribute) else None
                if name in ("glob", "rglob", "resolve", "absolute") and recv is not None:
                    return any(self(a) for a in args) or (name in ("resolve", "absolute") and self(recv))
                return any(self(a) for a in args)
            return False
        if isinstance(e, ast.Subscript):
            return not isinstance(e.slice, ast.Slice) and self(e.value)
        if isinstance(e, (ast.List, ast.Tuple, ast.Set)):
            return any(self(x) for x in e.elts)
        if isinstance(e, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            return self(e.elt)
        if isinstance(e, ast.DictComp):
            return self(e.value)
        if isinstance(e, ast.Dict):
            return any(self(v) for v in e.values if v is not None)
        if isinstance(e, ast.IfExp):
            return self(e.body) or self(e.orelse)
        if isinstance(e, ast.BoolOp):
            return any(self(v) for v in e.values)
        if isinstance(e, ast.NamedExpr):
            return self(e.value)
        return False


def _targets(t: ast.expr) -> Iterator[str]:
    if isinstance(t, ast.Name):
        yield t.id
    elif isinstance(t, (ast.Tuple, ast.List)):
        for x in t.elts:
            yield from _targets(x)
    elif isinstance(t, ast.Starred):
        yield from _targets(t.value)


def _scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Every node of ``scope`` without descending into nested function / class bodies."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(n))


def _taint(scope: ast.AST, inherited: Set[str], params: Set[str], returns: Set[str]) -> Set[str]:
    tainted = set(inherited) | set(params)
    nodes = list(_scope_nodes(scope))
    for n in [scope, *nodes]:       # a summary-ish NAME is a summary path wherever it is bound
        if isinstance(n, ast.arg) and n.arg in SUMMARY_PARAMS:
            tainted.add(n.arg)
        elif isinstance(n, ast.Name) and n.id in SUMMARY_PARAMS:
            tainted.add(n.id)
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = scope.args
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
            if arg is not None and arg.arg in SUMMARY_PARAMS:
                tainted.add(arg.arg)
    carry = _Carry(tainted, returns)
    changed = True
    while changed:
        changed = False
        for n in nodes:
            pairs: List[Tuple[ast.expr, Optional[ast.expr]]] = []
            if isinstance(n, ast.Assign):
                pairs = [(t, n.value) for t in n.targets]
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and n.value is not None:
                pairs = [(n.target, n.value)]
            elif isinstance(n, (ast.For, ast.AsyncFor)):
                pairs = [(n.target, n.iter)]
            elif isinstance(n, ast.comprehension):
                pairs = [(n.target, n.iter)]
            elif isinstance(n, ast.NamedExpr):
                pairs = [(n.target, n.value)]
            elif isinstance(n, ast.withitem) and n.optional_vars is not None:
                pairs = [(n.optional_vars, n.context_expr)] if _call_name(getattr(n.context_expr, "func", None) or ast.Name("")) not in _OPENERS else []
            for tgt, val in pairs:
                if carry(val):
                    for name in _targets(tgt):
                        if name not in tainted:
                            tainted.add(name)
                            changed = True
    return tainted


def _is_write_mode(call: ast.Call, pos: int) -> bool:
    mode: Optional[ast.expr] = call.args[pos] if len(call.args) > pos else None
    for kw in call.keywords:
        if kw.arg == "mode":
            mode = kw.value
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and any(c in mode.value for c in "wax")


def _reads(scope: ast.AST, tainted: Set[str], returns: Set[str]) -> Iterator[int]:
    carry = _Carry(tainted, returns)
    for n in _scope_nodes(scope):
        if not isinstance(n, ast.Call):
            continue
        name = _call_name(n.func)
        if name in _OPENERS and n.args and not isinstance(n.func, ast.Attribute) or (
                name == "open" and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                and n.func.value.id in ("gzip", "io", "builtins", "codecs") and n.args):
            if carry(n.args[0]) and not _is_write_mode(n, 1):
                yield n.lineno
        elif isinstance(n.func, ast.Attribute) and name in _PATH_READERS and not (
                isinstance(n.func.value, ast.Name) and n.func.value.id in ("gzip", "io", "builtins", "codecs")):
            if carry(n.func.value) and not (name == "open" and _is_write_mode(n, 0)):
                yield n.lineno


def scan_source(src: str, filename: str = "<src>") -> List[int]:
    """Line numbers of every direct ``*_summary.json`` READ in ``src``."""
    tree = ast.parse(src, filename)
    defs: Dict[str, List[ast.AST]] = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.setdefault(n.name, []).append(n)
    #: INTERPROCEDURAL, within the module: a carried path handed to a function defined HERE (by
    #: position or keyword) taints that parameter — ``_load_json(spath)`` is a summary read.
    param_taint: Dict[int, Set[str]] = {}
    #: functions defined HERE whose return value carries a summary path (``names()`` → a dict of them).
    returns: Set[str] = set()
    hits: List[int] = []

    def visit(scope: ast.AST, inherited: Set[str]) -> bool:
        grew = False
        tainted = _taint(scope, inherited, param_taint.get(id(scope), set()), returns)
        hits.extend(_reads(scope, tainted, returns))
        carry = _Carry(tainted, returns)
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) and scope.name not in returns:
            if any(isinstance(n, ast.Return) and carry(n.value) for n in _scope_nodes(scope)):
                returns.add(scope.name)
                grew = True
        for n in _scope_nodes(scope):
            if isinstance(n, ast.Call) and _call_name(n.func) in defs:
                for fd in defs[_call_name(n.func)]:
                    a = fd.args  # type: ignore[attr-defined]
                    pos = [x.arg for x in [*a.posonlyargs, *a.args]]
                    if pos and pos[0] in ("self", "cls") and isinstance(n.func, ast.Attribute):
                        pos = pos[1:]
                    got = {pos[i] for i, arg in enumerate(n.args) if i < len(pos) and carry(arg)}
                    got |= {kw.arg for kw in n.keywords if kw.arg and carry(kw.value)}
                    have = param_taint.setdefault(id(fd), set())
                    if not got <= have:
                        have |= got
                        grew = True
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                grew = visit(n, tainted if not isinstance(n, ast.ClassDef) else set()) or grew
        return grew

    while True:
        hits.clear()
        if not visit(tree, set()):
            break
    return sorted(set(hits))


def scan_tree(root: Path) -> List[str]:
    out = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel.endswith("_test.py") or rel == LOADER_MODULE or "/target/" in rel or rel.startswith("rust_sim/target"):
            continue
        try:
            src = p.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if SUFFIX not in src and not any(s in src for s in SUMMARY_PARAMS | SUMMARY_ATTRS):
            continue
        out.extend(f"{rel}:{ln}" for ln in scan_source(src, rel))
    return out


@pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_SUMMARY_READER_GATE") == "1",
                    reason="GEN3AI_SKIP_SUMMARY_READER_GATE=1 (explicit opt-out)")
def test_no_trace_summary_reader_bypasses_the_loader():
    hits = [h for h in scan_tree(src_root()) if h.split(":")[0] not in ALLOWLIST]
    assert not hits, (
        f"{len(hits)} direct *_summary.json read(s) bypass the loader — on a Rust-eval core trace "
        "(gen3_core_trace_v1) they see meta only and read ZERO decisions, silently. Use "
        "main.prober.core_trace.load_summary (expanded), load_summary_meta (meta only) or "
        "refuse_core_trace (a reader that cannot be sound on a core trace). The allowlist is EMPTY:\n  "
        + "\n  ".join(hits))


def test_the_allowlist_is_empty():
    assert ALLOWLIST == ()


def test_the_loader_module_exists_and_exports_the_three_entry_points():
    from main.prober import core_trace

    assert (src_root() / LOADER_MODULE).exists()
    for name in ("load_summary", "load_summary_meta", "refuse_core_trace"):
        assert callable(getattr(core_trace, name)), name


@pytest.mark.parametrize("src", [
    'import json\ndef f(p):\n    with open(p + "_summary.json") as fh:\n        return json.load(fh)\n',
    'import json\ndef f(prefix):\n    return json.load(open(f"{prefix}_summary.json"))\n',
    'import glob, json\ndef f(d):\n    for sp in sorted(glob.glob(d + "/*_summary.json")):\n        json.load(open(sp))\n',
    'import json, os\ndef f(d, b):\n    spath = os.path.join(d, b + "_summary.json")\n    with open(spath) as fh:\n        return json.load(fh)\n',
    'import json\ndef f(summary_path):\n    return json.load(open(summary_path))\n',
    'import json\ndef f(b):\n    return json.load(open(b.summary_path))\n',
    'from pathlib import Path\nimport json\ndef f(d):\n    return [json.loads(p.read_text()) for p in Path(d).rglob("*_summary.json")]\n',
    'import json, glob\nfiles = glob.glob("x/**/*_summary.json")\nfor f in files:\n    doc = json.load(open(f))\n',
    'import json, os\ndef _load(path):\n    with open(path) as fh:\n        return json.load(fh)\ndef g(d, b):\n    return _load(os.path.join(d, b + "_summary.json"))\n',
    'import json, sys\nsumm_path = sys.argv[1]\nsumm = json.load(open(summ_path))\n',
    # a nested helper RETURNING a dict of summary paths (the shape of the deleted Lane H gate's compare_traces)
    'import json\ndef f(a, b):\n    def names(r):\n        return {str(p): p for p in r.rglob("*_summary.json")}\n'
    '    pn = names(b)\n    for rel in pn:\n        json.loads(pn[rel].read_text())\n',
    'import json, glob\ndef paths(d):\n    return sorted(glob.glob(d + "/*_summary.json"))\n'
    'def g(d):\n    for p in paths(d):\n        json.load(open(p))\n',
])
def test_the_scan_sees_every_reader_shape(src):
    assert scan_source(src), src


@pytest.mark.parametrize("src", [
    # the sibling files: the suffix is STRIPPED or REPLACED
    'import numpy as np\ndef f(sp):\n    return np.load(open(sp[: -len("_summary.json")] + "_states.npz", "rb"))\n',
    'def f(b):\n    return open(b.summary_path[: -len("_summary.json")] + "_reconstruction.json").read()\n',
    'def f(smf):\n    return open(smf.replace("_summary.json", "_states.npz"), "rb")\n',
    'import glob\ndef f(d):\n    for sp in glob.glob(d + "/*_summary.json"):\n        base = sp[: -len("_summary.json")]\n        open(base + "_reconstruction.json")\n',
    # a WRITER
    'import json\ndef f(p, s):\n    with open(f"{p}_summary.json", "w") as fh:\n        json.dump(s, fh)\n',
    # the loader
    'from main.prober.core_trace import load_summary\ndef f(summary_path):\n    return load_summary(summary_path)\n',
    # a pool / anchor summary.json is another file
    'import json, os\ndef f(d):\n    return json.load(open(os.path.join(d, "summary.json")))\n',
])
def test_the_scan_does_not_flag_siblings_writers_or_the_loader(src):
    assert scan_source(src) == [], src
