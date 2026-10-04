"""Does every reader of the eval COUNT ledger DECLARE what it reads? And do the ledger's closed lists equal the
design's tables? The static gate (design_evaluation.md §0b.4 item 6).

**The bug class.** The ledger's value is that every estimate is a DECLARED read: which purposes, which regime,
which requests (its own, its family, or any), whether the games that selected a node are excluded, which legacy
flags it tolerates, and whether its error bar is conditional or across runs (§0b.7, §0c). A module that globs the
shards itself, or calls ``read`` with a declaration assembled at run time, silently skips every one of those
rules — pooled regimes, a peeked request, a double-counted replay — and nothing raises.

**The rules** (outside the ledger's own code, :data:`LEDGER_MODULES`; test files are exempt):

1. NO RAW ACCESS: the ledger's internals (``store`` — the scans and files, ``queue`` — the fold, ``audit``, and the
   reader's ``live_rows`` / ``_select``) are never imported or reached through an alias.
2. EVERY READ IS DECLARED: a call to ``read`` / ``read_by_regime`` passes, as its first argument (or ``decl=``),
   a ``ReaderDecl(...)`` call or a name bound at MODULE level to one; and that call spells out EVERY field by
   keyword — ``name`` (a string literal), ``purposes`` (``frozenset({...})`` of literal purposes, or
   ``ALL_PURPOSES``), ``regime`` (a ``RegimeFilter(...)``), ``requests`` / ``selection`` / ``inference`` (literals
   from their vocabularies), ``flags_ok`` (``frozenset()`` or ``frozenset({...})`` of literal flags), and
   ``decision_kind`` exactly when ``requests="family"``.
3. NO DIRECT FILE ACCESS: no string constant names a ledger file (``ledger.<writer>.jsonl``) or a path inside a
   ledger root (``_ledger/<sub>``).
4. THE CLOSED LISTS equal design_evaluation.md §0b.2's "closed lists, as built" table, in both directions (the
   ``recipe_doc_gate`` pattern).

**The allowlist is EMPTY.** A new reader is fixed at the source. Opt out explicitly (never silently):
``GEN3AI_SKIP_LEDGER_READER_GATE=1``.

**Blind spots, stated:** a read reached through ``getattr`` / ``importlib`` with a computed name, or a declaration
built by a function in another module. The scan follows import aliases and module-level names, not data flow.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

from utils.paths import repo_path, src_root

pytestmark = pytest.mark.static

#: The ledger's own code: the package and its operator CLI.
LEDGER_MODULES = ("agents/training/eval_ledger/", "main/eval_ledger.py")
#: EMPTY by rule.
ALLOWLIST: Tuple[str, ...] = ()
PACKAGE = "agents.training.eval_ledger"
RAW_SUBMODULES = frozenset({"store", "queue", "audit"})
RAW_NAMES = frozenset({"live_rows", "_select", "scan_rows", "scan_events", "scan_decisions", "scan_references",
                       "iter_jsonl", "row_shards", "shard_has_unit"})
READ_FUNCS = frozenset({"read", "read_by_regime"})
DECL_FIELDS = ("name", "purposes", "regime", "requests", "selection", "flags_ok", "inference")
DOC = ("designs", "endstate", "design_evaluation.md")
CLOSED_LISTS = ("PURPOSES", "REQUEST_KINDS", "PLAYER_KINDS", "SEAT_RULES", "FLAGS", "PROTOCOLS", "DECISION_KINDS",
                "GROUP_SEQUENTIAL_KINDS", "EVENT_KINDS")
_FILE_PATTERNS = (re.compile(r"ledger\.[^\s/]*\.jsonl"), re.compile(r"(^|[/\\])_ledger[/\\][A-Za-z]"))


def _vocab() -> Dict[str, frozenset]:
    from agents.training.eval_ledger import reader as R
    from agents.training.eval_ledger import schema as S

    return {"purposes": frozenset(S.PURPOSES), "flags": frozenset(S.FLAGS), "requests": frozenset(R.REQUEST_SCOPES),
            "selection": frozenset(R.SELECTIONS), "inference": frozenset(R.INFERENCES),
            "decision_kind": frozenset(S.GROUP_SEQUENTIAL_KINDS)}


def _docstring_ids(tree: ast.AST) -> Set[int]:
    out: Set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                out.add(id(first.value))
    return out


class _Aliases:
    """Names in a module bound to the ledger package, its submodules, or names imported from it."""

    def __init__(self, tree: ast.AST):
        self.pkg: Set[str] = set()                    # aliases of the package itself
        self.sub: Dict[str, str] = {}                 # alias -> submodule name
        self.names: Dict[str, Tuple[str, str]] = {}   # local name -> (module, imported name)
        self.raw_imports: List[Tuple[int, str]] = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.name == PACKAGE and a.asname:
                        self.pkg.add(a.asname)
                    elif a.name.startswith(PACKAGE + "."):
                        subm = a.name[len(PACKAGE) + 1:].split(".")[0]
                        if subm in RAW_SUBMODULES:
                            self.raw_imports.append((n.lineno, a.name))
                        if a.asname:
                            self.sub[a.asname] = subm
            elif isinstance(n, ast.ImportFrom) and n.module:
                if n.module == "agents.training":
                    for a in n.names:
                        if a.name == "eval_ledger":
                            self.pkg.add(a.asname or a.name)
                elif n.module == PACKAGE or n.module.startswith(PACKAGE + "."):
                    subm = n.module[len(PACKAGE) + 1:].split(".")[0] if n.module != PACKAGE else ""
                    if subm in RAW_SUBMODULES:
                        self.raw_imports.append((n.lineno, n.module))
                    for a in n.names:
                        local = a.asname or a.name
                        if n.module == PACKAGE and a.name in RAW_SUBMODULES:
                            self.raw_imports.append((n.lineno, f"{n.module}.{a.name}"))
                        if a.name in RAW_NAMES:
                            self.raw_imports.append((n.lineno, f"{n.module}.{a.name}"))
                        if n.module == PACKAGE and a.name in ("reader", "writer", "schema", "cells"):
                            self.sub[local] = a.name
                        else:
                            self.names[local] = (n.module, a.name)

    def resolves_to(self, e: ast.expr, name: str) -> bool:
        """Is ``e`` the ledger's ``name`` (``L.name``, ``R.name`` on the reader, or an imported ``name``)?"""
        if isinstance(e, ast.Name):
            mod_name = self.names.get(e.id)
            return mod_name is not None and mod_name[1] == name
        if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name):
            return e.attr == name and (e.value.id in self.pkg or e.value.id in self.sub)
        return False


def _literal_strs(e: ast.expr) -> Optional[List[str]]:
    if isinstance(e, (ast.Set, ast.List, ast.Tuple)) and all(isinstance(x, ast.Constant) and isinstance(x.value, str)
                                                             for x in e.elts):
        return [x.value for x in e.elts]  # type: ignore[attr-defined]
    return None


def _frozenset_literal(e: ast.expr) -> Optional[List[str]]:
    """``frozenset()`` -> []; ``frozenset({...})`` / ``frozenset((...))`` of string literals -> the strings."""
    if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "frozenset" and not e.keywords:
        if not e.args:
            return []
        if len(e.args) == 1:
            return _literal_strs(e.args[0])
    return None


def _check_decl_call(call: ast.Call, al: _Aliases, vocab: Dict[str, frozenset]) -> List[str]:
    """Why a ``ReaderDecl(...)`` call is not a spelled-out declaration (EMPTY = it is)."""
    bad: List[str] = []
    if call.args:
        bad.append("positional arguments (spell every field by keyword)")
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    if any(k.arg is None for k in call.keywords):
        bad.append("a **kwargs expansion")
    missing = [f for f in DECL_FIELDS if f not in kw]
    if missing:
        bad.append(f"missing fields {missing}")
    if "name" in kw and not (isinstance(kw["name"], ast.Constant) and isinstance(kw["name"].value, str)):
        bad.append("name: not a string literal")
    if "purposes" in kw:
        p = kw["purposes"]
        if al.resolves_to(p, "ALL_PURPOSES"):
            pass
        else:
            lit = _frozenset_literal(p)
            if lit is None or not lit:
                bad.append("purposes: not frozenset({...literal purposes}) or ALL_PURPOSES")
            elif set(lit) - vocab["purposes"]:
                bad.append(f"purposes: {sorted(set(lit) - vocab['purposes'])} are not purposes")
    if "regime" in kw and not (isinstance(kw["regime"], ast.Call) and al.resolves_to(kw["regime"].func, "RegimeFilter")):
        bad.append("regime: not a RegimeFilter(...) call")
    for f in ("requests", "selection", "inference"):
        if f in kw:
            v = kw[f]
            if not (isinstance(v, ast.Constant) and v.value in vocab[f]):
                bad.append(f"{f}: not a literal in {sorted(vocab[f])}")
    if "flags_ok" in kw:
        lit = _frozenset_literal(kw["flags_ok"])
        if lit is None:
            bad.append("flags_ok: not frozenset() / frozenset({...literal flags})")
        elif set(lit) - vocab["flags"]:
            bad.append(f"flags_ok: {sorted(set(lit) - vocab['flags'])} are not flags")
    req = kw.get("requests")
    family = isinstance(req, ast.Constant) and req.value == "family"
    dk = kw.get("decision_kind")
    if family and not (isinstance(dk, ast.Constant) and dk.value in vocab["decision_kind"]):
        bad.append("decision_kind: a family read names its group-sequential decision kind as a literal")
    if not family and dk is not None:
        bad.append("decision_kind: only a family read names one")
    return bad


def scan_source(src: str, filename: str = "<src>") -> List[str]:
    """``"<line>: <why>"`` for every rule broken in ``src`` (rules 1-3)."""
    tree = ast.parse(src, filename)
    al = _Aliases(tree)
    vocab = _vocab()
    hits = [f"{ln}: raw ledger access ({what}) — read through eval_ledger.read with a ReaderDecl"
            for ln, what in al.raw_imports]
    module_decls: Dict[str, ast.Call] = {}
    for n in getattr(tree, "body", []):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and isinstance(n.value, ast.Call) \
                and al.resolves_to(n.value.func, "ReaderDecl"):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    module_decls[t.id] = n.value
    docs = _docstring_ids(tree)
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            if n.value.id in al.pkg and (n.attr in RAW_SUBMODULES or n.attr in RAW_NAMES):
                hits.append(f"{n.lineno}: raw ledger access (eval_ledger.{n.attr})")
            elif n.value.id in al.sub and n.attr in RAW_NAMES:
                hits.append(f"{n.lineno}: raw ledger access ({al.sub[n.value.id]}.{n.attr})")
        if isinstance(n, ast.Call) and any(al.resolves_to(n.func, f) for f in READ_FUNCS):
            arg = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == "decl"), None)
            if arg is None:
                hits.append(f"{n.lineno}: a ledger read without a declaration")
            elif isinstance(arg, ast.Call) and al.resolves_to(arg.func, "ReaderDecl"):
                why = _check_decl_call(arg, al, vocab)
                if why:
                    hits.append(f"{n.lineno}: the inline ReaderDecl is not spelled out: {'; '.join(why)}")
            elif isinstance(arg, ast.Name) and arg.id in module_decls:
                why = _check_decl_call(module_decls[arg.id], al, vocab)
                if why:
                    hits.append(f"{n.lineno}: ReaderDecl {arg.id} is not spelled out: {'; '.join(why)}")
            else:
                hits.append(f"{n.lineno}: a ledger read whose declaration is not a ReaderDecl(...) call or a "
                            "module-level name bound to one")
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            if any(p.search(n.value) for p in _FILE_PATTERNS):
                hits.append(f"{n.lineno}: a string naming a ledger file / path ({n.value[:60]!r}) — the ledger's "
                            "files are opened by its own code only")
    return sorted(set(hits), key=lambda h: int(h.split(":")[0]))


def scan_tree(root: Path) -> List[str]:
    out = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        if rel.endswith("_test.py") or "/target/" in rel or any(rel == m or rel.startswith(m) for m in LEDGER_MODULES):
            continue
        try:
            src = p.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if "eval_ledger" not in src and "ledger." not in src and "_ledger" not in src:
            continue
        out.extend(f"{rel}:{h}" for h in scan_source(src, rel))
    return out


def parse_closed_lists(doc: str) -> Dict[str, List[str]]:
    """The doc's "closed lists, as built" table: ``{LIST: [values]}`` from rows ``| `LIST` | `a`, `b`, ... |``."""
    out: Dict[str, List[str]] = {}
    for line in doc.splitlines():
        m = re.match(r"^\|\s*`([A-Z_]+)`\s*\|(.*)\|\s*$", line)
        if m and m.group(1) in CLOSED_LISTS:
            if m.group(1) in out:
                raise AssertionError(f"the doc's closed-list table names {m.group(1)} twice")
            out[m.group(1)] = re.findall(r"`([^`]+)`", m.group(2))
    return out


def closed_list_problems(doc: str) -> List[str]:
    from agents.training.eval_ledger import schema as S

    table = parse_closed_lists(doc)
    probs = []
    for name in CLOSED_LISTS:
        code = list(getattr(S, name))
        got = table.get(name)
        if got is None:
            probs.append(f"{name}: no row in the doc's closed-list table")
            continue
        if len(set(got)) != len(got):
            probs.append(f"{name}: the doc lists a value twice")
        if set(got) != set(code):
            probs.append(f"{name}: code-only {sorted(set(code) - set(got))}, doc-only {sorted(set(got) - set(code))}")
    return probs


_SKIP = pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_LEDGER_READER_GATE") == "1",
                           reason="GEN3AI_SKIP_LEDGER_READER_GATE=1 (explicit opt-out)")


@_SKIP
def test_every_ledger_reader_declares_its_scope():
    hits = [h for h in scan_tree(src_root()) if h.split(":")[0] not in ALLOWLIST]
    assert not hits, (
        f"{len(hits)} undeclared / raw ledger read(s). Every read goes through eval_ledger.read (or read_by_regime) "
        "with a spelled-out ReaderDecl — purposes, regime, requests, selection, flags_ok, inference (design_"
        "evaluation.md §0b.7). The allowlist is EMPTY:\n  " + "\n  ".join(hits))


@_SKIP
def test_the_closed_lists_equal_the_design_tables():
    probs = closed_list_problems(repo_path(*DOC).read_text())
    assert not probs, ("the ledger's closed lists and design_evaluation.md §0b.2's 'closed lists, as built' table "
                       "differ — change BOTH in one commit:\n  " + "\n  ".join(probs))


def test_the_allowlist_is_empty():
    assert ALLOWLIST == ()


_GOOD_DECL = ('DECL = L.ReaderDecl(name="x", purposes=frozenset({"ab"}), regime=L.RegimeFilter(), requests="own", '
              'selection="include", flags_ok=frozenset(), inference="conditional")\n')


@pytest.mark.parametrize("src", [
    # raw access
    "from agents.training.eval_ledger import store\n",
    "from agents.training.eval_ledger.store import scan_rows\n",
    "import agents.training.eval_ledger.queue as q\n",
    "from agents.training import eval_ledger as L\nrows = L.store.scan_rows(p)\n",
    "from agents.training.eval_ledger import reader as R\nrows = R.live_rows(p)\n",
    "from agents.training.eval_ledger.audit import audit\n",
    # undeclared reads
    "from agents.training import eval_ledger as L\nL.read(root='x')\n",
    "from agents.training import eval_ledger as L\ndef f(d):\n    return L.read(d)\n",
    "from agents.training.eval_ledger import read\ndef f(decl):\n    return read(decl, request_id='r')\n",
    "from agents.training import eval_ledger as L\nd = make()\nL.read(d)\n",
    # declarations that are not spelled out
    "from agents.training import eval_ledger as L\nD = L.ReaderDecl(name='x', purposes=frozenset({'ab'}), "
    "regime=L.RegimeFilter(), requests='own', selection='include', flags_ok=frozenset())\nL.read(D)\n",
    "from agents.training import eval_ledger as L\nD = L.ReaderDecl(name='x', purposes=P, regime=L.RegimeFilter(), "
    "requests='own', selection='include', flags_ok=frozenset(), inference='conditional')\nL.read(D)\n",
    "from agents.training import eval_ledger as L\nD = L.ReaderDecl(name='x', purposes=frozenset({'study'}), "
    "regime=L.RegimeFilter(), requests='own', selection='include', flags_ok=frozenset(), inference='conditional')\n"
    "L.read(D)\n",
    "from agents.training import eval_ledger as L\nD = L.ReaderDecl(name='x', purposes=frozenset({'ab'}), "
    "regime=L.RegimeFilter(), requests=MODE, selection='include', flags_ok=frozenset(), inference='conditional')\n"
    "L.read(D)\n",
    "from agents.training import eval_ledger as L\nD = L.ReaderDecl(name='x', purposes=frozenset({'ab'}), "
    "regime=L.RegimeFilter(), requests='family', selection='include', flags_ok=frozenset(), "
    "inference='across_runs')\nL.read(D, family='f')\n",
    "from agents.training import eval_ledger as L\nD = L.ReaderDecl(name='x', purposes=frozenset({'ab'}), "
    "regime=L.RegimeFilter(), requests='own', selection='include', flags_ok=frozenset({'no_such_flag'}), "
    "inference='conditional')\nL.read(D)\n",
    "from agents.training import eval_ledger as L\nL.read(L.ReaderDecl('x', frozenset({'ab'}), L.RegimeFilter(), "
    "'own', 'include', frozenset(), 'conditional'))\n",
    "from agents.training import eval_ledger as L\nD = L.ReaderDecl(**KW)\nL.read_by_regime(D)\n",
    # a function-local declaration is not a module-level one
    "from agents.training import eval_ledger as L\ndef f():\n" + "    " + _GOOD_DECL + "    return L.read(DECL)\n",
    # direct file access
    "import glob\nfiles = glob.glob(d + '/rows/h2h/ledger.*.jsonl')\n",
    "import os\np = os.path.join(models, '_ledger/rows')\n",
])
def test_the_scan_sees_every_undeclared_or_raw_reader(src):
    assert scan_source(src), src


@pytest.mark.parametrize("src", [
    "from agents.training import eval_ledger as L\n" + _GOOD_DECL + "rows = L.read(DECL, request_id='r')\n",
    "from agents.training import eval_ledger as L\n" + _GOOD_DECL + "rows = L.read_by_regime(decl=DECL)\n",
    "from agents.training.eval_ledger import ReaderDecl, RegimeFilter, read, ALL_PURPOSES\n"
    "D = ReaderDecl(name='y', purposes=ALL_PURPOSES, regime=RegimeFilter(mirrored=True), requests='any', "
    "selection='exclude', flags_ok=frozenset({'digest_unrecorded'}), inference='across_runs')\nread(D)\n",
    "from agents.training import eval_ledger as L\nD = L.ReaderDecl(name='f', purposes=frozenset({'ab'}), "
    "regime=L.RegimeFilter(), requests='family', selection='include', flags_ok=frozenset(), "
    "inference='across_runs', decision_kind='ab_verdict')\nL.read(D, family='f')\n",
    "from agents.training import eval_ledger as L\nrows = L.read(L.ReaderDecl(name='z', purposes=frozenset(('cycle',"
    ")), regime=L.RegimeFilter(), requests='any', selection='include', flags_ok=frozenset(), "
    "inference='conditional'))\n",
    # writers, schema helpers and docstrings are not reads
    '"""Rows go to <root>/rows/h2h/ledger.<writer>.jsonl."""\nfrom agents.training import eval_ledger as L\n'
    "w = L.LedgerWriter(None, producer='h2h')\nL.validate_row(r)\n",
    # an unrelated `read`
    "import pandas as pd\npd.read_csv('x')\ndef read(x):\n    return x\nread(3)\n",
])
def test_the_scan_passes_declared_reads_writers_and_docstrings(src):
    assert scan_source(src) == [], (src, scan_source(src))


def test_the_closed_list_check_has_teeth():
    doc = repo_path(*DOC).read_text()
    assert closed_list_problems(doc) == []
    typo = "\n".join(line.replace("`monitor`", "`montior`") if line.startswith("| `PURPOSES` |") else line
                     for line in doc.splitlines())
    assert any("PURPOSES" in p for p in closed_list_problems(typo))
    extra = "\n".join(line.replace("`balanced`", "`balanced`, `p2_only`") if line.startswith("| `SEAT_RULES` |")
                      else line for line in doc.splitlines())
    assert any("SEAT_RULES" in p and "p2_only" in p for p in closed_list_problems(extra))
    gone = "\n".join(line for line in doc.splitlines() if not line.startswith("| `FLAGS` |"))
    assert any("FLAGS: no row" in p for p in closed_list_problems(gone))
