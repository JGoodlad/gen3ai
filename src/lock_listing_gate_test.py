"""Does any code but ``utils.procfs`` read the ``/proc/locks`` listing? The static gate of the lock readings.

**The bug class.** ``/proc/locks`` is handed out one 4096-byte page per ``read()`` and resumed by position, so a lock
released between two reads makes the listing SKIP a live entry: an entry listed did exist, an entry NOT listed proves
nothing. Seven readers acted on the negative (2026-11-09): a held GPU lease read as dead, a held re-entrance marker as
stale, an ancestor's hold as somebody else's, a held heavy slot as free (its job's memory cap not charged: one job
more than the box's budget admits), a held gate slot as free, a working lock holder as absent (its waiter flagged as stalled), a live chain
supervisor as having no lock.

**The rule.** The listing has ONE reader, ``src/utils/procfs.py``, and its raw readings do not leave it:

1. nothing else names ``procfs.read_locks`` / ``procfs._read_listing`` (a call, a reference, an import);
2. nothing else calls the module-level ``procfs.holders`` (the LISTED holders: its ``[]`` is the unverified negative);
3. no other non-test code spells the listing's path (``/proc/locks``, ``…/locks``, ``join(proc, "locks")``) or the
   ``lslocks`` tool in a string that is not a docstring, and no shell script under ``src/`` or ``scripts/`` names
   either outside a comment;
4. the one cheap reading that stops before the all-process walk, ``LockTable.listed_holders``, is called by its ONE
   declared caller (:data:`UNVERIFIED_READER`), which repeats the verified reading before anybody acts on it;
5. only a test imports ``utils.lock_listing_testing`` (it REPLACES the reading with one that misses a live lock);
6. the WAITERS of a lock (``LockTable.waiters`` / ``procfs.waiters``) are read by ONE declared caller
   (:data:`DISPLAY_READER`), and it only DISPLAYS them: a blocked request has no ``fdinfo`` line, so "nobody waits" can
   never be verified and nothing may act on it. A new reader of ``.waiters`` in non-test code is declared here, with
   its reason, or it asks the question another way (``holds_lock`` about a pid it knows).

What a caller uses instead: ``procfs.holds_lock(pid, key)`` for a pid it knows; ``procfs.LockTable().holders(key,
known)`` / ``procfs.lock_holders(key)`` for "who holds it / is it free"; a non-blocking ``flock`` to take it.

**The declared exceptions** (classes, never single favours; there is no allowlist): THE OWNER (the module, its test,
``utils.lock_listing_testing`` which replaces its reading with one that misses a live lock, and this gate); tests are
exempt from rules 3 and 5 only (a fake ``/proc`` tree has to write its ``locks`` file).

Opt out explicitly (never silently): ``GEN3AI_SKIP_LOCK_LISTING_GATE=1``.

**Blind spots, stated:** a path assembled at run time from pieces none of which ends in ``/locks``; a reader in a
file outside ``src/`` and ``scripts/`` or in another language (the repo's other trees, ``tools/`` and the root
``conftest.py``, were read for one on 2026-11-10 and hold none; a session script under ``~/.claude`` is outside any
gate); the DECLARED display caller of rule 6 acting on its waiters one day (the gate cannot tell a display from an
action, it only keeps the sites to the one that was reviewed); and a holder that only another user's processes keep
open, which ``LockTable`` itself reads from the listing alone.
"""
from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import pytest

from utils.paths import repo_path

pytestmark = pytest.mark.static

#: THE OWNER (repo-relative): the one reader, its test, the test support that replaces its reading, and this gate.
OWNER = ("src/utils/procfs.py", "src/utils/procfs_test.py", "src/utils/lock_listing_testing.py",
         "src/lock_listing_gate_test.py")
#: Where code is scanned.
TREES = ("src/", "scripts/")
#: Rule 1: the raw readings of the listing.
RAW_READERS = frozenset({"read_locks", "_read_listing"})
#: Rule 2: the module-level function whose empty answer is the unverified negative.
LISTED_ONLY = "holders"
#: Rule 4: ``(file, function)`` of the one caller of ``LockTable.listed_holders``.
UNVERIFIED_READER = ("src/utils/heavy_slots.py", "_slot_state")
#: Rule 6: ``(file, function)`` of the one caller of ``waiters``: ``--status``'s queue, a display (a queue shown can be
#: one waiter short, never a slot shown free).
DISPLAY_READER = ("src/utils/gate_lock_status.py", "collect")
_MODULE = "utils.procfs"
#: Rule 5: the test support that makes the reading miss a live lock.
TEST_SUPPORT = "lock_listing_testing"
_SHELL_SUFFIXES = frozenset({".sh", ".bash", ""})


def _is_test(rel: str) -> bool:
    return rel.endswith("_test.py") or rel.rsplit("/", 1)[-1] == "conftest.py"


def _names_the_listing(value: str) -> bool:
    return "proc/locks" in value or "lslocks" in value or value.endswith("/locks")


def _docstrings(tree: ast.AST) -> Set[int]:
    """ids of the string constants that are statements of their own (docstrings and bare-string comments)."""
    return {id(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}


def _is_str(node: ast.AST, value: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def scan_python(text: str, rel: str) -> List[Tuple[int, str]]:
    """``(line, what)`` for every breach of the six rules in the python source ``text`` of the file ``rel``."""
    tree = ast.parse(text)
    hits: Set[Tuple[int, str]] = set()
    aliases: Set[str] = set()                    # the names this file binds to the module utils.procfs
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "utils":
            aliases.update(a.asname or a.name for a in n.names if a.name == "procfs")
        elif isinstance(n, ast.Import):
            aliases.update(a.asname for a in n.names if a.name == _MODULE and a.asname)
        elif isinstance(n, ast.ImportFrom) and n.module == _MODULE:
            for a in n.names:
                if a.name in RAW_READERS or a.name == LISTED_ONLY or a.name == "*":
                    hits.add((n.lineno, f"imports `{a.name}` from {_MODULE} (the listing's raw reading)"))

    def is_module(node: ast.AST) -> bool:
        return ((isinstance(node, ast.Name) and node.id in aliases)
                or (isinstance(node, ast.Attribute) and ast.unparse(node) == _MODULE))

    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in RAW_READERS or isinstance(n, ast.Name) and n.id in RAW_READERS:
            name = n.attr if isinstance(n, ast.Attribute) else n.id
            hits.add((n.lineno, f"names `{name}` (the listing's raw reading)"))
        elif isinstance(n, ast.Attribute) and n.attr == LISTED_ONLY and is_module(n.value):
            hits.add((n.lineno, f"reads `{ast.unparse(n)}` (the LISTED holders: its empty answer proves nothing)"))
    # rule 4: the one caller of the cheap reading
    allowed: Set[int] = set()
    if rel == UNVERIFIED_READER[0]:
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == UNVERIFIED_READER[1]:
                allowed.update(id(x) for x in ast.walk(fn))
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == "listed_holders" and id(n) not in allowed:
            hits.add((n.lineno, "calls `listed_holders` (the reading that stops before the all-process walk) outside "
                                f"{UNVERIFIED_READER[0]}::{UNVERIFIED_READER[1]}"))
    if _is_test(rel):
        return sorted(hits)
    # rule 6: the waiters have one declared caller (a display); tests may read them (they build the queue they check)
    shown: Set[int] = set()
    if rel == DISPLAY_READER[0]:
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == DISPLAY_READER[1]:
                shown.update(id(x) for x in ast.walk(fn))
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == "waiters" and id(n) not in shown:
            hits.add((n.lineno, "reads `.waiters` (a blocked request has no fdinfo line: its absence proves nothing, so "
                                f"it is DISPLAYED only, by {DISPLAY_READER[0]}::{DISPLAY_READER[1]})"))
    # rule 5: the reading is replaced in tests only
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)) and (
                TEST_SUPPORT in (getattr(n, "module", None) or "") or any(TEST_SUPPORT in a.name for a in n.names)):
            hits.add((n.lineno, f"imports utils.{TEST_SUPPORT} (it makes the lock reading miss a live lock: tests only)"))
    # rule 3: the path of the listing, spelled
    docs = _docstrings(tree)
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and _names_the_listing(n.value):
            hits.add((n.lineno, f"spells the lock listing ({n.value[-40:]!r})"))
        elif isinstance(n, ast.Call) and any(_is_str(a, "locks") for a in n.args):
            hits.add((n.lineno, "builds a path to a `locks` file"))
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) and (_is_str(n.left, "locks") or _is_str(n.right, "locks")):
            hits.add((n.lineno, "builds a path to a `locks` file"))
    return sorted(hits)


def scan_shell(text: str) -> List[Tuple[int, str]]:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        code = line.split("#", 1)[0] if line.lstrip().startswith("#") or " #" in line else line
        if "proc/locks" in code or "lslocks" in code:
            out.append((i, "reads the lock listing from a shell script"))
    return out


def tracked_files(root: Path) -> List[str]:
    r = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
                        *TREES], capture_output=True, timeout=120, check=True)
    return sorted(p for p in r.stdout.decode("utf-8", "surrogateescape").split("\0") if p)


def scan(root: Path, files: Iterable[str]) -> List[str]:
    out: List[str] = []
    for rel in files:
        suffix = Path(rel).suffix.lower()
        if rel in OWNER or not rel.startswith(TREES) or (suffix != ".py" and suffix not in _SHELL_SUFFIXES):
            continue
        p = root / rel
        try:
            if not p.is_file() or p.is_symlink():
                continue
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if suffix == ".py":
            if not any(t in text for t in ("locks", LISTED_ONLY, "listed_holders", TEST_SUPPORT, "waiters")):
                continue                              # the cheap pass: nothing any rule could match
            try:
                found = scan_python(text, rel)
            except SyntaxError:
                continue                              # not this gate's finding (ruff's)
        else:
            if suffix == "" and not text.startswith("#!"):
                continue
            found = scan_shell(text)
        out.extend(f"{rel}:{n}: {what}" for n, what in found)
    return out


def test_only_procfs_reads_the_lock_listing() -> None:
    if os.environ.get("GEN3AI_SKIP_LOCK_LISTING_GATE"):
        pytest.skip("GEN3AI_SKIP_LOCK_LISTING_GATE is set")
    root = repo_path()
    hits = scan(root, tracked_files(root))
    assert not hits, (
        "the /proc/locks listing is read outside its ONE reader (utils.procfs); a listing can skip a live lock, so "
        "\"not listed\" is never \"not held\":\n  " + "\n  ".join(hits[:40]) + ("\n  …" if len(hits) > 40 else "")
        + "\nAsk instead: `procfs.holds_lock(pid, key)` about a pid you know (its own fdinfo); "
          "`procfs.LockTable().holders(key, known)` / `procfs.lock_holders(key)` for who holds a file or whether it "
          "is free; a non-blocking flock to take it. There is no allowlist.")


def test_the_one_unverified_reader_exists() -> None:
    """Rule 4 names a real function that really calls the cheap reading, and a verified one beside it."""
    rel, fn_name = UNVERIFIED_READER
    tree = ast.parse(repo_path(*rel.split("/")).read_text())
    fns = [f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == fn_name]
    assert len(fns) == 1, f"{rel} has no single `{fn_name}`: update UNVERIFIED_READER"
    attrs = {n.attr for n in ast.walk(fns[0]) if isinstance(n, ast.Attribute)}
    assert {"listed_holders", "holders"} <= attrs, f"{rel}::{fn_name} no longer makes both readings: drop rule 4's exception"


def test_the_one_display_reader_exists() -> None:
    """Rule 6 names a real function that really reads the waiters (and says so about its own shape)."""
    rel, fn_name = DISPLAY_READER
    tree = ast.parse(repo_path(*rel.split("/")).read_text())
    fns = [f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == fn_name]
    assert len(fns) == 1, f"{rel} has no single `{fn_name}`: update DISPLAY_READER"
    assert "waiters" in {n.attr for n in ast.walk(fns[0]) if isinstance(n, ast.Attribute)}, \
        f"{rel}::{fn_name} no longer reads the waiters: drop rule 6's exception"


def test_the_scanner_has_teeth() -> None:
    def what(src: str, rel: str = "src/x/y.py") -> List[int]:
        return [n for n, _ in scan_python(src, rel)]

    # rule 1: the raw reading, by any route, in code and in tests
    assert what("from utils import procfs\nlocks = procfs.read_locks()\n") == [2]
    assert what("from utils import procfs as P\nx = P._read_listing('p')\n", "src/x/y_test.py") == [2]
    assert what("from utils.procfs import read_locks\n") == [1]
    assert what("from utils.procfs import holders\n") == [1] and what("from utils.procfs import *\n") == [1]
    assert what("import utils.procfs\nf = utils.procfs.read_locks\n") == [2]
    # rule 2: the listed holders of the MODULE (any alias); a LockTable's verified answer is not it
    assert what("from utils import procfs as P\nfree = not P.holders(key)\n") == [2]
    assert what("from utils import core_split, procfs\nhs = procfs.holders(key, locks)\n", "src/x/y_test.py") == [2]
    assert what("import utils.procfs as pf\nhs = pf.holders(k)\n") == [2]
    assert what("import utils.procfs\nhs = utils.procfs.holders(k)\n") == [2]
    assert not what("from utils import procfs\nhs = procfs.LockTable().holders(key)\nt = procfs.lock_holders(key)\n")
    assert not what("from utils import procfs\nok = procfs.holds_lock(pid, key)\n")
    assert not what("holders = table.holders(key)\nself.holders(x)\n")
    # rule 3: the path, in code only; a docstring may explain it
    assert what("with open('/proc/locks') as f:\n    pass\n") == [1]
    assert what("p = f'{proc}/locks'\n") == [1] and what("p = os.path.join(proc, 'locks')\n") == [1]
    assert what("p = Path(proc) / 'locks'\n") == [1] and what("subprocess.run(['lslocks', '-o', 'PID'])\n") == [1]
    assert not what('"""Reads /proc/locks never; see utils.procfs."""\nx = {"locks": 1}\ny = "it unlocks"\n')
    assert not what("with open('/proc/locks') as f:\n    pass\np = os.path.join(proc, 'locks')\n", "src/x/y_test.py")
    # rule 4: the cheap reading has one caller
    cheap = "def _slot_state(t, key):\n    return t.listed_holders(key) or t.holders(key)\n"
    assert not what(cheap, UNVERIFIED_READER[0])
    assert what(cheap, "src/utils/gate_lock.py") == [2]
    assert what("def admission(t, key):\n    return t.listed_holders(key)\n", UNVERIFIED_READER[0]) == [2]
    # rule 6: the waiters are displayed by one declared caller, in code; a test builds the queue it checks
    show = "def collect(t, key):\n    return t.waiters(key)\n"
    assert not what(show, DISPLAY_READER[0])
    assert what(show, "src/utils/gate_lock.py") == [2] and what("w = procfs.waiters(key)\n") == [1]
    assert what("def idle(t, k):\n    return not t.waiters(k)\n", DISPLAY_READER[0]) == [2]
    assert what("def other(t, k):\n    return not t.waiters(k)\n" + show, DISPLAY_READER[0]) == [2]
    assert not what("w = procfs.waiters(key)\n", "src/utils/x_test.py") and not what("waiters = []\nwaiters.append(1)\n")
    # rule 5: the reading is replaced by tests only
    assert what("from utils import lock_listing_testing as T\n") == [1] and what("import utils.lock_listing_testing\n") == [1]
    assert what("from utils.lock_listing_testing import without\n") == [1]
    assert not what("from utils import lock_listing_testing as T\nT.install(T.without())\n", "src/x/y_test.py")
    # shell
    assert [n for n, _ in scan_shell("#!/bin/bash\n# reads /proc/locks: no\ngrep FLOCK /proc/locks\nlslocks | wc -l\n")] == [3, 4]


def test_the_scan_covers_code_and_tests_and_skips_only_the_owner(tmp_path: Path) -> None:
    bad = "from utils import procfs\nfree = not procfs.holders(k)\n"
    files = {"src/a/b.py": bad, "src/a/b_test.py": bad, "scripts/ops/w.py": "open('/proc/locks')\n",
             "scripts/ops/w.sh": "#!/bin/bash\ncat /proc/locks\n", "scripts/tool": "#!/bin/sh\nlslocks\n",
             "src/utils/procfs.py": bad + "open('/proc/locks')\n", "src/utils/procfs_test.py": bad,
             "tools/out_of_tree.py": bad, "src/a/notes.md": "/proc/locks", "src/a/clean.py": "x = 1\n",
             "src/a/data": "proc/locks but no shebang\n"}
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    assert sorted({h.split(":")[0] for h in scan(tmp_path, sorted(files))}) == [
        "scripts/ops/w.py", "scripts/ops/w.sh", "scripts/tool", "src/a/b.py", "src/a/b_test.py"]
    for rel in OWNER:
        assert repo_path(*rel.split("/")).is_file(), f"{rel} is gone: drop it from OWNER"
