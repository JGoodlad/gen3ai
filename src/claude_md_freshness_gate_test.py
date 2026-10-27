"""Do the instruction docs still describe THIS tree, within their budget? The static gate for DOCUMENTATION.

Sits beside `ruff_gate_test.py` and `file_size_gate_test.py` at the `src/` root because, like those, its subject
is the whole tree rather than any one package.

**Why it exists.** Every `CLAUDE.md` in this repo is loaded into an agent's context and read as
fact. A path that has moved and a flag that has been deleted do not merely go stale — they are
*actively believed*, and the believer then reports a false result or types a command that cannot
launch. That failure has a track record here: `main.checkargs` exists because a deleted flag in a
recorded argv costs a launch-crash-fix loop, and the root `CLAUDE.md`'s own c-family lesson is
that an allowlist entry can outlive its own fix and mislead every reader after it. Prose has no
compiler, so it needs a gate.

**Five checks, all mechanical.** None is a judgement about writing. Checks 1-3 scan every ALWAYS-CURRENT doc, not
only the `CLAUDE.md` files: `designs/ARCHITECTURE.md`, `designs/endstate/*.md`, the `designs/ops/` SOPs (+ the
runbook and `testing.md`), the seven `designs/<topic>/` leaf-detail trees (`training/`, `model/`, `rust_sim/`,
`prober/`, `launcher/`, `observation/`, `tools/`: `utils.instruction_budget.TREE_DIRS`), `docs/*.md` and every owned
`README.md` (`always_current_docs()`; the history trees — `measurements/`, the archives, `ai_v3/` — are out of scope on
purpose). A closed round's log is history and lives in `designs/research_state/claude_md_archive/`, not in a tree.

1. **Every repo-relative path a CLAUDE.md names must EXIST** (a `data/*.ts` citation is a Showdown source and
   resolves in `deps/pokemon-showdown`; a path gone on purpose is listed in `designs/deleted_flags.md`). Scope is the five owned prefixes
   (`src/`, `designs/`, `data/`, `tools/`, `scripts/`) — `deps/` is a submodule plus untracked
   build output, so its contents are not a promise this tree can keep in a fresh worktree. **A path git IGNORES**
   (`git check-ignore`: a build output such as `src/rust_env/target/selfcheck/libpokesim_env.so`) is unverifiable by
   design and is skipped: a fresh worktree has no Rust build, and a static gate must not depend on one. A tracked
   path stays strict.
2. **Every `--flag` a CLAUDE.md names must RESOLVE somewhere in the tree's own CLI surface** —
   or be listed in `designs/deleted_flags.md`, the history file for flags a document names on
   purpose because it is telling you what used to be true.
3. **Every `python -m <module>` a CLAUDE.md names must be RUNNABLE**: `<module>.py`, or a package
   with a `__main__.py`, under `src/` or the repo root — or a third-party tool in `_EXTERNAL_MODULES`.
4. **Every instruction file stays within its BYTE BUDGET** — the ratchet of `utils.instruction_budget`
   (rules there; numbers in `designs/ops/instruction_docs_budget.json`): every `CLAUDE.md`, slash command and
   `designs/<topic>/` tree doc (kind `tree`). Folded into this gate, not a gate of its own, so the root's pinned
   static-gate count does not move (audit U1).
5. **Every `<file> § <section>` pointer in any doc or code comment must land on a section** of the file it names
   (`utils.section_refs`: the forms, what resolves one and what is skipped are its docstring; `python -m
   utils.section_refs` lists the dangling ones). Scope is the whole tree, not only the `CLAUDE.md` files.

**Why the flag surface is an AST scan and not `build_parser()`.** Three reasons, in order of
weight. (a) `import main.train_rl_agent` pulls torch and the whole training stack — measured
1.15 s for that ONE parser, and this gate covers roughly thirty of them; a gate that costs ten
seconds gets excluded. (b) Several entry points live in `__main__.py`, and importing one of those
is the 2026-09-06 incident `entry_point_guard_test.py` records: a session imported
`main.launcher.__main__` to inspect its parser and the import *started a training run*. (c) The
scan reads the same `add_argument` literals the parsers are built from, so the two agree — that
was MEASURED against the live `build_parser()` while this gate was written: of 588 live options,
every single one the scan missed was an auto-generated `--no-` negation, which is why the
negation rule below exists rather than being a guess.

The honest limit of an AST scan is a flag whose name is computed at runtime rather than written
as a literal. `agents/model/flag_registry.py` generates argparse entries and was the obvious
candidate; it was checked and it does not do this. If a parser ever starts building option
strings from f-strings, this gate goes quiet about them, and the fix is to give that generator a
literal table the scan can see.

**What is NOT checked, deliberately.** Whether a document's *claims* are true. A measured number,
a design rationale, a "this is 22% of worker CPU" — none of that has a mechanical referent, and a
gate that pretended otherwise would either pass everything or block on prose review. This gate
checks the things that have a referent in the filesystem: names of files, flags and modules, and sizes.

**Cost: about a second of work** (a few seconds of pytest start-up) — it reads ~90 markdown files and
AST-parses the tree once. No cost marker, runs in the fast inner loop. Opt out explicitly:

    GEN3AI_SKIP_CLAUDE_MD_GATE=1 pytest src/ -q     # the whole gate
    GEN3AI_SKIP_USER_DOC_BUDGET=1 pytest src/ -q    # only the machine-local files' budget (see that test)
"""
from __future__ import annotations

import ast
import functools
import os
import pathlib
import re
import subprocess
from typing import Dict, List, Optional, Set, Tuple

import pytest

from utils import instruction_budget as budget
from utils import section_refs
from utils.paths import repo_root

pytestmark = pytest.mark.static   # the `static` budget tier (conftest._STATIC_BUDGET_BASE_S)

_REPO_ROOT = repo_root()

_SKIP = pytest.mark.skipif(
    os.environ.get("GEN3AI_SKIP_CLAUDE_MD_GATE") == "1",
    reason="GEN3AI_SKIP_CLAUDE_MD_GATE=1",
)

# --------------------------------------------------------------------------------------------
# Which documents are in scope
# --------------------------------------------------------------------------------------------

def claude_md_files() -> List[pathlib.Path]:
    return budget.claude_md_files(_REPO_ROOT)


# The ALWAYS-CURRENT docs the root `CLAUDE.md` names beside the `CLAUDE.md` files: they are read as fact exactly as
# a leaf is, so checks 1-3 scan them too (the byte budget, check 4, stays `CLAUDE.md`-only: it is their ratchet).
# Declared by name or glob, never inferred: a dated audit, a measurement record and a frozen archive are HISTORY
# and name deleted things on purpose (`designs/CLAUDE.md` lists which trees are which).
_ALWAYS_CURRENT_GLOBS = (
    "designs/ARCHITECTURE.md",
    "designs/STATUS.md",              # the one-page dashboard (its SHAs and date: `status_doc_gate_test.py`)
    "designs/endstate/*.md",
    "designs/ops/*_SOP.md",
    "designs/ops/training_runbook.md",
    "designs/ops/testing.md",
    "designs/learning/*.md",          # always-current concept explainers (`designs/CLAUDE.md`)
    "designs/ai_v12/launch_runbook.md",
    "README.md",
    "docs/*.md",
    *(f"designs/{tree}/**/*.md" for tree in budget.TREE_DIRS),   # the seven leaf-detail trees (`designs/CLAUDE.md`)
)
# Every `README.md` in the tree EXCEPT these: a vendored/untracked/build tree, and the HISTORY trees (frozen digraph,
# dated measurement records, archives).
_README_EXCLUDED_PARTS = ("deps", ".claude", ".pytest_cache", "node_modules", "target", "upstream", "ai_v3",
                          "measurements", "claude_md_archive", "memory_archive")


def _rel(doc: pathlib.Path) -> str:
    return doc.relative_to(_REPO_ROOT).as_posix() if doc.is_relative_to(_REPO_ROOT) else str(doc)


def always_current_docs() -> List[pathlib.Path]:
    """Every doc checks 1-3 scan: the `CLAUDE.md` files, the globs above, and every owned `README.md`."""
    docs = set(claude_md_files())
    for glob in _ALWAYS_CURRENT_GLOBS:
        docs.update(p for p in _REPO_ROOT.glob(glob) if p.is_file())
    for p in _REPO_ROOT.rglob("README.md"):
        if not any(part in _README_EXCLUDED_PARTS for part in p.relative_to(_REPO_ROOT).parts[:-1]):
            docs.add(p)
    return sorted(docs)


# --------------------------------------------------------------------------------------------
# (a) PATHS
# --------------------------------------------------------------------------------------------

# The five prefixes this repo OWNS. `deps/` is excluded on purpose: the submodule may be
# uninitialised and `deps/venv` is untracked, so "it exists on the box that wrote the sentence" is
# not a property a fresh clone shares.
_OWNED_PREFIXES = ("src", "designs", "data", "tools", "scripts")

_PATH_RE = re.compile(
    r"(?<![\w/.\-])((?:" + "|".join(_OWNED_PREFIXES) + r")/[A-Za-z0-9_./*<>{}\-]+)"
)

# A token is a PLACEHOLDER, not a path claim, if it carries any of these. `ai_vN` / `ai_vX` are
# the version-folder placeholders `designs/CLAUDE.md` teaches readers to substitute into.
_PLACEHOLDER_MARKS = ("<", ">", "*", "{", "}", "…")
_PLACEHOLDER_SEGMENTS = re.compile(r"(?:^|/)(?:ai_v[NX]|v[NX]|NNNN|N)(?:/|$)")

# Some CLAUDE.md files legitimately cite paths that are relative to something other than the repo
# root. Declared per file WITH THE REASON, never inferred, so a genuine typo cannot be absorbed by
# a base that happens to contain a same-named file.
_UPSTREAM_BASES: Tuple[Tuple[str, str], ...] = (
    ("deps/pokemon-showdown", "the upstream Showdown data/ the port was derived from"),
    ("deps/pokemon-showdown/dist", "the compiled upstream, cited by line number"),
)
#: A key ending in "/" covers every doc under that directory.
_EXTRA_BASES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    # The Rust port's leaf cites (1) its own crate layout — `src/turn.rs` means `src/rust_sim/src/turn.rs` (the doc's
    # own directory is already a base) — and (2) the upstream Showdown sources it was ported FROM, which is the whole
    # point of a port's evidence trail. Its `designs/rust_sim/` detail tree cites both, from outside the crate.
    "src/rust_sim/CLAUDE.md": _UPSTREAM_BASES,
    "designs/rust_sim/": (("src/rust_sim", "the crate's own layout: `src/turn.rs` means `src/rust_sim/src/turn.rs`"),
                          *_UPSTREAM_BASES),
}


def _extra_bases(rel_doc: str) -> Tuple[Tuple[str, str], ...]:
    out: Tuple[Tuple[str, str], ...] = ()
    for key, bases in _EXTRA_BASES.items():
        if rel_doc == key or (key.endswith("/") and rel_doc.startswith(key)):
            out += bases
    return out


# Tokens that MATCH the path regex but are prose, not path claims. Each needs a reason; the list
# may only shrink. Do not park a real broken path here.
_PROSE_NOT_A_PATH: Dict[str, str] = {
    # "regenerate after any data/derivation change" — English ("the data derivation"), not a file.
    "data/derivation": "prose: 'any data/derivation change' means the derivation step, not a file",
}


def _is_placeholder(token: str) -> bool:
    return (any(m in token for m in _PLACEHOLDER_MARKS)
            or bool(_PLACEHOLDER_SEGMENTS.search(token)))


_UPSTREAM_BASE = "deps/pokemon-showdown"


def _is_upstream_source(token: str) -> bool:
    """`data/moves.ts`, `data/mods/gen4/items.ts`: a Showdown SOURCE file the docs cite as evidence. This repo's own
    `data/` holds derived JSON and no TypeScript, so a `.ts` token under `data/` is upstream's by construction."""
    return token.startswith("data/") and token.endswith(".ts")


def _bases_for(rel_doc: str, token: str = "") -> List[pathlib.Path]:
    """Where a path named in `rel_doc` is allowed to resolve, in order."""
    if _is_upstream_source(token):
        # An uninitialised submodule is not a documentation defect: no base, and `test_every_..._path_exists`
        # skips the token (a fresh worktree without the submodule must not fail on Showdown citations).
        base = _REPO_ROOT / _UPSTREAM_BASE
        return [base] if (base / "data").is_dir() else []
    bases = [_REPO_ROOT, _REPO_ROOT / pathlib.PurePosixPath(rel_doc).parent]
    for extra, _reason in _extra_bases(rel_doc):
        base = _REPO_ROOT / extra
        # An uninitialised submodule is not a documentation defect. Skip the base rather than
        # failing every citation into it.
        if base.exists():
            bases.append(base)
    return bases


def _git_ignored(rel_path: str, root: pathlib.Path) -> bool:
    """True when git ignores `rel_path` (repo-relative, need not exist): `git check-ignore -q` exits 0. Any other
    outcome (not ignored, no git, not a repo) is False, so the gate stays strict wherever it cannot tell."""
    try:
        r = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", "--", rel_path],
                           capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def unverifiable_by_design(rel_doc: str, token: str, root: pathlib.Path = _REPO_ROOT) -> bool:
    """A claimed path that git ignores is a build output (or other untracked local state): it exists in the checkout
    that built it and not in a fresh worktree, so its absence is not a documentation defect."""
    for base in _bases_for(rel_doc, token):
        try:
            rel = (base / token).relative_to(root).as_posix()
        except ValueError:
            continue
        if _git_ignored(rel, root):
            return True
    return False


def collect_path_claims(docs: Optional[List[pathlib.Path]] = None) -> List[Tuple[str, int, str]]:
    """Every (doc, line, path) an always-current doc asserts exists."""
    claims = []
    for doc in (always_current_docs() if docs is None else docs):
        rel = _rel(doc)
        for lineno, line in enumerate(doc.read_text().splitlines(), 1):
            for m in _PATH_RE.finditer(line):
                token = m.group(1).rstrip(".,;:)`'\"/")   # a directory named with its slash is the directory
                if _is_placeholder(token) or token in _PROSE_NOT_A_PATH:
                    continue
                claims.append((rel, lineno, token))
    return claims


# --------------------------------------------------------------------------------------------
# (b) FLAGS — the tree's own CLI surface
# --------------------------------------------------------------------------------------------

_PY_ROOTS = ("src", "tools")

# Non-Python entry points that a CLAUDE.md documents by flag: the Rust binaries, the Node harness
# and bridge, and the bootstrap script.
_LITERAL_GLOBS = (
    "src/rust_sim/**/*.rs",
    "src/rust_sim/**/*.js",
    "src/utils/bridge/*.js",
    "scripts/**/*.sh",
    "scripts/**/*.py",
)

# Prefilter for the Python scan: does this file contain a quoted `--something` at all?
_MAYBE_FLAG_RE = re.compile(r"""["']--[A-Za-z]""")

_LITERAL_FLAG_RE = re.compile(r"""["'](--[A-Za-z][A-Za-z0-9_-]*)["']""")
# A shell `case` arm: `--with-rust)`.
_SH_CASE_FLAG_RE = re.compile(r"(?<![\w\-])(--[A-Za-z][A-Za-z0-9-]*)\)")
# A string constant that IS a flag name (`"--neural-opponent" in sys.argv`), as opposed to a
# docstring or a message that merely mentions one.
_FLAG_CONSTANT_RE = re.compile(r"^--[A-Za-z][A-Za-z0-9_-]*$")


def _harvests_bare_constants(rel: str) -> bool:
    """May a BARE `"--x"` string constant in this file count as a flag the tree accepts?

    Yes for production modules and for run-directly scripts, because several of those parse
    `sys.argv` by hand — `eval_sharding_fuzz_test.py`'s `"--neural-opponent" in sys.argv` is a real
    flag with no `add_argument` behind it.

    NO for ordinary `*_test.py` files, and that exclusion is load-bearing rather than tidy:
    `checkargs_test.py` passes `--pubval-mode` precisely to assert that it is UNKNOWN. Counting
    that literal as CLI surface made a deleted flag read as live, which would have let a stale
    document naming it pass forever. A test asserting a flag's absence must not be the reason it
    looks present. `*_fuzz_test.py` is exempt from the exclusion — by this repo's own naming
    convention those are scripts you run directly, not collected tests.
    """
    name = rel.rsplit("/", 1)[-1]
    return (not name.endswith("_test.py")) or name.endswith("_fuzz_test.py")


def _docstring_nodes(tree: ast.AST) -> Set[int]:
    """id() of every Constant that is a module/class/function docstring.

    Docstrings are where a script's `Run:` header lives, and a header may name a flag the script
    no longer takes. Excluding them keeps the surface to flags the code actually READS.
    """
    out: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                out.add(id(body[0].value))
    return out


@functools.lru_cache(maxsize=1)
def cli_flag_surface() -> frozenset:
    """Every option string the tree's own entry points accept, dash-normalised.

    Cached: three of the four tests need it, and re-deriving it per test tripled this gate's
    cost for no new information. A test run is a single snapshot of the tree by construction.
    """
    flags: Set[str] = set()

    for root_name in _PY_ROOTS:
        root = _REPO_ROOT / root_name
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            rel = str(p.relative_to(_REPO_ROOT))
            text = p.read_text(errors="replace")
            # Cheap prefilter: a file with no `"--x"` substring at all cannot contribute an
            # option string, and parsing it is pure cost. Measured: 2.9 s -> 0.7 s over ~800
            # files, with a byte-identical surface.
            if not _MAYBE_FLAG_RE.search(text):
                continue
            try:
                tree = ast.parse(text, filename=rel)
            except SyntaxError:
                continue
            skip = _docstring_nodes(tree)
            bare_ok = _harvests_bare_constants(rel)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    fname = fn.attr if isinstance(fn, ast.Attribute) else (
                        fn.id if isinstance(fn, ast.Name) else "")
                    if fname == "add_argument":
                        for a in node.args:
                            if (isinstance(a, ast.Constant) and isinstance(a.value, str)
                                    and a.value.startswith("--")):
                                flags.add(a.value)
                elif (bare_ok and isinstance(node, ast.Constant)
                        and isinstance(node.value, str)
                        and id(node) not in skip
                        and _FLAG_CONSTANT_RE.match(node.value)):
                    flags.add(node.value)

    for glob in _LITERAL_GLOBS:
        for p in _REPO_ROOT.glob(glob):
            if "/target/" in str(p):
                continue
            text = p.read_text(errors="replace")
            flags.update(m.group(1) for m in _LITERAL_FLAG_RE.finditer(text))
            if p.suffix == ".sh":
                flags.update(m.group(1) for m in _SH_CASE_FLAG_RE.finditer(text))

    return frozenset(f.replace("_", "-") for f in flags)


# Flags belonging to tools this repo INVOKES but does not own. Each names its tool, so the list
# cannot quietly become a dumping ground for our own dead flags.
_EXTERNAL_TOOL_FLAGS: Dict[str, str] = {
    "--collect-only": "pytest",
    "--gate": "the usage-limits skill's check_usage.py",
    "--durations": "pytest",
    "--dist": "pytest-xdist",
    "--select": "ruff",
    "--exclude": "ruff",
    "--extra-index-url": "pip",
    "--manifest-path": "cargo",
    "--no-fail-fast": "cargo test",
    "--init": "git submodule",
    "--oneline": "git log",
    "--git-common-dir": "git rev-parse",
    "--sort": "ps -eo",
    "--no-security": "pokemon-showdown (the vendored server)",
    "--headless": "chrome",
    "--disable-web-security": "chrome",
    "--user-data-dir": "chrome",
    "--window-size": "chrome",
    "--dump-dom": "chrome",
    "--force-dark-mode": "chrome",
    "--password-store": "chrome",
    "--virtual-time-budget": "chrome",
    "--accent": "chrome",
    "--ff-only": "git pull / git merge",
    "--rebase": "git pull",
    "--prune": "conda env update",
    "--full": "pgrep (the hook that refuses it)",
    "--logdir": "tensorboard",
    "--native": "py-spy",
    "--query-gpu": "nvidia-smi",
    "--strict-equality": "mypy",
    "--weekly-threshold": "~/.claude/skills/usage-limits/check_usage.py (a user skill, not this repo)",
}

# Metasyntactic flag names: prose ABOUT flags rather than a flag anyone can pass.
_PROSE_PLACEHOLDER_FLAGS: Dict[str, str] = {
    "--flag": "stands for 'some flag' in pinned_argv_test's description",
    "--no-no-fork-pool-seed": "the name that would have been generated had the flag been "
                              "declared negatively — named to explain why it was not",
    "--dashed-form": "prose: 'grepped in both the --dashed-form and the underscored form'",
}

_DELETED_FLAGS_DOC = "designs/deleted_flags.md"
_HISTORY_FLAG_RE = re.compile(r"`(--[A-Za-z][A-Za-z0-9_-]*)`")
_HISTORY_PATH_RE = re.compile(r"`((?:" + "|".join(_OWNED_PREFIXES) + r")/[A-Za-z0-9_./\-]+)`")


def _history_doc_text() -> str:
    p = _REPO_ROOT / _DELETED_FLAGS_DOC
    return p.read_text() if p.is_file() else ""


def historical_flags() -> Set[str]:
    """The flag named in the FIRST CELL of each table row — never one mentioned in a note.

    Reading the whole file would let a row's prose ("the surviving split is `--compile-opponents`")
    silently grant history status to a live flag, which is the opposite of what this list is for.
    """
    out: Set[str] = set()
    for line in _history_doc_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) < 2:
            continue
        for f in _HISTORY_FLAG_RE.findall(cells[1]):
            out.add(f.replace("_", "-"))
    return out


def historical_paths() -> Set[str]:
    return set(_HISTORY_PATH_RE.findall(_history_doc_text()))


_DOC_FLAG_RE = re.compile(r"(?<![\w\-])--[A-Za-z][A-Za-z0-9_-]*")


def collect_flag_claims(docs: Optional[List[pathlib.Path]] = None) -> List[Tuple[str, int, str]]:
    claims = []
    for doc in (always_current_docs() if docs is None else docs):
        rel = _rel(doc)
        for lineno, line in enumerate(doc.read_text().splitlines(), 1):
            for m in _DOC_FLAG_RE.finditer(line):
                # `--anneal-lr-*` and `--win-prob-pbrs-…` name a FAMILY; the regex stops at the
                # glob, leaving a trailing dash.
                token = m.group(0).rstrip("-")
                claims.append((rel, lineno, token))
    return claims


def flag_resolves(flag: str, surface: Set[str], history: Set[str]) -> bool:
    n = flag.replace("_", "-")
    if n in surface or n in history:
        return True
    if n in _EXTERNAL_TOOL_FLAGS or n in _PROSE_PLACEHOLDER_FLAGS:
        return True
    # `--no-x` is generated by argparse's BooleanOptionalAction and this tree's `BoolFlag`, so it
    # never appears as a literal. Measured: every live option the AST scan missed was one of these.
    if n.startswith("--no-") and ("--" + n[5:]) in surface:
        return True
    # A prefix left by a family glob (`--anneal-lr-*`, `--zarch-*`): accept if it prefixes a
    # real flag OR a listed historical one — a family can be deleted as a family.
    if any(k.startswith(n + "-") for k in surface) or any(k.startswith(n + "-") for k in history):
        return True
    return False


# --------------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------------

@_SKIP
def test_every_claude_md_path_exists():
    history = historical_paths()
    misses = []
    for rel, lineno, token in collect_path_claims():
        if token in history:
            continue
        bases = _bases_for(rel, token)
        if (not bases and _is_upstream_source(token)) or any((base / token).exists() for base in bases):
            continue
        if unverifiable_by_design(rel, token):
            continue
        misses.append(f"  {rel}:{lineno}  {token}")

    assert not misses, (
        "A CLAUDE.md names a path that does not exist. Every one of these is read as fact by "
        "the next agent to load the file.\n\n"
        + "\n".join(sorted(misses))
        + "\n\nFix by CORRECTING the path. If the path is gone on purpose and the sentence is "
        f"telling the reader what USED to be there, list it in `{_DELETED_FLAGS_DOC}` with the "
        "entry that deleted it. If the token is prose rather than a path claim, add it to "
        "`_PROSE_NOT_A_PATH` WITH THE REASON."
    )


@_SKIP
def test_every_claude_md_flag_resolves():
    surface = cli_flag_surface()
    history = historical_flags()

    assert len(surface) > 300, (
        f"The CLI flag surface scan found only {len(surface)} flags, which is far below the "
        "~585 this tree carries — the scan is broken, and a broken scan reports every "
        "documented flag as dead. Check that `src/` is present and parseable."
    )

    misses: Dict[str, List[str]] = {}
    for rel, lineno, flag in collect_flag_claims():
        if flag_resolves(flag, surface, history):
            continue
        misses.setdefault(flag, []).append(f"{rel}:{lineno}")

    lines = [f"  {f:38s} {', '.join(sites[:3])}" for f, sites in sorted(misses.items())]
    assert not misses, (
        "A CLAUDE.md names a --flag that resolves in no parser in this tree. A documented flag "
        "that cannot be typed costs a launch-crash-fix loop (see `main.checkargs`).\n\n"
        + "\n".join(lines)
        + f"\n\nFix by correcting the spelling, or — if the flag is DELETED, DEMOTED off the CLI, "
        f"or PROPOSED but not yet built, and the sentence naming it is deliberate history — add "
        f"it to `{_DELETED_FLAGS_DOC}` with its citation. External tools' flags go in "
        "`_EXTERNAL_TOOL_FLAGS` NAMED WITH THEIR TOOL."
    )


@_SKIP
def test_the_history_doc_exists_and_is_not_a_dumping_ground():
    """`designs/deleted_flags.md` is the ONE escape hatch, so it has to stay legible.

    Two properties. It must exist (a missing history file turns the flag half of this gate into
    a wall of failures the next person routes around with a blanket skip). And every entry must
    carry a CITATION — the version or commit that deleted, demoted or proposed it — because an
    unexplained entry is indistinguishable from a flag someone silently gave up on, which is the
    c-family failure the root CLAUDE.md records.
    """
    doc = _REPO_ROOT / _DELETED_FLAGS_DOC
    assert doc.is_file(), (
        f"{_DELETED_FLAGS_DOC} is missing. It is the history list this gate reads; without it "
        "every deliberately-historical flag mention fails."
    )
    text = doc.read_text()
    rows = [ln for ln in text.splitlines()
            if ln.startswith("|") and _HISTORY_FLAG_RE.search(ln)]
    assert rows, f"{_DELETED_FLAGS_DOC} lists no flags — did its table format change?"

    uncited = [ln.split("|")[1].strip() for ln in rows
               if not re.search(r"v\d+|`gen3_[a-z0-9_]+`|20\d\d-\d\d-\d\d|[0-9a-f]{7,40}", ln)]
    assert not uncited, (
        "These entries name no citation (a version like `v88`, a signature like "
        "`gen3_dead_flag_purge_v1`, a date, or a commit sha):\n  "
        + "\n  ".join(uncited)
        + f"\n\nAn uncited entry in {_DELETED_FLAGS_DOC} is a claim nobody can check."
    )


@_SKIP
def test_the_history_doc_may_only_shrink():
    """A flag listed as history must no longer be in the live CLI surface.

    Same ratchet as `file_size_gate_test`'s grandfather list and the ruff handoff list: an entry
    that stops being true has to LEAVE, because a stale entry silently suppresses a real finding
    forever. If a flag comes back, delete its row.
    """
    surface = cli_flag_surface()
    resurrected = sorted(f for f in historical_flags() if f in surface)
    assert not resurrected, (
        "These flags are listed as history but EXIST in the live CLI surface again:\n  "
        + "\n  ".join(resurrected)
        + f"\n\nRemove their rows from {_DELETED_FLAGS_DOC} — while they are listed, the gate "
        "cannot notice if they are deleted a second time."
    )


# --------------------------------------------------------------------------------------------
# (c) MODULES — every `python -m <module>` must be runnable
# --------------------------------------------------------------------------------------------

# Third-party modules a CLAUDE.md runs with `-m`. Each names what it is, so the list cannot quietly absorb one of
# our own dead modules.
_EXTERNAL_MODULES: Dict[str, str] = {
    "pytest": "the test runner",
    "mypy": "the type checker (the mypy gate)",
}

# `python -m x`, `python3.11 -u -m x`, `"$PY" -m x`, `${GEN3AI_PYTHON} -m x`. The trailing group catches a
# placeholder (`main.<meter>`); `…` never matches the name at all. There is no history escape (unlike paths
# and flags): a sentence about a deleted module names it without `python -m`.
_MODULE_RE = re.compile(r"(?:\bpython3?(?:\.\d+)?|\$\{?(?:PY|GEN3AI_PYTHON)\}?\"?)(?:\s+-[A-Za-z]+)*?\s+-m\s+"
                        r"([A-Za-z_][\w.]*)([<{*]?)")

_MODULE_BASES = ("src", "")   # `src/` (the package root) and the repo root (`tools.…`, run from it)


def collect_module_claims(docs: List[pathlib.Path]) -> List[Tuple[str, int, str]]:
    claims = []
    for doc in docs:
        rel = _rel(doc)
        for lineno, line in enumerate(doc.read_text().splitlines(), 1):
            for m in _MODULE_RE.finditer(line):
                if m.group(2) or m.group(1).endswith("."):   # a placeholder, not a module claim
                    continue
                claims.append((rel, lineno, m.group(1)))
    return claims


def module_resolves(module: str, root: pathlib.Path = _REPO_ROOT) -> bool:
    """Is `python -m module` runnable from `root`? A plain module file, or a package WITH a `__main__.py`."""
    if module.split(".")[0] in _EXTERNAL_MODULES:
        return True
    for base in _MODULE_BASES:
        target = root / base / pathlib.Path(*module.split("."))
        if target.with_suffix(".py").is_file() or (target / "__main__.py").is_file():
            return True
    return False


@_SKIP
def test_every_claude_md_python_module_resolves():
    claims = collect_module_claims(always_current_docs())
    assert len(claims) > 20, f"only {len(claims)} `python -m` claims found: the scan is broken"
    misses = [f"  {rel}:{lineno}  python -m {mod}" for rel, lineno, mod in claims if not module_resolves(mod)]
    assert not misses, (
        "A CLAUDE.md runs a `python -m` target that is not runnable in this tree (no `<module>.py`, and no "
        "package with a `__main__.py`):\n\n" + "\n".join(misses)
        + "\n\nFix by correcting the module path. A third-party tool goes in `_EXTERNAL_MODULES` NAMED WITH "
        "WHAT IT IS."
    )


def test_meta_the_module_check_has_teeth(tmp_path):
    """Revert-proof: a missing module, and a package with no `__main__.py`, FAIL; a placeholder is not a claim."""
    (tmp_path / "src" / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("")
    (tmp_path / "src" / "pkg" / "tool.py").write_text("")
    (tmp_path / "src" / "pkg" / "sub" / "__init__.py").write_text("")
    assert module_resolves("pkg.tool", tmp_path)
    assert not module_resolves("pkg.gone", tmp_path)
    assert not module_resolves("pkg.sub", tmp_path), "a package without __main__.py cannot be run with -m"
    (tmp_path / "src" / "pkg" / "sub" / "__main__.py").write_text("")
    assert module_resolves("pkg.sub", tmp_path)
    assert module_resolves("pytest", tmp_path)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "fetch.py").write_text("")
    assert module_resolves("tools.fetch", tmp_path), "the repo root is a base too"

    doc = tmp_path / "CLAUDE.md"
    doc.write_text('run `python -m pkg.tool`, `"$PY" -m pkg.gone --x`, `python3 -m main.<meter>`, '
                   "`python -m …`, `pytest -m slow`, `${PY} -m pkg.sub`, `python -m main.ops.*`, "
                   "`python -m a.{b,c}`, `python3.11 -u -m tools.fetch`, `$GEN3AI_PYTHON -m pkg.tool`.\n")
    assert [c[2] for c in collect_module_claims([doc])] == [
        "pkg.tool", "pkg.gone", "pkg.sub", "tools.fetch", "pkg.tool"]


def test_meta_the_scan_scope_reaches_every_always_current_doc():
    """Revert-proof: the docs the root `CLAUDE.md` calls always-current are SCANNED (it once stopped at `CLAUDE.md`,
    and `designs/endstate/` named deleted flags and modules unchecked); the HISTORY trees are not."""
    scanned = {_rel(p) for p in always_current_docs()}
    must = {"CLAUDE.md", "README.md", "designs/ARCHITECTURE.md", "designs/ops/ORCHESTRATOR_SOP.md",
            "designs/ops/TRAINING_RUN_SOP.md", "designs/ops/EXTERNAL_ANCHORS_SOP.md", "designs/ops/training_runbook.md",
            "designs/ops/testing.md", "docs/DEVELOPING.md", "src/rust_sim/README.md",
            "designs/endstate/README.md", "designs/endstate/program_rust_core.md",
            # the seven leaf-detail trees (one doc of each): their stale paths, flags and modules were once unchecked
            "designs/training/compile_flags.md", "designs/model/versioning.md", "designs/rust_sim/module_map.md",
            "designs/prober/web_service.md", "designs/launcher/README.md", "designs/observation/README.md",
            "designs/tools/README.md"}
    assert must <= scanned, sorted(must - scanned)
    assert {t for r in scanned for t in budget.TREE_DIRS if r.startswith(f"designs/{t}/")} == set(budget.TREE_DIRS)
    assert any(r.startswith("designs/endstate/") and r.endswith("search_ledger.md") for r in scanned)
    history = [r for r in scanned if "/measurements/" in r or "/ai_v3/" in r or "claude_md_archive" in r
               or r.startswith(("deps/", ".claude/")) or r == "designs/CHANGELOG.md"]
    assert not history, history


def test_meta_a_non_claude_md_doc_is_checked_for_paths_flags_and_modules(tmp_path):
    """Revert-proof: the three claim collectors read ANY doc handed to them, and the gate's verdict on each claim is
    the same one it gives a `CLAUDE.md`."""
    doc = tmp_path / "design_x.md"
    doc.write_text("see `src/gone/never_was.py`, run `--no-such-flag-xyz`, and `python -m main.gone_module_xyz`.\n"
                   "upstream `data/moves.ts`, a dir `src/main/rust_core_m5/`.\n")
    assert [c[2] for c in collect_path_claims([doc])] == ["src/gone/never_was.py", "data/moves.ts", "src/main/rust_core_m5"]
    assert not any((base / "src/gone/never_was.py").exists() for base in _bases_for(str(doc), "src/gone/never_was.py"))
    assert [c[2] for c in collect_flag_claims([doc])] == ["--no-such-flag-xyz"]
    assert not flag_resolves("--no-such-flag-xyz", set(), set())
    assert [c[2] for c in collect_module_claims([doc])] == ["main.gone_module_xyz"]
    assert not module_resolves("main.gone_module_xyz")


def test_meta_a_rust_sim_detail_doc_may_cite_the_crate_layout_and_nothing_else_may():
    """Revert-proof: `designs/rust_sim/*.md` resolves a crate-relative path (`src/turn.rs`), another tree does not."""
    crate = _REPO_ROOT / "src" / "rust_sim"
    in_crate = next(p for p in sorted(crate.glob("src/*.rs")))
    token = in_crate.relative_to(crate).as_posix()   # e.g. `src/turn.rs`
    assert any((b / token).exists() for b in _bases_for("designs/rust_sim/module_map.md", token))
    assert not any((b / token).exists() for b in _bases_for("designs/training/compile_flags.md", token))
    assert not any((b / token).exists() for b in _bases_for("designs/rust_sim.md", token))   # a prefix, not a substring


def test_meta_a_git_ignored_path_is_unverifiable_by_design_and_a_tracked_one_is_not(tmp_path):
    """Revert-proof (the fresh-worktree defect): a doc citing a build output git ignores (`target/...`, absent in a
    worktree with no Rust build) must not fail the gate, while a missing TRACKED path still must."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("/target\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / ".gitignore").write_text("/rust_env/target\n")
    assert _git_ignored("target/selfcheck/lib.so", tmp_path)
    assert _git_ignored("src/rust_env/target/selfcheck/libpokesim_env.so", tmp_path)
    assert not _git_ignored("src/rust_env/src/gone.rs", tmp_path)
    assert not _git_ignored("designs/gone.md", tmp_path)
    # outside any repo: strict (cannot tell => not ignored)
    assert not _git_ignored("target/x", pathlib.Path("/"))


def test_the_citation_that_broke_a_fresh_worktree_is_skipped_by_the_live_gate():
    """The live path the defect hit: `src/rust_env/target/...so` is ignored in THIS repo, so the verdict does not
    depend on whether this checkout built it."""
    token = "src/rust_env/target/selfcheck/libpokesim_env.so"
    assert unverifiable_by_design("designs/prober/session_scans.md", token)
    assert not unverifiable_by_design("designs/prober/session_scans.md", "src/rust_env/src/never_was.rs")


def test_meta_an_upstream_ts_citation_resolves_against_the_submodule_only():
    assert _is_upstream_source("data/mods/gen4/items.ts")
    assert not _is_upstream_source("data/pokemon/gen3_pokedex.json")
    assert not _is_upstream_source("src/foo.ts")
    submodule = _REPO_ROOT / _UPSTREAM_BASE
    assert _bases_for("designs/ARCHITECTURE.md", "data/moves.ts") == (
        [submodule] if (submodule / "data").is_dir() else [])
    assert _bases_for("designs/ARCHITECTURE.md", "data/moves.json")[0] == _REPO_ROOT


# --------------------------------------------------------------------------------------------
# (d) BYTE BUDGETS — the ratchet (rules: `utils.instruction_budget`)
# --------------------------------------------------------------------------------------------

def _fix_hint(problems: List[str]) -> str:
    return (
        "\n".join("  " + p for p in problems)
        + f"\n\nThe budget is principles §3 (`designs/ops/instruction_docs_principles.md`); the numbers are "
        f"`{budget.BUDGET_FILE}`. A grown file must SHRINK (replace, don't append: the deletion test is "
        "principle 1). `python -m utils.instruction_budget` prints the census."
    )


@_SKIP
def test_repo_instruction_docs_respect_the_byte_budget():
    b = budget.load_budget()
    docs = budget.repo_docs(_REPO_ROOT)
    assert {"CLAUDE.md", ".claude/commands/gen3ai-ship.md"} <= set(docs), f"the doc scan is broken: {sorted(docs)}"
    problems = budget.check(docs, b["type_caps"], b["grandfathered"])
    assert not problems, "An instruction doc broke its byte budget:\n\n" + _fix_hint(problems)


@_SKIP
def test_user_instruction_files_respect_the_byte_budget():
    """`MEMORY.md` and `~/.claude/agents/*.md` are machine-local (outside git): checked only where they exist.
    A failure here is NOT your diff — the orchestrator owns these files: report it as a FINDING, and only then
    opt this one check out with `GEN3AI_SKIP_USER_DOC_BUDGET=1`."""
    if os.environ.get("GEN3AI_SKIP_USER_DOC_BUDGET") == "1":
        pytest.skip("GEN3AI_SKIP_USER_DOC_BUDGET=1 (a machine-local file is over budget and its owner knows)")
    docs = budget.user_docs()
    if not docs:
        pytest.skip(f"no user instruction files on this machine (looked for the auto-memory index and "
                    f"{budget.claude_config_dir() / 'agents'}/*.md)")
    b = budget.load_budget()
    problems = budget.check(docs, b["type_caps"], b["grandfathered"], user_scope=True)
    assert not problems, ("A machine-local instruction file broke its byte budget (its owner is the orchestrator "
                          "or the owner, not this diff):\n\n" + _fix_hint(problems))


#: The caps the meta tests below use (any consistent set would do).
_CAPS = {"root": 16000, "leaf": 12000, "tree": 120000, "command": 12000, "memory": 10000, "agent": 2500}

#: The rows of principles §3's table, by kind (`designs/CLAUDE.md` is a leaf).
_PRINCIPLES_ROWS = {"root `CLAUDE.md`": "root", "`MEMORY.md`": "memory", "a code leaf `CLAUDE.md`": "leaf",
                    "`designs/CLAUDE.md` (a map)": "leaf", "a `designs/<topic>/` tree doc": "tree",
                    "a subagent definition file": "agent", "a slash command": "command"}


def test_the_budget_file_matches_the_principles_table():
    """The budget file's type caps and targets are principles §3's table, row by row (1 KB = 1,000 bytes)."""
    b = budget.load_budget()
    text = (_REPO_ROOT / "designs" / "ops" / "instruction_docs_principles.md").read_text()
    section = text[text.index("## 3. Budgets"):text.index("## 4.")]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] in _PRINCIPLES_ROWS:
            target, cap = (int(float(re.match(r"[\d.]+", c).group()) * 1000) for c in cells[1:])
            rows[cells[0]] = (target, cap)
    assert set(rows) == set(_PRINCIPLES_ROWS), f"principles §3's table rows changed: {sorted(rows)}"
    for label, (target, cap) in rows.items():
        kind = _PRINCIPLES_ROWS[label]
        assert (b["type_targets"][kind], b["type_caps"][kind]) == (target, cap), label
    assert set(b["type_caps"]) == set(b["type_targets"]) == set(_PRINCIPLES_ROWS.values())


def _leaf(size: int) -> "budget.Doc":
    return budget.Doc("leaf", size)


def test_meta_a_tree_doc_is_held_to_its_own_cap():
    assert budget.check({"designs/x/a.md": budget.Doc("tree", 120001)}, _CAPS, {})[0].startswith("OVER BUDGET")
    assert budget.check({"designs/x/a.md": budget.Doc("tree", 120000)}, _CAPS, {}) == []
    assert budget.check({"designs/x/a.md": _leaf(120000)}, _CAPS, {})[0].startswith("OVER BUDGET")   # a leaf is 12 KB


def test_meta_a_new_file_over_its_type_cap_fails_and_one_at_it_passes():
    assert budget.check({"a/CLAUDE.md": _leaf(12001)}, _CAPS, {})[0].startswith("OVER BUDGET")
    assert budget.check({"a/CLAUDE.md": _leaf(12000)}, _CAPS, {}) == []
    assert budget.check({"CLAUDE.md": budget.Doc("root", 16001)}, _CAPS, {})[0].startswith("OVER BUDGET")


def test_meta_a_grandfathered_file_may_not_grow_a_byte():
    assert budget.check({"a/CLAUDE.md": _leaf(20000)}, _CAPS, {"a/CLAUDE.md": 20000}) == []
    assert budget.check({"a/CLAUDE.md": _leaf(20001)}, _CAPS, {"a/CLAUDE.md": 20000})[0].startswith("OVER BUDGET")


def test_meta_a_shrunk_grandfathered_file_must_lower_its_cap():
    g = {"a/CLAUDE.md": 20000}
    assert budget.check({"a/CLAUDE.md": _leaf(20000 - budget.LOWER_SLACK)}, _CAPS, g) == []
    assert budget.check({"a/CLAUDE.md": _leaf(20000 - budget.LOWER_SLACK - 1)}, _CAPS, g)[0].startswith("LOWER")


def test_meta_a_grandfathered_file_back_under_its_type_cap_must_leave_the_list():
    assert budget.check({"a/CLAUDE.md": _leaf(12000)}, _CAPS, {"a/CLAUDE.md": 12500})[0].startswith("STALE")


def test_meta_a_user_file_is_held_to_its_cap_only():
    g = {budget.MEMORY_KEY: 12000}
    over = budget.check({budget.MEMORY_KEY: budget.Doc("memory", 12001)}, _CAPS, g, user_scope=True)
    assert over and over[0].startswith("OVER BUDGET")
    assert budget.check({budget.MEMORY_KEY: budget.Doc("memory", 9000)}, _CAPS, g, user_scope=True) == []
    assert budget.check({budget.MEMORY_KEY: budget.Doc("memory", 10001)}, _CAPS, {}, user_scope=True)


def test_meta_a_grandfathered_repo_entry_for_a_missing_file_fails_but_a_user_one_does_not():
    g = {"gone/CLAUDE.md": 20000, budget.MEMORY_KEY: 12000}
    assert budget.check({}, _CAPS, g) == [
        f"MISSING FILE: {budget.BUDGET_FILE} grandfathers gone/CLAUDE.md, which does not exist. Remove the entry."]
    assert budget.check({}, _CAPS, g, user_scope=True) == []


def test_meta_the_scans_find_each_kind(tmp_path):
    (tmp_path / "x" / "deps").mkdir(parents=True)
    (tmp_path / ".claude" / "commands").mkdir(parents=True)
    (tmp_path / ".claude" / "worktrees" / "w").mkdir(parents=True)
    (tmp_path / "designs" / "training").mkdir(parents=True)
    (tmp_path / "designs" / "ops").mkdir()
    for rel in ("CLAUDE.md", "x/CLAUDE.md", "x/deps/CLAUDE.md", ".claude/commands/c.md",
                ".claude/worktrees/w/CLAUDE.md", "designs/training/t.md", "designs/ops/o.md"):
        (tmp_path / rel).write_text("abc")
    assert budget.repo_docs(tmp_path) == {"CLAUDE.md": budget.Doc("root", 3), "x/CLAUDE.md": budget.Doc("leaf", 3),
                                          "designs/training/t.md": budget.Doc("tree", 3),   # `ops/` is not a tree
                                          ".claude/commands/c.md": budget.Doc("command", 3)}

    main = pathlib.Path("/w/my.repo")
    config = tmp_path / "cfg"
    assert budget.user_docs(main, config) == {}
    memory = budget.memory_index_path(main, config)
    assert memory == config / "projects" / "-w-my-repo" / "memory" / "MEMORY.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("12345")
    (config / "agents").mkdir()
    (config / "agents" / "a.md").write_text("12")
    assert budget.user_docs(main, config) == {budget.MEMORY_KEY: budget.Doc("memory", 5),
                                              budget.AGENT_KEY_PREFIX + "a.md": budget.Doc("agent", 2)}


# --------------------------------------------------------------------------------------------
# (e) SECTION POINTERS — `<file> § <section>` must land on a section (`utils.section_refs`)
# --------------------------------------------------------------------------------------------

@_SKIP
def test_every_section_pointer_resolves():
    bad, counts = section_refs.scan(_REPO_ROOT)
    assert counts["checked"] > 500, f"only {counts['checked']} section pointers checked: the scan is broken"
    assert not bad, (
        "A doc or code comment points at a section that does not exist (`python -m utils.section_refs` lists "
        "them):\n\n"
        + "\n".join(f"  {r.source}:{r.line}  {r.target} § {r.what}  -- {why}" for r, why in bad)
        + "\n\nRepoint the pointer to the section's NEW home (grep the heading's words; a section that moved to "
        "`designs/<tree>/` is cited there), or state the fact instead of citing a section that is gone."
    )


def _scan(files):
    return section_refs.scan_files(sorted(files), files.__getitem__)


_REF_DOC = "# Top\n\n## 4. The four\n\n### 4.2 Sub\n\n## Copying text\n\nA **bold lead phrase here.** line\n\n## (c) The byte test\n"


def test_meta_the_section_pointer_check_has_teeth():
    """Revert-proof: each pointer form FAILS when its section is missing and PASSES when it exists."""
    files = {"d/target.md": _REF_DOC, "d/js/probe.js": "// B. The CATEGORY FLIP\n// body\n"}
    good = ("`d/target.md` §4.2, `target.md` § 4, `target.md` §c, `target.md` § \"Copying text\", "
            "`target.md` → \"Copying text\", the \"Copying text\" section of `d/target.md`, "
            "\"Copying text\" in `target.md`, `target.md` § Copying, `d/js/probe.js` section B, "
            "`target.md` § \"bold lead phrase\" (a phrase in the body)")
    bad = ("`d/target.md` §5.1, `target.md` § 9, `target.md` §z, `target.md` § \"Gone away\", "
           "`target.md` → \"Gone away\", \"Gone away\" in `target.md`, `d/js/probe.js` section Q")
    dangling, counts = _scan({**files, "d/good.md": good, "d/bad.md": bad})
    assert [(r.source, r.what) for r, _ in dangling if r.source == "d/good.md"] == []
    got = sorted(r.what for r, _ in dangling if r.source == "d/bad.md")
    assert got == sorted(["5.1", "9", "z", '"Gone away"', "Q"]), got   # one row per distinct pointer
    assert counts["checked"] >= 12


def test_meta_the_section_pointer_check_skips_what_it_cannot_judge():
    files = {
        "d/target.md": _REF_DOC,
        "d/amb/target.md": "## Other\n",
        # a bare name with two homes passes when ANY has the section; a missing file FAILS
        "d/a.md": "`target.md` § 4 and `nowhere.md` § 4",
        # a hard-wrapped comment is joined, so a title that wraps still resolves
        "d/c.rs": '//! see "Copying\n//! text" in `d/target.md`\n',
        # history as a SOURCE is not checked; a `§` in a code string literal is not a pointer
        "designs/CHANGELOG.md": "`target.md` § 99",
        "d/t.py": 'x = f"{name}.md § " + y\n',
        # the section belongs to the file named AFTER it
        "d/b.md": "`target.md`, §2.1 of `d/amb/target.md` and \"Other\" in `d/amb/target.md`",
    }
    dangling, _ = _scan(files)
    assert [(r.source, r.target) for r, _ in dangling] == [("d/a.md", "nowhere.md")], dangling
