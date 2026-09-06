"""Do the CLAUDE.md files still describe THIS tree? The static gate for DOCUMENTATION.

Sits beside `ruff_gate_test.py`, `file_size_gate_test.py` and `strict_api_lock_test.py` at the
`src/` root because, like those, its subject is the whole tree rather than any one package.

**Why it exists.** Every `CLAUDE.md` in this repo is loaded into an agent's context and read as
fact. A path that has moved and a flag that has been deleted do not merely go stale — they are
*actively believed*, and the believer then reports a false result or types a command that cannot
launch. That failure has a track record here: `main.checkargs` exists because a deleted flag in a
recorded argv costs a launch-crash-fix loop, and the root `CLAUDE.md`'s own c-family lesson is
that an allowlist entry can outlive its own fix and mislead every reader after it. Prose has no
compiler, so it needs a gate.

**Two checks, both mechanical.** Neither is a judgement about writing.

1. **Every repo-relative path a CLAUDE.md names must EXIST.** Scope is the five owned prefixes
   (`src/`, `designs/`, `data/`, `tools/`, `scripts/`) — `deps/` is a submodule plus untracked
   build output, so its contents are not a promise this tree can keep in a fresh worktree.
2. **Every `--flag` a CLAUDE.md names must RESOLVE somewhere in the tree's own CLI surface** —
   or be listed in `designs/deleted_flags.md`, the history file for flags a document names on
   purpose because it is telling you what used to be true.

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
checks the two things that have a referent in the filesystem: names of files, and names of flags.

**Cost: well under a second** — it reads ~13 markdown files and AST-parses the tree once. No cost
marker, runs in the fast inner loop. Opt out explicitly:

    GEN3AI_SKIP_CLAUDE_MD_GATE=1 pytest src/ -q
"""
from __future__ import annotations

import ast
import functools
import os
import pathlib
import re
from typing import Dict, List, Set, Tuple

import pytest

from utils.paths import repo_root

_REPO_ROOT = repo_root()

_SKIP = pytest.mark.skipif(
    os.environ.get("GEN3AI_SKIP_CLAUDE_MD_GATE") == "1",
    reason="GEN3AI_SKIP_CLAUDE_MD_GATE=1",
)

# --------------------------------------------------------------------------------------------
# Which documents are in scope
# --------------------------------------------------------------------------------------------

# `deps/` is a git submodule whose working tree may be empty, and `.claude/worktrees/` holds other
# agents' checkouts of this same repo — scanning either would report another tree's state as this
# one's.
_EXCLUDED_DIR_PARTS = ("deps", ".claude", "node_modules", "target")


def claude_md_files() -> List[pathlib.Path]:
    out = []
    for p in _REPO_ROOT.rglob("CLAUDE.md"):
        rel = p.relative_to(_REPO_ROOT)
        if any(part in _EXCLUDED_DIR_PARTS for part in rel.parts[:-1]):
            continue
        out.append(p)
    return sorted(out)


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
_EXTRA_BASES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    # The Rust port's leaf cites (1) its own crate layout — `src/turn.rs` means
    # `src/rust_sim/src/turn.rs` — and (2) the upstream Showdown sources it was ported FROM,
    # which is the whole point of a port's evidence trail.
    "src/rust_sim/CLAUDE.md": (
        ("deps/pokemon-showdown", "the upstream Showdown data/ the port was derived from"),
        ("deps/pokemon-showdown/dist", "the compiled upstream, cited by line number"),
    ),
}

# Tokens that MATCH the path regex but are prose, not path claims. Each needs a reason; the list
# may only shrink. Do not park a real broken path here.
_PROSE_NOT_A_PATH: Dict[str, str] = {
    # "regenerate after any data/derivation change" — English ("the data derivation"), not a file.
    "data/derivation": "prose: 'any data/derivation change' means the derivation step, not a file",
}


def _is_placeholder(token: str) -> bool:
    return (any(m in token for m in _PLACEHOLDER_MARKS)
            or bool(_PLACEHOLDER_SEGMENTS.search(token)))


def _bases_for(rel_doc: str) -> List[pathlib.Path]:
    """Where a path named in `rel_doc` is allowed to resolve, in order."""
    bases = [_REPO_ROOT, _REPO_ROOT / pathlib.PurePosixPath(rel_doc).parent]
    for extra, _reason in _EXTRA_BASES.get(rel_doc, ()):
        base = _REPO_ROOT / extra
        # An uninitialised submodule is not a documentation defect. Skip the base rather than
        # failing every citation into it.
        if base.exists():
            bases.append(base)
    return bases


def collect_path_claims() -> List[Tuple[str, int, str]]:
    """Every (doc, line, path) a CLAUDE.md asserts exists."""
    claims = []
    for doc in claude_md_files():
        rel = str(doc.relative_to(_REPO_ROOT))
        for lineno, line in enumerate(doc.read_text().splitlines(), 1):
            for m in _PATH_RE.finditer(line):
                token = m.group(1).rstrip(".,;:)`'\"")
                if _is_placeholder(token) or token in _PROSE_NOT_A_PATH:
                    continue
                claims.append((rel, lineno, token))
    return claims


# --------------------------------------------------------------------------------------------
# (b) FLAGS — the tree's own CLI surface
# --------------------------------------------------------------------------------------------

_PY_ROOTS = ("src", "tools")
_PY_EXCLUDED = ("src/poke_env",)  # a vendored fork; its CLI is not ours to document

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
            if rel.startswith(_PY_EXCLUDED):
                continue
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
    "--durations": "pytest",
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
    "--accent": "chrome",
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


def collect_flag_claims() -> List[Tuple[str, int, str]]:
    claims = []
    for doc in claude_md_files():
        rel = str(doc.relative_to(_REPO_ROOT))
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
        if any((base / token).exists() for base in _bases_for(rel)):
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
