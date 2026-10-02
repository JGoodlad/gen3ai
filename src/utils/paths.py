"""Path discovery — the ONE place that knows how deep a module sits in the tree.

Before this module the depth arithmetic (`Path(__file__).resolve().parents[3]`,
`os.path.dirname(os.path.dirname(os.path.dirname(...)))`) was hand-written at ~25 sites. Every
copy is individually correct and collectively fragile: moving a file one directory changes the
right answer, and nothing tells you which of the 25 you just broke. Here the arithmetic exists
once, and :func:`repo_root` is cross-checked against ``git rev-parse`` by ``paths_test.py``.

**Three questions, three answers, and they are NOT interchangeable.**

===========================================  =============================  ==================
Question                                     Helper                         Mechanism
===========================================  =============================  ==================
Where is the checkout this code came from?   :func:`repo_root` /            ``__file__``
                                             :func:`repo_path`
Where is ``src/``?                           :func:`src_root` /             ``__file__``
                                             :func:`src_path`
Where is ``models/`` (the run archive)?      :func:`main_models_dir`        ``git``
Where does a run WRITE (create / resolve)?   :func:`run_archive_dir`,       ``git`` + a refusal
                                             :func:`checked_run_dir`
===========================================  =============================  ==================

The third is a different question and it is the one that keeps biting. **In a git worktree
:func:`repo_root` is the WORKTREE**, which has no ``models/`` — the run archive lives only in the
main checkout. A test that wants an archived run must reach across, which is what
:func:`main_models_dir` does, via ``utils.git.get_main_repo_root()`` — the same
``--git-common-dir`` logic the launcher uses to find the main checkout. Getting this wrong is
invisible: the directory is simply absent and a skip-if-missing test skips forever.

**The fourth question is the one that destroyed runs.** :func:`main_models_dir` is a READER's
accessor (``None`` = skip). A run that is being CREATED is a different caller: it must never fall
back to a cwd-relative ``models/``, because inside a linked worktree that directory is deleted
SILENTLY with the worktree (2026-09-23 eight runs; 2026-10-01/02 three agents' run data nearly).
:func:`run_archive_dir` is the ONE answer for "where does a run directory land" — in the main
checkout ``./models`` (nothing changes), from a worktree the MAIN checkout's ``models/``, anywhere
``$GEN3AI_MODELS_DIR`` when set — and :func:`checked_run_dir` is the ONE refusal for an explicit
run dir that would land inside a linked worktree's own ``models/``.

**Why the first two are ``__file__``-relative and not git.** They are used at import time in
production modules, and they must work in a checkout with no ``.git`` at all (a source tarball, a
container COPY). ``utils.git.get_repo_root()`` shells out to git and answers the same question a
different way; it stays for callers that specifically want git's opinion.

**When ``__file__``-relative is still the RIGHT answer and this module is the wrong tool:** a
module locating a file that ships *beside it* (``Path(__file__).parent / "local_sim_bridge.js"``)
is not doing repo-root discovery at all, and routing it through the repo root would make a local
fact depend on a global one. Leave those alone.
"""
import os
from pathlib import Path
from typing import Optional

#: The ``src/`` directory — this file is ``src/utils/paths.py``, so ``parents[1]``.
_SRC_ROOT = Path(__file__).resolve().parents[1]

#: The checkout root that contains ``src/``. In a git worktree this is the WORKTREE root.
_REPO_ROOT = _SRC_ROOT.parent

#: Env var that pins the run archive explicitly, for a checkout whose ``models/`` lives elsewhere.
MODELS_DIR_ENV_VAR = "GEN3AI_MODELS_DIR"

#: Env var that pins the harvest artifact archive — see :func:`harvest_dir`.
HARVEST_DIR_ENV_VAR = "GEN3AI_HARVEST_DIR"

#: Set by the root ``conftest.py`` for the whole pytest process tree: while it is set and
#: ``$GEN3AI_MODELS_DIR`` is NOT, :func:`run_archive_dir` REFUSES to name the real archive. A test
#: that creates a run directory must point ``$GEN3AI_MODELS_DIR`` at a tmp dir, so a test that
#: forgot FAILS instead of writing into the owner's run archive. Not a production switch.
RUN_ARCHIVE_SEAL_ENV_VAR = "GEN3AI_RUN_ARCHIVE_SEALED"


def repo_root() -> Path:
    """The checkout root — the directory holding ``src/``, ``data/``, ``designs/``, ``deps/``.

    Inside a git worktree this is the **worktree** root, matching
    ``utils.git.get_repo_root()``. It is derived from ``__file__``, so it needs no git and no
    subprocess and is correct at import time.
    """
    return _REPO_ROOT


def src_root() -> Path:
    """The ``src/`` directory — the import root for ``agents`` / ``main`` / ``utils``."""
    return _SRC_ROOT


def repo_path(*parts: str) -> Path:
    """``repo_root()`` joined with ``parts``. The path need not exist."""
    return _REPO_ROOT.joinpath(*parts)


def src_path(*parts: str) -> Path:
    """``src_root()`` joined with ``parts``. The path need not exist."""
    return _SRC_ROOT.joinpath(*parts)


def main_models_dir() -> Optional[Path]:
    """The **main checkout's** ``models/`` run archive, or ``None`` if there is no archive here.

    ``models/`` is not committed and exists only on a machine that has actually trained. It also
    exists only in the MAIN checkout — a git worktree does not have one — so this resolves the
    main checkout via ``git rev-parse --git-common-dir`` rather than :func:`repo_root`.

    Returning ``None`` rather than a non-existent path is deliberate: every caller is a test that
    must SKIP when the archive is absent (a fresh contributor clone has no ``models/``), and a
    ``None`` cannot be silently joined into a path that then globs to nothing.

    ``$GEN3AI_MODELS_DIR`` overrides and is AUTHORITATIVE — when it is set and is not a
    directory the answer is ``None``, never a quiet fall-back to somewhere else. That is what
    makes it usable as the seam the skip path is tested through (point it at an empty directory
    and every caller must take its skip).
    """
    override = os.environ.get(MODELS_DIR_ENV_VAR)
    if override:
        cand = Path(override)
        return cand if cand.is_dir() else None

    from utils.git import get_main_repo_root
    try:
        # Anchored at THIS file's checkout, so the answer does not depend on the caller's cwd.
        cand = Path(get_main_repo_root(cwd=str(_REPO_ROOT))) / "models"
        if cand.is_dir():
            return cand
    except Exception:
        pass  # not a git checkout (source tarball / container COPY) — fall through
    # Last resort: an archive beside THIS checkout. Covers a non-git tree, and anyone who
    # trains inside a worktree rather than the main checkout.
    local = repo_path("models")
    return local if local.is_dir() else None


# --------------------------------------------------------------------------------------------
# Where a run WRITES — the CREATING side of the archive (never a cwd-relative ``models/``)
# --------------------------------------------------------------------------------------------

def _archive_anchor() -> Path:
    """The checkout the run archive is asked FROM — the one this code came from.

    A function (not a bare ``_REPO_ROOT`` read) so a test can stand "in a linked worktree" for the
    archive questions ONLY, without moving :func:`repo_root` — which every ``data/`` and
    ``designs/`` read depends on. Production never rebinds it."""
    return _REPO_ROOT


class RunArchiveError(ValueError):
    """A run directory cannot be placed safely — a typed FATAL_CONFIG refusal.

    ``exit_code`` is ``TrainExitCode.FATAL_CONFIG`` (3; pinned by ``paths_test``), and
    ``main.exit_codes._FATAL_BY_NAME`` maps this class by NAME, so it exits 3 wherever it is raised
    — the launcher does not restart it, because a restart would meet the same refusal."""

    exit_code = 3


def is_linked_worktree(root: Optional[Path] = None) -> bool:
    """True when ``root`` (default: this checkout) is a LINKED git worktree.

    A linked worktree's ``.git`` is a FILE (``gitdir: <main>/.git/worktrees/<name>``); the main
    checkout and an ordinary clone have a ``.git`` DIRECTORY, a source tarball has none. No
    subprocess, so it is correct when git itself is the thing that failed."""
    return (Path(root) if root is not None else _REPO_ROOT).joinpath(".git").is_file()


def _main_checkout_of(here: Path) -> Optional[Path]:
    from utils.git import get_main_repo_root
    try:
        return Path(get_main_repo_root(cwd=str(here)))
    except Exception:
        return None


def run_archive_dir(anchor: Optional[Path] = None) -> Path:
    """The directory a run is CREATED in and RESOLVED against: ``<archive>/<run>``.

    1. ``$GEN3AI_MODELS_DIR`` when set — AUTHORITATIVE, and it must be an existing directory
       (set-but-missing is a refusal, never a quiet fall-back, never a typo that mints a tree).
    2. otherwise the MAIN checkout's ``models/`` via git's common dir. In the main checkout that is
       ``./models`` — nothing changes, and a first run on a fresh clone still creates it. From a
       linked worktree it is main's, so the run outlives the worktree.

    Raises :class:`RunArchiveError` — naming ``$GEN3AI_MODELS_DIR`` — when there is NO archive to
    land in: a linked worktree whose main checkout has no ``models/``, or a linked worktree git
    cannot resolve. It never answers with the worktree's own ``models/``.

    Unlike :func:`main_models_dir` (a READER: ``None`` means skip) this is the CREATOR's accessor:
    it never returns ``None`` and never falls back to a cwd- or worktree-relative directory.
    ``anchor`` is the checkout asked about (default: the one this code came from); the tests build
    a throwaway repo + worktree and ask it from there.
    """
    here = Path(anchor) if anchor is not None else _archive_anchor()
    override = os.environ.get(MODELS_DIR_ENV_VAR)
    if override:
        cand = Path(os.path.abspath(override))
        if cand.is_dir():
            return cand
        raise RunArchiveError(
            f"${MODELS_DIR_ENV_VAR}={override!r} is not a directory — it is authoritative, so "
            f"there is no fall-back to another archive. Create it, or unset ${MODELS_DIR_ENV_VAR} "
            f"to use the main checkout's models/.")
    if anchor is None and os.environ.get(RUN_ARCHIVE_SEAL_ENV_VAR):
        raise RunArchiveError(
            f"the run archive is SEALED (${RUN_ARCHIVE_SEAL_ENV_VAR} is set — this is a test "
            f"process) and ${MODELS_DIR_ENV_VAR} is not: a test must point ${MODELS_DIR_ENV_VAR} at "
            f"a tmp dir before anything creates a run directory, so it can never write into the "
            f"real run archive.")
    main = _main_checkout_of(here)
    if main is None:
        if is_linked_worktree(here):
            raise RunArchiveError(
                f"{str(here)!r} is a linked git worktree and git could not name the main checkout, "
                f"so there is no run archive to land in (a worktree's own models/ is deleted with "
                f"the worktree). Set ${MODELS_DIR_ENV_VAR} to the archive.")
        return here / "models"          # not a git checkout: no worktree to lose a run with
    cand = main / "models"
    if cand.is_dir() or not is_linked_worktree(here):
        return cand
    raise RunArchiveError(
        f"the main checkout {str(main)!r} has no models/ directory, and {str(here)!r} is a linked "
        f"worktree (its own models/ is deleted with it): there is no run archive to land in. "
        f"Create {str(cand)!r}, or set ${MODELS_DIR_ENV_VAR} to the archive.")


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _linked_worktree_roots(here: Path) -> "list[str]":
    """Every LINKED worktree of this repo, ``here`` first when it is one (git failing costs the
    others, never ``here`` — that needs no git)."""
    roots: "list[str]" = [str(here)] if is_linked_worktree(here) else []
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "worktree", "list", "--porcelain"], text=True,
            stderr=subprocess.DEVNULL, cwd=str(here))
    except Exception:
        return roots
    listed = [ln[len("worktree "):] for ln in out.splitlines() if ln.startswith("worktree ")]
    for wt in listed[1:]:               # the first entry is the main worktree
        if wt not in roots:
            roots.append(wt)
    return roots


def checked_run_dir(path: "str | os.PathLike[str]", *, anchor: Optional[Path] = None) -> str:
    """``path`` as an ABSOLUTE run-dir string — or a typed refusal when it would land inside a
    linked worktree's own ``models/``.

    That directory is not committed, so a clean ``git status`` hides it and removing the worktree
    deletes the run with it (2026-09-23: eight runs). The check is on the REAL path, so a worktree
    whose ``models`` is a symlink OUT to the main archive (or a run dir that is one) is correctly
    allowed, and an archive entry that symlinks INTO a worktree is correctly refused.

    Apply it to EVERY run dir a process is about to create or write into — derived (``run_archive_dir``)
    or explicit (``--run-dir``, a resumed checkpoint's own dir, a meter's default ``--out``).
    """
    here = Path(anchor) if anchor is not None else _archive_anchor()
    p = os.path.abspath(os.fspath(path))
    real = os.path.realpath(p)
    for wt in _linked_worktree_roots(here):
        wt_real = os.path.realpath(wt)
        wt_models = os.path.realpath(os.path.join(wt, "models"))
        if not _inside(wt_models, wt_real):
            continue                    # `models` links OUT of the worktree (e.g. to main's) — safe
        if _inside(real, wt_models):
            try:
                archive = f"the run archive {str(run_archive_dir(anchor))!r}"
            except RunArchiveError:
                archive = "the main checkout's models/"
            raise RunArchiveError(
                f"run directory {p!r} is inside the OWN models/ of the linked worktree {wt!r} — "
                f"a worktree's models/ is not committed and is deleted SILENTLY when the worktree "
                f"is removed (2026-09-23: eight runs lost). Runs land in {archive}: omit --run-dir "
                f"to let the default apply, or name a path there (or set ${MODELS_DIR_ENV_VAR}).")
    return p


def new_run_dir(name: str, *, anchor: Optional[Path] = None) -> str:
    """``<run_archive_dir()>/<name>`` — the directory a NEW run named ``name`` lands in, checked."""
    return checked_run_dir(run_archive_dir(anchor) / name, anchor=anchor)


def resolve_archive_ref(path: str, *, anchor: Optional[Path] = None) -> str:
    """A relative ``models/<run>/…`` reference that the cwd does not hold, re-anchored at the archive.

    The READ-side companion of :func:`run_archive_dir`, for the one reference a launch must turn
    into a run dir — ``--model models/<run>/checkpoints/x.zip`` typed in a worktree, where
    ``models/`` does not exist. A FALLBACK, never an override: an absolute path, one the cwd has, one
    that is not ``models/…``-rooted, and one the archive does not have are returned unchanged (so
    the caller's own error still names what the user typed)."""
    if not path or os.path.isabs(path) or os.path.exists(path):
        return path
    norm = path.replace("\\", "/")
    if norm != "models" and not norm.startswith("models/"):
        return path
    try:
        archive = run_archive_dir(anchor)
    except RunArchiveError:
        return path
    cand = archive.joinpath(*[x for x in norm.split("/")[1:] if x])
    return str(cand) if cand.exists() else path


def models_skip_reason() -> str:
    """The message a test should skip with when :func:`main_models_dir` returns ``None``.

    Says what is missing AND that its absence is expected off the owner's box, so a contributor
    reading a skip does not go looking for a broken test.
    """
    return (
        f"no run archive: the main checkout has no models/ directory. models/ is NOT committed — "
        f"it exists only on a machine that has trained. Expected on a fresh clone / CI; set "
        f"${MODELS_DIR_ENV_VAR} if your archive lives elsewhere."
    )


def trace_glob(run_name: str) -> Optional[str]:
    """The ``eval_traces`` glob for one archived run, or ``None`` if the run is not on this box.

    Tests that need REAL decision states name a specific run (they depend on that run's obs
    layout / generation), so a missing archive and a missing *run* are the same skip.
    """
    models = main_models_dir()
    if models is None:
        return None
    run_dir = models / run_name
    if not run_dir.is_dir():
        return None
    return str(run_dir / "eval_traces" / "**" / "*_states.npz")


def harvest_dir(create: bool = False) -> Path:
    """The run-agnostic archive for HARVEST artifacts — label shards, fine-tuned heads, meters.

    A fifth question, and it is deliberately not any of the four above. Harvest outputs are
    **generated, large, and belong to no single run**: they are mined from many runs' traces and
    consumed by tooling that must not write into ``models/`` (an archive that is read-only by
    convention, so a probe can never corrupt the thing it is probing).

    Unlike :func:`main_models_dir` this **returns a path that need not exist** and never ``None``:
    a caller here is a producer that is about to create it, not a test that must skip. ``create=True``
    makes it. It is anchored at the MAIN checkout for the same reason ``models/`` is — a worktree
    is deleted when its agent finishes, and an hours-long harvest that vanishes with it is worse
    than one that is merely inconvenient to find. ``$GEN3AI_HARVEST_DIR`` overrides and is
    AUTHORITATIVE (no quiet fall-back), which is also the seam the tests point at a tmpdir.
    """
    override = os.environ.get(HARVEST_DIR_ENV_VAR)
    if override:
        cand = Path(override)
    else:
        cand = None
        from utils.git import get_main_repo_root
        try:
            cand = Path(get_main_repo_root(cwd=str(_REPO_ROOT))) / "harvest"
        except Exception:
            cand = None  # not a git checkout — fall through to this checkout
        if cand is None:
            cand = repo_path("harvest")
    if create:
        cand.mkdir(parents=True, exist_ok=True)
    return cand


def run_skip_reason(run_name: str) -> str:
    """The message a test should skip with when :func:`trace_glob` returns ``None``."""
    return (
        f"no eval traces for run {run_name!r}: models/ lives only in the main checkout and is "
        f"not committed, so this run is absent on any box that did not train it. Expected on a "
        f"fresh clone / CI; set ${MODELS_DIR_ENV_VAR} to point at an archive that has it."
    )
