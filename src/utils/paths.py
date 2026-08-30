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
===========================================  =============================  ==================

The third is a different question and it is the one that keeps biting. **In a git worktree
:func:`repo_root` is the WORKTREE**, which has no ``models/`` — the run archive lives only in the
main checkout. A test that wants an archived run must reach across, which is what
:func:`main_models_dir` does, via ``utils.git.get_main_repo_root()`` — the same
``--git-common-dir`` logic the launcher uses to find the main checkout. Getting this wrong is
invisible: the directory is simply absent and a skip-if-missing test skips forever.

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

    A fourth question, and it is deliberately not any of the three above. Harvest outputs are
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
