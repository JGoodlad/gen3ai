"""The rewrite itself: a throwaway CLONE of the source, then ``git filter-repo`` on that clone. Never the real repo.

**The plan, in order** (``python -m utils.history_rewrite plan`` prints it; ``rewrite`` runs it):

1. ``git clone --no-local <source> <work>/clone``: a full, independent copy (no hardlinks into the source's objects).
2. ``git fetch <source> +<oid>:refs/history-rewrite/keep/<oid> ...`` then ``git repack -a -d`` (filter-repo aborts on a
   clone that is not ONE fresh pack): the commits that descend from the introducing
   commit but no ref reaches (a detached worktree pin, an abandoned branch tip). Without a ref they would not be
   exported, would get no new SHA, and every citation of them would be unmappable. These refs are LOCAL BOOKKEEPING:
   the runbook never pushes ``refs/history-rewrite/``.
3. ``git filter-repo --replace-text <rules> --replace-message <rules> --file-info-callback <callback>
   --prune-empty never --prune-degenerate never --preserve-commit-encoding --preserve-commit-hashes``: the purge (the rules replace the global
   identities everywhere; the callback replaces the scoped ones in the account-data files only: ``rules.py``). ``--prune-empty never`` keeps the commit graph isomorphic (so the SHA map
   is a bijection and no citation is orphaned by a commit that vanished); ``--preserve-commit-encoding`` keeps a
   commit with a non-UTF-8 ``encoding`` header (one exists, long before the names) from changing its SHA and shifting
   every descendant. ``--preserve-commit-hashes`` stops filter-repo from rewriting the SHAs a commit MESSAGE cites: by
   default a commit OUTSIDE the descendants (the 2026-10-25 N1-P pin commit, whose message cites a later commit) changes
   SHA just because its message was updated, and every tag NAME and doc that cites it goes stale for nothing. Messages
   keep the old SHAs they cite, like every other immutable record; the map resolves them.
4. ``.git/filter-repo/commit-map`` and ``ref-map`` are copied beside the clone for ``map``.

``git filter-repo`` is a single Python file (``git_filter_repo.py``) that is NOT installed here: :func:`fetch_filter_repo`
downloads the pinned PyPI wheel (sha256 checked against the value PyPI published for 2.47.0) and unpacks that one file.

A signed commit loses its signature under filter-repo 2.47 (fast-export strips it) and so changes SHA. The four such
commits in this repository live only on a remote-only cloud branch that contains neither introducing commit; the runbook
removes branches that do not contain it from the mirror clone BEFORE the rewrite, and ``verify`` names any changed
commit that is not a descendant of the introducing commit.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from utils.history_rewrite.gitobjs import commit_oids, run_git

FILTER_REPO_VERSION = "2.47.0"
FILTER_REPO_WHEEL = f"git_filter_repo-{FILTER_REPO_VERSION}-py3-none-any.whl"
#: sha256 of that wheel as PyPI publishes it (https://pypi.org/pypi/git-filter-repo/2.47.0/json, checked 2026-10-27).
FILTER_REPO_WHEEL_SHA256 = "2cd04929b9024e83e65db571cbe36aec65ead0cb5f9ec5abe42158654af5ad83"
KEEP_NAMESPACE = "refs/history-rewrite/keep/"


@dataclass(frozen=True)
class Step:
    label: str
    argv: list[str]
    cwd: Optional[str] = None


def fetch_filter_repo(dest: str | os.PathLike[str]) -> Path:
    """Download + verify + unpack ``git_filter_repo.py`` into ``dest`` (outside any repo); return its path."""
    d = Path(dest)
    script = d / "git_filter_repo.py"
    if script.exists():
        return script
    d.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "pip", "download", f"git-filter-repo=={FILTER_REPO_VERSION}", "--no-deps",
                        "--disable-pip-version-check", "-q", "-d", tmp], check=True)
        wheel = Path(tmp) / FILTER_REPO_WHEEL
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        if digest != FILTER_REPO_WHEEL_SHA256:
            raise RuntimeError(f"git-filter-repo wheel sha256 {digest} != pinned {FILTER_REPO_WHEEL_SHA256}: refuse")
        with zipfile.ZipFile(wheel) as zf:
            script.write_bytes(zf.read("git_filter_repo.py"))
    return script


def unreachable_descendants(source: str, clone: str, introducing: str) -> list[str]:
    """Commits of ``source``'s object database that descend from ``introducing`` and that no ref of ``clone`` reaches."""
    base = run_git(source, "rev-parse", f"{introducing}^{{commit}}").decode().strip()
    universe = ("\n".join(commit_oids(source)) + "\n").encode()
    desc = set(run_git(source, "rev-list", "--stdin", f"--ancestry-path={base}", f"^{base}^", stdin=universe).decode().split())
    reached = set(run_git(clone, "rev-list", "--all").decode().split())
    return sorted(desc - reached)


def plan_steps(source: str, work: str, rules: str, callback: str, filter_repo: str, keep: list[str]) -> list[Step]:
    clone = os.path.join(work, "clone")
    refspecs = [f"+{o}:{KEEP_NAMESPACE}{o}" for o in keep]
    return [
        Step("clone", ["git", "clone", "--no-local", "--quiet", source, clone]),
        Step(f"keep-refs ({len(keep)} commits no ref reaches)",
             ["git", "-c", "uploadpack.allowAnySHA1InWant=true", "-C", clone, "fetch", "--no-tags", "--quiet", source, *refspecs]),
        Step("repack (filter-repo refuses a clone that is not one fresh pack)", ["git", "-C", clone, "repack", "-a", "-d", "-q"]),
        Step("filter-repo", [sys.executable, filter_repo, "--replace-text", rules, "--replace-message", rules,
                             "--file-info-callback", callback,
                             "--prune-empty", "never", "--prune-degenerate", "never", "--preserve-commit-encoding",
                             "--preserve-commit-hashes"],
             cwd=clone),
    ]


def run_rewrite(source: str, work: str, rules: str, callback: str, filter_repo: str, *, introducing: str,
                log: Callable[[str], None] = print) -> str:
    """Execute the plan into ``work`` (must not exist or be empty, and not inside ``source``); return the clone path."""
    src, wk = Path(source).resolve(), Path(work).resolve()
    if wk == src or src in wk.parents or wk in src.parents:
        raise ValueError("--work must be a separate directory, not inside (or around) the source repository")
    if wk.exists() and any(wk.iterdir()):
        raise ValueError(f"{wk} is not empty: refusing to reuse a work directory")
    wk.mkdir(parents=True, exist_ok=True)
    clone = str(wk / "clone")
    clone_step, _, repack_step, filter_step = plan_steps(str(src), str(wk), rules, callback, filter_repo, [])
    log(f"[1/4] {clone_step.label}")
    subprocess.run(clone_step.argv, check=True)
    keep = unreachable_descendants(str(src), clone, introducing)
    keep_step = plan_steps(str(src), str(wk), rules, callback, filter_repo, keep)[1]
    log(f"[2/4] {keep_step.label}")
    if keep:
        subprocess.run(keep_step.argv, check=True)
    log(f"[3/4] {repack_step.label}")
    subprocess.run(repack_step.argv, check=True)
    log("[4/4] filter-repo (the purge; minutes, single-threaded)")
    subprocess.run(filter_step.argv, cwd=filter_step.cwd, check=True)
    for name in ("commit-map", "ref-map"):
        shutil.copy2(os.path.join(clone, ".git", "filter-repo", name), wk / name)
    return clone
