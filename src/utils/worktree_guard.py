"""WORKTREE REMOVAL GUARD — refuse to delete a git worktree that holds RUN DATA.

    python -m utils.worktree_guard --main <main checkout> <worktree>     # exit 0 safe · 3 refuse

🚨 THE INCIDENT (ledger 2026-09-23, *EIGHT RUN DIRECTORIES DESTROYED*). Early launcher worktrees
(`.claude/worktrees/gen*-run-08xx`) wrote each run directory INSIDE the worktree and left only a
SYMLINK in the main checkout's `models/`. `models/` is gitignored, so a clean `git status` hid the
data from every pre-removal check (uncommitted edits, untracked files, unlanded commits, live
processes), and `git worktree remove --force` deleted eight v9 generation-ladder runs. A removal
whose safety check is blind to the one thing it destroys is the defect; this module is the check.

TWO RULES, each a refusal (a :class:`Hazard`):

* **(a) a symlink into the tree** — any entry of the main checkout's `models/` (and of
  `$GEN3AI_MODELS_DIR`, when set), one level into each real run directory, plus the main
  checkout's own top-level entries, that is a symlink resolving INTO the worktree. That is the
  incident's exact shape, and it is refused whatever the size.
* **(b) untracked/ignored content above a threshold** — everything `git ls-files --others` lists
  (untracked AND ignored, which `--force` deletes alike), MINUS an explicit allowlist of benign
  build/cache paths, summed; above :data:`DEFAULT_THRESHOLD_BYTES` (50 MiB) every offending entry
  is reported with its size. Sizes are APPARENT (`st_size`, symlinks not followed), so a sparse
  file counts at its full length — what a removal would lose, not the blocks it happens to use.

Plus the degenerate one: the "worktree" IS the main checkout.

⚠️ THE HONEST LIMIT. A run directory under 50 MiB that NOTHING in the main checkout links to is
not protected by rule (b), and nothing here looks inside a submodule (`git ls-files --others` does
not descend into one). Anything the guard cannot establish — `git` failing, an unreadable tree —
is an ERROR (exit 2), which every caller treats as a refusal: this guard fails CLOSED.

Stdlib only, and run with ``-m`` rather than by file path: a script run by path puts ``src/utils/``
first on ``sys.path``, where the ``utils/logging`` package would shadow the standard library's.
Callers: ``scripts/land.sh`` (step 4) and the launcher's own pin removal
(``main.launcher.worktree``). Tests: ``src/utils/worktree_guard_test.py``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

#: Above this much non-allowlisted untracked/ignored content, a removal is REFUSED.
DEFAULT_THRESHOLD_BYTES = 50 * 1024 * 1024

#: A path with ANY of these components is a build/cache artifact, never run data.
BENIGN_DIR_NAMES = frozenset({
    "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".hypothesis",
    ".ipynb_checkpoints", "node_modules",
})

#: A path ending in one of these is a build/cache artifact.
BENIGN_SUFFIXES = (".pyc", ".pyo", ".egg-info")

#: Repo-relative paths (and everything under them) that are build artifacts: the per-worktree
#: cargo target and lockfile, and the two Showdown build symlinks the worktree setup creates.
BENIGN_PATHS = (
    "src/rust_sim/target",
    "src/rust_sim/Cargo.lock",
    "deps/pokemon-showdown/dist",
    "deps/pokemon-showdown/node_modules",
)

EXIT_SAFE = 0
EXIT_ERROR = 2
EXIT_REFUSED = 3


class GuardError(RuntimeError):
    """The guard could not establish that the removal is safe. Callers REFUSE on it."""


@dataclass(frozen=True)
class Hazard:
    kind: str       # "models_symlink" | "untracked_data" | "is_main_checkout"
    path: str
    detail: str
    size: int = 0

    def line(self) -> str:
        size = f"  [{_human(self.size)}]" if self.size else ""
        return f"{self.kind:<16} {self.path}{size} — {self.detail}"


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if x < 1024 or unit == "GiB":
            return f"{x:.1f} {unit}" if unit != "B" else f"{int(x)} B"
        x /= 1024
    return f"{n} B"


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


# --------------------------------------------------------------------------------------------
# Rule (a): symlinks in the main checkout that resolve into the worktree
# --------------------------------------------------------------------------------------------

def _symlink_candidates(main_checkout: str) -> Iterable[str]:
    """Every path whose symlink-ness is checked: main's top level, and `models/` (plus
    `$GEN3AI_MODELS_DIR`) to depth 2 — a run dir, and one level into a REAL run dir."""
    seen = set()
    roots = [os.path.join(main_checkout, "models")]
    env = os.environ.get("GEN3AI_MODELS_DIR")
    if env:
        roots.append(env)
    try:
        top = [e.path for e in os.scandir(main_checkout)]
    except OSError as exc:
        raise GuardError(f"cannot list the main checkout {main_checkout!r}: {exc}") from exc
    for p in top:
        seen.add(p)
        yield p
    for root in roots:
        if not os.path.isdir(root):
            continue
        with os.scandir(root) as it:
            entries = list(it)
        for e in entries:
            if e.path not in seen:
                seen.add(e.path)
                yield e.path
            if e.is_dir(follow_symlinks=False):
                try:
                    with os.scandir(e.path) as sub:
                        for s in sub:
                            if s.is_symlink():
                                yield s.path
                except OSError:
                    continue


def symlink_hazards(main_checkout: str, worktree: str) -> List[Hazard]:
    wt_real = os.path.realpath(worktree)
    out: List[Hazard] = []
    for p in _symlink_candidates(main_checkout):
        if not os.path.islink(p):
            continue
        target = os.path.realpath(p)
        if _inside(target, wt_real):
            out.append(Hazard(
                "models_symlink", p,
                f"symlink resolves INTO the worktree ({os.readlink(p)} -> {target}); "
                "removing the worktree deletes what it points at"))
    return out


# --------------------------------------------------------------------------------------------
# Rule (b): untracked + ignored content that is not a known build/cache artifact
# --------------------------------------------------------------------------------------------

def is_benign(rel: str) -> bool:
    rel = rel.rstrip("/")
    parts = rel.split("/")
    if any(p in BENIGN_DIR_NAMES for p in parts):
        return True
    if rel.endswith(BENIGN_SUFFIXES):
        return True
    return any(rel == b or rel.startswith(b + "/") for b in BENIGN_PATHS)


def untracked_entries(worktree: str) -> List[str]:
    """Every untracked OR ignored path, whole untracked directories collapsed to ``dir/``."""
    proc = subprocess.run(
        ["git", "-C", worktree, "ls-files", "--others", "--directory", "-z"],
        capture_output=True)
    if proc.returncode != 0:
        raise GuardError(f"`git ls-files --others` failed in {worktree!r}: "
                         f"{proc.stderr.decode(errors='replace').strip()}")
    return [p for p in proc.stdout.decode(errors="surrogateescape").split("\0") if p]


def apparent_size(worktree: str, rel: str) -> int:
    """Bytes a removal would lose under ``<worktree>/<rel>`` — apparent sizes, symlinks NOT
    followed, and every allowlisted path INSIDE it skipped. The allowlist is applied during the
    walk, by repo-relative path, because ``ls-files --directory`` collapses a wholly-untracked
    PARENT (``src/rust_sim/`` in a tree that tracks nothing under it) and its benign child
    (``src/rust_sim/target``) would otherwise be counted as run data."""
    path = os.path.join(worktree, rel)
    try:
        st = os.lstat(path)
    except OSError:
        return 0
    if not os.path.isdir(path) or os.path.islink(path):
        return st.st_size
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        here = os.path.relpath(dirpath, worktree).replace(os.sep, "/")
        dirnames[:] = [d for d in dirnames if not is_benign(f"{here}/{d}")]
        for f in filenames:
            if is_benign(f"{here}/{f}"):
                continue
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                continue
    return total


def untracked_hazards(worktree: str, threshold_bytes: int = DEFAULT_THRESHOLD_BYTES,
                      ) -> Tuple[List[Hazard], int]:
    """``(hazards, total_bytes)``. Hazards are empty unless the non-benign total is ABOVE the
    threshold, in which case every non-benign entry is reported, largest first."""
    sized: List[Tuple[int, str]] = []
    for rel in untracked_entries(worktree):
        if is_benign(rel):
            continue
        sized.append((apparent_size(worktree, rel), rel))
    total = sum(s for s, _ in sized)
    if total <= threshold_bytes:
        return [], total
    sized.sort(reverse=True)
    return [Hazard("untracked_data", rel,
                   "untracked/ignored and not a known build artifact", size)
            for size, rel in sized if size > 0], total


# --------------------------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------------------------

def find_hazards(main_checkout: str, worktree: str, *,
                 threshold_bytes: int = DEFAULT_THRESHOLD_BYTES) -> List[Hazard]:
    """Every reason removing ``worktree`` would destroy data. Empty ⇒ safe to remove.

    Raises :class:`GuardError` when safety cannot be established; treat that as a refusal.
    A worktree directory that no longer exists holds nothing, and returns no hazard.
    """
    main_real = os.path.realpath(main_checkout)
    if not os.path.isdir(worktree):
        return []
    wt_real = os.path.realpath(worktree)
    if wt_real == main_real:
        return [Hazard("is_main_checkout", worktree, "this IS the main checkout")]
    hazards = symlink_hazards(main_real, wt_real)
    more, _total = untracked_hazards(wt_real, threshold_bytes)
    return hazards + more


def report(hazards: Sequence[Hazard], worktree: str,
           threshold_bytes: int = DEFAULT_THRESHOLD_BYTES) -> str:
    lines = [f"WORKTREE GUARD — REFUSING to remove {worktree}: it holds data a removal would "
             "DESTROY (incident 2026-09-23: eight run directories lost this way).",
             f"  {len(hazards)} hazard(s):"]
    lines += [f"    {h.line()}" for h in hazards[:40]]
    if len(hazards) > 40:
        lines.append(f"    … and {len(hazards) - 40} more")
    if any(h.kind == "untracked_data" for h in hazards):
        total = sum(h.size for h in hazards if h.kind == "untracked_data")
        lines.append(f"  untracked/ignored non-build content totals {_human(total)} "
                     f"(threshold {_human(threshold_bytes)}).")
    lines += ["  FIX: move the data out (a run dir belongs in the MAIN checkout's models/, "
              "replacing its symlink) or delete it deliberately, then remove the worktree.",
              "  Build/cache paths never count: " + ", ".join(sorted(BENIGN_DIR_NAMES))
              + ", " + ", ".join(BENIGN_PATHS) + "."]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m utils.worktree_guard",
                                description="Refuse to remove a worktree that holds run data.")
    p.add_argument("--main", required=True, dest="main_checkout",
                   help="the MAIN checkout (whose models/ is scanned for symlinks into the tree)")
    p.add_argument("worktree")
    p.add_argument("--threshold-mb", type=float, default=DEFAULT_THRESHOLD_BYTES / 2 ** 20)
    args = p.parse_args(argv)
    threshold = int(args.threshold_mb * 2 ** 20)
    try:
        hazards = find_hazards(args.main_checkout, args.worktree, threshold_bytes=threshold)
    except (GuardError, OSError) as exc:
        print(f"WORKTREE GUARD — ERROR, refusing: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if hazards:
        print(report(hazards, args.worktree, threshold), file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_SAFE


if __name__ == "__main__":
    sys.exit(main())
