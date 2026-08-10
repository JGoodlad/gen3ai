"""Which runs this server will open, and the confinement that decides it.

THE THREAT. The hosted instance is unauthenticated for reading, so the run selector is a string
an anonymous visitor controls. If that string were ever joined onto a path, `../../../etc/passwd`
would be one request away.

THE DESIGN THAT MAKES THAT UNREPRESENTABLE. **No client string is ever joined to a path.** The
server enumerates the children of the models root itself, and a request's `run` value is only ever
tested for MEMBERSHIP in that enumerated set. A traversal string cannot appear in a directory
listing, so it cannot select anything. This is deliberately not "sanitise the input" — sanitising
is a blocklist you can be wrong about; membership is a allowlist you cannot.

SYMLINKS, AND WHY THE TOP LEVEL IS SPECIAL. The repo's own layout depends on them: the launcher
isolates each run in a git worktree and surfaces it into `models/` as a symlink, so on this box the
five newest runs — the entire current generation — are links into
`.claude/worktrees/<name>/models/run_<ts>`. A blanket "refuse all symlinks" would hide exactly the
runs anyone wants to look at.

So the rule is asymmetric, by the owner's decision (2026-08-09):

  * A **direct child of the models root** may be a symlink. It was placed there by whoever
    administers the box, it is resolved ONCE here at enumeration, and the resolved path becomes
    that run's canonical root. A client still cannot name anything that is not in the listing.
  * **Nothing deeper may be.** A run containing a symlink at any depth is REFUSED outright, not
    silently skipped — a link planted inside a run is the one remaining way a file outside the run
    could be read, and it should be loud.

`RunAccessError` carries a message safe to render: it never echoes back the path it rejected, so a
probe cannot use error text to map the filesystem.
"""

from __future__ import annotations

import os
import re

# A run's name is the directory's own basename, which the server produced. This is a
# belt-and-braces reject of anything that could not be a plain directory entry we listed —
# membership in `list_runs()` is the actual gate.
_SANE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")

# What makes a directory a "run" rather than some other thing sitting in models/. `eval_traces`
# is what the web views read; the others let a run still be listed (and reported as empty) rather
# than vanishing confusingly.
_RUN_MARKERS = ("eval_traces", "metadata.json", "model_config.json", "checkpoints", "latest.txt")


class RunAccessError(Exception):
    """A run was not selectable. The message is safe to show a user."""


class RunStore:
    """Enumerates selectable runs under a models root and resolves a name to a real path.

    `root` may be a models directory (picker mode) or a single run directory (pinned mode — the
    set of selectable runs is then exactly that one run, and the root is NOT widened to its
    parent; pointing at one run must never make its siblings reachable).
    """

    def __init__(self, root: str) -> None:
        raw = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(raw):
            raise RunAccessError("that path is not a directory")
        # Resolve the root once. Everything downstream compares against the resolved form, so a
        # symlinked models root (or a run dir given directly) is handled here and nowhere else.
        self.root = os.path.realpath(raw)
        self.pinned = _looks_like_run(self.root)
        self._symlink_audit: "dict[str, str | None]" = {}

    # -- enumeration ---------------------------------------------------------------------

    def list_runs(self) -> "list[dict]":
        """Every selectable run, newest first.

        Each row is `{name, path, has_traces, linked}` — `linked` marks a run reached through an
        owner-placed top-level symlink, so the UI can say so rather than pretending the layout is
        flat.
        """
        if self.pinned:
            return [self._row(os.path.basename(self.root), self.root, linked=False)]

        rows = []
        try:
            entries = list(os.scandir(self.root))
        except OSError:
            return []
        for entry in entries:
            if not _SANE_NAME.match(entry.name):
                continue
            linked = entry.is_symlink()
            # `is_dir()` follows the link — intended, and the only place a link is followed.
            if not entry.is_dir():
                continue
            target = os.path.realpath(entry.path)
            if not _looks_like_run(target):
                continue
            rows.append(self._row(entry.name, target, linked=linked))
        rows.sort(key=lambda r: (r["mtime"], r["name"]), reverse=True)
        return rows

    def _row(self, name: str, path: str, *, linked: bool) -> dict:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        return {"name": name, "path": path, "linked": linked, "mtime": mtime,
                "has_traces": os.path.isdir(os.path.join(path, "eval_traces"))}

    def default_run(self) -> "str | None":
        rows = self.list_runs()
        return rows[0]["name"] if rows else None

    # -- resolution ----------------------------------------------------------------------

    def resolve(self, name: "str | None") -> str:
        """A run name -> its real directory. Raises `RunAccessError` for anything else.

        The name is matched against the enumerated listing, and the path returned is the one the
        SERVER produced. The input is never joined, normalised or cleaned up — a value that is not
        already a run we listed simply does not resolve.

        Note there is deliberately no `_SANE_NAME` check here. Membership is the entire gate, so
        the pattern would add nothing — and applying it here actively broke PINNED mode, where the
        name is the server's own directory basename and may legitimately contain a space or a
        leading underscore: `list_runs()` advertised such a run while `resolve()` refused it, so
        the app 404'd on its own and only run. The pattern still guards ENUMERATION, which is
        where an odd name in a shared `models/` directory is worth skipping.
        """
        if not name:
            raise RunAccessError("no run selected")
        if len(name) > 512:                       # nothing we list is remotely this long
            raise RunAccessError("no such run")
        for row in self.list_runs():
            if row["name"] == name:
                return self._audited(row["path"])
        raise RunAccessError("no such run")

    def _audited(self, path: str) -> str:
        """Refuse a run that contains a symlink anywhere inside it.

        Cached: the walk is ~7k entries on a real run (tens of ms), but it would otherwise run on
        every request. A run whose contents change to add a link after the first look is not
        re-checked until restart — an accepted limit, since planting one requires write access to
        the box, at which point the symlink is not the interesting capability.
        """
        if path in self._symlink_audit:
            bad = self._symlink_audit[path]
            if bad:
                raise RunAccessError(f"that run contains a symlink ({bad}) and will not be opened")
            return path
        try:
            bad = _first_symlink(path)
        except _WalkError as exc:
            # NOT cached: an unreadable subtree may be transient (a directory mid-delete, a
            # permission being fixed), and caching a refusal would make that permanent until
            # restart. Re-walking on the next request is cheap next to serving a run whose
            # contents we could not actually inspect.
            raise RunAccessError(
                "that run could not be fully inspected for symlinks "
                f"({exc.err.__class__.__name__}) and will not be opened") from exc
        self._symlink_audit[path] = bad
        if bad:
            raise RunAccessError(f"that run contains a symlink ({bad}) and will not be opened")
        return path


def _looks_like_run(path: str) -> bool:
    return any(os.path.exists(os.path.join(path, m)) for m in _RUN_MARKERS)


def _first_symlink(root: str) -> "str | None":
    """The run-relative path of the first symlink found under `root`, or None.

    `followlinks=False` keeps the walk from descending THROUGH a link; the explicit checks are
    what notice one is there at all. Returns a run-relative path (never absolute) because the
    message is rendered to a user.

    **`onerror` is load-bearing.** `os.walk`'s default silently DISCARDS an OSError and yields
    nothing for that subtree — so an unreadable directory (mode 0o000, or one being deleted under
    us) hid a symlink inside it and the audit returned "clean". An audit whose whole job is to
    refuse a symlink must not fail open on the one input designed to defeat it, so any walk error
    is raised and becomes a refusal. Over-refusing an unreadable run is the right way to be wrong.
    """
    error = []
    os_walk = os.walk(root, followlinks=False, onerror=error.append)
    for dirpath, dirnames, filenames in os_walk:
        if error:
            raise _WalkError(error[0])
        for entry in list(dirnames) + list(filenames):
            full = os.path.join(dirpath, entry)
            if os.path.islink(full):
                return os.path.relpath(full, root)
    if error:
        raise _WalkError(error[0])
    return None


class _WalkError(Exception):
    """An OSError during the symlink audit. Carried out to `_audited`, which turns it into a
    refusal — never into a pass."""

    def __init__(self, err: OSError) -> None:
        super().__init__(err)
        self.err = err
