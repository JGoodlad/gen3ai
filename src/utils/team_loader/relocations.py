"""Team files that MOVED, and the one resolver every recorded team path goes through.

Runs record their team files as the repo-relative paths they were LAUNCHED with
(``cli_args.trainee_team(s)``, ``--distill-teacher T:FILE``), and those records are immutable. When
a file is relocated, every archived run naming it would otherwise die in ``FileNotFoundError`` the
next time it is used as a fold-back opponent, a ``'<run>:*'`` teacher, or read by a meter.

``data/teams/relocations.json`` is the COMMITTED, explicit map ``old -> new`` (both relative to
``data/``, like a manifest's ``file``). :func:`resolve_team_file` consults it ONLY for a path that
does not exist, never guesses (no glob, no basename search), and says so on stderr when it fires.
A path that neither exists nor is in the map is returned unchanged, so the caller's own
"file does not exist" error still names the path the user typed.

The file's bytes never change in a relocation, so every recorded fingerprint (``pin_sha`` /
``pin_shas``, which are UNSTRIPPED sha1 prefixes) still matches after resolution.
"""
from __future__ import annotations

import functools
import json
import os
import sys

RELOCATIONS_FILE = ("data", "teams", "relocations.json")
_MARKER = "data/teams/"


@functools.lru_cache(maxsize=1)
def relocation_map() -> "dict[str, str]":
    """``{"teams/sample/x.txt": "teams/promoted/x.txt", ...}`` from the committed map. Raises if the
    file is missing: it is COMMITTED, and a resolver that silently resolved nothing would turn every
    relocated path back into a bare ``FileNotFoundError`` with no hint why."""
    from utils.paths import repo_path
    path = repo_path(*RELOCATIONS_FILE)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    moved = raw["moved"]
    for old, new in moved.items():
        if not (old.startswith("teams/") and new.startswith("teams/")):
            raise ValueError(f"{path}: relocation {old!r} -> {new!r} is not data/-relative "
                             "(both sides must start with 'teams/')")
    return dict(moved)


def resolve_team_file(path: str, *, quiet: bool = False) -> str:
    """``path`` if it exists; else its relocated home per ``data/teams/relocations.json``; else
    ``path`` unchanged. Keeps whatever prefix ``path`` carried (relative or absolute)."""
    if not path or os.path.exists(path):
        return path
    norm = path.replace(os.sep, "/")
    i = norm.rfind(_MARKER)
    if i < 0:
        return path
    prefix, rel = norm[:i], norm[i + len("data/"):]
    new = relocation_map().get(rel)
    if new is None:
        return path
    out = f"{prefix}data/{new}"
    if not quiet:
        print(f"📦 [TEAM PATH] {path} was relocated → {out} (data/teams/relocations.json)",
              file=sys.stderr)
    return out
