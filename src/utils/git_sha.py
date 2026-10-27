"""Resolve a commit SHA a document, sidecar or tag NAME cites, across the history rewrite.

After the one-time history rewrite (``designs/ops/history_rewrite.md``) every commit from the rewrite point on has a
new SHA. A sidecar's ``original_command`` / ``pin_history``, an archive record, a frozen ledger entry and a tag name
cite the OLD one, and none of those may be edited. ``designs/ops/history_rewrite_sha_map.tsv`` (old -> new, 40-hex,
changed commits only) is the key; :func:`resolve` consults it FIRST and git second:

* the token is a prefix of exactly one OLD sha in the map -> ``mapped``: the NEW full sha (checked against git: if the
  token also names a different LIVE commit the answer is ``ambiguous``, never a silent pick);
* a prefix of several old shas -> ``ambiguous`` (the candidates are listed);
* otherwise git decides: ``git`` (a live commit, unchanged by the rewrite or cited after it), ``ambiguous`` or
  ``unknown``.

Map first, git second, because for a while after the rewrite the local object database still holds the OLD commits
(a pinned worktree, the reflog until it is expired): asking git first would silently return the commit that still
carries the purged names. With no rewrite the map is empty and this is ``git rev-parse``.

``python -m utils.git_sha <sha-or-prefix>...`` prints ``<query> -> <full sha>  (<status>)`` per query; the exit code
is 0 when all resolve, 1 when one is ``unknown``, 2 when one is ``ambiguous``. The module is deliberately
self-contained (stdlib + :mod:`utils.paths`): it outlives the one-shot rewrite tooling that produced the map.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.paths import repo_path, repo_root

MAP_PATH = repo_path("designs", "ops", "history_rewrite_sha_map.tsv")
_HEX = re.compile(r"^[0-9a-f]{4,40}$")
_FULL = re.compile(r"^[0-9a-f]{40}$")

MAPPED, GIT, AMBIGUOUS, UNKNOWN = "mapped", "git", "ambiguous", "unknown"

_cache: dict[tuple[str, int, int], dict[str, str]] = {}


@dataclass(frozen=True)
class Resolution:
    query: str
    status: str  # mapped | git | ambiguous | unknown
    full: Optional[str] = None
    candidates: tuple[str, ...] = ()
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (MAPPED, GIT) and self.full is not None


class ShaError(LookupError):
    """A SHA that did not resolve to exactly one commit (``.resolution`` says why)."""

    def __init__(self, resolution: Resolution):
        super().__init__(f"{resolution.query}: {resolution.status}"
                         + (f" ({resolution.note})" if resolution.note else "")
                         + (f" candidates {[c[:12] for c in resolution.candidates]}" if resolution.candidates else ""))
        self.resolution = resolution


def load_map(path: str | Path | None = None) -> dict[str, str]:
    """``{old: new}`` from the TSV; an absent or header-only file is the empty map. Cached by (path, mtime, size)."""
    p = Path(path) if path is not None else MAP_PATH
    try:
        st = p.stat()
    except FileNotFoundError:
        return {}
    key = (str(p), st.st_mtime_ns, st.st_size)
    if key not in _cache:
        pairs: dict[str, str] = {}
        for line in p.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            old, _, new = line.partition("\t")
            if not (_FULL.match(old) and _FULL.match(new)):
                raise ValueError(f"{p}: malformed SHA map row {line[:60]!r}")
            pairs[old] = new
        _cache[key] = pairs
    return _cache[key]


def _git_commit(token: str, repo: Optional[str]) -> tuple[Optional[str], bool]:
    """``(full sha or None, ambiguous)`` of ``token`` as a commit in ``repo``'s object database."""
    proc = subprocess.run(["git", "-C", repo or str(repo_root()), "rev-parse", "--verify", "--quiet", f"{token}^{{commit}}"],
                          capture_output=True, text=True)
    out = proc.stdout.strip()
    if proc.returncode == 0 and _FULL.match(out):
        return out, False
    return None, "ambiguous" in proc.stderr


def resolve(token: str, *, repo: Optional[str] = None, map_path: str | Path | None = None) -> Resolution:
    """Resolve one full or abbreviated SHA (see the module docstring for the order and the meanings)."""
    tok = token.strip().lower()
    if not _HEX.match(tok):
        return Resolution(token, UNKNOWN, note="not a 4-40 hex SHA")
    pairs = load_map(map_path)
    olds = sorted(o for o in pairs if o.startswith(tok))
    if len(olds) > 1:
        return Resolution(token, AMBIGUOUS, candidates=tuple(olds), note="prefix of several rewritten commits")
    if len(olds) == 1:
        old, new = olds[0], pairs[olds[0]]
        live, live_ambiguous = _git_commit(tok, repo)
        if live_ambiguous or (live is not None and live not in (old, new)):
            return Resolution(token, AMBIGUOUS, candidates=tuple(x for x in (old, live) if x),
                              note="names a rewritten commit AND a different live one: cite more digits")
        return Resolution(token, MAPPED, full=new, candidates=(old,))
    live, live_ambiguous = _git_commit(tok, repo)
    if live is not None:
        return Resolution(token, GIT, full=live)
    if live_ambiguous:
        return Resolution(token, AMBIGUOUS, note="ambiguous in git")
    return Resolution(token, UNKNOWN, note="not a commit here, and not a rewritten one")


def new_sha(token: str, *, repo: Optional[str] = None, map_path: str | Path | None = None) -> str:
    """The full CURRENT sha of what ``token`` cites; raises :class:`ShaError` unless it resolves to one commit."""
    r = resolve(token, repo=repo, map_path=map_path)
    if not r.ok or r.full is None:
        raise ShaError(r)
    return r.full


def old_sha(token: str, *, map_path: str | Path | None = None) -> list[str]:
    """The OLD full shas a NEW sha (or prefix) replaced; ``[]`` when it replaced none."""
    tok = token.strip().lower()
    return sorted(o for o, n in load_map(map_path).items() if n.startswith(tok))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m utils.git_sha", description=__doc__.split("\n\n")[0] if __doc__ else None)
    ap.add_argument("shas", nargs="+", help="full or abbreviated (4-40 hex) commit SHAs")
    ap.add_argument("--map", default=None, help=f"the SHA map (default {MAP_PATH})")
    ap.add_argument("--repo", default=None, help="the repository to ask (default: this checkout)")
    args = ap.parse_args(argv)
    worst = 0
    for sha in args.shas:
        r = resolve(sha, repo=args.repo, map_path=args.map)
        print(f"{sha} -> {r.full or '-'}  ({r.status}{': ' + r.note if r.note else ''})")
        worst = max(worst, {MAPPED: 0, GIT: 0, UNKNOWN: 1, AMBIGUOUS: 2}[r.status])
    return worst


if __name__ == "__main__":
    sys.exit(main())
