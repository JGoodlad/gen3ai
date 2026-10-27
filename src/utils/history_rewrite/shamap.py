"""The old -> new commit SHA map: read filter-repo's ``commit-map``, validate it, write the in-repo TSV.

**Formats.** filter-repo leaves ``.git/filter-repo/commit-map`` (a header line, then ``<old> <new>`` per line,
EVERY rewritten commit, unchanged ones included, a pruned one mapped to 40 zeros) and ``.git/filter-repo/ref-map``
(``<old> <new> <refname>``). The TSV this module writes keeps the CHANGED commits only, two tab-separated 40-hex
columns, ``#`` comment lines ahead of them; it is the file ``designs/ops/history_rewrite_sha_map.tsv`` and
``utils.git_sha`` read.

**No pruned commits.** The rewrite runs with ``--prune-empty never``, so the map is a bijection on the changed
commits. A pruned entry (40 zeros) would make a citation unmappable, so :func:`validate` refuses it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from utils.git_sha import load_map
from utils.history_rewrite.gitobjs import FULL_SHA, NULL_SHA

HEADER_STATUS_NONE = "# status: NO REWRITE HAS HAPPENED (this file holds no rows until `python -m utils.history_rewrite map` writes it)"


def read_commit_map(path: str | Path) -> dict[str, str]:
    """filter-repo's ``commit-map`` as ``{old: new}`` (all rows, including unchanged)."""
    pairs: dict[str, str] = {}
    lines = Path(path).read_text().splitlines()
    for line in lines[1:]:  # line 0 is the "old                                      new" header
        fields = line.split()
        if len(fields) == 2:
            pairs[fields[0]] = fields[1]
    return pairs


def read_ref_map(path: str | Path) -> list[tuple[str, str, str]]:
    """filter-repo's ``ref-map`` rows as ``(old, new, refname)``."""
    rows = []
    for line in Path(path).read_text().splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3 and FULL_SHA.match(fields[0]):
            rows.append((fields[0], fields[1], fields[2]))
    return rows


def changed_only(pairs: dict[str, str]) -> dict[str, str]:
    return {o: n for o, n in pairs.items() if o != n}


def validate(pairs: dict[str, str]) -> None:
    """Refuse a map that is not a clean bijection of 40-hex ids (a pruned commit, a duplicate, a bad row)."""
    for old, new in pairs.items():
        if not FULL_SHA.match(old) or not FULL_SHA.match(new):
            raise ValueError(f"map row is not 40-hex: {old[:12]!r} -> {new[:12]!r}")
        if new == NULL_SHA:
            raise ValueError(f"commit {old[:12]} was PRUNED (maps to zeros): rerun with --prune-empty never")
        if old == new:
            raise ValueError(f"unchanged commit {old[:12]} in a changed-only map")
    if len(set(pairs.values())) != len(pairs):
        raise ValueError("two old commits map to one new commit: not a bijection")
    if set(pairs) & set(pairs.values()):
        raise ValueError("an old id is also a new id: citations would be ambiguous")


def write_tsv(path: str | Path, pairs: dict[str, str], header: Iterable[str]) -> None:
    validate(pairs)
    body = [h if h.startswith("#") else f"# {h}" for h in header]
    body += [f"{o}\t{n}" for o, n in sorted(pairs.items())]
    Path(path).write_text("\n".join(body) + "\n")


def read_tsv(path: str | Path) -> dict[str, str]:
    """The in-repo map as ``{old: new}`` (``utils.git_sha.load_map``'s parser, plus :func:`validate`)."""
    pairs = dict(load_map(path))
    validate(pairs)
    return pairs
