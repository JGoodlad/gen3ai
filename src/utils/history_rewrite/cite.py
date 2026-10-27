"""The citation rewriter: point every SHA a tracked text file cites at the commit's NEW sha.

**What is rewritten.** A token is a run of 7-40 lowercase hex digits not touching an ASCII letter or digit. It is
rewritten ONLY when it is a prefix of exactly one commit of the OLD universe (every commit object of the original
repository, reachable or not) and that commit is in the map. Everything else is left alone and counted:

========================  ==================================================================================
``not_a_commit``          no old commit starts with it (a sha256 fragment, a seed, a blob id): not a citation
``unchanged_commit``      one old commit, and the rewrite did not change it
``ambiguous_unchanged``   several old commits, none rewritten
``ambiguous_changed``     several old commits, at least one rewritten: REFUSED and listed (cite more digits)
``embedded``              touches ``_`` or ``-``: a tag name, a worktree or directory name (``pin_9c033489``)
========================  ==================================================================================

The new token keeps the old one's LENGTH (so a byte-budgeted doc keeps its size), unless that prefix is ambiguous among
the NEW commits, when it is lengthened to the shortest unique one (``lengthened``).

**Where it does not write.** ``FROZEN`` paths (append-only or generated, and the map itself) and content-pinned files
(a file whose sha256 / blob id another tracked file cites: editing it would break that pin) are scanned and COUNTED
but never written; the report lists them, and ``utils.git_sha`` resolves their old citations. Binary files are skipped.
"""
from __future__ import annotations

import bisect
import hashlib
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from utils.history_rewrite.gitobjs import resolve_all, run_git

TOKEN = re.compile(rb"(?<![0-9A-Za-z])[0-9a-f]{7,40}(?![0-9A-Za-z])")
HEX64 = re.compile(rb"(?<![0-9A-Za-z])[0-9a-f]{64}(?![0-9A-Za-z])")
HEX40 = re.compile(rb"(?<![0-9A-Za-z])[0-9a-f]{40}(?![0-9A-Za-z])")

#: Never written: append-only / generated / the map itself. A prefix ending ``/`` is a directory.
FROZEN: tuple[str, ...] = (
    "designs/ops/history_rewrite_sha_map.tsv",       # the map (rewriting its old column would destroy it)
    "designs/research_state/ledger.md",              # the research ledger: never edited
    "designs/research_state/ledger_index.md",        # GENERATED from the ledger (python -m main.ledger_index)
    "designs/research_state/claude_md_archive/",     # append-only archive
    "designs/CHANGELOG.md",                          # append-only: a new entry per version
)
SKIP_ADJACENT = (b"_", b"-")
MAX_FILE_BYTES = 64 * 1024 * 1024


def is_frozen(path: str, frozen: Iterable[str] = FROZEN) -> bool:
    return any(path == f or (f.endswith("/") and path.startswith(f)) for f in frozen)


class CommitIndex:
    """Sorted full ids of a set of commits; ``prefix(p)`` lists those starting with ``p`` (bisect, no scan)."""

    def __init__(self, oids: Iterable[str]):
        self.oids = sorted(set(oids))

    def prefix(self, p: str) -> list[str]:
        i = bisect.bisect_left(self.oids, p)
        out = []
        while i < len(self.oids) and self.oids[i].startswith(p):
            out.append(self.oids[i])
            i += 1
        return out

    def __contains__(self, oid: str) -> bool:
        return bool(self.prefix(oid)) and oid in self.prefix(oid)

    def shortest_unique(self, full: str, at_least: int) -> str:
        for n in range(at_least, 41):
            if len(self.prefix(full[:n])) == 1:
                return full[:n]
        return full


@dataclass(frozen=True)
class Decision:
    kind: str  # rewritten | lengthened | not_a_commit | unchanged_commit | ambiguous_unchanged | ambiguous_changed
    new: Optional[str] = None  # the replacement token (rewritten / lengthened)
    old_full: Optional[str] = None
    candidates: tuple[str, ...] = ()


NOT_A_COMMIT = Decision("not_a_commit")


class Rewriter:
    def __init__(self, mapping: dict[str, str], old_index: CommitIndex, new_index: CommitIndex):
        self.map = mapping
        self.old = old_index
        self.new = new_index
        self._memo: dict[str, Decision] = {}

    def decide(self, tok: str) -> Decision:
        d = self._memo.get(tok)
        if d is not None:
            return d
        matches = self.old.prefix(tok)
        if not matches:
            return NOT_A_COMMIT      # not memoised: millions of distinct hex words (sha256 fragments, ids) would fill memory
        elif len(matches) > 1:
            d = (Decision("ambiguous_changed", candidates=tuple(matches)) if any(m in self.map for m in matches)
                 else Decision("ambiguous_unchanged"))
        else:
            old_full = matches[0]
            if old_full not in self.map:
                d = Decision("unchanged_commit")
            else:
                new_full = self.map[old_full]
                short = new_full[:len(tok)]
                if len(self.new.prefix(short)) == 1:
                    d = Decision("rewritten", new=short, old_full=old_full)
                else:
                    d = Decision("lengthened", new=self.new.shortest_unique(new_full, len(tok)), old_full=old_full)
        self._memo[tok] = d
        return d

    def rewrite(self, data: bytes, *, apply: bool = True) -> tuple[bytes, Counter, list[tuple[int, str, Decision]]]:
        """``(new data, occurrences by decision kind, [(byte offset, token, decision)] of refusals)``."""
        kinds: Counter = Counter()
        refusals: list[tuple[int, str, Decision]] = []

        def sub(m: re.Match[bytes]) -> bytes:
            tok = m.group().decode()
            s, e = m.span()
            if (s > 0 and data[s - 1:s] in SKIP_ADJACENT) or data[e:e + 1] in SKIP_ADJACENT:
                if self.decide(tok).kind in ("rewritten", "lengthened", "ambiguous_changed"):
                    kinds["embedded"] += 1
                return m.group()
            d = self.decide(tok)
            kinds[d.kind] += 1
            if d.kind == "ambiguous_changed":
                refusals.append((s, tok, d))
            if d.new is not None:
                return d.new.encode()
            return m.group()

        out = TOKEN.sub(sub, data)
        return (out if apply else data), kinds, refusals


@dataclass
class FileFacts:
    path: str
    kinds: Counter
    refusals: list[tuple[int, int, str]]  # (line, offset, token)


@dataclass
class CiteResult:
    files_scanned: int = 0
    files_binary: int = 0
    files_changed: int = 0
    written: bool = False
    kinds: Counter = field(default_factory=Counter)  # over rewritable files
    per_file: dict[str, Counter] = field(default_factory=dict)  # rewritable files with an edit
    frozen: dict[str, Counter] = field(default_factory=dict)  # frozen files: what WOULD have been rewritten
    pinned: dict[str, Counter] = field(default_factory=dict)
    pinned_by: dict[str, list[str]] = field(default_factory=dict)  # pinned file -> files citing its hash
    refusals: list[tuple[str, int, str, tuple[str, ...]]] = field(default_factory=list)  # (path, line, token, candidates)
    edits: dict[str, str] = field(default_factory=dict)  # old token -> new token (distinct, rewritten + lengthened)
    lengthened: dict[str, str] = field(default_factory=dict)

    @property
    def rewritten_tokens(self) -> int:
        return self.kinds["rewritten"] + self.kinds["lengthened"]


def tracked_files(tree: str) -> list[tuple[str, str]]:
    """``[(path, blob oid)]`` of regular tracked files in ``tree`` (symlinks and gitlinks excluded)."""
    out = run_git(tree, "ls-files", "-s", "-z").decode(errors="replace")
    files = []
    for rec in out.split("\0"):
        if not rec:
            continue
        meta, _, path = rec.partition("\t")
        mode, oid, _stage = meta.split()
        if mode in ("100644", "100755"):
            files.append((path, oid))
    return files


def find_pinned(tree: str, files: list[tuple[str, str]], texts: dict[str, bytes]) -> dict[str, list[str]]:
    """``{pinned path: [files citing its hash]}``: a tracked file whose sha256 / git blob id another tracked file names."""
    cited: dict[str, set[str]] = {}
    for path, data in texts.items():
        for rx in (HEX64, HEX40):
            for m in rx.finditer(data):
                cited.setdefault(m.group().decode(), set()).add(path)
    pinned: dict[str, list[str]] = {}
    for path, oid in files:
        body = texts.get(path)
        keys = [oid]
        if body is not None:
            keys.append(hashlib.sha256(body).hexdigest())
        for k in keys:
            citers = cited.get(k, set()) - {path}
            if citers:
                pinned.setdefault(path, []).extend(sorted(citers))
    return pinned


def rewrite_tree(tree: str, rw: Rewriter, *, apply: bool, frozen: Iterable[str] = FROZEN,
                 pin_guard: bool = True) -> CiteResult:
    """Run the rewriter over every tracked text file of ``tree`` (``apply`` writes; otherwise a dry run)."""
    res = CiteResult(written=apply)
    files = tracked_files(tree)
    texts: dict[str, bytes] = {}
    for path, _oid in files:
        full = os.path.join(tree, path)
        try:
            if os.path.getsize(full) > MAX_FILE_BYTES:
                continue
            data = Path(full).read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            res.files_binary += 1
            continue
        texts[path] = data
    pinned = find_pinned(tree, files, texts) if pin_guard else {}
    res.pinned_by = pinned
    for path, data in texts.items():
        res.files_scanned += 1
        out, kinds, refusals = rw.rewrite(data, apply=True)
        if not kinds and not refusals:
            continue
        bucket = res.frozen if is_frozen(path, frozen) else res.pinned if path in pinned else None
        if bucket is not None:
            if kinds["rewritten"] or kinds["lengthened"] or kinds["ambiguous_changed"] or kinds["embedded"]:
                bucket[path] = kinds
            continue
        res.kinds.update(kinds)
        for off, tok, d in refusals:
            res.refusals.append((path, data.count(b"\n", 0, off) + 1, tok, d.candidates))
        if out != data:
            res.files_changed += 1
            res.per_file[path] = kinds
            if apply:
                Path(os.path.join(tree, path)).write_bytes(out)
    for tok, d in rw._memo.items():
        if d.new is not None:
            res.edits[tok] = d.new
            if d.kind == "lengthened":
                res.lengthened[tok] = d.new
    return res


def verify_edits(res: CiteResult, rw: Rewriter, new_repo: str) -> list[str]:
    """Problems with the distinct (old token -> new token) edits, checked against GIT's own resolution in ``new_repo``.

    Each new token must resolve (``<tok>^{commit}``) to exactly the mapped new commit of the old commit it replaced.
    """
    problems = []
    revs = {new: f"{new}^{{commit}}" for new in set(res.edits.values())}
    got = resolve_all(new_repo, revs.values())
    for old_tok, new_tok in res.edits.items():
        want = rw.map[rw.decide(old_tok).old_full or ""]
        have = got.get(revs[new_tok])
        if have != want:
            problems.append(f"{old_tok} -> {new_tok}: git says {have}, the map says {want[:12]}")
    return problems
