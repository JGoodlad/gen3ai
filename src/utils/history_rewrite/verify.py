"""Checks on a rewritten clone: the names are gone, the graph is intact, the map is sound.

Each ``check_*`` returns a :class:`Check` (name, ok, one-line detail, optional ``items``); a detail speaks of identities by
index and never prints a name. The set answers the dry run's questions:

* ``names``       no identity form in ANY reachable blob, commit message, tag message, ref name or path name;
* ``variants``    no identity ID hiding behind punctuation / spacing in any version of the files the two commits touched
                  (informational: a short id can appear by chance across a normalised page, so a hit is "review", not "fail");
* ``graph``       the commit count is unchanged and every map row is a (old, new) pair of commits with the same
                  author / committer / dates / parent count (nothing was pruned or reordered);
* ``changed_set`` the commits whose SHA changed are exactly the descendants of the introducing commit (a changed
                  commit outside that set is named: that is collateral damage to the SHA space, e.g. a stripped signature);
* ``tip_trees``   the tree at each shared branch tip is byte-identical to the original's (names were already gone from
                  the current files, so the purge must not touch the tip).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from utils.history_rewrite.gitobjs import reachable_objects, run_git, stream_objects
from utils.history_rewrite.names import INTRODUCING, Identity, NameMatcher, audit_history


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    items: list[str] = field(default_factory=list)
    informational: bool = False  # a miss is "REVIEW", never a failure (a short id can match by chance)

    def line(self) -> str:
        return f"[{'PASS' if self.ok else 'REVIEW' if self.informational else 'FAIL'}] {self.name}: {self.detail}"


def check_names(clone: str, identities: list[Identity], waived: dict[int, tuple[str, ...]] | None = None) -> Check:
    """No identity stands alone anywhere, except a SCOPED identity at the paths the rules waived (listed, never silent)."""
    waived = waived or {}
    a = audit_history(clone, identities, before=None)
    bad, left = [], []
    for idx, rec in a.per_identity.items():
        extra = sorted(p for p in rec.paths if p not in set(waived.get(idx, ())))
        if extra:
            bad.append(f"#{idx:02d} still stands alone in {len(extra)} path(s): {extra[:3]}")
        elif rec.blobs or (idx in waived and a.commit_messages[idx]):
            left.append(f"#{idx:02d} waived: {len(rec.paths)} path(s) and {a.commit_messages[idx]} commit message(s) left as they were")
        if a.commit_messages[idx] and idx not in waived:
            bad.append(f"#{idx:02d} still in {a.commit_messages[idx]} commit message(s)")
    for what, counter in (("tag message", a.tag_messages), ("ref name", a.ref_names), ("path name", a.paths)):
        bad += [f"#{i:02d} still in {n} {what}(s)" for i, n in counter.items() if n]
    return Check("names", not bad, f"{a.scanned_blobs} blobs ({a.scanned_bytes / 1e6:.0f} MB) + messages + refs + paths scanned; "
                                   f"{len(bad)} identities still present; {len(left)} scoped identities have waived occurrences",
                 bad + left)


def check_variants(clone: str, source: str, identities: list[Identity], commits: tuple[str, ...] = INTRODUCING) -> Check:
    """Normalised (lowercase, alphanumerics only) search of every version of the files the introducing commits touched."""
    paths = set()
    for c in commits:
        paths |= set(run_git(source, "show", "--format=", "--name-only", c).decode().splitlines())
    objs = reachable_objects(clone, all_refs=True)
    want = [o for o, p in objs.items() if p in paths]
    tids = {i.idx: i.tid.encode() for i in identities}
    hits: dict[int, set[str]] = {}
    for oid, typ, data in stream_objects(clone, want):
        if typ != "blob":
            continue
        norm = re.sub(rb"[^a-z0-9]", b"", data.lower())
        for idx, t in tids.items():
            if t in norm:
                hits.setdefault(idx, set()).add(objs[oid])
    detail = f"{len(want)} versions of {len(paths)} touched files; identities with a normalised-text hit: {sorted(hits)}"
    return Check("variants", not hits, detail, [f"#{i:02d} in {sorted(p)}" for i, p in sorted(hits.items())], informational=True)


def _commit_facts(repo: str, oids: list[str]) -> dict[str, tuple[str, ...]]:
    """``{oid: (author, committer, dates, parent count)}`` through one ``git log --no-walk``."""
    out = run_git(repo, "log", "--no-walk=unsorted", "--stdin", "--format=%H%x1f%an%x1f%ae%x1f%aI%x1f%cn%x1f%ce%x1f%cI%x1f%P",
                  stdin=("\n".join(oids) + "\n").encode()).decode(errors="replace")
    facts = {}
    for line in out.splitlines():
        f = line.split("\x1f")
        facts[f[0]] = (*f[1:7], str(len(f[7].split())))
    return facts


def check_graph(clone: str, source: str, pairs: dict[str, str], clone_all: dict[str, str]) -> Check:
    """Commit count and per-commit identity (see the module doc). ``clone_all`` is the FULL commit-map (all rows)."""
    n_old = len(clone_all)
    n_new = len(set(clone_all.values()))
    problems = []
    if n_old != n_new:
        problems.append(f"{n_old} old commits map onto {n_new} new ones")
    old_f = _commit_facts(source, list(pairs))
    new_f = _commit_facts(clone, list(pairs.values()))
    bad = [o[:12] for o, n in pairs.items() if old_f.get(o) != new_f.get(n)]
    if bad:
        problems.append(f"{len(bad)} changed commits differ in author / committer / date / parent count: {bad[:5]}")
    return Check("graph", not problems, f"{n_old} commits in the commit-map -> {n_new} distinct; {len(pairs)} changed; "
                                         f"metadata equal on all changed", problems)


def check_changed_set(source: str, pairs: dict[str, str], clone_all: dict[str, str], introducing: str) -> Check:
    """The commits whose SHA changed must be exactly the rewritten commits that descend from ``introducing``."""
    base = run_git(source, "rev-parse", f"{introducing}^{{commit}}").decode().strip()

    def descendants(oids: list[str]) -> set[str]:
        out = run_git(source, "rev-list", "--stdin", f"--ancestry-path={base}", f"^{base}^", stdin=("\n".join(oids) + "\n").encode())
        return set(out.decode().split())

    collateral = sorted(set(pairs) - descendants(list(pairs)))  # changed, yet not a descendant
    unchanged = sorted(descendants(list(clone_all)) - set(pairs))  # a descendant the rewrite left alone (the map is partial?)
    ok = not collateral and not unchanged
    return Check("changed_set", ok, f"{len(pairs)} changed commits; changed but NOT a descendant of the introducing commit "
                                    f"{len(collateral)}; descendant but unchanged {len(unchanged)}",
                 [f"collateral {c[:12]}" for c in collateral[:20]] + [f"unchanged {c[:12]}" for c in unchanged[:20]])


def check_tip_trees(clone: str, source: str, ref_map: list[tuple[str, str, str]], refs: tuple[str, ...] = ("refs/heads/main",)) -> Check:
    """The tree at each named ref's tip, old vs new, from filter-repo's own ``ref-map`` SNAPSHOT (``old new refname``).

    The snapshot, not the live source ref: ``main`` keeps moving while a dry run works on a clone, and a live tip the
    clone never had would read as a mismatch.
    """
    problems, same = [], 0
    for old_tip, new_tip, ref in ref_map:
        if ref not in refs:
            continue
        old_tree = run_git(source, "rev-parse", f"{old_tip}^{{tree}}", check=False).decode().strip()
        new_tree = run_git(clone, "rev-parse", f"{new_tip}^{{tree}}", check=False).decode().strip()
        if old_tree and old_tree == new_tree:
            same += 1
        else:
            problems.append(f"{ref}: tree {old_tree[:12] or 'missing'} -> {new_tree[:12] or 'missing'}")
    return Check("tip_trees", not problems and same > 0, f"{same} tip tree(s) byte-identical to the original", problems)


def check_tree_names(tree: str, identities: list[Identity], allow: frozenset[str] = frozenset()) -> Check:
    """No tracked or untracked-but-not-ignored file of ``tree`` holds an identity standing alone (run before every commit
    of this tooling: the names must never ride into one). ``allow`` = paths known to hold a waived occurrence."""
    matcher = NameMatcher(identities)
    listing = run_git(tree, "ls-files", "-co", "--exclude-standard", "-z").decode(errors="replace").split("\0")
    bad, scanned = [], 0
    for rel in filter(None, listing):
        try:
            with open(os.path.join(tree, rel), "rb") as f:
                data = f.read()
        except OSError:
            continue
        scanned += 1
        found = sorted({h.idx for h in matcher.scan(data) if h.boundary_ok})
        if found and rel not in allow:
            bad.append(f"{rel}: #{', #'.join(f'{i:02d}' for i in found)}")
    return Check("tree_names", not bad, f"{scanned} files scanned, {len(bad)} hold a name", bad)
