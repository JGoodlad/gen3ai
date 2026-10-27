"""What the rewrite CANNOT re-point: the report the operator reads before the force-push.

Everything here is READ-ONLY. Sections:

1. **Tracked files left on old SHAs** (from the citation pass): frozen paths, content-pinned files, refused ambiguous
   tokens, identifier-embedded tokens.
2. **Refs**: which local branches / tags / remote-tracking refs move, and tag NAMES that embed an old SHA (kept).
3. **Worktrees**: each linked worktree's HEAD and what to do with it (a worktree pinned to a rewritten commit must be
   re-created at the new SHA; its directory name may embed the old one).
4. **Files outside git**: run sidecars in ``models/`` (IMMUTABLE: ``original_command`` / ``pin_history`` / ``lineage``),
   the archive, scratch directories: counts of citations of rewritten commits, by JSON field where there is one.
5. **Outside this machine**: GitHub's own copies of the old commits, forks, other clones. Static text: nothing here can
   be read or changed from the box.

A "citation of a rewritten commit" is a token the :class:`Rewriter` would rewrite (or refuse as ambiguous), so the
numbers are the size of the gap ``utils.git_sha`` has to cover.
"""
from __future__ import annotations

import fnmatch
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from utils.history_rewrite.cite import TOKEN, CiteResult, Rewriter
from utils.history_rewrite.gitobjs import run_git

RELEVANT = ("rewritten", "lengthened", "ambiguous_changed")
TEXT_SUFFIXES = (".json", ".jsonl", ".md", ".txt", ".log", ".tsv", ".csv", ".yaml", ".yml", ".sh", ".py", ".toml")


@dataclass
class ScanSpec:
    label: str
    root: str
    patterns: tuple[str, ...] = ("**/*",)  # fnmatch against the path relative to root
    max_bytes: int = 16 * 1024 * 1024


@dataclass
class ScanSummary:
    spec: ScanSpec
    files_scanned: int = 0
    files_hit: int = 0
    tokens_hit: int = 0
    commits: set[str] = field(default_factory=set)
    by_field: Counter = field(default_factory=Counter)
    by_subdir: Counter = field(default_factory=Counter)
    ambiguous: int = 0
    unreadable: int = 0


def _json_fields(node: object, rw: Rewriter, path: str, out: Counter, commits: set[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            _json_fields(v, rw, f"{path}.{k}" if path else k, out, commits)
    elif isinstance(node, list):
        for v in node:
            _json_fields(v, rw, path + "[]", out, commits)
    elif isinstance(node, str):
        for m in TOKEN.finditer(node.encode(errors="replace")):
            d = rw.decide(m.group().decode())
            if d.kind in ("rewritten", "lengthened"):
                out[path] += 1
                if d.old_full:
                    commits.add(d.old_full)


def scan(spec: ScanSpec, rw: Rewriter) -> ScanSummary:
    """Count the citations of rewritten commits in the files ``spec`` selects (never opens a file for writing)."""
    s = ScanSummary(spec)
    root = Path(spec.root)
    if not root.is_dir():
        return s
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = os.path.relpath(dirpath, root)
        dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
        for fn in filenames:
            rel = fn if rel_dir == "." else os.path.join(rel_dir, fn)
            if not any(fnmatch.fnmatch(rel, p) for p in spec.patterns):
                continue
            full = os.path.join(dirpath, fn)
            try:
                if os.path.islink(full) or os.path.getsize(full) > spec.max_bytes:
                    continue
                data = Path(full).read_bytes()
            except OSError:
                s.unreadable += 1
                continue
            if b"\0" in data[:8192]:
                continue
            s.files_scanned += 1
            hits = 0
            for m in TOKEN.finditer(data):
                d = rw.decide(m.group().decode())
                if d.kind in ("rewritten", "lengthened"):
                    hits += 1
                    if d.old_full:
                        s.commits.add(d.old_full)
                elif d.kind == "ambiguous_changed":
                    s.ambiguous += 1
            if not hits:
                continue
            s.files_hit += 1
            s.tokens_hit += hits
            s.by_subdir[rel.split(os.sep)[0]] += hits
            if fn.endswith(".json"):
                try:
                    _json_fields(json.loads(data), rw, "", s.by_field, set())
                except ValueError:
                    s.by_field["(unparseable json)"] += hits
    return s


def render_scan(s: ScanSummary, top: int = 12) -> list[str]:
    lines = [f"- `{s.spec.label}` ({s.spec.root}; {', '.join(s.spec.patterns)}): {s.files_scanned} text files scanned, "
             f"{s.files_hit} cite a rewritten commit ({s.tokens_hit} citations, {len(s.commits)} distinct commits"
             + (f", {s.ambiguous} ambiguous" if s.ambiguous else "") + (f", {s.unreadable} unreadable" if s.unreadable else "") + ")"]
    for f, n in s.by_field.most_common(top):
        lines.append(f"    - field `{f}`: {n}")
    for d, n in s.by_subdir.most_common(top):
        lines.append(f"    - under `{d}`: {n}")
    return lines


def worktree_rows(repo: str, pairs: dict[str, str], models_dir: Optional[str] = None, known: Optional[set[str]] = None) -> list[dict]:
    """One row per worktree: path, HEAD, branch, locked, what is using it, and the action."""
    rows: list[dict] = []
    cur: dict = {}
    out = run_git(repo, "worktree", "list", "--porcelain").decode(errors="replace")
    for line in out.splitlines() + [""]:
        if not line:
            if cur:
                rows.append(cur)
            cur = {}
            continue
        key, _, val = line.partition(" ")
        cur[key] = val if val else True
    for n, r in enumerate(rows):
        head = r.get("HEAD", "")
        r["main"] = n == 0  # `git worktree list` prints the primary checkout first
        r["new"] = pairs.get(head)
        r["busy"] = worktree_busy(r["worktree"], models_dir) if r["new"] and not r["main"] else []
        r["embeds"] = [t for t in (m.group().decode() for m in TOKEN.finditer(r.get("worktree", "").encode()))
                       if any(o.startswith(t) for o in pairs)]
        if r["new"] and r["main"]:
            r["action"] = f"the primary checkout: move its branch to {r['new'][:12]} (clean first, `git reset --hard` to the new tip)"
        elif r["new"]:
            r["action"] = (f"move in place: reset --keep {r['new'][:12]} (its branch moves with it)" if r.get("branch")
                           else f"move in place: checkout --detach {r['new'][:12]}")
        elif known is not None and head not in known:
            r["action"] = STALE
        else:
            r["action"] = "none: HEAD is not a rewritten commit"
    return rows


def worktree_busy(path: str, models_dir: Optional[str] = None) -> list[str]:
    """Why a worktree must NOT be removed or moved now: a process stands in it, or a run directory links into it.

    Read-only (``/proc/*/cwd`` and the top level of ``models/``): the check that would have prevented the worktree
    removals that destroyed runs (a ``models/<run>`` symlink into a worktree dies with it).
    """
    reasons: list[str] = []
    real = os.path.realpath(path)
    for entry in os.scandir("/proc") if os.path.isdir("/proc") else []:
        if entry.name.isdigit():
            try:
                cwd = os.path.realpath(os.readlink(f"/proc/{entry.name}/cwd"))
            except OSError:
                continue
            if cwd == real or cwd.startswith(real + os.sep):
                reasons.append(f"pid {entry.name} has its cwd inside")
    if models_dir and os.path.isdir(models_dir):
        for e in os.scandir(models_dir):
            if e.is_symlink():
                target = os.path.realpath(e.path)
                if target == real or target.startswith(real + os.sep):
                    reasons.append(f"models/{e.name} links into it")
    return reasons


STALE = "NOT IN THE SNAPSHOT (moved or created after the clone was taken: redo the dry run)"


def ref_rows(repo: str, pairs: dict[str, str], rw: Rewriter, known: Optional[set[str]] = None) -> dict[str, list[str]]:
    """Local refs by what the rewrite does to them (``known`` = every commit of the clone's snapshot: a tip outside it is stale)."""
    out = run_git(repo, "for-each-ref", "--format=%(refname)\t%(objecttype)\t%(objectname)\t%(*objectname)").decode()
    moved: list[str] = []
    stay: list[str] = []
    stale: list[str] = []
    remote: list[str] = []
    tag_names: list[str] = []
    for line in out.splitlines():
        ref, _typ, oid, peeled = (line.split("\t") + ["", "", ""])[:4]
        tip = peeled or oid
        bucket = (remote if ref.startswith("refs/remotes/") else moved if tip in pairs
                  else stale if known is not None and tip not in known else stay)
        bucket.append(ref)
        if ref.startswith("refs/tags/"):
            for m in TOKEN.finditer(ref.encode()):
                if rw.decide(m.group().decode()).kind in RELEVANT:
                    tag_names.append(ref)
                    break
    rows = {"moved": moved, "unchanged": stay, "remote-tracking (stale after the push; `git fetch --prune`)": remote,
            "tag names that embed an old SHA (the name is kept)": tag_names}
    if stale:
        rows[STALE] = stale
    return rows


OUTSIDE = (
    "GitHub keeps every old commit reachable BY SHA after a force-push (the web UI, API and `git fetch <sha>`) until its own "
    "garbage collection runs; only GitHub Support can purge them and the cached views (open a ticket naming the old SHAs "
    "of the two introducing commits).",
    "Forks and other clones of the repository (anyone who ever cloned or forked it) still hold the names; nothing here reaches them.",
    "Other copies on this machine: `~/gen3ai_archive` and any bundle, tarball or `git clone` made before the rewrite hold "
    "the old objects; this report only COUNTS citations in text files there.",
    "A `git bundle` / mirror taken as the rollback copy keeps the names by design; delete it only when the owner is satisfied.",
    "Local object database: after the refs move, the old commits stay reachable from the reflog and from any worktree "
    "still on an old SHA; `git reflog expire --expire=now --all && git gc --prune=now` removes them once none is left.",
)


def render_report(cite: CiteResult, *, refs: Optional[dict[str, list[str]]], worktrees: Optional[list[dict]],
                  scans: Iterable[ScanSummary], mapped_commits: int) -> str:
    L = ["# History rewrite: what cannot be re-pointed", "",
         f"{mapped_commits} commits change SHA. Every citation below stays on an OLD sha; `python -m utils.git_sha <sha>` "
         "resolves it through `designs/ops/history_rewrite_sha_map.tsv`.", ""]
    L += ["## 1. Tracked files left on old SHAs", ""]
    for title, bucket in (("frozen (append-only / generated / the map)", cite.frozen), ("content-pinned (another file cites their hash)", cite.pinned)):
        total = sum(sum(c[k] for k in RELEVANT + ("embedded",)) for c in bucket.values())
        L.append(f"- {title}: {len(bucket)} files, {total} citations")
        for p, c in sorted(bucket.items(), key=lambda kv: -sum(kv[1][k] for k in RELEVANT))[:15]:
            n = sum(c[k] for k in RELEVANT)
            by = f"; cited by {', '.join(cite.pinned_by.get(p, [])[:3])}" if bucket is cite.pinned else ""
            L.append(f"    - `{p}`: {n} rewritable citations{by}")
    L.append(f"- refused as AMBIGUOUS (cite more digits, then re-run): {len(cite.refusals)}")
    for path, line, tok, cands in cite.refusals[:30]:
        L.append(f"    - `{path}:{line}` `{tok}` matches {len(cands)} old commits")
    L.append(f"- identifier-embedded (touch `_` or `-`: tag names, worktree / directory names, run ids) and so left alone: "
             f"{cite.kinds['embedded']}")
    L.append(f"- lengthened to stay unique among the new commits: {len(cite.lengthened)}")
    L.append("")
    if refs is not None:
        L += ["## 2. Refs", ""]
        for k, v in refs.items():
            L.append(f"- {k}: {len(v)}" + (f" ({', '.join(f'`{r}`' for r in v[:6])}{' ...' if len(v) > 6 else ''})" if v and len(v) <= 40 else ""))
        L.append("")
    if worktrees is not None:
        L += ["## 3. Worktrees", ""]
        need = [w for w in worktrees if w["new"]]
        L.append(f"- {len(worktrees)} worktrees (primary included); {len(need)} sit on a rewritten commit")
        for w in need:
            L.append(f"    - `{w['worktree']}` HEAD {w['HEAD'][:12]}: {w['action']}"
                     + (" [LOCKED]" if w.get("locked") else "") + (f" [dir name embeds {w['embeds']}]" if w["embeds"] else "")
                     + (f" [BUSY: {'; '.join(w['busy'])}]" if w["busy"] else ""))
        L.append("")
    L += ["## 4. Files outside git (cannot be edited; resolve with `utils.git_sha`)", ""]
    for s in scans:
        L += render_scan(s)
    L += ["", "## 5. Outside this machine", ""] + [f"- {t}" for t in OUTSIDE]
    return "\n".join(L) + "\n"


def local_plan(repo: str, pairs: dict[str, str], models_dir: Optional[str] = None) -> list[str]:
    """The commands that move THIS repository's refs and worktrees onto the rewritten history (printed, never run).

    Order: ``git fetch`` brings the pushed objects and tags in, then each local ref / worktree is moved with a command that
    REFUSES rather than clobbers: ``update-ref <ref> <new> <old>`` checks the old value, ``reset --keep`` /
    ``checkout --detach`` refuse over conflicting local changes. A worktree is moved IN PLACE (its directory, untracked
    build output and ``models/`` links survive; only files whose content differs between the old and new commit change),
    never removed and re-created, and one that is BUSY is marked so.
    """
    L = ["# generated by `python -m utils.history_rewrite local-plan`: REVIEW, then run. Each line is idempotent-or-refusing.",
         "set -euo pipefail", f"cd {repo}", "git fetch --force --prune origin '+refs/heads/*:refs/remotes/origin/*' '+refs/tags/*:refs/tags/*'", ""]
    wts = {w.get("branch", "").removeprefix("refs/heads/"): w for w in worktree_rows(repo, pairs, models_dir) if w.get("branch")}
    out = run_git(repo, "for-each-ref", "--format=%(refname)\t%(objectname)", "refs/heads").decode().splitlines()
    L.append("# local branches (not checked out anywhere): the old value is the lease")
    for line in out:
        ref, _, oid = line.partition("\t")
        br = ref.removeprefix("refs/heads/")
        if oid in pairs and br not in wts:
            L.append(f"git update-ref {ref} {pairs[oid]} {oid}")
    L.append("")
    L.append("# worktrees, moved in place (clean + not busy first; --keep / checkout refuse over local changes)")
    for w in worktree_rows(repo, pairs, models_dir):
        if not w["new"]:
            continue
        flags = (" # PRIMARY CHECKOUT: clean first" if w["main"] else "") + (f" # BUSY: {'; '.join(w['busy'])}: DO NOT RUN YET" if w["busy"] else "")
        cmd = (f"git -C {w['worktree']} reset --keep {w['new']}" if w.get("branch") else f"git -C {w['worktree']} checkout --detach {w['new']}")
        L.append(f"test -z \"$(git -C {w['worktree']} status --porcelain)\" && {cmd}{flags}")
    L += ["", "# PHASE 2, only after every call site that reads a recorded sha goes through utils.git_sha and the owner agrees:",
          "#   git reflog expire --expire=now --all && git gc --prune=now      # drops the old objects (and the names) locally"]
    return L


def push_plan(repo: str, pairs: dict[str, str], clone: str) -> list[str]:
    """The ONE atomic force-push of the rewritten refs (printed, never run; reads only local refs, no network).

    Only refs that exist on origin (as ``repo``'s ``refs/remotes/origin/*`` and ``refs/tags/*`` last saw them) AND whose tip
    moved are pushed. Each is leased on its OLD value (``--force-with-lease=<ref>:<old>``), so a push that raced a newer
    origin state is refused. ``refs/history-rewrite/`` (the keep-refs) is local bookkeeping and is never pushed.
    """
    url = run_git(repo, "remote", "get-url", "origin").decode().strip()
    leases: list[str] = []
    specs: list[str] = []
    notes: list[str] = []
    for line in run_git(repo, "for-each-ref", "--format=%(refname)\t%(objectname)", "refs/remotes/origin").decode().splitlines():
        ref, _, oid = line.partition("\t")
        name = ref.removeprefix("refs/remotes/origin/")
        if name == "HEAD" or oid not in pairs:
            continue
        if run_git(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False).strip():
            leases.append(f"--force-with-lease=refs/heads/{name}:{oid}")
            specs.append(f"refs/heads/{name}:refs/heads/{name}")
        else:
            notes.append(f"# origin/{name} holds a rewritten commit but there is NO local branch {name}: `git branch {name} origin/{name}` and redo the rewrite")
    for line in run_git(repo, "for-each-ref", "--format=%(refname)\t%(objectname)\t%(*objectname)", "refs/tags").decode().splitlines():
        ref, _, rest = line.partition("\t")
        oid, _, peeled = rest.partition("\t")
        if (peeled or oid) in pairs:
            leases.append(f"--force-with-lease={ref}:{oid}")
            specs.append(f"{ref}:{ref}")
    L = ["# generated by `python -m utils.history_rewrite push-plan`: NOT RUN BY ANY TOOL. The owner runs it, once, after § Preconditions.",
         f"cd {clone}", f"git remote add origin {url}        # filter-repo removed it on purpose", *notes]
    if specs:
        L.append("git push --atomic \\")
        L += [f"  {x} \\" for x in leases]
        L.append("  origin " + " ".join(specs))
    return L
