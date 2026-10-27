"""The account names to purge: derive them, match them, and never print them.

**The names are data, not source.** They live ONLY in a file outside the repository (or gitignored, mode 0600): this
module refuses to write one inside a work tree that does not ignore it, and every report it builds speaks of an
identity by its INDEX (``#07``), never by its text. ``redact`` is the one place a form could leak into a message and
it replaces it.

**Derivation (the rule, so a reader can check it).** Commit 14ae4203 introduced
``designs/research_state/measurements/human_replay_quality_2026-10-25/results/results.json``, whose ``accounts``
block carried raw Showdown account names as the keys and ``name`` values of ``flagged`` (61) and the ``name`` of each
``top25`` entry (25, of which 9 are also flagged). Commit 742decb2 added prose beside it. An *identity* is one
Showdown account (``toid`` form: lowercase alphanumerics); its *forms* are the spellings found for it (the id key,
the display name). ``audit`` then searches ALL history for every form, so a spelling the derivation missed in the prose
shows up as a hit by path.

**Matching.** Case-insensitive, and a hit is a form not touching an ASCII letter or digit on either side (so the id
``abc`` is not found inside ``abcdef``). A form found INSIDE a longer token anywhere is reported as *contaminated*;
``rules`` then replaces it with a boundary-checked regex instead of a plain literal.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from utils.history_rewrite.gitobjs import reachable_objects, run_git, stream_objects

SCHEMA = "gen3_history_rewrite_names_v1"
#: The commits that introduced the names (the brief's), by the abbreviation the docs cite.
INTRODUCING = ("14ae4203", "742decb2")
RESULTS_PATH = "designs/research_state/measurements/human_replay_quality_2026-10-25/results/results.json"
#: Replacement text for identity N (stable, distinct per identity so a JSON object keyed by name keeps distinct keys).
REPLACEMENT_FMT = "redacted-account-{idx:02d}"
MIN_FORM_LEN = 4
_ALNUM = frozenset(b"abcdefghijklmnopqrstuvwxyz0123456789")


def toid(text: str) -> str:
    """Showdown's id of a name: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


@dataclass(frozen=True)
class Identity:
    idx: int
    tid: str
    forms: tuple[str, ...]
    tier: str  # "flagged" (the owner-approved 61) | "top25" (only in the top-25 list: the owner's call)

    @property
    def replacement(self) -> bytes:
        return REPLACEMENT_FMT.format(idx=self.idx).encode()


def derive_identities(repo: str, commit: str = INTRODUCING[0]) -> list[Identity]:
    """The identities in ``results.json`` as ``commit`` introduced it (see the module docstring for the rule)."""
    blob = run_git(repo, "show", f"{commit}:{RESULTS_PATH}")
    accounts = json.loads(blob)["accounts"]
    forms: dict[str, set[str]] = {}
    tier: dict[str, str] = {}
    for key, rec in accounts["flagged"].items():
        t = toid(key)
        forms.setdefault(t, set()).update({key, rec["name"]})
        tier[t] = "flagged"
    for rec in accounts["top25"]:
        t = toid(rec["name"])
        forms.setdefault(t, set()).add(rec["name"])
        tier.setdefault(t, "top25")
    out = []
    for i, t in enumerate(sorted(forms), start=1):
        fs = tuple(sorted(forms[t]))
        if any(len(f) < MIN_FORM_LEN or not f.isascii() or "\n" in f or "==>" in f for f in fs):
            raise ValueError(f"identity #{i:02d} has a form too short / non-ASCII / unquotable to match safely: refuse")
        out.append(Identity(i, t, fs, tier[t]))
    return out


def introducing_files(repo: str, commits: Iterable[str]) -> set[str]:
    """The paths the introducing commits touched (the candidates for the account-data scope)."""
    files: set[str] = set()
    for c in commits:
        files |= set(run_git(repo, "show", "--format=", "--name-only", c).decode(errors="replace").splitlines())
    return files


def assert_out_of_repo(path: str | os.PathLike[str], repo: str) -> None:
    """Refuse a names file inside ``repo``'s work tree unless git ignores it (the names must never be committable)."""
    p = Path(path).resolve()
    top = Path(run_git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top == p or top in p.parents:
        ignored = subprocess.run(["git", "-C", str(top), "check-ignore", "-q", str(p)]).returncode == 0
        if not ignored:
            raise ValueError(f"{p} is inside {top} and not gitignored: the names file must live outside the repo")


def save_identities(path: str | os.PathLike[str], identities: list[Identity], repo: str) -> None:
    assert_out_of_repo(path, repo)
    payload = {"schema": SCHEMA, "introducing": list(INTRODUCING),
               "identities": [{"idx": i.idx, "tid": i.tid, "forms": list(i.forms), "tier": i.tier} for i in identities]}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=1)
    os.chmod(p, 0o600)


def load_identities(path: str | os.PathLike[str], tiers: Optional[Iterable[str]] = None) -> list[Identity]:
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} file")
    keep = set(tiers) if tiers else None
    return [Identity(r["idx"], r["tid"], tuple(r["forms"]), r["tier"]) for r in data["identities"]
            if keep is None or r["tier"] in keep]


def redact(text: str, identities: list[Identity]) -> str:
    """``text`` with every form replaced by its ``#NN``: for any message built from data that might hold a name."""
    for ident in identities:
        for f in sorted(ident.forms, key=len, reverse=True):
            text = re.sub(re.escape(f), f"#{ident.idx:02d}", text, flags=re.I)
    return text


# ---------------------------------------------------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------------------------------------------------
def _trie_pattern(words: list[bytes]) -> bytes:
    """A regex alternation of ``words`` as a prefix trie, LONGEST first (an optional tail is greedy)."""
    root: dict = {}
    for w in words:
        node = root
        for c in w:
            node = node.setdefault(c, {})
        node[None] = True

    def emit(node: dict) -> bytes:
        alts = [re.escape(bytes([c])) + emit(node[c]) for c in sorted(k for k in node if k is not None)]
        if not alts:
            return b""
        if len(alts) == 1 and None not in node:
            return alts[0]
        return b"(?:" + b"|".join(alts) + b")" + (b"?" if None in node else b"")

    return b"(?:" + b"|".join(re.escape(bytes([c])) + emit(root[c]) for c in sorted(root)) + b")"


@dataclass(frozen=True)
class Hit:
    start: int
    end: int
    idx: int
    spelling: bytes  # the exact bytes found (original case)
    boundary_ok: bool


class NameMatcher:
    """Find every identity form in a byte string; ``boundary_ok`` says whether it stands alone (see the module doc)."""

    def __init__(self, identities: list[Identity]):
        self.by_form: dict[bytes, int] = {}
        for ident in identities:
            for f in {*ident.forms, ident.tid}:
                self.by_form[f.lower().encode()] = ident.idx
        self._lens = sorted({len(f) for f in self.by_form}, reverse=True)
        self._rx = re.compile(_trie_pattern(sorted(self.by_form)))

    def scan(self, data: bytes) -> list[Hit]:
        low = data.lower()
        hits: list[Hit] = []
        pos, n = 0, len(low)
        while True:
            m = self._rx.search(low, pos)
            if m is None:
                break
            s, e = m.span()
            left_ok = s == 0 or low[s - 1] not in _ALNUM
            chosen = None
            # the longest form first; on a failed boundary try the shorter forms that share this start
            for ln in self._lens:
                if ln > e - s or s + ln > n:
                    continue
                form = low[s:s + ln]
                idx = self.by_form.get(form)
                if idx is None:
                    continue
                right_ok = s + ln == n or low[s + ln] not in _ALNUM
                if left_ok and right_ok:
                    chosen = (s + ln, idx, True)
                    break
                if chosen is None:
                    chosen = (s + ln, idx, False)
            if chosen is None:
                pos = s + 1
                continue
            end, idx, ok = chosen
            hits.append(Hit(s, end, idx, data[s:end], ok))
            pos = end
        return hits


# ---------------------------------------------------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class IdentityHits:
    blobs: set[str] = field(default_factory=set)
    paths: set[str] = field(default_factory=set)
    spellings: Counter = field(default_factory=Counter)  # exact bytes -> boundary-OK occurrences
    contaminated: Counter = field(default_factory=Counter)  # exact bytes -> occurrences inside a longer token (NOT hits)
    pre_existing_blobs: int = 0  # hit blobs already reachable BEFORE the introducing commit


@dataclass
class Audit:
    scanned_blobs: int = 0
    scanned_bytes: int = 0
    per_identity: dict[int, IdentityHits] = field(default_factory=dict)
    commit_messages: Counter = field(default_factory=Counter)  # idx -> commits whose message holds the identity
    tag_messages: Counter = field(default_factory=Counter)
    ref_names: Counter = field(default_factory=Counter)
    paths: Counter = field(default_factory=Counter)  # idx -> tracked path names holding the identity

    @property
    def total_hits(self) -> int:
        return (sum(len(h.blobs) for h in self.per_identity.values()) + sum(self.commit_messages.values())
                + sum(self.tag_messages.values()) + sum(self.ref_names.values()) + sum(self.paths.values()))

    def summary_lines(self, identities: list[Identity]) -> list[str]:
        """The human summary; every path is passed through :func:`redact` (a path could itself hold a name)."""
        lines = [f"blobs scanned {self.scanned_blobs} ({self.scanned_bytes / 1e6:.0f} MB); identities hit in blobs "
                 f"{sum(1 for h in self.per_identity.values() if h.blobs)}; hit blobs "
                 f"{len(set().union(*[h.blobs for h in self.per_identity.values()])) if self.per_identity else 0}",
                 f"commit messages {sum(self.commit_messages.values())}; tag messages {sum(self.tag_messages.values())}; "
                 f"ref names {sum(self.ref_names.values())}; tracked path names {sum(self.paths.values())}"]
        pre = {i: h.pre_existing_blobs for i, h in self.per_identity.items() if h.pre_existing_blobs}
        lines.append(f"hit blobs that PRE-DATE the introducing commit: {sum(pre.values())} (identities {sorted(pre)})")
        contaminated = sorted(i for i, h in self.per_identity.items() if h.contaminated)
        lines.append(f"identities with a form inside a longer token (regex rule needed): {contaminated}")
        files: Counter = Counter()
        for h in self.per_identity.values():
            files.update(h.paths)
        lines.append(f"files holding a name (first path of each hit blob, {len(files)}):")
        lines += [f"  {n:>4} identities  {redact(p, identities)}" for p, n in files.most_common()]
        return lines


def audit_history(repo: str, identities: list[Identity], *, before: Optional[str] = INTRODUCING[0],
                  max_blob_bytes: int = 256 * 1024 * 1024, progress: Optional[Callable[[int], None]] = None) -> Audit:
    """Scan every reachable blob, commit message, tag message, ref name and tree path for the identities.

    ``before`` names the introducing commit: a hit blob also reachable from ``before^`` is counted as PRE-EXISTING
    (the derivation's claim, that the names arrived with that commit, would then be false for that identity).
    """
    matcher = NameMatcher(identities)
    audit = Audit(per_identity={i.idx: IdentityHits() for i in identities})

    def standing_alone(text: bytes) -> set[int]:
        return {h.idx for h in matcher.scan(text) if h.boundary_ok}

    objs = reachable_objects(repo, all_refs=True)
    old_objs = set(reachable_objects(repo, f"{before}^")) if before else set()
    n = 0
    for oid, typ, data in stream_objects(repo, objs):
        if typ == "commit":
            msg = data.split(b"\n\n", 1)[1] if b"\n\n" in data else b""
            for idx in standing_alone(msg):
                audit.commit_messages[idx] += 1
            continue
        if typ == "tag":
            for idx in standing_alone(data):
                audit.tag_messages[idx] += 1
            continue
        if typ != "blob":
            continue
        n += 1
        if progress and n % 5000 == 0:
            progress(n)
        if len(data) > max_blob_bytes:
            continue
        audit.scanned_blobs += 1
        audit.scanned_bytes += len(data)
        for h in matcher.scan(data):
            rec = audit.per_identity[h.idx]
            if h.boundary_ok:
                rec.spellings[h.spelling] += 1
                rec.blobs.add(oid)
                rec.paths.add(objs[oid])
            else:
                rec.contaminated[h.spelling] += 1
    for rec in audit.per_identity.values():
        rec.pre_existing_blobs = sum(1 for b in rec.blobs if b in old_objs)
    for path in set(objs.values()):
        for idx in standing_alone(path.encode()):
            audit.paths[idx] += 1
    refs = run_git(repo, "for-each-ref", "--format=%(refname)").decode().splitlines()
    for ref in refs:
        for idx in standing_alone(ref.encode()):
            audit.ref_names[idx] += 1
    return audit
