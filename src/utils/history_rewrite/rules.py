"""The ``git filter-repo`` rules, built from an audit of the history.

**Two kinds of identity.** Most account names are inert strings that occur ONLY in the account-data files the two
commits created (``SCOPE_DIR``). A few are not: one is a plain English phrase, one is the player in an
older public replay fixture, one is a Smogon forum curator's name. A global replacement would corrupt prose or
rewrite data that is not ours to rewrite. So:

* **global** identity (every standing-alone hit is in an account-data file): replaced EVERYWHERE, by plain
  ``literal:`` rules (a ``bytes.replace`` is memchr-fast; a case-insensitive regex per identity over 3.5 GB is hours)
  using the EXACT spellings the audit found;
* **scoped** identity (it ALSO stands alone in a blob that predates the introducing commit, or outside the
  account-data files): replaced ONLY inside the account-data files, by a boundary-checked regex in a
  ``--file-info-callback`` (the one filter-repo hook that sees the path). Its other occurrences are WAIVED: the
  audit and ``verify`` list them by identity index and path so the owner can overrule.

A spelling that also occurs INSIDE a longer token somewhere (``red fixed`` for ``red f``) can never be a literal: it
gets a boundary regex (global identity) or stays scoped.

filter-repo runs the literals, then the regexes, over a text blob (no NUL in its first 8 KB); a binary blob is not
touched, which is why ``verify`` rescans every blob afterwards rather than trusting the rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from utils.history_rewrite.names import Audit, Identity

#: The account-data files the two commits created live here; a scoped identity is replaced only in them.
SCOPE_DIR = "designs/research_state/measurements/human_replay_quality_2026-10-25/"
#: Text a later rule must not be able to match inside an earlier rule's replacement.
_TOKEN_TEXT = b"redacted-account-0123456789"


def _boundary(spelling: bytes) -> bytes:
    return b"(?<![A-Za-z0-9])" + re.escape(spelling) + b"(?![A-Za-z0-9])"


@dataclass(frozen=True)
class Rules:
    literals: tuple[tuple[bytes, int], ...]  # global, (spelling, identity idx), longest first
    regexes: tuple[tuple[bytes, int], ...]  # global, boundary-checked
    scoped: tuple[tuple[bytes, int], ...]  # boundary-checked, applied only to ``scope_paths``
    scope_paths: tuple[str, ...]
    waived: dict[int, tuple[str, ...]]  # scoped identity -> the paths where its standing-alone hits are LEFT

    def render(self, identities: list[Identity]) -> bytes:
        """The ``--replace-text`` / ``--replace-message`` file: ``literal:`` lines, then ``regex:`` lines (global only)."""
        token = {i.idx: i.replacement for i in identities}
        lines = [b"literal:" + sp + b"==>" + token[idx] for sp, idx in self.literals]
        lines += [b"regex:" + _boundary(sp) + b"==>" + token[idx] for sp, idx in self.regexes]
        return b"\n".join(lines) + b"\n"

    def render_callback(self, identities: list[Identity]) -> str:
        """The body of ``--file-info-callback``: the global rules everywhere, the scoped ones on ``scope_paths`` only."""
        token = {i.idx: i.replacement for i in identities}
        scoped = [(_boundary(sp), token[idx]) for sp, idx in self.scoped]
        paths = [p.encode() for p in self.scope_paths]
        return (
            "if mode in (b'120000', b'160000'):\n"
            "    return (filename, mode, blob_id)\n"
            "contents = value.get_contents_by_identifier(blob_id)\n"
            "if contents is None or value.is_binary(contents):\n"
            "    return (filename, mode, blob_id)\n"
            "new = value.apply_replace_text(contents)\n"
            f"if filename in {paths!r}:\n"
            f"    for rx, repl in {scoped!r}:\n"
            "        new = re.sub(rx, repl, new)\n"
            "if new != contents:\n"
            "    blob_id = value.insert_file_with_contents(new)\n"
            "return (filename, mode, blob_id)\n")

    @property
    def counts(self) -> tuple[int, int, int]:
        return len(self.literals), len(self.regexes), len(self.scoped)


def build_rules(audit: Audit, identities: list[Identity], scope_paths: frozenset[str] | set[str]) -> Rules:
    """Rules for every spelling the audit saw, plus each derived form (so a form absent from history is still ruled)."""
    literal: dict[bytes, int] = {}
    regex: dict[bytes, int] = {}
    scoped: dict[bytes, int] = {}
    waived: dict[int, tuple[str, ...]] = {}
    for ident in identities:
        rec = audit.per_identity.get(ident.idx)
        spellings = {f.encode() for f in ident.forms} | {ident.tid.encode()}
        contaminated: set[bytes] = set()
        outside: set[str] = set()
        if rec is not None:
            spellings |= set(rec.spellings) | set(rec.contaminated)
            contaminated = set(rec.contaminated)
            outside = {p for p in rec.paths if p not in scope_paths}
        is_scoped = rec is not None and (rec.pre_existing_blobs > 0 or bool(outside))
        for sp in spellings:
            if sp.lower() in _TOKEN_TEXT:
                raise ValueError(f"identity #{ident.idx:02d}: a spelling occurs inside the replacement token: a later rule would re-match it")
            for bad in (b"==>", b"\n", b"\r"):
                if bad in sp:
                    raise ValueError(f"identity #{ident.idx:02d}: a spelling holds a character the rules file cannot carry")
            if is_scoped:
                scoped[sp] = ident.idx
            elif sp in contaminated:
                regex[sp] = ident.idx
            else:
                literal[sp] = ident.idx
        if is_scoped:
            waived[ident.idx] = tuple(sorted(outside)) or ("(only blobs that predate the introducing commit)",)
    for sp in set(literal) & set(regex):  # a spelling clean somewhere and contaminated elsewhere is contaminated
        del literal[sp]
    owners: dict[bytes, int] = {}
    for sp, idx in [*literal.items(), *regex.items(), *scoped.items()]:
        if owners.setdefault(sp.lower(), idx) != idx:
            raise ValueError("two identities share one spelling: the derivation is wrong")
    order = lambda items: tuple(sorted(items, key=lambda kv: (-len(kv[0]), kv[0])))  # noqa: E731
    return Rules(order(literal.items()), order(regex.items()), order(scoped.items()), tuple(sorted(scope_paths)), waived)
