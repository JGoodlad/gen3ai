"""Do the ``<doc> § <section>`` pointers in our docs and code comments still land on a section?

A pointer such as ``designs/training/forks.md §14.3`` or ``testing.md "One torch thread"`` is believed by whoever
follows it. The freshness gate checks that a *path* exists; this checks the *section* it names (audit 2026-10-19
F3 / §5.4: eight pointers had dangled unnoticed). Run ``python -m utils.section_refs`` to list what dangles.

**What is a pointer.** A ``.md``/``.rs``/``.py``/``.js``/``.html``/``.toml``/``.sh`` file name (backticked or not)
followed by ``§`` / ``section`` and a number (``§4``, ``§14.3``, ``§6b``, ``§3–§5``), a letter (``§e``) or a title;
or by a quoted title (``file`` → "Title", ``file`` "Title"); or a quoted title followed by ``in`` / ``of`` /
``from`` and the file ("Title" in ``file``). A hard-wrapped comment or paragraph is joined first.

**What resolves it.** The file name resolves by repo-relative path, relative to the referring file, or by bare
name (a bare name with several matches passes when ANY of them has the section; never a guess). In a Markdown
target a number or letter must be the leading label of a heading, ``(c)`` heading, list item, bold lead or table
row; a title must be inside a heading, else (any length) anywhere in the text. In a source-file target only
comment labels (``// L. ...``) and quoted phrases are checked; a number there names the *referring* doc's section.

**What is not checked.** History as a source (ledgers, CHANGELOG, ``ai_vN``, ``deleted_flags.md``, measurement
dirs); ``§`` inside a string literal of code that handles pointers; a pointer with no file name beside it
(``the launcher leaf § "X"``); an unquoted title that starts with a stop word. ``scan`` counts these.
"""
from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple

from utils.paths import repo_root

# Files whose own pointers are history (dated rows, frozen digraphs), not promises about the tree.
_HISTORY_SOURCE_PREFIXES = (
    "designs/research_state/", "designs/ai_v", "designs/CHANGELOG.md", "designs/deleted_flags.md",
)
# The checker's own pointer FIXTURES (the gate's meta-tests) are examples, not claims about the tree.
_SELF_SOURCES = ("src/claude_md_freshness_gate_test.py", "src/utils/section_refs.py")
_SOURCE_SUFFIXES = (".md", ".py", ".rs", ".js", ".html", ".toml", ".sh")
_TARGET_SUFFIXES = (".md", ".rs", ".py", ".js", ".html", ".toml", ".sh")

_FILE = r"[\w./\-]+\.(?:md|rs|py|js|html|toml|sh)(?![\w])"
# `<file>` § 4 / <file>, § 4 / <file> (§4) / `<file>` section "x"
_REF_RE = re.compile(
    r"(?P<file>" + _FILE + r")`?\)?,?\s*(?:\(|—|-)?\s*(?:§|section\s)\s*"
    r"(?P<what>[\"“`*][^\"”`*\n]{2,80}[\"”`*]|[A-Za-z]?\d[\w.]*(?:\s*[–\-/]\s*§?\s*[A-Za-z]?\d[\w.]*)*|[A-Za-z][\w'\-]*(?:\s[A-Z][\w'\-]*)*)(?!\.\d)"
)
# `<file>` "Title" / `<file>` → "Title" / `<file>` → *Title*  (no §)
_FILE_TITLE_RE = re.compile(
    r"(?P<file>" + _FILE + r")`?\)?[`\s,;:()—\-→>]{0,8}(?:section\s+)?(?:\"(?P<q1>[^\"\n]{3,100})\"|“(?P<q2>[^”\n]{3,100})”|→\s*\*(?P<q3>[^*\n]{3,100})\*)"
)
# "Title" in `<file>`  /  the "Title" section of `<file>`
_TITLE_FILE_RE = re.compile(
    r"(?:\"(?P<q1>[^\"\n]{3,100})\"|“(?P<q2>[^”\n]{3,100})”)`?,?\s+(?:section\s+|chapter\s+)?(?:in|of|from|at)\s+(?:the\s+)?`?(?P<file>" + _FILE + r")"
)
_NUM_RE = re.compile(r"[A-Za-z]?\d[\w.]*")
_STOPWORDS = {"the", "a", "an", "in", "of", "on", "to", "for", "and", "or", "is", "are", "as", "at", "by", "it"}

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_LABEL_RE = re.compile(r"^\s*(?:[-*|>]\s*)*(?:\*\*|`)?\s*§?\s*([A-Za-z]?\d[\w.]*)[.):\s*`|—–-]")


class Ref(NamedTuple):
    source: str
    line: int
    target: str
    what: str


class Anchors(NamedTuple):
    labels: Set[str]
    titles: List[str]
    body: str = ""   # the whole text, normalised: a quoted phrase that is not a heading may still be findable


def tracked_files(root: Optional[Path] = None) -> List[str]:
    root = root or repo_root()
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=root, capture_output=True, text=True, check=True).stdout
    return sorted(p for p in out.split("\n") if p and (root / p).is_file())


def _norm(text: str) -> str:
    text = re.sub(r"[`*_\"“”]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _label_variants(label: str) -> Set[str]:
    label = label.rstrip(".")
    return {label.lower()}


_CODE_LABEL_RE = re.compile(r"^\s*(?://[/!]?|#+|\*)\s*([A-Za-z]\d?|\d+(?:\.\d+)*[a-z]?)[.):]\s")


def parse_anchors(text: str, code: bool = False) -> Anchors:
    """Every section label (``14.3``, ``6b``, ``A4``) and heading title a Markdown text offers.

    ``code=True`` reads a source file instead: its labels are comment lines such as ``// L. NON-WRAP byte forms``,
    and it has no headings (a quoted title is then found only as a phrase in the text).
    """
    labels: Set[str] = set()
    titles: List[str] = []
    if code:
        for line in text.split("\n"):
            m = _CODE_LABEL_RE.match(line)
            if m:
                labels.add(m.group(1).lower())
        return Anchors(labels, titles, _norm(text))
    in_fence = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        h = _HEADING_RE.match(line)
        if h:
            title = h.group(1)
            titles.append(_norm(title))
            m = re.match(r"(?:§\s*)?([A-Za-z]?\d[\w.]*)(?:[.):\s—–-]|$)", title.strip("*` "))
            if m:
                labels |= _label_variants(m.group(1))
            m = re.match(r"([A-Za-z])[.):](?:\s|$)", title.strip("*` "))   # "C. Plateau"
            if m:
                labels.add(m.group(1).lower())
            m = re.match(r"\(([A-Za-z0-9]{1,3})\)", title.strip("*` "))   # "(c) The byte-comparison TEST strategy"
            if m:
                labels.add(m.group(1).lower())
            # "Step 3" / "Part A4" / "Section 5" style headings
            m = re.match(r"(?:step|part|section|stage|phase|tier|unit|lane|milestone|appendix)\s+([A-Za-z]?\d[\w.]*)(?:[.):\s—–-]|$)", title.strip("*` "), re.I)
            if m:
                labels |= _label_variants(m.group(1))
            continue
        m = _LABEL_RE.match(line)
        if m:
            labels |= _label_variants(m.group(1))
        # an inline bold/table label opening a paragraph: "**§4.2 Foo.**"
    return Anchors(labels, titles, _norm(text))


_COMMENT_PREFIX_RE = re.compile(r"^\s*(?://[/!]?|#+(?=\s)|\*(?!\*)|>)\s?")


_OF_OTHER_FILE_RE = re.compile(r"[`\s,]*(?:of|in)\s+(?:the\s+)?`*" + _FILE)
_HAS_FILE_RE = re.compile(r"\.(?:md|rs|py|js|html|toml|sh)\b")
_CODE_LINE_RE = re.compile(
    r"^(?:def |class |return\b|assert\b|import |from \S+ import|if |elif |else:|for |while |with |try:|except\b|@|"
    r"let |const |var |fn |pub |use |mod |function |\w[\w.\[\]]*\s*(?:[-+*/|&]?=[^=]|\()|[)\]}]|\"\"\"|\'\'\'|[\"\'][^\"\']*[\"\'][,)])"
)


def logical_lines(text: str, suffix: str) -> List[Tuple[str, List[Tuple[int, int]]]]:
    """``(joined text, [(char offset, physical line)])`` with a hard-wrapped sentence/comment joined back up.

    A pointer's title often wraps across a ``//!`` / ``#`` / docstring line break; joining makes it matchable.
    A blank line, a heading, a list item, a table row, a fence line or a line of code ends a paragraph.
    """
    out: List[Tuple[str, List[Tuple[int, int]]]] = []
    cur: Optional[List] = None   # [text, marks, was_code]
    is_md = suffix == ".md"
    for i, raw in enumerate(text.split("\n"), 1):
        line = raw if is_md else _COMMENT_PREFIX_RE.sub("", raw, count=1)
        stripped = line.strip()
        code_like = (not is_md) and bool(_CODE_LINE_RE.match(stripped)) and not _COMMENT_PREFIX_RE.match(raw)
        starts_block = (not stripped or code_like or stripped.startswith(("#", "|", "```", "- ", "* ", "+ "))
                        or re.match(r"\d+[.)]\s", stripped) is not None)
        if cur is not None and not starts_block and not cur[2]:
            cur[1].append((len(cur[0]) + 1, i))
            cur[0] += " " + stripped
            continue
        if cur is not None:
            out.append((cur[0], cur[1]))
            cur = None
        if stripped:
            cur = [stripped, [(0, i)], code_like]
    if cur is not None:
        out.append((cur[0], cur[1]))
    return out


def _line_at(marks: List[Tuple[int, int]], offset: int) -> int:
    line = marks[0][1]
    for off, ln in marks:
        if off <= offset:
            line = ln
    return line


def find_refs(source: str, text: str) -> List[Ref]:
    refs: List[Ref] = []
    seen = set()
    for line, marks in logical_lines(text, PurePosixPath(source).suffix):
        if not _HAS_FILE_RE.search(line):
            continue
        found = []
        for m in _REF_RE.finditer(line):
            if _OF_OTHER_FILE_RE.match(line, m.end()):
                continue   # the section belongs to the file named after it ("§2.1 of <file>")
            found.append((m.start(), m.group("file"), m.group("what").strip()))
        for m in _FILE_TITLE_RE.finditer(line):
            if _OF_OTHER_FILE_RE.match(line, m.end()):
                continue   # the quoted title belongs to the file named after it ("... in <file>")
            title = m.group("q1") or m.group("q2") or m.group("q3")
            found.append((m.start(), m.group("file"), '"%s"' % title))
        for m in _TITLE_FILE_RE.finditer(line):
            title = m.group("q1") or m.group("q2")
            found.append((m.start(), m.group("file"), '"%s"' % title))
        for off, target, what in found:
            lineno = _line_at(marks, off)
            if (lineno, target, what) not in seen:
                seen.add((lineno, target, what))
                refs.append(Ref(source, lineno, target, what))
    return refs


def resolve_target(ref: Ref, by_name: Dict[str, List[str]], all_files: Set[str]) -> List[str]:
    """Candidate repo paths for a reference's file token (empty = no such file)."""
    tok = ref.target.lstrip("./")
    src_dir = str(PurePosixPath(ref.source).parent)
    for cand in (tok, str(PurePosixPath(src_dir) / tok)):
        norm = str(PurePosixPath(cand))
        if norm in all_files:
            return [norm]
    if "/" in tok:
        suffix = [f for f in by_name.get(PurePosixPath(tok).name, []) if f.endswith("/" + tok) or f == tok]
        return suffix
    cands = by_name.get(tok, [])
    local = [c for c in cands if str(PurePosixPath(c).parent) == src_dir]
    return local or cands


def _what_labels(what: str) -> List[str]:
    if re.fullmatch(r"[A-Za-z]\.?", what.strip()):
        return [what.strip().rstrip(".").lower()]
    return [m.lower().rstrip(".") for m in _NUM_RE.findall(what)][:1] + \
           [m.lower().rstrip(".") for m in re.findall(r"§\s*([A-Za-z]?\d[\w.]*)", what)]


_CODE_JUNK_RE = re.compile(r"^[\"“`*]\s*[)\]}:;,.=$|&<>/\\]|[\[\]{}]|\),|\"\)|^\"\s|\s\"$|^\"\$")


def section_exists(what: str, anchors: Anchors) -> Optional[bool]:
    """True/False, or None when the reference form cannot be judged."""
    w = what.strip()
    if _CODE_JUNK_RE.search(w):
        return None   # a ``§`` inside a string literal of CODE that handles pointers, not a pointer
    if w[0] in "\"“`*":
        title = _norm(w).lstrip("# ").rstrip(".,:;")
        title = re.split(r"…|\.\.\.", title)[0].strip() or title
        if any(title in t for t in anchors.titles):
            return True
        return len(title) >= 4 and title in anchors.body
    if re.match(r"[A-Za-z]?\d", w) or re.fullmatch(r"[A-Za-z]\.?", w):
        return all(lab in anchors.labels for lab in _what_labels(w))
    first = w.split()[0].lower()
    if first in _STOPWORDS:
        return None
    return any(t.startswith(first) or t.lstrip("§ ").startswith(first) for t in anchors.titles)


def scan(root: Optional[Path] = None) -> Tuple[List[Tuple[Ref, str]], Dict[str, int]]:
    """(dangling references with a reason, counts of checked / unchecked kinds) over this checkout."""
    root = root or repo_root()
    return scan_files(tracked_files(root), lambda rel: (root / rel).read_text(errors="replace"))


def scan_files(files: List[str], read: Callable[[str], str]) -> Tuple[List[Tuple[Ref, str]], Dict[str, int]]:
    """The scan over ``files`` (repo-relative paths) read through ``read`` -- the seam the meta-tests use."""
    all_files = set(files)
    by_name: Dict[str, List[str]] = defaultdict(list)
    for f in files:
        if f.endswith(_TARGET_SUFFIXES):
            by_name[PurePosixPath(f).name].append(f)
    anchor_cache: Dict[str, Anchors] = {}

    def anchors_of(rel: str) -> Anchors:
        if rel not in anchor_cache:
            anchor_cache[rel] = parse_anchors(read(rel), code=not rel.endswith(".md"))
        return anchor_cache[rel]

    bad: List[Tuple[Ref, str]] = []
    counts = {"checked": 0, "unchecked_non_md": 0, "unchecked_form": 0, "unresolved_file": 0}
    for rel in files:
        if not rel.endswith(_SOURCE_SUFFIXES) or rel.startswith(_HISTORY_SOURCE_PREFIXES) or rel in _SELF_SOURCES:
            continue
        text = read(rel)
        for ref in find_refs(rel, text):
            if _CODE_JUNK_RE.search(ref.what):
                counts["unchecked_form"] += 1   # a ``§`` inside a string literal of CODE that handles pointers
                continue
            cands = resolve_target(ref, by_name, all_files)
            if not cands:
                counts["unresolved_file"] += 1
                bad.append((ref, "no such file"))
                continue
            md = [c for c in cands if c.endswith(".md")]
            if not md:
                md = cands   # a source file: comment labels ("// L. ...") and quoted phrases only
                if not re.match(r"[\"“`*]|[A-Za-z]\.?$", ref.what):   # a number there names the REFERRING doc's section
                    counts["unchecked_non_md"] += 1
                    continue
            verdicts = [section_exists(ref.what, anchors_of(c)) for c in md]
            if all(v is None for v in verdicts):
                counts["unchecked_form"] += 1
                continue
            counts["checked"] += 1
            if not any(verdicts):
                bad.append((ref, "no such section in " + ", ".join(md)))
    return bad, counts


def main() -> int:
    bad, counts = scan()
    print(counts)
    for ref, why in bad:
        print(f"{ref.source}:{ref.line}: {ref.target} § {ref.what}  -- {why}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
