"""Does any file but the owner SPELL a trainer or agent CPU list? The static gate of ``utils.core_split``.

**The bug class.** The box's CPUs are split between the trainer and agent work (root ``CLAUDE.md`` rule 5b), and
the split can change (``designs/ops/core_split.md``). Before the owner, a dozen files each spelled a list. One of them
left on the old list after a switch puts agent work on the trainer's cores, or the trainer on the agents', and nothing
raises: a run just gets slower. So the lists live in ONE module, and everything else reads it::

    from utils import core_split;  core_split.agent_cpus()              # python, read when the job is launched
    taskset -c "$(python -m utils.core_split agent-cpus)" ...           # a shell script, a doc's command block

**The rule.** No tracked text file contains the trainer or the agent list of ANY split the owner accepts
(``core_split.SPLITS``) in ``taskset -c``'s form, and no CODE file (:data:`CODE_SUFFIXES`) contains one as a
comma-separated run of the CPU ids. A doc that records how a past measurement ran names the split instead ("the 5+3
split's agent cores").

**The declared exceptions** (classes, never single favours; there is no allowlist):

* THE OWNER: the module, its test, this gate, and its design doc (which tabulates the lists).
* FROZEN RECORDS: everything under ``designs/research_state/`` (measurement records and their scripts, the
  append-only ledger, the archive). A record says what ran; it is never a tool. :data:`LIVE_KITS` are the files
  under that tree that ARE still tools: they are scanned.
* APPEND-ONLY AND DATED RECORDS: ``designs/CHANGELOG.md``, and a file whose name carries its date
  (``…_2026-10-19.md``): a review or audit of that day.

Opt out explicitly (never silently): ``GEN3AI_SKIP_CORE_SPLIT_GATE=1``.

**Blind spots, stated:** a list assembled at run time (``f"{a}-{b}"``, ``range(5, 8)``), a list in an untracked
file or a session script outside the repo, and a kit under the frozen tree that is run again without being named in
:data:`LIVE_KITS`.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Pattern, Tuple

import pytest

from utils import core_split
from utils.paths import repo_path

pytestmark = pytest.mark.static

#: THE OWNER (repo-relative).
OWNER = ("src/utils/core_split.py", "src/utils/core_split_test.py", "src/core_split_literal_gate_test.py",
         "designs/ops/core_split.md")
#: FROZEN RECORDS: this tree keeps what it recorded.
FROZEN_TREE = "designs/research_state/"
#: Files under the frozen tree that are still tools (a launch kit in use): scanned like any other code.
LIVE_KITS = ("designs/research_state/measurements/x35_breakers_stage_2026-10-17/launch_x35.sh",
             "designs/research_state/measurements/boundary1_scratch_2026-11-09/looks/drive_look.sh")
APPEND_ONLY = ("designs/CHANGELOG.md",)
_DATED = re.compile(r"_20\d\d-\d\d-\d\d(?:[._][^/]*)?$")
#: Text files: by suffix, plus extensionless scripts. A binary or oversized file is not read.
TEXT_SUFFIXES = frozenset({".py", ".sh", ".bash", ".md", ".txt", ".json", ".jsonl", ".toml", ".yaml", ".yml", ".cfg", ".ini",
                           ".js", ".mjs", ".ts", ".rs", ".service", ".timer", ".conf", ".env", ""})
MAX_BYTES = 4_000_000
#: Where a list can also hide as a sequence of ints (a tuple, a list literal).
CODE_SUFFIXES = frozenset({".py", ".sh", ".bash", ".js", ".mjs", ".ts", ".rs", ".toml", ""})


def _sides() -> List[Tuple[str, str, Tuple[int, ...]]]:
    return [(f"the {n} split's {side}", text, ids) for n in core_split.SPLITS for s in (core_split.parse_split(n),)
            for side, text, ids in (("trainer", s.trainer_list, s.trainer_ids), ("agent", s.agent_list, s.agent_ids))]


def patterns(code: bool = True) -> List[Tuple[str, Pattern[str]]]:
    """``(what, regex)`` for both lists of every split the owner accepts; ``code``: the CPU-id spelling too."""
    out: List[Tuple[str, Pattern[str]]] = []
    for what, text, ids in _sides():
        out.append((f"{what} list", re.compile(rf"(?<![\d-]){re.escape(text)}(?![\d-])")))
        if code:
            seq = r"\s*,\s*".join(str(c) for c in ids)
            out.append((f"{what} CPU ids", re.compile(rf"(?<![\d.])\b{seq}\b(?!\s*,\s*\d)")))
    return out


_PATTERNS = {True: patterns(True), False: patterns(False)}
_LIST_TOKENS = tuple(text for _, text, _ in _sides())
_ID_TOKENS = tuple(",".join(map(str, ids)) for _, _, ids in _sides())


def excepted(rel: str) -> bool:
    if rel in LIVE_KITS:
        return False
    return (rel in OWNER or rel in APPEND_ONLY or rel.startswith(FROZEN_TREE) or bool(_DATED.search(rel.rsplit("/", 1)[-1])))


def scan_text(text: str, code: bool = True) -> List[Tuple[int, str]]:
    """``(line number, what)`` for every spelled list in ``text``; ``code``: the CPU-id spelling too."""
    # the cheap pass (substring searches; 130 MB of tracked text holds no list): the regexes run on candidates only
    if not (any(t in text for t in _LIST_TOKENS) or (code and any(t in "".join(text.split()) for t in _ID_TOKENS))):
        return []
    hits = {(text.count("\n", 0, m.start()) + 1, what) for what, rx in _PATTERNS[code] for m in rx.finditer(text)}
    return sorted(hits)


def tracked_files(root: Path) -> List[str]:
    r = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                       capture_output=True, timeout=120, check=True)
    return sorted(p for p in r.stdout.decode("utf-8", "surrogateescape").split("\0") if p)


def scan(root: Path, files: Iterable[str]) -> List[str]:
    out: List[str] = []
    for rel in files:
        suffix = Path(rel).suffix.lower()
        if excepted(rel) or suffix not in TEXT_SUFFIXES:
            continue
        p = root / rel
        try:
            if not p.is_file() or p.is_symlink() or p.stat().st_size > MAX_BYTES:
                continue
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        out.extend(f"{rel}:{n}: {what}" for n, what in scan_text(text, code=suffix in CODE_SUFFIXES))
    return out


def test_no_file_but_the_owner_spells_a_core_list() -> None:
    if os.environ.get("GEN3AI_SKIP_CORE_SPLIT_GATE"):
        pytest.skip("GEN3AI_SKIP_CORE_SPLIT_GATE is set")
    root = repo_path()
    hits = scan(root, tracked_files(root))
    assert not hits, (
        "a trainer / agent CPU list is spelled outside its ONE owner (utils.core_split):\n  " + "\n  ".join(hits[:40])
        + ("\n  …" if len(hits) > 40 else "")
        + "\nRead it instead: `core_split.agent_cpus()` / `core_split.trainer_cpus()` in python (when the job is "
          "launched, not at import); `\"$(python -m utils.core_split agent-cpus)\"` in a shell script or a doc's command; "
          "\"the 5+3 split's agent cores\" in prose about a past measurement. There is no allowlist.")


def test_the_scanner_has_teeth() -> None:
    """Every spelling of every split is caught; look-alikes are not."""
    for name in core_split.SPLITS:
        s = core_split.parse_split(name)
        for text, ids in ((s.trainer_list, s.trainer_ids), (s.agent_list, s.agent_ids)):
            assert scan_text(f"x\nnice -n 19 taskset -c {text} cmd\n") == [(2, scan_text(f"taskset -c {text}")[0][1])]
            assert scan_text(f'CORES = "{text}"') and scan_text(f"`{text}`")
            assert scan_text(f"CPUS = ({', '.join(map(str, ids))})") and scan_text(f"[{','.join(map(str, ids))}]")
    d = core_split.DEFAULT
    assert not scan_text(f"2026-1{d.agent_list}")                 # glued to a longer number or range: not the list
    assert not scan_text(f"{d.agent_list}-17") and not scan_text(f"{', '.join(map(str, d.agent_ids))}, 16")
    assert not scan_text("taskset -c 0-15") and not scan_text("cores 5-7") and not scan_text("list(range(16))")
    assert not scan_text('taskset -c "$(python -m utils.core_split agent-cpus)"')
    ids = ", ".join(map(str, d.agent_ids))
    assert scan_text(f"cores {d.agent_list}", code=False) and not scan_text(f"a table row: {ids}", code=False)


def test_the_exceptions_are_the_declared_classes_and_a_live_kit_is_scanned(tmp_path: Path) -> None:
    assert all(excepted(f) for f in OWNER + APPEND_ONLY)
    assert excepted("designs/research_state/measurements/x_2026-10-01/read.py") and excepted("designs/research_state/ledger.md")
    assert excepted("designs/ops/instruction_docs_audit_2026-10-19.md") and excepted("designs/endstate/reviews/x63_review_2026-10-27.md")
    for scanned in (*LIVE_KITS, "CLAUDE.md", "src/main/x35/kakuna.py", "designs/ops/testing.md", "scripts/ops/x.sh",
                    "designs/endstate/design_2026_plan.md", "src/utils/core_split_ab_2026.py"):
        assert not excepted(scanned), scanned
    for kit in LIVE_KITS:
        assert repo_path(*kit.split("/")).is_file(), f"{kit} is gone: drop it from LIVE_KITS"
    # the scan itself, on a made-up tree: the frozen record passes, the live kit and the code fail
    lst = core_split.DEFAULT.agent_list
    files = {"src/a.py": f'X = "{lst}"\n', "designs/research_state/measurements/m/run.sh": f"taskset -c {lst} x\n",
             LIVE_KITS[0]: f"taskset -c {lst} x\n", "docs/x.md": "the agent cores\n", "bin.png": lst,
             "data/t.json": "[" + ", ".join(map(str, core_split.DEFAULT.agent_ids)) + "]"}
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    assert [h.split(":")[0] for h in scan(tmp_path, sorted(files))] == sorted([LIVE_KITS[0], "src/a.py"])
