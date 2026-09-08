"""Generate the ledger's TITLE INDEX — `designs/research_state/ledger_index.md`.

🚨 **THIS NEVER TOUCHES `ledger.md`.** The ledger is append-only and is the record; this writes a
derived sibling that is regenerated from scratch every time, so the two can never disagree in a way
that needs adjudicating — if they differ, the index is simply out of date and one command fixes it.

WHY IT EXISTS. The ledger is one file of ~13,700 lines and grows with every landing. `ugrep` hit
complexity errors on it twice on 2026-09-06, and a registration that cannot be found is a
registration that gets re-made. The index is ~430 lines: every heading, its line number, and its
date when it carries one.

WHY IT LIVES IN `src/main/` RATHER THAN `tools/`. `tools/` is the ACQUISITION layer — the only
place that knows the three upstreams, deriving files into `data/` (`tools/CLAUDE.md`). This knows
no upstream and writes nothing under `data/`; it is an offline CLI over the repo's own documents,
which is exactly the `src/main/*.py` family (`main.lineage`, `main.sidecar_audit`, `main.dose`, …).
`tools/promote_teams` was moved to `src/main/` for the same reason and its leaf says so. The
second reason is mechanical: `src/ledger_index_gate_test.py` has to IMPORT the generator to compare
against the committed file, and `tools/` is not on `sys.path`.

THE HEADING CONVENTION, verified against the whole file (2026-09-07: 426 headings, 389 dated).
The ledger's own README says entries are cited by their landing sha "since the headings are prose",
so there is no id to key on — the only stable structure is the heading tree and the dates in it:

* An **ENTRY** heading carries an ISO date somewhere in its text. Both eras are covered by that one
  rule, and they differ: the early era writes `## Gen-3 40M gate (2026-08-08) — …` and
  `### ⚠️ CORRECTION (2026-08-11, same day) …` (the date parenthesised, mid-title), the current era
  writes `### 2026-09-07 · TECH DEBT · …` (the date leading, `·`-delimited). A leading date plus its
  separator is stripped from the title, because the index prints it in its own column; a
  parenthesised mid-title date is left exactly where the author put it.
* A heading with **no** date is either the file's front matter (the seven `##` sections before the
  first dated heading) or a SUBSECTION inside an entry (`### What fired`, `### The divergence`).

Rather than guess which, the index emits EVERY heading, nested by its own level, with a date column
that is blank when the heading has none. Nothing in the ledger can be lost to a convention guess
that turns out wrong — the cost is ~30 extra lines.

Fenced code blocks are skipped. No heading is inside one today; a future entry pasting a shell
heredoc would put one there.

USAGE
    export PYTHONPATH=$PYTHONPATH:src
    python3 -m main.ledger_index               # print the index
    python3 -m main.ledger_index --write       # regenerate designs/research_state/ledger_index.md
    python3 -m main.ledger_index --check       # exit 1 if the committed file is stale
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from utils.paths import repo_path

LEDGER = ("designs", "research_state", "ledger.md")
INDEX = ("designs", "research_state", "ledger_index.md")

REGEN_COMMAND = "export PYTHONPATH=$PYTHONPATH:src && python3 -m main.ledger_index --write"

_HEADING = re.compile(r"^(#{2,6})[ \t]+(\S.*?)[ \t]*$")
_ISO_DATE = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")
# A LEADING date plus the separator the era used: `2026-09-07 · `, `2026-08-10 — `, `2026-08-10 - `,
# `2026-09-07: `. Anything else (a parenthesised mid-title date) is left in place.
_LEADING_DATE = re.compile(r"^(20\d\d-\d\d-\d\d)\s*(?:[·—–\-:]\s*)?")


@dataclass(frozen=True)
class Heading:
    line: int           # 1-based line number in ledger.md
    level: int          # 2 for `##`, 3 for `###`, …
    date: str           # "" when the heading carries none
    title: str          # the heading text, leading date stripped when it led


def parse_headings(text: str) -> list[Heading]:
    """Every `##`..`######` heading outside a fenced code block, in file order."""
    out: list[Heading] = []
    fenced = False
    for n, raw in enumerate(text.split("\n"), start=1):
        if raw.startswith("```") or raw.startswith("~~~"):
            fenced = not fenced
            continue
        if fenced:
            continue
        m = _HEADING.match(raw)
        if not m:
            continue
        body = m.group(2)
        found = _ISO_DATE.search(body)
        date = found.group(1) if found else ""
        title = _LEADING_DATE.sub("", body).strip() or body
        out.append(Heading(line=n, level=len(m.group(1)), date=date, title=title))
    return out


def render_index(text: str) -> str:
    """The full `ledger_index.md`, ending in exactly one newline."""
    headings = parse_headings(text)
    n_lines = len(text.split("\n"))
    dated = [h for h in headings if h.date]
    dates = sorted(h.date for h in dated)
    span = f"{dates[0]} → {dates[-1]}" if dates else "(none)"
    top = min((h.level for h in headings), default=2)

    body = [
        "# Ledger — title index",
        "",
        "<!-- GENERATED FILE — DO NOT EDIT BY HAND.",
        f"     Regenerate with:  {REGEN_COMMAND}",
        "     Generator: src/main/ledger_index.py · currency gate: src/ledger_index_gate_test.py -->",
        "",
        "One line per heading in [`ledger.md`](ledger.md) — its line number, its date when it",
        "carries one, and its title — so a registration can be found without a regex over a",
        f"{n_lines:,}-line file. **The ledger itself is append-only and is never edited by this**;",
        "this file is regenerated from scratch, so a difference between the two means the index is",
        "stale, never that the ledger is wrong.",
        "",
        "**An ENTRY is a heading carrying an ISO date** — that one rule covers both eras (the early",
        "`## Gen-3 40M gate (2026-08-08) — …` and the current `### 2026-09-07 · TECH DEBT · …`).",
        "Undated headings — the file's front matter, and the subsections inside an entry — are",
        "listed too, with no date column and nested by their own heading level, so nothing can be",
        "lost to a convention guess that turns out wrong.",
        "Entries are still CITED by their landing sha (`README.md`) — this indexes, it does not",
        "rename.",
        "",
        f"**{len(headings)} headings · {len(dated)} dated · {span} · ledger {n_lines:,} lines.**",
        "",
    ]
    for h in headings:
        indent = "  " * (h.level - top)
        stamp = f" `{h.date}` ·" if h.date else ""
        body.append(f"{indent}- `L{h.line:05d}` ·{stamp} {h.title}")
    body.append("")
    return "\n".join(body)


def _read_ledger(root: Path | None = None) -> str:
    path = repo_path(*LEDGER) if root is None else root.joinpath(*LEDGER)
    return path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="regenerate designs/research_state/ledger_index.md")
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed index is stale")
    args = ap.parse_args(argv)

    rendered = render_index(_read_ledger())
    index_path = repo_path(*INDEX)

    if args.write:
        index_path.write_text(rendered, encoding="utf-8")
        print(f"wrote {index_path} ({len(rendered.splitlines())} lines)")
        return 0
    if args.check:
        current = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
        if current == rendered:
            print("ledger_index.md is CURRENT")
            return 0
        print(f"ledger_index.md is STALE — regenerate with:  {REGEN_COMMAND}", file=sys.stderr)
        return 1
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
