"""LEDGER INDEX CURRENCY GATE — `designs/research_state/ledger_index.md` matches `ledger.md`.

🚨 **A STALE INDEX IS WORSE THAN NO INDEX**, and that is why this is a hard gate rather than a
warning. The index exists so a registration can be FOUND (`ugrep` hit complexity errors on the
13.8k-line ledger twice on 2026-09-06); an index missing the last three entries answers "not
registered" for something that is, which is the absence-is-not-a-zero class the ledger itself keeps
recording. The failure is one command away from fixed and the regeneration is deterministic, so the
gate costs a landing seconds and buys the index's only useful property.

WHY NOT A GENERATION STEP IN `scripts/land.sh` INSTEAD. Considered and rejected on mechanics, not
taste: `land.sh` gates a worktree and then PUSHES it — it never commits, by design. A generation
step there would write a file into the worktree *after* the commit it belongs to, leaving the change
uncommitted and unlanded while the gate went green. To make it work `land.sh` would have to start
committing on the author's behalf in the step that lands code, which is a much larger change to a
script whose whole virtue is that it does four legible things.

WHAT THIS COSTS A CONCURRENT LANDING. Two agents that both append a ledger entry will both
regenerate, and the index will conflict on rebase. **The resolution is never a manual merge: take
either side and re-run the generator** — the index is a pure function of the ledger, so the rebased
ledger determines it completely. That is written into `designs/research_state/README.md` beside the
append-only rule.

Unmarked, so it runs in every tier including the routine gate, and it is picked up by `land.sh`'s
`src/*_gate_test.py` step. Opt out with `GEN3AI_SKIP_LEDGER_INDEX_GATE=1`.
"""
import os

import pytest

from main.ledger_index import REGEN_COMMAND, parse_headings, render_index
from utils.paths import repo_path

pytestmark = pytest.mark.skipif(
    os.environ.get("GEN3AI_SKIP_LEDGER_INDEX_GATE") == "1",
    reason="GEN3AI_SKIP_LEDGER_INDEX_GATE=1")

_LEDGER = repo_path("designs", "research_state", "ledger.md")
_INDEX = repo_path("designs", "research_state", "ledger_index.md")


def test_the_committed_index_is_current():
    assert _INDEX.exists(), (
        f"designs/research_state/ledger_index.md is MISSING — regenerate with:  {REGEN_COMMAND}")
    want = render_index(_LEDGER.read_text(encoding="utf-8"))
    got = _INDEX.read_text(encoding="utf-8")
    if got == want:
        return
    want_lines, got_lines = want.split("\n"), got.split("\n")
    missing = [ln for ln in want_lines if ln.startswith("- `L") or ln.lstrip().startswith("- `L")]
    have = set(got_lines)
    first = next((ln for ln in missing if ln not in have), "(the header block)")
    pytest.fail(
        "designs/research_state/ledger_index.md is STALE — the ledger has entries the index does "
        f"not (or the generator changed).\n  first line not in the committed index: {first[:160]}\n"
        f"  index {len(got_lines)} lines, regenerated {len(want_lines)} lines\n"
        f"  REGENERATE WITH:  {REGEN_COMMAND}\n"
        "  (on a rebase conflict in this file take EITHER side and re-run the generator — the "
        "index is a pure function of the ledger)")


def test_a_planted_ledger_entry_makes_the_gate_fire():
    """The gate is proved to FIRE, not just to pass — a currency check that cannot fail is a
    green light with no bulb in it."""
    ledger = _LEDGER.read_text(encoding="utf-8")
    committed = _INDEX.read_text(encoding="utf-8")
    planted = ledger + "\n### 2099-01-01 · PLANTED · this entry is not in the committed index\n"
    assert render_index(planted) != committed
    assert "PLANTED" in render_index(planted)


def test_both_heading_eras_parse_as_dated_entries():
    """The ledger spans two heading conventions and the index must not silently lose either.
    Early era: the date parenthesised mid-title. Current era: the date leading, `·`-delimited."""
    doc = "\n".join([
        "# Ledger",
        "## Status",
        "## Gen-3 40M gate (2026-08-08) — the concat re-read",
        "### ⚠️ CORRECTION (2026-08-11, same day) to the ladder entry above",
        "#### RESOLUTION (2026-08-14) — the battle test",
        "### 2026-09-07 · TECH DEBT · mode-flag doc gate",
        "### 2026-08-10 — STEP 0 of the pair-reduction plan",
        "### What fired",
        "",
    ])
    hs = parse_headings(doc)
    assert [h.level for h in hs] == [2, 2, 3, 4, 3, 3, 3]
    assert [h.date for h in hs] == [
        "", "2026-08-08", "2026-08-11", "2026-08-14", "2026-09-07", "2026-08-10", ""]
    # A LEADING date and its separator are stripped (the index prints a date column); a
    # parenthesised mid-title date is left exactly where the author put it.
    assert hs[4].title == "TECH DEBT · mode-flag doc gate"
    assert hs[5].title == "STEP 0 of the pair-reduction plan"
    assert hs[1].title == "Gen-3 40M gate (2026-08-08) — the concat re-read"
    assert hs[6].title == "What fired"


def test_headings_inside_a_fenced_block_are_not_entries():
    """No ledger heading is inside a fence today; an entry pasting a shell heredoc would put one
    there, and a `# comment` must not become an index line."""
    doc = "\n".join(["## 2026-09-07 · real", "```bash", "## 2026-09-07 · not a heading", "```",
                     "### 2026-09-08 · also real", ""])
    assert [h.title for h in parse_headings(doc)] == ["real", "also real"]


def test_the_generator_never_writes_the_ledger():
    """`render_index` is pure: the ledger is append-only and nothing here may touch it."""
    before = _LEDGER.read_bytes()
    render_index(before.decode("utf-8"))
    assert _LEDGER.read_bytes() == before
