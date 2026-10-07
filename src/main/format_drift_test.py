"""The format-spec DRIFT check: the committed master snapshot matches the spec, the pinned engine differs by
exactly the declared gaps, and every PLANTED master difference (a new / removed ban, clause, Uber, clause-list
entry, or a changed board-state clause body) FAILS loudly. Offline and deterministic."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agents.gen3_data import format_spec as fs
from main import format_drift as fd


def test_the_committed_master_snapshot_matches_the_spec():
    assert fd.snapshot_commit() == fs.MASTER_SNAPSHOT_COMMIT
    assert fd.compare(fd.read_facts(fd.SNAPSHOT_DIR)) == []
    assert fd.check(fd.SNAPSHOT_DIR) == 0


def test_the_pinned_engine_differs_by_exactly_the_declared_gaps():
    diffs = fd.compare(fd.read_facts(fd.PINNED_ROOT))
    assert {(d.key, d.item) for d in diffs} == set(fs.PINNED_DIFFERENCES)
    assert all(d.kind == "removed" for d in diffs)
    assert fd.pinned_gaps(diffs) == ([], [])
    assert fd.check(fd.PINNED_ROOT, pinned=True) == 0


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    dst = tmp_path / "master"
    shutil.copytree(fd.SNAPSHOT_DIR, dst)
    return dst


def _edit(root: Path, rel: str, old: str, new: str) -> None:
    p = root / rel
    text = p.read_text()
    assert old in text, (rel, old)
    p.write_text(text.replace(old, new, 1))


PLANTS = [
    # a NEW ban on master's gen3ou banlist
    ("new ban", fd.FORMATS_TS, "'Quick Claw', ", "'Quick Claw', 'King\\'s Rock', ",
     ("format_banlist", "new", "King's Rock")),
    # a ban LIFTED on master
    ("removed ban", fd.FORMATS_TS, ", 'Swagger'", "", ("format_banlist", "removed", "Swagger")),
    # a NEW clause on the format
    ("new clause", fd.FORMATS_TS, "'Speed Pass Clause'", "'Speed Pass Clause', 'Sleep Moves Clause'",
     ("format_ruleset", "new", "Sleep Moves Clause")),
    # a clause dropped from gen 3's Standard
    ("standard", fd.GEN3_RULESETS, "'Sleep Clause Mod', 'Switch", "'Switch",
     ("standard", "removed", "Sleep Clause Mod")),
    # Standard AG gains a rule
    ("standard ag", fd.GEN4_RULESETS, "'Endless Battle Clause',\n\t\t],", "'Endless Battle Clause', 'Team Preview',\n\t\t],",
     ("standard_ag", "new", "Team Preview")),
    # a species enters Uber
    ("uber", fd.GEN3_FORMATS_DATA, "salamence: {\n\t\ttier: \"OU\"", "salamence: {\n\t\ttier: \"Uber\"",
     ("ubers", "new", "salamence")),
    # the Evasion Items Clause list changes
    ("evasion items", fd.BASE_RULESETS, "banlist: ['Bright Powder', 'Lax Incense']",
     "banlist: ['Bright Powder', 'Lax Incense', 'Zoom Lens']", ("evasionitemsclause.banlist", "new", "Zoom Lens")),
    # One Boost Passer loses an effect
    ("obp", fd.BASE_RULESETS, "'recycle', ", "", ("oneboostpasserclause.boostingEffects", "removed", "recycle")),
    # Accuracy Trap gains a trapper
    ("trap", fd.BASE_RULESETS, "'spiderweb', ", "'spiderweb', 'ingrain', ",
     ("accuracytrapclause.trapping", "new", "ingrain")),
    # Sleep Clause Mod's mechanics change (counts a Rest sleeper): a TEXT drift
    ("sleep body", fd.BASE_RULESETS, "if (!pokemon.statusState.source?.isAlly(pokemon)) {", "if (true) {",
     ("sleepclausemod", "text", "")),
]


@pytest.mark.parametrize("name,rel,old,new,want", PLANTS, ids=[p[0] for p in PLANTS])
def test_a_planted_master_difference_fails(tree, name, rel, old, new, want):
    _edit(tree, rel, old, new)
    diffs = fd.compare(fd.read_facts(tree))
    assert [(d.key, d.kind, d.item) for d in diffs] == [want]
    assert fd.check(tree) == 1


def test_an_unparseable_tree_fails(tree):
    _edit(tree, fd.FORMATS_TS, 'name: "[Gen 3] OU"', 'name: "[Gen 3] OU (old)"')
    with pytest.raises(fd.FormatParseError):
        fd.read_facts(tree)
    assert fd.check(tree) == 1


def test_a_closed_pin_gap_must_be_deleted(tree):
    """The pinned check fails when a declared gap no longer shows (a pin bump that adds Quick Claw)."""
    pinned_like = tree
    _edit(pinned_like, fd.FORMATS_TS, "'Quick Claw', ", "")
    _edit(pinned_like, fd.BASE_RULESETS, "'recycle', ", "")
    assert fd.check(pinned_like, pinned=True) == 0                   # == the pin: exactly the declared gaps
    _edit(pinned_like, fd.FORMATS_TS, "'Soundproof', ", "'Soundproof', 'Quick Claw', ")
    assert fd.check(pinned_like, pinned=True) == 1                   # Quick Claw's gap closed → stale entry


def test_ladder_drift_scan_runs_the_format_check():
    import inspect

    from main import ladder_drift_scan
    assert "format_drift" in inspect.getsource(ladder_drift_scan.main)
