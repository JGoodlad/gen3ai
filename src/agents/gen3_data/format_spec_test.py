"""The gen3ou FORMAT SPEC is complete, one-story-per-rule, and every derived list matches the pinned source.

Each derived declaration (the Uber tier, the OHKO moves, Accuracy Trap's accuracy moves, the ability-locked
species, the rule closure) is RE-DERIVED here from `deps/pokemon-showdown`, so a hand edit that drifts from the
engine fails; the comparison with MASTER is `main/format_drift_test.py`.
"""
from __future__ import annotations

import re

import pytest

from agents import gen3_data
from agents.gen3_data import format_spec as fs
from main import format_drift
from utils.paths import repo_path

PS = repo_path("deps", "pokemon-showdown")


def _text(rel: str) -> str:
    return (PS / rel).read_text(encoding="utf-8")


def test_every_rule_has_exactly_one_story_a_reason_and_a_source():
    seen = set()
    for r in fs.RULES:
        assert isinstance(r.story, fs.Story), r.name
        assert r.reason.strip() and r.source.strip(), r.name
        assert r.id not in seen, r.name
        seen.add(r.id)
        for b in r.bans:
            assert b.source and b.via and b.ids, b
            assert r.story in (fs.Story.PRIOR_ZERO, fs.Story.TEAM_BUILDING), (r.name, r.story)
            # a PRIOR_ZERO rule bans entities (or a species-move); combos that ban no entity are team-building
            assert (b.kind == "combo") == (r.story is fs.Story.TEAM_BUILDING), (r.name, b.name)


def test_the_rule_table_is_the_pinned_rule_closure():
    """Every rule gen3ou's ruleset reaches (format entry → gen 3 Standard → gen 4 Standard AG) is declared, and
    nothing else is (the two banlist pseudo-rules apart)."""
    facts = format_drift.read_facts(PS)
    named = set(facts.format_ruleset) | set(facts.standard) | set(facts.standard_ag)
    declared = {r.name for r in fs.RULES} - {"Banlist entities", "Banlist combos"}
    assert declared == named
    by_parent = {r.name: r.parent for r in fs.RULES}
    assert all(by_parent[n] == "format" for n in facts.format_ruleset)
    assert all(by_parent[n] == "Standard" for n in facts.standard)
    assert all(by_parent[n] == "Standard AG" for n in facts.standard_ag)


def test_uber_species_and_their_cited_lines():
    fd = _text("data/mods/gen3/formats-data.ts").splitlines()
    facts = format_drift.read_facts(PS)
    assert {sid for sid, _ in fs.UBER_SPECIES} == set(facts.ubers)
    for sid, line in fs.UBER_SPECIES:
        assert fd[line - 1] == f"\t{sid}: {{", (sid, line)
        assert 'tier: "Uber"' in fd[line], (sid, line)


def _gen3_move_blocks():
    text = _text("data/moves.ts")
    for m in re.finditer(r"^\t(\w+): \{\n.*?^\t\},$", text, re.M | re.S):
        num = re.search(r"^\t\tnum: (-?\d+),", m.group(0), re.M)
        if num and 0 < int(num.group(1)) <= 354:
            yield m.group(1), m.group(0)


def test_ohko_bans_are_every_gen3_ohko_move_with_its_line():
    lines = _text("data/moves.ts").splitlines()
    derived = {mid for mid, b in _gen3_move_blocks() if re.search(r"^\t\tohko: ", b, re.M)}
    assert {b.ids[0] for b in fs.OHKO_BANS} == derived
    for b in fs.OHKO_BANS:
        line = int(b.source.split(":")[2].split(" ")[0])
        assert lines[line - 1].strip().startswith("ohko:"), b


def test_accuracy_trap_accuracy_moves_are_derived_from_the_dex():
    """`data/rulesets.ts` accuracytrapclause: a move whose own `boosts.accuracy < 0`, or a 100 %-chance
    secondary lowering accuracy — over the gen-3 moves."""
    derived = set()
    for mid, b in _gen3_move_blocks():
        own = re.search(r"^\t\tboosts: \{(.*?)\}", b, re.M | re.S)
        sec = re.search(r"^\t\tsecondary: \{(.*?)^\t\t\}", b, re.M | re.S)
        if own and re.search(r"accuracy: -\d", own.group(1)):
            derived.add(mid)
        if sec and re.search(r"chance: 100,", sec.group(1)) and re.search(r"accuracy: -\d", sec.group(1)):
            derived.add(mid)
        assert "secondaries: [" not in b or mid not in fs.ACCURACY_TRAP_ACCURACY_MOVES
    assert derived == set(fs.ACCURACY_TRAP_ACCURACY_MOVES)


def test_ability_locked_species_are_derived_from_the_gen3_dex():
    """A species every one of whose gen-3 abilities (slots 0/1, abilities of gen <= 3) is banned has no legal
    set: the spec must name it so its species prior can be 0."""
    text = _text("data/pokedex.ts")
    banned = fs.GEN3OU.banned_abilities
    derived = set()
    for m in re.finditer(r"^\t(\w+): \{\n\t\tnum: (\d+),.*?^\t\tabilities: \{([^}]*)\}", text, re.M | re.S):
        sid, num = m.group(1), int(m.group(2))
        if not (0 < num <= 386) or gen3_data.species.get(sid) is None:
            continue
        slots = dict(re.findall(r"(\w+): \"([^\"]+)\"", m.group(3)))
        abilities = {fs.to_id(v) for k, v in slots.items() if k in ("0", "1")}
        legal = {a for a in abilities if gen3_data.abilities.get(a) is not None}
        if legal and legal <= banned:
            derived.add(sid)
    assert derived == {sid for sid, _ in fs.ABILITY_LOCKED_SPECIES}
    assert {b.ids[0] for b in fs.ABILITY_LOCKED_BANS} <= fs.GEN3OU.banned_species


@pytest.mark.parametrize("kind,ids", [
    ("item", ("quickclaw", "brightpowder", "laxincense")),
    ("move", ("assist", "swagger", "doubleteam", "minimize", "fissure", "sheercold")),
    ("ability", ("sandveil", "soundproof")),
    ("species", ("mewtwo", "deoxysspeed", "wobbuffet", "mrmime", "sandslash")),
])
def test_every_banned_entity_is_a_real_gen3_id(kind, ids):
    dex = {"item": gen3_data.items, "move": gen3_data.moves, "ability": gen3_data.abilities,
           "species": gen3_data.species}[kind]
    for i in ids:
        assert fs.GEN3OU.is_banned(kind, i), i
        assert dex.get(i) is not None, i
    for b in fs.GEN3OU.bans:
        if b.kind == kind and b.ids[0] != "acupressure":          # Acupressure: gen 4, on the list verbatim
            assert dex.get(b.ids[0]) is not None, b


def test_species_move_ban_and_combo_ids():
    assert fs.GEN3OU.is_banned("move", "ingrain", species="smeargle")
    assert not fs.GEN3OU.is_banned("move", "ingrain", species="celebi")
    assert not fs.GEN3OU.is_banned("move", "ingrain")
    for b in fs.GEN3OU.combo_bans:
        for i in b.ids:
            assert gen3_data.moves.get(i) is not None or gen3_data.species.get(i) is not None, b


def test_pinned_differences_are_master_only_bans():
    assert ("format_banlist", "Quick Claw") in fs.PINNED_DIFFERENCES
    qc = next(b for b in fs.GEN3OU.bans if b.ids == ("quickclaw",))
    assert not qc.on_pinned and fs.MASTER_SNAPSHOT_COMMIT[:8] in qc.source
    assert all(b.on_pinned for b in fs.GEN3OU.bans if b.ids != ("quickclaw",))


def test_without_rule_moves_the_board_state_readers(monkeypatch):
    assert fs.sleep_clause_mod() and fs.freeze_clause_mod()
    monkeypatch.setattr(fs, "_ACTIVE", fs.without_rule(fs.GEN3OU, "sleepclausemod"))
    assert not fs.sleep_clause_mod() and fs.freeze_clause_mod()
    monkeypatch.setattr(fs, "_ACTIVE", fs.without_rule(fs.GEN3OU, "freezeclausemod"))
    assert fs.sleep_clause_mod() and not fs.freeze_clause_mod()


def test_the_format_id_is_the_one_the_rust_env_core_accepts():
    """Every training / eval game is created by the Rust env core, which REFUSES any other format."""
    spec_rs = repo_path("src", "rust_env", "src", "core", "spec.rs").read_text()
    assert f'if self.format_id != "{fs.FORMAT_ID}"' in spec_rs
