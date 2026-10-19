"""The Hidden Power repair (``gen3_hp_repair_v1``): only an untyped all-31 HP is re-sampled, onto the set's side,
from Smogon's species usage, with the validator's IVs; everything else stays byte-identical. Each test fails on
revert of the rule it names; the mechanics constants are re-read from the pinned Showdown source."""
from __future__ import annotations

import re

import pytest

from agents.gen3_data import hp_repair as h
from utils.paths import repo_path

# The corpus's first team's lead (the owner's 2026-09-25 example): Choice Band Adamant Dugtrio, untyped HP.
DUGTRIO = """Dugtrio @ Choice Band
Ability: Arena Trap
EVs: 0 HP / 252 Atk / 0 Def / 0 SpA / 4 SpD / 252 Spe
Adamant Nature
IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
- Earthquake
- Rock Slide
- Aerial Ace
- Hidden Power"""

ZAPDOS = """Zapdos @ Leftovers
Ability: Pressure
EVs: 252 HP / 0 Atk / 0 Def / 4 SpA / 0 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Hidden Power
- Toxic
- Substitute"""

TYPED = DUGTRIO.replace("- Hidden Power", "- Hidden Power [Ice]")
BY_IVS = DUGTRIO.replace("IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe",
                         "IVs: 31 HP / 31 Atk / 31 Def / 30 SpA / 30 SpD / 31 Spe")   # HP Ground by IVs
NO_HP = DUGTRIO.replace("- Hidden Power", "- Substitute")


def _lines(text: str) -> list:
    return text.split("\n")


def test_target_rule_untyped_all31_only():
    for text in (TYPED, BY_IVS, NO_HP):
        out, recs = h.repair_team(text, 1)
        assert recs == [] and out == text            # untouched, byte for byte
    assert len(h.repair_team(DUGTRIO, 1)[1]) == 1
    assert len(h.repair_team(ZAPDOS, 1)[1]) == 1     # no IVs line = all 31 = a target too


def test_dugtrio_gets_physical_type_with_validator_ivs_and_only_two_lines_change():
    out, (rec,) = h.repair_team(DUGTRIO, h.team_seed(7, "team_x"))
    assert rec.votes.side == h.PHYSICAL and (rec.votes.evs, rec.votes.nature, rec.votes.moves) == (1, 1, 1)
    assert h.move_side(rec.hp_type) == h.PHYSICAL and rec.hp_type != "Dark"
    changed = [(a, b) for a, b in zip(_lines(DUGTRIO), _lines(out)) if a != b]
    assert len(_lines(out)) == len(_lines(DUGTRIO)) and len(changed) == 2
    assert f"- Hidden Power [{rec.hp_type}]" in _lines(out)
    assert h.hp_iv_line(rec.hp_type) in _lines(out)
    assert h.hidden_power_type_of(out) == rec.hp_type
    assert rec.prior_source == "species"
    assert set(rec.restricted) <= {t for t in h.HP_TYPES if h.move_side(t) == h.PHYSICAL}
    assert abs(sum(rec.restricted.values()) - 1.0) < 1e-12


def test_missing_iv_line_is_inserted_before_the_first_move():
    out, (rec,) = h.repair_team(ZAPDOS, 3)
    lines = _lines(out)
    assert rec.votes.side == h.SPECIAL and h.move_side(rec.hp_type) == h.SPECIAL
    assert lines[lines.index("- Thunderbolt") - 1] == h.hp_iv_line(rec.hp_type)
    assert len(lines) == len(_lines(ZAPDOS)) + 1


def test_restricted_distribution_is_smogons_species_usage_on_the_side():
    from agents.gen3_data import priors
    src, dist = h.side_distribution("salamence", h.PHYSICAL)
    raw = {t: p for t, p in priors.hidden_power("salamence").items() if h.move_side(t) == h.PHYSICAL}
    tot = sum(raw.values())
    assert src == "species" and dist == pytest.approx({t: p / tot for t, p in raw.items()})
    _, both = h.side_distribution("salamence", None)
    assert both == pytest.approx(priors.hidden_power("salamence"))


def test_fallback_is_the_format_wide_smogon_usage(monkeypatch):
    from agents.gen3_data import priors
    monkeypatch.setattr(priors, "hidden_power", lambda sp: {"grass": 1.0})   # no physical mass
    src, dist = h.side_distribution("salamence", h.PHYSICAL)
    fmt = h.format_hp_usage()
    tot = sum(p for t, p in fmt.items() if h.move_side(t) == h.PHYSICAL)
    assert src == "format" and dist == pytest.approx(
        {t: p / tot for t, p in fmt.items() if h.move_side(t) == h.PHYSICAL and p > 0})


def test_deterministic_per_team_seed_and_recorded_draw():
    a = h.repair_team(DUGTRIO + "\n\n" + ZAPDOS, h.team_seed(20261018, "team_000001"))
    b = h.repair_team(DUGTRIO + "\n\n" + ZAPDOS, h.team_seed(20261018, "team_000001"))
    assert a[0] == b[0] and [r.as_dict() for r in a[1]] == [r.as_dict() for r in b[1]]
    assert h.team_seed(20261018, "team_000001") != h.team_seed(20261018, "team_000002")
    for r in a[1]:
        assert h.draw_type(r.restricted, r.u).capitalize() == r.hp_type


def test_side_votes_and_stat_tiebreak():
    # Return (BP 0, variable power) votes physical; Seismic Toss (fixed damage) does not vote.
    v = h.judge_side("snorlax", {"hp": 252, "def": 252}, "careful", ["return", "seismictoss"])
    assert (v.evs, v.nature, v.moves, v.side) == (0, 1, 1, h.PHYSICAL)
    v = h.judge_side("snorlax", {"hp": 252, "def": 252}, "hardy", ["seismictoss"])
    assert (v.evs, v.nature, v.moves) == (0, 0, 0) and v.decided_by == "stat"
    assert v.side == h.PHYSICAL                       # base Atk 110 > SpA 65
    v = h.judge_side("salamence", {"atk": 252, "spa": 0}, "modest", ["flamethrower"])
    assert (v.evs, v.nature, v.moves, v.side) == (1, -1, -1, h.SPECIAL)


def test_gen3_stat_formula_and_the_speed_tie_change():
    # Jolly 252 Spe Salamence (base 100): 31 Spe = 328, HP Fire's 30 Spe = 327 (pinned against Showdown's
    # stored stats in hp_repair_showdown_integration_test.py). A mirror that tied now strictly loses.
    nat = 1.1
    assert h.gen3_stat("spe", 100, 31, 252, 100, nat) == 328
    for t in ("Fire", "Psychic", "Rock", "Fighting"):
        assert h.HP_IVS[t]["spe"] == 30
        assert h.gen3_stat("spe", 100, h.HP_IVS[t]["spe"], 252, 100, nat) == 327
    assert all(h.HP_IVS[t]["spe"] == 31 for t in h.HP_IVS if t not in ("Fire", "Psychic", "Rock", "Fighting"))


def test_hp_ivs_is_the_one_table():
    # Pinned to typechart.ts's HPivs (fillStats, 31) in hidden_power_test.py.
    from agents.gen3_data import hidden_power

    assert h.HP_IVS is hidden_power.HP_IVS


def test_special_types_are_showdowns_gen3_list():
    for rel in (("data", "mods", "gen3", "scripts.ts"), ("data", "mods", "gen3", "moves.ts")):
        src = repo_path("deps", "pokemon-showdown", *rel).read_text()
        (lst,) = set(re.findall(r"const specialTypes = \[([^\]]*)\]", src))
        assert {s.strip().strip("'\"").lower() for s in lst.split(",")} == h.SPECIAL_TYPES
