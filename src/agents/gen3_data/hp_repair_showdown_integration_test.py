"""The Hidden Power repair against the PINNED Showdown (Node): every repaired set validates, plays the type it is
written with at power 70 (``dex.getHiddenPower``), keeps its IVs through the validator, and its stored Speed moves
by the 30 Spe IV (the speed-tie pin); the variable-power attack list is Showdown's."""
from __future__ import annotations

import json
import subprocess

import pytest

from agents.gen3_data import hp_repair as h
from agents.gen3_data.hp_repair_test import DUGTRIO, ZAPDOS
from utils.bridge.hp_set_check import check_teams
from utils.paths import repo_path

pytestmark = pytest.mark.integration

SALAMENCE = """Salamence @ Leftovers
Ability: Intimidate
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Dragon Dance
- Earthquake
- Rock Slide
- Hidden Power"""


def _forced(text: str, hp_type: str) -> str:
    """The repair's own writing path with the type pinned (restricted to one type)."""
    out = text.replace("- Hidden Power", f"- Hidden Power [{hp_type}]")
    return out.replace("Jolly Nature", "Jolly Nature\n" + h.hp_iv_line(hp_type))


def test_repaired_sets_validate_and_play_their_written_type():
    teams = []
    for seed in range(12):
        out, recs = h.repair_team(DUGTRIO + "\n\n" + ZAPDOS, seed)
        assert len(recs) == 2
        teams.append((out, recs))
    verdicts = check_teams([t for t, _ in teams])
    for (_, recs), v in zip(teams, verdicts):
        assert v["valid"], v["errors"]
        for rec in recs:
            s = v["sets"][rec.slot]
            assert s["hp_type"] == rec.hp_type and s["power"] == 70
            assert s["validator_hp_type"] == rec.hp_type
            assert s["ivs_in"] == rec.ivs == s["ivs_after_validate"]


def test_speed_tie_pin_hp_fire_salamence_is_one_point_slower():
    texts = [SALAMENCE] + [_forced(SALAMENCE, t) for t in ("Fire", "Psychic", "Rock", "Fighting", "Flying")]
    verdicts = check_teams(texts)
    spe = [v["sets"][0]["stats"]["spe"] for v in verdicts]
    assert all(v["valid"] for v in verdicts), [v["errors"] for v in verdicts]
    assert verdicts[0]["sets"][0]["hp_type"] == "Dark"
    assert spe == [328, 327, 327, 327, 327, 328]          # a Dark / Flying mirror no longer ties a Fire one
    assert spe[1] == h.gen3_stat("spe", 100, 30, 252, 100, 1.1)


def test_variable_power_attacks_are_showdowns():
    js = ("const {Dex}=require(process.argv[1]);const d=Dex.mod('gen3');const o=[];"
          "for(const m of d.moves.all()){if(m.gen>3||m.isNonstandard||m.category==='Status'||m.basePower>0)continue;"
          "if(m.id==='hiddenpower'||m.damage||m.damageCallback||m.ohko)continue;o.push(m.id)}"
          "console.log(JSON.stringify(o.sort()))")
    p = subprocess.run(["node", "-e", js, str(repo_path("deps", "pokemon-showdown"))],
                       capture_output=True, text=True, check=True)
    derived = set(json.loads(p.stdout))
    # Bide (BP 0, no callback) returns double the damage taken: it does not use the attacking stat.
    assert derived - {"bide"} == h.VARIABLE_POWER_ATTACKS


def test_every_type_ivs_survive_the_validator_at_power_70():
    names = [t.capitalize() for t in h.HP_TYPES]
    verdicts = check_teams([_forced(SALAMENCE, t) for t in names])
    for t, v in zip(names, verdicts):
        s = v["sets"][0]
        assert v["valid"], (t, v["errors"])
        assert (s["hp_type"], s["power"], s["validator_hp_type"]) == (t, 70, t)
        assert s["ivs_in"] == h.HP_IVS[t] == s["ivs_after_validate"]
