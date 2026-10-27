"""A Hidden Power that names its type but carries NO IV line plays its NAMED type, with Showdown's validator IVs, in
training (our pack + the Rust engine) exactly as in Showdown (H-CTX FINDING, 2026-10-27).

The class: 736 of the training pool's 1,912 Hidden Power sets are written ``- Hidden Power [Bug]`` with no ``IVs:``
line (``data/teams``; d3_setmut_v1 carries 2,234 more). Three readings were possible:

(a) the named type, with the IVs the validator writes (``sim/team-validator.ts`` ~1154-1164:
    ``fillStats(HPivs, 31)`` when every IV is 31);
(b) the type the all-31 IVs give (Dark in gen 3);
(c) the named type with the all-31 stats: what Showdown's SIM plays from an UNVALIDATED pack (``sim/pokemon.ts``
    ~357-410: ``set.hpType`` comes from the move, ``hpPower`` from the IVs).

``Teams.import`` itself never fills them: its bracket fill (``sim/teams.ts`` ~539-546) is guarded by ``!set.ivs``, and
``Teams.import`` creates every set with all-31 IVs first (~615), so only the validator decides. Training plays (a):
``fix_gen3_hp_ivs`` writes the validator's table before the pack (and ``gen3_team_pack_oracle_v1`` refuses any pool
pack Showdown would read differently). This file pins that the pack IS (a) for both spellings, that the three
readings are distinguishable in a real battle, and that the Rust engine plays each of them byte for byte as
Showdown's own sim does.

The live-server rejection that raised the FINDING ("Aerodactyl has Hidden Power Bug, but its IVs are for Hidden
Power Water") came from the KAKUNA side: Metamon's poke-env packs ``TeambuilderPokemon.HP_TO_IVS``, a table in the
formula's bit order (hp, atk, def, spe, spa, spd) written into the pack's (hp, atk, def, spa, spd, spe), so Bug's row
reads as Water. Our side's pack passes the same validator unchanged.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
from typing import Dict, List

import pytest

from agents.gen3_data import hidden_power
from utils.bridge.hp_set_check import check_teams
from utils.bridge.sim_bridge_bin import _BRIDGE_JS, _ENV_OVERRIDE, expected_bin_path
from utils.teambuilder import Gen3Teambuilder

pytestmark = pytest.mark.sim

#: Verbatim from the pool (``data/teams/sample/9283210847f806ee.txt``): Choice Band, so Hidden Power Bug is PHYSICAL
#: (gen 3 splits by type) and its damage reads the Attack the IVs set; Dark would be special.
AERO_BRACKET = """Aerodactyl @ Choice Band
Ability: Rock Head
EVs: 224 Atk / 32 SpD / 252 Spe
Jolly Nature
- Earthquake
- Rock Slide
- Double-Edge
- Hidden Power [Bug]
"""
AERO_SPACE = AERO_BRACKET.replace("Hidden Power [Bug]", "Hidden Power Bug")
#: Bug is 4x on Celebi (Grass / Psychic), Dark 2x: the type is visible in the damage line.
CELEBI = """Celebi @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 4 Def / 252 SpD
Calm Nature
- Calm Mind
- Recover
- Psychic
- Leech Seed
"""
#: The two packs training never plays, written by hand: (c) the unvalidated pack, (b) the all-31 bare Hidden Power.
PACK_C = "Aerodactyl||choiceband|rockhead|earthquake,rockslide,doubleedge,hiddenpowerbug|Jolly|,224,,,32,252|||||"
PACK_B = "Aerodactyl||choiceband|rockhead|earthquake,rockslide,doubleedge,hiddenpower|Jolly|,224,,,32,252|||||"
SEED = [1, 2, 3, 4]
#: Showdown's battle stats (atk, def, spa, spd, spe) for the validator's Bug IVs (atk / def / spd 30) and for all 31.
STATS_A = {"atk": 301, "def": 165, "spa": 140, "spd": 193, "spe": 394}
STATS_ALL31 = {"atk": 302, "def": 166, "spa": 140, "spd": 194, "spe": 394}


def _pack(paste: str) -> str:
    (packed,) = Gen3Teambuilder([paste]).packed_teams
    return packed


def _play(argv: List[str], p1: str, p2: str) -> Dict[str, List[str]]:
    """One turn (p1 Hidden Power, p2 Calm Mind), then p2 forfeits: each side's protocol chunks, ``|t:|`` dropped."""
    cmds = ("START " + json.dumps({"formatid": "gen3ou", "seed": SEED, "p1": {"name": "a", "team": p1},
                                   "p2": {"name": "b", "team": p2}})
            + "\nCHOOSE p1 move 4\nCHOOSE p2 move 1\nFORCELOSE p2\n")
    p = subprocess.run(argv, input=cmds, capture_output=True, text=True, timeout=120, check=False)
    assert p.returncode == 0 and "__END__" in p.stdout, f"{argv[0]} exit {p.returncode}: {p.stderr[-1500:]}"
    out: Dict[str, List[str]] = {"p1": [], "p2": []}
    for line in p.stdout.splitlines():
        tag, _, b64 = line.partition(" ")
        if tag in out:
            chunk = base64.b64decode(b64).decode()
            out[tag].append("\n".join(x for x in chunk.split("\n") if not x.startswith("|t:|")))
    return out


def _facts(out: Dict[str, List[str]]) -> Dict[str, object]:
    """What p1's own request shows (stats, the Hidden Power slot) and the turn's damage line on Celebi."""
    lines = [x for chunk in out["p1"] for x in chunk.split("\n")]
    req = next(json.loads(x[len("|request|"):]) for x in lines if x.startswith("|request|{"))
    mon = req["side"]["pokemon"][0]
    hp_slot = next(m["move"] for m in req["active"][0]["moves"] if m["id"] == "hiddenpower")
    damage = next(x for x in lines if x.startswith("|-damage|p2a: Celebi|"))
    return {"stats": mon["stats"], "move_ids": mon["moves"], "hp_slot": hp_slot, "damage": damage}


def _node() -> List[str]:
    return ["node", _BRIDGE_JS]


def _prebuilt_rust() -> str:
    override = os.environ.get(_ENV_OVERRIDE)
    if override and os.path.exists(override):
        return override
    path = expected_bin_path("sim_bridge")
    if not path.is_file():
        pytest.skip(f"no pre-built rust sim_bridge (set {_ENV_OVERRIDE} or build src/rust_sim); "
                    "skipping to avoid a cargo build in the routine suite")
    return str(path)


@pytest.mark.parametrize("paste", [AERO_BRACKET, AERO_SPACE], ids=["bracket", "space"])
def test_our_pack_is_the_validators_ivs_for_both_spellings(paste: str) -> None:
    (verdict,) = check_teams([paste])
    assert verdict["valid"], verdict["errors"]
    (s,) = verdict["sets"]
    # Showdown's import leaves every IV at 31 (its bracket fill is dead code); the validator writes Bug's.
    assert s["ivs_in"] == {k: 31 for k in hidden_power.STATS}
    assert s["ivs_after_validate"] == hidden_power.ivs_of("Bug")
    assert (s["validator_hp_type"], s["hp_type"]) == ("Bug", "Dark")
    assert _pack(paste) == ("Aerodactyl||choiceband|rockhead|earthquake,rockslide,doubleedge,hiddenpowerbug|Jolly|"
                            ",224,,,32,252||,30,30,,30,|||")


def test_the_three_readings_play_differently_in_showdown_and_training_plays_a() -> None:
    celebi = _pack(CELEBI)
    a = _facts(_play(_node(), _pack(AERO_BRACKET), celebi))
    c = _facts(_play(_node(), PACK_C, celebi))
    b = _facts(_play(_node(), PACK_B, celebi))
    assert a == {"stats": STATS_A, "move_ids": ["earthquake", "rockslide", "doubleedge", "hiddenpowerbug"],
                 "hp_slot": "Hidden Power Bug 70", "damage": "|-damage|p2a: Celebi|3/100"}
    assert (c["stats"], c["hp_slot"], c["damage"]) == (STATS_ALL31, "Hidden Power Bug 70", "|-damage|p2a: Celebi|2/100")
    assert (b["stats"], b["hp_slot"], b["damage"]) == (STATS_ALL31, "Hidden Power Dark 70",
                                                       "|-damage|p2a: Celebi|89/100")


@pytest.mark.parametrize("arm", ["a", "c", "b"])
def test_the_rust_engine_plays_each_reading_byte_for_byte_as_showdown(arm: str) -> None:
    rust = [_prebuilt_rust()]
    p1 = {"a": _pack(AERO_BRACKET), "c": PACK_C, "b": PACK_B}[arm]
    celebi = _pack(CELEBI)
    node_out, rust_out = _play(_node(), p1, celebi), _play(rust, p1, celebi)
    assert len(node_out["p1"]) > 3, "a battle that relayed almost nothing proves nothing"
    for side in ("p1", "p2"):
        assert rust_out[side] == node_out[side], f"arm ({arm}) {side}: the Rust engine's stream differs from Showdown's"
