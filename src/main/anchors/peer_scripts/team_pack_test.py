"""The Metamon peer's team pack (gen3_peer_showdown_pack_v1, ``EXTERNAL_ANCHORS_SOP.md`` H22): every team file packed
as a real Showdown server plays it, checked by the pack oracle, or no play.

The two defects the sweep found on Kakuna's side are the fixtures: forum72 ``sft431``'s Gyarados (an explicit
``Happiness: 0`` with Frustration, which poke-env packed as 255, base power 1) and a named Hidden Power with no IV line
(poke-env's mis-ordered table played another type). The legacy packer is modelled by the pack poke-env wrote for each
(its exact strings, from ``TeambuilderPokemon`` in Metamon's env).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from main.anchors.peer_scripts import team_pack as tp
from utils.paths import src_path

pytestmark = pytest.mark.integration

BRIDGE = str(src_path("utils", "bridge"))

#: forum72 sft431's Gyarados, verbatim.
GYARADOS = """Gyarados (M) @ Leftovers
Ability: Intimidate
Happiness: 0
EVs: 88 HP / 224 Atk / 196 Spe
Jolly Nature
IVs: 30 HP / 30 Atk / 30 SpD / 30 Spe
- Dragon Dance
- Earthquake
- Frustration
- Hidden Power [Rock]
"""
#: What upstream poke-env packed for it: happiness dropped (``happiness or ""``), so the server plays 255.
GYARADOS_POKEENV = ("Gyarados||leftovers|intimidate|dragondance,earthquake,frustration,hiddenpowerrock|Jolly|"
                    "88,224,,,,196|M|30,30,,,30,30|||")
#: A pool set: a named Hidden Power, no IV line (the validator writes Bug's IVs).
AERODACTYL = """Aerodactyl @ Choice Band
Ability: Rock Head
EVs: 224 Atk / 32 SpD / 252 Spe
Jolly Nature
- Earthquake
- Rock Slide
- Double-Edge
- Hidden Power [Bug]
"""
#: poke-env's pack: its HP_TO_IVS["bug"] row (formula bit order written as stat order), which reads as Water.
AERODACTYL_POKEENV = ("Aerodactyl||choiceband|rockhead|earthquake,rockslide,doubleedge,hiddenpowerbug|Jolly|"
                      ",224,,,32,252||,,,30,,30|||")


def _files(tmp_path: Path, **texts: str) -> list:
    out = []
    for name, text in texts.items():
        p = tmp_path / f"{name}.gen3ou_team"
        p.write_text(text)
        out.append(str(p))
    return out


def test_the_pack_is_what_showdown_plays_and_the_legacy_difference_is_named(tmp_path) -> None:
    files = _files(tmp_path, sft431=GYARADOS, aero=AERODACTYL)
    legacy = {GYARADOS: GYARADOS_POKEENV, AERODACTYL: AERODACTYL_POKEENV}
    packs, record = tp.build(files, "gen3ou", BRIDGE, legacy_pack=legacy.__getitem__)
    gyarados, aero = tp.pack_of(packs, files[0]), tp.pack_of(packs, files[1])
    assert gyarados.endswith("|M|30,30,,,30,30|||0,Rock,,,,"), gyarados   # happiness 0, as Showdown reads it
    assert "|,30,30,,30,|" in aero, aero                       # the validator's Bug IVs (atk / def / spd 30)
    assert record["schema"] == tp.PEER_PACK_SCHEMA and record["n_files"] == 2
    assert record["peer_pack_changed"] is True
    diffs = {c["file"]: c["diffs"] for c in record["changed"]}
    assert diffs["sft431.gen3ou_team"] == ["mon 1 (gyarados) happiness: legacy 255, now 0"]
    assert any("hiddenpower: legacy Water:70, now Bug:70" in d for d in diffs["aero.gen3ou_team"])


def test_a_set_the_legacy_packer_played_the_same_keeps_the_legacy_regime(tmp_path) -> None:
    files = _files(tmp_path, aero=AERODACTYL)
    packs, _ = tp.build(files, "gen3ou", BRIDGE)
    _, record = tp.build(files, "gen3ou", BRIDGE, legacy_pack=lambda text: tp.pack_of(packs, files[0]))
    assert record["peer_pack_changed"] is False and record["changed"] == []


def test_a_legacy_packer_that_crashed_is_a_change(tmp_path) -> None:
    def crash(text: str) -> str:
        raise ValueError("poke-env could not split the nickname")

    _, record = tp.build(_files(tmp_path, aero=AERODACTYL), "gen3ou", BRIDGE, legacy_pack=crash)
    assert record["peer_pack_changed"] is True
    assert record["changed"][0]["diffs"] == ["the legacy packer could not pack this file"]


def test_a_file_the_validator_refuses_never_plays_and_is_named(tmp_path) -> None:
    files = _files(tmp_path, ok=AERODACTYL, banned=AERODACTYL.replace("Aerodactyl", "Mewtwo"))
    with pytest.raises(tp.TeamPackError, match=r"banned\.gen3ou_team"):
        tp.build(files, "gen3ou", BRIDGE)


def test_the_oracle_refuses_a_pack_that_is_not_showdowns_reading(tmp_path, monkeypatch) -> None:
    """The guard on its own: had the peer kept poke-env's pack, startup would refuse, naming the file and the set
    (fails if the oracle check is removed from ``build``)."""
    files = _files(tmp_path, sft431=GYARADOS)
    monkeypatch.setattr(tp, "showdown_packs", lambda pastes, legacy, fmt, bridge_dir: [
        {"packed": GYARADOS_POKEENV, "errors": [], "legacy_diffs": []} for _ in pastes])
    with pytest.raises(tp.TeamPackError, match=r"sft431\.gen3ou_team: mon 1 \(gyarados\) happiness: Showdown plays 0"):
        tp.build(files, "gen3ou", BRIDGE)


def test_a_file_not_packed_at_startup_never_plays(tmp_path) -> None:
    packs, _ = tp.build(_files(tmp_path, aero=AERODACTYL), "gen3ou", BRIDGE)
    with pytest.raises(tp.TeamPackError, match="not packed and verified at startup"):
        tp.pack_of(packs, str(tmp_path / "other.gen3ou_team"))
    with pytest.raises(tp.TeamPackError, match="no team files"):
        tp.build([], "gen3ou", BRIDGE)
