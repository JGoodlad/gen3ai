"""Unit tests for the reconstruction record + capture registry (no Node, no battles)."""

import json
import os

import pytest

from utils.bridge import reconstruction
from utils.bridge.reconstruction import (
    RECON_SUFFIX,
    ReconstructionRecord,
    decode_packed_team,
    offer_record,
    pop_record,
    register_trace_prefix,
)


# Minimal REAL packed mons (12 pipe-fields each) so team decode paths work on
# the fixture record too.
_PACKED1 = "Zapdos||leftovers|pressure|thunderbolt,hiddenpowerice,substitute,batonpass|Timid|4,,,252,,252||,2,,30,,|||"
_PACKED2 = "Skarmory||leftovers|keeneye|spikes,drillpeck,protect,roar|Careful|252,,,,252,4|||||"


def _raw(tag_hint: str = "x") -> dict:
    """A minimal, structurally valid raw record (as the bridge JS emits it)."""
    return {
        "v": 1,
        "format_id": "gen3ou",
        "prng_seed": "sodium,abcdef0123456789",
        "input_log": [
            '>start {"formatid":"gen3ou","seed":"sodium,abcdef0123456789"}',
            f'>player p1 {{"name":"Alice","team":"{_PACKED1}"}}',
            f'>player p2 {{"name":"Bob","team":"{_PACKED2}"}}',
            ">p1 move thunderbolt",
            ">p2 move surf",
        ],
        "commands": [["p1", "move 1"], ["p2", "move 2"], ["forcelose", "p2"]],
        "hint": tag_hint,
    }


# ---------------------------------------------------------------------------
# Record type
# ---------------------------------------------------------------------------

def test_record_dict_roundtrip():
    rec = ReconstructionRecord.from_dict({**_raw(), "battle_tag": "battle-gen3ou-7",
                                          "trainee_username": "Alice"})
    rec2 = ReconstructionRecord.from_dict(rec.to_dict())
    assert rec2 == rec
    assert rec2.battle_tag == "battle-gen3ou-7"
    assert rec2.trainee_username == "Alice"
    assert rec2.commands[2] == ("forcelose", "p2")


def test_record_file_roundtrip(tmp_path):
    rec = ReconstructionRecord.from_dict(_raw())
    path = tmp_path / "r.json"
    rec.save(path)
    assert ReconstructionRecord.load(path) == rec


def test_record_derived_views():
    rec = ReconstructionRecord.from_dict(_raw())
    assert rec.start_options()["seed"] == "sodium,abcdef0123456789"
    players = rec.players()
    assert players["p1"]["name"] == "Alice"
    assert players["p2"]["team"] == _PACKED2
    assert rec.username("p2") == "Bob"
    assert rec.packed_team("p1") == _PACKED1
    assert rec.side_of("Bob") == "p2"
    with pytest.raises(KeyError):
        rec.side_of("Mallory")


def test_record_without_start_line_raises():
    raw = _raw()
    raw["input_log"] = [l for l in raw["input_log"] if not l.startswith(">start")]
    with pytest.raises(ValueError):
        ReconstructionRecord.from_dict(raw).start_options()


def test_decode_packed_team_fields_and_omission_defaults(monkeypatch):
    """Pins the packed format's omission conventions: empty IV field = all 31,
    a PARTIAL field (",0,,,,") fills blanks with 31, empty EV field = all 0,
    omitted level = 100. The full-pool sim cross-check lives in
    packed_team_decode_integration_test.py; this is the cheap shape guard."""
    monkeypatch.setattr(reconstruction, "_ALIAS_CACHE", {})  # pure unit: no node
    packed = (
        "Jirachi||leftovers|serenegrace|calmmind,icepunch,thunderbolt,hiddenpowerfighting"
        "|Quiet|252,,68,140,,48||,,30,30,30,30|||"
        "]Celebi||leftovers|naturalcure|leechseed,recover,psychic,calmmind"
        "|Bold|252,,220,,,36||,0,,,,|||"
        "]Snorlax||leftovers|immunity|bodyslam,earthquake,selfdestruct,shadowball"
        "|Adamant||||||"
    )
    mons = decode_packed_team(packed)
    assert [m["species"] for m in mons] == ["jirachi", "celebi", "snorlax"]

    jirachi = mons[0]
    assert jirachi["moves"] == ["calmmind", "icepunch", "thunderbolt", "hiddenpowerfighting"]
    assert jirachi["item"] == "leftovers" and jirachi["ability"] == "serenegrace"
    assert jirachi["nature"] == "quiet" and jirachi["level"] == 100
    assert jirachi["evs"] == {"hp": 252, "atk": 0, "def": 68, "spa": 140, "spd": 0, "spe": 48}
    # partial IVs ",,30,30,30,30" → blanks (hp, atk) default to 31
    assert jirachi["ivs"] == {"hp": 31, "atk": 31, "def": 30, "spa": 30, "spd": 30, "spe": 30}

    celebi = mons[1]
    assert celebi["ivs"]["atk"] == 0   # the explicit 0 survives
    assert all(celebi["ivs"][s] == 31 for s in ("hp", "def", "spa", "spd", "spe"))

    snorlax = mons[2]
    assert snorlax["evs"] == {s: 0 for s in ("hp", "atk", "def", "spa", "spd", "spe")}
    assert snorlax["ivs"] == {s: 31 for s in ("hp", "atk", "def", "spa", "spd", "spe")}


def test_decode_resolves_sim_aliases(monkeypatch):
    """A pool packed string can carry a sim ALIAS id (a team export wrote
    "Wisp"); the decode must emit the canonical id everything else in the
    system speaks — caught live by the full-pool sweep (team 548's Gengar)."""
    monkeypatch.setattr(reconstruction, "_ALIAS_CACHE", {"wisp": "Will-O-Wisp"})
    packed = ("Gengar||leftovers|levitate|wisp,icepunch,firepunch,explosion"
              "|Timid|252,,,,,252|||||")
    (gengar,) = decode_packed_team(packed)
    assert gengar["moves"] == ["willowisp", "icepunch", "firepunch", "explosion"]


def test_team_details_decodes_the_recorded_side(monkeypatch):
    monkeypatch.setattr(reconstruction, "_ALIAS_CACHE", {})  # pure unit: no node
    rec = ReconstructionRecord.from_dict(_raw())
    details = rec.team_details("p2")
    assert details == decode_packed_team(_PACKED2)
    assert details[0]["species"] == "skarmory"
    assert details[0]["moves"] == ["spikes", "drillpeck", "protect", "roar"]
    # p1's Zapdos carries the gen3 HP-Ice IV spread — the omniscient detail the
    # one-sided obs never sees but review tooling can now read.
    zapdos = rec.team_details("p1")[0]
    assert zapdos["ivs"]["atk"] == 2 and zapdos["ivs"]["spa"] == 30


# ---------------------------------------------------------------------------
# Capture registry — the __RECON__ ↔ trace-prefix join, both arrival orders
# ---------------------------------------------------------------------------

def test_join_record_first_then_prefix(tmp_path):
    tag = "battle-unit-join-a"
    assert offer_record(tag, _raw()) is None  # no prefix yet → stashed
    prefix = str(tmp_path / "loss_001")
    path = register_trace_prefix(tag, prefix, extra={"trainee_username": "Alice"})
    assert path == prefix + RECON_SUFFIX
    saved = json.loads(open(path).read())
    assert saved["battle_tag"] == tag
    assert saved["trainee_username"] == "Alice"
    assert saved["prng_seed"].startswith("sodium,")
    # consumed: a second offer stashes fresh (nothing pending anymore)
    assert pop_record(tag) is None


def test_join_prefix_first_then_record(tmp_path):
    tag = "battle-unit-join-b"
    prefix = str(tmp_path / "win_004")
    assert register_trace_prefix(tag, prefix) is None  # record not arrived → pending
    path = offer_record(tag, _raw())
    assert path == prefix + RECON_SUFFIX
    assert os.path.exists(path)
    # ReconstructionRecord.load reads what the join wrote
    rec = ReconstructionRecord.load(path)
    assert rec.battle_tag == tag


def test_pop_record_returns_and_clears():
    tag = "battle-unit-pop"
    offer_record(tag, _raw())
    rec = pop_record(tag)
    assert rec is not None and rec.battle_tag == tag
    assert pop_record(tag) is None


def test_registry_is_bounded_fifo():
    cap = reconstruction._REGISTRY_CAP
    tags = [f"battle-unit-evict-{i}" for i in range(cap + 5)]
    for t in tags:
        offer_record(t, _raw(t))
    # the 5 oldest were evicted, the newest cap survive
    for t in tags[:5]:
        assert pop_record(t) is None
    for t in tags[5:]:
        assert pop_record(t) is not None


def test_pending_prefixes_bounded_too(tmp_path):
    cap = reconstruction._REGISTRY_CAP
    tags = [f"battle-unit-pend-{i}" for i in range(cap + 3)]
    for t in tags:
        register_trace_prefix(t, str(tmp_path / t))
    # oldest 3 evicted: offering their record stashes instead of writing
    assert offer_record(tags[0], _raw()) is None
    pop_record(tags[0])  # clean up the stash
    # newest still pending: offering writes
    assert offer_record(tags[-1], _raw()) == str(tmp_path / tags[-1]) + RECON_SUFFIX
