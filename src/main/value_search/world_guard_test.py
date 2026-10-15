"""The world consistency guard on hand-built worlds (no engine: the engine's view is written from the entries, then
tampered): a consistent world passes, and each planted disagreement — the 2026-10-15 mis-paired never-seen record
among them — is named."""
from types import SimpleNamespace

import pytest

from agents.info_levers.packed import parse_set, toid
from main.value_search import world_guard as G

TRUE = ["Tyranitar||leftovers|sandstream|dragondance,rockslide,earthquake,hiddenpowerbug|Adamant|,252,,,,252|M|,30,30,,30,|||",
        "Skarmory||leftovers|keeneye|spikes,whirlwind,drillpeck,rest|Impish|252,,252,,4,|F||||",
        "Metagross||choiceband|clearbody|meteormash,earthquake,explosion,rockslide|Adamant|252,252,,,4,||||||",
        "Blissey||leftovers|naturalcure|softboiled,seismictoss,toxic,aromatherapy|Bold|252,,252,,4,|F||||"]
#: the searched side saw Tyranitar and Skarmory; Metagross and Blissey are never-seen
ROOT = SimpleNamespace(entries=TRUE, sets=[parse_set(e) for e in TRUE],
                       sids=["tyranitar", "skarmory", "metagross", "blissey"],
                       kept={"tyranitar": {"rockslide"}, "skarmory": {"spikes"}})
#: a world: the seen mons in their slots (true sets), a drawn Snorlax and the TRUE Metagross set, placed at slot 3
SNORLAX = "Snorlax||leftovers|immunity|bodyslam,curse,rest,earthquake|Careful|252,,4,,252,|M||||"
WORLD = [TRUE[0], TRUE[1], SNORLAX, TRUE[2]]


def _fact(match):
    return {"how": "drawn", "source": "", "match": match}


def _facts(world):
    out = []
    for i, e in enumerate(world):
        sid = G.species_of(e)
        seen = sid in ROOT.kept
        true_mon = seen or sid in ("metagross", "blissey")
        fa = {"slot": i, "species": sid, "seen": seen, **{c: _fact(true_mon) for c in ("item", "ability", "spread",
                                                                                       "moves")}}
        if not seen:
            fa["species_fact"] = {"how": "drawn", "source": "", "match": true_mon, "slot_match": sid == ROOT.sids[i]}
            fa["true_mon"] = true_mon
        out.append(fa)
    return out


def _view(world):
    out = []
    for e in world:
        f = parse_set(e)
        st = G.expected_stats(e)
        out.append({"details": f[0], "item": toid(f[2]), "baseAbility": toid(f[3]), "condition": f"{st['hp']}/{st['hp']}",
                    "stats": {k: v for k, v in st.items() if k != "hp"},
                    "moves": [m + ("102" if m == "return" else "") for m in G.typed_moves(f)]})
    return out


def test_a_consistent_world_passes():
    assert G.world_problems(ROOT, "]".join(WORLD), _facts(WORLD), _view(WORLD)) == []


def test_the_mis_paired_never_seen_record_is_caught():
    """THE 2026-10-15 GIGO: the placement put Metagross at slot 3 and Snorlax at 2, the records stayed where each
    was DRAWN (Metagross's at 2): slot <-> species fails at both slots."""
    fl = _facts(WORLD)
    fl[2], fl[3] = dict(fl[3], slot=2), dict(fl[2], slot=3)
    bad = G.world_problems(ROOT, "]".join(WORLD), fl, _view(WORLD))
    assert any("slot 2: entry snorlax, fact record metagross" in b for b in bad), bad
    assert any("slot 3: entry metagross, fact record snorlax" in b for b in bad), bad


@pytest.mark.parametrize("plant, needle", [
    ("engine_species", "engine skarmory"),
    ("engine_item", "item entry"),
    ("engine_moves", "moves entry"),
    ("engine_stats", "stats entry"),
    ("slot_field", "names slot 0"),
    ("species_match", "species match"),
    ("item_flag", "item match"),
    ("seen_moved", "moved from its true slot"),
])
def test_each_planted_disagreement_is_named(plant, needle):
    world, fl, view = list(WORLD), _facts(WORLD), _view(WORLD)
    if plant == "engine_species":
        view[2] = dict(view[2], details="Skarmory, F")
    elif plant == "engine_item":
        view[3] = dict(view[3], item="leftovers")
    elif plant == "engine_moves":
        view[2] = dict(view[2], moves=["bodyslam", "curse", "rest", "shadowball"])
    elif plant == "engine_stats":
        view[0] = dict(view[0], stats=dict(view[0]["stats"], spe=view[0]["stats"]["spe"] + 1))
    elif plant == "slot_field":
        fl[2] = dict(fl[2], slot=0)
    elif plant == "species_match":
        fl[3] = dict(fl[3], species_fact=dict(fl[3]["species_fact"], match=False))   # the old slot-specific flag
    elif plant == "item_flag":
        fl[2] = dict(fl[2], item=_fact(True))                                         # Snorlax holds no truth
    elif plant == "seen_moved":
        world[0], world[1] = world[1], world[0]
        fl, view = _facts(world), _view(world)
    bad = G.world_problems(ROOT, "]".join(world), fl, view)
    assert any(needle in b for b in bad), (plant, bad)


def test_a_packed_alias_is_the_engine_move_and_a_move_twice_is_caught():
    gengar = "Gengar||leftovers|levitate|thunderbolt,icepunch,wisp,explosion|Timid|,,,252,4,252|M||||"
    world = [TRUE[0], TRUE[1], gengar, TRUE[2]]
    view = _view(world)
    view[2] = dict(view[2], moves=["thunderbolt", "icepunch", "willowisp", "explosion"])
    assert not [b for b in G.world_problems(ROOT, "]".join(world), _facts(world), view) if "move" in b]
    twice = gengar.replace("thunderbolt", "willowisp")
    world[2] = twice
    view = _view(world)
    view[2] = dict(view[2], moves=["willowisp", "icepunch", "willowisp", "explosion"])
    assert any("a move twice" in b for b in G.world_problems(ROOT, "]".join(world), _facts(world), view))


def test_exact_stats_follow_the_gen3_formula():
    st = G.expected_stats(TRUE[1])                    # Skarmory 65/80/140/40/70/70, Impish, 252 HP / 252 Def
    assert st == {"hp": 334, "atk": 196, "def": 416, "spa": 104, "spd": 177, "spe": 176}
