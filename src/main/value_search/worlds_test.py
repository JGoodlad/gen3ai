"""The world sampler's pure parts: Hidden Power IVs, the packed-set fields, the move draw."""
import numpy as np

from main.value_search import worlds as W


def test_hp_ivs_round_trip_every_type_at_max_power():
    for t in W.HP_TYPES:
        ivs = W.hp_ivs(t)
        assert W.hp_type_of(ivs) == t
        assert all(x in (30, 31) for x in ivs)        # bit 1 set everywhere ⇒ power 70


def test_hp_type_of_known_spreads():
    # the classic gen-3 HP Grass (30 Atk / 30 SpA) and HP Ice (30 Atk / 30 Def) spreads, packed order
    assert W.hp_type_of([31, 30, 31, 30, 31, 31]) == "grass"
    assert W.hp_type_of([31, 30, 30, 31, 31, 31]) == "ice"
    assert W.hp_type_of([31] * 6) == "dark"


def test_parse_set_and_ivs():
    e = "Tyranitar||leftovers|sandstream|dragondance,rockslide,hiddenpowerbug,focuspunch|Adamant|184,104,,,4,216||,,,,30,30|||"
    f = W.parse_set(e)
    assert len(f) == 12 and f[0] == "Tyranitar" and f[4].split(",")[2] == "hiddenpowerbug"
    assert W.set_ivs(f) == [31, 31, 31, 31, 30, 30]
    assert W.set_ivs(W.parse_set("Snorlax||leftovers|immunity|curse|Adamant|||||||")) == [31] * 6


def test_log_commands():
    log = {"cmds": ["CHOOSE p1 switch Snorlax", "CHOOSE p2 move thunderbolt", "FORCELOSE p2"]}
    assert W.log_commands(log) == [["p1", "switch Snorlax"], ["p2", "move thunderbolt"], ["forcelose", "p2"]]
    assert W.log_commands(log, 1) == [["p1", "switch Snorlax"]]


class _FakePrior:
    def moves(self, sid):
        return {"hiddenpowerfire": 0.5, "hiddenpowergrass": 0.5, "earthquake": 0.9, "fly": 0.9,
                "rockslide": 0.6, "protect": 0.3, "toxic": 0.2}


def test_sample_moves_keeps_slots_and_never_draws_a_banned_move_or_two_hidden_powers():
    rng = np.random.default_rng(1)
    for _ in range(200):
        mv = W.sample_moves(rng, _FakePrior(), "x", {1: "toxic"}, None)
        assert mv[1] == "toxic" and len(set(mv)) == 4
        assert "fly" not in mv
        assert sum(m.startswith("hiddenpower") for m in mv) <= 1
        mv2 = W.sample_moves(rng, _FakePrior(), "x", {}, "grass")
        assert "hiddenpowerfire" not in mv2
