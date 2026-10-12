"""The world sampler's pure parts: Hidden Power IVs, the packed-set fields, the move draw, the gender placement, the
extra reveal lines and the choice attribution."""
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


def _team(*entries):
    return "]".join(entries)


_RATIO = "Tyranitar||leftovers|sandstream|dragondance,rockslide,earthquake,hiddenpowerbug|Adamant|||||||"
_RATIO2 = "Skarmory||leftovers|keeneye|spikes,whirlwind,drillpeck,rest|Impish|||||||"
_NONE = "Metagross||leftovers|clearbody|meteormash,earthquake,explosion,agility|Adamant|||||||"
_FIXED = "Blissey||leftovers|naturalcure|softboiled,seismictoss,toxic,aromatherapy|Bold|||||||"


def test_gender_draw_slots_follow_the_engine_rule():
    # a ratio'd species with no gender named draws; genderless / fixed-gender / named do not
    named = _RATIO.split("|")
    named[7] = "F"
    t = _team(_RATIO, _NONE, _FIXED, "|".join(named), _RATIO2, _NONE)
    assert W.gender_draw_slots(t) == {0, 4}


def test_place_genders_puts_ratio_mons_on_the_true_draw_slots_and_names_every_other_gender():
    rng = np.random.default_rng(3)
    # never-seen slots 2..5; the truth drew at 3 and 5; the world drew two ratio'd mons and two that cannot draw
    ents = [_NONE, _NONE, _NONE, _RATIO, _FIXED, _RATIO2]
    for _ in range(50):
        out = W.place_genders(rng, ents, [2, 3, 4, 5], {3, 5})
        assert out is not None
        assert sorted(W.toid(W.parse_set(e)[0]) for e in out[2:]) == sorted(
            W.toid(W.parse_set(e)[0]) for e in ents[2:])                        # the same mons, permuted
        assert W.gender_draw_slots("]".join(out)) & {2, 3, 4, 5} == {3, 5}     # draws exactly where the truth drew
        for i in (2, 4):
            f = W.parse_set(out[i])
            sid = W.toid(f[0])
            assert f[7] == {"metagross": "", "blissey": "F"}.get(sid, f[7]) and f[7] in ("", "M", "F")
    # fewer ratio'd world mons than true draws: no world (the residual, counted by the caller)
    assert W.place_genders(rng, [_NONE, _RATIO, _NONE], [0, 1, 2], {0, 1}) is None


def test_reveals_reads_focus_punch_tightening_and_a_blocked_choice():
    lines = ["|switch|p2a: Heracross|Heracross, M|100/100", "|-singleturn|p2a: Heracross|move: Focus Punch",
             "|switch|p2a: Skarmory|Skarmory, F|100/100", "|cant|p2a: Skarmory|move: Taunt|Spikes",
             "|-start|p2a: Skarmory|move: Leech Seed"]
    res = {"chunks": [[0, lines]], "trackers": [[{"obs": {"b64": "x"}, "after": 0}], []]}
    seen, _ = W.reveals(res, 0, 0)
    assert seen == {"heracross": {"focuspunch"}, "skarmory": {"spikes"}}    # Leech Seed was used ON it


def test_opp_actives_pairs_each_command_with_its_own_request():
    import json

    def rq(sp, wait=False):
        return "|request|" + json.dumps({"wait": True} if wait else
                                        {"side": {"pokemon": [{"details": f"{sp}, M", "active": True}]}})
    res = {"chunks": [[1, [rq("Tyranitar")]], [1, [rq("x", wait=True)]], [1, [rq("Skarmory"), rq("Skarmory")]]]}
    assert W.opp_actives(res, 1, 3) == ["tyranitar", "skarmory", "skarmory"]
