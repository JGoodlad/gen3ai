"""The world sampler's pure parts: Hidden Power IVs, the packed-set fields, the move draw, the gender placement, the
extra reveal lines and the choice attribution."""
import numpy as np
import pytest

from main.value_search import worlds as W


def test_hp_ivs_round_trip_every_type_at_max_power():
    for t in W.HP_TYPES:
        ivs = W.hp_ivs(t)
        assert W.hp_type_of(ivs) == t
        assert all(x in (30, 31) for x in ivs)        # bit 1 set everywhere ⇒ power 70


def test_every_hidden_power_iv_pattern_is_enumerated_canonical_first_and_the_screen_state_round_trips():
    from agents.info_levers.packed import hp_iv_masks
    from main.value_search import world_screen as WS

    seen = set()
    for t in W.HP_TYPES:
        masks = hp_iv_masks(t)
        assert masks[0] == W.hp_ivs(t) and len(set(map(tuple, masks))) == len(masks)
        assert all(W.hp_type_of(m) == t and all(x in (30, 31) for x in m) for m in masks)
        seen |= {tuple(m) for m in masks}
        for k in range(len(masks)):
            name = t if k == 0 else f"{t}/{k}"
            assert WS.hp_split(name) == (t, k) and list(WS.ivs_of(name)) == masks[k]
    assert len(seen) == 64, "the 16 types partition all 64 30/31 patterns"
    # a pool set's NON-canonical Hidden Power IVs (HP Grass with 30 HP, not the canonical 30 Atk / 30 SpA) are a state
    f = W.parse_set("Jolteon||leftovers|voltabsorb|thunderbolt,hiddenpower,batonpass,substitute|Timid|"
                    "4,,,252,,252||30,31,30,30,31,30|||")
    st = WS.true_state(f)
    ty, k = WS.hp_split(st[4])
    assert ty == W.hp_type_of(W.set_ivs(f)) and k > 0 and list(WS.ivs_of(st[4])) == W.set_ivs(f)


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


# ------------------------------------------------------------------------------------------------
# the honest builder's pure parts (design_information_levers.md §7)
# ------------------------------------------------------------------------------------------------

def test_the_pad_lets_any_never_seen_mon_sit_on_a_true_draw_slot_and_pinned_slots_stay():
    from utils.team_packing import GENDER_DRAW_PAD as PAD

    rng = np.random.default_rng(5)
    # the truth drew at 2 and 3; the world drew two mons that cannot draw (no pad: no world; with it: padded)
    ents = [_NONE, _NONE, _NONE, _FIXED]
    assert W.place_genders(rng, ents, [2, 3], {2, 3}) is None
    for _ in range(20):
        out = W.place_genders(rng, ents, [2, 3], {2, 3}, pad=PAD)
        assert out is not None and W.gender_draw_slots("]".join(out)) & {2, 3} == {2, 3}
        assert all(W.parse_set(e)[7].endswith(PAD) for e in out[2:])
    # a PINNED slot (a lever put its mon there) keeps its mon; it draws iff the true slot drew
    out = W.place_genders(rng, [_NONE, _RATIO, _FIXED], [1, 2], {1}, pinned={1, 2}, pad=PAD)
    assert [W.toid(W.parse_set(e)[0]) for e in out] == ["metagross", "tyranitar", "blissey"]
    assert W.gender_draw_slots("]".join(out)) == {1}


def test_the_divergence_score_and_hint_point_the_search():
    ref = ["|move|p2a: Swampert|Rock Slide|p1a: Zapdos", "|-damage|p1a: Zapdos|79/375", "|-damage|p2a: Swampert|50/100"]
    soft = ["|move|p2a: Swampert|Rock Slide|p1a: Zapdos", "|-damage|p1a: Zapdos|177/375"]
    assert 1.0 < W.divergence_score(soft, ref) < 2.0
    assert W.divergence_score(soft, ref) < W.divergence_score(["|move|p2a: Swampert|Rock Slide|p1a: Zapdos",
                                                               "|-damage|p1a: Zapdos|90/375"], ref)
    assert W.divergence_hint(soft, ref, "p2") == ("power", 1)              # ours kept more HP: hit harder
    bulky = ref[:2] + ["|-damage|p2a: Swampert|60/100"]
    assert W.divergence_hint(bulky, ref, "p2") == ("bulk", 1)              # theirs kept more HP: less bulk
    slow = ["|move|p1a: Zapdos|Thunderbolt|p2a: Swampert"]
    assert W.divergence_hint(slow, ref, "p2") == ("speed", -1)             # ours acted first: too slow
    assert W.divergence_hint(ref, ref, "p2") is None and W.divergence_score(ref, ref) == 3.0


def test_draw_moves_follows_the_screened_hidden_power():
    from agents.info_levers import spec as S

    rng = np.random.default_rng(2)
    prior = {"earthquake": 0.9, "rockslide": 0.6, "hiddenpowerice": 0.3, "hiddenpowergrass": 0.2, "protect": 0.4,
             "toxic": 0.3}
    for _ in range(30):
        mv, _f = W.WD.draw_moves(rng, prior, {0: "earthquake"}, 4, None, [], "fire", hp_rule="force")
        assert "hiddenpowerfire" in mv and sum(m.startswith("hiddenpower") for m in mv) == 1
        mv, _f = W.WD.draw_moves(rng, prior, {0: "earthquake"}, 4, None, [], None, hp_rule="none")
        assert not any(m.startswith("hiddenpower") for m in mv)
    # a told move set matches the truth with its Hidden Power of ANOTHER type (the type is a SPREAD fact)
    mv, f = W.WD.draw_moves(rng, prior, {}, 4, S.parse("moves:seen").lever("moves", "seen"),
                            ["earthquake", "rockslide", "hiddenpowerice", "protect"], "fire", hp_rule="force")
    assert f["true_hits"] == 4 and f["match"] and "hiddenpowerfire" in mv


def test_align_to_truth_puts_true_moves_in_their_true_slots():
    out = W.WD.align_to_truth(["a", "toxic", "x", "earthquake"], ["a", "earthquake", "rest", "toxic"], {0})
    assert out == ["a", "earthquake", "x", "toxic"]


def test_the_spread_prior_gives_every_legal_spread_mass_and_none_to_an_illegal_one():
    from main.value_search import world_screen as WS

    sp = WS.SpreadPrior({("adamant", (0, 252, 0, 0, 4, 252)): 0.7, ("jolly", (0, 252, 0, 0, 4, 252)): 0.3})
    assert sp.logp("adamant", (0, 252, 0, 0, 4, 252)) > sp.logp("modest", (16, 0, 252, 176, 12, 52)) > float("-inf")
    assert sp.logp("adamant", (252, 252, 252, 0, 0, 0)) == float("-inf")         # 756 EVs
    rng = np.random.default_rng(0)
    for _ in range(200):
        nat, evs = sp.draw(rng)
        assert sum(evs) <= 510 and all(0 <= e <= 252 for e in evs) and sp.logp(nat, evs) > float("-inf")


def test_the_directed_move_goes_the_way_the_hint_says():
    from main.value_search import world_screen as WS

    m = WS.MonSampler(key=WS.ScreenKey(slot=0, hp="none"), sid="x", f_true=W.parse_set(_RATIO),
                      items={"leftovers": 0.9, "choiceband": 0.1}, abilities={"sandstream": 1.0},
                      spreads=WS.SpreadPrior({("adamant", (0, 252, 0, 0, 4, 252)): 1.0}), hp_prior={}, hp_any=0.0,
                      rng=np.random.default_rng(4), needed=1)
    s = ("leftovers", "sandstream", "adamant", (100, 100, 100, 100, 100, 0), None)
    up = [m.guided(s, ("power", 1)) for _ in range(300)]
    assert all(t[3][1] >= 100 and t[3][3] >= 100 for t in up)                  # never less Atk / SpA
    assert any(t[0] == "choiceband" for t in up)                               # a Choice Band is a way to hit harder
    down = [m.guided(s, ("bulk", 1)) for _ in range(300)]
    assert all(t[3][0] <= 100 and t[3][2] <= 100 and t[3][4] <= 100 for t in down)
    faster = [m.guided(s, ("speed", -1)) for _ in range(300)]
    assert all(t[3][5] >= 0 for t in faster) and any(t[3][5] > 0 for t in faster)


def test_place_genders_reports_the_permutation_it_applied_and_reporting_it_changes_no_draw():
    """GIGO 2026-10-15: the builder's never-seen fact records must MOVE WITH their mons through the gender placement,
    so the placement reports where each mon went (``src_of``); asking for it spends no extra random number."""
    from utils.team_packing import GENDER_DRAW_PAD as PAD

    ents = [_NONE, _NONE, _NONE, _RATIO, _FIXED, _RATIO2]
    moved = 0
    for seed in range(40):
        plain = W.place_genders(np.random.default_rng(seed), ents, [1, 2, 3, 4, 5], {3, 5}, pinned={1}, pad=PAD)
        src_of: dict = {}
        rng = np.random.default_rng(seed)
        out = W.place_genders(rng, ents, [1, 2, 3, 4, 5], {3, 5}, pinned={1}, pad=PAD, src_of=src_of)
        assert out == plain                                                     # no draw changed
        assert sorted(src_of) == [1, 2, 3, 4, 5] and sorted(src_of.values()) == [1, 2, 3, 4, 5]
        assert src_of[1] == 1                                                   # the pinned slot stays
        for slot, src in src_of.items():                                        # each slot holds its source's mon
            assert W.parse_set(out[slot])[0] == W.parse_set(ents[src])[0]
        moved += any(s != d for s, d in src_of.items())
    assert moved > 0, "the fixture must exercise a real permutation"


def test_a_prior_naming_a_move_by_a_name_variant_is_refused_not_drawn_twice():
    """POOLSETS (2026-10-15): its move table held ``wisp`` beside ``willowisp``; a world drew Will-O-Wisp twice. The
    draw refuses such a table at the seam, whoever built it."""
    import pytest

    from agents.info_levers.packed import PackedError

    table = {"willowisp": 0.5, "wisp": 0.1, "thunderbolt": 0.9, "icepunch": 0.6, "explosion": 0.3}
    with pytest.raises(PackedError, match="wisp"):
        W.WD.draw_moves(np.random.default_rng(1), table, {}, 4, None, [], None)
    ok = {k: v for k, v in table.items() if k != "wisp"}
    got, _ = W.WD.draw_moves(np.random.default_rng(1), ok, {}, 4, None, [], None)
    assert len(set(got)) == 4


def test_the_true_team_is_read_through_packed_and_an_unresolvable_id_throws_before_any_replay():
    import pytest

    from agents.info_levers.packed import PackedError

    bad = "Gengar||cb|levitate|icepunch|Timid|||||||"
    log = {"teams": ["Skarmory||leftovers|keeneye|spikes|Impish|||||||", bad], "cmds": []}
    with pytest.raises(PackedError, match="item 'cb'"):
        W.prepare_root(log, 0, "p1", "k")


def test_select_carries_each_worlds_own_log_rolls_and_records_and_copies_them():
    n = 5
    logs = [{"wid": j, "rolls": [[j, 3]]} for j in range(n)]
    w = W.Worlds(logs=logs, teams=[f"t{j}" for j in range(n)], tries=list(range(n)), leak={"x": 1},
                 revealed={"a": ["b"]}, stats=[{"j": j} for j in range(n)], facts=[[{"j": j}] for j in range(n)],
                 dice="rolls", log_w=[-float(j) for j in range(n)], rolls=[[[j, 3]] for j in range(n)])
    s = w.select([4, 1, 1])
    assert [lg["wid"] for lg in s.logs] == [4, 1, 1] and s.teams == ["t4", "t1", "t1"] and s.tries == [4, 1, 1]
    assert [lg["rolls"] for lg in s.logs] == [[[4, 3]], [[1, 3]], [[1, 3]]] and s.rolls == [[[4, 3]], [[1, 3]], [[1, 3]]]
    assert s.log_w == [-4.0, -1.0, -1.0] and [f[0]["j"] for f in s.facts] == [4, 1, 1]
    assert s.leak == {"x": 1} and s.dice == "rolls" and s.revealed == {"a": ["b"]}
    s.logs[1]["rolls"].append([9, 9])                     # a copy: the source is untouched, and so is its twin
    assert w.logs[1]["rolls"] == [[1, 3]] and s.logs[2]["rolls"] == [[1, 3]]
    with pytest.raises(IndexError):
        w.select([5])
    seed = W.Worlds(logs=logs, teams=[f"t{j}" for j in range(n)], tries=[1] * n, leak={}, revealed={})
    assert seed.select([0]).log_w == [] and seed.select([0]).rolls == []
