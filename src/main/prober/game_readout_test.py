"""`engine.readout` — `/game`'s model panels from a battle capture (pure: numpy dicts, no torch).

The capture here is hand-built in the exact shape `model_capture.capture` returns, so each rule is
pinned on its own: the label → column mapping (the training rule: a move beyond the seats is
OTHER_move, a switch-in not among the slots is OTHER_species, a bare Hidden Power matches any HP
seat), the battle's calibration, the seat labels and the chosen action's token."""

from __future__ import annotations

import numpy as np
import pytest

from main.prober.engine import readout as R

EQ, HP_ICE, HP_BARE, ICEBEAM = 89, 0, 237, 58      # move nums (HP_ICE resolved below)
SKARM, BLISSEY, TYRANITAR, SWAMPERT = 227, 242, 248, 260


@pytest.fixture(scope="module")
def nums():
    from agents.gen3_data import moves
    return {"hp_ice": int(moves.raw()["hiddenpowerice"]["num"])}


def _cap(nums, *, p=None):
    """One decision, K = 2 seats: Earthquake (seen), Hidden Power Ice (guess); OTHER_move; six slots
    (slot 0 revealed Swampert, slots 1-2 hypotheses Skarmory / Blissey, the rest dead); OTHER_species."""
    k = 2
    F = k + 2 + 6
    cand = np.zeros((1, F), np.int64)
    cand[0, 0], cand[0, 1] = EQ, nums["hp_ice"]
    cand[0, k + 1 + 0], cand[0, k + 1 + 1], cand[0, k + 1 + 2] = SWAMPERT, SKARM, BLISSEY
    live = np.zeros((1, F), bool)
    live[0, [0, 1, 2, 3, 4, 5, F - 1]] = True
    if p is None:
        p = np.zeros((1, F), np.float32)
        p[0, [0, 1, 2, 3, 4, 5, F - 1]] = [0.4, 0.1, 0.1, 0.05, 0.15, 0.1, 0.1]
    return {"intent_p": p, "intent_live": live, "intent_cand": cand, "intent_k": k,
            "intent_hp_seat": np.zeros((1, k), bool),
            "seat_revealed": np.array([[True, False]]), "seat_nums": cand[:, :k], "seat_live": np.array([[True, True]]),
            "seat_pi": np.array([[1.0, 0.4]], np.float32),
            "slot_is_hyp": np.array([[False, True, True, False, False, False]]),
            "slot_species": np.array([[0, SKARM, BLISSEY, 0, 0, 0]]),
            "slot_pi": np.array([[1.0, 0.6, 0.3, 1.0, 1.0, 1.0]], np.float32),
            "teams": [{"our": ["zapdos"] * 6, "opp": ["swampert", "", "", "", "", ""]}]}


def _inv(opp_action):
    return {"outcome": {"opp": {"action": opp_action}}}


def test_a_seen_move_is_labelled_on_its_seat(nums):
    v = R.intent_view(_cap(nums), 0, R.opp_actual_action(_inv("earthquake")))
    assert v["actual_col"] == 0 and v["top_hit"] and v["p_actual"] == pytest.approx(0.4)


def test_a_move_beyond_the_seats_is_OTHER_move(nums):
    v = R.intent_view(_cap(nums), 0, R.opp_actual_action(_inv("icebeam")))
    assert v["actual_col"] == 2
    assert next(c for c in v["candidates"] if c["col"] == 2)["kind"] == "other_move"


def test_a_bare_hidden_power_matches_the_typed_seat_but_not_the_reverse(nums):
    assert R.intent_view(_cap(nums), 0, R.opp_actual_action(_inv("hiddenpower")))["actual_col"] == 1
    assert R._move_matches("hiddenpowerice", "hiddenpower")
    assert not R._move_matches("hiddenpowerice", "hiddenpowerfire")


def test_a_switch_to_a_hypothesis_lands_on_its_slot_and_an_unguessed_one_on_OTHER_species(nums):
    v = R.intent_view(_cap(nums), 0, R.opp_actual_action(_inv("switched_to:skarmory")))
    assert v["actual_col"] == 2 + 1 + 1
    v = R.intent_view(_cap(nums), 0, R.opp_actual_action(_inv("switched_to:tyranitar")))
    assert v["actual_col"] == 9 and v["candidates"][-1]["kind"] == "other_species"


def test_a_replacement_or_nothing_is_no_label():
    for a in ("swampert_sent_in", "unknown", "", "none"):
        assert R.opp_actual_action(_inv(a)) is None
    assert R.opp_actual_action(_inv("earthquake → skarmory_sent_in")) == {"kind": "move", "id": "earthquake"}


def test_a_guess_says_whether_it_is_on_their_true_team(nums):
    v = R.intent_view(_cap(nums), 0, None, opp_team_ids=["swampert", "skarmory", "zapdos"])
    by = {c["col"]: c for c in v["candidates"]}
    assert by[4]["on_team"] is True and by[5]["on_team"] is False and by[3]["on_team"] is None


def test_calibration_over_the_labelled_decisions_only(nums):
    views = [R.intent_view(_cap(nums), 0, R.opp_actual_action(_inv("earthquake"))),
             R.intent_view(_cap(nums), 0, R.opp_actual_action(_inv("icebeam"))),
             R.intent_view(_cap(nums), 0, None)]
    c = R.intent_calibration(views)
    assert c["n"] == 2
    assert c["mean_p_actual"] == pytest.approx((0.4 + 0.1) / 2)
    assert c["top1"] == pytest.approx(0.5)
    assert c["log_loss"] == pytest.approx((-np.log(0.4) - np.log(0.1)) / 2)
    assert R.intent_calibration([None, R.intent_view(_cap(nums), 0, None)]) is None


def test_hypotheses_name_the_revealed_mon_and_the_guesses(nums):
    cap = _cap(nums)
    cap.update(other_mass=np.array([2.1], np.float32), other_live=np.array([True]),
               other_any=np.array([0.9], np.float32), tail_idx=np.array([[TYRANITAR, 0]]),
               tail_p=np.array([[0.3, 0.0]], np.float32), move_other_mass=np.array([0.7], np.float32))
    h = R.hypotheses_view(cap, 0, opp_team_ids=["swampert", "blissey"])
    assert h["slots"][0]["revealed"] and h["slots"][0]["species"] == "Swampert"
    assert h["slots"][1]["species"] == "Skarmory" and h["slots"][1]["on_team"] is False
    assert h["slots"][2]["on_team"] is True and h["slots"][2]["presence"] == pytest.approx(0.3)
    assert [t["species"] for t in h["other_top"]] == ["Tyranitar"]
    assert h["active_moves"][0] == {"move": "Earthquake", "seen": True, "presence": 1.0}


def test_seat_labels_follow_the_layout_and_the_chosen_token():
    layout = {"n_tokens": 13 + 4 + 2 + 6 + 1 + 3, "base": 13, "board_seats": [12], "n_e3": 4, "k_e4": 2,
              "n_tail": 6, "has_other": True, "n_events": 3, "ok": True}
    cap = {"slot_is_hyp": np.array([[False, True, False, False, False, False]]),
           "slot_species": np.array([[0, SKARM, 0, 0, 0, 0]]),
           "seat_nums": np.array([[EQ, ICEBEAM]]), "seat_live": np.array([[True, True]]),
           "seat_revealed": np.array([[True, False]])}
    teams = {"our": ["zapdos", "", "", "", "", ""], "opp": ["swampert", "", "", "", "", ""]}
    labels = ["switch:zapdos"] + [""] * 5 + ["thunderbolt", "", "", "", ""]
    tl = R.token_labels(layout, teams, cap, 0, labels)
    assert len(tl) == layout["n_tokens"]
    assert tl[0]["label"] == "our Zapdos" and tl[7]["label"] == "their Skarmory (guess)"
    assert tl[12]["group"] == "board" and tl[13]["label"] == "our move: Thunderbolt"
    assert tl[17]["label"] == "their move: Earthquake (seen)" and tl[18]["label"] == "their move: Ice Beam (guess)"
    assert tl[25]["label"] == "a mon not on our list" and tl[-1]["label"] == "event −1"
    assert R.chosen_token(2, layout) == 2 and R.chosen_token(7, layout) == 14 and R.chosen_token(10, layout) is None
    bad = dict(layout, ok=False)
    assert R.token_labels(bad, teams, cap, 0, labels)[3]["label"] == "token 3"


def test_attention_summary_ranks_keys_and_skips_padded_ones():
    T = 4
    A = np.zeros((1, 1, 1, T, T), np.float16)
    A[0, 0, 0, 0] = [0.1, 0.6, 0.0, 0.3]
    cap = {"attention": A, "key_masked": np.array([[False, False, True, False]]),
           "layout": {"ok": True, "base": 13}}
    labels = [{"label": f"t{i}", "group": "g"} for i in range(T)]
    s = R.attention_summary(cap, 0, labels, action=0, our_active_slot=None)
    assert [k["token"] for k in s["rows"]["chosen"]["keys"]] == [1, 3, 0]
    assert R.attention_matrix(cap, 0, None, None).shape == (T, T)


def test_operator_view_greys_the_move_resolution_columns_when_not_built():
    op = {"outgoing": {"moves": [{"low": 0.2, "high": 0.3, "crit": 0.6, "pko": 0.0}] * 4, "p_outspeed": 0.7},
          "status_landing": [{"p_land": 0.9, "known": 1.0}] + [{"p_land": 0.0, "known": 0.0}] * 3,
          "incoming": []}
    cap = {"op": [op]}
    labels = [""] * 6 + ["toxic", "earthquake", "", ""]
    legal = [False] * 6 + [True, True, False, False]
    v = R.operator_view(cap, 0, labels, legal)
    assert [m["move"] for m in v["moves"]] == ["Toxic", "Earthquake"]
    assert v["moves"][0]["p_land"] == pytest.approx(0.9) and v["moves"][1]["p_land"] is None
    assert not v["move_resolution_built"] and v["moves"][0]["p_resolve"] is None
    cap["mr_move"] = np.zeros((1, 4, 3), np.float32)
    cap["mr_move"][0, 0] = [0.8, 0.1, 0.5]
    cap["mr_move_coords"] = ["p_resolve", "x", "p_ko_first"]
    v = R.operator_view(cap, 0, labels, legal)
    assert v["move_resolution_built"] and v["moves"][0]["p_resolve"] == pytest.approx(0.8)
    assert v["moves"][0]["p_ko_first"] == pytest.approx(0.5)
    assert R.operator_view({}, 0, labels, legal) is None
